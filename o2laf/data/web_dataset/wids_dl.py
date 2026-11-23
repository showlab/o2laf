# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import fcntl
import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from urllib.parse import urlparse

recent_downloads = deque(maxlen=1000)

open_objects = {}
max_open_objects = 100


def copy_file(remote, local):
    remote = urlparse(remote)
    assert remote.scheme in ['file', '']
    # use absolute path
    remote = os.path.abspath(remote.path)
    local = urlparse(local)
    assert local.scheme in ['file', '']
    local = os.path.abspath(local.path)
    if remote == local:
        return
    # check if the local file exists
    shutil.copyfile(remote, local)


default_cmds = {
    'posixpath': copy_file,
    'file': copy_file,
}


def download_file_no_log(remote, local, handlers=default_cmds):
    """Copy a local shard into the cache.

    O²LaF training manifests are filesystem paths. Remote and ``pipe:`` schemes
    are rejected instead of being executed through a shell.
    """

    schema = urlparse(remote).scheme
    if schema is None or schema == '':
        schema = 'posixpath'
    handler = handlers.get(schema)
    if handler is None:
        raise ValueError(f'Unsupported shard URL scheme: {schema!r}. Only local files are allowed.')
    handler(remote, local)
    return local


def download_file(remote, local, handlers=default_cmds, verbose=False):
    start = time.time()
    try:
        return download_file_no_log(remote, local, handlers=handlers)
    finally:
        recent_downloads.append((remote, local, time.time(), time.time() - start))
        if verbose:
            print(
                'downloaded',
                remote,
                'to',
                local,
                'in',
                time.time() - start,
                'seconds',
                file=sys.stderr,
            )


# ----------------------------------------------------------------------------
# Optional shared on-disk shard cache (opt-in via env).
#
# By default ``download_and_open`` opens the shard in place, so on networked
# storage every mmap page-fault is a small random remote read. With caching
# enabled we copy the whole shard ONCE to a local directory (sequential read)
# and mmap the local copy thereafter. The cache is shared by all worker
# processes / ranks on a node (one file per shard basename) and bounded by a
# byte budget with LRU(mtime) eviction.
#
#   WIDS_DISK_CACHE=1            enable copy-to-local (default off)
#   WIDS_DISK_CACHE_MAX_GB=3000  total byte budget for cached *.tar
#
# Eviction never removes a shard that a reader is actively holding: readers take
# a shared (LOCK_SH) flock on the cached fd and the evictor only unlinks shards
# it can briefly LOCK_EX|NB.
# ----------------------------------------------------------------------------

def _disk_cache_enabled():
    return os.environ.get('WIDS_DISK_CACHE', '0') == '1'


def _disk_cache_budget_bytes():
    try:
        gb = float(os.environ.get('WIDS_DISK_CACHE_MAX_GB', '3000'))
    except ValueError:
        gb = 3000.0
    return int(gb * (1024 ** 3))


def _cached_tar_entries(cache_dir):
    """Return ([(path, size, mtime), ...], total_bytes) for cached *.tar files."""
    entries = []
    total = 0
    try:
        with os.scandir(cache_dir) as it:
            for e in it:
                if not e.name.endswith('.tar'):
                    continue
                try:
                    st = e.stat()
                except OSError:
                    continue
                entries.append((e.path, st.st_size, st.st_mtime))
                total += st.st_size
    except FileNotFoundError:
        pass
    return entries, total


def _evict_to_budget(cache_dir, max_bytes, incoming_bytes):
    """Unlink oldest, currently-unused cached shards until the budget fits."""
    entries, total = _cached_tar_entries(cache_dir)
    if total + incoming_bytes <= max_bytes:
        return
    entries.sort(key=lambda x: x[2])  # oldest mtime first (LRU)
    for path, size, _ in entries:
        if total + incoming_bytes <= max_bytes:
            break
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY)
            # Only evict if no reader currently holds a shared lock on it.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.unlink(path)
            total -= size
        except (OSError, BlockingIOError):
            # In use, already gone, or unlockable -- skip.
            pass
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def _open_with_shared_lock(path, mode):
    """Open ``path`` and take a best-effort shared flock (released on close)."""
    result = open(path, mode)
    try:
        fcntl.flock(result.fileno(), fcntl.LOCK_SH)
    except OSError:
        # Filesystem without flock support: degrade to no read-lock (still safe;
        # eviction may redundantly re-copy an in-use shard but never corrupts).
        pass
    return result


def _fetch_to_cache(remote, local, verbose=False):
    """Copy ``remote`` -> ``local`` atomically if missing/incomplete; LRU-evict.

    Returns the path to open (``local`` on success, else ``remote`` as fallback).
    """
    try:
        remote_size = os.path.getsize(remote)
    except OSError:
        return remote

    # Hit: complete local copy already present -> bump LRU recency.
    if os.path.exists(local) and os.path.getsize(local) == remote_size:
        try:
            os.utime(local, None)
        except OSError:
            pass
        return local

    cache_dir = os.path.dirname(local) or '.'
    os.makedirs(cache_dir, exist_ok=True)
    lock_path = local + '.lock'

    lock_fd = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # serialize copies of THIS shard

        # Re-check after acquiring the lock (another proc may have copied it).
        if os.path.exists(local) and os.path.getsize(local) == remote_size:
            try:
                os.utime(local, None)
            except OSError:
                pass
            return local

        _evict_to_budget(cache_dir, _disk_cache_budget_bytes(), remote_size)

        tmp = f'{local}.tmp.{os.getpid()}'
        try:
            shutil.copyfile(remote, tmp)
            os.replace(tmp, local)  # atomic publish
            if verbose:
                print(f'[wids-cache] cached {remote} -> {local} ({remote_size} B)', file=sys.stderr)
            return local
        except OSError as e:
            if verbose:
                print(f'[wids-cache] copy failed for {remote}: {e}; using remote', file=sys.stderr)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return remote
    except OSError:
        return remote
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass


def download_and_open(remote, local, mode='rb', handlers=default_cmds, verbose=False):
    assert os.path.exists(remote), 'File must be downloaded to local first.'

    # Optional: copy the shard to a local cache and read from there.
    if _disk_cache_enabled() and local and os.path.abspath(local) != os.path.abspath(remote):
        path = _fetch_to_cache(remote, local, verbose=verbose)
        result = _open_with_shared_lock(path, mode) if path == local else open(path, mode)
    else:
        result = open(remote, mode)

    if open_objects is not None:
        for k, v in list(open_objects.items()):
            if v.closed:
                del open_objects[k]
        if len(open_objects) > max_open_objects:
            raise RuntimeError('Too many open objects')
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        key = tuple(str(x) for x in [remote, local, mode, current_time])
        open_objects[key] = result
    return result
