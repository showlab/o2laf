"""Filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: str | os.PathLike) -> str:
    """SHA256 of a file, read incrementally so large checkpoints stay out of memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | os.PathLike, payload) -> None:
    """Write ``payload`` as JSON via a temporary file and an atomic rename."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)
