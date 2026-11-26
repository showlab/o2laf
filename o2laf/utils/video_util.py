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

import os
import random
import tempfile
import warnings
from io import BytesIO
from typing import Optional

import numpy as np

# Decord's decoder thread count. Under many concurrent readers its threaded
# decoder can fail with `avcodec_send_packet ... (-11 vs. 0)`: -11 is EAGAIN, a
# back-pressure signal rather than a corrupt file, which decord treats as fatal.
# One thread is the documented workaround; parallelism comes from dataloader
# workers instead. Decoded pixels are unchanged. Files that still fail are read
# with PyAV (see `VideoLoader._switch_to_pyav`).
DECORD_NUM_THREADS = 1


class VideoLoader:
    def __init__(
        self,
        fp: str | BytesIO,
        name: Optional[str] = None,
    ):
        self.video_reader = None
        self.temp_video_name = None
        self.video_path = None
        self._backend = None
        self._decord_error = None

        if isinstance(fp, str):
            name = name or fp
            self.video_path = fp
        elif isinstance(fp, BytesIO):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                self.temp_video_name = temp_video.name
                temp_video.write(fp.read())
                temp_video.flush()
                name = name or self.temp_video_name
                self.video_path = self.temp_video_name
        else:
            raise ValueError(f'Type {type(fp)} is not supported')
        self.name = name

        try:
            self._open_decord()
        except Exception as decord_error:
            # Decord 0.6.x cannot open every codec/container combination seen
            # in web-video corpora. In particular, builds without AV1 support
            # report "cannot find video stream", and streams whose codec pixel
            # format is unknown until the first decoded frame fail while
            # creating Decord's filter graph with pix_fmt=-1. PyAV/FFmpeg can
            # decode both cases, so retain Decord as the fast path and switch
            # only the affected sample to a sequential PyAV reader.
            try:
                self._switch_to_pyav(decord_error)
            except Exception:
                self.close()
                raise

    def _open_decord(self):
        import decord

        self.video_reader = decord.VideoReader(
            self.video_path, num_threads=DECORD_NUM_THREADS
        )
        self._backend = 'decord'

    def _release_decord_reader(self):
        # VideoReader has no public close(). Dropping the last Python reference
        # invokes its destructor, which frees the native reader and joins its
        # decoder thread. Do this before unlinking a temporary input file.
        video_reader = self.video_reader
        self.video_reader = None
        if video_reader is not None:
            del video_reader

    def _switch_to_pyav(self, decord_error):
        # Do not retain the exception itself: its traceback can hold a bound
        # get_batch() frame and therefore keep the failed native reader alive.
        self._decord_error = (
            f'{type(decord_error).__name__}: {decord_error}'
        )
        self._release_decord_reader()
        try:
            self._read_pyav_metadata()
        except Exception as pyav_error:
            raise RuntimeError(
                f'{self.name} could not be opened by Decord or PyAV. '
                f'Decord: {type(decord_error).__name__}: {decord_error}; '
                f'PyAV: {type(pyav_error).__name__}: {pyav_error}'
            ) from pyav_error
        self._backend = 'pyav'

    def _read_pyav_metadata(self):
        import av

        with av.open(self.video_path) as container:
            stream = next(
                (candidate for candidate in container.streams
                 if candidate.type == 'video'),
                None,
            )
            if stream is None:
                raise RuntimeError('MP4 sample does not contain a video stream.')

            fps_fraction = (
                stream.average_rate
                or stream.guessed_rate
                or stream.base_rate
            )
            if fps_fraction is None:
                raise RuntimeError('MP4 video stream does not report an FPS.')
            fps = float(fps_fraction)
            if not np.isfinite(fps) or fps <= 0:
                raise RuntimeError(f'Invalid video FPS: {fps}.')

            # nb_frames is frequently stale in clipped web videos. Their
            # stream duration and average rate agree with the actual decoded
            # frame count, while nb_frames can be off by hundreds of frames.
            # Prefer duration-derived count and retain nb_frames as a fallback
            # for containers that do not expose stream duration.
            frame_count = 0
            if stream.duration is not None and stream.time_base is not None:
                duration_seconds = float(stream.duration * stream.time_base)
                if np.isfinite(duration_seconds) and duration_seconds > 0:
                    frame_count = int(round(duration_seconds * fps))
            if frame_count <= 0:
                frame_count = int(stream.frames or 0)
            if frame_count <= 0:
                raise RuntimeError(
                    'MP4 video stream does not report a usable frame count.'
                )

            self._pyav_video_stream_index = stream.index
            self._pyav_frame_count = frame_count
            self._pyav_fps = fps

    def close(self):
        self._release_decord_reader()
        if self.temp_video_name is not None:
            if os.path.exists(self.temp_video_name):
                os.remove(self.temp_video_name)
            self.temp_video_name = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_frame_count(self):
        if self._backend == 'decord':
            return len(self.video_reader)
        return self._pyav_frame_count

    def get_fps(self) -> Optional[float]:
        if self._backend == 'decord':
            return self.video_reader.get_avg_fps()
        return self._pyav_fps

    @staticmethod
    def _to_numpy(frames):
        if isinstance(frames, np.ndarray):
            return frames
        if hasattr(frames, 'asnumpy'):
            return frames.asnumpy()
        if hasattr(frames, 'numpy'):
            return frames.numpy()
        return np.asarray(frames)

    def _get_batch_decord(self, frame_idxs):
        frames = self.video_reader.get_batch(frame_idxs)
        return self._to_numpy(frames)

    def _get_batch_pyav(self, frame_idxs):
        import av

        requested = [int(index) for index in frame_idxs]
        wanted = set(requested)
        last_wanted = max(wanted)
        decoded = {}
        decoded_count = 0

        with av.open(self.video_path) as container:
            stream = next(
                (
                    candidate
                    for candidate in container.streams
                    if candidate.index == self._pyav_video_stream_index
                ),
                None,
            )
            if stream is None:
                raise RuntimeError(
                    'The video stream disappeared when reopening the MP4.'
                )
            # Parallelism already comes from DataLoader workers. Keeping each
            # fallback decoder single-threaded avoids multiplying native
            # decoder threads in the same way as the Decord fast path.
            stream.codec_context.thread_count = 1
            for frame in container.decode(stream):
                if decoded_count in wanted:
                    decoded[decoded_count] = frame.to_ndarray(format='rgb24')
                    if len(decoded) == len(wanted):
                        break
                decoded_count += 1
                if decoded_count > last_wanted:
                    break

        missing = sorted(wanted.difference(decoded))
        if missing:
            raise RuntimeError(
                f'{self.name} decoded only {decoded_count} frames with PyAV; '
                f'requested source frame(s) {missing[:8]}.'
            )
        return np.stack([decoded[index] for index in requested], axis=0)

    def _get_batch(self, frame_idxs):
        if self._backend == 'pyav':
            return self._get_batch_pyav(frame_idxs)

        try:
            return self._get_batch_decord(frame_idxs)
        except Exception as decord_error:
            # Some Decord failures (notably decoder EAGAIN and filter-graph
            # setup failures) are raised only on the first get_batch(), after
            # VideoReader construction and metadata queries succeeded.
            self._switch_to_pyav(decord_error)
            try:
                return self._get_batch_pyav(frame_idxs)
            except Exception as pyav_error:
                raise RuntimeError(
                    f'{self.name} failed to decode with Decord and the PyAV '
                    f'fallback. Decord: {type(decord_error).__name__}: '
                    f'{decord_error}; PyAV: {type(pyav_error).__name__}: '
                    f'{pyav_error}'
                ) from pyav_error

    def get_frames(self, frame_idxs):
        """Decode the requested 0-based frame indices as uint8 ``[N,H,W,3]``."""
        return self._get_batch(frame_idxs)

    def get_clip(self, num_frames: int, target_fps=None, random_sample=False, source_frame_start=0):
        """Decode ``num_frames`` frames at ``target_fps`` starting at or after ``source_frame_start``.

        With ``random_sample`` the clip starts at a random resampled frame,
        otherwise at the first one.
        """
        original_fps = self.get_fps()
        total_frames = self.get_frame_count()
        source_frame_start = int(source_frame_start)
        source_frame_end = total_frames
        if not 0 <= source_frame_start < source_frame_end <= total_frames:
            raise ValueError(
                'Source frame range must satisfy '
                f'0 <= start < end <= {total_frames}, got '
                f'[{source_frame_start}, {source_frame_end}).'
            )

        if target_fps is None:
            target_fps = original_fps

        if original_fps and target_fps > original_fps:
            # select_frame_indices cannot upsample: with a stride above 1 every
            # frame passes the accumulator test, so it returns the clip at its
            # native rate and raises nothing. The caller then trains or evaluates
            # on slower motion per frame than it asked for, silently. Warn rather
            # than raise -- a corpus of millions of clips should not stop for one
            # low-rate outlier -- but do not let it pass unnoticed. The fps values
            # are rounded so the number of distinct messages stays small and
            # Python's default "once per unique message" filter keeps this quiet.
            warnings.warn(
                f'Requested {target_fps} fps from a {round(original_fps)} fps '
                f'source; frames are returned at the native rate, so the motion '
                f'per frame is lower than requested.',
                RuntimeWarning,
                stacklevel=2,
            )

        frame_idxs = select_frame_indices(
            source_frame_end - source_frame_start,
            original_fps,
            target_fps=target_fps
        )
        frame_idxs = [source_frame_start + index for index in frame_idxs]
        if len(frame_idxs) < num_frames:
            raise RuntimeError(f'{self.name} only has {len(frame_idxs)} frames at {target_fps} fps, expected {num_frames} frames.')

        if random_sample and len(frame_idxs) > num_frames:
            start_idx = random.randint(0, len(frame_idxs) - num_frames)
        else:
            start_idx = 0
        frame_idxs = frame_idxs[start_idx:start_idx + num_frames]
        return self._get_batch(frame_idxs)


def select_frame_indices(total_frames, original_fps, target_fps):

    accumulator_step = target_fps / original_fps
    accumulator = 0.0

    frame_idxs = [0]

    for i in range(1, total_frames):
        accumulator += accumulator_step

        if accumulator >= 1.0:
            frame_idxs.append(i)
            accumulator -= 1.0

    return frame_idxs
