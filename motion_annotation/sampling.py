"""Timestamp-exact 16-fps window sampling and random-access video decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class SampledWindow:
    """One 41-frame model window and its causal Wan VAE transition intervals."""

    window_id: int
    frame_indices: np.ndarray
    frame_timestamps: np.ndarray
    transition_starts: np.ndarray
    transition_ends: np.ndarray
    padded: bool


def build_windows(
    *,
    num_source_frames: int,
    source_fps: float,
    model_fps: float = 16.0,
    num_model_frames: int = 41,
    temporal_compression: int = 4,
    stride_model_frames: int = 40,
) -> list[SampledWindow]:
    """Build source frame indices and transition bins covering an entire clip.

    Uses the exact accumulator downsampler from O²LaF training so sampled frames
    match training-time temporal alignment. The final window repeats the last
    sampled frame only when needed, and padded tail transitions whose endpoints
    collapse are excluded via ``transition_ends > transition_starts``.
    """
    if num_source_frames < 1 or source_fps <= 0 or model_fps <= 0:
        raise ValueError('num_source_frames, source_fps, and model_fps must be positive.')
    if (num_model_frames - 1) % temporal_compression:
        raise ValueError('num_model_frames - 1 must be divisible by temporal_compression.')
    if stride_model_frames < 1:
        raise ValueError('stride_model_frames must be positive.')

    accumulator_step = float(model_fps) / float(source_fps)
    if accumulator_step > 1.0 + 1e-12:
        raise ValueError(
            f'model_fps ({model_fps}) exceeds source_fps ({source_fps}); '
            'pass effective_fps = min(model_fps, source_fps) to avoid upsampling.'
        )
    accumulator_step = min(accumulator_step, 1.0)

    accumulator = 0.0
    sampled = [0]
    for source_index in range(1, int(num_source_frames)):
        accumulator += accumulator_step
        if accumulator >= 1.0:
            sampled.append(source_index)
            accumulator -= 1.0
    sampled_arr = np.asarray(sampled, dtype=np.int64)

    starts = list(range(0, max(len(sampled_arr) - 1, 1), int(stride_model_frames)))
    endpoints = np.arange(0, int(num_model_frames), int(temporal_compression), dtype=np.int64)
    result: list[SampledWindow] = []
    for window_id, model_start in enumerate(starts):
        source_indices = sampled_arr[model_start:model_start + int(num_model_frames)]
        padded = len(source_indices) < int(num_model_frames)
        if padded:
            source_indices = np.pad(
                source_indices,
                (0, int(num_model_frames) - len(source_indices)),
                mode='edge',
            )
        frame_timestamps = source_indices.astype(np.float64) / float(source_fps)
        transition_source_indices = source_indices[endpoints]
        transition_starts = transition_source_indices[:-1].astype(np.float64) / float(source_fps)
        transition_ends = transition_source_indices[1:].astype(np.float64) / float(source_fps)
        valid = transition_ends > transition_starts
        result.append(
            SampledWindow(
                window_id=window_id,
                frame_indices=source_indices,
                frame_timestamps=frame_timestamps,
                transition_starts=transition_starts[valid],
                transition_ends=transition_ends[valid],
                padded=padded,
            )
        )
    return result


class FrameSource:
    """Random-access RGB video/GIF reader (Decord with PyAV fallback)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._reader = None
        self._frames: np.ndarray | None = None
        if Path(self.path).suffix.lower() == '.gif':
            self._decode_pyav(default_fps=100.0 / 7.0)
            return
        try:
            import decord

            self._reader = decord.VideoReader(self.path, ctx=decord.cpu(0))
            self.fps = float(self._reader.get_avg_fps())
        except Exception:  # noqa: BLE001 - fall back to PyAV for uncommon codecs or CPU envs
            self._reader = None
            self._decode_pyav(default_fps=None)

    def _decode_pyav(self, default_fps: float | None) -> None:
        decoded = []
        with av.open(self.path) as container:
            stream = next(
                (candidate for candidate in container.streams if candidate.type == 'video'),
                None,
            )
            if stream is None:
                raise ValueError(f'{self.path}: no video stream found.')
            fps_value = stream.average_rate or stream.guessed_rate or stream.base_rate or default_fps
            if fps_value is None or float(fps_value) <= 0:
                raise ValueError(f'{self.path}: could not determine video FPS.')
            self.fps = float(fps_value)
            for frame in container.decode(stream):
                decoded.append(frame.to_ndarray(format='rgb24'))
        if not decoded:
            raise ValueError(f'{self.path}: decoded no RGB frames.')
        self._frames = np.stack(decoded)

    def __len__(self) -> int:
        return len(self._frames) if self._frames is not None else len(self._reader)

    def get_batch(self, indices) -> np.ndarray:
        """Return uint8 RGB frames of shape ``[T, H, W, 3]``."""
        index_list = np.asarray(indices, dtype=np.int64)
        if self._frames is not None:
            return self._frames[index_list]
        values = self._reader.get_batch(index_list.tolist())
        return values.asnumpy() if hasattr(values, 'asnumpy') else np.asarray(values)

    def close(self) -> None:
        self._reader = None
        self._frames = None

    def __enter__(self) -> 'FrameSource':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
