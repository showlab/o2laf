"""How the action encoder covers a source video of any length.

The action encoder was trained on clips of one fixed length (41 RGB frames, 10
latent transitions). A source video is read in windows of that length, and only
the first half of each window's transitions is kept. The next window starts
where the kept transitions end, so the kept actions tile the video once, with
no gap and no overlap. Once fewer than a full window of frames remains, the
last window is shortened to just the transitions still needed.

Every window starts on the temporal grid of the VAE (4 RGB frames per latent),
so the RGB clock and the action clock stay aligned. Up to three trailing RGB
frames that do not complete a transition are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderWindow:
    """One action-encoder pass, and which of its transitions are kept."""

    start_rgb_frame: int
    num_rgb_frames: int
    kept_transitions: int  # the first ``kept_transitions`` of this window's transitions


def usable_transitions(num_rgb_frames: int, temporal_compression_ratio: int = 4) -> int:
    """Latent transitions a clip of ``num_rgb_frames`` covers; trailing partial ones are dropped."""
    if num_rgb_frames < temporal_compression_ratio + 1:
        raise ValueError(
            f'A source video needs at least {temporal_compression_ratio + 1} frames, got {num_rgb_frames}.'
        )
    return (int(num_rgb_frames) - 1) // int(temporal_compression_ratio)


def plan_encoder_windows(
    num_rgb_frames: int,
    *,
    window_rgb_frames: int,
    temporal_compression_ratio: int = 4,
) -> list[EncoderWindow]:
    """Windows that cover ``num_rgb_frames`` with each window's first half of transitions.

    ``window_rgb_frames`` is the clip length the action encoder was trained on.
    """
    ratio = int(temporal_compression_ratio)
    window = int(window_rgb_frames)
    if ratio < 1:
        raise ValueError('temporal_compression_ratio must be positive.')
    if window < ratio + 1 or (window - 1) % ratio:
        raise ValueError(f'window_rgb_frames must be {ratio}k+1 with k >= 1, got {window}.')
    keep = max(1, (window - 1) // ratio // 2)
    total = usable_transitions(num_rgb_frames, ratio)
    aligned_rgb_frames = 1 + total * ratio

    windows = []
    done = 0
    while done < total:
        kept = min(keep, total - done)
        start = done * ratio
        length = window if start + window <= aligned_rgb_frames else 1 + kept * ratio
        windows.append(EncoderWindow(start_rgb_frame=start, num_rgb_frames=length, kept_transitions=kept))
        done += kept
    return windows
