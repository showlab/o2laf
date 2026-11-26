"""Exact bounded-memory two-pass backward across variable-length clip windows."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def recompute_feature_backward(
    recompute: Sequence[Callable[[], torch.Tensor]],
    gradients: Sequence[torch.Tensor],
) -> None:
    """Recompute window features one segment at a time and apply clip-head gradients.

    First pass runs the encoder under ``no_grad`` over all windows of a clip,
    concatenates the detached window features, and backpropagates the clip-level
    head loss to obtain exact gradients on each window's feature tensor.
    Second pass replays each window with autograd enabled and immediately frees
    its activations, bounding GPU memory to one microbatch of windows while
    producing gradients mathematically identical to whole-clip backprop.
    """
    if len(recompute) != len(gradients):
        raise ValueError(f'Got {len(recompute)} recompute functions and {len(gradients)} gradients.')
    for index, (function, gradient) in enumerate(zip(recompute, gradients)):
        features = function()
        if features.shape != gradient.shape:
            raise RuntimeError(
                f'Segment {index} recomputed shape {tuple(features.shape)} != gradient shape {tuple(gradient.shape)}.'
            )
        if not torch.isfinite(features).all():
            raise RuntimeError(f'Segment {index} recomputed non-finite features.')
        torch.autograd.backward(features, gradient.to(features.dtype))


def split_feature_gradient(
    gradient: torch.Tensor,
    lengths: Sequence[int],
) -> list[torch.Tensor]:
    """Split a concatenated ``[T, D]`` clip gradient into per-window tensors."""
    if gradient.ndim != 2:
        raise ValueError(f'Expected [T, D] gradient, got {tuple(gradient.shape)}.')
    sizes = [int(length) for length in lengths]
    if any(length < 1 for length in sizes):
        raise ValueError(f'Segment lengths must be positive, got {sizes}.')
    if sum(sizes) != gradient.shape[0]:
        raise ValueError(f'Segment lengths sum to {sum(sizes)}, gradient has {gradient.shape[0]} rows.')
    return list(gradient.split(sizes, dim=0))
