# Copyright (c) 2026 Show Lab
#
# From the latent action model of Olaf-World, released under the MIT License;
# see LICENSE in this directory.
#
# SPDX-License-Identifier: MIT
"""Small rotary embedding used by the vendored LAM."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _rotate_half(tensor: Tensor) -> Tensor:
    first = tensor[..., ::2]
    second = tensor[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Checkpoint-compatible rotary embedding with parameter name ``freqs``."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, dim, 2).float() / float(dim))
        )
        self.freqs = nn.Parameter(frequencies, requires_grad=False)

    def rotate_queries_or_keys(
        self,
        tensor: Tensor,
        freqs: Tensor | None = None,
    ) -> Tensor:
        base = self.freqs if freqs is None else freqs
        sequence_length = tensor.shape[-2]
        positions = torch.arange(
            sequence_length,
            device=tensor.device,
            dtype=base.dtype,
        )
        angles = torch.einsum(
            "n,d->nd", positions, base.to(tensor.device)
        )
        angles = torch.repeat_interleave(angles, 2, dim=-1).to(tensor.dtype)
        shape = (*([1] * (tensor.ndim - 2)), sequence_length, -1)
        cosine = angles.cos().view(*shape)
        sine = angles.sin().view(*shape)
        return tensor * cosine + _rotate_half(tensor) * sine
