# Copyright (c) 2026 Show Lab
#
# From the latent action model of Olaf-World, released under the MIT License;
# see LICENSE in this directory.
#
# SPDX-License-Identifier: MIT
"""Encoder-only pixel latent-action model."""

from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn
from torch import Tensor

from .blocks import (
    SpatioTemporalTransformer,
    patchify,
)


class LatentActionModel(nn.Module):
    """Infer one Gaussian action for every adjacent-frame transition."""

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_temporal: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.latent_dim = int(latent_dim)
        self.patch_size = int(patch_size)
        patch_token_dim = int(in_dim) * self.patch_size**2

        self.action_prompt = nn.Parameter(
            torch.empty(1, 1, 1, patch_token_dim)
        )
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=self.model_dim,
            out_dim=self.model_dim,
            num_blocks=int(enc_blocks),
            num_heads=int(num_heads),
            dropout=float(dropout),
            causal_temporal=bool(causal_temporal),
            gradient_checkpointing=bool(gradient_checkpointing),
        )
        self.fc = nn.Linear(self.model_dim, self.latent_dim * 2)

        # ``dec_blocks`` remains in the signature because it is part of the
        # checkpoint architecture metadata. The frozen inference package never
        # constructs the reconstruction decoder.
        del dec_blocks

    def encode(
        self,
        videos: Tensor,
        sample_mode: Literal["sample", "mean"] = "sample",
    ) -> Dict[str, Tensor]:
        batch, time = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        prompt = self.action_prompt.expand(batch, time, -1, -1)
        hidden = self.encoder(torch.cat([prompt, patches], dim=2))
        hidden = hidden[:, 1:, 0].reshape(
            batch * (time - 1), self.model_dim
        )
        moments = torch.nan_to_num(self.fc(hidden).float())
        z_mu, z_logvar = torch.chunk(moments, 2, dim=1)
        z_logvar = z_logvar.clamp(-10.0, 10.0)
        if sample_mode == "sample":
            z_rep = z_mu + torch.randn_like(z_mu) * torch.exp(
                0.5 * z_logvar
            )
        elif sample_mode == "mean":
            z_rep = z_mu
        else:
            raise ValueError(
                f"sample_mode must be 'sample' or 'mean', got {sample_mode!r}"
            )
        z_rep = z_rep.reshape(batch, time - 1, 1, self.latent_dim)
        return {
            "patches": patches,
            "z_rep": z_rep,
            "z_mu": z_mu.reshape(batch, time - 1, self.latent_dim),
            "z_logvar": z_logvar.reshape(
                batch, time - 1, self.latent_dim
            ),
        }

    def forward(
        self,
        videos: Tensor,
        sample_mode: Literal["sample", "mean"] = "sample",
    ) -> Dict[str, Tensor]:
        outputs = self.encode(videos, sample_mode=sample_mode)
        outputs.pop("patches", None)
        return outputs
