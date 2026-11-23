# Copyright (c) 2026 Show Lab
#
# From the latent action model of Olaf-World, released under the MIT License;
# see LICENSE in this directory.
#
# SPDX-License-Identifier: MIT
"""Checkpoint-compatible transformer blocks for the pixel LAM encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .embeddings import RotaryEmbedding


def patchify(videos: Tensor, size: int) -> Tensor:
    """Convert ``[B,T,H,W,C]`` video into per-frame spatial patch tokens."""
    _, _, height, width, _ = videos.shape
    videos = videos[
        :, :, : height - (height % size), : width - (width % size), :
    ]
    return rearrange(
        videos,
        "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)",
        hp=size,
        wp=size,
    )


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(max_len).float().unsqueeze(1)
        exponent = (
            torch.arange(0, model_dim, 2).float()
            * -(math.log(10000.0) / model_dim)
        )
        divisor = torch.exp(exponent)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pos_enc", encoding, persistent=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + self.pos_enc[
            : hidden_states.shape[2]
        ].to(dtype=hidden_states.dtype)


class SelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rot_emb: bool = False,
    ) -> None:
        super().__init__()
        self.heads = int(num_heads)
        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.rot_emb = bool(rot_emb)
        if self.rot_emb:
            self.rotary_embedding = RotaryEmbedding(
                dim=model_dim // num_heads
            )

    def forward(
        self,
        hidden_states: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        query, key, value = (
            rearrange(
                projection(hidden_states),
                "b n (h d) -> b h n d",
                h=self.heads,
            )
            for projection in (self.to_q, self.to_k, self.to_v)
        )
        if self.rot_emb:
            query = self.rotary_embedding.rotate_queries_or_keys(
                query, self.rotary_embedding.freqs
            ).contiguous()
            key = self.rotary_embedding.rotate_queries_or_keys(
                key, self.rotary_embedding.freqs
            ).contiguous()
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
            dropout_p=0.0,
        )
        return self.to_out(
            rearrange(output, "b h n d -> b n (h d)")
        )


class SpatioTemporalBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spatial_attn = SelfAttention(
            model_dim, num_heads, dropout=dropout
        )
        self.temporal_attn = SelfAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            rot_emb=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(
        self,
        hidden_states: Tensor,
        causal_temporal: bool = False,
    ) -> Tensor:
        time, spatial = hidden_states.shape[1:3]
        hidden_states = rearrange(
            hidden_states, "b t s e -> (b t) s e"
        )
        hidden_states = hidden_states + self.spatial_attn(
            self.norm1(hidden_states)
        )
        hidden_states = rearrange(
            hidden_states, "(b t) s e -> b t s e", t=time
        )

        hidden_states = rearrange(
            hidden_states, "b t s e -> (b s) t e"
        )
        hidden_states = hidden_states + self.temporal_attn(
            self.norm2(hidden_states),
            is_causal=causal_temporal,
        )
        hidden_states = rearrange(
            hidden_states, "(b s) t e -> b t s e", s=spatial
        )
        return hidden_states + self.ffn(self.norm3(hidden_states))


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_temporal: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(model_dim, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)
        self.causal_temporal = bool(causal_temporal)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.pos_enc(self.ffn(hidden_states))
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    self.causal_temporal,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states, self.causal_temporal
                )
        return self.out(hidden_states)
