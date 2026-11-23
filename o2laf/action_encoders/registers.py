"""Transition registers interleaved into an RGB backbone's token sequence.

An RGB action encoder reads one action per adjacent pair of temporal states
from a learned register token placed between states ``t`` and ``t+1``. The
layout, the attention contract and the Gaussian head depend only on the number
of states and the tokens per state, so they live here rather than in a backbone.

Attention contract: video tokens read video tokens of their own and earlier
states (bidirectional within a state), never registers. A register reads the
video tokens of exactly its two states, never another register.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _check(num_states: int, tokens_per_state: int) -> tuple[int, int]:
    states = int(num_states)
    spatial = int(tokens_per_state)
    if states < 2 or spatial < 1:
        raise ValueError(f"Need >=2 states and >=1 spatial token, got {states}, {spatial}")
    return states, spatial


def sequence_length(num_states: int, tokens_per_state: int) -> int:
    """Length of the interleaved sequence."""
    states, spatial = int(num_states), int(tokens_per_state)
    return (states - 1) * (spatial + 1) + spatial


def build_layout(
    num_states: int, tokens_per_state: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(frame_index, is_action, position_ids)`` for the interleaved tokens.

    ``frame_index`` of a register is the earlier state of its transition.
    ``position_ids`` are flattened patch positions for video tokens and
    ``-(t + 2)`` for the register of transition ``t``, which is how the V-JEPA
    RoPE attention tells them apart.
    """
    states, spatial = _check(num_states, tokens_per_state)
    frame_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []
    for state in range(states):
        frame_parts.append(torch.full((spatial,), state, dtype=torch.long, device=device))
        action_parts.append(torch.zeros(spatial, dtype=torch.bool, device=device))
        position_parts.append(
            torch.arange(state * spatial, (state + 1) * spatial, dtype=torch.long, device=device)
        )
        if state < states - 1:
            frame_parts.append(torch.tensor([state], dtype=torch.long, device=device))
            action_parts.append(torch.ones(1, dtype=torch.bool, device=device))
            position_parts.append(torch.tensor([-(state + 2)], dtype=torch.long, device=device))
    return torch.cat(frame_parts), torch.cat(action_parts), torch.cat(position_parts)


def build_attention_mask(num_states: int, tokens_per_state: int, device: torch.device) -> torch.Tensor:
    """Boolean ``True = may attend`` mask over the interleaved sequence."""
    frame, is_action, _ = build_layout(num_states, tokens_per_state, device)
    query_frame = frame[:, None]
    key_frame = frame[None, :]
    query_action = is_action[:, None]
    key_video = (~is_action)[None, :]
    video_reach = key_video & (key_frame <= query_frame)
    pair_reach = key_video & ((key_frame == query_frame) | (key_frame == query_frame + 1))
    return torch.where(query_action, pair_reach, video_reach)


def interleave_registers(video: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Interleave ``[B,F,S,D]`` video and a shared ``[1,1,D]`` query into ``[B,N,D]``."""
    if video.ndim != 4 or query.ndim != 3:
        raise ValueError(f"Expected video [B,F,S,D] and query [1,1,D], got {tuple(video.shape)} and {tuple(query.shape)}")
    batch, states, spatial, width = video.shape
    _check(states, spatial)
    if query.shape[-1] != width:
        raise ValueError(f"Query width {query.shape[-1]} != video width {width}")
    registers = query.expand(batch, states - 1, 1, width)
    prefix = torch.cat([video[:, :-1], registers], dim=2).flatten(1, 2)
    return torch.cat([prefix, video[:, -1]], dim=1)


def extract_registers(sequence: torch.Tensor, num_states: int, tokens_per_state: int) -> torch.Tensor:
    """Pull the ``F-1`` register rows out of ``[B,N,D]`` as ``[B,F-1,D]``."""
    states, spatial = _check(num_states, tokens_per_state)
    expected = sequence_length(states, spatial)
    if sequence.ndim != 3 or sequence.shape[1] != expected:
        raise ValueError(f"Sequence {tuple(sequence.shape)} does not match the interleaved layout length {expected}")
    batch, width = sequence.shape[0], sequence.shape[-1]
    block = spatial + 1
    grouped = sequence[:, : (states - 1) * block].reshape(batch, states - 1, block, width)
    return grouped[:, :, -1]


class TransitionRegisterHead(nn.Module):
    """Shared register query plus a diagonal-Gaussian projection of each register."""

    def __init__(
        self,
        width: int,
        latent_dim: int,
        query_init_std: float = 0.02,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.action_query = nn.Parameter(torch.empty(1, 1, int(width)))
        nn.init.normal_(self.action_query, mean=0.0, std=float(query_init_std))
        self.norm = nn.LayerNorm(int(width))
        self.to_moments = nn.Linear(int(width), 2 * int(latent_dim))
        nn.init.normal_(self.to_moments.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.to_moments.bias)
        self.latent_dim = int(latent_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(self, registers: torch.Tensor) -> dict[str, torch.Tensor]:
        moments = self.to_moments(self.norm(registers))
        moments = torch.nan_to_num(moments.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mu, logvar = moments.chunk(2, dim=-1)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        if self.training:
            action = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            action = mu
        return {"action": action.to(registers.dtype), "action_mu": mu, "action_logvar": logvar}
