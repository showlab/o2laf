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
"""Action-conditioned Wan 2.2 world model.

The world model predicts the flow-matching velocity of a video given its clean
first latent frame and one latent action per later frame. Actions enter as a
per-frame bias on the timestep embedding, *before* ``time_proj``, so they mix
nonlinearly with the timestep through pretrained weights and reach all six
AdaLN channels of every block.

The same class serves both attention regimes:

* causal: frame ``t`` attends to frames ``0..t`` (block mask via flex attention);
* bidirectional: every frame attends to every frame (stock SDPA).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from diffusers import WanTransformer3DModel
from diffusers.utils import apply_lora_scale

from o2laf.utils.flex_attention import (
    build_block_mask,
    cached_block_mask,
    install_flex_attention,
    pad_sequence_dim,
    padded_length,
    verify_flex_attention,
)
from o2laf.utils.registry import WORLD_MODEL_REGISTRY


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather an FSDP2 DTensor; pass a plain tensor through unchanged."""
    return tensor.full_tensor() if hasattr(tensor, 'full_tensor') else tensor


class ActionEmbedder(nn.Module):
    """``gamma * Linear(action)``: maps an action code to a timestep-embedding bias.

    ``gamma`` is a learnable scalar gain. Read it through :meth:`effective_gain`
    rather than on its own: ``gamma`` and the projection weight are
    multiplicatively coupled, so only their product says how strongly actions
    reach the model.
    """

    def __init__(self, in_features: int, out_features: int, gain_init: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Shape [1] rather than 0-dim: FSDP2's fully_shard rejects 0-dim
        # parameters. Broadcasting is identical.
        self.gamma = nn.Parameter(torch.full((1,), float(gain_init)))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.gamma * self.linear(actions)

    def _active_adapters(self) -> tuple[str, ...]:
        active = getattr(self.linear, 'active_adapters', None)
        if active is None:
            active = getattr(self.linear, 'active_adapter', None)
        if active is None:
            return ()
        return (active,) if isinstance(active, str) else tuple(active)

    @torch.no_grad()
    def composite_weight(self) -> torch.Tensor:
        """The projection's effective weight, including any LoRA adapter.

        Under an ``all-linear`` LoRA target ``self.linear`` is a PEFT
        ``lora.Linear`` and ``self.linear.weight`` resolves to the base weight
        alone, which would hide most of the trainable path.
        """
        if getattr(self.linear, 'lora_magnitude_vector', None):
            raise NotImplementedError('effective_gain cannot account for DoRA magnitude scaling.')
        base = getattr(self.linear, 'base_layer', self.linear)
        weight = _full_tensor(base.weight.detach()).float()
        lora_A = getattr(self.linear, 'lora_A', None)
        lora_B = getattr(self.linear, 'lora_B', None)
        if lora_A is None or lora_B is None:
            return weight
        for name in self._active_adapters():
            if name not in lora_A or name not in lora_B:
                continue
            a = _full_tensor(lora_A[name].weight.detach()).float()
            b = _full_tensor(lora_B[name].weight.detach()).float()
            scaling = float(getattr(self.linear, 'scaling', {}).get(name, 1.0))
            weight = weight + scaling * (b @ a)
        return weight

    @torch.no_grad()
    def effective_gain(self) -> float:
        """``|gamma| * ||W||_F / sqrt(out_features)``, the action-path strength.

        Every tensor read here may be an FSDP2 DTensor and is all-gathered, so
        this is a collective and must be called on all ranks.
        """
        gamma = _full_tensor(self.gamma.detach()).float()
        weight = self.composite_weight()
        scale = math.sqrt(weight.shape[0])
        return float(gamma.reshape(-1)[0].abs() * weight.norm() / scale)


@WORLD_MODEL_REGISTRY.register()
class WanWorldModel(WanTransformer3DModel):
    """Wan 2.2 transformer conditioned on one action per latent frame."""

    def setup_world_model(
        self,
        *,
        num_latent_frames: int,
        causal: bool,
        action_dim: int,
        action_gain_init: float,
    ) -> None:
        """Configure attention and attach the action embedder.

        Call after ``from_pretrained()`` and before PEFT, so that PEFT sees the
        final module tree.
        """
        self.register_to_config(
            chunk_partition=[1] * int(num_latent_frames),
            causal_attention=bool(causal),
            action_dim=int(action_dim),
            action_gain_init=float(action_gain_init),
        )
        if causal:
            install_flex_attention(self.blocks)
            verify_flex_attention(self.blocks, label='world model')
        inner_dim = int(self.config.num_attention_heads * self.config.attention_head_dim)
        self.action_embedder = ActionEmbedder(
            in_features=int(action_dim),
            out_features=inner_dim,
            gain_init=float(action_gain_init),
        )

    @property
    def causal(self) -> bool:
        return bool(getattr(self.config, 'causal_attention', True))

    def compute_action_bias(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed packed actions ``[B, F, r, D]`` into a per-frame bias ``[B, F, C]``.

        Each action of a latent frame's group is embedded independently and the
        group is mean-pooled. Frame 0 is produced by no transition, so its bias is
        exactly zero; otherwise the embedder's own bias would leak a constant into
        the clean context frame.
        """
        if not hasattr(self, 'action_embedder'):
            raise RuntimeError('setup_world_model() must run before passing actions.')
        action_dim = int(self.config.action_dim)
        if actions.ndim != 4:
            raise ValueError(f'actions must be packed to [B, F, r, D]; got {tuple(actions.shape)}.')
        if actions.shape[-1] != action_dim:
            raise ValueError(f'action width {actions.shape[-1]} != configured action_dim {action_dim}.')
        embeds = self.action_embedder(actions).mean(dim=2)
        return torch.cat([torch.zeros_like(embeds[:, :1]), embeds[:, 1:]], dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
        action_keep_mask: torch.Tensor | None = None,
        temb_bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | SimpleNamespace:
        """Velocity prediction.

        Pass either ``actions`` (packed ``[B, F, r, D]``, embedded here) or a
        precomputed ``temb_bias`` (``[B, F, C]``), not both. With neither, no
        action reaches the model. ``action_keep_mask`` ``[B]`` zeroes the bias of
        the samples it marks False, which is the training-time action dropout.
        """
        if actions is not None:
            if temb_bias is not None:
                raise ValueError('Pass `actions` xor `temb_bias`, not both.')
            temb_bias = self.compute_action_bias(actions)
            if action_keep_mask is not None:
                temb_bias = temb_bias * action_keep_mask.to(temb_bias.dtype).view(-1, 1, 1)
        return self._denoise(hidden_states, temb_bias=temb_bias, **kwargs)

    def _inject_action_bias(self, temb, temb_bias, ts_seq_len, num_frames, chunk_frames):
        """Add the per-frame action bias to ``temb`` and re-run ``time_proj``.

        Runs while ``temb`` is still per frame, before the frame-to-token
        expansion, so row ``f`` of the bias lines up with row ``f`` of ``temb``.
        Each check below guards a silent action-to-frame misalignment.
        """
        if ts_seq_len is None or temb.ndim != 3:
            raise ValueError(
                'temb_bias requires per-frame timesteps: pass timestep with shape [B, F]. '
                f'Got ts_seq_len={ts_seq_len}, temb.ndim={temb.ndim}.'
            )
        if temb.shape[1] != num_frames:
            raise ValueError(f'temb has {temb.shape[1]} rows but {num_frames} frames were expected.')
        if chunk_frames != num_frames:
            raise ValueError(
                f'chunk_partition covers {chunk_frames} frames but the input has {num_frames}; '
                'the causal mask and the action bias would disagree about frame boundaries.'
            )
        expected_bias_shape = (temb.shape[0], num_frames, temb.shape[2])
        if tuple(temb_bias.shape) != expected_bias_shape:
            raise ValueError(f'temb_bias must be {expected_bias_shape}, got {tuple(temb_bias.shape)}.')

        temb = temb + temb_bias.to(temb.dtype)
        condition_embedder = self.condition_embedder
        projection_dtype = condition_embedder.time_proj.weight.dtype
        timestep_proj = condition_embedder.time_proj(condition_embedder.act_fn(temb.to(projection_dtype)))
        return temb, timestep_proj

    def _causal_block_mask(self, chunk_partition, token_per_frame, real_seq_len, padded_seq_len, device):
        key = (tuple(int(length) for length in chunk_partition), int(token_per_frame), int(real_seq_len),
               int(padded_seq_len), str(device))
        return cached_block_mask(
            self, key,
            lambda: self._build_causal_block_mask(chunk_partition, token_per_frame, real_seq_len, padded_seq_len, device),
        )

    @staticmethod
    def _build_causal_block_mask(chunk_partition, token_per_frame, real_seq_len, padded_seq_len, device):
        frame_idx = torch.cat(
            [
                torch.full((chunk_len * token_per_frame,), chunk_idx, device=device, dtype=torch.long)
                for chunk_idx, chunk_len in enumerate(chunk_partition)
            ],
            dim=0,
        )
        frame_idx = torch.cat([frame_idx, frame_idx.new_zeros(padded_seq_len - real_seq_len)], dim=0)

        def mask_mod(b, h, q_idx, kv_idx):
            is_padding = (q_idx >= real_seq_len) | (kv_idx >= real_seq_len)
            return (frame_idx[q_idx] >= frame_idx[kv_idx]) & ~is_padding

        return build_block_mask(mask_mod, padded_seq_len, device)

    @apply_lora_scale('attention_kwargs')
    def _denoise(
        self,
        hidden_states: torch.Tensor,
        chunk_partition: list[int] | None = None,
        timestep: torch.LongTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        r_timestep: torch.LongTensor | None = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        temb_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | SimpleNamespace:
        # `r_timestep` is accepted for call-site symmetry with the scheduler's
        # (t, r) step and is not used by plain flow matching.
        del r_timestep, attention_kwargs
        if chunk_partition is None:
            chunk_partition = list(self.config.chunk_partition)
        causal = self.causal

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = hidden_states.contiguous()
        sequence_length = hidden_states.shape[1]

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if temb_bias is not None:
            temb, timestep_proj = self._inject_action_bias(
                temb,
                temb_bias,
                ts_seq_len=ts_seq_len,
                num_frames=post_patch_num_frames,
                chunk_frames=sum(chunk_partition),
            )
        if ts_seq_len is not None:
            condition_seq_len = temb.shape[1]
            if condition_seq_len != sequence_length:
                repeat = sequence_length // condition_seq_len
                temb = temb.repeat_interleave(repeat, dim=1)
                timestep_proj = timestep_proj.repeat_interleave(repeat, dim=1)
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        token_per_frame = post_patch_height * post_patch_width
        real_seq_len = sum(chunk_partition) * token_per_frame
        if causal:
            padded_seq_len = padded_length(real_seq_len)
            attention_mask = self._causal_block_mask(
                chunk_partition, token_per_frame, real_seq_len, padded_seq_len, hidden_states.device
            )
            hidden_states = pad_sequence_dim(hidden_states, padded_seq_len, dim=1)
            if timestep_proj.ndim == 4:
                timestep_proj = pad_sequence_dim(timestep_proj, padded_seq_len, dim=1)
            rotary_emb = tuple(pad_sequence_dim(freqs, padded_seq_len, dim=1) for freqs in rotary_emb)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # The mask is left installed on return: under gradient checkpointing the
        # blocks are re-run during backward and must see the same mask.
        checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing
        for block in self.blocks:
            if causal:
                block.attn1.processor.block_mask = attention_mask
            if checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        hidden_states = hidden_states[:, :sequence_length]

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3).contiguous()

        if not return_dict:
            return output
        return SimpleNamespace(sample=output)
