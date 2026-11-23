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
"""Action encoder built on the Wan 2.2 DiT trunk.

The encoder reads Wan VAE latents and emits one Gaussian action code per latent
transition. One learnable query token is interleaved into Wan's own token
sequence after every latent frame, so it is shaped by all pretrained blocks.

The attention mask has three clauses:

* video -> video is frame-causal (frame ``t`` reads frames ``0..t``), so
  ``a_t`` depends only on frames ``0..t+1``;
* video -> action is forbidden, so video tokens compute exactly what they would
  without the extra tokens;
* action -> video reads only the two frames ``{t, t+1}`` of its transition, and
  action -> action is forbidden, so the codes cannot agree on a clip-level label.

The head reads the query of transition ``t`` directly (pairwise readout).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import WanTransformer3DModel
from diffusers.utils import apply_lora_scale

from peft import LoraConfig, get_peft_model

from o2laf.action_encoders.base import ActionEncoderSpec, register_action_encoder_spec
from o2laf.action_encoders.posterior import GaussianActionHead
from o2laf.utils.dist_util import fsdp2_wrap
from o2laf.utils.lora_util import LORA_ADAPTER_NAME, LORA_STATE_KEYS, filter_learnable_module
from o2laf.utils.flex_attention import (
    WanFlexAttnProcessor,
    build_block_mask,
    cached_block_mask,
    install_flex_attention,
    pad_sequence_dim,
    padded_length,
    verify_flex_attention,
)
from o2laf.utils.registry import ACTION_ENCODER_REGISTRY


class PaddedFlexAttnProcessor(WanFlexAttnProcessor):
    """Flex self-attention that pads the sequence around the attention call.

    The encoder cannot pad ``hidden_states`` before its block loop, because the
    per-frame timestep embedding is broadcast by ``repeat_interleave`` and needs
    the unpadded length. Padding rows are zeros, the mask rejects them in both
    directions, and the output is sliced back before it leaves this call.
    """

    def __init__(self) -> None:
        super().__init__()
        self.padded_seq_len = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.block_mask is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        if self.padded_seq_len is None:
            raise RuntimeError('block_mask was installed without padded_seq_len; they must be set together.')
        real_seq_len = hidden_states.shape[1]
        padded_seq_len = int(self.padded_seq_len)
        if padded_seq_len < real_seq_len:
            raise RuntimeError(
                f'padded_seq_len={padded_seq_len} is shorter than the sequence {real_seq_len}; '
                'the block mask does not match this input.'
            )
        hidden_states = pad_sequence_dim(hidden_states, padded_seq_len, dim=1)
        if rotary_emb is not None:
            rotary_emb = tuple(pad_sequence_dim(freqs, padded_seq_len, dim=1) for freqs in rotary_emb)
        output = super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        return output[:, :real_seq_len]


class InSequenceReadout(nn.Module):
    """The per-frame query token, its RoPE angle, and the Gaussian head.

    Every parameter the readout adds lives here, under ``idm_head`` in the
    encoder, so parameter grouping and checkpointing can select it by name.
    Length-agnostic: one query weight is expanded to however many frames the
    input has.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        attention_head_dim: int = 128,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        query_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        # Small init keeps the token's residual stream dominated by what it reads.
        self.action_query = nn.Parameter(torch.empty(1, self.input_dim))
        if float(query_init_std) > 0.0:
            nn.init.normal_(self.action_query, mean=0.0, std=float(query_init_std))
        else:
            nn.init.zeros_(self.action_query)

        # Spatial RoPE for the token, parameterised by angle so it is always a
        # rotation. Zero is the identity: the token carries a purely temporal
        # position. The released configs keep it frozen at zero.
        head_dim = int(attention_head_dim)
        self.temporal_width = head_dim - 2 * (head_dim // 3)
        self.spatial_width = 2 * (head_dim // 3)
        if self.spatial_width % 2 != 0:
            raise ValueError(f'attention_head_dim={head_dim} gives an odd spatial RoPE width {self.spatial_width}.')
        self.action_rope_angle = nn.Parameter(torch.zeros(self.spatial_width // 2))

        self.query_norm = nn.LayerNorm(self.input_dim)
        self.outputs = GaussianActionHead(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
        )

    def spatial_rope_columns(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` of width ``spatial_width``, laid out like Wan's own RoPE buffer."""
        angle = self.action_rope_angle.float()
        cos = torch.cos(angle).repeat_interleave(2).to(dtype)
        sin = torch.sin(angle).repeat_interleave(2).to(dtype)
        return cos, sin

    def transition_features(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Per-frame query tokens ``[B, F, D]`` to per-transition features ``[B, F-1, D]``.

        Query ``t`` has read both frames of transition ``t -> t+1``, so it is the
        transition feature directly.
        """
        if action_tokens.ndim != 3:
            raise ValueError(f'Expected per-frame action tokens [B,F,D], got {tuple(action_tokens.shape)}.')
        if action_tokens.shape[1] < 2:
            raise ValueError(f'Need >= 2 latent frames to form a transition, got {action_tokens.shape[1]}.')
        if action_tokens.shape[-1] != self.input_dim:
            raise ValueError(f'Expected width {self.input_dim}, got {action_tokens.shape[-1]}.')
        return self.query_norm(action_tokens)[:, :-1]

    def forward(self, action_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.outputs(self.transition_features(action_tokens))


@ACTION_ENCODER_REGISTRY.register()
class WanDiTActionEncoder(WanTransformer3DModel):
    """Wan DiT trunk that reads one latent action per latent transition."""

    def setup_readout(
        self,
        latent_dim: int = 32,
        query_init_std: float = 0.02,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        """Attach the readout and the masked attention. Call before PEFT."""
        inner_dim = int(self.config.num_attention_heads * self.config.attention_head_dim)
        self.latent_action_dim = int(latent_dim)
        self.idm_head = InSequenceReadout(
            input_dim=inner_dim,
            latent_dim=int(latent_dim),
            attention_head_dim=int(self.config.attention_head_dim),
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
            query_init_std=float(query_init_std),
        )
        install_flex_attention(self.blocks, PaddedFlexAttnProcessor)
        verify_flex_attention(self.blocks, PaddedFlexAttnProcessor, label='action encoder')

    @staticmethod
    def in_sequence_mask_mod(num_frames: int, tokens_per_frame: int, device: torch.device):
        """Token predicate for the interleaved ``[video tokens | 1 query]`` per-frame layout.

        Returns ``(mask_mod, real_seq_len, padded_seq_len)``; split out so the
        predicate can be checked on CPU without a flex-attention kernel.
        """
        per_frame = int(tokens_per_frame) + 1
        real_seq_len = int(num_frames) * per_frame
        padded_seq_len = padded_length(real_seq_len)
        position = torch.arange(padded_seq_len, device=device)
        frame_index = position // per_frame
        is_action = (position % per_frame) >= int(tokens_per_frame)

        def mask_mod(b, h, q_idx, kv_idx):
            q_frame = frame_index[q_idx]
            kv_frame = frame_index[kv_idx]
            q_is_action = is_action[q_idx]
            kv_is_video = ~is_action[kv_idx]
            action_reach = (kv_frame == q_frame) | (kv_frame == q_frame + 1)
            video_reach = kv_frame <= q_frame
            allow = torch.where(q_is_action, kv_is_video & action_reach, kv_is_video & video_reach)
            is_padding = (q_idx >= real_seq_len) | (kv_idx >= real_seq_len)
            return allow & ~is_padding

        return mask_mod, real_seq_len, padded_seq_len

    @staticmethod
    def interleave_per_frame(
        sequence: torch.Tensor, extra: torch.Tensor, num_frames: int, tokens_per_frame: int
    ) -> torch.Tensor:
        """Insert ``extra`` after each frame's block of tokens, on dim 1.

        Works for hidden states ``[B, S, D]`` and RoPE frequencies alike, so the
        two always agree about the layout.
        """
        batch_size = sequence.shape[0]
        trailing = tuple(sequence.shape[2:])
        expected = int(num_frames) * int(tokens_per_frame)
        if sequence.shape[1] != expected:
            raise ValueError(f'Expected {expected} = {num_frames}x{tokens_per_frame} tokens on dim 1, got {sequence.shape[1]}.')
        blocks = sequence.reshape(batch_size, int(num_frames), int(tokens_per_frame), *trailing)
        extra = extra.expand(batch_size, int(num_frames), 1, *trailing)
        return torch.cat([blocks, extra], dim=2).flatten(1, 2)

    @staticmethod
    def extract_interleaved(sequence: torch.Tensor, num_frames: int, tokens_per_frame: int) -> torch.Tensor:
        """Pull back out what :meth:`interleave_per_frame` put in."""
        batch_size = sequence.shape[0]
        trailing = tuple(sequence.shape[2:])
        per_frame = int(tokens_per_frame) + 1
        expected = int(num_frames) * per_frame
        if sequence.shape[1] != expected:
            raise ValueError(f'Expected {expected} = {num_frames}x{per_frame} tokens on dim 1, got {sequence.shape[1]}.')
        blocks = sequence.reshape(batch_size, int(num_frames), per_frame, *trailing)
        return blocks[:, :, int(tokens_per_frame)]

    def build_in_sequence_rotary_emb(
        self, rotary_emb: tuple[torch.Tensor, torch.Tensor], num_frames: int, tokens_per_frame: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Give each query token its frame's temporal RoPE and the learned spatial angle."""
        head_dim = int(self.config.attention_head_dim)
        split_sizes = [head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3]
        freqs_cos, freqs_sin = rotary_emb
        if freqs_cos.shape[-1] != head_dim:
            raise RuntimeError(f'Expected RoPE frequencies of width {head_dim}, got {freqs_cos.shape[-1]}.')
        temporal_cos = self.rope.freqs_cos.split(split_sizes, dim=1)[0][:num_frames]
        temporal_sin = self.rope.freqs_sin.split(split_sizes, dim=1)[0][:num_frames]
        spatial_cos, spatial_sin = self.idm_head.spatial_rope_columns(temporal_cos.dtype)
        token_cos = torch.cat([temporal_cos, spatial_cos.unsqueeze(0).expand(num_frames, -1)], dim=-1).view(
            1, num_frames, 1, 1, head_dim
        )
        token_sin = torch.cat([temporal_sin, spatial_sin.unsqueeze(0).expand(num_frames, -1)], dim=-1).view(
            1, num_frames, 1, 1, head_dim
        )
        return (
            self.interleave_per_frame(freqs_cos, token_cos.to(freqs_cos.dtype), num_frames, tokens_per_frame),
            self.interleave_per_frame(freqs_sin, token_sin.to(freqs_sin.dtype), num_frames, tokens_per_frame),
        )

    def encode_query_tokens(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the trunk with query tokens in the sequence; return them as ``[B, F, D]``."""
        _, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        if num_frames % p_t != 0 or height % p_h != 0 or width % p_w != 0:
            raise ValueError(
                f'Latent shape {(num_frames, height, width)} is not divisible by patch size {(p_t, p_h, p_w)}.'
            )
        post_patch_num_frames = num_frames // p_t
        tokens_per_frame = (height // p_h) * (width // p_w)
        if post_patch_num_frames < 2:
            raise ValueError(f'Need >= 2 post-patch frames to have a transition, got {post_patch_num_frames}.')

        rotary_emb = self.rope(hidden_states)
        rotary_emb = self.build_in_sequence_rotary_emb(rotary_emb, post_patch_num_frames, tokens_per_frame)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
        hidden_states = self.interleave_per_frame(
            hidden_states,
            self.idm_head.action_query.to(hidden_states.dtype).view(1, 1, 1, -1),
            post_patch_num_frames,
            tokens_per_frame,
        )
        sequence_length = hidden_states.shape[1]

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        _, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            if ts_seq_len != sequence_length:
                if sequence_length % ts_seq_len != 0:
                    raise ValueError(f'Cannot expand {ts_seq_len} timestep embeddings to {sequence_length} tokens.')
                timestep_proj = timestep_proj.repeat_interleave(sequence_length // ts_seq_len, dim=1)
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Left installed on return: under gradient checkpointing the blocks are
        # re-run during backward and must see the same mask.
        mask_mod, _, padded_seq_len = self.in_sequence_mask_mod(
            post_patch_num_frames, tokens_per_frame, hidden_states.device
        )
        block_mask = cached_block_mask(
            self, (int(post_patch_num_frames), int(tokens_per_frame), str(hidden_states.device)),
            lambda: build_block_mask(mask_mod, padded_seq_len, hidden_states.device),
        )
        for block in self.blocks:
            block.attn1.processor.block_mask = block_mask
            block.attn1.processor.padded_seq_len = padded_seq_len

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        return self.extract_interleaved(hidden_states, post_patch_num_frames, tokens_per_frame)

    @apply_lora_scale('attention_kwargs')
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        r_timestep: torch.LongTensor | None = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Latents ``[B, C, F, H, W]`` to ``{'action', 'action_mu', 'action_logvar'}`` with F-1 codes."""
        del r_timestep, attention_kwargs
        if not hasattr(self, 'idm_head'):
            raise RuntimeError('setup_readout() must run after from_pretrained() and before encoding.')
        tokens = self.encode_query_tokens(hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image)
        return self.idm_head(tokens)


@register_action_encoder_spec('wan_dit')
class WanDiTSpec(ActionEncoderSpec):
    """Build and run the Wan DiT action encoder.

    This encoder's preprocessing is the VAE: it reads the clip as latents, which
    the world model needs anyway, so it emits one code per latent transition and
    consumes the whole clip rather than a window of RGB frames.
    """

    codes_per_transition = 1
    action_ratio_cfg_key = 'action_compression_ratio'
    legacy_contract_state_keys = ()
    reads_pixels = False
    window_rgb_frames = None
    unsupported_action_cfg_keys = ()

    def prepare_trainer(self, trainer, action_cfg):
        trainer.readout_query_init_std = float((action_cfg or {}).get('query_init_std', 0.02))

    # States written before this was fixed recorded the base model's local path,
    # which changes from machine to machine and says nothing about the action space.
    LEGACY_CONTRACT_FIELDS = ('model_name',)

    def build_contract(self, trainer, idm_cfg):
        return {
            'version': 1,
            'architecture': 'wan_dit',
            'action_compression_ratio': trainer.action_compression_ratio,
        }

    def build_encoder(self, trainer, idm_cfg, dtype):
        """Wan DiT action encoder with the in-sequence readout, adapted and FSDP2-wrapped."""
        encoder = ACTION_ENCODER_REGISTRY.get(idm_cfg['model_type']).from_pretrained(
            idm_cfg['model_name'], subfolder='transformer', torch_dtype=dtype
        )
        # Before PEFT, so that PEFT sees the final module tree.
        encoder.setup_readout(latent_dim=trainer.latent_action_dim, query_init_std=float(trainer.readout_query_init_std))

        if 'lora_config' in idm_cfg:
            lora_cfg = idm_cfg['lora_config']
            target_modules = filter_learnable_module(encoder, lora_cfg['target_module_name'])
            if not target_modules:
                raise ValueError(f'Action encoder LoRA target selection matched no linear modules: {lora_cfg["target_module_name"]}.')
            encoder = get_peft_model(
                encoder,
                LoraConfig(
                    r=int(lora_cfg['lora_rank']),
                    lora_alpha=int(lora_cfg['lora_alpha']),
                    target_modules=target_modules,
                    lora_dropout=0.0,
                    bias='none',
                ),
                adapter_name=LORA_ADAPTER_NAME,
            )
            # The readout is fully trainable already; an adapter on it would be
            # stepped at two learning rates at once.
            adapted_head = [
                name for name, _ in encoder.named_parameters()
                if 'idm_head' in name and any(key in name for key in LORA_STATE_KEYS)
            ]
            if adapted_head:
                raise ValueError(
                    f'Action encoder LoRA reached {len(adapted_head)} readout parameters; restrict '
                    f'target_module_name to the trunk (e.g. ["blocks."]). First: {adapted_head[0]}.'
                )

        trainer.set_params_to_optimize(encoder, idm_cfg['param_names_to_optimize'], idm_cfg.get('param_names_to_exclude', []))
        if idm_cfg.get('enable_gradient_checkpoint', False):
            encoder.enable_gradient_checkpointing()
        return fsdp2_wrap(encoder, transformer_block_clsname='WanTransformerBlock')

    def validate_state_contract(self, state, expected, label):
        """Check the contract when the state carries one.

        States from the research code record no contract at all; their action
        space is still identified, by the hash of the encoder's weights
        (``action_space_id``), which world model training binds to. A missing
        contract is therefore not an unknown origin and must not be rejected.
        """
        if state.get('idm_contract') is None and state.get('idm_backend') is None:
            return
        contract = state.get('idm_contract')
        if isinstance(contract, dict):
            contract = {key: value for key, value in contract.items() if key not in self.LEGACY_CONTRACT_FIELDS}
            state = {**state, 'idm_contract': contract}
        super().validate_state_contract(state, expected, label)

    def encode(self, encoder, *, latents=None, pixel_values=None, prompt_embeds=None):
        """Run the encoder on clean latents at timestep 0."""
        del pixel_values  # this encoder's preprocessing is the VAE the caller already ran
        timestep = torch.zeros(latents.shape[:1] + latents.shape[2:3], device=latents.device, dtype=torch.float32)
        outputs = encoder(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            r_timestep=timestep,
        )
        if not isinstance(outputs, dict) or 'action' not in outputs:
            raise RuntimeError(f'The action encoder did not return an action dict; got {type(outputs)}.')
        return outputs
