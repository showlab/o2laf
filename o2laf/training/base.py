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
"""Pieces shared by every trainer that runs on top of Wan 2.2."""

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.models import AutoencoderKLWan
from einops import rearrange
from transformers import AutoTokenizer, UMT5EncoderModel

from o2laf.schedulers.flow_match_euler import FlowMatchEulerScheduler
from o2laf.utils.dist_util import fsdp2_wrap
from o2laf.utils.ema_util import ShardEMA
from o2laf.utils.logger_util import get_logger

TEXT_MAX_LENGTH = 512


class WanTrainerBase(nn.Module, ConfigMixin):
    """Wan text encoder, VAE, scheduler and parameter selection for the trainers.

    ``self.world_model`` is the module that the training loop clips, averages
    with EMA and checkpoints by default.
    """

    def setup_wan_components(self, base_model, *, text_encoder: bool, vae: bool, scheduler_cfg, dtype, device):
        """Load the frozen Wan text encoder and VAE, and the flow-matching scheduler."""
        if text_encoder:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder='tokenizer', use_fast=False)
            self.text_encoder = UMT5EncoderModel.from_pretrained(base_model, subfolder='text_encoder')
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
            self.text_encoder = fsdp2_wrap(
                self.text_encoder, transformer_block_clsname='UMT5Block', sync_module_state=False
            )
        if vae:
            self.vae = AutoencoderKLWan.from_pretrained(base_model, subfolder='vae', torch_dtype=dtype)
            self.vae.requires_grad_(False)
            self.vae.eval()
            self.vae = self.vae.to(device=device)
        if scheduler_cfg is not None:
            self.scheduler = FlowMatchEulerScheduler.from_pretrained(base_model, subfolder='scheduler', **scheduler_cfg)

    def set_params_to_optimize(self, model, param_names_to_optimize, param_names_to_exclude):
        """Make exactly the parameters whose names match ``param_names_to_optimize`` trainable."""

        def is_keyword_in_param_name(name, keyword_list):
            return any(keyword in name for keyword in keyword_list)

        params_to_optimize = []
        params_to_fix = []
        for name, param in model.named_parameters():
            if is_keyword_in_param_name(name, param_names_to_optimize) and not is_keyword_in_param_name(
                name, param_names_to_exclude
            ):
                param.requires_grad = True
                params_to_optimize.append(param)
                get_logger().info(f'optimizer params: {name}')
            else:
                param.requires_grad = False
                params_to_fix.append(param)
                get_logger().info(f'fix params: {name}')
        get_logger().info(f'#Trained Parameters: {sum([p.numel() for p in params_to_optimize]) / 1e6} M')
        get_logger().info(f'#Fixed Parameters: {sum([p.numel() for p in params_to_fix]) / 1e6} M')

    def set_ema_model(self, decay, warmup_step=0, decay_type='fixed'):
        if decay is not None:
            self.ema = ShardEMA(self.world_model, decay=decay, warmup_step=warmup_step, decay_type=decay_type)
            get_logger().info(f'enable EMA training with decay_type {decay_type}, decay {decay}, warmup_step: {warmup_step}')

    @torch.no_grad()
    def encode_latents(self, videos):
        """Pixels ``[B, T, C, H, W]`` in ``[-1, 1]`` to normalised VAE posterior means ``[B, C, F, H, W]``."""
        videos = rearrange(videos, 'b t c h w -> b c t h w')
        moments = self.vae._encode(videos)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_inverse_std = (1.0 / torch.tensor(self.vae.config.latents_std)).view(1, -1, 1, 1, 1)
        mu, _ = torch.chunk(moments, 2, dim=1)
        latents_mean = latents_mean.to(device=mu.device)
        latents_inverse_std = latents_inverse_std.to(device=mu.device)
        return ((mu.float() - latents_mean) * latents_inverse_std).to(mu)

    @torch.no_grad()
    def encode_text_embedding(self, prompts, device):
        """UMT5 embeddings of ``prompts``, zero-padded to ``TEXT_MAX_LENGTH`` tokens."""
        text_inputs = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=TEXT_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        return torch.stack(
            [torch.cat([u, u.new_zeros(TEXT_MAX_LENGTH - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )
