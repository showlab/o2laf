"""Latent action learning: an action encoder and a causal world model, trained jointly.

The action encoder reads a clip and emits one Gaussian action code per latent
transition. The world model is Wan 2.2 with a frame-causal attention mask and
the actions injected as per-frame timestep-embedding biases. Both are trained
end to end with the world model's flow-matching loss plus ``kl_beta`` times the
KL of the action posterior to N(0, I).

The world model is first-frame conditioned: latent frame 0 is the clean encoded
first frame at timestep 0, later frames share one noise level, and the loss
covers only the predicted frames. With appearance given by frame 0, the action
code is pushed to carry motion rather than clip identity.
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import export_to_video
from einops import repeat
from peft import LoraConfig, get_peft_model
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from o2laf.action_encoders import build_action_encoder_spec
from o2laf.action_encoders.base import (
    GROUP_ACTION_EMBEDDER,
    GROUP_ACTION_ENCODER_LORA,
    GROUP_ACTION_ENCODER_OTHER,
    GROUP_READOUT,
    GROUP_WORLD_MODEL_LORA,
    GROUP_WORLD_MODEL_OTHER,
    PARAM_GROUP_NAMES,
    resolve_codes_per_transition,
    unwrap_peft,
)
from o2laf.action_encoders.posterior import kl_divergence, kl_per_dimension
from o2laf.training.base import WanTrainerBase
from o2laf.utils.action_util import (
    NULL_ACTION_NO_BIAS,
    NULL_ACTION_ZERO_CODE,
    check_latent_action_frame_budget,
    drop_action_groups,
    pack_action_groups,
    shuffle_action_groups,
)
from o2laf.utils.dist_util import fsdp2_wrap, is_main_process, load_full_state_dict, reduce_loss
from o2laf.utils.logger_util import get_logger
from o2laf.utils.lora_util import (
    LORA_ADAPTER_NAME,
    LORA_STATE_KEYS,
    filter_learnable_module,
    is_peft_model,
    validate_lora_config,
)
from o2laf.utils.registry import TRAINER_REGISTRY
from o2laf.world_models import build_world_model_from_config

# Action sources of the evaluation matrix that are not clip names.
ZERO_ACTION = '__zero__'
SHUFFLED_ACTION = '__shuffle__'


@TRAINER_REGISTRY.register()
class LatentActionTrainer(WanTrainerBase):
    """Jointly train an action encoder and a causal world model."""

    config_name = 'latent_action_trainer.json'
    world_model_causal = True
    # Codes per latent transition. The Wan DiT encoder reads latents and emits
    # one; RGB encoders run faster and override this.
    action_compression_ratio = 1
    # This trainer's dropout zeroes the action *code*, so the model learns
    # "no action" as the bias a zero code produces. Inference must ask for the
    # same thing; see utils.action_util.NULL_ACTION_*.
    null_action = NULL_ACTION_ZERO_CODE
    # The optimizer spans two FSDP2 trees, which the training loop's generic
    # checkpoint path cannot map, so this trainer saves its own distributed state.
    lightweight_checkpoint_is_distributed = True

    @register_to_config
    def __init__(
        self,
        transformer_cfg=None,
        idm_cfg=None,
        text_encoder_cfg=None,
        vae_cfg=None,
        scheduler_cfg=None,
        action_cfg=None,
        dtype=torch.float,
        device='cpu',
    ):
        nn.Module.__init__(self)
        action_cfg = action_cfg or {}

        self.latent_action_dim = int(action_cfg.get('latent_action_dim', 32))
        self.setup_action_encoder_spec(idm_cfg, action_cfg)
        self.readout_query_init_std = float(action_cfg.get('query_init_std', 0.02))
        self.beta = float(action_cfg.get('beta', 2e-3))
        # Per-transition probability of hiding an action from the world model;
        # see `drop_actions`.
        self.action_dropout_prob = float(action_cfg.get('action_dropout_prob', 0.0))
        if not 0.0 <= self.action_dropout_prob < 1.0:
            raise ValueError(f'action_dropout_prob must be in [0, 1), got {self.action_dropout_prob}.')
        self._action_drop_frac = 0.0

        # Fraction of samples whose predicted frames are noised into the
        # high-sigma band; see `draw_high_sigma_mask`.
        self.high_sigma_prob = float(action_cfg.get('high_sigma_prob', 0.0))
        if not 0.0 <= self.high_sigma_prob <= 1.0:
            raise ValueError(f'high_sigma_prob must be in [0, 1], got {self.high_sigma_prob}.')
        # The band is [high_sigma_min, high_sigma_max] in shifted sigma.
        self.high_sigma_min = float(action_cfg.get('high_sigma_min', 0.995))
        self.high_sigma_max = float(action_cfg.get('high_sigma_max', 1.0))
        if not 0.0 < self.high_sigma_min <= self.high_sigma_max <= 1.0:
            raise ValueError(
                'high_sigma_min / high_sigma_max are shifted sigmas and must satisfy 0 < min <= max <= 1, '
                f'got min={self.high_sigma_min}, max={self.high_sigma_max}.'
            )
        # None weights band samples like the peak of the gaussian loss weight.
        high_sigma_weight = action_cfg.get('high_sigma_weight', None)
        self.high_sigma_weight = None if high_sigma_weight is None else float(high_sigma_weight)
        self._high_sigma_frac = 0.0

        self.num_rgb_frames = int(action_cfg.get('num_rgb_frames', 41))
        self.temporal_compression_ratio = int(action_cfg.get('temporal_compression_ratio', 4))
        self.num_latent_frames = check_latent_action_frame_budget(self.num_rgb_frames, self.temporal_compression_ratio)
        # Clean latent frames at the head of the clip, held at timestep 0 and
        # excluded from the loss. 1 matches Wan 2.2 TI2V's image conditioning.
        self.num_context_frames = int(action_cfg.get('num_context_frames', 1))
        if not 0 <= self.num_context_frames < self.num_latent_frames:
            raise ValueError(
                f'num_context_frames={self.num_context_frames} must be in [0, {self.num_latent_frames}).'
            )

        self.fixed_prompt = action_cfg.get('fixed_prompt', 'a video')
        self.world_model_lora_lr = action_cfg.get('decoder_lora_lr')
        self.action_embedder_lr = action_cfg.get('action_embedder_lr')
        self.action_encoder_lora_lr = action_cfg.get('encoder_lora_lr')
        self.readout_lr = action_cfg.get('readout_lr')
        self.action_encoder_max_grad_norm = action_cfg.get('idm_max_grad_norm')
        self.kl_active_threshold = float(action_cfg.get('kl_active_threshold', 0.01))
        # Warm start from a portable latent-action state, restoring the halves
        # listed in `latent_action_state_parts`.
        self.latent_action_state_path = action_cfg.get('latent_action_state_path')
        self.latent_action_state_load_path = (
            action_cfg.get('latent_action_state_load_path') or self.latent_action_state_path
        )
        parts = action_cfg.get('latent_action_state_parts', ('decoder', 'encoder'))
        parts = (parts,) if isinstance(parts, str) else tuple(parts)
        unknown = set(parts) - {'decoder', 'encoder'}
        if unknown:
            raise ValueError(
                f"latent_action_state_parts got unknown entries {sorted(unknown)}; valid entries are 'decoder' and 'encoder'."
            )
        if self.latent_action_state_path and not parts:
            raise ValueError('latent_action_state_path is set but latent_action_state_parts is empty.')
        self.latent_action_state_parts = parts

        self.world_model_param_names_to_optimize = tuple((transformer_cfg or {}).get('param_names_to_optimize', ()))
        self.world_model_param_names_to_exclude = tuple((transformer_cfg or {}).get('param_names_to_exclude', ()))
        self.action_encoder_param_names_to_optimize = tuple((idm_cfg or {}).get('param_names_to_optimize', ()))
        self.action_encoder_param_names_to_exclude = tuple((idm_cfg or {}).get('param_names_to_exclude', ()))
        self.world_model_lora_config = (transformer_cfg or {}).get('lora_config')
        self.action_encoder_lora_config = (idm_cfg or {}).get('lora_config')
        self._cached_prompt_embedding = None
        # Evaluation clips are fixed for the whole run, so encode them once.
        self._cached_eval_latents = None

        if transformer_cfg is not None:
            self.world_model = self.build_world_model(transformer_cfg, action_cfg, dtype=dtype)
        if idm_cfg is not None:
            self.action_encoder = self.build_action_encoder(idm_cfg, action_cfg, dtype=dtype)

        self.setup_wan_components(
            transformer_cfg['model_name'],
            text_encoder=text_encoder_cfg is not None,
            vae=vae_cfg is not None,
            scheduler_cfg=scheduler_cfg,
            dtype=dtype,
            device=device,
        )

        if self.latent_action_state_path:
            self.load_latent_action_weights(self.latent_action_state_load_path)

    def setup_action_encoder_spec(self, idm_cfg, action_cfg):
        """Choose the action encoder named by the config and take its fixed facts.

        Safe to call more than once: a subclass that needs the spec before it
        calls ``super().__init__`` calls this first, and reads nothing but the
        config, so the second call reproduces the first.
        """
        self.encoder_name = str((idm_cfg or {}).get('backend', 'wan_dit'))
        self.encoder_spec = build_action_encoder_spec(self.encoder_name)
        self.action_compression_ratio = resolve_codes_per_transition(self.encoder_spec, action_cfg)
        self.encoder_spec.prepare_trainer(self, action_cfg or {})

    # -- construction ------------------------------------------------------

    def build_world_model(self, transformer_cfg, action_cfg, dtype):
        """Action-conditioned Wan world model, adapted and FSDP2-wrapped."""
        world_model = build_world_model_from_config(
            transformer_cfg,
            action_cfg,
            num_latent_frames=self.num_latent_frames,
            causal=self.world_model_causal,
            dtype=dtype,
        )
        # Registered before PEFT/FSDP so they join every model state dict.
        self._register_world_model_buffers(world_model)

        merge_init_cfg = transformer_cfg.get('merge_init_cfg')
        if merge_init_cfg is not None:
            if 'lora_config' in transformer_cfg:
                raise ValueError('merge_init_cfg cannot combine with a world model lora_config of its own.')
            world_model = self._merge_source_world_model_lora(world_model, merge_init_cfg)

        if 'lora_config' in transformer_cfg:
            lora_cfg = transformer_cfg['lora_config']
            target_modules = filter_learnable_module(world_model, lora_cfg['target_module_name'])
            if not target_modules:
                raise ValueError(f'World model LoRA target selection matched no linear modules: {lora_cfg["target_module_name"]}.')
            world_model = get_peft_model(
                world_model,
                LoraConfig(
                    r=int(lora_cfg['lora_rank']),
                    lora_alpha=int(lora_cfg['lora_alpha']),
                    target_modules=target_modules,
                    lora_dropout=0.0,
                    bias='none',
                ),
                adapter_name=LORA_ADAPTER_NAME,
            )

        self.set_params_to_optimize(
            world_model, transformer_cfg['param_names_to_optimize'], transformer_cfg.get('param_names_to_exclude', [])
        )
        if transformer_cfg.get('enable_gradient_checkpoint', False):
            world_model.enable_gradient_checkpointing()
        if 'lora_config' not in transformer_cfg:
            # Full fine-tuning: the action embedder is created in fp32 after
            # from_pretrained, and FSDP2 needs one parameter dtype per tree.
            world_model = world_model.to(dtype=dtype)
        return fsdp2_wrap(world_model, transformer_block_clsname='WanTransformerBlock')

    def _merge_source_world_model_lora(self, world_model, merge_init_cfg):
        """Initialise from a latent-action run's world model by folding its LoRA into the trunk.

        Rebuilds the source adapters, loads them, merges ``B @ A`` into the base
        weights and unloads them, leaving a plain trunk for full fine-tuning.
        Runs before ``set_params_to_optimize`` and FSDP2, since PEFT's merge
        needs plain tensors.
        """
        from o2laf.inference import hub
        from o2laf.training.frozen_encoder import read_world_model_half

        path = str(merge_init_cfg['path'])
        stored, source_lora_config = read_world_model_half(path)
        if not source_lora_config:
            raise ValueError(f'{path} records no world model LoRA config, so it cannot be rebuilt as adapters.')
        expected = merge_init_cfg.get('lora_config')
        if expected is not None:
            validate_lora_config(expected, source_lora_config, 'Merge-init world model')

        target_modules = filter_learnable_module(world_model, source_lora_config['target_module_name'])
        if not target_modules:
            raise ValueError(f'Merge-init target selection matched no linear modules: {source_lora_config["target_module_name"]}.')
        rank = int(source_lora_config['lora_rank'])
        alpha = int(source_lora_config['lora_alpha'])
        adapted = get_peft_model(
            world_model,
            LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules, lora_dropout=0.0, bias='none'),
            adapter_name=LORA_ADAPTER_NAME,
        )
        # Either format's names, matched onto the rebuilt adapter tree; what has
        # no match is checked below exactly as a strict load would report it.
        by_hub_key = {hub.to_hub_key(name): name for name in adapted.state_dict()}
        source = {by_hub_key[hub.to_hub_key(name)]: tensor for name, tensor in stored.items()
                  if hub.to_hub_key(name) in by_hub_key}
        unmatched = [name for name in stored if hub.to_hub_key(name) not in by_hub_key]
        missing, unexpected = adapted.load_state_dict(source, strict=False)
        unexpected = list(unexpected) + unmatched
        stranded = sorted(
            name for name in unexpected if 'action_embedder' in name or any(key in name for key in LORA_STATE_KEYS)
        )
        if stranded:
            raise RuntimeError(
                f'{len(stranded)} source world model tensors had no home in the rebuilt adapter tree, e.g. {stranded[:3]}.'
            )
        absent = sorted(name for name in missing if any(key in name for key in LORA_STATE_KEYS))
        if absent:
            raise RuntimeError(f'{len(absent)} rebuilt adapter tensors were absent from {path}, e.g. {absent[:3]}.')

        # A silent identity merge would train happily on a trunk that inherited nothing.
        adapter_norm_squared = None
        adapter_count = 0
        for name, parameter in adapted.named_parameters():
            if 'lora_B' not in name:
                continue
            adapter_count += 1
            contribution = parameter.detach().float().pow(2).sum()
            adapter_norm_squared = contribution if adapter_norm_squared is None else adapter_norm_squared + contribution
        if not adapter_count:
            raise RuntimeError(f'The rebuilt adapter tree for {path} has no lora_B tensors.')
        adapter_norm = float(adapter_norm_squared.sqrt())
        if adapter_norm == 0.0:
            raise RuntimeError(f'All {adapter_count} lora_B tensors from {path} are zero; merging would change nothing.')

        merged = adapted.merge_and_unload()
        # merge_and_unload leaves a stray `peft_config` attribute behind, which
        # would route checkpointing down the LoRA path.
        for module in merged.modules():
            module.__dict__.pop('peft_config', None)
        survivors = sorted(name for name, _ in merged.named_parameters() if any(key in name for key in LORA_STATE_KEYS))
        if survivors:
            raise RuntimeError(f'{len(survivors)} adapter tensors survived merge_and_unload, e.g. {survivors[:3]}.')
        if is_peft_model(merged):
            raise RuntimeError('The merged world model still reports as a PEFT model.')
        get_logger().info(
            'Merged %d source world model LoRA adapters (rank %d, alpha %d, lora_B norm %.6g) from %s into the trunk.',
            adapter_count, rank, alpha, adapter_norm, path,
        )
        return merged

    def _register_world_model_buffers(self, world_model):
        """Hook for trainers that bind the world model to a frozen action encoder."""
        del world_model

    def build_action_encoder(self, idm_cfg, action_cfg, dtype):
        del action_cfg
        self._idm_contract = self.encoder_spec.build_contract(self, idm_cfg)
        return self.encoder_spec.build_encoder(self, idm_cfg, dtype)

    def set_ema_model(self, decay, warmup_step=0, decay_type='fixed'):
        """EMA is refused: averaging only the world model would pair it with a live encoder."""
        if decay is None:
            return
        raise NotImplementedError(
            'EMA is not supported by LatentActionTrainer: it would average only the world model and '
            'leave the action encoder out. Remove train.ema_decay from the config.'
        )

    # -- conditioning ------------------------------------------------------

    def fixed_prompt_embedding(self, batch_size, device):
        """One cached embedding of the constant prompt, broadcast over the batch.

        The same constant conditions both models in training and evaluation, so
        it carries no information about the clip and cannot compete with the
        action code.
        """
        if self._cached_prompt_embedding is None:
            embedding = self.encode_text_embedding([self.fixed_prompt], device=device)
            self._cached_prompt_embedding = embedding.detach()
        embedding = self._cached_prompt_embedding.to(device=device)
        if embedding.shape[0] != batch_size:
            embedding = embedding.expand(batch_size, -1, -1)
        return embedding

    # -- training ----------------------------------------------------------

    def encode_actions(self, latents, prompt_embeds, batch=None):
        """Run the action encoder on one clip, in whichever form it reads."""
        pixels = batch.get('pixel_values') if batch is not None else None
        outputs = self.encoder_spec.encode(
            self.action_encoder, latents=latents, pixel_values=pixels, prompt_embeds=prompt_embeds
        )
        expected = (latents.shape[0], self.expected_num_actions, self.latent_action_dim)
        for key in ('action', 'action_mu', 'action_logvar'):
            if key not in outputs or tuple(outputs[key].shape) != expected:
                actual = None if key not in outputs else tuple(outputs[key].shape)
                raise RuntimeError(f'The {self.encoder_name} action encoder returned {key} of shape {actual}, expected {expected}.')
        return outputs

    @property
    def expected_num_actions(self) -> int:
        return self.action_compression_ratio * (self.num_latent_frames - 1)

    @property
    def kl_beta(self):
        return self.beta

    def drop_actions(self, action):
        """Drop whole transitions, so the unconditional query is "no action here".

        With ~10 transitions and p = 0.1 a clip almost never loses every action,
        so what the world model learns as unconditional is one transition's
        action removed rather than a silent clip.
        """
        action, self._action_drop_frac = drop_action_groups(action, self.action_compression_ratio, self.action_dropout_prob)
        return action

    def pack_actions(self, action):
        """``[B, rK, D]`` codes to ``[B, K+1, r, D]`` with a zero frame-0 anchor."""
        return pack_action_groups(action, self.action_compression_ratio)

    def draw_high_sigma_mask(self, batch_size, device):
        """Which samples have their predicted frames noised into the high-sigma band.

        At ordinary noise levels the world model can read most of the motion
        straight from its own noisy input, so the action is nearly redundant.
        Near ``sigma = 1`` it cannot, and the action becomes the only carrier of
        motion. The default schedule gives that regime almost no gradient, so
        this sends a fraction ``high_sigma_prob`` of samples there.
        """
        if not self.high_sigma_prob:
            self._high_sigma_frac = 0.0
            return None
        mask = torch.rand(batch_size, device=device) < self.high_sigma_prob
        self._high_sigma_frac = float(mask.float().mean())
        return mask

    def _inverse_shift(self, sigma):
        """Raw ``t`` that the scheduler's ``apply_shift`` maps to ``sigma``."""
        k = float(self.scheduler.config.shift)
        if k == 1.0:
            return float(sigma)
        return float(sigma) / (k - (k - 1.0) * float(sigma))

    def _high_sigma_raw_timestep(self, mask, raw_timestep):
        """Redraw the masked samples' raw ``t`` uniformly inside the (unshifted) band."""
        t_lo = self._inverse_shift(self.high_sigma_min)
        t_hi = 1.0 if self.high_sigma_max >= 1.0 else self._inverse_shift(self.high_sigma_max)
        redrawn = t_lo + (t_hi - t_lo) * torch.rand_like(raw_timestep)
        return torch.where(mask, redrawn, raw_timestep)

    def flow_matching_loss(
        self,
        latents,
        packed_actions,
        prompt_embeds,
        *,
        noise=None,
        raw_timestep=None,
        context_length=None,
        apply_train_weight=True,
    ):
        """Flow-matching loss on the predicted frames.

        Training draws its own noise and timestep. Evaluation passes one fixed
        ``noise`` and ``raw_timestep`` (in [0, 1], before the scheduler shift) to
        every action variant, so loss differences cannot be noise variance.
        """
        batch_size, _, num_frames, _, _ = latents.shape
        chunk_partition = [1] * num_frames
        context_length = self.num_context_frames if context_length is None else int(context_length)

        if noise is None:
            noise = torch.randn_like(latents)
        elif noise.shape != latents.shape:
            raise ValueError(f'noise shape {tuple(noise.shape)} != latents {tuple(latents.shape)}')
        # One noise level per sample, shared by every predicted frame; shaped
        # [B, F] because frame 0 needs its own timestep of 0.
        high_sigma_mask = None
        if raw_timestep is None:
            raw_timestep = torch.rand(batch_size, dtype=torch.float32, device=latents.device)
            high_sigma_mask = self.draw_high_sigma_mask(batch_size, latents.device)
            if high_sigma_mask is not None:
                raw_timestep = self._high_sigma_raw_timestep(high_sigma_mask, raw_timestep)
        else:
            raw_timestep = raw_timestep.to(device=latents.device, dtype=torch.float32).reshape(batch_size)
        timestep = repeat(raw_timestep, 'b -> b f', b=batch_size, f=num_frames).contiguous()
        timestep = (self.scheduler.apply_shift(timestep) * self.scheduler.config.num_train_timesteps).to(device=latents.device)

        noisy_latents = self.scheduler.scale_noise(latents, timestep[:, None, :], noise)

        # Clean context frames at timestep 0. `torch.cat` rather than in-place
        # assignment, since `scale_noise`'s output is on the autograd graph.
        if context_length > 0:
            noisy_latents = torch.cat([latents[:, :, :context_length], noisy_latents[:, :, context_length:]], dim=2)
            timestep = torch.cat([torch.zeros_like(timestep[:, :context_length]), timestep[:, context_length:]], dim=1)

        weight = self.scheduler.get_train_weight(timestep).to(latents.device)
        if high_sigma_mask is not None:
            # The gaussian weight is ~0 near sigma = 1, which would cancel the
            # band; band samples get a fixed weight instead.
            replacement = (
                float(self.scheduler.linear_timesteps_weights.max())
                if self.high_sigma_weight is None
                else self.high_sigma_weight
            )
            weight = torch.where(high_sigma_mask[:, None], torch.full_like(weight, replacement), weight)

        velocity = self.world_model(
            noisy_latents,
            chunk_partition=chunk_partition,
            timestep=timestep,
            r_timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            actions=packed_actions,
            return_dict=False,
        )
        target = noise - latents
        # Reduced per frame so that context frames can be sliced off and each
        # frame gets its own timestep weight.
        per_frame = ((velocity.float() - target.float()) ** 2).mean(dim=(1, 3, 4))
        if not apply_train_weight:
            return per_frame[:, context_length:].mean()
        return (per_frame[:, context_length:] * weight[:, context_length:]).mean()

    def train_step(self, batch, iters=-1):
        del iters
        self.world_model.train()
        self.action_encoder.train()

        pixels = batch['pixel_values'].to(device=self.config.device, dtype=self.config.dtype)
        latents = self.encode_latents(pixels)
        num_frames = latents.shape[2]
        if num_frames != self.num_latent_frames:
            raise ValueError(
                f'Batch has {num_frames} latent frames but the model was built for {self.num_latent_frames} '
                f'(num_rgb_frames={self.num_rgb_frames}).'
            )

        prompt_embeds = self.fixed_prompt_embedding(latents.shape[0], latents.device)
        action_outputs = self.encode_actions(latents, prompt_embeds, batch={**batch, 'pixel_values': pixels})
        action_mu = action_outputs['action_mu']
        action_logvar = action_outputs['action_logvar']
        kl = kl_divergence(action_mu, action_logvar)

        # Dropout sits between the code and the world model, so the KL and every
        # diagnostic below are measured on the full code.
        packed_actions = self.pack_actions(self.drop_actions(action_outputs['action']))
        flow_loss = self.flow_matching_loss(latents, packed_actions, prompt_embeds)

        beta = self.kl_beta
        (flow_loss + beta * kl).backward()

        # The training loop clips only the world model.
        encoder_grad_norm = None
        if self.action_encoder_max_grad_norm:
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.action_encoder.parameters() if p.requires_grad],
                max_norm=float(self.action_encoder_max_grad_norm),
            ).item()

        kl_nats = reduce_loss(kl)
        return self._diagnostics(flow_loss, kl_nats, beta, action_mu, action_logvar, encoder_grad_norm)

    @staticmethod
    def _clip_anova(action_mu):
        """Share of ``mu``'s variance that is constant within a clip: ``(clip_share, step_var)``.

        Law of total variance over the clip index, bias-corrected for estimating
        each clip mean from ``steps`` transitions. 1.0 means a per-clip label,
        0.0 a pure per-transition control. Compare against the same statistic of
        the data's ground-truth actions, not against zero.
        """
        clips, steps = int(action_mu.shape[0]), int(action_mu.shape[1])
        if clips < 2 or steps < 2:
            return 0.0, 0.0
        mu = action_mu.float()
        clip_var_raw = float(mu.mean(dim=1).var(dim=0, unbiased=False).mean())
        step_var_raw = float(mu.var(dim=1, unbiased=False).mean())
        step_var = step_var_raw * steps / (steps - 1)
        clip_var = max(clip_var_raw - step_var / steps, 0.0)
        total = clip_var + step_var
        return (clip_var / total if total > 0 else 0.0), step_var

    @torch.no_grad()
    def _diagnostics(self, flow_loss, kl_nats, beta, action_mu, action_logvar, encoder_grad_norm):
        """Per-step metrics that tell a healthy action bottleneck from a collapsed one.

        * ``kl_nats``: KL per action. ``kl_mean_share`` is the part coming from
          the mean term; a rising KL at a falling mean share is the posterior
          sharpening, not learning.
        * ``active_dims``: code dimensions above ``kl_active_threshold``.
        * ``action_std`` / ``action_sigma`` / ``action_snr``: spread of ``mu``,
          mean posterior std, and their squared ratio.
        * ``action_const_share``: the part of ``mu``'s spread that is one fixed
          vector shared by every sample; ``action_snr_batch`` excludes it.
        * ``clip_share``: see :meth:`_clip_anova`.
        * ``gamma`` / ``action_gain``: the world model's action gain.
        """
        per_dimension = kl_per_dimension(action_mu, action_logvar)
        flow_value = float(reduce_loss(flow_loss))
        kl_weighted = float(beta) * float(kl_nats)
        total = abs(flow_value) + abs(kl_weighted)
        action_std = float(action_mu.float().std())
        action_sigma = float(action_logvar.float().mul(0.5).exp().mean())
        # Law of total variance over the dimension index.
        mu_flat = action_mu.float().reshape(-1, action_mu.shape[-1])
        within_dim = float(mu_flat.var(dim=0, unbiased=False).mean())
        between_dim = float(mu_flat.mean(dim=0).var(unbiased=False))
        variance_total = within_dim + between_dim
        action_std_batch = math.sqrt(within_dim)
        # Averaged over ranks like kl_nats, which it is divided by below.
        kl_mean_term = float(reduce_loss(0.5 * action_mu.detach().float().square().reshape(-1, action_mu.shape[-1]).mean(dim=0).sum()))
        clip_share, step_var = self._clip_anova(action_mu)
        diagnostics = {
            'flow_loss': flow_value,
            'kl_nats': float(kl_nats),
            'kl_weighted': kl_weighted,
            'kl_share': (abs(kl_weighted) / total) if total > 0 else 0.0,
            'kl_beta': float(beta),
            'kl_mean_share': (kl_mean_term / float(kl_nats) if float(kl_nats) > 0 else 0.0),
            'active_dims': float((per_dimension > self.kl_active_threshold).sum()),
            'action_std': action_std,
            'action_abs_mean': float(action_mu.float().abs().mean()),
            'logvar_mean': float(action_logvar.float().mean()),
            'action_sigma': action_sigma,
            'action_snr': ((action_std / action_sigma) ** 2 if action_sigma > 0 else 0.0),
            'action_const_share': (between_dim / variance_total if variance_total > 0 else 0.0),
            'action_snr_batch': ((action_std_batch / action_sigma) ** 2 if action_sigma > 0 else 0.0),
            'clip_share': clip_share,
            'action_step_std': math.sqrt(step_var),
        }
        if self.action_dropout_prob:
            diagnostics['action_drop_frac'] = self._action_drop_frac
        if self.high_sigma_prob:
            diagnostics['high_sigma_frac'] = self._high_sigma_frac
        embedder = getattr(unwrap_peft(self.world_model), 'action_embedder', None)
        if embedder is not None:
            gamma = embedder.gamma.detach().float()
            if hasattr(gamma, 'full_tensor'):  # DTensor under FSDP2
                gamma = gamma.full_tensor()
            diagnostics['gamma'] = float(gamma.reshape(-1)[0])
            diagnostics['action_gain'] = embedder.effective_gain()
        if encoder_grad_norm is not None:
            diagnostics['action_encoder_grad_norm'] = float(encoder_grad_norm)
        return diagnostics

    # -- evaluation --------------------------------------------------------

    @torch.no_grad()
    def load_eval_clip(self, path, height, width, fps, num_frames=None):
        """Read one clip as ``[1, T, C, H, W]`` in ``[-1, 1]``, with the training transform.

        The temporal crop is deterministic, so successive evaluations of the
        same checkpoint are comparable.
        """
        from torchvision.transforms import Compose, Normalize, Resize

        from o2laf.utils.video_util import VideoLoader

        num_frames = int(num_frames or self.num_rgb_frames)
        with VideoLoader(str(path)) as loader:
            frames = loader.get_clip(num_frames, target_fps=fps, random_sample=False)
        video = torch.from_numpy(frames / 255.0).float().permute(0, 3, 1, 2).contiguous()
        transform = Compose([
            Resize((int(height), int(width))),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform(video).unsqueeze(0)

    @torch.no_grad()
    def eval_clip_latents(self, eval_cfg):
        """Cached ``{name: latents [1, C, F, H, W]}`` for the fixed evaluation clips."""
        if self._cached_eval_latents is not None:
            return self._cached_eval_latents
        height = int(eval_cfg.get('height', 480))
        width = int(eval_cfg.get('width', 832))
        fps = int(eval_cfg.get('fps', 16))
        num_frames = int(eval_cfg.get('num_frames', self.num_rgb_frames))
        expected_latent_frames = check_latent_action_frame_budget(num_frames, self.temporal_compression_ratio)

        latents = {}
        for entry in eval_cfg['clips']:
            name = entry['name']
            video = self.load_eval_clip(entry['path'], height, width, int(entry.get('fps', fps)), num_frames=num_frames)
            video = video.to(device=self.config.device, dtype=self.config.dtype)
            encoded = self.encode_latents(video)
            if encoded.shape[2] != expected_latent_frames:
                raise ValueError(
                    f'Eval clip {name} encoded to {encoded.shape[2]} latent frames, expected '
                    f'{expected_latent_frames} for {num_frames} RGB frames.'
                )
            latents[name] = encoded
        self._cached_eval_latents = latents
        return latents

    def shuffle_action_codes(self, action, seed=0):
        """Deterministically permute transitions; an identity permutation is rolled by one."""
        return shuffle_action_groups(action, self.action_compression_ratio, seed)[0]

    @torch.no_grad()
    def matched_action_gate_metrics(self, clips, codes, prompt_embeds, eval_cfg):
        """Flow loss with the clip's own, zero, shuffled and another clip's actions.

        Every variant of a clip shares one CPU-drawn noise and timestep, so the
        differences measure the action alone. ``swap`` pairs each clip with the
        next one in a ring (``gate_swap_mode: ring``) or with every other clip.
        """
        names = list(clips)
        seeds = eval_cfg.get('gate_loss_seeds', [0, 1, 2, 3])
        if isinstance(seeds, int):
            seeds = list(range(int(seeds)))
        seeds = [int(seed) for seed in seeds]
        if not seeds:
            return {}
        swap_mode = str(eval_cfg.get('gate_swap_mode', 'ring'))
        if swap_mode not in ('ring', 'all'):
            raise ValueError(f"gate_swap_mode must be 'ring' or 'all', got {swap_mode!r}.")

        values = {'action': [], 'zero': [], 'shuffle': [], 'swap': []}
        context_length = self.num_context_frames
        for seed in seeds:
            for index, name in enumerate(names):
                latents = clips[name]
                generator = torch.Generator(device='cpu').manual_seed(seed)
                noise = torch.randn(latents.shape, generator=generator, dtype=torch.float32).to(
                    device=latents.device, dtype=latents.dtype
                )
                raw_timestep = torch.rand(latents.shape[0], generator=generator, dtype=torch.float32).to(latents.device)
                own = codes[name]
                variants = {
                    'action': self.pack_actions(own),
                    'zero': self.eval_no_action_condition(own),
                    'shuffle': self.pack_actions(self.shuffle_action_codes(own, seed=seed)),
                }
                for label, packed in variants.items():
                    loss = self.flow_matching_loss(
                        latents, packed, prompt_embeds,
                        noise=noise, raw_timestep=raw_timestep, context_length=context_length,
                    )
                    values[label].append(float(loss))
                partners = (
                    [names[(index + 1) % len(names)]]
                    if swap_mode == 'ring'
                    else [candidate for candidate in names if candidate != name]
                )
                for partner in partners:
                    loss = self.flow_matching_loss(
                        latents, self.pack_actions(codes[partner]), prompt_embeds,
                        noise=noise, raw_timestep=raw_timestep, context_length=context_length,
                    )
                    values['swap'].append(float(loss))

        means = {label: sum(samples) / len(samples) for label, samples in values.items()}
        action = means['action']
        zero = means['zero']
        return {
            'gate/loss_action': action,
            'gate/loss_zero': zero,
            'gate/loss_shuffle': means['shuffle'],
            'gate/loss_swap': means['swap'],
            'gate/action_benefit': zero - action,
            'gate/relative_action_benefit': ((zero - action) / zero if abs(zero) > 1e-12 else 0.0),
            'gate/shuffle_penalty': means['shuffle'] - action,
            'gate/swap_penalty': means['swap'] - action,
            'gate/num_seeds': float(len(seeds)),
            'gate/num_clips': float(len(names)),
            'gate/num_swap_pairs': float(len(values['swap']) / len(seeds)),
        }

    @torch.no_grad()
    def generate(self, packed_actions, prompt_embeds, anchor=None, num_steps=20, seed=0):
        """Euler-sample a whole clip under the world model's attention mask.

        Not autoregressive: all predicted frames share one noise level and are
        denoised jointly. The clean anchor is pinned at timestep 0 throughout,
        exactly as in training.

        Args:
            packed_actions: ``[1, F, r, D]`` from :meth:`pack_actions`, or None.
            anchor: clean latents ``[1, C, >=num_context_frames, H, W]``, or None.
        """
        device = self.config.device
        reference = next(iter(self._cached_eval_latents.values()))
        num_latent_frames = int(reference.shape[2])
        shape = (1, reference.shape[1], num_latent_frames) + reference.shape[3:]

        # Seeded on CPU so the noise is identical on every rank and across runs.
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        latents = torch.randn(shape, generator=generator, dtype=torch.float32)
        latents = latents.to(device=device, dtype=self.config.dtype)

        chunk_partition = [1] * num_latent_frames
        self.scheduler.set_timesteps(int(num_steps), device=device)
        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, timesteps.new_tensor([0])])
        context_length = self.num_context_frames if anchor is not None else 0

        for index, t in enumerate(timesteps[:-1]):
            r = timesteps[index + 1]
            if t == r:
                continue
            step_t = t.reshape(1).expand(1, num_latent_frames).contiguous()
            step_r = r.reshape(1).expand(1, num_latent_frames).contiguous()
            if context_length > 0:
                latents[:, :, :context_length] = anchor[:, :, :context_length]
                step_t[:, :context_length] = 0
                step_r[:, :context_length] = 0
            velocity = self.world_model(
                latents,
                chunk_partition=chunk_partition,
                timestep=step_t.to(device),
                r_timestep=step_r.to(device),
                encoder_hidden_states=prompt_embeds,
                actions=packed_actions,
                return_dict=False,
            )
            latents = self.scheduler.step(velocity, t, latents, r).prev_sample

        # `scheduler.step` advances the context frames too; re-pin them.
        if context_length > 0:
            latents[:, :, :context_length] = anchor[:, :, :context_length]
        return latents

    @torch.no_grad()
    def decode_to_frames(self, latents):
        """Latents to a list of ``[H, W, 3]`` float frames in ``[0, 1]``."""
        z_dim = self.vae.config.z_dim
        mean = torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
        inverse_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1)
        mean = mean.to(device=latents.device, dtype=self.vae.dtype)
        inverse_std = inverse_std.to(device=latents.device, dtype=self.vae.dtype)
        decoded = self.vae.decode(latents.to(self.vae.dtype) / inverse_std + mean, return_dict=False)[0]
        decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
        return list(decoded[0].permute(1, 2, 3, 0).cpu().numpy())

    @torch.no_grad()
    def latent_action_eval(self, cfg, global_step=0):
        """Generate every (first frame, action source) pair of the fixed clips and score it.

        Row ``f`` takes appearance from clip ``f``'s first frame; each cell takes
        motion from a different action source: every clip (``transfer_mode:
        matrix``) or the clip itself and the next one (``ring``), plus a shuffled
        and a zero-action control. One seed for the whole matrix, so a difference
        along a row is caused only by the action and down a column only by the
        anchor. Metrics cover predicted frames only.
        """
        eval_cfg = (cfg.get('val') or {}).get('latent_action_eval') or {}
        if not eval_cfg.get('clips'):
            return None

        num_steps = int(eval_cfg.get('num_inference_steps', 20))
        save_video = bool(eval_cfg.get('save_video', True))
        seed = int(eval_cfg.get('seed', 0))
        transfer_mode = str(eval_cfg.get('transfer_mode', 'matrix'))
        if transfer_mode not in ('matrix', 'ring'):
            raise ValueError(f"transfer_mode must be 'matrix' or 'ring', got {transfer_mode!r}.")
        output_root = osp.join(cfg['path']['visualization'], 'latent_action_eval')

        was_training = self.world_model.training
        self.world_model.eval()
        self._eval_action_encoder_eval()
        try:
            clips = self.eval_clip_latents(eval_cfg)
            names = list(clips)
            if len(names) < 2:
                raise ValueError(f'latent_action_eval needs at least two clips; got {names}.')
            prompt_embeds = self.fixed_prompt_embedding(1, self.config.device)
            # eval() makes the Gaussian heads return the posterior mean.
            codes = self.eval_action_codes(clips, prompt_embeds, eval_cfg)
            action_sources = self.eval_action_sources(names, transfer_mode)

            context = self.num_context_frames
            predicted = slice(context, None)

            def mse(a, b):
                difference = a.float()[:, :, predicted] - b.float()[:, :, predicted]
                return float((difference**2).mean())

            def latent_motion(x):
                """Consecutive latent differences over predicted frames only."""
                x = x.float()
                return x[:, :, context + 1 :] - x[:, :, context:-1]

            generated = {}
            for name in names:
                anchor = clips[name][:, :, :context] if context > 0 else None
                for source in action_sources[name]:
                    if source == ZERO_ACTION:
                        packed_actions = self.eval_no_action_condition(codes[name])
                    elif source == SHUFFLED_ACTION:
                        packed_actions = self.pack_actions(self.shuffle_action_codes(codes[name], seed=seed))
                    else:
                        packed_actions = self.pack_actions(codes[source])
                    # Every rank generates: an FSDP2 forward needs all ranks.
                    generated[(name, source)] = self.generate(
                        packed_actions, prompt_embeds, anchor=anchor, num_steps=num_steps, seed=seed
                    )

            metrics = {}
            for name in names:
                reference = clips[name].float()
                own = generated[(name, name)]
                zero = generated[(name, ZERO_ACTION)]
                metrics[f'{name}/recon_mse'] = mse(own, reference)
                metrics[f'{name}/recon_mse_zero'] = mse(zero, reference)
                metrics[f'{name}/action_sensitivity'] = mse(own, zero)
                metrics[f'{name}/shuffle_sensitivity'] = mse(own, generated[(name, SHUFFLED_ACTION)])
                for source in action_sources[name]:
                    if source in (name, ZERO_ACTION, SHUFFLED_ACTION):
                        continue
                    metrics[f'{name}/divergence_from_{source}'] = mse(own, generated[(name, source)])

            metrics.update(self.transfer_consistency_metrics(names, action_sources, generated, latent_motion))
            if bool(eval_cfg.get('compute_gate_losses', True)):
                metrics.update(self.matched_action_gate_metrics(clips, codes, prompt_embeds, eval_cfg))

            save_state_steps = {str(step) for step in eval_cfg.get('save_state_steps', [])}
            if str(global_step) in save_state_steps:
                # The portable latent-action state of this step.
                state = self.export_latent_action_state()
                if is_main_process():
                    state_directory = osp.join(output_root, f'iter_{global_step}')
                    os.makedirs(state_directory, exist_ok=True)
                    state['global_step'] = int(global_step)
                    # Via a temporary file: a crash mid-write must not leave a
                    # truncated state that a later run would load.
                    state_path = osp.join(state_directory, 'latent_action_state.pt')
                    torch.save(state, state_path + '.tmp')
                    os.replace(state_path + '.tmp', state_path)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            if save_video and is_main_process():
                entries = {entry['name']: entry for entry in eval_cfg['clips']}
                for name in names:
                    fps = int(entries[name].get('fps', eval_cfg.get('fps', 16)))
                    directory = osp.join(output_root, f'iter_{global_step}', name)
                    os.makedirs(directory, exist_ok=True)
                    source_video = entries[name].get('path')
                    ground_truth_path = osp.join(directory, 'ground_truth.mp4')
                    if source_video and osp.isfile(source_video):
                        shutil.copyfile(source_video, ground_truth_path)
                    else:
                        export_to_video(
                            self.decode_to_frames(clips[name].float().to(self.config.dtype)),
                            output_video_path=ground_truth_path,
                            fps=fps,
                        )
                    # The directory fixes the first frame; the file names the action source.
                    for source in action_sources[name]:
                        if source == ZERO_ACTION:
                            stem = 'zero_action'
                        elif source == SHUFFLED_ACTION:
                            stem = 'shuffled_action'
                        elif source == name:
                            stem = 'self'
                        else:
                            stem = f'action_from_{source}'
                        export_to_video(
                            self.decode_to_frames(generated[(name, source)]),
                            output_video_path=osp.join(directory, f'{stem}.mp4'),
                            fps=fps,
                        )
                metrics_path = osp.join(output_root, f'iter_{global_step}', 'metrics.json')
                with open(metrics_path, 'w', encoding='utf-8') as handle:
                    json.dump(metrics, handle, indent=2, sort_keys=True)
        finally:
            if was_training:
                self.world_model.train()
            self._eval_action_encoder_restore()
        return metrics

    def _eval_action_encoder_eval(self):
        self.action_encoder.eval()

    def _eval_action_encoder_restore(self):
        self.action_encoder.train()

    def eval_action_codes(self, clips, prompt_embeds, eval_cfg):
        """Action codes for each evaluation clip, read as the encoder reads them."""
        height = int(eval_cfg.get('height', 480))
        width = int(eval_cfg.get('width', 832))
        fps = int(eval_cfg.get('fps', 16))
        num_frames = int(eval_cfg.get('num_frames', self.num_rgb_frames))
        codes = {}
        for entry in eval_cfg['clips']:
            name = str(entry['name'])
            batch = None
            if self.encoder_spec.reads_pixels:
                # Only an encoder that reads pixels needs the clip decoded again,
                # at the clip's own frame rate as its latents were; the others
                # take the latents these clips were encoded into.
                clip_fps = int(entry.get('fps', fps))
                pixels = self.load_eval_clip(entry['path'], height, width, clip_fps, num_frames=num_frames).to(
                    device=self.config.device, dtype=self.config.dtype
                )
                batch = {'pixel_values': pixels}
            codes[name] = self.encode_actions(clips[name], prompt_embeds, batch=batch)['action']
        return codes

    def eval_no_action_condition(self, reference_code):
        """Actions for the no-action control, in whichever form this run's dropout trained.

        ``None`` means no action bias reaches the world model at all; packed
        zeros mean it receives the bias a zero code produces.
        """
        if self.null_action == NULL_ACTION_NO_BIAS:
            return None
        return self.pack_actions(torch.zeros_like(reference_code))

    @staticmethod
    def eval_action_sources(names, transfer_mode):
        """``{first-frame clip: [action source, ...]}`` for the evaluation matrix."""
        if transfer_mode == 'matrix':
            per_name = {name: list(names) for name in names}
        else:
            per_name = {name: [name, names[(index + 1) % len(names)]] for index, name in enumerate(names)}
        for name in names:
            per_name[name] = per_name[name] + [SHUFFLED_ACTION, ZERO_ACTION]
        return per_name

    @staticmethod
    def transfer_consistency_metrics(names, action_sources, generated, latent_motion):
        """Does one action drive the same motion on every first frame?

        * ``transfer/consistency_{a}``: disagreement between the latent motion
          action ``a`` induces on different first frames (lower is better).
        * ``transfer/separation_{f}``: disagreement between different actions on
          first frame ``f`` (higher is better).
        * ``transfer/snr``: mean separation over mean consistency.
        """
        metrics = {}

        def pairwise_mean(tensors):
            total, count = 0.0, 0
            for i in range(len(tensors)):
                for j in range(i + 1, len(tensors)):
                    total += float(((tensors[i] - tensors[j]) ** 2).mean())
                    count += 1
            return total / count if count else None

        consistency, separation = [], []
        for source in names:
            motions = [latent_motion(generated[(name, source)]) for name in names if (name, source) in generated]
            value = pairwise_mean(motions)
            if value is not None:
                metrics[f'transfer/consistency_{source}'] = value
                consistency.append(value)
        for name in names:
            motions = [
                latent_motion(generated[(name, source)])
                for source in action_sources[name]
                if source not in (ZERO_ACTION, SHUFFLED_ACTION)
            ]
            value = pairwise_mean(motions)
            if value is not None:
                metrics[f'transfer/separation_{name}'] = value
                separation.append(value)
        if consistency and separation:
            mean_consistency = sum(consistency) / len(consistency)
            if mean_consistency > 0:
                metrics['transfer/snr'] = (sum(separation) / len(separation)) / mean_consistency
        return metrics

    def sample(self, cfg, global_step=0):
        """Run the fixed-clip latent-action evaluation, optionally on EMA weights."""
        eval_cfg = (cfg.get('val') or {}).get('latent_action_eval') or {}
        use_ema = bool(eval_cfg.get('use_ema', False)) and hasattr(self, 'ema')
        if use_ema:
            self.ema.store(self.world_model)
            self.ema.copy_to(self.world_model)
        try:
            metrics = self.latent_action_eval(cfg, global_step=global_step)
        finally:
            if use_ema:
                self.ema.restore(self.world_model)
        if metrics and is_main_process():
            report = ', '.join(f'{key} {value:.4f}' for key, value in metrics.items())
            get_logger().info(f'[latent-action eval @ {global_step}] {report}')
        return metrics

    def get_optimizer_param_groups(self):
        """Named parameter groups across both models, with per-group learning rates."""
        groups = {name: [] for name in PARAM_GROUP_NAMES}
        for name, parameter in self.world_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if 'action_embedder' in name:
                groups[GROUP_ACTION_EMBEDDER].append(parameter)
            elif any(key in name for key in LORA_STATE_KEYS):
                groups[GROUP_WORLD_MODEL_LORA].append(parameter)
            else:
                groups[GROUP_WORLD_MODEL_OTHER].append(parameter)

        if hasattr(self, 'action_encoder'):
            for name, parameter in self.action_encoder.named_parameters():
                if not parameter.requires_grad:
                    continue
                if 'idm_head' in name:
                    groups[GROUP_READOUT].append(parameter)
                elif any(key in name for key in LORA_STATE_KEYS):
                    groups[GROUP_ACTION_ENCODER_LORA].append(parameter)
                else:
                    groups[GROUP_ACTION_ENCODER_OTHER].append(parameter)

        parameter_groups = []
        for name, learning_rate in (
            (GROUP_WORLD_MODEL_LORA, self.world_model_lora_lr),
            (GROUP_ACTION_EMBEDDER, self.action_embedder_lr),
            (GROUP_ACTION_ENCODER_LORA, self.action_encoder_lora_lr),
            (GROUP_READOUT, self.readout_lr),
            (GROUP_WORLD_MODEL_OTHER, None),
            (GROUP_ACTION_ENCODER_OTHER, None),
        ):
            parameters = groups[name]
            if not parameters:
                continue
            group = {'name': name, 'params': parameters}
            if learning_rate is not None:
                group['lr'] = float(learning_rate)
            parameter_groups.append(group)

        if not parameter_groups:
            raise RuntimeError('No trainable parameters were found in either model.')
        encoder_groups = {GROUP_ACTION_ENCODER_LORA, GROUP_READOUT, GROUP_ACTION_ENCODER_OTHER}
        if any(group['name'] in encoder_groups for group in parameter_groups):
            # The encoder re-rates its own groups. A frozen encoder (world model
            # training) has none, and nothing of it to re-rate.
            parameter_groups = self.encoder_spec.configure_optimizer_groups(self, parameter_groups)
        for group in parameter_groups:
            count = sum(p.numel() for p in group['params']) / 1e6
            get_logger().info(f'param group {group["name"]}: {count:.3f} M, lr={group.get("lr", "default")}')
        return parameter_groups

    # -- checkpointing -----------------------------------------------------

    def validate_idm_contract(self, state, label):
        """Refuse an encoder state whose action space was defined by a different encoder."""
        self.encoder_spec.validate_state_contract(state or {}, self._idm_contract, label)

    def load_latent_action_weights(self, path):
        """Restore the halves of a latent-action state named in ``latent_action_state_parts``.

        ``path`` is a ``latent_action_state.pt`` written by this trainer, or a
        released model (its directory or its ``latent_action_model/``), so a run
        can continue from a published checkpoint. Weights only: optimizer, step
        and RNG are untouched. Every trainable tensor of a restored half must be
        present in the source.
        """
        if Path(path).is_dir():
            return self._load_released_latent_action_model(Path(path))
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        # Read once, memory-mapped: the identity check needs only a few small keys
        # and the weights are copied out tensor by tensor below.
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True) if rank == 0 else None
        metadata = [None]
        if rank == 0:
            keys = ('idm_backend', 'idm_contract', *self.encoder_spec.legacy_contract_state_keys)
            metadata[0] = {key: state[key] for key in keys if key in state}
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(metadata, src=0)
        self.validate_idm_contract(metadata[0], str(path))

        for module, key, tag in (
            (self.world_model, 'decoder', 'world model'),
            (self.action_encoder, 'encoder', 'action encoder'),
        ):
            if key not in self.latent_action_state_parts:
                get_logger().info(f'Skipping the {tag} half of {path} by config.')
                continue
            load_full_state_dict(
                module,
                state[key] if state is not None else {},
                required={name for name, parameter in module.named_parameters() if parameter.requires_grad},
                label=tag,
                source_name=str(path),
                device=self.config.device,
                unexpected='warn',
            )
        get_logger().info(f'Loaded latent-action state from {path}')

    def _load_released_latent_action_model(self, path: Path):
        """Restore from a released ``latent_action_model/``; see :mod:`o2laf.inference.hub`."""
        from o2laf.inference import hub

        component = path / hub.LATENT_ACTION_MODEL_DIR if (path / hub.MODEL_INDEX_NAME).is_file() else path
        config = hub.load_component_config(component)
        if config.get('_class_name') != 'LatentActionModel':
            raise ValueError(f'{component} is not a released latent action model.')
        encoder_config, world_model_config = config[hub.ACTION_ENCODER_PART], config[hub.WORLD_MODEL_PART]
        if encoder_config.get('action_encoder') != self.encoder_spec.name:
            raise RuntimeError(f'{component} holds a {encoder_config.get("action_encoder")!r} action encoder, '
                               f'not this run\'s {self.encoder_spec.name!r}.')
        if int(encoder_config['latent_dim']) != self.latent_action_dim:
            raise RuntimeError(f'{component} has latent_dim {encoder_config["latent_dim"]}, not {self.latent_action_dim}.')
        validate_lora_config(self.world_model_lora_config, world_model_config.get('lora_config'), 'world model')
        validate_lora_config(self.action_encoder_lora_config, encoder_config.get('lora_config'), 'action encoder')
        self.validate_idm_contract(
            {key: encoder_config[key] for key in ('idm_backend', 'idm_contract') if key in encoder_config}, str(component)
        )

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        for module, part, key, tag in (
            (self.world_model, hub.WORLD_MODEL_PART, 'decoder', 'world model'),
            (self.action_encoder, hub.ACTION_ENCODER_PART, 'encoder', 'action encoder'),
        ):
            if key not in self.latent_action_state_parts:
                get_logger().info(f'Skipping the {tag} part of {component} by config.')
                continue
            source = {}
            if rank == 0:
                stored = hub.load_component_state(component, part)
                names = [name for name, _ in module.named_parameters()] + [name for name, _ in module.named_buffers()]
                mapping = hub.hub_keys_to_model_keys(stored, names)
                source = {mapping[name]: tensor for name, tensor in stored.items()}
            load_full_state_dict(
                module,
                source,
                required={name for name, parameter in module.named_parameters() if parameter.requires_grad},
                label=tag,
                source_name=str(component),
                device=self.config.device,
                unexpected='warn',
            )
        get_logger().info(f'Loaded the released latent action model from {component} (step {config.get("global_step")}).')

    def _checkpoint_module(self):
        """Both trained trees under one root, for the distributed state-dict APIs.

        The optimizer owns parameters from two FSDP2 trees, and the distributed
        APIs map parameters through the single module they are handed. Built
        fresh each call and never stored, so neither model appears twice in this
        trainer's own module tree.
        """
        return nn.ModuleDict({'decoder': self.world_model, 'encoder': self.action_encoder})

    def _collect_trainable(self, module, tag, param_names_to_optimize=None, param_names_to_exclude=()):
        """Full CPU state dict of one model's trainable tensors, on rank 0 only."""
        # Select by the configured name filters when given, so the exported set is
        # exactly what the config declares trainable -- the same filters
        # set_params_to_optimize applied -- rather than whatever else may have
        # had requires_grad switched on since.
        if param_names_to_optimize is None:
            trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
        else:
            include = tuple(param_names_to_optimize)
            exclude = tuple(param_names_to_exclude)
            trainable = {
                name
                for name, _ in module.named_parameters()
                if any(keyword in name for keyword in include) and not any(keyword in name for keyword in exclude)
            }
        state = get_model_state_dict(
            module, options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
        )
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return None
        keep = {name: value.detach().cpu() for name, value in state.items() if name in trainable}
        absent = trainable - set(keep)
        if absent:
            raise RuntimeError(
                f'The full distributed state dict for {tag} omitted {len(absent)} trainable tensors, e.g. {sorted(absent)[:3]}.'
            )
        if not keep:
            raise RuntimeError(f'The full distributed state dict for {tag} contained no trainable tensors.')
        return keep

    def export_latent_action_state(self) -> dict | None:
        """The portable state: both models' trainable tensors plus geometry metadata."""
        world_model_state = self._collect_trainable(
            self.world_model, 'the world model',
            self.world_model_param_names_to_optimize, self.world_model_param_names_to_exclude,
        )
        action_encoder_state = self._collect_trainable(
            self.action_encoder, 'the action encoder',
            self.action_encoder_param_names_to_optimize, self.action_encoder_param_names_to_exclude,
        )
        if world_model_state is None or action_encoder_state is None:
            return None
        return {
            'decoder': world_model_state,
            'encoder': action_encoder_state,
            'latent_action_dim': self.latent_action_dim,
            'num_rgb_frames': self.num_rgb_frames,
            'num_latent_frames': self.num_latent_frames,
            'beta': self.beta,
            'decoder_lora_config': self.world_model_lora_config,
            'encoder_lora_config': self.action_encoder_lora_config,
            # Which encoder defined this action space, so a later run cannot
            # load it into a different one.
            'idm_backend': self.encoder_spec.name,
            'idm_contract': dict(self._idm_contract),
        }

    def _gather_optimizer_state(self, optimizer):
        """Full, unsharded optimizer state for both models, or None on failure.

        A failure is logged rather than raised: losing the moments costs a few
        dozen steps, while raising would kill the run at the moment it tries to
        make itself recoverable.
        """
        try:
            return get_optimizer_state_dict(
                self._checkpoint_module(), optimizer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        except Exception as error:  # noqa: BLE001 - see the docstring
            get_logger().error(
                f'Could not gather the optimizer state ({type(error).__name__}: {error}). '
                'Saving weights only; a resume from this checkpoint restarts AdamW moments from zero.'
            )
            return None

    def get_lightweight_train_state(self, optimizer, lr_scheduler, global_step, world_size, random_states):
        """Trainable tensors of both models plus optimizer state. Every rank must call this."""
        payload = self.export_latent_action_state()
        # A collective; it must run on every rank before the rank-0 early return.
        optimizer_state = self._gather_optimizer_state(optimizer)
        if payload is None:
            return None
        return {
            **payload,
            'optimizer': optimizer_state,
            'lr_scheduler': lr_scheduler.state_dict(),
            'global_step': int(global_step),
            'world_size': int(world_size),
            'random_states': random_states,
        }

    def load_lightweight_train_state(self, state, optimizer, lr_scheduler):
        self.validate_idm_contract(state, 'lightweight training state')
        if 'decoder_lora_config' in state:
            validate_lora_config(self.world_model_lora_config, state['decoder_lora_config'], 'world model')
        if 'encoder_lora_config' in state:
            validate_lora_config(self.action_encoder_lora_config, state['encoder_lora_config'], 'action encoder')
        for module, key, tag in (
            (self.world_model, 'decoder', 'world model'),
            (self.action_encoder, 'encoder', 'action encoder'),
        ):
            # FSDP2 parameters are DTensors even at world size 1, so the
            # distributed loader converts the full tensors to each placement.
            load_full_state_dict(
                module,
                state[key],
                required={name for name, parameter in module.named_parameters() if parameter.requires_grad},
                label=tag,
                source_name='the lightweight checkpoint',
                device=self.config.device,
                unexpected='warn',
            )
        optimizer_state = state.get('optimizer')
        if optimizer_state is None:
            get_logger().warning('The checkpoint carries no optimizer state, so AdamW moments restart from zero.')
        else:
            set_optimizer_state_dict(
                self._checkpoint_module(),
                optimizer,
                optim_state_dict=optimizer_state,
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
            )
        lr_scheduler.load_state_dict(state['lr_scheduler'])
