"""Action encoder built on a randomly initialised spatio-temporal transformer.

This is the paper's "Scratch-LAM" baseline: the pixel latent-action encoder of
:mod:`o2laf.third_party.pixel_lam`, trained from scratch against the same world
model as the pretrained encoders. It reads every RGB frame and emits one action
per adjacent-frame transition, i.e. four per Wan latent transition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from o2laf.action_encoders.base import (
    GROUP_ACTION_ENCODER_LORA,
    GROUP_ACTION_ENCODER_OTHER,
    GROUP_READOUT,
    ActionEncoderSpec,
    register_action_encoder_spec,
)
from o2laf.third_party.pixel_lam import LatentActionModel
from o2laf.utils.dist_util import fsdp2_wrap
from o2laf.utils.logger_util import get_logger
from o2laf.utils.registry import ACTION_ENCODER_REGISTRY


@ACTION_ENCODER_REGISTRY.register()
class STTransformerActionEncoder(nn.Module):
    """Frame-rate pixel encoder emitting one Gaussian action per RGB transition."""

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        model_dim: int = 1024,
        enc_blocks: int = 16,
        num_heads: int = 16,
        patch_size: int = 16,
        dropout: float = 0.0,
        height: int = 272,
        width: int = 480,
        in_dim: int = 3,
    ) -> None:
        super().__init__()
        if height % patch_size or width % patch_size:
            # Patchify would silently crop the remainder.
            raise ValueError(f"height={height} and width={width} must both be multiples of patch_size={patch_size}")
        self.lam = LatentActionModel(
            in_dim=int(in_dim),
            model_dim=int(model_dim),
            latent_dim=int(latent_dim),
            patch_size=int(patch_size),
            enc_blocks=int(enc_blocks),
            dec_blocks=0,
            num_heads=int(num_heads),
            dropout=float(dropout),
            causal_temporal=True,
            gradient_checkpointing=False,
        )
        self.latent_action_dim = int(latent_dim)
        self.patch_size = int(patch_size)
        self.height = int(height)
        self.width = int(width)

    def enable_gradient_checkpointing(self) -> None:
        self.lam.encoder.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.lam.encoder.gradient_checkpointing = False

    def preprocess(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Wan RGB ``[B,T,C,H,W]`` in ``[-1,1]`` to ``[B,T,H,W,C]`` in ``[0,1]`` at the encoder resolution."""
        if pixel_values.ndim != 5 or pixel_values.shape[2] != 3:
            raise ValueError(f"pixel_values must be [B,T,3,H,W], got {tuple(pixel_values.shape)}")
        batch, frames, channels, _, _ = pixel_values.shape
        video = F.interpolate(
            pixel_values.reshape(batch * frames, channels, *pixel_values.shape[3:]),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        video = video.add(1.0).mul(0.5).clamp(0.0, 1.0)
        video = video.reshape(batch, frames, channels, self.height, self.width)
        target_dtype = self.lam.fc.weight.dtype
        return video.permute(0, 1, 3, 4, 2).contiguous().to(target_dtype)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        video = self.preprocess(pixel_values)
        outputs = self.lam.encode(video, sample_mode="sample" if self.training else "mean")
        # `z_rep` arrives as [B, T-1, 1, D]; the trainer contract is [B, T-1, D].
        action = outputs["z_rep"].squeeze(2)
        return {"action": action.to(video.dtype), "action_mu": outputs["z_mu"], "action_logvar": outputs["z_logvar"]}


@register_action_encoder_spec('st_transformer')
class STTransformerSpec(ActionEncoderSpec):
    """Construct the randomly initialised ST-Transformer action encoder for the trainers."""

    action_ratio_cfg_key = 'st_transformer_action_compression_ratio'
    codes_per_transition = 4
    legacy_contract_state_keys = ('scratch_lam_idm_contract',)

    def prepare_trainer(self, trainer, action_cfg):
        action_cfg = action_cfg or {}
        configured = action_cfg.get('st_transformer_lr')
        if configured is None:
            configured = action_cfg.get('readout_lr', 1.0e-4)
        trainer.st_transformer_lr = float(configured)

    def build_contract(self, trainer, idm_cfg):
        return {
            'version': 1,
            # The architecture tag is part of the checkpoint contract and keeps
            # its historical value so released encoder states still load.
            'architecture': 'external_lam.LatentActionModel',
            'model_dim': int(idm_cfg.get('model_dim', 1024)),
            'enc_blocks': int(idm_cfg.get('enc_blocks', 16)),
            'num_heads': int(idm_cfg.get('num_heads', 16)),
            'patch_size': int(idm_cfg.get('patch_size', 16)),
            'height': int(idm_cfg.get('height', 272)),
            'width': int(idm_cfg.get('width', 480)),
            'causal_temporal': True,
            'action_compression_ratio': trainer.action_compression_ratio,
        }

    def build_encoder(self, trainer, idm_cfg, dtype):
        del dtype
        if 'lora_config' in idm_cfg:
            raise ValueError('The ST-Transformer action encoder trains every parameter and does not support LoRA.')
        encoder = STTransformerActionEncoder(
            latent_dim=trainer.latent_action_dim,
            model_dim=int(idm_cfg.get('model_dim', 1024)),
            enc_blocks=int(idm_cfg.get('enc_blocks', 16)),
            num_heads=int(idm_cfg.get('num_heads', 16)),
            patch_size=int(idm_cfg.get('patch_size', 16)),
            dropout=float(idm_cfg.get('dropout', 0.0)),
            height=int(idm_cfg.get('height', 272)),
            width=int(idm_cfg.get('width', 480)),
        )
        trainer.set_params_to_optimize(
            encoder, idm_cfg['param_names_to_optimize'], idm_cfg.get('param_names_to_exclude', [])
        )
        if not any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError('No ST-Transformer parameter is trainable.')
        if idm_cfg.get('enable_gradient_checkpoint', False):
            encoder.enable_gradient_checkpointing()
        return fsdp2_wrap(encoder, transformer_block_clsname='SpatioTemporalBlock')

    def encode(self, encoder, *, latents=None, pixel_values=None, prompt_embeds=None):
        """The ST-Transformer reads the clip's pixels directly."""
        del latents, prompt_embeds
        return encoder(pixel_values=pixel_values)

    def configure_optimizer_groups(self, trainer, groups):
        groups = super().configure_optimizer_groups(trainer, groups)
        retuned = 0
        for group in groups:
            if group['name'] == GROUP_ACTION_ENCODER_OTHER:
                group['lr'] = trainer.st_transformer_lr
                get_logger().info('ST-Transformer encoder group: lr=%.3g', group['lr'])
                retuned += 1
        if retuned != 1:
            raise RuntimeError(
                f'Expected exactly one {GROUP_ACTION_ENCODER_OTHER!r} group to carry the ST-Transformer encoder, found {retuned}.'
            )
        stray = [group['name'] for group in groups if group['name'] in (GROUP_ACTION_ENCODER_LORA, GROUP_READOUT)]
        if stray:
            raise RuntimeError(f'Unexpected ST-Transformer parameter groups: {stray}.')
        return groups
