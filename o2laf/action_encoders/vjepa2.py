"""Action encoder built on the pretrained V-JEPA 2 video encoder.

The V-JEPA trunk is made causal at tubelet granularity. One shared register is
interleaved between every pair of temporal states and reads exactly the two
states of its transition (see :mod:`o2laf.action_encoders.registers`). Each
tubelet holds two RGB frames, so the encoder emits two actions per Wan latent
transition.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint

from o2laf.action_encoders.base import ActionEncoderSpec, register_action_encoder_spec, unwrap_peft
from o2laf.action_encoders.registers import (
    TransitionRegisterHead,
    build_attention_mask,
    build_layout,
    extract_registers,
    interleave_registers,
)
from o2laf.third_party.vjepa2 import vision_transformer
from o2laf.utils.dist_util import fsdp2_wrap
from o2laf.utils.file_util import sha256_file
from o2laf.utils.logger_util import get_logger
from o2laf.utils.lora_util import LORA_ADAPTER_NAME, LORA_STATE_KEYS, filter_learnable_module
from o2laf.utils.registry import ACTION_ENCODER_REGISTRY


def _rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """V-JEPA 2 rotary transform, duplicating the frequency vector as the release does.

    A reimplementation of ``rotate_queries_or_keys`` from V-JEPA 2 (Meta, MIT;
    vendored in ``o2laf/third_party/vjepa2``), kept bit-compatible with the
    pretrained checkpoint.
    """
    width = x.shape[-1]
    if width % 2:
        raise ValueError(f"RoPE width must be even, got {width}")
    omega = torch.arange(width // 2, dtype=x.dtype, device=x.device)
    omega = 1.0 / (10000 ** (omega / (width / 2.0)))
    frequency = torch.einsum("...,f->...f", pos.to(x.dtype), omega)
    sin = frequency.sin().repeat(1, 1, 1, 2)
    cos = frequency.cos().repeat(1, 1, 1, 2)
    paired = x.unflatten(-1, (-1, 2))
    first, second = paired.unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return (x * cos + rotated * sin).to(x.dtype)


class RegisterRoPEAttention(nn.Module):
    """V-JEPA RoPE attention that gives registers a temporal-only position."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        # Reuse the loaded modules so state-dict names stay `blocks.*.attn.{qkv,proj}`.
        self.qkv = source.qkv
        self.proj = source.proj
        self.attn_drop = source.attn_drop
        self.proj_drop = source.proj_drop
        self.proj_drop_prob = float(source.proj_drop_prob)
        self.num_heads = int(source.num_heads)
        self.head_dim = int(source.head_dim)
        self.scale = float(source.scale)
        self.d_dim = int(source.d_dim)
        self.h_dim = int(source.h_dim)
        self.w_dim = int(source.w_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        height_patches: int,
        width_patches: int,
    ) -> torch.Tensor:
        batch, length, width = x.shape
        if position_ids.shape != (batch, length):
            raise ValueError(f"position_ids must be {(batch, length)}, got {tuple(position_ids.shape)}")
        if attention_mask.shape != (length, length):
            raise ValueError(f"attention_mask must be {(length, length)}, got {tuple(attention_mask.shape)}")

        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, width // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        # Video IDs are flattened patch positions; negative IDs are registers,
        # which get temporal RoPE only (zero height/width = identity rotation).
        special = position_ids < 0
        video_ids = position_ids.clamp_min(0)
        tokens_per_frame = int(height_patches) * int(width_patches)
        temporal = video_ids // tokens_per_frame
        spatial = video_ids - temporal * tokens_per_frame
        height = spatial // int(width_patches)
        horizontal = spatial - height * int(width_patches)
        temporal = torch.where(special, -position_ids - 1, temporal)
        height = torch.where(special, torch.zeros_like(height), height)
        horizontal = torch.where(special, torch.zeros_like(horizontal), horizontal)

        def heads(value_: torch.Tensor) -> torch.Tensor:
            return value_[:, None, :].expand(-1, self.num_heads, -1)

        temporal = heads(temporal)
        height = heads(height)
        horizontal = heads(horizontal)

        offset = 0
        query_d = _rotate_queries_or_keys(query[..., offset : offset + self.d_dim], temporal)
        key_d = _rotate_queries_or_keys(key[..., offset : offset + self.d_dim], temporal)
        offset += self.d_dim
        query_h = _rotate_queries_or_keys(query[..., offset : offset + self.h_dim], height)
        key_h = _rotate_queries_or_keys(key[..., offset : offset + self.h_dim], height)
        offset += self.h_dim
        query_w = _rotate_queries_or_keys(query[..., offset : offset + self.w_dim], horizontal)
        key_w = _rotate_queries_or_keys(key[..., offset : offset + self.w_dim], horizontal)
        offset += self.w_dim
        query = torch.cat([query_d, query_h, query_w, query[..., offset:]], dim=-1)
        key = torch.cat([key_d, key_h, key_w, key[..., offset:]], dim=-1)

        # Additive float mask: boolean SDPA polarity is version-dependent.
        if attention_mask.dtype == torch.bool:
            bias = torch.zeros(attention_mask.shape, dtype=query.dtype, device=query.device)
            attention_mask = bias.masked_fill(~attention_mask, float("-inf"))
        else:
            attention_mask = attention_mask.to(dtype=query.dtype, device=query.device)

        dropout = self.proj_drop_prob if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.proj_drop(self.proj(attended))


class RegisterBlock(nn.Module):
    """V-JEPA block whose whole residual step is one FSDP-visible forward.

    FSDP2 installs parameter hooks on the module passed to ``fully_shard``;
    calling ``block.norm1`` / ``block.attn`` directly would bypass them.
    Checkpoint names stay ``backbone.blocks.N.{norm1,attn,norm2,mlp}``.
    """

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.norm1 = source.norm1
        self.attn = source.attn
        self.drop_path = source.drop_path
        self.norm2 = source.norm2
        self.mlp = source.mlp

    def forward(self, hidden, position_ids, attention_mask, height_patches, width_patches):
        attended = self.attn(
            self.norm1(hidden),
            position_ids=position_ids,
            attention_mask=attention_mask,
            height_patches=height_patches,
            width_patches=width_patches,
        )
        hidden = hidden + self.drop_path(attended)
        return hidden + self.drop_path(self.mlp(self.norm2(hidden)))


@ACTION_ENCODER_REGISTRY.register()
class VJEPA2ActionEncoder(nn.Module):
    """Causal V-JEPA 2 video encoder emitting one action per tubelet transition."""

    checkpoint_ignored_keys = frozenset({'pos_embed'})
    fsdp_block_classname = 'RegisterBlock'

    def __init__(
        self,
        *,
        checkpoint: str,
        checkpoint_key: str = "target_encoder",
        architecture: str = "vit_giant_xformers",
        image_size: int = 256,
        pretrained_num_frames: int = 64,
        patch_size: int = 16,
        tubelet_size: int = 2,
        latent_dim: int = 32,
        query_init_std: float = 0.02,
        prepend_first_frame: bool = True,
        checkpoint_sha256: str | None = None,
        checkpoint_size: int | None = None,
        checkpoint_resolution: int = 384,
        allow_resolution_override: bool = False,
        verify_checkpoint_hash: bool = False,
        load_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if int(tubelet_size) != 2:
            raise ValueError(f"VJEPA2ActionEncoder requires tubelet_size=2, got {tubelet_size}")
        try:
            builder = getattr(vision_transformer, str(architecture))
        except AttributeError as exc:
            raise ValueError(f"Unknown V-JEPA architecture {architecture!r}") from exc
        self.backbone = builder(
            img_size=(int(image_size), int(image_size)),
            patch_size=int(patch_size),
            num_frames=int(pretrained_num_frames),
            tubelet_size=int(tubelet_size),
            use_sdpa=True,
            use_rope=True,
            uniform_power=False,
            use_silu=False,
            wide_silu=True,
        )
        if int(image_size) != int(checkpoint_resolution):
            if not bool(allow_resolution_override):
                raise ValueError(
                    f"image_size={image_size} differs from checkpoint_resolution={checkpoint_resolution}; "
                    "set allow_resolution_override=true only for an intentional resolution change"
                )
            get_logger().warning(
                "V-JEPA resolution override: checkpoint=%d, input=%d", int(checkpoint_resolution), int(image_size)
            )
        if load_checkpoint:
            self.load_pretrained_encoder(
                checkpoint,
                checkpoint_key,
                expected_sha256=checkpoint_sha256,
                expected_size=checkpoint_size,
                verify_sha256=bool(verify_checkpoint_hash),
            )

        wrapped_blocks = []
        for block in self.backbone.blocks:
            source = block.attn
            missing = [name for name in ("qkv", "proj", "head_dim", "d_dim", "h_dim", "w_dim") if not hasattr(source, name)]
            if missing:
                raise TypeError(f"V-JEPA block does not use the expected RoPE attention; missing {missing}")
            block.attn = RegisterRoPEAttention(source)
            wrapped_blocks.append(RegisterBlock(block))
        self.backbone.blocks = nn.ModuleList(wrapped_blocks)

        self.idm_head = TransitionRegisterHead(
            width=int(self.backbone.embed_dim),
            latent_dim=int(latent_dim),
            query_init_std=float(query_init_std),
        )
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.prepend_first_frame = bool(prepend_first_frame)
        self.gradient_checkpointing = False
        self.latent_action_dim = int(latent_dim)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1), persistent=False)

    def load_pretrained_encoder(self, path, checkpoint_key, *, expected_sha256, expected_size, verify_sha256) -> None:
        source = Path(path)
        if expected_size is not None and source.stat().st_size != int(expected_size):
            raise RuntimeError(f"V-JEPA checkpoint size {source.stat().st_size} != expected {expected_size}: {source}")
        if expected_sha256 is not None and verify_sha256:
            actual = sha256_file(source)
            if actual != str(expected_sha256):
                raise RuntimeError(f"V-JEPA checkpoint SHA256 {actual} != expected {expected_sha256}: {source}")
        checkpoint = torch.load(str(source), map_location="cpu", weights_only=True, mmap=True)
        if checkpoint_key not in checkpoint:
            raise KeyError(f"{source} has no {checkpoint_key!r}; available top-level keys: {sorted(checkpoint)}")
        state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in checkpoint[checkpoint_key].items()
        }
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        material_missing = [key for key in missing if key not in self.checkpoint_ignored_keys]
        material_unexpected = [key for key in unexpected if key not in self.checkpoint_ignored_keys]
        if material_missing or material_unexpected:
            raise RuntimeError(
                f"V-JEPA encoder checkpoint mismatch: missing={material_missing[:8]}, unexpected={material_unexpected[:8]}"
            )
        del checkpoint
        get_logger().info("Loaded V-JEPA encoder key %s from %s", checkpoint_key, source)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def preprocess(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Wan RGB ``[B,T,C,H,W]`` in ``[-1,1]`` to V-JEPA ``[B,C,T,H,W]``.

        Center square crop, resize to ``image_size``, ImageNet normalisation,
        then a repeated first frame and an even frame count for tubelets.
        """
        if pixel_values.ndim != 5 or pixel_values.shape[2] != 3:
            raise ValueError(f"pixel_values must be [B,T,3,H,W], got {tuple(pixel_values.shape)}")
        batch, frames, channels, height, width = pixel_values.shape
        size = min(height, width)
        top = (height - size) // 2
        left = (width - size) // 2
        video = pixel_values[:, :, :, top : top + size, left : left + size]
        video = video.reshape(batch * frames, channels, size, size)
        video = F.interpolate(video, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        video = video.reshape(batch, frames, channels, self.image_size, self.image_size)
        video = video.add(1.0).mul(0.5).clamp(0.0, 1.0)
        mean = self.image_mean.to(device=video.device, dtype=video.dtype)
        std = self.image_std.to(device=video.device, dtype=video.dtype)
        video = (video - mean) / std
        if self.prepend_first_frame:
            video = torch.cat([video[:, :1], video], dim=1)
        if video.shape[1] % self.tubelet_size:
            video = torch.cat([video, video[:, -1:]], dim=1)
        target_dtype = self.backbone.patch_embed.proj.weight.dtype
        return video.permute(0, 2, 1, 3, 4).contiguous().to(target_dtype)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        video = self.preprocess(pixel_values)
        _, _, frames, height, width = video.shape
        if frames % self.tubelet_size:
            raise RuntimeError("preprocess failed to make the frame count even")
        states = frames // self.tubelet_size
        height_patches = height // self.patch_size
        width_patches = width // self.patch_size
        spatial = height_patches * width_patches

        hidden = self.backbone.patch_embed(video)
        expected = states * spatial
        if hidden.shape[1] != expected:
            raise RuntimeError(f"Patch embed returned {hidden.shape[1]} tokens, expected {states}x{spatial}={expected}")
        hidden = hidden.reshape(hidden.shape[0], states, spatial, hidden.shape[-1])
        hidden = interleave_registers(hidden, self.idm_head.action_query.to(hidden.dtype))
        _, _, position = build_layout(states, spatial, hidden.device)
        position = position.unsqueeze(0).expand(hidden.shape[0], -1)
        attention_mask = build_attention_mask(states, spatial, hidden.device)

        for block in self.backbone.blocks:
            # Calling the whole block is load-bearing under FSDP2 (see RegisterBlock).
            if self.gradient_checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(
                    lambda value, module=block: module(value, position, attention_mask, height_patches, width_patches),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, position, attention_mask, height_patches, width_patches)
        hidden = self.backbone.norm(hidden)
        return self.idm_head(extract_registers(hidden, states, spatial))


@register_action_encoder_spec('vjepa2')
class VJEPA2Spec(ActionEncoderSpec):
    """Construct the causal V-JEPA2 ViT-g action encoder for the trainers."""

    action_ratio_cfg_key = 'vjepa_action_compression_ratio'
    codes_per_transition = 2
    legacy_contract_state_keys = ('vjepa_idm_contract',)
    # Older states pinned the base checkpoint's byte size and SHA-256 in the contract.
    LEGACY_CONTRACT_FIELDS = ('checkpoint_sha256', 'checkpoint_size')

    def build_contract(self, trainer, idm_cfg):
        return {
            'version': 1,
            'checkpoint_key': str(idm_cfg.get('checkpoint_key', 'target_encoder')),
            'architecture': str(idm_cfg.get('architecture', 'vit_giant_xformers')),
            'image_size': int(idm_cfg.get('image_size', 256)),
            'checkpoint_resolution': int(idm_cfg.get('checkpoint_resolution', 384)),
            'pretrained_num_frames': int(idm_cfg.get('pretrained_num_frames', 64)),
            'patch_size': int(idm_cfg.get('patch_size', 16)),
            'tubelet_size': int(idm_cfg.get('tubelet_size', 2)),
            'prepend_first_frame': bool(idm_cfg.get('prepend_first_frame', True)),
            'action_compression_ratio': trainer.action_compression_ratio,
        }

    def validate_state_contract(self, state, expected, label):
        cleaned = dict(state)
        for key in ('idm_contract', *self.legacy_contract_state_keys):
            contract = cleaned.get(key)
            if isinstance(contract, dict):
                cleaned[key] = {k: v for k, v in contract.items() if k not in self.LEGACY_CONTRACT_FIELDS}
        super().validate_state_contract(cleaned, expected, label)

    def build_encoder(self, trainer, idm_cfg, dtype):
        if not bool(idm_cfg.get('causal_temporal', True)):
            raise ValueError('The V-JEPA2 action encoder requires causal_temporal=true.')
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        checkpoint_sha256 = idm_cfg.get('checkpoint_sha256')
        checkpoint_size = idm_cfg.get('checkpoint_size')
        encoder = VJEPA2ActionEncoder(
            checkpoint=idm_cfg['model_name'],
            checkpoint_key=idm_cfg.get('checkpoint_key', 'target_encoder'),
            architecture=idm_cfg.get('architecture', 'vit_giant_xformers'),
            image_size=int(idm_cfg.get('image_size', 256)),
            pretrained_num_frames=int(idm_cfg.get('pretrained_num_frames', 64)),
            patch_size=int(idm_cfg.get('patch_size', 16)),
            tubelet_size=int(idm_cfg.get('tubelet_size', 2)),
            latent_dim=trainer.latent_action_dim,
            query_init_std=float(idm_cfg.get('query_init_std', 0.02)),
            prepend_first_frame=bool(idm_cfg.get('prepend_first_frame', True)),
            checkpoint_sha256=str(checkpoint_sha256) if checkpoint_sha256 else None,
            checkpoint_size=int(checkpoint_size) if checkpoint_size is not None else None,
            checkpoint_resolution=int(idm_cfg.get('checkpoint_resolution', 384)),
            allow_resolution_override=bool(idm_cfg.get('allow_resolution_override', False)),
            verify_checkpoint_hash=bool(idm_cfg.get('verify_checkpoint_hash', False)),
            # Other ranks receive the weights through fsdp2_wrap's state sync.
            load_checkpoint=(rank == 0),
        )
        encoder.requires_grad_(False)

        if 'lora_config' in idm_cfg:
            lora_cfg = idm_cfg['lora_config']
            targets = filter_learnable_module(encoder, lora_cfg['target_module_name'])
            if not targets:
                raise ValueError('V-JEPA2 LoRA target selection matched no modules.')
            encoder = get_peft_model(
                encoder,
                LoraConfig(
                    r=int(lora_cfg['lora_rank']),
                    lora_alpha=int(lora_cfg['lora_alpha']),
                    target_modules=targets,
                    lora_dropout=float(lora_cfg.get('lora_dropout', 0.0)),
                    bias='none',
                ),
                adapter_name=LORA_ADAPTER_NAME,
            )
            adapted_head = [
                name for name, _ in encoder.named_parameters()
                if 'idm_head' in name and any(key in name for key in LORA_STATE_KEYS)
            ]
            if adapted_head:
                raise ValueError(f'V-JEPA2 LoRA reached the fully trainable readout head: {adapted_head[0]}.')

        trainer.set_params_to_optimize(
            encoder, idm_cfg['param_names_to_optimize'], idm_cfg.get('param_names_to_exclude', [])
        )
        base = unwrap_peft(encoder)
        base.idm_head.requires_grad_(True)
        if idm_cfg.get('enable_gradient_checkpoint', False):
            base.enable_gradient_checkpointing()
        if 'lora_config' not in idm_cfg:
            encoder = encoder.to(dtype=dtype)
        return fsdp2_wrap(encoder, transformer_block_clsname=base.fsdp_block_classname)

    def encode(self, encoder, *, latents=None, pixel_values=None, prompt_embeds=None):
        """V-JEPA2 reads the clip's pixels directly."""
        del latents, prompt_embeds
        return encoder(pixel_values=pixel_values)
