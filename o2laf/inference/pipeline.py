"""Standalone inference: encode actions from a source video and generate from a first frame.

Two modes share one action encoder:

* ``causal``: the causal world model trained jointly with the encoder;
* ``bidirectional``: the fully fine-tuned bidirectional world model trained
  against the frozen encoder.

A model is a directory in the layout of :mod:`o2laf.inference.hub`, local or on
the Hugging Face Hub. The base model (Wan 2.2 TI2V-5B) supplies the VAE, the
scheduler and the transformer the action encoder and causal world model adapt.
Its text encoder is loaded only for a model whose fixed prompt has no embedding
packaged with this library (:mod:`o2laf.inference.prompts`).
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from peft import LoraConfig, get_peft_model

from o2laf.action_encoders.base import build_action_encoder_spec, unwrap_peft
from o2laf.action_encoders.st_transformer import STTransformerActionEncoder
from o2laf.action_encoders.vjepa2 import VJEPA2ActionEncoder
from o2laf.action_encoders.wan_dit import WanDiTActionEncoder
from o2laf.inference import hub
from o2laf.inference.encoder_windows import plan_encoder_windows, usable_transitions
from o2laf.inference.prompts import fixed_prompt_embeds
from o2laf.inference.sampling import MAX_RGB_FRAMES, ActionGuidance, sample_with_guidance
from o2laf.schedulers.flow_match_euler import FlowMatchEulerScheduler
from o2laf.utils.action_util import (
    NULL_ACTION_NO_BIAS,
    prepend_action_anchor,
    process_actions_for_latent_space,
    validate_null_action,
)
from o2laf.utils.lora_util import LORA_ADAPTER_NAME, LORA_STATE_KEYS, filter_learnable_module
from o2laf.world_models import build_world_model_from_config

INFERENCE_MODES = hub.INFERENCE_MODES


def _cast_and_freeze(module, device, dtype):
    module = module.to(device=device, dtype=dtype)
    wrong = [
        (name, parameter.dtype)
        for name, parameter in module.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    ]
    if wrong:
        raise RuntimeError(f'{len(wrong)} parameters were not cast to {dtype}, e.g. {wrong[:3]}.')
    module.requires_grad_(False)
    module.eval()
    return module


def _selected_parameter_names(module, config):
    include = tuple(config.get('param_names_to_optimize', ()))
    exclude = tuple(config.get('param_names_to_exclude', ()))
    return {
        name
        for name, _ in module.named_parameters()
        if any(keyword in name for keyword in include) and not any(keyword in name for keyword in exclude)
    }


def _load_selected_state(module, stored, config, label):
    """Load stored tensors into ``module``; every tensor its training optimised must be among them."""
    mapping = hub.hub_keys_to_model_keys(stored, module.state_dict())
    source = {mapping[name]: tensor for name, tensor in stored.items()}
    required = _selected_parameter_names(module, config)
    absent = required - set(source)
    if absent:
        raise RuntimeError(f'{len(absent)} required {label} tensors are absent, e.g. {sorted(absent)[:3]}.')
    missing, unexpected = module.load_state_dict(source, strict=False)
    if unexpected:
        raise RuntimeError(f'{len(unexpected)} unexpected {label} tensors, e.g. {unexpected[:3]}.')
    missing_required = required.intersection(missing)
    if missing_required:
        raise RuntimeError(f'{len(missing_required)} required {label} tensors did not load, e.g. {sorted(missing_required)[:3]}.')


def _attach_lora(module, lora_cfg, label):
    targets = filter_learnable_module(module, lora_cfg['target_module_name'])
    if not targets:
        raise ValueError(f'{label} LoRA selection matched no modules.')
    return get_peft_model(
        module,
        LoraConfig(
            r=int(lora_cfg['lora_rank']),
            lora_alpha=int(lora_cfg['lora_alpha']),
            target_modules=targets,
            lora_dropout=float(lora_cfg.get('lora_dropout', 0.0)),
            bias='none',
        ),
        adapter_name=LORA_ADAPTER_NAME,
    )


def _build_world_model(state: dict[str, torch.Tensor], config: dict[str, Any], base_model: str, device, dtype,
                       **load_kwargs):
    world_model = build_world_model_from_config(
        {'model_name': base_model, 'model_type': config['_class_name']},
        {'latent_action_dim': config['action_dim'], 'gamma_init': config['action_gain_init']},
        num_latent_frames=int(config['num_latent_frames']),
        causal=bool(config['causal']),
        dtype=dtype,
        **load_kwargs,
    )
    if config.get('lora_config') is not None:
        world_model = _attach_lora(world_model, config['lora_config'], 'world model')
    label = f'{"causal" if config["causal"] else "bidirectional"} world model'
    _load_selected_state(world_model, state, config, label)
    return _cast_and_freeze(world_model, device, dtype)


def _build_wan_dit_encoder(config, base_model, dtype, **load_kwargs):
    encoder = WanDiTActionEncoder.from_pretrained(base_model, subfolder='transformer', torch_dtype=dtype, **load_kwargs)
    encoder.setup_readout(latent_dim=int(config['latent_dim']), query_init_std=float(config['query_init_std']))
    return encoder


def _build_vjepa2_encoder(config, base_action_encoder):
    """The trained parts come from the release; the frozen ViT-g trunk is the user's own copy."""
    if not base_action_encoder:
        raise ValueError(
            'This model needs the V-JEPA 2 ViT-g checkpoint it was trained on. Pass base_action_encoder=<path to '
            f'{config.get("base_action_encoder_name", "vitg-384.pt")}>.'
        )
    return VJEPA2ActionEncoder(
        checkpoint=str(base_action_encoder),
        checkpoint_key=str(config.get('checkpoint_key', 'target_encoder')),
        architecture=str(config.get('architecture', 'vit_giant_xformers')),
        image_size=int(config['image_size']),
        pretrained_num_frames=int(config['pretrained_num_frames']),
        patch_size=int(config['patch_size']),
        tubelet_size=int(config['tubelet_size']),
        latent_dim=int(config['latent_dim']),
        query_init_std=float(config['query_init_std']),
        prepend_first_frame=bool(config['prepend_first_frame']),
        checkpoint_resolution=int(config.get('checkpoint_resolution', 384)),
        allow_resolution_override=bool(config.get('allow_resolution_override', False)),
        load_checkpoint=True,
    )


def _build_st_transformer_encoder(config):
    """Trained from scratch, so the release holds every weight it needs."""
    return STTransformerActionEncoder(
        latent_dim=int(config['latent_dim']),
        model_dim=int(config['model_dim']),
        enc_blocks=int(config['enc_blocks']),
        num_heads=int(config['num_heads']),
        patch_size=int(config['patch_size']),
        height=int(config['height']),
        width=int(config['width']),
    )


def _build_action_encoder(state: dict[str, torch.Tensor], config: dict[str, Any], base_model: str, device, dtype,
                          base_action_encoder=None, **load_kwargs):
    name = config.get('action_encoder')
    if name == 'wan_dit':
        encoder = _build_wan_dit_encoder(config, base_model, dtype, **load_kwargs)
    elif name == 'vjepa2':
        encoder = _build_vjepa2_encoder(config, base_action_encoder)
    elif name == 'st_transformer':
        encoder = _build_st_transformer_encoder(config)
    else:
        raise NotImplementedError(f'Unknown action encoder {name!r}; expected wan_dit, vjepa2 or st_transformer.')
    if config.get('lora_config') is not None:
        encoder = _attach_lora(encoder, config['lora_config'], 'action encoder')
        adapted_head = [
            name for name, _ in encoder.named_parameters()
            if 'idm_head' in name and any(key in name for key in LORA_STATE_KEYS)
        ]
        if adapted_head:
            raise RuntimeError('Action encoder LoRA unexpectedly reached the readout head.')
    _load_selected_state(encoder, state, config, 'action encoder')
    return _cast_and_freeze(encoder, device, dtype)


def _check_action_space(index, encoder_config, world_model_config, world_model_name) -> str:
    """Every part must describe the same action space, or the world model reads codes it never saw."""
    action_space_id = str(index['action_space_id'])
    for label, recorded in (('action encoder', encoder_config.get('action_space_id')),
                            (f'world model {world_model_name!r}', world_model_config.get('action_space_id'))):
        if recorded != action_space_id:
            raise RuntimeError(f'The {label} was trained in action space {recorded}, but the model is {action_space_id}.')
    return action_space_id


@dataclass
class LatentActionPipeline:
    """One action encoder plus one world model (causal or bidirectional), ready to sample."""

    model_dir: Path
    model_index: dict[str, Any]
    action_space_id: str
    mode: str
    # What this mode's world model learned "no action" to be, from the dropout
    # its training used. Guidance and the no-action control must ask for the
    # same thing, or the unconditional branch is off-distribution.
    null_action: str
    world_model: torch.nn.Module
    action_encoder: torch.nn.Module
    vae: AutoencoderKLWan
    scheduler: FlowMatchEulerScheduler
    prompt_embeddings: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    # The clip length this mode's world model was trained on, and its default.
    num_rgb_frames: int
    num_latent_frames: int
    num_context_frames: int
    # The clip length the action encoder was trained on; longer source videos
    # are read in windows of this length (see o2laf.inference.encoder_windows).
    encoder_window_rgb_frames: int
    # Which encoder this model carries, and whether it reads the clip's pixels
    # or the VAE latents of those pixels.
    action_encoder_name: str
    encoder_reads_pixels: bool
    action_dim: int
    action_compression_ratio: int
    height: int
    width: int
    fps: int
    world_model_step: int

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        mode: str = 'bidirectional',
        base_model: str | Path | None = None,
        base_action_encoder: str | Path | None = None,
        device: str | torch.device = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        verify_action_space: bool = False,
    ) -> 'LatentActionPipeline':
        """Load a model from a Hub repo id or a local directory.

        Args:
            mode: ``'bidirectional'`` (the model the paper evaluates) or ``'causal'``.
            base_model: the Wan 2.2 TI2V-5B base, as a Hub id or local path;
                defaults to the one the model records.
            base_action_encoder: the frozen trunk an action encoder was trained
                on, for the encoders that have one (V-JEPA 2's ViT-g checkpoint).
            verify_action_space: recompute the action encoder's content hash and
                check it against the recorded action space (a few seconds).
        """
        if mode not in INFERENCE_MODES:
            raise ValueError(f'mode must be one of {list(INFERENCE_MODES)}, got {mode!r}.')
        device = torch.device(device)
        root = hub.resolve_model_dir(
            pretrained_model_name_or_path, mode=mode, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only, token=token,
        )
        index = hub.read_model_index(root)
        encoder_at, world_model_at = hub.component_paths(index, mode)
        mode_config = index['modes'][mode]
        encoder_config = hub.load_part_config(root, encoder_at)
        world_model_config = hub.load_part_config(root, world_model_at)
        action_space_id = _check_action_space(index, encoder_config, world_model_config, world_model_at['component'])
        encoder_state = hub.load_part_state(root, encoder_at)
        if verify_action_space:
            actual = 'sha256:' + hub.tensor_content_sha256(encoder_state)
            if actual != action_space_id:
                raise RuntimeError(f'The action encoder weights hash to {actual}, not the recorded {action_space_id}.')
        # Where to load the base model from; the embedding depends on which one
        # the model was trained on, which is what the index records.
        base_model_path = str(base_model or index['base_model'])

        # The base model comes from the same Hub cache, credentials and offline
        # setting as this model; its own revision is whatever base_model names.
        base_kwargs = {'cache_dir': cache_dir, 'local_files_only': local_files_only, 'token': token}
        prompt_embeddings = fixed_prompt_embeds(
            base_model=index['base_model'], prompt=index['fixed_prompt'], base_model_path=base_model_path,
            device=device, dtype=dtype, **base_kwargs,
        )
        vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder='vae', torch_dtype=dtype, **base_kwargs).to(device)
        vae.requires_grad_(False)
        vae.eval()
        world_model = _build_world_model(
            hub.load_part_state(root, world_model_at), world_model_config, base_model_path, device, dtype, **base_kwargs
        )
        action_encoder = _build_action_encoder(
            encoder_state, encoder_config, base_model_path, device, dtype,
            base_action_encoder=base_action_encoder, **base_kwargs,
        )
        del encoder_state
        scheduler = FlowMatchEulerScheduler.from_pretrained(
            base_model_path, subfolder='scheduler', **mode_config['scheduler_cfg'], **base_kwargs
        )

        pipeline = cls(
            model_dir=root,
            model_index=index,
            action_space_id=action_space_id,
            mode=mode,
            null_action=validate_null_action(mode_config['null_action']),
            world_model=world_model,
            action_encoder=action_encoder,
            vae=vae,
            scheduler=scheduler,
            prompt_embeddings=prompt_embeddings,
            device=device,
            dtype=dtype,
            num_rgb_frames=int(mode_config['num_rgb_frames']),
            num_latent_frames=int(world_model_config['num_latent_frames']),
            num_context_frames=int(mode_config['num_context_frames']),
            encoder_window_rgb_frames=int(encoder_config['num_rgb_frames']),
            action_encoder_name=str(encoder_config['action_encoder']),
            encoder_reads_pixels=bool(build_action_encoder_spec(encoder_config['action_encoder']).reads_pixels),
            action_dim=int(index['action_dim']),
            action_compression_ratio=int(index['codes_per_transition']),
            height=int(mode_config['height']),
            width=int(mode_config['width']),
            fps=int(mode_config['fps']),
            world_model_step=int(world_model_config['global_step']),
        )
        gc.collect()
        return pipeline

    def preprocess_frames(self, frames) -> torch.Tensor:
        """uint8 ``[T,H,W,3]`` frames to normalised model pixels ``[1,T,3,H,W]``."""
        # Decoded video frames are often a read-only array; copy rather than wrap
        # it, which torch warns about. ``.float()`` below copies anyway.
        pixels = torch.as_tensor(np.array(frames)) if isinstance(frames, np.ndarray) else torch.as_tensor(frames)
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError(f'frames must be [T,H,W,3], got {tuple(pixels.shape)}.')
        pixels = pixels.float().permute(0, 3, 1, 2).div_(255.0)
        pixels = F.interpolate(pixels, size=(self.height, self.width), mode='bilinear', align_corners=False, antialias=True)
        pixels = pixels.sub_(0.5).div_(0.5)
        return pixels.unsqueeze(0)

    @torch.no_grad()
    def encode_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Normalised ``[B,T,3,H,W]`` pixels to Wan latents."""
        if pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError(f'pixels must be [B,T,3,H,W], got {tuple(pixels.shape)}.')
        video = pixels.permute(0, 2, 1, 3, 4).contiguous().to(device=self.device, dtype=self.dtype)
        moments = self.vae._encode(video)
        mean, _ = torch.chunk(moments, 2, dim=1)
        latent_mean = torch.as_tensor(self.vae.config.latents_mean, device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        latent_std = torch.as_tensor(self.vae.config.latents_std, device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        return ((mean.float() - latent_mean) / latent_std).to(self.dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalised latents to ``[B,T,3,H,W]`` pixels in ``[0,1]``."""
        z_dim = int(self.vae.config.z_dim)
        latent_mean = torch.as_tensor(self.vae.config.latents_mean, device=latents.device, dtype=self.vae.dtype).view(1, z_dim, 1, 1, 1)
        inverse_std = 1.0 / torch.as_tensor(self.vae.config.latents_std, device=latents.device, dtype=self.vae.dtype).view(
            1, z_dim, 1, 1, 1
        )
        decoded = self.vae.decode(latents.to(self.vae.dtype) / inverse_std + latent_mean, return_dict=False)[0]
        decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
        return decoded.permute(0, 2, 1, 3, 4).contiguous()

    @property
    def temporal_compression_ratio(self) -> int:
        return int(self.model_index['temporal_compression_ratio'])

    def num_latent_frames_for(self, num_rgb_frames: int) -> int:
        """Latent frames of a ``num_rgb_frames`` clip, which must be ``1 + 4k`` frames long."""
        ratio = self.temporal_compression_ratio
        if num_rgb_frames < 1 + ratio or (num_rgb_frames - 1) % ratio:
            raise ValueError(f'A clip must be {ratio}k+1 frames long with k >= 1, got {num_rgb_frames}.')
        return (num_rgb_frames - 1) // ratio + 1

    def num_rgb_frames_for(self, num_latent_frames: int) -> int:
        return 1 + (num_latent_frames - 1) * self.temporal_compression_ratio

    @property
    def encoder_spec(self):
        return build_action_encoder_spec(self.action_encoder_name)

    @torch.no_grad()
    def encode_source_actions(self, frames) -> torch.Tensor:
        """Actions ``[1, K, D]`` of a source video of any length, as uint8 ``[T,H,W,3]`` frames at ``fps``.

        The action encoder reads the video in windows of the length it was
        trained on and keeps the first half of each window's transitions
        (:func:`~o2laf.inference.encoder_windows.plan_encoder_windows`). Each
        window is VAE-encoded on its own. ``K`` is ``(T - 1) // 4`` transitions
        times the codes per transition; up to three trailing frames are dropped.
        """
        num_frames = len(frames)
        if num_frames > MAX_RGB_FRAMES:
            raise ValueError(f'The source video has {num_frames} frames; at most {MAX_RGB_FRAMES} are supported.')
        windows = plan_encoder_windows(
            num_frames, window_rgb_frames=self.encoder_window_rgb_frames,
            temporal_compression_ratio=self.temporal_compression_ratio,
        )
        codes = self.action_compression_ratio
        pieces = []
        for window in windows:
            clip = frames[window.start_rgb_frame:window.start_rgb_frame + window.num_rgb_frames]
            pixels = self.preprocess_frames(clip)
            # Each encoder's preprocessing is its own: the Wan VAE for wan_dit,
            # a square crop for vjepa2, a resize for st_transformer.
            actions = (
                self._encode_actions(pixels=pixels) if self.encoder_reads_pixels
                else self._encode_actions(latents=self.encode_latents(pixels))
            )
            pieces.append(actions[:, :window.kept_transitions * codes])
        actions = torch.cat(pieces, dim=1)
        expected = usable_transitions(num_frames, self.temporal_compression_ratio) * codes
        if actions.shape[1] != expected:
            raise RuntimeError(f'Stitched {actions.shape[1]} action codes, expected {expected}.')
        return actions

    @torch.no_grad()
    def encode_actions_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Posterior-mean actions ``[B, F-1, D]`` from the latents of one clip of this mode's length, in one pass.

        :meth:`encode_source_actions` is what the commands use; this single
        pass is how training reads its clips. Only the encoders that read VAE
        latents have this path.
        """
        if self.encoder_reads_pixels:
            raise TypeError(
                f'The {self.action_encoder_name} action encoder reads pixels, not latents; '
                'use encode_source_actions().'
            )
        if latents.shape[2] != self.num_latent_frames:
            raise ValueError(f'Expected {self.num_latent_frames} latent frames, got {latents.shape[2]}.')
        return self._encode_actions(latents=latents)

    def _encode_actions(self, *, latents: torch.Tensor | None = None, pixels: torch.Tensor | None = None) -> torch.Tensor:
        """Posterior-mean actions, handing the encoder whichever form of the clip it reads."""
        if pixels is not None:
            # preprocess_frames leaves the clip on the CPU; encode_latents moves
            # it for the encoders that read latents, so the pixel path must too.
            pixels = pixels.to(device=self.device, dtype=self.dtype)
        reference = pixels if latents is None else latents
        prompt = self.prompt_embeddings
        if prompt.shape[0] != reference.shape[0]:
            prompt = prompt.expand(reference.shape[0], -1, -1)
        outputs = self.encoder_spec.encode(
            self.action_encoder, latents=latents, pixel_values=pixels, prompt_embeds=prompt,
        )
        return outputs['action_mu']

    def num_latent_frames_for_actions(self, actions: torch.Tensor) -> int:
        """Latent frames a clip driven by ``actions`` ``[B, K, D]`` has: one per transition plus the first frame."""
        codes = self.action_compression_ratio
        if actions.ndim != 3 or actions.shape[2] != self.action_dim or actions.shape[1] < codes or actions.shape[1] % codes:
            raise ValueError(
                f'Expected actions [B, k*{codes}, {self.action_dim}] with k >= 1, got {tuple(actions.shape)}.'
            )
        return actions.shape[1] // codes + self.num_context_frames

    def pack_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return process_actions_for_latent_space(
            prepend_action_anchor(actions.to(self.dtype)),
            num_latent_frames=self.num_latent_frames_for_actions(actions),
            compression_ratio=self.action_compression_ratio,
        )

    @torch.no_grad()
    def no_action_condition(self, actions: torch.Tensor) -> torch.Tensor | None:
        """The actions that stand for "no action" in this mode.

        The counterpart of
        :meth:`~o2laf.training.latent_action.LatentActionTrainer.eval_no_action_condition`:
        ``None`` for no bias at all, an all-zero code otherwise.
        """
        if self.null_action == NULL_ACTION_NO_BIAS:
            return None
        return torch.zeros_like(actions)

    def null_action_bias(self, actions: torch.Tensor) -> torch.Tensor | None:
        """The bias that stands for "no action here" in this mode.

        ``None`` means no bias at all. Zero-code mode instead embeds an all-zero
        code, which still carries the embedder's own bias.
        """
        if self.null_action == NULL_ACTION_NO_BIAS:
            return None
        return self.compute_action_bias(torch.zeros_like(actions))

    def compute_action_bias(self, actions: torch.Tensor) -> torch.Tensor:
        return unwrap_peft(self.world_model).compute_action_bias(self.pack_actions(actions))

    def latent_shape(self, num_latent_frames: int | None = None) -> tuple[int, ...]:
        scale = int(self.vae.config.scale_factor_spatial)
        frames = self.num_latent_frames if num_latent_frames is None else int(num_latent_frames)
        return (1, int(self.vae.config.z_dim), frames, self.height // scale, self.width // scale)

    def predict_velocity(self, latents, action_bias, timestep, next_timestep):
        """World model velocity for ``latents`` under a precomputed action bias (None = no action)."""
        prompt = self.prompt_embeddings
        if prompt.shape[0] != latents.shape[0]:
            prompt = prompt.expand(latents.shape[0], -1, -1)
        return self.world_model(
            latents,
            chunk_partition=[1] * latents.shape[2],
            timestep=timestep.to(self.device),
            r_timestep=next_timestep.to(self.device),
            encoder_hidden_states=prompt,
            temb_bias=action_bias,
            return_dict=False,
        )

    def _resolve_num_latent_frames(self, actions: torch.Tensor | None, num_latent_frames: int | None) -> int:
        """The clip length: the actions fix it; without actions it is ``num_latent_frames`` or this mode's default."""
        if actions is not None:
            implied = self.num_latent_frames_for_actions(actions)
            if num_latent_frames is not None and int(num_latent_frames) != implied:
                raise ValueError(f'{actions.shape[1]} action codes drive {implied} latent frames, not {num_latent_frames}.')
            num_latent_frames = implied
        frames = self.num_latent_frames if num_latent_frames is None else int(num_latent_frames)
        num_rgb_frames = self.num_rgb_frames_for(frames)
        if frames <= self.num_context_frames or num_rgb_frames > MAX_RGB_FRAMES:
            raise ValueError(
                f'A clip of {num_rgb_frames} frames is outside the supported range '
                f'({self.num_rgb_frames_for(self.num_context_frames + 1)} to {MAX_RGB_FRAMES}).'
            )
        if frames > self.num_latent_frames:
            warnings.warn(
                f'Generating {num_rgb_frames} frames; the {self.mode} world model was trained on '
                f'{self.num_rgb_frames}, so the rest is extrapolated.',
                stacklevel=3,
            )
        return frames

    @torch.no_grad()
    def generate_latents(
        self,
        *,
        actions: torch.Tensor | None,
        first_frame_latents: torch.Tensor,
        num_latent_frames: int | None = None,
        num_steps: int = 20,
        seed: int = 0,
        guidance: ActionGuidance = ActionGuidance(),
    ) -> torch.Tensor:
        """Sample latents that start at ``first_frame_latents`` and follow ``actions`` (None = no action).

        The clip has one latent frame per action transition plus the first
        frame. Without actions, ``num_latent_frames`` sets the length (default:
        this mode's training length).
        """
        num_latent_frames = self._resolve_num_latent_frames(actions, num_latent_frames)
        bias = self.compute_action_bias(actions) if actions is not None else None
        null_bias = self.null_action_bias(actions) if actions is not None else None
        return sample_with_guidance(
            predict=self.predict_velocity,
            scheduler=self.scheduler,
            latent_shape=self.latent_shape(num_latent_frames),
            action_bias=bias,
            anchor=first_frame_latents,
            num_context_frames=self.num_context_frames,
            num_steps=num_steps,
            seed=seed,
            device=self.device,
            dtype=self.dtype,
            guidance=guidance,
            null_bias=null_bias,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        actions: torch.Tensor | None,
        first_frame,
        num_latent_frames: int | None = None,
        num_steps: int = 20,
        seed: int = 0,
        guidance: ActionGuidance = ActionGuidance(),
    ) -> torch.Tensor:
        """A video ``[1, T, 3, H, W]`` in ``[0, 1]`` that starts at ``first_frame`` and follows ``actions``.

        ``first_frame`` is one uint8 RGB frame, ``[H, W, 3]`` or ``[1, H, W, 3]``,
        resized to the model's resolution. See :meth:`generate_latents` for the
        other arguments.
        """
        # A decoded frame is often a read-only array; preprocess_frames copies it.
        first_frame = torch.as_tensor(np.array(first_frame)) if isinstance(first_frame, np.ndarray) else torch.as_tensor(first_frame)
        if first_frame.ndim == 3:
            first_frame = first_frame.unsqueeze(0)
        first_frame_latents = self.encode_latents(self.preprocess_frames(first_frame))
        latents = self.generate_latents(
            actions=actions, first_frame_latents=first_frame_latents, num_latent_frames=num_latent_frames,
            num_steps=num_steps, seed=seed, guidance=guidance,
        )
        return self.decode_latents(latents)
