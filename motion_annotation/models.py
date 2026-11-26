"""Unified Wan DiT action-encoder builders for all four motion-annotation modes."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from motion_annotation import ADAPTATION_MODES

DEFAULT_BASE_MODEL = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
DEFAULT_FIXED_PROMPT = 'a video'
DEFAULT_LORA_CONFIG = {
    'lora_rank': 128,
    'lora_alpha': 128,
    'target_module_name': ['blocks.'],
}
DEFAULT_EXCLUDE_PARAMS = ('action_rope_angle',)
LORA_PARAMETER_KEYS = (
    'lora_A',
    'lora_B',
    'lora_embedding_A',
    'lora_embedding_B',
    'lora_magnitude_vector',
)


def unwrap_peft_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying ``WanDiTActionEncoder`` from a PEFT wrapper."""
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def trainable_parameter_contract(model: torch.nn.Module) -> dict[str, list[int]]:
    """Serializable name -> shape contract for every trainable parameter."""
    return {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def assert_matching_trainable_contracts(
    first: dict[str, list[int]],
    second: dict[str, list[int]],
    *,
    first_name: str = 'first',
    second_name: str = 'second',
) -> None:
    """Assert two models optimize the exact same parameter names and shapes."""
    if first != second:
        first_keys = set(first)
        second_keys = set(second)
        mismatches = [
            (key, first[key], second[key])
            for key in sorted(first_keys & second_keys)
            if first[key] != second[key]
        ]
        raise ValueError(
            f'Trainable parameter contracts differ between {first_name} and {second_name}: '
            f'only_{first_name}={sorted(first_keys - second_keys)[:5]}, '
            f'only_{second_name}={sorted(second_keys - first_keys)[:5]}, '
            f'shape_mismatches={mismatches[:5]}.'
        )


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract only trainable tensors onto CPU for compact checkpointing."""
    names = set(trainable_parameter_contract(model))
    result = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in names
    }
    if set(result) != names:
        raise RuntimeError(
            'model.state_dict() does not cover the trainable parameter contract: '
            f'missing={sorted(names - set(result))[:5]}.'
        )
    return result


def load_trainable_state_dict(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    """Restore trainable tensors with strict name and shape verification."""
    expected = trainable_parameter_contract(model)
    if set(state) != set(expected):
        raise ValueError(
            'Trainable state keys do not match model contract: '
            f'missing={sorted(set(expected) - set(state))[:5]}, '
            f'unexpected={sorted(set(state) - set(expected))[:5]}.'
        )
    current = model.state_dict()
    for name, value in state.items():
        if list(value.shape) != expected[name]:
            raise ValueError(
                f'Trainable tensor {name} has shape {list(value.shape)}, expected {expected[name]}.'
            )
        current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))


def configure_wan_trainability(
    model: torch.nn.Module,
    *,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE_PARAMS,
) -> dict[str, list[str]]:
    """Train trunk LoRA and pre-Gaussian readout; keep Gaussian head frozen."""
    model.requires_grad_(False)
    groups: dict[str, list[str]] = {'encoder_lora': [], 'transition_readout': []}
    for name, parameter in model.named_parameters():
        if any(key in name for key in exclude):
            continue
        if any(key in name for key in LORA_PARAMETER_KEYS):
            parameter.requires_grad_(True)
            groups['encoder_lora'].append(name)
        elif 'idm_head' in name and '.outputs.' not in name:
            parameter.requires_grad_(True)
            groups['transition_readout'].append(name)
    if not groups['encoder_lora']:
        raise RuntimeError('Wan LoRA adaptation found no trainable LoRA tensors.')
    if not groups['transition_readout']:
        raise RuntimeError('Wan LoRA adaptation found no trainable transition-readout tensors.')
    accidental_gaussian = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and '.outputs.' in name
    ]
    if accidental_gaussian:
        raise RuntimeError(
            f'Gaussian projection must stay frozen during pre-Gaussian adaptation: {accidental_gaussian[:3]}.'
        )
    return groups


def optimizer_parameter_groups(
    model: torch.nn.Module,
    *,
    encoder_lr: float,
    readout_lr: float | None = None,
) -> list[dict[str, Any]]:
    """Create non-overlapping AdamW parameter groups for LoRA and readout."""
    lora = []
    readout = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(key in name for key in LORA_PARAMETER_KEYS):
            lora.append(parameter)
        elif 'idm_head' in name:
            readout.append(parameter)
        else:
            raise RuntimeError(f'Unexpected trainable parameter outside LoRA/readout: {name}.')
    groups = []
    if lora:
        groups.append({'name': 'encoder_lora', 'params': lora, 'lr': float(encoder_lr)})
    if readout:
        groups.append(
            {
                'name': 'transition_readout',
                'params': readout,
                'lr': float(readout_lr if readout_lr is not None else encoder_lr),
            }
        )
    if not groups:
        raise RuntimeError('Adaptation model has no trainable parameters.')
    return groups


def wan_transition_feature_dim(model: torch.nn.Module) -> int:
    """Return the pre-Gaussian transition feature width (3072 for Wan 2.2 5B)."""
    base = unwrap_peft_model(model)
    head = getattr(base, 'idm_head', None)
    width = getattr(head, 'input_dim', None)
    if width is None:
        raise TypeError('Wan action encoder idm_head does not expose input_dim.')
    return int(width)


def wan_forward_features(
    model: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    return_action_mu: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Differentiable Wan DiT pre-Gaussian transition features (and optional 32-D ``action_mu``).

    Runs ``encode_query_tokens`` on the underlying ``WanDiTActionEncoder`` (whose
    blocks carry PEFT LoRA layers in-place) followed by ``idm_head.transition_features``.
    """
    base = unwrap_peft_model(model)
    action_tokens = base.encode_query_tokens(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
    )
    transition = base.idm_head.transition_features(action_tokens)
    if transition.ndim != 3 or transition.shape[1] != action_tokens.shape[1] - 1:
        raise RuntimeError(
            f'Expected transition features [B, {action_tokens.shape[1] - 1}, D], got {tuple(transition.shape)}.'
        )
    if not return_action_mu:
        return transition
    posterior = base.idm_head.outputs(transition)
    return {
        'transition_features': transition,
        'action_mu': posterior['action_mu'],
    }


def _attach_wan_lora(encoder: torch.nn.Module, lora_cfg: dict[str, Any]) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    from o2laf.utils.lora_util import LORA_ADAPTER_NAME, LORA_STATE_KEYS, filter_learnable_module

    targets = filter_learnable_module(encoder, lora_cfg['target_module_name'])
    if not targets:
        raise ValueError(f'LoRA target selection matched no linear modules: {lora_cfg["target_module_name"]!r}.')
    wrapped = get_peft_model(
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
        name
        for name, _ in wrapped.named_parameters()
        if 'idm_head' in name and any(key in name for key in LORA_STATE_KEYS)
    ]
    if adapted_head:
        raise ValueError(
            f'LoRA unexpectedly targeted readout parameters ({adapted_head[0]}); '
            'restrict target_module_name to the DiT trunk (e.g. ["blocks."]).'
        )
    return wrapped


def build_wan_action_encoder(
    mode: str,
    *,
    model: str | Path | None = None,
    base_model: str | Path | None = None,
    device: str | torch.device = 'cuda',
    base_dtype: torch.dtype = torch.bfloat16,
    lora_rank: int = 128,
    lora_alpha: int = 128,
    latent_dim: int = 32,
    query_init_std: float = 0.02,
    enable_gradient_checkpointing: bool = True,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the Wan DiT action encoder for any of the four adaptation modes.

    * ``wan_base_lora``: Wan 2.2 TI2V-5B base + fresh LoRA + fresh readout (trainable).
    * ``ssl_lora``: O²LaF SSL checkpoint loaded; trunk LoRA + readout trainable,
      Gaussian projection frozen.
    * ``ssl_action_mu`` / ``ssl_pre_gaussian``: O²LaF SSL checkpoint loaded and
      fully frozen for feature extraction.
    """
    if mode not in ADAPTATION_MODES:
        raise ValueError(f'Unknown adaptation mode {mode!r}; expected one of {ADAPTATION_MODES}.')

    from o2laf.action_encoders.wan_dit import WanDiTActionEncoder
    from o2laf.inference import hub

    base_kwargs = {
        'cache_dir': cache_dir,
        'local_files_only': local_files_only,
        'token': token,
    }
    load_o2laf_weights = mode in ('ssl_action_mu', 'ssl_pre_gaussian', 'ssl_lora')
    if load_o2laf_weights and not model:
        raise ValueError(f'Mode {mode!r} requires --model (an O²LaF Hub repo id or local model directory).')

    lora_cfg: dict[str, Any] = {
        'lora_rank': int(lora_rank),
        'lora_alpha': int(lora_alpha),
        'target_module_name': list(DEFAULT_LORA_CONFIG['target_module_name']),
    }
    exclude_params = DEFAULT_EXCLUDE_PARAMS
    base_model_id = DEFAULT_BASE_MODEL
    fixed_prompt = DEFAULT_FIXED_PROMPT
    action_space_id = None
    encoder_state = None
    encoder_config = None
    resolved_model_dir = None

    if model:
        fetch = 'action_encoder' if load_o2laf_weights else 'metadata'
        resolved_model_dir = hub.resolve_model_dir(
            model,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            fetch=fetch,
        )
        index = hub.read_model_index(resolved_model_dir)
        encoder_at = dict(index['action_encoder'])
        encoder_config = hub.load_part_config(resolved_model_dir, encoder_at)
        if encoder_config.get('action_encoder') != 'wan_dit':
            raise ValueError(
                f'motion_annotation supports Wan DiT action encoders; '
                f'{model!r} uses {encoder_config.get("action_encoder")!r}.'
            )
        base_model_id = str(index['base_model'])
        fixed_prompt = str(index['fixed_prompt'])
        latent_dim = int(encoder_config['latent_dim'])
        query_init_std = float(encoder_config['query_init_std'])
        if encoder_config.get('lora_config') is not None:
            lora_cfg = dict(encoder_config['lora_config'])
        exclude_params = tuple(encoder_config.get('param_names_to_exclude', DEFAULT_EXCLUDE_PARAMS))
        action_space_id = str(index['action_space_id'])
        if load_o2laf_weights:
            encoder_state = hub.load_part_state(resolved_model_dir, encoder_at)

    base_model_path = str(base_model or base_model_id)
    encoder = WanDiTActionEncoder.from_pretrained(
        base_model_path,
        subfolder='transformer',
        torch_dtype=base_dtype,
        **base_kwargs,
    )
    encoder.setup_readout(latent_dim=int(latent_dim), query_init_std=float(query_init_std))
    encoder = _attach_wan_lora(encoder, lora_cfg)

    if encoder_state is not None:
        mapping = hub.hub_keys_to_model_keys(encoder_state, encoder.state_dict())
        source = {mapping[key]: tensor for key, tensor in encoder_state.items()}
        include = tuple((encoder_config or {}).get('param_names_to_optimize', ()))
        required = {
            name
            for name, _ in encoder.named_parameters()
            if any(k in name for k in include) and not any(k in name for k in exclude_params)
        }
        absent = required - set(source)
        if absent:
            raise RuntimeError(f'{len(absent)} required O²LaF encoder tensors are absent, e.g. {sorted(absent)[:3]}.')
        missing, unexpected = encoder.load_state_dict(source, strict=False)
        if unexpected:
            raise RuntimeError(f'{len(unexpected)} unexpected O²LaF encoder tensors, e.g. {unexpected[:3]}.')
        missing_required = required.intersection(missing)
        if missing_required:
            raise RuntimeError(f'{len(missing_required)} required O²LaF tensors did not load, e.g. {sorted(missing_required)[:3]}.')
        del encoder_state, source
        gc.collect()

    trainable_groups: dict[str, list[str]] = {}
    if mode in ('wan_base_lora', 'ssl_lora'):
        trainable_groups = configure_wan_trainability(encoder, exclude=exclude_params)
        if enable_gradient_checkpointing:
            unwrap_peft_model(encoder).enable_gradient_checkpointing()
        encoder = encoder.to(device=device)
    else:
        encoder = encoder.to(device=device, dtype=base_dtype)
        encoder.requires_grad_(False)
        encoder.eval()

    feature_kind = 'action_mu' if mode == 'ssl_action_mu' else 'transition_features'
    feature_dim = int(latent_dim) if mode == 'ssl_action_mu' else wan_transition_feature_dim(encoder)
    metadata = {
        'mode': mode,
        'family': 'wan2.2_ti2v_5b',
        'model': str(model) if model else None,
        'resolved_model_dir': str(resolved_model_dir) if resolved_model_dir else None,
        'action_space_id': action_space_id,
        'base_model_id': base_model_id,
        'base_model_path': base_model_path,
        'fixed_prompt': fixed_prompt,
        'base_dtype': str(base_dtype),
        'lora_config': lora_cfg,
        'latent_dim': int(latent_dim),
        'query_init_std': float(query_init_std),
        'feature_kind': feature_kind,
        'feature_dim': feature_dim,
        'trainable_groups': trainable_groups,
        'trainable_contract': trainable_parameter_contract(encoder),
    }
    return encoder, metadata


# -- Wan latent cache validation & clip provider --------------------------------


def valid_wan_latent_archive(
    path: Path,
    row: dict,
    expected_shape: tuple[int, int, int, int] | None = None,
) -> bool:
    """Validate a cached per-video Wan VAE latent archive."""
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        latents = payload['latents']
        lengths = payload['valid_transition_lengths']
        starts = payload['transition_starts']
        ends = payload['transition_ends']
        indices = payload['sampled_frame_indices']
        if latents.ndim != 5:
            return False
        if expected_shape is not None and tuple(latents.shape[1:]) != expected_shape:
            return False
        if latents.dtype != torch.bfloat16 or not torch.isfinite(latents).all():
            return False
        max_transitions = int(latents.shape[2]) - 1
        if lengths.shape != (latents.shape[0],) or torch.any((lengths < 1) | (lengths > max_transitions)):
            return False
        if len(starts) != len(ends) or len(starts) != latents.shape[0]:
            return False
        previous_end = None
        for length, window_starts, window_ends in zip(lengths, starts, ends):
            expected = int(length)
            if window_starts.shape != window_ends.shape or window_starts.shape != (expected,):
                return False
            if not torch.isfinite(window_starts).all() or not torch.isfinite(window_ends).all():
                return False
            if torch.any(window_ends <= window_starts):
                return False
            if expected > 1 and torch.any(window_starts[1:] < window_ends[:-1] - 1.0e-9):
                return False
            if previous_end is not None and float(window_starts[0]) < previous_end - 1.0e-9:
                return False
            previous_end = float(window_ends[-1])
        if indices.ndim != 2 or indices.shape[0] != latents.shape[0]:
            return False
        if str(payload['video_id']) != str(row['video_id']):
            return False
        source_num_frames = int(payload['source_num_frames'])
        if 'num_frames' in row and source_num_frames != int(row['num_frames']):
            return False
        if indices.numel() and (
            torch.any(indices < 0)
            or torch.any(indices >= source_num_frames)
            or torch.any(indices[:, 1:] < indices[:, :-1])
        ):
            return False
    except (OSError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    return True


class WanLatentProvider:
    """Streams cached Wan VAE latents through ``WanDiTActionEncoder`` with two-pass recompute."""

    def __init__(
        self,
        latent_cache: str | Path,
        model: torch.nn.Module,
        prompt_embeds: torch.Tensor,
        device: torch.device,
        microbatch: int = 2,
    ) -> None:
        self.root = Path(latent_cache)
        self.model = model
        self.prompt = prompt_embeds.to(device=device, dtype=torch.bfloat16)
        self.device = device
        self.microbatch = int(microbatch)

    def path(self, row: dict) -> Path:
        safe = str(row['video_id']).replace('/', '_')
        split = str(row.get('split', 'unsplit'))
        candidate = self.root / split / f'{safe}.pt'
        if candidate.is_file():
            return candidate
        # Allow looking up a cached clip even if split label changed between runs.
        for match in self.root.glob(f'*/{safe}.pt'):
            return match
        return candidate

    def load_record(self, row: dict) -> dict[str, Any]:
        archive = self.path(row)
        if not archive.is_file():
            raise FileNotFoundError(f'Missing cached Wan latent archive: {archive}. Run cache_latents.py first.')
        payload = torch.load(archive, map_location='cpu', weights_only=True, mmap=True)
        if str(payload['video_id']) != str(row['video_id']):
            raise ValueError(f'{archive}: video_id mismatch ({payload["video_id"]!r} != {row["video_id"]!r}).')
        return {
            'latents': payload['latents'],
            'lengths': [int(value) for value in payload['valid_transition_lengths']],
        }

    def encode_batch(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
        timestep = torch.zeros(latents.shape[0], latents.shape[2], device=self.device, dtype=torch.float32)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return wan_forward_features(
                self.model,
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=self.prompt.expand(latents.shape[0], -1, -1),
                return_action_mu=False,
            )

    @torch.no_grad()
    def detached_segments(self, record: dict[str, Any], *, requires_grad: bool = True) -> list[torch.Tensor]:
        result = []
        for start in range(0, len(record['latents']), self.microbatch):
            latents = record['latents'][start:start + self.microbatch]
            values = self.encode_batch(latents)
            for offset, feature in enumerate(values):
                length = record['lengths'][start + offset]
                segment = feature[:length].float().detach()
                if not torch.isfinite(segment).all():
                    raise RuntimeError('Wan detached transition features are non-finite.')
                result.append(segment.requires_grad_(True) if requires_grad else segment)
        return result

    def backward_segments(
        self,
        record: dict[str, Any],
        gradients: list[torch.Tensor],
        *,
        expected: list[torch.Tensor] | None = None,
        equivalence_atol: float = 1.0e-4,
    ) -> float:
        cursor = 0
        maximum_difference = 0.0
        for start in range(0, len(record['latents']), self.microbatch):
            latents = record['latents'][start:start + self.microbatch]
            values = self.encode_batch(latents)
            if not torch.isfinite(values).all():
                raise RuntimeError('Wan recomputed transition features are non-finite.')
            gradient = torch.zeros_like(values)
            for offset in range(len(values)):
                length = record['lengths'][start + offset]
                if expected is not None:
                    diff = float((values[offset, :length].float() - expected[cursor].detach().float()).abs().max())
                    maximum_difference = max(maximum_difference, diff)
                    if diff > float(equivalence_atol):
                        raise RuntimeError(
                            f'Wan recompute diverged from forward pass: max_abs_diff={diff} > {equivalence_atol}.'
                        )
                gradient[offset, :length] = gradients[cursor].to(gradient.dtype)
                if not torch.isfinite(gradient[offset, :length]).all():
                    raise RuntimeError('Wan transition-feature gradient is non-finite.')
                cursor += 1
            torch.autograd.backward(values, gradient)
        if cursor != len(gradients):
            raise RuntimeError('Wan segment-gradient count mismatch.')
        return maximum_difference

    @torch.no_grad()
    def features(self, row: dict) -> torch.Tensor:
        record = self.load_record(row)
        segments = self.detached_segments(record, requires_grad=False)
        return torch.cat(segments, dim=0)
