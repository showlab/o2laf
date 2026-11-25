"""Export training outputs to the released-model layout (:mod:`o2laf.inference.hub`).

::

    o2laf-export \\
      --latent-action-state  <run>/visualization/latent_action_eval/iter_10000/latent_action_state.pt \\
      --latent-action-config <run>/config.yaml \\
      --world-model-checkpoint <run2>/models/step_10000.pt \\
      --world-model-config     <run2>/config.yaml \\
      --output o2laf-dit-mixed

The latent-action state becomes one component, the latent action model (the
action encoder and the causal world model it was trained with); the
world-model checkpoint gives the bidirectional world model. Export needs no GPU:
the fixed prompt's embedding ships with the package, not with the model. Before anything is
written, the bidirectional world model's own record of the encoder it was
trained against -- the SHA256 of the encoder state and its geometry -- is
checked against the state being exported, so a model cannot be published with a
world model paired to a different action space.

Weights keep the dtype they were trained in. ``model_index.json`` is written
last, so an interrupted export never leaves a directory that loads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from o2laf.action_encoders.base import build_action_encoder_spec
from o2laf.inference import hub
from o2laf.inference.prompts import packaged_prompt_embeds
from o2laf.training.frozen_encoder import FROZEN_ENCODER_ID_BUFFER
from o2laf.training.latent_action import LatentActionTrainer
from o2laf.training.world_model import WorldModelTrainer
from o2laf.utils.action_util import check_latent_action_frame_budget
from o2laf.utils.file_util import sha256_file, write_json_atomic
from o2laf.utils.logger_util import get_logger
from o2laf.utils.lora_util import validate_lora_config

DEFAULT_BASE_MODEL = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
# Each mode's null action is whatever its trainer's dropout taught, read from
# the trainer rather than restated, so the two cannot drift apart.
NULL_ACTION_BY_MODE = {
    'causal': LatentActionTrainer.null_action,
    'bidirectional': WorldModelTrainer.null_action,
}
ACTION_LIBRARY_SCHEMA = 'o2laf_atomic_action_library_v1'
# Library arrays that describe the actions; ``names`` are strings and go to JSON.
_LIBRARY_ARRAYS = (
    'actions', 'actions_video_balanced', 'actions_transition_mean', 'transition_std',
    'video_mean_std', 'bootstrap_std', 'support', 'video_count', 'gt_labels',
)


def read_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict) or 'models' not in config:
        raise ValueError(f'{path} is not a training config.')
    return config


# -- geometry and identity, as the training configs record them ------------------


def resolve_geometry(config) -> tuple[int, int, int]:
    """Inference resolution and frame rate, read from a training config."""
    dataset = config['datasets']['train']
    if 'preprocess_video_cfg' in dataset:
        preprocess = dataset['preprocess_video_cfg']
        return int(preprocess['height']), int(preprocess['width']), int(preprocess['fps'])
    if all(key in dataset for key in ('input_h', 'input_w', 'fps')):
        return int(dataset['input_h']), int(dataset['input_w']), int(dataset['fps'])
    if dataset.get('type') == 'WeightedInterleaveDataset':
        evaluation = config['val']['latent_action_eval']
        return int(evaluation['height']), int(evaluation['width']), int(evaluation['fps'])
    raise ValueError('Could not resolve inference geometry from the config.')


def recorded_encoder_sha256(source: dict[str, torch.Tensor]) -> str:
    """The encoder-state SHA256 a world-model checkpoint carries in ``fixed_idm_sha256_bytes``."""
    if 'fixed_idm_sha256_bytes' not in source:
        raise KeyError('The world model state has no fixed_idm_sha256_bytes buffer.')
    return bytes(int(item) for item in source['fixed_idm_sha256_bytes'].detach().cpu().flatten()).hex()


def encoder_fps_values(fixed_idm_cfg) -> list[int]:
    """The frame rates a frozen encoder was trained at.

    A run at several rates lists ``fps_values``; a single-rate run names ``fps``.
    """
    values = fixed_idm_cfg.get('fps_values') or [fixed_idm_cfg.get('fps', 16)]
    return [int(value) for value in values]


# The encoder family each world-model checkpoint records, as numbered by the
# research trainers that wrote the buffer.
ENCODER_FAMILIES = {'wan_dit': 1, 'vjepa2': 2, 'st_transformer': 3}


def validate_recorded_geometry(source, *, encoder_step, action_cfg, fps_values, action_dim,
                               backend: str = 'wan_dit') -> None:
    """Check the world model's ``fixed_idm_contract`` buffer against the encoder being exported.

    Two layouts exist: the original Wan DiT one, and the one with an explicit
    encoder-family field. The family, the codes each transition emits and the
    encoder's window all identify which encoder the world model was trained
    against, so a mismatched pairing cannot be exported.
    """
    value = source.get('fixed_idm_contract')
    if value is None:
        raise KeyError('The world model state has no fixed_idm_contract buffer.')
    actual = tuple(int(item) for item in value.detach().cpu().flatten())
    spec = build_action_encoder_spec(backend)
    family = ENCODER_FAMILIES[backend]
    codes = int(spec.codes_per_transition)
    window = int(spec.window_rgb_frames or 0)
    rgb_frames = int(action_cfg['num_rgb_frames'])
    temporal_ratio = int(action_cfg.get('temporal_compression_ratio', 4))
    latent_frames = check_latent_action_frame_budget(rgb_frames, temporal_ratio)
    fps_sentinel = fps_values[0] if len(fps_values) == 1 else 0
    fps_suffix = tuple(fps_values) if len(fps_values) > 1 else ()
    # (version, family, step, source rgb, source latent, rgb, latent, ratio,
    #  dim, fps, actions per latent step, encoder window)
    with_family = (1, family, int(encoder_step), 41, 11, rgb_frames, latent_frames, temporal_ratio,
                   int(action_dim), fps_sentinel, codes, window)
    allowed = {with_family + fps_suffix}
    if backend == 'wan_dit':
        # Runs that predate the family field; only Wan DiT has any.
        # (version, step, source rgb, source latent, rgb, latent, ratio, dim,
        #  fps, causal encoder, pairwise readout, readout reads history)
        allowed.add((1, int(encoder_step), 41, 11, rgb_frames, latent_frames, temporal_ratio,
                     int(action_dim), fps_sentinel, 1, 1, 0) + fps_suffix)
    if actual not in allowed:
        raise RuntimeError(
            f'World model encoder geometry mismatch: checkpoint {actual}, expected '
            f'{" or ".join(str(item) for item in sorted(allowed))}.'
        )


# -- components -----------------------------------------------------------------


def _trained_parameters(model_cfg) -> dict[str, Any]:
    return {
        'lora_config': model_cfg.get('lora_config'),
        'param_names_to_optimize': list(model_cfg.get('param_names_to_optimize', ())),
        'param_names_to_exclude': list(model_cfg.get('param_names_to_exclude', ())),
    }


def load_latent_action_state(path) -> dict[str, Any]:
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(state, dict):
        raise TypeError('The latent-action state must be a mapping.')
    for key in ('encoder', 'decoder'):
        if not isinstance(state.get(key), dict) or not state[key]:
            raise KeyError(f'The latent-action state has no non-empty {key!r} half.')
    return state


def _wan_dit_component(idm_cfg, action_cfg) -> dict[str, Any]:
    """Reads Wan VAE latents with the Wan DiT trunk, which comes from the base model."""
    del idm_cfg
    return {
        '_class_name': 'WanDiTActionEncoder',
        'base_model_subfolder': 'transformer',
        'query_init_std': float(action_cfg.get('query_init_std', 0.02)),
        'codes_per_transition': 1,
    }


def _vjepa2_component(idm_cfg, action_cfg) -> dict[str, Any]:
    """Reads pixels with a frozen V-JEPA 2 ViT-g trunk the user supplies separately."""
    ratio = int(action_cfg.get('vjepa_action_compression_ratio', 2))
    if ratio != 2:
        raise NotImplementedError(f'The V-JEPA 2 action encoder emits 2 codes per transition, not {ratio}.')
    component = {
        '_class_name': 'VJEPA2ActionEncoder',
        'checkpoint_key': str(idm_cfg.get('checkpoint_key', 'target_encoder')),
        'architecture': str(idm_cfg.get('architecture', 'vit_giant_xformers')),
        'image_size': int(idm_cfg.get('image_size', 256)),
        'pretrained_num_frames': int(idm_cfg.get('pretrained_num_frames', 64)),
        'patch_size': int(idm_cfg.get('patch_size', 16)),
        'tubelet_size': int(idm_cfg.get('tubelet_size', 2)),
        'prepend_first_frame': bool(idm_cfg.get('prepend_first_frame', True)),
        'checkpoint_resolution': int(idm_cfg.get('checkpoint_resolution', 384)),
        # The trunk is pretrained at one resolution and run at another; without
        # this the encoder refuses to rebuild.
        'allow_resolution_override': bool(idm_cfg.get('allow_resolution_override', False)),
        'query_init_std': float(idm_cfg.get('query_init_std', 0.02)),
        'codes_per_transition': ratio,
        # Which trunk this encoder was trained on (supplied separately by the user).
        'base_action_encoder_name': Path(str(idm_cfg['model_name'])).name,
    }
    return component


def _st_transformer_component(idm_cfg, action_cfg) -> dict[str, Any]:
    """Trained from scratch, so the release carries every weight it needs."""
    ratio = int(action_cfg.get('st_transformer_action_compression_ratio',
                               action_cfg.get('scratch_lam_action_compression_ratio', 4)))
    if ratio != 4:
        raise NotImplementedError(f'The ST-Transformer action encoder emits 4 codes per transition, not {ratio}.')
    return {
        '_class_name': 'STTransformerActionEncoder',
        'model_dim': int(idm_cfg.get('model_dim', 1024)),
        'enc_blocks': int(idm_cfg.get('enc_blocks', 16)),
        'num_heads': int(idm_cfg.get('num_heads', 16)),
        'patch_size': int(idm_cfg.get('patch_size', 16)),
        'height': int(idm_cfg.get('height', 272)),
        'width': int(idm_cfg.get('width', 480)),
        'codes_per_transition': ratio,
    }


ENCODER_COMPONENTS = {
    'wan_dit': _wan_dit_component,
    'vjepa2': _vjepa2_component,
    'st_transformer': _st_transformer_component,
}


# The research configs name the encoder by the class they build rather than by
# a backend key. Reading `backend` alone would silently default every one of
# them to Wan DiT and mislabel the release.
RESEARCH_MODEL_TYPES = {
    'WanLatentActionTransformer3DModel': 'wan_dit',
    'VJEPALatentActionModel': 'vjepa2',
    'VJEPA21LatentActionModel': 'vjepa2',
    'ScratchLAMLatentActionModel': 'st_transformer',
}


def resolve_encoder_backend(idm_cfg) -> str:
    """Which action encoder a training config trains, from either naming."""
    backend = idm_cfg.get('backend')
    if backend is None:
        model_type = idm_cfg.get('model_type')
        if model_type is None:
            raise ValueError('The action encoder config names neither a backend nor a model_type.')
        if model_type not in RESEARCH_MODEL_TYPES:
            raise NotImplementedError(
                f'Unknown action encoder model_type {model_type!r}; expected one of '
                f'{", ".join(sorted(RESEARCH_MODEL_TYPES))}.'
            )
        backend = RESEARCH_MODEL_TYPES[model_type]
    backend = 'wan_dit' if backend == 'dit' else str(backend)
    if backend not in ENCODER_COMPONENTS:
        raise NotImplementedError(
            f'Released inference supports {", ".join(sorted(ENCODER_COMPONENTS))}, not {backend!r}.'
        )
    return backend


def action_encoder_component(state, config) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    models = config['models']
    idm_cfg, action_cfg = models['idm_cfg'], models['action_cfg']
    backend = resolve_encoder_backend(idm_cfg)
    validate_lora_config(idm_cfg.get('lora_config'), state.get('encoder_lora_config'), 'action encoder')
    latent_dim = int(action_cfg.get('latent_action_dim', 32))
    if int(state.get('latent_action_dim', -1)) != latent_dim:
        raise RuntimeError('The latent-action state latent_action_dim does not match its config.')
    tensors = {hub.to_hub_key(name): tensor for name, tensor in state['encoder'].items()}
    component = {
        'action_encoder': backend,
        'latent_dim': latent_dim,
        'num_rgb_frames': int(action_cfg['num_rgb_frames']),
        'global_step': int(state['global_step']),
        **ENCODER_COMPONENTS[backend](idm_cfg, action_cfg),
        **_trained_parameters(idm_cfg),
    }
    # What identifies the encoder's action space to a later run that loads the
    # release (world model training, continued training); absent from states
    # written by the research code.
    for key in ('idm_backend', 'idm_contract'):
        if state.get(key) is not None:
            component[key] = state[key]
    return component, tensors


def config_trains_causal_world_model(config) -> bool:
    """Whether a training config's world model is causal.

    The trainer decides it (``world_model_causal``). Configs written for the
    research code name a trainer this package does not have, and state it as
    ``action_cfg.decoder_causal`` instead, which must then be present.
    """
    from o2laf.training import build_trainer

    try:
        return bool(build_trainer(config['train']['trainer']).world_model_causal)
    except KeyError:
        action_cfg = config['models']['action_cfg']
        if 'decoder_causal' not in action_cfg:
            raise ValueError(
                f'Cannot tell whether {config["train"]["trainer"]!r} trains a causal world model: it is not a '
                'trainer of this package, and the config does not set action_cfg.decoder_causal.'
            ) from None
        return bool(action_cfg['decoder_causal'])


# Where a checkpoint written by TrainingRunner keeps each kind of weights; the
# second name is the research code's, for checkpoints it wrote.
CHECKPOINT_WEIGHT_KEYS = {'ema': ('ema',), 'model': ('model', 'model_state_dict_g')}


def world_model_component(config, *, causal: bool, global_step: int) -> dict[str, Any]:
    models = config['models']
    transformer_cfg, action_cfg = models['transformer_cfg'], models['action_cfg']
    if config_trains_causal_world_model(config) != causal:
        expected = 'causal' if causal else 'bidirectional'
        actual = 'bidirectional' if causal else 'causal'
        raise ValueError(f'The config trains a {actual} world model, not the {expected} one expected here.')
    num_rgb_frames = int(action_cfg['num_rgb_frames'])
    return {
        '_class_name': 'WanWorldModel',
        'base_model_subfolder': 'transformer',
        'causal': causal,
        'num_rgb_frames': num_rgb_frames,
        'num_latent_frames': check_latent_action_frame_budget(num_rgb_frames, int(action_cfg.get('temporal_compression_ratio', 4))),
        'action_dim': int(action_cfg.get('latent_action_dim', 32)),
        'action_gain_init': float(action_cfg.get('gamma_init', 0.5)),
        'global_step': int(global_step),
        **_trained_parameters(transformer_cfg),
    }


def causal_world_model_component(state, config) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    transformer_cfg = config['models']['transformer_cfg']
    validate_lora_config(transformer_cfg.get('lora_config'), state.get('decoder_lora_config'), 'causal world model')
    component = world_model_component(config, causal=True, global_step=int(state['global_step']))
    tensors = {hub.to_hub_key(name): tensor for name, tensor in state['decoder'].items()}
    return component, tensors


def bidirectional_world_model_component(checkpoint_path, config, *, action_space_id: str, encoder_state_sha256: str,
                                        encoder_step: int, weights: str = 'ema') -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """The fine-tuned world model, after checking it was trained against this encoder.

    A world model trained by this package records the encoder's
    ``action_space_id``. The released one predates that and records the SHA256
    of the encoder's ``.pt`` file plus a geometry tuple instead; it is checked
    in that form, against the state file being exported.
    """
    models = config['models']
    if models['transformer_cfg'].get('lora_config') is not None:
        raise ValueError('The bidirectional world model is fully fine-tuned and has no LoRA.')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    keys = CHECKPOINT_WEIGHT_KEYS[weights]
    key = next((key for key in keys if isinstance(payload.get(key), dict) and payload[key]), None)
    if key is None:
        raise KeyError(f'{checkpoint_path} has no non-empty {weights!r} weights (looked for {list(keys)}).')
    source = payload[key]
    source = dict(source)

    if FROZEN_ENCODER_ID_BUFFER in source:
        recorded = 'sha256:' + bytes(int(item) for item in source.pop(FROZEN_ENCODER_ID_BUFFER).flatten()).hex()
        if recorded != action_space_id:
            raise RuntimeError(
                f'The world model was trained against action space {recorded}, but the encoder being exported is '
                f'{action_space_id}.'
            )
    else:
        validate_legacy_binding(source, config, encoder_state_sha256=encoder_state_sha256, encoder_step=encoder_step)

    step = int(payload.get('global_step', -1))
    if step < 0:
        raise ValueError(f'{checkpoint_path} records no valid global_step.')
    component = world_model_component(config, causal=False, global_step=step)
    component['weights'] = weights
    return component, source


# Stage-1 states that were re-saved after their world models were trained:
# serialisation metadata changed the file hash while the tensors stayed identical.
EQUIVALENT_ENCODER_STATES = {
    '1b05e80fa7695524dc49ba25c28c4fb6a022a9054076e16819bf2578f56fa2c7':
        '17ab1e77b101fe95cdae0570d34eb40a2938117b92e353c0b21ed0f151155700',
}


def validate_legacy_binding(source, config, *, encoder_state_sha256: str, encoder_step: int) -> None:
    """The binding the released world model carries: the encoder file's SHA256 and a geometry tuple."""
    fixed_idm_cfg, action_cfg = config['models']['fixed_idm_cfg'], config['models']['action_cfg']
    recorded = recorded_encoder_sha256(source)
    configured = str(fixed_idm_cfg['state_sha256']).lower()
    equivalent = EQUIVALENT_ENCODER_STATES.get(recorded)
    if recorded == configured and equivalent == encoder_state_sha256:
        get_logger().info(
            'Encoder state %s was re-saved as %s; the tensors are the same, so the binding holds.',
            recorded, encoder_state_sha256,
        )
    elif not recorded == configured == encoder_state_sha256:
        raise RuntimeError(
            f'The world model was trained against encoder state {recorded} (config says {configured}), '
            f'but the state being exported is {encoder_state_sha256}.'
        )
    if int(fixed_idm_cfg['source_global_step']) != int(encoder_step):
        raise RuntimeError(f'The world model config names encoder step {fixed_idm_cfg["source_global_step"]}, not {encoder_step}.')
    validate_recorded_geometry(
        source,
        encoder_step=encoder_step,
        action_cfg=action_cfg,
        fps_values=encoder_fps_values(fixed_idm_cfg),
        action_dim=int(action_cfg.get('latent_action_dim', 32)),
        backend=resolve_encoder_backend(config['models']['idm_cfg']),
    )
    # Training-time bookkeeping; the release states the binding through action_space_id.
    source.pop('fixed_idm_sha256_bytes')
    source.pop('fixed_idm_contract')


def mode_entry(config, *, world_model: str, null_action: str) -> dict[str, Any]:
    action_cfg = config['models']['action_cfg']
    height, width, fps = resolve_geometry(config)
    return {
        'world_model': world_model,
        'null_action': null_action,
        'num_rgb_frames': int(action_cfg['num_rgb_frames']),
        'num_context_frames': int(action_cfg.get('num_context_frames', 1)),
        'height': height,
        'width': width,
        'fps': fps,
        'scheduler_cfg': dict(config['models'].get('scheduler_cfg') or {}),
    }


# -- action library -------------------------------------------------------------


def sanitize_action_library(metadata_path, *, encoder_state_sha256: str, action_space_id: str,
                            codes_per_transition: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """A library's arrays and a metadata record with no internal paths or per-video sources.

    The library must have been extracted with the encoder state being exported;
    it is then rebound to the released model's action space.
    """
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    source_checkpoint = metadata.get('source_checkpoint') or {}
    if str(source_checkpoint.get('sha256', '')).lower() != encoder_state_sha256:
        raise RuntimeError(
            f'{metadata_path} was extracted with encoder state {source_checkpoint.get("sha256")}, '
            f'not {encoder_state_sha256}.'
        )
    with np.load(metadata_path.with_name(metadata['npz']), allow_pickle=False) as arrays:
        names = [str(name) for name in arrays['names']]
        tensors = {key: torch.from_numpy(np.ascontiguousarray(arrays[key])) for key in _LIBRARY_ARRAYS if key in arrays.files}
    if [entry['name'] for entry in metadata['classes']] != names:
        raise ValueError(f'{metadata_path} classes disagree with the stored names {names}.')
    actions = tensors['actions']
    if actions.ndim != 2 or actions.shape[0] != len(names):
        raise ValueError(f'Library actions must be [num_actions, action_dim], got {tuple(actions.shape)}.')

    source = metadata.get('source') or {}
    record = {
        'schema_version': ACTION_LIBRARY_SCHEMA,
        'action_space_id': action_space_id,
        'perspective': metadata.get('perspective'),
        'names': names,
        'action_dim': int(actions.shape[1]),
        'codes_per_transition': int(codes_per_transition),
        'feature': source_checkpoint.get('feature'),
        'num_rgb_frames': source_checkpoint.get('evaluation_num_rgb_frames'),
        'classes': metadata['classes'],
        'selection': metadata.get('selection'),
        'aggregation': metadata.get('aggregation'),
        'bootstrap': metadata.get('bootstrap'),
        # Counts only: the per-video records name internal archives.
        'source': {key: source[key] for key in ('splits', 'videos_per_split', 'video_count', 'transition_count') if key in source},
    }
    return record, tensors


def save_action_library(directory: Path, stem: str, record, tensors) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weights_name = f'{stem}.safetensors'
    hub.save_tensors(tensors, directory / weights_name)
    write_json_atomic(directory / f'{stem}.json', {**record, 'weights': weights_name})


# -- the whole model ------------------------------------------------------------


# Hand-written files in an output directory that a re-export keeps.
KEPT_ON_OVERWRITE = ('README.md',)


def export(*, latent_action_state, latent_action_config, output, world_model_checkpoint=None, world_model_config=None,
           world_model_weights='ema', base_model=DEFAULT_BASE_MODEL, action_libraries=(), overwrite=False,
           max_shard_size='5GB') -> Path:
    """Read and check every input, then write the release into place in one step.

    Nothing is written until every check has passed, and the release is built in
    a temporary sibling directory that replaces ``output`` only once complete:
    a failed export leaves an existing ``output`` exactly as it was. A
    hand-written model card (``README.md``) in ``output`` is carried over.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f'{output} is not empty; pass overwrite=True to replace it.')
    if (world_model_checkpoint is None) != (world_model_config is None):
        raise ValueError('Pass the world-model checkpoint and its config together.')

    # -- read and check everything ---------------------------------------------
    la_config = read_config(latent_action_config)
    print(f'hashing {latent_action_state} ...', flush=True)
    state_sha256 = sha256_file(latent_action_state)
    state = load_latent_action_state(latent_action_state)

    # The latent action model: both halves of the state, as they were trained.
    encoder_config, encoder_tensors = action_encoder_component(state, la_config)
    action_space_id = hub.action_space_id(encoder_tensors)
    causal_config, causal_tensors = causal_world_model_component(state, la_config)
    encoder_config['action_space_id'] = causal_config['action_space_id'] = action_space_id
    latent_action_model = {
        '_class_name': 'LatentActionModel',
        'global_step': int(state['global_step']),
        hub.ACTION_ENCODER_PART: encoder_config,
        hub.WORLD_MODEL_PART: causal_config,
    }
    latent_action_tensors = {f'{hub.ACTION_ENCODER_PART}.{name}': tensor for name, tensor in encoder_tensors.items()}
    latent_action_tensors.update({f'{hub.WORLD_MODEL_PART}.{name}': tensor for name, tensor in causal_tensors.items()})

    joint = {'component': hub.LATENT_ACTION_MODEL_DIR}
    modes = {'causal': mode_entry(la_config, world_model={**joint, 'part': hub.WORLD_MODEL_PART},
                                  null_action=NULL_ACTION_BY_MODE['causal'])}
    prompts = {str(la_config['models']['action_cfg'].get('fixed_prompt', 'a video'))}
    fps_values = None
    world_config = world_tensors = None
    if world_model_checkpoint is not None:
        wm_config = read_config(world_model_config)
        world_config, world_tensors = bidirectional_world_model_component(
            world_model_checkpoint, wm_config, action_space_id=action_space_id, encoder_state_sha256=state_sha256,
            encoder_step=int(state['global_step']), weights=world_model_weights,
        )
        world_config['action_space_id'] = action_space_id
        modes['bidirectional'] = mode_entry(wm_config, world_model={'component': hub.WORLD_MODEL_DIR, 'part': None},
                                            null_action=NULL_ACTION_BY_MODE['bidirectional'])
        prompts.add(str(wm_config['models']['action_cfg'].get('fixed_prompt', 'a video')))
        fixed_idm_cfg = wm_config['models']['fixed_idm_cfg']
        fps_values = encoder_fps_values(fixed_idm_cfg)
    if len(prompts) != 1:
        raise ValueError(f'The two parts were trained with different fixed prompts: {sorted(prompts)}.')
    prompt = prompts.pop()
    if packaged_prompt_embeds(base_model, prompt) is None:
        print(f'note: no packaged embedding of {prompt!r} for {base_model}; inference will load the text encoder.')

    libraries = []
    for library in action_libraries:
        record, library_tensors = sanitize_action_library(
            library, encoder_state_sha256=state_sha256, action_space_id=action_space_id,
            codes_per_transition=int(encoder_config['codes_per_transition']),
        )
        libraries.append((Path(library).stem, record, library_tensors))

    index = {
        '_class_name': 'LatentActionPipeline',
        'format': hub.MODEL_FORMAT,
        'base_model': base_model,
        'fixed_prompt': prompt,
        'action_space_id': action_space_id,
        'action_encoder': {**joint, 'part': hub.ACTION_ENCODER_PART},
        'action_dim': int(encoder_config['latent_dim']),
        'codes_per_transition': int(encoder_config['codes_per_transition']),
        'temporal_compression_ratio': int(la_config['models']['action_cfg'].get('temporal_compression_ratio', 4)),
        'modes': modes,
    }
    if fps_values is not None:
        index['fps_values'] = fps_values
    if libraries:
        index['action_library'] = {'directory': 'action_library', 'libraries': [stem for stem, _, _ in libraries]}

    # -- write, then swap into place -------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f'.{output.name}.export-{os.getpid()}')
    shutil.rmtree(staging, ignore_errors=True)
    try:
        hub.save_component(staging / hub.LATENT_ACTION_MODEL_DIR, latent_action_model, latent_action_tensors,
                           max_shard_size=max_shard_size)
        if world_config is not None:
            hub.save_component(staging / hub.WORLD_MODEL_DIR, world_config, world_tensors, max_shard_size=max_shard_size)
        for stem, record, library_tensors in libraries:
            save_action_library(staging / 'action_library', stem, record, library_tensors)
        for name in KEPT_ON_OVERWRITE:
            if (output / name).is_file():
                shutil.copy2(output / name, staging / name)
        # Last: until this exists, the directory is not a model.
        write_json_atomic(staging / hub.MODEL_INDEX_NAME, index)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    retired = output.with_name(f'.{output.name}.replaced-{os.getpid()}')
    if output.exists():
        os.replace(output, retired)
    os.replace(staging, output)
    shutil.rmtree(retired, ignore_errors=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--latent-action-state', type=Path, required=True)
    parser.add_argument('--latent-action-config', type=Path, required=True)
    parser.add_argument('--world-model-checkpoint', type=Path)
    parser.add_argument('--world-model-config', type=Path)
    parser.add_argument('--world-model-weights', choices=('ema', 'model'), default='ema')
    parser.add_argument('--base-model', default=DEFAULT_BASE_MODEL, help='Hub id recorded in the model index.')
    parser.add_argument('--action-library', type=Path, action='append', default=[], help='Library JSON; repeatable.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = export(
        latent_action_state=args.latent_action_state,
        latent_action_config=args.latent_action_config,
        world_model_checkpoint=args.world_model_checkpoint,
        world_model_config=args.world_model_config,
        world_model_weights=args.world_model_weights,
        base_model=args.base_model,
        action_libraries=args.action_library,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
