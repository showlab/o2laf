"""The frozen action encoder a world model is trained against.

World model training extracts every action with one frozen encoder, so the
world model only understands that encoder's action space: paired with another
encoder -- even one of the same architecture from a different run -- it would
generate garbage without any error. The pairing is therefore recorded in the
world model itself.

The encoder is identified by its ``action_space_id``, a hash of its weights
(:func:`o2laf.inference.hub.action_space_id`). It is computed from the source,
never typed into a config, and it is the same id a released model records, so a
world model can be trained on a ``latent_action_state.pt`` from a latent-action
run or on a released model directory alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from o2laf.inference import hub

# The world-model buffer holding the encoder's action_space_id (32 hash bytes).
FROZEN_ENCODER_ID_BUFFER = 'frozen_action_encoder_id'


@dataclass
class FrozenEncoderSource:
    """One encoder state, read from either source format.

    ``encoder`` keeps the names it was stored under; map them onto a module with
    :func:`o2laf.inference.hub.hub_keys_to_model_keys` after :func:`~o2laf.inference.hub.to_hub_key`.
    """

    path: str
    encoder: dict[str, torch.Tensor] = field(repr=False)
    metadata: dict[str, Any]

    @property
    def action_space_id(self) -> str:
        return self.metadata['action_space_id']


def is_released_model(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and (
        (path / hub.MODEL_INDEX_NAME).is_file() or (path / hub.COMPONENT_CONFIG_NAME).is_file()
    )


def latent_action_model_dir(path: str | Path) -> Path:
    """The ``latent_action_model/`` component, given it or the model directory containing it."""
    path = Path(path)
    return path / hub.LATENT_ACTION_MODEL_DIR if (path / hub.MODEL_INDEX_NAME).is_file() else path


def read_frozen_encoder_source(path: str | Path) -> FrozenEncoderSource:
    """The encoder half of a latent-action state, from a ``.pt`` or a released model."""
    if is_released_model(path):
        component = latent_action_model_dir(path)
        config = hub.load_component_config(component)
        if config.get('_class_name') != 'LatentActionModel':
            raise ValueError(f'{component} is not a released latent action model.')
        encoder_config = config[hub.ACTION_ENCODER_PART]
        world_model_config = config[hub.WORLD_MODEL_PART]
        encoder = hub.load_component_state(component, hub.ACTION_ENCODER_PART)
        metadata = {
            'global_step': int(config['global_step']),
            'latent_dim': int(encoder_config['latent_dim']),
            'num_rgb_frames': int(encoder_config['num_rgb_frames']),
            'num_latent_frames': int(world_model_config['num_latent_frames']),
            'encoder_name': encoder_config.get('action_encoder'),
            'encoder_lora_config': encoder_config.get('lora_config'),
            'contract_state': {key: encoder_config[key] for key in ('idm_backend', 'idm_contract') if key in encoder_config},
        }
    else:
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if not isinstance(state, dict):
            raise TypeError(f'Expected a state mapping in {path}, got {type(state).__name__}.')
        encoder = state.get('encoder')
        if not isinstance(encoder, dict) or not encoder:
            raise RuntimeError(f'{path} contains no non-empty encoder state.')
        missing = [key for key in ('global_step', 'latent_action_dim', 'num_rgb_frames', 'num_latent_frames') if key not in state]
        if missing:
            raise KeyError(f'{path} is missing {missing}.')
        metadata = {
            'global_step': int(state['global_step']),
            'latent_dim': int(state['latent_action_dim']),
            'num_rgb_frames': int(state['num_rgb_frames']),
            'num_latent_frames': int(state['num_latent_frames']),
            'encoder_name': state.get('idm_backend'),
            'encoder_lora_config': state.get('encoder_lora_config'),
            # What an encoder spec checks to refuse a state written by another encoder.
            'contract_state': {key: state[key] for key in state if key == 'idm_backend' or 'contract' in key},
        }
    metadata['action_space_id'] = hub.action_space_id(encoder)
    return FrozenEncoderSource(path=str(path), encoder=dict(encoder), metadata=metadata)


def read_world_model_half(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    """The causal world model trained with the encoder, and its LoRA config, from either source format."""
    if is_released_model(path):
        component = latent_action_model_dir(path)
        config = hub.load_component_config(component)
        return hub.load_component_state(component, hub.WORLD_MODEL_PART), config[hub.WORLD_MODEL_PART].get('lora_config')
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    decoder = state.get('decoder')
    if not isinstance(decoder, dict) or not decoder:
        raise ValueError(f'{path} has no non-empty world model state.')
    return dict(decoder), state.get('decoder_lora_config')


def id_bytes(action_space_id: str) -> torch.Tensor:
    """The buffer form of an ``action_space_id``."""
    digest = action_space_id.removeprefix('sha256:')
    if len(digest) != 64:
        raise ValueError(f'Not an action_space_id: {action_space_id!r}.')
    return torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8)


def recorded_action_space_id(world_model) -> str:
    """The ``action_space_id`` a world model was trained against, from its buffer."""
    value = getattr(world_model, FROZEN_ENCODER_ID_BUFFER, None)
    if value is None:
        raise RuntimeError(f'The world model has no {FROZEN_ENCODER_ID_BUFFER} buffer.')
    if hasattr(value, 'full_tensor'):
        value = value.full_tensor()
    return 'sha256:' + bytes(int(item) for item in value.detach().cpu().flatten()).hex()


def assert_frozen_encoder(world_model, expected: str) -> None:
    """Ensure the world model still names the encoder its actions come from."""
    actual = recorded_action_space_id(world_model)
    if actual != expected:
        raise RuntimeError(f'The world model was trained against action space {actual}, but its encoder is {expected}.')
