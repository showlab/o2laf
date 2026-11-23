"""The contract between the trainers and one action encoder.

Every action encoder reads the same thing -- the RGB video the dataset produces
-- and differs only in how it turns that video into tokens its trunk
understands. The Wan DiT encoder VAE-encodes it to latents; V-JEPA2 crops it
square and resizes; the ST-Transformer resizes. The VAE is a preprocessing step
like the others, not a different kind of input, so there is one ``encode`` and
no branching on what an encoder "reads".

A spec also declares the rest of what a trainer needs to know about its encoder:
how many codes it emits per latent transition, how to build it, and what
identifies the action space it defines in a checkpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

_ACTION_ENCODER_SPECS = {}

# Optimizer parameter-group names, shared by the trainer that creates the groups
# and the encoders that re-rate them. They are defined once because the two
# sides are in different modules: when these were string literals on both sides,
# renaming the trainer's groups silently disabled an encoder's learning rate.
GROUP_WORLD_MODEL_LORA = 'world_model_lora'
GROUP_ACTION_EMBEDDER = 'action_embedder'
GROUP_ACTION_ENCODER_LORA = 'action_encoder_lora'
GROUP_READOUT = 'readout'
GROUP_WORLD_MODEL_OTHER = 'world_model_other'
GROUP_ACTION_ENCODER_OTHER = 'action_encoder_other'

PARAM_GROUP_NAMES = (
    GROUP_WORLD_MODEL_LORA,
    GROUP_ACTION_EMBEDDER,
    GROUP_ACTION_ENCODER_LORA,
    GROUP_READOUT,
    GROUP_WORLD_MODEL_OTHER,
    GROUP_ACTION_ENCODER_OTHER,
)


def unwrap_peft(module):
    """Return the inner module under an optional PEFT wrapper."""
    return module.get_base_model() if hasattr(module, 'get_base_model') else module


def register_action_encoder_spec(name):
    """Register a spec class under the name used by ``idm_cfg.backend``."""

    def decorator(spec_class):
        if name in _ACTION_ENCODER_SPECS:
            raise KeyError(f'Action encoder spec {name!r} is already registered.')
        spec_class.name = name
        _ACTION_ENCODER_SPECS[name] = spec_class
        return spec_class

    return decorator


def build_action_encoder_spec(name):
    """Construct a registered action encoder spec."""
    if name not in _ACTION_ENCODER_SPECS:
        available = ', '.join(sorted(_ACTION_ENCODER_SPECS)) or '<empty>'
        raise KeyError(f'Unknown action encoder {name!r}. Available: {available}.')
    return _ACTION_ENCODER_SPECS[name]()


class ActionEncoderSpec(ABC):
    """Build, run and describe one action encoder."""

    name = None
    # Codes emitted per latent transition. Fixed by the architecture, not a
    # free choice: a config may state it, but only to agree with it.
    codes_per_transition = None
    action_ratio_cfg_key = None
    legacy_contract_state_keys = ()
    # Whether ``encode`` reads the clip's pixels (True) or the VAE latents the
    # caller already holds (False). Callers decode a clip only when it is True.
    reads_pixels = True
    # RGB frames one encoder pass consumes, for encoders that run in windows
    # over a longer clip. ``None`` means the encoder takes the whole clip.
    window_rgb_frames = 41
    # ``action_cfg`` keys that describe the Wan DiT in-sequence readout, which
    # only that encoder has. Accepting them elsewhere would silently describe an
    # encoder the run does not train.
    unsupported_action_cfg_keys = ('query_init_std',)

    def prepare_trainer(self, trainer, action_cfg):
        """Set encoder-specific trainer attributes before common construction."""

    @abstractmethod
    def build_contract(self, trainer, idm_cfg):
        """Return metadata that identifies the action space this encoder defines."""

    @abstractmethod
    def build_encoder(self, trainer, idm_cfg, dtype):
        """Construct, adapt, select trainable parameters, and FSDP-wrap the encoder."""

    @abstractmethod
    def encode(self, encoder, *, latents=None, pixel_values=None, prompt_embeds=None) -> dict:
        """Return the action posterior for one clip, reading whichever form this encoder reads.

        Callers pass every form they already hold; an encoder takes the one it
        needs and ignores the rest. The latents are the ones the world model
        will condition, so the VAE is never run a second time here.
        """

    def validate_state_contract(self, state, expected, label):
        """Refuse an encoder state whose action space was defined by a different encoder."""
        name = state.get('idm_backend')
        if name is not None and name != self.name:
            raise RuntimeError(f'{label} was written by the {name!r} action encoder, not {self.name!r}.')
        actual = state.get('idm_contract')
        if actual is None:
            for key in self.legacy_contract_state_keys:
                if state.get(key) is not None:
                    actual = state[key]
                    break
        if not isinstance(actual, dict):
            raise RuntimeError(f'{label} has no valid encoder contract; refusing to load an action space of unknown origin.')
        if actual != expected:
            differing = {
                key: (expected.get(key), actual.get(key))
                for key in set(expected) | set(actual)
                if expected.get(key) != actual.get(key)
            }
            raise RuntimeError(f'{label} encoder contract mismatch: {differing}.')

    def configure_optimizer_groups(self, trainer, groups):
        """Apply encoder-specific optimizer rates after common grouping."""
        unknown = [group['name'] for group in groups if group['name'] not in PARAM_GROUP_NAMES]
        if unknown:
            raise RuntimeError(f'Unknown optimizer parameter groups {unknown}; expected a subset of {list(PARAM_GROUP_NAMES)}.')
        return groups


def resolve_codes_per_transition(spec, action_cfg):
    """The encoder's code rate, checking any config that restates it.

    The rate is fixed by the architecture, so a config may only agree with it.
    """
    action_cfg = action_cfg or {}
    present = [key for key in spec.unsupported_action_cfg_keys if key in action_cfg]
    if present:
        raise ValueError(f'action_cfg keys {present} describe a readout the {spec.name!r} action encoder does not have.')
    required = int(spec.codes_per_transition)
    stated = action_cfg.get(spec.action_ratio_cfg_key)
    if stated is not None and int(stated) != required:
        raise ValueError(f'{spec.name} emits {required} codes per latent transition; {spec.action_ratio_cfg_key}={stated} disagrees.')
    return required
