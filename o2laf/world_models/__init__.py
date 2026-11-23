"""World models: action-conditioned video generators."""

from o2laf.utils.registry import WORLD_MODEL_REGISTRY
from o2laf.world_models import wan  # noqa: F401  (registration)


def build_world_model(name):
    """Return the world model class registered under ``name``."""
    return WORLD_MODEL_REGISTRY.get(name)


def build_world_model_from_config(transformer_cfg, action_cfg, *, num_latent_frames, causal, dtype, model_class=None,
                                  **load_kwargs):
    """Construct an action-conditioned world model from the config, ready to adapt.

    Training and inference agree up to this point and diverge after it: training
    attaches PEFT, selects trainable parameters and wraps in FSDP2; inference
    loads weights and freezes. Only the construction is shared, so that which
    config key means what is written down once.

    ``load_kwargs`` (``cache_dir``, ``token``, ...) go to ``from_pretrained``.
    ``model_class`` bypasses the registry lookup of ``transformer_cfg['model_type']``;
    tests use it to substitute a stand-in for the 5B-parameter model.
    """
    if model_class is None:
        model_class = build_world_model(transformer_cfg['model_type'])
    world_model = model_class.from_pretrained(
        transformer_cfg['model_name'], subfolder='transformer', torch_dtype=dtype, **load_kwargs
    )
    world_model.setup_world_model(
        num_latent_frames=int(num_latent_frames),
        causal=bool(causal),
        action_dim=int((action_cfg or {}).get('latent_action_dim', 32)),
        action_gain_init=float((action_cfg or {}).get('gamma_init', 0.5)),
    )
    return world_model


__all__ = ['build_world_model', 'build_world_model_from_config']
