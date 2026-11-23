"""Action encoders: models that infer latent actions from video.

One file per encoder, named by backbone architecture:

* :mod:`~o2laf.action_encoders.wan_dit` reads Wan VAE latents with the Wan DiT trunk;
* :mod:`~o2laf.action_encoders.vjepa2` reads RGB with a pretrained V-JEPA 2 encoder;
* :mod:`~o2laf.action_encoders.st_transformer` reads RGB with a randomly initialised
  spatio-temporal transformer.
"""

from o2laf.action_encoders import st_transformer, vjepa2, wan_dit  # noqa: F401  (registration)
from o2laf.action_encoders.base import build_action_encoder_spec
from o2laf.utils.registry import ACTION_ENCODER_REGISTRY


def build_action_encoder(name):
    """Return the action encoder class registered under ``name``."""
    return ACTION_ENCODER_REGISTRY.get(name)


__all__ = ['build_action_encoder', 'build_action_encoder_spec']
