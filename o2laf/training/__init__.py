"""Trainers: what a training run optimises.

* :class:`~o2laf.training.latent_action.LatentActionTrainer` jointly learns an
  action encoder and a causal world model. Which action encoder is a config
  choice, not a different trainer.
* :class:`~o2laf.training.world_model.WorldModelTrainer` freezes a trained
  action encoder and fully fine-tunes a bidirectional world model.
"""

from o2laf.training import latent_action, world_model  # noqa: F401  (registration)
from o2laf.utils.registry import TRAINER_REGISTRY


def build_trainer(name):
    """Return the trainer class registered under ``name``."""
    return TRAINER_REGISTRY.get(name)


__all__ = ['build_trainer']
