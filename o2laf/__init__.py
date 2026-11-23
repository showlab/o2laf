"""O²LaF: latent action learning and action-conditioned world models on Wan 2.2."""

__version__ = '0.1.0'

__all__ = ['ActionGuidance', 'LatentActionPipeline']


def __getattr__(name):
    # Lazy, so that importing a training module does not load the inference stack.
    if name == 'LatentActionPipeline':
        from o2laf.inference.pipeline import LatentActionPipeline

        return LatentActionPipeline
    if name == 'ActionGuidance':
        from o2laf.inference.sampling import ActionGuidance

        return ActionGuidance
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
