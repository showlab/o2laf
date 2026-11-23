"""Minimal V-JEPA2 model-source snapshot used by the latent-action backend."""

from .vision_transformer import VisionTransformer, vit_giant_xformers

__all__ = ['VisionTransformer', 'vit_giant_xformers']
