# V-JEPA2 source snapshot

- Upstream: <https://github.com/facebookresearch/vjepa2>
- Commit: `204698b45b3712590f06245fbfba32d3be539812`
- License: MIT; see `LICENSE`

This directory contains only the source closure required to construct the
registered `vit_giant_xformers` video encoder:

- `vision_transformer.py`
- `modules.py`
- `patch_embed.py`
- `pos_embs.py`
- `mask_utils.py`
- `tensors.py`

Dataset, training, evaluation, and robot-control code are intentionally not
vendored.

The upstream source files retain Meta's original copyright and MIT headers.
Local changes are limited to:

- changing `vision_transformer.py` from checkout-root `src.*` imports to
  package-relative imports;
- changing the deprecated `timm.models.layers` import to its equivalent
  `timm.layers` location.

Architecture and parameter names are unchanged.
