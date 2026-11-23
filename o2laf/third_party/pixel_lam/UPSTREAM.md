# Pixel latent-action encoder: source provenance

The files in this directory come from the latent action model of
**Olaf-World** (Show Lab), released under the MIT License. The license text is
in `LICENSE` next to this file.

- Project: Olaf-World, <https://showlab.github.io/Olaf-World/>
- License: MIT, Copyright (c) 2026 Show Lab

Only the encoder-side modules are kept -- spatio-temporal transformer blocks,
rotary embedding, patchification and the Gaussian action head -- for the
randomly initialised ST-Transformer action encoder. Checkpoint wrappers,
training loops, decoders, datasets and teacher encoders are excluded.

Architecture and parameter names are unchanged; the only edits are
package-relative imports and the license header.
