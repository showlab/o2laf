"""Shared utilities for per-frame action conditioning.

An action encoder emits ``r`` codes per latent transition: the Wan DiT encoder
reads latents and emits one, V-JEPA2 emits two and the ST-Transformer four.
Everything that depends only on ``r`` -- packing to latent resolution, dropout,
shuffling -- lives here, so the transition, not the single code, is the unit of
every grouped operation. At ``r = 1`` each group holds one code and these
degenerate to the per-transition forms.
"""

from __future__ import annotations

import torch


# What "no action" means when the world model is asked for an unconditional
# prediction. The two differ only by the action embedder's own bias: a zero code
# still reaches the model as ``gamma * linear.bias``, while no bias reaches it as
# exactly nothing. They coincide at initialisation, because that bias starts at
# zero, and diverge as it trains -- in the released causal model by about 4% of a
# typical action's magnitude, which classifier-free guidance then amplifies.
#
# Each trainer's action dropout teaches the model one of the two, so inference
# must ask for the same one. A model records which in its own metadata rather
# than relying on a global default.
NULL_ACTION_ZERO_CODE = 'zero_code'
NULL_ACTION_NO_BIAS = 'no_bias'
NULL_ACTION_CONVENTIONS = (NULL_ACTION_ZERO_CODE, NULL_ACTION_NO_BIAS)


def validate_null_action(convention: str) -> str:
    """Return ``convention`` if it names a null-action convention, else raise."""
    if convention not in NULL_ACTION_CONVENTIONS:
        raise ValueError(f'Unknown null action {convention!r}; expected one of {NULL_ACTION_CONVENTIONS}.')
    return convention


def prepend_action_anchor(z: torch.Tensor) -> torch.Tensor:
    """Prepend a zero frame-0 anchor row to per-transition action codes.

    Turns per-transition codes ``[B, T-1, D]`` into a per-frame action matrix
    ``[B, T, D]`` whose first row is all zeros, because no transition produced
    the first frame.

    Args:
        z: Per-transition codes ``[B, T-1, D]``.

    Returns:
        ``[B, T, D]`` with ``out[:, 0] == 0``.
    """
    if z.dim() != 3:
        raise ValueError(f'z must be [B, T-1, D], got {tuple(z.shape)}')
    batch, _, dim = z.shape
    anchor = z.new_zeros(batch, 1, dim)
    return torch.cat([anchor, z], dim=1)


def process_actions_for_latent_space(
    actions: torch.Tensor,
    num_latent_frames: int,
    compression_ratio: int = 4,
) -> torch.Tensor:
    """Pack an action sequence ``[B, T, D]`` to latent resolution ``[B, F, r, D]``.

    Keep-first semantics:

    - group 0 repeats ``actions[:, 0]`` ``r`` times, because VAE latent frame 0
      encodes the initial, action-free frame;
    - group ``i >= 1`` takes the window of ``r`` actions starting at
      ``1 + (i-1)*r``, mirroring the WAN VAE temporal compression.

    With ``compression_ratio=1`` and ``T == num_latent_frames`` this is exactly
    ``actions.unsqueeze(2)``: latent frame ``i`` receives ``actions[:, i]``. Since
    the caller prepends a zero anchor, latent frame ``t+1`` receives transition
    code ``a_t``, which is the alignment the AdaLN injection assumes.

    Args:
        actions: Float tensor ``[B, T, D]``.
        num_latent_frames: Number of latent temporal frames F.
        compression_ratio: VAE temporal compression ratio r.

    Returns:
        Tensor ``[B, F, r, D]``.
    """
    if actions.dim() != 3:
        raise ValueError(f'actions must be [B, T, D], got {tuple(actions.shape)}')
    batch, num_actions, dim = actions.shape
    ratio = int(compression_ratio)
    frames = int(num_latent_frames)
    if ratio < 1:
        raise ValueError(f'compression_ratio must be >= 1, got {ratio}')
    if frames < 1:
        raise ValueError(f'num_latent_frames must be >= 1, got {frames}')

    out = torch.zeros(
        batch, frames, ratio, dim, device=actions.device, dtype=actions.dtype
    )
    out[:, 0] = actions[:, 0:1].expand(batch, ratio, dim)
    for index in range(1, frames):
        start = 1 + (index - 1) * ratio
        if start >= num_actions:
            continue
        chunk = actions[:, start:min(start + ratio, num_actions)]
        available = chunk.shape[1]
        if available < ratio:
            pad = torch.zeros(
                batch, ratio - available, dim,
                device=actions.device, dtype=actions.dtype,
            )
            chunk = torch.cat([chunk, pad], dim=1)
        out[:, index] = chunk
    return out


def check_latent_action_frame_budget(
    num_rgb_frames: int, compression_ratio: int = 4
) -> int:
    """Return the latent frame count for ``num_rgb_frames``, or raise.

    WAN's VAE is causal with temporal stride ``r``, so ``T`` RGB frames become
    ``(T - 1) // r + 1`` latent frames and ``T - 1`` must be divisible by ``r``.
    Valid choices at ``r=4`` include 17, 33, 41, 65 and 81 frames.
    """
    if num_rgb_frames < 1 + compression_ratio:
        raise ValueError(
            f'num_rgb_frames={num_rgb_frames} is too short for '
            f'compression_ratio={compression_ratio}'
        )
    if (num_rgb_frames - 1) % compression_ratio != 0:
        raise ValueError(
            f'(num_rgb_frames - 1) must be divisible by {compression_ratio}; '
            f'got {num_rgb_frames}. Use e.g. 17, 33, 41, 65, or 81.'
        )
    return (num_rgb_frames - 1) // compression_ratio + 1


def pack_action_groups(actions: torch.Tensor, ratio: int) -> torch.Tensor:
    """Pack ``[B, r*K, D]`` codes into ``[B, K+1, r, D]`` behind a zero anchor.

    Latent frame 0 encodes the initial, action-free frame, so it gets the zero
    anchor and latent frame ``t+1`` gets the ``r`` codes of transition ``t``.
    """
    if actions.ndim != 3 or actions.shape[1] % ratio:
        raise ValueError(f'Expected actions [B, {ratio}K, D], got {tuple(actions.shape)}.')
    num_latent_frames = actions.shape[1] // ratio + 1
    return process_actions_for_latent_space(
        prepend_action_anchor(actions), num_latent_frames=num_latent_frames, compression_ratio=ratio
    )


def drop_action_groups(actions: torch.Tensor, ratio: int, prob: float) -> tuple[torch.Tensor, float]:
    """Zero the codes of whole transitions, independently per sample and transition.

    Dropping a transition's ``r`` codes together is what makes dropout mean
    "this transition has no action" rather than "this transition is partly
    described". The code is zeroed rather than the embedder output, so a dropped
    transition still receives ``gamma * linear.bias``; KL is still charged on the
    full posterior. Returns the actions and the fraction dropped.
    """
    if actions.ndim != 3 or actions.shape[1] % ratio:
        raise ValueError(f'Expected actions [B, {ratio}K, D], got {tuple(actions.shape)}.')
    if not prob:
        return actions, 0.0
    groups = actions.shape[1] // ratio
    # One keep bit per transition, drawn before any broadcast so the number of
    # random values consumed depends on the transition count alone.
    keep = torch.rand(actions.shape[0], groups, 1, device=actions.device) >= prob
    dropped_fraction = 1.0 - float(keep.float().mean())
    keep = keep.repeat_interleave(ratio, dim=1)
    return actions * keep.to(actions.dtype), dropped_fraction


def shuffle_action_groups(actions: torch.Tensor, ratio: int, seed: int) -> tuple[torch.Tensor, list[int]]:
    """Permute ``[B, r*K, D]`` actions in groups of ``ratio``, keeping the order inside each group.

    The permutation is drawn on the CPU from ``seed``; an identity draw is rolled
    by one so the result always differs. Returns the actions and the order.
    """
    if actions.ndim != 3 or actions.shape[1] % ratio:
        raise ValueError(f'Expected actions [B, {ratio}K, D], got {tuple(actions.shape)}.')
    groups = actions.shape[1] // ratio
    if groups < 2:
        raise ValueError('Temporal shuffle needs at least two action groups.')
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    order = torch.randperm(groups, generator=generator)
    if torch.equal(order, torch.arange(groups)):
        order = order.roll(1)
    grouped = actions.reshape(actions.shape[0], groups, ratio, actions.shape[-1])
    return grouped[:, order.to(actions.device)].flatten(1, 2).contiguous(), order.tolist()
