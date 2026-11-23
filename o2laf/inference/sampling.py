"""Euler sampling of the world model, with optional classifier-free guidance on the actions.

Guidance combines a conditional and an unconditional velocity,
``v = v_uncond + scale * (v_cond - v_uncond)``. ``scale == 1`` returns the
conditional velocity exactly and skips the unconditional forward. The prompt is
a constant, so the action is the only guided condition.

The unconditional branch replaces the action bias with the *null-action* bias,
which is what the model's training taught it "no action" looks like: either no
bias at all, or the bias a zero action code produces. See
``utils.action_util.NULL_ACTION_*``.

Two forms of "unconditional" are available:

* ``full``: every action frame's bias is removed at once (two forwards per step);
* ``per_frame``: frame ``t`` comes from the branch in which only frame ``t``'s
  bias is removed (one extra forward per action frame, batched).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

UNCONDITIONAL_MODES = ('full', 'per_frame')


@dataclass(frozen=True)
class ActionGuidance:
    """Classifier-free guidance settings for the action condition.

    Attributes:
        scale: guidance scale; 1.0 disables guidance.
        unconditional_mode: ``'full'`` or ``'per_frame'`` (see module docstring).
        guidance_sigma_min: if set, guide only at steps with sigma >= this value.
        variant_batch_size: unconditional variants per forward in ``per_frame`` mode.
    """

    scale: float = 1.0
    unconditional_mode: str = 'full'
    guidance_sigma_min: float | None = None
    variant_batch_size: int = 4

    def __post_init__(self):
        if self.scale < 0:
            raise ValueError(f'scale must be nonnegative, got {self.scale}.')
        if self.unconditional_mode not in UNCONDITIONAL_MODES:
            raise ValueError(f'unconditional_mode must be one of {UNCONDITIONAL_MODES}, got {self.unconditional_mode!r}.')
        if self.guidance_sigma_min is not None and not 0.0 < self.guidance_sigma_min < 1.0:
            raise ValueError(f'guidance_sigma_min must be in (0, 1), got {self.guidance_sigma_min}.')
        if self.variant_batch_size < 1:
            raise ValueError('variant_batch_size must be positive.')

    @property
    def active(self) -> bool:
        return self.scale != 1.0


# The longest video either world model generates, in RGB frames (7.5 s at
# 16 fps). Both world models run at any length up to this. Past the length
# they were trained on (81 or 41 frames), they extrapolate.
MAX_RGB_FRAMES = 121


def combine_guidance(conditional: torch.Tensor, unconditional: torch.Tensor, scale: float) -> torch.Tensor:
    """``unconditional + scale * (conditional - unconditional)``; exactly ``conditional`` at scale 1."""
    if conditional.shape != unconditional.shape:
        raise ValueError(
            f'Guidance branches have different shapes: {tuple(conditional.shape)} and {tuple(unconditional.shape)}.'
        )
    if scale == 1.0:
        return conditional
    return unconditional + float(scale) * (conditional - unconditional)


def action_frame_indices(num_latent_frames: int) -> tuple[int, ...]:
    """Latent frames that carry an action bias: every frame after the clean first one."""
    if num_latent_frames < 2:
        raise ValueError('Action guidance requires at least two latent frames.')
    return tuple(range(1, num_latent_frames))


def build_unconditional_biases(
    action_bias: torch.Tensor, mode: str, *, action_frames: Sequence[int], null_bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Unconditional bias variants: the null-action bias everywhere (``full``) or one frame at a time (``per_frame``).

    ``null_bias`` is what the model was taught "no action here" looks like. It
    defaults to zeros, which is the no-bias convention; a model trained with
    zero-code dropout passes the bias a zero code produces instead.
    """
    if action_bias.ndim != 3 or action_bias.shape[0] != 1:
        raise ValueError(f'action_bias must be [1, F, C], got {tuple(action_bias.shape)}.')
    if null_bias is None:
        null_bias = torch.zeros_like(action_bias)
    elif null_bias.shape != action_bias.shape:
        raise ValueError(f'null_bias must be {tuple(action_bias.shape)}, got {tuple(null_bias.shape)}.')
    if mode == 'full':
        return null_bias
    if mode != 'per_frame':
        raise ValueError(f'Unknown unconditional mode {mode!r}.')
    frames = tuple(int(frame) for frame in action_frames)
    if not frames:
        raise ValueError('per_frame guidance needs at least one action frame.')
    if min(frames) < 0 or max(frames) >= action_bias.shape[1]:
        raise ValueError(f'Action frames {frames} are outside [0, {action_bias.shape[1]}).')
    variants = action_bias.expand(len(frames), -1, -1).clone()
    for row, frame in enumerate(frames):
        variants[row, frame] = null_bias[0, frame]
    return variants


def gather_per_frame_unconditional(
    variants: torch.Tensor, action_frames: Sequence[int], conditional: torch.Tensor
) -> torch.Tensor:
    """Assemble each action frame from the variant that removed that frame's bias."""
    frames = tuple(int(frame) for frame in action_frames)
    if variants.shape[0] != len(frames):
        raise ValueError(f'Expected {len(frames)} variants, got {variants.shape[0]}.')
    if variants.shape[1:] != conditional.shape[1:]:
        raise ValueError(
            f'Variant shape {tuple(variants.shape[1:])} does not match conditional shape {tuple(conditional.shape[1:])}.'
        )
    if conditional.ndim != 5:
        raise ValueError(f'Velocity must be [B, C, F, H, W], got {tuple(conditional.shape)}.')
    result = conditional.clone()
    for row, frame in enumerate(frames):
        result[:, :, frame] = variants[row:row + 1, :, frame]
    return result


def _forward_variants(predict, latents, bias_variants, timestep, next_timestep, batch_size):
    outputs = []
    for start in range(0, bias_variants.shape[0], batch_size):
        block = bias_variants[start:start + batch_size]
        count = block.shape[0]
        outputs.append(
            predict(
                latents.expand(count, -1, -1, -1, -1),
                block,
                timestep.expand(count, -1),
                next_timestep.expand(count, -1),
            )
        )
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def sample_with_guidance(
    *,
    predict: Callable[[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor], torch.Tensor],
    scheduler,
    latent_shape: tuple[int, ...],
    action_bias: torch.Tensor | None,
    anchor: torch.Tensor | None,
    num_context_frames: int,
    num_steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    guidance: ActionGuidance,
    null_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler-sample a clip from CPU-seeded noise, pinning the clean context frames throughout.

    ``predict(latents, action_bias, timestep, next_timestep)`` returns the velocity.
    """
    if len(latent_shape) != 5 or latent_shape[0] != 1:
        raise ValueError(f'latent_shape must be [1, C, F, H, W], got {latent_shape}.')
    if num_steps < 1:
        raise ValueError('num_steps must be positive.')
    if not 0 <= num_context_frames < latent_shape[2]:
        raise ValueError(f'num_context_frames={num_context_frames} is invalid for {latent_shape[2]} latent frames.')
    if anchor is None and num_context_frames:
        raise ValueError('A nonzero context length requires an anchor tensor.')
    if anchor is not None and anchor.shape[2] < num_context_frames:
        raise ValueError(f'Anchor has {anchor.shape[2]} frames but {num_context_frames} are required.')
    if guidance.active and action_bias is None:
        raise ValueError('Action guidance requires an action bias.')

    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    latents = torch.randn(latent_shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    scheduler.set_timesteps(int(num_steps), device=device)
    timesteps = torch.cat([scheduler.timesteps, scheduler.timesteps.new_tensor([0])])
    num_latent_frames = latent_shape[2]
    action_frames = action_frame_indices(num_latent_frames)
    bias_variants = (
        build_unconditional_biases(
            action_bias, guidance.unconditional_mode, action_frames=action_frames, null_bias=null_bias
        )
        if guidance.active
        else None
    )
    train_timesteps = float(scheduler.config.num_train_timesteps)

    for index, timestep_scalar in enumerate(timesteps[:-1]):
        next_scalar = timesteps[index + 1]
        if timestep_scalar == next_scalar:
            continue
        timestep = timestep_scalar.reshape(1).expand(1, num_latent_frames).contiguous()
        next_timestep = next_scalar.reshape(1).expand(1, num_latent_frames).contiguous()
        if num_context_frames:
            latents[:, :, :num_context_frames] = anchor[:, :, :num_context_frames]
            timestep[:, :num_context_frames] = 0
            next_timestep[:, :num_context_frames] = 0

        conditional = predict(latents, action_bias, timestep, next_timestep)
        sigma = float(timestep_scalar) / train_timesteps
        guide = guidance.active and (guidance.guidance_sigma_min is None or sigma >= guidance.guidance_sigma_min)
        if guide:
            variants = _forward_variants(
                predict, latents, bias_variants, timestep, next_timestep, guidance.variant_batch_size
            )
            unconditional = (
                variants
                if guidance.unconditional_mode == 'full'
                else gather_per_frame_unconditional(variants, action_frames, conditional)
            )
            velocity = combine_guidance(conditional, unconditional, guidance.scale)
        else:
            velocity = conditional
        latents = scheduler.step(velocity, timestep_scalar, latents, next_scalar).prev_sample

    if num_context_frames:
        latents[:, :, :num_context_frames] = anchor[:, :, :num_context_frames]
    return latents
