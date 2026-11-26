"""Raw pairwise action-query attention extraction and visualisation for Wan DiT encoders.

During a Wan DiT latent action encoder forward pass, each transition ``t -> t+1``
has one interleaved action query token ``a_t`` that attends exclusively to the
video tokens of latent frames ``t`` (source) and ``t+1`` (target).

This module captures the post-projection, post-QK-norm, post-RoPE query and key
tensors at a requested transformer block (1-indexed, ``1..30``) and recomputes
the exact joint-softmax attention distribution over all ``2 * H * W`` video keys
for a single head (0-indexed, ``0..23``). Source and target share one softmax
denominator (``source_mass + target_mass = 1``) and one video-global colour
scale; no head/block aggregation, conditional normalisation, thresholding,
debiasing, Gaussian blur, or temporal smoothing is applied.
"""

from __future__ import annotations

import gc
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from PIL import Image, ImageDraw, ImageFont

from o2laf.action_encoders.base import unwrap_peft
from o2laf.action_encoders.wan_dit import PaddedFlexAttnProcessor, WanDiTActionEncoder
from o2laf.inference import hub
from o2laf.inference.pipeline import _build_action_encoder
from o2laf.inference.prompts import fixed_prompt_embeds
from o2laf.utils.file_util import write_json_atomic
from o2laf.utils.video_util import VideoLoader, select_frame_indices

DEFAULT_BLOCK = 30
DEFAULT_HEAD = 11
DEFAULT_WINDOW_FRAMES = 121
DEFAULT_PERCENTILE = 99.0
DEFAULT_GAMMA = 0.8
DEFAULT_ALPHA = 0.6
DEFAULT_ALPHA_POWER = 1.1

GAP_WIDTH = 8
CAPTION_HEIGHT = 58


@dataclass(frozen=True)
class AttentionRenderConfig:
    """Display parameters for side-by-side source|target attention overlays."""

    percentile: float = DEFAULT_PERCENTILE
    gamma: float = DEFAULT_GAMMA
    alpha: float = DEFAULT_ALPHA
    alpha_power: float = DEFAULT_ALPHA_POWER

    def __post_init__(self) -> None:
        if not 0.0 < float(self.percentile) <= 100.0:
            raise ValueError(f'percentile must lie in (0, 100], got {self.percentile}.')
        if float(self.gamma) <= 0.0:
            raise ValueError(f'gamma must be positive, got {self.gamma}.')
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError(f'alpha must lie in [0, 1], got {self.alpha}.')
        if float(self.alpha_power) <= 0.0:
            raise ValueError(f'alpha_power must be positive, got {self.alpha_power}.')

    def to_dict(self) -> dict[str, Any]:
        return {
            'cmap': 'turbo',
            'interpolation': 'bicubic',
            'percentile': float(self.percentile),
            'gamma': float(self.gamma),
            'alpha': float(self.alpha),
            'alpha_power': float(self.alpha_power),
            'gaussian_blur': False,
        }


@dataclass
class AttentionResult:
    """Extracted raw pairwise attention maps and aligned RGB transition endpoints."""

    maps: np.ndarray  # [T, 2, H, W] float32 joint-softmax probabilities
    source_frames: np.ndarray  # [T, H_rgb, W_rgb, 3] uint8
    target_frames: np.ndarray  # [T, H_rgb, W_rgb, 3] uint8
    source_frame_indices: np.ndarray  # [T] int64 native source-video indices
    target_frame_indices: np.ndarray  # [T] int64 native source-video indices
    block: int
    head: int
    target_fps: float
    window_frames: int
    windows: list[dict[str, Any]]


def _resample_indices(total_frames: int, source_fps: float, target_fps: float) -> list[int]:
    if total_frames < 1:
        return []
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError('source_fps and target_fps must be positive.')
    if source_fps <= target_fps + 1e-6:
        return list(range(total_frames))
    return select_frame_indices(total_frames, source_fps, target_fps)


def adjacent_temporal_windows(
    num_frames: int,
    *,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    temporal_compression_ratio: int = 4,
) -> list[tuple[int, int]]:
    """Adjacent windows sharing one endpoint; each length is ``1 mod ratio``."""
    ratio = int(temporal_compression_ratio)
    window = int(window_frames)
    if ratio < 1:
        raise ValueError('temporal_compression_ratio must be positive.')
    if window < 1 + ratio or (window - 1) % ratio:
        raise ValueError(f'window_frames must be {ratio}k+1 with k >= 1, got {window}.')
    windows = []
    start = 0
    while start + ratio < int(num_frames):
        available = int(num_frames) - start
        length = min(window, 1 + ratio * ((available - 1) // ratio))
        if length < 1 + ratio:
            break
        windows.append((start, length))
        start += window - 1
    return windows


def pairwise_attention_from_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    num_frames: int,
    *,
    grid_height: int,
    grid_width: int,
    heads: Sequence[int] | None = None,
) -> torch.Tensor:
    """Exact pairwise action-query attention over the complete ``2*H*W`` key set.

    Args:
        query, key: Post-projection, post-QK-norm, post-RoPE tensors shaped
            ``[B, num_heads, padded_seq_len, head_dim]``.
        num_frames: Number of latent frames ``F`` (produces ``F - 1`` transitions).
        grid_height, grid_width: Post-patch spatial token grid dimensions.
        heads: Optional subset of 0-indexed attention heads to compute.

    Returns:
        Float32 tensor ``[B, F - 1, H_sel, 2, grid_height, grid_width]`` where
        axis 3 is ``(source, target)`` and sums jointly to 1 per transition/head.
    """
    if query.ndim != 4 or key.ndim != 4 or query.shape != key.shape:
        raise ValueError(f'query and key must share shape [B,H,S,D], got {tuple(query.shape)} and {tuple(key.shape)}.')
    num_frames = int(num_frames)
    if num_frames < 2:
        raise ValueError(f'pairwise attention needs >= 2 latent frames, got {num_frames}.')
    grid_height, grid_width = int(grid_height), int(grid_width)
    tokens_per_frame = grid_height * grid_width
    per_frame = tokens_per_frame + 1
    real_length = num_frames * per_frame
    if query.shape[2] < real_length:
        raise ValueError(f'q/k sequence length {query.shape[2]} is shorter than {real_length}.')

    if heads is not None:
        num_heads = int(query.shape[1])
        head_indices = [int(h) for h in heads]
        invalid = [h for h in head_indices if h < 0 or h >= num_heads]
        if invalid:
            raise ValueError(f'Head indices must lie in [0, {num_heads - 1}], got {invalid}.')
        query = query[:, head_indices]
        key = key[:, head_indices]

    scale = 1.0 / math.sqrt(query.shape[-1])
    maps = []
    for transition in range(num_frames - 1):
        source_begin = transition * per_frame
        source_span = slice(source_begin, source_begin + tokens_per_frame)
        action_index = source_begin + tokens_per_frame
        target_begin = (transition + 1) * per_frame
        target_span = slice(target_begin, target_begin + tokens_per_frame)

        action = query[:, :, action_index, :].float()
        video = torch.cat([key[:, :, source_span, :], key[:, :, target_span, :]], dim=2).float()
        logits = torch.einsum('bhd,bhnd->bhn', action, video) * scale
        weights = torch.softmax(logits, dim=-1)
        maps.append(weights.reshape(query.shape[0], query.shape[1], 2, grid_height, grid_width))
    return torch.stack(maps, dim=1)


class _CapturingPaddedFlexAttnProcessor(PaddedFlexAttnProcessor):
    """Capture exact pairwise attention at ``_attend`` without altering outputs."""

    def __init__(
        self,
        *,
        block: int,
        num_frames: int,
        grid_height: int,
        grid_width: int,
        heads: tuple[int, ...] | None,
        sink: dict[int, torch.Tensor],
    ) -> None:
        super().__init__()
        self._block = int(block)
        self._num_frames = int(num_frames)
        self._grid_height = int(grid_height)
        self._grid_width = int(grid_width)
        self._heads = heads
        self._sink = sink

    def _attend(self, query, key, value, attention_mask):
        self._sink[self._block] = (
            pairwise_attention_from_qk(
                query,
                key,
                self._num_frames,
                grid_height=self._grid_height,
                grid_width=self._grid_width,
                heads=self._heads,
            )
            .detach()
            .cpu()
        )
        return super()._attend(query, key, value, attention_mask)


@contextmanager
def capture_pairwise_attention(
    action_encoder: torch.nn.Module,
    *,
    blocks: Sequence[int],
    num_frames: int,
    grid_height: int,
    grid_width: int,
    heads: Sequence[int] | None = None,
):
    """Temporarily instrument selected 1-indexed blocks of a ``WanDiTActionEncoder``."""
    base = unwrap_peft(action_encoder)
    if not isinstance(base, WanDiTActionEncoder):
        raise TypeError(
            f'Pairwise action-query attention is only defined for WanDiTActionEncoder, got {type(base).__name__}.'
        )
    num_blocks = len(base.blocks)
    requested = tuple(sorted(set(int(block) for block in blocks)))
    if not requested or any(block < 1 or block > num_blocks for block in requested):
        raise ValueError(f'blocks must lie in [1, {num_blocks}], got {list(blocks)}.')
    head_tuple = tuple(int(h) for h in heads) if heads is not None else None
    if head_tuple is not None:
        num_heads = int(base.config.num_attention_heads)
        if not head_tuple or any(h < 0 or h >= num_heads for h in head_tuple):
            raise ValueError(f'heads must lie in [0, {num_heads - 1}], got {list(head_tuple)}.')

    captured: dict[int, torch.Tensor] = {}
    originals: dict[int, Any] = {}
    for block in requested:
        attn = base.blocks[block - 1].attn1
        originals[block] = attn.processor
        attn.set_processor(
            _CapturingPaddedFlexAttnProcessor(
                block=block,
                num_frames=num_frames,
                grid_height=grid_height,
                grid_width=grid_width,
                heads=head_tuple,
                sink=captured,
            )
        )
    try:
        yield captured
    finally:
        for block, processor in originals.items():
            base.blocks[block - 1].attn1.set_processor(processor)


def head_clip_scale(maps: np.ndarray, percentile: float = DEFAULT_PERCENTILE) -> float:
    """One shared colour scale over ``[T, 2, H, W]`` (all frames, both sides)."""
    scale = float(np.percentile(np.asarray(maps, dtype=np.float32), float(percentile)))
    return max(scale, 1e-12)


def render_attention_overlay(
    frame: np.ndarray,
    attention: np.ndarray,
    *,
    scale: float,
    config: AttentionRenderConfig = AttentionRenderConfig(),
) -> np.ndarray:
    """Overlay a single ``[H, W]`` attention map onto an RGB uint8 frame."""
    normalized = np.clip(np.asarray(attention, dtype=np.float32) / float(scale), 0.0, 1.0)
    normalized = cv2.resize(normalized, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    normalized = np.clip(normalized, 0.0, 1.0)
    heatmap = np.power(normalized, float(config.gamma))
    colour = cv2.applyColorMap(np.round(heatmap * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB).astype(np.float32)
    opacity = (float(config.alpha) * np.power(heatmap, float(config.alpha_power)))[..., None]
    blended = (1.0 - opacity) * frame.astype(np.float32) + opacity * colour
    return np.clip(blended, 0.0, 255.0).round().astype(np.uint8)


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    for directory in (
        Path('/usr/share/fonts/truetype/dejavu'),
        Path('/usr/share/fonts/dejavu'),
        Path('/Library/Fonts'),
    ):
        candidate = directory / name
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_caption_bar(
    *,
    panel_width: int,
    block: int,
    head: int,
    source_mass: float,
    target_mass: float,
    source_index: int,
    target_index: int,
    transition: int,
    label_font: ImageFont.ImageFont | None = None,
    small_font: ImageFont.ImageFont | None = None,
) -> np.ndarray:
    """Caption strip placed below the side-by-side SOURCE | TARGET panels."""
    label_font = label_font or _load_font(20, bold=True)
    small_font = small_font or _load_font(16)
    width = 2 * int(panel_width) + GAP_WIDTH
    image = Image.new('RGB', (width, CAPTION_HEIGHT), (20, 22, 26))
    draw = ImageDraw.Draw(image)
    entries = (
        (
            0,
            f'SOURCE  |  block {block:02d} head {head:02d}  |  mass {source_mass:.1%}',
            f'frame {source_index}  |  transition {transition}',
        ),
        (
            int(panel_width) + GAP_WIDTH,
            f'TARGET  |  block {block:02d} head {head:02d}  |  mass {target_mass:.1%}',
            f'frame {target_index}  |  transition {transition}',
        ),
    )
    for x, title, subtitle in entries:
        draw.text((x + 10, 6), title, fill=(247, 247, 247), font=label_font)
        draw.text((x + 10, 34), subtitle, fill=(185, 190, 198), font=small_font)
    draw.rectangle((int(panel_width), 0, int(panel_width) + GAP_WIDTH - 1, image.height), fill=(238, 238, 238))
    return np.asarray(image)


def render_attention_video(
    result: AttentionResult,
    output_path: str | Path,
    *,
    fps: float | None = None,
    config: AttentionRenderConfig = AttentionRenderConfig(),
) -> dict[str, Any]:
    """Render side-by-side SOURCE | TARGET raw attention MP4 and return summary stats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_fps = float(fps) if fps is not None else float(result.target_fps) / 4.0
    if render_fps <= 0.0:
        raise ValueError(f'output fps must be positive, got {render_fps}.')

    scale = head_clip_scale(result.maps, percentile=config.percentile)
    label_font = _load_font(20, bold=True)
    small_font = _load_font(16)
    gap = np.full((result.source_frames.shape[1], GAP_WIDTH, 3), 238, dtype=np.uint8)
    source_masses = []
    target_masses = []

    writer = imageio.get_writer(
        str(output_path),
        fps=render_fps,
        codec='libx264',
        quality=8,
        macro_block_size=1,
    )
    try:
        for transition in range(len(result.maps)):
            pair = result.maps[transition]
            src_mass = float(pair[0].sum())
            tgt_mass = float(pair[1].sum())
            source_masses.append(src_mass)
            target_masses.append(tgt_mass)
            left = render_attention_overlay(result.source_frames[transition], pair[0], scale=scale, config=config)
            right = render_attention_overlay(result.target_frames[transition], pair[1], scale=scale, config=config)
            pixels = np.concatenate([left, gap, right], axis=1)
            caption = render_caption_bar(
                panel_width=left.shape[1],
                block=result.block,
                head=result.head,
                source_mass=src_mass,
                target_mass=tgt_mass,
                source_index=int(result.source_frame_indices[transition]),
                target_index=int(result.target_frame_indices[transition]),
                transition=transition,
                label_font=label_font,
                small_font=small_font,
            )
            writer.append_data(np.concatenate([pixels, caption], axis=0))
    finally:
        writer.close()

    return {
        'output': str(output_path.resolve()),
        'output_fps': render_fps,
        'num_transitions': int(len(result.maps)),
        'head_clip_scale': scale,
        'source_mass_mean': float(np.mean(source_masses)),
        'target_mass_mean': float(np.mean(target_masses)),
        'vis_config': config.to_dict(),
    }


def save_attention_npz(result: AttentionResult, path: str | Path) -> Path:
    """Save raw attention probabilities and aligned RGB frames as a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        attention=result.maps.astype(np.float32),
        source_rgb=result.source_frames.astype(np.uint8),
        target_rgb=result.target_frames.astype(np.uint8),
        source_frame_indices=result.source_frame_indices.astype(np.int64),
        target_frame_indices=result.target_frame_indices.astype(np.int64),
        block=np.asarray(int(result.block), dtype=np.int64),
        head=np.asarray(int(result.head), dtype=np.int64),
        target_fps=np.asarray(float(result.target_fps), dtype=np.float32),
    )
    return path


@dataclass
class ActionAttentionExtractor:
    """Encoder-only loader for extracting raw pairwise attention without loading a world model."""

    model_dir: Path
    model_index: dict[str, Any]
    action_space_id: str
    action_encoder: torch.nn.Module
    vae: AutoencoderKLWan
    prompt_embeddings: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    height: int
    width: int
    fps: int
    temporal_compression_ratio: int
    action_encoder_step: int

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        base_model: str | Path | None = None,
        device: str | torch.device = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        verify_action_space: bool = False,
    ) -> 'ActionAttentionExtractor':
        """Load only the Wan DiT action encoder and VAE from a Hub model or local directory."""
        device = torch.device(device)
        root = hub.resolve_model_dir(
            pretrained_model_name_or_path,
            fetch='action_encoder',
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        index = hub.read_model_index(root)
        encoder_at = index['action_encoder']
        encoder_config = hub.load_part_config(root, encoder_at)
        encoder_name = str(encoder_config.get('action_encoder'))
        if encoder_name != 'wan_dit':
            raise ValueError(
                f'Action-query attention visualisation requires a wan_dit latent action encoder, '
                f'but {pretrained_model_name_or_path} uses {encoder_name!r}.'
            )
        action_space_id = str(index['action_space_id'])
        if encoder_config.get('action_space_id') != action_space_id:
            raise RuntimeError(
                f'The action encoder was trained in action space {encoder_config.get("action_space_id")}, '
                f'but the model index records {action_space_id}.'
            )
        encoder_state = hub.load_part_state(root, encoder_at)
        if verify_action_space:
            actual = 'sha256:' + hub.tensor_content_sha256(encoder_state)
            if actual != action_space_id:
                raise RuntimeError(f'The action encoder weights hash to {actual}, not the recorded {action_space_id}.')

        base_model_path = str(base_model or index['base_model'])
        base_kwargs = {'cache_dir': cache_dir, 'local_files_only': local_files_only, 'token': token}
        prompt_embeddings = fixed_prompt_embeds(
            base_model=index['base_model'],
            prompt=index['fixed_prompt'],
            base_model_path=base_model_path,
            device=device,
            dtype=dtype,
            **base_kwargs,
        )
        vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder='vae', torch_dtype=dtype, **base_kwargs).to(device)
        vae.requires_grad_(False)
        vae.eval()
        action_encoder = _build_action_encoder(encoder_state, encoder_config, base_model_path, device, dtype, **base_kwargs)
        del encoder_state

        mode_config = index['modes'].get('bidirectional') or next(iter(index['modes'].values()))
        extractor = cls(
            model_dir=root,
            model_index=index,
            action_space_id=action_space_id,
            action_encoder=action_encoder,
            vae=vae,
            prompt_embeddings=prompt_embeddings,
            device=device,
            dtype=dtype,
            height=int(mode_config['height']),
            width=int(mode_config['width']),
            fps=int(mode_config['fps']),
            temporal_compression_ratio=int(index['temporal_compression_ratio']),
            action_encoder_step=int(encoder_config['global_step']),
        )
        gc.collect()
        return extractor

    @classmethod
    def from_pipeline(cls, pipeline) -> 'ActionAttentionExtractor':
        """Reuse an already-loaded ``LatentActionPipeline`` without loading extra weights."""
        base = unwrap_peft(pipeline.action_encoder)
        if not isinstance(base, WanDiTActionEncoder):
            raise ValueError(
                f'Action-query attention visualisation requires a wan_dit latent action encoder, '
                f'got {pipeline.action_encoder_name!r}.'
            )
        encoder_cfg = hub.load_part_config(pipeline.model_dir, pipeline.model_index['action_encoder'])
        return cls(
            model_dir=pipeline.model_dir,
            model_index=pipeline.model_index,
            action_space_id=pipeline.action_space_id,
            action_encoder=pipeline.action_encoder,
            vae=pipeline.vae,
            prompt_embeddings=pipeline.prompt_embeddings,
            device=pipeline.device,
            dtype=pipeline.dtype,
            height=pipeline.height,
            width=pipeline.width,
            fps=pipeline.fps,
            temporal_compression_ratio=pipeline.temporal_compression_ratio,
            action_encoder_step=int(encoder_cfg['global_step']),
        )

    @torch.no_grad()
    def _encode_window(self, frames: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        tensor = torch.as_tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2)
        resized = F.interpolate(tensor, size=(self.height, self.width), mode='bilinear', align_corners=False, antialias=True)
        endpoints = resized[::self.temporal_compression_ratio].round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

        pixels = resized.div(255.0).sub_(0.5).div_(0.5).unsqueeze(0)
        video = pixels.permute(0, 2, 1, 3, 4).contiguous().to(device=self.device, dtype=self.dtype)
        moments = self.vae._encode(video)
        mean, _ = torch.chunk(moments, 2, dim=1)
        latent_mean = torch.as_tensor(self.vae.config.latents_mean, device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        latent_std = torch.as_tensor(self.vae.config.latents_std, device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        latents = ((mean.float() - latent_mean) / latent_std).to(self.dtype)
        if endpoints.shape[0] != latents.shape[2]:
            raise RuntimeError(f'RGB endpoint/latent mismatch: {endpoints.shape[0]} vs {latents.shape[2]}.')
        return latents, endpoints

    @torch.no_grad()
    def extract(
        self,
        video_path: str | Path,
        *,
        block: int = DEFAULT_BLOCK,
        head: int = DEFAULT_HEAD,
        target_fps: float | None = None,
        source_fps: float | None = None,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
    ) -> AttentionResult:
        """Extract raw pairwise attention ``[T, 2, H, W]`` across a full video."""
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f'Video not found: {video_path}.')

        base = unwrap_peft(self.action_encoder)
        _, patch_h, patch_w = base.config.patch_size
        effective_fps = float(target_fps) if target_fps is not None else float(self.fps)

        with VideoLoader(str(video_path), name=str(video_path)) as loader:
            native_fps = float(source_fps or loader.get_fps())
            picks = _resample_indices(loader.get_frame_count(), native_fps, effective_fps)
            windows = adjacent_temporal_windows(
                len(picks),
                window_frames=int(window_frames),
                temporal_compression_ratio=self.temporal_compression_ratio,
            )
            if not windows:
                raise ValueError(
                    f'{video_path} has only {len(picks)} frames at {effective_fps:g} fps; '
                    f'need at least {1 + self.temporal_compression_ratio} frames for one transition.'
                )

            source_frames: list[np.ndarray] = []
            target_frames: list[np.ndarray] = []
            source_indices: list[int] = []
            target_indices: list[int] = []
            attention_chunks: list[np.ndarray] = []
            window_reports: list[dict[str, Any]] = []

            for window_index, (start, length) in enumerate(windows):
                indices = [picks[i] for i in range(start, start + length)]
                decoded = loader.get_frames(indices)
                latents, endpoints = self._encode_window(decoded)
                num_latent_frames = int(latents.shape[2])
                grid_height = int(latents.shape[3]) // int(patch_h)
                grid_width = int(latents.shape[4]) // int(patch_w)
                timestep = torch.zeros(1, num_latent_frames, device=self.device, dtype=torch.float32)

                with capture_pairwise_attention(
                    self.action_encoder,
                    blocks=(int(block),),
                    num_frames=num_latent_frames,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    heads=(int(head),),
                ) as captured:
                    self.action_encoder(
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=self.prompt_embeddings,
                        r_timestep=timestep,
                    )

                endpoint_indices = indices[::self.temporal_compression_ratio]
                if len(endpoint_indices) != num_latent_frames:
                    raise RuntimeError(f'{video_path.name} window {window_index}: endpoint/latent mismatch.')

                # captured[block] is [1, F-1, 1, 2, H, W]; squeeze batch and single-head axes.
                attention_chunks.append(captured[int(block)][0, :, 0].numpy().astype(np.float32))
                source_frames.extend(endpoints[:-1])
                target_frames.extend(endpoints[1:])
                source_indices.extend(endpoint_indices[:-1])
                target_indices.extend(endpoint_indices[1:])
                window_reports.append({
                    'window': window_index,
                    'resampled_start': int(start),
                    'rgb_frames': int(length),
                    'latent_frames': num_latent_frames,
                    'transitions': num_latent_frames - 1,
                    'source_frame_span': [int(indices[0]), int(indices[-1])],
                })
                del decoded, latents
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        return AttentionResult(
            maps=np.concatenate(attention_chunks, axis=0),
            source_frames=np.stack(source_frames, axis=0),
            target_frames=np.stack(target_frames, axis=0),
            source_frame_indices=np.asarray(source_indices, dtype=np.int64),
            target_frame_indices=np.asarray(target_indices, dtype=np.int64),
            block=int(block),
            head=int(head),
            target_fps=effective_fps,
            window_frames=int(window_frames),
            windows=window_reports,
        )

    def visualize(
        self,
        video_path: str | Path,
        output_path: str | Path,
        *,
        block: int = DEFAULT_BLOCK,
        head: int = DEFAULT_HEAD,
        target_fps: float | None = None,
        source_fps: float | None = None,
        output_fps: float | None = None,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        render_config: AttentionRenderConfig = AttentionRenderConfig(),
        save_npz: str | Path | None = None,
    ) -> dict[str, Any]:
        """Extract attention, render the MP4, write its JSON manifest, and optionally save raw NPZ."""
        video_path = Path(video_path)
        output_path = Path(output_path)
        result = self.extract(
            video_path,
            block=block,
            head=head,
            target_fps=target_fps,
            source_fps=source_fps,
            window_frames=window_frames,
        )
        render_summary = render_attention_video(result, output_path, fps=output_fps, config=render_config)
        npz_path = save_attention_npz(result, save_npz) if save_npz is not None else None
        manifest = {
            'schema_version': 'o2laf_attention_viz_v1',
            'model': str(self.model_dir),
            'action_space_id': self.action_space_id,
            'action_encoder_step': self.action_encoder_step,
            'video': str(video_path.resolve()),
            'block': result.block,
            'head': result.head,
            'target_fps': result.target_fps,
            'window_frames': result.window_frames,
            'windows': result.windows,
            'attention_semantics': {
                'raw_joint_softmax': True,
                'source_target_shared_denominator': True,
                'single_block_single_head': True,
                'normalization_scope': 'all frames, source and target, for this (block, head) clip',
            },
            **render_summary,
            'npz': str(npz_path.resolve()) if npz_path is not None else None,
        }
        manifest_path = output_path.with_suffix('.json')
        write_json_atomic(manifest_path, manifest)
        return {'manifest': manifest, 'manifest_path': manifest_path, 'npz_path': npz_path}
