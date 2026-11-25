"""Argument parsing, video I/O and manifests shared by the inference commands."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from o2laf.inference.sampling import MAX_RGB_FRAMES, UNCONDITIONAL_MODES, ActionGuidance
from o2laf.utils.file_util import write_json_atomic
from o2laf.utils.video_util import VideoLoader, select_frame_indices

VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.webm'}
DEFAULT_CFG_SCALE = 3.0


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model', required=True, help='Hub model (e.g. YuxinJ/O2LaF/o2laf_dit_mixed) or local model directory.')
    parser.add_argument('--base-model', help='Wan 2.2 TI2V-5B base (repo id or path); defaults to the one the model records.')
    parser.add_argument('--base-action-encoder',
                        help='Frozen action-encoder trunk checkpoint, for encoders that need one (e.g. V-JEPA 2 vitg-384.pt).')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--verify-action-space', action='store_true',
                        help="Re-hash the action encoder's weights and check them against the recorded action space.")


def load_pipeline(args, mode: str):
    from o2laf.inference.pipeline import LatentActionPipeline

    return LatentActionPipeline.from_pretrained(
        args.model, mode=mode, base_model=args.base_model, base_action_encoder=args.base_action_encoder,
        device=args.device, dtype=torch_dtype(args.dtype), verify_action_space=args.verify_action_space,
    )


def add_generation_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument('--first-frame', type=Path, required=required,
                        help='The frame the video starts from: an image, or a video (its frame 0).')
    parser.add_argument('--output', type=Path, required=required, help='Output .mp4; a .json manifest is written next to it.')
    parser.add_argument('--cfg-scale', type=float, default=None,
                        help=f'Action guidance scale (default {DEFAULT_CFG_SCALE:g}); 1 disables guidance.')
    parser.add_argument('--unconditional-mode', choices=UNCONDITIONAL_MODES, default='full')
    parser.add_argument('--guidance-sigma-min', type=float)
    parser.add_argument('--variant-batch-size', type=int, default=4)
    parser.add_argument('--num-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)


def torch_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == 'bf16' else torch.float32


def require_device(device: str) -> None:
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')


def guidance_from_args(args) -> ActionGuidance:
    return ActionGuidance(
        scale=DEFAULT_CFG_SCALE if args.cfg_scale is None else args.cfg_scale,
        unconditional_mode=args.unconditional_mode,
        guidance_sigma_min=args.guidance_sigma_min,
        variant_batch_size=args.variant_batch_size,
    )


def load_video(path: Path, num_frames: int, fps: int) -> np.ndarray:
    """The first ``num_frames`` frames of ``path`` resampled to ``fps``, as uint8 ``[T,H,W,3]``."""
    with VideoLoader(str(path), name=str(path)) as loader:
        frames = loader.get_clip(num_frames, target_fps=fps, random_sample=False)
    if frames.shape[0] != num_frames:
        raise RuntimeError(f'{path} returned {frames.shape[0]} frames, expected {num_frames}.')
    return np.asarray(frames, dtype=np.uint8)


def load_source_video(path: Path, fps: int, max_frames: int = MAX_RGB_FRAMES) -> np.ndarray:
    """The whole of ``path`` resampled to ``fps``, up to its first ``max_frames`` frames, as uint8 ``[T,H,W,3]``."""
    with VideoLoader(str(path), name=str(path)) as loader:
        available = len(select_frame_indices(loader.get_frame_count(), loader.get_fps(), fps))
        num_frames = min(available, int(max_frames))
        if available > num_frames:
            print(f'{path}: using the first {num_frames} of its {available} frames at {fps} fps.')
        frames = loader.get_clip(num_frames, target_fps=fps, random_sample=False)
    return np.asarray(frames, dtype=np.uint8)


def load_first_frame(path: Path, fps: int) -> np.ndarray:
    """One RGB frame ``[1,H,W,3]`` from an image, or frame 0 of a video."""
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return load_video(path, 1, fps)
    with Image.open(path) as image:
        frame = np.asarray(image.convert('RGB'), dtype=np.uint8)
    return frame[None]


def save_video_and_manifest(pipeline, latents: torch.Tensor, output: Path, manifest: dict) -> None:
    """Decode ``latents``, write ``output`` and its JSON manifest, and print both paths."""
    generated = pipeline.decode_latents(latents)[0]
    # export_to_video takes float frames in [0, 1] and scales them itself. Frames
    # already scaled to uint8 would be multiplied by 255 again and wrap modulo
    # 256 into a colour negative -- which is what this function once wrote.
    frames = generated.permute(0, 2, 3, 1).float().clamp(0, 1).cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(list(frames), str(output), fps=pipeline.fps)
    manifest = {**manifest, 'fps': pipeline.fps, 'num_rgb_frames': int(frames.shape[0]), 'output': str(output.resolve())}
    manifest_path = output.with_suffix('.json')
    write_json_atomic(manifest_path, manifest)
    print(f'video: {output}')
    print(f'manifest: {manifest_path}')


def model_manifest(pipeline) -> dict:
    """The fields that identify which model produced an output."""
    return {
        'model': str(pipeline.model_dir),
        'mode': pipeline.mode,
        'null_action': pipeline.null_action,
        'world_model_step': pipeline.world_model_step,
        'action_space_id': pipeline.action_space_id,
    }


def guidance_manifest(guidance: ActionGuidance, args) -> dict:
    return {
        'cfg_scale': guidance.scale,
        'unconditional_mode': guidance.unconditional_mode,
        'guidance_sigma_min': guidance.guidance_sigma_min,
        'num_steps': args.num_steps,
        'seed': args.seed,
    }
