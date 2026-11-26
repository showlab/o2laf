"""Cache causal Wan 2.2 VAE latents for all 41-frame windows in a JSONL index.

One shared latent cache serves all four motion-annotation modes:
- ``wan_base_lora`` and ``ssl_lora`` fine-tune directly on these cached latents;
- ``ssl_action_mu`` and ``ssl_pre_gaussian`` extract frozen features from them.

Supports single-GPU or multi-GPU (``torchrun``) execution and resumes safely.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan

from motion_annotation.checkpointing import write_json_atomic
from motion_annotation.data import read_rows, safe_video_filename
from motion_annotation.models import DEFAULT_BASE_MODEL, valid_wan_latent_archive
from motion_annotation.sampling import FrameSource, build_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Cache Wan 2.2 VAE latents for a video JSONL index.')
    parser.add_argument('--video-index', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--base-model',
        default=DEFAULT_BASE_MODEL,
        help='Wan 2.2 TI2V-5B Hub repo id or local path (default: Wan-AI/Wan2.2-TI2V-5B-Diffusers).',
    )
    parser.add_argument('--splits', nargs='+', help='Optional subset of splits to cache (e.g. train val test).')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--model-fps', type=float, default=16.0)
    parser.add_argument('--num-rgb-frames', type=int, default=41)
    parser.add_argument('--stride-model-frames', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=2, help='41-frame windows encoded per VAE forward pass.')
    parser.add_argument('--limit-videos', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    return parser.parse_args()


def preprocess_frames(frames: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Convert uint8 ``[B, T, H, W, 3]`` frames to normalized ``[B, 3, T, height, width]`` in ``[-1, 1]``."""
    batch, count = frames.shape[:2]
    pixels = frames.flatten(0, 1).float().permute(0, 3, 1, 2).div_(255.0)
    pixels = F.interpolate(pixels, size=(height, width), mode='bilinear', align_corners=False, antialias=True)
    pixels = pixels.sub_(0.5).div_(0.5)
    return pixels.unflatten(0, (batch, count)).permute(0, 2, 1, 3, 4).contiguous()


@torch.inference_mode()
def encode_latents(
    frames: torch.Tensor,
    vae: AutoencoderKLWan,
    device: torch.device,
    *,
    height: int,
    width: int,
    expected_latent_frames: int,
) -> torch.Tensor:
    video = preprocess_frames(frames, height=height, width=width).to(device=device, dtype=torch.bfloat16)
    moments = vae._encode(video)
    mean, _ = torch.chunk(moments, 2, dim=1)
    latent_mean = torch.as_tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latent_std = torch.as_tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latents = ((mean.float() - latent_mean) / latent_std).to(torch.bfloat16)
    if int(latents.shape[2]) != int(expected_latent_frames):
        raise RuntimeError(
            f'Wan VAE returned {latents.shape[2]} latent frames, expected {expected_latent_frames}.'
        )
    return latents.cpu()


def archive_path(root: Path, row: dict) -> Path:
    return root / str(row.get('split', 'unsplit')) / f'{safe_video_filename(row["video_id"])}.pt'


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Wan VAE latent caching.')

    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device('cuda', local_rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=12))

    temporal_compression = 4
    if (args.num_rgb_frames - 1) % temporal_compression != 0:
        raise ValueError(f'--num-rgb-frames ({args.num_rgb_frames}) must satisfy (F - 1) % 4 == 0.')
    expected_latent_frames = 1 + (args.num_rgb_frames - 1) // temporal_compression

    rows = read_rows(args.video_index, set(args.splits) if args.splits else None)
    if args.limit_videos is not None:
        rows = rows[:args.limit_videos]
    if not rows:
        raise ValueError('No videos selected from --video-index.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json_atomic(
            args.output_dir / 'provenance.json',
            {
                'version': 'o2laf_wan_latents_v1',
                'video_index': str(args.video_index.resolve()),
                'base_model': str(args.base_model),
                'dtype': 'bfloat16',
                'height': int(args.height),
                'width': int(args.width),
                'model_fps': float(args.model_fps),
                'num_rgb_frames': int(args.num_rgb_frames),
                'stride_model_frames': int(args.stride_model_frames),
                'vae_temporal_compression': temporal_compression,
            },
        )
    if world_size > 1:
        dist.barrier()

    vae = AutoencoderKLWan.from_pretrained(
        args.base_model,
        subfolder='vae',
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    z_dim = int(vae.config.z_dim)
    spatial_scale = int(getattr(vae.config, 'scale_factor_spatial', 16))
    expected_shape = (z_dim, expected_latent_frames, args.height // spatial_scale, args.width // spatial_scale)

    shard = rows[rank::world_size]
    for index, row in enumerate(shard, 1):
        target = archive_path(args.output_dir, row)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix('.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.is_file() and not args.overwrite and valid_wan_latent_archive(target, row, expected_shape):
                if index % 50 == 0 or index == len(shard):
                    print(f'[rank {rank}/{world_size}] skip valid ({index}/{len(shard)})', flush=True)
                continue

            with FrameSource(row['video_path']) as reader:
                source_fps = float(reader.fps)
                source_num_frames = len(reader)
                effective_fps = min(float(args.model_fps), source_fps)
                windows = build_windows(
                    num_source_frames=source_num_frames,
                    source_fps=source_fps,
                    model_fps=effective_fps,
                    num_model_frames=int(args.num_rgb_frames),
                    temporal_compression=temporal_compression,
                    stride_model_frames=int(args.stride_model_frames),
                )
                latent_batches = []
                for start in range(0, len(windows), args.batch_size):
                    selected = windows[start:start + args.batch_size]
                    frame_batch = torch.from_numpy(
                        np.stack([reader.get_batch(window.frame_indices) for window in selected])
                    )
                    latent_batches.append(
                        encode_latents(
                            frame_batch,
                            vae,
                            device,
                            height=args.height,
                            width=args.width,
                            expected_latent_frames=expected_latent_frames,
                        )
                    )

            payload = {
                'version': 'o2laf_wan_latent_video_v1',
                'video_id': str(row['video_id']),
                'split': str(row.get('split', 'unsplit')),
                'latents': torch.cat(latent_batches, dim=0),
                'valid_transition_lengths': torch.tensor(
                    [len(window.transition_starts) for window in windows], dtype=torch.int64
                ),
                'sampled_frame_indices': torch.from_numpy(
                    np.stack([window.frame_indices for window in windows])
                ),
                'transition_starts': [torch.from_numpy(window.transition_starts) for window in windows],
                'transition_ends': [torch.from_numpy(window.transition_ends) for window in windows],
                'source_fps': source_fps,
                'source_num_frames': source_num_frames,
                'effective_model_fps': effective_fps,
            }
            tmp = target.with_suffix('.tmp.pt')
            torch.save(payload, tmp)
            tmp.replace(target)

        if index % 20 == 0 or index == len(shard):
            print(f'[rank {rank}/{world_size}] cached {index}/{len(shard)}: {row["video_id"]}', flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({'cached_videos': len(rows), 'output_dir': str(args.output_dir.resolve())}, indent=2))


if __name__ == '__main__':
    main()
