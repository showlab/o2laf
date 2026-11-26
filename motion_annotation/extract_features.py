"""Extract frozen O²LaF Wan DiT features from cached Wan VAE latents.

Extracts 32-D posterior-mean ``action_mu`` (for mode ``ssl_action_mu``) and/or
3072-D pre-Gaussian ``transition_features`` (for mode ``ssl_pre_gaussian``) in
one forward pass per video.
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

from motion_annotation.checkpointing import write_json_atomic
from motion_annotation.data import read_rows, safe_video_filename
from motion_annotation.models import build_wan_action_encoder, wan_forward_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract frozen O²LaF Wan DiT features from cached latents.')
    parser.add_argument(
        '--model',
        required=True,
        help='O²LaF Wan DiT model (Hub id e.g. YuxinJ/O2LaF/o2laf_dit_mixed or local directory).',
    )
    parser.add_argument(
        '--base-model',
        help='Optional local copy of Wan-AI/Wan2.2-TI2V-5B-Diffusers (defaults to model_index.json base_model).',
    )
    parser.add_argument('--video-index', type=Path, required=True)
    parser.add_argument('--latent-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--feature-kind',
        choices=('both', 'action_mu', 'transition_features'),
        default='both',
        help='Which representations to save in each per-video .npz archive (default: both).',
    )
    parser.add_argument('--splits', nargs='+')
    parser.add_argument('--batch-size', type=int, default=4, help='41-frame windows per encoder forward pass.')
    parser.add_argument('--limit-videos', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    return parser.parse_args()


def feature_path(root: Path, row: dict) -> Path:
    return root / str(row.get('split', 'unsplit')) / f'{safe_video_filename(row["video_id"])}.npz'


def latent_path(root: Path, row: dict) -> Path:
    safe = safe_video_filename(row['video_id'])
    candidate = root / str(row.get('split', 'unsplit')) / f'{safe}.pt'
    if candidate.is_file():
        return candidate
    for match in root.glob(f'*/{safe}.pt'):
        return match
    return candidate


def valid_feature_archive(path: Path, row: dict, required_keys: tuple[str, ...]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data['video_id'].item()) != str(row['video_id']):
                return False
            starts = data['transition_starts']
            ends = data['transition_ends']
            if starts.shape != ends.shape or starts.ndim != 1 or not len(starts):
                return False
            for key in required_keys:
                values = data[key]
                if values.ndim != 2 or len(values) != len(starts) or not np.isfinite(values).all():
                    return False
    except (OSError, KeyError, ValueError):
        return False
    return True


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Wan DiT feature extraction.')

    from o2laf.inference.prompts import fixed_prompt_embeds

    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device('cuda', local_rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=12))

    required_keys = (
        ('action_mu', 'transition_features')
        if args.feature_kind == 'both'
        else (args.feature_kind,)
    )
    rows = read_rows(args.video_index, set(args.splits) if args.splits else None)
    if args.limit_videos is not None:
        rows = rows[:args.limit_videos]
    if not rows:
        raise ValueError('No videos selected from --video-index.')

    encoder, metadata = build_wan_action_encoder(
        'ssl_pre_gaussian',
        model=args.model,
        base_model=args.base_model,
        device=device,
        base_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    prompt_embeds = fixed_prompt_embeds(
        base_model=metadata['base_model_id'],
        prompt=metadata['fixed_prompt'],
        base_model_path=metadata['base_model_path'],
        device=device,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json_atomic(
            args.output_dir / 'provenance.json',
            {
                'version': 'o2laf_frozen_features_v1',
                'model': str(args.model),
                'action_space_id': metadata['action_space_id'],
                'base_model_id': metadata['base_model_id'],
                'video_index': str(args.video_index.resolve()),
                'latent_cache': str(args.latent_cache.resolve()),
                'feature_kind': args.feature_kind,
                'keys': list(required_keys),
                'action_mu_dim': int(metadata['latent_dim']),
                'transition_feature_dim': int(metadata['feature_dim']),
            },
        )
    if world_size > 1:
        dist.barrier()

    shard = rows[rank::world_size]
    with torch.inference_mode():
        for index, row in enumerate(shard, 1):
            target = feature_path(args.output_dir, row)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.with_suffix('.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if target.is_file() and not args.overwrite and valid_feature_archive(target, row, required_keys):
                    continue

                archive = latent_path(args.latent_cache, row)
                if not archive.is_file():
                    raise FileNotFoundError(f'Missing latent archive: {archive}. Run cache_latents.py first.')
                payload = torch.load(archive, map_location='cpu', weights_only=True, mmap=True)
                latents = payload['latents']
                lengths = [int(v) for v in payload['valid_transition_lengths']]

                mu_segments = []
                trans_segments = []
                for start in range(0, len(latents), args.batch_size):
                    batch = latents[start:start + args.batch_size].to(device=device, dtype=torch.bfloat16)
                    timestep = torch.zeros(batch.shape[0], batch.shape[2], device=device, dtype=torch.float32)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        outputs = wan_forward_features(
                            encoder,
                            hidden_states=batch,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds.expand(batch.shape[0], -1, -1),
                            return_action_mu=True,
                        )
                    for offset in range(len(batch)):
                        length = lengths[start + offset]
                        if 'action_mu' in required_keys:
                            mu_segments.append(outputs['action_mu'][offset, :length].float().cpu().numpy())
                        if 'transition_features' in required_keys:
                            trans_segments.append(outputs['transition_features'][offset, :length].float().cpu().numpy())

                starts_np = np.concatenate([t.numpy() for t in payload['transition_starts']], axis=0)
                ends_np = np.concatenate([t.numpy() for t in payload['transition_ends']], axis=0)
                arrays: dict[str, np.ndarray] = {
                    'video_id': np.asarray(str(row['video_id'])),
                    'split': np.asarray(str(row.get('split', 'unsplit'))),
                    'transition_starts': starts_np,
                    'transition_ends': ends_np,
                }
                if 'action_mu' in required_keys:
                    arrays['action_mu'] = np.concatenate(mu_segments, axis=0).astype(np.float32)
                if 'transition_features' in required_keys:
                    arrays['transition_features'] = np.concatenate(trans_segments, axis=0).astype(np.float32)

                tmp = target.with_suffix('.tmp.npz')
                np.savez(tmp, **arrays)
                tmp.replace(target)

            if index % 50 == 0 or index == len(shard):
                print(f'[rank {rank}/{world_size}] extracted {index}/{len(shard)}: {row["video_id"]}', flush=True)
            if index % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({'extracted_videos': len(rows), 'output_dir': str(args.output_dir.resolve())}, indent=2))


if __name__ == '__main__':
    main()
