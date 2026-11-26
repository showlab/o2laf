"""Visualise raw pairwise action-query attention of a Wan DiT latent action encoder.

    o2laf-attention --model <repo id or dir> --video video.mp4 --output out.mp4

Only the latent action encoder and Wan VAE are loaded (the world model is not
downloaded). By default, block 30 (1-indexed) and head 11 (0-indexed) are
visualised side-by-side over source and target transition frames using one
clip-global colour scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from o2laf.inference.attention import (
    DEFAULT_ALPHA,
    DEFAULT_ALPHA_POWER,
    DEFAULT_BLOCK,
    DEFAULT_GAMMA,
    DEFAULT_HEAD,
    DEFAULT_PERCENTILE,
    DEFAULT_WINDOW_FRAMES,
    ActionAttentionExtractor,
    AttentionRenderConfig,
)
from o2laf.inference.cli.common import require_device, torch_dtype


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='Hub model (e.g. YuxinJ/O2LaF/o2laf_dit_mixed) or local model directory.')
    parser.add_argument('--base-model', help='Wan 2.2 TI2V-5B base (repo id or path); defaults to the one the model records.')
    parser.add_argument('--video', type=Path, required=True, help='Input video whose latent action attention is visualised.')
    parser.add_argument('--output', type=Path, required=True, help='Output .mp4; a .json manifest is written next to it.')
    parser.add_argument('--block', type=int, default=DEFAULT_BLOCK, help=f'1-indexed Wan DiT block to inspect (default {DEFAULT_BLOCK}).')
    parser.add_argument('--head', type=int, default=DEFAULT_HEAD, help=f'0-indexed attention head to inspect (default {DEFAULT_HEAD}).')
    parser.add_argument('--fps', type=float, default=None, help='Resampling FPS before VAE encoding; defaults to the model FPS.')
    parser.add_argument('--output-fps', type=float, default=None, help='Output video FPS; defaults to fps / 4 (latent transition cadence).')
    parser.add_argument(
        '--window-frames',
        type=int,
        default=DEFAULT_WINDOW_FRAMES,
        help=f'RGB frames per adjacent encoder window (must be 4k+1; default {DEFAULT_WINDOW_FRAMES}).',
    )
    parser.add_argument('--percentile', type=float, default=DEFAULT_PERCENTILE, help=f'Clip-global probability percentile (default {DEFAULT_PERCENTILE:g}).')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA, help=f'Heatmap gamma exponent (default {DEFAULT_GAMMA:g}).')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help=f'Maximum overlay opacity (default {DEFAULT_ALPHA:g}).')
    parser.add_argument('--alpha-power', type=float, default=DEFAULT_ALPHA_POWER, help=f'Opacity curve exponent (default {DEFAULT_ALPHA_POWER:g}).')
    parser.add_argument('--save-npz', type=Path, default=None, help='Optional .npz path to save raw [T, 2, H, W] attention probabilities.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument(
        '--verify-action-space',
        action='store_true',
        help="Re-hash the action encoder's weights and check them against the recorded action space.",
    )
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    require_device(args.device)
    render_config = AttentionRenderConfig(
        percentile=args.percentile,
        gamma=args.gamma,
        alpha=args.alpha,
        alpha_power=args.alpha_power,
    )
    extractor = ActionAttentionExtractor.from_pretrained(
        args.model,
        base_model=args.base_model,
        device=args.device,
        dtype=torch_dtype(args.dtype),
        verify_action_space=args.verify_action_space,
    )
    written = extractor.visualize(
        args.video,
        args.output,
        block=args.block,
        head=args.head,
        target_fps=args.fps,
        output_fps=args.output_fps,
        window_frames=args.window_frames,
        render_config=render_config,
        save_npz=args.save_npz,
    )
    manifest = written['manifest']
    print(
        f'block {manifest["block"]:02d} head {manifest["head"]:02d}: '
        f'{manifest["num_transitions"]} transitions '
        f'(source mass {manifest["source_mass_mean"]:.1%}, target mass {manifest["target_mass_mean"]:.1%})'
    )
    print(f'video: {args.output}')
    print(f'manifest: {written["manifest_path"]}')
    if written['npz_path'] is not None:
        print(f'npz: {written["npz_path"]}')


if __name__ == '__main__':
    main()
