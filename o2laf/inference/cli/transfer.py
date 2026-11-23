"""Zero-shot action transfer: apply the actions of a source video to a new first frame.

    o2laf-transfer --model <repo id or dir> --source-video source.mp4 --first-frame frame.png --output out.mp4

The source video is resampled to the model's frame rate and used whole, up to
its first 121 frames (7.5 s at 16 fps). The output has as many frames as the
source, rounded down to 1 + 4k.

``--control`` selects what drives the generation:

* ``transfer``: the source video's actions, from the given first frame (default);
* ``self``: the source video's actions from its own first frame (reconstruction);
* ``zero``: no action at all;
* ``shuffle``: the source video's actions in a shuffled temporal order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from o2laf.inference.cli.common import (
    add_generation_arguments,
    add_model_arguments,
    guidance_from_args,
    guidance_manifest,
    load_first_frame,
    load_pipeline,
    load_source_video,
    model_manifest,
    require_device,
    save_video_and_manifest,
)
from o2laf.utils.action_util import shuffle_action_groups

CONTROL_MODES = ('transfer', 'self', 'zero', 'shuffle')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument('--mode', choices=('causal', 'bidirectional'), default='bidirectional')
    parser.add_argument('--source-video', type=Path, required=True, help='The video whose actions are transferred.')
    parser.add_argument('--control', choices=CONTROL_MODES, default='transfer')
    add_generation_arguments(parser, required=False)
    return validate_args(parser, parser.parse_args(argv))


def validate_args(parser, args):
    if args.output is None:
        parser.error('--output is required.')
    # `self` reconstructs the source video from its own first frame; every
    # other control drives a new first frame.
    if args.control != 'self' and args.first_frame is None:
        parser.error(f'--first-frame is required for --control {args.control}.')
    if args.control == 'self' and args.first_frame is not None:
        parser.error('--control self uses the source video\'s own first frame; omit --first-frame.')
    if args.control == 'zero':
        if args.cfg_scale not in (None, 1.0):
            parser.error('--control zero has no actions to guide; omit --cfg-scale.')
        args.cfg_scale = 1.0
    return args


@torch.no_grad()
def main():
    args = parse_args()
    require_device(args.device)
    pipeline = load_pipeline(args, args.mode)
    source_frames = load_source_video(args.source_video, pipeline.fps)
    actions = pipeline.encode_source_actions(source_frames)
    num_latent_frames = pipeline.num_latent_frames_for_actions(actions)
    print(f'{len(source_frames)} source frames -> {pipeline.num_rgb_frames_for(num_latent_frames)} output frames')

    first_frame = source_frames[:1] if args.control == 'self' else load_first_frame(args.first_frame, pipeline.fps)
    shuffle_order = None
    if args.control == 'zero':
        actions = pipeline.no_action_condition(actions)
    elif args.control == 'shuffle':
        actions, shuffle_order = shuffle_action_groups(actions, pipeline.action_compression_ratio, args.seed)

    guidance = guidance_from_args(args)
    first_frame_latents = pipeline.encode_latents(pipeline.preprocess_frames(first_frame))
    latents = pipeline.generate_latents(
        actions=actions, first_frame_latents=first_frame_latents, num_latent_frames=num_latent_frames,
        num_steps=args.num_steps, seed=args.seed, guidance=guidance,
    )
    save_video_and_manifest(
        pipeline,
        latents,
        args.output,
        {
            'schema_version': 'o2laf_zero_shot_transfer_v1',
            **model_manifest(pipeline),
            'source_video': str(args.source_video.resolve()),
            'source_video_frames': len(source_frames),
            'first_frame': str(args.first_frame.resolve()) if args.first_frame is not None else None,
            'control': args.control,
            'shuffle_order': shuffle_order,
            **guidance_manifest(guidance, args),
        },
    )


if __name__ == '__main__':
    main()
