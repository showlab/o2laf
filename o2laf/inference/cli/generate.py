"""Generate from one named action of the model's atomic action library and a first frame.

    o2laf-generate --model <repo id or dir> --action <name> --first-frame frame.png --output out.mp4

The video is 81 frames (5 s at 16 fps) unless ``--num-frames`` asks for another
length: ``1 + 4k`` frames, at most 121.

``--list`` prints the shipped libraries and their actions instead, without
downloading or loading any weights. ``--library`` picks a library when the model
ships several; the default is the first one its index lists.
"""

from __future__ import annotations

import argparse

import torch

from o2laf.inference import hub
from o2laf.inference.action_library import AtomicActionLibrary, library_names
from o2laf.inference.cli.common import (
    add_generation_arguments,
    add_model_arguments,
    guidance_from_args,
    guidance_manifest,
    load_first_frame,
    load_pipeline,
    model_manifest,
    require_device,
    save_video_and_manifest,
)
from o2laf.inference.sampling import MAX_RGB_FRAMES


DEFAULT_NUM_FRAMES = 81


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument('--library', help='Which shipped library; defaults to the first the model lists.')
    parser.add_argument('--action', help='Name of the library action to apply.')
    parser.add_argument('--list', action='store_true', help='List the shipped libraries and actions, then exit.')
    parser.add_argument('--num-frames', type=int, default=DEFAULT_NUM_FRAMES,
                        help=f'Frames to generate: 1 + 4k, at most {MAX_RGB_FRAMES} (default {DEFAULT_NUM_FRAMES}).')
    add_generation_arguments(parser, required=False)
    args = parser.parse_args(argv)
    if not args.list:
        missing = [flag for flag, value in (('--action', args.action), ('--first-frame', args.first_frame),
                                            ('--output', args.output)) if value is None]
        if missing:
            parser.error(f'the following arguments are required unless --list is given: {", ".join(missing)}')
    if args.num_frames < 5 or (args.num_frames - 1) % 4 or args.num_frames > MAX_RGB_FRAMES:
        parser.error(f'--num-frames must be 1 + 4k between 5 and {MAX_RGB_FRAMES}, got {args.num_frames}.')
    return args


def list_actions(args) -> None:
    root = hub.resolve_model_dir(args.model, fetch='metadata')
    index = hub.read_model_index(root)
    names = library_names(index)
    if not names:
        print(f'{args.model} ships no action library.')
        return
    for position, name in enumerate(names):
        library = AtomicActionLibrary.from_model_dir(root, index, name)
        default = ' (default)' if position == 0 else ''
        print(f'{name}{default}: {" ".join(library.names())}')


@torch.no_grad()
def main():
    args = parse_args()
    if args.list:
        list_actions(args)
        return
    require_device(args.device)
    pipeline = load_pipeline(args, 'bidirectional')
    library = AtomicActionLibrary.from_pipeline(pipeline, args.library)
    library.get(args.action)  # fail on an unknown name before generating anything
    print(f'library {library.name}, action {args.action}')
    actions = library.tensor(
        args.action,
        groups=pipeline.num_latent_frames_for(args.num_frames) - pipeline.num_context_frames,
        device=pipeline.device,
        dtype=pipeline.dtype,
    )
    first_frame = pipeline.encode_latents(pipeline.preprocess_frames(load_first_frame(args.first_frame, pipeline.fps)))
    guidance = guidance_from_args(args)
    latents = pipeline.generate_latents(
        actions=actions, first_frame_latents=first_frame, num_steps=args.num_steps, seed=args.seed, guidance=guidance
    )
    action = library.get(args.action)
    save_video_and_manifest(
        pipeline,
        latents,
        args.output,
        {
            'schema_version': 'o2laf_atomic_action_rollout_v1',
            **model_manifest(pipeline),
            'library': library.name,
            'action': action.name,
            'action_support': action.support,
            'first_frame': str(args.first_frame.resolve()),
            **guidance_manifest(guidance, args),
        },
    )


if __name__ == '__main__':
    main()
