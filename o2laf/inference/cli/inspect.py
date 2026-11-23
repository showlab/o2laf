"""Summarise a released model without loading any weights.

    o2laf-inspect <repo id or dir> [--verify-action-space]
"""

from __future__ import annotations

import argparse

from o2laf.inference import hub


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help='Hub model (e.g. YuxinJ/O2LaF/o2laf_dit_mixed) or local model directory.')
    parser.add_argument('--verify-action-space', action='store_true',
                        help="Re-hash the action encoder's weights and check them against the recorded action space.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Metadata only; the action encoder's weights only when asked to verify them.
    root = hub.resolve_model_dir(args.model, fetch='action_encoder' if args.verify_action_space else 'metadata')
    index = hub.read_model_index(root)
    print(f'model: {root}')
    for key in ('base_model', 'action_space_id', 'fixed_prompt', 'action_dim', 'codes_per_transition', 'fps_values'):
        if key in index:
            print(f'{key}: {index[key]}')
    encoder = hub.load_part_config(root, index['action_encoder'])
    print(f'action encoder: {encoder["action_encoder"]}, step {encoder["global_step"]} '
          f'(in {index["action_encoder"]["component"]}/)')
    for name, mode in index['modes'].items():
        location = mode['world_model']
        step = hub.load_part_config(root, location)['global_step']
        print(f'mode {name}: {mode["num_rgb_frames"]} frames at {mode["height"]}x{mode["width"]}, {mode["fps"]} fps, '
              f'null action {mode["null_action"]}, world model step {step} (in {location["component"]}/)')
    library = index.get('action_library')
    if library:
        print(f'action libraries: {", ".join(library["libraries"])}')
    if args.verify_action_space:
        actual = 'sha256:' + hub.tensor_content_sha256(hub.load_part_state(root, index['action_encoder']))
        status = 'ok' if actual == index['action_space_id'] else f'MISMATCH ({actual})'
        print(f'action space check: {status}')
        if actual != index['action_space_id']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
