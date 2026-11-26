"""Prepare leak-audited CameraBench train/val/test JSONL indexes.

CameraBench public test clips (GIFs) and official 15-class binary task JSONLs
are publicly distributed by the CameraBench authors. The supervised training
archive (``camerabench_train.zip`` + MP4 videos) is request-controlled by the
CameraBench authors and is not redistributed here.

This script builds deterministic, source-group-disjoint train/val splits
(1,114 train / 197 validation labeled clips by default) and/or the official
969-clip public test index in the unified JSONL schema used by all four
adaptation modes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import av

from motion_annotation.checkpointing import write_json_atomic
from motion_annotation.data import (
    CAMERABENCH_CLASSES,
    choose_grouped_split,
    extract_sparse_labels,
    read_official_questions,
    source_group,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare CameraBench train/val/test JSONL indexes.')
    parser.add_argument(
        '--task-dir',
        type=Path,
        required=True,
        help='Directory containing the 15 official CameraBench binary_classification/*.jsonl files.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for video_index.jsonl, split_manifest.json, and summary.json.',
    )
    parser.add_argument(
        '--annotations-zip',
        type=Path,
        help='Path to camerabench_train.zip (request-controlled from CameraBench authors).',
    )
    parser.add_argument(
        '--train-videos',
        type=Path,
        help='Directory containing extracted CameraBench training MP4 clips.',
    )
    parser.add_argument(
        '--test-videos',
        type=Path,
        help='Directory containing CameraBench public test GIF (or MP4) clips.',
    )
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--split-seed', type=int, default=20261026)
    parser.add_argument('--workers', type=int, default=16)
    return parser.parse_args()


def probe_video(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = next(candidate for candidate in container.streams if candidate.type == 'video')
        fps = float(stream.average_rate or stream.guessed_rate or stream.base_rate or (100.0 / 7.0))
        frames = int(stream.frames or 0)
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        if not frames:
            frames = sum(1 for _ in container.decode(stream))
        if duration is None:
            duration = frames / fps
        return {
            'num_frames': frames,
            'nominal_fps': fps,
            'nominal_duration_sec': duration,
            'width': int(stream.width),
            'height': int(stream.height),
            'codec': str(stream.codec_context.name),
        }


def build_test_rows(task_dir: Path, test_videos: Path) -> list[dict]:
    """Build the 969-clip official CameraBench public test index from task JSONLs."""
    labels_by_clip: dict[str, dict[str, str]] = defaultdict(dict)
    mp4_by_id: dict[str, str] = {}
    for task_path in sorted(task_dir.glob('*.jsonl')):
        task = task_path.stem
        if task not in CAMERABENCH_CLASSES:
            raise ValueError(f'Unexpected CameraBench task file: {task_path}.')
        for line in task_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            mp4_name = PurePosixPath(str(record['image'])).name
            video_id = PurePosixPath(mp4_name).stem
            answer = str(record['label']).strip().title()
            if answer not in ('Yes', 'No'):
                raise ValueError(f'Unexpected test label {answer!r} in {task_path}.')
            prev = labels_by_clip[video_id].get(task)
            if prev is not None and prev != answer:
                raise ValueError(f'Conflicting test label for {video_id} / {task}: {prev} vs {answer}.')
            labels_by_clip[video_id][task] = answer
            mp4_by_id[video_id] = mp4_name

    rows = []
    missing = []
    for video_id in sorted(labels_by_clip):
        gif_path = test_videos / f'{video_id}.gif'
        mp4_path = test_videos / mp4_by_id[video_id]
        if gif_path.is_file():
            video_path = gif_path
        elif mp4_path.is_file():
            video_path = mp4_path
        else:
            missing.append(video_id)
            continue
        rows.append(
            {
                'video_id': video_id,
                'mp4_name': mp4_by_id[video_id],
                'video_path': str(video_path.resolve()),
                'source_group': source_group(mp4_by_id[video_id]),
                'split': 'test',
                'labels': dict(sorted(labels_by_clip[video_id].items())),
            }
        )
    if missing:
        raise FileNotFoundError(
            f'Missing {len(missing)} CameraBench test videos under {test_videos}, e.g. {missing[:5]}.'
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.annotations_zip is None and args.test_videos is None:
        raise ValueError('Provide --annotations-zip + --train-videos and/or --test-videos.')
    if (args.annotations_zip is None) != (args.train_videos is None):
        raise ValueError('--annotations-zip and --train-videos must be provided together.')

    question_to_task = read_official_questions(args.task_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_rows: list[dict] = []
    if args.test_videos is not None:
        test_rows = build_test_rows(args.task_dir, args.test_videos)
        write_jsonl(args.output_dir / 'video_index_test.jsonl', test_rows)

    labeled_train_val_rows: list[dict] = []
    split_manifest = None
    label_summary = None
    exact_overlap: list[str] = []
    source_overlap: list[str] = []

    if args.annotations_zip is not None:
        with zipfile.ZipFile(args.annotations_zip) as archive:
            balanced = json.load(archive.open('cam_motion/balanced_vqa.json'))
            imbalanced = json.load(archive.open('cam_motion/imb_raw.json'))
            captions = json.load(archive.open('cam_motion/captionset.json'))

        labels, label_summary = extract_sparse_labels(
            (('imb_raw', imbalanced), ('balanced_vqa', balanced)),
            question_to_task,
        )
        caption_by_video: dict[str, str] = {}
        for row in captions:
            video = PurePosixPath(str(row['videos'][0])).name
            caption = str(row['messages'][1]['content']).strip()
            previous = caption_by_video.setdefault(video, caption)
            if previous != caption:
                raise ValueError(f'Inconsistent captions for {video}.')

        video_names = sorted(caption_by_video)
        paths = {name: args.train_videos / name for name in video_names}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'Missing {len(missing)} training videos: {missing[:5]}.')

        train_videos, val_videos, split_manifest = choose_grouped_split(
            labels,
            val_fraction=args.val_fraction,
            seed=args.split_seed,
        )
        if train_videos & val_videos:
            raise RuntimeError('Train/validation clip overlap detected.')
        if {source_group(v) for v in train_videos} & {source_group(v) for v in val_videos}:
            raise RuntimeError('Source-group leakage across train and validation splits.')

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            metadata = dict(zip(video_names, pool.map(probe_video, paths.values())))

        if test_rows:
            test_names = {str(row['mp4_name']) for row in test_rows}
            test_groups = {source_group(name) for name in test_names}
            exact_overlap = sorted(set(video_names) & test_names)
            source_overlap = sorted({source_group(name) for name in video_names} & test_groups)
            if exact_overlap:
                raise RuntimeError(f'Official train/test clips overlap: {exact_overlap[:5]}.')

        all_train_val_rows = []
        for name in video_names:
            split_name = 'val' if name in val_videos else 'train'
            row = {
                'video_id': Path(name).stem,
                'mp4_name': name,
                'video_path': str(paths[name].resolve()),
                'source_group': source_group(name),
                'split': split_name,
                'labels': labels.get(name, {}),
                'caption': caption_by_video[name],
                **metadata[name],
            }
            all_train_val_rows.append(row)
            if name in labels:
                labeled_train_val_rows.append(row)

        write_jsonl(args.output_dir / 'video_index_train_val_all.jsonl', all_train_val_rows)
        write_jsonl(args.output_dir / 'video_index_train_val.jsonl', labeled_train_val_rows)
        write_json_atomic(args.output_dir / 'split_manifest.json', split_manifest)

    combined = labeled_train_val_rows + test_rows
    write_jsonl(args.output_dir / 'video_index.jsonl', combined)

    summary = {
        'version': 'o2laf_camerabench_index_v1',
        'classes': list(CAMERABENCH_CLASSES),
        'train_videos': sum(row['split'] == 'train' for row in combined),
        'val_videos': sum(row['split'] == 'val' for row in combined),
        'test_videos': sum(row['split'] == 'test' for row in combined),
        'exact_train_test_overlap': len(exact_overlap),
        'source_group_overlap_with_public_test': len(source_overlap),
        'label_summary': label_summary,
    }
    write_json_atomic(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
