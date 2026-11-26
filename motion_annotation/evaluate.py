"""Unified evaluator for all four motion-annotation modes (single or ensemble).

Evaluates one or more checkpoints from ``train_frozen.py`` (Modes 2 & 3) or
``train_finetune.py`` (Modes 1 & 4) on either:
- Official CameraBench public test (``--camerabench-task-dir``), emitting exact
  official per-task score JSON files plus 15-class macro AP and macro ROC-AUC;
- Any custom JSONL test split (``--video-index`` + ``--split``).
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from motion_annotation.checkpointing import write_json_atomic
from motion_annotation.data import (
    CAMERABENCH_CLASSES,
    binary_average_precision,
    binary_roc_auc,
    classification_metrics,
    labels_to_arrays,
    read_rows,
    safe_video_filename,
)
from motion_annotation.heads import build_head
from motion_annotation.models import (
    WanLatentProvider,
    build_wan_action_encoder,
    load_trainable_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate trained motion-annotation checkpoint(s) on test clips.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        action='append',
        required=True,
        help='Path to best.pt; pass multiple times to average probabilities across seeds.',
    )
    parser.add_argument('--video-index', type=Path, required=True, help='JSONL video index containing the test split.')
    parser.add_argument('--split', default='test', help='Split name in --video-index to evaluate (default: test).')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-name', default='o2laf_annotator', help='Identifier recorded in output score files.')
    parser.add_argument(
        '--features',
        type=Path,
        help='Frozen feature directory (required for frozen_head checkpoints from train_frozen.py).',
    )
    parser.add_argument(
        '--latent-cache',
        type=Path,
        help='Cached Wan VAE latent directory (required for finetuned_encoder checkpoints from train_finetune.py).',
    )
    parser.add_argument('--model', help='Optional override for O²LaF model path (for ssl_lora checkpoints).')
    parser.add_argument('--base-model', help='Optional override for Wan 2.2 TI2V-5B base model path.')
    parser.add_argument('--window-microbatch', type=int, default=2)
    parser.add_argument(
        '--camerabench-task-dir',
        type=Path,
        help='Optional CameraBench binary_classification/ dir to emit official per-task score JSON files.',
    )
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--local-files-only', action='store_true')
    return parser.parse_args()


def _find_feature_file(root: Path, row: dict) -> Path:
    safe = safe_video_filename(row['video_id'])
    candidate = root / str(row.get('split', 'unsplit')) / f'{safe}.npz'
    if candidate.is_file():
        return candidate
    for match in root.glob(f'*/{safe}.npz'):
        return match
    raise FileNotFoundError(f'Missing feature archive for {row["video_id"]} under {root}.')


@torch.inference_mode()
def predict_frozen_checkpoints(
    payloads: Sequence[dict],
    test_rows: Sequence[dict],
    *,
    feature_root: Path,
    classes: Sequence[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    feature_keys = {str(p['feature_key']) for p in payloads}
    if len(feature_keys) != 1:
        raise ValueError(f'Ensemble frozen checkpoints must share one feature_key; got {sorted(feature_keys)}.')
    feature_key = next(iter(feature_keys))

    heads = []
    means = []
    stds = []
    for payload in payloads:
        head = build_head(
            str(payload['head']),
            int(payload['input_dim']),
            len(classes),
            hidden_dim=int(payload.get('hidden_dim', 128)),
            dropout=float(payload.get('dropout', 0.0)),
        ).to(device)
        head.load_state_dict(payload['head_state'], strict=True)
        head.eval()
        heads.append(head)
        means.append(torch.as_tensor(payload['feature_mean'], device=device, dtype=torch.float32).view(1, 1, -1))
        stds.append(torch.as_tensor(payload['feature_std'], device=device, dtype=torch.float32).view(1, 1, -1))

    predictions: dict[str, np.ndarray] = {}
    for row in test_rows:
        path = _find_feature_file(feature_root, row)
        with np.load(path, allow_pickle=False) as data:
            values = torch.from_numpy(data[feature_key].astype(np.float32)).unsqueeze(0).to(device)
        probs = np.zeros(len(classes), dtype=np.float64)
        for head, mean, std in zip(heads, means, stds):
            logits = head((values - mean) / std)[0]
            probs += torch.sigmoid(logits).cpu().numpy()
        predictions[str(row['video_id'])] = probs / len(heads)
    return predictions


@torch.inference_mode()
def predict_finetuned_checkpoints(
    checkpoint_paths: Sequence[Path],
    payloads: Sequence[dict],
    test_rows: Sequence[dict],
    *,
    latent_cache: Path,
    classes: Sequence[str],
    device: torch.device,
    model_override: str | None,
    base_model_override: str | None,
    window_microbatch: int,
    local_files_only: bool,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    if device.type != 'cuda':
        raise RuntimeError('CUDA is required to evaluate finetuned Wan DiT checkpoints.')

    from o2laf.inference.prompts import fixed_prompt_embeds

    accumulated = {str(row['video_id']): np.zeros(len(classes), dtype=np.float64) for row in test_rows}
    for ckpt_idx, (ckpt_path, payload) in enumerate(zip(checkpoint_paths, payloads)):
        cache_path = output_dir / 'checkpoint_predictions' / f'ckpt_{ckpt_idx}_seed{payload.get("seed", 0)}.json'
        ckpt_stat = ckpt_path.stat()
        ckpt_sig = {'path': str(ckpt_path.resolve()), 'size': int(ckpt_stat.st_size), 'mtime_ns': int(ckpt_stat.st_mtime_ns)}
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            if (
                cached.get('checkpoint') == ckpt_sig
                and cached.get('classes') == list(classes)
                and set(cached.get('predictions', {})) == set(accumulated)
            ):
                for vid, scores in cached['predictions'].items():
                    accumulated[vid] += np.asarray(scores, dtype=np.float64)
                print(f'Reused cached predictions for {ckpt_path.name}', flush=True)
                continue

        meta = payload['model_metadata']
        mode = str(payload['mode'])
        model_ref = model_override or meta.get('model')
        base_model_ref = base_model_override or meta.get('base_model_path') or meta.get('base_model_id')
        lora_cfg = meta.get('lora_config') or {}
        encoder, built_meta = build_wan_action_encoder(
            mode,
            model=model_ref,
            base_model=base_model_ref,
            device=device,
            base_dtype=torch.bfloat16,
            lora_rank=int(lora_cfg.get('lora_rank', 128)),
            lora_alpha=int(lora_cfg.get('lora_alpha', 128)),
            latent_dim=int(meta.get('latent_dim', 32)),
            query_init_std=float(meta.get('query_init_std', 0.02)),
            enable_gradient_checkpointing=False,
            local_files_only=local_files_only,
        )
        if built_meta['trainable_contract'] != meta['trainable_contract']:
            raise ValueError(f'{ckpt_path}: reconstructed model trainable contract differs from checkpoint.')
        load_trainable_state_dict(encoder, payload['model_trainable_state'])
        encoder.eval()

        head = build_head(
            str(payload['head']),
            int(meta['feature_dim']),
            len(classes),
            hidden_dim=int(payload.get('hidden_dim', 128)),
            dropout=float(payload.get('dropout', 0.0)),
        ).to(device)
        head.load_state_dict(payload['head_state'], strict=True)
        head.eval()

        prompt_embeds = fixed_prompt_embeds(
            base_model=built_meta['base_model_id'],
            prompt=built_meta['fixed_prompt'],
            base_model_path=built_meta['base_model_path'],
            device=device,
            dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        provider = WanLatentProvider(latent_cache, encoder, prompt_embeds, device, microbatch=window_microbatch)

        per_ckpt: dict[str, list[float]] = {}
        for number, row in enumerate(test_rows, 1):
            features = provider.features(row)
            probs = torch.sigmoid(head(features.unsqueeze(0))[0]).float().cpu().numpy()
            vid = str(row['video_id'])
            per_ckpt[vid] = [float(x) for x in probs]
            accumulated[vid] += probs
            if number % 100 == 0 or number == len(test_rows):
                print(f'[{ckpt_path.name}] evaluated {number}/{len(test_rows)}', flush=True)

        write_json_atomic(
            cache_path,
            {
                'checkpoint': ckpt_sig,
                'classes': list(classes),
                'predictions': per_ckpt,
            },
        )
        del provider, head, encoder
        gc.collect()
        torch.cuda.empty_cache()

    return {vid: values / len(payloads) for vid, values in accumulated.items()}


def evaluate_camerabench_official(
    task_dir: Path,
    test_rows: Sequence[dict],
    clip_scores: dict[str, dict[str, float]],
    *,
    output_dir: Path,
    model_name: str,
    checkpoint_paths: Sequence[Path],
    mode: str,
) -> dict:
    video_paths = {str(row['video_id']): str(row.get('video_path', '')) for row in test_rows}
    per_class = {}
    score_files = []
    for task in CAMERABENCH_CLASSES:
        task_path = task_dir / f'{task}.jsonl'
        if not task_path.is_file():
            raise FileNotFoundError(f'Missing CameraBench task file: {task_path}.')
        task_records = [json.loads(line) for line in task_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        outputs = []
        targets = []
        scores = []
        for idx, record in enumerate(task_records):
            video_id = Path(record['image']).stem
            if video_id not in clip_scores:
                raise KeyError(f'Test clip {video_id!r} required by {task_path.name} is missing from predictions.')
            score = clip_scores[video_id][task]
            target = int(str(record['label']).strip().lower() == 'yes')
            targets.append(target)
            scores.append(score)
            outputs.append(
                {
                    'sample_id': str(idx),
                    'video_path': video_paths.get(video_id, record['image']),
                    'question': record['question'],
                    'ground_truth_label': record['label'],
                    'score': score,
                    'method': f'o2laf_{mode}',
                    'error': None,
                }
            )
        per_class[task] = {
            'average_precision': binary_average_precision(np.asarray(targets), np.asarray(scores)),
            'roc_auc': binary_roc_auc(np.asarray(targets), np.asarray(scores)),
            'num_samples': len(targets),
            'num_positive': int(np.sum(targets)),
            'num_negative': int(len(targets) - np.sum(targets)),
        }
        out_file = output_dir / f'classification_scores_{model_name}_{task}.json'
        write_json_atomic(
            out_file,
            {
                'metadata': {
                    'method_type': 'O2LaF_Motion_Annotator',
                    'model_name': model_name,
                    'mode': mode,
                    'checkpoints': [str(p.resolve()) for p in checkpoint_paths],
                    'ensemble_size': len(checkpoint_paths),
                    'split_name': task,
                },
                'scores': outputs,
            },
        )
        score_files.append(str(out_file))
    return {
        'official_15_macro_ap': float(np.mean([per_class[t]['average_precision'] for t in CAMERABENCH_CLASSES])),
        'official_15_macro_auc': float(np.mean([per_class[t]['roc_auc'] for t in CAMERABENCH_CLASSES])),
        'per_class': per_class,
        'official_score_files': score_files,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payloads = [torch.load(path, map_location='cpu', weights_only=False) for path in args.checkpoint]

    ckpt_types = {str(p['checkpoint_type']) for p in payloads}
    if len(ckpt_types) != 1:
        raise ValueError(f'Cannot mix frozen_head and finetuned_encoder checkpoints: {sorted(ckpt_types)}.')
    ckpt_type = next(iter(ckpt_types))

    modes = {str(p['mode']) for p in payloads}
    if len(modes) != 1:
        raise ValueError(f'All ensemble checkpoints must share one mode; got {sorted(modes)}.')
    mode = next(iter(modes))

    classes = tuple(payloads[0]['classes'])
    if any(tuple(p['classes']) != classes for p in payloads[1:]):
        raise ValueError('All ensemble checkpoints must share identical class lists and ordering.')

    test_rows = read_rows(args.video_index, {args.split})
    if not test_rows:
        raise ValueError(f'No rows found for split {args.split!r} in {args.video_index}.')
    test_rows.sort(key=lambda row: str(row['video_id']))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_type == 'frozen_head':
        if args.features is None:
            raise ValueError('--features is required to evaluate frozen_head checkpoints.')
        probs_by_id = predict_frozen_checkpoints(
            payloads,
            test_rows,
            feature_root=args.features,
            classes=classes,
            device=device,
        )
    elif ckpt_type == 'finetuned_encoder':
        if args.latent_cache is None:
            raise ValueError('--latent-cache is required to evaluate finetuned_encoder checkpoints.')
        probs_by_id = predict_finetuned_checkpoints(
            args.checkpoint,
            payloads,
            test_rows,
            latent_cache=args.latent_cache,
            classes=classes,
            device=device,
            model_override=args.model,
            base_model_override=args.base_model,
            window_microbatch=args.window_microbatch,
            local_files_only=args.local_files_only,
            output_dir=args.output_dir,
        )
    else:
        raise ValueError(f'Unknown checkpoint_type {ckpt_type!r}.')

    clip_scores = {
        vid: {task: float(values[idx]) for idx, task in enumerate(classes)}
        for vid, values in probs_by_id.items()
    }
    write_json_atomic(args.output_dir / 'clip_predictions.json', clip_scores)

    metrics: dict = {
        'version': 'o2laf_motion_annotation_eval_v1',
        'model_name': args.model_name,
        'mode': mode,
        'checkpoint_type': ckpt_type,
        'checkpoints': [str(p.resolve()) for p in args.checkpoint],
        'ensemble_size': len(args.checkpoint),
        'classes': list(classes),
        'num_clips': len(test_rows),
    }

    if args.camerabench_task_dir is not None:
        if classes != CAMERABENCH_CLASSES:
            raise ValueError('Official CameraBench evaluation requires checkpoints trained on the 15 CameraBench classes.')
        metrics.update(
            evaluate_camerabench_official(
                args.camerabench_task_dir,
                test_rows,
                clip_scores,
                output_dir=args.output_dir,
                model_name=args.model_name,
                checkpoint_paths=args.checkpoint,
                mode=mode,
            )
        )
    else:
        labeled = [row for row in test_rows if row.get('labels')]
        if labeled:
            targets, masks = zip(*(labels_to_arrays(row['labels'], classes) for row in labeled))
            scores = np.stack([probs_by_id[str(row['video_id'])] for row in labeled])
            metrics.update(
                classification_metrics(
                    np.stack(targets),
                    np.stack(masks),
                    scores,
                    classes=classes,
                )
            )

    write_json_atomic(args.output_dir / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
