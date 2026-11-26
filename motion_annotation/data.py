"""Dataset schemas, CameraBench utilities, sparse-label batching, and metrics."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

CAMERABENCH_CLASSES: tuple[str, ...] = (
    'Move_Down',
    'Move_In',
    'Move_Left',
    'Move_Out',
    'Move_Right',
    'Move_Up',
    'Pan_Left',
    'Pan_Right',
    'Roll_Clockwise',
    'Roll_Counterclockwise',
    'Static',
    'Tilt_Down',
    'Tilt_Up',
    'Zoom_In',
    'Zoom_Out',
)


# -- Generic sparse multi-label helpers -----------------------------------------


def parse_binary_label(value: Any) -> int:
    """Parse a binary label accepting Yes/No, True/False, or 1/0."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
        raise ValueError(f'Binary label integer must be 0 or 1, got {value!r}.')
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('yes', 'true', '1'):
            return 1
        if lowered in ('no', 'false', '0'):
            return 0
    raise ValueError(f'Unsupported binary label value {value!r}; expected Yes/No, 1/0, or True/False.')


def resolve_classes(rows: Sequence[dict], classes_arg: str | Sequence[str] | None = None) -> tuple[str, ...]:
    """Resolve class list from explicit argument, preset name, or JSONL rows."""
    if classes_arg is None or classes_arg == 'auto':
        found = sorted({str(task) for row in rows for task in (row.get('labels') or {})})
        if not found:
            raise ValueError('Could not infer classes: no observed labels found in the provided rows.')
        if set(found) <= set(CAMERABENCH_CLASSES) and len(found) == len(CAMERABENCH_CLASSES):
            return CAMERABENCH_CLASSES
        return tuple(found)
    if isinstance(classes_arg, str):
        if classes_arg.lower() == 'camerabench':
            return CAMERABENCH_CLASSES
        candidate = Path(classes_arg)
        if candidate.is_file():
            loaded = json.loads(candidate.read_text(encoding='utf-8'))
            if not isinstance(loaded, list) or not loaded:
                raise ValueError(f'{candidate} must contain a non-empty JSON list of class names.')
            return tuple(str(item) for item in loaded)
        parsed = tuple(item.strip() for item in classes_arg.split(',') if item.strip())
        if not parsed:
            raise ValueError(f'Could not parse any class names from {classes_arg!r}.')
        return parsed
    resolved = tuple(str(item) for item in classes_arg)
    if not resolved:
        raise ValueError('classes must be non-empty.')
    return resolved


def labels_to_arrays(labels: dict[str, Any], classes: Sequence[str] = CAMERABENCH_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    """Convert a sparse ``{class: Yes/No|1/0}`` mapping into ``(target, mask)`` arrays."""
    targets = np.zeros(len(classes), dtype=np.float32)
    mask = np.zeros(len(classes), dtype=bool)
    for index, task in enumerate(classes):
        if task not in labels or labels[task] is None:
            continue
        mask[index] = True
        targets[index] = float(parse_binary_label(labels[task]))
    return targets, mask


def read_rows(path: str | Path, splits: set[str] | None = None) -> list[dict]:
    """Read a JSONL video index, optionally filtering by ``split``."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if splits is not None:
        rows = [row for row in rows if str(row.get('split', 'unsplit')) in splits]
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(target)


def safe_video_filename(video_id: str) -> str:
    return str(video_id).replace('/', '_')


# -- CameraBench official preparation helpers -----------------------------------


def normalize_question(text: str) -> str:
    """Normalize CameraBench prompt wrappers without altering question meaning."""
    text = text.replace('<video>', '')
    text = re.sub(r'\s*Please only answer Yes or No\.\s*$', '', text, flags=re.IGNORECASE)
    return ' '.join(text.strip().lower().split())


def source_group(mp4_name: str) -> str:
    """Source-video identifier preceding CameraBench clip suffixes."""
    stem = PurePosixPath(mp4_name).stem
    return stem.split('.', 1)[0]


def read_official_questions(task_dir: Path) -> dict[str, str]:
    """Map exact official binary questions to the 15 CameraBench primitive names."""
    result: dict[str, str] = {}
    for path in sorted(Path(task_dir).glob('*.jsonl')):
        first = next(
            json.loads(line)
            for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        )
        task = path.stem
        if task not in CAMERABENCH_CLASSES:
            raise ValueError(f'Unexpected CameraBench task {task!r}.')
        question = normalize_question(str(first['question']))
        if question in result and result[question] != task:
            raise ValueError(f'Question maps to multiple tasks: {question!r}.')
        result[question] = task
    if set(result.values()) != set(CAMERABENCH_CLASSES):
        raise ValueError(
            f'Official task dir {task_dir} does not cover all 15 CameraBench classes; '
            f'missing={sorted(set(CAMERABENCH_CLASSES) - set(result.values()))}.'
        )
    return result


def extract_sparse_labels(
    datasets: Iterable[tuple[str, list[dict]]],
    question_to_task: dict[str, str],
) -> tuple[dict[str, dict[str, str]], dict]:
    """Extract exact official-question sparse labels from CameraBench SFT JSONs.

    Unasked classes stay missing (masked during training). Repeated answers
    across records or files must agree; conflicts raise ``ValueError``.
    """
    evidence: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    total_rows = 0
    mapped_rows = 0
    source_rows: dict[str, int] = {}
    for source, rows in datasets:
        source_rows[source] = len(rows)
        total_rows += len(rows)
        for row in rows:
            question = normalize_question(str(row['messages'][0]['content']))
            task = question_to_task.get(question)
            if task is None:
                continue
            answer = str(row['messages'][1]['content']).strip().lower()
            if answer not in {'yes', 'no'}:
                raise ValueError(f'Unexpected binary answer {answer!r}.')
            video = PurePosixPath(str(row['videos'][0])).name
            evidence[video][task].append((source, answer))
            mapped_rows += 1

    labels: dict[str, dict[str, str]] = {}
    origin_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    conflicts = []
    for video, task_map in evidence.items():
        labels[video] = {}
        for task, values in task_map.items():
            answers = {answer for _, answer in values}
            if len(answers) != 1:
                conflicts.append({'video': video, 'task': task, 'evidence': values})
                continue
            labels[video][task] = next(iter(answers)).title()
            for source in {source for source, _ in values}:
                origin_counts[task][source] += 1
    if conflicts:
        raise ValueError(f'Conflicting CameraBench labels: {conflicts[:5]}.')

    summary = {
        'total_rows': total_rows,
        'mapped_rows': mapped_rows,
        'unmapped_rows': total_rows - mapped_rows,
        'source_rows': source_rows,
        'labeled_videos': len(labels),
        'coverage_histogram': {
            str(count): sum(len(tasks) == count for tasks in labels.values())
            for count in sorted({len(tasks) for tasks in labels.values()})
        },
        'fully_labeled_videos': sum(len(tasks) == len(CAMERABENCH_CLASSES) for tasks in labels.values()),
        'per_class': {},
    }
    for task in CAMERABENCH_CLASSES:
        observed = [tasks[task] for tasks in labels.values() if task in tasks]
        summary['per_class'][task] = {
            'observed': len(observed),
            'positive': observed.count('Yes'),
            'negative': observed.count('No'),
            'source_video_count': len(
                {source_group(video) for video, tasks in labels.items() if task in tasks}
            ),
            'origins': dict(origin_counts[task]),
        }
    return labels, summary


def choose_grouped_split(
    labels_by_video: dict[str, dict[str, Any]],
    *,
    classes: Sequence[str] = CAMERABENCH_CLASSES,
    val_fraction: float = 0.15,
    seed: int = 20261026,
    candidates: int = 5000,
) -> tuple[set[str], set[str], dict]:
    """Choose a deterministic source-group-disjoint split covering all classes."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('val_fraction must be in (0, 1).')
    videos = sorted(labels_by_video)
    groups = sorted({source_group(video) for video in videos})
    group_rows: dict[str, list[str]] = defaultdict(list)
    for video in videos:
        group_rows[source_group(video)].append(video)

    total_observed = np.zeros(len(classes), dtype=np.float64)
    total_positive = np.zeros(len(classes), dtype=np.float64)
    for video in videos:
        targets, mask = labels_to_arrays(labels_by_video[video], classes)
        total_observed += mask
        total_positive += targets * mask

    rng = np.random.default_rng(seed)
    best = None
    target_videos = len(videos) * val_fraction
    for candidate_index in range(candidates):
        order = rng.permutation(groups)
        selected: list[str] = []
        selected_videos = 0
        for group in order:
            if selected_videos >= target_videos and selected:
                break
            selected.append(str(group))
            selected_videos += len(group_rows[str(group)])
        val_groups = set(selected)
        val_observed = np.zeros(len(classes), dtype=np.float64)
        val_positive = np.zeros(len(classes), dtype=np.float64)
        for group in val_groups:
            for video in group_rows[group]:
                targets, mask = labels_to_arrays(labels_by_video[video], classes)
                val_observed += mask
                val_positive += targets * mask
        train_observed = total_observed - val_observed
        train_positive = total_positive - val_positive
        val_negative = val_observed - val_positive
        train_negative = train_observed - train_positive
        valid = np.all(
            (val_positive > 0) & (val_negative > 0) & (train_positive > 0) & (train_negative > 0)
        )
        if not valid:
            continue
        observed_ratio = val_observed / np.maximum(total_observed, 1)
        positive_ratio = val_positive / np.maximum(total_positive, 1)
        video_ratio = selected_videos / len(videos)
        score = (
            4.0 * abs(video_ratio - val_fraction)
            + float(np.mean(np.abs(observed_ratio - val_fraction)))
            + float(np.mean(np.abs(positive_ratio - val_fraction)))
            + float(np.max(np.abs(positive_ratio - val_fraction)))
        )
        key = (score, candidate_index)
        if best is None or key < best[0]:
            best = (
                key,
                val_groups,
                {
                    'score': score,
                    'candidate_index': candidate_index,
                    'video_fraction': video_ratio,
                    'val_observed': val_observed.astype(int).tolist(),
                    'val_positive': val_positive.astype(int).tolist(),
                    'train_observed': train_observed.astype(int).tolist(),
                    'train_positive': train_positive.astype(int).tolist(),
                },
            )
    if best is None:
        raise RuntimeError('Failed to find a source-group-disjoint split covering all classes.')
    val_groups = best[1]
    val_videos = {video for video in videos if source_group(video) in val_groups}
    train_videos = set(videos) - val_videos
    details = {
        'version': 'camerabench_source_group_split_v1',
        'seed': seed,
        'candidate_trials': candidates,
        'target_val_fraction': val_fraction,
        'classes': list(classes),
        'train_videos': len(train_videos),
        'val_videos': len(val_videos),
        'train_groups': len({source_group(video) for video in train_videos}),
        'val_groups': len(val_groups),
        'val_group_ids': sorted(val_groups),
        **best[2],
    }
    return train_videos, val_videos, details


# -- Pure-NumPy binary metrics (exact match to scikit-learn) --------------------


def _binary_metric_inputs(targets: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets.shape != scores.shape or not len(targets):
        raise ValueError('Binary metrics require equally sized non-empty inputs.')
    if not np.isin(targets, (0, 1)).all():
        raise ValueError('Binary metric targets must contain only 0 and 1.')
    if not np.isfinite(scores).all():
        raise ValueError('Binary metric scores must be finite.')
    targets = targets.astype(np.int64, copy=False)
    if np.unique(targets).size != 2:
        raise ValueError('Binary metrics require both positive and negative targets.')
    return targets, scores


def _binary_score_curve(targets: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets, scores = _binary_metric_inputs(targets, scores)
    order = np.argsort(scores, kind='mergesort')[::-1]
    ordered_targets = targets[order]
    ordered_scores = scores[order]
    threshold_indices = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(ordered_scores) - 1]
    true_positives = np.cumsum(ordered_targets, dtype=np.float64)[threshold_indices]
    false_positives = (1 + threshold_indices - true_positives).astype(np.float64, copy=False)
    return true_positives, false_positives


def binary_average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    """Non-interpolated binary average precision, matching scikit-learn."""
    true_positives, false_positives = _binary_score_curve(targets, scores)
    previous_true_positives = np.r_[0.0, true_positives[:-1]]
    precision = true_positives / (true_positives + false_positives)
    recall_increment = (true_positives - previous_true_positives) / true_positives[-1]
    return float(np.sum(precision * recall_increment))


def binary_roc_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Binary ROC AUC with tie handling, matching scikit-learn."""
    true_positives, false_positives = _binary_score_curve(targets, scores)
    tpr = np.r_[0.0, true_positives / true_positives[-1]]
    fpr = np.r_[0.0, false_positives / false_positives[-1]]
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) * 0.5))


def classification_metrics(
    targets: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    classes: Sequence[str] = CAMERABENCH_CLASSES,
) -> dict:
    """Compute macro AP, macro ROC-AUC, micro AP, and per-class stats on sparse labels."""
    per_class = {}
    aps = []
    aucs = []
    for index, task in enumerate(classes):
        observed = masks[:, index].astype(bool)
        target = targets[observed, index]
        score = scores[observed, index]
        if len(np.unique(target)) != 2:
            raise ValueError(f'{task}: evaluation requires both positive and negative labels.')
        ap = binary_average_precision(target, score)
        auc = binary_roc_auc(target, score)
        aps.append(ap)
        aucs.append(auc)
        per_class[task] = {
            'average_precision': ap,
            'roc_auc': auc,
            'observed': int(len(target)),
            'positive': int(target.sum()),
            'negative': int(len(target) - target.sum()),
        }
    flat = masks.astype(bool)
    return {
        'macro_average_precision': float(np.mean(aps)),
        'macro_roc_auc': float(np.mean(aucs)),
        'micro_average_precision': binary_average_precision(targets[flat], scores[flat]),
        'per_class': per_class,
    }


# -- Label-budget subsetting, class weights, and batching -----------------------


def annotation_subset(
    rows: list[dict],
    *,
    classes: Sequence[str] = CAMERABENCH_CLASSES,
    fraction: float,
    seed: int,
) -> tuple[list[dict], dict]:
    """Select an exact stratified sparse-label budget of ``(video, class)`` annotations."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError('fraction must be in (0, 1].')
    examples: list[tuple[int, str, int]] = []
    for row_index, row in enumerate(rows):
        for task in classes:
            if task in row['labels'] and row['labels'][task] is not None:
                examples.append((row_index, task, parse_binary_label(row['labels'][task])))
    target = min(len(examples), max(1, int(round(len(examples) * fraction))))
    if target == len(examples):
        copied = [{**row, 'labels': dict(row['labels'])} for row in rows]
        return copied, {
            'fraction': fraction,
            'seed': seed,
            'available_annotations': len(examples),
            'selected_annotations': len(examples),
            'selected_videos': len(copied),
        }

    strata: dict[tuple[str, int], list[tuple[int, str, int]]] = defaultdict(list)
    for example in examples:
        strata[(example[1], example[2])].append(example)
    keys = [(task, value) for task in classes for value in (1, 0)]
    if set(strata) != set(keys):
        raise ValueError('Stratified annotation subset requires positive and negative train labels per class.')

    counts = np.asarray([len(strata[key]) for key in keys], dtype=np.int64)
    allocation = np.zeros(len(keys), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if target >= len(keys):
        allocation[:] = 1
        remaining = target - len(keys)
        capacities = counts - 1
        if remaining:
            ideal = remaining * capacities / capacities.sum()
            floor = np.minimum(np.floor(ideal).astype(np.int64), capacities)
            allocation += floor
            leftover = remaining - int(floor.sum())
            tie = rng.random(len(keys))
            order = sorted(
                range(len(keys)),
                key=lambda idx: (ideal[idx] - floor[idx], tie[idx]),
                reverse=True,
            )
            for idx in order:
                if leftover == 0:
                    break
                if allocation[idx] < counts[idx]:
                    allocation[idx] += 1
                    leftover -= 1
    else:
        chosen = rng.choice(len(keys), size=target, replace=False, p=counts / counts.sum())
        allocation[chosen] = 1

    selected: set[tuple[int, str]] = set()
    for idx, key in enumerate(keys):
        count = int(allocation[idx])
        if count:
            candidates = strata[key]
            chosen_indices = rng.choice(len(candidates), size=count, replace=False)
            selected.update((candidates[int(i)][0], candidates[int(i)][1]) for i in chosen_indices)

    subset = []
    for row_index, row in enumerate(rows):
        kept = {
            task: value
            for task, value in row['labels'].items()
            if (row_index, task) in selected
        }
        if kept:
            subset.append({**row, 'labels': kept})
    return subset, {
        'fraction': fraction,
        'seed': seed,
        'available_annotations': len(examples),
        'selected_annotations': target,
        'selected_videos': len(subset),
    }


def class_weights_np(
    rows: Sequence[dict],
    *,
    classes: Sequence[str] = CAMERABENCH_CLASSES,
    cap: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """Compute capped inverse-frequency positive weights per class."""
    positive = np.zeros(len(classes), dtype=np.float64)
    negative = np.zeros(len(classes), dtype=np.float64)
    for row in rows:
        target, mask = labels_to_arrays(row['labels'], classes)
        positive += target * mask
        negative += (1.0 - target) * mask
    if np.any(positive == 0) or np.any(negative == 0):
        missing = [
            classes[i] for i in range(len(classes)) if positive[i] == 0 or negative[i] == 0
        ]
        raise ValueError(f'Training split needs both positive and negative labels for every class; missing: {missing}.')
    weights = np.minimum(negative / positive, float(cap)).astype(np.float32)
    details = {
        task: {
            'positive': int(positive[index]),
            'negative': int(negative[index]),
            'pos_weight': float(weights[index]),
        }
        for index, task in enumerate(classes)
    }
    return weights, details


def epoch_batches(
    rows: list[dict],
    *,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
    clips_per_rank: int,
) -> list[list[dict]]:
    """Deterministic global shuffle partitioned across ranks without duplication."""
    if clips_per_rank < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid distributed batch parameters.')
    order = np.random.default_rng(int(seed) + 1_000_003 * int(epoch)).permutation(len(rows))
    global_batch = int(world_size) * int(clips_per_rank)
    steps = math.ceil(len(rows) / global_batch)
    result = []
    for step in range(steps):
        start = step * global_batch + rank * clips_per_rank
        stop = min(start + clips_per_rank, (step + 1) * global_batch, len(rows))
        result.append([rows[int(index)] for index in order[start:stop]])
    return result


# -- Torch batching & loss helpers (lazy torch import keeps module CPU-friendly) -


def labels_for_rows(rows: Sequence[dict], classes: Sequence[str], device):
    import torch

    targets, masks = zip(*(labels_to_arrays(row['labels'], classes) for row in rows))
    return (
        torch.from_numpy(np.stack(targets)).to(device),
        torch.from_numpy(np.stack(masks)).to(device),
    )


def pad_feature_sequences(sequences: list):
    import torch

    if not sequences:
        raise ValueError('Cannot pad an empty feature batch.')
    width = int(sequences[0].shape[1])
    maximum = max(int(sequence.shape[0]) for sequence in sequences)
    padded = []
    masks = []
    for sequence in sequences:
        if sequence.ndim != 2 or sequence.shape[1] != width or not len(sequence):
            raise ValueError('Feature sequences must be non-empty [T, D] tensors with a common width.')
        padding = maximum - len(sequence)
        padded.append(torch.nn.functional.pad(sequence, (0, 0, 0, padding)))
        masks.append(
            torch.cat(
                [
                    torch.ones(len(sequence), dtype=torch.bool, device=sequence.device),
                    torch.zeros(padding, dtype=torch.bool, device=sequence.device),
                ]
            )
        )
    return torch.stack(padded), torch.stack(masks)


def masked_bce_sum(logits, targets, mask, *, pos_weight):
    import torch

    values = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    return (values * mask).sum()


def masked_bce_mean(logits, targets, mask, *, pos_weight):
    observed = mask.sum().clamp_min(1)
    return masked_bce_sum(logits, targets, mask, pos_weight=pos_weight) / observed
