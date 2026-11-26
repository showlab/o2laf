"""Train a lightweight multi-label head on frozen O²LaF Wan DiT features.

Covers Modes 2 and 3:
- ``ssl_action_mu``: frozen O²LaF 32-D posterior mean ``action_mu`` + head.
- ``ssl_pre_gaussian``: frozen O²LaF 3072-D pre-Gaussian ``transition_features`` + head.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from motion_annotation.checkpointing import (
    save_checkpoint_atomic,
    seed_everything,
    write_json_atomic,
)
from motion_annotation.data import (
    annotation_subset,
    class_weights_np,
    classification_metrics,
    labels_to_arrays,
    masked_bce_mean,
    read_rows,
    resolve_classes,
    safe_video_filename,
)
from motion_annotation.heads import HEAD_ARCHITECTURES, build_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a supervised head on frozen O²LaF features (Modes 2 & 3).')
    parser.add_argument(
        '--mode',
        choices=('ssl_action_mu', 'ssl_pre_gaussian'),
        required=True,
        help='ssl_action_mu (32-D action_mu) or ssl_pre_gaussian (3072-D transition_features).',
    )
    parser.add_argument('--features', type=Path, required=True, help='Feature directory written by extract_features.py.')
    parser.add_argument('--video-index', type=Path, required=True, help='JSONL index with train and val splits.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--feature-key', help='Override .npz array key (defaults to action_mu or transition_features).')
    parser.add_argument(
        '--classes',
        default='auto',
        help='Class list: "auto" (infer from train split), "camerabench", comma-separated names, or a JSON file.',
    )
    parser.add_argument('--head', choices=HEAD_ARCHITECTURES, default='direct_linear')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-label-fraction', type=float, default=1.0)
    parser.add_argument('--subset-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--pos-weight-cap', type=float, default=20.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


class FrozenFeatureDataset(Dataset):
    """Variable-length per-video feature sequences paired with sparse binary labels."""

    def __init__(
        self,
        rows: Sequence[dict],
        feature_root: Path,
        *,
        classes: Sequence[str],
        feature_key: str,
        preload: bool = True,
    ) -> None:
        self.rows = list(rows)
        self.feature_root = Path(feature_root)
        self.classes = tuple(classes)
        self.feature_key = feature_key
        self._items = [self._load(row) for row in self.rows] if preload else None

    def __len__(self) -> int:
        return len(self.rows)

    def feature_path(self, row: dict) -> Path:
        safe = safe_video_filename(row['video_id'])
        split = str(row.get('split', 'unsplit'))
        candidate = self.feature_root / split / f'{safe}.npz'
        if candidate.is_file():
            return candidate
        for match in self.feature_root.glob(f'*/{safe}.npz'):
            return match
        return candidate

    def _load(self, row: dict) -> dict:
        path = self.feature_path(row)
        if not path.is_file():
            raise FileNotFoundError(f'Missing feature archive {path}. Run extract_features.py first.')
        with np.load(path, allow_pickle=False) as data:
            if self.feature_key not in data:
                raise KeyError(f'{path} does not contain key {self.feature_key!r}; available={list(data.files)}.')
            values = data[self.feature_key].astype(np.float32)
            archive_id = str(data['video_id'].item())
        if archive_id != str(row['video_id']):
            raise ValueError(f'{path}: video_id {archive_id!r} != {row["video_id"]!r}.')
        if values.ndim != 2 or not len(values):
            raise ValueError(f'{path}: {self.feature_key} must be a non-empty [T, D] array.')
        if not np.isfinite(values).all():
            raise ValueError(f'{path}: non-finite values in {self.feature_key}.')
        target, label_mask = labels_to_arrays(row['labels'], self.classes)
        return {
            'video_id': str(row['video_id']),
            'features': torch.from_numpy(values),
            'target': torch.from_numpy(target),
            'label_mask': torch.from_numpy(label_mask),
        }

    def __getitem__(self, index: int) -> dict:
        if self._items is not None:
            return self._items[index]
        return self._load(self.rows[index])


def collate_feature_batch(items: list[dict]) -> dict:
    if not items:
        raise ValueError('Cannot collate an empty batch.')
    feature_dim = int(items[0]['features'].shape[1])
    max_length = max(int(item['features'].shape[0]) for item in items)
    batch = torch.zeros(len(items), max_length, feature_dim, dtype=torch.float32)
    sequence_mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    for index, item in enumerate(items):
        values = item['features']
        if int(values.shape[1]) != feature_dim:
            raise ValueError('Mixed feature dimensions in batch.')
        length = int(values.shape[0])
        batch[index, :length] = values
        sequence_mask[index, :length] = True
    return {
        'video_id': [item['video_id'] for item in items],
        'features': batch,
        'sequence_mask': sequence_mask,
        'target': torch.stack([item['target'] for item in items]),
        'label_mask': torch.stack([item['label_mask'] for item in items]),
    }


def compute_channel_stats(dataset: FrozenFeatureDataset) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = None
    square = None
    for item in dataset:
        values = item['features'].numpy().astype(np.float64)
        count += len(values)
        cur_total = values.sum(axis=0)
        cur_square = np.square(values).sum(axis=0)
        total = cur_total if total is None else total + cur_total
        square = cur_square if square is None else square + cur_square
    if not count:
        raise ValueError('Cannot compute statistics for an empty dataset.')
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def standardize(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    classes: Sequence[str],
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[dict, dict]:
    model.eval()
    losses = []
    targets = []
    masks = []
    scores = []
    video_ids = []
    for batch in loader:
        features = standardize(batch['features'].to(device), mean, std)
        sequence_mask = batch['sequence_mask'].to(device)
        target = batch['target'].to(device)
        label_mask = batch['label_mask'].to(device)
        logits = model(features, sequence_mask)
        losses.append(float(masked_bce_mean(logits, target, label_mask, pos_weight=pos_weight)))
        targets.append(target.cpu().numpy())
        masks.append(label_mask.cpu().numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
        video_ids.extend(batch['video_id'])
    targets_arr = np.concatenate(targets)
    masks_arr = np.concatenate(masks)
    scores_arr = np.concatenate(scores)
    metrics = classification_metrics(targets_arr, masks_arr, scores_arr, classes=classes)
    metrics['loss'] = float(np.mean(losses))
    predictions = {
        video_id: {task: float(scores_arr[row_idx, cls_idx]) for cls_idx, task in enumerate(classes)}
        for row_idx, video_id in enumerate(video_ids)
    }
    return metrics, predictions


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'metrics.json').is_file() and not args.overwrite:
        raise FileExistsError(f'Completed run exists at {args.output_dir}; pass --overwrite to replace.')

    seed_everything(args.seed)
    feature_key = args.feature_key or ('action_mu' if args.mode == 'ssl_action_mu' else 'transition_features')
    all_rows = read_rows(args.video_index)
    train_rows = [row for row in all_rows if row.get('split') == 'train' and row.get('labels')]
    val_rows = [row for row in all_rows if row.get('split') == 'val' and row.get('labels')]
    if not train_rows or not val_rows:
        raise ValueError('Video index must contain non-empty labeled train and val splits.')

    classes = resolve_classes(train_rows, args.classes)
    train_rows, subset_details = annotation_subset(
        train_rows,
        classes=classes,
        fraction=args.train_label_fraction,
        seed=args.subset_seed,
    )

    train_dataset = FrozenFeatureDataset(train_rows, args.features, classes=classes, feature_key=feature_key)
    val_dataset = FrozenFeatureDataset(val_rows, args.features, classes=classes, feature_key=feature_key)
    measured_mean, measured_std = compute_channel_stats(train_dataset)
    input_dim = int(len(measured_mean))

    if args.head == 'direct_linear':
        feature_mean_np = np.zeros(input_dim, dtype=np.float32)
        feature_std_np = np.ones(input_dim, dtype=np.float32)
        normalization = 'per_transition_layernorm_only'
    else:
        feature_mean_np = measured_mean
        feature_std_np = measured_std
        normalization = 'train_split_channel_standardization'

    pos_weight_np, weight_details = class_weights_np(train_rows, classes=classes, cap=args.pos_weight_cap)
    device = torch.device(args.device)
    model = build_head(
        args.head,
        input_dim,
        len(classes),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    feature_mean = torch.from_numpy(feature_mean_np).to(device)
    feature_std = torch.from_numpy(feature_std_np).to(device)
    pos_weight = torch.from_numpy(pos_weight_np).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        collate_fn=collate_feature_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_feature_batch,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {
        'version': 'o2laf_motion_annotation_frozen_v1',
        'checkpoint_type': 'frozen_head',
        'mode': args.mode,
        'features': str(args.features.resolve()),
        'video_index': str(args.video_index.resolve()),
        'feature_key': feature_key,
        'head': args.head,
        'classes': list(classes),
        'seed': int(args.seed),
        'train_label_fraction': float(args.train_label_fraction),
        'subset_seed': int(args.subset_seed),
        'annotation_subset': subset_details,
        'input_dim': input_dim,
        'hidden_dim': int(args.hidden_dim),
        'dropout': float(args.dropout),
        'batch_size': int(args.batch_size),
        'epochs': int(args.epochs),
        'patience': int(args.patience),
        'learning_rate': float(args.learning_rate),
        'weight_decay': float(args.weight_decay),
        'pos_weight_cap': float(args.pos_weight_cap),
        'feature_normalization': normalization,
        'train_videos': len(train_rows),
        'val_videos': len(val_rows),
        'train_class_weights': weight_details,
        'parameters': sum(p.numel() for p in model.parameters()),
    }
    write_json_atomic(args.output_dir / 'config.json', config)

    history = []
    best_key = None
    stale = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            features = standardize(batch['features'].to(device), feature_mean, feature_std)
            sequence_mask = batch['sequence_mask'].to(device)
            target = batch['target'].to(device)
            label_mask = batch['label_mask'].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, sequence_mask)
            loss = masked_bce_mean(logits, target, label_mask, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        scheduler.step()

        val_metrics, _ = evaluate_loader(
            model,
            val_loader,
            classes=classes,
            device=device,
            mean=feature_mean,
            std=feature_std,
            pos_weight=pos_weight,
        )
        row = {
            'epoch': epoch,
            'train_loss': float(np.mean(batch_losses)),
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'val': val_metrics,
        }
        history.append(row)
        key = (
            val_metrics['macro_average_precision'],
            val_metrics['macro_roc_auc'],
            -val_metrics['loss'],
            -epoch,
        )
        if best_key is None or key > best_key:
            best_key = key
            stale = 0
            save_checkpoint_atomic(
                args.output_dir / 'best.pt',
                {
                    'version': 'o2laf_motion_annotation_frozen_v1',
                    'checkpoint_type': 'frozen_head',
                    'mode': args.mode,
                    'head': args.head,
                    'input_dim': input_dim,
                    'hidden_dim': int(args.hidden_dim),
                    'dropout': float(args.dropout),
                    'classes': list(classes),
                    'feature_key': feature_key,
                    'feature_mean': feature_mean_np,
                    'feature_std': feature_std_np,
                    'head_state': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    'epoch': epoch,
                    'seed': int(args.seed),
                    'validation_metrics': val_metrics,
                    'config': config,
                },
            )
        else:
            stale += 1
        write_json_atomic(args.output_dir / 'history.json', history)
        print(
            f'epoch={epoch:03d} train_loss={row["train_loss"]:.5f} '
            f'val_mAP={val_metrics["macro_average_precision"]:.5f} '
            f'val_AUC={val_metrics["macro_roc_auc"]:.5f} stale={stale}',
            flush=True,
        )
        if stale >= args.patience:
            break

    best = torch.load(args.output_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['head_state'])
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_feature_batch,
    )
    train_metrics, _ = evaluate_loader(
        model,
        train_eval_loader,
        classes=classes,
        device=device,
        mean=feature_mean,
        std=feature_std,
        pos_weight=pos_weight,
    )
    val_metrics, val_predictions = evaluate_loader(
        model,
        val_loader,
        classes=classes,
        device=device,
        mean=feature_mean,
        std=feature_std,
        pos_weight=pos_weight,
    )
    metrics = {
        'version': 'o2laf_motion_annotation_frozen_v1',
        'mode': args.mode,
        'head': args.head,
        'best_epoch': int(best['epoch']),
        'epochs_ran': len(history),
        'elapsed_seconds': time.time() - start_time,
        'train': train_metrics,
        'validation': val_metrics,
    }
    write_json_atomic(args.output_dir / 'metrics.json', metrics)
    write_json_atomic(args.output_dir / 'validation_predictions.json', val_predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
