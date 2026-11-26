"""End-to-end LoRA + readout fine-tuning on Wan 2.2 DiT for motion annotation.

Covers Modes 1 and 4 with identical trainable parameter contracts:
- ``wan_base_lora``: Wan 2.2 TI2V-5B base + fresh rank-128 LoRA + fresh readout (3072-D).
- ``ssl_lora``: O²LaF Wan DiT checkpoint + rank-128 LoRA + readout fine-tuning (3072-D).

Both modes freeze the Gaussian posterior projection and ``action_rope_angle``,
use exact two-pass whole-clip recomputation backward over cached Wan VAE
latents, and support single-GPU or multi-GPU (``torchrun``) training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from motion_annotation.checkpointing import (
    build_constant_with_warmup_scheduler,
    optimizer_state_to_parameter_devices,
    recursive_to_cpu,
    resume_compatibility_config,
    save_checkpoint_atomic,
    seed_everything,
    write_json_atomic,
)
from motion_annotation.data import (
    class_weights_np,
    classification_metrics,
    epoch_batches,
    labels_for_rows,
    masked_bce_sum,
    pad_feature_sequences,
    read_rows,
    resolve_classes,
)
from motion_annotation.distributed import (
    all_gather_objects,
    all_reduce_gradients,
    all_reduce_scalar,
    assert_parameters_synchronized,
    barrier,
    broadcast_parameters,
    finish_distributed,
    initialize_distributed,
)
from motion_annotation.heads import HEAD_ARCHITECTURES, build_head
from motion_annotation.models import (
    DEFAULT_BASE_MODEL,
    WanLatentProvider,
    build_wan_action_encoder,
    load_trainable_state_dict,
    optimizer_parameter_groups,
    trainable_parameter_contract,
    trainable_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fine-tune Wan DiT LoRA + readout for motion annotation (Modes 1 & 4).')
    parser.add_argument(
        '--mode',
        choices=('wan_base_lora', 'ssl_lora'),
        required=True,
        help='wan_base_lora (base Wan 2.2 + fresh LoRA) or ssl_lora (O²LaF SSL + LoRA FT).',
    )
    parser.add_argument(
        '--model',
        help='O²LaF Wan DiT model (required for ssl_lora; optional metadata reference for wan_base_lora).',
    )
    parser.add_argument(
        '--base-model',
        default=None,
        help=f'Wan 2.2 TI2V-5B Hub repo id or local path (default: model base_model or {DEFAULT_BASE_MODEL}).',
    )
    parser.add_argument('--video-index', type=Path, help='JSONL index with train and val splits (optional with --audit-only).')
    parser.add_argument('--latent-cache', type=Path, help='Cached Wan VAE latents dir (optional with --audit-only).')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--classes',
        default='auto',
        help='Class list: "auto" (infer from train split), "camerabench", comma-separated names, or a JSON file.',
    )
    parser.add_argument('--head', choices=HEAD_ARCHITECTURES, default='direct_linear')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lora-rank', type=int, default=128)
    parser.add_argument('--lora-alpha', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--encoder-lora-lr', type=float, default=5e-5)
    parser.add_argument('--readout-lr', type=float, default=1e-4)
    parser.add_argument('--classifier-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.95)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--clips-per-rank', type=int, default=8)
    parser.add_argument('--window-microbatch', type=int, default=2)
    parser.add_argument('--max-grad-norm', type=float, default=5.0)
    parser.add_argument('--pos-weight-cap', type=float, default=20.0)
    parser.add_argument('--limit-train-videos', type=int)
    parser.add_argument('--limit-val-videos', type=int)
    parser.add_argument('--audit-only', action='store_true', help='Write trainable parameter contract and exit.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    return parser.parse_args()


def gradient_norm(parameters) -> float:
    values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).norm().item())


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    head: torch.nn.Module,
    provider: WanLatentProvider,
    rows: Sequence[dict],
    *,
    classes: Sequence[str],
    rank: int,
    world_size: int,
) -> tuple[dict, dict]:
    model.eval()
    head.eval()
    local = []
    for row in rows[rank::world_size]:
        features = provider.features(row)
        logits = head(features.unsqueeze(0))[0]
        target, mask = labels_for_rows([row], classes, features.device)
        local.append(
            {
                'video_id': str(row['video_id']),
                'target': target[0].cpu().numpy(),
                'mask': mask[0].cpu().numpy(),
                'score': torch.sigmoid(logits).float().cpu().numpy(),
            }
        )
    gathered = [item for shard in all_gather_objects(local) for item in shard]
    gathered.sort(key=lambda item: item['video_id'])
    targets = np.stack([item['target'] for item in gathered])
    masks = np.stack([item['mask'] for item in gathered])
    scores = np.stack([item['score'] for item in gathered])
    metrics = classification_metrics(targets, masks, scores, classes=classes)
    predictions = {
        item['video_id']: {task: float(item['score'][idx]) for idx, task in enumerate(classes)}
        for item in gathered
    }
    model.train()
    head.train()
    return metrics, predictions


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Wan DiT fine-tuning.')

    from o2laf.inference.prompts import fixed_prompt_embeds

    rank, world_size, local_rank = initialize_distributed()
    device = torch.device('cuda', local_rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    seed_everything(args.seed)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    pos_weight_np = None
    weight_details: dict = {}
    if args.audit_only and args.video_index is None:
        classes = resolve_classes([], 'camerabench' if args.classes == 'auto' else args.classes)
    else:
        if args.video_index is None:
            raise ValueError('--video-index is required unless --audit-only is set.')
        if not args.audit_only and args.latent_cache is None:
            raise ValueError('--latent-cache is required unless --audit-only is set.')
        all_rows = read_rows(args.video_index)
        train_rows = [row for row in all_rows if row.get('split') == 'train' and row.get('labels')]
        val_rows = [row for row in all_rows if row.get('split') == 'val' and row.get('labels')]
        if not train_rows or not val_rows:
            raise ValueError('Video index must contain non-empty labeled train and val splits.')
        classes = resolve_classes(train_rows, args.classes)
        pos_weight_np, weight_details = class_weights_np(train_rows, classes=classes, cap=args.pos_weight_cap)
        if args.limit_train_videos is not None:
            train_rows = train_rows[:args.limit_train_videos]
        if args.limit_val_videos is not None:
            val_rows = val_rows[:args.limit_val_videos]

    model, model_metadata = build_wan_action_encoder(
        args.mode,
        model=args.model,
        base_model=args.base_model,
        device=device,
        base_dtype=torch.bfloat16,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        enable_gradient_checkpointing=True,
        local_files_only=args.local_files_only,
    )
    head = build_head(
        args.head,
        int(model_metadata['feature_dim']),
        len(classes),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    broadcast_parameters(model.parameters())
    broadcast_parameters(head.parameters())
    local_contract = {
        'model': trainable_parameter_contract(model),
        'head': trainable_parameter_contract(head),
    }
    contracts = all_gather_objects(local_contract)
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise RuntimeError('Distributed ranks constructed mismatched trainable parameter contracts.')
    counts = {
        'model_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'model_total': sum(p.numel() for p in model.parameters()),
        'head_trainable': sum(p.numel() for p in head.parameters()),
    }
    init_sync = {
        'model': assert_parameters_synchronized(model.parameters(), tag='wan encoder'),
        'head': assert_parameters_synchronized(head.parameters(), tag='classifier head'),
    }

    if args.audit_only:
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            audit = {
                'version': 'o2laf_motion_annotation_audit_v1',
                'mode': args.mode,
                'seed': int(args.seed),
                'classes': list(classes),
                'model_metadata': model_metadata,
                'parameters': counts,
                'initialization_sync': init_sync,
                'trainable_contract': local_contract,
            }
            write_json_atomic(args.output_dir / 'audit.json', audit)
            print(json.dumps({'mode': args.mode, 'parameters': counts}, indent=2))
        barrier()
        finish_distributed()
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'metrics.json').is_file() and not args.overwrite:
        raise FileExistsError(f'Completed run exists at {args.output_dir}; pass --overwrite to replace.')

    prompt_embeds = fixed_prompt_embeds(
        base_model=model_metadata['base_model_id'],
        prompt=model_metadata['fixed_prompt'],
        base_model_path=model_metadata['base_model_path'],
        device=device,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    provider = WanLatentProvider(
        args.latent_cache,
        model,
        prompt_embeds,
        device,
        microbatch=args.window_microbatch,
    )
    pos_weight = torch.from_numpy(pos_weight_np).to(device)

    groups = optimizer_parameter_groups(
        model,
        encoder_lr=args.encoder_lora_lr,
        readout_lr=args.readout_lr,
    )
    groups.append({'name': 'classifier', 'params': list(head.parameters()), 'lr': float(args.classifier_lr)})
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=float(args.weight_decay),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
    )
    scheduler = build_constant_with_warmup_scheduler(optimizer, warmup_steps=args.warmup_steps)

    run_config = {
        'version': 'o2laf_motion_annotation_finetune_v1',
        'checkpoint_type': 'finetuned_encoder',
        'mode': args.mode,
        'head': args.head,
        'hidden_dim': int(args.hidden_dim),
        'dropout': float(args.dropout),
        'classes': list(classes),
        'seed': int(args.seed),
        'world_size': int(world_size),
        'video_index': str(args.video_index.resolve()),
        'latent_cache': str(args.latent_cache.resolve()),
        'model_metadata': model_metadata,
        'parameters': counts,
        'initialization_sync': init_sync,
        'trainable_contract': local_contract,
        'train_videos': len(train_rows),
        'validation_videos': len(val_rows),
        'clips_per_rank': int(args.clips_per_rank),
        'window_microbatch': int(args.window_microbatch),
        'max_epochs': int(args.max_epochs),
        'patience': int(args.patience),
        'warmup_steps': int(args.warmup_steps),
        'max_grad_norm': float(args.max_grad_norm),
        'class_weights': weight_details,
        'optimizer_groups': [
            {'name': group['name'], 'lr': group['lr'], 'parameters': sum(p.numel() for p in group['params'])}
            for group in groups
        ],
    }
    if rank == 0:
        write_json_atomic(args.output_dir / 'config.json', run_config)

    resume_config = resume_compatibility_config(run_config)
    history = []
    best_key = None
    stale = 0
    global_step = 0
    start_epoch = 1
    latest_path = args.output_dir / 'latest.pt'

    if latest_path.is_file() and not args.overwrite:
        latest = torch.load(latest_path, map_location='cpu', weights_only=False)
        if latest.get('config') != resume_config:
            raise ValueError(f'Existing {latest_path} has an incompatible config; pass --overwrite to start fresh.')
        load_trainable_state_dict(model, latest['model_trainable_state'])
        head.load_state_dict(latest['head_state'], strict=True)
        optimizer.load_state_dict(latest['optimizer'])
        optimizer_state_to_parameter_devices(optimizer)
        scheduler.load_state_dict(latest['scheduler'])
        broadcast_parameters(model.parameters())
        broadcast_parameters(head.parameters())
        history = latest['history']
        best_key = tuple(latest['best_key']) if latest['best_key'] is not None else None
        stale = int(latest['stale'])
        global_step = int(latest['global_step'])
        start_epoch = int(latest['epoch']) + 1
        if rank == 0:
            print(f'Resumed {latest_path} at epoch={start_epoch} global_step={global_step}', flush=True)

    start_time = time.time()
    all_parameters = list(model.parameters()) + list(head.parameters())
    model.train()
    head.train()

    for epoch in range(start_epoch, args.max_epochs + 1):
        batches = epoch_batches(
            train_rows,
            seed=args.seed,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            clips_per_rank=args.clips_per_rank,
        )
        epoch_losses = []
        epoch_gradients = []
        for step, local_rows in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            records = []
            detached_by_clip = []
            for row in local_rows:
                record = provider.load_record(row)
                segments = provider.detached_segments(record)
                records.append(record)
                detached_by_clip.append(segments)

            local_count = sum(
                sum(1 for task in classes if task in row['labels'] and row['labels'][task] is not None)
                for row in local_rows
            )
            global_count = all_reduce_scalar(local_count, device)
            if global_count <= 0:
                raise RuntimeError('Distributed batch contains no observed labels.')

            local_loss_sum = torch.zeros((), device=device)
            recompute_diff = 0.0
            if local_rows:
                clip_features = [torch.cat(segments, dim=0) for segments in detached_by_clip]
                padded, sequence_mask = pad_feature_sequences(clip_features)
                targets, label_mask = labels_for_rows(local_rows, classes, device)
                logits = head(padded, sequence_mask)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Classifier logits are non-finite.')
                local_loss_sum = masked_bce_sum(logits, targets, label_mask, pos_weight=pos_weight)
                if not torch.isfinite(local_loss_sum):
                    raise RuntimeError('Masked BCE loss is non-finite.')
                (local_loss_sum / global_count).backward()

                verify_recompute = global_step == 0
                for record, segments in zip(records, detached_by_clip):
                    gradients = []
                    for segment in segments:
                        if segment.grad is None:
                            raise RuntimeError('Detached transition segment received no gradient.')
                        gradients.append(segment.grad)
                    recompute_diff = max(
                        recompute_diff,
                        provider.backward_segments(
                            record,
                            gradients,
                            expected=segments if verify_recompute else None,
                        ),
                    )

            all_reduce_gradients(all_parameters)
            group_norms = {group['name']: gradient_norm(group['params']) for group in groups}
            if any(not math.isfinite(v) for v in group_norms.values()):
                raise RuntimeError(f'Non-finite optimizer-group gradient norms: {group_norms}.')
            if local_rows:
                for required_group in ('encoder_lora', 'transition_readout', 'classifier'):
                    if group_norms.get(required_group, 0.0) == 0.0:
                        raise RuntimeError(f'{required_group} received zero gradient.')

            total_norm = float(torch.nn.utils.clip_grad_norm_(all_parameters, float(args.max_grad_norm)))
            if not math.isfinite(total_norm):
                raise RuntimeError(f'Non-finite total gradient norm: {total_norm}.')
            optimizer.step()
            scheduler.step()

            global_loss_sum = all_reduce_scalar(float(local_loss_sum.detach()), device)
            epoch_losses.append(global_loss_sum / global_count)
            epoch_gradients.append(
                {**group_norms, 'total_before_clip': total_norm, 'recompute_max_abs_diff': recompute_diff}
            )
            global_step += 1
            if rank == 0:
                print(
                    f'epoch={epoch} step={step}/{len(batches)} global_step={global_step} '
                    f'loss={epoch_losses[-1]:.6f} grads={group_norms}',
                    flush=True,
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            del records, detached_by_clip
            gc.collect()
            torch.cuda.empty_cache()

        val_metrics, val_predictions = evaluate_validation(
            model,
            head,
            provider,
            val_rows,
            classes=classes,
            rank=rank,
            world_size=world_size,
        )
        row = {
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': float(np.mean(epoch_losses)),
            'gradient_norms': {
                key: float(np.mean([item[key] for item in epoch_gradients]))
                for key in epoch_gradients[0]
            },
            'learning_rates': {group['name']: float(group['lr']) for group in optimizer.param_groups},
            'validation': val_metrics,
        }
        history.append(row)
        key = (val_metrics['macro_average_precision'], val_metrics['macro_roc_auc'], -epoch)
        if best_key is None or key > best_key:
            best_key = key
            stale = 0
            if rank == 0:
                save_checkpoint_atomic(
                    args.output_dir / 'best.pt',
                    {
                        'version': 'o2laf_motion_annotation_finetune_v1',
                        'checkpoint_type': 'finetuned_encoder',
                        'mode': args.mode,
                        'head': args.head,
                        'hidden_dim': int(args.hidden_dim),
                        'dropout': float(args.dropout),
                        'classes': list(classes),
                        'seed': int(args.seed),
                        'epoch': epoch,
                        'model_metadata': model_metadata,
                        'model_trainable_state': trainable_state_dict(model),
                        'head_state': {name: value.detach().cpu() for name, value in head.state_dict().items()},
                        'validation_metrics': val_metrics,
                        'config': resume_config,
                    },
                )
                write_json_atomic(args.output_dir / 'validation_predictions.json', val_predictions)
        else:
            stale += 1

        if rank == 0:
            write_json_atomic(args.output_dir / 'history.json', history)
            save_checkpoint_atomic(
                latest_path,
                {
                    'version': 'o2laf_motion_annotation_latest_v1',
                    'epoch': epoch,
                    'global_step': global_step,
                    'stale': stale,
                    'best_key': best_key,
                    'history': history,
                    'model_trainable_state': trainable_state_dict(model),
                    'head_state': {name: value.detach().cpu() for name, value in head.state_dict().items()},
                    'optimizer': recursive_to_cpu(optimizer.state_dict()),
                    'scheduler': scheduler.state_dict(),
                    'config': resume_config,
                },
            )
            print(
                f'epoch={epoch:03d} train_loss={row["train_loss"]:.5f} '
                f'val_mAP={val_metrics["macro_average_precision"]:.5f} '
                f'val_AUC={val_metrics["macro_roc_auc"]:.5f} stale={stale}',
                flush=True,
            )

        stop = (stale >= args.patience) or (args.max_steps is not None and global_step >= args.max_steps)
        stop_tensor = torch.tensor(int(stop), device=device)
        if dist.is_initialized():
            dist.broadcast(stop_tensor, src=0)
        if int(stop_tensor.item()):
            break

    if rank == 0:
        best_row = max(
            history,
            key=lambda item: (
                item['validation']['macro_average_precision'],
                item['validation']['macro_roc_auc'],
                -item['epoch'],
            ),
        )
        metrics = {
            'version': 'o2laf_motion_annotation_finetune_v1',
            'mode': args.mode,
            'head': args.head,
            'best_epoch': int(best_row['epoch']),
            'best_validation': best_row['validation'],
            'epochs_ran': len(history),
            'global_steps': global_step,
            'elapsed_seconds': time.time() - start_time,
        }
        write_json_atomic(args.output_dir / 'metrics.json', metrics)
        print(json.dumps(metrics, indent=2))
    barrier()
    finish_distributed()


if __name__ == '__main__':
    main()
