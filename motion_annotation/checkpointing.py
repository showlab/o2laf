"""Atomic JSON/checkpoint IO, reproducibility seeding, and LR scheduling."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def write_json_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(target)


def save_checkpoint_atomic(path: str | Path, payload: dict) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(target)


def recursive_to_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(item) for item in value)
    return value


def optimizer_state_to_parameter_devices(optimizer) -> None:
    import torch

    for parameter, state in optimizer.state.items():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor) and value.device != parameter.device:
                state[key] = value.to(parameter.device)


def build_constant_with_warmup_scheduler(optimizer, *, warmup_steps: int):
    import torch

    steps = int(warmup_steps)
    if steps < 0:
        raise ValueError(f'warmup_steps must be non-negative, got {steps}.')

    def multiplier(current_step: int) -> float:
        if steps == 0:
            return 1.0
        return min(1.0, float(current_step + 1) / float(steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def resume_compatibility_config(run_config: dict) -> dict:
    """Drop runtime diagnostics that should not invalidate checkpoint resume."""
    result = dict(run_config)
    result.pop('initialization_sync', None)
    return result
