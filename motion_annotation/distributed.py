"""Lightweight explicit data-parallel helpers for recompute-based fine-tuning."""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import timedelta

import torch
import torch.distributed as dist


def initialize_distributed(timeout_hours: int = 2) -> tuple[int, int, int]:
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', timeout=timedelta(hours=timeout_hours))
    return rank, world_size, local_rank


def broadcast_parameters(
    parameters: Iterable[torch.nn.Parameter],
    *,
    source: int = 0,
    bucket_size_mb: int = 128,
) -> None:
    """Broadcast trainable parameters from rank ``source`` to all ranks."""
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not dist.is_initialized():
        return
    grouped: dict[tuple[torch.device, torch.dtype], list[torch.nn.Parameter]] = {}
    for parameter in params:
        grouped.setdefault((parameter.device, parameter.dtype), []).append(parameter)
    limit = int(bucket_size_mb) * 1024 * 1024
    for group in grouped.values():
        bucket: list[torch.nn.Parameter] = []
        size = 0
        for parameter in group:
            bytes_count = parameter.numel() * parameter.element_size()
            if bucket and size + bytes_count > limit:
                _broadcast_bucket(bucket, source=source)
                bucket = []
                size = 0
            bucket.append(parameter)
            size += bytes_count
        if bucket:
            _broadcast_bucket(bucket, source=source)


def _broadcast_bucket(parameters: list[torch.nn.Parameter], *, source: int) -> None:
    values = [parameter.detach() for parameter in parameters]
    flat = torch._utils._flatten_dense_tensors(values)
    dist.broadcast(flat, src=int(source))
    for parameter, broadcasted in zip(parameters, torch._utils._unflatten_dense_tensors(flat, values)):
        parameter.data.copy_(broadcasted)


def all_reduce_scalar(value: float | int, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_gradients(
    parameters: Iterable[torch.nn.Parameter],
    *,
    bucket_size_mb: int = 128,
) -> None:
    """Sum gradients across ranks in dense dtype/device-compatible buckets."""
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not dist.is_initialized():
        return
    grouped: dict[tuple[torch.device, torch.dtype], list[torch.nn.Parameter]] = {}
    for parameter in params:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        grouped.setdefault((parameter.device, parameter.grad.dtype), []).append(parameter)
    limit = int(bucket_size_mb) * 1024 * 1024
    for group in grouped.values():
        bucket: list[torch.nn.Parameter] = []
        size = 0
        for parameter in group:
            gradient = parameter.grad
            bytes_count = gradient.numel() * gradient.element_size()
            if bucket and size + bytes_count > limit:
                _reduce_bucket(bucket)
                bucket = []
                size = 0
            bucket.append(parameter)
            size += bytes_count
        if bucket:
            _reduce_bucket(bucket)


def _reduce_bucket(parameters: list[torch.nn.Parameter]) -> None:
    gradients = [parameter.grad for parameter in parameters]
    flat = torch._utils._flatten_dense_tensors(gradients)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    for parameter, reduced in zip(parameters, torch._utils._unflatten_dense_tensors(flat, gradients)):
        parameter.grad.copy_(reduced)


def all_gather_objects(value):
    if not dist.is_initialized():
        return [value]
    result = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(result, value)
    return result


def assert_parameters_synchronized(
    parameters: Iterable[torch.nn.Parameter],
    *,
    tag: str,
    atol: float = 1.0e-6,
) -> dict[str, float]:
    """Verify identical trainable weights across all ranks."""
    params = [parameter for parameter in parameters if parameter.requires_grad]
    count = sum(parameter.numel() for parameter in params)
    checksum = sum(float(parameter.detach().double().sum().item()) for parameter in params)
    square_checksum = sum(float(parameter.detach().double().square().sum().item()) for parameter in params)
    summary = {
        'parameter_count': int(count),
        'checksum': checksum,
        'square_checksum': square_checksum,
    }
    if not dist.is_initialized():
        return summary
    device = params[0].device if params else torch.device('cuda')
    values = torch.tensor([float(count), checksum, square_checksum], device=device, dtype=torch.float64)
    minimum = values.clone()
    maximum = values.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    diffs = (maximum - minimum).abs().cpu().tolist()
    if diffs[0] != 0.0 or diffs[1] > atol or diffs[2] > atol:
        raise RuntimeError(f'{tag} parameters differ across ranks: diffs={diffs}.')
    return summary


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def finish_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
