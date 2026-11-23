# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import socket
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import split_group
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        return int(probe.getsockname()[1])


def dist_init() -> None:
    """Initialise the default process group, under torchrun or as a one-process run.

    Run without torchrun, this sets up a real one-rank group rather than none,
    so every collective the training code issues behaves the same at one GPU as
    at many. A torchrun job whose rendezvous fails raises instead of carrying on
    as a one-process job, which would change the batch size and typically turn
    the real error into a misleading out-of-memory.
    """
    if is_dist_initialized():
        return
    if 'RANK' not in os.environ:
        os.environ.update(
            RANK='0', LOCAL_RANK='0', WORLD_SIZE='1', LOCAL_WORLD_SIZE='1',
            MASTER_ADDR='127.0.0.1', MASTER_PORT=str(_free_port()),
        )
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    init_method = f'tcp://[{host}]:{port}' if ':' in host else f'tcp://{host}:{port}'

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend='nccl',
        init_method=init_method,
        timeout=timedelta(minutes=30),
        device_id=torch.device('cuda', local_rank),
    )


def fsdp2_wrap(
    module: nn.Module,
    sharding_strategy: str = 'hybrid_full',
    mixed_precision: bool = True,
    transformer_block_clsname: str = None,
    sync_module_state: bool = True
):
    if sync_module_state and dist.is_initialized():
        # Rank 0 may spend many minutes reading a large checkpoint from shared storage
        # while the other ranks construct an empty model.  Synchronize on the
        # 30-minute default process group before DeviceMesh creates subgroups;
        # those subgroups otherwise use PyTorch's shorter default timeout and
        # can expire before rank 0 reaches the first collective.
        dist.barrier()

    if sync_module_state:
        full_state_dict = module.state_dict() if dist.get_rank() == 0 else {}

    # build mixed precision policy
    if mixed_precision:
        mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = MixedPrecisionPolicy()

    # build device mesh
    world_size = int(os.environ['WORLD_SIZE'])
    if 'LOCAL_WORLD_SIZE' not in os.environ and world_size > 1:
        # Defaulting to 1 would treat every GPU as its own node and turn hybrid
        # sharding into plain replication: the job runs, and runs out of memory.
        raise RuntimeError('LOCAL_WORLD_SIZE is not set; launch with torchrun, or set it to the GPUs per node.')
    gpus_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    num_nodes = world_size // gpus_per_node

    if sharding_strategy == 'hybrid_full' or sharding_strategy == 'hybrid_zero2':
        # DeviceMesh normally creates these groups through ncclCommSplit without
        # passing a timeout.  PyTorch then falls back to NCCL's 10-minute
        # default even though the parent process group above is configured for
        # 30 minutes.  A first-step compile/data straggler can therefore kill
        # an otherwise healthy job in mesh_replication while the default group
        # is still well within its timeout.  Create the same two mesh groups
        # explicitly so every collective uses the process-group timeout.
        mesh = torch.arange(world_size, dtype=torch.int).reshape(
            num_nodes, gpus_per_node
        )
        process_group_timeout = timedelta(
            minutes=int(os.environ.get('O2LAF_PROCESS_GROUP_TIMEOUT_MINUTES', '30'))
        )
        replication_group = split_group(
            parent_pg=dist.group.WORLD,
            split_ranks=mesh.transpose(0, 1).tolist(),
            timeout=process_group_timeout,
            group_desc='mesh_replication',
        )
        sharding_group = split_group(
            parent_pg=dist.group.WORLD,
            split_ranks=mesh.tolist(),
            timeout=process_group_timeout,
            group_desc='mesh_sharding',
        )
        if replication_group is None or sharding_group is None:
            raise RuntimeError('current rank is missing from the hybrid FSDP mesh')
        device_mesh = DeviceMesh.from_group(
            [replication_group, sharding_group],
            'cuda',
            mesh=mesh,
            mesh_dim_names=('replication', 'sharding'),
        )
    else:
        device_mesh = init_device_mesh('cuda', (world_size,))

    if transformer_block_clsname is not None:
        for m in module.modules():
            if type(m).__name__ == transformer_block_clsname:
                fully_shard(m, mesh=device_mesh, reshard_after_forward=True, mp_policy=mixed_precision_policy, offload_policy=OffloadPolicy())
    fully_shard(module, mesh=device_mesh, reshard_after_forward=True, mp_policy=mixed_precision_policy, offload_policy=OffloadPolicy())

    if sync_module_state:
        options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
        set_model_state_dict(module, full_state_dict, options=options)

    return module


def dist_barrier():
    if dist.is_initialized():
        dist.barrier()


def get_dist_rank() -> int:
    return int(os.environ.get('RANK', 0))


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def is_dist_initialized() -> bool:
    return torch.distributed.is_initialized()


def destroy_process_group():
    if is_dist_initialized():
        dist_barrier()
        torch.distributed.destroy_process_group()


def exit_failed_rank(code: int = 1) -> None:
    """End this process now, without shutting down the process group.

    Shutting down an NCCL group waits for every peer, and a peer still inside a
    collective never arrives: a rank that failed alone would hang there until the
    NCCL timeout. Exiting lets torchrun see the failure and stop the other ranks.
    """
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def reduce_loss(loss):
    """Mean of ``loss`` over all ranks.

    Returns the local value when no process group exists, which is the identity
    at one rank and is what makes the diagnostics testable. The latent-action
    CPU suite exercises ``_diagnostics`` directly against a stub, and an
    unconditional ``all_reduce`` put every metric in that path -- KL shares,
    ``action_const_share``, the SNR split -- behind a live cluster, so the tests
    written for them could not run at all.
    """
    rt = loss.detach().clone()
    if not is_dist_initialized():
        return rt.item()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt.item()


def raise_if_any_rank_failed(error: str | None, device, message: str) -> None:
    """All-reduce a per-rank failure flag and raise the same error on every rank.

    Validation that only rank 0 can run (it alone holds a loaded file) must not
    leave the other ranks waiting in their next collective.
    """
    from o2laf.utils.logger_util import get_logger  # logger_util imports this module

    if error is not None:
        get_logger().error('%s: %s', message, error)
    success = torch.tensor(float(error is None), device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if not bool(success.item()):
        raise RuntimeError(f'{message}; see the error logged by the failing rank.')


def load_full_state_dict(module, source, *, required, label, source_name, device, unexpected='raise') -> None:
    """Load a full (unsharded) state dict into an FSDP2 module from rank 0, checking keys first.

    ``set_model_state_dict(..., full_state_dict=True, strict=False)`` reports no
    missing keys, so absent tensors are detected here instead, by comparing
    ``required`` (parameter names that must be restored) with ``source``'s keys
    on rank 0. Other ranks may pass an empty ``source``. ``unexpected`` is
    ``'raise'``, ``'warn'`` or ``'ignore'`` for source keys the module lacks.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    error = None
    if rank == 0:
        names = set(source)
        absent = sorted(set(required) - names)
        known = {name for name, _ in module.named_parameters()} | {name for name, _ in module.named_buffers()}
        extra = sorted(names - known)
        if absent:
            error = f'{len(absent)} required {label} tensors are absent from {source_name}, e.g. {absent[:3]}.'
        elif extra and unexpected == 'raise':
            error = f'{len(extra)} unexpected {label} tensors in {source_name}, e.g. {extra[:3]}.'
        elif extra and unexpected == 'warn':
            from o2laf.utils.logger_util import get_logger  # logger_util imports this module

            get_logger().warning(f'{len(extra)} unexpected {label} keys in {source_name}, e.g. {extra[:3]}.')
    raise_if_any_rank_failed(error, device, f'Could not load the {label} state')
    set_model_state_dict(
        module,
        source if rank == 0 else {},
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False),
    )
