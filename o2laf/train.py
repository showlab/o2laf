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
"""Training entry point.

    torchrun --nproc_per_node=8 -m o2laf.train config_path=configs/latent_action/wan_dit_mira.yaml

Any config key can be overridden on the command line, e.g. ``train.total_iter=100``.
"""

import gc
import os
import random
import shutil
import time
import traceback

import numpy as np
import torch
import torch.distributed
import torch.utils.checkpoint
from omegaconf import OmegaConf
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from transformers import get_constant_schedule_with_warmup

from o2laf.data import build_dataloader
from o2laf.training import build_trainer
from o2laf.utils.dist_util import (
    destroy_process_group,
    dist_barrier,
    dist_init,
    exit_failed_rank,
    get_dist_rank,
    get_world_size,
    is_main_process,
)
from o2laf.utils.logger_util import MessageLogger, dict2str, get_logger, set_path_logger, setup_wandb
from o2laf.utils.lora_util import get_lora_state_dict, is_peft_model

# A default, not an override: read when CUDA first allocates, so an explicit
# setting in the environment still wins.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,garbage_collection_threshold:0.8')


class TrainingRunner:
    """Owns the loop: data, optimizer, LR schedule, EMA, logging, evaluation and checkpoints.

    What is optimised -- models, loss, evaluation -- is the trainer selected by
    ``train.trainer`` (see :mod:`o2laf.training`).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.setup_dist_env()
        self.setup_seed()

        set_path_logger(self.cfg)
        get_logger().info(dict2str(self.cfg))

        self.trainer = build_trainer(self.cfg['train']['trainer'])(
            **self.cfg['models'],
            device=self.device,
            dtype=self.dtype,
        )
        self.wandb_logger = None
        self.global_step = 'final'

        if self.cfg['mode'] == 'train':
            self.setup_wandb()
            self.setup_ema_model()
            self.train_dataloader = build_dataloader(self.cfg['datasets']['train'])
            self.setup_optimizer()
            self.setup_lr_scheduler()
            self.global_step = 0
            self.resume_checkpoint()
            self.msg_logger = MessageLogger(self.cfg, self.global_step)

    def evaluate(self):
        """Run the trainer's fixed-clip evaluation, if one is configured."""
        if (self.cfg.get('val') or {}).get('latent_action_eval'):
            self.trainer.sample(self.cfg, global_step=self.global_step)
        self.free_memory()

    def train(self):
        get_logger().info('***** Running training *****')
        get_logger().info(f'  Num examples = {len(self.train_dataloader.dataset)}')
        get_logger().info(f'  GPUs = {get_world_size()}')
        get_logger().info(f'  Total train batch size = {self.train_dataloader.cfg.global_batch_size}')
        get_logger().info(f"  Total optimization steps = {self.cfg['train']['total_iter']}")

        total_iter = self.cfg['train']['total_iter']
        if self.global_step < total_iter:
            self.init_train_dataloader()
        else:
            get_logger().info('Checkpoint step %d already reached total_iter=%d.', self.global_step, total_iter)

        while self.global_step < total_iter:
            if self.global_step != 0 and self.global_step % self.cfg['logger']['save_eval_checkpoint_freq'] == 0:
                self.save_checkpoint(f'step_{self.global_step}', only_model_state_dict=True)

            if self.global_step % self.cfg['val']['sample_freq'] == 0 and not (
                self.global_step == 0 and self.cfg['val'].get('skip_first_eval', False)
            ):
                self.evaluate()

            step_begin_time = time.time()
            batch = self.train_dataloader.get_next_batch()
            self.optimizer.zero_grad(set_to_none=True)
            loss_dict = self.trainer.train_step(batch, iters=self.global_step)
            if 'max_grad_norm' in self.cfg['train']:
                loss_dict['world_model_grad_norm'] = torch.nn.utils.clip_grad_norm_(
                    self.trainer.world_model.parameters(), max_norm=self.cfg['train']['max_grad_norm']
                ).item()
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.cfg['train'].get('ema_decay'):
                self.trainer.ema.step(self.trainer.world_model, step=self.global_step)
            log_dict = {'step_time': time.time() - step_begin_time}

            if self.global_step % self.cfg['logger']['print_freq'] == 0:
                log_dict.update({'iter': self.global_step, 'lr': self.lr_scheduler.get_last_lr()[0]})
                log_dict.update(loss_dict)
                self.msg_logger(log_dict)
                if is_main_process() and self.wandb_logger:
                    self.wandb_logger.log(
                        {f'train/{k}': v for k, v in log_dict.items()}, step=self.global_step, commit=True
                    )

            self.global_step += 1
            if self.global_step % self.cfg['logger']['save_checkpoint_freq'] == 0:
                self.save_checkpoint('checkpoint')
            if self.global_step % self.cfg['logger'].get('clean_cache_freq', 100) == 0:
                self.free_memory()

        # Evaluation and the periodic export run before an optimizer step, so
        # the final step needs an explicit opt-in.
        if self.cfg['val'].get('sample_at_end', False):
            self.evaluate()
        if (
            self.cfg['logger'].get('save_eval_checkpoint_at_end', False)
            and self.global_step != 0
            and self.global_step % self.cfg['logger']['save_eval_checkpoint_freq'] == 0
        ):
            self.save_checkpoint(f'step_{self.global_step}', only_model_state_dict=True)

    def setup_optimizer(self):
        optimizer_cfg = dict(self.cfg['train']['optimizer'])
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type != 'AdamW':
            raise ValueError(f'Unsupported optimizer type: {optimizer_type}')
        self.optimizer = torch.optim.AdamW(self.trainer.get_optimizer_param_groups(), **optimizer_cfg)

    def setup_lr_scheduler(self):
        if self.cfg['train']['lr_scheduler'] != 'constant_with_warmup':
            raise NotImplementedError(self.cfg['train']['lr_scheduler'])
        self.lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=self.cfg['train']['warmup_iter']
        )

    def init_train_dataloader(self):
        data_loader_length = len(self.train_dataloader.data_loader)
        get_logger().info(f'Data_loader_length: {data_loader_length}')
        epoch = self.global_step // data_loader_length
        batch_iter = self.global_step - data_loader_length * epoch
        self.train_dataloader.set_state(epoch, batch_iter)
        get_logger().info(f'Init dataloader at epoch: {epoch}, batch_iter {batch_iter}')

    def setup_wandb(self):
        if is_main_process() and self.cfg['logger'].get('use_wandb', False):
            self.wandb_logger = setup_wandb(name=self.cfg['name'], save_dir=self.cfg['path']['log'])

    def setup_ema_model(self):
        if self.cfg['train'].get('ema_decay'):
            self.trainer.set_ema_model(
                decay=self.cfg['train']['ema_decay'],
                warmup_step=self.cfg['train'].get('ema_warmup_step', 0),
                decay_type=self.cfg['train'].get('ema_type', 'fixed'),
            )

    def setup_dist_env(self):
        dist_init()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.dtype = torch.bfloat16 if self.cfg['mixed_precision'] else torch.float32
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.rank, self.world_size = get_dist_rank(), get_world_size()

    def setup_seed(self) -> None:
        seed = self.rank + self.cfg['manual_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def get_random_states(self):
        """Every rank's Python/NumPy/Torch RNG states, gathered for an exact resume."""
        numpy_random_state = np.random.get_state()
        numpy_random_state = (
            numpy_random_state[0],
            numpy_random_state[1].tolist(),
            numpy_random_state[2],
            numpy_random_state[3],
            numpy_random_state[4],
        )
        random_states_single_rank = {
            'random_state': random.getstate(),
            'numpy_random_state': numpy_random_state,
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state(),
        }
        if not torch.distributed.is_initialized():
            return {f'rank_{self.rank}': random_states_single_rank}
        gathered = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, random_states_single_rank)
        return {f'rank_{rank}': state for rank, state in enumerate(gathered)}

    def set_random_states(self, random_states):
        """Restore this rank's RNG state; report whether it was available."""
        key = f'rank_{self.rank}'
        if key not in random_states:
            get_logger().warning(f'Checkpoint has no RNG state for {key}; keeping the seeded startup state.')
            return False
        state = random_states[key]
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_random_state'])
        torch.set_rng_state(state['torch_rng_state'])
        torch.cuda.set_rng_state(state['torch_cuda_rng_state'])
        return True

    def uses_lightweight_checkpoint(self):
        """Whether the trainer saves its own (portable) training state instead of the generic DCP one."""
        if getattr(self.trainer, 'disable_lightweight_checkpoint', False):
            return False
        if not hasattr(self.trainer, 'get_lightweight_train_state'):
            return False
        return self.world_size == 1 or getattr(self.trainer, 'lightweight_checkpoint_is_distributed', False)

    def get_train_state(self, only_model_state_dict: bool = False):
        """The generic distributed training state of ``trainer.world_model``."""
        world_model = self.trainer.world_model
        train_states = {'global_step': self.global_step, 'world_size': self.world_size}
        if self.cfg['train'].get('ema_decay'):
            if only_model_state_dict:
                if is_peft_model(world_model):
                    train_states['ema'] = get_lora_state_dict(world_model, 'ema', self.trainer.ema)
                else:
                    train_states['ema'] = self.trainer.ema.single_state_dict(world_model)
            else:
                train_states['ema'] = self.trainer.ema.state_dict()
        if only_model_state_dict:
            if is_peft_model(world_model):
                train_states['model'] = get_lora_state_dict(world_model, 'model')
            else:
                train_states['model'] = get_model_state_dict(world_model)
            return train_states
        train_states['model'], train_states['optimizer'] = get_state_dict(world_model, self.optimizer)
        train_states['lr_scheduler'] = self.lr_scheduler.state_dict()
        train_states['random_states'] = self.get_random_states()
        return train_states

    def resume_checkpoint(self):
        """Resume from the last checkpoint under ``path.models``, if any."""
        load_path = self.cfg['path'].get('pretrain_network') or self.cfg['path']['models']
        checkpoint_path = os.path.join(load_path, 'checkpoint')
        lightweight_path = os.path.join(load_path, 'lightweight_checkpoint.pt')
        if self.uses_lightweight_checkpoint() and os.path.exists(lightweight_path):
            get_logger().info(f'Resuming from lightweight checkpoint {lightweight_path}')
            state = torch.load(lightweight_path, map_location='cpu', weights_only=False)
            self.trainer.load_lightweight_train_state(state, self.optimizer, self.lr_scheduler)
            self.global_step = int(state['global_step'])
            rng_restored = False
            if self.world_size == state['world_size']:
                rng_restored = self.set_random_states(state['random_states'])
            else:
                get_logger().warning(
                    'Checkpoint world size %s differs from %s; keeping freshly seeded RNG states.',
                    state['world_size'], self.world_size,
                )
            get_logger().info('Resumed at step %d; RNG restored=%s.', self.global_step, rng_restored)
            dist_barrier()
            return
        if not os.path.exists(checkpoint_path):
            get_logger().info('No checkpoint found. Starting a new training run.')
            return

        get_logger().info(f'Resuming from checkpoint {checkpoint_path}')
        train_states = self.get_train_state()
        torch.distributed.checkpoint.load(train_states, checkpoint_id=checkpoint_path)
        self.global_step = train_states['global_step']
        set_state_dict(
            self.trainer.world_model,
            self.optimizer,
            model_state_dict=train_states['model'],
            optim_state_dict=train_states['optimizer'],
            options=StateDictOptions(strict=True),
        )
        if self.cfg['train'].get('ema_decay'):
            self.trainer.ema.load_state_dict(train_states['ema'])
        self.lr_scheduler.load_state_dict(train_states['lr_scheduler'])
        if self.world_size == train_states['world_size']:
            self.set_random_states(train_states['random_states'])
        else:
            get_logger().warning(
                f"Checkpoint world size {train_states['world_size']} differs from {self.world_size}; "
                'keeping freshly seeded RNG states.'
            )
        get_logger().info(f'Resumed model, optimizer, EMA and LR schedule at step {self.global_step}.')
        dist_barrier()

    def save_checkpoint(self, model_name, only_model_state_dict=False):
        """Save the rolling training checkpoint (``checkpoint``) or a model export (``step_N``)."""
        output_dir = self.cfg['path']['models']
        self.free_memory()

        if model_name == 'checkpoint':
            assert not only_model_state_dict, 'the training checkpoint saves the optimizer too'
            if self.uses_lightweight_checkpoint():
                lightweight_path = os.path.join(output_dir, 'lightweight_checkpoint.pt')
                tmp_path = lightweight_path + '.tmp'
                # Every rank calls this: the state gather is collective.
                state = self.trainer.get_lightweight_train_state(
                    self.optimizer, self.lr_scheduler, self.global_step, self.world_size, self.get_random_states()
                )
                if is_main_process():
                    assert state is not None, 'rank 0 must receive the lightweight state'
                    torch.save(state, tmp_path)
                    os.replace(tmp_path, lightweight_path)
                    get_logger().info(f'Saved lightweight training checkpoint to {lightweight_path}')
                dist_barrier()
                return
        elif self.uses_lightweight_checkpoint():
            # A model export: get_train_state covers trainer.world_model only,
            # which for latent action learning would drop the action encoder.
            # Export the portable state, both halves, instead.
            state = self.trainer.export_latent_action_state()  # collective: every rank
            if is_main_process():
                state['global_step'] = int(self.global_step)
                export_path = os.path.join(output_dir, f'{model_name}_latent_action_state.pt')
                torch.save(state, export_path + '.tmp')
                os.replace(export_path + '.tmp', export_path)
                get_logger().info(f'Saved latent-action state to {export_path}')
            dist_barrier()
            return

        train_states = self.get_train_state(only_model_state_dict=only_model_state_dict)

        if model_name == 'checkpoint':
            save_path_ = os.path.join(output_dir, 'checkpoint_')
            save_path = os.path.join(output_dir, 'checkpoint')
            if is_main_process():
                shutil.rmtree(save_path_, ignore_errors=True)
            dist_barrier()
            torch.distributed.checkpoint.save(train_states, checkpoint_id=save_path_)
            dist_barrier()
            if is_main_process():
                shutil.rmtree(save_path, ignore_errors=True)
                shutil.move(save_path_, save_path)
            dist_barrier()
            get_logger().info(f'Saved state to {save_path}')
        else:
            assert only_model_state_dict, 'a model export saves the model state only'
            dist_barrier()
            suffix = '_lora' if is_peft_model(self.trainer.world_model) else ''
            dcp_path = os.path.join(output_dir, f'{model_name}{suffix}_dcp')
            export_path = os.path.join(output_dir, f'{model_name}{suffix}.pt')
            tmp_export_path = export_path + '.tmp'
            if is_main_process():
                shutil.rmtree(dcp_path, ignore_errors=True)
                if os.path.exists(tmp_export_path):
                    os.remove(tmp_export_path)
            dist_barrier()
            torch.distributed.checkpoint.save(train_states, checkpoint_id=dcp_path)
            dist_barrier()
            del train_states
            self.free_memory()
            if is_main_process():
                dcp_to_torch_save(dcp_path, tmp_export_path)
                os.replace(tmp_export_path, export_path)
                shutil.rmtree(dcp_path, ignore_errors=True)
                get_logger().info(f'Saved model export to {export_path}')
            dist_barrier()
            return

        del train_states
        self.free_memory()

    def free_memory(self):
        dist_barrier()
        torch.cuda.empty_cache()
        gc.collect()
        dist_barrier()


USAGE = """usage: o2laf-train config_path=<config.yaml> [key.path=value ...]

Trains with the config at config_path; any further key=value pair overrides that
config entry, e.g. train.total_iter=1000 or datasets.train.meta_path=/data/meta.json.
Launch with torchrun for more than one GPU:

    torchrun --nproc_per_node=8 -m o2laf.train config_path=configs/latent_action/wan_dit_mira.yaml
"""


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if not argv or any(arg in ('-h', '--help') for arg in argv):
        print(USAGE, end='')
        raise SystemExit(0 if argv else 2)
    cli_cfg = OmegaConf.from_dotlist(argv)
    if 'config_path' not in cli_cfg:
        print(USAGE, end='')
        raise SystemExit('config_path=<config.yaml> is required.')
    cfg = OmegaConf.merge(OmegaConf.load(cli_cfg.config_path), cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        runner = TrainingRunner(cfg)
        if cfg['mode'] == 'train':
            runner.train()
        else:
            runner.evaluate()
    except Exception:
        if get_world_size() > 1:
            # A rank that fails alone must not wait for the others: they are
            # blocked in their next collective, so a barrier -- or even shutting
            # the process group down -- would hang until the NCCL timeout and
            # hold back this traceback. Report it and exit; torchrun stops the rest.
            traceback.print_exc()
            exit_failed_rank()
        destroy_process_group()
        raise
    destroy_process_group()


if __name__ == '__main__':
    main()
