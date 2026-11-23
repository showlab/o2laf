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

import dataclasses
import datetime
import logging
import os
import os.path
import os.path as osp
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

from .dist_util import is_main_process


def _format_cuda_memory() -> str:
    if not torch.cuda.is_available():
        return ''
    d = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(d) / (1024**3)
    reserved = torch.cuda.memory_reserved(d) / (1024**3)
    peak = torch.cuda.max_memory_allocated(d) / (1024**3)
    return f'[cuda:{d} alloc {alloc:.2f}GiB reserved {reserved:.2f}GiB peak_alloc {peak:.2f}GiB] '


def _format_log_value(value) -> str:
    if abs(value) < 1e-4:
        return f'{value:.3e}'
    return f'{value:.4f}'


# ----------- file/logger util ----------
def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def make_exp_dirs(cfg):
    """Make dirs for experiments."""
    path_opt = cfg['path'].copy()
    # An existing experiment directory is reused as it is: that is how a run resumes.
    if cfg['mode'] == 'train':
        os.makedirs(path_opt.pop('experiments_root'), exist_ok=True)
    else:
        os.makedirs(path_opt.pop('results_root'), exist_ok=True)
    for key, path in path_opt.items():
        if ('strict_load' in key) or ('pretrain_network' in key) or ('resume' in key) or ('param_key' in key) or ('lora_path' in key):
            continue
        else:
            os.makedirs(path, exist_ok=True)


def copy_cfg_file(cfg, experiments_root):
    """Write the config the run actually uses -- command-line overrides applied -- next to its outputs."""
    import sys

    import yaml

    filename = osp.join(experiments_root, osp.basename(cfg['config_path']))
    with open(filename, 'w', encoding='utf-8') as handle:
        handle.write(f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {" ".join(sys.argv)}\n'
                     '# The resolved config, with every command-line override applied.\n\n')
        yaml.safe_dump(cfg, handle, sort_keys=False)


def set_path_logger(cfg):

    if 'path' not in cfg:
        cfg['path'] = {}

    if cfg['mode'] == 'train':
        experiments_root = cfg['path'].get(
            'experiments_root', osp.join('experiments', cfg['name'])
        )
        cfg['path']['experiments_root'] = experiments_root
        cfg['path']['models'] = osp.join(experiments_root, 'models')
        cfg['path']['log'] = experiments_root
        cfg['path']['visualization'] = osp.join(experiments_root, 'visualization')
    else:
        results_root = osp.join('results', cfg['name'])
        cfg['path']['results_root'] = results_root
        cfg['path']['log'] = results_root
        cfg['path']['visualization'] = osp.join(results_root, 'visualization')

    # Handle the output folder creation
    if is_main_process():
        make_exp_dirs(cfg)

        if cfg['mode'] == 'train':
            copy_cfg_file(cfg, cfg['path']['experiments_root'])
            log_file = osp.join(cfg['path']['log'], f"train_{cfg['name']}_{get_time_str()}.log")
            set_logger(log_file)
        else:
            copy_cfg_file(cfg, cfg['path']['results_root'])
            log_file = osp.join(cfg['path']['log'], f"test_{cfg['name']}_{get_time_str()}.log")
            set_logger(log_file)
    else:
        get_logger().propagate = False
        get_logger().setLevel(logging.ERROR)
    dist.barrier()


def get_logger():
    logger = logging.getLogger('o2laf')
    return logger


def set_logger(log_file=None):
    # Make one log on every process with the configuration for debugging.
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    log_level = logging.INFO

    logger = get_logger()
    logger.propagate = False
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter(format_str))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))

    logger.addHandler(stream_handler)


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """
    def __init__(self, opt, start_iter=1):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.start_time = time.time()

    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        current_iter = log_vars.pop('iter')

        message = f'[{self.exp_name}][Iter:{current_iter:8,d}] '

        # time and estimated time
        total_time = time.time() - self.start_time
        time_sec_avg = total_time / (current_iter - self.start_iter + 1)
        eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        message += f'[eta: {eta_str}] '

        # other items, especially losses
        for k, v in log_vars.items():
            if 'lr' in k:
                message += f'{k}: {v:.3e} '
            else:
                message += f'{k}: {_format_log_value(v)} '

        message += _format_cuda_memory()
        get_logger().info(message)


def setup_wandb(name, save_dir):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            'You are trying to use wandb which is not currently installed. '
            'Please install it using pip install wandb'
        )
    wandb_config = WANDB_CONFIG()
    wandb_config.name = name
    wandb_config.dir = save_dir

    # auto resume
    wandb_latest_dir = os.path.join(save_dir, 'wandb', 'latest-run')

    if os.path.exists(wandb_latest_dir):
        run_id = os.readlink(wandb_latest_dir).split('-')[-1]
        wandb_config.id = run_id

    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    return run


@dataclass
class WANDB_CONFIG:
    project: str = 'o2laf'
    entity: Optional[str] = None
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
    name: Optional[str] = None
    dir: Optional[str] = None
    resume: str = 'allow'
    id: Optional[str] = None
