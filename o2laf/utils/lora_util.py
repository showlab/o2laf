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

import torch


# PEFT adapter name used for every LoRA in this package. It appears in
# checkpoint keys (``...lora_A.real.weight``), so it cannot change.
LORA_ADAPTER_NAME = 'real'

LORA_STATE_KEYS = ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B', 'lora_magnitude_vector')
LORA_AUXILIARY_STATE_KEYS = ('idm_head', 'action_embedder')


def canonical_lora_config(config):
    """Return the architecture-defining LoRA fields in a comparable form."""
    if config is None:
        return None
    targets = config.get('target_module_name')
    if not isinstance(targets, str):
        targets = tuple(targets)
    return {
        'lora_rank': int(config['lora_rank']),
        'lora_alpha': int(config['lora_alpha']),
        'target_module_name': targets,
    }


def validate_lora_config(configured, checkpoint, tag):
    """Reject loading a checkpoint into a differently shaped LoRA tree."""
    configured = canonical_lora_config(configured)
    checkpoint = canonical_lora_config(checkpoint)
    if configured != checkpoint:
        raise ValueError(
            f'{tag} LoRA config mismatch: checkpoint has {checkpoint}, '
            f'but the YAML requests {configured}. Use the YAML that produced '
            'the checkpoint or start from a fresh experiment directory.'
        )


def filter_learnable_module(model, target_module_name):
    if target_module_name == 'all-linear':
        return target_module_name
    else:
        assert isinstance(target_module_name, list)
        target_modules = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                for target in target_module_name:
                    if target in name:
                        target_modules.add(name)
        return target_modules


def is_peft_model(model):
    return any(hasattr(module, 'peft_config') for module in model.modules())


def get_lora_state_dict(transformer, state_type, ema=None):
    if state_type == 'model':
        return {
            name: param.detach()
            for name, param in transformer.named_parameters()
            if any(key in name for key in LORA_STATE_KEYS + LORA_AUXILIARY_STATE_KEYS)
        }
    else:
        assert state_type == 'ema'
        ema_state_dict = ema.state_dict()
        lora_ema_state_dict = {}

        for name, _ in transformer.named_parameters():
            if not any(key in name for key in LORA_STATE_KEYS + LORA_AUXILIARY_STATE_KEYS):
                continue

            ema_key = name.replace('.', '_')
            if ema_key in ema_state_dict:
                lora_ema_state_dict[name] = ema_state_dict[ema_key]

        return lora_ema_state_dict
