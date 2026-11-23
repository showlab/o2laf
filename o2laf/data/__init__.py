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
"""Datasets and the distributed dataloader."""

from copy import deepcopy

from o2laf.data import video_tar_dataset  # noqa: F401  (registration)
from o2laf.data.dataloader import BaseDataLoader
from o2laf.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']


def build_dataset(dataset_opt):
    """Build the dataset named by ``dataset_opt['type']`` from the remaining keys."""
    dataset_opt = deepcopy(dataset_opt)
    dataset_type = dataset_opt.pop('type')
    return DATASET_REGISTRY.get(dataset_type)(**dataset_opt)


def build_dataloader(dataset_opt):
    dataset_opt = deepcopy(dataset_opt)
    dataloader_cfg = dataset_opt.pop('dataloader_cfg')
    return BaseDataLoader(build_dataset(dataset_opt), dataloader_cfg)
