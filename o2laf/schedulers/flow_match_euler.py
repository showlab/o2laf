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
"""Shifted flow-matching schedule, training weights, and the Euler sampler step."""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class FlowMatchEulerSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMatchEulerScheduler(SchedulerMixin, ConfigMixin):
    """Flow matching with ``x_t = (1 - sigma) x_0 + sigma * noise`` and a shifted sigma grid.

    Timesteps are ``sigma * num_train_timesteps``. ``shift`` maps a uniform
    ``t`` to ``sigma = shift * t / (1 + (shift - 1) * t)``.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        weight_type: str = 'gaussian',
    ):
        if weight_type != 'gaussian':
            raise ValueError(f"Only weight_type='gaussian' is supported, got {weight_type!r}.")
        self.set_timesteps(num_train_timesteps, device='cpu')
        self.set_train_weight()

    def set_train_weight(self):
        """Gaussian per-timestep loss weights over the full training grid.

        The training grid is kept separate from ``self.timesteps``, which
        ``set_timesteps`` replaces with a short inference grid during sampling.
        """
        self.train_timesteps = self.timesteps.detach().clone()
        x = self.train_timesteps
        y = torch.exp(-2 * ((x - self.config.num_train_timesteps / 2) / self.config.num_train_timesteps) ** 2)
        y_shifted = y - y.min()
        self.linear_timesteps_weights = y_shifted * (self.config.num_train_timesteps / y_shifted.sum())

    @torch.no_grad()
    def get_train_weight(self, timesteps):
        train_timesteps = self.train_timesteps
        timestep_id = torch.argmin(
            (train_timesteps.unsqueeze(1) - timesteps.flatten().unsqueeze(0).to(train_timesteps.device)).abs(),
            dim=0,
        ).reshape(timesteps.shape)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights.to(timesteps.device)

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        timestep = timestep.float()
        timestep = timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (noise.ndim - timestep.ndim)))
        sample = timestep * noise + (1.0 - timestep) * sample
        return sample.to(noise.dtype)

    def apply_shift(self, sigmas):
        sigmas = sigmas.float()
        if self.config.shift == 1.0:
            return sigmas
        return self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

    def set_timesteps(self, num_inference_steps: Optional[int] = None, device: Union[str, torch.device] = None):
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device)[:-1]
        timesteps = self.apply_shift(timesteps)
        self.timesteps = timesteps * self.config.num_train_timesteps

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.FloatTensor,
        sample: torch.FloatTensor,
        next_timestep: torch.FloatTensor,
        return_dict: bool = True,
    ):
        """One Euler step of the velocity ``model_output`` from ``timestep`` to ``next_timestep``."""
        timestep = timestep.float() / self.config.num_train_timesteps
        next_timestep = next_timestep.float() / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (model_output.ndim - timestep.ndim)))
        next_timestep = next_timestep.view(*next_timestep.shape, *([1] * (model_output.ndim - next_timestep.ndim)))
        prev_sample = sample - (timestep - next_timestep) * model_output
        if not return_dict:
            return (prev_sample.to(model_output.dtype),)
        return FlowMatchEulerSchedulerOutput(prev_sample=prev_sample.to(model_output.dtype))
