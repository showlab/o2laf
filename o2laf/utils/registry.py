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

"""Small name-to-object registries used by AnyFlow-style builders."""


class Registry:
    """Map stable configuration names to Python objects."""

    def __init__(self, name):
        self._name = str(name)
        self._obj_map = {}

    @property
    def name(self):
        return self._name

    def _do_register(self, name, obj):
        if name in self._obj_map:
            raise KeyError(f"An object named '{name}' is already registered in '{self._name}'.")
        self._obj_map[name] = obj

    def register(self, obj=None):
        """Register an object directly or use this method as a decorator."""

        if obj is None:

            def decorator(function_or_class):
                self._do_register(function_or_class.__name__, function_or_class)
                return function_or_class

            return decorator

        self._do_register(obj.__name__, obj)
        return obj

    def get(self, name):
        if name not in self._obj_map:
            available = ", ".join(sorted(self._obj_map)) or "<empty>"
            raise KeyError(f"No object named '{name}' is registered in '{self._name}'. Available: {available}.")
        return self._obj_map[name]

    def keys(self):
        return self._obj_map.keys()

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def __len__(self):
        return len(self._obj_map)


TRAINER_REGISTRY = Registry("trainers")
ACTION_ENCODER_REGISTRY = Registry("action_encoders")
WORLD_MODEL_REGISTRY = Registry("world_models")
DATASET_REGISTRY = Registry("datasets")
