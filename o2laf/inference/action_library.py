"""Named atomic actions shipped with a model.

A library is a ``<name>.json`` record and a ``<name>.safetensors`` file of
arrays in the model's ``action_library/`` directory. Each action is one group of
``codes_per_transition`` codes, meant to be repeated across every transition of
a generation ("walk forward", "turn left", ...).

A library is bound to the action space it was extracted in, and refuses to load
into a model whose action encoder defines a different one: the same code means
a different motion under a different encoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

LIBRARY_SCHEMA = 'o2laf_atomic_action_library_v1'


@dataclass(frozen=True)
class AtomicAction:
    name: str
    values: torch.Tensor  # [codes_per_transition, action_dim]
    support: int | None
    metadata: dict[str, Any]


class AtomicActionLibrary:
    """The atomic actions of one library, checked against the model they drive."""

    def __init__(self, metadata_path: str | Path, *, action_space_id: str, action_dim: int, codes_per_transition: int):
        metadata_path = Path(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('schema_version') != LIBRARY_SCHEMA:
            raise ValueError(f'{metadata_path} has schema {metadata.get("schema_version")!r}, not {LIBRARY_SCHEMA!r}.')
        if metadata.get('action_space_id') != action_space_id:
            raise RuntimeError(
                f'{metadata_path} was extracted in action space {metadata.get("action_space_id")}, '
                f'but this model is {action_space_id}.'
            )
        arrays = load_file(str(metadata_path.with_name(metadata['weights'])))
        names = list(metadata['names'])
        actions = arrays['actions'].float()
        if len(set(names)) != len(names):
            raise ValueError(f'{metadata_path} repeats action names.')
        if actions.ndim == 2:
            actions = actions[:, None, :]
        expected = (len(names), int(codes_per_transition), int(action_dim))
        if tuple(actions.shape) != expected:
            raise ValueError(f'{metadata_path} actions are {tuple(actions.shape)}, expected {expected}.')
        if not torch.isfinite(actions).all():
            raise ValueError(f'{metadata_path} actions contain NaN or infinity.')
        support = arrays.get('support')
        classes = {entry['name']: entry for entry in metadata.get('classes', []) if isinstance(entry, dict)}
        self.actions = {
            name: AtomicAction(
                name=name,
                values=actions[index].clone(),
                support=int(support[index]) if support is not None else None,
                metadata=classes.get(name, {}),
            )
            for index, name in enumerate(names)
        }
        self.metadata = metadata
        self.metadata_path = metadata_path
        self.action_dim = int(action_dim)
        self.codes_per_transition = int(codes_per_transition)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, index: dict[str, Any], name: str | None = None) -> 'AtomicActionLibrary':
        """A library shipped with a model: ``name``, or the first one its index lists.

        Needs only the index and the library files, not the model's weights.
        """
        available = library_names(index)
        if not available:
            raise FileNotFoundError(f'{model_dir} ships no action library.')
        name = name or available[0]
        if name not in available:
            raise KeyError(f'Unknown action library {name!r}; this model ships {available}.')
        return cls(
            Path(model_dir) / index['action_library']['directory'] / f'{name}.json',
            action_space_id=index['action_space_id'],
            action_dim=int(index['action_dim']),
            codes_per_transition=int(index['codes_per_transition']),
        )

    @classmethod
    def from_pipeline(cls, pipeline, name: str | None = None) -> 'AtomicActionLibrary':
        """A library shipped with ``pipeline``'s model; see :meth:`from_model_dir`."""
        return cls.from_model_dir(pipeline.model_dir, pipeline.model_index, name)

    @property
    def name(self) -> str:
        return self.metadata_path.stem

    def names(self) -> tuple[str, ...]:
        return tuple(self.actions)

    def get(self, name: str) -> AtomicAction:
        if name not in self.actions:
            raise KeyError(f'Unknown action {name!r}; available: {list(self.actions)}.')
        return self.actions[name]

    def tensor(self, name: str, *, groups: int, device, dtype) -> torch.Tensor:
        """``[1, groups * codes_per_transition, action_dim]``: one action repeated over every transition."""
        if groups < 1:
            raise ValueError('groups must be positive.')
        values = self.get(name).values.repeat(groups, 1)
        return values.reshape(1, groups * self.codes_per_transition, self.action_dim).to(device=device, dtype=dtype)


def library_names(index: dict[str, Any]) -> list[str]:
    """The action libraries a model's index lists, in order; the first is the default."""
    entry = index.get('action_library') or {}
    return list(entry.get('libraries', ()))
