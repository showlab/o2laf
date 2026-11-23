"""The on-disk layout of a released model, as published on Hugging Face.

::

    <model>/
      model_index.json        what the model is and which files make it up
      latent_action_model/    the action encoder and the causal world model, as
                              latent action learning trained them: one config,
                              tensors under ``action_encoder.`` and ``world_model.``
      world_model/            the bidirectional world model (sharded when large)
      action_library/         optional atomic actions

The latent action model is one component because it is one trained object: the
encoder and the causal world model were optimised jointly, a run that continues
training restores both, and neither is meaningful paired with another's
partner. Bidirectional inference reads only its ``action_encoder.`` tensors.

The fixed prompt's embedding is not part of a model: it depends only on the base
model's text encoder and the prompt, so it ships with the package
(:mod:`o2laf.inference.prompts`).

Weights are stored exactly as trained, in the dtype they were trained in.
Tensor names are the model's own parameter names with PEFT's wrapper removed:
no ``base_model.model.`` prefix, no ``base_layer`` segment, no adapter name. A
file therefore does not depend on how the model was wrapped when it was
trained, and :func:`hub_keys_to_model_keys` puts the wrapper back for whatever
module it is loaded into.

Writing and reading live together here so the format is defined in one place.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from o2laf.utils.action_util import NULL_ACTION_CONVENTIONS
from o2laf.utils.file_util import write_json_atomic
from o2laf.utils.lora_util import LORA_ADAPTER_NAME

MODEL_FORMAT = 'o2laf_model_v1'
MODEL_INDEX_NAME = 'model_index.json'
COMPONENT_CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'model.safetensors'
WEIGHTS_INDEX_NAME = 'model.safetensors.index.json'
INFERENCE_MODES = ('causal', 'bidirectional')

LATENT_ACTION_MODEL_DIR = 'latent_action_model'
WORLD_MODEL_DIR = 'world_model'
# Parts of the latent action model; each is also the key of its config section.
ACTION_ENCODER_PART = 'action_encoder'
WORLD_MODEL_PART = 'world_model'

# PEFT's wrapper prefix, at the start of a module path -- which, inside a joint
# component, is after the part's own prefix (``action_encoder.base_model...``).
_PEFT_PREFIX = re.compile(r'(^|\.)base_model\.model\.')
_BASE_LAYER = '.base_layer'
# ``<module>.lora_A.real.weight`` (or ``lora_B``, ``lora_magnitude_vector``,
# ...) with this package's adapter name. Only that name is stripped: matching
# any segment would also strip the ``weight`` of an already-clean name, and the
# mapping must be safe to apply twice.
_PEFT_ADAPTER = re.compile(rf'\.(lora_[A-Za-z_]+)\.{re.escape(LORA_ADAPTER_NAME)}(?=\.|$)')

_REQUIRED_INDEX_KEYS = (
    'format',
    'base_model',
    'fixed_prompt',
    'action_space_id',
    'action_encoder',
    'action_dim',
    'codes_per_transition',
    'temporal_compression_ratio',
    'modes',
)
_REQUIRED_MODE_KEYS = (
    'world_model', 'null_action', 'num_rgb_frames', 'num_context_frames', 'height', 'width', 'fps', 'scheduler_cfg',
)


def save_tensors(tensors: Mapping[str, torch.Tensor], path: str | Path, metadata: dict[str, str] | None = None) -> None:
    """``safetensors.save_file``, readable by others.

    safetensors creates its files owner-only, which leaves a model directory
    shared on a cluster unreadable to everyone else. The mode is set outright
    rather than derived from the umask, which can only be read by changing it --
    process-wide, and so not safely while other threads create files.
    """
    save_file(dict(tensors), str(path), metadata={'format': 'pt', **(metadata or {})})
    os.chmod(path, 0o644)


# -- tensor names -------------------------------------------------------------


def to_hub_key(name: str) -> str:
    """A parameter name with PEFT's wrapper removed, with or without a part prefix."""
    name = _PEFT_PREFIX.sub(r'\1', name)
    name = name.replace(_BASE_LAYER + '.', '.')
    return _PEFT_ADAPTER.sub(r'.\1', name)


def hub_keys_to_model_keys(hub_keys: Iterable[str], model_keys: Iterable[str]) -> dict[str, str]:
    """Map each hub tensor name to the name it has in ``model_keys``.

    The model may or may not be PEFT-wrapped; the mapping is found by stripping
    the model's own names the same way the hub names were stripped. A hub name with no counterpart, or with several, is an
    error rather than a guess.
    """
    by_hub_key: dict[str, list[str]] = {}
    for model_key in model_keys:
        by_hub_key.setdefault(to_hub_key(model_key), []).append(model_key)
    mapping, missing, ambiguous = {}, [], []
    for hub_key in hub_keys:
        candidates = by_hub_key.get(hub_key, [])
        if not candidates:
            missing.append(hub_key)
        elif len(candidates) > 1:
            ambiguous.append((hub_key, candidates))
        else:
            mapping[hub_key] = candidates[0]
    if missing:
        raise KeyError(f'{len(missing)} stored tensors have no parameter in the model, e.g. {missing[:3]}.')
    if ambiguous:
        raise KeyError(f'{len(ambiguous)} stored tensors match several parameters, e.g. {ambiguous[:1]}.')
    return mapping


def tensor_content_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """A digest of names, dtypes, shapes and bytes, independent of file layout.

    Two files holding the same tensors hash alike however they are sharded or
    ordered, so this identifies weights rather than a particular serialisation.
    """
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('utf-8'))
        digest.update(json.dumps(list(tensor.shape)).encode('utf-8'))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def action_space_id(encoder_state: Mapping[str, torch.Tensor]) -> str:
    """The identity of the action space an encoder defines: a hash of its weights.

    Names are compared with PEFT's wrapper removed, so a training state and its
    released form give the same id. World model training records it, the release
    records it, and action libraries are bound to it -- one definition for all.
    """
    return 'sha256:' + tensor_content_sha256({to_hub_key(name): tensor for name, tensor in encoder_state.items()})


# -- components -----------------------------------------------------------------


def save_component(directory: str | Path, config: Mapping[str, Any], state: Mapping[str, torch.Tensor], *,
                   max_shard_size: int | str = '5GB') -> list[str]:
    """Write one component's ``config.json`` and weights; return the weight file names."""
    from huggingface_hub import split_torch_state_dict_into_shards

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    renamed = {to_hub_key(name): name for name in state}
    if len(renamed) != len(state):
        raise ValueError('Two tensors map to one stored name once the PEFT wrapper is removed.')
    tensors = {hub_key: state[name].detach().cpu().contiguous() for hub_key, name in renamed.items()}

    split = split_torch_state_dict_into_shards(tensors, filename_pattern='model{suffix}.safetensors',
                                               max_shard_size=max_shard_size)
    for filename, names in split.filename_to_tensors.items():
        save_tensors({name: tensors[name] for name in names}, directory / filename)
    if split.is_sharded:
        write_json_atomic(directory / WEIGHTS_INDEX_NAME, {'metadata': split.metadata, 'weight_map': split.tensor_to_filename})
    write_json_atomic(directory / COMPONENT_CONFIG_NAME, dict(config))
    return sorted(split.filename_to_tensors)


def load_component_config(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / COMPONENT_CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f'Component has no {COMPONENT_CONFIG_NAME}: {path}.')
    config = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise TypeError(f'{path} must contain a JSON object.')
    return config


def load_component_state(directory: str | Path, part: str | None = None) -> dict[str, torch.Tensor]:
    """The tensors of one component, from a single file or its shards.

    With ``part``, only the tensors stored under ``<part>.`` are read -- the
    others are never loaded -- and they are returned without that prefix.
    """
    directory = Path(directory)
    prefix = f'{part}.' if part else ''
    index_path = directory / WEIGHTS_INDEX_NAME
    if not index_path.is_file():
        single = directory / WEIGHTS_NAME
        if not single.is_file():
            raise FileNotFoundError(f'Component has neither {WEIGHTS_NAME} nor {WEIGHTS_INDEX_NAME}: {directory}.')
        return _read_tensors(single, prefix)

    index = json.loads(index_path.read_text(encoding='utf-8'))
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f'{index_path} has no weight_map.')
    # The weight map alone cannot reveal a dropped entry whose tensor had a
    # shard to itself: that shard is simply never read. The shard names and the
    # recorded total size are independent witnesses to what was written.
    filenames = sorted(set(weight_map.values()))
    expected = _expected_shard_names(filenames)
    if filenames != expected:
        raise ValueError(f'{index_path} references shards {filenames}, but their names say {expected}.')
    stored: dict[str, int] = {}
    state: dict[str, torch.Tensor] = {}
    for filename in filenames:
        with safe_open(str(directory / filename), framework='pt') as handle:
            for name in handle.keys():
                if name in stored:
                    raise ValueError(f'Tensor {name!r} is stored in more than one shard.')
                tensor_slice = handle.get_slice(name)
                stored[name] = _stored_bytes(tensor_slice.get_shape(), tensor_slice.get_dtype())
                if name.startswith(prefix):
                    state[name[len(prefix):]] = handle.get_tensor(name)
    if stored.keys() != weight_map.keys():
        missing = sorted(weight_map.keys() - stored.keys())
        extra = sorted(stored.keys() - weight_map.keys())
        raise ValueError(f'Shards disagree with {WEIGHTS_INDEX_NAME}: missing {missing[:3]}, unlisted {extra[:3]}.')
    recorded = (index.get('metadata') or {}).get('total_size')
    if recorded is not None and int(recorded) != sum(stored.values()):
        raise ValueError(f'{index_path} records {recorded} bytes of weights, but the shards hold {sum(stored.values())}.')
    if prefix and not state:
        raise KeyError(f'{directory} stores no tensors under {prefix!r}.')
    return state


_DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1, 'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
                'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}


def _stored_bytes(shape, dtype: str) -> int:
    count = 1
    for size in shape:
        count *= int(size)
    return count * _DTYPE_BYTES[dtype]


def _read_tensors(path: Path, prefix: str) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework='pt') as handle:
        state = {name[len(prefix):]: handle.get_tensor(name) for name in handle.keys() if name.startswith(prefix)}
    if prefix and not state:
        raise KeyError(f'{path} stores no tensors under {prefix!r}.')
    return state


def _expected_shard_names(filenames: list[str]) -> list[str]:
    """The complete set of ``model-0000k-of-0000n`` names the first one implies."""
    match = re.fullmatch(r'model-\d+-of-(\d+)\.safetensors', filenames[0])
    if match is None:
        raise ValueError(f'Unexpected shard name {filenames[0]!r}.')
    width, count = len(match.group(1)), int(match.group(1))
    return [f'model-{index:0{width}d}-of-{count:0{width}d}.safetensors' for index in range(1, count + 1)]


# -- the model index ------------------------------------------------------------


def read_model_index(root: str | Path) -> dict[str, Any]:
    """Read and check ``model_index.json``.

    Components are checked when they are loaded rather than here: a download
    for one mode deliberately leaves out the other mode's world model.
    """
    root = Path(root)
    path = root / MODEL_INDEX_NAME
    if not path.is_file():
        raise FileNotFoundError(f'Not an O²LaF model directory (no {MODEL_INDEX_NAME}): {root}.')
    index = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(index, dict):
        raise TypeError(f'{path} must contain a JSON object.')
    if index.get('format') != MODEL_FORMAT:
        raise ValueError(f'{path} has format {index.get("format")!r}; this package reads {MODEL_FORMAT!r}.')
    missing = [key for key in _REQUIRED_INDEX_KEYS if key not in index]
    if missing:
        raise KeyError(f'{path} is missing {missing}.')
    modes = index['modes']
    if not isinstance(modes, dict) or not modes or set(modes) - set(INFERENCE_MODES):
        raise ValueError(f'{path} modes must be a non-empty subset of {INFERENCE_MODES}, got {modes!r}.')
    for name, mode in modes.items():
        absent = [key for key in _REQUIRED_MODE_KEYS if key not in mode]
        if absent:
            raise KeyError(f'{path} mode {name!r} is missing {absent}.')
        if mode['null_action'] not in NULL_ACTION_CONVENTIONS:
            raise ValueError(f'{path} mode {name!r} has unknown null_action {mode["null_action"]!r}.')
    return index


def component_paths(index: Mapping[str, Any], mode: str) -> tuple[dict[str, str], dict[str, str]]:
    """Where the action encoder and ``mode``'s world model live.

    Each is ``{'component': <directory>, 'part': <section or None>}``: a part
    names the config section and tensor prefix inside a joint component.
    """
    if mode not in index['modes']:
        raise ValueError(f'This model has no {mode!r} mode; it has {sorted(index["modes"])}.')
    return dict(index['action_encoder']), dict(index['modes'][mode]['world_model'])


def load_part_config(root: str | Path, location: Mapping[str, str]) -> dict[str, Any]:
    """The config of the part at ``location``: its section of a joint component, or the whole component."""
    config = load_component_config(Path(root) / location['component'])
    part = location.get('part')
    if not part:
        return config
    if part not in config:
        raise KeyError(f'{location["component"]}/{COMPONENT_CONFIG_NAME} has no {part!r} section.')
    return dict(config[part])


def load_part_state(root: str | Path, location: Mapping[str, str]) -> dict[str, torch.Tensor]:
    return load_component_state(Path(root) / location['component'], location.get('part'))


FETCH_MODES = ('mode', 'metadata', 'action_encoder')


def resolve_model_dir(pretrained_model_name_or_path: str | Path, *, mode: str = 'bidirectional', revision: str | None = None,
                      cache_dir: str | Path | None = None, local_files_only: bool = False,
                      token: str | bool | None = None, fetch: str = 'mode') -> Path:
    """A local directory holding a model, downloading from the Hub if given a repo id.

    A Hub model is ``<org>/<repo>/<model>``: one repository holds several
    models, each in its own subfolder, e.g. ``YuxinJ/O2LaF/o2laf_dit_mixed``.
    ``<org>/<repo>`` alone names a model at the repository root.

    ``fetch`` says how much of a Hub model to download:

    * ``'mode'``: everything ``mode`` needs to run -- the latent action model,
      that mode's world model and the action library. Causal inference never
      downloads the 10 GB bidirectional world model.
    * ``'metadata'``: the index, every component's ``config.json`` and the
      action library; no weights at all.
    * ``'action_encoder'``: the metadata, plus the component holding the action
      encoder.

    A local directory is returned as it is.
    """
    if fetch not in FETCH_MODES:
        raise ValueError(f'fetch must be one of {FETCH_MODES}, got {fetch!r}.')
    name = str(pretrained_model_name_or_path)
    local = Path(name).expanduser()
    if local.is_dir():
        return local.resolve()
    if _looks_like_a_path(name):
        raise FileNotFoundError(f'No model directory at {local}. (A Hub model is named "<org>/<name>".)')

    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    repo_id, subfolder = _split_hub_name(name)
    prefix = f'{subfolder}/' if subfolder else ''
    hub = {'revision': revision, 'cache_dir': cache_dir, 'local_files_only': local_files_only, 'token': token}
    try:
        index_file = hf_hub_download(repo_id, prefix + MODEL_INDEX_NAME, **hub)
    except EntryNotFoundError as error:
        if isinstance(error, LocalEntryNotFoundError):
            raise
        raise FileNotFoundError(_no_model_here(repo_id, subfolder, revision, token)) from error
    index = read_model_index_file(Path(index_file))
    # Every component the model has, from the index alone: `mode` is only
    # looked up below, when its weights are asked for, so a model without the
    # default mode can still be described.
    every_component = {index['action_encoder']['component']}
    every_component |= {mode_entry['world_model']['component'] for mode_entry in index['modes'].values()}
    patterns = [MODEL_INDEX_NAME, *(f'{component}/{COMPONENT_CONFIG_NAME}' for component in sorted(every_component))]
    if fetch == 'mode':
        weights = {location['component'] for location in component_paths(index, mode)}
    elif fetch == 'action_encoder':
        weights = {index['action_encoder']['component']}
    else:
        weights = set()
    patterns += [f'{component}/*' for component in sorted(weights)]
    if index.get('action_library'):
        patterns.append(f'{index["action_library"]["directory"]}/*')
    snapshot = Path(snapshot_download(repo_id, allow_patterns=[prefix + pattern for pattern in patterns], **hub))
    return snapshot / subfolder if subfolder else snapshot


def _looks_like_a_path(name: str) -> bool:
    """Whether ``name`` was meant as a local path rather than a Hub model (``org/repo`` or ``org/repo/model``)."""
    return name.startswith(('/', './', '../', '~', '.\\')) or name in ('.', '..') or name.count('/') > 2


def _split_hub_name(name: str) -> tuple[str, str | None]:
    """``org/repo/model`` to ``('org/repo', 'model')``; ``org/repo`` to ``('org/repo', None)``."""
    parts = name.split('/')
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(f'{name!r} is not a Hub model: expected "<org>/<repo>" or "<org>/<repo>/<model>".')
    return '/'.join(parts[:2]), parts[2] if len(parts) == 3 else None


def _no_model_here(repo_id: str, subfolder: str | None, revision, token) -> str:
    """Explain a missing index, naming the models the repository does hold."""
    where = f'{repo_id}/{subfolder}' if subfolder else repo_id
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo_id, revision=revision, token=token)
    except Exception:  # noqa: BLE001 - the listing only improves the message
        return f'{where} holds no {MODEL_INDEX_NAME}.'
    models = sorted(f'{repo_id}/{path.rsplit("/", 1)[0]}' for path in files if path.endswith(f'/{MODEL_INDEX_NAME}'))
    if not models:
        return f'{where} holds no {MODEL_INDEX_NAME}, and neither does any folder of {repo_id}.'
    return f'{where} is not a model. {repo_id} holds: {", ".join(models)}.'


def read_model_index_file(path: Path) -> dict[str, Any]:
    """The index alone, before the rest of the model is present to check it against."""
    index = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(index, dict) or index.get('format') != MODEL_FORMAT:
        raise ValueError(f'{path} is not an {MODEL_FORMAT} model index.')
    return index
