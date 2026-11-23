"""The fixed prompt's text embedding.

Every model in this package conditions on one constant prompt ("a video"), so
its UMT5 embedding depends only on the base model's text encoder and the prompt
text -- not on any O²LaF checkpoint. It ships with the package, tagged with both,
and inference uses it instead of loading the 11 GB text encoder. A model trained
with another prompt or base model finds no packaged match and has its prompt
encoded on load, as before.
"""

from __future__ import annotations

import gc
from importlib import resources
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

PROMPT_EMBEDS_KEY = 'prompt_embeds'
TEXT_MAX_LENGTH = 512
PACKAGED_DIR = 'assets/prompt_embeds'


def save_prompt_embeds(path: str | Path, embeds: torch.Tensor, *, base_model: str, prompt: str) -> None:
    from o2laf.inference.hub import save_tensors

    metadata = {'base_model': base_model, 'prompt': prompt, 'max_length': str(TEXT_MAX_LENGTH)}
    save_tensors({PROMPT_EMBEDS_KEY: embeds.detach().cpu().contiguous()}, path, metadata=metadata)


def load_prompt_embeds(path: str | Path) -> tuple[torch.Tensor, dict[str, str]]:
    with safe_open(str(path), framework='pt') as handle:
        names = list(handle.keys())
        if names != [PROMPT_EMBEDS_KEY]:
            raise KeyError(f'{path} must hold exactly one tensor named {PROMPT_EMBEDS_KEY!r}, found {names}.')
        return handle.get_tensor(PROMPT_EMBEDS_KEY), dict(handle.metadata() or {})


def packaged_prompt_embeds(base_model: str, prompt: str) -> torch.Tensor | None:
    """The shipped embedding of ``prompt`` under ``base_model``'s text encoder, if there is one."""
    directory = resources.files('o2laf').joinpath(PACKAGED_DIR)
    if not directory.is_dir():
        return None
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if not entry.name.endswith('.safetensors'):
            continue
        with resources.as_file(entry) as path:
            embeds, metadata = load_prompt_embeds(path)
        if metadata.get('base_model') == base_model and metadata.get('prompt') == prompt:
            return embeds
    return None


@torch.no_grad()
def encode_prompt(model_path: str, prompt: str, device, dtype, **load_kwargs) -> torch.Tensor:
    """UMT5 embedding of ``prompt``, zero-padded to the text length Wan expects."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder='tokenizer', use_fast=False, **load_kwargs)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path, subfolder='text_encoder', torch_dtype=dtype, **load_kwargs
    ).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    inputs = tokenizer(
        [prompt],
        padding='max_length',
        max_length=TEXT_MAX_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    length = int(mask[0].sum())
    embeddings = text_encoder(ids, mask).last_hidden_state[:, :length]
    embeddings = F.pad(embeddings, (0, 0, 0, TEXT_MAX_LENGTH - length)).to(dtype)
    del tokenizer, text_encoder, inputs, ids, mask
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings


def fixed_prompt_embeds(*, base_model: str, prompt: str, base_model_path: str, device, dtype, **load_kwargs) -> torch.Tensor:
    """The packaged embedding when one matches, else ``prompt`` encoded with the base model's text encoder.

    ``base_model`` is the id the model was trained on, which is what the
    embedding depends on; ``base_model_path`` is where to load it from if it
    has to be computed.
    """
    embeds = packaged_prompt_embeds(base_model, prompt)
    if embeds is None:
        embeds = encode_prompt(base_model_path, prompt, device, dtype, **load_kwargs)
    return embeds.to(device=device, dtype=dtype)
