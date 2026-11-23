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
"""Flex-attention helpers shared by the action encoder and the world model.

Both models restrict self-attention with a token-level predicate. The predicate
is turned into a ``BlockMask`` here, and ``WanFlexAttnProcessor`` replaces the
stock Wan self-attention so that the mask installed on ``processor.block_mask``
is actually honoured.
"""

import math

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_wan import _get_qkv_projections
from torch.nn.attention.flex_attention import BlockMask, flex_attention

flex_attention = torch.compile(flex_attention, dynamic=True)

BLOCK_SIZE = 128


def padded_length(seq_len: int) -> int:
    """Round ``seq_len`` up to the flex-attention block size."""
    return int(math.ceil(seq_len / float(BLOCK_SIZE)) * BLOCK_SIZE)


def pad_sequence_dim(tensor: torch.Tensor, padded_seq_len: int, dim: int = 1) -> torch.Tensor:
    """Zero-pad ``tensor`` along ``dim`` up to ``padded_seq_len``."""
    seq_len = tensor.shape[dim]
    if seq_len == padded_seq_len:
        return tensor
    if seq_len > padded_seq_len:
        raise ValueError(f'Cannot pad sequence dim {dim} from {seq_len} down to {padded_seq_len}')
    pad_shape = list(tensor.shape)
    pad_shape[dim] = padded_seq_len - seq_len
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)


def _ordered_block_columns(block_mask):
    return torch.argsort(block_mask.to(torch.int32), dim=-1, descending=True, stable=True).to(torch.int32)


def build_block_mask(mask_mod, seq_len: int, device) -> BlockMask:
    """Square ``BlockMask`` for a token predicate over ``seq_len`` positions.

    Built on the host from the predicate evaluated one 128-row strip at a time,
    so no flex-attention kernel is needed to construct it.
    """
    seq_len = int(seq_len)
    num_blocks = math.ceil(seq_len / BLOCK_SIZE)
    kv_num_blocks = torch.zeros((1, 1, num_blocks), dtype=torch.int32, device=device)
    kv_indices = torch.zeros((1, 1, num_blocks, max(1, num_blocks)), dtype=torch.int32, device=device)
    full_kv_num_blocks = torch.zeros_like(kv_num_blocks)
    full_kv_indices = torch.zeros_like(kv_indices)

    batch_index = torch.tensor(0, device=device)
    head_index = torch.tensor(0, device=device)
    kv_index = torch.arange(seq_len, device=device)[None, :]
    padded_len = num_blocks * BLOCK_SIZE
    with torch.no_grad():
        for q_block in range(num_blocks):
            q_start = q_block * BLOCK_SIZE
            q_end = min(q_start + BLOCK_SIZE, seq_len)
            q_index = torch.arange(q_start, q_end, device=device)[:, None]
            allowed = mask_mod(batch_index, head_index, q_index, kv_index)
            if not torch.is_tensor(allowed):
                allowed = torch.as_tensor(allowed, device=device)
            allowed = torch.broadcast_to(allowed.to(torch.bool), (q_end - q_start, seq_len))
            if padded_len > seq_len:
                allowed = torch.cat([allowed, allowed.new_zeros((q_end - q_start, padded_len - seq_len))], dim=1)
            if q_end - q_start < BLOCK_SIZE:
                allowed = torch.cat([allowed, allowed.new_zeros((BLOCK_SIZE - (q_end - q_start), padded_len))], dim=0)

            block_sum = allowed.reshape(BLOCK_SIZE, num_blocks, BLOCK_SIZE).sum(dim=(0, 2))
            block_all = block_sum == BLOCK_SIZE * BLOCK_SIZE
            block_partial = (block_sum > 0) & ~block_all

            kv_indices[0, 0, q_block] = _ordered_block_columns(block_partial)
            full_kv_indices[0, 0, q_block] = _ordered_block_columns(block_all)
            kv_num_blocks[0, 0, q_block] = int(block_partial.sum().item())
            full_kv_num_blocks[0, 0, q_block] = int(block_all.sum().item())

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=(BLOCK_SIZE, BLOCK_SIZE),
        mask_mod=mask_mod,
        seq_lengths=(seq_len, seq_len),
    )


# Masks kept per module; a handful of geometries at most, so a small bound only
# guards against a caller that varies its shapes without end.
BLOCK_MASK_CACHE_SIZE = 8


def cached_block_mask(owner, key, build):
    """``build()``'s mask for ``key``, built once per owner and reused.

    A mask depends only on the sequence geometry, never on the activations, and
    building one costs a Python loop with two host-device syncs per 128-token
    block -- on every forward pass, if it is not cached.
    """
    cache = owner.__dict__.setdefault('_block_mask_cache', {})
    mask = cache.get(key)
    if mask is None:
        if len(cache) >= BLOCK_MASK_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        mask = cache[key] = build()
    return mask


def _apply_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class WanFlexAttnProcessor:
    """Wan self-attention that applies ``self.block_mask`` through flex attention.

    With ``block_mask`` unset it falls back to plain SDPA, so the same processor
    serves masked and unmasked calls.
    """

    def __init__(self):
        self.block_mask = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        if attention_mask is None:
            attention_mask = self.block_mask
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attention_mask is None:
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        else:
            hidden_states = flex_attention(query, key, value, block_mask=attention_mask)

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def install_flex_attention(blocks, processor_class=WanFlexAttnProcessor) -> None:
    """Replace every block's self-attention processor with ``processor_class``."""
    for block in blocks:
        block.attn1.set_processor(processor_class())


def verify_flex_attention(blocks, processor_class=WanFlexAttnProcessor, label='model') -> None:
    """Fail if any block still uses a processor that would ignore ``block_mask``.

    Assigning ``block_mask`` to the stock diffusers processor parks an attribute
    nobody reads, so attention silently stays unmasked.
    """
    stock = [index for index, block in enumerate(blocks) if not isinstance(block.attn1.processor, processor_class)]
    if stock:
        raise RuntimeError(
            f'{len(stock)} of {len(blocks)} {label} blocks do not use {processor_class.__name__}, '
            f'so the block mask would be silently ignored. First offending block: {stock[0]}.'
        )
