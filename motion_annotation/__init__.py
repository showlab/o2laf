"""Supervised motion annotation with O²LaF Wan DiT latent action encoders.

Supports four controlled adaptation modes on CameraBench or any custom
multi-label video JSONL dataset:

1. ``wan_base_lora``: Wan 2.2 TI2V-5B base + fresh rank-128 LoRA + fresh readout (3072-D).
2. ``ssl_action_mu``: Frozen O²LaF encoder, 32-D posterior mean ``action_mu`` + head.
3. ``ssl_pre_gaussian``: Frozen O²LaF encoder, 3072-D pre-Gaussian features + head.
4. ``ssl_lora``: O²LaF encoder + rank-128 LoRA + readout fine-tuning (3072-D, exact
   trainable parameter contract parity with ``wan_base_lora``).
"""

ADAPTATION_MODES = (
    'wan_base_lora',
    'ssl_action_mu',
    'ssl_pre_gaussian',
    'ssl_lora',
)

__all__ = ['ADAPTATION_MODES']
