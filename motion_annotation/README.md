# IDM as a Motion Annotator

Repurpose O²LaF's Wan 2.2 DiT latent action encoder (IDM) as a supervised
motion annotator on **CameraBench** (15 camera-motion primitives, achieving
state-of-the-art 89.9%) or any custom multi-label video dataset.

No extra packages are required beyond the base `o2laf` install.

## Four Adaptation Modes

All four modes share one Wan 2.2 VAE latent cache and one sparse masked-BCE
objective, enabling controlled comparisons across initialization and readout
depth:

| Mode | Backbone init | Features | Trainable weights | Entrypoint |
|---|---|---|---|---|
| `wan_base_lora` | Base `Wan2.2-TI2V-5B` (no O²LaF SSL) | 3072-D pre-Gaussian | Fresh rank-128 trunk LoRA (`blocks.`) + readout (`action_query`, `query_norm`) + head | `train_finetune.py` |
| `ssl_action_mu` | O²LaF IDM (frozen) | 32-D `action_mu` | Classification head only | `extract_features.py` + `train_frozen.py` |
| `ssl_pre_gaussian` | O²LaF IDM (frozen) | 3072-D pre-Gaussian | Classification head only | `extract_features.py` + `train_frozen.py` |
| `ssl_lora` | O²LaF IDM | 3072-D pre-Gaussian | Same rank-128 trunk LoRA + readout contract as `wan_base_lora` + head | `train_finetune.py` |

Key design details:

- **Matched trainable contract:** `wan_base_lora` and `ssl_lora` optimize the
  exact same trunk LoRA and pre-Gaussian readout tensors; the Gaussian
  projection (`idm_head.outputs`) and `action_rope_angle` stay frozen.
- **Exact bounded-memory clip backward:** Long videos spanning multiple
  41-frame windows use two-pass recomputation (`recompute.py`), bounding GPU
  memory by `--window-microbatch` while computing exact whole-clip gradients.
- **No text encoder required:** Reuses the fixed `"a video"` UMT5 embedding
  packaged with `o2laf`. Hub weights download automatically on first use (pass
  `--model <local_dir> --base-model <wan_dir>` for local copies).

## 1. Prepare Data

### Option A: CameraBench (15 camera-motion primitives)

1. Download the public CameraBench test clips (`videos_gif/`) from
   [`syCen/CameraBench`](https://huggingface.co/datasets/syCen/CameraBench) and
   the official 15-class binary task JSONLs (`binary_classification/*.jsonl`)
   from the CameraBench benchmark release.
2. Request the official supervised training archive (`camerabench_train.zip`
   and training MP4 clips) from the CameraBench authors. Because access is
   controlled by the authors, raw videos and annotations are not redistributed
   here.
3. Build the leak-audited index (deterministic source-group-disjoint split of
   1,114 train / 197 validation clips, plus the 969 public test clips):

```bash
python -m motion_annotation.prepare_camerabench \
    --task-dir /path/to/CameraBench/binary_classification \
    --annotations-zip /path/to/camerabench_train.zip \
    --train-videos /path/to/camerabench_train/videos \
    --test-videos /path/to/CameraBench/videos_gif \
    --output-dir work/camerabench_index
```

To prepare only the public test split (without `camerabench_train.zip`), omit
`--annotations-zip` and `--train-videos`.

### Option B: Custom Multi-Label Video Dataset

Provide a JSONL file with one clip per line. Labels can be sparse: omitted
classes are treated as unknown and masked out of the BCE loss rather than
assumed negative:

```json
{"video_id": "clip_001", "video_path": "/data/videos/clip_001.mp4", "split": "train", "labels": {"Pan_Left": 1, "Zoom_In": 0}}
{"video_id": "clip_002", "video_path": "/data/videos/clip_002.mp4", "split": "val", "labels": {"Pan_Left": "No", "Zoom_In": "Yes"}}
{"video_id": "clip_003", "video_path": "/data/videos/clip_003.mp4", "split": "test", "labels": {"Pan_Left": 0, "Zoom_In": 1}}
```

Labels accept `1`/`0`, `true`/`false`, or `"Yes"`/`"No"`. Classes are inferred
automatically from the `train` split (`--classes auto`) or specified explicitly
(`--classes camerabench`, `--classes Pan_Left,Zoom_In`, or a JSON list file).

## 2. Cache Wan 2.2 VAE Latents (Shared Across All 4 Modes)

```bash
torchrun --nproc_per_node=4 -m motion_annotation.cache_latents \
    --video-index work/camerabench_index/video_index.jsonl \
    --base-model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --output-dir work/latents
```

For a single GPU, run `python -m motion_annotation.cache_latents ...` directly.
Re-running resumes safely and skips valid cached clips.

## 3. Frozen IDM Modes (`ssl_action_mu` / `ssl_pre_gaussian`)

Extract both 32-D `action_mu` and 3072-D pre-Gaussian `transition_features` in
a single pass over the cached latents:

```bash
torchrun --nproc_per_node=4 -m motion_annotation.extract_features \
    --model YuxinJ/O2LaF/o2laf_dit_mixed \
    --video-index work/camerabench_index/video_index.jsonl \
    --latent-cache work/latents \
    --feature-kind both \
    --output-dir work/features_o2laf_dit_mixed
```

Train lightweight heads on either representation (runs in seconds on CPU or
GPU; `--head` supports `direct_linear`, `mean_linear`, `mean_mlp`, `attention`):

```bash
# Frozen 32-D action_mu
python -m motion_annotation.train_frozen \
    --mode ssl_action_mu \
    --features work/features_o2laf_dit_mixed \
    --video-index work/camerabench_index/video_index.jsonl \
    --head direct_linear \
    --seed 0 \
    --output-dir work/runs/ssl_action_mu_seed0

# Frozen 3072-D pre-Gaussian transition features
python -m motion_annotation.train_frozen \
    --mode ssl_pre_gaussian \
    --features work/features_o2laf_dit_mixed \
    --video-index work/camerabench_index/video_index.jsonl \
    --head direct_linear \
    --seed 0 \
    --output-dir work/runs/ssl_pre_gaussian_seed0
```

## 4. End-to-End LoRA + Readout Fine-Tuning (`wan_base_lora` / `ssl_lora`)

```bash
# Base Wan 2.2 TI2V-5B + fresh rank-128 LoRA + fresh readout
torchrun --nproc_per_node=4 -m motion_annotation.train_finetune \
    --mode wan_base_lora \
    --base-model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --video-index work/camerabench_index/video_index.jsonl \
    --latent-cache work/latents \
    --head direct_linear \
    --seed 0 \
    --output-dir work/runs/wan_base_lora_seed0

# O²LaF SSL IDM + rank-128 LoRA + readout fine-tuning
torchrun --nproc_per_node=4 -m motion_annotation.train_finetune \
    --mode ssl_lora \
    --model YuxinJ/O2LaF/o2laf_dit_mixed \
    --video-index work/camerabench_index/video_index.jsonl \
    --latent-cache work/latents \
    --head direct_linear \
    --seed 0 \
    --output-dir work/runs/ssl_lora_seed0
```

Pass `--audit-only` to write `audit.json` and verify exact trainable parameter
contract parity between `wan_base_lora` and `ssl_lora` without running
training. Training resumes automatically from `latest.pt` if interrupted.

## 5. Evaluate (Single Checkpoint or Multi-Seed Ensemble)

`evaluate.py` evaluates checkpoints from all four modes. Pass `--checkpoint`
multiple times to average predicted probabilities across seeds:

```bash
# Frozen modes (ssl_action_mu / ssl_pre_gaussian)
python -m motion_annotation.evaluate \
    --checkpoint work/runs/ssl_pre_gaussian_seed0/best.pt \
    --video-index work/camerabench_index/video_index.jsonl \
    --features work/features_o2laf_dit_mixed \
    --camerabench-task-dir /path/to/CameraBench/binary_classification \
    --model-name ssl_pre_gaussian \
    --output-dir work/eval/ssl_pre_gaussian

# Fine-tuned modes (wan_base_lora / ssl_lora)
python -m motion_annotation.evaluate \
    --checkpoint work/runs/ssl_lora_seed0/best.pt \
    --checkpoint work/runs/ssl_lora_seed1/best.pt \
    --checkpoint work/runs/ssl_lora_seed2/best.pt \
    --video-index work/camerabench_index/video_index.jsonl \
    --latent-cache work/latents \
    --camerabench-task-dir /path/to/CameraBench/binary_classification \
    --model-name ssl_lora \
    --output-dir work/eval/ssl_lora
```

For custom datasets, omit `--camerabench-task-dir` to score directly against
the `labels` field of `--split test` in `--video-index`.
