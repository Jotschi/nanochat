# nanochat-vlm — Implementation State

> Status: **IMPLEMENTED, TRAINED & VERIFIED** (2026-08-23)
>
> This document records what has been built to achieve [GOAL.md](./GOAL.md) — adding
> native vision capability to nanochat to make it a Vision-Language Model (VLM) — and
> how to interact with the implementation. It is a living record; update it as the
> implementation evolves.
>
> Three full end-to-end training runs (base LLM → ViT pretrain → VLM SFT → inference) were
> completed. v1 (54M) and v2 (146.5M, scaled up) ran on a single RTX 4090 with a from-scratch
> ViT; v3 (~147M) ran on a remote H200 with a DINOv2 ViT-B/14 pretrained encoder. See
> [report/vlm_train_v1.html](../report/vlm_train_v1.html),
> [report/vlm_train_v2.html](../report/vlm_train_v2.html), and
> [report/vlm_train_v3.html](../report/vlm_train_v3.html) for the detailed training reports,
> loss curves, and eval examples.

## 1. What was done

A lightweight, native-PyTorch vision encoder (ViT) was added and wired into the existing
LLaMA-style GPT, following the GOAL's architectural direction: a patch-based ViT converts
an image into a contiguous prefix of visual-token embeddings that the language model
consumes. No heavyweight multimodal framework (HF Transformers, timm, torchvision model
abstractions) was introduced — the only new dependency is **Pillow** for image loading.

### New files

| File | Purpose |
|------|---------|
| `nanochat/vit.py` | ViT encoder: `ViTConfig` + `ViT`. Patch embed (Conv2d), learned positional embed, bidirectional transformer blocks (full attention via SDPA), and a `projector: Linear(embed_dim, n_embd)` that maps patch features into the LLM hidden dim. Follows nanochat conventions: no bias, RMSNorm, relu² MLP, meta-device init. |
| `nanochat/vlm.py` | `VLM` wrapper combining `ViT` + `GPT`. `forward(image, idx, targets, kv_cache, loss_reduction)` encodes the image → visual embeds, embeds the text, concatenates, and calls `gpt.forward(..., embeds=combined_embeds)`. Also `generate()`, `set_llm_trainable()`, `setup_optimizer()`. |
| `scripts/coco_data.py` | COCO val2014 data pipeline: `prepare_coco()` (download images + captions → parquet), `COCODataset`, `collate_coco_batch`, `load_image()` (PIL → resize 128×128 → normalize [0,1]→[-1,1] → tensor). |
| `scripts/vit_pretrain.py` | **Stage 1** training: load a base model, freeze the GPT, train only the ViT + projector on COCO image→caption pairs (AdamW). |
| `scripts/vlm_sft.py` | **Stage 2** training: load the stage-1 VLM, unfreeze everything, fine-tune on image-text conversations (MuonAdamW). Loss only on assistant tokens. |
| `scripts/vlm_cli.py` | Inference / demo CLI: image + text prompt → generated response. |
| `scripts/vlm_smoke_test.py` | Self-contained end-to-end verification: tiny tokenizer, tiny VLM, learnable synthetic image→caption mapping, checkpoint save/load round-trip, generation. |
| `tests/test_vit.py` | 12 unit tests: config, forward shape, determinism, gradients, init, attention. |
| `tests/test_vlm.py` | 11 unit tests: construction, forward (loss/logits/masking/reduction), generate, GPT backward compat. |

### Modified files

| File | Change |
|------|--------|
| `nanochat/gpt.py` | `GPT.forward()` gained an optional `embeds=None` argument. When provided, it skips `self.transformer.wte(idx)` and uses `embeds` directly (still applies `.to(COMPUTE_DTYPE)` and `norm()`). Fully backward compatible — text-only callers are unaffected. |
| `nanochat/tokenizer.py` | Added `"<\|image\|>"` to `SPECIAL_TOKENS`. Vocab size increases by 1 when the tokenizer is retrained. |
| `nanochat/checkpoint_manager.py` | Added `build_vlm(checkpoint_dir, step, device, phase)`, `load_vlm_from_dir(...)`, and `load_vlm(source, ...)` (source ∈ `vlm` \| `vlm_sft`). Loads `gpt_config` + `vit_config` + `image_token_id` from the checkpoint's meta JSON, builds on the meta device, and loads the state dict. |
| `pyproject.toml` | Added `"pillow>=10.0.0"` to dependencies. |

### Architecture at a glance

```
image (3,128,128)                text prompt
        │                            │
   ┌────▼─────┐                 ┌────▼─────┐
   │   ViT    │                 │  GPT.wte │  (text token embeds)
   │ +projector│                └────┬─────┘
   └────┬─────┘                      │
   visual_embeds (64, n_embd)        │
        │                            │
        └──────────┬─────────────────┘
                   ▼
        combined_embeds (64 + T_text, n_embd)
                   │
                   ▼
            GPT transformer (causal)
                   │
                   ▼
              logits → loss / next token
```

Visual tokens occupy a fixed **64-token** prefix (128/16 = 8×8 patches). In the token-id
sequence, the visual positions carry `image_token_id` so the value-embedding lookup is
well-defined, while the actual values come from the ViT.

### Two-stage training

1. **Stage 1 — ViT pretrain** (`scripts/vit_pretrain.py`): LLM frozen, only ViT + projector
   trained on image→caption pairs. Teaches the vision encoder to emit embeddings the frozen
   LLM can turn into caption text.
2. **Stage 2 — VLM SFT** (`scripts/vlm_sft.py`): full model unfrozen, fine-tuned on
   image-text conversations; loss masked to assistant tokens only.

## 2. Verification

- `pytest tests/ -m "not slow"` → **81 passed** (no regressions to the text-only path).
- `python -m scripts.vlm_smoke_test` → **PASSED** (loss 6.24 → 0.53, checkpoint round-trip
  weights match, generation works).
- No lint errors in any new/modified file; all scripts import cleanly.

## 2b. Training run v1 (2026-08-23, single RTX 4090)

A full end-to-end run was completed. Details, loss curves, and eval examples are in
[report/vlm_train_v1.html](../report/vlm_train_v1.html).

| Stage | Config | Steps | Result |
|-------|--------|-------|--------|
| Base LLM (`d8`) | depth 8, n_embd 512, seq 512, vocab 8192 | 5,000 (~82M tok) | loss 3.62 → 3.18 |
| Stage 1 ViT pretrain (`vlm_d8`) | ViT 256/4/4, frozen LLM, AdamW 3e-4 | 3,000 | val loss 4.16 → 3.16 |
| Stage 2 VLM SFT (`vlmsft_d8`) | full VLM, MuonAdamW | 3,000 | val loss 5.03 → 2.21 (min @ 1500) |
| Inference | `vlm_cli`, temp 0.7 | — | fluent, image-conditioned COCO captions |

Checkpoints: `~/.cache/nanochat/{base_checkpoints/d8, vlm_checkpoints/vlm_d8, vlm_sft_checkpoints/vlmsft_d8}`.

**Bugs fixed during the run** (all in this repo):
1. `base_train` `--total-batch-size` is in tokens and must be a multiple of
   `device_batch_size × max_seq_len` (used 16384).
2. `coco_data.py`: captions are not hosted standalone — now downloads
   `annotations_trainval2014.zip` and extracts `annotations/captions_val2014.json`.
3. `coco_data.py`: `val2014.zip` extracts into a nested `val2014/val2014/` — added a flatten
   step and skip-download-when-present.
4. `coco_data.py` `COCODataset`: `UnboundLocalError` on `start` in the train branch — fixed.
5. `vlm_sft.py`: SFT `targets[i]` shape mismatch — assign to `targets[i, :max_len-1]`.
6. `vlm.py` `setup_optimizer`: 4-D Conv2d `patch_embed` weight was routed to Muon (2-D only) —
   changed routing to `param.ndim == 2` so Conv2d/pos_embed use AdamW.

## 2c. Training run v2 — scaled up (2026-08-23, single RTX 4090)

A scaled-up retrain. Details, loss curves, and v1-vs-v2 eval examples are in
[report/vlm_train_v2.html](../report/vlm_train_v2.html).

| Stage | Config | Steps | Result |
|-------|--------|-------|--------|
| Base LLM (`d12`) | depth 12, n_embd 768, seq 512, vocab 8192 | 15,000 (~245M tok) | loss 3.92 → 2.85 |
| Stage 1 ViT pretrain (`vlm_d12`) | ViT 384/6/6, frozen LLM, AdamW 3e-4 | 5,000 | val loss 3.83 → 2.75 |
| Stage 2 VLM SFT (`vlmsft_d12`) | full VLM (146.5M), MuonAdamW | 3,000 | val loss 4.65 → 2.20 (best @ 900, saved) |
| Inference | `vlm_cli`, temp 0.7 | — | more varied, scene-aware captions (2/6 subject match vs v1's 1/6) |

Checkpoints: `~/.cache/nanochat/{base_checkpoints/d12, vlm_checkpoints/vlm_d12, vlm_sft_checkpoints/vlmsft_d12}`.

**Code change in this run:** `scripts/vlm_sft.py` now tracks the best-validation weights and saves
those (with best step + val loss in meta) instead of the final step — SFT overfits past the val
minimum, so the final checkpoint is usually worse than the best.

## 2d. Training run v3 — DINOv2 pretrained encoder (2026-08-23, remote H200)

Swapped the from-scratch ViT for a **DINOv2 ViT-B/14** pretrained encoder (86M params,
self-supervised on ~142M images). Details, loss curves, and v1/v2/v3 eval examples are in
[report/vlm_train_v3.html](../report/vlm_train_v3.html).

| Stage | Config | Steps | Result |
|-------|--------|-------|--------|
| Base LLM (`d12`) | reused from v2 | — | loss 2.85 |
| Stage 1 projector (`vlm_d12`) | DINOv2 frozen, LLM frozen, train 589,824-param projector, AdamW | 2,000 | val loss 3.95 → 2.68 (min 2.6770) |
| Stage 2 VLM SFT (`vlmsft_d12`) | full VLM (~147M), DINOv2 @ 0.1× LR (AdamW), LLM+projector full LR | 3,000 | val loss 4.43 → 1.81 (best @ 900, saved) |
| Inference | `vlm_cli`, temp 0.7 | — | **6/6 correct main subject** (vs v2's 2/6, v1's 1/6) |

Checkpoints (remote, under `/workspace/nanochat-vlm/.cache/nanochat/`):
`{base_checkpoints/d12, vlm_checkpoints/vlm_d12, vlm_sft_checkpoints/vlmsft_d12}`.

**Code changes in this run:**
- `nanochat/vit.py`: added `DinoViTConfig` + `DinoViT` (LayerNorm + GELU + fused QKV +
  LayerScale, matching DINOv2 so the 175 pretrained keys load 1:1). `DinoViT` drops the CLS
  token, bicubic-interpolates `pos_embed` from 37×37 → 16×16, and projects 256 patch features
  into the LLM embedding space.
- `nanochat/vlm.py`: `VLM(..., encoder="vit"|"dinov2")`; `freeze_vit_backbone()` (only the
  projector trainable); `setup_optimizer()` trains the DINOv2 backbone with AdamW (betas
  0.9/0.95) instead of Muon to preserve pretrained features.
- `scripts/coco_data.py`: `--normalize {unit,imagenet}` — DINOv2 uses ImageNet mean/std.
- `scripts/vit_pretrain.py` / `scripts/vlm_sft.py`: `--encoder`, `--dinov2-weights`,
  `--normalize`, `--vit-lr-scale` args; `torch._dynamo.config.cache_size_limit = 64` (DINOv2
  has ~15 distinct param shapes → fused-kernel recompiles). Encoder type stored in checkpoint
  meta so `load_vlm` rebuilds the right ViT and the CLI applies the right normalization.

**Hardware note:** trained on a remote `NVIDIA H200 NVL` (140 GiB, sm_90) over SSH
(`ssh -p 15021 defaultuser@infom1.apa.at`), all files under `/workspace/nanochat-vlm` (home is
not persisted on that host). The H200 is shared with other users' jobs; the nanochat-vlm jobs
use only ~13.5 GiB and run alongside them.

## 3. Cheat sheet

All commands run from the repo root with the project venv active:

```bash
cd /home/defaultuser/workspaces/metaloom/nanochat-vlm
source .venv/bin/activate
```

### 0. Environment

```bash
# Install / sync dependencies (uv-managed venv)
uv sync

# Verify the GPU is visible (RTX 4090 = GPU 0, RTX 3060 = GPU 1)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1. Dataset curation & processing

```bash
# (a) Base text pretraining data — download ~170 shards (enough for a GPT-2-scale model).
#     Adjust -n up/down for larger/smaller base models.
python -m nanochat.dataset -n 170

# (b) COCO val2014 image-caption data (one-time; ~5000 images, ~37K captions).
#     Stored under ~/.cache/nanochat/coco/
python -m scripts.coco_data
```

### 2. Tokenizer

```bash
# Retrain the BPE tokenizer so it includes the new <|image|> special token.
# (Vocab grows by 1 vs. the text-only tokenizer.)
python -m scripts.tok_train
```

### 3. Base (text-only) model

```bash
# Train a base LLM. Defaults target a full-size model; for a quick small run:
python -m scripts.base_train \
    --depth=12 --max-seq-len=2048 --device-batch-size=16 \
    --total-batch-size=512 --num-iterations=50000

# Tiny CPU smoke run:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 \
    --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
```

### 4. Stage 1 — ViT pretrain (frozen LLM)

```bash
# Requires: base model in ~/.cache/nanochat/base_checkpoints/ and COCO data.
python -m scripts.vit_pretrain

# Tiny CPU smoke run:
python -m scripts.vit_pretrain --num-iterations=10 --device-batch-size=2 \
    --total-batch-size=8 --eval-every=-1 --device-type=cpu
```

Key flags: `--model-tag` / `--model-step` (which base model to freeze), `--image-size`
(128), `--patch-size` (16), `--vit-dim` (256), `--vit-layers` (4), `--vit-heads` (4),
`--vit-lr` (3e-4), `--max-coco-images` (cap for tiny runs).

### 5. Stage 2 — VLM SFT (full fine-tune)

```bash
# Requires: stage-1 VLM in ~/.cache/nanochat/vlm_checkpoints/ (or a base model for a fresh ViT).
python -m scripts.vlm_sft

# Start from a base model with a fresh (untrained) ViT instead of stage 1:
python -m scripts.vlm_sft --source base

# Tiny CPU smoke run:
python -m scripts.vlm_sft --num-iterations=10 --device-batch-size=2 \
    --total-batch-size=8 --eval-every=-1 --device-type=cpu
```

Key flags: `--source vlm|base`, `--model-tag` / `--model-step`, `--embedding-lr` (0.05),
`--unembedding-lr` (0.001), `--matrix-lr` (0.02, Muon), `--vit-lr` (3e-4).

### 6. Inference / demo

```bash
# Describe an image (loads the latest vlm_sft checkpoint by default):
python -m scripts.vlm_cli --image photo.jpg --prompt "Describe this image."

# Point at a specific checkpoint and tune sampling:
python -m scripts.vlm_cli --image photo.jpg --prompt "What is in this image?" \
    --source vlm_sft --model-tag vlmsft_d12 --temperature 0.7 --top-k 50 --max-tokens 256
```

### 7. Tests & smoke test

```bash
# Full non-slow suite (text-only + VLM):
python -m pytest tests/ -m "not slow"

# Just the VLM-related tests:
python -m pytest tests/test_vit.py tests/test_vlm.py tests/test_tokenizer.py -v

# End-to-end self-contained smoke test (no data download, no GPU required):
python -m scripts.vlm_smoke_test
```

### 8. End-to-end from scratch (full pipeline)

```bash
source .venv/bin/activate
python -m nanochat.dataset -n 170      # 1. base text data
python -m scripts.tok_train            # 2. tokenizer (with <|image|>)
python -m scripts.base_train           # 3. base LLM
python -m scripts.coco_data            # 4. COCO image-caption data
python -m scripts.vit_pretrain         # 5. Stage 1: ViT (frozen LLM)
python -m scripts.vlm_sft              # 6. Stage 2: full VLM fine-tune
python -m scripts.vlm_cli --image photo.jpg --prompt "Describe this image."   # 7. demo
```

## 4. Checkpoint layout

| Source | Directory (under `~/.cache/nanochat/`) |
|--------|----------------------------------------|
| Base text model | `base_checkpoints/<model_tag>/` |
| Stage 1 VLM (ViT pretrain) | `vlm_checkpoints/<model_tag>/` |
| Stage 2 VLM (SFT) | `vlm_sft_checkpoints/<model_tag>/` |

`load_vlm(source, device, phase, model_tag=None, step=None)` resolves `source` to the
directory above, picks the largest model tag / last step when not given, and returns
`(vlm, tokenizer, meta_data)`.

## 5. Notes & gotchas

- **FA3 vs SDPA on CPU**: the GPU torch build sets `USE_FA3=True` globally, but the FA3
  kernel only exists on CUDA sm80+/sm90 with bf16. The VLM unit tests force SDPA via an
  autouse fixture (`fa_module._override_impl = 'sdpa'`, `fa_module.USE_FA3 = False`) so they
  pass on CPU.
- **ViT zero gradients on step 1**: GPT-style zero-init projections (`c_proj=0`)
  legitimately yield zero gradients for attention/MLP weights on the first step; only
  `patch_embed`, `pos_embed`, and `projector` are guaranteed non-zero.
- **`loss_reduction='none'`** returns a flattened `(B*T,)` tensor, not `(B, T)`.
- **`build_vlm` on CPU/MPS** upcasts bf16 weights to float32; compare with `.float()` when
  round-tripping.
- **`GPT.forward(embeds=...)`** is backward compatible — the text-only path is unchanged
  when `embeds` is `None`.
