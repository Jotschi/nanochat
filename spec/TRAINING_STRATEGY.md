# Training strategy

## The one rule

> **Never set a training horizon the run will not reach.**

Every nanochat LR schedule is expressed as a fraction of `num_iterations`. If `num_iterations` is
larger than the number of steps actually run, the model trains at peak LR forever and warmdown never
happens. That single mistake destroyed the previous model (see [FINDINGS.md](FINDINGS.md) §1).

Corollaries:
- Express repetition as **epochs**, never as a dataset-duplication multiplier. The LR schedule can
  see epochs; it cannot see `CustomJSON(5000, …)`.
- Always confirm in the log that `lrm` reaches ~0 at the final step.

## Scale

Corpus: **133,204 stories / 128.3M chars ≈ 34M tokens** (see [DATA.md](DATA.md)).

Model: **d12**, `--vocab-size 8192`.

| | |
| --- | --- |
| `model_dim` | `depth × aspect_ratio` = 12 × 64 = 768 |
| `n_head` | `model_dim / head_dim` = 768 / 128 = 6 |
| non-embedding params | ≈ 12 · n_layer · n_embd² ≈ 85M |
| embedding + unembedding | 2 × 8192 × 768 ≈ 12.6M |
| **total** | **≈ 98M** |

Rationale: the previous attempt used d15/d16 at vocab 65,536, where embeddings were ~⅔ of the model
and most of the vocabulary saw almost no gradient. At 128M chars of German, an 8,192-token vocabulary
keeps the embedding table small relative to the transformer. If `tok_eval` reports a poor compression
ratio (below ~3.5 chars/token), 16,384 is comfortably supportable at this corpus size — decide
**before** spending GPU hours, because the vocabulary is baked into every downstream checkpoint.

## Token budget

Do **not** rely on `--target-param-data-ratio` here. Its default of 12 implies ~1B tokens ≈ 30 epochs
over a 34M-token corpus. Instead:

```
num_iterations = epochs × corpus_tokens / total_batch_size
```

Start at **3–4 epochs**. `python -m nanochat.kleiner_astronaut --stats` prints `corpus_tokens` and the
implied iteration count for a given batch size.

## Hardware

Single **RTX 4090** (24 GB, SM 89). The RTX 3060 (12 GB, SM 86) is not worth pairing under DDP — it
would halve throughput and cap the device batch size.

- `CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1`
- `--window-pattern L` — Flash Attention 3 is Hopper-only, so attention falls back to SDPA, and
  `nanochat/flash_attention.py` warns loudly for any non-`L` pattern on the fallback path.
- **No `--fp8`** — Hopper only.
- bf16 is selected automatically (`COMPUTE_DTYPE`).

## Stage settings

### Pretraining (`scripts/base_train`)

Upstream defaults are sound: `--warmup-steps 40`, `--warmdown-ratio 0.65`, `--final-lr-frac 0.05`,
cosine-decayed weight decay. Override only `--depth`, `--num-iterations`, `--device-batch-size` and
`--window-pattern`.

### SFT (`scripts/chat_sft`)

Use upstream's schedule as-is — it is the direct fix for the previous failure:

| Flag | Value | Why |
| --- | --- | --- |
| `--init-lr-frac` | 0.8 | starts below the pretraining peak instead of at it |
| `--warmdown-ratio` | 0.5 | half the run is warmdown |
| `--final-lr-frac` | 0.0 | actually reaches zero |
| `--num-iterations` | -1 | exactly one epoch, horizon known in advance |

SFT data: ~25.7K conversations (14,289 four-turn from v5 + 11,420 two-turn from v6). Repeat via
epochs if needed, never via row duplication.

## Red flags during a run

| Symptom | Meaning |
| --- | --- |
| `lrm` flat at 1.00 late in the run | horizon exceeds the actual step count — **stop the run** |
| step-1 loss above `ln(vocab_size)` | targets are wrong or special tokens are unseen; investigate before burning hours |
| loss plateaus at ~7 nats and stops moving | collapse to the unigram distribution |
| `train_loss` ≪ `val_loss` and diverging | memorization; reduce epochs or model size |
| `val_bpb` rising | the model is being damaged, not trained |
