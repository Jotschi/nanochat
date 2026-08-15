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

Corpus: **133,204 stories / 128,303,041 chars = 28.0M tokens** measured with the trained
tokenizer (see [DATA.md](DATA.md)).

Model: **d12**, `--vocab-size 8192`.

The vocabulary size is settled: at 8,192 the tokenizer compresses this corpus at
**4.58 chars/token**, beating GPT-2 (2.67) and GPT-4 (3.48) on the same text. There is no
case for going to 16,384 — that would only inflate the embedding table.

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

Do **not** rely on `--target-param-data-ratio` here. Its default of 12 implies ~1B tokens ≈ 36 epochs
over a 28M-token corpus. Instead:

```
num_iterations = epochs × corpus_tokens / total_batch_size
```

`python -m nanochat.kleiner_astronaut --stats` prints `corpus_tokens` (measured with the real
tokenizer when one exists) and the implied iteration count. `runs/astronaut.sh` derives
`--num-iterations` from it automatically.

**Batch size: 131,072 tokens, not the usual 524,288.** At a 512k batch this corpus gives only 65
optimizer steps per epoch, which is far too few updates. A quarter of that gives 214 steps/epoch.
nanochat rescales the LR by `sqrt(B/B_ref)` on its own, so this needs no matching LR change.

At 4 epochs that is **855 iterations**. Repeating data up to ~4 epochs is close to as good as fresh
data (Muennighoff et al., *Scaling Data-Constrained Language Models*); past that returns diminish but
stay positive out to ~16. Raise `EPOCHS` while `val_bpb` still falls, stop when it turns.

## Hardware

Single **RTX 4090** (24 GB, SM 89). The RTX 3060 (12 GB, SM 86) is not worth pairing under DDP — it
would halve throughput and cap the device batch size.

**The 4090 is shared with the LLM server.** Stages 1 and 2 of the data pipeline need llama.cpp or
vLLM, and a 24B model in that container occupies ~20 GB, leaving nothing for training — a training
run started underneath it dies with `torch.OutOfMemoryError` a few hundred MiB short. `nvidia-smi`
will name the holder:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
docker ps --format '{{.Names}} {{.Image}}'
```

Options when they collide, in order of preference:
1. Run the two phases at different times — generation is a one-off, training is not.
2. Put the short stages (SFT, evals) on the 3060: `CUDA_VISIBLE_DEVICES=1` plus a smaller
   `DEVICE_BATCH_SIZE` (4 fits comfortably; pretraining at 16 peaks at 13.0 GiB and will not fit).
3. Pin the LLM server to the 3060 instead and keep the 4090 for training, if the model is small
   enough to fit in 12 GB.

`runs/sandbox.sh` honours `CUDA_VISIBLE_DEVICES` from the environment and only defaults it to 0.

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
