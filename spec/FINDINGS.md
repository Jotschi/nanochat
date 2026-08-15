# Findings

Why the previous attempt (Nov–Dec 2025) failed, and every defect found while reconstructing it.

## 1. Mid-training destroyed a healthy base model

Evidence: `nanochat-old/wandb/run-20251126_220518-8f72y886/` (`config.yaml` + `output.log`), code at
commit `cb3179f`.

```
Step 00000 | Validation bpb: 0.2982      <- healthy base model
step 00001 | loss: 13.070236 | lrm: 1.00
step 00002 | loss: 16.036624 | lrm: 1.00 <- above uniform-random (ln 65536 = 11.09)
step 00150 | Validation bpb: 1.7252      <- destroyed inside 150 steps
step 00401 | loss:  6.900375 | lrm: 1.00
step 01801 | loss:  6.880404 | lrm: 1.00
step 02381 | loss:  6.870882 | lrm: -0.00 <- flat at ~6.88 for 2200 steps
Step 02381 | Validation bpb: 1.6045      <- 5h21m bought back nothing
```

### Primary cause: the learning rate never decayed, at full pretraining magnitude

`num_iterations = 100000`, `init_lr_frac = 1.0`, and

```python
def get_lr_multiplier(progress):
    # first 80% of training: no decay, then linearly ramp down to 0.
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2
```

with `progress = it / num_iterations`. The run stopped at step 2381 → `progress = 0.024`, so **`lrm`
stayed pinned at 1.00 for the entire run**. Mid-training restarted a converged model at the full
pretraining peak LR (`embedding_lr = 0.2`, `matrix_lr = 0.02`) and never warmed down.

### Second cause: ~200 epochs over 15,600 conversations

`CustomJSON(5000, filepath=…)` duplicated every conversation 5,000×. The log's
`Dataset len: 78000000` ÷ 5000 = **15,600 unique conversations** (val: 810). At
`total_batch_size = 524288`, 2,381 steps ≈ **1.25B tokens** — roughly 200 epochs over the same data,
at constant peak LR. The epoch-end stop had been commented out:

```python
if cursor >= dataset_size:
    cursor -= dataset_size
    #if split == "train":
    #    last_step = True     # <- disabled
```

### Third cause: the model was far too large for the data

d15/d16 with `vocab_size = 65536` is ~200M parameters, ~134M of them embedding/unembedding. Most of a
65k vocabulary receives almost no gradient from a German story corpus. `val_bpb 0.2982` reflected
memorization rather than generalization, and the validation conversations were built from training
stories, so validation was contaminated.

### Why the 6.88 plateau is diagnostic

6.88 nats ≈ 9.9 bits is approximately the **context-free unigram entropy** of the token distribution.
A model sitting there has learned marginal token frequencies and nothing conditional — consistent
with a blown-out embedding table that the constant peak LR prevented from recovering.

Corroborating run `run-20251127_184201-dybml0fd`: `train/loss 6.9159` while `val/bpb 0.1818`.
Also `run-20251205_193847-ilvx8ulv` (`chat_sft`): `train_loss 0.0412` vs `val_loss 8.5029` — textbook
memorization.

## 2. Data-path defects

| Where | Defect |
| --- | --- |
| `KleinerAstronautJsonlHandler.loadWords`, `KleinerAstronautParquetHandler` | `word2Needle` derived from **`word1`**, not `word2` — the second keyword was never verified against the story, so `rl_key2` is unreliable **in the data itself** |
| `tasks/customjson.py` | The same bug mirrored downstream: the `rl_key2` branch re-reads `rl_key1`, so `rl_key2` was never enforced during RL either |
| `tasks/customjson.py` | `key[:-2]` stemming copied from `QAGenerator` but without its `length > 5 && !isNumber` guards → short keys mangled |
| `tasks/customjson.py` | Reward keys read off the message dict while only `role`/`content` are validated → missing keys silently return `True`, i.e. **reward 1.0 for every rollout** |
| `Dataset2ChatTest` | Validation slice is the **first 5% of the file, unshuffled**; `val_size >= 0` also gives one extra line |
| `SplitDatasetTest` | Shuffles, but **unseeded** → a different split on every run |
| both converters | `Charset.defaultCharset()` rather than explicit UTF-8; file reopened per line via `writeStringToFile(..., append=true)` |
| `Dataset2ChatTest` | The QA turns were commented out — the last traces were request→story only, 2 messages |
| stage 1/2 | All RNG unseeded, temperature 1.0, 24/10 virtual threads appending concurrently → non-reproducible; executors `awaitTermination`-ed but never `shutdown()` |
| `AbstractGeneratorTest` | `getOllamaURL()` read key `ollama.host` while the config defined `ollama.url` → always null (moot now, Ollama removed) |

### Measured impact of the `word2Needle` bug

The rebuilt converter only attaches a reward key when the keyword can actually be
found in the text it describes. Over all 25,709 conversations:

| Story-turn reward keys retained | Conversations | Share |
| --- | --- | --- |
| both `rl_key1` and `rl_key2` | 14,947 | 58.1% |
| one of the two | 8,143 | 31.7% |
| neither | 2,619 | 10.2% |

So **~42% of story turns carried at least one keyword that does not occur in the
reference story**. Under the old code those keys were shipped anyway, giving the
RL stage targets it could not satisfy. The answer-turn key survives in 14,289 of
14,289 cases (100%) — `QAGenerator`'s quality gate did verify that one.

## 2b. First successful base run (d12, 855 iterations)

The failure mode from §1 is gone. `lrm` decayed all the way to 0.05 by the final
step, i.e. **the LR schedule actually completed** rather than sitting pinned at
1.00. 14.9 minutes on one RTX 4090, 58% bf16 MFU, ~125k tok/s, peak 13.0 GiB.

Samples after pretraining are coherent German children's stories, e.g.

> Es war einmal ein kleiner Astronaut namens Kippo, der in einem Raumschiff
> namens "Sternenwind" lebte. Eines Tages entdeckte er auf seinem Bildschirm
> einen riesigen, leuchtenden Kometen, der direkt auf den Planeten "Mondlicht"
> zuraste. …

**Validation bpb turned upward.** Minimum 1.3139, final 1.4006. With warmdown
driving the LR to ~0 the last evaluation should normally be the best one, so a
rise means the model is overfitting the corpus rather than that the schedule is
wrong.

## 2c. Where the corpus runs out: the 6-epoch sweep

`logs/sweep_e6_partial.log`, d12, 1,282-iteration schedule, evaluating every 50 steps:

| step | val bpb | | step | val bpb |
| --- | --- | --- | --- | --- |
| 0 | 2.9274 | | 400 | 1.3470 |
| 50 | 1.7459 | | 450 | 1.3376 |
| 100 | 1.6447 | | 500 | 1.3325 |
| 150 | 1.5283 | | **550** | **1.3296** ← minimum |
| 200 | 1.4373 | | 600 | 1.3487 |
| 250 | 1.3917 | | 650 | 1.3503 |
| 300 | 1.3676 | | 700 | 1.3572 |
| 350 | 1.3537 | | | |

The turn at ~550 steps (≈2.6 epochs) is real overfitting, not a schedule
artifact: `warmdown_ratio 0.65` means warmdown began at step 449, so the learning
rate was **already decaying** when validation started getting worse. The run was
stopped at 700 once the trend was unambiguous.

**A checkpoint from the middle of a long schedule is not the model you want** —
at step 550 of 1,282 the LR is still mid-warmdown. The horizon has to be chosen
so the schedule *ends* near the minimum, which is why the next run is a full
600-iteration schedule rather than an early checkpoint of this one.

**Process note:** the first run was piped through `tail -60`, which discarded
every intermediate evaluation. Long runs write to
`$NANOCHAT_BASE_DIR/logs/` instead.

## 3. Environment findings

- **Upstream deleted the mid-training stage.** No `scripts/mid_train.py`, no `tasks/customjson.py`,
  no `nanochat/report.py`, no `configurator.py`, no `base_loss.py`, no vendored `rustbpe/` crate.
  Scripts moved from the `configurator` idiom to argparse with **dashed** flags.
- **The architecture changed**, so Nov/Dec 2025 checkpoints cannot be loaded: value embeddings,
  `resid_lambdas`, `x0_lambdas`, smear gate, sliding-window attention, RoPE base 10000 → 100000.
  `load_state_dict(strict=True)` and the new keys are not patched by `_patch_missing_keys`.
- **No FA3 on this box** (RTX 4090 = SM 89, RTX 3060 = SM 86). SDPA fallback; use
  `--window-pattern L` and no `--fp8` (that needs Hopper).
- **Old runtime artifacts are gone** — the tokenizer and every checkpoint lived in a podman
  bind-mount that was deleted. Keep `NANOCHAT_BASE_DIR` on a real host path.

## 4. Dataset version lineage (corrects an earlier assumption)

There was never a `conversations_v4`. Two independent series:

- **Base data**: `_enhanced` (v1) → `_v2` → `_v3` → `_v4`/`_v4_hashed` → `_v5_combined` → `_v6`
- **Conversations**: `conversations.jsonl` → `_v2` → **`_v3_{train,val}`** → `_v5_{train,val}` →
  unversioned `conversations_{train,val}.jsonl`

The 15,600 train / 810 val artifact is the **v3** output from commit `a30c3ea "Update RL keys"` —
the same commit that introduced both `rl_key1`/`rl_key2` and the 5% split.
