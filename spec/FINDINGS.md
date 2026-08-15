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
so the schedule *ends* near the minimum.

### Horizon runs

| run | steps | `lrm` floor | min val bpb | final val bpb | gap |
| --- | --- | --- | --- | --- | --- |
| first | 855 | 0.05 | 1.3139 | 1.4006 | +6.6% |
| sweep | 1,282 (stopped at 700) | 0.05 | 1.3296 @550 | — | rising |
| horizon fix | 600 | 0.05 | 1.3227 @500 | 1.3330 | +0.8% |
| + anneal to zero | 600 | 0.00 | 1.3231 @500 | 1.3300 | +0.5% |

Shortening the horizon did nearly all the work, cutting the gap from 6.6% to
0.8%. Annealing to zero (`--final-lr-frac 0`, against upstream's 0.05) took it to
0.5% — worth keeping, but it confirms the 5% LR floor was a minor contributor
rather than the cause. What remains is the model saturating the corpus: the last
~15% of any schedule overfits slightly, whatever the LR does.

**Stopping here.** 1.3227, 1.3231 and 1.3300 are all within 0.5% of each other,
so the plateau is flat and further horizon tuning buys nothing measurable. The
d12 base model used downstream is the 600-step, fully-annealed one.

**Process note:** the first run was piped through `tail -60`, which discarded
every intermediate evaluation. Long runs write to
`$NANOCHAT_BASE_DIR/logs/` instead.

## 2d. SFT and the first end-to-end model

d12, 74 steps (2 epochs over 24,407 conversations), on the 3060 at
`DEVICE_BATCH_SIZE=4` because the 4090 was hosting llama.cpp. 10.5 minutes.

**The December failure mode does not reproduce.** Step-0 validation bpb is
**1.3428** against the base model's **1.3300** — essentially continuous. Compare
the old run, where step 1 was 13.07, *above* uniform-random for the vocabulary.
Training loss rises from 2.03 to a peak of 3.54 around step 21 as the model meets
the chat special tokens for the first time, then recovers to 2.46 as warmdown
proceeds. `lrm` reaches 0.00.

**SFT overfits at 2 epochs.** Validation bpb went 1.2678 → 1.3664. One epoch is
the obvious next thing to try.

### First numbers on the actual goal

`scripts/chat_astronaut_eval -i sft --max-problems 120`:

| metric | result |
| --- | --- |
| Keyword adherence (story contains both requested words) | 32/97 = **32.99%** |
| Question answering (answer contains `answer_word`) | 47/120 = **39.17%** |

Keyword adherence is the weaker of the two, which is consistent with the training
data: only 58.1% of story turns carry both reward keys (§2), so the two-keyword
signal is diluted.

### Qualitative check

```
USER  : Schreib ein Abenteuer von Mira in das Raumschiff nur mit Aris
ASSIST: In einem riesigen Raumschiff namens "Sternenblitz" lebt der kleine
        Astronaut Mira. Eines Tages entdeckt er auf einem fernen Planeten ein
        seltsames, blinkendes Licht, das wie ein großer Diamant aussieht. ...
USER  : Wer fliegt mit dem Raumschiff?     ASSIST: Mira fliegt mit dem Raumschiff.
USER  : Wo spielt die Geschichte?          ASSIST: In der fernen Galaxie.
USER  : Was hat Mira entdeckt?             ASSIST: Mira hat einen großen Diamanten entdeckt.
```

The last answer is grounded reading comprehension — the story described the light
as looking like "ein großer Diamant" and the model retrieved it. Note the story
also drops "Aris", which is what the 33% keyword score measures.

Stories end abruptly and sometimes degenerate ("Es kommen direkt aus dem Licht,
das aus dem Licht kommt"). That traces to the training data: v6 stories are
`softClamp`ed to 300 characters, so the model never saw many long, well-resolved
endings.

### Open question

Each SFT evaluation prints **two** `Validation bpb` lines with different values
(1.3428 and 1.2678 at step 0; 1.4303 and 1.3664 at step 74). There is exactly one
`print0` site, one `if` guard, and one process, and `last_step` is correctly
guarded by `split == "train"` so the val loader is not ending training early.
Unexplained. It does not change the conclusion — both series rise by ~0.09 — but
it is worth understanding before trusting the absolute numbers.

## 2e. vLLM serving: the architecture has moved ahead of transformers

`vllm serve` runs nanochat models through transformers' `NanoChatForCausalLM`
(vLLM has no native kernel; it uses `--model-impl transformers`). That
implementation targets an **older nanochat architecture than this repo trains**.

`python vllm/convert_to_hf.py --check` on the d12 SFT checkpoint:

```
Tensors    : 91
Mapped     : 74
Unsupported by transformers' NanoChat (17 tensors):
  per-layer value embeddings added to V
  gate blending the value embedding into V
  token-smearing gate over the previous token / scale for the smear gate
  backout of the first block's contribution
  per-layer residual stream scaling / re-blending of the initial embedding
```

Those 17 tensors carry trained weights. Dropping them does not raise — the model
loads and generates, just not the model that was trained — so the converter
refuses unless given `--force`.

One difference *is* benign: nanochat normalises with parameter-free
`F.rms_norm` while transformers uses learnable RMSNorm. An all-ones weight makes
them identical, and the converter synthesises those.

Installed for reference: vLLM 0.27.1, transformers 5.15.0 (ships `models/nanochat`),
torch 2.13.0+cu130, in `vllm/.venv` — deliberately separate from the training venv,
which pins torch 2.9.1.

The serving path itself is **verified** against `nanochat-students/nanochat-d20`
on the 3060: server up, `vllm/test.sh` returning both completions.

`vllm/start-vllm.sh` runs the **official `vllm/vllm-openai` container** rather
than pip-installing vLLM. The deciding reason is the CUDA toolkit: the
gpu-sandbox image is a runtime-only PyTorch base, so a pip install of vLLM there
dies at engine start with

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

because flashinfer JIT-compiles its kernels. (Attention was unaffected — vLLM
picked FLASH_ATTN — only the sampler reaches for flashinfer, so
`VLLM_USE_FLASHINFER_SAMPLER=0` also works as a pip-install workaround.) The
official image carries the toolkit and needs no venv, which also keeps the
training venv's pinned torch 2.9.1 untouched.

Two more fixes, both baked into the script:

| symptom | cause | fix |
| --- | --- | --- |
| every chat request returns HTTP 400 `User messages must contain string content` | vLLM's OpenAI server rewrites content into a list of parts; nanochat's chat template indexes it as a plain string | `--chat-template-content-format string` |
| an index selects the wrong card | with a 4090 and a 3060 the default CUDA ordering is fastest-first, so `CUDA_VISIBLE_DEVICES=1` need not be the 3060 | select at the docker layer: `--gpus '"device=1"'` |

The 400 initially looked like the model returning empty completions, because
`curl -f` exits non-zero and prints nothing on an HTTP error. `vllm/test.sh`
deliberately omits `-f` and surfaces the server's message.

Versions confirmed working: vLLM 0.27.1, transformers 5.15.0 (ships
`models/nanochat`), torch 2.13.0+cu130.

## 2f. The vLLM-compatible architecture is *better* on this corpus

vLLM support for nanochat is real:
[transformers#41634](https://github.com/huggingface/transformers/pull/41634) merged
on 27 Nov 2025 with explicit vLLM work in its commits ("move attention into func
and add kwarg to all signatures (for vllm)", "nanochat config is in all (fixes
vllm)"). Verified here by serving `nanochat-d20` through
`vllm/vllm-openai:latest` with `--model-impl transformers`.

To close the architecture gap from our side, `GPTConfig.hf_compatible`
(`base_train --hf-compatible`) drops the components added after that port: value
embeddings and their gates, the token-smear gate, the backout term, and the
per-layer residual/x0 lambdas. They are *not constructed* rather than merely
unused — an `nn.Parameter` that exists lands in the checkpoint regardless. State
dict goes from 91 tensors to exactly the **74** the converter maps.

The expectation was that this would cost accuracy. It did the opposite:

| d12, 600 steps, `--final-lr-frac 0` | min val bpb | final val bpb |
| --- | --- | --- |
| full architecture | 1.3231 @500 | 1.3300 (rose over the last 100 steps) |
| `--hf-compatible` | **1.3209 @600** | **1.3209** |

The simplified model is better *and* its curve is still falling at the horizon,
so it satisfies G2 cleanly where the full architecture did not. The likely reason
is capacity: value embeddings alone add `6 × vocab × kv_dim` parameters, and §2c
already showed this 28M-token corpus saturating a d12. Those components are
presumably worth their parameters at nanochat's intended data scale; at ours they
are extra surface to overfit.

SFT on top of it, **1 epoch** (2 overfit last time): val bpb 1.2426 → **1.2135**,
last eval is the minimum. The end-to-end model now serves under vLLM:

```
USER  : Schreib ein Abenteuer von Mira in das Raumschiff nur mit Aris
ASSIST: In einem weit entfernten Teil des Weltraums, wo die Sterne wie funkelnde
        Diamanten leuchten, lebte ein kleiner Astronaut namens Mira. ... Es war
        ein Quasar, ein riesiger, leuchtender Stern ...
USER  : Was hat Mira entdeckt?
ASSIST: Mira wollte herausfinden, was das für ein Quasar war.
```

Three traps on the way there, all now handled in `vllm/convert_to_hf.py`:

1. **`checkpoint_manager._patch_missing_keys` fought the new flag.** It injects
   `resid_lambdas`/`x0_lambdas` defaults for old checkpoints, then
   `load_state_dict(strict=True)` rejects them as *unexpected* keys on an
   hf_compatible model. It now skips patching when the config says so.
2. **Norms are parameter-free in transformers' NanoChat too.** Synthesising
   all-ones `input_layernorm`/`q_norm` weights — a reasonable guess, and what the
   first version did — makes vLLM fail with "There is no module or parameter
   named model.layers.0.input_layernorm.weight". The reference d20 has exactly
   `20*6+2 = 122` tensors; ours has `12*6+2 = 74`, a 1:1 match with no synthesis.
3. **The tokenizer needed rebuilding, and needed checking.** tiktoken stores only
   token→rank, so merges are recovered by re-running BPE per token against
   lower-rank merges. The converter round-trips samples through both tokenizers
   and fails on any id mismatch — a silently wrong tokenizer presents as "the
   model is just bad", and German umlauts are exactly where a lossy byte→str
   conversion breaks.

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
