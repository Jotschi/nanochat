# Progress

## Phase 0 — Repo unification

- [x] Create `kleiner-astronaut` branch in the nanochat checkout
- [x] Write the `spec/` notes tree (GOALS, FINDINGS, DATA, TRAINING_STRATEGY, PROGRESS)
- [x] Import `dataset-processor` via `git subtree add --prefix=dataset-processor` (8 commits preserved)
- [x] Copy the 13 in-flight generator changes and commit them separately
- [x] Scope `.gitignore` so `dataset/` (180 MB) and the real `config/settings.properties` stay out
- [x] Add `config/settings.properties.example` with placeholders (no internal hostnames)

## Phase 1 — Make the Java generator build

- [x] Delete `Models.java` and every Ollama code path/test
- [x] Reduce `VLLMModel` to the 3-method `LargeLanguageModel` contract (drop `providerType()`)
- [x] Use `new OpenAILLMProvider()` everywhere; add `llm()`/`model()` helpers
- [x] Drop `ollama.url`; add `llm.model`/`llm.context.window`; fix `nanochat.cache.dir`
- [x] Compile with `release 21` (gpu-sandbox ships Temurin 21 and 25, not 24)
- [x] `mvn -q compile` and `test-compile` succeed

## Phase 2 — Fix and promote the converters

- [x] `Dataset2Chat` → `main()`, **QA turns re-enabled** (14,289 four-turn + 11,420 two-turn)
- [x] `SplitDataset` → `main()`
- [x] Seeded, story-level split shared by both converters (hash bucket, no shuffle)
- [x] Fix `word2Needle`, and only attach reward keys that really occur in the text
- [x] Explicit UTF-8; single buffered writer per split
- [x] `shutdown()` the executors
- [x] Exclude the infinite-loop ETL jobs from surefire
- [x] 19 offline tests green (`Split`, `Words`, `Dataset2Chat`, `SplitDataset`)

## Phase 3 — nanochat data bridge

- [x] `nanochat/kleiner_astronaut.py`: dedup by `hash`, parquet shards, held-out split last
- [x] `NANOCHAT_DATA_DIR` override in `nanochat/dataset.py`, raising when it points nowhere
- [x] `--stats` prints corpus size and the implied `--num-iterations`
- [x] Round-trip verified through `parquets_iter_batched`: 133,204 rows / 128,303,041 chars

## Phase 4 — Training code

- [x] `tasks/kleiner_astronaut.py` (replaces `CustomJSON`; reward bugs fixed; no `it` multiplier)
- [x] `--kleiner-astronaut` flag in `scripts/chat_sft.py`, ChatCORE auto-disabled
- [x] German sample prompts via `NANOCHAT_SAMPLE_PROMPTS` in `base_train` and `base_eval`
- [x] `scripts/chat_astronaut_eval.py` — keyword adherence + question answering
- [x] `runs/sandbox.sh` (gpu-sandbox wrapper, resolves the real host bind path)
- [x] `runs/astronaut.sh` (full pipeline, horizon derived from the corpus)

## Phase 5 — Verification

- [x] `mvn -q test` green — 19 tests, no LLM endpoint needed
- [x] Parquet round-trip: 133,204 rows / 128,303,041 chars; last shard is the held-out split
- [x] Tokenizer: **4.58 chars/token** at vocab 8192, beating GPT-2 (2.67) and GPT-4 (3.48)
- [x] Smoke run `--depth=4 --num-iterations=20`: loop works, German text already emerging
- [x] Full base run — d12, **600 iterations**, `lrm` reaches 0.00, final val bpb **1.3300**
- [x] Horizon located: val bpb bottoms at ~500-550 steps (see FINDINGS §2c)
- [x] SFT run — 74 steps, `lrm` reaches 0.00, step-0 val bpb **1.3428** vs base 1.3300 (no blow-up)
- [x] `chat_astronaut_eval` baseline — story keywords **32.99%**, question answering **39.17%**
- [x] End-to-end: writes a German story on request, then answers questions about it correctly
- [x] Upstream tests still pass — 58 passed

      bash runs/sandbox.sh bash -c 'uv run --group dev python -m pytest tests/ -m "not slow" -q'

      Note the `python -m`: plain `uv run pytest` collects nothing because the repo root is
      not on `sys.path`, and every test module fails to import.

## Measurements so far

| Quantity | Value |
| --- | --- |
| Unique stories | 133,204 |
| Characters | 128,303,041 |
| Tokens (vocab 8192) | 28,003,629 |
| Compression | 4.58 chars/token |
| Conversations | 25,709 (14,289 four-turn, 11,420 two-turn) |
| Held out | 6,644 stories / 1,302 conversations (~5%) |
| Story turns keeping both reward keys | 58.1% |

## Next steps, in order of expected value

1. **SFT for 1 epoch instead of 2.** Val bpb rose 1.2678 → 1.3664 over two epochs.
   `SFT_EPOCHS=1`.
2. **Raise `STORY_MAX_LEN`.** v6 stories are clamped to 300 characters in
   `KleinerAstronautJsonlHandler`, so the model never learns to finish a story —
   generations trail off or repeat. The raw corpus averages 963 characters.
   Regenerating stage 2 with a larger clamp needs the LLM endpoint.
3. **Regenerate QA coverage over the full corpus.** Only 14,289 of 133,204 stories
   have a question/answer pair; the rest can only teach request → story. This is
   the single biggest lever on the 39% QA number, and needs the LLM endpoint.
4. **Improve reward-key quality.** 42% of story turns lack a verifiable second
   keyword, which caps keyword adherence. Fixing `word2Needle` (done) helps future
   generations; existing rows keep the gap.
5. **RL on the keyword reward** (`scripts/chat_rl.py` with `KleinerAstronaut`),
   once SFT is settled.

## Open questions

- Vocab size: **settled at 8192** by the compression ratio.
- Whether to include the 27,908 published HF stories (`--include-hf`); different generation,
  avg 1,957 chars vs 963, so it shifts the length distribution.
- Extending QA coverage beyond the 14,289 v5 rows needs an LLM endpoint; deferred.
- Whether 4 epochs is the right stopping point — raise while `val_bpb` still falls.
