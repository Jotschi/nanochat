# Progress

## Phase 0 — Repo unification

- [x] Create `kleiner-astronaut` branch in the nanochat checkout
- [x] Write the `spec/` notes tree (GOALS, FINDINGS, DATA, TRAINING_STRATEGY, PROGRESS)
- [ ] Import `dataset-processor` via `git subtree add --prefix=dataset-processor`
- [ ] Copy the 13 in-flight (uncommitted) generator changes and commit them separately
- [ ] Carry forward `.gitignore` for `dataset-processor/dataset/` and `config/`
- [ ] Add `config/settings.properties.example` with placeholders (no internal hostnames)

## Phase 1 — Make the Java generator build

- [ ] Delete `Models.java` and every Ollama code path/test
- [ ] Reduce `VLLMModel` to the 3-method `LargeLanguageModel` contract (drop `providerType()`)
- [ ] Use `new OpenAILLMProvider()` everywhere
- [ ] Drop `ollama.url` from settings; fix `nanochat.cache.dir` to a path that exists
- [ ] Resolve the `<release>24</release>` vs Temurin 21/25 mismatch
- [ ] Confirm `mvn -q compile` succeeds after installing genai-utils + hash-utils

## Phase 2 — Fix and promote the converters

- [ ] `Dataset2Chat` → `main()`, **re-enable the QA turns** (4-turn when `question`/`answer` present)
- [ ] `SplitDataset` → `main()`
- [ ] Seeded, story-level split shared by both converters
- [ ] Fix `word2Needle` to derive from `word2`
- [ ] Explicit UTF-8; single buffered writer
- [ ] `shutdown()` the executors; seed the RNGs
- [ ] Exclude the infinite-loop ETL jobs from surefire

## Phase 3 — nanochat data bridge

- [ ] `nanochat/kleiner_astronaut.py`: dedup by `hash`, write parquet shards, held-out split last
- [ ] `NANOCHAT_DATA_DIR` env override in `nanochat/dataset.py`
- [ ] `--stats` mode printing corpus chars/tokens and the implied iteration count

## Phase 4 — Training code

- [ ] `tasks/kleiner_astronaut.py` (replaces `CustomJSON`, reward bugs fixed, no `it` multiplier)
- [ ] Wire the task into `scripts/chat_sft.py` behind a flag
- [ ] German sample prompts in `scripts/base_eval.py`
- [ ] German story-QA eval for `scripts/chat_eval.py`
- [ ] `runs/astronaut.sh`

## Phase 5 — Verification

- [ ] `mvn -q test` green (converter tests against `mock-llm-server`)
- [ ] Parquet round-trip: 133,204 rows / 128,303,136 chars; last shard is the held-out split
- [ ] `tok_eval` compression ratio 3.5–4.5 chars/token → confirm or revise vocab size
- [ ] Smoke run `--depth=4 --num-iterations=20`, loss falls from ≈9.01
- [ ] Full base run; `val_bpb` monotone, `lrm` reaches ~0
- [ ] SFT run; step-0 loss sane, train/val move together
- [ ] `chat_cli` end-to-end: story request, then a follow-up question
- [ ] `pytest tests/ -m "not slow"` still passes

## Open questions

- Vocab 8192 vs 16384 — settled by the `tok_eval` compression ratio in Phase 5.
- Whether to include the 27,908 HF stories (`--include-hf`); different generation, avg 1,957 chars
  vs 963, so it shifts the length distribution.
- Extending QA coverage beyond the 14,289 v5 rows needs an LLM endpoint; deferred.
