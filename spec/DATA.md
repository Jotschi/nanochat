# Data

## Pipeline

```
stage 1  StoryGenerator          word lists + LLM  ->  dataset/stories*.jsonl        (raw stories)
stage 2  StoryProcessor          stories + LLM     ->  dataset/kleiner_astronaut_qa_v*.jsonl
stage 3a Dataset2Chat            qa_v*             ->  conversations_{train,val}.jsonl   (SFT/RL)
stage 3b SplitDataset            qa_v*             ->  astronaut_basedata/..._{train,test}.jsonl
stage 4  nanochat.kleiner_astronaut   stories*.jsonl -> base_data_astronaut/*.parquet  (pretraining)
```

Stages 1 and 2 need an OpenAI-compatible LLM endpoint (vLLM or llama.cpp). Stages 3 and 4 do not.
**We do not need to re-run stages 1–2** — 133K stories already exist on disk.

## Corpus sizes (measured)

| Source | Rows | Chars |
| --- | --- | --- |
| `dataset/stories.jsonl` | 9,997 | 9,272,181 |
| `dataset/stories_done/stories.jsonl` | 29,864 | 29,018,546 |
| `dataset/stories_done/stories2.jsonl` | 75,149 | 73,163,183 |
| `dataset/stories_done/stories3.jsonl` | 7,651 | 6,693,636 |
| `dataset/stories_done/stories4.jsonl` | 867 | 791,712 |
| `dataset/stories_done/stories_full_v2_hashed.jsonl` | 9,736 | 9,423,208 |
| **deduplicated by `hash`** | **133,204** | **128,303,136** |

Average 963 chars/story; ≈ **34M tokens** at ~3.8 chars/token.

For comparison, the published HF dataset `Jotschi/kleiner-astronaut` is a different, older generation
(Phi-3, April 2024): 22,326 train + 5,582 test = 27,908 stories / 54.6M chars, avg 1,957 chars, with
schema `text`, `topic`, `word_1`, `word_2`, `adjective_1`, `adjective_2`, `verb`. It is **opt-in**
here via `--include-hf`.

## Schemas

### Stories (stage 1/2 output) — the pretraining corpus

```json
{"hash":"5b638b57e71644ecc63512b3d913f385","text":"Ein Raumschiff namens \"Sternschnupp\" …",
 "verb":"helfen","word":"Trolle","topic":"Mondlandung","spaceWord":"Komet",
 "adjective1":"Spannend","adjective2":"Abenteuerlich","names":["Cosmo","Zora"]}
```
`hash` is MD5 of the story text (content-addressed dedup + stage-2 resume). `names` is filtered to
those actually occurring in the story.

### `kleiner_astronaut_qa_v5_combined.jsonl` — 14,289 rows, the **full** schema

```
hash, request, request_word_1, request_word_2, story,
story_adj_1, story_adj_2, story_topic, story_verb, story_word_1, story_word_2,
story_target_len, story_names, story_start,
question, question_typ, answer, answer_word
```
`question_typ` is one of nine German W-Fragen, balanced: `Wo` 1,689 · `Was` 1,683 · `Womit` 1,657 ·
`Wer` 1,646 · `Wie` 1,599 · `Mit wem` 1,577 · `Weshalb` 1,486 · `Warum` 1,478 · `Wann` 1,474.

### `kleiner_astronaut_qa_v6.jsonl` — 11,420 rows, reduced

```json
{"hash":"…","request":"Schreib ein Abenteuer von Mira in das Raumschiff nur mit Aris",
 "request_word_1":"Mira","request_word_2":"Aris","story":"Unter dem funkelnden Sternenhimmel …"}
```
QA generation was commented out when v6 was produced. Note `story` here is `softClamp`ed to
`STORY_MAX_LEN = 300` (mean 363 chars), far shorter than the raw stories — **pretraining must read
the raw `text`, not the v6 `story`.**

### Conversations — one JSON *array* per line

4-turn (from v5, what we want):
```json
[{"role":"user","content":"<request>"},
 {"role":"assistant","content":"<story>","rl_key1":"<request_word_1>","rl_key2":"<request_word_2>"},
 {"role":"user","content":"<question>"},
 {"role":"assistant","content":"<answer>","rl_key1":"<answer_word>"}]
```
2-turn (from v6, fallback when `question`/`answer` are absent): the first two messages only.

`rl_key1`/`rl_key2` are siblings of `role`/`content`, **on assistant turns only**. They are the RL
reward keys: the German nouns/names the generated text must contain. Vert.x `JsonObject` preserves
insertion order, so key order is always `role, content, rl_key1, rl_key2`.

## Regenerating

```bash
# stages 3a/3b only (no LLM needed)
cd dataset-processor
mvn -q compile exec:java -Dexec.mainClass=de.jotschi.ai.converter.stage3.Dataset2Chat
mvn -q compile exec:java -Dexec.mainClass=de.jotschi.ai.converter.stage3.SplitDataset

# stage 4: materialize the pretraining parquet shards
python -m nanochat.kleiner_astronaut
```

Stages 1–2 additionally need `cluster.url` in `dataset-processor/config/settings.properties` pointing
at an OpenAI-compatible server hosting `mistralai/Mistral-Small-24B-Instruct-2501`.

## Reproducibility

Stage 1/2 output is **not** reproducible: unseeded RNG, temperature 1.0, concurrent appends, and the
server-side sampling seed is not forwarded. That is acceptable because the generated corpus is
checked in as data. Stages 3 and 4 **are** reproducible — seeded split, deterministic conversion.
