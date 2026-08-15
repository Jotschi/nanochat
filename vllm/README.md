# Serving the Kleiner Astronaut model with vLLM

```
start-vllm.sh     run the official vLLM container, OpenAI API on :8000
test.sh           two chat queries against the running server
convert_to_hf.py  nanochat checkpoint -> HuggingFace format
```

```bash
# one shell
bash vllm/start-vllm.sh
# another shell
bash vllm/test.sh
```

Verified working end to end with our own model (d12, `--hf-compatible`, SFT):

```
USER  : Schreib ein Abenteuer von Mira in das Raumschiff nur mit Aris
ASSIST: In einem weit entfernten Teil des Weltraums, wo die Sterne wie funkelnde
        Diamanten leuchten, lebte ein kleiner Astronaut namens Mira. ... Es war
        ein Quasar, ein riesiger, leuchtender Stern ...
USER  : Was hat Mira entdeckt?
ASSIST: Mira wollte herausfinden, was das für ein Quasar war.
```

To check the serving path alone, without any conversion:

```bash
MODEL=nanochat-students/nanochat-d20 bash vllm/start-vllm.sh
```

Nothing to install: `start-vllm.sh` runs `vllm/vllm-openai:latest` directly and
pulls it on first use. It starts its own container, so it does **not** go through
`runs/sandbox.sh`.

The official image is used rather than pip-installing vLLM because it ships the
CUDA toolkit. The gpu-sandbox image is a runtime-only PyTorch base, so flashinfer
cannot JIT-compile its kernels there and the engine dies with:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

## Flags that are not obvious

| flag | why |
| --- | --- |
| `--model-impl transformers` | nanochat has no native vLLM kernel; it runs through transformers' `NanoChatForCausalLM` |
| `--chat-template-content-format string` | the OpenAI server otherwise rewrites message content into a list of parts, but nanochat's chat template indexes `content` as a plain string — every request 400s with `User messages must contain string content` |
| `--enforce-eager` | recommended by the reference model card; skips CUDA graph capture, which is slow and memory-hungry for no gain at this size |
| `--gpus '"device=1"'` | with a 4090 and a 3060 in one box the default CUDA ordering is fastest-first, so an index can select a different card than `nvidia-smi` shows. Selecting at the docker layer removes the ambiguity |
| `--ipc=host` | vLLM needs shared memory between its worker processes |

`test.sh` deliberately does **not** use `curl -f`: with `-f` an HTTP 400 exits
non-zero and prints nothing, so a chat-template error looks exactly like the model
returning an empty completion. It reads the body and prints the server's message.

## Serving our own model: train it with `--hf-compatible`

`vllm serve` works for nanochat because **transformers** ships a
`NanoChatForCausalLM` ([PR #41634](https://github.com/huggingface/transformers/pull/41634),
merged 27 Nov 2025, with explicit vLLM work in its commits).

**That implementation targets an older nanochat architecture than this repo
trains by default.** A default checkpoint carries 17 tensors it has never heard
of, and `convert_to_hf.py` refuses to write a model that drops them:

| default checkpoint has | `NanoChatConfig` / `NanoChatModel` |
| --- | --- |
| `value_embeds.{0,2,4,...}.weight` (6 tensors) | absent |
| `transformer.h.N.attn.ve_gate.weight` (6 tensors) | absent |
| `smear_gate.weight`, `smear_lambda` | absent |
| `resid_lambdas`, `x0_lambdas`, `backout_lambda` | absent |

The fix is to train with `--hf-compatible`, which does not construct those
components at all (see `GPTConfig.hf_compatible`). The resulting state dict is
exactly `n_layer * 6 + 2` tensors, a 1:1 match for what transformers expects:

```bash
# base
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 --hf-compatible --window-pattern=L --model-tag=d12hf \
    --num-iterations=600 --total-batch-size=131072 --final-lr-frac=0.0
# sft
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --kleiner-astronaut --model-tag=d12hf --model-step=600 --final-lr-frac=0.0
# export + serve
python vllm/convert_to_hf.py --model-tag d12hf --source sft
bash vllm/start-vllm.sh
```

On this corpus `--hf-compatible` is not a downgrade — it reached a *lower*
validation bpb than the full architecture (1.3209 vs 1.3300). See
spec/FINDINGS.md §2f.

### Norms are parameter-free — do not synthesise them

transformers' NanoChat keeps nanochat's parameter-free RMSNorm. A converted model
is six weights per layer plus `embed_tokens` and `lm_head`, and **nothing else** —
`nanochat-students/nanochat-d20` has exactly 20*6+2 = 122 tensors. Helpfully
adding all-ones `input_layernorm` / `q_norm` weights (a reasonable-sounding guess,
and what an earlier version of this script did) makes vLLM reject the load:

```
ValueError: There is no module or parameter named 'model.layers.0.input_layernorm.weight'
in TransformersForCausalLM
```

## Tokenizer

`convert_to_hf.py` also exports the tokenizer. nanochat stores a pickled
tiktoken/rustbpe vocabulary, which has no merge list — tiktoken keeps only
token→rank — so the merges are recovered by re-running BPE on each token
restricted to lower-rank merges. Byte tokens are mapped through GPT-2's
byte↔unicode table for the `ByteLevel` representation, and the pre-tokenizer is
the same two stages tiktoken uses: nanochat's split regex, then byte-level.

A tokenizer that silently disagrees with the one the model trained on presents as
"the model is just bad", so the converter **round-trips sample strings through
both tokenizers and fails if the ids differ** — including German umlauts and
whitespace runs, which is where a lossy byte→str conversion breaks first.

A chat template matching `render_conversation` is written to
`tokenizer_config.json`, so `apply_chat_template` reproduces the
`<|user_start|>` / `<|assistant_start|>` framing the model was trained on.
