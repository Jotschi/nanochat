# Serving the Kleiner Astronaut model with vLLM

```
setup.sh          create vllm/.venv with vLLM (separate from the training venv)
convert_to_hf.py  nanochat checkpoint -> HuggingFace format
start-vllm.sh     vllm serve <model>
test.sh           two chat queries against the running server
```

Quick smoke test with a known-good public model (no conversion needed):

```bash
bash vllm/setup.sh
bash runs/sandbox.sh bash -c 'MODEL=nanochat-students/nanochat-d20 GPU=1 bash vllm/start-vllm.sh'
bash vllm/test.sh    # from another shell, on the host
```

This path is **verified working** on this box: vLLM 0.27.1 + transformers 5.15.0
serving `nanochat-d20` on the 3060, `test.sh` returning both completions. The
answers are English nonsense because d20 is a general English model, not our
German story model — the point of the smoke test is the serving path, not the
content.

## Read this before converting our own checkpoint

`vllm serve` works for nanochat models because **transformers** ships a
`NanoChatForCausalLM`, and vLLM can run it through its Transformers backend
(`--model-impl transformers`). The reference model
[nanochat-students/nanochat-d20](https://huggingface.co/nanochat-students/nanochat-d20)
is served exactly that way.

**That implementation targets an older nanochat architecture than the one this
repo trains.** Checked against `transformers/models/nanochat` on `main`:

| our checkpoint has | `NanoChatConfig` / `NanoChatModel` |
| --- | --- |
| `value_embeds.{0,2,4,...}.weight` (6 tensors) | absent |
| `transformer.h.N.attn.ve_gate.weight` (6 tensors) | absent |
| `smear_gate.weight`, `smear_lambda` | absent |
| `resid_lambdas`, `x0_lambdas`, `backout_lambda` | absent |
| `window_pattern` (sliding-window attention) | absent |
| parameter-free `F.rms_norm` | learnable `q_norm` / `k_norm` / layernorms |

So 16 tensors and 4 scalars in our d12 checkpoint have no destination. Dropping
them does not fail — it produces a model that **loads and generates nonsense**,
which is worse than an error. `convert_to_hf.py` therefore refuses by default and
lists what it could not map.

The parameter-free norms are the one benign difference: they are equivalent to a
learnable RMSNorm with an all-ones weight, and the converter emits those.

### Three ways forward

1. **Wait for transformers to catch up.** The upstream nanochat architecture
   moved (value embeddings, smear gate, residual lambdas); the port will follow.
   Nothing to do here but re-run the converter later.
2. **Write custom modeling code** and ship it in the exported repo via
   `auto_map`, then serve with `--trust-remote-code`. Note that vLLM's
   Transformers backend has requirements around attention plumbing, and
   per-layer value embeddings interact with the paged KV cache — this is a
   project, not an afternoon.
3. **Train an architecture-compatible model.** `--window-pattern L` already
   disables sliding-window attention, but value embeddings, the smear gate and
   the lambdas are unconditional in `nanochat/gpt.py` and would need config
   switches to turn off. Costs a retrain (~15 min at d12) and some accuracy.

Until one of those lands, the working way to talk to our model is nanochat's own
engine:

```bash
bash runs/sandbox.sh .venv/bin/python -m scripts.chat_cli -i sft \
  -p "Schreib ein Abenteuer von Mira im Raumschiff"
```

## Sandbox gotchas

The gpu-sandbox image is the **runtime** flavour of the PyTorch base image: it has
no CUDA toolkit, so `nvcc` and `/usr/local/cuda` are missing. flashinfer
JIT-compiles its kernels and dies on start with

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

Attention is unaffected (vLLM picks FLASH_ATTN); only the sampler reaches for
flashinfer, so `start-vllm.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0`. Rebuild the
sandbox from a `-devel` base if you want flashinfer.

It also sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, because with an RTX 4090 and an RTX
3060 in one box vLLM warns that the default fastest-first ordering can make
`CUDA_VISIBLE_DEVICES=1` select a different card than `nvidia-smi` reports.

Serving from inside the sandbox needs two more things, both already handled:
`runs/sandbox.sh` publishes port 8000 (override with `SANDBOX_PUBLISH="8001:8001"`),
and `start-vllm.sh` binds `0.0.0.0` rather than `127.0.0.1` — a loopback bind
inside the container is not reachable through a published port.

One more, and it is the least obvious: vLLM's OpenAI server rewrites message
content into a list of parts (`[{"type": "text", ...}]`) by default, but
nanochat's chat template indexes `content` as a plain string and raises
`User messages must contain string content`. Every chat request 400s.
`start-vllm.sh` passes `--chat-template-content-format string`.

That failure is easy to misread, which is why `test.sh` deliberately does *not*
use `curl -f`: with `-f` a 400 exits non-zero and prints nothing, so a template
error is indistinguishable from the model returning an empty completion. It reads
the body and prints the server's own message instead.

## Tokenizer

The converter also needs to emit an HF tokenizer. nanochat stores a pickled
rustbpe/tiktoken vocabulary (`$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`), which
`convert_to_hf.py` rebuilds into `tokenizer.json` using the `tokenizers` library,
along with a chat template for the `<|user_start|>` / `<|assistant_start|>`
special tokens. That part is architecture-independent and works today.
