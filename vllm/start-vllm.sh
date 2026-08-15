#!/bin/bash
#
# Serve a nanochat model with vLLM, OpenAI-compatible API on :8000.
#
#   bash vllm/start-vllm.sh
#   MODEL=nanochat-students/nanochat-d20 bash vllm/start-vllm.sh
#   PORT=8001 GPU=1 bash vllm/start-vllm.sh
#
# Run it inside the sandbox if you prefer the container toolchain:
#   bash runs/sandbox.sh bash vllm/start-vllm.sh
#
# Requires vllm/setup.sh to have been run once.

set -o nounset
set -o errexit

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")

# Default to our converted checkpoint; fall back to the reference model, which is
# useful for checking that the serving path itself works. See README.md for why
# our own checkpoint may not convert yet.
NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}
MODEL=${MODEL:-$NANOCHAT_BASE_DIR/hf_export/d12-sft}

PORT=${PORT:-8000}
# 0.0.0.0 rather than 127.0.0.1: inside the sandbox container, a loopback bind is
# not reachable through docker's published port. runs/sandbox.sh publishes 8000.
HOST=${HOST:-0.0.0.0}
# The 4090 is often busy hosting llama.cpp for the data generator; the 3060 has
# plenty of room for a 0.1B model. See spec/TRAINING_STRATEGY.md.
GPU=${GPU:-1}
MAX_LEN=${MAX_LEN:-2048}
# nanochat models are tiny, so there is no point reserving most of the card.
GPU_MEM_FRAC=${GPU_MEM_FRAC:-0.35}

VENV="$HERE/.venv"
if [ ! -x "$VENV/bin/vllm" ]; then
    echo "vLLM not installed. Run: bash vllm/setup.sh" >&2
    exit 1
fi

if [ -d "$MODEL" ] && [ ! -f "$MODEL/config.json" ]; then
    echo "$MODEL exists but has no config.json - convert the checkpoint first:" >&2
    echo "  bash runs/sandbox.sh .venv/bin/python vllm/convert_to_hf.py" >&2
    exit 1
fi

echo "Model : $MODEL"
echo "GPU   : $GPU"
echo "URL   : http://$HOST:$PORT/v1"
echo

# --model-impl transformers: nanochat has no native vLLM kernel; it runs through
#   the Transformers backend.
# --enforce-eager: what the reference model card recommends. Skips CUDA graph
#   capture, which is a slow, memory-hungry step that buys little on a model
#   this small.
# CUDA_DEVICE_ORDER: this box has an RTX 4090 and an RTX 3060, and vLLM warns
# that with mixed devices the default (fastest-first) ordering can make
# CUDA_VISIBLE_DEVICES=1 select a different card than nvidia-smi shows.
#
# VLLM_USE_FLASHINFER_SAMPLER=0: flashinfer JIT-compiles its kernels and needs
# nvcc, but the gpu-sandbox image is the runtime flavour with no CUDA toolkit -
# engine start dies with "Could not find nvcc and default cuda_home
# '/usr/local/cuda' doesn't exist". Attention itself is fine (FLASH_ATTN); only
# the sampler pulls flashinfer in. Rebuild the sandbox from a -devel base if you
# want flashinfer.
exec env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$GPU" \
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
    "$VENV/bin/vllm" serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --model-impl transformers \
    --enforce-eager \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_FRAC" \
    --served-model-name kleiner-astronaut \
    --chat-template-content-format string \
    "$@"
# --chat-template-content-format string: by default the OpenAI server rewrites
# message content into a list of parts ([{"type":"text",...}]), but nanochat's
# chat template indexes content as a plain string and raises
# "User messages must contain string content", surfacing as a 400 on every
# chat request.
