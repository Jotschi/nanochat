#!/bin/bash
#
# Serve a nanochat model with the official vLLM container.
# OpenAI-compatible API on :8000.
#
#   bash vllm/start-vllm.sh                                       # our converted model
#   MODEL=nanochat-students/nanochat-d20 bash vllm/start-vllm.sh   # reference model
#   PORT=8001 GPU=0 bash vllm/start-vllm.sh
#
# Run it directly on the host - this starts its own container, so it does NOT go
# through runs/sandbox.sh. Then, from another shell:
#
#   bash vllm/test.sh
#
# Why the official image rather than pip-installing vLLM: it ships the CUDA
# toolkit. The gpu-sandbox image is a runtime-only base, so flashinfer cannot
# JIT-compile its kernels there and the engine dies with
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist".

set -o nounset
set -o errexit

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")

IMAGE=${IMAGE:-vllm/vllm-openai:latest}
NAME=${NAME:-nanochat-vllm}
PORT=${PORT:-8000}
# The 4090 usually hosts llama.cpp for the data generator; a nanochat model fits
# comfortably on the 3060. See spec/TRAINING_STRATEGY.md.
GPU=${GPU:-1}
MAX_LEN=${MAX_LEN:-2048}
GPU_MEM_FRAC=${GPU_MEM_FRAC:-0.35}

# Docker resolves -v sources on the *real host*, which is not this filesystem
# when the shell itself runs in a container. Resolve the bind mount rather than
# assuming. Same trick as runs/sandbox.sh - getting it wrong silently mounts an
# empty directory.
# ${USER:-$(id -un)}: USER is not exported in every shell (cron, docker exec, some
# non-login shells) and `set -o nounset` turns that into a hard failure.
GUEST_METALOOM=${GUEST_METALOOM:-/home/${USER:-$(id -un)}/workspaces/metaloom}
if [ -z "${HOST_METALOOM:-}" ]; then
    HOST_METALOOM=$(awk -v t="$GUEST_METALOOM" '$5 == t { print $4; exit }' /proc/self/mountinfo || true)
    HOST_METALOOM=${HOST_METALOOM:-$GUEST_METALOOM}
fi

NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$GUEST_METALOOM/nanochat-cache}
MODEL=${MODEL:-$NANOCHAT_BASE_DIR/hf_export/d12hf-sft}

# Keep downloaded hub models across container restarts.
HF_CACHE_GUEST=$GUEST_METALOOM/.hf-cache
HF_CACHE_HOST=$HOST_METALOOM/.hf-cache
mkdir -p "$HF_CACHE_GUEST"

# Validate a local model path before handing it to vLLM. Without this, a path
# that does not exist is passed through and transformers tries to read it as a
# Hub repo id, reporting
#   OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
# which says nothing about the actual problem.
case "$MODEL" in
    /*|./*|../*)
        if [ ! -d "$MODEL" ]; then
            echo "No such model directory: $MODEL" >&2
            echo >&2
            echo "Nothing has been exported yet. Check whether this checkpoint can be:" >&2
            echo "  bash runs/sandbox.sh .venv/bin/python vllm/convert_to_hf.py --check" >&2
            echo >&2
            echo "Expect it to report unsupported tensors: transformers' NanoChat targets" >&2
            echo "an older architecture than this repo trains, so convert_to_hf.py refuses" >&2
            echo "rather than writing a model that loads but generates nonsense." >&2
            echo "See vllm/README.md. To check that serving itself works, use the" >&2
            echo "reference model instead:" >&2
            echo "  MODEL=nanochat-students/nanochat-d20 bash vllm/start-vllm.sh" >&2
            exit 1
        fi
        if [ ! -f "$MODEL/config.json" ]; then
            echo "$MODEL has no config.json - it is not a HuggingFace model directory." >&2
            echo "Re-export it with: bash runs/sandbox.sh .venv/bin/python vllm/convert_to_hf.py" >&2
            exit 1
        fi
        ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "Image : $IMAGE"
echo "Model : $MODEL"
echo "GPU   : $GPU"
echo "URL   : http://127.0.0.1:$PORT/v1"
echo

# --gpus '"device=N"' rather than CUDA_VISIBLE_DEVICES: with a 4090 and a 3060 in
#   one box the default CUDA ordering is fastest-first, so an index can select a
#   different card than nvidia-smi shows. Selecting the device at the docker
#   layer avoids the ambiguity entirely.
# --ipc=host: vLLM needs shared memory for its worker processes.
# --model-impl transformers: nanochat has no native vLLM kernel; it runs through
#   the Transformers backend.
# --enforce-eager: what the reference model card recommends. Skips CUDA graph
#   capture, a slow memory-hungry step that buys little on a model this small.
# --chat-template-content-format string: the OpenAI server otherwise rewrites
#   message content into a list of parts, but nanochat's chat template indexes
#   content as a plain string and every request 400s with
#   "User messages must contain string content".
# -it only when there is a terminal, so this also works from scripts, nohup and CI
# ("the input device is not a TTY" otherwise).
TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAGS=(-it)
fi

exec docker run --rm "${TTY_FLAGS[@]}" \
    --name "$NAME" \
    --gpus "\"device=$GPU\"" \
    --ipc=host \
    -p "$PORT:8000" \
    -v "$HOST_METALOOM:$GUEST_METALOOM" \
    -v "$HF_CACHE_HOST:/root/.cache/huggingface" \
    "$IMAGE" \
    --model "$MODEL" \
    --served-model-name kleiner-astronaut \
    --model-impl transformers \
    --enforce-eager \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_FRAC" \
    --chat-template-content-format string \
    "$@"
