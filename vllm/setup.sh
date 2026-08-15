#!/bin/bash
#
# Create vllm/.venv with vLLM and the nanochat-aware transformers.
#
#   bash vllm/setup.sh
#
# Deliberately a SEPARATE venv from the repo's .venv: the training environment
# pins torch==2.9.1, and vLLM pins its own torch. Installing vLLM into the
# training venv would silently move torch underneath every training script.

set -o nounset
set -o errexit

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV="$HERE/.venv"

command -v uv >/dev/null || { echo "uv is required (it ships in the gpu-sandbox image)" >&2; exit 1; }

uv venv --python 3.12 "$VENV"

# transformers must be new enough to carry models/nanochat. The reference model
# card pins a branch; on a current release it is upstream, so try the release
# first and fall back to the branch.
VLLM_VERSION=${VLLM_VERSION:-}
if [ -n "$VLLM_VERSION" ]; then
    uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"
else
    uv pip install --python "$VENV/bin/python" vllm
fi

if ! "$VENV/bin/python" -c "from transformers.models.nanochat import NanoChatForCausalLM" 2>/dev/null; then
    echo
    echo "Installed transformers has no models/nanochat; using the porting branch."
    uv pip install --python "$VENV/bin/python" \
        "git+https://github.com/huggingface/transformers.git@nanochat-implementation"
fi

echo
"$VENV/bin/python" - <<'PY'
import torch, transformers, vllm
print("vllm        ", vllm.__version__)
print("transformers", transformers.__version__)
print("torch       ", torch.__version__, "cuda", torch.cuda.is_available())
try:
    from transformers.models.nanochat import NanoChatForCausalLM  # noqa: F401
    print("nanochat    available")
except Exception as e:
    print("nanochat    NOT available:", e)
PY

echo
echo "Next:"
echo "  MODEL=nanochat-students/nanochat-d20 bash vllm/start-vllm.sh"
echo "  bash vllm/test.sh"
