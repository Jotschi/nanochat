#!/bin/bash
#
# Run a command inside the gpu-sandbox container.
#
#   bash runs/sandbox.sh python -m nanochat.kleiner_astronaut --stats
#   bash runs/sandbox.sh bash runs/astronaut.sh
#   bash runs/sandbox.sh            # interactive shell
#
# Why the flags are what they are (see spec/FINDINGS.md):
#
#   HOST_METALOOM   Docker resolves -v source paths on the *real host*, which is
#                   not the filesystem this shell sees when the shell itself runs
#                   in a container. Here /home/defaultuser/workspaces/metaloom is
#                   a bind of /home/jotschi/workspaces/metaloom, and the real
#                   host also has an unrelated /home/defaultuser/workspaces. Get
#                   this wrong and the container silently mounts an empty
#                   directory. Auto-detected from /proc/self/mountinfo.
#                   It is mounted back at the same path this shell uses, so
#                   absolute paths mean the same thing on both sides.
#
#   --gpus all      The image README and copilot/gpu-sandbox/run.sh use CDI
#                   (--device nvidia.com/gpu=all), but this host has no /etc/cdi
#                   and no nvidia-ctk. Docker does have the nvidia runtime
#                   registered, so --gpus all is the working form. run.sh also
#                   defaults to podman, which is not installed.
#
#   SANDBOX_USER=root
#                   The bind mount appears as uid 0 inside the container, so the
#                   default 'defaultuser' cannot write to it. The entrypoint
#                   honours SANDBOX_USER.
#
#   CUDA_VISIBLE_DEVICES=0
#                   Train on the 4090 only. Pairing it with the 12 GB 3060 under
#                   DDP would halve throughput and cap the device batch size.

set -o nounset
set -o errexit

IMAGE=${IMAGE:-swdev-docker.docker.apa-it.at/infomgmt/gpu-sandbox:latest}

# Path as this shell sees it.
GUEST_METALOOM=${GUEST_METALOOM:-/home/defaultuser/workspaces/metaloom}

# Same directory as the docker daemon sees it. Resolve the bind mount rather
# than assuming, so this keeps working outside the copilot pod too.
if [ -z "${HOST_METALOOM:-}" ]; then
    HOST_METALOOM=$(awk -v target="$GUEST_METALOOM" '$5 == target { print $4; exit }' /proc/self/mountinfo || true)
    HOST_METALOOM=${HOST_METALOOM:-$GUEST_METALOOM}
fi

REPO=${REPO:-$GUEST_METALOOM/nanochat}
CACHE=${NANOCHAT_BASE_DIR:-$GUEST_METALOOM/nanochat-cache}
UV_CACHE=$GUEST_METALOOM/.uv-cache
# uv honours .python-version (3.10) and downloads a managed interpreter for it.
# That must live on the mounted volume: the default is under $HOME, which is
# inside the --rm container, so .venv/bin/python would dangle after every run.
UV_PYTHON_DIR=$GUEST_METALOOM/.uv-python
M2=$GUEST_METALOOM/.m2

mkdir -p "$CACHE" "$UV_CACHE" "$UV_PYTHON_DIR" "$M2"

if [ "$#" -eq 0 ]; then
    set -- bash
fi

# -it only when there is a terminal, so this works from scripts and CI too.
TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAGS=(-it)
fi

# Publish ports so a server started inside the container is reachable from the
# host, e.g. `bash runs/sandbox.sh bash vllm/start-vllm.sh` then `bash vllm/test.sh`
# from another shell. Space-separated "host:container" pairs.
PUBLISH_FLAGS=()
for mapping in ${SANDBOX_PUBLISH:-8000:8000}; do
    PUBLISH_FLAGS+=(-p "$mapping")
done

exec docker run --rm "${TTY_FLAGS[@]}" \
    --gpus all \
    --shm-size 32g \
    "${PUBLISH_FLAGS[@]}" \
    -e SANDBOX_USER=root \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    -e NANOCHAT_BASE_DIR="$CACHE" \
    -e NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$CACHE/base_data_astronaut}" \
    -e UV_CACHE_DIR="$UV_CACHE" \
    -e UV_PYTHON_INSTALL_DIR="$UV_PYTHON_DIR" \
    -e MAVEN_OPTS="-Dmaven.repo.local=$M2" \
    -e WANDB_RUN="${WANDB_RUN:-dummy}" \
    -v "$HOST_METALOOM:$GUEST_METALOOM" \
    -w "$REPO" \
    "$IMAGE" "$@"
