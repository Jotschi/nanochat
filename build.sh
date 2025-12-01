#!/bin/bash

set -o nounset
set -o errexit

VERSION=$(cat .version)
IMAGE_NAME=nanochat-train

podman build -t $IMAGE_NAME:$VERSION .

mkdir -p .cache
mkdir -p .rustup

podman run --rm -it \
    --device nvidia.com/gpu=all \
    --name nanochat-train \
    --shm-size 1g \
	-v .:/opt/build \
        -v .cargo:/root/.cargo \
	-v .cache:/root/.cache \
	-v .rustup:/root/.rustup \
    $IMAGE_NAME:$VERSION  bash
