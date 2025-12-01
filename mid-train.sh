#!/bin/bash

source .venv/bin/activate

WANDB_RUN=d15-mid

torchrun --standalone --nproc_per_node=1 -m scripts.mid_train -- --run=$WANDB_RUN