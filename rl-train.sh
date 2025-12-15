#!/bin/bash

source .venv/bin/activate

WANDB_RUN=d15-sft

torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl -- --run=$WANDB_RUN