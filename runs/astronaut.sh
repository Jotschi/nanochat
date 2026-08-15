#!/bin/bash
#
# Kleiner Astronaut: train a small German model that writes children's astronaut
# stories and answers a question about the story it just told.
#
# Run it inside the sandbox:
#   bash runs/sandbox.sh bash runs/astronaut.sh
#
# Or a single stage:
#   STAGES=data,tok bash runs/sandbox.sh bash runs/astronaut.sh
#
# See spec/TRAINING_STRATEGY.md before changing any hyperparameter. The one rule:
# never set a training horizon the run will not reach, or the LR never decays
# and the model is destroyed rather than trained.

set -o nounset
set -o errexit
set -o pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}
export NANOCHAT_DATA_DIR=${NANOCHAT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_astronaut}
mkdir -p "$NANOCHAT_BASE_DIR"

WANDB_RUN=${WANDB_RUN:-dummy}
STAGES=${STAGES:-data,tok,base,base_eval,sft,chat_eval}

# --- model / training knobs --------------------------------------------------
DEPTH=${DEPTH:-12}                       # n_embd = DEPTH * 64 = 768
VOCAB_SIZE=${VOCAB_SIZE:-8192}           # small corpus -> small vocab
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
# 2^17, not the usual 2^19. The corpus is only ~34M tokens, so a 512k-token batch
# would give just 65 optimizer steps per epoch. A quarter of that gives 258, which
# is a far healthier number of updates. nanochat rescales the LR by sqrt(B/B_ref)
# automatically, so this does not need a matching LR change.
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-131072}
# 4 epochs over the ~34M token story corpus. Repeating data up to ~4 epochs is
# close to as good as fresh data (Muennighoff et al., "Scaling Data-Constrained
# Language Models"); beyond that returns diminish but stay positive to ~16.
# Watch val_bpb: raise this while it still falls, stop when it turns.
EPOCHS=${EPOCHS:-4}
SFT_EPOCHS=${SFT_EPOCHS:-2}              # over the 25.7k conversations
# No --fp8: that needs Hopper, and this is an RTX 4090 (SM 89).
# --window-pattern L: Flash Attention 3 is Hopper-only, so attention falls back
# to SDPA, which warns loudly for any other pattern.
WINDOW_PATTERN=${WINDOW_PATTERN:-L}

VENV=.venv/bin
TORCHRUN="$VENV/torchrun --standalone --nproc_per_node=1"
PY="$VENV/python"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# --- German sample prompts for base_eval ------------------------------------
export NANOCHAT_SAMPLE_PROMPTS="Es war einmal
Im Wurmloch
Die Rakete
Der Name vom Roboter ist
Sein Jetpack war
In der Zeitmaschine
Die Sonne war"
export NANOCHAT_SAMPLE_TOKENS=${NANOCHAT_SAMPLE_TOKENS:-96}

# -----------------------------------------------------------------------------
# Data: conversations from the Java generator, then parquet shards for pretraining
if has_stage data; then
    echo "=== Stage: data ==="
    (cd dataset-processor \
        && mvn -q -o compile exec:java -Dexec.mainClass=de.jotschi.ai.converter.stage3.Dataset2Chat \
             -Dexec.args="--out-dir $NANOCHAT_BASE_DIR")
    $PY -m nanochat.kleiner_astronaut --out-dir "$NANOCHAT_DATA_DIR"
fi

# -----------------------------------------------------------------------------
# Tokenizer
if has_stage tok; then
    echo "=== Stage: tok ==="
    # max-chars above the corpus size so the whole thing is used.
    $PY -m scripts.tok_train --vocab-size="$VOCAB_SIZE" --max-chars=200000000
    $PY -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Pretraining
if has_stage base; then
    echo "=== Stage: base ==="
    # Derive the horizon from the measured corpus rather than from
    # --target-param-data-ratio, whose default of 12 would imply ~30 epochs here.
    NUM_ITERATIONS=$($PY -m nanochat.kleiner_astronaut --stats \
        --total-batch-size "$TOTAL_BATCH_SIZE" --epochs "$EPOCHS" 2>/dev/null \
        | awk -v e="$EPOCHS" '$1 == e && $2 == "epoch(s)" { gsub(/,/, "", $NF); print $NF }')
    if [ -z "${NUM_ITERATIONS:-}" ]; then
        echo "Could not derive num_iterations from the corpus stats" >&2
        exit 1
    fi
    echo "Corpus horizon: $EPOCHS epoch(s) => --num-iterations $NUM_ITERATIONS"

    # --core-metric-every=-1: CORE is an English benchmark suite (ARC, HellaSwag,
    # SQuAD, ...). On a German story model it reports noise and costs minutes.
    $TORCHRUN -m scripts.base_train -- \
        --depth="$DEPTH" \
        --num-iterations="$NUM_ITERATIONS" \
        --max-seq-len="$MAX_SEQ_LEN" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        --window-pattern="$WINDOW_PATTERN" \
        --core-metric-every=-1 \
        --run="$WANDB_RUN"
fi

if has_stage base_eval; then
    echo "=== Stage: base_eval ==="
    # CORE is an English benchmark suite; only bpb and samples mean anything here.
    $TORCHRUN -m scripts.base_eval -- \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --eval=bpb,sample
fi

# -----------------------------------------------------------------------------
# SFT: teach the chat special tokens and the request -> story -> question -> answer shape
if has_stage sft; then
    echo "=== Stage: sft ==="
    $TORCHRUN -m scripts.chat_sft -- \
        --kleiner-astronaut \
        --kleiner-astronaut-epochs="$SFT_EPOCHS" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --run="$WANDB_RUN"
fi

if has_stage chat_eval; then
    echo "=== Stage: chat_eval ==="
    $PY -m scripts.chat_astronaut_eval -i sft
fi

echo
echo "Done. Chat with it:"
echo "  bash runs/sandbox.sh $PY -m scripts.chat_cli -i sft -p 'Schreib ein Abenteuer von Mira im Raumschiff'"
