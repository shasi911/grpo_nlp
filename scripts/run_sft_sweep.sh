#!/usr/bin/env bash
# Run SFT experiments for all dataset sizes, then the filtered variant.
# Adjust --lr / --batch-size if you need to tune further.
set -euo pipefail

LR=2e-5
BS=16
EPOCHS=3
DEVICE=cuda

# Part 1: vary dataset size
for N in 128 256 512 1024 full; do
    echo "=== SFT num-examples=$N ==="
    python scripts/run_sft.py \
        --num-examples $N \
        --lr $LR \
        --batch-size $BS \
        --num-epochs $EPOCHS \
        --device $DEVICE \
        --use-sgd \
        --output-dir "outputs/sft_n${N}"
done

# Part 2: filtered dataset (correct answers only, full dataset)
echo "=== SFT filtered ==="
python scripts/run_sft.py \
    --filtered \
    --num-examples full \
    --lr $LR \
    --batch-size $BS \
    --num-epochs $EPOCHS \
    --device $DEVICE \
    --use-sgd \
    --output-dir "outputs/sft_filtered"
