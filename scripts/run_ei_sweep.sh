#!/usr/bin/env bash
# Expert Iteration sweep over rollouts G, SFT epochs, and batch size Db.
set -euo pipefail

DEVICE=cuda
N_EI_STEPS=5
LR=2e-5

# ---- Vary rollouts G (Db=1024, sft_epochs=1) ----
for G in 4 8; do
    echo "=== EI G=${G} db=1024 sft_epochs=1 ==="
    python scripts/run_expert_iteration.py \
        --rollouts $G \
        --db 1024 \
        --sft-epochs 1 \
        --n-ei-steps $N_EI_STEPS \
        --lr $LR \
        --device $DEVICE \
        --output-dir "outputs/ei_G${G}_db1024_ep1"
done

# ---- Vary SFT epochs (G=4, Db=1024) ----
for EP in 1 3; do
    echo "=== EI G=4 db=1024 sft_epochs=${EP} ==="
    python scripts/run_expert_iteration.py \
        --rollouts 4 \
        --db 1024 \
        --sft-epochs $EP \
        --n-ei-steps $N_EI_STEPS \
        --lr $LR \
        --device $DEVICE \
        --output-dir "outputs/ei_G4_db1024_ep${EP}"
done

# ---- Vary Db (G=4, sft_epochs=1) ----
for DB in 512 1024 2048; do
    echo "=== EI G=4 db=${DB} sft_epochs=1 ==="
    python scripts/run_expert_iteration.py \
        --rollouts 4 \
        --db $DB \
        --sft-epochs 1 \
        --n-ei-steps $N_EI_STEPS \
        --lr $LR \
        --device $DEVICE \
        --output-dir "outputs/ei_G4_db${DB}_ep1"
done
