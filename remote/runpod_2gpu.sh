#!/bin/bash
# ============================================================
# Parameter Golf — Test Run on 2×RTX 4090 (48GB total)
# Expected: ~15-20 minutes, 2x faster per step than 1×GPU
# ============================================================
set -e

cd /workspace/parameter-golf 2>/dev/null || cd ~/parameter-golf

echo "=============================="
echo " Parameter Golf — 2×RTX 4090"
echo " Test Run (~15-20 min)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Verify 2 GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Expected 2 GPUs but found $GPU_COUNT. Falling back to 1GPU mode."
    exec bash runpod_1gpu.sh
fi
echo "GPUs detected: $GPU_COUNT"

# Training config for 2x RTX 4090
export SEED=42
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=0  # non-avoid early_stop

# 2× batch: gas=4 on 2GPU → micro-batch = 393216/(2×4) = 49152 tokens/GPU → ~16GB/GPU
export TRAIN_BATCH_TOKENS=393216
export TRAIN_SEQ_LEN=2048
export VAL_BATCH_SIZE=1048576
export EVAL_SEQ_LEN=2048

# Full model architecture
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3

# Input representations
export BIGRAM_VOCAB_SIZE=4096
export BIGRAM_DIM=128
export TRIGRAM_VOCAB_SIZE=4096
export TRIGRAM_DIM=128

# Attention features
export XSA_LAST_N=4
export ROPE_DIMS=16
export GATED_ATTENTION=1
export VALUE_RESIDUAL=1
export LN_SCALE=1

# Value Embedding
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"

# Depth recurrence
export DEPTH_RECUR_LAYERS="4,5"
export DEPTH_RECUR_PASSES=2

# QAT + warmdown
export LATE_QAT_THRESHOLD=0.15
export WARMDOWN_ITERS=1500

# Weight averaging
export SWA_ENABLED=1
export SWA_EVERY=50

# LR Warmup
export LR_WARMUP_STEPS=50

# EMA late start
export EMA_START_FRAC=0.4

# Optimizer
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=500
export WARMUP_STEPS=10
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3

# torch.compile ON
export TORCH_COMPILE=1

# Disable TTT for speed
export TTT_ENABLED=0

# Logging
export TRAIN_LOG_EVERY=100
export VAL_LOSS_EVERY=1000
export EVAL_STRIDE=128

echo "Starting training (2×GPU)..."
echo ""

torchrun --standalone --nproc_per_node=2 train_gpt.py

echo ""
echo "=============================="
echo " Training Complete!"
echo "=============================="

# Check artifact size
if [ -f "final_model.int8.ptz" ]; then
    SIZE=$(stat -c%s "final_model.int8.ptz" 2>/dev/null || stat -f%z "final_model.int8.ptz")
    CODE=$(wc -c < train_gpt.py)
    TOTAL=$((SIZE + CODE))
    echo "Model:  ${SIZE} bytes"
    echo "Code:   ${CODE} bytes"
    echo "Total:  ${TOTAL} bytes / 16,000,000 limit"
    if [ $TOTAL -le 16000000 ]; then
        echo "✅ WITHIN 16MB LIMIT"
    else
        echo "❌ OVER 16MB LIMIT!"
    fi
fi
