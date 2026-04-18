#!/bin/bash
# ============================================================
# Parameter Golf — PRODUCTION RUN on 8×H100 SXM (RunPod)
# Expected: ~10 minutes, target val_bpb < 1.12
# ============================================================
set -e

cd /workspace/parameter-golf 2>/dev/null || cd ~/parameter-golf

echo "=============================="
echo " Parameter Golf — 8×H100 SXM"
echo " PRODUCTION RUN (10 min)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Verify 8 GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 8 ]; then
    echo "ERROR: Expected 8 GPUs but found $GPU_COUNT"
    exit 1
fi
echo "GPUs detected: $GPU_COUNT"

# ===================== PRODUCTION CONFIG =====================
export SEED=2024
export ITERATIONS=12000
export MAX_WALLCLOCK_SECONDS=600  # 10 min hard limit

# Full batch: 786432 tokens, gas=1 on 8 GPUs
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export VAL_BATCH_SIZE=524288
export EVAL_SEQ_LEN=2048

# Full model architecture
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3

# Input representations
export BIGRAM_VOCAB_SIZE=2048
export BIGRAM_DIM=128
# TrigramHash disabled (saves ~590K params)
export TRIGRAM_VOCAB_SIZE=0

# Attention features
export XSA_LAST_N=4
export ROPE_DIMS=16
export GATED_ATTENTION=0
export VALUE_RESIDUAL=0
export LN_SCALE=1

# Value Embedding
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"

# Depth recurrence DISABLED
export DEPTH_RECUR_LAYERS=""
export DEPTH_RECUR_PASSES=1

# QAT + warmdown
export LATE_QAT_THRESHOLD=0.15
export WARMDOWN_ITERS=3500

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
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMUP_STEPS=10
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3

# torch.compile ON (H100 Hopper architecture)
export TORCH_COMPILE=1

# Legal TTT — set to 1 to enable after base BPB validated
export TTT_ENABLED=0
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

# Logging
export TRAIN_LOG_EVERY=200
export VAL_LOSS_EVERY=2000
export EVAL_STRIDE=16         # Production: stride 16 for best BPB
export EVAL_TEMPERATURE=0.90  # Temperature scaling for LeakyReLU²

echo "Starting PRODUCTION training..."
echo ""

torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "========================================="
echo " PRODUCTION Training Complete!"
echo "========================================="

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

echo ""
echo "To submit:"
echo "  1. Download final_model.int8.ptz and train_gpt.py"
echo "  2. Create PR to openai/parameter-golf"
