#!/bin/bash
# ============================================================
# Parameter Golf — Test Run on 1×RTX 4090 (RunPod / VastAI)
# Expected: ~15-20 minutes, validates code + serialization
# ============================================================
set -e

cd /workspace/parameter-golf 2>/dev/null || cd ~/parameter-golf

echo "=============================="
echo " Parameter Golf — 1×RTX 4090"
echo " Test Run (~15-20 min)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Training config for 1x RTX 4090
export SEED=42
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=0  # non-avoid early_stop

# Effective batch 2x larger than minimal, fits 24GB VRAM
# gas=8 on 1GPU → micro-batch = 196608/8 = 24576 tokens → ~12GB VRAM
export TRAIN_BATCH_TOKENS=196608
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
# TrigramHash disabled (not used by any top submission, saves ~590K params)
export TRIGRAM_VOCAB_SIZE=0

# Attention features
export XSA_LAST_N=4
export ROPE_DIMS=16
export GATED_ATTENTION=0  # Not used by SOTA
export VALUE_RESIDUAL=0   # Not used by SOTA
export LN_SCALE=1

# Value Embedding
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"

# Depth recurrence DISABLED (proven ineffective by 3 researchers, 250+ experiments)
export DEPTH_RECUR_LAYERS=""
export DEPTH_RECUR_PASSES=1

# QAT + warmdown (adjusted for longer 1GPU run)
export LATE_QAT_THRESHOLD=0.15
export WARMDOWN_ITERS=1500

# Weight averaging
export SWA_ENABLED=1
export SWA_EVERY=50

# LR Warmup (NEW: fixes loss spike)
export LR_WARMUP_STEPS=50

# EMA late start (NEW: fixes EMA pollution)
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

# torch.compile ON (4090 has enough SMs)
export TORCH_COMPILE=1

# Disable TTT for speed (test arch only)
export TTT_ENABLED=0

# Logging
export TRAIN_LOG_EVERY=100
export VAL_LOSS_EVERY=1000
export EVAL_STRIDE=64        # 64 for test, 16 for production
export EVAL_TEMPERATURE=0.90  # Temperature scaling for LeakyReLU² (free -0.005 BPB)

echo "Starting training..."
echo ""

torchrun --standalone --nproc_per_node=1 train_gpt.py

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
