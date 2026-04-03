#!/bin/bash
# ============================================================
# Parameter Golf — RunPod Setup Script
# Image: runpod/parameter-golf:latest (PyTorch 2.9.1, CUDA 12.8)
# ============================================================
set -e

echo "======================================"
echo " Parameter Golf — RunPod Setup"
echo "======================================"

cd /workspace

# Step 1: Clone official repo (for data download script)
echo "[1/3] Cloning repository..."
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Step 2: Download dataset
echo "[2/3] Downloading FineWeb dataset (sp1024)..."
if [ -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo "  Dataset already downloaded, skipping."
else
    python3 data/cached_challenge_fineweb.py --variant sp1024
    echo "  ✅ Dataset downloaded"
fi

# Step 3: Verify environment
echo "[3/3] Verifying environment..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'    GPU {i}: {name} ({mem:.0f}GB)')
try:
    from flash_attn_interface import flash_attn_func
    print('  FlashAttention 3: ✅')
except ImportError:
    print('  FlashAttention 3: ❌ (will use SDPA fallback)')
"

echo ""
echo "======================================"
echo " Setup Complete!"
echo "======================================"
echo ""
echo "Next: Start training with one of:"
echo "  1×h100:  bash runpod_h100.sh"
echo ""
echo "⚠️  Always run inside tmux to prevent SSH disconnect killing training!"
echo "  tmux new -s train"
echo ""
