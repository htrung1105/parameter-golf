#!/bin/bash
# ============================================================
# Parameter Golf — Setup for any remote GPU container
# Run this ONCE after SSH into the container
# ============================================================
set -e

echo "=============================="
echo " Parameter Golf Remote Setup"
echo "=============================="

# 1. Install system dependencies (gcc needed for torch.compile/Triton)
echo ""
echo "[1/5] Installing system dependencies..."
apt-get update -qq

if ! command -v gcc &> /dev/null; then
    apt-get install -y -qq gcc > /dev/null 2>&1
    echo "gcc installed"
else
    echo "gcc already available"
fi

if ! command -v git &> /dev/null; then
    apt-get install -y -qq git > /dev/null 2>&1
    echo "git installed"
else
    echo "git already available"
fi

if ! command -v tmux &> /dev/null; then
    apt-get install -y -qq tmux > /dev/null 2>&1
    echo "tmux installed (use to prevent SSH disconnect kills)"
else
    echo "tmux already available"
fi

# 2. System info
echo ""
echo "[2/5] System Info"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null || \
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch check skipped"

# 3. Install Python dependencies
echo ""
echo "[3/5] Installing Python dependencies..."
pip install --quiet sentencepiece numpy huggingface-hub datasets tqdm 2>/dev/null || \
pip3 install --quiet sentencepiece numpy huggingface-hub datasets tqdm 2>/dev/null
echo "Python dependencies installed"

# 4. Clone repo (if not already cloned)
echo ""
echo "[4/5] Setting up repository..."
cd /workspace 2>/dev/null || cd ~
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf
echo "Repository cloned"

# 5. Download dataset (1 shard for test, 80 for production)
SHARDS="${1:-1}"
echo ""
echo "[5/5] Downloading dataset ($SHARDS shards)..."
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    python data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$SHARDS" 2>/dev/null || \
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$SHARDS"
else
    echo "Dataset already exists, skipping download"
fi

# 6. Done
echo ""
echo "[Final] Setup complete!"
echo ""
echo "Next: Upload train_gpt.py and run scripts"
echo "  scp -P PORT train_gpt.py root@HOST:/workspace/parameter-golf/"
echo "  scp -P PORT runpod_1gpu.sh root@HOST:/workspace/parameter-golf/"
echo "  scp -P PORT runpod_2gpu.sh root@HOST:/workspace/parameter-golf/"
echo "  scp -P PORT runpod_8gpu.sh root@HOST:/workspace/parameter-golf/"
echo ""
echo "Usage:"
echo "  cd parameter-golf"
echo ""
echo "  # -- Test train 1xGPU --"
echo "  chmod +x runpod_1gpu.sh"
echo "  bash runpod_1gpu.sh"
echo ""
echo "  # -- Test train 2xGPU --"
echo "  chmod +x runpod_2gpu.sh"
echo "  bash runpod_2gpu.sh"
echo ""
echo "  # -- Full train --"
echo "  chmod +x runpod_8gpu.sh"
echo "  bash runpod_8gpu.sh"