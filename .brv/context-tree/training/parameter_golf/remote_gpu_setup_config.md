---
title: Remote GPU Setup Config
tags: []
related: [training/parameter_golf/project_snapshot.md]
keywords: []
importance: 50
recency: 1
maturity: draft
createdAt: '2026-04-01T07:48:38.458Z'
updatedAt: '2026-04-01T07:48:38.458Z'
---
## Raw Concept
**Task:**
Document remote GPU environment and recommended training configuration values for parameter-golf experiments

**Changes:**
- Recorded chiasegpu.vn remote GPU environment details
- Captured TRAIN_BATCH_TOKENS values for 1 GPU, 2 GPU, and 8 GPU setups
- Captured test versus production values for wallclock, warmdown, and TTT flags
- Noted single-4090 OOM constraint at TRAIN_BATCH_TOKENS=786432

**Files:**
- remote/remote_gpu_guide.md
- remote/runpod_1gpu.sh
- remote/runpod_2gpu.sh
- remote/runpod_8gpu.sh

**Flow:**
provision remote GPU -> install gcc for torch.compile -> use python entrypoint -> choose batch tokens by GPU count -> choose test or production runtime knobs -> launch training

**Timestamp:** 2026-04-01

**Patterns:**
- `^TRAIN_BATCH_TOKENS=(196608|393216|786432)$` - Supported batch token values called out for 1 GPU, 2 GPU, and 8 GPU launch configurations
- `^MAX_WALLCLOCK_SECONDS=(900|600)$` - Recommended wallclock values for test and production profiles
- `^WARMDOWN_ITERS=(1500|3500)$` - Recommended warmdown iteration counts for test and production profiles
- `^TTT_ENABLED=(0|1)$` - Toggle used to disable TTT for test runs and enable it for production runs

## Narrative
### Structure
The remote setup targets a chiasegpu.vn machine with one RTX 4090 24GB GPU and a PyTorch 2.9.1+cu128 environment. Operational notes specify that shell commands should invoke python instead of python3, and gcc must be installed through apt before relying on torch.compile.

### Dependencies
This setup depends on a Linux environment where apt is available for installing gcc, a CUDA-compatible PyTorch 2.9.1+cu128 build, and launch scripts that scale TRAIN_BATCH_TOKENS according to 1-GPU, 2-GPU, or 8-GPU configurations.

### Highlights
Recommended batch-token settings are 196608 for 1 GPU, 393216 for 2 GPUs, and 786432 for 8 GPUs. A single RTX 4090 runs out of memory at 786432 tokens. Test runs use MAX_WALLCLOCK_SECONDS=900, WARMDOWN_ITERS=1500, and TTT_ENABLED=0, while production uses MAX_WALLCLOCK_SECONDS=600, WARMDOWN_ITERS=3500, and TTT_ENABLED=1.

### Examples
Example test profile: TRAIN_BATCH_TOKENS=196608, MAX_WALLCLOCK_SECONDS=900, WARMDOWN_ITERS=1500, TTT_ENABLED=0 on a single 4090. Example production profile: choose GPU-count batch tokens, then set MAX_WALLCLOCK_SECONDS=600, WARMDOWN_ITERS=3500, and TTT_ENABLED=1.

## Facts
- **remote_gpu_provider**: Remote GPU provider is chiasegpu.vn [environment]
- **remote_gpu_hardware**: Remote training setup uses 1x RTX 4090 with 24GB VRAM [environment]
- **pytorch_version**: Remote environment uses PyTorch 2.9.1+cu128 [environment]
- **python_command**: Remote commands use python rather than python3 [convention]
- **torch_compile_system_dependency**: The remote machine needs apt install gcc for torch.compile [environment]
- **train_batch_tokens_1gpu**: TRAIN_BATCH_TOKENS is 196608 for 1 GPU runs [project]
- **train_batch_tokens_2gpu**: TRAIN_BATCH_TOKENS is 393216 for 2 GPU runs [project]
- **train_batch_tokens_8gpu**: TRAIN_BATCH_TOKENS is 786432 for 8 GPU runs [project]
- **single_4090_oom_batch_tokens**: TRAIN_BATCH_TOKENS 786432 causes OOM on a single RTX 4090 [project]
- **max_wallclock_seconds_test**: MAX_WALLCLOCK_SECONDS is 900 for test runs [project]
- **max_wallclock_seconds_production**: MAX_WALLCLOCK_SECONDS is 600 for production runs [project]
- **warmdown_iters_test**: WARMDOWN_ITERS is 1500 for test runs [project]
- **warmdown_iters_production**: WARMDOWN_ITERS is 3500 for production runs [project]
- **ttt_enabled_test**: TTT_ENABLED is 0 for test runs [project]
- **ttt_enabled_production**: TTT_ENABLED is 1 for production runs [project]
