# Smoke test for RTX 3050 Ti (4GB VRAM)
# Purpose: Verify code runs, serialization works, artifact valid
# Expected: ~3-5 minutes, train_loss decreasing, artifact generated

$env:SEED = "42"
$env:ITERATIONS = "30"
$env:MAX_WALLCLOCK_SECONDS = "300"

# Tiny batch to fit in 4GB VRAM
$env:TRAIN_BATCH_TOKENS = "4096"
$env:TRAIN_SEQ_LEN = "512"
$env:VAL_BATCH_SIZE = "2048"
$env:EVAL_SEQ_LEN = "512"

# Reduced model for VRAM
$env:NUM_LAYERS = "4"
$env:MODEL_DIM = "256"
$env:NUM_HEADS = "4"
$env:NUM_KV_HEADS = "2"
$env:MLP_MULT = "2"

# Tiny n-gram tables
$env:BIGRAM_VOCAB_SIZE = "512"
$env:BIGRAM_DIM = "64"
$env:TRIGRAM_VOCAB_SIZE = "0"

# Disable heavy features
$env:DEPTH_RECUR_PASSES = "1"
$env:DEPTH_RECUR_LAYERS = ""
$env:TTT_ENABLED = "0"
$env:GATED_ATTENTION = "0"
$env:VALUE_RESIDUAL = "0"
$env:VE_ENABLED = "0"
$env:SWA_ENABLED = "0"
$env:XSA_LAST_N = "0"
$env:ROPE_DIMS = "0"
$env:LN_SCALE = "0"
$env:LATE_QAT_THRESHOLD = "0"
$env:EVAL_STRIDE = "128"
$env:EVAL_TEMPERATURE = "1.0"

# New: bug-fix settings
$env:LR_WARMUP_STEPS = "5"
$env:EMA_START_FRAC = "0.3"
$env:WARMUP_STEPS = "2"
$env:WARMDOWN_ITERS = "10"

# torch.compile OFF (3050 Ti lacks SMs)
$env:TORCH_COMPILE = "0"

# Logging
$env:TRAIN_LOG_EVERY = "5"
$env:VAL_LOSS_EVERY = "0"

# Single-GPU without DDP: do NOT set RANK/WORLD_SIZE
Remove-Item Env:RANK -ErrorAction SilentlyContinue
Remove-Item Env:WORLD_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:LOCAL_RANK -ErrorAction SilentlyContinue
Remove-Item Env:MASTER_ADDR -ErrorAction SilentlyContinue
Remove-Item Env:MASTER_PORT -ErrorAction SilentlyContinue

Write-Host "=== Parameter Golf Smoke Test ===" -ForegroundColor Cyan
Write-Host "GPU: RTX 3050 Ti (4GB VRAM)" -ForegroundColor Yellow
Write-Host "Model: 4L/256d/4H (mini) - 30 steps" -ForegroundColor Yellow
Write-Host "Fixes: LR warmup, EMA late start, lzma compression, no trigram" -ForegroundColor Yellow
Write-Host ""

python train_gpt.py
