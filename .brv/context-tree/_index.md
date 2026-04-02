---
children_hash: bb6fe9d5157cddaf0b6f697ba9fc922e5c47d282bf92f06f0113c827adc10120
compression_ratio: 0.8499095840867993
condensation_order: 3
covers: [README.md, training/_index.md]
covers_token_total: 1659
summary_level: d3
token_count: 1410
type: summary
---
# Training Knowledge Overview

The `training` knowledge tree centers on the `parameter_golf` effort: a compact language-model training project that combines architecture design, `train_gpt.py` stabilization, remote GPU execution, and submission-grade benchmarking. For details, drill into `parameter_golf/context.md`, `parameter_golf/architecture_snapshot.md`, `parameter_golf/project_snapshot.md`, `parameter_golf/train_gpt_bugfix_benchmark_update.md`, `parameter_golf/remote_gpu_setup_config.md`, and `parameter_golf/execution_next_steps_and_submission_criteria.md`.

## Core Topic: `parameter_golf`

### Scope
`parameter_golf` tracks the full training workflow:
- define the compact model architecture
- improve quality and correctness in `train_gpt.py`
- validate on `1x RTX 4090`
- scale remote execution to multi-GPU setups
- package results under `records/track_10min_16mb/`

### Repeatedly Referenced Project Assets
- `train_gpt.py`
- `remote/remote_gpu_guide.md`
- `remote/runpod_1gpu.sh`
- `remote/runpod_2gpu.sh`
- `remote/runpod_8gpu.sh`
- `records/track_10min_16mb/`

## Architecture Summary
From `parameter_golf/architecture_snapshot.md` and echoed by `parameter_golf/project_snapshot.md`, the current model design is an efficiency-tuned compact Transformer variant with these fixed decisions:

- `27.9M` parameters
- `11` layers
- model width `512d`
- `1024`-token BPE vocabulary
- `8` query heads, `4` KV heads
- BigramHash + TrigramHash features with `4096` entries and `dim128`
- SmearGate causal mechanism
- LeakyReLU(0.5)^2 MLP with `3x` expansion
- Partial RoPE on `16/64` dimensions
- gated attention with value residuals
- XSA enabled only in the last `4` layers
- layer scaling `1/sqrt(layer+1)`
- depth recurrence on layers `4` and `5` with `2` passes
- Int6 quantization-aware training is part of the active design state

Relationship between entries:
- `architecture_snapshot.md` = canonical specification
- `project_snapshot.md` = concise current-state restatement

## Training State and Metrics
`parameter_golf/project_snapshot.md` and `parameter_golf/train_gpt_bugfix_benchmark_update.md` define the current performance direction.

### Current Progress
- `10` bugs fixed in `train_gpt.py`
- BPB improved from about `1.42` to about `1.31`
- validation performed on `1x RTX 4090`
- artifact size: `15.79MB`

### Benchmark Values
From `train_gpt_bugfix_benchmark_update.md`:
- `raw_bpb=1.3217`
- `sliding_bpb=1.3077`
- previous sliding BPB: `1.4226`
- improvement: `0.115`

### Interpreted Pattern
Recent quality gains are attributed primarily to the `train_gpt.py` fix pass, with QAT timing and scheduler behavior treated as especially high-impact.

## `train_gpt.py` Fix Cluster
The main implementation changes are recorded in `parameter_golf/train_gpt_bugfix_benchmark_update.md`:

- compare raw, EMA, and SWA checkpoints instead of ignoring SWA
- delay EMA start to `40%`
- set LR warmup to `50` steps
- preserve optimizer momentum
- improve fit/quality formatting
- use `torch.load(..., weights_only=...)`
- remove eval gradient accumulation steps
- use combined warmup × warmdown scheduling
- initialize `atms` to `0`
- start QAT only after warmup

Key relationship:
- the QAT-after-warmup guard is the immediate checkpoint before larger-scale runs

## Remote GPU Execution Model
`parameter_golf/remote_gpu_setup_config.md` defines the remote environment and scaling rules.

### Environment
- provider: `chiasegpu.vn`
- baseline hardware: `1x RTX 4090 24GB`
- PyTorch: `2.9.1+cu128`
- use `python` instead of `python3`
- install `gcc` with `apt` for `torch.compile`

### Batch Token Scaling
- `TRAIN_BATCH_TOKENS=196608` for `1 GPU`
- `TRAIN_BATCH_TOKENS=393216` for `2 GPU`
- `TRAIN_BATCH_TOKENS=786432` for `8 GPU`
- `TRAIN_BATCH_TOKENS=786432` causes OOM on a single `RTX 4090`

### Runtime Profiles
Test profile:
- `MAX_WALLCLOCK_SECONDS=900`
- `WARMDOWN_ITERS=1500`
- `TTT_ENABLED=0`

Production profile:
- `MAX_WALLCLOCK_SECONDS=600`
- `WARMDOWN_ITERS=3500`
- `TTT_ENABLED=1`

### Preserved Config Patterns
- `^TRAIN_BATCH_TOKENS=(196608|393216|786432)$`
- `^MAX_WALLCLOCK_SECONDS=(900|600)$`
- `^WARMDOWN_ITERS=(1500|3500)$`
- `^TTT_ENABLED=(0|1)$`

## Execution Plan and Submission Criteria
`parameter_golf/execution_next_steps_and_submission_criteria.md` converts project state into an ordered plan:

1. apply the `train_gpt.py` QAT bugfix
2. upload fixed code to remote
3. retest on `1x RTX 4090` for `30–60` minutes
4. optionally iterate on `2x RTX 4090`
5. run production evaluation on `8x H100 SXM`
6. use seeds `1337`, `42`, `2025`
7. package outputs in `records/track_10min_16mb/`
8. validate BPB and statistical significance against SOTA

### Acceptance Targets
- target BPB: below `1.12`
- current SOTA reference: `1.1194`
- acceptance requires `SOTA+0.005 nats` improvement with significance
- significance threshold: `p < 0.01`

### Required Submission Bundle
- `README.md`
- `submission.json`
- `train_gpt.py`
- `3` logs

## Entry Relationship Map
- `parameter_golf/context.md` defines the overall scope of compact LM training, experiments, bugfixes, and run planning.
- `parameter_golf/architecture_snapshot.md` is the architectural source of truth.
- `parameter_golf/project_snapshot.md` gives the active project status and next checkpoint.
- `parameter_golf/train_gpt_bugfix_benchmark_update.md` explains the main source of recent BPB improvement.
- `parameter_golf/remote_gpu_setup_config.md` captures environment assumptions and launch configuration.
- `parameter_golf/execution_next_steps_and_submission_criteria.md` defines the path from validated fixes to submission-quality runs.