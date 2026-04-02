---
children_hash: b63839352fdf47a370470dd1a7d3e91aa72719f54b6e0a8f39356987fda4f56e
compression_ratio: 0.45042492917847027
condensation_order: 1
covers: [architecture_snapshot.md, context.md, execution_next_steps_and_submission_criteria.md, project_snapshot.md, remote_gpu_setup_config.md, train_gpt_bugfix_benchmark_update.md]
covers_token_total: 3883
summary_level: d1
token_count: 1749
type: summary
---
# parameter_golf Structural Summary

## Overview
The `parameter_golf` topic captures the current state of a compact LM training effort centered on `train_gpt.py`, remote GPU execution scripts under `remote/`, and result packaging under `records/track_10min_16mb/`. Across the entries, the main thread is: define the compact 27.9M architecture, fix training issues in `train_gpt.py`, validate on RTX 4090, then scale to 8xH100 SXM for submission-quality evaluation.

See:
- `context.md`
- `architecture_snapshot.md`
- `project_snapshot.md`
- `train_gpt_bugfix_benchmark_update.md`
- `remote_gpu_setup_config.md`
- `execution_next_steps_and_submission_criteria.md`

## Core Architecture
From `architecture_snapshot.md` and reinforced by `project_snapshot.md`, the model is an 11-layer, 512d Transformer-style architecture with compact efficiency-oriented modifications:

- 27.9M parameters
- 1024-token BPE vocabulary
- BigramHash + TrigramHash features with 4096 entries and dim128 embeddings
- 8 query heads, 4 KV heads
- SmearGate causal mechanism
- LeakyReLU(0.5)^2 MLP with 3x expansion
- Partial RoPE on 16/64 dimensions
- gated attention + value residuals
- XSA enabled only in the last 4 layers
- layer norm scaling: `1/sqrt(layer+1)`
- depth recurrence on layers 4 and 5 for 2 passes
- project snapshot also calls out Int6 quantization-aware training (QAT)

High-level architecture flow:
- tokenizer setup
- hashed n-gram features
- causal/gated attention stack
- XSA in final layers
- recurrent passes on layers 4 and 5

For exact architecture drill-down:
- `architecture_snapshot.md`

## Project State and Performance Trajectory
`project_snapshot.md` summarizes the current program status, while `train_gpt_bugfix_benchmark_update.md` explains the main recent improvement source.

Key state:
- project uses an 11-layer 512-dimensional Transformer with XSA, depth recurrence, BigramHash, and Int6 QAT
- main execution entry points are `train_gpt.py` and `remote/runpod_1gpu.sh`
- 10 bugs were fixed
- BPB improved from 1.42 to 1.31 at the project-snapshot level

`train_gpt_bugfix_benchmark_update.md` gives the more specific benchmark evidence:
- benchmark performed on 1x RTX 4090
- `raw_bpb=1.3217`
- `sliding_bpb=1.3077`
- previous sliding BPB: `1.4226`
- improvement: `0.115`
- artifact size: `15.79MB`

For status and metrics:
- `project_snapshot.md`
- `train_gpt_bugfix_benchmark_update.md`

## train_gpt.py Bugfix Set
The central implementation work is in `train_gpt.py`. `train_gpt_bugfix_benchmark_update.md` records 10 fixes tied directly to improved evaluation:

- compare raw, EMA, and SWA instead of leaving SWA unused
- move EMA start later to 40%
- set LR warmup to 50 steps
- keep optimizer momentum intact
- improve fit/quality formatting
- use `torch.load` with `weights_only`
- remove eval gradient accumulation steps
- apply combined warmup-times-warmdown schedule
- set `atms` init to 0
- guard QAT so it starts only after warmup

Important relationships:
- EMA timing, LR warmup, and QAT gating are treated as quality-critical
- the QAT guard/fix is the immediate validation target before production scaling
- benchmark gains are attributed to this bugfix pass

For exact bugfix list and benchmark:
- `train_gpt_bugfix_benchmark_update.md`

## Remote Execution and Environment Configuration
`remote_gpu_setup_config.md` documents the remote operating environment and launch configuration values. The setup is tied to:

Files:
- `remote/remote_gpu_guide.md`
- `remote/runpod_1gpu.sh`
- `remote/runpod_2gpu.sh`
- `remote/runpod_8gpu.sh`

Environment facts:
- provider: `chiasegpu.vn`
- hardware baseline: `1x RTX 4090 24GB`
- PyTorch: `2.9.1+cu128`
- use `python`, not `python3`
- install `gcc` via `apt` for `torch.compile`

Batch-token scaling by GPU count:
- `TRAIN_BATCH_TOKENS=196608` for 1 GPU
- `TRAIN_BATCH_TOKENS=393216` for 2 GPU
- `TRAIN_BATCH_TOKENS=786432` for 8 GPU
- `TRAIN_BATCH_TOKENS=786432` causes OOM on a single RTX 4090

Runtime profile knobs:
- test profile:
  - `MAX_WALLCLOCK_SECONDS=900`
  - `WARMDOWN_ITERS=1500`
  - `TTT_ENABLED=0`
- production profile:
  - `MAX_WALLCLOCK_SECONDS=600`
  - `WARMDOWN_ITERS=3500`
  - `TTT_ENABLED=1`

Exact config patterns preserved in the entry:
- `^TRAIN_BATCH_TOKENS=(196608|393216|786432)$`
- `^MAX_WALLCLOCK_SECONDS=(900|600)$`
- `^WARMDOWN_ITERS=(1500|3500)$`
- `^TTT_ENABLED=(0|1)$`

For environment and launch configuration:
- `remote_gpu_setup_config.md`

## Execution Plan and Hardware Progression
`execution_next_steps_and_submission_criteria.md` turns the project state into an ordered execution plan:

1. apply the `train_gpt.py` QAT bug10 fix
2. upload fixed `train_gpt.py` to remote
3. run a longer `1xRTX4090` retest for `30–60 minutes`
4. optionally use `2xRTX4090` for faster iteration
5. run production evaluation on `8xH100 SXM`
6. use production seeds: `1337`, `42`, `2025`
7. assemble submission artifacts in `records/track_10min_16mb/`
8. validate BPB and statistical significance against SOTA

This extends the shorter progression in `project_snapshot.md`:
- QAT re-test first
- 8xH100 production run after validation

For execution sequencing:
- `project_snapshot.md`
- `execution_next_steps_and_submission_criteria.md`

## Submission and Acceptance Criteria
The submission requirements are concentrated in `execution_next_steps_and_submission_criteria.md`.

Target and benchmark references:
- target BPB: below `1.12`
- current SOTA: `1.1194`

Acceptance rules:
- submission requires `SOTA+0.005 nats` improvement with statistical significance
- significance threshold: `p < 0.01`
- production must use 3 seeds: `1337`, `42`, `2025`

Required artifact packaging under:
- `records/track_10min_16mb/`

Required bundle contents:
- `README.md`
- `submission.json`
- `train_gpt.py`
- 3 logs

For final packaging and rules:
- `execution_next_steps_and_submission_criteria.md`

## Cross-Entry Relationships
- `context.md` provides the topic-level frame: 11-layer 512d Transformer, XSA, depth recurrence, BigramHash, Int6 QAT, BPB improvement tracking, experiments under `records/`, remote execution under `remote/`.
- `architecture_snapshot.md` gives the precise architecture specification.
- `project_snapshot.md` gives the compact current-state summary and next checkpoint.
- `train_gpt_bugfix_benchmark_update.md` explains why quality improved and quantifies the improvement.
- `remote_gpu_setup_config.md` provides the environment constraints and launch parameters needed to execute the plan.
- `execution_next_steps_and_submission_criteria.md` defines the operational path from fixed code to final competition submission.

## Practical Drill-Down Guide
- Architecture details: `architecture_snapshot.md`
- Current project state: `project_snapshot.md`
- `train_gpt.py` fixes and benchmark numbers: `train_gpt_bugfix_benchmark_update.md`
- Remote hardware/config values: `remote_gpu_setup_config.md`
- Next actions and submission rules: `execution_next_steps_and_submission_criteria.md`