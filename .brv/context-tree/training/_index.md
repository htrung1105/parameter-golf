---
children_hash: 54cb9b42a1edb37856fbc29c14bd20e57c5ab0ff9d44e3ea4c682cb2ba021658
compression_ratio: 0.7909407665505227
condensation_order: 2
covers: [context.md, parameter_golf/_index.md]
covers_token_total: 2009
summary_level: d2
token_count: 1589
type: summary
---
# training Structural Summary

## Domain Overview
The `training` domain captures Parameter Golf model-training knowledge: architecture definition, experiment progress, `train_gpt.py` optimization/bugfix work, remote GPU execution setup, and submission-oriented run planning.

Primary drill-down:
- `parameter_golf/context.md`
- `parameter_golf/architecture_snapshot.md`
- `parameter_golf/project_snapshot.md`
- `parameter_golf/train_gpt_bugfix_benchmark_update.md`
- `parameter_golf/remote_gpu_setup_config.md`
- `parameter_golf/execution_next_steps_and_submission_criteria.md`

## Main Topic: `parameter_golf`
`parameter_golf` is the active training topic for the compact LM effort. The core workflow across entries is:

1. define and stabilize the compact architecture
2. fix quality-impacting issues in `train_gpt.py`
3. validate on `1x RTX 4090`
4. scale to larger remote runs
5. package artifacts under `records/track_10min_16mb/` for submission-grade evaluation

Key project files and locations repeatedly referenced:
- `train_gpt.py`
- `remote/remote_gpu_guide.md`
- `remote/runpod_1gpu.sh`
- `remote/runpod_2gpu.sh`
- `remote/runpod_8gpu.sh`
- `records/track_10min_16mb/`

## Architecture Pattern
From `parameter_golf/architecture_snapshot.md` and `parameter_golf/project_snapshot.md`, the model is a compact efficiency-tuned Transformer variant with these persistent architectural decisions:

- 27.9M parameters
- 11 layers
- width `512d`
- `1024`-token BPE vocabulary
- `8` query heads, `4` KV heads
- BigramHash + TrigramHash features, `4096` entries, `dim128`
- SmearGate causal mechanism
- LeakyReLU(0.5)^2 MLP with `3x` expansion
- Partial RoPE on `16/64` dimensions
- gated attention with value residuals
- XSA enabled only in the final `4` layers
- layer scaling by `1/sqrt(layer+1)`
- depth recurrence on layers `4` and `5` with `2` passes
- Int6 quantization-aware training called out in project state

Relationship:
- `architecture_snapshot.md` is the precise architecture reference
- `project_snapshot.md` is the compact “current-state” restatement of the same design

## Training Progress and Benchmark Direction
`parameter_golf/project_snapshot.md` and `parameter_golf/train_gpt_bugfix_benchmark_update.md` together define the current performance trajectory.

Reported state:
- 10 bugs fixed in `train_gpt.py`
- BPB improved from roughly `1.42` to `1.31`
- benchmark validation was done on `1x RTX 4090`
- artifact size is `15.79MB`

More specific benchmark values from `train_gpt_bugfix_benchmark_update.md`:
- `raw_bpb=1.3217`
- `sliding_bpb=1.3077`
- previous sliding BPB: `1.4226`
- improvement: `0.115`

Pattern:
- recent quality gains are attributed directly to the `train_gpt.py` fix pass
- QAT timing and scheduler behavior are treated as especially sensitive levers

## `train_gpt.py` Fix Cluster
The central implementation topic is the bugfix set summarized in `parameter_golf/train_gpt_bugfix_benchmark_update.md`. The recorded fixes are:

- compare raw, EMA, and SWA checkpoints instead of leaving SWA unused
- move EMA start later to `40%`
- set LR warmup to `50` steps
- preserve optimizer momentum
- improve fit/quality formatting
- use `torch.load(..., weights_only=...)`
- remove eval gradient accumulation steps
- use combined warmup × warmdown scheduling
- initialize `atms` to `0`
- gate QAT so it starts only after warmup

Key relationship:
- the QAT-start guard is the immediate validation checkpoint before larger production runs

## Remote Execution Configuration
`parameter_golf/remote_gpu_setup_config.md` captures the execution environment and launch scaling rules for remote training.

Environment facts:
- provider: `chiasegpu.vn`
- baseline hardware: `1x RTX 4090 24GB`
- PyTorch: `2.9.1+cu128`
- use `python` rather than `python3`
- install `gcc` via `apt` to support `torch.compile`

Batch-token scaling:
- `TRAIN_BATCH_TOKENS=196608` for `1 GPU`
- `TRAIN_BATCH_TOKENS=393216` for `2 GPU`
- `TRAIN_BATCH_TOKENS=786432` for `8 GPU`
- `TRAIN_BATCH_TOKENS=786432` OOMs on a single `RTX 4090`

Runtime profiles:
- test:
  - `MAX_WALLCLOCK_SECONDS=900`
  - `WARMDOWN_ITERS=1500`
  - `TTT_ENABLED=0`
- production:
  - `MAX_WALLCLOCK_SECONDS=600`
  - `WARMDOWN_ITERS=3500`
  - `TTT_ENABLED=1`

Preserved config patterns:
- `^TRAIN_BATCH_TOKENS=(196608|393216|786432)$`
- `^MAX_WALLCLOCK_SECONDS=(900|600)$`
- `^WARMDOWN_ITERS=(1500|3500)$`
- `^TTT_ENABLED=(0|1)$`

## Execution Sequence and Submission Logic
`parameter_golf/execution_next_steps_and_submission_criteria.md` turns the technical state into an ordered run plan:

1. apply the `train_gpt.py` QAT bugfix
2. upload fixed code to remote
3. retest on `1x RTX 4090` for `30–60 minutes`
4. optionally iterate on `2x RTX 4090`
5. run production evaluation on `8x H100 SXM`
6. use seeds `1337`, `42`, `2025`
7. package outputs in `records/track_10min_16mb/`
8. validate BPB and significance against SOTA

Submission targets and constraints:
- target BPB: below `1.12`
- current SOTA reference: `1.1194`
- acceptance requires `SOTA+0.005 nats` improvement with statistical significance
- significance threshold: `p < 0.01`

Required bundle contents:
- `README.md`
- `submission.json`
- `train_gpt.py`
- `3` logs

## Entry Relationships
- `parameter_golf/context.md` frames the topic scope: compact LM training, architecture, experiments, bugfixes, and run plans.
- `parameter_golf/architecture_snapshot.md` provides the canonical model specification.
- `parameter_golf/project_snapshot.md` summarizes current status, metric direction, and the next validation checkpoint.
- `parameter_golf/train_gpt_bugfix_benchmark_update.md` explains the main source of recent BPB improvement.
- `parameter_golf/remote_gpu_setup_config.md` defines remote environment assumptions and launch parameters.
- `parameter_golf/execution_next_steps_and_submission_criteria.md` defines the path from validated bugfix to submission-quality multi-seed production runs.

## Drill-Down Guide
- Architecture details: `parameter_golf/architecture_snapshot.md`
- Status snapshot: `parameter_golf/project_snapshot.md`
- Bugfixes and benchmark deltas: `parameter_golf/train_gpt_bugfix_benchmark_update.md`
- Remote environment and run scripts: `parameter_golf/remote_gpu_setup_config.md`
- Next actions and acceptance rules: `parameter_golf/execution_next_steps_and_submission_criteria.md`