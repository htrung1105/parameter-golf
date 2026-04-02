---
title: Train GPT Bugfix Benchmark Update
tags: []
keywords: []
importance: 50
recency: 1
maturity: draft
createdAt: '2026-04-01T07:35:23.722Z'
updatedAt: '2026-04-01T07:35:23.722Z'
---
## Raw Concept
**Task:**
Document 10 bug fixes and resulting benchmark improvement in train_gpt.py

**Changes:**
- Compared raw, EMA, and SWA instead of leaving SWA unused
- Moved EMA start later to 40 percent to avoid pollution
- Set LR warmup to 50 steps
- Kept optimizer momentum intact
- Improved fit and quality formatting
- Used torch.load with weights_only
- Removed eval gradient accumulation steps
- Applied combined warmup times warmdown schedule
- Set atms init to 0
- Guarded QAT to start only after warmup

**Files:**
- train_gpt.py

**Flow:**
apply 10 train_gpt.py fixes -> run test run 2 on 1xRTX4090 -> compare sliding BPB against previous result -> measure artifact size

**Timestamp:** 2026-04-01

## Narrative
### Structure
This update records a focused bugfix pass in train_gpt.py followed by a benchmark check on a single RTX 4090. The note ties implementation corrections directly to observed model quality and artifact-size outcomes.

### Dependencies
The benchmark depends on train_gpt.py, the corrected EMA and SWA handling, the warmup and warmdown learning-rate schedule, and a 1xRTX4090 test environment. Reported numbers also depend on comparing against the previous sliding BPB baseline of 1.4226.

### Highlights
Test run 2 reported raw_bpb=1.3217 and sliding_bpb=1.3077 with a 15.79MB artifact. Relative to the previous sliding score of 1.4226, the run improved by 0.115 BPB, indicating materially better evaluation after the fixes.

### Examples
Bugfix list: SWA compare enabled; EMA late start 40pct; LR warmup 50 steps; keep optimizer momentum; fit+quality format fix; torch.load weights_only; eval gas removed; warmup*warmdown schedule; atms init 0; QAT guard step>warmup.

## Facts
- **bugfix_count**: 10 bugs were fixed in train_gpt.py [project]
- **ema_late_start**: EMA start was moved later to 40 percent to avoid pollution [project]
- **lr_warmup_steps**: LR warmup is 50 steps [project]
- **qat_guard**: QAT is guarded to start only after warmup [project]
- **raw_bpb**: Test run 2 on 1xRTX4090 achieved raw_bpb 1.3217 [environment]
- **sliding_bpb**: Test run 2 on 1xRTX4090 achieved sliding_bpb 1.3077 [environment]
- **artifact_size_mb**: Artifact size was 15.79MB [environment]
- **previous_sliding_bpb**: Previous sliding BPB was 1.4226 [project]
- **sliding_bpb_improvement**: Sliding BPB improved by 0.115 [project]
