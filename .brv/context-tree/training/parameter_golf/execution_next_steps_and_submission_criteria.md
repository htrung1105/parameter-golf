---
title: Execution Next Steps And Submission Criteria
tags: []
related: [training/parameter_golf/project_snapshot.md, training/parameter_golf/train_gpt_bugfix_benchmark_update.md, training/parameter_golf/remote_gpu_setup_config.md]
keywords: []
importance: 50
recency: 1
maturity: draft
createdAt: '2026-04-01T07:51:56.707Z'
updatedAt: '2026-04-01T07:51:56.707Z'
---
## Raw Concept
**Task:**
Document immediate next steps for parameter golf training, remote execution, and submission readiness

**Changes:**
- Prioritized uploading a fixed train_gpt.py with the QAT bug10 fix to the remote environment
- Planned a longer single-GPU retest on 1xRTX4090 for 30-60 minutes
- Added 2xRTX4090 as an optional faster-iteration path
- Defined production evaluation on 8xH100 SXM using seeds 1337, 42, and 2025
- Specified submission artifact packaging under records/track_10min_16mb/
- Set target BPB and submission significance criteria against the current SOTA

**Files:**
- train_gpt.py
- records/track_10min_16mb/
- README.md

**Flow:**
apply QAT bug10 fix -> upload fixed train_gpt.py to remote -> run 1xRTX4090 retest for 30-60min -> optionally move to 2xRTX4090 for iteration speed -> run production 8xH100 SXM with seeds 1337,42,2025 -> assemble submission artifacts in records/track_10min_16mb/ -> validate BPB and significance against SOTA

**Timestamp:** 2026-04-01

## Narrative
### Structure
This note captures a concise execution plan for moving from a local code fix to remote retesting, then to production-scale validation and final competition submission packaging. The plan is sequenced around progressively stronger hardware: first a fixed remote upload, then a longer 1xRTX4090 validation run, then optional 2xRTX4090 iteration, and finally 8xH100 SXM production evaluation.

### Dependencies
Execution depends on having the corrected train_gpt.py available on the remote environment, access to RTX4090 and H100 hardware, and the ability to store final artifacts under records/track_10min_16mb/. Final acceptance also depends on statistical validation across the specified seed set and meeting the competition improvement threshold relative to the stated SOTA.

### Highlights
The performance target is BPB below 1.12 while the current SOTA is recorded as 1.1194. Submission readiness is not just a packaging task: it also requires SOTA+0.005 nats improvement with statistical significance, explicitly targeting p<0.01, plus a complete artifact bundle containing README.md, submission.json, train_gpt.py, and three logs.

### Rules
Submission requires SOTA+0.005 nats improvement with statistical significance.
Production 8xH100 SXM should use 3 seeds: 1337, 42, 2025.
Create submission records/track_10min_16mb/ with README.md, submission.json, train_gpt.py, 3 logs.

### Examples
Example execution order: upload fixed train_gpt.py with QAT bug10 fix, run a 30-60 minute 1xRTX4090 retest, then promote to 8xH100 SXM runs with seeds 1337, 42, and 2025 before packaging the submission directory.

## Facts
- **next_code_fix**: The immediate code change to deploy is a train_gpt.py QAT bug10 fix. [project]
- **single_gpu_retest_duration**: A longer 1xRTX4090 retest should run for 30 to 60 minutes. [project]
- **candidate_iteration_hardware**: 2xRTX4090 is being considered for faster iteration. [project]
- **production_hardware**: Production evaluation target hardware is 8xH100 SXM. [project]
- **production_seeds**: Production runs should use seeds 1337, 42, and 2025. [project]
- **submission_directory**: Submission artifacts should be created under records/track_10min_16mb/. [project]
- **submission_artifacts**: Submission packaging requires README.md, submission.json, train_gpt.py, and 3 logs. [convention]
- **target_bpb**: The target BPB is below 1.12. [project]
- **current_sota_bpb**: The current SOTA is 1.1194. [project]
- **submission_acceptance_rule**: Submission requires SOTA+0.005 nats improvement with statistical significance. [convention]
- **significance_threshold**: Statistical significance should reach p<0.01. [convention]
