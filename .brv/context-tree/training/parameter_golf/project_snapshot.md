---
title: Project Snapshot
tags: []
keywords: []
importance: 50
recency: 1
maturity: draft
createdAt: '2026-04-01T07:26:43.680Z'
updatedAt: '2026-04-01T07:26:43.680Z'
---
## Raw Concept
**Task:**
Document current Parameter Golf project snapshot and immediate execution plan

**Changes:**
- Fixed 10 bugs
- Improved BPB from 1.42 to 1.31
- Identified re-test of QAT fix as the next checkpoint
- Planned 8xH100 production run after validation

**Files:**
- train_gpt.py
- remote/runpod_1gpu.sh

**Flow:**
current architecture summary -> bug-fix outcome -> QAT re-test -> 8xH100 production run

**Timestamp:** 2026-04-01

## Narrative
### Structure
This snapshot describes the Parameter Golf project as an 11-layer, 512-dimensional Transformer configuration augmented with XSA, depth recurrence, BigramHash, and Int6 quantization-aware training. It identifies the main implementation and execution entry points as train_gpt.py and remote/runpod_1gpu.sh.

### Dependencies
Progress depends on validating the QAT fix before scaling to the planned 8xH100 production run. The documented next step implies that production execution should follow only after the re-test confirms the fix behaves as expected.

### Highlights
Ten bugs were fixed, improving bits-per-byte from 1.42 to 1.31. The project is positioned for a larger production training phase once the QAT re-test is complete.

### Examples
Grouped extracted subjects: model_architecture, model_features, bpb_improvement, key_files, next_steps

## Facts
- **model_architecture**: The Parameter Golf project uses an 11-layer 512-dimensional Transformer. [project]
- **model_features**: The model includes XSA, depth recurrence, BigramHash, and Int6 QAT. [project]
- **bpb_improvement**: Fixing 10 bugs improved BPB from 1.42 to 1.31. [project]
- **key_files**: Key files are train_gpt.py and remote/runpod_1gpu.sh. [project]
- **next_steps**: The next steps are to re-test the QAT fix and then run 8xH100 production. [project]
