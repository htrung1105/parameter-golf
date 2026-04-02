---
title: Architecture Snapshot
tags: []
keywords: []
importance: 50
recency: 1
maturity: draft
createdAt: '2026-04-01T07:33:28.082Z'
updatedAt: '2026-04-01T07:33:28.082Z'
---
## Raw Concept
**Task:**
Document a compact parameter-golf model architecture snapshot

**Changes:**
- Captured architecture specification for 27.9M parameter model

**Flow:**
Tokenizer setup -> hashed n-gram features -> causal/gated attention stack -> XSA in final layers -> recurrent passes on layers 4 and 5

**Timestamp:** 2026-04-01

## Narrative
### Structure
This snapshot describes a 27.9M-parameter parameter-golf architecture with 1024-token BPE vocabulary, hashed bigram/trigram features, 11 transformer-style layers at width 512, and a mixed attention stack with 8 query heads and 4 KV heads.

### Dependencies
The design depends on BigramHash+TrigramHash features at 4096 entries with dim128 embeddings, SmearGate causal processing, Partial RoPE over 16 of 64 dimensions, and XSA enabled only in the last four layers. It also depends on depth recurrence that reprocesses layers 4 and 5 for two passes.

### Highlights
Notable features include LeakyReLU(0.5)^2 MLP blocks with 3x expansion, gated attention, value residuals, and layer-norm scaling by 1/sqrt(layer+1). The context is a compact architecture summary rather than a training log or evaluation result.

### Examples
Architecture details: 27.9M params, vocab 1024 BPE, BigramHash+TrigramHash 4096 dim128, SmearGate causal, 11 layers 512d 8H 4KV, MLP LeakyReLU(0.5)^2 3x expansion, Partial RoPE 16/64, Gated Attention, Value Residuals, XSA last 4 layers, LN Scale 1/sqrt(layer+1), depth recurrence layers 4,5 x2 passes.

## Facts
- **parameter_count**: Model architecture has 27.9M parameters [project]
- **vocab_size**: Tokenizer vocabulary size is 1024 BPE [project]
- **hash_features**: Hash features use BigramHash and TrigramHash with 4096 entries and dimension 128 [project]
- **causal_mechanism**: Model uses SmearGate causal mechanism [project]
- **layer_shape**: Architecture has 11 layers with width 512, 8 attention heads, and 4 KV heads [project]
- **mlp_activation**: MLP uses LeakyReLU(0.5)^2 with 3x expansion [project]
- **partial_rope**: Partial RoPE is applied with 16 of 64 dimensions [project]
- **attention_features**: Model uses gated attention and value residuals [project]
- **xsa_layers**: XSA is enabled in the last 4 layers [project]
- **ln_scale**: Layer norm scaling uses 1/sqrt(layer+1) [project]
- **depth_recurrence**: Depth recurrence runs layers 4 and 5 for two passes [project]
