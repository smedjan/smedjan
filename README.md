# AndreAI

Pure Rust LLM training and inference engine. Zero Python. Zero PyTorch. Zero cloud dependency.

16.4K lines of Rust + GPU kernels. Trains, fine-tunes, aligns, and deploys transformer language models from scratch.

## Why

Own the stack. Every line of code, every GPU kernel, every byte of the model. No frameworks, no runtimes, no dependencies you don't control. Runs on macOS today and NVIDIA CUDA hosts next; AndreOS is a planned native target once its GPU backend lands.

## Architecture

```
model.rs / attention.rs     ← Transformer (RoPE, GQA, SwiGLU, weight-tied lm_head)
tensor.rs                   ← GPU tensor operations
autograd.rs                 ← Tape-based reverse-mode autodiff
      ↓ (backend-agnostic above this line)
metal/   cuda/               ← GPU backends (compile-time selected)
```

**Two GPU backends, one codebase:**

```bash
cargo build --release                                    # Metal (macOS/Apple Silicon)
cargo build --release --features cuda --no-default-features    # CUDA (NVIDIA)
```

Checkpoints are portable across supported backends. Train on Mac, resume on H100, and carry the same checkpoint format into future backends.

## Features

### Training Pipeline

| Stage | Command | Description |
|-------|---------|-------------|
| Pre-train | `andreai train` | Train from scratch on raw text |
| Distill | `andreai train --teacher-checkpoint teacher.bin` | Knowledge distillation (KL + CE) |
| SFT | `andreai sft` | Supervised fine-tuning on instruction data |
| DPO | `andreai dpo` | Direct Preference Optimization (alignment) |
| Evaluate | `andreai eval` | Benchmark across 8 categories |

### Inference

| Feature | Command |
|---------|---------|
| Generate | `andreai generate --checkpoint model.bin --prompt "Hello"` |
| Speculative | `andreai generate --speculative --draft-checkpoint draft.bin` |
| Streaming | `andreai generate --stream` |

### Data Pipeline

| Command | Description |
|---------|-------------|
| `andreai tokenizer` | Train BPE tokenizer on corpus |
| `andreai prepare` | Convert text to binary training format |
| `andreai process` | Quality filtering, deduplication |
| `andreai mix` | Mix datasets with custom ratios |
| `andreai dpo-prepare` | Convert JSONL preference pairs to binary |

### Model

- **Architecture**: Decoder-only transformer with pre-norm (RMSNorm)
- **Attention**: Multi-Head or Grouped Query (GQA) via `--kv-heads`
- **Activation**: SwiGLU feed-forward
- **Position**: Rotary Position Embeddings (RoPE) with NTK-aware scaling
- **Weight tying**: Embedding matrix shared with language model head
- **Sizes**: 2M (tiny) to 6.5B (8b), plus fully custom via `--size custom --dim --layers --heads`

### Optimization

- **FP16 mixed precision**: Half-precision shared memory + FP16 input matmuls with float accumulator
- **Gradient accumulation**: `--grad-accum N` (effective_batch = batch_size * N)
- **Gradient checkpointing**: `--gradient-checkpointing` (trade compute for memory)
- **AdamW** with cosine warmup scheduler and optional warm restarts (`--lr-restart`)
- **Gradient clipping** with NaN/Inf detection and zeroing

### Production

- **Checkpoint resume**: `--resume state_5000.bin` (saves optimizer state + model + step)
- **Validation loss**: `--val-dataset val.bin` (eval every checkpoint interval)
- **Early stopping**: Stops after 3 intervals without validation improvement
- **Training CSV log**: `{checkpoint_dir}/train.csv` with step, loss, lr, tok/s, tokens
- **Quantization**: `andreai quantize` (Q4/Q8 post-training)

## Performance

Throughput (removed: broken benchmark):

| Machine | Original | Optimized | Speedup |
|---------|----------|-----------|---------|
| Mac mini M1 | benchmark removed (broken bench era) |
| MacBook Air M3 | benchmark removed (broken bench era) |
| **Combined** | **294 tok/s** | **~1,550 sustained** | **5.3x** |

Key optimizations:
- Batched matmul shaders (96 GPU dispatches → 6 per attention layer)
- FP16 mixed precision (half shared memory + half input reads)
- Merged forward+backward GPU command batch
- Clamped FP16 casts (prevents NaN from half overflow)
- RoPE sincos single instruction

98M Chinchilla-optimal training (2B tokens): ~15 days on both machines.

## GPU Kernels

46 CUDA + 35 Metal kernels covering:

**Compute**: tiled matmul (6 variants), batched matmul (6 variants), FP16-input matmul (6 variants)

**Normalization**: softmax, RMS norm, fused residual+RMS norm

**Activation**: SiLU, SiLU-gate (fused), RoPE forward/backward

**Training**: cross-entropy loss, KL divergence (distillation), AdamW update, gradient clipping (L2 norm + NaN check)

**Inference**: argmax, temperature scaling, causal masking, KV cache operations

**Utility**: FP16 cast (with clamp), transpose permutation, embedding lookup/backward, buffer copy

All kernels use FP16 shared memory with float accumulator (mixed precision).

## File Structure

```
src/
  main.rs          (749)   15 CLI commands
  model.rs         (488)   Transformer architecture, 8 presets + custom
  attention.rs     (484)   Multi-head attention, GQA, KV cache
  tensor.rs        (830)   GPU tensor operations
  autograd.rs      (901)   Tape-based autodiff, gradient checkpointing
  train.rs         (428)   Training loop, grad accum, validation, resume
  generate.rs      (632)   Inference, speculative decoding
  dpo.rs          (1171)   Direct Preference Optimization
  sft.rs           (633)   Supervised fine-tuning
  loss.rs          (171)   Cross-entropy + distillation loss
  optim.rs         (165)   AdamW, cosine warmup with restarts
  checkpoint.rs    (317)   Save/load model + optimizer state
  tokenizer.rs     (370)   BPE train/encode/decode
  data.rs          (239)   Mmap dataset, DataLoader
  datapipe.rs      (480)   Quality filter, dedup, mixing
  eval.rs          (365)   Benchmark (8 categories)
  quantize.rs      (506)   Q4/Q8 quantization
  api.rs           (115)   Rust embedding/integration API
  tests.rs        (1463)   81 tests
  gpu.rs            (16)   Backend feature gate

  metal/                   macOS Metal backend
    shaders.rs    (2505)   35 MSL kernels
    compute.rs    (1165)   Dispatch functions
    mod.rs         (469)   MetalContext, buffer pool, command batching

  cuda/                    NVIDIA CUDA backend
    kernels.rs    (1128)   46 CUDA kernels
    compute.rs     (207)   Dispatch functions
    mod.rs         (107)   CudaContext (cudarc)

```

**16,416 lines total. 81 tests. 56 commits.**

## Quick Start

```bash
# Train a tokenizer
andreai tokenizer --input corpus.txt --vocab-size 8192 --output tokenizer.bin

# Prepare training data
andreai prepare --input corpus.txt --tokenizer tokenizer.bin --output train.bin

# Train a model
andreai train \
  --dataset train.bin \
  --tokenizer tokenizer.bin \
  --size medium \
  --batch-size 4 \
  --seq-len 256 \
  --steps 50000 \
  --lr 3e-4 \
  --checkpoint-dir checkpoints/

# Resume after crash
andreai train \
  --dataset train.bin \
  --tokenizer tokenizer.bin \
  --size medium \
  --steps 50000 \
  --resume checkpoints/state_25000.bin

# Generate text
andreai generate \
  --checkpoint checkpoints/final.bin \
  --tokenizer tokenizer.bin \
  --prompt "The" \
  --stream

# DPO alignment
andreai dpo-prepare --input prefs.jsonl --output prefs.bin --tokenizer tokenizer.bin
andreai dpo \
  --checkpoint checkpoints/sft_final.bin \
  --ref-checkpoint checkpoints/sft_final.bin \
  --dataset prefs.bin \
  --tokenizer tokenizer.bin \
  --beta 0.1
```

## License

Private. All rights reserved.
