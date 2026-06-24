<p align="center">
  <a href="https://smedjan.dev"><img src="https://raw.githubusercontent.com/smedjan/smedjan/main/docs/banner.png" alt="SMEDJAN — pure-Rust LLM engine. Own the stack." width="840"></a>
</p>

<p align="center">
  <a href="https://crates.io/crates/smedjan"><img src="https://img.shields.io/crates/v/smedjan?style=flat-square&amp;color=ff7a2f&amp;labelColor=0d1014" alt="crates.io"></a>
  <a href="https://crates.io/crates/smedjan"><img src="https://img.shields.io/crates/d/smedjan?style=flat-square&amp;color=ff7a2f&amp;labelColor=0d1014" alt="downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/crates/l/smedjan?style=flat-square&amp;color=ff7a2f&amp;labelColor=0d1014" alt="MIT license"></a>
  <a href="https://smedjan.dev"><img src="https://img.shields.io/badge/site-smedjan.dev-ff7a2f?style=flat-square&amp;labelColor=0d1014" alt="website"></a>
</p>

<p align="center"><strong>Pure-Rust LLM training and inference engine — zero Python, zero PyTorch, zero cloud.</strong></p>

<p align="center">
  <a href="https://smedjan.dev"><strong>Website</strong></a> &nbsp;·&nbsp;
  <a href="https://crates.io/crates/smedjan">crates.io</a> &nbsp;·&nbsp;
  <a href="https://smedjan.dev/docs">Docs</a> &nbsp;·&nbsp;
  <a href="LICENSE">MIT</a>
</p>

*Smedjan* is Swedish for "the smithy" — the forge where you make your own tools. The whole stack is here: every line of code, every GPU kernel, every byte of the model. ~45K lines of Rust that train, fine-tune, align, quantize, and serve decoder-only transformer language models from scratch on your own hardware.

## Why

Own the stack. No frameworks, no runtimes, no dependencies you don't control — the entire dependency tree is a handful of small crates (`clap`, `rand`, `memmap2`, `byteorder`, and the GPU FFI bindings). Train on a laptop, resume on a datacenter GPU, and carry the same checkpoint format between them.

## Architecture

```
model.rs / attention.rs     ← Transformer (RoPE, GQA, SwiGLU, weight-tied lm_head)
tensor.rs                   ← GPU tensor operations
autograd.rs                 ← Tape-based reverse-mode autodiff
      ↓ (backend-agnostic above this line)
metal/   cuda/              ← GPU backends (compile-time selected)
```

**Two GPU backends, one codebase:**

```bash
cargo build --release                                          # Metal (macOS / Apple Silicon)
cargo build --release --no-default-features --features cuda    # CUDA (NVIDIA)
```

Checkpoints are portable across supported backends. Train on a Mac, resume on an H100, keep the same checkpoint format.

## Install

Requires a recent stable Rust toolchain ([rustup](https://rustup.rs)).

```bash
git clone https://github.com/smedjan/smedjan.git
cd smedjan
cargo build --release        # builds ./target/release/smedjan
```

On macOS the Metal backend is the default and needs no extra setup. For CUDA you need the CUDA toolkit (12.x) installed; build with `--no-default-features --features cuda`.

## Quick Start

```bash
# Train a BPE tokenizer
smedjan tokenizer --input corpus.txt --vocab-size 8192 --output tokenizer.bin

# Prepare training data
smedjan prepare --input corpus.txt --tokenizer tokenizer.bin --output train.bin

# Train a model
smedjan train \
  --dataset train.bin \
  --tokenizer tokenizer.bin \
  --size medium \
  --batch-size 4 \
  --seq-len 256 \
  --steps 50000 \
  --lr 3e-4 \
  --checkpoint-dir checkpoints/

# Resume after a crash (restores model + optimizer state + step)
smedjan train --dataset train.bin --tokenizer tokenizer.bin --size medium \
  --steps 50000 --resume checkpoints/state_25000.bin

# Generate text
smedjan generate \
  --checkpoint checkpoints/final.bin \
  --tokenizer tokenizer.bin \
  --prompt "The" \
  --stream

# DPO alignment
smedjan dpo-prepare --input prefs.jsonl --output prefs.bin --tokenizer tokenizer.bin
smedjan dpo \
  --checkpoint checkpoints/sft_final.bin \
  --ref-checkpoint checkpoints/sft_final.bin \
  --dataset prefs.bin \
  --tokenizer tokenizer.bin \
  --beta 0.1
```

## Features

### Training pipeline

| Stage | Command | Description |
|-------|---------|-------------|
| Pre-train | `smedjan train` | Train from scratch on raw text |
| Distill | `smedjan train --teacher-checkpoint teacher.bin` | Knowledge distillation (KL + CE) |
| SFT | `smedjan sft` | Supervised fine-tuning on instruction data |
| DPO | `smedjan dpo` | Direct Preference Optimization (alignment) |
| Evaluate | `smedjan eval` | Benchmark across categories |

### Inference

| Feature | Command |
|---------|---------|
| Generate | `smedjan generate --checkpoint model.bin --prompt "Hello"` |
| Speculative | `smedjan generate --speculative --draft-checkpoint draft.bin` |
| Streaming | `smedjan generate --stream` |

### Data pipeline

| Command | Description |
|---------|-------------|
| `smedjan tokenizer` | Train a BPE tokenizer on a corpus |
| `smedjan prepare` | Convert text to binary training format |
| `smedjan process` | Quality filtering and deduplication |
| `smedjan mix` | Mix datasets with custom ratios |
| `smedjan dpo-prepare` | Convert JSONL preference pairs to binary |

### Model

- **Architecture**: Decoder-only transformer, pre-norm (RMSNorm)
- **Attention**: Multi-Head or Grouped-Query (GQA) via `--kv-heads`; also Linear, SSM (Mamba-2/SSD), RWKV, MLA, and block-sparse mixers
- **Activation**: SwiGLU feed-forward; Mixture-of-Experts routing available
- **Position**: Rotary Position Embeddings (RoPE) with NTK-aware and YaRN scaling
- **Weight tying**: Embedding matrix shared with the language-model head
- **Sizes**: tiny (2M) through 6.5B, plus fully custom via `--size custom --dim --layers --heads`

### Optimization

- **Mixed precision**: FP16 / BF16 input matmuls with float accumulators
- **Gradient accumulation**: `--grad-accum N` (effective batch = batch × N)
- **Gradient checkpointing**: `--gradient-checkpointing` (trade compute for memory)
- **Optimizers**: AdamW (cosine warmup, optional warm restarts), Muon / NorMuon, hybrid per-parameter-group, BitNet ternary training
- **Gradient clipping** with NaN/Inf detection and zeroing

### Interop & production

- **Checkpoint resume**: `--resume state_5000.bin` (model + optimizer state + step)
- **Validation & early stopping**: `--val-dataset val.bin`, stops after N intervals without improvement
- **Quantization**: `smedjan quantize` (Q4 / Q8 post-training)
- **safetensors I/O**: zero-dependency import/export with F32/BF16/F16 weights, plus `smedjan import-hf` (HuggingFace `config.json` → model) for continued-training retrofits
- **GGUF export**: real GGML `f32` / `q8_0` / `q4_0` blocks (32-byte aligned, 1-D norms kept f32), validated against the reference GGUF dequantizer. Valid GGML weight container; turnkey `llama.cpp` *inference* (tokenizer embedding + RoPE/QK-norm parity) is on the roadmap

## Performance

Measured with `smedjan bench` on an Apple M1 Mac mini (16 GB) — batch 4, sequence length 128, hardware simdgroup-MMA path (the default). Real throughput, not theoretical peaks:

| Preset | Inference (forward) | Decode (1 tok, KV cache) | Train (fwd+bwd) |
|--------|--------------------:|-------------------------:|----------------:|
| `small` (7.2M · d256/6L) | 22,900 tok/s | 173 tok/s | 4,400 tok/s |
| `medium` (45M · d512/12L) | 5,090 tok/s | 65 tok/s | 1,150 tok/s |

The hardware simdgroup-MMA matmul path is on by default (bit-identical to the scalar kernels) and runs ~1.3–1.4× faster — on `medium`, inference 3,600 → 5,090 tok/s and training 785 → 1,150 tok/s. Measure the scalar fallback with `smedjan bench --no-simdgroup-matmul`. Other Metal-pass wins: batched matmul shaders, FP16 mixed precision with float accumulators, a merged forward+backward command batch, and single-instruction RoPE sincos. Reproduce on your own hardware with `smedjan bench --size <preset>`.

### NVIDIA (CUDA)

On an RTX 4090 the matmul path runs on cuBLAS TF32 tensor cores. Forward inference reaches ~70,000 tok/s on a 214M model — about 4× the initial portable kernels — and a multi-block gradient-norm reduction roughly halved per-step GPU time, lifting checkpointed training throughput ~1.6× (`medium`, batch 16, seq 256). The full 269-test suite passes on **both** Metal and CUDA, and checkpoints are portable between them. Build with `--no-default-features --features cuda`.

## Module map

```
src/
  main.rs        CLI entry point and subcommands
  model.rs       Transformer architecture, presets + custom sizes
  attention.rs   Multi-head & grouped-query attention, RoPE, KV cache
  linear_attention.rs · ssm.rs · rwkv.rs · mla.rs
                 Alternative sequence mixers (Linear, Mamba-2/SSD, RWKV, MLA, block-sparse)
  tensor.rs      GPU tensor operations
  gpu.rs         Backend-agnostic GPU dispatch facade
  autograd.rs    Tape-based autodiff, gradient checkpointing
  train.rs       Training loop, grad accum, validation, resume
  generate.rs    Inference, sampling, speculative decoding
  dpo.rs         Direct Preference Optimization
  sft.rs         Supervised fine-tuning
  distill.rs     Knowledge distillation (KL + CE teacher transfer)
  loss.rs        Cross-entropy + distillation loss
  optim.rs       AdamW, Muon/NorMuon, schedulers
  checkpoint.rs  Save/load model + optimizer state
  tokenizer.rs   BPE train / encode / decode
  data.rs        Mmap dataset + DataLoader
  datapipe.rs    Quality filtering, dedup, mixing
  eval.rs        Benchmark harness
  quantize.rs    Q4/Q8 quantization, GGUF export
  safetensors.rs safetensors + HF-Llama interop
  api.rs         Rust embedding/integration API

  metal/         macOS Metal backend (MSL kernels, dispatch, buffer pool)
  cuda/          NVIDIA CUDA backend (cudarc)
```

~45K lines of Rust, 260+ tests.

## Testing

The test suite runs against the GPU backend, so it must be run on real hardware:

```bash
# macOS / Apple Silicon
cargo test --release

# GPU tests that touch shared device state run serially
cargo test --release -- --include-ignored --test-threads=1

# NVIDIA
cargo test --release --no-default-features --features cuda
```

## Roadmap

- Faithful **bit-exact** HF *inference* parity (half-split RoPE, fixed QK-norm). The `config.json` → model + F32/BF16/F16 import path already works for continued training (`smedjan import-hf`); reproducing HF inference to the bit is the remaining piece.
- Turnkey `llama.cpp` inference from an exported GGUF (embed the tokenizer vocab + the same RoPE/QK-norm parity as above; the GGML weight blocks themselves are already correct)
- Chunked O(N) RWKV forward (the SSM chunked path already exists) — RWKV already trains via the stable materialised WKV
- Stronger long-context (NIAH / RULER) curves on better-trained checkpoints (the eval harness ships: `smedjan eval --longctx`)

## License

[MIT](LICENSE) © Andrei Dodu
