# Smedjan Backlog

Capability audit + inference engine roadmap, 2026-07-01.

## Current State

- **v0.1.3** on crates.io, 293 tests, 0 clippy warnings, 0 `#[allow]`
- ~51.8K lines Rust (36.3K src + 15.6K backends)
- Two GPU backends: Metal (269/269) + CUDA (269/269), checkpoints portable
- 26 CLI subcommands
- Training pipeline: pretrain → SFT → DPO → distill → self-distill (all functional)
- Inference: greedy, temperature, top-p, top-k, min-p, typical, repetition penalty, streaming, batched, speculative decoding
- 6 attention variants: softmax/flash, linear, SSM/Mamba-2, RWKV, MLA, block-sparse, Gated DeltaNet
- MoE routing (ReMoE-style, top-K, shared expert, load balancing)
- Q4/Q8 quantization with GPU inference
- safetensors I/O (F32/BF16/F16), HF Llama + Qwen3.5 import, GGUF export
- 5 optimizers: AdamW, AdamW8bit, Muon, Sophia, Lion

---

## BACKLOG — by priority

### P0 — Inference engine (blocks the AI service business model)

#### 1. OpenAI-compatible HTTP serving layer
- **Gap**: No HTTP server anywhere. `api.rs` is a Rust embedding API (library calls), not a network service.
- **Need**: `axum` + `tokio` HTTP server with `/v1/chat/completions`, `/v1/completions`, `/v1/models` endpoints.
- **Why**: opencode, Hugin, Cursor, and any OpenAI-compatible client can call Smedjan as a backend.
- **Spec**: streaming (SSE), batched generation, model loading/unloading, config via CLI/env.
- **Effort**: ~2-3 days. Foundation exists (`api.rs` has `Smedjan::load`, `generate`, `generate_streaming`).

#### 2. MoE expert offloading (VRAM → RAM cache)
- **Gap**: All MoE experts resident in GPU memory (`experts: Vec<ExpertFFN>` allocated upfront). Model size bounded by VRAM.
- **Need**: LRU expert cache in VRAM, prefetch from RAM via PCIe based on router predictions.
- **Why**: Required to fit DeepSeek-V4-Flash (284B MoE, 256 experts) on 64GB VRAM — only 13B activated, but all experts must be accessible.
- **Spec**: 
  - Expert weights stored in RAM (mmap'd from safetensors)
  - LRU cache of active experts in VRAM (configurable size)
  - Router prediction prefetch (prefetch top-K+1 experts before gate computation completes)
  - Async PCIe transfer (overlap compute with transfer)
- **Effort**: ~1-2 weeks. GPU MoE kernels (`gpu_moe_gather`/`gpu_moe_scatter_add`) already exist but are unwired.

#### 3. Wire GPU MoE kernels into hot path
- **Gap**: `model.rs::moe_ffn` (line 790) uses a CPU-orchestrated per-expert loop (`for expert_idx in 0..self.n_experts { ... }`) with `scale_rows`. The fused GPU gather/scatter kernels exist in both backends but are only called from tests.
- **Need**: Replace the CPU loop with `gpu_moe_gather` → batched expert matmul → `gpu_moe_scatter_add`.
- **Why**: The CPU loop launches N separate GPU dispatches per layer; the fused kernel does it in 2 dispatches. Critical for MoE inference throughput.
- **Effort**: ~2-3 days. Kernels are written and tested; needs model.rs integration.

#### 4. FP4 quantization (Blackwell tensor cores)
- **Gap**: Zero FP4/FP8 support. Only Q4 (int4) and Q8 (int8) exist.
- **Need**: FP4 (e4m3) weight storage with FP8 activation, native Blackwell tensor core kernels.
- **Why**: DeepSeek-V4-Flash is natively FP4+FP8. Running it in FP16 would balloon to 568GB. FP4 keeps active experts at ~7GB in VRAM.
- **Spec**:
  - FP4 dequantize kernel (FP4 → FP16/BF16 on the fly in the matmul tile)
  - FP8 activation storage
  - cuBLASLt FP4 GEMM (or custom kernel if cuBLASLt doesn't support it)
  - Checkpoint format extension for FP4 tensors
- **Effort**: ~2-3 weeks. New kernels needed. Q4 dequant-in-tile pattern is the template.

#### 5. Expert-parallel multi-GPU
- **Gap**: Single-device only. `MetalContext::new()` / `CudaContext::new()` return one `Arc<Self>` for one device.
- **Need**: Multi-context orchestration — each GPU holds a subset of experts, router sends tokens to whichever GPU has the needed expert.
- **Why**: 2x RTX 5090 (64GB total) needed to hold enough experts for V4-Flash. Expert-parallel is simpler than tensor-parallel (no cross-GPU all-reduce on activations).
- **Spec**:
  - Multi-context manager (Vec<Arc<CudaContext>>)
  - Expert-to-GPU assignment (static or dynamic based on VRAM)
  - Cross-GPU token routing (CUDA IPC or host-staged)
  - KV cache per-GPU (no sharing needed for expert-parallel)
- **Effort**: ~1-2 weeks. Foundation is single-context CUDA; multi-context is additive.

#### 6. Continuous batching / request scheduling
- **Gap**: `generate_batch` does fixed-shape batched decode (equal-length prompts from a file). No dynamic batching.
- **Need**: In-flight request queue, dynamic batch formation, prefill/decode interleaving, paged-attention for variable-length sequences.
- **Why**: Production serving needs concurrent requests of varying length. Static batching wastes GPU on padding.
- **Spec**:
  - Request queue with priority
  - Dynamic batch formation (join new requests into running batch at decode step)
  - Paged KV cache (block-based, variable-length sequences)
  - Prefill/decode interleaving (prefill new request while decoding others)
- **Effort**: ~1-2 weeks. KV cache is per-layer; paged attention is a rewrite of `kv_cache.rs`.

### P1 — Correctness & feature gaps

#### 7. Alternative mixer KV-cache decode
- **Gap**: Linear, SSM, RWKV, and block-sparse mixers `assert_eq!(seq_q_len, seq_k)` — training/prefill only. No decode path (seq_q=1, seq_k=N).
- **Need**: Recurrent state form for each mixer (linear attention has a fixed-size state; SSM has selective state; RWKV has WKV state).
- **Why**: Can't serve models with non-softmax attention without this. Qwen3.5 hybrid (Gated DeltaNet) already has decode for FA layers but not DeltaNet layers.
- **Effort**: ~1 week per mixer. Linear attention is easiest (state = K^T V, O(1) update).

#### 8. Qwen3.5 speculative-decode placeholder embedding
- **Gap**: `spec_decode.rs:215` uses a hash-based `placeholder_embed` (`(tid*31 + j) % 100`) instead of the real quantized embedding. Used in the verify path.
- **Need**: Use `q_embed` (the real Q4 quantized embedding) in the verify forward pass.
- **Why**: Speculative decoding with degraded verify logits causes incorrect rejections/acceptances.
- **Effort**: ~1 day. The `q_embed` function exists; just needs to be called instead of the placeholder.

#### 9. GGUF import
- **Gap**: GGUF is export-only. No `load_gguf`/`import_gguf`.
- **Need**: Read GGUF files (f32, q8_0, q4_0, q4_K, q5_K, q6_K) into Smedjan model weights.
- **Why**: Turnkey llama.cpp inference from exported GGUFs (README roadmap item). Also lets Smedjan load any GGUF model from HuggingFace.
- **Effort**: ~3-5 days. GGUF format is documented; Q4_K/Q5_K/Q6_K block dequant is the bulk.

#### 10. CUDA YaRN RoPE
- **Gap**: `cuda/compute.rs:629` — `yarn_scale` is ignored on CUDA (plain RoPE only). Works on Metal.
- **Need**: Port the YaRN frequency-blend kernel to CUDA (mirror the Metal `rope_yarn` shader).
- **Why**: Long-context extension (YaRN) is a no-op on CUDA. Can't train/serve long-context models on CUDA with YaRN scaling.
- **Effort**: ~1 day. Metal kernel is the reference; CUDA kernel is a direct port.

### P2 — Performance

#### 11. bf16/fp16 activation storage
- **Gap**: Activations stored as fp32. Memory bandwidth is the training bottleneck (measured: matmul is only ~5% of step time on small models).
- **Need**: Store activations in bf16/fp16, halving memory traffic for the bottleneck ops (attention, layernorm, elementwise).
- **Why**: ~2x training speedup on memory-bound workloads. The bf16 GEMM itself gave ~0% (measured), but bf16 ACTIVATIONS attack the real bottleneck.
- **Effort**: ~1 week. Touches the entire autograd tape.

#### 12. Flash attention in training path
- **Gap**: Flash attention exists for inference but the training path materializes the full seq^2 attention scores (~1GB at seq256 b16).
- **Need**: Flash attention with backward pass (gradient through the fused kernel).
- **Why**: ~2-3x training speedup for seq>=256. The forward flash kernel exists; backward is the gap.
- **Effort**: ~1 week. Metal flash-attn backward kernel + CUDA equivalent.

#### 13. Kernel fusion / CUDA graphs
- **Gap**: ~294 tape ops/step, each a separate GPU kernel launch. Launch overhead + round-trip dominates.
- **Need**: Fuse norm+matmul+activation into single kernels. Use CUDA graphs to capture the entire step.
- **Why**: ~2x training speedup from reducing launch overhead. Measured: 430ms/step is non-matmul overhead.
- **Effort**: ~2 weeks. Fused kernels are bespoke per layer type; CUDA graphs need stable tensor addresses.

### P3 — Ecosystem

#### 14. Distributed training (data-parallel)
- **Gap**: Single-device training. `--grad-accum` is the only scaling knob.
- **Need**: Data-parallel gradient all-reduce across GPUs.
- **Why**: Train larger models faster. Multi-GPU training is table stakes for a serious engine.
- **Effort**: ~2-3 weeks. NCCL integration + gradient sync.

#### 15. Tensor-parallel inference
- **Gap**: No tensor-parallel sharding.
- **Need**: Split model layers across GPUs (pipeline-parallel) or split each layer's matmul across GPUs (tensor-parallel).
- **Why**: Serve models too large for one GPU. Complementary to expert-parallel (item 5).
- **Effort**: ~2-3 weeks. All-reduce on activations after each layer.

#### 16. Spot preemption handling
- **Gap**: No preemption detection or recovery.
- **Need**: Detect spot preemption, save KV cache state, relaunch on new instance, restore.
- **Why**: The AI service business model runs on spot B300s ($3.35/hr vs $9.16/hr dedicated). Spot instances can be claimed at any time. The service must handle preemption gracefully.
- **Spec**:
  - Periodic KV cache checkpoint (every N tokens)
  - Preemption detection (health check from client side)
  - Fast model reload + KV restore on new instance
  - Automatic failover (client retries to new instance)
- **Effort**: ~1 week. KV cache serialization is the bulk.

---

## Dependency graph

```
P0:
  1. HTTP serving layer ──────────────────┐
  3. Wire GPU MoE kernels ──────────────┐ │
  2. MoE expert offloading ───────────┐ │ │
  4. FP4 quantization ───────────────┐ │ │ │
  5. Expert-parallel multi-GPU ─────┤ │ │ │
  6. Continuous batching ───────────┤ │ │ │ │
                                   │ │ │ │ │
P1:                                │ │ │ │ │
  7. Mixer KV-cache decode         │ │ │ │ │
  8. Qwen3.5 spec-decode embed     │ │ │ │ │
  9. GGUF import                   │ │ │ │ │
  10. CUDA YaRN RoPE               │ │ │ │ │
                                   │ │ │ │ │
P2:                                │ │ │ │ │
  11. bf16 activations             │ │ │ │ │
  12. Flash attn training          │ │ │ │ │
  13. Kernel fusion / CUDA graphs  │ │ │ │ │
                                   │ │ │ │ │
P3:                                │ │ │ │ │
  14. Distributed training         │ │ │ │ │
  15. Tensor-parallel inference    │ │ │ │ │
  16. Spot preemption handling ────┘ │ │ │ │
                                     │ │ │ │
                                     ▼ ▼ ▼ ▼
                              ┌──────────────┐
                              │  V4-Flash    │
                              │  on consumer │
                              │  GPU at      │
                              │  20+ tok/s   │
                              └──────────────┘
```

Items 1+3 are the fastest path to a working inference server (days, not weeks).
Items 2+4+5 are the path to V4-Flash on consumer hardware (weeks).
Item 6 is the path to production serving (weeks).
Item 16 is the path to spot-instance viability (days, after 1+6).

---

## What's NOT on the backlog (already done)

- ✅ Training pipeline (pretrain, SFT, DPO, distill, self-distill)
- ✅ 6 attention variants (training/prefill)
- ✅ MoE routing + load balancing
- ✅ Q4/Q8 quantization + GPU inference
- ✅ safetensors I/O (F32/BF16/F16)
- ✅ HF Llama + Qwen3.5 import
- ✅ GGUF export (f32, q8_0, q4_0)
- ✅ 5 optimizers (AdamW, AdamW8bit, Muon, Sophia, Lion)
- ✅ Speculative decoding (main model)
- ✅ Gradient checkpointing
- ✅ 26 CLI subcommands
- ✅ Metal + CUDA backends at 100% parity (269/269)
- ✅ Portable checkpoints (Metal ↔ CUDA)
- ✅ BPE tokenizer (train/encode/decode)
- ✅ Data pipeline (mmap, dedup, mix, pack)
- ✅ Eval harness (perplexity, long-context NIAH/RULER)
- ✅ LoRA (Qwen3.5)
- ✅ Rust embedding API (`Smedjan::load`, `generate`, `generate_streaming`)
