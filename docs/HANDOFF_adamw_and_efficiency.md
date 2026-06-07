# Handoff — AdamW hardening + efficiency roadmap (10× capacity @ ⅒ usage)

Written end of the session that fixed gradient-checkpointing, drained the gap list, and root-caused
the AdamW instability. Tip at write time: `e26330e` on `origin/main` (Forgejo, `localhost:3300`).
All claims below were grounded against the tree, not assumed.

---

## 0. Snapshot — what's already done

- **Gradient checkpointing**: re-enabled; root cause was buffer-pool corruption in `clear_tape` /
  `clear_tape_keep_grads` (recycling buffers still referenced as inputs) + checkpoint cleanup +
  recompute pool aliasing. Fixed; exact grad equivalence test (`gradient_checkpointing_matches_standard`,
  `#[ignore]` — needs `--test-threads=1`, the GPU layer is single-threaded).
- **clippy.toml deleted**: all 31 `too_many_arguments` + the `large_enum_variant` fixed by real
  refactors (dim/param structs, `Box<TrainArgs>`, deleted dead fused/persistent paths). **0 `#[allow]`
  in `src/`.**
- **Gap features shipped + tested**: end-to-end convergence smoke test (uses Muon), `eval::perplexity`
  + CLI, min-p / locally-typical sampling, **bf16 tiled matmul** (`matmul_tiled_bf16`, opt-in),
  batched generation (`generate_batch`, `--batch-file`), GPT-2 `merges.txt` import
  (`BpeTokenizer::import_gpt2_merges`).
- **AdamW instability — FIXED (both root causes)**:
  1. `fce7f8c` — RMSNorm backward `inv_rms^3` exploded (~1e8) on a collapsed activation row → clamp
     `inv_rms ≤ 1/√1e-3` and clamp `grad_input` to ±1e3 (also stops cross-layer compounding). Forward
     untouched → existing checkpoints bit-identical.
  2. `a5cd0eb` — AdamW `eps` 1e-8 → **1e-5**: with `beta2=0.95` the denominator collapsed when
     gradients shrank. Now AdamW goes from oscillating/diverging (spiking ~1e5) to **stable + descending
     below uniform**. Guard: `adamw_training_stays_bounded_no_grad_explosion`.
- **Verified on BOTH M3 (dev) and M1 (mini)**: full suite 130/130 on each; bf16 (`bfloat`) compiles and
  matches on M1. Cross-Apple-Silicon portable.
- **Still blocked**: CUDA backend (22/89 stub, no NVIDIA hardware — the mini is Apple Silicon, not a
  CUDA box); distributed/multi-GPU (needs the network plumbing — the mini is the second device for it).

Build/test: `cargo test --no-default-features --features metal`. Serial GPU tests:
`-- --include-ignored --test-threads=1`. Mini: `ssh mini`, repo at `~/projects/andreai` (has local WIP
on `main` — DON'T touch it; verify in a throwaway worktree: `git worktree add /tmp/x origin/main`).

---

## 1. AdamW — "hammer until perfect"

State: stable + descends, but **still slower than Muon on full-batch overfit** (~0.001 loss/step →
thousands of steps to fully memorize vs Muon's ~150). Swept `beta2∈{0.95,0.999}`, `eps∈{1e-8..1e-4}`,
LR, cosine decay — none closed it. That residual is the **diagonal-vs-matrix preconditioning** gap
(why Muon/Shampoo exist), not a bug. To push it further:

1. **Muon+AdamW hybrid (highest value).** The canonical Muon recipe is Muon for 2-D matrices, AdamW
   for 1-D params (embeddings, norms, biases) and the LM head. The harness has BOTH optimizers
   (`src/optim.rs`) but never combines them per-parameter. This is the standard way to get Muon's
   full-batch speed without its embedding/norm pathologies. Build a `HybridOptimizer { muon, adamw }`
   that routes each `ParamState` by `shape.len()`.
2. **Investigate the `beta2=0.999` anomaly.** It made the loss *worse* (3.75→4.99) where 0.95 didn't —
   counterintuitive. Likely a bias-correction × warmup interaction; instrument `m_hat`, `v_hat`, the
   denominator, and the per-step update norm. May reveal a third subtle bug.
3. **Update-norm clipping** (clip the AdamW *update* `m̂/(√v̂+ε)`, not the gradient). Bounds overshoot
   at the source — more principled than the global grad clip, which lets one exploded component
   dominate the *direction* after normalization.
4. **Per-tensor (not global) gradient clipping.** A single exploded tensor currently corrupts the
   clipped direction for everything. Per-tensor norm clipping preserves the rest.
5. **Root-cause the activation collapse itself** (not just clamp the backward). *Why* does a RMSNorm
   input row go to `mean_sq→0` under AdamW? If it's a dead SwiGLU gate or a zeroed projection, fixing
   the cause removes the need for the clamp.
6. **Verify on real (large-batch) training** that eps=1e-5 + the RMSNorm clamp don't regress loss/throughput
   — they shouldn't (eps negligible when `√v̂≫ε`; clamp only touches degenerate rows), but confirm on a
   real run before declaring victory.
7. **Make `eps`/`beta1`/`beta2` configurable** via `TrainArgs`/CLI (they're `pub` fields but hardcoded
   in `AdamW::new`). Different regimes want different values.
8. **8-bit AdamW states** (see §2.2) — folds into the efficiency goal.
9. Optional optimizers worth comparing on this harness: Adafactor (memory), AdamW-mini, SOAP, Lion
   (have it), gradient centralization, Sophia (have it).

---

## 2. Efficiency roadmap — 10× capacity @ ⅒ usage

Prioritized by impact on the goal. "Capacity" = model size / context; "usage" = compute / memory /
energy. Grounded absences are flagged **[MISSING]**.

### TIER 1 — biggest wins

1. **simdgroup_matrix matmul [MISSING — the single biggest training speedup].**
   `src/metal/shaders.rs` matmuls are 100% hand-rolled tiled kernels (74 manual-MAC markers, 0
   `simdgroup_matrix`). M-series GPUs have hardware matrix units reachable from MSL via
   `simdgroup_matrix<half,8,8>` + `simdgroup_multiply_accumulate` (Metal 3+, works on M1→M4, all
   verified-present here). Rewriting `matmul_tiled` / `batched_matmul_tiled` / flash-attention inner
   loops on these can give **2–8× matmul throughput** — i.e. most of the "⅒ usage" for training in one
   change. Keep the hand-rolled path as a fallback for odd shapes. Also evaluate **MPSGraph** for
   matmul/attention as an alternative to hand-rolling.
2. **8-bit optimizer states [MISSING].** AdamW `m`/`v` are fp32 (`src/optim.rs`). Block-wise int8
   quantization (bitsandbytes-style) → **4× less optimizer memory**. Combined with the existing
   **GaLore** (`galore_rank`) and **Muon** (no `v` at all), this is the direct lever for fitting much
   bigger models on the same RAM → "10× capacity."
3. **MLA — Multi-head Latent Attention [MISSING; only GQA present].** Compress K/V into a low-rank
   latent (DeepSeek-V2/V3): **10–50× KV-cache shrink** → 10× longer context and ⅒ inference memory. Add
   as a new `AttnKind` (the enum already exists from the linear/ssm/rwkv work) so it slots into the
   hybrid-topology config.
4. **Make bf16 the training matmul default.** The `bf16` kernel is shipped but opt-in; training still
   uses the fp16 path that clamps at ±65504. Routing training matmuls through `matmul_tiled_bf16` gives
   fp32 range + fp16-ish bandwidth → removes a whole class of overflow instability (related to the
   AdamW work) at ~no cost.
5. **Sequence packing / varlen [MISSING].** Pack multiple short sequences into one row with a
   block-diagonal mask (cu_seqlens style) instead of padding to max length. For mixed-length SFT/pretrain
   data this is a large, free throughput win (no wasted compute on pad tokens).

### TIER 2

6. **KV-cache quantization (int8/int4) [MISSING].** fp32 KV cache today. Quantizing it → 4× longer
   context / less memory, complementary to MLA.
7. **Distributed data-parallel (use the mini).** Single-device today. Two Apple-Silicon boxes (M3 + M1
   mini) over the network with gradient all-reduce → ~2× compute and the path to N-node. This is the
   "multi-GPU" gap, now *verifiable* because there's a second device.
8. **Continuous batching + paged attention.** Inference is single-sequence (`generate`) / equal-length
   batch (`generate_batch`). vLLM-style paged KV + continuous batching → much higher serving throughput.
9. **Kernel fusion with simdgroup_matrix.** Metal dispatch is ~300 µs/kernel; the harness batches command
   buffers but per-op kernels remain. The *dead* fused/persistent paths were deleted this session —
   rebuild them as *working* fused (norm+matmul, attention, SwiGLU-FFN) on top of simdgroup_matrix.
10. **FlashAttention-2/3 partitioning.** `flash_attention` exists; re-do its work partitioning + use
    hardware matrix units; consider FP8 paths for future hardware.

### TIER 3

11. **Mixture-of-Depths** — dynamic per-token layer skipping (capacity without proportional compute).
12. **Stochastic rounding** for bf16/fp8 training (preserves sub-ULP updates — matters for the small
    gradients that drove the AdamW instability).
13. **bf16 activations + KV** in the forward (memory) once bf16 is the matmul default.
14. **GPTQ/AWQ inference quantization** (have gguf export + basic `quantize.rs`; add calibrated 4-bit).
15. **Speculative decoding upgrades** (Medusa/EAGLE heads; have a basic draft-model path).
16. **Dropless / expert-choice MoE routing** (MoE present via `n_experts`).
17. **Multi-token-prediction tuning** (present via `n_predict`).

### What's already strong (don't rebuild)
GQA, RoPE+NTK scaling, SwiGLU, sliding-window attn, FlashAttention, **linear/SSM/RWKV O(N) mixers**
(huge for long context), **MoE**, **BitNet 1.58-bit** (massive capacity/usage win), **Muon/Sophia/Lion**,
**GaLore**, WSD/cosine schedules, SFT/DPO/distillation, gradient checkpointing (fixed), gguf export,
speculative decoding, zero-copy unified-memory tensors, well-tuned release profile (LTO, codegen-units=1).

---

## 3. The goal, mapped

- **10× capacity**: MoE (have) + MLA (KV compression) + BitNet (have) + 8-bit-optimizer/GaLore (fit
  bigger) + linear/SSM/RWKV (have, O(N) context).
- **⅒ usage**: **simdgroup_matrix** (compute) + bf16 default (precision/memory) + KV quant + sequence
  packing + 8-bit optimizer + quantized inference (have basics) + distributed (amortize).

Suggested order to maximize ROI: **(1) simdgroup_matrix** → **(2) 8-bit optimizer** → **(4) bf16 default**
→ **(3) MLA** → **(5) sequence packing**, interleaving the **Muon+AdamW hybrid** for the AdamW thread.

---

## 4. DELIVERED (session after the handoff — all tested, clippy-clean, on `origin/main`)

The entire suggested ROI sequence above, plus the bulk of the AdamW thread. Full suite **145 passed /
0 failed / 4 ignored** (the 4 ignored = long multi-step GPU trajectories that need
`--test-threads=1`; the GPU layer is single-threaded by design). Build is warning-free with **0
`#[allow]`** kept — test-only surfaces are exercised by `api::gpu_diagnostic` at runtime.

**AdamW thread**
- **Muon+AdamW hybrid** (`HybridOptimizer`, `Optimizer::Hybrid`, `--optimizer hybrid`): routes by
  ROLE via `Transformer::force_adamw_param_ids()` — hidden 2-D matrices → Muon, embeddings/tied
  head/MoE routers/1-D norms → hardened AdamW. Per-group LR scales (`--muon-lr-scale`,
  `--adamw-lr-scale`). Fixed a **latent bug** in Muon's own AdamW fallback (it hardcoded eps=1e-8 —
  the denominator-collapse bug — and decayed norm weights; now uses `AdamWHyper`/`no_decay`).
- **Configurable eps/beta1/beta2** via `AdamWHyper` + `TrainConfig` + CLI (defaults unchanged).
- **Update-norm clipping** (`update_clip`): per-element clamp on the normalized update in the kernel +
  `step_cpu`, bounds overshoot at the source. **Per-tensor grad clip** (`--per-tensor-clip`).
- **beta2=0.999 anomaly — INVESTIGATED, not a bug.** Grounded by `beta2_high_overshoots_but_is_not_a_bug`:
  under a 100× gradient jump, 0.999's slow `v` lags → under-sized denom → it overshoots harder than
  0.95 (jump Δ 0.0052 vs 0.0044), but both stay finite + bounded. It's the diagonal-second-moment
  non-stationarity tradeoff (why warmup + beta2=0.95 are the defaults), not a third bug.

**Efficiency thread**
- **simdgroup_matrix MMA** (`matmul_simdgroup`, `matmul_simdgroup_f16`): hardware matrix units. The
  naive 32×32/1-simdgroup port measured **0.98×** (occupancy-bound, grounded — not assumed); the
  64×64/4-simdgroup version measures **1.29× at 1024³** (964→1247 GFLOP/s on M3), bit-identical to the
  hand-rolled fp16. Opt-in (`--simdgroup-matmul`). Below the theoretical 2–8× — double-buffering /
  larger tiles are the next lever.
- **8-bit optimizer** (`AdamW8bit`, `--optimizer adamw-8bit`): block-wise int8 moments, **3.94×**
  optimizer-memory reduction. KEY fix (caught by the fidelity test): linear-quant of `v` underflows
  (v=g² spans ~1000× per block → small entries → 0 → v̂≈0 → update explodes); store int8 of **√v**
  instead → tracks fp32 to max_diff **4e-4**.
- **bf16 default-matmul option** (`--bf16-matmul`): fp32 range, no fp16 ±65504 clamp (preserves 1e5).
- **MLA** (`AttnKind::Mla`, `--mla-latent-dim`, `src/mla.rs`): K/V from a shared low-rank latent →
  **16×** KV-cache shrink measured; checkpoint format **v9** persists `mla_latent_dim`.
- **Sequence packing** (`datapipe::pack_sequences`, `Tensor::causal_doc_mask`): block-diagonal mask;
  packed attention proven **bit-identical** (max_diff 0.0) to per-sequence — zero leakage.

**Still genuinely open (need resources beyond unit tests, not code-blocked):**
- **#5 root-cause the RMSNorm activation collapse itself** (the clamp treats the symptom). Needs
  per-layer activation-norm instrumentation across a real diverging run to find the dead gate /
  zeroed projection.
- **#6 verify the AdamW hardening + the new opt-in paths (simdgroup/bf16/8-bit/MLA/packing) on a real
  large-batch multi-hour run** — unit tests prove correctness + boundedness; end-to-end loss/throughput
  on real data is an operator-run.
- simdgroup beyond 1.29× (double-buffer/larger tiles); MLA decoupled-RoPE keys + KV-cache-compressed
  incremental decode; wiring `pack_sequences` into the training `DataLoader`.
