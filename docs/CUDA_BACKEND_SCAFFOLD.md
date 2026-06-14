# CUDA backend — scaffold & rented-box playbook

Status as of 2026-06-14 (`origin/main` 1c2a2e7). The Metal backend is the reference and is
complete + verified; this doc is the turnkey plan to bring the **CUDA** backend to training-parity
on a rented NVIDIA box. Everything CUDA here is **UNVERIFIED until built on an NVIDIA GPU** — the Mac
has no CUDA toolkit, so `cargo build --features cuda` cannot even compile-check locally. Do the kernel
work on the box, where the compiler + a GPU verify each step. This doc is the checklist, not the code.

## Current state (grounded)

- `src/cuda/mod.rs` — `MetalContext` alias over `CudaDevice`, `GpuBuffer = CudaSlice<f32>`, nvrtc PTX
  compile of `kernels::ALL_KERNELS`. Buffer pool is a no-op (cudarc owns allocation). OK as a base.
- `src/cuda/compute.rs` — **22 of the 106** `gpu_*` functions the shared code calls. Forward-partial:
  has `matmul` (+ `trans_a`/`trans_b`), `softmax`, `rms_norm`, add/mul/scale/fill/copy, `silu_gate`,
  `rope`, `cross_entropy`, `reduce_sum`, `adamw_update`, `embedding_lookup`, casts, `causal_mask`.
- `src/cuda/kernels.rs:554` — `// Backward kernels (stubs — to be completed)`.
- **No batched matmul** in CUDA at all → attention can't run (forward OR backward).

### Architectural blockers to clear FIRST (before any kernel work)

1. **`main()` is Metal-hardcoded** — `let ctx = metal::MetalContext::new()` and
   `metal::compute::set_simdgroup_matmul(true)` are unconditional. A `--features cuda --no-default-features`
   build won't compile `main.rs`. Fix: `#[cfg]`-alias the backend module (e.g. `use crate::metal as gpu`
   / `use crate::cuda as gpu`) and route `main()` through it; OR drive CUDA bring-up through a
   backend-generic test/bench binary first and defer `main()`.
2. **The `simdgroup` MMA path is Metal-only.** `Tensor::matmul` routes to `gpu_matmul_simdgroup_f16`
   when `simdgroup_matmul_enabled()` (now default-on). CUDA has no simdgroup kernels. For CUDA either
   make `cuda::compute::set_simdgroup_matmul` a no-op that keeps the flag **false**, or `#[cfg]` the
   simdgroup branch out of `Tensor::matmul`. CUDA's fast GEMM should be cuBLAS or a tiled kernel, not a
   simdgroup port. Do NOT stub `gpu_*_simdgroup` for CUDA — they shouldn't be reached.

## The gap: functions the BACKWARD calls but CUDA lacks (the training work-list)

Required by `src/autograd.rs` (backward) and missing from `cuda/compute.rs`, grouped by the minimal
dense-transformer training path vs. advanced paths. Each needs the `compute.rs` wrapper + the CUDA C
kernel in `kernels.rs` + registration in `KERNEL_NAMES`.

### Tier A — minimal dense LM training (do these first, in order)
- `gpu_rms_norm_backward`
- `gpu_silu_gate_backward`, `gpu_silu_backward`
- `gpu_softmax_backward`
- `gpu_batched_matmul`, `gpu_batched_matmul_trans_a`, `gpu_batched_matmul_trans_b`  ← attention (fwd reuse + bwd)
- `gpu_embedding_backward`
- `gpu_transpose_perm_forward`, `gpu_transpose_perm_backward`  ← QKV head reshape
- `gpu_rope_backward_copy`, `gpu_transpose_rope_backward`
- `gpu_scale_copy`, `gpu_scale_rows`, `gpu_row_dot_reduce`, `gpu_concat_cols`  ← used in norm/attn backward
- `gpu_relu_backward`
- forward gaps the above depend on: `gpu_rms_norm_residual`, `gpu_silu`, `gpu_broadcast_rows`,
  `gpu_transpose_2d`, `gpu_scaled_causal_softmax`, `gpu_repeat_kv`(+`_backward` for GQA)

### Tier B — defer (not needed for a first dense-LM smoke)
- Flash attention: `gpu_flash_attention_forward/backward`, `gpu_flash_attn_precompute_d` (use the
  standard scores→softmax→context path on CUDA first; gate flash off)
- Block-sparse: `gpu_gather_blocks(_backward)`, `gpu_block_*`, masks
- MoE: `gpu_moe_gather`, `gpu_moe_scatter_add`
- BitNet: `gpu_ternary_*`
- Alt optimizers: `gpu_sophia_update`, `gpu_lion_update`, `gpu_adamw_8bit_update`, `gpu_muon_frob_normalize`, `gpu_ema_update`, `gpu_inv_sqrt_bc`
- Misc: `gpu_argmax`, `gpu_temperature_scale`, `gpu_kl_divergence`, `gpu_logsumexp`, etc.

The forward matmul backward (`backward_matmul`) already has its CUDA deps (`gpu_matmul`,
`gpu_matmul_trans_a/_b` exist) — so the **linear-layer backward should work on CUDA once it compiles**;
the missing pieces above are norm/activation/attention/embedding backward.

## Verification protocol (per kernel — mirror the Metal tests)

For each new CUDA kernel, prove it before moving on:
1. **CPU-reference unit test** — random inputs, compute the op in plain Rust on CPU, assert the CUDA
   kernel matches within tol. (Mirror `matmul_simdgroup_trans_a_matches_scalar` style, but vs CPU since
   there's no Metal on the box.)
2. **Grad-check** — finite-difference vs analytic for the backward ops.
3. **Training smoke** — `bench`/a tiny `train` run: loss finite + decreasing for ~50 steps.

Gate: a CUDA training step's loss must track the Metal training step's loss on the same seed/config.

## Rented-box workflow (RunPod RTX 4090, ~$0.34/hr, per-minute billing)

```bash
# 0. On the Mac: push latest to origin (already done), then rsync the tree up
#    (origin is the localhost:3300 Forgejo tunnel — not reachable from the box, so rsync).
rsync -az --exclude target --exclude .git ~/projects/andreai/  user@BOX:~/andreai/

# 1. On the box: toolchain (most CUDA images have nvcc; add rust)
nvidia-smi                          # confirm GPU + driver
curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"

# 2. First build — expect compile errors (missing cuda::compute fns + main() metal-hardcoding).
#    Clear the architectural blockers above, then iterate kernel-by-kernel.
cd ~/andreai
cargo build --release --no-default-features --features cuda 2>&1 | tail -40

# 3. Per-kernel: implement → unit-test vs CPU → grad-check
cargo test --release --no-default-features --features cuda <name> -- --nocapture

# 4. Training smoke once Tier A is in
./target/release/andreai train --size tiny --steps 50 ...   # loss finite + decreasing

# 5. Pull results back; STOP the instance when idle (per-minute billing)
rsync -az --exclude target user@BOX:~/andreai/src/cuda/ ~/projects/andreai/src/cuda/
```

Cost for finishing Tier A + smoke: a few GPU-hours over ~2 days ≈ **$5–20** on a 4090. Don't rent an
H100 for this — save it for actual scale training once the backend is green.

## Definition of done (this scaffold's target)

`cargo build --no-default-features --features cuda` green; Tier-A kernels each CPU-match + grad-check
green; a `train --size tiny` smoke on the GPU shows finite, decreasing loss tracking the Metal run.
Then the Mac stops being the ceiling and fp8/fp4 + multi-GPU become the next (separate) campaign.

---

## RUNTIME BRING-UP PROGRESS (2026-06-15, overnight)

**Status: CUDA backend COMPILES (0 errors) and RUNS.** nvrtc compiles all kernels for sm_120 on the
RTX 5090 (CUDA 12.8); the model initialises; the forward executes through embedding + linear matmuls
+ batched attention matmuls, panicking at the first not-yet-wired kernel. Metal green throughout
(verified each step). Commits: `52322bd` (compiles), `149b8b1` (matmul batch + fp32 path).

**Design decisions taken:**
- CUDA stays **fp32** for now: `cast_to_f16` is a no-op (shares the fp32 buffer); the f16 matmul
  entry points delegate to the fp32 kernels. (f16 perf path is a later optimisation.)
- Typed-buffer abstraction: `Buf`=Arc<CudaSlice<f32>>, `BufU32`=Arc<CudaSlice<u32>>, with
  `u32_to_buf`/`buf_as_u32` transmute helpers so the untyped Metal tape (`Vec<Buf>`) can hold u32
  token/sel buffers. `buf_write_bytes`/`buf_bytes` do dtoh/htod for the Metal unified-memory paths.
- Metal-only CPU-pointer paths (`step_cpu`, opt-state load, 8bit init, `api` harness) are cfg-gated.

**Remaining = kernel wiring (mechanical), then port the fused ones:**
Most kernels already exist in `ALL_KERNELS` + `KERNEL_NAMES`; the `cuda/compute.rs` stub just needs a
launch (mirror an existing wrapper; grid from the kernel's blockIdx convention). Compose the fused:
- `transpose_rope` = `transpose_perm_forward` then `rope`; `transpose_rope_backward` = `rope_backward`
  then `transpose_perm_backward`. (Both component kernels exist.)
- Wire stub→existing kernel: `transpose_perm_forward`/`_backward`, `rope_copy`/`_backward_copy`,
  `rms_norm_residual`, `scaled_causal_softmax`(+`_window`), `silu`/`silu_backward`/`silu_gate_backward`,
  `rms_norm_backward`, `softmax_backward`, `embedding_backward`, `scale_rows`, `row_dot_reduce`,
  `broadcast_rows`, `slice_cols`, `concat_cols`, `repeat_kv`(+`_backward`), `transpose_2d`, `zero_rows`.
- DEFER (no CUDA kernel; advanced, off the minimal dense-LM path): flash-attention(±), block-sparse
  gather/mask, MoE gather/scatter, ternary/BitNet, sophia/lion/8bit, muon_frob.

**Resume loop:** replace next `unimplemented!` stub → push → `bench --size tiny` → fix next panic →
repeat until forward green, then backward, then `train --size tiny --steps 50` smoke (loss finite +
decreasing). Build env: `CUDA_PATH=/usr/local/cuda-12.8 LD_LIBRARY_PATH=$CUDA_PATH/lib64`,
`cargo build --release --no-default-features --features cuda`.
