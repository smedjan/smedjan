# CUDA backend — scaffold & rented-box playbook

The Metal backend is the reference and is complete + verified; this doc is the turnkey plan for
CUDA runtime proof on a rented NVIDIA box. CUDA compile parity is now checked locally with
`cargo check --no-default-features --features cuda`, but CUDA execution remains **UNVERIFIED until run
on an NVIDIA GPU**. Do runtime/kernel work on the box, where the compiler, memcheck, and a GPU verify
each step.

## Current state (grounded)

- `src/cuda/mod.rs` — `MetalContext` alias over `CudaDevice`, `GpuBuffer = CudaSlice<f32>`, nvrtc PTX
  compile of `kernels::ALL_KERNELS`. Buffer pool is a no-op (cudarc owns allocation). OK as a base.
- `src/cuda/compute.rs` — shared backend surface compiles, including batched matmul, GQA composition,
  backward wrappers, optimizer wrappers, and advanced-path wrappers. No active `unimplemented!`
  wrappers remain.
- `src/cuda/kernels.rs` — CUDA kernel source is compiled by nvrtc when `CudaContext` starts; real
  correctness still needs NVIDIA runtime execution plus `compute-sanitizer`.

### Architectural blockers to clear FIRST (before any kernel work)

1. **Runtime hardware is the blocker.** Mac compile parity is not CUDA correctness. Use a rented
   NVIDIA box and run `compute-sanitizer` on at least a one-step train plus a finite-loss training
   smoke before treating CUDA as production-ready.
2. **The `simdgroup` MMA path is Metal-only.** CUDA aliases the simdgroup entry points to its tiled
   kernels for API parity. CUDA's eventual fast GEMM should be cuBLAS or a CUDA-native tiled kernel,
   not a Metal simdgroup port.

## Runtime proof checklist

The old missing-wrapper work-list is closed: the shared CUDA backend surface compiles and has no
active `unimplemented!` wrappers. The remaining production question is runtime proof on NVIDIA
hardware. Prove these paths on the box before calling CUDA production-ready:

### Tier A — minimal dense LM training
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

### Tier B — advanced paths
- Flash attention: `gpu_flash_attention_forward/backward`, `gpu_flash_attn_precompute_d`
- Block-sparse: `gpu_gather_blocks(_backward)`, `gpu_block_*`, masks
- MoE: `gpu_moe_gather`, `gpu_moe_scatter_add`
- BitNet: `gpu_ternary_*`
- Alt optimizers: `gpu_sophia_update`, `gpu_lion_update`, `gpu_adamw_8bit_update`, `gpu_muon_frob_normalize`, `gpu_ema_update`, `gpu_inv_sqrt_bc`
- Misc: `gpu_argmax`, `gpu_temperature_scale`, `gpu_kl_divergence`, `gpu_logsumexp`, etc.

Each path needs at least one CPU/Metal-reference unit test, one real CUDA execution, and one
`compute-sanitizer` pass that includes forward + backward + optimizer.

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

# 2. First build — should compile. If it does not, fix compile parity before runtime work.
cd ~/andreai
cargo build --release --no-default-features --features cuda 2>&1 | tail -40

# 3. Runtime proof: unit-test vs CPU/Metal reference, then grad-check
cargo test --release --no-default-features --features cuda <name> -- --nocapture

# 4. Training smoke
./target/release/andreai train --size tiny --steps 50 ...   # loss finite + decreasing

# 5. Pull results back; STOP the instance when idle (per-minute billing)
rsync -az --exclude target user@BOX:~/andreai/src/cuda/ ~/projects/andreai/src/cuda/
```

Cost for finishing Tier A + smoke: a few GPU-hours over ~2 days ≈ **$5–20** on a 4090. Don't rent an
H100 for this — save it for actual scale training once the backend is green.

## Definition of done (this scaffold's target)

`cargo build --no-default-features --features cuda` green; Tier-A and enabled Tier-B kernels each
CPU/Metal-match + grad-check green; `compute-sanitizer` reports 0 errors for a one-step train; a
`train --size tiny --steps 50` smoke on the GPU shows finite, decreasing loss tracking the Metal run.
Then the Mac stops being the ceiling and fp8/fp4 + multi-GPU become the next campaign.

---

## Historical runtime bring-up progress (2026-06-15, overnight)

At this point in history the CUDA backend compiled and started running. nvrtc compiled all kernels for sm_120 on the
RTX 5090 (CUDA 12.8); the model initialises; the forward executes through embedding + linear matmuls
+ batched attention matmuls, then stopped at the first not-yet-wired kernel. Metal green throughout
(verified each step). Commits: `52322bd` (compiles), `149b8b1` (matmul batch + fp32 path).

**Design decisions taken:**
- CUDA stays **fp32** for now: `cast_to_f16` is a no-op (shares the fp32 buffer); the f16 matmul
  entry points delegate to the fp32 kernels. (f16 perf path is a later optimisation.)
- Typed-buffer abstraction: `Buf`=Arc<CudaSlice<f32>>, `BufU32`=Arc<CudaSlice<u32>>, with
  `u32_to_buf`/`buf_as_u32` transmute helpers so the untyped Metal tape (`Vec<Buf>`) can hold u32
  token/sel buffers. `buf_write_bytes`/`buf_bytes` do dtoh/htod for the Metal unified-memory paths.
- Metal-only CPU-pointer paths (`step_cpu`, opt-state load, 8bit init, `api` harness) are cfg-gated.

Current state has moved past this wiring loop. Use the runtime proof checklist above, not the old
stub-replacement loop. Build env on the NVIDIA box remains:
`CUDA_PATH=/usr/local/cuda-12.8 LD_LIBRARY_PATH=$CUDA_PATH/lib64 cargo build --release --no-default-features --features cuda`.

---

## TRAINING BRING-UP (2026-06-15, cont.) — full forward+backward path memcheck-clean

All Tier-A kernels wired (one `python3` batch over the stubs + composed fused ones); `bench --size
tiny` runs every section (forward 45k tok/s, decode, **Training Forward+Backward 13.3k tok/s** vs Metal
M1 3.3k, checkpointed, roofline) with exit 0. Then drove an actual `train` to expose what the bench's
non-asserting run hid:

**Bugs found + fixed (each verified, Metal kept green):**
- **`gpu_scaled_causal_softmax` OOB (false-green).** `SoftmaxDims.total_rows` is ALREADY
  `batch_heads*seq_q` (see `Tensor::scaled_causal_softmax`); the composed wrapper multiplied by `seq_q`
  AGAIN for the copy/scale size, passed `total_rows` (not `total_rows/seq_q`) as causal_mask
  batch_heads, and `total_rows*seq_q` as softmax rows. Overran the `[total_rows,seq_k]` score buffer by
  ~seq_q×. At bench dims it silently overran into valid pool memory (so "forward ran" was a LIE — it
  was corrupting); at train dims it faulted `CUDA_ERROR_ILLEGAL_ADDRESS`. Fix: `n=total_rows*seq_k`,
  `causal_mask(total_rows/seq_q, seq_q, seq_k)`, `softmax rows=total_rows`. **Lesson: "kernel ran" ≠
  "kernel correct" — sanitize, don't trust a clean exit.**
- **Raw `result::memcpy_htod_sync` needs the ctx bound.** `write_u32_to_buffer` / `buf_write_bytes`
  (the in-place htod for the loss workspace + checkpoint writes) used the raw driver memcpy, which —
  unlike the safe `device.htod_*` wrappers — does not make the primary context current on the thread.
  Added `buf.device().bind_to_thread()` before the copy + a sized assert.
- **Bench KV-cache overflow (harness, both backends).** Decode hardcoded a 16-token prefill but cache
  capacity == `config.max_seq_len` (= `--seq-len`). Sized the prefill to leave room for warmup+iters.
- **`gpu_mega_ffn` is Metal-only.** Gated `use_mega` on `cfg!(feature="metal")`; CUDA decode falls
  back to the primitive `rms_norm_residual`+`swiglu_ffn` path.

**Diagnostic recipe (worked):** `CUDA_LAUNCH_BLOCKING=1` to localize, then nvrtc
`--generate-line-info` + `compute-sanitizer --tool memcheck` to NAME the faulting kernel + caller +
the exact OOB address. **`compute-sanitizer` on a full 1-step train (fwd+bwd+loss+clip+optimizer):
0 errors.** Run detached on the box (`nohup ... > log &`) — it's 50–100× slower and an idle SSH will
time out mid-run; poll the log. Keep a persistent SSH master (`ssh -fNM -o ServerAliveInterval=30`).

**Status: training is memcheck-clean; 50-step `train` loss-curve verification in progress.**
