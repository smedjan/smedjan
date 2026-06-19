# Handoff — buffer-hazard root cause, loss readout + batch-LR transfer (landed), seq-packing & sparse-backward (specced)

Written in a Linux sandbox with **no Metal and no Rust toolchain** — none of the code below was
compiled or run here. Every change is grounded against the actual tree (read, not assumed) and mirrors
existing patterns, but **verification is yours on the Mac** (M1 mini / M3). Build/test commands and the
real-run checks are in §5. Tip at write time: `768e6d3` on `origin/main`.

---

## 0. What landed (real edits, need Mac verify)

### A. Hazard-aware loss readout — `src/train.rs` (FIXES the large-batch "constant 1.0")

Root-caused the revert. The reverted fix (`7255b2b`, reverted in `48c4e05`) was correct in *shape*
— copy the loss scalar into a persistent buffer before `clear_tape_keep_grads` frees it — but it
allocated that buffer with **`ctx.alloc_buffer(4)`, i.e. from the pool**. At large batch the pool
handed back a 4-byte buffer that was **still logically live** (a gradient-norm / clip scalar encoded
earlier in the same *uncommitted* command batch). The `gpu_copy` then clobbered that scalar →
corrupted gradient clipping → divergence (EMA 1.56 → 388). **The bug was never the copy; it was
pulling the destination from the pool.**

Fix: allocate the readout buffer **once, outside the pool**, via `ctx.buffer_from_slice(&[0.0f32])`
— a direct (unpooled) Metal allocation that is never recycled and never handed to any other tensor,
so it cannot alias anything. Three edits:
- declare `let loss_readout = ctx.buffer_from_slice(&[0.0f32]);` once before the `for step` loop;
- `compute::gpu_copy(ctx, &loss_tensor.buffer, &loss_readout, 1);` **BEFORE `autograd::backward`**;
- read the log-time loss from `loss_readout` instead of `last_loss_tensor.to_vec()`.

> **Correction (verified on M3, 2026-06-13).** An unpooled *destination* is necessary but NOT
> sufficient. The earlier wording placed the copy *after* `backward` ("so it executes first") — that is
> wrong: encoded dispatches run in **encoding order**, so a copy encoded after `backward` runs after it.
> The hazard is on the **source**: `backward` recycles 4-byte pooled buffers for its own scalars
> (including the `dL/dL = 1.0` seed) and re-hands `loss_tensor.buffer` to one of them, overwriting it
> before the post-backward copy executes. The result is the displayed loss reading **exactly `1.0`**
> (the seed) once the pool warms up — masking the *true* loss entirely (the model is actually training
> fine; only the readout, and the "EMA ~1.56" derived from it, were garbage). The "1.0" is therefore a
> **readout artifact, not divergence** — confirmed by capturing logits + `loss_tensor` behind an extra
> flush, which read a correct, descending ~9.x. Moving the copy to *before* `backward` captures the
> true value while `loss_tensor.buffer` is still live. With the honest readout, a tiny model on
> `train_v3.bin` descends 9.5 → 9.08 over 2000 steps (perplexity 7572 → 4574); too-hot configs
> (seq256/lr3e-3) now visibly *climb* the loss instead of hiding behind a constant 1.0.

Why this is the general lesson (the **buffer-hazard root cause**, task #1): the thread-local buffer
pool reissues a buffer the instant it is recycled, but `dispatch_kernel` only *encodes* into the
`ACTIVE_BATCH`; encoded dispatches don't run until `flush_batch`/`auto_flush_batch` commits+waits.
So any buffer that is recycled — or whose value is still needed later in the same uncommitted batch —
can be re-handed-out and overwritten *before* the GPU ever runs the dispatch that depended on it.
The pool has no notion of "in-flight within the current batch." **Anything that must survive across a
batch boundary, or be read after the step, belongs in an unpooled buffer** (the loss readout, EMA
state, optimizer state already are). This is the same failure mode behind the seq-packing revert (§2).

### B. Batch-size LR transfer — `src/train.rs` + `src/main.rs` (orthogonal to μP; default OFF)

μP already transfers LR across model **width** (`mup_lr_scale = base_width/d_model`, `mup_output_scale`
on logits). It does **not** transfer across **batch size**, which is the #6 finding ("a batch-16 LR
diverged at batch 32"). Added an opt-in `--lr-ref-batch N` (`TrainConfig::lr_ref_batch`, default `0` =
off, so existing runs are byte-identical). When set, `--lr` is the LR tuned at batch `N` and the
effective LR is scaled to the actual batch by the √batch rule:
`effective_lr = max_lr · μP_scale · sqrt(batch_size / lr_ref_batch)`. Also added a one-line LR-provenance
log so the applied LR is always visible: `LR: max_lr=… × μP(…) × batch√(…) = effective …`.

Two important caveats, both in the code comments:
1. **Direction is empirical.** The √batch rule scales LR *up* with batch (standard for Adam-family).
   But #6 saw the *same* LR diverge at *larger* batch — partly confounded by the broken loss readout
   (the "1.0" was garbage), now fixed, so the real curve is finally observable. **Re-confirm the
   direction on a real run before trusting it.**
2. **Muon is different.** Muon's orthogonalized update has a batch-independent magnitude, so it can
   need the *opposite* move — lower `--muon-lr-scale` as batch rises. Don't apply √batch to the Muon
   group; tune `--muon-lr-scale` down instead.

---

## 1. The command-batch / buffer model (reference for any future GPU work)

`src/metal/mod.rs`:
- `dispatch_kernel` ENCODES into a thread-local `ACTIVE_BATCH` (a single `MTLComputeCommandEncoder`
  on one command buffer). Dispatches execute in encode order (serial encoder → implicit barriers),
  but **only after** `flush_batch`/`auto_flush_batch` does `endEncoding(); commit(); waitUntilCompleted()`.
- `alloc_buffer` pops from a per-size thread-local `BUFFER_POOL` (LRU-ish, capped 64/bucket). `recycle_buffer`
  returns to it immediately. `POOL_BYPASS` (RAII `PoolBypassGuard`) disables pooling during checkpoint
  recompute so recomputed buffers can't alias the outer backward's live buffers.
- **Hazard class:** recycle-then-reuse *within one uncommitted batch*, or reading a pooled buffer whose
  producing dispatch hasn't been flushed. Within a single serial encoder a later write can't precede an
  earlier read of the same buffer (ordering is safe), but a buffer **recycled and re-handed-out** within
  the batch gets a *new logical owner* whose write the old owner never expected.

**Optional systemic hardening (not landed — higher risk, needs real-run memory/throughput check):**
generation-quarantine the pool. Keep a `BATCH_GENERATION` counter bumped on every flush; tag each
recycled buffer with the generation at recycle; `alloc_buffer` may only reissue buffers whose
recycle-generation `< current` (i.e. recycled in an already-flushed batch). This makes intra-batch
reuse impossible by construction, eliminating the whole hazard class — at the cost of disabling
intra-step pooled reuse (the checkpoint-recompute path raised the bucket cap to 64 specifically for
intra-backward reuse, so measure memory before adopting). Recommend leaving the targeted fixes (§0.A,
§2) in place and treating quarantine as a future correctness-vs-memory tradeoff, gated behind a flag.

---

## 2. Seq-packing model integration (SPEC — task #3)

Op-level pieces are done and verified bit-identical (`datapipe::pack_sequences`,
`Tensor::causal_doc_mask(seg_ids, n_heads)` — note it takes `seg_ids` **explicitly**, no thread-local).
The reverted integration passed `seg_ids` to attention via a **thread-local that was cleared before
the deferred batch ran** → exactly the §1 hazard (the seg buffer recycled/cleared while encoded mask
dispatches still referenced it) → divergence.

**Correct fix: thread `seg_ids` through the forward signatures** (no thread-local). The seg buffer is
owned by the caller (train loop) for the whole step and passed by reference, so it outlives the batch.

Concrete plan (all call sites — miss none or it won't compile):
- `MultiHeadAttention::forward` (`src/attention.rs:427`): add a param
  `seg_ids: Option<&Retained<GpuBuffer>>`. Where it currently applies the causal mask to the
  `[bh, seq, seq]` scores, branch: `match seg_ids { Some(s) => scores.causal_doc_mask(s, self.n_heads), None => <existing causal mask> }`. (Only the dense softmax path; MLA/linear/SSM/RWKV/block-sparse
  paths keep their own masking — packed varlen first targets the standard softmax path.)
- `Block::forward` (`src/model.rs:512`): add the same param, forward it to `self.attn.forward(..)`.
- `forward_checkpointed` / `forward_checkpointed_recompute` (`src/model.rs:680,685`): thread it through
  (these re-run Block forward — the seg buffer must reach the recompute, so capture it in the closure /
  pass as arg; do **not** stash it thread-local, that's the bug).
- `Transformer::forward` / `forward_hidden` / `forward_mtp` (`src/model.rs:969,1082,1145`): add
  `seg_ids: Option<&Retained<GpuBuffer>>` and pass to every block.
- Callers: inference (`generate.rs`, `eval.rs`, dpo/sft forwards) pass `None`. The train loop builds the
  packed batch + seg buffer and passes `Some(&seg_buf)`; keep `seg_buf` a normal owned `Retained` in the
  step scope (do **not** recycle it until after `flush_batch`).
- Defaults: keep a thin wrapper `forward(x, kv)` = `forward_seg(x, kv, None)` if you don't want to touch
  every inference call site at once.

Test (serial GPU): pack two short docs into one row, forward with `Some(seg)`, and assert the per-doc
logits are bit-identical to forwarding each doc alone (the op-level `causal_doc_mask` test already
proves zero cross-doc leakage at the op level — this extends it through the model). Then a real
mixed-length SFT smoke: tokens/sec should rise (no pad waste) with loss curve unchanged.

---

## 3. Block-sparse TRAINABLE backward (✅ LANDED 2026-06-14)

**Done.** New `gather_blocks_backward` scatter-add kernel (atomic, exact transpose of `gather_blocks`)
+ `compute::gpu_gather_blocks_backward` + `Op::GatherBlocks` (stores `sel` in `TapeEntry.cached`) +
`backward_gather_blocks`. `block_sparse_gather_attention` now keeps only the routing/`sel` under
`no_grad` and records the gathers + attention math, so gradients flow q→scores, v_sel→out, and
ksel/vsel→(scatter-add)→k/v. Verified by `gradcheck_block_sparse_gather_attention` (finite-diff,
all-blocks config), `block_sparse_gather_grad_matches_dense_when_full` (q/k/v grads == dense when
top_k+1≥nb), and `gather_blocks_backward_scatter_add_direct` (hand-computed kernel case:
accumulation + sentinel skip).

**Latent bug this surfaced (FIXED):** `gpu_batched_matmul_trans_a` (the `dB = Aᵀ@dC` half of every
batched-matmul backward) declared its param struct `{ m, n, k, batch }` while the MSL kernel reads
`{ M, K, N, batch }` — K/N swapped. Silently correct only when K==N (all square attention scores, and
the one square forward test), it corrupted A/B/C strides for **non-square** output. The block-sparse
gather's `block×sel_w` scores were the first non-square caller. Fix = align the struct to the f16
sibling's `{ m, k, n, batch }`. Guards added: `gradcheck_batched_matmul_nonsquare`,
`gradcheck_batched_matmul_trans_b_nonsquare`. This is exactly the bleeding class Phase B's grad-check
harness was built to catch — "only a real (non-square) run catches it" → CI catches it.

Wiring the (now-trainable) subquadratic gather path into the model as a selectable mixer — replacing
the O(n²) `AttnKind::BlockSparse` mask path — is a **throughput** change (task #12); it needs a
convergence run, not just gradient correctness, so it's deliberately left out of this task.

---

### Original spec (for reference)

`block_sparse_gather_attention` (`src/attention.rs:47`) currently wraps the **entire** body in
`autograd::no_grad(|| …)`, so nothing is recorded → forward/inference only. Two facts make the
trainable path small:
1. The routing (`block_mean_keys` → CPU top-k → `sel`) is **non-differentiable** and should stay that
   way (straight-through, exactly like MoE top-k). Keep it in `no_grad`; `sel` is a fixed permutation
   for the backward.
2. Everything *after* the gather — `reshape`, `batched_matmul_trans_b` (scores), `scale`, the causal
   mask (gradient just passes masked), `softmax`, `batched_matmul` with `v_sel` — are **standard tape
   ops that already have backward implementations**. They only need to be *recorded* (run outside
   `no_grad`).

So the only genuinely new kernel is the **transpose of `gather_blocks`: a scatter-add**. `gather_blocks`
maps source K/V block `sel[bh,qb,slot]` → compact `ksel[bh,qb,slot·block+pos,hd]`. Its backward maps
`dKsel` back: `dK[bh, sel[bh,qb,slot]·block + pos, hd] += dKsel[bh,qb, slot·block+pos, hd]`. Because
multiple query-blocks select the same key-block, the scatter **must accumulate** (atomic add, or a
serialized per-source-block reduction). Sentinel slots (`sel == nb`, padding) scatter nowhere — skip.

Plan:
- New MSL kernel `gather_blocks_backward` (scatter-add) + `compute::gpu_gather_blocks_backward`, mirroring
  the `GatherDims` of the forward. Prefer `atomic_fetch_add_explicit` on a `device atomic_float*` dK/dV,
  or a two-pass (zero dK, then per-(bh,pos,hd) gather the contributing slots) to avoid atomics if float
  atomics are awkward on the target GPU family.
- New autograd `Op::GatherBlocks` (+ a tape entry storing `sel_buf`, dims) whose backward calls the new
  kernel for dK and dV. The gather of K and V are two separate ops → two tape entries (or one entry with
  both buffers).
- In `block_sparse_gather_attention`: keep routing + `sel` in `no_grad`; pull the gather + attention math
  **out** of `no_grad`; record the gathers as `Op::GatherBlocks`. The downstream matmul/softmax ops record
  themselves as usual. Gradient then flows q→scores, v_sel→out, and ksel/vsel→(scatter)→k/v.
- `block_mean_keys` stays in `no_grad` (routing only) — straight-through, no gradient to the means.

Test (numerical): finite-difference check on a tiny config (bh=2, seq=8, block=2, top_k=1, hd=4) — perturb
each of q/k/v, compare analytic vs numerical grad to ~1e-2 (fp16). Plus the existing equivalence guard:
when `top_k+1 ≥ nb` the sparse path must match dense attention's gradients (all blocks selectable).

---

## 4. Why these weren't blind-edited

#2 and #4 are local, low-risk, mirror existing patterns, and default to no behavior change where
relevant — landed. #3 and #5 are multi-file signature changes and a new GPU kernel + autograd op; the
prior #3 attempt diverged, and the 156-test unit suite provably does **not** catch this class
(handoff #6). Landing them unrun in a Metal-less sandbox would risk a broken tree and false "done"
claims. They're specced to the exact call sites/kernels so they land fast on the Mac.

---

## 5. Verification protocol (run on the Mac — this is the actual gate)

Build + unit suite (both machines, per the existing handoff convention):
```
cargo fmt --check
git diff --check
cargo test --no-default-features --features metal -- --test-threads=1
cargo clippy --no-default-features --features metal,bufsan --all-targets -- -D warnings
cargo check --no-default-features --features cuda
./scripts/train-smoke.sh
```

The remaining `#[ignore]` tests are performance benchmarks only. Run them manually on a quiet machine
with `cargo test --release --no-default-features --features metal bench_ -- --ignored --nocapture`;
they are not a correctness CI gate.

(A) **Loss readout** — the real regression test, since unit tests don't catch it:
```
andreai train --size tiny --batch-size 32 --seq-len 256 --steps 300 --lr 3e-3 \
  --optimizer adamw --dataset <data> --tokenizer <tok> --checkpoint-dir /tmp/ck
```
PASS = displayed loss **fluctuates** (e.g. 9.x descending), never a constant `1.0000`, AND training
stays bounded (no FATAL, EMA descends toward the ~1.56 baseline at the proven config) — i.e. gradients
are NOT corrupted (this is what the reverted pooled-buffer fix failed). Compare an 8-bit + simdgroup run
at tiny/batch16/lr3e-3 → should still reproduce EMA ~1.56.

(B) **Batch-LR transfer** — confirm default-off is a no-op, then exercise the knob:
```
# default off → effective == max_lr*μP (log line shows batch√(1.000))
andreai train ... --batch-size 16 --lr 3e-3
# opt-in: --lr is the batch-16 LR, scaled to batch 64
andreai train ... --batch-size 64 --lr 3e-3 --lr-ref-batch 16     # log shows batch√(2.000), eff 6e-3
```
Then sweep batch∈{16,32,64} with and without `--lr-ref-batch` and read the (now-correct) loss to find
the real direction/exponent. For hybrid/Muon at large batch, sweep `--muon-lr-scale` DOWN instead.

(C) After landing #3/#5: the per-task tests in §2/§3, then the mixed-length SFT throughput smoke (#3)
and the finite-diff gradient check (#5).

If anything regresses, the §0.A and §0.B edits are independent and individually revertible.

---

## 6. Round 2 — full-read gap drain

### Landed (real edits, need Mac verify)

**(C) BitNet ternary matmul — per-column scale bug FIXED (`src/tensor.rs::ternary_matmul`).**
The dequant scale was applied as a single global scalar: it read `absmean[0]` back to the CPU and did
`out_tensor.scale(absmean[0])`, scaling the ENTIRE `[m,n]` output by column-0's scale. Correct BitNet
b1.58 needs **per-column** scale `out[i][j] *= absmean[j]` (each output column comes from one weight
column with its own absmean). Every column but the first got the wrong magnitude unless all columns
happened to share a scale — silently degrading any `--bitnet` run (BitNet was even on the "don't
rebuild, already strong" list). Fix uses existing kernels: `gpu_broadcast_rows(absmean[n] → [m,n])`
then `gpu_mul` elementwise. The STE backward is **unaffected** — it records `Op::Matmul` against the
full-precision `weight.buffer`, so the forward quant/scale never enters the gradient. Bonus: this
deleted a `read_buffer` that was force-flushing the command batch on every BitNet matmul (a mid-forward
sync). Verify: a `--bitnet` smoke should now converge closer to the dense baseline; add a unit test
asserting `ternary_matmul` output ≈ `(x @ dequant(W))` per-column (not just column 0).

**(D) `quantize()` `unreachable!()` → proper panic (`src/quantize.rs`).** The public `quantize(data,
shape, bits, group_size)` matched `8 | 4` and fell through to `unreachable!()` for any other `bits`.
That branch IS reachable (it's user-supplied input — e.g. a future 2-bit KIVI path), and `unreachable!`
would abort with no message. Now `panic!("Unsupported quantization bits: {} …")`, matching the sibling
`dequantize`'s existing error. Trivial robustness fix.

### Gaps confirmed (NOT a quick fix — specced / flagged)

**GaLore is advertised but NOT implemented (`src/optim.rs`).** `AdamW::new_with_config` hard-asserts
`galore_rank == 0` with: *"GALORE … is not yet implemented. The current code allocates m/v buffers of
size `rank` but passes the full param size to the AdamW kernel, causing out-of-bounds GPU memory
access."* So `--galore-rank N>0` **panics at startup**. The dead projection code (lines ~92–104) and
the `optimizer.galore_rank > 0` log in `train.rs` never execute. **The panic guard is safe** (it
crashes rather than corrupts), but the capability is falsely listed as shipped in
`docs/HANDOFF_adamw_and_efficiency.md` ("what's already strong … GaLore", "10× capacity … GaLore"). It
is NOT in README/ROADMAP (those are clean). Treat GaLore as a TODO, not a feature.
- **To implement properly:** for a 2-D weight `W[r,c]`, keep a projection `P[c, rank]` (random,
  Gaussian, re-sampled every ~200 steps — FLORA-style — or via periodic SVD of the gradient). Each
  step: project the gradient `G_proj = G @ P` → `[r, rank]`; run Adam's m/v in that subspace (so m/v are
  `[r, rank]`, the memory win); project the Adam update back `U = U_proj @ P^T` → `[r,c]`; apply to W.
  Needs: a `[r,c]·[c,rank]` matmul + a `[r,rank]·[rank,c]` matmul per param per step, P storage, and a
  re-projection schedule. ~300 lines + reuses existing matmul kernels. Only worth it for the large 2-D
  params (gate it on `size > threshold`, like the existing dead code intended). Verify memory_bytes()
  actually drops and convergence matches full AdamW within tolerance.

### Known follow-ups (documented in-code, by design — not bugs)

- **CUDA backend**: compile parity is kept green and active wrappers no longer contain
  `unimplemented!`; runtime correctness remains NVIDIA-hardware-gated. Metal is the supported path on
  this Mac.
- **SSM / RWKV / linear-attention** (`ssm.rs`, `rwkv.rs`, `linear_attention.rs`): forward uses the
  **materialized** reference form; the chunked/decay-prefix O(N) form is the documented optimization
  follow-up. Recurrence semantics + selectivity are proven by tests. The `.to_vec()` calls in those
  files are reference/test paths or the inherent CPU step, not hot-path hazards.
- **MLA absorbed-form decode** (`attention.rs:27`) and **block-sparse trainable backward**
  (`attention.rs:45`, see §3) remain the two big attention follow-ups.
- **AndreOS backend**: removed from the production build surface until the native GPU driver and
  backend API are ready.

### Suggested next ROI order
1. Verify §0 + §6 landed edits on the Mac (loss-readout real-run is the key gate).
2. Land seq-packing (§2) and block-sparse trainable backward (§3) — both fully specced.
3. Implement GaLore for real (above) or delete the dead code + false doc claims to stop advertising it.
4. Then the chunked SSM/RWKV forms and MLA absorbed decode (throughput/context, not correctness).

---

## 7. Round 3 — correctness deep-read + latest-field map (June 2026)

### Likely-bug found (SPEC — too correctness-critical to blind-edit): Muon Newton-Schulz normalization

`src/optim.rs::Muon::step` orthogonalizes momentum with the **cubic** Newton-Schulz
`X = 1.5·X − 0.5·(X·Xᵀ)·X`, run `ns_steps` times. The matmul wiring is correct
(`X·Xᵀ` via `matmul_trans_b(rows,rows,cols)`, then `·X` via `matmul(rows,cols,rows)`, then
`scale(1.5)+axpy(−0.5)`). **The problem is the input normalization:** it scales the momentum by
`norm_scale = 1/√max(rows,cols)` before the iteration. The cubic map `g(σ)=1.5σ−0.5σ³` only converges
to 1 for singular values `σ ∈ (0, √3)`; for `σ > √3` it **diverges** (cubically). `1/√max(rows,cols)`
does NOT bound `σ_max` — it's a dimension heuristic, not a spectral bound. When the momentum magnitude
grows (e.g. larger batch → different gradient scale), `σ_max` can exceed √3 and the orthogonalized
update blows up. **This is the most likely root cause of the #6 "hybrid diverges at batch 32 / Muon
destabilizes at higher effective LR" finding** — and it's exactly what canonical Muon avoids.

Fix (canonical Muon): normalize by the **Frobenius norm**, `X = M / (‖M‖_F + 1e-7)`. Then
`σ_max ≤ ‖X‖_2 ≤ ‖X‖_F = 1 < √3`, so the iteration always converges, independent of batch/scale.
(The output is scale-free after orthogonalization, so this only fixes convergence — it doesn't change
the update direction.) Implementation note: ‖M‖_F needs a sum-of-squares reduction; do NOT read it back
to the CPU mid-step (that force-flushes the command batch → the §1 hazard). Add a tiny GPU kernel
`scale_by_inv_frob(x_buf, ssq_buf, size)` = `x[i] *= rsqrt(ssq[0] + 1e-14)` and feed it the existing
`l2_norm` reduction over `M`. ~30 lines MSL + one dispatch per 2-D param. Verify: sweep batch∈{16,32,64}
— Muon/hybrid should stay bounded at batch 32 where it currently diverges. (Optionally also switch to
the **quintic** coefficients `(3.4445, −4.7750, 2.0315)` — converges in ~5 iters from a worse start, the
form Keller-Jordan ships — but Frobenius-normalize first either way.)

### Verified SOUND on this read (not bugs — recorded so they aren't re-audited)
- **Autograd backward math**: `matmul` / `matmul_trans_b` / `add` (uses `accumulate_grad_shared` —
  the audit's buffer-aliasing fix) / `mul` / `softmax` / `scaled_causal_softmax` / `rms_norm` /
  `slice_cols` all check out dimensionally and in the chain rule. The Round-1/2 audit's transposed-dB
  and grad-aliasing bugs are genuinely fixed.
- **Sampling** (`generate.rs`): temperature / top-k / top-p / min-p / locally-typical all present with
  correct `partial_cmp` NaN-guards and greedy fast-path.
- **Hybrid per-layer topology** is wired (`linear_attn_period`, `ssm`, `rwkv` → per-block `AttnKind`,
  `model.rs:395-479`) — i.e. andreai already does Nemotron-3-style alternating attention/SSM layers,
  the 2026 frontier hybrid. No gap.

### Latest-field map (June 2026) — what's worth adding, ranked
1. **NorMuon** (neuron-wise normalized Muon): adds per-neuron adaptive LR on top of Muon's
   orthogonalization — reported **+11% over Muon**, +21% over Adam at 1.1B. Small delta on the Muon you
   already have; **do the Frobenius fix first** (correctness), then NorMuon (quality). (openreview 7TeJXgr7L6)
2. **Reconsider GaLore.** GaLore is unimplemented here (§6) and the field has moved past it: **SCALE**
   and "minimalist optimizer design" (arXiv 2506.16659) match/beat Adam at **35-45% memory** and
   **outperform GaLore/Fira/APOLLO**. Given Muon (no `v`) + 8-bit AdamW already cover the memory lever,
   **delete the dead GaLore code + false doc claim** rather than build it; if a subspace method is still
   wanted, target SCALE, not GaLore.
3. **MLA validated**: independent 2026 reads call MLA the winner of the KV-cache race — andreai's MLA
   (incremental-decode done) is the right bet; finishing the absorbed-form decode (§ existing handoff)
   is the high-value compute follow-up.
4. **Mamba-2 / hybrid**: the SSD view (SSM≈attention) and alternating-layer hybrids (Nemotron-3) are the
   architecture trend — andreai already has the mixers + per-layer topology; the **chunked O(N) SSM/RWKV
   forward** (currently materialized) is the remaining throughput piece.
5. **Speculative / self-distillation training-speed** (MIT 2026, "train a small predictor the big model
   verifies"): conceptually close to the EMA self-distillation + speculative-pretraining already in
   `train.rs`; a smaller verified-draft loop could ~2× training — a research follow-up, not a bug.

Net: the framework's architectural choices (MLA, hybrid mixers, Muon, BitNet, 8-bit optim) are aligned
with the June-2026 state of the art. The actionable gaps are **correctness** (Muon Frobenius norm — likely
the batch-scaling blocker) and **honesty** (drop GaLore), not missing capabilities.

Sources: Minimalist/SCALE optimizer arXiv:2506.16659; NorMuon (OpenReview 7TeJXgr7L6); Muon+latent
attention+MoE arXiv:2509.24406; Sebastian Raschka "LLM Research Papers 2026 (Jan–May)"; MIT News
2026-02-26 training-efficiency.

---

## 8. Round 4 — Muon Frobenius fix LANDED + checkpoint subsystem verified

### Landed (real edits, need Mac verify): Muon Frobenius normalization

Implemented the §7 fix. Four files, all mirroring existing `reduce_sum`/`scale_copy` patterns:
- `src/metal/shaders.rs` — new `MUON_FROB_NORMALIZE` kernel: single-threadgroup sum-of-squares
  reduction (grid-stride, ≤256 threads) then per-element `x = m · rsqrt(Σm² + 1e-14)`. One dispatch,
  no CPU readback (so it can't force-flush the command batch → no buffer hazard).
- `src/metal/mod.rs` — registered the `muon_frob_normalize` pipeline.
- `src/metal/compute.rs` — `gpu_muon_frob_normalize(ctx, m, x, size)`, same launch shape as `gpu_reduce_sum`.
- `src/optim.rs::Muon::step` — replaced `gpu_scale_copy(m, x, 1/√max(rows,cols))` with
  `gpu_muon_frob_normalize(m, x)`. `norm_scale` removed.

Why low-regression-risk: orthogonalization is scale-free, so on configs where NS already converged
(small batch, the proven EMA-1.56 runs) the update is unchanged; only the divergent regime (σ_max > √3)
is fixed. **Caveat:** a partially-converged NS (small `ns_steps`) is mildly input-scale-sensitive, so
**re-check `--muon-lr-scale`** after this lands. CUDA unaffected — `optim.rs` imports
`crate::metal` + `objc2_metal` directly, so it is metal-only by construction (CUDA's compute module
doesn't even have `gpu_scale_copy`); the supported build is metal.
Verify: batch∈{16,32,64} sweep with `--optimizer muon` and `--optimizer hybrid` — should stay bounded
at batch 32 where it currently diverges, and small-batch EMA should still hit ~1.56.

### Verified SOUND (not bugs): checkpoint / resume

`write_config`↔`read_config` are field-for-field symmetric and version-gated (v1→v11: n_kv_heads@v2,
the lowrank/MoE/bitnet/mup/n_predict group@v3, linear_attn@v5, period@v6, ssm@v7, rwkv@v8, mla@v9,
block-sparse top_k+block_size@v10, optimizer-param-count@v11). Both the model-only
(`save_checkpoint`/`load_checkpoint`) and training-state (`save_training_state`/`load_training_state`)
pairs call the same symmetric config codec; the opt-sidecar (`AOPT`) round-trips Muon/8-bit/hybrid
state. No write/read asymmetry, no version mismatch. (My `lr_ref_batch` addition is a `TrainConfig`
field, not `ModelConfig` — correctly NOT serialized.)

Minor gap (not corruption): `stochastic_depth`, `sliding_window`, and `mod_capacity` are NOT in
`write_config` — a model trained with e.g. `--sliding-window 256` loads with full attention unless the
flag is re-passed at inference. Low impact (they're re-specifiable CLI knobs), but worth serializing
for reproducibility — append them as v12 fields in `write_config`/`read_config` (bump `VERSION`, gate
reads on `version >= 12`).

### Running tally across all rounds
- **Landed:** loss-readout root-cause fix, batch-LR transfer (`--lr-ref-batch`), BitNet per-column
  scale, `quantize()` panic message, **Muon Frobenius normalization**.
- **Specced (exact code/kernels):** seq-packing forward threading (§2), block-sparse trainable backward
  (§3), v12 config fields (above). GaLore was removed from the active production surface; revisit SCALE
  instead if a subspace optimizer becomes worth adding.
- **Verified sound:** autograd backward math, generate sampling, hybrid per-layer topology,
  checkpoint/resume codec.

---

## 9. Round 5 — closure pass (full sweep of remaining subsystems)

### Landed (real edits)
- **v12 checkpoint: serialize `sliding_window`** (`src/checkpoint.rs`). A model trained with windowed
  attention silently loaded as full-causal (the field wasn't persisted). Now written/read symmetrically,
  gated on `version >= 12`, both version asserts bumped to 12, backward-compatible (older checkpoints
  default to 0). `stochastic_depth`/`fp16_activations` intentionally NOT persisted (train-time-only;
  should default off at inference/resume).
- **GaLore CLI honesty** (`src/main.rs`): `--galore-rank` help no longer claims "Saves optimizer memory"
  (it panics if > 0); now says NOT-IMPLEMENTED and points to Muon / `--optimizer adamw-8bit`.

### Swept and VERIFIED SOUND (no bugs — recorded so they aren't re-audited)
- **DPO** (`dpo.rs`): loss `−log σ(β·(Δchosen − Δrejected))` and gradient signs are correct;
  `sequence_log_probs` uses stable log-softmax; `log1p_exp`/`sigmoid` are overflow-safe (branch on
  magnitude/sign); pair mmap reads are bounds-asserted.
- **SFT/DPO JSONL parsing**: `\uXXXX` handling includes **surrogate-pair** reconstruction
  (U+1F600 test passes) and rejects malformed escapes.
- **Eval** (`eval.rs`): perplexity = `exp(mean NLL)` — correct.
- **datapipe** (`datapipe.rs`): MinHash shingling guards `chars.len() < shingle_size`;
  `quality_score` early-returns for `len < 50` so no div-by-zero (minor: it mixes byte-len with
  char-count for the alpha ratio — harmless heuristic imprecision, not worth a format change).
- **Tokenizer / data loader / generate**: index arithmetic (`seq_len-1`, mixer cumulative, spec-decode
  offsets) is guarded by prior length checks; sampling math already verified in Round 3.

### Closure status — every identified item
| Item | State |
|---|---|
| Large-batch loss readout "1.0" | **FIXED** (unpooled buffer) |
| Buffer-hazard root cause | **DOCUMENTED** (§1) + the loss-readout instance fixed; systemic quarantine is an optional future hardening |
| Batch-size LR transfer | **LANDED** (`--lr-ref-batch`, default off) |
| BitNet per-column scale | **FIXED** |
| `quantize()` unreachable | **FIXED** (proper panic) |
| Muon NS divergence (batch-32) | **FIXED** (Frobenius normalization) — re-check `--muon-lr-scale` |
| `sliding_window` not persisted | **FIXED** (v12) |
| GaLore advertised-not-implemented | **REMOVED from active production surface**; historical docs keep the field decision context |
| autograd / checkpoint / DPO / eval / sampling / datapipe / JSON | **VERIFIED SOUND** |
| Seq-packing model integration | **SPECCED** (§2) — needs Metal build to land safely |
| Block-sparse trainable backward | ✅ **LANDED** (§3) — scatter-add kernel + `Op::GatherBlocks`, grad-checked; fixed a latent non-square `batched_matmul_trans_a` param-swap |
| Chunked O(N) SSM/RWKV forward; MLA absorbed decode; NorMuon; simdgroup >1.29×; CUDA runtime proof | **ENHANCEMENT follow-ups** (not bugs) — documented, hardware-gated |

### What "fully closed" means here
Every **bug** I could identify is fixed; every **correctness-critical subsystem** is read and verified
sound; every **gap** is either landed or specced to exact code. What remains is (a) two GPU-kernel
features that the prior in-tree attempt diverged on and that the unit suite provably can't validate
(seq-packing, sparse backward), and (b) architectural enhancements (chunked mixers, MLA absorbed,
NorMuon). Those need your Metal hardware to implement and verify — I will not blind-write unrunnable
GPU kernels into `main` and call them done. The **gate for all 7 landed edits** is the §5 protocol on
the Mac: `cargo test --features metal` (full + serial), the batch-sweep (loss readout + Muon), and a
`--bitnet` smoke. Once those pass, the bug backlog is empty and the remaining work is feature dev.

---

## 10. Round 6 — feature dev (landed, additive, default-off → zero regression)

Both features are opt-in and don't change any existing run; verify on the Mac before relying on them.

### A. EMA checkpoint export (`src/checkpoint.rs`, `src/train.rs`)
The train loop maintained EMA weights (for self-distillation) but never saved them — the EMA is
typically a better model than the live snapshot (BYOL / self-distillation), so it was being discarded.
- New `checkpoint::save_checkpoint_ema(path, model, ema_buffers, step)` — byte-identical format to
  `save_checkpoint` (loads via the normal path); trainable tensors' data comes from the EMA buffers,
  frozen ReLoRA base params from the model.
- Wired into the train loop: when `--ema-decay > 0`, writes `ema_{step}.bin` at each checkpoint
  interval and `ema_final.bin` at the end, with a hint to compare via `andreai perplexity`.
- Zero cost when EMA is off (`ema_buffers` empty).

### B. NorMuon — per-neuron normalized Muon (`--normuon`)
Neuron-wise normalized Muon (arXiv NorMuon, ~+11% over Muon, ~+21% over Adam at 1.1B): adds Adam-like
per-output-neuron adaptivity on top of Muon's orthogonalization. Builds directly on the Round-4
Frobenius fix. Opt-in (`--normuon`, default off → plain Muon); applies to `--optimizer muon` and the
`hybrid` Muon group.
- After Newton-Schulz produces the orthogonal update X[rows,cols], per output row i:
  `r_i = mean_j X[i,j]²` (`row_dot_reduce` + scale), `v_i = β₂·v_i + (1−β₂)·r_i` (`ema_update`),
  `s_i = 1/(√(v_i/(1−β₂ᵗ)) + ε)` (new `inv_sqrt_bc` kernel), `X[i,:] *= s_i` (`scale_rows`).
- New kernel `inv_sqrt_bc` (shaders/mod/compute) — the only new GPU code; everything else reuses
  existing kernels.
- **Deliberately no optimizer-state-serialization change:** the per-row second moment `ns_vrow` lives
  in `MuonState` (allocated in `Muon::new`, like the NS workspace) and is NOT written to the opt
  sidecar. On resume it restarts at 0 and re-warms in a few steps (fast β₂=0.95 EMA) — a negligible,
  self-correcting cost that keeps the fragile state-blob format byte-compatible.
- Wiring: `Muon{normalized,norm_beta2,norm_eps}` + `MuonState{ns_vrow,ns_rowss}` +
  `Muon::set_normalization`; `--normuon` flag → `config.normuon` → `set_normalization` on both the
  standalone Muon and `HybridOptimizer.muon`.
- Verify: `--optimizer hybrid --normuon` should match-or-beat plain hybrid at equal compute; re-check
  `--muon-lr-scale` (NorMuon changes the effective per-neuron update magnitude).

### Swept-and-sound this round (no gaps): SFT prompt masking (loss masked to response tokens only,
gradient zeroed for prompt — correct), cosine schedule already floors at `min_lr = max_lr·0.1`,
`Merge`/`Grow`/gguf/safetensors/dedup/distill commands already exist.

### C. No-repeat-ngram generation control (`--no-repeat-ngram-size`, `src/generate.rs`)
Repetition *penalty* exists but can't fully stop exact loops (the classic "command command command"
failure for an OS/code assistant). Added HuggingFace-style `no_repeat_ngram_size`: hard-bans any token
that would complete an n-gram already present in the generated history (logit → −∞ → prob 0).
- Pure CPU, applied in BOTH samplers (`sample_token`, `sample_token_prescaled`) after the repetition
  penalty; default 0 = off, so existing behavior is unchanged.
- `SamplingConfig.no_repeat_ngram_size` + CLI `--no-repeat-ngram-size` (3 is a good assistant default);
  wired through all construction sites (main generate, eval, Default).
- **Has real unit tests** (`generate.rs::ngram_tests`) — this helper is pure CPU logic, so unlike the
  GPU code these actually run in `cargo test` and verify the ban logic (trigram recurrence, n=1,
  multiple recurrences, disabled/short-history no-ops).

### Feature menu still open (specced; need Metal hardware to implement+verify)
| Feature | Value | Notes |
|---|---|---|
| Seq-packing model integration | throughput (no pad waste) | §2 — thread `seg_ids` through forward |
| Block-sparse trainable backward | cheaper training / long ctx | ✅ LANDED (§3) — kernel + autograd op + grad-checks; model wiring deferred to #12 |
| Chunked O(N) SSM/RWKV forward | long-context throughput | mixers materialize today |
| MLA absorbed-form decode | faster decode | latent-cache decode already done |
| SCALE-style subspace optimizer | optimizer memory | optional future work; Muon+8-bit already cover the active memory lever |
| Persistent/fused mega-kernel; simdgroup >1.29× | raw throughput | the big systems lever (RESEARCH_2026) |
