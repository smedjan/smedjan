# AndreAI Audit Round 2 — Findings

Generated: 2025-01-28
Status: **Pending implementation**
Previous round: commit 915b0b8 (all Round 1 fixes landed)

---

## Priority 1 — CRITICAL

### RISK-4 / BUG-6: `accumulate_grad` in-place add shares buffers unsafely

**File:** `autograd.rs:195-205`
**Severity:** CRITICAL — silent gradient corruption during training
**Introduced by:** Round 1 RISK-2 fix (in-place gradient accumulation)

The Round 1 optimization changed `accumulate_grad` to use `gpu_add_inplace` instead of allocating a new buffer. The problem: when the first call to `accumulate_grad` finds no existing gradient for a tensor, it inserts the passed `grad` buffer directly (via `Retained::clone`, which is a refcount bump, not a copy). If the same `grad` buffer is then passed to `accumulate_grad` for a DIFFERENT tensor, both tensors now share the same underlying Metal buffer.

Later, when any backward op calls `gpu_add_inplace` on one tensor's gradient, it mutates the shared buffer, corrupting the other tensor's gradient.

**Affected backward ops** (pass same grad to multiple `accumulate_grad` calls):
- `backward_add_rms_norm` (lines 440-441) — `grad_sum` shared between inputs[0] and inputs[1]
- `backward_add` (lines 458-460) — `out_grad` shared between both inputs
- Any future broadcast/split backward

**Fix:** On first insert (no existing grad), allocate a fresh buffer and copy:
```rust
fn accumulate_grad(ctx: &Arc<MetalContext>, tensor_id: usize, grad: &Retained<GpuBuffer>, size: usize) {
    GRADS.with(|grads| {
        let mut grads = grads.borrow_mut();
        if let Some(existing) = grads.get(&tensor_id) {
            // In-place accumulate: existing += grad
            compute::gpu_add_inplace(ctx, existing, grad, size as u32);
        } else {
            // MUST copy — the same grad buffer may be accumulated to other tensors.
            let owned = ctx.alloc_buffer(size * 4);
            compute::gpu_copy(ctx, grad, &owned, size as u32);
            grads.insert(tensor_id, owned);
        }
    });
}
```

---

## Priority 2 — BUG

### BUG-4: `backward_batched_matmul` dB produces transposed gradient

**File:** `autograd.rs:793-801`
**Severity:** HIGH — wrong weight gradients in batched matmul backward

For `C[b] = A[b] @ B[b]` where A:[B,M,K], B:[B,K,N], C:[B,M,N]:
- Correct: `dB[b] = A[b]^T @ dC[b]` → shape [K,N]
- Code calls: `gpu_matmul_trans_a(dc_sub, a_sub, db_sub, m, n, k)`
- This computes: `dC^T @ A = [N,M] @ [M,K] = [N,K]` — **transposed**

**Fix:** Swap arguments:
```rust
// Before (WRONG):
compute::gpu_matmul_trans_a(ctx, &dc_sub, &a_sub, &db_sub, m as u32, n as u32, k as u32);

// After (CORRECT):
compute::gpu_matmul_trans_a(ctx, &a_sub, &dc_sub, &db_sub, k as u32, m as u32, n as u32);
```

`gpu_matmul_trans_a(A, B, C, M, K, N)` computes `C[K,N] = A^T[K,M] @ B[M,N]`.
So `gpu_matmul_trans_a(a_sub, dc_sub, db_sub, k, m, n)` computes `db[M,N] = a^T[M,K] @ dc[K,N]` — wait, that's wrong too. Let me be precise:

The function signature: `gpu_matmul_trans_a(A, B, C, M, K, N)` where A is [M,K] stored row-major, computes `C[K,N] = A^T[K,M] @ B[M,N]`.

We need `dB[K,N] = A^T[K,M] @ dC[M,N]`. So:
- A_param = A_sub (shape [M,K])
- B_param = dC_sub (shape [M,N])
- M_param = M, K_param = K, N_param = N

```rust
compute::gpu_matmul_trans_a(ctx, &a_sub, &dc_sub, &db_sub, m as u32, k as u32, n as u32);
```

---

## Priority 3 — PERFORMANCE

### PERF-8: Batched matmul forward — O(batch) serial GPU dispatches

**File:** `tensor.rs:712-740`

`batched_matmul_trans_b` loops over batch dimension, allocating 3 sub-buffers and dispatching `gpu_matmul_trans_b` per batch element. For 32 attention heads, that's 96 buffer allocations + 32 matmul dispatches.

**Fix:** Add a `BATCHED_MATMUL_TRANS_B` MSL shader that handles the full [B,M,K] @ [B,N,K]^T = [B,M,N] in a single dispatch using a 3D thread grid (batch × tile_row × tile_col). Each threadgroup computes one 32×32 tile for one batch element.

### PERF-9: Batched matmul backward — O(batch) serial dispatches × 3

**File:** `autograd.rs:735-802`

Same pattern as PERF-8 but worse — 3 sub-buffer allocations + 2 matmul dispatches per batch element (dA and dB). For 32 heads: 192 allocations + 64 matmul dispatches.

**Fix:** Add `BATCHED_MATMUL` and `BATCHED_MATMUL_TRANS_A` shaders. Use strided buffer access with batch offset instead of per-batch sub-buffer copies.

### PERF-10: Batched matmul_trans_b backward — same O(batch) pattern

**File:** `autograd.rs` (backward_batched_matmul_trans_b)

Same serial loop pattern for the trans_b variant used in attention score @ V backward.

**Fix:** Same batched kernel approach as PERF-8/9.

### PERF-11: `backward_slice_flat` zeros entire source buffer

**File:** `autograd.rs:690-698`

For a slice of `length` elements from a `source_size` tensor, it `gpu_fill`-zeros all `source_size` elements, then copies `length` elements at the offset. The zero-fill is O(source_size) when only the non-sliced region needs zeroing.

**Fix:** Use a conditional zero kernel that only zeros elements outside [offset, offset+length), or just accept the cost since slices are typically a small fraction of ops.

---

## Priority 4 — RISK

### RISK-5: Checkpoint recompute pins GPU memory for full sub-tape

**File:** `autograd.rs:524-620`

The checkpoint backward re-runs the forward to build a sub-tape, then walks it backward. The sub-tape holds `Retained<GpuBuffer>` for ALL intermediate tensors simultaneously. For a transformer block with ~20 ops, this pins ~20 intermediate buffers during backward.

**Fix:** Process and drop sub-tape entries incrementally:
```rust
while let Some(sub_entry) = sub_tape.pop() {
    // process backward for sub_entry
    // sub_entry is dropped here, releasing its GPU buffers
}
```
Instead of iterating with `for sub_entry in sub_tape.iter().rev()` which keeps all entries alive.

---

## Verification checklist

After implementing all fixes:
- [ ] `cargo check` — zero errors
- [ ] `cargo test` — 55/55 pass
- [ ] Verify `accumulate_grad` never shares buffers (add a test with `backward_add`)
- [ ] Verify batched matmul dB gradient shape matches expected (add a numerical gradient test)
- [ ] Benchmark batched matmul before/after kernel change

ROUND 2 AUDIT — COMPLETE FINDINGS

    BUGS

    BUG-4: backward_batched_matmul dB computation produces transposed result
    File: autograd.rs:793-801
    For C[b] = A[b] @ B[b] where B is [B, K, N], dB should be [K,N].
    Code calls gpu_matmul_trans_a(dc_sub, a_sub, db_sub, m, n, k) which computes dC^T @ A = [N,M]@[M,K] = [N,K].
    But dB should be A^T @ dC = [K,M]@[M,N] = [K,N].
    Result is [N,K] instead of [K,N] — transposed gradient. The correct call should be gpu_matmul_trans_a(a_sub, dc_sub, db_sub, k, m, n) to get A^T @ dC = [K,M]@[M,N] = [K,N].

    BUG-5: backward_slice_flat zeros the full source tensor, then copies grad into offset
    File: autograd.rs:690-698
    Uses gpu_fill to zero a buffer of source_size elements, then gpu_buffer_copy to write length elements at an offset. This is correct but wasteful — the zero-fill is O(source_size) when only the
    non-sliced region needs zeroing. More importantly, if the same source tensor is sliced multiple times, the second accumulate_grad call will in-place-add into the first gradient, which IS correct after
    our RISK-2 fix. No correctness bug here, just perf.

    BUG-6: backward_add_rms_norm accumulates the same grad_sum to both inputs
    File: autograd.rs:440-441
    accumulate_grad(ctx, entry.inputs[0], &grad_sum, rows * cols);
    accumulate_grad(ctx, entry.inputs[1], &grad_sum, rows * cols);
    After RISK-2 fix (in-place add), the first accumulate_grad call for inputs[0] will ADD grad_sum into the existing gradient. If inputs[0] already has a gradient, gpu_add_inplace modifies the existing
    buffer. But then the SECOND call passes the same grad_sum buffer — this is fine because gpu_add_inplace modifies a (existing), not b (grad_sum). Not a bug — grad_sum is read-only in add_inplace.

    Wait — actually check again. The first call: if inputs[0] has no existing grad, it inserts grad_sum directly (clone of the Retained). The second call: if inputs[1] also has no existing grad, it inserts
    another clone of grad_sum. Now both inputs[0] and inputs[1] share the same underlying Metal buffer. If later backward ops call gpu_add_inplace on one of them, it mutates the shared buffer, corrupting
    the other. THIS IS A BUG introduced by our RISK-2 fix.

    The old code (gpu_add into a new buffer) was safe because it always created a new buffer. The new in-place code shares the buffer on first insert. Fix: always clone/copy the buffer before inserting, or
    only use in-place when existing grad already exists.

    PERF-8: batched_matmul_trans_b forward — O(batch) serial dispatches
    File: tensor.rs:712-740
    Loops over batch, allocating sub-buffers and dispatching individual matmul_trans_b per batch element. For 32 attention heads, that's 32 alloc+copy+matmul dispatches. Needs a strided batched matmul
    kernel.

    PERF-9: backward_batched_matmul — O(batch) serial dispatches with 3 allocs per iteration
    File: autograd.rs:735-802
    Same pattern: loops over batches with per-element alloc+copy+matmul. For 32 heads, that's 96 alloc+dispatch operations per backward batched matmul (3 sub-buffers × 32 batches).

    PERF-10: backward_batched_matmul_trans_b — same O(batch) serial pattern
    File: autograd.rs:668+ (need to verify exact location)
    Same issue as PERF-9 but for the trans_b variant.

    PERF-11: backward_slice_flat zeros entire source buffer unnecessarily
    File: autograd.rs:690-698
    For a slice of 100 elements from a 10000-element tensor, it zeros all 10000 elements then copies 100. Could use targeted zero (zero only the non-sliced regions) or use a scatter-style kernel.

    RISK-4: accumulate_grad in-place add shares buffers between multiple gradient consumers
    File: autograd.rs:195-205
    When the first accumulate_grad call inserts a gradient buffer (no existing grad), subsequent in-place additions to OTHER tensors that received the same buffer will corrupt it. This affects any backward
    op that passes the same grad buffer to multiple accumulate_grad calls (e.g., backward_add_rms_norm at lines 440-441, backward_add at lines 458-460 which passes out_grad to both inputs, and any
    split/broadcast gradient).
