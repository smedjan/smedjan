use crate::metal::{compute, GpuBuffer, MetalContext};
use objc2::rc::Retained;
use objc2_metal::MTLBuffer;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Global tensor ID counter.
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

pub fn next_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// A closure that re-runs a checkpointed forward pass to recover the sub-tape.
/// Takes the MetalContext and returns the sub-tape entries produced by the recomputed forward.
pub type RecomputeFn = Box<dyn Fn(&Arc<MetalContext>) -> Vec<TapeEntry>>;

/// Operations tracked by the tape.
#[derive(Debug, Clone)]
pub enum Op {
    Matmul,
    MatmulTransB,
    Add,
    Mul,
    Softmax,
    RmsNorm { eps: f32 },
    Silu,
    Reshape,
    CrossEntropy,
    Embedding,
    Scale { factor: f32 },
    Transpose {
        batch: usize,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        forward_dir: bool, // true = bsh→bhs, false = bhs→bsh
    },
    RoPE { seq_len: u32, head_dim: u32, offset: u32, theta: f32 },
    Checkpoint { layer_idx: usize },
    /// Slice a contiguous region from a flat buffer. offset and length in elements.
    Slice { offset: usize, length: usize, source_size: usize },
    /// Concatenate multiple tensors along first dimension (used to reassemble heads).
    ConcatParts { part_sizes: Vec<usize> },
    /// Batched matrix multiply: A[b] @ B[b] for each batch element.
    /// A: [B, M, K], B: [B, K, N] → C: [B, M, N]
    BatchedMatmul,
    /// Batched matrix multiply with B transposed: A[b] @ B[b]^T for each batch element.
    /// A: [B, M, K], B: [B, N, K] → C: [B, M, N]
    BatchedMatmulTransB,
    /// Fused SiLU-gate: output = silu(gate) * up
    /// gate and up are the two inputs. Backward: d_gate = d_out * up * silu'(gate), d_up = d_out * silu(gate)
    SiluGate,
    /// Fused residual add + RMS norm: output = rms_norm(a + b, weight, eps)
    /// inputs: [a, b, weight], cached: (a+b) buffer for backward
    RmsNormResidual { eps: f32 },
    /// GQA KV head expansion: repeat each KV head group_size times.
    /// Backward: sum group_size gradient blocks back into each KV head.
    RepeatKv { n_kv_heads: usize, group_size: usize, head_block: usize },
    /// Per-row scaling: out[r][c] = input[r][c] * scales[r]
    ScaleRows { rows: usize, cols: usize },
    /// Flash Attention: fused Q@K^T → mask → softmax → @V
    /// inputs: [Q, K, V], output: O, cached: O (for backward D computation)
    FlashAttention { batch_heads: usize, seq_q: usize, seq_k: usize, head_dim: usize, kv_offset: u32 },
}

/// A single entry on the autodiff tape.
pub struct TapeEntry {
    pub op: Op,
    pub inputs: Vec<usize>,     // TensorIds of inputs
    pub output: usize,          // TensorId of output
    pub input_buffers: Vec<Retained<GpuBuffer>>,
    pub output_buffer: Retained<GpuBuffer>,
    pub shapes: Vec<Vec<usize>>, // TODO: replace with SmallVec or fixed arrays to eliminate heap alloc
    pub cached: Option<Retained<GpuBuffer>>,
}

thread_local! {
    static TAPE: RefCell<Vec<TapeEntry>> = const { RefCell::new(Vec::new()) };
    static GRADS: RefCell<HashMap<usize, Retained<GpuBuffer>>> = RefCell::new(HashMap::new());
    static NO_GRAD: RefCell<bool> = const { RefCell::new(false) };
    static RECOMPUTE_REGISTRY: RefCell<HashMap<usize, RecomputeFn>> = RefCell::new(HashMap::new());
}

/// Check if we're currently recording ops.
pub fn is_recording() -> bool {
    NO_GRAD.with(|ng| !*ng.borrow())
}

/// Return diagnostic info about the current tape: (num_ops, total_output_bytes).
/// Reads output_buffer size from each tape entry to compute total activation memory.
pub fn tape_stats() -> (usize, usize) {
    TAPE.with(|tape| {
        let tape = tape.borrow();
        let num_ops = tape.len();
        let total_bytes: usize = tape.iter()
            .map(|entry| entry.output_buffer.length())
            .sum();
        (num_ops, total_bytes)
    })
}

/// Record an operation on the tape.
pub fn record(entry: TapeEntry) {
    if is_recording() {
        TAPE.with(|tape| tape.borrow_mut().push(entry));
    }
}

/// Clear the tape and all stored gradients.
pub fn clear_tape() {
    TAPE.with(|tape| {
        let entries = tape.borrow_mut().drain(..).collect::<Vec<_>>();
        for entry in entries {
            // Recycle output and cached buffers — these are unique to the tape entry.
            // Input buffers are shared refs to source tensors and MUST NOT be recycled
            // (model parameters still hold references to them).
            MetalContext::recycle_buffer(entry.output_buffer);
            if let Some(cached) = entry.cached {
                MetalContext::recycle_buffer(cached);
            }
        }
    });
    GRADS.with(|grads| grads.borrow_mut().clear());
}

/// Clear the tape entries (freeing activations) but preserve accumulated gradients.
/// Used in gradient accumulation: after each micro-step's backward pass we free the tape
/// to reclaim activation memory, but keep gradients so the next micro-step accumulates on top.
pub fn clear_tape_keep_grads() {
    use objc2_metal::MTLBuffer;

    // Collect gradient buffer contents() pointers so we don't recycle shared buffers.
    // A buffer in GRADS that's also in the tape (output_buffer/cached) must NOT be recycled,
    // or the next micro-step's forward pass would overwrite the gradient via pool reuse.
    let grad_ptrs: std::collections::HashSet<usize> = GRADS.with(|grads| {
        grads.borrow().values().map(|buf| {
            buf.contents().as_ptr() as usize
        }).collect()
    });

    TAPE.with(|tape| {
        let entries = tape.borrow_mut().drain(..).collect::<Vec<_>>();
        for entry in entries {
            let out_ptr = entry.output_buffer.contents().as_ptr() as usize;
            if !grad_ptrs.contains(&out_ptr) {
                MetalContext::recycle_buffer(entry.output_buffer);
            }
            if let Some(cached) = entry.cached {
                let cached_ptr = cached.contents().as_ptr() as usize;
                if !grad_ptrs.contains(&cached_ptr) {
                    MetalContext::recycle_buffer(cached);
                }
            }
        }
    });
}

/// Zero all stored gradient buffers. Call after optimizer.step() when using gradient
/// accumulation to prepare for the next accumulation cycle.
pub fn zero_grads() {
    GRADS.with(|grads| grads.borrow_mut().clear());
}

/// Scale all stored gradient buffers by a constant factor.
/// Used in gradient accumulation to average gradients: scale by 1/grad_accum_steps.
pub fn scale_grads(ctx: &Arc<MetalContext>, factor: f32) {
    GRADS.with(|grads| {
        let grads = grads.borrow();
        for (_id, grad_buf) in grads.iter() {
            let size = grad_buf.length() / 4; // f32 = 4 bytes
            compute::gpu_scale(ctx, grad_buf, size as u32, factor);
        }
    });
}

/// Clear all registered recompute functions. Call alongside `clear_tape()`.
pub fn clear_recompute_registry() {
    RECOMPUTE_REGISTRY.with(|reg| reg.borrow_mut().clear());
}

/// Register a recompute closure for a given layer index.
/// During backward, the `Op::Checkpoint` arm looks this up to re-run the forward.
pub fn register_recompute(layer_idx: usize, f: RecomputeFn) {
    RECOMPUTE_REGISTRY.with(|reg| {
        reg.borrow_mut().insert(layer_idx, f);
    });
}

/// Run a closure while temporarily capturing tape entries onto a fresh sub-tape.
/// The main tape is swapped out, the closure runs (recording onto the empty tape),
/// and the captured sub-tape entries are returned. The main tape is then restored.
/// The caller decides what to do with the sub-tape — it is NOT kept on the main tape.
pub fn checkpoint_forward<F, R>(f: F) -> (R, Vec<TapeEntry>)
where
    F: FnOnce() -> R,
{
    // Swap the main tape with an empty one
    let main_tape = TAPE.with(|tape| {
        let mut t = tape.borrow_mut();
        std::mem::take(&mut *t)
    });

    // Run the closure — all record() calls go to the now-empty TAPE
    let result = f();

    // Extract the sub-tape (everything the closure recorded)
    let sub_tape = TAPE.with(|tape| {
        let mut t = tape.borrow_mut();
        std::mem::take(&mut *t)
    });

    // Restore the main tape
    TAPE.with(|tape| {
        *tape.borrow_mut() = main_tape;
    });

    (result, sub_tape)
}

/// RAII guard that restores NO_GRAD state on drop (panic-safe).
struct NoGradGuard {
    prev: bool,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        NO_GRAD.with(|ng| *ng.borrow_mut() = self.prev);
    }
}

/// Execute a closure with gradient computation disabled.
/// Panic-safe: restores recording state even if the closure panics.
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = NO_GRAD.with(|ng| {
        let prev = *ng.borrow();
        *ng.borrow_mut() = true;
        prev
    });
    let _guard = NoGradGuard { prev };
    f()
}

/// Get the gradient buffer for a tensor ID.
pub fn get_grad(tensor_id: usize) -> Option<Retained<GpuBuffer>> {
    GRADS.with(|grads| grads.borrow().get(&tensor_id).cloned())
}

/// Store a gradient buffer for a tensor ID, accumulating if one already exists.
/// On first insert, uses the buffer directly (refcount clone — zero cost).
/// Callers that pass the SAME buffer to multiple accumulate_grad calls
/// (backward_add, backward_add_rms_norm) MUST use accumulate_grad_shared instead.
fn accumulate_grad(ctx: &Arc<MetalContext>, tensor_id: usize, grad: &Retained<GpuBuffer>, size: usize) {
    GRADS.with(|grads| {
        let mut grads = grads.borrow_mut();
        if let Some(existing) = grads.get(&tensor_id) {
            compute::gpu_add_inplace(ctx, existing, grad, size as u32);
        } else {
            grads.insert(tensor_id, grad.clone());
        }
    });
}

/// Like accumulate_grad but copies the buffer on first insert.
/// Use this ONLY when the same grad buffer is passed to multiple tensor IDs
/// (e.g. backward_add passes out_grad to both inputs).
fn accumulate_grad_shared(ctx: &Arc<MetalContext>, tensor_id: usize, grad: &Retained<GpuBuffer>, size: usize) {
    GRADS.with(|grads| {
        let mut grads = grads.borrow_mut();
        if let Some(existing) = grads.get(&tensor_id) {
            compute::gpu_add_inplace(ctx, existing, grad, size as u32);
        } else {
            let owned = ctx.alloc_buffer(size * 4);
            compute::gpu_copy(ctx, grad, &owned, size as u32);
            grads.insert(tensor_id, owned);
        }
    });
}

/// Run backward pass from a loss tensor. The loss should be a scalar.
pub fn backward(ctx: &Arc<MetalContext>, loss_id: usize) {
    // Initialize loss gradient as 1.0
    let ones = ctx.alloc_buffer(4);
    compute::gpu_fill(ctx, &ones, 1, 1.0);
    GRADS.with(|grads| grads.borrow_mut().insert(loss_id, ones));

    // Take ownership of the tape so it's not borrowed during backward.
    // This is critical for gradient checkpointing: backward_checkpoint calls
    // checkpoint_forward which needs to borrow TAPE. If we held a borrow here,
    // that would panic with "RefCell already borrowed".
    let tape = TAPE.with(|tape| std::mem::take(&mut *tape.borrow_mut()));
    {
        // Walk tape in reverse
        for entry in tape.iter().rev() {
            let out_grad = GRADS.with(|grads| grads.borrow().get(&entry.output).cloned());
            let out_grad = match out_grad {
                Some(g) => g,
                None => continue, // No gradient flows to this op
            };

            match &entry.op {
                Op::Matmul => {
                    backward_matmul(ctx, entry, &out_grad);
                }
                Op::MatmulTransB => {
                    backward_matmul_trans_b(ctx, entry, &out_grad);
                }
                Op::Add => {
                    backward_add(ctx, entry, &out_grad);
                }
                Op::Mul => {
                    backward_mul(ctx, entry, &out_grad);
                }
                Op::Softmax => {
                    backward_softmax(ctx, entry, &out_grad);
                }
                Op::RmsNorm { eps } => {
                    backward_rms_norm(ctx, entry, &out_grad, *eps);
                }
                Op::RmsNormResidual { eps } => {
                    backward_rms_norm_residual(ctx, entry, &out_grad, *eps);
                }
                Op::RepeatKv { n_kv_heads, group_size, head_block } => {
                    backward_repeat_kv(ctx, entry, &out_grad, *n_kv_heads, *group_size, *head_block);
                }
                Op::ScaleRows { rows, cols } => {
                    backward_scale_rows(ctx, entry, &out_grad, *rows, *cols);
                }
                Op::FlashAttention { batch_heads, seq_q, seq_k, head_dim, kv_offset } => {
                    backward_flash_attention(ctx, entry, &out_grad, *batch_heads, *seq_q, *seq_k, *head_dim, *kv_offset);
                }
                Op::Silu => {
                    backward_silu(ctx, entry, &out_grad);
                }
                Op::SiluGate => {
                    backward_silu_gate(ctx, entry, &out_grad);
                }
                Op::Reshape => {
                    backward_reshape(ctx, entry, &out_grad);
                }
                Op::CrossEntropy => {
                    // Cross-entropy backward is computed in the forward pass (fused).
                    // The gradient is already stored in the cached buffer.
                    if let Some(grad_logits) = &entry.cached {
                        let size: usize = entry.shapes[0].iter().product();
                        accumulate_grad(ctx, entry.inputs[0], grad_logits, size);
                    }
                }
                Op::Embedding => {
                    backward_embedding(ctx, entry, &out_grad);
                }
                Op::Scale { factor } => {
                    backward_scale(ctx, entry, &out_grad, *factor);
                }
                Op::Transpose { batch, seq_len, n_heads, head_dim, forward_dir } => {
                    backward_transpose(ctx, entry, &out_grad, &TransposeParams {
                        batch: *batch, seq_len: *seq_len, n_heads: *n_heads, head_dim: *head_dim, forward_dir: *forward_dir,
                    });
                }
                Op::Checkpoint { layer_idx } => {
                    backward_checkpoint(ctx, entry, &out_grad, *layer_idx);
                }
                Op::Slice { offset, length, source_size } => {
                    backward_slice(ctx, entry, &out_grad, *offset, *length, *source_size);
                }
                Op::ConcatParts { part_sizes } => {
                    backward_concat_parts(ctx, entry, &out_grad, part_sizes);
                }
                Op::BatchedMatmul => {
                    backward_batched_matmul(ctx, entry, &out_grad);
                }
                Op::BatchedMatmulTransB => {
                    backward_batched_matmul_trans_b(ctx, entry, &out_grad);
                }
                Op::RoPE { seq_len, head_dim, offset, theta } => {
                    backward_rope(ctx, entry, &out_grad, *seq_len, *head_dim, *offset, *theta);
                }
            }
        }
    }
    // Restore the tape (in case anything needs it later, though typically clear_tape is called)
    TAPE.with(|t| *t.borrow_mut() = tape);
}

/// Cast a float buffer to half for FP16 matmul backward. Returns FP16 buffer (size*2 bytes).
fn cast_buf_f16(ctx: &Arc<MetalContext>, buf: &Retained<GpuBuffer>, num_elements: usize) -> Retained<GpuBuffer> {
    let f16_buf = ctx.alloc_buffer(num_elements * 2);
    compute::gpu_cast_f32_to_f16(ctx, buf, &f16_buf, num_elements as u32);
    f16_buf
}

fn backward_matmul(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    // C = A @ B → dA = dC @ B^T, dB = A^T @ dC
    let a_shape = &entry.shapes[0];
    let b_shape = &entry.shapes[1];
    let rank_a = a_shape.len();
    let rank_b = b_shape.len();

    let m = a_shape[rank_a - 2];
    let k = a_shape[rank_a - 1];
    let n = b_shape[rank_b - 1];

    // FP16 backward: cast to half for bandwidth savings (shared mem is half anyway)
    let grad_f16 = cast_buf_f16(ctx, out_grad, m * n);
    let b_f16 = cast_buf_f16(ctx, &entry.input_buffers[1], k * n);
    let a_f16 = cast_buf_f16(ctx, &entry.input_buffers[0], m * k);

    // dA = dC @ B^T : [M, N] @ [N, K] = [M, K]
    let da_buf = ctx.alloc_buffer(m * k * 4);
    compute::gpu_matmul_trans_b_f16(ctx, &grad_f16, &b_f16, &da_buf, m as u32, k as u32, n as u32);
    accumulate_grad(ctx, entry.inputs[0], &da_buf, m * k);

    // dB = A^T @ dC : [K, M] @ [M, N] = [K, N]
    let db_buf = ctx.alloc_buffer(k * n * 4);
    compute::gpu_matmul_trans_a_f16(ctx, &a_f16, &grad_f16, &db_buf, m as u32, k as u32, n as u32);
    accumulate_grad(ctx, entry.inputs[1], &db_buf, k * n);
}

fn backward_matmul_trans_b(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    // C = A @ B^T where A:[M,K], B:[N,K], C:[M,N]
    // dA = dC @ B : [M,N] @ [N,K] = [M,K]
    // dB = dC^T @ A : [N,M] @ [M,K] = [N,K]  — but dC^T @ A = (A^T @ dC)^T hmm
    // Actually: dB_ij = sum_m dC_mi * A_mj → dB = dC^T @ A
    // matmul_trans_b(dC, B_transposed_back) ... this gets complicated. Let's do it directly.

    let a_shape = &entry.shapes[0];
    let b_shape = &entry.shapes[1];
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];

    // FP16 backward with clamped casts
    let grad_f16 = cast_buf_f16(ctx, out_grad, m * n);
    let b_f16 = cast_buf_f16(ctx, &entry.input_buffers[1], n * k);
    let a_f16 = cast_buf_f16(ctx, &entry.input_buffers[0], m * k);

    // dA = dC @ B : [M,N] @ [N,K] = [M,K]
    let da_buf = ctx.alloc_buffer(m * k * 4);
    compute::gpu_matmul_f16(ctx, &grad_f16, &b_f16, &da_buf, m as u32, k as u32, n as u32);
    accumulate_grad(ctx, entry.inputs[0], &da_buf, m * k);

    // dB = dC^T @ A : [N,M] @ [M,K] = [N,K]
    let db_buf = ctx.alloc_buffer(n * k * 4);
    compute::gpu_matmul_trans_a_f16(ctx, &grad_f16, &a_f16, &db_buf, m as u32, n as u32, k as u32);
    accumulate_grad(ctx, entry.inputs[1], &db_buf, n * k);
}

fn backward_add(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    // d(A + B) = dA = grad, dB = grad
    // SHARED: same out_grad passed to both inputs — must copy on first insert
    let size: usize = entry.shapes[0].iter().product();
    accumulate_grad_shared(ctx, entry.inputs[0], out_grad, size);
    accumulate_grad_shared(ctx, entry.inputs[1], out_grad, size);
}

fn backward_mul(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    // d(A * B) → dA = grad * B, dB = grad * A
    let size: usize = entry.shapes[0].iter().product();

    let da_buf = ctx.alloc_buffer(size * 4);
    compute::gpu_mul(ctx, out_grad, &entry.input_buffers[1], &da_buf, size as u32);
    accumulate_grad(ctx, entry.inputs[0], &da_buf, size);

    let db_buf = ctx.alloc_buffer(size * 4);
    compute::gpu_mul(ctx, out_grad, &entry.input_buffers[0], &db_buf, size as u32);
    accumulate_grad(ctx, entry.inputs[1], &db_buf, size);
}

fn backward_softmax(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    let shape = &entry.shapes[0];
    let cols = *shape.last().unwrap();
    let rows: usize = shape.iter().product::<usize>() / cols;

    let softmax_out = entry.cached.as_ref().expect("softmax backward needs cached output");
    let grad_input = ctx.alloc_buffer(rows * cols * 4);
    compute::gpu_softmax_backward(ctx, softmax_out, out_grad, &grad_input, rows as u32, cols as u32);
    accumulate_grad(ctx, entry.inputs[0], &grad_input, rows * cols);
}

fn backward_rms_norm(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>, eps: f32) {
    let input_shape = &entry.shapes[0];
    let cols = *input_shape.last().unwrap();
    let rows: usize = input_shape.iter().product::<usize>() / cols;

    let grad_input = ctx.alloc_buffer(rows * cols * 4);
    let grad_weight = ctx.alloc_buffer(cols * 4);

    compute::gpu_rms_norm_backward(
        ctx,
        &entry.input_buffers[0], // input
        &entry.input_buffers[1], // weight
        out_grad,
        &grad_input,
        &grad_weight,
        &compute::RmsNormBackwardParams {
            rows: rows as u32,
            cols: cols as u32,
            eps,
        },
    );

    accumulate_grad(ctx, entry.inputs[0], &grad_input, rows * cols);
    accumulate_grad(ctx, entry.inputs[1], &grad_weight, cols);
}

/// Backward for fused residual+RMSNorm: rms_norm(a + b, weight, eps)
/// The cached buffer stores (a+b), which is the effective "input" to rms_norm.
/// Gradients flow equally to both a and b.
fn backward_rms_norm_residual(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>, eps: f32) {
    let input_shape = &entry.shapes[0]; // shape of a (and b)
    let cols = *input_shape.last().unwrap();
    let rows: usize = input_shape.iter().product::<usize>() / cols;

    let grad_sum = ctx.alloc_buffer(rows * cols * 4);
    let grad_weight = ctx.alloc_buffer(cols * 4);

    // The cached buffer is (a+b), which was the effective input to rms_norm
    let sum_buf = entry.cached.as_ref().expect("RmsNormResidual requires cached (a+b) buffer");

    compute::gpu_rms_norm_backward(
        ctx,
        sum_buf,                    // input to rms_norm was (a+b)
        &entry.input_buffers[2],    // weight
        out_grad,
        &grad_sum,
        &grad_weight,
        &compute::RmsNormBackwardParams {
            rows: rows as u32,
            cols: cols as u32,
            eps,
        },
    );

    // grad flows equally to both a and b
    // SHARED: same grad_sum buffer — must copy on first insert
    accumulate_grad_shared(ctx, entry.inputs[0], &grad_sum, rows * cols);
    accumulate_grad_shared(ctx, entry.inputs[1], &grad_sum, rows * cols);
    accumulate_grad(ctx, entry.inputs[2], &grad_weight, cols);
}

fn backward_silu(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    let size: usize = entry.shapes[0].iter().product();
    let grad_input = ctx.alloc_buffer(size * 4);
    compute::gpu_silu_backward(ctx, &entry.input_buffers[0], out_grad, &grad_input, size as u32);
    accumulate_grad(ctx, entry.inputs[0], &grad_input, size);
}

fn backward_silu_gate(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    let size: usize = entry.shapes[0].iter().product();
    let grad_gate = ctx.alloc_buffer(size * 4);
    let grad_up = ctx.alloc_buffer(size * 4);
    compute::gpu_silu_gate_backward(
        ctx,
        &entry.input_buffers[0], // gate
        &entry.input_buffers[1], // up
        out_grad,
        &grad_gate,
        &grad_up,
        size as u32,
    );
    accumulate_grad(ctx, entry.inputs[0], &grad_gate, size);
    accumulate_grad(ctx, entry.inputs[1], &grad_up, size);
}

fn backward_reshape(_ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    // Reshape backward: just pass the gradient through (same data, different shape)
    let size: usize = entry.shapes[0].iter().product();
    accumulate_grad(_ctx, entry.inputs[0], out_grad, size);
}

fn backward_embedding(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    // entry.input_buffers[0] = tokens, input_buffers[1] = embedding matrix
    // shapes[0] = tokens shape, shapes[1] = embedding shape [vocab, dim]
    let vocab = entry.shapes[1][0];
    let dim = entry.shapes[1][1];
    let n_tokens: usize = entry.shapes[0].iter().product();

    let grad_embeddings = ctx.alloc_buffer(vocab * dim * 4);
    compute::gpu_embedding_backward(
        ctx,
        &entry.input_buffers[0], // tokens
        out_grad,
        &grad_embeddings,
        n_tokens as u32,
        dim as u32,
    );
    accumulate_grad(ctx, entry.inputs[1], &grad_embeddings, vocab * dim);
}

fn backward_scale(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>, factor: f32) {
    let size: usize = entry.shapes[0].iter().product();
    let grad_input = ctx.alloc_buffer(size * 4);
    compute::gpu_copy(ctx, out_grad, &grad_input, size as u32);
    compute::gpu_scale(ctx, &grad_input, size as u32, factor);
    accumulate_grad(ctx, entry.inputs[0], &grad_input, size);
}

/// RoPE backward: apply inverse rotation to propagate gradients through RoPE.
/// The forward pass rotates by angle θ, so backward rotates by -θ.
fn backward_rope(
    ctx: &Arc<MetalContext>,
    entry: &TapeEntry,
    out_grad: &Retained<GpuBuffer>,
    seq_len: u32,
    head_dim: u32,
    offset: u32,
    theta: f32,
) {
    let size: usize = entry.shapes[0].iter().product();
    let total_rows = entry.shapes[0][0] as u32;
    let grad_input = ctx.alloc_buffer(size * 4);
    compute::gpu_copy(ctx, out_grad, &grad_input, size as u32);
    compute::gpu_rope_backward(ctx, &grad_input, total_rows, seq_len, head_dim, offset, theta);
    accumulate_grad(ctx, entry.inputs[0], &grad_input, size);
}

/// Checkpoint backward: re-run the forward pass to recover the sub-tape,
/// inject the output gradient, walk the sub-tape in reverse, and extract
/// the input gradient for the original input tensor.
fn backward_checkpoint(
    ctx: &Arc<MetalContext>,
    entry: &TapeEntry,
    out_grad: &Retained<GpuBuffer>,
    layer_idx: usize,
) {
    // Temporarily remove the recompute fn from the registry so we can call it
    // without holding a RefCell borrow (the fn itself will borrow TAPE via
    // checkpoint_forward). We put it back after calling.
    let recompute = RECOMPUTE_REGISTRY.with(|reg| reg.borrow_mut().remove(&layer_idx))
        .expect("missing recompute fn for checkpoint layer");

    // Re-run the forward pass to get the sub-tape
    let sub_tape = recompute(ctx);

    // Put the recompute fn back for potential future use (e.g., gradient accumulation)
    RECOMPUTE_REGISTRY.with(|reg| reg.borrow_mut().insert(layer_idx, recompute));

    // The sub-tape's last entry's output should correspond to entry.output
    // (the checkpoint output tensor). Inject the output gradient for it.
    if let Some(last_sub_entry) = sub_tape.last() {
        let sub_output_id = last_sub_entry.output;
        let output_size: usize = last_sub_entry.shapes.last()
            .map(|s| s.iter().product())
            .unwrap_or_else(|| {
                entry.shapes.last().map(|s| s.iter().product()).unwrap_or(0)
            });
        accumulate_grad(ctx, sub_output_id, out_grad, output_size);
    }

    // Save first entry's input info before consuming the tape
    let first_input_id = sub_tape.first().map(|e| e.inputs[0]);
    let input_size: usize = entry.shapes[0].iter().product();

    // Walk the sub-tape in reverse, CONSUMING entries to free GPU buffers incrementally.
    // This is critical: without this, all intermediate buffers stay pinned until the
    // entire sub-tape is processed, causing memory pressure on 16GB devices.
    let mut sub_tape_rev: Vec<TapeEntry> = sub_tape;
    sub_tape_rev.reverse();
    for sub_entry in sub_tape_rev.drain(..) {
        let sub_out_grad = GRADS.with(|grads| grads.borrow().get(&sub_entry.output).cloned());
        let sub_out_grad = match sub_out_grad {
            Some(g) => g,
            None => continue,
        };

        match &sub_entry.op {
            Op::Matmul => backward_matmul(ctx, &sub_entry, &sub_out_grad),
            Op::MatmulTransB => backward_matmul_trans_b(ctx, &sub_entry, &sub_out_grad),
            Op::Add => backward_add(ctx, &sub_entry, &sub_out_grad),
            Op::Mul => backward_mul(ctx, &sub_entry, &sub_out_grad),
            Op::Softmax => backward_softmax(ctx, &sub_entry, &sub_out_grad),
            Op::RmsNorm { eps } => backward_rms_norm(ctx, &sub_entry, &sub_out_grad, *eps),
            Op::RmsNormResidual { eps } => backward_rms_norm_residual(ctx, &sub_entry, &sub_out_grad, *eps),
            Op::RepeatKv { n_kv_heads, group_size, head_block } => {
                backward_repeat_kv(ctx, &sub_entry, &sub_out_grad, *n_kv_heads, *group_size, *head_block);
            }
            Op::ScaleRows { rows, cols } => {
                backward_scale_rows(ctx, &sub_entry, &sub_out_grad, *rows, *cols);
            }
            Op::FlashAttention { batch_heads, seq_q, seq_k, head_dim, kv_offset } => {
                backward_flash_attention(ctx, &sub_entry, &sub_out_grad, *batch_heads, *seq_q, *seq_k, *head_dim, *kv_offset);
            }
            Op::Silu => backward_silu(ctx, &sub_entry, &sub_out_grad),
            Op::SiluGate => backward_silu_gate(ctx, &sub_entry, &sub_out_grad),
            Op::Reshape => backward_reshape(ctx, &sub_entry, &sub_out_grad),
            Op::CrossEntropy => {
                if let Some(grad_logits) = &sub_entry.cached {
                    let size: usize = sub_entry.shapes[0].iter().product();
                    accumulate_grad(ctx, sub_entry.inputs[0], grad_logits, size);
                }
            }
            Op::Embedding => backward_embedding(ctx, &sub_entry, &sub_out_grad),
            Op::Scale { factor } => backward_scale(ctx, &sub_entry, &sub_out_grad, *factor),
            Op::Transpose { batch, seq_len, n_heads, head_dim, forward_dir } => {
                backward_transpose(ctx, &sub_entry, &sub_out_grad, &TransposeParams {
                    batch: *batch, seq_len: *seq_len, n_heads: *n_heads, head_dim: *head_dim, forward_dir: *forward_dir,
                });
            }
            Op::Checkpoint { layer_idx: nested_idx } => {
                backward_checkpoint(ctx, &sub_entry, &sub_out_grad, *nested_idx);
            }
            Op::Slice { offset, length, source_size } => {
                backward_slice(ctx, &sub_entry, &sub_out_grad, *offset, *length, *source_size);
            }
            Op::ConcatParts { part_sizes } => {
                backward_concat_parts(ctx, &sub_entry, &sub_out_grad, part_sizes);
            }
            Op::BatchedMatmul => {
                backward_batched_matmul(ctx, &sub_entry, &sub_out_grad);
            }
            Op::BatchedMatmulTransB => {
                backward_batched_matmul_trans_b(ctx, &sub_entry, &sub_out_grad);
            }
            Op::RoPE { seq_len, head_dim, offset, theta } => {
                backward_rope(ctx, &sub_entry, &sub_out_grad, *seq_len, *head_dim, *offset, *theta);
            }
        }
    }

    // Extract gradient for the checkpoint's input tensor.
    // We saved first_input_id before consuming the sub-tape.
    if let Some(sub_input_id) = first_input_id {
        if let Some(input_grad) = GRADS.with(|grads| grads.borrow().get(&sub_input_id).cloned()) {
            accumulate_grad(ctx, entry.inputs[0], &input_grad, input_size);
        }
    }

    // Clean up sub-tape gradients for intermediate tensors to free memory.
    // We keep gradients for tensor IDs that match the checkpoint entry's inputs
    // (already accumulated above) and any parameter tensor IDs (which are in the
    // main GRADS map and were accumulated by the sub-tape backward ops).
    // The sub-tape intermediate IDs can be dropped.
    // Note: We don't explicitly clean up here because the sub-tape Vec is dropped
    // at end of scope, and the GRADS entries for sub-tape intermediates will be
    // overwritten or cleared when clear_tape() is called at the end of the step.
}

/// Parameters for the transpose backward pass.
pub struct TransposeParams {
    pub batch: usize,
    pub seq_len: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub forward_dir: bool,
}

/// Transpose backward: the inverse permutation.
/// Forward: [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim]
/// Backward: apply the reverse permutation to the gradient.
fn backward_transpose(
    ctx: &Arc<MetalContext>,
    entry: &TapeEntry,
    out_grad: &Retained<GpuBuffer>,
    tp: &TransposeParams,
) {
    let TransposeParams { batch, seq_len, n_heads, head_dim, forward_dir } = *tp;
    let size = batch * seq_len * n_heads * head_dim;

    let grad_buf = if forward_dir {
        // Forward was bsh→bhs. Backward is bhs→bsh.
        // out_grad is [batch*n_heads, seq, head_dim], need [batch*seq, n_heads*head_dim]
        let output = ctx.alloc_buffer(size * 4);
        compute::gpu_transpose_perm_backward(
            ctx,
            out_grad,
            &output,
            batch as u32,
            seq_len as u32,
            n_heads as u32,
            head_dim as u32,
        );
        output
    } else {
        // Forward was bhs→bsh. Backward is bsh→bhs.
        // out_grad is [batch*seq, n_heads*head_dim], need [batch*n_heads, seq, head_dim]
        // Use the forward transpose kernel (bsh→bhs is exactly the forward permutation)
        let output = ctx.alloc_buffer(size * 4);
        compute::gpu_transpose_perm_forward(
            ctx,
            out_grad,
            &output,
            batch as u32,
            seq_len as u32,
            n_heads as u32,
            head_dim as u32,
        );
        output
    };

    accumulate_grad(ctx, entry.inputs[0], &grad_buf, size);
}

/// Slice backward: scatter gradient back into the source tensor's gradient at the correct offset.
fn backward_slice(
    ctx: &Arc<MetalContext>,
    entry: &TapeEntry,
    out_grad: &Retained<GpuBuffer>,
    offset: usize,
    length: usize,
    source_size: usize,
) {
    // Create a zero buffer the size of the source, then copy the slice gradient into it at offset
    let grad_source = ctx.alloc_buffer(source_size * 4);
    compute::gpu_fill(ctx, &grad_source, source_size as u32, 0.0);

    // Copy out_grad (length elements) into grad_source at offset — fully on GPU
    compute::gpu_buffer_copy(ctx, out_grad, &grad_source, 0, offset as u32, length as u32);

    accumulate_grad(ctx, entry.inputs[0], &grad_source, source_size);
}

/// ConcatParts backward: split the gradient and distribute to each input part.
fn backward_concat_parts(
    ctx: &Arc<MetalContext>,
    entry: &TapeEntry,
    out_grad: &Retained<GpuBuffer>,
    part_sizes: &[usize],
) {
    let total: usize = part_sizes.iter().sum();
    let expected: usize = entry.shapes.last().map(|s| s.iter().product()).unwrap_or(0);
    assert_eq!(total, expected, "backward_concat_parts: part_sizes sum {} != output size {}", total, expected);
    let mut offset = 0u32;
    for (i, &part_size) in part_sizes.iter().enumerate() {
        let part_grad = ctx.alloc_buffer(part_size * 4);
        compute::gpu_buffer_copy(ctx, out_grad, &part_grad, offset, 0, part_size as u32);
        accumulate_grad(ctx, entry.inputs[i], &part_grad, part_size);
        offset += part_size as u32;
    }
}

/// BatchedMatmul backward: C[b] = A[b] @ B[b]
/// dA[b] = dC[b] @ B[b]^T, dB[b] = A[b]^T @ dC[b]
fn backward_batched_matmul(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    let a_shape = &entry.shapes[0]; // [B, M, K]
    let b_shape = &entry.shapes[1]; // [B, K, N]

    let batches = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    // Batched backward — FP32 (small attention dims, cast overhead not worth it)
    // dA = dC @ B^T : [B,M,N] @ [B,N,K] = [B,M,K]
    let da_total = ctx.alloc_buffer(batches * m * k * 4);
    compute::gpu_batched_matmul_trans_b(ctx, out_grad, &entry.input_buffers[1], &da_total, batches as u32, m as u32, k as u32, n as u32);

    // dB = A^T @ dC : [B,K,M] @ [B,M,N] = [B,K,N]
    let db_total = ctx.alloc_buffer(batches * k * n * 4);
    compute::gpu_batched_matmul_trans_a(ctx, &entry.input_buffers[0], out_grad, &db_total, batches as u32, m as u32, k as u32, n as u32);

    accumulate_grad(ctx, entry.inputs[0], &da_total, batches * m * k);
    accumulate_grad(ctx, entry.inputs[1], &db_total, batches * k * n);
}

/// BatchedMatmulTransB backward: C[b] = A[b] @ B[b]^T
/// dA[b] = dC[b] @ B[b], dB[b] = dC[b]^T @ A[b]
fn backward_batched_matmul_trans_b(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    let a_shape = &entry.shapes[0]; // [B, M, K]
    let b_shape = &entry.shapes[1]; // [B, N, K]

    let batches = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[1];

    // Batched backward — FP32
    // dA = dC @ B : [B,M,N] @ [B,N,K] = [B,M,K]
    let da_total = ctx.alloc_buffer(batches * m * k * 4);
    compute::gpu_batched_matmul(ctx, out_grad, &entry.input_buffers[1], &da_total, batches as u32, m as u32, k as u32, n as u32);

    // dB = dC^T @ A : [B,N,M] @ [B,M,K] = [B,N,K]
    let db_total = ctx.alloc_buffer(batches * n * k * 4);
    compute::gpu_batched_matmul_trans_a(ctx, out_grad, &entry.input_buffers[0], &db_total, batches as u32, m as u32, n as u32, k as u32);

    accumulate_grad(ctx, entry.inputs[0], &da_total, batches * m * k);
    accumulate_grad(ctx, entry.inputs[1], &db_total, batches * n * k);
}

/// RepeatKv backward: sum group_size gradient blocks back into each KV head.
/// out_grad: [n_kv_heads * group_size, seq, head_dim], input grad: [n_kv_heads, seq, head_dim]
fn backward_repeat_kv(
    ctx: &Arc<MetalContext>,
    entry: &TapeEntry,
    out_grad: &Retained<GpuBuffer>,
    n_kv_heads: usize,
    group_size: usize,
    head_block: usize,
) {
    // For each KV head, sum the gradients from its group_size copies
    let kv_grad = ctx.alloc_buffer(n_kv_heads * head_block * 4);

    for h in 0..n_kv_heads {
        // First copy: initialize the KV head's gradient with the first group member
        let src_offset = (h * group_size) * head_block;
        let dst_offset = h * head_block;
        compute::gpu_buffer_copy(ctx, out_grad, &kv_grad, src_offset as u32, dst_offset as u32, head_block as u32);

        // Remaining copies: add (in-place) the other group members
        for g in 1..group_size {
            let src_offset = (h * group_size + g) * head_block;
            // Need a temp buffer to add from
            let tmp = ctx.alloc_buffer(head_block * 4);
            compute::gpu_buffer_copy(ctx, out_grad, &tmp, src_offset as u32, 0, head_block as u32);
            // Add tmp into kv_grad at the right offset — use a sub-buffer view
            let dst_sub = ctx.alloc_buffer(head_block * 4);
            compute::gpu_buffer_copy(ctx, &kv_grad, &dst_sub, dst_offset as u32, 0, head_block as u32);
            compute::gpu_add_inplace(ctx, &dst_sub, &tmp, head_block as u32);
            compute::gpu_buffer_copy(ctx, &dst_sub, &kv_grad, 0, dst_offset as u32, head_block as u32);
        }
    }

    accumulate_grad(ctx, entry.inputs[0], &kv_grad, n_kv_heads * head_block);
}

/// Flash Attention backward: compute dQ, dK, dV using Flash Attention backward kernel.
/// entry.inputs: [Q_id, K_id, V_id]
/// entry.cached: O (forward output, for D computation)
fn backward_flash_attention(
    ctx: &Arc<MetalContext>,
    entry: &TapeEntry,
    out_grad: &Retained<GpuBuffer>, // dO
    batch_heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    kv_offset: u32,
) {
    let total_q_rows = batch_heads * seq_q;
    let total_k_rows = batch_heads * seq_k;

    // Get O from cached (saved during forward)
    let o_buf = entry.cached.as_ref().expect("FlashAttention backward requires cached O buffer");

    // Precompute D[i] = sum_j(dO[i][j] * O[i][j])
    let d_buf = ctx.alloc_buffer(total_q_rows * 4);
    compute::gpu_flash_attn_precompute_d(ctx, out_grad, o_buf, &d_buf, total_q_rows as u32, head_dim as u32);

    // Allocate gradient buffers
    let dq_buf = ctx.alloc_buffer(total_q_rows * head_dim * 4);
    let dk_buf = ctx.alloc_buffer(total_k_rows * head_dim * 4);
    let dv_buf = ctx.alloc_buffer(total_k_rows * head_dim * 4);

    // Zero dK and dV (they accumulate from multiple query rows)
    compute::gpu_fill(ctx, &dk_buf, (total_k_rows * head_dim) as u32, 0.0);
    compute::gpu_fill(ctx, &dv_buf, (total_k_rows * head_dim) as u32, 0.0);

    // Run Flash Attention backward kernel
    compute::gpu_flash_attention_backward(
        ctx,
        &entry.input_buffers[0], // Q
        &entry.input_buffers[1], // K
        &entry.input_buffers[2], // V
        o_buf,                    // O
        out_grad,                 // dO
        &d_buf,                   // D
        &dq_buf, &dk_buf, &dv_buf,
        batch_heads as u32, seq_q as u32, seq_k as u32, head_dim as u32, kv_offset,
    );

    // Accumulate gradients
    accumulate_grad(ctx, entry.inputs[0], &dq_buf, total_q_rows * head_dim); // dQ
    accumulate_grad(ctx, entry.inputs[1], &dk_buf, total_k_rows * head_dim); // dK
    accumulate_grad(ctx, entry.inputs[2], &dv_buf, total_k_rows * head_dim); // dV
}

/// ScaleRows backward: d_input = d_out * scales (per-row), d_scales = rowsum(d_out * input)
fn backward_scale_rows(
    ctx: &Arc<MetalContext>,
    entry: &TapeEntry,
    out_grad: &Retained<GpuBuffer>,
    rows: usize,
    cols: usize,
) {
    // d_input[r][c] = d_out[r][c] * scales[r]
    let d_input = ctx.alloc_buffer(rows * cols * 4);
    compute::gpu_scale_rows(ctx, out_grad, &entry.input_buffers[1], &d_input, rows as u32, cols as u32);
    accumulate_grad(ctx, entry.inputs[0], &d_input, rows * cols);

    // d_scales[r] = sum_c(d_out[r][c] * input[r][c]) — single GPU dispatch
    let d_scales = ctx.alloc_buffer(rows * 4);
    compute::gpu_row_dot_reduce(ctx, out_grad, &entry.input_buffers[0], &d_scales, rows as u32, cols as u32);
    accumulate_grad(ctx, entry.inputs[1], &d_scales, rows);
}
