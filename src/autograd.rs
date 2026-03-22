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
}

/// A single entry on the autodiff tape.
pub struct TapeEntry {
    pub op: Op,
    pub inputs: Vec<usize>,     // TensorIds of inputs
    pub output: usize,          // TensorId of output
    pub input_buffers: Vec<Retained<GpuBuffer>>,
    pub output_buffer: Retained<GpuBuffer>,
    pub shapes: Vec<Vec<usize>>,
    pub cached: Option<Retained<GpuBuffer>>, // Cached forward-pass data for backward
}

thread_local! {
    static TAPE: RefCell<Vec<TapeEntry>> = RefCell::new(Vec::new());
    static GRADS: RefCell<HashMap<usize, Retained<GpuBuffer>>> = RefCell::new(HashMap::new());
    static NO_GRAD: RefCell<bool> = RefCell::new(false);
    static RECOMPUTE_REGISTRY: RefCell<HashMap<usize, RecomputeFn>> = RefCell::new(HashMap::new());
}

/// Check if we're currently recording ops.
pub fn is_recording() -> bool {
    NO_GRAD.with(|ng| !*ng.borrow())
}

/// Record an operation on the tape.
pub fn record(entry: TapeEntry) {
    if is_recording() {
        TAPE.with(|tape| tape.borrow_mut().push(entry));
    }
}

/// Clear the tape and all stored gradients.
pub fn clear_tape() {
    TAPE.with(|tape| tape.borrow_mut().clear());
    GRADS.with(|grads| grads.borrow_mut().clear());
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
fn accumulate_grad(ctx: &Arc<MetalContext>, tensor_id: usize, grad: &Retained<GpuBuffer>, size: usize) {
    GRADS.with(|grads| {
        let mut grads = grads.borrow_mut();
        if let Some(existing) = grads.get(&tensor_id) {
            // Accumulate: existing += grad
            let out = ctx.alloc_buffer(size * 4);
            compute::gpu_add(ctx, existing, grad, &out, size as u32);
            grads.insert(tensor_id, out);
        } else {
            grads.insert(tensor_id, grad.clone());
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
                Op::Silu => {
                    backward_silu(ctx, entry, &out_grad);
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
                    backward_transpose(ctx, entry, &out_grad, *batch, *seq_len, *n_heads, *head_dim, *forward_dir);
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
            }
        }
    }
    // Restore the tape (in case anything needs it later, though typically clear_tape is called)
    TAPE.with(|t| *t.borrow_mut() = tape);
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

    // dA = dC @ B^T : [M, N] @ [N, K] = [M, K]
    let da_buf = ctx.alloc_buffer(m * k * 4);
    compute::gpu_matmul_trans_b(ctx, out_grad, &entry.input_buffers[1], &da_buf, m as u32, k as u32, n as u32);
    accumulate_grad(ctx, entry.inputs[0], &da_buf, m * k);

    // dB = A^T @ dC : [K, M] @ [M, N] = [K, N]
    // A^T @ dC is equivalent to dC^T @ A transposed... let's use the identity:
    // dB = A^T @ dC. We need matmul with A transposed.
    // matmul_trans_b computes A @ B^T, but we need A^T @ B.
    // A^T @ dC = (dC^T @ A)^T. For 2D: we can do matmul_trans_b(dC, A) which gives dC @ A^T = [N,M]@...
    // Actually let's just do it differently: dB[i,j] = sum_m A[m,i] * dC[m,j]
    // This is the same as: dB = matmul(A^T, dC) where A^T is [K, M]
    // We can use matmul_trans_b(dC^T, ...) — no, let's keep it simple with a transposed-A matmul.
    // For now, we'll read and transpose A on CPU, then matmul.
    let a_data = MetalContext::read_buffer(&entry.input_buffers[0], m * k);
    let mut a_t = vec![0.0f32; k * m];
    for r in 0..m {
        for c in 0..k {
            a_t[c * m + r] = a_data[r * k + c];
        }
    }
    let a_t_buf = ctx.buffer_from_slice(&a_t);
    let db_buf = ctx.alloc_buffer(k * n * 4);
    compute::gpu_matmul(ctx, &a_t_buf, out_grad, &db_buf, k as u32, n as u32, m as u32);
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

    // dA = dC @ B : [M,N] @ [N,K] = [M,K]
    let da_buf = ctx.alloc_buffer(m * k * 4);
    compute::gpu_matmul(ctx, out_grad, &entry.input_buffers[1], &da_buf, m as u32, k as u32, n as u32);
    accumulate_grad(ctx, entry.inputs[0], &da_buf, m * k);

    // dB = dC^T @ A : [N,M] @ [M,K] = [N,K]
    // Read dC, transpose, then matmul
    let dc_data = MetalContext::read_buffer(out_grad, m * n);
    let mut dc_t = vec![0.0f32; n * m];
    for r in 0..m {
        for c in 0..n {
            dc_t[c * m + r] = dc_data[r * n + c];
        }
    }
    let dc_t_buf = ctx.buffer_from_slice(&dc_t);
    let db_buf = ctx.alloc_buffer(n * k * 4);
    compute::gpu_matmul(ctx, &dc_t_buf, &entry.input_buffers[0], &db_buf, n as u32, k as u32, m as u32);
    accumulate_grad(ctx, entry.inputs[1], &db_buf, n * k);
}

fn backward_add(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    // d(A + B) = dA = grad, dB = grad
    let size: usize = entry.shapes[0].iter().product();
    accumulate_grad(ctx, entry.inputs[0], out_grad, size);
    accumulate_grad(ctx, entry.inputs[1], out_grad, size);
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
        rows as u32,
        cols as u32,
        eps,
    );

    accumulate_grad(ctx, entry.inputs[0], &grad_input, rows * cols);
    accumulate_grad(ctx, entry.inputs[1], &grad_weight, cols);
}

fn backward_silu(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    let size: usize = entry.shapes[0].iter().product();
    let grad_input = ctx.alloc_buffer(size * 4);
    compute::gpu_silu_backward(ctx, &entry.input_buffers[0], out_grad, &grad_input, size as u32);
    accumulate_grad(ctx, entry.inputs[0], &grad_input, size);
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
        vocab as u32,
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
                // Fallback: use the checkpoint entry's output shape
                entry.shapes.last().map(|s| s.iter().product()).unwrap_or(0)
            });
        accumulate_grad(ctx, sub_output_id, out_grad, output_size);
    }

    // Walk the sub-tape in reverse, computing gradients for each op
    for sub_entry in sub_tape.iter().rev() {
        let sub_out_grad = GRADS.with(|grads| grads.borrow().get(&sub_entry.output).cloned());
        let sub_out_grad = match sub_out_grad {
            Some(g) => g,
            None => continue,
        };

        match &sub_entry.op {
            Op::Matmul => backward_matmul(ctx, sub_entry, &sub_out_grad),
            Op::MatmulTransB => backward_matmul_trans_b(ctx, sub_entry, &sub_out_grad),
            Op::Add => backward_add(ctx, sub_entry, &sub_out_grad),
            Op::Mul => backward_mul(ctx, sub_entry, &sub_out_grad),
            Op::Softmax => backward_softmax(ctx, sub_entry, &sub_out_grad),
            Op::RmsNorm { eps } => backward_rms_norm(ctx, sub_entry, &sub_out_grad, *eps),
            Op::Silu => backward_silu(ctx, sub_entry, &sub_out_grad),
            Op::Reshape => backward_reshape(ctx, sub_entry, &sub_out_grad),
            Op::CrossEntropy => {
                if let Some(grad_logits) = &sub_entry.cached {
                    let size: usize = sub_entry.shapes[0].iter().product();
                    accumulate_grad(ctx, sub_entry.inputs[0], grad_logits, size);
                }
            }
            Op::Embedding => backward_embedding(ctx, sub_entry, &sub_out_grad),
            Op::Scale { factor } => backward_scale(ctx, sub_entry, &sub_out_grad, *factor),
            Op::Transpose { batch, seq_len, n_heads, head_dim, forward_dir } => {
                backward_transpose(ctx, sub_entry, &sub_out_grad, *batch, *seq_len, *n_heads, *head_dim, *forward_dir);
            }
            Op::Checkpoint { layer_idx: nested_idx } => {
                backward_checkpoint(ctx, sub_entry, &sub_out_grad, *nested_idx);
            }
            Op::Slice { offset, length, source_size } => {
                backward_slice(ctx, sub_entry, &sub_out_grad, *offset, *length, *source_size);
            }
            Op::ConcatParts { part_sizes } => {
                backward_concat_parts(ctx, sub_entry, &sub_out_grad, part_sizes);
            }
            Op::BatchedMatmul => {
                backward_batched_matmul(ctx, sub_entry, &sub_out_grad);
            }
            Op::BatchedMatmulTransB => {
                backward_batched_matmul_trans_b(ctx, sub_entry, &sub_out_grad);
            }
        }
    }

    // The first sub-tape entry's first input should be the checkpoint's input tensor.
    // Extract its gradient and accumulate it for the original input tensor ID.
    if let Some(first_sub_entry) = sub_tape.first() {
        let sub_input_id = first_sub_entry.inputs[0];
        if let Some(input_grad) = GRADS.with(|grads| grads.borrow().get(&sub_input_id).cloned()) {
            let input_size: usize = entry.shapes[0].iter().product();
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

/// Transpose backward: the inverse permutation.
/// Forward: [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim]
/// Backward: apply the reverse permutation to the gradient.
fn backward_transpose(
    ctx: &Arc<MetalContext>,
    entry: &TapeEntry,
    out_grad: &Retained<GpuBuffer>,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    forward_dir: bool,
) {
    let size = batch * seq_len * n_heads * head_dim;
    let grad_data = MetalContext::read_buffer(out_grad, size);
    let mut grad_input = vec![0.0f32; size];

    if forward_dir {
        // Forward was bsh→bhs. Backward is bhs→bsh.
        // out_grad is [batch*n_heads, seq, head_dim], need [batch*seq, n_heads*head_dim]
        for b in 0..batch {
            for s in 0..seq_len {
                for h in 0..n_heads {
                    for d in 0..head_dim {
                        let src_idx = (b * n_heads + h) * seq_len * head_dim + s * head_dim + d;
                        let dst_idx = (b * seq_len + s) * n_heads * head_dim + h * head_dim + d;
                        grad_input[dst_idx] = grad_data[src_idx];
                    }
                }
            }
        }
    } else {
        // Forward was bhs→bsh. Backward is bsh→bhs.
        for b in 0..batch {
            for s in 0..seq_len {
                for h in 0..n_heads {
                    for d in 0..head_dim {
                        let src_idx = (b * seq_len + s) * n_heads * head_dim + h * head_dim + d;
                        let dst_idx = (b * n_heads + h) * seq_len * head_dim + s * head_dim + d;
                        grad_input[dst_idx] = grad_data[src_idx];
                    }
                }
            }
        }
    }

    let grad_buf = ctx.buffer_from_slice(&grad_input);
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

    // Copy out_grad (length elements) into grad_source at offset
    let grad_data = MetalContext::read_buffer(out_grad, length);
    unsafe {
        let dst = (grad_source.contents().as_ptr() as *mut f32).add(offset);
        std::ptr::copy_nonoverlapping(grad_data.as_ptr(), dst, length);
    }

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
    let grad_data = MetalContext::read_buffer(out_grad, total);

    let mut offset = 0;
    for (i, &part_size) in part_sizes.iter().enumerate() {
        let part_grad = ctx.buffer_from_slice(&grad_data[offset..offset + part_size]);
        accumulate_grad(ctx, entry.inputs[i], &part_grad, part_size);
        offset += part_size;
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

    let a_data = MetalContext::read_buffer(&entry.input_buffers[0], batches * m * k);
    let b_data = MetalContext::read_buffer(&entry.input_buffers[1], batches * k * n);
    let dc_data = MetalContext::read_buffer(out_grad, batches * m * n);

    let da_total = ctx.alloc_buffer(batches * m * k * 4);
    let db_total = ctx.alloc_buffer(batches * k * n * 4);

    for b in 0..batches {
        let dc_off = b * m * n;
        let a_off = b * m * k;
        let b_off = b * k * n;

        let dc_sub = ctx.buffer_from_slice(&dc_data[dc_off..dc_off + m * n]);
        let b_sub = ctx.buffer_from_slice(&b_data[b_off..b_off + k * n]);

        // dA[b] = dC[b] @ B[b]^T : [M, N] @ [N, K] = [M, K]
        let da_sub = ctx.alloc_buffer(m * k * 4);
        compute::gpu_matmul_trans_b(ctx, &dc_sub, &b_sub, &da_sub, m as u32, k as u32, n as u32);
        let da_vals = MetalContext::read_buffer(&da_sub, m * k);
        unsafe {
            let dst = (da_total.contents().as_ptr() as *mut f32).add(a_off);
            std::ptr::copy_nonoverlapping(da_vals.as_ptr(), dst, m * k);
        }

        // dB[b] = A[b]^T @ dC[b] : [K, M] @ [M, N] = [K, N]
        let a_sub_data = &a_data[a_off..a_off + m * k];
        let mut a_t = vec![0.0f32; k * m];
        for r in 0..m {
            for c in 0..k {
                a_t[c * m + r] = a_sub_data[r * k + c];
            }
        }
        let a_t_buf = ctx.buffer_from_slice(&a_t);
        let db_sub = ctx.alloc_buffer(k * n * 4);
        compute::gpu_matmul(ctx, &a_t_buf, &dc_sub, &db_sub, k as u32, n as u32, m as u32);
        let db_vals = MetalContext::read_buffer(&db_sub, k * n);
        unsafe {
            let dst = (db_total.contents().as_ptr() as *mut f32).add(b_off);
            std::ptr::copy_nonoverlapping(db_vals.as_ptr(), dst, k * n);
        }
    }

    accumulate_grad(ctx, entry.inputs[0], &da_total, batches * m * k);
    accumulate_grad(ctx, entry.inputs[1], &db_total, batches * k * n);
}

/// BatchedMatmulTransB backward: C[b] = A[b] @ B[b]^T
/// where A: [B, M, K], B: [B, N, K], C: [B, M, N]
/// dA[b] = dC[b] @ B[b] : [M, N] @ [N, K] = [M, K]
/// dB[b] = dC[b]^T @ A[b] : [N, M] @ [M, K] = [N, K]
fn backward_batched_matmul_trans_b(ctx: &Arc<MetalContext>, entry: &TapeEntry, out_grad: &Retained<GpuBuffer>) {
    let a_shape = &entry.shapes[0]; // [B, M, K]
    let b_shape = &entry.shapes[1]; // [B, N, K]

    let batches = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[1];

    let a_data = MetalContext::read_buffer(&entry.input_buffers[0], batches * m * k);
    let b_data = MetalContext::read_buffer(&entry.input_buffers[1], batches * n * k);
    let dc_data = MetalContext::read_buffer(out_grad, batches * m * n);

    let da_total = ctx.alloc_buffer(batches * m * k * 4);
    let db_total = ctx.alloc_buffer(batches * n * k * 4);

    for b in 0..batches {
        let dc_off = b * m * n;
        let a_off = b * m * k;
        let b_off = b * n * k;

        let dc_sub = ctx.buffer_from_slice(&dc_data[dc_off..dc_off + m * n]);
        let b_sub = ctx.buffer_from_slice(&b_data[b_off..b_off + n * k]);
        let a_sub = ctx.buffer_from_slice(&a_data[a_off..a_off + m * k]);

        // dA[b] = dC[b] @ B[b] : [M, N] @ [N, K] = [M, K]
        let da_sub = ctx.alloc_buffer(m * k * 4);
        compute::gpu_matmul(ctx, &dc_sub, &b_sub, &da_sub, m as u32, k as u32, n as u32);
        let da_vals = MetalContext::read_buffer(&da_sub, m * k);
        unsafe {
            let dst = (da_total.contents().as_ptr() as *mut f32).add(a_off);
            std::ptr::copy_nonoverlapping(da_vals.as_ptr(), dst, m * k);
        }

        // dB[b] = dC[b]^T @ A[b] : [N, M] @ [M, K] = [N, K]
        let dc_sub_data = &dc_data[dc_off..dc_off + m * n];
        let mut dc_t = vec![0.0f32; n * m];
        for r in 0..m {
            for c in 0..n {
                dc_t[c * m + r] = dc_sub_data[r * n + c];
            }
        }
        let dc_t_buf = ctx.buffer_from_slice(&dc_t);
        let db_sub = ctx.alloc_buffer(n * k * 4);
        compute::gpu_matmul(ctx, &dc_t_buf, &a_sub, &db_sub, n as u32, k as u32, m as u32);
        let db_vals = MetalContext::read_buffer(&db_sub, n * k);
        unsafe {
            let dst = (db_total.contents().as_ptr() as *mut f32).add(b_off);
            std::ptr::copy_nonoverlapping(db_vals.as_ptr(), dst, n * k);
        }
    }

    accumulate_grad(ctx, entry.inputs[0], &da_total, batches * m * k);
    accumulate_grad(ctx, entry.inputs[1], &db_total, batches * n * k);
}
