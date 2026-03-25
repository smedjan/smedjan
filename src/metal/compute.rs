use super::{GpuBuffer, GpuComputeEncoder, MetalContext};
use objc2_metal::{MTLComputeCommandEncoder as MTLComputeCommandEncoderTrait, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

/// Round up to the next power of 2, clamped to [1, 256].
/// Required for threadgroup reductions in all row-wise kernels.
#[inline]
fn next_power_of_2_clamped(n: u64) -> u64 {
    // Clamp to 256 BEFORE computing next_power_of_two to avoid u32 overflow
    // when n > 2^31 (e.g., large vocab sizes). Values > 256 always clamp to 256.
    let clamped = n.min(256).max(1) as u32;
    let p = clamped.next_power_of_two().min(256);
    p as u64
}

/// Create a params buffer from a repr(C) struct.
#[inline]
fn params_buffer<T>(ctx: &Arc<MetalContext>, params: &T) -> objc2::rc::Retained<GpuBuffer> {
    let ptr = NonNull::new(params as *const T as *mut c_void).unwrap();
    unsafe {
        ctx.device
            .newBufferWithBytes_length_options(
                ptr,
                std::mem::size_of::<T>(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("Failed to create params buffer")
    }
}

/// Helper to bind buffers to a compute encoder.
fn bind_buffer(encoder: &GpuComputeEncoder, buf: &GpuBuffer, index: usize) {
    unsafe { encoder.setBuffer_offset_atIndex(Some(buf), 0, index); }
}

/// Dispatch helper: encode compute command, set buffers, dispatch threadgroups.
/// Uses command batching when active (encode-only, no commit/wait).
/// Falls back to sync dispatch when no batch is active.
macro_rules! dispatch_sync {
    ($ctx:expr, $kernel:expr, $grid:expr, $tg:expr, $($idx:expr => $buf:expr),+ $(,)?) => {{
        let grid = $grid;
        let tg = $tg;
        $ctx.dispatch_kernel($kernel, grid, tg, false, |encoder| {
            $(bind_buffer(encoder, $buf, $idx);)+
        });
    }};
}

/// Dispatch using dispatchThreads (automatic threadgroup tiling by Metal).
/// Uses command batching when active.
macro_rules! dispatch_threads_sync {
    ($ctx:expr, $kernel:expr, $total:expr, $tg:expr, $($idx:expr => $buf:expr),+ $(,)?) => {{
        let total = $total;
        let tg = $tg;
        $ctx.dispatch_kernel($kernel, total, tg, true, |encoder| {
            $(bind_buffer(encoder, $buf, $idx);)+
        });
    }};
}

/// C = A @ B where A:[M,K], B:[K,N], C:[M,N]
pub fn gpu_matmul(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32 }
    let params = Params { m, n, k };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (m as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, 1);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "matmul_tiled", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// C = A @ B^T where A:[M,K], B:[N,K], C:[M,N]
pub fn gpu_matmul_trans_b(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32 }
    let params = Params { m, n, k };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (m as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, 1);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "matmul_tiled_trans_b", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// Batched C[b] = A[b] @ B[b] for all b in [0, batch). Single GPU dispatch.
/// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N]
pub fn gpu_batched_matmul(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, batch: u32, m: u32, n: u32, k: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32, batch: u32 }
    let params = Params { m, n, k, batch };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (m as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, batch as u64);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "batched_matmul_tiled", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// Batched C[b] = A[b] @ B[b]^T for all b. Single GPU dispatch.
/// A: [batch, M, K], B: [batch, N, K], C: [batch, M, N]
pub fn gpu_batched_matmul_trans_b(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, batch: u32, m: u32, n: u32, k: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32, batch: u32 }
    let params = Params { m, n, k, batch };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (m as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, batch as u64);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "batched_matmul_tiled_trans_b", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// Batched C[b] = A[b]^T @ B[b] for all b. Single GPU dispatch.
/// A: [batch, M, K] (transposed to [K,M]), B: [batch, M, N], C: [batch, K, N]
pub fn gpu_batched_matmul_trans_a(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, batch: u32, m: u32, k: u32, n: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32, batch: u32 }
    let params = Params { m, n, k, batch };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (k as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, batch as u64);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "batched_matmul_tiled_trans_a", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// Row-wise softmax. input/output: [rows, cols]
pub fn gpu_softmax(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32) {
    #[repr(C)]
    struct Params { rows: u32, cols: u32 }
    let params = Params { rows, cols };
    let params_buf = params_buffer(ctx, &params);

    let threads_per_group = next_power_of_2_clamped(cols as u64);
    let grid = MetalContext::size(rows as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);

    dispatch_sync!(ctx, "softmax", grid, tg,
        0 => input, 1 => output, 2 => &params_buf
    );
}

/// RMS layer normalization.
pub fn gpu_rms_norm(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    output: &GpuBuffer,
    rows: u32,
    cols: u32,
    eps: f32,
) {
    #[repr(C)]
    struct Params { rows: u32, cols: u32, eps: f32 }
    let params = Params { rows, cols, eps };
    let params_buf = params_buffer(ctx, &params);

    let threads_per_group = next_power_of_2_clamped(cols as u64);
    let grid = MetalContext::size(rows as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);

    dispatch_sync!(ctx, "rms_norm", grid, tg,
        0 => input, 1 => weight, 2 => output, 3 => &params_buf
    );
}

/// Fused residual add + RMS norm: output = rms_norm(input + residual, weight, eps)
/// Also stores (input + residual) in sum_out for backward pass.
#[allow(clippy::too_many_arguments)]
pub fn gpu_rms_norm_residual(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    residual: &GpuBuffer,
    weight: &GpuBuffer,
    output: &GpuBuffer,
    sum_out: &GpuBuffer,
    rows: u32,
    cols: u32,
    eps: f32,
) {
    #[repr(C)]
    struct Params { rows: u32, cols: u32, eps: f32 }
    let params = Params { rows, cols, eps };
    let params_buf = params_buffer(ctx, &params);

    let threads_per_group = next_power_of_2_clamped(cols as u64);
    let grid = MetalContext::size(rows as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);

    dispatch_sync!(ctx, "rms_norm_residual", grid, tg,
        0 => input, 1 => residual, 2 => weight, 3 => output, 4 => sum_out, 5 => &params_buf
    );
}

/// Apply RoPE in-place.
pub fn gpu_rope(
    ctx: &Arc<MetalContext>,
    data: &GpuBuffer,
    total_rows: u32,
    seq_len: u32,
    head_dim: u32,
    offset: u32,
    theta: f32,
) {
    #[repr(C)]
    struct Params { seq_len: u32, head_dim: u32, total_rows: u32, offset: u32, theta: f32 }
    let params = Params { seq_len, head_dim, total_rows, offset, theta };
    let params_buf = params_buffer(ctx, &params);

    let half_dim = head_dim / 2;
    let total = MetalContext::size(seq_len as u64, total_rows as u64, half_dim as u64);
    let tg = MetalContext::size(
        8.min(seq_len as u64).max(1),
        8.min(total_rows as u64).max(1),
        8.min(half_dim as u64).max(1),
    );

    dispatch_threads_sync!(ctx, "rope", total, tg,
        0 => data, 1 => &params_buf
    );
}

/// RoPE backward: apply inverse rotation (rotate by -θ) to propagate gradients.
pub fn gpu_rope_backward(
    ctx: &Arc<MetalContext>,
    data: &GpuBuffer,
    total_rows: u32,
    seq_len: u32,
    head_dim: u32,
    offset: u32,
    theta: f32,
) {
    #[repr(C)]
    struct Params { seq_len: u32, head_dim: u32, total_rows: u32, offset: u32, theta: f32 }
    let params = Params { seq_len, head_dim, total_rows, offset, theta };
    let params_buf = params_buffer(ctx, &params);

    let half_dim = head_dim / 2;
    let total = MetalContext::size(seq_len as u64, total_rows as u64, half_dim as u64);
    let tg = MetalContext::size(
        8.min(seq_len as u64).max(1),
        8.min(total_rows as u64).max(1),
        8.min(half_dim as u64).max(1),
    );

    dispatch_threads_sync!(ctx, "rope_backward", total, tg,
        0 => data, 1 => &params_buf,
    );
}

/// Elementwise addition: c = a + b
pub fn gpu_add(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "add", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// In-place add: a += b. Avoids allocating a third buffer for gradient accumulation.
pub fn gpu_add_inplace(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "add_inplace", grid, tg,
        0 => a, 1 => b, 2 => &params_buf
    );
}

/// Elementwise multiply: c = a * b
pub fn gpu_mul(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "mul", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// SiLU activation
pub fn gpu_silu(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "silu", grid, tg,
        0 => input, 1 => output, 2 => &params_buf
    );
}

/// Fused SiLU-gate: output = silu(gate) * up (one kernel, one buffer)
pub fn gpu_silu_gate(ctx: &Arc<MetalContext>, gate: &GpuBuffer, up: &GpuBuffer, output: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "silu_gate", grid, tg,
        0 => gate, 1 => up, 2 => output, 3 => &params_buf
    );
}

/// Fused cross-entropy loss.
pub fn gpu_cross_entropy(
    ctx: &Arc<MetalContext>,
    logits: &GpuBuffer,
    targets: &GpuBuffer,
    losses: &GpuBuffer,
    grad_logits: &GpuBuffer,
    batch_size: u32,
    vocab_size: u32,
) {
    #[repr(C)]
    struct Params { batch_size: u32, vocab_size: u32 }
    let params = Params { batch_size, vocab_size };
    let params_buf = params_buffer(ctx, &params);

    let threads_per_group = next_power_of_2_clamped(vocab_size as u64);
    let grid = MetalContext::size(batch_size as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);

    dispatch_sync!(ctx, "cross_entropy", grid, tg,
        0 => logits, 1 => targets, 2 => losses, 3 => grad_logits, 4 => &params_buf
    );
}

/// Reduce sum: output[0] = sum(input)
pub fn gpu_reduce_sum(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let grid = MetalContext::size(1, 1, 1);
    let tg = MetalContext::size(next_power_of_2_clamped(size as u64), 1, 1);

    dispatch_sync!(ctx, "reduce_sum", grid, tg,
        0 => input, 1 => output, 2 => &params_buf
    );
}

/// AdamW optimizer hyperparameters.
pub struct AdamWHyperparams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u32,
}

/// AdamW fused update.
pub fn gpu_adamw_update(
    ctx: &Arc<MetalContext>,
    param: &GpuBuffer,
    grad: &GpuBuffer,
    m: &GpuBuffer,
    v: &GpuBuffer,
    size: u32,
    hp: &AdamWHyperparams,
) {
    let AdamWHyperparams { lr, beta1, beta2, eps, weight_decay, step } = *hp;
    #[repr(C)]
    struct Params {
        size: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
    }
    let params = Params {
        size,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1: 1.0 - beta1.powi(step as i32),
        bias_correction2: 1.0 - beta2.powi(step as i32),
    };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "adamw_update", grid, tg,
        0 => param, 1 => grad, 2 => m, 3 => v, 4 => &params_buf
    );
}

/// Embedding lookup
pub fn gpu_embedding_lookup(
    ctx: &Arc<MetalContext>,
    tokens: &GpuBuffer,
    embeddings: &GpuBuffer,
    output: &GpuBuffer,
    n_tokens: u32,
    dim: u32,
) {
    #[repr(C)]
    struct Params { n_tokens: u32, dim: u32 }
    let params = Params { n_tokens, dim };
    let params_buf = params_buffer(ctx, &params);

    let total = MetalContext::size(dim as u64, n_tokens as u64, 1);
    let tg = MetalContext::size(16.min(dim as u64).max(1), 16.min(n_tokens as u64).max(1), 1);

    dispatch_threads_sync!(ctx, "embedding_lookup", total, tg,
        0 => tokens, 1 => embeddings, 2 => output, 3 => &params_buf
    );
}

/// Apply causal mask
pub fn gpu_causal_mask(
    ctx: &Arc<MetalContext>,
    scores: &GpuBuffer,
    batch_heads: u32,
    seq_q: u32,
    seq_k: u32,
    offset: u32,
) {
    #[repr(C)]
    struct Params { batch_heads: u32, seq_q: u32, seq_k: u32, offset: u32 }
    let params = Params { batch_heads, seq_q, seq_k, offset };
    let params_buf = params_buffer(ctx, &params);

    let total = MetalContext::size(seq_k as u64, seq_q as u64, batch_heads as u64);
    let tg = MetalContext::size(
        8.min(seq_k as u64).max(1),
        8.min(seq_q as u64).max(1),
        4.min(batch_heads as u64).max(1),
    );

    dispatch_threads_sync!(ctx, "causal_mask", total, tg,
        0 => scores, 1 => &params_buf
    );
}

/// Compute L2 norm. Returns the norm value (includes GPU→CPU readback).
pub fn gpu_l2_norm(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32) -> f32 {
    let output = ctx.alloc_buffer(std::mem::size_of::<f32>());
    gpu_l2_norm_into(ctx, data, size, &output);
    MetalContext::read_buffer(&output, 1)[0]
}

/// Compute L2 norm into a pre-allocated output buffer (no CPU readback).
/// Use this in batched contexts to avoid breaking the command batch.
pub fn gpu_l2_norm_into(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, output: &GpuBuffer) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let grid = MetalContext::size(1, 1, 1);
    let tg = MetalContext::size(next_power_of_2_clamped(size as u64), 1, 1);

    dispatch_sync!(ctx, "l2_norm", grid, tg,
        0 => data, 1 => output, 2 => &params_buf
    );
}

/// Compute L2 norm (sum of squares) and NaN/Inf check into a pre-allocated output buffer.
/// Output buffer must hold 2 floats: [0] = sum_of_squares, [1] = has_nan_or_inf (1.0 or 0.0).
/// Returns raw sum_sq (not sqrt) for accumulation across multiple params.
pub fn gpu_l2_norm_check_into(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, output: &GpuBuffer) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let grid = MetalContext::size(1, 1, 1);
    let tg = MetalContext::size(next_power_of_2_clamped(size as u64), 1, 1);

    dispatch_sync!(ctx, "l2_norm_check", grid, tg,
        0 => data, 1 => output, 2 => &params_buf
    );
}

/// Compute L2 norm and NaN/Inf check. Returns (sum_of_squares, has_nan).
/// Includes GPU→CPU readback. For batched use, prefer gpu_l2_norm_check_into.
pub fn gpu_l2_norm_check(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32) -> (f32, bool) {
    let output = ctx.alloc_buffer(std::mem::size_of::<f32>() * 2);
    gpu_l2_norm_check_into(ctx, data, size, &output);
    let vals = MetalContext::read_buffer(&output, 2);
    (vals[0], vals[1] > 0.5)
}

/// Scale buffer in-place
pub fn gpu_scale(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, scale: f32) {
    #[repr(C)]
    struct Params { size: u32, scale: f32 }
    let params = Params { size, scale };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "scale", grid, tg,
        0 => data, 1 => &params_buf
    );
}

/// Fill buffer with a constant
pub fn gpu_fill(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, value: f32) {
    #[repr(C)]
    struct Params { size: u32, value: f32 }
    let params = Params { size, value };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "fill", grid, tg,
        0 => data, 1 => &params_buf
    );
}

/// Copy buffer: dst = src
pub fn gpu_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "copy_buffer", grid, tg,
        0 => src, 1 => dst, 2 => &params_buf
    );
}

/// SiLU backward
pub fn gpu_silu_backward(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_input: &GpuBuffer,
    size: u32,
) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "silu_backward", grid, tg,
        0 => input, 1 => grad_output, 2 => grad_input, 3 => &params_buf
    );
}

/// Fused SiLU-gate backward: computes grad_gate and grad_up in one kernel.
pub fn gpu_silu_gate_backward(
    ctx: &Arc<MetalContext>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_gate: &GpuBuffer,
    grad_up: &GpuBuffer,
    size: u32,
) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "silu_gate_backward", grid, tg,
        0 => gate, 1 => up, 2 => grad_output, 3 => grad_gate, 4 => grad_up, 5 => &params_buf
    );
}

/// RMS norm backward shape and epsilon parameters.
pub struct RmsNormBackwardParams {
    pub rows: u32,
    pub cols: u32,
    pub eps: f32,
}

/// RMS norm backward
pub fn gpu_rms_norm_backward(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_input: &GpuBuffer,
    grad_weight: &GpuBuffer,
    params: &RmsNormBackwardParams,
) {
    let RmsNormBackwardParams { rows, cols, eps } = *params;
    // Zero grad_weight first
    gpu_fill(ctx, grad_weight, cols, 0.0);

    #[repr(C)]
    struct Params { rows: u32, cols: u32, eps: f32 }
    let params = Params { rows, cols, eps };
    let params_buf = params_buffer(ctx, &params);

    let threads_per_group = next_power_of_2_clamped(cols as u64);
    let grid = MetalContext::size(rows as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);

    dispatch_sync!(ctx, "rms_norm_backward", grid, tg,
        0 => input, 1 => weight, 2 => grad_output, 3 => grad_input, 4 => grad_weight, 5 => &params_buf
    );
}

/// Softmax backward
pub fn gpu_softmax_backward(
    ctx: &Arc<MetalContext>,
    softmax_out: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_input: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    #[repr(C)]
    struct Params { rows: u32, cols: u32 }
    let params = Params { rows, cols };
    let params_buf = params_buffer(ctx, &params);

    let threads_per_group = next_power_of_2_clamped(cols as u64);
    let grid = MetalContext::size(rows as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);

    dispatch_sync!(ctx, "softmax_backward", grid, tg,
        0 => softmax_out, 1 => grad_output, 2 => grad_input, 3 => &params_buf
    );
}

/// Embedding backward: scatter-add gradients into embedding matrix.
/// Uses 1D threadgroup dispatch (one threadgroup per dim position) with
/// threadgroup-local accumulation to reduce atomic contention on common tokens.
pub fn gpu_embedding_backward(
    ctx: &Arc<MetalContext>,
    tokens: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_embeddings: &GpuBuffer,
    n_tokens: u32,
    dim: u32,
) {
    // Zero only the rows that will be touched by scatter-add, instead of
    // zeroing the entire vocab_size × dim matrix. For typical training
    // (batch=4, seq=512, vocab=32K), this zeros ~2K rows instead of 32K.
    gpu_zero_rows(ctx, tokens, grad_embeddings, n_tokens, dim);

    #[repr(C)]
    struct Params { n_tokens: u32, dim: u32 }
    let params = Params { n_tokens, dim };
    let params_buf = params_buffer(ctx, &params);

    // One threadgroup per dim position. Each threadgroup has up to 256 threads
    // that split n_tokens among themselves with local accumulation per token_id.
    let threads_per_group = 256u64.min(n_tokens as u64).max(1);
    let grid = MetalContext::size(dim as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);

    dispatch_sync!(ctx, "embedding_backward", grid, tg,
        0 => tokens, 1 => grad_output, 2 => grad_embeddings, 3 => &params_buf
    );
}

/// Zero only the rows of a matrix that correspond to given token IDs.
/// Used before embedding backward scatter-add to avoid zeroing the full matrix.
pub fn gpu_zero_rows(
    ctx: &Arc<MetalContext>,
    tokens: &GpuBuffer,
    matrix: &GpuBuffer,
    n_tokens: u32,
    dim: u32,
) {
    #[repr(C)]
    struct Params { n_tokens: u32, dim: u32 }
    let params = Params { n_tokens, dim };
    let params_buf = params_buffer(ctx, &params);

    let total = (n_tokens as u64) * (dim as u64);
    let tpg = 256u64;
    let groups = total.div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "zero_rows", grid, tg,
        0 => tokens, 1 => matrix, 2 => &params_buf
    );
}

/// GPU 2D matrix transpose: out[j,i] = in[i,j]. in:[rows,cols], out:[cols,rows]
pub fn gpu_transpose_2d(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    #[repr(C)]
    struct Params { rows: u32, cols: u32 }
    let params = Params { rows, cols };
    let params_buf = params_buffer(ctx, &params);

    let size = (rows * cols) as u64;
    let tpg = 256u64;
    let groups = size.div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "transpose_2d", grid, tg,
        0 => input, 1 => output, 2 => &params_buf
    );
}

/// C = A^T @ B where A:[M,K] row-major, B:[M,N], C:[K,N]
pub fn gpu_matmul_trans_a(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    k: u32,
    n: u32,
) {
    #[repr(C)]
    struct Params { m: u32, k: u32, n: u32 }
    let params = Params { m, k, n };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (k as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, 1);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "matmul_trans_a_tiled", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// Buffer-to-buffer copy with offsets (all in floats, not bytes).
pub fn gpu_buffer_copy(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    src_offset: u32,
    dst_offset: u32,
    count: u32,
) {
    #[repr(C)]
    struct Params { src_offset: u32, dst_offset: u32, count: u32 }
    let params = Params { src_offset, dst_offset, count };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let groups = (count as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "buffer_copy", grid, tg,
        0 => src, 1 => dst, 2 => &params_buf
    );
}

/// Attention transpose permutation backward (GPU).
/// Input: grad [batch*n_heads, seq, head_dim]
/// Output: grad [batch*seq, n_heads*head_dim]
pub fn gpu_transpose_perm_backward(
    ctx: &Arc<MetalContext>,
    grad_in: &GpuBuffer,
    grad_out: &GpuBuffer,
    batch: u32,
    seq: u32,
    n_heads: u32,
    head_dim: u32,
) {
    #[repr(C)]
    struct Params { batch: u32, seq: u32, n_heads: u32, head_dim: u32 }
    let params = Params { batch, seq, n_heads, head_dim };
    let params_buf = params_buffer(ctx, &params);

    let total = (batch * seq * n_heads * head_dim) as u64;
    let tpg = 256u64;
    let groups = total.div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "transpose_perm_backward", grid, tg,
        0 => grad_in, 1 => grad_out, 2 => &params_buf
    );
}

/// Forward attention transpose (GPU).
/// Input: [batch*seq, n_heads*head_dim]
/// Output: [batch*n_heads, seq, head_dim]
pub fn gpu_transpose_perm_forward(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    batch: u32,
    seq: u32,
    n_heads: u32,
    head_dim: u32,
) {
    #[repr(C)]
    struct Params { batch: u32, seq: u32, n_heads: u32, head_dim: u32 }
    let params = Params { batch, seq, n_heads, head_dim };
    let params_buf = params_buffer(ctx, &params);

    let total = (batch * seq * n_heads * head_dim) as u64;
    let tpg = 256u64;
    let groups = total.div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "transpose_perm_forward", grid, tg,
        0 => input, 1 => output, 2 => &params_buf
    );
}

/// Apply gradient mask: zero out rows where mask[pos] == 0.
/// grad: [positions, vocab_size], mask: [positions] as u32 (0 or 1).
pub fn gpu_gradient_mask(
    ctx: &Arc<MetalContext>,
    grad: &GpuBuffer,
    mask: &GpuBuffer,
    positions: u32,
    vocab_size: u32,
) {
    #[repr(C)]
    struct Params { total: u32, vocab_size: u32 }
    let total = positions * vocab_size;
    let params = Params { total, vocab_size };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let grid = MetalContext::size((total as u64).div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "gradient_mask", grid, tg,
        0 => grad, 1 => mask, 2 => &params_buf
    );
}

/// Batched strided copy: src [bh, src_seq_len, dim] (contiguous) →
/// dst [bh, dst_stride, dim] at seq offset dst_offset per batch-head.
/// Single GPU dispatch replaces O(bh) individual buffer_copy calls.
pub fn gpu_strided_batch_copy(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    bh: u32,
    src_seq_len: u32,
    dst_stride: u32,
    dst_offset: u32,
    dim: u32,
) {
    #[repr(C)]
    struct Params { bh: u32, src_seq_len: u32, dst_stride: u32, dst_offset: u32, dim: u32 }
    let params = Params { bh, src_seq_len, dst_stride, dst_offset, dim };
    let params_buf = params_buffer(ctx, &params);

    let total_threads = bh as u64 * src_seq_len as u64 * dim as u64;
    let tpg = 256u64;
    let grid = MetalContext::size(total_threads.div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "strided_batch_copy", grid, tg,
        0 => src, 1 => dst, 2 => &params_buf
    );
}

/// Compact strided copy: src [bh, src_stride, dim] (strided, first seq_len valid) →
/// dst [bh, seq_len, dim] (contiguous). Reverse of strided_batch_copy.
/// Single GPU dispatch replaces O(bh) individual buffer_copy calls.
pub fn gpu_compact_strided_copy(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    bh: u32,
    seq_len: u32,
    src_stride: u32,
    dim: u32,
) {
    #[repr(C)]
    struct Params { bh: u32, seq_len: u32, src_stride: u32, dim: u32 }
    let params = Params { bh, seq_len, src_stride, dim };
    let params_buf = params_buffer(ctx, &params);

    let total_threads = bh as u64 * seq_len as u64 * dim as u64;
    let tpg = 256u64;
    let grid = MetalContext::size(total_threads.div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "compact_strided_copy", grid, tg,
        0 => src, 1 => dst, 2 => &params_buf
    );
}

/// GPU argmax: find the index of the maximum value in a float buffer.
/// Returns a single u32 — reads back only 4 bytes instead of the full buffer.
/// Uses a single threadgroup of 256 threads for parallel reduction.
pub fn gpu_argmax(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32) -> u32 {
    let result_buf = ctx.alloc_buffer(std::mem::size_of::<u32>());

    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);

    let grid = MetalContext::size(1, 1, 1);
    let tg = MetalContext::size(256, 1, 1);

    dispatch_sync!(ctx, "argmax", grid, tg,
        0 => data, 1 => &result_buf, 2 => &params_buf
    );

    MetalContext::read_buffer_u32(&result_buf, 1)[0]
}

/// GPU temperature scaling: divide logits by temperature in-place.
/// Operates on a sub-range [offset, offset + count) of the buffer.
/// This avoids copying the full logits tensor to CPU just for scaling.
pub fn gpu_temperature_scale(
    ctx: &Arc<MetalContext>,
    data: &GpuBuffer,
    offset: u32,
    count: u32,
    temperature: f32,
) {
    #[repr(C)]
    struct Params { offset: u32, count: u32, inv_temperature: f32 }
    let params = Params { offset, count, inv_temperature: 1.0 / temperature };
    let params_buf = params_buffer(ctx, &params);

    let tpg = 256u64;
    let grid = MetalContext::size((count as u64).div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_sync!(ctx, "temperature_scale", grid, tg,
        0 => data, 1 => &params_buf
    );
}

/// KL divergence: KL(softmax(teacher/T) || softmax(student/T))
/// teacher_logits, student_logits: [batch, vocab] flat f32 buffers
/// losses: [batch] per-sample KL divergence
/// grad_student: [batch * vocab] raw gradient w.r.t. student logits: (1/T) * (q - p) / batch
pub fn gpu_kl_divergence(
    ctx: &Arc<MetalContext>,
    teacher_logits: &GpuBuffer,
    student_logits: &GpuBuffer,
    losses: &GpuBuffer,
    grad_student: &GpuBuffer,
    batch_size: u32,
    vocab_size: u32,
    temperature: f32,
) {
    #[repr(C)]
    struct Params { batch_size: u32, vocab_size: u32, temperature: f32 }
    let params = Params { batch_size, vocab_size, temperature };
    let params_buf = params_buffer(ctx, &params);

    let threads_per_group = next_power_of_2_clamped(vocab_size as u64);
    let grid = MetalContext::size(batch_size as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);

    dispatch_sync!(ctx, "kl_divergence", grid, tg,
        0 => teacher_logits, 1 => student_logits, 2 => losses, 3 => grad_student, 4 => &params_buf
    );
}
