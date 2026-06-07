use super::{GpuBuffer, GpuComputeEncoder, MetalContext};
use objc2_metal::{MTLComputeCommandEncoder as MTLComputeCommandEncoderTrait, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

/// Round up to the next power of 2, clamped to [1, 256].
/// Required for threadgroup reductions in all row-wise kernels.
/// 256 threads × 2 shared arrays × 4 bytes = 2KB — fits in 32KB threadgroup memory.
#[inline]
fn next_power_of_2_clamped(n: u64) -> u64 {
    let clamped = n.clamp(1, 256) as u32;
    let p = clamped.next_power_of_two().min(256);
    p as u64
}

/// Create a params buffer from a repr(C) struct.
/// Create a params buffer. Uses newBufferWithBytes (not the pool) because params
/// buffers are live within a command batch — pooling would cause aliasing between
/// kernels encoded in the same batch.
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

    // Auto-select narrow kernel when N is small — eliminates 50% wasted compute at N=16
    if n <= 32 {
        let tile_n = 16u64;
        let tile_m = 32u64;
        let groups_x = (n as u64).div_ceil(tile_n);
        let groups_y = (m as u64).div_ceil(tile_m);
        let grid = MetalContext::size(groups_x, groups_y, 1);
        let tg = MetalContext::size(32, 1, 1);
        dispatch_sync!(ctx, "matmul_narrow", grid, tg,
            0 => a, 1 => b, 2 => c, 3 => &params_buf
        );
    } else {
        let tile = 32u64;
        let groups_x = (n as u64).div_ceil(tile);
        let groups_y = (m as u64).div_ceil(tile);
        let grid = MetalContext::size(groups_x, groups_y, 1);
        let tg = MetalContext::size(64, 1, 1);
        dispatch_sync!(ctx, "matmul_tiled", grid, tg,
            0 => a, 1 => b, 2 => c, 3 => &params_buf
        );
    }
}

/// Full-FP32 tiled matmul: C = A @ B with no fp16 cast/clamp (precise path). Always the 32×32
/// tiled kernel (no narrow specialisation). Slower than gpu_matmul but full precision and range.
pub fn gpu_matmul_fp32(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32 }
    let params = Params { m, n, k };
    let params_buf = params_buffer(ctx, &params);
    let tile = 32u64;
    let grid = MetalContext::size((n as u64).div_ceil(tile), (m as u64).div_ceil(tile), 1);
    let tg = MetalContext::size(64, 1, 1);
    dispatch_sync!(ctx, "matmul_tiled_fp32", grid, tg, 0 => a, 1 => b, 2 => c, 3 => &params_buf);
}

/// BF16 tiled matmul: C = A @ B with `bfloat` shared tiles — fp32 range (no ±65504 clamp), ~half
/// fp32 bandwidth, bf16 mantissa precision. The range-safe mixed-precision matmul.
pub fn gpu_matmul_bf16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32 }
    let params = Params { m, n, k };
    let params_buf = params_buffer(ctx, &params);
    let tile = 32u64;
    let grid = MetalContext::size((n as u64).div_ceil(tile), (m as u64).div_ceil(tile), 1);
    let tg = MetalContext::size(64, 1, 1);
    dispatch_sync!(ctx, "matmul_tiled_bf16", grid, tg, 0 => a, 1 => b, 2 => c, 3 => &params_buf);
}

/// Cast float32 buffer to float16. Output buffer must be size * 2 bytes.
pub fn gpu_cast_f32_to_f16(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    let size_buf = params_buffer(ctx, &size);
    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "cast_f32_to_f16", grid, tg,
        0 => input, 1 => output, 2 => &size_buf
    );
}

/// Cast float16 buffer to float32. Output buffer must be size * 4 bytes.
pub fn gpu_cast_f16_to_f32(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    let size_buf = params_buffer(ctx, &size);
    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "cast_f16_to_f32", grid, tg,
        0 => input, 1 => output, 2 => &size_buf
    );
}

/// C(f32) = A(f16) @ B(f16) — FP16 inputs, FP32 output. Halves memory bandwidth.
pub fn gpu_matmul_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32 }
    let params = Params { m, n, k };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (m as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, 1);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "matmul_tiled_f16", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// C(f32) = A(f16) @ B(f16)^T — FP16 inputs, FP32 output.
pub fn gpu_matmul_trans_b_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32 }
    let params = Params { m, n, k };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (m as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, 1);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "matmul_tiled_trans_b_f16", grid, tg,
        0 => a, 1 => b, 2 => c, 3 => &params_buf
    );
}

/// C(f32) = A(f16)^T @ B(f16) — FP16 inputs, FP32 output.
pub fn gpu_matmul_trans_a_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, k: u32, n: u32) {
    #[repr(C)]
    struct Params { m: u32, k: u32, n: u32 }
    let params = Params { m, k, n };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (k as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, 1);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "matmul_trans_a_tiled_f16", grid, tg,
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

/// Batched C[b](f32) = A[b](f16) @ B[b](f16). Single dispatch, FP16 inputs.
/// Dimensions for a batched matmul dispatch: C[batch, M, N]. K is the contraction dim.
#[derive(Clone, Copy)]
pub struct BatchedDims { pub batch: u32, pub m: u32, pub n: u32, pub k: u32 }

/// RoPE-over-flat-rows dims (rope_copy / rope_backward_copy).
#[derive(Clone, Copy)]
pub struct RopeDims { pub total_rows: u32, pub seq_len: u32, pub head_dim: u32, pub offset: u32, pub theta: f32 }
/// Transpose+RoPE dims (transpose_rope / _backward).
#[derive(Clone, Copy)]
pub struct TrRopeDims { pub batch: u32, pub seq: u32, pub n_heads: u32, pub head_dim: u32, pub offset: u32, pub theta: f32 }
/// Flash-attention dims.
#[derive(Clone, Copy)]
pub struct FlashDims { pub batch_heads: u32, pub seq_q: u32, pub seq_k: u32, pub head_dim: u32, pub kv_offset: u32 }
/// Scaled-causal-softmax dims (`window` = u32::MAX means no window).
#[derive(Clone, Copy)]
pub struct SoftmaxDims { pub total_rows: u32, pub seq_q: u32, pub seq_k: u32, pub scale: f32, pub kv_offset: u32 }
/// Lion optimizer hyperparameters.
#[derive(Clone, Copy)]
pub struct LionParams { pub lr: f32, pub beta1: f32, pub beta2: f32, pub weight_decay: f32 }
/// Sophia optimizer hyperparameters.
#[derive(Clone, Copy)]
pub struct SophiaParams { pub lr: f32, pub beta1: f32, pub beta2: f32, pub eps: f32, pub rho: f32, pub weight_decay: f32 }

/// Fused residual-add + RMS-norm dims.
#[derive(Clone, Copy)]
pub struct RmsResDims { pub rows: u32, pub cols: u32, pub eps: f32 }
/// Strided batch-copy layout.
#[derive(Clone, Copy)]
pub struct StridedCopyDims { pub bh: u32, pub src_seq_len: u32, pub dst_stride: u32, pub dst_offset: u32, pub dim: u32 }
/// KL-divergence distillation dims.
#[derive(Clone, Copy)]
pub struct KlDims { pub batch_size: u32, pub vocab_size: u32, pub temperature: f32 }
/// Mega-FFN dims.
#[derive(Clone, Copy)]
pub struct MegaFfnDims { pub batch_tokens: u32, pub d_model: u32, pub d_ff: u32, pub eps: f32 }
/// SwiGLU FFN weight matrices.
#[derive(Clone, Copy)]
pub struct FfnWeights<'a> { pub w1: &'a GpuBuffer, pub w2: &'a GpuBuffer, pub w3: &'a GpuBuffer }
/// Fused norm+matmul dims.
#[derive(Clone, Copy)]
pub struct NormMatmulDims { pub m: u32, pub n: u32, pub k: u32, pub eps: f32 }
/// Flash-attention backward buffers.
#[derive(Clone, Copy)]
pub struct FlashBwdBufs<'a> {
    pub q: &'a GpuBuffer, pub k: &'a GpuBuffer, pub v: &'a GpuBuffer,
    pub output: &'a GpuBuffer, pub d_out: &'a GpuBuffer, pub d_buf: &'a GpuBuffer,
    pub dq: &'a GpuBuffer, pub dk: &'a GpuBuffer, pub dv: &'a GpuBuffer,
}

pub fn gpu_batched_matmul_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims) {
    let BatchedDims { batch, m, n, k } = d;
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32, batch: u32 }
    let params = Params { m, n, k, batch };
    let params_buf = params_buffer(ctx, &params);
    let tile = 32u64;
    let grid = MetalContext::size((n as u64).div_ceil(tile), (m as u64).div_ceil(tile), batch as u64);
    let tg = MetalContext::size(64, 1, 1);
    dispatch_sync!(ctx, "batched_matmul_tiled_f16", grid, tg, 0 => a, 1 => b, 2 => c, 3 => &params_buf);
}

/// Batched C[b](f32) = A[b](f16) @ B[b](f16)^T. Single dispatch, FP16 inputs.
pub fn gpu_batched_matmul_trans_b_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims) {
    let BatchedDims { batch, m, n, k } = d;
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32, batch: u32 }
    let params = Params { m, n, k, batch };
    let params_buf = params_buffer(ctx, &params);
    let tile = 32u64;
    let grid = MetalContext::size((n as u64).div_ceil(tile), (m as u64).div_ceil(tile), batch as u64);
    let tg = MetalContext::size(64, 1, 1);
    dispatch_sync!(ctx, "batched_matmul_tiled_trans_b_f16", grid, tg, 0 => a, 1 => b, 2 => c, 3 => &params_buf);
}

/// Batched C[b](f32) = A[b](f16)^T @ B[b](f16). Single dispatch, FP16 inputs.
pub fn gpu_batched_matmul_trans_a_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims) {
    let BatchedDims { batch, m, n, k } = d;
    #[repr(C)]
    struct Params { m: u32, k: u32, n: u32, batch: u32 }
    let params = Params { m, k, n, batch };
    let params_buf = params_buffer(ctx, &params);
    let tile = 32u64;
    let grid = MetalContext::size((n as u64).div_ceil(tile), (k as u64).div_ceil(tile), batch as u64);
    let tg = MetalContext::size(64, 1, 1);
    dispatch_sync!(ctx, "batched_matmul_tiled_trans_a_f16", grid, tg, 0 => a, 1 => b, 2 => c, 3 => &params_buf);
}

/// Batched C[b] = A[b] @ B[b] for all b in [0, batch). Single GPU dispatch.
/// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N]
pub fn gpu_batched_matmul(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims) {
    let BatchedDims { batch, m, n, k } = d;
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
pub fn gpu_batched_matmul_trans_b(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims) {
    let BatchedDims { batch, m, n, k } = d;
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

/// GQA-aware batched C[b] = A[b] @ B[b/group_size]^T. Eliminates repeat_kv copy.
/// A: [batch_q, M, K], B: [batch_kv, N, K], C: [batch_q, M, N]
/// batch_q = batch * n_heads, batch_kv = batch * n_kv_heads, group_size = n_heads / n_kv_heads
pub fn gpu_batched_matmul_gqa_trans_b(
    ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer,
    d: BatchedDims, group_size: u32,
) {
    let BatchedDims { batch: batch_q, m, n, k } = d;
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32, batch: u32, group_size: u32 }
    let params = Params { m, n, k, batch: batch_q, group_size };
    let params_buf = params_buffer(ctx, &params);
    let tile = 32u64;
    let grid = MetalContext::size((n as u64).div_ceil(tile), (m as u64).div_ceil(tile), batch_q as u64);
    let tg = MetalContext::size(64, 1, 1);
    dispatch_sync!(ctx, "batched_matmul_gqa_trans_b", grid, tg, 0 => a, 1 => b, 2 => c, 3 => &params_buf);
}

/// GQA-aware batched C[b] = A[b] @ B[b/group_size]. Eliminates repeat_kv copy.
pub fn gpu_batched_matmul_gqa(
    ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer,
    d: BatchedDims, group_size: u32,
) {
    let BatchedDims { batch: batch_q, m, n, k } = d;
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32, batch: u32, group_size: u32 }
    let params = Params { m, n, k, batch: batch_q, group_size };
    let params_buf = params_buffer(ctx, &params);
    let tile = 32u64;
    let grid = MetalContext::size((n as u64).div_ceil(tile), (m as u64).div_ceil(tile), batch_q as u64);
    let tg = MetalContext::size(64, 1, 1);
    dispatch_sync!(ctx, "batched_matmul_gqa", grid, tg, 0 => a, 1 => b, 2 => c, 3 => &params_buf);
}

/// Batched C[b] = A[b]^T @ B[b] for all b. Single GPU dispatch.
/// A: [batch, M, K] (transposed to [K,M]), B: [batch, M, N], C: [batch, K, N]
pub fn gpu_batched_matmul_trans_a(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims) {
    let BatchedDims { batch, m, n, k } = d;
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
pub fn gpu_rms_norm_residual(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer, residual: &GpuBuffer, weight: &GpuBuffer,
    output: &GpuBuffer, sum_out: &GpuBuffer, d: RmsResDims,
) {
    let RmsResDims { rows, cols, eps } = d;
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

/// Out-of-place RoPE forward: dst = rotate(src, θ). Single dispatch replaces copy + in-place.
pub fn gpu_rope_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, d: RopeDims) {
    let RopeDims { total_rows, seq_len, head_dim, offset, theta } = d;
    #[repr(C)]
    struct Params { seq_len: u32, head_dim: u32, total_rows: u32, offset: u32, theta: f32 }
    let params = Params { seq_len, head_dim, total_rows, offset, theta };
    let params_buf = params_buffer(ctx, &params);
    let half_dim = head_dim / 2;
    let total = MetalContext::size(seq_len as u64, total_rows as u64, half_dim as u64);
    let tg = MetalContext::size(
        8.min(seq_len as u64).max(1), 8.min(total_rows as u64).max(1), 8.min(half_dim as u64).max(1),
    );
    dispatch_threads_sync!(ctx, "rope_copy", total, tg, 0 => src, 1 => dst, 2 => &params_buf);
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

/// Out-of-place RoPE backward: dst = rotate(src, -θ). Single dispatch replaces copy + in-place.
pub fn gpu_rope_backward_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, d: RopeDims) {
    let RopeDims { total_rows, seq_len, head_dim, offset, theta } = d;
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

    dispatch_threads_sync!(ctx, "rope_backward_copy", total, tg,
        0 => src, 1 => dst, 2 => &params_buf,
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

/// Flash Attention Forward: fused Q@K^T → mask → softmax → @V in one kernel.
/// Q,K,V: [batch_heads, seq, head_dim], O: [batch_heads, seq_q, head_dim]
pub fn gpu_flash_attention_forward(
    ctx: &Arc<MetalContext>, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer, o: &GpuBuffer, d: FlashDims,
) {
    let FlashDims { batch_heads, seq_q, seq_k, head_dim, kv_offset } = d;
    #[repr(C)]
    struct Params { seq_q: u32, seq_k: u32, head_dim: u32, batch_heads: u32, scale: f32, kv_offset: u32 }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let params = Params { seq_q, seq_k, head_dim, batch_heads, scale, kv_offset };
    let params_buf = params_buffer(ctx, &params);

    let br = 32u64; // query block size — matches FA_BR in shader
    let q_blocks = (seq_q as u64).div_ceil(br);
    let grid = MetalContext::size(batch_heads as u64, q_blocks, 1);
    let tg = MetalContext::size(br, 1, 1); // one thread per query row in block

    dispatch_sync!(ctx, "flash_attention_forward", grid, tg,
        0 => q, 1 => k, 2 => v, 3 => o, 4 => &params_buf
    );
}

/// Precompute D[i] = sum_j(dO[i][j] * O[i][j]) for Flash Attention backward.
pub fn gpu_flash_attn_precompute_d(
    ctx: &Arc<MetalContext>,
    d_out: &GpuBuffer, output: &GpuBuffer, d_buf: &GpuBuffer,
    total_rows: u32, head_dim: u32,
) {
    let total_rows_buf = params_buffer(ctx, &total_rows);
    let head_dim_buf = params_buffer(ctx, &head_dim);
    let tpg = 256u64;
    let groups = (total_rows as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups * tpg, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_threads_sync!(ctx, "flash_attn_precompute_d", grid, tg,
        0 => d_out, 1 => output, 2 => d_buf, 3 => &total_rows_buf, 4 => &head_dim_buf
    );
}

/// Flash Attention Backward: compute dQ, dK, dV.
pub fn gpu_flash_attention_backward(ctx: &Arc<MetalContext>, b: FlashBwdBufs, d: FlashDims) {
    let FlashBwdBufs { q, k, v, output, d_out, d_buf, dq, dk, dv } = b;
    let FlashDims { batch_heads, seq_q, seq_k, head_dim, kv_offset } = d;
    #[repr(C)]
    struct Params { seq_q: u32, seq_k: u32, head_dim: u32, batch_heads: u32, scale: f32, kv_offset: u32 }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let params = Params { seq_q, seq_k, head_dim, batch_heads, scale, kv_offset };
    let params_buf = params_buffer(ctx, &params);

    let br = 32u64;
    let q_blocks = (seq_q as u64).div_ceil(br);
    let grid = MetalContext::size(batch_heads as u64, q_blocks, 1);
    let tg = MetalContext::size(br, 1, 1);

    dispatch_sync!(ctx, "flash_attention_backward", grid, tg,
        0 => q, 1 => k, 2 => v, 3 => output, 4 => d_out, 5 => d_buf,
        6 => dq, 7 => dk, 8 => dv, 9 => &params_buf
    );
}

/// BitNet: ternary matmul C = A(float) @ W(ternary packed).
/// W is packed as 2 bits per weight, 16 per u32. No floating point multiply.
pub fn gpu_ternary_matmul(ctx: &Arc<MetalContext>, a: &GpuBuffer, w_packed: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32 }
    let params = Params { m, n, k };
    let params_buf = params_buffer(ctx, &params);

    let grid = MetalContext::size(n as u64, m as u64, 1);
    let tg = MetalContext::size(n.min(32) as u64, m.min(32) as u64, 1);

    dispatch_sync!(ctx, "ternary_matmul", grid, tg,
        0 => a, 1 => w_packed, 2 => c, 3 => &params_buf
    );
}

/// BitNet: compute absmean per column for ternary quantization threshold.
pub fn gpu_ternary_absmean(ctx: &Arc<MetalContext>, weights: &GpuBuffer, absmean: &GpuBuffer, rows: u32, cols: u32) {
    let rows_buf = params_buffer(ctx, &rows);
    let cols_buf = params_buffer(ctx, &cols);
    let grid = MetalContext::size(cols as u64, 1, 1);
    let tg = MetalContext::size(cols.min(256) as u64, 1, 1);
    dispatch_threads_sync!(ctx, "ternary_absmean", grid, tg,
        0 => weights, 1 => absmean, 2 => &rows_buf, 3 => &cols_buf
    );
}

/// BitNet: pack float weights to ternary (2 bits per weight, 16 per u32).
pub fn gpu_ternary_pack(ctx: &Arc<MetalContext>, weights: &GpuBuffer, absmean: &GpuBuffer, packed: &GpuBuffer, rows: u32, cols: u32) {
    let rows_buf = params_buffer(ctx, &rows);
    let cols_buf = params_buffer(ctx, &cols);
    let packed_rows = rows.div_ceil(16);
    let grid = MetalContext::size(cols as u64, packed_rows as u64, 1);
    let tg = MetalContext::size(cols.min(32) as u64, packed_rows.min(32) as u64, 1);
    dispatch_sync!(ctx, "ternary_pack", grid, tg,
        0 => weights, 1 => absmean, 2 => packed, 3 => &rows_buf, 4 => &cols_buf
    );
}

/// Lion optimizer update: simpler than AdamW, 2x less memory (no variance buffer).
pub fn gpu_lion_update(ctx: &Arc<MetalContext>, param: &GpuBuffer, grad: &GpuBuffer, m: &GpuBuffer, size: u32, p: LionParams) {
    let LionParams { lr, beta1, beta2, weight_decay } = p;
    #[repr(C)]
    struct Params { lr: f32, beta1: f32, beta2: f32, weight_decay: f32 }
    let params = Params { lr, beta1, beta2, weight_decay };
    let params_buf = params_buffer(ctx, &params);
    let size_buf = params_buffer(ctx, &size);
    let tpg = 256u64;
    let grid = MetalContext::size((size as u64).div_ceil(tpg) * tpg, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_threads_sync!(ctx, "lion_update", grid, tg,
        0 => param, 1 => grad, 2 => m, 3 => &params_buf, 4 => &size_buf
    );
}

/// Sophia optimizer: second-order with diagonal Hessian estimate. 2x faster convergence.
pub fn gpu_sophia_update(ctx: &Arc<MetalContext>, param: &GpuBuffer, grad: &GpuBuffer, m: &GpuBuffer, h: &GpuBuffer, size: u32, p: SophiaParams) {
    let SophiaParams { lr, beta1, beta2, eps, rho, weight_decay } = p;
    #[repr(C)]
    struct Params { lr: f32, beta1: f32, beta2: f32, eps: f32, rho: f32, weight_decay: f32 }
    let params = Params { lr, beta1, beta2, eps, rho, weight_decay };
    let params_buf = params_buffer(ctx, &params);
    let size_buf = params_buffer(ctx, &size);
    let tpg = 256u64;
    let grid = MetalContext::size((size as u64).div_ceil(tpg) * tpg, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_threads_sync!(ctx, "sophia_update", grid, tg,
        0 => param, 1 => grad, 2 => m, 3 => h, 4 => &params_buf, 5 => &size_buf
    );
}

/// Scale each row by a different scalar: output[r][c] = input[r][c] * scales[r]
pub fn gpu_scale_rows(ctx: &Arc<MetalContext>, input: &GpuBuffer, scales: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32) {
    let rows_buf = params_buffer(ctx, &rows);
    let cols_buf = params_buffer(ctx, &cols);
    let grid = MetalContext::size(cols as u64, rows as u64, 1);
    let tg = MetalContext::size(cols.min(32) as u64, rows.min(32) as u64, 1);
    dispatch_sync!(ctx, "scale_rows", grid, tg,
        0 => input, 1 => scales, 2 => output, 3 => &rows_buf, 4 => &cols_buf
    );
}

/// Row-wise dot reduce: output[r] = sum_c(a[r][c] * b[r][c]). Single dispatch.
pub fn gpu_row_dot_reduce(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32) {
    let rows_buf = params_buffer(ctx, &rows);
    let cols_buf = params_buffer(ctx, &cols);
    let tpg = 256u64;
    let groups = (rows as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups * tpg, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_threads_sync!(ctx, "row_dot_reduce", grid, tg,
        0 => a, 1 => b, 2 => output, 3 => &rows_buf, 4 => &cols_buf
    );
}

/// MoE: gather tokens for one expert into contiguous buffer.
pub fn gpu_moe_gather(ctx: &Arc<MetalContext>, input: &GpuBuffer, indices: &GpuBuffer, gathered: &GpuBuffer, n_routed: u32, dim: u32) {
    let n_buf = params_buffer(ctx, &n_routed);
    let d_buf = params_buffer(ctx, &dim);
    let grid = MetalContext::size(n_routed as u64, dim as u64, 1);
    let tg = MetalContext::size(n_routed.min(32) as u64, dim.min(32) as u64, 1);
    dispatch_sync!(ctx, "moe_gather", grid, tg,
        0 => input, 1 => indices, 2 => gathered, 3 => &n_buf, 4 => &d_buf
    );
}

/// MoE: scatter-add weighted expert output back to combined output.
pub fn gpu_moe_scatter_add(ctx: &Arc<MetalContext>, expert_out: &GpuBuffer, indices: &GpuBuffer, weights: &GpuBuffer, combined: &GpuBuffer, n_routed: u32, dim: u32) {
    let n_buf = params_buffer(ctx, &n_routed);
    let d_buf = params_buffer(ctx, &dim);
    let grid = MetalContext::size(n_routed as u64, dim as u64, 1);
    let tg = MetalContext::size(n_routed.min(32) as u64, dim.min(32) as u64, 1);
    dispatch_sync!(ctx, "moe_scatter_add", grid, tg,
        0 => expert_out, 1 => indices, 2 => weights, 3 => combined, 4 => &n_buf, 5 => &d_buf
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
    gpu_causal_mask_window(ctx, scores, batch_heads, seq_q, seq_k, offset, 0)
}

/// Causal mask with optional sliding window.
pub fn gpu_causal_mask_window(
    ctx: &Arc<MetalContext>,
    scores: &GpuBuffer,
    batch_heads: u32,
    seq_q: u32,
    seq_k: u32,
    offset: u32,
    window: u32,
) {
    #[repr(C)]
    struct Params { batch_heads: u32, seq_q: u32, seq_k: u32, offset: u32, window: u32 }
    let params = Params { batch_heads, seq_q, seq_k, offset, window };
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
pub fn gpu_strided_batch_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, d: StridedCopyDims) {
    let StridedCopyDims { bh, src_seq_len, dst_stride, dst_offset, dim } = d;
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
    teacher_logits: &GpuBuffer, student_logits: &GpuBuffer, losses: &GpuBuffer, grad_student: &GpuBuffer,
    d: KlDims,
) {
    let KlDims { batch_size, vocab_size, temperature } = d;
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

/// Fused scale + causal mask + softmax. Input: [total_rows, seq_k].
pub fn gpu_scaled_causal_softmax(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, d: SoftmaxDims) {
    let SoftmaxDims { total_rows, seq_q, seq_k, scale, kv_offset } = d;
    gpu_scaled_causal_softmax_window(ctx, input, output, SoftmaxDims { total_rows, seq_q, seq_k, scale, kv_offset }, 0)
}

/// Scaled causal softmax with optional sliding window.
pub fn gpu_scaled_causal_softmax_window(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, d: SoftmaxDims, window: u32) {
    let SoftmaxDims { total_rows, seq_q, seq_k, scale, kv_offset } = d;
    #[repr(C)]
    struct Params { seq_q: u32, seq_k: u32, scale: f32, kv_offset: u32, window: u32 }
    let params = Params { seq_q, seq_k, scale, kv_offset, window };
    let params_buf = params_buffer(ctx, &params);
    let threads_per_group = next_power_of_2_clamped(seq_k as u64);
    let grid = MetalContext::size(total_rows as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);
    dispatch_sync!(ctx, "scaled_causal_softmax", grid, tg,
        0 => input, 1 => output, 2 => &params_buf
    );
}

/// MEGA-KERNEL: Full SwiGLU FFN block in one dispatch.
/// Computes: output = x + (silu(norm(x) @ W1) * (norm(x) @ W3)) @ W2
/// For d_model ≤ 256, d_ff ≤ 1024 (fits in threadgroup memory).
/// Eliminates 5 dispatches + 4 intermediate buffer allocations.
pub fn gpu_mega_ffn(
    ctx: &Arc<MetalContext>, x: &GpuBuffer, norm_w: &GpuBuffer, w: FfnWeights, output: &GpuBuffer, d: MegaFfnDims,
) {
    let FfnWeights { w1, w2, w3 } = w;
    let MegaFfnDims { batch_tokens, d_model, d_ff, eps } = d;
    assert!(d_model <= 2048, "mega_ffn requires d_model <= 2048 (got {})", d_model);
    assert!(d_ff <= 4096, "mega_ffn requires d_ff <= 4096 (got {})", d_ff);

    #[repr(C)]
    struct Params { batch_tokens: u32, d_model: u32, d_ff: u32, eps: f32 }
    let params = Params { batch_tokens, d_model, d_ff, eps };
    let params_buf = params_buffer(ctx, &params);

    // One threadgroup per token, 256 threads per group
    let grid = MetalContext::size(batch_tokens as u64, 1, 1);
    let tg = MetalContext::size(256, 1, 1);

    dispatch_sync!(ctx, "mega_ffn", grid, tg,
        0 => x, 1 => norm_w, 2 => w1, 3 => w2, 4 => w3, 5 => output, 6 => &params_buf
    );
}

/// Fused transpose [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim] + RoPE.
/// Eliminates the intermediate buffer and 1 dispatch (transpose + RoPE → 1 kernel).
pub fn gpu_transpose_rope(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, d: TrRopeDims) {
    let TrRopeDims { batch, seq, n_heads, head_dim, offset, theta } = d;
    #[repr(C)]
    struct Params { batch: u32, seq: u32, n_heads: u32, head_dim: u32, offset: u32, theta: f32 }
    let params = Params { batch, seq, n_heads, head_dim, offset, theta };
    let params_buf = params_buffer(ctx, &params);

    let total = (batch * n_heads * seq * head_dim) as u64;
    let tpg = 256u64;
    let groups = total.div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_threads_sync!(ctx, "transpose_rope", grid, tg,
        0 => input, 1 => output, 2 => &params_buf
    );
}

/// Backward for fused transpose+RoPE: inverse RoPE + inverse transpose in one dispatch.
pub fn gpu_transpose_rope_backward(ctx: &Arc<MetalContext>, grad_out: &GpuBuffer, grad_in: &GpuBuffer, d: TrRopeDims) {
    let TrRopeDims { batch, seq, n_heads, head_dim, offset, theta } = d;
    #[repr(C)]
    struct Params { batch: u32, seq: u32, n_heads: u32, head_dim: u32, offset: u32, theta: f32 }
    let params = Params { batch, seq, n_heads, head_dim, offset, theta };
    let params_buf = params_buffer(ctx, &params);

    let total = (batch * seq * n_heads * head_dim) as u64;
    let tpg = 256u64;
    let groups = total.div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);

    dispatch_threads_sync!(ctx, "transpose_rope_backward", grid, tg,
        0 => grad_out, 1 => grad_in, 2 => &params_buf
    );
}

/// Pre-compute inv_rms per row: inv_rms[i] = 1/sqrt(mean(A[i]^2) + eps)
pub fn gpu_compute_inv_rms(ctx: &Arc<MetalContext>, input: &GpuBuffer, inv_rms: &GpuBuffer,
    rows: u32, cols: u32, eps: f32)
{
    #[repr(C)]
    struct Params { rows: u32, cols: u32, eps: f32 }
    let params = Params { rows, cols, eps };
    let params_buf = params_buffer(ctx, &params);
    let tpg = next_power_of_2_clamped(cols as u64);
    let grid = MetalContext::size(rows as u64, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "compute_inv_rms", grid, tg, 0 => input, 1 => inv_rms, 2 => &params_buf);
}

/// Fused RMSNorm + Matmul: C = rms_norm(A, weight, eps) @ B in 2 dispatches.
/// Eliminates the intermediate [M, K] normalized buffer.
pub fn gpu_fused_norm_matmul(
    ctx: &Arc<MetalContext>, a: &GpuBuffer, norm_weight: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: NormMatmulDims,
) {
    let NormMatmulDims { m, n, k, eps } = d;
    // Phase 1: compute inv_rms per row
    let inv_rms = ctx.alloc_buffer(m as usize * 4);
    gpu_compute_inv_rms(ctx, a, &inv_rms, m, k, eps);

    // Phase 2: fused norm+matmul
    #[repr(C)]
    struct Params { m: u32, n: u32, k: u32, eps: f32 }
    let params = Params { m, n, k, eps };
    let params_buf = params_buffer(ctx, &params);

    let tile = 32u64;
    let groups_x = (n as u64).div_ceil(tile);
    let groups_y = (m as u64).div_ceil(tile);
    let grid = MetalContext::size(groups_x, groups_y, 1);
    let tg = MetalContext::size(64, 1, 1);

    dispatch_sync!(ctx, "fused_norm_matmul", grid, tg,
        0 => a, 1 => norm_weight, 2 => b, 3 => c, 4 => &inv_rms, 5 => &params_buf
    );
}

/// AXPY: y[i] += alpha * x[i]. Fused scale+add in 1 dispatch.
pub fn gpu_axpy(ctx: &Arc<MetalContext>, y: &GpuBuffer, x: &GpuBuffer, size: u32, alpha: f32) {
    #[repr(C)]
    struct Params { size: u32, alpha: f32 }
    let params = Params { size, alpha };
    let params_buf = params_buffer(ctx, &params);
    let tpg = 256u64;
    let grid = MetalContext::size((size as u64).div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "axpy", grid, tg, 0 => y, 1 => x, 2 => &params_buf);
}

/// ReLU: output[i] = max(input[i], 0)
pub fn gpu_relu(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);
    let tpg = 256u64;
    let grid = MetalContext::size((size as u64).div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "relu", grid, tg, 0 => input, 1 => output, 2 => &params_buf);
}

/// Broadcast a `[cols]` vector to `[rows, cols]` (out[r*cols+c] = vec[c]). Direct copy.
pub fn gpu_broadcast_rows(ctx: &Arc<MetalContext>, vec: &GpuBuffer, out: &GpuBuffer, rows: u32, cols: u32) {
    #[repr(C)]
    struct Params { rows: u32, cols: u32 }
    let params = Params { rows, cols };
    let params_buf = params_buffer(ctx, &params);
    let total = (rows * cols) as u64;
    let tpg = 256u64;
    let grid = MetalContext::size(total.div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "broadcast_rows", grid, tg, 0 => vec, 1 => out, 2 => &params_buf);
}

/// Elementwise exp: output = exp(input) (input clamped to ≤80 for overflow safety).
pub fn gpu_exp(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);
    let tpg = 256u64;
    let grid = MetalContext::size((size as u64).div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "exp_fwd", grid, tg, 0 => input, 1 => output, 2 => &params_buf);
}

/// ReLU backward: grad_input = grad_output * (input > 0)
pub fn gpu_relu_backward(ctx: &Arc<MetalContext>, input: &GpuBuffer, grad_output: &GpuBuffer, grad_input: &GpuBuffer, size: u32) {
    #[repr(C)]
    struct Params { size: u32 }
    let params = Params { size };
    let params_buf = params_buffer(ctx, &params);
    let tpg = 256u64;
    let grid = MetalContext::size((size as u64).div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "relu_backward", grid, tg, 0 => input, 1 => grad_output, 2 => grad_input, 3 => &params_buf);
}

/// EMA update: ema[i] = decay * ema[i] + (1-decay) * src[i]. Single dispatch for all elements.
pub fn gpu_ema_update(ctx: &Arc<MetalContext>, ema: &GpuBuffer, src: &GpuBuffer, size: u32, decay: f32) {
    #[repr(C)]
    struct Params { size: u32, decay: f32 }
    let params = Params { size, decay };
    let params_buf = params_buffer(ctx, &params);
    let tpg = 256u64;
    let grid = MetalContext::size((size as u64).div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "ema_update", grid, tg, 0 => ema, 1 => src, 2 => &params_buf);
}

/// LogSumExp per row: output[i] = log(sum_j(exp(input[i*cols + j]))). Numerically stable.
pub fn gpu_logsumexp(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32) {
    #[repr(C)]
    struct Params { rows: u32, cols: u32 }
    let params = Params { rows, cols };
    let params_buf = params_buffer(ctx, &params);
    let threads_per_group = next_power_of_2_clamped(cols as u64);
    let grid = MetalContext::size(rows as u64, 1, 1);
    let tg = MetalContext::size(threads_per_group, 1, 1);
    dispatch_sync!(ctx, "logsumexp", grid, tg, 0 => input, 1 => output, 2 => &params_buf);
}

/// Out-of-place scale: dst[i] = src[i] * factor. Replaces copy+scale_inplace (2→1 dispatch).
pub fn gpu_scale_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, size: u32, scale: f32) {
    #[repr(C)]
    struct Params { size: u32, scale: f32 }
    let params = Params { size, scale };
    let params_buf = params_buffer(ctx, &params);
    let tpg = 256u64;
    let groups = (size as u64).div_ceil(tpg);
    let grid = MetalContext::size(groups, 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "scale_copy", grid, tg, 0 => src, 1 => dst, 2 => &params_buf);
}

/// Column-wise copy: src[rows, src_cols] → dst[rows, dst_cols] at col_offset.
pub fn gpu_concat_cols(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer,
    rows: u32, src_cols: u32, dst_cols: u32, col_offset: u32)
{
    #[repr(C)]
    struct Params { rows: u32, src_cols: u32, dst_cols: u32, col_offset: u32 }
    let params = Params { rows, src_cols, dst_cols, col_offset };
    let params_buf = params_buffer(ctx, &params);
    let total = (rows as u64) * (src_cols as u64);
    let tpg = 256u64;
    let grid = MetalContext::size(total.div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "concat_cols", grid, tg, 0 => src, 1 => dst, 2 => &params_buf);
}

/// Column-wise slice: extract cols [offset..offset+dst_cols) from [rows, src_cols].
pub fn gpu_slice_cols(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer,
    rows: u32, src_cols: u32, dst_cols: u32, col_offset: u32)
{
    #[repr(C)]
    struct Params { rows: u32, src_cols: u32, dst_cols: u32, col_offset: u32 }
    let params = Params { rows, src_cols, dst_cols, col_offset };
    let params_buf = params_buffer(ctx, &params);
    let total = (rows as u64) * (dst_cols as u64);
    let tpg = 256u64;
    let grid = MetalContext::size(total.div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "slice_cols", grid, tg, 0 => src, 1 => dst, 2 => &params_buf);
}

/// Single-kernel KV head expansion for GQA forward.
pub fn gpu_repeat_kv(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer,
    n_kv_total: u32, group_size: u32, seq_len: u32, head_dim: u32)
{
    #[repr(C)]
    struct Params { n_kv_total: u32, group_size: u32, seq_len: u32, head_dim: u32 }
    let params = Params { n_kv_total, group_size, seq_len, head_dim };
    let params_buf = params_buffer(ctx, &params);
    let total = (n_kv_total as u64) * (group_size as u64) * (seq_len as u64) * (head_dim as u64);
    let tpg = 256u64;
    let grid = MetalContext::size(total.div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "repeat_kv", grid, tg, 0 => input, 1 => output, 2 => &params_buf);
}

/// Single-kernel backward for repeat_kv: sum group_size gradient blocks.
pub fn gpu_repeat_kv_backward(ctx: &Arc<MetalContext>, out_grad: &GpuBuffer, kv_grad: &GpuBuffer,
    n_kv_total: u32, group_size: u32, seq_len: u32, head_dim: u32)
{
    #[repr(C)]
    struct Params { n_kv_total: u32, group_size: u32, seq_len: u32, head_dim: u32 }
    let params = Params { n_kv_total, group_size, seq_len, head_dim };
    let params_buf = params_buffer(ctx, &params);
    let total = (n_kv_total as u64) * (seq_len as u64) * (head_dim as u64);
    let tpg = 256u64;
    let grid = MetalContext::size(total.div_ceil(tpg), 1, 1);
    let tg = MetalContext::size(tpg, 1, 1);
    dispatch_sync!(ctx, "repeat_kv_backward", grid, tg, 0 => out_grad, 1 => kv_grad, 2 => &params_buf);
}
