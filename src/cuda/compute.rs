//! CUDA compute dispatch functions — same API as metal/compute.rs.
//! Each function launches a CUDA kernel with the appropriate grid/block dimensions.

use super::MetalContext; // aliased CudaContext
use cudarc::driver::{CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

type GpuBuffer = CudaSlice<f32>;

fn launch_cfg(threads: u32, blocks: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (threads, 1, 1),
        grid_dim: (blocks, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_cfg_2d(bx: u32, by: u32, tx: u32, ty: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (tx, ty, 1),
        grid_dim: (bx, by, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_cfg_3d(bx: u32, by: u32, bz: u32, tx: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (tx, 1, 1),
        grid_dim: (bx, by, bz),
        shared_mem_bytes: 0,
    }
}

// ===== Matmul =====

pub fn gpu_matmul(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let tile = 32u32;
    let cfg = launch_cfg_3d(n.div_ceil(tile), m.div_ceil(tile), 1, 64);
    let f = ctx.device.get_func("andreai", "matmul_tiled").unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k)) }.unwrap();
}

pub fn gpu_matmul_trans_b(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let tile = 32u32;
    let cfg = launch_cfg_3d(n.div_ceil(tile), m.div_ceil(tile), 1, 64);
    let f = ctx
        .device
        .get_func("andreai", "matmul_tiled_trans_b")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k)) }.unwrap();
}

pub fn gpu_matmul_trans_a(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    k: u32,
    n: u32,
) {
    let tile = 32u32;
    let cfg = launch_cfg_3d(n.div_ceil(tile), k.div_ceil(tile), 1, 64);
    let f = ctx
        .device
        .get_func("andreai", "matmul_trans_a_tiled")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, k, n)) }.unwrap();
}

// ===== Softmax / Norm =====

fn next_power_of_2_clamped(n: u64) -> u64 {
    let clamped = n.min(256).max(1) as u32;
    clamped.next_power_of_two().min(256) as u64
}

pub fn gpu_softmax(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    let tpg = next_power_of_2_clamped(cols as u64) as u32;
    let cfg = launch_cfg_2d(rows, 1, tpg, 1);
    let f = ctx.device.get_func("andreai", "softmax").unwrap();
    unsafe { f.launch(cfg, (input, output, rows, cols)) }.unwrap();
}

pub fn gpu_rms_norm(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    output: &GpuBuffer,
    rows: u32,
    cols: u32,
    eps: f32,
) {
    let tpg = next_power_of_2_clamped(cols as u64) as u32;
    let cfg = launch_cfg_2d(rows, 1, tpg, 1);
    let f = ctx.device.get_func("andreai", "rms_norm").unwrap();
    unsafe { f.launch(cfg, (input, weight, output, rows, cols, eps)) }.unwrap();
}

// ===== Element-wise =====

pub fn gpu_add(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "add_kernel").unwrap();
    unsafe { f.launch(cfg, (a, b, c, size)) }.unwrap();
}

pub fn gpu_add_inplace(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "add_inplace").unwrap();
    unsafe { f.launch(cfg, (a, b, size)) }.unwrap();
}

pub fn gpu_mul(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "mul_kernel").unwrap();
    unsafe { f.launch(cfg, (a, b, c, size)) }.unwrap();
}

pub fn gpu_scale(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, scale: f32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "scale_kernel").unwrap();
    unsafe { f.launch(cfg, (data, size, scale)) }.unwrap();
}

pub fn gpu_fill(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, value: f32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "fill_kernel").unwrap();
    unsafe { f.launch(cfg, (data, size, value)) }.unwrap();
}

pub fn gpu_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "copy_kernel").unwrap();
    unsafe { f.launch(cfg, (src, dst, size)) }.unwrap();
}

pub fn gpu_silu_gate(
    ctx: &Arc<MetalContext>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    output: &GpuBuffer,
    size: u32,
) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "silu_gate").unwrap();
    unsafe { f.launch(cfg, (gate, up, output, size)) }.unwrap();
}

// ===== RoPE =====

pub fn gpu_rope(
    ctx: &Arc<MetalContext>,
    data: &GpuBuffer,
    total_rows: u32,
    seq_len: u32,
    head_dim: u32,
    offset: u32,
    theta: f32,
) {
    let pairs = head_dim / 2;
    let cfg = launch_cfg_3d(total_rows, seq_len, 1, pairs);
    let f = ctx.device.get_func("andreai", "rope").unwrap();
    unsafe { f.launch(cfg, (data, total_rows, seq_len, head_dim, offset, theta)) }.unwrap();
}

// ===== Loss =====

pub fn gpu_cross_entropy(
    ctx: &Arc<MetalContext>,
    logits: &GpuBuffer,
    targets: &CudaSlice<u32>,
    losses: &GpuBuffer,
    grad: &GpuBuffer,
    batch: u32,
    vocab: u32,
) {
    let tpg = next_power_of_2_clamped(vocab as u64) as u32;
    let cfg = launch_cfg_2d(batch, 1, tpg, 1);
    let f = ctx.device.get_func("andreai", "cross_entropy").unwrap();
    unsafe { f.launch(cfg, (logits, targets, losses, grad, batch, vocab)) }.unwrap();
    // Loss is the mean over batch; scale grad by 1/batch to match (Metal does this in-kernel).
    // Done here in Rust because the in-kernel `batch` was unreliable for this scale under nvrtc.
    gpu_scale(ctx, grad, batch * vocab, 1.0 / batch as f32);
}

pub fn gpu_reduce_sum(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    let tpg = next_power_of_2_clamped(size as u64) as u32;
    let cfg = launch_cfg(tpg, 1);
    let f = ctx.device.get_func("andreai", "reduce_sum").unwrap();
    unsafe { f.launch(cfg, (input, output, size)) }.unwrap();
}

// ===== Optimizer =====

pub struct AdamWHyperparams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u32,
    /// Per-element ceiling on the normalized update. 0 = disabled. (Mirrors metal.)
    pub update_clip: f32,
}

pub fn gpu_adamw_update(
    ctx: &Arc<MetalContext>,
    param: &GpuBuffer,
    grad: &GpuBuffer,
    m: &GpuBuffer,
    v: &GpuBuffer,
    size: u32,
    hp: &AdamWHyperparams,
) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "adamw_update").unwrap();
    unsafe {
        f.launch(
            cfg,
            (
                param,
                grad,
                m,
                v,
                size,
                hp.lr,
                hp.beta1,
                hp.beta2,
                hp.eps,
                hp.weight_decay,
                hp.step as i32,
                hp.update_clip,
            ),
        )
    }
    .unwrap();
}

// ===== Embedding =====

pub fn gpu_embedding_lookup(
    ctx: &Arc<MetalContext>,
    tokens: &CudaSlice<u32>,
    table: &GpuBuffer,
    output: &GpuBuffer,
    seq_len: u32,
    dim: u32,
) {
    let cfg = launch_cfg_3d(seq_len, dim.div_ceil(256), 1, 256);
    let f = ctx.device.get_func("andreai", "embedding_lookup").unwrap();
    unsafe { f.launch(cfg, (table, tokens, output, seq_len, dim)) }.unwrap();
}

// ===== Cast =====

pub fn gpu_cast_f32_to_f16(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    size: u32,
) {
    let n_words = size.div_ceil(2); // 2 halves packed per 4-byte word
    let cfg = launch_cfg(256, n_words.div_ceil(256));
    let f = ctx.device.get_func("andreai", "cast_f32_to_f16").unwrap();
    unsafe { f.launch(cfg, (input, output, size)) }.unwrap();
}

// ===== Norm check =====

pub fn gpu_l2_norm_check_into(
    ctx: &Arc<MetalContext>,
    data: &GpuBuffer,
    size: u32,
    output: &GpuBuffer,
) {
    let tpg = next_power_of_2_clamped(size as u64) as u32;
    let cfg = launch_cfg(tpg, 1);
    let f = ctx.device.get_func("andreai", "l2_norm_check").unwrap();
    unsafe { f.launch(cfg, (data, output, size)) }.unwrap();
}

pub fn gpu_l2_norm_check(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32) -> (f32, bool) {
    let output = ctx.alloc_buffer(8);
    gpu_l2_norm_check_into(ctx, data, size, &output);
    let vals = MetalContext::read_buffer(&output, 2);
    (vals[0], vals[1] > 0.5)
}

pub fn gpu_buffer_copy(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    src_offset: u32,
    dst_offset: u32,
    count: u32,
) {
    let cfg = launch_cfg(256, count.div_ceil(256));
    let f = ctx.device.get_func("andreai", "buffer_copy").unwrap();
    unsafe { f.launch(cfg, (src, dst, src_offset, dst_offset, count)) }.unwrap();
}

// Causal mask
pub fn gpu_causal_mask(
    ctx: &Arc<MetalContext>,
    scores: &GpuBuffer,
    batch_heads: u32,
    seq_q: u32,
    seq_k: u32,
    offset: u32,
) {
    let cfg = launch_cfg_3d(batch_heads, seq_q, seq_k.div_ceil(256), 256);
    let f = ctx.device.get_func("andreai", "causal_mask").unwrap();
    unsafe { f.launch(cfg, (scores, batch_heads, seq_q, seq_k, offset)) }.unwrap();
}

// ===== AUTO-GENERATED CUDA STUBS (mirror metal/compute.rs; unimplemented until ported) =====
#[derive(Clone, Copy)]
pub struct BatchedDims {
    pub batch: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

#[derive(Clone, Copy)]
pub struct RopeDims {
    pub total_rows: u32,
    pub seq_len: u32,
    pub head_dim: u32,
    pub offset: u32,
    pub theta: f32,
}

#[derive(Clone, Copy)]
pub struct TrRopeDims {
    pub batch: u32,
    pub seq: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    pub offset: u32,
    pub theta: f32,
}

#[derive(Clone, Copy)]
pub struct FlashDims {
    pub batch_heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
    pub kv_offset: u32,
}

#[derive(Clone, Copy)]
pub struct SoftmaxDims {
    pub total_rows: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub scale: f32,
    pub kv_offset: u32,
}

#[derive(Clone, Copy)]
pub struct LionParams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
}

#[derive(Clone, Copy)]
pub struct SophiaParams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub rho: f32,
    pub weight_decay: f32,
}

#[derive(Clone, Copy)]
pub struct RmsResDims {
    pub rows: u32,
    pub cols: u32,
    pub eps: f32,
}

#[derive(Clone, Copy)]
pub struct StridedCopyDims {
    pub bh: u32,
    pub src_seq_len: u32,
    pub dst_stride: u32,
    pub dst_offset: u32,
    pub dim: u32,
}

#[derive(Clone, Copy)]
pub struct KlDims {
    pub batch_size: u32,
    pub vocab_size: u32,
    pub temperature: f32,
}

#[derive(Clone, Copy)]
pub struct MegaFfnDims {
    pub batch_tokens: u32,
    pub d_model: u32,
    pub d_ff: u32,
    pub eps: f32,
}

#[derive(Clone, Copy)]
pub struct FfnWeights<'a> {
    pub w1: &'a GpuBuffer,
    pub w2: &'a GpuBuffer,
    pub w3: &'a GpuBuffer,
}

#[derive(Clone, Copy)]
pub struct NormMatmulDims {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub eps: f32,
}

#[derive(Clone, Copy)]
pub struct FlashBwdBufs<'a> {
    pub q: &'a GpuBuffer,
    pub k: &'a GpuBuffer,
    pub v: &'a GpuBuffer,
    pub output: &'a GpuBuffer,
    pub d_out: &'a GpuBuffer,
    pub d_buf: &'a GpuBuffer,
    pub dq: &'a GpuBuffer,
    pub dk: &'a GpuBuffer,
    pub dv: &'a GpuBuffer,
}

pub struct Adam8Buffers<'a> {
    pub m_q: &'a GpuBuffer,
    pub v_q: &'a GpuBuffer,
    pub m_scale: &'a GpuBuffer,
    pub v_scale: &'a GpuBuffer,
}

#[derive(Clone, Copy)]
pub struct BlockMeanDims {
    pub bh: u32,
    pub seq: u32,
    pub hd: u32,
    pub nb: u32,
    pub block_size: u32,
}

#[derive(Clone, Copy)]
pub struct BlockSparseDims {
    pub bh: u32,
    pub seq: u32,
    pub nb: u32,
    pub block_size: u32,
    pub top_k: u32,
}

#[derive(Clone, Copy)]
pub struct GatherDims {
    pub bh: u32,
    pub nb: u32,
    pub seq: u32,
    pub hd: u32,
    pub block: u32,
    pub k_sel: u32,
}

pub struct RmsNormBackwardParams {
    pub rows: u32,
    pub cols: u32,
    pub eps: f32,
}

pub fn gpu_matmul_fp32(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let tile = 32u32;
    let cfg = launch_cfg_3d(n.div_ceil(tile), m.div_ceil(tile), 1, 64);
    let f = ctx.device.get_func("andreai", "matmul_tiled_fp32").unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k)) }.unwrap();
}

pub fn gpu_matmul_bf16(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let tile = 32u32;
    let cfg = launch_cfg_3d(n.div_ceil(tile), m.div_ceil(tile), 1, 64);
    let f = ctx.device.get_func("andreai", "matmul_tiled_bf16").unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k)) }.unwrap();
}

// No hardware MMA path on CUDA: the "simdgroup" matmul = the precise fp32 tiled kernel (matches the
// fp32 reference the tests check against at 1e-3); the trans/f16 variants alias their tiled equivalents.
pub fn gpu_matmul_simdgroup(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    gpu_matmul_fp32(ctx, a, b, c, m, n, k)
}

pub fn gpu_matmul_simdgroup_f16(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    gpu_matmul_f16(ctx, a, b, c, m, n, k)
}

pub fn gpu_matmul_trans_b_simdgroup(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    gpu_matmul_trans_b(ctx, a, b, c, m, n, k)
}

pub fn gpu_matmul_trans_a_simdgroup(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    k: u32,
    n: u32,
) {
    gpu_matmul_trans_a(ctx, a, b, c, m, k, n)
}

pub fn gpu_cast_f16_to_f32(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    size: u32,
) {
    let n_words = size.div_ceil(2); // unpack 2 halves per 4-byte word (matches cast_f32_to_f16)
    let cfg = launch_cfg(256, n_words.div_ceil(256));
    let f = ctx.device.get_func("andreai", "cast_f16_to_f32").unwrap();
    unsafe { f.launch(cfg, (input, output, size)) }.unwrap();
}

// Real FP16-input matmuls: read packed-half buffers (from cast_to_f16, now a real pack on CUDA).
// Tiling/compute identical to the f32 kernels (which cast f32→half in-tile) → bit-identical values,
// half the input bandwidth. m/n/k = logical dims; inputs are half-packed (2 halves per 4-byte word).
pub fn gpu_matmul_f16(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let cfg = launch_cfg_3d(n.div_ceil(32), m.div_ceil(32), 1, 64);
    let f = ctx.device.get_func("andreai", "matmul_tiled_f16").unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k)) }.unwrap();
}
pub fn gpu_matmul_trans_b_f16(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let cfg = launch_cfg_3d(n.div_ceil(32), m.div_ceil(32), 1, 64);
    let f = ctx
        .device
        .get_func("andreai", "matmul_tiled_trans_b_f16")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k)) }.unwrap();
}
pub fn gpu_matmul_trans_a_f16(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    k: u32,
    n: u32,
) {
    let cfg = launch_cfg_3d(n.div_ceil(32), k.div_ceil(32), 1, 64);
    let f = ctx
        .device
        .get_func("andreai", "matmul_trans_a_tiled_f16")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, k, n)) }.unwrap();
}
pub fn gpu_batched_matmul_f16(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
) {
    let BatchedDims { batch, m, n, k } = d;
    let cfg = launch_cfg_3d(n.div_ceil(32), m.div_ceil(32), batch, 64);
    let f = ctx
        .device
        .get_func("andreai", "batched_matmul_tiled_f16")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k, batch)) }.unwrap();
}
pub fn gpu_batched_matmul_trans_b_f16(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
) {
    let BatchedDims { batch, m, n, k } = d;
    let cfg = launch_cfg_3d(n.div_ceil(32), m.div_ceil(32), batch, 64);
    let f = ctx
        .device
        .get_func("andreai", "batched_matmul_tiled_trans_b_f16")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k, batch)) }.unwrap();
}
pub fn gpu_batched_matmul_trans_a_f16(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
) {
    let BatchedDims { batch, m, n, k } = d;
    let cfg = launch_cfg_3d(n.div_ceil(32), k.div_ceil(32), batch, 64);
    let f = ctx
        .device
        .get_func("andreai", "batched_matmul_tiled_trans_a_f16")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, k, n, batch)) }.unwrap();
}

pub fn gpu_batched_matmul(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
) {
    let BatchedDims { batch, m, n, k } = d;
    let cfg = launch_cfg_3d(n.div_ceil(32), m.div_ceil(32), batch, 64);
    let f = ctx
        .device
        .get_func("andreai", "batched_matmul_tiled")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k, batch)) }.unwrap();
}

pub fn gpu_batched_matmul_simdgroup(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
) {
    gpu_batched_matmul(ctx, a, b, c, d)
}

pub fn gpu_batched_matmul_trans_b_simdgroup(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
) {
    gpu_batched_matmul_trans_b(ctx, a, b, c, d)
}

pub fn gpu_batched_matmul_trans_a_simdgroup(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
) {
    gpu_batched_matmul_trans_a(ctx, a, b, c, d)
}

pub fn gpu_batched_matmul_trans_b(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
) {
    let BatchedDims { batch, m, n, k } = d;
    let cfg = launch_cfg_3d(n.div_ceil(32), m.div_ceil(32), batch, 64);
    let f = ctx
        .device
        .get_func("andreai", "batched_matmul_tiled_trans_b")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k, batch)) }.unwrap();
}

pub fn gpu_batched_matmul_gqa_trans_b(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
    group_size: u32,
) {
    assert!(group_size > 0, "GQA group_size must be positive");
    assert_eq!(
        d.batch % group_size,
        0,
        "GQA batch_heads must be divisible by group_size"
    );
    if group_size == 1 {
        return gpu_batched_matmul_trans_b(ctx, a, b, c, d);
    }
    let expanded_b = ctx.alloc_buffer((d.batch as usize * d.n as usize * d.k as usize) * 4);
    gpu_repeat_kv(
        ctx,
        b,
        &expanded_b,
        d.batch / group_size,
        group_size,
        d.n,
        d.k,
    );
    gpu_batched_matmul_trans_b(ctx, a, &expanded_b, c, d);
}

pub fn gpu_batched_matmul_gqa(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
    group_size: u32,
) {
    assert!(group_size > 0, "GQA group_size must be positive");
    assert_eq!(
        d.batch % group_size,
        0,
        "GQA batch_heads must be divisible by group_size"
    );
    if group_size == 1 {
        return gpu_batched_matmul(ctx, a, b, c, d);
    }
    let expanded_b = ctx.alloc_buffer((d.batch as usize * d.k as usize * d.n as usize) * 4);
    gpu_repeat_kv(
        ctx,
        b,
        &expanded_b,
        d.batch / group_size,
        group_size,
        d.k,
        d.n,
    );
    gpu_batched_matmul(ctx, a, &expanded_b, c, d);
}

pub fn gpu_batched_matmul_trans_a(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: BatchedDims,
) {
    let BatchedDims { batch, m, n, k } = d;
    let cfg = launch_cfg_3d(n.div_ceil(32), k.div_ceil(32), batch, 64);
    let f = ctx
        .device
        .get_func("andreai", "batched_matmul_tiled_trans_a")
        .unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, k, n, batch)) }.unwrap();
}

pub fn gpu_rms_norm_residual(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    residual: &GpuBuffer,
    weight: &GpuBuffer,
    output: &GpuBuffer,
    sum_out: &GpuBuffer,
    d: RmsResDims,
) {
    let cfg = launch_cfg_2d(d.rows, 1, next_power_of_2_clamped(d.cols as u64) as u32, 1);
    let f = ctx.device.get_func("andreai", "rms_norm_residual").unwrap();
    unsafe {
        f.launch(
            cfg,
            (
                input, residual, weight, output, sum_out, d.rows, d.cols, d.eps,
            ),
        )
    }
    .unwrap();
}

pub fn gpu_rope_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, d: RopeDims) {
    let n = d.total_rows * d.seq_len * d.head_dim;
    gpu_copy(ctx, src, dst, n);
    gpu_rope(
        ctx,
        dst,
        d.total_rows,
        d.seq_len,
        d.head_dim,
        d.offset,
        d.theta,
    );
}

pub fn gpu_rope_backward(
    ctx: &Arc<MetalContext>,
    data: &GpuBuffer,
    total_rows: u32,
    seq_len: u32,
    head_dim: u32,
    offset: u32,
    theta: f32,
) {
    let cfg = launch_cfg_3d(total_rows, seq_len, 1, head_dim / 2);
    let f = ctx.device.get_func("andreai", "rope_backward").unwrap();
    unsafe {
        f.launch(
            cfg,
            (data, data, total_rows, seq_len, head_dim, offset, theta),
        )
    }
    .unwrap();
}

pub fn gpu_rope_backward_copy(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    d: RopeDims,
) {
    let n = d.total_rows * d.seq_len * d.head_dim;
    gpu_copy(ctx, src, dst, n);
    gpu_rope_backward(
        ctx,
        dst,
        d.total_rows,
        d.seq_len,
        d.head_dim,
        d.offset,
        d.theta,
    );
}

pub fn gpu_silu(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "silu").unwrap();
    unsafe { f.launch(cfg, (input, output, size)) }.unwrap();
}

// 9 scalars exceed cudarc's 12-arg launch tuple when combined with 6 buffers, so pass them as one
// by-value POD struct (matches the kernel's `struct Adam8Params`).
#[repr(C)]
#[derive(Clone, Copy)]
struct Adam8Params {
    size: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    update_clip: f32,
}
unsafe impl DeviceRepr for Adam8Params {}

pub fn gpu_adamw_8bit_update(
    ctx: &Arc<MetalContext>,
    param: &GpuBuffer,
    grad: &GpuBuffer,
    state: &Adam8Buffers,
    size: u32,
    hp: &AdamWHyperparams,
) {
    let Adam8Buffers {
        m_q,
        v_q,
        m_scale,
        v_scale,
    } = *state;
    let AdamWHyperparams {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
        update_clip,
    } = *hp;
    let p = Adam8Params {
        size,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1: 1.0 - beta1.powi(step as i32),
        bias_correction2: 1.0 - beta2.powi(step as i32),
        update_clip,
    };
    let n_blocks = size.div_ceil(ADAM8_BLOCK as u32);
    let cfg = launch_cfg(ADAM8_BLOCK as u32, n_blocks); // one block (256 threads) per param-block
    let f = ctx.device.get_func("andreai", "adamw_8bit_update").unwrap();
    unsafe { f.launch(cfg, (param, grad, m_q, v_q, m_scale, v_scale, p)) }.unwrap();
}

pub fn gpu_flash_attention_forward(
    ctx: &Arc<MetalContext>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    d: FlashDims,
) {
    let FlashDims {
        batch_heads,
        seq_q,
        seq_k,
        head_dim,
        kv_offset,
    } = d;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_blocks = seq_q.div_ceil(32);
    let cfg = launch_cfg_3d(batch_heads, q_blocks, 1, 32); // grid (bh, q_blocks), 32 threads/block (one per query row)
    let f = ctx
        .device
        .get_func("andreai", "flash_attention_forward")
        .unwrap();
    unsafe {
        f.launch(
            cfg,
            (
                q,
                k,
                v,
                o,
                batch_heads,
                seq_q,
                seq_k,
                head_dim,
                scale,
                kv_offset,
            ),
        )
    }
    .unwrap();
}

pub fn gpu_flash_attn_precompute_d(
    ctx: &Arc<MetalContext>,
    d_out: &GpuBuffer,
    output: &GpuBuffer,
    d_buf: &GpuBuffer,
    total_rows: u32,
    head_dim: u32,
) {
    let cfg = launch_cfg(256, total_rows.div_ceil(256));
    let f = ctx
        .device
        .get_func("andreai", "flash_attn_precompute_d")
        .unwrap();
    unsafe { f.launch(cfg, (d_out, output, d_buf, total_rows, head_dim)) }.unwrap();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FlashBwdParams {
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    batch_heads: u32,
    scale: f32,
    kv_offset: u32,
}
unsafe impl DeviceRepr for FlashBwdParams {}

pub fn gpu_flash_attention_backward(ctx: &Arc<MetalContext>, b: FlashBwdBufs, d: FlashDims) {
    let FlashBwdBufs {
        q,
        k,
        v,
        output,
        d_out,
        d_buf,
        dq,
        dk,
        dv,
    } = b;
    let FlashDims {
        batch_heads,
        seq_q,
        seq_k,
        head_dim,
        kv_offset,
    } = d;
    // dq written fresh; dk/dv are atomic scatter targets pre-zeroed by the caller (autograd). 9 buffers
    // + 6 scalars exceed cudarc's 12-arg cap, so the scalars go in one by-value DeviceRepr struct.
    let p = FlashBwdParams {
        seq_q,
        seq_k,
        head_dim,
        batch_heads,
        scale: 1.0 / (head_dim as f32).sqrt(),
        kv_offset,
    };
    let q_blocks = seq_q.div_ceil(32);
    let cfg = launch_cfg_3d(batch_heads, q_blocks, 1, 32);
    let f = ctx
        .device
        .get_func("andreai", "flash_attention_backward")
        .unwrap();
    unsafe { f.launch(cfg, (q, k, v, output, d_out, d_buf, dq, dk, dv, p)) }.unwrap();
}

pub fn gpu_ternary_matmul(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    w_packed: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let cfg = launch_cfg_2d(n.div_ceil(16), m.div_ceil(16), 16, 16);
    let f = ctx.device.get_func("andreai", "ternary_matmul").unwrap();
    unsafe { f.launch(cfg, (a, w_packed, c, m, n, k)) }.unwrap();
}

pub fn gpu_ternary_absmean(
    ctx: &Arc<MetalContext>,
    weights: &GpuBuffer,
    absmean: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    let cfg = launch_cfg(256, cols.div_ceil(256));
    let f = ctx.device.get_func("andreai", "ternary_absmean").unwrap();
    unsafe { f.launch(cfg, (weights, absmean, rows, cols)) }.unwrap();
}

pub fn gpu_ternary_pack(
    ctx: &Arc<MetalContext>,
    weights: &GpuBuffer,
    absmean: &GpuBuffer,
    packed: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    let packed_rows = rows.div_ceil(16);
    let cfg = launch_cfg_2d(cols.div_ceil(16), packed_rows.div_ceil(16), 16, 16);
    let f = ctx.device.get_func("andreai", "ternary_pack").unwrap();
    unsafe { f.launch(cfg, (weights, absmean, packed, rows, cols)) }.unwrap();
}

pub fn gpu_lion_update(
    ctx: &Arc<MetalContext>,
    param: &GpuBuffer,
    grad: &GpuBuffer,
    m: &GpuBuffer,
    size: u32,
    p: LionParams,
) {
    let LionParams {
        lr,
        beta1,
        beta2,
        weight_decay,
    } = p;
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "lion_update").unwrap();
    unsafe { f.launch(cfg, (param, grad, m, size, lr, beta1, beta2, weight_decay)) }.unwrap();
}

pub fn gpu_sophia_update(
    ctx: &Arc<MetalContext>,
    param: &GpuBuffer,
    grad: &GpuBuffer,
    m: &GpuBuffer,
    h: &GpuBuffer,
    size: u32,
    p: SophiaParams,
) {
    let SophiaParams {
        lr,
        beta1,
        beta2,
        eps,
        rho,
        weight_decay,
    } = p;
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "sophia_update").unwrap();
    unsafe {
        f.launch(
            cfg,
            (
                param,
                grad,
                m,
                h,
                size,
                lr,
                beta1,
                beta2,
                eps,
                rho,
                weight_decay,
            ),
        )
    }
    .unwrap();
}

pub fn gpu_scale_rows(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    scales: &GpuBuffer,
    output: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    let cfg = launch_cfg(256, (rows * cols).div_ceil(256));
    let f = ctx.device.get_func("andreai", "scale_rows").unwrap();
    unsafe { f.launch(cfg, (input, scales, output, rows, cols)) }.unwrap();
}

pub fn gpu_row_dot_reduce(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    output: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    let cfg = launch_cfg(256, rows.div_ceil(256));
    let f = ctx.device.get_func("andreai", "row_dot_reduce").unwrap();
    unsafe { f.launch(cfg, (a, b, output, rows, cols)) }.unwrap();
}

pub fn gpu_moe_gather(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    indices: &CudaSlice<u32>,
    gathered: &GpuBuffer,
    n_routed: u32,
    dim: u32,
) {
    let cfg = launch_cfg_2d(n_routed.div_ceil(16), dim.div_ceil(16), 16, 16);
    let f = ctx.device.get_func("andreai", "moe_gather").unwrap();
    unsafe { f.launch(cfg, (input, indices, gathered, n_routed, dim)) }.unwrap();
}

pub fn gpu_moe_scatter_add(
    ctx: &Arc<MetalContext>,
    expert_out: &GpuBuffer,
    indices: &CudaSlice<u32>,
    weights: &GpuBuffer,
    combined: &GpuBuffer,
    n_routed: u32,
    dim: u32,
) {
    let cfg = launch_cfg_2d(n_routed.div_ceil(16), dim.div_ceil(16), 16, 16);
    let f = ctx.device.get_func("andreai", "moe_scatter_add").unwrap();
    unsafe { f.launch(cfg, (expert_out, indices, weights, combined, n_routed, dim)) }.unwrap();
}

pub fn gpu_causal_mask_window(
    ctx: &Arc<MetalContext>,
    scores: &GpuBuffer,
    batch_heads: u32,
    seq_q: u32,
    seq_k: u32,
    offset: u32,
    window: u32,
) {
    let cfg = launch_cfg_3d(batch_heads, seq_q, seq_k.div_ceil(256), 256);
    let f = ctx
        .device
        .get_func("andreai", "causal_mask_window")
        .unwrap();
    unsafe { f.launch(cfg, (scores, batch_heads, seq_q, seq_k, offset, window)) }.unwrap();
}

pub fn gpu_causal_doc_mask(
    ctx: &Arc<MetalContext>,
    scores: &GpuBuffer,
    seg_ids: &CudaSlice<u32>,
    batch_heads: u32,
    seq: u32,
    n_heads: u32,
) {
    let cfg = launch_cfg_3d(batch_heads, seq, seq.div_ceil(256), 256);
    let f = ctx.device.get_func("andreai", "causal_doc_mask").unwrap();
    unsafe { f.launch(cfg, (scores, seg_ids, batch_heads, seq, n_heads)) }.unwrap();
}

pub fn gpu_block_mean_keys(
    ctx: &Arc<MetalContext>,
    k: &GpuBuffer,
    out: &GpuBuffer,
    d: BlockMeanDims,
) {
    let BlockMeanDims {
        bh,
        seq,
        hd,
        nb,
        block_size,
    } = d;
    let total = bh * nb * hd;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx.device.get_func("andreai", "block_mean_keys").unwrap();
    unsafe { f.launch(cfg, (k, out, bh, seq, hd, nb, block_size)) }.unwrap();
}

pub fn gpu_block_sparse_mask(
    ctx: &Arc<MetalContext>,
    scores: &GpuBuffer,
    block_scores: &GpuBuffer,
    d: BlockSparseDims,
) {
    let BlockSparseDims {
        bh,
        seq,
        nb,
        block_size,
        top_k,
    } = d;
    let total = bh * seq * seq;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx
        .device
        .get_func("andreai", "block_sparse_topk_mask")
        .unwrap();
    unsafe { f.launch(cfg, (scores, block_scores, bh, seq, nb, block_size, top_k)) }.unwrap();
}

pub fn gpu_gather_blocks(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    sel: &CudaSlice<u32>,
    out: &GpuBuffer,
    d: GatherDims,
) {
    let GatherDims {
        bh,
        nb,
        seq,
        hd,
        block,
        k_sel,
    } = d;
    let total = bh * nb * k_sel * block * hd;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx.device.get_func("andreai", "gather_blocks").unwrap();
    unsafe { f.launch(cfg, (src, sel, out, bh, nb, seq, hd, block, k_sel)) }.unwrap();
}

pub fn gpu_gather_blocks_backward(
    ctx: &Arc<MetalContext>,
    d_out: &GpuBuffer,
    sel: &CudaSlice<u32>,
    d_src: &GpuBuffer,
    d: GatherDims,
) {
    let GatherDims {
        bh,
        nb,
        seq,
        hd,
        block,
        k_sel,
    } = d;
    gpu_fill(ctx, d_src, bh * seq * hd, 0.0); // scatter-add accumulator must start zeroed (matches metal)
    let total = bh * nb * k_sel * block * hd;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx
        .device
        .get_func("andreai", "gather_blocks_backward")
        .unwrap();
    unsafe { f.launch(cfg, (d_out, sel, d_src, bh, nb, seq, hd, block, k_sel)) }.unwrap();
}

pub fn gpu_gather_causal_mask(
    ctx: &Arc<MetalContext>,
    scores: &GpuBuffer,
    sel: &CudaSlice<u32>,
    bh_nb: u32,
    nb: u32,
    block: u32,
    k_sel: u32,
) {
    let total = bh_nb * block * k_sel * block;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx
        .device
        .get_func("andreai", "gather_causal_mask")
        .unwrap();
    unsafe { f.launch(cfg, (scores, sel, bh_nb, nb, block, k_sel)) }.unwrap();
}

pub fn gpu_l2_norm(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32) -> f32 {
    let out = ctx.alloc_buffer(8); // l2_norm_check writes [sum_sq, nan_flag]
    gpu_l2_norm_into(ctx, data, size, &out);
    MetalContext::read_buffer(&out, 2)[0]
}

pub fn gpu_l2_norm_into(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, output: &GpuBuffer) {
    let tpg = next_power_of_2_clamped(size as u64) as u32;
    let cfg = launch_cfg(tpg, 1);
    let f = ctx.device.get_func("andreai", "l2_norm_check").unwrap();
    unsafe { f.launch(cfg, (data, output, size)) }.unwrap();
}

pub fn gpu_silu_backward(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_input: &GpuBuffer,
    size: u32,
) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "silu_backward").unwrap();
    unsafe { f.launch(cfg, (input, grad_output, grad_input, size)) }.unwrap();
}

pub fn gpu_silu_gate_backward(
    ctx: &Arc<MetalContext>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_gate: &GpuBuffer,
    grad_up: &GpuBuffer,
    size: u32,
) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx
        .device
        .get_func("andreai", "silu_gate_backward")
        .unwrap();
    unsafe { f.launch(cfg, (gate, up, grad_output, grad_gate, grad_up, size)) }.unwrap();
}

pub fn gpu_rms_norm_backward(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_input: &GpuBuffer,
    grad_weight: &GpuBuffer,
    params: &RmsNormBackwardParams,
) {
    let cfg = launch_cfg_2d(
        params.rows,
        1,
        next_power_of_2_clamped(params.cols as u64) as u32,
        1,
    );
    let f = ctx.device.get_func("andreai", "rms_norm_backward").unwrap();
    let clamp_on: u32 = if rmsnorm_clamp_enabled() { 1 } else { 0 };
    unsafe {
        f.launch(
            cfg,
            (
                input,
                weight,
                grad_output,
                grad_input,
                grad_weight,
                params.rows,
                params.cols,
                params.eps,
                clamp_on,
            ),
        )
    }
    .unwrap();
}

pub fn gpu_softmax_backward(
    ctx: &Arc<MetalContext>,
    softmax_out: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_input: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    let cfg = launch_cfg_2d(rows, 1, next_power_of_2_clamped(cols as u64) as u32, 1);
    let f = ctx.device.get_func("andreai", "softmax_backward").unwrap();
    unsafe { f.launch(cfg, (softmax_out, grad_output, grad_input, rows, cols)) }.unwrap();
}

pub fn gpu_embedding_backward(
    ctx: &Arc<MetalContext>,
    tokens: &CudaSlice<u32>,
    grad_output: &GpuBuffer,
    grad_embeddings: &GpuBuffer,
    n_tokens: u32,
    dim: u32,
) {
    // The optimizer and gradient clipping consume the whole embedding gradient
    // tensor, so pooled/stale rows must be zero even when no token touched them.
    gpu_fill(ctx, grad_embeddings, grad_embeddings.len() as u32, 0.0);
    let cfg = launch_cfg_3d(n_tokens, dim.div_ceil(256), 1, 256);
    let f = ctx
        .device
        .get_func("andreai", "embedding_backward")
        .unwrap();
    unsafe { f.launch(cfg, (tokens, grad_output, grad_embeddings, n_tokens, dim)) }.unwrap();
}

pub fn gpu_transpose_2d(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    let total = rows * cols;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx.device.get_func("andreai", "transpose_2d").unwrap();
    unsafe { f.launch(cfg, (input, output, rows, cols)) }.unwrap();
}

pub fn gpu_transpose_perm_backward(
    ctx: &Arc<MetalContext>,
    grad_in: &GpuBuffer,
    grad_out: &GpuBuffer,
    batch: u32,
    seq: u32,
    n_heads: u32,
    head_dim: u32,
) {
    let total = batch * seq * n_heads * head_dim;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx
        .device
        .get_func("andreai", "transpose_perm_backward")
        .unwrap();
    // Contract (matches Metal): 1st buffer param = source [bh,seq,hd] (read), 2nd = dest [b,s,d] (write).
    // The kernel reads arg0, writes arg1 — so pass (grad_in, grad_out) in that order.
    unsafe { f.launch(cfg, (grad_in, grad_out, batch, seq, n_heads, head_dim)) }.unwrap();
}

pub fn gpu_transpose_perm_forward(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    batch: u32,
    seq: u32,
    n_heads: u32,
    head_dim: u32,
) {
    let total = batch * seq * n_heads * head_dim;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx
        .device
        .get_func("andreai", "transpose_perm_forward")
        .unwrap();
    unsafe { f.launch(cfg, (input, output, batch, seq, n_heads, head_dim)) }.unwrap();
}

pub fn gpu_gradient_mask(
    ctx: &Arc<MetalContext>,
    grad: &GpuBuffer,
    mask: &CudaSlice<u32>,
    positions: u32,
    vocab_size: u32,
) {
    let total = positions * vocab_size;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx.device.get_func("andreai", "gradient_mask").unwrap();
    unsafe { f.launch(cfg, (grad, mask, total, vocab_size)) }.unwrap();
}

pub fn gpu_strided_batch_copy(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    d: StridedCopyDims,
) {
    let StridedCopyDims {
        bh,
        src_seq_len,
        dst_stride,
        dst_offset,
        dim,
    } = d;
    let copy_len = src_seq_len * dim;
    let cfg = launch_cfg_2d(bh, copy_len.div_ceil(256), 256, 1);
    let f = ctx
        .device
        .get_func("andreai", "strided_batch_copy")
        .unwrap();
    // src/dst row = head; per-head linear copy of (src_seq_len*dim) elems, dim-scaled strides/offset.
    unsafe {
        f.launch(
            cfg,
            (
                src,
                dst,
                0u32,
                dst_offset * dim,
                copy_len,
                src_seq_len * dim,
                dst_stride * dim,
                bh,
            ),
        )
    }
    .unwrap();
}

pub fn gpu_compact_strided_copy(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    bh: u32,
    seq_len: u32,
    src_stride: u32,
    dim: u32,
) {
    let copy_len = seq_len * dim;
    let cfg = launch_cfg_2d(bh, copy_len.div_ceil(256), 256, 1);
    let f = ctx
        .device
        .get_func("andreai", "compact_strided_copy")
        .unwrap();
    // gather strided src rows into compact dst; row = head, dim-scaled strides.
    unsafe {
        f.launch(
            cfg,
            (src, dst, src_stride * dim, seq_len * dim, copy_len, bh),
        )
    }
    .unwrap();
}

pub fn gpu_argmax(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32) -> u32 {
    let result = ctx.buffer_from_u32_slice(&[0u32]);
    let threads = next_power_of_2_clamped(size as u64) as u32;
    let cfg = launch_cfg(threads, 1); // single block, grid-stride over size
    let f = ctx.device.get_func("andreai", "argmax").unwrap();
    unsafe { f.launch(cfg, (data, result.as_ref(), size)) }.unwrap();
    MetalContext::read_buffer_u32(result.as_ref(), 1)[0]
}

pub fn gpu_temperature_scale(
    ctx: &Arc<MetalContext>,
    data: &GpuBuffer,
    offset: u32,
    count: u32,
    temperature: f32,
) {
    let inv_temperature = 1.0 / temperature;
    let cfg = launch_cfg(256, count.div_ceil(256));
    let f = ctx.device.get_func("andreai", "temperature_scale").unwrap();
    unsafe { f.launch(cfg, (data, offset, count, inv_temperature)) }.unwrap();
}

pub fn gpu_kl_divergence(
    ctx: &Arc<MetalContext>,
    teacher_logits: &GpuBuffer,
    student_logits: &GpuBuffer,
    losses: &GpuBuffer,
    grad_student: &GpuBuffer,
    d: KlDims,
) {
    let cfg = launch_cfg_2d(
        d.batch_size,
        1,
        next_power_of_2_clamped(d.vocab_size as u64) as u32,
        1,
    );
    let f = ctx.device.get_func("andreai", "kl_divergence").unwrap();
    unsafe {
        f.launch(
            cfg,
            (
                teacher_logits,
                student_logits,
                losses,
                grad_student,
                d.batch_size,
                d.vocab_size,
                d.temperature,
            ),
        )
    }
    .unwrap();
}

pub fn gpu_scaled_causal_softmax(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    d: SoftmaxDims,
) {
    // SoftmaxDims.total_rows ALREADY = batch_heads * seq_q (see Tensor::scaled_causal_softmax).
    // The score buffer is [total_rows, seq_k]; causal_mask wants the real batch_heads = total_rows/seq_q.
    let SoftmaxDims {
        total_rows,
        seq_q,
        seq_k,
        scale,
        kv_offset,
    } = d;
    let n = total_rows * seq_k;
    gpu_copy(ctx, input, output, n);
    gpu_scale(ctx, output, n, scale);
    gpu_causal_mask(ctx, output, total_rows / seq_q, seq_q, seq_k, kv_offset);
    gpu_softmax(ctx, output, output, total_rows, seq_k);
}

pub fn gpu_scaled_causal_softmax_window(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    d: SoftmaxDims,
    window: u32,
) {
    let SoftmaxDims {
        total_rows,
        seq_q,
        seq_k,
        scale,
        kv_offset,
    } = d;
    let n = total_rows * seq_k;
    gpu_copy(ctx, input, output, n);
    gpu_scale(ctx, output, n, scale);
    gpu_causal_mask_window(
        ctx,
        output,
        total_rows / seq_q,
        seq_q,
        seq_k,
        kv_offset,
        window,
    );
    gpu_softmax(ctx, output, output, total_rows, seq_k);
}

pub fn gpu_mega_ffn(
    ctx: &Arc<MetalContext>,
    x: &GpuBuffer,
    norm_w: &GpuBuffer,
    w: FfnWeights,
    output: &GpuBuffer,
    d: MegaFfnDims,
) {
    // CUDA composes the SwiGLU FFN from primitives (no monolithic fused kernel): output =
    // x + (silu(rms_norm(x) @ w1) * (rms_norm(x) @ w3)) @ w2 — bit-identical to the standard path.
    let FfnWeights { w1, w2, w3 } = w;
    let MegaFfnDims {
        batch_tokens,
        d_model,
        d_ff,
        eps,
    } = d;
    let (bt, dm, df) = (batch_tokens, d_model, d_ff);
    let normed = ctx.alloc_buffer((bt * dm * 4) as usize);
    gpu_rms_norm(ctx, x, norm_w, &normed, bt, dm, eps);
    let gate = ctx.alloc_buffer((bt * df * 4) as usize);
    gpu_matmul(ctx, &normed, w1, &gate, bt, df, dm);
    let up = ctx.alloc_buffer((bt * df * 4) as usize);
    gpu_matmul(ctx, &normed, w3, &up, bt, df, dm);
    let hidden = ctx.alloc_buffer((bt * df * 4) as usize);
    gpu_silu_gate(ctx, &gate, &up, &hidden, bt * df);
    let down = ctx.alloc_buffer((bt * dm * 4) as usize);
    gpu_matmul(ctx, &hidden, w2, &down, bt, dm, df);
    gpu_add(ctx, x, &down, output, bt * dm);
}

pub fn gpu_transpose_rope(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    d: TrRopeDims,
) {
    gpu_transpose_perm_forward(ctx, input, output, d.batch, d.seq, d.n_heads, d.head_dim);
    gpu_rope(
        ctx,
        output,
        d.batch * d.n_heads,
        d.seq,
        d.head_dim,
        d.offset,
        d.theta,
    );
}

pub fn gpu_transpose_rope_backward(
    ctx: &Arc<MetalContext>,
    grad_out: &GpuBuffer,
    grad_in: &GpuBuffer,
    d: TrRopeDims,
) {
    gpu_rope_backward(
        ctx,
        grad_out,
        d.batch * d.n_heads,
        d.seq,
        d.head_dim,
        d.offset,
        d.theta,
    );
    // grad_out (rope'd [bh,seq,hd]) is the source; grad_in ([b,seq,d]) is the dest.
    gpu_transpose_perm_backward(
        ctx, grad_out, grad_in, d.batch, d.seq, d.n_heads, d.head_dim,
    );
}

pub fn gpu_compute_inv_rms(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    inv_rms: &GpuBuffer,
    rows: u32,
    cols: u32,
    eps: f32,
) {
    let tpg = next_power_of_2_clamped(cols as u64) as u32;
    let cfg = launch_cfg(tpg, rows); // one block per row
    let f = ctx.device.get_func("andreai", "compute_inv_rms").unwrap();
    unsafe { f.launch(cfg, (input, inv_rms, rows, cols, eps)) }.unwrap();
}

pub fn gpu_fused_norm_matmul(
    ctx: &Arc<MetalContext>,
    a: &GpuBuffer,
    norm_weight: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    d: NormMatmulDims,
) {
    // C = rms_norm(A[m,k], norm_weight) @ B[k,n]. Composed from primitives (CUDA has no fused kernel).
    let NormMatmulDims { m, n, k, eps } = d;
    let normed = ctx.alloc_buffer((m * k * 4) as usize);
    gpu_rms_norm(ctx, a, norm_weight, &normed, m, k, eps);
    gpu_matmul(ctx, &normed, b, c, m, n, k);
}

pub fn gpu_axpy(ctx: &Arc<MetalContext>, y: &GpuBuffer, x: &GpuBuffer, size: u32, alpha: f32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "axpy").unwrap();
    unsafe { f.launch(cfg, (y, x, size, alpha)) }.unwrap();
}

pub fn gpu_relu(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "relu").unwrap();
    unsafe { f.launch(cfg, (input, output, size)) }.unwrap();
}

pub fn gpu_broadcast_rows(
    ctx: &Arc<MetalContext>,
    vec: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    let cfg = launch_cfg(256, (rows * cols).div_ceil(256));
    let f = ctx.device.get_func("andreai", "broadcast_rows").unwrap();
    unsafe { f.launch(cfg, (vec, out, rows, cols)) }.unwrap();
}

pub fn gpu_exp(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "exp_kernel").unwrap();
    unsafe { f.launch(cfg, (input, output, size)) }.unwrap();
}

pub fn gpu_relu_backward(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    grad_output: &GpuBuffer,
    grad_input: &GpuBuffer,
    size: u32,
) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "relu_backward").unwrap();
    unsafe { f.launch(cfg, (input, grad_output, grad_input, size)) }.unwrap();
}

pub fn gpu_ema_update(
    ctx: &Arc<MetalContext>,
    ema: &GpuBuffer,
    src: &GpuBuffer,
    size: u32,
    decay: f32,
) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "ema_update").unwrap();
    unsafe { f.launch(cfg, (ema, src, size, decay)) }.unwrap();
}

pub fn gpu_logsumexp(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    rows: u32,
    cols: u32,
) {
    let tpg = next_power_of_2_clamped(cols as u64) as u32;
    let cfg = launch_cfg(tpg, rows); // one block per row
    let f = ctx.device.get_func("andreai", "logsumexp").unwrap();
    unsafe { f.launch(cfg, (input, output, rows, cols)) }.unwrap();
}

pub fn gpu_scale_copy(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    size: u32,
    scale: f32,
) {
    gpu_copy(ctx, src, dst, size);
    gpu_scale(ctx, dst, size, scale);
}

pub fn gpu_muon_frob_normalize(ctx: &Arc<MetalContext>, m: &GpuBuffer, x: &GpuBuffer, size: u32) {
    let threads = next_power_of_2_clamped(size as u64) as u32; // single block, ≤256 (power of 2)
    let cfg = launch_cfg(threads, 1);
    let f = ctx
        .device
        .get_func("andreai", "muon_frob_normalize")
        .unwrap();
    unsafe { f.launch(cfg, (m, x, size)) }.unwrap();
}

pub fn gpu_inv_sqrt_bc(
    ctx: &Arc<MetalContext>,
    v: &GpuBuffer,
    out: &GpuBuffer,
    size: u32,
    bias_correction: f32,
    eps: f32,
) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "inv_sqrt_bc").unwrap();
    unsafe { f.launch(cfg, (v, out, size, bias_correction, eps)) }.unwrap();
}

pub fn gpu_cautious_mask(
    ctx: &Arc<MetalContext>,
    update: &GpuBuffer,
    grad: &GpuBuffer,
    keep: &GpuBuffer,
    size: u32,
) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "cautious_mask").unwrap();
    unsafe { f.launch(cfg, (update, grad, keep, size)) }.unwrap();
}

pub fn gpu_cautious_scale(ctx: &Arc<MetalContext>, x: &GpuBuffer, kept_sum: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "cautious_scale").unwrap();
    unsafe { f.launch(cfg, (x, kept_sum, size)) }.unwrap();
}

pub fn gpu_concat_cols(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    rows: u32,
    src_cols: u32,
    dst_cols: u32,
    col_offset: u32,
) {
    let cfg = launch_cfg(256, (rows * src_cols).div_ceil(256));
    let f = ctx.device.get_func("andreai", "concat_cols").unwrap();
    unsafe { f.launch(cfg, (src, dst, rows, src_cols, dst_cols, col_offset)) }.unwrap();
}

pub fn gpu_slice_cols(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    rows: u32,
    src_cols: u32,
    dst_cols: u32,
    col_offset: u32,
) {
    let cfg = launch_cfg(256, (rows * dst_cols).div_ceil(256));
    let f = ctx.device.get_func("andreai", "slice_cols").unwrap();
    unsafe { f.launch(cfg, (src, dst, rows, src_cols, dst_cols, col_offset)) }.unwrap();
}

pub fn gpu_repeat_kv(
    ctx: &Arc<MetalContext>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    n_kv_total: u32,
    group_size: u32,
    seq_len: u32,
    head_dim: u32,
) {
    let total = n_kv_total * group_size * seq_len * head_dim;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx.device.get_func("andreai", "repeat_kv").unwrap();
    unsafe {
        f.launch(
            cfg,
            (input, output, n_kv_total, group_size, seq_len, head_dim),
        )
    }
    .unwrap();
}

pub fn gpu_repeat_kv_backward(
    ctx: &Arc<MetalContext>,
    out_grad: &GpuBuffer,
    kv_grad: &GpuBuffer,
    n_kv_total: u32,
    group_size: u32,
    seq_len: u32,
    head_dim: u32,
) {
    let total = n_kv_total * seq_len * head_dim;
    let cfg = launch_cfg(256, total.div_ceil(256));
    let f = ctx
        .device
        .get_func("andreai", "repeat_kv_backward")
        .unwrap();
    unsafe {
        f.launch(
            cfg,
            (out_grad, kv_grad, n_kv_total, group_size, seq_len, head_dim),
        )
    }
    .unwrap();
}

// ===== Matmul-path flags. On CUDA "simdgroup" has no hardware analogue, so the flag selects
// fp32-precise vs the default fp16-tiled matmul (the simdgroup wrappers below alias accordingly). =====
thread_local! {
    static SIMDGROUP_MATMUL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub fn set_simdgroup_matmul(on: bool) -> bool {
    SIMDGROUP_MATMUL.with(|c| c.replace(on))
}
pub fn simdgroup_matmul_enabled() -> bool {
    SIMDGROUP_MATMUL.with(|c| c.get())
}
thread_local! {
    static BF16_MATMUL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub fn set_bf16_matmul(on: bool) -> bool {
    BF16_MATMUL.with(|c| c.replace(on))
}
pub fn bf16_matmul_enabled() -> bool {
    BF16_MATMUL.with(|c| c.get())
}
thread_local! {
    static RMSNORM_CLAMP: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
pub fn set_rmsnorm_clamp(on: bool) -> bool {
    RMSNORM_CLAMP.with(|c| c.replace(on))
}
fn rmsnorm_clamp_enabled() -> bool {
    RMSNORM_CLAMP.with(|c| c.get())
}
pub const ADAM8_BLOCK: usize = 256;
