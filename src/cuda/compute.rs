//! CUDA compute dispatch functions — same API as metal/compute.rs.
//! Each function launches a CUDA kernel with the appropriate grid/block dimensions.

use super::MetalContext; // aliased CudaContext
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
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

pub fn gpu_matmul(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    let tile = 32u32;
    let cfg = launch_cfg_3d(n.div_ceil(tile), m.div_ceil(tile), 1, 64);
    let f = ctx.device.get_func("andreai", "matmul_tiled").unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k)) }.unwrap();
}

pub fn gpu_matmul_trans_b(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    let tile = 32u32;
    let cfg = launch_cfg_3d(n.div_ceil(tile), m.div_ceil(tile), 1, 64);
    let f = ctx.device.get_func("andreai", "matmul_tiled_trans_b").unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, n, k)) }.unwrap();
}

pub fn gpu_matmul_trans_a(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, k: u32, n: u32) {
    let tile = 32u32;
    let cfg = launch_cfg_3d(n.div_ceil(tile), k.div_ceil(tile), 1, 64);
    let f = ctx.device.get_func("andreai", "matmul_trans_a_tiled").unwrap();
    unsafe { f.launch(cfg, (a, b, c, m, k, n)) }.unwrap();
}

// ===== Softmax / Norm =====

fn next_power_of_2_clamped(n: u64) -> u64 {
    let clamped = n.min(256).max(1) as u32;
    clamped.next_power_of_two().min(256) as u64
}

pub fn gpu_softmax(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32) {
    let tpg = next_power_of_2_clamped(cols as u64) as u32;
    let cfg = launch_cfg_2d(rows, 1, tpg, 1);
    let f = ctx.device.get_func("andreai", "softmax").unwrap();
    unsafe { f.launch(cfg, (input, output, rows, cols)) }.unwrap();
}

pub fn gpu_rms_norm(ctx: &Arc<MetalContext>, input: &GpuBuffer, weight: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32, eps: f32) {
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

pub fn gpu_silu_gate(ctx: &Arc<MetalContext>, gate: &GpuBuffer, up: &GpuBuffer, output: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "silu_gate").unwrap();
    unsafe { f.launch(cfg, (gate, up, output, size)) }.unwrap();
}

// ===== RoPE =====

pub fn gpu_rope(ctx: &Arc<MetalContext>, data: &GpuBuffer, total_rows: u32, seq_len: u32, head_dim: u32, offset: u32, theta: f32) {
    let pairs = head_dim / 2;
    let cfg = launch_cfg_3d(total_rows, seq_len, 1, pairs);
    let f = ctx.device.get_func("andreai", "rope").unwrap();
    unsafe { f.launch(cfg, (data, total_rows, seq_len, head_dim, offset, theta)) }.unwrap();
}

// ===== Loss =====

pub fn gpu_cross_entropy(ctx: &Arc<MetalContext>, logits: &GpuBuffer, targets: &CudaSlice<u32>, losses: &GpuBuffer, grad: &GpuBuffer, batch: u32, vocab: u32) {
    let tpg = next_power_of_2_clamped(vocab as u64) as u32;
    let cfg = launch_cfg_2d(batch, 1, tpg, 1);
    let f = ctx.device.get_func("andreai", "cross_entropy").unwrap();
    unsafe { f.launch(cfg, (logits, targets, losses, grad, batch, vocab)) }.unwrap();
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
}

pub fn gpu_adamw_update(ctx: &Arc<MetalContext>, param: &GpuBuffer, grad: &GpuBuffer, m: &GpuBuffer, v: &GpuBuffer, size: u32, hp: &AdamWHyperparams) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "adamw_update").unwrap();
    unsafe { f.launch(cfg, (param, grad, m, v, size, hp.lr, hp.beta1, hp.beta2, hp.eps, hp.weight_decay, hp.step as i32)) }.unwrap();
}

// ===== Embedding =====

pub fn gpu_embedding_lookup(ctx: &Arc<MetalContext>, table: &GpuBuffer, tokens: &CudaSlice<u32>, output: &GpuBuffer, seq_len: u32, dim: u32) {
    let cfg = launch_cfg_2d(seq_len, 1, dim.min(1024), 1);
    let f = ctx.device.get_func("andreai", "embedding_lookup").unwrap();
    unsafe { f.launch(cfg, (table, tokens, output, seq_len, dim)) }.unwrap();
}

// ===== Cast =====

pub fn gpu_cast_f32_to_f16(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32) {
    let cfg = launch_cfg(256, size.div_ceil(256));
    let f = ctx.device.get_func("andreai", "cast_f32_to_f16").unwrap();
    unsafe { f.launch(cfg, (input, output, size)) }.unwrap();
}

// ===== Norm check =====

pub fn gpu_l2_norm_check_into(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, output: &GpuBuffer) {
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

pub fn gpu_buffer_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, src_offset: u32, dst_offset: u32, count: u32) {
    let cfg = launch_cfg(256, count.div_ceil(256));
    let f = ctx.device.get_func("andreai", "buffer_copy").unwrap();
    unsafe { f.launch(cfg, (src, dst, src_offset, dst_offset, count)) }.unwrap();
}

// Causal mask
pub fn gpu_causal_mask(ctx: &Arc<MetalContext>, scores: &GpuBuffer, batch_heads: u32, seq_q: u32, seq_k: u32, offset: u32) {
    let cfg = launch_cfg_3d(batch_heads, seq_q, 1, seq_k.min(1024));
    let f = ctx.device.get_func("andreai", "causal_mask").unwrap();
    unsafe { f.launch(cfg, (scores, batch_heads, seq_q, seq_k, offset)) }.unwrap();
}

// ===== AUTO-GENERATED CUDA STUBS (mirror metal/compute.rs; unimplemented until ported) =====
#[derive(Clone, Copy)]
pub struct BatchedDims { pub batch: u32, pub m: u32, pub n: u32, pub k: u32 }

#[derive(Clone, Copy)]
pub struct RopeDims { pub total_rows: u32, pub seq_len: u32, pub head_dim: u32, pub offset: u32, pub theta: f32 }

#[derive(Clone, Copy)]
pub struct TrRopeDims { pub batch: u32, pub seq: u32, pub n_heads: u32, pub head_dim: u32, pub offset: u32, pub theta: f32 }

#[derive(Clone, Copy)]
pub struct FlashDims { pub batch_heads: u32, pub seq_q: u32, pub seq_k: u32, pub head_dim: u32, pub kv_offset: u32 }

#[derive(Clone, Copy)]
pub struct SoftmaxDims { pub total_rows: u32, pub seq_q: u32, pub seq_k: u32, pub scale: f32, pub kv_offset: u32 }

#[derive(Clone, Copy)]
pub struct LionParams { pub lr: f32, pub beta1: f32, pub beta2: f32, pub weight_decay: f32 }

#[derive(Clone, Copy)]
pub struct SophiaParams { pub lr: f32, pub beta1: f32, pub beta2: f32, pub eps: f32, pub rho: f32, pub weight_decay: f32 }

#[derive(Clone, Copy)]
pub struct RmsResDims { pub rows: u32, pub cols: u32, pub eps: f32 }

#[derive(Clone, Copy)]
pub struct StridedCopyDims { pub bh: u32, pub src_seq_len: u32, pub dst_stride: u32, pub dst_offset: u32, pub dim: u32 }

#[derive(Clone, Copy)]
pub struct KlDims { pub batch_size: u32, pub vocab_size: u32, pub temperature: f32 }

#[derive(Clone, Copy)]
pub struct MegaFfnDims { pub batch_tokens: u32, pub d_model: u32, pub d_ff: u32, pub eps: f32 }

#[derive(Clone, Copy)]
pub struct FfnWeights<'a> { pub w1: &'a GpuBuffer, pub w2: &'a GpuBuffer, pub w3: &'a GpuBuffer }

#[derive(Clone, Copy)]
pub struct NormMatmulDims { pub m: u32, pub n: u32, pub k: u32, pub eps: f32 }

#[derive(Clone, Copy)]
pub struct FlashBwdBufs<'a> {
    pub q: &'a GpuBuffer, pub k: &'a GpuBuffer, pub v: &'a GpuBuffer,
    pub output: &'a GpuBuffer, pub d_out: &'a GpuBuffer, pub d_buf: &'a GpuBuffer,
    pub dq: &'a GpuBuffer, pub dk: &'a GpuBuffer, pub dv: &'a GpuBuffer,
}

pub struct Adam8Buffers<'a> {
    pub m_q: &'a GpuBuffer,
    pub v_q: &'a GpuBuffer,
    pub m_scale: &'a GpuBuffer,
    pub v_scale: &'a GpuBuffer,
}

#[derive(Clone, Copy)]
pub struct BlockMeanDims { pub bh: u32, pub seq: u32, pub hd: u32, pub nb: u32, pub block_size: u32 }

#[derive(Clone, Copy)]
pub struct BlockSparseDims { pub bh: u32, pub seq: u32, pub nb: u32, pub block_size: u32, pub top_k: u32 }

#[derive(Clone, Copy)]
pub struct GatherDims { pub bh: u32, pub nb: u32, pub seq: u32, pub hd: u32, pub block: u32, pub k_sel: u32 }

pub struct RmsNormBackwardParams {
    pub rows: u32,
    pub cols: u32,
    pub eps: f32,
}

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_matmul_fp32(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32)  { unimplemented!("cuda backend: gpu_matmul_fp32 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_matmul_bf16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32)  { unimplemented!("cuda backend: gpu_matmul_bf16 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_matmul_simdgroup(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32)  { unimplemented!("cuda backend: gpu_matmul_simdgroup not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_matmul_simdgroup_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32)  { unimplemented!("cuda backend: gpu_matmul_simdgroup_f16 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_matmul_trans_b_simdgroup(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32)  { unimplemented!("cuda backend: gpu_matmul_trans_b_simdgroup not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_matmul_trans_a_simdgroup(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, k: u32, n: u32)  { unimplemented!("cuda backend: gpu_matmul_trans_a_simdgroup not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_cast_f16_to_f32(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32)  { unimplemented!("cuda backend: gpu_cast_f16_to_f32 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_matmul_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32)  { unimplemented!("cuda backend: gpu_matmul_f16 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_matmul_trans_b_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32)  { unimplemented!("cuda backend: gpu_matmul_trans_b_f16 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_matmul_trans_a_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, k: u32, n: u32)  { unimplemented!("cuda backend: gpu_matmul_trans_a_f16 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims)  { unimplemented!("cuda backend: gpu_batched_matmul_f16 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_trans_b_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims)  { unimplemented!("cuda backend: gpu_batched_matmul_trans_b_f16 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_trans_a_f16(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims)  { unimplemented!("cuda backend: gpu_batched_matmul_trans_a_f16 not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims)  { unimplemented!("cuda backend: gpu_batched_matmul not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_simdgroup(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims)  { unimplemented!("cuda backend: gpu_batched_matmul_simdgroup not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_trans_b_simdgroup(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims)  { unimplemented!("cuda backend: gpu_batched_matmul_trans_b_simdgroup not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_trans_a_simdgroup(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims)  { unimplemented!("cuda backend: gpu_batched_matmul_trans_a_simdgroup not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_trans_b(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims)  { unimplemented!("cuda backend: gpu_batched_matmul_trans_b not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_gqa_trans_b( ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims, group_size: u32, )  { unimplemented!("cuda backend: gpu_batched_matmul_gqa_trans_b not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_gqa( ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims, group_size: u32, )  { unimplemented!("cuda backend: gpu_batched_matmul_gqa not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_batched_matmul_trans_a(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: BatchedDims)  { unimplemented!("cuda backend: gpu_batched_matmul_trans_a not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_rms_norm_residual( ctx: &Arc<MetalContext>, input: &GpuBuffer, residual: &GpuBuffer, weight: &GpuBuffer, output: &GpuBuffer, sum_out: &GpuBuffer, d: RmsResDims, )  { unimplemented!("cuda backend: gpu_rms_norm_residual not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_rope_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, d: RopeDims)  { unimplemented!("cuda backend: gpu_rope_copy not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_rope_backward( ctx: &Arc<MetalContext>, data: &GpuBuffer, total_rows: u32, seq_len: u32, head_dim: u32, offset: u32, theta: f32, )  { unimplemented!("cuda backend: gpu_rope_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_rope_backward_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, d: RopeDims)  { unimplemented!("cuda backend: gpu_rope_backward_copy not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_silu(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32)  { unimplemented!("cuda backend: gpu_silu not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_adamw_8bit_update( ctx: &Arc<MetalContext>, param: &GpuBuffer, grad: &GpuBuffer, state: &Adam8Buffers, size: u32, hp: &AdamWHyperparams, )  { unimplemented!("cuda backend: gpu_adamw_8bit_update not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_flash_attention_forward( ctx: &Arc<MetalContext>, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer, o: &GpuBuffer, d: FlashDims, )  { unimplemented!("cuda backend: gpu_flash_attention_forward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_flash_attn_precompute_d( ctx: &Arc<MetalContext>, d_out: &GpuBuffer, output: &GpuBuffer, d_buf: &GpuBuffer, total_rows: u32, head_dim: u32, )  { unimplemented!("cuda backend: gpu_flash_attn_precompute_d not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_flash_attention_backward(ctx: &Arc<MetalContext>, b: FlashBwdBufs, d: FlashDims)  { unimplemented!("cuda backend: gpu_flash_attention_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_ternary_matmul(ctx: &Arc<MetalContext>, a: &GpuBuffer, w_packed: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32)  { unimplemented!("cuda backend: gpu_ternary_matmul not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_ternary_absmean(ctx: &Arc<MetalContext>, weights: &GpuBuffer, absmean: &GpuBuffer, rows: u32, cols: u32)  { unimplemented!("cuda backend: gpu_ternary_absmean not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_ternary_pack(ctx: &Arc<MetalContext>, weights: &GpuBuffer, absmean: &GpuBuffer, packed: &GpuBuffer, rows: u32, cols: u32)  { unimplemented!("cuda backend: gpu_ternary_pack not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_lion_update(ctx: &Arc<MetalContext>, param: &GpuBuffer, grad: &GpuBuffer, m: &GpuBuffer, size: u32, p: LionParams)  { unimplemented!("cuda backend: gpu_lion_update not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_sophia_update(ctx: &Arc<MetalContext>, param: &GpuBuffer, grad: &GpuBuffer, m: &GpuBuffer, h: &GpuBuffer, size: u32, p: SophiaParams)  { unimplemented!("cuda backend: gpu_sophia_update not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_scale_rows(ctx: &Arc<MetalContext>, input: &GpuBuffer, scales: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32)  { unimplemented!("cuda backend: gpu_scale_rows not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_row_dot_reduce(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32)  { unimplemented!("cuda backend: gpu_row_dot_reduce not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_moe_gather(ctx: &Arc<MetalContext>, input: &GpuBuffer, indices: &GpuBuffer, gathered: &GpuBuffer, n_routed: u32, dim: u32)  { unimplemented!("cuda backend: gpu_moe_gather not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_moe_scatter_add(ctx: &Arc<MetalContext>, expert_out: &GpuBuffer, indices: &GpuBuffer, weights: &GpuBuffer, combined: &GpuBuffer, n_routed: u32, dim: u32)  { unimplemented!("cuda backend: gpu_moe_scatter_add not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_causal_mask_window( ctx: &Arc<MetalContext>, scores: &GpuBuffer, batch_heads: u32, seq_q: u32, seq_k: u32, offset: u32, window: u32, )  { unimplemented!("cuda backend: gpu_causal_mask_window not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_causal_doc_mask( ctx: &Arc<MetalContext>, scores: &GpuBuffer, seg_ids: &GpuBuffer, batch_heads: u32, seq: u32, n_heads: u32, )  { unimplemented!("cuda backend: gpu_causal_doc_mask not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_block_mean_keys(ctx: &Arc<MetalContext>, k: &GpuBuffer, out: &GpuBuffer, d: BlockMeanDims)  { unimplemented!("cuda backend: gpu_block_mean_keys not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_block_sparse_mask(ctx: &Arc<MetalContext>, scores: &GpuBuffer, block_scores: &GpuBuffer, d: BlockSparseDims)  { unimplemented!("cuda backend: gpu_block_sparse_mask not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_gather_blocks(ctx: &Arc<MetalContext>, src: &GpuBuffer, sel: &GpuBuffer, out: &GpuBuffer, d: GatherDims)  { unimplemented!("cuda backend: gpu_gather_blocks not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_gather_blocks_backward(ctx: &Arc<MetalContext>, d_out: &GpuBuffer, sel: &GpuBuffer, d_src: &GpuBuffer, d: GatherDims)  { unimplemented!("cuda backend: gpu_gather_blocks_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_gather_causal_mask(ctx: &Arc<MetalContext>, scores: &GpuBuffer, sel: &GpuBuffer, bh_nb: u32, nb: u32, block: u32, k_sel: u32)  { unimplemented!("cuda backend: gpu_gather_causal_mask not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_l2_norm(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32) -> f32 { unimplemented!("cuda backend: gpu_l2_norm not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_l2_norm_into(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, output: &GpuBuffer)  { unimplemented!("cuda backend: gpu_l2_norm_into not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_silu_backward( ctx: &Arc<MetalContext>, input: &GpuBuffer, grad_output: &GpuBuffer, grad_input: &GpuBuffer, size: u32, )  { unimplemented!("cuda backend: gpu_silu_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_silu_gate_backward( ctx: &Arc<MetalContext>, gate: &GpuBuffer, up: &GpuBuffer, grad_output: &GpuBuffer, grad_gate: &GpuBuffer, grad_up: &GpuBuffer, size: u32, )  { unimplemented!("cuda backend: gpu_silu_gate_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_rms_norm_backward( ctx: &Arc<MetalContext>, input: &GpuBuffer, weight: &GpuBuffer, grad_output: &GpuBuffer, grad_input: &GpuBuffer, grad_weight: &GpuBuffer, params: &RmsNormBackwardParams, )  { unimplemented!("cuda backend: gpu_rms_norm_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_softmax_backward( ctx: &Arc<MetalContext>, softmax_out: &GpuBuffer, grad_output: &GpuBuffer, grad_input: &GpuBuffer, rows: u32, cols: u32, )  { unimplemented!("cuda backend: gpu_softmax_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_embedding_backward( ctx: &Arc<MetalContext>, tokens: &GpuBuffer, grad_output: &GpuBuffer, grad_embeddings: &GpuBuffer, n_tokens: u32, dim: u32, )  { unimplemented!("cuda backend: gpu_embedding_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_zero_rows( ctx: &Arc<MetalContext>, tokens: &GpuBuffer, matrix: &GpuBuffer, n_tokens: u32, dim: u32, )  { unimplemented!("cuda backend: gpu_zero_rows not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_transpose_2d( ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32, )  { unimplemented!("cuda backend: gpu_transpose_2d not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_transpose_perm_backward( ctx: &Arc<MetalContext>, grad_in: &GpuBuffer, grad_out: &GpuBuffer, batch: u32, seq: u32, n_heads: u32, head_dim: u32, )  { unimplemented!("cuda backend: gpu_transpose_perm_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_transpose_perm_forward( ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, batch: u32, seq: u32, n_heads: u32, head_dim: u32, )  { unimplemented!("cuda backend: gpu_transpose_perm_forward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_gradient_mask( ctx: &Arc<MetalContext>, grad: &GpuBuffer, mask: &GpuBuffer, positions: u32, vocab_size: u32, )  { unimplemented!("cuda backend: gpu_gradient_mask not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_strided_batch_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, d: StridedCopyDims)  { unimplemented!("cuda backend: gpu_strided_batch_copy not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_compact_strided_copy( ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, bh: u32, seq_len: u32, src_stride: u32, dim: u32, )  { unimplemented!("cuda backend: gpu_compact_strided_copy not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_argmax(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32) -> u32 { unimplemented!("cuda backend: gpu_argmax not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_temperature_scale( ctx: &Arc<MetalContext>, data: &GpuBuffer, offset: u32, count: u32, temperature: f32, )  { unimplemented!("cuda backend: gpu_temperature_scale not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_kl_divergence( ctx: &Arc<MetalContext>, teacher_logits: &GpuBuffer, student_logits: &GpuBuffer, losses: &GpuBuffer, grad_student: &GpuBuffer, d: KlDims, )  { unimplemented!("cuda backend: gpu_kl_divergence not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_scaled_causal_softmax(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, d: SoftmaxDims)  { unimplemented!("cuda backend: gpu_scaled_causal_softmax not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_scaled_causal_softmax_window(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, d: SoftmaxDims, window: u32)  { unimplemented!("cuda backend: gpu_scaled_causal_softmax_window not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_mega_ffn( ctx: &Arc<MetalContext>, x: &GpuBuffer, norm_w: &GpuBuffer, w: FfnWeights, output: &GpuBuffer, d: MegaFfnDims, )  { unimplemented!("cuda backend: gpu_mega_ffn not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_transpose_rope(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, d: TrRopeDims)  { unimplemented!("cuda backend: gpu_transpose_rope not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_transpose_rope_backward(ctx: &Arc<MetalContext>, grad_out: &GpuBuffer, grad_in: &GpuBuffer, d: TrRopeDims)  { unimplemented!("cuda backend: gpu_transpose_rope_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_compute_inv_rms(ctx: &Arc<MetalContext>, input: &GpuBuffer, inv_rms: &GpuBuffer, rows: u32, cols: u32, eps: f32)  { unimplemented!("cuda backend: gpu_compute_inv_rms not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_fused_norm_matmul( ctx: &Arc<MetalContext>, a: &GpuBuffer, norm_weight: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, d: NormMatmulDims, )  { unimplemented!("cuda backend: gpu_fused_norm_matmul not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_axpy(ctx: &Arc<MetalContext>, y: &GpuBuffer, x: &GpuBuffer, size: u32, alpha: f32)  { unimplemented!("cuda backend: gpu_axpy not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_relu(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32)  { unimplemented!("cuda backend: gpu_relu not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_broadcast_rows(ctx: &Arc<MetalContext>, vec: &GpuBuffer, out: &GpuBuffer, rows: u32, cols: u32)  { unimplemented!("cuda backend: gpu_broadcast_rows not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_exp(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, size: u32)  { unimplemented!("cuda backend: gpu_exp not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_relu_backward(ctx: &Arc<MetalContext>, input: &GpuBuffer, grad_output: &GpuBuffer, grad_input: &GpuBuffer, size: u32)  { unimplemented!("cuda backend: gpu_relu_backward not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_ema_update(ctx: &Arc<MetalContext>, ema: &GpuBuffer, src: &GpuBuffer, size: u32, decay: f32)  { unimplemented!("cuda backend: gpu_ema_update not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_logsumexp(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32)  { unimplemented!("cuda backend: gpu_logsumexp not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_scale_copy(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, size: u32, scale: f32)  { unimplemented!("cuda backend: gpu_scale_copy not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_muon_frob_normalize(ctx: &Arc<MetalContext>, m: &GpuBuffer, x: &GpuBuffer, size: u32)  { unimplemented!("cuda backend: gpu_muon_frob_normalize not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_inv_sqrt_bc(ctx: &Arc<MetalContext>, v: &GpuBuffer, out: &GpuBuffer, size: u32, bias_correction: f32, eps: f32)  { unimplemented!("cuda backend: gpu_inv_sqrt_bc not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_concat_cols(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, rows: u32, src_cols: u32, dst_cols: u32, col_offset: u32)  { unimplemented!("cuda backend: gpu_concat_cols not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_slice_cols(ctx: &Arc<MetalContext>, src: &GpuBuffer, dst: &GpuBuffer, rows: u32, src_cols: u32, dst_cols: u32, col_offset: u32)  { unimplemented!("cuda backend: gpu_slice_cols not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_repeat_kv(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, n_kv_total: u32, group_size: u32, seq_len: u32, head_dim: u32)  { unimplemented!("cuda backend: gpu_repeat_kv not yet ported") }

#[allow(unused_variables, clippy::too_many_arguments)]
pub fn gpu_repeat_kv_backward(ctx: &Arc<MetalContext>, out_grad: &GpuBuffer, kv_grad: &GpuBuffer, n_kv_total: u32, group_size: u32, seq_len: u32, head_dim: u32)  { unimplemented!("cuda backend: gpu_repeat_kv_backward not yet ported") }

// ===== Matmul-path flags: Metal-only (simdgroup MMA / bf16 / rmsnorm-clamp). No-ops on CUDA. =====
pub fn set_simdgroup_matmul(_on: bool) -> bool { false }
pub fn simdgroup_matmul_enabled() -> bool { false }
pub fn set_bf16_matmul(_on: bool) -> bool { false }
pub fn bf16_matmul_enabled() -> bool { false }
pub fn set_rmsnorm_clamp(_on: bool) -> bool { false }
pub const ADAM8_BLOCK: usize = 256;
