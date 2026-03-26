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
