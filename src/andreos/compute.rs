//! AndreOS compute dispatch — direct GPU kernel launches.
//!
//! Same API as metal/compute.rs and cuda/compute.rs.
//! On AndreOS, these compile to direct hardware submissions
//! instead of framework API calls.
//!
//! The kernel code itself is the same math (matmul, softmax, etc.)
//! but compiled to GPU ISA at build time via andreos-gpu toolchain,
//! not runtime-compiled via Metal/CUDA.

use super::{GpuBuffer, MetalContext};
use std::sync::Arc;

// Kernel IDs — indices into the pre-compiled kernel table.
// On AndreOS, kernels are compiled at build time and linked into the binary.
const MATMUL_TILED: u32 = 0;
const MATMUL_TILED_TRANS_B: u32 = 1;
const MATMUL_TRANS_A: u32 = 2;
const BATCHED_MATMUL: u32 = 3;
const BATCHED_MATMUL_TRANS_B: u32 = 4;
const BATCHED_MATMUL_TRANS_A: u32 = 5;
const SOFTMAX: u32 = 6;
const RMS_NORM: u32 = 7;
const ROPE: u32 = 8;
const CROSS_ENTROPY: u32 = 9;
const ADAMW: u32 = 10;
const ADD: u32 = 11;
const ADD_INPLACE: u32 = 12;
const MUL: u32 = 13;
const SCALE: u32 = 14;
const FILL: u32 = 15;
const SILU_GATE: u32 = 16;
const EMBEDDING: u32 = 17;
const REDUCE_SUM: u32 = 18;
// ... etc

/// Dispatch helpers — these are the same function signatures as Metal/CUDA.
/// The only difference is HOW they submit to the GPU.

pub fn gpu_matmul(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    let grid = ((n + 31) / 32, (m + 31) / 32, 1);
    let block = (64, 1, 1);
    ctx.dispatch_kernel_direct(
        MATMUL_TILED,
        grid, block,
        &[a.ptr as _, b.ptr as _, c.ptr as _],
        &[a.size_bytes, b.size_bytes, c.size_bytes],
    );
}

pub fn gpu_matmul_trans_b(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, m: u32, n: u32, k: u32) {
    let grid = ((n + 31) / 32, (m + 31) / 32, 1);
    let block = (64, 1, 1);
    ctx.dispatch_kernel_direct(
        MATMUL_TILED_TRANS_B,
        grid, block,
        &[a.ptr as _, b.ptr as _, c.ptr as _],
        &[a.size_bytes, b.size_bytes, c.size_bytes],
    );
}

pub fn gpu_softmax(ctx: &Arc<MetalContext>, input: &GpuBuffer, output: &GpuBuffer, rows: u32, cols: u32) {
    let tpg = (cols as u64).min(256).max(1).next_power_of_two().min(256) as u32;
    ctx.dispatch_kernel_direct(
        SOFTMAX,
        (rows, 1, 1), (tpg, 1, 1),
        &[input.ptr as _, output.ptr as _],
        &[input.size_bytes, output.size_bytes],
    );
}

pub fn gpu_add(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer, size: u32) {
    ctx.dispatch_kernel_direct(
        ADD,
        ((size + 255) / 256, 1, 1), (256, 1, 1),
        &[a.ptr as _, b.ptr as _, c.ptr as _],
        &[a.size_bytes, b.size_bytes, c.size_bytes],
    );
}

pub fn gpu_add_inplace(ctx: &Arc<MetalContext>, a: &GpuBuffer, b: &GpuBuffer, size: u32) {
    ctx.dispatch_kernel_direct(
        ADD_INPLACE,
        ((size + 255) / 256, 1, 1), (256, 1, 1),
        &[a.ptr as _, b.ptr as _],
        &[a.size_bytes, b.size_bytes],
    );
}

pub fn gpu_scale(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, _scale: f32) {
    ctx.dispatch_kernel_direct(
        SCALE,
        ((size + 255) / 256, 1, 1), (256, 1, 1),
        &[data.ptr as _],
        &[data.size_bytes],
    );
}

pub fn gpu_fill(ctx: &Arc<MetalContext>, data: &GpuBuffer, size: u32, _value: f32) {
    ctx.dispatch_kernel_direct(
        FILL,
        ((size + 255) / 256, 1, 1), (256, 1, 1),
        &[data.ptr as _],
        &[data.size_bytes],
    );
}

// AdamW hyperparams — same struct as Metal/CUDA
pub struct AdamWHyperparams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u32,
}

pub fn gpu_adamw_update(ctx: &Arc<MetalContext>, param: &GpuBuffer, grad: &GpuBuffer, m: &GpuBuffer, v: &GpuBuffer, size: u32, _hp: &AdamWHyperparams) {
    ctx.dispatch_kernel_direct(
        ADAMW,
        ((size + 255) / 256, 1, 1), (256, 1, 1),
        &[param.ptr as _, grad.ptr as _, m.ptr as _, v.ptr as _],
        &[param.size_bytes, grad.size_bytes, m.size_bytes, v.size_bytes],
    );
}

// ... remaining dispatch functions follow the same pattern.
// Every function has the SAME signature as metal/compute.rs.
// Only the dispatch mechanism changes: direct hardware write vs framework API.
