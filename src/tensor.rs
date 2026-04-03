use crate::autograd::{self, Op, TapeEntry};
use crate::metal::{compute, GpuBuffer, MetalContext};
use objc2::rc::Retained;
use objc2_metal::MTLBuffer;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for tensors on the autodiff tape.
pub type TensorId = usize;

// Thread-local FP16 cast cache: avoids redundant float→half casts for weights.
thread_local! {
    static F16_CAST_CACHE: RefCell<HashMap<usize, Retained<GpuBuffer>>> = RefCell::new(HashMap::new());
}

// Thread-local ternary weight cache: avoids requantizing every matmul call.
// Key: buffer pointer. Value: (packed_ternary, absmean) buffers.
thread_local! {
    static TERNARY_CACHE: RefCell<HashMap<usize, (Retained<GpuBuffer>, Retained<GpuBuffer>)>> = RefCell::new(HashMap::new());
}

/// A tensor backed by Metal shared memory with automatic differentiation support.
#[derive(Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub buffer: Retained<GpuBuffer>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub ctx: Arc<MetalContext>,
}

// SAFETY: Tensor's non-Send fields are Metal GPU buffers. All GPU dispatch
// in this codebase is synchronous (waitUntilCompleted), and Metal buffers
// on Apple Silicon are safe to reference from any thread.
//
// INVARIANT: The autograd tape (TAPE, GRADS) and buffer pool (BUFFER_POOL)
// are thread-local. All training/inference must run on a single thread.
// If multi-threaded training is ever needed, these must be moved to
// Arc<Mutex<>> or use crossbeam scoped threads with explicit synchronization.
// The batch mode (begin_batch/flush_batch) is also thread-local and must not
// span threads.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    /// Create a tensor from raw float data.
    pub fn from_slice(ctx: &Arc<MetalContext>, data: &[f32], shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "Data length {} != shape product {}", data.len(), expected);
        let buffer = ctx.buffer_from_slice(data);
        let id = autograd::next_id();
        Self { id, buffer, shape, requires_grad: false, ctx: Arc::clone(ctx) }
    }

    /// Create a tensor from an existing GPU buffer (no copy).
    pub fn from_buffer(ctx: Arc<MetalContext>, buffer: Retained<GpuBuffer>, shape: Vec<usize>) -> Self {
        let id = autograd::next_id();
        Self { id, buffer, shape, requires_grad: false, ctx }
    }

    /// Create a tensor of zeros.
    pub fn zeros(ctx: &Arc<MetalContext>, shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let buffer = ctx.alloc_buffer(size * 4);
        compute::gpu_fill(ctx, &buffer, size as u32, 0.0);
        let id = autograd::next_id();
        Self { id, buffer, shape, requires_grad: false, ctx: Arc::clone(ctx) }
    }

    /// Create a tensor filled with a value.
    pub fn full(ctx: &Arc<MetalContext>, shape: Vec<usize>, value: f32) -> Self {
        let size: usize = shape.iter().product();
        let buffer = ctx.alloc_buffer(size * 4);
        compute::gpu_fill(ctx, &buffer, size as u32, value);
        let id = autograd::next_id();
        Self { id, buffer, shape, requires_grad: false, ctx: Arc::clone(ctx) }
    }

    /// Create a parameter tensor (requires_grad=true) with random normal initialization.
    pub fn randn(ctx: &Arc<MetalContext>, shape: Vec<usize>, std_dev: f32) -> Self {
        use rand::Rng;
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..size)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rng.gen::<f32>().max(1e-7);
                let u2: f32 = rng.gen();
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) * std_dev
            })
            .collect();
        let buffer = ctx.buffer_from_slice(&data);
        let id = autograd::next_id();
        Self { id, buffer, shape, requires_grad: true, ctx: Arc::clone(ctx) }
    }

    /// Create a parameter tensor initialized with ones (for norm weights).
    pub fn ones(ctx: &Arc<MetalContext>, shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let buffer = ctx.alloc_buffer(size * 4);
        compute::gpu_fill(ctx, &buffer, size as u32, 1.0);
        let id = autograd::next_id();
        Self { id, buffer, shape, requires_grad: true, ctx: Arc::clone(ctx) }
    }

    /// Mark this tensor as a parameter (requires gradient).
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Mark this tensor as not requiring gradient.
    pub fn detach(&self) -> Self {
        let id = autograd::next_id();
        Self {
            id,
            buffer: self.buffer.clone(),
            shape: self.shape.clone(),
            requires_grad: false,
            ctx: Arc::clone(&self.ctx),
        }
    }

    /// Number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Read data back to CPU.
    pub fn to_vec(&self) -> Vec<f32> {
        MetalContext::read_buffer(&self.buffer, self.numel())
    }

    /// Zero-copy read access to GPU buffer contents as a slice.
    /// On Apple Silicon with shared memory, this is a direct pointer — no copy.
    /// The slice is valid only while no GPU writes are pending on this buffer.
    pub fn as_slice(&self) -> &[f32] {
        MetalContext::buffer_as_slice(&self.buffer, self.numel())
    }

    /// Read data into a pre-allocated buffer — avoids allocation in hot loops.
    pub fn read_into(&self, dst: &mut [f32]) {
        assert!(dst.len() >= self.numel(), "destination buffer too small");
        MetalContext::read_buffer_into(&self.buffer, &mut dst[..self.numel()]);
    }

    // ===== Operations =====

    /// BitNet matmul: self @ weight where weight is quantized to ternary {-1,0,+1}.
    /// self: [M, K] float, weight: [K, N] float (quantized on-the-fly).
    /// Uses Straight-Through Estimator: forward uses ternary, backward uses float.
    /// BitNet b1.58 matmul: quantize weight to ternary, compute via add/subtract.
    /// Caches packed ternary weights per step (cleared with FP16 cache after optimizer).
    pub fn ternary_matmul(&self, weight: &Tensor) -> Tensor {
        let m = self.shape[0];
        let k = self.shape[1];
        let n = weight.shape[1];
        assert_eq!(self.shape[1], weight.shape[0], "ternary_matmul K mismatch");

        let packed_rows = (k + 15) / 16;

        // Check ternary cache (same pattern as FP16 cache)
        let cache_key = weight.buffer.contents().as_ptr() as usize;
        let cached = TERNARY_CACHE.with(|c| c.borrow().get(&cache_key).cloned());

        let (packed_buf, absmean_buf) = if let Some((p, a)) = cached {
            (p, a)
        } else {
            // Quantize: compute absmean threshold per column, pack to ternary
            let absmean = self.ctx.alloc_buffer(n * 4);
            compute::gpu_ternary_absmean(&self.ctx, &weight.buffer, &absmean, k as u32, n as u32);
            let packed = self.ctx.alloc_buffer(packed_rows * n * 4);
            compute::gpu_ternary_pack(&self.ctx, &weight.buffer, &absmean, &packed, k as u32, n as u32);
            TERNARY_CACHE.with(|c| c.borrow_mut().insert(cache_key, (packed.clone(), absmean.clone())));
            (packed, absmean)
        };

        // Ternary matmul: add/subtract only, no float multiply
        let out_buf = self.ctx.alloc_buffer(m * n * 4);
        compute::gpu_ternary_matmul(&self.ctx, &self.buffer, &packed_buf, &out_buf, m as u32, n as u32, k as u32);

        // Scale output by absmean per column: out[i][j] *= absmean[j]
        // This restores the magnitude lost by ternary quantization
        let out_tensor = Tensor::from_buffer(Arc::clone(&self.ctx), out_buf, vec![m, n]);
        let absmean_tensor = Tensor::from_buffer(Arc::clone(&self.ctx), absmean_buf, vec![n]);
        // Broadcast scale: use scale_rows with transposed logic
        // For column scaling, we'd need a scale_cols op. Approximate with mean scale for now.
        // Read first absmean value as approximate global scale
        // (proper per-column scaling needs a scale_cols kernel — future optimization)
        let absmean_data = MetalContext::read_buffer(&absmean_tensor.buffer, n.min(1));
        let mean_scale = if !absmean_data.is_empty() && absmean_data[0] > 0.0 { absmean_data[0] } else { 1.0 };
        let out_scaled = out_tensor.scale(mean_scale);

        // STE: the scale() call above already records Op::Scale on the tape.
        // For backward, gradients flow through scale → out_tensor.
        // out_tensor was created from from_buffer (not on tape), but the matmul
        // backward is handled by recording the STE matmul entry:
        if self.requires_grad || weight.requires_grad || autograd::is_recording() {
            autograd::record(autograd::TapeEntry {
                op: autograd::Op::Matmul, // STE: backward treats ternary as regular matmul
                inputs: vec![self.id, weight.id],
                output: out_scaled.id,
                input_buffers: vec![self.buffer.clone(), weight.buffer.clone()],
                output_buffer: out_scaled.buffer.clone(),
                shapes: vec![self.shape.clone(), weight.shape.clone(), out_scaled.shape.clone()],
                cached: None,
            });
        }

        out_scaled
    }

    /// Per-row scaling: output[r][c] = self[r][c] * scales[r]
    /// self: [rows, cols], scales: [rows] → [rows, cols]
    /// On autograd tape — gradients flow to both input and scales.
    pub fn scale_rows(&self, scales: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "scale_rows requires 2D tensor");
        let rows = self.shape[0];
        let cols = self.shape[1];
        assert_eq!(scales.shape, vec![rows], "scales must be [rows]");

        let out_buf = self.ctx.alloc_buffer(rows * cols * 4);
        compute::gpu_scale_rows(&self.ctx, &self.buffer, &scales.buffer, &out_buf, rows as u32, cols as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf.clone(),
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || scales.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || scales.requires_grad || autograd::is_recording() {
            autograd::record(autograd::TapeEntry {
                op: autograd::Op::ScaleRows { rows, cols },
                inputs: vec![self.id, scales.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), scales.buffer.clone()],
                output_buffer: out_buf,
                shapes: vec![self.shape.clone(), scales.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Cast tensor contents to FP16 buffer with safe clamping.
    /// Uses thread-local cache: same buffer pointer → cached FP16 version.
    /// Call `Tensor::clear_f16_cache()` after optimizer step when weights change.
    pub fn cast_to_f16(&self) -> Retained<crate::metal::GpuBuffer> {
        let key = self.buffer.contents().as_ptr() as usize;

        let cached = F16_CAST_CACHE.with(|c| c.borrow().get(&key).cloned());
        if let Some(buf) = cached {
            return buf;
        }

        let size = self.numel();
        let f16_buf = self.ctx.alloc_buffer(size * 2);
        compute::gpu_cast_f32_to_f16(&self.ctx, &self.buffer, &f16_buf, size as u32);
        F16_CAST_CACHE.with(|c| c.borrow_mut().insert(key, f16_buf.clone()));
        f16_buf
    }

    /// FP16 roundtrip: cast FP32→FP16→FP32 to reduce activation precision.
    /// The roundtrip quantizes values to half precision (~0.1% loss).
    /// On the autograd tape, this is a no-op for gradient flow (identity backward).
    /// The purpose: when gradient checkpointing stores this tensor as a checkpoint
    /// input, the quantized version has less numerical variation, and the tape's
    /// input_buffers reference a buffer that was written from a smaller source.
    pub fn fp16_roundtrip(&self) -> Tensor {
        let size = self.numel();
        let f16_buf = self.ctx.alloc_buffer(size * 2);
        compute::gpu_cast_f32_to_f16(&self.ctx, &self.buffer, &f16_buf, size as u32);
        let f32_buf = self.ctx.alloc_buffer(size * 4);
        compute::gpu_cast_f16_to_f32(&self.ctx, &f16_buf, &f32_buf, size as u32);

        // Record as identity on tape — gradient passes through unchanged
        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id, buffer: f32_buf, shape: self.shape.clone(),
            requires_grad: self.requires_grad, ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Reshape, // Reuse Reshape — it's a gradient passthrough
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Clear weight caches. Call after optimizer step when weights change.
    pub fn clear_f16_cache() {
        F16_CAST_CACHE.with(|c| c.borrow_mut().clear());
        TERNARY_CACHE.with(|c| c.borrow_mut().clear());
    }

    /// Clear FP16 cache and recycle buffers to the pool instead of dropping.
    pub fn clear_f16_cache_recycle() {
        F16_CAST_CACHE.with(|c| {
            for (_key, buf) in c.borrow_mut().drain() {
                MetalContext::recycle_buffer(buf);
            }
        });
        TERNARY_CACHE.with(|c| c.borrow_mut().clear());
    }

    /// Matrix multiplication: self @ other
    /// self: [..., M, K], other: [..., K, N] → [..., M, N]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let rank_a = self.shape.len();
        let rank_b = other.shape.len();
        assert!(rank_a >= 2 && rank_b >= 2, "matmul requires at least 2D tensors");

        let m = self.shape[rank_a - 2];
        let k = self.shape[rank_a - 1];
        let k2 = other.shape[rank_b - 2];
        let n = other.shape[rank_b - 1];
        assert_eq!(k, k2, "matmul inner dimensions must match: {} vs {}", k, k2);

        // Compute batch dimensions
        let batch_a: usize = self.shape[..rank_a - 2].iter().product();
        let batch_b: usize = other.shape[..rank_b - 2].iter().product();
        let batch = batch_a.max(batch_b);
        assert!(batch_a == 1 || batch_b == 1 || batch_a == batch_b,
            "Incompatible batch dimensions");

        let total_m = batch * m;
        let out_size = total_m * n;
        let out_buf = self.ctx.alloc_buffer(out_size * 4);

        // FP16 matmul with clamped cast — prevents NaN from half overflow
        if batch == 1 {
            let a_f16 = self.cast_to_f16();
            let b_f16 = other.cast_to_f16();
            compute::gpu_matmul_f16(&self.ctx, &a_f16, &b_f16, &out_buf, m as u32, n as u32, k as u32);
        } else {
            // Sequential batch dispatch — each batch element is a separate matmul
            // All operations stay on GPU — no CPU readback
            let a_stride = if batch_a == 1 { 0 } else { m * k };
            let b_stride = if batch_b == 1 { 0 } else { k * n };
            for b in 0..batch {
                let a_offset = b * a_stride;
                let b_offset = b * b_stride;
                let c_offset = b * m * n;

                let a_sub = self.ctx.alloc_buffer(m * k * 4);
                compute::gpu_buffer_copy(&self.ctx, &self.buffer, &a_sub, a_offset as u32, 0, (m * k) as u32);

                let b_sub = self.ctx.alloc_buffer(k * n * 4);
                compute::gpu_buffer_copy(&self.ctx, &other.buffer, &b_sub, b_offset as u32, 0, (k * n) as u32);

                let c_sub = self.ctx.alloc_buffer(m * n * 4);
                compute::gpu_matmul(&self.ctx, &a_sub, &b_sub, &c_sub, m as u32, n as u32, k as u32);

                compute::gpu_buffer_copy(&self.ctx, &c_sub, &out_buf, 0, c_offset as u32, (m * n) as u32);
            }
        }

        let mut out_shape = self.shape[..rank_a - 2].to_vec();
        if batch > 1 && batch_a == 1 {
            out_shape = other.shape[..rank_b - 2].to_vec();
        }
        out_shape.push(m);
        out_shape.push(n);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: out_shape,
            requires_grad: self.requires_grad || other.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || other.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Matmul,
                inputs: vec![self.id, other.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), other.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), other.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Matrix multiply where B (other) is detached — no gradient flows to B.
    /// Used for ReLoRA frozen base weights: forward uses them, backward skips them.
    /// Gradients still flow to self (the input activations).
    pub fn matmul_detached(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul_detached expects 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul_detached expects 2D tensors");
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        assert_eq!(k, other.shape[0], "inner dimensions must match");

        let out_buf = self.ctx.alloc_buffer(m * n * 4);
        let a_f16 = self.cast_to_f16();
        let b_f16 = other.cast_to_f16();
        compute::gpu_matmul_f16(&self.ctx, &a_f16, &b_f16, &out_buf, m as u32, n as u32, k as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: vec![m, n],
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::MatmulDetachedB,
                inputs: vec![self.id, other.id], // other.id recorded but won't get grad
                input_buffers: vec![self.buffer.clone(), other.buffer.clone()],
                output: out_id,
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), other.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Matrix multiply with B transposed: self @ other^T
    /// self: [M, K], other: [N, K] → [M, N]
    pub fn matmul_trans_b(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul_trans_b expects 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul_trans_b expects 2D tensors");
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[0];
        assert_eq!(k, other.shape[1], "Inner dim mismatch");

        let out_buf = self.ctx.alloc_buffer(m * n * 4);
        // FP16 path: cast inputs to half, halves bandwidth
        let a_f16 = self.cast_to_f16();
        let b_f16 = other.cast_to_f16();
        compute::gpu_matmul_trans_b_f16(&self.ctx, &a_f16, &b_f16, &out_buf, m as u32, n as u32, k as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: vec![m, n],
            requires_grad: self.requires_grad || other.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || other.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::MatmulTransB,
                inputs: vec![self.id, other.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), other.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), other.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Elementwise addition.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "add shape mismatch: {:?} vs {:?}", self.shape, other.shape);
        let size = self.numel();
        let out_buf = self.ctx.alloc_buffer(size * 4);
        compute::gpu_add(&self.ctx, &self.buffer, &other.buffer, &out_buf, size as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || other.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || other.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Add,
                inputs: vec![self.id, other.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), other.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), other.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Elementwise multiply.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "mul shape mismatch");
        let size = self.numel();
        let out_buf = self.ctx.alloc_buffer(size * 4);
        compute::gpu_mul(&self.ctx, &self.buffer, &other.buffer, &out_buf, size as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || other.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || other.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Mul,
                inputs: vec![self.id, other.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), other.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), other.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Row-wise softmax (last dimension).
    pub fn softmax(&self) -> Tensor {
        assert!(!self.shape.is_empty(), "softmax needs at least 1D");
        let cols = *self.shape.last().unwrap();
        let rows: usize = self.numel() / cols;
        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_softmax(&self.ctx, &self.buffer, &out_buf, rows as u32, cols as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf.clone(),
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Softmax,
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: Some(out_buf), // cache softmax output for backward
            });
        }

        out
    }

    /// RMS normalization over the last dimension.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
        let cols = *self.shape.last().unwrap();
        let rows: usize = self.numel() / cols;
        assert_eq!(weight.shape, vec![cols], "norm weight shape mismatch");

        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_rms_norm(&self.ctx, &self.buffer, &weight.buffer, &out_buf, rows as u32, cols as u32, eps);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || weight.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || weight.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::RmsNorm { eps },
                inputs: vec![self.id, weight.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), weight.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), weight.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Fused residual add + RMS norm: output = rms_norm(self + residual, weight, eps)
    /// Saves 1 kernel dispatch + 1 temp buffer vs separate add() + rms_norm().
    pub fn rms_norm_residual(&self, residual: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        assert_eq!(self.shape, residual.shape, "rms_norm_residual shape mismatch");
        let cols = *self.shape.last().unwrap();
        let rows: usize = self.numel() / cols;
        assert_eq!(weight.shape, vec![cols], "norm weight shape mismatch");

        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        let sum_buf = self.ctx.alloc_buffer(self.numel() * 4); // stores (self + residual)
        compute::gpu_rms_norm_residual(
            &self.ctx, &self.buffer, &residual.buffer, &weight.buffer,
            &out_buf, &sum_buf, rows as u32, cols as u32, eps,
        );

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || residual.requires_grad || weight.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || residual.requires_grad || weight.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::RmsNormResidual { eps },
                inputs: vec![self.id, residual.id, weight.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), residual.buffer.clone(), weight.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), residual.shape.clone(), weight.shape.clone(), out.shape.clone()],
                cached: Some(sum_buf),
            });
        }

        out
    }

    /// Fused add + RMS norm that also returns the sum tensor.
    /// Computes: sum = self + residual, out = rms_norm(sum, weight, eps)
    /// Returns (out, sum) — saves 1 kernel dispatch vs separate add + rms_norm.
    pub fn rms_norm_residual_with_sum(&self, residual: &Tensor, weight: &Tensor, eps: f32) -> (Tensor, Tensor) {
        assert_eq!(self.shape, residual.shape, "rms_norm_residual shape mismatch");
        let cols = *self.shape.last().unwrap();
        let rows: usize = self.numel() / cols;
        assert_eq!(weight.shape, vec![cols], "norm weight shape mismatch");

        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        let sum_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_rms_norm_residual(
            &self.ctx, &self.buffer, &residual.buffer, &weight.buffer,
            &out_buf, &sum_buf, rows as u32, cols as u32, eps,
        );

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || residual.requires_grad || weight.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        let sum_id = autograd::next_id();
        let sum = Tensor {
            id: sum_id,
            buffer: sum_buf.clone(),
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || residual.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || residual.requires_grad || weight.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::RmsNormResidual { eps },
                inputs: vec![self.id, residual.id, weight.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), residual.buffer.clone(), weight.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), residual.shape.clone(), weight.shape.clone(), out.shape.clone()],
                cached: Some(sum_buf),
            });
        }

        (out, sum)
    }

    /// SiLU activation: x * sigmoid(x)
    pub fn silu(&self) -> Tensor {
        let size = self.numel();
        let out_buf = self.ctx.alloc_buffer(size * 4);
        compute::gpu_silu(&self.ctx, &self.buffer, &out_buf, size as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Silu,
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Fused SiLU-gate: output = silu(self) * other
    /// Replaces separate silu() + mul() with a single kernel dispatch.
    pub fn silu_gate(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "silu_gate shape mismatch");
        let size = self.numel();
        let out_buf = self.ctx.alloc_buffer(size * 4);
        compute::gpu_silu_gate(&self.ctx, &self.buffer, &other.buffer, &out_buf, size as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad || other.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || other.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::SiluGate,
                inputs: vec![self.id, other.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), other.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), other.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Reshape tensor (view, no data copy if contiguous).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(self.numel(), new_numel, "reshape: incompatible sizes {} vs {}", self.numel(), new_numel);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: self.buffer.clone(),
            shape: new_shape,
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Reshape,
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Apply RoPE positional encoding in-place.
    /// self shape: [batch_heads, seq_len, head_dim]
    pub fn apply_rope(&self, offset: u32, theta: f32) -> Tensor {
        assert_eq!(self.shape.len(), 3, "rope expects 3D: [batch_heads, seq, head_dim]");
        let total_rows = self.shape[0];
        let seq_len = self.shape[1];
        let head_dim = self.shape[2];

        // Out-of-place RoPE: dst = rotate(src, θ) in 1 dispatch (was copy + in-place = 2)
        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_rope_copy(&self.ctx, &self.buffer, &out_buf,
            total_rows as u32, seq_len as u32, head_dim as u32, offset, theta);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::RoPE { seq_len: seq_len as u32, head_dim: head_dim as u32, offset, theta },
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Apply causal mask to attention scores.
    /// self shape: [batch_heads, seq_q, seq_k]
    /// Gradient passes through unchanged for non-masked positions (masked positions get zero grad).
    pub fn causal_mask(&self, offset: u32) -> Tensor {
        assert_eq!(self.shape.len(), 3);
        let batch_heads = self.shape[0];
        let seq_q = self.shape[1];
        let seq_k = self.shape[2];

        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_copy(&self.ctx, &self.buffer, &out_buf, self.numel() as u32);
        compute::gpu_causal_mask(&self.ctx, &out_buf, batch_heads as u32, seq_q as u32, seq_k as u32, offset);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        // Record passthrough on tape — gradient flows through unchanged.
        // Masked positions already get -inf → softmax gives 0 weight → zero gradient naturally.
        // No special backward needed: just pass gradient through like Reshape.
        if autograd::is_recording() {
            autograd::record(autograd::TapeEntry {
                op: Op::Reshape, // passthrough: same data, gradient flows through
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone()],
                cached: None,
            });
        }
        out
    }

    /// Fused RMSNorm + Matmul: computes rms_norm(self, weight, eps) @ b in 2 dispatches
    /// instead of 3 (norm + matmul). Eliminates the intermediate normalized buffer.
    /// self: [M, K], weight: [K], b: [K, N] → output: [M, N]
    pub fn fused_norm_matmul(&self, weight: &Tensor, b: &Tensor, eps: f32) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let m = self.shape[0];
        let k = self.shape[1];
        let n = b.shape[1];
        assert_eq!(b.shape[0], k);
        assert_eq!(weight.shape[0], k);

        let out_buf = self.ctx.alloc_buffer(m * n * 4);
        compute::gpu_fused_norm_matmul(
            &self.ctx, &self.buffer, &weight.buffer, &b.buffer, &out_buf,
            m as u32, n as u32, k as u32, eps,
        );

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id, buffer: out_buf, shape: vec![m, n],
            requires_grad: self.requires_grad || b.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        // For backward: we need the original input (for norm backward) and b (for matmul backward).
        // Record as a standard Matmul op — the norm is absorbed into the computation.
        // The backward will compute gradients as if the normalized input was the matmul input.
        // This is an approximation — the true backward needs the norm Jacobian — but for
        // inference and forward-only mode, it's exact. For training backward, use the
        // separate norm + matmul path (which we keep as the default).
        if self.requires_grad || b.requires_grad || autograd::is_recording() {
            // Record norm + matmul as two ops for correct backward
            // The fused kernel only saves forward compute — backward still needs both ops
            // For now, record as a Matmul with the original input (backward will be approximate)
            autograd::record(TapeEntry {
                op: Op::Matmul,
                inputs: vec![self.id, b.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), b.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), b.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// ReLU activation: output = max(input, 0). Tape-tracked for gradient flow.
    pub fn relu(&self) -> Tensor {
        let size = self.numel();
        let out_buf = self.ctx.alloc_buffer(size * 4);
        compute::gpu_relu(&self.ctx, &self.buffer, &out_buf, size as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id, buffer: out_buf, shape: self.shape.clone(),
            requires_grad: self.requires_grad, ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Relu,
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }
        out
    }

    /// Causal mask with sliding window: mask future AND positions beyond window distance.
    pub fn causal_mask_window(&self, offset: u32, window: u32) -> Tensor {
        assert_eq!(self.shape.len(), 3);
        let batch_heads = self.shape[0];
        let seq_q = self.shape[1];
        let seq_k = self.shape[2];

        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_copy(&self.ctx, &self.buffer, &out_buf, self.numel() as u32);
        compute::gpu_causal_mask_window(&self.ctx, &out_buf, batch_heads as u32, seq_q as u32, seq_k as u32, offset, window);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if autograd::is_recording() {
            autograd::record(autograd::TapeEntry {
                op: Op::Reshape,
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone()],
                cached: None,
            });
        }
        out
    }

    /// Fused scale + causal mask + softmax in one kernel.
    /// self shape: [batch_heads, seq_q, seq_k] (raw attention scores)
    /// Returns softmax(self * scale, causal_masked).
    pub fn scaled_causal_softmax(&self, scale: f32, kv_offset: u32) -> Tensor {
        assert_eq!(self.shape.len(), 3, "scaled_causal_softmax needs 3D [batch_heads, seq_q, seq_k]");
        let batch_heads = self.shape[0];
        let seq_q = self.shape[1];
        let seq_k = self.shape[2];
        let total_rows = batch_heads * seq_q;

        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_scaled_causal_softmax(
            &self.ctx, &self.buffer, &out_buf,
            total_rows as u32, seq_q as u32, seq_k as u32, scale, kv_offset,
        );

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf.clone(),
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::ScaledCausalSoftmax { scale },
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: Some(out_buf),
            });
        }

        out
    }

    /// Column-wise slice: extract columns [col_offset..col_offset+dst_cols) from [rows, src_cols].
    /// Returns [rows, dst_cols]. Tape-tracked for gradient flow.
    pub fn slice_cols(&self, col_offset: usize, dst_cols: usize) -> Tensor {
        assert_eq!(self.shape.len(), 2, "slice_cols needs 2D, got {:?}", self.shape);
        let rows = self.shape[0];
        let src_cols = self.shape[1];
        assert!(col_offset + dst_cols <= src_cols);

        let out_buf = self.ctx.alloc_buffer(rows * dst_cols * 4);
        compute::gpu_slice_cols(&self.ctx, &self.buffer, &out_buf,
            rows as u32, src_cols as u32, dst_cols as u32, col_offset as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: vec![rows, dst_cols],
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::SliceCols { rows, src_cols, dst_cols, col_offset },
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Slice a contiguous region from the flat buffer. Tape-tracked for gradient flow.
    /// Returns a tensor with `length` elements starting at `offset` in the flat buffer.
    pub fn slice_flat(&self, offset: usize, length: usize, new_shape: Vec<usize>) -> Tensor {
        let source_size = self.numel();
        assert!(offset + length <= source_size, "slice out of bounds");
        assert_eq!(length, new_shape.iter().product::<usize>(), "shape doesn't match length");

        // Copy the slice into a new buffer using GPU buffer copy — no CPU roundtrip
        let out_buf = self.ctx.alloc_buffer(length * 4);
        compute::gpu_buffer_copy(&self.ctx, &self.buffer, &out_buf, offset as u32, 0, length as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: new_shape,
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Slice { offset, length, source_size },
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Concatenate multiple tensors along flat dimension. Tape-tracked.
    pub fn concat_flat(tensors: &[&Tensor], new_shape: Vec<usize>) -> Tensor {
        let ctx = &tensors[0].ctx;
        let total: usize = tensors.iter().map(|t| t.numel()).sum();
        assert_eq!(total, new_shape.iter().product::<usize>(), "shape doesn't match total");

        let mut part_sizes = Vec::with_capacity(tensors.len());
        let mut input_ids = Vec::with_capacity(tensors.len());
        let mut input_bufs = Vec::with_capacity(tensors.len());

        // Concatenate on GPU using buffer_copy — no CPU roundtrip
        let out_buf = ctx.alloc_buffer(total * 4);
        let mut offset = 0u32;
        for t in tensors {
            let n = t.numel();
            compute::gpu_buffer_copy(ctx, &t.buffer, &out_buf, 0, offset, n as u32);
            part_sizes.push(n);
            input_ids.push(t.id);
            input_bufs.push(t.buffer.clone());
            offset += n as u32;
        }
        let any_grad = tensors.iter().any(|t| t.requires_grad);
        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: new_shape,
            requires_grad: any_grad,
            ctx: Arc::clone(ctx),
        };

        if autograd::is_recording() {
            let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).chain(std::iter::once(out.shape.clone())).collect();
            autograd::record(TapeEntry {
                op: Op::ConcatParts { part_sizes },
                inputs: input_ids,
                output: out_id,
                input_buffers: input_bufs,
                output_buffer: out.buffer.clone(),
                shapes,
                cached: None,
            });
        }

        out
    }

    /// Batched matrix multiplication: self[b] @ other[b] for each batch element.
    /// self: [B, M, K], other: [B, K, N] → result: [B, M, N]
    /// Records a single Op::BatchedMatmul tape entry.
    pub fn batched_matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 3, "batched_matmul expects 3D tensors, got {:?}", self.shape);
        assert_eq!(other.shape.len(), 3, "batched_matmul expects 3D tensors, got {:?}", other.shape);
        let batches = self.shape[0];
        let m = self.shape[1];
        let k = self.shape[2];
        assert_eq!(other.shape[0], batches, "batch dim mismatch: {} vs {}", batches, other.shape[0]);
        assert_eq!(other.shape[1], k, "inner dim mismatch: {} vs {}", k, other.shape[1]);
        let n = other.shape[2];

        let out_buf = self.ctx.alloc_buffer(batches * m * n * 4);

        // Batched matmul — uses FP32 inputs (small attention dims don't benefit from FP16 cast overhead)
        compute::gpu_batched_matmul(&self.ctx, &self.buffer, &other.buffer, &out_buf, batches as u32, m as u32, n as u32, k as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: vec![batches, m, n],
            requires_grad: self.requires_grad || other.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || other.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::BatchedMatmul,
                inputs: vec![self.id, other.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), other.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), other.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Batched matrix multiply with B transposed: self[b] @ other[b]^T for each batch element.
    /// self: [B, M, K], other: [B, N, K] → result: [B, M, N]
    /// Records a single Op::BatchedMatmulTransB tape entry.
    pub fn batched_matmul_trans_b(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 3, "batched_matmul_trans_b expects 3D tensors, got {:?}", self.shape);
        assert_eq!(other.shape.len(), 3, "batched_matmul_trans_b expects 3D tensors, got {:?}", other.shape);
        let batches = self.shape[0];
        let m = self.shape[1];
        let k = self.shape[2];
        assert_eq!(other.shape[0], batches, "batch dim mismatch: {} vs {}", batches, other.shape[0]);
        assert_eq!(other.shape[2], k, "inner dim mismatch: {} vs {}", k, other.shape[2]);
        let n = other.shape[1];

        let out_buf = self.ctx.alloc_buffer(batches * m * n * 4);

        // Batched matmul trans_b — uses FP32 inputs (small attention dims)
        compute::gpu_batched_matmul_trans_b(&self.ctx, &self.buffer, &other.buffer, &out_buf, batches as u32, m as u32, n as u32, k as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: vec![batches, m, n],
            requires_grad: self.requires_grad || other.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || other.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::BatchedMatmulTransB,
                inputs: vec![self.id, other.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone(), other.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), other.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }

    /// Scale: self * scalar
    pub fn scale(&self, factor: f32) -> Tensor {
        // Fused out-of-place: dst = src * factor in 1 dispatch (was copy + scale = 2)
        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_scale_copy(&self.ctx, &self.buffer, &out_buf, self.numel() as u32, factor);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            ctx: Arc::clone(&self.ctx),
        };

        if self.requires_grad || autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Scale { factor },
                inputs: vec![self.id],
                output: out_id,
                input_buffers: vec![self.buffer.clone()],
                output_buffer: out.buffer.clone(),
                shapes: vec![self.shape.clone(), out.shape.clone()],
                cached: None,
            });
        }

        out
    }
}
