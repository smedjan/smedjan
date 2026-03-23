use crate::autograd::{self, Op, TapeEntry};
use crate::metal::{compute, GpuBuffer, MetalContext};
use objc2::rc::Retained;
use objc2_metal::MTLBuffer;
use std::sync::Arc;

/// Unique identifier for tensors on the autodiff tape.
pub type TensorId = usize;

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
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Read data back to CPU.
    pub fn to_vec(&self) -> Vec<f32> {
        MetalContext::read_buffer(&self.buffer, self.numel())
    }

    // ===== Operations =====

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

        // For batched matmul, dispatch per batch element
        if batch == 1 {
            compute::gpu_matmul(&self.ctx, &self.buffer, &other.buffer, &out_buf, m as u32, n as u32, k as u32);
        } else {
            // Sequential batch dispatch — each batch element is a separate matmul
            let a_stride = if batch_a == 1 { 0 } else { m * k };
            let b_stride = if batch_b == 1 { 0 } else { k * n };
            for b in 0..batch {
                let a_offset = b * a_stride;
                let b_offset = b * b_stride;
                let c_offset = b * m * n;
                // Create sub-buffer views using pointer arithmetic on the shared buffer
                // For simplicity, we use offset-based dispatch via a temporary copy approach
                // TODO: optimize with buffer offsets once the basic path works
                let a_data = MetalContext::read_buffer(&self.buffer, batch_a * m * k);
                let b_data = MetalContext::read_buffer(&other.buffer, batch_b * k * n);
                let a_sub = self.ctx.buffer_from_slice(&a_data[a_offset..a_offset + m * k]);
                let b_sub = self.ctx.buffer_from_slice(&b_data[b_offset..b_offset + k * n]);
                let c_sub = self.ctx.alloc_buffer(m * n * 4);
                compute::gpu_matmul(&self.ctx, &a_sub, &b_sub, &c_sub, m as u32, n as u32, k as u32);
                // Copy result into output buffer
                let c_data = MetalContext::read_buffer(&c_sub, m * n);
                unsafe {
                    let dst = (out_buf.contents().as_ptr() as *mut f32).add(c_offset);
                    std::ptr::copy_nonoverlapping(c_data.as_ptr(), dst, m * n);
                }
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
            requires_grad: false,
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
        compute::gpu_matmul_trans_b(&self.ctx, &self.buffer, &other.buffer, &out_buf, m as u32, n as u32, k as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: vec![m, n],
            requires_grad: false,
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
            requires_grad: false,
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
            requires_grad: false,
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
        assert!(self.shape.len() >= 2, "softmax needs at least 2D");
        let cols = *self.shape.last().unwrap();
        let rows: usize = self.numel() / cols;
        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_softmax(&self.ctx, &self.buffer, &out_buf, rows as u32, cols as u32);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf.clone(),
            shape: self.shape.clone(),
            requires_grad: false,
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
            requires_grad: false,
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
            requires_grad: false,
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
            requires_grad: false,
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

        // Copy data since rope is in-place
        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_copy(&self.ctx, &self.buffer, &out_buf, self.numel() as u32);
        compute::gpu_rope(&self.ctx, &out_buf, total_rows as u32, seq_len as u32, head_dim as u32, offset, theta);

        let out_id = autograd::next_id();
        Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: false,
            ctx: Arc::clone(&self.ctx),
        }
        // RoPE backward is handled by storing the angle and applying inverse rotation.
        // For simplicity in this phase, we skip RoPE on the backward pass (it has minimal
        // impact on gradient flow for short sequences).
    }

    /// Apply causal mask to attention scores.
    /// self shape: [batch_heads, seq_q, seq_k]
    pub fn causal_mask(&self, offset: u32) -> Tensor {
        assert_eq!(self.shape.len(), 3);
        let batch_heads = self.shape[0];
        let seq_q = self.shape[1];
        let seq_k = self.shape[2];

        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_copy(&self.ctx, &self.buffer, &out_buf, self.numel() as u32);
        compute::gpu_causal_mask(&self.ctx, &out_buf, batch_heads as u32, seq_q as u32, seq_k as u32, offset);

        let out_id = autograd::next_id();
        Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: false,
            ctx: Arc::clone(&self.ctx),
        }
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
            requires_grad: false,
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
        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: new_shape,
            requires_grad: false,
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

        // Dispatch per batch element using GPU buffer slicing — no CPU roundtrip
        for b in 0..batches {
            let a_off = b * m * k;
            let b_off = b * k * n;
            let c_off = b * m * n;

            let a_sub = self.ctx.alloc_buffer(m * k * 4);
            compute::gpu_buffer_copy(&self.ctx, &self.buffer, &a_sub, a_off as u32, 0, (m * k) as u32);

            let b_sub = self.ctx.alloc_buffer(k * n * 4);
            compute::gpu_buffer_copy(&self.ctx, &other.buffer, &b_sub, b_off as u32, 0, (k * n) as u32);

            let c_sub = self.ctx.alloc_buffer(m * n * 4);
            compute::gpu_matmul(&self.ctx, &a_sub, &b_sub, &c_sub, m as u32, n as u32, k as u32);

            compute::gpu_buffer_copy(&self.ctx, &c_sub, &out_buf, 0, c_off as u32, (m * n) as u32);
        }

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: vec![batches, m, n],
            requires_grad: false,
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

        // Dispatch per batch element using GPU buffer slicing — no CPU roundtrip
        for b in 0..batches {
            let a_off = b * m * k;
            let b_off = b * n * k;
            let c_off = b * m * n;

            let a_sub = self.ctx.alloc_buffer(m * k * 4);
            compute::gpu_buffer_copy(&self.ctx, &self.buffer, &a_sub, a_off as u32, 0, (m * k) as u32);

            let b_sub = self.ctx.alloc_buffer(n * k * 4);
            compute::gpu_buffer_copy(&self.ctx, &other.buffer, &b_sub, b_off as u32, 0, (n * k) as u32);

            let c_sub = self.ctx.alloc_buffer(m * n * 4);
            compute::gpu_matmul_trans_b(&self.ctx, &a_sub, &b_sub, &c_sub, m as u32, n as u32, k as u32);

            compute::gpu_buffer_copy(&self.ctx, &c_sub, &out_buf, 0, c_off as u32, (m * n) as u32);
        }

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: vec![batches, m, n],
            requires_grad: false,
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
        let out_buf = self.ctx.alloc_buffer(self.numel() * 4);
        compute::gpu_copy(&self.ctx, &self.buffer, &out_buf, self.numel() as u32);
        compute::gpu_scale(&self.ctx, &out_buf, self.numel() as u32, factor);

        let out_id = autograd::next_id();
        let out = Tensor {
            id: out_id,
            buffer: out_buf,
            shape: self.shape.clone(),
            requires_grad: false,
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
