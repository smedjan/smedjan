pub mod compute;
pub mod shaders;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice,
    MTLLibrary, MTLCreateSystemDefaultDevice, MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

/// Type alias for the buffer pool: maps buffer size → reusable Metal buffer list.
type BufferPool = HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>;

// Buffer pool: caches Metal buffers by size to avoid repeated allocations.
// Training steps allocate the same buffer sizes every iteration, so reuse is high.
//
// THREAD-LOCAL: This pool is per-thread. All GPU work (training, inference,
// autograd) must run on a single thread. If multi-threaded dispatch is ever
// needed, replace with a global Arc<Mutex<BufferPool>> or per-thread pools
// with cross-thread buffer transfer.
thread_local! {
    static BUFFER_POOL: RefCell<BufferPool> =
        RefCell::new(HashMap::new());
    static POOL_STATS: RefCell<(usize, usize)> = const { RefCell::new((0, 0)) }; // (hits, misses)
    // Allocation size log for debugging pool behavior. Enable with enable_alloc_log().
    static ALLOC_SIZE_LOG: RefCell<(bool, HashMap<usize, usize>)> = RefCell::new((false, HashMap::new()));
}

// Link CoreGraphics — required for MTLCreateSystemDefaultDevice
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// Type aliases for the objc2-metal protocol objects.
pub type GpuDevice = ProtocolObject<dyn MTLDevice>;
pub type GpuQueue = ProtocolObject<dyn MTLCommandQueue>;
pub type GpuBuffer = ProtocolObject<dyn MTLBuffer>;
pub type GpuPipeline = ProtocolObject<dyn MTLComputePipelineState>;
pub type GpuCommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
pub type GpuComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// Active command batch for kernel fusion. Accumulates multiple kernel dispatches
/// into a single command buffer, then commits and waits once.
/// This eliminates the per-kernel commit+wait overhead (~50-80% of wall time).
struct CommandBatch {
    cmd: Retained<GpuCommandBuffer>,
    encoder_count: usize,
}

thread_local! {
    /// Active command batch. When Some, kernels encode into this batch.
    /// When None, kernels create and commit individual command buffers (legacy sync path).
    static ACTIVE_BATCH: RefCell<Option<CommandBatch>> = const { RefCell::new(None) };
}

/// Metal context: device, command queue, and pre-compiled compute pipelines.
pub struct MetalContext {
    pub device: Retained<GpuDevice>,
    pub queue: Retained<GpuQueue>,
    pub pipelines: HashMap<&'static str, Retained<GpuPipeline>>,
}

// SAFETY: MetalContext is only used from a single thread in practice.
// Metal's command queue and device are thread-safe on Apple Silicon,
// and all GPU dispatch in this codebase is synchronous (waitUntilCompleted).
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl MetalContext {
    pub fn new() -> Arc<Self> {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let queue = device.newCommandQueue().expect("Failed to create command queue");

        let mut pipelines = HashMap::new();

        let shader_sources: &[(&str, &str)] = &[
            ("matmul_tiled", shaders::MATMUL_TILED),
            ("matmul_tiled_trans_b", shaders::MATMUL_TILED_TRANS_B),
            ("softmax", shaders::SOFTMAX),
            ("rms_norm", shaders::RMS_NORM),
            ("rms_norm_residual", shaders::RMS_NORM_RESIDUAL),
            ("rope", shaders::ROPE),
            ("rope_backward", shaders::ROPE_BACKWARD),
            ("add", shaders::ADD),
            ("add_inplace", shaders::ADD_INPLACE),
            ("mul", shaders::MUL),
            ("silu", shaders::SILU),
            ("silu_gate", shaders::SILU_GATE),
            ("cross_entropy", shaders::CROSS_ENTROPY),
            ("reduce_sum", shaders::REDUCE_SUM),
            ("adamw_update", shaders::ADAMW_UPDATE),
            ("embedding_lookup", shaders::EMBEDDING_LOOKUP),
            ("causal_mask", shaders::CAUSAL_MASK),
            ("l2_norm", shaders::L2_NORM),
            ("l2_norm_check", shaders::L2_NORM_CHECK),
            ("scale", shaders::SCALE),
            ("fill", shaders::FILL),
            ("copy_buffer", shaders::COPY),
            ("silu_backward", shaders::SILU_BACKWARD),
            ("silu_gate_backward", shaders::SILU_GATE_BACKWARD),
            ("rms_norm_backward", shaders::RMS_NORM_BACKWARD),
            ("softmax_backward", shaders::SOFTMAX_BACKWARD),
            ("embedding_backward", shaders::EMBEDDING_BACKWARD),
            ("zero_rows", shaders::ZERO_ROWS),
            ("transpose_2d", shaders::TRANSPOSE_2D),
            ("matmul_trans_a_tiled", shaders::MATMUL_TRANS_A),
            ("buffer_copy", shaders::BUFFER_COPY),
            ("transpose_perm_backward", shaders::TRANSPOSE_PERM_BACKWARD),
            ("transpose_perm_forward", shaders::TRANSPOSE_PERM_FORWARD),
            ("gradient_mask", shaders::GRADIENT_MASK),
            ("argmax", shaders::ARGMAX),
            ("temperature_scale", shaders::TEMPERATURE_SCALE),
            ("strided_batch_copy", shaders::STRIDED_BATCH_COPY),
            ("compact_strided_copy", shaders::COMPACT_STRIDED_COPY),
            ("batched_matmul_tiled", shaders::BATCHED_MATMUL_TILED),
            ("batched_matmul_tiled_trans_b", shaders::BATCHED_MATMUL_TILED_TRANS_B),
            ("batched_matmul_tiled_trans_a", shaders::BATCHED_MATMUL_TILED_TRANS_A),
            ("kl_divergence", shaders::KL_DIVERGENCE),
            ("flash_attention_forward", shaders::FLASH_ATTENTION_FORWARD),
            ("flash_attn_precompute_d", shaders::FLASH_ATTENTION_BACKWARD),
            ("moe_scatter_add", shaders::MOE_SCATTER_ADD),
            ("scale_rows", shaders::SCALE_ROWS),
            ("row_dot_reduce", shaders::ROW_DOT_REDUCE),
            ("lion_update", shaders::LION_UPDATE),
            ("sophia_update", shaders::SOPHIA_UPDATE),
            ("ternary_matmul", shaders::TERNARY_MATMUL),
            ("ternary_absmean", shaders::TERNARY_QUANTIZE),
            ("ternary_pack", shaders::TERNARY_QUANTIZE),
            ("moe_gather", shaders::MOE_GATHER),
            ("flash_attention_backward", shaders::FLASH_ATTENTION_BACKWARD),
            ("cast_f32_to_f16", shaders::CAST_F32_TO_F16),
            ("cast_f16_to_f32", shaders::CAST_F16_TO_F32),
            ("matmul_tiled_f16", shaders::MATMUL_TILED_F16),
            ("matmul_tiled_trans_b_f16", shaders::MATMUL_TILED_TRANS_B_F16),
            ("matmul_trans_a_tiled_f16", shaders::MATMUL_TRANS_A_F16),
            ("batched_matmul_tiled_f16", shaders::BATCHED_MATMUL_TILED_F16),
            ("batched_matmul_tiled_trans_b_f16", shaders::BATCHED_MATMUL_TILED_TRANS_B_F16),
            ("batched_matmul_tiled_trans_a_f16", shaders::BATCHED_MATMUL_TILED_TRANS_A_F16),
            ("repeat_kv", shaders::REPEAT_KV),
            ("repeat_kv_backward", shaders::REPEAT_KV_BACKWARD),
            ("scaled_causal_softmax", shaders::SCALED_CAUSAL_SOFTMAX),
            ("scale_copy", shaders::SCALE_COPY),
            ("rope_copy", shaders::ROPE_COPY),
            ("rope_backward_copy", shaders::ROPE_BACKWARD_COPY),
            ("matmul_narrow", shaders::MATMUL_NARROW),
            ("axpy", shaders::AXPY),
            ("relu", shaders::RELU),
            ("relu_backward", shaders::RELU_BACKWARD),
            ("ema_update", shaders::EMA_UPDATE),
            ("logsumexp", shaders::LOGSUMEXP),
            ("concat_cols", shaders::CONCAT_COLS),
            ("slice_cols", shaders::SLICE_COLS),
        ];

        let compile_options = MTLCompileOptions::new();

        for (kernel_name, source) in shader_sources {
            let ns_source = NSString::from_str(source);

            let library = device
                .newLibraryWithSource_options_error(&ns_source, Some(&compile_options))
                .unwrap_or_else(|e| {
                    panic!("Failed to compile shader '{}': {}", kernel_name, e)
                });

            let ns_name = NSString::from_str(kernel_name);
            let function = library
                .newFunctionWithName(&ns_name)
                .unwrap_or_else(|| {
                    panic!("Failed to get function '{}' from library", kernel_name)
                });

            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .unwrap_or_else(|e| {
                    panic!("Failed to create pipeline for '{}': {}", kernel_name, e)
                });

            pipelines.insert(*kernel_name, pipeline);
        }

        Arc::new(Self {
            device,
            queue,
            pipelines,
        })
    }

    /// Allocate a shared-mode Metal buffer (CPU + GPU accessible, zero-copy on M1).
    /// Checks the buffer pool first for a cached buffer of the same size.
    pub fn alloc_buffer(&self, size_bytes: usize) -> Retained<GpuBuffer> {
        // Debug: track allocation sizes on first step
        ALLOC_SIZE_LOG.with(|log| {
            let mut l = log.borrow_mut();
            if l.0 { // logging enabled
                *l.1.entry(size_bytes).or_insert(0) += 1;
            }
        });

        // Try pool first
        let pooled = BUFFER_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            if let Some(list) = p.get_mut(&size_bytes) {
                if let Some(buf) = list.pop() {
                    POOL_STATS.with(|s| s.borrow_mut().0 += 1);
                    return Some(buf);
                }
            }
            None
        });
        if let Some(buf) = pooled {
            return buf;
        }
        POOL_STATS.with(|s| s.borrow_mut().1 += 1);
        self.device
            .newBufferWithLength_options(
                size_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("Failed to allocate Metal buffer")
    }

    /// Return a buffer to the pool for reuse. Call when a buffer is no longer needed.
    pub fn recycle_buffer(buf: Retained<GpuBuffer>) {
        let size = buf.length();
        BUFFER_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            let list = p.entry(size).or_insert_with(Vec::new);
            // Cap pool size per bucket to avoid unbounded memory growth
            if list.len() < 32 {
                list.push(buf);
            }
        });
    }

    /// Get buffer pool statistics: (hits, misses)
    pub fn pool_stats() -> (usize, usize) {
        POOL_STATS.with(|s| *s.borrow())
    }

    /// Enable/disable allocation size logging for debugging.
    pub fn enable_alloc_log(enabled: bool) {
        ALLOC_SIZE_LOG.with(|log| {
            let mut l = log.borrow_mut();
            l.0 = enabled;
            l.1.clear();
        });
    }

    /// Dump allocation size log: prints each unique size and count, sorted by count desc.
    pub fn dump_alloc_log(label: &str) {
        ALLOC_SIZE_LOG.with(|log| {
            let l = log.borrow();
            let mut sizes: Vec<_> = l.1.iter().collect();
            sizes.sort_by(|a, b| b.1.cmp(a.1));
            let total: usize = sizes.iter().map(|(_, c)| **c).sum();
            eprintln!("[ALLOC LOG] {} — {} unique sizes, {} total allocs:", label, sizes.len(), total);
            for (size, count) in sizes.iter().take(20) {
                eprintln!("  {:>10} bytes × {:>4}", size, count);
            }
        });
    }

    /// Clear the buffer pool (e.g., between training runs to free memory)
    pub fn clear_pool() {
        BUFFER_POOL.with(|pool| pool.borrow_mut().clear());
        POOL_STATS.with(|s| *s.borrow_mut() = (0, 0));
    }

    /// Return total bytes cached in the buffer pool and the number of buffers.
    pub fn pool_memory_bytes() -> (usize, usize) {
        BUFFER_POOL.with(|pool| {
            let p = pool.borrow();
            let mut total_bytes = 0usize;
            let mut total_bufs = 0usize;
            for (size, bufs) in p.iter() {
                total_bytes += size * bufs.len();
                total_bufs += bufs.len();
            }
            (total_bytes, total_bufs)
        })
    }

    /// Allocate a buffer and initialize with float data.
    pub fn buffer_from_slice(&self, data: &[f32]) -> Retained<GpuBuffer> {
        let byte_len = std::mem::size_of_val(data);
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Failed to create buffer from slice")
        }
    }

    /// Write u32 data into an existing buffer (shared memory, zero-copy).
    /// Buffer must be at least data.len() * 4 bytes.
    pub fn write_u32_to_buffer(buf: &GpuBuffer, data: &[u32]) {
        use objc2_metal::MTLBuffer;
        assert!(buf.length() >= data.len() * 4);
        unsafe {
            let ptr = buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    /// Allocate a buffer and initialize with u32 data.
    pub fn buffer_from_u32_slice(&self, data: &[u32]) -> Retained<GpuBuffer> {
        let byte_len = std::mem::size_of_val(data);
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Failed to create buffer from u32 slice")
        }
    }

    /// Read float data back from a buffer.
    pub fn read_buffer(buf: &GpuBuffer, count: usize) -> Vec<f32> {
        // Auto-flush any active batch to ensure GPU data is committed
        Self::auto_flush_batch();
        let mut result = vec![0.0f32; count];
        unsafe {
            let ptr = buf.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
        }
        result
    }

    /// Read float data into a pre-allocated slice — zero allocation in the hot path.
    /// Returns the number of elements actually copied (min of count and slice length).
    pub fn read_buffer_into(buf: &GpuBuffer, dst: &mut [f32]) -> usize {
        Self::auto_flush_batch();
        let count = dst.len();
        unsafe {
            let ptr = buf.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), count);
        }
        count
    }

    /// Get a direct read-only pointer to buffer contents — true zero-copy.
    /// SAFETY: The returned slice is valid only while no GPU writes to this buffer.
    /// Caller must ensure the batch is flushed before calling.
    pub fn buffer_as_slice(buf: &GpuBuffer, count: usize) -> &[f32] {
        Self::auto_flush_batch();
        unsafe {
            let ptr = buf.contents().as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, count)
        }
    }

    /// Read u32 data back from a buffer.
    pub fn read_buffer_u32(buf: &GpuBuffer, count: usize) -> Vec<u32> {
        // Auto-flush any active batch to ensure GPU data is committed
        Self::auto_flush_batch();
        let mut result = vec![0u32; count];
        unsafe {
            let ptr = buf.contents().as_ptr() as *const u32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
        }
        result
    }

    /// Auto-flush the active batch if one exists, then restart it.
    /// Called before any GPU→CPU data read to ensure coherence.
    /// The batch is restarted so subsequent kernel dispatches continue batching.
    fn auto_flush_batch() {
        ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            if let Some(cb) = b.take() {
                if cb.encoder_count > 0 {
                    cb.cmd.commit();
                    cb.cmd.waitUntilCompleted();
                }
                // Note: batch is NOT restarted — we'd need &self (the queue) to create
                // a new command buffer. Since read_buffer is a static method, callers
                // must call begin_batch() again if they want to continue batching.
            }
        });
    }

    /// Get a pipeline by name, panics if not found.
    pub fn pipeline(&self, name: &str) -> &GpuPipeline {
        self.pipelines
            .get(name)
            .unwrap_or_else(|| panic!("Unknown pipeline: {}", name))
    }

    /// Helper to create an MTLSize.
    pub fn size(width: u64, height: u64, depth: u64) -> MTLSize {
        MTLSize {
            width: width as usize,
            height: height as usize,
            depth: depth as usize,
        }
    }

    /// Device name string.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Begin a command batch. All subsequent GPU kernel dispatches will be encoded
    /// into a single command buffer instead of individual commit+wait cycles.
    /// Call `flush_batch()` when you need results.
    pub fn begin_batch(&self) {
        ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            // If a batch is already active, flush it first (e.g., auto-flush consumed it partially)
            if let Some(cb) = b.take() {
                if cb.encoder_count > 0 {
                    cb.cmd.commit();
                    cb.cmd.waitUntilCompleted();
                }
            }
            let cmd = self.queue.commandBuffer().expect("Failed to create command buffer");
            *b = Some(CommandBatch { cmd, encoder_count: 0 });
        });
    }

    /// Flush the current command batch: commit and wait for GPU completion.
    /// Returns the number of kernels that were batched.
    pub fn flush_batch(&self) -> usize {
        ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            if let Some(cb) = b.take() {
                if cb.encoder_count > 0 {
                    cb.cmd.commit();
                    cb.cmd.waitUntilCompleted();
                }
                cb.encoder_count
            } else {
                0
            }
        })
    }

    /// Flush without waiting — commit the command buffer but don't block.
    /// The GPU runs in parallel with the CPU. Call flush_batch() or
    /// wait_batch() before reading any GPU buffers.
    pub fn flush_batch_async(&self) -> usize {
        ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            if let Some(cb) = b.take() {
                if cb.encoder_count > 0 {
                    cb.cmd.commit();
                    // Don't wait — GPU runs while CPU prepares next batch
                }
                cb.encoder_count
            } else {
                0
            }
        })
    }

    /// Wait for all submitted GPU work to complete.
    /// Call this before reading buffer contents.
    pub fn wait_gpu(&self) {
        // Submit and wait for any active batch
        self.flush_batch();
    }

    /// Check if a command batch is currently active.
    pub fn batch_active() -> bool {
        ACTIVE_BATCH.with(|batch| batch.borrow().is_some())
    }

    /// Encode a kernel dispatch into the active batch, or create a one-off sync dispatch.
    /// This is the core dispatch primitive used by all compute functions.
    /// When batching: encodes into the shared command buffer (no commit/wait).
    /// When not batching: creates a one-off command buffer, commits, waits (legacy path).
    pub fn dispatch_kernel(
        &self,
        pipeline_name: &str,
        grid: MTLSize,
        threadgroup: MTLSize,
        use_dispatch_threads: bool,
        bind: impl FnOnce(&GpuComputeEncoder),
    ) {
        let pipeline = self.pipelines.get(pipeline_name)
            .unwrap_or_else(|| panic!("Unknown pipeline: {}", pipeline_name));

        ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            if let Some(ref mut cb) = *b {
                // Batched path: encode into the shared command buffer
                let encoder = cb.cmd.computeCommandEncoder().expect("Failed to create encoder");
                encoder.setComputePipelineState(pipeline);
                bind(&encoder);
                if use_dispatch_threads {
                    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
                } else {
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
                }
                encoder.endEncoding();
                cb.encoder_count += 1;
            } else {
                // Unbatched path: one-off command buffer with sync wait
                let cmd = self.queue.commandBuffer().expect("Failed to create command buffer");
                let encoder = cmd.computeCommandEncoder().expect("Failed to create encoder");
                encoder.setComputePipelineState(pipeline);
                bind(&encoder);
                if use_dispatch_threads {
                    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
                } else {
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
                }
                encoder.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
        });
    }
}
