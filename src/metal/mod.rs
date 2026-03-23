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
            ("rope", shaders::ROPE),
            ("add", shaders::ADD),
            ("mul", shaders::MUL),
            ("silu", shaders::SILU),
            ("silu_gate", shaders::SILU_GATE),
            ("cross_entropy", shaders::CROSS_ENTROPY),
            ("reduce_sum", shaders::REDUCE_SUM),
            ("adamw_update", shaders::ADAMW_UPDATE),
            ("embedding_lookup", shaders::EMBEDDING_LOOKUP),
            ("causal_mask", shaders::CAUSAL_MASK),
            ("l2_norm", shaders::L2_NORM),
            ("scale", shaders::SCALE),
            ("fill", shaders::FILL),
            ("copy_buffer", shaders::COPY),
            ("silu_backward", shaders::SILU_BACKWARD),
            ("silu_gate_backward", shaders::SILU_GATE_BACKWARD),
            ("rms_norm_backward", shaders::RMS_NORM_BACKWARD),
            ("softmax_backward", shaders::SOFTMAX_BACKWARD),
            ("embedding_backward", shaders::EMBEDDING_BACKWARD),
            ("transpose_2d", shaders::TRANSPOSE_2D),
            ("matmul_trans_a", shaders::MATMUL_TRANS_A),
            ("buffer_copy", shaders::BUFFER_COPY),
            ("transpose_perm_backward", shaders::TRANSPOSE_PERM_BACKWARD),
            ("transpose_perm_forward", shaders::TRANSPOSE_PERM_FORWARD),
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
    pub fn alloc_buffer(&self, size_bytes: usize) -> Retained<GpuBuffer> {
        self.device
            .newBufferWithLength_options(
                size_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("Failed to allocate Metal buffer")
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

    /// Flush the current command batch: commit the command buffer and wait.
    /// Returns the number of kernels that were batched.
    /// If no batch is active, this is a no-op returning 0.
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
