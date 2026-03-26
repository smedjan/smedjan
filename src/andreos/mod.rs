//! AndreOS GPU backend — direct hardware access, zero framework overhead.
//!
//! On AndreOS there is no Metal, no CUDA, no kernel. This module talks
//! directly to the GPU through memory-mapped registers and unified memory.
//!
//! Architecture:
//!   1. GPU command processor accessible via MMIO
//!   2. Unified memory — CPU and GPU share the same physical address space
//!   3. Compute kernels compiled to GPU ISA at build time (no runtime compilation)
//!   4. No command buffers, no encoders, no validation — raw dispatch
//!
//! This eliminates the 92% framework overhead measured on macOS Metal,
//! potentially pushing GPU utilization from 8% to 20-40% for small models.

pub mod compute;

use std::sync::Arc;

/// GPU buffer — just a raw pointer into unified memory.
/// On AndreOS, CPU and GPU share the same address space.
/// No allocation API, no refcounting, no Metal/CUDA wrapper.
pub struct GpuBuffer {
    pub ptr: *mut f32,
    pub size_bytes: usize,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    pub fn len_floats(&self) -> usize {
        self.size_bytes / 4
    }

    /// Direct read access (zero-copy on unified memory)
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len_floats()) }
    }

    /// Direct write access
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len_floats()) }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // Return to unified memory allocator
        unsafe {
            crate::andreos::gpu_free(self.ptr, self.size_bytes);
        }
    }
}

/// GPU context — minimal state for direct hardware access.
/// Named MetalContext for source compatibility.
pub struct MetalContext {
    /// GPU command processor base address (MMIO)
    cmd_base: *mut u8,
    /// Unified memory allocator state
    heap_base: *mut u8,
    heap_size: usize,
    heap_offset: std::sync::atomic::AtomicUsize,
}

unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

/// Allocate from unified memory (bump allocator — simple, fast, no fragmentation)
unsafe fn gpu_alloc(ctx: &MetalContext, size_bytes: usize) -> *mut f32 {
    // Align to 256 bytes (GPU cache line)
    let aligned = (size_bytes + 255) & !255;
    let offset = ctx.heap_offset.fetch_add(aligned, std::sync::atomic::Ordering::Relaxed);
    assert!(offset + aligned <= ctx.heap_size, "GPU unified memory exhausted");
    ctx.heap_base.add(offset) as *mut f32
}

/// Free unified memory (no-op for bump allocator; reset on clear)
unsafe fn gpu_free(_ptr: *mut f32, _size: usize) {
    // Bump allocator doesn't free individual allocations.
    // Call MetalContext::reset_heap() between training steps if needed.
}

impl MetalContext {
    /// Initialize direct GPU access.
    /// On AndreOS, this maps the GPU command processor and unified memory region.
    pub fn new() -> Arc<Self> {
        // These addresses come from AndreOS device tree / hardware discovery
        // TODO: wire to actual AndreOS GPU driver (andreos-gpu crate)
        let cmd_base = std::ptr::null_mut(); // placeholder
        let heap_size = 8 * 1024 * 1024 * 1024; // 8GB unified memory
        let heap_base = std::ptr::null_mut(); // placeholder — will be mmap'd region

        Arc::new(Self {
            cmd_base,
            heap_base,
            heap_size,
            heap_offset: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Allocate a GPU buffer from unified memory.
    pub fn alloc_buffer(&self, size_bytes: usize) -> GpuBuffer {
        let ptr = unsafe { gpu_alloc(self, size_bytes) };
        // Zero-initialize
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size_bytes); }
        GpuBuffer { ptr, size_bytes }
    }

    /// Create buffer from f32 slice (memcpy — same address space)
    pub fn buffer_from_slice(&self, data: &[f32]) -> GpuBuffer {
        let size_bytes = data.len() * 4;
        let buf = self.alloc_buffer(size_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.ptr, data.len());
        }
        buf
    }

    /// Read buffer contents (zero-copy — just cast the pointer)
    pub fn read_buffer(buf: &GpuBuffer, count: usize) -> Vec<f32> {
        buf.as_slice()[..count].to_vec()
    }

    pub fn device_name(&self) -> String {
        "AndreOS Direct GPU".to_string()
    }

    /// Reset heap allocator (call between training runs to reclaim memory)
    pub fn reset_heap(&self) {
        self.heap_offset.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn recycle_buffer(_buf: GpuBuffer) {}
    pub fn pool_stats() -> (usize, usize) { (0, 0) }

    // No command batching needed — dispatches go directly to hardware
    pub fn begin_batch(&self) {}
    pub fn flush_batch(&self) -> usize { 0 }
    pub fn flush_batch_async(&self) -> usize { 0 }
    pub fn batch_active() -> bool { false }

    pub fn size(x: u64, y: u64, z: u64) -> (u32, u32, u32) {
        (x as u32, y as u32, z as u32)
    }

    /// Submit a compute kernel directly to the GPU command processor.
    /// On Metal: encode → commit → wait (3 API calls + validation)
    /// On AndreOS: write command descriptor → doorbell → poll completion
    pub fn dispatch_kernel_direct(
        &self,
        _kernel_id: u32,      // pre-compiled kernel index
        _grid: (u32, u32, u32),
        _block: (u32, u32, u32),
        _args: &[*const u8],  // raw buffer pointers
        _arg_sizes: &[usize],
    ) {
        // TODO: wire to AndreOS GPU command processor
        // This is where the magic happens — one register write to submit work
        //
        // Pseudocode for Apple Silicon GPU:
        //   let cmd = CommandDescriptor {
        //       kernel_addr: self.kernel_table[kernel_id],
        //       grid_dim: grid,
        //       block_dim: block,
        //       arg_buffer: args_packed,
        //       completion_signal: &self.completion_flag,
        //   };
        //   write_volatile(self.cmd_base as *mut CommandDescriptor, cmd);
        //   write_volatile(self.doorbell_addr, 1); // ring doorbell
        //   while read_volatile(&self.completion_flag) == 0 {} // poll
        //
        // This replaces:
        //   Metal: MTLCommandBuffer + MTLComputeCommandEncoder + setBuffer × N +
        //          dispatchThreadgroups + endEncoding + commit + waitUntilCompleted
        //   (7+ Objective-C message sends with runtime dispatch)
        //
        //   CUDA: cuLaunchKernel + cuCtxSynchronize
        //   (2 driver API calls with kernel-mode transitions)
        //
        //   AndreOS: 2 memory writes + 1 poll loop (pure userspace, zero syscalls)
    }
}
