//! AndreOS GPU backend — direct hardware access via andreos-gpu crate.
//!
//! Zero framework overhead. CPU and GPU share unified memory.
//! Compute dispatch is 2 memory writes + 1 poll (vs Metal's 7+ ObjC messages).
//!
//! Depends on: andreos-gpu crate (../andreos/gpu or crates.io when published)

pub mod compute;

use std::sync::Arc;

// When building with andreos feature, import the real GPU crate.
// For now, we use the same interface but with placeholder types
// until Cargo.toml dependency is wired.

/// GPU buffer — raw pointer into unified memory.
pub struct GpuBuffer {
    pub ptr: *mut f32,
    pub size_bytes: usize,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    pub fn len_floats(&self) -> usize { self.size_bytes / 4 }
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len_floats()) }
    }
    pub fn contents(&self) -> std::ptr::NonNull<std::ffi::c_void> {
        unsafe { std::ptr::NonNull::new_unchecked(self.ptr as *mut std::ffi::c_void) }
    }
    pub fn length(&self) -> usize { self.size_bytes }
}

/// MetalContext-compatible wrapper around andreos-gpu::GpuContext.
///
/// Named MetalContext for source compatibility with tensor.rs, autograd.rs etc.
/// When andreos feature is active, this replaces the Metal implementation entirely.
pub struct MetalContext {
    /// Heap base + offset for bump allocation
    heap_base: *mut u8,
    heap_size: usize,
    heap_offset: std::sync::atomic::AtomicUsize,

    // When andreos-gpu is wired as a dependency:
    // pub gpu: andreos_gpu::GpuContext,
    //
    // The dispatch_kernel_direct method maps directly to gpu.dispatch().
    // alloc_buffer maps to gpu.alloc().
    // No command buffers, no encoders, no framework.
}

unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl MetalContext {
    pub fn new() -> Arc<Self> {
        // TODO: initialize from andreos-gpu::GpuContext::new()
        // let gpu = andreos_gpu::GpuContext::init();
        let heap_size = 8 * 1024 * 1024 * 1024; // 8GB
        Arc::new(Self {
            heap_base: std::ptr::null_mut(),
            heap_size,
            heap_offset: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn alloc_buffer(&self, size_bytes: usize) -> GpuBuffer {
        let aligned = (size_bytes + 255) & !255;
        let offset = self.heap_offset.fetch_add(aligned, std::sync::atomic::Ordering::Relaxed);
        assert!(offset + aligned <= self.heap_size, "GPU memory exhausted");
        let ptr = unsafe { self.heap_base.add(offset) as *mut f32 };
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size_bytes); }
        GpuBuffer { ptr, size_bytes }
    }

    pub fn buffer_from_slice(&self, data: &[f32]) -> GpuBuffer {
        let buf = self.alloc_buffer(data.len() * 4);
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf.ptr, data.len()); }
        buf
    }

    pub fn buffer_from_u32_slice(&self, data: &[u32]) -> GpuBuffer {
        let buf = self.alloc_buffer(data.len() * 4);
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const f32, buf.ptr, data.len()); }
        buf
    }

    pub fn read_buffer(buf: &GpuBuffer, count: usize) -> Vec<f32> {
        buf.as_slice()[..count].to_vec()
    }

    pub fn device_name(&self) -> String { "AndreOS GPU (AGX)".to_string() }
    pub fn recycle_buffer(_buf: GpuBuffer) {}
    pub fn pool_stats() -> (usize, usize) { (0, 0) }
    pub fn begin_batch(&self) {}
    pub fn flush_batch(&self) -> usize { 0 }
    pub fn flush_batch_async(&self) -> usize { 0 }
    pub fn batch_active() -> bool { false }
    pub fn size(x: u64, y: u64, z: u64) -> (u32, u32, u32) { (x as u32, y as u32, z as u32) }

    pub fn reset_heap(&self) {
        self.heap_offset.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Direct GPU dispatch — maps to andreos_gpu::GpuContext::dispatch().
    /// When andreos-gpu is wired: self.gpu.dispatch(kernel_id, grid, block, args, arg_sizes)
    /// 2 memory writes + 1 poll. Zero syscalls.
    pub fn dispatch_kernel_direct(
        &self,
        _kernel_id: u32,
        _grid: (u32, u32, u32),
        _block: (u32, u32, u32),
        _args: &[*const u8],
        _arg_sizes: &[usize],
    ) {
        // Wire to andreos-gpu:
        //   self.gpu.dispatch(kernel_id, grid, block, args, arg_sizes);
        //   self.gpu.wait_complete();
        //
        // The andreos-gpu dispatch() does:
        //   1. Write command descriptor to ring buffer
        //   2. Ring doorbell (MMIO write)
        //   3. Poll completion_stamp until GPU increments it
        //
        // Total: 2 writes + 1 poll = ~1μs per dispatch
        // vs Metal: 7+ ObjC messages = ~10-50μs per dispatch
    }
}
