//! CUDA GPU backend for AndreAI.
//!
//! Provides the same public API as the Metal backend:
//! - `CudaContext` (aliased as `MetalContext` for compatibility)
//! - `GpuBuffer` type
//! - `compute` module with gpu_* dispatch functions
//!
//! Uses `cudarc` for CUDA driver API bindings.

pub mod compute;
pub mod kernels;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::driver::sys::CUdeviceptr;
use std::collections::HashMap;
use std::sync::Arc;

/// GPU buffer type for CUDA — wraps a device memory allocation.
pub type GpuBuffer = CudaSlice<f32>;

/// CUDA context: device handle, pre-compiled kernels, buffer pool.
/// Named MetalContext for source compatibility with the rest of the codebase.
pub struct MetalContext {
    pub device: Arc<CudaDevice>,
}

unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl MetalContext {
    pub fn new() -> Arc<Self> {
        let device = CudaDevice::new(0).expect("Failed to initialize CUDA device");

        // Load all PTX kernels
        let ptx = cudarc::nvrtc::compile_ptx(kernels::ALL_KERNELS)
            .expect("Failed to compile CUDA kernels");
        device.load_ptx(ptx, "andreai", &kernels::KERNEL_NAMES)
            .expect("Failed to load CUDA kernels");

        Arc::new(Self { device })
    }

    /// Allocate a GPU buffer (device memory).
    pub fn alloc_buffer(&self, size_bytes: usize) -> CudaSlice<f32> {
        let n_floats = size_bytes / 4;
        self.device.alloc_zeros::<f32>(n_floats)
            .expect("Failed to allocate CUDA buffer")
    }

    /// Create buffer from f32 slice (host→device copy).
    pub fn buffer_from_slice(&self, data: &[f32]) -> CudaSlice<f32> {
        self.device.htod_sync_copy(data)
            .expect("Failed to copy to device")
    }

    /// Create buffer from u32 slice.
    pub fn buffer_from_u32_slice(&self, data: &[u32]) -> CudaSlice<u32> {
        self.device.htod_sync_copy(data)
            .expect("Failed to copy u32 to device")
    }

    /// Read f32 data from device buffer.
    pub fn read_buffer(buf: &CudaSlice<f32>, count: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; count];
        buf.device().dtoh_sync_copy_into(buf, &mut result)
            .expect("Failed to copy from device");
        result
    }

    /// Read u32 data from device buffer.
    pub fn read_buffer_u32(buf: &CudaSlice<u32>, count: usize) -> Vec<u32> {
        let mut result = vec![0u32; count];
        buf.device().dtoh_sync_copy_into(buf, &mut result)
            .expect("Failed to copy u32 from device");
        result
    }

    /// Return device name.
    pub fn device_name(&self) -> String {
        format!("CUDA GPU {}", self.device.ordinal())
    }

    /// Create a size tuple for CUDA grid/block dimensions.
    pub fn size(x: u64, y: u64, z: u64) -> (u32, u32, u32) {
        (x as u32, y as u32, z as u32)
    }

    /// Recycle buffer (no-op for CUDA — cudarc handles deallocation).
    pub fn recycle_buffer(_buf: CudaSlice<f32>) {
        // Drop handles deallocation
    }

    /// Pool stats (placeholder — CUDA uses cudarc's allocator).
    pub fn pool_stats() -> (usize, usize) {
        (0, 0)
    }

    // Command batching — CUDA uses streams, no explicit batching needed.
    // These are no-ops for API compatibility with Metal backend.
    pub fn begin_batch(&self) {}
    pub fn flush_batch(&self) -> usize { 0 }
    pub fn flush_batch_async(&self) -> usize { 0 }
    pub fn batch_active() -> bool { false }

    // Auto-flush is a no-op on CUDA
    fn auto_flush_batch() {}
}
