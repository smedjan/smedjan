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

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::driver::sys::CUdeviceptr;
use std::collections::HashMap;
use std::sync::Arc;

/// GPU buffer type for CUDA — wraps a device memory allocation.
pub type GpuBuffer = CudaSlice<f32>;

/// Backend-agnostic buffer handle. Arc so `.clone()` is a cheap refcount share (mirrors Metal's
/// `Retained<MTLBuffer>`); shared code stores tensors/tape buffers as `crate::gpu::Buf`.
pub type Buf = Arc<CudaSlice<f32>>;

/// Device-pointer address as usize — for pool/dedup/cache keys (mirrors Metal's `buf_addr`).
#[inline]
pub fn buf_addr(b: &Buf) -> usize {
    use cudarc::driver::DevicePtr;
    *b.device_ptr() as usize
}

/// u32 index/token buffer handle (tokens, targets, seg_ids). Metal stores these in the same untyped
/// GpuBuffer; CUDA needs a distinct typed slice.
pub type BufU32 = Arc<CudaSlice<u32>>;

/// Byte length of a buffer (element count × 4 bytes).
#[inline]
pub fn buf_len_bytes(b: &Buf) -> usize {
    (**b).len() * 4
}

/// Write host bytes into a device buffer (htod). Mirrors Metal's unified-memory write; bytes are
/// reinterpreted as f32 then synchronously copied into the live device allocation.
#[inline]
pub fn buf_write_bytes(buf: &Buf, bytes: &[u8]) {
    let n = bytes.len() / 4;
    let mut f = vec![0f32; n];
    for i in 0..n {
        f[i] = f32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]);
    }
    buf.device().bind_to_thread().expect("bind CUDA context");
    unsafe { cudarc::driver::result::memcpy_htod_sync(*buf.device_ptr(), &f).expect("htod buf_write_bytes"); }
}

/// Reinterpret a u32 buffer handle as a Buf (f32) for storage in the untyped tape Vec<Buf>.
/// CudaSlice<u32> and CudaSlice<f32> have identical layout (device ptr + len, both 4-byte elems);
/// the tape only holds it to hand back to embedding_backward, which reinterprets it back as u32.
pub fn u32_to_buf(b: BufU32) -> Buf {
    unsafe { std::mem::transmute::<Arc<CudaSlice<u32>>, Arc<CudaSlice<f32>>>(b) }
}

/// Inverse of `u32_to_buf`: view a tape-stored Buf as the u32 buffer it really is.
pub fn buf_as_u32(b: &Buf) -> BufU32 {
    unsafe { std::mem::transmute::<Arc<CudaSlice<f32>>, Arc<CudaSlice<u32>>>(b.clone()) }
}

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

        // Compile all kernels to PTX. nvrtc needs the toolkit include dir for <cuda_fp16.h> etc.;
        // derive it from CUDA_PATH (set at build/run time) with a sane fallback.
        let cuda_inc = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".into()) + "/include";
        let opts = cudarc::nvrtc::CompileOptions {
            include_paths: vec![cuda_inc],
            // line info lets compute-sanitizer name the faulting kernel + source line.
            options: vec!["--generate-line-info".to_string()],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(kernels::ALL_KERNELS, opts)
            .expect("Failed to compile CUDA kernels");
        device.load_ptx(ptx, "andreai", &kernels::KERNEL_NAMES)
            .expect("Failed to load CUDA kernels");

        Arc::new(Self { device })
    }

    /// Allocate a GPU buffer (device memory).
    pub fn alloc_buffer(&self, size_bytes: usize) -> Buf {
        let n_floats = size_bytes.div_ceil(4); // round up: byte-sized (int8) allocs must not under-provision
        Arc::new(self.device.alloc_zeros::<f32>(n_floats)
            .expect("Failed to allocate CUDA buffer"))
    }

    /// Create buffer from f32 slice (host→device copy).
    pub fn buffer_from_slice(&self, data: &[f32]) -> Buf {
        Arc::new(self.device.htod_sync_copy(data)
            .expect("Failed to copy to device"))
    }

    /// Create buffer from u32 slice.
    pub fn buffer_from_u32_slice(&self, data: &[u32]) -> BufU32 {
        Arc::new(self.device.htod_sync_copy(data)
            .expect("Failed to copy u32 to device"))
    }

    /// Allocate a zeroed u32 device buffer of `count` elements.
    pub fn alloc_buffer_u32(&self, count: usize) -> BufU32 {
        Arc::new(self.device.alloc_zeros::<u32>(count).expect("Failed to allocate u32 CUDA buffer"))
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

    /// Overwrite an existing u32 device buffer with host data (htod into the live allocation).
    /// Mirrors Metal's unified-memory in-place write; here a low-level synchronous htod memcpy.
    pub fn write_u32_to_buffer(buf: &CudaSlice<u32>, data: &[u32]) {
        assert!(buf.len() >= data.len(),
            "write_u32_to_buffer: dst {} < src {} elems", buf.len(), data.len());
        // Raw driver memcpy requires the device's primary context current on this thread;
        // the safe htod_* wrappers bind it, result::* does not. Bind before copying.
        buf.device().bind_to_thread().expect("bind CUDA context");
        unsafe {
            cudarc::driver::result::memcpy_htod_sync(*buf.device_ptr(), data)
                .expect("Failed to write u32 to device buffer");
        }
    }

    /// Return device name.
    pub fn device_name(&self) -> String {
        format!("CUDA GPU {}", self.device.ordinal())
    }

    /// Create a size tuple for CUDA grid/block dimensions.
    pub fn size(x: u64, y: u64, z: u64) -> (u32, u32, u32) {
        (x as u32, y as u32, z as u32)
    }

    /// Recycle buffer (no-op for CUDA — Arc drop handles deallocation).
    pub fn recycle_buffer(_buf: Buf) {
        // Drop handles deallocation
    }

    /// Pool stats (placeholder — CUDA uses cudarc's allocator).
    pub fn pool_stats() -> (usize, usize) {
        (0, 0)
    }

    // Alloc-logging is a Metal buffer-pool diagnostic; no-op on CUDA.
    pub fn enable_alloc_log(_on: bool) {}
    pub fn dump_alloc_log(_label: &str) {}

    /// Read a device buffer into a host slice (dtoh). Mirrors Metal's read_buffer_into.
    pub fn read_buffer_into(buf: &CudaSlice<f32>, dst: &mut [f32]) {
        buf.device().dtoh_sync_copy_into(buf, dst).expect("dtoh_sync_copy_into");
    }

    /// Block until all queued GPU work finishes. CUDA copies here are already synchronous; sync anyway.
    pub fn wait_gpu(&self) {
        self.device.synchronize().ok();
    }

    /// Metal returns a zero-copy host view of unified memory; CUDA device memory has no host slice.
    /// Callers on CUDA must use `to_vec()` / `read_buffer` instead. Stubbed to keep the API surface.
    pub fn buffer_as_slice(_buf: &CudaSlice<f32>, _n: usize) -> &'static [f32] {
        unimplemented!("cuda: buffer_as_slice has no zero-copy host view — use to_vec()/read_buffer")
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

/// No-op on CUDA: there is no thread-local buffer pool to bypass (cudarc owns allocation).
/// Mirrors metal's RAII PoolBypassGuard so shared checkpoint-recompute code compiles unchanged.
pub struct PoolBypassGuard;
impl PoolBypassGuard {
    pub fn new() -> Self { PoolBypassGuard }
}
impl Default for PoolBypassGuard {
    fn default() -> Self { Self::new() }
}
