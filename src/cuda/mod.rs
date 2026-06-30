//! CUDA GPU backend for Smedjan.
//!
//! Provides the same public API as the Metal backend:
//! - `CudaContext` (aliased as `MetalContext` for compatibility)
//! - `GpuBuffer` type
//! - `compute` module with gpu_* dispatch functions
//!
//! Uses `cudarc` for CUDA driver API bindings.

pub mod compute;
pub mod kernels;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Global free list of recycled device buffers, keyed by `(device instance, element count)`.
///
/// Why this is aliasing-safe without Metal's generation/quarantine machinery: each `CudaDevice`
/// issues our kernels in order on its own single stream. A buffer is returned here only once its
/// last Rust-side reference drops (`Arc::strong_count == 1`), so nothing else holds it; and because
/// every reissue's zeroing memset and every subsequent kernel are queued on that same stream *after*
/// whatever last read the buffer, the GPU can't reorder a stale read past the reuse. That ordering
/// is exactly what Metal's deferred command-buffer model lacks, which is why this needs no quarantine.
///
/// The device half of the key is essential: a process can hold several `CudaDevice` instances (every
/// `test_ctx()` makes a fresh one, each with its *own* stream), and reusing a buffer recycled under
/// one stream on a *different* stream would race the zeroing memset against the first stream's still
/// pending work. Keying by device confines reuse to a single stream — full reuse for the one context
/// a real run holds, zero cross-context reuse in the test suite.
/// Pool type alias to avoid clippy::type_complexity on the two sites that hold it.
type BufferPool = Mutex<HashMap<(usize, usize), Vec<CudaSlice<f32>>>>;

static BUFFER_POOL: OnceLock<BufferPool> = OnceLock::new();
/// (hits, misses) since process start — surfaced by `pool_stats()` for the training/bench readout.
static POOL_HITS: AtomicUsize = AtomicUsize::new(0);
static POOL_MISSES: AtomicUsize = AtomicUsize::new(0);
/// >0 while any `PoolBypassGuard` is live (checkpoint recompute / grad-accum): allocations come
/// > fresh and recycles are dropped, mirroring the Metal bypass. Process-global is sound because
/// > the GPU compute path is single-threaded.
static POOL_BYPASS: AtomicUsize = AtomicUsize::new(0);
/// Cap per size bucket, mirroring Metal, so idle VRAM can't grow unbounded across varied seq lengths.
const POOL_BUCKET_CAP: usize = 64;

fn buffer_pool() -> &'static BufferPool {
    BUFFER_POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Stable per-instance id for a `CudaDevice` (its heap address). Distinguishes contexts that share
/// GPU ordinal 0 but own different streams, so the pool never reuses a buffer across streams.
#[inline]
fn device_id(dev: &Arc<CudaDevice>) -> usize {
    Arc::as_ptr(dev) as usize
}

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
        f[i] = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    buf.device().bind_to_thread().expect("bind CUDA context");
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(*buf.device_ptr(), &f)
            .expect("htod buf_write_bytes");
    }
}

/// Read a slice of bytes from a device buffer at the given offset (dtoh). On Metal shared-memory
/// buffers this is a direct CPU memcpy; on CUDA the buffer lives in device memory, so we copy the
/// requested byte range back to the host synchronously. Used by `Q4Matmul::gather_row` to pull one
/// int4 weight row + its group scales/biases back to the CPU for dequantization (a single row of
/// d=4096 is ~512 U32s + 128 groups — small enough that the dtoh is cheaper than a GPU kernel).
pub fn buf_read_bytes(buf: &Buf, offset: usize, length: usize) -> Vec<u8> {
    // The buffer is a CudaSlice<f32> (4-byte elements). `offset` and `length` are in bytes, and
    // may not be 4-byte aligned (e.g. BF16 scales use 2-byte stride). We copy the full enclosing
    // f32 elements, then slice the byte range out of the host Vec.
    let elem_offset = offset / 4;
    let elem_end = (offset + length).div_ceil(4); // ceil to include the partial tail element
    let n_elems = elem_end - elem_offset;
    // dtoh_sync_copy_into copies from the START of the given CudaSlice; to read a sub-range we
    // need to offset the device pointer. Use the raw memcpy with a byte-offset device pointer.
    let base_ptr = *buf.device_ptr();
    let byte_offset = elem_offset * 4;
    let mut host = vec![0.0f32; n_elems];
    buf.device().bind_to_thread().expect("bind CUDA context");
    unsafe {
        // CUdeviceptr is an integer (u64) device address; add the byte offset to advance it.
        // memcpy_dtoh_sync takes (dst: &mut [T], src: CUdeviceptr).
        let offset_ptr = base_ptr + byte_offset as cudarc::driver::sys::CUdeviceptr;
        cudarc::driver::result::memcpy_dtoh_sync(&mut host, offset_ptr)
            .expect("dtoh buf_read_bytes");
    }
    let bytes: Vec<u8> = host.iter().flat_map(|f| f.to_le_bytes()).collect();
    let byte_offset_in_slice = offset - elem_offset * 4;
    bytes[byte_offset_in_slice..byte_offset_in_slice + length].to_vec()
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
    /// cuBLAS handle with TF32 tensor-core math — the fast path (training/inference forward).
    pub cublas_fast: cudarc::cublas::CudaBlas,
    /// cuBLAS handle with strict fp32 math — the precise path (gradchecks, where TF32's ~1e-3
    /// mantissa would make finite-difference checks fail just like the old fp16 kernels did).
    pub cublas_precise: cudarc::cublas::CudaBlas,
}

unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl MetalContext {
    pub fn new() -> Arc<Self> {
        let device = CudaDevice::new(0).expect("Failed to initialize CUDA device");

        // Compile all kernels to PTX. nvrtc needs the toolkit include dir for <cuda_fp16.h> etc.;
        // derive it from CUDA_PATH (set at build/run time) with a sane fallback.
        let cuda_inc =
            std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".into()) + "/include";
        let opts = cudarc::nvrtc::CompileOptions {
            include_paths: vec![cuda_inc],
            // line info lets compute-sanitizer name the faulting kernel + source line.
            options: vec!["--generate-line-info".to_string()],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(kernels::ALL_KERNELS, opts)
            .expect("Failed to compile CUDA kernels");
        device
            .load_ptx(ptx, "smedjan", kernels::KERNEL_NAMES)
            .expect("Failed to load CUDA kernels");

        // cuBLAS GEMM uses the GPU's tensor cores — orders of magnitude faster than the hand-rolled
        // tiled kernels. Two handles: TF32 tensor-core math for speed, strict fp32 for precision.
        let cublas_fast =
            cudarc::cublas::CudaBlas::new(device.clone()).expect("Failed to init cuBLAS (fast)");
        unsafe {
            cudarc::cublas::sys::lib().cublasSetMathMode(
                *cublas_fast.handle(),
                cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
            );
        }
        let cublas_precise =
            cudarc::cublas::CudaBlas::new(device.clone()).expect("Failed to init cuBLAS (precise)");

        Arc::new(Self {
            device,
            cublas_fast,
            cublas_precise,
        })
    }

    /// Allocate a GPU buffer (device memory). Checks the recycle pool first (unless bypassed),
    /// falling back to a fresh `cudaMalloc`. Returns zeroed memory either way, preserving the
    /// long-standing contract — the win is replacing a per-op cudaMalloc syscall (driver lock +
    /// page setup) with a same-stream memset of an already-mapped allocation in the steady state.
    pub fn alloc_buffer(&self, size_bytes: usize) -> Buf {
        let n_floats = size_bytes.div_ceil(4); // round up: byte-sized (int8) allocs must not under-provision
        // Pop a recycled same-device, same-size buffer under the lock; release it before the GPU op.
        let reused = if POOL_BYPASS.load(Ordering::Relaxed) == 0 {
            let key = (device_id(&self.device), n_floats);
            let mut pool = buffer_pool().lock().unwrap();
            pool.get_mut(&key).and_then(|list| list.pop())
        } else {
            None
        };
        let buf = if let Some(mut slice) = reused {
            self.device
                .memset_zeros(&mut slice)
                .expect("Failed to zero recycled CUDA buffer");
            POOL_HITS.fetch_add(1, Ordering::Relaxed);
            Arc::new(slice)
        } else {
            POOL_MISSES.fetch_add(1, Ordering::Relaxed);
            Arc::new(
                self.device
                    .alloc_zeros::<f32>(n_floats)
                    .expect("Failed to allocate CUDA buffer"),
            )
        };
        // This device address is about to be written fresh. A previous tensor that lived here may
        // still have an fp16/ternary conversion cached under this same pointer (those caches clear
        // only after the optimizer step), and the pool deliberately reissues the same address — so
        // without this a later `cast_to_f16` could return the previous tensor's data. Mirrors Metal.
        crate::tensor::Tensor::invalidate_conversion_cache(buf_addr(&buf));
        buf
    }

    /// Create buffer from f32 slice (host→device copy).
    pub fn buffer_from_slice(&self, data: &[f32]) -> Buf {
        let buf = Arc::new(
            self.device
                .htod_sync_copy(data)
                .expect("Failed to copy to device"),
        );
        // Same staleness hazard as alloc_buffer: this fresh allocation may land on the address of a
        // just-freed buffer whose fp16/ternary conversion is still cached, so drop any stale entry.
        // (Metal's harness caught this as a perturbed from_slice input silently read through a cached
        // fp16 cast → numeric gradient of exactly 0.)
        crate::tensor::Tensor::invalidate_conversion_cache(buf_addr(&buf));
        buf
    }

    /// Create buffer from u32 slice.
    pub fn buffer_from_u32_slice(&self, data: &[u32]) -> BufU32 {
        Arc::new(
            self.device
                .htod_sync_copy(data)
                .expect("Failed to copy u32 to device"),
        )
    }

    /// Allocate a zeroed u32 device buffer of `count` elements.
    pub fn alloc_buffer_u32(&self, count: usize) -> BufU32 {
        Arc::new(
            self.device
                .alloc_zeros::<u32>(count)
                .expect("Failed to allocate u32 CUDA buffer"),
        )
    }

    /// Read f32 data from device buffer.
    pub fn read_buffer(buf: &CudaSlice<f32>, count: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; count];
        buf.device()
            .dtoh_sync_copy_into(buf, &mut result)
            .expect("Failed to copy from device");
        result
    }

    /// Read u32 data from device buffer.
    pub fn read_buffer_u32(buf: &CudaSlice<u32>, count: usize) -> Vec<u32> {
        let mut result = vec![0u32; count];
        buf.device()
            .dtoh_sync_copy_into(buf, &mut result)
            .expect("Failed to copy u32 from device");
        result
    }

    /// Overwrite an existing u32 device buffer with host data (htod into the live allocation).
    /// Mirrors Metal's unified-memory in-place write; here a low-level synchronous htod memcpy.
    pub fn write_u32_to_buffer(buf: &CudaSlice<u32>, data: &[u32]) {
        assert!(
            buf.len() >= data.len(),
            "write_u32_to_buffer: dst {} < src {} elems",
            buf.len(),
            data.len()
        );
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

    /// Return a buffer to the recycle pool for same-size reuse. Only pools buffers with no other
    /// live reference (`Arc::strong_count == 1`) — reissuing a still-shared buffer would alias a
    /// tensor still in use. Clone-recycle sites (which Metal's deferred model tolerates) carry an
    /// extra refcount and so fall through to a plain `Arc` drop (`cudaFree`) here, which is correct.
    pub fn recycle_buffer(buf: Buf) {
        if POOL_BYPASS.load(Ordering::Relaxed) > 0 {
            return; // bypassed region: drop instead of pooling
        }
        if let Ok(slice) = Arc::try_unwrap(buf) {
            // Key by the buffer's *own* device so it can only ever be reissued on the stream that
            // last touched it (see BUFFER_POOL docs) — `slice.device()` is that owning device.
            let key = (device_id(&slice.device()), slice.len());
            let mut pool = buffer_pool().lock().unwrap();
            let list = pool.entry(key).or_default();
            if list.len() < POOL_BUCKET_CAP {
                list.push(slice);
            }
            // bucket full → drop (cudaFree) to bound idle VRAM
        }
    }

    /// Pool statistics: (hits, misses) since process start.
    pub fn pool_stats() -> (usize, usize) {
        (
            POOL_HITS.load(Ordering::Relaxed),
            POOL_MISSES.load(Ordering::Relaxed),
        )
    }

    /// Drop every pooled buffer and reset stats (e.g. between training runs to free VRAM).
    pub fn clear_pool() {
        buffer_pool().lock().unwrap().clear();
        POOL_HITS.store(0, Ordering::Relaxed);
        POOL_MISSES.store(0, Ordering::Relaxed);
    }

    // Alloc-logging is a Metal buffer-pool diagnostic; no-op on CUDA.
    pub fn enable_alloc_log(_on: bool) {}
    pub fn dump_alloc_log(_label: &str) {}

    /// Read a device buffer into a host slice (dtoh). Mirrors Metal's read_buffer_into.
    pub fn read_buffer_into(buf: &CudaSlice<f32>, dst: &mut [f32]) {
        buf.device()
            .dtoh_sync_copy_into(buf, dst)
            .expect("dtoh_sync_copy_into");
    }

    /// Block until all queued GPU work finishes. CUDA copies here are already synchronous; sync anyway.
    pub fn wait_gpu(&self) {
        self.device.synchronize().ok();
    }

    // Command batching — CUDA uses streams, no explicit batching needed.
    // These are no-ops for API compatibility with Metal backend.
    pub fn begin_batch(&self) {}
    pub fn flush_batch(&self) -> usize {
        0
    }
    pub fn flush_batch_async(&self) -> usize {
        0
    }
    pub fn batch_active() -> bool {
        false
    }

    // Auto-flush is a no-op on CUDA
    fn auto_flush_batch() {}
}

/// RAII guard that suspends buffer pooling for its lifetime. While any guard is live, `alloc_buffer`
/// allocates fresh and `recycle_buffer` drops instead of pooling. Used around checkpoint recompute
/// and grad-accum, where reissuing a pooled buffer the outer backward still references would corrupt
/// gradients. Mirrors the Metal guard so shared code compiles and behaves identically.
pub struct PoolBypassGuard;
impl PoolBypassGuard {
    pub fn new() -> Self {
        POOL_BYPASS.fetch_add(1, Ordering::Relaxed);
        PoolBypassGuard
    }
}
impl Default for PoolBypassGuard {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for PoolBypassGuard {
    fn drop(&mut self) {
        POOL_BYPASS.fetch_sub(1, Ordering::Relaxed);
    }
}
