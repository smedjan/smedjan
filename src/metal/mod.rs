pub mod compute;
pub mod shaders;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::cell::{Cell, RefCell};
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
    // When > 0, alloc_buffer skips the pool (fresh Metal alloc) and recycle_buffer is a no-op.
    // Set during gradient-checkpoint recompute so its buffers can't alias the outer backward's
    // still-referenced pooled buffers.
    static POOL_BYPASS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(feature = "bufsan")]
thread_local! {
    /// Free-pool buffer addresses already NaN-poisoned since their last recycle. Avoids rewriting
    /// the entire unchanged pool after every synchronous dispatch in sanitizer builds.
    static POISONED_POOL_ADDRS: RefCell<std::collections::HashSet<usize>> = RefCell::new(std::collections::HashSet::new());
}

// ---------------------------------------------------------------------------
// Buffer-pool sanitizer (feature = "bufsan"). Compiles to nothing when off.
//
// The hazard (see docs/HANDOFF_buffer_hazard_and_followups.md §1): the pool reissues a
// buffer the instant it is recycled, but dispatch_kernel only ENCODES into the active
// command batch — nothing runs until flush. A buffer recycled-then-reissued within one
// uncommitted batch gets a new owner whose write clobbers data the old owner still needs.
//
// Two instruments, both keyed off a per-flush generation counter:
//   * POISON: a buffer recycled in generation G is only provably dead once G has been
//     flushed (commit+wait). So we NaN-fill pooled buffers at FLUSH time (never at recycle
//     time — that would corrupt reads still legitimately pending in the current batch). A
//     later read-before-overwrite (use-after-recycle) or a too-small dispatch then surfaces
//     as NaN instead of silently-wrong data.
//   * QUARANTINE (opt-in): alloc may only reissue a buffer whose recycle-generation is
//     strictly less than the current generation, i.e. recycled in an already-flushed batch.
//     This makes intra-batch recycle→reissue aliasing impossible by construction — the exact
//     loss-readout-class failure. Used as a differential: a correct run is unchanged by it.
// ---------------------------------------------------------------------------
thread_local! {
    /// Bumped on every WAITING flush (commit+wait). Marks command-batch boundaries so the pool
    /// can distinguish "recycled in an already-completed batch" (safe to reissue) from "recycled
    /// in the current uncommitted batch" (reissue would alias a dispatch the GPU hasn't run yet).
    static BATCH_GENERATION: Cell<u64> = const { Cell::new(0) };
    /// buffer address → generation at which it was recycled.
    static RECYCLE_GEN: RefCell<HashMap<usize, u64>> = RefCell::new(HashMap::new());
    /// Pool quarantine — DEFAULT ON. Alloc reissues only buffers recycled in an already-flushed
    /// generation, making intra-batch recycle→reissue aliasing impossible by construction. This is
    /// the fix for silent gradient corruption that surfaced as loss divergence at seq_len ≥ 256
    /// (the pooled attention-backward buffers aliased a still-pending dispatch). Toggleable so the
    /// `bufsan` differential test can compare against the old unsafe behaviour; production keeps it
    /// on. `SMEDJAN_NO_POOL` (full bypass) remains as the even-stronger diagnostic.
    static QUARANTINE: Cell<bool> = const { Cell::new(true) };
}

#[inline]
fn buf_contents_addr(buf: &GpuBuffer) -> usize {
    use objc2_metal::MTLBuffer;
    buf.contents().as_ptr() as usize
}

/// Called at the end of every WAITING flush (commit+wait). Bumps the command-batch generation so
/// quarantine can release buffers recycled in now-completed batches. Under `bufsan` it also
/// NaN-poisons the (now provably-dead) pooled buffers so a use-after-recycle surfaces as NaN.
fn on_flush() {
    BATCH_GENERATION.with(|g| g.set(g.get() + 1));
    #[cfg(feature = "bufsan")]
    bufsan_poison_pool();
}

#[cfg(feature = "bufsan")]
fn bufsan_poison_buffer(buf: &GpuBuffer) {
    use objc2_metal::MTLBuffer;
    let nan = f32::from_bits(0x7FC0_0000);
    let n = buf.length() / 4;
    let ptr = buf.contents().as_ptr() as *mut f32;
    // SAFETY: shared-storage buffer; caller only invokes this for free-pool buffers or freshly
    // allocated/reissued buffers before handing them to a new owner.
    unsafe {
        for i in 0..n {
            *ptr.add(i) = nan;
        }
    }
}

#[cfg(feature = "bufsan")]
fn bufsan_poison_pool() {
    BUFFER_POOL.with(|pool| {
        let p = pool.borrow();
        POISONED_POOL_ADDRS.with(|poisoned| {
            let mut poisoned = poisoned.borrow_mut();
            for bufs in p.values() {
                for b in bufs {
                    let addr = buf_contents_addr(b);
                    if poisoned.insert(addr) {
                        bufsan_poison_buffer(b);
                    }
                }
            }
        });
    });
}

/// RAII guard that bypasses the buffer pool while alive (used during checkpoint recompute).
pub struct PoolBypassGuard;

impl PoolBypassGuard {
    pub fn new() -> Self {
        POOL_BYPASS.with(|b| b.set(b.get() + 1));
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
        POOL_BYPASS.with(|b| b.set(b.get() - 1));
    }
}

// Link CoreGraphics — required for MTLCreateSystemDefaultDevice
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Type aliases for the objc2-metal protocol objects.
pub type GpuDevice = ProtocolObject<dyn MTLDevice>;
pub type GpuQueue = ProtocolObject<dyn MTLCommandQueue>;
pub type GpuBuffer = ProtocolObject<dyn MTLBuffer>;
pub type GpuPipeline = ProtocolObject<dyn MTLComputePipelineState>;
pub type GpuCommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
pub type GpuComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// Backend-agnostic buffer handle. Metal: a refcounted `Retained<MTLBuffer>` (cheap `.clone()` = share).
/// Shared code uses `crate::gpu::Buf` instead of naming objc2 types directly, so CUDA can substitute
/// `Arc<CudaSlice<f32>>` (also cheap-clone) behind the same alias.
pub type Buf = objc2::rc::Retained<GpuBuffer>;

/// u32 index/token buffer handle. Metal buffers are untyped, so this is the same as `Buf`.
pub type BufU32 = objc2::rc::Retained<GpuBuffer>;

/// Address of a buffer's contents, as usize — for pool/dedup/cache keys. Replaces direct
/// `objc2_metal::MTLBuffer::contents()` calls in shared code so the backend stays swappable.
#[inline]
pub fn buf_addr(b: &Buf) -> usize {
    use objc2_metal::MTLBuffer;
    b.contents().as_ptr() as usize
}

/// Byte length of a buffer. Replaces direct `MTLBuffer::length()` calls in shared code.
#[inline]
pub fn buf_len_bytes(b: &Buf) -> usize {
    use objc2_metal::MTLBuffer;
    b.length()
}

/// Write host bytes into a buffer (unified memory; zero-copy). Replaces inline `.contents()`
/// memcpy in checkpoint/quantize so CUDA can substitute an htod copy.
#[inline]
pub fn buf_write_bytes(buf: &Buf, bytes: &[u8]) {
    use objc2_metal::MTLBuffer;
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buf.contents().as_ptr() as *mut u8,
            bytes.len(),
        );
    }
}

/// Read a slice of bytes from a GPU buffer at the given offset. On Metal shared-memory
/// buffers, this is a direct memcpy from the buffer's contents — no GPU dispatch needed.
pub fn buf_read_bytes(buf: &Buf, offset: usize, length: usize) -> Vec<u8> {
    use objc2_metal::MTLBuffer;
    let mut out = vec![0u8; length];
    unsafe {
        std::ptr::copy_nonoverlapping(
            (buf.contents().as_ptr() as *const u8).add(offset),
            out.as_mut_ptr(),
            length,
        );
    }
    out
}

/// On Metal, buffers are untyped so the u32/f32 handles are the same — identity conversions.
#[inline]
pub fn u32_to_buf(b: BufU32) -> Buf {
    b
}
#[inline]
pub fn buf_as_u32(b: &Buf) -> BufU32 {
    b.clone()
}

/// Active command batch for kernel fusion. Accumulates multiple kernel dispatches
/// into a single command buffer, then commits and waits once.
/// This eliminates the per-kernel commit+wait overhead (~50-80% of wall time).
struct CommandBatch {
    cmd: Retained<GpuCommandBuffer>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    dispatch_count: usize,
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
        let queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        let mut pipelines = HashMap::new();

        let shader_sources: &[(&str, &str)] = &[
            ("matmul_tiled", shaders::MATMUL_TILED),
            ("matmul_tiled_fp32", shaders::MATMUL_TILED_FP32),
            ("matmul_tiled_bf16", shaders::MATMUL_TILED_BF16),
            ("matmul_simdgroup", shaders::MATMUL_SIMDGROUP),
            ("matmul_simdgroup_f16", shaders::MATMUL_SIMDGROUP_F16),
            (
                "batched_matmul_simdgroup",
                shaders::BATCHED_MATMUL_SIMDGROUP,
            ),
            (
                "batched_matmul_simdgroup_trans_b",
                shaders::BATCHED_MATMUL_SIMDGROUP_TRANS_B,
            ),
            (
                "batched_matmul_simdgroup_trans_a",
                shaders::BATCHED_MATMUL_SIMDGROUP_TRANS_A,
            ),
            (
                "matmul_simdgroup_trans_b",
                shaders::MATMUL_SIMDGROUP_TRANS_B,
            ),
            ("matmul_qint4_trans_b", shaders::MATMUL_QINT4_TRANS_B),
            ("matmul_qint4_decode", shaders::MATMUL_QINT4_DECODE),
            (
                "matmul_simdgroup_trans_a",
                shaders::MATMUL_SIMDGROUP_TRANS_A,
            ),
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
            ("adamw_8bit_update", shaders::ADAMW_8BIT_UPDATE),
            ("embedding_lookup", shaders::EMBEDDING_LOOKUP),
            ("causal_mask", shaders::CAUSAL_MASK),
            ("causal_doc_mask", shaders::CAUSAL_DOC_MASK),
            ("block_mean_keys", shaders::BLOCK_MEAN_KEYS),
            ("block_sparse_topk_mask", shaders::BLOCK_SPARSE_TOPK_MASK),
            ("gather_blocks", shaders::GATHER_BLOCKS),
            ("gather_blocks_backward", shaders::GATHER_BLOCKS_BACKWARD),
            ("gather_causal_mask", shaders::GATHER_CAUSAL_MASK),
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
            ("transpose_rope", shaders::TRANSPOSE_ROPE),
            ("transpose_rope_backward", shaders::TRANSPOSE_ROPE_BACKWARD),
            ("gradient_mask", shaders::GRADIENT_MASK),
            ("argmax", shaders::ARGMAX),
            ("temperature_scale", shaders::TEMPERATURE_SCALE),
            ("strided_batch_copy", shaders::STRIDED_BATCH_COPY),
            ("compact_strided_copy", shaders::COMPACT_STRIDED_COPY),
            ("batched_matmul_tiled", shaders::BATCHED_MATMUL_TILED),
            (
                "batched_matmul_tiled_trans_b",
                shaders::BATCHED_MATMUL_TILED_TRANS_B,
            ),
            (
                "batched_matmul_tiled_trans_a",
                shaders::BATCHED_MATMUL_TILED_TRANS_A,
            ),
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
            (
                "flash_attention_backward",
                shaders::FLASH_ATTENTION_BACKWARD,
            ),
            ("cast_f32_to_f16", shaders::CAST_F32_TO_F16),
            ("cast_f16_to_f32", shaders::CAST_F16_TO_F32),
            ("matmul_tiled_f16", shaders::MATMUL_TILED_F16),
            (
                "matmul_tiled_trans_b_f16",
                shaders::MATMUL_TILED_TRANS_B_F16,
            ),
            ("matmul_trans_a_tiled_f16", shaders::MATMUL_TRANS_A_F16),
            (
                "batched_matmul_tiled_f16",
                shaders::BATCHED_MATMUL_TILED_F16,
            ),
            (
                "batched_matmul_tiled_trans_b_f16",
                shaders::BATCHED_MATMUL_TILED_TRANS_B_F16,
            ),
            (
                "batched_matmul_tiled_trans_a_f16",
                shaders::BATCHED_MATMUL_TILED_TRANS_A_F16,
            ),
            ("repeat_kv", shaders::REPEAT_KV),
            ("repeat_kv_backward", shaders::REPEAT_KV_BACKWARD),
            ("scaled_causal_softmax", shaders::SCALED_CAUSAL_SOFTMAX),
            ("scale_copy", shaders::SCALE_COPY),
            ("muon_frob_normalize", shaders::MUON_FROB_NORMALIZE),
            ("inv_sqrt_bc", shaders::INV_SQRT_BC),
            ("rope_copy", shaders::ROPE_COPY),
            ("rope_copy_cached", shaders::ROPE_COPY_CACHED),
            ("rope_backward_copy", shaders::ROPE_BACKWARD_COPY),
            ("matmul_narrow", shaders::MATMUL_NARROW),
            ("axpy", shaders::AXPY),
            ("relu", shaders::RELU),
            ("relu_backward", shaders::RELU_BACKWARD),
            ("exp_fwd", shaders::EXP),
            ("log_fwd", shaders::LOG),
            ("softplus_fwd", shaders::SOFTPLUS),
            ("rms_norm_gated", shaders::RMS_NORM_GATED),
            ("broadcast_rows", shaders::BROADCAST_ROWS),
            ("ema_update", shaders::EMA_UPDATE),
            ("cautious_mask", shaders::CAUTIOUS_MASK),
            ("cautious_scale", shaders::CAUTIOUS_SCALE),
            ("logsumexp", shaders::LOGSUMEXP),
            ("concat_cols", shaders::CONCAT_COLS),
            ("slice_cols", shaders::SLICE_COLS),
            (
                "batched_matmul_gqa_trans_b",
                shaders::BATCHED_MATMUL_GQA_TRANS_B,
            ),
            ("batched_matmul_gqa", shaders::BATCHED_MATMUL_GQA),
            ("mega_ffn", shaders::MEGA_FFN),
        ];

        let compile_options = MTLCompileOptions::new();

        for (kernel_name, source) in shader_sources {
            let ns_source = NSString::from_str(source);

            let library = device
                .newLibraryWithSource_options_error(&ns_source, Some(&compile_options))
                .unwrap_or_else(|e| panic!("Failed to compile shader '{}': {}", kernel_name, e));

            let ns_name = NSString::from_str(kernel_name);
            let function = library
                .newFunctionWithName(&ns_name)
                .unwrap_or_else(|| panic!("Failed to get function '{}' from library", kernel_name));

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

    /// Diagnostic: when `SMEDJAN_NO_POOL` is set in the environment, every `alloc_buffer`
    /// bypasses the size-keyed reuse pool and returns a brand-new buffer. This removes
    /// intra-batch buffer aliasing (the #1 known GPU-correctness hazard: a pooled buffer
    /// recycled then re-handed-out while a still-pending dispatch needs it) as a variable
    /// when bisecting gradient-corruption bugs. Process-global, env read once. Slower + more
    /// memory, so OFF by default; pure diagnostic, never a correctness dependency.
    fn pool_disabled() -> bool {
        use std::sync::OnceLock;
        static D: OnceLock<bool> = OnceLock::new();
        *D.get_or_init(|| std::env::var("SMEDJAN_NO_POOL").is_ok())
    }

    /// Allocate a shared-mode Metal buffer (CPU + GPU accessible, zero-copy on M1).
    /// Checks the buffer pool first for a cached buffer of the same size.
    pub fn alloc_buffer(&self, size_bytes: usize) -> Retained<GpuBuffer> {
        // Debug: track allocation sizes on first step
        ALLOC_SIZE_LOG.with(|log| {
            let mut l = log.borrow_mut();
            if l.0 {
                // logging enabled
                *l.1.entry(size_bytes).or_insert(0) += 1;
            }
        });

        // Try pool first (unless bypassed — e.g. during checkpoint recompute, where reusing a
        // pooled buffer the outer backward still references would corrupt gradients).
        let bypass = POOL_BYPASS.with(|b| b.get()) > 0 || Self::pool_disabled();
        let pooled = if bypass {
            None
        } else {
            BUFFER_POOL.with(|pool| {
                let mut p = pool.borrow_mut();
                if let Some(list) = p.get_mut(&size_bytes) {
                    // Quarantine (default): reissue only a buffer recycled in an already-flushed
                    // generation, so a buffer recycled in THIS uncommitted batch can't be handed
                    // back out and overwritten while a still-pending dispatch needs it. Forcing a
                    // fresh allocation when every pooled buffer is same-generation is what keeps
                    // gradients correct at seq_len ≥ 256.
                    if QUARANTINE.with(|q| q.get()) {
                        let cur = BATCH_GENERATION.with(|g| g.get());
                        let pick = RECYCLE_GEN.with(|rg| {
                            let rg = rg.borrow();
                            list.iter().position(|b| {
                                rg.get(&buf_contents_addr(b))
                                    .copied()
                                    .is_none_or(|g| g < cur)
                            })
                        });
                        return match pick {
                            Some(i) => {
                                let buf = list.remove(i);
                                RECYCLE_GEN.with(|rg| {
                                    rg.borrow_mut().remove(&buf_contents_addr(&buf));
                                });
                                #[cfg(feature = "bufsan")]
                                POISONED_POOL_ADDRS.with(|p| {
                                    p.borrow_mut().remove(&buf_contents_addr(&buf));
                                });
                                POOL_STATS.with(|s| s.borrow_mut().0 += 1);
                                Some(buf)
                            }
                            None => None, // all same-generation → force a fresh allocation
                        };
                    }
                    if let Some(buf) = list.pop() {
                        RECYCLE_GEN.with(|rg| {
                            rg.borrow_mut().remove(&buf_contents_addr(&buf));
                        });
                        #[cfg(feature = "bufsan")]
                        POISONED_POOL_ADDRS.with(|p| {
                            p.borrow_mut().remove(&buf_contents_addr(&buf));
                        });
                        POOL_STATS.with(|s| s.borrow_mut().0 += 1);
                        return Some(buf);
                    }
                }
                None
            })
        };
        let buf = if let Some(buf) = pooled {
            buf
        } else {
            POOL_STATS.with(|s| s.borrow_mut().1 += 1);
            self.device
                .newBufferWithLength_options(size_bytes, MTLResourceOptions::StorageModeShared)
                .expect("Failed to allocate Metal buffer")
        };
        // This buffer's contents are about to be written fresh. Its address may previously have
        // belonged to a now-freed buffer whose fp16/ternary conversion is still cached (those
        // caches are keyed by address and only fully cleared after the optimizer step). Drop any
        // such stale entry so a later cast_to_f16/ternary on this address can't get a false hit.
        use objc2_metal::MTLBuffer;
        crate::tensor::Tensor::invalidate_conversion_cache(buf.contents().as_ptr() as usize);
        #[cfg(feature = "bufsan")]
        bufsan_poison_buffer(&buf);
        buf
    }

    /// Return a buffer to the pool for reuse. Call when a buffer is no longer needed.
    pub fn recycle_buffer(buf: Retained<GpuBuffer>) {
        // While the pool is bypassed (checkpoint recompute), drop the buffer instead of pooling —
        // pooling here could hand a still-referenced buffer to a later allocation.
        if POOL_BYPASS.with(|b| b.get()) > 0 {
            return;
        }
        let size = buf.length();
        let addr = buf_contents_addr(&buf);
        BUFFER_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            let list = p.entry(size).or_default();
            if list.iter().any(|b| buf_contents_addr(b) == addr) {
                return;
            }
            // Cap pool size per bucket to avoid unbounded memory growth.
            // 64 allows gradient-checkpointed backward to find reusable buffers
            // across layer recomputes (was 32, causing ~63% miss rate).
            if list.len() < 64 {
                // Stamp the recycle generation so quarantine won't reissue this buffer until the
                // batch it was recycled in has been flushed (committed + waited).
                let gen_val = BATCH_GENERATION.with(|g| g.get());
                RECYCLE_GEN.with(|rg| {
                    rg.borrow_mut().insert(addr, gen_val);
                });
                #[cfg(feature = "bufsan")]
                POISONED_POOL_ADDRS.with(|p| {
                    p.borrow_mut().remove(&addr);
                });
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
            eprintln!(
                "[ALLOC LOG] {} — {} unique sizes, {} total allocs:",
                label,
                sizes.len(),
                total
            );
            for (size, count) in sizes.iter().take(20) {
                eprintln!("  {:>10} bytes × {:>4}", size, count);
            }
        });
    }

    /// Clear the buffer pool (e.g., between training runs to free memory)
    pub fn clear_pool() {
        BUFFER_POOL.with(|pool| pool.borrow_mut().clear());
        POOL_STATS.with(|s| *s.borrow_mut() = (0, 0));
        RECYCLE_GEN.with(|rg| rg.borrow_mut().clear());
        BATCH_GENERATION.with(|g| g.set(0));
        #[cfg(feature = "bufsan")]
        POISONED_POOL_ADDRS.with(|p| p.borrow_mut().clear());
    }

    /// Enable/disable buffer-pool quarantine. ON by default (the gradient-corruption fix). When on,
    /// alloc reissues only buffers recycled in an already-flushed generation, eliminating
    /// intra-batch recycle→reissue aliasing. Exposed mainly so the `bufsan` differential test can
    /// toggle it; production should leave it on.
    pub fn set_pool_quarantine(on: bool) {
        QUARANTINE.with(|q| q.set(on));
    }

    /// Current command-batch generation (count of waiting flushes since start/clear).
    pub fn pool_generation() -> u64 {
        BATCH_GENERATION.with(|g| g.get())
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
        let buf = unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Failed to create buffer from slice")
        };
        // Same staleness hazard as alloc_buffer: this fresh buffer may reuse the address of a
        // just-freed buffer whose fp16/ternary conversion is still cached. Drop any stale entry
        // so a later cast_to_f16/ternary on this address can't return a false hit. (Surfaced by
        // the grad-check harness: a perturbed from_slice input was silently ignored by a cached
        // fp16 cast → numeric gradient of exactly 0.)
        use objc2_metal::MTLBuffer;
        crate::tensor::Tensor::invalidate_conversion_cache(buf.contents().as_ptr() as usize);
        buf
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
    /// Allocate a zeroed u32 buffer of `count` elements (untyped on Metal → reuse alloc_buffer).
    pub fn alloc_buffer_u32(&self, count: usize) -> Buf {
        self.alloc_buffer(count * 4)
    }

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
        let flushed = ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            if let Some(cb) = b.take()
                && cb.dispatch_count > 0
            {
                cb.encoder.endEncoding();
                cb.cmd.commit();
                cb.cmd.waitUntilCompleted();
                return true;
            }
            false
        });
        if flushed {
            on_flush();
        }
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
    /// into a single compute encoder on one command buffer. This amortizes encoder
    /// creation overhead (~300μs) across all dispatches instead of per-dispatch.
    /// Call `flush_batch()` when you need results.
    pub fn begin_batch(&self) {
        let flushed = ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            // If a batch is already active, flush it first
            let mut did = false;
            if let Some(cb) = b.take()
                && cb.dispatch_count > 0
            {
                cb.encoder.endEncoding();
                cb.cmd.commit();
                cb.cmd.waitUntilCompleted();
                did = true;
            }
            let cmd = self
                .queue
                .commandBuffer()
                .expect("Failed to create command buffer");
            let encoder = cmd
                .computeCommandEncoder()
                .expect("Failed to create encoder");
            *b = Some(CommandBatch {
                cmd,
                encoder,
                dispatch_count: 0,
            });
            did
        });
        if flushed {
            on_flush();
        }
    }

    /// Flush the current command batch: end encoder, commit, and wait for GPU completion.
    /// Returns the number of kernels that were batched.
    pub fn flush_batch(&self) -> usize {
        let count = ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            if let Some(cb) = b.take() {
                if cb.dispatch_count > 0 {
                    cb.encoder.endEncoding();
                    cb.cmd.commit();
                    cb.cmd.waitUntilCompleted();
                }
                cb.dispatch_count
            } else {
                0
            }
        });
        if count > 0 {
            on_flush();
        }
        count
    }

    /// Flush without waiting — commit the command buffer but don't block.
    /// The GPU runs in parallel with the CPU. Call flush_batch() or
    /// wait_batch() before reading any GPU buffers.
    pub fn flush_batch_async(&self) -> usize {
        ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            if let Some(cb) = b.take() {
                if cb.dispatch_count > 0 {
                    cb.encoder.endEncoding();
                    cb.cmd.commit();
                    // Don't wait — GPU runs while CPU prepares next batch
                }
                cb.dispatch_count
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
        let pipeline = self
            .pipelines
            .get(pipeline_name)
            .unwrap_or_else(|| panic!("Unknown pipeline: {}", pipeline_name));

        let completed_sync_dispatch = ACTIVE_BATCH.with(|batch| {
            let mut b = batch.borrow_mut();
            if let Some(ref mut cb) = *b {
                // Batched path: encode into the persistent compute encoder (no encoder create/destroy overhead)
                cb.encoder.setComputePipelineState(pipeline);
                bind(&cb.encoder);
                if use_dispatch_threads {
                    cb.encoder
                        .dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
                } else {
                    cb.encoder
                        .dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
                }
                cb.dispatch_count += 1;
                false
            } else {
                // Unbatched path: one-off command buffer with sync wait
                let cmd = self
                    .queue
                    .commandBuffer()
                    .expect("Failed to create command buffer");
                let encoder = cmd
                    .computeCommandEncoder()
                    .expect("Failed to create encoder");
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
                true
            }
        });
        if completed_sync_dispatch {
            on_flush();
        }
    }
}
