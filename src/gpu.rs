//! GPU backend abstraction — compile-time selection via features.
//!
//! `cargo build` (default) → Metal backend (macOS/Apple Silicon)
//! `cargo build --features cuda --no-default-features` → CUDA backend (NVIDIA)
//!
//! Both backends export the same public API:
//!   - GpuContext (MetalContext or CudaContext)
//!   - GpuBuffer type
//!   - compute module (gpu_matmul, gpu_softmax, etc.)
//!
//! All other code (model.rs, tensor.rs, autograd.rs) uses `crate::gpu::*`
//! and is backend-agnostic.

#[cfg(feature = "metal")]
pub use crate::metal::*;

#[cfg(feature = "cuda")]
pub use crate::cuda::*;
