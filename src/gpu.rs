//! GPU backend abstraction — compile-time selection via features.
//!
//! `cargo build`                                        → Metal (macOS)
//! `cargo build --features cuda --no-default-features`  → CUDA (NVIDIA)
//! `cargo build --features andreos --no-default-features` → AndreOS (direct HW)
//!
//! Backends re-exported here. Use `crate::gpu::MetalContext` etc. for portable code.

// Backend-agnostic re-exports. Use `crate::gpu::GpuContext` for portable code.
#[cfg(feature = "metal")]
pub use crate::metal::{MetalContext as GpuContext};

#[cfg(feature = "cuda")]
pub use crate::cuda::{MetalContext, compute};
#[cfg(feature = "cuda")]
pub use crate::cuda::MetalContext as GpuContext;

#[cfg(feature = "andreos")]
pub use crate::andreos::{MetalContext, compute};
#[cfg(feature = "andreos")]
pub use crate::andreos::MetalContext as GpuContext;
