//! GPU backend abstraction — compile-time selection via features.
//!
//! `cargo build`                                          → Metal (macOS)
//! `cargo build --features cuda --no-default-features`    → CUDA (NVIDIA)
//! `cargo build --features andreos --no-default-features` → AndreOS (direct HW)
//!
//! Portable code refers to `crate::gpu::{MetalContext, GpuBuffer, Buf, compute, buf_addr,
//! buf_len_bytes, ...}` instead of naming a backend (or objc2) directly, so one cfg switch swaps the
//! whole backend. Each backend module exposes the same surface; CUDA substitutes `Arc<CudaSlice>`
//! for `Buf` so `.clone()` stays a cheap share. Glob re-export so every backend symbol the shared
//! code uses (contexts, buffer types, pool guards, helpers, the `compute` module) is available here.

#[cfg(feature = "metal")]
pub use crate::metal::*;
#[cfg(feature = "metal")]
pub use crate::metal::MetalContext as GpuContext;

#[cfg(all(feature = "cuda", not(feature = "metal")))]
pub use crate::cuda::*;
#[cfg(all(feature = "cuda", not(feature = "metal")))]
pub use crate::cuda::MetalContext as GpuContext;

#[cfg(all(feature = "andreos", not(feature = "metal"), not(feature = "cuda")))]
pub use crate::andreos::*;
#[cfg(all(feature = "andreos", not(feature = "metal"), not(feature = "cuda")))]
pub use crate::andreos::MetalContext as GpuContext;
