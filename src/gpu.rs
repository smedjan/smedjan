//! GPU backend abstraction — compile-time selection via features.
//!
//! `cargo build`                                        → Metal (macOS)
//! `cargo build --features cuda --no-default-features`  → CUDA (NVIDIA)
//! `cargo build --features andreos --no-default-features` → AndreOS (direct HW)
//!
//! All backends export identical public API. Model code is backend-agnostic.

#[cfg(feature = "metal")]
pub use crate::metal::*;

#[cfg(feature = "cuda")]
pub use crate::cuda::*;

#[cfg(feature = "andreos")]
pub use crate::andreos::*;
