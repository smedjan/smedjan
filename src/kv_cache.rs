//! KV-cache for the Qwen3.5 full-attention layers during autoregressive decode.
//!
//! At decode time, each full-attention layer computes K and V for the *current* token only.
//! The KV-cache stores all previous K/V vectors so attention can be computed over the full
//! context without re-running the previous tokens through the model. This makes decode
//! O(1) per new token for the attention layers (the 24 DeltaNet layers are already O(1) —
//! they maintain a fixed-size recurrent state).
//!
//! The cache grows by `[n_kv, head_dim]` per token per layer. For 8 full-attention layers with
//! n_kv=4, head_dim=256: 8 * 4 * 256 * 4 bytes = 32 KB per token — negligible vs the 5 GB model.

use crate::gpu::MetalContext;
use crate::tensor::Tensor;
use std::sync::Arc;

/// KV-cache for one full-attention layer. Stores K and V across all decode steps.
/// K_cache: `[seq_len, n_kv, head_dim]` (grows by one row per decode step)
/// V_cache: same shape
pub struct KvCache {
    pub k_cache: Vec<f32>, // [max_seq, n_kv * head_dim]
    pub v_cache: Vec<f32>, // [max_seq, n_kv * head_dim]
    pub seq_len: usize,    // current number of cached tokens
    pub n_kv: usize,
    pub head_dim: usize,
}

impl KvCache {
    pub fn new(max_seq: usize, n_kv: usize, head_dim: usize) -> Self {
        let kv_size = n_kv * head_dim;
        KvCache {
            k_cache: vec![0.0f32; max_seq * kv_size],
            v_cache: vec![0.0f32; max_seq * kv_size],
            seq_len: 0,
            n_kv,
            head_dim,
        }
    }

    /// Append one token's K and V to the cache. `k_new` and `v_new` are `[n_kv * head_dim]` f32.
    pub fn append(&mut self, k_new: &[f32], v_new: &[f32]) {
        let kv_size = self.n_kv * self.head_dim;
        assert_eq!(k_new.len(), kv_size, "K new size mismatch");
        assert_eq!(v_new.len(), kv_size, "V new size mismatch");
        let offset = self.seq_len * kv_size;
        self.k_cache[offset..offset + kv_size].copy_from_slice(k_new);
        self.v_cache[offset..offset + kv_size].copy_from_slice(v_new);
        self.seq_len += 1;
    }

    /// Get all cached K as a tensor `[seq_len, n_kv * head_dim]`.
    pub fn k_tensor(&self, ctx: &Arc<MetalContext>) -> Tensor {
        let kv_size = self.n_kv * self.head_dim;
        let data = &self.k_cache[..self.seq_len * kv_size];
        Tensor::from_slice(ctx, data, vec![self.seq_len, kv_size])
    }

    /// Get all cached V as a tensor `[seq_len, n_kv * head_dim]`.
    pub fn v_tensor(&self, ctx: &Arc<MetalContext>) -> Tensor {
        let kv_size = self.n_kv * self.head_dim;
        let data = &self.v_cache[..self.seq_len * kv_size];
        Tensor::from_slice(ctx, data, vec![self.seq_len, kv_size])
    }

    /// Reset the cache (start a new generation).
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

/// Per-layer KV-caches for the full model. Index by layer index.
pub struct ModelKvCache {
    pub caches: Vec<Option<KvCache>>, // Some for full-attn layers, None for DeltaNet
}

impl ModelKvCache {
    pub fn new(max_seq: usize, n_kv: usize, head_dim: usize, is_full_attention: &[bool]) -> Self {
        let caches = is_full_attention
            .iter()
            .map(|&is_full| {
                if is_full {
                    Some(KvCache::new(max_seq, n_kv, head_dim))
                } else {
                    None
                }
            })
            .collect();
        ModelKvCache { caches }
    }

    pub fn reset(&mut self) {
        for c in &mut self.caches {
            if let Some(c) = c {
                c.reset();
            }
        }
    }
}
