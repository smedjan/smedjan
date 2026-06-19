//! Multi-head Latent Attention (MLA) — DeepSeek-V2/V3 KV-cache compression.
//!
//! Standard attention caches K and V at `2 · n_kv_heads · head_dim` floats per token. MLA instead
//! caches a single shared low-rank latent `c = x @ W_dkv` of dim `d_c ≪ n_kv_heads · head_dim`, and
//! reconstructs K and V on the fly via up-projections. The cache shrinks by
//! `2 · n_kv_heads · head_dim / d_c` (typically 10–50×) → 10× longer context and ⅒ the inference KV
//! memory, which is the lever this gives the "10× capacity" goal.
//!
//! This module is the differentiable core (plain matmuls); `MultiHeadAttention` wires it in as
//! `AttnKind::Mla`. The attention math over the reconstructed (Q, K, V) is unchanged from softmax
//! attention, so RoPE/GQA/causal-masking all apply as usual.

use crate::tensor::Tensor;

/// Compress and reconstruct K, V through a shared low-rank latent.
///
/// * `x`      — `[n_tokens, d_model]` layer input (already flattened over batch·seq).
/// * `w_dkv`  — `[d_model, d_c]` KV down-projection. Its output `c` is the cacheable latent.
/// * `w_uk`   — `[d_c, kv_dim]` key up-projection.
/// * `w_uv`   — `[d_c, kv_dim]` value up-projection.
///
/// Returns `(c, k, v)`:
/// * `c` — `[n_tokens, d_c]` the latent (what an MLA KV cache stores).
/// * `k` — `[n_tokens, kv_dim]` reconstructed keys.
/// * `v` — `[n_tokens, kv_dim]` reconstructed values.
///
/// Fully differentiable — gradients flow into `w_dkv`, `w_uk`, `w_uv`.
pub fn mla_kv(
    x: &Tensor,
    w_dkv: &Tensor,
    w_uk: &Tensor,
    w_uv: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let c = x.matmul(w_dkv); // [n_tokens, d_c] — the compressed latent
    let k = c.matmul(w_uk); // [n_tokens, kv_dim]
    let v = c.matmul(w_uv); // [n_tokens, kv_dim]
    (c, k, v)
}

/// Per-token KV-cache footprint (in floats) of standard attention vs MLA, and the shrink factor.
/// Standard caches K and V (`2 · kv_dim`); MLA caches the latent (`d_c`).
pub fn cache_footprint(kv_dim: usize, d_c: usize) -> (usize, usize, f32) {
    let standard = 2 * kv_dim;
    let mla = d_c;
    (standard, mla, standard as f32 / mla.max(1) as f32)
}
