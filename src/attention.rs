use crate::autograd::{self, Op, TapeEntry};
use crate::metal::{compute, MetalContext};
use crate::tensor::Tensor;
use std::sync::Arc;

/// Multi-head attention with rotary positional encoding, KV cache, and
/// Grouped Query Attention (GQA) support.
///
/// When `n_kv_heads < n_heads`, K and V projections are smaller (kv_dim instead
/// of d_model). Each KV head serves `group_size = n_heads / n_kv_heads` query heads.
/// When `n_kv_heads == n_heads`, this is standard Multi-Head Attention.
pub struct MultiHeadAttention {
    pub w_q: Tensor, // [d_model, d_model]
    pub w_k: Tensor, // [d_model, kv_dim] where kv_dim = head_dim * n_kv_heads
    pub w_v: Tensor, // [d_model, kv_dim]
    pub w_o: Tensor, // [d_model, d_model]
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
    pub rope_theta: f32,
}

/// KV cache for autoregressive inference.
/// Pre-allocates to max_seq_len capacity to avoid O(n^2) reallocation
/// during autoregressive generation. Only new KV pairs are copied each step.
pub struct KvCache {
    pub k: Option<Tensor>, // [batch * n_heads, capacity, head_dim] (only [0..len] is valid)
    pub v: Option<Tensor>, // [batch * n_heads, capacity, head_dim]
    pub len: usize,        // current number of cached positions
    pub capacity: usize,   // pre-allocated max positions
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None, len: 0, capacity: 0 }
    }

    /// Create a pre-allocated KV cache with capacity for max_seq_len positions.
    pub fn with_capacity(ctx: &Arc<MetalContext>, batch_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let total_floats = batch_heads * max_seq_len * head_dim;
        let k_buf = ctx.alloc_buffer(total_floats * 4);
        let v_buf = ctx.alloc_buffer(total_floats * 4);
        Self {
            k: Some(Tensor::from_buffer(Arc::clone(ctx), k_buf, vec![batch_heads, max_seq_len, head_dim])),
            v: Some(Tensor::from_buffer(Arc::clone(ctx), v_buf, vec![batch_heads, max_seq_len, head_dim])),
            len: 0,
            capacity: max_seq_len,
        }
    }

    pub fn cached_len(&self) -> usize {
        self.len
    }

    /// Truncate the KV cache to `new_len` positions.
    /// Used by speculative decoding to roll back rejected draft tokens.
    /// For pre-allocated caches, this simply adjusts the length counter —
    /// the stale data beyond `new_len` is never read because attention
    /// only looks at `[0..len]`. For legacy caches, we must also shrink
    /// the actual tensors.
    pub fn truncate(&mut self, new_len: usize) {
        assert!(new_len <= self.len, "truncate: new_len {} > current len {}", new_len, self.len);
        if self.capacity > 0 {
            // Pre-allocated path: just move the length counter back.
            // The buffer space from new_len..old_len is "dead" and will be
            // overwritten by the next update_kv_cache call.
            self.len = new_len;
        } else {
            // Legacy path: rebuild tensors from the first new_len positions.
            if new_len == 0 {
                self.k = None;
                self.v = None;
                self.len = 0;
            } else {
                // Extract geometry from existing tensors before mutating self
                let (bh, old_len, head_dim, ctx_clone) = {
                    let k = self.k.as_ref().expect("legacy cache must have k");
                    (k.shape[0], k.shape[1], k.shape[2], Arc::clone(&k.ctx))
                };

                let k_buf = ctx_clone.alloc_buffer(bh * new_len * head_dim * 4);
                let v_buf = ctx_clone.alloc_buffer(bh * new_len * head_dim * 4);

                // Copy [0..new_len] from old tensors (which have stride = old_len)
                compute::gpu_compact_strided_copy(
                    &ctx_clone, &self.k.as_ref().unwrap().buffer, &k_buf,
                    bh as u32, new_len as u32, old_len as u32, head_dim as u32,
                );
                compute::gpu_compact_strided_copy(
                    &ctx_clone, &self.v.as_ref().unwrap().buffer, &v_buf,
                    bh as u32, new_len as u32, old_len as u32, head_dim as u32,
                );

                self.k = Some(Tensor::from_buffer(Arc::clone(&ctx_clone), k_buf, vec![bh, new_len, head_dim]));
                self.v = Some(Tensor::from_buffer(ctx_clone, v_buf, vec![bh, new_len, head_dim]));
                self.len = new_len;
            }
        }
    }
}

impl MultiHeadAttention {
    pub fn new(ctx: &Arc<MetalContext>, d_model: usize, n_heads: usize, n_kv_heads: usize, rope_theta: f32) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        assert!(n_kv_heads <= n_heads, "n_kv_heads must be <= n_heads");
        assert_eq!(n_heads % n_kv_heads, 0, "n_heads must be divisible by n_kv_heads");
        let head_dim = d_model / n_heads;
        let kv_dim = head_dim * n_kv_heads;

        // Xavier initialization
        let std_q = (2.0 / (d_model + d_model) as f32).sqrt();
        let std_kv = (2.0 / (d_model + kv_dim) as f32).sqrt();

        Self {
            w_q: Tensor::randn(ctx, vec![d_model, d_model], std_q),
            w_k: Tensor::randn(ctx, vec![d_model, kv_dim], std_kv),
            w_v: Tensor::randn(ctx, vec![d_model, kv_dim], std_kv),
            w_o: Tensor::randn(ctx, vec![d_model, d_model], std_q),
            n_heads,
            n_kv_heads,
            head_dim,
            d_model,
            rope_theta,
        }
    }

    /// Forward pass with Grouped Query Attention support.
    /// x: [batch, seq_len, d_model]
    /// Returns: [batch, seq_len, d_model]
    ///
    /// When n_kv_heads < n_heads, K and V are projected to kv_dim = head_dim * n_kv_heads,
    /// then expanded via repeat_kv to match n_heads before attention computation.
    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut KvCache>,
    ) -> Tensor {
        let batch = x.shape[0];
        let seq_len = x.shape[1];
        let d_model = x.shape[2];
        assert_eq!(d_model, self.d_model);

        // Flatten batch*seq for matmul: [batch*seq, d_model]
        let x_flat = x.reshape(vec![batch * seq_len, d_model]);

        // Project Q, K, V — these go through the tape via matmul
        let q = x_flat.matmul(&self.w_q); // [batch*seq, d_model]
        let k = x_flat.matmul(&self.w_k); // [batch*seq, kv_dim]
        let v = x_flat.matmul(&self.w_v); // [batch*seq, kv_dim]

        // Transpose Q: [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim]
        let bh_kv = batch * self.n_kv_heads;
        let q = transpose_bsh_to_bhs(&q, batch, seq_len, self.n_heads, self.head_dim);
        // Transpose K, V: [batch*seq, n_kv_heads*head_dim] → [batch*n_kv_heads, seq, head_dim]
        let k = transpose_bsh_to_bhs(&k, batch, seq_len, self.n_kv_heads, self.head_dim);
        let v = transpose_bsh_to_bhs(&v, batch, seq_len, self.n_kv_heads, self.head_dim);

        // Apply RoPE to Q and K
        let offset = match &kv_cache {
            Some(cache) => cache.cached_len() as u32,
            None => 0,
        };
        let q = q.apply_rope(offset, self.rope_theta);
        let k = k.apply_rope(offset, self.rope_theta);

        // Handle KV cache (inference only — no tape needed)
        // Cache stores n_kv_heads, not n_heads
        let (k_full, v_full) = match kv_cache {
            Some(cache) => update_kv_cache(cache, &k, &v, bh_kv, self.head_dim),
            None => (k, v),
        };

        // Expand KV heads to match Q heads for attention computation.
        // k_full: [batch*n_kv_heads, seq_k, head_dim] → [batch*n_heads, seq_k, head_dim]
        let group_size = self.n_heads / self.n_kv_heads;
        let seq_k = k_full.shape[1];
        let (k_expanded, v_expanded) = if group_size == 1 {
            // Standard MHA — no expansion needed
            (k_full, v_full)
        } else {
            (
                repeat_kv(&k_full, bh_kv, seq_k, self.head_dim, group_size),
                repeat_kv(&v_full, bh_kv, seq_k, self.head_dim, group_size),
            )
        };

        // --- Attention computation ---
        let bh = batch * self.n_heads;
        let seq_q_len = q.shape[1];
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Use Flash Attention when not recording gradients (inference).
        // For training, use standard 4-op path (backward is well-tested).
        // Flash Attention backward kernel is Phase 1b optimization.
        let attn_cat = if !autograd::is_recording() {
            // FLASH ATTENTION: fused Q@K^T → mask → softmax → @V
            // One kernel, O(n) memory, never materializes N×N scores.
            let attn_out_buf = q.ctx.alloc_buffer(bh * seq_q_len * self.head_dim * 4);
            compute::gpu_flash_attention_forward(
                &q.ctx,
                &q.buffer, &k_expanded.buffer, &v_expanded.buffer, &attn_out_buf,
                bh as u32, seq_q_len as u32, seq_k as u32, self.head_dim as u32, offset,
            );
            Tensor {
                id: autograd::next_id(),
                buffer: attn_out_buf,
                shape: vec![bh, seq_q_len, self.head_dim],
                requires_grad: false,
                ctx: Arc::clone(&q.ctx),
            }
        } else {
            // STANDARD ATTENTION: 4 separate ops (backward-compatible)
            let scores = q.batched_matmul_trans_b(&k_expanded);
            let scores = scores.scale(scale);
            let scores = scores.causal_mask(offset);
            let weights = scores.softmax();
            weights.batched_matmul(&v_expanded)
        };

        // Transpose [bh, seq, head_dim] back to [batch*seq, d_model]
        let attn_combined = transpose_bhs_to_bsh(&attn_cat, batch, seq_len, self.n_heads, self.head_dim);

        // Output projection
        let out = attn_combined.matmul(&self.w_o); // [batch*seq, d_model]
        out.reshape(vec![batch, seq_len, d_model])
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o]
    }
}

/// Transpose [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim]
/// Records a tape entry so gradients flow through.
fn transpose_bsh_to_bhs(
    t: &Tensor,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Tensor {
    let bh = batch * n_heads;
    let size = bh * seq_len * head_dim;
    let out_buf = t.ctx.alloc_buffer(size * 4);

    // GPU transpose — no CPU roundtrip
    compute::gpu_transpose_perm_forward(
        &t.ctx,
        &t.buffer,
        &out_buf,
        batch as u32,
        seq_len as u32,
        n_heads as u32,
        head_dim as u32,
    );
    let out_id = autograd::next_id();
    let result = Tensor {
        id: out_id,
        buffer: out_buf.clone(),
        shape: vec![bh, seq_len, head_dim],
        requires_grad: false,
        ctx: Arc::clone(&t.ctx),
    };

    // Record transpose on tape for backward
    if t.requires_grad || autograd::is_recording() {
        autograd::record(TapeEntry {
            op: Op::Transpose {
                batch,
                seq_len,
                n_heads,
                head_dim,
                forward_dir: true, // bsh → bhs
            },
            inputs: vec![t.id],
            output: out_id,
            input_buffers: vec![t.buffer.clone()],
            output_buffer: out_buf,
            shapes: vec![t.shape.clone(), result.shape.clone()],
            cached: None,
        });
    }

    result
}

/// Transpose [batch*n_heads, seq, head_dim] → [batch*seq, n_heads*head_dim]
/// Records a tape entry (reverse direction) so gradients flow through.
fn transpose_bhs_to_bsh(
    t: &Tensor,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Tensor {
    let d_model = n_heads * head_dim;
    let size = batch * seq_len * d_model;
    let out_buf = t.ctx.alloc_buffer(size * 4);

    // GPU transpose — bhs→bsh is the backward of bsh→bhs
    compute::gpu_transpose_perm_backward(
        &t.ctx,
        &t.buffer,
        &out_buf,
        batch as u32,
        seq_len as u32,
        n_heads as u32,
        head_dim as u32,
    );
    let out_id = autograd::next_id();
    let result = Tensor {
        id: out_id,
        buffer: out_buf.clone(),
        shape: vec![batch * seq_len, d_model],
        requires_grad: false,
        ctx: Arc::clone(&t.ctx),
    };

    if t.requires_grad || autograd::is_recording() {
        autograd::record(TapeEntry {
            op: Op::Transpose {
                batch,
                seq_len,
                n_heads,
                head_dim,
                forward_dir: false, // bhs → bsh (reverse direction)
            },
            inputs: vec![t.id],
            output: out_id,
            input_buffers: vec![t.buffer.clone()],
            output_buffer: out_buf,
            shapes: vec![t.shape.clone(), result.shape.clone()],
            cached: None,
        });
    }

    result
}

/// Update KV cache by concatenating new K/V with cached K/V.
/// This is only used during inference (no_grad), so no tape needed.
fn update_kv_cache(
    cache: &mut KvCache,
    k_new: &Tensor,
    v_new: &Tensor,
    bh: usize,
    head_dim: usize,
) -> (Tensor, Tensor) {
    let new_len = k_new.shape[1];

    if cache.capacity > 0 {
        // Pre-allocated path: copy new KV data into the buffer at the right offset.
        // No reallocation, no copying old data — O(new_len) per step instead of O(total_len).
        let k_cache = cache.k.as_ref().expect("pre-allocated cache must have k");
        let v_cache = cache.v.as_ref().expect("pre-allocated cache must have v");
        let old_len = cache.len;
        let total_len = old_len + new_len;
        assert!(total_len <= cache.capacity, "KV cache overflow: {} + {} > {}", old_len, new_len, cache.capacity);

        // Copy new K, V into cache at offset = old_len (single batched dispatch per tensor)
        compute::gpu_strided_batch_copy(
            &k_cache.ctx, &k_new.buffer, &k_cache.buffer,
            bh as u32, new_len as u32, cache.capacity as u32, old_len as u32, head_dim as u32,
        );
        compute::gpu_strided_batch_copy(
            &v_cache.ctx, &v_new.buffer, &v_cache.buffer,
            bh as u32, new_len as u32, cache.capacity as u32, old_len as u32, head_dim as u32,
        );

        cache.len = total_len;

        // Return views that cover [0..total_len] of the cache.
        // We create tensors that reference sub-regions via buffer_copy to a contiguous buffer
        // because attention needs contiguous [bh, total_len, head_dim] layout, not strided.
        let k_view_buf = k_cache.ctx.alloc_buffer(bh * total_len * head_dim * 4);
        let v_view_buf = k_cache.ctx.alloc_buffer(bh * total_len * head_dim * 4);
        compute::gpu_compact_strided_copy(
            &k_cache.ctx, &k_cache.buffer, &k_view_buf,
            bh as u32, total_len as u32, cache.capacity as u32, head_dim as u32,
        );
        compute::gpu_compact_strided_copy(
            &v_cache.ctx, &v_cache.buffer, &v_view_buf,
            bh as u32, total_len as u32, cache.capacity as u32, head_dim as u32,
        );

        let k_full = Tensor::from_buffer(Arc::clone(&k_cache.ctx), k_view_buf, vec![bh, total_len, head_dim]);
        let v_full = Tensor::from_buffer(Arc::clone(&v_cache.ctx), v_view_buf, vec![bh, total_len, head_dim]);
        (k_full, v_full)
    } else {
        // Legacy path: concat and reallocate (used when cache was created with new())
        match (&cache.k, &cache.v) {
            (Some(k_old), Some(v_old)) => {
                let old_len = k_old.shape[1];

                let k_full = concat_seq(k_old, k_new, bh, old_len, new_len, head_dim);
                let v_full = concat_seq(v_old, v_new, bh, old_len, new_len, head_dim);

                cache.k = Some(k_full.clone());
                cache.v = Some(v_full.clone());
                cache.len = old_len + new_len;

                (k_full, v_full)
            }
            _ => {
                cache.k = Some(k_new.clone());
                cache.v = Some(v_new.clone());
                cache.len = new_len;
                (k_new.clone(), v_new.clone())
            }
        }
    }
}

/// Concatenate along sequence dimension: [bh, len_a, dim] + [bh, len_b, dim] → [bh, len_a+len_b, dim]
/// GPU-resident — no CPU roundtrip via to_vec().
fn concat_seq(
    a: &Tensor,
    b: &Tensor,
    bh: usize,
    len_a: usize,
    len_b: usize,
    dim: usize,
) -> Tensor {
    let total_len = len_a + len_b;
    let out_buf = a.ctx.alloc_buffer(bh * total_len * dim * 4);

    // Copy a's data: src [bh, len_a, dim] → dst [bh, total_len, dim] at offset 0
    compute::gpu_strided_batch_copy(
        &a.ctx, &a.buffer, &out_buf,
        bh as u32, len_a as u32, total_len as u32, 0, dim as u32,
    );
    // Copy b's data: src [bh, len_b, dim] → dst [bh, total_len, dim] at offset len_a
    compute::gpu_strided_batch_copy(
        &a.ctx, &b.buffer, &out_buf,
        bh as u32, len_b as u32, total_len as u32, len_a as u32, dim as u32,
    );

    Tensor::from_buffer(Arc::clone(&a.ctx), out_buf, vec![bh, total_len, dim])
}

/// Expand KV heads for Grouped Query Attention.
/// Input: [n_kv_heads_total, seq, head_dim] where n_kv_heads_total = batch * n_kv_heads
/// Output: [n_heads_total, seq, head_dim] where n_heads_total = batch * n_heads
///
/// Each KV head is repeated `group_size` times contiguously:
///   output[h] = input[h / group_size]
///
/// This is a GPU buffer copy operation — each KV head's [seq, head_dim] block
/// is copied `group_size` times into the output.
fn repeat_kv(
    kv: &Tensor,
    n_kv_total: usize,
    seq_len: usize,
    head_dim: usize,
    group_size: usize,
) -> Tensor {
    let n_heads_total = n_kv_total * group_size;
    let head_block = seq_len * head_dim;
    let out_buf = kv.ctx.alloc_buffer(n_heads_total * head_block * 4);

    for h in 0..n_kv_total {
        let src_offset = h * head_block;
        for g in 0..group_size {
            let dst_offset = (h * group_size + g) * head_block;
            compute::gpu_buffer_copy(
                &kv.ctx,
                &kv.buffer,
                &out_buf,
                src_offset as u32,
                dst_offset as u32,
                head_block as u32,
            );
        }
    }

    let out_id = autograd::next_id();
    let out = Tensor {
        id: out_id,
        buffer: out_buf.clone(),
        shape: vec![n_heads_total, seq_len, head_dim],
        requires_grad: kv.requires_grad,
        ctx: Arc::clone(&kv.ctx),
    };

    // Record on tape: backward sums group_size gradient blocks into each KV head
    if kv.requires_grad || autograd::is_recording() {
        autograd::record(autograd::TapeEntry {
            op: autograd::Op::RepeatKv { n_kv_heads: n_kv_total, group_size, head_block },
            inputs: vec![kv.id],
            output: out_id,
            input_buffers: vec![kv.buffer.clone()],
            output_buffer: out_buf,
            shapes: vec![kv.shape.clone(), out.shape.clone()],
            cached: None,
        });
    }

    out
}
