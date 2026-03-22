use crate::autograd::{self, Op, TapeEntry};
use crate::metal::{compute, GpuBuffer, MetalContext};
use crate::tensor::Tensor;
use objc2::rc::Retained;
use std::sync::Arc;

/// Multi-head attention with rotary positional encoding and KV cache support.
pub struct MultiHeadAttention {
    pub w_q: Tensor, // [d_model, d_model]
    pub w_k: Tensor, // [d_model, d_model]
    pub w_v: Tensor, // [d_model, d_model]
    pub w_o: Tensor, // [d_model, d_model]
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
}

/// KV cache for autoregressive inference.
pub struct KvCache {
    pub k: Option<Tensor>, // [batch * n_heads, cached_len, head_dim]
    pub v: Option<Tensor>, // [batch * n_heads, cached_len, head_dim]
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn cached_len(&self) -> usize {
        match &self.k {
            Some(k) => k.shape[1],
            None => 0,
        }
    }
}

impl MultiHeadAttention {
    pub fn new(ctx: &Arc<MetalContext>, d_model: usize, n_heads: usize) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let head_dim = d_model / n_heads;

        // Xavier initialization
        let std_dev = (2.0 / (d_model + d_model) as f32).sqrt();

        Self {
            w_q: Tensor::randn(ctx, vec![d_model, d_model], std_dev),
            w_k: Tensor::randn(ctx, vec![d_model, d_model], std_dev),
            w_v: Tensor::randn(ctx, vec![d_model, d_model], std_dev),
            w_o: Tensor::randn(ctx, vec![d_model, d_model], std_dev),
            n_heads,
            head_dim,
            d_model,
        }
    }

    /// Forward pass.
    /// x: [batch, seq_len, d_model]
    /// Returns: [batch, seq_len, d_model]
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
        let k = x_flat.matmul(&self.w_k);
        let v = x_flat.matmul(&self.w_v);

        // Transpose [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim]
        // This is a physical memory rearrangement that must go through the tape.
        let bh = batch * self.n_heads;
        let q = transpose_bsh_to_bhs(&q, batch, seq_len, self.n_heads, self.head_dim);
        let k = transpose_bsh_to_bhs(&k, batch, seq_len, self.n_heads, self.head_dim);
        let v = transpose_bsh_to_bhs(&v, batch, seq_len, self.n_heads, self.head_dim);

        // Apply RoPE to Q and K
        let offset = match &kv_cache {
            Some(cache) => cache.cached_len() as u32,
            None => 0,
        };
        let q = q.apply_rope(offset, 10000.0);
        let k = k.apply_rope(offset, 10000.0);

        // Handle KV cache (inference only — no tape needed)
        let (k_full, v_full, seq_k) = match kv_cache {
            Some(cache) => {
                let (k_full, v_full) = update_kv_cache(cache, &k, &v, bh, self.head_dim);
                let seq_k = k_full.shape[1];
                (k_full, v_full, seq_k)
            }
            None => (k, v, seq_len),
        };

        // --- Attention computation ---
        // Q: [bh, seq_q, head_dim], K: [bh, seq_k, head_dim], V: [bh, seq_k, head_dim]
        //
        // Strategy: flatten batch_heads into the first dimension of a 2D matmul.
        // For Q @ K^T: reshape Q to [bh*seq_q, head_dim], K to [bh*seq_k, head_dim]
        // But we need per-head matmuls, not one giant matmul.
        //
        // Correct approach: treat each head independently as a 2D matmul.
        // Flatten Q → [bh * seq_q, head_dim] and K → [bh * seq_k, head_dim]
        // won't work because matmul would mix heads.
        //
        // We must do per-head matmul. To keep this on the tape, we use the
        // batched matmul support in tensor.rs (which handles batch>1 via loop).
        // But that also uses CPU readbacks...
        //
        // SIMPLEST CORRECT APPROACH for training: for each batch-head, do a
        // tape-tracked 2D matmul. This creates tape entries per head which is
        // more tape entries but each one flows gradients correctly.

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_outs: Vec<Tensor> = Vec::with_capacity(bh);
        let head_size = self.head_dim;

        // q/k_full/v_full are [bh, seq, head_dim] — flat buffer of bh * seq * head_dim elements.
        // Use slice_flat to extract each head's data WITH tape tracking.
        for i in 0..bh {
            let q_offset = i * seq_len * head_size;
            let q_head = q.slice_flat(q_offset, seq_len * head_size, vec![seq_len, head_size]);

            let k_offset = i * seq_k * head_size;
            let k_head = k_full.slice_flat(k_offset, seq_k * head_size, vec![seq_k, head_size]);

            let v_offset = i * seq_k * head_size;
            let v_head = v_full.slice_flat(v_offset, seq_k * head_size, vec![seq_k, head_size]);

            // scores = Q @ K^T / sqrt(d_k) — tape-tracked
            let scores = q_head.matmul_trans_b(&k_head);
            let scores = scores.scale(scale);

            // Causal mask
            let scores_3d = scores.reshape(vec![1, seq_len, seq_k]);
            let masked = scores_3d.causal_mask(offset);
            let masked_2d = masked.reshape(vec![seq_len, seq_k]);

            // Softmax — tape-tracked
            let weights = masked_2d.softmax();

            // output = weights @ V — tape-tracked
            let head_out = weights.matmul(&v_head); // [seq_q, head_dim]
            attn_outs.push(head_out);
        }

        // Concatenate all head outputs using tape-tracked concat.
        // First concat to [bh * seq_q, head_dim], then transpose back.
        let attn_refs: Vec<&Tensor> = attn_outs.iter().collect();
        let attn_cat = Tensor::concat_flat(&attn_refs, vec![bh * seq_len, head_size]);

        // Transpose [bh, seq, head_dim] back to [batch*seq, d_model] using tape-tracked transpose
        let attn_3d = attn_cat.reshape(vec![bh, seq_len, head_size]);
        let attn_combined = transpose_bhs_to_bsh(&attn_3d, batch, seq_len, self.n_heads, head_size);

        // Output projection — goes through tape
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
    let data = t.to_vec();
    let bh = batch * n_heads;
    let mut out = vec![0.0f32; bh * seq_len * head_dim];

    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..n_heads {
                for d in 0..head_dim {
                    let src_idx = (b * seq_len + s) * n_heads * head_dim + h * head_dim + d;
                    let dst_idx = (b * n_heads + h) * seq_len * head_dim + s * head_dim + d;
                    out[dst_idx] = data[src_idx];
                }
            }
        }
    }

    let out_buf = t.ctx.buffer_from_slice(&out);
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
    let data = t.to_vec();
    let d_model = n_heads * head_dim;
    let mut out = vec![0.0f32; batch * seq_len * d_model];

    for b in 0..batch {
        for h in 0..n_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    let src_idx = (b * n_heads + h) * seq_len * head_dim + s * head_dim + d;
                    let dst_idx = (b * seq_len + s) * d_model + h * head_dim + d;
                    out[dst_idx] = data[src_idx];
                }
            }
        }
    }

    let out_buf = t.ctx.buffer_from_slice(&out);
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

    match (&cache.k, &cache.v) {
        (Some(k_old), Some(v_old)) => {
            let old_len = k_old.shape[1];
            let total_len = old_len + new_len;

            let k_full = concat_seq(k_old, k_new, bh, old_len, new_len, head_dim);
            let v_full = concat_seq(v_old, v_new, bh, old_len, new_len, head_dim);

            cache.k = Some(k_full.clone());
            cache.v = Some(v_full.clone());

            let _ = total_len;
            (k_full, v_full)
        }
        _ => {
            cache.k = Some(k_new.clone());
            cache.v = Some(v_new.clone());
            (k_new.clone(), v_new.clone())
        }
    }
}

/// Concatenate along sequence dimension: [bh, len_a, dim] + [bh, len_b, dim] → [bh, len_a+len_b, dim]
fn concat_seq(
    a: &Tensor,
    b: &Tensor,
    bh: usize,
    len_a: usize,
    len_b: usize,
    dim: usize,
) -> Tensor {
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let total_len = len_a + len_b;
    let mut out = vec![0.0f32; bh * total_len * dim];

    for i in 0..bh {
        let a_start = i * len_a * dim;
        let out_start = i * total_len * dim;
        out[out_start..out_start + len_a * dim]
            .copy_from_slice(&a_data[a_start..a_start + len_a * dim]);
        let b_start = i * len_b * dim;
        let out_offset = out_start + len_a * dim;
        out[out_offset..out_offset + len_b * dim]
            .copy_from_slice(&b_data[b_start..b_start + len_b * dim]);
    }

    Tensor::from_slice(&a.ctx, &out, vec![bh, total_len, dim])
}
