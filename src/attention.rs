use crate::autograd::{self, Op, TapeEntry};
use crate::gpu::{compute, MetalContext};
use crate::tensor::Tensor;
use std::sync::Arc;

/// Token-mixing kind for the attention block. `Softmax` is standard scaled-dot-product
/// attention; `Linear` is O(N) softmax-free kernel attention (see `crate::linear_attention`).
/// Both share the identical Q/K/V/O projections, so switching adds no parameters and keeps
/// checkpoints byte-compatible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttnKind {
    Softmax,
    Linear,
    /// Selective state-space (Mamba-2/SSD) mixer: linear attention with an input-dependent
    /// per-head decay gate (see `crate::ssm`). Reuses the Q/K/V/O projections; adds one small
    /// decay-gate projection `ssm_loga`.
    Ssm,
    /// RWKV-style time mixing (see `crate::rwkv`): per-channel decayed WKV with a SiLU receptance
    /// gate (Q projection = receptance). Adds per-channel decay `rwkv_w` and bonus `rwkv_u`.
    /// Token-shift is omitted on this path (proven in the standalone module); the WKV recurrence
    /// is the core. Materialised form → short-sequence training (chunked stable form is follow-up).
    Rwkv,
    /// Multi-head Latent Attention (DeepSeek-V2/V3): K and V are reconstructed from a shared low-rank
    /// latent `c = x @ W_dkv` (dim d_c ≪ kv_dim) via up-projections `W_uk`, `W_uv`. The latent `c` is
    /// what an MLA KV cache stores → 10–50× cache shrink. Attention itself is standard softmax over
    /// the reconstructed K/V; the W_q/W_o projections are unchanged. (Decoupled-RoPE keys — the
    /// DeepSeek refinement that keeps RoPE out of the absorbed up-proj — are a documented follow-up;
    /// here RoPE is applied to reconstructed K as usual.) Requires attn_rank == 0.
    Mla,
    /// Block-sparse attention with learned top-k block routing (MoBA / DeepSeek-NSA family — the
    /// quality-preserving sparse attention behind subquadratic LLMs). K/V are split into blocks; each
    /// query attends only to its own block (causally) + the top-k PAST blocks scored by
    /// query · block-mean-key. Content-based + trainable end-to-end, so it keeps full-attention
    /// quality at a fraction of the attended positions. Reuses the standard Q/K/V/O projections (no
    /// new params). Full-sequence forward; uses `block_size` + `block_sparse_top_k`.
    BlockSparse,
}

/// SUBQUADRATIC block-sparse attention (training + inference). This GATHERS only the selected
/// key/value blocks and runs attention over them, so the score compute is O(n · (top_k+1) · block)
/// instead of O(n²) — the genuine subquadratic speedup. Per-query-block routing: each query block
/// attends its own block + the top_k past blocks by block-mean-query · block-mean-key. TRAINABLE:
/// the gather + attention math is recorded on the tape and the gather's scatter-add backward is
/// gradcheck-verified, so gradients flow to q/k/v (the routing selection itself is straight-through,
/// like MoE top-k). This is the path `AttnKind::BlockSparse` uses for training/prefill. The inline
/// `block_sparse_mask` path (used only as a seq % block != 0 fallback) is forward-correct but NOT
/// trainable — it records an `Op::Reshape` passthrough, so no gradient flows through block selection
/// and loss stays pinned at init. q/k/v: [bh, seq, hd] (expanded), seq % block == 0. Returns
/// [bh, seq, hd].
pub fn block_sparse_gather_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    block: usize,
    top_k: usize,
) -> Tensor {
    assert_eq!(
        q.shape.len(),
        3,
        "block_sparse_gather expects [bh, seq, hd]"
    );
    let (bh, seq, hd) = (q.shape[0], q.shape[1], q.shape[2]);
    assert_eq!(
        seq % block,
        0,
        "block_sparse_gather requires seq % block == 0"
    );
    let nb = seq / block;
    let k_sel = (top_k + 1).min(nb); // own block + up to top_k past; capped at nb
    let ctx = Arc::clone(&q.ctx);

    // ROUTING (block-mean → CPU top-k → `sel`) is non-differentiable and stays in `no_grad`:
    // straight-through, exactly like MoE top-k. `sel` is a fixed gather permutation that the
    // backward reuses to scatter gradients back to the selected source blocks.
    let sel_buf = autograd::no_grad(|| {
        let mean_q = q.block_mean_keys(block); // [bh, nb, hd]
        let mean_k = k.block_mean_keys(block);
        let block_scores = mean_q.batched_matmul_trans_b(&mean_k).to_vec(); // [bh, nb, nb]

        // CPU top-k selection per (head, query-block): slot 0 = own block, then top_k past blocks by
        // score; remaining slots = sentinel (nb) → padding (zero-gathered, fully masked).
        let mut sel = vec![nb as u32; bh * nb * k_sel];
        for bh_i in 0..bh {
            for qb in 0..nb {
                let base = (bh_i * nb + qb) * k_sel;
                sel[base] = qb as u32; // own block
                if qb > 0 && top_k > 0 {
                    let srow = (bh_i * nb + qb) * nb;
                    let mut past: Vec<(usize, f32)> =
                        (0..qb).map(|kb| (kb, block_scores[srow + kb])).collect();
                    past.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    for (i, &(kb, _)) in past.iter().take(top_k).enumerate() {
                        sel[base + 1 + i] = kb as u32;
                    }
                }
            }
        }
        ctx.buffer_from_u32_slice(&sel)
    });

    let dims = compute::GatherDims {
        bh: bh as u32,
        nb: nb as u32,
        seq: seq as u32,
        hd: hd as u32,
        block: block as u32,
        k_sel: k_sel as u32,
    };

    // Gather + attention math are RECORDED (outside `no_grad`) so gradients flow:
    // q → scores → q, v_sel → out → V, and ksel → scores → (scatter-add) → K. The gathers are
    // `Op::GatherBlocks`; the downstream reshape/matmul/scale/softmax record themselves as usual.
    let k_sel_t = gather_blocks_recorded(k, &sel_buf, dims); // [bh*nb, k_sel*block, hd]
    let v_sel_t = gather_blocks_recorded(v, &sel_buf, dims);

    // Attention over the gathered blocks: [bh*nb, block, hd] @ [bh*nb, sel_w, hd]^T.
    let q_bnq = q.reshape(vec![bh * nb, block, hd]);
    let scale = 1.0 / (hd as f32).sqrt();
    let scores = q_bnq.batched_matmul_trans_b(&k_sel_t).scale(scale); // [bh*nb, block, sel_w]
                                                                      // In-place causal mask sets out-of-causal/sentinel keys to -inf BEFORE softmax → ~0 weight and
                                                                      // ~0 gradient there. Safe under autograd: softmax caches its own output for backward, and the
                                                                      // upstream scale/matmul backwards read their inputs (q_bnq, k_sel_t), not this masked buffer.
    compute::gpu_gather_causal_mask(
        &ctx,
        &scores.buffer,
        &sel_buf,
        (bh * nb) as u32,
        nb as u32,
        block as u32,
        k_sel as u32,
    );
    let weights = scores.softmax();
    let out = weights.batched_matmul(&v_sel_t); // [bh*nb, block, hd]
    out.reshape(vec![bh, seq, hd])
}

/// Gather selected source blocks into a compact [bh*nb, k_sel*block, hd] tensor, recording an
/// `Op::GatherBlocks` tape entry (when recording) so the backward scatter-adds gradients back to
/// `src`. `sel_buf` is the fixed routing permutation (computed non-differentiably by the caller).
fn gather_blocks_recorded(
    src: &Tensor,
    sel_buf: &crate::gpu::BufU32,
    dims: compute::GatherDims,
) -> Tensor {
    let (bh, nb, hd, block, k_sel) = (
        dims.bh as usize,
        dims.nb as usize,
        dims.hd as usize,
        dims.block as usize,
        dims.k_sel as usize,
    );
    let sel_w = k_sel * block;
    let out_buf = src.ctx.alloc_buffer(bh * nb * sel_w * hd * 4);
    compute::gpu_gather_blocks(&src.ctx, &src.buffer, sel_buf, &out_buf, dims);
    let out = Tensor::from_buffer(Arc::clone(&src.ctx), out_buf, vec![bh * nb, sel_w, hd]);
    if autograd::is_recording() {
        autograd::record(TapeEntry {
            op: Op::GatherBlocks {
                bh: dims.bh,
                nb: dims.nb,
                seq: dims.seq,
                hd: dims.hd,
                block: dims.block,
                k_sel: dims.k_sel,
            },
            inputs: vec![src.id],
            output: out.id,
            input_buffers: vec![src.buffer.clone()],
            output_buffer: out.buffer.clone(),
            shapes: vec![src.shape.clone()],
            cached: Some(crate::gpu::u32_to_buf(sel_buf.clone())),
        });
    }
    out
}

/// Multi-head attention with rotary positional encoding, KV cache, and
/// Grouped Query Attention (GQA) support.
///
/// When `n_kv_heads < n_heads`, K and V projections are smaller (kv_dim instead
/// of d_model). Each KV head serves `group_size = n_heads / n_kv_heads` query heads.
/// When `n_kv_heads == n_heads`, this is standard Multi-Head Attention.
pub struct MultiHeadAttention {
    pub w_q: Tensor, // [d_model, d_model] or [d_model, rank] if low-rank
    pub w_k: Tensor, // [d_model, kv_dim] or [d_model, rank]
    pub w_v: Tensor, // [d_model, kv_dim] or [d_model, rank]
    pub w_o: Tensor, // [d_model, d_model] or [d_model, rank]
    // Low-rank: W = U × V. These are the V matrices (U is stored in w_q/k/v/o above)
    pub w_q_v: Tensor,    // [rank, d_model]
    pub w_k_v: Tensor,    // [rank, kv_dim]
    pub w_v_v: Tensor,    // [rank, kv_dim]
    pub w_o_v: Tensor,    // [rank, d_model]
    pub attn_rank: usize, // 0 = full rank
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
    pub rope_theta: f32,
    pub qk_norm_weight: Tensor, // [head_dim] — QK-norm weight (ones for fixed normalization)
    pub sliding_window: usize,  // 0=full causal, >0=attend only last W positions
    pub attn_kind: AttnKind,    // Softmax (default), Linear, Ssm, or Rwkv
    pub ssm_loga: Tensor, // [d_model, n_heads] — SSM per-head decay-gate projection (used iff Ssm)
    pub rwkv_w: Tensor, // [head_dim] — RWKV per-channel decay (rate = exp(rwkv_w) > 0; used iff Rwkv)
    pub rwkv_u: Tensor, // [head_dim] — RWKV per-channel current-token bonus (used iff Rwkv)
    pub mla_dc: usize,  // MLA latent dim d_c (0 = MLA off); used iff Mla
    pub w_dkv: Tensor, // [d_model, d_c] — MLA KV down-projection (its output is the cacheable latent)
    pub w_uk: Tensor,  // [d_c, kv_dim] — MLA key up-projection
    pub w_uv: Tensor,  // [d_c, kv_dim] — MLA value up-projection
    pub block_size: usize, // block-sparse attention block length (used iff BlockSparse)
    pub block_sparse_top_k: usize, // block-sparse attention: # past blocks attended per query (iff BlockSparse)
}

/// KV cache for autoregressive inference.
/// Pre-allocates to max_seq_len capacity to avoid O(n^2) reallocation
/// during autoregressive generation. Only new KV pairs are copied each step.
pub struct KvCache {
    pub k: Option<Tensor>, // [batch * n_heads, capacity, head_dim] (only [0..len] is valid)
    pub v: Option<Tensor>, // [batch * n_heads, capacity, head_dim]
    pub len: usize,        // current number of cached positions
    pub capacity: usize,   // pre-allocated max positions
    /// MLA latent cache: [batch, len, d_c]. When MLA is the mixer, the decode path caches this small
    /// shared latent (10–50× smaller than K/V) and reconstructs K/V from it each step. None otherwise.
    pub latent: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
            capacity: 0,
            latent: None,
        }
    }

    /// Create a pre-allocated KV cache with capacity for max_seq_len positions.
    pub fn with_capacity(
        ctx: &Arc<MetalContext>,
        batch_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        let total_floats = batch_heads * max_seq_len * head_dim;
        let k_buf = ctx.alloc_buffer(total_floats * 4);
        let v_buf = ctx.alloc_buffer(total_floats * 4);
        Self {
            k: Some(Tensor::from_buffer(
                Arc::clone(ctx),
                k_buf,
                vec![batch_heads, max_seq_len, head_dim],
            )),
            v: Some(Tensor::from_buffer(
                Arc::clone(ctx),
                v_buf,
                vec![batch_heads, max_seq_len, head_dim],
            )),
            len: 0,
            capacity: max_seq_len,
            latent: None,
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
        assert!(
            new_len <= self.len,
            "truncate: new_len {} > current len {}",
            new_len,
            self.len
        );
        // MLA latent cache: keep the first new_len latent rows (it's the source of truth for the MLA
        // decode path, independent of the K/V buffers). Rebuild as a contiguous [batch, new_len, d_c].
        if let Some(lat) = self.latent.take() {
            if new_len == 0 {
                self.latent = None;
            } else {
                let (b, old, dc, lctx) = (
                    lat.shape[0],
                    lat.shape[1],
                    lat.shape[2],
                    Arc::clone(&lat.ctx),
                );
                let buf = lctx.alloc_buffer(b * new_len * dc * 4);
                compute::gpu_compact_strided_copy(
                    &lctx,
                    &lat.buffer,
                    &buf,
                    b as u32,
                    new_len as u32,
                    old as u32,
                    dc as u32,
                );
                self.latent = Some(Tensor::from_buffer(lctx, buf, vec![b, new_len, dc]));
            }
        }
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
                    &ctx_clone,
                    &self.k.as_ref().unwrap().buffer,
                    &k_buf,
                    bh as u32,
                    new_len as u32,
                    old_len as u32,
                    head_dim as u32,
                );
                compute::gpu_compact_strided_copy(
                    &ctx_clone,
                    &self.v.as_ref().unwrap().buffer,
                    &v_buf,
                    bh as u32,
                    new_len as u32,
                    old_len as u32,
                    head_dim as u32,
                );

                self.k = Some(Tensor::from_buffer(
                    Arc::clone(&ctx_clone),
                    k_buf,
                    vec![bh, new_len, head_dim],
                ));
                self.v = Some(Tensor::from_buffer(
                    ctx_clone,
                    v_buf,
                    vec![bh, new_len, head_dim],
                ));
                self.len = new_len;
            }
        }
    }
}

impl MultiHeadAttention {
    pub fn new(
        ctx: &Arc<MetalContext>,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        rope_theta: f32,
    ) -> Self {
        Self::new_with_rank(ctx, d_model, n_heads, n_kv_heads, rope_theta, 0)
    }

    /// Create attention with scaled-down random init. scale × normal init std.
    pub fn new_scaled(
        ctx: &Arc<MetalContext>,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        rope_theta: f32,
        rank: usize,
        scale: f32,
    ) -> Self {
        let head_dim = d_model / n_heads;
        let kv_dim = head_dim * n_kv_heads;
        let z = || Tensor::zeros(ctx, vec![1]);

        let (w_q, w_k, w_v, w_o, w_q_v, w_k_v, w_v_v, w_o_v) = if rank > 0 {
            let u_std = (2.0 / (d_model + rank) as f32).sqrt() * scale;
            let vq_std = (2.0 / (rank + d_model) as f32).sqrt() * scale;
            let vk_std = (2.0 / (rank + kv_dim) as f32).sqrt() * scale;
            (
                Tensor::randn(ctx, vec![d_model, rank], u_std),
                Tensor::randn(ctx, vec![d_model, rank], u_std),
                Tensor::randn(ctx, vec![d_model, rank], u_std),
                Tensor::randn(ctx, vec![d_model, rank], u_std),
                Tensor::randn(ctx, vec![rank, d_model], vq_std),
                Tensor::randn(ctx, vec![rank, kv_dim], vk_std),
                Tensor::randn(ctx, vec![rank, kv_dim], vk_std),
                Tensor::randn(ctx, vec![rank, d_model], vq_std),
            )
        } else {
            let std_q = (2.0 / (d_model + d_model) as f32).sqrt() * scale;
            let std_kv = (2.0 / (d_model + kv_dim) as f32).sqrt() * scale;
            (
                Tensor::randn(ctx, vec![d_model, d_model], std_q),
                Tensor::randn(ctx, vec![d_model, kv_dim], std_kv),
                Tensor::randn(ctx, vec![d_model, kv_dim], std_kv),
                Tensor::randn(ctx, vec![d_model, d_model], std_q),
                z(),
                z(),
                z(),
                z(),
            )
        };
        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            w_q_v,
            w_k_v,
            w_v_v,
            w_o_v,
            attn_rank: rank,
            n_heads,
            n_kv_heads,
            head_dim,
            d_model,
            rope_theta,
            qk_norm_weight: Tensor::ones(ctx, vec![head_dim]),
            sliding_window: 0,
            attn_kind: AttnKind::Softmax,
            ssm_loga: Tensor::randn(
                ctx,
                vec![d_model, n_heads],
                (1.0 / d_model as f32).sqrt() * scale,
            ),
            rwkv_w: Tensor::randn(ctx, vec![head_dim], 0.01),
            rwkv_u: Tensor::randn(ctx, vec![head_dim], 0.01),
            mla_dc: 0,
            w_dkv: z(),
            w_uk: z(),
            w_uv: z(),
            block_size: 64,
            block_sparse_top_k: 0,
        }
    }

    pub fn new_with_rank(
        ctx: &Arc<MetalContext>,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        rope_theta: f32,
        rank: usize,
    ) -> Self {
        assert_eq!(d_model % n_heads, 0);
        assert!(n_kv_heads <= n_heads);
        assert_eq!(n_heads % n_kv_heads, 0);
        let head_dim = d_model / n_heads;
        let kv_dim = head_dim * n_kv_heads;

        let z = || Tensor::zeros(ctx, vec![1]);

        let (w_q, w_k, w_v, w_o, w_q_v, w_k_v, w_v_v, w_o_v) = if rank > 0 {
            let u_std = (2.0 / (d_model + rank) as f32).sqrt();
            let vq_std = (2.0 / (rank + d_model) as f32).sqrt();
            let vk_std = (2.0 / (rank + kv_dim) as f32).sqrt();
            (
                Tensor::randn(ctx, vec![d_model, rank], u_std),  // Q_U
                Tensor::randn(ctx, vec![d_model, rank], u_std),  // K_U
                Tensor::randn(ctx, vec![d_model, rank], u_std),  // V_U
                Tensor::randn(ctx, vec![d_model, rank], u_std),  // O_U
                Tensor::randn(ctx, vec![rank, d_model], vq_std), // Q_V
                Tensor::randn(ctx, vec![rank, kv_dim], vk_std),  // K_V
                Tensor::randn(ctx, vec![rank, kv_dim], vk_std),  // V_V
                Tensor::randn(ctx, vec![rank, d_model], vq_std), // O_V
            )
        } else {
            let std_q = (2.0 / (d_model + d_model) as f32).sqrt();
            let std_kv = (2.0 / (d_model + kv_dim) as f32).sqrt();
            (
                Tensor::randn(ctx, vec![d_model, d_model], std_q),
                Tensor::randn(ctx, vec![d_model, kv_dim], std_kv),
                Tensor::randn(ctx, vec![d_model, kv_dim], std_kv),
                Tensor::randn(ctx, vec![d_model, d_model], std_q),
                z(),
                z(),
                z(),
                z(),
            )
        };

        let qk_norm_weight = Tensor::ones(ctx, vec![head_dim]);

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            w_q_v,
            w_k_v,
            w_v_v,
            w_o_v,
            attn_rank: rank,
            n_heads,
            n_kv_heads,
            head_dim,
            d_model,
            rope_theta,
            qk_norm_weight,
            sliding_window: 0,
            attn_kind: AttnKind::Softmax,
            ssm_loga: Tensor::randn(ctx, vec![d_model, n_heads], (1.0 / d_model as f32).sqrt()),
            rwkv_w: Tensor::randn(ctx, vec![head_dim], 0.01),
            rwkv_u: Tensor::randn(ctx, vec![head_dim], 0.01),
            mla_dc: 0,
            w_dkv: z(),
            w_uk: z(),
            w_uv: z(),
            block_size: 64,
            block_sparse_top_k: 0,
        }
    }

    /// Enable Multi-head Latent Attention with latent dim `d_c`. Allocates the KV down-projection
    /// and the K/V up-projections, and switches the mixer to `AttnKind::Mla`. Requires attn_rank==0
    /// (MLA replaces the K/V projections; combining with low-rank Q/V is unsupported here).
    pub fn enable_mla(&mut self, ctx: &Arc<MetalContext>, d_c: usize) {
        assert!(d_c > 0, "MLA latent dim must be > 0");
        assert_eq!(
            self.attn_rank, 0,
            "MLA requires attn_rank == 0 (low-rank Q/V is unsupported with MLA)"
        );
        let kv_dim = self.head_dim * self.n_kv_heads;
        let down_std = (2.0 / (self.d_model + d_c) as f32).sqrt();
        let up_std = (2.0 / (d_c + kv_dim) as f32).sqrt();
        self.w_dkv = Tensor::randn(ctx, vec![self.d_model, d_c], down_std);
        self.w_uk = Tensor::randn(ctx, vec![d_c, kv_dim], up_std);
        self.w_uv = Tensor::randn(ctx, vec![d_c, kv_dim], up_std);
        self.mla_dc = d_c;
        self.attn_kind = AttnKind::Mla;
    }

    /// Enable block-sparse (MoBA/NSA-style) attention: each query attends to its own block + the
    /// top-k past blocks. Reuses the existing Q/K/V/O projections (no new params).
    pub fn enable_block_sparse(&mut self, top_k: usize, block_size: usize) {
        assert!(
            top_k > 0 && block_size > 0,
            "block-sparse needs top_k>0 and block_size>0"
        );
        self.block_sparse_top_k = top_k;
        self.block_size = block_size;
        self.attn_kind = AttnKind::BlockSparse;
    }

    /// MLA incremental decode caching the latent `c` (not K/V). Appends `c_new` to the latent cache,
    /// reconstructs K/V from the FULL cached latent (K=c@W_uk, V=c@W_uv), RoPEs at absolute positions,
    /// and attends the new tokens' queries causally. Cache stores `c` ([batch,total,d_c]) — 10–50×
    /// smaller than storing K/V. Mathematically equal to the prefill MLA forward (linearity + causal).
    fn mla_cached_forward(&self, x: &Tensor, cache: &mut KvCache) -> Tensor {
        let batch = x.shape[0];
        let new_seq = x.shape[1];
        let d_model = x.shape[2];
        let d_c = self.mla_dc;
        let hd = self.head_dim;
        let x_flat = x.reshape(vec![batch * new_seq, d_model]);

        let q = x_flat.matmul(&self.w_q); // [batch*new_seq, n_heads*hd]
        let c_new = x_flat
            .matmul(&self.w_dkv)
            .reshape(vec![batch, new_seq, d_c]);

        // Append c_new to the latent cache → c_all [batch, total, d_c].
        let old_len = cache.latent.as_ref().map_or(0, |c| c.shape[1]);
        let c_all = match cache.latent.take() {
            Some(c_old) => concat_seq(&c_old, &c_new, batch, old_len, new_seq, d_c),
            None => c_new,
        };
        let total = old_len + new_seq;

        // Reconstruct K/V from the FULL latent (per-token linear, so cached == prefill).
        let c_flat = c_all.reshape(vec![batch * total, d_c]);
        let k_all = c_flat.matmul(&self.w_uk); // [batch*total, kv_dim]
        let v_all = c_flat.matmul(&self.w_uv);
        cache.latent = Some(c_all);
        cache.len = total;

        // Transpose + RoPE: K at absolute positions 0..total, Q (new tokens) at old_len..total.
        let q = fused_transpose_rope(
            &q,
            batch,
            new_seq,
            self.n_heads,
            hd,
            old_len as u32,
            RopeParams::plain(self.rope_theta),
        );
        let k = fused_transpose_rope(
            &k_all,
            batch,
            total,
            self.n_kv_heads,
            hd,
            0,
            RopeParams::plain(self.rope_theta),
        );
        let v = transpose_bsh_to_bhs(&v_all, batch, total, self.n_kv_heads, hd);
        let q = if self.d_model >= 512 {
            q.rms_norm(&self.qk_norm_weight, 1e-6)
        } else {
            q
        };
        let k = if self.d_model >= 512 {
            k.rms_norm(&self.qk_norm_weight, 1e-6)
        } else {
            k
        };

        // GQA expand K/V heads to match Q heads.
        let group_size = self.n_heads / self.n_kv_heads;
        let bh_kv = batch * self.n_kv_heads;
        let (k, v) = if group_size > 1 {
            (
                repeat_kv(&k, bh_kv, total, hd, group_size),
                repeat_kv(&v, bh_kv, total, hd, group_size),
            )
        } else {
            (k, v)
        };

        // Causal attention: new queries (positions old_len..total) over all keys (0..total).
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = q.batched_matmul_trans_b(&k).scale(scale); // [bh, new_seq, total]
        let weights = scores.causal_mask(old_len as u32).softmax();
        let attn = weights.batched_matmul(&v); // [bh, new_seq, hd]

        let attn_combined = transpose_bhs_to_bsh(&attn, batch, new_seq, self.n_heads, hd);
        let out = attn_combined.matmul(&self.w_o);
        out.reshape(vec![batch, new_seq, d_model])
    }

    /// Forward pass with Grouped Query Attention support.
    /// x: [batch, seq_len, d_model]
    /// Returns: [batch, seq_len, d_model]
    ///
    /// When n_kv_heads < n_heads, K and V are projected to kv_dim = head_dim * n_kv_heads,
    /// then expanded via repeat_kv to match n_heads before attention computation.
    /// Thin wrapper: standard forward with no sequence packing.
    pub fn forward(&self, x: &Tensor, kv_cache: Option<&mut KvCache>) -> Tensor {
        self.forward_seg(x, kv_cache, None)
    }

    /// Forward with optional packed-sequence segment ids. When `seg_ids` is Some, the standard dense
    /// softmax path applies a causal + per-document mask so attention stays within each packed
    /// sequence. seg_ids is honored only on the standard MHA/GQA path (training/prefill, offset 0).
    /// The caller owns the seg buffer for the whole step — threaded by reference, NOT a thread-local
    /// (the cleared-thread-local-before-deferred-batch hazard reverted the first attempt; see
    /// HANDOFF_buffer_hazard_and_followups.md §2). Special mixers (MLA/linear/SSM/RWKV/block-sparse)
    /// keep their own masking and ignore it.
    pub fn forward_seg(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut KvCache>,
        seg_ids: Option<&crate::gpu::Buf>,
    ) -> Tensor {
        debug_assert!(
            seg_ids.is_none()
                || !matches!(
                    self.attn_kind,
                    AttnKind::Linear
                        | AttnKind::Ssm
                        | AttnKind::Rwkv
                        | AttnKind::BlockSparse
                        | AttnKind::Mla
                ),
            "seq-packing (seg_ids) is only honored on the standard MHA/GQA dense path"
        );
        // MLA incremental decode: cache the small latent `c` (not K/V) and reconstruct K/V from the
        // full cached latent each step → 10–50× smaller KV cache. Isolated early-return so the main
        // forward (training + non-MLA) is untouched. Equivalent to the prefill MLA forward (linearity
        // + causality), verified by the incremental==full test.
        let kv_cache = if self.attn_kind == AttnKind::Mla {
            match kv_cache {
                Some(cache) => return self.mla_cached_forward(x, cache),
                None => None,
            }
        } else {
            kv_cache
        };

        let batch = x.shape[0];
        let seq_len = x.shape[1];
        let d_model = x.shape[2];
        assert_eq!(d_model, self.d_model);

        // Flatten batch*seq for matmul: [batch*seq, d_model]
        let x_flat = x.reshape(vec![batch * seq_len, d_model]);

        // Project Q, K, V — separate matmuls (fewer dispatches than fused concat+slice)
        let (q, k, v) = if self.attn_kind == AttnKind::Mla {
            // MLA: K,V reconstructed from a shared low-rank latent c = x @ W_dkv (the cacheable
            // compression). Q/O unchanged. K,V keep the standard [n_tokens, kv_dim] shape, so the
            // rest of the pipeline (transpose, RoPE, GQA, softmax) is identical.
            let q = x_flat.matmul(&self.w_q);
            let (_c, k, v) = crate::mla::mla_kv(&x_flat, &self.w_dkv, &self.w_uk, &self.w_uv);
            (q, k, v)
        } else if self.attn_rank > 0 {
            (
                x_flat.matmul(&self.w_q).matmul(&self.w_q_v),
                x_flat.matmul(&self.w_k).matmul(&self.w_k_v),
                x_flat.matmul(&self.w_v).matmul(&self.w_v_v),
            )
        } else {
            (
                x_flat.matmul(&self.w_q),
                x_flat.matmul(&self.w_k),
                x_flat.matmul(&self.w_v),
            )
        };

        // Fused transpose + RoPE: [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim] with rotation.
        // Eliminates intermediate buffer and saves 2 dispatches per Q/K (transpose + RoPE → 1 each).
        let bh_kv = batch * self.n_kv_heads;
        let offset = match &kv_cache {
            Some(cache) => cache.cached_len() as u32,
            None => 0,
        };
        let q = fused_transpose_rope(
            &q,
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
            offset,
            RopeParams::plain(self.rope_theta),
        );
        let k = fused_transpose_rope(
            &k,
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
            offset,
            RopeParams::plain(self.rope_theta),
        );
        // V only needs transpose (no RoPE)
        let v = transpose_bsh_to_bhs(&v, batch, seq_len, self.n_kv_heads, self.head_dim);

        // QK-norm: only at d_model≥512 where attention entropy collapse is a real risk.
        // At d<512, the overhead (~4% throughput) isn't worth it.
        let q = if self.d_model >= 512 {
            q.rms_norm(&self.qk_norm_weight, 1e-6)
        } else {
            q
        };
        let k = if self.d_model >= 512 {
            k.rms_norm(&self.qk_norm_weight, 1e-6)
        } else {
            k
        };

        // Handle KV cache (inference only — no tape needed)
        // Cache stores n_kv_heads, not n_heads
        let (k_full, v_full) = match kv_cache {
            Some(cache) => update_kv_cache(cache, &k, &v, bh_kv, self.head_dim),
            None => (k, v),
        };

        // --- Attention computation ---
        let group_size = self.n_heads / self.n_kv_heads;
        let seq_k = k_full.shape[1];
        let bh = batch * self.n_heads;
        let seq_q_len = q.shape[1];

        // Linear (O(N) kernel) attention and the SSM mixer reuse the expanded, non-strided K/V layout.
        let linear = self.attn_kind == AttnKind::Linear;
        let ssm = self.attn_kind == AttnKind::Ssm;
        let rwkv = self.attn_kind == AttnKind::Rwkv;
        let block_sparse = self.attn_kind == AttnKind::BlockSparse;

        // GQA strided path: skip repeat_kv copy, use GQA-aware matmuls directly.
        // Only for inference (no tape) — training backward needs expanded K/V for gradient flow.
        // Block-sparse needs the expanded K/V (block-mean + dense scores), so it's excluded too.
        // The strided GQA matmuls are a Metal-only kernel; CUDA falls back to the repeat_kv path.
        let use_gqa_strided = cfg!(feature = "metal")
            && !linear
            && !ssm
            && !rwkv
            && !block_sparse
            && group_size > 1
            && !autograd::is_recording();

        // For training or MHA: expand KV heads to match Q heads
        let (k_for_attn, v_for_attn) = if use_gqa_strided {
            // Strided: pass unexpanded K/V, GQA matmuls handle indexing
            (k_full, v_full)
        } else if group_size > 1 {
            (
                repeat_kv(&k_full, bh_kv, seq_k, self.head_dim, group_size),
                repeat_kv(&v_full, bh_kv, seq_k, self.head_dim, group_size),
            )
        } else {
            (k_full, v_full)
        };

        // Linear attention: O(N) softmax-free kernel mixing over the full sequence.
        // Operates on the expanded [bh, seq, hd] Q/K/V (same layout as the standard path).
        // Decode-with-KV-cache (seq_q != seq_k) needs the recurrent state form — not yet
        // wired — so linear attention currently runs in the no-cache (training/prefill) path.
        let attn_cat = if linear {
            assert_eq!(
                seq_q_len, seq_k,
                "linear attention currently supports full-sequence forward only (seq_q == seq_k); \
                 incremental KV-cache decode needs the recurrent state path"
            );
            crate::linear_attention::linear_attention(&q, &k_for_attn, &v_for_attn)
        } else if ssm {
            assert_eq!(
                seq_q_len, seq_k,
                "SSM mixer currently supports full-sequence forward only (seq_q == seq_k)"
            );
            // Per-head, input-dependent log-decay gate: loga = -relu(x @ W_loga) ≤ 0 (decay ∈ (0,1]).
            // RoPE on Q/K is harmless here — the SSM's positional signal is carried by the decay.
            let loga_raw = x_flat.matmul(&self.ssm_loga); // [n_tokens, n_heads]
            let loga_bh = transpose_bsh_to_bhs(&loga_raw, batch, seq_len, self.n_heads, 1); // [bh, seq, 1]
            let loga = loga_bh.reshape(vec![bh, seq_q_len]).relu().scale(-1.0);
            crate::ssm::ssm(&q, &k_for_attn, &v_for_attn, &loga)
        } else if rwkv {
            assert_eq!(
                seq_q_len, seq_k,
                "RWKV mixer currently supports full-sequence forward only (seq_q == seq_k)"
            );
            // Per-channel WKV with decay rate exp(rwkv_w) > 0 and bonus rwkv_u; SiLU receptance gate
            // uses the Q projection as the receptance r. (Token-shift omitted on this path.)
            let actual_w = self.rwkv_w.exp();
            let wkv_out = crate::rwkv::wkv(&k_for_attn, &v_for_attn, &actual_w, &self.rwkv_u);
            q.silu().mul(&wkv_out)
        } else if block_sparse {
            assert_eq!(
                seq_q_len, seq_k,
                "block-sparse attention is full-sequence forward only (seq_q == seq_k)"
            );
            // MoBA/NSA: own block + top-k past blocks selected by block-mean score. Route training
            // and prefill through the trainable GATHER path (block_sparse_gather_attention): its
            // scatter-add backward is gradcheck-verified, so gradients flow. The inline
            // block_sparse_mask path below records only an Op::Reshape passthrough and does NOT train
            // (loss pinned at init). The gather path needs seq % block == 0; otherwise fall back to
            // the (forward-correct, non-training) mask path. NOTE: block-sparse training also relies
            // on the step-level pool bypass in train.rs — a residual pooled-mode buffer aliasing in
            // the gather path makes the buffer pool corrupt its gradients; pooled-mode is a follow-up.
            if seq_q_len % self.block_size == 0 {
                block_sparse_gather_attention(
                    &q,
                    &k_for_attn,
                    &v_for_attn,
                    self.block_size,
                    self.block_sparse_top_k,
                )
            } else {
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                let block_means = k_for_attn.block_mean_keys(self.block_size); // [bh, nb, hd], no tape
                let block_scores = q.batched_matmul_trans_b(&block_means); // [bh, seq, nb]
                let scores = q.batched_matmul_trans_b(&k_for_attn).scale(scale); // [bh, seq, seq]
                let masked = scores.block_sparse_mask(
                    &block_scores,
                    self.block_size,
                    self.block_sparse_top_k,
                );
                masked.softmax().batched_matmul(&v_for_attn)
            }
        } else if seq_q_len >= 2048 && !use_gqa_strided && seg_ids.is_none() {
            let attn_out_buf = q.ctx.alloc_buffer(bh * seq_q_len * self.head_dim * 4);
            compute::gpu_flash_attention_forward(
                &q.ctx,
                &q.buffer,
                &k_for_attn.buffer,
                &v_for_attn.buffer,
                &attn_out_buf,
                compute::FlashDims {
                    batch_heads: bh as u32,
                    seq_q: seq_q_len as u32,
                    seq_k: seq_k as u32,
                    head_dim: self.head_dim as u32,
                    kv_offset: offset,
                },
            );

            let attn_out_id = autograd::next_id();
            let attn = Tensor {
                id: attn_out_id,
                buffer: attn_out_buf.clone(),
                shape: vec![bh, seq_q_len, self.head_dim],
                requires_grad: q.requires_grad,
                ctx: Arc::clone(&q.ctx),
            };

            if autograd::is_recording() {
                autograd::record(autograd::TapeEntry {
                    op: autograd::Op::FlashAttention {
                        batch_heads: bh,
                        seq_q: seq_q_len,
                        seq_k,
                        head_dim: self.head_dim,
                        kv_offset: offset,
                    },
                    inputs: vec![q.id, k_for_attn.id, v_for_attn.id],
                    output: attn_out_id,
                    input_buffers: vec![
                        q.buffer.clone(),
                        k_for_attn.buffer.clone(),
                        v_for_attn.buffer.clone(),
                    ],
                    output_buffer: attn_out_buf.clone(),
                    shapes: vec![
                        q.shape.clone(),
                        k_for_attn.shape.clone(),
                        v_for_attn.shape.clone(),
                        attn.shape.clone(),
                    ],
                    cached: Some(attn_out_buf),
                });
            }
            attn
        } else if use_gqa_strided {
            // GQA strided: Q@K^T and attn@V use modular head indexing, no KV copy
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let scores_buf = q.ctx.alloc_buffer(bh * seq_q_len * seq_k * 4);
            compute::gpu_batched_matmul_gqa_trans_b(
                &q.ctx,
                &q.buffer,
                &k_for_attn.buffer,
                &scores_buf,
                compute::BatchedDims {
                    batch: bh as u32,
                    m: seq_q_len as u32,
                    n: seq_k as u32,
                    k: self.head_dim as u32,
                },
                group_size as u32,
            );
            let scores =
                Tensor::from_buffer(Arc::clone(&q.ctx), scores_buf, vec![bh, seq_q_len, seq_k]);
            let scores = scores.scale(scale);
            let scores = if self.sliding_window > 0 {
                scores.causal_mask_window(offset, self.sliding_window as u32)
            } else {
                scores.causal_mask(offset)
            };
            let weights = scores.softmax();
            let attn_buf = q.ctx.alloc_buffer(bh * seq_q_len * self.head_dim * 4);
            compute::gpu_batched_matmul_gqa(
                &q.ctx,
                &weights.buffer,
                &v_for_attn.buffer,
                &attn_buf,
                compute::BatchedDims {
                    batch: bh as u32,
                    m: seq_q_len as u32,
                    n: self.head_dim as u32,
                    k: seq_k as u32,
                },
                group_size as u32,
            );
            Tensor::from_buffer(
                Arc::clone(&q.ctx),
                attn_buf,
                vec![bh, seq_q_len, self.head_dim],
            )
        } else {
            // Fused scale+mask+softmax: 1 dispatch instead of 3
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let scores = q.batched_matmul_trans_b(&k_for_attn);
            let weights = if let Some(seg) = seg_ids {
                // Packed varlen (training/prefill, offset 0): causal + per-document mask keeps
                // attention inside each packed sequence. Falls back here (not the fused kernel)
                // because the doc mask needs the explicit [bh, seq, seq] scores.
                scores
                    .scale(scale)
                    .causal_doc_mask(seg, self.n_heads)
                    .softmax()
            } else if self.sliding_window > 0 {
                // Windowed attention can't use fused kernel — fall back to separate ops
                let scores = scores.scale(scale);
                let scores = scores.causal_mask_window(offset, self.sliding_window as u32);
                scores.softmax()
            } else {
                scores.scaled_causal_softmax(scale, offset)
            };
            weights.batched_matmul(&v_for_attn)
        };

        // Transpose [bh, seq, head_dim] back to [batch*seq, d_model]
        let attn_combined =
            transpose_bhs_to_bsh(&attn_cat, batch, seq_len, self.n_heads, self.head_dim);

        // Output projection
        let out = if self.attn_rank > 0 {
            attn_combined.matmul(&self.w_o).matmul(&self.w_o_v)
        } else {
            attn_combined.matmul(&self.w_o)
        };
        out.reshape(vec![batch, seq_len, d_model])
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        if self.attn_kind == AttnKind::Mla {
            // MLA: Q/O projections + QK-norm + the latent down/up projections. The direct w_k/w_v
            // are unused (K,V come from the latent), so they are NOT trained or checkpointed.
            return vec![
                &self.w_q,
                &self.w_o,
                &self.qk_norm_weight,
                &self.w_dkv,
                &self.w_uk,
                &self.w_uv,
            ];
        }
        let mut params = if self.attn_rank > 0 {
            vec![
                &self.w_q,
                &self.w_q_v,
                &self.w_k,
                &self.w_k_v,
                &self.w_v,
                &self.w_v_v,
                &self.w_o,
                &self.w_o_v,
            ]
        } else {
            vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o]
        };
        params.push(&self.qk_norm_weight);
        if self.attn_kind == AttnKind::Ssm {
            params.push(&self.ssm_loga); // trained only for SSM layers
        }
        if self.attn_kind == AttnKind::Rwkv {
            params.push(&self.rwkv_w); // per-channel decay
            params.push(&self.rwkv_u); // per-channel bonus
        }
        params
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RopeParams {
    pub theta: f32,
    pub yarn_scale: f32,
    pub yarn_orig_max: f32,
}

impl RopeParams {
    pub(crate) fn plain(theta: f32) -> Self {
        Self {
            theta,
            yarn_scale: 1.0,
            yarn_orig_max: 0.0,
        }
    }
}

/// Transpose [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim]
/// Records a tape entry so gradients flow through.
/// Fused transpose [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim] + RoPE.
/// Saves 2 dispatches (transpose + RoPE → 1) and eliminates the intermediate buffer.
pub(crate) fn fused_transpose_rope(
    t: &Tensor,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    offset: u32,
    rope: RopeParams,
) -> Tensor {
    let bh = batch * n_heads;
    let size = bh * seq_len * head_dim;
    let out_buf = t.ctx.alloc_buffer(size * 4);

    compute::gpu_transpose_rope(
        &t.ctx,
        &t.buffer,
        &out_buf,
        compute::TrRopeDims {
            batch: batch as u32,
            seq: seq_len as u32,
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            offset,
            theta: rope.theta,
            yarn_scale: rope.yarn_scale,
            yarn_orig_max: rope.yarn_orig_max,
        },
    );

    let out_id = autograd::next_id();
    let result = Tensor {
        id: out_id,
        buffer: out_buf.clone(),
        shape: vec![bh, seq_len, head_dim],
        requires_grad: false,
        ctx: Arc::clone(&t.ctx),
    };

    if t.requires_grad || autograd::is_recording() {
        autograd::record(TapeEntry {
            op: Op::TransposeRoPE {
                batch,
                seq_len,
                n_heads,
                head_dim,
                offset,
                theta: rope.theta,
                yarn_scale: rope.yarn_scale,
                yarn_orig_max: rope.yarn_orig_max,
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

pub(crate) fn transpose_bsh_to_bhs(
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
pub(crate) fn transpose_bhs_to_bsh(
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
        assert!(
            total_len <= cache.capacity,
            "KV cache overflow: {} + {} > {}",
            old_len,
            new_len,
            cache.capacity
        );

        // Copy new K, V into cache at offset = old_len (single batched dispatch per tensor)
        compute::gpu_strided_batch_copy(
            &k_cache.ctx,
            &k_new.buffer,
            &k_cache.buffer,
            compute::StridedCopyDims {
                bh: bh as u32,
                src_seq_len: new_len as u32,
                dst_stride: cache.capacity as u32,
                dst_offset: old_len as u32,
                dim: head_dim as u32,
            },
        );
        compute::gpu_strided_batch_copy(
            &v_cache.ctx,
            &v_new.buffer,
            &v_cache.buffer,
            compute::StridedCopyDims {
                bh: bh as u32,
                src_seq_len: new_len as u32,
                dst_stride: cache.capacity as u32,
                dst_offset: old_len as u32,
                dim: head_dim as u32,
            },
        );

        cache.len = total_len;

        // Return views that cover [0..total_len] of the cache.
        // We create tensors that reference sub-regions via buffer_copy to a contiguous buffer
        // because attention needs contiguous [bh, total_len, head_dim] layout, not strided.
        let k_view_buf = k_cache.ctx.alloc_buffer(bh * total_len * head_dim * 4);
        let v_view_buf = k_cache.ctx.alloc_buffer(bh * total_len * head_dim * 4);
        compute::gpu_compact_strided_copy(
            &k_cache.ctx,
            &k_cache.buffer,
            &k_view_buf,
            bh as u32,
            total_len as u32,
            cache.capacity as u32,
            head_dim as u32,
        );
        compute::gpu_compact_strided_copy(
            &v_cache.ctx,
            &v_cache.buffer,
            &v_view_buf,
            bh as u32,
            total_len as u32,
            cache.capacity as u32,
            head_dim as u32,
        );

        let k_full = Tensor::from_buffer(
            Arc::clone(&k_cache.ctx),
            k_view_buf,
            vec![bh, total_len, head_dim],
        );
        let v_full = Tensor::from_buffer(
            Arc::clone(&v_cache.ctx),
            v_view_buf,
            vec![bh, total_len, head_dim],
        );
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
fn concat_seq(a: &Tensor, b: &Tensor, bh: usize, len_a: usize, len_b: usize, dim: usize) -> Tensor {
    let total_len = len_a + len_b;
    let out_buf = a.ctx.alloc_buffer(bh * total_len * dim * 4);

    // Copy a's data: src [bh, len_a, dim] → dst [bh, total_len, dim] at offset 0
    compute::gpu_strided_batch_copy(
        &a.ctx,
        &a.buffer,
        &out_buf,
        compute::StridedCopyDims {
            bh: bh as u32,
            src_seq_len: len_a as u32,
            dst_stride: total_len as u32,
            dst_offset: 0,
            dim: dim as u32,
        },
    );
    // Copy b's data: src [bh, len_b, dim] → dst [bh, total_len, dim] at offset len_a
    compute::gpu_strided_batch_copy(
        &a.ctx,
        &b.buffer,
        &out_buf,
        compute::StridedCopyDims {
            bh: bh as u32,
            src_seq_len: len_b as u32,
            dst_stride: total_len as u32,
            dst_offset: len_a as u32,
            dim: dim as u32,
        },
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
pub fn repeat_kv(
    kv: &Tensor,
    n_kv_total: usize,
    seq_len: usize,
    head_dim: usize,
    group_size: usize,
) -> Tensor {
    let n_heads_total = n_kv_total * group_size;
    let head_block = seq_len * head_dim;
    let out_buf = kv.ctx.alloc_buffer(n_heads_total * head_block * 4);

    compute::gpu_repeat_kv(
        &kv.ctx,
        &kv.buffer,
        &out_buf,
        n_kv_total as u32,
        group_size as u32,
        seq_len as u32,
        head_dim as u32,
    );

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
            op: autograd::Op::RepeatKv {
                n_kv_heads: n_kv_total,
                group_size,
                seq_len,
                head_dim,
            },
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
