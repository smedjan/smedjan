//! Linear (kernelised) attention — softmax-free token mixing.
//!
//! Standard attention computes `softmax(Q Kᵀ / √d) V`, which materialises an N×N
//! score matrix and is therefore O(N²) in the sequence length N. Linear attention
//! replaces the softmax with a feature map φ and exploits associativity. In the
//! non-causal limit `φ(Q) (φ(K)ᵀ V)` runs in O(N); for a causal language model the
//! masked form is used:
//!
//! ```text
//!     Oᵢ = Σ_{j ≤ i} (φ(qᵢ) · φ(kⱼ)) vⱼ
//! ```
//!
//! This module provides the **masked reference** implementation. It materialises the
//! [N,N] score matrix (so it is O(N²) memory like softmax attention — the genuine
//! O(N) chunked recurrence lands in Stage B and is validated *against* this form).
//! Crucially it is composed **entirely from existing differentiable tensor ops**, so
//! the autograd tape supplies the backward pass for free — no hand-written backward
//! kernel that could silently disagree with the forward.
//!
//! ## Feature map
//! φ(x) = relu(x) + 1. Strictly positive (plain relu can zero an entire row, giving a
//! dead output and a zero denominator); monotonic; cheap; expressible from existing ops.
//!
//! ## Output normalisation
//! The numerator is RMS-normalised over the head dimension (a `rms_norm` with unit
//! weight) instead of being divided by the key-sum denominator `φ(qᵢ)·Σ_{j≤i} φ(kⱼ)`.
//! This avoids needing a GPU division/reciprocal kernel and is a standard stabiliser
//! for linear-attention / SSM token mixers (cf. RetNet's post-mix GroupNorm).

#![allow(dead_code)] // wired into the model in Stage C; until then only the tests exercise it.

use crate::metal::MetalContext;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Positive, **magnitude-bounded** feature map: φ(x) = rms_norm(relu(x) + 1) over the head dim.
///
/// `relu(x)+1` keeps φ strictly positive (a zero row would give a dead output). The RMS-norm
/// is the stability key: because linear attention here is denominator-free (the numerator is
/// normalised once at the end, not divided by Σφ(K) per position), an *unbounded* φ lets the
/// numerator `Σ_{j≤i}(φ(q_i)·φ(k_j))v_j` grow with the input scale and overflow fp16 inside the
/// GPU matmuls → NaN gradients on unlucky random inits. RMS-normalising φ caps it independent of
/// input scale (this is exactly QK-norm, which the softmax path also uses), so training is stable.
/// Built from existing differentiable ops → autograd backward for free.
fn feature_map(x: &Tensor) -> Tensor {
    let hd = *x.shape.last().expect("feature_map needs a non-empty shape");
    let ones = Tensor::full(&x.ctx, x.shape.clone(), 1.0);
    let phi = x.relu().add(&ones);
    let unit = Tensor::ones(&x.ctx, vec![hd]);
    phi.rms_norm(&unit, 1e-6)
}

/// Lower-triangular causal mask `[bh, rows, cols]`, replicated across the `bh` batch.
/// `mask[b,i,j] = 1` if `j ≤ i + row_offset` else `0`. Constant (no gradient needed).
fn causal_mask_tensor(
    ctx: &Arc<MetalContext>,
    bh: usize,
    rows: usize,
    cols: usize,
    row_offset: i64,
) -> Tensor {
    let mut data = vec![0.0f32; bh * rows * cols];
    for i in 0..rows {
        let limit = i as i64 + row_offset;
        for j in 0..cols {
            if (j as i64) <= limit {
                for b in 0..bh {
                    data[b * rows * cols + i * cols + j] = 1.0;
                }
            }
        }
    }
    Tensor::from_slice(ctx, &data, vec![bh, rows, cols])
}

/// Causal linear-attention **numerator**, masked O(N²) reference form (un-normalised).
/// `q,k,v: [bh, seq, hd]` → `[bh, seq, hd]`.
pub fn masked_reference(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    assert_eq!(q.shape.len(), 3, "linear attention expects [bh, seq, hd]");
    assert_eq!(q.shape, k.shape, "q and k must share shape");
    assert_eq!(q.shape[0], v.shape[0]);
    assert_eq!(q.shape[1], v.shape[1]);
    let bh = q.shape[0];
    let seq = q.shape[1];

    let qf = feature_map(q); // [bh, seq, hd]
    let kf = feature_map(k); // [bh, seq, hd]
    let scores = qf.batched_matmul_trans_b(&kf); // [bh, seq, seq] = φ(Q) φ(K)ᵀ
    let mask = causal_mask_tensor(&q.ctx, bh, seq, seq, 0);
    let masked = scores.mul(&mask); // zero the strictly-upper triangle
    masked.batched_matmul(v) // [bh, seq, hd]
}

/// Strictly-lower-triangular ones matrix `[bh, n, n]`: `L[b,c,c'] = 1` iff `c' < c`.
/// `L @ X` is the **exclusive** prefix sum of `X` along its first (chunk) axis.
fn strict_lower_tri(ctx: &Arc<MetalContext>, bh: usize, n: usize) -> Tensor {
    let mut data = vec![0.0f32; bh * n * n];
    for c in 0..n {
        for cp in 0..c {
            for b in 0..bh {
                data[(b * n + c) * n + cp] = 1.0;
            }
        }
    }
    Tensor::from_slice(ctx, &data, vec![bh, n, n])
}

/// Pick a chunk size that divides `seq`, closest to √seq from above (balances the
/// O(seq·C) intra cost against the O(seq·hd²/C) inter cost). Falls back to `seq`
/// (single chunk → exact masked form) when `seq` is prime.
fn pick_chunk(seq: usize) -> usize {
    let target = (seq as f64).sqrt() as usize;
    let target = target.max(1);
    let mut fallback = seq;
    for c in 1..=seq {
        if seq % c == 0 {
            if c >= target {
                return c;
            }
            fallback = c;
        }
    }
    fallback
}

/// Causal linear-attention NUMERATOR, **O(N) chunked-parallel** form (un-normalised).
///
/// Splits the sequence into `seq/C` chunks and computes, with no `[seq,seq]` matrix and
/// no sequential Python-style loop:
///   * intra-chunk: masked `φ(Q_c)φ(K_c)ᵀ V_c` over the small `[C,C]` block (parallel);
///   * inter-chunk: `φ(Q_c) · S_c`, where `S_c = Σ_{c'<c} φ(K_{c'})ᵀV_{c'}` is the
///     exclusive prefix sum of chunk summaries, computed as a strictly-lower-triangular
///     matmul `L·U` over the chunk axis.
///
/// Memory is O(seq·C + seq·hd²/C) = O(N) at `C≈hd`, vs O(N²) for the masked form.
/// Mathematically identical to `masked_reference` (validated by the equivalence test).
/// Requires `seq % chunk == 0`. `q,k,v: [bh, seq, hd]` → `[bh, seq, hd]`.
pub fn chunked(q: &Tensor, k: &Tensor, v: &Tensor, chunk: usize) -> Tensor {
    assert_eq!(q.shape.len(), 3, "linear attention expects [bh, seq, hd]");
    assert_eq!(q.shape, k.shape, "q and k must share shape");
    let bh = q.shape[0];
    let seq = q.shape[1];
    let hd = q.shape[2];
    assert!(chunk > 0 && seq % chunk == 0, "seq {seq} must be divisible by chunk {chunk}");
    let nc = seq / chunk;

    let qf = feature_map(q);
    let kf = feature_map(k);
    // Fold chunks into the batch dimension — a pure contiguous reshape ([b][c·C+r] ≡ [b·nc+c][r]).
    let qfb = qf.reshape(vec![bh * nc, chunk, hd]);
    let kfb = kf.reshape(vec![bh * nc, chunk, hd]);
    let vb = v.reshape(vec![bh * nc, chunk, hd]);

    // Intra-chunk, all chunks in parallel: [bh·nc, C, hd].
    let a = qfb.batched_matmul_trans_b(&kfb); // [bh·nc, C, C]
    let mask = causal_mask_tensor(&q.ctx, bh * nc, chunk, chunk, 0);
    let intra = a.mul(&mask).batched_matmul(&vb);

    // Per-chunk summary U_c = φ(K_c)ᵀ V_c, then exclusive prefix sum over chunks via L·U.
    let u = kfb.batched_matmul_trans_a(&vb); // [bh·nc, hd, hd]
    let u_flat = u.reshape(vec![bh, nc, hd * hd]);
    let l = strict_lower_tri(&q.ctx, bh, nc); // [bh, nc, nc]
    let s = l.batched_matmul(&u_flat).reshape(vec![bh * nc, hd, hd]); // S_c per chunk

    // Inter-chunk: φ(Q_c) · S_c → [bh·nc, C, hd]; add intra; restore [bh, seq, hd].
    let inter = qfb.batched_matmul(&s);
    inter.add(&intra).reshape(vec![bh, seq, hd])
}

/// Full linear-attention core: O(N) chunked numerator + RMS-norm over the head dimension.
/// The unit weight makes it a pure normalisation (no learned scale → no new parameters),
/// keeping checkpoints byte-compatible with the existing model.
/// `q,k,v: [bh, seq, hd]` → `[bh, seq, hd]`.
pub fn linear_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    let seq = q.shape[1];
    let hd = q.shape[2];
    let num = chunked(q, k, v, pick_chunk(seq)); // [bh, seq, hd], O(N) memory
    let unit = Tensor::ones(&q.ctx, vec![hd]);
    // eps=1.0 (not the usual 1e-6): early positions have a near-zero numerator, and a tiny eps
    // makes 1/rms blow up there → exploding gradients → training divergence. eps=1.0 keeps the
    // normalisation effective for normal-magnitude numerators while bounding the gradient for
    // tiny ones (normalise-when-large, pass-through-when-small) → stable training.
    num.rms_norm(&unit, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd;
    use crate::metal::MetalContext;

    fn ctx() -> Arc<MetalContext> {
        MetalContext::new()
    }

    /// CPU ground truth for the masked numerator: φ(x)=rms_norm(relu(x)+1) over hd, causal sum.
    /// q,k,v laid out [bh, seq, hd] row-major. Mirrors the GPU feature_map exactly.
    fn cpu_masked_reference(q: &[f32], k: &[f32], v: &[f32], bh: usize, seq: usize, hd: usize) -> Vec<f32> {
        let phi = |x: f32| x.max(0.0) + 1.0;
        // φ then RMS-norm over the head dim, per (b, position).
        let normfeat = |src: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; bh * seq * hd];
            for p in 0..bh * seq {
                let base = p * hd;
                let mut ss = 0.0f32;
                for d in 0..hd {
                    let f = phi(src[base + d]);
                    ss += f * f;
                }
                let inv = 1.0 / (ss / hd as f32 + 1e-6).sqrt();
                for d in 0..hd {
                    out[base + d] = phi(src[base + d]) * inv;
                }
            }
            out
        };
        let qn = normfeat(q);
        let kn = normfeat(k);
        let mut out = vec![0.0f32; bh * seq * hd];
        for b in 0..bh {
            for i in 0..seq {
                for j in 0..=i {
                    // score = φ̂(qᵢ) · φ̂(kⱼ)  (φ̂ = RMS-normalised φ)
                    let mut s = 0.0f32;
                    for d in 0..hd {
                        s += qn[(b * seq + i) * hd + d] * kn[(b * seq + j) * hd + d];
                    }
                    for d in 0..hd {
                        out[(b * seq + i) * hd + d] += s * v[(b * seq + j) * hd + d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn masked_reference_matches_cpu() {
        let ctx = ctx();
        let (bh, seq, hd) = (2usize, 4usize, 3usize);
        // Deterministic, mixed-sign inputs.
        let q: Vec<f32> = (0..bh * seq * hd).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3).collect();
        let k: Vec<f32> = (0..bh * seq * hd).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.25).collect();
        let v: Vec<f32> = (0..bh * seq * hd).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.5).collect();

        let got = autograd::no_grad(|| {
            let qt = Tensor::from_slice(&ctx, &q, vec![bh, seq, hd]);
            let kt = Tensor::from_slice(&ctx, &k, vec![bh, seq, hd]);
            let vt = Tensor::from_slice(&ctx, &v, vec![bh, seq, hd]);
            masked_reference(&qt, &kt, &vt).to_vec()
        });
        let want = cpu_masked_reference(&q, &k, &v, bh, seq, hd);

        assert_eq!(got.len(), want.len());
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 0.02 * (1.0 + w.abs()),
                "mismatch at {idx}: gpu={g} cpu={w}"
            );
        }
    }

    /// Causality: a future value must not affect a past output position.
    #[test]
    fn output_is_causal() {
        let ctx = ctx();
        let (bh, seq, hd) = (1usize, 5usize, 2usize);
        let q: Vec<f32> = (0..bh * seq * hd).map(|i| ((i % 5) as f32 - 2.0) * 0.4).collect();
        let k: Vec<f32> = (0..bh * seq * hd).map(|i| ((i % 3) as f32 - 1.0) * 0.6).collect();
        let mut v: Vec<f32> = (0..bh * seq * hd).map(|i| ((i % 4) as f32 - 1.5) * 0.5).collect();

        let out_a = autograd::no_grad(|| {
            let qt = Tensor::from_slice(&ctx, &q, vec![bh, seq, hd]);
            let kt = Tensor::from_slice(&ctx, &k, vec![bh, seq, hd]);
            let vt = Tensor::from_slice(&ctx, &v, vec![bh, seq, hd]);
            masked_reference(&qt, &kt, &vt).to_vec()
        });
        // Perturb the LAST position's value vector.
        for d in 0..hd {
            v[(seq - 1) * hd + d] += 3.0;
        }
        let out_b = autograd::no_grad(|| {
            let qt = Tensor::from_slice(&ctx, &q, vec![bh, seq, hd]);
            let kt = Tensor::from_slice(&ctx, &k, vec![bh, seq, hd]);
            let vt = Tensor::from_slice(&ctx, &v, vec![bh, seq, hd]);
            masked_reference(&qt, &kt, &vt).to_vec()
        });
        // Positions 0..seq-1 must be unchanged; only the last may differ.
        for i in 0..seq - 1 {
            for d in 0..hd {
                let idx = i * hd + d;
                assert!(
                    (out_a[idx] - out_b[idx]).abs() <= 1e-4,
                    "causality violated at pos {i}: {} vs {}",
                    out_a[idx],
                    out_b[idx]
                );
            }
        }
    }

    /// The O(N) chunked-parallel form must equal the O(N²) masked reference exactly
    /// (up to fp16 matmul tolerance), for every chunk size that divides the sequence —
    /// including the extremes C=seq (single chunk) and C=1 (pure recurrence).
    #[test]
    fn chunked_matches_masked_reference() {
        let ctx = ctx();
        let (bh, seq, hd) = (2usize, 12usize, 4usize);
        let q: Vec<f32> = (0..bh * seq * hd).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3).collect();
        let k: Vec<f32> = (0..bh * seq * hd).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.25).collect();
        let v: Vec<f32> = (0..bh * seq * hd).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.5).collect();

        autograd::no_grad(|| {
            let qt = Tensor::from_slice(&ctx, &q, vec![bh, seq, hd]);
            let kt = Tensor::from_slice(&ctx, &k, vec![bh, seq, hd]);
            let vt = Tensor::from_slice(&ctx, &v, vec![bh, seq, hd]);
            let reference = masked_reference(&qt, &kt, &vt).to_vec();
            // 12 = 1·12 = 2·6 = 3·4 = 4·3 = 6·2 = 12·1 — exercise several chunkings incl. C=1 and C=seq.
            for &c in &[1usize, 2, 3, 4, 6, 12] {
                let got = chunked(&qt, &kt, &vt, c).to_vec();
                for (idx, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
                    assert!(
                        (g - r).abs() <= 0.02 * (1.0 + r.abs()),
                        "chunk={c} idx={idx}: chunked={g} masked={r}"
                    );
                }
            }
        });
    }

    /// Gradients flow end-to-end through the composed core (autograd-for-free check).
    #[test]
    fn gradient_flows() {
        let ctx = ctx();
        let (bh, seq, hd) = (1usize, 3usize, 4usize);
        let n = bh * seq * hd;
        let mk = |scale: f32, off: f32| -> Vec<f32> {
            (0..n).map(|i| ((i * 9 % 17) as f32 - 8.0) * scale + off).collect()
        };
        let qt = Tensor::from_slice(&ctx, &mk(0.2, 0.1), vec![bh, seq, hd]).with_grad();
        let kt = Tensor::from_slice(&ctx, &mk(0.15, -0.1), vec![bh, seq, hd]).with_grad();
        let vt = Tensor::from_slice(&ctx, &mk(0.3, 0.0), vec![bh, seq, hd]).with_grad();

        let out = linear_attention(&qt, &kt, &vt); // [bh, seq, hd]
        // Reduce to a scalar so backward()'s size-1 seed is correct: loss = sum(out).
        let flat = out.reshape(vec![1, n]);
        let ones = Tensor::ones(&ctx, vec![n, 1]);
        let loss = flat.matmul(&ones); // [1, 1]
        autograd::backward(&ctx, loss.id);

        for (name, id) in [("q", qt.id), ("k", kt.id), ("v", vt.id)] {
            let g = autograd::get_grad(id).unwrap_or_else(|| panic!("no grad for {name}"));
            let gv = Tensor::from_buffer(Arc::clone(&ctx), g, vec![bh, seq, hd]).to_vec();
            assert!(gv.iter().all(|x| x.is_finite()), "non-finite grad for {name}");
            assert!(gv.iter().any(|x| x.abs() > 1e-6), "all-zero grad for {name}");
        }
        autograd::zero_grads();
    }
}
