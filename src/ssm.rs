//! Selective state-space (Mamba-2 / SSD-style) token mixer.
//!
//! Mamba-2's *state-space duality* showed that a diagonal selective SSM is exactly
//! **decayed linear attention**: with a per-position, input-dependent log-decay `logā_t ≤ 0`
//! (so the per-step decay `exp(logā_t) ∈ (0,1]`), the scan
//!
//! ```text
//!     S_t = exp(logā_t)·S_{t-1} + k_tᵀ v_t ,   o_t = q_t · S_t
//! ```
//!
//! unrolls to
//!
//! ```text
//!     o_t = Σ_{j≤t} exp(A_t − A_j) (q_t·k_j) v_j ,   A_t = Σ_{s≤t} logā_s .
//! ```
//!
//! `S` is a fixed `[hd,hd]` state — there is **no KV cache that grows with the sequence**, which
//! is the whole point of the SSM family. The *selectivity* is that `logā` (and q/k/v) are produced
//! by input-dependent projections, so the model learns what to remember vs. forget per token.
//!
//! This module implements the materialised `O(seq²)` reference form (a `[seq,seq]` decay matrix),
//! composed entirely from existing differentiable ops + `exp` + `cumsum-via-triangular-matmul`, so
//! the autograd tape supplies the backward pass for free. It is validated against a CPU reference.
//! (The chunked `O(N)` SSD form — same trans_a state primitive as linear attention, plus a
//! decay-weighted chunk prefix — is the follow-up optimisation; the recurrence semantics are here.)

#![allow(dead_code)] // wired into the model via the block system; until then the tests exercise it.

use crate::metal::MetalContext;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Inclusive lower-triangular ones `[bh, n, n]`: `L[b,t,j] = 1` iff `j ≤ t`.
/// `L @ x` is the inclusive prefix sum (cumsum) of `x` along its first axis.
fn lower_tri_inclusive(ctx: &Arc<MetalContext>, bh: usize, n: usize) -> Tensor {
    let mut data = vec![0.0f32; bh * n * n];
    for t in 0..n {
        for j in 0..=t {
            for b in 0..bh {
                data[(b * n + t) * n + j] = 1.0;
            }
        }
    }
    Tensor::from_slice(ctx, &data, vec![bh, n, n])
}

/// Causal (j ≤ t) 0/1 mask `[bh, n, n]`.
fn causal_mask(ctx: &Arc<MetalContext>, bh: usize, n: usize) -> Tensor {
    lower_tri_inclusive(ctx, bh, n)
}

/// Selective SSM **numerator** (un-normalised), materialised `O(seq²)` form.
/// `q,k,v: [bh, seq, hd]`, `loga: [bh, seq]` (per-position log-decay, ≤ 0) → `[bh, seq, hd]`.
pub fn ssm_decayed_numerator(q: &Tensor, k: &Tensor, v: &Tensor, loga: &Tensor) -> Tensor {
    assert_eq!(q.shape.len(), 3, "ssm expects [bh, seq, hd]");
    assert_eq!(q.shape, k.shape);
    let bh = q.shape[0];
    let seq = q.shape[1];
    assert_eq!(loga.shape, vec![bh, seq], "loga must be [bh, seq]");

    // Cumulative log-decay A_t = Σ_{s≤t} loga_s, via inclusive-triangular matmul.
    let ltri = lower_tri_inclusive(&q.ctx, bh, seq);
    let loga3 = loga.reshape(vec![bh, seq, 1]);
    let a = ltri.batched_matmul(&loga3); // [bh, seq, 1]

    // Outer difference D[t,j] = A_t − A_j, built with ones-matmuls (no broadcast op needed).
    let ones_row = Tensor::ones(&q.ctx, vec![bh, 1, seq]);
    let ones_col = Tensor::ones(&q.ctx, vec![bh, seq, 1]);
    let a_col = a.batched_matmul(&ones_row); // [bh,seq,seq], entry = A_t (constant across j)
    let a_t = a.reshape(vec![bh, 1, seq]);
    let a_row = ones_col.batched_matmul(&a_t); // [bh,seq,seq], entry = A_j (constant across t)
    let diff = a_col.add(&a_row.scale(-1.0)); // A_t − A_j

    // Decay mask M[t,j] = exp(A_t − A_j) for j ≤ t, else 0.
    // For j>t this is exp(positive) (clamped in-kernel) but the causal mask zeroes it.
    let decay = diff.exp();
    let m = decay.mul(&causal_mask(&q.ctx, bh, seq));

    // o_t = Σ_{j≤t} M[t,j] (q_t·k_j) v_j
    let scores = q.batched_matmul_trans_b(k); // [bh,seq,seq] = q·k
    let weighted = scores.mul(&m);
    weighted.batched_matmul(v) // [bh, seq, hd]
}

/// Full SSM token mixer: selective decayed scan + RMS-norm over the head dim (eps=1.0, the same
/// well-conditioned output normalisation as the linear-attention path).
pub fn ssm(q: &Tensor, k: &Tensor, v: &Tensor, loga: &Tensor) -> Tensor {
    let num = ssm_decayed_numerator(q, k, v, loga);
    let hd = q.shape[2];
    let unit = Tensor::ones(&q.ctx, vec![hd]);
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

    /// CPU ground truth: A=cumsum(loga); o_t = Σ_{j≤t} exp(A_t−A_j)(q_t·k_j)v_j.
    fn cpu_ssm(q: &[f32], k: &[f32], v: &[f32], loga: &[f32], bh: usize, seq: usize, hd: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; bh * seq * hd];
        for b in 0..bh {
            // inclusive cumsum of loga
            let mut a = vec![0.0f32; seq];
            let mut acc = 0.0f32;
            for t in 0..seq {
                acc += loga[b * seq + t];
                a[t] = acc;
            }
            for t in 0..seq {
                for j in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[(b * seq + t) * hd + d] * k[(b * seq + j) * hd + d];
                    }
                    let w = (a[t] - a[j]).exp() * dot;
                    for d in 0..hd {
                        out[(b * seq + t) * hd + d] += w * v[(b * seq + j) * hd + d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn ssm_matches_cpu() {
        let ctx = ctx();
        let (bh, seq, hd) = (2usize, 5usize, 3usize);
        let q: Vec<f32> = (0..bh * seq * hd).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.2).collect();
        let k: Vec<f32> = (0..bh * seq * hd).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.2).collect();
        let v: Vec<f32> = (0..bh * seq * hd).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.4).collect();
        // log-decay ≤ 0 (per-step decay in (0,1]).
        let loga: Vec<f32> = (0..bh * seq).map(|i| -0.1 - ((i % 4) as f32) * 0.3).collect();

        let got = autograd::no_grad(|| {
            let qt = Tensor::from_slice(&ctx, &q, vec![bh, seq, hd]);
            let kt = Tensor::from_slice(&ctx, &k, vec![bh, seq, hd]);
            let vt = Tensor::from_slice(&ctx, &v, vec![bh, seq, hd]);
            let lt = Tensor::from_slice(&ctx, &loga, vec![bh, seq]);
            ssm_decayed_numerator(&qt, &kt, &vt, &lt).to_vec()
        });
        let want = cpu_ssm(&q, &k, &v, &loga, bh, seq, hd);
        assert_eq!(got.len(), want.len());
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() <= 0.02 * (1.0 + w.abs()), "ssm mismatch at {idx}: gpu={g} cpu={w}");
        }
    }

    /// Stronger decay (more negative logā) must down-weight distant past more — a sanity check
    /// that the selectivity actually attenuates: output at the last position with heavy decay
    /// differs from the no-decay case.
    #[test]
    fn decay_attenuates_history() {
        let ctx = ctx();
        let (bh, seq, hd) = (1usize, 6usize, 2usize);
        let q: Vec<f32> = (0..bh * seq * hd).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
        let k: Vec<f32> = (0..bh * seq * hd).map(|i| ((i % 3) as f32 - 1.0) * 0.4).collect();
        let v: Vec<f32> = (0..bh * seq * hd).map(|i| ((i % 4) as f32 - 1.5) * 0.5).collect();

        let run = |decay: f32| -> Vec<f32> {
            let loga = vec![decay; bh * seq];
            autograd::no_grad(|| {
                let qt = Tensor::from_slice(&ctx, &q, vec![bh, seq, hd]);
                let kt = Tensor::from_slice(&ctx, &k, vec![bh, seq, hd]);
                let vt = Tensor::from_slice(&ctx, &v, vec![bh, seq, hd]);
                let lt = Tensor::from_slice(&ctx, &loga, vec![bh, seq]);
                ssm_decayed_numerator(&qt, &kt, &vt, &lt).to_vec()
            })
        };
        let near_zero = run(-0.001); // almost no decay
        let strong = run(-2.0); // strong decay
        let diff: f32 = near_zero.iter().zip(strong.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-2, "decay had no effect on the output (selectivity broken)");
    }

    /// Gradients flow end-to-end, including through the decay gate `loga` (via exp).
    #[test]
    fn gradient_flows_including_decay() {
        let ctx = ctx();
        let (bh, seq, hd) = (1usize, 4usize, 3usize);
        let n = bh * seq * hd;
        let q = Tensor::from_slice(&ctx, &(0..n).map(|i| ((i * 9 % 17) as f32 - 8.0) * 0.1).collect::<Vec<_>>(), vec![bh, seq, hd]).with_grad();
        let k = Tensor::from_slice(&ctx, &(0..n).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1).collect::<Vec<_>>(), vec![bh, seq, hd]).with_grad();
        let v = Tensor::from_slice(&ctx, &(0..n).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2).collect::<Vec<_>>(), vec![bh, seq, hd]).with_grad();
        let loga = Tensor::from_slice(&ctx, &(0..bh * seq).map(|i| -0.2 - (i % 3) as f32 * 0.2).collect::<Vec<_>>(), vec![bh, seq]).with_grad();

        let out = ssm(&q, &k, &v, &loga); // [bh, seq, hd]
        let ones = Tensor::ones(&ctx, vec![n, 1]);
        let loss = out.reshape(vec![1, n]).matmul(&ones);
        autograd::backward(&ctx, loss.id);

        for (name, id, shape) in [
            ("q", q.id, vec![bh, seq, hd]),
            ("k", k.id, vec![bh, seq, hd]),
            ("v", v.id, vec![bh, seq, hd]),
            ("loga", loga.id, vec![bh, seq]),
        ] {
            let g = autograd::get_grad(id).unwrap_or_else(|| panic!("no grad for {name}"));
            let gv = Tensor::from_buffer(Arc::clone(&ctx), g, shape).to_vec();
            assert!(gv.iter().all(|x| x.is_finite()), "non-finite grad for {name}");
            assert!(gv.iter().any(|x| x.abs() > 1e-6), "all-zero grad for {name}");
        }
        autograd::zero_grads();
    }
}
