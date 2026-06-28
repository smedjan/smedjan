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

use crate::gpu::MetalContext;
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

/// STRICT lower-triangular ones `[bh, n, n]`: `L[b,t,j] = 1` iff `j < t` (diagonal excluded).
fn strict_lower(ctx: &Arc<MetalContext>, bh: usize, n: usize) -> Tensor {
    let mut data = vec![0.0f32; bh * n * n];
    for t in 0..n {
        for j in 0..t {
            for b in 0..bh {
                data[(b * n + t) * n + j] = 1.0;
            }
        }
    }
    Tensor::from_slice(ctx, &data, vec![bh, n, n])
}

/// Chunked O(seq·chunk) SSM numerator (Mamba-2 / SSD form): mathematically IDENTICAL to
/// `ssm_decayed_numerator` (verified by `ssm_chunked_matches_materialized`) but avoids the full
/// O(seq²) score matrix. Splits the sequence into `nc = seq/chunk` chunks and sums:
///   • intra-chunk — the materialised SSM within each chunk (reuses the verified primitive), and
///   • inter-chunk — `o_t = g_t · (q_t · S_p)` where the running state `S_p = Σ_{p'<p} Λ(p',p)·KV_p'`
///     collapses the whole key history into an `[hd,hd]` matrix per chunk. `KV_p'[d,e]=Σ_i h·k·v`,
///     and the chunk-level decay matrix `Λ[p,p']=exp(CGs_p − CG_p')` (strictly lower) is applied as a
///     single batched `[nc,nc]@[nc,hd²]` matmul — no sequential scan.
/// Decay factorisation (exact): with chunk-local cumsum `AL`, `g=exp(AL)`, `h=exp(logγ−AL)`,
/// `logγ=AL[:,−1]`, one has `g_t·Λ(p',p)·h_j = exp(A_t−A_j)`. All `loga ≤ 0` ⇒ every exp ∈ (0,1]
/// (numerically safe). Requires `seq % chunk == 0`. `q,k,v:[bh,seq,hd]`, `loga:[bh,seq]`.
pub fn ssm_decayed_numerator_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    loga: &Tensor,
    chunk: usize,
) -> Tensor {
    let bh = q.shape[0];
    let seq = q.shape[1];
    let hd = q.shape[2];
    assert!(
        seq.is_multiple_of(chunk),
        "chunked SSM requires seq % chunk == 0 (got seq={seq}, chunk={chunk})"
    );
    let nc = seq / chunk;
    let bn = bh * nc;
    let ctx = Arc::clone(&q.ctx);

    // Chunk-major view: [bh, seq, hd] ≡ [bh*nc, chunk, hd] (contiguous, free reshape).
    let qc = q.reshape(vec![bn, chunk, hd]);
    let kc = k.reshape(vec![bn, chunk, hd]);
    let vc = v.reshape(vec![bn, chunk, hd]);
    let logac = loga.reshape(vec![bn, chunk]);

    // Intra-chunk: the materialised SSM over each chunk (j and t in the same chunk, j ≤ t).
    let o_intra = ssm_decayed_numerator(&qc, &kc, &vc, &logac); // [bn, chunk, hd]
    if nc == 1 {
        return o_intra.reshape(vec![bh, seq, hd]); // single chunk → no cross-chunk term
    }

    // Chunk-local inclusive cumsum AL[bn, chunk].
    let ltri_c = lower_tri_inclusive(&ctx, bn, chunk);
    let al = ltri_c
        .batched_matmul(&logac.reshape(vec![bn, chunk, 1]))
        .reshape(vec![bn, chunk]);

    // Per-token decays: g_t = exp(AL_t); h_j = exp(logγ − AL_j) with logγ = AL[:, chunk-1].
    let g = al.exp();
    let chunk_total = al.slice_cols(chunk - 1, 1); // logγ per chunk [bn, 1]
    let ones_1c = Tensor::ones(&ctx, vec![1, chunk]);
    let ct_bc = chunk_total.matmul(&ones_1c); // broadcast logγ over the chunk → [bn, chunk]
    let h = ct_bc.add(&al.scale(-1.0)).exp();

    // Per-chunk KV state: KV[d,e] = Σ_i h_i·k_i[d]·v_i[e] = (h⊙k)ᵀ @ v → [bn, hd, hd].
    let kh = kc
        .reshape(vec![bn * chunk, hd])
        .scale_rows(&h.reshape(vec![bn * chunk]))
        .reshape(vec![bn, chunk, hd]);
    let kv_chunks = kh
        .batched_matmul_trans_a(&vc)
        .reshape(vec![bh, nc, hd * hd]); // [bh, nc, hd²]

    // Chunk-level decay matrix Λ[bh, nc, nc]: Λ[p,p'] = exp(CGs_p − CG_p') for p' < p, else 0.
    let glog = chunk_total.reshape(vec![bh, nc]); // logγ per (b, p)
    let ltri_nc = lower_tri_inclusive(&ctx, bh, nc);
    let cg = ltri_nc
        .batched_matmul(&glog.reshape(vec![bh, nc, 1]))
        .reshape(vec![bh, nc]); // inclusive cumsum
    let cgs = cg.add(&glog.scale(-1.0)); // exclusive cumsum CGs_p = CG_p − logγ_p
    let ones_row = Tensor::ones(&ctx, vec![bh, 1, nc]);
    let ones_col = Tensor::ones(&ctx, vec![bh, nc, 1]);
    let cgs_col = cgs.reshape(vec![bh, nc, 1]).batched_matmul(&ones_row); // CGs_p across p'
    let cg_row = ones_col.batched_matmul(&cg.reshape(vec![bh, 1, nc])); // CG_p' across p
    let lam = cgs_col
        .add(&cg_row.scale(-1.0))
        .exp()
        .mul(&strict_lower(&ctx, bh, nc));

    // S_states[bh, nc, hd²] = Λ @ KV ; reshaped to [bn, hd, hd] (chunk-inner, aligns with qc).
    let s_states = lam.batched_matmul(&kv_chunks).reshape(vec![bn, hd, hd]);

    // Inter-chunk output: o_t = g_t · (q_t @ S_p).
    let o_inter = qc.batched_matmul(&s_states); // [bn, chunk, hd]
    let o_inter = o_inter
        .reshape(vec![bn * chunk, hd])
        .scale_rows(&g.reshape(vec![bn * chunk]))
        .reshape(vec![bn, chunk, hd]);

    o_intra.add(&o_inter).reshape(vec![bh, seq, hd])
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
    let hd = q.shape[2];
    let seq = q.shape[1];
    // Long sequences: the O(seq·chunk) chunked SSD form (verified identical to the materialised
    // O(seq²) form by `ssm_chunked_matches_materialized`) avoids the full seq×seq score matrix. Engage
    // only when it clearly wins and the seq divides evenly; small seq stays on the cheap materialised
    // path. Pick the largest chunk (64 then 32) that divides seq.
    let num = if seq >= 256 {
        match [64usize, 32].into_iter().find(|&c| seq.is_multiple_of(c)) {
            Some(chunk) => ssm_decayed_numerator_chunked(q, k, v, loga, chunk),
            None => ssm_decayed_numerator(q, k, v, loga),
        }
    } else {
        ssm_decayed_numerator(q, k, v, loga)
    };
    let unit = Tensor::ones(&q.ctx, vec![hd]);
    num.rms_norm(&unit, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd;
    use crate::gpu::MetalContext;

    fn ctx() -> Arc<MetalContext> {
        MetalContext::new()
    }

    /// CPU ground truth: A=cumsum(loga); o_t = Σ_{j≤t} exp(A_t−A_j)(q_t·k_j)v_j.
    fn cpu_ssm(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        loga: &[f32],
        bh: usize,
        seq: usize,
        hd: usize,
    ) -> Vec<f32> {
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
        let q: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.2)
            .collect();
        let k: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.2)
            .collect();
        let v: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.4)
            .collect();
        // log-decay ≤ 0 (per-step decay in (0,1]).
        let loga: Vec<f32> = (0..bh * seq)
            .map(|i| -0.1 - ((i % 4) as f32) * 0.3)
            .collect();

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
            assert!(
                (g - w).abs() <= 0.02 * (1.0 + w.abs()),
                "ssm mismatch at {idx}: gpu={g} cpu={w}"
            );
        }
    }

    /// The chunked O(seq·chunk) SSD form must be BIT-FOR-(fp)-EQUAL to the materialised O(seq²) form
    /// for every chunk size that divides seq — exercises intra-chunk + the cross-chunk Λ@KV state
    /// (nc>1 so the inter term is non-trivial), the strict-lower decay matrix, and the decay
    /// factorisation g·Λ·h == exp(A_t−A_j).
    #[test]
    fn ssm_chunked_matches_materialized() {
        let ctx = ctx();
        let (bh, seq, hd) = (2usize, 12usize, 4usize); // chunk ∈ {2,3,4,6} all give nc>1
        let n = bh * seq * hd;
        let mk_data = |s: usize| {
            (0..n)
                .map(|i| (((i * 7 + s * 13) % 17) as f32 - 8.0) * 0.1)
                .collect::<Vec<f32>>()
        };
        let loga: Vec<f32> = (0..bh * seq)
            .map(|i| -0.1 - ((i % 4) as f32) * 0.2)
            .collect();
        let (q, k, v) = (mk_data(1), mk_data(2), mk_data(3));
        autograd::no_grad(|| {
            let qt = Tensor::from_slice(&ctx, &q, vec![bh, seq, hd]);
            let kt = Tensor::from_slice(&ctx, &k, vec![bh, seq, hd]);
            let vt = Tensor::from_slice(&ctx, &v, vec![bh, seq, hd]);
            let lt = Tensor::from_slice(&ctx, &loga, vec![bh, seq]);
            let mat = ssm_decayed_numerator(&qt, &kt, &vt, &lt).to_vec();
            for &chunk in &[2usize, 3, 4, 6] {
                let chk = ssm_decayed_numerator_chunked(&qt, &kt, &vt, &lt, chunk).to_vec();
                let md = mat
                    .iter()
                    .zip(&chk)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    chk.iter().all(|x| x.is_finite()),
                    "chunk={chunk}: non-finite"
                );
                assert!(
                    md < 2e-3,
                    "chunk={chunk} (nc={}): chunked vs materialised max_diff={md:.6}",
                    seq / chunk
                );
            }
        });
    }

    /// Stronger decay (more negative logā) must down-weight distant past more — a sanity check
    /// that the selectivity actually attenuates: output at the last position with heavy decay
    /// differs from the no-decay case.
    #[test]
    fn decay_attenuates_history() {
        let ctx = ctx();
        let (bh, seq, hd) = (1usize, 6usize, 2usize);
        let q: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.3)
            .collect();
        let k: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i % 3) as f32 - 1.0) * 0.4)
            .collect();
        let v: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i % 4) as f32 - 1.5) * 0.5)
            .collect();

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
        let diff: f32 = near_zero
            .iter()
            .zip(strong.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-2,
            "decay had no effect on the output (selectivity broken)"
        );
    }

    /// Gradients flow end-to-end, including through the decay gate `loga` (via exp).
    #[test]
    fn gradient_flows_including_decay() {
        let ctx = ctx();
        let (bh, seq, hd) = (1usize, 4usize, 3usize);
        let n = bh * seq * hd;
        let q = Tensor::from_slice(
            &ctx,
            &(0..n)
                .map(|i| ((i * 9 % 17) as f32 - 8.0) * 0.1)
                .collect::<Vec<_>>(),
            vec![bh, seq, hd],
        )
        .with_grad();
        let k = Tensor::from_slice(
            &ctx,
            &(0..n)
                .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1)
                .collect::<Vec<_>>(),
            vec![bh, seq, hd],
        )
        .with_grad();
        let v = Tensor::from_slice(
            &ctx,
            &(0..n)
                .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2)
                .collect::<Vec<_>>(),
            vec![bh, seq, hd],
        )
        .with_grad();
        let loga = Tensor::from_slice(
            &ctx,
            &(0..bh * seq)
                .map(|i| -0.2 - (i % 3) as f32 * 0.2)
                .collect::<Vec<_>>(),
            vec![bh, seq],
        )
        .with_grad();

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
            assert!(
                gv.iter().all(|x| x.is_finite()),
                "non-finite grad for {name}"
            );
            assert!(
                gv.iter().any(|x| x.abs() > 1e-6),
                "all-zero grad for {name}"
            );
        }
        autograd::zero_grads();
    }
}

#[cfg(test)]
mod chunked_grad {
    use super::*;
    use crate::autograd;

    fn ctx() -> Arc<MetalContext> {
        MetalContext::new()
    }

    /// The chunked SSD backward must be IDENTICAL to the materialised backward for q/k/v/loga (the
    /// chunked form is an exact algebraic refactor, so the gradients must match). This is the right
    /// check rather than finite-diff: loga position 0 has a structurally-ZERO true gradient (it
    /// cancels from every causal decay A_t−A_j, being a uniform shift of A), so a central difference
    /// there is pure fp noise amplified by 1/2ε — it would spuriously fail a finite-diff check while
    /// the analytic gradient is correctly ~0. Comparing the two analytic backwards is noise-immune.
    #[test]
    fn ssm_chunked_grad_matches_materialized() {
        let ctx = ctx();
        let (bh, seq, hd, chunk) = (2usize, 12usize, 4usize, 4usize); // nc=3
        let n = bh * seq * hd;
        let mk_data = |s: usize| {
            (0..n)
                .map(|i| (((i * 7 + s * 13) % 17) as f32 - 8.0) * 0.1)
                .collect::<Vec<f32>>()
        };
        let la: Vec<f32> = (0..bh * seq)
            .map(|i| -0.2 - ((i % 4) as f32) * 0.2)
            .collect();
        let (qd, kd, vd) = (mk_data(1), mk_data(2), mk_data(3));

        // Return (dq, dk, dv, dloga) for either the materialised or the chunked numerator.
        let grads = |chunked: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            autograd::clear_tape();
            autograd::zero_grads();
            let q = Tensor::from_slice(&ctx, &qd, vec![bh, seq, hd]).with_grad();
            let k = Tensor::from_slice(&ctx, &kd, vec![bh, seq, hd]).with_grad();
            let v = Tensor::from_slice(&ctx, &vd, vec![bh, seq, hd]).with_grad();
            let l = Tensor::from_slice(&ctx, &la, vec![bh, seq]).with_grad();
            let out = if chunked {
                ssm_decayed_numerator_chunked(&q, &k, &v, &l, chunk)
            } else {
                ssm_decayed_numerator(&q, &k, &v, &l)
            };
            // Deterministic non-uniform seed so every output element contributes to the loss.
            let seed: Vec<f32> = (0..n)
                .map(|i| (((i * 13 + 5) % 11) as f32 - 5.0) * 0.1)
                .collect();
            let seed_t = Tensor::from_slice(&ctx, &seed, vec![n, 1]);
            let loss = out.reshape(vec![1, n]).matmul(&seed_t);
            autograd::backward(&ctx, loss.id);
            let fetch = |id: usize, len: usize| {
                Tensor::from_buffer(Arc::clone(&ctx), autograd::get_grad(id).unwrap(), vec![len])
                    .to_vec()
            };
            (
                fetch(q.id, n),
                fetch(k.id, n),
                fetch(v.id, n),
                fetch(l.id, bh * seq),
            )
        };
        let (mq, mk, mv, ml) = grads(false);
        let (cq, ck, cv, cl) = grads(true);
        let maxd = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        for (name, m, c) in [
            ("dq", &mq, &cq),
            ("dk", &mk, &ck),
            ("dv", &mv, &cv),
            ("dloga", &ml, &cl),
        ] {
            let d = maxd(m, c);
            assert!(
                c.iter().all(|x| x.is_finite()),
                "{name}: non-finite chunked grad"
            );
            assert!(
                d < 2e-3,
                "{name}: chunked vs materialised grad max_diff={d:.6}"
            );
        }
        autograd::clear_tape();
    }
}
