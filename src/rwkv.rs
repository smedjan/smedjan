//! RWKV-6-style time mixing — attention-free, RNN-like recurrent token mixing.
//!
//! Two signature pieces:
//!
//! 1. **Token shift** — mix each token with the previous one before projection. The shift
//!    `x_{t-1}` (with `x_{-1}=0`) is a one-step causal shift, expressed here as a constant
//!    sub-diagonal matmul so it stays differentiable and composes from existing ops.
//!
//! 2. **WKV** — a per-channel decayed weighted sum (no QKᵀ matrix; each channel `d` evolves with
//!    its own decay `w[d]`):
//!
//! ```text
//!     wkv_t[d] = Σ_{i<t} exp(-(t-1-i)·w[d] + k_i[d]) v_i[d]  +  exp(u[d] + k_t[d]) v_t[d]
//! ```
//!
//! Computed stably as a per-channel decay matmul, with the channel axis folded into the batch so an
//! existing batched matmul does the work: with `ek = exp(k)`, `P = ek·v`, build the strict-lower
//! decay `D[(b,d),t,i] = exp(-(t-1-i)·w[d])` for `i < t` (zero otherwise). Every exponent is `≤ 0`
//! (decay `w > 0`), so `D ∈ (0, 1]` — no `exp(i·w)` factor that overflows fp16/loses fp32 precision.
//! Then `past_t[d] = Σ_{i<t} D[t,i,d]·P_i[d]` (a `[bh·hd,seq,seq] @ [bh·hd,seq,1]` batched matmul, the
//! channel `d` carried in the batch via the same `head_dim=1` transpose the SSM/attention paths use)
//! and `wkv_t = past_t + exp(u)·P_t`. Composed from existing ops + `exp`, so the autograd tape
//! supplies the backward for free. Validated against a CPU reference at short and long sequence.
//!
//! Earlier this used the algebraically-equivalent `exp(-t·w)·cumsum_i(exp(i·w)·P_i)` factorization,
//! which is exact in real arithmetic but collapses in fp32/fp16: `exp(i·w)` reaches `~e^63 ≈ 2e27` at
//! seq 64 with `w ≈ 1` (the trained decay is `exp(rwkv_w) ≈ 1`), overflowing fp16 matmul accumulators
//! and drowning early positions below fp32 epsilon — which pinned RWKV training loss flat. The direct
//! decay form above removes the blowup entirely.
//!
//! Note: this materialised form is `O(bh·hd·seq²)` memory (`≈ batch·dim·seq²`); for long sequences the
//! chunked/running-state scan is the production follow-up. Correctness and trainability are proven here.

#[cfg(test)]
use crate::gpu::MetalContext;
use crate::tensor::Tensor;
#[cfg(test)]
use std::sync::Arc;

/// One-step causal shift matrix `[bh, seq, seq]`: `Shift[t,i] = 1` iff `i == t-1`.
#[cfg(test)]
fn shift_matrix(ctx: &Arc<MetalContext>, bh: usize, seq: usize) -> Tensor {
    let mut data = vec![0.0f32; bh * seq * seq];
    for t in 1..seq {
        for b in 0..bh {
            data[(b * seq + t) * seq + (t - 1)] = 1.0;
        }
    }
    Tensor::from_slice(ctx, &data, vec![bh, seq, seq])
}

/// Broadcast a per-channel vector `[hd]` to `[bh, seq, hd]` (constant across batch & time),
/// differentiably, via `ones[bh·seq,1] @ vec[1,hd]`.
fn broadcast_hd(vec_hd: &Tensor, bh: usize, seq: usize) -> Tensor {
    let hd = vec_hd.shape[0];
    // Direct broadcast kernel (was ones[bh·seq,1] @ vec[1,hd] — a wasteful K=1 outer-product matmul).
    vec_hd.broadcast_rows(bh * seq).reshape(vec![bh, seq, hd])
}

/// Token shift: `out_t = x_{t-1}` (with `x_{-1}=0`). `x: [bh, seq, hd]` → `[bh, seq, hd]`.
#[cfg(test)]
pub fn token_shift(x: &Tensor) -> Tensor {
    let bh = x.shape[0];
    let seq = x.shape[1];
    shift_matrix(&x.ctx, bh, seq).batched_matmul(x)
}

/// RWKV WKV (numerator). `k,v: [bh, seq, hd]`; `w,u: [hd]` (decay rate `w>0`, bonus `u`).
/// → `[bh, seq, hd]`.
pub fn wkv(k: &Tensor, v: &Tensor, w: &Tensor, u: &Tensor) -> Tensor {
    assert_eq!(k.shape.len(), 3, "wkv expects [bh, seq, hd]");
    assert_eq!(k.shape, v.shape);
    let bh = k.shape[0];
    let seq = k.shape[1];
    let hd = k.shape[2];
    assert_eq!(w.shape, vec![hd]);
    assert_eq!(u.shape, vec![hd]);

    let ek = k.exp(); // exp(k) — RWKV weights values by exp(key)
    let p = ek.mul(v); // P = ek · v   [bh, seq, hd]

    // Strict-lower per-channel decay D[(b,d),t,i] = exp(-(t-1-i)·w[d]) for i < t, else 0.
    // negr[t,i] = -(t-1-i) and mask[t,i] = 1 over the strict lower triangle (constant, per-call upload
    // like the other mixers' index matrices). Folding channel d into the batch lets one batched matmul
    // do the per-channel contraction.
    let bhd = bh * hd;
    let mut negr = vec![0.0f32; seq * seq];
    let mut mask = vec![0.0f32; seq * seq];
    for t in 0..seq {
        for i in 0..t {
            negr[t * seq + i] = -((t - 1 - i) as f32);
            mask[t * seq + i] = 1.0;
        }
    }
    let negr_b = Tensor::from_slice(&k.ctx, &negr, vec![seq * seq]).broadcast_rows(bhd); // [bhd, seq*seq]
    let mask_t = Tensor::from_slice(&k.ctx, &mask, vec![seq * seq])
        .broadcast_rows(bhd)
        .reshape(vec![bhd, seq, seq]);

    // w tiled across the batch: w_tiled[b·hd + d] = w[d]; then E = -(t-1-i)·w[d] = negr · w_tiled per row.
    // All exponents ≤ 0 → exp ∈ (0,1], fp16-safe. The strict-lower mask zeroes i ≥ t (where negr=0 → exp=1).
    let w_tiled = w.broadcast_rows(bh).reshape(vec![bhd]); // [bh, hd] → [bh*hd]
    let d2 = negr_b
        .scale_rows(&w_tiled)
        .exp()
        .reshape(vec![bhd, seq, seq])
        .mul(&mask_t); // [bhd, seq, seq]

    // P with channel folded into batch: P2[(b,d),i,0] = P[b,i,d]; out2[(b,d),t,0] = Σ_{i<t} D·P2.
    let p2 = crate::attention::transpose_bsh_to_bhs(&p, bh, seq, hd, 1).reshape(vec![bhd, seq, 1]);
    let out2 = d2.batched_matmul(&p2); // [bhd, seq, 1]
                                       // transpose_bhs_to_bsh returns the flattened [bh·seq, hd]; reshape back to [bh, seq, hd].
    let past =
        crate::attention::transpose_bhs_to_bsh(&out2, bh, seq, hd, 1).reshape(vec![bh, seq, hd]);

    // Current-token bonus: exp(u[d])·P_t[d].
    let expu = broadcast_hd(&u.exp(), bh, seq);
    let current = expu.mul(&p);
    past.add(&current)
}

/// Full RWKV time-mix core: token-shift-mixed projections feeding WKV, with a SiLU receptance
/// gate, then an output RMS-norm (eps=1.0, the same well-conditioned norm as the other mixers).
/// Here r,k,v are already-projected `[bh, seq, hd]` (the block does the shift+projection); this
/// applies the receptance gate and WKV. `w,u: [hd]`.
#[cfg(test)]
pub fn time_mix(r: &Tensor, k: &Tensor, v: &Tensor, w: &Tensor, u: &Tensor) -> Tensor {
    let out = wkv(k, v, w, u); // [bh, seq, hd]
    let gated = r.silu().mul(&out); // SiLU receptance gate (sigmoid unavailable; silu is a sound gate)
    let hd = k.shape[2];
    let unit = Tensor::ones(&k.ctx, vec![hd]);
    gated.rms_norm(&unit, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd;
    use crate::gpu::MetalContext;

    fn ctx() -> Arc<MetalContext> {
        MetalContext::new()
    }

    #[test]
    fn token_shift_matches_cpu() {
        let ctx = ctx();
        let (bh, seq, hd) = (2usize, 4usize, 3usize);
        let x: Vec<f32> = (0..bh * seq * hd).map(|i| i as f32 * 0.1).collect();
        let got = autograd::no_grad(|| {
            token_shift(&Tensor::from_slice(&ctx, &x, vec![bh, seq, hd])).to_vec()
        });
        // out[b,t] = x[b,t-1], out[b,0] = 0
        for b in 0..bh {
            for t in 0..seq {
                for d in 0..hd {
                    let want = if t == 0 {
                        0.0
                    } else {
                        x[(b * seq + (t - 1)) * hd + d]
                    };
                    assert!((got[(b * seq + t) * hd + d] - want).abs() < 5e-3);
                }
            }
        }
    }

    /// CPU ground truth for the WKV numerator (RWKV-6 form).
    fn cpu_wkv(
        k: &[f32],
        v: &[f32],
        w: &[f32],
        u: &[f32],
        bh: usize,
        seq: usize,
        hd: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; bh * seq * hd];
        for b in 0..bh {
            for t in 0..seq {
                for d in 0..hd {
                    let mut s = 0.0f32;
                    for i in 0..t {
                        s += (-((t - 1 - i) as f32) * w[d] + k[(b * seq + i) * hd + d]).exp()
                            * v[(b * seq + i) * hd + d];
                    }
                    s += (u[d] + k[(b * seq + t) * hd + d]).exp() * v[(b * seq + t) * hd + d];
                    out[(b * seq + t) * hd + d] = s;
                }
            }
        }
        out
    }

    #[test]
    fn wkv_matches_cpu() {
        let ctx = ctx();
        let (bh, seq, hd) = (2usize, 5usize, 3usize);
        let k: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.15)
            .collect();
        let v: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.4)
            .collect();
        let w: Vec<f32> = (0..hd).map(|d| 0.3 + d as f32 * 0.2).collect(); // decay rate > 0
        let u: Vec<f32> = (0..hd).map(|d| 0.1 - d as f32 * 0.05).collect();

        let got = autograd::no_grad(|| {
            let kt = Tensor::from_slice(&ctx, &k, vec![bh, seq, hd]);
            let vt = Tensor::from_slice(&ctx, &v, vec![bh, seq, hd]);
            let wt = Tensor::from_slice(&ctx, &w, vec![hd]);
            let ut = Tensor::from_slice(&ctx, &u, vec![hd]);
            wkv(&kt, &vt, &wt, &ut).to_vec()
        });
        let want = cpu_wkv(&k, &v, &w, &u, bh, seq, hd);
        for (idx, (g, ww)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - ww).abs() <= 0.02 * (1.0 + ww.abs()),
                "wkv mismatch at {idx}: gpu={g} cpu={ww}"
            );
        }
    }

    /// The stable form must stay exact at the long-sequence / `w ≈ 1` regime where the old
    /// `exp(i·w)` factorization collapsed (`e^40 ≈ 2e17` overflows fp16, drowns early positions in
    /// fp32). The trained decay is `exp(rwkv_w) ≈ 1`, so this is the real operating point.
    #[test]
    fn wkv_stable_at_long_seq() {
        let ctx = ctx();
        let (bh, seq, hd) = (2usize, 40usize, 4usize);
        let k: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.3)
            .collect();
        let w: Vec<f32> = (0..hd).map(|d| 0.9 + d as f32 * 0.05).collect(); // ≈ exp(rwkv_w) ≈ 1
        let u: Vec<f32> = (0..hd).map(|d| 0.1 - d as f32 * 0.03).collect();

        let got = autograd::no_grad(|| {
            let kt = Tensor::from_slice(&ctx, &k, vec![bh, seq, hd]);
            let vt = Tensor::from_slice(&ctx, &v, vec![bh, seq, hd]);
            let wt = Tensor::from_slice(&ctx, &w, vec![hd]);
            let ut = Tensor::from_slice(&ctx, &u, vec![hd]);
            wkv(&kt, &vt, &wt, &ut).to_vec()
        });
        let want = cpu_wkv(&k, &v, &w, &u, bh, seq, hd);
        assert!(
            got.iter().all(|x| x.is_finite()),
            "non-finite wkv output at seq {seq}"
        );
        for (idx, (g, ww)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - ww).abs() <= 0.02 * (1.0 + ww.abs()),
                "wkv mismatch at {idx} (seq {seq}, w≈1): gpu={g} cpu={ww}"
            );
        }
    }

    /// Gradients flow end-to-end through token-shift, WKV, the decay w and bonus u, and the gate.
    #[test]
    fn gradient_flows() {
        let ctx = ctx();
        let (bh, seq, hd) = (1usize, 4usize, 3usize);
        let n = bh * seq * hd;
        let mk = |f: fn(usize) -> f32| (0..n).map(f).collect::<Vec<_>>();
        let x = Tensor::from_slice(
            &ctx,
            &mk(|i| ((i * 9 % 17) as f32 - 8.0) * 0.1),
            vec![bh, seq, hd],
        )
        .with_grad();
        let r = token_shift(&x).add(&x); // exercise token-shift in the graph
        let k = Tensor::from_slice(
            &ctx,
            &mk(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1),
            vec![bh, seq, hd],
        )
        .with_grad();
        let v = Tensor::from_slice(
            &ctx,
            &mk(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2),
            vec![bh, seq, hd],
        )
        .with_grad();
        let w = Tensor::from_slice(
            &ctx,
            &(0..hd).map(|d| 0.3 + d as f32 * 0.1).collect::<Vec<_>>(),
            vec![hd],
        )
        .with_grad();
        let u = Tensor::from_slice(
            &ctx,
            &(0..hd).map(|d| 0.1 * d as f32).collect::<Vec<_>>(),
            vec![hd],
        )
        .with_grad();

        let out = time_mix(&r, &k, &v, &w, &u);
        let ones = Tensor::ones(&ctx, vec![n, 1]);
        let loss = out.reshape(vec![1, n]).matmul(&ones);
        autograd::backward(&ctx, loss.id);

        for (name, id, shape) in [
            ("x(shift)", x.id, vec![bh, seq, hd]),
            ("k", k.id, vec![bh, seq, hd]),
            ("v", v.id, vec![bh, seq, hd]),
            ("w", w.id, vec![hd]),
            ("u", u.id, vec![hd]),
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
