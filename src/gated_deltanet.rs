//! Gated DeltaNet — the linear-attention mixer used by Qwen3.5 / Qwen3-Next (24 of its 32 layers).
//!
//! Per head, with state matrix `S_t ∈ R^{d×d}` (key-dim × value-dim), the gated delta rule is:
//!
//! ```text
//!   r_t  = S_{t-1}ᵀ k_t                  (retrieve current value estimate for key k_t)
//!   Δv_t = β_t (v_t − r_t)               (delta correction, β_t ∈ (0,1] = write strength)
//!   S_t  = g_t · S_{t-1} + k_t Δv_tᵀ     (gated state update, g_t ∈ (0,1] = decay/forget)
//!   o_t  = S_tᵀ q_t                      (read with the query)
//! ```
//!
//! At `g_t = 1` this is exactly the (ungated) DeltaNet rule — the correctness anchor.
//!
//! Like `rwkv.rs`, this is a **materialized, parallel** formulation composed from existing
//! differentiable ops (so the autograd tape supplies the backward for free, on both Metal and
//! CUDA). smedjan has no per-timestep sequence slice, so we never loop over `t`; instead we unroll
//! the recurrence into full-sequence matrices.
//!
//! Unrolling `S_t = Σ_{i≤t} G_{i→t} k_i Δv_iᵀ` with the cumulative gate `G_{i→t} = Π_{i<j≤t} g_j`:
//!
//!   O = (tril≤(Dout ⊙ QKᵀ)) · ΔV                         …(1)  read-out
//!   ΔV = (I + A)⁻¹ (β ⊙ V),   A[i,j] = β_i·Dδ[i,j]·(k_i·k_j), strict-lower   …(2)  delta solve
//!
//! where `Dout[t,i] = exp(logc_t − logc_i)` (i ≤ t), `Dδ[i,j] = exp(logc_{i-1} − logc_j)` (j < i),
//! and `logc_t = Σ_{j≤t} log g_j` is the cumulative log-decay. We take **`log_g` (≤ 0) as input**
//! (the gate is produced in log space by the arch, same as RWKV's decay), so every decay exponent
//! is `≤ 0` → `exp ∈ (0,1]`, fp16-safe, no overflow. `A` is strictly-lower (nilpotent), so
//! `(I+A)⁻¹` is computed exactly by the finite iteration `ΔV ← βV − A·ΔV` (`seq` steps).
//!
//! O(bh·seq²·d) memory; the chunked running-state scan is the production follow-up. Correctness and
//! trainability are proven here against a CPU reference (see tests), exactly as RWKV/linear-attn were.

use crate::tensor::Tensor;
#[cfg(test)]
use crate::gpu::MetalContext;
#[cfg(test)]
use std::sync::Arc;

/// Lower-triangular ones `[seq,seq]` (incl. diagonal if `incl_diag`, else strict) tiled to `[bh,seq,seq]`.
fn tri_mask(ctx: &std::sync::Arc<MetalContext>, bh: usize, seq: usize, incl_diag: bool) -> Tensor {
    let mut m = vec![0.0f32; seq * seq];
    for t in 0..seq {
        for i in 0..seq {
            if (incl_diag && i <= t) || (!incl_diag && i < t) {
                m[t * seq + i] = 1.0;
            }
        }
    }
    Tensor::from_slice(ctx, &m, vec![seq * seq])
        .broadcast_rows(bh)
        .reshape(vec![bh, seq, seq])
}

/// Tile a per-(bh,seq) vector `v:[bh,seq]` into `[bh,seq,seq]` with **row t** = v_t (const across cols).
fn col_tile(v: &Tensor, bh: usize, seq: usize) -> Tensor {
    let ones_row = Tensor::ones(&v.ctx, vec![bh, 1, seq]);
    v.reshape(vec![bh, seq, 1]).batched_matmul(&ones_row) // [bh,seq,1]@[bh,1,seq] = [bh,seq,seq]
}

/// Tile a per-(bh,seq) vector `v:[bh,seq]` into `[bh,seq,seq]` with **col i** = v_i (const across rows).
fn row_tile(v: &Tensor, bh: usize, seq: usize) -> Tensor {
    let ones_col = Tensor::ones(&v.ctx, vec![bh, seq, 1]);
    ones_col.batched_matmul(&v.reshape(vec![bh, 1, seq])) // [bh,seq,1]@[bh,1,seq] = [bh,seq,seq]
}

fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    a.add(&b.scale(-1.0))
}

/// Gated delta rule (materialized). `q,k,v: [bh,seq,d]`; `log_g,beta: [bh,seq]` (log_g ≤ 0).
/// Returns `o: [bh,seq,d]`. Caller normalizes q/k and produces log_g/beta in the right ranges.
pub fn gated_delta_rule(q: &Tensor, k: &Tensor, v: &Tensor, log_g: &Tensor, beta: &Tensor) -> Tensor {
    assert_eq!(q.shape.len(), 3, "expect [bh,seq,d]");
    let (bh, seq, d) = (q.shape[0], q.shape[1], q.shape[2]);
    assert_eq!(k.shape, vec![bh, seq, d]);
    assert_eq!(v.shape, vec![bh, seq, d]);
    assert_eq!(log_g.shape, vec![bh, seq]);
    assert_eq!(beta.shape, vec![bh, seq]);

    // cumulative log-decay logc_t = Σ_{j≤t} log_g_j  (via lower-incl-tri matmul)
    let tri_incl = tri_mask(&q.ctx, bh, seq, true); // [bh,seq,seq]
    let logc = tri_incl
        .batched_matmul(&log_g.reshape(vec![bh, seq, 1]))
        .reshape(vec![bh, seq]); // [bh,seq]
    let logc_prev = sub(&logc, log_g); // logc_{t-1} = logc_t − log_g_t

    // KKᵀ and QKᵀ : [bh,seq,seq]
    let kk = k.batched_matmul_trans_b(k); // KK[t,i] = k_t·k_i
    let qk = q.batched_matmul_trans_b(k); // QK[t,i] = q_t·k_i

    // read-out decay Dout[t,i] = exp(logc_t − logc_i), masked i ≤ t
    let mask_incl = tri_mask(&q.ctx, bh, seq, true);
    let dout = sub(&col_tile(&logc, bh, seq), &row_tile(&logc, bh, seq))
        .exp()
        .mul(&mask_incl);
    let m_read = dout.mul(&qk).mul(&mask_incl); // (Dout ⊙ QKᵀ), lower-incl

    // delta-solve matrix A[i,j] = β_i · exp(logc_{i-1} − logc_j) · KK[i,j], strict-lower
    let mask_strict = tri_mask(&q.ctx, bh, seq, false);
    let ddelta = sub(&col_tile(&logc_prev, bh, seq), &row_tile(&logc, bh, seq))
        .exp()
        .mul(&mask_strict);
    let a_nodiag = ddelta.mul(&kk).mul(&mask_strict);
    // scale row i by β_i : reshape to [bh*seq, seq], scale_rows by beta[bh*seq]
    let beta_flat = beta.reshape(vec![bh * seq]);
    let a_mat = a_nodiag
        .reshape(vec![bh * seq, seq])
        .scale_rows(&beta_flat)
        .reshape(vec![bh, seq, seq]);

    // RHS = β ⊙ V : scale each value row v_i by β_i
    let rhs = v
        .reshape(vec![bh * seq, d])
        .scale_rows(&beta_flat)
        .reshape(vec![bh, seq, d]);

    // Solve (I+A) ΔV = RHS exactly: ΔV ← RHS − A·ΔV, `seq` iterations (A nilpotent ⇒ exact).
    let mut dv = rhs.clone();
    for _ in 0..seq {
        dv = sub(&rhs, &a_mat.batched_matmul(&dv));
    }

    // O = M_read · ΔV
    m_read.batched_matmul(&dv)
}

/// Shift `x:[bh,seq,c]` by `lag` timesteps into the past: `out[t]=x[t-lag]` (zero for `t<lag`).
fn shift_by(x: &Tensor, lag: usize, bh: usize, seq: usize, _c: usize) -> Tensor {
    if lag == 0 {
        return x.clone();
    }
    let mut s = vec![0.0f32; seq * seq];
    for t in lag..seq {
        s[t * seq + (t - lag)] = 1.0;
    }
    Tensor::from_slice(&x.ctx, &s, vec![seq * seq])
        .broadcast_rows(bh)
        .reshape(vec![bh, seq, seq])
        .batched_matmul(x)
}

/// Depthwise **causal conv1d** (Qwen3.5 `linear_conv_kernel_dim`). `x:[bh,seq,c]`, `kernel:[c,kw]`
/// (`kernel[c,kw-1]` = current-tap weight). `out[b,t,c] = Σ_j kernel[c,j]·x[b, t-(kw-1)+j, c]`.
pub fn causal_conv1d(x: &Tensor, kernel: &Tensor, kw: usize) -> Tensor {
    let (bh, seq, c) = (x.shape[0], x.shape[1], x.shape[2]);
    assert_eq!(kernel.shape, vec![c, kw]);
    let mut acc: Option<Tensor> = None;
    for j in 0..kw {
        let lag = kw - 1 - j;
        let shifted = shift_by(x, lag, bh, seq, c);
        let kj = kernel.slice_cols(j, 1).reshape(vec![c]); // kernel[:,j]
        let kj_b = kj.broadcast_rows(bh * seq).reshape(vec![bh, seq, c]);
        let term = shifted.mul(&kj_b);
        acc = Some(match acc {
            None => term,
            Some(a) => a.add(&term),
        });
    }
    acc.unwrap()
}

/// Constant selection matrix `[in_dim, out_dim]` with `M[i, base+i] = 1` (scatter a slice into cols).
fn select_into(
    ctx: &std::sync::Arc<MetalContext>,
    in_dim: usize,
    out_dim: usize,
    base: usize,
) -> Tensor {
    let mut m = vec![0.0f32; in_dim * out_dim];
    for i in 0..in_dim {
        m[i * out_dim + (base + i)] = 1.0;
    }
    Tensor::from_slice(ctx, &m, vec![in_dim, out_dim])
}

/// **Partial RoPE** (Qwen3.5 `partial_rotary_factor`): rotate only the first `rot_dim` of `head_dim`,
/// pass the rest through unchanged. Reassembled with constant selection matmuls (concat_flat is a
/// flat append, not a per-row column concat).
pub fn partial_rope(x: &Tensor, rot_dim: usize, offset: u32, theta: f32) -> Tensor {
    let (bh, seq, hd) = (x.shape[0], x.shape[1], x.shape[2]);
    if rot_dim >= hd {
        return x.apply_rope(offset, theta);
    }
    let x2 = x.reshape(vec![bh * seq, hd]);
    let xr = x2
        .slice_cols(0, rot_dim)
        .reshape(vec![bh, seq, rot_dim])
        .apply_rope(offset, theta)
        .reshape(vec![bh * seq, rot_dim]);
    let xp = x2.slice_cols(rot_dim, hd - rot_dim);
    let p_rot = select_into(&x.ctx, rot_dim, hd, 0);
    let p_pass = select_into(&x.ctx, hd - rot_dim, hd, rot_dim);
    xr.matmul(&p_rot)
        .add(&xp.matmul(&p_pass))
        .reshape(vec![bh, seq, hd])
}

/// Sigmoid with no new kernel: `sigmoid(x) = softmax([x, 0])[..,0]`. Row-wise softmax over a 2-wide
/// matrix gives `[σ(x), 1−σ(x)]`; take column 0. Composed-ops → autograd backward for free.
pub fn sigmoid(x: &Tensor) -> Tensor {
    let n = x.numel();
    let e = Tensor::from_slice(&x.ctx, &[1.0, 0.0], vec![1, 2]); // x·[1,0] → [x, 0] per row
    x.reshape(vec![n, 1])
        .matmul(&e)
        .softmax()
        .slice_cols(0, 1)
        .reshape(x.shape.clone())
}

/// Qwen3.5 attention **output gate**: `out ⊙ sigmoid(gate)`.
pub fn output_gate(out: &Tensor, gate: &Tensor) -> Tensor {
    out.mul(&sigmoid(gate))
}

/// GQA head expansion for the asymmetric DeltaNet (Qwen3.5: 16 key heads → 32 value heads).
/// Repeat each of `n_in` head-blocks `repeat` times along the feature axis via a constant
/// block-selection matmul. `x:[rows, n_in*d]` → `[rows, n_in*repeat*d]`; out head `h*repeat+r` == in `h`.
pub fn expand_heads(x: &Tensor, n_in: usize, repeat: usize, d: usize) -> Tensor {
    if repeat == 1 {
        return x.clone();
    }
    let rows = x.shape[0];
    assert_eq!(x.shape, vec![rows, n_in * d]);
    let out_dim = n_in * repeat * d;
    let mut e = vec![0.0f32; (n_in * d) * out_dim];
    for h in 0..n_in {
        for r in 0..repeat {
            for j in 0..d {
                e[(h * d + j) * out_dim + (h * repeat + r) * d + j] = 1.0;
            }
        }
    }
    x.matmul(&Tensor::from_slice(&x.ctx, &e, vec![n_in * d, out_dim]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd;

    fn ctx() -> Arc<MetalContext> {
        MetalContext::new()
    }

    /// CPU ground truth: the literal sequential recurrence.
    #[allow(clippy::too_many_arguments)]
    fn cpu_gdr(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        log_g: &[f32],
        beta: &[f32],
        bh: usize,
        seq: usize,
        d: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; bh * seq * d];
        for b in 0..bh {
            let mut s = vec![0.0f32; d * d]; // S[key,val], row=key
            for t in 0..seq {
                let g = (log_g[b * seq + t]).exp();
                let bet = beta[b * seq + t];
                let koff = (b * seq + t) * d;
                // r = Sᵀ k  (r[val] = Σ_key S[key,val] k[key])
                let mut r = vec![0.0f32; d];
                for val in 0..d {
                    let mut acc = 0.0;
                    for key in 0..d {
                        acc += s[key * d + val] * k[koff + key];
                    }
                    r[val] = acc;
                }
                // Δv = β (v − r)
                let mut dv = vec![0.0f32; d];
                for val in 0..d {
                    dv[val] = bet * (v[koff + val] - r[val]);
                }
                // S = g·S + k Δvᵀ
                for key in 0..d {
                    for val in 0..d {
                        s[key * d + val] = g * s[key * d + val] + k[koff + key] * dv[val];
                    }
                }
                // o = Sᵀ q
                for val in 0..d {
                    let mut acc = 0.0;
                    for key in 0..d {
                        acc += s[key * d + val] * q[koff + key];
                    }
                    out[koff + val] = acc;
                }
            }
        }
        out
    }

    fn run(bh: usize, seq: usize, d: usize, gconst: f32) -> (Vec<f32>, Vec<f32>) {
        let ctx = ctx();
        let n = bh * seq * d;
        let q: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.12).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.3).collect();
        let log_g: Vec<f32> = (0..bh * seq).map(|_| gconst.ln()).collect();
        let beta: Vec<f32> = (0..bh * seq).map(|i| 0.3 + (i % 5) as f32 * 0.12).collect();
        let got = autograd::no_grad(|| {
            gated_delta_rule(
                &Tensor::from_slice(&ctx, &q, vec![bh, seq, d]),
                &Tensor::from_slice(&ctx, &k, vec![bh, seq, d]),
                &Tensor::from_slice(&ctx, &v, vec![bh, seq, d]),
                &Tensor::from_slice(&ctx, &log_g, vec![bh, seq]),
                &Tensor::from_slice(&ctx, &beta, vec![bh, seq]),
            )
            .to_vec()
        });
        let want = cpu_gdr(&q, &k, &v, &log_g, &beta, bh, seq, d);
        (got, want)
    }

    #[test]
    fn matches_cpu_gated() {
        let (got, want) = run(2, 6, 4, 0.9);
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 0.02 * (1.0 + w.abs()),
                "mismatch at {idx}: gpu={g} cpu={w}"
            );
        }
    }

    #[test]
    fn reduces_to_deltanet_at_g1() {
        // g = 1 (log_g = 0) must equal the ungated delta rule.
        let (got, want) = run(2, 5, 4, 1.0);
        assert!(got.iter().all(|x| x.is_finite()));
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= 0.02 * (1.0 + w.abs()));
        }
    }

    #[test]
    fn gradient_flows() {
        let ctx = ctx();
        let (bh, seq, d) = (1usize, 4usize, 3usize);
        let n = bh * seq * d;
        let mk = |f: fn(usize) -> f32, len: usize| (0..len).map(f).collect::<Vec<_>>();
        let q = Tensor::from_slice(&ctx, &mk(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1, n), vec![bh, seq, d]).with_grad();
        let k = Tensor::from_slice(&ctx, &mk(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1, n), vec![bh, seq, d]).with_grad();
        let v = Tensor::from_slice(&ctx, &mk(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2, n), vec![bh, seq, d]).with_grad();
        let log_g = Tensor::from_slice(&ctx, &mk(|i| -0.05 - (i % 3) as f32 * 0.02, bh * seq), vec![bh, seq]).with_grad();
        let beta = Tensor::from_slice(&ctx, &mk(|i| 0.3 + (i % 4) as f32 * 0.1, bh * seq), vec![bh, seq]).with_grad();
        let out = gated_delta_rule(&q, &k, &v, &log_g, &beta);
        let ones = Tensor::ones(&ctx, vec![n, 1]);
        let loss = out.reshape(vec![1, n]).matmul(&ones);
        autograd::backward(&ctx, loss.id);
        for (name, id) in [("q", q.id), ("k", k.id), ("v", v.id), ("log_g", log_g.id), ("beta", beta.id)] {
            let g = autograd::get_grad(id).unwrap_or_else(|| panic!("no grad for {name}"));
            let gv = Tensor::from_buffer(Arc::clone(&ctx), g, vec![1]).to_vec();
            assert!(gv.iter().all(|x| x.is_finite()), "non-finite grad {name}");
        }
        autograd::zero_grads();
    }

    #[test]
    fn conv1d_matches_cpu() {
        let ctx = ctx();
        let (bh, seq, c, kw) = (2usize, 5usize, 3usize, 4usize);
        let n = bh * seq * c;
        let x: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
        let ker: Vec<f32> = (0..c * kw).map(|i| ((i * 3 % 5) as f32 - 2.0) * 0.3).collect();
        let got = autograd::no_grad(|| {
            causal_conv1d(
                &Tensor::from_slice(&ctx, &x, vec![bh, seq, c]),
                &Tensor::from_slice(&ctx, &ker, vec![c, kw]),
                kw,
            )
            .to_vec()
        });
        let mut want = vec![0.0f32; n];
        for b in 0..bh {
            for t in 0..seq {
                for ch in 0..c {
                    let mut acc = 0.0;
                    for j in 0..kw {
                        let lag = kw - 1 - j;
                        if t >= lag {
                            acc += ker[ch * kw + j] * x[(b * seq + (t - lag)) * c + ch];
                        }
                    }
                    want[(b * seq + t) * c + ch] = acc;
                }
            }
        }
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= 1e-3 * (1.0 + w.abs()), "conv1d gpu={g} cpu={w}");
        }
    }

    #[test]
    fn partial_rope_passthrough_and_full() {
        let ctx = ctx();
        let (bh, seq, hd) = (2usize, 4usize, 6usize);
        let x: Vec<f32> = (0..bh * seq * hd)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let (full, part) = autograd::no_grad(|| {
            let xt = Tensor::from_slice(&ctx, &x, vec![bh, seq, hd]);
            (
                xt.apply_rope(0, 10000.0).to_vec(),
                partial_rope(&xt, 4, 0, 10000.0).to_vec(),
            )
        });
        // pass columns [4,6) must be unchanged by partial rope (rot_dim=4)
        for b in 0..bh {
            for t in 0..seq {
                for d in 4..hd {
                    let idx = (b * seq + t) * hd + d;
                    assert!((part[idx] - x[idx]).abs() < 1e-4, "pass col {d} changed");
                }
            }
        }
        // rot_dim == hd reproduces full apply_rope
        let eq = autograd::no_grad(|| {
            partial_rope(&Tensor::from_slice(&ctx, &x, vec![bh, seq, hd]), hd, 0, 10000.0).to_vec()
        });
        for (a, b) in eq.iter().zip(full.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn sigmoid_matches_cpu() {
        let ctx = ctx();
        let x: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.5).collect();
        let got = autograd::no_grad(|| sigmoid(&Tensor::from_slice(&ctx, &x, vec![3, 4])).to_vec());
        for (g, xi) in got.iter().zip(x.iter()) {
            let w = 1.0 / (1.0 + (-xi).exp());
            assert!((g - w).abs() < 1e-3, "sigmoid {g} vs {w}");
        }
    }

    #[test]
    fn expand_heads_matches_cpu() {
        let ctx = ctx();
        let (rows, n_in, repeat, d) = (3usize, 2usize, 2usize, 3usize);
        let x: Vec<f32> = (0..rows * n_in * d).map(|i| i as f32 * 0.5).collect();
        let got = autograd::no_grad(|| {
            expand_heads(
                &Tensor::from_slice(&ctx, &x, vec![rows, n_in * d]),
                n_in,
                repeat,
                d,
            )
            .to_vec()
        });
        let outdim = n_in * repeat * d;
        for row in 0..rows {
            for h in 0..n_in {
                for r in 0..repeat {
                    for j in 0..d {
                        let want = x[row * (n_in * d) + h * d + j];
                        let g = got[row * outdim + (h * repeat + r) * d + j];
                        assert!((g - want).abs() < 1e-4, "expand mismatch");
                    }
                }
            }
        }
    }
}
