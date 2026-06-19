# Handoff — drain the bleeding points & gaps

**For a fresh session.** Goal: take andreai's correctness backlog to actually-zero, then
close the specced gaps. Build is **Mac + Metal only** (air/mini have cargo + Metal; the
Linux/cowork sandbox cannot build this — every claim below is gated on a host run).

Origin = self-hosted Forgejo (`http://localhost:3300/andrei/andreai`, via the `forge`
tunnel). Land linearly on `main` (andreai is NOT under the redofy/hugin squash-only rule).

---

## TL;DR priority order

1. **Phase A — verify, don't assume.** Run the §5 gate (below) on the Mac. The bug fixes
   are *coded* but the closure table is ahead of a real run. The loss-readout fix is now
   committed (`e9eb136`); confirm honest loss on a batch sweep. **This empties the bug
   backlog for real.**
2. **Phase B — drain the bleeding *class*, not just instances.** Build the GPU-correctness
   harness. This is the highest-leverage work in the repo: today a kernel bug only shows up
   in a full training run (the 81-test suite provably can't see the buffer-hazard / numeric
   class — see the loss-readout and BitNet-per-column bugs). Convert "only a real run
   catches it" → "CI catches it."
3. **Phase C — close the bounded gaps** (each already specced): seq-packing and the remaining
   throughput work (chunked RWKV, MLA absorbed form, fused kernels), then CUDA runtime proof on
   NVIDIA hardware.

Do **not** turn this into a treadmill: chasing every new paper or rebuilding the
data/eval/distributed ecosystem is a standing tax, not a gap you close. The architectural
bets (MLA, hybrid mixers, Muon/NorMuon, BitNet, 8-bit optim) are already aligned with the
June-2026 SOTA (see `HANDOFF_buffer_hazard_and_followups.md` §7 field map). The actionable
work is **correctness infrastructure** + the **bounded** items below.

---

## Phase A — verify the landed fixes (the actual gate)

Canonical protocol (from `HANDOFF_buffer_hazard_and_followups.md` §5):

```bash
cargo fmt --check
git diff --check
cargo test --no-default-features --features metal
cargo test --no-default-features --features metal -- --include-ignored --test-threads=1  # serial GPU tests
cargo clippy --no-default-features --features metal,bufsan --all-targets -- -D warnings
cargo check --no-default-features --features cuda
```

Then the run-level gates that the unit suite can't cover:
- **Loss readout** (`e9eb136`): train a tiny model across batch ∈ {16, 32, 64}; the logged
  loss must descend (≈9.x → lower), never pin to a constant `1.0`. Too-hot configs should
  visibly climb, not hide.
- **Muon stability**: same batch sweep with `--optimizer muon` and `hybrid`; confirm no
  divergence at batch 32 (the Frobenius fix). **Re-check `--muon-lr-scale`** (and with
  `--normuon`, per-neuron magnitude changed).
- **`--bitnet` smoke**: one short run; confirms the per-column ternary scale.

Per the closure table (§8 of the buffer-hazard handoff), once these pass the **bug backlog
is empty** and everything after is feature work. If any fails, that's the first bleeding
point — fix it before Phase B.

---

## Phase B — the GPU-correctness harness (drain the bleeding class)

This is the piece that's missing and that makes the whole from-scratch-Metal path safe.
Four parts, each finite, each compounding:

1. **Finite-difference gradient checks per op/kernel.** For matmul/add/mul/softmax/rms_norm/
   slice and every optimizer step: perturb each input element by ±ε, compare numeric grad to
   the analytic grad from `autograd::backward`, gate on relative error. Catches the Muon-NS
   and per-column-scale class directly. Runs as a `#[test]` (small tensors) so it lives in CI.
2. **Golden tests vs a reference**, op by op: same inputs through a trusted oracle (a CPU
   reference impl, or PyTorch/MPS exported fixtures checked into `tests/golden/`), compare
   within tolerance. Pins hand-written kernels to known-correct output.
3. **Buffer-pool sanitizer** — the #1 hazard (see `HANDOFF_buffer_hazard_and_followups.md`
   §1). Add a debug build mode to `src/metal/mod.rs` that (a) fills every recycled buffer
   with a NaN/sentinel and (b) tags buffers with a generation counter, asserting on any read
   of a buffer recycled-but-not-yet-flushed. This turns the use-after-recycle class — the
   shared root cause of the loss-readout AND seq-packing reverts — into an immediate panic in
   CI instead of silent corruption in a real run. (The §1 doc also notes the optional
   systemic fix: generation-quarantine the pool — measure memory cost first.)
4. **Mac CI runner.** `mini` is always-on with Metal. Wire a self-hosted runner (or a cron
   that pulls `origin/main`, runs the §5 protocol + the new harness, and reports) so GPU
   tests gate every push — not a manual host run. This is what closes the "all verification
   is host-side / nothing in CI" gap permanently.

After Phase B, re-landing kernels (seq-packing, sparse backward) is safe: the harness, not a
hope-and-a-prayer training run, is the gate.

---

## Phase C — close the bounded gaps (all specced)

Land order (ROI, from §6/§7 of the buffer-hazard handoff):

| Gap | Where it's specced | Notes |
|---|---|---|
| **Seq-packing model integration** | `HANDOFF_buffer_hazard…` §2 | Op-level packing exists; model-integration reverted once (buffer hazard). Re-land **behind Phase B's sanitizer**. |
| **Block-sparse trainable backward** | §3 | ✅ LANDED — scatter-add kernel + `Op::GatherBlocks`, grad-checked; fixed a latent non-square `batched_matmul_trans_a` param-swap. |
| **Delete GaLore** | §6 / §7 #2 | Active production surface no longer exposes GaLore; keep historical context only, or target SCALE if a subspace method is revisited. |
| **Chunked O(N) SSM forward** | §7 #4 | ✅ **LANDED** — `ssm_decayed_numerator_chunked` (Mamba-2/SSD), wired into `ssm()` for seq≥256 (largest of {64,32} dividing seq). Intra-chunk = the materialised primitive per chunk; inter-chunk = `o=g·(q@S_p)` with state `S=Λ@KV` — a chunk-level decay matrix applied as ONE batched `[nc,nc]@[nc,hd²]`, **no sequential scan**. Decay factorisation `g_t·Λ(p',p)·h_j = exp(A_t−A_j)` verified algebraically + `ssm_chunked_matches_materialized` (forward, every chunk size) + `ssm_chunked_grad_matches_materialized` (backward q/k/v/loga). RWKV chunking still open — design done, gated on one missing primitive (see RWKV row). **Non-bug noted:** loga position 0 has a STRUCTURALLY-ZERO gradient (uniform shift of A, cancels from every `A_t−A_j`) — a finite-diff there is fp-noise×1/2ε and spuriously "fails"; verify SSM loga grads by analytic-equivalence, not central differences. |
| **Chunked O(N) RWKV (wkv) forward** | §7 #4 | Existing `wkv` backward now VERIFIED (`gradcheck_wkv` — sound, fp16-precision on the exp-sensitive w/u params; widen rel-tol to 12% there, k/v at 5%). The materialised `wkv` is O(seq²) (the `lower_tri @ (exp(t·w)·p)` cumsum) AND overflow-unstable for long seq (absolute `exp(±t·w)`, clamp-papered). Chunked design (DERIVED, stable): `wkv_t = intra[p,a] + exp(−a·w)·R_p`, where intra = the verified `wkv` applied per chunk (stable since within-chunk positions <C), and the cross-chunk state is the relative-decay scan `R_{p+1}=γ·R_p+KV_p` (γ=exp(−C·w)≤1, KV_p=Σ_b exp(−(C−1−b)w)·p_{p,b}). Verify against `cpu_wkv` (stable, any seq). **BLOCKER — needs ONE new differentiable primitive:** assembling the per-chunk results back into the bh-outer `[bh,seq,hd]` layout requires a differentiable seq-concat OR a general 3D axis-swap transpose. `concat_seq` exists but does NOT record a tape entry (inference-only); `concat_flat` is dim-0 only (gives chunk-outer `p·bh+b`, wrong interleave). Add a grad-checked `transpose_last2`/`concat_dim1`, then the chunked composition is pure existing-ops. RWKV decay is per-channel (not per-position like SSM), so SSM's batched `Λ@KV` trick does not transfer. |
| **MLA absorbed-form decode** | §7 #3 | **NOT a contained refactor — needs an architecture change.** `mla_cached_forward` applies `fused_transpose_rope` to BOTH q and the reconstructed k (attention.rs:426-427), so RoPE sits between q and k. The absorbed identity `scores=(q·W_ukᵀ)·cᵀ` and `out=(softmax·c)·W_uv` only holds WITHOUT RoPE on that path. Doing it requires DeepSeek **decoupled RoPE**: split the head into a small rope sub-dim (separate shared K_rope) + a nope sub-dim that carries the absorbed latent — new projections, changed cache layout, **changed checkpoint format, and a convergence run** (not just decode-equivalence). Surface as an arch decision; do not blind-land. |
| **Flash Attention** | `ROADMAP.md` Phase 1 | Tiled fwd+bwd kernels existed but were UNVERIFIED (only a constructability test). ✅ Now grad-checked + dense-equivalence tested — which caught & fixed a **partial-last-q-block** bug: out-of-range query threads returned before the cooperative K/V tile load, leaving `K/V_shared` unloaded → wrong out+grad for any seq_q not a multiple of 32 with >1 q-block (fwd+bwd both fixed). CUDA port still pending. |
| **CUDA runtime proof** | closure table | Metal is the source of truth; CUDA compiles but runtime correctness is NVIDIA-hardware-gated. Only if CUDA targets matter. |

Each needs Metal hardware to implement + verify — by design these were **specced, not
blind-written**, so they land fast once you're on the Mac with the harness in place.

### Phase B grad-check coverage EXTENDED to the whole attention kernel path (2026-06-14)

The B1 finite-diff harness now covers every custom/fused backward kernel on the hot path — the
class that previously had *no* numerical check and that hid two real bugs this session:

- **`batched_matmul_trans_a`** (the `dB = Aᵀ@dC` half of every batched-matmul backward) had its
  param struct fields swapped (`{m,n,k}` vs kernel `{M,K,N}`) — silently correct only for K==N
  (all square attention scores), wrong for non-square. Caught by `gradcheck_batched_matmul_nonsquare`
  / `…_trans_b_nonsquare`. Fixed.
- **Flash Attention** partial-last-q-block (see table above). Caught by `gradcheck_flash_attention`
  + `flash_attention_matches_dense_causal`. Fixed.

New grad-checks, all PASS (kernels verified sound): `scaled_causal_softmax`, `apply_rope`,
`fused_transpose_rope`, `transpose_bsh_to_bhs`, `transpose_bhs_to_bsh`, `rms_norm_residual`,
`scale_rows`, `repeat_kv` (GQA), `flash_attention`, `block_sparse_gather_attention`. Remaining
unchecked customs are low-risk (`embedding` = the validated scatter-add precedent, `concat_parts`,
`slice`, `matmul_detached_b` STE, `checkpoint` recompute). SSM/RWKV are pure compositions of
already-grad-checked primitives (forward also CPU-verified), so their backward is correct by
construction. **Lesson: a custom fused backward with only a "gradient is finite/non-zero" test is
unverified — finite-diff it.**

---

## Ground state (verified this session, air / M-series Mac)

- `main` was at `49645cf` and in sync with `origin/main`; **uncommitted, M3-verified
  loss-readout fix was sitting in the tree** — now committed (`e9eb136`, compiles clean via
  `cargo check --no-default-features --features metal`). Its real-run re-confirmation is
  Phase A.
- Canonical state of every prior item: `docs/HANDOFF_buffer_hazard_and_followups.md` §8
  closure table (10 audit rounds; 7 bug fixes + 3 features landed, all "need Mac verify").
- Code reality at the time: ~16.4K lines, 81 tests (none covered the GPU hazard class then —
  Phase B), Metal + CUDA plus an unshipped AndreOS target; Metal is canonical.
- Field alignment + what NOT to build: §7 field map (drop GaLore; MLA/hybrid/Muon/BitNet are
  the right bets).

## The one-line decision frame

If **the engine is the product**, Phases A→C *are* the roadmap and this is exactly the work.
If **models are the product**, do Phase A + Phase B (so the engine stops bleeding) and then
spend your time training, not rebuilding PyTorch's ecosystem.
