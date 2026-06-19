# Verification — Phase A (landed fixes) + Phase B (GPU-correctness harness)

Run on **air (MacBook Air M3, 16 GB, Metal 4)**, 2026-06-14. This closes Phase A of
`HANDOFF_drain_bleeding_and_gaps.md` (verify the landed fixes for real) and lands Phase B
(the correctness harness). Specs are canonical in `HANDOFF_buffer_hazard_and_followups.md`.

## Phase A — static gate (§5 protocol)

| Gate | Result |
|---|---|
| `cargo test --no-default-features --features metal` | **177 passed, 0 failed, 4 ignored** (the 162 prior + 10 grad-checks + 5 goldens) |
| `cargo clippy --no-default-features --features metal --all-targets -- -D warnings` | **clean** (0 warnings, exit 0) |
| `cargo fmt --check` / `git diff --check` | **clean** |
| `cargo clippy … --features metal,bufsan …` | **clean** (0 warnings, exit 0) |
| `cargo check --no-default-features --features cuda` | **passes**; runtime proof remains NVIDIA-hardware-gated |
| `./scripts/train-smoke.sh` | **passes**; self-contained CLI tokenizer→prepare→train smoke matrix |
| `cargo test … -- --test-threads=1` | run by the mini CI runner (`scripts/ci-mac.sh`); correctness tests are serial, ignored tests are performance benchmarks only |
| 0 `#[allow]` in `src/` | holds (the only grep hit is a comment asserting it) |

> The serial run + the bufsan suite now execute automatically on **mini** on every push (launchd agent
> `nu.andreai.ci` → `scripts/ci-mac-poll.sh` → `scripts/ci-mac.sh`), closing the "nothing in CI" gap.

## Phase A — run gate (loss readout + Muon/BitNet), tiny model on `train_v3.bin`

Short `--warmup 50` so LR reaches full within the run. Loss is column 2 of `train.csv`.
"pinned1.0=no" means the **honest readout** is working (`e9eb136`): the displayed loss shows
real dynamics, never the constant `1.0000` artifact. "bounded" means no NaN/FATAL/divergence.

| Run | first → last (min) | honest? | bounded? |
|---|---|---|---|
| adamw b16 (700 steps, lr2e-3) | 9.574 → 9.567 (min 9.479) | yes (70 distinct) | yes |
| adamw b32 | 9.513 → 9.502 (min 9.487) | yes | yes |
| adamw b64 | 9.508 → 9.497 (min 9.496) | yes | yes |
| muon b16 | 9.486 → 9.463 (min 9.460) | yes | yes |
| **muon b32** | **9.496 → 9.450 (min 9.450)** | yes | **yes — Frobenius fix holds, no batch-32 divergence** |
| muon b64 | 9.509 → 9.479 (min 9.475) | yes | yes |
| muon b32 (lr1e-3) | 9.464 → 9.493 (min 9.464) | yes | yes |
| hybrid b32 | 9.471 → 9.456 (min 9.456) | yes | yes |
| normuon b32 (scale 1.0) | 9.503 → **13.607** (min 9.369) | yes | climbs — see recheck below |
| bitnet b16 | 9.449 → 9.491 (min 9.449) | yes | yes (per-column ternary scale) |

Reading: loss-readout honesty is confirmed across every config (none pinned to 1.0). AdamW,
Muon, and hybrid stay bounded at batch 32 (and 16/64) — the Newton-Schulz Frobenius-normalization
fix holds. The descent is modest because this is a 2M-param model in < 1 epoch on real data;
convergence itself is separately proven by the unit suite's overfit tests. NorMuon at the default
`--muon-lr-scale 1.0` runs **too hot** and the (honest) readout shows the climb — exactly the
handoff caveat that NorMuon changes per-neuron update magnitude and needs `--muon-lr-scale` retuned.

### NorMuon — verified finding (NOT a bug; needs a lower `--lr`)

First, a correction: **`--muon-lr-scale` is hybrid-only** — `HybridOptimizer` does
`muon.step(lr · muon_lr_scale)` (optim.rs:871), but standalone `--optimizer muon` does `muon.step(lr)`.
So tuning `--muon-lr-scale` on `--optimizer muon --normuon` is a **no-op**; an initial recheck that varied
it (1.0/0.5/0.3/0.15) only sampled run-to-run variance (13.6–18.0, all climbing). Standalone NorMuon must
be retuned via `--lr`. Doing that (batch32, 200 steps):

| `--optimizer muon --normuon` `--lr` | first → last | verdict |
|---|---|---|
| 3e-3 | 9.50 → 13–18 (run variance) | climbs (too hot) |
| 1e-4 | 9.509 → 9.467 | **stable, descends** |
| 3e-5 | 9.540 → 9.493 | **stable, descends** |

So NorMuon is **correct, not a bug** — it descends like plain Muon once `--lr` is dropped ~30×+. Why the
high LR sensitivity: plain Muon feeds a Frobenius-normalized update (`‖X‖_F = 1`); NorMuon then row-scales
by `s_i = 1/(√v̂_i + ε)` with `v̂_i ≈ RMS(row)² ≈ 1/(rows·cols)`, so `s_i ≈ √(rows·cols)` — ~128–256× for
the tiny model's 2-D weights — making the update that much larger, so plain-Muon LRs are far too hot.

**Recommended follow-ups:** (1) make NorMuon's magnitude transferable — after the per-neuron row scaling,
re-Frobenius-normalize `X` (reuse `gpu_muon_frob_normalize`) so its update size matches plain Muon's while
keeping the per-neuron *relative* adaptation; then `--lr` transfers between `muon`/`normuon`. Needs the
NorMuon paper to confirm it preserves the +11%. (2) Until then, document that `--normuon` needs a much
lower `--lr` (standalone) or `--muon-lr-scale` (hybrid). (3) `--muon-lr-scale` applying only to the hybrid
group is a sharp edge worth a CLI help note.

## Phase B — GPU-correctness harness (drains the bleeding class)

The unit suite previously could not see the buffer-hazard / numeric class; a kernel bug only
surfaced in a full training run. This converts "only a real run catches it" → "CI catches it."

- **B1 — finite-difference grad checks** (`tests::suite::gradcheck_*`, 10 ops): perturb each
  input ±ε, compare numeric grad to the analytic grad from the real `autograd::backward`. Covers
  matmul, matmul_trans_b, add, mul, softmax, rms_norm, slice_cols, silu, silu_gate, scale.
- **B2 — golden tests vs CPU reference** (`tests::suite::golden_*`, 5): pins the hand-written
  forward kernels to known-correct output. Includes a **BitNet per-column-scale lock**
  (`golden_ternary_matmul_per_column_scale`) — W built so per-column `absmean` genuinely differs,
  so a regression to a single global `absmean[0]` fails for every column but the first.
- **B3 — buffer-pool sanitizer** (feature `bufsan`, `src/metal/mod.rs`): a generation counter
  bumped per flush; pooled buffers are **NaN-poisoned at flush time** (never at recycle time —
  that would corrupt reads still legitimately pending in the batch), so a use-after-recycle or a
  too-small dispatch surfaces as NaN. Opt-in **quarantine** forbids intra-batch reissue (the
  loss-readout-class hazard). Tests `bufsan_training_stays_finite_under_poison` and
  `bufsan_quarantine_matches_default` pass — the current training path is clean of the class.
- **B4 — Mac CI runner** (`scripts/ci-mac.sh`): owns a host-wide GPU lock, runs the §5 protocol +
  formatting gates + CUDA compile parity + self-contained training smokes + the bufsan suite, exits
  non-zero on any failure. Intended for an always-on Apple-Silicon Mac (mini) via launchd/cron.

### Bug found and fixed by the harness (first run)

`grad_check` for matmul reported `numeric = 0.0000` exactly — a perturbed `from_slice` input was
silently ignored. Root cause: `MetalContext::buffer_from_slice` (used by `Tensor::from_slice`)
never invalidated the address-keyed fp16/ternary conversion cache, unlike `alloc_buffer`, so a
fresh buffer at a recycled address inherited a stale fp16 cast. Fixed by mirroring `alloc_buffer`'s
`invalidate_conversion_cache` call in `buffer_from_slice`. Regression-free (177/177).
