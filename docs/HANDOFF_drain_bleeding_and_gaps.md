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
3. **Phase C — close the bounded gaps** (each already specced): seq-packing, block-sparse
   trainable backward, delete GaLore, then throughput (chunked SSM/RWKV, MLA absorbed,
   Flash Attention), then CUDA backward.

Do **not** turn this into a treadmill: chasing every new paper or rebuilding the
data/eval/distributed ecosystem is a standing tax, not a gap you close. The architectural
bets (MLA, hybrid mixers, Muon/NorMuon, BitNet, 8-bit optim) are already aligned with the
June-2026 SOTA (see `HANDOFF_buffer_hazard_and_followups.md` §7 field map). The actionable
work is **correctness infrastructure** + the **bounded** items below.

---

## Phase A — verify the landed fixes (the actual gate)

Canonical protocol (from `HANDOFF_buffer_hazard_and_followups.md` §5):

```bash
cargo test --no-default-features --features metal
cargo test --no-default-features --features metal -- --include-ignored --test-threads=1  # serial GPU tests
cargo clippy --no-default-features --features metal --all-targets   # expect 0 warnings, 0 #[allow]
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
| **Block-sparse trainable backward** | §3 | Needs a new **scatter-add** Metal kernel. Forward (gather) already landed. |
| **Delete GaLore** | §6 / §7 #2 | Dead code that panics at `--galore-rank>0`; field moved to SCALE (arXiv 2506.16659). Remove code + any stale claim, or target SCALE — do **not** build GaLore. |
| **Chunked O(N) SSM/RWKV forward** | §7 #4 | Currently materialized; throughput, not correctness. |
| **MLA absorbed-form decode** | §7 #3 | Incremental decode done; absorbed form is the high-value compute follow-up. |
| **Flash Attention** | `ROADMAP.md` Phase 1 | Highest throughput lever (2× seq=256, 4×+ seq=1024); tiled fwd+bwd kernels, then CUDA port. |
| **CUDA backward stubs** | closure table | Metal is the source of truth; CUDA backward is incomplete. Only if CUDA targets matter. |

Each needs Metal hardware to implement + verify — by design these were **specced, not
blind-written**, so they land fast once you're on the Mac with the harness in place.

---

## Ground state (verified this session, air / M-series Mac)

- `main` was at `49645cf` and in sync with `origin/main`; **uncommitted, M3-verified
  loss-readout fix was sitting in the tree** — now committed (`e9eb136`, compiles clean via
  `cargo check --no-default-features --features metal`). Its real-run re-confirmation is
  Phase A.
- Canonical state of every prior item: `docs/HANDOFF_buffer_hazard_and_followups.md` §8
  closure table (10 audit rounds; 7 bug fixes + 3 features landed, all "need Mac verify").
- Code reality: ~16.4K lines, 81 tests (none cover the GPU hazard class — that's Phase B),
  Metal + CUDA + AndreOS backends; Metal is canonical.
- Field alignment + what NOT to build: §7 field map (drop GaLore; MLA/hybrid/Muon/BitNet are
  the right bets).

## The one-line decision frame

If **the engine is the product**, Phases A→C *are* the roadmap and this is exactly the work.
If **models are the product**, do Phase A + Phase B (so the engine stops bleeding) and then
spend your time training, not rebuilding PyTorch's ecosystem.
