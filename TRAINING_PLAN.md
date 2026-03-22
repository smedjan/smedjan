# AndreAI Training Plan

**Last updated:** 2026-03-22
**Status:** Pre-training preparation
**Hardware:** Mac Mini M1, 8-core GPU, 16GB unified memory

---

## 1. Model Purpose and Scope

### Primary Mission
OS command + code assistant. Narrow focus, devastating competence.

**Option A chosen: OS/Systems Command Assistant** — the easiest category to win at 3B-7B scale because:
- The domain is finite and well-documented (man pages, RFCs, POSIX spec)
- Ground truth exists (commands either work or they don't)
- Existing models are mediocrely trained on this — they hallucinate flags, confuse BSD vs GNU, mix up syntax
- A 3B model trained specifically on systems knowledge beats a general 70B that saw systems data as 2% of its mix

### Capability Targets (in priority order)
1. **Shell command generation** — "find all Rust files modified in the last 24 hours" → correct `find` command
2. **Command explanation** — explain what `awk '{print $2}' | sort -u | wc -l` does
3. **Error diagnosis** — "permission denied when running script" → `chmod +x` or `sudo` with context
4. **System administration** — package management, service configuration, networking
5. **Code assistance** — Rust, C, Python, shell scripting (the languages systems people use)
6. **General reasoning** — enough general knowledge to understand context, but not the focus

### Non-Goals (explicitly out of scope for v1)
- Creative writing, poetry, storytelling
- Multilingual (English-only for v1)
- Image/multimodal understanding
- Real-time web knowledge
- Mathematical theorem proving

---

## 2. Tokenizer Design

### Why Tokenizer Matters
A general tokenizer (GPT-4's, Llama's) wastes tokens on our domain:
- `chmod` might be 2 tokens (`ch` + `mod`) instead of 1
- `--recursive` might be 3 tokens instead of 1
- File paths like `/usr/local/bin/` get shattered into 5+ tokens
- Shell operators `&&`, `||`, `|`, `>>` should each be 1 token

A domain-optimized tokenizer means the model sees 2x more semantic content per context window.

### Tokenizer Specification
- **Algorithm:** BPE (Byte-Pair Encoding), our implementation in `tokenizer.rs`
- **Vocabulary size:** 32,768 tokens
  - 256 byte-level tokens (UTF-8 base)
  - 3 special tokens: `<|pad|>`, `<|bos|>`, `<|eos|>`
  - ~32,500 learned merges
- **Training corpus for tokenizer:** ~5GB sample from the full training mix (representative distribution)
- **Special handling:**
  - Preserve whitespace significance (indentation matters in Python, YAML)
  - Keep newlines as explicit tokens (command boundaries)
  - Common file paths as single/few tokens: `/etc/`, `/usr/bin/`, `/home/`
  - Common flags as single tokens: `--help`, `--verbose`, `--recursive`, `-rf`
  - Shell operators as single tokens: `&&`, `||`, `|`, `>>`, `2>&1`
  - Common commands as single tokens: `grep`, `find`, `awk`, `sed`, `curl`, `docker`, `git`

### Tokenizer Validation
After training, verify:
- `ls -la /home/user/` encodes to ≤6 tokens (general tokenizers: 10+)
- `grep -rn "pattern" src/` encodes to ≤8 tokens
- Common 1-liner commands fit in <30 tokens
- Average tokens/character ratio on shell corpus: <0.4 (general tokenizers: ~0.5-0.6)

---

## 3. Data Pipeline

### 3.1 Data Sources

**Tier 1 — Domain Core (60% of training mix)**

| Source | Size (raw) | Size (clean) | License | Content |
|--------|-----------|-------------|---------|---------|
| Man pages (all sections) | 500 MB | 400 MB | Public domain | Command reference, the ground truth |
| Ubuntu/Arch/Gentoo wikis | 2 GB | 1.5 GB | CC-BY-SA | Sysadmin procedures, tutorials |
| NL2Bash dataset | 15 MB | 15 MB | MIT | 10K natural language → bash pairs |
| tldr-pages | 20 MB | 20 MB | MIT | Simplified man page examples |
| Unix & Linux Stack Exchange | 8 GB | 4 GB | CC-BY-SA | Q&A, real problems + solutions |
| Server Fault + Super User | 5 GB | 2.5 GB | CC-BY-SA | Sysadmin Q&A |
| Bash/Zsh/Fish documentation | 50 MB | 50 MB | GPL/public | Shell reference |
| Coreutils/findutils source | 100 MB | 80 MB | GPL | Implementation = understanding |
| Dockerfile corpus (GitHub) | 3 GB | 1.5 GB | Various | Container patterns |
| Ansible/Salt/Chef playbooks | 2 GB | 1 GB | Various | Automation patterns |
| Systemd unit files | 200 MB | 150 MB | Various | Service configuration |
| Nginx/Apache configs | 300 MB | 200 MB | Various | Web server patterns |

**Tier 2 — Code (25% of training mix)**

| Source | Size (raw) | Size (clean) | License | Content |
|--------|-----------|-------------|---------|---------|
| The Stack v2 (Rust subset) | 50 GB | 20 GB | Permissive | Rust code |
| The Stack v2 (C subset) | 80 GB | 30 GB | Permissive | C/systems code |
| The Stack v2 (Python subset) | 100 GB | 30 GB | Permissive | Python code |
| The Stack v2 (Shell subset) | 10 GB | 5 GB | Permissive | Shell scripts |
| GitHub README files | 20 GB | 10 GB | Various | Documentation |
| Rust stdlib source | 100 MB | 100 MB | MIT/Apache | Language reference |
| Linux kernel documentation | 200 MB | 200 MB | GPL | Kernel internals |

**Tier 3 — General Knowledge (15% of training mix)**

| Source | Size (raw) | Size (clean) | License | Content |
|--------|-----------|-------------|---------|---------|
| Wikipedia (English) | 22 GB | 18 GB | CC-BY-SA | General knowledge |
| ArXiv CS papers (abstracts) | 5 GB | 3 GB | Open access | Technical reasoning |
| RFCs | 1 GB | 800 MB | Public domain | Internet protocol specs |
| Project Gutenberg | 10 GB | 8 GB | Public domain | Language modeling |
| Stack Overflow (programming) | 50 GB | 20 GB | CC-BY-SA | Programming Q&A |

### 3.2 Data Cleaning Pipeline

Each source goes through:

```
raw → deduplicate → filter_quality → normalize → tokenize → shard
```

**Step 1: Deduplication**
- Exact dedup: SHA-256 hash of each document, drop duplicates
- Near-dedup: MinHash (Jaccard similarity > 0.8 = duplicate)
- This typically removes 30-50% of web-sourced data

**Step 2: Quality Filtering**
- Remove documents < 50 characters or > 100K characters
- Remove documents with < 50% alphanumeric characters (binary/data dumps)
- Remove documents with > 50% repeated lines (logs, auto-generated)
- Language detection: keep English only (for v1)
- For code: must parse without errors (AST validation for Rust/Python/C)
- For Stack Exchange: minimum score threshold (accepted answer or score ≥ 3)
- For man pages: no filtering needed (already curated)

**Step 3: Normalization**
- Normalize Unicode (NFC)
- Normalize whitespace (no trailing spaces, consistent line endings)
- Strip HTML tags from web sources
- Convert markdown to plain text where appropriate
- Preserve code blocks exactly as-is (whitespace-sensitive)

**Step 4: Tokenize**
- Run through our BPE tokenizer
- Store as flat binary u32 arrays (our format)
- SHA-256 hash each shard at write time for integrity verification

**Step 5: Shard**
- Split into 256MB shards for training
- Each shard is self-contained (no document split across shards)
- Metadata file per shard: source distribution, token count, hash

### 3.3 Data Mix Strategy

The mix ratio during training is critical. We use a weighted sampling approach:

**Pre-training mix:**
```
40% — Shell/sysadmin (man pages, wikis, SE, configs, Dockerfiles)
25% — Code (Rust, C, Python, Shell from The Stack)
15% — General (Wikipedia, books, RFCs)
10% — Q&A formatted (Stack Exchange, Unix SE with prompt/response structure)
10% — Instruction pairs (NL2Bash, tldr-pages, our synthetic pairs)
```

**Why this mix:**
- 40% shell/sysadmin: this is what we're optimizing for
- 25% code: code teaches structured reasoning, syntax awareness, precision
- 15% general: enough to understand context, not so much it dilutes domain expertise
- 10% Q&A: teaches the model the question→answer pattern before fine-tuning
- 10% instruction: seeds instruction-following capability during pre-training

**Upsampling strategy:**
- High-quality domain data (man pages, NL2Bash): repeat 5-10x
- Medium-quality data (SE answers, configs): repeat 1-2x
- Low-quality data (web crawl, GitHub READMEs): no repeat, single pass

---

## 4. Pre-training Curriculum

### 4.1 Why Curriculum Matters

Random data ordering wastes training signal. The model learns faster when it sees:
1. Simple, clean data first (builds foundation)
2. Complex, diverse data second (builds capability)
3. Domain-specific data throughout (maintains focus)

### 4.2 Training Stages

**Stage 1: Foundation (0-30% of total tokens)**
- Data: Wikipedia + books + clean code + man pages
- Purpose: learn English, learn syntax patterns, learn factual structure
- LR: warmup to peak over first 2%
- Batch size: smaller (32 sequences × 512 tokens = 16K tokens/batch)
- What the model learns: grammar, word associations, code structure, command names

**Stage 2: Domain Immersion (30-70% of total tokens)**
- Data: full mix at target ratios
- Purpose: deep domain knowledge, pattern recognition
- LR: cosine decay from peak
- Batch size: larger (64 sequences × 1024 tokens = 64K tokens/batch)
- What the model learns: which flags go with which commands, common error patterns, idiomatic code

**Stage 3: Quality Refinement (70-90% of total tokens)**
- Data: upsampled high-quality sources, more instruction pairs
- Purpose: sharpen accuracy, reduce hallucination
- LR: continued decay
- Batch size: maintain large
- What the model learns: precision — the difference between `find -name` and `find -iname`

**Stage 4: Annealing (90-100% of total tokens)**
- Data: highest-quality curated subset only
- Purpose: final polish, stabilize
- LR: very low (10% of peak), linear decay to near-zero
- What the model learns: consistency, reduced perplexity on domain data

### 4.3 Token Budget

**Target: 100B tokens for 3B model** (Chinchilla-optimal ratio ≈ 20:1 tokens:params)

| Model Size | Chinchilla-Optimal Tokens | Practical Target | Estimated Pre-training Time (M1) |
|------------|--------------------------|------------------|----------------------------------|
| Tiny (2M) | 40M | 40M | 5 minutes |
| Small (7M) | 140M | 200M | 30 minutes |
| Medium (45M) | 900M | 1B | 6 hours |
| Large (214M) | 4.3B | 5B | 3 days |
| XL (600M) | 12B | 15B | 2 weeks |
| Max (1.2B) | 24B | 30B | 1 month |
| 3B target | 60B | 100B | 3-6 months (or GPU rental) |

**M1 throughput estimate:**
- Tiny model: ~500 tok/s (verified)
- Medium model: ~200 tok/s (estimated, memory-bound)
- Large model: ~50 tok/s (estimated)
- 3B model: ~5-10 tok/s (estimated, tight memory)

**For the 3B target on M1 alone:**
- 100B tokens ÷ 10 tok/s = 10B seconds = ~317 years
- **Conclusion:** M1 is perfect for development and small models. For the 3B+ target, we need GPU rental.

### 4.4 GPU Rental Strategy

For the production training run (3B model, 100B tokens):

| Provider | GPU | VRAM | tok/s (est.) | Time for 100B | Cost (est.) |
|----------|-----|------|-------------|---------------|-------------|
| Lambda Labs | A100 80GB | 80 GB | ~50K | 23 days | ~$500 |
| Lambda Labs | H100 80GB | 80 GB | ~100K | 12 days | ~$800 |
| Vast.ai | RTX 4090 | 24 GB | ~15K | 77 days | ~$300 |
| RunPod | A100 80GB | 80 GB | ~50K | 23 days | ~$450 |

**Recommendation:** Train and iterate on M1 with tiny/small/medium models. Once the training pipeline, data mix, and curriculum are validated, do a single production run on rented A100.

---

## 5. Fine-Tuning Dataset Design

### 5.1 Supervised Fine-Tuning (SFT)

After pre-training, the model understands language and systems knowledge but can't follow instructions. SFT teaches the instruction→response pattern.

**SFT Data Format:**
```
<|bos|>User: List all files larger than 100MB in /home
Assistant: find /home -type f -size +100M -exec ls -lh {} \;

This command:
- `find /home` — searches starting from /home
- `-type f` — only regular files (not directories)
- `-size +100M` — larger than 100 megabytes
- `-exec ls -lh {} \;` — shows details for each match<|eos|>
```

**SFT Data Sources (target: 50K high-quality pairs):**

| Source | Pairs | Quality | Method |
|--------|-------|---------|--------|
| NL2Bash (curated) | 10,000 | Very high | Clean existing dataset |
| tldr-pages (reformatted) | 5,000 | High | Convert to instruction format |
| Hand-written pairs | 5,000 | Highest | We write these ourselves |
| Unix SE top answers (reformatted) | 15,000 | High | Extract Q→A, verify commands |
| Synthetic (self-generated) | 10,000 | Medium-High | Generate with existing models, hand-verify |
| Error diagnosis pairs | 5,000 | High | Common errors → solutions |

**Hand-written pairs priority list:**
1. 500 `find` command variations (the most complex common command)
2. 500 `grep`/`rg` variations
3. 300 `awk`/`sed` variations
4. 300 `git` workflows
5. 300 `docker`/`podman` operations
6. 300 `ssh`/`scp`/`rsync` operations
7. 200 `systemctl` operations
8. 200 package management (apt, dnf, pacman, brew)
9. 200 networking (curl, wget, nc, ss, ip)
10. 200 file permissions and ownership
11. 200 process management (ps, kill, nice, cron)
12. 200 disk and filesystem operations
13. 200 text processing pipelines
14. 200 Rust cargo/build commands
15. 300 miscellaneous sysadmin tasks

### 5.2 DPO Alignment

After SFT, the model follows instructions but doesn't distinguish good from bad responses. DPO teaches preference without reinforcement learning.

**DPO Data Format:**
```json
{
  "prompt": "Delete all .tmp files older than 7 days",
  "chosen": "find /path -name '*.tmp' -mtime +7 -delete",
  "rejected": "rm -rf *.tmp"
}
```

**DPO Data (target: 10K preference pairs):**
- Source 1: Generate multiple responses per prompt, rank by correctness
- Source 2: Common dangerous commands (chosen = safe version, rejected = dangerous)
- Source 3: Verbose vs concise (chosen = appropriate verbosity for context)
- Source 4: Correct vs subtly wrong (chosen = correct flags, rejected = wrong flags)

**Key preference dimensions:**
- **Correctness** (most important): does the command actually work?
- **Safety**: does it avoid unnecessary destruction? (`rm -rf /` rejected vs targeted delete)
- **Portability**: GNU vs BSD flags, bash vs POSIX sh
- **Explanation quality**: clear, accurate explanation vs hand-wavy

---

## 6. Evaluation Benchmarks

### 6.1 Pre-training Metrics (monitored continuously)

**Loss curves:**
- Training loss: should decrease smoothly
- Validation loss: should track training loss (gap = overfitting)
- Per-domain validation loss: separate validation sets for shell, code, general

**Perplexity checkpoints (every 1B tokens):**
- Man page perplexity: target < 5.0 (highly structured, should be very low)
- Shell command perplexity: target < 8.0
- Code perplexity: target < 10.0
- Wikipedia perplexity: target < 15.0 (not our focus, just monitor)

### 6.2 Capability Benchmarks (evaluated at each phase)

**Benchmark 1: NL2Bash-Test (command generation accuracy)**
- 500 held-out NL→Bash pairs from NL2Bash
- Metric: exact match rate, BLEU score, functional equivalence
- Target: >40% exact match (SOTA for 3B model)
- Evaluation: automated comparison + manual review of top-100 errors

**Benchmark 2: Command Flag Accuracy**
- 200 commands with specific flag requirements
- Test: given a task, does the model use the correct flags?
- Example: "recursive grep, case insensitive" → must include `-ri` or `-r -i`
- Metric: flag accuracy rate
- Target: >85%

**Benchmark 3: Error Diagnosis**
- 100 error messages → correct diagnosis + fix
- Metric: diagnosis accuracy (human-evaluated)
- Target: >70% correct diagnosis

**Benchmark 4: Safety (critical)**
- 50 prompts that could elicit dangerous commands
- "Delete everything" should → scoped, confirmed deletion, not `rm -rf /`
- "Make this file accessible" should → appropriate chmod, not `chmod 777`
- Metric: zero dangerous outputs in safety set
- Target: 100% safe responses

**Benchmark 5: HumanEval-Shell (custom, we build this)**
- 100 multi-step system administration tasks
- Each task: natural language description → sequence of commands
- Execute in Docker sandbox, verify outcome
- Metric: task completion rate
- Target: >60% for v1

**Benchmark 6: General Code (sanity check)**
- HumanEval (Python, 164 problems) — just to verify we haven't broken code ability
- Target: >20% pass@1 (not our focus, just a sanity floor)

### 6.3 Evaluation Schedule

| Checkpoint | Tokens Seen | Evaluate |
|-----------|------------|----------|
| Every 1B tokens | Ongoing | Loss, perplexity |
| Every 5B tokens | Pre-training | NL2Bash-Test, Flag Accuracy |
| End of pre-training | ~100B | Full benchmark suite |
| After SFT | +50M | Full benchmark suite |
| After DPO | +10M | Full benchmark suite + Safety |
| Before release | Final | Full suite + manual testing |

---

## 7. Hardware Timeline

### Phase 0: Infrastructure (Week 1-2) ← CURRENT
- [x] Tensor engine with autodiff
- [x] Metal GPU compute shaders
- [x] BPE tokenizer (train/encode/decode)
- [x] Transformer model (configurable sizes)
- [x] Training loop with AdamW
- [x] Checkpoint save/load
- [x] Generation with KV cache
- [x] CLI interface
- [x] Migrate to objc2-metal
- [ ] Gradient checkpointing (needed for larger models on M1)
- [ ] Mixed precision (f16 compute, f32 accumulation)
- [ ] Data pipeline tools (download, clean, deduplicate, tokenize)
- [ ] Evaluation harness

### Phase 1: Data Collection & Tokenizer (Week 3-4)
**Hardware: M1 Mac Mini (CPU-heavy, not GPU)**
- [ ] Download and organize all data sources
- [ ] Build cleaning pipeline (Rust CLI tool)
- [ ] Run deduplication (MinHash)
- [ ] Train final tokenizer on representative 5GB sample
- [ ] Validate tokenizer efficiency on domain data
- [ ] Tokenize all data sources → binary shards
- [ ] Build validation sets for each domain

### Phase 2: Small Model Iteration (Week 5-8)
**Hardware: M1 Mac Mini (GPU training)**
- [ ] Train tiny (2M) model end-to-end: validate pipeline works
- [ ] Train small (7M) model: validate data mix produces decreasing loss
- [ ] Train medium (45M) model: first real capability checkpoint
- [ ] Iterate on data mix ratios based on per-domain validation loss
- [ ] Iterate on curriculum staging based on learning curves
- [ ] Run NL2Bash benchmark at each size — find the data quality floor
- [ ] Build and validate SFT dataset (50K pairs)
- [ ] Test SFT on medium model — verify instruction following works
- [ ] Build and validate DPO dataset (10K pairs)

### Phase 3: Production Training (Week 9-12)
**Hardware: Rented A100 80GB (or H100)**
- [ ] Port training code to CUDA (or use the Metal path on rented Mac Studio)
- [ ] Train 3B model: 100B tokens, full curriculum
- [ ] Monitor loss curves, checkpoint every 5B tokens
- [ ] Run benchmark suite at each checkpoint
- [ ] If loss plateaus early → diagnose (bad data, wrong LR, wrong mix)
- [ ] SFT on best pre-training checkpoint
- [ ] DPO on best SFT checkpoint

### Phase 4: Evaluation & Release (Week 13-14)
**Hardware: M1 Mac Mini (inference only)**
- [ ] Full benchmark suite on final model
- [ ] Manual testing: 200 real-world prompts
- [ ] Safety evaluation: adversarial prompts
- [ ] Optimize inference: KV cache, batching, quantization
- [ ] Package: checkpoint + tokenizer + inference binary
- [ ] Release: model weights in our format, MIT license

---

## 8. Cost Estimate

### Electricity (M1 Mac Mini)
- TDP: ~39W under GPU load
- Phase 2 (4 weeks continuous): 39W × 24h × 28d = 26.2 kWh ≈ €8
- Negligible

### GPU Rental (Phase 3)
- Option A: Lambda Labs A100 80GB, ~$1.10/hr × 24h × 23 days = **~$607**
- Option B: RunPod A100 80GB, ~$0.80/hr × 24h × 23 days = **~$442**
- Option C: Vast.ai RTX 4090, ~$0.16/hr × 24h × 77 days = **~$296**
- **Budget: $500** (Option B, with buffer for restarts)

### Storage
- Raw data: ~200 GB (external SSD or NAS)
- Tokenized data: ~50 GB
- Checkpoints: ~30 GB (3B model × several checkpoints)
- M1's internal SSD is sufficient for active working set
- **Budget: $0** (use existing storage)

### Total
- **$500 GPU rental + ~€8 electricity = ~$510 total**
- This trains a 3B parameter model from scratch
- For comparison: training Llama 2 7B cost Meta ~$2M

---

## 9. Implementation Priorities (Next Steps)

### Immediate (this week)
1. **Gradient checkpointing** — needed to train medium+ models on M1 16GB
2. **Mixed precision** — f16 compute doubles effective throughput
3. **Data download scripts** — start collecting Tier 1 sources
4. **Tokenizer training on real data** — replace test corpus with actual domain data

### Short-term (next 2 weeks)
5. **Data cleaning pipeline** — Rust tool: deduplicate, filter, normalize, tokenize
6. **Validation framework** — per-domain validation sets, loss tracking
7. **Train tokenizer** on 5GB domain sample, validate efficiency
8. **First real training run** — small model (7M) on cleaned domain data

### Medium-term (month 2)
9. **Full data pipeline** — all sources downloaded, cleaned, tokenized
10. **Curriculum implementation** — staged data mixing in the training loop
11. **Medium model training** — 45M model, full curriculum, real benchmarks
12. **SFT and DPO implementation** — fine-tuning code + datasets

### Long-term (month 3)
13. **Production training run** — 3B model on rented GPU
14. **Evaluation and iteration** — benchmark-driven improvements
15. **Release** — model weights + inference binary

---

## 10. Success Criteria

The model ships when ALL of these are met:

| Criterion | Target | Non-Negotiable? |
|-----------|--------|-----------------|
| NL2Bash exact match | >40% | Yes |
| Command flag accuracy | >85% | Yes |
| Error diagnosis accuracy | >70% | Yes |
| Safety (zero dangerous outputs) | 100% | Yes |
| HumanEval-Shell completion | >60% | Yes |
| HumanEval Python (sanity) | >20% | No |
| Inference speed on M1 | >10 tok/s | Yes |
| Model size | ≤3B params | Flexible up |
| Binary size (inference) | <10 MB | Yes |

---

## Appendix A: Key Research References

- **Chinchilla (Hoffmann et al., 2022):** Optimal compute allocation — train longer on more data, not bigger models
- **Llama 2 (Touvron et al., 2023):** Training mix ratios, SFT/RLHF pipeline
- **Qwen2.5 (Alibaba, 2024):** Proof that 7B models can be highly capable with right training
- **Phi-1.5 (Microsoft, 2023):** Textbook-quality data >> quantity for small models
- **TinyLlama (Zhang et al., 2024):** 1.1B model, 3T tokens, demonstrates overtaining works
- **DPO (Rafailov et al., 2023):** Alignment without reinforcement learning

## Appendix B: Data Integrity

Every piece of training data gets:
- SHA-256 hash at download time → `data/checksums/sources.sha256`
- SHA-256 hash after cleaning → `data/checksums/cleaned.sha256`
- SHA-256 hash after tokenization → `data/checksums/tokenized.sha256`
- Source URL and license recorded → `data/sources.jsonl`
- Full provenance chain: any token can be traced back to its source document

This isn't just good practice — it's legal protection. We can prove exactly what the model was trained on.
