#!/usr/bin/env bash
# End-to-end training smoke for CI. Builds its own tiny tokenizer/dataset under target/
# so a fresh clone can exercise the real CLI/train loop without external data.
set -euo pipefail
export CARGO_INCREMENTAL=0

REPO="${ANDREAI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${ANDREAI_TRAIN_SMOKE_DIR:-$REPO/target/ci-train-smoke}"
BIN="$REPO/target/release/andreai"
LOG_DIR="$OUT/logs"
CORPUS="$OUT/corpus.txt"
TOKENIZER="$OUT/tokenizer.bin"
DATASET="$OUT/dataset.bin"
BAD_TOKENIZER="$OUT/bad-tokenizer.bin"
BAD_CHECKPOINT="$OUT/bad-checkpoint.bin"
BAD_SHARD="$OUT/bad-shard.bin"

rm -rf "$OUT"
mkdir -p "$LOG_DIR"

cat > "$CORPUS" <<'CORPUS'
AndreAI trains compact language models on local Metal GPUs.
The runtime must keep losses finite, write checkpoints, and exercise optimizer state.
Short smoke runs cover dense attention, fused CE, gradient checkpointing, BitNet, SSM, block sparse routing, and hybrid Muon.
The corpus repeats enough structure for the tiny tokenizer and dataset pipeline.
AndreAI trains compact language models on local Metal GPUs.
The runtime must keep losses finite, write checkpoints, and exercise optimizer state.
Short smoke runs cover dense attention, fused CE, gradient checkpointing, BitNet, SSM, block sparse routing, and hybrid Muon.
The corpus repeats enough structure for the tiny tokenizer and dataset pipeline.
CORPUS

run_logged() {
  local name=$1; shift
  local log="$LOG_DIR/$name.log"
  echo "---- $name ----"
  if "$@" > "$log" 2>&1; then
    echo "PASS: $name"
  else
    echo "FAIL: $name (see $log)"
    tail -40 "$log"
    exit 1
  fi
}

run_reject_logged() {
  local name=$1
  local pattern=$2
  shift 2
  local log="$LOG_DIR/$name.log"
  echo "---- reject:$name ----"
  if "$@" > "$log" 2>&1; then
    echo "FAIL: reject:$name unexpectedly succeeded"
    tail -60 "$log"
    exit 1
  fi
  if ! grep -Fq -- "$pattern" "$log"; then
    echo "FAIL: reject:$name did not report '$pattern'"
    tail -60 "$log"
    exit 1
  fi
  if grep -Fq "panicked at" "$log"; then
    echo "FAIL: reject:$name reported via panic"
    tail -60 "$log"
    exit 1
  fi
  echo "PASS: reject:$name"
}

run_reject_train() {
  local name=$1
  local pattern=$2
  shift 2
  local ckpt="$OUT/$name"
  local log="$LOG_DIR/$name.log"
  echo "---- reject:$name ----"
  if "$BIN" train \
    --dataset "$DATASET" \
    --tokenizer "$TOKENIZER" \
    --size tiny \
    --batch-size 2 \
    --seq-len 16 \
    --steps 1 \
    --warmup 1 \
    --lr 0.001 \
    --checkpoint-dir "$ckpt" \
    "$@" > "$log" 2>&1; then
    echo "FAIL: reject:$name unexpectedly succeeded"
    tail -60 "$log"
    exit 1
  fi
  if ! grep -Fq -- "$pattern" "$log"; then
    echo "FAIL: reject:$name did not report '$pattern'"
    tail -60 "$log"
    exit 1
  fi
  if grep -Fq "panicked at" "$log"; then
    echo "FAIL: reject:$name reported via panic"
    tail -60 "$log"
    exit 1
  fi
  echo "PASS: reject:$name"
}

run_train() {
  local name=$1; shift
  local ckpt="$OUT/$name"
  local log="$LOG_DIR/$name.log"
  echo "---- train:$name ----"
  if "$BIN" train \
    --dataset "$DATASET" \
    --tokenizer "$TOKENIZER" \
    --size tiny \
    --batch-size 2 \
    --seq-len 16 \
    --steps 5 \
    --warmup 1 \
    --lr 0.001 \
    --checkpoint-dir "$ckpt" \
    "$@" > "$log" 2>&1; then
    :
  else
    echo "FAIL: train:$name (see $log)"
    tail -60 "$log"
    exit 1
  fi
  if grep -E 'FATAL|NaN detected|loss is (NaN|inf|-inf)' "$log" >/dev/null; then
    echo "FAIL: train:$name emitted a fatal/non-finite loss signal"
    tail -60 "$log"
    exit 1
  fi
  if ! grep -q "Training complete" "$log"; then
    echo "FAIL: train:$name did not complete"
    tail -60 "$log"
    exit 1
  fi
  if [[ ! -s "$ckpt/final.bin" || ! -s "$ckpt/state_final.bin" ]]; then
    echo "FAIL: train:$name did not write final checkpoints"
    tail -60 "$log"
    exit 1
  fi
  echo "PASS: train:$name"
}

run_resume_train() {
  local name=$1; shift
  local base="$OUT/${name}_base"
  local resumed="$OUT/${name}_resume"
  local base_log="$LOG_DIR/${name}_base.log"
  local resume_log="$LOG_DIR/${name}_resume.log"
  echo "---- resume:$name ----"
  if "$BIN" train \
    --dataset "$DATASET" \
    --tokenizer "$TOKENIZER" \
    --size tiny \
    --batch-size 2 \
    --seq-len 16 \
    --steps 2 \
    --warmup 1 \
    --lr 0.001 \
    --checkpoint-dir "$base" \
    "$@" > "$base_log" 2>&1; then
    :
  else
    echo "FAIL: resume:$name base run (see $base_log)"
    tail -60 "$base_log"
    exit 1
  fi
  if [[ ! -s "$base/state_final.bin" ]]; then
    echo "FAIL: resume:$name base run did not write state_final.bin"
    tail -60 "$base_log"
    exit 1
  fi
  if "$BIN" train \
    --dataset "$DATASET" \
    --tokenizer "$TOKENIZER" \
    --size tiny \
    --batch-size 2 \
    --seq-len 16 \
    --steps 4 \
    --warmup 1 \
    --lr 0.001 \
    --checkpoint-dir "$resumed" \
    --resume "$base/state_final.bin" \
    "$@" > "$resume_log" 2>&1; then
    :
  else
    echo "FAIL: resume:$name resume run (see $resume_log)"
    tail -80 "$resume_log"
    exit 1
  fi
  if grep -E 'FATAL|NaN detected|loss is (NaN|inf|-inf)' "$base_log" "$resume_log" >/dev/null; then
    echo "FAIL: resume:$name emitted a fatal/non-finite loss signal"
    tail -80 "$resume_log"
    exit 1
  fi
  if ! grep -q "Resuming at step 2/4" "$resume_log"; then
    echo "FAIL: resume:$name did not continue at the next saved step"
    tail -80 "$resume_log"
    exit 1
  fi
  if [[ -s "$base/state_final.bin.opt" ]] && ! grep -q "Restored '" "$resume_log"; then
    echo "FAIL: resume:$name did not restore optimizer sidecar state"
    tail -80 "$resume_log"
    exit 1
  fi
  local last_step last_tokens
  read -r last_step last_tokens < <(awk -F, 'NR > 1 { step=$1; tokens=$6 } END { print step, tokens }' "$resumed/train.csv")
  if [[ "$last_step" != "3" || "$last_tokens" != "128" ]]; then
    echo "FAIL: resume:$name ended at step=$last_step tokens=$last_tokens, expected step=3 tokens=128"
    cat "$resumed/train.csv"
    exit 1
  fi
  echo "PASS: resume:$name"
}

cd "$REPO" || { echo "FAIL: repo $REPO not found"; exit 2; }

run_logged build cargo build --release --no-default-features --features metal
printf 'not-a-checkpoint' > "$BAD_CHECKPOINT"
run_reject_logged info_bad_checkpoint "not a valid AndreAI checkpoint" "$BIN" info --checkpoint "$BAD_CHECKPOINT"
run_reject_logged tokenizer_missing_input "Failed to read input file" "$BIN" tokenizer --input "$OUT/missing-corpus.txt" --vocab-size 260 --output "$OUT/missing-tokenizer.bin"
run_reject_logged prepare_missing_tokenizer "Failed to load tokenizer" "$BIN" prepare --input "$CORPUS" --tokenizer "$OUT/missing-tokenizer.bin" --output "$OUT/missing-dataset.bin"
printf 'not-a-tokenizer' > "$BAD_TOKENIZER"
run_reject_logged prepare_bad_tokenizer "not a valid tokenizer file" "$BIN" prepare --input "$CORPUS" --tokenizer "$BAD_TOKENIZER" --output "$OUT/bad-tokenizer-dataset.bin"
run_logged tokenizer "$BIN" tokenizer --input "$CORPUS" --vocab-size 260 --output "$TOKENIZER"
run_logged prepare "$BIN" prepare --input "$CORPUS" --tokenizer "$TOKENIZER" --output "$DATASET"
printf '\001\002\003' > "$BAD_SHARD"

run_reject_logged mix_zero_weight "data mixing weights must sum to > 0" "$BIN" mix --shards "$DATASET:0" --output "$OUT/mix-zero.bin"
run_reject_logged mix_malformed_shard "byte length must be a multiple of 4" "$BIN" mix --shards "$BAD_SHARD:1" --output "$OUT/mix-bad.bin"
run_reject_logged train_missing_dataset "Failed to verify dataset" "$BIN" train --dataset "$OUT/missing-dataset.bin" --tokenizer "$TOKENIZER" --size tiny --batch-size 2 --seq-len 16 --steps 1 --warmup 1 --lr 0.001 --checkpoint-dir "$OUT/train_missing_dataset"
run_reject_logged train_unknown_size "Unknown model size" "$BIN" train --dataset "$DATASET" --tokenizer "$TOKENIZER" --size definitely-not-real --batch-size 2 --seq-len 16 --steps 1 --warmup 1 --lr 0.001 --checkpoint-dir "$OUT/train_unknown_size"
run_reject_logged train_custom_missing_dim "--dim required for custom size" "$BIN" train --dataset "$DATASET" --tokenizer "$TOKENIZER" --size custom --layers 1 --heads 1 --batch-size 2 --seq-len 16 --steps 1 --warmup 1 --lr 0.001 --checkpoint-dir "$OUT/train_custom_missing_dim"
run_reject_train invalid_optimizer "unsupported optimizer" --optimizer definitely-not-real
run_reject_train invalid_lr_schedule "unsupported lr_schedule" --lr-schedule lunar
run_reject_train invalid_yarn_scale "--yarn-scale must be finite and >= 1.0" --yarn-scale 0
run_train adamw
run_resume_train adamw_resume
run_train checkpoint_fused --gradient-checkpointing --fused-ce
run_train hybrid_normuon --optimizer hybrid --normuon
run_resume_train hybrid_resume --optimizer hybrid --normuon
run_train bitnet_accum --grad-accum 2 --bitnet
run_train ssm --ssm
run_train block_sparse --block-sparse-top-k 1 --block-size 4
run_train linear_period --linear-attn-period 2
run_train yarn --yarn-scale 2.0

echo "ALL TRAIN SMOKES PASS"
