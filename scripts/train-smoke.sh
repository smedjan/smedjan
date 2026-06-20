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
SFT_JSONL="$OUT/sft.jsonl"
DPO_JSONL="$OUT/dpo.jsonl"
DPO_BIN="$OUT/dpo.bin"
QBIN="$OUT/adamw.qbin"

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

run_sft() {
  local name=$1
  local ckpt=$2
  local out_dir="$OUT/$name"
  local log="$LOG_DIR/$name.log"
  echo "---- sft:$name ----"
  if "$BIN" sft \
    --checkpoint "$ckpt" \
    --tokenizer "$TOKENIZER" \
    --data "$SFT_JSONL" \
    --steps 1 \
    --warmup 0 \
    --lr 0.0001 \
    --batch-size 2 \
    --seq-len 64 \
    --output-dir "$out_dir" > "$log" 2>&1; then
    :
  else
    echo "FAIL: sft:$name (see $log)"
    tail -80 "$log"
    exit 1
  fi
  if grep -E 'FATAL|loss is (NaN|inf|-inf)|non-finite' "$log" >/dev/null; then
    echo "FAIL: sft:$name emitted a fatal/non-finite loss signal"
    tail -80 "$log"
    exit 1
  fi
  if [[ ! -s "$out_dir/sft_final.bin" ]]; then
    echo "FAIL: sft:$name did not write final checkpoint"
    tail -80 "$log"
    exit 1
  fi
  echo "PASS: sft:$name"
}

run_dpo() {
  local name=$1
  local ckpt=$2
  local out_dir="$OUT/$name"
  local log="$LOG_DIR/$name.log"
  echo "---- dpo:$name ----"
  if "$BIN" dpo \
    --checkpoint "$ckpt" \
    --ref-checkpoint "$ckpt" \
    --tokenizer "$TOKENIZER" \
    --dataset "$DPO_BIN" \
    --steps 1 \
    --warmup 0 \
    --lr 0.000001 \
    --beta 0.1 \
    --max-seq-len 64 \
    --output-dir "$out_dir" > "$log" 2>&1; then
    :
  else
    echo "FAIL: dpo:$name (see $log)"
    tail -80 "$log"
    exit 1
  fi
  if grep -E 'FATAL|loss is (NaN|inf|-inf)|non-finite' "$log" >/dev/null; then
    echo "FAIL: dpo:$name emitted a fatal/non-finite loss signal"
    tail -80 "$log"
    exit 1
  fi
  if [[ ! -s "$out_dir/dpo_final.bin" ]]; then
    echo "FAIL: dpo:$name did not write final checkpoint"
    tail -80 "$log"
    exit 1
  fi
  echo "PASS: dpo:$name"
}

run_conversion_smokes() {
  local ckpt=$1
  echo "---- convert:quantize_q8 ----"
  run_logged quantize_q8 "$BIN" quantize --checkpoint "$ckpt" --output "$QBIN" --bits 8
  if [[ ! -s "$QBIN" ]]; then
    echo "FAIL: convert:quantize_q8 did not write $QBIN"
    exit 1
  fi

  echo "---- convert:generate_qbin ----"
  run_logged generate_qbin "$BIN" generate --checkpoint "$QBIN" --tokenizer "$TOKENIZER" --prompt "AndreAI" --max-tokens 1 --temperature 0

  echo "---- convert:export_gguf ----"
  run_logged export_gguf "$BIN" export-gguf --checkpoint "$ckpt" --output "$OUT/model.gguf" --quant f32
  if [[ ! -s "$OUT/model.gguf" ]]; then
    echo "FAIL: convert:export_gguf did not write output"
    exit 1
  fi

  echo "---- convert:export_safetensors ----"
  run_logged export_safetensors "$BIN" export-safetensors --checkpoint "$ckpt" --output "$OUT/model.safetensors"
  if [[ ! -s "$OUT/model.safetensors" ]]; then
    echo "FAIL: convert:export_safetensors did not write output"
    exit 1
  fi
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
run_reject_logged eval_bad_longctx_length "invalid --longctx-lengths entry 'abc'" "$BIN" eval --checkpoint "$BAD_CHECKPOINT" --tokenizer "$OUT/missing-tokenizer.bin" --longctx --longctx-lengths abc --longctx-depths 0.5
run_reject_logged eval_bad_longctx_depth "--longctx-depths entries must be finite and in [0, 1]" "$BIN" eval --checkpoint "$BAD_CHECKPOINT" --tokenizer "$OUT/missing-tokenizer.bin" --longctx --longctx-lengths 64 --longctx-depths 1.5
run_reject_logged generate_bad_temperature "--temperature must be finite and >= 0" "$BIN" generate --checkpoint "$BAD_CHECKPOINT" --tokenizer "$OUT/missing-tokenizer.bin" --temperature=-1
run_reject_logged generate_bad_top_p "--top-p must be finite and in (0, 1]" "$BIN" generate --checkpoint "$BAD_CHECKPOINT" --tokenizer "$OUT/missing-tokenizer.bin" --top-p 0
run_reject_logged generate_bad_max_tokens "--max-tokens must be greater than 0" "$BIN" generate --checkpoint "$BAD_CHECKPOINT" --tokenizer "$OUT/missing-tokenizer.bin" --max-tokens 0
run_reject_logged generate_bad_draft_tokens "--draft-tokens must be greater than 0" "$BIN" generate --checkpoint "$BAD_CHECKPOINT" --tokenizer "$OUT/missing-tokenizer.bin" --speculative --draft-tokens 0
run_reject_logged tokenizer_missing_input "Failed to read input file" "$BIN" tokenizer --input "$OUT/missing-corpus.txt" --vocab-size 260 --output "$OUT/missing-tokenizer.bin"
run_reject_logged prepare_missing_tokenizer "Failed to load tokenizer" "$BIN" prepare --input "$CORPUS" --tokenizer "$OUT/missing-tokenizer.bin" --output "$OUT/missing-dataset.bin"
printf 'not-a-tokenizer' > "$BAD_TOKENIZER"
run_reject_logged prepare_bad_tokenizer "not a valid tokenizer file" "$BIN" prepare --input "$CORPUS" --tokenizer "$BAD_TOKENIZER" --output "$OUT/bad-tokenizer-dataset.bin"
run_logged tokenizer "$BIN" tokenizer --input "$CORPUS" --vocab-size 260 --output "$TOKENIZER"
run_logged prepare "$BIN" prepare --input "$CORPUS" --tokenizer "$TOKENIZER" --output "$DATASET"
printf '\001\002\003' > "$BAD_SHARD"

run_reject_logged mix_zero_weight "data mixing weights must sum to > 0" "$BIN" mix --shards "$DATASET:0" --output "$OUT/mix-zero.bin"
run_reject_logged mix_malformed_shard "byte length must be a multiple of 4" "$BIN" mix --shards "$BAD_SHARD:1" --output "$OUT/mix-bad.bin"
cat > "$SFT_JSONL" <<'SFT'
{"prompt":"Say hello.","response":"Hello."}
{"prompt":"Name the runtime.","response":"AndreAI runs local training."}
SFT
cat > "$DPO_JSONL" <<'DPO'
{"prompt":"Say hello.","chosen":"Hello.","rejected":"Goodbye."}
DPO
run_reject_logged sft_invalid_seq_len "seq_len must be greater than 0" "$BIN" sft --checkpoint "$BAD_CHECKPOINT" --tokenizer "$TOKENIZER" --data "$SFT_JSONL" --seq-len 0
run_reject_logged dpo_invalid_beta "beta must be finite and > 0" "$BIN" dpo --checkpoint "$BAD_CHECKPOINT" --ref-checkpoint "$BAD_CHECKPOINT" --tokenizer "$TOKENIZER" --dataset "$OUT/missing-dpo.bin" --beta 0
run_reject_logged distill_zero_samples "n_samples must be greater than 0" "$BIN" distill --output "$OUT/distill-zero.jsonl" --n-samples 0
run_reject_logged distill_missing_api_key "api_key must not be empty" "$BIN" distill --api-url https://api.openai.com/v1/chat/completions --model gpt-4o --output "$OUT/distill-openai.jsonl" --n-samples 1
run_reject_logged quantize_bad_bits "bits must be 4 or 8" "$BIN" quantize --checkpoint "$BAD_CHECKPOINT" --output "$OUT/bad-bits.qbin" --bits 3
run_reject_logged train_missing_dataset "Failed to verify dataset" "$BIN" train --dataset "$OUT/missing-dataset.bin" --tokenizer "$TOKENIZER" --size tiny --batch-size 2 --seq-len 16 --steps 1 --warmup 1 --lr 0.001 --checkpoint-dir "$OUT/train_missing_dataset"
run_reject_logged train_unknown_size "Unknown model size" "$BIN" train --dataset "$DATASET" --tokenizer "$TOKENIZER" --size definitely-not-real --batch-size 2 --seq-len 16 --steps 1 --warmup 1 --lr 0.001 --checkpoint-dir "$OUT/train_unknown_size"
run_reject_logged train_custom_missing_dim "--dim required for custom size" "$BIN" train --dataset "$DATASET" --tokenizer "$TOKENIZER" --size custom --layers 1 --heads 1 --batch-size 2 --seq-len 16 --steps 1 --warmup 1 --lr 0.001 --checkpoint-dir "$OUT/train_custom_missing_dim"
run_reject_train invalid_optimizer "unsupported optimizer" --optimizer definitely-not-real
run_reject_train invalid_lr_schedule "unsupported lr_schedule" --lr-schedule lunar
run_reject_train invalid_yarn_scale "--yarn-scale must be finite and >= 1.0" --yarn-scale 0
run_reject_train invalid_grad_accum "grad_accum_steps must be greater than 0" --grad-accum 0
run_reject_train invalid_dropout "dropout must be finite and in [0, 1)" --dropout 1
run_reject_train invalid_moe_top_k "--top-k-experts must be in 1..=--n-experts" --n-experts 2 --top-k-experts 0
run_train adamw
run_reject_logged export_gguf_invalid_quant "unsupported GGUF quantization" "$BIN" export-gguf --checkpoint "$OUT/adamw/final.bin" --output "$OUT/invalid.gguf" --quant q5
run_conversion_smokes "$OUT/adamw/final.bin"
run_sft sft_tiny "$OUT/adamw/final.bin"
run_logged dpo_prepare "$BIN" dpo-prepare --input "$DPO_JSONL" --output "$DPO_BIN" --tokenizer "$TOKENIZER"
run_dpo dpo_tiny "$OUT/adamw/final.bin"
run_resume_train adamw_resume
run_train checkpoint_fused --gradient-checkpointing --fused-ce
run_train hybrid_normuon --optimizer hybrid --normuon
run_resume_train hybrid_resume --optimizer hybrid --normuon
run_train bitnet_accum --grad-accum 2 --bitnet
run_train ssm --ssm
run_train block_sparse --block-sparse-top-k 1 --block-size 4
run_train linear_period --linear-attn-period 2
run_train yarn --yarn-scale 2.0
run_train moe_preset --n-experts 2 --top-k-experts 1
if ! grep -Fq "n_experts=2, top_k_experts=1" "$LOG_DIR/moe_preset.log"; then
  echo "FAIL: train:moe_preset did not build the requested MoE preset"
  tail -60 "$LOG_DIR/moe_preset.log"
  exit 1
fi

echo "ALL TRAIN SMOKES PASS"
