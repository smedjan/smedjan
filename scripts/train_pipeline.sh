#!/bin/bash
# AndreAI Full Training Pipeline
# Runs: pre-train → eval → SFT → eval → quantize
# Usage: ./scripts/train_pipeline.sh
# Logs to: logs/pipeline_$(date).log

set -euo pipefail

PROJ="/Users/Andrei/projects/andreai"
BIN="$PROJ/target/release/andreai"
DATA="$PROJ/data"
LOG_DIR="$PROJ/logs"
CKPT_DIR="$DATA/checkpoints"
SFT_DIR="$DATA/sft_checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$SFT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Build first
log "=== Building release binary ==="
cd "$PROJ"
cargo build --release 2>&1 | tail -1 | tee -a "$LOG"

# ============================================================
# Phase 1: Pre-training
# ============================================================
log "=== Phase 1: Pre-training ==="
log "Dataset: $DATA/train_v3.bin"
log "Tokenizer: $DATA/tokenizer_v2.bin"

$BIN train \
  --dataset "$DATA/train_v3.bin" \
  --tokenizer "$DATA/tokenizer_v2.bin" \
  --size custom --dim 256 --layers 6 --heads 4 \
  --batch-size 32 \
  --seq-len 256 \
  --steps 15000 \
  --lr 1e-3 \
  --warmup 1000 \
  --checkpoint-dir "$CKPT_DIR" 2>&1 | tee -a "$LOG"

log "Pre-training complete."

# ============================================================
# Phase 2: Evaluate pre-trained model
# ============================================================
log "=== Phase 2: Evaluate pre-trained model ==="

$BIN eval \
  --checkpoint "$CKPT_DIR/final.bin" \
  --tokenizer "$DATA/tokenizer_v2.bin" 2>&1 | tee -a "$LOG"

log "Pre-training eval complete."

# ============================================================
# Phase 3: SFT Fine-tuning
# ============================================================
log "=== Phase 3: SFT Fine-tuning ==="

$BIN sft \
  --checkpoint "$CKPT_DIR/final.bin" \
  --tokenizer "$DATA/tokenizer_v2.bin" \
  --data "$DATA/sft_combined.jsonl" \
  --steps 5000 \
  --lr 2e-5 \
  --batch-size 4 \
  --seq-len 256 \
  --warmup 200 \
  --output-dir "$SFT_DIR" 2>&1 | tee -a "$LOG"

log "SFT complete."

# ============================================================
# Phase 4: Evaluate SFT model
# ============================================================
log "=== Phase 4: Evaluate SFT model ==="

$BIN eval \
  --checkpoint "$SFT_DIR/final.bin" \
  --tokenizer "$DATA/tokenizer_v2.bin" 2>&1 | tee -a "$LOG"

log "SFT eval complete."

# ============================================================
# Phase 5: Quantize
# ============================================================
log "=== Phase 5: Quantize ==="

$BIN quantize \
  --checkpoint "$SFT_DIR/final.bin" \
  --output "$SFT_DIR/model.q8.qbin" \
  --bits 8 2>&1 | tee -a "$LOG"

$BIN quantize \
  --checkpoint "$SFT_DIR/final.bin" \
  --output "$SFT_DIR/model.q4.qbin" \
  --bits 4 2>&1 | tee -a "$LOG"

log "Quantization complete."

# ============================================================
# Phase 6: Test generation from all models
# ============================================================
log "=== Phase 6: Generation test ==="

for model in "$CKPT_DIR/final.bin" "$SFT_DIR/final.bin" "$SFT_DIR/model.q8.qbin" "$SFT_DIR/model.q4.qbin"; do
  if [ -f "$model" ]; then
    log "--- $(basename $model) ---"
    $BIN generate \
      --checkpoint "$model" \
      --tokenizer "$DATA/tokenizer_v2.bin" \
      --prompt "find all files larger than 100MB" \
      --max-tokens 50 \
      --temperature 0.3 2>&1 | tee -a "$LOG"
  fi
done

log "=== Pipeline complete ==="
log "Log: $LOG"
log "Checkpoints: $CKPT_DIR/final.bin, $SFT_DIR/final.bin"
log "Quantized: $SFT_DIR/model.q8.qbin, $SFT_DIR/model.q4.qbin"
