#!/usr/bin/env bash
# Waits for the background smoke-train to finish, then runs the full calibration battery on the
# trained checkpoint and writes /tmp/sj_cal/REPORT.md. Run via Bash run_in_background so the harness
# re-invokes when train+calibrate are both done.
set -u
cd ~/projects/smedjan
BIN=./target/release/smedjan
OUT=/tmp/sj_cal
CKPT=$OUT/final.bin
TOK=data/tokenizer_v2.bin
HELD=$OUT/heldout.txt
VENVPY=/private/tmp/claude-501/-Users-Andrei/a667c1bd-e4c7-4712-84c7-dc4d14ed9327/scratchpad/ggufvenv/bin/python
R=$OUT/REPORT.md

# 1. Wait for training to complete (it prints "Final EMA loss" at the end).
until grep -q "Final EMA loss" $OUT/train.log 2>/dev/null; do sleep 20; done
sleep 8

{
  echo "# Smedjan smoke-train + calibration"
  echo
  echo "## Training — Smedjan-7M (d256/6L), seq256, batch32, 1500 steps, AdamW + cosine, train_v3"
  echo '```'
  grep -E "^step" $OUT/train.log | head -1
  grep -E "^step" $OUT/train.log | sed -n '0~150p' | tail -10
  grep -E "^step" $OUT/train.log | tail -1
  grep -E "Final EMA loss|Best:|Peak throughput|Avg throughput|Epochs" $OUT/train.log | tail -5
  echo '```'
} > $R

echo -e "\n## Perplexity — held-out text" >> $R
echo '```' >> $R
$BIN perplexity --checkpoint $CKPT --tokenizer $TOK --file $HELD 2>&1 | grep -ivE "Metal device|loaded tensor|Loading checkpoint|Model initialized|Checkpoint loaded|Smedjan v" | tail -8 >> $R
echo '```' >> $R

echo -e "\n## Generation — greedy-ish sample (temp 0.7, 60 tok)" >> $R
echo '```' >> $R
$BIN generate --checkpoint $CKPT --tokenizer $TOK --prompt "The system" --max-tokens 60 --temperature 0.7 2>&1 | grep -ivE "Metal device|loaded tensor|Loading checkpoint|Model initialized|Checkpoint loaded|Smedjan v" | tail -10 >> $R
echo '```' >> $R

echo -e "\n## Long-context — NIAH / RULER (L256, depths 0/0.5/1.0)" >> $R
echo '```' >> $R
$BIN eval --checkpoint $CKPT --tokenizer $TOK --longctx --longctx-lengths 256 --longctx-depths 0.0,0.5,1.0 2>&1 | grep -iE "category|niah|multikey|freqagg|vartrace|Total examples|%" | tail -10 >> $R
echo '```' >> $R

echo -e "\n## Throughput — bench (small, b4 seq128, 20 iters)" >> $R
echo '```' >> $R
$BIN bench --size small --batch-size 4 --seq-len 128 --iters 20 2>&1 | grep -iE "tok/s|forward|decode|train|fwd|throughput" | tail -8 >> $R
echo '```' >> $R

echo -e "\n## Quantization calibration — GGUF on the TRAINED model" >> $R
echo '```' >> $R
for q in f32 q8_0 q4_0; do
  $BIN export-gguf --checkpoint $CKPT --output $OUT/sj_$q.gguf --quant $q 2>&1 | grep -i exported >> $R
done
# Reference-dequantizer error vs f32 (the real quant-quality calibration on trained weights).
if [ -x "$VENVPY" ]; then
  "$VENVPY" - "$OUT" >> $R 2>/dev/null <<'PY'
import sys, numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
import gguf.quants as q
out=sys.argv[1]
def deq(t):
    return (t.data.astype(np.float32).ravel() if t.tensor_type==GGMLQuantizationType.F32
            else q.dequantize(t.data,t.tensor_type).astype(np.float32).ravel())
ref={t.name:deq(t) for t in GGUFReader(f"{out}/sj_f32.gguf").tensors}
for lab,path in [("Q8_0",f"{out}/sj_q8_0.gguf"),("Q4_0",f"{out}/sj_q4_0.gguf")]:
    w=0.0;nq=0
    for t in GGUFReader(path).tensors:
        if t.tensor_type==GGMLQuantizationType.F32: continue
        nq+=1; g=q.dequantize(t.data,t.tensor_type).astype(np.float32).ravel(); r=ref[t.name]
        n=min(len(g),len(r)); w=max(w,np.abs(g[:n]-r[:n]).max()/max(1e-6,np.abs(r[:n]).max()))
    print(f"{lab}: {nq} quantized tensors, worst rel err vs f32 = {w:.3%}")
PY
fi
echo '```' >> $R
echo -e "\nCALIBRATION_DONE" >> $R
