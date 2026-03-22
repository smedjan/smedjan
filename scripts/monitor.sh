#!/bin/bash
# AndreAI Training Monitor
# Checks training progress and alerts on completion/failure
# Usage: ./scripts/monitor.sh [interval_seconds]
# Default: checks every 60 seconds

INTERVAL=${1:-60}
PROJ="/Users/Andrei/projects/andreai"
CKPT_DIR="$PROJ/data/checkpoints"

echo "=== AndreAI Training Monitor ==="
echo "Checking every ${INTERVAL}s. Ctrl+C to stop."
echo ""

last_ckpt_count=0

while true; do
    # Check if training process is running
    pid=$(pgrep -f "andreai train" 2>/dev/null | head -1)

    if [ -z "$pid" ]; then
        echo "[$(date +%H:%M:%S)] Training NOT running."

        # Check if checkpoint exists (training completed)
        if [ -f "$CKPT_DIR/final.bin" ]; then
            size=$(ls -lh "$CKPT_DIR/final.bin" 2>/dev/null | awk '{print $5}')
            echo "[$(date +%H:%M:%S)] Final checkpoint exists: $size"
            echo "[$(date +%H:%M:%S)] Training appears COMPLETE."

            # Run quick eval
            echo "[$(date +%H:%M:%S)] Running eval..."
            "$PROJ/target/release/andreai" eval \
                --checkpoint "$CKPT_DIR/final.bin" \
                --tokenizer "$PROJ/data/tokenizer_v2.bin" 2>/dev/null | grep -E "Exact|Partial|Total"

            echo ""
            echo "Next steps:"
            echo "  1. Run SFT: andreai sft --checkpoint data/checkpoints/final.bin --tokenizer data/tokenizer_v2.bin --data data/sft_combined.jsonl --steps 5000 --lr 2e-5"
            echo "  2. Or run full pipeline: ./scripts/train_pipeline.sh"
            break
        else
            echo "[$(date +%H:%M:%S)] No checkpoint found. Training may have failed."
            break
        fi
    fi

    # Get process info
    cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ')
    mem=$(ps -p "$pid" -o %mem= 2>/dev/null | tr -d ' ')
    time_info=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')

    # Count checkpoints
    ckpt_count=$(ls "$CKPT_DIR"/step_*.bin 2>/dev/null | wc -l | tr -d ' ')

    # Check for new checkpoint
    new_ckpt=""
    if [ "$ckpt_count" -gt "$last_ckpt_count" ]; then
        latest=$(ls -t "$CKPT_DIR"/step_*.bin 2>/dev/null | head -1)
        new_ckpt=" NEW: $(basename "$latest")"
        last_ckpt_count=$ckpt_count
    fi

    echo "[$(date +%H:%M:%S)] PID=$pid CPU=${cpu}% MEM=${mem}% elapsed=$time_info checkpoints=$ckpt_count$new_ckpt"

    sleep "$INTERVAL"
done
