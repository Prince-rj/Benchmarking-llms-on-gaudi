#!/bin/bash

PROMPT_FILE="prompts.txt"
PYTHON_SCRIPT="benchmark_vllm.py"
POWER_LOG="power_log.csv"

echo "🟢 Starting power monitoring..."
hl-smi -Q timestamp,power.draw -f csv -l 1 > "$POWER_LOG" &
POWER_PID=$!

# Give it 1-2s to stabilize
sleep 2

echo "🚀 Running benchmarks from $PROMPT_FILE..."
python "$PYTHON_SCRIPT"

echo "🛑 Stopping power logging..."
kill $POWER_PID
wait $POWER_PID 2>/dev/null

# Process power log
echo ""
echo "🔋 === Power Summary ==="
POWER_VALUES=$(tail -n +2 "$POWER_LOG" | cut -d',' -f2 | sed 's/ W//g' | tr -d '[:space:]')
SUM=0
COUNT=0
MAX=0

for VALUE in $POWER_VALUES; do
  VALUE_INT=${VALUE%.*}
  SUM=$((SUM + VALUE_INT))
  COUNT=$((COUNT + 1))
  if (( VALUE_INT > MAX )); then
    MAX=$VALUE_INT
  fi
done

AVG=$((SUM / COUNT))

echo "Samples: $COUNT"
echo "Average Power: $AVG W"
echo "Peak Power: $MAX W"
echo "Benchmark Results: benchmark_results.csv"
echo "Power Log: $POWER_LOG"

