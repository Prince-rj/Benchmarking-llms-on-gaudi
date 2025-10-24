#!/bin/bash

# Your Python inference script
PYTHON_SCRIPT="benchmark_llamma2.py"

# Temporary power log file
POWER_LOG="power_log.csv"

# Start monitoring power (every second), run in background
hl-smi -Q timestamp,power.draw -f csv -l 1 > "$POWER_LOG" &
POWER_PID=$!

# Run your Python model
mpirun --allow-run-as-root -np 1     python "$PYTHON_SCRIPT"

# Kill power monitoring after model finishes
kill $POWER_PID
wait $POWER_PID 2>/dev/null

# Skip header and extract power values, remove ' W' suffix
POWER_VALUES=$(tail -n +2 "$POWER_LOG" | cut -d',' -f2 | sed 's/ W//g' | tr -d '[:space:]')

# Initialize variables
SUM=0
COUNT=0
MAX=0

for VALUE in $POWER_VALUES; do
  VALUE_INT=${VALUE%.*}  # Remove decimal if any
  SUM=$((SUM + VALUE_INT))
  COUNT=$((COUNT + 1))
  if (( VALUE_INT > MAX )); then
    MAX=$VALUE_INT
  fi
done

# Print results
echo ""
echo "power_usage in power_log.csv"
