#!/bin/bash

# Your Python inference script
PYTHON_SCRIPT="deep.py"

# Path to the model
MODEL_PATH="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-70B-Instruct"

# Tensor parallel size
MP_SIZE=8

# Data type

# Temporary power log file
POWER_LOG="power_log.csv"

# Start monitoring HPU power every second, run in background
hl-smi -Q timestamp,power.draw -f csv -l 1 > "$POWER_LOG" &
POWER_PID=$!

# Run the Python model with mpirun
mpirun --allow-run-as-root -np $MP_SIZE \
       python "$PYTHON_SCRIPT" \
       --model-path "$MODEL_PATH" \
       --mp-size $MP_SIZE \
       --dtype bfloat16 

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

# Calculate average power
if (( COUNT > 0 )); then
  AVG=$((SUM / COUNT))
else
  AVG=0
fi

# Print results
echo ""
echo "HPU Power Usage Summary:"
echo "Average Power: ${AVG} W"
echo "Maximum Power: ${MAX} W"
echo "Raw power log saved in $POWER_LOG"

