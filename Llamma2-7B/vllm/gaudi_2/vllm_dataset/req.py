import subprocess
import time
import re
import csv
import os
import signal
from datasets import load_dataset
import requests
import json

# ---------------- CONFIG ----------------
VLLM_URL = "http://localhost:8000/v1/completions"
MODEL_ID = "/mnt/weka/data/pytorch/llama2/Llama-2-7b-chat-hf/"
OUTPUT_CSV = "alpaca_benchmark_results.csv"
POWER_LOG = "power_log.csv"
NUM_SAMPLES = None  # Set to None to run all samples
MAX_TOKENS = 512
# ----------------------------------------

def start_power_logging():
    """Start hl-smi power logging."""
    print("🟢 Starting power monitoring...")
    power_proc = subprocess.Popen(
        ["hl-smi", "-Q", "timestamp,power.draw", "-f", "csv", "-l", "1"],
        stdout=open(POWER_LOG, "w"),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # allow readings to stabilize
    return power_proc


def stop_power_logging(power_proc):
    """Stop hl-smi power monitoring and summarize results."""
    print("🛑 Stopping power logging...")
    power_proc.send_signal(signal.SIGINT)
    power_proc.wait(timeout=5)

    power_values = []
    with open(POWER_LOG, "r") as f:
        for line in f:
            if "power.draw" in line or "timestamp" in line:
                continue
            match = re.search(r"(\d+(\.\d+)?)\s*W", line)
            if match:
                power_values.append(float(match.group(1)))

    if not power_values:
        print("⚠️ No power data recorded.")
        return

    avg_power = sum(power_values) / len(power_values)
    max_power = max(power_values)

    print("\n📊 === Power Summary ===")
    print(f"Samples: {len(power_values)}")
    print(f"Average Power: {avg_power:.1f} W")
    print(f"Peak Power: {max_power:.1f} W")
    print(f"Power Log: {POWER_LOG}")


def build_prompt(sample):
    """Format Alpaca sample into prompt."""
    if sample["input"]:
        return f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nResponse:"
    else:
        return f"Instruction: {sample['instruction']}\nResponse:"


def run_prompt(prompt):
    """Send prompt to vLLM and compute token timings."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "stream": True
    }

    start_time = time.time()
    response = requests.post(VLLM_URL, headers=headers, json=payload, stream=True)

    timestamps = []
    token_count = 0
    output_text = ""
    pattern = re.compile(r"data:\s*({.*})")

    print("\n🧠 LLM Response:")

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        match = pattern.search(line)
        if match:
            timestamps.append(time.time())
            try:
                data = json.loads(match.group(1))
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("text", "")
                    print(content, end="", flush=True)
                    output_text += content
                    token_count += 1
            except Exception:
                continue

    print()
    end_time = time.time()

    if len(timestamps) < 2:
        print("⚠️ No valid token timestamps — skipping.")
        return None

    ttft = timestamps[0] - start_time
    itl = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    mean_itl = sum(itl) / len(itl)
    median_itl = sorted(itl)[len(itl) // 2]
    throughput = token_count / (end_time - start_time)

    return {
        "prompt": prompt[:60].replace("\n", " ") + "...",
        "tokens_generated": token_count,
        "ttft": round(ttft, 4),
        "mean_itl": round(mean_itl, 4),
        "median_itl": round(median_itl, 4),
        "throughput": round(throughput, 2)
    }


def main():
    # Start power monitoring
    power_proc = start_power_logging()

    # ✅ Load Alpaca dataset properly
    if NUM_SAMPLES:
        dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{NUM_SAMPLES}]")
    else:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

    print(f"✅ Loaded {len(dataset)} Alpaca samples")

    # Prepare CSV for results
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = ["prompt", "tokens_generated", "ttft", "mean_itl", "median_itl", "throughput"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, sample in enumerate(dataset, 1):
            prompt = build_prompt(sample)
            print(f"\n[{idx}/{len(dataset)}] Running: {prompt[:60]}...")
            result = run_prompt(prompt)
            if result:
                writer.writerow(result)
            else:
                print(f"⚠️ Failed for sample {idx}")

    # Stop and summarize power logging
    stop_power_logging(power_proc)

    print(f"\n✅ Benchmark complete. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

