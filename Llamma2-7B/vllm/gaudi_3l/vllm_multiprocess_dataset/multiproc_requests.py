import requests
import time
import csv
import re
import os
import signal
import subprocess
import multiprocessing as mp
from datasets import load_dataset
import json

# ---------------- CONFIG ----------------
VLLM_URL = "http://localhost:8000/v1/completions"
MODEL_ID = "/mnt/weka/data/pytorch/llama2/Llama-2-7b-chat-hf/"
OUTPUT_CSV = "alpaca_benchmark_results.csv"
POWER_LOG = "power_log.csv"
NUM_SAMPLES = 520        # None → load all (≈52,002 samples)
BATCH_SIZE = 8           # run this many in parallel
MAX_TOKENS = 512
# ----------------------------------------


def start_power_logging():
    """Start hl-smi power logging."""
    print("🟢 Starting power monitoring...")
    proc = subprocess.Popen(
        ["hl-smi", "-Q", "timestamp,power.draw", "-f", "csv", "-l", "1"],
        stdout=open(POWER_LOG, "w"),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def stop_power_logging(proc):
    """Stop power logging and print stats."""
    print("\n🛑 Stopping power logging...")
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=5)

    power_values = []
    with open(POWER_LOG, "r") as f:
        for line in f:
            if "timestamp" in line or "power.draw" in line:
                continue
            match = re.search(r"(\d+(\.\d+)?)\s*W", line)
            if match:
                power_values.append(float(match.group(1)))

    if not power_values:
        print("⚠️ No power data recorded.")
        return

    avg_power = sum(power_values) / len(power_values)
    max_power = max(power_values)
    print(f"📊 Samples: {len(power_values)}")
    print(f"📊 Average Power: {avg_power:.1f} W")
    print(f"📊 Peak Power: {max_power:.1f} W")
    print(f"📁 Power Log: {POWER_LOG}")


def build_prompt(sample):
    """Format Alpaca sample to LLM prompt."""
    if sample["input"]:
        return f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nResponse:"
    else:
        return f"Instruction: {sample['instruction']}\nResponse:"


def run_prompt(prompt):
    """Query vLLM endpoint with a single prompt and collect performance metrics."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "stream": True
    }

    start_time = time.time()
    try:
        response = requests.post(VLLM_URL, headers=headers, json=payload, stream=True, timeout=180)
    except Exception as e:
        print(f"⚠️ Request failed: {e}")
        return None

    timestamps = []
    token_count = 0
    output_text = ""
    pattern = re.compile(r"data:\s*({.*})")

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        match = pattern.search(line)
        if match:
            timestamps.append(time.time())
            try:
                data = json.loads(match.group(1))
                if "choices" in data and len(data["choices"]) > 0:
                    token = data["choices"][0].get("text", "")
                    output_text += token
                    token_count += 1
            except json.JSONDecodeError:
                continue

    end_time = time.time()
    if len(timestamps) < 2:
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


def worker(prompt):
    """Worker function for multiprocessing."""
    try:
        return run_prompt(prompt)
    except Exception as e:
        print(f"⚠️ Worker failed: {e}")
        return None


def batched_run(prompts, batch_size=8):
    """Run prompts in parallel batches using multiprocessing."""
    results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        print(f"\n🚀 Running batch {i // batch_size + 1}/{total_batches} ({len(batch)} prompts)")
        with mp.Pool(batch_size) as pool:
            batch_results = pool.map(worker, batch)
        results.extend(batch_results)
    return results


def main():
    # Start power logging
    power_proc = start_power_logging()

    # ✅ Load full Alpaca dataset
    if NUM_SAMPLES:
        dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{NUM_SAMPLES}]")
    else:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = [build_prompt(s) for s in dataset]
    print(f"✅ Loaded {len(prompts)} prompts from Alpaca dataset")

    # Run all prompts in parallel batches
    start = time.time()
    results = batched_run(prompts, batch_size=BATCH_SIZE)
    end = time.time()

    # Stop power monitoring
    stop_power_logging(power_proc)

    # Write benchmark results
    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["prompt", "tokens_generated", "ttft", "mean_itl", "median_itl", "throughput"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r:
                writer.writerow(r)

    print(f"\n✅ Benchmark complete. {len([r for r in results if r])} successful prompts.")
    print(f"⏱️ Total Time: {end - start:.2f}s")
    print(f"📁 Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

