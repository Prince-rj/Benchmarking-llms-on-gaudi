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
import base64
from io import BytesIO

# ---------------- CONFIG ----------------
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "/mnt/weka/data/pytorch/Qwen/Qwen2.5-VL-7B-Instruct/"
OUTPUT_CSV_PREFIX = "cord_benchmark_batch"  # Will append batch size
POWER_LOG_PREFIX = "power_log_batch"        # Will append batch size
NUM_SAMPLES = 100        # None → load all CORD-v2 samples
BATCH_SIZES = [2, 4, 8, 12, 16]  # Different batch sizes to test
MAX_TOKENS = 2048        # Model max context is 3072, leaving room for input tokens
PROMPT_TEXT = "Describe this image in detail, including all visible text, layout, and elements."
# ----------------------------------------


def start_power_logging(power_log_file):
    """Start hl-smi power logging."""
    print("🟢 Starting power monitoring...")
    proc = subprocess.Popen(
        ["hl-smi", "-Q", "timestamp,power.draw", "-f", "csv", "-l", "1"],
        stdout=open(power_log_file, "w"),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def stop_power_logging(proc, power_log_file):
    """Stop power logging and print stats."""
    print("\n🛑 Stopping power logging...")
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=5)

    power_values = []
    with open(power_log_file, "r") as f:
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
    print(f"📁 Power Log: {power_log_file}")


def image_to_base64(pil_image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def run_vision_request(image_data_tuple):
    """Query vLLM endpoint with an image and collect performance metrics."""
    sample_idx, pil_image = image_data_tuple
    
    # Convert PIL image to base64
    image_data = image_to_base64(pil_image)
    
    # Prepare the request data
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }
        ],
        "max_tokens": MAX_TOKENS,
        "stream": True
    }
    
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    try:
        response = requests.post(VLLM_URL, headers=headers, json=payload, stream=True, timeout=300)
    except Exception as e:
        print(f"⚠️ Request failed for sample {sample_idx}: {e}")
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
                    delta = data["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    output_text += token
                    if token:
                        token_count += 1
            except json.JSONDecodeError:
                continue

    end_time = time.time()
    if len(timestamps) < 2:
        return None

    ttft = timestamps[0] - start_time
    itl = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    mean_itl = sum(itl) / len(itl) if itl else 0
    median_itl = sorted(itl)[len(itl) // 2] if itl else 0
    total_time = end_time - start_time
    throughput = token_count / total_time if total_time > 0 else 0

    return {
        "sample_idx": sample_idx,
        "tokens_generated": token_count,
        "ttft": round(ttft, 4),
        "mean_itl": round(mean_itl, 4),
        "median_itl": round(median_itl, 4),
        "throughput": round(throughput, 2),
        "total_time": round(total_time, 2)
    }


def worker(image_data_tuple):
    """Worker function for multiprocessing."""
    try:
        return run_vision_request(image_data_tuple)
    except Exception as e:
        print(f"⚠️ Worker failed: {e}")
        return None


def batched_run(image_data_list, batch_size=8):
    """Run image requests in parallel batches using multiprocessing."""
    results = []
    total_batches = (len(image_data_list) + batch_size - 1) // batch_size

    for i in range(0, len(image_data_list), batch_size):
        batch = image_data_list[i:i + batch_size]
        print(f"\n🚀 Running batch {i // batch_size + 1}/{total_batches} ({len(batch)} images)")
        with mp.Pool(batch_size) as pool:
            batch_results = pool.map(worker, batch)
        results.extend(batch_results)
    return results


def main():
    # Load CORD-v2 dataset once
    print("📥 Loading CORD-v2 dataset...")
    if NUM_SAMPLES:
        dataset = load_dataset("naver-clova-ix/cord-v2", split=f"train[:{NUM_SAMPLES}]")
    else:
        dataset = load_dataset("naver-clova-ix/cord-v2", split="train")

    # Prepare image data tuples (index, PIL Image)
    image_data_list = [(idx, sample["image"]) for idx, sample in enumerate(dataset)]
    print(f"✅ Loaded {len(image_data_list)} images from CORD-v2 dataset")
    
    # Run benchmarks for each batch size
    all_results = {}
    
    for batch_size in BATCH_SIZES:
        print(f"\n{'='*60}")
        print(f"🚀 STARTING BENCHMARK FOR BATCH SIZE: {batch_size}")
        print(f"{'='*60}\n")
        
        # Create filenames for this batch size
        output_csv = f"{OUTPUT_CSV_PREFIX}_{batch_size}.csv"
        power_log = f"{POWER_LOG_PREFIX}_{batch_size}.csv"
        
        # Start power logging
        power_proc = start_power_logging(power_log)

        # Run all images in parallel batches
        start = time.time()
        results = batched_run(image_data_list, batch_size=batch_size)
        end = time.time()

        # Stop power monitoring
        stop_power_logging(power_proc, power_log)

        # Write benchmark results
        with open(output_csv, "w", newline="") as f:
            fieldnames = ["sample_idx", "tokens_generated", "ttft", "mean_itl", "median_itl", "throughput", "total_time"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                if r:
                    writer.writerow(r)

        successful_results = [r for r in results if r]
        total_time = end - start
        
        print(f"\n✅ Batch size {batch_size} complete. {len(successful_results)} successful requests.")
        print(f"⏱️ Total Time: {total_time:.2f}s")
        
        # Calculate aggregate metrics
        if successful_results:
            avg_ttft = sum(r["ttft"] for r in successful_results) / len(successful_results)
            avg_throughput = sum(r["throughput"] for r in successful_results) / len(successful_results)
            avg_tokens = sum(r["tokens_generated"] for r in successful_results) / len(successful_results)
            avg_mean_itl = sum(r["mean_itl"] for r in successful_results) / len(successful_results)
            
            print(f"📊 Average TTFT: {avg_ttft:.4f}s")
            print(f"📊 Average Mean ITL: {avg_mean_itl:.4f}s")
            print(f"📊 Average Throughput: {avg_throughput:.2f} tokens/s")
            print(f"📊 Average Tokens Generated: {avg_tokens:.1f}")
            
            # Store summary results
            all_results[batch_size] = {
                "batch_size": batch_size,
                "total_time": round(total_time, 2),
                "successful_requests": len(successful_results),
                "avg_ttft": round(avg_ttft, 4),
                "avg_mean_itl": round(avg_mean_itl, 4),
                "avg_throughput": round(avg_throughput, 2),
                "avg_tokens": round(avg_tokens, 1),
                "csv_file": output_csv,
                "power_log": power_log
            }
        
        print(f"📁 Results saved to: {output_csv}")
    
    # Write summary report
    summary_file = "benchmark_summary.csv"
    with open(summary_file, "w", newline="") as f:
        fieldnames = ["batch_size", "total_time", "successful_requests", "avg_ttft", 
                     "avg_mean_itl", "avg_throughput", "avg_tokens", "csv_file", "power_log"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for batch_size in BATCH_SIZES:
            if batch_size in all_results:
                writer.writerow(all_results[batch_size])
    
    print(f"\n{'='*60}")
    print(f"✅ ALL BENCHMARKS COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Summary saved to: {summary_file}")
    print(f"\nResults by batch size:")
    for batch_size in BATCH_SIZES:
        if batch_size in all_results:
            result = all_results[batch_size]
            print(f"  Batch {batch_size:2d}: {result['total_time']:7.2f}s | "
                  f"Throughput: {result['avg_throughput']:6.2f} tok/s | "
                  f"TTFT: {result['avg_ttft']:.4f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
