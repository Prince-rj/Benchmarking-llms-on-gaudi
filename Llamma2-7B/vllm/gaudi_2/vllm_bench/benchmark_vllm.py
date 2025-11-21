import subprocess
import time
import re
import csv

VLLM_URL = "http://localhost:8000/v1/completions"
MODEL_ID = "/software/data/llama_inference/Llama-2-7b-hf/"
PROMPT_FILE = "prompts.txt"
OUTPUT_CSV = "benchmark_results.csv"
MAX_TOKENS = 1000

def run_prompt(prompt):
    import json

    payload = f"""{{
        "model": "{MODEL_ID}",
        "prompt": "{prompt}",
        "max_tokens": {MAX_TOKENS},
        "stream": true
    }}"""

    start_time = time.time()

    process = subprocess.Popen(
        ["curl", VLLM_URL, "-H", "Content-Type: application/json", "-N", "-d", payload],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    timestamps = []
    token_count = 0
    pattern = re.compile(r'data:\s*({.*})')
    print("\n🧠 LLM Response:")

    for line in process.stdout:
        line = line.strip()
        if line.startswith("data: "):
            match = pattern.search(line)
            if match:
                token_time = time.time()
                timestamps.append(token_time)

                try:
                    json_data = json.loads(match.group(1))
                    if 'choices' in json_data and len(json_data['choices']) > 0:
                        content = json_data['choices'][0].get("text", "")
                        print(content, end="", flush=True)  # Print the token without newline
                        token_count += 1
                except json.JSONDecodeError:
                    continue

    print()  # newline after full output
    end_time = time.time()

    if len(timestamps) < 2:
        return None  # Failed inference or short response

    ttft = timestamps[0] - start_time
    itl = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    mean_itl = sum(itl) / len(itl)
    median_itl = sorted(itl)[len(itl) // 2]
    throughput = token_count / (end_time - start_time)

    return {
        "prompt": prompt,
        "tokens_generated": token_count,
        "ttft": round(ttft, 4),
        "mean_itl": round(mean_itl, 4),
        "median_itl": round(median_itl, 4),
        "throughput": round(throughput, 2)
    }


def main():
    with open(PROMPT_FILE, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = ["prompt", "tokens_generated", "ttft", "mean_itl", "median_itl", "throughput"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, prompt in enumerate(prompts, 1):
            print(f"[{idx}/{len(prompts)}] Running: {prompt[:60]}...")
            result = run_prompt(prompt)
            if result:
                writer.writerow(result)
            else:
                print(f"⚠️ Failed to generate tokens for: {prompt}")

    print(f"\n✅ Benchmark complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()


