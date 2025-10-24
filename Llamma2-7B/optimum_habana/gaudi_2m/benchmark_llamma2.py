import numpy as np
import time, subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

MODEL_PATH = "/software/data/llama_inference/Llama-2-7b-hf"  # change to your local model path
DEVICE = "hpu"  # Gaudi uses HPU
TOKENS_TO_GENERATE = 1000
PROMPT = "what is Llamma2 explain in in very detail along with architecture."


# ---- Timing Streamer for TTFT & ITL ----
class TimingStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.timestamps = []

    def on_finalized_text(self, text, stream_end=False):
        self.timestamps.append(time.time())
        super().on_finalized_text(text, stream_end)

# ---- Load Model ----
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

# ---- Benchmark ----
streamer = TimingStreamer(tokenizer)
print("Running benchmark...")


start_time = time.time()
output = model.generate(**inputs, max_new_tokens=TOKENS_TO_GENERATE, streamer=streamer)
end_time = time.time()

running = False

# ---- Compute Metrics ----
timestamps = streamer.timestamps
ttft = timestamps[0] - start_time
itl = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
mean_itl = np.mean(itl)
median_itl = np.median(itl)

gen_tokens = output.shape[1] - inputs.input_ids.shape[1]
total_time = end_time - start_time
throughput = gen_tokens / total_time


# ---- Results ----
print("\n=== Benchmark Results ===")
print(f"Prompt length: {inputs.input_ids.shape[1]} tokens")
print(f"Generated tokens: {gen_tokens}")
print(f"Time to First Token (TTFT): {ttft:.4f} sec")
print(f"Inter-Token Latency (Mean): {mean_itl:.4f} sec")
print(f"Inter-Token Latency (Median): {median_itl:.4f} sec")
print(f"Throughput: {throughput:.2f} tokens/sec")

