import os
import time
import numpy as np
import torch
import torch.distributed as dist
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

MODEL_PATH = "/software/data/llama_inference/Llama-2-7b-hf"
TOKENS_TO_GENERATE = 1000
PROMPT = "what is Llama2 explain in very detail along with architecture."

class TimingStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.timestamps = []
    def on_finalized_text(self, text, stream_end=False):
        self.timestamps.append(time.time())
        super().on_finalized_text(text, stream_end)

# Distributed init
dist.init_process_group("hccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.hpu.set_device(local_rank)

if local_rank == 0:
    print(f"Running on {dist.get_world_size()} HPUs...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
adapt_transformers_to_gaudi()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": local_rank},   # Explicitly map to HPU per rank
    low_cpu_mem_usage=True
)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt").to("hpu")
streamer = TimingStreamer(tokenizer)

start_time = time.time()
output = model.generate(**inputs, max_new_tokens=TOKENS_TO_GENERATE, streamer=streamer)
end_time = time.time()

if local_rank == 0:
    timestamps = streamer.timestamps
    ttft = timestamps[0] - start_time
    itl = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    mean_itl = np.mean(itl)
    median_itl = np.median(itl)
    gen_tokens = output.shape[1] - inputs.input_ids.shape[1]
    total_time = end_time - start_time
    throughput = gen_tokens / total_time

    print("\n=== Benchmark Results ===")
    print(f"Prompt length: {inputs.input_ids.shape[1]} tokens")
    print(f"Generated tokens: {gen_tokens}")
    print(f"Time to First Token (TTFT): {ttft:.4f} sec")
    print(f"Inter-Token Latency (Mean): {mean_itl:.4f} sec")
    print(f"Inter-Token Latency (Median): {median_itl:.4f} sec")
    print(f"Throughput: {throughput:.2f} tokens/sec")

