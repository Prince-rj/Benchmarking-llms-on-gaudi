import time 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import deepspeed
from habana_profile import HabanaProfile 
# Model path
model_id = "/software/data/llama_inference/Llama-2-7b-hf/"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# Load model with DeepSpeed inference engine
ds_engine = deepspeed.init_inference(
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32),
    mp_size=1,
    dtype=torch.float32,
    replace_with_kernel_inject=True
)

model = ds_engine.module

# Set up the generation pipeline (not used directly here, just model and tokenizer)
prompt = "Explain Llamma2 and deepspeed inference"
inputs = tokenizer(prompt, return_tensors="pt")

# Move input to model device
for k in inputs:
    inputs[k] = inputs[k].to(model.device)

# Generation parameters with randomness
generation_kwargs = {
    "max_new_tokens": 500,
    "do_sample": True,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.95,
    "return_dict_in_generate": True,
    "output_scores": True
}

# Warmup
_ = model.generate(**inputs, max_new_tokens=50, do_sample=True)

# Actual inference
start_time = time.time()
with torch.no_grad():
    HabanaProfile.enable()
    profiler = HabanaProfile(warmup=1, active=3, output_dir="./hpu_profile", wait=1)
    profiler.start()
    profiler.step()
    profiler.step()
    generated = model.generate(**inputs, **generation_kwargs)
    profiler.stop()
end_time = time.time()

# Decode the output
decoded = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)
print("Generated text:\n", decoded)

# Benchmarking
total_time = end_time - start_time
num_tokens = generated.sequences.shape[1] - inputs['input_ids'].shape[1]

# Compute inter-token latencies
scores = generated.scores
token_times = np.diff(np.linspace(start_time, end_time, len(scores) + 1))

ttft = token_times[0]
inter_latencies = token_times[1:]

print("\n=== Benchmark Metrics ===")
print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")
print(f"Generated tokens: {num_tokens}")
print(f"Time to First Token (TTFT): {ttft:.4f} sec")
print(f"Inter-Token Latency (Mean): {np.mean(inter_latencies):.4f} sec")
print(f"Inter-Token Latency (Median): {np.median(inter_latencies):.4f} sec")
print(f"Throughput: {num_tokens / total_time:.2f} tokens/sec")

