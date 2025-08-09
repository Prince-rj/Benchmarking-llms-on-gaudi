import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import deepspeed

# Model path
model_id = "/software/data/llama_inference/Llama-2-7b-hf/"  # adjust as per your setup

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with DeepSpeed inference engine
ds_engine = deepspeed.init_inference(
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32),
    mp_size=1,               # Number of GPUs / HPUs if using model parallelism
    dtype=torch.float32,     # Use bfloat16/float16 if supported and safe
    replace_method="auto",   # Replace transformer layers with optimized kernels
    replace_with_kernel_inject=True
)

model = ds_engine.module  # Get the actual model from the DeepSpeed engine

# Set up the generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Input prompt
prompt = "Explain Llamma2 and deepspeed inference"
inputs = tokenizer(prompt, return_tensors="pt")

# Move input to model device (e.g., HPU)
for k in inputs:
    inputs[k] = inputs[k].to(model.device)

# Start benchmark
max_new_tokens = 50

# Warmup (optional but recommended)
_ = model.generate(**inputs, max_new_tokens=5)

# Actual inference
start_time = time.time()
with torch.no_grad():
    generated_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True)
end_time = time.time()

# Decode the output
decoded = tokenizer.decode(generated_tokens.sequences[0], skip_special_tokens=True)
print("Generated text:\n", decoded)

# Benchmarking
total_time = end_time - start_time
num_tokens = generated_tokens.sequences.shape[1] - inputs['input_ids'].shape[1]

# Compute inter-token latencies
scores = generated_tokens.scores  # one per token after the prompt
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

