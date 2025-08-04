import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Path to your model
model_id = "/software/data/llama_inference/Llama-2-7b-hf/"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt and generation config
prompt = "Hello, my name is"
max_new_tokens = 50

# Warmup (first call may include compilation on HPU)
pipe(prompt, max_new_tokens=5)

# Tokenize prompt and calculate input length
input_tokens = tokenizer(prompt, return_tensors="pt")
input_token_count = input_tokens["input_ids"].shape[1]

# Run and benchmark
start_time = time.time()
output = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
end_time = time.time()

generated_text = output[0]["generated_text"]
generated_tokens = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]

# Inter-token timings (simulate, since HF pipeline doesn’t give per-token latencies directly)
# If you want real per-token latency, you must run with `generate()` looped over each token (less efficient)
inter_token_latencies = []  # simulate single-token generation for accurate timings
current_input = input_tokens
generated_token_count = 0

# Optional accurate measurement using manual generation
# Not mandatory unless per-token stats needed
manual = True
if manual:
    current_input = tokenizer(prompt, return_tensors="pt")
    current_input = {k: v.to(model.device) for k, v in current_input.items()}
    generated = current_input["input_ids"]
    past_key_values = None
    latencies = []
    for _ in range(max_new_tokens):
        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=True)
        t1 = time.time()
        latencies.append(t1 - t0)

        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token_id], dim=-1)
        past_key_values = outputs.past_key_values

    ttft = latencies[0]
    inter_latencies = latencies[1:]
    mean_latency = np.mean(inter_latencies)
    median_latency = np.median(inter_latencies)
    total_tokens = len(latencies)
    total_time = sum(latencies)
else:
    ttft = end_time - start_time
    mean_latency = ttft / max_new_tokens
    median_latency = mean_latency
    total_tokens = max_new_tokens
    total_time = end_time - start_time

throughput = total_tokens / total_time

# Print metrics
print(f"=== Benchmark Results ===")
print(f"Prompt length: {input_token_count} tokens")
print(f"Generated tokens: {total_tokens}")
print(f"Time to First Token (TTFT): {ttft:.4f} sec")
print(f"Inter-Token Latency (Mean): {mean_latency:.4f} sec")
print(f"Inter-Token Latency (Median): {median_latency:.4f} sec")
print(f"Throughput: {throughput:.2f} tokens/sec")
print(f"Total time: {total_time:.4f} sec")

