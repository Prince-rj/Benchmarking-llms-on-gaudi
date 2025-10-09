#!/usr/bin/env python3
# deep.py
import os
import argparse
import time
import numpy as np
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ---------------------------
# Timing Streamer for TTFT & ITL
# ---------------------------
class TimingStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.timestamps = []

    def on_finalized_text(self, text, stream_end=False):
        self.timestamps.append(time.time())
        super().on_finalized_text(text, stream_end)

# ---------------------------
# Argument Parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to LLaMA model")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16","fp32"])
    parser.add_argument("--mp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--prompt", type=str, default="Write a detailed 1000-word helpful sentence about Gaudi inference:")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    return parser.parse_args()

def get_dtype(dtype_str):
    if dtype_str == "bfloat16": return torch.bfloat16
    if dtype_str == "float16": return torch.float16
    return torch.float32

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    dtype = get_dtype(args.dtype)

    # Get MPI rank (rank 0 prints output)
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    if rank == 0:
        print(f"Loading model from {args.model_path} (dtype={args.dtype})")

    # Load model to CPU first (low memory usage)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="cpu",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    if rank == 0:
        print(f"Initializing DeepSpeed inference with TP size={args.mp_size}")

    # DeepSpeed tensor parallel
    model = deepspeed.init_inference(
        model,
        dtype=dtype,
        tensor_parallel={"tp_size": args.mp_size}
    )   

    # Set HPU device
    device = torch.device("hpu")

    # Encode prompt
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Benchmarking streamer
    streamer = TimingStreamer(tokenizer)

    # ---- Generation & Timing ----
    if rank == 0:
        print("Running inference and collecting benchmarks...")

    start_time = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            do_sample=True,               # enable sampling to prevent EOS stopping early
            early_stopping=False,         # ignore early stopping
            pad_token_id=tokenizer.eos_token_id,  # use EOS token as padding
            eos_token_id=None,
            num_beams=1
        )
    end_time = time.time()

    # ---- Compute Metrics (only rank 0) ----
    if rank == 0:
        timestamps = streamer.timestamps
        if len(timestamps) > 0:
            ttft = timestamps[0] - start_time
            itl = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            mean_itl = np.mean(itl) if itl else 0
            median_itl = np.median(itl) if itl else 0
        else:
            ttft = mean_itl = median_itl = 0 

        gen_tokens = output.shape[1] - inputs['input_ids'].shape[1]
        total_time = end_time - start_time
        throughput = gen_tokens / total_time if total_time > 0 else 0

        # Print benchmark results
        print("\n=== Benchmark Results ===")
        print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")
        print(f"Generated tokens: {gen_tokens}")
        print(f"Time to First Token (TTFT): {ttft:.4f} sec")
        print(f"Inter-Token Latency (Mean): {mean_itl:.4f} sec")
        print(f"Inter-Token Latency (Median): {median_itl:.4f} sec")
        print(f"Throughput: {throughput:.2f} tokens/sec")

        # Decode and print generated text
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print("\n=== Generated Output ===")
        print(result)

if __name__ == "__main__":
    main()

