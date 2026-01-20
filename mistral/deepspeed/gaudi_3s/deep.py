import os
import time
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import habana_frameworks.torch.core as htcore  # HPU ops (mark_step)

###############################################################################
# Config
###############################################################################
MODEL_ID = "/mnt/weka/data/pytorch/mistral/Mistral-7B-Instruct-v0.3/"
PROMPT = "Explain mistral  and DeepSpeed inference."
MAX_NEW_TOKENS = 1000
DTYPE = torch.bfloat16
BACKEND = "hccl"  # Habana comms backend

###############################################################################
# Helpers
###############################################################################
def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def barrier():
    if is_dist():
        dist.barrier()

###############################################################################
# Init distributed (supports both mpirun and single-process runs)
###############################################################################
def init_distributed():
    # If launched with mpirun/torchrun, env vars will be set
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    if world_size > 1 and not is_dist():
        dist.init_process_group(
            backend=BACKEND,
            init_method=f"env://",
            rank=rank,
            world_size=world_size,
        )

    # Map process to HPU
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if hasattr(torch, "hpu"):
        torch.hpu.set_device(local_rank)
    return local_rank

###############################################################################
# Main
###############################################################################
def main():
    local_rank = init_distributed()
    rank = get_rank()
    world_size = get_world_size()

    if rank == 0:
        print(f"[Info] World size: {world_size}, using HPU local_rank={local_rank}")
        print(f"[Info] Loading tokenizer from: {MODEL_ID}")

    # Tokenizer (no 'pipeline' to avoid extra heavy imports)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    if rank == 0:
        print(f"[Info] Loading model on rank {rank} ...")

    # Load model with DeepSpeed Inference (tensor-parallel across world_size)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    ds_engine = deepspeed.init_inference(
        base_model,
        mp_size=world_size,           # shard across N HPUs
        dtype=DTYPE,
        replace_method="auto",
        replace_with_kernel_inject=True
    )

    # DeepSpeed returns an engine; actual model is in .module
    model = ds_engine.module
    # Ensure model is on HPU
    if hasattr(model, "to"):
        model = model.to("hpu")

    # Prepare inputs
    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {k: v.to("hpu") for k, v in inputs.items()}

    # Optional warmup to stabilize kernels
    torch.set_grad_enabled(False)
    _ = model.generate(**inputs, max_new_tokens=5)
    htcore.mark_step()

    barrier()  # sync before timing
    if rank == 0:
        print("[Info] Starting timed generation...")

    # Timed generation
    start_time = time.time()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_scores=True
        )
        htcore.mark_step()
    end_time = time.time()

    # Only rank 0 decodes/prints (others have identical sequences)
    if rank == 0:
        decoded = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)
        print("\nGenerated text:\n", decoded)

        # Benchmark metrics (approximate per-token timing via linear split)
        total_time = end_time - start_time
        num_gen_tokens = generated.sequences.shape[1] - inputs["input_ids"].shape[1]
        scores = generated.scores if hasattr(generated, "scores") and generated.scores is not None else []
        # If scores list length != num_gen_tokens (can happen with some configs), fall back to num_gen_tokens
        steps = len(scores) if len(scores) > 0 else max(1, num_gen_tokens)
        token_times = np.diff(np.linspace(start_time, end_time, steps + 1))

        ttft = token_times[0] if len(token_times) > 0 else total_time
        inter_latencies = token_times[1:] if len(token_times) > 1 else np.array([total_time])

        print("\n=== Benchmark Metrics (rank 0) ===")
        print(f"World size: {world_size}")
        print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")
        print(f"Generated tokens: {num_gen_tokens}")
        print(f"Total time: {total_time:.4f} sec")
        print(f"Time to First Token (TTFT): {ttft:.4f} sec")
        print(f"Inter-Token Latency (Mean): {np.mean(inter_latencies):.4f} sec")
        print(f"Inter-Token Latency (Median): {np.median(inter_latencies):.4f} sec")
        print(f"Throughput: {num_gen_tokens / total_time:.2f} tokens/sec")

    barrier()  # ensure clean exit across ranks

if __name__ == "__main__":
    main()

