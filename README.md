# Benchmarking Large Language Models on Intel Habana Gaudi AI Accelerators

This repository contains all benchmarking scripts, analysis notebooks, logs, and documentation for:

**Benchmarking Large Language Models on Intel Habana Gaudi AI Accelerators**

The project provides a vendor-neutral, reproducible evaluation of LLM inference across Gaudi2 and Gaudi3 accelerator generations, covering latency, throughput, energy efficiency, and scaling behavior for both text-only and multi-modal (vision-language) models.

---

## 📌 Objectives

- Establish unified baselines for **TTFT, ITL, throughput, J/token, and EDP**.
- Compare performance of:
  - **Llama2-7B**
  - **Llama3-8B**
  - **Llama3.1-70B**
  - **Qwen2.5-VL-7B-Instruct**
- Evaluate across **three inference frameworks**:
  - vLLM (Habana fork)
  - DeepSpeed Inference
  - Optimum Habana
- Analyze scaling:
  - Single-card  
  - Multi-card  
  - Tensor Parallelism (TP=1/2/4/8/etc.)
- Characterize **multi-modal inference overhead** for document understanding workloads.

---

## 🧠 Models Benchmarked

| Model | Params | Type | Usage |
|-------|--------|------|-------|
| **Llama2-7B** | 7B | Text-only | Baseline |
| **Llama3-8B** | 8B | Text-only | Latency–energy sweet spot |
| **Llama3.1-70B** | 70B | Text-only | Large-model scaling |
| **Qwen2.5-VL-7B-Instruct** | 7B | Vision-Language | Document + image inputs |

---

## 🖥 Hardware Setup (Vendor-Neutral Abstraction)

| Config | Description |
|--------|-------------|
| `gaudi_2` | 7nm, 96GB HBM, 2.5 TB/s BW |
| `gaudi_3` | 5nm, 128GB HBM, 3.7 TB/s BW, native FP8 |
| `gaudi_3s` / `gaudi_3m` / `gaudi_3l` | Scaled small / medium / large multi-card setups |

---

## 📦 Software Stack

- **Habana SynapseAI**
- **vLLM (Habana fork)** with continuous batching + KV Cache Paging  
- **DeepSpeed** for tensor & pipeline parallelism  
- **Optimum Habana** for compiler/runtime fused inference  
- **Power logging** using `hl-smi` at 1 Hz  
- **Python 3.10+**  

---

## 📊 Metrics Recorded

### Latency
- **TTFT** — Time To First Token  
- **ITL** — Inter Token Latency  
- **Throughput** — Tokens/sec  

### Energy
- **Power traces** (1 Hz sampling)
- **Energy per token** = W / tok/s  
- **Energy-Delay Product (EDP)**

### Multi-modal
- Image encoding latency  
- Fusion overhead  
- Batch scaling effects  

---

## 🚀 Running Benchmarks

### 1. Setup Environment

```bash
pip install transformers==4.35.2 optimum[habana] deepspeed
2. Install vLLM Habana Fork
bash
Copy code
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git checkout 18423b3ea006bc09083635a386d5fc10f581ddc9
pip install -r requirements-hpu.txt
python setup.py develop
3. Run Benchmarks (Example: vLLM)
bash
Copy code
python benchmark_vllm.py --model meta-llama/Llama-2-7b-hf --tp 1
4. Power Logging
bash
Copy code
hl-smi --power --interval 1 --log power_trace.csv
📈 Summary of Key Findings
8B models provide the best efficiency (3.84 J/token, 122 tok/s).

70B models incur ~3× higher energy/token due to KV-cache and memory bandwidth limits.

Gaudi3 shows pipelined TPC utilization, improving overlap and reducing idle gaps.

Vision-language models (Qwen2.5-VL):

4–5× higher TTFT

3–5× higher energy/token

Require minimum TP=4 for stable large-batch inference

Continuous batching (vLLM) maximizes throughput.

Distributed inference (DeepSpeed) achieves lowest TTFT for single queries.

📚 Repository Structure
bash
Copy code
/benchmarks
    benchmark_vllm.py
    benchmark_deepspeed.py
    benchmark_optimum.py

/scripts
    power_logger.sh
    parse_metrics.py

/logs
    *.csv  (TTFT, ITL, throughput)
    power_traces/

/analysis
    notebooks/
    plots/

/configs
    gaudi2/
    gaudi3/
📑 Thesis Document
The full thesis PDF is included in this repository:

Copy code
mtech_thesis_21_11_2025.pdf
📝 Citation
If you use this work, please cite:

nginx
Copy code
Prince Raj, "Benchmarking Large Language Models on Intel Habana Gaudi AI Accelerators," IIT Ropar, 2025.
🧭 Future Work
Add quantized (INT8/INT4) baselines

Speculative decoding benchmarks

Collective communication profiling

End-to-end agentic workload testing

📬 Contact
Prince Raj
MTech Artificial Intelligence
IIT Ropar
Email: (your email here)
