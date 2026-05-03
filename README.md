# Benchmarking Large Language Models on Intel Habana Gaudi AI Accelerators

This repository contains all benchmarking scripts, analysis notebooks, logs, and documentation for:

**Benchmarking Large Language Models on Intel Habana Gaudi AI Accelerators**

The project provides a vendor-neutral, reproducible evaluation of LLM inference across Gaudi2 and Gaudi3 accelerator generations, covering latency, throughput, energy efficiency, cost, and scaling behavior across diverse model architectures.

---

## 📌 Objectives

- Establish unified baselines for:
  - TTFT (Time to First Token)
  - ITL (Inter Token Latency)
  - Throughput (tokens/sec)
  - Energy per token (J/token)
  - Energy-Delay Product (EDP)
- Benchmark across:
  - Dense transformer models
  - Sparse Mixture-of-Experts (MoE) models
  - Vision-Language models
- Compare performance across Gaudi2 vs Gaudi3
- Evaluate single-card and multi-card scaling
- Build a data-driven deployment framework for LLM inference

---

## 🧠 Models Benchmarked

### Dense Models
- Mistral-7B
- Llama2-7B
- Llama3-8B
- Llama3.1-70B

### Mixture-of-Experts (MoE)
- Mixtral-8x7B
- Mixtral-8x22B

### Vision-Language
- Qwen2.5-VL-7B-Instruct

---

## 🖥 Hardware Setup

- Gaudi 2
- Gaudi 3
- Multi-card configurations (small, medium, large)

---

## 📦 Software Stack

- SynapseAI
- vLLM (Habana fork)
- DeepSpeed
- Optimum Habana
- hl-smi power logging

---

## 📊 Metrics

- TTFT
- ITL
- Throughput
- Energy per token
- EDP

---

## 🚀 Running Benchmarks

pip install transformers optimum[habana] deepspeed

python benchmark_vllm.py --model mistralai/Mistral-7B --tp 1

hl-smi --power --interval 1 --log power_trace.csv

---

## 📈 Key Findings

- Gaudi3 excels in throughput and latency
- Gaudi2 is more energy efficient
- MoE models behave differently from dense models
- Trade-offs exist across all metrics
- No single configuration is optimal

---

## 📬 Contact

Prince Raj  
MTech AI, IIT Ropar  
2024AIM1012@iitrpr.ac.in
