## FlashAttention Benchmark
This repository presents a controlled systems benchmark comparing three attention implementations—Naive Attention, PyTorch Scaled Dot-Product Attention (SDPA), and FlashAttention—with respect to runtime, GPU memory usage, and throughput across varying sequence lengths.

The goal is not to showcase peak numbers, but to understand scaling behavior and identify the regimes in which FlashAttention provides meaningful benefits.

### Motivation
Attention is the dominant bottleneck in modern Transformer models, especially at long sequence lengths. While FlashAttention is widely adopted, its advantages are often described qualitatively.

This project asks a more precise question:

At what sequence lengths and under what conditions does FlashAttention outperform standard attention implementations—and why?

### What This Repository Contains
- Three attention implementations
  - Naive Attention (explicit QK.T materialization)
  - PyTorch SDPA (fused kernel baseline)
  - FlashAttention v2 (memory-efficient tiled attention)
- A minimal Transformer block
  - QKV projections
  - Swappable attention backend
  - Residual connection and LayerNorm
  - No MLP, no dropout (attention-dominated by design)
- Benchmarking utilities
  - CUDA-synchronized runtime measurement
  - Peak GPU memory tracking
  - Token throughput calculation
- Controlled experiments
  - Sequence length sweeps
  - Fixed batch size, embedding dimension, and head count
  - Fair dtype handling across backends
- Analysis
  - Runtime vs sequence length
  - Memory usage vs sequence length
  - Throughput vs sequence length
 
### Experimental Setup
- Model: Single Transformer block
- Batch size: Fixed
- Embedding dimension / heads: Fixed
- Variable: Sequence length
- Precision
  - Naive / SDPA: FP32
  - FlashAttention: FP16
- Device: Single NVIDIA GPU
- Inputs: Random hidden-state tensors (post-embedding activations)
Random inputs are used intentionally, as attention kernel performance depends on tensor shape, dtype, and memory access patterns—not token semantics.

### Key Results
- Naive attention
  - Quadratic memory growth (O(T^2))
  - Becomes impractical at moderate sequence lengths
- SDPA
  - Substantially better than naive attention
  - Still exhibits increasing memory pressure as sequence length grows
- FlashAttention
  - Near-linear memory scaling
  - Superior runtime and throughput at medium to large sequence lengths
  - Slight overhead at very small sequence lengths due to kernel setup costs
Overall, FlashAttention’s primary advantage is memory efficiency, which in turn enables better runtime scaling and sustained throughput for long-context workloads.

### Notes on Reproducibility
- Raw CSV outputs are hardware-dependent and intentionally not version-controlled.
- Plots are included for reference.
- Each experiment uses a fresh model instance to avoid CUDA memory contamination.
- CUDA synchronization and warm-up runs are used to ensure accurate timing.

### Takeaway
FlashAttention does not merely provide constant-factor speedups.
It fundamentally changes the memory scaling behavior of attention, making long-context Transformers practical on modern hardware.

















