# FlashAttention in Raw CUDA
**ME 759 Final Project — High-Performance Computing**

[![GitHub Pages](https://img.shields.io/badge/Interactive%20Report-Live-818cf8?style=for-the-badge&logo=github)](https://batra98.github.io/me759-flashattention)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A bare-metal **CUDA C++** implementation of the FlashAttention forward pass (Dao et al. 2022) on an NVIDIA Tesla T4. This project demonstrates how transforming attention from a memory-bound algorithm to an IO-aware algorithm reduces global memory traffic from $O(N^2)$ to $O(N)$.

---

## 1. Introduction and Motivation

Transformers dominate modern AI, but self-attention has quadratic computing complexity $O(N^2)$. The standard mathematical formulation requires materializing the full $N \times N$ attention matrix $S = QK^T / \sqrt{d}$ in Global Memory (HBM). For a sequence length of $N=4096$, the reads and writes of this $S$ matrix create severe memory bottlenecks preventing optimal GPU utilization.

**The Solution:** This project implements FlashAttention completely from scratch in raw CUDA C++. By heavily utilizing Shared Memory (SRAM) caching and mathematically reformulating softmax into a single-pass "online softmax", we keep the entire $N \times N$ matrix strictly inside the multiprocessor registers, reducing the core bottleneck (HBM traffic) to $O(N)$.

---

## 2. Methodology & Algorithm Design

### The Baseline (Naive) Implementation
To form a solid experimental baseline, `naive_attention.cu` was implemented natively:
1. **`qkt_kernel`**: Computes $S = QK^T / \sqrt{d}$ and saves the full $N \times N$ matrix to HBM.
2. **`softmax_kernel`**: Loads rows from HBM, finds a moving max for numerical stability, exponentiates, and normalizes (requiring multiple trips to HBM).
3. **`pv_kernel`**: Loads $P$ and $V$ from HBM, computing and writing output $O$.

*The Flaw:* This approach scales quadratically in memory capacity but exponentially worse in traffic due to **uncoalesced global reads**. Strided memory block fetching causes immense 32-way cache-line thrashing on typical GPU architectures.

### FlashAttention (Tiled, Shared Memory) Implementation
`flash_attention_v1.cu` solves this by restricting computation inside on-chip boundaries:
- **Hardware constraints:** The Tesla T4 features 64 KB of addressable shared memory per SM. We restrict our tile blocks to $B_r = 32$, $B_c = 32$, $d = 64$, ensuring matrix subsets fit inside a tight 24 KB memory footprint per block.
- **Tiling Mechanics:** The outer loop processes tiles of $Q$. The inner loop streams tiles of $K$ and $V$ entirely sequentially, generating perfectly coalesced `float4` loads.
- **Online Softmax:** The maximum $m_i$ and normalizer $l_i$ states are recursively tracked in the thread registers. 
By multiplying old accumulations by an attenuation factor (scaling equations), we never need to access the past $S$ matrix scores to rebuild the correct probabilities! Intermediate calculations are never written out to the GPU's memory bus.

---

## 3. Experimental Setup and Build

**Hardware Environment:** 
- **GPU:** NVIDIA Tesla T4 (sm_75, Turing Architecture)
- **Power:** 8.1 TFLOPs FP32 throughput, 320 GB/s HBM Bandwidth, 64 KB SRAM/SM.
- **Compiler:** `nvcc` via CMake with `-O3 --extended-lambda --expt-relaxed-constexpr -lineinfo`

**Build Instructions (Linux):**
```bash
git clone https://github.com/batra98/me759-flashattention.git
cd me759-flashattention
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Usage & Testing:**
```bash
# Validity check against the naive baseline (Requires RMSE < 1e-3)
./flash_attn --mode correctness --seq_len 1024

# Interactive Latency Benchmark
./flash_attn --mode flash --seq_len 4096 --warmup 5 --iters 20
```

---

## 4. Results & Benchmarks

The core argument of this project relies on hardware profiling extracted directly from the GPU performance counters using **Nsight Compute (NCU)**. We monitored L1TEX byte metrics against global operational loads and stores to view the exact algorithmic hardware traffic.

### Wall-Clock Latency (Speedup)
| N    | Naive (ms) | Flash (ms) | Speedup |
|------|-----------|-----------|---------|
| 512  | 2.14      | 1.37      | 1.6×    |
| 1024 | 3.70      | 1.69      | 2.2×    |
| 2048 | 8.66      | 3.75      | 2.3×    |
| **4096** | **27.12** | **14.96** | **1.8×**|
| 8192 | 112.31    | 58.70     | 1.9×    |

Because the algorithm does not bypass the raw $O(N^2)$ FLOP requirement, speedup is naturally clipped around 2× vs standard implementation; memory latency is removed but compute latency remains.

### HBM Cache Evictions (The Crown Jewel)
At $N=4096$, the Nsight Compute hardware trackers reported the following raw data streams:
- **Naive Traffic:** ~25.8 GB global reads, ~1.16 GB global writes.
- **Flash Traffic:** ~2.21 GB global reads, ~8.39 MB global writes.

This illustrates the IO awareness: **FlashAttention triggers an $\approx 11.7\times$ reduction in Global HBM Reads and an astonishing $\approx 138\times$ reduction in Global HBM Writes**, totally eliminating the memory wall bottlenecks that cap standard attention heads.

---

## 5. Conclusions & Future Extensibility

This project built a verifiable `libflash` parallelized logic directly in CUDA. For future hardware extensions, standard floating point matrices could be substituted with Tensor Cores leveraging NVIDIA `wmma` API layouts, effectively removing the raw FP32 compute bounds and generating exponential multiplier walls into the timing speedups. 

---

## Algorithm Reference
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022).  
*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*  
NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
