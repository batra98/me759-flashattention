# me759-flashattention

[![GitHub Pages](https://img.shields.io/badge/Interactive%20Report-Live-818cf8?style=for-the-badge&logo=github)](https://batra98.github.io/me759-flashattention)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A bare-metal **CUDA C++** implementation of the FlashAttention forward pass on an NVIDIA Tesla T4 (sm_75), featuring:

- **Online Softmax** — incremental max/sum rescaling, no N×N scratch space
- **Shared-memory tiling** — Q tile cached across K/V loop; BR = BC = 32, D = 64  
- **O(N) HBM traffic** vs O(N²) for naive attention
- **Interactive report** at [batra98.github.io/me759-flashattention](https://batra98.github.io/me759-flashattention)

## Build (on GCP/Linux)

```bash
git clone https://github.com/batra98/me759-flashattention.git
cd me759-flashattention
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

```bash
# Correctness check (RMSE naive vs flash, should be < 1e-3)
./flash_attn --mode correctness --seq_len 1024

# Timing benchmark
./flash_attn --mode naive  --seq_len 4096 --warmup 5 --iters 20
./flash_attn --mode flash  --seq_len 4096 --warmup 5 --iters 20

# Full sweep → CSV
chmod +x ../benchmarks/run_bench.sh
../benchmarks/run_bench.sh ./flash_attn ../benchmarks/results/
```

## NCU Profiling (HBM traffic)

```bash
sudo ncu --metrics \
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
  ./flash_attn --mode naive --seq_len 4096 --warmup 0 --iters 1
```

## Results (Tesla T4, d=64)

| N    | Naive (ms) | Flash (ms) | Speedup |
|------|-----------|-----------|---------|
| 512  | 2.14      | 1.37      | 1.6×    |
| 1024 | 3.70      | 1.69      | 2.2×    |
| 2048 | 8.66      | 3.75      | 2.3×    |
| 4096 | 27.12     | 14.96     | 1.8×    |
| 8192 | 112.31    | 58.70     | 1.9×    |

## Algorithm Reference

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022).  
*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*  
NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

## Course

ME 759 — High-Performance Computing for Engineering Applications  
University of Wisconsin–Madison, Spring 2026
