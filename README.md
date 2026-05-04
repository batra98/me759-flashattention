# FlashAttention in Raw CUDA (ME/CS/ECE 759)

[![GitHub Pages](https://img.shields.io/badge/Interactive%20Report-Live-818cf8?style=for-the-badge&logo=github)](https://batra98.github.io/me759-flashattention)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

🎥 **Video Walkthrough:** A complete video walkthrough is available [on YouTube](https://youtu.be/a39IGxQ6rKs).
 
CUDA implementation of a FlashAttention-style forward pass (Dao et al., 2022) with naive, tiled, causal, warp-centric, and WMMA variants. Reported timings and NCU data were taken on a **Google Cloud Platform** VM with a **Tesla T4** (not the department Euler cluster: toolchain issues, non-T4 GPUs, and **Nsight Compute** / `ncu` needing **`sudo`**, which Euler does not provide—see **`FinalProject/README.md`** and the PDF).

**Graders and reproducibility:** All course code, benchmarks, and build or run instructions are under **`FinalProject/`**. Start with **[`FinalProject/README.md`](FinalProject/README.md)**. The LaTeX source for the final report is **`final_report.tex`** at this repository root.
