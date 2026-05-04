# ME/CS/ECE 759 Final Project: FlashAttention in Raw CUDA

This folder is the **course submission root** (name must be exactly `FinalProject/`). It contains a from-scratch CUDA forward pass for single-head attention: a three-kernel naive baseline, tiled FlashAttention-style kernels with online softmax, a causal variant that skips future tiles, a warp-centric FP32 experiment (v2), and FP16 WMMA score paths (including a double-buffer entry point that matches WMMA on Turing).

**Interactive write-up:** [GitHub Pages](https://batra98.github.io/me759-flashattention) (built from `docs/`). **Full narrative, citations, and methodology:** `../final_report.tex` at the repository root.

---

## Current state

| Area | Status |
|------|--------|
| **Hardware target** | NVIDIA Tesla T4 (`sm_75`); CMake builds for architecture 75. |
| **Correctness** | Each mode is checked against the naive FP32 reference (`--mode correctness`); WMMA modes use a slightly looser RMSE tolerance because of FP16 score accumulation. |
| **Latency sweep** | Checked-in `data/results/timing.csv`: N ∈ {512, 1024, 2048, 4096, 8192}, d = 64, 5 warmup + 20 timed iterations per point. |
| **Profiling** | Checked-in `data/results/hbm_traffic.csv`: Nsight Compute L1TEX global byte counters via `benchmarks/run_ncu_profile.sh`. |
| **Figures** | `data/results/plots/` and `docs/assets/` PNGs can be regenerated with `python/plot_results.py` from the CSVs above. |
| **v2 (warp-centric)** | Correct but **slower than naive at large N** in this repo: smaller Q tiles double how often K/V stream, which shows up clearly in `hbm_traffic.csv` and the write-up. |
| **WMMA** | Lowest read traffic among non-causal variants here; **causal skipping is not fused** into the WMMA kernel yet, so `flash_causal` still wins wall-clock at large N. |
| **`flash_wmma_db`** | Same kernel as WMMA on T4; a real async pipeline would need newer hardware. |

Everything below is reproducible from a clean clone using the commands in **Build** through **Plots**.

---

## Results snapshot

All numbers below come from the **committed** `data/results/timing.csv` and `hbm_traffic.csv` (Tesla T4, FP32 I/O except WMMA score tiles, head dimension **d = 64**).

### Mean latency (ms)

| Mode | N = 4096 | N = 8192 |
|------|----------|----------|
| `naive` | 27.12 | 113.55 |
| `naive_causal` | 17.92 | 73.66 |
| `flash` | 15.05 | 58.68 |
| `flash_causal` | 9.52 | **30.65** |
| `flash_v2` | 66.33 | 252.88 |
| `flash_wmma` | 15.79 | 50.19 |
| `flash_wmma_db` | 15.79 | 50.18 |

**How to read this:** Tiled `flash` is about **1.9× faster** than `naive` at N = 8192 (59 ms vs 114 ms). **Causal** Flash wins overall because it does less work and loads roughly half the K/V data of the dense kernel. WMMA beats dense `flash` at N = 8192 (about 50 ms vs 59 ms) but does not beat `flash_causal`. **v2** is a deliberate counterexample: softmax fusion does not help if smaller Q tiles force extra passes over K/V (visible in both timers and NCU).

### HBM traffic (NCU, snapshot at N = 4096)

Global **read** volume drops from about **25.8 GB** (`naive`) to **2.2 GB** (`flash`) and **1.1 GB** (`flash_causal`); WMMA is near **0.54 GB** read at the same N. Tiled kernels cluster near **8 MB** **writes** (output only), versus about **1.16 GB** for naive (materialized score matrix). The committed `hbm_traffic.csv` and the roofline-style figure from `plot_results.py` contain the full sweep.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/` | CUDA sources; `main.cu` CLI and dispatch |
| `build/` | Out-of-tree build output (create locally; ignored by git) |
| `data/results/` | `timing.csv`, `hbm_traffic.csv`, `plots/*.png` |
| `benchmarks/` | `run_bench.sh`, `run_ncu_profile.sh` |
| `python/` | `plot_results.py`, `reference.py` |
| `docs/` | GitHub Pages site and `docs/assets/` figures |

**Requirements:** Linux, NVIDIA GPU with CUDA (tested on T4), CMake 3.18+, CUDA toolkit with `nvcc`.

---

## Build

```bash
git clone https://github.com/batra98/me759-flashattention.git
cd me759-flashattention/FinalProject
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
```

The binary is `build/flash_attn`.

---

## Correctness

```bash
./flash_attn --mode correctness --target flash --seq_len 2048
```

---

## Benchmarks (timing CSV)

From `FinalProject/build/`:

```bash
bash ../benchmarks/run_bench.sh ./flash_attn ../data/results
```

Writes `../data/results/timing.csv`. Limit modes, for example:

```bash
MODES="naive flash flash_causal" bash ../benchmarks/run_bench.sh ./flash_attn ../data/results
```

---

## Nsight Compute (HBM CSV)

Requires `ncu` on `PATH` (often `/usr/local/cuda/bin`). From `FinalProject/build/`:

```bash
sudo bash ../benchmarks/run_ncu_profile.sh ./flash_attn ../data/results/hbm_traffic.csv
```

---

## Plots

Python 3 with **matplotlib** (and a writable `MPLCONFIGDIR` if your home cache is not writable):

```bash
cd /path/to/me759-flashattention/FinalProject
python3 python/plot_results.py
```

Refreshes `data/results/plots/*.png` and `docs/assets/*.png`.

---

## Euler (Slurm, timing only)

On department clusters you typically **do not have sudo**, so **Nsight Compute profiling is skipped** here. The batch script follows the same Slurm style as course homework in `repo759` (`#SBATCH -p instruction`, `--gres=gpu:1`, optional `module load` lines you can uncomment).

1. Clone the repo on Euler and `cd` into **`FinalProject/`**.
2. Set **`FLASHATTN_CUDA_ARCH`** to match the GPU (for example **`80`** on A100, **`75`** on T4/Turing). The default in CMake is **75**.
3. Uncomment and fix **`module load`** lines in `benchmarks/euler_flash_attn_timing.sh` if `cmake` / `nvcc` are not already on your `PATH`.
4. Submit:

```bash
export FLASHATTN_CUDA_ARCH=80   # example; use 75 on sm_75 GPUs
sbatch benchmarks/euler_flash_attn_timing.sh
```

The script requests **`#SBATCH --time=00:30:00`** so it stays under the `instruction` partition wall-clock cap (longer jobs show Slurm reason **PartitionTimeLimit** and never start).

Each job writes **`data/euler_runs/<JOBID>/`**: `job_info.txt`, `nvidia_smi.txt`, a fresh **`build/`**, **`timing.csv`**, and Slurm’s `slurm-<JOBID>.out` / `.err` in the directory from which you ran `sbatch`. That tree is **gitignored** so you can `scp` results back without fighting Git.

For a shorter test, set **`MODES`** before `sbatch` (same as `run_bench.sh`), for example: `export MODES="naive flash"`.
