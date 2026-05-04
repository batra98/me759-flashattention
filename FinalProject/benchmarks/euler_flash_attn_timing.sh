#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --job-name=fa759_timing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
# Must be <= instruction partition max wall time (1:00:00 is rejected as PartitionTimeLimit).
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Slurm batch script for UW CSL Euler-style clusters (ME 759 / repo759 patterns).
# One GPU node: CMake + make + timing sweep only (no NCU / no sudo).
#
# Usage (from FinalProject/ on the cluster):
#   export FLASHATTN_CUDA_ARCH=80    # example: A100; omit or use 75 for sm_75
#   sbatch benchmarks/euler_flash_attn_timing.sh
#
# Logs: slurm-%j.out / slurm-%j.err in the directory where you ran sbatch.

set -eu
setopt PIPE_FAIL

# --- Modules: uncomment and edit to match your Euler stack (see repo759 HW09 CUDA jobs). ---
# module purge
# module load nvidia/cuda/12.2.0
# module load cmake/3.28.3
# module load gnu13/13.2.0

# Optional: limit kernels for a quick smoke test, e.g. export MODES="naive flash"
# export MODES="naive naive_causal flash flash_causal flash_v2 flash_wmma flash_wmma_db"

# FinalProject/ (parent of benchmarks/)
ROOT=${0:A:h:h}
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-${ROOT}/data/euler_runs}"
RUN_DIR="${OUT_ROOT}/${SLURM_JOB_ID:-manual}"
mkdir -p "${RUN_DIR}"

{
  echo "timestamp=$(date -Iseconds 2>/dev/null || date)"
  echo "hostname=$(hostname)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
  echo "FLASHATTN_CUDA_ARCH=${FLASHATTN_CUDA_ARCH:-75}"
  echo "MODES=${MODES:-<default all modes>}"
} | tee "${RUN_DIR}/job_info.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "${RUN_DIR}/nvidia_smi.txt"
else
  echo "nvidia-smi not in PATH" | tee "${RUN_DIR}/nvidia_smi.txt"
fi

export FLASHATTN_CUDA_ARCH="${FLASHATTN_CUDA_ARCH:-75}"

BUILD_DIR="${RUN_DIR}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLASHATTN_CUDA_ARCH="${FLASHATTN_CUDA_ARCH}"

make -j"${SLURM_CPUS_PER_TASK:-4}"

echo "=== correctness smoke (flash vs naive ref) ==="
./flash_attn --mode correctness --target flash --seq_len 1024

echo "=== timing sweep (writes ${RUN_DIR}/timing.csv) ==="
bash "${ROOT}/benchmarks/run_bench.sh" "${BUILD_DIR}/flash_attn" "${RUN_DIR}"

echo "Done. CSV: ${RUN_DIR}/timing.csv"
ls -la "${RUN_DIR}"
