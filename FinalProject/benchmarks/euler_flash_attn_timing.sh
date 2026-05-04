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
# Slurm batch script for UW CSL Euler (ME 759 / repo759-style Slurm headers).
# One GPU node: CMake + make + timing sweep only (no NCU / no sudo).
#
# Usage (from FinalProject/ on the cluster):
#   export FLASHATTN_CUDA_ARCH=80    # example: A100; use 75 for T4 / Turing
#   sbatch benchmarks/euler_flash_attn_timing.sh
#
# Optional overrides (see module spider on Euler):
#   export EULER_MODULE_GNU=gnu15/15.1.0
#   export EULER_MODULE_CUDA=nvidia/cuda/12.2.0
#   export EULER_MODULE_CMAKE=cmake/4.1.2
#   export EULER_SKIP_MODULES=1    # you already loaded nvcc, g++, cmake on PATH
#
# Logs: slurm-%j.out / slurm-%j.err in the directory where you ran sbatch.

set -eu
setopt PIPE_FAIL

# FinalProject root: under sbatch, $0 points at a copy under /var/spool/slurmd — use submit dir.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SUBMIT=${SLURM_SUBMIT_DIR:A}
  if [[ -f "${SUBMIT}/CMakeLists.txt" ]]; then
    ROOT=$SUBMIT
  elif [[ -f "${SUBMIT}/../CMakeLists.txt" ]]; then
    ROOT=${SUBMIT:h}
  elif [[ -f "${SUBMIT}/FinalProject/CMakeLists.txt" ]]; then
    ROOT="${SUBMIT}/FinalProject"
  else
    print -u2 "euler_flash_attn_timing.sh: cannot find CMakeLists.txt. Run sbatch from FinalProject/, from FinalProject/benchmarks/, or from the repo root (with FinalProject/)."
    exit 1
  fi
else
  ROOT=${0:A:h:h}
fi
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
fizsh

# --- Toolchain: defaults match `module spider` on engr Euler (gnu15, nvidia/cuda, cmake/4.1.x). ---
if [[ -z "${EULER_SKIP_MODULES:-}" ]]; then
  module purge 2>/dev/null || true
  if [[ -n "${EULER_MODULE_GNU:-}" ]]; then
    module load "$EULER_MODULE_GNU"
  else
    module load gnu15/15.1.0 || module load gnu13/13.2.0
  fi
  if [[ -n "${EULER_MODULE_CUDA:-}" ]]; then
    module load "$EULER_MODULE_CUDA"
  else
    module load nvidia/cuda/12.2.0 || module load nvidia/cuda/11.8.0 || module load nvidia/cuda/12.5.0
  fi
  if [[ -n "${EULER_MODULE_CMAKE:-}" ]]; then
    module load "$EULER_MODULE_CMAKE"
  else
    module load cmake/4.1.2 || module load cmake/4.1.0
  fi
  print "\n=== module list ===" >>"${RUN_DIR}/job_info.txt"
  module list 2>&1 >>"${RUN_DIR}/job_info.txt" || true
fi

if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
  export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-$CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11; do
    [[ -x "$d/bin/nvcc" ]] || continue
    export CUDA_HOME=$d CUDAToolkit_ROOT=$d
    export PATH="$d/bin:${PATH}"
    break
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  print -u2 "nvcc not found after modules. Try: module spider nvidia/cuda && module spider cmake"
  print -u2 "Set EULER_MODULE_CUDA / EULER_MODULE_CMAKE or EULER_SKIP_MODULES=1."
  exit 1
fi
export CUDACXX="${CUDACXX:-$(command -v nvcc)}"
if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
  print -u2 "No g++/c++ in PATH after modules. Load gnu15 (see script)."
  exit 1
fi

# Optional: limit kernels for a quick smoke test, e.g. export MODES="naive flash"
# export MODES="naive naive_causal flash flash_causal flash_v2 flash_wmma flash_wmma_db"

export FLASHATTN_CUDA_ARCH="${FLASHATTN_CUDA_ARCH:-75}"

BUILD_DIR="${RUN_DIR}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

NVCC=$(command -v nvcc)
CXXBIN=$(command -v g++ 2>/dev/null || command -v c++)
cmake "${ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLASHATTN_CUDA_ARCH="${FLASHATTN_CUDA_ARCH}" \
  -DCMAKE_CUDA_COMPILER="${NVCC}" \
  -DCMAKE_CXX_COMPILER="${CXXBIN}"

make -j"${SLURM_CPUS_PER_TASK:-4}"

echo "=== correctness smoke (flash vs naive ref) ==="
./flash_attn --mode correctness --target flash --seq_len 1024

echo "=== timing sweep (writes ${RUN_DIR}/timing.csv) ==="
bash "${ROOT}/benchmarks/run_bench.sh" "${BUILD_DIR}/flash_attn" "${RUN_DIR}"

echo "Done. CSV: ${RUN_DIR}/timing.csv"
ls -la "${RUN_DIR}"
