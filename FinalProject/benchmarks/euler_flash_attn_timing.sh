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
# UW CSL Euler: build + timing sweep (no NCU). From FinalProject/: sbatch benchmarks/euler_flash_attn_timing.sh
# Edit the three module lines or FLASHATTN_CUDA_ARCH below if your node stack differs.

set -eu
setopt PIPE_FAIL

# FinalProject root (sbatch copies script under /var/spool/slurmd — do not use $0 for ROOT).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SUBMIT=${SLURM_SUBMIT_DIR:A}
  if [[ -f "${SUBMIT}/CMakeLists.txt" ]]; then
    ROOT=$SUBMIT
  elif [[ -f "${SUBMIT}/../CMakeLists.txt" ]]; then
    ROOT=${SUBMIT:h}
  elif [[ -f "${SUBMIT}/FinalProject/CMakeLists.txt" ]]; then
    ROOT="${SUBMIT}/FinalProject"
  else
    print -u2 "Run sbatch from FinalProject/, FinalProject/benchmarks/, or repo root (with FinalProject/)."
    exit 1
  fi
else
  ROOT=${0:A:h:h}
fi
cd "$ROOT"

RUN_DIR="${ROOT}/data/euler_runs/${SLURM_JOB_ID:-manual}"
mkdir -p "${RUN_DIR}"

{
  echo "timestamp=$(date -Iseconds 2>/dev/null || date)"
  echo "hostname=$(hostname)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
} | tee "${RUN_DIR}/job_info.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "${RUN_DIR}/nvidia_smi.txt"
else
  echo "nvidia-smi not in PATH" | tee "${RUN_DIR}/nvidia_smi.txt"
fi

[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
[[ -f /usr/share/lmod/lmod/init/zsh ]] && ! whence module >/dev/null 2>&1 && source /usr/share/lmod/lmod/init/zsh

module purge 2>/dev/null || true
module load gnu15/15.1.0
module load nvidia/cuda/12.2.0
module load cmake/4.1.2

print "\n=== module list ===" >>"${RUN_DIR}/job_info.txt"
module list 2>&1 >>"${RUN_DIR}/job_info.txt" || true

if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
  export CUDAToolkit_ROOT="${CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

command -v nvcc >/dev/null 2>&1 || { print -u2 "nvcc not on PATH after module load."; exit 1 }
command -v g++ >/dev/null 2>&1 || { print -u2 "g++ not on PATH after module load."; exit 1 }

# sm_80 = A100 etc.; use 75 for T4 / Turing — must match the GPU you requested.
FLASHATTN_CUDA_ARCH=80

export CUDACXX="$(command -v nvcc)"

BUILD_DIR="${RUN_DIR}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLASHATTN_CUDA_ARCH="${FLASHATTN_CUDA_ARCH}" \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++)"

make -j"${SLURM_CPUS_PER_TASK:-4}"

echo "=== correctness smoke ==="
./flash_attn --mode correctness --target flash --seq_len 1024

echo "=== timing sweep → ${RUN_DIR}/timing.csv ==="
bash "${ROOT}/benchmarks/run_bench.sh" "${BUILD_DIR}/flash_attn" "${RUN_DIR}"

echo "Done. CSV: ${RUN_DIR}/timing.csv"
ls -la "${RUN_DIR}"
