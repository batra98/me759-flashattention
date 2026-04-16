#!/bin/bash
# run_ncu_profile.sh — NCU HBM traffic profiling for all N values
# Usage: sudo bash run_ncu_profile.sh [binary] [output_csv]
set -euo pipefail

BIN="${1:-./flash_attn}"
OUT="${2:-hbm_traffic.csv}"
NCU="/usr/local/cuda/bin/ncu"

export HOME=/tmp
export PATH=/usr/local/cuda/bin:$PATH

echo "mode,seq_len,d_head,bytes_read_MB,bytes_write_MB" > "$OUT"

SEQ_LENS=(512 1024 2048 4096 8192)

echo "======================================="
echo " NCU HBM Traffic Profiling"
echo " $(date)"
echo "======================================="

for N in "${SEQ_LENS[@]}"; do
    for MODE in naive flash; do
        printf "N=%-5d mode=%-5s ... " "$N" "$MODE"

        # Capture NCU output
        RAW=$($NCU --csv --metrics \
            l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
            "$BIN" --mode "$MODE" --seq_len "$N" --d_head 64 --warmup 0 --iters 1 2>&1)

        # Sum reads and writes across all kernels (NCU CSV has one row per kernel)
        # Extract metric values — they're in bytes in CSV mode
        TOTAL_READ=$(echo "$RAW" | grep "l1tex__t_bytes_pipe_lsu_mem_global_op_ld" | \
            awk -F',' '{sum += $NF} END {printf "%.2f", sum/1048576}')
        TOTAL_WRITE=$(echo "$RAW" | grep "l1tex__t_bytes_pipe_lsu_mem_global_op_st" | \
            awk -F',' '{sum += $NF} END {printf "%.2f", sum/1048576}')

        echo "$MODE,$N,64,$TOTAL_READ,$TOTAL_WRITE" >> "$OUT"
        printf "read=%s MB  write=%s MB\n" "$TOTAL_READ" "$TOTAL_WRITE"
    done
done

echo ""
echo "Saved → $OUT"
cat "$OUT"
