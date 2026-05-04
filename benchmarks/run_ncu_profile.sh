#!/bin/bash
# run_ncu_profile.sh — NCU HBM traffic profiling for all N values and modes
# Usage: sudo bash run_ncu_profile.sh [binary] [output_csv]
set -euo pipefail

BIN="${1:-./flash_attn}"
OUT="${2:-hbm_traffic.csv}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
TMPF="/tmp/ncu_out.txt"

DEFAULT_MODES="naive naive_causal flash flash_causal flash_v2 flash_wmma flash_wmma_db"
MODES="${MODES:-$DEFAULT_MODES}"

export HOME=/tmp
export PATH=/usr/local/cuda/bin:$PATH

echo "mode,seq_len,d_head,bytes_read_MB,bytes_write_MB" > "$OUT"

SEQ_LENS=(512 1024 2048 4096 8192)

echo "======================================="
echo " NCU HBM Traffic Profiling"
echo " $(date)"
echo " MODES=$MODES"
echo "======================================="

for N in "${SEQ_LENS[@]}"; do
    for MODE in $MODES; do
        printf "N=%-5d mode=%-14s ... " "$N" "$MODE"

        $NCU --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
            "$BIN" --mode "$MODE" --seq_len "$N" --d_head 64 --warmup 0 --iters 1 \
            > "$TMPF" 2>&1

        TOTAL_READ=$(grep "op_ld.sum" "$TMPF" | awk '{
            val=$NF; unit=$(NF-1);
            if(unit=="Gbyte") val=val*1024;
            if(unit=="Kbyte") val=val/1024;
            if(unit=="byte") val=val/1048576;
            sum+=val} END{printf "%.2f",sum}')

        TOTAL_WRITE=$(grep "op_st.sum" "$TMPF" | awk '{
            val=$NF; unit=$(NF-1);
            if(unit=="Gbyte") val=val*1024;
            if(unit=="Kbyte") val=val/1024;
            if(unit=="byte") val=val/1048576;
            sum+=val} END{printf "%.2f",sum}')

        echo "$MODE,$N,64,$TOTAL_READ,$TOTAL_WRITE" >> "$OUT"
        printf "read=%s MB  write=%s MB\n" "$TOTAL_READ" "$TOTAL_WRITE"
    done
done

rm -f "$TMPF"
echo ""
echo "Saved → $OUT"
cat "$OUT"
