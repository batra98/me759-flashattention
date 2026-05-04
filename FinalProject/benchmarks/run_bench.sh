#!/bin/bash
set -euo pipefail
BIN="${1:-./build/flash_attn}"
OUT="${2:-./data/results}"
# Override with env MODES="naive flash" for a subset
DEFAULT_MODES="naive naive_causal flash flash_causal flash_v2 flash_wmma flash_wmma_db"
MODES="${MODES:-$DEFAULT_MODES}"
mkdir -p "$OUT"
CSV="$OUT/timing.csv"
echo "mode,seq_len,d_head,ms" > "$CSV"
for N in 512 1024 2048 4096 8192; do
    for MODE in $MODES; do
        LINE=$("$BIN" --mode "$MODE" --seq_len "$N" --warmup 5 --iters 20 --csv)
        echo "$LINE" >> "$CSV"
        echo "$LINE"
    done
done
echo "Saved → $CSV"
