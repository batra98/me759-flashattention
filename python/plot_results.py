#!/usr/bin/env python3
"""Read benchmark CSVs and generate PNG plots for the final PDF report."""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        headers = f.readline().strip().split(',')
        rows = []
        for line in f:
            if not line.strip(): continue
            vals = line.strip().split(',')
            rows.append(dict(zip(headers, vals)))
    return rows

def main():
    timing = read_csv("benchmarks/results/timing.csv")
    hbm = read_csv("benchmarks/results/hbm_traffic.csv")

    if not timing:
        print("Error: Could not find benchmarks/results/timing.csv")
        return

    os.makedirs("benchmarks/results/plots", exist_ok=True)

    # ---------------------------------------------------------
    # 1. Plot Timing (Latency)
    # ---------------------------------------------------------
    seq_lens = sorted(list(set(int(r['seq_len']) for r in timing)))
    
    naive_ms = [float(next(r['ms'] for r in timing if r['mode'] == 'naive' and int(r['seq_len']) == N)) for N in seq_lens]
    flash_ms = [float(next(r['ms'] for r in timing if r['mode'] == 'flash' and int(r['seq_len']) == N)) for N in seq_lens]

    plt.figure(figsize=(8, 5))
    plt.plot(seq_lens, naive_ms, marker='o', label='Naive Attention', color='#f43f5e', linewidth=2)
    plt.plot(seq_lens, flash_ms, marker='o', label='FlashAttention', color='#38bdf8', linewidth=2)
    
    plt.title("Attention Latency vs Sequence Length (Tesla T4)")
    plt.xlabel("Sequence Length (N)")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(seq_lens)
    
    plt.savefig("benchmarks/results/plots/latency_plot.png", dpi=300, bbox_inches='tight')
    print("Saved -> benchmarks/results/plots/latency_plot.png")
    plt.close()

    # ---------------------------------------------------------
    # 2. Plot HBM Traffic (Reads + Writes)
    # ---------------------------------------------------------
    if not hbm:
        print("Note: hbm_traffic.csv not found, skipping HBM plots.")
        return

    naive_hbm = []
    flash_hbm = []
    
    for N in seq_lens:
        nv = next((r for r in hbm if r['mode'] == 'naive' and int(r['seq_len']) == N), None)
        fl = next((r for r in hbm if r['mode'] == 'flash' and int(r['seq_len']) == N), None)
        
        if nv: naive_hbm.append(float(nv['bytes_read_MB']) + float(nv['bytes_write_MB']))
        else: naive_hbm.append(0)
            
        if fl: flash_hbm.append(float(fl['bytes_read_MB']) + float(fl['bytes_write_MB']))
        else: flash_hbm.append(0)

    plt.figure(figsize=(8, 5))
    
    x = np.arange(len(seq_lens))
    width = 0.35
    
    plt.bar(x - width/2, naive_hbm, width, label='Naive HBM', color='#f43f5e')
    plt.bar(x + width/2, flash_hbm, width, label='Flash HBM', color='#38bdf8')
    
    plt.title("Total HBM Traffic vs Sequence Length (Tesla T4)")
    plt.xlabel("Sequence Length (N)")
    plt.ylabel("Total HBM Traffic (MB) - Log Scale")
    plt.yscale('log')
    plt.xticks(x, seq_lens)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig("benchmarks/results/plots/hbm_plot.png", dpi=300, bbox_inches='tight')
    print("Saved -> benchmarks/results/plots/hbm_plot.png")
    plt.close()

if __name__ == "__main__":
    main()
