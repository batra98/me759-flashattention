#!/usr/bin/env python3
"""Read benchmark CSVs and print summary statistics."""
import sys

def read_csv(path):
    with open(path) as f:
        headers = f.readline().strip().split(',')
        rows = []
        for line in f:
            vals = line.strip().split(',')
            rows.append(dict(zip(headers, vals)))
    return rows

def main():
    timing = read_csv("benchmarks/results/timing.csv")
    naive = {r['seq_len']: float(r['ms']) for r in timing if r['mode'] == 'naive'}
    flash = {r['seq_len']: float(r['ms']) for r in timing if r['mode'] == 'flash'}

    print("=== Timing Summary ===")
    print(f"{'N':>6}  {'Naive (ms)':>10}  {'Flash (ms)':>10}  {'Speedup':>8}")
    for n in sorted(naive.keys(), key=int):
        sp = naive[n] / flash[n]
        print(f"{n:>6}  {naive[n]:>10.4f}  {flash[n]:>10.4f}  {sp:>7.1f}x")

if __name__ == "__main__":
    main()
