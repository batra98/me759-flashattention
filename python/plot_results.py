#!/usr/bin/env python3
"""Read benchmark CSVs and generate PNG plots for the ME759 final report."""
import os
import sys
import matplotlib.pyplot as plt

D_HEAD = 64
# Tesla T4 peak FP32 ~ 8.1 TFLOPS; memory bandwidth ~320 GB/s (reference lines only)
PEAK_TFLOPS_FP32 = 8.1
PEAK_GB_S = 320.0


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        headers = f.readline().strip().split(",")
        rows = []
        for line in f:
            if not line.strip():
                continue
            vals = line.strip().split(",")
            rows.append(dict(zip(headers, vals)))
    return rows


def main():
    timing = read_csv("benchmarks/results/timing.csv")
    hbm = read_csv("benchmarks/results/hbm_traffic.csv")

    if not timing:
        print("Error: Could not find benchmarks/results/timing.csv", file=sys.stderr)
        sys.exit(1)

    os.makedirs("benchmarks/results/plots", exist_ok=True)
    os.makedirs("docs/assets", exist_ok=True)

    seq_lens = sorted({int(r["seq_len"]) for r in timing})
    modes = []
    for r in timing:
        m = r["mode"]
        if m not in modes:
            modes.append(m)

    palette = [
        "#f43f5e",
        "#fb7185",
        "#38bdf8",
        "#22d3ee",
        "#a78bfa",
        "#34d399",
        "#fbbf24",
        "#94a3b8",
    ]
    color = {m: palette[i % len(palette)] for i, m in enumerate(modes)}

    # --- Latency ---
    plt.figure(figsize=(9, 5.5))
    for m in modes:
        ys = [
            float(next(r["ms"] for r in timing if r["mode"] == m and int(r["seq_len"]) == N))
            for N in seq_lens
        ]
        plt.plot(seq_lens, ys, marker="o", label=m, color=color[m], linewidth=2)
    plt.title("Attention kernel latency vs sequence length (Tesla T4)")
    plt.xlabel("Sequence length N")
    plt.ylabel("Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=8, loc="upper left")
    plt.xticks(seq_lens)
    plt.tight_layout()
    plt.savefig("benchmarks/results/plots/latency_plot.png", dpi=300)
    plt.savefig("docs/assets/latency_plot.png", dpi=300)
    plt.close()
    print("Saved latency_plot.png")

    # --- Speedup vs naive ---
    baseline = "naive"
    if any(r["mode"] == baseline for r in timing):
        plt.figure(figsize=(9, 5.5))
        for m in modes:
            if m == baseline:
                continue
            ratios = []
            for N in seq_lens:
                t0 = float(next(r["ms"] for r in timing if r["mode"] == baseline and int(r["seq_len"]) == N))
                t1 = float(next(r["ms"] for r in timing if r["mode"] == m and int(r["seq_len"]) == N))
                ratios.append(t0 / t1 if t1 > 0 else 0.0)
            plt.plot(seq_lens, ratios, marker="o", label=m + " vs naive", color=color[m], linewidth=2)
        plt.axhline(1.0, color="#64748b", linestyle="--", linewidth=1)
        plt.title("Speedup vs naive baseline")
        plt.xlabel("Sequence length N")
        plt.ylabel("Speedup (×)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=8, loc="upper left")
        plt.xticks(seq_lens)
        plt.tight_layout()
        plt.savefig("benchmarks/results/plots/speedup_plot.png", dpi=300)
        plt.savefig("docs/assets/speedup_plot.png", dpi=300)
        plt.close()
        print("Saved speedup_plot.png")

    # --- HBM traffic ---
    if hbm:
        plt.figure(figsize=(9, 5.5))
        for m in modes:
            if not any(r["mode"] == m for r in hbm):
                continue
            tot = []
            for N in seq_lens:
                row = next((r for r in hbm if r["mode"] == m and int(r["seq_len"]) == N), None)
                if row:
                    tot.append(float(row["bytes_read_MB"]) + float(row["bytes_write_MB"]))
                else:
                    tot.append(0.0)
            plt.plot(seq_lens, tot, marker="o", label=m, color=color[m], linewidth=2)
        plt.yscale("log")
        plt.title("Total HBM traffic (read + write, NCU L1TEX global bytes)")
        plt.xlabel("Sequence length N")
        plt.ylabel("Traffic (MB, log scale)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=8, loc="upper left")
        plt.xticks(seq_lens)
        plt.tight_layout()
        plt.savefig("benchmarks/results/plots/hbm_plot.png", dpi=300)
        plt.savefig("docs/assets/hbm_plot.png", dpi=300)
        plt.close()
        print("Saved hbm_plot.png")

    # --- Roofline-style plot (needs HBM CSV for intensity axis) ---
    if hbm:
        plt.figure(figsize=(8, 5.5))
        for m in modes:
            xs = []
            ys = []
            for N in seq_lens:
                row = next((r for r in timing if r["mode"] == m and int(r["seq_len"]) == N), None)
                if not row:
                    continue
                ms = float(row["ms"])
                if ms <= 0:
                    continue
                flops = 4.0 * N * N * D_HEAD
                achieved_tflops = (flops / (ms * 1e-3)) / 1e12
                hrow = next((r for r in hbm if r["mode"] == m and int(r["seq_len"]) == N), None)
                if not hrow:
                    continue
                bytes_hbm = (float(hrow["bytes_read_MB"]) + float(hrow["bytes_write_MB"])) * (1024**2)
                ai = flops / bytes_hbm if bytes_hbm > 0 else 0.0
                xs.append(ai)
                ys.append(achieved_tflops)
            if xs:
                plt.plot(xs, ys, marker="o", label=m, color=color[m], linewidth=2)

        plt.xlabel("Operational intensity (FLOP / byte HBM, approximate)")
        plt.ylabel("Achieved throughput (TFLOP/s, approximate)")
        plt.title("Roofline-style snapshot (NCU bytes + analytical FLOP count)")
        plt.axhline(PEAK_TFLOPS_FP32, color="#94a3b8", linestyle="--", label=f"~{PEAK_TFLOPS_FP32} TFLOP/s ref")
        plt.axhline(PEAK_GB_S / 1024.0, color="#cbd5e1", linestyle=":", label="memory ref (scaled)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=7, loc="best")
        plt.tight_layout()
        plt.savefig("benchmarks/results/plots/roofline_plot.png", dpi=300)
        plt.savefig("docs/assets/roofline_plot.png", dpi=300)
        plt.close()
        print("Saved roofline_plot.png")
    else:
        print("Note: benchmarks/results/hbm_traffic.csv missing — skipped roofline_plot.png")


if __name__ == "__main__":
    main()
