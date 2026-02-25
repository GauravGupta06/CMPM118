"""
visualize_lzc_energy.py — Visualize LZC energy measurement results.

Usage:
    python visualize_lzc_energy.py
    python visualize_lzc_energy.py --input lzc_energy_SHD.txt
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------- config ----------
ENERGY_TABLE = "lzc_energy_UCI_HAR.txt"
CLOCK_HZ = 96_000_000  # STM32F411 @ 96 MHz


def load_data(path):
    """Load energy, cycles, and LZC score from output file."""
    energies, cycles, scores = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                energies.append(float(parts[0]))
                cycles.append(int(parts[1]))
                scores.append(int(parts[2]))
            elif len(parts) == 2:
                # Legacy format (energy, score) — estimate cycles
                energies.append(float(parts[0]))
                cycles.append(int(float(parts[0]) / 1.8e-10))
                scores.append(int(parts[1]))
    return np.array(energies), np.array(cycles), np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Visualize LZC Energy Results")
    parser.add_argument("--input", type=str, default=ENERGY_TABLE,
                        help="Input energy table file")
    args = parser.parse_args()

    energies, cycles, scores = load_data(args.input)
    n = len(energies)

    # Derive dataset name from filename
    base = os.path.basename(args.input)
    dataset_name = base.replace("lzc_energy_", "").replace(".txt", "")

    # Convert to microjoules for readability
    energies_uj = energies * 1e6

    # Latency in milliseconds
    latency_ms = cycles / CLOCK_HZ * 1000

    print(f"Loaded {n} samples ({dataset_name})")
    print(f"Energy:   min={energies_uj.min():.1f} µJ, max={energies_uj.max():.1f} µJ, "
          f"mean={energies_uj.mean():.1f} µJ, std={energies_uj.std():.1f} µJ")
    print(f"Cycles:   min={cycles.min()}, max={cycles.max()}, mean={cycles.mean():.0f}")
    print(f"Latency:  min={latency_ms.min():.3f} ms, max={latency_ms.max():.3f} ms, "
          f"mean={latency_ms.mean():.3f} ms")
    print(f"LZC:      min={scores.min()}, max={scores.max()}, "
          f"mean={scores.mean():.1f}, std={scores.std():.1f}")

    # ---------- styling ----------
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"LZC Energy Measurement — {dataset_name} — STM32F411 (Cortex-M4) → STM32L476 Projection",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ---------- 1. Scatter: Energy vs LZC Score ----------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(scores, energies_uj, alpha=0.3, s=10, c=scores,
                cmap="viridis", edgecolors="none")
    z = np.polyfit(scores, energies_uj, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(scores.min(), scores.max(), 100)
    ax1.plot(x_fit, p(x_fit), "r-", linewidth=2,
             label=f"Linear fit (r={np.corrcoef(scores, energies_uj)[0,1]:.3f})")
    ax1.set_xlabel("LZC Score")
    ax1.set_ylabel("Energy (µJ)")
    ax1.set_title("Energy vs LZC Score")
    ax1.legend(fontsize=8)

    # ---------- 2. Histogram: Energy Distribution ----------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(energies_uj, bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax2.axvline(energies_uj.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {energies_uj.mean():.1f} µJ")
    ax2.set_xlabel("Energy (µJ)")
    ax2.set_ylabel("Count")
    ax2.set_title("Energy Distribution")
    ax2.legend(fontsize=8)

    # ---------- 3. Histogram: LZC Score Distribution ----------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(scores, bins=40, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax3.axvline(scores.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {scores.mean():.1f}")
    ax3.set_xlabel("LZC Score")
    ax3.set_ylabel("Count")
    ax3.set_title("LZC Score Distribution")
    ax3.legend(fontsize=8)

    # ---------- 4. Latency Distribution ----------
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.hist(latency_ms, bins=50, color="#FF9800", edgecolor="white", alpha=0.8)
    ax4.axvline(latency_ms.mean(), color="red", linestyle="--", linewidth=2,
                label=f"Mean = {latency_ms.mean():.3f} ms")
    ax4.axvline(np.median(latency_ms), color="#1565C0", linestyle="--", linewidth=1.5,
                label=f"Median = {np.median(latency_ms):.3f} ms")
    ax4.set_xlabel("Latency (ms)")
    ax4.set_ylabel("Count")
    ax4.set_title("Latency Distribution (per-sample inference time @ 96 MHz)")
    ax4.legend(fontsize=8)

    # ---------- 5. Summary Stats Box ----------
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    corr = np.corrcoef(scores, energies_uj)[0, 1]
    total_energy = energies.sum()

    stats_text = (
        f"Dataset:           {dataset_name}\n"
        f"Samples:           {n}\n"
        f"\n"
        f"── Energy ──\n"
        f"Mean:              {energies_uj.mean():.1f} µJ\n"
        f"Median:            {np.median(energies_uj):.1f} µJ\n"
        f"Std:               {energies_uj.std():.1f} µJ\n"
        f"Min:               {energies_uj.min():.1f} µJ\n"
        f"Max:               {energies_uj.max():.1f} µJ\n"
        f"Total:             {total_energy*1e3:.2f} mJ\n"
        f"\n"
        f"── Latency ──\n"
        f"Mean:              {latency_ms.mean():.3f} ms\n"
        f"Median:            {np.median(latency_ms):.3f} ms\n"
        f"Min:               {latency_ms.min():.3f} ms\n"
        f"Max:               {latency_ms.max():.3f} ms\n"
        f"\n"
        f"── LZC ──\n"
        f"Mean:              {scores.mean():.1f}\n"
        f"Range:             [{scores.min()}, {scores.max()}]\n"
        f"\n"
        f"── Correlation ──\n"
        f"r(Energy, LZC):    {corr:.4f}\n"
        f"\n"
        f"── Hardware ──\n"
        f"Measured on:       STM32F411CEU6\n"
        f"Projected to:      STM32L476 (180 pJ/cyc)\n"
        f"Mean cycles:       {cycles.mean():.0f}"
    )

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"))

    # Save with dataset name
    output_png = f"lzc_energy_plot_{dataset_name}.png"
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {output_png}")
    plt.show()


if __name__ == "__main__":
    main()
