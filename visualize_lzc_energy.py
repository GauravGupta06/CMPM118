"""
visualize_lzc_energy.py — Visualize LZC energy measurement results.

Usage:
    python visualize_lzc_energy.py
    python visualize_lzc_energy.py --input lzc_energy_table.txt
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------- config ----------
ENERGY_TABLE = "lzc_energy_UCI_HAR.txt"


def load_data(path):
    """Load energy and LZC score from output file."""
    energies, scores = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                energies.append(float(parts[0]))
                scores.append(int(parts[1]))
    return np.array(energies), np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Visualize LZC Energy Results")
    parser.add_argument("--input", type=str, default=ENERGY_TABLE,
                        help="Input energy table file")
    args = parser.parse_args()

    energies, scores = load_data(args.input)
    n = len(energies)

    # Convert to microjoules for readability
    energies_uj = energies * 1e6

    print(f"Loaded {n} samples")
    print(f"Energy:  min={energies_uj.min():.1f} µJ, max={energies_uj.max():.1f} µJ, "
          f"mean={energies_uj.mean():.1f} µJ, std={energies_uj.std():.1f} µJ")
    print(f"LZC:     min={scores.min()}, max={scores.max()}, "
          f"mean={scores.mean():.1f}, std={scores.std():.1f}")

    # ---------- styling ----------
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("LZC Energy Measurement — STM32F411 (Cortex-M4) → STM32L476 Projection",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ---------- 1. Scatter: Energy vs LZC Score ----------
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(scores, energies_uj, alpha=0.3, s=10, c=scores,
                          cmap="viridis", edgecolors="none")
    z = np.polyfit(scores, energies_uj, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(scores.min(), scores.max(), 100)
    ax1.plot(x_fit, p(x_fit), "r-", linewidth=2, label=f"Linear fit (r={np.corrcoef(scores, energies_uj)[0,1]:.3f})")
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

    # ---------- 4. Energy per Sample (time series) ----------
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.plot(energies_uj, linewidth=0.5, alpha=0.7, color="#FF5722")
    # Rolling average
    window = 50
    if n > window:
        rolling = np.convolve(energies_uj, np.ones(window)/window, mode="valid")
        ax4.plot(np.arange(window//2, window//2 + len(rolling)), rolling,
                 linewidth=2, color="#1565C0", label=f"{window}-sample rolling avg")
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Energy (µJ)")
    ax4.set_title("Energy per Sample (ordered by dataset index)")
    ax4.legend(fontsize=8)

    # ---------- 5. Summary Stats Box ----------
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    corr = np.corrcoef(scores, energies_uj)[0, 1]
    total_energy = energies.sum()
    cycles_approx = energies / 1.8e-10

    stats_text = (
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
        f"── LZC ──\n"
        f"Mean:              {scores.mean():.1f}\n"
        f"Std:               {scores.std():.1f}\n"
        f"Range:             [{scores.min()}, {scores.max()}]\n"
        f"\n"
        f"── Correlation ──\n"
        f"r(Energy, LZC):    {corr:.4f}\n"
        f"\n"
        f"── Hardware ──\n"
        f"Measured on:       STM32F411CEU6\n"
        f"Projected to:      STM32L476 (180 pJ/cyc)\n"
        f"Mean cycles:       {cycles_approx.mean():.0f}"
    )

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"))

    plt.savefig("lzc_energy_plot_UCI_HAR.png", dpi=150, bbox_inches="tight")
    print("\nSaved → lzc_energy_plot_UCI_HAR.png")
    plt.show()


if __name__ == "__main__":
    main()
