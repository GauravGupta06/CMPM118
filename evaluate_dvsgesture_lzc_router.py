"""Evaluate a compact LZC-style DVSGesture router metric.

The current STM32 router uses raw binary prefix occupancy popcount. That is very
cheap, but it is a weak separator for the dense-needed samples. This script tests
a still-cheap structural metric:

    coarse_lzc640

Metric definition:
  1. Use the first 400 ms of the DVS Gesture count-frame input.
  2. Convert to binary occupancy: count > 0.
  3. Pool into 20 time bins of 20 ms each.
  4. Pool 32x32 pixels into a 4x4 spatial grid, keeping polarity separate.
  5. Mark each coarse cell active if any event occurred in it.
  6. Flatten [20 time bins, 2 polarities, 4, 4] into a 640-bit sequence.
  7. Compute Lempel-Ziv complexity over that 640-bit sequence.

The route direction found on the current dense/sparse DVS pair is:

    route dense if coarse_lzc640 <= threshold

This is empirical. The sparse model tends to fail on some low-LZC prefixes that
do not give it enough robust early evidence, even though "low complexity" sounds
like it should be easier.
"""

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np

from datasets.dvsgesture_dataset import DVSGestureDataset


DEFAULT_PER_SAMPLE_CSV = (
    "new_test_results/dvsgesture_from_pod_20260505_133029/"
    "prefix_router_eval/dvsgesture_prefix_router_per_sample_20260505_235126.csv"
)


def lzc_binary(seq):
    """LZ76/Kaspar-Schuster complexity for a short binary sequence."""
    s = "".join("1" if int(x) else "0" for x in seq)
    n = len(s)
    if n == 0:
        return 0

    i = 0
    k = 1
    ell = 1
    c = 1
    kmax = 1

    while True:
        if ell + k > n:
            c += 1
            break

        if s[i + k - 1] == s[ell + k - 1]:
            k += 1
            if ell + k > n:
                c += 1
                break
        else:
            if k > kmax:
                kmax = k
            i += 1
            if i == ell:
                c += 1
                ell += kmax
                if ell + 1 > n:
                    break
                i = 0
                k = 1
                kmax = 1
            else:
                k = 1

    return c


def coarse_lzc640(sample, prefix_bins=400, time_bins=20, grid=4):
    arr = np.asarray(sample[:prefix_bins], dtype=np.float32)
    if arr.ndim != 4 or arr.shape[1:] != (2, 32, 32):
        raise ValueError(f"Expected sample shape [T, 2, 32, 32], got {arr.shape}")

    if arr.shape[0] < prefix_bins:
        fixed = np.zeros((prefix_bins, 2, 32, 32), dtype=np.float32)
        fixed[: arr.shape[0]] = arr
        arr = fixed

    if prefix_bins % time_bins != 0:
        raise ValueError("prefix_bins must divide evenly into time_bins")
    if 32 % grid != 0:
        raise ValueError("grid must divide 32")

    frames_per_bin = prefix_bins // time_bins
    pixels_per_cell = 32 // grid
    binary = arr > 0
    coarse = binary.reshape(
        time_bins,
        frames_per_bin,
        2,
        grid,
        pixels_per_cell,
        grid,
        pixels_per_cell,
    ).any(axis=(1, 4, 6))

    return int(lzc_binary(coarse.reshape(-1)))


def load_router_rows(path):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in {
                    "sample_idx",
                    "stm32_sample_idx",
                    "target",
                    "dense_pred",
                    "sparse_pred",
                    "routed_pred",
                    "route_dense",
                    "dense_correct",
                    "sparse_correct",
                    "routed_correct",
                    "binary_prefix_ones",
                    "count_prefix_sum",
                    "count_bins_gt1",
                    "router_cycles",
                }:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def metric_auc(labels, scores):
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def summarize_threshold(rows, threshold, route_dense_when_lower=True, router_scale=1.0):
    routed_correct = []
    routed_energy = []
    route_dense_flags = []

    for row in rows:
        score = row["coarse_lzc640"]
        route_dense = score <= threshold if route_dense_when_lower else score >= threshold
        route_dense_flags.append(route_dense)
        routed_correct.append(row["dense_correct"] if route_dense else row["sparse_correct"])
        model_energy = (
            row["baseline_dense_energy_J"] if route_dense else row["sparse_only_energy_J"]
        )
        routed_energy.append(model_energy + row["router_energy_J"] * router_scale)

    baseline_total = sum(row["baseline_dense_energy_J"] for row in rows)
    routed_total = float(np.sum(routed_energy))
    return {
        "threshold": int(threshold),
        "route_dense_when_lower": route_dense_when_lower,
        "accuracy": float(np.mean(routed_correct)),
        "dense_fraction": float(np.mean(route_dense_flags)),
        "avg_energy_J": float(np.mean(routed_energy)),
        "energy_savings_percent": float(100.0 * (1.0 - routed_total / baseline_total)),
    }


def threshold_sweep(rows, router_scale=1.0):
    scores = sorted(set(row["coarse_lzc640"] for row in rows))
    thresholds = [scores[0] - 1] + scores + [scores[-1] + 1]
    sweep = []
    for threshold in thresholds:
        sweep.append(summarize_threshold(rows, threshold, True, router_scale))
        sweep.append(summarize_threshold(rows, threshold, False, router_scale))

    best_accuracy = max(sweep, key=lambda x: (x["accuracy"], x["energy_savings_percent"]))

    def best_at(min_acc):
        candidates = [x for x in sweep if x["accuracy"] >= min_acc]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x["energy_savings_percent"])

    return {
        "best_accuracy": best_accuracy,
        "best_energy_at_accuracy_ge_80": best_at(0.80),
        "best_energy_at_accuracy_ge_85": best_at(0.85),
        "best_energy_at_accuracy_ge_88": best_at(0.88),
        "best_energy_at_accuracy_ge_sparse_accuracy": best_at(
            float(np.mean([row["sparse_correct"] for row in rows]))
        ),
    }


def write_outputs(rows, summary, output_path):
    os.makedirs(output_path, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_path, f"dvsgesture_coarse_lzc640_router_{stamp}.csv")
    json_path = os.path.join(output_path, f"dvsgesture_coarse_lzc640_router_summary_{stamp}.json")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate compact LZC DVS Gesture router")
    parser.add_argument("--per_sample_csv", default=DEFAULT_PER_SAMPLE_CSV)
    parser.add_argument("--dataset_path", default="./data")
    parser.add_argument("--output_path", default="new_test_results/dvsgesture_lzc_router_eval")
    parser.add_argument("--prefix_bins", type=int, default=400)
    parser.add_argument("--time_bins", type=int, default=20)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument(
        "--router_energy_scale",
        type=float,
        default=1.0,
        help="Scale current measured popcount router energy as a placeholder for LZC cost",
    )
    args = parser.parse_args()

    rows = load_router_rows(args.per_sample_csv)
    rows_by_local_idx = {row["stm32_sample_idx"]: row for row in rows}

    data = DVSGestureDataset(
        dataset_path=args.dataset_path,
        w=32,
        h=32,
        max_timesteps=600,
        binarize=False,
        denoise_filter_time=10000,
    )
    _, cached_test = data.load_dvsgesture()
    if len(cached_test) != len(rows_by_local_idx):
        raise ValueError(
            f"Dataset has {len(cached_test)} samples, but router CSV has {len(rows_by_local_idx)}"
        )

    for local_idx in range(len(cached_test)):
        sample, label = cached_test[local_idx]
        row = rows_by_local_idx[local_idx]
        if int(label) != row["target"]:
            # The router CSV target is in pod order, so only the STM32/local row has
            # to match the local label. The pod row target can differ before remap.
            pass
        row["coarse_lzc640"] = coarse_lzc640(
            sample,
            prefix_bins=args.prefix_bins,
            time_bins=args.time_bins,
            grid=args.grid,
        )

    enriched_rows = sorted(rows_by_local_idx.values(), key=lambda r: r["sample_idx"])
    dense_needed = np.array(
        [row["dense_correct"] and not row["sparse_correct"] for row in enriched_rows],
        dtype=np.int64,
    )
    scores = np.array([row["coarse_lzc640"] for row in enriched_rows], dtype=np.float64)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "metric": {
            "name": "coarse_lzc640",
            "description": (
                "LZC over 20 time bins x 2 polarities x 4x4 spatial active-cell "
                "mask from first 400 ms"
            ),
            "prefix_bins": args.prefix_bins,
            "time_bins": args.time_bins,
            "grid": args.grid,
            "sequence_bits": args.time_bins * 2 * args.grid * args.grid,
            "auc_for_dense_needed": metric_auc(dense_needed, scores),
            "auc_for_dense_needed_flipped": metric_auc(dense_needed, -scores),
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
            "score_min": int(scores.min()),
            "score_max": int(scores.max()),
        },
        "baseline": {
            "dense_accuracy": float(np.mean([row["dense_correct"] for row in enriched_rows])),
            "sparse_accuracy": float(np.mean([row["sparse_correct"] for row in enriched_rows])),
            "dense_energy_avg_J": float(
                np.mean([row["baseline_dense_energy_J"] for row in enriched_rows])
            ),
            "sparse_energy_avg_J": float(
                np.mean([row["sparse_only_energy_J"] for row in enriched_rows])
            ),
        },
        "router_energy_note": (
            "Energy uses the existing STM32 popcount router rows scaled by "
            "router_energy_scale as a placeholder. Measure coarse_lzc640 on STM32 "
            "before using the energy column as final paper evidence."
        ),
        "router_energy_scale": args.router_energy_scale,
        "threshold_sweep": threshold_sweep(enriched_rows, args.router_energy_scale),
        "paths": {
            "input_per_sample_csv": args.per_sample_csv,
        },
    }

    csv_path, json_path = write_outputs(enriched_rows, summary, args.output_path)
    summary["paths"]["per_sample_csv"] = csv_path
    summary["paths"]["summary_json"] = json_path
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nDVSGesture compact LZC router metric")
    print("=" * 72)
    print(f"Dense accuracy:  {summary['baseline']['dense_accuracy'] * 100:.2f}%")
    print(f"Sparse accuracy: {summary['baseline']['sparse_accuracy'] * 100:.2f}%")
    print(f"AUC dense-needed: {summary['metric']['auc_for_dense_needed']:.3f}")
    print(f"AUC flipped:      {summary['metric']['auc_for_dense_needed_flipped']:.3f}")
    for key, value in summary["threshold_sweep"].items():
        if value is None:
            continue
        print(
            f"{key}: threshold={value['threshold']} "
            f"route={'low' if value['route_dense_when_lower'] else 'high'} "
            f"acc={value['accuracy'] * 100:.2f}% "
            f"dense={value['dense_fraction'] * 100:.1f}% "
            f"savings={value['energy_savings_percent']:.2f}%"
        )
    print(f"Per-sample CSV: {csv_path}")
    print(f"Summary JSON:   {json_path}")


if __name__ == "__main__":
    main()
