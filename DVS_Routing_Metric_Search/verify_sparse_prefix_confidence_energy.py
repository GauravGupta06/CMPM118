"""Verify DVSGesture sparse-prefix confidence router energy.

This helper recomputes the operating points reported by
evaluate_all_dvsgesture_router_metrics.py, then applies those route decisions to
the per-sample energy columns emitted by router_dvsgesture.py.
"""

import argparse
import csv

import numpy as np


def load_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def summarize(name, score, dense_correct, sparse_correct, dense_energy, sparse_energy, max_dense_fraction=None):
    best = None
    for threshold in np.unique(score):
        route_dense = score >= threshold
        dense_fraction = float(route_dense.mean())
        if max_dense_fraction is not None and dense_fraction > max_dense_fraction:
            continue

        routed_correct = np.where(route_dense, dense_correct, sparse_correct)
        accuracy = float(routed_correct.mean())
        candidate = (accuracy, -dense_fraction, threshold, route_dense.copy())
        if best is None or candidate > best:
            best = candidate

    if best is None:
        raise RuntimeError(f"No threshold found for {name}")

    accuracy, neg_dense_fraction, threshold, route_dense = best
    dense_fraction = -neg_dense_fraction
    routed_model_energy = np.where(route_dense, dense_energy, sparse_energy)
    dense_avg_energy = float(dense_energy.mean())
    routed_avg_energy = float(routed_model_energy.mean())

    return {
        "name": name,
        "accuracy": accuracy,
        "dense_fraction": dense_fraction,
        "threshold_oriented": float(threshold),
        "avg_model_energy_J": routed_avg_energy,
        "avg_model_energy_mJ": routed_avg_energy * 1000.0,
        "avg_energy_saved_J": dense_avg_energy - routed_avg_energy,
        "avg_energy_saved_mJ": (dense_avg_energy - routed_avg_energy) * 1000.0,
        "energy_savings_percent": 100.0 * (1.0 - routed_model_energy.sum() / dense_energy.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per_sample_csv",
        default="new_test_results/dvsgesture/prefix_router_eval/dvsgesture_prefix_router_per_sample_20260524_193020.csv",
    )
    parser.add_argument(
        "--confidence_cache",
        default="DVS_Routing_Metric_Search/sparse_prefix_confidence_cache_p460.npz",
    )
    args = parser.parse_args()

    rows = load_rows(args.per_sample_csv)
    rows_by_local_idx = {int(row["stm32_sample_idx"]): row for row in rows}
    aligned = [rows_by_local_idx[idx] for idx in range(len(rows_by_local_idx))]

    cache = np.load(args.confidence_cache)
    score = -cache["sparse_prefix_margin"]

    dense_correct = np.array([int(row["dense_correct"]) for row in aligned], dtype=bool)
    sparse_correct = np.array([int(row["sparse_correct"]) for row in aligned], dtype=bool)
    dense_energy = np.array([float(row["baseline_dense_energy_J"]) for row in aligned])
    sparse_energy = np.array([float(row["sparse_only_energy_J"]) for row in aligned])

    if len(score) != len(aligned):
        raise RuntimeError(f"Cache has {len(score)} samples, CSV has {len(aligned)}")

    summaries = [
        summarize("best_any_threshold", score, dense_correct, sparse_correct, dense_energy, sparse_energy),
        summarize(
            "best_dense_fraction_le_60",
            score,
            dense_correct,
            sparse_correct,
            dense_energy,
            sparse_energy,
            max_dense_fraction=0.60,
        ),
        summarize(
            "best_dense_fraction_le_75",
            score,
            dense_correct,
            sparse_correct,
            dense_energy,
            sparse_energy,
            max_dense_fraction=0.75,
        ),
    ]

    print(f"dense_accuracy={dense_correct.mean():.12f}")
    print(f"sparse_accuracy={sparse_correct.mean():.12f}")
    print(f"dense_avg_energy_mJ={dense_energy.mean() * 1000.0:.12f}")
    print(f"sparse_avg_energy_mJ={sparse_energy.mean() * 1000.0:.12f}")
    for row in summaries:
        print(
            "{name}: accuracy={accuracy:.12f}, dense_fraction={dense_fraction:.12f}, "
            "threshold_oriented={threshold_oriented:.12f}, "
            "avg_model_energy_mJ={avg_model_energy_mJ:.12f}, "
            "avg_energy_saved_mJ={avg_energy_saved_mJ:.12f}, "
            "energy_savings_percent={energy_savings_percent:.12f}".format(**row)
        )


if __name__ == "__main__":
    main()
