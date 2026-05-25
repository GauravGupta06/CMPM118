"""Evaluate cheap handcrafted DVSGesture router features.

This script tests whether simple prefix statistics can identify samples where
the dense model is correct and the sparse model is wrong. It does not estimate
energy. The point is to check signal quality before moving anything to STM32.
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from datasets.dvsgesture_dataset import DVSGestureDataset


DEFAULT_PER_SAMPLE_CSV = (
    "new_test_results/dvsgesture_from_pod_20260505_133029/"
    "prefix_router_eval/dvsgesture_prefix_router_per_sample_20260505_235126.csv"
)


INT_COLUMNS = {
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
}


def safe_div(a, b):
    return float(a) / (float(b) + 1e-9)


def entropy_norm(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    total = x.sum()
    if total <= 0 or len(x) <= 1:
        return 0.0
    p = x[x > 0] / total
    return float(-(p * np.log2(p)).sum() / math.log2(len(x)))


def gini(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    weights = 2 * np.arange(1, n + 1) - n - 1
    return float(weights.dot(x) / (n * total))


def lzc_binary(seq):
    s = "".join("1" if int(v) else "0" for v in seq)
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


def load_router_rows(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            parsed = {}
            for key, value in row.items():
                parsed[key] = int(value) if key in INT_COLUMNS else float(value)
            rows.append(parsed)
    return rows


def sample_features(sample, prefix_bins=400):
    arr = np.asarray(sample[:prefix_bins], dtype=np.float32)
    if arr.shape[0] < prefix_bins:
        fixed = np.zeros((prefix_bins, 2, 32, 32), dtype=np.float32)
        fixed[: arr.shape[0]] = arr
        arr = fixed

    binary = arr > 0
    time_binary = binary.sum(axis=(1, 2, 3)).astype(np.float64)
    time_counts = arr.sum(axis=(1, 2, 3)).astype(np.float64)
    prefix_ones = float(time_binary.sum())
    count_sum = float(time_counts.sum())

    polarity_binary = binary.sum(axis=(0, 2, 3)).astype(np.float64)
    polarity_counts = arr.sum(axis=(0, 2, 3)).astype(np.float64)

    quarter_binary = np.array(
        [binary[i * 100 : (i + 1) * 100].sum() for i in range(4)],
        dtype=np.float64,
    )
    quarter_counts = np.array(
        [arr[i * 100 : (i + 1) * 100].sum() for i in range(4)],
        dtype=np.float64,
    )

    active_time = np.flatnonzero(time_binary > 0)
    if len(active_time):
        span = float(active_time[-1] - active_time[0] + 1)
        center = float(np.arange(prefix_bins).dot(time_binary) / (prefix_ones + 1e-9))
        temporal_std = float(
            np.sqrt((((np.arange(prefix_bins) - center) ** 2) * time_binary).sum() / (prefix_ones + 1e-9))
        )
    else:
        span = 0.0
        center = 0.0
        temporal_std = 0.0

    coarse = binary.reshape(20, 20, 2, 4, 8, 4, 8).any(axis=(1, 4, 6))
    coarse_per_time = coarse.sum(axis=(1, 2, 3)).astype(np.float64)

    spatial_counts = binary.reshape(400, 2, 4, 8, 4, 8).sum(axis=(0, 3, 5)).astype(np.float64)
    spatial = spatial_counts.sum(axis=0)
    center_cells = spatial[1:3, 1:3].sum()
    edge_cells = spatial.sum() - center_cells

    timeline_10ms = (time_binary.reshape(40, 10).sum(axis=1) > 0).astype(np.uint8)

    return {
        "prefix_ones": prefix_ones,
        "count_sum": count_sum,
        "count_gt1": float((arr > 1).sum()),
        "active_ms": float((time_binary > 0).sum()),
        "activity_span_ms": span,
        "temporal_center": center,
        "temporal_std": temporal_std,
        "time_max": float(time_binary.max()),
        "time_mean_nonzero": float(time_binary[time_binary > 0].mean()) if np.any(time_binary > 0) else 0.0,
        "time_burstiness": safe_div(time_binary.max(), time_binary.mean()),
        "time_gini": gini(time_binary),
        "time_entropy": entropy_norm(time_binary),
        "q0_ones": quarter_binary[0],
        "q1_ones": quarter_binary[1],
        "q2_ones": quarter_binary[2],
        "q3_ones": quarter_binary[3],
        "q0_count": quarter_counts[0],
        "q1_count": quarter_counts[1],
        "q2_count": quarter_counts[2],
        "q3_count": quarter_counts[3],
        "early200_ones": float(quarter_binary[:2].sum()),
        "late200_ones": float(quarter_binary[2:].sum()),
        "early_late_diff": float(quarter_binary[:2].sum() - quarter_binary[2:].sum()),
        "late_early_ratio": safe_div(quarter_binary[2:].sum(), quarter_binary[:2].sum()),
        "last_first_ratio": safe_div(quarter_binary[3], quarter_binary[0]),
        "polarity_absdiff": float(abs(polarity_binary[0] - polarity_binary[1])),
        "polarity_balance": safe_div(abs(polarity_binary[0] - polarity_binary[1]), prefix_ones),
        "polarity_count_balance": safe_div(abs(polarity_counts[0] - polarity_counts[1]), count_sum),
        "coarse_active_cells": float(coarse.sum()),
        "coarse_active_time_bins": float((coarse_per_time > 0).sum()),
        "coarse_time_max": float(coarse_per_time.max()),
        "coarse_time_mean": float(coarse_per_time.mean()),
        "coarse_time_std": float(coarse_per_time.std()),
        "coarse_time_gini": gini(coarse_per_time),
        "coarse_time_entropy": entropy_norm(coarse_per_time),
        "spatial_active_4x4": float((spatial > 0).sum()),
        "spatial_max": float(spatial.max()),
        "spatial_gini": gini(spatial),
        "spatial_entropy": entropy_norm(spatial),
        "center_edge_ratio": safe_div(center_cells, edge_cells),
        "x_spread_entropy": entropy_norm(spatial.sum(axis=0)),
        "y_spread_entropy": entropy_norm(spatial.sum(axis=1)),
        "timeline_10ms_lzc": float(lzc_binary(timeline_10ms)),
        "count_per_one": safe_div(count_sum, prefix_ones),
    }


def rank01(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return ranks
    return ranks / (len(values) - 1)


def threshold_stats(score, y_dense_needed, dense_correct, sparse_correct):
    fpr, tpr, thresholds = roc_curve(y_dense_needed, score)
    gmean = np.sqrt(tpr * (1 - fpr))
    idx = int(np.argmax(gmean))
    route_dense = score >= thresholds[idx]
    routed_correct = np.where(route_dense, dense_correct, sparse_correct)
    return {
        "threshold": float(thresholds[idx]),
        "tpr": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "gmean": float(gmean[idx]),
        "routed_accuracy": float(routed_correct.mean()),
        "dense_fraction": float(route_dense.mean()),
    }


def single_feature_scores(X, names, y, dense_correct, sparse_correct):
    scored = []
    for idx, name in enumerate(names):
        values = X[:, idx]
        if np.all(values == values[0]):
            continue
        auc = roc_auc_score(y, values)
        if auc < 0.5:
            direction = "lower"
            oriented = -values
            auc = 1.0 - auc
        else:
            direction = "higher"
            oriented = values
        stats = threshold_stats(oriented, y, dense_correct, sparse_correct)
        scored.append(
            {
                "feature": name,
                "auc": float(auc),
                "direction_for_dense_needed": direction,
                **stats,
            }
        )
    return sorted(scored, key=lambda row: row["auc"], reverse=True)


def greedy_rank_router(X, names, single_scores, y, dense_correct, sparse_correct):
    oriented = []
    for row in single_scores:
        values = X[:, names.index(row["feature"])]
        if row["direction_for_dense_needed"] == "lower":
            values = -values
        oriented.append((row["feature"], row["auc"], rank01(values)))

    selected = []
    remaining = list(oriented)
    current = None
    best_auc = 0.0

    while remaining and len(selected) < 8:
        best = None
        for item in remaining:
            candidate_score = item[2] if current is None else current + item[2]
            auc = roc_auc_score(y, candidate_score)
            if auc < 0.5:
                auc = 1.0 - auc
                candidate_score = -candidate_score
            if best is None or auc > best[0]:
                best = (auc, item, candidate_score)

        if selected and best[0] <= best_auc + 0.002:
            break

        best_auc = best[0]
        selected.append({"feature": best[1][0], "single_feature_auc": float(best[1][1])})
        current = best[2]
        remaining = [item for item in remaining if item[0] != best[1][0]]

    if roc_auc_score(y, current) < 0.5:
        current = -current

    return {
        "selected_features": selected,
        "auc": float(roc_auc_score(y, current)),
        **threshold_stats(current, y, dense_correct, sparse_correct),
    }


def cross_validated_models(X, y, dense_correct, sparse_correct):
    models = {
        "logreg_l1": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                C=0.25,
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
            ),
        ),
        "tree_d2": DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=118,
        ),
        "random_forest_d2": RandomForestClassifier(
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=118,
        ),
        "extra_trees_d2": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=118,
        ),
        "gradient_boost_d2": GradientBoostingClassifier(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.03,
            random_state=118,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=118)
    out = {}
    for name, model in models.items():
        probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        out[name] = {
            "cv_auc": float(roc_auc_score(y, probs)),
            **threshold_stats(probs, y, dense_correct, sparse_correct),
        }
    return out


def write_summary(summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"dvsgesture_feature_router_summary_{stamp}.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate cheap DVS Gesture router features")
    parser.add_argument("--per_sample_csv", default=DEFAULT_PER_SAMPLE_CSV)
    parser.add_argument("--dataset_path", default="./data")
    parser.add_argument("--output_dir", default="new_test_results/dvsgesture_feature_router_eval")
    parser.add_argument("--prefix_bins", type=int, default=400)
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
    _, test_dataset = data.load_dvsgesture()
    if len(test_dataset) != len(rows_by_local_idx):
        raise ValueError(
            f"Dataset has {len(test_dataset)} samples but router CSV has {len(rows_by_local_idx)}"
        )

    feature_rows = []
    aligned_rows = []
    for local_idx in range(len(test_dataset)):
        sample, _ = test_dataset[local_idx]
        feature_rows.append(sample_features(sample, prefix_bins=args.prefix_bins))
        aligned_rows.append(rows_by_local_idx[local_idx])

    feature_names = list(feature_rows[0].keys())
    X = np.array([[row[name] for name in feature_names] for row in feature_rows], dtype=np.float64)
    dense_correct = np.array([row["dense_correct"] for row in aligned_rows], dtype=bool)
    sparse_correct = np.array([row["sparse_correct"] for row in aligned_rows], dtype=bool)
    dense_needed = (dense_correct & ~sparse_correct).astype(int)

    singles = single_feature_scores(X, feature_names, dense_needed, dense_correct, sparse_correct)
    greedy = greedy_rank_router(X, feature_names, singles, dense_needed, dense_correct, sparse_correct)
    cv_models = cross_validated_models(X, dense_needed, dense_correct, sparse_correct)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(dense_needed)),
        "target": "dense_needed = dense_correct and not sparse_correct",
        "dense_needed_count": int(dense_needed.sum()),
        "baseline": {
            "dense_accuracy": float(dense_correct.mean()),
            "sparse_accuracy": float(sparse_correct.mean()),
        },
        "best_single_features": singles[:15],
        "greedy_rank_vote_router": greedy,
        "cross_validated_tiny_models": cv_models,
        "note": (
            "The greedy rank-vote score is exploratory and fit/evaluated on the same "
            "264 samples. The cross-validated tiny model AUCs are a better warning "
            "about likely generalization."
        ),
    }
    out_path = write_summary(summary, args.output_dir)

    print("\nDVSGesture cheap feature router")
    print("=" * 72)
    print(f"Dense accuracy:  {summary['baseline']['dense_accuracy'] * 100:.2f}%")
    print(f"Sparse accuracy: {summary['baseline']['sparse_accuracy'] * 100:.2f}%")
    print(f"Dense-needed positives: {summary['dense_needed_count']} / {summary['n_samples']}")
    print("\nBest single feature:")
    best = singles[0]
    print(
        f"  {best['feature']} | AUC={best['auc']:.3f} | "
        f"direction={best['direction_for_dense_needed']} | "
        f"routed_acc={best['routed_accuracy'] * 100:.2f}%"
    )
    print("\nGreedy rank-vote router:")
    print(f"  AUC={greedy['auc']:.3f}")
    print(f"  routed_acc={greedy['routed_accuracy'] * 100:.2f}%")
    print(f"  dense_fraction={greedy['dense_fraction'] * 100:.1f}%")
    print("  selected_features=" + ", ".join(row["feature"] for row in greedy["selected_features"]))
    print("\nCross-validated warning check:")
    for name, row in cv_models.items():
        print(
            f"  {name}: cv_auc={row['cv_auc']:.3f}, "
            f"routed_acc={row['routed_accuracy'] * 100:.2f}%, "
            f"dense_fraction={row['dense_fraction'] * 100:.1f}%"
        )
    print(f"\nSummary JSON: {out_path}")


if __name__ == "__main__":
    main()
