"""Centralized DVSGesture router-metric search.

This file is intentionally self-contained. It evaluates many early-prefix
metrics for deciding whether a sample should use the high-accuracy DVS model or
the low-spike DVS model. It does not estimate energy.

Target label for AUC:
    dense_needed = dense_correct and not sparse_correct

Default routing rule per metric:
    Orient the score so larger means "more likely dense_needed".
    Pick the ROC/G-mean threshold.
    Route dense when oriented_score >= threshold.

All metrics use only the first 400 ms unless their name explicitly says model
confidence. Model-confidence metrics run the sparse model on the first 400 ms
only, not the full sample.
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.dvsgesture_dataset import DVSGestureDataset
from models.dvsgesture_model import DVSGestureSNN


DEFAULT_PER_SAMPLE_CSV = (
    "new_test_results/dvsgesture_from_pod_20260505_133029/"
    "prefix_router_eval/dvsgesture_prefix_router_per_sample_20260505_235126.csv"
)
DEFAULT_SPARSE_MODEL = (
    "new_test_results/dvsgesture_from_pod_20260505_133029/"
    "new_sparse/models/accepted_new_sparse_acc7188_spikes6314.pth"
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


def sobel_energy(frame):
    gx = (
        -frame[:-2, :-2]
        - 2 * frame[1:-1, :-2]
        - frame[2:, :-2]
        + frame[:-2, 2:]
        + 2 * frame[1:-1, 2:]
        + frame[2:, 2:]
    )
    gy = (
        -frame[:-2, :-2]
        - 2 * frame[:-2, 1:-1]
        - frame[:-2, 2:]
        + frame[2:, :-2]
        + 2 * frame[2:, 1:-1]
        + frame[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.std()), float(mag.mean()), float(mag.max())


def fft_highfreq_energy(frame):
    freq = np.fft.fft2(frame)
    mag = np.abs(freq)
    h, w = mag.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    shifted = np.fft.fftshift(mag)
    mask = dist > (min(h, w) * 0.20)
    return float(shifted[mask].sum() / (shifted.sum() + 1e-9))


def load_router_rows(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            parsed = {}
            for key, value in row.items():
                parsed[key] = int(value) if key in INT_COLUMNS else float(value)
            rows.append(parsed)
    return rows


def fixed_prefix(sample, prefix_bins):
    arr = np.asarray(sample[:prefix_bins], dtype=np.float32)
    if arr.shape[0] < prefix_bins:
        fixed = np.zeros((prefix_bins, 2, 32, 32), dtype=np.float32)
        fixed[: arr.shape[0]] = arr
        arr = fixed
    return arr


def sample_metrics(sample, prefix_bins=400):
    arr = fixed_prefix(sample, prefix_bins)
    binary = arr > 0
    summed_frames = arr.sum(axis=1)          # [T, H, W]
    binary_frames = binary.sum(axis=1) > 0   # [T, H, W]

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
        center_t = float(np.arange(prefix_bins).dot(time_binary) / (prefix_ones + 1e-9))
        temporal_std = float(
            np.sqrt((((np.arange(prefix_bins) - center_t) ** 2) * time_binary).sum() / (prefix_ones + 1e-9))
        )
    else:
        span = 0.0
        center_t = 0.0
        temporal_std = 0.0

    # 20 x 2 x 4 x 4 coarse occupancy. For the default 400 ms prefix this is
    # 20 ms per bin; for other prefix lengths the number of coarse bins stays
    # fixed and the frames-per-bin changes.
    coarse_time_bins = 20
    if prefix_bins % coarse_time_bins != 0:
        raise ValueError("prefix_bins must divide evenly into 20 coarse time bins")
    frames_per_coarse_bin = prefix_bins // coarse_time_bins
    coarse = binary.reshape(
        coarse_time_bins,
        frames_per_coarse_bin,
        2,
        4,
        8,
        4,
        8,
    ).any(axis=(1, 4, 6))
    coarse_bits = coarse.reshape(-1).astype(np.uint8)
    coarse_per_time = coarse.sum(axis=(1, 2, 3)).astype(np.float64)

    # Space-time event cloud statistics on a 20 x 4 x 4 grid.
    cloud = binary.reshape(
        coarse_time_bins,
        frames_per_coarse_bin,
        2,
        4,
        8,
        4,
        8,
    ).sum(axis=(1, 2, 4, 6)).astype(np.float64)
    coords = []
    weights = []
    for ti in range(coarse_time_bins):
        for yi in range(4):
            for xi in range(4):
                w = cloud[ti, yi, xi]
                if w > 0:
                    coords.append([ti / max(coarse_time_bins - 1, 1), yi / 3.0, xi / 3.0])
                    weights.append(w)
    if weights:
        coords = np.asarray(coords, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        mean = (coords * weights[:, None]).sum(axis=0) / weights.sum()
        centered = coords - mean
        cov = (centered * weights[:, None]).T.dot(centered) / weights.sum()
        eig = np.linalg.eigvalsh(cov)
        eig = np.maximum(eig, 0.0)
        eventcloud_trace = float(eig.sum())
        eventcloud_anisotropy = float(eig[-1] / (eig.sum() + 1e-9))
        eventcloud_linearity = float((eig[-1] - eig[-2]) / (eig[-1] + 1e-9))
    else:
        eventcloud_trace = 0.0
        eventcloud_anisotropy = 0.0
        eventcloud_linearity = 0.0

    spatial_counts = binary.reshape(prefix_bins, 2, 4, 8, 4, 8).sum(axis=(0, 3, 5)).astype(np.float64)
    spatial = spatial_counts.sum(axis=0)
    center_cells = spatial[1:3, 1:3].sum()
    edge_cells = spatial.sum() - center_cells

    # Time surface / surface of active events style summaries.
    last_time = np.full((2, 32, 32), -1.0, dtype=np.float32)
    for t in range(prefix_bins):
        active = binary[t]
        last_time[active] = t
    seen = last_time >= 0
    recency = np.zeros_like(last_time, dtype=np.float32)
    recency[seen] = (prefix_bins - 1 - last_time[seen]) / max(prefix_bins - 1, 1)
    latest_age_mean = float(recency[seen].mean()) if seen.any() else 0.0
    latest_age_std = float(recency[seen].std()) if seen.any() else 0.0
    latest_seen_fraction = float(seen.mean())

    # Video SI/TI style: aggregate to 20 count frames, then Sobel and frame diff.
    coarse_count_frames = summed_frames.reshape(
        coarse_time_bins,
        frames_per_coarse_bin,
        32,
        32,
    ).sum(axis=1)
    si_std = []
    si_mean = []
    si_max = []
    fft_hf = []
    for frame in coarse_count_frames:
        a, b, c = sobel_energy(frame)
        si_std.append(a)
        si_mean.append(b)
        si_max.append(c)
        fft_hf.append(fft_highfreq_energy(frame))
    diff = np.diff(coarse_count_frames, axis=0)
    ti_sad = np.abs(diff).reshape(19, -1).sum(axis=1) if len(diff) else np.array([0.0])
    ti_mse = (diff * diff).reshape(19, -1).mean(axis=1) if len(diff) else np.array([0.0])

    # Coarse frame transition counts and run changes.
    coarse_time_active = coarse_per_time > 0
    ten_ms_bins = max(prefix_bins // 10, 1)
    transition_10ms = np.abs(
        np.diff((time_binary.reshape(ten_ms_bins, 10).sum(axis=1) > 0).astype(int))
    ).sum()
    transition_coarse_cells = np.abs(np.diff(coarse.astype(np.int16), axis=0)).sum()

    features = {
        # Existing/simple metrics.
        "popcount_prefix_ones": prefix_ones,
        "count_sum": count_sum,
        "count_gt1": float((arr > 1).sum()),
        "coarse_active_cells": float(coarse_bits.sum()),
        "coarse_lzc640": float(lzc_binary(coarse_bits)),
        # Temporal density/shape.
        "active_ms": float((time_binary > 0).sum()),
        "activity_span_ms": span,
        "temporal_center": center_t,
        "temporal_std": temporal_std,
        "time_max": float(time_binary.max()),
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
        "early_late_diff": float(quarter_binary[:2].sum() - quarter_binary[2:].sum()),
        "late_early_ratio": safe_div(quarter_binary[2:].sum(), quarter_binary[:2].sum()),
        "last_first_ratio": safe_div(quarter_binary[3], quarter_binary[0]),
        # Polarity.
        "polarity_absdiff": float(abs(polarity_binary[0] - polarity_binary[1])),
        "polarity_balance": safe_div(abs(polarity_binary[0] - polarity_binary[1]), prefix_ones),
        "polarity_count_balance": safe_div(abs(polarity_counts[0] - polarity_counts[1]), count_sum),
        # Spatial/coarse spread.
        "coarse_active_time_bins": float((coarse_per_time > 0).sum()),
        "coarse_time_max": float(coarse_per_time.max()),
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
        # Event-cloud / time-surface / video complexity inspired metrics.
        "eventcloud_trace": eventcloud_trace,
        "eventcloud_anisotropy": eventcloud_anisotropy,
        "eventcloud_linearity": eventcloud_linearity,
        "latest_age_mean": latest_age_mean,
        "latest_age_std": latest_age_std,
        "latest_seen_fraction": latest_seen_fraction,
        "siti_spatial_sobel_std_mean": float(np.mean(si_std)),
        "siti_spatial_sobel_std_max": float(np.max(si_std)),
        "siti_spatial_sobel_mean_mean": float(np.mean(si_mean)),
        "siti_temporal_sad_mean": float(np.mean(ti_sad)),
        "siti_temporal_sad_max": float(np.max(ti_sad)),
        "siti_temporal_mse_mean": float(np.mean(ti_mse)),
        "vca_lite_fft_highfreq_mean": float(np.mean(fft_hf)),
        "vca_lite_fft_highfreq_max": float(np.max(fft_hf)),
        "transition_10ms_active": float(transition_10ms),
        "transition_coarse_cells": float(transition_coarse_cells),
        "timeline_10ms_lzc": float(lzc_binary(time_binary.reshape(ten_ms_bins, 10).sum(axis=1) > 0)),
    }
    features["count_per_one"] = safe_div(count_sum, prefix_ones)
    features["coarse_per_one"] = safe_div(features["coarse_active_cells"], prefix_ones)
    return features


def metric_table_for_scores(metric_scores, y_dense_needed, dense_correct, sparse_correct):
    table = []
    for name, score_values in metric_scores.items():
        score = np.asarray(score_values, dtype=np.float64)
        if np.all(score == score[0]):
            continue
        raw_auc = roc_auc_score(y_dense_needed, score)
        if raw_auc < 0.5:
            oriented = -score
            auc = 1.0 - raw_auc
            direction = "lower"
        else:
            oriented = score
            auc = raw_auc
            direction = "higher"
        fpr, tpr, thresholds = roc_curve(y_dense_needed, oriented)
        gmean = np.sqrt(tpr * (1 - fpr))
        idx = int(np.argmax(gmean))
        route_dense = oriented >= thresholds[idx]
        routed_correct = np.where(route_dense, dense_correct, sparse_correct)

        # Also compute a best-routed-accuracy threshold for diagnosis. This can
        # collapse to all dense, so the G-mean result is the primary one.
        best_acc = -1.0
        best_dense_frac = 0.0
        best_acc_dense_le_60 = -1.0
        best_acc_dense_le_60_frac = 0.0
        best_acc_dense_le_75 = -1.0
        best_acc_dense_le_75_frac = 0.0
        min_dense_for_875_acc = None
        min_dense_for_875_frac = None
        for threshold in np.unique(oriented):
            rd = oriented >= threshold
            acc = float(np.where(rd, dense_correct, sparse_correct).mean())
            dense_frac = float(rd.mean())
            if (acc, -dense_frac) > (best_acc, -best_dense_frac):
                best_acc = acc
                best_dense_frac = dense_frac
            if dense_frac <= 0.60 and (acc, -dense_frac) > (best_acc_dense_le_60, -best_acc_dense_le_60_frac):
                best_acc_dense_le_60 = acc
                best_acc_dense_le_60_frac = dense_frac
            if dense_frac <= 0.75 and (acc, -dense_frac) > (best_acc_dense_le_75, -best_acc_dense_le_75_frac):
                best_acc_dense_le_75 = acc
                best_acc_dense_le_75_frac = dense_frac
            if acc >= 0.875 and (
                min_dense_for_875_frac is None or dense_frac < min_dense_for_875_frac
            ):
                min_dense_for_875_acc = acc
                min_dense_for_875_frac = dense_frac

        table.append(
            {
                "metric": name,
                "auc": float(auc),
                "raw_auc": float(raw_auc),
                "dense_needed_direction": direction,
                "gmean": float(gmean[idx]),
                "tpr": float(tpr[idx]),
                "fpr": float(fpr[idx]),
                "threshold_oriented": float(thresholds[idx]),
                "routed_accuracy": float(routed_correct.mean()),
                "dense_fraction": float(route_dense.mean()),
                "best_routed_accuracy_any_threshold": best_acc,
                "best_routed_dense_fraction": best_dense_frac,
                "best_routed_accuracy_dense_le_60": best_acc_dense_le_60,
                "best_routed_dense_fraction_dense_le_60": best_acc_dense_le_60_frac,
                "best_routed_accuracy_dense_le_75": best_acc_dense_le_75,
                "best_routed_dense_fraction_dense_le_75": best_acc_dense_le_75_frac,
                "min_dense_fraction_for_87p5_accuracy": min_dense_for_875_frac,
                "accuracy_at_min_dense_fraction_for_87p5": min_dense_for_875_acc,
            }
        )
    return sorted(table, key=lambda row: row["auc"], reverse=True)


def add_rank_vote_and_cv(metric_scores, rows, y, dense_correct, sparse_correct):
    names = list(metric_scores.keys())
    X = np.array([metric_scores[name] for name in names], dtype=np.float64).T

    valid = [idx for idx in range(len(names)) if not np.all(X[:, idx] == X[0, idx])]
    names = [names[idx] for idx in valid]
    X = X[:, valid]

    single_rows = metric_table_for_scores(
        {names[i]: X[:, i] for i in range(len(names))},
        y,
        dense_correct,
        sparse_correct,
    )

    def rank01(values):
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks / max(len(values) - 1, 1)

    oriented = []
    for row in single_rows:
        values = X[:, names.index(row["metric"])]
        if row["dense_needed_direction"] == "lower":
            values = -values
        oriented.append((row["metric"], row["auc"], rank01(values)))

    selected = []
    current = None
    best_auc = 0.0
    remaining = list(oriented)
    while remaining and len(selected) < 8:
        best = None
        for item in remaining:
            candidate = item[2] if current is None else current + item[2]
            auc = roc_auc_score(y, candidate)
            if auc < 0.5:
                auc = 1 - auc
                candidate = -candidate
            if best is None or auc > best[0]:
                best = (auc, item, candidate)
        if selected and best[0] <= best_auc + 0.002:
            break
        best_auc = best[0]
        selected.append({"metric": best[1][0], "single_auc": float(best[1][1])})
        current = best[2]
        remaining = [item for item in remaining if item[0] != best[1][0]]

    rows.extend(
        metric_table_for_scores(
            {"rank_vote_input_features": current},
            y,
            dense_correct,
            sparse_correct,
        )
    )
    rows[-1]["selected_metrics"] = selected
    rows[-1]["note"] = "Exploratory same-set rank-vote over input-only metrics."

    # Conservative warning check: tiny models with cross-validated predictions.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=118)
    models = {
        "cv_logreg_l1_input_features": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.25, penalty="l1", solver="liblinear", class_weight="balanced"),
        ),
        "cv_tree_depth2_input_features": DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=12, class_weight="balanced", random_state=118
        ),
        "cv_random_forest_depth2_input_features": RandomForestClassifier(
            n_estimators=300, max_depth=2, min_samples_leaf=8, class_weight="balanced", random_state=118
        ),
        "cv_extra_trees_depth2_input_features": ExtraTreesClassifier(
            n_estimators=300, max_depth=2, min_samples_leaf=8, class_weight="balanced", random_state=118
        ),
        "cv_gradient_boost_depth2_input_features": GradientBoostingClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.03, random_state=118
        ),
    }
    for name, model in models.items():
        try:
            probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            continue
        cv_row = metric_table_for_scores({name: probs}, y, dense_correct, sparse_correct)[0]
        cv_row["note"] = "Cross-validated tiny model; useful as a generalization warning."
        rows.append(cv_row)


def load_sparse_prefix_confidence(args, test_dataset):
    cache_path = os.path.join(
        args.output_dir,
        f"sparse_prefix_confidence_cache_p{args.prefix_bins}.npz",
    )
    if os.path.exists(cache_path) and not args.recompute_model_metrics:
        cache = np.load(cache_path)
        return {key: cache[key] for key in cache.files}

    device = torch.device(args.device)
    hp = DVSGestureSNN.load_hyperparams(args.sparse_model_path, device=device)
    model = DVSGestureSNN(
        input_size=hp.get("input_size", None),
        n_frames=hp.get("n_frames", 600),
        tau_mem=hp.get("tau_mem", None),
        spike_lam=hp.get("spike_lam", 0.0),
        model_type=hp.get("model_type", "sparse"),
        device=device,
        num_classes=hp.get("num_classes", 11),
        lr=hp.get("lr", 0.001),
        dt=hp.get("dt", 0.001),
        threshold=hp.get("threshold", 1.0),
        has_bias=hp.get("has_bias", False),
        beta=hp.get("beta", 0.93),
        surrogate_slope=hp.get("surrogate_slope", 9.70),
        reset_mechanism=hp.get("reset_mechanism", "subtract"),
    )
    model.load_model(args.sparse_model_path)
    model.eval()

    max_rates = []
    margins = []
    entropies = []
    output_spikes_total = []
    nonzero_classes = []
    prefix_preds = []

    arrays = []
    for idx in range(len(test_dataset)):
        sample, _ = test_dataset[idx]
        arrays.append(fixed_prefix(sample, args.prefix_bins))
        if len(arrays) == args.model_batch_size or idx == len(test_dataset) - 1:
            batch = torch.tensor(np.stack(arrays), dtype=torch.float32, device=device)
            with torch.no_grad():
                output_spikes, _ = model(batch, record=False)
                rates = output_spikes.sum(dim=1) / output_spikes.shape[1]
                sorted_rates, preds = torch.sort(rates, dim=1, descending=True)
                probs = torch.softmax(rates, dim=1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1) / math.log(rates.shape[1])
            max_rates.extend(sorted_rates[:, 0].cpu().numpy().tolist())
            margins.extend((sorted_rates[:, 0] - sorted_rates[:, 1]).cpu().numpy().tolist())
            entropies.extend(entropy.cpu().numpy().tolist())
            output_spikes_total.extend(output_spikes.sum(dim=(1, 2)).cpu().numpy().tolist())
            nonzero_classes.extend((rates > 0).sum(dim=1).cpu().numpy().tolist())
            prefix_preds.extend(preds[:, 0].cpu().numpy().tolist())
            arrays = []

    out = {
        "sparse_prefix_max_rate": np.asarray(max_rates, dtype=np.float64),
        "sparse_prefix_margin": np.asarray(margins, dtype=np.float64),
        "sparse_prefix_entropy": np.asarray(entropies, dtype=np.float64),
        "sparse_prefix_output_spikes": np.asarray(output_spikes_total, dtype=np.float64),
        "sparse_prefix_nonzero_classes": np.asarray(nonzero_classes, dtype=np.float64),
        "sparse_prefix_pred": np.asarray(prefix_preds, dtype=np.int64),
    }
    np.savez(cache_path, **out)
    return out


def write_outputs(args, summary, table):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "dvsgesture_router_metric_search_table.csv")
    json_path = os.path.join(args.output_dir, "dvsgesture_router_metric_search_summary.json")

    fieldnames = [
        "metric",
        "auc",
        "raw_auc",
        "dense_needed_direction",
        "gmean",
        "tpr",
        "fpr",
        "routed_accuracy",
        "dense_fraction",
        "accuracy_gap_vs_dense",
        "best_routed_accuracy_any_threshold",
        "best_routed_dense_fraction",
        "best_routed_accuracy_dense_le_60",
        "best_routed_dense_fraction_dense_le_60",
        "best_routed_accuracy_dense_le_75",
        "best_routed_dense_fraction_dense_le_75",
        "accuracy_at_min_dense_fraction_for_87p5",
        "min_dense_fraction_for_87p5_accuracy",
        "note",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Search DVSGesture router metrics")
    parser.add_argument("--per_sample_csv", default=DEFAULT_PER_SAMPLE_CSV)
    parser.add_argument("--dataset_path", default="./data")
    parser.add_argument("--output_dir", default="DVS_Routing_Metric_Search")
    parser.add_argument("--prefix_bins", type=int, default=400)
    parser.add_argument("--include_model_metrics", action="store_true")
    parser.add_argument("--recompute_model_metrics", action="store_true")
    parser.add_argument("--sparse_model_path", default=DEFAULT_SPARSE_MODEL)
    parser.add_argument("--model_batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_router_rows(args.per_sample_csv)
    rows_by_local = {row["stm32_sample_idx"]: row for row in rows}

    data = DVSGestureDataset(
        dataset_path=args.dataset_path,
        w=32,
        h=32,
        max_timesteps=600,
        binarize=False,
        denoise_filter_time=10000,
    )
    _, test_dataset = data.load_dvsgesture()
    if len(test_dataset) != len(rows_by_local):
        raise ValueError(
            f"Dataset has {len(test_dataset)} samples but router CSV has {len(rows_by_local)}"
        )

    feature_rows = []
    aligned = []
    for local_idx in range(len(test_dataset)):
        sample, _ = test_dataset[local_idx]
        feature_rows.append(sample_metrics(sample, prefix_bins=args.prefix_bins))
        aligned.append(rows_by_local[local_idx])

    dense_correct = np.asarray([row["dense_correct"] for row in aligned], dtype=bool)
    sparse_correct = np.asarray([row["sparse_correct"] for row in aligned], dtype=bool)
    dense_needed = (dense_correct & ~sparse_correct).astype(int)
    dense_accuracy = float(dense_correct.mean())
    sparse_accuracy = float(sparse_correct.mean())

    metric_names = list(feature_rows[0].keys())
    metric_scores = {
        name: np.asarray([row[name] for row in feature_rows], dtype=np.float64)
        for name in metric_names
    }

    table = metric_table_for_scores(metric_scores, dense_needed, dense_correct, sparse_correct)
    add_rank_vote_and_cv(metric_scores, table, dense_needed, dense_correct, sparse_correct)

    if args.include_model_metrics:
        model_scores = load_sparse_prefix_confidence(args, test_dataset)
        target = np.asarray([row["target"] for row in aligned], dtype=np.int64)
        model_metric_scores = {
            "sparse_prefix_low_confidence_max_rate": -model_scores["sparse_prefix_max_rate"],
            "sparse_prefix_low_confidence_margin": -model_scores["sparse_prefix_margin"],
            "sparse_prefix_entropy": model_scores["sparse_prefix_entropy"],
            "sparse_prefix_output_spikes": model_scores["sparse_prefix_output_spikes"],
            "sparse_prefix_nonzero_classes": model_scores["sparse_prefix_nonzero_classes"],
            "sparse_prefix_pred_wrong_oracle_warning": (
                model_scores["sparse_prefix_pred"] != target
            ).astype(np.float64),
        }
        model_rows = metric_table_for_scores(
            model_metric_scores,
            dense_needed,
            dense_correct,
            sparse_correct,
        )
        for row in model_rows:
            row["note"] = (
                "Model-assisted prefix metric. Uses sparse model on first 400 ms; "
                "not a pure input-complexity metric."
            )
        table.extend(model_rows)

    for row in table:
        row["accuracy_gap_vs_dense"] = float(dense_accuracy - row["routed_accuracy"])
        row.setdefault("note", "")

    table = sorted(table, key=lambda row: row["auc"], reverse=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(dense_needed)),
        "prefix_bins": args.prefix_bins,
        "target_for_auc": "dense_needed = dense_correct and not sparse_correct",
        "dense_needed_count": int(dense_needed.sum()),
        "baseline": {
            "dense_accuracy": dense_accuracy,
            "sparse_accuracy": sparse_accuracy,
        },
        "research_families_tested": [
            "event count and density",
            "temporal burstiness and entropy",
            "polarity balance",
            "coarse spatial occupancy",
            "coarse LZC",
            "space-time event-cloud covariance",
            "time-surface / recency summaries",
            "SI/TI-style Sobel and frame-difference video complexity",
            "VCA-lite high-frequency spectral energy",
            "transition-count/run-change metrics",
            "optional sparse-prefix confidence metrics",
        ],
        "table": table,
        "paths": {},
    }
    csv_path, json_path = write_outputs(args, summary, table)
    summary["paths"] = {"csv": csv_path, "json": json_path}
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nDVSGesture router metric search")
    print("=" * 86)
    print(f"Dense accuracy:  {dense_accuracy * 100:.2f}%")
    print(f"Sparse accuracy: {sparse_accuracy * 100:.2f}%")
    print(f"Dense-needed positives: {dense_needed.sum()} / {len(dense_needed)}")
    print("\nTop metrics by AUC:")
    print(f"{'metric':45s} {'AUC':>7s} {'routed':>9s} {'gap':>8s} {'dense%':>8s}")
    for row in table[:20]:
        print(
            f"{row['metric'][:45]:45s} "
            f"{row['auc']:7.3f} "
            f"{row['routed_accuracy'] * 100:8.2f}% "
            f"{row['accuracy_gap_vs_dense'] * 100:7.2f}% "
            f"{row['dense_fraction'] * 100:7.1f}%"
        )
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
