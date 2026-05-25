"""Evaluate non-learned SHD router metrics for dense-vs-sparse routing.

This script asks a narrow question:

    Can a cheap, MCU-portable scalar metric separate samples where the dense
    SHD model is needed from samples where the sparse model is good enough?

The positive label is:

    dense_needed = dense_correct and not sparse_correct

All metrics are deterministic input-spike descriptors or legacy precomputed
router values. No router model is trained here.
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import tonic
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

try:
    from lempel_ziv_complexity import lempel_ziv_complexity
except Exception:  # pragma: no cover - optional dependency
    lempel_ziv_complexity = None

from datasets.shd_dataset import SHDDataset
from models.shd_model_paper import SHDSNN


METRIC_METADATA = {
    "total_spikes": {
        "family": "spike_count",
        "stm32_port": "integer accumulation over input bits/events",
        "energy_status": "requires STM32 measurement",
    },
    "active_frames": {
        "family": "temporal_activity",
        "stm32_port": "integer count of nonzero time bins",
        "energy_status": "requires STM32 measurement",
    },
    "active_channels": {
        "family": "channel_activity",
        "stm32_port": "integer count of channels with at least one spike",
        "energy_status": "requires STM32 measurement",
    },
    "duration_bins": {
        "family": "temporal_extent",
        "stm32_port": "track first/last nonzero frame",
        "energy_status": "requires STM32 measurement",
    },
    "max_frame_spikes": {
        "family": "burstiness",
        "stm32_port": "integer max over frame spike counts",
        "energy_status": "requires STM32 measurement",
    },
    "frame_count_var": {
        "family": "burstiness",
        "stm32_port": "integer sum and sum-of-squares; no sqrt needed",
        "energy_status": "requires STM32 measurement",
    },
    "frame_fano": {
        "family": "variability",
        "stm32_port": "integer sum and sum-of-squares plus one division",
        "energy_status": "requires STM32 measurement",
    },
    "temporal_hhi": {
        "family": "concentration",
        "stm32_port": "sum frame_count^2 / total^2; no logs",
        "energy_status": "requires STM32 measurement",
    },
    "channel_hhi": {
        "family": "concentration",
        "stm32_port": "sum channel_count^2 / total^2; no logs",
        "energy_status": "requires STM32 measurement",
    },
    "temporal_entropy": {
        "family": "entropy",
        "stm32_port": "histogram entropy; use fixed-point or LUT log2 on STM32",
        "energy_status": "requires STM32 measurement",
    },
    "channel_entropy": {
        "family": "entropy",
        "stm32_port": "histogram entropy; use fixed-point or LUT log2 on STM32",
        "energy_status": "requires STM32 measurement",
    },
    "temporal_centroid": {
        "family": "temporal_distribution",
        "stm32_port": "weighted integer sum of time-bin index by frame spike count",
        "energy_status": "requires STM32 measurement",
    },
    "temporal_spread": {
        "family": "temporal_distribution",
        "stm32_port": "weighted integer sum and sum-of-squares over time-bin index",
        "energy_status": "requires STM32 measurement",
    },
    "channel_centroid": {
        "family": "spectral_distribution",
        "stm32_port": "weighted integer sum of channel index by channel spike count",
        "energy_status": "requires STM32 measurement",
    },
    "channel_spread": {
        "family": "spectral_distribution",
        "stm32_port": "weighted integer sum and sum-of-squares over channel index",
        "energy_status": "requires STM32 measurement",
    },
    "temporal_total_variation": {
        "family": "transition_count",
        "stm32_port": "sum abs(frame_count[t] - frame_count[t-1])",
        "energy_status": "requires STM32 measurement",
    },
    "channel_total_variation": {
        "family": "spectral_roughness",
        "stm32_port": "sum abs(channel_count[c] - channel_count[c-1])",
        "energy_status": "requires STM32 measurement",
    },
    "temporal_gini": {
        "family": "concentration",
        "stm32_port": "sort-free Gini approximation is possible; exact version requires sorting",
        "energy_status": "requires STM32 measurement",
    },
    "channel_gini": {
        "family": "concentration",
        "stm32_port": "sort-free Gini approximation is possible; exact version requires sorting",
        "energy_status": "requires STM32 measurement",
    },
    "active_frame_transitions": {
        "family": "transition_count",
        "stm32_port": "count binary changes in active/inactive time bins",
        "energy_status": "requires STM32 measurement",
    },
    "longest_silence": {
        "family": "burstiness",
        "stm32_port": "single pass run-length count over active frames",
        "energy_status": "requires STM32 measurement",
    },
    "burst_count": {
        "family": "burstiness",
        "stm32_port": "count inactive-to-active transitions",
        "energy_status": "requires STM32 measurement",
    },
    "channel_span": {
        "family": "spectral_extent",
        "stm32_port": "track min/max active channel",
        "energy_status": "requires STM32 measurement",
    },
    "early_late_ratio": {
        "family": "temporal_distribution",
        "stm32_port": "two integer accumulators and one division",
        "energy_status": "requires STM32 measurement",
    },
    "legacy_lzc_score": {
        "family": "legacy_lzc",
        "stm32_port": "already implemented/measured previously, but may be stale for current cache",
        "energy_status": "measured if legacy LZC energy file matches preprocessing",
    },
}


def load_lzc_records(path):
    if not path:
        return []
    records = []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 3:
                records.append(
                    {
                        "energy_J": float(parts[0]),
                        "cycles": int(parts[1]),
                        "lzc_score": float(parts[2]),
                    }
                )
            elif len(parts) == 2:
                records.append(
                    {
                        "energy_J": float(parts[0]),
                        "cycles": 0,
                        "lzc_score": float(parts[1]),
                    }
                )
    return records


def make_model(model_path, device):
    hp = SHDSNN.load_hyperparams(model_path, device=device)
    model = SHDSNN(
        input_size=hp.get("input_size", 700),
        n_frames=hp.get("n_frames", 1400),
        tau_mem=hp.get("tau_mem", 0.02),
        tau_syn=hp.get("tau_syn", 0.005),
        spike_lam=hp.get("spike_lam", 0.0),
        model_type=hp.get("model_type", "baseline"),
        device=device,
        num_classes=hp.get("num_classes", 20),
        lr=hp.get("lr", 0.001),
        dt=hp.get("dt", 0.001),
        threshold=hp.get("threshold", 1.0),
        has_bias=hp.get("has_bias", True),
        hidden_size=hp.get("hidden_size", 512),
        rate_lam=hp.get("rate_lam", 0.001),
        target_rate=hp.get("target_rate", 14.0),
        delay_mode=hp.get("delay_mode", "none"),
        max_delay_steps=hp.get("max_delay_steps", 62),
    )
    model.load_model(model_path)
    model.net.eval()
    return model, hp


def predict_and_count(model, data):
    output, _, recording = model.net(data, record=True)
    logits = output.mean(dim=1)
    preds = logits.argmax(dim=1)

    batch = data.shape[0]
    per_sample_spikes = torch.zeros(batch, device=data.device)
    for value in recording.values():
        if isinstance(value, dict) and "spikes" in value:
            spikes = value["spikes"]
            per_sample_spikes += spikes.reshape(batch, -1).sum(dim=1)
    return preds, per_sample_spikes


def entropy_from_counts(counts, total, eps=1e-12):
    if total <= 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / float(total)
    return float(-(probs * np.log2(probs + eps)).sum())


def weighted_centroid_and_spread(counts):
    total = float(counts.sum())
    if total <= 0:
        return 0.0, 0.0
    idx = np.arange(len(counts), dtype=np.float64)
    centroid = float((idx * counts).sum() / total)
    spread = float((((idx - centroid) ** 2) * counts).sum() / total)
    return centroid, spread


def gini_from_counts(counts):
    # Exact Gini is useful for offline ranking. If it wins, use an STM32
    # approximation or benchmark the sort cost separately.
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return 0.0
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    sorted_counts = np.sort(counts)
    idx = np.arange(1, counts.size + 1, dtype=np.float64)
    return float((2.0 * (idx * sorted_counts).sum()) / (counts.size * total) - (counts.size + 1.0) / counts.size)


def longest_false_run(active):
    best = 0
    cur = 0
    for val in active:
        if val:
            if cur > best:
                best = cur
            cur = 0
        else:
            cur += 1
    return max(best, cur)


def lzc_binary_array(arr):
    if lempel_ziv_complexity is None:
        return math.nan
    bits = "".join("1" if v else "0" for v in arr.astype(np.uint8).ravel())
    return float(lempel_ziv_complexity(bits))


def compute_metrics_for_sample(x, dt, prefix_bins, include_lzc):
    # x shape: [T, C], current SHD cache is binary raster, but use >0 defensively.
    xb = x > 0
    frame_counts = xb.sum(axis=1).astype(np.float64)
    channel_counts = xb.sum(axis=0).astype(np.float64)
    active_frames = frame_counts > 0
    active_channels = channel_counts > 0
    total = float(frame_counts.sum())
    t_len = xb.shape[0]

    metrics = {}
    metrics["total_spikes"] = total
    metrics["active_frames"] = float(active_frames.sum())
    metrics["active_channels"] = float(active_channels.sum())
    metrics["spike_density"] = total / float(xb.size)
    metrics["mean_frame_spikes"] = float(frame_counts.mean())
    metrics["max_frame_spikes"] = float(frame_counts.max()) if t_len else 0.0
    metrics["frame_count_var"] = float(frame_counts.var()) if t_len else 0.0
    metrics["frame_fano"] = float(frame_counts.var() / (frame_counts.mean() + 1e-12))
    metrics["temporal_total_variation"] = float(np.abs(np.diff(frame_counts)).sum()) if t_len > 1 else 0.0
    metrics["channel_total_variation"] = (
        float(np.abs(np.diff(channel_counts)).sum()) if channel_counts.size > 1 else 0.0
    )
    metrics["active_frame_transitions"] = (
        float(np.abs(np.diff(active_frames.astype(np.int8))).sum()) if t_len > 1 else 0.0
    )
    metrics["burst_count"] = float(np.logical_and(active_frames, np.r_[True, ~active_frames[:-1]]).sum())
    metrics["longest_silence"] = float(longest_false_run(active_frames))

    if active_frames.any():
        nonzero = np.flatnonzero(active_frames)
        first = int(nonzero[0])
        last = int(nonzero[-1])
        metrics["first_active_bin"] = float(first)
        metrics["last_active_bin"] = float(last)
        metrics["duration_bins"] = float(last - first + 1)
    else:
        metrics["first_active_bin"] = 0.0
        metrics["last_active_bin"] = 0.0
        metrics["duration_bins"] = 0.0

    if active_channels.any():
        chans = np.flatnonzero(active_channels)
        metrics["channel_span"] = float(int(chans[-1]) - int(chans[0]) + 1)
    else:
        metrics["channel_span"] = 0.0

    denom = total * total + 1e-12
    metrics["temporal_hhi"] = float((frame_counts * frame_counts).sum() / denom)
    metrics["channel_hhi"] = float((channel_counts * channel_counts).sum() / denom)
    metrics["temporal_entropy"] = entropy_from_counts(frame_counts, total)
    metrics["channel_entropy"] = entropy_from_counts(channel_counts, total)
    metrics["temporal_gini"] = gini_from_counts(frame_counts)
    metrics["channel_gini"] = gini_from_counts(channel_counts)
    metrics["temporal_centroid"], metrics["temporal_spread"] = weighted_centroid_and_spread(frame_counts)
    metrics["channel_centroid"], metrics["channel_spread"] = weighted_centroid_and_spread(channel_counts)

    half = t_len // 2
    early = float(frame_counts[:half].sum())
    late = float(frame_counts[half:].sum())
    metrics["early_late_ratio"] = early / (late + 1.0)
    metrics["early_fraction"] = early / (total + 1.0)
    metrics["late_fraction"] = late / (total + 1.0)
    lower = float(channel_counts[: channel_counts.size // 2].sum())
    upper = float(channel_counts[channel_counts.size // 2 :].sum())
    metrics["low_high_channel_ratio"] = lower / (upper + 1.0)
    metrics["high_channel_fraction"] = upper / (total + 1.0)

    for p in prefix_bins:
        p = min(p, t_len)
        prefix = xb[:p]
        prefix_frame_counts = frame_counts[:p]
        name = f"prefix_{p}ms"
        metrics[f"{name}_spikes"] = float(prefix.sum())
        metrics[f"{name}_spike_fraction"] = float(prefix.sum()) / (total + 1.0)
        metrics[f"{name}_active_frames"] = float((prefix_frame_counts > 0).sum())
        metrics[f"{name}_max_frame_spikes"] = float(prefix_frame_counts.max()) if p > 0 else 0.0
        metrics[f"{name}_temporal_tv"] = (
            float(np.abs(np.diff(prefix_frame_counts)).sum()) if p > 1 else 0.0
        )
        if include_lzc and p <= 100:
            metrics[f"{name}_time_lzc"] = lzc_binary_array(prefix_frame_counts > 0)
            # 700 * 100 = 70k bits; still feasible, and a plausible early-LZC router.
            metrics[f"{name}_flat_lzc"] = lzc_binary_array(prefix)

    if include_lzc:
        metrics["time_active_lzc"] = lzc_binary_array(active_frames)
        metrics["channel_active_lzc"] = lzc_binary_array(active_channels)

    return metrics


def gmean_threshold(y_true, values):
    fpr, tpr, thresholds = roc_curve(y_true, values)
    gmean = np.sqrt(tpr * (1.0 - fpr))
    idx = int(np.argmax(gmean))
    return float(thresholds[idx]), float(gmean[idx])


def evaluate_metric(name, values, labels, dense_correct, sparse_correct, dense_spikes, sparse_spikes, energy_per_spike):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if mask.sum() != len(values) or len(np.unique(values[mask])) < 2:
        return None

    raw_auc = roc_auc_score(labels[mask], values[mask])
    if raw_auc >= 0.5:
        score = values
        direction = "higher_routes_dense"
        oriented_auc = raw_auc
    else:
        score = -values
        direction = "lower_routes_dense"
        oriented_auc = 1.0 - raw_auc

    threshold, gmean = gmean_threshold(labels, score)
    route_dense = score >= threshold
    routed_correct = np.where(route_dense, dense_correct, sparse_correct)
    routed_spikes = np.where(route_dense, dense_spikes, sparse_spikes)
    baseline_energy = dense_spikes * energy_per_spike
    routed_model_energy = routed_spikes * energy_per_spike

    out = {
        "metric": name,
        "family": METRIC_METADATA.get(name, {}).get("family", "derived"),
        "direction": direction,
        "auc": float(oriented_auc),
        "raw_auc": float(raw_auc),
        "threshold_on_oriented_score": threshold,
        "gmean": gmean,
        "route_dense_fraction": float(route_dense.mean()),
        "route_dense_count": int(route_dense.sum()),
        "route_sparse_count": int((~route_dense).sum()),
        "routed_accuracy": float(routed_correct.mean()),
        "routed_avg_spikes": float(routed_spikes.mean()),
        "model_only_avg_energy_J": float(routed_model_energy.mean()),
        "model_only_energy_savings_percent": float(
            100.0 * (1.0 - routed_model_energy.sum() / baseline_energy.sum())
        ),
        "stm32_port": METRIC_METADATA.get(name, {}).get("stm32_port", "simple scalar feature; inspect code"),
        "energy_status": METRIC_METADATA.get(name, {}).get("energy_status", "requires STM32 measurement"),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate SHD input complexity metrics.")
    parser.add_argument("--dense_model_path", required=True)
    parser.add_argument("--sparse_model_path", required=True)
    parser.add_argument("--dataset_path", default="./data")
    parser.add_argument("--output_path", default="./new_test_results")
    parser.add_argument("--lzc_file", default="LZC_Energy/lzc_energy_SHD.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_frames", type=int, default=1400)
    parser.add_argument("--net_dt", type=float, default=0.001)
    parser.add_argument("--paper_energy_mJ", type=float, default=0.42)
    parser.add_argument("--prefix_ms", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    parser.add_argument("--include_lzc", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    dense_model, dense_hp = make_model(args.dense_model_path, device)
    sparse_model, sparse_hp = make_model(args.sparse_model_path, device)

    data = SHDDataset(
        dataset_path=args.dataset_path,
        NUM_CHANNELS=700,
        NUM_POLARITIES=1,
        n_frames=args.n_frames,
        net_dt=args.net_dt,
        use_polarity=False,
    )
    _, cached_test = data.load_shd()
    test_loader = DataLoader(
        cached_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=tonic.collation.PadTensors(batch_first=True),
    )

    lzc_records = load_lzc_records(args.lzc_file) if args.lzc_file else []
    prefix_bins = [int(round(ms / 1000.0 / args.net_dt)) for ms in args.prefix_ms]

    rows = []
    idx = 0
    with torch.no_grad():
        for batch_data, labels in test_loader:
            batch_data = batch_data.to(device).float()
            labels = labels.to(device)
            dense_preds, dense_spikes = predict_and_count(dense_model, batch_data)
            sparse_preds, sparse_spikes = predict_and_count(sparse_model, batch_data)

            batch_np = batch_data.detach().cpu().numpy()
            for i in range(batch_np.shape[0]):
                if args.max_samples is not None and idx >= args.max_samples:
                    break
                metrics = compute_metrics_for_sample(
                    batch_np[i],
                    args.net_dt,
                    prefix_bins=prefix_bins,
                    include_lzc=args.include_lzc,
                )
                if idx < len(lzc_records):
                    metrics["legacy_lzc_score"] = lzc_records[idx]["lzc_score"]
                    metrics["legacy_lzc_energy_J"] = lzc_records[idx]["energy_J"]
                    metrics["legacy_lzc_cycles"] = lzc_records[idx]["cycles"]

                label = int(labels[i].item())
                d_pred = int(dense_preds[i].item())
                s_pred = int(sparse_preds[i].item())
                row = {
                    "sample_idx": idx,
                    "label": label,
                    "dense_pred": d_pred,
                    "sparse_pred": s_pred,
                    "dense_correct": d_pred == label,
                    "sparse_correct": s_pred == label,
                    "dense_needed": (d_pred == label) and (s_pred != label),
                    "dense_spikes": float(dense_spikes[i].item()),
                    "sparse_spikes": float(sparse_spikes[i].item()),
                }
                row.update(metrics)
                rows.append(row)
                idx += 1
            print(f"Processed {idx}/{len(cached_test)} samples")
            if args.max_samples is not None and idx >= args.max_samples:
                break

    dense_correct = np.array([r["dense_correct"] for r in rows], dtype=bool)
    sparse_correct = np.array([r["sparse_correct"] for r in rows], dtype=bool)
    dense_needed = np.array([r["dense_needed"] for r in rows], dtype=np.int64)
    dense_spikes = np.array([r["dense_spikes"] for r in rows], dtype=np.float64)
    sparse_spikes = np.array([r["sparse_spikes"] for r in rows], dtype=np.float64)
    energy_per_spike = (args.paper_energy_mJ * 1e-3) / dense_spikes.mean()

    metric_names = [
        k
        for k in rows[0].keys()
        if k
        not in {
            "sample_idx",
            "label",
            "dense_pred",
            "sparse_pred",
            "dense_correct",
            "sparse_correct",
            "dense_needed",
            "dense_spikes",
            "sparse_spikes",
            "legacy_lzc_energy_J",
            "legacy_lzc_cycles",
        }
    ]

    metric_results = []
    if dense_needed.sum() == 0 or dense_needed.sum() == len(dense_needed):
        print("Dense-needed labels contain one class only; AUC cannot be computed.")
    else:
        for name in metric_names:
            result = evaluate_metric(
                name,
                [r[name] for r in rows],
                dense_needed,
                dense_correct,
                sparse_correct,
                dense_spikes,
                sparse_spikes,
                energy_per_spike,
            )
            if result is not None:
                metric_results.append(result)
    metric_results.sort(key=lambda r: (r["auc"], r["routed_accuracy"]), reverse=True)

    baseline_energy = dense_spikes * energy_per_spike
    sparse_energy = sparse_spikes * energy_per_spike
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(rows),
        "labels": {
            "dense_needed_positive_count": int(dense_needed.sum()),
            "dense_needed_positive_fraction": float(dense_needed.mean()),
            "dense_correct_sparse_wrong_definition": "dense_correct and not sparse_correct",
        },
        "model_metrics": {
            "dense_accuracy": float(dense_correct.mean()),
            "sparse_accuracy": float(sparse_correct.mean()),
            "dense_avg_spikes": float(dense_spikes.mean()),
            "sparse_avg_spikes": float(sparse_spikes.mean()),
            "energy_per_spike_J": float(energy_per_spike),
            "dense_baseline_avg_energy_J": float(baseline_energy.mean()),
            "sparse_only_avg_energy_J": float(sparse_energy.mean()),
            "sparse_only_model_energy_savings_percent": float(
                100.0 * (1.0 - sparse_energy.sum() / baseline_energy.sum())
            ),
        },
        "metric_results": metric_results,
        "top_metrics": metric_results[:20],
        "paths": {
            "dense_model_path": args.dense_model_path,
            "sparse_model_path": args.sparse_model_path,
            "lzc_file": args.lzc_file,
        },
        "notes": [
            "AUC is oriented so values above 0.5 indicate separability regardless of whether high or low raw metric routes dense.",
            "Router energy for new metrics is not estimated here; each candidate needs STM32 measurement before total-energy claims.",
            "Legacy LZC rows may not match the current 700-channel, 1ms, T1400 cache unless regenerated.",
        ],
        "hyperparams": {"dense": dense_hp, "sparse": sparse_hp},
    }

    out_dir = os.path.join(args.output_path, "shd", "complexity_metrics")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"shd_complexity_metrics_{stamp}.json")
    csv_path = os.path.join(out_dir, f"shd_complexity_metrics_{stamp}.csv")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("SHD INPUT COMPLEXITY METRIC EVALUATION")
    print("=" * 78)
    print(f"Samples: {len(rows)}")
    print(f"Dense accuracy:  {dense_correct.mean()*100:.2f}%")
    print(f"Sparse accuracy: {sparse_correct.mean()*100:.2f}%")
    print(
        f"Dense-needed positives: {dense_needed.sum()} "
        f"({dense_needed.mean()*100:.2f}% of samples)"
    )
    print(f"Dense spikes/sample:  {dense_spikes.mean():.2f}")
    print(f"Sparse spikes/sample: {sparse_spikes.mean():.2f}")
    print("\nTop metrics:")
    for result in metric_results[:15]:
        print(
            f"  {result['metric']:<32} AUC={result['auc']:.4f} "
            f"dir={result['direction']:<20} routed_acc={result['routed_accuracy']*100:.2f}% "
            f"dense_route={result['route_dense_fraction']*100:.1f}% "
            f"model_savings={result['model_only_energy_savings_percent']:.2f}%"
        )
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved per-sample CSV: {csv_path}")


if __name__ == "__main__":
    main()
