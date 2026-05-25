"""Measured DVSGesture prefix router.

This is the DVS Gesture equivalent of router.py for the SHD experiment. It uses
the snnTorch DVSGesture models and the measured STM32 prefix-router energy file:

    Prefix_Router_Energy/dvsgesture_prefix_router_binary_energy_DVSGesture_T600_dt1ms_p400ms_32x32_binary.txt

Each row in that file is one DVSGesture test sample in deterministic dataset
order:

    energy_J cycles binary_ones route_dense nbytes nbits expected_binary_ones
    expected_route_dense count_prefix_sum count_bins_gt1

For every DVS Gesture test sample, this script:

1. Runs the high-accuracy and sparse snnTorch models with spike recording.
2. Counts model spikes per sample.
3. Converts spikes to energy using the original paper-matched reference model
   energy calibration.
4. Adds the matching measured STM32 router energy row for that sample.
5. Writes per-sample routed energy and a summary JSON.

Example:
    python router_dvsgesture.py \\
      --dense_model_path /workspace/new_test_results/dvsgesture/dense/models/current_new_dense_acc8906_spikes26024.pth \\
      --sparse_model_path /workspace/new_test_results/dvsgesture/new_sparse/models/accepted_new_sparse_acc7188_spikes6314.pth \\
      --prefix_energy_file /workspace/Prefix_Router_Energy/dvsgesture_prefix_router_binary_energy_DVSGesture_T600_dt1ms_p400ms_32x32_binary.txt \\
      --expected_file /workspace/Prefix_Router_Energy/dvsgesture_prefix_router_binary_expected_DVSGesture_T600_dt1ms_p400ms_32x32_binary.csv
"""

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import torch
import tonic
from torch.utils.data import DataLoader

from datasets.dvsgesture_dataset import DVSGestureDataset
from models.dvsgesture_model import DVSGestureSNN


DEFAULT_DENSE_MODEL = (
    "new_test_results/dvsgesture/dense/models/current_new_dense_acc8906_spikes26024.pth"
)
DEFAULT_SPARSE_MODEL = (
    "new_test_results/dvsgesture/new_sparse/models/accepted_new_sparse_acc7188_spikes6314.pth"
)
DEFAULT_PREFIX_ENERGY = (
    "Prefix_Router_Energy/"
    "dvsgesture_prefix_router_binary_energy_DVSGesture_T600_dt1ms_p400ms_32x32_binary.txt"
)
DEFAULT_EXPECTED = (
    "Prefix_Router_Energy/"
    "dvsgesture_prefix_router_binary_expected_DVSGesture_T600_dt1ms_p400ms_32x32_binary.csv"
)
DEFAULT_REFERENCE_AVG_SPIKES = 257304.70075757575
DEFAULT_PAPER_ENERGY_J = 0.459


def load_prefix_router_records(path):
    records = []
    with open(path, "r") as fh:
        for sample_idx, line in enumerate(fh):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 10:
                raise ValueError(
                    f"Expected 10 columns in {path} line {sample_idx + 1}, "
                    f"got {len(parts)}: {line!r}"
                )
            records.append(
                {
                    "sample_idx": sample_idx,
                    "router_energy_J": float(parts[0]),
                    "router_cycles": int(parts[1]),
                    "binary_prefix_ones": int(parts[2]),
                    "route_dense": bool(int(parts[3])),
                    "nbytes": int(parts[4]),
                    "nbits": int(parts[5]),
                    "expected_binary_prefix_ones": int(parts[6]),
                    "expected_route_dense": bool(int(parts[7])),
                    "count_prefix_sum": int(parts[8]),
                    "count_bins_gt1": int(parts[9]),
                }
            )
    print(f"Loaded {len(records)} measured DVS prefix-router rows from {path}")
    return records


def load_expected_records(path):
    if path is None:
        return None
    expected = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            expected.append(
                {
                    "sample_idx": int(row["sample_idx"]),
                    "label": int(row["label"]),
                    "binary_prefix_ones": int(row["binary_prefix_ones"]),
                    "count_prefix_sum": int(row["count_prefix_sum"]),
                    "count_bins_gt1": int(row["count_bins_gt1"]),
                    "route_dense": bool(int(row["route_dense"])),
                }
            )
    print(f"Loaded {len(expected)} expected DVS prefix rows from {path}")
    return expected


def attach_expected_to_records(router_records, expected_records):
    if expected_records is None:
        return router_records
    if len(router_records) != len(expected_records):
        raise ValueError(
            f"Measured rows ({len(router_records)}) and expected rows ({len(expected_records)}) differ"
        )

    combined = []
    for measured, expected in zip(router_records, expected_records):
        if measured["sample_idx"] != expected["sample_idx"]:
            raise ValueError(
                f"Measured/expected sample index mismatch: "
                f"{measured['sample_idx']} vs {expected['sample_idx']}"
            )
        if measured["binary_prefix_ones"] != expected["binary_prefix_ones"]:
            raise ValueError(f"Measured/expected prefix score mismatch at {measured['sample_idx']}")
        if measured["count_prefix_sum"] != expected["count_prefix_sum"]:
            raise ValueError(f"Measured/expected count-sum mismatch at {measured['sample_idx']}")
        if measured["count_bins_gt1"] != expected["count_bins_gt1"]:
            raise ValueError(f"Measured/expected gt1-count mismatch at {measured['sample_idx']}")
        if measured["route_dense"] != expected["route_dense"]:
            raise ValueError(f"Measured/expected route mismatch at {measured['sample_idx']}")

        row = dict(measured)
        row["label"] = expected["label"]
        combined.append(row)
    return combined


def record_key(label, binary_prefix_ones, count_prefix_sum, count_bins_gt1):
    return (
        int(label),
        int(binary_prefix_ones),
        int(count_prefix_sum),
        int(count_bins_gt1),
    )


def make_record_matcher(records):
    matcher = {}
    for record in records:
        if "label" not in record:
            raise ValueError("Expected records with labels for fingerprint matching")
        key = record_key(
            record["label"],
            record["binary_prefix_ones"],
            record["count_prefix_sum"],
            record["count_bins_gt1"],
        )
        matcher.setdefault(key, []).append(record)
    return matcher


def sample_prefix_fingerprint(sample, label, prefix_bins):
    prefix = sample[:prefix_bins]
    binary_prefix_ones = int((prefix > 0).sum().detach().cpu().item())
    count_prefix_sum = int(round(float(prefix.sum().detach().cpu().item())))
    count_bins_gt1 = int((prefix > 1).sum().detach().cpu().item())
    return record_key(label, binary_prefix_ones, count_prefix_sum, count_bins_gt1)


def make_model(model_path, device, fallback_model_type):
    hp = DVSGestureSNN.load_hyperparams(model_path, device=device)
    model = DVSGestureSNN(
        input_size=hp.get("input_size", None),
        n_frames=hp.get("n_frames", 600),
        tau_mem=hp.get("tau_mem", None),
        spike_lam=hp.get("spike_lam", 0.0),
        model_type=hp.get("model_type", fallback_model_type),
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
    model.load_model(model_path)
    model.eval()
    return model, hp


def predict_and_count(model, data):
    output_spikes, recordings = model(data, record=True)
    spike_rates = output_spikes.sum(dim=1) / output_spikes.shape[1]
    preds = spike_rates.argmax(dim=1)

    batch = data.shape[0]
    per_sample_spikes = torch.zeros(batch, device=data.device)
    for spk_tensor in recordings.values():
        per_sample_spikes += spk_tensor.reshape(batch, -1).sum(dim=1)
    return preds, per_sample_spikes


def validate_records(rows, expected_rows, labels):
    if expected_rows is None:
        return
    if len(rows) != len(expected_rows):
        raise ValueError(
            f"Measured rows ({len(rows)}) and expected rows ({len(expected_rows)}) differ"
        )
    if len(labels) != len(expected_rows):
        raise ValueError(
            f"Dataset labels ({len(labels)}) and expected rows ({len(expected_rows)}) differ"
        )
    for i, (measured, expected, label) in enumerate(zip(rows, expected_rows, labels)):
        if measured["sample_idx"] != i or expected["sample_idx"] != i:
            raise ValueError(f"Sample index mismatch at {i}")
        if measured["binary_prefix_ones"] != expected["binary_prefix_ones"]:
            raise ValueError(f"Prefix score mismatch at sample {i}")
        if measured["route_dense"] != expected["route_dense"]:
            raise ValueError(f"Route mismatch at sample {i}")
        if expected["label"] != int(label):
            raise ValueError(
                f"Label order mismatch at sample {i}: expected CSV has "
                f"{expected['label']}, dataset has {int(label)}"
            )
    print("Measured STM32 rows, expected CSV, and dataset labels match.")


def compute_threshold_rows(base_rows, threshold):
    out = []
    for row in base_rows:
        updated = dict(row)
        updated["route_dense"] = row["binary_prefix_ones"] >= threshold
        out.append(updated)
    return out


def summarize_rows(rows, route_threshold):
    dense_correct = np.array([r["dense_correct"] for r in rows], dtype=bool)
    sparse_correct = np.array([r["sparse_correct"] for r in rows], dtype=bool)
    routed_correct = np.array([r["routed_correct"] for r in rows], dtype=bool)
    route_dense = np.array([r["route_dense"] for r in rows], dtype=bool)

    dense_spikes = np.array([r["dense_spikes"] for r in rows], dtype=np.float64)
    sparse_spikes = np.array([r["sparse_spikes"] for r in rows], dtype=np.float64)
    routed_spikes = np.array([r["routed_model_spikes"] for r in rows], dtype=np.float64)

    baseline_dense_energy = np.array(
        [r["baseline_dense_energy_J"] for r in rows], dtype=np.float64
    )
    sparse_only_energy = np.array(
        [r["sparse_only_energy_J"] for r in rows], dtype=np.float64
    )
    routed_model_energy = np.array(
        [r["routed_model_energy_J"] for r in rows], dtype=np.float64
    )
    router_energy = np.array([r["router_energy_J"] for r in rows], dtype=np.float64)
    routed_total_energy = np.array(
        [r["routed_total_energy_J"] for r in rows], dtype=np.float64
    )
    scores = np.array([r["binary_prefix_ones"] for r in rows], dtype=np.float64)
    dense_needed = dense_correct & (~sparse_correct)

    auc_value = None
    if dense_needed.any() and (~dense_needed).any():
        try:
            from sklearn.metrics import roc_auc_score

            auc_value = float(roc_auc_score(dense_needed.astype(np.int64), scores))
        except Exception:
            auc_value = None

    dense_route_correct = dense_correct[route_dense] if route_dense.any() else np.array([])
    sparse_route_correct = sparse_correct[~route_dense] if (~route_dense).any() else np.array([])

    return {
        "n_samples": len(rows),
        "accuracy": {
            "dense": float(dense_correct.mean()),
            "sparse": float(sparse_correct.mean()),
            "routed": float(routed_correct.mean()),
            "dense_route_accuracy": (
                float(dense_route_correct.mean()) if dense_route_correct.size else None
            ),
            "sparse_route_accuracy": (
                float(sparse_route_correct.mean()) if sparse_route_correct.size else None
            ),
            "oracle_best_of_dense_sparse": float((dense_correct | sparse_correct).mean()),
            "both_correct": float((dense_correct & sparse_correct).mean()),
            "both_wrong": float((~dense_correct & ~sparse_correct).mean()),
            "dense_correct_sparse_wrong": float(dense_needed.mean()),
            "sparse_correct_dense_wrong": float((sparse_correct & (~dense_correct)).mean()),
        },
        "route_counts": {
            "dense": int(route_dense.sum()),
            "sparse": int((~route_dense).sum()),
            "dense_fraction": float(route_dense.mean()),
            "sparse_fraction": float((~route_dense).mean()),
        },
        "router_metric": {
            "name": "binary_prefix_occupancy_popcount_400ms",
            "threshold": route_threshold,
            "auc_for_dense_needed": auc_value,
            "binary_prefix_ones_mean": float(scores.mean()),
            "binary_prefix_ones_median": float(np.median(scores)),
            "binary_prefix_ones_min": int(scores.min()),
            "binary_prefix_ones_max": int(scores.max()),
            "count_prefix_sum_mean": float(np.mean([r["count_prefix_sum"] for r in rows])),
            "count_bins_gt1_mean": float(np.mean([r["count_bins_gt1"] for r in rows])),
        },
        "avg_spikes_per_sample": {
            "dense": float(dense_spikes.mean()),
            "sparse": float(sparse_spikes.mean()),
            "routed_model": float(routed_spikes.mean()),
        },
        "energy": {
            "baseline_dense_only_avg_J": float(baseline_dense_energy.mean()),
            "baseline_dense_only_total_J": float(baseline_dense_energy.sum()),
            "sparse_only_avg_J": float(sparse_only_energy.mean()),
            "sparse_only_total_J": float(sparse_only_energy.sum()),
            "routed_model_only_avg_J": float(routed_model_energy.mean()),
            "routed_model_only_total_J": float(routed_model_energy.sum()),
            "router_avg_J": float(router_energy.mean()),
            "router_total_J": float(router_energy.sum()),
            "routed_total_avg_J": float(routed_total_energy.mean()),
            "routed_total_J": float(routed_total_energy.sum()),
            "energy_gain_model_only_J": float(
                baseline_dense_energy.sum() - routed_model_energy.sum()
            ),
            "energy_gain_with_router_J": float(
                baseline_dense_energy.sum() - routed_total_energy.sum()
            ),
            "energy_savings_model_only_percent": float(
                100.0 * (1.0 - routed_model_energy.sum() / baseline_dense_energy.sum())
            ),
            "energy_savings_with_router_percent": float(
                100.0 * (1.0 - routed_total_energy.sum() / baseline_dense_energy.sum())
            ),
            "router_overhead_percent_of_dense_baseline": float(
                100.0 * router_energy.sum() / baseline_dense_energy.sum()
            ),
        },
        "router_cycles": {
            "mean": float(np.mean([r["router_cycles"] for r in rows])),
            "median": float(np.median([r["router_cycles"] for r in rows])),
            "min": int(np.min([r["router_cycles"] for r in rows])),
            "max": int(np.max([r["router_cycles"] for r in rows])),
        },
    }


def summarize_threshold(
    threshold,
    prefix_scores,
    dense_correct,
    sparse_correct,
    dense_energy,
    sparse_energy,
    router_energy,
):
    route_dense = prefix_scores >= threshold
    routed_correct = np.where(route_dense, dense_correct, sparse_correct)
    routed_model_energy = np.where(route_dense, dense_energy, sparse_energy)
    routed_total = routed_model_energy + router_energy
    baseline_total = dense_energy.sum()
    return {
        "threshold": int(threshold),
        "accuracy": float(routed_correct.mean()),
        "dense_fraction": float(route_dense.mean()),
        "avg_energy_J": float(routed_total.mean()),
        "energy_savings_percent": float(100.0 * (1.0 - routed_total.sum() / baseline_total)),
    }


def threshold_sweep(rows):
    prefix_scores = np.array([r["binary_prefix_ones"] for r in rows], dtype=np.int64)
    dense_correct = np.array([r["dense_correct"] for r in rows], dtype=bool)
    sparse_correct = np.array([r["sparse_correct"] for r in rows], dtype=bool)
    dense_energy = np.array([r["baseline_dense_energy_J"] for r in rows], dtype=np.float64)
    sparse_energy = np.array([r["sparse_only_energy_J"] for r in rows], dtype=np.float64)
    router_energy = np.array([r["router_energy_J"] for r in rows], dtype=np.float64)

    thresholds = sorted(set(prefix_scores.tolist() + [int(prefix_scores.min()) - 1, int(prefix_scores.max()) + 1]))
    sweep = [
        summarize_threshold(
            th,
            prefix_scores,
            dense_correct,
            sparse_correct,
            dense_energy,
            sparse_energy,
            router_energy,
        )
        for th in thresholds
    ]

    best_accuracy = max(sweep, key=lambda x: (x["accuracy"], x["energy_savings_percent"]))
    best_energy_above_80 = max(
        [x for x in sweep if x["accuracy"] >= 0.80],
        key=lambda x: x["energy_savings_percent"],
        default=None,
    )
    best_energy_above_85 = max(
        [x for x in sweep if x["accuracy"] >= 0.85],
        key=lambda x: x["energy_savings_percent"],
        default=None,
    )
    best_energy_above_sparse = max(
        [x for x in sweep if x["accuracy"] >= float(sparse_correct.mean())],
        key=lambda x: x["energy_savings_percent"],
        default=None,
    )

    return {
        "best_accuracy": best_accuracy,
        "best_energy_at_accuracy_ge_80": best_energy_above_80,
        "best_energy_at_accuracy_ge_85": best_energy_above_85,
        "best_energy_at_accuracy_ge_sparse_accuracy": best_energy_above_sparse,
    }


def write_outputs(rows, summary, args, dense_hp, sparse_hp):
    out_dir = os.path.join(args.output_path, "dvsgesture", "prefix_router_eval")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"dvsgesture_prefix_router_per_sample_{stamp}.csv")
    json_path = os.path.join(out_dir, f"dvsgesture_prefix_router_summary_{stamp}.json")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        **summary,
        "timestamp": datetime.now().isoformat(),
        "paths": {
            "dense_model_path": args.dense_model_path,
            "sparse_model_path": args.sparse_model_path,
            "prefix_router_energy_file": args.prefix_energy_file,
            "expected_file": args.expected_file,
            "per_sample_csv": csv_path,
            "summary_json": json_path,
        },
        "calibration": {
            "paper_energy_J": args.paper_energy_J,
            "reference_avg_spikes_per_sample": args.reference_avg_spikes_per_sample,
            "energy_per_spike_J": (
                args.paper_energy_J / args.reference_avg_spikes_per_sample
            ),
        },
        "hyperparams": {
            "dense": dense_hp,
            "sparse": sparse_hp,
        },
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    return csv_path, json_path


def run_dvsgesture_prefix_router(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    dense_model, dense_hp = make_model(args.dense_model_path, device, "dense")
    sparse_model, sparse_hp = make_model(args.sparse_model_path, device, "new_sparse")

    data = DVSGestureDataset(
        dataset_path=args.dataset_path,
        w=args.w,
        h=args.h,
        max_timesteps=args.max_timesteps,
        binarize=args.binarize,
        denoise_filter_time=args.denoise_filter_time,
        cache_tag=args.cache_tag,
    )
    _, cached_test = data.load_dvsgesture()
    test_loader = DataLoader(
        cached_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True),
    )

    router_records = load_prefix_router_records(args.prefix_energy_file)
    expected_records = load_expected_records(args.expected_file)
    router_records = attach_expected_to_records(router_records, expected_records)
    if len(router_records) != len(cached_test):
        raise ValueError(
            f"Router rows ({len(router_records)}) do not match test set ({len(cached_test)})"
        )
    record_matcher = make_record_matcher(router_records) if expected_records else None
    prefix_bins = int(round((args.prefix_ms / 1000.0) / args.net_dt))

    energy_per_spike = args.paper_energy_J / args.reference_avg_spikes_per_sample
    rows = []
    labels_seen = []
    sample_idx = 0

    dense_model.eval()
    sparse_model.eval()
    with torch.no_grad():
        for batch_idx, (batch_data, targets) in enumerate(test_loader):
            batch_data = batch_data.to(device).float()
            targets = targets.to(device)

            dense_preds, dense_spikes = predict_and_count(dense_model, batch_data)
            sparse_preds, sparse_spikes = predict_and_count(sparse_model, batch_data)

            for i in range(targets.shape[0]):
                if args.threshold is not None:
                    # The measured row is still used for energy/cycles; only the route
                    # decision is overridden for threshold sweeps.
                    pass
                target = int(targets[i].detach().cpu().item())
                if record_matcher is not None:
                    key = sample_prefix_fingerprint(batch_data[i], target, prefix_bins)
                    matches = record_matcher.get(key)
                    if not matches:
                        raise ValueError(
                            "Could not match dataset sample to measured STM32 row. "
                            f"eval_sample_idx={sample_idx}, key={key}"
                        )
                    record = matches.pop(0)
                else:
                    record = router_records[sample_idx]

                if args.threshold is not None:
                    route_dense = record["binary_prefix_ones"] >= args.threshold
                else:
                    route_dense = record["route_dense"]

                labels_seen.append(target)
                dense_pred = int(dense_preds[i].detach().cpu().item())
                sparse_pred = int(sparse_preds[i].detach().cpu().item())
                dense_spike_count = float(dense_spikes[i].detach().cpu().item())
                sparse_spike_count = float(sparse_spikes[i].detach().cpu().item())
                dense_energy = dense_spike_count * energy_per_spike
                sparse_energy = sparse_spike_count * energy_per_spike

                routed_pred = dense_pred if route_dense else sparse_pred
                routed_spikes = dense_spike_count if route_dense else sparse_spike_count
                routed_model_energy = dense_energy if route_dense else sparse_energy
                routed_total_energy = routed_model_energy + record["router_energy_J"]

                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "stm32_sample_idx": record["sample_idx"],
                        "target": target,
                        "dense_pred": dense_pred,
                        "sparse_pred": sparse_pred,
                        "routed_pred": routed_pred,
                        "route_dense": int(route_dense),
                        "dense_correct": int(dense_pred == target),
                        "sparse_correct": int(sparse_pred == target),
                        "routed_correct": int(routed_pred == target),
                        "binary_prefix_ones": record["binary_prefix_ones"],
                        "count_prefix_sum": record["count_prefix_sum"],
                        "count_bins_gt1": record["count_bins_gt1"],
                        "router_cycles": record["router_cycles"],
                        "router_energy_J": record["router_energy_J"],
                        "dense_spikes": dense_spike_count,
                        "sparse_spikes": sparse_spike_count,
                        "routed_model_spikes": routed_spikes,
                        "baseline_dense_energy_J": dense_energy,
                        "sparse_only_energy_J": sparse_energy,
                        "routed_model_energy_J": routed_model_energy,
                        "routed_total_energy_J": routed_total_energy,
                    }
                )
                sample_idx += 1

            if (batch_idx + 1) % args.progress_every == 0:
                print(f"Processed {sample_idx}/{len(router_records)} samples")

    if record_matcher is None:
        validate_records(router_records, expected_records, labels_seen)
    else:
        unused = sum(len(v) for v in record_matcher.values())
        if unused != 0:
            raise ValueError(f"{unused} measured STM32 rows were not matched to dataset samples")
        print("Matched all STM32 rows to pod dataset samples by prefix fingerprint.")

    route_threshold = args.threshold if args.threshold is not None else args.measured_threshold
    summary = summarize_rows(rows, route_threshold)
    summary["threshold_sweep"] = threshold_sweep(rows)
    csv_path, json_path = write_outputs(rows, summary, args, dense_hp, sparse_hp)

    print("\nDVSGesture measured prefix-router summary")
    print("=" * 72)
    print(f"Samples: {summary['n_samples']}")
    print(f"Dense accuracy:  {summary['accuracy']['dense'] * 100:.2f}%")
    print(f"Sparse accuracy: {summary['accuracy']['sparse'] * 100:.2f}%")
    print(f"Routed accuracy: {summary['accuracy']['routed'] * 100:.2f}%")
    print(
        "Route split: "
        f"{summary['route_counts']['dense']} dense "
        f"({summary['route_counts']['dense_fraction'] * 100:.1f}%), "
        f"{summary['route_counts']['sparse']} sparse "
        f"({summary['route_counts']['sparse_fraction'] * 100:.1f}%)"
    )
    print(f"Dense avg spikes/sample:  {summary['avg_spikes_per_sample']['dense']:.2f}")
    print(f"Sparse avg spikes/sample: {summary['avg_spikes_per_sample']['sparse']:.2f}")
    print(f"Routed avg model spikes/sample: {summary['avg_spikes_per_sample']['routed_model']:.2f}")
    print(f"Baseline dense energy/sample: {summary['energy']['baseline_dense_only_avg_J'] * 1000:.3f} mJ")
    print(f"Sparse-only energy/sample:    {summary['energy']['sparse_only_avg_J'] * 1000:.3f} mJ")
    print(f"Router energy/sample:         {summary['energy']['router_avg_J'] * 1e6:.3f} uJ")
    print(f"Routed total energy/sample:   {summary['energy']['routed_total_avg_J'] * 1000:.3f} mJ")
    print(
        "Energy savings with router:  "
        f"{summary['energy']['energy_savings_with_router_percent']:.2f}%"
    )
    print(f"Prefix metric AUC for dense-needed cases: {summary['router_metric']['auc_for_dense_needed']}")
    print(f"Per-sample CSV: {csv_path}")
    print(f"Summary JSON:   {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run measured DVSGesture prefix router")
    parser.add_argument("--dense_model_path", type=str, default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--sparse_model_path", type=str, default=DEFAULT_SPARSE_MODEL)
    parser.add_argument("--prefix_energy_file", type=str, default=DEFAULT_PREFIX_ENERGY)
    parser.add_argument("--expected_file", type=str, default=DEFAULT_EXPECTED)
    parser.add_argument("--dataset_path", type=str, default="./data")
    parser.add_argument("--output_path", type=str, default="new_test_results")
    parser.add_argument("--w", type=int, default=32)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--max_timesteps", type=int, default=600)
    parser.add_argument("--prefix_ms", type=int, default=400)
    parser.add_argument("--net_dt", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--denoise_filter_time", type=int, default=10000)
    parser.add_argument("--cache_tag", type=str, default=None)
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Override measured route decisions with binary_prefix_ones >= threshold",
    )
    parser.add_argument(
        "--measured_threshold",
        type=int,
        default=10000,
        help="Threshold used when the STM32 measurement file was generated",
    )
    parser.add_argument("--paper_energy_J", type=float, default=DEFAULT_PAPER_ENERGY_J)
    parser.add_argument(
        "--reference_avg_spikes_per_sample",
        type=float,
        default=DEFAULT_REFERENCE_AVG_SPIKES,
    )
    parser.add_argument("--progress_every", type=int, default=2)
    args = parser.parse_args()

    run_dvsgesture_prefix_router(args)


if __name__ == "__main__":
    main()
