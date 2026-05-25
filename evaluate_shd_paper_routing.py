"""Evaluate paper-matched SHD dense/sparse models and LZC routing metrics."""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import tonic
from scipy.stats import entropy
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from datasets.shd_dataset import SHDDataset
from models.shd_model_paper import SHDSNN


def load_lzc_records(path):
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

    return preds, logits, per_sample_spikes


def optimal_lzc_threshold(records):
    y_true = np.array(
        [1 if (r["dense_correct"] and not r["sparse_correct"]) else 0 for r in records],
        dtype=np.int64,
    )
    lzc = np.array([r["lzc_score"] for r in records], dtype=np.float64)

    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return {
            "threshold": float(np.median(lzc)),
            "roc_auc": None,
            "positives": positives,
            "negatives": negatives,
            "note": "ROC undefined because only one class was present for dense-needed labels.",
        }

    fpr, tpr, thresholds = roc_curve(y_true, lzc)
    roc_auc = auc(fpr, tpr)
    gmean = np.sqrt(tpr * (1.0 - fpr))
    idx = int(np.argmax(gmean))
    return {
        "threshold": float(thresholds[idx]),
        "roc_auc": float(roc_auc),
        "gmean": float(gmean[idx]),
        "positives": positives,
        "negatives": negatives,
    }


def summarize(records, threshold, paper_energy_j):
    dense_correct = np.array([r["dense_correct"] for r in records], dtype=bool)
    sparse_correct = np.array([r["sparse_correct"] for r in records], dtype=bool)
    dense_spikes = np.array([r["dense_spikes"] for r in records], dtype=np.float64)
    sparse_spikes = np.array([r["sparse_spikes"] for r in records], dtype=np.float64)
    lzc_energy = np.array([r["lzc_energy_J"] for r in records], dtype=np.float64)
    lzc_scores = np.array([r["lzc_score"] for r in records], dtype=np.float64)

    dense_avg_spikes = float(dense_spikes.mean())
    sparse_avg_spikes = float(sparse_spikes.mean())
    energy_per_spike_j = paper_energy_j / dense_avg_spikes

    route_dense = lzc_scores >= threshold
    route_sparse = ~route_dense
    routed_correct = np.where(route_dense, dense_correct, sparse_correct)
    routed_spikes = np.where(route_dense, dense_spikes, sparse_spikes)

    baseline_model_energy = dense_spikes * energy_per_spike_j
    sparse_model_energy = sparse_spikes * energy_per_spike_j
    routed_model_energy = routed_spikes * energy_per_spike_j
    routed_total_energy = routed_model_energy + lzc_energy

    dense_route_correct = dense_correct[route_dense] if route_dense.any() else np.array([])
    sparse_route_correct = sparse_correct[route_sparse] if route_sparse.any() else np.array([])

    return {
        "n_samples": len(records),
        "dense_accuracy": float(dense_correct.mean()),
        "sparse_accuracy": float(sparse_correct.mean()),
        "routed_accuracy": float(routed_correct.mean()),
        "dense_routed_accuracy": float(dense_route_correct.mean()) if dense_route_correct.size else None,
        "sparse_routed_accuracy": float(sparse_route_correct.mean()) if sparse_route_correct.size else None,
        "route_counts": {
            "dense": int(route_dense.sum()),
            "sparse": int(route_sparse.sum()),
            "dense_fraction": float(route_dense.mean()),
            "sparse_fraction": float(route_sparse.mean()),
        },
        "avg_spikes_per_sample": {
            "dense": dense_avg_spikes,
            "sparse": sparse_avg_spikes,
            "routed_model_only": float(routed_spikes.mean()),
        },
        "energy": {
            "paper_dense_energy_J_per_inference": paper_energy_j,
            "energy_per_spike_J": float(energy_per_spike_j),
            "baseline_dense_only_avg_J": float(baseline_model_energy.mean()),
            "baseline_dense_only_total_J": float(baseline_model_energy.sum()),
            "sparse_only_avg_J": float(sparse_model_energy.mean()),
            "sparse_only_total_J": float(sparse_model_energy.sum()),
            "lzc_avg_J": float(lzc_energy.mean()),
            "lzc_total_J": float(lzc_energy.sum()),
            "routed_model_only_avg_J": float(routed_model_energy.mean()),
            "routed_model_only_total_J": float(routed_model_energy.sum()),
            "routed_with_lzc_avg_J": float(routed_total_energy.mean()),
            "routed_with_lzc_total_J": float(routed_total_energy.sum()),
            "energy_gain_model_only_J": float(baseline_model_energy.sum() - routed_model_energy.sum()),
            "energy_gain_with_lzc_J": float(baseline_model_energy.sum() - routed_total_energy.sum()),
            "energy_savings_model_only_percent": float(
                100.0 * (1.0 - routed_model_energy.sum() / baseline_model_energy.sum())
            ),
            "energy_savings_with_lzc_percent": float(
                100.0 * (1.0 - routed_total_energy.sum() / baseline_model_energy.sum())
            ),
        },
        "lzc": {
            "score_mean": float(lzc_scores.mean()),
            "score_min": float(lzc_scores.min()),
            "score_max": float(lzc_scores.max()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SHD dense/sparse routing metrics.")
    parser.add_argument("--dense_model_path", required=True)
    parser.add_argument("--sparse_model_path", required=True)
    parser.add_argument("--dataset_path", default="./data")
    parser.add_argument("--lzc_file", default="LZC_Energy/lzc_energy_SHD.txt")
    parser.add_argument("--output_path", default="./new_test_results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--paper_energy_mJ", type=float, default=0.42)
    parser.add_argument("--n_frames", type=int, default=1400)
    parser.add_argument("--net_dt", type=float, default=0.001)
    parser.add_argument("--cache_tag", default=None)
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
        cache_tag=args.cache_tag,
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

    lzc_records = load_lzc_records(args.lzc_file)
    if len(lzc_records) < len(cached_test):
        raise ValueError(f"LZC file has {len(lzc_records)} rows but test set has {len(cached_test)} samples")

    records = []
    idx = 0
    with torch.no_grad():
        for batch_idx, (batch_data, labels) in enumerate(test_loader):
            batch_data = batch_data.to(device).float()
            labels = labels.to(device)

            dense_preds, dense_logits, dense_spikes = predict_and_count(dense_model, batch_data)
            sparse_preds, sparse_logits, sparse_spikes = predict_and_count(sparse_model, batch_data)
            dense_probs = torch.softmax(dense_logits, dim=1)
            sparse_probs = torch.softmax(sparse_logits, dim=1)

            for i in range(batch_data.shape[0]):
                lzc = lzc_records[idx]
                label = int(labels[i].item())
                dense_pred = int(dense_preds[i].item())
                sparse_pred = int(sparse_preds[i].item())
                records.append(
                    {
                        "sample_idx": idx,
                        "label": label,
                        "dense_pred": dense_pred,
                        "sparse_pred": sparse_pred,
                        "dense_correct": dense_pred == label,
                        "sparse_correct": sparse_pred == label,
                        "dense_spikes": float(dense_spikes[i].item()),
                        "sparse_spikes": float(sparse_spikes[i].item()),
                        "dense_entropy": float(entropy(dense_probs[i].detach().cpu().numpy())),
                        "sparse_entropy": float(entropy(sparse_probs[i].detach().cpu().numpy())),
                        "lzc_score": float(lzc["lzc_score"]),
                        "lzc_energy_J": float(lzc["energy_J"]),
                        "lzc_cycles": int(lzc["cycles"]),
                    }
                )
                idx += 1

            print(f"Processed {idx}/{len(cached_test)} samples")

    threshold_info = optimal_lzc_threshold(records)
    threshold = threshold_info["threshold"]
    summary = summarize(records, threshold, args.paper_energy_mJ * 1e-3)
    summary["threshold"] = threshold_info
    summary["paths"] = {
        "dense_model_path": args.dense_model_path,
        "sparse_model_path": args.sparse_model_path,
        "lzc_file": args.lzc_file,
    }
    summary["hyperparams"] = {
        "dense": dense_hp,
        "sparse": sparse_hp,
    }
    summary["timestamp"] = datetime.now().isoformat()

    out_dir = os.path.join(args.output_path, "shd", "routing_eval")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"shd_routing_eval_{stamp}.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 72)
    print("SHD PAPER-MATCHED ROUTING EVALUATION")
    print("=" * 72)
    print(f"Samples: {summary['n_samples']}")
    print(f"Dense accuracy:  {summary['dense_accuracy'] * 100:.2f}%")
    print(f"Sparse accuracy: {summary['sparse_accuracy'] * 100:.2f}%")
    print(f"Routed accuracy: {summary['routed_accuracy'] * 100:.2f}%")
    print(f"Threshold:       {threshold}")
    print(f"ROC AUC:         {threshold_info.get('roc_auc')}")
    print(f"Route counts:    {summary['route_counts']}")
    print(f"Dense spikes/sample:  {summary['avg_spikes_per_sample']['dense']:.2f}")
    print(f"Sparse spikes/sample: {summary['avg_spikes_per_sample']['sparse']:.2f}")
    print(f"Routed model spikes/sample: {summary['avg_spikes_per_sample']['routed_model_only']:.2f}")
    print("\nEnergy:")
    print(f"  Baseline dense-only avg J:       {summary['energy']['baseline_dense_only_avg_J']:.6e}")
    print(f"  Baseline dense-only total J:     {summary['energy']['baseline_dense_only_total_J']:.6e}")
    print(f"  Sparse-only avg J:               {summary['energy']['sparse_only_avg_J']:.6e}")
    print(f"  Routed model-only avg J:         {summary['energy']['routed_model_only_avg_J']:.6e}")
    print(f"  Routed with LZC avg J:           {summary['energy']['routed_with_lzc_avg_J']:.6e}")
    print(f"  Routed with LZC total J:         {summary['energy']['routed_with_lzc_total_J']:.6e}")
    print(f"  Energy gain model-only J:        {summary['energy']['energy_gain_model_only_J']:.6e}")
    print(f"  Energy gain with LZC J:          {summary['energy']['energy_gain_with_lzc_J']:.6e}")
    print(f"  Energy savings model-only %:     {summary['energy']['energy_savings_model_only_percent']:.2f}%")
    print(f"  Energy savings with LZC %:       {summary['energy']['energy_savings_with_lzc_percent']:.2f}%")
    print(f"\nSaved JSON: {json_path}")


if __name__ == "__main__":
    main()
