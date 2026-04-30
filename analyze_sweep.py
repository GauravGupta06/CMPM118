"""Evaluate sweep checkpoints and rank every dense x sparse pairing.

Reads runs.csv produced by sweep_uci_har.py, evaluates test accuracy and
avg spikes/neuron/timestep for each checkpoint, then scores all cross-pairings.

Scoring:
- Hard filter: dense_acc > sparse_acc AND spike_ratio in [5, 200]
- Soft score : accuracy_gap_pts + log10(clip(spike_ratio, 1, 50))
    -> rewards wider accuracy gap, caps the benefit of extreme spike ratios
       (we don't want 500x where sparse becomes irrelevant)
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from datasets.uci_har import UCIHARDataset
from models.uci_har_model import UCIHARSNN


def build_model_from_checkpoint(model_path, device):
    hp = UCIHARSNN.load_hyperparams(model_path, device="cpu")
    model = UCIHARSNN(
        input_size=hp["input_size"],
        n_frames=hp["n_frames"],
        tau_mem=hp["tau_mem"],
        tau_syn=hp["tau_syn"],
        spike_lam=hp["spike_lam"],
        model_type=hp["model_type"],
        device=device,
        num_classes=hp["num_classes"],
        dt=hp["dt"],
        threshold=hp["threshold"],
        max_spikes_per_dt=hp.get("max_spikes_per_dt", 1),
        dropout_p=hp.get("dropout_p", 0.0),
    )
    model.load_model(model_path)
    return model, hp


def evaluate_checkpoint(model_path, dataset_path, device, batch_size=64, spike_batches=20):
    model, hp = build_model_from_checkpoint(model_path, device)
    data = UCIHARDataset(
        dataset_path=dataset_path,
        n_frames=hp["n_frames"],
        time_first=True,
        normalize=True,
    )
    _, test_ds = data.load_uci_har()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    acc = model.validate_model(test_loader)
    spikes = model.get_avg_spike_count(test_loader, max_batches=spike_batches)
    return acc, spikes


def score_pair(acc_gap_pts, spike_ratio):
    if spike_ratio <= 0 or math.isinf(spike_ratio):
        return -1e9
    clipped = max(1.0, min(spike_ratio, 50.0))
    return acc_gap_pts + math.log10(clipped)


def passes_filter(dense_acc, sparse_acc, spike_ratio, min_ratio=5.0, max_ratio=200.0):
    if dense_acc <= sparse_acc:
        return False
    if not (min_ratio <= spike_ratio <= max_ratio):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", type=str, required=True,
                    help="Path to sweep dir containing runs.csv (e.g. /workspace/sweeps/sweep1)")
    ap.add_argument("--dataset_path", type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--spike_batches", type=int, default=20)
    ap.add_argument("--min_ratio", type=float, default=5.0)
    ap.add_argument("--max_ratio", type=float, default=200.0)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    runs_csv = sweep_dir / "runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found at {runs_csv}")

    with open(runs_csv) as f:
        runs = list(csv.DictReader(f))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Evaluating {len(runs)} checkpoints on {device}\n")

    results = []
    for r in runs:
        mp = r["model_path"]
        if not Path(mp).exists():
            print(f"[MISS] {r['name']}: {mp} (skipping)")
            continue
        try:
            acc, spikes = evaluate_checkpoint(
                mp, args.dataset_path, device,
                batch_size=args.batch_size, spike_batches=args.spike_batches,
            )
        except Exception as e:
            print(f"[ERR] {r['name']}: {e}")
            continue

        enriched = dict(r)
        enriched["test_acc"] = acc
        enriched["test_acc_pct"] = acc * 100.0
        enriched["avg_spikes_per_dt"] = spikes
        results.append(enriched)
        print(f"[OK]  {r['name']:<22} acc={acc*100:6.2f}%  spikes/dt={spikes:.6f}")

    if not results:
        print("\nNo evaluable checkpoints. Exiting.")
        return

    eval_csv = sweep_dir / "runs_eval.csv"
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {eval_csv}")

    dense = [r for r in results if r["model_type"] == "dense"]
    sparse = [r for r in results if r["model_type"] == "sparse"]

    pairings = []
    for d in dense:
        for s in sparse:
            gap = (d["test_acc"] - s["test_acc"]) * 100.0
            if s["avg_spikes_per_dt"] > 0:
                ratio = d["avg_spikes_per_dt"] / s["avg_spikes_per_dt"]
            else:
                ratio = float("inf")
            ok = passes_filter(d["test_acc"], s["test_acc"], ratio,
                               args.min_ratio, args.max_ratio)
            sc = score_pair(gap, ratio) if ok else -1e9
            pairings.append({
                "dense": d["name"],
                "sparse": s["name"],
                "dense_acc_pct": round(d["test_acc"] * 100.0, 3),
                "sparse_acc_pct": round(s["test_acc"] * 100.0, 3),
                "acc_gap_pts": round(gap, 3),
                "dense_spikes": d["avg_spikes_per_dt"],
                "sparse_spikes": s["avg_spikes_per_dt"],
                "spike_ratio": ratio if not math.isinf(ratio) else -1.0,
                "passes_filter": ok,
                "score": round(sc, 4),
            })

    pairings.sort(key=lambda x: x["score"], reverse=True)

    pair_csv = sweep_dir / "pairings.csv"
    with open(pair_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairings[0].keys()))
        w.writeheader()
        w.writerows(pairings)
    print(f"Wrote {pair_csv}\n")

    print(f"{'='*96}")
    print(f"All pairings ranked by score (filter: gap>0, ratio in [{args.min_ratio}, {args.max_ratio}])")
    print(f"{'='*96}")
    header = f"{'dense':<22} {'sparse':<22} {'d_acc':>7} {'s_acc':>7} {'gap':>7} {'d_spk':>10} {'s_spk':>10} {'ratio':>8} {'score':>8}  flt"
    print(header)
    print("-" * len(header))
    for p in pairings:
        tag = "PASS" if p["passes_filter"] else "fail"
        ratio_str = f"{p['spike_ratio']:.1f}" if p["spike_ratio"] >= 0 else "inf"
        print(
            f"{p['dense']:<22} {p['sparse']:<22} "
            f"{p['dense_acc_pct']:6.2f}% {p['sparse_acc_pct']:6.2f}% "
            f"{p['acc_gap_pts']:+6.2f} "
            f"{p['dense_spikes']:10.6f} {p['sparse_spikes']:10.6f} "
            f"{ratio_str:>8} {p['score']:8.3f}  {tag}"
        )

    print(f"\nBest pairing: dense={pairings[0]['dense']}  sparse={pairings[0]['sparse']}  "
          f"score={pairings[0]['score']:.3f}")


if __name__ == "__main__":
    main()
