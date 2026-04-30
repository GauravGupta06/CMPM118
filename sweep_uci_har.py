"""Automated UCI HAR dense/sparse sweep.

Runs a fixed set of (dense, sparse) training configs sequentially via subprocess,
each writing checkpoints under --output_path /workspace. Logs one row per run to
runs.csv so analyze_sweep.py can evaluate + score every dense x sparse pairing.

Does NOT modify train_UCI_HAR.py or models/uci_har_model.py.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_PATH = "/workspace"

DENSE_CONFIGS = [
    {
        "name": "dense_t03_d00",
        "threshold": 0.3,
        "dropout_p": 0.0,
        "max_spikes_per_dt": 5,
        "spike_lam": 0.0,
        "tau_mem": 0.1,
    },
    {
        "name": "dense_t03_d01",
        "threshold": 0.3,
        "dropout_p": 0.1,
        "max_spikes_per_dt": 5,
        "spike_lam": 0.0,
        "tau_mem": 0.1,
    },
    {
        "name": "dense_t02_d01",
        "threshold": 0.2,
        "dropout_p": 0.1,
        "max_spikes_per_dt": 5,
        "spike_lam": 0.0,
        "tau_mem": 0.1,
    },
]

SPARSE_CONFIGS = [
    {
        "name": "sparse_t20_lam5e5",
        "threshold": 2.0,
        "dropout_p": 0.2,
        "max_spikes_per_dt": 1,
        "spike_lam": 5e-5,
        "tau_mem": 0.005,
    },
    {
        "name": "sparse_t20_lam0",
        "threshold": 2.0,
        "dropout_p": 0.2,
        "max_spikes_per_dt": 1,
        "spike_lam": 0.0,
        "tau_mem": 0.005,
    },
    {
        "name": "sparse_t30_lam0",
        "threshold": 3.0,
        "dropout_p": 0.2,
        "max_spikes_per_dt": 1,
        "spike_lam": 0.0,
        "tau_mem": 0.005,
    },
]


def build_cmd(cfg, model_type, epochs, batch_size, n_frames, lr, dataset_path):
    prefix = "dense" if model_type == "dense" else "sparse"
    return [
        sys.executable, "train_UCI_HAR.py",
        "--model_type", model_type,
        "--n_frames", str(n_frames),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--dataset_path", dataset_path,
        "--output_path", OUTPUT_PATH,
        f"--{prefix}_threshold", str(cfg["threshold"]),
        f"--{prefix}_dropout_p", str(cfg["dropout_p"]),
        f"--{prefix}_max_spikes_per_dt", str(cfg["max_spikes_per_dt"]),
        f"--{prefix}_spike_lam", str(cfg["spike_lam"]),
        f"--{prefix}_tau_mem", str(cfg["tau_mem"]),
    ]


def read_counter(model_type):
    counter_path = Path(OUTPUT_PATH) / "uci_har" / model_type / "experiment_counter.txt"
    if not counter_path.exists():
        return 0
    try:
        return int(counter_path.read_text().strip())
    except ValueError:
        return 0


def run_config(cfg, model_type, epochs, batch_size, n_frames, lr, dataset_path, log_dir):
    print(f"\n{'='*60}\n[{time.strftime('%H:%M:%S')}] Starting {cfg['name']} ({model_type})\n{'='*60}", flush=True)
    before = read_counter(model_type)
    start = time.time()

    cmd = build_cmd(cfg, model_type, epochs, batch_size, n_frames, lr, dataset_path)
    print("CMD:", " ".join(cmd), flush=True)

    log_file = log_dir / f"{cfg['name']}.log"
    with open(log_file, "w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    elapsed = time.time() - start
    after = read_counter(model_type)
    take_num = after  # counter is incremented in save_model before being written

    model_path = (
        Path(OUTPUT_PATH) / "uci_har" / model_type / "models"
        / f"Take{take_num}_T{n_frames}_Epochs{epochs}.pth"
    )

    row = {
        "name": cfg["name"],
        "model_type": model_type,
        "take_num": take_num,
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "elapsed_sec": round(elapsed, 1),
        "returncode": result.returncode,
        "counter_before": before,
        "counter_after": after,
        "log_file": str(log_file),
        "epochs": epochs,
        "n_frames": n_frames,
        "batch_size": batch_size,
        "lr": lr,
        "cfg_threshold": cfg["threshold"],
        "cfg_dropout_p": cfg["dropout_p"],
        "cfg_max_spikes_per_dt": cfg["max_spikes_per_dt"],
        "cfg_spike_lam": cfg["spike_lam"],
        "cfg_tau_mem": cfg["tau_mem"],
    }

    status = "OK" if (result.returncode == 0 and model_path.exists()) else "FAIL"
    print(f"[{status}] {cfg['name']} done in {elapsed/60:.1f} min -> {model_path}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_frames", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dataset_path", type=str, default="./data")
    ap.add_argument("--sweep_name", type=str, default="sweep1")
    ap.add_argument("--only", type=str, default="all", choices=["all", "dense", "sparse"],
                    help="Run only one side of the sweep")
    args = ap.parse_args()

    sweep_dir = Path(OUTPUT_PATH) / "sweeps" / args.sweep_name
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / "runs.csv"

    if args.only == "dense":
        queue = [("dense", c) for c in DENSE_CONFIGS]
    elif args.only == "sparse":
        queue = [("sparse", c) for c in SPARSE_CONFIGS]
    else:
        queue = [("dense", c) for c in DENSE_CONFIGS] + [("sparse", c) for c in SPARSE_CONFIGS]

    print(f"Sweep: {args.sweep_name}  runs: {len(queue)}  epochs/run: {args.epochs}")
    print(f"Output: {sweep_dir}")
    print(f"Checkpoints go to: {OUTPUT_PATH}/uci_har/{{dense,sparse}}/models/")

    rows = []
    t0 = time.time()
    for i, (model_type, cfg) in enumerate(queue, 1):
        print(f"\n### [{i}/{len(queue)}] ###", flush=True)
        row = run_config(
            cfg, model_type, args.epochs, args.batch_size, args.n_frames,
            args.lr, args.dataset_path, log_dir
        )
        rows.append(row)

        # Write CSV after every run so a crash doesn't lose progress
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{time.strftime('%H:%M:%S')}] runs.csv updated ({len(rows)} rows)", flush=True)

    total = time.time() - t0
    print(f"\nSweep complete in {total/3600:.2f} h. CSV: {csv_path}")
    print(f"Next step: python analyze_sweep.py --sweep_dir {sweep_dir}")


if __name__ == "__main__":
    main()
