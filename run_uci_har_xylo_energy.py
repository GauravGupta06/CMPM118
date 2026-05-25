"""Run current UCI HAR dense/sparse SNNs on Xylo Audio 3 and save energy metrics.

This script is intentionally UCI-only. It reuses the Xylo deployment flow from
``router_xylo.py`` / ``xylo.py`` but removes routing logic so the dense and
sparse models can be measured independently on the same UCI HAR test samples.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from datasets.uci_har import UCIHARDataset
from models.uci_har_model import UCIHARSNN


def import_xylo_modules():
    """Import Xylo modules only when hardware execution is requested."""
    import samna
    from rockpool.devices.xylo import find_xylo_hdks
    from rockpool.devices.xylo import syns65302 as xa3
    from rockpool.nn.combinators import Sequential as RockpoolSequential
    from rockpool.transform import quantize_methods as q

    return samna, find_xylo_hdks, xa3, RockpoolSequential, q


def load_model(model_path):
    """Load a saved UCIHARSNN checkpoint using its stored hyperparameters."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint or "hyperparams" not in checkpoint:
        raise ValueError(f"Checkpoint is missing state_dict/hyperparams: {model_path}")

    hp = checkpoint["hyperparams"]
    model = UCIHARSNN(
        input_size=hp["input_size"],
        n_frames=hp["n_frames"],
        hidden_size=hp.get("hidden_size", 256),
        tau_mem=hp["tau_mem"],
        tau_syn=hp["tau_syn"],
        spike_lam=hp["spike_lam"],
        model_type=hp["model_type"],
        device=torch.device("cpu"),
        num_classes=hp["num_classes"],
        dt=hp["dt"],
        threshold=hp["threshold"],
        max_spikes_per_dt=hp.get("max_spikes_per_dt", 1),
        dropout_p=hp.get("dropout_p", 0.0),
        output_layer=hp.get("output_layer", "exp_syn"),
        output_readout=hp.get("output_readout", "auto"),
    )
    model.net.load_state_dict(checkpoint["state_dict"])
    model.net.eval()
    model.convert_for_hardware()
    return model, hp


def load_test_samples(args):
    """Load UCI HAR test samples in the same encoding used for current models."""
    data = UCIHARDataset(
        dataset_path=args.dataset_path,
        n_frames=args.n_frames,
        time_first=True,
        normalize=True,
        binarize=False,
        input_encoding=args.input_encoding,
        event_count_max=args.event_count_max,
        event_count_clip=args.event_count_clip,
    )
    _, test_dataset = data.load_uci_har()

    samples = []
    labels = []
    n = len(test_dataset) if args.num_samples is None else min(args.num_samples, len(test_dataset))
    for idx in range(n):
        sample, label = test_dataset[idx]
        samples.append(np.asarray(sample).astype(np.int32))
        labels.append(int(label))

    return samples, labels


def cpu_accuracy(model, samples, labels, batch_size=128):
    """Sanity-check model accuracy in PyTorch before measuring hardware energy."""
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = np.stack(samples[start:start + batch_size]).astype(np.float32)
            targets = torch.tensor(labels[start:start + batch_size], dtype=torch.long)
            x = torch.tensor(batch, dtype=torch.float32)
            model.net.reset_state()
            output, _, recording = model.net(x, record=True)
            logits = model._readout_logits(output, recording)
            correct += int((logits.argmax(1) == targets).sum().item())
            total += int(targets.numel())
    return correct / total if total else 0.0


def collect_serializable_modules(seq_module):
    """Collect modules that can be serialized to a Rockpool graph, skipping dropout."""
    import copy

    selected = []
    for name, child in seq_module._modules.items():
        try:
            child.as_graph()
            selected.append(copy.deepcopy(child))
            print(f"[TAKE] module {name}: {child.__class__.__name__}")
        except Exception:
            if len(child._modules) == 0:
                print(f"[SKIP] module {name}: {child.__class__.__name__}")
                continue

            for gname, gchild in child._modules.items():
                try:
                    gchild.as_graph()
                    selected.append(copy.deepcopy(gchild))
                    print(f"[TAKE] module {name}.{gname}: {gchild.__class__.__name__}")
                except Exception:
                    print(f"[SKIP] module {name}.{gname}: {gchild.__class__.__name__}")

    if not selected:
        raise RuntimeError("No Xylo-serializable modules found.")
    return selected


def build_xylo_config(model, dt):
    """Convert a UCIHARSNN Rockpool model into a validated Xylo Audio 3 config."""
    samna, _, xa3, RockpoolSequential, q = import_xylo_modules()

    clean_net = RockpoolSequential(*collect_serializable_modules(model.net))
    graph = clean_net.as_graph()
    spec = xa3.mapper(
        graph,
        weight_dtype="float",
        threshold_dtype="float",
        dash_dtype="float",
    )
    spec["dt"] = dt
    quant_spec = q.channel_quantize(**spec)

    for key in ["dt", "aliases"]:
        if key in spec:
            quant_spec[key] = spec[key]

    for key in ["dash_mem", "dash_mem_out", "dash_syn", "dash_syn_2", "dash_syn_out"]:
        if key in quant_spec:
            quant_spec[key] = np.abs(quant_spec[key]).astype(np.uint8)

    config, is_valid, msg = xa3.config_from_specification(**quant_spec)
    if not is_valid:
        raise ValueError(f"Invalid Xylo configuration: {msg}")

    is_valid2, msg2 = samna.xyloAudio3.validate_configuration(config)
    if not is_valid2:
        raise ValueError(f"Samna validation failed: {msg2}")

    return config


def predict_from_xylo_output(output, recorded, output_readout):
    """Match the current UCI model readout: mean Isyn/Vmem if requested."""
    readout = output_readout or "auto"
    if readout == "auto":
        readout = "isyn"

    if readout == "isyn" and recorded.get("Isyn_out") is not None:
        arr = np.asarray(recorded["Isyn_out"])
    elif readout == "vmem" and recorded.get("Vmem_out") is not None:
        arr = np.asarray(recorded["Vmem_out"])
    else:
        arr = np.asarray(output)

    if arr.ndim == 1:
        logits = arr
    elif arr.ndim == 2:
        logits = arr.mean(axis=0)
    elif arr.ndim == 3:
        logits = arr.mean(axis=0).mean(axis=0)
    else:
        logits = arr.reshape(-1, arr.shape[-1]).mean(axis=0)

    return int(np.argmax(logits))


def run_model_on_xylo(model_name, model, hp, hdk, samples, labels, args):
    """Deploy one model to Xylo and collect per-sample energy metrics."""
    _, _, xa3, _, _ = import_xylo_modules()

    config = build_xylo_config(model, dt=hp["dt"])
    print(f"[{model_name}] hidden weights: {len(config.hidden.weights)}")
    print(f"[{model_name}] readout weights: {len(config.readout.weights)}")

    mod_samna = xa3.XyloSamna(hdk, config, dt=hp["dt"], power_frequency=args.power_frequency)

    per_sample = []
    failed = []
    correct = 0
    total_energy = 0.0
    total_time = 0.0

    for idx, (sample, label) in enumerate(zip(samples, labels)):
        try:
            output, _, recorded = mod_samna(sample, record=True, record_power=True)
            analog = recorded.get("analog_power")
            digital = recorded.get("digital_power")
            io = recorded.get("io_power")
            inf_duration = float(recorded.get("inf_duration", 0.0))

            if analog is not None and digital is not None and io is not None:
                total_power = np.asarray(analog) + np.asarray(digital) + np.asarray(io)
                power_dt = inf_duration / len(total_power) if len(total_power) else 0.0
                energy_j = float(np.sum(total_power) * power_dt)
                mean_power = float(np.mean(total_power)) if len(total_power) else 0.0
                peak_power = float(np.max(total_power)) if len(total_power) else 0.0
            else:
                energy_j = 0.0
                mean_power = 0.0
                peak_power = 0.0

            pred = predict_from_xylo_output(output, recorded, hp.get("output_readout", "auto"))
            is_correct = int(pred == label)
            correct += is_correct
            total_energy += energy_j
            total_time += inf_duration

            row = {
                "index": idx,
                "label": int(label),
                "pred": int(pred),
                "correct": is_correct,
                "energy_j": energy_j,
                "inf_duration_s": inf_duration,
                "mean_power_w": mean_power,
                "peak_power_w": peak_power,
                "input_events": int(np.sum(sample)),
                "recorded_spikes": int(np.sum(recorded.get("Spikes", 0))) if recorded.get("Spikes") is not None else 0,
            }
            per_sample.append(row)

            if args.print_every and ((idx + 1) % args.print_every == 0 or idx == 0):
                print(
                    f"[{model_name}] {idx + 1}/{len(samples)} "
                    f"acc={correct / len(per_sample):.4f} "
                    f"energy={energy_j:.6e} J time={inf_duration:.4f}s"
                )
        except Exception as exc:
            failed.append({"index": idx, "error": str(exc)})
            print(f"[{model_name}] ERROR sample {idx}: {exc}")

    total = len(per_sample)
    return {
        "model_name": model_name,
        "model_type": hp.get("model_type", model_name),
        "model_accuracy": correct / total if total else 0.0,
        "total_samples": total,
        "failed_samples": failed,
        "total_energy_j": total_energy,
        "avg_energy_j": total_energy / total if total else 0.0,
        "total_inf_duration_s": total_time,
        "avg_inf_duration_s": total_time / total if total else 0.0,
        "per_sample": per_sample,
    }


def save_results(results, args):
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "created_at": stamp,
        "dataset_path": args.dataset_path,
        "input_encoding": args.input_encoding,
        "event_count_max": args.event_count_max,
        "event_count_clip": args.event_count_clip,
        "num_samples_requested": args.num_samples,
        "models": {
            name: {k: v for k, v in result.items() if k != "per_sample"}
            for name, result in results.items()
        },
    }

    summary_path = os.path.join(args.output_dir, f"uci_har_xylo_energy_summary_{stamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for name, result in results.items():
        csv_path = os.path.join(args.output_dir, f"uci_har_xylo_{name}_per_sample_{stamp}.csv")
        rows = result["per_sample"]
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[SAVE] {name} per-sample CSV: {csv_path}")

    print(f"[SAVE] summary JSON: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure UCI HAR dense/sparse models on Xylo Audio 3.")
    parser.add_argument("--dense_model", default="new_test_results/uci_har/dense/models/Take5_T128_Epochs120.pth")
    parser.add_argument("--sparse_model", default="new_test_results/uci_har/sparse/models/Take2_T128_Epochs120.pth")
    parser.add_argument("--dataset_path", default="./data")
    parser.add_argument("--output_dir", default="new_test_results/uci_har/xylo_energy")
    parser.add_argument("--which", choices=["dense", "sparse", "both"], default="both")
    parser.add_argument("--num_samples", type=int, default=None, help="Optional test-sample limit for quick runs.")
    parser.add_argument("--n_frames", type=int, default=128)
    parser.add_argument("--input_encoding", default="signed_count_6",
                        choices=["binary", "positive_count", "unsigned_count", "signed_count_6"])
    parser.add_argument("--event_count_max", type=int, default=3)
    parser.add_argument("--event_count_clip", type=float, default=3.0)
    parser.add_argument("--power_frequency", type=float, default=20.0)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--skip_cpu_check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    samna, find_xylo_hdks, _, _, _ = import_xylo_modules()
    _ = samna

    samples, labels = load_test_samples(args)
    print(f"Loaded {len(samples)} UCI HAR test samples.")

    models = {}
    if args.which in {"dense", "both"}:
        print(f"Loading dense model: {args.dense_model}")
        models["dense"] = load_model(args.dense_model)
    if args.which in {"sparse", "both"}:
        print(f"Loading sparse model: {args.sparse_model}")
        models["sparse"] = load_model(args.sparse_model)

    if not args.skip_cpu_check:
        for name, (model, _) in models.items():
            acc = cpu_accuracy(model, samples, labels)
            print(f"[CPU sanity] {name}: accuracy={acc * 100:.2f}%")

    hdk_nodes, _, _ = find_xylo_hdks()
    if not hdk_nodes:
        raise RuntimeError("No Xylo Audio 3 HDK found.")
    hdk = hdk_nodes[0]

    results = {}
    for name, (model, hp) in models.items():
        print("\n" + "=" * 80)
        print(f"Running {name} model on Xylo")
        print("=" * 80)
        results[name] = run_model_on_xylo(name, model, hp, hdk, samples, labels, args)

    save_results(results, args)

    print("\nSummary")
    for name, result in results.items():
        print(
            f"{name}: acc={result['model_accuracy'] * 100:.2f}% "
            f"avg_energy={result['avg_energy_j']:.6e} J "
            f"avg_time={result['avg_inf_duration_s']:.6f}s "
            f"samples={result['total_samples']} failed={len(result['failed_samples'])}"
        )


if __name__ == "__main__":
    main()
