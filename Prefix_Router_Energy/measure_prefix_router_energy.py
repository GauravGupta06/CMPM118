"""
measure_prefix_router_energy.py - STM32 energy pipeline for the SHD prefix router.

This measures the selected non-learned SHD router:

    prefix_400ms_spikes >= 6993 -> dense
    otherwise                   -> sparse

The host writes one line per SHD test sample. Each line contains 400 comma-
separated frame spike counts from the paper-matched SHD preprocessing
([T=1400, C=700], 1 ms bins). The STM32 parses the line, then the DWT cycle
counter measures only:

    sum(first_400_frame_counts) + threshold comparison

Usage:
    # Generate input/expected files only.
    python Prefix_Router_Energy/measure_prefix_router_energy.py --dry-run

    # Measure on a flashed STM32 board.
    python Prefix_Router_Energy/measure_prefix_router_energy.py --port COM3
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.shd_dataset import SHDDataset  # noqa: E402


ENERGY_PER_CYCLE_J = 1.8e-10
DEFAULT_N_FRAMES = 1400
DEFAULT_NET_DT = 0.001
DEFAULT_PREFIX_MS = 400
DEFAULT_THRESHOLD = 6993
DEFAULT_DENSE_ENERGY_MJ = 0.420
DEFAULT_ROUTED_MODEL_ENERGY_MJ = 0.3058191164648503


def default_stem(n_frames, net_dt, prefix_ms):
    dt_ms = int(round(net_dt * 1000))
    return f"SHD_T{n_frames}_dt{dt_ms}ms_p{prefix_ms}ms"


def resolve_defaults(args):
    stem = default_stem(args.n_frames, args.net_dt, args.prefix_ms)
    input_path = args.input or os.path.join(
        args.output_dir, f"prefix_router_input_{stem}.txt"
    )
    expected_path = args.expected or os.path.join(
        args.output_dir, f"prefix_router_expected_{stem}.csv"
    )
    output_path = args.output or os.path.join(
        args.output_dir, f"prefix_router_energy_{stem}.txt"
    )
    summary_path = args.summary or os.path.join(
        args.output_dir, f"prefix_router_energy_{stem}_summary.json"
    )
    return input_path, expected_path, output_path, summary_path


def load_shd_test_dataset(args):
    data = SHDDataset(
        dataset_path=args.dataset_path,
        NUM_CHANNELS=700,
        NUM_POLARITIES=1,
        n_frames=args.n_frames,
        net_dt=args.net_dt,
        use_polarity=False,
        cache_tag=args.cache_tag,
        rebuild_cache=args.rebuild_cache,
    )
    _, cached_test = data.load_shd()
    return cached_test


def sample_to_prefix_counts(sample, prefix_bins):
    arr = np.asarray(sample)
    if arr.ndim != 2:
        raise ValueError(f"Expected SHD sample shape [T, C], got {arr.shape}")
    prefix = arr[:prefix_bins] > 0
    return prefix.sum(axis=1).astype(np.uint16)


def write_prefix_input_files(test_dataset, input_path, expected_path, args):
    prefix_bins = int(round((args.prefix_ms / 1000.0) / args.net_dt))
    if prefix_bins <= 0 or prefix_bins > args.n_frames:
        raise ValueError(f"Invalid prefix bins: {prefix_bins}")

    os.makedirs(os.path.dirname(input_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(expected_path) or ".", exist_ok=True)

    max_line_len = 0
    n_dense = 0
    prefix_scores = []

    print(f"Writing STM32 input: {input_path}")
    print(f"Writing expected CSV: {expected_path}")
    print(f"Samples: {len(test_dataset)} | prefix bins: {prefix_bins}")

    with open(input_path, "w", newline="") as in_fh, open(
        expected_path, "w", newline=""
    ) as exp_fh:
        writer = csv.DictWriter(
            exp_fh,
            fieldnames=[
                "sample_idx",
                "label",
                "prefix_ms",
                "prefix_bins",
                "prefix_spikes",
                "route_dense",
            ],
        )
        writer.writeheader()

        for idx in range(len(test_dataset)):
            sample, label = test_dataset[idx]
            counts = sample_to_prefix_counts(sample, prefix_bins)
            prefix_score = int(counts.sum())
            route_dense = int(prefix_score >= args.threshold)
            n_dense += route_dense
            prefix_scores.append(prefix_score)

            line = ",".join(str(int(x)) for x in counts)
            max_line_len = max(max_line_len, len(line))
            in_fh.write(line + "\n")
            writer.writerow(
                {
                    "sample_idx": idx,
                    "label": int(label),
                    "prefix_ms": args.prefix_ms,
                    "prefix_bins": prefix_bins,
                    "prefix_spikes": prefix_score,
                    "route_dense": route_dense,
                }
            )

    print(f"Max input line length: {max_line_len} characters")
    if max_line_len >= 4095:
        print("WARNING: line length reaches STM32 buffer limit of 4096 bytes.")
    print(f"Mean prefix spikes: {float(np.mean(prefix_scores)):.2f}")
    print(f"Dense-route fraction from threshold {args.threshold}: {n_dense / len(test_dataset):.3f}")


def load_expected(expected_path):
    if not expected_path or not os.path.exists(expected_path):
        return None
    expected = []
    with open(expected_path, newline="") as fh:
        for row in csv.DictReader(fh):
            expected.append(
                {
                    "prefix_spikes": int(row["prefix_spikes"]),
                    "route_dense": int(row["route_dense"]),
                }
            )
    return expected


def read_response(ser):
    for _ in range(5):
        response = ser.readline().decode(errors="replace").strip()
        if not response:
            continue
        if response == "READY":
            continue
        return response
    return ""


def run_board(port, input_path, expected_path=None, baud=115200, timeout=10):
    import serial
    from time import sleep

    with open(input_path) as fh:
        lines = [line.strip() for line in fh if line.strip()]
    expected = load_expected(expected_path)

    print(f"Connecting to STM32 on {port}...")
    ser = serial.Serial(port, baud, timeout=timeout)
    sleep(2)
    ser.reset_input_buffer()

    print(f"Sending {len(lines)} SHD samples...")
    results = []
    mismatches = 0

    for idx, line in enumerate(lines):
        ser.write((line + "\n").encode())
        response = read_response(ser)

        if not response:
            print(f"  ERROR: no response for sample {idx}")
            results.append((0, -1, -1, -1, -1, -1))
            mismatches += 1
            continue

        parts = response.split()
        if len(parts) != 4:
            print(f"  ERROR: unexpected response for sample {idx}: {response!r}")
            results.append((0, -1, -1, -1, -1, -1))
            mismatches += 1
            continue

        cycles = int(parts[0])
        prefix_score = int(parts[1])
        route_dense = int(parts[2])
        n_counts = int(parts[3])
        expected_score = -1
        expected_route = -1

        if expected is not None and idx < len(expected):
            expected_score = expected[idx]["prefix_spikes"]
            expected_route = expected[idx]["route_dense"]
            if prefix_score != expected_score or route_dense != expected_route:
                mismatches += 1
                print(
                    f"  MISMATCH sample {idx}: board score/route="
                    f"{prefix_score}/{route_dense}, expected="
                    f"{expected_score}/{expected_route}"
                )

        results.append(
            (cycles, prefix_score, route_dense, n_counts, expected_score, expected_route)
        )

        if (idx + 1) % 250 == 0 or idx == len(lines) - 1:
            print(
                f"  [{idx + 1}/{len(lines)}] cycles={cycles}, "
                f"prefix={prefix_score}, route_dense={route_dense}"
            )

    ser.write(b"DONE\n")
    sleep(0.5)
    ser.close()

    if mismatches:
        print(f"WARNING: {mismatches} board/expected mismatches or missing responses.")
    else:
        print("All board outputs matched expected prefix scores/routes.")

    return results


def summarize_results(results, args, output_path, summary_path):
    valid = [r for r in results if r[0] > 0]
    if not valid:
        raise RuntimeError("No valid STM32 results to summarize.")

    cycles = np.array([r[0] for r in valid], dtype=np.float64)
    prefix_scores = np.array([r[1] for r in valid], dtype=np.float64)
    route_dense = np.array([r[2] for r in valid], dtype=np.int64)
    energies_j = cycles * args.energy_per_cycle
    router_mj = float(energies_j.mean() * 1000.0)
    total_routed_mj = args.routed_model_energy_mJ + router_mj
    savings_percent = 100.0 * (1.0 - total_routed_mj / args.dense_energy_mJ)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(results),
        "n_valid": len(valid),
        "router": {
            "metric": "prefix_400ms_spikes",
            "prefix_ms": args.prefix_ms,
            "threshold": args.threshold,
            "route_rule": "dense if prefix_spikes >= threshold else sparse",
        },
        "stm32_projection": {
            "energy_per_cycle_J": args.energy_per_cycle,
            "mean_cycles": float(cycles.mean()),
            "median_cycles": float(np.median(cycles)),
            "min_cycles": int(cycles.min()),
            "max_cycles": int(cycles.max()),
            "mean_router_energy_J": float(energies_j.mean()),
            "mean_router_energy_uJ": float(energies_j.mean() * 1e6),
            "median_router_energy_uJ": float(np.median(energies_j) * 1e6),
        },
        "prefix_scores": {
            "mean": float(prefix_scores.mean()),
            "median": float(np.median(prefix_scores)),
            "min": int(prefix_scores.min()),
            "max": int(prefix_scores.max()),
            "dense_route_fraction": float(route_dense.mean()),
        },
        "system_energy": {
            "dense_only_mJ_per_sample": args.dense_energy_mJ,
            "routed_model_only_mJ_per_sample": args.routed_model_energy_mJ,
            "router_mJ_per_sample": router_mj,
            "routed_total_mJ_per_sample": total_routed_mj,
            "energy_savings_percent_vs_dense": float(savings_percent),
        },
        "paths": {
            "output": output_path,
            "summary": summary_path,
        },
    }

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nSTM32 prefix-router summary")
    print("=" * 72)
    print(f"Valid samples: {len(valid)} / {len(results)}")
    print(f"Mean cycles/sample: {cycles.mean():.2f}")
    print(f"Mean router energy: {energies_j.mean() * 1e6:.4f} uJ/sample")
    print(f"Routed total energy: {total_routed_mj:.6f} mJ/sample")
    print(f"Energy savings vs dense: {savings_percent:.2f}%")
    print(f"Summary JSON: {summary_path}")

    return summary


def write_energy_table(results, output_path, args, summary_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Writing energy table: {output_path}")
    with open(output_path, "w") as fh:
        for cycles, prefix_score, route_dense, n_counts, expected_score, expected_route in results:
            energy = cycles * args.energy_per_cycle if cycles > 0 else 0.0
            fh.write(
                f"{energy} {cycles} {prefix_score} {route_dense} "
                f"{n_counts} {expected_score} {expected_route}\n"
            )
    summarize_results(results, args, output_path, summary_path)


def main():
    parser = argparse.ArgumentParser(description="Measure SHD prefix-router cycles on STM32.")
    parser.add_argument("--port", default="COM3", help="STM32 serial port, e.g. COM3")
    parser.add_argument("--input", default=None, help="Pre-generated prefix input file")
    parser.add_argument("--expected", default=None, help="Expected prefix score CSV")
    parser.add_argument("--output", default=None, help="Output energy table")
    parser.add_argument("--summary", default=None, help="Output summary JSON")
    parser.add_argument("--output_dir", default="Prefix_Router_Energy")
    parser.add_argument("--dataset_path", default="./data")
    parser.add_argument("--n_frames", type=int, default=DEFAULT_N_FRAMES)
    parser.add_argument("--net_dt", type=float, default=DEFAULT_NET_DT)
    parser.add_argument("--prefix_ms", type=int, default=DEFAULT_PREFIX_MS)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--cache_tag", default=None)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--energy_per_cycle", type=float, default=ENERGY_PER_CYCLE_J)
    parser.add_argument("--dense_energy_mJ", type=float, default=DEFAULT_DENSE_ENERGY_MJ)
    parser.add_argument("--routed_model_energy_mJ", type=float, default=DEFAULT_ROUTED_MODEL_ENERGY_MJ)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Generate input files only")
    args = parser.parse_args()

    input_path, expected_path, output_path, summary_path = resolve_defaults(args)

    if args.input:
        print(f"Using existing input file: {input_path}")
    else:
        test_dataset = load_shd_test_dataset(args)
        write_prefix_input_files(test_dataset, input_path, expected_path, args)

    if args.dry_run:
        print("\nDry-run complete. Flash the STM32 firmware, then rerun without --dry-run.")
        print(f"Input:    {input_path}")
        print(f"Expected: {expected_path}")
        return

    results = run_board(
        args.port,
        input_path,
        expected_path=expected_path if os.path.exists(expected_path) else None,
        baud=args.baud,
        timeout=args.timeout,
    )
    write_energy_table(results, output_path, args, summary_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
