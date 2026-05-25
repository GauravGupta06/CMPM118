"""
measure_prefix_router_binary_energy.py - STM32 measurement for the SHD router
from packed binary model input.

This is the more rigorous version of the prefix-router measurement. It sends
the first 400 ms of each SHD sample as the binarized model input:

    400 * 700 = 280000 bits = 35000 packed bytes

The STM32 stores the packed bytes, then the timed region popcounts those bytes
and applies the router threshold:

    prefix_spikes >= 6993 -> dense

Usage:
    python Prefix_Router_Energy/measure_prefix_router_binary_energy.py --dry-run
    python Prefix_Router_Energy/measure_prefix_router_binary_energy.py --port COM3
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


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


def resolve_paths(args):
    stem = default_stem(args.n_frames, args.net_dt, args.prefix_ms)
    input_path = args.input or os.path.join(
        args.output_dir, f"prefix_router_binary_input_{stem}.hex"
    )
    expected_path = args.expected or os.path.join(
        args.output_dir, f"prefix_router_binary_expected_{stem}.csv"
    )
    output_path = args.output or os.path.join(
        args.output_dir, f"prefix_router_binary_energy_{stem}.txt"
    )
    summary_path = args.summary or os.path.join(
        args.output_dir, f"prefix_router_binary_energy_{stem}_summary.json"
    )
    return input_path, expected_path, output_path, summary_path


def load_shd_test_dataset(args):
    from datasets.shd_dataset import SHDDataset

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


def sample_to_packed_prefix(sample, prefix_bins):
    import numpy as np

    arr = np.asarray(sample)
    if arr.ndim != 2:
        raise ValueError(f"Expected SHD sample shape [T, C], got {arr.shape}")
    bits = (arr[:prefix_bins] > 0).astype(np.uint8).reshape(-1)
    packed = np.packbits(bits)
    return packed, int(bits.sum()), int(bits.size)


def write_binary_input_files(test_dataset, input_path, expected_path, args):
    prefix_bins = int(round((args.prefix_ms / 1000.0) / args.net_dt))
    os.makedirs(os.path.dirname(input_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(expected_path) or ".", exist_ok=True)

    print(f"Writing packed binary STM32 input: {input_path}")
    print(f"Writing expected CSV: {expected_path}")
    print(f"Samples: {len(test_dataset)} | prefix bins: {prefix_bins}")

    dense_count = 0
    byte_lengths = []
    bit_lengths = []
    prefix_scores = []

    with open(input_path, "w") as in_fh, open(expected_path, "w", newline="") as exp_fh:
        writer = csv.DictWriter(
            exp_fh,
            fieldnames=[
                "sample_idx",
                "label",
                "prefix_ms",
                "prefix_bins",
                "prefix_bits",
                "packed_bytes",
                "prefix_spikes",
                "route_dense",
            ],
        )
        writer.writeheader()

        n = len(test_dataset)
        if args.max_samples is not None:
            n = min(n, args.max_samples)

        for idx in range(n):
            sample, label = test_dataset[idx]
            packed, prefix_score, nbits = sample_to_packed_prefix(sample, prefix_bins)
            route_dense = int(prefix_score >= args.threshold)
            dense_count += route_dense
            byte_lengths.append(int(packed.size))
            bit_lengths.append(nbits)
            prefix_scores.append(prefix_score)

            in_fh.write(packed.tobytes().hex() + "\n")
            writer.writerow(
                {
                    "sample_idx": idx,
                    "label": int(label),
                    "prefix_ms": args.prefix_ms,
                    "prefix_bins": prefix_bins,
                    "prefix_bits": nbits,
                    "packed_bytes": int(packed.size),
                    "prefix_spikes": prefix_score,
                    "route_dense": route_dense,
                }
            )

            if (idx + 1) % 250 == 0 or idx == n - 1:
                print(f"  wrote {idx + 1}/{n}")

    print(f"Packed bytes/sample: {byte_lengths[0] if byte_lengths else 0}")
    print(f"Bits/sample: {bit_lengths[0] if bit_lengths else 0}")
    mean_prefix = sum(prefix_scores) / len(prefix_scores) if prefix_scores else 0.0
    print(f"Mean prefix spikes: {mean_prefix:.2f}")
    print(f"Dense-route fraction: {dense_count / len(prefix_scores):.3f}")


def load_expected(expected_path):
    expected = []
    with open(expected_path, newline="") as fh:
        for row in csv.DictReader(fh):
            expected.append(
                {
                    "prefix_spikes": int(row["prefix_spikes"]),
                    "route_dense": int(row["route_dense"]),
                    "packed_bytes": int(row["packed_bytes"]),
                    "prefix_bits": int(row["prefix_bits"]),
                }
            )
    return expected


def read_response(ser, allow_ready=True):
    for _ in range(10):
        response = ser.readline().decode(errors="replace").strip()
        if not response:
            continue
        if allow_ready and response == "READY":
            continue
        return response
    return ""


def expect_ok(ser, context):
    response = read_response(ser)
    if not response.startswith("OK"):
        raise RuntimeError(f"STM32 did not acknowledge {context}: {response!r}")
    return response


def run_board(args, input_path, expected_path):
    import serial
    from time import sleep

    expected = load_expected(expected_path)
    with open(input_path) as fh:
        lines = [line.strip() for line in fh if line.strip()]

    print(f"Connecting to STM32 on {args.port}...")
    ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    sleep(2)
    ser.reset_input_buffer()

    results = []
    mismatches = 0
    n = len(lines)
    print(f"Sending {n} packed-binary samples...")

    for idx, hex_line in enumerate(lines):
        exp = expected[idx]
        nbytes = len(hex_line) // 2
        nbits = exp["prefix_bits"]

        ser.write(f"BEGIN {nbytes} {nbits}\n".encode())
        expect_ok(ser, f"BEGIN sample {idx}")

        chunk_hex_chars = args.chunk_bytes * 2
        for start in range(0, len(hex_line), chunk_hex_chars):
            chunk = hex_line[start : start + chunk_hex_chars]
            ser.write(("DATA " + chunk + "\n").encode())
            expect_ok(ser, f"DATA sample {idx}")

        ser.write(b"RUN\n")
        response = read_response(ser)
        parts = response.split()
        if len(parts) != 5:
            raise RuntimeError(f"Unexpected RUN response for sample {idx}: {response!r}")

        cycles = int(parts[0])
        prefix_score = int(parts[1])
        route_dense = int(parts[2])
        returned_nbytes = int(parts[3])
        returned_nbits = int(parts[4])

        if (
            prefix_score != exp["prefix_spikes"]
            or route_dense != exp["route_dense"]
            or returned_nbytes != exp["packed_bytes"]
            or returned_nbits != exp["prefix_bits"]
        ):
            mismatches += 1
            print(
                f"  MISMATCH sample {idx}: got "
                f"{prefix_score}/{route_dense}/{returned_nbytes}/{returned_nbits}, "
                f"expected {exp}"
            )

        results.append(
            (
                cycles,
                prefix_score,
                route_dense,
                returned_nbytes,
                returned_nbits,
                exp["prefix_spikes"],
                exp["route_dense"],
            )
        )

        if (idx + 1) % 50 == 0 or idx == n - 1:
            print(
                f"  [{idx + 1}/{n}] cycles={cycles}, "
                f"prefix={prefix_score}, route_dense={route_dense}"
            )

    ser.write(b"DONE\n")
    sleep(0.5)
    ser.close()

    if mismatches:
        print(f"WARNING: {mismatches} mismatches.")
    else:
        print("All board outputs matched expected prefix scores/routes.")

    return results


def summarize(results, args, output_path, summary_path):
    valid = [r for r in results if r[0] > 0]
    if not valid:
        raise RuntimeError("No valid STM32 results.")

    cycles = [float(r[0]) for r in valid]
    prefix_scores = [float(r[1]) for r in valid]
    route_dense = [float(r[2]) for r in valid]
    energies = [cycle * args.energy_per_cycle for cycle in cycles]

    def mean(values):
        return sum(values) / len(values)

    def median(values):
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    router_mj = float(mean(energies) * 1000.0)
    routed_total_mj = args.routed_model_energy_mJ + router_mj
    savings = 100.0 * (1.0 - routed_total_mj / args.dense_energy_mJ)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "measurement": "packed_binary_prefix_router",
        "n_samples": len(results),
        "n_valid": len(valid),
        "router": {
            "metric": "prefix_400ms_spikes",
            "input_representation": "packed binarized SHD raster, first 400 ms",
            "threshold": args.threshold,
        },
        "stm32_projection": {
            "energy_per_cycle_J": args.energy_per_cycle,
            "mean_cycles": float(mean(cycles)),
            "median_cycles": float(median(cycles)),
            "min_cycles": int(min(cycles)),
            "max_cycles": int(max(cycles)),
            "mean_router_energy_uJ": float(mean(energies) * 1e6),
            "median_router_energy_uJ": float(median(energies) * 1e6),
        },
        "prefix_scores": {
            "mean": float(mean(prefix_scores)),
            "median": float(median(prefix_scores)),
            "min": int(min(prefix_scores)),
            "max": int(max(prefix_scores)),
            "dense_route_fraction": float(mean(route_dense)),
        },
        "system_energy": {
            "dense_only_mJ_per_sample": args.dense_energy_mJ,
            "routed_model_only_mJ_per_sample": args.routed_model_energy_mJ,
            "router_mJ_per_sample": router_mj,
            "routed_total_mJ_per_sample": routed_total_mj,
            "energy_savings_percent_vs_dense": float(savings),
        },
        "paths": {
            "output": output_path,
            "summary": summary_path,
        },
    }

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nPacked-binary STM32 prefix-router summary")
    print("=" * 72)
    print(f"Valid samples: {len(valid)} / {len(results)}")
    print(f"Mean cycles/sample: {mean(cycles):.2f}")
    print(f"Mean router energy: {mean(energies) * 1e6:.4f} uJ/sample")
    print(f"Routed total energy: {routed_total_mj:.6f} mJ/sample")
    print(f"Energy savings vs dense: {savings:.2f}%")
    print(f"Summary JSON: {summary_path}")


def write_output(results, args, output_path, summary_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        for row in results:
            cycles, prefix_score, route_dense, nbytes, nbits, exp_score, exp_route = row
            energy = cycles * args.energy_per_cycle if cycles > 0 else 0.0
            fh.write(
                f"{energy} {cycles} {prefix_score} {route_dense} "
                f"{nbytes} {nbits} {exp_score} {exp_route}\n"
            )
    summarize(results, args, output_path, summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Measure SHD packed-binary prefix router on STM32."
    )
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--input", default=None)
    parser.add_argument("--expected", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=None)
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
    parser.add_argument("--chunk_bytes", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path, expected_path, output_path, summary_path = resolve_paths(args)

    if args.input:
        print(f"Using existing packed-binary input: {input_path}")
    else:
        test_dataset = load_shd_test_dataset(args)
        write_binary_input_files(test_dataset, input_path, expected_path, args)

    if args.dry_run:
        print("\nDry-run complete. Flash packed-binary STM32 firmware, then rerun without --dry-run.")
        print(f"Input:    {input_path}")
        print(f"Expected: {expected_path}")
        return

    results = run_board(args, input_path, expected_path)
    write_output(results, args, output_path, summary_path)


if __name__ == "__main__":
    main()
