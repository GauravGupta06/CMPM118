"""
measure_dvsgesture_prefix_router_binary_energy.py

STM32 measurement for a DVSGesture prefix router from packed binary input.

The trained DVSGesture model currently receives 1 ms count frames with shape:

    [T, 2, 32, 32]

This script measures a router metric on a binary-clipped version of that same
prefix representation:

    metric bit = 1 if count_frame[t, polarity, y, x] > 0 else 0

For the default 400 ms prefix:

    400 * 2 * 32 * 32 = 819200 bits = 102400 packed bytes

USB transfer and hex parsing happen outside the STM32 timed region. The timed
region popcounts the packed binary prefix and applies a threshold.

Usage:
    python Prefix_Router_Energy/measure_dvsgesture_prefix_router_binary_energy.py --dry-run
    python Prefix_Router_Energy/measure_dvsgesture_prefix_router_binary_energy.py --port COM3
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
DEFAULT_W = 32
DEFAULT_H = 32
DEFAULT_MAX_TIMESTEPS = 600
DEFAULT_NET_DT = 0.001
DEFAULT_PREFIX_MS = 400
DEFAULT_DENOISE_FILTER_TIME = 10000

# Placeholder only. Router threshold should be selected after evaluating the
# DVS router score/AUC. The threshold barely affects cycle count because the
# timed region is dominated by popcount.
DEFAULT_THRESHOLD = 10000


def default_stem(max_timesteps, net_dt, prefix_ms, w, h):
    dt_ms = int(round(net_dt * 1000))
    return f"DVSGesture_T{max_timesteps}_dt{dt_ms}ms_p{prefix_ms}ms_{w}x{h}_binary"


def resolve_paths(args):
    stem = default_stem(args.max_timesteps, args.net_dt, args.prefix_ms, args.w, args.h)
    input_path = args.input or os.path.join(
        args.output_dir, f"dvsgesture_prefix_router_binary_input_{stem}.hex"
    )
    expected_path = args.expected or os.path.join(
        args.output_dir, f"dvsgesture_prefix_router_binary_expected_{stem}.csv"
    )
    output_path = args.output or os.path.join(
        args.output_dir, f"dvsgesture_prefix_router_binary_energy_{stem}.txt"
    )
    summary_path = args.summary or os.path.join(
        args.output_dir, f"dvsgesture_prefix_router_binary_energy_{stem}_summary.json"
    )
    return input_path, expected_path, output_path, summary_path


def load_dvsgesture_test_dataset(args):
    from datasets.dvsgesture_dataset import DVSGestureDataset

    def load_once(rebuild_cache):
        data = DVSGestureDataset(
            dataset_path=args.dataset_path,
            w=args.w,
            h=args.h,
            max_timesteps=args.max_timesteps,
            binarize=False,
            denoise_filter_time=args.denoise_filter_time,
            cache_tag=args.cache_tag,
            rebuild_cache=rebuild_cache,
        )
        return data.load_dvsgesture()

    _, cached_test = load_once(args.rebuild_cache)
    if len(cached_test) == 0 and not args.rebuild_cache:
        print(
            "Selected DVSGesture cache contains 0 test samples. "
            "Rebuilding that cache once..."
        )
        _, cached_test = load_once(True)

    if len(cached_test) == 0:
        raise RuntimeError(
            "DVSGesture test dataset has 0 samples after loading. "
            "Your local cache/dataset is empty or incomplete. Try rerunning with "
            "`--rebuild_cache`, and confirm data/DVSGesture/ contains the raw dataset."
        )
    return cached_test


def sample_to_packed_binary_prefix(sample, prefix_bins, h, w):
    import numpy as np

    arr = np.asarray(sample)
    if arr.ndim != 4:
        raise ValueError(f"Expected DVSGesture sample shape [T, 2, H, W], got {arr.shape}")
    if arr.shape[1:] != (2, h, w):
        raise ValueError(f"Expected [T, 2, {h}, {w}], got {arr.shape}")

    fixed = np.zeros((prefix_bins, 2, h, w), dtype=np.uint8)
    n_copy = min(prefix_bins, arr.shape[0])
    prefix_counts = arr[:n_copy]
    fixed[:n_copy] = prefix_counts > 0

    bits = fixed.reshape(-1)
    packed = np.packbits(bits)

    return {
        "packed": packed,
        "prefix_ones": int(bits.sum()),
        "prefix_count_sum": int(prefix_counts.sum()),
        "prefix_count_gt1": int((prefix_counts > 1).sum()),
        "prefix_bits": int(bits.size),
        "prefix_bins": int(prefix_bins),
    }


def write_binary_input_files(test_dataset, input_path, expected_path, args):
    prefix_bins = int(round((args.prefix_ms / 1000.0) / args.net_dt))
    os.makedirs(os.path.dirname(input_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(expected_path) or ".", exist_ok=True)

    print(f"Writing packed binary STM32 input: {input_path}")
    print(f"Writing expected CSV: {expected_path}")
    n = len(test_dataset)
    if args.max_samples is not None:
        n = min(n, args.max_samples)
    if n == 0:
        raise RuntimeError(
            "No DVSGesture samples available for STM32 input generation. "
            "The selected cache is empty; rerun with --rebuild_cache or fix dataset_path."
        )

    print(f"Samples: {n} | prefix bins: {prefix_bins}")

    dense_count = 0
    byte_lengths = []
    bit_lengths = []
    prefix_scores = []
    prefix_count_sums = []
    prefix_gt1 = []

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
                "binary_prefix_ones",
                "count_prefix_sum",
                "count_bins_gt1",
                "route_dense",
            ],
        )
        writer.writeheader()

        for idx in range(n):
            sample, label = test_dataset[idx]
            row = sample_to_packed_binary_prefix(sample, prefix_bins, args.h, args.w)
            packed = row["packed"]
            prefix_score = row["prefix_ones"]
            route_dense = int(prefix_score >= args.threshold)

            dense_count += route_dense
            byte_lengths.append(int(packed.size))
            bit_lengths.append(row["prefix_bits"])
            prefix_scores.append(prefix_score)
            prefix_count_sums.append(row["prefix_count_sum"])
            prefix_gt1.append(row["prefix_count_gt1"])

            in_fh.write(packed.tobytes().hex() + "\n")
            writer.writerow(
                {
                    "sample_idx": idx,
                    "label": int(label),
                    "prefix_ms": args.prefix_ms,
                    "prefix_bins": prefix_bins,
                    "prefix_bits": row["prefix_bits"],
                    "packed_bytes": int(packed.size),
                    "binary_prefix_ones": prefix_score,
                    "count_prefix_sum": row["prefix_count_sum"],
                    "count_bins_gt1": row["prefix_count_gt1"],
                    "route_dense": route_dense,
                }
            )

            if (idx + 1) % 50 == 0 or idx == n - 1:
                print(f"  wrote {idx + 1}/{n}")

    mean_prefix = sum(prefix_scores) / len(prefix_scores) if prefix_scores else 0.0
    mean_count_sum = sum(prefix_count_sums) / len(prefix_count_sums) if prefix_count_sums else 0.0
    mean_gt1 = sum(prefix_gt1) / len(prefix_gt1) if prefix_gt1 else 0.0
    print(f"Packed bytes/sample: {byte_lengths[0] if byte_lengths else 0}")
    print(f"Bits/sample: {bit_lengths[0] if bit_lengths else 0}")
    print(f"Mean binary prefix ones: {mean_prefix:.2f}")
    print(f"Mean count prefix sum: {mean_count_sum:.2f}")
    print(f"Mean count bins > 1: {mean_gt1:.2f}")
    print(f"Dense-route fraction at threshold {args.threshold}: {dense_count / len(prefix_scores):.3f}")


def load_expected(expected_path):
    expected = []
    with open(expected_path, newline="") as fh:
        for row in csv.DictReader(fh):
            expected.append(
                {
                    "binary_prefix_ones": int(row["binary_prefix_ones"]),
                    "route_dense": int(row["route_dense"]),
                    "packed_bytes": int(row["packed_bytes"]),
                    "prefix_bits": int(row["prefix_bits"]),
                    "count_prefix_sum": int(row["count_prefix_sum"]),
                    "count_bins_gt1": int(row["count_bins_gt1"]),
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
            prefix_score != exp["binary_prefix_ones"]
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
                exp["binary_prefix_ones"],
                exp["route_dense"],
                exp["count_prefix_sum"],
                exp["count_bins_gt1"],
            )
        )

        if (idx + 1) % 25 == 0 or idx == n - 1:
            print(
                f"  [{idx + 1}/{n}] cycles={cycles}, "
                f"binary_ones={prefix_score}, route_dense={route_dense}"
            )

    ser.write(b"DONE\n")
    sleep(0.5)
    ser.close()

    if mismatches:
        print(f"WARNING: {mismatches} mismatches.")
    else:
        print("All board outputs matched expected prefix scores/routes.")

    return results


def mean(values):
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarize(results, args, output_path, summary_path):
    valid = [r for r in results if r[0] > 0]
    if not valid:
        raise RuntimeError("No valid STM32 results.")

    cycles = [float(r[0]) for r in valid]
    binary_scores = [float(r[1]) for r in valid]
    route_dense = [float(r[2]) for r in valid]
    count_sums = [float(r[7]) for r in valid]
    count_gt1 = [float(r[8]) for r in valid]
    energies = [cycle * args.energy_per_cycle for cycle in cycles]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "measurement": "dvsgesture_packed_binary_prefix_router",
        "n_samples": len(results),
        "n_valid": len(valid),
        "router": {
            "metric": "binary_prefix_ones",
            "input_representation": "binary-clipped DVSGesture count frames, first prefix_ms",
            "threshold": args.threshold,
            "prefix_ms": args.prefix_ms,
            "w": args.w,
            "h": args.h,
            "max_timesteps": args.max_timesteps,
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
            "binary_ones_mean": float(mean(binary_scores)),
            "binary_ones_median": float(median(binary_scores)),
            "binary_ones_min": int(min(binary_scores)),
            "binary_ones_max": int(max(binary_scores)),
            "count_sum_mean": float(mean(count_sums)),
            "count_bins_gt1_mean": float(mean(count_gt1)),
            "dense_route_fraction": float(mean(route_dense)),
        },
        "paths": {
            "output": output_path,
            "summary": summary_path,
        },
    }

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nDVSGesture packed-binary STM32 prefix-router summary")
    print("=" * 72)
    print(f"Valid samples: {len(valid)} / {len(results)}")
    print(f"Mean cycles/sample: {mean(cycles):.2f}")
    print(f"Mean router energy: {mean(energies) * 1e6:.4f} uJ/sample")
    print(f"Mean binary prefix ones: {mean(binary_scores):.2f}")
    print(f"Mean count prefix sum: {mean(count_sums):.2f}")
    print(f"Summary JSON: {summary_path}")


def write_output(results, args, output_path, summary_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        for row in results:
            (
                cycles,
                prefix_score,
                route_dense,
                nbytes,
                nbits,
                exp_score,
                exp_route,
                count_sum,
                count_gt1,
            ) = row
            energy = cycles * args.energy_per_cycle if cycles > 0 else 0.0
            fh.write(
                f"{energy} {cycles} {prefix_score} {route_dense} "
                f"{nbytes} {nbits} {exp_score} {exp_route} {count_sum} {count_gt1}\n"
            )
    summarize(results, args, output_path, summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Measure DVSGesture packed-binary prefix router on STM32."
    )
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--input", default=None)
    parser.add_argument("--expected", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output_dir", default="Prefix_Router_Energy")
    parser.add_argument("--dataset_path", default="./data")
    parser.add_argument("--w", type=int, default=DEFAULT_W)
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--max_timesteps", type=int, default=DEFAULT_MAX_TIMESTEPS)
    parser.add_argument("--net_dt", type=float, default=DEFAULT_NET_DT)
    parser.add_argument("--prefix_ms", type=int, default=DEFAULT_PREFIX_MS)
    parser.add_argument("--denoise_filter_time", type=int, default=DEFAULT_DENOISE_FILTER_TIME)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--cache_tag", default=None)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--energy_per_cycle", type=float, default=ENERGY_PER_CYCLE_J)
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
        test_dataset = load_dvsgesture_test_dataset(args)
        write_binary_input_files(test_dataset, input_path, expected_path, args)

    if args.dry_run:
        print("\nDry-run complete. Flash DVS packed-binary STM32 firmware, then rerun without --dry-run.")
        print(f"Input:    {input_path}")
        print(f"Expected: {expected_path}")
        print("Note: The default 400 ms DVS prefix is 102400 packed bytes/sample.")
        return

    results = run_board(args, input_path, expected_path)
    write_output(results, args, output_path, summary_path)


if __name__ == "__main__":
    main()
