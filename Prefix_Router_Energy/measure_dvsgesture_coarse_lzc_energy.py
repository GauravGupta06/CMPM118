"""
measure_dvsgesture_coarse_lzc_energy.py

STM32 measurement for the DVSGesture coarse-LZC router metric.

Metric definition:
  1. Use the first 400 ms of the DVSGesture count-frame input.
  2. Convert to binary occupancy: count > 0.
  3. Pool 400 x 1 ms frames into 20 coarse time bins of 20 ms each.
  4. Pool each 32x32 frame into a 4x4 spatial grid, keeping polarity separate.
  5. Each coarse cell is binary: 1 if any event occurred, else 0.
  6. Flatten [20 time bins, 2 polarities, 4, 4] into 640 bits.
  7. Compute Lempel-Ziv complexity over those 640 bits.

The STM32 receives the same full packed binary prefix used by the popcount
router:

    400 * 2 * 32 * 32 = 819200 bits = 102400 packed bytes/sample

USB transfer and hex parsing happen outside the timed region. The timed region
builds the coarse occupancy map, computes LZC, and applies the threshold.

Usage:
    python Prefix_Router_Energy/measure_dvsgesture_coarse_lzc_energy.py --dry-run
    python Prefix_Router_Energy/measure_dvsgesture_coarse_lzc_energy.py --port COM3
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
DEFAULT_TIME_BINS = 20
DEFAULT_GRID = 4
DEFAULT_DENOISE_FILTER_TIME = 10000

# Offline sweep on the current DVS dense/sparse pair found that routing dense
# when coarse_lzc640 <= 53 keeps routed accuracy close to dense. The score file
# can still be threshold-swept later, so this only affects the STM32 route flag.
DEFAULT_THRESHOLD = 53


def default_stem(max_timesteps, net_dt, prefix_ms, time_bins, grid):
    dt_ms = int(round(net_dt * 1000))
    return (
        f"DVSGesture_T{max_timesteps}_dt{dt_ms}ms_p{prefix_ms}ms_"
        f"{time_bins}tb_{grid}x{grid}"
    )


def resolve_paths(args):
    stem = default_stem(
        args.max_timesteps,
        args.net_dt,
        args.prefix_ms,
        args.time_bins,
        args.grid,
    )
    input_path = args.input or os.path.join(
        args.output_dir, f"dvsgesture_coarse_lzc_input_{stem}.hex"
    )
    expected_path = args.expected or os.path.join(
        args.output_dir, f"dvsgesture_coarse_lzc_expected_{stem}.csv"
    )
    output_path = args.output or os.path.join(
        args.output_dir, f"dvsgesture_coarse_lzc_energy_{stem}.txt"
    )
    summary_path = args.summary or os.path.join(
        args.output_dir, f"dvsgesture_coarse_lzc_energy_{stem}_summary.json"
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
            "Try rerunning with --rebuild_cache and confirm data/DVSGesture has "
            "the raw dataset archives."
        )
    return cached_test


def lzc_binary(seq):
    """LZ76/Kaspar-Schuster complexity for a short binary sequence."""
    s = "".join("1" if int(x) else "0" for x in seq)
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


def sample_to_metric_row(sample, prefix_bins, h, w, time_bins, grid):
    import numpy as np

    arr = np.asarray(sample)
    if arr.ndim != 4:
        raise ValueError(f"Expected DVSGesture sample shape [T, 2, H, W], got {arr.shape}")
    if arr.shape[1:] != (2, h, w):
        raise ValueError(f"Expected [T, 2, {h}, {w}], got {arr.shape}")
    if prefix_bins % time_bins != 0:
        raise ValueError("prefix_bins must divide evenly by time_bins")
    if h != w or h % grid != 0:
        raise ValueError("Only square frames with h == w and grid dividing h are supported")

    fixed = np.zeros((prefix_bins, 2, h, w), dtype=np.uint8)
    n_copy = min(prefix_bins, arr.shape[0])
    prefix_counts = arr[:n_copy]
    fixed[:n_copy] = prefix_counts > 0

    bits = fixed.reshape(-1)
    packed = np.packbits(bits)

    frames_per_bin = prefix_bins // time_bins
    pixels_per_cell = h // grid
    coarse = fixed.reshape(
        time_bins,
        frames_per_bin,
        2,
        grid,
        pixels_per_cell,
        grid,
        pixels_per_cell,
    ).any(axis=(1, 4, 6))
    coarse_bits = coarse.reshape(-1).astype(np.uint8)

    return {
        "packed": packed,
        "coarse_lzc640": int(lzc_binary(coarse_bits)),
        "coarse_active_cells": int(coarse_bits.sum()),
        "binary_prefix_ones": int(bits.sum()),
        "count_prefix_sum": int(prefix_counts.sum()),
        "count_bins_gt1": int((prefix_counts > 1).sum()),
        "prefix_bits": int(bits.size),
        "prefix_bins": int(prefix_bins),
        "coarse_bits": int(coarse_bits.size),
    }


def write_input_files(test_dataset, input_path, expected_path, args):
    prefix_bins = int(round((args.prefix_ms / 1000.0) / args.net_dt))
    os.makedirs(os.path.dirname(input_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(expected_path) or ".", exist_ok=True)

    print(f"Writing packed binary STM32 input: {input_path}")
    print(f"Writing expected CSV: {expected_path}")
    n = len(test_dataset)
    if args.max_samples is not None:
        n = min(n, args.max_samples)
    if n == 0:
        raise RuntimeError("No DVSGesture samples available for STM32 input generation.")

    print(f"Samples: {n} | prefix bins: {prefix_bins}")

    lzc_scores = []
    active_cells = []
    binary_ones = []
    count_sums = []
    count_gt1 = []
    route_dense_count = 0
    byte_lengths = []
    bit_lengths = []

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
                "time_bins",
                "grid",
                "coarse_bits",
                "coarse_lzc640",
                "coarse_active_cells",
                "binary_prefix_ones",
                "count_prefix_sum",
                "count_bins_gt1",
                "route_dense",
            ],
        )
        writer.writeheader()

        for idx in range(n):
            sample, label = test_dataset[idx]
            row = sample_to_metric_row(
                sample,
                prefix_bins,
                args.h,
                args.w,
                args.time_bins,
                args.grid,
            )
            packed = row["packed"]
            route_dense = int(row["coarse_lzc640"] <= args.threshold)

            lzc_scores.append(row["coarse_lzc640"])
            active_cells.append(row["coarse_active_cells"])
            binary_ones.append(row["binary_prefix_ones"])
            count_sums.append(row["count_prefix_sum"])
            count_gt1.append(row["count_bins_gt1"])
            route_dense_count += route_dense
            byte_lengths.append(int(packed.size))
            bit_lengths.append(row["prefix_bits"])

            in_fh.write(packed.tobytes().hex() + "\n")
            writer.writerow(
                {
                    "sample_idx": idx,
                    "label": int(label),
                    "prefix_ms": args.prefix_ms,
                    "prefix_bins": prefix_bins,
                    "prefix_bits": row["prefix_bits"],
                    "packed_bytes": int(packed.size),
                    "time_bins": args.time_bins,
                    "grid": args.grid,
                    "coarse_bits": row["coarse_bits"],
                    "coarse_lzc640": row["coarse_lzc640"],
                    "coarse_active_cells": row["coarse_active_cells"],
                    "binary_prefix_ones": row["binary_prefix_ones"],
                    "count_prefix_sum": row["count_prefix_sum"],
                    "count_bins_gt1": row["count_bins_gt1"],
                    "route_dense": route_dense,
                }
            )

            if (idx + 1) % 50 == 0 or idx == n - 1:
                print(f"  wrote {idx + 1}/{n}")

    print(f"Packed bytes/sample: {byte_lengths[0] if byte_lengths else 0}")
    print(f"Bits/sample: {bit_lengths[0] if bit_lengths else 0}")
    print(f"Coarse bits/sample: {args.time_bins * 2 * args.grid * args.grid}")
    print(f"Mean coarse LZC: {mean(lzc_scores):.2f}")
    print(f"Mean coarse active cells: {mean(active_cells):.2f}")
    print(f"Mean binary prefix ones: {mean(binary_ones):.2f}")
    print(f"Mean count prefix sum: {mean(count_sums):.2f}")
    print(f"Mean count bins > 1: {mean(count_gt1):.2f}")
    print(f"Dense-route fraction at LZC <= {args.threshold}: {route_dense_count / len(lzc_scores):.3f}")


def load_expected(expected_path):
    expected = []
    with open(expected_path, newline="") as fh:
        for row in csv.DictReader(fh):
            expected.append(
                {
                    "coarse_lzc640": int(row["coarse_lzc640"]),
                    "coarse_active_cells": int(row["coarse_active_cells"]),
                    "route_dense": int(row["route_dense"]),
                    "packed_bytes": int(row["packed_bytes"]),
                    "prefix_bits": int(row["prefix_bits"]),
                    "binary_prefix_ones": int(row["binary_prefix_ones"]),
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

    if len(lines) != len(expected):
        raise RuntimeError(f"Input has {len(lines)} rows but expected CSV has {len(expected)} rows")

    print(f"Connecting to STM32 on {args.port}...")
    ser = serial.Serial(args.port, args.baud, timeout=args.timeout, write_timeout=args.timeout)
    sleep(2)
    ser.reset_input_buffer()

    results = []
    mismatches = 0
    print(f"Sending {len(lines)} packed-binary samples...")

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
        if len(parts) != 7:
            raise RuntimeError(f"Unexpected RUN response for sample {idx}: {response!r}")

        cycles = int(parts[0])
        coarse_lzc = int(parts[1])
        coarse_active = int(parts[2])
        route_dense = int(parts[3])
        returned_nbytes = int(parts[4])
        returned_nbits = int(parts[5])
        binary_ones_returned = int(parts[6])

        if (
            coarse_lzc != exp["coarse_lzc640"]
            or coarse_active != exp["coarse_active_cells"]
            or route_dense != exp["route_dense"]
            or returned_nbytes != exp["packed_bytes"]
            or returned_nbits != exp["prefix_bits"]
            or binary_ones_returned != exp["binary_prefix_ones"]
        ):
            mismatches += 1
            print(
                f"  MISMATCH sample {idx}: got "
                f"lzc={coarse_lzc}, active={coarse_active}, route={route_dense}, "
                f"nbytes={returned_nbytes}, nbits={returned_nbits}, ones={binary_ones_returned}; "
                f"expected {exp}"
            )

        results.append(
            (
                cycles,
                coarse_lzc,
                coarse_active,
                route_dense,
                returned_nbytes,
                returned_nbits,
                binary_ones_returned,
                exp["coarse_lzc640"],
                exp["coarse_active_cells"],
                exp["route_dense"],
                exp["count_prefix_sum"],
                exp["count_bins_gt1"],
            )
        )

        if (idx + 1) % 25 == 0 or idx == len(lines) - 1:
            print(
                f"  [{idx + 1}/{len(lines)}] cycles={cycles}, "
                f"coarse_lzc640={coarse_lzc}, active_cells={coarse_active}, "
                f"route_dense={route_dense}"
            )

    ser.write(b"DONE\n")
    sleep(0.5)
    ser.close()

    if mismatches:
        print(f"WARNING: {mismatches} mismatches.")
    else:
        print("All board outputs matched expected coarse-LZC scores/routes.")

    return results


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarize(results, args, output_path, summary_path):
    valid = [r for r in results if r[0] > 0]
    if not valid:
        raise RuntimeError("No valid STM32 results.")

    cycles = [float(r[0]) for r in valid]
    lzc_scores = [float(r[1]) for r in valid]
    active_cells = [float(r[2]) for r in valid]
    route_dense = [float(r[3]) for r in valid]
    binary_ones = [float(r[6]) for r in valid]
    count_sums = [float(r[10]) for r in valid]
    count_gt1 = [float(r[11]) for r in valid]
    energies = [cycle * args.energy_per_cycle for cycle in cycles]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "measurement": "dvsgesture_coarse_lzc640_prefix_router",
        "n_samples": len(results),
        "n_valid": len(valid),
        "router": {
            "metric": "coarse_lzc640",
            "input_representation": "binary-clipped DVSGesture count frames, first prefix_ms",
            "route_rule": "dense if coarse_lzc640 <= threshold",
            "threshold": args.threshold,
            "prefix_ms": args.prefix_ms,
            "time_bins": args.time_bins,
            "grid": args.grid,
            "coarse_bits": args.time_bins * 2 * args.grid * args.grid,
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
        "scores": {
            "coarse_lzc640_mean": float(mean(lzc_scores)),
            "coarse_lzc640_median": float(median(lzc_scores)),
            "coarse_lzc640_min": int(min(lzc_scores)),
            "coarse_lzc640_max": int(max(lzc_scores)),
            "coarse_active_cells_mean": float(mean(active_cells)),
            "binary_prefix_ones_mean": float(mean(binary_ones)),
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

    print("\nDVSGesture coarse-LZC STM32 prefix-router summary")
    print("=" * 72)
    print(f"Valid samples: {len(valid)} / {len(results)}")
    print(f"Mean cycles/sample: {mean(cycles):.2f}")
    print(f"Mean router energy: {mean(energies) * 1e6:.4f} uJ/sample")
    print(f"Mean coarse LZC: {mean(lzc_scores):.2f}")
    print(f"Mean coarse active cells: {mean(active_cells):.2f}")
    print(f"Dense-route fraction: {mean(route_dense):.3f}")
    print(f"Summary JSON: {summary_path}")


def write_output(results, args, output_path, summary_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        for row in results:
            (
                cycles,
                coarse_lzc,
                coarse_active,
                route_dense,
                nbytes,
                nbits,
                binary_ones,
                exp_lzc,
                exp_active,
                exp_route,
                count_sum,
                count_gt1,
            ) = row
            energy = cycles * args.energy_per_cycle if cycles > 0 else 0.0
            fh.write(
                f"{energy} {cycles} {coarse_lzc} {coarse_active} {route_dense} "
                f"{nbytes} {nbits} {binary_ones} "
                f"{exp_lzc} {exp_active} {exp_route} {count_sum} {count_gt1}\n"
            )
    summarize(results, args, output_path, summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Measure DVSGesture coarse-LZC prefix router on STM32."
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
    parser.add_argument("--time_bins", type=int, default=DEFAULT_TIME_BINS)
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID)
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
        write_input_files(test_dataset, input_path, expected_path, args)

    if args.dry_run:
        print("\nDry-run complete. Flash DVS coarse-LZC STM32 firmware, then rerun without --dry-run.")
        print(f"Input:    {input_path}")
        print(f"Expected: {expected_path}")
        print("Note: The default 400 ms DVS prefix is 102400 packed bytes/sample.")
        return

    results = run_board(args, input_path, expected_path)
    write_output(results, args, output_path, summary_path)


if __name__ == "__main__":
    main()
