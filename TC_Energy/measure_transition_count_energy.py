"""
measure_transition_count_energy.py — Transition Count Energy Measurement Pipeline

Sends binarized spike data to the STM32 board over USB serial, collects hardware
DWT cycle counts + transition count scores, converts cycles to energy (Joules),
and writes results to an output file.

Usage (dry-run on Nautilus pod — generate input file to /workspace, no board needed):
    python measure_transition_count_energy.py --dry-run --output_dir /workspace

Usage (with pre-generated input file, from local machine with board connected):
    python measure_transition_count_energy.py --port COM3 --input tc_input_UCI_HAR.txt

Usage (generate input from UCI HAR, then measure):
    python measure_transition_count_energy.py --port COM3 --dataset_path ./data
"""

import os
import sys
import argparse
import numpy as np

# Add project root for imports
sys.path.insert(0, os.path.dirname(__file__))

from datasets.uci_har import UCIHARDataset

# ---------- constants ----------
INPUT_FILE  = "tc_input_UCI_HAR.txt"
ENERGY_TABLE = "tc_energy_UCI_HAR.txt"

# STM32L476 (Cortex-M4, low-power target): 180 pJ per cycle
ENERGY_PER_CYCLE = 1.8e-10  # joules


def load_test_data(dataset_path, n_frames=128):
    """Load UCI HAR test set with binarization, deterministic order."""
    ds = UCIHARDataset(
        dataset_path=dataset_path,
        n_frames=n_frames,
        time_first=True,
        normalize=True,
        binarize=True,
    )
    _, test_dataset = ds.load_uci_har()
    return test_dataset


def write_input_file(test_dataset, output_path):
    """Flatten each (128, 9) binarized sample to a 1152-char string of 0s/1s."""
    n = len(test_dataset)
    print(f"Writing {n} samples to {output_path}...")

    line = ""
    with open(output_path, "w") as f:
        for i in range(n):
            sample, _ = test_dataset[i]           # (128, 9) tensor
            flat = np.asarray(sample).reshape(-1)  # 1152 floats
            line = "".join(str(int(x)) for x in flat)
            f.write(line + "\n")

    print(f"  -> {output_path} written ({n} lines, {len(line)} chars each)")


def run_board(port, input_file, baud=115200, timeout=10):
    """Send each line of input file to STM32 over serial, collect results."""
    import serial
    from time import sleep

    with open(input_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    n = len(lines)
    results = []

    print(f"Connecting to {port}...")
    ser = serial.Serial(port, baud, timeout=timeout)
    sleep(2)  # wait for USB CDC to stabilise

    ser.reset_input_buffer()

    print(f"Sending {n} samples to STM32 (this takes ~{n * 0.03:.0f}s)...")

    for i, line in enumerate(lines):
        ser.write((line + "\n").encode())
        response = ser.readline().decode().strip()

        if not response:
            print(f"  WARNING: no response for sample {i}, retrying...")
            sleep(0.5)
            response = ser.readline().decode().strip()

        if not response:
            print(f"  ERROR: no response for sample {i}, skipping")
            results.append((0, -1))
            continue

        parts = response.split()
        if len(parts) != 2:
            print(f"  ERROR: unexpected response '{response}' for sample {i}")
            results.append((0, -1))
            continue

        cycles = int(parts[0])
        tc     = int(parts[1])
        results.append((cycles, tc))

        if (i + 1) % 500 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] last: {cycles} cycles, TC={tc}")

    ser.write(b"DONE\n")
    sleep(0.5)
    ser.close()

    return results


def write_energy_table(results, energy_per_cycle, output_path):
    """Convert cycles to Joules and write output file."""
    n = len(results)
    print(f"Writing {n} results to {output_path}...")

    with open(output_path, "w") as f:
        for cycles, tc in results:
            energy = cycles * energy_per_cycle  # Joules
            f.write(f"{energy} {cycles} {tc}\n")

    energies = [cycles * energy_per_cycle for cycles, tc in results if cycles > 0]
    if energies:
        print(f"  Energy range: {min(energies):.2e} — {max(energies):.2e} J")
        print(f"  Mean energy:  {np.mean(energies):.2e} J")
        print(f"  -> {output_path} written")


def main():
    parser = argparse.ArgumentParser(
        description="Transition Count Energy Measurement via STM32"
    )
    parser.add_argument("--port", type=str, default="COM3",
                        help="Serial port for STM32 (e.g. COM3 on Windows)")
    parser.add_argument("--input", type=str, default=None,
                        help="Pre-generated input file (skips dataset loading)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output energy table file")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to write input and energy files (default: cwd)")
    parser.add_argument("--dataset_path", type=str, default="./data",
                        help="Path to UCI HAR dataset root")
    parser.add_argument("--n_frames", type=int, default=128,
                        help="Number of time steps per sample")
    parser.add_argument("--energy_per_cycle", type=float, default=ENERGY_PER_CYCLE,
                        help="Energy per CPU cycle in Joules (default: 180 pJ for STM32L476)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only generate input file, don't communicate with board")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve input/output paths
    if args.input:
        input_path = args.input
    else:
        input_path = os.path.join(args.output_dir, INPUT_FILE)

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.output_dir, ENERGY_TABLE)

    if args.input:
        print(f"Using pre-generated input file: {input_path}")
    else:
        print("Loading UCI HAR test dataset (binarized)...")
        test_dataset = load_test_data(args.dataset_path, args.n_frames)
        print(f"  -> {len(test_dataset)} test samples loaded")
        write_input_file(test_dataset, input_path)

    if args.dry_run:
        print(f"\nDry-run mode: input file written to {input_path}")
        print("Copy this file to your local machine, then run with --input and --port.")
        return

    results = run_board(args.port, input_path)
    write_energy_table(results, args.energy_per_cycle, output_path)
    print("\nDone!")


if __name__ == "__main__":
    main()
