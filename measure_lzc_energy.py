"""
measure_lzc_energy.py — LZC Energy Measurement Pipeline

Loads UCI HAR test data (binarized), sends each sample to the STM32 board
over USB serial, collects hardware DWT cycle counts + LZC scores, converts
cycles to energy (Joules), and writes results to an output file.

Usage (on Windows, with board connected):
    python measure_lzc_energy.py --port COM3

Dry-run (no board needed, just generates input file):
    python measure_lzc_energy.py --dry-run
"""

import os
import sys
import argparse
import numpy as np

# Add project root for imports
sys.path.insert(0, os.path.dirname(__file__))

from datasets.uci_har import UCIHARDataset

# ---------- constants ----------
INPUT_FILE = "lzc_input_UCI_HAR.txt"
ENERGY_TABLE = "lzc_energy_UCI_HAR.txt"

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


def write_input_file(test_dataset):
    """Flatten each (128, 9) binarized sample to a 1152-char string of 0s/1s."""
    n = len(test_dataset)
    print(f"Writing {n} samples to {INPUT_FILE}...")

    with open(INPUT_FILE, "w") as f:
        for i in range(n):
            sample, _ = test_dataset[i]           # (128, 9) tensor
            flat = np.asarray(sample).reshape(-1)  # 1152 floats
            line = "".join(str(int(x)) for x in flat)
            f.write(line + "\n")

    print(f"  → {INPUT_FILE} written ({n} lines, {len(line)} chars each)")


def run_board(port, baud=115200, timeout=10):
    """Send each line of lzc_input.txt to STM32 over serial, collect results."""
    import serial
    from time import sleep

    # Read input lines
    with open(INPUT_FILE) as f:
        lines = [l.strip() for l in f if l.strip()]

    n = len(lines)
    results = []

    print(f"Connecting to {port}...")
    ser = serial.Serial(port, baud, timeout=timeout)
    sleep(2)  # wait for USB CDC to stabilise

    # Drain any buffered data (e.g. READY message)
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
        lzc = int(parts[1])
        results.append((cycles, lzc))

        # Progress update
        if (i + 1) % 500 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] last: {cycles} cycles, LZC={lzc}")

    # Signal completion
    ser.write(b"DONE\n")
    sleep(0.5)
    ser.close()

    return results


def write_energy_table(results, energy_per_cycle):
    """Convert cycles to Joules and write output file."""
    n = len(results)
    print(f"Writing {n} results to {ENERGY_TABLE}...")

    with open(ENERGY_TABLE, "w") as f:
        for cycles, lzc in results:
            energy = cycles * energy_per_cycle  # Joules
            f.write(f"{energy} {lzc}\n")

    # Summary statistics
    energies = [cycles * energy_per_cycle for cycles, lzc in results if cycles > 0]
    if energies:
        print(f"  Energy range: {min(energies):.2e} — {max(energies):.2e} J")
        print(f"  Mean energy:  {np.mean(energies):.2e} J")
        print(f"  → {ENERGY_TABLE} written")


def main():
    parser = argparse.ArgumentParser(description="LZC Energy Measurement via STM32")
    parser.add_argument("--port", type=str, default="COM3",
                        help="Serial port for STM32 (e.g. COM3 on Windows)")
    parser.add_argument("--dataset_path", type=str, default="./data",
                        help="Path to UCI HAR dataset")
    parser.add_argument("--n_frames", type=int, default=128,
                        help="Number of time steps per sample")
    parser.add_argument("--energy_per_cycle", type=float, default=ENERGY_PER_CYCLE,
                        help="Energy per CPU cycle in Joules (default: STM32L476 @ 180pJ)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only generate input file, don't communicate with board")
    args = parser.parse_args()

    # Step 1: Load dataset
    print("Loading UCI HAR test dataset (binarized)...")
    test_dataset = load_test_data(args.dataset_path, args.n_frames)
    print(f"  → {len(test_dataset)} test samples loaded")

    # Step 2: Write input file
    write_input_file(test_dataset)

    if args.dry_run:
        print("\nDry-run mode: stopping before board communication.")
        print(f"Verify {INPUT_FILE} has the expected content.")
        return

    # Step 3: Send to board, collect results
    results = run_board(args.port)

    # Step 4: Write energy table
    write_energy_table(results, args.energy_per_cycle)

    print("\nDone!")


if __name__ == "__main__":
    main()