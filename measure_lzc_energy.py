"""
LZC Energy Measurement - Data Preparation Script

This script prepares UCI HAR dataset samples for LZC energy measurement.
It loads the dataset in a fixed deterministic order, binarizes and flattens
each sample, and writes them to input files for the C LZC implementation.

Steps 1-4 of the measurement pipeline:
1. Load dataset in fixed deterministic order
2. For each sample, binarize (>0 threshold)
3. Flatten time-major: [t0_ch0, t0_ch1, ..., t0_ch8, t1_ch0, ..., t127_ch8]
4. Write to input file (one line per sample, one for train, one for test)

Usage:
    python measure_lzc_energy.py --dataset_path ./data --output_dir ./data

Output files:
    - data/lzc_inputs_train.txt  (flattened binary strings, one per line)
    - data/lzc_inputs_test.txt   (flattened binary strings, one per line)
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.uci_har import UCIHARDataset


def sample_to_binary_string(sample: torch.Tensor) -> str:
    """
    Convert a pre-binarized sample to a binary string for LZC computation.

    Flattens in time-major order: [t0_ch0, t0_ch1, ..., t0_chC-1, t1_ch0, ...]

    Args:
        sample: Tensor of shape [T, C], already binarized (0s and 1s)

    Returns:
        String of '0' and '1' characters
    """
    if torch.is_tensor(sample):
        sample = sample.cpu().numpy()
    # Flatten in time-major order (row-major for [T, C])
    flattened = sample.astype(int).flatten()
    return ''.join(str(x) for x in flattened)


def process_dataset(dataset, output_path: str, split_name: str):
    """
    Process entire dataset and write to output file.

    Args:
        dataset: Dataset to process
        output_path: Path to output file
        split_name: Name for logging (e.g., "train", "test")
    """
    print(f"Processing {split_name} dataset ({len(dataset)} samples)...")

    # Use DataLoader with shuffle=False for deterministic order
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Single-threaded for deterministic order
        drop_last=False
    )

    with open(output_path, 'w') as f:
        for idx, (sample, label) in enumerate(dataloader):
            # sample shape: [1, T, C] -> squeeze to [T, C]
            sample = sample.squeeze(0)

            # Convert to binary string
            binary_string = sample_to_binary_string(sample)

            # Write line
            f.write(binary_string + '\n')

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} samples...")

    print(f"  Wrote {len(dataset)} samples to {output_path}")

    # Verify line count
    with open(output_path, 'r') as f:
        line_count = sum(1 for _ in f)
    print(f"  Verified: {line_count} lines in output file")

    # Print sample info
    sample, _ = dataset[0]
    if torch.is_tensor(sample):
        sample = sample.numpy()
    print(f"  Sample shape: {sample.shape}")
    print(f"  Flattened length: {sample.size}")
    print(f"  Binary string length: {len(sample_to_binary_string(torch.tensor(sample)))}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare UCI HAR data for LZC energy measurement"
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./data',
        help='Path to dataset directory (default: ./data)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Directory for output files (default: ./data)'
    )
    parser.add_argument(
        '--n_frames',
        type=int,
        default=128,
        help='Number of time frames (default: 128)'
    )

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Output file paths
    train_output = os.path.join(args.output_dir, 'lzc_inputs_train.txt')
    test_output = os.path.join(args.output_dir, 'lzc_inputs_test.txt')

    print("="*60)
    print("LZC Energy Measurement - Data Preparation")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"N frames: {args.n_frames}")
    print()

    # Load UCI HAR dataset with binarization
    print("Loading UCI HAR dataset...")
    loader = UCIHARDataset(
        dataset_path=args.dataset_path,
        n_frames=args.n_frames,
        time_first=True,   # Shape: [T, C]
        normalize=True,    # Z-score normalize (same as training)
        binarize=True      # Binarize: values > 0 become 1, else 0
    )

    train_dataset, test_dataset = loader.load_uci_har()

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()

    # Process train dataset
    process_dataset(train_dataset, train_output, "train")
    print()

    # Process test dataset
    process_dataset(test_dataset, test_output, "test")
    print()

    print("="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"Train input file: {train_output}")
    print(f"Test input file: {test_output}")
    print()
    print("Next steps:")
    print("  1. Compile lzc.c: gcc -o lzc lzc.c")
    print("  2. Run with QEMU for energy measurement")
    print("  3. Energy output files will be:")
    print(f"     - {args.output_dir}/lzc_energy_train.txt")
    print(f"     - {args.output_dir}/lzc_energy_test.txt")


if __name__ == "__main__":
    main()
