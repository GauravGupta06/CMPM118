"""Unified training script for Rockpool SNN models on UCI HAR dataset."""

import torch
import tonic
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from datasets.uci_har import load_uci_har
from models.uci_har_model import UCIHARSNN


def main():
    parser = argparse.ArgumentParser(description="Train Rockpool SNN on UCIHAR")
    parser.add_argument('--model_type', type=str, default='dense',
                        choices=['sparse', 'dense'])
    parser.add_argument('--n_frames', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--output_path', type=str, default='./results')

    args = parser.parse_args()

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print(f"Using device: {device}")
    print("=" * 60)

    # Load dataset
    print("\nLoading UCI HAR dataset...")
    train_ds, test_ds, num_classes = load_uci_har(
        dataset_path=args.dataset_path,
        n_frames=args.n_frames
    )

    print(f"Dataset loaded: {num_classes} classes")

    # Create dataloaders with GPU optimizations
    use_cuda = device.type == 'cuda'
    print(f"\nCreating dataloaders (num_workers={args.num_workers}, pin_memory={use_cuda})...")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=use_cuda
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda
    )

    # Model hyperparameters (sparse vs dense)
    # NOTE: spike_lam=0 for now to maximize accuracy
    if args.model_type == 'sparse':
        tau_mem = 0.02
        spike_lam = 1e-4
        hidden_size = 64
        print(f"\nTraining SPARSE model:")
        print(f"   - tau_mem: {tau_mem}")
        print(f"   - spike_lam: {spike_lam} (disabled)")
    else:  # dense
        tau_mem = 0.1
        spike_lam = 0.0
        hidden_size = 128
        print(f"\nTraining DENSE model:")
        print(f"   - tau_mem: {tau_mem}")
        print(f"   - spike_lam: {spike_lam} (disabled)")

    # Input size for SHD dataset (700 frequency bins)
    input_size = 700
    print(f"   - input_size: {input_size}")

    # Create model
    print(f"\nCreating UCIHARSNN model...")
    print(f"   - learning_rate: {args.lr}")

    model = UCIHARSNN(
    input_size=9,
    hidden_size=64 if args.model_type == "sparse" else 128,
    n_frames=args.n_frames,
    tau_mem=tau_mem,
    spike_lam=spike_lam,
    model_type=args.model_type,
    device=device,
    num_classes=6,
    lr=args.lr
    )

    print(f"\nModel architecture:")
    print(model.net)

    # Train
    print("\n" + "="*60)
    print(f"Training {args.model_type} model for {args.epochs} epochs...")
    print("="*60)

    model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        print_every=20
    )

    # Save
    print("\nSaving model...")
    model.save_model(base_path=args.output_path)

    # Final evaluation
    final_acc = model.validate_model(test_loader)
    print(f"\n{'='*60}")
    print(f"Final Test Accuracy: {final_acc * 100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
