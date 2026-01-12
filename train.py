"""Unified training script for Rockpool SNN models on SHD dataset."""

import torch
import tonic
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_shd
from models import SHDSNN_FC


def main():
    parser = argparse.ArgumentParser(description="Train Rockpool SNN on SHD")
    parser.add_argument('--model_type', type=str, default='dense', choices=['sparse', 'dense'],
                        help='Model type: sparse (fewer spikes) or dense (more spikes)')
    parser.add_argument('--n_frames', type=int, default=100,
                        help='Number of time steps')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (try 128+ for A10 GPU)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default 0.001)')
    parser.add_argument('--dataset_path', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--output_path', type=str, default='./results',
                        help='Path to save the trained model (use /workspace for Kubernetes PVC)')

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("=" * 60)
    print(f"Using device: {device}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading SHD dataset...")
    cached_train, cached_test, num_classes = load_shd(
        dataset_path=args.dataset_path,
        n_frames=args.n_frames
    )
    print(f"Dataset loaded: {num_classes} classes")

    # Create dataloaders with GPU optimizations
    use_cuda = device.type == 'cuda'
    print(f"\nCreating dataloaders (num_workers={args.num_workers}, pin_memory={use_cuda})...")
    train_loader = torch.utils.data.DataLoader(
        cached_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    test_loader = torch.utils.data.DataLoader(
        cached_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    # Model hyperparameters (sparse vs dense)
    # NOTE: spike_lam=0 for now to maximize accuracy
    if args.model_type == 'sparse':
        tau_mem = 0.1
        spike_lam = 0.0
        print(f"\nTraining SPARSE model:")
        print(f"   - tau_mem: {tau_mem}")
        print(f"   - spike_lam: {spike_lam} (disabled)")
    else:  # dense
        tau_mem = 0.1
        spike_lam = 0.0
        print(f"\nTraining DENSE model:")
        print(f"   - tau_mem: {tau_mem}")
        print(f"   - spike_lam: {spike_lam} (disabled)")

    # Input size for SHD dataset (700 frequency bins)
    input_size = 700
    print(f"   - input_size: {input_size}")

    # Create model
    print(f"\nCreating SHDSNN_FC model...")
    print(f"   - learning_rate: {args.lr}")
    model = SHDSNN_FC(
        input_size=input_size,
        n_frames=args.n_frames,
        tau_mem=tau_mem,
        spike_lam=spike_lam,
        model_type=args.model_type,
        device=device,
        num_classes=num_classes,
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
