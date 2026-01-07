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
    parser.add_argument('--reduce_to_16', action='store_true',
                        help='Reduce 700 → 16 features (Xylo-compatible)')
    parser.add_argument('--n_frames', type=int, default=100,
                        help='Number of time steps')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--dataset_path', type=str, default='./data',
                        help='Path to dataset directory')

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("=" * 60)
    print(f"Using device: {device}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading SHD dataset (reduce_to_16={args.reduce_to_16})...")
    cached_train, cached_test, num_classes = load_shd(
        dataset_path=args.dataset_path,
        n_frames=args.n_frames,
        reduce_to_16=args.reduce_to_16
    )
    print(f"Dataset loaded: {num_classes} classes")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = torch.utils.data.DataLoader(
        cached_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=7,
        drop_last=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    test_loader = torch.utils.data.DataLoader(
        cached_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=7,
        drop_last=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    # Model hyperparameters (sparse vs dense)
    if args.model_type == 'sparse':
        tau_mem = 0.01
        spike_lam = 1e-6
        print(f"\n🔥 Training SPARSE model:")
        print(f"   - tau_mem: {tau_mem} (faster decay, less integration)")
        print(f"   - spike_lam: {spike_lam} (strong spike penalty)")
    else:  # dense
        tau_mem = 0.02
        spike_lam = 1e-8
        print(f"\n🔥 Training DENSE model:")
        print(f"   - tau_mem: {tau_mem} (slower decay, more integration)")
        print(f"   - spike_lam: {spike_lam} (weak spike penalty)")

    # Input size depends on binning
    input_size = 16 if args.reduce_to_16 else 700
    print(f"   - input_size: {input_size}")

    # Create model
    print(f"\nCreating SHDSNN_FC model...")
    model = SHDSNN_FC(
        input_size=input_size,
        n_frames=args.n_frames,
        tau_mem=tau_mem,
        spike_lam=spike_lam,
        model_type=args.model_type,
        device=device,
        num_classes=num_classes
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
    model.save_model(base_path="../results")

    # Final evaluation
    final_acc = model.validate_model(test_loader)
    print(f"\n{'='*60}")
    print(f"Final Test Accuracy: {final_acc * 100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
