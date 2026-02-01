"""Unified training script for Rockpool SNN models on UCI HAR dataset."""

import torch
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from datasets.uci_har import UCIHARDataset
from models.uci_har_model import UCIHARSNN_FC
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Train Rockpool SNN on UCI HAR")

    parser.add_argument('--model_type', type=str, default='dense', choices=['sparse', 'dense'],
                        help='Model type: sparse (fewer spikes) or dense (more spikes)')

    parser.add_argument('--n_frames', type=int, default=128,
                        help='Number of time steps (UCI HAR windows are typically 128)')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default 0.001)')

    parser.add_argument('--dataset_path', type=str, default='./data',
                        help='Path to dataset directory (should contain "UCI HAR Dataset/")')

    parser.add_argument('--output_path', type=str, default='./results',
                        help='Path to save the trained model')

    # UCI HAR specifics
    parser.add_argument('--NUM_CHANNELS', type=int, default=9,
                        help='Number of input channels (UCI HAR has 9 inertial channels)')

    parser.add_argument('--net_dt', type=float, default=0.02,
                        help='Time step in seconds (UCI HAR is ~50Hz => dt=0.02)')

    parser.add_argument('--normalize', action='store_true',
                        help='Apply per-sample z-score normalization (recommended)')

    args = parser.parse_args()

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")      
    elif torch.cuda.is_available():
        device = torch.device("cuda")     
    else:
        device = torch.device("cpu")

    # Load dataset
    data = UCIHARDataset(
        dataset_path=args.dataset_path,
        n_frames=args.n_frames,
        time_first=True,       # keep [T, C] internally
        normalize=args.normalize
    )

    cached_train, cached_test = data.load_uci_har()

    # DataLoaders
    # NOTE:
    # - Unlike SHD, UCI HAR samples are fixed length (usually 128), so no tonic padding collation needed.
    # - We still stack to match BaseSNNModel expectation of [T, B, C] later in training.
    train_loader = DataLoader(
        cached_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        cached_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Hyperparameters (sparse vs dense)
    # With continuous input, sparsity is mainly controlled by tau_mem / threshold / spike_lam
    if args.model_type == 'sparse':
        tau_mem = 0.08
        spike_lam = 5e-4
    else:  # dense
        tau_mem = 0.1
        spike_lam = 0.0
        print(f"   - spike_lam: {spike_lam} (disabled)")

    # Create model
    model = UCIHARSNN_FC(
        input_size=args.NUM_CHANNELS,
        n_frames=args.n_frames,
        tau_mem=tau_mem,
        tau_syn=0.07, #changed from 0.05 to 0.07
        spike_lam=spike_lam,
        model_type=args.model_type,
        device=device,
        num_classes=6,
        lr=args.lr,
        dt=args.net_dt,
        threshold=0.9, #changed from 1.0 to 0.9
        has_bias=True
    )

    # Train
    model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        print_every=20
    )

    # Save
    print("\nSaving model...")
    save_dir = os.path.join(args.output_path, args.model_type)
    model.save_model(base_path=save_dir)

    # Final evaluation
    final_acc = model.validate_model(test_loader) 
    print(f"\n{'='*60}")
    print(f"Final Test Accuracy: {final_acc * 100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
