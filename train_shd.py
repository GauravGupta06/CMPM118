"""Unified training script for Rockpool SNN models on SHD dataset."""
import torch
import tonic
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from datasets.shd_dataset import SHDDataset
from models.shd_model import SHDSNN_FC
from torch.utils.data import DataLoader



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
    parser.add_argument('--NUM_CHANNELS', type=int, default=700,
                        help='Number of channels')
    parser.add_argument('--NUM_POLARITIES', type=int, default=2,
                        help='Number of polarities')
    parser.add_argument('--net_dt', type=float, default=10e-3,
                        help='Time step')

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  

    # Load dataset
    data = SHDDataset(
        dataset_path=args.dataset_path,
        NUM_CHANNELS=args.NUM_CHANNELS, 
        NUM_POLARITIES=args.NUM_POLARITIES,
        n_frames=args.n_frames,
        net_dt=args.net_dt
    )
    
    cached_train, cached_test = data.load_shd()

    # Create dataloaders with GPU optimizations
    use_cuda = device.type == 'cuda'

    train_loader = DataLoader(
        cached_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True)
    )

    test_loader = DataLoader(
        cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True)
    )

    # Model hyperparameters (sparse vs dense)
    # NOTE: spike_lam=0 for now to maximize accuracy
    if args.model_type == 'sparse':
        tau_mem = 0.1
        spike_lam = 0.0
    else:  # dense
        tau_mem = 0.1
        spike_lam = 0.0
        print(f"   - spike_lam: {spike_lam} (disabled)")

    # Create model
    model = SHDSNN_FC(
        input_size=args.NUM_CHANNELS,
        n_frames=args.n_frames,
        tau_mem=tau_mem,
        spike_lam=spike_lam,
        model_type=args.model_type,
        device=device,
        num_classes=20,
        lr=args.lr,
        dt=args.net_dt,
        threshold=1.0,
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
    model.save_model(base_path=args.output_path)

    # Final evaluation
    final_acc = model.validate_model(test_loader)
    print(f"\n{'='*60}")
    print(f"Final Test Accuracy: {final_acc * 100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
