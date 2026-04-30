"""Training script for DVSGesture dataset.

Uses the conv architecture from Arfa et al. (2025) with paper-matched hyperparameters.
"""
import torch
import torch.multiprocessing
import argparse
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))

from datasets.dvsgesture_dataset import DVSGestureDataset
from models.dvsgesture_model import DVSGestureSNN
from torch.utils.data import DataLoader
import tonic


def main():
    parser = argparse.ArgumentParser(description="Train Rockpool Conv SNN on DVSGesture (Arfa et al. 2025)")
    parser.add_argument('--model_type', type=str, default='dense', choices=['sparse', 'dense', 'reference'],
                        help='Model type: sparse (spike reg), dense (less reg), or reference (no reg, paper-exact)')
    parser.add_argument('--w', type=int, default=32, help='Width')
    parser.add_argument('--h', type=int, default=32, help='Height')
    parser.add_argument('--max_timesteps', type=int, default=600, help='Max timesteps (1ms bins)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--threshold', type=float, default=1.0, help='LIF firing threshold')
    parser.add_argument('--spike_lam', type=float, default=None, help='Override spike_lam (default: auto from model_type)')
    parser.add_argument('--tau_mem', type=float, default=None, help='Override tau_mem (default: 0.01378 from beta=0.93)')
    parser.add_argument('--dataset_path', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--output_path', type=str, default='./results', help='Path to save the trained model')

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset
    data = DVSGestureDataset(
        dataset_path=args.dataset_path,
        w=args.w,
        h=args.h,
        max_timesteps=args.max_timesteps,
    )
    cached_train, cached_test = data.load_dvsgesture()

    # Create dataloaders — PadTensors required for variable-length sequences
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

    # Model hyperparameters
    tau_mem = args.tau_mem if args.tau_mem is not None else -0.001 / math.log(0.93)

    if args.spike_lam is not None:
        spike_lam = args.spike_lam
    elif args.model_type == 'sparse':
        spike_lam = 1e-5
    elif args.model_type == 'reference':
        spike_lam = 0.0
    else:
        spike_lam = 0.0

    print(f"\nHyperparameters:")
    print(f"  model_type: {args.model_type}")
    print(f"  tau_mem:    {tau_mem:.5f}")
    print(f"  threshold:  {args.threshold}")
    print(f"  spike_lam:  {spike_lam}")
    print(f"  lr:         {args.lr}")
    print(f"  epochs:     {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  max_timesteps: {args.max_timesteps}")

    # Create model — conv architecture, no tau_syn, no input_size needed
    model = DVSGestureSNN(
        input_size=None,
        n_frames=args.max_timesteps,
        tau_mem=tau_mem,
        spike_lam=spike_lam,
        model_type=args.model_type,
        device=device,
        num_classes=data.get_num_classes(),
        lr=args.lr,
        dt=0.001,
        threshold=args.threshold,
        has_bias=False,
    )

    total_params = sum(p.numel() for p in model.parameters() if 'conv' in str(p) or 'fc' in str(p))
    print(f"  Model architecture: {model.architecture}")
    print(f"  Total weight params: {sum(p.numel() for n, p in model.named_parameters() if 'conv' in n or 'fc' in n)}")

    # Train
    model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        print_every=15
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
