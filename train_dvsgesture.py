"""Training script for DVSGesture dataset."""
import torch
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from datasets.dvsgesture_dataset import DVSGestureDataset
from models.dvsgesture_model import DVSGestureSNN
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Train Rockpool SNN on DVSGesture")
    parser.add_argument('--model_type', type=str, default='dense', choices=['sparse', 'dense'],
                        help='Model type: sparse (fewer spikes) or dense (more spikes)')
    parser.add_argument('--w', type=int, default=32, help='Width')
    parser.add_argument('--h', type=int, default=32, help='Height')
    parser.add_argument('--n_frames', type=int, default=32, help='Number of time steps')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset_path', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--output_path', type=str, default='./results', help='Path to save the trained model')

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load dataset
    data = DVSGestureDataset(
        dataset_path=args.dataset_path,
        w=args.w,
        h=args.h,
        n_frames=args.n_frames,
    )
    cached_train, cached_test = data.load_dvsgesture()

    # Create dataloaders
    train_loader = DataLoader(
        cached_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model hyperparameters
    if args.model_type == 'sparse':
        tau_mem = 0.1
        spike_lam = 0.0
    else:
        tau_mem = 0.1
        spike_lam = 0.0
        print(f"   - spike_lam: {spike_lam} (disabled)")

    # Create model
    model = DVSGestureSNN(
        input_size=args.w * args.h * 2,
        n_frames=args.n_frames,
        tau_mem=tau_mem,
        tau_syn=0.1,
        spike_lam=spike_lam,
        model_type=args.model_type,
        device=device,
        num_classes=data.get_num_classes(),
        lr=1e-5,
        dt=0.01,
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
