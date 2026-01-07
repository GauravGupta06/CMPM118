import torch
import tonic
import sys
import os

# Add parent directory to path for LoadDataset import
sys.path.insert(0, os.path.dirname(__file__))

from LoadDataset import load_dataset
from RockpoolSNN_model import DVSGestureSNN_FC, SHDSNN


def train_dvsgesture():
    """Train DVSGesture model with Rockpool."""
    # Setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Model hyperparameters
    w = 32
    h = 32
    n_frames = 32

    # Load dataset
    print("\nLoading DVSGesture dataset...")
    cached_train, cached_test, num_classes = load_dataset(
        dataset_name="DVSGesture",
        dataset_path="./data",
        w=w,
        h=h,
        n_frames=n_frames
    )

    # Create data loaders
    active_cores = 7
    train_loader = torch.utils.data.DataLoader(
        cached_train,
        batch_size=8,
        shuffle=True,
        num_workers=active_cores,
        drop_last=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    test_loader = torch.utils.data.DataLoader(
        cached_test,
        batch_size=8,
        shuffle=False,
        num_workers=active_cores,
        drop_last=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    # Create Rockpool model
    print("\nCreating Rockpool SNN model...")
    model = DVSGestureSNN_FC(
        w=w,
        h=h,
        n_frames=n_frames,
        tau_mem=0.02,  # 20ms membrane time constant
        spike_lam=1e-7,
        model_type="dense",
        device=device,
        num_classes=num_classes
    )

    print(f"\nModel architecture:")
    print(model.net)

    # Train model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=150,
        print_every=15
    )

    # Save model
    print("\nSaving model...")
    model.save_model(base_path="../results")

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    final_acc = model.validate_model(test_loader)
    print(f"Final Test Accuracy: {final_acc * 100:.2f}%")

    return model


def train_shd():
    """Train SHD model with Rockpool."""
    # Setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Model hyperparameters
    freq_bins = 700
    n_frames = 100

    # Load dataset
    print("\nLoading SHD dataset...")
    cached_train, cached_test, num_classes = load_dataset(
        dataset_name="SHD",
        dataset_path="./data",
        w=freq_bins,
        h=1,
        n_frames=n_frames
    )

    # Create data loaders
    active_cores = 7
    train_loader = torch.utils.data.DataLoader(
        cached_train,
        batch_size=32,
        shuffle=True,
        num_workers=active_cores,
        drop_last=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    test_loader = torch.utils.data.DataLoader(
        cached_test,
        batch_size=32,
        shuffle=False,
        num_workers=active_cores,
        drop_last=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    # Create Rockpool model
    print("\nCreating Rockpool SNN model for SHD...")
    model = SHDSNN(
        freq_bins=freq_bins,
        n_frames=n_frames,
        tau_mem=0.02,
        spike_lam=1e-7,
        model_type="dense",
        device=device,
        num_classes=num_classes
    )

    print(f"\nModel architecture:")
    print(model.net)

    # Train model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=200,
        print_every=20
    )

    # Save model
    print("\nSaving model...")
    model.save_model(base_path="../results")

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    final_acc = model.validate_model(test_loader)
    print(f"Final Test Accuracy: {final_acc * 100:.2f}%")

    return model


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Rockpool SNN models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="DVSGesture",
        choices=["DVSGesture", "SHD"],
        help="Dataset to train on"
    )

    args = parser.parse_args()

    if args.dataset == "DVSGesture":
        model = train_dvsgesture()
    elif args.dataset == "SHD":
        model = train_shd()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("\nTraining complete!")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
