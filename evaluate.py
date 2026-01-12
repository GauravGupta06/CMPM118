"""Evaluation script for trained Rockpool SNN models on SHD dataset."""

import torch
import tonic
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_shd
from models import SHDSNN_FC


def evaluate_model_on_test(model, test_loader, device):
    """
    Evaluate model and return accuracy + average spikes.

    Returns:
        (accuracy, avg_spikes_per_sample)
    """
    accuracy = model.validate_model(test_loader)

    # Compute average spikes
    total_spikes = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            spk_rec, spike_count = model.forward_pass(data)

            batch_size = data.shape[1] if data.dim() >= 2 else 1
            total_spikes += float(spike_count)
            total_samples += int(batch_size)

    avg_spikes = total_spikes / total_samples if total_samples > 0 else 0.0
    return float(accuracy), float(avg_spikes)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Rockpool SNN on SHD")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model file (.pth)')
    parser.add_argument('--model_type', type=str, default='dense', choices=['sparse', 'dense'],
                        help='Model type: sparse or dense')
    parser.add_argument('--n_frames', type=int, default=100,
                        help='Number of time steps')
    parser.add_argument('--dataset_path', type=str, default='./data',
                        help='Path to dataset directory')

    args = parser.parse_args()

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

    test_loader = torch.utils.data.DataLoader(
        cached_test,
        batch_size=1,
        shuffle=False,
        num_workers=7,
        drop_last=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    # Model hyperparameters
    input_size = 700
    tau_mem = 0.01 if args.model_type == 'sparse' else 0.02
    spike_lam = 1e-6 if args.model_type == 'sparse' else 1e-8

    print(f"\nCreating {args.model_type} model...")
    print(f"   - input_size: {input_size}")
    print(f"   - tau_mem: {tau_mem}")
    print(f"   - spike_lam: {spike_lam}")

    model = SHDSNN_FC(
        input_size=input_size,
        n_frames=args.n_frames,
        tau_mem=tau_mem,
        spike_lam=spike_lam,
        model_type=args.model_type,
        device=device,
        num_classes=num_classes
    )

    # Load weights
    print(f"\nLoading model from: {args.model_path}")
    model.load_model(args.model_path)

    # Evaluate
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)

    accuracy, avg_spikes = evaluate_model_on_test(model, test_loader, device)

    print(f"\nResults:")
    print(f"  - Accuracy: {accuracy * 100:.2f}%")
    print(f"  - Average spikes per sample: {avg_spikes:.2f}")
    print("="*60)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
