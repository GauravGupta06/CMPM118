"""Test script for DVSGesture model.

Loads a trained checkpoint, runs inference on the test set,
and reports accuracy, spike counts, and energy estimates.
"""
import torch
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from datasets.dvsgesture_dataset import DVSGestureDataset
from models.dvsgesture_model import DVSGestureSNN
from torch.utils.data import DataLoader
import tonic


def main():
    parser = argparse.ArgumentParser(description="Test DVSGesture SNN model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_timesteps', type=int, default=600)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cpu")
    print(f"Using device: {device}")

    # Load hyperparams from checkpoint
    hyperparams = DVSGestureSNN.load_hyperparams(args.model_path, device=device)

    # Create model with saved hyperparams
    model = DVSGestureSNN(
        input_size=hyperparams.get('input_size', None),
        n_frames=hyperparams.get('n_frames', args.max_timesteps),
        tau_mem=hyperparams.get('tau_mem', 0.01378),
        spike_lam=hyperparams.get('spike_lam', 0.0),
        model_type=hyperparams.get('model_type', 'dense'),
        device=device,
        num_classes=hyperparams.get('num_classes', 11),
        lr=hyperparams.get('lr', 0.003),
        dt=hyperparams.get('dt', 0.001),
        threshold=hyperparams.get('threshold', 1.0),
        has_bias=hyperparams.get('has_bias', False),
    )

    # Load weights
    model.load_model(args.model_path)

    # Load test dataset
    data = DVSGestureDataset(
        dataset_path=args.dataset_path,
        w=32,
        h=32,
        max_timesteps=args.max_timesteps,
    )
    _, cached_test = data.load_dvsgesture()

    test_loader = DataLoader(
        cached_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True)
    )

    # Run inference on test set
    model.eval()
    correct = 0
    total = 0
    total_spikes = 0
    total_timestep_energy = 0.0

    print("\nRunning inference on test set...")

    with torch.no_grad():
        for batch_idx, (batch_data, targets) in enumerate(test_loader):
            batch_data = batch_data.to(device).float()
            targets = targets.to(device)

            # Run inference with spike recording
            logits, spike_count = model.run_inference(batch_data, record=True)

            # Predictions
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            total_spikes += spike_count

            # Per-timestep energy for this batch
            batch_energies = model.estimate_energy(batch_data, method='per_timestep')
            total_timestep_energy += sum(batch_energies)

            if (batch_idx + 1) % 1 == 0:
                print(f"  Processed {total} samples...")
                break

    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0
    avg_spikes_per_inference = total_spikes / total if total > 0 else 0
    avg_timestep_energy = total_timestep_energy / total if total > 0 else 0.0

    # Energy per spike calibration from Arfa et al.: 459 mJ total for reference model
    energy_per_spike = 459e-3 / avg_spikes_per_inference if avg_spikes_per_inference > 0 else 0.0

    # Per-spike energy estimate
    per_spike_energy = avg_spikes_per_inference * energy_per_spike

    # Print results
    model_type = hyperparams.get('model_type', 'unknown')
    architecture = hyperparams.get('architecture', 'unknown')

    print(f"\n{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Architecture: {architecture}")
    print(f"Model type: {model_type}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Total test samples: {total}")
    print(f"\nSPIKE COUNTS:")
    print(f"  Total spikes (entire test set): {total_spikes:,}")
    print(f"  Avg spikes per inference: {avg_spikes_per_inference:,.0f}")
    print(f"\nENERGY ESTIMATES:")
    print(f"  Per-timestep energy: {avg_timestep_energy * 1000:.3f} mJ/inference (0.765 mJ × avg_timesteps)")
    print(f"  Per-spike calibration: {energy_per_spike:.3e} J/spike (459mJ / avg_spikes)")
    print(f"  Per-spike energy: {per_spike_energy * 1000:.3f} mJ/inference (avg_spikes × energy_per_spike)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
