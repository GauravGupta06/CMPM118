"""Unified training script for Rockpool SNN models on SHD dataset."""
import torch
import tonic
import argparse
import sys
import os
import json
from datetime import datetime
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from datasets.shd_dataset import SHDDataset
# <-- Use the SHD model adapted to the paper (saved as models/shd_model_paper.py)
from models.shd_model_paper import SHDSNN
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
    parser.add_argument('--paper_energy_mJ', type=float, default=0.46,
                        help='Reference paper average energy per inference in mJ (default 0.46 mJ - Loihi2 feedforward w/delay Table II)')

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

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

    # Create model (using the paper-like SHD model)
    model = SHDSNN(
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
    print(f"{'='*60}\n")

    # ------------------------
    # Spike counting & energy estimate
    # ------------------------
    # For accurate per-sample spike counts, evaluate with batch_size=1
    print("Computing spike statistics on test set (batch_size=1 recommended for exact per-sample counts)...")
    energy_eval_loader = DataLoader(
        cached_test, batch_size=1, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True)
    )

    spike_stats = model.compute_spike_counts_on_loader(energy_eval_loader, verbose=True)
    print("\nSpike stats:")
    print(f"  n_samples: {spike_stats['n_samples']}")
    print(f"  total_spikes: {spike_stats['total_spikes']:.2f}")
    print(f"  avg_spikes_per_sample: {spike_stats['avg_spikes_per_sample']:.4f}")

    # Convert paper energy from mJ -> J
    paper_energy_J = float(args.paper_energy_mJ) * 1e-3
    print(f"\nUsing paper reference energy: {args.paper_energy_mJ} mJ/inference ({paper_energy_J:.6e} J)")

    energy_est = model.estimate_energy_per_spike_from_paper(
        paper_energy_joules_per_inference=paper_energy_J,
        spike_stats=spike_stats,
        paper_note=f"paper_energy_mJ={args.paper_energy_mJ}"
    )

    print("\nEstimated energy metrics (based on paper normalization):")
    print(f"  Energy per spike (J): {energy_est['energy_per_spike_J']:.6e}")
    print(f"  Estimated total energy for measured test set (J): {energy_est['estimated_total_energy_J']:.6e}")
    print(f"  Paper energy (J/inference): {energy_est['paper_energy_j_per_inference']:.6e}")
    print("\nNote: This estimate assumes energy scales roughly linearly with spike count on the referenced hardware. "
          "Report this assumption when using these figures.\n")

    def estimate_energy_from_paper(total_spikes, avg_spikes_per_sample, n_samples,
                                paper_energy_j=4.6e-4,  # 0.46 mJ -> J
                                uncertainty_frac=0.5):
        """
        Returns a dict with energy-per-spike, per-sample energy estimates, totals and uncertainty.
        uncertainty_frac: reporting uncertainty multiplier (e.g. 0.5 => ±50%) to reflect model->hardware mismatch.
        """
        energy_per_spike = paper_energy_j / avg_spikes_per_sample if avg_spikes_per_sample > 0 else float('nan')
        estimated_total_energy = total_spikes * energy_per_spike
        estimated_energy_per_sample = estimated_total_energy / n_samples if n_samples>0 else float('nan')

        # Add reporting uncertainty range (conservative)
        low = estimated_total_energy * (1 - uncertainty_frac)
        high = estimated_total_energy * (1 + uncertainty_frac)

        return {
            'paper_energy_per_inference_J': paper_energy_j,
            'energy_per_spike_J': energy_per_spike,
            'n_samples': n_samples,
            'total_spikes': total_spikes,
            'avg_spikes_per_sample': avg_spikes_per_sample,
            'estimated_total_energy_J': estimated_total_energy,
            'estimated_energy_per_sample_J': estimated_energy_per_sample,
            'estimated_energy_total_J_uncertainty': [low, high],
            'uncertainty_fraction': uncertainty_frac,
            'timestamp': datetime.now().isoformat()
        }

    summary = estimate_energy_from_paper(
        total_spikes=spike_stats['total_spikes'],
        avg_spikes_per_sample=spike_stats['avg_spikes_per_sample'],
        n_samples=spike_stats['n_samples'],
    )

    # Create unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs('results/shd/energy_estimates', exist_ok=True)

    filename = f"results/shd/energy_estimates/shd_energy_estimate_{timestamp}.json"

    with open(filename, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved energy estimate to {filename}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()