"""
Router for Rockpool SNN models.
Analyzes complexity of samples and routes between sparse and dense models for energy efficiency.

Usage:
    python simplified_router.py --sparse_model_path <path> --dense_model_path <path> [options]

Example:
    python simplified_router.py \
        --sparse_model_path /workspace/sparse/small/models/Rockpool_Sparse_Take2_HAR_Input9_T128_FC_Rockpool_Epochs30.pth \
        --dense_model_path  /workspace/dense/large/models/Rockpool_Non_Sparse_Take4_HAR_Input9_T128_FC_Rockpool_Epochs30.pth
"""

import numpy as np
import os
import tonic
import torch
from lempel_ziv_complexity import lempel_ziv_complexity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import entropy
import json
from datetime import datetime
import argparse
import sys



# SHD
from datasets.shd_dataset import SHDDataset
from models.shd_model import SHDSNN

# DVSGesture
from datasets.dvsgesture_dataset import DVSGestureDataset
from models.dvsgesture_model import DVSGestureSNN




from torch.utils.data import DataLoader


# ========== COMPLEXITY METRICS ==========

def count_spikes_from_recording(recording_dict):
    """
    Count total spikes from model recording dictionary.
    Works with Rockpool's Sequential recording format for ANY spiking neuron type.

    Args:
        recording_dict: Recording dictionary from model forward pass with record=True

    Returns:
        int: Total number of spikes across all layers and timesteps
    """
    total_spikes = 0

    for layer_name, layer_data in recording_dict.items():
        if torch.is_tensor(layer_data):
            # Check if this tensor contains spike data (binary: only 0s and 1s)
            # Spikes are always binary regardless of neuron type (LIF, Izhikevich, AdEx, etc.)
            unique_vals = torch.unique(layer_data)
            is_binary = len(unique_vals) <= 2 and torch.all((unique_vals >= 0) & (unique_vals <= 1))
            if is_binary and layer_data.numel() > 0:
                total_spikes += layer_data.sum().item()
        elif isinstance(layer_data, dict) and 'spikes' in layer_data:
            # Fallback for other recording formats
            spikes = layer_data['spikes']
            total_spikes += spikes.sum().item()

    return int(total_spikes)


def compute_lzc_from_events(events):
    """
    Compute Lempel-Ziv Complexity from spike events.

    Args:
        events: Spike tensor

    Returns:
        float: LZC score
    """
    if torch.is_tensor(events):
        events = events.cpu().numpy()

    spike_seq = (events).astype(int).flatten()
    spike_seq_string = ''.join(map(str, spike_seq.tolist()))
    lz_score = lempel_ziv_complexity(spike_seq_string)
    return lz_score


# ========== PRECOMPUTED LZC LOADING ==========

def load_lzc_from_file(path):
    """
    Load precomputed LZC values from an energy file.
    Each line is: <energy> <cycles> <lzc_score>

    Returns:
        list of dicts with keys: energy, cycles, lzc_score
    """
    records = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                records.append({
                    'energy': float(parts[0]),
                    'cycles': int(parts[1]),
                    'lzc_score': int(parts[2])
                })
            elif len(parts) == 2:
                # Legacy 2-column format
                records.append({
                    'energy': float(parts[0]),
                    'cycles': 0,
                    'lzc_score': int(parts[1])
                })
    print(f"  Loaded {len(records)} precomputed LZC values from {path}")
    return records


def evaluate_models_on_dataset(dataLoader, sparse_model, dense_model, precomputed_lzc):
    """
    Evaluate both models on entire dataset using batched inference.
    Only collects what's needed for threshold finding: LZC value, predictions, true_complex.
    """
    results = []
    sample_idx = 0
    
    for batch in dataLoader:
        events, labels = batch
        batch_size = events.shape[0]
        events = events.to(sparse_model.device)
        
        with torch.no_grad():
            sparse_output, _, _ = sparse_model.net(events, record=False)
            dense_output, _, _ = dense_model.net(events, record=False)
        
        sparse_preds = sparse_output.mean(dim=1).argmax(1)
        dense_preds = dense_output.mean(dim=1).argmax(1)
        
        for i in range(batch_size):
            label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
            sparse_pred = sparse_preds[i].item()
            dense_pred = dense_preds[i].item()
            lz_value = precomputed_lzc[sample_idx]['lzc_score']
            
            true_complex = 1 if (dense_pred == label and sparse_pred != label) else 0
            
            results.append({
                'label': label,
                'lz_value': lz_value,
                'sparse_pred': sparse_pred,
                'dense_pred': dense_pred,
                'true_complex': true_complex,
            })
            sample_idx += 1
            if sample_idx % 10 == 0:
                print(f"Processed {sample_idx} samples")
    
    return results


def threshold_sweep_and_roc(results, dataset_name="unknown", plotting_only=False):
    """
    Perform threshold sweep, compute ROC-AUC curve, and find optimal LZC threshold.

    Args:
        results: Per-sample results from evaluate_models_on_dataset
        dataset_name: Name of the dataset ("shd", "dvsgesture", "uci_har")
        plotting_only: If True, only return values without printing/plotting

    Returns:
        tuple: (optimal_threshold, roc_auc)
    """
    # Ground truth: 1 if dense model was needed, 0 if sparse sufficed
    y_true = np.array([r['true_complex'] for r in results])
    lz_scores = np.array([r['lz_value'] for r in results])
    fpr, tpr, thresholds = roc_curve(y_true, lz_scores)
    roc_auc = auc(fpr, tpr)
    gmean = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmean)
    optimal_threshold = thresholds[idx]

    if plotting_only:
        return optimal_threshold, roc_auc

    print(f"Optimal LZC threshold: {optimal_threshold:.4f} (G-mean={gmean[idx]:.4f}) (AUC={roc_auc:.4f})")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.scatter(fpr[idx], tpr[idx], color='red', label=f'Optimal G-mean\n(Threshold={optimal_threshold:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for LZC-based Routing')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("results/ROC_curves", exist_ok=True)
    
    # Initialize counter file if it doesn't exist
    counter_file_path = "results/ROC_curves/roc_counter.txt"
    if not os.path.exists(counter_file_path):
        with open(counter_file_path, "w") as f:
            f.write("0")
    
    # Read and increment counter
    with open(counter_file_path, "r") as f:
        counter = int(f.read().strip())
    counter += 1
    with open(counter_file_path, "w") as f:
        f.write(str(counter))
    
    graph_save_path = f"results/ROC_curves/{dataset_name}_Take{counter}.png"
    plt.savefig(graph_save_path)
    print(f"ROC curve saved to: {graph_save_path}")
    plt.close()

    return optimal_threshold, roc_auc


# Energy per spike constant (placeholder — set to your model's value)
ENERGY_PER_SPIKE = 5e-9 # Joules per spike (TODO: set correct value)


def route_and_evaluate(dataLoader, sparse_model, dense_model, optimal_threshold, results, precomputed_lzc):
    """
    Route samples and compute accuracy + energy metrics.
    Runs single-sample inference with record=True on the routed model for exact spike counts.

    Reports three metric blocks:
      1. Baseline: dense-only accuracy + energy (no LZC cost)
      2. Sparse-routed: accuracy + energy (LZC + sparse model) for routed samples
      3. Dense-routed: accuracy + energy (LZC + dense model) for routed samples
    """
    print("\nRouting and evaluating with threshold:", optimal_threshold, "\n")

    # Accumulators
    route_counts = {'sparse': 0, 'dense': 0}
    correct_sparse_routed = 0
    correct_dense_routed = 0
    sparse_energies = []   # per-sample total energy (LZC + model) for sparse-routed
    dense_energies = []    # per-sample total energy (LZC + model) for dense-routed
    baseline_energies = [] # per-sample dense-only energy (no LZC cost)

    # Overall accuracy on entire dataset (from evaluate_models_on_dataset predictions)
    total_samples = len(results)
    sparse_correct_all = sum(1 for r in results if r['sparse_pred'] == r['label'])
    dense_correct_all = sum(1 for r in results if r['dense_pred'] == r['label'])
    sparse_acc_overall = sparse_correct_all / total_samples
    dense_acc_overall = dense_correct_all / total_samples

    sample_idx = 0

    for batch in dataLoader:
        events, labels = batch
        batch_size = events.shape[0]
        events = events.to(sparse_model.device)

        # Process each sample in the batch individually for routing
        for i in range(batch_size):
            if sample_idx >= len(results):
                break

            single_event = events[i:i+1]  # [1, T, C]
            label = results[sample_idx]['label']
            lz_value = results[sample_idx]['lz_value']
            lzc_energy = precomputed_lzc[sample_idx]['energy']

            with torch.no_grad():
                # --- Baseline: always run dense (no LZC cost) ---
                _, _, dense_rec = dense_model.net(single_event, record=True)
                baseline_spikes = count_spikes_from_recording(dense_rec)
                baseline_energies.append(baseline_spikes * ENERGY_PER_SPIKE)

                # --- Routing decision ---
                if lz_value < optimal_threshold:
                    # Route to sparse
                    route_counts['sparse'] += 1
                    _, _, sparse_rec = sparse_model.net(single_event, record=True)
                    model_spikes = count_spikes_from_recording(sparse_rec)
                    model_energy = model_spikes * ENERGY_PER_SPIKE
                    sparse_energies.append(lzc_energy + model_energy)

                    if results[sample_idx]['sparse_pred'] == label:
                        correct_sparse_routed += 1
                else:
                    # Route to dense (reuse the recording from baseline)
                    route_counts['dense'] += 1
                    model_spikes = baseline_spikes  # already computed above
                    model_energy = model_spikes * ENERGY_PER_SPIKE
                    dense_energies.append(lzc_energy + model_energy)

                    if results[sample_idx]['dense_pred'] == label:
                        correct_dense_routed += 1

            sample_idx += 1
            if sample_idx % 10 == 0:
                print(f"  Routed {sample_idx}/{total_samples} samples")

    # ==================== METRICS ====================
    n_sparse = route_counts['sparse']
    n_dense = route_counts['dense']
    total_routed = n_sparse + n_dense

    # Routed accuracy
    acc_sparse_routed = correct_sparse_routed / n_sparse if n_sparse > 0 else 0
    acc_dense_routed = correct_dense_routed / n_dense if n_dense > 0 else 0
    total_correct_routed = correct_sparse_routed + correct_dense_routed
    acc_overall_routed = total_correct_routed / total_routed if total_routed > 0 else 0

    # Average energies
    avg_sparse_energy = np.mean(sparse_energies) if sparse_energies else 0
    avg_dense_energy = np.mean(dense_energies) if dense_energies else 0
    avg_baseline_energy = np.mean(baseline_energies) if baseline_energies else 0
    avg_routed_energy = np.mean(sparse_energies + dense_energies) if (sparse_energies + dense_energies) else 0

    # ==================== REPORTING ====================
    print("\n" + "="*60)
    print("BASELINE (Dense-Only, No Router)")
    print("="*60)
    print(f"  Accuracy:     {dense_acc_overall*100:.2f}%")
    print(f"  Avg Energy:   {avg_baseline_energy:.4e} J  (no LZC cost)")

    print("\n" + "="*60)
    print("SPARSE-ROUTED SAMPLES")
    print("="*60)
    print(f"  Samples routed:           {n_sparse}")
    print(f"  Accuracy (routed):        {acc_sparse_routed*100:.2f}%")
    print(f"  Accuracy (entire dataset): {sparse_acc_overall*100:.2f}%")
    print(f"  Accuracy improvement:     {((acc_sparse_routed/sparse_acc_overall - 1)*100 if sparse_acc_overall > 0 else 0):.2f}%")
    print(f"  Avg Energy (LZC + model): {avg_sparse_energy:.4e} J")

    print("\n" + "="*60)
    print("DENSE-ROUTED SAMPLES")
    print("="*60)
    print(f"  Samples routed:           {n_dense}")
    print(f"  Accuracy (routed):        {acc_dense_routed*100:.2f}%")
    print(f"  Accuracy (entire dataset): {dense_acc_overall*100:.2f}%")
    print(f"  Accuracy improvement:     {((acc_dense_routed/dense_acc_overall - 1)*100 if dense_acc_overall > 0 else 0):.2f}%")
    print(f"  Avg Energy (LZC + model): {avg_dense_energy:.4e} J")

    print("\n" + "="*60)
    print("OVERALL ROUTED")
    print("="*60)
    print(f"  Total samples:    {total_routed}")
    print(f"  Overall accuracy: {acc_overall_routed*100:.2f}%")
    print(f"  Avg Energy:       {avg_routed_energy:.4e} J")
    print(f"  Energy savings vs baseline: {((1 - avg_routed_energy/avg_baseline_energy)*100 if avg_baseline_energy > 0 else 0):.2f}%")
    print("="*60 + "\n")

    return {
        'baseline_accuracy': dense_acc_overall,
        'baseline_avg_energy': avg_baseline_energy,
        'sparse_routed_accuracy': acc_sparse_routed,
        'sparse_avg_energy': avg_sparse_energy,
        'dense_routed_accuracy': acc_dense_routed,
        'dense_avg_energy': avg_dense_energy,
        'overall_routed_accuracy': acc_overall_routed,
        'overall_avg_energy': avg_routed_energy,
        'route_counts': route_counts,
    }


# ========== MAIN ==========

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Router for Rockpool SHD models - analyzes complexity and routes between sparse/dense models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 700 frequency bins
  python router.py \\
    --sparse_model_path ./results/small/models/Sparse_Take1.pth \\
    --dense_model_path ./results/large/models/Dense_Take1.pth

  # Custom hyperparameters
  python router.py \\
    --sparse_model_path ./results/small/models/Sparse.pth \\
    --dense_model_path ./results/large/models/Dense.pth \\
    --tau_mem_sparse 0.015 \\
    --tau_mem_dense 0.025
        """
    )

    # Required arguments
    parser.add_argument('--sparse_model_path', type=str, required=True,
                       help='Path to pre-trained sparse model (.pth file)')
    parser.add_argument('--dense_model_path', type=str, required=True,
                       help='Path to pre-trained dense model (.pth file)')

    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, default='./data',
                       help='Path to dataset cache directory (default: ./data)')
    parser.add_argument('--input_size', type=int, default=700,
                       help='Number of frequency bins (default: 700)')
    parser.add_argument('--n_frames', type=int, default=100,
                       help='Number of temporal bins (default: 100)')

    # Model hyperparameter arguments
    parser.add_argument('--tau_mem_sparse', type=float, default=0.01,
                       help='Membrane time constant for sparse model (default: 0.01)')
    parser.add_argument('--tau_mem_dense', type=float, default=0.02,
                       help='Membrane time constant for dense model (default: 0.02)')
    parser.add_argument('--spike_lam_sparse', type=float, default=1e-6,
                       help='Spike regularization for sparse model (default: 1e-6)')
    parser.add_argument('--spike_lam_dense', type=float, default=1e-8,
                       help='Spike regularization for dense model (default: 1e-8)')
    
    # Additional arguments (matching train_shd.py)
    parser.add_argument('--NUM_CHANNELS', type=int, default=700,
                       help='Number of frequency channels (default: 700)')
    parser.add_argument('--NUM_POLARITIES', type=int, default=2,
                       help='Number of polarities (default: 2)')
    parser.add_argument('--net_dt', type=float, default=10e-3,
                       help='Simulation time step in seconds (default: 10e-3)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers (default: 4)')
    parser.add_argument('--model_type', type=str, default='dense',
                       help='Model type for both models (default: dense)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')

    # Parse arguments
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Sparse model path: {args.sparse_model_path}")
    print(f"Dense model path:  {args.dense_model_path}")
    print(f"Dataset path:      {args.dataset_path}")
    print(f"Input size:        {args.input_size} frequency bins")
    print(f"N frames:          {args.n_frames}")
    print(f"Tau mem (sparse):  {args.tau_mem_sparse}")
    print(f"Tau mem (dense):   {args.tau_mem_dense}")
    print(f"Spike lam (sparse):{args.spike_lam_sparse}")
    print(f"Spike lam (dense): {args.spike_lam_dense}")
    print("="*80 + "\n")

    # ==================== DATASET LOADING ====================
    # Uncomment ONE of the following dataset sections:

    # ----- SHD Dataset -----
    print("Loading SHD dataset...")
    data = SHDDataset(
        dataset_path=args.dataset_path,
        NUM_CHANNELS=args.NUM_CHANNELS,
        NUM_POLARITIES=args.NUM_POLARITIES,
        n_frames=args.n_frames,
        net_dt=args.net_dt
    )
    _, cached_test = data.load_shd()
    test_loader = DataLoader(
        cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=False,
        collate_fn=tonic.collation.PadTensors(batch_first=True)
    )
    # Limit to 3 batches for faster testing
    import itertools
    test_loader = list(itertools.islice(test_loader, 5))

    # ----- DVSGesture Dataset -----
    # print("Loading DVSGesture dataset...")
    # data = DVSGestureDataset(
    #     dataset_path=args.dataset_path,
    #     NUM_CHANNELS=args.NUM_CHANNELS,
    #     NUM_POLARITIES=args.NUM_POLARITIES,
    #     n_frames=args.n_frames,
    #     net_dt=args.net_dt
    # )
    # _, cached_test = data.load_dvsgesture()
    # test_loader = DataLoader(
    #     cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
    #     num_workers=args.num_workers, pin_memory=True,
    #     collate_fn=tonic.collation.PadTensors(batch_first=True)
    # )

    # ==================== END DATASET LOADING ====================



    

    # Load hyperparameters from checkpoint files
    print("\nLoading hyperparameters from checkpoints...")

    sparse_hp = SHDSNN.load_hyperparams(args.sparse_model_path)
    dense_hp = SHDSNN.load_hyperparams(args.dense_model_path)

    # sparse_hp = DVSGestureSNN.load_hyperparams(args.sparse_model_path)
    # dense_hp = DVSGestureSNN.load_hyperparams(args.dense_model_path)


    

    # ==================== MODEL CREATION ====================
    # Uncomment ONE of the following model sections (must match dataset above):

    # ----- SHD Models -----
    print("\nCreating SHD sparse model...")
    sparse_model = SHDSNN(
        input_size=sparse_hp['input_size'],
        n_frames=sparse_hp['n_frames'],
        tau_mem=sparse_hp['tau_mem'],
        tau_syn=sparse_hp['tau_syn'],
        spike_lam=sparse_hp['spike_lam'],
        model_type=sparse_hp['model_type'],
        device=device,
        num_classes=sparse_hp['num_classes'],
        dt=sparse_hp['dt'],
        threshold=sparse_hp['threshold'],
        has_bias=sparse_hp['has_bias']
    )
    print("Creating SHD dense model...")
    dense_model = SHDSNN(
        input_size=dense_hp['input_size'],
        n_frames=dense_hp['n_frames'],
        tau_mem=dense_hp['tau_mem'],
        tau_syn=dense_hp['tau_syn'],
        spike_lam=dense_hp['spike_lam'],
        model_type=dense_hp['model_type'],
        device=device,
        num_classes=dense_hp['num_classes'],
        dt=dense_hp['dt'],
        threshold=dense_hp['threshold'],
        has_bias=dense_hp['has_bias']
    )






    # ----- DVSGesture Models -----
    # print("\nCreating DVSGesture sparse model...")
    # sparse_model = DVSGestureSNN(
    #     input_size=sparse_hp['input_size'],
    #     n_frames=sparse_hp['n_frames'],
    #     tau_mem=sparse_hp['tau_mem'],
    #     tau_syn=sparse_hp['tau_syn'],
    #     spike_lam=sparse_hp['spike_lam'],
    #     model_type=sparse_hp['model_type'],
    #     device=device,
    #     num_classes=sparse_hp['num_classes'],
    #     dt=sparse_hp['dt'],
    #     threshold=sparse_hp['threshold'],
    #     has_bias=sparse_hp['has_bias']
    # )
    # print("Creating DVSGesture dense model...")
    # dense_model = DVSGestureSNN(
    #     input_size=dense_hp['input_size'],
    #     n_frames=dense_hp['n_frames'],
    #     tau_mem=dense_hp['tau_mem'],
    #     tau_syn=dense_hp['tau_syn'],
    #     spike_lam=dense_hp['spike_lam'],
    #     model_type=dense_hp['model_type'],
    #     device=device,
    #     num_classes=dense_hp['num_classes'],
    #     dt=dense_hp['dt'],
    #     threshold=dense_hp['threshold'],
    #     has_bias=dense_hp['has_bias']
    # )

    # ==================== END MODEL CREATION ====================

    # Load pre-trained weights
    print(f"\nLoading sparse model from: {args.sparse_model_path}")
    sparse_model.load_model(args.sparse_model_path)

    print(f"Loading dense model from: {args.dense_model_path}")
    dense_model.load_model(args.dense_model_path)

    # Main execution
    print("\n" + "="*80)
    print("EVERYTHING LOADED SUCCESSFULLY")
    print("="*80 + "\n")
    print("Starting evaluation...\n")

    # Load precomputed LZC energy metrics
    lzc_file = 'LZC_Energy/lzc_energy_SHD.txt'
    print(f"\nLoading precomputed LZC values from: {lzc_file}")
    precomputed_lzc = load_lzc_from_file(lzc_file)

    # 1. Evaluate both models on all test samples (using precomputed LZC)
    results = evaluate_models_on_dataset(test_loader, sparse_model, dense_model, precomputed_lzc)

    # 2. Find optimal routing threshold
    optimal_threshold, roc_auc = threshold_sweep_and_roc(results, dataset_name="shd")
    # optimal_threshold, roc_auc = threshold_sweep_and_roc(results, dataset_name="dvsgesture")
    
    # 3. Route samples and evaluate with energy metrics
    metrics = route_and_evaluate(test_loader, sparse_model, dense_model, optimal_threshold, results, precomputed_lzc)


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
