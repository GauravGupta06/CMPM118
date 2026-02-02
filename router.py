"""
Router for Rockpool SNN models.
Analyzes complexity of samples and routes between sparse and dense models for energy efficiency.

Usage:
    python router.py --sparse_model_path <path> --dense_model_path <path> [options]

Example:
    python router.py \
        --sparse_model_path ./workspace/large/models/Rockpool_Non_Sparse_Take1_HAR_Input9_T128_FC_Rockpool_Epochs1.pth \
        --dense_model_path ./workspace/large/models/Rockpool_Non_Sparse_Take1_HAR_Input9_T128_FC_Rockpool_Epochs1.pth
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
from models.shd_model import SHDSNN_FC

# DVSGesture
from datasets.dvsgesture_dataset import DVSGestureDataset
from models.dvsgesture_model import DVSGestureSNN_FC

# UCI HAR
from datasets.uci_har import UCIHARDataset
from models.uci_har_model import UCIHARSNN_FC


from core.base_model import BaseSNNModel
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


def compute_shannon_entropy_from_events(events):
    """
    Compute Shannon entropy from spike events.

    Args:
        events: Spike tensor

    Returns:
        float: Shannon entropy value
    """
    flattened = events.cpu().numpy().astype(int).flatten()

    values, counts = np.unique(flattened, return_counts=True)
    probs = counts / counts.sum()

    entropy_value = entropy(probs, base=2)

    return entropy_value


def compute_isi_entropy_from_events(events, num_bins=30):
    """
    Compute Inter-Spike Interval (ISI) entropy from spike events.

    Args:
        events: Spike tensor
        num_bins: Number of bins for histogram

    Returns:
        float: ISI entropy value
    """
    # Handle PyTorch tensors
    if hasattr(events, 'cpu'):  # Check if it's a PyTorch tensor
        events_np = events.cpu().numpy()
        if events_np.ndim > 1:
            # If it's a multi-dimensional tensor, flatten it
            timestamps = np.sort(events_np.flatten())
        else:
            timestamps = np.sort(events_np)
    elif isinstance(events, np.ndarray) and 't' in events.dtype.names:
        timestamps = np.sort(events['t'])
    elif isinstance(events, (list, np.ndarray)):
        timestamps = np.sort(np.array(events))
    elif isinstance(events, dict) and 't' in events:
        timestamps = np.sort(np.array(events['t']))
    else:
        raise ValueError("ISI entropy expects event data with timestamps (events['t']).")

    if len(timestamps) < 2:
        return 0.0

    isis = np.diff(timestamps)

    if np.all(isis == 0):
        return 0.0

    hist, bin_edges = np.histogram(isis, bins=num_bins, density=True)
    hist = hist[hist > 0]
    probs = hist / np.sum(hist)

    isi_entropy = entropy(probs, base=2)

    return isi_entropy


# ========== CORE FUNCTIONS ==========

def evaluate_models_on_dataset(dataLoader, sparse_model, dense_model):
    """
    Evaluate both sparse and dense models on entire dataset.

    Args:
        dataLoader: Test data loader
        sparse_model: Sparse SNN model
        dense_model: Dense SNN model`

    Returns:
        list: Results with per-sample metrics
    """
    results = []

    for batch in dataLoader:
        events, label = batch
        # batch size should be 1 for this function.
        # events shape will be different based on the dataset.

        # Move data to same device as model
        events = events.to(sparse_model.device)

        label = label.item()
        # convert label from tensor to int. 

        # Compute complexity
        lz_value = compute_lzc_from_events(events)

        # Get predictions and spike counts with recording enabled
        # events already in [B, T, features] format from batch_first=True
        with torch.no_grad():
            sparse_output, _, sparse_recording = sparse_model.net(events, record=True)
            dense_output, _, dense_recording = dense_model.net(events, record=True)

        # Count ALL spikes from ALL spiking layers
        spike_count_sparse = count_spikes_from_recording(sparse_recording)
        spike_count_dense = count_spikes_from_recording(dense_recording)

        # Get predictions
        sparse_logits = sparse_output.mean(dim=1)
        dense_logits = dense_output.mean(dim=1)

        sparse_pred = sparse_logits.argmax(1).item()
        dense_pred = dense_logits.argmax(1).item()

        # Set as complex IF dense_pred matches label and sparse_pred does NOT
        if dense_pred == label and sparse_pred != label:
            true_complex = 1
        else:
            true_complex = 0

        results.append({
            'label': label,
            'lz_value': lz_value,
            'sparse_pred': sparse_pred,
            'dense_pred': dense_pred,
            'true_complex': true_complex,
            'dense_spikes': spike_count_dense,
            'sparse_spikes': spike_count_sparse
        })
    return results


def threshold_sweep_and_roc(results, sparse_model, dense_model, plotting_only=False):
    """
    Perform threshold sweep, compute ROC-AUC curve, and find optimal LZC threshold.

    Args:
        results: Per-sample results from evaluate_models_on_dataset
        sparse_model: Sparse model (for extracting parameters)
        dense_model: Dense model (for extracting parameters)
        plotting_only: If True, only return values without printing/plotting

    Returns:
        tuple: (optimal_threshold, roc_auc, average_spike_dense, average_spike_sparse)
    """
    # Ground truth: 1 if dense model was needed, 0 if sparse sufficed
    y_true = np.array([r['true_complex'] for r in results])
    lz_scores = np.array([r['lz_value'] for r in results])
    fpr, tpr, thresholds = roc_curve(y_true, lz_scores)
    roc_auc = auc(fpr, tpr)
    gmean = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmean)
    optimal_threshold = thresholds[idx]

    average_spike_dense = np.mean([r['dense_spikes'] for r in results])
    average_spike_sparse = np.mean([r['sparse_spikes'] for r in results])

    if plotting_only:
        return (
            optimal_threshold,
            roc_auc,
            average_spike_dense,
            average_spike_sparse
        )

    print(f"average spike dense: {average_spike_dense:.2f}")
    print(f"average spike sparse: {average_spike_sparse:.2f}")

    print(f"Optimal LZC threshold: {optimal_threshold:.4f} (G-mean={gmean[idx]:.4f}) (AUC={roc_auc:.4f})")

    # Extract parameters from models
    input_size = sparse_model.input_size
    n_frames = sparse_model.n_frames

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
    graph_save_path = f"results/ROC_curves/Input{input_size}_T{n_frames}_Rockpool.png"
    plt.savefig(graph_save_path)
    plt.show()

    return optimal_threshold, roc_auc, average_spike_dense, average_spike_sparse


def route_and_evaluate(dataLoader, sparse_model, dense_model, optimal_threshold, results):
    """
    Route samples to appropriate model and evaluate accuracy.

    Args:
        dataLoader: Test data loader
        sparse_model: Sparse SNN model
        dense_model: Dense SNN model
        optimal_threshold: LZC threshold for routing
        results: Pre-computed results from evaluate_models_on_dataset

    Returns:
        tuple: (total_accuracy, accuracy_dense_routed, accuracy_sparse_routed, route_counts)
    """
    print("\nRouting and evaluating with threshold:", optimal_threshold, "\n")
    correct_sparse = 0
    correct_dense = 0
    route_counts = {'sparse': 0, 'dense': 0}

    lz_values = [r['lz_value'] for r in results]

    for i, batch in enumerate(dataLoader):
        events, label = batch

        # Move data to same device as model
        events = events.to(sparse_model.device)

        # Extract label as int
        label = label.item()

        lz_value = lz_values[i]

        with torch.no_grad():
            if lz_value < optimal_threshold:
                route_counts['sparse'] += 1
                sparse_output, _, _ = sparse_model.net(events, record=False)
                sparse_logits = sparse_output.mean(dim=1)
                pred = sparse_logits.argmax(1).item()
                if pred == label:
                    correct_sparse += 1
            else:
                route_counts['dense'] += 1
                dense_output, _, _ = dense_model.net(events, record=False)
                dense_logits = dense_output.mean(dim=1)
                pred = dense_logits.argmax(1).item()
                if pred == label:
                    correct_dense += 1

    accuracy_dense_routed = correct_dense / route_counts['dense'] if route_counts['dense'] > 0 else 0
    accuracy_sparse_routed = correct_sparse / route_counts['sparse'] if route_counts['sparse'] > 0 else 0

    total_correct = correct_dense + correct_sparse
    total_samples = route_counts['sparse'] + route_counts['dense']
    total_accuracy = total_correct / total_samples

    # Getting the accuracy on each model for the entire dataset
    total_samples = len(results)

    # Sparse model accuracy
    sparse_correct = sum(1 for r in results if r['sparse_pred'] == r['label'])
    sparse_accuracy_overall = sparse_correct / total_samples

    # Dense model accuracy
    dense_correct = sum(1 for r in results if r['dense_pred'] == r['label'])
    dense_accuracy_overall = dense_correct / total_samples

    sparse_accuracy_improvement = accuracy_sparse_routed/sparse_accuracy_overall - 1
    dense_accuracy_improvement = accuracy_dense_routed/dense_accuracy_overall - 1

    print(f"Sparse model accuracy on entire dataset: {sparse_accuracy_overall*100: .2f}%")
    print(f"Sparse model accuracy AFTER routing: {accuracy_sparse_routed*100: .2f}%")
    print(f"Sparse model accuracy improvement: {sparse_accuracy_improvement*100: .2f}%")
    print(f"Samples routed to sparse model: {route_counts['sparse']}\n")

    print(f"Dense model accuracy on entire dataset: {dense_accuracy_overall*100: .2f}%")
    print(f"Dense model accuracy AFTER routing: {accuracy_dense_routed*100: .2f}%")
    print(f"Dense model accuracy improvement: {dense_accuracy_improvement*100: .2f}%")
    print(f"Samples routed to dense model: {route_counts['dense']}\n")

    print(f"Overall Accuracy after routing: {total_accuracy*100: .2f}%")
    print(f"Total Samples: {total_samples}")

    return total_accuracy, accuracy_dense_routed, accuracy_sparse_routed, route_counts


def lzc_vs_accuracy_plot(results, sparse_model, dense_model):
    """
    Plot overall accuracy vs LZC routing threshold.

    Args:
        results: Per-sample results
        sparse_model: Sparse model (for extracting parameters)
        dense_model: Dense model (for extracting parameters)
    """
    print("\nLZC vs. Accuracy Analysis:")
    lz_values = np.array([r['lz_value'] for r in results])
    total_samples = len(results)
    accuracy_at_threshold = []
    threshold_range = np.linspace(lz_values.min(), lz_values.max(), 50)
    sparse_accuracy_overall = sum(1 for r in results if r['sparse_pred'] == r['label']) / total_samples
    dense_accuracy_overall = sum(1 for r in results if r['dense_pred'] == r['label']) / total_samples

    for threshold in threshold_range:
        correct_total = 0

        for r in results:
            lz_value = r['lz_value']

            if lz_value < threshold:
                pred = r['sparse_pred']
            else:
                pred = r['dense_pred']

            if pred == r['label']:
                correct_total += 1

        accuracy = correct_total / total_samples
        accuracy_at_threshold.append(accuracy)

    optimal_threshold, _, _, _ = threshold_sweep_and_roc(results, sparse_model, dense_model, plotting_only=True)
    optimal_threshold = float(optimal_threshold)
    best_acc_idx = int(np.searchsorted(threshold_range, optimal_threshold))
    best_acc_idx = np.clip(best_acc_idx, 0, len(threshold_range) - 1)
    best_accuracy = accuracy_at_threshold[best_acc_idx]

    # Extract parameters from model
    input_size = sparse_model.input_size
    n_frames = sparse_model.n_frames

    plt.figure(figsize=(10, 7))
    plt.plot(threshold_range, accuracy_at_threshold, marker='o', linestyle='-', markersize=4, label='Routed Model Accuracy')

    plt.axhline(y=sparse_accuracy_overall, color='r', linestyle='--', label=f'Sparse Only ({sparse_accuracy_overall:.4f})')
    plt.axhline(y=dense_accuracy_overall, color='g', linestyle='--', label=f'Dense Only ({dense_accuracy_overall:.4f})')

    plt.scatter(optimal_threshold, best_accuracy, color='k', s=100, zorder=5, label=f'ROC Optimal Threshold ({optimal_threshold:.4f})')

    plt.xlabel('LZC Threshold for Routing')
    plt.ylabel('Overall Model Accuracy')
    plt.title('Overall Accuracy vs. LZC Routing Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs('results/LZC_vs_Accuracy', exist_ok=True)
    graph_save_path = f"results/LZC_vs_Accuracy/Input{input_size}_T{n_frames}_Rockpool.png"
    plt.savefig(graph_save_path)
    plt.show()
    print("Saved LZC vs. Accuracy graph to:", graph_save_path)


def save_run_to_json(
    results,
    optimal_threshold,
    roc_auc,
    route_counts,
    accuracy_dense_routed,
    accuracy_sparse_routed,
    total_accuracy,
    average_spike_dense,
    average_spike_sparse,
    sparse_model,
    dense_model
):
    """
    Save run results to JSON file.

    Args:
        results: Per-sample results
        optimal_threshold: Optimal LZC threshold
        roc_auc: ROC AUC score
        route_counts: Dictionary with routing counts
        accuracy_dense_routed: Accuracy on dense-routed samples
        accuracy_sparse_routed: Accuracy on sparse-routed samples
        total_accuracy: Overall routed accuracy
        average_spike_dense: Average spikes for dense model
        average_spike_sparse: Average spikes for sparse model
        sparse_model: Sparse model (for extracting parameters)
        dense_model: Dense model (for extracting parameters)
    """
    os.makedirs("results/run_logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract parameters from models
    input_size = sparse_model.input_size
    n_frames = sparse_model.n_frames
    tau_mem_sparse = sparse_model.tau_mem
    tau_mem_dense = dense_model.tau_mem
    spike_lam_sparse = sparse_model.spike_lam
    spike_lam_dense = dense_model.spike_lam

    save_path = f"results/run_logs/run_Input{input_size}_T{n_frames}_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": timestamp,
            "sparse_model": {
                "input_size": input_size,
                "n_frames": n_frames,
                "tau_mem": tau_mem_sparse,
                "spike_lam": spike_lam_sparse,
                "model_type": "sparse"
            },
            "dense_model": {
                "input_size": input_size,
                "n_frames": n_frames,
                "tau_mem": tau_mem_dense,
                "spike_lam": spike_lam_dense,
                "model_type": "dense"
            },
        },
        "roc_results": {
            "optimal_threshold": float(optimal_threshold),
            "roc_auc": float(roc_auc)
        },
        "routing_metrics": {
            "overall_accuracy": float(total_accuracy),
            "dense_routed_accuracy": float(accuracy_dense_routed),
            "sparse_routed_accuracy": float(accuracy_sparse_routed),
            "route_counts": route_counts,
            "avg_dense_spikes": float(average_spike_dense),
            "avg_sparse_spikes": float(average_spike_sparse)
        },
        "per_sample_results": results
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved run results to: {save_path}\n")


def print_latex_table(total_accuracy,
                      accuracy_dense_routed,
                      accuracy_sparse_routed,
                      avg_dense_spikes,
                      avg_sparse_spikes,
                      route_counts,
                      roc_auc,
                      optimal_threshold):
    """
    Print LaTeX table with routing performance metrics.

    Args:
        total_accuracy: Overall routed accuracy
        accuracy_dense_routed: Accuracy on dense-routed samples
        accuracy_sparse_routed: Accuracy on sparse-routed samples
        avg_dense_spikes: Average spikes for dense model
        avg_sparse_spikes: Average spikes for sparse model
        route_counts: Dictionary with routing counts
        roc_auc: ROC AUC score
        optimal_threshold: Optimal LZC threshold
    """
    print("\n\n===================== LATEX TABLE =====================\n")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{l c}")
    print(r"\hline")
    print(r"Metric & Value \\")
    print(r"\hline")
    print(fr"Optimal threshold & {optimal_threshold:.2f} \\")
    print(fr"ROC-AUC & {roc_auc:.3f} \\")
    print(fr"Total accuracy & {total_accuracy:.3f} \\")
    print(fr"Dense-route accuracy & {accuracy_dense_routed:.3f} \\")
    print(fr"Sparse-route accuracy & {accuracy_sparse_routed:.3f} \\")
    print(fr"Avg dense spikes & {avg_dense_spikes:.1f} \\")
    print(fr"Avg sparse spikes & {avg_sparse_spikes:.1f} \\")
    print(fr"Samples to dense & {route_counts['dense']} \\")
    print(fr"Samples to sparse & {route_counts['sparse']} \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Routing performance and spike activity.}")
    print(r"\end{table}")
    print("\n=======================================================\n")


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
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation (default: 1 for routing)')
    parser.add_argument('--num_workers', type=int, default=4,
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
    # print("Loading SHD dataset...")
    # data = SHDDataset(
    #     dataset_path=args.dataset_path,
    #     NUM_CHANNELS=args.NUM_CHANNELS,
    #     NUM_POLARITIES=args.NUM_POLARITIES,
    #     n_frames=args.n_frames,
    #     net_dt=args.net_dt
    # )
    # _, cached_test = data.load_shd()
    # test_loader = DataLoader(
    #     cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
    #     num_workers=args.num_workers, pin_memory=True,
    #     collate_fn=tonic.collation.PadTensors(batch_first=True)
    # )

    # ----- UCI-HAR Dataset -----
    print("Loading UCI-HAR dataset...")
    data = UCIHARDataset(
        dataset_path=args.dataset_path,
        n_frames=128,
        time_first=True,
        normalize=True
    )
    _, cached_test = data.load_uci_har()
    test_loader = DataLoader(
        cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True
    )

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
    sparse_hp = BaseSNNModel.load_hyperparams(args.sparse_model_path)
    dense_hp = BaseSNNModel.load_hyperparams(args.dense_model_path)

    print(f"Sparse model: tau_mem={sparse_hp['tau_mem']}, tau_syn={sparse_hp['tau_syn']}, spike_lam={sparse_hp['spike_lam']}")
    print(f"Dense model:  tau_mem={dense_hp['tau_mem']}, tau_syn={dense_hp['tau_syn']}, spike_lam={dense_hp['spike_lam']}")

    # ==================== MODEL CREATION ====================
    # Uncomment ONE of the following model sections (must match dataset above):

    # ----- SHD Models -----
    # print("\nCreating SHD sparse model...")
    # sparse_model = SHDSNN_FC(
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
    # print("Creating SHD dense model...")
    # dense_model = SHDSNN_FC(
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

    # ----- UCI-HAR Models -----
    print("\nCreating UCI-HAR sparse model...")
    sparse_model = UCIHARSNN_FC(
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
    print("Creating UCI-HAR dense model...")
    dense_model = UCIHARSNN_FC(
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
    # sparse_model = DVSGestureSNN_FC(
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
    # dense_model = DVSGestureSNN_FC(
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

    # 1. Evaluate both models on all test samples
    results = evaluate_models_on_dataset(test_loader, sparse_model, dense_model)

    # 2. Plot LZC vs accuracy sweep
    lzc_vs_accuracy_plot(results, sparse_model, dense_model)

    # 3. Find optimal threshold via ROC analysis
    optimal_threshold, roc_auc, avg_dense_spikes, avg_sparse_spikes = \
        threshold_sweep_and_roc(results, sparse_model, dense_model)

    # 4. Route samples and evaluate
    total_accuracy, accuracy_dense_routed, accuracy_sparse_routed, route_counts = \
        route_and_evaluate(test_loader, sparse_model, dense_model, optimal_threshold, results)

    # 5. Save results to JSON
    save_run_to_json(
        results, optimal_threshold, roc_auc, route_counts,
        accuracy_dense_routed, accuracy_sparse_routed, total_accuracy,
        avg_dense_spikes, avg_sparse_spikes,
        sparse_model, dense_model
    )

    # 6. Print LaTeX table
    print_latex_table(
        total_accuracy, accuracy_dense_routed, accuracy_sparse_routed,
        avg_dense_spikes, avg_sparse_spikes, route_counts,
        roc_auc, optimal_threshold
    )


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
