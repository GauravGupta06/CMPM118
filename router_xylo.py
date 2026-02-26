"""
Router for Rockpool SNN models.
Analyzes complexity of samples and routes between sparse and dense models for energy efficiency.

Usage:
    python router.py --sparse_model_path <path> --dense_model_path <path> [options]

Example:
    python router.py \
        --sparse_model_path ./workspace/small/models/Rockpool_Sparse_Take1_HAR_Input9_T128_FC_Rockpool_Epochs100.pth \
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

from rockpool.devices.xylo import find_xylo_hdks
from rockpool.devices.xylo import syns65302 as xa3 # XyloAudio 3
from rockpool.transform import quantize_methods as q
import samna

# UCI HAR
from datasets.uci_har import UCIHARDataset
from models.uci_har_model import UCIHARSNN



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

# ========== CORE FUNCTIONS ==========

def evaluate_models_on_dataset(dataLoader, sparse_model, dense_model):
    """
    Evaluate both sparse and dense models on entire dataset.
    Supports batched forward passes for efficiency.

    Args:
        dataLoader: Test data loader (can have batch_size > 1)
        sparse_model: Sparse SNN model
        dense_model: Dense SNN model

    Returns:
        list: Results with per-sample metrics
    """
    results = []
    count = 0
    
    for batch in dataLoader:
        events, labels = batch
        batch_size = events.shape[0]
        
        # Move data to same device as model
        events = events.to(sparse_model.device)
        
        # Batched forward pass (efficient!)
        with torch.no_grad():
            sparse_output, _, sparse_recording = sparse_model.net(events, record=True)
            dense_output, _, dense_recording = dense_model.net(events, record=True)
        
        # Get predictions for entire batch
        sparse_logits = sparse_output.mean(dim=1)  # [B, num_classes]
        dense_logits = dense_output.mean(dim=1)    # [B, num_classes]
        sparse_preds = sparse_logits.argmax(1)     # [B]
        dense_preds = dense_logits.argmax(1)       # [B]
        
        # Process each sample in the batch for per-sample metrics
        for i in range(batch_size):
            # Get single sample for LZC computation
            single_event = events[i:i+1]  # Keep batch dim
            lz_value = compute_lzc_from_events(single_event)
            
            # Extract per-sample values
            label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
            sparse_pred = sparse_preds[i].item()
            dense_pred = dense_preds[i].item()
            
            # Set as complex IF dense_pred matches label and sparse_pred does NOT
            if dense_pred == label and sparse_pred != label:
                true_complex = 1
            else:
                true_complex = 0
            
            # Spike counts (approximate per-sample from batch recording)
            spike_count_sparse = count_spikes_from_recording(sparse_recording) // batch_size
            spike_count_dense = count_spikes_from_recording(dense_recording) // batch_size
            
            results.append({
                'label': label,
                'lz_value': lz_value,
                'sparse_pred': sparse_pred,
                'dense_pred': dense_pred,
                'true_complex': true_complex,
                'dense_spikes': spike_count_dense,
                'sparse_spikes': spike_count_sparse
            })
        
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} samples")
    
    return results


def threshold_sweep_and_roc(results, sparse_model, dense_model, dataset_name="unknown", plotting_only=False):
    """
    Perform threshold sweep, compute ROC-AUC curve, and find optimal LZC threshold.

    Args:
        results: Per-sample results from evaluate_models_on_dataset
        sparse_model: Sparse model (for extracting parameters)
        dense_model: Dense model (for extracting parameters)
        dataset_name: Name of the dataset ("shd", "dvsgesture", "uci_har")
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
    plt.show()

    return optimal_threshold, roc_auc, average_spike_dense, average_spike_sparse


def route_and_evaluate(dataLoader, sparse_model, dense_model, optimal_threshold, results):
    """
    Route samples to appropriate model and evaluate accuracy.
    Supports batched dataloaders.

    Args:
        dataLoader: Test data loader (can have batch_size > 1)
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

    # Flatten results into a sample index
    sample_idx = 0

    for batch in dataLoader:
        events, labels = batch
        batch_size = events.shape[0]

        # Move data to same device as model
        events = events.to(sparse_model.device)

        # Process each sample in the batch individually for routing
        for i in range(batch_size):
            if sample_idx >= len(results):
                break
                
            single_event = events[i:i+1]  # Keep batch dim [1, T, C]
            label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
            lz_value = results[sample_idx]['lz_value']

            with torch.no_grad():
                if lz_value < optimal_threshold:
                    route_counts['sparse'] += 1
                    sparse_output, _, _ = sparse_model.net(single_event, record=False)
                    sparse_logits = sparse_output.mean(dim=1)
                    pred = sparse_logits.argmax(1).item()
                    if pred == label:
                        correct_sparse += 1
                else:
                    route_counts['dense'] += 1
                    dense_output, _, _ = dense_model.net(single_event, record=False)
                    dense_logits = dense_output.mean(dim=1)
                    pred = dense_logits.argmax(1).item()
                    if pred == label:
                        correct_dense += 1
            
            sample_idx += 1

    accuracy_dense_routed = correct_dense / route_counts['dense'] if route_counts['dense'] > 0 else 0
    accuracy_sparse_routed = correct_sparse / route_counts['sparse'] if route_counts['sparse'] > 0 else 0

    total_correct = correct_dense + correct_sparse
    total_samples = route_counts['sparse'] + route_counts['dense']
    total_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Getting the accuracy on each model for the entire dataset
    total_samples = len(results)

    # Sparse model accuracy
    sparse_correct = sum(1 for r in results if r['sparse_pred'] == r['label'])
    sparse_accuracy_overall = sparse_correct / total_samples

    # Dense model accuracy
    dense_correct = sum(1 for r in results if r['dense_pred'] == r['label'])
    dense_accuracy_overall = dense_correct / total_samples

    sparse_accuracy_improvement = accuracy_sparse_routed/sparse_accuracy_overall - 1 if sparse_accuracy_overall > 0 else 0
    dense_accuracy_improvement = accuracy_dense_routed/dense_accuracy_overall - 1 if dense_accuracy_overall > 0 else 0

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

# ========== XYLO ==========

def collect_samples_from_loader(test_loader):
    """
    Flatten test_loader (list of batches) into a list of numpy samples and labels
    in the same order evaluate_models_on_dataset iterates.
    Each sample is returned as a numpy array with shape [T, C] (time-first).
    """
    samples = []
    labels = []
    for batch in test_loader:
        events, labs = batch
        # ensure tensor -> numpy
        if torch.is_tensor(events):
            ev_np = events.cpu().numpy()
        else:
            ev_np = np.asarray(events)
        if torch.is_tensor(labs):
            labs_np = labs.cpu().numpy()
        else:
            labs_np = np.asarray(labs)
        # ev_np shape expected: [B, T, C]
        B = ev_np.shape[0]
        for i in range(B):
            samples.append(ev_np[i])       # shape [T, C]
            labels.append(int(labs_np[i]))
    return samples, labels


def build_xylo_config_from_graph(graph, dt):
    """
    Given a Rockpool graph (net.as_graph()), map to Xylo spec, quantize,
    and return a validated Xylo config object.

    Returns: (config, quant_spec)
    """
    spec = xa3.mapper(
        graph,
        weight_dtype='float',
        threshold_dtype='float',
        dash_dtype='float'
    )
    spec['dt'] = dt

    quant_spec = q.channel_quantize(**spec)

    for key in ['dt', 'aliases']:
        if key in spec:
            quant_spec[key] = spec[key]

    for key in ["dash_mem", "dash_mem_out", "dash_syn", "dash_syn_2", "dash_syn_out"]:
        if key in quant_spec:
            quant_spec[key] = np.abs(quant_spec[key]).astype(np.uint8)

    config, is_valid, msg = xa3.config_from_specification(**quant_spec)
    if not is_valid:
        raise ValueError(f"Invalid Xylo configuration: {msg}")

    # Extra sanity check with samna
    is_valid2, msg2 = samna.xyloAudio3.validate_configuration(config)
    if not is_valid2:
        raise ValueError(f"Samna validation failed: {msg2}")

    return config, quant_spec


def run_samples_on_xylo(modSamna, samples, record_power=True, verbose=True, dtype=np.int32):
    """
    Run a list of samples (each shape [T, C]) through the provided modSamna instance.
    Returns per-sample metrics and an aggregate summary.

    Output:
      stats = {
        'per_sample': [
            {'energy_j': float, 'mean_power': float, 'peak_power': float,
             'inf_duration': float, 'n_input_spikes': int, 'n_output_spikes': int,
             'pred': int, 'label': maybe None}
        ],
        'total_energy': float,
        'avg_energy_per_sample': float,
        'total_samples': int,
        'failed_indices': [i, ...]
      }
    """
    per_sample = []
    total_energy = 0.0
    failed = []

    for i, s in enumerate(samples):
        try:
            # Ensure numpy array in expected dtype/shape
            sample_np = np.asarray(s).astype(dtype)

            # If modSamna expects shape [T, N] for a single sample that's what we pass
            output, state, recorded = modSamna(sample_np, record=True, record_power=record_power)

            # recorded keys we've seen: analog_power, digital_power, io_power, inf_duration, Spikes, times
            analog = recorded.get('analog_power')
            digital = recorded.get('digital_power')
            io = recorded.get('io_power')

            if analog is None or digital is None or io is None:
                # fallback: treat energy as 0 and log warning
                if verbose:
                    print(f"[WARN] sample {i}: power traces missing in recorded, skipping energy calc")
                energy_j = 0.0
                mean_power = 0.0
                peak_power = 0.0
                inf_duration = float(recorded.get('inf_duration', 0.0))
            else:
                total_power = np.asarray(analog) + np.asarray(digital) + np.asarray(io)
                inf_duration = float(recorded.get('inf_duration', np.nan))
                if np.isnan(inf_duration) or inf_duration == 0 or len(total_power) == 0:
                    # fallback safe: set energy to sum*dt where dt guessed from sample length
                    # but prefer recorded inf_duration
                    power_dt = inf_duration / len(total_power) if (not np.isnan(inf_duration) and len(total_power)>0) else (inf_duration if not np.isnan(inf_duration) else 0.0)
                    energy_j = float(np.sum(total_power) * power_dt) if power_dt>0 else float(np.sum(total_power)) # best effort
                else:
                    power_dt = inf_duration / len(total_power)
                    energy_j = float(np.sum(total_power) * power_dt)

                mean_power = float(np.mean(total_power)) if len(total_power)>0 else 0.0
                peak_power = float(np.max(total_power)) if len(total_power)>0 else 0.0

            # output -> predictions (robust handling: could be numpy or tensor)
            if isinstance(output, np.ndarray):
                out_np = output
            elif torch.is_tensor(output):
                out_np = output.cpu().numpy()
            else:
                out_np = np.asarray(output)

            # try to reduce to class logits: [T, C] or [T, B, C] etc -> sum over time
            try:
                if out_np.ndim == 2:
                    # [T, C] -> average over time
                    logits = out_np.mean(axis=0)
                elif out_np.ndim == 3:
                    # If shape [T, B, C] assume B==1 and reduce to [T, C]
                    if out_np.shape[1] == 1:
                        logits = out_np.mean(axis=0).mean(axis=0)  # (T,1,C)->(1,C) -> mean over time
                    else:
                        # fallback: mean over first axis
                        logits = out_np.mean(axis=0).mean(axis=0)
                elif out_np.ndim == 1:
                    logits = out_np
                else:
                    logits = out_np.mean()
                pred = int(np.argmax(logits))
            except Exception:
                pred = None

            # spike counts if available
            n_output_spikes = int(np.sum(recorded.get('Spikes', 0))) if recorded.get('Spikes') is not None else 0
            # input spikes: if you have input spike trains count, but here sample is the input
            n_input_spikes = int(np.sum(sample_np)) if sample_np is not None else 0

            per_sample.append({
                'index': i,
                'energy_j': energy_j,
                'mean_power': mean_power,
                'peak_power': peak_power,
                'inf_duration': inf_duration,
                'n_output_spikes': n_output_spikes,
                'n_input_spikes': n_input_spikes,
                'pred': pred
            })

            total_energy += energy_j

            if verbose and (i % 10 == 0):
                print(f"[XYLO] processed sample {i}: energy {energy_j:.3e} J, inf_time {inf_duration:.3f}s")
        except Exception as e:
            failed.append((i, str(e)))
            if verbose:
                print(f"[ERROR] running sample {i} on XYLO: {e}")
            # continue with next sample
            continue

    total_samples = len(per_sample)
    avg_energy = total_energy / total_samples if total_samples > 0 else 0.0

    stats = {
        'per_sample': per_sample,
        'total_energy': total_energy,
        'avg_energy_per_sample': avg_energy,
        'total_samples': total_samples,
        'failed_indices': failed
    }
    return stats

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
    parser.add_argument('--input_size', type=int, default=9,
                       help='Number of frequency bins (default: 700)')
    parser.add_argument('--n_frames', type=int, default=128,
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
    parser.add_argument('--net_dt', type=float, default=0.02,
                       help='Simulation time step in seconds (default: 10e-3)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation (default: 1 for routing)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers (default: 4)')
    parser.add_argument('--model_type', type=str, default='dense',
                       help='Model type for both models (default: dense)')

    # Parse arguments
    args = parser.parse_args()
    """
    python router_xylo.py --sparse_model_path workspace/uci_har/sparse/models/Take1_T128_Epochs20.pth --dense_model_path workspace/uci_har/dense/models/Take2_T128_Epochs20.pth
    """

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

    # ----- UCI-HAR Dataset -----
    print("Loading UCI-HAR dataset...")
    data = UCIHARDataset(
        dataset_path=args.dataset_path,
        n_frames=128,
        time_first=True,
        normalize=True,
        binarize=True
    )
    _, cached_test = data.load_uci_har()
    test_loader = DataLoader(
        cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Limit to 200 samples for faster testing
    import itertools
    test_loader = list(itertools.islice(test_loader, 75))

    # ==================== END DATASET LOADING ====================

    # Load hyperparameters from checkpoint files
    print("\nLoading hyperparameters from checkpoints...")
    sparse_hp = UCIHARSNN.load_hyperparams(args.sparse_model_path)
    dense_hp = UCIHARSNN.load_hyperparams(args.dense_model_path)

    print(f"Sparse model: tau_mem={sparse_hp['tau_mem']}, tau_syn={sparse_hp['tau_syn']}, spike_lam={sparse_hp['spike_lam']}")
    print(f"Dense model:  tau_mem={dense_hp['tau_mem']}, tau_syn={dense_hp['tau_syn']}, spike_lam={dense_hp['spike_lam']}")

    # ==================== MODEL CREATION ====================

    # ----- UCI-HAR Models -----
    print("\nCreating UCI-HAR sparse model...")
    sparse_model = UCIHARSNN(
        input_size=sparse_hp['input_size'],
        n_frames=sparse_hp['n_frames'],
        tau_mem=sparse_hp['tau_mem'],
        tau_syn=sparse_hp['tau_syn'],
        spike_lam=sparse_hp['spike_lam'],
        #model_type=sparse_hp['model_type'],
        model_type="sparse",
        device=device,
        num_classes=sparse_hp['num_classes'],
        dt=sparse_hp['dt'],
        threshold=sparse_hp['threshold'],
        has_bias=sparse_hp['has_bias']
    )
    print("Creating UCI-HAR dense model...")
    dense_model = UCIHARSNN(
        input_size=dense_hp['input_size'],
        n_frames=dense_hp['n_frames'],
        tau_mem=dense_hp['tau_mem'],
        tau_syn=dense_hp['tau_syn'],
        spike_lam=dense_hp['spike_lam'],
        #model_type=dense_hp['model_type'],
        model_type="dense",
        device=device,
        num_classes=dense_hp['num_classes'],
        dt=dense_hp['dt'],
        threshold=dense_hp['threshold'],
        has_bias=dense_hp['has_bias']
    )

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

    # # 2. Plot LZC vs accuracy sweep
    # lzc_vs_accuracy_plot(results, sparse_model, dense_model)

    # 3. Find optimal threshold via ROC analysis
    
    # ----- UCI-HAR -----
    optimal_threshold, roc_auc, avg_dense_spikes, avg_sparse_spikes = threshold_sweep_and_roc(results, sparse_model, dense_model, dataset_name="uci_har")

    # 4. Route samples and evaluate
    total_accuracy, accuracy_dense_routed, accuracy_sparse_routed, route_counts = \
        route_and_evaluate(test_loader, sparse_model, dense_model, optimal_threshold, results)
    
    # 5. Use Xylo to show energy metrics
    # ------------- usage: compute buckets and run each model on HDK  ----------------
    print("\n" + "="*80)
    print("XYLO")
    print("="*80  + "\n")

    # After you have `results` and `test_loader` and you have created hdk and both graphs:
    # 1) collect raw samples in same order
    samples_list, labels_list = collect_samples_from_loader(test_loader)

    # 2) compute the optimal_threshold (you already did). We'll assume you have it:
    # optimal_threshold, roc_auc, avg_dense_spikes, avg_sparse_spikes = threshold_sweep_and_roc(...)

    # 3) bucket indices
    sparse_indices = [i for i, r in enumerate(results) if r['lz_value'] < float(optimal_threshold)]
    dense_indices = [i for i, r in enumerate(results) if r['lz_value'] >= float(optimal_threshold)]

    print(f"Samples routed to sparse model: {len(sparse_indices)}")
    print(f"Samples routed to dense model:  {len(dense_indices)}")

    # 4) bucket sample arrays
    sparse_samples = [samples_list[i] for i in sparse_indices]
    dense_samples = [samples_list[i] for i in dense_indices]

    # 5) build configs for both graphs (use your previously created sparse_graph/dense_graph)
    sparse_graph = sparse_model.net.as_graph()
    print(sparse_graph) # GraphHolder "TorchSequential_xxx" with N input nodes -> M output nodes

    dense_graph = dense_model.net.as_graph()
    print(dense_graph)

    sparse_config, sparse_quant_spec = build_xylo_config_from_graph(sparse_graph, args.net_dt)
    dense_config, dense_quant_spec = build_xylo_config_from_graph(dense_graph, args.net_dt)

    print("Sparse config: hidden neurons", len(sparse_config.hidden.weights))
    print("Dense config:  hidden neurons", len(dense_config.hidden.weights))

    # 6) load each model on same HDK, run its sample bucket and collect energy stats
    # (we recreate a modSamna instance for each config)
    hdk_nodes, support_modules, versions = find_xylo_hdks()
    if not hdk_nodes:
        raise RuntimeError("No Xylo HDK found")
    hdk = hdk_nodes[0]

    print("Deploying sparse model to HDK...")
    modSamna_sparse = xa3.XyloSamna(hdk, sparse_config, dt=args.net_dt, power_frequency=20.0)
    sparse_stats = run_samples_on_xylo(modSamna_sparse, sparse_samples, record_power=True, verbose=True)

    print("Deploying dense model to HDK...")
    modSamna_dense = xa3.XyloSamna(hdk, dense_config, dt=args.net_dt, power_frequency=20.0)
    dense_stats = run_samples_on_xylo(modSamna_dense, dense_samples, record_power=True, verbose=True)

    # 7) Summarize
    print("\nXYLO ENERGY SUMMARY")
    print("--------------------")
    print("Sparse model: samples:", sparse_stats['total_samples'])
    print("Sparse total energy (J):", sparse_stats['total_energy'])
    print("Sparse avg energy/sample (J):", sparse_stats['avg_energy_per_sample'])
    print("Sparse failed runs:", len(sparse_stats['failed_indices']))

    print("Dense model: samples:", dense_stats['total_samples'])
    print("Dense total energy (J):", dense_stats['total_energy'])
    print("Dense avg energy/sample (J):", dense_stats['avg_energy_per_sample'])
    print("Dense failed runs:", len(dense_stats['failed_indices']))
    
    # ===== ENERGY SAVINGS CALCULATION =====

    energy_routed = sparse_stats['total_energy'] + dense_stats['total_energy']
    energy_dense_only = dense_stats['avg_energy_per_sample'] * len(results)
    energy_saved = energy_dense_only - energy_routed

    print("\nENERGY COMPARISON")
    print("--------------------")
    print(f"Dense-only total energy (estimated): {energy_dense_only:.6f} J")
    print(f"Routed total energy:                {energy_routed:.6f} J")
    print(f"Energy saved:                       {energy_saved:.6f} J")
    print(f"Percent energy saved:               {(energy_saved / energy_dense_only) * 100:.2f}%")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
