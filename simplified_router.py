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


def threshold_sweep_and_roc(results, dataset_name="unknown", plotting_only=False):
    """
    Perform threshold sweep, compute ROC-AUC curve, and find optimal LZC threshold.

    Args:
        results: Per-sample results from evaluate_models_on_dataset
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
    
    # Limit to 200 samples for faster testing
    import itertools
    test_loader = list(itertools.islice(test_loader, 3))

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
    sparse_hp = UCIHARSNN.load_hyperparams(args.sparse_model_path)
    dense_hp = UCIHARSNN.load_hyperparams(args.dense_model_path)

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
    sparse_model = UCIHARSNN(
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
    dense_model = UCIHARSNN(
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

    # ----- UCI-HAR -----
    optimal_threshold, roc_auc, avg_dense_spikes, avg_sparse_spikes = threshold_sweep_and_roc(results, dataset_name="uci_har")
    
    # 4. Route samples and evaluate
    total_accuracy, accuracy_dense_routed, accuracy_sparse_routed, route_counts = \
        route_and_evaluate(test_loader, sparse_model, dense_model, optimal_threshold, results)


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
