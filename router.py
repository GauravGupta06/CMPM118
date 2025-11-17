# make sure to have do "pip install PyQT6" otherwise the plt graph for the ROC curve might not show up. 
import argparse
import json
import os

import numpy as np
import tonic
import torch
from lempel_ziv_complexity import lempel_ziv_complexity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import entropy

# user made imports
from SNN_model_inheritance import DVSGestureSNN
from LoadDataset import load_dataset
from neuromorphic.energy_profiler import (
    NeuromorphicHardwareConfig,
    NullEnergyProfiler,
    RouterEnergyProfiler,
    XyloEnergyProfiler,
)

 # Model hyperparameters
w_large = 32
h_large = 32
n_frames_large = 32

w_small = 32
h_small = 32
n_frames_small = 32

def compute_lzc_from_events(events):

    # for events to be passed into the lempel_ziv_complexity() function, it needs to be a string. To do this, we convert events
    # which is originally a tensor, into a numpy array, and then convert that into a string before passing it into lempel_ziv_complexity(). 
    if torch.is_tensor(events):
        events = events.cpu().numpy()


    spike_seq = (events).astype(int).flatten()
    spike_seq_string = ''.join(map(str, spike_seq.tolist()))
    lz_score = lempel_ziv_complexity(spike_seq_string)
    return lz_score

def compute_shannon_entropy_from_events(events):
    flattened = events.cpu().numpy().astype(int).flatten()

    values, counts = np.unique(flattened, return_counts=True)
    probs = counts / counts.sum()

    entropy_value = entropy(probs, base=2)

    return entropy_value

def compute_isi_entropy_from_events(events, num_bins = 30):
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

def evaluate_models_on_dataset(
    dataLoader,
    sparse_model,
    dense_model,
    router_energy_profiler: RouterEnergyProfiler,
    bin_size=0.005,
):
    results = []

    for batch in dataLoader:
        events, label = batch



        # the events size has the a batch dimension as well. But because our batch size is just 1, we have to get rid of this deminsion
        if events.dim() >= 4:
            events = events.squeeze(1) # this will remove the dimension in index 1

        # for the labels, we need it to be of type int. So we check to see if its a tensor or not, and if it is, then we convert it to int
        if torch.is_tensor(label):
            if label.dim() == 0:
                label = label.item()
            else:
                label = label[0].item() if label.shape[0] == 1 else label.item()



        lz_value = compute_lzc_from_events(events)
        router_energy = router_energy_profiler.record()
        sparse_pred, spike_count_sparse, sparse_energy = sparse_model.predict_sample(events)
        dense_pred, spike_count_dense, dense_energy = dense_model.predict_sample(events)
        
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
            'sparse_spikes': spike_count_sparse,
            'dense_energy': dense_energy,
            'sparse_energy': sparse_energy,
            'router_energy': router_energy,
        })
    return results


def format_energy(energy_j: float) -> str:
    if energy_j >= 1e-3:
        return f"{energy_j * 1e3:.4f} mJ"
    if energy_j >= 1e-6:
        return f"{energy_j * 1e6:.2f} µJ"
    return f"{energy_j * 1e9:.2f} nJ"


def threshold_sweep_and_roc(results):
    """Threshold sweep, ROC-AUC curve, and optimal LZC threshold."""
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
    avg_router_energy = np.mean([r['router_energy'] for r in results])
    avg_dense_energy = np.mean([r['dense_energy'] for r in results])
    avg_sparse_energy = np.mean([r['sparse_energy'] for r in results])
    avg_total_dense_path = np.mean([r['router_energy'] + r['dense_energy'] for r in results])
    avg_total_sparse_path = np.mean([r['router_energy'] + r['sparse_energy'] for r in results])

    print(f"average spike dense: {average_spike_dense:.2f}")
    print(f"average spike sparse: {average_spike_sparse:.2f}")
    print(f"Average router energy per sample: {format_energy(avg_router_energy)}")
    print(f"Average dense-model energy per sample: {format_energy(avg_dense_energy)}")
    print(f"Average sparse-model energy per sample: {format_energy(avg_sparse_energy)}")
    print(f"Average total energy if always dense: {format_energy(avg_total_dense_path)}")
    print(f"Average total energy if always sparse: {format_energy(avg_total_sparse_path)}")

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

    graph_save_path = f"results/ROC_curves:{w_small}x{h_small}_T{n_frames_small}_Large:{w_large}x{h_large}_T{n_frames_large}.png"
    plt.savefig(graph_save_path)
    plt.show()
    
    return optimal_threshold


def route_and_evaluate(
    dataLoader,
    sparse_model,
    dense_model,
    optimal_threshold,
    results,
    energy_metrics_path: str = None,
):
    """Route samples to appropriate model and evaluate accuracy."""
    print("\nRouting and evaluating with threshold:", optimal_threshold, "\n")
    correct_sparse = 0
    correct_dense = 0
    route_counts = {'sparse': 0, 'dense': 0}

    lz_values = [r['lz_value'] for r in results]
    total_energy_consumption = 0.0
    energy_totals = {'router': 0.0, 'dense': 0.0, 'sparse': 0.0}

    for i, batch in enumerate(dataLoader):
        events, label = batch



        # the events size has the a batch dimension as well. But because our batch size is just 1, we have to get rid of this deminsion
        if events.dim() >= 4:
            events = events.squeeze(1) # this will remove the dimension in index 1

        # for the labels, we need it to be of type int. So we check to see if its a tensor or not, and if it is, then we convert it to int
        if torch.is_tensor(label):
            if label.dim() == 0:
                label = label.item()
            else:
                label = label[0].item() if label.shape[0] == 1 else label.item()



        lz_value = lz_values[i]
        sample_result = results[i]
        energy_totals['router'] += sample_result['router_energy']

        if lz_value < optimal_threshold:
            route_counts['sparse'] += 1
            pred, _, _ = sparse_model.predict_sample(events)
            energy_totals['sparse'] += sample_result['sparse_energy']
            total_energy_consumption += sample_result['router_energy'] + sample_result['sparse_energy']
            if pred == label:
                correct_sparse += 1
        else:
            route_counts['dense'] += 1
            pred, _, _ = dense_model.predict_sample(events)
            energy_totals['dense'] += sample_result['dense_energy']
            total_energy_consumption += sample_result['router_energy'] + sample_result['dense_energy']
            if pred == label:
                correct_dense += 1
    
    accuracy_dense_routed = correct_dense / route_counts['dense'] if route_counts['dense'] > 0 else 0
    accuracy_sparse_routed = correct_sparse / route_counts['sparse'] if route_counts['sparse'] > 0 else 0

    total_correct = correct_dense + correct_sparse
    total_routed_samples = route_counts['sparse'] + route_counts['dense']
    total_accuracy = total_correct / total_routed_samples
    avg_energy_per_inference = (
        total_energy_consumption / total_routed_samples if total_routed_samples else 0.0
    )
    always_dense_energy = sum(r['router_energy'] + r['dense_energy'] for r in results)
    always_sparse_energy = sum(r['router_energy'] + r['sparse_energy'] for r in results)
    energy_savings_vs_dense = (
        1 - (total_energy_consumption / always_dense_energy)
        if always_dense_energy
        else 0.0
    )






    # getting the accuracy on each model based for the entire dataset. 
    dataset_samples = len(results)

    # Sparse model accuracy
    sparse_correct = sum(1 for r in results if r['sparse_pred'] == r['label'])
    sparse_accuracy_overall = sparse_correct / dataset_samples

    # Dense model accuracy
    dense_correct = sum(1 for r in results if r['dense_pred'] == r['label'])
    dense_accuracy_overall = dense_correct / dataset_samples

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
    print(f"Total Samples: {dataset_samples}")
    print(f"Average energy per inference (router + model): {format_energy(avg_energy_per_inference)}")
    print(f"Energy savings vs always running dense model: {energy_savings_vs_dense * 100: .2f}%")

    energy_summary = {
        'avg_energy_per_inference_j': avg_energy_per_inference,
        'total_energy_consumption_j': total_energy_consumption,
        'avg_router_energy_j': energy_totals['router'] / dataset_samples if dataset_samples else 0.0,
        'avg_sparse_energy_j': (
            energy_totals['sparse'] / route_counts['sparse'] if route_counts['sparse'] else 0.0
        ),
        'avg_dense_energy_j': (
            energy_totals['dense'] / route_counts['dense'] if route_counts['dense'] else 0.0
        ),
        'always_dense_energy_j': always_dense_energy,
        'always_sparse_energy_j': always_sparse_energy,
        'energy_savings_vs_dense_pct': energy_savings_vs_dense * 100,
    }

    if energy_metrics_path:
        os.makedirs(os.path.dirname(energy_metrics_path) or '.', exist_ok=True)
        with open(energy_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(energy_summary, f, indent=2)
        print(f"Energy metrics saved to {energy_metrics_path}")

    return (
        total_accuracy,
        accuracy_dense_routed,
        accuracy_sparse_routed,
        route_counts,
        energy_summary,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="SNN router with energy profiling")
    parser.add_argument(
        '--hardware-backend',
        choices=['xylo', 'none'],
        default='xylo',
        help='Neuromorphic backend used for energy estimation.',
    )
    parser.add_argument('--spike-energy-pj', type=float, default=23.6)
    parser.add_argument('--synaptic-event-energy-pj', type=float, default=4.1)
    parser.add_argument('--neuron-leak-energy-pj', type=float, default=0.5)
    parser.add_argument('--static-energy-pj', type=float, default=0.0)
    parser.add_argument('--router-energy-pj', type=float, default=9.2)
    parser.add_argument('--synaptic-scaling', type=float, default=4.0)
    parser.add_argument('--calibration-factor', type=float, default=1.0)
    parser.add_argument(
        '--energy-metrics-path',
        type=str,
        default='results/energy_metrics.json',
        help='Where to store aggregated energy statistics.',
    )
    return parser.parse_args()


def build_model_profiler(args):
    if args.hardware_backend == 'none':
        return NullEnergyProfiler()

    config = NeuromorphicHardwareConfig(
        name=args.hardware_backend,
        spike_energy_pj=args.spike_energy_pj,
        synaptic_event_energy_pj=args.synaptic_event_energy_pj,
        neuron_leak_energy_pj=args.neuron_leak_energy_pj,
        router_energy_pj=args.router_energy_pj,
        static_energy_pj=args.static_energy_pj,
    )
    return XyloEnergyProfiler(
        config=config,
        synaptic_scaling=args.synaptic_scaling,
        calibration_factor=args.calibration_factor,
    )

def main(args=None):
    if args is None:
        args = parse_args()
    # Setup device
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(device)



    # Load in the preprocessed dataset
    # dataset_root = f"data/dvsgesture/{w_large}x{h_large}_T{n_frames_large}"
    # dataset = tonic.DiskCachedDataset(None, cache_path=f"{dataset_root}/test")

    cached_train, cached_test, num_classes = load_dataset(
        dataset_name="DVSGesture",  # or "ASLDVS"
        dataset_path = "./data",
        #dataset_path='/home/gauravgupta/CMPM118/data',
        w=32,
        h=32,
        n_frames=32
    )


    active_cores = 1
    test_loader = torch.utils.data.DataLoader(
        cached_test, 
        batch_size=1,  # Process one sample at a time for routing
        shuffle=False,  # Don't shuffle for consistent evaluation
        num_workers=active_cores,  # Use multiple cores
        drop_last=False,  # Keep all samples
        collate_fn=tonic.collation.PadTensors(batch_first=False)
    )



    dense_energy_profiler = build_model_profiler(args)
    sparse_energy_profiler = build_model_profiler(args)
    router_energy_profiler = RouterEnergyProfiler(router_energy_pj=args.router_energy_pj)

    # Create and load dense model
    dense_model = DVSGestureSNN(
        w=w_large,
        h=h_large,
        n_frames=n_frames_large,
        beta=0.8,
        spike_lam=0,
        slope=25,
        model_type="dense",
        device=device,
        energy_profiler=dense_energy_profiler,
    )
    dense_model.load_model("results/large/models/Non_Sparse_Take6_32x32_T32.pth")

    # Create and load sparse model
    sparse_model = DVSGestureSNN(
        w=w_small,
        h=h_small,
        n_frames=n_frames_small,
        beta=0.4,
        spike_lam=1e-7,
        slope=25,
        model_type="sparse",
        device=device,
        energy_profiler=sparse_energy_profiler,
    )
    sparse_model.load_model("results/small/models/Sparse_Take47_32x32_T32.pth")


    # Main execution
    print("\n")
    print("---------------------------------- EVERYTHING LOADED SUCCESSFULLY ----------------------------------")
    print("\n")
    print("starting evaluation")

# Optional: Histogram of LZC values
# LZCValues = []
# for (events_dense, label_dense) in cached_test_dense:
#     lz_value = compute_lzc_from_events(events_dense)
#     LZCValues.append(lz_value)
# plt.figure()
# plt.hist(LZCValues, bins=30)
# plt.xlabel("LZC Value")
# plt.ylabel("Frequency")
# plt.title("Histogram of LZC Values")
# plt.show()

    results = evaluate_models_on_dataset(
        test_loader,
        sparse_model,
        dense_model,
        router_energy_profiler,
    )
    optimal_threshold = threshold_sweep_and_roc(results)

    route_and_evaluate(
        test_loader,
        sparse_model,
        dense_model,
        optimal_threshold,
        results,
        energy_metrics_path=args.energy_metrics_path,
    )

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
