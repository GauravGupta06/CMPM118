# make sure to have do "pip install PyQT6" otherwise the plt graph for the ROC curve might not show up. 
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

# user made imports
from SNN_model_inheritance import DVSGestureSNN
from LoadDataset import load_dataset

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

def evaluate_models_on_dataset(dataLoader, sparse_model, dense_model, bin_size=0.005):
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
        sparse_pred, spike_count_sparse = sparse_model.predict_sample(events)
        dense_pred, spike_count_dense = dense_model.predict_sample(events)
        
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


def threshold_sweep_and_roc(results, plotting_only=False):
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

    graph_save_path = f"results/ROC_curves:{w_small}x{h_small}_T{n_frames_small}_Large:{w_large}x{h_large}_T{n_frames_large}.png"
    plt.savefig(graph_save_path)
    plt.show()
    
    return optimal_threshold, roc_auc, average_spike_dense, average_spike_sparse



def route_and_evaluate(dataLoader, sparse_model, dense_model, optimal_threshold, results):
    """Route samples to appropriate model and evaluate accuracy."""
    print("\nRouting and evaluating with threshold:", optimal_threshold, "\n")
    correct_sparse = 0
    correct_dense = 0
    route_counts = {'sparse': 0, 'dense': 0}

    lz_values = [r['lz_value'] for r in results]

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

        if lz_value < optimal_threshold:
            route_counts['sparse'] += 1
            pred, _ = sparse_model.predict_sample(events)
            if pred == label:
                correct_sparse += 1
        else:
            route_counts['dense'] += 1
            pred, _ = dense_model.predict_sample(events)
            if pred == label:
                correct_dense += 1
    
    accuracy_dense_routed = correct_dense / route_counts['dense'] if route_counts['dense'] > 0 else 0
    accuracy_sparse_routed = correct_sparse / route_counts['sparse'] if route_counts['sparse'] > 0 else 0

    total_correct = correct_dense + correct_sparse
    total_samples = route_counts['sparse'] + route_counts['dense']
    total_accuracy = total_correct / total_samples






    # getting the accuracy on each model based for the entire dataset. 
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
    print(f"Samples routed to sparse model: {route_counts['dense']}\n")




    print(f"Overall Accuracy after routing: {total_accuracy*100: .2f}%")
    print(f"Total Samples: {total_samples}")

    return total_accuracy, accuracy_dense_routed, accuracy_sparse_routed, route_counts



def lzc_vs_accuracy_plot(results):
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

    optimal_threshold, _, _, _ = threshold_sweep_and_roc(results, plotting_only=True)
    optimal_threshold = float(optimal_threshold)
    best_acc_idx = int(np.searchsorted(threshold_range, optimal_threshold))
    best_acc_idx = np.clip(best_acc_idx, 0, len(threshold_range) - 1)
    best_accuracy = accuracy_at_threshold[best_acc_idx]
    
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
    
    os.makedirs('results', exist_ok=True)
    graph_save_path = f"results/LZC_vs_Accuracy:{w_small}x{h_small}_T{n_frames_small}_Large:{w_large}x{h_large}_T{n_frames_large}.png"
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
    average_spike_sparse
):
    os.makedirs("results/run_logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/run_logs/run_{w_small}x{h_small}_T{n_frames_small}_large_{w_large}x{h_large}_T{n_frames_large}_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": timestamp,
            "small_model": {
                "w": w_small,
                "h": h_small,
                "T": n_frames_small
            },
            "large_model": {
                "w": w_large,
                "h": h_large,
                "T": n_frames_large
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


def main():
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



    # Create and load dense model
    dense_model = DVSGestureSNN(
        w=w_large,
        h=h_large,
        n_frames=n_frames_large,
        beta=0.8,
        spike_lam=0,
        slope=25,
        model_type="dense",
        device=device
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
        device=device
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

    results = evaluate_models_on_dataset(test_loader, sparse_model, dense_model)

    lzc_vs_accuracy_plot(results)

    optimal_threshold, roc_auc, avg_dense_spikes, avg_sparse_spikes = threshold_sweep_and_roc(results)

    total_accuracy, accuracy_dense_routed, accuracy_sparse_routed, route_counts = route_and_evaluate(
        test_loader, sparse_model, dense_model, optimal_threshold, results
    )

    save_run_to_json(
    results=results,
    optimal_threshold=optimal_threshold,
    roc_auc=roc_auc,
    route_counts=route_counts,
    accuracy_dense_routed=accuracy_dense_routed,
    accuracy_sparse_routed=accuracy_sparse_routed,
    total_accuracy=total_accuracy,
    average_spike_dense=avg_dense_spikes,
    average_spike_sparse=avg_sparse_spikes
    )

    print_latex_table(
    total_accuracy,
    accuracy_dense_routed,
    accuracy_sparse_routed,
    avg_dense_spikes,
    avg_sparse_spikes,
    route_counts,
    roc_auc,
    optimal_threshold
    )

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
