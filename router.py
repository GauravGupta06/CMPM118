# make sure to have do "pip install PyQT6" otherwise the plt graph for the ROC curve might not show up. 
import numpy as np
import tonic
import torch
from lempel_ziv_complexity import lempel_ziv_complexity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# user made imports
from SNN_model import SNNModel
from LoadDataset import load_dataset

# Setup device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Model hyperparameters
w_large = 32
h_large = 32
n_frames_large = 32

w_small = 32
h_small = 32
n_frames_small = 32





# Load in the preprocessed dataset
# dataset_root = f"data/dvsgesture/{w_large}x{h_large}_T{n_frames_large}"
# dataset = tonic.DiskCachedDataset(None, cache_path=f"{dataset_root}/test")

cached_train, dataset, num_classes = load_dataset(
    dataset_name="DVSGesture",  # or "ASLDVS"
    dataset_path='/home/gauravgupta/CMPM118/data',
    w=32,
    h=32,
    n_frames=32
)



# Create and load dense model
dense_model = SNNModel(
    w=w_large,
    h=h_large,
    n_frames=n_frames_large,
    beta=0.8,
    spike_lam= 0,
    slope=25,
    model_type="dense",
    device=device
)
dense_model.load_model("results/large/models/Non_Sparse_Take6_32x32_T32.pth")

# Create and load sparse model
sparse_model = SNNModel(
    w=w_small,
    h=h_small,
    n_frames=n_frames_small,
    beta=0.4,
    spike_lam= 1e-7,
    slope=25,
    model_type="sparse",
    device=device
)
sparse_model.load_model("results/small/models/Sparse_Take47_32x32_T32.pth")


def compute_lzc_from_events(events):
    spike_seq = (events).astype(int).flatten()
    spike_seq_string = ''.join(map(str, spike_seq.tolist()))
    lz_score = lempel_ziv_complexity(spike_seq_string)
    return lz_score


def evaluate_models_on_dataset(dataset, sparse_model, dense_model, bin_size=0.005):
    results = []
    for (events, label) in dataset:
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
    
    return optimal_threshold


def route_and_evaluate(dataset, sparse_model, dense_model, optimal_threshold, results, lz_values=None):
    """Route samples to appropriate model and evaluate accuracy."""
    print("\nRouting and evaluating with threshold:", optimal_threshold, "\n")
    correct_sparse = 0
    correct_dense = 0
    route_counts = {'sparse': 0, 'dense': 0}

    if lz_values is None:
        lz_values = [compute_lzc_from_events(e) for e, _ in dataset]

    for i, (events, label) in enumerate(dataset):
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

results = evaluate_models_on_dataset(dataset, sparse_model, dense_model)
optimal_threshold = threshold_sweep_and_roc(results)

lz_values = [compute_lzc_from_events(e) for e, _ in dataset]
route_and_evaluate(dataset, sparse_model, dense_model, optimal_threshold, results, lz_values)