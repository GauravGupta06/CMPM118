import numpy as np
from lempel_ziv_complexity import lempel_ziv_complexity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. Preprocess DVSGesture Events for LZC
def preprocess_dvs_events(events, bin_size, frame_shape=(128, 128)):
    min_t, max_t = events['t'][0], events['t'][-1]
    num_bins = int((max_t - min_t) / bin_size) + 1
    spike_train_on = np.zeros((num_bins, *frame_shape), dtype=np.uint8)
    spike_train_off = np.zeros((num_bins, *frame_shape), dtype=np.uint8)

    for t, x, y, p in zip(events['t'], events['x'], events['y'], events['p']):
        bin_idx = int((t - min_t) / bin_size)
        if p == 1:
            spike_train_on[bin_idx, y, x] = 1
        else:
            spike_train_off[bin_idx, y, x] = 1

    seq_on = spike_train_on.flatten()
    seq_off = spike_train_off.flatten()
    lz_input_seq = np.concatenate([seq_on, seq_off])
    lz_input_str = ''.join(map(str, lz_input_seq))
    return lz_input_str

# 2. Evaluate both models on the dataset, storing all needed info.
def evaluate_models_on_dataset(dataset, sparse_model, dense_model, bin_size=0.005):
    results = []
    for (events, label) in dataset:
        lz_input_str = preprocess_dvs_events(events, bin_size)
        lz_value = lempel_ziv_complexity(lz_input_str)
        sparse_pred = sparse_model.predict(events)
        dense_pred = dense_model.predict(events)
        # Choose which model did better for this input
        # Here you decide which model is actually more accurate!
        # Example: assume ground truth label; set as complex IF dense_pred matches label and sparse_pred does NOT
        # Adjust logic as best fits your data and what you mean by "complex"
        if dense_pred == label and sparse_pred != label:
            true_complex = 1
        else:
            true_complex = 0
        results.append({
            'label': label,
            'lz_value': lz_value,
            'sparse_pred': sparse_pred,
            'dense_pred': dense_pred,
            'true_complex': true_complex
        })
    return results

# 3. Threshold sweep, ROC-AUC curve, and optimal LZC threshold
def threshold_sweep_and_roc(results):
    # Ground truth: 1 if dense model was needed, 0 if sparse sufficed
    y_true = np.array([r['true_complex'] for r in results])
    lz_scores = np.array([r['lz_value'] for r in results])
    fpr, tpr, thresholds = roc_curve(y_true, lz_scores)
    roc_auc = auc(fpr, tpr)
    gmean = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmean)
    optimal_threshold = thresholds[idx]
    print(f"Optimal LZC threshold: {optimal_threshold:.4f} (G-mean={gmean[idx]:.4f}) (AUC={roc_auc:.4f})")
    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.scatter(fpr[idx], tpr[idx], color='red', label=f'Optimal G-mean\n(Threshold={optimal_threshold:.4f})')
    plt.plot([0,1],[0,1],'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for LZC-based Routing')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return optimal_threshold

# ---- USAGE EXAMPLE ----
# dataset = ...   # list of (events, label) samples from DVSGesture
# sparse_model = ... # your trained model
# dense_model = ... # your trained model
# results = evaluate_models_on_dataset(dataset, sparse_model, dense_model)
# optimal_threshold = threshold_sweep_and_roc(results)