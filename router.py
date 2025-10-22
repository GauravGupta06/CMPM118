import numpy as np
import tonic
import torch
import snntorch as snn
import torch.nn as nn
from lempel_ziv_complexity import lempel_ziv_complexity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#larger model net
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device) 

w = 64
h = 64
n_frames = 100

# Load in the large preprocessed datast
cache_root_dense = f"data/dvsgesture/{w}x{h}_T{n_frames}"
cached_test_dense= tonic.DiskCachedDataset(None, cache_path=f"{cache_root_dense}/test")


# this is to find the output layer size (the total number of connections the last layer of 11 neurons will have). 
# This value (flattenedSize), will be used when we construct the archecture of the CSNN. 
test_input = torch.zeros((1, 2, w, h))  # 2 polarity channels
x = nn.Conv2d(2, 12, 5)(test_input)
x = nn.MaxPool2d(2)(x)
x = nn.Conv2d(12, 32, 5)(x)
x = nn.MaxPool2d(2)(x)
print("Output shape before flatten:", x.shape)
print("Flattened size:", x.numel())
flattenedSize = x.numel()

grad = snn.surrogate.fast_sigmoid(slope=25)
beta = 0.5 #this is the decay rate of the pontential inside each neuron. The lower this value, the fewer the spikes 

dense_model = nn.Sequential(
    nn.Conv2d(2, 12, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
    nn.Conv2d(12, 32, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(flattenedSize, 11),   # make sure 800 matches flattenedSize
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
).to(device)

model_path = "results/large/models/Large_Take4.pth"
dense_model.load_state_dict(torch.load(model_path, map_location=device))
dense_model.eval()
print("Model loaded successfully.")

#small model net
w = 32
h = 32
n_frames = 5


# Load in the small preprocessed datast
cache_root_sparse = f"data/dvsgesture/{w}x{h}_T{n_frames}"
cached_test_sparse = tonic.DiskCachedDataset(None, cache_path=f"{cache_root_sparse}/test")


# this is to find the output layer size (the total number of connections the last layer of 11 neurons will have). 
# This value (flattenedSize), will be used when we construct the archecture of the CSNN. 
test_input = torch.zeros((1, 2, w, h))  # 2 polarity channels
x = nn.Conv2d(2, 8, 3)(test_input)
x = nn.MaxPool2d(2)(x)
print("Output shape before flatten:", x.shape)
print("Flattened size:", x.numel())
flattenedSize = x.numel()

grad = snn.surrogate.fast_sigmoid(slope=25)
beta = 0.5

sparse_model = nn.Sequential(
    nn.Conv2d(2, 8, 3), # in_channels, out_channels, kernel_size
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(flattenedSize, 11),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
).to(device)

model_path = "results/small/models/Small_Take2_32x32_T5.pth"
sparse_model.load_state_dict(torch.load(model_path, map_location=device))
sparse_model.eval()
print("Model loaded successfully.")



def forward_pass(net, data):
    spk_rec = []
    snn.utils.reset(net)
    with torch.no_grad():
        for t in range(data.size(0)):          # data: [T, 2, H, W]
            x = data[t].unsqueeze(0).to(device) # -> [1, 2, H, W]
            spk_out, _ = net(x)
            spk_rec.append(spk_out)             # [1, 11]
    return torch.stack(spk_rec)  


def predict_sample(frames, net):
    frames = torch.tensor(frames, dtype=torch.float)  # [T, 2, H, W]
    spk_rec = forward_pass(net, frames)
    counts = spk_rec.sum(0)            # [1, 11]
    return counts.argmax(1).item()


def compute_lzc_from_events(events):
    spike_seq = (events['p'] > 0).astype(int).flatten()
    spike_seq_string = ''.join(map(str, spike_seq.tolist()))
    lz_score = lempel_ziv_complexity(spike_seq_string)
    return lz_score

# 2. Evaluate both models on the dataset, storing all needed info.
def evaluate_models_on_dataset(dataset_sparse, dataset_dense, sparse_model, dense_model, bin_size=0.005):
    results = []
    for (events_sparse, label_sparse),(events_dense, label_dense) in zip(dataset_sparse, dataset_dense):
        lz_value = lempel_ziv_complexity(events_dense) # we have to figure out which events to calculate the LZC score
        sparse_pred = sparse_model.predict_sample(events_sparse)
        dense_pred = dense_model.predict_sample(events_dense)
        # Choose which model did better for this input
        # Here you decide which model is actually more accurate!
        # Example: assume ground truth label; set as complex IF dense_pred matches label and sparse_pred does NOT
        # Adjust logic as best fits your data and what you mean by "complex"
        # expected to have same label
        if dense_pred == label_dense and sparse_pred != label_sparse:
            true_complex = 1
        else:
            true_complex = 0
        results.append({
            'label': label_dense,
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

print("\n")
print ("---------------------------------- EVERYTHING LOADED SUCCESSFULLY ----------------------------------")
print("\n")



evaluate_models_on_dataset(cached_test_sparse, cached_test_dense, sparse_model, dense_model)

