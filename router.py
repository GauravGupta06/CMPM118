# make sure to have do "pip install PyQT6" otherwise the plt graph for the ROC curve might not show up. 

import numpy as np
import tonic
import torch
import snntorch as snn
from snntorch import utils
import torch.nn as nn
from lempel_ziv_complexity import lempel_ziv_complexity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



#larger model net 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device) 

w_large = 64
h_large = 64
n_frames_large = 100

# Load in the large preprocessed datast
cache_root_dense = f"data/dvsgesture/{w_large}x{h_large}_T{n_frames_large}"
cached_test_dense= tonic.DiskCachedDataset(None, cache_path=f"{cache_root_dense}/test")


# this is to find the output layer size (the total number of connections the last layer of 11 neurons will have). 
# This value (flattenedSize), will be used when we construct the archecture of the CSNN. 
test_input = torch.zeros((1, 2, w_large, h_large))  # 2 polarity channels
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
w_small = 32
h_small = 32
n_frames_small = 10


# Load in the small preprocessed datast
cache_root_sparse = f"data/dvsgesture/{w_small}x{h_small}_T{n_frames_small}"
cached_test_sparse = tonic.DiskCachedDataset(None, cache_path=f"{cache_root_sparse}/test")


# this is to find the output layer size (the total number of connections the last layer of 11 neurons will have). 
# This value (flattenedSize), will be used when we construct the archecture of the CSNN. 
test_input = torch.zeros((1, 2, w_small, h_small))  # 2 polarity channels
x = nn.Conv2d(2, 8, 3)(test_input)
x = nn.MaxPool2d(2)(x)
print("Output shape before flatten:", x.shape)
print("Flattened size:", x.numel())
flattenedSize = x.numel()

grad = snn.surrogate.fast_sigmoid(slope=25)
beta = 0.15

sparse_model = nn.Sequential(
    nn.Conv2d(2, 8, 3), # in_channels, out_channels, kernel_size
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(flattenedSize, 11),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
).to(device)

model_path = "results/small/models/Small_Take6_32x32_T10.pth"
sparse_model.load_state_dict(torch.load(model_path, map_location=device))
sparse_model.eval()
print("Model loaded successfully.")





















def forward_pass(net, data):
    utils.reset(net)
    spk_rec = []
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
    # print(events.shape)
    spike_seq = (events).astype(int).flatten()
    spike_seq_string = ''.join(map(str, spike_seq.tolist()))
    lz_score = lempel_ziv_complexity(spike_seq_string)
    return lz_score

# 2. Evaluate both models on the dataset, storing all needed info.
def evaluate_models_on_dataset(dataset_sparse, dataset_dense, sparse_model, dense_model, bin_size=0.005):
    results = []
    for (events_sparse, label_sparse),(events_dense, label_dense) in zip(dataset_sparse, dataset_dense):
        lz_value = compute_lzc_from_events(events_dense) # we have to figure out which events to calculate the LZC score
        sparse_pred = predict_sample(events_sparse, sparse_model)
        dense_pred = predict_sample(events_dense, dense_model)
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




    graph_save_path = f"results/large/graphs/Small:{w_small}x{h_small}_T{n_frames_small}_Large:{w_large}x{h_large}_T{n_frames_large}.png"
    plt.savefig(graph_save_path)


    plt.show()
    return optimal_threshold

def route_and_evaluate(dataset_sparse, dataset_dense, sparse_model, dense_model, optimal_threshold, lz_values = None):
    print("\nRouting and evaluating with threshold:", optimal_threshold)
    correct_sparse = 0
    correct_dense = 0
    route_counts = {'sparse': 0, 'dense': 0}

    if lz_values is None:
        lz_values = [compute_lzc_from_events(e) for e, _ in dataset_dense]

    for i,(events_sparse, label_sparse), (events_dense, label_dense) in enumerate(zip(dataset_sparse, dataset_dense)):
        lz_value = lz_values[i]

        if lz_value < optimal_threshold:
            route_counts['sparse'] += 1
            pred = predict_sample(events_sparse, sparse_model)
            if pred == label_sparse:
                correct_sparse += 1
        else:
            route_counts['dense'] += 1
            pred = predict_sample(events_dense, dense_model)
            if pred == label_dense:
                correct_dense += 1
    
    accuracy_dense = correct_dense/route_counts['dense'] if route_counts['dense'] > 0 else 0
    accuracy_sparse = correct_sparse/route_counts['sparse'] if route_counts['sparse'] > 0 else 0

    total_correct = correct_dense + correct_sparse
    total_samples = route_counts['sparse'] + route_counts['dense']
    total_accuracy = total_correct/total_samples

    print(f"\n Dense Model accuracy after routing: {accuracy_dense*100: .2f}%")
    print(f"\n Sparse Model accuracy after routing: {accuracy_sparse*100: .2f}%")
    print(f"\n Overall Accuracy after routing: {total_accuracy*100: .2f}%")
    print(f"Routed {route_counts['sparse']} samples to SMALL model")
    print(f"Routed {route_counts['dense']} samples to LARGE model")

    return total_accuracy, accuracy_dense, accuracy_sparse, route_counts



print("\n")
print ("---------------------------------- EVERYTHING LOADED SUCCESSFULLY ----------------------------------")
print("\n") 
print("starting evaluation")


results = evaluate_models_on_dataset(cached_test_sparse, cached_test_dense, sparse_model, dense_model)
optimal_threshold = threshold_sweep_and_roc(results)

lz_values = [compute_lzc_from_events(e) for e, _ in cached_test_dense]
route_and_evaluate(cached_test_sparse, cached_test_dense, sparse_model, dense_model, optimal_threshold, lz_values)





