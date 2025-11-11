import numpy as np
import torch, torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import csv
import tonic
from lempel_ziv_complexity import lempel_ziv_complexity

import sys
sys.path.insert(0, "../.")
from SNN_model import SNNModel
from LoadDataset import load_dataset

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Dataset parameters
width = 32
height = 32
n_frames = 32

# Model hyperparameters
w_large = 32
h_large = 32
n_frames_large = 32

w_small = 32
h_small = 32
n_frames_small = 32


cached_train, cached_test, num_classes = load_dataset(
    dataset_name="DVSGesture",  # or "ASLDVS"
    dataset_path='/Users/q-bh/repos/CMPM118/data',
    w=width,
    h=height,
    n_frames=n_frames
)


# -- LOAD DENSE SNN --
dense_model = SNNModel(
    w=w_large,
    h=h_large,
    n_frames=n_frames_large,
    beta= 0.8,
    spike_lam= 0,
    slope= 25,
    model_type="dense",
    device=device
)

model_path = "../results/large/models/Non_Sparse_Take91_32x32_T32_B0.6_SpkLam0_Epochs200.pth"
dense_model.load_model(model_path)

print("Dense model loaded successfully.")

# -- LOAD SPARSE SNN --
sparse_model = SNNModel(
    w=w_small,
    h=h_small,
    n_frames=n_frames_small,
    beta=0.4,
    spike_lam= 1e-7,
    slope= 25,
    model_type="sparse",
    device=device
)

model_path = "../results/small/models/Sparse_Take47_32x32_T32.pth"
sparse_model.load_model(model_path)

print("Sparse model loaded successfully.")



cache_root = f"../data/dvsgesture/{width}x{height}_T{n_frames}"

# -- WRITE DATA INTO data_train.csv --
"""
cached_train = tonic.DiskCachedDataset(None, cache_path=f"{cache_root}/train")

print("Creating data_train.csv...")

with open("data_train.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["idx", "lzc", "entropy", "std", "spike_count"])

with open("data_train.csv", "a", newline="") as file:
    writer = csv.writer(file)

    for i, (frames, label) in enumerate(cached_train):
        flat_seq = (frames > 0).astype(int).flatten()
        
        # LZC
        lzc = lempel_ziv_complexity(''.join(map(str, flat_seq.tolist())))

        # Entropy
        p = np.mean(flat_seq)
        if p in [0, 1]:
            entropy = 0
        else:
            entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        # Standard Deviation
        std = np.std(flat_seq)

        # Spike Count
        spike_count = np.sum(flat_seq)

        writer.writerow([i, lzc, entropy, std, spike_count])

print("data_train.csv has been created")
"""



# -- WRITE DATA INTO data_test.csv (it is the same process) --
"""
cached_test = tonic.DiskCachedDataset(None, cache_path=f"{cache_root}/test")

print("Creating data_test.csv...")

with open("data_test.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["idx", "lzc", "entropy", "std", "spike_count"])

with open("data_test.csv", "a", newline="") as file:
    writer = csv.writer(file)

    for i, (frames, label) in enumerate(cached_test):
        flat_seq = (frames > 0).astype(int).flatten()
        
        # LZC
        lzc = lempel_ziv_complexity(''.join(map(str, flat_seq.tolist())))

        # Entropy
        p = np.mean(flat_seq)
        if p in [0, 1]:
            entropy = 0
        else:
            entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        # Standard Deviation
        std = np.std(flat_seq)

        # Spike Count
        spike_count = np.sum(flat_seq)

        writer.writerow([i, lzc, entropy, std, spike_count])
        
print("data_test.csv has been created")
"""

# -----------------------------------------------------------------------------------------------------------------------------

# -- CREATING SNN CSVs --
"""
Truth Table:
+----------------+---------------+------------+-----------+
| correct_sparse | correct_dense | use_sparse | use_dense |
+----------------+---------------+------------+-----------+
|            1.0 |           1.0 | True       | False     |
|            1.0 |           0.0 | True       | False     | <- This will essentially never happen though
|            0.0 |           1.0 | False      | True      |
|            0.0 |           0.0 | False      | True      |
+----------------+---------------+------------+-----------+
"""

active_cores = 0


# -- WRITE DATA INTO data_train_snn.csv --
"""
train_loader = torch.utils.data.DataLoader(cached_train, batch_size=1, num_workers = active_cores, drop_last=True, 
                                           collate_fn=tonic.collation.PadTensors(batch_first=False))

print("Creating data_train_snn.csv...")

with open("data_train_snn.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["sparse_result", "dense_results", "use_sparse", "use_dense"])

with open("data_train_snn.csv", "a", newline="") as file:
    writer = csv.writer(file)

    correct_dense, correct_sparse = 0, 0
    for idx, (frames, label) in enumerate(train_loader):
        pred_dense, _ = dense_model.forward_pass(frames)
        pred_sparse, _ = sparse_model.forward_pass(frames)

        correct_dense = SF.accuracy_rate(pred_dense, label) * frames.shape[1]
        correct_sparse = SF.accuracy_rate(pred_sparse, label) * frames.shape[1]

        if correct_sparse:
            use_sparse = True
            use_dense = False
        else:
            use_sparse = False
            use_dense = True

        writer.writerow([idx, correct_sparse, correct_dense, use_sparse, use_dense])

print("data_train_snn.csv has been created")
"""

# -- WRITE DATA INTO data_test_snn.csv --
"""
test_loader = torch.utils.data.DataLoader(cached_test, batch_size=1, num_workers = active_cores, drop_last=True, 
                                          collate_fn=tonic.collation.PadTensors(batch_first=False))

print("Creating data_test_snn.csv...")

with open("data_test_snn.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["sparse_result", "dense_results", "use_sparse", "use_dense"])

with open("data_test_snn.csv", "a", newline="") as file:
    writer = csv.writer(file)

    correct_dense, correct_sparse = 0, 0
    for idx, (frames, label) in enumerate(test_loader):
        pred_dense, _ = dense_model.forward_pass(frames)
        pred_sparse, _ = sparse_model.forward_pass(frames)

        correct_dense = SF.accuracy_rate(pred_dense, label) * frames.shape[1]
        correct_sparse = SF.accuracy_rate(pred_sparse, label) * frames.shape[1]

        if correct_sparse:
            use_sparse = True
            use_dense = False
        else:
            use_sparse = False
            use_dense = True

        writer.writerow([idx, correct_sparse, correct_dense, use_sparse, use_dense])

print("data_test_snn.csv has been created")
"""