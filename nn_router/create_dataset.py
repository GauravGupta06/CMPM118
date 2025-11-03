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

width = 32
height = 32
n_frames = 32


cached_train, cached_test, num_classes = load_dataset(
    dataset_name="DVSGesture",  # or "ASLDVS"
    dataset_path='/Users/q-bh/repos/CMPM118/data',
    w=width,
    h=height,
    n_frames=n_frames
)


# -- LOAD DENSE SNN --
dense_model = SNNModel(
    w=width,
    h=height,
    n_frames=n_frames,
    beta= 0.6,
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
    w=width,
    h=height,
    n_frames=n_frames,
    beta= 0.6,
    spike_lam= 0,
    slope= 25,
    model_type="dense",
    device=device
)

model_path = "../results/small/models/Sparse_Take47_32x32_T32.pth"
sparse_model.load_model(model_path)

print("Sparse model loaded successfully.")



cache_root = f"../data/dvsgesture/{width}x{height}_T{n_frames}"

# -- WRITE DATA INTO data_train.csv --
cached_train = tonic.DiskCachedDataset(None, cache_path=f"{cache_root}/train")
"""
print("Creating data_train.csv...")

with open("data_train.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["lzc", "entropy", "std", "spike_count"])

with open("data_train.csv", "a", newline="") as file:
    writer = csv.writer(file)

    for frames, label in cached_train:
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

        writer.writerow([lzc, entropy, std, spike_count])

print("data_train.csv has been created")
"""



# -- WRITE DATA INTO data_test.csv (it is the same process) --
cached_test = tonic.DiskCachedDataset(None, cache_path=f"{cache_root}/test")
"""
print("Creating data_test.csv...")
with open("data_test.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["lzc", "entropy", "std", "spike_count"])
with open("data_test.csv", "a", newline="") as file:
    writer = csv.writer(file)
    for frames, label in cached_test:
        flat_seq = (frames > 0).astype(int).flatten()
        lzc = lempel_ziv_complexity(''.join(map(str, flat_seq.tolist())))
        p = np.mean(flat_seq)
        if p in [0, 1]:
            entropy = 0
        else:
            entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        std = np.std(flat_seq)
        spike_count = np.sum(flat_seq)
        writer.writerow([lzc, entropy, std, spike_count])
print("data_test.csv has been created")
"""




def forward_pass(net, data):
    spk_rec = []
    snn.utils.reset(net)
    with torch.no_grad():
        for t in range(data.size(0)):          # data: [T, 2, H, W]
            x = data[t].unsqueeze(0).to(device) # -> [1, 2, H, W]
            spk_out, _ = net(x)
            spk_rec.append(spk_out)             # [1, 11]
    return torch.stack(spk_rec)  


def predict_sample(net, frames):
    frames = torch.tensor(frames, dtype=torch.float)  # [T, 2, H, W]
    spk_rec = forward_pass(net, frames)
    counts = spk_rec.sum(0)            # [1, 11]
    return counts.argmax(1).item()

# -- WRITE DATA INTO data_nn.csv --
print("Creating data_snn.csv...")
"""
with open("data_snn.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["use_small", "use_large"])

with open("data_train.csv", "a", newline="") as file:
    writer = csv.writer(file)

    for frames, label in cached_train:
        pred_l = predict_sample(net_l, frames)
        pred_s = predict_sample(net_s, frames)

        writer.writerow([None, None])
        break
"""
print("data_snn.csv has been created")