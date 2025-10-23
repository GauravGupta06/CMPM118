import numpy as np
import torch, torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import snntorch as snn
import csv
import tonic
from lempel_ziv_complexity import lempel_ziv_complexity

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

grad = snn.surrogate.fast_sigmoid(slope=25)
beta = 0.5

# -- LOAD LARGE SNN --
w,h=64,64
n_frames=100
test_input = torch.zeros((1, 2, w, h))  # 2 polarity channels
x = nn.Conv2d(2, 12, 5)(test_input)
x = nn.MaxPool2d(2)(x)
x = nn.Conv2d(12, 32, 5)(x)
x = nn.MaxPool2d(2)(x)
print("Output shape before flatten:", x.shape)
print("Flattened size:", x.numel())
flattenedSize = x.numel()

net_l = nn.Sequential(
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

model_path = "../results/large/models/Large_Take4.pth"
net_l.load_state_dict(torch.load(model_path, map_location=device))
print("Large model loaded successfully.")

# -- LOAD SMALL SNN --
w,h=32,32
n_frames=5
test_input = torch.zeros((1, 2, w, h))  # 2 polarity channels
x = nn.Conv2d(2, 8, 3)(test_input)
x = nn.MaxPool2d(2)(x)
print("Output shape before flatten:", x.shape)
print("Flattened size:", x.numel())
flattenedSize = x.numel()

net_s = nn.Sequential(
    nn.Conv2d(2, 8, 3), # in_channels, out_channels, kernel_size
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(flattenedSize, 11),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
).to(device)

model_path = "../results/small/models/Small_Take2_32x32_T5.pth"
net_s.load_state_dict(torch.load(model_path, map_location=device))
print("Small model loaded successfully.")



cache_root = f"../data/dvsgesture/{w}x{h}_T{n_frames}"

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

print("data_snn.csv has been created")