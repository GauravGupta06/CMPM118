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
from scipy.stats import entropy as shannon_entropy

import sys
sys.path.insert(0, "../.")
from SNN_model_inheritance import DVSGestureSNN
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
dense_model = DVSGestureSNN(
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
sparse_model = DVSGestureSNN(
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

def extract_time_features(frames, n_windows=8, burst_thresh=None, max_fft_bins=8):
    T = frames.shape[0]
    spike_counts_per_frame = (frames > 0).astype(int).sum(axis=(1,2,3)).astype(np.float32)

    # --- 1. Total spike count ---
    total_spikes = float(spike_counts_per_frame.sum()) + 1e-9    

    # --- 2-3. Mean & Standard Deviation ---
    spikes_per_frame_mean = float(spike_counts_per_frame.mean())
    spikes_per_frame_std = float(spike_counts_per_frame.std())

    # --- 4. LZC ---
    flat_seq = (frames > 0).astype(int).flatten()
    lzc = lempel_ziv_complexity(''.join(map(str, flat_seq.tolist())))

    # --- 5. Center of Mass (weighted average of spike times; measures when the most activity occurs) ---
    times = np.arange(T)
    com = float((times * spike_counts_per_frame).sum() / total_spikes) / max(1, T-1)  # normalized 0..1

    # --- 6-7. Inter-spike intervals (ISI) aggregated across the whole frame sequence ---
    # expand spike times by count so each spike gets a timestamp (frame index)
    expanded_times = np.repeat(times, spike_counts_per_frame.astype(int))
    isis = np.diff(expanded_times).astype(float)
    isi_mean = float(np.mean(isis))

    # discrete distribution for ISI entropy
    hist, _ = np.histogram(isis, bins=min(20, len(isis)), density=True)
    isi_entropy = float(shannon_entropy(hist + 1e-12))

    # --- 8-10. Burst features ---
    # define burst frame if spike_counts_per_frame >= burst_thresh (or >= 2 or >= mean)
    if burst_thresh is None:
        burst_thresh = max(2, int(np.ceil(np.maximum(1.0, spikes_per_frame_mean))))  # fallback heuristic
    burst_mask = spike_counts_per_frame >= burst_thresh
    # find runs of consecutive True in burst_mask
    bursts = []
    if burst_mask.any():
        # find run-length encoding
        diff = np.diff(np.concatenate(([0], burst_mask.view(np.int8), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        bursts = [ends[i] - starts[i] for i in range(len(starts))]
    n_bursts = float(len(bursts))
    avg_burst_len = float(np.mean(bursts)) if bursts else 0.0
    spikes_in_bursts = float(sum(spike_counts_per_frame[burst_mask])) if bursts else 0.0
    frac_spikes_in_bursts = spikes_in_bursts / total_spikes

    # --- Windowed stats (preserve temporal drift) ---
    # split into n_windows (last window possibly smaller)
    w_feats = []
    edges = np.linspace(0, T, n_windows+1, dtype=int)
    window_counts = []
    for i in range(n_windows):
        a, b = edges[i], edges[i+1]
        if b <= a:
            seg = np.array([])
        else:
            seg = spike_counts_per_frame[a:b]
        s = float(seg.sum()) if seg.size else 0.0
        window_counts.append(s)
        # entropy within window (binary per-frame presence/absence)
        if seg.size:
            p = (seg > 0).astype(int)
            # entropy of presence vs absence
            p_mean = p.mean()
            if p_mean in (0.0, 1.0):
                win_ent = 0.0
            else:
                win_ent = float(-(p_mean*np.log2(p_mean) + (1-p_mean)*np.log2(1-p_mean)))
        else:
            win_ent = 0.0
        w_feats.extend([s, win_ent])   # keep two numbers per window

    # --- Autocorrelation features (lags 1..L) ---
    L = min(10, max(1, T//4))
    acorrs = []
    if T > 2:
        scf = spike_counts_per_frame - spike_counts_per_frame.mean()
        denom = (scf * scf).sum()
        for lag in range(1, L+1):
            if denom == 0:
                ac = 0.0
            else:
                ac = float((scf[:-lag] * scf[lag:]).sum() / denom)
            acorrs.append(ac)
    else:
        acorrs = [0.0] * L

    # --- FFT / spectral features ---
    # rfft on spike_counts_per_frame (small smoothing first)
    if T >= 4:
        arr = spike_counts_per_frame - spike_counts_per_frame.mean()
        spec = np.abs(np.fft.rfft(arr, n=max(4, 2**int(np.ceil(np.log2(T))))))  # power spectrum
        # collapse to a small number of frequency-band features
        spec = spec[1: max_fft_bins+1] if spec.size > 1 else np.zeros(max_fft_bins)
        # if too short, pad
        if spec.size < max_fft_bins:
            spad = np.zeros(max_fft_bins)
            spad[:spec.size] = spec
            spec = spad
        spec = spec / (spec.sum() + 1e-12)
        spec_feats = spec.tolist()
    else:
        spec_feats = [0.0] * max_fft_bins

    # --- final vector assembly ---
    feats = [
        total_spikes,
        spikes_per_frame_mean,
        spikes_per_frame_std,
        lzc,
        com,
        isi_mean, isi_entropy,
        n_bursts, avg_burst_len, frac_spikes_in_bursts,
    ]
    # append window features
    feats.extend(w_feats)        # 2*n_windows
    # autocorrs
    feats.extend(acorrs)        # L values
    # spectral features
    feats.extend(spec_feats)    # max_fft_bins values

    return np.array(feats, dtype=np.float32)

# -- WRITE DATA INTO data_train_v2.csv --

cached_train = tonic.DiskCachedDataset(None, cache_path=f"{cache_root}/train")

print("Creating data_train_v2.csv...")

with open("data_train_v2.csv", "w", newline="") as file:
    writer = csv.writer(file)

    test_vec = extract_time_features(cached_train[0][0])
    header = ["idx"] + [f"feat_{i}" for i in range(len(test_vec))]
    writer.writerow(header)

    for i, (frames, label) in enumerate(cached_train):
        flat_seq = (frames > 0).astype(int).flatten()
        
        writer.writerow([i] + list(extract_time_features(frames)))
        

print("data_train_v2.csv has been created")


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
    writer.writerow(["idx", "sparse_result", "dense_results", "use_sparse", "use_dense"])

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
    writer.writerow(["idx", "sparse_result", "dense_results", "use_sparse", "use_dense"])

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