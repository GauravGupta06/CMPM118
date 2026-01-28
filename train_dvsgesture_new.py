"""
Full 1028x2=2048 input features, single file, no unnecessary abstractions.
"""

import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import tonic
from tonic import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from rockpool.nn.modules import LIFTorch, LinearTorch, ExpSynTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant

# =============================================================================
# Data Transform
# =============================================================================

class ToRaster:
    """Convert DVSGesture events to raster with full channels and polarity."""

    def __init__(self, width=32, height=32, num_channels=1024, num_polarities=2, sample_T=100, dt_seconds=10e-3):
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.num_polarities = num_polarities
        self.sample_T = sample_T
        self.total_features = self.width * self.height * num_polarities
        self.dt_seconds = dt_seconds

    def __call__(self, events):
        if len(events) == 0:
            return np.zeros((self.sample_T, self.total_features), dtype=np.float32)
        
        times = np.floor(events["t"].astype(int) / (self.dt_seconds * 1e6)).astype(int)

        x = events["x"].astype(int)
        y = events["y"].astype(int)
        p = events["p"].astype(int)
        
        #channels = events["x"].astype(int) * self.num_polarities + events["p"].astype(int) 
        channels = (y * self.width + x) * self.num_polarities + p

        # by multiplying with num_polarities, scale the values from 0-1024 to 0-2048 for the index.
        # this allows us to index into the raster array. 
        # we then add the polarity to get which of the two possible indexes for the raster we add the event to. 
        # Note that every event is a spike, regardless of polarity. The reason we have the + polarity is to 
        # allow us find the specific column for that specific channel (becuase each channel has 2 columns). 

        valid = (channels >= 0) & (channels < self.total_features) & (times >= 0)
        if not np.any(valid):
            return np.zeros((self.sample_T, self.total_features), dtype=np.float32)
        
        max_t = max(self.sample_T, times.max() + 1)
        raster = np.zeros((max_t, self.total_features), dtype=np.float32)

        np.add.at(raster, (times[valid], channels[valid]), 1.0)

        # Pad or truncate to sample_T
        if raster.shape[0] < self.sample_T:
            pad = np.zeros((self.sample_T - raster.shape[0], self.total_features), dtype=np.float32)
            raster = np.concatenate([raster, pad], axis=0)
        else:
            raster = raster[:self.sample_T, :]
        # the reason we are doing the code above is because the input data is a variable length.
        # some audio samples are longer than others, so we need to pad or truncate to a fixed length.
        # we are padding with 0s if the sample is shorter than the fixed length.
        # we are truncating if the sample is longer than the fixed length.
        
        # Its important to keep the data size the exact same even though it doesn't come in the same size. 
        # This is because the model is going to be trained on batches of data, and if the data size is different
        # for each batch, the model will not be able to learn effectively. Also, the model will not be able to 
        # make predictions on new data if the data size is different. 


        # Binarize
        raster = (raster > 0).astype(np.float32) * 0.5
        return torch.from_numpy(raster)



# =============================================================================
# Main
# =============================================================================

##################

# create a pad-collater once
pad_collate = tonic.collation.PadTensors(batch_first=True)

def collate_flatten(batch):
    """
    batch: list of tuples (frames, label)
    - frames after pad_collate will be a Tensor (B, T, C, H, W)
    - this function returns (frames_flat, labels) where
    frames_flat has shape (B, T, C*H*W) which Rockpool/Linear expects.
    """
    frames_padded, labels = pad_collate(batch)   # frames_padded: (B, T, C, H, W)
    # ensure it's a torch.Tensor
    B, T, C, H, W = frames_padded.shape
    #print(B,T,C,H,W, frames_padded.shape)
    frames_flat = frames_padded.view(B, T, C * H * W)  # (B, T, features)
    return frames_flat, labels

##################

def main():
    parser = argparse.ArgumentParser(description="DVSGesture Training")
    parser.add_argument('--w', type=int, default=32)
    parser.add_argument('--h', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)    # Default: 1e-5
    parser.add_argument('--hidden1', type=int, default=256)  # Default: 256
    parser.add_argument('--hidden2', type=int, default=128)  # Default: 128
    parser.add_argument('--sample_T', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--output_path', type=str, default='./results/large/models')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Constants
    NUM_CHANNELS = args.w * args.h
    NUM_POLARITIES = 2
    INPUT_SIZE = NUM_CHANNELS * NUM_POLARITIES  # 2048
    NUM_CLASSES = 11
    NET_DT = 10e-3

    # ==========================================================================
    # Load Data
    # ==========================================================================
    print("\nLoading DVSGesture dataset...")

    transform = transforms.Compose([
        transforms.Downsample(time_factor=1e-6 / NET_DT, spatial_factor=(args.w / 128, args.h / 128)),
        ToRaster(width=args.w, height=args.h, num_polarities=2, sample_T=args.sample_T, dt_seconds=NET_DT),
    ])

    # The downsample multiplies each time step by 1e-6 / NET_DT. 
    # This converts the micro seconds data into milliseconds, which is the bin size we want. 
    # So its important to note that we don't have like 100 bins instantly. 
    # Its AROUND 100 bins and then we crop or add bins later on in the raster class
    # The ToRaster class then converts the data into a raster with the specified sample_T 
    # The torch.tensor converts the data into a torch tensor

    cache_root = f"{args.data_path}/DVSGesture/{args.w}x{args.h}_T{args.sample_T}/"
    cache_exists = os.path.exists(f"{cache_root}/train") and os.path.exists(f"{cache_root}/test")

    if not cache_exists:
        train_dataset = tonic.datasets.DVSGesture(save_to=args.data_path, transform=transform, train=True)
        test_dataset = tonic.datasets.DVSGesture(save_to=args.data_path, transform=transform, train=False)
    else:
        train_dataset = None
        test_dataset = None

    cached_train = tonic.DiskCachedDataset(train_dataset, cache_path=f"{cache_root}/train")
    cached_test = tonic.DiskCachedDataset(test_dataset, cache_path=f"{cache_root}/test")

    train_loader = DataLoader(
        cached_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_flatten
    )

    test_loader = DataLoader(
        cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_flatten
    )

    print(f"Train: {len(cached_train)} samples, Test: {len(cached_test)} samples")

    # ==========================================================================
    # Build Network
    # ==========================================================================
    print(f"\nBuilding network: {INPUT_SIZE} → {args.hidden1} → {args.hidden2} → {NUM_CLASSES}") # 2048 -> 256 -> 128 -> 11

    tau_mem = Constant(0.1)
    tau_syn = Constant(0.1)
    threshold = Constant(1.0)
    bias = Constant(0.01)

    # TO TRY: Rework model architecture to have more layers but (preferably) same parameters, ex: 2048->1024->256->64->11
    """
    Notes:
    - Increasing hidden layer sizes made the model perform the same and overfit more
    - Decreasing hidden layer sizes (128 -> 32) made the model take longer to get to the usual accuracy, but with slightly less overfitting
    - 2048 -> 1024 -> 256 -> 64 -> 11 w/ 1e-6 LR is unsuccessful
    """
    net = Sequential(
        LinearTorch((INPUT_SIZE, args.hidden1), has_bias=True),
        LIFTorch(args.hidden1, tau_mem=tau_mem, tau_syn=tau_syn, threshold=threshold, bias=bias, dt=NET_DT, has_rec=False),
        LinearTorch((args.hidden1, args.hidden2), has_bias=True),
        LIFTorch(args.hidden2, tau_mem=tau_mem, tau_syn=tau_syn, threshold=threshold, bias=bias, dt=NET_DT, has_rec=False),
        LinearTorch((args.hidden2, NUM_CLASSES), has_bias=True),
        ExpSynTorch(NUM_CLASSES, dt=NET_DT, tau=Constant(5e-3))
    ).to(device)

    # Normalize parameters to start off around 0
    for p in net.parameters().astorch():
        if p.dim() > 1:
            p.data.normal_(0, 0.01)

    total_params = sum(p.numel() for p in net.parameters().astorch())
    print(f"Total parameters: {total_params:,}")

    # ==========================================================================
    # Training Setup
    # ==========================================================================
    
    optimizer = torch.optim.Adam(net.parameters().astorch(), lr=args.lr, weight_decay=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    best_test_acc = 0.0

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print(f"\nTraining for {args.epochs} epochs...")

    acc_hist = []
    test_acc_hist = []
    best_hist = []
    for epoch in range(args.epochs):
        # Train
        net.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for data, labels in pbar:
            data, labels = data.to(device).float(), labels.to(device)

            optimizer.zero_grad()
            output, _, _ = net(data)
            #print("output min/max:", output.min().item(), output.max().item())
            #print("total spikes:", output.sum().item())

            # Cumsum integration (from Rockpool tutorial)
            #logits = torch.cumsum(output, dim=1)[:, -1, :]
            logits = output[:, -1, :]

            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters().astorch(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*train_correct/train_total:.1f}%")


        # Test
        net.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device).float(), labels.to(device)
                output, _, _ = net(data)
                #logits = torch.cumsum(output, dim=1)[:, -1, :]
                logits = output[:, -1, :]
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs(args.output_path, exist_ok=True)
            torch.save(net.state_dict(), f"{args.output_path}/Rockpool_Non_Sparse_Take103_DVSGesture_Input1024_T32_FC_Rockpool_Epochs100.pth")

        acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)
        best_hist.append(best_test_acc)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%, Best={best_test_acc:.1f}%")

    print(f"\nDone! Best test accuracy: {best_test_acc:.1f}%")


    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(acc_hist)
    axes[0].set_title("Train Set Accuracy")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0,100)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])

    axes[1].plot(test_acc_hist)
    axes[1].set_title("Test Set Accuracy")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0,100)
    axes[1].set_yticks([0, 20, 40, 60, 80, 100])

    axes[2].plot(best_hist)
    axes[2].set_title("Best")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0,100)
    axes[2].set_yticks([0, 20, 40, 60, 80, 100])

    plt.savefig("./results/large/graphs/Rockpool_Non_Sparse_Take103_DVSGesture_Input1024_T32_FC_Rockpool_Epochs100")
    plt.show()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
