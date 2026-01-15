"""
Simple SHD Training Script - Following Rockpool tutorial style.
Full 700x2=1400 input features, single file, no unnecessary abstractions.
"""

import argparse
import os
import torch
import torch.nn as nn
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
    """Convert SHD events to raster with full channels and polarity."""

    def __init__(self, num_channels=700, num_polarities=2, sample_T=100):
        self.num_channels = num_channels
        self.num_polarities = num_polarities
        self.sample_T = sample_T
        self.total_features = num_channels * num_polarities  # 1400

    def __call__(self, events):
        max_t = int(events["t"].max()) + 1
        raster = np.zeros((max_t, self.total_features), dtype=np.float32)

        times = events["t"].astype(int)
        channels = events["x"].astype(int) * self.num_polarities + events["p"].astype(int) 
        # by multiplying with num_polarities, scale the values from 0-700 to 0-1400 for the index.
        # this allows us to index into the raster array. 
        # we then add the polarity to get which of the two possible indexes for the raster we add the event to. 
        # Note that every event is a spike, regardless of polarity. The reason we have the + polarity is to 
        # allow us find the specific column for that specific channel (becuase each channel has 2 columns). 

        valid = (channels < self.total_features)
        np.add.at(raster, (times[valid], channels[valid]), 1)

        # Pad or truncate to sample_T
        if raster.shape[0] < self.sample_T:
            pad = np.zeros((self.sample_T - raster.shape[0], self.total_features), dtype=np.float32)
            raster = np.concatenate([raster, pad], axis=0)
        else:
            raster = raster[:self.sample_T, :]
        # the reason we are doing the code above is becuase the input data is a variable length.
        # some audio samples are longer than others, so we need to pad or truncate to a fixed length.
        # we are padding with 0s if the sample is shorter than the fixed length.
        # we are truncating if the sample is longer than the fixed length.
        
        # Its important to keep the data size the exact same even though it doesn't come in the same size. 
        # This is because the model is going to be trained on batches of data, and if the data size is different
        # for each batch, the model will not be able to learn effectively. Also, the model will not be able to 
        # make predictions on new data if the data size is different. 


        # Binarize
        return (raster > 0).astype(np.float32)



# =============================================================================
# Main
# =============================================================================



def main():
    parser = argparse.ArgumentParser(description="SHD Training")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=128)
    parser.add_argument('--sample_T', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--output_path', type=str, default='./results')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Constants
    NUM_CHANNELS = 700
    NUM_POLARITIES = 2
    INPUT_SIZE = NUM_CHANNELS * NUM_POLARITIES  # 1400
    NUM_CLASSES = 20
    NET_DT = 10e-3

    # ==========================================================================
    # Load Data
    # ==========================================================================
    print("\nLoading SHD dataset...")

    transform = transforms.Compose([
        transforms.Downsample(time_factor=1e-6 / NET_DT, spatial_factor=1.0),
        ToRaster(NUM_CHANNELS, NUM_POLARITIES, args.sample_T),
        torch.tensor,
    ])

    # The downsample multiplies each time step by 1e-6 / NET_DT. 
    # This converts the micro seconds data into milliseconds, which is the bin size we want. 
    # So its important to note that we don't have like 100 bins inistantly. 
    # Its AROUND 100 bins and then we crop or add bins later on in the raster class
    # The ToRaster class then converts the data into a raster with the specified sample_T 
    # The torch.tensor converts the data into a torch tensor


    train_data = datasets.SHD(args.data_path, train=True, transform=transform)
    test_data = datasets.SHD(args.data_path, train=False, transform=transform)
    # Apply the transform to the data when you load it in using tonic datasets function

    cache_path = f"{args.data_path}/cache/shd_full_{args.sample_T}T"
    cached_train = tonic.DiskCachedDataset(train_data, cache_path=f"{cache_path}/train")
    cached_test = tonic.DiskCachedDataset(test_data, cache_path=f"{cache_path}/test")

    train_loader = DataLoader(
        cached_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True)
    )

    test_loader = DataLoader(
        cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True)
    )

    print(f"Train: {len(cached_train)} samples, Test: {len(cached_test)} samples")

    # ==========================================================================
    # Build Network
    # ==========================================================================
    print(f"\nBuilding network: {INPUT_SIZE} → {args.hidden1} → {args.hidden2} → {NUM_CLASSES}")

    net = Sequential(
        LinearTorch((INPUT_SIZE, args.hidden1), has_bias=False),
        LIFTorch(args.hidden1, tau_mem=Constant(0.1), tau_syn=Constant(0.1),
                 threshold=Constant(1.0), bias=Constant(0.), dt=NET_DT, has_rec=False),
        LinearTorch((args.hidden1, args.hidden2), has_bias=False),
        LIFTorch(args.hidden2, tau_mem=Constant(0.1), tau_syn=Constant(0.1),
                 threshold=Constant(1.0), bias=Constant(0.), dt=NET_DT, has_rec=False),
        LinearTorch((args.hidden2, NUM_CLASSES), has_bias=False),
        ExpSynTorch(NUM_CLASSES, dt=NET_DT, tau=Constant(5e-3))
    ).to(device)

    total_params = sum(p.numel() for p in net.parameters().astorch())
    print(f"Total parameters: {total_params:,}")

    # ==========================================================================
    # Training Setup
    # ==========================================================================
    optimizer = torch.optim.Adam(net.parameters().astorch(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_test_acc = 0.0

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # Train
        net.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for data, labels in pbar:
            data, labels = data.to(device).float(), labels.to(device)

            optimizer.zero_grad()
            output, _, _ = net(data)

            # Cumsum integration (from Rockpool tutorial)
            logits = torch.cumsum(output, dim=1)[:, -1, :]

            loss = loss_fn(logits, labels)
            loss.backward()
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
                logits = torch.cumsum(output, dim=1)[:, -1, :]
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs(args.output_path, exist_ok=True)
            torch.save(net.state_dict(), f"{args.output_path}/best_model.pth")

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%, Best={best_test_acc:.1f}%")

    print(f"\nDone! Best test accuracy: {best_test_acc:.1f}%")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
