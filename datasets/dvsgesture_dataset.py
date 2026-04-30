"""DVSGesture dataset loader matching Arfa et al. (2025) preprocessing.

Preprocessing pipeline:
  1. Denoise (1s temporal, 1px spatial default)
  2. Downsample 128x128 → 32x32
  3. ToFrame with 1ms time-window binning
  4. Binarize
  5. Cap at max_timesteps (600) frames
  Output shape: [T, 2, 32, 32] where T ≤ max_timesteps

IMPORTANT: DataLoader MUST use tonic.collation.PadTensors(batch_first=True) as collate_fn
since samples have variable-length temporal dimensions. PadTensors pads shorter sequences
with zeros, which simply causes membrane leak in LIF neurons — no special handling needed.
"""

import os
import tonic
import numpy as np


class DVSGestureDataset:
    """DVSGesture dataset loader with Arfa et al. (2025) preprocessing."""

    def __init__(self, dataset_path, w=32, h=32, max_timesteps=600):
        self.dataset_path = dataset_path
        self.w = w
        self.h = h
        self.max_timesteps = max_timesteps
        self.num_classes = 11
        self.sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)

    def load_dvsgesture(self):
        """Load DVSGesture dataset with caching."""
        cache_path = f"{self.dataset_path}/dvsgesture/{self.w}x{self.h}_tw1ms_T{self.max_timesteps}"
        cache_exists = os.path.exists(f"{cache_path}/train") and os.path.exists(f"{cache_path}/test")

        # Transform pipeline matching Arfa et al. (2025):
        # - Denoise: remove isolated events (1px spatial, 1s temporal neighborhood)
        # - Downsample: 128x128 → 32x32
        # - ToFrame: 1ms bins (time_window=1000 since DVS timestamps are in µs)
        # - Binarize: counts → 0/1
        # - Temporal cap: max 600 timesteps (matches paper's SpiNNaker2 SRAM limit)
        # - NO flatten: keep [T, 2, 32, 32] for conv architecture
        max_t = self.max_timesteps
        transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=1000000),  # 1s in µs
            tonic.transforms.Downsample(spatial_factor=(self.w / self.sensor_size[1], self.h / self.sensor_size[0])),
            tonic.transforms.ToFrame(sensor_size=(self.w, self.h, 2), time_window=1000),  # 1ms bins
            lambda x: (x > 0).astype(np.float32),  # Binarize
            lambda x: x[:max_t],  # Cap at max_timesteps
            # NO flatten — output is [T, 2, 32, 32] where T ≤ max_timesteps
        ])

        # Load raw datasets only if cache doesn't exist
        train_dataset = None if cache_exists else tonic.datasets.DVSGesture(save_to=self.dataset_path, transform=transform, train=True)
        test_dataset = None if cache_exists else tonic.datasets.DVSGesture(save_to=self.dataset_path, transform=transform, train=False)

        # Create cached datasets
        cached_train = tonic.DiskCachedDataset(train_dataset, cache_path=f"{cache_path}/train")
        cached_test = tonic.DiskCachedDataset(test_dataset, cache_path=f"{cache_path}/test")

        # Populate cache on first load
        if not cache_exists:
            print(f"Caching dataset to {cache_path}...")
            print("Caching training data...")
            for _ in cached_train:
                pass
            print("Caching test data...")
            for _ in cached_test:
                pass
            print("Caching complete!")

        return cached_train, cached_test

    def get_num_classes(self):
        return self.num_classes
