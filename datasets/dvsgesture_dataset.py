"""DVSGesture dataset loader."""

import os
import tonic
import numpy as np


class DVSGestureDataset:
    """DVSGesture dataset loader."""

    def __init__(self, dataset_path, w=32, h=32, n_frames=32):
        self.dataset_path = dataset_path
        self.w = w
        self.h = h
        self.n_frames = n_frames
        self.num_classes = 11
        self.sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)

    def load_dvsgesture(self):
        """Load DVSGesture dataset with caching."""
        cache_path = f"{self.dataset_path}/dvsgesture/{self.w}x{self.h}_T{self.n_frames}"
        cache_exists = os.path.exists(f"{cache_path}/train") and os.path.exists(f"{cache_path}/test")

        # Create transforms - output shape: [T, C*H*W] = [n_frames, w*h*2]
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(spatial_factor=(self.w/self.sensor_size[1], self.h/self.sensor_size[0])),
            tonic.transforms.ToFrame(sensor_size=(self.w, self.h, 2), n_time_bins=self.n_frames),
            lambda x: (x > 0).astype(np.float32),  # Binarize: counts → 0/1
            lambda x: x.reshape(x.shape[0], -1),   # Flatten: [T, C, H, W] → [T, C*H*W]
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
