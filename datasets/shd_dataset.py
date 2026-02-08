"""SHD dataset loader."""

import os
import tonic
import numpy as np


class SHDDataset:
    """SHD dataset loader."""

    def __init__(self, dataset_path, NUM_CHANNELS=700, NUM_POLARITIES=2, n_frames=100, net_dt=10e-3):
        self.dataset_path = dataset_path
        self.NUM_CHANNELS = NUM_CHANNELS
        self.NUM_POLARITIES = NUM_POLARITIES
        self.n_frames = n_frames
        self.net_dt = net_dt
        self.num_classes = 20
        self.total_features = NUM_CHANNELS * NUM_POLARITIES  # 1400

    def _to_raster(self, events):
        """Convert SHD events to raster with full channels and polarity."""
        max_t = int(events["t"].max()) + 1
        raster = np.zeros((max_t, self.total_features), dtype=np.float32)

        times = events["t"].astype(int)
        channels = events["x"].astype(int) * self.NUM_POLARITIES + events["p"].astype(int)

        valid = (channels < self.total_features)
        np.add.at(raster, (times[valid], channels[valid]), 1)

        # Pad or truncate to n_frames
        if raster.shape[0] < self.n_frames:
            pad = np.zeros((self.n_frames - raster.shape[0], self.total_features), dtype=np.float32)
            raster = np.concatenate([raster, pad], axis=0)
        else:
            raster = raster[:self.n_frames, :]

        # Binarize
        return (raster > 0).astype(np.float32)

    def load_shd(self):
        """Load SHD dataset with caching."""
        cache_path = f"{self.dataset_path}/shd/{self.NUM_CHANNELS}x{self.NUM_POLARITIES}_T{self.n_frames}"
        cache_exists = os.path.exists(f"{cache_path}/train") and os.path.exists(f"{cache_path}/test")

        # Create transform - output shape: [T, features] = [n_frames, NUM_CHANNELS * NUM_POLARITIES]
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(time_factor=(1e-6/self.net_dt), spatial_factor=1.0),
            self._to_raster,
        ])

        # Load raw datasets only if cache doesn't exist
        train_dataset = None if cache_exists else tonic.datasets.SHD(save_to=self.dataset_path, transform=transform, train=True)
        test_dataset = None if cache_exists else tonic.datasets.SHD(save_to=self.dataset_path, transform=transform, train=False)

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
