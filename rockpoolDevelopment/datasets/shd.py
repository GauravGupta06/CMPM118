"""SHD dataset loader with optional frequency binning."""

import os
import tonic
import torch
import numpy as np
from core.base_dataset import NeuromorphicDataset


class FrequencyBinning:
    """Tonic transform: Bin 700 frequencies → 16 bands."""

    def __init__(self, n_bins=16):
        self.n_bins = n_bins

    def __call__(self, events):
        """
        Bin frequencies from 700 → n_bins.
        Input shape: [T, 2, 1, 700]
        Output shape: [T, 2, 1, n_bins]
        """
        T, C, H, freq = events.shape

        # Split into n_bins and average each bin
        bins = np.array_split(events, self.n_bins, axis=-1)
        binned = np.array([b.mean(axis=-1, keepdims=True) for b in bins])
        binned = np.concatenate(binned, axis=-1)  # [T, 2, 1, n_bins]

        return binned


class SHDDataset(NeuromorphicDataset):
    """SHD dataset loader with optional frequency binning."""

    def __init__(self, dataset_path, n_frames=100, reduce_to_16=False):
        """
        Args:
            dataset_path: Root path for dataset storage
            n_frames: Number of temporal bins
            reduce_to_16: If True, bin 700 frequencies → 16 bands (Xylo-compatible)
        """
        super().__init__(dataset_path, n_frames)
        self.reduce_to_16 = reduce_to_16
        self.sensor_size = (700, 1, 2)
        self.num_classes = 20

    def _get_transforms(self):
        """Create tonic transforms for SHD dataset."""
        transforms_list = [
            tonic.transforms.ToFrame(
                sensor_size=self.sensor_size,
                n_time_bins=self.n_frames
            ),
        ]

        if self.reduce_to_16:
            transforms_list.append(FrequencyBinning(n_bins=16))

        return tonic.transforms.Compose(transforms_list)

    def _load_raw_dataset(self, train=True):
        """Load SHD dataset from tonic."""
        return tonic.datasets.SHD(
            save_to=self.dataset_path,
            transform=self._get_transforms(),
            train=train
        )

    def _get_cache_path(self):
        """Generate cache path based on configuration."""
        freq_str = "16bands" if self.reduce_to_16 else "700x1"
        return f"{self.dataset_path}/shd/{freq_str}_T{self.n_frames}"

    def get_num_classes(self):
        """Return number of classes for SHD."""
        return self.num_classes


def load_shd(dataset_path, n_frames=100, reduce_to_16=False):
    """
    Load SHD dataset.

    Args:
        dataset_path: Root path for dataset storage
        n_frames: Number of temporal bins
        reduce_to_16: If True, bin 700 frequencies → 16 bands (Xylo-compatible)

    Returns:
        (cached_train, cached_test, num_classes)
    """
    dataset = SHDDataset(dataset_path, n_frames, reduce_to_16)
    return dataset.create_datasets()
