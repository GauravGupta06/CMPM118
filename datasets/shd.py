"""SHD dataset loader."""

import os
import tonic
import torch
import numpy as np
from core.base_dataset import NeuromorphicDataset


class SHDDataset(NeuromorphicDataset):
    """SHD dataset loader."""

    def __init__(self, dataset_path, n_frames=100):
        """
        Args:
            dataset_path: Root path for dataset storage
            n_frames: Number of temporal bins
        """
        super().__init__(dataset_path, n_frames)
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
        return f"{self.dataset_path}/shd/700x1_T{self.n_frames}"

    def get_num_classes(self):
        """Return number of classes for SHD."""
        return self.num_classes


def load_shd(dataset_path, n_frames=100):
    """
    Load SHD dataset.

    Args:
        dataset_path: Root path for dataset storage
        n_frames: Number of temporal bins

    Returns:
        (cached_train, cached_test, num_classes)
    """
    dataset = SHDDataset(dataset_path, n_frames)
    return dataset.create_datasets()
