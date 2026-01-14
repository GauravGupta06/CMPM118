"""DVSGesture dataset loader."""

import os
import tonic
import torch
import numpy as np
from core.base_dataset import NeuromorphicDataset


class DVSGestureDataset(NeuromorphicDataset):
    """DVSGesture dataset loader."""

    def __init__(self, dataset_path, w=32, h=32, n_frames=32): # Width & height added
        """
        Args:
            dataset_path: Root path for dataset storage
            w: Width of samples
            h: Height of samples
            n_frames: Number of temporal bins
        """
        super().__init__(dataset_path, n_frames) # dataset_path should simply be in the main directory (in this case, CMPM118)
        self.sensor_size = tonic.datasets.DVSGesture.sensor_size # (128, 128, 2)
        self.num_classes = 11 # Changed
        self.w = w # Added
        self.h = h # Added

    def _get_transforms(self):
        """Create tonic transforms for DVSGesture dataset."""
        transforms_list = [
            tonic.transforms.ToFrame(
                sensor_size=self.sensor_size,
                n_time_bins=self.n_frames
            ),
        ]

        return tonic.transforms.Compose(transforms_list)

    def _load_raw_dataset(self, train=True):
        """Load DVSGesture dataset from tonic."""
        return tonic.datasets.DVSGesture(
            save_to=self.dataset_path,
            transform=self._get_transforms(),
            train=train
        )

    def _get_cache_path(self):
        """Generate cache path based on configuration."""
        return f"{self.dataset_path}/DVSGesture/{self.w}x{self.h}_T{self.n_frames}" # Changed to DVSGesture path

    def get_num_classes(self):
        """Return number of classes for DVSGesture."""
        return self.num_classes


def load_dvsgesture(dataset_path, n_frames=100):
    """
    Load DVSGesture dataset.

    Args:
        dataset_path: Root path for dataset storage
        n_frames: Number of temporal bins

    Returns:
        (cached_train, cached_test, num_classes)
    """
    dataset = DVSGestureDataset(dataset_path, n_frames)
    return dataset.create_datasets()
