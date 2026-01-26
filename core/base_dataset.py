"""Base dataset class for neuromorphic dataset loading."""

import os
from abc import ABC, abstractmethod
import tonic


class NeuromorphicDataset(ABC):
    """Base class for neuromorphic dataset loading."""

    def __init__(self, dataset_path, n_frames, **kwargs):
        self.dataset_path = dataset_path
        self.n_frames = n_frames
        self.kwargs = kwargs

    @abstractmethod
    def _get_transforms(self):
        """Return tonic transforms for this dataset."""
        pass

    @abstractmethod
    def _load_raw_dataset(self, train=True):
        """Load dataset from tonic."""
        pass

    @abstractmethod
    def _get_cache_path(self):
        """Return cache directory path."""
        pass

    def create_datasets(self):
        """
        Create train/test datasets with caching.
        Returns: (cached_train, cached_test)
        """
        cache_root = self._get_cache_path()
        cache_exists = os.path.exists(f"{cache_root}/train") and \
                      os.path.exists(f"{cache_root}/test")

        if not cache_exists:
            train_dataset = self._load_raw_dataset(train=True)
            test_dataset = self._load_raw_dataset(train=False)
        else:
            train_dataset = None
            test_dataset = None

        cached_train = tonic.DiskCachedDataset(train_dataset, cache_path=f"{cache_root}/train")
        cached_test = tonic.DiskCachedDataset(test_dataset, cache_path=f"{cache_root}/test")

        # Force cache population on first load
        if not cache_exists:
            print(f"Caching dataset to {cache_root}...")
            print("Caching training data...")
            for _ in cached_train:
                pass
            print("Caching test data...")
            for _ in cached_test:
                pass
            print("Caching complete!")

        return cached_train, cached_test

    @abstractmethod
    def get_num_classes(self):
        """Return number of classes."""
        pass