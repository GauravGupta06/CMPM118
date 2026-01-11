"""Dataset module with neuromorphic dataset loaders."""

from .shd import load_shd, SHDDataset

DATASET_REGISTRY = {
    'SHD': load_shd,
}

__all__ = ['load_shd', 'SHDDataset', 'DATASET_REGISTRY']
