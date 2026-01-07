"""Core module with base classes for SNN models and datasets."""

from .base_model import BaseSNNModel
from .base_dataset import NeuromorphicDataset

__all__ = ['BaseSNNModel', 'NeuromorphicDataset']
