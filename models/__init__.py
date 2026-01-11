"""Models module with SNN model implementations."""

from .shd_model import SHDSNN_FC

MODEL_REGISTRY = {
    'SHDSNN_FC': SHDSNN_FC,
}

__all__ = ['SHDSNN_FC', 'MODEL_REGISTRY']
