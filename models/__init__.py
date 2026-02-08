"""Models module with SNN model implementations."""

from .shd_model import SHDSNN
from .uci_har_model import UCIHARSNN
from .dvsgesture_model import DVSGestureSNN

MODEL_REGISTRY = {
    'SHDSNN': SHDSNN,
    'UCIHARSNN': UCIHARSNN,
    'DVSGestureSNN': DVSGestureSNN,
}

__all__ = ['SHDSNN', 'UCIHARSNN', 'DVSGestureSNN', 'MODEL_REGISTRY']
