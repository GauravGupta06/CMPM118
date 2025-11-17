"""Energy estimation helpers for neuromorphic simulations.

The goal of this module is to provide a lightweight, hardware-agnostic API for
estimating how much energy an SNN inference or router operation consumes. The
implementation intentionally does not depend on vendor SDKs (e.g., Xylo or
Loihi) so that it can run inside CI and unit tests, while still exposing knobs
that match the parameters published for those platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import collections


@dataclass
class NeuromorphicHardwareConfig:
    """Configuration constants that describe a neuromorphic target."""

    name: str = "xylo"
    spike_energy_pj: float = 23.6
    synaptic_event_energy_pj: float = 4.1
    neuron_leak_energy_pj: float = 0.0
    router_energy_pj: float = 9.2
    static_energy_pj: float = 0.0
    description: str = ""
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "spike_energy_pj": self.spike_energy_pj,
            "synaptic_event_energy_pj": self.synaptic_event_energy_pj,
            "neuron_leak_energy_pj": self.neuron_leak_energy_pj,
            "router_energy_pj": self.router_energy_pj,
            "static_energy_pj": self.static_energy_pj,
            "description": self.description,
            **self.metadata,
        }


class EnergyAccumulator:
    """Utility that keeps running statistics for Joule estimates."""

    def __init__(self) -> None:
        self.total_energy_j = 0.0
        self.samples = 0
        self.by_tag = collections.defaultdict(float)

    def record(self, energy_j: float, model_tag: Optional[str] = None) -> None:
        self.total_energy_j += energy_j
        self.samples += 1
        if model_tag:
            self.by_tag[model_tag] += energy_j

    def average(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.total_energy_j / self.samples

    def summary(self) -> Dict[str, float]:
        summary = {
            "total_energy_j": self.total_energy_j,
            "num_samples": self.samples,
            "average_energy_j": self.average(),
        }
        if self.by_tag:
            summary.update({f"total_{tag}_energy_j": energy for tag, energy in self.by_tag.items()})
        return summary


class EnergyProfiler:
    """Abstract interface for profilers."""

    def __init__(self) -> None:
        self.accumulator = EnergyAccumulator()

    def estimate_inference(
        self,
        spike_count: float,
        *,
        model_tag: Optional[str] = None,
        synaptic_events: Optional[float] = None,
    ) -> float:
        raise NotImplementedError

    def summary(self) -> Dict[str, float]:
        return self.accumulator.summary()


class NullEnergyProfiler(EnergyProfiler):
    """Profiler that always returns zero Joules."""

    def estimate_inference(
        self,
        spike_count: float,
        *,
        model_tag: Optional[str] = None,
        synaptic_events: Optional[float] = None,
    ) -> float:
        energy = 0.0
        self.accumulator.record(energy, model_tag=model_tag)
        return energy


class XyloEnergyProfiler(EnergyProfiler):
    """Closed-form approximation of the Xylo energy model."""

    def __init__(
        self,
        config: Optional[NeuromorphicHardwareConfig] = None,
        *,
        synaptic_scaling: float = 4.0,
        calibration_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config or NeuromorphicHardwareConfig()
        self.synaptic_scaling = synaptic_scaling
        self.calibration_factor = calibration_factor

    def estimate_inference(
        self,
        spike_count: float,
        *,
        model_tag: Optional[str] = None,
        synaptic_events: Optional[float] = None,
    ) -> float:
        synaptic_events = (
            synaptic_events if synaptic_events is not None else spike_count * self.synaptic_scaling
        )
        energy_pj = (
            spike_count * self.config.spike_energy_pj
            + synaptic_events * self.config.synaptic_event_energy_pj
            + self.config.neuron_leak_energy_pj
            + self.config.static_energy_pj
        )
        energy_j = energy_pj * 1e-12 * self.calibration_factor
        self.accumulator.record(energy_j, model_tag=model_tag)
        return energy_j


class RouterEnergyProfiler:
    """Tracks router-side decision energy."""

    def __init__(self, router_energy_pj: float = 0.0) -> None:
        self.router_energy_pj = router_energy_pj
        self.accumulator = EnergyAccumulator()

    def record(self, operations: float = 1.0) -> float:
        energy_j = max(operations, 0.0) * self.router_energy_pj * 1e-12
        self.accumulator.record(energy_j, model_tag="router")
        return energy_j

    def average(self) -> float:
        return self.accumulator.average()

    def summary(self) -> Dict[str, float]:
        return self.accumulator.summary()


__all__ = [
    "NeuromorphicHardwareConfig",
    "EnergyProfiler",
    "NullEnergyProfiler",
    "XyloEnergyProfiler",
    "RouterEnergyProfiler",
]
