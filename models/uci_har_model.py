"""Fully-connected SNN model for UCI HAR dataset using Rockpool."""

import torch
import numpy as np
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.base_model import BaseSNNModel


class UCIHARSNN(BaseSNNModel):
    """
    Fully-connected SNN for UCI HAR dataset.

    Architecture:
        9 → hidden → hidden → 6

    - No polarity (continuous IMU signals)
    - Xylo-compatible (FC + LIF only)
    - Sparse vs Dense via tau_mem, spike_lam, hidden_size
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        n_frames,
        tau_mem=0.02,
        spike_lam=1e-7,
        model_type="dense",
        device=None,
        num_classes=6,
        lr=0.001
    ):
        self.input_size = input_size          # 9
        self.hidden_size = hidden_size        # 64 (sparse) / 128 (dense)

        super().__init__(
            n_frames=n_frames,
            tau_mem=tau_mem,
            spike_lam=spike_lam,
            model_type=model_type,
            device=device,
            num_classes=num_classes,
            lr=lr
        )

    def _build_network(self):
        """
        UCI HAR FC SNN:
        9 → hidden → hidden → 6
        """

        threshold = 0.5  # good for continuous IMU signals

        input_size = self.input_size  # 9
        hidden_size = 128
        threshold = 1.0

        net = Sequential(
            LinearTorch((input_size, hidden_size)),
            LIFTorch(hidden_size, tau_mem=self.tau_mem, threshold=threshold, dt=self.dt),
            LinearTorch((hidden_size, hidden_size)),
            LIFTorch(hidden_size, tau_mem=self.tau_mem, threshold=threshold, dt=self.dt),
            LinearTorch((hidden_size, self.num_classes))  # output layer
        )


        return net.to(self.device)

    def _prepare_input(self, data):
        """
        Input shape from DataLoader:
            [T, B, 1, 9]
        """
        data = data.squeeze(2)      # [T, B, 9]
        data = data.transpose(0, 1) # [B, T, 9]
        return data

    def _get_save_params(self):
        return f"UCIHAR_Input{self.input_size}_Hidden{self.hidden_size}_T{self.n_frames}_FC"
