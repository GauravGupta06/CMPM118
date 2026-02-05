"""Fully-connected SNN model for UCI HAR dataset using Rockpool."""

import torch
import torch.nn as nn
import numpy as np
from rockpool.nn.modules import LIFTorch, LinearTorch, ExpSynTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.base_model import BaseSNNModel


class UCIHARSNN_FC(BaseSNNModel):
    """
    Fully-connected SNN for UCI HAR dataset using Rockpool.

    Architecture:
        9 → 128 → 64 → 32 → 6

    Input:
        - 9 IMU channels
        - 128 timesteps
        - continuous-valued signals (not spike events)

    This model injects HAR signals as input current into LIF neurons.
    """

    def __init__(
        self,
        input_size=9,
        n_frames=128,
        tau_mem=0.1,
        tau_syn=0.05,
        spike_lam=0.0,
        model_type="dense",
        device=None,
        num_classes=6,
        lr=0.001,
        dt=0.02,
        threshold=1.0,
        has_bias=True,
    ):
        """
        Args:
            input_size: number of IMU channels (9)
            n_frames: number of timesteps (128)
            tau_mem: membrane time constant
            tau_syn: synaptic time constant
            spike_lam: spike regularization
            model_type: "dense" or "sparse"
            num_classes: 6 HAR activities
            dt: timestep in seconds (50Hz → 0.02s)
        """

        self.dt = dt
        self.threshold = threshold
        self.has_bias = has_bias
        self.input_size = input_size
        self.tau_syn = tau_syn

        super().__init__(
            n_frames=n_frames,
            tau_mem=tau_mem,
            spike_lam=spike_lam,
            model_type=model_type,
            device=device,
            num_classes=num_classes,
            lr=lr,
            dt=self.dt,
            threshold=threshold,
            has_bias=has_bias,
        )

    def _build_network(self):
        """
        Build FC architecture:
            9 → 128 → 64 → 32 → 6

        Uses recurrent LIF layers for temporal modeling.
        """

        net = Sequential(
            LinearTorch((self.input_size, 128), has_bias=self.has_bias),
            LIFTorch(
                128,
                tau_mem=Constant(self.tau_mem),
                tau_syn=Constant(self.tau_syn),
                threshold=Constant(self.threshold),
                bias=Constant(0.0),
                dt=self.dt,
                has_rec=True,
            ),

            LinearTorch((128, 64), has_bias=self.has_bias),
            LIFTorch(
                64,
                tau_mem=Constant(self.tau_mem),
                tau_syn=Constant(self.tau_syn),
                threshold=Constant(self.threshold),
                bias=Constant(0.0),
                dt=self.dt,
                has_rec=True,
            ),

            LinearTorch((64, 32), has_bias=self.has_bias),
            LIFTorch(
                32,
                tau_mem=Constant(self.tau_mem),
                tau_syn=Constant(self.tau_syn),
                threshold=Constant(self.threshold),
                bias=Constant(0.0),
                dt=self.dt,
                has_rec=True,
            ),

            LinearTorch((32, self.num_classes), has_bias=self.has_bias),

            LIFTorch(
                self.num_classes,
                tau_mem=Constant(self.tau_mem),
                tau_syn=Constant(self.tau_syn),
                threshold=Constant(self.threshold),
                bias=Constant(0.0),
                dt=self.dt,
                has_rec=True,
            )
        ).to(self.device)

        self._init_small_recurrent_weights(net)
        return net

    def _init_small_recurrent_weights(self, net):
        """Prevent spike explosions in recurrent layers."""
        with torch.no_grad():
            for name, param in net.named_parameters():
                if "rec" in name.lower():
                    param.data *= 0.01
                    print(f"Scaled {name} by 0.01")

    def _prepare_input(self, data):
        """
        Prepare HAR input for Rockpool.

        DataLoader already provides:
            data: [B, T, C]

        Rockpool expects:
            [B, T, C]
        """

        x = data.float()

        # Increase input current so LIF neurons actually respond
        x = x * 2.0

        # Safety clamp
        x = torch.clamp(x, -5.0, 5.0)

        return x


    def _get_save_params(self):
        """Used for checkpoint naming."""
        return f"HAR_Input{self.input_size}_T{self.n_frames}_FC_Rockpool"
