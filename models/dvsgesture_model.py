"""Fully-connected SNN model for DVSGesture dataset using Rockpool."""

import torch
import torch.nn as nn
import numpy as np
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.base_model import BaseSNNModel


class DVSGestureSNN_FC(BaseSNNModel):
    """
    Fully-connected SNN for DVSGesture dataset using Rockpool.
    Architecture: input_size → 256 → 128 → num_classes (For DVSGesture, input_size=(w, h, 2), num_classes=11)
    Xylo-compatible (no Conv layers).

    Sparse vs Dense differentiation via hyperparameters:
    - Sparse: tau_mem=0.01, spike_lam=1e-6 → fewer spikes
    - Dense: tau_mem=0.02, spike_lam=1e-8 → more spikes
    """

    def __init__(self, input_size, n_frames, tau_mem=0.02, tau_syn=0.02, spike_lam=1e-7,
                 model_type="dense", device=None, num_classes=11, lr=0.001, dt=0.001, threshold=1.0, has_bias=True):
        """
        Args:
            input_size: Number of input features per time step, or (HxWxC)
            n_frames: Number of time steps
            tau_mem: Membrane time constant in seconds (0.01 for sparse, 0.02 for dense)
            tau_syn: Synaptic time constant in seconds
            spike_lam: Spike regularization (1e-6 for sparse, 1e-8 for dense)
            model_type: "sparse" or "dense" (for tracking/saving)
            device: torch device
            num_classes: Number of output classes (11 for DVSGesture)
            lr: Learning rate
            dt: Simulation timestep in seconds
            threshold: Spike threshold
            has_bias: Whether linear layers have bias
        """
        self.dt = dt
        self.threshold = threshold
        self.has_bias = has_bias
        self.input_size = input_size
        self.tau_syn = tau_syn
        super().__init__(n_frames, tau_mem, spike_lam, model_type, device, num_classes, lr=lr, dt=self.dt, threshold=threshold, has_bias=has_bias)

    def _build_network(self):
        """
        Build FC architecture: input_size → 256 → 128 → num_classes
        Uses Rockpool's Sequential + LinearTorch + LIFTorch
        """
        net = Sequential(
            LinearTorch((self.input_size, 256), has_bias=self.has_bias),
            LIFTorch(256, tau_mem=Constant(self.tau_mem), tau_syn=Constant(self.tau_syn),
                     threshold=Constant(self.threshold), dt=self.dt),
            LinearTorch((256, 128), has_bias=self.has_bias),
            LIFTorch(128, tau_mem=Constant(self.tau_mem), tau_syn=Constant(self.tau_syn),
                     threshold=Constant(self.threshold), dt=self.dt),
            LinearTorch((128, self.num_classes), has_bias=self.has_bias),
            LIFTorch(self.num_classes, tau_mem=Constant(self.tau_mem), tau_syn=Constant(self.tau_syn),
                     threshold=Constant(self.threshold), dt=self.dt),
        )

        return net.to(self.device)

    def _prepare_input(self, data):
        """
        Prepare DVSGesture input for Rockpool.
        Input: [T, B, C, 1, freq_bins] from tonic (spike counts per bin)
        Output: [B, T, C*freq_bins] for Rockpool (binary spikes)
        """
        # T, B = data.size(0), data.size(1)
        # x = data.transpose(0, 1)   # [B, T, C, 1, freq_bins]
        # x = x.squeeze(3)           # [B, T, C, freq_bins]
        # x = x.flatten(2)           # [B, T, C*freq_bins]

        # # Convert spike counts to binary (spike happened or not)
        # x = (x > 0).float()
        
        # return x
        return data

    def _get_save_params(self):
        """Get parameters for save filename."""
        return f"Input{self.input_size}_T{self.n_frames}_FC_Rockpool"
