"""Fully-connected SNN model for SHD dataset using Rockpool."""

import torch
import torch.nn as nn
import numpy as np
<<<<<<< HEAD
from rockpool.nn.modules import LIFTorch, LinearTorch, ExpSynTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant
=======
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential
>>>>>>> feature/uci_har

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.base_model import BaseSNNModel


class SHDSNN_FC(BaseSNNModel):
    """
    Fully-connected SNN for SHD dataset using Rockpool.
<<<<<<< HEAD
    Architecture: (input_size*2) → 128 → 64 → 32 → 20 (with recurrent connections)
=======
    Architecture: (input_size*2) → 256 → 128 → 20
>>>>>>> feature/uci_har
    Note: SHD has 2 channels (polarity), so total input = input_size * 2
    Xylo-compatible (no Conv layers).

    Sparse vs Dense differentiation via hyperparameters:
    - Sparse: tau_mem=0.01, spike_lam=1e-6 → fewer spikes
    - Dense: tau_mem=0.02, spike_lam=1e-8 → more spikes
    """

<<<<<<< HEAD
    def __init__(self, input_size, n_frames, tau_mem=0.1, tau_syn=0.1, spike_lam=0.0,
                 model_type="dense", device=None, num_classes=20, lr=0.001, dt=10e-3, threshold=1.0, has_bias=True):
=======
    def __init__(self, input_size, n_frames, tau_mem=0.02, spike_lam=1e-7,
                 model_type="dense", device=None, num_classes=20, lr=0.001):
>>>>>>> feature/uci_har
        """
        Args:
            input_size: Number of frequency bins (700 for SHD). Total input features = input_size * 2
            n_frames: Number of time steps
<<<<<<< HEAD
            tau_mem: Membrane time constant in seconds (default 0.1 = 100ms like Rockpool tutorial)
            tau_syn: Synaptic time constant in seconds (default 0.1 = 100ms)
            spike_lam: Spike regularization (default 0.0 = disabled)
=======
            tau_mem: Membrane time constant in seconds (0.01 for sparse, 0.02 for dense)
            spike_lam: Spike regularization (1e-6 for sparse, 1e-8 for dense)
>>>>>>> feature/uci_har
            model_type: "sparse" or "dense" (for tracking/saving)
            device: torch device
            num_classes: Number of output classes (20 for SHD)
            lr: Learning rate
        """
<<<<<<< HEAD
        self.dt = dt
        self.threshold = threshold
        self.has_bias = has_bias
        self.input_size = input_size
        self.tau_syn = tau_syn
        super().__init__(n_frames, tau_mem, spike_lam, model_type, device, num_classes, lr=lr, dt=self.dt, threshold=threshold, has_bias=has_bias)

    def _build_network(self):
        """
        Build FC architecture with recurrence: (input_size*2) → 128 → 64 → 32 → 20
        Note: input_size is multiplied by 2 because SHD has 2 channels (polarity)
        Uses Rockpool's Sequential + LinearTorch + LIFTorch with recurrent connections
=======
        self.input_size = input_size
        super().__init__(n_frames, tau_mem, spike_lam, model_type, device, num_classes, lr=lr)

    def _build_network(self):
        """
        Build FC architecture: (input_size*2) → 256 → 128 → 20
        Note: input_size is multiplied by 2 because SHD has 2 channels (polarity)
        Uses Rockpool's Sequential + LinearTorch + LIFTorch
>>>>>>> feature/uci_har
        """
        # SHD has 2 channels, so actual input features = input_size * 2
        actual_input_size = self.input_size * 2

<<<<<<< HEAD
        # Use Constant() to keep LIF parameters fixed during training (only train weights)
        # has_rec=True enables recurrent connections within each LIF layer (good for temporal data)
        net = Sequential(
            LinearTorch((actual_input_size, 128), has_bias=self.has_bias),
            LIFTorch(128, tau_mem=Constant(self.tau_mem), tau_syn=Constant(self.tau_syn), 
                     threshold=Constant(1.0), bias=Constant(0.), dt=self.dt, has_rec=True),
            LinearTorch((128, 64), has_bias=self.has_bias),
            LIFTorch(64, tau_mem=Constant(self.tau_mem), tau_syn=Constant(self.tau_syn),
                     threshold=Constant(1.0), bias=Constant(0.), dt=self.dt, has_rec=True),
            LinearTorch((64, 32), has_bias=self.has_bias),
            LIFTorch(32, tau_mem=Constant(self.tau_mem), tau_syn=Constant(self.tau_syn),
                     threshold=Constant(1.0), bias=Constant(0.), dt=self.dt, has_rec=True),
            LinearTorch((32, self.num_classes), has_bias=self.has_bias),
            # ExpSynTorch output layer: produces smooth synaptic current instead of spikes
            ExpSynTorch(self.num_classes, dt=self.dt, tau=Constant(5e-3)),
        ).to(self.device)

        # Initialize recurrent weights to be SMALL to prevent cascade explosions
        self._init_small_recurrent_weights(net)
        
        return net
    
    def _init_small_recurrent_weights(self, net):
        """Scale down recurrent weights to prevent spike cascades."""
        with torch.no_grad():
            for name, param in net.named_parameters():
                if 'w_rec' in name.lower() or 'rec' in name.lower():
                    # Scale recurrent weights to 1% of original
                    param.data *= 0.01
                    print(f"Scaled {name} by 0.01 for stability")
=======
        # Higher threshold makes it harder to spike, resulting in sparser/binary output
        threshold = 1

        net = Sequential(
            LinearTorch((actual_input_size, 256), has_bias=True),
            LIFTorch(256, tau_mem=self.tau_mem, threshold=threshold, dt=self.dt),
            LinearTorch((256, 128), has_bias=True),
            LIFTorch(128, tau_mem=self.tau_mem, threshold=threshold, dt=self.dt),
            LinearTorch((128, self.num_classes), has_bias=True),
            LIFTorch(self.num_classes, tau_mem=self.tau_mem, threshold=threshold, dt=self.dt),
        )

        return net.to(self.device)
>>>>>>> feature/uci_har

    def _prepare_input(self, data):
        """
        Prepare SHD input for Rockpool.
        Input: [T, B, C, 1, freq_bins] from tonic (spike counts per bin)
        Output: [B, T, C*freq_bins] for Rockpool (binary spikes)
        """
        T, B = data.size(0), data.size(1)
        x = data.transpose(0, 1)  # [B, T, C, 1, freq_bins]
        x = x.squeeze(3)           # [B, T, C, freq_bins]
        x = x.flatten(2)           # [B, T, C*freq_bins]

        # Convert spike counts to binary (spike happened or not)
<<<<<<< HEAD
        x = (x > 0).float()
=======
        
>>>>>>> feature/uci_har

        return x

    def _get_save_params(self):
        """Get parameters for save filename."""
        return f"Input{self.input_size}_T{self.n_frames}_FC_Rockpool"
