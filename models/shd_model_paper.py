"""Fully-connected SNN model for SHD dataset using Rockpool.

Adapted to match Meszaros et al. (Loihi2/SHD) paper:
 - Feedforward architecture with two hidden layers of 512 LIF neurons each.
 - Utilities added to compute spike counts over a DataLoader and
   to estimate energy-per-spike using a paper-reported average
   inference energy (Joules / inference).
"""

import torch
import torch.nn as nn
from rockpool.nn.modules import LIFTorch, LinearTorch, ExpSynTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant
import math


class SHDSNN:
    """
    Fully-connected SNN for SHD dataset.

    Architecture (paper-like feedforward): input_size -> 512 -> 512 -> num_classes
      - LIF layers (LIFTorch) after each hidden linear layer
      - ExpSynTorch as output integrator (non-firing output)

    NOTE:
      - SHD inputs were described in the paper as 700 cochlea channels.
        Pass input_size accordingly (e.g., 700).
      - For accurate per-sample spike counts, use a DataLoader with batch_size=1
        when calling `compute_spike_counts_on_loader`. With batch_size>1 the
        method will average spikes per sample within each batch (best-effort).
    """

    def __init__(self, input_size, n_frames, tau_mem=0.1, tau_syn=0.1, spike_lam=0.0,
                 model_type="dense", device=None, num_classes=20, lr=0.001, dt=1e-3, threshold=1.0, has_bias=True):
        # prefer user-specified device; fallback to cuda -> cpu
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.input_size = input_size        # e.g., 700 cochlea channels
        self.n_frames = n_frames
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.spike_lam = spike_lam
        self.model_type = model_type
        self.num_classes = num_classes
        self.lr = lr
        self.dt = dt
        self.threshold = threshold
        self.has_bias = has_bias
        self.epochs = None

        # Paper architecture: 2 hidden layers with 512 LIF neurons (for SHD)
        hidden_size = 512

        # Build network: input_size -> 512 -> 512 -> num_classes
        self.net = Sequential(
            LinearTorch((self.input_size*2, hidden_size), has_bias=has_bias),
            LIFTorch(hidden_size, tau_mem=Constant(tau_mem), tau_syn=Constant(tau_syn),
                     threshold=Constant(threshold), bias=Constant(0.), dt=dt, has_rec=False),
            LinearTorch((hidden_size, hidden_size), has_bias=has_bias),
            LIFTorch(hidden_size, tau_mem=Constant(tau_mem), tau_syn=Constant(tau_syn),
                     threshold=Constant(threshold), bias=Constant(0.), dt=dt, has_rec=False),
            LinearTorch((hidden_size, num_classes), has_bias=has_bias),
            # ExpSynTorch output layer: integrates synaptic input for classification (non-spiking)
            ExpSynTorch(num_classes, dt=dt, tau=Constant(5e-3)),
        ).to(self.device)

        # Training components
        self.optimizer = torch.optim.Adam(
            [p for _, p in self.net.named_parameters()], lr=lr, betas=(0.9, 0.999)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )

        # History tracking
        self.loss_hist = []
        self.acc_hist = []
        self.test_acc_hist = []

    # ---------------------------
    # Training / eval helpers
    # ---------------------------
    def train_model(self, train_loader, test_loader, num_epochs=150, print_every=15):
        """Train the model."""
        print("starting training")
        self.epochs = num_epochs
        cnt = 0

        for epoch in range(num_epochs):
            for batch, (data, targets) in enumerate(train_loader):
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                self.net.train()
                self.optimizer.zero_grad()

                output, _, recording = self.net(data, record=True)
                logits = output.mean(dim=1)

                # Count spikes from LIF layers only (not ExpSynTorch output)
                spike_count = torch.tensor(0.0, device=self.device)
                # Best-effort: if recording corresponds to batch, we sum and use it directly;
                # for per-sample exact counts prefer batch_size=1 in train_loader.
                for key, value in recording.items():
                    # value might be a tensor or dict with 'spk'/'spikes' etc.
                    if isinstance(value, dict):
                        # common rockpool keys: 'spk', 's' or 'spikes'
                        candidate = None
                        for k in ('spk', 's', 'spikes'):
                            if k in value:
                                candidate = value[k]
                                break
                        if candidate is not None:
                            spike_count += candidate.sum()
                        else:
                            # sum numeric items if possible
                            for vv in value.values():
                                if torch.is_tensor(vv):
                                    spike_count += vv.sum()
                    elif torch.is_tensor(value):
                        spike_count += value.sum()
                    else:
                        # fallback: try to coerce to tensor (numpy)
                        try:
                            import numpy as _np
                            arr = _np.asarray(value)
                            spike_count += torch.tensor(arr.sum(), device=self.device)
                        except Exception:
                            pass

                loss = self.loss_fn(logits, targets) + spike_count * self.spike_lam

                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in self.net.named_parameters()], max_norm=1.0)
                self.optimizer.step()

                self.loss_hist.append(loss.item())
                acc = ((logits.argmax(1) == targets).sum().item()) / targets.size(0)
                self.acc_hist.append(acc)

                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iter {batch} | Loss: {loss.item():.4f} | Train Acc: {acc*100:.2f}%")
                    test_acc = self.validate_model(test_loader)
                    self.test_acc_hist.append(test_acc)
                    print(f"Test Acc: {test_acc*100:.2f}% | LR: {self.optimizer.param_groups[0]['lr']:.6f}\n")
                    self.scheduler.step(test_acc)

                cnt += 1

    def validate_model(self, test_loader):
        """Validate the model on test data."""
        self.net.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                output, _, _ = self.net(data)
                logits = output.mean(dim=1)
                correct += (logits.argmax(1) == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0

    # ---------------------------
    # Spike counting & energy estimate utilities
    # ---------------------------
    @staticmethod
    def _count_spikes_in_recording(recording):
        """
        Helper: best-effort total spike count in a recording dict returned by rockpool net(..., record=True).
        Returns scalar (float or int).
        """
        total = 0.0
        for key, value in recording.items():
            if isinstance(value, dict):
                # try common keys
                for k in ('spk', 's', 'spikes'):
                    v = value.get(k, None)
                    if v is not None:
                        if hasattr(v, 'sum'):
                            total += float(v.sum().item())
                        else:
                            try:
                                import numpy as _np
                                total += float(_np.sum(v))
                            except Exception:
                                pass
                        break
                else:
                    # sum any tensor-like entries
                    for vv in value.values():
                        if hasattr(vv, 'sum'):
                            total += float(vv.sum().item())
            elif torch.is_tensor(value):
                total += float(value.sum().item())
            else:
                # try numpy coercion
                try:
                    import numpy as _np
                    arr = _np.asarray(value)
                    total += float(_np.sum(arr))
                except Exception:
                    pass
        return total

    def compute_spike_counts_on_loader(self, data_loader, verbose=False):
        """
        Run the network on every sample in data_loader (best if batch_size=1) and return:
           - total_spikes (float)
           - avg_spikes_per_sample (float)
           - spikes_per_sample (list of floats)  (approx when batch_size>1)
        NOTE: for exact per-sample counts set data_loader.batch_size==1
        """
        self.net.eval()
        spikes_list = []
        total_spikes = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # batch is (data, targets) or (data,) depending on loader
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    data = batch[0]
                else:
                    data = batch

                # Ensure data on device
                data = data.to(self.device).float()

                output, _, recording = self.net(data, record=True)

                # Count spikes in whole recording (may be batch summed). Use helper.
                batch_spikes_total = float(self._count_spikes_in_recording(recording))

                # Determine batch size: look for dimension in output or data
                # We'll try common shapes: data: [B, T, C] or [B, C, T] etc.
                batch_size = 1
                if torch.is_tensor(data):
                    if data.dim() >= 1:
                        batch_size = int(data.shape[0])
                else:
                    # fallback
                    batch_size = 1

                # Distribute total across samples in batch (approx if recording aggregated)
                if batch_size <= 0:
                    batch_size = 1
                per_sample = batch_spikes_total / float(batch_size)

                # append per-sample value batch_size times (so length matches sample count)
                for _ in range(batch_size):
                    spikes_list.append(per_sample)

                total_spikes += batch_spikes_total
                total_samples += batch_size

                if verbose and (batch_idx % 50 == 0):
                    print(f"[spike_count] processed batch {batch_idx}: batch_spikes_total={batch_spikes_total:.1f}, batch_size={batch_size}")

        avg_spikes = (total_spikes / total_samples) if total_samples > 0 else float('nan')

        return {
            'total_spikes': float(total_spikes),
            'n_samples': int(total_samples),
            'avg_spikes_per_sample': float(avg_spikes),
            'spikes_per_sample': spikes_list
        }

    def estimate_energy_per_spike_from_paper(self, paper_energy_joules_per_inference, spike_stats=None, paper_note=None):
        """
        Given a paper-reported average inference energy (Joule per inference), and
        spike counts (preferably the output of compute_spike_counts_on_loader),
        compute estimated energy per spike and estimated total energy for your measured dataset.

        Args:
          - paper_energy_joules_per_inference: float (Joules per inference reported by the reference paper)
          - spike_stats: dict returned by compute_spike_counts_on_loader (or None)
              If None, this method will raise. Provide spike counts measured for your model.
          - paper_note: optional string to identify which paper/figure the number came from.

        Returns:
          dict with:
            - energy_per_spike_J
            - estimated_total_energy_J (paper_energy_per_inference / avg_spikes_per_sample * total_spikes)
            - avg_spikes_per_sample
            - total_spikes
            - n_samples
            - paper_energy_joules_per_inference
            - paper_note
        """
        if spike_stats is None:
            raise ValueError("spike_stats must be provided (use compute_spike_counts_on_loader to get it).")

        total_spikes = float(spike_stats.get('total_spikes', 0.0))
        n_samples = int(spike_stats.get('n_samples', 0))
        avg_spikes = float(spike_stats.get('avg_spikes_per_sample', float('nan')))

        if n_samples <= 0:
            raise ValueError("spike_stats must contain n_samples > 0")

        # If average spikes per sample is zero (no spikes) -> undefined
        if avg_spikes == 0 or math.isnan(avg_spikes):
            energy_per_spike = float('nan')
        else:
            energy_per_spike = float(paper_energy_joules_per_inference / avg_spikes)

        # Estimate total energy for the dataset (using paper energy normalized by spikes)
        if math.isnan(energy_per_spike):
            estimated_total_energy = float('nan')
        else:
            estimated_total_energy = energy_per_spike * total_spikes

        result = {
            'paper_energy_j_per_inference': float(paper_energy_joules_per_inference),
            'paper_note': paper_note,
            'total_spikes': total_spikes,
            'n_samples': n_samples,
            'avg_spikes_per_sample': avg_spikes,
            'energy_per_spike_J': energy_per_spike,
            'estimated_total_energy_J': estimated_total_energy
        }

        return result

    # ---------------------------
    # Save / load routines (unchanged from your original design)
    # ---------------------------
    def save_model(self, base_path="./results", counter_file="experiment_counter.txt"):
        """Save model and training graphs."""
        import matplotlib.pyplot as plt
        import os

        os.makedirs(f"{base_path}/shd/{self.model_type}/models", exist_ok=True)
        os.makedirs(f"{base_path}/shd/{self.model_type}/graphs", exist_ok=True)

        # Counter file
        counter_path = f"{base_path}/shd/{self.model_type}/{counter_file}"
        if not os.path.exists(counter_path):
            os.makedirs(os.path.dirname(counter_path), exist_ok=True)
            with open(counter_path, "w") as f:
                f.write("0")

        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(self.acc_hist)
        axes[0].set_title("Train Set Accuracy")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Accuracy")

        axes[1].plot(self.test_acc_hist)
        axes[1].set_title("Test Set Accuracy")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Accuracy")

        axes[2].plot(self.loss_hist)
        axes[2].set_title("Loss History")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Loss")

        # Read and increment counter
        with open(counter_path, "r") as f:
            num = int(f.read().strip()) + 1
        with open(counter_path, "w") as f:
            f.write(str(num))

        # Save paths
        model_path = f"{base_path}/shd/{self.model_type}/models/Take{num}_T{self.n_frames}_Epochs{self.epochs}.pth"
        graph_path = f"{base_path}/shd/{self.model_type}/graphs/Take{num}_T{self.n_frames}_Epochs{self.epochs}.png"

        # Save checkpoint
        torch.save({
            'state_dict': self.net.state_dict(),
            'hyperparams': {
                'input_size': self.input_size,
                'n_frames': self.n_frames,
                'tau_mem': self.tau_mem,
                'tau_syn': self.tau_syn,
                'spike_lam': self.spike_lam,
                'model_type': self.model_type,
                'num_classes': self.num_classes,
                'dt': self.dt,
                'threshold': self.threshold,
                'has_bias': self.has_bias,
            }
        }, model_path)

        plt.savefig(graph_path)
        plt.show()

        print(f"Model saved to: {model_path}")
        print(f"Graph saved to: {graph_path}")

    def load_model(self, model_path):
        """Load model weights from file."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['state_dict'])
        else:
            self.net.load_state_dict(checkpoint)
        self.net.eval()
        print(f"Model loaded from: {model_path}")

    @staticmethod
    def load_hyperparams(model_path, device='cpu'):
        """Load hyperparameters from checkpoint file."""
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'hyperparams' in checkpoint:
            return checkpoint['hyperparams']
        else:
            raise ValueError(f"Checkpoint at {model_path} does not contain hyperparams. "
                             "Was it saved with an older version?")