"""Convolutional SNN model for DVSGesture dataset using Rockpool.

Architecture matches Arfa et al. (2025) Table I:
  Input:    [B, T, 2, 32, 32]
  Conv0:    2→16, k=5, s=2, p=1  → 16×15×15
  LIF0:     3600 neurons
  Conv1:    16→16, k=3, s=1, p=1 → 16×15×15
  LIF1:     3600 neurons
  Pool0:    AvgPool 2×2           → 16×7×7
  Conv2:    16→8, k=3, s=1, p=1  → 8×7×7
  LIF2:     392 neurons
  Pool1:    AvgPool 2×2           → 8×3×3
  Flatten:  72
  FC0:      72→256
  LIF3:     256 neurons
  FC1:      256→11
  LIF4:     11 neurons (output)
  Total:    25,504 parameters
"""

import torch
import torch.nn as nn
import math
from rockpool.nn.modules import LIFTorch
from rockpool.parameters import Constant


class DVSGestureSNN(nn.Module):
    """
    Convolutional SNN for DVSGesture matching Arfa et al. (2025).
    Mixes PyTorch nn.Conv2d/Linear with Rockpool LIFTorch neurons.
    """

    # Energy constants from Arfa et al. Table VII / Section IV-C
    ENERGY_PER_TIMESTEP = 0.765e-3  # Joules per 1ms frame on SpiNNaker2
    ENERGY_PER_SPIKE = None  # Not calibrated — set externally if using per_spike method

    def __init__(self, input_size=None, n_frames=600, tau_mem=0.01378, spike_lam=1e-7,
                 model_type="dense", device=None, num_classes=11, lr=0.003, dt=0.001,
                 threshold=1.0, has_bias=False):
        super().__init__()

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.input_size = input_size  # kept for backward compat, not used in conv arch
        self.n_frames = n_frames
        self.tau_mem = tau_mem
        self.spike_lam = spike_lam
        self.model_type = model_type
        self.num_classes = num_classes
        self.lr = lr
        self.dt = dt
        self.threshold = threshold
        self.has_bias = has_bias
        self.architecture = 'conv_arfa'
        self.epochs = None

        # --- Conv block 1 ---
        self.conv0 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=1, bias=False)
        self.lif0 = LIFTorch(
            (16 * 15 * 15,),
            tau_mem=Constant(tau_mem), threshold=Constant(threshold),
            bias=Constant(0.0), dt=dt, max_spikes_per_dt=1
        )

        # --- Conv block 2 ---
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.lif1 = LIFTorch(
            (16 * 15 * 15,),
            tau_mem=Constant(tau_mem), threshold=Constant(threshold),
            bias=Constant(0.0), dt=dt, max_spikes_per_dt=1
        )
        self.pool0 = nn.AvgPool2d(kernel_size=2, stride=2)

        # --- Conv block 3 ---
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.lif2 = LIFTorch(
            (8 * 7 * 7,),
            tau_mem=Constant(tau_mem), threshold=Constant(threshold),
            bias=Constant(0.0), dt=dt, max_spikes_per_dt=1
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # --- FC block ---
        self.flat = nn.Flatten()
        self.fc0 = nn.Linear(72, 256, bias=False)
        self.lif3 = LIFTorch(
            (256,),
            tau_mem=Constant(tau_mem), threshold=Constant(threshold),
            bias=Constant(0.0), dt=dt, max_spikes_per_dt=1
        )
        self.fc1 = nn.Linear(256, 11, bias=False)
        self.lif4 = LIFTorch(
            (11,),
            tau_mem=Constant(tau_mem), threshold=Constant(threshold),
            bias=Constant(0.0), dt=dt, max_spikes_per_dt=1
        )

        # Weight initialization: small normal (0, 0.01)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.data.normal_(0, 0.01)

        # Move to device
        self.to(self.device)

        # Training components
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )

        # History tracking
        self.loss_hist = []
        self.acc_hist = []
        self.test_acc_hist = []

        # All LIF layers for easy iteration
        self._lif_layers = [self.lif0, self.lif1, self.lif2, self.lif3, self.lif4]

    def _reset_lif_states(self):
        """Reset all LIF neuron states."""
        for lif in self._lif_layers:
            lif.reset_state()

    def forward(self, x, record=False):
        """
        Forward pass through the conv SNN.

        Args:
            x: Input tensor [batch, T, 2, 32, 32]
            record: If True, record spike outputs from each LIF layer

        Returns:
            output_spikes: [batch, T, 11] — per-timestep output spikes
            recordings: dict of per-layer spike tensors if record=True, else {}
        """
        batch, T, C, H, W = x.shape
        recordings = {}

        # Reset all LIF states at start of each forward pass
        self._reset_lif_states()

        # --- Conv block 1: Conv0 → LIF0 ---
        # Batch conv across all timesteps: [batch, T, 2, 32, 32] → [batch*T, 2, 32, 32]
        x_flat = x.reshape(batch * T, C, H, W)
        conv0_flat = self.conv0(x_flat)                   # [batch*T, 16, 15, 15]
        conv0_seq = conv0_flat.reshape(batch, T, -1)      # [batch, T, 3600]

        lif0_out, _, lif0_rec = self.lif0(conv0_seq, record=True)  # [batch, T, 3600]
        if record:
            recordings['lif0'] = lif0_out

        # --- Conv block 2: Conv1 → LIF1 → Pool0 ---
        # Batch conv: reshape LIF0 spikes back to spatial
        conv1_flat = lif0_out.reshape(batch * T, 16, 15, 15)
        conv1_out = self.conv1(conv1_flat)                # [batch*T, 16, 15, 15]
        conv1_seq = conv1_out.reshape(batch, T, -1)       # [batch, T, 3600]

        lif1_out, _, lif1_rec = self.lif1(conv1_seq, record=True)  # [batch, T, 3600]
        if record:
            recordings['lif1'] = lif1_out

        # Pool0: batch pool across all timesteps
        pool0_flat = lif1_out.reshape(batch * T, 16, 15, 15)
        pool0_out = self.pool0(pool0_flat)                # [batch*T, 16, 7, 7]
        pool0_seq = pool0_out.reshape(batch, T, -1)       # [batch, T, 784]

        # --- Conv block 3: Conv2 → LIF2 → Pool1 ---
        conv2_flat = pool0_seq.reshape(batch * T, 16, 7, 7)
        conv2_out = self.conv2(conv2_flat)                # [batch*T, 8, 7, 7]
        conv2_seq = conv2_out.reshape(batch, T, -1)       # [batch, T, 392]

        lif2_out, _, lif2_rec = self.lif2(conv2_seq, record=True)  # [batch, T, 392]
        if record:
            recordings['lif2'] = lif2_out

        # Pool1: batch pool across all timesteps
        pool1_flat = lif2_out.reshape(batch * T, 8, 7, 7)
        pool1_out = self.pool1(pool1_flat)                # [batch*T, 8, 3, 3]
        pool1_seq = pool1_out.reshape(batch, T, -1)       # [batch, T, 72]

        # --- FC block: FC0 → LIF3 → FC1 → LIF4 ---
        # nn.Linear handles [batch, T, features] natively — applies to last dim
        fc0_seq = self.fc0(pool1_seq)                     # [batch, T, 256]

        lif3_out, _, lif3_rec = self.lif3(fc0_seq, record=True)  # [batch, T, 256]
        if record:
            recordings['lif3'] = lif3_out

        fc1_seq = self.fc1(lif3_out)                      # [batch, T, 11]

        lif4_out, _, lif4_rec = self.lif4(fc1_seq, record=True)  # [batch, T, 11]
        if record:
            recordings['lif4'] = lif4_out

        return lif4_out, recordings

    def train_model(self, train_loader, test_loader, num_epochs=200, print_every=15):
        """Train the model using MSE loss on spike counts vs one-hot targets."""
        print("Starting training")
        self.epochs = num_epochs
        cnt = 0

        for epoch in range(num_epochs):
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                self.train()
                self.optimizer.zero_grad()

                # Forward pass
                output_spikes, recordings = self(data, record=True)

                # Normalize output spike counts by T → spike rates in [0, 1]
                T = output_spikes.shape[1]
                spike_counts = output_spikes.sum(dim=1) / T

                # Soft targets: 0.8 for correct class, 0.2 for others
                one_hot = torch.full((targets.size(0), self.num_classes), 0.2, device=self.device)
                one_hot.scatter_(1, targets.unsqueeze(1), 0.8)
                # **`scatter_(1, indices, 1.0)`** means: along dimension 1 (columns), go to the column specified by each index, and put 1.0 there.


                # MSE loss + spike regularization
                loss = self.loss_fn(spike_counts, one_hot)

                # Spike regularization across all LIF layers
                total_spikes = torch.tensor(0.0, device=self.device)
                for key, spk_tensor in recordings.items():
                    total_spikes = total_spikes + spk_tensor.sum()
                loss = loss + self.spike_lam * total_spikes

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                self.loss_hist.append(loss.item())

                # Calculate accuracy: argmax of summed output spikes
                preds = spike_counts.argmax(dim=1)
                acc = (preds == targets).sum().item() / targets.size(0)
                self.acc_hist.append(acc)

                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iter {batch_idx} | Loss: {loss.item():.2f} | Train Acc: {acc*100:.2f}%")
                    test_acc = self.validate_model(test_loader)
                    self.test_acc_hist.append(test_acc)
                    print(f"Test Acc: {test_acc*100:.2f}% | LR: {self.optimizer.param_groups[0]['lr']:.6f}\n")
                    self.scheduler.step(test_acc)

                cnt += 1

    def validate_model(self, test_loader):
        """Validate the model on test data. Prediction = argmax of summed output spikes."""
        self.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                output_spikes, _ = self(data, record=False)
                T = output_spikes.shape[1]
                spike_counts = output_spikes.sum(dim=1) / T  # Normalized spike rates
                preds = spike_counts.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0

    def count_spikes(self, data):
        """
        Run inference and return total spike count per sample.

        Args:
            data: input tensor [batch, T, C, H, W]

        Returns:
            list of ints, one per sample in batch
        """
        self.eval()
        with torch.no_grad():
            data = data.to(self.device).float()
            _, recordings = self(data, record=True)

            batch_size = data.shape[0]
            counts = []
            for i in range(batch_size):
                sample_total = 0
                for key, spk_tensor in recordings.items():
                    sample_total += spk_tensor[i].sum().item()
                counts.append(int(sample_total))
            return counts

    def estimate_energy(self, data, method='per_timestep'):
        """
        Estimate energy per sample based on Arfa et al. SpiNNaker2 measurements.

        Methods:
            'per_timestep': energy = num_actual_timesteps * 0.765e-3 J
                (num_actual_timesteps = number of non-padding timesteps per sample)
            'per_spike': energy = total_spikes * self.ENERGY_PER_SPIKE

        Returns:
            list of floats (Joules), one per sample
        """
        if method == 'per_timestep':
            # Count non-zero timesteps per sample (non-padding frames)
            batch_size = data.shape[0]
            energies = []
            for i in range(batch_size):
                sample = data[i]  # [T, C, H, W]
                # A padding frame is all zeros
                non_pad = (sample.abs().sum(dim=tuple(range(1, sample.dim()))) > 0).sum().item()
                energies.append(non_pad * self.ENERGY_PER_TIMESTEP)
            return energies
        elif method == 'per_spike':
            if self.ENERGY_PER_SPIKE is None:
                raise ValueError("ENERGY_PER_SPIKE not calibrated. Set model.ENERGY_PER_SPIKE first.")
            spike_counts = self.count_spikes(data)
            return [s * self.ENERGY_PER_SPIKE for s in spike_counts]
        else:
            raise ValueError(f"Unknown energy method: {method}")

    def run_inference(self, data, record=False):
        """
        Framework-agnostic inference entry point for the router.

        Returns:
            logits: [batch, num_classes] — summed output spikes
            spike_count: int total spikes if record=True, else 0
        """
        self.eval()
        with torch.no_grad():
            data = data.to(self.device).float()
            output_spikes, recordings = self(data, record=record)
            logits = output_spikes.sum(dim=1)  # [batch, num_classes]

            spike_count = 0
            if record:
                for key, spk_tensor in recordings.items():
                    spike_count += int(spk_tensor.sum().item())

            return logits, spike_count

    def save_model(self, base_path="./results", counter_file="experiment_counter.txt"):
        """Save model and training graphs."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os

        os.makedirs(f"{base_path}/dvsgesture/{self.model_type}/models", exist_ok=True)
        os.makedirs(f"{base_path}/dvsgesture/{self.model_type}/graphs", exist_ok=True)

        # Counter file
        counter_path = f"{base_path}/dvsgesture/{self.model_type}/{counter_file}"
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
        model_path = f"{base_path}/dvsgesture/{self.model_type}/models/Take{num}_T{self.n_frames}_Epochs{self.epochs}.pth"
        graph_path = f"{base_path}/dvsgesture/{self.model_type}/graphs/Take{num}_T{self.n_frames}_Epochs{self.epochs}.png"

        # Save checkpoint — tau_syn intentionally absent
        torch.save({
            'state_dict': self.state_dict(),
            'hyperparams': {
                'input_size': self.input_size,
                'n_frames': self.n_frames,
                'tau_mem': self.tau_mem,
                'spike_lam': self.spike_lam,
                'model_type': self.model_type,
                'num_classes': self.num_classes,
                'dt': self.dt,
                'threshold': self.threshold,
                'has_bias': self.has_bias,
                'architecture': 'conv_arfa',
                'beta': 0.93,
                # NOTE: tau_syn is intentionally absent
            }
        }, model_path)

        plt.savefig(graph_path)
        plt.close(fig)

        print(f"Model saved to: {model_path}")
        print(f"Graph saved to: {graph_path}")

    def load_model(self, model_path):
        """Load model weights from file. Handles both old and new checkpoint formats."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'])
        else:
            self.load_state_dict(checkpoint)
        self.eval()
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
