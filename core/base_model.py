"""Base SNN model class for Rockpool-based implementations.

COPY-PASTE DROP-IN REPLACEMENT.

Key fix:
- Always route batch data through `_prepare_input()` before calling `self.net(...)`.
This makes SHD (which needs flatten/binarize) and UCI HAR (already [B,T,C]) work
without changing the rest of your pipeline.

Assumption:
- Your DataLoader returns BATCH-FIRST tensors: [B, T, ...]
  (This matches your SHD training script using PadTensors(batch_first=True).)
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential


class BaseSNNModel(ABC):
    """Base class for Rockpool SNN models with common functionality."""

    def __init__(
        self,
        n_frames,
        tau_mem,
        spike_lam,
        model_type="sparse",
        device=None,
        num_classes=None,
        lr=0.001,
        dt=0.001,
        threshold=1.0,
        has_bias=True,
    ):
        """
        Args:
            n_frames: Number of time steps
            tau_mem: Membrane time constant (seconds)
            spike_lam: Spike regularization coefficient
            model_type: "sparse" or "dense" (for saving purposes)
            device: torch device
            num_classes: Number of output classes
            lr: Learning rate
            dt: simulation timestep
            threshold: neuron threshold
            has_bias: linear layer bias
        """
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.n_frames = n_frames
        self.tau_mem = tau_mem
        self.model_type = model_type
        self.spike_lam = spike_lam
        self.epochs = None
        self.num_classes = num_classes
        self.dt = dt
        self.lr = lr
        self.threshold = threshold
        self.has_bias = has_bias

        # Build network
        self.net = self._build_network()

        # Training components
        torch_params = [p for _, p in self.net.named_parameters()]
        self.optimizer = torch.optim.Adam(torch_params, lr=self.lr, betas=(0.9, 0.999))
        self.loss_fn = nn.CrossEntropyLoss()

        # LR scheduler (based on test accuracy)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )


        # History tracking
        self.loss_hist = []
        self.acc_hist = []
        self.test_acc_hist = []

    # -----------------------------
    # Required overrides
    # -----------------------------
    @abstractmethod
    def _build_network(self):
        """Build the neural network architecture. Must be implemented by child classes."""
        pass

    @abstractmethod
    def _get_save_params(self):
        """Get dataset-specific parameters for save filename."""
        pass

    # -----------------------------
    # Input preparation
    # -----------------------------
    def _prepare_input(self, data: torch.Tensor) -> torch.Tensor:
        """
        Prepare input data for Rockpool format.

        DEFAULT behavior:
        - expects batch-first input: [B, T, ...]
        - flattens everything after time into features -> [B, T, F]

        Child classes SHOULD override this when they need custom shaping, e.g.:
        - SHD: [B,T,C,1,freq] -> squeeze -> flatten -> binarize
        - HAR: [B,T,C] -> (optionally normalize/clamp) -> return
        """
        # data: [B, T, ...]
        if data.dim() < 3:
            raise ValueError(f"Expected data with at least 3 dims [B,T,...], got shape {tuple(data.shape)}")

        x = data.flatten(2)  # [B, T, F]
        return x

    # -----------------------------
    # Forward helper (optional)
    # -----------------------------
    def forward_pass(self, data: torch.Tensor):
        """
        Dataset-agnostic forward pass through the network.

        Args:
            data: input batch, batch-first [B, T, ...] (recommended)

        Returns:
            spk_rec: output activity [T, B, num_classes]
            spike_count: total spike/activity count (by default sums output)
        """
        x = self._prepare_input(data).to(self.device).float()  # [B, T, F]
        output, state_dict, recording_dict = self.net(x)       # output: [B, T, num_classes]

        spike_count = self._count_spikes(output)
        spk_rec = output.transpose(0, 1)  # [T, B, num_classes]
        return spk_rec, spike_count

    def _count_spikes(self, output: torch.Tensor) -> torch.Tensor:
        """Count total spikes/activity from output."""
        return output.sum()

    # -----------------------------
    # Training / evaluation
    # -----------------------------
    def train_model(self, train_loader, test_loader, num_epochs=150, print_every=15):
        """Train the model using a dataset-agnostic training loop."""
        print("starting training")
        self.epochs = num_epochs
        cnt = 0

        for epoch in range(num_epochs):
            for batch, (data, targets) in enumerate(iter(train_loader)):
                # data: [B, T, ...] (batch-first)
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                self.net.train()
                self.optimizer.zero_grad()

                # IMPORTANT FIX: always prepare input for Rockpool
                x = self._prepare_input(data)        # [B, T, F]
                output, _, _ = self.net(x)

                # Rockpool may return either [B, T, C] or [T, B, C] depending on module versions.
                # Make it batch-first: [B, T, C]
                if output.dim() == 3 and output.shape[0] != targets.shape[0] and output.shape[1] == targets.shape[0]:
                    output = output.transpose(0, 1)



                # Stable readout: mean over time
                logits = output.mean(dim=1)          # [B, num_classes]
                loss = self.loss_fn(logits, targets)

                loss.backward()

                # Gradient clipping (helps with recurrence)
                torch.nn.utils.clip_grad_norm_([p for _, p in self.net.named_parameters()], max_norm=1.0)

                self.optimizer.step()

                self.loss_hist.append(loss.item())

                # Train accuracy
                acc = ((logits.argmax(1) == targets).sum().item()) / targets.size(0)
                self.acc_hist.append(acc)

                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iteration {batch} \nTrain Loss: {loss.item():.2f}")
                    print(f"Train Accuracy: {acc * 100:.2f}%")

                    test_acc = self.validate_model(test_loader)
                    self.test_acc_hist.append(test_acc)
                    print(f"Test Accuracy: {test_acc * 100:.2f}%")

                    # Step scheduler on test accuracy
                    self.scheduler.step(test_acc)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Current LR: {current_lr:.6f}\n")

                cnt += 1

    def validate_model(self, test_loader):
        """Validate the model on test data and return accuracy."""
        self.net.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for (data, targets) in test_loader:
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                # IMPORTANT FIX: always prepare input for Rockpool
                x = self._prepare_input(data)      # [B, T, F]
                output, _, _ = self.net(x)

                # Rockpool may return either [B, T, C] or [T, B, C] depending on module versions.
                # Make it batch-first: [B, T, C]
                if output.dim() == 3 and output.shape[0] != targets.shape[0] and output.shape[1] == targets.shape[0]:
                    output = output.transpose(0, 1)

                logits = output.mean(dim=1)        # [B, num_classes]
                correct += (logits.argmax(1) == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0

    # -----------------------------
    # Saving / loading / prediction
    # -----------------------------
    def save_model(self, base_path="results", counter_file="experiment_counter.txt"):
        """
        Save model weights and training graphs.

        Args:
            base_path: Base directory path (should contain 'small' or 'large' subdirs)
            counter_file: Name of the counter file
        """
        import matplotlib.pyplot as plt
        import os

        subdir = "small" if self.model_type == "sparse" else "large"

        models_dir = f"{base_path}/{subdir}/models"
        graphs_dir = f"{base_path}/{subdir}/graphs"
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)

        experiment_counter_file_path = f"{base_path}/{subdir}/{counter_file}"
        if not os.path.exists(experiment_counter_file_path):
            os.makedirs(os.path.dirname(experiment_counter_file_path), exist_ok=True)
            with open(experiment_counter_file_path, "w") as f:
                f.write("0")

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

        with open(experiment_counter_file_path, "r") as f:
            num_str = f.read().strip()
            num = int(num_str)

        num += 1
        with open(experiment_counter_file_path, "w") as f:
            f.write(str(num))

        model_name = "Rockpool_Sparse" if self.model_type == "sparse" else "Rockpool_Non_Sparse"
        model_params = self._get_save_params()

        model_save_path = f"{base_path}/{subdir}/models/{model_name}_Take{num}_{model_params}_Epochs{self.epochs}.pth"
        graph_save_path = f"{base_path}/{subdir}/graphs/{model_name}_Take{num}_{model_params}_Epochs{self.epochs}.png"

        torch.save(self.net.state_dict(), model_save_path)

        plt.savefig(graph_save_path)
        plt.show()

        print(f"Model saved to: {model_save_path}")
        print(f"Graph saved to: {graph_save_path}")

    def load_model(self, model_path):
        """Load model weights from file."""
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()
        print(f"Model loaded from: {model_path}")

    def predict_sample(self, frames: torch.Tensor):
        """
        Predict a single sample.
        Expects frames as either:
          - [T, ...] (no batch) OR
          - [B, T, ...] (already batched)

        Returns:
            pred_class (int), spike_count (float)
        """
        frames = frames.detach().clone().float()

        # If user passes [T,...], add batch dim -> [1, T, ...]
        if frames.dim() >= 2 and frames.shape[0] == self.n_frames:
            frames = frames.unsqueeze(0)

        with torch.no_grad():
            spk_rec, spike_count = self.forward_pass(frames)
            counts = spk_rec.sum(0)  # [B, num_classes]
            return counts.argmax(1).item(), float(spike_count.item())
