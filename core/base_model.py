"""Base SNN model class for Rockpool-based implementations."""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential


class BaseSNNModel(ABC):
    """Base class for Rockpool SNN models with common functionality."""

    def __init__(self, n_frames, tau_mem, spike_lam, model_type="sparse", device=None, num_classes=None, lr=0.001, dt=0.001, threshold=1.0, has_bias=True):
        """
        Args:
            n_frames: Number of time steps
            tau_mem: Membrane time constant (in seconds, e.g., 0.02 for 20ms)
            spike_lam: Spike regularization coefficient
            model_type: "sparse" or "dense" (for saving purposes)
            device: torch device
            num_classes: Number of output classes
            lr: Learning rate (default 0.001)
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
        # Note: Rockpool's .parameters() returns module names, we need named_parameters()
        torch_params = [p for name, p in self.net.named_parameters()]
        self.optimizer = torch.optim.Adam(torch_params, lr=self.lr, betas=(0.9, 0.999))
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Learning rate scheduler - reduces lr by factor of 0.5 when test accuracy plateaus
        # patience=10 means wait 10 evaluations before reducing lr (more stable for recurrent nets)

        """
        Kevin:
        Removed `verbose=True` from the below line. I don't know if this
        is something that anyone else needs, but in case it is, I'll leave
        this comment here so you know what changed.

        For me, I couldn't run my code unless I removed it.
        """
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )

        # History tracking
        self.loss_hist = []
        self.acc_hist = []
        self.test_acc_hist = []

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
        # if data.dim() < 3:
        #     raise ValueError(f"Expected data with at least 3 dims [B,T,...], got shape {tuple(data.shape)}")

        # x = data.flatten(2)  # [B, T, F]
        # return x
        pass

    # -----------------------------
    # Forward helper (optional)
    # -----------------------------
    def forward_pass(self, data: torch.Tensor):
        """
        Dataset-agnostic forward pass through the network.

        Args:
            data: Input tensor of shape [T, B, ...] where T is time, B is batch

        Returns:
            spk_rec: Output spikes [T, B, num_classes]
            spike_count: Total number of spikes across all neurons
        """
        # Rockpool expects [B, T, features], but tonic gives us [T, B, C, H, W]
        # We need to reshape based on the specific model
        spike_count = torch.tensor(0.0, device=self.device)

        # Process through network
        # For Rockpool, we need to reshape to [B, T, features]
        T, B = data.size(0), data.size(1)
        x = self._prepare_input(data)  # [B, T, features]

        # Forward through network
        # Rockpool returns (output, state_dict, recording_dict)
        output, state_dict, recording_dict = self.net(x)

        # Count spikes from all LIF layers
        spike_count = self._count_spikes(output)

        # Reshape output back to [T, B, num_classes] for compatibility
        spk_rec = output.transpose(0, 1)  # [T, B, num_classes]

        return spk_rec, spike_count

    # def _prepare_input(self, data):
    #     """
    #     Prepare input data for Rockpool format.
    #     Override in child classes if needed.
    #     """
    #     # Default: flatten spatial dimensions and transpose to [B, T, features]
    #     T, B = data.size(0), data.size(1)
    #     x = data.transpose(0, 1)  # [B, T, ...]
    #     x = x.flatten(2)  # [B, T, features]
    #     return x

    def _count_spikes(self, output):
        """Count total spikes from output."""
        # For Xylo-compatible models, count output spikes
        return output.sum()

    def train_model(self, train_loader, test_loader, num_epochs=150, print_every=15):
        """Train the model using dataset-agnostic training loop."""
        print("starting training")
        self.epochs = num_epochs
        cnt = 0

        for epoch in range(num_epochs):
            for batch, (data, targets) in enumerate(iter(train_loader)):
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                self.net.train()

                self.optimizer.zero_grad()
                output, _, _ = self.net(data)

                # Use MEAN over time instead of cumsum - much more stable for recurrent networks
                # This prevents exploding values regardless of spike cascade magnitude
                logits = output.mean(dim=1)  # [B, num_classes] - average activity per class
                
                loss = self.loss_fn(logits, targets)

                loss.backward()
                
                # Gradient clipping to prevent exploding gradients with recurrence
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in self.net.named_parameters()], max_norm=1.0
                )
                
                self.optimizer.step()

                self.loss_hist.append(loss.item())

                # Calculate accuracy
                acc = ((logits.argmax(1) == targets).sum().item()) / targets.size(0)
                self.acc_hist.append(acc)

                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iteration {batch} \nTrain Loss: {loss.item():.2f}")
                    print(f"Train Accuracy: {acc * 100:.2f}%")
                    test_acc = self.validate_model(test_loader)
                    self.test_acc_hist.append(test_acc)
                    print(f"Test Accuracy: {test_acc * 100:.2f}%")
                    
                    # Step the learning rate scheduler based on test accuracy
                    self.scheduler.step(test_acc)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Current LR: {current_lr:.6f}\n")

                cnt += 1

    def spike_regularizer(self, spike_count, lam):
        """Apply spiking penalty."""
        return lam * spike_count

    def validate_model(self, test_loader):
        """
        Validate the model on test data.

        Args:
            test_loader: DataLoader for test data

        Returns:
            accuracy: Test accuracy
        """
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

    def save_model(self, base_path="./results", counter_file="experiment_counter.txt"):
        """
        Args:
            base_path: Base directory path (should contain 'small' or 'large' subdirs)
            counter_file: Name of the counter file
        """
        import matplotlib.pyplot as plt
        import os

        # Determine subdirectory based on model type
        subdir = "small" if self.model_type == "sparse" else "large"

        # Create necessary directories if they don't exist
        models_dir = f"{base_path}/{subdir}/models"
        graphs_dir = f"{base_path}/{subdir}/graphs"
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)

        # Initialize counter file if it doesn't exist
        experiment_counter_file_path = f"{base_path}/{subdir}/{counter_file}"
        if not os.path.exists(experiment_counter_file_path):
            os.makedirs(os.path.dirname(experiment_counter_file_path), exist_ok=True)
            with open(experiment_counter_file_path, "w") as f:
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
        experiment_counter_file_path = f"{base_path}/{subdir}/{counter_file}"
        with open(experiment_counter_file_path, "r") as f:
            num_str = f.read().strip()
            num = int(num_str)

        num += 1

        with open(experiment_counter_file_path, "w") as f:
            f.write(str(num))

        # Save model and graph
        model_name = "Rockpool_Sparse" if self.model_type == "sparse" else "Rockpool_Non_Sparse"
        model_params = self._get_save_params()
        model_save_path = f"{base_path}/{subdir}/models/{model_name}_Take{num}_{model_params}_Epochs{self.epochs}.pth"
        graph_save_path = f"{base_path}/{subdir}/graphs/{model_name}_Take{num}_{model_params}_Epochs{self.epochs}.png"

        # Save Rockpool model state with hyperparameters
        checkpoint = {
            'state_dict': self.net.state_dict(),
            'hyperparams': {
                'input_size': getattr(self, 'input_size', None),
                'n_frames': self.n_frames,
                'tau_mem': self.tau_mem,
                'tau_syn': getattr(self, 'tau_syn', 0.1),
                'spike_lam': self.spike_lam,
                'model_type': self.model_type,
                'num_classes': self.num_classes,
                'dt': self.dt,
                'threshold': self.threshold,
                'has_bias': self.has_bias,
            }
        }
        torch.save(checkpoint, model_save_path)

        plt.savefig(graph_save_path)
        plt.show()

        print(f"Model saved to: {model_save_path}")
        print(f"Graph saved to: {graph_save_path}")

    @abstractmethod
    def _get_save_params(self):
        """Get dataset-specific parameters for save filename."""
        pass

    def load_model(self, model_path):
        """Load model weights from file."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        # Handle both new format (dict with 'state_dict') and old format (just state_dict)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['state_dict'])
        else:
            self.net.load_state_dict(checkpoint)
        self.net.eval()
        print(f"Model loaded from: {model_path}")

    @staticmethod
    def load_hyperparams(model_path, device='cpu'):
        """Load hyperparameters from checkpoint file without instantiating model."""
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'hyperparams' in checkpoint:
            return checkpoint['hyperparams']
        else:
            raise ValueError(f"Checkpoint at {model_path} does not contain hyperparams. "
                           "Was it saved with an older version?")

    def predict_sample(self, frames):
        """Predict with spike counting."""
        frames = frames.detach().clone().float()
        with torch.no_grad():
            spk_rec, spike_count = self.forward_pass(frames.unsqueeze(1))
            counts = spk_rec.sum(0)
            return counts.argmax(1).item(), spike_count.item()
