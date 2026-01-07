import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential


class BaseSNNModel(ABC):
    """Base class for Rockpool SNN models with common functionality."""

    def __init__(self, n_frames, tau_mem, spike_lam, model_type="sparse", device=None, num_classes=None):
        """
        Args:
            n_frames: Number of time steps
            tau_mem: Membrane time constant (in seconds, e.g., 0.02 for 20ms)
            spike_lam: Spike regularization coefficient
            model_type: "sparse" or "dense" (for saving purposes)
            device: torch device
            num_classes: Number of output classes
        """
        self.n_frames = n_frames
        self.tau_mem = tau_mem
        self.model_type = model_type
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.spike_lam = spike_lam
        self.epochs = None
        self.num_classes = num_classes
        self.dt = 0.001  # 1ms timestep for Xylo compatibility

        # Build network
        self.net = self._build_network()

        # Training components
        # Note: Rockpool's .parameters() returns module names, we need named_parameters()
        torch_params = [p for name, p in self.net.named_parameters()]
        self.optimizer = torch.optim.Adam(torch_params, lr=0.0001, betas=(0.9, 0.999))
        self.loss_fn = nn.CrossEntropyLoss()

        # History tracking
        self.loss_hist = []
        self.acc_hist = []
        self.test_acc_hist = []

    @abstractmethod
    def _build_network(self):
        """Build the neural network architecture. Must be implemented by child classes."""
        pass

    def forward_pass(self, data):
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

    def _prepare_input(self, data):
        """
        Prepare input data for Rockpool format.
        Override in child classes if needed.
        """
        # Default: flatten spatial dimensions and transpose to [B, T, features]
        T, B = data.size(0), data.size(1)
        x = data.transpose(0, 1)  # [B, T, ...]
        x = x.flatten(2)  # [B, T, features]
        return x

    def _count_spikes(self, output):
        """Count total spikes from output."""
        # For Xylo-compatible models, count output spikes
        return output.sum()

    def train_model(self, train_loader, test_loader, num_epochs=150, print_every=15):
        """Train the model using dataset-agnostic training loop."""
        self.epochs = num_epochs
        cnt = 0

        for epoch in range(num_epochs):
            for batch, (data, targets) in enumerate(iter(train_loader)):
                data = data.to(self.device)
                targets = targets.to(self.device)

                self.net.train()

                spk_rec, spike_count = self.forward_pass(data)

                # Use spike counts over time for classification
                # Sum spikes over time: [T, B, num_classes] -> [B, num_classes]
                spike_counts = spk_rec.sum(0)

                loss = self.loss_fn(spike_counts, targets) + self.spike_regularizer(spike_count, lam=self.spike_lam)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.loss_hist.append(loss.item())

                # Calculate accuracy
                predicted = spike_counts.argmax(1)
                acc = (predicted == targets).float().mean()
                self.acc_hist.append(acc.item())

                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iteration {batch} \nTrain Loss: {loss.item():.2f}")
                    print(f"Train Accuracy: {acc.item() * 100:.2f}%")
                    test_acc = self.validate_model(test_loader)
                    self.test_acc_hist.append(test_acc)
                    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

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
                data, targets = data.to(self.device), targets.to(self.device)
                spk_rec, _ = self.forward_pass(data)

                # Sum spikes over time for classification
                spike_counts = spk_rec.sum(0)
                predicted = spike_counts.argmax(1)

                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0

    def save_model(self, base_path="results", counter_file="experiment_counter.txt"):
        """
        Args:
            base_path: Base directory path (should contain 'small' or 'large' subdirs)
            counter_file: Name of the counter file
        """
        import matplotlib.pyplot as plt

        # Determine subdirectory based on model type
        subdir = "small" if self.model_type == "sparse" else "large"

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

        # Save Rockpool model state
        torch.save(self.net.state_dict(), model_save_path)

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
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()
        print(f"Model loaded from: {model_path}")

    def predict_sample(self, frames):
        """Predict with spike counting."""
        frames = frames.detach().clone().float()
        with torch.no_grad():
            spk_rec, spike_count = self.forward_pass(frames.unsqueeze(1))
            counts = spk_rec.sum(0)
            return counts.argmax(1).item(), spike_count.item()


class DVSGestureSNN_FC(BaseSNNModel):
    """
    Fully-connected SNN for DVSGesture using Rockpool, optimized for Xylo deployment.
    Architecture: 2048 → 128 → 64 → 11
    """

    def __init__(self, w, h, n_frames, tau_mem=0.02, spike_lam=1e-7,
                 model_type="dense", device=None, num_classes=11):
        """
        Args:
            w: Width of input frames
            h: Height of input frames
            n_frames: Number of time steps
            tau_mem: Membrane time constant (seconds)
            spike_lam: Spike regularization coefficient
            model_type: "sparse" or "dense"
            device: torch device
            num_classes: Number of output classes
        """
        self.w = w
        self.h = h
        self.input_size = w * h * 2  # 32*32*2 = 2048
        super().__init__(n_frames, tau_mem, spike_lam, model_type, device, num_classes)

    def _build_network(self):
        """
        Build Rockpool FC architecture: 2048 → 128 → 64 → 11
        Uses LIFTorch for PyTorch compatibility during training
        """
        # Create sequential network
        net = Sequential(
            LinearTorch((self.input_size, 128), has_bias=True),
            LIFTorch(128, tau_mem=self.tau_mem, dt=self.dt),
            LinearTorch((128, 64), has_bias=True),
            LIFTorch(64, tau_mem=self.tau_mem, dt=self.dt),
            LinearTorch((64, self.num_classes), has_bias=True),
            LIFTorch(self.num_classes, tau_mem=self.tau_mem, dt=self.dt),
        )

        # Move to device
        return net.to(self.device)

    def _prepare_input(self, data):
        """
        Prepare DVSGesture input for Rockpool.
        Input: [T, B, C, H, W] from tonic
        Output: [B, T, features] for Rockpool
        """
        T, B = data.size(0), data.size(1)
        x = data.transpose(0, 1)  # [B, T, C, H, W]
        x = x.flatten(2)  # [B, T, C*H*W] = [B, T, 2048]
        return x

    def _get_save_params(self):
        return f"{self.w}x{self.h}_T{self.n_frames}_FC_Rockpool"

    def to_xylo_compatible(self):
        """
        Convert trained Rockpool FC model to Xylo format.
        Returns XyloSim object with energy tracking.
        Uses Xylo-Audio 3 (syns63300) chip.
        """
        from rockpool.devices.xylo.syns63300 import XyloSim, config_from_specification

        # Extract weights from Rockpool LinearTorch modules
        # Get all modules from the Sequential container
        linear_modules = []
        for name, module in self.net.named_modules():
            if isinstance(module, LinearTorch):
                linear_modules.append(module)

        if len(linear_modules) != 3:
            raise ValueError(f"Expected 3 Linear layers, found {len(linear_modules)}")

        # Extract weights (Rockpool stores as [in, out])
        w_in_layer1 = linear_modules[0].weight.data.cpu().numpy()   # [2048, 128]
        w_layer1_to_layer2 = linear_modules[1].weight.data.cpu().numpy()  # [128, 64]
        w_out_from_layer2 = linear_modules[2].weight.data.cpu().numpy()  # [64, 11]

        # Combine into Xylo format:
        # - Input weights: only connect to first layer (128 neurons)
        #   Shape: [2048, 192] with zeros for neurons 128-192
        Nhidden = 128 + 64  # Total hidden neurons = 192
        w_in = np.zeros((2048, Nhidden))
        w_in[:, :128] = w_in_layer1  # Only first 128 neurons receive input

        # - Recurrent weights: layer1 (0-127) → layer2 (128-191)
        #   Shape: [192, 192] block structure
        w_rec = np.zeros((Nhidden, Nhidden))
        w_rec[:128, 128:] = w_layer1_to_layer2  # Layer 1 → Layer 2 [128, 64]
        # No recurrent connections within layers (feedforward only)

        # - Output weights: only read from second layer (neurons 128-191)
        #   Shape: [192, 11] with zeros for neurons 0-127
        w_out = np.zeros((Nhidden, 11))
        w_out[128:, :] = w_out_from_layer2  # Only layer 2 connects to output

        # Quantize to 8-bit integers
        def quantize(w):
            """Quantize float weights to 8-bit range [-128, 127]"""
            w_min, w_max = w.min(), w.max()
            if w_max == w_min:  # Handle all-zero case
                return np.zeros_like(w, dtype=np.int8), 1.0
            scale = 127.0 / max(abs(w_min), abs(w_max))
            quantized = np.round(w * scale).clip(-128, 127).astype(np.int8)
            return quantized, scale

        w_in_q, scale_in = quantize(w_in)
        w_rec_q, scale_rec = quantize(w_rec)
        w_out_q, scale_out = quantize(w_out)

        # De-quantize for config (Xylo will re-quantize internally)
        w_in_dequant = w_in_q.astype(float) / scale_in
        w_rec_dequant = w_rec_q.astype(float) / scale_rec
        w_out_dequant = w_out_q.astype(float) / scale_out

        # Create Xylo configuration using config_from_specification
        # Returns (config, is_valid, error_message)
        config_result = config_from_specification(
            weights_in=w_in_dequant,
            weights_rec=w_rec_dequant,
            weights_out=w_out_dequant,
            dash_mem=np.ones(Nhidden) * self.tau_mem,  # 192 hidden neurons
            dash_mem_out=np.ones(self.num_classes) * self.tau_mem,
            threshold=np.ones(Nhidden) * 1.0,
            threshold_out=np.ones(self.num_classes) * 1.0,
            weight_shift_in=0,
            weight_shift_rec=0,
            weight_shift_out=0,
        )

        # Unpack config result
        config, is_valid, error_msg = config_result
        if not is_valid:
            raise ValueError(f"Invalid Xylo configuration: {error_msg}")

        # Create XyloSim instance
        xylo_model = XyloSim.from_config(config, dt=self.dt)

        metadata = {
            'quantization_scales': {
                'input': float(scale_in),
                'recurrent': float(scale_rec),
                'output': float(scale_out)
            },
            'architecture': f"{self.input_size}→128→64→{self.num_classes}",
            'tau_mem': float(self.tau_mem),
            'dt': float(self.dt),
            'total_params': int(self.input_size * 128 + 128 * 64 + 64 * self.num_classes)
        }

        return xylo_model, metadata


class SHDSNN(BaseSNNModel):
    """
    SNN model for Spiking Heidelberg Digits (SHD) dataset using Rockpool.
    Uses 1D convolutions for frequency bin processing.
    """

    def __init__(self, freq_bins, n_frames, tau_mem=0.02, spike_lam=1e-7,
                 model_type="sparse", device=None, num_classes=20):
        """
        Args:
            freq_bins: Number of frequency bins (700 for SHD)
            n_frames: Number of time steps
            tau_mem: Membrane time constant (seconds)
            spike_lam: Spike regularization coefficient
            model_type: "sparse" or "dense"
            device: torch device
            num_classes: Number of output classes (20 for SHD)
        """
        self.freq_bins = freq_bins
        super().__init__(n_frames, tau_mem, spike_lam, model_type, device, num_classes=num_classes)

    def _build_network(self):
        """
        Build a 1D convolutional network for SHD's 700 frequency bins.
        Note: This architecture is NOT Xylo-compatible (uses convolutions).
        """
        # Determine flattened dimension after convolutions/pooling
        with torch.no_grad():
            test_input = torch.zeros((1, self.n_frames, 2 * self.freq_bins), device=self.device)
            x = test_input.permute(0, 2, 1)  # [B, features, T]
            x = nn.Conv1d(2 * self.freq_bins, 32, kernel_size=5, padding=2).to(self.device)(x)
            x = nn.MaxPool1d(kernel_size=2)(x)
            x = nn.Conv1d(32, 64, kernel_size=5, padding=2).to(self.device)(x)
            x = nn.MaxPool1d(kernel_size=2)(x)
            x = nn.Conv1d(64, 64, kernel_size=3, padding=1).to(self.device)(x)
            x = x.permute(0, 2, 1)  # Back to [B, T, features]
            linear_in_features = x.shape[-1]

        # Build network with PyTorch modules for conv layers
        # Note: This is a hybrid approach - not pure Rockpool
        # For Xylo deployment, you'd need a fully-connected version
        class SHDNet(nn.Module):
            def __init__(self, freq_bins, linear_in_features, num_classes, tau_mem, dt, device):
                super().__init__()
                self.freq_bins = freq_bins
                self.device = device

                # Convolutional layers (PyTorch)
                self.conv1 = nn.Conv1d(2 * freq_bins, 32, kernel_size=5, padding=2)
                self.pool1 = nn.MaxPool1d(kernel_size=2)
                self.lif1 = LIFTorch(32, tau_mem=tau_mem, dt=dt)

                self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
                self.pool2 = nn.MaxPool1d(kernel_size=2)
                self.lif2 = LIFTorch(64, tau_mem=tau_mem, dt=dt)

                self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
                self.lif3 = LIFTorch(64, tau_mem=tau_mem, dt=dt)

                # Fully-connected output
                self.fc = LinearTorch((linear_in_features, num_classes), has_bias=True)
                self.lif_out = LIFTorch(num_classes, tau_mem=tau_mem, dt=dt)

            def forward(self, x):
                # x: [B, T, features]
                B, T, F = x.shape

                # Process through conv layers (need to handle time dimension)
                outputs = []
                for t in range(T):
                    xt = x[:, t, :].unsqueeze(1)  # [B, 1, features]
                    xt = xt.permute(0, 2, 1)  # [B, features, 1]

                    xt = self.pool1(torch.relu(self.conv1(xt)))
                    xt = self.pool2(torch.relu(self.conv2(xt)))
                    xt = torch.relu(self.conv3(xt))

                    xt = xt.permute(0, 2, 1)  # [B, T_reduced, features]
                    outputs.append(xt)

                x = torch.cat(outputs, dim=1)  # Combine time steps

                # Fully-connected output
                x, state = self.fc(x)
                x, state = self.lif_out(x)

                return x, {}

        net = SHDNet(self.freq_bins, linear_in_features, self.num_classes,
                     self.tau_mem, self.dt, self.device)
        return net.to(self.device)

    def _prepare_input(self, data):
        """
        Prepare SHD input for network.
        Input: [T, B, C, 1, freq_bins] from tonic
        Output: [B, T, 2*freq_bins] for network
        """
        T, B = data.size(0), data.size(1)
        x = data.transpose(0, 1)  # [B, T, C, 1, freq_bins]
        x = x.squeeze(3)  # [B, T, C, freq_bins]
        x = x.flatten(2)  # [B, T, C*freq_bins]
        return x

    def _get_save_params(self):
        return f"Freq{self.freq_bins}_T{self.n_frames}_Rockpool"
