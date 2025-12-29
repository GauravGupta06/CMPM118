import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from abc import ABC, abstractmethod


class BaseSNNModel(ABC):
    """Base class for SNN models with common functionality."""

    def __init__(self, n_frames, beta, spike_lam, slope=25, model_type="sparse", device=None, num_classes=None):
        self.n_frames = n_frames
        self.beta = beta
        self.slope = slope
        self.model_type = model_type
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.spike_lam = spike_lam
        self.epochs = None
        self.num_classes = num_classes
        
        # Build network
        self.grad = snn.surrogate.fast_sigmoid(slope=self.slope)
        self.net = self._build_network()
        
        # Training components
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
        
        # History tracking
        self.loss_hist = []
        self.acc_hist = []
        self.test_acc_hist = []
    
    @abstractmethod
    def _build_network(self):
        """Build the neural network architecture. Must be implemented by child classes."""
        pass
    
    def forward_pass(self, data):
        """Dataset-agnostic forward pass through the network."""
        utils.reset(self.net)
        spk_rec = []
        spike_count = torch.tensor(0.0, device=self.device)

        for t in range(data.size(0)):
            x = data[t].to(self.device)

            for layer in self.net:
                x = layer(x)
                if isinstance(layer, snn.Leaky):
                    if isinstance(x, tuple):
                        spikes, mem = x
                    else:
                        spikes = x
                    spike_count = spike_count + spikes.sum()
                    x = spikes

            spk_rec.append(x)

        return torch.stack(spk_rec), spike_count

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
                loss = self.loss_fn(spk_rec, targets) + self.spike_regularizer(spike_count, lam=self.spike_lam)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.loss_hist.append(loss.item())
                acc = SF.accuracy_rate(spk_rec, targets)
                self.acc_hist.append(acc)

                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iteration {batch} \nTrain Loss: {loss.item():.2f}")
                    print(f"Train Accuracy: {acc * 100:.2f}%")
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
        for (data, targets) in test_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            spk_rec, _ = self.forward_pass(data)
            correct += SF.accuracy_rate(spk_rec, targets) * data.shape[1]
            total += data.shape[1]
        return correct / total
    
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
        model_name = "Sparse" if self.model_type == "sparse" else "Non_Sparse"
        model_params = self._get_save_params()
        model_save_path = f"{base_path}/{subdir}/models/{model_name}_Take{num}_{model_params}_Epochs{self.epochs}.pth"
        graph_save_path = f"{base_path}/{subdir}/graphs/{model_name}_Take{num}_{model_params}_Epochs{self.epochs}.png"
        pvc_save_path = f"/workspace/models/{model_name}_Take{num}_{model_params}_Epochs{self.epochs}.pth"

        torch.save(self.net.state_dict(), model_save_path)
        torch.save(self.net.state_dict(), pvc_save_path)

        
        plt.savefig(graph_save_path)
        plt.show()
        
        print(f"Model saved to: {model_save_path}")
        print(f"Graph saved to: {graph_save_path}")
    
    @abstractmethod
    def _get_save_params(self):
        """Get dataset-specific parameters for save filename."""
        pass
    
    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.net.eval()
        print(f"Model loaded from: {model_path}")

    def predict_sample(self, frames):
        """Predict with spike counting."""
        frames = frames.detach().clone().float()
        with torch.no_grad():
            spk_rec, spike_count = self.forward_pass(frames.unsqueeze(1))
            counts = spk_rec.sum(0)
            return counts.argmax(1).item(), spike_count.item()












class DVSGestureSNN(BaseSNNModel):
    """SNN model for DVS Gesture dataset."""

    def __init__(self, w, h, n_frames, beta, spike_lam, slope=25, model_type="sparse", device=None, num_classes=11):
        self.w = w
        self.h = h
        super().__init__(n_frames, beta, spike_lam, slope, model_type, device, num_classes=num_classes)

    def _build_network(self):
        test_input = torch.zeros((1, 2, self.w, self.h))
        test_input = test_input.to(self.device)
        x = nn.Conv2d(2, 12, 5).to(self.device)(test_input)
        x = nn.MaxPool2d(2)(x)
        x = nn.Conv2d(12, 32, 5).to(self.device)(x)
        x = nn.MaxPool2d(2)(x)

        net = nn.Sequential(
            nn.Conv2d(2, 12, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True),
            nn.Conv2d(12, 32, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(x.numel(), self.num_classes),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True, output=True)
        ).to(self.device)

        return net

    def _get_save_params(self):
        return f"{self.w}x{self.h}_T{self.n_frames}_B{self.beta}_SpkLam{self.spike_lam}"













class DVSGestureSNN_FC(BaseSNNModel):
    """
    Fully-connected SNN for DVSGesture, optimized for Xylo deployment.
    Uses smaller hidden layers (128→64) to reduce overfitting.
    """

    def __init__(self, w, h, n_frames, beta, spike_lam, slope=25, 
                 model_type="sparse", device=None, num_classes=11):
        self.w = w
        self.h = h
        self.input_size = w * h * 2  # 32*32*2 = 2048
        super().__init__(n_frames, beta, spike_lam, slope, model_type, device, num_classes)

    def _build_network(self):
        """
        Optimized FC architecture: 2048 → 128 → 64 → 11
        Total params: ~270k (less overfitting risk)
        """
        net = nn.Sequential(
            nn.Flatten(start_dim=1),  # [Batch, 2, 32, 32] → [Batch, 2048]
            nn.Linear(self.input_size, 128),  # 2048 → 128
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True),
            nn.Linear(128, 64),  # 128 → 64
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True),
            nn.Linear(64, self.num_classes),  # 64 → 11
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True, output=True)
        ).to(self.device)
        return net

    def _get_save_params(self):
        return f"{self.w}x{self.h}_T{self.n_frames}_FC"

    def to_xylo_compatible(self):
        """
        Convert trained FC model to Xylo format.
        Returns XyloSim object with energy tracking.
        """
        import numpy as np
        from rockpool.devices.xylo import XyloSim
        
        # Extract linear layers
        linear_layers = [layer for layer in self.net if isinstance(layer, nn.Linear)]
        
        if len(linear_layers) != 3:
            raise ValueError(f"Expected 3 Linear layers, found {len(linear_layers)}")
        
        # Extract weights (transpose for Xylo: [in, out])
        w_in = linear_layers[0].weight.data.T.cpu().numpy()   # [2048, 128]
        w_rec = linear_layers[1].weight.data.T.cpu().numpy()  # [128, 64]
        w_out = linear_layers[2].weight.data.T.cpu().numpy()  # [64, 11]
        
        # Quantize to 8-bit integers
        def quantize(w):
            """Quantize float weights to 8-bit range [-128, 127]"""
            w_min, w_max = w.min(), w.max()
            scale = 127.0 / max(abs(w_min), abs(w_max))
            quantized = np.round(w * scale).clip(-128, 127).astype(np.int8)
            return quantized, scale
        
        w_in_q, scale_in = quantize(w_in)
        w_rec_q, scale_rec = quantize(w_rec)
        w_out_q, scale_out = quantize(w_out)
        
        # Convert beta to tau (membrane time constant)
        dt = 0.001  # 1ms timestep
        tau_mem = -dt / np.log(self.beta) if 0 < self.beta < 1 else 0.02
        
        # Create Xylo configuration
        config = {
            'weights_in': w_in_q.astype(float) / scale_in,
            'weights_rec': w_rec_q.astype(float) / scale_rec,
            'weights_out': w_out_q.astype(float) / scale_out,
            'dash_mem': np.ones(128 + 64) * tau_mem,  # Hidden neurons (128+64)
            'dash_mem_out': np.ones(self.num_classes) * tau_mem,  # Output neurons
            'threshold': np.ones(128 + 64) * 1.0,
            'threshold_out': np.ones(self.num_classes) * 1.0,
            'weight_shift_in': 0,
            'weight_shift_rec': 0,
            'weight_shift_out': 0,
        }
        
        # Create XyloSim instance
        xylo_model = XyloSim.from_config(**config)
        
        metadata = {
            'quantization_scales': {
                'input': float(scale_in),
                'recurrent': float(scale_rec),
                'output': float(scale_out)
            },
            'architecture': f"{self.input_size}→128→64→{self.num_classes}",
            'original_beta': float(self.beta),
            'tau_mem': float(tau_mem),
            'total_params': int(self.input_size * 128 + 128 * 64 + 64 * self.num_classes)
        }
        
        return xylo_model, metadata













class SHDSNN(BaseSNNModel):


    """SNN model for Spiking Heidelberg Digits (SHD) dataset."""

    def __init__(self, freq_bins, n_frames, beta, spike_lam, slope=25, model_type="sparse", device=None, num_classes=20):
        self.freq_bins = freq_bins
        super().__init__(n_frames, beta, spike_lam, slope, model_type, device, num_classes=num_classes)

    def _build_network(self):
        """Build a 1D convolutional network tailored to SHD's 700 frequency bins."""
        test_input = torch.zeros((1, 2, 1, self.freq_bins), device=self.device)

        # Determine flattened dimension after the temporal convolutions/pooling.
        with torch.no_grad():
            x = torch.flatten(test_input, start_dim=2)
            x = nn.Conv1d(2, 32, kernel_size=5, padding=2).to(self.device)(x)
            x = nn.MaxPool1d(kernel_size=2)(x)
            x = nn.Conv1d(32, 64, kernel_size=5, padding=2).to(self.device)(x)
            x = nn.MaxPool1d(kernel_size=2)(x)
            x = nn.Conv1d(64, 64, kernel_size=3, padding=1).to(self.device)(x)
            x = torch.flatten(x, start_dim=1)
            linear_in_features = x.shape[-1]

        net = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(linear_in_features, self.num_classes),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True, output=True),
        ).to(self.device)

        return net

    def _get_save_params(self):
        return f"Freq{self.freq_bins}_T{self.n_frames}_B{self.beta}_SpkLam{self.spike_lam}"