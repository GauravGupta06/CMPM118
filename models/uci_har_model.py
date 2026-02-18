"""Fully-connected SNN model for UCI HAR dataset using Rockpool."""

import torch
import torch.nn as nn
from rockpool.nn.modules import LIFTorch, LinearTorch, ExpSynTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant


class DropoutModule(torch.nn.Module):
    """Dropout wrapper compatible with Rockpool Sequential."""
    def __init__(self, p=0.2):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=p)
    
    def forward(self, x, record=False):
        return self.dropout(x), {}, {}


class UCIHARSNN:
    """
    Fully-connected SNN for UCI HAR dataset.
    Architecture: 9 → 128 → 64 → 32 → 6
    """

    def __init__(self, input_size=9, n_frames=128, tau_mem=0.1, tau_syn=0.05, spike_lam=0.0,
                 model_type="dense", device=None, num_classes=6, lr=0.001, dt=0.02, threshold=1.0):
        
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.input_size = input_size
        self.n_frames = n_frames
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.spike_lam = spike_lam
        self.model_type = model_type
        self.num_classes = num_classes
        self.lr = lr
        self.dt = dt
        self.threshold = threshold

        self.epochs = None

        # Build network: 9 → 256 → dropout → 6 (single wide LIF layer)
        self.net = Sequential(
            LinearTorch((input_size, 256), has_bias=False),
            LIFTorch(256, tau_mem=Constant(tau_mem), tau_syn=Constant(tau_syn), 
                     threshold=Constant(threshold), bias=Constant(0.0), dt=dt, 
                     has_rec=False, max_spikes_per_dt=1),
            DropoutModule(p=0.2),
            LinearTorch((256, num_classes), has_bias=False),
            ExpSynTorch(num_classes, dt=dt, tau=Constant(tau_syn))       
        ).to(self.device)

        # Training components
        self.optimizer = torch.optim.Adam(
            [p for _, p in self.net.named_parameters()], lr=lr, betas=(0.9, 0.999), weight_decay=1e-4
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )

        # History tracking
        self.loss_hist = []
        self.acc_hist = []
        self.test_acc_hist = []

    def train_model(self, train_loader, test_loader, num_epochs=150, print_every=15):
        """Train the model."""
        print("starting training")
        self.epochs = num_epochs
        cnt = 0

        for epoch in range(num_epochs):
            for batch, (data, targets) in enumerate(train_loader):
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                # Forward pass
                self.net.train()
                self.net.reset_state()
                self.optimizer.zero_grad()
                
                output, _, recording = self.net(data, record=True)
                summed = torch.cumsum(output, dim=1)[:, -1, :]  # integrate over time, take last
                
                # Count spikes from LIF layers only
                spike_count = torch.tensor(0.0, device=self.device)
                for key, value in recording.items():
                    layer_idx = int(key.split('_')[0])
                    if isinstance(self.net[layer_idx], LIFTorch):
                        if isinstance(value, dict):
                            spike_count += value.get('spikes', torch.tensor(0)).sum()
                        else:
                            spike_count += value.sum()

                loss = self.loss_fn(summed, targets) + spike_count * self.spike_lam

                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in self.net.named_parameters()], max_norm=1.0)
                self.optimizer.step()

                self.loss_hist.append(loss.item())  
                acc = ((summed.argmax(1) == targets).sum().item()) / targets.size(0)
                self.acc_hist.append(acc)

                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iter {batch} | Loss: {loss.item():.2f} | Train Acc: {acc*100:.2f}%")
                    test_acc = self.validate_model(test_loader)
                    self.test_acc_hist.append(test_acc)   
                    print(f"Test Acc: {test_acc*100:.2f}% | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
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

                self.net.reset_state()
                output, _, _ = self.net(data)
                summed = torch.cumsum(output, dim=1)[:, -1, :]  # Total spikes per output neuron (0-128 range)
                correct += (summed.argmax(1) == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0

    def convert_for_hardware(self):
        """Swap ExpSynTorch output layer → LIFTorch for Xylo deployment.
        
        Call this after loading trained weights. The Linear layer weights
        carry over — only the output layer changes from smooth to spiking.
        """
        self.net[4] = LIFTorch(
            self.num_classes,
            tau_mem=Constant(self.tau_mem),
            tau_syn=Constant(self.tau_syn),
            threshold=Constant(self.threshold),
            bias=Constant(0.0),
            dt=self.dt,
            has_rec=False,
            max_spikes_per_dt=1,
        ).to(self.device)
        print("Converted output layer: ExpSynTorch → LIFTorch (hardware-compatible)")

    def get_avg_spike_count(self, dataloader, max_batches=10):
        """Calculate average spikes per neuron per timestep across all LIF layers.

        Args:
            dataloader: DataLoader to evaluate on
            max_batches: Maximum number of batches to evaluate (for speed)

        Returns:
            float: Average spikes per neuron per timestep
        """
        self.net.eval()
        total_spikes = 0
        total_neuron_timesteps = 0

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                data = data.to(self.device).float()
                output, _, recording = self.net(data, record=True)

                # Count spikes from all LIF layers
                for key, value in recording.items():
                    layer_idx = int(key.split('_')[0])
                    if isinstance(self.net[layer_idx], LIFTorch):
                        if isinstance(value, dict):
                            spikes = value.get('spikes', torch.tensor(0))
                        else:
                            spikes = value

                        # spikes shape: [batch, time, neurons]
                        batch_size = spikes.shape[0]
                        timesteps = spikes.shape[1]
                        neurons = spikes.shape[2]

                        total_spikes += spikes.sum().item()
                        total_neuron_timesteps += batch_size * timesteps * neurons

        return total_spikes / total_neuron_timesteps if total_neuron_timesteps > 0 else 0.0

    def get_layer_spike_stats(self, dataloader, max_batches=5):
        """Get detailed spike statistics per layer.

        Returns:
            dict: Per-layer spike statistics (avg, max per timestep)
        """
        self.net.eval()
        layer_stats = {}

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                data = data.to(self.device).float()
                output, _, recording = self.net(data, record=True)

                for key, value in recording.items():
                    layer_idx = int(key.split('_')[0])
                    if isinstance(self.net[layer_idx], LIFTorch):
                        if isinstance(value, dict):
                            spikes = value.get('spikes', torch.tensor(0))
                        else:
                            spikes = value

                        if key not in layer_stats:
                            layer_stats[key] = {
                                'total_spikes': 0,
                                'max_per_timestep': 0,
                                'count': 0
                            }

                        layer_stats[key]['total_spikes'] += spikes.sum().item()
                        layer_stats[key]['max_per_timestep'] = max(
                            layer_stats[key]['max_per_timestep'],
                            spikes.max().item()
                        )
                        layer_stats[key]['count'] += spikes.numel()

        # Compute averages
        for key in layer_stats:
            if layer_stats[key]['count'] > 0:
                layer_stats[key]['avg'] = layer_stats[key]['total_spikes'] / layer_stats[key]['count']
            else:
                layer_stats[key]['avg'] = 0

        return layer_stats

    def save_model(self, base_path="./results", counter_file="experiment_counter.txt"):
        """Save model and training graphs."""
        import matplotlib.pyplot as plt
        import os

        os.makedirs(f"{base_path}/uci_har/{self.model_type}/models", exist_ok=True)
        os.makedirs(f"{base_path}/uci_har/{self.model_type}/graphs", exist_ok=True)

        # Counter file
        counter_path = f"{base_path}/uci_har/{self.model_type}/{counter_file}"
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
        model_path = f"{base_path}/uci_har/{self.model_type}/models/Take{num}_T{self.n_frames}_Epochs{self.epochs}.pth"
        graph_path = f"{base_path}/uci_har/{self.model_type}/graphs/Take{num}_T{self.n_frames}_Epochs{self.epochs}.png"

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
