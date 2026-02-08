"""Fully-connected SNN model for DVSGesture dataset using Rockpool."""

import torch
import torch.nn as nn
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant


class DVSGestureSNN:
    """
    Fully-connected SNN for DVSGesture dataset.
    Architecture: input_size → 256 → 128 → num_classes
    """

    def __init__(self, input_size, n_frames, tau_mem=0.02, tau_syn=0.02, spike_lam=1e-7,
                 model_type="dense", device=None, num_classes=11, lr=0.001, dt=0.001, threshold=1.0, has_bias=True):
        
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
        self.has_bias = has_bias
        self.epochs = None

        # Build network
        self.net = Sequential(
            LinearTorch((input_size, 256), has_bias=has_bias),
            LIFTorch(256, tau_mem=Constant(tau_mem), tau_syn=Constant(tau_syn), threshold=Constant(threshold), dt=dt),
            LinearTorch((256, 128), has_bias=has_bias),
            LIFTorch(128, tau_mem=Constant(tau_mem), tau_syn=Constant(tau_syn), threshold=Constant(threshold), dt=dt),
            LinearTorch((128, num_classes), has_bias=has_bias),
            LIFTorch(num_classes, tau_mem=Constant(tau_mem), tau_syn=Constant(tau_syn), threshold=Constant(threshold), dt=dt),
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
                
                # Count spikes from LIF layers only
                spike_count = torch.tensor(0.0, device=self.device)
                for key, value in recording.items():
                    layer_idx = int(key.split('_')[0])
                    if isinstance(self.net[layer_idx], LIFTorch):
                        if isinstance(value, dict):
                            spike_count += value.get('spk', torch.tensor(0)).sum()
                        else:
                            spike_count += value.sum()

                loss = self.loss_fn(logits, targets) + spike_count * self.spike_lam

                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in self.net.named_parameters()], max_norm=1.0)
                self.optimizer.step()

                self.loss_hist.append(loss.item())

                # Calculate accuracy
                acc = ((logits.argmax(1) == targets).sum().item()) / targets.size(0)
                self.acc_hist.append(acc)

                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iter {batch} | Loss: {loss.item():.2f} | Train Acc: {acc*100:.2f}%")
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

    def save_model(self, base_path="./results", counter_file="experiment_counter.txt"):
        """Save model and training graphs."""
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

        # Save checkpoint
        torch.save({
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

