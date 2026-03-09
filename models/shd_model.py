"""Fully-connected SNN model for SHD dataset using Rockpool.

Architecture matches the EventProp paper (Nowotny et al. / Balazs et al.):
  - Feedforward: Input -> LIF(hidden) -> LIF(hidden) -> ExpSyn(num_classes)
  - Recurrent:   Input -> LIF(hidden, rec) -> ExpSyn(num_classes)
  - Output neurons are non-firing leaky integrators (ExpSynTorch)
  - Classification: argmax of max-over-time output voltages
  - Firing rate regularisation targeting 14 Hz on hidden neurons
  - Dead neuron bump: +0.002 to input weights of neurons that fired 0 spikes in an epoch
"""

import torch
import torch.nn as nn
from rockpool.nn.modules import LIFTorch, LinearTorch, ExpSynTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant


class SHDSNN:
    """
    SNN for SHD matching the EventProp paper architecture.

    arch='feedforward': Input -> LIF(hidden) -> LIF(hidden) -> ExpSyn(num_classes)
    arch='recurrent':   Input -> LIF(hidden, rec) -> ExpSyn(num_classes)

    model_type controls sparsity penalty:
      'dense' / 'baseline': spike_lam=0 (no sparsity penalty)
      'sparse':              spike_lam>0 (encourages fewer spikes)
    """

    def __init__(self, input_size, n_frames, tau_mem=0.02, tau_syn=0.02,
                 arch='feedforward', hidden_size=512,
                 spike_lam=0.0, rate_lam=1e-3, target_rate=14.0,
                 model_type="dense", device=None, num_classes=20,
                 lr=0.001, dt=10e-3, threshold=1.0, has_bias=True):

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.input_size = input_size
        self.n_frames = n_frames
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.arch = arch
        self.hidden_size = hidden_size
        self.spike_lam = spike_lam
        self.rate_lam = rate_lam
        self.target_rate = target_rate   # Hz
        self.model_type = model_type
        self.num_classes = num_classes
        self.lr = lr
        self.dt = dt
        self.threshold = threshold
        self.has_bias = has_bias
        self.epochs = None

        # SHD has 2 polarities, so actual input = input_size * 2
        actual_input = input_size * 2

        lif_kwargs = dict(
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(tau_syn),
            threshold=Constant(threshold),
            bias=Constant(0.01),
            dt=dt,
        )

        if arch == 'recurrent':
            # Single hidden LIF layer with recurrent connections
            self.net = Sequential(
                LinearTorch((actual_input, hidden_size), has_bias=has_bias),
                LIFTorch(hidden_size, has_rec=True, **lif_kwargs),
                LinearTorch((hidden_size, num_classes), has_bias=has_bias),
                ExpSynTorch(num_classes, dt=dt, tau=Constant(tau_syn)),
            ).to(self.device)

            # Scale down recurrent weights to prevent spike cascades
            with torch.no_grad():
                for name, param in self.net.named_parameters():
                    if 'w_rec' in name.lower():
                        param.data *= 0.01
                        print(f"Scaled {name} by 0.01 for stability")

        else:  # feedforward (2-layer, paper default)
            self.net = Sequential(
                LinearTorch((actual_input, hidden_size), has_bias=has_bias),
                LIFTorch(hidden_size, has_rec=False, **lif_kwargs),
                LinearTorch((hidden_size, hidden_size), has_bias=has_bias),
                LIFTorch(hidden_size, has_rec=False, **lif_kwargs),
                LinearTorch((hidden_size, num_classes), has_bias=has_bias),
                ExpSynTorch(num_classes, dt=dt, tau=Constant(tau_syn)),
            ).to(self.device)

        self.optimizer = torch.optim.Adam(
            [p for _, p in self.net.named_parameters()], lr=lr, betas=(0.9, 0.999)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )

        self.loss_hist = []
        self.acc_hist = []
        self.test_acc_hist = []

    def _lif_layer_indices(self):
        """Return indices of LIFTorch layers in self.net."""
        return [i for i, layer in enumerate(self.net) if isinstance(layer, LIFTorch)]

    def _linear_before(self, lif_idx):
        """Return the LinearTorch layer immediately before lif_idx."""
        for i in range(lif_idx - 1, -1, -1):
            if isinstance(self.net[i], LinearTorch):
                return self.net[i]
        return None

    def _collect_hidden_spikes(self, recording):
        """Return spike tensors from LIF hidden layers only."""
        spike_tensors = []
        for key, value in recording.items():
            layer_idx = int(key.split('_')[0])
            if isinstance(self.net[layer_idx], LIFTorch):
                spikes = value.get('spk', None) if isinstance(value, dict) else value
                if spikes is not None:
                    spike_tensors.append(spikes)
        return spike_tensors

    def _dead_neuron_bump(self, epoch_spikes_per_layer):
        """
        For each hidden LIF layer, find neurons that fired 0 spikes across
        the entire epoch and add 0.002 to their input LinearTorch weights.
        epoch_spikes_per_layer: list of 1-D tensors [hidden_size] (summed over epoch batches).
        """
        lif_indices = self._lif_layer_indices()
        for lif_idx, epoch_spikes in zip(lif_indices, epoch_spikes_per_layer):
            dead_mask = (epoch_spikes == 0)
            n_dead = dead_mask.sum().item()
            if n_dead == 0:
                continue
            linear = self._linear_before(lif_idx)
            if linear is None:
                continue
            with torch.no_grad():
                # weight shape: (in_features, out_features)
                linear.weight[:, dead_mask] += 0.002
            print(f"  Dead neuron bump: layer {lif_idx} — {n_dead} dead neurons bumped")

    def train_model(self, train_loader, test_loader, num_epochs=200, print_every=20):
        """Train the model."""
        print(f"Starting training — arch={self.arch}, hidden={self.hidden_size}, "
              f"dt={self.dt*1e3:.0f}ms, target_rate={self.target_rate}Hz, "
              f"rate_lam={self.rate_lam}, spike_lam={self.spike_lam}")
        self.epochs = num_epochs

        # Expected spikes per neuron per timestep at target_rate Hz
        target_per_step = self.target_rate * self.dt
        lif_indices = self._lif_layer_indices()
        cnt = 0

        for epoch in range(num_epochs):
            # Accumulate epoch-level spike counts per LIF layer for dead neuron bump
            epoch_spikes = [torch.zeros(self.hidden_size, device=self.device)
                            for _ in lif_indices]

            for batch, (data, targets) in enumerate(train_loader):
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                self.net.train()
                self.optimizer.zero_grad()

                output, _, recording = self.net(data, record=True)

                # Classification: max over time of output voltages (paper: max of voltage integrals)
                logits = output.max(dim=1).values  # [batch, num_classes]

                # Collect hidden spike tensors
                hidden_spikes = self._collect_hidden_spikes(recording)

                # Firing rate regularisation: penalise deviation from target_rate
                rate_loss = torch.tensor(0.0, device=self.device)
                for i, spikes in enumerate(hidden_spikes):
                    mean_rate = spikes.float().mean()
                    rate_loss = rate_loss + (mean_rate - target_per_step) ** 2
                    # Accumulate for dead neuron bump (sum spikes per neuron across batch & time)
                    epoch_spikes[i] += spikes.float().sum(dim=(0, 1))

                rate_loss = self.rate_lam * rate_loss

                # Optional sparsity penalty (spike_lam > 0 for sparse model)
                spike_loss = torch.tensor(0.0, device=self.device)
                if self.spike_lam > 0:
                    for spikes in hidden_spikes:
                        spike_loss = spike_loss + spikes.float().sum()
                    spike_loss = self.spike_lam * spike_loss

                loss = self.loss_fn(logits, targets) + rate_loss + spike_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in self.net.named_parameters()], max_norm=1.0
                )
                self.optimizer.step()

                self.loss_hist.append(loss.item())
                acc = (logits.argmax(1) == targets).sum().item() / targets.size(0)
                self.acc_hist.append(acc)

                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iter {batch} | Loss: {loss.item():.2f} | "
                          f"Train Acc: {acc*100:.2f}%")
                    test_acc = self.validate_model(test_loader)
                    self.test_acc_hist.append(test_acc)
                    print(f"Test Acc: {test_acc*100:.2f}% | "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.6f}\n")
                    self.scheduler.step(test_acc)

                cnt += 1

            # Dead neuron bump at end of each epoch
            self._dead_neuron_bump(epoch_spikes)

    def validate_model(self, test_loader):
        """Validate the model on test data."""
        self.net.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device).float()
                targets = targets.to(self.device)

                output, _, _ = self.net(data)
                logits = output.max(dim=1).values  # max over time
                correct += (logits.argmax(1) == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0

    def save_model(self, base_path="./results", counter_file="experiment_counter.txt"):
        """Save model weights and training graphs."""
        import matplotlib.pyplot as plt
        import os

        os.makedirs(f"{base_path}/shd/{self.model_type}/models", exist_ok=True)
        os.makedirs(f"{base_path}/shd/{self.model_type}/graphs", exist_ok=True)

        counter_path = f"{base_path}/shd/{self.model_type}/{counter_file}"
        if not os.path.exists(counter_path):
            os.makedirs(os.path.dirname(counter_path), exist_ok=True)
            with open(counter_path, "w") as f:
                f.write("0")

        with open(counter_path, "r") as f:
            num = int(f.read().strip()) + 1
        with open(counter_path, "w") as f:
            f.write(str(num))

        tag = f"Take{num}_{self.arch}_H{self.hidden_size}_T{self.n_frames}_Epochs{self.epochs}"
        model_path = f"{base_path}/shd/{self.model_type}/models/{tag}.pth"
        graph_path = f"{base_path}/shd/{self.model_type}/graphs/{tag}.png"

        torch.save({
            'state_dict': self.net.state_dict(),
            'hyperparams': {
                'input_size': self.input_size,
                'n_frames': self.n_frames,
                'tau_mem': self.tau_mem,
                'tau_syn': self.tau_syn,
                'arch': self.arch,
                'hidden_size': self.hidden_size,
                'rate_lam': self.rate_lam,
                'target_rate': self.target_rate,
                'spike_lam': self.spike_lam,
                'model_type': self.model_type,
                'num_classes': self.num_classes,
                'dt': self.dt,
                'threshold': self.threshold,
                'has_bias': self.has_bias,
            }
        }, model_path)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(self.acc_hist)
        axes[0].set_title("Train Accuracy")
        axes[0].set_xlabel("Iteration")
        axes[1].plot(self.test_acc_hist)
        axes[1].set_title("Test Accuracy")
        axes[1].set_xlabel("Validation checkpoint")
        axes[2].plot(self.loss_hist)
        axes[2].set_title("Loss")
        axes[2].set_xlabel("Iteration")
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        print(f"Model saved to: {model_path}")
        print(f"Graph saved to: {graph_path}")

    def load_model(self, model_path):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['state_dict'])
        else:
            self.net.load_state_dict(checkpoint)
        self.net.eval()
        print(f"Model loaded from: {model_path}")

    @staticmethod
    def load_hyperparams(model_path, device='cpu'):
        """Load hyperparameters from checkpoint."""
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'hyperparams' in checkpoint:
            return checkpoint['hyperparams']
        raise ValueError(f"No hyperparams found in {model_path}")
