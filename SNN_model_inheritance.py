import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from abc import ABC, abstractmethod


class BaseSNNModel(ABC):
    """Base class for SNN models with common functionality."""
    
    def __init__(self, n_frames, beta, spike_lam, slope=25, model_type="sparse", device=None):
        self.n_frames = n_frames
        self.beta = beta
        self.slope = slope
        self.model_type = model_type
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.spike_lam = spike_lam
        self.epochs = None
        
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
    
    @abstractmethod
    def forward_pass(self, data):
        """
        Forward pass through the network. Must be implemented by child classes.
        
        Args:
            data: Input tensor
            
        Returns:
            spk_rec: Stacked spike recordings
            spike_count: Total spike count
        """
        pass
    
    @abstractmethod
    def train_model(self, train_loader, test_loader, num_epochs=150, print_every=15):
        """
        Train the model. Must be implemented by child classes.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            num_epochs: Number of training epochs
            print_every: Print stats every N iterations
        """
        pass
    
    def spike_regularizer(self, spike_count, lam=1e-4):
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


class DVSGestureSNN(BaseSNNModel):
    """SNN model for DVS Gesture dataset."""
    
    def __init__(self, w, h, n_frames, beta, spike_lam, slope=25, model_type="sparse", device=None):
        self.w = w
        self.h = h
        super().__init__(n_frames, beta, spike_lam, slope, model_type, device)
    
    def _build_network(self):
        test_input = torch.zeros((1, 2, self.w, self.h))
        test_input = test_input.to(self.device)
        x = nn.Conv2d(2, 12, 5)(test_input)
        x = nn.MaxPool2d(2)(x)
        x = nn.Conv2d(12, 32, 5)(x)
        x = nn.MaxPool2d(2)(x)
    
        net = nn.Sequential(
            nn.Conv2d(2, 12, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True),
            nn.Conv2d(12, 32, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(x.numel(), 11),
            snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True, output=True)
        ).to(self.device)

        return net
    
    def forward_pass(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: Input tensor of shape [T, batch, 2, H, W] or [T, 2, H, W]
            
        Returns:
            spk_rec: Stacked spike recordings
            spike_count: Total spike count
        """
        utils.reset(self.net)
        spk_rec = []
        spike_count = torch.tensor(0., device=self.device)
        
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
        self.epochs = num_epochs
        """
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            num_epochs: Number of training epochs
            print_every: Print stats every N iterations
        """
        cnt = 0
        for epoch in range(num_epochs):
            for batch, (data, targets) in enumerate(iter(train_loader)):
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                self.net.train()
                
                # Forward pass
                spk_rec, spike_count = self.forward_pass(data)
                loss = self.loss_fn(spk_rec, targets) + self.spike_regularizer(spike_count, lam = self.spike_lam)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Store history
                self.loss_hist.append(loss.item())
                acc = SF.accuracy_rate(spk_rec, targets)
                self.acc_hist.append(acc)
                
                # Print and validate
                if cnt % print_every == 0:
                    print(f"Epoch {epoch}, Iteration {batch} \nTrain Loss: {loss.item():.2f}")
                    print(f"Train Accuracy: {acc * 100:.2f}%")
                    test_acc = self.validate_model(test_loader)
                    self.test_acc_hist.append(test_acc)
                    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")
                
                cnt += 1
    
    def _get_save_params(self):
        return f"{self.w}x{self.h}_T{self.n_frames}_B{self.beta}_SpkLam{self.spike_lam}"


class SHDSNN(BaseSNNModel):
    """SNN model for Spiking Heidelberg Digits (SHD) dataset."""
    
    def __init__(self, n_inputs, n_frames, beta, spike_lam, slope=25, model_type="sparse", device=None):
        self.n_inputs = n_inputs
        super().__init__(n_frames, beta, spike_lam, slope, model_type, device)
    
    def _build_network(self):
        """Build network architecture for SHD."""
        # Implement your SHD architecture here
        pass
    
    def forward_pass(self, data):
        """Forward pass for SHD data."""
        # Implement your SHD forward pass here
        pass
    
    def train_model(self, train_loader, test_loader, num_epochs=150, print_every=15):
        """Train model on SHD data."""
        # Implement your SHD training loop here
        # Will likely be similar to DVS but may have dataset-specific differences
        pass
    
    def _get_save_params(self):
        return f"Inputs{self.n_inputs}_T{self.n_frames}_B{self.beta}_SpkLam{self.spike_lam}"