import os
import numpy as np
import torch
from torch.utils.data import Dataset
from core.base_dataset import NeuromorphicDataset
from torch.utils.data import DataLoader
from core.base_model import BaseSNNModel

class UCIHARDataset(NeuromorphicDataset):
    def __init__(self, dataset_path, n_frames=128):
        super().__init__(dataset_path, n_frames)
        self.num_classes = 6
        self.num_features = 9

    def _get_transforms(self):
        return None
    
    def _load_raw_dataset(self, train=True):
        """
        Load raw UCI HAR signals.
        Returns list of (frames, label).
        """
        split = "train" if train else "test"
        base = os.path.join(self.dataset_path, "UCI_HAR_Dataset", split)

        signal_names = [
            "body_acc_x_", "body_acc_y_", "body_acc_z_",
            "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
            "total_acc_x_", "total_acc_y_", "total_acc_z_"
        ]

        signals = []
        for name in signal_names:
            path = os.path.join(
                base,
                "Inertial Signals",
                f"{name}{split}.txt"
            )
            signals.append(np.loadtxt(path))

        # Shape: [N, T, F]
        X = np.stack(signals, axis=-1)

        # ---- NORMALIZE ----
        # zero-mean, unit variance per sample
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)

        y = np.loadtxt(
            os.path.join(base, f"y_{split}.txt")
        ).astype(int) - 1  # labels 0–5

        return list(zip(X, y))
    
    def _get_cache_path(self):
        """Cache path consistent with SHD style."""
        return f"{self.dataset_path}/uci_har/T{self.n_frames}"

    def get_num_classes(self):
        return self.num_classes
    


class TorchUCIHAR(Dataset):
    """Torch wrapper for cached UCI HAR dataset."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]

        # [T, F] → [T, 1, F]
        x = torch.from_numpy(x).float().unsqueeze(1)
        y = torch.tensor(y).long()

        return x, y
    

    
def load_uci_har(dataset_path, n_frames=128):
    """
    Load UCI HAR dataset (SHD-style API) without tonic caching.

    Args:
        dataset_path: Root dataset directory
        n_frames: Temporal length (default 128)

    Returns:
        train_ds, test_ds, num_classes
    """
    # Initialize dataset object
    dataset = UCIHARDataset(dataset_path, n_frames)

    # Load raw train/test data
    train_raw = dataset._load_raw_dataset(train=True)
    test_raw  = dataset._load_raw_dataset(train=False)
    num_classes = dataset.get_num_classes()

    # Wrap in Torch Dataset
    train_ds = TorchUCIHAR(train_raw)
    test_ds  = TorchUCIHAR(test_raw)

    return train_ds, test_ds, num_classes




train_ds, test_ds, num_classes = load_uci_har(
    dataset_path="./data",
    n_frames=128
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False
)

