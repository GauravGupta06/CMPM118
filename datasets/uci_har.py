"""UCI HAR dataset loader (file-style similar to SHD loader)."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from core.base_dataset import NeuromorphicDataset


# -----------------------------
# Transforms
# -----------------------------
class Normalize:
    """
    Per-sample z-score normalization across time for each channel.
    Input:  (T, C) or (C, T)
    Output: same shape
    """

    def __init__(self, eps: float = 1e-6, time_first: bool = True):
        self.eps = eps
        self.time_first = time_first

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (T, C) if time_first else (C, T)
        if self.time_first:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
        else:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
        return (x - mean) / (std + self.eps)


class ToTensor:
    """Convert numpy array to torch.float32 tensor."""
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32)


# -----------------------------
# Raw Dataset
# -----------------------------
class UCIHARRaw(Dataset):
    """
    Reads UCI HAR pre-windowed inertial signals:
      - 9 channels, each sample is 128 timesteps
      - labels 1..6
    Returns: (X, y) where X is (T, C) float32 by default.
    """

    SIGNAL_FILES = [
        "body_acc_x_{}.txt",
        "body_acc_y_{}.txt",
        "body_acc_z_{}.txt",
        "body_gyro_x_{}.txt",
        "body_gyro_y_{}.txt",
        "body_gyro_z_{}.txt",
        "total_acc_x_{}.txt",
        "total_acc_y_{}.txt",
        "total_acc_z_{}.txt",
    ]

    def __init__(self, root: str, split: str = "train", transform=None, time_first: bool = True):
        """
        Args:
            root: path to folder that contains "UCI HAR Dataset"
                  e.g. dataset_path/UCI HAR Dataset/...
            split: "train" or "test"
            transform: callable applied to X
            time_first: if True return (T, C); else (C, T)
        """
        super().__init__()
        assert split in {"train", "test"}, "split must be 'train' or 'test'"

        self.root = root
        self.split = split
        self.transform = transform
        self.time_first = time_first

        base = os.path.join(root, "UCI_HAR_Dataset", split)
        signals_dir = os.path.join(base, "Inertial Signals")

        # Load 9 signals -> shape per file: (N, 128)
        signals = []
        for pattern in self.SIGNAL_FILES:
            fp = os.path.join(signals_dir, pattern.format(split))
            if not os.path.exists(fp):
                raise FileNotFoundError(
                    f"Missing file: {fp}\n"
                    f"Expected UCI HAR extracted under: {os.path.join(root, 'UCI HAR Dataset')}"
                )
            arr = np.loadtxt(fp, dtype=np.float32)  # (N, 128)
            signals.append(arr)

        # Stack into (N, 9, 128)
        X = np.stack(signals, axis=1)

        # Labels file -> (N,)
        y_path = os.path.join(base, f"y_{split}.txt")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing labels file: {y_path}")
        y = np.loadtxt(y_path, dtype=np.int64)

        # Convert labels 1..6 -> 0..5
        y = y - 1

        # Return as (N, T, C) or (N, C, T)
        if self.time_first:
            # (N, 9, 128) -> (N, 128, 9)
            X = np.transpose(X, (0, 2, 1))

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

        # Infer sizes
        self.n_samples = self.X.shape[0]
        self.T = self.X.shape[1] if self.time_first else self.X.shape[2]
        self.C = self.X.shape[2] if self.time_first else self.X.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (T, C) if time_first else (C, T)
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        # ensure label is torch long if user wants
        return x, int(y)


# -----------------------------
# Main Dataset Loader (like SHD)
# -----------------------------
class UCIHARDataset(NeuromorphicDataset):
    """UCI HAR dataset loader (windowed IMU signals)."""

    def __init__(
        self,
        dataset_path: str,
        n_frames: int = 128,
        time_first: bool = True,
        normalize: bool = True,
    ):
        """
        Args:
            dataset_path: Root path for dataset storage (contains "UCI HAR Dataset" folder)
            n_frames: number of timesteps per window (UCI HAR is typically 128)
            time_first: output shape is (T, C) if True else (C, T)
            normalize: apply per-sample z-score normalization per channel
        """
        super().__init__(dataset_path, n_frames)
        self.num_classes = 6
        self.time_first = time_first
        self.normalize = normalize

        # UCI HAR channels: 9
        self.NUM_CHANNELS = 9
        self.sensor_size = (self.NUM_CHANNELS, 1, 1)  # keep similar "shape" notion

    def _get_transforms(self):
        """Create transforms for UCI HAR."""
        ops = []
        if self.normalize:
            ops.append(Normalize(time_first=self.time_first))
        ops.append(ToTensor())
        return lambda x: ops[-1](ops[0](x)) if len(ops) == 2 else ops[0](x)

    def _load_raw_dataset(self, train=True):
        """Load UCI HAR dataset from local extracted files."""
        split = "train" if train else "test"
        ds = UCIHARRaw(
            root=self.dataset_path,
            split=split,
            transform=self._get_transforms(),
            time_first=self.time_first,
        )

        # Optional: enforce / truncate / pad to n_frames to match your framework
        # UCI HAR is already fixed-length (typically 128), but keep it consistent:
        if ds.T != self.n_frames:
            # Wrap dataset to pad/truncate to n_frames
            return _PadTruncWrapper(ds, n_frames=self.n_frames, time_first=self.time_first)
        return ds

    def _get_cache_path(self):
        """Generate cache path based on configuration."""
        fmt = "Tfirst" if self.time_first else "Cfirst"
        norm = "norm" if self.normalize else "nonorm"
        return f"{self.dataset_path}/uci_har/{fmt}_{norm}_T{self.n_frames}"

    def get_num_classes(self):
        """Return number of classes for UCI HAR."""
        return self.num_classes

    def load_uci_har(self):
        """Load UCI HAR dataset."""
        return self.create_datasets()


class _PadTruncWrapper(Dataset):
    """Pads/truncates each sample to n_frames along time axis."""

    def __init__(self, base_ds: Dataset, n_frames: int, time_first: bool = True):
        self.base_ds = base_ds
        self.n_frames = n_frames
        self.time_first = time_first

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        # x is torch.Tensor either (T, C) or (C, T)
        if self.time_first:
            T, C = x.shape
            if T < self.n_frames:
                pad = torch.zeros((self.n_frames - T, C), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            else:
                x = x[: self.n_frames, :]
        else:
            C, T = x.shape
            if T < self.n_frames:
                pad = torch.zeros((C, self.n_frames - T), dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, : self.n_frames]
        return x, y
