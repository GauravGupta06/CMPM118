"""UCI HAR dataset loader."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tonic


class Compose:
    """Compose multiple transforms together (picklable for multiprocessing)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Normalize:
    """Per-sample z-score normalization across time for each channel."""

    def __init__(self, eps=1e-6, time_first=True):
        self.eps = eps
        self.time_first = time_first

    def __call__(self, x):
        if self.time_first:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
        else:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
        return (x - mean) / (std + self.eps)


class Binarize:
    """Binarize values using >0 threshold (rate encoding for SNNs).

    Converts normalized continuous values to binary spikes:
    - Values > 0 become 1
    - Values <= 0 become 0

    This is consistent with Xylo hardware rate encoding.
    """

    def __call__(self, x):
        return (x > 0).astype(np.float32)


class ToTensor:
    """Convert numpy array to torch.float32 tensor."""
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float32)


class UCIHARRaw(Dataset):
    """Raw UCI HAR dataset - reads pre-windowed inertial signals."""

    SIGNAL_FILES = [
        "body_acc_x_{}.txt", "body_acc_y_{}.txt", "body_acc_z_{}.txt",
        "body_gyro_x_{}.txt", "body_gyro_y_{}.txt", "body_gyro_z_{}.txt",
        "total_acc_x_{}.txt", "total_acc_y_{}.txt", "total_acc_z_{}.txt",
    ]

    def __init__(self, root, split="train", transform=None, time_first=True):
        super().__init__()
        assert split in {"train", "test"}

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
                raise FileNotFoundError(f"Missing file: {fp}")
            signals.append(np.loadtxt(fp, dtype=np.float32))

        # Stack into (N, 9, 128)
        X = np.stack(signals, axis=1)

        # Labels 1..6 -> 0..5
        y_path = os.path.join(base, f"y_{split}.txt")
        y = np.loadtxt(y_path, dtype=np.int64) - 1

        # Convert to (N, T, C) or (N, C, T)
        if self.time_first:
            X = np.transpose(X, (0, 2, 1))  # (N, 128, 9)

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.n_samples = self.X.shape[0]
        self.T = self.X.shape[1] if self.time_first else self.X.shape[2]
        self.C = self.X.shape[2] if self.time_first else self.X.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(y)


class _PadTruncWrapper(Dataset):
    """Pads/truncates each sample to n_frames along time axis."""

    def __init__(self, base_ds, n_frames, time_first=True):
        self.base_ds = base_ds
        self.n_frames = n_frames
        self.time_first = time_first

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        if self.time_first:
            T, C = x.shape
            if T < self.n_frames:
                pad = torch.zeros((self.n_frames - T, C), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            else:
                x = x[:self.n_frames, :]
        else:
            C, T = x.shape
            if T < self.n_frames:
                pad = torch.zeros((C, self.n_frames - T), dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, :self.n_frames]
        return x, y


class UCIHARDataset:
    """UCI HAR dataset loader."""

    def __init__(self, dataset_path, n_frames=128, time_first=True, normalize=True, binarize=False):
        self.dataset_path = dataset_path
        self.n_frames = n_frames
        self.time_first = time_first
        self.normalize = normalize
        self.binarize = binarize
        self.num_classes = 6
        self.NUM_CHANNELS = 9

    def _get_transforms(self):
        """Create transforms for UCI HAR.

        Transform order: Normalize -> Binarize -> ToTensor
        """
        ops = []
        if self.normalize:
            ops.append(Normalize(time_first=self.time_first))
        if self.binarize:
            ops.append(Binarize())
        ops.append(ToTensor())
        return Compose(ops)

    def _load_raw_dataset(self, train=True):
        """Load UCI HAR dataset from local extracted files."""
        split = "train" if train else "test"
        ds = UCIHARRaw(
            root=self.dataset_path,
            split=split,
            transform=self._get_transforms(),
            time_first=self.time_first,
        )
        if ds.T != self.n_frames:
            return _PadTruncWrapper(ds, n_frames=self.n_frames, time_first=self.time_first)
        return ds

    def load_uci_har(self):
        """Load UCI HAR dataset with caching."""
        fmt = "Tfirst" if self.time_first else "Cfirst"
        norm = "norm" if self.normalize else "nonorm"
        binar = "bin" if self.binarize else "nobin"
        cache_path = f"{self.dataset_path}/uci_har/{fmt}_{norm}_{binar}_T{self.n_frames}"
        cache_exists = os.path.exists(f"{cache_path}/train") and os.path.exists(f"{cache_path}/test")

        # Load raw datasets only if cache doesn't exist
        train_dataset = None if cache_exists else self._load_raw_dataset(train=True)
        test_dataset = None if cache_exists else self._load_raw_dataset(train=False)

        # Create cached datasets
        cached_train = tonic.DiskCachedDataset(train_dataset, cache_path=f"{cache_path}/train")
        cached_test = tonic.DiskCachedDataset(test_dataset, cache_path=f"{cache_path}/test")

        # Populate cache on first load
        if not cache_exists:
            print(f"Caching dataset to {cache_path}...")
            print("Caching training data...")
            for _ in cached_train:
                pass
            print("Caching test data...")
            for _ in cached_test:
                pass
            print("Caching complete!")

        return cached_train, cached_test

    def get_num_classes(self):
        return self.num_classes
