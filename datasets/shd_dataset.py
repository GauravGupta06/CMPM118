"""SHD dataset loader."""

import os
import tonic
from tonic import transforms
import torch
import numpy as np
from core.base_dataset import NeuromorphicDataset



class ToRaster:
    """Convert SHD events to raster with full channels and polarity."""

    def __init__(self, num_channels=700, num_polarities=2, sample_T=100):
        self.num_channels = num_channels
        self.num_polarities = num_polarities
        self.sample_T = sample_T
        self.total_features = num_channels * num_polarities  # 1400

    def __call__(self, events):
        max_t = int(events["t"].max()) + 1
        raster = np.zeros((max_t, self.total_features), dtype=np.float32)

        times = events["t"].astype(int)
        channels = events["x"].astype(int) * self.num_polarities + events["p"].astype(int) 
        # by multiplying with num_polarities, scale the values from 0-700 to 0-1400 for the index.
        # this allows us to index into the raster array. 
        # we then add the polarity to get which of the two possible indexes for the raster we add the event to. 
        # Note that every event is a spike, regardless of polarity. The reason we have the + polarity is to 
        # allow us find the specific column for that specific channel (becuase each channel has 2 columns). 

        valid = (channels < self.total_features)
        np.add.at(raster, (times[valid], channels[valid]), 1)

        # Pad or truncate to sample_T
        if raster.shape[0] < self.sample_T:
            pad = np.zeros((self.sample_T - raster.shape[0], self.total_features), dtype=np.float32)
            raster = np.concatenate([raster, pad], axis=0)
        else:
            raster = raster[:self.sample_T, :]
        # the reason we are doing the code above is becuase the input data is a variable length.
        # some audio samples are longer than others, so we need to pad or truncate to a fixed length.
        # we are padding with 0s if the sample is shorter than the fixed length.
        # we are truncating if the sample is longer than the fixed length.
        
        # Its important to keep the data size the exact same even though it doesn't come in the same size. 
        # This is because the model is going to be trained on batches of data, and if the data size is different
        # for each batch, the model will not be able to learn effectively. Also, the model will not be able to 
        # make predictions on new data if the data size is different. 


        # Binarize
        return (raster > 0).astype(np.float32)






class SHDDataset(NeuromorphicDataset):
    """SHD dataset loader."""

    def __init__(self, dataset_path, NUM_CHANNELS=700, NUM_POLARITIES=2, n_frames=100, net_dt=10e-3):
        """
        Args:
            dataset_path: Root path for dataset storage
            n_frames: Number of temporal bins
        """
        super().__init__(dataset_path, n_frames)
        self.num_classes = 20
        self.NUM_CHANNELS = NUM_CHANNELS
        self.NUM_POLARITIES = NUM_POLARITIES
        self.net_dt = net_dt
        self.sensor_size = (self.NUM_CHANNELS, 1, self.NUM_POLARITIES)


    def _get_transforms(self):
        """Create tonic transforms for SHD dataset."""
        transform = transforms.Compose([
            transforms.Downsample(time_factor= (1e-6/self.net_dt), spatial_factor=1.0),
            ToRaster(self.NUM_CHANNELS, self.NUM_POLARITIES, self.n_frames),
            torch.tensor,
        ])

        return transform

    def _load_raw_dataset(self, train=True):
        """Load SHD dataset from tonic."""
        return tonic.datasets.SHD(save_to=self.dataset_path, transform=self._get_transforms(), train=train)

    def _get_cache_path(self):
        """Generate cache path based on configuration."""
        return f"{self.dataset_path}/shd/700x1_T{self.n_frames}"

    def get_num_classes(self):
        """Return number of classes for SHD."""
        return self.num_classes
    def load_shd(self):
        """Load SHD dataset."""
        return self.create_datasets()



