# All imports go here
import numpy as np
import numpy.lib.recfunctions as rf
import tonic
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from lempel_ziv_complexity import lempel_ziv_complexity



from SNN_model_inheritance import SHDSNN
from LoadDataset import load_dataset



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("using " + str(device))



freq_bins = 700
height = 1
n_frames = 100



cached_train, cached_test, num_classes = load_dataset(
    dataset_name="SHD",
    dataset_path='data',
    w=freq_bins,
    h=height,
    n_frames=n_frames,
)



# cached_train, cached_test, num_classes = load_dataset(
#     dataset_name="ASLDVS",  # or "DVSGesture"
#     dataset_path='data',
#     w=width,
#     h=height,
#     n_frames=n_frames
# )


# Create and load dense model
dense_model = SHDSNN(
    freq_bins=freq_bins,
    n_frames=n_frames,
    beta= 0.6,
    spike_lam= 0,
    slope= 25,
    model_type="dense",
    device=device,
    num_classes=num_classes,
)



num_epochs = 200
active_cores = 4



train_loader = torch.utils.data.DataLoader(cached_train, batch_size=64, shuffle=True, num_workers = active_cores, drop_last=True, 
                                           collate_fn=tonic.collation.PadTensors(batch_first=False))
test_loader = torch.utils.data.DataLoader(cached_test, batch_size=32, shuffle=True, num_workers = active_cores, drop_last=True, 
                                          collate_fn=tonic.collation.PadTensors(batch_first=False))



print("starting training")
dense_model.train_model(train_loader, test_loader, num_epochs = num_epochs)
dense_model.save_model()


