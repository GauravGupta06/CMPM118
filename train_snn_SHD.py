# %%
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



from SNN_model import SNNModel
from LoadDataset import load_dataset

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("using " + str(device))

# %%
dataset_name = "SHD"
dataset_path = "/home/gauravgupta/CMPM118/data"  # change path if needed

if dataset_name == "DVSGesture":
    w, h, n_frames = 32, 32, 32 # typical temporal bin count for DVSGesture
if dataset_name == "SHD":
    w, h, n_frames = 700, 1, 100 # typical temporal bin count for DVSGesture

cached_train, cached_test, num_classes = load_dataset(
    dataset_name=dataset_name,
    dataset_path=dataset_path,
    w=w,
    h=h,
    n_frames=n_frames,
)

# %%



