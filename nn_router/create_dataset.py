import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
from torchvision.transforms import v2
import pandas as pd
import matplotlib.pyplot as plt
import snntorch as snn
import csv

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

grad = snn.surrogate.fast_sigmoid(slope=25)
beta = 0.5

# -- LOAD LARGE SNN --
w,h=64,64
n_frames=100
test_input = torch.zeros((1, 2, w, h))  # 2 polarity channels
x = nn.Conv2d(2, 12, 5)(test_input)
x = nn.MaxPool2d(2)(x)
x = nn.Conv2d(12, 32, 5)(x)
x = nn.MaxPool2d(2)(x)
print("Output shape before flatten:", x.shape)
print("Flattened size:", x.numel())
flattenedSize = x.numel()

net = nn.Sequential(
    nn.Conv2d(2, 12, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
    nn.Conv2d(12, 32, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(flattenedSize, 11),   # make sure 800 matches flattenedSize
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
).to(device)

model_path = "../results/large/models/Large_Take4.pth"
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()
print("Large model loaded successfully.")

# -- LOAD SMALL SNN --
w,h=32,32
n_frames=5
test_input = torch.zeros((1, 2, w, h))  # 2 polarity channels
x = nn.Conv2d(2, 8, 3)(test_input)
x = nn.MaxPool2d(2)(x)
print("Output shape before flatten:", x.shape)
print("Flattened size:", x.numel())
flattenedSize = x.numel()

net_s = nn.Sequential(
    nn.Conv2d(2, 8, 3), # in_channels, out_channels, kernel_size
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(flattenedSize, 11),
    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
).to(device)

model_path = "../results/small/models/Small_Take2_32x32_T5.pth"
net_s.load_state_dict(torch.load(model_path, map_location=device))
net_s.eval()
print("Small model loaded successfully.")

# -- WRITE DATA --
with open("data.csv", "a") as file:
    writer = csv.writer(file)
    writer.writerow([1, 2, 3, 4]) # test