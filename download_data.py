"""
Assortment of libraries because I'm too lazy to pick out which one to not keep. Use or don't use

When done, do something like:

`python -m venv venv
source venv/bin/activate
pip install -r requirements.txt`
(Make sure your Python is a version that's >= 3.10 and < 3.14 or else tonic won't install)

For me (Kevin):
`python3.13 -m venv venv
source venv/bin/activate
[install libraries]
pip3 freeze > requirements.txt`
"""

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import snntorch
import tonic

# REPLACE FILE NAME HERE
file_name = "/Users/q-bh/repos/CMPM118/data"
tonic.datasets.DVSGesture(save_to=file_name, train=True)
tonic.datasets.DVSGesture(save_to=file_name, train=False)