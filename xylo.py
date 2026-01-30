import json
import numpy as np
from rockpool.devices.xylo import find_xylo_hdks
from rockpool.devices.xylo import syns65302 as xa3 # XyloAudio 3
from rockpool.nn.modules import LIFTorch, LinearTorch, ExpSynTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant
from rockpool.transform import quantize_methods as q
import samna
import torch

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from datasets.uci_har import UCIHARDataset

# Load your trained Rockpool network
device = None
dt = 0.02
has_bias = False
tau_mem = 0.1
tau_syn = 0.05
threshold = 1.0

net = Sequential(
    LinearTorch((9, 128), has_bias=has_bias),
    LIFTorch(
        128,
        tau_mem=Constant(tau_mem),
        tau_syn=Constant(tau_syn),
        threshold=Constant(threshold),
        bias=Constant(0.0),
        dt=dt,
        has_rec=True,
    ),

    LinearTorch((128, 64), has_bias=has_bias),
    LIFTorch(
        64,
        tau_mem=Constant(tau_mem),
        tau_syn=Constant(tau_syn),
        threshold=Constant(threshold),
        bias=Constant(0.0),
        dt=dt,
        has_rec=True,
    ),

    LinearTorch((64, 32), has_bias=has_bias),
    LIFTorch(
        32,
        tau_mem=Constant(tau_mem),
        tau_syn=Constant(tau_syn),
        threshold=Constant(threshold),
        bias=Constant(0.0),
        dt=dt,
        has_rec=True,
    ),

    LinearTorch((32, 6), has_bias=has_bias),

    # non-spiking readout
    #ExpSynTorch(6, dt=dt, tau=Constant(5e-3)),

    LIFTorch(
        6,
        tau_mem=Constant(tau_mem),
        tau_syn=Constant(tau_syn),
        threshold=Constant(threshold),
        bias=Constant(0.0),
        dt=dt,
        has_rec=False,
    ),
).to(device)

checkpoint = torch.load("./results/large/models//Rockpool_Non_Sparse_Take102_HAR_Input9_T128_FC_Rockpool_Epochs100.pth", map_location='cpu')

"""
Can't use as_graph() @ line 76 with ExpSynTorch, so strict=False says to ignore the removed ExpSynTorch layer
"""
net.load_state_dict(checkpoint, strict=False)

net.eval()

# Extract the graph from your Rockpool network
graph = net.as_graph()
print(graph)
# GraphHolder "TorchSequential_xxx" with N input nodes -> M output nodes

# Map graph to hardware specification
spec = xa3.mapper(
    graph,
    weight_dtype='float', # Keep float for quantization step
    threshold_dtype='float',
    dash_dtype='float'
)
spec['dt'] = dt # Preserve timestep

# Channel-wise quantization (recommended)
quant_spec = q.channel_quantize(**spec)

# Restore parameters that may be dropped
for key in ['dt', 'aliases']:
    if key in spec:
        quant_spec[key] = spec[key]

# Enforce hardware data types for decay parameters
for key in ["dash_mem", "dash_mem_out", "dash_syn", "dash_syn_2", "dash_syn_out"]:
    if key in quant_spec:
        quant_spec[key] = np.abs(quant_spec[key]).astype(np.uint8)

# Convert numpy arrays to JSON-serializable format
json_spec = {
    k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in quant_spec.items()
}

with open('hw_config.json', 'w') as f:
    json.dump(json_spec, f, indent=4)

# Option 1: Random Poisson spikes - type: numpy.ndarray
#sample_spikes = (np.random.rand(num_samples, num_steps, n_input) < 0.05).astype(np.float32)

# Option 2: Real data from your application
dataset_path = "./data"   # folder that contains "UCI_HAR_Dataset"
loader = UCIHARDataset(
    dataset_path=dataset_path,
    n_frames=128,
    time_first=True,
    normalize=True
)

train_ds, test_ds = loader.load_uci_har()   # returns (cached_train, cached_test)

sample_spikes, label = train_ds[0]
sample_spikes = np.asarray(sample_spikes)

# Simple rate encoding
sample_spikes = (sample_spikes > 0).astype(np.int32)

# Save whichever you choose
np.save('sample_spikes.npy', sample_spikes)

# Convert specification to hardware configuration
config, is_valid, msg = xa3.config_from_specification(**quant_spec)
print("Hidden neurons:", len(config.hidden.weights))

"""
len(config.readout.weights) = 32, but there's a small chance that it should instead be 6.

If it IS the case that it's supposed to be 6 when it's actually 32, I'm not sure how we could
fix that, since I don't think changing the model architecture would fix it.

But otherwise, if it's not a problem, this can be ignored
"""
print("Output neurons:", len(config.readout.weights))

if not is_valid:
    raise ValueError(f"Invalid configuration: {msg}")

# Create simulator
modSim = xa3.XyloSim.from_config(config)

# Run inference
output, _, recorded = modSim(sample_spikes, record=True)

print("\n" + "="*60)
print("Below is the error-prone section")
print("="*60)

# Connect to HDK
hdk_nodes, support_modules, versions = find_xylo_hdks()
if not hdk_nodes:
    raise RuntimeError("No Xylo HDK found")
hdk = hdk_nodes[0]
          
"""
I don't think that these configurations are needed. When I keep these, they produce errors,
but without them, the code works up until the modSamna portion

config = xa3.XyloConfiguration()
config.input.weights = np.array(quant_spec['weights_in'],dtype=np.int8
config.hidden.weights = np.array(quant_spec['weights_rec'],dtype=np.int8
config.readout.weights = np.array(quant_spec['weights_out'],dtype=np.int8)
# ... configure neurons ...
"""

is_valid, msg = samna.xyloAudio3.validate_configuration(config)
if not is_valid:
    raise ValueError(f"Invalid configuration: {msg}")

# Deploy
modSamna = xa3.XyloSamna(hdk, config, dt=dt, power_frequency=20.)

# Run inference
"""
From what I have tried, the line below always crashes my Python and produces an error:

"TimeoutError: Timeout after 5.0s when reading current timestep."

From DeepSeek:

"Power is the most likely issue - USB-C to USB-A adapters on MacBooks often don't provide enough power for the HDK"

"This is a known issue with Xylo HDK. Try these steps:
    1. UNPLUG the HDK from USB
    2. Wait 10 seconds
    3. PLUG it back in
    4. Wait 5 seconds for device enumeration
    5. Run this script again

If that doesn't work:
    1. Restart your computer
    2. Try a different USB port
    3. Check if other processes are using the HDK

If all else fails, fall back to simulation results only."

Unplugging and restarting hasn't worked for me. I can only
assume we might need a non-Mac laptop to run it? I have no
clue and I've been at this for >2.5 hours by now
"""
output, state, recorded = modSamna(sample_spikes, record=True)

