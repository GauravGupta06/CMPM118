# Rockpool Development - SHD-Focused SNN Implementation

This directory contains a modular Rockpool-based implementation of Spiking Neural Networks (SNNs) for the **SHD (Spiking Heidelberg Digits)** dataset, optimized for Xylo neuromorphic hardware deployment.

## Overview

This is a refactored, production-ready codebase with a clean, extensible architecture:
- Modular design with core base classes
- SHD-focused implementation (fully-connected, Xylo-compatible)
- Support for sparse vs. dense models via hyperparameters
- Optional frequency binning (700 → 16 features) for Xylo input constraints

## Directory Structure

```
rockpoolDevelopment/
├── core/                      # Base classes
│   ├── __init__.py
│   ├── base_model.py         # BaseSNNModel (training, inference, I/O)
│   └── base_dataset.py       # NeuromorphicDataset (abstract interface)
├── datasets/                  # Dataset loaders
│   ├── __init__.py           # Registry
│   └── shd.py                # SHD loading + optional 700→16 binning
├── models/                    # Model implementations
│   ├── __init__.py           # Registry
│   └── shd_model.py          # SHDSNN_FC (fully-connected)
├── legacy/                    # Old implementation (reference)
│   ├── RockpoolSNN_model.py
│   ├── LoadDataset.py
│   ├── train_rockpool.py
│   ├── run_rockpool.py
│   └── test_setup.py
├── train.py                   # Unified training script
├── evaluate.py                # Evaluation script
└── README.md                  # This file
```

## Model Architecture

### SHDSNN_FC (Fully-Connected)
```
Input: [T, B, 2, 1, freq_bins]  # freq_bins = 700 or 16, 2 channels (polarity)
  ↓ Flatten to [B, T, 2*freq_bins]
  ↓
Linear(2*freq_bins → 256) + LIF
  ↓
Linear(256 → 128) + LIF
  ↓
Linear(128 → 20) + LIF (output)

Total parameters:
  - 700 freq bins: ~540k params (1400 → 256 → 128 → 20)
  - 16 freq bins: ~40k params (32 → 256 → 128 → 20)

Hidden neurons: 384 (256 + 128)
Total input features: freq_bins * 2 (due to 2 polarity channels)
```

## Sparse vs. Dense Models

**Key insight:** Both models use **the same architecture**, differentiated **only by hyperparameters**:

| Parameter | Sparse | Dense | Effect |
|-----------|--------|-------|--------|
| `tau_mem` | 0.01 | 0.02 | Membrane time constant (shorter = less integration) |
| `spike_lam` | 1e-6 | 1e-8 | Spike regularization (higher = fewer spikes) |
| `model_type` | "sparse" | "dense" | Label for saving/tracking |

**Expected spike activity (700-input):**
- Sparse: ~1,000-2,000 spikes/sample (lower energy)
- Dense: ~3,000-5,000 spikes/sample (higher accuracy)
- Ratio: ~2-3x difference

**Expected accuracy (700-input):**
- Sparse: 55-65% on SHD test set
- Dense: 60-70% on SHD test set
- Trade-off: Energy vs. accuracy

## Frequency Binning (700 → 16)

The SHD dataset has 700 frequency channels. For Xylo deployment, we can reduce this to 16 bands:

```python
# Simple averaging approach
n_bins = 16
bin_size = 700 // n_bins  # ~44 frequencies per bin
bins = np.array_split(frequencies_700, n_bins)
reduced = np.array([bin.mean(axis=0) for bin in bins])  # [16, ...]
```

When enabled (`--reduce_to_16`):
- Model input: `16*2 = 32` features (instead of 1400)
- Architecture: `32 → 256 → 128 → 20`
- Xylo-compatible (fits input constraints)
- ~40k parameters (vs. ~540k for full resolution)

## Usage

### Training a Model

```bash
cd rockpoolDevelopment

# Train dense model (700 features, full resolution)
python train.py --model_type dense --epochs 200

# Train sparse model (700 features)
python train.py --model_type sparse --epochs 200

# Train dense model (16 features, Xylo-compatible)
python train.py --model_type dense --reduce_to_16 --epochs 200

# Custom configuration
python train.py \
  --model_type dense \
  --reduce_to_16 \
  --n_frames 100 \
  --epochs 200 \
  --batch_size 32 \
  --dataset_path ./data
```

**Training options:**
- `--model_type`: `sparse` or `dense` (default: `dense`)
- `--reduce_to_16`: Enable frequency binning (default: False)
- `--n_frames`: Number of time steps (default: 100)
- `--epochs`: Training epochs (default: 200)
- `--batch_size`: Batch size (default: 32)
- `--dataset_path`: Dataset directory (default: `./data`)

### Evaluating a Model

```bash
# Evaluate dense model (700 features)
python evaluate.py \
  --model_path ../results/large/models/Rockpool_Non_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth \
  --model_type dense

# Evaluate sparse model (700 features)
python evaluate.py \
  --model_path ../results/small/models/Rockpool_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth \
  --model_type sparse

# Evaluate with 16-feature binning
python evaluate.py \
  --model_path ../results/large/models/Rockpool_Non_Sparse_Take1_Input16_T100_FC_Rockpool_Epochs200.pth \
  --model_type dense \
  --reduce_to_16
```

**Evaluation options:**
- `--model_path`: Path to trained model file (required)
- `--model_type`: `sparse` or `dense` (must match training)
- `--reduce_to_16`: Enable frequency binning (must match training)
- `--n_frames`: Number of time steps (must match training)
- `--dataset_path`: Dataset directory (default: `./data`)

## Python API Usage

### Training

```python
import torch
from datasets import load_shd
from models import SHDSNN_FC

# Load dataset
cached_train, cached_test, num_classes = load_shd(
    dataset_path='./data',
    n_frames=100,
    reduce_to_16=False  # Set True for 16-band binning
)

# Create model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = SHDSNN_FC(
    input_size=700,      # or 16 if reduce_to_16=True
    n_frames=100,
    tau_mem=0.02,        # 0.01 for sparse, 0.02 for dense
    spike_lam=1e-8,      # 1e-6 for sparse, 1e-8 for dense
    model_type="dense",  # "sparse" or "dense"
    device=device,
    num_classes=20
)

# Train
model.train_model(train_loader, test_loader, num_epochs=200)

# Save
model.save_model(base_path="../results")
```

### Evaluation

```python
import torch
from models import SHDSNN_FC

# Create model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = SHDSNN_FC(
    input_size=700,
    n_frames=100,
    tau_mem=0.02,
    spike_lam=1e-8,
    model_type="dense",
    device=device,
    num_classes=20
)

# Load weights
model.load_model("path/to/model.pth")

# Evaluate
accuracy = model.validate_model(test_loader)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

## Xylo Hardware Constraints

The Xylo neuromorphic chip has specific limitations:

1. **Neuron Budget**: ~1000 hidden neurons total
   - Our model uses 256 + 128 = 384 neurons ✓

2. **Architecture**: Only fully-connected layers
   - SHDSNN_FC is compatible ✓
   - No Conv1D layers ✓

3. **Input Size**: Limited input neurons
   - 700-input: Requires external preprocessing or use as external input
   - 16-input: Fits within Xylo input constraints ✓

4. **Quantization**: 8-bit integer weights
   - Can be handled during Xylo conversion ✓

## Key Rockpool Quirks

This implementation preserves critical Rockpool-specific patterns:

### 1. Parameter Extraction
```python
# Rockpool's .parameters() returns module names, need named_parameters()
torch_params = [p for name, p in self.net.named_parameters()]
self.optimizer = torch.optim.Adam(torch_params, lr=0.0001)
```

### 2. Network Return Format
```python
# Rockpool returns (output, state_dict, recording_dict)
output, state_dict, recording_dict = self.net(x)
```

### 3. Input Format
```python
# Rockpool expects [B, T, features] (batch-first, time-second)
# Tonic gives [T, B, ...] (time-first)
# Must transpose: x = data.transpose(0, 1)
```

## Dataset Information

**SHD (Spiking Heidelberg Digits):**
- Audio dataset of spoken digits (0-9) + 10 German digits
- 20 classes total
- 700 frequency channels (cochlea model)
- Train: 8,156 samples
- Test: 2,264 samples

**Caching:**
- First load will download and cache the dataset
- Subsequent loads use cached data for faster training
- Cache location: `{dataset_path}/shd/{config}_T{n_frames}/`

## Results Directory Structure

Models are saved to `../results/` with the following structure:

```
results/
├── small/                     # Sparse models
│   ├── models/
│   │   └── Rockpool_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth
│   ├── graphs/
│   │   └── Rockpool_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.png
│   └── experiment_counter.txt
└── large/                     # Dense models
    ├── models/
    │   └── Rockpool_Non_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth
    ├── graphs/
    │   └── Rockpool_Non_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.png
    └── experiment_counter.txt
```

## Extension Guide

To add a new dataset:

1. Create `datasets/new_dataset.py`:
   ```python
   from core.base_dataset import NeuromorphicDataset

   class NewDataset(NeuromorphicDataset):
       def _get_transforms(self): ...
       def _load_raw_dataset(self, train=True): ...
       def _get_cache_path(self): ...
       def get_num_classes(self): ...
   ```

2. Add to `datasets/__init__.py`:
   ```python
   from .new_dataset import load_new_dataset
   DATASET_REGISTRY['NewDataset'] = load_new_dataset
   ```

To add a new model:

1. Create `models/new_model.py`:
   ```python
   from core.base_model import BaseSNNModel

   class NewModel(BaseSNNModel):
       def _build_network(self): ...
       def _prepare_input(self, data): ...
       def _get_save_params(self): ...
   ```

2. Add to `models/__init__.py`:
   ```python
   from .new_model import NewModel
   MODEL_REGISTRY['NewModel'] = NewModel
   ```

## Next Steps

1. Train sparse and dense models on full SHD (700 features)
2. Train models with 16-band binning
3. Compare sparse vs. dense spike counts and accuracy
4. Implement Xylo conversion (`.to_xylo_compatible()`)
5. Test on XyloSim simulator for energy estimates
6. Deploy to actual Xylo hardware

## References

- Rockpool: https://rockpool.ai/
- Xylo: https://synsense.ai/products/xylo/
- SHD Dataset: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
- Tonic: https://tonic.readthedocs.io/
