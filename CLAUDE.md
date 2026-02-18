# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SNN Router is a spiking neural network system that dynamically routes inputs between sparse and dense models based on input complexity metrics (Lempel-Ziv Complexity). This achieves energy efficiency similar to mixture-of-experts by sending simple inputs to lightweight sparse models and complex inputs to more accurate dense models.

**Core Framework:** Rockpool (with PyTorch backend) for SNN implementation using LIF neurons.

## Common Commands

### Training Models

```bash
# UCI HAR (Human Activity Recognition)
python train_UCI_HAR.py --model_type dense --n_frames 128 --epochs 100 --batch_size 64 --lr 1e-3

# DVSGesture (Neuromorphic Gesture Recognition)
python train_dvsgesture.py --model_type dense --n_frames 32 --w 32 --h 32 --epochs 200 --batch_size 32

# SHD (Neuromorphic Audio)
python train_shd.py --model_type dense --n_frames 100 --epochs 200 --batch_size 32 --NUM_CHANNELS 700
```

Use `--model_type sparse` for sparse variants. All scripts support `--dataset_path` and `--output_path`.

### Running Router

```bash
python router.py --sparse_model_path <path> --dense_model_path <path> --dataset uci_har|dvsgesture|shd
```

### Testing

```bash
python test_uci_har.py  # Uses hardcoded model path in script
```

### Docker

```bash
docker build -t snn-router:latest .
```

## Architecture

### Data Flow

```
Raw Events → Transform (resize, time-bin, binarize) → Cache (Tonic DiskCachedDataset)
→ DataLoader → Model Forward (record=True) → Extract recordings
→ Compute metrics (LZC, spikes) → Router decision
```

### Model Structure

All models in `models/` use Rockpool's `Sequential` with:
- `LinearTorch` layers for weights
- `LIFTorch` neurons with configurable tau_mem, threshold, spike_lam
- `ExpSynTorch` for output layer (critical for gradient stability)
- Forward pass uses `record=True` to capture layer-wise spike recordings

**Sparse vs Dense distinction:**
- Sparse: Lower tau_mem, higher threshold, spike_lam > 0 → fewer spikes
- Dense: Higher tau_mem, lower threshold, spike_lam = 0 → more spikes, higher accuracy

### Router Logic (`router.py`)

Computes Lempel-Ziv Complexity on input spike patterns to estimate input difficulty. Uses ROC curve analysis to find optimal threshold for routing decisions.

### Datasets (`datasets/`)

Each dataset module provides `get_dataloaders()` returning train/test DataLoaders:
- **uci_har.py**: 9 sensor channels → 6 activity classes
- **dvsgesture_dataset.py**: 32x32 DVS frames → 11 gesture classes
- **shd_dataset.py**: 700 frequency channels → 20 audio classes

## Key Hyperparameters

| Dataset | Time Steps | Sparse tau_mem | Dense tau_mem | Sparse threshold | Dense threshold |
|---------|------------|----------------|---------------|------------------|-----------------|
| UCI HAR | 128 | 0.005 | 0.1 | 2.0 | 0.5 |
| DVSGesture | 32 | - | 0.1 | - | 0.5 |
| SHD | 100 | - | 0.1 | - | 0.5 |

## Important Implementation Notes

- Always use `ExpSynTorch` for the final output layer to avoid exploding gradients
- Add small bias (0.01) to LIF layers to prevent dead neurons
- DVSGesture training requires very low learning rate (~1e-5) with weight downscaling
- Models save hyperparameters in checkpoint dict for reproducibility
- Use `model.reset_state()` between batches for proper temporal processing