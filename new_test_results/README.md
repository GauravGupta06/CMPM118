# Fresh Training Runs

This folder is for new clean training outputs only. Keep old `results/` and
`workspace/` artifacts as historical context, but do not mix new runs into them.

## UCI-HAR First Baseline

The current UCI-HAR trainer only has `dense` and `sparse` model modes. For the
first fresh baseline, use `dense` with no spike penalty as the paper/reference
baseline unless the source paper specifies different neuron parameters.

Run from the repository root:

```bash
.venv/bin/python train_UCI_HAR.py \
  --model_type dense \
  --n_frames 128 \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001 \
  --dataset_path ./data \
  --output_path ./new_test_results \
  --num_workers 0
```

On the GPU cluster, increase workers if the environment supports it:

```bash
python train_UCI_HAR.py \
  --model_type dense \
  --n_frames 128 \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001 \
  --dataset_path ./data \
  --output_path ./new_test_results \
  --num_workers 4
```

Expected output location:

```text
new_test_results/uci_har/dense/
```

Before using this run for energy-per-spike calibration, verify the source paper's
architecture, preprocessing, timestep, and neuron parameters match the trainer.

## DVSGesture Paper-Matched Baseline

The DVSGesture baseline is matched to Arfa et al., "Efficient Deployment of
Spiking Neural Networks on SpiNNaker2 for DVS Gesture Recognition Using
Neuromorphic Intermediate Representation." Use the full-precision P-SNN
baseline, not the quantized Q-SNN, for the reference spike count.

The paper-matched settings are 32x32 frames, 1 ms time-window slicing, summed
frame counts, 600 timestep cap, no conv/linear biases, SumPool layers, snnTorch
Leaky neurons with beta 0.93, threshold 1, fast-sigmoid surrogate slope 9.70,
subtract reset, batch size 32, MSE loss, Adam, gradient/weight clipping, and
200 max epochs. The paper reports 0.765 mJ per 1 ms frame, so the default
reference energy is 459 mJ/gesture at 600 timesteps.

Note: the released author code uses `Denoise(filter_time=10000)` even though
the paper prose describes the denoising less precisely. The local cache path
includes the denoise setting to avoid mixing old preprocessing runs.

On the GPU cluster:

```bash
python -u train_dvsgesture.py \
  --model_type reference \
  --w 32 \
  --h 32 \
  --max_timesteps 600 \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.0024 \
  --dataset_path ./data \
  --output_path /workspace/new_test_results \
  --num_workers 4 \
  --denoise_filter_time 10000 \
  --print_every 15 \
  --validate_every 1 \
  --energy_batch_size 32 \
  --paper_energy_per_timestep_mJ 0.765 \
  --early_stop_patience 20 \
  --min_delta 0.001
```

Only add `--binarize` for legacy experiments. The paper-matched path keeps
summed count frames.

Expected cache location:

```text
data/dvsgesture/32x32_tw1ms_T600_count_denoise10000/
```

Expected output location:

```text
new_test_results/dvsgesture/reference/
```

## SHD Paper-Matched Baseline

The SHD paper we matched against is Mészáros et al., "A Complete Pipeline for
deploying SNNs with Synaptic Delays on Loihi 2." The Rockpool trainer here
matches the no-delay feedforward baseline as closely as this codebase supports:
700 input channels, two 512-neuron LIF hidden layers, non-spiking output
integrator, 1 ms timestep, and 14 Hz target hidden firing rate.

The paper's learned-delay model uses EventProp in mlGeNN. This repository's
Rockpool trainer does not implement those learned per-synapse delays, so use the
paper's no-delay feedforward energy value, 0.42 mJ/inference, for this command.

Run from the repository root:

```bash
.venv/bin/python train_shd_paper.py \
  --model_type baseline \
  --n_frames 1400 \
  --net_dt 0.001 \
  --tau_mem 0.02 \
  --tau_syn 0.005 \
  --epochs 30 \
  --batch_size 32 \
  --lr 0.001 \
  --dataset_path ./data \
  --output_path ./new_test_results \
  --num_workers 0 \
  --print_every 50 \
  --validate_every 1 \
  --energy_batch_size 32 \
  --paper_energy_mJ 0.42 \
  --early_stop_patience 5 \
  --min_delta 0.002 \
  --target_test_acc 0.80
```

On the GPU cluster:

```bash
python train_shd_paper.py \
  --model_type baseline \
  --n_frames 1400 \
  --net_dt 0.001 \
  --tau_mem 0.02 \
  --tau_syn 0.005 \
  --epochs 30 \
  --batch_size 128 \
  --lr 0.001 \
  --dataset_path ./data \
  --output_path /workspace/new_test_results \
  --num_workers 4 \
  --print_every 10 \
  --validate_every 1 \
  --energy_batch_size 128 \
  --paper_energy_mJ 0.42 \
  --early_stop_patience 5 \
  --min_delta 0.002 \
  --target_test_acc 0.80
```

Only add `--rebuild_cache` when you intentionally want to delete and rebuild the
selected SHD cache. A normal training restart should reuse the existing cache.

Expected new cache location:

```text
data/shd/700ch_dt1ms_T1400/
```

Expected output location:

```text
new_test_results/shd/baseline/
```
