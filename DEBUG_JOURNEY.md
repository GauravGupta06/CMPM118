# SNN Training Debug Journey

## The Problem
When running `python train.py`, training immediately produced:
- Extremely high initial loss (119,005)
- NaN errors after ~20 iterations
- Model accuracy stuck at ~5% (random chance for 20 classes)

---

## Debug Points & What They Revealed

### Debug 1: Output Spike Values
```python
print(f"spk_rec min/max: {spk_rec.min():.4f} / {spk_rec.max():.4f}")
```
**Finding**: `spk_rec max = 2821.0` (should be 0 or 1 for binary spikes)

**What this meant**: LIF neurons were outputting values way larger than 1 per timestep. This is NOT how spiking neurons should behave - they should output 0 (no spike) or 1 (spike).

### Debug 2: Unique Values in LIF Output
```python
print(f"unique values: {torch.unique(lif1_out[0, 0, :10])}")
```
**Finding**: `tensor([0., 2., 5., 6., 7.])` - integers, but NOT binary

**What this meant**: Multiple spikes were happening per timestep because neurons were firing too easily.

### Debug 3: Spike Counts After Summation
```python
print(f"spike_counts range=[{spike_counts.min():.0f}, {spike_counts.max():.0f}]")
```
**Finding**: Range was `[0, 211029]` initially

**What this meant**: When summed over 100 timesteps, total spike counts were astronomical. These values go directly into CrossEntropyLoss.

---

## Root Causes Identified

### Root Cause 1: Input Data Not Preprocessed Correctly
**Discovery**: `tonic.transforms.ToFrame` produces **spike counts per bin**, not binary values.

- If 50 spikes occurred in a frequency bin, input value = 50 (not 1)
- These large input values caused excessive network activity

**Fix**: Binarize input in `_prepare_input()`:
```python
x = (x > 0).float()  # Convert to binary: spike happened or not
```

### Root Cause 2: LIF Threshold Too Low
**Discovery**: Default threshold = 1.0, but Linear layer outputs can be huge (sum of 1400 weighted inputs).

- If Linear output = 100 and threshold = 1, neuron fires ~100 times per timestep
- This is why we saw per-timestep values of 2821 instead of 0/1

**Fix**: Increase threshold to make spiking harder:
```python
threshold = 30  # Instead of 1.0
```

---

## Why NaN Occurred

### The CrossEntropyLoss + Softmax Problem

CrossEntropyLoss internally computes:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

With spike_counts like `[89, 45, 12, ...]`:
- `exp(89) = 4.4 x 10^38` - astronomically large
- `exp(280)` = infinity in float32
- `infinity / infinity = NaN`

**Key insight**: Even though dividing all values by the same number preserves relative rankings mathematically, the intermediate `exp()` calculations overflow BEFORE the division happens.

---

## What Loss Values Mean

| Loss Value | Meaning |
|------------|---------|
| ~3.0 | Random guessing (`-log(1/20) = 3.0` for 20 classes) |
| < 3.0 | Model is learning something |
| > 3.0 | Model is worse than random |
| NaN | Numerical overflow (spike counts too large) |

When loss gets stuck at exactly 3.0, it means the network outputs are all similar (often all zeros), so it predicts equal probability for all classes.

---

## Solutions We Tried (And Why)

### Band-aid Solutions (Avoided)
These would have masked the problem without fixing it:
- `log1p()` to compress large values
- Gradient clipping
- Weight decay
- MSE loss with temporal targets

### Proper Solutions (Implemented)
1. **Input binarization** - Fix the data at the source
2. **Higher threshold** - Make neurons behave biologically (sparse binary spikes)

---

## The Threshold Tuning Journey

| Threshold | Per-timestep Max | Spike Counts Range | Result |
|-----------|------------------|-------------------|--------|
| 1.0 (default) | 3868 | [0, 211029] | NaN |
| 10 | 5.0 | [0, 280] | NaN |
| 30 | 1.0 | [0, 11] | Works, but too sparse |
| 100 | 0.0 | [0, 0] | No spikes at all |

**Sweet spot**: Somewhere between 20-30 where you get binary spikes (max=1) with enough activity to learn.

---

## Key Learnings

### 1. SNNs Output Spikes, Not Activations
Unlike regular neural networks that output continuous activations, SNNs should output binary spikes (0 or 1) per timestep. When you see values > 1, something is misconfigured.

### 2. The Threshold Controls Spike Sparsity
- Low threshold = neurons fire easily = many spikes = potential overflow
- High threshold = neurons rarely fire = too sparse = no learning signal
- Must find balance based on your input magnitudes

### 3. Input Preprocessing Matters
The raw output from `tonic.ToFrame` is spike counts, not binary. If you feed large counts into the network, everything downstream explodes.

### 4. Debug the Data Flow
When training fails, trace the values at each step:
1. What are the input values? (should be 0/1 for spikes)
2. What does each layer output?
3. What are the final spike counts?
4. Are they reasonable for your loss function?

### 5. CrossEntropyLoss Expects Reasonable Logits
Values in range [-10, 10] are safe. Values > 100 will cause softmax overflow. If your spike counts are large, you need to normalize OR fix the root cause.

### 6. Loss = 3.0 Means Random Guessing
For 20-class classification, `-log(1/20) = 3.0`. If your loss stabilizes here, the model has no useful signal to learn from.

---

## Current Status
- Input binarization: Implemented
- Threshold tuning: In progress (finding sweet spot between 20-30)
- Next step: Find threshold that gives binary spikes (max=1) with spike_counts in range ~[0, 50] for meaningful learning

---

## Files Modified
- `models/shd_model.py`: Added input binarization, threshold parameter
- `core/base_model.py`: Added debug prints for spike analysis
