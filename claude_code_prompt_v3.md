# TASK: Rewrite DVSGesture SNN from FC to Conv Architecture (Staying in Rockpool) + Update Pipeline

## OVERVIEW

I have an SNN research project that routes DVSGesture inputs between sparse and dense models using Lempel-Ziv complexity. The DVSGesture model currently uses a fully-connected architecture in Rockpool. I need to change it to a **convolutional architecture matching a reference paper** while staying in the Rockpool framework.

**Read `paper.pdf` first** — it is Arfa et al. (2025), "Efficient Deployment of Spiking Neural Networks on SpiNNaker2 for DVS Gesture Recognition Using Neuromorphic Intermediate Representation." Extract and use the architecture and hyperparameter details from that paper.

---

## FILES TO MODIFY/CREATE

| File | Action |
|------|--------|
| `models/dvsgesture_model.py` | **REPLACE** — conv architecture matching paper, add energy methods |
| `datasets/dvsgesture_dataset.py` | **REPLACE** — time_window=1ms bins, 600 timestep cap, no flattening, add denoising |
| `router.py` | **MODIFY** — framework-agnostic energy/spike handling, proper dataset switching |
| `train_dvsgesture.py` | **MODIFY** — update to use new model/dataset interface and paper hyperparams |
| `models/shd_model.py` | **DO NOT TOUCH** |
| `datasets/shd_dataset.py` | **DO NOT TOUCH** |

---

## REFERENCE PAPER DETAILS (verify against paper.pdf)

### Target Architecture (Table I in paper)
```
Input:        2 × 32 × 32
Conv2D (0):   kernel=5×5, stride=2, padding=1, out_channels=16  → 16×15×15
LIF (1):      16×15×15
Conv2D (2):   kernel=3×3, stride=1, padding=1, out_channels=16  → 16×15×15
LIF (3):      16×15×15
SumPool (4):  kernel=2×2, stride=2                               → 16×7×7
Conv2D (5):   kernel=3×3, stride=1, padding=1, out_channels=8   → 8×7×7
LIF (6):      8×7×7
SumPool (7):  kernel=2×2, stride=2                               → 8×3×3
Flatten (8):  72
Linear (9):   72 → 256
LIF (10):     256
Linear (11):  256 → 11
LIF (12):     11
Output:       11 classes
```

**Padding math verification (do this before writing code):**
- Conv0: (32 + 2*1 - 5)/2 + 1 = 15 ✓
- Conv2: (15 + 2*1 - 3)/1 + 1 = 15 ✓
- Pool4: 15//2 = 7 ✓
- Conv5: (7 + 2*1 - 3)/1 + 1 = 7 ✓
- Pool7: 7//2 = 3 ✓
- Flatten: 8*3*3 = 72 ✓

### Hyperparameters (Table III in paper)
- **Beta (decay rate):** 0.93 → convert to tau_mem for Rockpool: `tau_mem = -dt / ln(0.93)` where dt=0.001 → tau_mem ≈ 0.01378s
- **NO tau_syn.** The paper uses a single-state LIF neuron (snn.Leaky) with only membrane decay, no separate synaptic time constant. In Rockpool's LIFTorch, either remove tau_syn entirely or set it to a negligibly small value equal to dt (0.001) so the synaptic filter is effectively a passthrough. **Do NOT use ExpSynTorch either** — it's not needed. If LIFTorch requires tau_syn as a parameter, set it to dt and treat it as a non-parameter. Remove tau_syn from all hyperparameter dicts, constructor signatures, and saved checkpoints.
- **Threshold:** 1.0
- **Bias:** False (no bias in conv or linear layers). LIF neuron bias = 0.
- **Reset mechanism:** subtract
- **Learning rate:** 0.003
- **Optimizer:** Adam
- **Loss function:** MSE on output spike counts vs one-hot targets
- **Batch size:** 32
- **Epochs:** 200
- **Total parameters:** 25,504

### Preprocessing (Sections II-D and III of paper)
- DVSGesture via Tonic
- Spatial downsampling 128×128 → 32×32
- 2 polarity channels preserved → input shape [T, 2, 32, 32]
- **Time-window binning: 1ms per frame** (use `time_window=1000` in Tonic's ToFrame, since DVS timestamps are in microseconds)
- **Cap at 600 timesteps maximum.** The paper truncated to ~600 timesteps on SpiNNaker2 due to SRAM limits and still achieved 94% accuracy. We match this. Add a transform step that crops the temporal dimension to at most 600 frames. If a sample has fewer than 600 frames, leave it as-is (PadTensors will handle padding in the DataLoader).
- Denoising: filter isolated events — use Tonic's denoise transform with appropriate params (paper says: events removed if no other events within 1px spatial and 1s temporal neighborhood)
- Binarize frames: (x > 0).astype(float32)
- Do NOT flatten — output must be [T, 2, 32, 32] where T ≤ 600

### Energy Numbers (Table VII and Section IV-C of paper)
- Energy per full gesture inference: **459 mJ** (for ~600 timesteps on SpiNNaker2)
- Energy per 1ms frame (per timestep): **0.765 mJ**
- Hardware: SpiNNaker2, 300 MHz, 0.8V, 147 PEs
- Since we use 1ms bins capped at 600 timesteps — exactly matching their on-chip setup — the 0.765 mJ/timestep number applies directly.

---

## FILE 1: `models/dvsgesture_model.py` — REPLACE

### Architecture

Since Rockpool's `Sequential` combinator expects Rockpool modules, and we need to mix `nn.Conv2d` (PyTorch) with `LIFTorch` (Rockpool), **write a manual forward pass** instead of using `Sequential`. Make the model class an `nn.Module` subclass that contains both PyTorch layers and Rockpool LIF layers.

The architecture must match Arfa et al. exactly:
```python
# Conv block 1
self.conv0 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=1, bias=False)
self.lif0 = LIFTorch(...)  # 16*15*15 = 3600 neurons
# Conv block 2  
self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
self.lif1 = LIFTorch(...)  # 16*15*15 = 3600 neurons
self.pool0 = nn.AvgPool2d(kernel_size=2, stride=2)  # approximate SumPool
# Conv block 3
self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
self.lif2 = LIFTorch(...)  # 8*7*7 = 392 neurons
self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
# FC block
self.flat = nn.Flatten()
self.fc0 = nn.Linear(72, 256, bias=False)
self.lif3 = LIFTorch(...)  # 256 neurons
self.fc1 = nn.Linear(256, 11, bias=False)
self.lif4 = LIFTorch(...)  # 11 neurons (output layer)
```

### LIF Configuration — NO tau_syn
```python
# Convert beta=0.93 to tau_mem:
# beta = exp(-dt/tau_mem) → tau_mem = -dt/ln(beta)
# With dt=0.001: tau_mem = -0.001/ln(0.93) ≈ 0.01378

tau_mem = 0.01378  # equivalent to beta=0.93 with dt=0.001
dt = 0.001
threshold = 1.0

# For LIFTorch calls:
# If LIFTorch requires tau_syn, set it to dt (0.001) so it's a passthrough.
# Set LIF bias to 0 (paper uses no bias).
# Use Constant() wrapper for all LIF params.
```

### Forward Pass Design

**IMPORTANT — LIFTorch shape handling:**
Rockpool's LIFTorch expects input shape `(batch, timesteps, features)`. For conv layers, you need to handle the spatial dimensions yourself.

**Recommended approach:** Process all timesteps through conv layers first, collecting into `[batch, T, C*H*W]`, then feed entire sequence to LIFTorch at once (letting it handle temporal state internally):

```python
def forward(self, x):
    # x shape: [batch, T, 2, 32, 32]
    batch, T, C, H, W = x.shape
    
    # Reset all LIF states
    # ... reset each lif layer ...
    
    # Process each timestep through conv layers, collect for LIF
    conv0_out = []
    for t in range(T):
        frame = x[:, t]  # [batch, 2, 32, 32]
        out = self.conv0(frame)  # [batch, 16, 15, 15]
        conv0_out.append(out.reshape(batch, -1))  # [batch, 3600]
    conv0_seq = torch.stack(conv0_out, dim=1)  # [batch, T, 3600]
    
    lif0_out, _, lif0_rec = self.lif0(conv0_seq, record=True)  # [batch, T, 3600]
    
    # Reshape back to spatial for next conv
    # ... continue pattern for each layer ...
```

OR process per-timestep all the way through. Either approach works — pick whichever is cleaner. **Test with a dummy input before proceeding.**

### Class Interface

**Constructor — remove tau_syn:**
```python
def __init__(self, input_size=None, n_frames=600, tau_mem=0.01378, spike_lam=1e-7,
             model_type="dense", device=None, num_classes=11, lr=0.003, dt=0.001, 
             threshold=1.0, has_bias=False):
```
- `input_size` kept for backward compat but not used in conv arch
- `tau_syn` is GONE from the signature
- Add `self.architecture = 'conv_arfa'`

**Keep all existing methods with updated logic:**

`train_model(self, train_loader, test_loader, num_epochs=200, print_every=15)`:
- Use **MSE loss** instead of CrossEntropyLoss
- Target: one-hot encoded labels → shape `(batch, num_classes)` as float
- Prediction: sum of output LIF spikes over all timesteps → shape `(batch, num_classes)`
- Loss = MSE(spike_counts, one_hot_target) + spike_lam * total_spikes_all_layers
- Keep gradient clipping (max_norm=1.0), Adam optimizer, ReduceLROnPlateau scheduler

`validate_model(self, test_loader)`:
- Prediction = argmax of summed output spikes over time

`save_model(self, base_path, counter_file)`:
- Same plot generation, same counter system

`load_model(self, model_path)`:
- Handle both old and new checkpoint formats gracefully

`load_hyperparams(model_path, device)` (static method):
- Same as before

**New methods:**

```python
def count_spikes(self, data):
    """
    Run inference and return total spike count per sample.
    Args: data — input tensor [batch, T, C, H, W]
    Returns: list of ints, one per sample in batch
    """

def estimate_energy(self, data, method='per_timestep'):
    """
    Estimate energy per sample based on Arfa et al. SpiNNaker2 measurements.
    
    Methods:
        'per_timestep': energy = num_actual_timesteps * 0.765e-3 J
            (num_actual_timesteps = number of non-padding timesteps per sample)
        'per_spike': energy = total_spikes * self.ENERGY_PER_SPIKE
    
    Returns: list of floats (Joules), one per sample
    """

def run_inference(self, data, record=False):
    """
    Framework-agnostic inference entry point for the router.
    Returns: (logits, spike_count)
        logits: [batch, num_classes] — summed output spikes
        spike_count: int total spikes if record=True, else 0
    """
```

Set `self.ENERGY_PER_SPIKE = None` as default. For `per_spike` method, raise error if not calibrated.
Set `self.ENERGY_PER_TIMESTEP = 0.765e-3` as class constant (from paper).

### Save Format
```python
{
    'state_dict': model_state_dict,
    'hyperparams': {
        'input_size': self.input_size,
        'n_frames': self.n_frames,
        'tau_mem': self.tau_mem,
        'spike_lam': self.spike_lam,
        'model_type': self.model_type,
        'num_classes': self.num_classes,
        'dt': self.dt,
        'threshold': self.threshold,
        'has_bias': self.has_bias,
        'architecture': 'conv_arfa',
        'beta': 0.93,
        # NOTE: tau_syn is intentionally absent
    }
}
```

---

## FILE 2: `datasets/dvsgesture_dataset.py` — REPLACE

### Changes from current version:
1. **Remove the flatten transform** — output shape is `[T, 2, 32, 32]` not `[T, 2*32*32]`
2. **Switch from `n_time_bins=32` to `time_window=1000`** (1ms bins)
3. **Add denoising** before downsampling
4. **Add 600 timestep cap** — crop temporal dimension to max 600 frames
5. **Keep binarization**
6. **Update cache path** to avoid conflicts with old cache

### Transform pipeline:
```python
transform = tonic.transforms.Compose([
    tonic.transforms.Denoise(filter_time=1000000),  # 1s in µs — verify Tonic API
    tonic.transforms.Downsample(spatial_factor=(self.w/self.sensor_size[1], self.h/self.sensor_size[0])),
    tonic.transforms.ToFrame(sensor_size=(self.w, self.h, 2), time_window=1000),  # 1ms bins
    lambda x: (x > 0).astype(np.float32),  # Binarize
    lambda x: x[:600],  # Cap at 600 timesteps — matches paper's on-chip setup
    # NO flatten — keep [T, 2, 32, 32] shape where T ≤ 600
])
```

### Class interface:
```python
class DVSGestureDataset:
    def __init__(self, dataset_path, w=32, h=32, max_timesteps=600):
        # n_frames is replaced by max_timesteps
    def load_dvsgesture(self):  # returns (cached_train, cached_test)
    def get_num_classes(self):  # returns 11
```

### IMPORTANT: Variable-length sequences
Even with the 600 cap, samples will have different T values (some gestures are shorter than 600ms). The DataLoader **must** use `tonic.collation.PadTensors(batch_first=True)` as collate_fn. Add a clear comment documenting this requirement.

---

## FILE 3: `router.py` — MODIFY

### Goals:
1. Work with both Rockpool SHD models (FC, old interface) and new DVSGesture model (conv, new interface)
2. Use model.run_inference() when available, fall back to old model.net() for SHD
3. Use model.estimate_energy() when available, fall back to spike-count proxy for SHD
4. Replace comment-toggling with proper --dataset argument

### Add arguments:
```python
parser.add_argument('--dataset', type=str, required=True, choices=['shd', 'dvsgesture'])
parser.add_argument('--energy_method', type=str, default='per_timestep', 
                    choices=['per_timestep', 'per_spike', 'spike_proxy'])
```

### Replace comment-toggle blocks with if/else on args.dataset

### Add framework-agnostic helpers:

```python
def get_predictions_and_spikes(model, data):
    """Works with both old Rockpool interface and new conv model interface."""
    if hasattr(model, 'run_inference'):
        logits, spike_count = model.run_inference(data, record=True)
        preds = logits.argmax(dim=1)
        return preds, spike_count
    else:
        # Rockpool fallback (SHD models)
        output, _, recording = model.net(data, record=True)
        logits = output.mean(dim=1)
        preds = logits.argmax(dim=1)
        spike_count = count_spikes_from_recording(recording)
        return preds, spike_count

def get_energy_estimate(model, data, method='per_timestep', precomputed_lzc_energy=0):
    """Falls back to spike proxy if model doesn't have estimate_energy."""
    if hasattr(model, 'estimate_energy') and method != 'spike_proxy':
        energies = model.estimate_energy(data, method=method)
        return [e + precomputed_lzc_energy for e in energies]
    else:
        _, spike_count = get_predictions_and_spikes(model, data)
        model_energy = spike_count * ENERGY_PER_SPIKE
        return [model_energy + precomputed_lzc_energy]
```

### Keep all existing functions — they're still needed for SHD fallback:
- `count_spikes_from_recording()`
- `compute_lzc_from_events()`
- `load_lzc_from_file()`
- `threshold_sweep_and_roc()`
- `evaluate_models_on_dataset()`
- `route_and_evaluate()`

### Update evaluate_models_on_dataset and route_and_evaluate to use the new helpers

### For DVSGesture DataLoader in router: use PadTensors collate_fn

---

## FILE 4: `train_dvsgesture.py` — MODIFY (file already exists)

The file exists with the current FC model interface. Update it to work with the new conv model and dataset.

### Key changes needed:
1. **Remove tau_syn** from model construction
2. **Remove input_size** calculation (conv model doesn't use it) — pass None or a dummy value
3. **Remove n_frames** from dataset constructor — use max_timesteps=600 instead
4. **Add PadTensors collate_fn** to both DataLoaders
5. **Update default hyperparams** to match paper:
   - lr=0.003
   - tau_mem = -0.001 / math.log(0.93)  (≈ 0.01378)
   - spike_lam: sparse=1e-5, dense=1e-8
   - threshold=1.0
   - dt=0.001
   - has_bias=False
   - epochs=200
   - batch_size=32
6. **Update model construction call** to match new constructor (no tau_syn)
7. **Import tonic** for PadTensors collate_fn
8. **Keep the overall structure** (argparse, device setup, train, save, final eval)

---

## CRITICAL CONSTRAINTS

1. **DO NOT TOUCH `models/shd_model.py` or `datasets/shd_dataset.py`.** They remain as-is.
2. **Test the conv architecture shapes.** After writing the model, create a dummy input of shape `[1, 100, 2, 32, 32]` (batch=1, T=100, C=2, H=32, W=32) and verify the forward pass produces correct output without errors. Print shapes at each stage.
3. **Rockpool LIFTorch expects `(batch, time, features)`.** For conv layers, reshape between spatial `(C, H, W)` and flat `(C*H*W)` around each LIFTorch call. Two approaches:
   - Process all T timesteps through a conv layer, collect outputs into `[batch, T, C*H*W]`, feed to LIFTorch, reshape LIFTorch output back to spatial for next conv. **This is preferred** — it's cleaner and lets LIFTorch manage temporal state.
   - OR loop per-timestep through everything, calling LIFTorch with `[batch, 1, features]` each step.
4. **LIFTorch state reset:** Reset all LIF neuron states at the start of each forward pass.
5. **600 timestep cap is enforced in the dataset**, not the model. The model handles whatever T it receives.
6. **Variable-length sequences:** PadTensors pads with zeros. Zero input to LIF neurons just causes leak, which is fine.
7. **Classification:** Prediction = argmax of total output spike counts over all timesteps.
8. **Device support:** CUDA, MPS (Apple Silicon), CPU.
9. **The model class should subclass `nn.Module`** so state_dict/load_state_dict work normally.
10. **AvgPool2d for SumPool:** Just use `nn.AvgPool2d(kernel_size=2, stride=2)`. Threshold compensates during training. Don't multiply by 4.
11. **Weight initialization:** Small normal `(0, 0.01)` for conv and linear layers.
12. **tau_syn handling:** If LIFTorch absolutely requires tau_syn as a constructor argument, set it to `dt` (0.001). But do NOT store it in hyperparams, do NOT expose it in the model constructor, and do NOT save it in checkpoints. It should be invisible to the user.

## ORDER OF OPERATIONS

1. Read `paper.pdf` and confirm architecture/hyperparameter details.
2. Create `datasets/dvsgesture_dataset.py`.
3. Create `models/dvsgesture_model.py`.
4. **Run a shape verification test** — dummy input, verify forward pass works, print all intermediate shapes.
5. Modify `train_dvsgesture.py`.
6. Modify `router.py`.
7. Run import checks on all files.
