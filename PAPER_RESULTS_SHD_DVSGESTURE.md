# SHD and DVSGesture Paper Results Handoff

Generated from the local CMPM118 artifacts after rerunning the main checks on
2026-05-24. UCI HAR is intentionally excluded here because the current plan is
not to report UCI HAR as a routed result.

## High-Level Paper Framing

The core result is that two SNNs with the same architecture but different
sparsity/energy behavior can be used together through a lightweight routing
decision to reduce energy while keeping accuracy close to the high-accuracy
model.

- SHD has the cleanest routed result: an input-prefix spike-count router measured
  on STM32 gives real net energy savings after router overhead.
- DVSGesture has two stories:
  - Pure input-complexity metrics were weak separators.
  - The best result came from a model-assisted sparse-prefix confidence router.
    This should be framed clearly as an early-confidence/adaptive low-power mode,
    not as a pure pre-inference entropy metric.

## Verification Status

The following main results were rerun locally on 2026-05-24:

- SHD measured STM32 prefix router: matched the paper numbers.
- DVSGesture measured STM32 binary-popcount prefix router: matched the saved
  numbers.
- DVSGesture sparse-prefix confidence metric search: matched the AUC and routed
  accuracy numbers.
- DVSGesture sparse-prefix confidence energy operating point: matched the
  reported `31.006 mJ` average model energy and `15.418 mJ` saved.
- DVSGesture coarse-LZC: matched when using the original measured/heavy
  STM32 energy scale.

## SHD Models

### Architecture and Model Details

| Item | Value |
|---|---|
| Dataset | Spiking Heidelberg Digits (SHD) |
| Model file/class | `models/shd_model_paper.py`, `SHDSNN` |
| Input size | 700 channels |
| Timesteps | 1400 at 1 ms |
| Hidden size | 512 |
| Output classes | 20 |
| Neuron parameters | `tau_mem=0.02`, `tau_syn=0.005`, threshold `1.0` |
| Bias | Enabled |
| Delay mode | None |
| Approx. trainable weight/bias parameters | 631,828 |
| Dense model path | `new_test_results/shd/baseline/models/best_baseline_T1400_dt0.001.pth` |
| Sparse model path | `new_test_results/shd/sparse/models/best_sparse_target6_epoch9_acc7652_preserved.pth` |

### SHD Dense/Sparse/Routed Result

| Setup | Accuracy | Avg spikes/sample | Avg energy/sample | Notes |
|---|---:|---:|---:|---|
| Dense | 82.77% | 27,734.65 | 0.420 mJ | High-accuracy reference model |
| Sparse | 76.41% | 14,155.82 | 0.214 mJ | Same architecture, stronger sparsity |
| Routed | 81.45% | 20,194.73 routed-model spikes | 0.336 mJ total | Includes STM32 router overhead |

### SHD Router Details

| Metric | Value |
|---|---:|
| Router metric | 400 ms prefix spike count |
| Router rule | Route to dense if prefix spike count `>= 6993` |
| AUC for dense-needed samples | 0.645 |
| Routed to dense | 41.5% |
| Routed to sparse | 58.5% |
| Accuracy loss vs dense | 1.32 percentage points |
| Energy saved vs dense | 0.084 mJ/sample |
| Percent energy saved vs dense | 20.06% |
| STM32 router cycles/sample | 166,303.54 |
| STM32 router energy/sample | 0.0299 mJ |
| Router energy as fraction of dense | about 7.13% |

### SHD Energy Calibration

| Item | Value |
|---|---:|
| Dense paper/reference energy | 0.420 mJ/inference |
| Dense avg spikes/sample | 27,734.65 |
| Energy per spike | `1.5143509308254347e-08 J/spike` |

### SHD Artifact Paths

| Artifact | Path |
|---|---|
| Latest verified summary JSON | `new_test_results/shd/prefix_router_eval/shd_prefix_router_summary_20260524_192849.json` |
| Latest verified per-sample CSV | `new_test_results/shd/prefix_router_eval/shd_prefix_router_per_sample_20260524_192849.csv` |
| STM32 measured router energy file | `Prefix_Router_Energy/prefix_router_binary_energy_SHD_T1400_dt1ms_p400ms.txt` |
| STM32 expected prefix file | `Prefix_Router_Energy/prefix_router_binary_expected_SHD_T1400_dt1ms_p400ms.csv` |

## DVSGesture Models

### Architecture and Model Details

| Item | Value |
|---|---|
| Dataset | DVS Gesture |
| Model file/class | `models/dvsgesture_model.py`, `DVSGestureSNN` |
| Architecture | Same Conv-SNN architecture for dense and sparse |
| Input | 32x32 DVS count-frame style input, two polarities |
| Important input note | The model training input is count-frame data, not purely binary-clipped input |
| Timesteps | 600 at 1 ms |
| Output classes | 11 |
| Trainable weights | 25,504 |
| Bias | Disabled |
| Neuron parameters | `beta=0.93`, threshold `1.0`, surrogate slope `9.70`, subtract reset |
| Dense/high-accuracy model path | `new_test_results/dvsgesture/dense/models/current_new_dense_acc8906_spikes26024.pth` |
| Sparse model path | `new_test_results/dvsgesture/new_sparse/models/accepted_new_sparse_acc7188_spikes6314.pth` |

### DVSGesture Dense/Sparse Model Result

| Setup | Accuracy | Avg spikes/sample | Avg energy/sample | Notes |
|---|---:|---:|---:|---|
| Dense / high-accuracy model | 89.39% | 26,024.25 | 46.424 mJ | Current DVS dense/reference-for-routing model |
| Sparse | 71.97% | 6,314.15 | 11.264 mJ | Accepted lower-power sparse model |

### DVSGesture Energy Calibration

These energy numbers are estimated from a paper/reference energy-per-spike
calibration, not direct hardware measurement of the full models.

| Item | Value |
|---|---:|
| Paper/reference energy | 459 mJ/inference |
| Reference avg spikes used for calibration | 257,304.70 spikes/sample |
| Energy per spike | `1.7838772422290688e-06 J/spike` |

## DVSGesture Routing Results

### Summary of Routing Options

| Router / Metric | AUC | Routed Accuracy | Routed to Dense | Energy Result | Verdict |
|---|---:|---:|---:|---:|---|
| Binary prefix occupancy popcount, 400 ms | 0.482 | 79.55% | 45.8% | 31.078 mJ total, 33.06% saved | Energy good, accuracy bad |
| Coarse-LZC 640-bit prefix | 0.571 flipped | 85.61% at useful threshold | 60.6% | 37.015 mJ total, 20.27% saved | Better, but metric direction/quality weak |
| Sparse-prefix confidence margin, 460 ms | 0.771 | 88.64% | 58.3% | 31.006 mJ model energy, 15.418 mJ saved | Best DVS result so far |

### DVSGesture Binary Popcount Router

This was the pure input-prefix STM32 metric.

| Metric | Value |
|---|---:|
| Router metric | Binary prefix occupancy popcount |
| Prefix | First 400 ms |
| Input to STM32 metric | Binary-clipped prefix, `count > 0` |
| Bits/sample | 819,200 bits |
| Packed bytes/sample | 102,400 bytes |
| Router rule | Route to dense if binary prefix ones `>= 10000` |
| AUC for dense-needed samples | 0.482 |
| Routed accuracy | 79.55% |
| Routed to dense | 45.8% |
| Routed to sparse | 54.2% |
| Dense avg energy/sample | 46.424 mJ |
| Sparse avg energy/sample | 11.264 mJ |
| Routed total energy/sample | 31.078 mJ |
| Energy savings vs dense | 33.06% |
| STM32 router cycles/sample | 435,340.83 |
| STM32 router energy/sample | 78.361 uJ |

Artifact paths:

| Artifact | Path |
|---|---|
| Latest verified summary JSON | `new_test_results/dvsgesture/prefix_router_eval/dvsgesture_prefix_router_summary_20260524_193020.json` |
| Latest verified per-sample CSV | `new_test_results/dvsgesture/prefix_router_eval/dvsgesture_prefix_router_per_sample_20260524_193020.csv` |
| STM32 measured router energy file | `Prefix_Router_Energy/dvsgesture_prefix_router_binary_energy_DVSGesture_T600_dt1ms_p400ms_32x32_binary.txt` |
| STM32 expected prefix file | `Prefix_Router_Energy/dvsgesture_prefix_router_binary_expected_DVSGesture_T600_dt1ms_p400ms_32x32_binary.csv` |

### DVSGesture Coarse-LZC Router

This was a compact input-complexity metric. It performed better than raw
popcount, but its AUC was still weak and the direction was unintuitive.

Metric definition:

1. Use the first 400 ms of DVS Gesture input.
2. Convert count frames to binary occupancy with `count > 0`.
3. Pool time into 20 bins of 20 ms each.
4. Pool 32x32 pixels into a 4x4 spatial grid.
5. Keep polarity separate.
6. Flatten `[20 time bins, 2 polarities, 4, 4]` into a 640-bit sequence.
7. Compute Lempel-Ziv complexity on the 640-bit sequence.

| Metric | Value |
|---|---:|
| AUC for dense-needed samples | 0.429 raw, 0.571 flipped |
| Useful route direction | Route dense for lower LZC |
| Threshold for 85%+ accuracy point | Route dense if LZC `<= 46` |
| Routed accuracy at that point | 85.61% |
| Routed to dense at that point | 60.6% |
| Avg energy/sample at that point | 37.015 mJ |
| Energy savings vs dense | 20.27% |
| Higher-accuracy operating point | 88.64% accuracy at LZC `<= 53` |
| Higher-accuracy energy result | -0.08% savings, effectively no net energy win |
| STM32 coarse-LZC measured cycles/sample | about 21,798,389 |
| STM32 coarse-LZC energy/sample | about 3.924 mJ |

Artifact paths:

| Artifact | Path |
|---|---|
| Latest verified summary JSON | `new_test_results/dvsgesture/lzc_router_eval/dvsgesture_coarse_lzc640_router_summary_20260524_193501.json` |
| Latest verified per-sample CSV | `new_test_results/dvsgesture/lzc_router_eval/dvsgesture_coarse_lzc640_router_20260524_193501.csv` |
| STM32 coarse-LZC energy summary | `Prefix_Router_Energy/dvsgesture_coarse_lzc_energy_DVSGesture_T600_dt1ms_p400ms_20tb_4x4_summary.json` |

### DVSGesture Sparse-Prefix Confidence Router

This is the strongest DVS routing result, but it is model-assisted. It runs the
sparse model on an initial prefix, checks confidence, and routes to dense if the
sparse model is not confident enough.

| Metric | Value |
|---|---:|
| Prefix | 460 ms sparse-model prefix |
| Router score | Low sparse-prefix confidence margin |
| AUC for dense-needed samples | 0.771 |
| Selected operating point | Best routed accuracy with dense fraction `<= 60%` |
| Routed accuracy | 88.64% |
| Dense accuracy | 89.39% |
| Accuracy loss vs dense | 0.76 percentage points |
| Routed to dense | 58.3% |
| Routed to sparse | 41.7% |
| Avg routed model energy/sample | 31.006 mJ |
| Avg energy saved vs dense | 15.418 mJ/sample |
| Percent model-energy savings | 33.21% |

Important caveat for the paper: this is not a pure pre-inference
input-complexity metric. It should be described as a model-assisted
early-confidence router or adaptive sparse-prefix decision. The energy number
above is the verified model-routing energy from dense/sparse model energy
columns. If the final implementation literally runs the sparse prefix and then
starts the dense model from scratch, the paper should explicitly account for
that prefix overhead.

Artifact paths:

| Artifact | Path |
|---|---|
| Metric search summary | `DVS_Routing_Metric_Search/dvsgesture_router_metric_search_summary.json` |
| Metric search table | `DVS_Routing_Metric_Search/dvsgesture_router_metric_search_table.csv` |
| Sparse prefix confidence cache | `DVS_Routing_Metric_Search/sparse_prefix_confidence_cache_p460.npz` |
| Energy verifier script | `DVS_Routing_Metric_Search/verify_sparse_prefix_confidence_energy.py` |

## Recommended Paper Tables

### Main SHD Table

Use the SHD measured-prefix router as the main clean result:

| Dataset | Dense Acc. | Sparse Acc. | Routed Acc. | Routed to Dense | Dense Energy | Routed Energy | Energy Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| SHD | 82.77% | 76.41% | 81.45% | 41.5% | 0.420 mJ | 0.336 mJ | 20.06% |

### Main DVSGesture Table

If the paper allows model-assisted routing, use the sparse-prefix confidence
router:

| Dataset | Dense Acc. | Sparse Acc. | Routed Acc. | AUC | Routed to Dense | Dense Energy | Routed Energy | Energy Saved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DVSGesture | 89.39% | 71.97% | 88.64% | 0.771 | 58.3% | 46.424 mJ | 31.006 mJ | 33.21% |

If the paper needs a pure input-complexity router only, DVSGesture should be
reported more cautiously:

- Binary popcount gives energy savings but loses too much accuracy.
- Coarse-LZC can reach 85.61% with 20.27% savings, but AUC is weak.
- Coarse-LZC can reach 88.64% accuracy, but then energy savings disappear.

## Main Caveats

1. SHD is the strongest fully clean story because the router is a cheap
   pre-inference input-prefix metric measured on STM32.
2. DVSGesture input-only complexity metrics did not separate dense-needed cases
   well enough.
3. DVSGesture sparse-prefix confidence is the best practical result, but it
   changes the story from pure entropy/complexity routing to model-assisted
   early-confidence routing.
4. DVSGesture model energy is estimated from reference energy-per-spike
   calibration, not from direct hardware measurement of the full DVS models.
5. UCI HAR routing was not run and should not be claimed as a routed result.
