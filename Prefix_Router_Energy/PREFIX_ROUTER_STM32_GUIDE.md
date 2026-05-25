# STM32 Prefix Router Energy Measurement Guide

This folder contains two STM32 measurement paths for the final SHD router:

```text
prefix_400ms_spikes >= 6993 -> dense
otherwise                   -> sparse
```

Use the packed-binary path for the final paper number. It measures the router
from the same binarized SHD raster representation that the SNN model receives.

Use the frame-count path only as a lightweight sanity check. It sends 400
precomputed per-frame spike counts, so it measures only summing those counts,
not computing them from the binary raster.

The goal is to replace the estimated router cost with a measured STM32 cycle count, using the same method as the earlier LZC and Transition Count measurements:

```text
cycles measured by DWT CYCCNT
energy = cycles * 1.8e-10 J/cycle
```

For the final packed-binary measurement, the timed region on STM32 only includes:

```c
popcount 35000 packed bytes from the first 400 ms of [400, 700] binary input
compare against threshold 6993
```

For the lightweight frame-count measurement, the timed region only includes:

```c
sum 400 one-millisecond frame spike counts
compare against threshold 6993
```

USB transfer and string parsing are outside the DWT measurement window.

## Files

```text
Prefix_Router_Energy/prefix_router_stm32.c
Prefix_Router_Energy/prefix_router_stm32_packed_binary.c
Prefix_Router_Energy/measure_prefix_router_energy.py
Prefix_Router_Energy/measure_prefix_router_binary_energy.py
```

`prefix_router_stm32_packed_binary.c` is the recommended firmware for final
measurement.

`prefix_router_stm32.c` is the simpler frame-count firmware.

The Python files generate input data, send samples to the board, validate the
returned score/route, convert cycles to energy, and write a summary JSON.

## Step 1: Generate SHD Packed-Binary Prefix Input

From the repo root:

```bash
python Prefix_Router_Energy/measure_prefix_router_binary_energy.py --dry-run
```

This creates:

```text
Prefix_Router_Energy/prefix_router_binary_input_SHD_T1400_dt1ms_p400ms.hex
Prefix_Router_Energy/prefix_router_binary_expected_SHD_T1400_dt1ms_p400ms.csv
```

Each input line is one SHD test sample. It contains the first 400 ms of the
model input as packed binary data:

```text
400 * 700 binary values = 280000 bits = 35000 packed bytes
```

The file is hex text because that is easy to stream over USB CDC, but the
content represents the binarized model input. The STM32 does not binarize it;
it only decodes hex outside the timed window, then popcounts packed binary
bytes inside the timed window.

The expected CSV stores the known prefix spike count and route decision for
validation.

## Step 2: Create STM32CubeIDE Project

Use the same setup as before:

1. Open STM32CubeMX.
2. Start an MCU project for `STM32F411CEU6`.
3. Set `System Core -> RCC -> HSE` to `Crystal/Ceramic Resonator`.
4. Set `Connectivity -> USB_OTG_FS -> Mode` to `Device_Only`.
5. Set `Middleware -> USB_DEVICE -> Class` to `Communication Device Class`.
6. Set `HCLK` to `96 MHz`.
7. Project name suggestion: `Prefix_Router_Energy_Metrics`.
8. Toolchain: `STM32CubeIDE`.
9. Generate code and open it in CubeIDE.

## Step 3: Paste STM32 Code

For the recommended packed-binary measurement, open:

```text
Prefix_Router_Energy/prefix_router_stm32_packed_binary.c
```

Paste the marked sections into:

```text
Core/Src/main.c
USB_DEVICE/App/usbd_cdc_if.c
```

For the lightweight frame-count sanity check, use:

```text
Prefix_Router_Energy/prefix_router_stm32.c
```

Important constants:

```c
#define PREFIX_BITS 280000
#define PREFIX_THRESHOLD 6993
```

Do not move USB parsing or hex decoding into the DWT measurement window. For
the packed-binary version, the measured region should stay:

```c
DWT->CYCCNT = 0;
uint32_t start = DWT->CYCCNT;
int prefix_score = popcount_prefix_bits(prefix_bits, prefix_nbytes);
int route_dense = route_dense_from_prefix_score(prefix_score);
uint32_t end = DWT->CYCCNT;
```

## Step 4: Build And Flash

1. Build in CubeIDE. Confirm `0 errors`.
2. For the final timing number, use an optimized build. Either switch to
   `Release`, or set the Debug compiler optimization to `-O2`:

```text
Project -> Properties -> C/C++ Build -> Settings
MCU GCC Compiler -> Optimization -> Optimization level = Optimize more (-O2)
Apply and Close
Clean Project
Build
```

The default Debug build uses `-O0`, which is useful for debugging but gives an
artificially high cycle count for this tight popcount loop.
3. Hold `BOOT0`.
4. Plug in USB-C.
5. Release `BOOT0`.
6. Open STM32CubeProgrammer.
7. Select `USB`, connect, open the generated `.elf`.
8. Download/flash.
9. Unplug and replug without holding `BOOT0`.
10. In Device Manager, note the COM port, for example `COM3`.

## Step 5: Run Measurement

From the repo root:

```bash
python Prefix_Router_Energy/measure_prefix_router_binary_energy.py --port COM3
```

Use your actual port.

The script writes:

```text
Prefix_Router_Energy/prefix_router_energy_SHD_T1400_dt1ms_p400ms.txt
Prefix_Router_Energy/prefix_router_energy_SHD_T1400_dt1ms_p400ms_summary.json
```

For the packed-binary path, the actual output filenames are:

```text
Prefix_Router_Energy/prefix_router_binary_energy_SHD_T1400_dt1ms_p400ms.txt
Prefix_Router_Energy/prefix_router_binary_energy_SHD_T1400_dt1ms_p400ms_summary.json
```

Packed-binary energy table format:

```text
energy_J cycles prefix_score route_dense nbytes nbits expected_prefix_score expected_route_dense
```

Frame-count energy table format:

```text
energy_J cycles prefix_score route_dense n_counts expected_prefix_score expected_route_dense
```

Summary JSON includes:

```text
mean cycles/sample
mean router energy in uJ/sample
routed total mJ/sample
energy savings vs dense-only
```

## Expected Outcome

Before measurement, our repo-backed estimate was:

```text
~36 cycles per prefix event/check equivalent
~16.7% full-system energy savings after router cost
```

The STM32 result from this pipeline should replace that estimate.

If the measured mean router energy is below:

```text
0.114 mJ/sample
```

then the router still saves energy versus dense-only. That is the break-even router budget:

```text
dense-only energy      = 0.420 mJ/sample
routed model-only      = 0.306 mJ/sample
available router cost  = 0.114 mJ/sample
```

## Why This Measurement Is Cleaner Than The Estimate

The previous `16.7%` number estimated cycles from measured Transition Count timings. This packed-binary pipeline measures the actual prefix router from the binarized SHD model input on the STM32 using the same DWT counter method, so after running it you can write:

```text
We measured the prefix router cycle count directly on STM32F411CEU6 from the binarized SHD input raster and projected energy using 180 pJ/cycle.
```

That is stronger than saying we estimated cycles per event.
