# DVSGesture Prefix Router Energy Measurement

This is the DVS Gesture companion to the SHD packed-binary STM32 measurement.

The current DVSGesture training pipeline uses 1 ms **count frames**, not binary
input frames:

```text
[T, 2, 32, 32] float32 count frames, capped at T=600
```

For an STM32-friendly router metric, this measurement clips the first prefix to
binary occupancy:

```text
metric bit = 1 if count_frame[t, polarity, y, x] > 0 else 0
```

Default prefix size:

```text
400 ms * 2 polarities * 32 * 32 = 819200 bits = 102400 packed bytes/sample
```

This is much larger than the SHD prefix measurement:

```text
SHD 400 ms prefix: 400 * 700 = 280000 bits = 35000 packed bytes/sample
DVSGesture 400 ms prefix: 819200 bits = 102400 packed bytes/sample
```

So DVS is about `2.93x` larger for the same 400 ms binary raster-style prefix.
That is why this file is separate from the SHD firmware.

## Files

```text
Prefix_Router_Energy/measure_dvsgesture_prefix_router_binary_energy.py
Prefix_Router_Energy/dvsgesture_prefix_router_stm32_packed_binary.c
```

## Generate Input Files

From the repo root:

```bash
python Prefix_Router_Energy/measure_dvsgesture_prefix_router_binary_energy.py --dry-run
```

This writes:

```text
Prefix_Router_Energy/dvsgesture_prefix_router_binary_input_DVSGesture_T600_dt1ms_p400ms_32x32_binary.hex
Prefix_Router_Energy/dvsgesture_prefix_router_binary_expected_DVSGesture_T600_dt1ms_p400ms_32x32_binary.csv
```

The expected CSV also records the original count-frame prefix sum and the
number of bins where the count was greater than one. Those values are useful
for analysis, but the STM32 timed metric is the binary popcount.

## Flash STM32 Firmware

Use the same CubeMX/CubeIDE project setup as the SHD router, but paste sections
from:

```text
Prefix_Router_Energy/dvsgesture_prefix_router_stm32_packed_binary.c
```

Important constants:

```c
#define MAX_PREFIX_BYTES 102400
#define PREFIX_BITS 819200
#define PREFIX_THRESHOLD 10000
```

`PREFIX_THRESHOLD` is only a placeholder. Choose the final threshold after the
DVS router AUC / energy analysis. It barely changes cycle count.

## RAM Warning

`MAX_PREFIX_BYTES=102400` allocates a 100 KB static buffer. The STM32F411CEU6
has 128 KB SRAM, so this may fit, but it is tight once USB buffers, stack, and
HAL state are included. If CubeIDE reports SRAM overflow, use one of these:

```text
1. reduce --prefix_ms, for example 200 ms gives 51200 packed bytes/sample
2. switch to a streaming/event-count metric for DVS instead of dense raster popcount
```

## Run Measurement

After flashing and reconnecting the board as a USB serial device:

```bash
python Prefix_Router_Energy/measure_dvsgesture_prefix_router_binary_energy.py --port COM3
```

Use your actual COM port.

Output table format:

```text
energy_J cycles binary_prefix_ones route_dense nbytes nbits expected_binary_ones expected_route_dense count_prefix_sum count_bins_gt1
```

Summary JSON includes mean cycles/sample and mean router energy in uJ/sample
using the same `1.8e-10 J/cycle` STM32L476 projection used for SHD.
