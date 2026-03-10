# STM32 Transition Count Cycle Counter — Setup Guide

**Hardware:** STM32F411CEU6 (Black Pill) + USB-C data cable
**Software:** STM32CubeMX + STM32CubeIDE + STM32CubeProgrammer (Windows or macOS)

> This guide mirrors the LZC Energy Guide exactly. The only differences are:
> - Project name: `TC_Energy_Metrics`
> - Algorithm: `transition_count` instead of `lzcomplexity` + `compute_lzc_from_events`
> - Run script: `measure_transition_count_energy.py`
> - macOS port: `/dev/cu.usbmodem396F336632331` (already confirmed plugged in)

---

## Install Tools

1. Download and install **STM32CubeMX** from https://www.st.com/en/development-tools/stm32cubemx.html
2. Download and install **STM32CubeIDE** from https://www.st.com/en/development-tools/stm32cubeide.html
3. Download and install **STM32CubeProgrammer** from https://www.st.com/en/development-tools/stm32cubeprog.html

## Configure Project in CubeMX

4. Open STM32CubeMX → **ACCESS TO MCU SELECTOR**
5. Search **STM32F411CEU6** → select **STM32F411CEUx** → **Start Project**
6. Left panel → **System Core → RCC** → set HSE = **Crystal/Ceramic Resonator**
7. Left panel → **Connectivity → USB_OTG_FS** → Mode = **Device_Only**
8. Left panel → **Middleware and Software Packs → USB_DEVICE** → Class = **Communication Device Class (Virtual Port Com)**
9. **Clock Configuration** tab → set HCLK to **96 MHz**, press Enter, let it auto-resolve
10. **Project Manager** tab → Project Name = **TC_Energy_Metrics**, Toolchain/IDE = **STM32CubeIDE**
11. Click **GENERATE CODE** → when done, click **Open Project** (opens in CubeIDE)

## Edit Code in CubeIDE (2 files)

### File 1: `Core/Src/main.c`

12. In `/* USER CODE BEGIN Includes */` paste:
```c
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "usbd_cdc_if.h"
```

13. In `/* USER CODE BEGIN PV */` paste:
```c
#define MAX_LINE_LEN 2048
extern volatile char     cdc_rx_buf[MAX_LINE_LEN];
extern volatile uint16_t cdc_rx_len;
extern volatile uint8_t  cdc_rx_ready;
```

14. In `/* USER CODE BEGIN 0 */` paste the transition_count algorithm + helper:
```c
/* Counts 0->1 and 1->0 transitions in the binary spike sequence. O(N). */
int transition_count(const int *events, int num_events) {
    int count = 0;
    for (int i = 1; i < num_events; i++) {
        if (events[i] != events[i - 1]) ++count;
    }
    return count;
}

void cdc_print(const char *msg) {
    CDC_Transmit_FS((uint8_t *)msg, strlen(msg));
    HAL_Delay(2);
}
```

15. In `/* USER CODE BEGIN 2 */` paste:
```c
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
DWT->CYCCNT = 0;
DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
HAL_Delay(1000);
cdc_print("READY\r\n");
```

16. In `/* USER CODE BEGIN 3 */` (inside the while loop) paste:
```c
    if (!cdc_rx_ready) continue;

    char line[MAX_LINE_LEN];
    int len = cdc_rx_len;
    memcpy(line, (const char *)cdc_rx_buf, len);
    line[len] = '\0';
    cdc_rx_ready = 0;
    cdc_rx_len = 0;

    if (len == 0) continue;
    if (strcmp(line, "DONE") == 0) { cdc_print("FINISHED\r\n"); continue; }

    int *events = (int *)malloc(len * sizeof(int));
    if (!events) { cdc_print("ERROR\r\n"); continue; }
    for (int i = 0; i < len; i++) events[i] = (line[i] == '1') ? 1 : 0;

    DWT->CYCCNT = 0;
    uint32_t start = DWT->CYCCNT;
    int tc = transition_count(events, len);
    uint32_t end = DWT->CYCCNT;
    uint32_t cycles = end - start;

    char out[64];
    snprintf(out, sizeof(out), "%lu %d\r\n", (unsigned long)cycles, tc);
    CDC_Transmit_FS((uint8_t *)out, strlen(out));
    HAL_Delay(2);
    free(events);
```

### File 2: `USB_DEVICE/App/usbd_cdc_if.c`

17. In `/* USER CODE BEGIN PRIVATE_VARIABLES */` paste:
```c
volatile char     cdc_rx_buf[2048];
volatile uint16_t cdc_rx_len = 0;
volatile uint8_t  cdc_rx_ready = 0;
```

18. Find `CDC_Receive_FS` function → replace the body inside `/* USER CODE BEGIN 6 */` with:
```c
  for (uint32_t i = 0; i < *Len; i++) {
      if (Buf[i] == '\n' || Buf[i] == '\r') {
          if (cdc_rx_len > 0) {
              cdc_rx_buf[cdc_rx_len] = '\0';
              cdc_rx_ready = 1;
          }
      } else if (!cdc_rx_ready && cdc_rx_len < 2047) {
          cdc_rx_buf[cdc_rx_len++] = Buf[i];
      }
  }
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
  USBD_CDC_ReceivePacket(&hUsbDeviceFS);
  return (USBD_OK);
```

## Build and Flash

19. In CubeIDE, click the **hammer icon** 🔨 to build. Verify: **0 errors, 0 warnings**
20. **Hold BOOT0 button** on the board → plug in USB-C → release BOOT0 after 1 second
21. Open **STM32CubeProgrammer** → top-right dropdown: change to **USB** → click **Connect**
22. Click **Open File** → browse to `Debug/TC_Energy_Metrics.elf` in your project folder
23. Click **Download** → wait for "File download complete"
24. **Unplug USB** → **replug without holding BOOT0** → board boots the TC firmware
25. Confirm port reappears:
    - **macOS:** `/dev/cu.usbmodem...` in terminal: `ls /dev/cu.usbmodem*`
    - **Windows:** check Device Manager → Ports (COM & LPT)

## Run the Pipeline

26. Input file already generated at `tc_input_UCI_HAR.txt` (2947 samples, 1152 chars each)
27. Open terminal in the repo folder, activate your venv
28. `pip install pyserial`
29. Run the measurement:

**macOS** (board currently detected at `/dev/cu.usbmodem396F336632331`):
```bash
python measure_transition_count_energy.py \
  --port /dev/cu.usbmodem396F336632331 \
  --input tc_input_UCI_HAR.txt \
  --output_dir TC_Energy
```

**Windows:**
```bash
python measure_transition_count_energy.py --port COM3 --input tc_input_UCI_HAR.txt --output_dir TC_Energy
```

30. Wait ~90 seconds for all 2947 samples to process
31. Results saved to `TC_Energy/tc_energy_UCI_HAR.txt` (format: `<energy_joules> <cycles> <tc_score>` per line)
