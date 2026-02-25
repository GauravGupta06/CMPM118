# STM32 LZC Cycle Counter — Setup Guide

**Hardware:** STM32F411CEU6 (Black Pill) + USB-C data cable  
**Software:** Windows PC with Python installed

---

## Install Tools

1. Download and install **STM32CubeMX** from https://www.st.com/en/development-tools/stm32cubemx.html (Windows version)
2. Download and install **STM32CubeIDE** from https://www.st.com/en/development-tools/stm32cubeide.html (Windows version)
3. Download and install **STM32CubeProgrammer** from https://www.st.com/en/development-tools/stm32cubeprog.html (Windows version — this includes USB DFU drivers)

## Configure Project in CubeMX

4. Open STM32CubeMX → **ACCESS TO MCU SELECTOR**
5. Search **STM32F411CEU6** → select **STM32F411CEUx** → **Start Project**
6. Left panel → **System Core → RCC** → set HSE = **Crystal/Ceramic Resonator**
7. Left panel → **Connectivity → USB_OTG_FS** → Mode = **Device_Only**
8. Left panel → **Middleware and Software Packs → USB_DEVICE** → Class = **Communication Device Class (Virtual Port Com)**
9. **Clock Configuration** tab → set HCLK to **96 MHz**, press Enter, let it auto-resolve
10. **Project Manager** tab → Project Name = **LZC_Energy_Metrics**, Toolchain/IDE = **STM32CubeIDE**
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

14. In `/* USER CODE BEGIN 0 */` paste the LZC algorithm + helpers (copy from `lzc_stm32.c` — the `lzcomplexity`, `compute_lzc_from_events`, and `cdc_print` functions):
```c
int lzcomplexity(char *ss) {
    int ii = 0, kk = 1, el = 1, kmax = 1, cc = 1, nn;
    nn = strlen(ss);
    while (1) {
        if (ss[ii + kk - 1] == ss[el + kk - 1]) {
            kk++;
            if ((el + kk) > nn) { ++cc; break; }
        } else {
            if (kk > kmax) kmax = kk;
            ++ii;
            if (ii == el) {
                ++cc;
                el += kmax;
                if ((el + 1) > nn) break;
                ii = 0; kk = 1; kmax = 1;
            } else { kk = 1; }
        }
    }
    return cc;
}

int compute_lzc_from_events(const int *events, int num_events) {
    char *s = (char *)malloc(num_events + 1);
    if (!s) return -1;
    for (int i = 0; i < num_events; i++) s[i] = events[i] ? '1' : '0';
    s[num_events] = '\0';
    int lz = lzcomplexity(s);
    free(s);
    return lz;
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
    int lzc = compute_lzc_from_events(events, len);
    uint32_t end = DWT->CYCCNT;
    uint32_t cycles = end - start;

    char out[64];
    snprintf(out, sizeof(out), "%lu %d\r\n", (unsigned long)cycles, lzc);
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
21. Open **Device Manager** → confirm **"STM32 BOOTLOADER"** appears under Universal Serial Bus devices (if yellow warning icon, STM32CubeProgrammer's driver install should have fixed it)
22. Open **STM32CubeProgrammer** → top-right dropdown: change to **USB** → click **Connect**
23. Click **Open File** → browse to `Debug\LZC_Energy_Metrics.elf` in your project folder
24. Click **Download** → wait for "File download complete"
25. **Unplug USB** → **replug without holding BOOT0** → board boots your firmware
26. Open **Device Manager** → confirm new **"USB Serial Device (COM3)"** under Ports (COM & LPT). Note the COM number.

## Run the Pipeline

27. Clone the repo on Windows. Ensure `data/UCI_HAR_Dataset/` is present.
28. Open terminal in the repo folder, activate your venv
29. `pip install pyserial`
30. `python measure_lzc_energy.py --port COM3` (use your actual COM number)
31. Wait ~90 seconds for all 2947 samples to process
32. Results saved to `lzc_energy_table.txt` (format: `<energy_joules> <lzc_score>` per line)
33. `python visualize_lzc_energy.py` to see the plots
