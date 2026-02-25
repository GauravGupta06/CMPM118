/* ============================================================
 * STM32F411CEU6 (Black Pill) — LZC Cycle Counter via USB CDC
 *
 * SETUP IN STM32CubeIDE:
 * 1. File → New STM32 Project → search "STM32F411CEU6" → select it
 * 2. In the .ioc graphical configurator:
 *    a. System Core → RCC → HSE = Crystal/Ceramic Resonator
 *    b. Connectivity → USB_OTG_FS → Mode = Device_Only
 *    c. Middleware → USB_DEVICE → Class = Communication Device Class (CDC)
 *    d. Clock Configuration tab → set HCLK to 96 MHz (needed for USB)
 * 3. Click "Generate Code" (gear icon)
 *
 * PASTE CODE:
 * 4. Open Core/Src/main.c → paste sections marked "main.c" below
 * 5. Open USB_DEVICE/App/usbd_cdc_if.c → paste section marked "usbd_cdc_if.c" below
 *
 * FLASHING (USB DFU — no ST-Link needed):
 * 6. Hold BOOT0 button on the board
 * 7. Plug in USB-C cable (while still holding BOOT0)
 * 8. Release BOOT0
 * 9. In STM32CubeIDE: Run → Run Configurations → STM32 Cortex-M C/C++ 
 *    Application → set "USB" as the method (or use dfu-util from terminal)
 *
 * After flashing, unplug and replug USB. The board appears as /dev/ttyACM0
 * ============================================================ */


/* ============================================================
 *  FILE: Core/Src/main.c
 * ============================================================ */

/* ---- PASTE INTO: USER CODE BEGIN Includes ---- */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "usbd_cdc_if.h"

/* ---- END ---- */


/* ---- PASTE INTO: USER CODE BEGIN PV (private variables) ---- */

#define MAX_LINE_LEN 2048

/* Receive buffer: filled by USB CDC callback in usbd_cdc_if.c */
extern volatile char     cdc_rx_buf[MAX_LINE_LEN];
extern volatile uint16_t cdc_rx_len;
extern volatile uint8_t  cdc_rx_ready;  /* 1 = complete line received */

/* ---- END ---- */


/* ---- PASTE INTO: USER CODE BEGIN 0 (before main function) ---- */

/* -------- LZC Algorithm (unchanged from original) -------- */

int lzcomplexity(char *ss) {
    int ii = 0, kk = 1, el = 1, kmax = 1, cc = 1, nn;
    nn = strlen(ss);

    while (1) {
        if (ss[ii + kk - 1] == ss[el + kk - 1]) {
            kk++;
            if ((el + kk) > nn) {
                ++cc;
                break;
            }
        } else {
            if (kk > kmax) {
                kmax = kk;
            }
            ++ii;
            if (ii == el) {
                ++cc;
                el += kmax;
                if ((el + 1) > nn) {
                    break;
                }
                ii = 0;
                kk = 1;
                kmax = 1;
            } else {
                kk = 1;
            }
        }
    }
    return cc;
}



int compute_lzc_from_events(const int *events, int num_events) {
    char *spike_seq_string = (char *)malloc(num_events + 1);
    if (!spike_seq_string) return -1;

    for (int i = 0; i < num_events; i++) {
        spike_seq_string[i] = events[i] ? '1' : '0';
    }
    spike_seq_string[num_events] = '\0';

    int lz_score = lzcomplexity(spike_seq_string);
    free(spike_seq_string);
    return lz_score;
}

/* -------- USB CDC send helper -------- */

void cdc_print(const char *msg) {
    CDC_Transmit_FS((uint8_t *)msg, strlen(msg));
    HAL_Delay(1);  /* small delay to let USB flush */
}

/* ---- END ---- */


/* ---- PASTE INTO: USER CODE BEGIN 2 (inside main, after init) ---- */

/* Enable DWT cycle counter */
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
DWT->CYCCNT = 0;
DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

/* Wait for USB to enumerate */
HAL_Delay(1000);
cdc_print("READY\r\n");

char line[MAX_LINE_LEN];
char out[64];

while (1) {
    /* Wait for a complete line from PC */
    if (!cdc_rx_ready) continue;

    /* Copy received data (volatile → local) */
    int len = cdc_rx_len;
    memcpy(line, (const char *)cdc_rx_buf, len);
    line[len] = '\0';

    /* Reset for next receive */
    cdc_rx_ready = 0;
    cdc_rx_len = 0;

    if (len == 0) continue;

    /* Check for end signal */
    if (strcmp(line, "DONE") == 0) {
        cdc_print("FINISHED\r\n");
        continue;
    }

    /* Convert '0'/'1' chars to int array */
    int *events = (int *)malloc(len * sizeof(int));
    if (!events) {
        cdc_print("ERROR\r\n");
        continue;
    }
    for (int i = 0; i < len; i++) {
        events[i] = (line[i] == '1') ? 1 : 0;
    }

    /* Measure exact CPU cycles */
    DWT->CYCCNT = 0;
    uint32_t start = DWT->CYCCNT;
    int lzc = compute_lzc_from_events(events, len);
    uint32_t end = DWT->CYCCNT;
    uint32_t cycles = end - start;

    /* Send result: "<cycles> <lzc_value>" */
    int slen = snprintf(out, sizeof(out), "%lu %d\r\n", (unsigned long)cycles, lzc);
    CDC_Transmit_FS((uint8_t *)out, slen);
    HAL_Delay(2);

    free(events);
}

/* ---- END ---- */


/* ============================================================
 *  FILE: USB_DEVICE/App/usbd_cdc_if.c
 *
 *  Find the CDC_Receive_FS function and REPLACE its body
 *  with the code below.
 *
 *  Also add the global variables at the top of the file
 *  (in USER CODE BEGIN PRIVATE_VARIABLES or after includes).
 * ============================================================ */

/* ---- ADD these globals near the top of usbd_cdc_if.c ---- */

volatile char     cdc_rx_buf[2048];
volatile uint16_t cdc_rx_len = 0;
volatile uint8_t  cdc_rx_ready = 0;

/* ---- END ---- */


/* ---- REPLACE the body of CDC_Receive_FS() with this ---- */

static int8_t CDC_Receive_FS(uint8_t *Buf, uint32_t *Len)
{
    /* Accumulate bytes until newline */
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

    /* Re-arm USB receive */
    USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
    USBD_CDC_ReceivePacket(&hUsbDeviceFS);
    return (USBD_OK);
}

/* ---- END ---- */
