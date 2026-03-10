/* ============================================================
 * STM32F411CEU6 (Black Pill) — Transition Count Cycle Counter via USB CDC
 *
 * Measures DWT CPU cycles for computing transition_count on binarized spike
 * sequences. Setup is identical to lzc_stm32.c — reuse the same CubeIDE
 * project and only swap the algorithm section in main.c.
 *
 * SETUP IN STM32CubeIDE (identical to LZC project):
 * 1. File → New STM32 Project → search "STM32F411CEU6" → select it
 * 2. In the .ioc graphical configurator:
 *    a. System Core → RCC → HSE = Crystal/Ceramic Resonator
 *    b. Connectivity → USB_OTG_FS → Mode = Device_Only
 *    c. Middleware → USB_DEVICE → Class = Communication Device Class (CDC)
 *    d. Clock Configuration tab → set HCLK to 96 MHz
 * 3. Click "Generate Code"
 * 4. Paste sections below into the two files marked
 *
 * FLASHING (USB DFU — no ST-Link needed):
 * 5. Hold BOOT0 button → plug USB-C → release BOOT0 after 1s
 * 6. STM32CubeProgrammer → USB → Connect → Open .elf → Download
 * 7. Unplug and replug USB (no BOOT0). Board appears as COM port.
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

/* -------- Transition Count Algorithm --------
 *
 * Counts the number of 0->1 and 1->0 transitions in the binary spike sequence.
 * For a sequence of length N there are at most N-1 possible transitions.
 * This is O(N) vs LZC's O(N log N), so cycle counts should be much lower.
 */
int transition_count(const int *events, int num_events) {
    int count = 0;
    for (int i = 1; i < num_events; i++) {
        if (events[i] != events[i - 1]) {
            ++count;
        }
    }
    return count;
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

    /* Copy received data (volatile -> local) */
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
    int tc = transition_count(events, len);
    uint32_t end = DWT->CYCCNT;
    uint32_t cycles = end - start;

    /* Send result: "<cycles> <transition_count_value>" */
    int slen = snprintf(out, sizeof(out), "%lu %d\r\n", (unsigned long)cycles, tc);
    CDC_Transmit_FS((uint8_t *)out, slen);
    HAL_Delay(2);

    free(events);
}

/* ---- END ---- */


/* ============================================================
 *  FILE: USB_DEVICE/App/usbd_cdc_if.c
 *
 *  Identical to the LZC project — same USB CDC receive buffering.
 *  If reusing the same CubeIDE project, this file needs no changes.
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
