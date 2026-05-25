/* ============================================================
 * STM32F411CEU6 (Black Pill) - SHD Prefix Router Cycle Counter
 *
 * Measures DWT CPU cycles for the final SHD router metric:
 *
 *     prefix_400ms_spikes >= 6993  -> dense
 *     otherwise                    -> sparse
 *
 * The host sends one line per SHD sample. Each line contains the first
 * 400 one-millisecond frame spike counts, comma-separated:
 *
 *     12,8,4,0,0,...,19
 *
 * Parsing and USB transfer are intentionally outside the timed region.
 * The timed region only sums the 400 uint16_t counts and applies the
 * threshold, which is the router compute cost we want.
 *
 * SETUP IN STM32CubeIDE:
 * 1. File -> New STM32 Project -> search "STM32F411CEU6" -> select it.
 * 2. System Core -> RCC -> HSE = Crystal/Ceramic Resonator.
 * 3. Connectivity -> USB_OTG_FS -> Mode = Device_Only.
 * 4. Middleware -> USB_DEVICE -> Class = Communication Device Class.
 * 5. Clock Configuration -> set HCLK to 96 MHz.
 * 6. Generate code, then paste the marked sections below.
 * ============================================================ */


/* ============================================================
 *  FILE: Core/Src/main.c
 * ============================================================ */

/* ---- PASTE INTO: USER CODE BEGIN Includes ---- */

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "usbd_cdc_if.h"

/* ---- END ---- */


/* ---- PASTE INTO: USER CODE BEGIN PV (private variables) ---- */

#define MAX_LINE_LEN 4096
#define PREFIX_BINS 400
#define PREFIX_THRESHOLD 6993

extern volatile char     cdc_rx_buf[MAX_LINE_LEN];
extern volatile uint16_t cdc_rx_len;
extern volatile uint8_t  cdc_rx_ready;

/* ---- END ---- */


/* ---- PASTE INTO: USER CODE BEGIN 0 (before main function) ---- */

int parse_prefix_counts(const char *line, uint16_t *counts, int max_counts) {
    int n = 0;
    const char *p = line;

    while (*p != '\0' && n < max_counts) {
        while (*p != '\0' && !isdigit((unsigned char)*p)) {
            ++p;
        }
        if (*p == '\0') {
            break;
        }

        unsigned int value = 0;
        while (isdigit((unsigned char)*p)) {
            value = value * 10u + (unsigned int)(*p - '0');
            ++p;
        }

        if (value > 65535u) {
            value = 65535u;
        }
        counts[n++] = (uint16_t)value;
    }

    return n;
}

int prefix_spike_count(const uint16_t *counts, int n_counts) {
    int total = 0;
    for (int i = 0; i < n_counts; ++i) {
        total += counts[i];
    }
    return total;
}

int route_dense_from_prefix_score(int prefix_score) {
    return prefix_score >= PREFIX_THRESHOLD;
}

void cdc_print(const char *msg) {
    CDC_Transmit_FS((uint8_t *)msg, strlen(msg));
    HAL_Delay(2);
}

/* ---- END ---- */


/* ---- PASTE INTO: USER CODE BEGIN 2 (inside main, after init) ---- */

CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
DWT->CYCCNT = 0;
DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
HAL_Delay(1000);
cdc_print("READY\r\n");

/* ---- END ---- */


/* ---- PASTE INTO: USER CODE BEGIN 3 (inside the while loop) ---- */

if (!cdc_rx_ready) {
    continue;
}

char line[MAX_LINE_LEN];
uint16_t prefix_counts[PREFIX_BINS];
char out[96];

int len = cdc_rx_len;
if (len >= MAX_LINE_LEN) {
    len = MAX_LINE_LEN - 1;
}
memcpy(line, (const char *)cdc_rx_buf, len);
line[len] = '\0';

cdc_rx_ready = 0;
cdc_rx_len = 0;

if (len == 0) {
    continue;
}
if (strcmp(line, "DONE") == 0) {
    cdc_print("FINISHED\r\n");
    continue;
}

int n_counts = parse_prefix_counts(line, prefix_counts, PREFIX_BINS);
if (n_counts <= 0) {
    cdc_print("ERROR 0 0 0\r\n");
    continue;
}

/* Measure only the router function: sum prefix counts + threshold. */
DWT->CYCCNT = 0;
uint32_t start = DWT->CYCCNT;
int prefix_score = prefix_spike_count(prefix_counts, n_counts);
int route_dense = route_dense_from_prefix_score(prefix_score);
uint32_t end = DWT->CYCCNT;
uint32_t cycles = end - start;

/* Send result: "<cycles> <prefix_score> <route_dense> <n_counts>" */
int slen = snprintf(
    out,
    sizeof(out),
    "%lu %d %d %d\r\n",
    (unsigned long)cycles,
    prefix_score,
    route_dense,
    n_counts
);
CDC_Transmit_FS((uint8_t *)out, slen);
HAL_Delay(2);

/* ---- END ---- */


/* ============================================================
 *  FILE: USB_DEVICE/App/usbd_cdc_if.c
 * ============================================================ */

/* ---- ADD these globals near the top of usbd_cdc_if.c ---- */

volatile char     cdc_rx_buf[4096];
volatile uint16_t cdc_rx_len = 0;
volatile uint8_t  cdc_rx_ready = 0;

/* ---- END ---- */


/* ---- REPLACE the body of CDC_Receive_FS() with this ---- */

static int8_t CDC_Receive_FS(uint8_t *Buf, uint32_t *Len)
{
    for (uint32_t i = 0; i < *Len; i++) {
        if (Buf[i] == '\n' || Buf[i] == '\r') {
            if (cdc_rx_len > 0) {
                cdc_rx_buf[cdc_rx_len] = '\0';
                cdc_rx_ready = 1;
            }
        } else if (!cdc_rx_ready && cdc_rx_len < 4095) {
            cdc_rx_buf[cdc_rx_len++] = Buf[i];
        }
    }

    USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
    USBD_CDC_ReceivePacket(&hUsbDeviceFS);
    return (USBD_OK);
}

/* ---- END ---- */
