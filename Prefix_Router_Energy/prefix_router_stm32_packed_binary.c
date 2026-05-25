/* ============================================================
 * STM32F411CEU6 (Black Pill) - SHD Packed-Binary Prefix Router
 *
 * Measures the router directly from the same binary representation used by
 * the SHD model input:
 *
 *     first 400 ms of [T=1400, C=700] binarized SHD raster
 *     400 * 700 = 280000 bits = 35000 packed bytes
 *
 * The host sends a bit-packed sample as hex chunks:
 *
 *     BEGIN 35000 280000
 *     DATA <hex bytes>
 *     DATA <hex bytes>
 *     ...
 *     RUN
 *
 * USB transfer and hex parsing happen outside the timed region. The timed
 * region only popcounts the packed binary input and applies the threshold:
 *
 *     prefix_spikes >= 6993 -> dense
 *
 * This is the rigorous measurement path if the router receives the same
 * binarized raster as the SNN model.
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

#define MAX_LINE_LEN 1200
#define MAX_PREFIX_BYTES 35000
#define PREFIX_BITS 280000
#define PREFIX_THRESHOLD 6993

extern volatile char     cdc_rx_buf[MAX_LINE_LEN];
extern volatile uint16_t cdc_rx_len;
extern volatile uint8_t  cdc_rx_ready;

static uint8_t prefix_bits[MAX_PREFIX_BYTES] __attribute__((aligned(4)));
static uint32_t prefix_nbytes = 0;
static uint32_t prefix_nbits = 0;
static uint32_t prefix_offset = 0;

/* ---- END ---- */


/* ---- PASTE INTO: USER CODE BEGIN 0 (before main function) ---- */

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int append_hex_bytes(const char *hex) {
    while (*hex != '\0') {
        while (*hex == ' ' || *hex == '\t') {
            ++hex;
        }
        if (*hex == '\0') {
            break;
        }

        int hi = hex_value(hex[0]);
        int lo = hex_value(hex[1]);
        if (hi < 0 || lo < 0) {
            return -1;
        }
        if (prefix_offset >= prefix_nbytes || prefix_offset >= MAX_PREFIX_BYTES) {
            return -2;
        }

        prefix_bits[prefix_offset++] = (uint8_t)((hi << 4) | lo);
        hex += 2;
    }
    return 0;
}

static inline uint32_t popcount32_swar(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

static int popcount_prefix_bits(const uint8_t *bits, uint32_t nbytes) {
    int total = 0;
    uint32_t nwords = nbytes / 4u;
    const uint32_t *words = (const uint32_t *)bits;

    for (uint32_t i = 0; i < nwords; ++i) {
        total += (int)popcount32_swar(words[i]);
    }

    for (uint32_t i = nwords * 4u; i < nbytes; ++i) {
        uint8_t v = bits[i];
        v = v - ((v >> 1) & 0x55u);
        v = (v & 0x33u) + ((v >> 2) & 0x33u);
        total += (int)((v + (v >> 4)) & 0x0Fu);
    }

    return total;
}

static int route_dense_from_prefix_score(int prefix_score) {
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

if (strncmp(line, "BEGIN ", 6) == 0) {
    unsigned long nbytes = 0;
    unsigned long nbits = 0;
    if (sscanf(line + 6, "%lu %lu", &nbytes, &nbits) != 2) {
        cdc_print("ERROR BEGIN\r\n");
        continue;
    }
    if (nbytes > MAX_PREFIX_BYTES || nbits > PREFIX_BITS) {
        cdc_print("ERROR SIZE\r\n");
        continue;
    }

    prefix_nbytes = (uint32_t)nbytes;
    prefix_nbits = (uint32_t)nbits;
    prefix_offset = 0;
    cdc_print("OK BEGIN\r\n");
    continue;
}

if (strncmp(line, "DATA ", 5) == 0) {
    int rc = append_hex_bytes(line + 5);
    if (rc != 0) {
        snprintf(out, sizeof(out), "ERROR DATA %d\r\n", rc);
        CDC_Transmit_FS((uint8_t *)out, strlen(out));
        HAL_Delay(2);
        continue;
    }
    snprintf(out, sizeof(out), "OK DATA %lu\r\n", (unsigned long)prefix_offset);
    CDC_Transmit_FS((uint8_t *)out, strlen(out));
    HAL_Delay(2);
    continue;
}

if (strcmp(line, "RUN") == 0) {
    if (prefix_nbytes == 0 || prefix_offset != prefix_nbytes) {
        snprintf(
            out,
            sizeof(out),
            "ERROR RUN %lu %lu\r\n",
            (unsigned long)prefix_offset,
            (unsigned long)prefix_nbytes
        );
        CDC_Transmit_FS((uint8_t *)out, strlen(out));
        HAL_Delay(2);
        continue;
    }

    DWT->CYCCNT = 0;
    uint32_t start = DWT->CYCCNT;
    int prefix_score = popcount_prefix_bits(prefix_bits, prefix_nbytes);
    int route_dense = route_dense_from_prefix_score(prefix_score);
    uint32_t end = DWT->CYCCNT;
    uint32_t cycles = end - start;

    snprintf(
        out,
        sizeof(out),
        "%lu %d %d %lu %lu\r\n",
        (unsigned long)cycles,
        prefix_score,
        route_dense,
        (unsigned long)prefix_nbytes,
        (unsigned long)prefix_nbits
    );
    CDC_Transmit_FS((uint8_t *)out, strlen(out));
    HAL_Delay(2);
    continue;
}

if (strcmp(line, "DONE") == 0) {
    cdc_print("FINISHED\r\n");
    continue;
}

cdc_print("ERROR CMD\r\n");

/* ---- END ---- */


/* ============================================================
 *  FILE: USB_DEVICE/App/usbd_cdc_if.c
 * ============================================================ */

/* ---- ADD these globals near the top of usbd_cdc_if.c ---- */

volatile char     cdc_rx_buf[1200];
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
        } else if (!cdc_rx_ready && cdc_rx_len < 1199) {
            cdc_rx_buf[cdc_rx_len++] = Buf[i];
        }
    }

    USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
    USBD_CDC_ReceivePacket(&hUsbDeviceFS);
    return (USBD_OK);
}

/* ---- END ---- */
