#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SAMPLES 10000
#define MAX_LINE_LENGTH 2048

/* Core LZC algorithm - this is what we measure */
int lzcomplexity(const char *s, int n) {
    int i = 0, k = 1, l = 1, k_max = 1, c = 1;
    while (l + k <= n) {
        if (s[i + k - 1] == s[l + k - 1]) {
            k++;
        } else {
            if (k > k_max) k_max = k;
            i++;
            if (i == l) {
                c++;
                l += k_max;
                i = 0;
                k = 1;
                k_max = 1;
            } else {
                k = 1;
            }
        }
    }
    return c + 1;
}

/* Cycle counter - replace with actual hardware counter for QEMU */
static unsigned long cycles_start, cycles_end;

void start_cycles(void) {
    /* TODO: For ARM Cortex-M, read DWT->CYCCNT here */
    cycles_start = 0;
}

void stop_cycles(void) {
    /* TODO: For ARM Cortex-M, read DWT->CYCCNT here */
    cycles_end = 0;
}

unsigned long get_cycles(void) {
    return cycles_end - cycles_start;
}

int main(int argc, char **argv) {
    static char samples[MAX_SAMPLES][MAX_LINE_LENGTH];
    static int lengths[MAX_SAMPLES];
    static unsigned long cycles[MAX_SAMPLES];
    int num_samples = 0;
    FILE *f;

    if (argc < 3) {
        printf("Usage: %s input.txt output.txt\n", argv[0]);
        return 1;
    }

    /* Load all samples into memory */
    f = fopen(argv[1], "r");
    while (fgets(samples[num_samples], MAX_LINE_LENGTH, f) && num_samples < MAX_SAMPLES) {
        int len = strlen(samples[num_samples]);
        if (len > 0 && samples[num_samples][len-1] == '\n') {
            samples[num_samples][len-1] = '\0';
            len--;
        }
        lengths[num_samples] = len;
        num_samples++;
    }
    fclose(f);

    /* Measure LZC for each sample */
    for (int i = 0; i < num_samples; i++) {
        start_cycles();
        lzcomplexity(samples[i], lengths[i]);
        stop_cycles();
        cycles[i] = get_cycles();
    }

    /* Write cycle counts */
    f = fopen(argv[2], "w");
    for (int i = 0; i < num_samples; i++) {
        fprintf(f, "%lu\n", cycles[i]);
    }
    fclose(f);

    printf("Processed %d samples\n", num_samples);
    return 0;
}
