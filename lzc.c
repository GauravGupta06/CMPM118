#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

<<<<<<< HEAD
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
=======

static inline uint64_t read_cycles() {
    uint64_t c;
    asm volatile("mrs %0, cntvct_el0" : "=r"(c));
    return c;
}

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
    if (!spike_seq_string) {
        return -1;
    }

    for (int i = 0; i < num_events; i++) {
        spike_seq_string[i] = events[i] ? '1' : '0';
    }
    spike_seq_string[num_events] = '\0';

    int lz_score = lzcomplexity(spike_seq_string);
    free(spike_seq_string);
    return lz_score;
}

/*
 * transition_count: count how many times the spike train changes value.
 * every 0->1 or 1->0 flip is one transition.
 *
 * silent (0000) and always-firing (1111) both score 0.
 * alternating (0101) scores the maximum (n-1).
 * more transitions = more complex temporal pattern.
 */
int transition_count(const int *events, int n) {
    int count = 0;
    for (int i = 1; i < n; i++) count += (events[i] != events[i-1]);
    return count;
}


#define MAX_LINE_LEN 16384

/*
 * Usage:
 *   ./lzc_qemu input.txt metrics.txt
 *
 * input.txt   : one binary string per line (000101010...)
 * metrics.txt : one line per sample: "<cycles> <lzc_value>"
 */
int main(int argc, char **argv) {

    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s input.txt metrics.txt\n", argv[0]);
        return 1;
    }

    FILE *fin  = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    if (!fin || !fout) {
        perror("File open failed");
        return 1;
    }

    char line[MAX_LINE_LEN];

    /*
     * Iterate through dataset in FIXED order:
     * one sample per line
     */
    while (fgets(line, MAX_LINE_LEN, fin)) {

        int len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
            len--;
        }

        /*
         * Convert '0'/'1' characters into int events[]
         * This matches compute_lzc_from_events()
         */
        int *events = (int *)malloc(len * sizeof(int));
        for (int i = 0; i < len; i++) {
            events[i] = (line[i] == '1') ? 1 : 0;
        }

        uint64_t start_cycles = read_cycles();
        int lzc = compute_lzc_from_events(events, len);
        uint64_t end_cycles = read_cycles();

        uint64_t lzc_start = read_cycles();
        int tc = transition_count(events, len);
        uint64_t lzc_end = read_cycles();

        /*
         * Write ONE metrics line per sample:
         * "<lzc_cycles> <lzc> <tc_cycles> <tc>"
         * Python will convert cycles -> Joules.
         */
        fprintf(fout, "%llu %d %llu %d\n",
                end_cycles - start_cycles, lzc,
                lzc_end - lzc_start, tc);

        free(events);
    }

    fclose(fin);
    fclose(fout);
    return 0;
}

>>>>>>> d4ab0a478f6113483a7bc408a7f6c48d9782421e
