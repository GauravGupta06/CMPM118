#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


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

        uint64_t cycles = end_cycles - start_cycles;

        /*
         * Write ONE metrics line per sample.
         * Python will convert cycles -> Joules.
         */
        fprintf(fout, "%llu %d\n", cycles, lzc);

        free(events);
    }

    fclose(fin);
    fclose(fout);
    return 0;
}

