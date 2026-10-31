/* DRAM read bandwidth with N threads: each sums its slice of a buffer much larger than L3. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc, char **argv){
    size_t gb = argc > 1 ? (size_t)atoi(argv[1]) : 8;
    size_t n = gb << 30;
    uint8_t *buf = _aligned_malloc(n, 4096);
    if(!buf){ fprintf(stderr,"no memory\n"); return 1; }
    memset(buf, 1, n);
    int ths[] = {1, 4, 8, 16, 32};
    for(size_t k = 0; k < sizeof ths / sizeof ths[0]; k++){
        int t = ths[k];
        double best = 0;
        for(int rep = 0; rep < 3; rep++){
            double t0 = now(); long long tot = 0;
            #pragma omp parallel num_threads(t) reduction(+:tot)
            {
                int id = omp_get_thread_num();
                size_t per = n / t, a = per * id, b = a + per;
                __m256i acc = _mm256_setzero_si256();
                for(size_t i = a; i < b; i += 64){
                    acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *)(buf + i)));
                    acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *)(buf + i + 32)));
                }
                long long v[4]; _mm256_storeu_si256((__m256i *)v, acc);
                tot += v[0] + v[1] + v[2] + v[3];
            }
            double dt = now() - t0, bw = n / dt / 1e9;
            if(bw > best) best = bw;
            if(tot == 42) printf(" ");
        }
        printf("%2d threads: %.1f GB/s read\n", t, best);
    }
    return 0;
}
