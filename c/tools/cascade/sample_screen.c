/* Compile twice with CASCADE_HEADER naming original or generated header. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include CASCADE_HEADER

static double now(void) {
#ifdef _WIN32
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t); QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    int v = atoi(argv[1]), shape = atoi(argv[2]);
    float temp = (float)atof(argv[3]), top = (float)atof(argv[4]);
    if (v != 131072 && v != 248320) return 2;
    float *lo = malloc((size_t)v * sizeof(float));
    if (!lo) return 3;
    unsigned state = 73;
    for (int i = 0; i < v; ++i) {
        state = state * 1664525u + 1013904223u;
        lo[i] = shape == 0 ? 0.f : (float)(state >> 8) / 16777216.f * 14.f - 7.f;
        if (shape == 1 && i < 16) lo[i] = 25.f - i * .25f;
    }
    srand(42);
    (void)serve_sample(lo, v, temp, top);
    printf("{\"tokens\":[");
    double times[3];
    int next[3];
    for (int k = 0; k < 3; ++k) {
        srand(100u + (unsigned)k);
        double start = now();
        int token = serve_sample(lo, v, temp, top);
        times[k] = (now() - start) * 1000.;
        next[k] = rand();
        printf("%s%d", k ? "," : "", token);
    }
    printf("],\"next_rng\":[%d,%d,%d],\"ms\":[%.9f,%.9f,%.9f]}\n",
           next[0], next[1], next[2], times[0], times[1], times[2]);
    free(lo);
    return 0;
}
