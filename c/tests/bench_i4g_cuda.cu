/* Throughput of the three ways to do one K3 dense projection at a prefill chunk
 * size: CPU grouped-int4, GPU GEMV-by-replication, GPU Tensor Core tiling.
 *
 * Reports the kernel time AND the time including coli_cuda_matmul's host round
 * trip, because the API copies x in and y out on every call. If the round trip
 * dominates, the conclusion is about the API, not the kernel -- the engine would
 * need activations to stay resident across a layer, which is a much larger
 * change than swapping a kernel.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "../backend_cuda.h"

extern "C" void i4g_ref(float *y, const float *x, const unsigned char *q4,
                        const float *scale, int S, int I, int O, int gs);

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
static unsigned long long sd = 0x2545F4914F6CDD1DULL;
static float rnd(void) {
    sd = sd * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((sd >> 40) % 2000001) / 1000000.0f - 1.0f;
}

int main(int argc, char **argv) {
    int S = argc > 1 ? std::atoi(argv[1]) : 32;
    int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    int dev[1] = {0};
    if (!coli_cuda_init(dev, 1)) { std::fprintf(stderr, "no CUDA device\n"); return 0; }

    struct { const char *nm; int I, O; } shp[] = {
        {"lat_down 7168->3584", 7168, 3584},
        {"lat_up   3584->7168", 3584, 7168},
        {"sh_gate  7168->6144", 7168, 6144},
        {"kda_qkv  7168->12288", 7168, 12288},
    };
    std::printf("S=%d, %d reps, per-call ms (lower is better)\n", S, reps);
    std::printf("%-22s %10s %10s %10s %8s\n", "shape", "cpu", "gpu-gemv", "gpu-tile", "tile/cpu");

    for (unsigned c = 0; c < sizeof(shp)/sizeof(shp[0]); c++) {
        int I = shp[c].I, O = shp[c].O, gs = 64;
        int rb = (I+1)/2, ng = (I+gs-1)/gs;
        unsigned char *q4 = (unsigned char*)std::malloc((size_t)O*rb);
        float *scale = (float*)std::malloc((size_t)O*ng*sizeof(float));
        float *x = (float*)std::malloc((size_t)S*I*sizeof(float));
        float *y = (float*)std::calloc((size_t)S*O, sizeof(float));
        for (size_t i = 0; i < (size_t)O*rb; i++) { sd = sd*6364136223846793005ULL+1442695040888963407ULL; q4[i]=(unsigned char)(sd>>40); }
        for (size_t i = 0; i < (size_t)O*ng; i++) scale[i] = 0.01f;
        for (size_t i = 0; i < (size_t)S*I; i++) x[i] = rnd();

        i4g_ref(y, x, q4, scale, S, I, O, gs);          /* warm caches/threads */
        double t0 = now();
        for (int r = 0; r < reps; r++) i4g_ref(y, x, q4, scale, S, I, O, gs);
        double tcpu = (now()-t0)/reps*1e3;

        double tg[2];
        for (int mode = 0; mode < 2; mode++) {
            setenv("COLI_CUDA_TC_W4A16_MIN", mode ? "1" : "1000000", 1);
            ColiCudaTensor *t = nullptr;
            coli_cuda_matmul(&t, y, x, q4, scale, 4, S, I, O, 0, gs);   /* upload once */
            double s0 = now();
            for (int r = 0; r < reps; r++)
                coli_cuda_matmul(&t, y, x, q4, scale, 4, S, I, O, 0, gs);
            tg[mode] = (now()-s0)/reps*1e3;
            if (t) coli_cuda_tensor_free(t);
        }
        std::printf("%-22s %10.3f %10.3f %10.3f %8.2fx\n",
                    shp[c].nm, tcpu, tg[0], tg[1], tcpu/tg[1]);
        std::free(q4); std::free(scale); std::free(x); std::free(y);
    }
    return 0;
}
