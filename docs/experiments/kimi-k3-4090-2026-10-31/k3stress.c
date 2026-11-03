/* k3stress.c -- the routed experts' kernels computed again and again on the CPU (k3_wave, every worker) and on
 * the GPU (k3cu_moe_*, a thread of its own, at the same time), every output compared with the first result,
 * bit for bit: an unstable core, clock or memory shows as outputs that differ. Random MXFP4 experts at Kimi K3's
 * widths over 2,637 rows (as tests/test_k3_cuda_experts.c makes them).
 * usage: k3stress SECONDS (built like tests/test_k3_cuda_experts, beside coli_cuda_k3.dll) */
#define main coli_k3_main_unused
#include "kimi_k3.c"
#undef main
#include "backend_cuda_k3.h"

static uint32_t rng = 20261103;
static uint32_t rnd32(void){ rng = rng * 1664525u + 1013904223u; return rng; }
static float rndf(void){ return ((float)(rnd32() >> 8) / 16777216.0f - 0.5f) * 4.0f; }

enum { LT = 3584, MI = 3072, NROWS = 2637, NE = 8 };
static const int RS[NE] = {1, 3, 9, 24, 70, 200, 640, 1100};
static Model *M; static Slot SL[NE]; static Slot *SLP[NE];
static size_t OFF[6], EB; static int FIRST[NE], TOT, *POS, WAVE[NE];
static int8_t *ZQ; static float *ZS, *YREF;
static volatile int g_stop;
static unsigned long long g_gpu_iter, g_gpu_bad, g_cpu_iter, g_cpu_bad;

static size_t cmp_outputs(const float *y){
    size_t d = 0;
    for(size_t i = 0; i < (size_t)TOT * LT; i++) d += memcmp(YREF + i, y + i, 4) != 0;
    return d;
}
static void *gpu_loop(void *arg){
    (void)arg;
    float *Yg = k3cu_host_alloc((size_t)TOT * LT * sizeof(float));
    if(!Yg){ printf("no pinned memory\n"); exit(1); }
    while(!g_stop){
        memset(Yg, 0xff, (size_t)TOT * LT * sizeof(float));
        if(!k3cu_moe_begin(ZQ, ZS, NROWS, LT, MI, 4.0f, 25.0f, EB, OFF, Yg, TOT)){ printf("begin failed\n"); exit(1); }
        int tags[64], seen = 0;
        for(int e = 0; e < NE; e++){
            if(!k3cu_moe_expert(SL[e].buf, POS + FIRST[e], RS[e], FIRST[e], e)){ printf("expert failed\n"); exit(1); }
            int n = k3cu_moe_poll(tags, 64); if(n > 0) seen += n;
        }
        while(seen < NE){ int n = k3cu_moe_poll(tags, 64); if(n < 0){ printf("poll failed\n"); exit(1); } seen += n; }
        if(!k3cu_moe_end()){ printf("end failed\n"); exit(1); }
        size_t d = cmp_outputs(Yg);
        if(d){ g_gpu_bad += d; printf("  GPU iteration %llu: %zu outputs differ\n", g_gpu_iter, d); fflush(stdout); }
        g_gpu_iter++;
    }
    k3cu_host_free(Yg);
    return NULL;
}

int main(int argc, char **argv){
    const double secs = argc > 1 ? atof(argv[1]) : 300;
    if(!k3cu_init(0)){ printf("no device or no coli_cuda_k3.dll\n"); return 1; }
    M = calloc(1, sizeof(Model));
    M->c.latent = LT; M->c.moe_inter = MI; M->c.situ_b1 = 4.0f; M->c.situ_b2 = 25.0f;
    M->e_w1p = (int64_t)MI * LT / 2; M->e_w1s = (int64_t)MI * LT / 32; M->e_w2p = (int64_t)LT * MI / 2; M->e_w2s = (int64_t)LT * MI / 32;
    OFF[0] = 0; OFF[1] = M->e_w1p; OFF[2] = OFF[1] + M->e_w1s; OFF[3] = OFF[2] + M->e_w2p; OFF[4] = OFF[3] + M->e_w2s; OFF[5] = OFF[4] + M->e_w1p;
    EB = OFF[5] + M->e_w1s;
    for(int e = 0; e < NE; e++){
        uint8_t *b = k3cu_host_alloc(EB);   /* pinned, as the read-ahead pool's slots are */
        for(size_t i = 0; i < EB; i++) b[i] = (uint8_t)(rnd32() >> 24);
        for(int k = 0; k < 3; k++){
            size_t s0 = OFF[2 * k + 1], s1 = s0 + (size_t)(k == 1 ? M->e_w2s : M->e_w1s);
            for(size_t i = s0; i < s1; i++) b[i] = (uint8_t)(110 + (rnd32() >> 24) % 21);
        }
        SL[e].buf = b; SLP[e] = &SL[e]; WAVE[e] = e;
    }
    float *z = malloc((size_t)NROWS * LT * sizeof(float));
    for(size_t i = 0; i < (size_t)NROWS * LT; i++){ z[i] = rndf(); if(rnd32() % 997 == 0) z[i] *= 30.0f; }
    ZQ = malloc((size_t)NROWS * LT); ZS = malloc((size_t)NROWS * (LT / 32) * sizeof(float));
    mxfp4_qx(z, NROWS, LT, ZQ, ZS);
    for(int e = 0; e < NE; e++) TOT += RS[e];
    POS = malloc((size_t)TOT * sizeof(int));
    for(int e = 0, f = 0; e < NE; e++){   /* each expert's rows distinct and ascending, as a step's are */
        FIRST[e] = f;
        for(int i = 0; i < RS[e]; i++) POS[f + i] = (int)((long long)i * NROWS / RS[e] + rnd32() % (uint32_t)(NROWS / RS[e]));
        f += RS[e];
    }
    int8_t *XQ = malloc((size_t)TOT * LT), *GQ = malloc((size_t)TOT * MI);
    float *XS = malloc((size_t)TOT * (LT / 32) * sizeof(float)), *GS = malloc((size_t)TOT * (MI / 32) * sizeof(float));
    float *G = malloc((size_t)TOT * MI * sizeof(float)), *Y = malloc((size_t)TOT * LT * sizeof(float));
    YREF = malloc((size_t)TOT * LT * sizeof(float));
    for(int e = 0; e < TOT; e++){
        memcpy(XQ + (size_t)e * LT, ZQ + (size_t)POS[e] * LT, LT);
        memcpy(XS + (size_t)e * (LT / 32), ZS + (size_t)POS[e] * (LT / 32), (LT / 32) * sizeof(float));
    }
    k3_wave(M, WAVE, NE, SLP, FIRST, RS, XQ, XS, NULL, G, GQ, GS, YREF);
    printf("%d experts, %d pairs at %dx%d; %.0f s of the CPU's k3_wave on %d threads beside the GPU's kernels\n",
           NE, TOT, LT, MI, secs, omp_get_max_threads());
    pthread_t th; pthread_create(&th, NULL, gpu_loop, NULL);
    const double t0 = now_s();
    while(now_s() - t0 < secs){
        memset(Y, 0xff, (size_t)TOT * LT * sizeof(float));
        k3_wave(M, WAVE, NE, SLP, FIRST, RS, XQ, XS, NULL, G, GQ, GS, Y);
        size_t d = cmp_outputs(Y);
        if(d){ g_cpu_bad += d; printf("  CPU iteration %llu: %zu outputs differ\n", g_cpu_iter, d); fflush(stdout); }
        g_cpu_iter++;
    }
    g_stop = 1; pthread_join(th, NULL);
    printf("CPU: %llu iterations, %llu outputs differed; GPU: %llu iterations, %llu outputs differed\n",
           g_cpu_iter, g_cpu_bad, g_gpu_iter, g_gpu_bad);
    return g_cpu_bad || g_gpu_bad ? 2 : 0;
}
