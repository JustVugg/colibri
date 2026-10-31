/* coli_cuda_k3.dll against kimi_k3.c, bit for bit.
 *
 * 1. The device's exp_f and tanhf (k3_exact.h and its tables) against exp_f.h's exp_f and the
 *    C library's tanhf: every 4099th float, or with --all-floats every one of the 2^32.
 * 2. A prompt step's routed experts: random MXFP4 experts of Kimi K3's widths (latent 3584,
 *    moe_inter 3072, SiTU's 4 and 25) and of the tiny model's, over rows quantized by
 *    mxfp4_qx, through k3_wave (the CPU's gate and up, SiTU, quantization and down) and
 *    through k3cu_moe_*: every output the same bytes. Experts of 1 row to more than a device
 *    piece (1024 rows), handed out over the device's 8 slots more than once.
 * Says "skipped" and passes when there is no CUDA device or no coli_cuda_k3.dll beside the
 * test or in c/ (build_k3_cuda.bat).
 *   make tests/test_k3_cuda_experts.exe CUDA_DLL=1 && tests/test_k3_cuda_experts.exe */
#define main coli_k3_main_unused
#include "../kimi_k3.c"
#undef main
#include "../backend_cuda_k3.h"

static uint32_t rng = 20261031;
static uint32_t rnd32(void){ rng = rng * 1664525u + 1013904223u; return rng; }
static float rndf(void){ return ((float)(rnd32() >> 8) / 16777216.0f - 0.5f) * 4.0f; }

static int check_exact(int all){
    const uint32_t chunk = 1u << 24;
    uint32_t *out = malloc((size_t)chunk * 4);
    unsigned long long n = 0, bad = 0;
    if(!out){ printf("OOM\n"); return 1; }
    for(int kind = 0; kind < 2; kind++)
        for(uint64_t first = 0; first < (1ull << 32); first += all ? chunk : (uint64_t)chunk * 97){
            if(!k3cu_exact_eval(kind, (uint32_t)first, chunk, out)){ printf("exact eval failed\n"); free(out); return 1; }
            unsigned long long b = 0;
            #pragma omp parallel for reduction(+:b) schedule(static)
            for(int64_t i = 0; i < (int64_t)chunk; i++){
                if(!all && i % 4099) continue;
                uint32_t xb = (uint32_t)(first + (uint64_t)i);
                float x; memcpy(&x, &xb, 4);
                float r = kind ? tanhf(x) : exp_f(x);
                uint32_t rb; memcpy(&rb, &r, 4);
                if(rb != out[i]){
                    if(b < 3) printf("  %s(%a = 0x%08x): device 0x%08x, host 0x%08x\n", kind ? "tanhf" : "exp_f", x, xb, out[i], rb);
                    b++;
                }
            }
            bad += b; n += all ? chunk : chunk / 4099 + 1;
        }
    printf("exp_f and tanhf on the device: %llu floats each%s, %llu differ\n", n / 2, all ? " (all)" : " (every 4099th)", bad);
    free(out);
    return bad != 0;
}

static int check_experts(int LT, int MI, float b1, float b2, const int *rs, int ne){
    Model *m = calloc(1, sizeof(Model));
    m->c.latent = LT; m->c.moe_inter = MI; m->c.situ_b1 = b1; m->c.situ_b2 = b2;
    m->e_w1p = (int64_t)MI * LT / 2; m->e_w1s = (int64_t)MI * LT / 32;
    m->e_w2p = (int64_t)LT * MI / 2; m->e_w2s = (int64_t)LT * MI / 32;
    size_t off[6] = { 0, (size_t)m->e_w1p, (size_t)(m->e_w1p + m->e_w1s), (size_t)(m->e_w1p + m->e_w1s + m->e_w2p),
                      (size_t)(m->e_w1p + m->e_w1s + m->e_w2p + m->e_w2s), 0 };
    off[5] = off[4] + (size_t)m->e_w1p;
    size_t eb = off[5] + (size_t)m->e_w1s;
    int nrows = 1200, tot = 0, gb1 = LT / 32, gb2 = MI / 32;
    for(int e = 0; e < ne; e++) tot += rs[e];
    /* the experts: random codes, scales of 2^-17 .. 2^3; the rows: mostly within +-2, a few up to 60 */
    Slot *slots = calloc((size_t)ne, sizeof(Slot)); Slot **sl = malloc((size_t)ne * sizeof(Slot *));
    for(int e = 0; e < ne; e++){
        uint8_t *b = malloc(eb);
        for(size_t i = 0; i < eb; i++) b[i] = (uint8_t)(rnd32() >> 24);
        for(int k = 0; k < 3; k++){
            size_t s0 = off[2 * k + 1], s1 = s0 + (size_t)(k == 1 ? m->e_w2s : m->e_w1s);
            for(size_t i = s0; i < s1; i++) b[i] = (uint8_t)(110 + (rnd32() >> 24) % 21);
        }
        slots[e].buf = b; sl[e] = &slots[e];
    }
    float *z = malloc((size_t)nrows * LT * sizeof(float));
    for(size_t i = 0; i < (size_t)nrows * LT; i++){ z[i] = rndf(); if(rnd32() % 997 == 0) z[i] *= 30.0f; }
    int8_t *zq = malloc((size_t)nrows * LT); float *zs = malloc((size_t)nrows * gb1 * sizeof(float));
    mxfp4_qx(z, nrows, LT, zq, zs);
    int *first = malloc((size_t)ne * sizeof(int)), *pos = malloc((size_t)tot * sizeof(int)), *wave = malloc((size_t)ne * sizeof(int));
    for(int e = 0, f = 0; e < ne; e++){ first[e] = f; wave[e] = e; for(int i = 0; i < rs[e]; i++) pos[f + i] = (int)(rnd32() % (uint32_t)nrows); f += rs[e]; }
    /* the CPU: k3_wave over the pairs' gathered inputs, as k3_waves_run hands them over */
    int8_t *XQ = malloc((size_t)tot * LT), *GQ = malloc((size_t)tot * MI);
    float *XS = malloc((size_t)tot * gb1 * sizeof(float)), *GS = malloc((size_t)tot * gb2 * sizeof(float));
    float *G = malloc((size_t)tot * MI * sizeof(float)), *Yc = malloc((size_t)tot * LT * sizeof(float));
    for(int e = 0; e < tot; e++){
        memcpy(XQ + (size_t)e * LT, zq + (size_t)pos[e] * LT, (size_t)LT);
        memcpy(XS + (size_t)e * gb1, zs + (size_t)pos[e] * gb1, (size_t)gb1 * sizeof(float));
    }
    k3_wave(m, wave, ne, sl, first, rs, XQ, XS, NULL, G, GQ, GS, Yc);
    /* the device: the same experts, twice over its slots */
    float *Yg = k3cu_host_alloc((size_t)tot * LT * sizeof(float));
    int bad = 0, seen = 0;
    if(!Yg){ printf("no pinned memory\n"); return 1; }
    memset(Yg, 0xff, (size_t)tot * LT * sizeof(float));
    if(!k3cu_moe_begin(zq, zs, nrows, LT, MI, b1, b2, eb, off, Yg, tot)){ printf("k3cu_moe_begin failed\n"); return 1; }
    int *got = calloc((size_t)ne, sizeof(int)), tags[64];
    for(int e = 0; e < ne; e++){
        if(!k3cu_moe_expert(slots[e].buf, pos + first[e], rs[e], first[e], e)){ printf("k3cu_moe_expert failed\n"); return 1; }
        int n = k3cu_moe_poll(tags, 64);
        for(int i = 0; i < n; i++){ got[tags[i]]++; seen++; }
    }
    while(seen < ne){
        int n = k3cu_moe_poll(tags, 64);
        if(n < 0){ printf("k3cu_moe_poll failed\n"); return 1; }
        for(int i = 0; i < n; i++){ got[tags[i]]++; seen++; }
    }
    if(!k3cu_moe_end()){ printf("k3cu_moe_end failed\n"); return 1; }
    for(int e = 0; e < ne; e++) if(got[e] != 1){ printf("  expert %d reported %d times\n", e, got[e]); bad++; }
    size_t diff = 0, firstd = (size_t)-1;
    for(size_t i = 0; i < (size_t)tot * LT; i++)
        if(memcmp(Yc + i, Yg + i, 4)){ if(firstd == (size_t)-1) firstd = i; diff++; }
    if(diff){
        size_t p = firstd / LT, o = firstd % LT;
        printf("  pair %zu output %zu: CPU %a, device %a\n", p, o, Yc[firstd], Yg[firstd]);
        bad++;
    }
    printf("experts %dx%d (SiTU %g, %g): %d experts, %d pairs, %zu outputs: %zu differ\n",
           LT, MI, b1, b2, ne, tot, (size_t)tot * LT, diff);
    k3cu_host_free(Yg);
    for(int e = 0; e < ne; e++) free(slots[e].buf);
    free(slots); free(sl); free(z); free(zq); free(zs); free(first); free(pos); free(wave);
    free(XQ); free(GQ); free(XS); free(GS); free(G); free(Yc); free(got); free(m);
    return bad;
}

/* 3. The dense matrices the device does not hold: k3cu_matmul against matmul_i4_grouped
 *    (fmt 4) and matmul_q (fmt 1), every output the same bytes. */
static int check_dense(int fmt, int S, int I, int O, int gs){
    size_t wb = fmt == 4 ? (size_t)O * (I / 2) : (size_t)O * I, ns = fmt == 4 ? (size_t)O * (I / gs) : (size_t)O;
    uint8_t *w = malloc(wb); float *sc = malloc(ns * sizeof(float));
    float *x = malloc((size_t)S * I * sizeof(float)), *yc = malloc((size_t)S * O * sizeof(float)), *yg = malloc((size_t)S * O * sizeof(float));
    for(size_t i = 0; i < wb; i++) w[i] = (uint8_t)(rnd32() >> 24);
    for(size_t i = 0; i < ns; i++) sc[i] = rndf() * 0.01f;
    for(size_t i = 0; i < (size_t)S * I; i++) x[i] = rndf();
    if(fmt == 4) matmul_i4_grouped(yc, x, w, sc, S, I, O, gs);
    else matmul_q(yc, x, (const int8_t *)w, sc, S, I, O);
    memset(yg, 0xff, (size_t)S * O * sizeof(float));
    int ok = k3cu_matmul(fmt, yg, x, w, sc, S, I, O, gs);
    size_t diff = 0;
    for(size_t i = 0; i < (size_t)S * O; i++) if(memcmp(yc + i, yg + i, 4)) diff++;
    printf("dense %s %dx%d%s%s, %d rows: %s%zu of %zu differ\n", fmt == 4 ? "int4" : "int8", O, I,
           fmt == 4 ? ", groups of " : "", fmt == 4 ? (gs == 64 ? "64" : gs == 32 ? "32" : "128") : "", S,
           ok ? "" : "FAILED to run; ", diff, (size_t)S * O);
    free(w); free(sc); free(x); free(yc); free(yg);
    return !ok || diff;
}

int main(int argc, char **argv){
    int all = argc > 1 && !strcmp(argv[1], "--all-floats");
    if(!k3cu_init(0)){ printf("no CUDA device or coli_cuda_k3.dll: skipped\n"); return 0; }
    int bad = check_exact(all);
    const int real[] = { 1, 9, 64, 300, 1100, 2, 17, 8, 33, 5, 3 };
    bad += check_experts(3584, 3072, 4.0f, 25.0f, real, (int)(sizeof real / sizeof real[0]));
    const int tiny[] = { 1, 7, 8, 9, 40 };
    bad += check_experts(64, 32, 1.0f, 1.0f, tiny, 5);
    bad += check_dense(4, 37, 7168, 200, 64);
    bad += check_dense(4, 1100, 1024, 96, 32);
    bad += check_dense(4, 9, 3584, 71, 128);
    bad += check_dense(1, 33, 7168, 100, 0);
    bad += check_dense(1, 1030, 512, 40, 0);
    k3cu_report();
    printf(bad ? "FAILED\n" : "ok\n");
    return bad ? 1 : 0;
}
