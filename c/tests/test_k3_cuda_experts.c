/* coli_cuda_k3.dll against kimi_k3.c, bit for bit.
 *
 * 1. The device's exp_f (in float pairs, and its fallback in doubles: k3_exact.h and its
 *    table) and tanhf against exp_f.h's exp_f and the C library's tanhf: every 4099th float,
 *    or with --all-floats every one of the 2^32.
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
    for(int kind = 0; kind < 3; kind++)   /* exp_f, tanhf, exp_f's fallback in doubles */
        for(uint64_t first = 0; first < (1ull << 32); first += all ? chunk : (uint64_t)chunk * 97){
            if(!k3cu_exact_eval(kind, (uint32_t)first, chunk, out)){ printf("exact eval failed\n"); free(out); return 1; }
            unsigned long long b = 0;
            #pragma omp parallel for reduction(+:b) schedule(static)
            for(int64_t i = 0; i < (int64_t)chunk; i++){
                if(!all && i % 4099) continue;
                uint32_t xb = (uint32_t)(first + (uint64_t)i);
                float x; memcpy(&x, &xb, 4);
                float r = kind == 1 ? tanhf(x) : exp_f(x);
                uint32_t rb; memcpy(&rb, &r, 4);
                if(rb != out[i]){
                    if(b < 3) printf("  %s(%a = 0x%08x): device 0x%08x, host 0x%08x\n", kind == 1 ? "tanhf" : kind ? "exp_f (doubles)" : "exp_f", x, xb, out[i], rb);
                    b++;
                }
            }
            bad += b; n += all ? chunk : chunk / 4099 + 1;
        }
    printf("exp_f, tanhf and exp_f's fallback on the device: %llu floats each%s, %llu differ\n", n / 3,
           all ? " (all)" : " (every 4099th)", bad);
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

/* 4. A prompt step's gated MLA attention: k3cu_mla against mla_forward's loop, copied here
 *    with the functions it calls (w_addrow, k3_mla_scores_g, softmax_, k3_rowdot8_i8,
 *    sigmoidf_): every output the same bytes. */
static int check_mla(int H, int qn, int qr, int vh, int kvl, int pos0, int C){
    const int N = pos0 + C, qh = qn + qr, O = H * (qn + vh);
    float *Lc = malloc((size_t)N * kvl * sizeof(float)), *Rc = malloc((size_t)N * (qr ? qr : 1) * sizeof(float));
    float *qv = malloc((size_t)C * H * qh * sizeof(float)), *gv = malloc((size_t)C * H * vh * sizeof(float));
    float *cc = malloc((size_t)C * H * vh * sizeof(float)), *cg = malloc((size_t)C * H * vh * sizeof(float));
    W kvb; memset(&kvb, 0, sizeof kvb);
    kvb.fmt = 1; kvb.O = O; kvb.I = kvl;
    kvb.q8 = malloc((size_t)O * kvl); kvb.s = malloc((size_t)O * sizeof(float));
    for(size_t i = 0; i < (size_t)N * kvl; i++) Lc[i] = rndf();
    for(size_t i = 0; i < (size_t)N * qr; i++) Rc[i] = rndf();
    for(size_t i = 0; i < (size_t)C * H * qh; i++) qv[i] = rndf();
    for(size_t i = 0; i < (size_t)C * H * vh; i++) gv[i] = rndf();
    for(size_t i = 0; i < (size_t)O * kvl; i++){ int v = (int)(rnd32() >> 24) - 128; kvb.q8[i] = (int8_t)(v < -127 ? -127 : v); }
    for(int r = 0; r < O; r++) kvb.s[r] = 0.004f * (1.0f + 0.1f * rndf());
    const float scale = 1.0f / sqrtf((float)qh);
    for(int tt = 0; tt < C; tt++){
        int nt = pos0 + tt + 1;
        const float *qvt = qv + (int64_t)tt * H * qh, *gvt = gv + (int64_t)tt * H * vh;
        float *ctxt = cc + (int64_t)tt * H * vh;
        const int ngr = (H + K3_MLA_G - 1) / K3_MLA_G;
        #pragma omp parallel for schedule(dynamic)
        for(int hg = 0; hg < ngr; hg++){
            const int h0 = hg * K3_MLA_G, nh = H - h0 < K3_MLA_G ? H - h0 : K3_MLA_G;
            float *qab = falloc((int64_t)K3_MLA_G * kvl), *scs = falloc((int64_t)K3_MLA_G * nt);
            const float *qrps[K3_MLA_G];
            for(int g = 0; g < nh; g++){
                const float *qp = qvt + (int64_t)(h0 + g) * qh;
                int rbase = (h0 + g) * (qn + vh);
                float *qabs = qab + (int64_t)g * kvl; memset(qabs, 0, kvl * sizeof(float));
                for(int d = 0; d < qn; d++) w_addrow(&kvb, rbase + d, qp[d], qabs);
                qrps[g] = qp + qn;
            }
            k3_mla_scores_g(scs, qab, qrps, nh, Lc, Rc, nt, kvl, qr, scale);
            for(int g = 0; g < nh; g++){
                const int h = h0 + g, rbase = h * (qn + vh);
                float *sc = scs + (int64_t)g * nt;
                softmax_(sc, nt);
                float clat[4096]; memset(clat, 0, kvl * sizeof(float));
                for(int t = 0; t < nt; t++){
                    const float *Lt = Lc + (int64_t)t * kvl; float s2 = sc[t];
                    for(int i = 0; i < kvl; i++) clat[i] += s2 * Lt[i];
                }
                float *cx = ctxt + (int64_t)h * vh;
                int d = 0;
                for(; d + 8 <= vh; d += 8){
                    float rd[8];
                    k3_rowdot8_i8(&kvb, rbase + qn + d, clat, rd);
                    for(int j = 0; j < 8; j++) cx[d + j] = rd[j] * sigmoidf_(gvt[(int64_t)h * vh + d + j]);
                }
                for(; d < vh; d++) cx[d] = w_rowdot(&kvb, rbase + qn + d, clat) * sigmoidf_(gvt[(int64_t)h * vh + d]);
            }
            free(qab); free(scs);
        }
    }
    memset(cg, 0xff, (size_t)C * H * vh * sizeof(float));
    int ok = k3cu_mla(Lc, Rc, N, pos0, C, qv, gv, kvb.q8, kvb.s, H, qn, qr, vh, kvl, scale, cg);
    size_t diff = 0, first = (size_t)-1;
    for(size_t i = 0; i < (size_t)C * H * vh; i++) if(memcmp(cc + i, cg + i, 4)){ if(first == (size_t)-1) first = i; diff++; }
    if(diff) printf("  output %zu: CPU %a, device %a\n", first, cc[first], cg[first]);
    printf("MLA %d heads (%d+%d, v %d, latent %d), rows %d..%d: %s%zu of %zu differ\n", H, qn, qr, vh, kvl,
           pos0, N - 1, ok ? "" : "FAILED to run; ", diff, (size_t)C * H * vh);
    free(Lc); free(Rc); free(qv); free(gv); free(cc); free(cg); free(kvb.q8); free(kvb.s);
    return !ok || diff;
}

/* 5. The bf16 router: k3cu_router against k3_router_scores, every score the same bytes. */
static int check_router(int C, int D, int E){
    Moe o; memset(&o, 0, sizeof o);
    o.router16 = malloc((size_t)E * D * sizeof(uint16_t));
    float *x = malloc((size_t)C * D * sizeof(float)), *sc = malloc((size_t)C * E * sizeof(float)), *sg = malloc((size_t)C * E * sizeof(float));
    for(size_t i = 0; i < (size_t)E * D; i++){ float v = rndf() * 0.02f; uint32_t b; memcpy(&b, &v, 4); o.router16[i] = (uint16_t)(b >> 16); }
    for(size_t i = 0; i < (size_t)C * D; i++) x[i] = rndf();
    k3_router_scores(&o, sc, x, C, D, E);
    memset(sg, 0xff, (size_t)C * E * sizeof(float));
    int ok = k3cu_router(sg, x, o.router16, C, D, E);
    size_t diff = 0;
    for(size_t i = 0; i < (size_t)C * E; i++) if(memcmp(sc + i, sg + i, 4)) diff++;
    printf("router %d experts x %d, %d rows: %s%zu of %zu differ\n", E, D, C, ok ? "" : "FAILED to run; ", diff, (size_t)C * E);
    free(o.router16); free(x); free(sc); free(sg);
    return !ok || diff;
}

/* 6. The shared experts: k3cu_shared against k3_shared (three matmul_i4_grouped and situf_). */
static void rand_i4(W *w, int O, int I, int gs){
    memset(w, 0, sizeof *w);
    w->fmt = 4; w->O = O; w->I = I; w->gs = gs;
    w->q4 = malloc((size_t)O * (I / 2)); w->s = malloc((size_t)O * (I / gs) * sizeof(float));
    for(size_t i = 0; i < (size_t)O * (I / 2); i++) w->q4[i] = (uint8_t)(rnd32() >> 24);
    for(size_t i = 0; i < (size_t)O * (I / gs); i++) w->s[i] = 0.01f * (1.0f + 0.2f * rndf());
}
static int check_shared(int C, int D, int MI, int ns, int gs){
    Model *m = calloc(1, sizeof(Model));
    m->c.hidden = D; m->c.moe_inter = MI; m->c.n_shared = ns; m->c.situ_b1 = 4.0f; m->c.situ_b2 = 25.0f;
    const int I = MI * ns;
    Moe o; memset(&o, 0, sizeof o);
    rand_i4(&o.sh_gate, I, D, gs); rand_i4(&o.sh_up, I, D, gs); rand_i4(&o.sh_down, D, I, gs);
    float *x = malloc((size_t)C * D * sizeof(float)), *sg = malloc((size_t)C * D * sizeof(float));
    for(size_t i = 0; i < (size_t)C * D; i++) x[i] = rndf();
    float *sc = k3_shared(m, &o, x, C);
    memset(sg, 0xff, (size_t)C * D * sizeof(float));
    int ok = k3cu_shared(sg, x, C, D, I, o.sh_gate.q4, o.sh_gate.s, o.sh_up.q4, o.sh_up.s, gs, o.sh_down.q4, o.sh_down.s, gs, 4.0f, 25.0f);
    size_t diff = 0;
    for(size_t i = 0; i < (size_t)C * D; i++) if(memcmp(sc + i, sg + i, 4)) diff++;
    printf("shared experts %dx%d (%d of %d), %d rows: %s%zu of %zu differ\n", D, I, ns, MI, C, ok ? "" : "FAILED to run; ", diff, (size_t)C * D);
    free(sc); free(sg); free(x);
    W *ws[3] = {&o.sh_gate, &o.sh_up, &o.sh_down};
    for(int k = 0; k < 3; k++){ free(ws[k]->q4); free(ws[k]->s); }
    free(m);
    return !ok || diff;
}

/* 7. A prompt step's KDA layer: kda_forward through k3cu_kda against its by-step path on the
 *    CPU, the output, the recurrent state and the conv windows the same bytes. The projections
 *    in the mask `card` (1 q, 2 k, 4 v, 8 g, 16 o) go to coli_cuda.dll first: the CPU path then
 *    multiplies them there (w_matmul), the GPU path on that copy in its own kernel; the others
 *    stay the host's. Rows in several of k3cu_kda's pieces. */
static int g_cuda_up = -1;
/* 3b. The same on coli_cuda.dll's copy of the matrix (its int4 nibbles flipped), k3cu_matmul_dev: a
 *     decode step's 1 to 4 rows (k3_mv4, k3_mv8), a few more (k3_dense), a prompt's (k3_dense_b,
 *     k3_dense_b8), pieces cut short by K3_MMBYTES; every output matmul_i4_grouped's or matmul_q's bytes. */
static int check_dense_dev(int fmt, int S, int I, int O, int gs){
    if(g_cuda_up < 0){ int dev = 0; g_cuda_up = coli_cuda_init(&dev, 1) ? 1 : 0; }
    if(!g_cuda_up){ printf("dense on the card: no coli_cuda.dll: skipped\n"); return 0; }
    size_t wb = fmt == 4 ? (size_t)O * (I / 2) : (size_t)O * I, ns = fmt == 4 ? (size_t)O * (I / gs) : (size_t)O;
    uint8_t *w = malloc(wb); float *sc = malloc(ns * sizeof(float));
    float *x = malloc((size_t)S * I * sizeof(float)), *yc = malloc((size_t)S * O * sizeof(float)), *yg = malloc((size_t)S * O * sizeof(float));
    for(size_t i = 0; i < wb; i++) w[i] = (uint8_t)(rnd32() >> 24);
    for(size_t i = 0; i < ns; i++) sc[i] = rndf() * 0.01f;
    for(size_t i = 0; i < (size_t)S * I; i++) x[i] = rndf();
    if(fmt == 4) matmul_i4_grouped(yc, x, w, sc, S, I, O, gs);
    else matmul_q(yc, x, (const int8_t *)w, sc, S, I, O);
    memset(yg, 0xff, (size_t)S * O * sizeof(float));
    ColiCudaTensor *t = NULL;
    const void *dw = NULL; const float *ds = NULL; int tgs = 0;
    int ok = coli_cuda_tensor_upload_g(&t, w, sc, fmt, I, O, 0, fmt == 4 ? gs : 0) &&
             coli_cuda_tensor_dev_ptrs(t, &dw, &ds, &tgs) == fmt && k3cu_matmul_dev(fmt, yg, x, dw, ds, S, I, O, tgs);
    size_t diff = 0, first = (size_t)-1;
    for(size_t i = 0; i < (size_t)S * O; i++) if(memcmp(yc + i, yg + i, 4)){ if(first == (size_t)-1) first = i; diff++; }
    if(diff) printf("  output %zu: CPU %a, device %a\n", first, yc[first], yg[first]);
    printf("dense on the card %s %dx%d%s%s, %d rows: %s%zu of %zu differ\n", fmt == 4 ? "int4" : "int8", O, I,
           fmt == 4 ? ", groups of " : "", fmt == 4 ? (gs == 64 ? "64" : gs == 32 ? "32" : "128") : "", S,
           ok ? "" : "FAILED to run; ", diff, (size_t)S * O);
    if(t) coli_cuda_tensor_free(t);
    free(w); free(sc); free(x); free(yc); free(yg);
    return !ok || diff;
}
static void kda_rand(float *p, size_t n, float s){ for(size_t i = 0; i < n; i++) p[i] = rndf() * s; }
static int check_kda(int C, int D, int H, int hd, int K, int card){
    const int P = H * hd;
    Model *m = calloc(1, sizeof(Model));
    Layer *l = calloc(1, sizeof(Layer));
    m->c.hidden = D; m->c.kda_heads = H; m->c.kda_hd = hd; m->c.kda_proj = P; m->c.conv_k = K;
    m->c.gate_lb = -5.0f; m->c.eps = 1e-5f;
    l->kda = 1;
    Kda *a = &l->a;
    W *ws[5] = {&a->q, &a->k, &a->v, &a->g, &a->o};
    for(int i = 0; i < 5; i++){ const int I = i < 4 ? D : P; rand_i4(ws[i], i < 4 ? P : D, I, I < 64 ? I : 64); }
    a->fa = malloc((size_t)hd * D * 4); a->fb = malloc((size_t)P * hd * 4); a->bp = malloc((size_t)H * D * 4);
    a->conv_q = malloc((size_t)P * K * 4); a->conv_k = malloc((size_t)P * K * 4); a->conv_v = malloc((size_t)P * K * 4);
    a->dt = malloc((size_t)P * 4); a->A = malloc((size_t)H * 4); a->onw = malloc((size_t)hd * 4);
    kda_rand(a->fa, (size_t)hd * D, 0.02f); kda_rand(a->fb, (size_t)P * hd, 0.2f); kda_rand(a->bp, (size_t)H * D, 0.02f);
    kda_rand(a->conv_q, (size_t)P * K, 0.5f); kda_rand(a->conv_k, (size_t)P * K, 0.5f); kda_rand(a->conv_v, (size_t)P * K, 0.5f);
    kda_rand(a->dt, (size_t)P, 0.5f);
    for(int h = 0; h < H; h++) a->A[h] = 1.0f + 0.25f * rndf();
    for(int e = 0; e < hd; e++) a->onw[e] = 1.0f + 0.1f * rndf();
    float *kst = malloc((size_t)H * hd * hd * 4), *win[3], *st0 = malloc((size_t)H * hd * hd * 4), *w0[3], *stc = malloc((size_t)H * hd * hd * 4), *wc[3];
    kda_rand(st0, (size_t)H * hd * hd, 0.1f);
    for(int w = 0; w < 3; w++){
        win[w] = malloc((size_t)P * K * 4); w0[w] = malloc((size_t)P * K * 4); wc[w] = malloc((size_t)P * K * 4);
        kda_rand(w0[w], (size_t)P * K, 1.0f);
    }
    m->kstate = &kst; m->cwq = &win[0]; m->cwk = &win[1]; m->cwv = &win[2];
    float *x = malloc((size_t)C * D * 4), *oc = malloc((size_t)C * D * 4), *og = malloc((size_t)C * D * 4);
    kda_rand(x, (size_t)C * D, 1.0f);
    int up = 1;
    if(card){
        if(g_cuda_up < 0){ int dev = 0; g_cuda_up = coli_cuda_init(&dev, 1) ? 1 : 0; g_k3_cuda_dev = 0; }
        if(!g_cuda_up){ printf("KDA: no coli_cuda.dll: the card's projections skipped\n"); card = 0; }
        for(int i = 0; i < 5 && card; i++)
            if(card >> i & 1)
                up &= coli_cuda_tensor_upload_g((ColiCudaTensor **)&ws[i]->cuda, ws[i]->q4, ws[i]->s, 4, ws[i]->I, ws[i]->O, 0, 64);
        g_k3_cuda_exact = 1;
    }
    /* the CPU's: the by-step path, every projection the host's */
    memcpy(kst, st0, (size_t)H * hd * hd * 4);
    for(int w = 0; w < 3; w++) memcpy(win[w], w0[w], (size_t)P * K * 4);
    g_k3_kda_gpu = 0; g_k3g_on = 0; g_k3_cuda_dense = 0;
    kda_forward(m, l, 0, x, C, oc);
    memcpy(stc, kst, (size_t)H * hd * hd * 4);
    for(int w = 0; w < 3; w++) memcpy(wc[w], win[w], (size_t)P * K * 4);
    /* the GPU's, the card's projections there */
    memcpy(kst, st0, (size_t)H * hd * hd * 4);
    for(int w = 0; w < 3; w++) memcpy(win[w], w0[w], (size_t)P * K * 4);
    g_k3_kda_gpu = 1; g_k3g_on = 1; m->t_kda_gpu = 0; g_k3_cuda_dense = card != 0;
    memset(og, 0xff, (size_t)C * D * 4);
    kda_forward(m, l, 0, x, C, og);
    const int ran = m->t_kda_gpu > 0;
    g_k3g_on = 0;
    size_t dout = 0, dst = 0, dwin = 0, first = (size_t)-1;
    for(size_t i = 0; i < (size_t)C * D; i++) if(memcmp(oc + i, og + i, 4)){ if(first == (size_t)-1) first = i; dout++; }
    for(size_t i = 0; i < (size_t)H * hd * hd; i++) if(memcmp(stc + i, kst + i, 4)) dst++;
    for(int w = 0; w < 3; w++) for(size_t i = 0; i < (size_t)P * K; i++) if(memcmp(wc[w] + i, win[w] + i, 4)) dwin++;
    if(dout) printf("  output %zu: CPU %a, device %a\n", first, oc[first], og[first]);
    printf("KDA %d heads of %d, %d columns, conv %d, card's projections 0x%02x, %d rows: %s%zu of %zu outputs, %zu of %zu state, "
           "%zu of %zu window values differ\n", H, hd, D, K, card, C, ran && up ? "" : "FAILED to run on the GPU; ",
           dout, (size_t)C * D, dst, (size_t)H * hd * hd, dwin, (size_t)3 * P * K);
    if(card){   /* the by-step path with the card's projections: w_matmul's k3cu_matmul_dev */
        memcpy(kst, st0, (size_t)H * hd * hd * 4);
        for(int w = 0; w < 3; w++) memcpy(win[w], w0[w], (size_t)P * K * 4);
        g_k3_kda_gpu = 0;
        memset(og, 0xff, (size_t)C * D * 4);
        kda_forward(m, l, 0, x, C, og);
        size_t d2 = 0;
        for(size_t i = 0; i < (size_t)C * D; i++) d2 += memcmp(oc + i, og + i, 4) != 0;
        for(size_t i = 0; i < (size_t)H * hd * hd; i++) d2 += memcmp(stc + i, kst + i, 4) != 0;
        printf("KDA by step, the card's projections 0x%02x, %d rows: %zu of the outputs and state differ\n", card, C, d2);
        dout += d2;
    }
    g_k3_cuda_dense = 0;
    for(int i = 0; i < 5; i++){ if(ws[i]->cuda) coli_cuda_tensor_free((ColiCudaTensor *)ws[i]->cuda); free(ws[i]->q4); free(ws[i]->s); }
    free(a->fa); free(a->fb); free(a->bp); free(a->conv_q); free(a->conv_k); free(a->conv_v); free(a->dt); free(a->A); free(a->onw);
    free(kst); free(st0); free(stc);
    for(int w = 0; w < 3; w++){ free(win[w]); free(w0[w]); free(wc[w]); }
    free(x); free(oc); free(og); free(l); free(m);
    return !ran || !up || dout || dst || dwin;
}

/* 7. A prompt step's gated MLA layer whole: mla_forward through k3cu_mla_layer against its
 *    call-by-call path (the CPU's core, w_matmul's matrices): the output and the cache rows it
 *    writes the same bytes. The matrices in the mask `card` (1 qa, 2 qb, 4 kva, 8 g, 16 o) go to
 *    coli_cuda.dll first; i4 makes g and o int4 in groups of 64 (else int8 rows, as qa, qb, kva
 *    and kv_b). `want`: whether the layer path takes the step (not with qb on the card: no
 *    quant_matmul_blocked shape, the call-by-call path's bytes still). */
static void rand_i8(W *w, int O, int I){
    memset(w, 0, sizeof *w);
    w->fmt = 1; w->O = O; w->I = I;
    w->q8 = malloc((size_t)O * I); w->s = malloc((size_t)O * sizeof(float));
    for(size_t i = 0; i < (size_t)O * I; i++){ int v = (int)(rnd32() >> 24) - 128; w->q8[i] = (int8_t)(v < -127 ? -127 : v); }
    for(int r = 0; r < O; r++) w->s[r] = 0.0004f * (1.0f + 0.1f * rndf());
}
static int check_mla_layer(int C, int pos0, int D, int H, int qn, int qr, int vh, int kvl, int ql, int card, int i4, int want){
    const int N = pos0 + C, qh = qn + qr, KD = kvl + qr;
    Model *m = calloc(1, sizeof(Model));
    Layer *l = calloc(1, sizeof(Layer));
    Cfg *c = &m->c;
    c->hidden = D; c->n_heads = H; c->qk_nope = qn; c->qk_rope = qr; c->qk_head = qh; c->v_head = vh; c->kv_lora = kvl;
    c->q_lora = ql; c->eps = 1e-5f; c->attn_scale = 1.0f / sqrtf((float)qh);
    Mla *a = &l->m;
    rand_i8(&a->qa, ql, D); rand_i8(&a->qb, H * qh, ql); rand_i8(&a->kva, KD, D); rand_i8(&a->kvb, H * (qn + vh), kvl);
    if(i4){ rand_i4(&a->g, H * vh, D, 64); rand_i4(&a->o, D, H * vh, 64); }
    else { rand_i8(&a->g, H * vh, D); rand_i8(&a->o, D, H * vh); }
    a->qa_ln = malloc((size_t)ql * 4); a->kva_ln = malloc((size_t)kvl * 4);
    for(int i = 0; i < ql; i++) a->qa_ln[i] = 1.0f + 0.1f * rndf();
    for(int i = 0; i < kvl; i++) a->kva_ln[i] = 1.0f + 0.1f * rndf();
    const size_t nl = (size_t)N * kvl, nr = (size_t)N * (qr ? qr : 1);
    float *Lc = malloc(nl * 4), *Rc = malloc(nr * 4), *L0 = malloc(nl * 4), *R0 = malloc(nr * 4), *Lg = malloc(nl * 4), *Rg = malloc(nr * 4);
    for(size_t i = 0; i < nl; i++) L0[i] = i < (size_t)pos0 * kvl ? rndf() : -1.0f;   /* the steps before; this one's rows written */
    for(size_t i = 0; i < nr; i++) R0[i] = i < (size_t)pos0 * qr ? rndf() : -1.0f;
    m->Lc = &Lc; m->Rc = &Rc;
    float *x = malloc((size_t)C * D * 4), *oc = malloc((size_t)C * D * 4), *og = malloc((size_t)C * D * 4);
    for(size_t i = 0; i < (size_t)C * D; i++) x[i] = rndf();
    W *ws[5] = {&a->qa, &a->qb, &a->kva, &a->g, &a->o};
    int up = 1;
    if(card){
        if(g_cuda_up < 0){ int dev = 0; g_cuda_up = coli_cuda_init(&dev, 1) ? 1 : 0; g_k3_cuda_dev = 0; }
        if(!g_cuda_up){ printf("MLA layer: no coli_cuda.dll: the card's matrices skipped\n"); card = 0; }
        for(int i = 0; i < 5 && card; i++)
            if(card >> i & 1)
                up &= ws[i]->fmt == 4 ? coli_cuda_tensor_upload_g((ColiCudaTensor **)&ws[i]->cuda, ws[i]->q4, ws[i]->s, 4, ws[i]->I, ws[i]->O, 0, 64)
                                      : coli_cuda_tensor_upload_g((ColiCudaTensor **)&ws[i]->cuda, ws[i]->q8, ws[i]->s, 1, ws[i]->I, ws[i]->O, 0, 0);
        g_k3_cuda_exact = 1;
    }
    /* the call-by-call path, its core and every matrix the CPU's */
    memcpy(Lc, L0, nl * 4); memcpy(Rc, R0, nr * 4);
    g_k3_mla_gpu = 0; g_k3g_on = 0; g_k3_cuda_dense = 0;
    mla_forward(m, l, 0, x, pos0, C, oc);
    memcpy(Lg, Lc, nl * 4); memcpy(Rg, Rc, nr * 4);
    /* the layer path, the card's matrices there */
    memcpy(Lc, L0, nl * 4); memcpy(Rc, R0, nr * 4);
    const int min_rows = g_k3g_min_rows;
    g_k3_mla_gpu = 1; g_k3g_on = 1; g_k3g_min_rows = 1; g_k3_cuda_dense = card != 0;
    const unsigned long long before = g_k3_mla_layers;
    memset(og, 0xff, (size_t)C * D * 4);
    mla_forward(m, l, 0, x, pos0, C, og);
    const int ran = g_k3_mla_layers > before;
    g_k3g_on = 0; g_k3g_min_rows = min_rows;
    size_t dout = 0, dl = 0, dr = 0, first = (size_t)-1;
    for(size_t i = 0; i < (size_t)C * D; i++) if(memcmp(oc + i, og + i, 4)){ if(first == (size_t)-1) first = i; dout++; }
    for(size_t i = 0; i < nl; i++) if(memcmp(Lg + i, Lc + i, 4)) dl++;
    for(size_t i = 0; i < nr; i++) if(memcmp(Rg + i, Rc + i, 4)) dr++;
    if(card){   /* call by call with the card's matrices: w_matmul's k3cu_matmul_dev */
        memcpy(Lc, L0, nl * 4); memcpy(Rc, R0, nr * 4);
        g_k3_mla_gpu = 0;
        float *o2 = malloc((size_t)C * D * 4);
        mla_forward(m, l, 0, x, pos0, C, o2);
        size_t d2 = 0;
        for(size_t i = 0; i < (size_t)C * D; i++) d2 += memcmp(oc + i, o2 + i, 4) != 0;
        for(size_t i = 0; i < nl; i++) d2 += memcmp(Lg + i, Lc + i, 4) != 0;
        printf("MLA call by call, the card's 0x%02x, rows %d..%d: %zu of the outputs and cache values differ\n", card, pos0, N - 1, d2);
        dout += d2;
        free(o2);
    }
    g_k3_cuda_dense = 0;
    if(dout) printf("  output %zu: CPU %a, device %a\n", first, oc[first], og[first]);
    printf("MLA layer %d heads (%d+%d, v %d, latent %d, q %d), %d columns, card's 0x%02x%s, rows %d..%d: %s%zu of %zu outputs, "
           "%zu of %zu cache values differ\n", H, qn, qr, vh, kvl, ql, D, card, i4 ? " (g, o int4)" : "", pos0, N - 1,
           !up ? "FAILED to upload; " : ran != want ? (want ? "FAILED to run on the GPU; " : "RAN where it should not; ") : "",
           dout, (size_t)C * D, dl + dr, nl + (size_t)N * qr);
    for(int i = 0; i < 5; i++){ if(ws[i]->cuda) coli_cuda_tensor_free((ColiCudaTensor *)ws[i]->cuda); free(ws[i]->q8); free(ws[i]->q4); free(ws[i]->s); }
    free(a->kvb.q8); free(a->kvb.s); free(a->qa_ln); free(a->kva_ln);
    free(Lc); free(Rc); free(L0); free(R0); free(Lg); free(Rg); free(x); free(oc); free(og); free(l); free(m);
    return !up || ran != want || dout || dl || dr;
}
int dense_dev(void){
    int bad = 0;
    const int rows[] = { 1, 2, 3, 4, 5, 37, 64, 300, 1030 };
    for(int i = 0; i < 9; i++) bad += check_dense_dev(4, rows[i], 7168, 200, 64) + check_dense_dev(1, rows[i], 7168, 200, 0);
    bad += check_dense_dev(4, 1, 12288, 72, 64) + check_dense_dev(4, 300, 12288, 72, 64);   /* KDA's o_proj */
    bad += check_dense_dev(1, 1, 12288, 100, 0) + check_dense_dev(1, 260, 12288, 100, 0);  /* MLA's o_proj */
    bad += check_dense_dev(1, 300, 1536, 300, 0);   /* qb */
    bad += check_dense_dev(4, 70, 1024, 96, 32) + check_dense_dev(4, 1, 1024, 96, 32);     /* groups of 32: k3_dense */
    bad += check_dense_dev(1, 100, 1040, 50, 0) + check_dense_dev(1, 1, 1032, 50, 0);      /* widths k3_dense_b8 and k3_mv8 leave to k3_dense */
    bad += check_dense_dev(1, 1100, 7168, 60000, 0);   /* pieces of 998 rows (K3_MMBYTES), the head's case */
    return bad;
}
int mla_layers(void){
    return check_mla_layer(300, 200, 7168, 96, 128, 64, 128, 512, 1536, 8 | 16, 0, 1)   /* g and o on the card: K3's own */
         + check_mla_layer(40, 0, 7168, 96, 128, 64, 128, 512, 1536, 0, 1, 1)          /* every matrix the host's */
         + check_mla_layer(130, 3000, 7168, 96, 128, 64, 128, 512, 1536, 1 | 4 | 8 | 16, 1, 1)   /* chunks cut by their scores */
         + check_mla_layer(50, 10, 7168, 96, 128, 64, 128, 512, 1536, 2 | 8 | 16, 0, 1)     /* qb on the card too */
         + check_mla_layer(3, 20, 7168, 96, 128, 64, 128, 512, 1536, 1 | 2 | 4 | 8 | 16, 0, 1)   /* 3 rows: every matrix k3_mv8's */
         + check_mla_layer(600, 50, 128, 8, 16, 8, 16, 32, 64, 0, 0, 1);                 /* the tiny model's shapes, 3 chunks */
}

int main(int argc, char **argv){
    int all = argc > 1 && !strcmp(argv[1], "--all-floats");
    if(!k3cu_init(0)){ printf("no CUDA device or coli_cuda_k3.dll: skipped\n"); return 0; }
    if(argc > 1 && !strcmp(argv[1], "--kda")){   /* the KDA layer's cases alone */
        int bad = check_kda(40, 128, 2, 16, 4, 0) + check_kda(1100, 7168, 96, 128, 4, 0x1f) + check_kda(600, 7168, 96, 128, 4, 0x12);
        k3cu_report();
        printf(bad ? "FAILED\n" : "ok\n");
        return bad ? 1 : 0;
    }
    if(argc > 1 && !strcmp(argv[1], "--dev")){   /* coli_cuda.dll's matrices alone */
        int bad = dense_dev();
        k3cu_report();
        printf(bad ? "FAILED\n" : "ok\n");
        return bad ? 1 : 0;
    }
    if(argc > 1 && !strcmp(argv[1], "--mla")){   /* the MLA layer's cases alone */
        int bad = mla_layers();
        k3cu_report();
        printf(bad ? "FAILED\n" : "ok\n");
        return bad ? 1 : 0;
    }
    int bad = check_exact(all);
    const int real[] = { 1, 9, 64, 300, 1100, 2, 17, 8, 33, 5, 3 };
    bad += check_experts(3584, 3072, 4.0f, 25.0f, real, (int)(sizeof real / sizeof real[0]));
    const int tiny[] = { 1, 7, 8, 9, 40 };
    bad += check_experts(64, 32, 1.0f, 1.0f, tiny, 5);
    bad += check_dense(4, 37, 7168, 200, 64);
    bad += check_dense(4, 1100, 1024, 96, 32);
    bad += check_dense(4, 9, 3584, 71, 128);
    bad += check_dense(4, 100, 7168, 64, 64);   /* 12 blocks of 8 rows and one of 4 */
    bad += check_dense(4, 300, 7168, 200, 64);  /* k3_dense_b: 9 blocks of 32 rows and one of 12, 192 outputs and 8 */
    bad += check_dense(4, 1030, 3584, 7168, 64);/* the latent's up: a 1024-row piece by k3_dense_b, 6 rows by k3_dense */
    bad += check_dense(1, 33, 7168, 100, 0);
    bad += check_dense(1, 1030, 512, 40, 0);    /* k3_dense_b8: a 1024-row piece; 6 rows by k3_dense */
    bad += check_dense(4, 1, 7168, 300, 64);    /* k3_mv4 */
    bad += check_dense(4, 3, 3584, 70, 64);
    bad += check_dense(1, 1, 7168, 300, 0);     /* k3_mv8 */
    bad += check_dense(1, 4, 1536, 70, 0);
    bad += dense_dev();
    bad += check_mla(96, 128, 64, 128, 512, 100, 40);
    bad += check_mla(96, 128, 64, 128, 512, 1000, 300);   /* two chunks of rows, the scores' blocks past nt skipped */
    bad += check_mla(96, 128, 64, 128, 512, 3000, 200);   /* chunks cut to their scores' 128 MB */
    bad += check_mla(8, 16, 8, 16, 32, 0, 37);
    bad += check_mla(8, 16, 8, 16, 32, 50, 600);          /* three chunks: a buffer used twice */
    bad += check_mla(5, 24, 8, 40, 64, 7, 300);           /* partial tiles every way */
    bad += check_router(37, 7168, 896);
    bad += check_router(70, 128, 13);
    bad += check_shared(40, 1024, 768, 2, 64);
    bad += check_shared(1030, 256, 128, 1, 32);
    bad += check_shared(300, 7168, 3072, 2, 64);   /* Kimi K3's, through k3_dense_b */
    bad += check_kda(40, 128, 2, 16, 4, 0);         /* the tiny model's shapes, every projection the host's */
    bad += check_kda(1100, 7168, 96, 128, 4, 0x1f); /* Kimi K3's, every projection on the card: 3 pieces */
    bad += check_kda(600, 7168, 96, 128, 4, 0x12);  /* k and o on the card, q, v and g the host's */
    bad += mla_layers();
    k3cu_report();
    printf(bad ? "FAILED\n" : "ok\n");
    return bad ? 1 : 0;
}
