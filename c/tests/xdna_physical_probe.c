/* W2-N7-I5 -- PHYSICAL qualification of the native XDNA lane.
 *
 * NOT a test gate: it requires an XDNA2 NPU, a working XRT installation, the
 * built coli_xdna.dll and the qualified F3 artifact bytes, none of which a
 * build machine is entitled to assume. It is built and run explicitly.
 *
 * Every step below is PRODUCTION code -- the production registry, the
 * production integrity check, the production loader, the production weight
 * preparation, the production hard-eligibility gates and the production
 * candidate function that the GLM shared gate/up call sites invoke. This file
 * supplies only a synthetic tensor and synthetic activations, and the oracles.
 *
 *   usage: xdna_physical_probe <artifact-root> <helper-dll> [M-list]
 *
 * M-list is a comma-separated list of M values; it defaults to the frozen
 * 1,32,64 (the M64 bucket). Passing 65,130,256 qualifies the M256 bucket
 * through this same owner, so neither bucket needs a throwaway probe.
 */

#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include "../backend_xdna.h"

#define PK 6144      /* the frozen F3 K */
#define PN 2048      /* the frozen F3 N */
#define PGS 64
#define COLI_PROBE_MAX_M 16

static int g_bad = 0;

static float b2f(unsigned short b){
    unsigned int u = (unsigned int)b << 16; float f; memcpy(&f,&u,4); return f;
}
static unsigned short f2b_local(float f){
    unsigned int u; memcpy(&u,&f,4);
    u += 0x7FFFu + ((u>>16)&1u);
    return (unsigned short)(u>>16);
}

/* ---- the two BF16 oracles, and why there are two -------------------------
 *
 * ORACLE A -- ACCEPTANCE. The criterion frozen by N6-A1C-1 and reused unchanged
 * by A1A: take the exact BF16 inputs the lane submits, accumulate in DOUBLE,
 * round once to f32, and accept on absolute 0.5 OR relative 0.05. A1C-1's own
 * note on this oracle is that "the only error present is the device's
 * accumulation order", which is precisely what it is built to tolerate. This is
 * a pre-declared research criterion, not a threshold chosen after seeing
 * output.
 *
 * ORACLE B -- CONTINUITY. The same BF16 inputs accumulated sequentially in f32,
 * which is the comparator A5-R1 reported bit-exactness against.
 *
 * They are kept separate because bit-exactness against Oracle B is NOT a
 * universal property, and this slice is where that became visible. A5-R1's
 * fixture used activations that are exact multiples of 2^-7 and weights that
 * are small multiples of 0.25 -- every value exactly representable in BF16, and
 * every partial sum exact -- so accumulation order could not matter and any
 * order gave the identical result. Once activations carry a full f32 mantissa,
 * partial sums round, order matters, and the device legitimately differs from
 * one particular host summation order. Oracle B is therefore REPORTED at both
 * fixtures and GATES at neither; Oracle A is the acceptance criterion. */

static const float ABS_TOL = 0.5f, REL_TOL = 0.05f;
static int accept_a1c1(float expected, float actual){
    float d = fabsf(expected - actual);
    if(d <= ABS_TOL) return 1;
    float norm = fabsf(expected) + fabsf(actual);
    return d <= norm * REL_TOL;
}

static void oracle_double(float *out, const float *x, const unsigned short *W,
                          int S, int K, int N){
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < S; i++)
        for(int j = 0; j < N; j++){
            double acc = 0.0;
            for(int k = 0; k < K; k++)
                acc += (double)b2f(f2b_local(x[(size_t)i*K+k])) * (double)b2f(W[(size_t)k*N+j]);
            out[(size_t)i*N+j] = (float)acc;
        }
}

static void oracle_bf16(float *out, const float *x, const unsigned short *W,
                        int S, int K, int N){
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < S; i++)
        for(int j = 0; j < N; j++){
            float acc = 0.0f;
            for(int k = 0; k < K; k++)
                acc += b2f(f2b_local(x[(size_t)i*K+k])) * b2f(W[(size_t)k*N+j]);
            out[(size_t)i*N+j] = acc;
        }
}

/* Report both oracles for one result. Returns 1 when the ACCEPTANCE criterion
 * holds. */
static int report_oracles(const float *y, const float *x, const unsigned short *W,
                          int S, int K, int N){
    float *refA = (float*)malloc((size_t)S*N*4);
    float *refB = (float*)malloc((size_t)S*N*4);
    oracle_double(refA, x, W, S, K, N);
    oracle_bf16(refB, x, W, S, K, N);
    size_t mism = 0, bitmis = 0, nan_n = 0, inf_n = 0;
    double maxabsA = 0, maxrelA = 0, maxabsB = 0;
    for(size_t i = 0; i < (size_t)S*N; i++){
        if(!accept_a1c1(refA[i], y[i])) mism++;
        double ae = fabs((double)y[i]-(double)refA[i]);
        if(ae > maxabsA) maxabsA = ae;
        double den = fabs((double)refA[i]);
        if(den > 1e-6 && ae/den > maxrelA) maxrelA = ae/den;
        if(y[i] != refB[i]) bitmis++;
        double be = fabs((double)y[i]-(double)refB[i]);
        if(be > maxabsB) maxabsB = be;
        if(y[i] != y[i]) nan_n++;
        else if(y[i] > 3.0e38f || y[i] < -3.0e38f) inf_n++;
    }
    printf("ORACLE_A_ACCEPT  mismatches=%llu (criterion abs<=0.5 OR rel<=0.05)  max_abs=%.9f  max_rel=%.9f\n",
           (unsigned long long)mism, maxabsA, maxrelA);
    printf("ORACLE_B_EXACT   bit_mismatches=%llu  max_abs=%.9f\n",
           (unsigned long long)bitmis, maxabsB);
    printf("NAN              %llu\nINF              %llu\n",
           (unsigned long long)nan_n, (unsigned long long)inf_n);
    free(refA); free(refB);
    return mism == 0;
}

int main(int argc, char **argv){
    if(argc < 3){ fprintf(stderr,"usage: xdna_physical_probe <artifact-root> <helper-dll> [M-list]\n"); return 2; }
    const char *root = argv[1], *helper = argv[2];

    /* Which M values to qualify. Default = the frozen M64 set; a caller may
     * pass e.g. 65,130,256 to qualify the M256 bucket through this same
     * owner. Only the shapes change -- every gate below is production. */
    int Ms[COLI_PROBE_MAX_M]; size_t nM = 0;
    if(argc > 3){
        const char *p = argv[3];
        while(*p && nM < COLI_PROBE_MAX_M){
            char *end; long v = strtol(p, &end, 10);
            if(end == p || v < 1 || v > 100000){
                fprintf(stderr,"bad M-list: %s\n", argv[3]); return 2; }
            Ms[nM++] = (int)v;
            p = end; while(*p == ',' || *p == ' ') p++;
        }
        if(!nM){ fprintf(stderr,"bad M-list: %s\n", argv[3]); return 2; }
    } else {
        Ms[0] = 1; Ms[1] = 32; Ms[2] = 64; nM = 3;
    }

    printf("W2-N7-I5 PHYSICAL XDNA QUALIFICATION\n");
    printf("PID              %lu\n", (unsigned long)GetCurrentProcessId());
    printf("ARTIFACT_ROOT    %s\n", root);
    printf("HELPER           %s\n", helper);
    printf("REGISTRY         production (not a test registry)\n");
    printf("M_VALUES        ");
    for(size_t c = 0; c < nM; c++) printf(" %d", Ms[c]);
    printf("\n");

    coli_xdna_test_set_helper_path(helper);
    coli_xdna_test_set_artifact_root(root);

    printf("BINDING          %s\n", coli_xdna_binding_text(coli_xdna_binding()));
    if(coli_xdna_binding() != COLI_XDNA_AVAILABLE){
        printf("RESULT           FAIL -- helper did not bind\n"); return 1; }
    printf("ABI              host %u\n", (unsigned)COLI_XDNA_ABI_VERSION);

    /* Production static verdict, on the production registry and the real bytes. */
    ColiXdnaRequest q;
    q.family = COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP;
    q.m = 64; q.k = PK; q.n = PN;
    q.in_dtype = COLI_XDNA_DT_BF16; q.weight_dtype = COLI_XDNA_DT_BF16;
    q.out_dtype = COLI_XDNA_DT_F32;
    ColiXdnaStatic st = coli_xdna_static_eligibility(&q, root);
    printf("STATIC           %s\n", coli_xdna_static_text(st));
    if(st != COLI_XDNA_STATIC_QUALIFIED){
        printf("RESULT           FAIL -- artifact integrity gate refused\n"); return 1; }

    /* Two genuinely different runtime weights: gate-like and up-like. */
    const int rb = (PK+1)/2, ng = (PK+PGS-1)/PGS;
    QT G, U;
    memset(&G,0,sizeof G); memset(&U,0,sizeof U);
    QT *ts[2] = { &G, &U };
    unsigned seeds[2] = { 20260824u, 77771111u };
    for(int t = 0; t < 2; t++){
        QT *w = ts[t];
        w->fmt = 4; w->I = PK; w->O = PN; w->gs = PGS;
        w->q4 = (uint8_t*)malloc((size_t)PN*rb);
        w->s  = (float*)malloc((size_t)PN*ng*sizeof(float));
        unsigned s = seeds[t];
        for(size_t i=0;i<(size_t)PN*rb;i++){ s=s*1664525u+1013904223u; w->q4[i]=(uint8_t)(s>>24); }
        for(size_t i=0;i<(size_t)PN*ng;i++){ s=s*1664525u+1013904223u;
            w->s[i] = 0.25f*(float)(1+((s>>26)&3)); }
    }

    coli_xdna_test_set_force_execution(1);

    const char *names[2] = { "sh_gate-like", "sh_up-like" };
    unsigned long long hashes[2][COLI_PROBE_MAX_M];

    for(int t = 0; t < 2; t++){
        for(size_t c = 0; c < nM; c++){
            int S = Ms[c];
            QT *w = ts[t];
            float *x = (float*)malloc((size_t)S*PK*4);
            unsigned s = 31337u + (unsigned)S + 1000u*(unsigned)t;
            for(size_t i=0;i<(size_t)S*PK;i++){ s=s*1664525u+1013904223u;
                x[i] = (float)((int)((s>>20)&255)-128)*0.0078125f; }
            float *y = (float*)malloc((size_t)S*PN*4);
            for(size_t i=0;i<(size_t)S*PN;i++) y[i] = -123456.0f;

            int d0 = coli_xdna_test_dispatches();
            double t0 = now_s();
            int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,
                                               &w->xdna, w->fmt, w->q4, w->s,
                                               w->I, w->O, w->gs, w->planar, y, x, S);
            double ms = (now_s()-t0)*1000.0;

            printf("\n-- %s  M=%d --\n", names[t], S);
            printf("HANDLED          %d\n", handled);
            printf("HARD             %s\n", coli_xdna_hard_text(coli_xdna_test_last_hard()));
            printf("EXEC             %s\n", coli_xdna_exec_text(coli_xdna_test_last_exec()));
            if(!handled){
                printf("HELPER_ERR       (see loader)\n");
                printf("RESULT           FAIL -- native execution declined or failed\n");
                g_bad = 1; free(x); free(y); continue;
            }
            printf("DISPATCHES       %d (+%d)\n", coli_xdna_test_dispatches(),
                   coli_xdna_test_dispatches()-d0);
            printf("ARTIFACT_OPENS   %d\n", coli_xdna_test_artifact_opens());
            printf("USERPTR_WRAPS    %d\n", coli_xdna_test_userptr_wraps());
            printf("PADDED_OPS       %d\n", coli_xdna_test_padded_operations());
            printf("ELAPSED_MS       %.3f\n", ms);

            const unsigned short *W = (const unsigned short*)coli_xdna_prepared_image(w->xdna);
            printf("PREPARED_PTR     %p  mod4096=%llu\n", (const void*)W,
                   (unsigned long long)((uintptr_t)W % 4096u));
            printf("PREPARED_BYTES   %llu\n",
                   (unsigned long long)coli_xdna_prepared_bytes(w->xdna));

            if(!report_oracles(y, x, W, S, PK, PN)) g_bad = 1;

            /* ---- ORACLE B: the exact current path, for characterisation ---- */
            float *refB = (float*)malloc((size_t)S*PN*4);
            matmul_qt(refB, x, w, S);
            double maxabsB = 0, maxrelB = 0, sse = 0;
            for(size_t i=0;i<(size_t)S*PN;i++){
                double a = (double)y[i], b = (double)refB[i], d = fabs(a-b);
                if(d>maxabsB) maxabsB=d;
                double den = fabs(b);
                if(den > 1e-12 && d/den > maxrelB) maxrelB = d/den;
                sse += d*d;
            }
            printf("VS_MATMUL_QT     max_abs=%.6f  max_rel=%.3f  rms=%.6f\n",
                   maxabsB, maxrelB, sqrt(sse/((double)S*PN)));

            unsigned long long h = 1469598103934665603ULL;
            for(size_t i=0;i<(size_t)S*PN*4;i++){ h ^= ((const unsigned char*)y)[i];
                h *= 1099511628211ULL; }
            hashes[t][c] = h;
            printf("OUTPUT_HASH      %016llx\n", h);

            free(refB); free(x); free(y);
        }
    }

    printf("\n-- dynamic runtime weights --\n");
    for(size_t c = 0; c < nM; c++)
        printf("M=%-3d  gate=%016llx  up=%016llx  distinct=%s\n", Ms[c],
               hashes[0][c], hashes[1][c],
               hashes[0][c]!=hashes[1][c] ? "YES" : "NO");
    for(size_t c = 0; c < nM; c++) if(hashes[0][c]==hashes[1][c]) g_bad = 1;

    printf("\nARTIFACT_OPENS_TOTAL %d  (artifacts opened across every M and both weights)\n",
           coli_xdna_test_artifact_opens());
    printf("DISPATCHES_TOTAL     %d\n", coli_xdna_test_dispatches());
    printf("COMPLETIONS_TOTAL    %d\n", coli_xdna_test_completions());
    printf("DEVICE_OPENS         %d\n", coli_xdna_test_device_opens());
    printf("CONVERSIONS          %d\n", coli_xdna_test_conversions());

    /* ---- characterising the difference against the CURRENT path -------------
     *
     * The activations above are all exact multiples of 2^-7 and the weights all
     * small multiples of 0.25, so every value is exactly representable in BF16
     * and the f32->BF16 conversion is lossless. That makes the current path and
     * the lane agree exactly -- a true result, but one that characterises
     * nothing, because it never exercises the conversion the two paths differ
     * over.
     *
     * This case uses activations with a full f32 mantissa, which BF16 genuinely
     * rounds. The lane must STILL be bit-exact against its own BF16 oracle; the
     * difference against matmul_qt is the real, expected representational gap,
     * reported and not tested against any invented threshold. */
    {
        int S = 64;
        float *x = (float*)malloc((size_t)S*PK*4);
        unsigned s = 0xC0FFEEu;
        for(size_t i=0;i<(size_t)S*PK;i++){
            s = s*1664525u+1013904223u;
            /* a generic f32 in [-1,1): 24 significant bits, not BF16-exact */
            x[i] = ((float)(s >> 8) / (float)(1u<<24)) * 2.0f - 1.0f;
        }
        float *y = (float*)malloc((size_t)S*PN*4);
        int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,
                                           &G.xdna, G.fmt, G.q4, G.s,
                                           G.I, G.O, G.gs, G.planar, y, x, S);
        printf("\n-- generic f32 activations (BF16 rounding is LOSSY here), M=%d --\n", S);
        printf("HANDLED          %d\n", handled);
        if(!handled) g_bad = 1;
        else {
            const unsigned short *W = (const unsigned short*)coli_xdna_prepared_image(G.xdna);
            if(!report_oracles(y, x, W, S, PK, PN)) g_bad = 1;

            float *refB = (float*)malloc((size_t)S*PN*4);
            matmul_qt(refB, x, &G, S);
            double maxabsB=0, maxrelB=0, sse=0, refmax=0;
            for(size_t i=0;i<(size_t)S*PN;i++){
                double a=(double)y[i], b=(double)refB[i], d=fabs(a-b);
                if(d>maxabsB) maxabsB=d;
                if(fabs(b)>refmax) refmax=fabs(b);
                if(fabs(b)>1e-12 && d/fabs(b) > maxrelB) maxrelB = d/fabs(b);
                sse += d*d;
            }
            printf("VS_MATMUL_QT     max_abs=%.6f  max_rel=%.3f  rms=%.6f  ref_max_abs=%.6f\n",
                   maxabsB, maxrelB, sqrt(sse/((double)S*PN)), refmax);
            printf("INTERPRETATION   expected representational gap: matmul_qt accumulates f32\n");
            printf("                 activations against dequantised int4; the lane is BF16\n");
            printf("                 throughout. max_rel is inflated by near-zero reference\n");
            printf("                 dot products and is NOT an error measure.\n");
            free(refB);
        }
        free(x); free(y);
    }

    /* ---- stale-wrapper control -------------------------------------------
     * Re-prepare a DIFFERENT weight into the SAME prepared object. Capacity is
     * retained, so the buffer pointer is unchanged. If the lane keys its
     * userptr wrapper on the pointer alone it skips the re-wrap, and the device
     * -- which snapshotted at wrap time via sync(BO_TO_DEVICE) -- computes with
     * the previous weight. */
    {
        int S = 64;
        float *x = (float*)malloc((size_t)S*PK*4);
        unsigned s = 0xBEEF01u;
        for(size_t i=0;i<(size_t)S*PK;i++){ s=s*1664525u+1013904223u;
            x[i] = (float)((int)((s>>20)&255)-128)*0.0078125f; }
        float *ya = (float*)malloc((size_t)S*PN*4);
        float *yb = (float*)malloc((size_t)S*PN*4);
        printf("\n-- stale-wrapper control: reprepare a different weight in place --\n");
        int ha = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, &G.xdna,
                                      G.fmt, G.q4, G.s, G.I, G.O, G.gs, G.planar, ya, x, S);
        const void *p1 = coli_xdna_prepared_image(G.xdna);
        int w0 = coli_xdna_test_userptr_wraps();
        coli_xdna_prepared_invalidate(G.xdna);
        ColiXdnaPrepResult pr = coli_xdna_prepare_from_fmt4(G.xdna, 4, U.q4, U.s, PK, PN, PGS);
        const void *p2 = coli_xdna_prepared_image(G.xdna);
        int hb = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, &G.xdna,
                                      G.fmt, G.q4, G.s, G.I, G.O, G.gs, G.planar, yb, x, S);
        printf("PREPARE_RC       %d   ptr_same %s\n", (int)pr, p1==p2?"YES":"no");
        printf("HANDLED          %d %d\n", ha, hb);
        printf("WRAPS            before=%d after=%d  rewrap=%s\n", w0,
               coli_xdna_test_userptr_wraps(),
               coli_xdna_test_userptr_wraps()>w0 ? "HAPPENED" : "SKIPPED");
        if(ha && hb){
            const unsigned short *W = (const unsigned short*)p2;
            float *ref = (float*)malloc((size_t)S*PN*4);
            oracle_double(ref, x, W, S, PK, PN);
            size_t mism=0, sameA=0;
            for(size_t i=0;i<(size_t)S*PN;i++){
                if(!accept_a1c1(ref[i], yb[i])) mism++;
                if(yb[i]==ya[i]) sameA++;
            }
            printf("SECOND_VS_ITS_OWN_WEIGHT  mismatches %llu / %d\n",
                   (unsigned long long)mism, S*PN);
            printf("SECOND_VS_FIRST_RESULT    identical  %llu / %d\n",
                   (unsigned long long)sameA, S*PN);
            if(mism){ printf("WEIGHT_VIEW      DEFECT -- device view not resynchronised\n"); g_bad = 1; }
            else      printf("WEIGHT_VIEW      coherent -- the re-wrap happened\n");
            free(ref);
        } else { printf("WEIGHT_VIEW      inconclusive (an operation declined)\n"); g_bad = 1; }
        free(x); free(ya); free(yb);
    }

    /* Negative control on live hardware: same shape, wrong family. */
    {
        float *x = (float*)malloc((size_t)8*PK*4); memset(x,0,(size_t)8*PK*4);
        float *y = (float*)malloc((size_t)8*PN*4);
        int before = coli_xdna_test_dispatches();
        int h2 = coli_xdna_try_matmul(COLI_XDNA_FAMILY_NONE, &G.xdna, G.fmt, G.q4, G.s,
                                      G.I, G.O, G.gs, G.planar, y, x, 8);
        printf("\nWRONG_FAMILY_HANDLED %d  dispatch_delta %d  (%s)\n", h2,
               coli_xdna_test_dispatches()-before,
               coli_xdna_hard_text(coli_xdna_test_last_hard()));
        if(h2 || coli_xdna_test_dispatches()!=before) g_bad = 1;
        free(x); free(y);
    }

    coli_xdna_execution_shutdown();
    coli_xdna_shutdown();
    printf("\nSHUTDOWN         clean\n");
    printf("RESULT           %s\n", g_bad ? "FAIL" : "PASS");
    return g_bad;
}
