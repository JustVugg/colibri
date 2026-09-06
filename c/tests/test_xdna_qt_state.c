/* The XDNA prepared-state side pointer inside QT.
 *
 * This owner compiles the real engine with -DCOLI_XDNA so the QT field, its
 * reset helper and the expert-slot wiring are exercised as the engine actually
 * builds them -- the same #define main / #include seam c/tests/test_i4_grouped.c
 * uses to reach static engine internals.
 *
 * The properties under test are ownership properties: a tensor starts with no
 * derived state, nothing creates it implicitly, dropping it never touches the
 * authoritative weight, and exactly one owner ever frees it.
 */

#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>

static int g_fail = 0;

static void ck(int cond, const char *what){
    printf("  %-60s %s\n", what, cond ? "ok" : "FAIL");
    if(!cond) g_fail = 1;
}

int main(void){
    printf("QT xdna side pointer\n");

    /* A tensor built the way the loader builds one: authoritative fmt=4 bytes,
     * and no derived state of any kind. */
    static unsigned char q4[256];
    static float scales[8];
    for(int i = 0; i < 256; i++) q4[i] = (unsigned char)(i * 3 + 5);
    for(int i = 0; i < 8; i++)   scales[i] = 0.5f * (float)(i + 1);

    QT t;
    memset(&t, 0, sizeof t);
    t.fmt = 4; t.I = 64; t.O = 4; t.gs = 64;
    t.q4 = q4; t.s = scales;

    ck(t.xdna == NULL, "a fresh tensor carries no prepared state");
    ck(coli_xdna_prepared_live_objects() == 0, "and none exists anywhere");

    /* Nothing implicit: asking the registry a question must not allocate. */
    ColiXdnaRequest q;
    memset(&q, 0, sizeof q);
    q.family = COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP;
    q.m = 64; q.k = 6144; q.n = 2048;
    q.in_dtype = COLI_XDNA_DT_BF16; q.weight_dtype = COLI_XDNA_DT_BF16;
    q.out_dtype = COLI_XDNA_DT_F32;
    (void)coli_xdna_registry_lookup(&q);
    (void)coli_xdna_artifact_status(&q, NULL);
    ck(t.xdna == NULL, "a registry query creates no prepared state");
    ck(coli_xdna_prepared_live_objects() == 0, "still none anywhere");

    /* Explicit creation is the only way. */
    t.xdna = coli_xdna_prepared_create();
    ck(t.xdna != NULL, "prepared state attaches on explicit request");
    ck(coli_xdna_prepare_begin(t.xdna, 64, 64, COLI_XDNA_DT_BF16) == 1, "begin");
    ck(coli_xdna_prepare_publish_success(t.xdna) == 1, "publish");
    ck(coli_xdna_prepared_total_bytes() == 64*64*2, "bytes accounted while attached");

    /* The reset the expert-slot reuse path calls. It must take the derived
     * state and leave the authoritative weight exactly as it was. */
    unsigned char q4_before[256]; float sc_before[8];
    memcpy(q4_before, q4, sizeof q4); memcpy(sc_before, scales, sizeof scales);
    const void *q4_ptr = t.q4; const void *s_ptr = t.s;
    int fmt_before = t.fmt, I_before = t.I, O_before = t.O, gs_before = t.gs;

    qt_xdna_reset(&t);

    ck(t.xdna == NULL, "reset detaches the derived state");
    ck(coli_xdna_prepared_total_bytes() == 0, "and returns its bytes");
    ck(coli_xdna_prepared_live_objects() == 0, "and frees the object exactly once");
    ck(t.q4 == q4_ptr && t.s == s_ptr, "authoritative pointers untouched");
    ck(memcmp(q4, q4_before, sizeof q4) == 0, "fmt4 bytes untouched");
    ck(memcmp(scales, sc_before, sizeof scales) == 0, "scales untouched");
    ck(t.fmt == fmt_before && t.I == I_before && t.O == O_before && t.gs == gs_before,
       "QT metadata untouched");

    /* Repeated reset, and reset of a tensor that never had derived state. */
    qt_xdna_reset(&t);
    qt_xdna_reset(&t);
    ck(t.xdna == NULL, "repeated reset is safe");

    QT plain;
    memset(&plain, 0, sizeof plain);
    plain.fmt = 4; plain.I = 8; plain.O = 8; plain.gs = 8;
    qt_xdna_reset(&plain);
    ck(plain.xdna == NULL, "reset of a tensor with no derived state is safe");


    /* ---- cross-check against the PRODUCTION decode kernel ----------------
     * The strongest oracle available: quant.h's matmul_i4_grouped is the code
     * the engine actually computes with. Feeding it a one-hot activation makes
     * y[o] equal the decoded weight at (o, i), so the prepared image can be
     * checked against the real kernel rather than against a second copy of my
     * own reading of the format. */
    printf("cross-check vs production matmul_i4_grouped\n");
    {
        const int I = 70, O = 5, gs = 64;          /* odd-ish I, partial 2nd group */
        const int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
        unsigned char *pq4 = (unsigned char*)malloc((size_t)O * rb);
        float *psc = (float*)malloc((size_t)O * ng * sizeof(float));
        unsigned st = 4242u;
        for(size_t k = 0; k < (size_t)O * rb; k++){ st = st*1664525u+1013904223u; pq4[k] = (unsigned char)(st >> 24); }
        for(size_t k = 0; k < (size_t)O * ng; k++){ st = st*1664525u+1013904223u;
            psc[k] = 0.25f * (float)(1 + ((st >> 26) & 3)); }

        ColiXdnaPrepared *pp = coli_xdna_prepared_create();
        ck(coli_xdna_prepare_from_fmt4(pp, 4, pq4, psc, I, O, gs) == COLI_XDNA_PREP_OK,
           "prepare from fmt4");
        const unsigned short *img = (const unsigned short*)coli_xdna_prepared_image(pp);

        float *x = (float*)calloc((size_t)I, sizeof(float));
        float *y = (float*)malloc((size_t)O * sizeof(float));
        size_t mism = 0;
        for(int i = 0; i < I; i++){
            memset(x, 0, (size_t)I * sizeof(float));
            x[i] = 1.0f;                            /* one-hot: y[o] = W[o][i] */
            matmul_i4_grouped(y, x, pq4, psc, 1, I, O, gs);
            for(int o = 0; o < O; o++){
                unsigned int u; memcpy(&u, &y[o], 4);
                u += 0x7FFFu + ((u >> 16) & 1u);
                unsigned short want = (unsigned short)(u >> 16);
                if(img[(size_t)i * O + o] != want) mism++;
            }
        }
        ck(mism == 0, "prepared image matches the production kernel decode exactly");
        free(x); free(y); free(pq4); free(psc);
        coli_xdna_prepared_release(&pp);
    }

    ck(coli_xdna_prepared_live_objects() == 0, "no object leaked");
    ck(coli_xdna_prepared_total_bytes() == 0, "no bytes leaked");

    printf("test_xdna_qt_state: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail;
}
