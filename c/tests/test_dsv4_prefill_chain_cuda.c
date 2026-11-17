/* V4_PREFILL_CHAIN: dsv4_cuda_attn_prefill_chain against the separate calls the
 * engine's batched prompt attention makes without it, on the same inputs: the
 * batched wq_b GEMM (dsv4_cuda_matmul_batch), the bf16 round, the per-head
 * rmsnorm and the forward RoPE on the host (the fused-multiply-add forms the
 * chain is given: 1, as the engine passes them), the cached sparse attention
 * (dsv4_cuda_sparse_attn_batch_cached, and _idx with a selection), the
 * inverse RoPE on the host and the grouped wo_a + wo_b
 * (dsv4_cuda_attention_output_batch). The outputs must match bit for bit, for
 * the plain and the indexed selection. Needs a GPU. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

enum { HEADS = 64, DIM = 512, ROPE = 64, QR = 1024, GROUPS = 8, ORANK = 1024, HIDDEN = 4096,
       WINDOW = 128, T = 48, START = 300, COMP = 75, SEL = 40, LAYER = 3 };

static unsigned g_seed = 2026;
static float frand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return ((g_seed >> 8) & 0xffff) / 65536.0f - 0.5f;
}
static unsigned urand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}
static float bf16_round(float x) {
    uint32_t bits;
    memcpy(&bits, &x, 4);
    bits += 0x7fffu + ((bits >> 16) & 1);
    bits &= 0xffff0000u;
    memcpy(&x, &bits, 4);
    return x;
}
/* e4m3 codes without the NaN pair (0x7f, 0xff) */
static Dsv4CudaTensor *fp8_tensor(int O, int I, int bf16_flavour) {
    size_t wn = (size_t)O * I, sn = (size_t)((O + 127) / 128) * ((I + 127) / 128);
    uint8_t *w = malloc(wn), *s = malloc(sn);
    Dsv4CudaTensor *t = NULL;
    if (!w || !s) return NULL;
    for (size_t i = 0; i < wn; i++) { uint8_t b = (uint8_t)(urand() % 0x70); w[i] = (uint8_t)(b | ((urand() & 1) << 7)); }
    for (size_t i = 0; i < sn; i++) s[i] = (uint8_t)(120 + urand() % 4);
    int ok = bf16_flavour ? dsv4_cuda_upload_fp8_bf16(&t, w, s, O, I, 0) : dsv4_cuda_upload_fp8(&t, w, s, O, I, 0);
    free(s); free(w);
    return ok ? t : NULL;
}
static void rope(float *v, const float *c, const float *s, int inverse) {
    for (int i = 0; i < ROPE / 2; i++) {
        float re = v[2 * i], im = v[2 * i + 1], sn = inverse ? -s[i] : s[i];
        float a = fmaf(re, c[i], -(im * sn)), b = fmaf(im, c[i], re * sn);
        v[2 * i] = bf16_round(a);
        v[2 * i + 1] = bf16_round(b);
    }
}

static int run(int indexed, Dsv4CudaTensor *wqb, Dsv4CudaTensor *wa, Dsv4CudaTensor *wb,
               const float *qa, const float *cosv, const float *sinv, const float *chunk,
               const float *sinks) {
    const float eps = 1e-6f, scale = 1.0f / sqrtf((float)DIM);
    int first = START - WINDOW + 1, meta[T * 3], *sel = malloc((size_t)T * SEL * sizeof(int));
    float *q = malloc((size_t)T * HEADS * DIM * sizeof(float)), *att = malloc((size_t)T * HEADS * DIM * sizeof(float));
    float *want = malloc((size_t)T * HIDDEN * sizeof(float)), *got = malloc((size_t)T * HIDDEN * sizeof(float));
    if (!sel || !q || !att || !want || !got) return 2;
    for (int t = 0; t < T; t++) {
        int position = START + t, wfirst = position - WINDOW + 1;
        meta[3 * t] = wfirst - first;
        meta[3 * t + 1] = position - wfirst + 1;
        meta[3 * t + 2] = indexed ? 1 + (int)(urand() % SEL) : 1 + t * (COMP - 1) / (T - 1);
        if (indexed)
            for (int i = 0; i < meta[3 * t + 2]; i++) {   /* distinct ordinals, any order */
                int o, dup;
                do {
                    o = (int)(urand() % COMP);
                    dup = 0;
                    for (int j = 0; j < i; j++) dup |= sel[t * SEL + j] == o;
                } while (dup);
                sel[t * SEL + i] = o;
            }
    }
    /* the separate calls */
    Dsv4CudaActivation *qa_d = dsv4_cuda_activation_create(0, (long long)T * QR);
    Dsv4CudaActivation *q_d = dsv4_cuda_activation_create(0, (long long)T * HEADS * DIM);
    Dsv4CudaActivation *out_d = dsv4_cuda_activation_create(0, (long long)T * HIDDEN);
    if (!qa_d || !q_d || !out_d || !dsv4_cuda_activation_upload(qa_d, qa, (long long)T * QR) ||
        !dsv4_cuda_matmul_batch(wqb, qa_d, T, q_d) ||
        !dsv4_cuda_activation_download(q, q_d, (long long)T * HEADS * DIM) || !dsv4_cuda_activation_sync(q_d)) {
        fprintf(stderr, "query projection failed\n");
        return 3;
    }
    for (int t = 0; t < T; t++)
        for (int h = 0; h < HEADS; h++) {
            float *v = q + ((size_t)t * HEADS + h) * DIM, ms = 0.0f;
            for (int i = 0; i < DIM; i++) v[i] = bf16_round(v[i]);
            for (int i = 0; i < DIM; i++) ms = fmaf(v[i], v[i], ms);
            float sc = 1.0f / sqrtf(ms / (float)DIM + eps);
            for (int i = 0; i < DIM; i++) v[i] = bf16_round(v[i] * sc);
            rope(v + DIM - ROPE, cosv + (size_t)t * (ROPE / 2), sinv + (size_t)t * (ROPE / 2), 0);
        }
    int done = indexed
        ? dsv4_cuda_sparse_attn_batch_cached_idx(0, LAYER, q, chunk, START, sinks, meta, sel, SEL, first,
                                                 COMP, HEADS, DIM, T, scale, att)
        : dsv4_cuda_sparse_attn_batch_cached(0, LAYER, q, chunk, START, sinks, meta, first, COMP, HEADS,
                                             DIM, T, scale, att);
    if (!done) { fprintf(stderr, "cached sparse attention refused\n"); return 3; }
    for (int t = 0; t < T; t++)
        for (int h = 0; h < HEADS; h++)
            rope(att + ((size_t)t * HEADS + h) * DIM + DIM - ROPE, cosv + (size_t)t * (ROPE / 2),
                 sinv + (size_t)t * (ROPE / 2), 1);
    if (!dsv4_cuda_activation_upload(q_d, att, (long long)T * HEADS * DIM) ||
        !dsv4_cuda_attention_output_batch(q_d, wa, wb, GROUPS, T, out_d) ||
        !dsv4_cuda_activation_download(want, out_d, (long long)T * HIDDEN) || !dsv4_cuda_activation_sync(out_d)) {
        fprintf(stderr, "output projection failed\n");
        return 3;
    }
    /* the chain */
    if (!dsv4_cuda_attn_prefill_chain(0, LAYER, qa, QR, wqb, eps, 1, cosv, sinv, ROPE, 1, chunk, START, sinks,
                                      meta, indexed ? sel : NULL, SEL, first, COMP, HEADS, DIM, scale, T, wa,
                                      GROUPS, wb, got)) {
        fprintf(stderr, "prefill chain refused\n");
        return 4;
    }
    long long bad = 0;
    for (size_t i = 0; i < (size_t)T * HIDDEN; i++) {
        if (!isfinite(want[i])) { fprintf(stderr, "non-finite reference output %zu\n", i); return 5; }
        if (memcmp(&want[i], &got[i], 4)) {
            if (!bad) fprintf(stderr, "first difference at %zu: separate %.9g chain %.9g\n", i, want[i], got[i]);
            bad++;
        }
    }
    printf("dsv4 prefill chain (%s selection): %lld of %d outputs differ\n", indexed ? "indexed" : "plain",
           bad, T * HIDDEN);
    dsv4_cuda_activation_free(out_d); dsv4_cuda_activation_free(q_d); dsv4_cuda_activation_free(qa_d);
    free(got); free(want); free(att); free(q); free(sel);
    return bad ? 6 : 0;
}

int main(void) {
    int dev = 0;
    if (!dsv4_cuda_init(&dev, 1)) { puts("dsv4 prefill chain: no CUDA device, skipped"); return 77; }
    Dsv4CudaTensor *wqb = fp8_tensor(HEADS * DIM, QR, 0), *wa = fp8_tensor(GROUPS * ORANK, HEADS * DIM / GROUPS, 1),
                   *wb = fp8_tensor(HIDDEN, GROUPS * ORANK, 0);
    float *qa = malloc((size_t)T * QR * sizeof(float)), *cosv = malloc((size_t)T * (ROPE / 2) * sizeof(float)),
          *sinv = malloc((size_t)T * (ROPE / 2) * sizeof(float)), *chunk = malloc((size_t)T * DIM * sizeof(float)),
          *row = malloc((size_t)DIM * sizeof(float)), *comp = malloc((size_t)COMP * DIM * sizeof(float)),
          sinks[HEADS];
    if (!wqb || !wa || !wb || !qa || !cosv || !sinv || !chunk || !row || !comp) return 2;
    for (size_t i = 0; i < (size_t)T * QR; i++) qa[i] = bf16_round(frand());
    for (int t = 0; t < T; t++)
        for (int i = 0; i < ROPE / 2; i++) {
            float angle = (START + t) * powf(10000.0f, -2.0f * i / ROPE);
            cosv[t * (ROPE / 2) + i] = cosf(angle);
            sinv[t * (ROPE / 2) + i] = sinf(angle);
        }
    for (size_t i = 0; i < (size_t)T * DIM; i++) chunk[i] = bf16_round(frand());
    for (int h = 0; h < HEADS; h++) sinks[h] = frand();
    for (size_t i = 0; i < (size_t)COMP * DIM; i++) comp[i] = bf16_round(frand());
    /* the device cache as the engine leaves it before the chunk: the ring
     * holds the positions before START, the compressed rows are appended */
    for (int p = 0; p < START; p++) {
        for (int d = 0; d < DIM; d++) row[d] = bf16_round(frand());
        if (!dsv4_cuda_kv_ring_append(0, LAYER, row, p, 1, WINDOW, DIM)) { fprintf(stderr, "ring append refused\n"); return 3; }
    }
    if (!dsv4_cuda_kv_comp_append(0, LAYER, comp, 0, COMP, DIM)) { fprintf(stderr, "comp append refused\n"); return 3; }
    int r = run(0, wqb, wa, wb, qa, cosv, sinv, chunk, sinks);
    if (!r) r = run(1, wqb, wa, wb, qa, cosv, sinv, chunk, sinks);
    dsv4_cuda_tensor_free(wb); dsv4_cuda_tensor_free(wa); dsv4_cuda_tensor_free(wqb);
    dsv4_cuda_shutdown();
    free(comp); free(row); free(chunk); free(sinv); free(cosv); free(qa);
    if (!r) puts("dsv4 prefill chain: OK (bit-identical to the separate calls)");
    return r;
}
