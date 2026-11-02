/* ggml_view.h -- a GGUF tensor behind a ColiTensorView (tensor.h, COLI_TENSOR_GGML).
 *
 * The view holds the tensor's blocks as the file has them (ggml_quants.h types: Q8_0,
 * Q4_K, IQ2_XS, MXFP4, ...); block_columns carries the ggml type. An engine that reads a
 * GGUF in place (st_gguf.h) hands such a view to its own matmul entries, which come here. */
#ifndef GGML_VIEW_H
#define GGML_VIEW_H

#include <stdint.h>
#include <stdlib.h>
#include "tensor.h"
#include "ggml_quants.h"

static inline int coli_ggml_view_type(const ColiTensorView *w) { return (int)w->block_columns; }

static inline int coli_ggml_view_ok(const ColiTensorView *w) {
    return w && w->format == COLI_TENSOR_GGML && w->data && w->rows >= 1 && w->columns >= 1 &&
           w->rows <= INT32_MAX && w->columns <= INT32_MAX &&
           gq_row_bytes(coli_ggml_view_type(w), w->columns) >= 0;
}

/* The type a GGUF matrix of block type `type` is held in for the CPU: its rows interleaved by
 * 16 (ggml_quants.h GQ_*_R16, the same bytes or a few percent more) when the type has that form,
 * the matrix whole groups of 16 rows and the build has their kernels; else `type`, the file's
 * rows. `env` names the engine's switch: set to 0, like GQ_EXACT=1, it keeps the file's rows. A
 * device or a row reader gets the file's rows back (gq_r16_unpack, gq_r16_dequant_row). */
static inline int coli_ggml_held_type(int type, int64_t rows, int64_t cols, const char *env) {
#ifdef GQ_R16_ON
    const char *e = env ? getenv(env) : NULL;
    const int r = gq_r16_of(type);
    if (!(e && *e == '0') && !gq_exact_mode() && r >= 0 && rows % 16 == 0 && cols % GQ_TYPES[type].blck == 0)
        return r;
#else
    (void)rows; (void)cols; (void)env;
#endif
    return type;
}

/* MXFP4 blocks as a GGUF holds them (one e8m0 byte, then 16 bytes of codes: value j in the low
 * nibble of byte j and j+16 in its high one) into the layout of the checkpoints that ship MXFP4
 * (one e8m0 byte per 32 values in `scales`, values 2i and 2i+1 in the low and high nibble of
 * byte i of `codes`): the same codes and the same scale bytes, block after block. */
static inline void coli_mxfp4_unblock(uint8_t *scales, uint8_t *codes, const uint8_t *blocks, int64_t nblocks) {
    for (int64_t b = 0; b < nblocks; b++) {
        const uint8_t *src = blocks + b * 17;
        uint8_t *dst = codes + b * 16;
        scales[b] = src[0];
        for (int i = 0; i < 16; i++) {
            const int e0 = 2 * i, e1 = 2 * i + 1;   /* source byte e % 16, nibble e / 16 */
            const unsigned lo = e0 < 16 ? src[1 + e0] & 15u : src[1 + e0 - 16] >> 4;
            const unsigned hi = e1 < 16 ? src[1 + e1] & 15u : src[1 + e1 - 16] >> 4;
            dst[i] = (uint8_t)(lo | hi << 4);
        }
    }
}

/* The activations quantized once per call into a buffer each thread keeps (gq_matmul
 * would allocate and free one per matrix: thousands a token). */
static inline gq_act *coli_ggml_act(const float *x, int S, int I) {
    static __thread gq_act a;
    return gq_act_set(&a, x, S, I) ? NULL : &a;
}

/* y[s][rows] = W x[s] for S rows of f32 activations: int8 activations per 32 and the
 * integer kernels, or the float reference under GQ_EXACT=1. 0, or -1. */
static inline int coli_ggml_matmul(float *y, const ColiTensorView *w, const float *x, int S) {
    if (!y || !x || S < 1 || !coli_ggml_view_ok(w)) return -1;
    const int t = coli_ggml_view_type(w), I = (int)w->columns, O = (int)w->rows;
    if (!gq_int_ok(t, I)) return gq_matmul_f32(y, x, (const uint8_t *)w->data, t, S, I, O);
    gq_act *a = coli_ggml_act(x, S, I);
    if (!a) return gq_matmul_f32(y, x, (const uint8_t *)w->data, t, S, I, O);
    gq_matmul_act(y, a, (const uint8_t *)w->data, t, O);
    return 0;
}

/* Two matrices of the same shape on the same activations (gate and up): one quantization
 * and one parallel region for both. 0, or -1. */
static inline int coli_ggml_matmul2(float *y1, const ColiTensorView *w1, float *y2,
                                    const ColiTensorView *w2, const float *x, int S) {
    if (!coli_ggml_view_ok(w1) || !coli_ggml_view_ok(w2) || w1->rows != w2->rows ||
        w1->columns != w2->columns || S < 1)
        return coli_ggml_matmul(y1, w1, x, S) || coli_ggml_matmul(y2, w2, x, S) ? -1 : 0;
    const int t1 = coli_ggml_view_type(w1), t2 = coli_ggml_view_type(w2), I = (int)w1->columns;
    if (!gq_int_ok(t1, I) || !gq_int_ok(t2, I))
        return coli_ggml_matmul(y1, w1, x, S) || coli_ggml_matmul(y2, w2, x, S) ? -1 : 0;
    gq_act *a = coli_ggml_act(x, S, I);
    if (!a) return coli_ggml_matmul(y1, w1, x, S) || coli_ggml_matmul(y2, w2, x, S) ? -1 : 0;
    if (S == 1) gq_matmul2_act(y1, (const uint8_t *)w1->data, t1, y2, (const uint8_t *)w2->data, t2, a, (int)w1->rows);
    else { gq_matmul_act(y1, a, (const uint8_t *)w1->data, t1, (int)w1->rows);
           gq_matmul_act(y2, a, (const uint8_t *)w2->data, t2, (int)w2->rows); }
    return 0;
}

#endif /* GGML_VIEW_H */
