/* The CPU's matrix unit beside the GPU in fast prompt batches (Apple only).
 *
 * The M1 has a matrix unit next to its CPU cores, reached through Accelerate's
 * BLAS: cblas_sgemm runs the 35B model's prompt products at 0.44-0.47 Tmac/s
 * from one thread, about 60% of what the GPU's matrix kernels do (0.75). With
 * both busy at once the GPU kept its speed and the CPU lost 8%. Some products
 * of a layer are read only well after the GPU could compute them, so the CPU
 * computes them while the GPU runs the steps in between: DeltaNet's z
 * projection, read by the gate after the recurrence; the gate half of the
 * attention's q projection, read after the attention; the shared expert, read
 * by the final sum after the routed experts. It also computes the routed
 * experts with the most choices (qg_cpu_experts), their int4 weights decoded,
 * and in the later batches of a long prompt the attention of the last rows
 * (qg_cpu_attn_rows).
 *
 * Fast mode only: BLAS adds the products in its own order, and each output's
 * scale is folded into its weights, so the results differ from the GPU
 * kernels' in the last bits. The weights are converted to floats on the way
 * (about 1 ms for a DeltaNet layer's z on one core, 0.3 ms for a routed
 * expert); the dense ones are kept while the same layer runs, so the windows
 * and batches of a long prompt's layer convert them once. With the scratch
 * that is 70 to 90 MB more while a prompt runs on the 35B model (about 110 MB
 * at 8192 positions, with the float copy of a KV head's cache), freed with the
 * batch tensors. QWEN36_CPU_MATMUL=0 leaves every product on the GPU. */
#ifndef QWEN36_CPU_MM_H
#define QWEN36_CPU_MM_H
#if defined(__APPLE__) && defined(__aarch64__)   /* NEON and the matrix unit: Apple Silicon only */
#define QCM_ON 1
#define ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#include <vecLib/thread_api.h>
#include <vecLib/vForce.h>
#undef I                                        /* complex.h's: a field of QW (I, O) */
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* A product of fewer rows stays on the GPU. */
#define QCM_MIN_ROWS 32

/* Float copies of int8 matrices stacked by rows, each row times its scale. */
typedef struct {
    float *w;
    size_t cap;                                 /* floats allocated */
    const void *src[2];                         /* the int8 matrices it holds: the weights do not move */
} QcmWeights;

static void qcm_weights_free(QcmWeights *b) {
    free(b->w);
    memset(b, 0, sizeof(*b));
}

/* b = the n (1 or 2) matrices q[k] (out x in int8, row scales sc[k]) stacked,
 * unless it holds them already. */
static int qcm_convert(QcmWeights *b, int n, const int8_t *const *q, const float *const *sc, int out, int in) {
    if (b->w && b->src[0] == q[0] && b->src[1] == (n > 1 ? q[1] : NULL)) return 1;
    size_t need = (size_t)n * out * in;
    if (need > b->cap) {
        float *w = realloc(b->w, need * sizeof(float));
        if (!w) return 0;
        b->w = w;
        b->cap = need;
    }
    for (int k = 0; k < n; k++)
        for (int o = 0; o < out; o++) {
            const int8_t *qr = q[k] + (size_t)o * in;
            float *wr = b->w + ((size_t)k * out + o) * in, s = sc[k][o];
            for (int i = 0; i < in; i++) wr[i] = (float)qr[i] * s;
        }
    b->src[0] = q[0];
    b->src[1] = n > 1 ? q[1] : NULL;
    return 1;
}

/* b = rows (r / group) * stride + offset + r % group, r < out, of the int8 matrix q
 * (row scales sc): one part of each head of a projection, unless b holds it. */
static int qcm_convert_rows(QcmWeights *b, const int8_t *q, const float *sc, int out, int in, int group, int stride,
                            int offset) {
    const int8_t *key = q + (size_t)offset * in;
    if (b->w && b->src[0] == key && !b->src[1]) return 1;
    size_t need = (size_t)out * in;
    if (need > b->cap) {
        float *w = realloc(b->w, need * sizeof(float));
        if (!w) return 0;
        b->w = w;
        b->cap = need;
    }
    for (int r = 0; r < out; r++) {
        size_t src = (size_t)(r / group) * stride + offset + r % group;
        const int8_t *qr = q + src * in;
        float *wr = b->w + (size_t)r * in, s = sc[src];
        for (int i = 0; i < in; i++) wr[i] = (float)qr[i] * s;
    }
    b->src[0] = key;
    b->src[1] = NULL;
    return 1;
}

/* y (rows x out, leading dimension ldy) = x (rows x in, leading dimension ldx) times w (out x in) transposed */
static void qcm_mm(float *y, int ldy, const float *x, int ldx, const float *w, int rows, int in, int out) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, out, in, 1.0f, x, ldx, w, in, 0.0f, y, ldy);
}

/* g = silu(g) * u over n values, as the GPU kernels round it: g / (1 + exp(-g)) * u
 * (the exponentials vectorized, a chunk at a time) */
static void qcm_silu_mul(float *g, const float *u, int n) {
    float t[256];
    for (int i0 = 0; i0 < n; i0 += 256) {
        int m = n - i0 < 256 ? n - i0 : 256;
        for (int i = 0; i < m; i++) t[i] = -g[i0 + i];
        vvexpf(t, t, &m);
        for (int i = 0; i < m; i++) g[i0 + i] = g[i0 + i] / (1.f + t[i]) * u[i0 + i];
    }
}

/* Planar int4 rows as an expert slot holds them: for 64 inputs 32 bytes, input
 * j < 32 in the low nibble of byte j and j >= 32 in the high one of byte j - 32,
 * value q - 8, times one scale per row and group of 64 -> rows x in floats. */
static void qcm_decode4(float *w, const uint8_t *q, const float *s, int rows, int in) {
    int groups = in / 64;
    for (size_t rb = 0; rb < (size_t)rows * groups; rb++) {
        const uint8_t *p = q + rb * 32;
        float *o = w + rb * 64;
        float32x4_t sc = vdupq_n_f32(s[rb]);
        for (int h = 0; h < 2; h++) {
            uint8x16_t v = vld1q_u8(p + 16 * h);
            int8x16_t part[2] = {vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(15))), vdupq_n_s8(8)),
                                 vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), vdupq_n_s8(8))};
            for (int k = 0; k < 2; k++) {       /* k 0: inputs 16 h .., k 1: inputs 32 + 16 h .. */
                int16x8_t a = vmovl_s8(vget_low_s8(part[k])), b = vmovl_s8(vget_high_s8(part[k]));
                float *d = o + 32 * k + 16 * h;
                vst1q_f32(d, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), sc));
                vst1q_f32(d + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a))), sc));
                vst1q_f32(d + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(b))), sc));
                vst1q_f32(d + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(b))), sc));
            }
        }
    }
}

/* Routed experts the CPU computes while the GPU computes the others: the slot
 * (gate, up, down in planar int4, then their scales) and the expert's choices. */
#define QCM_MAX_EXPERTS 64
#define QCM_EXPERT_ROWS 512                     /* choices per product: bounds the scratch */
typedef struct { const uint8_t *slot; const uint32_t *pairs; int n; } QcmExpert;

/* The CPU's part of a batch's MoE, one job on a serial queue while the main
 * thread encodes the routed experts: the shared expert, shared = (silu(post Wg)
 * * (post Wu)) Wd, each row times sigmoid(post . gate) when the model has a
 * gate; then the routed experts given to it, each choice's row of ey. */
typedef struct {
    QcmWeights gu, down;
    float *scratch;                             /* rows x 2 SI: gate | up, then h | up; and rows gate sums */
    size_t scratch_cap;
    float *ew, *ea;                             /* an expert's weights as floats; its choices' rows, g | u, y */
    size_t ew_cap, ea_cap;
    dispatch_queue_t queue;
    dispatch_group_t group;
    /* the job */
    const float *post, *gate;
    float *shared, *ey;
    const int8_t *q[3];
    const float *sc[3];
    QcmExpert ex[QCM_MAX_EXPERTS];
    size_t scales;                              /* byte offset of the scales in a slot */
    int rows, H, SI, I, K, nex, ok, ex_ok, running;
} QcmShared;

static int qcm_grow(float **p, size_t *cap, size_t need) {
    if (need <= *cap) return 1;
    float *q = realloc(*p, need * sizeof(float));
    if (!q) return 0;
    *p = q;
    *cap = need;
    return 1;
}

static int qcm_experts_run(QcmShared *j) {
    int H = j->H, I = j->I, K = j->K, R = QCM_EXPERT_ROWS;
    size_t n = (size_t)H * I;
    if (!j->nex) return 1;
    if (!qcm_grow(&j->ew, &j->ew_cap, 3 * n) || !qcm_grow(&j->ea, &j->ea_cap, (size_t)R * (2 * H + 2 * I))) return 0;
    float *a = j->ea, *gu = a + (size_t)R * H, *y = gu + (size_t)R * 2 * I;
    for (int x = 0; x < j->nex; x++) {
        const QcmExpert *e = &j->ex[x];
        const float *sc = (const float *)(e->slot + j->scales);
        qcm_decode4(j->ew, e->slot, sc, 2 * I, H);                       /* gate and up */
        qcm_decode4(j->ew + 2 * n, e->slot + n, sc + 2 * n / 64, H, I);  /* down */
        for (int c0 = 0; c0 < e->n; c0 += R) {
            int m = e->n - c0 < R ? e->n - c0 : R;
            for (int i = 0; i < m; i++)
                memcpy(a + (size_t)i * H, j->post + (size_t)((e->pairs[c0 + i] & 0x0fffffffu) / K) * H, H * sizeof(float));
            qcm_mm(gu, 2 * I, a, H, j->ew, m, H, 2 * I);
            for (int i = 0; i < m; i++) qcm_silu_mul(gu + (size_t)i * 2 * I, gu + (size_t)i * 2 * I + I, I);
            qcm_mm(y, H, gu, 2 * I, j->ew + 2 * n, m, I, H);
            for (int i = 0; i < m; i++)
                memcpy(j->ey + (size_t)(e->pairs[c0 + i] & 0x0fffffffu) * H, y + (size_t)i * H, H * sizeof(float));
        }
    }
    return 1;
}

static int qcm_shared_part(QcmShared *j) {
    int rows = j->rows, H = j->H, SI = j->SI;
    if (!qcm_grow(&j->scratch, &j->scratch_cap, (size_t)rows * 2 * SI + rows)) return 0;
    if (!qcm_convert(&j->gu, 2, j->q, j->sc, SI, H) || !qcm_convert(&j->down, 1, j->q + 2, j->sc + 2, H, SI)) return 0;
    float *gu = j->scratch, *gsum = j->scratch + (size_t)rows * 2 * SI;
    qcm_mm(gu, 2 * SI, j->post, H, j->gu.w, rows, H, 2 * SI);
    for (int r = 0; r < rows; r++) qcm_silu_mul(gu + (size_t)r * 2 * SI, gu + (size_t)r * 2 * SI + SI, SI);
    qcm_mm(j->shared, H, gu, 2 * SI, j->down.w, rows, SI, H);
    if (j->gate) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, H, 1.0f, j->post, H, j->gate, 1, 0.0f, gsum, 1);
        for (int r = 0; r < rows; r++) {
            float s = 1.f / (1.f + expf(-gsum[r]));
            float *y = j->shared + (size_t)r * H;
            for (int i = 0; i < H; i++) y[i] *= s;
        }
    }
    return 1;
}

static void qcm_shared_run(void *arg) {
    QcmShared *j = arg;
    j->ok = qcm_shared_part(j);
    j->ex_ok = qcm_experts_run(j);
}

static int qcm_shared_start(QcmShared *j) {
    if (!j->queue) {
        dispatch_queue_attr_t attr =
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
        j->queue = dispatch_queue_create("qwen36.cpu_mm", attr);
        j->group = dispatch_group_create();
        if (!j->queue || !j->group) return 0;
    }
    j->running = 1;
    dispatch_group_async_f(j->group, j->queue, j, qcm_shared_run);
    return 1;
}

/* Waits for the job; 0 if the shared expert failed (ex_ok tells the routed ones). */
static int qcm_shared_finish(QcmShared *j) {
    if (!j->running) return 1;
    dispatch_group_wait(j->group, DISPATCH_TIME_FOREVER);
    j->running = 0;
    return j->ok;
}

/* exp(x) for x <= 0, 4 at a time: 2^(x log2 e) as 2^n times the degree-6
 * Taylor polynomial of 2^f, |f| <= 1/2 (relative error about 1e-7, not expf's
 * rounding); below -87 it gives about 1e-38. */
static inline float32x4_t qcm_exp4(float32x4_t x) {
    float32x4_t t = vmulq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(1.44269504f));
    float32x4_t n = vrndnq_f32(t), f = vsubq_f32(t, n);
    float32x4_t p = vfmaq_f32(vdupq_n_f32(1.3333558e-3f), vdupq_n_f32(1.5403530e-4f), f);
    p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, f);
    p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(vcvtq_s32_f32(n), 23)));
}

/* s[0 .. n) = exp(s - max s), returns their sum (n a multiple of 4 or not) */
static float qcm_softmax_row(float *s, int n) {
    float32x4_t m4 = vdupq_n_f32(-INFINITY);
    int i = 0;
    for (; i + 4 <= n; i += 4) m4 = vmaxq_f32(m4, vld1q_f32(s + i));
    float top = vmaxvq_f32(m4);
    for (; i < n; i++) top = s[i] > top ? s[i] : top;
    float32x4_t neg = vdupq_n_f32(-top), sum4 = vdupq_n_f32(0.0f);
    for (i = 0; i + 4 <= n; i += 4) {
        float32x4_t e = qcm_exp4(vaddq_f32(vld1q_f32(s + i), neg));
        vst1q_f32(s + i, e);
        sum4 = vaddq_f32(sum4, e);
    }
    float sum = vaddvq_f32(sum4);
    for (; i < n; i++) {
        float x[4] = {s[i] - top, 0, 0, 0};
        s[i] = vgetq_lane_f32(qcm_exp4(vld1q_f32(x)), 0);
        sum += s[i];
    }
    return sum;
}

/* Prompt attention of a batch's last rows: for query head h, with KV head h /
 * (qh / kv), the scores scale q . k over the positions up to the row's own,
 * their softmax, then the sum of the values they weigh. q and out are rows x qh
 * x hd floats (the queries normed and rotated, as the GPU left them), the cache
 * kv x cap x hd halves (fast mode) or floats. Rows first .. rows of the batch,
 * at positions pos + first ..; a KV head's keys and values become floats once,
 * the scores go 256 rows at a time, each row's sum divides its output. */
#define QCM_ATTN_ROWS 256
typedef struct {
    float *k, *v, *s;
    size_t k_cap, v_cap, s_cap;
} QcmAttn;

static void qcm_attn_free(QcmAttn *a) {
    free(a->k);
    free(a->v);
    free(a->s);
    memset(a, 0, sizeof(*a));
}

static int qcm_attention(QcmAttn *a, float *out, const float *q, const void *kc, const void *vc, int half, int first,
                         int rows, int pos, int qh, int kv, int hd, int cap) {
    int n = pos + rows, group = qh / kv, CH = QCM_ATTN_ROWS;
    float inv[QCM_ATTN_ROWS];
    if (!qcm_grow(&a->s, &a->s_cap, (size_t)CH * n)) return 0;
    if (half && (!qcm_grow(&a->k, &a->k_cap, (size_t)n * hd) || !qcm_grow(&a->v, &a->v_cap, (size_t)n * hd))) return 0;
    float scale = 1.0f / sqrtf((float)hd);
    for (int kh = 0; kh < kv; kh++) {
        const float *K, *V;
        if (half) {
            const _Float16 *ks = (const _Float16 *)kc + (size_t)kh * cap * hd, *vs = (const _Float16 *)vc + (size_t)kh * cap * hd;
            for (size_t i = 0; i < (size_t)n * hd; i++) a->k[i] = (float)ks[i];
            for (size_t i = 0; i < (size_t)n * hd; i++) a->v[i] = (float)vs[i];
            K = a->k;
            V = a->v;
        } else {
            K = (const float *)kc + (size_t)kh * cap * hd;
            V = (const float *)vc + (size_t)kh * cap * hd;
        }
        for (int h = kh * group; h < (kh + 1) * group; h++)
            for (int r0 = first; r0 < rows; r0 += CH) {
                int m = rows - r0 < CH ? rows - r0 : CH, seen = pos + r0 + m;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, seen, hd, scale, q + ((size_t)r0 * qh + h) * hd,
                            qh * hd, K, hd, 0.0f, a->s, seen);
                for (int i = 0; i < m; i++) {
                    float *si = a->s + (size_t)i * seen;
                    int valid = pos + r0 + i + 1;
                    inv[i] = 1.0f / qcm_softmax_row(si, valid);
                    memset(si + valid, 0, (size_t)(seen - valid) * sizeof(float));
                }
                float *o = out + ((size_t)r0 * qh + h) * hd;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, hd, seen, 1.0f, a->s, seen, V, hd, 0.0f, o,
                            qh * hd);
                for (int i = 0; i < m; i++) {
                    float32x4_t f = vdupq_n_f32(inv[i]);
                    float *oi = o + (size_t)i * qh * hd;
                    for (int d = 0; d + 4 <= hd; d += 4) vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), f));
                }
            }
    }
    return 1;
}

/* The memory of a prompt: the converted weights and the scratch. The job must be finished. */
static void qcm_shared_release(QcmShared *j) {
    qcm_weights_free(&j->gu);
    qcm_weights_free(&j->down);
    free(j->scratch);
    free(j->ew);
    free(j->ea);
    j->scratch = j->ew = j->ea = NULL;
    j->scratch_cap = j->ew_cap = j->ea_cap = 0;
}

/* One BLAS thread: one core drives the matrix unit as fast as four (measured on
 * the M1), and leaves the others to the expert reads. */
static void qcm_init(void) {
#if defined(__MAC_15_0)
    if (__builtin_available(macOS 15.0, *)) BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
#endif
}

#else
#define QCM_ON 0                                 /* an Intel Mac: every product stays on the GPU */
#endif
#endif
