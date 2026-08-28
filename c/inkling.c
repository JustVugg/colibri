/* Pure-C inference engine for Thinking Machines "Inkling" (text-only), Stage A.
 * Goal, like olmoe.c before GLM-5.2: reproduce the EXACT token ids of the HF
 * transformers reference (ref_inkling.json from tools/make_tiny_inkling.py)
 * to validate the core math before scaling to the 975B checkpoint.
 *
 * Architecture (vs glm.c's MLA/RoPE/DSA — shares almost nothing):
 *  - hybrid attention: sliding-window layers (window=512, 16 KV heads) and
 *    global layers (8 KV heads) interleaved 5:1; conventional GQA, no RoPE
 *  - learned relative-position bias: r_proj(x) mixes a per-layer bank
 *    proj[d_rel, rel_extent] into one bias per backward distance
 *  - log-length scaling tau on global layers past n_floor tokens
 *  - depthwise-causal short convs (kernel 4, residual inside, fp32):
 *    on K and V inside attention, after attention, and after the MLP
 *  - MoE: sigmoid router + loss-free bias for top-k selection; combine
 *    weights are sigmoids of the raw logits jointly normalized over
 *    topk routed + n_shared shared experts, x route_scale x global_scale
 *  - logits: hidden / logits_mup_width_multiplier, sliced to unpadded vocab
 *
 * Dense weights (attn, norms, convs, router, shared experts, dense MLP)
 * resident in RAM as f32; routed experts streamed from disk per-expert out
 * of the fused [E, 2I, D] / [E, D, I] tensors, LRU-cached, optionally
 * int-quantized (bits=0 keeps them f32 for bit-exact oracle validation).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/select.h>                              /* serve-loop stdin poll (POSIX); inkling serves on Linux */
#endif
#include "st.h"
#include "tok.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "omp_tune.h"
#include "route_trace.h"
#include "kv_prefix.h"                          /* KV prefix reuse (shared) */                          /* shared routing telemetry (#700) */
#include "serve_codec.h"
#ifdef COLI_SEGMENT_ADAPTER
#include "segment_runtime.h"
#include "segment_adapters.h"
#include "segment_adapter_internal.h"
#endif
#ifdef COLI_EDGE_ADAPTER
#include "edge_runtime.h"
#include "edge_adapters.h"
#include "edge_tok_internal.h"
#endif
#ifdef COLI_CUDA
#include "backend_cuda_ink.h"
static int g_cuda = 0;
#endif
#ifdef COLI_METAL
/* Apple-GPU expert MoE (opt-in, COLI_METAL=1). Reuses colibri's batched
 * coli_metal_moe_block: inkling's container int4 (nibble-packed, -8 offset,
 * per-row scales) is bit-identical to the Metal fmt=2 kernel, and int8 to
 * fmt=1. Expert slots live in page-aligned per-layer slabs registered once
 * for zero-copy resolve — unified memory, no upload. Attention and the dense
 * path stay on the CPU (at high hit rates ~90% of decode is expert matmul). */
#include "backend_metal.h"
static int g_metal = 0;
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#define MAXL 256

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, vocab, unpad_vocab;
    int n_heads, n_kv, head_dim;          /* global ("hybrid") layers */
    int swa_heads, swa_kv, swa_hd;        /* sliding ("hybrid_sliding") layers */
    int window, d_rel, rel_extent, conv_k;
    double log_floor;                     /* <=0: log scaling off */
    float log_alpha;
    int n_experts, topk, n_shared, moe_inter, dense_inter;
    int eos;
    float eps, route_scale, mup;
    int audio_tok, mel_bins, mel_vocab;   /* DMel audio input (TMLv0 <|audio|> placeholder) */
    unsigned char local[MAXL];            /* 1 = sliding-window layer */
    unsigned char sparse[MAXL];           /* 1 = MoE layer, 0 = dense MLP */
} Cfg;

/* per-layer dims that depend on the attention type */
#define L_HEADS(c,i) ((c)->local[i] ? (c)->swa_heads : (c)->n_heads)
#define L_KV(c,i)    ((c)->local[i] ? (c)->swa_kv    : (c)->n_kv)
#define L_HD(c,i)    ((c)->local[i] ? (c)->swa_hd    : (c)->head_dim)
#define L_EXT(c,i)   ((c)->local[i] ? (c)->window    : (c)->rel_extent)

/* ---------- resident weights ----------
 * Large matmul weights keep their on-disk dtype in RAM: bf16 for the real
 * 975B checkpoint (f32 residents would need ~172 GB, over sabre's 187),
 * f32 for the tiny oracle (bit-exact validation). Under CUDA, bf16 tensors
 * move to VRAM (dev set, host freed): decode reads ~35 GB of residents per
 * token, so this trades the DDR5 bandwidth wall for VRAM bandwidth AND
 * frees the same RAM for the expert cache. */
/* q4/qs: densa pre-quantizzata int4 group-scaled (gs=64) da un container
 * separato, opzionale — vedi load_w. La densa bf16 di Inkling e' 49.4 GB e non
 * entra in 25 GB (il caricamento la espande pure a f32, ~99 GB al picco: e' li'
 * l'OOM). A int4-gs64 diventa ~15 GB. Nessun campo q4 = comportamento invariato. */
typedef struct { float *f; uint16_t *h; void *dev;
                 uint8_t *q4;        /* int4 nibble-packed (qbits=4) o int8 (qbits=8) */
                 float *qs;          /* scale: [rows*ng] se qbits=4, [rows] se qbits=8 */
                 int gs, qbits; int64_t qn; } Wt;   /* qn = byte in q4, per il guard OOB */

typedef struct {
    float *in_ln, *post_ln;
    Wt q, k, v, r, o;                     /* projections */
    float *qn, *kn;                       /* per-head rmsnorm [head_dim] */
    float *relp;                          /* [d_rel, ext] bias bank */
    float *k_cw, *v_cw, *a_cw, *m_cw;     /* sconv weights, [C*K] depthwise */
    /* dense layers */
    Wt dg, du, dd; float dgs;
    /* MoE layers */
    float *router, *rbias, rgs;           /* [E+ns, D], [E], scalar */
    Wt sh_g, sh_u, sh_d;                  /* shared experts [ns][I,D] etc. */
} Layer;

/* ---------- routed-expert cache: LRU + optional pinned set ----------
 * Container snapshots keep the expert rows PACKED in RAM (int4 stays 4-bit:
 * ~28 MB/expert instead of ~57 unpacked, so the same budget caches twice the
 * experts); the matmul kernels unpack nibbles in-register. */
typedef struct {
    int eid; uint64_t used;
    int pinned;                           /* never evicted (usage-history pin) */
    int filled;                           /* 0 while queued for a parallel fill */
    uint8_t *p13, *p2; float *s13, *s2;   /* container: packed rows + row scales */
    int8_t *q13, *q2;                     /* bits>0: runtime-quantized int8 */
    float *f13, *f2;                      /* bits==0: raw f32 (oracle) */
} Slot;
typedef struct {
    Slot *slots;
    int *slot_by_expert;                  /* expert id -> local slot, -1 if absent */
    int n, cap;
} LCache;

typedef struct {
    Cfg c;
    shards S;
    shards Sq;                            /* container densa int4-gs64 (opzionale) */
    int has_q, q_loaded;                  /* has_q: container presente; q_loaded: tensori presi da li */
    int64_t q_bytes;
    int quant_bits;                       /* 0 = f32 experts (oracle mode) */
    int xq;                               /* experts on disk are a colibri container (U8 + .qs) */
    Wt embed, lm_head;
    float *embed_norm, *final_norm;
    Wt audio_enc;                         /* [mel_bins*mel_vocab, D] embedding table */
    float *audio_norm;                    /* audio tower RMSNorm [D]; NULL = no audio */
    Layer *L;
    LCache *cache;
    int64_t rb13, rb2;                    /* container row-bytes (0 = not container) */
    uint32_t **eusage;                    /* per-layer expert selection counts */
    int npin;                             /* pinned experts per sparse layer */
    uint64_t clock, hits, miss;
    uint64_t ereq, euse;                  /* routed richiesti (topk) vs usati dopo TOPP */
    double t_fill, t_expert, t_shared, t_attn, t_route;   /* phase timers */
    float **K, **V; int kv_len, max_t;    /* per-layer [kv][kv_ring_rows][hd]; sliding layers are a t%window ring */
    float **cs[4];                        /* conv states, [n_layers][C*(K-1)] */
    double dense_load_s;
    /* KV prefix reuse: what the current K/V and conv states were built from.
     * See kv_prefix.h — recorded where the tokens are fed, never derived. */
    kv_prefix kvp;
} Model;

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
#endif
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }
static float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
static float siluf(float x) { return x / (1.f + expf(-x)); }
/* TOPP=p (0..1): top-p adattivo sui routed — tieni gli esperti fino al peso
 * cumulato p. 0 = spento (default): tutti i topk, calcolo invariato. */
static float g_topp = 0.f;

/* y[S,O] = x[S,I] @ W^T, W row-major [O,I] */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

#if defined(__AVX512BF16__) && defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_BF16_DOT 1
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* bf16-weight matmul: activations rounded to bf16 per row (matches the HF
 * bf16 reference numerics), hardware vdpbf16ps dot where available,
 * shift-to-f32 scalar otherwise. */
static void matmul_h(float *y, const float *x, const uint16_t *W, int S, int I, int O) {
#ifdef HAVE_BF16_DOT
    if (I % 32 == 0) {
        uint16_t *xh = malloc((size_t)S * I * sizeof(uint16_t));
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            uint16_t *xd = xh + (int64_t)s * I;
            for (int i = 0; i < I; i += 32) {
                __m512 a = _mm512_loadu_ps(xs + i), b = _mm512_loadu_ps(xs + i + 16);
                _mm512_storeu_si512(xd + i, (__m512i)_mm512_cvtne2ps_pbh(b, a));
            }
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint16_t *w = W + (int64_t)o * I;
            int s = 0;
            for (; s + 3 < S; s += 4) {
                const uint16_t *x0 = xh + (int64_t)(s+0)*I;
                const uint16_t *x1 = xh + (int64_t)(s+1)*I;
                const uint16_t *x2 = xh + (int64_t)(s+2)*I;
                const uint16_t *x3 = xh + (int64_t)(s+3)*I;
                __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                for (int i = 0; i < I; i += 32) {
                    __m512bh wb = (__m512bh)_mm512_loadu_si512(w + i);
                    a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_loadu_si512(x0 + i), wb);
                    a1 = _mm512_dpbf16_ps(a1, (__m512bh)_mm512_loadu_si512(x1 + i), wb);
                    a2 = _mm512_dpbf16_ps(a2, (__m512bh)_mm512_loadu_si512(x2 + i), wb);
                    a3 = _mm512_dpbf16_ps(a3, (__m512bh)_mm512_loadu_si512(x3 + i), wb);
                }
                y[(int64_t)(s+0)*O + o] = _mm512_reduce_add_ps(a0);
                y[(int64_t)(s+1)*O + o] = _mm512_reduce_add_ps(a1);
                y[(int64_t)(s+2)*O + o] = _mm512_reduce_add_ps(a2);
                y[(int64_t)(s+3)*O + o] = _mm512_reduce_add_ps(a3);
            }
            for (; s < S; s++) {
                const uint16_t *xs = xh + (int64_t)s*I;
                __m512 acc = _mm512_setzero_ps();
                for (int i = 0; i < I; i += 32)
                    acc = _mm512_dpbf16_ps(acc, (__m512bh)_mm512_loadu_si512(xs + i),
                                                (__m512bh)_mm512_loadu_si512(w + i));
                y[(int64_t)s*O + o] = _mm512_reduce_add_ps(acc);
            }
        }
        free(xh);
        return;
    }
#endif
    /* Prefill tile: decode each bf16 weight once for four independent rows.
     * Explicit mul+add (rather than a C expression that the compiler may fuse)
     * matches the non-FMA scalar oracle bit for bit.  Every SIMD lane still
     * accumulates i=0..I-1 in the original order. */
#if defined(__AVX2__)
    if (S > 1) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint16_t *w = W + (int64_t)o * I;
            int s = 0;
            for (; s + 3 < S; s += 4) {
                const float *x0 = x + (int64_t)(s+0)*I;
                const float *x1 = x + (int64_t)(s+1)*I;
                const float *x2 = x + (int64_t)(s+2)*I;
                const float *x3 = x + (int64_t)(s+3)*I;
                __m128 acc = _mm_setzero_ps();
                for (int i = 0; i < I; i++) {
                    union { uint32_t u; float f; } v = { (uint32_t)w[i] << 16 };
                    __m128 xv = _mm_set_ps(x3[i], x2[i], x1[i], x0[i]);
                    acc = _mm_add_ps(acc, _mm_mul_ps(xv, _mm_set1_ps(v.f)));
                }
                float a[4]; _mm_storeu_ps(a, acc);
                y[(int64_t)(s+0)*O + o] = a[0]; y[(int64_t)(s+1)*O + o] = a[1];
                y[(int64_t)(s+2)*O + o] = a[2]; y[(int64_t)(s+3)*O + o] = a[3];
            }
            for (; s < S; s++) {
                const float *xs = x + (int64_t)s * I;
                float acc = 0.f;
                for (int i = 0; i < I; i++) {
                    union { uint32_t u; float f; } v = { (uint32_t)w[i] << 16 };
                    acc += xs[i] * v.f;
                }
                y[(int64_t)s * O + o] = acc;
            }
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) {
                union { uint32_t u; float f; } v = { (uint32_t)w[i] << 16 };
                acc += xs[i] * v.f;
            }
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* dispatch on where the weight lives */
/* y[S,O] = x[S,I] @ W^T con W int4 GROUP-scaled: nibble +8, low = colonna pari,
 * una scala f32 ogni `gs` elementi lungo I (ng = ceil(I/gs) scale per riga).
 * Differenza da matmul_q4 (per-riga): la scala cambia DENTRO la riga, quindi
 * l'accumulo va chiuso a ogni gruppo invece che una volta sola a fine riga. */
#if defined(__AVX2__) && defined(__FMA__)
static inline float matmul_i4g_row_fma(const float *xs, const uint8_t *w,
                                       const float *sc, int I, int gs, int ng) {
    __m128 acc = _mm_setzero_ps();
    for (int g = 0; g < ng; g++) {
        int i0 = g*gs, i1 = i0+gs; if (i1 > I) i1 = I;
        __m128 part = _mm_setzero_ps();
        for (int i = i0; i < i1; i++) {
            uint8_t b = w[i >> 1]; int q = (i & 1) ? (b >> 4) : (b & 15);
            part = _mm_fmadd_ss(_mm_set_ss(xs[i]), _mm_set_ss((float)(q-8)), part);
        }
        acc = _mm_add_ss(acc, _mm_mul_ss(part, _mm_set_ss(sc[g])));
    }
    return _mm_cvtss_f32(acc);
}
#endif
static void matmul_i4g(float *y, const float *x, const uint8_t *p, const float *scale,
                       int S, int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
    int64_t rb = (I + 1) / 2;
#if defined(__AVX2__) && defined(__FMA__)
    /* Four prompt rows share nibble unpack and scale loads.  The explicit FMA
     * for each lane and the group-boundary mul+add match the scalar kernel's
     * generated operation order bit for bit. */
    if (S > 1) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *w = p + (int64_t)o * rb;
            const float *sc = scale + (int64_t)o * ng;
            int s = 0;
            for (; s + 3 < S; s += 4) {
                const float *x0 = x + (int64_t)(s+0)*I, *x1 = x + (int64_t)(s+1)*I;
                const float *x2 = x + (int64_t)(s+2)*I, *x3 = x + (int64_t)(s+3)*I;
                __m128 acc = _mm_setzero_ps();
                for (int g = 0; g < ng; g++) {
                    int i0 = g*gs, i1 = i0+gs; if (i1 > I) i1 = I;
                    __m128 part = _mm_setzero_ps();
                    for (int i = i0; i < i1; i++) {
                        uint8_t b = w[i >> 1]; int q = (i & 1) ? (b >> 4) : (b & 15);
                        __m128 xv = _mm_set_ps(x3[i], x2[i], x1[i], x0[i]);
                        part = _mm_fmadd_ps(xv, _mm_set1_ps((float)(q-8)), part);
                    }
                    acc = _mm_add_ps(acc, _mm_mul_ps(part, _mm_set1_ps(sc[g])));
                }
                float a[4]; _mm_storeu_ps(a, acc);
                y[(int64_t)(s+0)*O + o] = a[0]; y[(int64_t)(s+1)*O + o] = a[1];
                y[(int64_t)(s+2)*O + o] = a[2]; y[(int64_t)(s+3)*O + o] = a[3];
            }
            for (; s < S; s++) {
                const float *xs = x + (int64_t)s*I;
                y[(int64_t)s*O + o] = matmul_i4g_row_fma(xs, w, sc, I, gs, ng);
            }
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = p + (int64_t)o * rb;
        const float *sc = scale + (int64_t)o * ng;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
#if defined(__AVX2__) && defined(__FMA__)
            y[(int64_t)s * O + o] = matmul_i4g_row_fma(xs, w, sc, I, gs, ng);
#else
            float acc = 0.f;
            for (int g = 0; g < ng; g++) {
                int i0 = g * gs, i1 = i0 + gs; if (i1 > I) i1 = I;
                float part = 0.f;
                for (int i = i0; i < i1; i++) {
                    uint8_t b = w[i >> 1];
                    int q = (i & 1) ? (b >> 4) : (b & 0x0F);
                    part += xs[i] * (float)(q - 8);
                }
                acc += part * sc[g];               /* scala chiusa per gruppo */
            }
            y[(int64_t)s * O + o] = acc;
#endif
        }
    }
}
/* int8 per-riga (embed/lm_head: sensibili, non vanno a 4 bit) */
static void matmul_i8r(float *y, const float *x, const int8_t *q, const float *scale,
                       int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I; float sc = scale[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * (float)w[i];
            y[(int64_t)s * O + o] = acc * sc;
        }
    }
}

#ifdef COLI_INKLING_SHARED_BATCH_TEST
static uint64_t g_matmul_w_calls;
#endif
static void matmul_w(float *y, const float *x, Wt W, int S, int I, int O) {
#ifdef COLI_INKLING_SHARED_BATCH_TEST
    g_matmul_w_calls++;
#endif
#ifdef COLI_CUDA
    if (W.dev) {
        if (ink_cuda_matmul_bf16(y, x, W.dev, S, I, O) == 0) return;
        fprintf(stderr, "cuda matmul failed and host copy was freed\n"); exit(1);
    }
#endif
    if (W.q4) {
        /* guard: il container e' un file, non un invariante — se la geometria non
         * torna si esce invece di leggere fuori dal buffer. */
        int64_t need = W.qbits == 8 ? (int64_t)O * I : (int64_t)O * ((I + 1) / 2);
        if (need > W.qn) {
            fprintf(stderr, "dense q4: geometria incoerente (serve %lld B, ho %lld) I=%d O=%d\n",
                    (long long)need, (long long)W.qn, I, O); exit(1);
        }
        if (W.qbits == 8) matmul_i8r(y, x, (const int8_t*)W.q4, W.qs, S, I, O);
        else              matmul_i4g(y, x, W.q4, W.qs, S, I, O, W.gs);
        return;
    }
    if (W.f) matmul(y, x, W.f, S, I, O);
    else     matmul_h(y, x, W.h, S, I, O);
}

/* y[1,O] = x @ q^T, int8 weights + per-row scale. Fast path: activations
 * quantized Q8 per 32-block, VNNI (or maddubs) int8 dot — same family as
 * glm.c's IDOT kernels; IDOT=0 falls back to the byte-exact scalar route. */
#if defined(__AVX2__)
static inline __m256i i8dot_block(__m256i acc, __m256i a, __m256i b) {
    __m256i ax = _mm256_sign_epi8(a, a);        /* |a| as u8 */
    __m256i sy = _mm256_sign_epi8(b, a);        /* b * sign(a) */
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, ax, sy);
#else
    __m256i p = _mm256_maddubs_epi16(ax, sy);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
#endif
}
#endif
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__AVX2__)
    static int idot = -1;
    /* Opt-in (IDOT=1), not default: the fast path only exists under __AVX2__
     * and quantizes ACTIVATIONS per 32-block — the same model produced
     * different tokens on x86 vs ARM with the old on-by-default. Same class
     * and same fix as olmoe (#1044) and qwen36 (#712 review). */
    if (idot < 0) { const char *e = getenv("IDOT"); idot = (e && atoi(e)); }
    if (idot && I % 32 == 0 && I <= 8192) {
        int nb = I / 32;
        int8_t xi[8192]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*32;
            float am = 0.f; for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 32; i++) xi[b*32+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                __m256i vacc = i8dot_block(_mm256_setzero_si256(),
                                           _mm256_loadu_si256((const __m256i*)(xi + b*32)),
                                           _mm256_loadu_si256((const __m256i*)(w + b*32)));
                __m128i lo = _mm256_castsi256_si128(vacc), hi = _mm256_extracti128_si256(vacc, 1);
                __m128i s4 = _mm_add_epi32(lo, hi);
                s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
                acc += xs[b] * (float)_mm_cvtsi128_si32(s4);
            }
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

/* y[1,O] = x @ W^T with W kept PACKED int4 (low nibble = even column, +8
 * offset, per-row scale — the on-disk container layout, cached as-is).
 * Nibbles unpack in-register: same numeric result as unpack-to-int8 +
 * matmul_q, half the cache footprint. IDOT=0 keeps the byte-exact scalar. */
static void matmul_q4(float *y, const float *x, const uint8_t *p, const float *scale, int I, int O) {
#if defined(__AVX2__)
    static int idot = -1;
    /* Opt-in (IDOT=1), not default: the fast path only exists under __AVX2__
     * and quantizes ACTIVATIONS per 32-block — the same model produced
     * different tokens on x86 vs ARM with the old on-by-default. Same class
     * and same fix as olmoe (#1044) and qwen36 (#712 review). */
    if (idot < 0) { const char *e = getenv("IDOT"); idot = (e && atoi(e)); }
    if (idot && I % 32 == 0 && I <= 8192) {
        int nb = I / 32;
        int8_t xi[8192]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*32;
            float am = 0.f; for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 32; i++) xi[b*32+i] = (int8_t)lrintf(xb[i]*inv);
        }
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m256i b8 = _mm256_set1_epi8(8);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *w = p + (int64_t)o * (I/2);
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                __m128i by = _mm_loadu_si128((const __m128i*)(w + b*16));  /* 16 B = 32 nibbles */
                __m128i lo = _mm_and_si128(by, m4);                        /* even columns */
                __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);     /* odd columns  */
                __m256i nib = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi),  /* cols 16..31 */
                                               _mm_unpacklo_epi8(lo, hi)); /* cols  0..15 */
                nib = _mm256_sub_epi8(nib, b8);
                __m256i vacc = i8dot_block(_mm256_setzero_si256(),
                                           _mm256_loadu_si256((const __m256i*)(xi + b*32)), nib);
                __m128i l = _mm256_castsi256_si128(vacc), h = _mm256_extracti128_si256(vacc, 1);
                __m128i s4 = _mm_add_epi32(l, h);
                s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
                acc += xs[b] * (float)_mm_cvtsi128_si32(s4);
            }
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = p + (int64_t)o * (I/2);
        float acc = 0.f;
        for (int i = 0; i < I; i += 2) {
            uint8_t byte = w[i/2];
            acc += x[i]   * (float)((int)(byte & 0xF) - 8);
            acc += x[i+1] * (float)((int)(byte >> 4)  - 8);
        }
        y[o] = acc * scale[o];
    }
}

static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax / qmax; if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v >  qmax) v =  qmax;
            if (v < -qmax-1) v = -qmax-1;
            qr[i] = (int8_t)v;
        }
    }
}

/* rmsnorm computed in f64 accumulate like the f32->f32 reference */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- depthwise causal short conv, residual inside (fp32) ----------
 * seq[S,C] in-place: out[t] = sum_j w[c,j]*in[t+j-(K-1)] + in[t], history from
 * state[C*(K-1)] (raw pre-conv inputs), which is updated to the new tail. */
static void sconv_apply(float *seq, int S, int C, const float *w, float *state, int K) {
    int P = K - 1;
    #pragma omp parallel
    {
        float *col = malloc((P + S) * sizeof(float));
        #pragma omp for schedule(static)
        for (int ch = 0; ch < C; ch++) {
            for (int j = 0; j < P; j++) col[j] = state[(int64_t)ch*P + j];
            for (int t = 0; t < S; t++) col[P + t] = seq[(int64_t)t*C + ch];
            const float *wc = w + (int64_t)ch*K;
            for (int t = 0; t < S; t++) {
                float acc = 0.f;
                for (int j = 0; j < K; j++) acc += wc[j] * col[t + j];
                seq[(int64_t)t*C + ch] = acc + col[P + t];
            }
            for (int j = 0; j < P; j++) state[(int64_t)ch*P + j] = col[S + j];
        }
        free(col);
    }
}

/* ---------- config loading ----------
 * Accepts both the flat text config (tiny oracle via InklingForCausalLM) and
 * the full multimodal config.json (real checkpoint, fields under text_config). */
static double jnum(jval *o, const char *k, double dflt) {
    jval *v = json_get(o, k);
    return (v && v->t == J_NUM) ? v->num : dflt;
}

static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *root = json_parse(buf, &arena);
    jval *r = json_get(root, "text_config"); if (!r) r = root;

    c->hidden      = (int)jnum(r,"hidden_size",6144);
    c->n_layers    = (int)jnum(r,"num_hidden_layers",66);
    c->vocab       = (int)jnum(r,"vocab_size",201024);
    c->unpad_vocab = (int)jnum(r,"unpadded_vocab_size",c->vocab);
    c->n_heads     = (int)jnum(r,"num_attention_heads",64);
    c->n_kv        = (int)jnum(r,"num_key_value_heads",8);
    c->head_dim    = (int)jnum(r,"head_dim",128);
    c->swa_heads   = (int)jnum(r,"swa_num_attention_heads",c->n_heads);
    c->swa_kv      = (int)jnum(r,"swa_num_key_value_heads",16);
    c->swa_hd      = (int)jnum(r,"swa_head_dim",c->head_dim);
    c->window      = (int)jnum(r,"sliding_window_size",512);
    c->d_rel       = (int)jnum(r,"d_rel",16);
    c->rel_extent  = (int)jnum(r,"rel_extent",1024);
    c->log_floor   = jnum(r,"log_scaling_n_floor",0);
    c->log_alpha   = (float)jnum(r,"log_scaling_alpha",0.1);
    c->conv_k      = (int)jnum(r,"sconv_kernel_size", jnum(r,"conv_kernel_size",4));
    c->n_experts   = (int)jnum(r,"n_routed_experts",256);
    c->topk        = (int)jnum(r,"num_experts_per_tok",6);
    c->n_shared    = (int)jnum(r,"n_shared_experts",2);
    c->eps         = (float)jnum(r,"rms_norm_eps",1e-6);
    c->route_scale = (float)jnum(r,"route_scale",8.0);
    c->mup         = (float)jnum(r,"logits_mup_width_multiplier",24.0);
    /* eos lives at the top level in the real multimodal config, in the text
     * config for a flat snapshot; may be null (tiny oracle) */
    jval *eo = json_get(root,"eos_token_id");
    if (!eo || eo->t != J_NUM) eo = json_get(r,"eos_token_id");
    c->eos = (eo && eo->t == J_NUM) ? (int)eo->num : -1;
    /* DMel audio: placeholder id at the top level (absent in the shipped
     * config -> the TMLv0 constant), frame geometry under audio_config */
    c->audio_tok = (int)jnum(root,"audio_token_id",200023);
    jval *ac = json_get(root,"audio_config");
    c->mel_bins  = ac ? (int)jnum(ac,"n_mel_bins",80)     : 80;
    c->mel_vocab = ac ? (int)jnum(ac,"mel_vocab_size",16) : 16;
    /* real config.json: intermediate_size = MoE, dense_intermediate_size = dense.
     * HF-saved config (post_init applied): intermediate_size = dense, moe_intermediate_size = MoE. */
    jval *dis = json_get(r,"dense_intermediate_size");
    if (dis && dis->t == J_NUM) {
        c->dense_inter = (int)dis->num;
        c->moe_inter   = (int)jnum(r,"intermediate_size",3072);
    } else {
        c->dense_inter = (int)jnum(r,"intermediate_size",24576);
        c->moe_inter   = (int)jnum(r,"moe_intermediate_size",3072);
    }
    if (c->n_layers > MAXL) { fprintf(stderr,"n_layers %d > MAXL\n", c->n_layers); exit(1); }

    /* SEC: these loops run to n_layers -- a number from config.json, up to
     * MAXL -- while indexing a JSON array whose length is a different, equally
     * attacker-chosen number. "num_hidden_layers": 66 with a one-element
     * layer_types[] read kids[1..65] past a malloc'd array that starts at
     * capacity 8, then dereferenced whatever was there as a string. config.json
     * alone was enough; no weights and no valid snapshot were needed.
     *
     * A short array now means "not specified for these layers", so the loop
     * falls through to the same default it would have used had the key been
     * absent. The element type is checked too: kids[i]->str is a union member
     * that is only a valid pointer when t == J_STR. */
    #define LT_STR(arr, i) \
        ((arr) && (arr)->t == J_ARR && (i) < (arr)->len && \
         (arr)->kids[i] && (arr)->kids[i]->t == J_STR ? (arr)->kids[i]->str : NULL)

    /* attention layer types: explicit layer_types[] > local_layer_ids[] > (i+1)%6 rule */
    jval *lt = json_get(r,"layer_types");
    jval *ll = json_get(r,"local_layer_ids");
    for (int i = 0; i < c->n_layers; i++) {
        const char *ltype = LT_STR(lt, i);
        if (ltype) c->local[i] = (strcmp(ltype,"hybrid_sliding")==0);
        else if (ll && ll->t == J_ARR) {
            c->local[i] = 0;
            for (int j = 0; j < ll->len; j++)
                if (ll->kids[j] && ll->kids[j]->t == J_NUM &&
                    (int)ll->kids[j]->num == i) { c->local[i] = 1; break; }
        } else c->local[i] = ((i + 1) % 6) != 0;
    }
    /* MLP types: explicit mlp_layer_types[] > dense_mlp_idx (first k layers dense) */
    jval *mt = json_get(r,"mlp_layer_types");
    int dense_idx = (int)jnum(r,"dense_mlp_idx",0);
    for (int i = 0; i < c->n_layers; i++) {
        const char *mtype = LT_STR(mt, i);
        if (mtype) c->sparse[i] = (strcmp(mtype,"sparse")==0);
        else c->sparse[i] = (i >= dense_idx);
    }
    #undef LT_STR
    free(buf); free(arena);
}

/* ---------- weight loading ---------- */
static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}
static float load_scalar(Model *m, const char *name, float dflt) {
    if (!st_has(&m->S, name)) return dflt;
    /* SEC: `v` is four bytes on the stack, and st_read_f32 writes as many as
     * the FILE says the tensor holds -- it validates the header against itself
     * (numel*esz == nbytes) and cannot know the destination size. A crafted
     * snapshot declaring e.g. model.layers.0.mlp.gate.global_scale as F32
     * shape [4096] therefore drops 16 KiB of attacker bytes over this frame
     * and its return address; the reported crash had RSP fully controlled.
     *
     * st_read_f32_cap is the same read with the caller's capacity passed in.
     * A scalar's capacity is 1. Anything larger is a hostile or corrupt file
     * and stops the load, which is the behaviour every other bounds check in
     * st.h already has. */
    float v; st_read_f32_cap(&m->S, name, &v, 1, 0); return v;
}

/* chunked pread: a single pread caps at ~2.1 GB on Linux, and the bf16
 * embed/lm_head tensors are 2.47 GB — loop in 1 GB slices */
static void pread_all(int fd, void *buf, int64_t nb, int64_t off) {
    char *p = buf;
    while (nb > 0) {
        int64_t chunk = nb < (1<<30) ? nb : (1<<30);
        ssize_t got = pread(fd, p, (size_t)chunk, off);
        if (got <= 0) { perror("pread chunk"); exit(1); }
        p += got; off += got; nb -= got;
    }
}

/* big matmul weights keep their on-disk dtype resident: BF16 raw (real
 * checkpoint, halves RAM), anything else as f32 (tiny oracle: bit-exact).
 * gpu_ok: bf16 tensors move to VRAM while budget lasts (embed stays host —
 * it's a row lookup, not a matmul). */
/* Container densa pre-quantizzata (opzionale): <snap>/dense-int4g64/.
 * Se il tensore c'e' li' dentro lo carichiamo int4-gs64 (U8 + sidecar .qs, la
 * stessa convenzione che il container usa gia' per gli esperti); altrimenti si
 * prosegue col percorso bf16/f32 di sempre. Container assente = nessun cambio. */
static int load_w_quant(Model *m, const char *name, int64_t orig_numel, Wt *out) {
    if (!m->has_q) return 0;
    st_tensor *t = st_find(&m->Sq, name);
    if (!t || t->dtype != 3) return 0;              /* 3 = U8/byte grezzi */
    char qn[352]; snprintf(qn, sizeof(qn), "%s.qs", name);
    st_tensor *s = st_find(&m->Sq, qn);
    if (!s) return 0;                                /* senza scale non si decodifica */
    /* Geometria dedotta dai CONTEGGI, non dalle shape (st_tensor non le porta):
     *   int4-gs64 -> byte ~= numel/2 e scale ~= numel/64
     *   int8      -> byte  == numel   e scale == righe (numel/I, I ignoto qui:
     *                basta che le scale siano molte meno dei byte)
     * Il controllo forte sull'OOB lo fa matmul_w con l'I vero del chiamante. */
    Wt w = {0};
    if (t->nbytes == orig_numel && s->numel * 64 < (int64_t)t->nbytes) {
        w.qbits = 8;                                 /* int8 per riga */
    } else if (t->nbytes * 2 >= orig_numel && t->nbytes * 2 <= orig_numel + 2 * s->numel) {
        w.qbits = 4; w.gs = 64;                      /* int4 group-scaled */
    } else {
        fprintf(stderr, "[dense] %s: geometria non riconosciuta nel container, uso il bf16\n", name);
        return 0;                                    /* dubbio -> percorso originale */
    }
    w.q4 = malloc(t->nbytes); if (!w.q4) { fprintf(stderr, "OOM %s\n", name); exit(1); }
    pread_all(t->fd, w.q4, t->nbytes, t->off);
    w.qn = t->nbytes;
    w.qs = falloc(s->numel);
    st_read_f32(&m->Sq, qn, w.qs, 0);
    *out = w;
    m->q_loaded++; m->q_bytes += t->nbytes + (int64_t)s->numel * 4;
    return 1;
}

static Wt load_w(Model *m, const char *name, int gpu_ok) {
    Wt w = {0};
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "missing %s\n", name); exit(1); }
    if (load_w_quant(m, name, t->numel, &w)) return w;
    if (t->dtype == 0) {
        w.h = malloc(t->nbytes); if (!w.h) { fprintf(stderr,"OOM %s\n",name); exit(1); }
        pread_all(t->fd, w.h, t->nbytes, t->off);
#ifdef COLI_CUDA
        /* keep 3 GB VRAM headroom for the activation buffers + future tiers */
        if (g_cuda && gpu_ok && ink_cuda_free_bytes() > (size_t)t->nbytes + (3ULL<<30)) {
            w.dev = ink_cuda_upload(w.h, t->nbytes);
            if (w.dev) { free(w.h); w.h = NULL; }
        }
#else
        (void)gpu_ok;
#endif
    } else {
        w.f = falloc(t->numel);
        st_read_f32(&m->S, name, w.f, 0);
    }
    return w;
}
/* `off` conta ELEMENTI ed e' sempre un multiplo di I (si affetta per riga: un
 * esperto condiviso dal tensore fuso [E,R,I]), quindi I arriva dal chiamante —
 * st_tensor non porta le shape e non voglio indovinarle. */
static Wt wt_off_i(Wt w, int64_t off, int I) {
    Wt r = w;
    r.f = w.f ? w.f + off : NULL;
    r.h = w.h ? w.h + off : NULL;
    r.dev = w.dev ? (char*)w.dev + off*2 : NULL;    /* dev is always bf16 */
    if (w.q4) {
        int64_t row = off / I;
        if (w.qbits == 8) { r.q4 = w.q4 + off;        r.qs = w.qs + row; r.qn = w.qn - off; }
        else              { r.q4 = w.q4 + off / 2;    r.qs = w.qs + row * ((I + w.gs - 1) / w.gs);
                            r.qn = w.qn - off / 2; }
    }
    return r;
}
/* dequantizza UNA riga (int4-gs64: nibble +8, low = colonna pari, scala ogni gs) */
static void wt_deq_row(Wt w, int64_t row, float *out, int I) {
    if (w.qbits == 8) {
        const int8_t *q = (const int8_t*)w.q4 + row * I; float s = w.qs[row];
        for (int i = 0; i < I; i++) out[i] = (float)q[i] * s;
        return;
    }
    int ng = (I + w.gs - 1) / w.gs;
    const uint8_t *p = w.q4 + row * ((I + 1) / 2);
    const float *sc = w.qs + row * ng;
    for (int i = 0; i < I; i++) {
        uint8_t b = p[i >> 1];
        int q = (i & 1) ? (b >> 4) : (b & 0x0F);
        out[i] = (float)(q - 8) * sc[i / w.gs];
    }
}
static void wt_row_f32(Wt w, int64_t off, float *out, int n) {
    if (w.q4) { wt_deq_row(w, off / n, out, n); return; }   /* n = dim di contrazione */
    if (w.f) memcpy(out, w.f + off, n * sizeof(float));
    else for (int i = 0; i < n; i++) { union { uint32_t u; float f; } v = { (uint32_t)w.h[off + i] << 16 }; out[i] = v.f; }
}

/* f32 slice of a (possibly bf16/f16) tensor: element offset + count.
 * Needed to stream one expert out of the fused [E,2I,D]/[E,D,I] tensors. */
static void read_f32_slice(shards *S, const char *name, float *out, int64_t off, int64_t cnt) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->dtype == 3) { fprintf(stderr, "%s: U8 container has no f32 view\n", name); exit(1); }
    int esz = (t->dtype == 2) ? 4 : 2;
    void *raw = malloc((size_t)cnt * esz);
    if (!raw) { fprintf(stderr,"OOM slice %s\n",name); exit(1); }
    if (pread(t->fd, raw, (size_t)cnt*esz, t->off + off*esz) != (ssize_t)(cnt*esz)) { perror("pread slice"); exit(1); }
    if (t->dtype == 2) memcpy(out, raw, (size_t)cnt*4);
    else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < cnt; i++) out[i] = bf16_to_f32(p[i]); }
    else                    { uint16_t *p = raw; for (int64_t i = 0; i < cnt; i++) out[i] = f16_to_f32(p[i]); }
    free(raw);
    posix_fadvise(t->fd, t->off + off*esz, cnt*esz, POSIX_FADV_DONTNEED);
}

/* raw byte slice of a U8 container tensor */
static void read_u8_slice(shards *S, const char *name, uint8_t *out, int64_t boff, int64_t nb) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (pread(t->fd, out, (size_t)nb, t->off + boff) != (ssize_t)nb) { perror("pread u8 slice"); exit(1); }
    posix_fadvise(t->fd, t->off + boff, nb, POSIX_FADV_DONTNEED);
}

/* container rows -> int8: rowb==cols is int8 verbatim; rowb==cols/2 is packed
 * int4 (low nibble = even column, offset +8 — convert_inkling_int4.py / glm.c) */
static void unpack_rows(const uint8_t *raw, int8_t *q, int64_t rows, int64_t cols, int64_t rowb) {
    if (rowb == cols) { memcpy(q, raw, (size_t)(rows*cols)); return; }
    if (rowb*2 != cols) { fprintf(stderr, "container row size %ld vs cols %ld unsupported\n", (long)rowb, (long)cols); exit(1); }
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *b = raw + r*rowb;
        int8_t *qr = q + r*cols;
        for (int64_t j = 0; j < rowb; j++) {
            qr[2*j]   = (int8_t)((b[j] & 0xF) - 8);
            qr[2*j+1] = (int8_t)((b[j] >> 4) - 8);
        }
    }
}

static double mem_avail_bytes(void);

static void model_init_range(Model *m, const char *snap, int cap, int bits,
                             int layer_begin, int layer_end,
                             int load_boundaries, int allocate_state,
                             int init_telemetry) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    /* densa pre-quantizzata, se il container c'e' (INK_DENSE_Q4=0 la ignora).
     * Sta in una SOTTOCARTELLA apposta: nella dir dello snapshot i nomi tensore
     * andrebbero in collisione con i bf16 originali, che restano intatti. */
    {   char qd[2100]; snprintf(qd, sizeof(qd), "%s/dense-int4g64", snap);
        const char *off = getenv("INK_DENSE_Q4");
        struct stat qs;
        if (!(off && *off == '0') && stat(qd, &qs) == 0 && S_ISDIR(qs.st_mode)) {
            st_init(&m->Sq, qd);
            if (m->Sq.n > 0) { m->has_q = 1;
                fprintf(stderr, "[dense] container int4-gs64: %s (%d tensori)\n", qd, m->Sq.n); }
        }
    }
    Cfg *c = &m->c;
    if (layer_end == 0) layer_end = c->n_layers;
    if (layer_begin < 0 || layer_begin >= layer_end ||
        layer_end > c->n_layers) {
        fprintf(stderr, "invalid Inkling layer range [%d,%d) for %d layers\n",
                layer_begin, layer_end, c->n_layers);
        exit(1);
    }
    int D = c->hidden, K = c->conv_k;
    double t0 = now_s();
#ifdef COLI_CUDA
    if (!getenv("NOGPU")) {
        int dev = getenv("GPU_DEV") ? atoi(getenv("GPU_DEV")) : 0;
        if (ink_cuda_init(dev) == 0) {
            g_cuda = 1;
            fprintf(stderr, "[cuda] device %d ready, %.1f GB free — bf16 residents to VRAM\n",
                    dev, ink_cuda_free_bytes()/1e9);
        } else fprintf(stderr, "[cuda] init failed, running on CPU\n");
    }
#endif
#ifdef COLI_METAL
    {   const char *me = getenv("COLI_METAL");
        if (me && *me == '1' && !getenv("NOGPU")) {
            /* Residency set ON by default for inkling: without it every MoE
             * block pays per-buffer useResource churn — measured 0.17 vs 1.76
             * tok/s decode on Inkling-Small. COLI_METAL_RESSET=0 opts out. */
            setenv("COLI_METAL_RESSET", "1", 0);
            if (coli_metal_init()) {
                g_metal = 1;
                fprintf(stderr, "[metal] ready — batched expert MoE on the Apple GPU\n");
            } else fprintf(stderr, "[metal] init failed, running on CPU\n");
        }
    }
#endif
    if (load_boundaries) {
        m->embed      = load_w(m, "model.embed_tokens.weight", 0);
        m->embed_norm = st_has(&m->S,"model.embed_norm.weight") ? load_t(m,"model.embed_norm.weight") : NULL;
        m->final_norm = load_t(m, "model.norm.weight");
        m->lm_head    = load_w(m, "lm_head.weight", 1);
    }
    /* Inkling's audio "tower" is one embedding table + one RMSNorm. The int4
     * containers are text-only, so these usually arrive via an audio.safetensors
     * sidecar dropped in the snapshot dir (st_init indexes every *.safetensors).
     * Absent tensors = text-only engine, exactly as before. */
    if (load_boundaries && st_has(&m->S, "model.audio.encoder.weight")) {
        /* SEC (GHSA-w696): mel_bins/mel_vocab come straight from config.json with
         * no bounds. audio_embed_row indexes the table at (b*mel_vocab+v)*D with
         * b<mel_bins, v<mel_vocab — so the table must have exactly
         * mel_bins*mel_vocab rows or that index runs off the heap (bidirectional
         * OOB read, config-controlled). Reconcile the real element count against
         * the geometry before using it. */
        st_tensor *aet = st_find(&m->S, "model.audio.encoder.weight");
        if (c->mel_bins < 1 || c->mel_vocab < 1 || D < 1 ||
            (int64_t)c->mel_bins * c->mel_vocab > INT64_MAX / D ||
            !aet || aet->numel != (int64_t)c->mel_bins * c->mel_vocab * D) {
            fprintf(stderr, "[audio] rejected: encoder table has %lld elements, "
                    "config geometry is %d bins x %d levels x D=%d\n",
                    aet ? (long long)aet->numel : -1, c->mel_bins, c->mel_vocab, D);
            exit(1);
        }
        m->audio_enc  = load_w(m, "model.audio.encoder.weight", 0);
        m->audio_norm = load_t(m, "model.audio.final_norm.weight");
        fprintf(stderr, "[audio] DMel encoder loaded (%d bins x %d levels -> D=%d)\n",
                c->mel_bins, c->mel_vocab, D);
    }
    m->L = calloc(c->n_layers, sizeof(Layer));
    for (int j = 0; j < 4; j++)
        m->cs[j] = calloc(c->n_layers, sizeof(float*));
    char nm[320];
    for (int i = layer_begin; i < layer_end; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix)  snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        #define LDW(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_w(m,nm,1)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LDW(q, "self_attn.q_proj.weight"); LDW(k, "self_attn.k_proj.weight");
        LDW(v, "self_attn.v_proj.weight"); LDW(r, "self_attn.r_proj.weight");
        LDW(o, "self_attn.o_proj.weight");
        LD(qn,"self_attn.q_norm.weight"); LD(kn,"self_attn.k_norm.weight");
        LD(relp, "self_attn.rel_logits_proj.proj");
        LD(k_cw, "self_attn.k_sconv.conv1d.weight");
        LD(v_cw, "self_attn.v_sconv.conv1d.weight");
        LD(a_cw, "attn_sconv.conv1d.weight");
        LD(m_cw, "mlp_sconv.conv1d.weight");
        if (!c->sparse[i]) {
            LDW(dg, "mlp.gate_proj.weight"); LDW(du, "mlp.up_proj.weight"); LDW(dd, "mlp.down_proj.weight");
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.global_scale",i); l->dgs = load_scalar(m,nm,1.f);
        } else {
            LD(router, "mlp.gate.weight");
            LD(rbias,  "mlp.gate.e_score_correction_bias");
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.global_scale",i); l->rgs = load_scalar(m,nm,1.f);
            LDW(sh_g, "mlp.shared_experts.gate_proj");
            LDW(sh_u, "mlp.shared_experts.up_proj");
            LDW(sh_d, "mlp.shared_experts.down_proj");
#ifdef COLI_METAL
            /* Move the bf16 shared-expert weights into one page-aligned slab
             * (gates, then ups, then downs) and repoint the Wt handles into
             * it: the CPU path reads the same bytes it always did, and the
             * slab registers once so the GPU fmt=5 path can resolve it. */
            if (g_metal && l->sh_g.h && l->sh_u.h && l->sh_d.h) {
                int64_t I = c->moe_inter, ns = c->n_shared;
                size_t one = (size_t)ns*I*D*2, pg = 16384;
                size_t len = (3*one + pg - 1) / pg * pg;
                void *sl = NULL;
                if (!posix_memalign(&sl, pg, len)) {
                    memcpy((char*)sl,           l->sh_g.h, one);
                    memcpy((char*)sl + one,     l->sh_u.h, one);
                    memcpy((char*)sl + 2*one,   l->sh_d.h, one);
                    free(l->sh_g.h); free(l->sh_u.h); free(l->sh_d.h);
                    l->sh_g.h = (uint16_t*)sl;
                    l->sh_u.h = (uint16_t*)((char*)sl + one);
                    l->sh_d.h = (uint16_t*)((char*)sl + 2*one);
                    coli_metal_register(sl, len);
                }
            }
#endif
        }
        #undef LD
        #undef LDW
        /* conv states: raw inputs of the previous K-1 steps, zero-init */
        int kvdim = L_KV(c,i) * L_HD(c,i);
        for (int j = 0; j < 4; j++) if (allocate_state) {
            int C = (j < 2) ? kvdim : D;
            m->cs[j][i] = calloc((int64_t)C * (K-1), sizeof(float));
        }
    }
    /* container detection: converted snapshots store experts as U8 + .qs.
     * rb13/rb2 = bytes per packed row (D/2|D and I/2|I for int4|int8) */
    int64_t I = c->moe_inter, E = c->n_experts;
    for (int i = layer_begin; i < layer_end; i++) if (c->sparse[i]) {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",i);
        st_tensor *t = st_find(&m->S, nm);
        if (t && t->dtype == 3) {
            m->xq = 1;
            m->rb13 = t->nbytes / (E * 2*I);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",i);
            st_tensor *t2 = st_find(&m->S, nm);
            m->rb2 = t2->nbytes / (E * (int64_t)D);
            if (m->rb13 != D && m->rb13*2 != D) { fprintf(stderr,"unsupported container row size %lld\n",(long long)m->rb13); exit(1); }
        }
        break;
    }
    int nsp = 0; for (int i = layer_begin; i < layer_end; i++) nsp += c->sparse[i];
    int64_t slotb = m->xq ? m->rb13*2*I + m->rb2*D + (2*I+D)*4
                  : m->quant_bits ? 3*I*D + (2*I+D)*4 : 3*I*D*4;
    if (cap <= 0) {   /* auto: fit the LRU in available RAM, 20% + 4 GB headroom */
        double avail = mem_avail_bytes();
        cap = avail > 0 ? (int)((avail*0.80 - 4e9) / ((double)slotb * (nsp ? nsp : 1))) : 16;
        if (cap < 4) cap = 4;
        if (cap > c->n_experts) cap = c->n_experts;
        fprintf(stderr, "[cap auto] %d experts/layer (%.1f GB cache budget)\n",
                cap, (double)cap*slotb*nsp/1e9);
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = layer_begin; i < layer_end; i++) {
        LCache *lc = &m->cache[i];
        lc->cap = cap;
        if (c->sparse[i]) lc->slots = calloc(cap, sizeof(Slot));
        if (c->sparse[i]) {
            lc->slot_by_expert = malloc((size_t)E * sizeof(int));
            if (!lc->slot_by_expert) { fprintf(stderr,"OOM expert cache index\n"); exit(1); }
            for (int e = 0; e < E; e++) lc->slot_by_expert[e] = -1;
        }
    }
    /* container mode: slot storage is one page-aligned slab per sparse layer
     * (weights) plus one for the row scales, with every slot's pointers carved
     * out up front. Slots then recycle their region on eviction — no per-slot
     * malloc, and (under Metal) each slab registers ONCE for zero-copy GPU
     * resolve instead of churning register/unregister on every fill. */
    if (m->xq) {
        int64_t st13 = m->rb13*2*I, st2 = m->rb2*D;
        size_t pg = 16384;
        size_t wlen = ((size_t)cap*(st13+st2) + pg - 1) / pg * pg;
        size_t slen = ((size_t)cap*(2*I+D)*4 + pg - 1) / pg * pg;
        for (int i = layer_begin; i < layer_end; i++) {
            if (!c->sparse[i]) continue;
            void *wsl = NULL, *ssl = NULL;
            if (posix_memalign(&wsl, pg, wlen) || posix_memalign(&ssl, pg, slen)) {
                fprintf(stderr, "OOM expert slab layer %d (%zu MB)\n", i, (wlen+slen)>>20); exit(1); }
            for (int k = 0; k < cap; k++) {
                Slot *s = &m->cache[i].slots[k];
                s->p13 = (uint8_t*)wsl + (int64_t)k*(st13+st2);
                s->p2  = s->p13 + st13;
                s->s13 = (float*)ssl + (int64_t)k*(2*I+D);
                s->s2  = s->s13 + 2*I;
            }
#ifdef COLI_METAL
            if (g_metal) { coli_metal_register(wsl, wlen); coli_metal_register(ssl, slen); }
#endif
        }
    }
    /* usage counters; seeded from a previous run's history when present */
    if (init_telemetry) {
        rt_init("inkling", c->n_layers, E);
        for (int i = 0; i < c->n_layers; i++) if (!c->sparse[i]) rt_drop_row(i);
        rt_drop_row(c->n_layers);                 /* inkling has no MTP row */
        m->eusage = rt_counts_all();              /* alias: the bump sites stay as they are */
    }
    m->dense_load_s = now_s() - t0;
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    model_init_range(m, snap, cap, bits, 0, 0, 1, 1, 1);
}

static double mem_avail_bytes(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char ln[256]; double kb = 0;
    while (fgets(ln, sizeof(ln), f)) if (sscanf(ln, "MemAvailable: %lf", &kb) == 1) break;
    fclose(f);
    return kb * 1024.0;
#elif defined(__APPLE__)
    /* free + inactive + purgeable ~ Linux MemAvailable. Without this the auto
     * cap fell back to 16 experts/layer on a 128 GB Mac. */
    vm_size_t page = 0; host_page_size(mach_host_self(), &page);
    vm_statistics64_data_t vs; mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &n) != KERN_SUCCESS)
        return 0;
    return (double)(vs.free_count + vs.inactive_count + vs.purgeable_count) * page;
#else
    return 0;
#endif
}

/* ---------- routed-expert slots: serial bookkeeping, parallel fills ---------- */
#ifdef COLI_CACHE_INDEX_TEST
static uint64_t g_slot_index_probes;
#define SLOT_INDEX_PROBE() (g_slot_index_probes++)
#else
#define SLOT_INDEX_PROBE() ((void)0)
#endif
/* Cache identity is mutated only by slot_acquire(), which keeps this index in
 * lockstep with the slot. Fills change bytes/filled but never identity. The
 * validation makes a stale/corrupt entry a miss instead of serving another
 * expert's weights; production slot_find then touches the LRU clock. */
static Slot *slot_indexed(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer];
    if (eid < 0 || eid >= m->c.n_experts || !lc->slot_by_expert) return NULL;
    SLOT_INDEX_PROBE();
    int i = lc->slot_by_expert[eid];
    if (i < 0 || i >= lc->n || lc->slots[i].eid != eid) return NULL;
    return &lc->slots[i];
}
static Slot *slot_find(Model *m, int layer, int eid) {
    Slot *s = slot_indexed(m, layer, eid);
    if (s) s->used = ++m->clock;
    return s;
}

/* allocate a slot (or evict the LRU non-pinned one); serial callers only */
static Slot *slot_acquire(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer]; Cfg *c = &m->c;
    if (eid < 0 || eid >= c->n_experts || !lc->slot_by_expert) {
        fprintf(stderr,"layer %d: invalid expert id %d for cache index\n",layer,eid); exit(1); }
    int64_t D = c->hidden, I = c->moe_inter, n13 = 2*I*D, n2 = D*I;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        if (m->xq)              { /* slab-carved at model_init; nothing to allocate */ }
        else if (m->quant_bits) { s->q13 = malloc(n13); s->q2 = malloc(n2);
                                  s->s13 = falloc(2*I); s->s2 = falloc(D);
                                  if (!s->q13 || !s->q2) { fprintf(stderr,"OOM expert slot\n"); exit(1); } }
        else                    { s->f13 = falloc(n13); s->f2 = falloc(n2); }
    } else {
        int lru = -1;
        for (int i = 0; i < lc->n; i++)
            if (!lc->slots[i].pinned && (lru < 0 || lc->slots[i].used < lc->slots[lru].used)) lru = i;
        if (lru < 0) { fprintf(stderr, "layer %d: cache cap %d entirely pinned\n", layer, lc->cap); exit(1); }
        s = &lc->slots[lru];
    }
    int si = (int)(s - lc->slots);
    if (s->eid >= 0 && s->eid < c->n_experts &&
        lc->slot_by_expert[s->eid] == si)
        lc->slot_by_expert[s->eid] = -1;
    s->eid = eid; s->used = ++m->clock; s->filled = 0; s->pinned = 0;
    lc->slot_by_expert[eid] = si;
    return s;
}

/* pure I/O (+ optional requant): safe to run in parallel across slots */
static void slot_fill(Model *m, int layer, Slot *s) {
    Cfg *c = &m->c;
    int64_t D = c->hidden, I = c->moe_inter, n13 = 2*I*D, n2 = D*I;
    int64_t eid = s->eid;
    char nm[320], qs[340];
    if (m->xq) {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",layer);
        read_u8_slice(&m->S, nm, s->p13, eid*2*I*m->rb13, 2*I*m->rb13);
        snprintf(qs,sizeof(qs),"%s.qs",nm);
        read_f32_slice(&m->S, qs, s->s13, eid*2*I, 2*I);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",layer);
        read_u8_slice(&m->S, nm, s->p2, eid*D*m->rb2, D*m->rb2);
        snprintf(qs,sizeof(qs),"%s.qs",nm);
        read_f32_slice(&m->S, qs, s->s2, eid*D, D);
    } else if (m->quant_bits) {
        float *tmp = falloc(n13 > n2 ? n13 : n2);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",layer);
        read_f32_slice(&m->S, nm, tmp, eid*n13, n13);
        quantize_rows(tmp, s->q13, s->s13, 2*I, D, m->quant_bits);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",layer);
        read_f32_slice(&m->S, nm, tmp, eid*n2, n2);
        quantize_rows(tmp, s->q2, s->s2, D, I, m->quant_bits);
        free(tmp);
    } else {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",layer);
        read_f32_slice(&m->S, nm, s->f13, eid*n13, n13);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",layer);
        read_f32_slice(&m->S, nm, s->f2, eid*n2, n2);
    }
    s->filled = 1;
}

/* pin the top-N experts per sparse layer from a usage-history file (colibri
 * .coli_usage convention: one uint32 count per expert per layer). Pins are
 * regular cache slots flagged non-evictable, filled in parallel at startup.
 * Toggles: PIN=off (or PIN=0) skips cache warming entirely (no seeding, no
 * pins, cold LRU start); PIN_N=0 seeds the ranking from the history but pins
 * nothing; PIN=<path> uses an alternate history file; PIN_N=<n> pin depth. */
static void pins_load(Model *m, const char *snap) {
    Cfg *c = &m->c; int E = c->n_experts;
    char up[2048];
    const char *env = getenv("PIN");
    if (env && (!strcmp(env, "off") || !strcmp(env, "0"))) {
        fprintf(stderr, "[pin] cache warming disabled (PIN=%s)\n", env);
        return;
    }
    if (env) snprintf(up, sizeof(up), "%s", env);
    else snprintf(up, sizeof(up), "%s/.coli_usage", snap);
    /* Reads both layouts: the IKU1 block every previous inkling wrote, and the shared
     * text format. Dimension and identity checks moved into the reader, which refuses
     * by name instead of with a generic 'ignoring'. Counts land straight in eusage. */
    if (rt_load(up) <= 0) return;
    int cap = m->cache[0].cap;
    /* default: pin half the cap. Measured on the 975B: cap/4 (19/layer) gave
     * 83.6% hit / 0.32 tok/s; 40/layer gave 95.6% / 0.80 tok/s — decode fills
     * run at queue depth ~1, so every pinned expert removes a ~35ms stall. */
    m->npin = getenv("PIN_N") ? atoi(getenv("PIN_N")) : cap/2;
    if (m->npin > cap - 8) m->npin = cap - 8;
    if (m->npin < 0) m->npin = 0;
    Slot **ps = malloc((size_t)c->n_layers * m->npin * sizeof(Slot*));
    int *pl = malloc((size_t)c->n_layers * m->npin * sizeof(int));
    int np = 0;
    for (int i = 0; i < c->n_layers; i++) {
        const uint32_t *tmp = rt_counts(i);                /* seeded by rt_load above */
        if (!tmp || !c->sparse[i] || !m->npin) continue;
        for (int r = 0; r < m->npin; r++) {                /* top-N selection */
            int best = -1; uint32_t bv = 0;
            for (int e = 0; e < E; e++) {
                int taken = slot_indexed(m, i, e) != NULL;
                if (!taken && tmp[e] >= bv && tmp[e] > 0) { bv = tmp[e]; best = e; }
            }
            if (best < 0) break;
            Slot *s = slot_acquire(m, i, best);
            s->pinned = 1;
            ps[np] = s; pl[np] = i; np++;
        }
    }
    if (np) {
        double t0 = now_s();
        #pragma omp parallel for schedule(dynamic,1)
        for (int j = 0; j < np; j++) slot_fill(m, pl[j], ps[j]);
        fprintf(stderr, "[pin] %d experts pinned (%d/layer) from %s in %.1fs\n",
                np, m->npin, up, now_s()-t0);
    }
    free(ps); free(pl);
}

/* usage snapshot: rewritten after every generation run (same contract as
 * glm's .coli_usage — copy it aside if you need a stable ranking).
 * USAGE_SAVE=0 skips the rewrite (e.g. benchmark loops that would skew the
 * ranking); PIN=off also implies no save (that run never seeded counts). */
static int usage_save(Model *m, const char *snap) {
    (void)m;                              /* the counters live in route_trace.h now */
    char up[2048];
    const char *env = getenv("PIN");
    /* USAGE_SAVE=0 is honoured inside rt_save itself (#1039) */
    if (env && (!strcmp(env, "off") || !strcmp(env, "0"))) return 0;
    if (env) snprintf(up, sizeof(up), "%s", env);
    else snprintf(up, sizeof(up), "%s/.coli_usage", snap);
    /* One format for every engine now: sparse text with the dimension and identity
     * header. The IKU1 block this replaced is still readable on load, so a history
     * written by an older inkling keeps working and is rewritten in the new form. */
    return rt_save(up, 1);
}

/* KV rows actually kept per layer: sliding layers only ever attend to the last
 * `window` positions (t0 clamp in attention), so their cache is a ring of
 * `window` rows instead of max_t — at 32k context that is a ~64x cut on the
 * 5-of-6 sliding layers. Global layers keep the full max_t. Must be computed
 * from the SAME max_t the buffers were allocated with (m->max_t). */
static inline int kv_ring_rows(const Cfg *c, int li, int max_t) {
    return (c->local[li] && c->window > 0 && c->window < max_t) ? c->window : max_t;
}

/* ---------- attention (GQA + sliding/global + relative bias + K/V sconv) ---------- */
static void attention(Model *m, Layer *l, int li, float *x, int S, int pos0, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, H = L_HEADS(c,li), KV = L_KV(c,li), hd = L_HD(c,li), ext = L_EXT(c,li);
    int local = c->local[li];
    /* the ring made an over-run silent (t%win wraps instead of writing OOB), so
     * fail fast here: every caller sizes the cache via kv_alloc before stepping */
    if (pos0 + S > m->max_t) { fprintf(stderr, "attention: pos %d+%d exceeds kv alloc %d\n", pos0, S, m->max_t); exit(1); }
    int qdim = H*hd, kvdim = KV*hd, group = H/KV;
    float *q  = falloc((int64_t)S*qdim);
    float *k  = falloc((int64_t)S*kvdim);
    float *vv = falloc((int64_t)S*kvdim);
    float *rr = falloc((int64_t)S*H*c->d_rel);
    matmul_w(q,  x, l->q, S, D, qdim);
    matmul_w(k,  x, l->k, S, D, kvdim);
    matmul_w(vv, x, l->v, S, D, kvdim);
    matmul_w(rr, x, l->r, S, D, H*c->d_rel);
    /* short convs on K and V (sequence-wise, over the raw projections) */
    sconv_apply(k,  S, kvdim, l->k_cw, m->cs[0][li], c->conv_k);
    sconv_apply(vv, S, kvdim, l->v_cw, m->cs[1][li], c->conv_k);
    /* per-head q/k rmsnorm (scaling below is 1/hd, not 1/sqrt(hd), because of this) */
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H;  h++) rmsnorm_row(q + (int64_t)s*qdim  + h*hd, q + (int64_t)s*qdim  + h*hd, l->qn, hd, c->eps);
        for (int h = 0; h < KV; h++) rmsnorm_row(k + (int64_t)s*kvdim + h*hd, k + (int64_t)s*kvdim + h*hd, l->kn, hd, c->eps);
    }
    /* Rows for positions inside this batch are read from the k/vv scratch, not
     * the cache: with a ring, appending the whole batch up front could overwrite
     * history rows that earlier queries in the batch still need. The scratch
     * holds exactly what the cache would (post-sconv, post-rmsnorm), so the
     * arithmetic is unchanged; the cache is appended after scoring. */
    int win = kv_ring_rows(c, li, m->max_t);
    float scale = 1.f / (float)hd;
    float *ctx = falloc((int64_t)S*qdim);
    #pragma omp parallel
    {
        float *rl = malloc(ext * sizeof(float));
        float *sc = malloc((size_t)m->max_t * sizeof(float));
        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < H; h++) {
            for (int s = 0; s < S; s++) {
                int qpos = pos0 + s;
                int t0 = local && qpos - c->window + 1 > 0 ? qpos - c->window + 1 : 0;
                int tb = pos0 > t0 ? pos0 : t0;   /* first row served by the scratch */
                /* mix the relative-bias bank for this (token, head): rl[dist] */
                const float *rv = rr + (int64_t)s*H*c->d_rel + h*c->d_rel;
                for (int e = 0; e < ext; e++) {
                    float acc = 0.f;
                    for (int d = 0; d < c->d_rel; d++) acc += rv[d] * l->relp[(int64_t)d*ext + e];
                    rl[e] = acc;
                }
                /* tau: log-length scaling on global layers (f32, per query pos) */
                float tau = 1.f;
                if (!local && c->log_floor > 0) {
                    double en = (double)(qpos + 1) / c->log_floor;
                    if (en > 1.0) tau = 1.f + c->log_alpha * (float)log(en);
                }
                const float *qv = q + (int64_t)s*qdim + h*hd;
                const float *Kh = m->K[li] + ((int64_t)(h/group)*win)*hd;
                const float *Kb = k  + (int64_t)(h/group)*hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *kv = t < tb ? Kh + (int64_t)(t % win)*hd
                                             : Kb + (int64_t)(t - pos0)*kvdim;
                    float acc = 0.f;
                    for (int d = 0; d < hd; d++) acc += qv[d]*kv[d];
                    int dist = qpos - t;
                    sc[t - t0] = tau * (acc*scale + (dist < ext ? rl[dist] : 0.f));
                }
                int n = qpos - t0 + 1;
                softmax_row(sc, n);
                float *cx = ctx + (int64_t)s*qdim + h*hd;
                for (int d = 0; d < hd; d++) cx[d] = 0.f;
                const float *Vh = m->V[li] + ((int64_t)(h/group)*win)*hd;
                const float *Vb = vv + (int64_t)(h/group)*hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *vrow = t < tb ? Vh + (int64_t)(t % win)*hd
                                               : Vb + (int64_t)(t - pos0)*kvdim;
                    float a = sc[t - t0];
                    for (int d = 0; d < hd; d++) cx[d] += a * vrow[d];
                }
            }
        }
        free(rl); free(sc);
    }
    /* append K,V to the cache (ring on sliding layers); rows the ring would
     * overwrite within this same batch are skipped, they can never be read */
    int s0 = S - win > 0 ? S - win : 0;
    for (int s = s0; s < S; s++) for (int h = 0; h < KV; h++) {
        int t = pos0 + s;
        memcpy(m->K[li] + ((int64_t)h*win + t % win)*hd, k  + (int64_t)s*kvdim + h*hd, hd*sizeof(float));
        memcpy(m->V[li] + ((int64_t)h*win + t % win)*hd, vv + (int64_t)s*kvdim + h*hd, hd*sizeof(float));
    }
    matmul_w(out, ctx, l->o, S, qdim, D);
    free(q); free(k); free(vv); free(rr); free(ctx);
}

/* ---------- dense MLP ---------- */
static void dense_mlp(Model *m, Layer *l, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, I = c->dense_inter;
    float *g = falloc((int64_t)S*I), *u = falloc((int64_t)S*I);
    matmul_w(g, x, l->dg, S, D, I);
    matmul_w(u, x, l->du, S, D, I);
    for (int64_t i = 0; i < (int64_t)S*I; i++) g[i] = siluf(g[i]) * u[i];
    matmul_w(out, g, l->dd, S, I, D);
    for (int64_t i = 0; i < (int64_t)S*D; i++) out[i] *= l->dgs;
    free(g); free(u);
}

/* ---------- MoE: sigmoid router + bias top-k, joint routed+shared weights ----------
 * Three passes per layer call: (1) route every position and acquire slots,
 * (2) fill ALL missing experts in one parallel burst (the NVMe wants queue
 * depth — during prefill this batches the whole sequence's misses), then
 * (3) compute. */
/* shared experts for all S positions: gamma inside (before down_proj is
 * linear, so applied at the end). Factored out so the Metal path can run it
 * on the CPU while the last routed-expert round is in flight on the GPU.
 *
 * Prefill batches positions so each shared matrix is traversed once per chunk,
 * rather than once per token. Decode (S=1) deliberately stays on the original
 * scalar path: its scratch remains tiny and the compiler sees a one-row GEMV.
 * Bound the batch scratch to 64 MiB so a long prompt cannot turn this speedup
 * into a memory spike. INK_SHARED_BATCH=0 restores the scalar path; a positive
 * value sets a smaller maximum row count for deterministic A/B tests. */
static int shared_batch_rows(int S, int D, int I) {
    if (S <= 1) return 1;
    const char *env = getenv("INK_SHARED_BATCH");
    if (env) {
        int requested = atoi(env);
        if (requested <= 0) return 1;
        if (requested < S) S = requested;
    }
    int64_t row_bytes = ((int64_t)2*I + D) * (int64_t)sizeof(float);
    int64_t bounded = (64LL << 20) / (row_bytes > 0 ? row_bytes : 1);
    if (bounded < 1) bounded = 1;
    if (S > bounded) S = (int)bounded;
    return S;
}
static void shared_experts_cpu(Model *m, Layer *l, const float *x, int S,
                               float *out, const float *wgt,
                               float *g, float *u, float *hh) {
    Cfg *c = &m->c;
    int D = c->hidden, K = c->topk, I = c->moe_inter, ns = c->n_shared;
    if (ns <= 0 || S <= 0) return;
    int B = shared_batch_rows(S, D, I);
    if (B == 1) {
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s*D;
            float *os = out + (int64_t)s*D;
            const float *w = wgt + (int64_t)s*(K+ns);
            for (int j = 0; j < ns; j++) {
                matmul_w(g, xs, wt_off_i(l->sh_g, (int64_t)j*I*D, D), 1, D, I);
                matmul_w(u, xs, wt_off_i(l->sh_u, (int64_t)j*I*D, D), 1, D, I);
                for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                matmul_w(hh, g, wt_off_i(l->sh_d, (int64_t)j*D*I, I), 1, I, D);
                for (int d = 0; d < D; d++) os[d] += w[K+j] * hh[d];
            }
        }
        return;
    }
    float *bg = falloc((int64_t)2*B*I), *bu = bg + (int64_t)B*I;
    float *bh = falloc((int64_t)B*D);
    for (int base = 0; base < S; base += B) {
        int rows = S - base < B ? S - base : B;
        for (int j = 0; j < ns; j++) {
            matmul_w(bg, x + (int64_t)base*D,
                     wt_off_i(l->sh_g, (int64_t)j*I*D, D), rows, D, I);
            matmul_w(bu, x + (int64_t)base*D,
                     wt_off_i(l->sh_u, (int64_t)j*I*D, D), rows, D, I);
            for (int64_t q = 0; q < (int64_t)rows*I; q++) bg[q] = siluf(bg[q]) * bu[q];
            matmul_w(bh, bg, wt_off_i(l->sh_d, (int64_t)j*D*I, I), rows, I, D);
            for (int s = 0; s < rows; s++) {
                float *os = out + (int64_t)(base+s)*D;
                const float *w = wgt + (int64_t)(base+s)*(K+ns);
                const float *hs = bh + (int64_t)s*D;
                for (int d = 0; d < D; d++) os[d] += w[K+j] * hs[d];
            }
        }
    }
    free(bg); free(bh);
}

static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk, I = c->moe_inter, ns = c->n_shared;
    int ET = E + ns;
    float *logits = falloc((int64_t)S*ET);
    matmul(logits, x, l->router, S, D, ET);
    memset(out, 0, (int64_t)S*D*sizeof(float));
    int   *idx  = malloc((size_t)S*K*sizeof(int));
    int   *keff = malloc((size_t)S*sizeof(int));      /* routed effettivi per token (TOPP) */
    float *wgt  = malloc((size_t)S*(K+ns)*sizeof(float));
    Slot **use  = malloc((size_t)S*K*sizeof(Slot*));
    Slot **fill = malloc((size_t)S*K*sizeof(Slot*));
    int  *fl    = malloc((size_t)S*K*sizeof(int));
    int nfill = 0;
    /* pass 1: routing + slot bookkeeping (serial) */
    for (int s = 0; s < S; s++) {
        float *lg = logits + (int64_t)s*ET;
        int *si = idx + (int64_t)s*K;
        /* selection: sigmoid(routed) + correction bias, top-K */
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (si[j]==e){taken=1;break;}
                float ch = sigmoidf(lg[e]) + l->rbias[e];
                if (!taken && ch > bv) { bv = ch; best = e; }
            }
            /* SEC: all-NaN logits leave best at -1, and lg[-1] / eusage[-1] are
             * next. See rt_router_pick in route_trace.h. */
            si[kk] = rt_router_pick(best, kk, E, layer);
        }
        /* combine weights: sigmoids of the raw logits of (topK routed + shared),
         * normalized to sum 1 over all K+ns, x route_scale x gate.global_scale */
        float *w = wgt + (int64_t)s*(K+ns); float sum = 0.f;
        for (int kk = 0; kk < K; kk++)  { w[kk]   = sigmoidf(lg[si[kk]]); sum += w[kk]; }
        for (int j = 0; j < ns; j++)    { w[K+j]  = sigmoidf(lg[E+j]);    sum += w[K+j]; }
        for (int kk = 0; kk < K+ns; kk++) w[kk] *= c->route_scale * l->rgs / sum;
        /* TOPP: tieni i routed fino al peso cumulato p, scarta la coda. Stessa
         * semantica di colibri.c (g_topp) e kimi_k3.c (K3_TOPP): NON rinormalizza,
         * il peso scartato semplicemente non contribuisce. E' una leva di QUALITA'
         * — cambia il calcolo — quindi opt-in e annunciata all'avvio.
         * Su questo motore vale piu' che sugli altri: Inkling e' limitato dal disco
         * (topk=6 x 28 MB di esperto x 66 layer ~ 11 GB letti per token), e ogni
         * esperto scartato e' un esperto NON letto.
         * L'ordine di si[] segue sigmoid(logit)+bias mentre il peso e' sigmoid(logit)
         * senza bias, quindi i pesi non sono garantiti decrescenti: si riordina la
         * coppia (peso, id) prima di tagliare, come fa colibri.c. */
        keff[s] = K;
        m->ereq += K;
        if (g_topp > 0.f && g_topp < 1.f) {
            for (int a = 1; a < K; a++) { int ii = si[a]; float ww = w[a]; int b = a-1;
                while (b >= 0 && w[b] < ww) { w[b+1] = w[b]; si[b+1] = si[b]; b--; }
                w[b+1] = ww; si[b+1] = ii; }
            float tot = 1e-20f; for (int kk = 0; kk < K; kk++) tot += w[kk];
            float cum = 0;
            for (int kk = 0; kk < K; kk++) { cum += w[kk];
                if (cum >= g_topp * tot) { keff[s] = kk+1; break; } }
        }
        m->euse += keff[s];
    }
    /* Il ciclo sotto ACQUISISCE uno slot per ogni coppia (token, esperto) e ne tiene
     * il puntatore fino al calcolo. slot_acquire evince l'LRU quando la cache e'
     * piena — anche uno slot gia' consegnato in QUESTA chiamata: il puntatore resta
     * valido ma lo slot ora contiene un ALTRO esperto, e il modello calcola con i
     * pesi sbagliati. In silenzio: niente crash, solo output incoerente. Acquisire
     * tutte le S*K coppie in una volta pretendeva quindi una cache capace di tenere
     * tutti gli esperti distinti del batch (18 token x topk 6 = fino a 108 slot per
     * layer), e sotto quella soglia il modello sembrava rotto.
     * Fix: si lavora a GIRI di al piu' `cap` coppie — acquisisci, riempi, calcola e
     * accumula — cosi' nessuno slot puo' essere evinto mentre serve. L'output MoE e'
     * una somma pesata, quindi accumulare a giri da' lo stesso risultato di una
     * passata sola, e la cache puo' scendere fino a 1 slot per layer (piu' letture
     * da disco, ma memoria proporzionale a `cap` invece che al batch). */
    int cap = m->cache[layer].cap; if (cap < 1) cap = 1;
    float *g = falloc(2*I), *u = g + I, *hh = falloc(D);
    int q4 = m->xq && m->rb13*2 == D;   /* packed int4 vs int8 container */
    int shared_done = 0;                /* set when overlapped with the last GPU round */
    int64_t npair = (int64_t)S*K;
#ifdef COLI_METAL
    /* per-round scratch for the batched GPU submit: pairs grouped by expert,
     * activations packed in group order. Allocated once per moe() call. */
    float *mxg = NULL, *mrw = NULL; const void **mgp = NULL, **mup = NULL, **mdp = NULL;
    const float **mgs = NULL, **mus = NULL, **mds = NULL; Slot **mslot = NULL;
    int *mxoff = NULL, *mnr = NULL, *mrows = NULL, *mgi = NULL, *mfp = NULL;
    /* GPU for decode too, now that the residency set is on by default: with
     * it a decode block runs in ~3ms and measures 1.76 tok/s vs 0.84 on CPU
     * (Inkling-Small, M-series 128 GB). Without the set the same block paid
     * ~135ms of useResource churn and CPU won — INK_METAL_MIN_S=2 restores
     * the prefill-only gate if that regime ever returns. */
    int metal_min_s = getenv("INK_METAL_MIN_S") ? atoi(getenv("INK_METAL_MIN_S")) : 1;
    if (g_metal && m->xq && S >= metal_min_s) {
        mxg = falloc((int64_t)cap*D); mrw = falloc(cap);
        mgp = malloc(cap*sizeof(void*)); mup = malloc(cap*sizeof(void*)); mdp = malloc(cap*sizeof(void*));
        mgs = malloc(cap*sizeof(float*)); mus = malloc(cap*sizeof(float*)); mds = malloc(cap*sizeof(float*));
        mslot = malloc(cap*sizeof(Slot*));
        mxoff = malloc((cap+1)*sizeof(int)); mnr = malloc(cap*sizeof(int));
        mrows = malloc(cap*sizeof(int)); mgi = malloc(cap*sizeof(int)); mfp = malloc(cap*sizeof(int));
    }
    /* Shared experts as ONE bf16 (fmt=5) block, submitted BEFORE the routed
     * rounds so it rides the GPU while the CPU routes, fills, and packs.
     * Every token uses every shared expert, so group j is simply all S rows
     * with weight w[K+j]. Falls back to the CPU loop if begin refuses. */
    ColiMetalMoeHandle *sh_h = NULL;
    float *sxg = NULL, *srw = NULL;
    if (mxg && ns > 0 && ns <= 8 && l->sh_g.h && l->sh_u.h && l->sh_d.h &&
        !(getenv("INK_METAL_SHARED") && *getenv("INK_METAL_SHARED") == '0')) {
        double ts = now_s();
        sxg = falloc((int64_t)ns*S*D); srw = falloc((int64_t)ns*S);
        const void *sgp[8], *sup[8], *sdp[8];
        const float *sscale[8];
        int sxoff[9], snr[8];
        int *srows = malloc((size_t)ns*S*sizeof(int));
        for (int j = 0; j < ns; j++) {
            sgp[j] = l->sh_g.h + (int64_t)j*I*D;
            sup[j] = l->sh_u.h + (int64_t)j*I*D;
            sdp[j] = l->sh_d.h + (int64_t)j*D*I;
            sscale[j] = (const float*)sgp[j];        /* fmt=5 never reads scales */
            sxoff[j] = j*S; snr[j] = S;
            for (int s = 0; s < S; s++) {
                memcpy(sxg + ((int64_t)j*S + s)*D, x + (int64_t)s*D, (size_t)D*sizeof(float));
                srows[j*S + s] = s;
                srw[j*S + s] = wgt[(int64_t)s*(K+ns) + K + j];
            }
        }
        sxoff[ns] = ns*S;
        sh_h = coli_metal_moe_block_begin(ns, D, I, 5, 0, sgp, sup, sdp,
                                          sscale, sscale, sscale,
                                          sxg, sxoff, snr, srows, srw);
        free(srows);
        m->t_shared += now_s() - ts;
    }
#endif
    for (int64_t base = 0; base < npair; base += cap) {
        int64_t end = base + cap < npair ? base + cap : npair;
        nfill = 0;
        for (int64_t t = base; t < end; t++) {           /* acquisizione del giro */
            int s = (int)(t / K), kk = (int)(t % K);
            if (kk >= keff[s]) { use[t - base] = NULL; continue; }   /* scartato da TOPP */
            int eid = idx[(int64_t)s*K + kk];
            if (m->eusage && m->eusage[layer]) m->eusage[layer][eid]++;
            Slot *e = slot_find(m, layer, eid);
            if (e) m->hits++;
            else {
                m->miss++;
                e = slot_acquire(m, layer, eid);
                fill[nfill] = e; fl[nfill] = layer; nfill++;
            }
            use[t - base] = e;
        }
        if (nfill) {                                      /* riempimento in parallelo */
            double tf = now_s();
            #pragma omp parallel for schedule(dynamic,1)
            for (int j = 0; j < nfill; j++) slot_fill(m, fl[j], fill[j]);
            m->t_fill += now_s() - tf;
        }
        /* Validate before the CPU/Metal split: either backend must refuse a
         * cache slot whose weights belong to a different routed expert. */
        for (int64_t t = base; t < end; t++) {
            Slot *e = use[t - base];
            if (!e) continue;                              /* scartato da TOPP */
            int s = (int)(t / K), kk = (int)(t % K);
            if (e->eid != idx[(int64_t)s*K + kk]) {
                fprintf(stderr, "layer %d: cache served expert %d for requested expert %d\n",
                        layer, e->eid, idx[(int64_t)s*K + kk]);
                exit(1);
            }
        }
        double te = now_s();
#ifdef COLI_METAL
        if (mxg) {
            /* group this round's (token, expert) pairs by expert and submit ONE
             * command buffer; the kernel scatter-adds rw*hh into out with the
             * same accumulate semantics as the CPU loop below. 0 -> CPU. */
            int nb = 0;
            for (int64_t t = base; t < end; t++) {
                Slot *e = use[t - base];
                if (!e) { mgi[t - base] = -1; continue; }
                int gi = -1;
                for (int j = 0; j < nb; j++) if (mslot[j] == e) { gi = j; break; }
                if (gi < 0) { gi = nb++; mslot[gi] = e; mnr[gi] = 0; }
                mgi[t - base] = gi; mnr[gi]++;
            }
            if (nb) {
                mxoff[0] = 0;
                for (int j = 0; j < nb; j++) mxoff[j+1] = mxoff[j] + mnr[j];
                memcpy(mfp, mxoff, nb*sizeof(int));
                for (int64_t t = base; t < end; t++) {
                    int gi = mgi[t - base];
                    if (gi < 0) continue;
                    int s = (int)(t / K), kk = (int)(t % K);
                    int r = mfp[gi]++;
                    memcpy(mxg + (int64_t)r*D, x + (int64_t)s*D, (size_t)D*sizeof(float));
                    mrows[r] = s;
                    mrw[r] = wgt[(int64_t)s*(K+ns) + kk];
                }
                for (int j = 0; j < nb; j++) {
                    Slot *e = mslot[j];
                    mgp[j] = e->p13; mup[j] = e->p13 + (int64_t)I*m->rb13; mdp[j] = e->p2;
                    mgs[j] = e->s13; mus[j] = e->s13 + I; mds[j] = e->s2;
                }
                /* Last round: submit async and run the shared experts on the
                 * CPU while the GPU computes the routed ones — they are
                 * independent (both read x, both accumulate into out, and the
                 * GPU only touches out in _end's scatter-add, after the wait).
                 * On a GPU fault _end returns 0 and the CPU loop below redoes
                 * the round; shared_done stays set either way. */
                if (base + cap >= npair && !sh_h) {
                    ColiMetalMoeHandle *h = coli_metal_moe_block_begin(
                        nb, D, I, q4 ? 2 : 1, 0, mgp, mup, mdp, mgs, mus, mds,
                        mxg, mxoff, mnr, mrows, mrw);
                    if (h) {
                        double ts = now_s();
                        shared_experts_cpu(m, l, x, S, out, wgt, g, u, hh);
                        double sh = now_s() - ts;
                        m->t_shared += sh;
                        shared_done = 1;
                        int ok = coli_metal_moe_block_end(h, out);
                        m->t_expert += (now_s() - te) - sh;
                        if (ok) continue;                  /* round done on the GPU */
                        te = now_s();                      /* fault: CPU redo below */
                    }
                }
                if (coli_metal_moe_block(nb, D, I, q4 ? 2 : 1, 0, mgp, mup, mdp, mgs, mus, mds,
                                         mxg, mxoff, mnr, mrows, mrw, out, S)) {
                    m->t_expert += now_s() - te;
                    continue;                              /* round done on the GPU */
                }
            } else { m->t_expert += now_s() - te; continue; }
        }
#endif
        for (int64_t t = base; t < end; t++) {            /* calcolo + accumulo */
            int s = (int)(t / K), kk = (int)(t % K);
            Slot *e = use[t - base];
            if (!e) continue;                              /* scartato da TOPP */
            const float *xs = x + (int64_t)s*D;
            float *os = out + (int64_t)s*D;
            float *w = wgt + (int64_t)s*(K+ns);
            if (m->xq) {
                if (q4) {
                    matmul_q4(g, xs, e->p13, e->s13, D, 2*I);   /* gate rows then up rows */
                    for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                    matmul_q4(hh, g, e->p2, e->s2, I, D);
                } else {
                    matmul_q(g, xs, (int8_t*)e->p13, e->s13, D, 2*I);
                    for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                    matmul_q(hh, g, (int8_t*)e->p2, e->s2, I, D);
                }
            } else if (m->quant_bits) {
                matmul_q(g, xs, e->q13, e->s13, D, 2*I);
                for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                matmul_q(hh, g, e->q2, e->s2, I, D);
            } else {
                matmul(g, xs, e->f13, 1, D, 2*I);
                for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                matmul(hh, g, e->f2, 1, I, D);
            }
            for (int d = 0; d < D; d++) os[d] += w[kk] * hh[d];
        }
        m->t_expert += now_s() - te;
    }
#ifdef COLI_METAL
    /* GPU shared block: wait + scatter-add. A fault falls through to CPU. */
    if (sh_h) {
        double ts = now_s();
        if (coli_metal_moe_block_end(sh_h, out)) shared_done = 1;
        m->t_shared += now_s() - ts;
    }
#endif
    /* shared experts: una volta per token, fuori dai giri (non usano la cache).
     * Under Metal these run on the GPU as an fmt=5 block overlapped with the
     * routed rounds (or on the CPU during the last GPU round); either path
     * sets shared_done. */
    if (!shared_done) {
        double ts = now_s();
        shared_experts_cpu(m, l, x, S, out, wgt, g, u, hh);
        m->t_shared += now_s() - ts;
    }
    free(logits); free(idx); free(keff); free(wgt); free(use); free(fill); free(fl);
    free(g); free(hh);              /* u aliases g+I */
#ifdef COLI_METAL
    if (mxg) {
        free(mxg); free(mrw); free(mgp); free(mup); free(mdp);
        free(mgs); free(mus); free(mds); free(mslot);
        free(mxoff); free(mnr); free(mrows); free(mgi); free(mfp);
    }
    free(sxg); free(srw);
#endif
}

/* ---------- DMel audio embedding ----------
 * One frame = mel_bins u8 levels in [0, mel_vocab). Its decoder embedding is
 * sum_b E[b*mel_vocab + v_b], RMSNorm'd with the audio tower norm (eps is a
 * literal 1e-6 in the HF audio tower, independent of the text rms_norm_eps).
 * This row REPLACES the <|audio|> placeholder's text embedding — embed_norm
 * does not apply, matching masked_scatter in modeling_inkling.py. */
static void audio_embed_row(Model *m, const uint8_t *frame, float *out, float *tmp) {
    Cfg *c = &m->c; int D = c->hidden;
    memset(out, 0, (size_t)D * sizeof(float));
    for (int b = 0; b < c->mel_bins; b++) {
        int v = frame[b] < c->mel_vocab ? frame[b] : c->mel_vocab - 1;
        wt_row_f32(m->audio_enc, (int64_t)(b*c->mel_vocab + v)*D, tmp, D);
        for (int d = 0; d < D; d++) out[d] += tmp[d];
    }
    rmsnorm_row(out, out, m->audio_norm, D, 1e-6f);
}

/* count <|audio|> placeholders in a token sequence (0 if no audio tower) */
static int audio_tok_count(Model *m, const int *ids, int n) {
    if (!m->audio_norm) return 0;
    int k = 0;
    for (int i = 0; i < n; i++) k += (ids[i] == m->c.audio_tok);
    return k;
}

static void inkling_layers_forward_range(Model *m, float *x, int S, int pos0,
                                          int layer_begin, int layer_end) {
    Cfg *c = &m->c; int D = c->hidden;
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = layer_begin; i < layer_end; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++)
            rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D,
                        l->in_ln, D, c->eps);
        double ta = now_s();
        attention(m, l, i, nrm, S, pos0, tmp);
        m->t_attn += now_s() - ta;
        sconv_apply(tmp, S, D, l->a_cw, m->cs[2][i], c->conv_k);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++)
            rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D,
                        l->post_ln, D, c->eps);
        if (c->sparse[i]) moe(m, l, i, nrm, S, tmp);
        else dense_mlp(m, l, nrm, S, tmp);
        sconv_apply(tmp, S, D, l->m_cw, m->cs[3][i], c->conv_k);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    free(nrm); free(tmp);
}

/* ---------- one forward pass over S new tokens ----------
 * Returns malloc'd logits of the last token (unpadded vocab). If tf_out is
 * non-NULL also writes the per-position argmax (teacher-forcing check).
 * dmel: u8 [naud, mel_bins] frames consumed left-to-right by the <|audio|>
 * placeholder positions in ids (prefill only; decode steps pass NULL). */
static float *step_mm(Model *m, const int *ids, int S, int pos0, int *tf_out,
                      const uint8_t *dmel, int naud) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    float *arow = (dmel && naud > 0) ? falloc(D) : NULL;
    int aidx = 0;
    for (int s = 0; s < S; s++) {
        if (arow && m->audio_norm && ids[s] == c->audio_tok && aidx < naud) {
            audio_embed_row(m, dmel + (int64_t)aidx*c->mel_bins, x + (int64_t)s*D, arow);
            aidx++;
            continue;
        }
        wt_row_f32(m->embed, (int64_t)ids[s]*D, x + (int64_t)s*D, D);
        if (m->embed_norm) rmsnorm_row(x + (int64_t)s*D, x + (int64_t)s*D, m->embed_norm, D, c->eps);
    }
    free(arow);
    inkling_layers_forward_range(m, x, S, pos0, 0, c->n_layers);
    m->kv_len = pos0 + S;
    /* record what was just fed, at the positions it went to (kv_prefix.h).
     * Audio taints the record: every frame carries the same token id while the
     * mel payload differs, so ids alone cannot tell two clips apart. */
    kv_prefix_record(&m->kvp, ids, pos0, S);
    if (dmel && naud > 0) kv_prefix_taint(&m->kvp);
    float *last = falloc(D);
    float *logit = falloc(c->unpad_vocab);
    if (tf_out) {
        for (int s = 0; s < S; s++) {
            rmsnorm_row(last, x + (int64_t)s*D, m->final_norm, D, c->eps);
            for (int d = 0; d < D; d++) last[d] /= c->mup;
            matmul_w(logit, last, m->lm_head, 1, D, c->unpad_vocab);
            int best = 0; for (int i = 1; i < c->unpad_vocab; i++) if (logit[i] > logit[best]) best = i;
            tf_out[pos0 + s] = best;
        }
    }
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    for (int d = 0; d < D; d++) last[d] /= c->mup;
    matmul_w(logit, last, m->lm_head, 1, D, c->unpad_vocab);
    free(x); free(last);
    return logit;
}

static float *step(Model *m, const int *ids, int S, int pos0, int *tf_out) {
    return step_mm(m, ids, S, pos0, tf_out, NULL, 0);
}

static void state_reset(Model *m) {
    Cfg *c = &m->c;
    m->kv_len = 0;
    kv_prefix_clear(&m->kvp);
    for (int i = 0; i < c->n_layers; i++) {
        int kvdim = L_KV(c,i) * L_HD(c,i);
        for (int j = 0; j < 4; j++)
            memset(m->cs[j][i], 0, (int64_t)((j < 2) ? kvdim : c->hidden) * (c->conv_k-1) * sizeof(float));
    }
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    if (m->K && max_t <= m->max_t) return;   /* reuse across prompts when big enough */

    /* GROW, DO NOT RESTART.
     *
     * This used to free the buffers and allocate fresh ones, which discarded
     * every position already computed. That was invisible while each turn
     * re-prefilled anyway — but it defeats KV prefix reuse in exactly the case
     * reuse exists for: a conversation whose prompt is longer every turn asks
     * for a larger max_t every turn, so the state was thrown away just before
     * the point of using it.
     *
     * K/V are laid out [kv_head][rows][hd] with rows = kv_ring_rows(), so a
     * larger max_t changes the stride wherever rows follows max_t: those
     * contents cannot be realloc'd, they have to be re-laid-out head by head.
     * That copy costs a memcpy of what is already computed, which is nothing
     * beside re-running the prefill that produced it.
     *
     * Sliding layers whose ring is already window rows are the exception in
     * BOTH directions: their size does not depend on max_t (nothing to grow),
     * and a ring that has wrapped is not linear (position t lives at row
     * t % window), so the linear copy below would silently rotate it. Steal
     * the buffer instead — the slot map is unchanged, so it stays valid.
     * Every layer that does reach the copy IS linear: rows != old_rows only
     * happens when old_rows == old_max (a wrapped ring keeps rows == window
     * forever), and keep <= old_max <= rows there, so the copy fits. */
    float **oldK = m->K, **oldV = m->V;
    int old_max = m->max_t;
    int keep = (m->K && m->kv_len > 0 && m->kv_len <= max_t) ? m->kv_len : 0;

    m->max_t = max_t;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    int64_t ring = 0, full = 0;
    for (int i = 0; i < c->n_layers; i++) {
        int kv = L_KV(c,i), hd = L_HD(c,i);
        int64_t rows     = kv_ring_rows(c, i, max_t);
        int64_t old_rows = oldK ? kv_ring_rows(c, i, old_max) : 0;
        if (oldK && rows == old_rows) {
            m->K[i] = oldK[i]; m->V[i] = oldV[i];
            oldK[i] = NULL;    oldV[i] = NULL;
        } else {
            m->K[i] = falloc((int64_t)kv * rows * hd);
            m->V[i] = falloc((int64_t)kv * rows * hd);
            for (int h = 0; h < kv && keep; h++) {
                memcpy(m->K[i] + (int64_t)h * rows * hd,
                       oldK[i] + (int64_t)h * old_max * hd, (size_t)keep * hd * sizeof(float));
                memcpy(m->V[i] + (int64_t)h * rows * hd,
                       oldV[i] + (int64_t)h * old_max * hd, (size_t)keep * hd * sizeof(float));
            }
        }
        ring += 2 * (int64_t)kv * rows  * hd * (int64_t)sizeof(float);
        full += 2 * (int64_t)kv * max_t * hd * (int64_t)sizeof(float);
    }
    if (oldK) for (int i = 0; i < c->n_layers; i++) { free(oldK[i]); free(oldV[i]); }
    free(oldK); free(oldV);
    if (ring < full)
        fprintf(stderr, "[kv] %.1f MiB (ring buffers on sliding layers; full cache would be %.1f MiB)\n",
                ring/1048576.0, full/1048576.0);

    /* the record describes those same positions, so it survives with them --
     * unless its own allocation fails, in which case reuse simply stops. */
    if (kv_prefix_grow(&m->kvp, max_t, keep)) m->kv_len = keep;
    else                                      m->kv_len = 0;
}

/* greedy generation, olmoe.c-style */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out,
                     const uint8_t *dmel, int naud) {
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step_mm(m, prompt, np, 0, NULL, dmel, naud);
    int len = np;
    Cfg *c = &m->c;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->unpad_vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1, NULL);
    }
}

/* ---------- interactive prompt mode: greedy, streaming, stop on eos ---------- */
static void generate_stream(Model *m, Tok *T, const char *prompt, int n_new,
                            const uint8_t *dmel, int naud) {
    Cfg *c = &m->c;
    int cap = (int)strlen(prompt) + 16;
    int *ids = malloc(cap * sizeof(int));
    int np = tok_encode(T, prompt, (int)strlen(prompt), ids, cap);
    if (np <= 0) { fprintf(stderr, "empty prompt after tokenization\n"); return; }
    if (audio_tok_count(m, ids, np) != naud) {
        fprintf(stderr, "audio frames (%d) do not match <|audio|> placeholders (%d)%s\n",
                naud, audio_tok_count(m, ids, np),
                m->audio_norm ? "" : " — snapshot has no audio tensors (audio.safetensors)");
        free(ids); return;
    }
    kv_alloc(m, np + n_new + 8);
    printf("[%d prompt tokens%s] %s", np, naud ? " incl. audio" : "", naud ? "" : prompt);
    fflush(stdout);
    double t0 = now_s(), t1 = 0;
    float *logit = step_mm(m, ids, np, 0, NULL, dmel, naud);
    int len = np;
    char buf[512];
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->unpad_vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        if (s == 0) t1 = now_s();
        if (best == c->eos) { printf("\n[eos after %d tokens]", s); break; }
        int nb = tok_decode(T, &best, 1, buf, sizeof(buf)-1);
        buf[nb] = 0; fputs(buf, stdout); fflush(stdout);
        int one = best;
        len++;
        if (s == n_new - 1) break;
        logit = step(m, &one, 1, len - 1, NULL);
    }
    double dt = now_s() - t1;
    int gen = len - np;
    printf("\n[prefill %.1fs | %d tokens in %.1fs = %.2f tok/s | RSS %.1f GB]\n",
           t1 - t0, gen, dt, gen > 1 ? (gen-1)/dt : 0.0, rss_gb());
    /* One line, every engine, one format: `coli tune` sweeps scheduling knobs and
     * needs tokens-and-elapsed to compare candidates. Before this only colibri
     * emitted a parseable throughput line (REPLAY decode), so the tuner was
     * GLM-only and bannered the right model while launching the wrong engine
     * (#898). Printed to stdout, which is what autotune captures.
     * Tokens and seconds, not tok/s: the ratio is derived by the caller at full
     * precision (#852 -- two decimals of tok/s is one significant digit at the
     * rates this engine runs at). */
    printf("TUNE decode: %d tokens in %.3fs\n", gen > 1 ? gen - 1 : gen, dt);
    double wall = now_s() - t0;
#ifdef COLI_METAL
    if (g_metal) {
        uint64_t mok = 0, mfb = 0, mex = 0;
        coli_metal_moe_counts(&mok, &mfb, &mex);
        printf("[metal] %llu MoE blocks on GPU, %llu CPU fallbacks, %llu experts\n",
               (unsigned long long)mok, (unsigned long long)mfb, (unsigned long long)mex);
    }
#endif
    printf("[phases] fill %.1fs | expert-mm %.1fs | shared %.1fs | attn %.1fs | other %.1fs\n",
           m->t_fill, m->t_expert, m->t_shared, m->t_attn,
           wall - m->t_fill - m->t_expert - m->t_shared - m->t_attn);
    free(ids);
}

/* ---------- serve mode: openai_server.py engine protocol ----------
 * stdin:  SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>\n<payload>\n
 *         CANCEL <id>\n
 * stdout: READY sentinel once loaded, then per request a stream of
 *         DATA <id> <size>\n<bytes>\n frames and a final
 *         DONE <id> STAT <tok> <tps> <hit%> <rss> <prompt_tok> <len_limited>\n
 * Byte-identical to colibri.c's serve protocol so the shared openai_server.py
 * gateway drives inkling unchanged (v1: one request at a time; the KV slot arg
 * is accepted but every request re-prefills). */

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double rng_next(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}

/* temperature + top-p nucleus sampling; temp<=0 = greedy (the oracle path) */
typedef struct { float p; int i; } PI;
static int pi_desc(const void *a, const void *b) {
    float d = ((const PI*)b)->p - ((const PI*)a)->p;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}
static int sample_logits(const float *logit, int n, float temp, float top_p) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logit[i] > logit[best]) best = i;
    if (temp <= 0.f) return best;
    PI *c = malloc((size_t)n * sizeof(PI));
    double sum = 0;
    for (int i = 0; i < n; i++) {
        c[i].p = expf((logit[i] - logit[best]) / temp);
        c[i].i = i; sum += c[i].p;
    }
    qsort(c, n, sizeof(PI), pi_desc);
    double cut = (top_p > 0.f && top_p < 1.f) ? top_p * sum : sum;
    double acc = 0; int k = 0;
    while (k < n && acc < cut) acc += c[k++].p;
    double r = rng_next() * acc, run = 0;
    int pick = c[0].i;
    for (int i = 0; i < k; i++) { run += c[i].p; if (run >= r) { pick = c[i].i; break; } }
    free(c);
    return pick;
}

/* light repeat guard: recently emitted tokens get their logit divided by pen>1 */
static void apply_rep_penalty(float *logit, int n, const int *hist, int nhist, float pen) {
    if (pen <= 1.f) return;
    for (int i = 0; i < nhist; i++) {
        int t = hist[i];
        if (t < 0 || t >= n) continue;
        logit[t] = logit[t] > 0 ? logit[t] / pen : logit[t] * pen;
    }
}

/* reject a prompt that would overrun the served KV bound (CTX_MAX, default 8192) */
static const char *prompt_reject(int np, int want) {
    const char *cm = getenv("CTX_MAX");
    int ctx_max = cm ? atoi(cm) : 8192;
    if (np + want > ctx_max) return "context exceeds CTX_MAX";
    return NULL;
}

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen;
                 uint8_t *audio; int alen; } SReq;   /* raw DMel bytes after the payload */
#define SRV_QMAX 16
static SReq g_q[SRV_QMAX]; static int g_qn = 0;
static const ColiServeWireProfile inkling_wire = {
    .max_header_bytes = 511,
    .max_payload_bytes = 1u << 22,
    .max_extension_bytes = 1u << 26,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 0,
    .allow_extension_bytes = 1,
    .allow_prefix_hint = 0,
};

static void serve_request_dispose(SReq *request) {
    if (!request) return;
    free(request->payload);
    memset(request, 0, sizeof(*request));
}

static int stdin_readable(void) {
    /* Windows non ha fd_set/select in questa forma: la build falliva del tutto.
     * La versione portabile (con i fix #139/#195) vive in compat.h, incluso via st.h. */
    return coli_stdin_readable();
}

/* read one control line (+ payload for SUBMIT). cur_id: request in flight;
 * returns 1 if that request was cancelled, 0 otherwise, -1 on stdin EOF. */
static int serve_read_cmd(FILE *input, FILE *output, const char *cur_id) {
    ColiServeCommand command;
    ColiServeReadResult result=coli_serve_read_command(input,&inkling_wire,&command);
    if(result==COLI_SERVE_READ_EOF) return -1;
    if(result==COLI_SERVE_READ_BAD_FRAME) return -2;
    if(result==COLI_SERVE_READ_NOMEM){
        coli_serve_write_error(output,command.id,"out of memory"); return -2;
    }
    if(result==COLI_SERVE_READ_BAD_REQUEST&&
       command.kind==COLI_SERVE_COMMAND_SUBMIT){
        /* 6th field (optional, backward compatible): DMel byte count appended
         * verbatim after the text payload — frames x mel_bins u8 levels */
        /* SEC: max_tok was the one field nobody validated, and it is the one
         * the context check is built on. prompt_reject() asks
         * `np + want > ctx_max`; a negative `want` makes that sum smaller than
         * np, so the check passes for any prompt length. kv_alloc() is then
         * sized on the same np + max_tok + 8 and comes out shorter than the
         * prompt, and prefill writes past the end of the K/V cache -- a heap
         * overflow whose length and contents both follow from the request.
         *
         * The official gateway forces a positive integer, so this is not
         * reachable through openai_server.py. The SERVE protocol is public and
         * anything bridging it (socat, a custom gateway, a sidecar) exposes it
         * directly, so the check belongs here, next to the one plen already
         * has, rather than in one of its callers. */
        coli_serve_write_error(output,command.id,"bad submit header"); return -2;
    }
    if(result!=COLI_SERVE_READ_OK) return 0;
    if(command.kind==COLI_SERVE_COMMAND_CANCEL){
        int matched=cur_id&&!strcmp(command.id,cur_id);
        coli_serve_command_dispose(&command); return matched;
    }
    if(command.kind==COLI_SERVE_COMMAND_STOP){
        coli_serve_command_dispose(&command); return 0;
    }
    if(command.kind==COLI_SERVE_COMMAND_SUBMIT){
        if (g_qn < SRV_QMAX) {
            SReq *q = &g_q[g_qn++];
            snprintf(q->id, sizeof(q->id), "%s", command.id);
            q->max_tok=command.max_tokens; q->temp=command.temperature;
            q->top_p=command.top_p; q->plen=(int)command.payload_bytes;
            q->alen=(int)command.extension_bytes;
            q->audio=coli_serve_command_extension(&command);
            q->payload=(char*)coli_serve_command_take_payload(&command);
        } else coli_serve_write_error(output,command.id,"queue full");
    }
    coli_serve_command_dispose(&command);
    return 0;
}

static int serve_one(Model *m, Tok *T, SReq *q) {
    Cfg *c = &m->c;
    int cap = q->plen + 16;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { coli_serve_write_error(stdout,q->id,"empty prompt"); free(ids); return 0; }
    const char *bad = prompt_reject(np, q->max_tok);
    if (bad) { coli_serve_write_error(stdout,q->id,bad); free(ids); return 0; }
    /* audio: every <|audio|> placeholder must have exactly one DMel frame */
    int naud = q->alen / m->c.mel_bins;
    if (q->alen % m->c.mel_bins != 0 || audio_tok_count(m, ids, np) != naud) {
        char message[256];
        snprintf(message,sizeof(message),
                 "audio frames (%d) do not match <|audio|> placeholders (%d)%s",
                 naud,audio_tok_count(m,ids,np),
                 m->audio_norm?"":" — snapshot has no audio tensors");
        coli_serve_write_error(stdout,q->id,message); free(ids); return 0;
    }
    /* KV PREFIX REUSE (#639 for GLM; this engine re-prefilled every turn).
     * A chat client resends the whole transcript each turn, so turn N used to
     * re-process turns 1..N-1 from scratch — the cost of a message grew with
     * the conversation, and every replayed position pulled its experts off
     * disk again. When this prompt begins with the sequence the state already
     * holds, that state IS the state at that position: keep it and prefill
     * only the tail.
     *
     * Requirements, all necessary:
     *   - kv_alloc must not have grown (it frees the K/V fed[] describes), so
     *     the reuse decision is taken AFTER it
     *   - at least one new token, since the state cannot be rewound
     *   - no audio on either side: every audio frame carries the same token id,
     *     so ids alone cannot tell two different clips apart
     * Either the reused positions are token-identical or nothing is reused;
     * the emitted tokens are unchanged in both cases. */
    kv_alloc(m, np + q->max_tok + 8);
    /* naud>0 taints this turn before the comparison, not after: a request that
     * brings its own audio must not match a text-only state either. */
    if (naud > 0) kv_prefix_taint(&m->kvp);
    int reuse = kv_prefix_reuse(&m->kvp, ids, np);
    if (getenv("INK_PREFIX_LOG")) {
        /* Report the decision either way, with the reason when it is no. "It
         * did not get faster" is otherwise indistinguishable from "reuse is not
         * wired up", both for a user and for the CI gate. */
        if (reuse)
            fprintf(stderr, "[PREFIX] reusing %d of %d prompt tokens (%.0f%%)\n",
                    reuse, np, 100.0 * reuse / np);
        else
            fprintf(stderr, "[PREFIX] no reuse: held=%d cap=%d prompt=%d%s%s\n",
                    m->kvp.len, m->kvp.cap, np,
                    m->kvp.tainted ? " tainted" : "",
                    (m->kvp.len > 0 && m->kvp.len < np) ? " (diverged)" : "");
        fflush(stderr);
    }
    if (!reuse) state_reset(m);
    double t0 = now_s();
    uint64_t h0 = m->hits, m0 = m->miss;
    /* per-turn phase snapshot for the PROF line (timers accumulate globally) */
    double f0 = m->t_fill, e0 = m->t_expert, s0 = m->t_shared, a0 = m->t_attn;
    /* `reuse` is the ABSOLUTE position of the first fresh token: attention and
     * the KV slots are position-indexed, so this has to be the real offset. */
    float *logit = step_mm(m, ids + reuse, np - reuse, reuse, NULL, q->audio, naud);
    int len = np, gen = 0, limited = 1, cancelled = 0;
    char buf[512];
    /* repetition-penalty history: prompt tail + emitted tokens, ring of 128 */
    float rep = getenv("REP_PEN") ? atof(getenv("REP_PEN")) : 1.1f;
    int hist[128], nhist = 0;
    for (int i = (np > 128 ? np - 128 : 0); i < np; i++) hist[nhist++] = ids[i];
    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        apply_rep_penalty(logit, c->unpad_vocab, hist, nhist, rep);
        int tk = sample_logits(logit, c->unpad_vocab, q->temp, q->top_p);
        free(logit); logit = NULL;
        if (tk == c->eos) { limited = 0; break; }
        if (nhist < 128) hist[nhist++] = tk;
        else { memmove(hist, hist+1, 127*sizeof(int)); hist[127] = tk; }
        int nb = tok_decode(T, &tk, 1, buf, sizeof(buf)-1);
        coli_serve_write_data(stdout,q->id,buf,(size_t)nb);
        gen++; len++;
        while (stdin_readable()) {
            int r = serve_read_cmd(stdin, stdout, q->id);
            if (r < 0) { free(ids); return -1; }
            if (r > 0) { cancelled = 1; limited = 0; }
        }
        if (cancelled || s == q->max_tok - 1) break;
        logit = step(m, &tk, 1, len - 1, NULL);
    }
    free(logit);
    double dt = now_s() - t0;
    double tot = (double)(m->hits - h0 + m->miss - m0);
    ColiServeDone done={gen,dt>0?gen/dt:0.0,
                        tot?100.0*(m->hits-h0)/tot:0.0,rss_gb(),np,limited};
    char done_line[256];
    int done_bytes=coli_serve_format_done(done_line,sizeof(done_line),q->id,&done);
    if(done_bytes>0) fwrite(done_line,1,(size_t)done_bytes,stdout);
    /* PROF: per-turn phase timings for the dashboard (gateway schema — we map
     * expert_wait -> shared-expert compute, lm_head folded into 0). */
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %d\n", dt, np, gen,
           m->t_fill - f0, m->t_shared - s0, m->t_expert - e0, m->t_attn - a0, 0.0, gen + 1);
    fflush(stdout);
    free(ids);
    return 0;
}

/* ---------- dashboard protocol (HWINFO / TIERS / EMAP) ----------
 * Same stdout lines colibri.c emits for the web dashboard; the gateway parses
 * them and the Brain/Profiling pages render live expert-tier state. */
static void serve_hwinfo(Model *m) {
    char cpu[256] = ""; int cores = 0; double rt = 0, ra = 0;
    FILE *ci = fopen("/proc/cpuinfo", "r");
    if (ci) { char ln[256];
        while (fgets(ln, sizeof(ln), ci)) if (!strncmp(ln, "model name", 10)) {
            char *p = strchr(ln, ':'); if (p) { p++; while (*p == ' ') p++;
            int n = (int)strlen(p); if (n > 0 && p[n-1] == '\n') p[--n] = 0;
            snprintf(cpu, sizeof(cpu), "%s", p); } break; }
        fclose(ci); }
#ifdef _SC_NPROCESSORS_ONLN
    cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    FILE *mi = fopen("/proc/meminfo", "r");
    if (mi) { char ln[256]; double v = 0;
        while (fgets(ln, sizeof(ln), mi)) {
            if (sscanf(ln, "MemTotal: %lf", &v) == 1) rt = v/1e6;
            if (sscanf(ln, "MemAvailable: %lf", &v) == 1) ra = v/1e6;
        } fclose(mi); }
    int ngpu = 0; double vram = 0;
    const char *gpu = "";
#ifdef COLI_CUDA
    if (g_cuda) { ngpu = 1; vram = ink_cuda_free_bytes()/1e9; gpu = "CUDA device"; }
#endif
    (void)m;
    printf("HWINFO %d %.1f %.1f %d %.1f %s|%s\n", cores, rt, ra, ngpu, vram, cpu[0]?cpu:"unknown", gpu);
    fflush(stdout);
}

static void serve_tiers_emap(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts;
    int nsp = 0, filled = 0;
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) { nsp++; filled += m->cache[i].n; }
    int64_t I = c->moe_inter, D = c->hidden;
    int64_t slotb = m->xq ? m->rb13*2*I + m->rb2*D + (2*I+D)*4
                  : m->quant_bits ? 3*I*D + (2*I+D)*4 : 3*I*D*4;
    printf("TIERS 0 %d %d 0.00 %.2f\n", filled, nsp*E - filled, filled*(double)slotb/1e9);
    /* EMAP: 1 byte/expert hex — tier(2b: 0=disk 1=RAM)<<6 | heat(6b: log2 usage) */
    char *hex = malloc((size_t)nsp*E*2 + 1); int w = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (!c->sparse[i]) continue;
        for (int e = 0; e < E; e++) {
            Slot *resident = slot_indexed(m, i, e);
            int tier = resident && resident->filled;
            uint32_t u = m->eusage[i] ? m->eusage[i][e] : 0;
            int heat = 0; while (u) { heat++; u >>= 1; } if (heat > 63) heat = 63;
            int b = (tier << 6) | heat;
            hex[w++] = "0123456789abcdef"[b >> 4];
            hex[w++] = "0123456789abcdef"[b & 15];
        }
    }
    hex[w] = 0;
    printf("EMAP %d %d %s\n", nsp, E, hex);
    fflush(stdout); free(hex);
}

static void serve_loop(Model *m, Tok *T) {
    /* Before the sentinel: on Windows a TEXT-mode stdout rewrites the trailing \n
     * as \r\n, the gateway never matches it and waits forever (#748). Lives in
     * compat.h because colibri.c has had it since #195 and this engine was
     * written without it. */
    coli_serve_stdio_init();
    const char *sd = getenv("SEED");
    if (sd) g_rng ^= (uint64_t)strtoull(sd, NULL, 10);
    else g_rng ^= (uint64_t)time(NULL) * 2654435761u;
    /* the gateway reads a STAT line right after the READY sentinel (colibri
     * reports its load stats there) — match the handshake */
    coli_serve_write_ready(stdout,rss_gb());
    serve_hwinfo(m);
    serve_tiers_emap(m);
    for (;;) {
        while (!g_qn) if (serve_read_cmd(stdin, stdout, NULL) < 0) return;   /* blocks on stdin */
        SReq q = g_q[0];
        memmove(g_q, g_q+1, (size_t)(--g_qn) * sizeof(SReq));
        int fatal=serve_one(m,T,&q);
        serve_tiers_emap(m);
        serve_request_dispose(&q);
        if(fatal<0){ while(g_qn) serve_request_dispose(&g_q[--g_qn]); return; }
    }
}

/* ---------- ref_inkling.json harness ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { *n_out = 0; return NULL; }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

#ifndef INKLING_NO_MAIN
int main(int argc, char **argv) {
    /* OpenMP hot-thread tuning, same trick (and rationale) as glm.c: the
     * per-expert matmul regions are tiny and back-to-back; the default passive
     * wait policy parks the team between regions and re-wake latency dominates.
     * libgomp reads OMP_/GOMP_ vars before main(), so seed them and re-exec
     * once (COLI_OMP_TUNED guards the exec; COLI_NO_OMP_TUNE=1 disables).
     * NOT under CUDA — same exception glm.c makes: a spinning 24-thread team
     * starves the CUDA driver during every stream sync. */
#if !defined(COLI_CUDA) && !defined(__APPLE__)
    /* NOT on Apple Silicon: the active-spin team steals the shared SoC power
     * budget (same mechanism as the M5 Max Metal report) and measured strictly
     * worse on Inkling-Small even for pure-CPU decode — 0.50 vs 0.84 tok/s,
     * and 2x the prefill. Opt back in with COLI_OMP_TUNED=0 unset + Linux. */
    if (!getenv("COLI_OMP_TUNED") && !getenv("COLI_NO_OMP_TUNE")) {
        setenv("OMP_WAIT_POLICY","active",0);
        setenv("GOMP_SPINCOUNT","200000",0);
        setenv("OMP_PROC_BIND","close",0);
        setenv("OMP_DYNAMIC","FALSE",0);
        setenv("COLI_OMP_TUNED","1",1);
#ifdef __linux__
        execv("/proc/self/exe", argv);
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
#endif  /* !COLI_CUDA && !__APPLE__ */
    coli_omp_tune_threads("inkling");
    const char *snap = getenv("SNAP");
    if (!snap) { coli_print_launcher_help("Inkling"); return 1; }
    g_topp = getenv("TOPP") ? (float)atof(getenv("TOPP")) : 0.f;
    if (g_topp > 0.f && g_topp < 1.f)
        fprintf(stderr, "[TOPP] %.2f: routed experts kept to cumulative weight — "
                "fewer experts read per token, but the routing is TRIMMED "
                "(quality lever, A/B it before trusting the speed-up)\n", g_topp);
    else if (g_topp != 0.f) {
        fprintf(stderr, "TOPP must be in (0,1); %.3f ignored\n", g_topp); g_topp = 0.f;
    }
    /* flags: -p "prompt" [-n N] -> generate mode; positional: [cap] [bits] [ref.json] */
    const char *prompt = NULL, *pfile = NULL, *refpath = "ref_inkling.json";
    const char *audiopath = NULL;
    int cap = -1, bits = 0, n_new = 256, npos = 0, chat = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-f") && i+1 < argc) pfile = argv[++i];
        else if (!strcmp(argv[i], "-n") && i+1 < argc) n_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "--audio") && i+1 < argc) audiopath = argv[++i];
        else if (npos == 0) { cap = atoi(argv[i]); npos++; }
        else if (npos == 1) { bits = atoi(argv[i]); npos++; }
        else refpath = argv[i];
    }
    /* --audio <file>: raw u8 DMel frames, [n_frames, mel_bins] row-major —
     * what tinkernel-audio / the gateway DSP emit. Implies --chat (the audio
     * message needs TMLv0 framing to be in distribution). */
    uint8_t *dmel = NULL; long dmel_bytes = 0;
    if (audiopath) {
        FILE *af = fopen(audiopath, "rb");
        if (!af) { perror(audiopath); return 1; }
        fseek(af, 0, SEEK_END); dmel_bytes = ftell(af); fseek(af, 0, SEEK_SET);
        dmel = malloc((size_t)dmel_bytes);
        if (fread(dmel, 1, (size_t)dmel_bytes, af) != (size_t)dmel_bytes) {
            fprintf(stderr, "short read on %s\n", audiopath); return 1; }
        fclose(af);
        chat = 1;
        if (!prompt) prompt = "";
    }
    /* --chat: avvolge il prompt nel template di Inkling (sottoinsieme testuale di
     * chat_template.jinja, lo stesso che openai_server.py rende in render_chat_inkling):
     * token di ruolo + <|content_text|>, <|end_message|> a chiudere, il livello di
     * thinking come messaggio di sistema, e <|message_model|> come prompt di
     * generazione. Senza template un modello instruct riceve testo fuori
     * distribuzione. THINK=<0..1> alza lo sforzo di ragionamento (default 0). */
    char *chat_buf = NULL;
    if (chat && prompt && !dmel) {
        const char *eff = getenv("THINK") ? getenv("THINK") : "0";
        size_t need = strlen(prompt) + strlen(eff) + 256;
        chat_buf = malloc(need);
        if (!chat_buf) { fprintf(stderr, "OOM chat template\n"); return 1; }
        snprintf(chat_buf, need,
                 "<|message_user|><|content_text|>%s<|end_message|>"
                 "<|message_system|><|content_text|>Thinking effort level: %s<|end_message|>"
                 "<|message_model|>", prompt, eff);
        prompt = chat_buf;
    } else if (chat && dmel) {
        /* audio turn, TMLv0 framing (conformant with tml-renderers 0.1.0):
         * effort system message, optional text user message, then the audio
         * message — <|content_audio_input|>, one <|audio|> per frame,
         * <|audio_end|> — and <|message_model|> to hand the turn over.
         * Frame count is provisional here (mel_bins is read from config.json
         * at model_init); recomputed below once the model is loaded. */
        const char *eff = getenv("THINK") ? getenv("THINK") : "0";
        long nf_guess = dmel_bytes / 80;           /* placeholder emission only */
        size_t need = strlen(prompt) + strlen(eff) + (size_t)nf_guess*10 + 320;
        chat_buf = malloc(need);
        if (!chat_buf) { fprintf(stderr, "OOM chat template\n"); return 1; }
        char *w = chat_buf;
        w += sprintf(w, "<|message_system|><|content_text|>Thinking effort level: %s<|end_message|>", eff);
        if (prompt[0])
            w += sprintf(w, "<|message_user|><|content_text|>%s<|end_message|>", prompt);
        w += sprintf(w, "<|message_user|><|content_audio_input|>");
        for (long i = 0; i < nf_guess; i++) w += sprintf(w, "<|audio|>");
        sprintf(w, "<|audio_end|><|end_message|><|message_model|>");
        prompt = chat_buf;
    }
    if (cap < 0) cap = (prompt || pfile) ? 0 : 16;   /* generate mode defaults to RAM-sized auto cap */
    if (bits && (bits < 2 || bits > 8)) { fprintf(stderr, "quant_bits must be 0 (f32) or 2..8\n"); return 1; }

    /* SERVE=1: the openai_server.py gateway drives the engine over stdin/stdout
     * (READY handshake, SUBMIT/CANCEL, DATA/DONE frames) — same protocol colibri. */
    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        Model m; model_init(&m, snap, cap, bits);
        pins_load(&m, snap);
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok T; tok_load(&T, tkp);
        serve_loop(&m, &T);
        usage_save(&m, snap);
        return 0;
    }

    if (prompt || pfile) {
        Model m; model_init(&m, snap, cap, bits);
        printf("== Inkling C engine, %d layers, experts @ %s, cache %d/layer ==\n",
               m.c.n_layers, m.xq ? "container" : bits ? "int" : "f32", m.cache[0].cap);
        pins_load(&m, snap);
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok T; tok_load(&T, tkp);
        if (prompt) {
            int naud = dmel ? (int)(dmel_bytes / m.c.mel_bins) : 0;
            if (dmel && dmel_bytes % m.c.mel_bins != 0) {
                fprintf(stderr, "%s: %ld bytes is not a multiple of mel_bins=%d\n",
                        audiopath, dmel_bytes, m.c.mel_bins); return 1; }
            generate_stream(&m, &T, prompt, n_new, dmel, naud);
        }
        else {   /* -f: one prompt per line, model loaded once, usage accumulates */
            FILE *pf = fopen(pfile, "rb"); if (!pf) { perror(pfile); return 1; }
            char ln[8192]; int np = 0;
            while (fgets(ln, sizeof(ln), pf)) {
                size_t n = strlen(ln); while (n && (ln[n-1]=='\n'||ln[n-1]=='\r')) ln[--n]=0;
                if (!n || ln[0]=='#') continue;
                printf("\n===== prompt %d =====\n", ++np);
                state_reset(&m);
                generate_stream(&m, &T, ln, n_new, NULL, 0);
            }
            fclose(pf);
        }
        int saved = usage_save(&m, snap);
        double tot = m.hits + m.miss;
        printf("[cache] hit %.1f%% (%llu hit / %llu load)%s\n",
               tot ? 100.0*m.hits/tot : 0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss,
               saved ? " | usage history saved" : "");
        /* quanto ha tagliato TOPP, in modo che la leva sia misurabile e non creduta */
        if (g_topp > 0.f && m.ereq)
            printf("[topp] %.2f: %llu/%llu routed used (%.1f%% trimmed, "
                   "%.2f experts/token avg)\n", g_topp,
                   (unsigned long long)m.euse, (unsigned long long)m.ereq,
                   100.0*(double)(m.ereq-m.euse)/(double)m.ereq,
                   (double)m.euse/((double)m.ereq/(double)m.c.topk));
        return 0;
    }

    FILE *f = fopen(refpath, "rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull, ntf, ndm;
    int *pids  = read_int_array(ref,"prompt_ids",&np);
    int *full  = read_int_array(ref,"full_ids",&nfull);
    int *tfref = read_int_array(ref,"tf_pred",&ntf);
    /* optional audio oracle: "dmel" = flattened [n_frames, mel_bins] levels */
    int *dmint = read_int_array(ref,"dmel",&ndm);
    int ngen = nfull - np;

    Model m; model_init(&m, snap, cap, bits);
    printf("== Inkling C engine (Stage A), cache = %d experts/layer, experts @ %s ==\n",
           cap, m.xq ? "container (int4/int8 + .qs)" : bits ? "int (runtime quant)" : "f32");
    printf("cfg: D=%d L=%d V=%d(%d) heads=%d/%d kv=%d/%d hd=%d win=%d d_rel=%d ext=%d E=%d+%d topk=%d\n",
           m.c.hidden, m.c.n_layers, m.c.vocab, m.c.unpad_vocab, m.c.n_heads, m.c.swa_heads,
           m.c.n_kv, m.c.swa_kv, m.c.head_dim, m.c.window, m.c.d_rel, m.c.rel_extent,
           m.c.n_experts, m.c.n_shared, m.c.topk);
    printf("resident weights loaded in %.1fs | RSS: %.2f GB\n", m.dense_load_s, rss_gb());
    kv_alloc(&m, nfull + 8);

    uint8_t *rdmel = NULL; int rnaud = 0;
    if (dmint && ndm > 0) {
        if (ndm % m.c.mel_bins != 0) { fprintf(stderr, "dmel len %d not a multiple of mel_bins %d\n", ndm, m.c.mel_bins); return 1; }
        rnaud = ndm / m.c.mel_bins;
        rdmel = malloc((size_t)ndm);
        for (int i = 0; i < ndm; i++) rdmel[i] = (uint8_t)dmint[i];
        printf("audio oracle: %d DMel frames x %d bins\n", rnaud, m.c.mel_bins);
    }

    /* pass 1: teacher-forced argmax over the full reference sequence */
    if (tfref && ntf == nfull) {
        int *tf = malloc(nfull * sizeof(int));
        float *lg = step_mm(&m, full, nfull, 0, tf, rdmel, rnaud);
        free(lg);
        int ok = 0; for (int i = 0; i < nfull; i++) ok += (tf[i] == tfref[i]);
        printf("teacher-forced argmax: %d/%d match\n", ok, nfull);
        free(tf);
        state_reset(&m);
    }

    /* pass 2: greedy generation, token-for-token vs the oracle */
    int *out = malloc(nfull * sizeof(int));
    double t = now_s();
    generate(&m, pids, np, ngen, out, rdmel, rnaud);
    double dt = now_s() - t;
    int match = 0;
    printf("Reference: "); for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : "); for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, ngen);
    double tot = m.hits + m.miss;
#ifdef COLI_METAL
    if (g_metal) {
        uint64_t mok = 0, mfb = 0, mex = 0;
        coli_metal_moe_counts(&mok, &mfb, &mex);
        printf("[metal] %llu MoE blocks on GPU, %llu CPU fallbacks, %llu experts\n",
               (unsigned long long)mok, (unsigned long long)mfb, (unsigned long long)mex);
    }
#endif
    printf("PEAK RSS: %.2f GB | expert cache hit %.1f%% | %.2f tok/s\n",
           rss_gb(), tot?100.0*m.hits/tot:0.0, ngen/dt);
    free(buf); free(arena);
    return (match == ngen) ? 0 : 1;
}
#endif /* INKLING_NO_MAIN */

#ifdef COLI_SEGMENT_ADAPTER
/* ---------- engine-owned Segment adapter ------------------------------ */

typedef struct {
    Model model;
    uint32_t layer_begin, layer_end, context_tokens;
    pthread_mutex_t run_lock;
} InklingSegmentEngine;

typedef struct {
    InklingSegmentEngine *engine;
    float **K, **V;
    float **cs[4];
    uint32_t context_tokens, position;
} InklingSegmentSession;

static void inkling_segment_wt_destroy(Wt *weight) {
    if (!weight) return;
    free(weight->f); free(weight->h);
    free(weight->q4); free(weight->qs);
    memset(weight, 0, sizeof(*weight));
}

static void inkling_segment_layer_destroy(Layer *layer) {
    if (!layer) return;
    free(layer->in_ln); free(layer->post_ln);
    inkling_segment_wt_destroy(&layer->q);
    inkling_segment_wt_destroy(&layer->k);
    inkling_segment_wt_destroy(&layer->v);
    inkling_segment_wt_destroy(&layer->r);
    inkling_segment_wt_destroy(&layer->o);
    free(layer->qn); free(layer->kn); free(layer->relp);
    free(layer->k_cw); free(layer->v_cw);
    free(layer->a_cw); free(layer->m_cw);
    inkling_segment_wt_destroy(&layer->dg);
    inkling_segment_wt_destroy(&layer->du);
    inkling_segment_wt_destroy(&layer->dd);
    free(layer->router); free(layer->rbias);
    inkling_segment_wt_destroy(&layer->sh_g);
    inkling_segment_wt_destroy(&layer->sh_u);
    inkling_segment_wt_destroy(&layer->sh_d);
}

static void inkling_segment_model_destroy(InklingSegmentEngine *engine) {
    if (!engine) return;
    Model *model = &engine->model;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        inkling_segment_layer_destroy(&model->L[layer]);
        LCache *cache = &model->cache[layer];
        if (model->xq && cache->cap > 0 && cache->slots) {
            free(cache->slots[0].p13);
            free(cache->slots[0].s13);
        } else if (cache->slots) {
            for (int slot = 0; slot < cache->cap; slot++) {
                free(cache->slots[slot].q13); free(cache->slots[slot].q2);
                free(cache->slots[slot].s13); free(cache->slots[slot].s2);
                free(cache->slots[slot].f13); free(cache->slots[slot].f2);
            }
        }
        free(cache->slot_by_expert); free(cache->slots);
    }
    inkling_segment_wt_destroy(&model->embed);
    inkling_segment_wt_destroy(&model->lm_head);
    inkling_segment_wt_destroy(&model->audio_enc);
    free(model->embed_norm); free(model->final_norm); free(model->audio_norm);
    free(model->K); free(model->V);
    for (int state = 0; state < 4; state++) free(model->cs[state]);
    free(model->cache); free(model->L);
    kv_prefix_free(&model->kvp);
    st_destroy(&model->Sq); st_destroy(&model->S);
}

static int inkling_segment_engine_open(
    void **engine_impl, ColiSegmentCapabilities *capabilities,
    const ColiSegmentEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid Inkling Segment open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_SEGMENT_CAP_CPU))
        return coli_segment_adapter_error(error, error_size,
                                           "Inkling Segment currently supports CPU");

    Cfg config;
    memset(&config, 0, sizeof(config));
    load_cfg(&config, options->model_dir);
    if (options->layer_end > (uint32_t)config.n_layers)
        return coli_segment_adapter_error(error, error_size,
                                           "Inkling Segment range exceeds model");
    int bits = getenv("INK_SEGMENT_BITS")
        ? atoi(getenv("INK_SEGMENT_BITS")) : 0;
    if (bits && (bits < 2 || bits > 8))
        return coli_segment_adapter_error(error, error_size,
                                           "INK_SEGMENT_BITS must be 0 or 2..8");
    int sparse_layers = 0;
    for (uint32_t layer = options->layer_begin; layer < options->layer_end;
         layer++) sparse_layers += config.sparse[layer] != 0;
    int cap = 0;
    if (options->memory_limit_bytes && sparse_layers) {
        uint64_t slot_bytes = bits
            ? 3u * (uint64_t)config.moe_inter * config.hidden +
              (uint64_t)(2 * config.moe_inter + config.hidden) * sizeof(float)
            : 3u * (uint64_t)config.moe_inter * config.hidden * sizeof(float);
        uint64_t slots = slot_bytes
            ? options->memory_limit_bytes / slot_bytes /
              (uint64_t)sparse_layers : 0;
        cap = slots > (uint64_t)config.n_experts
            ? config.n_experts : (int)slots;
        if (cap < 1) cap = 1;
    }

    InklingSegmentEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory opening Inkling Segment");
    engine->layer_begin = options->layer_begin;
    engine->layer_end = options->layer_end;
    engine->context_tokens = options->context_tokens;
    if (pthread_mutex_init(&engine->run_lock, NULL)) {
        free(engine);
        return coli_segment_adapter_error(error, error_size,
                                           "cannot initialize Inkling Segment lock");
    }
    model_init_range(&engine->model, options->model_dir, cap, bits,
                     (int)options->layer_begin, (int)options->layer_end,
                     0, 0, 0);
    for (int state = 0; state < 4; state++) {
        free(engine->model.cs[state]);
        engine->model.cs[state] = NULL;
    }

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_SNAPSHOT |
                          COLI_SEGMENT_CAP_RANGE_NATIVE |
                          COLI_SEGMENT_CAP_MULTI_SESSION |
                          COLI_SEGMENT_CAP_CPU;
    coli_segment_capability_string(capabilities->engine_id,
                                   sizeof(capabilities->engine_id), "inkling");
    coli_segment_capability_string(capabilities->state_schema,
                                   sizeof(capabilities->state_schema),
                                   "inkling/kv-ring-conv-f32-v1");
    snprintf(capabilities->numeric_class,
             sizeof(capabilities->numeric_class),
             "inkling/expert-q%d/f32/cpu-v1", bits);
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = (uint32_t)config.hidden;
    capabilities->max_batch_rows = 128;
    capabilities->max_context_tokens = options->context_tokens;
    capabilities->num_layers = (uint32_t)config.n_layers;
    *engine_impl = engine;
    return 0;
}

static void inkling_segment_engine_destroy(void *engine_impl) {
    InklingSegmentEngine *engine = (InklingSegmentEngine *)engine_impl;
    if (!engine) return;
    inkling_segment_model_destroy(engine);
    pthread_mutex_destroy(&engine->run_lock);
    free(engine);
}

static void inkling_segment_session_free(InklingSegmentSession *session) {
    if (!session) return;
    if (session->engine) {
        for (uint32_t layer = session->engine->layer_begin;
             layer < session->engine->layer_end; layer++) {
            free(session->K ? session->K[layer] : NULL);
            free(session->V ? session->V[layer] : NULL);
            for (int state = 0; state < 4; state++)
                free(session->cs[state] ? session->cs[state][layer] : NULL);
        }
    }
    free(session->K); free(session->V);
    for (int state = 0; state < 4; state++) free(session->cs[state]);
    free(session);
}

static int inkling_segment_session_create(
    void *engine_impl, void **session_impl,
    const ColiSegmentSessionOptions *options, char *error, size_t error_size) {
    InklingSegmentEngine *engine = (InklingSegmentEngine *)engine_impl;
    if (!engine || !session_impl || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid Inkling Segment session");
    *session_impl = NULL;
    InklingSegmentSession *session = calloc(1, sizeof(*session));
    if (!session)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory creating Inkling session");
    session->engine = engine;
    session->context_tokens = options->context_tokens;
    size_t layers = (size_t)engine->model.c.n_layers;
    session->K = calloc(layers, sizeof(*session->K));
    session->V = calloc(layers, sizeof(*session->V));
    for (int state = 0; state < 4; state++)
        session->cs[state] = calloc(layers, sizeof(*session->cs[state]));
    if (!session->K || !session->V || !session->cs[0] || !session->cs[1] ||
        !session->cs[2] || !session->cs[3]) goto oom;

    Cfg *config = &engine->model.c;
    uint64_t state_bytes = 0;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        size_t cells;
        int kvdim = L_KV(config, layer) * L_HD(config, layer);
        int rows = kv_ring_rows(config, (int)layer,
                                (int)options->context_tokens);
        if (coli_segment_size_mul((size_t)L_KV(config, layer),
                                  (size_t)rows, &cells) ||
            coli_segment_size_mul(cells, (size_t)L_HD(config, layer),
                                  &cells)) goto oom;
        session->K[layer] = calloc(cells, sizeof(float));
        session->V[layer] = calloc(cells, sizeof(float));
        state_bytes += 2u * (uint64_t)cells * sizeof(float);
        for (int state = 0; state < 4; state++) {
            int width = state < 2 ? kvdim : config->hidden;
            if (coli_segment_size_mul((size_t)width,
                                      (size_t)(config->conv_k - 1),
                                      &cells)) goto oom;
            session->cs[state][layer] = calloc(cells, sizeof(float));
            state_bytes += (uint64_t)cells * sizeof(float);
        }
        if (!session->K[layer] || !session->V[layer] ||
            !session->cs[0][layer] || !session->cs[1][layer] ||
            !session->cs[2][layer] || !session->cs[3][layer]) goto oom;
    }
    if (options->memory_limit_bytes &&
        state_bytes > options->memory_limit_bytes) {
        inkling_segment_session_free(session);
        return coli_segment_adapter_error(error, error_size,
                                           "Inkling session state exceeds memory limit");
    }
    *session_impl = session;
    return 0;

oom:
    inkling_segment_session_free(session);
    return coli_segment_adapter_error(error, error_size,
                                       "out of memory allocating Inkling state");
}

static void inkling_segment_session_destroy(void *session_impl) {
    inkling_segment_session_free((InklingSegmentSession *)session_impl);
}

static int inkling_segment_session_run(void *session_impl,
                                       const ColiSegmentRunRequest *request,
                                       char *error, size_t error_size) {
    InklingSegmentSession *session = (InklingSegmentSession *)session_impl;
    if (!session || !request || request->position != session->position)
        return coli_segment_adapter_error(
            error, error_size, "Inkling Segment requires contiguous positions");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return coli_segment_adapter_error(error, error_size,
                                           "Inkling Segment run cancelled");
    InklingSegmentEngine *engine = session->engine;
    if (request->output != request->input)
        memcpy(request->output, request->input, request->input_bytes);
    pthread_mutex_lock(&engine->run_lock);
    Model *model = &engine->model;
    model->K = session->K; model->V = session->V;
    for (int state = 0; state < 4; state++) model->cs[state] = session->cs[state];
    model->max_t = (int)session->context_tokens;
    model->kv_len = (int)session->position;
    inkling_layers_forward_range(model, (float *)request->output,
                                  (int)request->rows, (int)request->position,
                                  (int)engine->layer_begin,
                                  (int)engine->layer_end);
    model->K = NULL; model->V = NULL;
    for (int state = 0; state < 4; state++) model->cs[state] = NULL;
    model->max_t = 0; model->kv_len = 0;
    pthread_mutex_unlock(&engine->run_lock);
    session->position += request->rows;
    return 0;
}

static int inkling_segment_spans(
    InklingSegmentSession *session, uint32_t position,
    ColiSegmentStateSpan **spans_output, size_t *count_output,
    char *error, size_t error_size) {
    Cfg *config = &session->engine->model.c;
    size_t capacity = 0;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++)
        capacity += (size_t)(2 * L_KV(config, layer) + 4);
    ColiSegmentStateSpan *spans = capacity
        ? calloc(capacity, sizeof(*spans)) : NULL;
    if (capacity && !spans)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory describing Inkling state");
    size_t count = 0;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++) {
        int kv = L_KV(config, layer), hd = L_HD(config, layer);
        int rows = kv_ring_rows(config, (int)layer,
                                (int)session->context_tokens);
        size_t live_rows = config->local[layer] ? (size_t)rows : position;
        size_t stride = (size_t)rows * hd;
        size_t row_bytes = live_rows * hd * sizeof(float);
        for (int which = 0; which < 2; which++) {
            float *state = which ? session->V[layer] : session->K[layer];
            for (int head = 0; head < kv; head++)
                spans[count++] = (ColiSegmentStateSpan){
                    state + (size_t)head * stride, row_bytes};
        }
        int kvdim = kv * hd;
        for (int state = 0; state < 4; state++) {
            int width = state < 2 ? kvdim : config->hidden;
            spans[count++] = (ColiSegmentStateSpan){
                session->cs[state][layer],
                (size_t)width * (config->conv_k - 1) * sizeof(float)};
        }
    }
    *spans_output = spans; *count_output = count;
    return 0;
}

static int inkling_segment_session_snapshot(
    void *session_impl, ColiSegmentWriteFn write_fn, void *write_user_data,
    char *error, size_t error_size) {
    InklingSegmentSession *session = (InklingSegmentSession *)session_impl;
    ColiSegmentStateSpan *spans = NULL;
    size_t count = 0, payload_bytes;
    if (!session || inkling_segment_spans(session, session->position, &spans,
                                          &count, error, error_size))
        return -1;
    if (coli_segment_spans_size(spans, count, &payload_bytes)) {
        free(spans);
        return coli_segment_adapter_error(error, error_size,
                                           "Inkling snapshot size overflow");
    }
    ColiSegmentSnapshotHeader header;
    coli_segment_snapshot_header_init(
        &header, "inkling", session->engine->layer_begin,
        session->engine->layer_end, session->context_tokens, session->position,
        payload_bytes, coli_segment_spans_hash(spans, count));
    int result = coli_segment_stream_write(
        write_fn, write_user_data, &header, sizeof(header), error, error_size);
    if (!result)
        result = coli_segment_spans_write(spans, count, write_fn,
                                          write_user_data, error, error_size);
    free(spans);
    return result;
}

static int inkling_segment_session_restore(
    void *session_impl, ColiSegmentReadFn read_fn, void *read_user_data,
    char *error, size_t error_size) {
    InklingSegmentSession *session = (InklingSegmentSession *)session_impl;
    ColiSegmentSnapshotHeader header;
    if (!session || coli_segment_stream_read(read_fn, read_user_data, &header,
                                             sizeof(header), error, error_size))
        return -1;
    ColiSegmentStateSpan *spans = NULL;
    size_t count = 0, payload_bytes;
    if (inkling_segment_spans(session, header.position, &spans, &count,
                              error, error_size)) return -1;
    if (coli_segment_spans_size(spans, count, &payload_bytes) ||
        coli_segment_snapshot_header_valid(
            &header, "inkling", session->engine->layer_begin,
            session->engine->layer_end, session->context_tokens,
            payload_bytes, error, error_size)) {
        free(spans); return -1;
    }
    int result = coli_segment_spans_restore(
        spans, count, header.payload_hash, read_fn, read_user_data,
        error, error_size);
    free(spans);
    if (!result) session->position = header.position;
    return result;
}

static const ColiSegmentAdapter inkling_segment_adapter = {
    sizeof(ColiSegmentAdapter), COLI_SEGMENT_ABI_VERSION, "inkling",
    inkling_segment_engine_open, inkling_segment_engine_destroy,
    inkling_segment_session_create, inkling_segment_session_destroy,
    inkling_segment_session_run, inkling_segment_session_snapshot,
    inkling_segment_session_restore, {0}
};

int coli_inkling_segment_adapter_register(void) {
    return coli_segment_adapter_register(&inkling_segment_adapter);
}
#endif /* COLI_SEGMENT_ADAPTER */

#ifdef COLI_EDGE_ADAPTER
/* ---------- engine-owned model Edge adapter --------------------------- */

typedef struct {
    Model model;
    Tok tokenizer;
} InklingEdgeEngine;

static void inkling_edge_wt_destroy(Wt *weight) {
    if (!weight) return;
    free(weight->f); free(weight->h); free(weight->q4); free(weight->qs);
    memset(weight, 0, sizeof(*weight));
}

static uint64_t inkling_edge_wt_bytes(const Wt *weight, uint64_t rows,
                                      uint64_t columns) {
    if (weight->f) return rows * columns * sizeof(float);
    if (weight->h) return rows * columns * sizeof(uint16_t);
    if (!weight->q4) return 0;
    uint64_t scales = weight->qbits == 4 && weight->gs > 0
        ? rows * ((columns + (uint64_t)weight->gs - 1u) /
                  (uint64_t)weight->gs)
        : rows;
    return (uint64_t)weight->qn + scales * sizeof(float);
}

static void inkling_edge_engine_destroy(void *engine_impl) {
    InklingEdgeEngine *engine = (InklingEdgeEngine *)engine_impl;
    if (!engine) return;
    inkling_edge_wt_destroy(&engine->model.embed);
    inkling_edge_wt_destroy(&engine->model.lm_head);
    free(engine->model.embed_norm); free(engine->model.final_norm);
    st_destroy(&engine->model.Sq); st_destroy(&engine->model.S);
    tok_free(&engine->tokenizer);
    free(engine);
}

static int inkling_edge_engine_open(
    void **engine_impl, ColiEdgeCapabilities *capabilities,
    const ColiEdgeEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return coli_edge_adapter_error(error, error_size,
                                       "invalid Inkling Edge open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_EDGE_CAP_CPU))
        return coli_edge_adapter_error(error, error_size,
                                       "Inkling Edge supports CPU only");
    InklingEdgeEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory opening Inkling Edge");
    Model *model = &engine->model;
    load_cfg(&model->c, options->model_dir);
    st_init(&model->S, options->model_dir);
    char quantized_dir[4096];
    snprintf(quantized_dir, sizeof(quantized_dir), "%s/dense-int4g64",
             options->model_dir);
    const char *disable_quantized = getenv("INK_DENSE_Q4");
    struct stat quantized_stat;
    if (!(disable_quantized && *disable_quantized == '0') &&
        stat(quantized_dir, &quantized_stat) == 0 &&
        S_ISDIR(quantized_stat.st_mode)) {
        st_init(&model->Sq, quantized_dir);
        model->has_q = model->Sq.n > 0;
    }
    model->embed = load_w(model, "model.embed_tokens.weight", 0);
    model->embed_norm = st_has(&model->S, "model.embed_norm.weight")
        ? load_t(model, "model.embed_norm.weight") : NULL;
    model->final_norm = load_t(model, "model.norm.weight");
    model->lm_head = load_w(model, "lm_head.weight", 1);
    char tokenizer_path[4096];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             options->model_dir);
    tok_load(&engine->tokenizer, tokenizer_path);

    Cfg *config = &model->c;
    uint64_t resident = inkling_edge_wt_bytes(
        &model->embed, (uint64_t)config->vocab, (uint64_t)config->hidden);
    resident += inkling_edge_wt_bytes(
        &model->lm_head, (uint64_t)config->vocab, (uint64_t)config->hidden);
    resident += (uint64_t)config->hidden * sizeof(float);
    if (model->embed_norm)
        resident += (uint64_t)config->hidden * sizeof(float);
    if (options->memory_limit_bytes && resident > options->memory_limit_bytes) {
        inkling_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error, error_size,
                                       "Inkling Edge exceeds memory limit");
    }
    int bits = getenv("INK_SEGMENT_BITS")
        ? atoi(getenv("INK_SEGMENT_BITS")) : 0;
    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_EDGE_ABI_VERSION;
    capabilities->flags = COLI_EDGE_CAP_TOKENIZE |
                          COLI_EDGE_CAP_DETOKENIZE |
                          COLI_EDGE_CAP_GREEDY | COLI_EDGE_CAP_LOGITS |
                          COLI_EDGE_CAP_CPU;
    coli_edge_capability_string(capabilities->engine_id,
                                sizeof(capabilities->engine_id), "inkling");
    coli_edge_capability_string(capabilities->state_schema,
                                sizeof(capabilities->state_schema),
                                "inkling/kv-ring-conv-f32-v1");
    snprintf(capabilities->numeric_class,
             sizeof(capabilities->numeric_class),
             "inkling/expert-q%d/f32/cpu-v1", bits);
    coli_edge_capability_string(capabilities->tokenizer_class,
                                sizeof(capabilities->tokenizer_class),
                                "inkling/o200k-byte-bpe-v1");
    capabilities->state_dtype = COLI_EDGE_DTYPE_F32;
    capabilities->state_width = (uint32_t)config->hidden;
    capabilities->vocab_size = (uint32_t)config->unpad_vocab;
    capabilities->max_batch_rows = 128;
    capabilities->max_context_tokens = UINT32_MAX;
    capabilities->num_layers = (uint32_t)config->n_layers;
    capabilities->bos_token_id = -1;
    capabilities->eos_token_id = config->eos;
    capabilities->resident_bytes = resident;
    *engine_impl = engine;
    return 0;
}

static int inkling_edge_tokenize(
    void *engine_impl, const char *text, size_t text_bytes,
    int32_t *token_ids, size_t token_capacity, size_t *token_count,
    char *error, size_t error_size) {
    InklingEdgeEngine *engine = (InklingEdgeEngine *)engine_impl;
    return coli_edge_tok_tokenize(&engine->tokenizer, text, text_bytes,
                                  token_ids, token_capacity, token_count,
                                  error, error_size);
}

static int inkling_edge_detokenize(
    void *engine_impl, const int32_t *token_ids, size_t token_count,
    char *text, size_t text_capacity, size_t *text_bytes,
    char *error, size_t error_size) {
    InklingEdgeEngine *engine = (InklingEdgeEngine *)engine_impl;
    return coli_edge_tok_detokenize(&engine->tokenizer, token_ids, token_count,
                                    text, text_capacity, text_bytes,
                                    error, error_size);
}

static int inkling_edge_embed(void *engine_impl,
                              const ColiEdgeEmbedRequest *request,
                              char *error, size_t error_size) {
    InklingEdgeEngine *engine = (InklingEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *output = (float *)request->output;
    for (uint32_t row = 0; row < request->rows; row++) {
        int token = request->token_ids[row];
        if (token < 0 || token >= config->vocab)
            return coli_edge_adapter_error(error, error_size,
                                           "Inkling token ID is out of range");
        float *state = output + (size_t)row * config->hidden;
        wt_row_f32(engine->model.embed, (int64_t)token * config->hidden,
                   state, config->hidden);
        if (engine->model.embed_norm)
            rmsnorm_row(state, state, engine->model.embed_norm,
                        config->hidden, config->eps);
    }
    return 0;
}

static int inkling_edge_select(void *engine_impl,
                               const ColiEdgeSelectRequest *request,
                               char *error, size_t error_size) {
    InklingEdgeEngine *engine = (InklingEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    float *logits = falloc(config->unpad_vocab);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(logits); free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "Inkling Edge selection cancelled");
        }
        rmsnorm_row(normalized, input + (size_t)row * config->hidden,
                    engine->model.final_norm, config->hidden, config->eps);
        for (int item = 0; item < config->hidden; item++)
            normalized[item] /= config->mup;
        matmul_w(logits, normalized, engine->model.lm_head,
                 1, config->hidden, config->unpad_vocab);
        if (coli_edge_argmax(logits, (uint32_t)config->unpad_vocab,
                            &request->token_ids[row],
                            request->scores ? &request->scores[row] : NULL)) {
            free(logits); free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "Inkling Edge head failed");
        }
    }
    free(logits); free(normalized);
    return 0;
}

static int inkling_edge_logits(void *engine_impl,
                               const ColiEdgeLogitsRequest *request,
                               char *error, size_t error_size) {
    InklingEdgeEngine *engine = (InklingEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "Inkling Edge logits cancelled");
        }
        rmsnorm_row(normalized, input + (size_t)row * config->hidden,
                    engine->model.final_norm, config->hidden, config->eps);
        for (int item = 0; item < config->hidden; item++)
            normalized[item] /= config->mup;
        matmul_w(request->logits + (size_t)row * config->unpad_vocab,
                 normalized, engine->model.lm_head,
                 1, config->hidden, config->unpad_vocab);
    }
    free(normalized);
    return 0;
}

static const ColiEdgeAdapter inkling_edge_adapter = {
    sizeof(ColiEdgeAdapter), COLI_EDGE_ABI_VERSION, "inkling",
    inkling_edge_engine_open, inkling_edge_engine_destroy,
    inkling_edge_tokenize, inkling_edge_detokenize,
    inkling_edge_embed, inkling_edge_select, inkling_edge_logits, {0}
};

int coli_inkling_edge_adapter_register(void) {
    return coli_edge_adapter_register(&inkling_edge_adapter);
}
#endif /* COLI_EDGE_ADAPTER */
