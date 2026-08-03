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
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/select.h>                              /* serve-loop stdin poll (POSIX); inkling serves on Linux */
#endif
#include "st.h"
#include "tok.h"
#include "route_trace.h"                          /* shared routing telemetry (#700) */
#ifdef COLI_CUDA
#include "backend_cuda_ink.h"
static int g_cuda = 0;
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
typedef struct { Slot *slots; int n, cap; } LCache;

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
    Layer *L;
    LCache *cache;
    int64_t rb13, rb2;                    /* container row-bytes (0 = not container) */
    uint32_t **eusage;                    /* per-layer expert selection counts */
    int npin;                             /* pinned experts per sparse layer */
    uint64_t clock, hits, miss;
    uint64_t ereq, euse;                  /* routed richiesti (topk) vs usati dopo TOPP */
    double t_fill, t_expert, t_shared, t_attn, t_route;   /* phase timers */
    float **K, **V; int kv_len, max_t;    /* per-layer [kv][max_t][hd] */
    float **cs[4];                        /* conv states, [n_layers][C*(K-1)] */
    double dense_load_s;
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
            for (int s = 0; s < S; s++) {
                const uint16_t *xs = xh + (int64_t)s * I;
                __m512 acc = _mm512_setzero_ps();
                for (int i = 0; i < I; i += 32)
                    acc = _mm512_dpbf16_ps(acc, (__m512bh)_mm512_loadu_si512(xs + i),
                                                (__m512bh)_mm512_loadu_si512(w + i));
                y[(int64_t)s * O + o] = _mm512_reduce_add_ps(acc);
            }
        }
        free(xh);
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
static void matmul_i4g(float *y, const float *x, const uint8_t *p, const float *scale,
                       int S, int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
    int64_t rb = (I + 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = p + (int64_t)o * rb;
        const float *sc = scale + (int64_t)o * ng;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
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

static void matmul_w(float *y, const float *x, Wt W, int S, int I, int O) {
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
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
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
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
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

    /* attention layer types: explicit layer_types[] > local_layer_ids[] > (i+1)%6 rule */
    jval *lt = json_get(r,"layer_types");
    jval *ll = json_get(r,"local_layer_ids");
    for (int i = 0; i < c->n_layers; i++) {
        if (lt && lt->t == J_ARR) c->local[i] = (strcmp(lt->kids[i]->str,"hybrid_sliding")==0);
        else if (ll && ll->t == J_ARR) {
            c->local[i] = 0;
            for (int j = 0; j < ll->len; j++) if ((int)ll->kids[j]->num == i) { c->local[i] = 1; break; }
        } else c->local[i] = ((i + 1) % 6) != 0;
    }
    /* MLP types: explicit mlp_layer_types[] > dense_mlp_idx (first k layers dense) */
    jval *mt = json_get(r,"mlp_layer_types");
    int dense_idx = (int)jnum(r,"dense_mlp_idx",0);
    for (int i = 0; i < c->n_layers; i++) {
        if (mt && mt->t == J_ARR) c->sparse[i] = (strcmp(mt->kids[i]->str,"sparse")==0);
        else c->sparse[i] = (i >= dense_idx);
    }
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
    float v; st_read_f32(&m->S, name, &v, 0); return v;
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

static void model_init(Model *m, const char *snap, int cap, int bits) {
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
    m->embed      = load_w(m, "model.embed_tokens.weight", 0);
    m->embed_norm = st_has(&m->S,"model.embed_norm.weight") ? load_t(m,"model.embed_norm.weight") : NULL;
    m->final_norm = load_t(m, "model.norm.weight");
    m->lm_head    = load_w(m, "lm_head.weight", 1);
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[320];
    for (int i = 0; i < c->n_layers; i++) {
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
        }
        #undef LD
        #undef LDW
        /* conv states: raw inputs of the previous K-1 steps, zero-init */
        int kvdim = L_KV(c,i) * L_HD(c,i);
        for (int j = 0; j < 4; j++) {
            if (!m->cs[j]) m->cs[j] = calloc(c->n_layers, sizeof(float*));
            int C = (j < 2) ? kvdim : D;
            m->cs[j][i] = calloc((int64_t)C * (K-1), sizeof(float));
        }
    }
    /* container detection: converted snapshots store experts as U8 + .qs.
     * rb13/rb2 = bytes per packed row (D/2|D and I/2|I for int4|int8) */
    int64_t I = c->moe_inter, E = c->n_experts;
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) {
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
    int nsp = 0; for (int i = 0; i < c->n_layers; i++) nsp += c->sparse[i];
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
    for (int i = 0; i < c->n_layers; i++) { m->cache[i].cap = cap; m->cache[i].slots = calloc(cap, sizeof(Slot)); }
    /* usage counters; seeded from a previous run's history when present */
    rt_init("inkling", c->n_layers, E);
    for (int i = 0; i < c->n_layers; i++) if (!c->sparse[i]) rt_drop_row(i);
    rt_drop_row(c->n_layers);                     /* inkling has no MTP row */
    m->eusage = rt_counts_all();                  /* alias: the bump sites stay as they are */
    m->dense_load_s = now_s() - t0;
}

static double mem_avail_bytes(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char ln[256]; double kb = 0;
    while (fgets(ln, sizeof(ln), f)) if (sscanf(ln, "MemAvailable: %lf", &kb) == 1) break;
    fclose(f);
    return kb * 1024.0;
#else
    return 0;
#endif
}

/* ---------- routed-expert slots: serial bookkeeping, parallel fills ---------- */
static Slot *slot_find(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        lc->slots[i].used = ++m->clock;
        return &lc->slots[i];
    }
    return NULL;
}

/* allocate a slot (or evict the LRU non-pinned one); serial callers only */
static Slot *slot_acquire(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer]; Cfg *c = &m->c;
    int64_t D = c->hidden, I = c->moe_inter, n13 = 2*I*D, n2 = D*I;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        if (m->xq)              { s->p13 = malloc((size_t)(m->rb13*2*I)); s->p2 = malloc((size_t)(m->rb2*D));
                                  s->s13 = falloc(2*I); s->s2 = falloc(D);
                                  if (!s->p13 || !s->p2) { fprintf(stderr,"OOM expert slot\n"); exit(1); } }
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
    s->eid = eid; s->used = ++m->clock; s->filled = 0; s->pinned = 0;
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
                int taken = 0;
                for (int z = 0; z < r; z++) if (ps[np-r+z]->eid == e) { taken = 1; break; }
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
    const char *sv = getenv("USAGE_SAVE");
    if (sv && *sv == '0') return 0;
    if (env && (!strcmp(env, "off") || !strcmp(env, "0"))) return 0;
    if (env) snprintf(up, sizeof(up), "%s", env);
    else snprintf(up, sizeof(up), "%s/.coli_usage", snap);
    /* One format for every engine now: sparse text with the dimension and identity
     * header. The IKU1 block this replaced is still readable on load, so a history
     * written by an older inkling keeps working and is rewritten in the new form. */
    return rt_save(up, 1);
}

/* ---------- attention (GQA + sliding/global + relative bias + K/V sconv) ---------- */
static void attention(Model *m, Layer *l, int li, float *x, int S, int pos0, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, H = L_HEADS(c,li), KV = L_KV(c,li), hd = L_HD(c,li), ext = L_EXT(c,li);
    int local = c->local[li];
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
    /* append K,V to the cache */
    for (int s = 0; s < S; s++) for (int h = 0; h < KV; h++) {
        int t = pos0 + s;
        memcpy(m->K[li] + ((int64_t)h*m->max_t + t)*hd, k  + (int64_t)s*kvdim + h*hd, hd*sizeof(float));
        memcpy(m->V[li] + ((int64_t)h*m->max_t + t)*hd, vv + (int64_t)s*kvdim + h*hd, hd*sizeof(float));
    }
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
                const float *Kh = m->K[li] + ((int64_t)(h/group)*m->max_t)*hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *kv = Kh + (int64_t)t*hd;
                    float acc = 0.f;
                    for (int d = 0; d < hd; d++) acc += qv[d]*kv[d];
                    int dist = qpos - t;
                    sc[t - t0] = tau * (acc*scale + (dist < ext ? rl[dist] : 0.f));
                }
                int n = qpos - t0 + 1;
                softmax_row(sc, n);
                float *cx = ctx + (int64_t)s*qdim + h*hd;
                for (int d = 0; d < hd; d++) cx[d] = 0.f;
                const float *Vh = m->V[li] + ((int64_t)(h/group)*m->max_t)*hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *vrow = Vh + (int64_t)t*hd;
                    float a = sc[t - t0];
                    for (int d = 0; d < hd; d++) cx[d] += a * vrow[d];
                }
            }
        }
        free(rl); free(sc);
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
            si[kk] = best;
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
    int64_t npair = (int64_t)S*K;
    for (int64_t base = 0; base < npair; base += cap) {
        int64_t end = base + cap < npair ? base + cap : npair;
        nfill = 0;
        for (int64_t t = base; t < end; t++) {           /* acquisizione del giro */
            int s = (int)(t / K), kk = (int)(t % K);
            if (kk >= keff[s]) { use[t - base] = NULL; continue; }   /* scartato da TOPP */
            int eid = idx[(int64_t)s*K + kk];
            if (m->eusage[layer]) m->eusage[layer][eid]++;
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
        double te = now_s();
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
    /* shared experts: una volta per token, fuori dai giri (non usano la cache) */
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s*D;
        float *os = out + (int64_t)s*D;
        float *w = wgt + (int64_t)s*(K+ns);
        double ts = now_s();
        /* gamma inside (before down_proj is linear, so applied at the end) */
        for (int j = 0; j < ns; j++) {
            matmul_w(g, xs, wt_off_i(l->sh_g, (int64_t)j*I*D, D), 1, D, I);
            matmul_w(u, xs, wt_off_i(l->sh_u, (int64_t)j*I*D, D), 1, D, I);
            for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
            matmul_w(hh, g, wt_off_i(l->sh_d, (int64_t)j*D*I, I), 1, I, D);
            for (int d = 0; d < D; d++) os[d] += w[K+j] * hh[d];
        }
        m->t_shared += now_s() - ts;
    }
    free(logits); free(idx); free(keff); free(wgt); free(use); free(fill); free(fl);
    free(g); free(hh);              /* u aliases g+I */
}

/* ---------- one forward pass over S new tokens ----------
 * Returns malloc'd logits of the last token (unpadded vocab). If tf_out is
 * non-NULL also writes the per-position argmax (teacher-forcing check). */
static float *step(Model *m, const int *ids, int S, int pos0, int *tf_out) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) {
        wt_row_f32(m->embed, (int64_t)ids[s]*D, x + (int64_t)s*D, D);
        if (m->embed_norm) rmsnorm_row(x + (int64_t)s*D, x + (int64_t)s*D, m->embed_norm, D, c->eps);
    }
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        double ta = now_s();
        attention(m, l, i, nrm, S, pos0, tmp);
        m->t_attn += now_s() - ta;
        sconv_apply(tmp, S, D, l->a_cw, m->cs[2][i], c->conv_k);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        if (c->sparse[i]) moe(m, l, i, nrm, S, tmp);
        else dense_mlp(m, l, nrm, S, tmp);
        sconv_apply(tmp, S, D, l->m_cw, m->cs[3][i], c->conv_k);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->kv_len = pos0 + S;
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
    free(x); free(nrm); free(tmp); free(last);
    return logit;
}

static void state_reset(Model *m) {
    Cfg *c = &m->c;
    m->kv_len = 0;
    for (int i = 0; i < c->n_layers; i++) {
        int kvdim = L_KV(c,i) * L_HD(c,i);
        for (int j = 0; j < 4; j++)
            memset(m->cs[j][i], 0, (int64_t)((j < 2) ? kvdim : c->hidden) * (c->conv_k-1) * sizeof(float));
    }
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    if (m->K && max_t <= m->max_t) return;   /* reuse across prompts when big enough */
    if (m->K) for (int i = 0; i < c->n_layers; i++) { free(m->K[i]); free(m->V[i]); }
    free(m->K); free(m->V);
    m->max_t = max_t;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)L_KV(c,i) * max_t * L_HD(c,i));
        m->V[i] = falloc((int64_t)L_KV(c,i) * max_t * L_HD(c,i));
    }
}

/* greedy generation, olmoe.c-style */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0, NULL);
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
static void generate_stream(Model *m, Tok *T, const char *prompt, int n_new) {
    Cfg *c = &m->c;
    int cap = (int)strlen(prompt) + 16;
    int *ids = malloc(cap * sizeof(int));
    int np = tok_encode(T, prompt, (int)strlen(prompt), ids, cap);
    if (np <= 0) { fprintf(stderr, "empty prompt after tokenization\n"); return; }
    kv_alloc(m, np + n_new + 8);
    printf("[%d prompt tokens] %s", np, prompt); fflush(stdout);
    double t0 = now_s(), t1 = 0;
    float *logit = step(m, ids, np, 0, NULL);
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
    double wall = now_s() - t0;
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

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen; } SReq;
#define SRV_QMAX 16
static SReq g_q[SRV_QMAX]; static int g_qn = 0;

static int stdin_readable(void) {
    /* Windows non ha fd_set/select in questa forma: la build falliva del tutto.
     * La versione portabile (con i fix #139/#195) vive in compat.h, incluso via st.h. */
    return coli_stdin_readable();
}

/* read one control line (+ payload for SUBMIT). cur_id: request in flight;
 * returns 1 if that request was cancelled, 0 otherwise, -1 on stdin EOF. */
static int serve_read_cmd(const char *cur_id) {
    char ln[512];
    if (!fgets(ln, sizeof(ln), stdin)) return -1;
    char cmd[16], id[64];
    if (sscanf(ln, "%15s %63s", cmd, id) < 2) return 0;
    if (!strcmp(cmd, "CANCEL")) return cur_id && !strcmp(id, cur_id);
    if (!strcmp(cmd, "SUBMIT")) {
        int slot, plen, max_tok; float temp, top_p;
        if (sscanf(ln, "%*s %*s %d %d %d %f %f", &slot, &plen, &max_tok, &temp, &top_p) != 5 ||
            plen < 0 || plen > (1<<22)) { printf("ERROR %s bad submit header\n", id); fflush(stdout); return 0; }
        (void)slot;
        char *pl = malloc((size_t)plen + 1);
        if (fread(pl, 1, (size_t)plen, stdin) != (size_t)plen) { free(pl); return -1; }
        int nl = fgetc(stdin); (void)nl;
        pl[plen] = 0;
        if (g_qn < SRV_QMAX) {
            SReq *q = &g_q[g_qn++];
            snprintf(q->id, sizeof(q->id), "%s", id);
            q->max_tok = max_tok; q->temp = temp; q->top_p = top_p;
            q->payload = pl; q->plen = plen;
        } else { printf("ERROR %s queue full\n", id); fflush(stdout); free(pl); }
    }
    return 0;
}

static void serve_one(Model *m, Tok *T, SReq *q) {
    Cfg *c = &m->c;
    int cap = q->plen + 16;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { printf("ERROR %s empty prompt\n", q->id); fflush(stdout); free(ids); return; }
    const char *bad = prompt_reject(np, q->max_tok);
    if (bad) { printf("ERROR %s %s\n", q->id, bad); fflush(stdout); free(ids); return; }
    state_reset(m);
    kv_alloc(m, np + q->max_tok + 8);
    double t0 = now_s();
    uint64_t h0 = m->hits, m0 = m->miss;
    /* per-turn phase snapshot for the PROF line (timers accumulate globally) */
    double f0 = m->t_fill, e0 = m->t_expert, s0 = m->t_shared, a0 = m->t_attn;
    float *logit = step(m, ids, np, 0, NULL);
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
        printf("DATA %s %d\n", q->id, nb);
        fwrite(buf, 1, (size_t)nb, stdout);
        fputc('\n', stdout); fflush(stdout);
        gen++; len++;
        while (stdin_readable()) {
            int r = serve_read_cmd(q->id);
            if (r < 0) { free(ids); return; }
            if (r > 0) { cancelled = 1; limited = 0; }
        }
        if (cancelled || s == q->max_tok - 1) break;
        logit = step(m, &tk, 1, len - 1, NULL);
    }
    free(logit);
    double dt = now_s() - t0;
    double tot = (double)(m->hits - h0 + m->miss - m0);
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", q->id, gen,
           dt > 0 ? gen/dt : 0.0, tot ? 100.0*(m->hits-h0)/tot : 0.0, rss_gb(), np, limited);
    /* PROF: per-turn phase timings for the dashboard (gateway schema — we map
     * expert_wait -> shared-expert compute, lm_head folded into 0). */
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %d\n", dt, np, gen,
           m->t_fill - f0, m->t_shared - s0, m->t_expert - e0, m->t_attn - a0, 0.0, gen + 1);
    fflush(stdout);
    free(ids);
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
        LCache *lc = &m->cache[i];
        for (int e = 0; e < E; e++) {
            int tier = 0;
            for (int z = 0; z < lc->n; z++) if (lc->slots[z].eid == e && lc->slots[z].filled) { tier = 1; break; }
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
    coli_serve_binary_mode();   /* #748: TEXT-mode stdout mangles the READY sentinel */
    setvbuf(stdin, NULL, _IONBF, 0);
    const char *sd = getenv("SEED");
    if (sd) g_rng ^= (uint64_t)strtoull(sd, NULL, 10);
    else g_rng ^= (uint64_t)time(NULL) * 2654435761u;
    /* the gateway reads a STAT line right after the READY sentinel (colibri
     * reports its load stats there) — match the handshake */
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", rss_gb());
    fflush(stdout);
    serve_hwinfo(m);
    serve_tiers_emap(m);
    for (;;) {
        while (!g_qn) if (serve_read_cmd(NULL) < 0) return;   /* blocks on stdin */
        SReq q = g_q[0];
        memmove(g_q, g_q+1, (size_t)(--g_qn) * sizeof(SReq));
        serve_one(m, T, &q);
        serve_tiers_emap(m);
        free(q.payload);
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

int main(int argc, char **argv) {
    /* OpenMP hot-thread tuning, same trick (and rationale) as glm.c: the
     * per-expert matmul regions are tiny and back-to-back; the default passive
     * wait policy parks the team between regions and re-wake latency dominates.
     * libgomp reads OMP_/GOMP_ vars before main(), so seed them and re-exec
     * once (COLI_OMP_TUNED guards the exec; COLI_NO_OMP_TUNE=1 disables).
     * NOT under CUDA — same exception glm.c makes: a spinning 24-thread team
     * starves the CUDA driver during every stream sync. */
#ifndef COLI_CUDA
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
#endif  /* !COLI_CUDA */
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
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
    int cap = -1, bits = 0, n_new = 256, npos = 0, chat = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-f") && i+1 < argc) pfile = argv[++i];
        else if (!strcmp(argv[i], "-n") && i+1 < argc) n_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (npos == 0) { cap = atoi(argv[i]); npos++; }
        else if (npos == 1) { bits = atoi(argv[i]); npos++; }
        else refpath = argv[i];
    }
    /* --chat: avvolge il prompt nel template di Inkling (sottoinsieme testuale di
     * chat_template.jinja, lo stesso che openai_server.py rende in render_chat_inkling):
     * token di ruolo + <|content_text|>, <|end_message|> a chiudere, il livello di
     * thinking come messaggio di sistema, e <|message_model|> come prompt di
     * generazione. Senza template un modello instruct riceve testo fuori
     * distribuzione. THINK=<0..1> alza lo sforzo di ragionamento (default 0). */
    char *chat_buf = NULL;
    if (chat && prompt) {
        const char *eff = getenv("THINK") ? getenv("THINK") : "0";
        size_t need = strlen(prompt) + strlen(eff) + 256;
        chat_buf = malloc(need);
        if (!chat_buf) { fprintf(stderr, "OOM chat template\n"); return 1; }
        snprintf(chat_buf, need,
                 "<|message_user|><|content_text|>%s<|end_message|>"
                 "<|message_system|><|content_text|>Thinking effort level: %s<|end_message|>"
                 "<|message_model|>", prompt, eff);
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
        if (prompt) generate_stream(&m, &T, prompt, n_new);
        else {   /* -f: one prompt per line, model loaded once, usage accumulates */
            FILE *pf = fopen(pfile, "rb"); if (!pf) { perror(pfile); return 1; }
            char ln[8192]; int np = 0;
            while (fgets(ln, sizeof(ln), pf)) {
                size_t n = strlen(ln); while (n && (ln[n-1]=='\n'||ln[n-1]=='\r')) ln[--n]=0;
                if (!n || ln[0]=='#') continue;
                printf("\n===== prompt %d =====\n", ++np);
                state_reset(&m);
                generate_stream(&m, &T, ln, n_new);
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
    int np, nfull, ntf;
    int *pids  = read_int_array(ref,"prompt_ids",&np);
    int *full  = read_int_array(ref,"full_ids",&nfull);
    int *tfref = read_int_array(ref,"tf_pred",&ntf);
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

    /* pass 1: teacher-forced argmax over the full reference sequence */
    if (tfref && ntf == nfull) {
        int *tf = malloc(nfull * sizeof(int));
        float *lg = step(&m, full, nfull, 0, tf);
        free(lg);
        int ok = 0; for (int i = 0; i < nfull; i++) ok += (tf[i] == tfref[i]);
        printf("teacher-forced argmax: %d/%d match\n", ok, nfull);
        free(tf);
        state_reset(&m);
    }

    /* pass 2: greedy generation, token-for-token vs the oracle */
    int *out = malloc(nfull * sizeof(int));
    double t = now_s();
    generate(&m, pids, np, ngen, out);
    double dt = now_s() - t;
    int match = 0;
    printf("Reference: "); for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : "); for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, ngen);
    double tot = m.hits + m.miss;
    printf("PEAK RSS: %.2f GB | expert cache hit %.1f%% | %.2f tok/s\n",
           rss_gb(), tot?100.0*m.hits/tot:0.0, ngen/dt);
    free(buf); free(arena);
    return (match == ngen) ? 0 : 1;
}
