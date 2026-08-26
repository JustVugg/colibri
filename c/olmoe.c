/* Motore di inferenza OLMoE in C puro, con EXPERT-STREAMING dal disco.
 * Porting del motore Python (engine.py). Obiettivo Stadio A: produrre gli STESSI
 * token id del riferimento (ref.json) -> valida il core prima di scalare a GLM-5.2.
 *
 * Densa (embed, attn, router, norme, lm_head) residente in RAM (float32).
 * Expert letti dal disco on-demand via pread, cache LRU per-layer; le pagine
 * restano nel page cache cosi' i miss LRU non tornano su disco (EXPERT_DROP=1
 * ripristina fadvise(DONTNEED) per macchine con poca RAM).
 * Matmul multi-thread con OpenMP (niente BLAS).
 *
 * ENV VARS:
 *   PILOT=0/1/2/3 : 0=no prefetch, 1=1-layer lookahead, 2=2-layer, 3=3-layer lookahead
 *   HOT=N         : pin top-N hot experts per layer permanently (never evict)
 *   WARMUP=N      : tokens before hot pinning activates (default 5)
 *   WIDE=N        : prefetch top-K*N candidates (default 1, try 2 or 3)
 *   SMOOTH=F      : EMA coefficient for routing momentum (default 0.3, range 0.0-0.95)
 *   CONF_LIMIT=F  : cumulative gate probability threshold for prefetch cutoff (default 0.92)
 *   PILOT_EVICT_GUARD=0/1 : 1=enable LFRU prefetch eviction guard (default), 0=disable
 *   EXPERT_DROP=0/1: 1=fadvise(DONTNEED) after each expert read (old behaviour,
 *                    for RAM-tight boxes); 0=keep pages cached (default)
 *   (expert queue is sorted by eid for SSD read locality)
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
#include <unistd.h>
#endif
#include "st.h"
#ifdef _OPENMP
#include <omp.h>   /* omp_set_num_threads/omp_get_max_threads per omp_tune.h */
#endif
#include "omp_tune.h"
#include "route_trace.h"                    /* shared routing telemetry (#700) */
#include "serve_codec.h"
#ifdef COLI_SEGMENT_ADAPTER
#include "segment_runtime.h"
#include "segment_adapters.h"
#include "segment_adapter_internal.h"
#endif

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#define sleep_ms(ms) usleep((ms) * 1000)
#endif



/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, topk, inter, vocab;
    float theta, eps; int norm_topk;
    int stop_ids[8], n_stop;   /* unused (no model-specific hardcoded stops) — chat mode's
                                 * stop set comes entirely from the tokenizer's own special-
                                 * token flags via sample.h's stops_arm_tok */
} Cfg;

/* ---------- pesi densi per-layer ---------- */
typedef struct {
    float *in_ln, *post_ln, *q, *k, *v, *o, *qn, *kn, *gate;
} Layer;

/* ---------- cache LRU degli expert (pesi QUANTIZZATI) ----------
 * Ogni weight [out,in] tenuto come int8 (per-riga) + scala float per riga.
 * Cosi' la RAM-cache scende da 4 byte/param (f32) a 1 byte/param: e' il
 * meccanismo che fa stare GLM-5.2 nei 15 GB. dequant-on-use nel matmul. */
/* pinned=1 means this slot is strongly preferred to keep (hot expert); it will
 * not be evicted during normal LRU eviction, but may be displaced under extreme
 * cache pressure when all slots are pinned or in-flight. */
typedef struct { int eid; int pinned; int8_t *g, *u, *d; float *gs, *us, *ds; uint64_t used; } Slot;
typedef struct {
    Slot *slots;
    int *slot_by_expert;                  /* expert id -> resident slot, -1 if absent */
    int n, cap;
} LCache;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;
    float *embed, *lm_head, *final_norm;
    Layer *L;
    LCache *cache;          /* [n_layers] */
    uint64_t clock, hits, miss;
    float **K, **V; int kv_len, max_t;
    double dense_load_s;
    /* IMPROVEMENT 2: expert frequency heatmap */
    uint32_t **freq;                   /* per-layer expert counts, owned by route_trace.h */
    int freq_token_count, hot_pinned, hot_n, warmup_tokens;
    int token_count;
    /* PREDICTION IMPROVEMENT A: per-layer EMA of gate logits across tokens.
     * momentum_logits[l*E .. (l+1)*E-1] = EMA of gate outputs for layer l.
     * Used exclusively by the PILOT prefetcher to stabilise routing predictions
     * across tokens; does NOT affect actual MoE routing (pr is unchanged). */
    float *momentum_logits; /* [n_layers * n_experts], EMA of gate logits */
    float pilot_smooth;     /* SMOOTH env: EMA coefficient 0.0-0.9 (default 0.3) */
    uint8_t *is_pinned;     /* [n_layers * n_experts], 1 if expert is globally pinned */
    uint8_t *is_queued;     /* [n_layers * n_experts], 1 if expert is currently in the prefetch queue */
    float pilot_conf_limit; /* CONF_LIMIT env: cumulative gate probability threshold (e.g. 0.92) */
    uint64_t *last_access;  /* [n_layers * n_experts], clock time when expert was last accessed */
} Model;

static pthread_mutex_t g_pilot_mx = PTHREAD_MUTEX_INITIALIZER;
static struct { int l, e; } pilot_q[4096];
static volatile unsigned pilot_r = 0, pilot_w = 0;
static Model *pilot_m = NULL;
static int g_pilot = 0;
static int g_wide  = 1;  /* IMPROVEMENT 4: top-K * g_wide candidates prefetched */
static int g_pilot_evict_guard = 1; /* PILOT_EVICT_GUARD=0 to disable LFRU prefetch eviction guard */
static int g_expert_drop = 0;       /* EXPERT_DROP=1 restores fadvise(DONTNEED) after expert reads */
static int g_fused3 = 0;            /* FUSED3=1: AVX2 activation quant + gate/up pair matmul
                                     * (fused_simd.h: quant_x_q8_avx2, matmul_q_idot_v3,
                                     * matmul_q_idot_pair_v3). Exact integer arithmetic only —
                                     * bit-identical to the stock matmul_q path; OFF by default. */

static uint64_t lfru_score(uint32_t heat, uint64_t last, uint64_t clock) {
    uint64_t age = (clock > last) ? (clock - last) : 0;
    uint64_t recent = (age < 255) ? (255 - age) : 0;
    return ((uint64_t)heat << 8) | recent;
}

static void pilot_prefetch(Model *m, int lnext, const float *x, int S);
static void *pilot_worker(void *arg);
static void ensure_pilot_worker_started(Model *m);
static void slot_ensure_allocated(Model *m, Slot *s);

#ifdef COLI_CACHE_INDEX_TEST
static uint64_t g_slot_index_probes;
#endif

/* Runtime callers hold g_pilot_mx.  The defensive eid check is intentional:
 * an index bug must degrade to a miss, never serve another expert's weights. */
static Slot *slot_indexed(Model *m, int layer, int eid) {
    if (layer < 0 || layer >= m->c.n_layers || eid < 0 ||
        eid >= m->c.n_experts) return NULL;
    LCache *lc = &m->cache[layer];
    if (!lc->slot_by_expert) return NULL;
#ifdef COLI_CACHE_INDEX_TEST
    g_slot_index_probes++;
#endif
    int i = lc->slot_by_expert[eid];
    if (i < 0 || i >= lc->n || lc->slots[i].eid != eid) return NULL;
    return &lc->slots[i];
}

static void cache_unindex(Model *m, int layer, Slot *s) {
    LCache *lc = &m->cache[layer];
    int eid = s->eid, i = (int)(s - lc->slots);
    if (lc->slot_by_expert && eid >= 0 && eid < m->c.n_experts &&
        lc->slot_by_expert[eid] == i)
        lc->slot_by_expert[eid] = -1;
}

static void cache_hide(Model *m, int layer, Slot *s) {
    cache_unindex(m, layer, s);
    s->eid = -1;
}

static void cache_publish(Model *m, int layer, Slot *s, int eid) {
    LCache *lc = &m->cache[layer];
    cache_unindex(m, layer, s);
    s->eid = eid;
    if (lc->slot_by_expert && eid >= 0 && eid < m->c.n_experts)
        lc->slot_by_expert[eid] = (int)(s - lc->slots);
}

static void ensure_pilot_worker_started(Model *m) {
    if (!pilot_m) {
        pilot_m = m;
        pthread_t t;
        if (pthread_create(&t, NULL, pilot_worker, NULL) != 0) {
            fprintf(stderr, "Error: Failed to create pilot prefetch worker thread\n");
            exit(1);
        }
        pthread_detach(t);
    }
}

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }  /* macOS: byte */
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }        /* Linux: KB */
#endif
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* chat mode only (main()'s CHAT=1 path): sampling temperature/top-p and the
 * tokenizer + stop-set machinery. g_temp<=0 -> greedy (sample.h's pick_tok).
 * Declared before #include "sample.h" — it references these by name, and
 * Cfg/falloc above, without its own extern declarations. */
static float g_temp = 0.7f;   /* TEMP env overrides */
static float g_nuc  = 0.95f;  /* NUCLEUS env overrides */
#include "sample.h"

/* y[S,O] = x[S,I] @ W^T,  W e' [O,I] row-major */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            #pragma omp simd reduction(+:acc)
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* y[1,O] = x[1,I] @ W^T con W quantizzato: q[O,I] int8 + scala per riga.
 * W[o,i] ~= q[o,i]*scale[o]  ->  y[o] = scale[o] * sum_i x[i]*q[o,i].
 * Su ARM: attivazione quantizzata Q8_0 (scala per blocco di 16) + dot int8
 * NEON (sdot dove c'e' dotprod) — stessa famiglia IDOT di glm.c, IDOT=0 per
 * la via scalare byte-esatta. Misurato 2.7x end-to-end su M5.
 *
 * NB: la quantizzazione delle ATTIVAZIONI rende questo percorso non
 * equivalente alla via scalare (issue #1044). Vale per NEON come per AVX2:
 * IDOT e' opt-in (IDOT=1), non piu' attivo di default. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
    return vaddvq_s32(acc);
}
#define HAVE_FAST_DOT_I8 1
#elif defined(__AVX2__)
#include <immintrin.h>
/* x86 counterpart of the NEON path above (was scalar-only here before —
 * the only fast path was ARM, so x86 boxes silently used the scalar
 * fallback even when AVX2 was available).
 * Sign-extend both int8 vectors to int16 (exact, no precision loss) then
 * madd+horizontal-sum in int32: pure integer arithmetic, so THIS DOT is
 * bit-for-bit identical to a scalar int8 dot, just vectorized.
 *
 * That exactness does NOT extend to the branch that calls it. matmul_q's IDOT
 * path quantizes the ACTIVATIONS to Q8_0 per 16-block before calling this,
 * which the scalar fallback does not do -- so the two paths differ. This
 * comment previously read as if it covered the whole path, which is how
 * issue #1044 stayed invisible. See the note at the idot default below. */
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    __m256i va16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
    __m256i vb16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)b));
    __m256i prod = _mm256_madd_epi16(va16, vb16);           /* 8 x int32, adjacent pairs summed */
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(prod), _mm256_extracti128_si256(prod, 1));
    __m128i hi64   = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64  = _mm_add_epi32(sum128, hi64);
    __m128i hi32   = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32  = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}
#define HAVE_FAST_DOT_I8 1
#endif
/* Test-only hook, compiled out of the shipping binary.
 *
 * matmul_q reads IDOT once into a static, so a test cannot exercise both paths
 * in one process without it. Guarded by OLMOE_TESTING so production builds have
 * neither the global nor the branch: tests/test_olmoe_matmul_q.c defines it. */
#ifdef OLMOE_TESTING
int matmul_q_idot_force = -1;
static inline void matmul_q_reset_for_test(void) { matmul_q_idot_force = -1; }
#endif

#if defined(__AVX2__)
#include "fused_simd.h"   /* FUSED3=1: quant_x_q8_avx2 + matmul_q_idot{,_pair}_v3 (bit-exact) */
#endif

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(HAVE_FAST_DOT_I8)
    /* IDOT is OPT-IN. It quantizes the ACTIVATIONS to Q8_0 per 16-block (below),
     * which the scalar fallback never does, so the two paths are not numerically
     * equivalent: measured 5/12 vs 12/12 matching tokens against a reference
     * (issue #1044). Before 2c4e9de x86 had no fast dot at all, so IDOT=1 fell
     * through to the exact scalar path and x86 was token-exact by accident; that
     * commit silently made every AVX2 box lossy by default. Defaulting to off
     * restores the behaviour users actually had.
     *
     * Cost of turning it on: ~+5.8e-4 relative error per dot from the activation
     * quantization (colibri.c:910-915 measures the same mechanism at ~+0.117
     * nats/token on GLM, which is why GLM keeps q/k/v off IDOT). On AVX2-only
     * hardware it is also not faster: 11.01 vs 10.67 tok/s measured on an
     * i5-9600K, n=4 interleaved. Set IDOT=1 to opt in knowingly. */
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = (e && *e == '1'); }
#ifdef OLMOE_TESTING
    if (matmul_q_idot_force >= 0) idot = matmul_q_idot_force;
#endif
    if (idot && I % 16 == 0 && I <= 4096) {
        int nb = I / 16; int8_t xi[4096]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*16;
            float am = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 16; i++) xi[b*16+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b]*(float)dot_i8_16(xi+b*16, w+b*16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        #pragma omp simd reduction(+:acc)
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}


/* rmsnorm su una riga di lunghezza D, in-place su out (out puo' essere == x) */
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

/* ---------- caricamento ---------- */
/* config.json arrives from an untrusted mirror: a missing key was a NULL-deref
 * (json_get(...)->num), so require each dimension present + numeric. */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0 || n>(256L<<20)){ fprintf(stderr,"%s: config.json missing or larger than 256 MB\n",path); exit(1); }  /* SEC-9 */
    char *buf = malloc((size_t)n+1); if(!buf){ fprintf(stderr,"OOM reading %s\n",path); exit(1); }
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); } buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    c->hidden    = (int)req_num(r,"hidden_size");
    c->n_layers  = (int)req_num(r,"num_hidden_layers");
    c->n_heads   = (int)req_num(r,"num_attention_heads");
    c->n_kv_heads= (int)req_num(r,"num_key_value_heads");
    c->n_experts = (int)req_num(r,"num_experts");
    c->topk      = (int)req_num(r,"num_experts_per_tok");
    c->inter     = (int)req_num(r,"intermediate_size");
    c->vocab     = (int)req_num(r,"vocab_size");
    /* range-check so bad dims can't drive a later malloc(inter*hidden) / div-by-zero */
    if(c->hidden<1||c->hidden>(1<<20) || c->n_heads<1||c->n_heads>(1<<16) ||
       c->inter<1||c->inter>(1<<24) || c->vocab<1||c->vocab>(1<<24) ||
       c->n_layers<1||c->n_layers>4096 || c->n_experts<1||c->n_experts>(1<<20) ||
       c->n_kv_heads<1 || c->topk<1||c->topk>c->n_experts){
        fprintf(stderr,"config.json: dimension out of range\n"); exit(1); }
    c->head_dim  = c->hidden / c->n_heads;
    jval *th = json_get(r,"rope_theta");  c->theta = th ? (float)th->num : 10000.f;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-5f;
    jval *nt = json_get(r,"norm_topk_prob"); c->norm_topk = (nt && nt->t==J_BOOL) ? nt->boolean : 0;
    free(buf); free(arena);
}

static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);   /* densa: niente DONTNEED, resta residente */
    return p;
}

static void model_init_range(Model *m, const char *snap, int cap, int bits,
                             int layer_begin, int layer_end,
                             int load_boundaries, int init_telemetry) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    if (layer_end == 0) layer_end = c->n_layers;
    if (layer_begin < 0 || layer_end > c->n_layers ||
        layer_begin >= layer_end) {
        fprintf(stderr, "invalid OLMoE layer range [%d,%d) for %d layers\n",
                layer_begin, layer_end, c->n_layers);
        exit(1);
    }
    double t0 = now_s();
    if (load_boundaries) {
        m->embed      = load_t(m, "model.embed_tokens.weight");
        m->lm_head    = load_t(m, "lm_head.weight");
        m->final_norm = load_t(m, "model.norm.weight");
    }
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = layer_begin; i < layer_end; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LD(q, "self_attn.q_proj.weight"); LD(k, "self_attn.k_proj.weight");
        LD(v, "self_attn.v_proj.weight"); LD(o, "self_attn.o_proj.weight");
        LD(qn,"self_attn.q_norm.weight"); LD(kn,"self_attn.k_norm.weight");
        LD(gate, "mlp.gate.weight");
        #undef LD
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = layer_begin; i < layer_end; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc(cap, sizeof(Slot));
        m->cache[i].slot_by_expert = malloc((size_t)c->n_experts * sizeof(int));
        if (!m->cache[i].slot_by_expert) { fprintf(stderr,"OOM expert cache index\n"); exit(1); }
        for (int e = 0; e < c->n_experts; e++) m->cache[i].slot_by_expert[e] = -1;
    }
    /* IMPROVEMENT 2: frequency heatmap for hot expert pinning */
    if (init_telemetry) {
        rt_init("olmoe", c->n_layers, c->n_experts);
        rt_drop_row(c->n_layers);                 /* every layer routes; no MTP row */
        m->freq = rt_counts_all();                /* read sites keep their shape */
        const char *up = getenv("COLI_USAGE");    /* optional history seed */
        if (up && *up) {
            int64_t h = rt_load(up);
            if (h > 0)
                fprintf(stderr,
                        "[USAGE] expert history: %lld selections (%s)\n",
                        (long long)h, up);
        }
    } else {
        /* A process can host several ranges. Keep their optional counters
         * detached from route_trace.h's process-global singleton. */
        m->freq = calloc((size_t)c->n_layers, sizeof(*m->freq));
    }
    m->hot_pinned = 0; m->freq_token_count = 0;
    m->hot_n         = getenv("HOT")    ? atoi(getenv("HOT"))    : 0;
    m->warmup_tokens = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    m->token_count = 0;
    /* PREDICTION A: routing momentum — EMA of gate logits across tokens.
     * Initialized to zero; first token sets EMA = fresh logits. */
    m->momentum_logits = calloc((size_t)c->n_layers * c->n_experts, sizeof(float));
    float sv = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    if (sv < 0.f) sv = 0.f; if (sv > 0.95f) sv = 0.95f;
    m->pilot_smooth = sv;
    m->is_pinned = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    m->is_queued = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    m->last_access = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint64_t));
    float cl = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;
    if (cl < 0.1f) cl = 0.1f; if (cl > 1.0f) cl = 1.0f;
    m->pilot_conf_limit = cl;
    m->dense_load_s = now_s() - t0;

    /* Persistent hot pinning belongs to the standalone runtime. A Segment
     * range must never enqueue expert reads for layers it does not own. */
    char pinpath[512];
    snprintf(pinpath, sizeof(pinpath), "%s/hot_pinned.bin", snap);
    FILE *pinf = init_telemetry ? fopen(pinpath, "rb") : NULL;
    if (pinf) {
        size_t expected_size = (size_t)c->n_layers * c->n_experts;
        if (fread(m->is_pinned, 1, expected_size, pinf) == expected_size) {
            m->hot_pinned = 1;
            printf("[HOT] Loaded persistent pinning from %s\n", pinpath);
            
            if (g_pilot) {
                ensure_pilot_worker_started(m);
                for (int l = 0; l < c->n_layers; l++) {
                    for (int e = 0; e < c->n_experts; e++) {
                        if (m->is_pinned[l * c->n_experts + e]) {
                            unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                            unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                            if (w - r < 4096) {
                                pilot_q[w & 4095].l = l; pilot_q[w & 4095].e = e;
                                pthread_mutex_lock(&g_pilot_mx);
                                m->is_queued[l * c->n_experts + e] = 1;
                                pthread_mutex_unlock(&g_pilot_mx);
                                __atomic_store_n(&pilot_w, w + 1, __ATOMIC_RELEASE);
                            }
                        }
                    }
                }
                printf("[HOT] Pre-loading pinned experts into cache...\n");
                double t_wait = now_s();
                while (1) {
                    unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                    unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE);
                    if (r == w) break;
                    sleep_ms(2);
                }
                printf("[HOT] Pre-loaded in %.1fs!\n", now_s() - t_wait);
            }
        }
        fclose(pinf);
    }
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    model_init_range(m, snap, cap, bits, 0, 0, 1, 1);
}

static void slot_ensure_allocated(Model *m, Slot *s) {
    if (s->g) return;
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden;
    int64_t nd = (int64_t)c->hidden * c->inter;
    int8_t *w_block = malloc(ng + ng + nd);
    if (!w_block) {
        fprintf(stderr, "Error: Out of memory allocating slot weights block\n");
        exit(1);
    }
    s->g = w_block;
    s->u = w_block + ng;
    s->d = w_block + ng + ng;
    float *s_block = falloc(c->inter + c->inter + c->hidden);
    s->gs = s_block;
    s->us = s_block + c->inter;
    s->ds = s_block + c->inter + c->inter;
    s->pinned = 0;
}

static void load_expert_merged(Model *m, int layer, int eid, Slot *s) {
    char nm[256], qsnm[256];
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.merged_weight", layer, eid);
    snprintf(qsnm, sizeof(qsnm), "model.layers.%d.mlp.experts.%d.qs", layer, eid);
    /* SEC: st_init skips its numel*esz==nbytes cross-check for dtype-3 (U8/I8)
     * tensors, so a crafted header from an untrusted mirror can declare nbytes
     * far larger than the config-sized destination and st_read_raw would write
     * past w_block (heap overflow); an oversized .qs likewise overruns s->gs.
     * slot_ensure_allocated sizes those buffers exactly ng+ng+nd bytes and
     * inter+inter+hidden floats, so require an exact match. Reject, never
     * repair — same contract as qt_resolve_fmt in the GLM engine. */
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->inter * cc->hidden, nd = (int64_t)cc->hidden * cc->inter;
    int64_t want_w = ng + ng + nd;
    int64_t want_s = (int64_t)cc->inter + cc->inter + cc->hidden;
    st_tensor *tw = st_find(&m->S, nm), *ts = st_find(&m->S, qsnm);
    if (!tw || tw->nbytes != want_w) {
        fprintf(stderr, "%s: expert weight is %lld bytes — expected %lld for [inter=%d,hidden=%d], "
                "refusing (untrusted container)\n", nm, (long long)(tw ? tw->nbytes : -1),
                (long long)want_w, cc->inter, cc->hidden); exit(1); }
    if (!ts || ts->numel != want_s) {
        fprintf(stderr, "%s: scale array is %lld elems — expected %lld, refusing (untrusted container)\n",
                qsnm, (long long)(ts ? ts->numel : -1), (long long)want_s); exit(1); }
    st_read_raw(&m->S, nm, s->g, g_expert_drop);
    st_read_f32(&m->S, qsnm, s->gs, 0);  /* scales are F32; use typed reader for dtype safety */
}

/* ---------- cache expert: ritorna i pesi quantizzati (q+scale) da cache o disco ---------- */
static void expert_get(Model *m, int layer, int eid, Slot **out) {
    LCache *lc = &m->cache[layer];
    pthread_mutex_lock(&g_pilot_mx);
    Slot *hit = slot_indexed(m, layer, eid);
    if (hit) {
        m->hits++; hit->used = ++m->clock; *out = hit;
        if (m->last_access) m->last_access[layer * m->c.n_experts + eid] = m->clock;
        pthread_mutex_unlock(&g_pilot_mx);
        return;
    }
    m->miss++;
    Cfg *c = &m->c;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        slot_ensure_allocated(m, s);
    } else {
        /* LRU eviction — skip pinned and in-flight (eid==-1) slots */
        int lru = -1;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
        }
        if (lru < 0) {
            /* All slots are pinned or in-flight; find oldest non-in-flight slot
             * (may be pinned, but never select one currently being loaded). */
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid < 0) continue; /* never evict in-flight */
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
        }
        while (lru < 0) {
            /* EVERY slot is in flight: each buffer is owned by an unlocked pread
             * in the pilot worker (or a demand load) that will publish into it.
             * The old last resort (lru=0) stole such a slot mid-load — two writers
             * racing the same slab, then whichever published last decided the
             * expert id the resident bytes answered to. Wait for a publish instead
             * and rescan; in-flight always drains because a load either finishes
             * or the process is already dead in the water. */
            pthread_mutex_unlock(&g_pilot_mx);
            sleep_ms(1);
            pthread_mutex_lock(&g_pilot_mx);
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid < 0) continue;
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
        }
        s = &lc->slots[lru];
        s->pinned = 0;
    }
    cache_hide(m, layer, s);
    s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);

    load_expert_merged(m, layer, eid, s);

    pthread_mutex_lock(&g_pilot_mx);
    cache_publish(m, layer, s, eid);
    s->pinned = m->is_pinned[layer * c->n_experts + eid];
    s->used = ++m->clock;
    if (m->last_access) m->last_access[layer * c->n_experts + eid] = m->clock;
    *out = s;
    pthread_mutex_unlock(&g_pilot_mx);
}

/* ---------- IMPROVEMENT 2: pin top-N hot experts per layer ---------- */
static void pin_hot_experts(Model *m) {
    Cfg *c = &m->c;
    if (m->hot_n <= 0 || m->hot_pinned) return;
    m->hot_pinned = 1;
    
    int is_dynamic = (m->hot_n >= 100);
    double thresh = is_dynamic ? (double)m->hot_n / 1000.0 : 0.0;
    
    int pinned_total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        uint32_t *freq_l = m->freq[l];
        if (!freq_l) continue;                    /* a layer with no row cannot be ranked */

        uint64_t layer_total = 0;
        for (int e = 0; e < c->n_experts; e++) layer_total += freq_l[e];
        if (layer_total == 0) continue;

        int max_pin = m->cache[l].cap - 8;
        if (max_pin < 4) max_pin = 4;
        
        int hn = is_dynamic ? max_pin : (m->hot_n < c->n_experts ? m->hot_n : c->n_experts);
        if (hn > 256) hn = 256;
        int hot_eids[256];
        int actual_hn = 0;
        
        for (int k = 0; k < hn; k++) {
            int best = -1; uint32_t bv = 0;
            for (int e = 0; e < c->n_experts; e++) {
                int already = 0;
                for (int j = 0; j < k; j++) if (hot_eids[j] == e) { already = 1; break; }
                if (!already && freq_l[e] > bv) { bv = freq_l[e]; best = e; }
            }
            if (best < 0 || bv == 0) break;
            if (is_dynamic && bv < thresh * layer_total) break;
            hot_eids[k] = best;
            actual_hn++;
        }
        
        for (int k = 0; k < actual_hn; k++) {
            int eid = hot_eids[k];
            m->is_pinned[l * c->n_experts + eid] = 1;

            int found = 0;
            pthread_mutex_lock(&g_pilot_mx);
            Slot *resident = slot_indexed(m, l, eid);
            if (resident) { resident->pinned = 1; found = 1; }
            pthread_mutex_unlock(&g_pilot_mx);
            if (!found && g_pilot > 0) {
                /* Only enqueue when the prefetch worker is active (PILOT>0). */
                ensure_pilot_worker_started(m);
                unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                int gidx = l * c->n_experts + eid;
                pthread_mutex_lock(&g_pilot_mx);
                int already = m->is_queued[gidx];
                if (!already && w - r < 4096) {
                    pilot_q[w & 4095].l = l; pilot_q[w & 4095].e = eid;
                    m->is_queued[gidx] = 1;
                    __atomic_store_n(&pilot_w, w + 1, __ATOMIC_RELEASE);
                }
                pthread_mutex_unlock(&g_pilot_mx);
            }
            pinned_total++;
        }
    }
    if (is_dynamic) {
        printf("[HOT] Dynamic Pinned %d experts total (thresh=%.1f%%) after %d warmup tokens\n",
               pinned_total, thresh * 100.0, m->freq_token_count);
    } else {
        printf("[HOT] Pinned %d experts (top-%d/layer) after %d warmup tokens\n",
               pinned_total, m->hot_n, m->freq_token_count);
    }
}


/* ---------- RoPE su un vettore di una testa (head_dim) a posizione assoluta pos ---------- */
static void rope_head(float *x, int pos, const Cfg *c) {
    int h = c->head_dim / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(c->theta, -2.0f * j / c->head_dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* attenzione sui token nuovi x[S,hidden]; pos_base = posizione assoluta del primo token nuovo */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c; int H = c->n_heads, hd = c->head_dim, D = c->hidden;
    float *q = falloc((int64_t)S*D), *k = falloc((int64_t)S*D), *vv = falloc((int64_t)S*D);
    matmul(q, x, l->q, S, D, D);
    matmul(k, x, l->k, S, D, D);
    matmul(vv, x, l->v, S, D, D);
    /* qk-norm sull'intero vettore hidden, poi RoPE per testa */
    for (int s = 0; s < S; s++) {
        rmsnorm_row(q + (int64_t)s*D, q + (int64_t)s*D, l->qn, D, c->eps);
        rmsnorm_row(k + (int64_t)s*D, k + (int64_t)s*D, l->kn, D, c->eps);
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) { rope_head(q + (int64_t)s*D + hh*hd, pos, c); rope_head(k + (int64_t)s*D + hh*hd, pos, c); }
    }
    /* scrive k,v nella kv-cache alle posizioni pos_base..pos_base+S-1 */
    for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
        int t = pos_base + s;
        memcpy(m->K[layer] + ((int64_t)hh*m->max_t + t)*hd, k + (int64_t)s*D + hh*hd, hd*sizeof(float));
        memcpy(m->V[layer] + ((int64_t)hh*m->max_t + t)*hd, vv + (int64_t)s*D + hh*hd, hd*sizeof(float));
    }
    int Tk = pos_base + S;             /* numero di key totali disponibili */
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*D);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int qpos = pos_base + s;
            const float *qv = q + (int64_t)s*D + hh*hd;
            float sc[4096];
            for (int t = 0; t <= qpos; t++) {          /* causale: t <= qpos */
                const float *kv = m->K[layer] + ((int64_t)hh*m->max_t + t)*hd;
                float acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd]*kv[dd];
                sc[t] = acc * scale;
            }
            softmax_row(sc, qpos+1);
            float *cx = ctx + (int64_t)s*D + hh*hd;
            for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
            for (int t = 0; t <= qpos; t++) {
                const float *vrow = m->V[layer] + ((int64_t)hh*m->max_t + t)*hd;
                float a = sc[t];
                for (int dd = 0; dd < hd; dd++) cx[dd] += a * vrow[dd];
            }
        }
    }
    (void)Tk;
    matmul(out, ctx, l->o, S, D, D);
    free(q); free(k); free(vv); free(ctx);
}

/* MoE sui token x[S,hidden] -> out[S,hidden] */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S*E);
    matmul(logits, x, l->gate, S, D, E);
    memset(out, 0, (int64_t)S*D*sizeof(float));
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        if (m->momentum_logits && m->pilot_smooth > 0.f) {
            float *ema = m->momentum_logits + (int64_t)layer * E;
            int is_zero = 1;
            for (int e = 0; e < E; e++) { if (ema[e] != 0.f) { is_zero = 0; break; } }
            if (is_zero) {
                for (int e = 0; e < E; e++) ema[e] = pr[e];
            } else {
                for (int e = 0; e < E; e++) {
                    ema[e] = (1.f - m->pilot_smooth) * pr[e] + m->pilot_smooth * ema[e];
                }
            }
        }

        softmax_row(pr, E);
        /* top-K indici (selezione parziale) */
        int idx[64]; float val[64];
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
                if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
            }
            /* SEC: all-NaN probabilities leave best at -1, which reaches
             * expert_get() and then last_access[layer*E - 1] -- a heap write at
             * a negative index. See rt_router_pick in route_trace.h. */
            best = rt_router_pick(best, kk, E, layer);
            idx[kk] = best; val[kk] = pr[best];
        }
        if (c->norm_topk) { float sm=0; for(int kk=0;kk<K;kk++) sm+=val[kk]; for(int kk=0;kk<K;kk++) val[kk]/=sm; }
        /* IMPROVEMENT 2: update activation heatmap (before pinning activates) */
        if (!m->hot_pinned && m->freq) {
            uint32_t *freq_l = m->freq[layer];
            if (freq_l) for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) freq_l[idx[kk]]++;
        }
        const float *xs = x + (int64_t)s*D;
        for (int kk = 0; kk < K; kk++) {
            Slot *e; expert_get(m, layer, idx[kk], &e);
#if defined(__AVX2__)
            /* FUSED3: same contract as matmul_q's IDOT fast branch (IDOT env,
             * dims %16==0, <=4096) — outside it the stock calls below run
             * unchanged. Exact integer arithmetic only: bit-identical output
             * (verified by memcmp in tests/bench_fused3.c). OFF by default. */
            static int idot_moe = -1;
            if (idot_moe < 0) { const char *ie = getenv("IDOT"); idot_moe = !(ie && *ie == '0'); }
            if (g_fused3 && idot_moe && D % 16 == 0 && D <= 4096 && I % 16 == 0 && I <= 4096) {
                matmul_q_idot_pair_v3(g, u, xs, e->g, e->gs, e->u, e->us, D, I);   /* gate+up share one quant of xs */
                for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
                matmul_q_idot_v3(hh, g, e->d, e->ds, I, D);                        /* down_proj [D,I] */
            } else
#endif
            {
            matmul_q(g, xs, e->g, e->gs, D, I);     /* gate_proj [I,D] */
            matmul_q(u, xs, e->u, e->us, D, I);     /* up_proj   [I,D] */
            for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
            matmul_q(hh, g, e->d, e->ds, I, D);     /* down_proj [D,I] */
            }
            float w = val[kk];
            float *os = out + (int64_t)s*D;
            for (int d = 0; d < D; d++) os[d] += w * hh[d];
        }
    }
    free(logits); free(g); free(u); free(hh);
}

static void layers_forward_range(Model *m, float *x, int S, int pos_base,
                                 int layer_begin, int layer_end,
                                 int allow_prefetch) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = layer_begin; i < layer_end; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        /* IMPROVEMENT 1: PILOT=1 -> 1-layer lookahead */
        if (allow_prefetch && g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        moe(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];

        /* PREDICTION IMPROVEMENT C (Residual gate trick):
         * PILOT=2 -> prefetch layer i+2 using completed state x (containing MoE residual). */
        if (allow_prefetch && g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
        if (allow_prefetch && g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
        
    }
    free(nrm); free(tmp);
}

static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    if (g_pilot && m->token_count > 0) {
        /* Flush stale prefetch requests: clear is_queued so pilot_realload
         * will skip any entries still sitting in pilot_q for the previous
         * token.  We deliberately do NOT move pilot_w backwards; that would
         * break the ring-buffer invariant (pilot_r could exceed pilot_w if
         * the worker consumed an entry concurrently).  The worker will drain
         * the stale slots harmlessly because pilot_realload already exits
         * early when the expert is already cached or is_queued is clear. */
        pthread_mutex_lock(&g_pilot_mx);
        memset(m->is_queued, 0, (size_t)c->n_layers * c->n_experts);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    layers_forward_range(m, x, S, pos_base, 0, c->n_layers, 1);
    /* count actual tokens processed (S>1 during prefill) */
    m->token_count += S; m->freq_token_count += S;
    if (!m->hot_pinned && m->hot_n > 0 && m->freq_token_count >= m->warmup_tokens)
        pin_hot_experts(m);
    m->kv_len = pos_base + S;
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, last, m->lm_head, 1, D, c->vocab);
    free(x); free(last);
    return logit;
}

static void pilot_realload(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer];
    Cfg *c = &m->c;

    pthread_mutex_lock(&g_pilot_mx);
    /* Early-exit if entry was flushed (is_queued cleared) while waiting. */
    if (!m->is_queued[layer * c->n_experts + eid]) {
        pthread_mutex_unlock(&g_pilot_mx);
        return;
    }
    if (slot_indexed(m, layer, eid)) {
        m->is_queued[layer * c->n_experts + eid] = 0;
        pthread_mutex_unlock(&g_pilot_mx);
        return;
    }
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        slot_ensure_allocated(m, s);
    } else {
        /* LRU eviction — skip pinned and in-flight (eid==-1) slots */
        int lru = -1;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
        }
        if (lru < 0) {
            m->is_queued[layer * c->n_experts + eid] = 0;
            pthread_mutex_unlock(&g_pilot_mx);
            return; /* all pinned/in-flight, skip */
        }

        /* LFRU eviction guard: don't displace a warm resident expert with a speculation */
        if (g_pilot_evict_guard && m->freq && m->freq[layer] && m->last_access &&
            lc->slots[lru].eid >= 0) {
            int vid = lc->slots[lru].eid;
            uint64_t vs = lfru_score(m->freq[layer][vid], m->last_access[layer * c->n_experts + vid], m->clock);
            uint64_t cs = lfru_score(m->freq[layer][eid], m->last_access[layer * c->n_experts + eid], m->clock);
            if (cs <= vs + (vs >> 2) + (4u << 8)) {
                m->is_queued[layer * c->n_experts + eid] = 0;
                pthread_mutex_unlock(&g_pilot_mx);
                return; /* drop speculation */
            }
        }

        s = &lc->slots[lru]; s->pinned = 0;
    }
    cache_hide(m, layer, s); s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);

    load_expert_merged(m, layer, eid, s);

    pthread_mutex_lock(&g_pilot_mx);
    cache_publish(m, layer, s, eid);
    s->pinned = m->is_pinned[layer * c->n_experts + eid];
    s->used = ++m->clock;
    if (m->last_access) m->last_access[layer * c->n_experts + eid] = m->clock;
    m->is_queued[layer * c->n_experts + eid] = 0;
    pthread_mutex_unlock(&g_pilot_mx);
}

static void *pilot_worker(void *arg) {
    (void)arg;
    while (1) {
        unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
        unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE);
        if (r == w) {
            sleep_ms(1);
            continue;
        }
        int layer = pilot_q[r & 4095].l;
        int eid = pilot_q[r & 4095].e;
        pilot_realload(pilot_m, layer, eid);
        __atomic_store_n(&pilot_r, r + 1, __ATOMIC_RELEASE);
    }
    return NULL;
}

static void pilot_prefetch(Model *m, int lnext, const float *x, int S) {
    if (lnext < 0 || lnext >= m->c.n_layers) return;
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts;
    ensure_pilot_worker_started(m);
    float *logits = falloc((int64_t)S * E);
    Layer *l = &m->L[lnext];

    // PREDICTION IMPROVEMENT B: Apply RMSNorm to x using destination layer's post_ln
    // This scales inputs to the distribution expected by l->gate.
    float *nrm_x = falloc((int64_t)S * D);
    for (int s = 0; s < S; s++) {
        rmsnorm_row(nrm_x + (int64_t)s * D, x + (int64_t)s * D, l->post_ln, D, c->eps);
    }

    matmul(logits, nrm_x, l->gate, S, D, E);
    free(nrm_x);

    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s * E;

        // PREDICTION IMPROVEMENT A: Apply routing momentum (EMA of gate logits)
        float *blended = pr;
        float *ema = m->momentum_logits + (int64_t)lnext * E;
        if (m->pilot_smooth > 0.f) {
            blended = falloc(E);
            int is_zero = 1;
            for (int e = 0; e < E; e++) { if (ema[e] != 0.f) { is_zero = 0; break; } }
            if (is_zero) {
                for (int e = 0; e < E; e++) {
                    ema[e] = pr[e];
                    blended[e] = pr[e];
                }
            } else {
                for (int e = 0; e < E; e++) {
                    blended[e] = (1.f - m->pilot_smooth) * pr[e] + m->pilot_smooth * ema[e];
                    ema[e] = blended[e]; // update EMA
                }
            }
        }

        int cand = 0;
        int idx[128];

        float max_logit = -1e30f;
        for (int e = 0; e < E; e++) { if (blended[e] > max_logit) max_logit = blended[e]; }
        float *exps = falloc(E);
        float sum_exps = 0.f;
        for (int e = 0; e < E; e++) {
            exps[e] = expf(blended[e] - max_logit);
            sum_exps += exps[e];
        }

        float cum_sum = 0.f;
        int min_cand = c->topk;
        int max_cand = c->topk * g_wide;
        if (max_cand < min_cand) max_cand = min_cand;
        if (max_cand > 128) max_cand = 128; /* idx[] buffer bound */
        if (max_cand > E) max_cand = E;

        for (int kk = 0; kk < max_cand; kk++) {
            int best = -1; float bv = -1.f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (idx[j] == e) { taken=1; break; }
                if (!taken && exps[e] > bv) { bv = exps[e]; best = e; }
            }
            if (best < 0) break;
            idx[kk] = best;
            cum_sum += bv;
            cand++;
            if (cum_sum >= m->pilot_conf_limit * sum_exps && cand >= min_cand) {
                break;
            }
        }
        free(exps);

        if (blended != pr) free(blended);

        /* IMPROVEMENT 5: sort candidates by eid for sequential SSD read locality */
        for (int a = 0; a < cand-1; a++)
            for (int b = a+1; b < cand; b++)
                if (idx[b] >= 0 && (idx[a] < 0 || idx[a] > idx[b])) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }

        for (int kk = 0; kk < cand; kk++) {
            int eid = idx[kk];
            if (eid < 0) continue;
            int found = 0;
            pthread_mutex_lock(&g_pilot_mx);
            found = slot_indexed(m, lnext, eid) != NULL;
            pthread_mutex_unlock(&g_pilot_mx);
            if (!found) {
                int gidx = lnext * E + eid;
                pthread_mutex_lock(&g_pilot_mx);
                int already_queued = m->is_queued[gidx];
                if (!already_queued) {
                    m->is_queued[gidx] = 1;
                }
                pthread_mutex_unlock(&g_pilot_mx);

                if (!already_queued) {
                    unsigned w2 = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                    unsigned r2 = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                    if (w2 - r2 < 4096) {
                        pilot_q[w2 & 4095].l = lnext;
                        pilot_q[w2 & 4095].e = eid;
                        __atomic_store_n(&pilot_w, w2 + 1, __ATOMIC_RELEASE);
                    } else {
                        pthread_mutex_lock(&g_pilot_mx);
                        m->is_queued[gidx] = 0;
                        pthread_mutex_unlock(&g_pilot_mx);
                    }
                }
            }
        }
    }
    free(logits);
}


/* generazione greedy. prompt[np] -> riempie out[np+n_new] */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
    m->max_t = np + n_new;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0);          /* PREFILL */
    int len = np;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1);          /* DECODE */
    }
}

/* teacher-forced NLL of full_ids[np..nfull): feed the REFERENCE token at each step
 * (never the argmax), accumulate -log softmax(logits)[next_ref]. A loss meter for
 * throughput experiments: same engine path as decode, so hit rate/speed stay
 * comparable, but quality is measured as perplexity instead of exact-match.
 * Cross-checked vs HF transformers bf16 on identical token ids: engine (int8
 * experts) 12.11 ppl vs reference 12.25 (#108). Enabled by PPL=1. */
static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    m->max_t = nfull;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }
    double nll = 0; int scored = 0;
    float *logit = step(m, full, np, 0);              /* prefill on the prompt */
    for (int i = np; i < nfull; i++) {
        /* log softmax(logit)[full[i]] without materializing the softmax */
        float mx = logit[0]; for (int v = 1; v < c->vocab; v++) if (logit[v] > mx) mx = logit[v];
        double Z = 0; for (int v = 0; v < c->vocab; v++) Z += exp((double)logit[v] - mx);
        nll += -((double)logit[full[i]] - mx - log(Z));
        scored++;
        free(logit); logit = NULL;
        if (i == nfull - 1) break;
        logit = step(m, &full[i], 1, i);              /* teacher forcing */
    }
    if (logit) free(logit);
    *nll_out = nll / scored;
    return scored;
}

/* ---------- interactive chat mode (CHAT=1) ---------- */
/* OLMoE-Instruct's real template (tokenizer_config.json's chat_template):
 *   {{bos_token}}<|user|>\n{msg}\n<|assistant|>\n{reply}{eos_token}\n<|user|>\n...
 * bos_token == eos_token == "|||IP_ADDRESS|||" (a genuine OLMoE tokenizer quirk,
 * a PII-scrubbing artifact repurposed as the BOS/EOS marker — not a bug here).
 * It's an added special token, so tok_encode() tokenizes the literal string as
 * one atomic id, same as any other added token. <|user|>/<|assistant|> are NOT
 * special tokens in this tokenizer — just plain text the model was trained to
 * treat as turn markers.
 * Turn 1 gets bos_token with no separator before "<|user|>"; every later turn
 * gets eos_token+"\n" first (closing the previous assistant turn we never
 * explicitly appended to history, matching the is_stop-skips-append design
 * below) before its own "<|user|>...". */
static int fmt_user_turn(char *out, int cap, const char *msg, int first_turn) {
    int n = first_turn
        ? snprintf(out, cap, "|||IP_ADDRESS|||<|user|>\n%s\n<|assistant|>\n", msg)
        : snprintf(out, cap, "|||IP_ADDRESS|||\n<|user|>\n%s\n<|assistant|>\n", msg);
    return (n < 0 || n >= cap) ? -1 : n;
}

/* KV cache allocated ONCE for ctx_cap and never reallocated — hist_len only
 * grows (or resets to 0 on /reset), so every turn after the first reuses the
 * previous turns' cached keys/values: real multi-turn context, not a fresh
 * generate() call per message. ctx_cap capped at 4096: attention()'s per-head
 * score buffer (sc[4096]) is fixed-size, any position >= 4096 would overflow it. */
static void run_chat(Model *m, Tok *T, int ctx_cap) {
    Cfg *c = &m->c;
    m->max_t = ctx_cap;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }

    int tok_eos = tok_id_of(T, "|||IP_ADDRESS|||");
    stops_arm_tok(c, tok_eos, T);

    int max_new = getenv("MAX_NEW") ? atoi(getenv("MAX_NEW")) : 512;
    if (max_new < 1) max_new = 1;

    int *hist = malloc((size_t)ctx_cap * sizeof(int));
    int hist_len = 0;
    int first_turn = 1;

    char *line = malloc(8192);
    char *turn = malloc(8192 + 64);
    int  *newids = malloc(8192 * sizeof(int));
    int  *gen = malloc((size_t)max_new * sizeof(int));
    char *outbuf = malloc(65536);

    fprintf(stderr, "olmoe chat — SNAP=%s, ctx=%d, TEMP=%.2f, NUCLEUS=%.2f\n"
                     "  type a message and press enter; /reset clears context; Ctrl-D exits\n",
            getenv("SNAP"), ctx_cap, g_temp, g_nuc);

    for (;;) {
        printf("\n> "); fflush(stdout);
        if (!fgets(line, 8192, stdin)) break;
        size_t L = strlen(line);
        while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (L == 0) continue;
        if (!strcmp(line, "/reset")) { hist_len = 0; first_turn = 1; fprintf(stderr, "[chat] context reset\n"); continue; }

        int tn = fmt_user_turn(turn, 8192 + 64, line, first_turn);
        if (tn < 0) { fprintf(stderr, "[chat] message too long, skipped\n"); continue; }
        first_turn = 0;
        int nn = tok_encode(T, turn, tn, newids, 8192);

        if (hist_len + nn + max_new > ctx_cap) {
            fprintf(stderr, "[chat] context window full (%d/%d tokens) — /reset to start over\n",
                    hist_len + nn, ctx_cap);
            continue;
        }

        float *logit = step(m, newids, nn, hist_len);
        hist_len += nn;

        /* Every token counted in hist_len must have gone through step() exactly
         * once (that's what writes its KV-cache slot) — otherwise a later turn's
         * attention would read an uninitialized slot. So even on the turn's LAST
         * token we still call step() once to populate its KV before breaking;
         * only its returned logit (which nothing will consume) is discarded. */
        int ngen = 0;
        for (int s = 0; s < max_new; s++) {
            int nt = pick_tok(logit, c->vocab, -1);
            free(logit); logit = NULL;
            if (is_stop(nt)) break;
            hist[hist_len] = nt; gen[ngen++] = nt; hist_len++;
            int room_left = (s < max_new - 1) && (hist_len < ctx_cap);
            logit = step(m, &nt, 1, hist_len - 1);
            if (!room_left) {
                if (hist_len >= ctx_cap) fprintf(stderr, "\n[chat] context window full mid-reply — /reset to start over\n");
                free(logit); logit = NULL;
                break;
            }
        }

        int outn = tok_decode(T, gen, ngen, outbuf, 65535);
        outbuf[outn] = 0;
        printf("%s\n", outbuf);
        fflush(stdout);
    }
    free(line); free(turn); free(newids); free(gen); free(outbuf); free(hist);
}

/* ---------- serve mode: openai_server.py engine protocol ----------
 * stdin:  SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>\n<payload>\n
 *         CANCEL <id>\n
 * stdout: READY sentinel once loaded, then per request a stream of
 *         DATA <id> <size>\n<bytes>\n frames and a final
 *         DONE <id> STAT <tok> <tps> <hit%> <rss> <prompt_tok> <len_limited>\n
 * Byte-identical to colibri.c's serve protocol (inkling.c documents it in
 * full above its own SUBMIT handling) so the shared openai_server.py gateway
 * drives olmoe unchanged.
 *
 * v1 scope, same as Inkling's first serve mode: one request in flight, full
 * re-prefill every turn, no cross-request KV reuse. The payload arrives
 * already rendered by openai_server.py's render_chat_olmoe (bos/eos turn
 * markers and all) -- this engine tokenizes it as-is, the same way run_chat()
 * feeds fmt_user_turn()'s output to tok_encode() above. Because nothing here
 * persists state across requests, a fresh prefill at pos_base=0 is enough to
 * start clean: attention() only ever reads positions [0, kv_len), so the
 * previous request's leftover K/V contents past the new prompt's length are
 * never touched, the same invariant CHAT mode's /reset already relies on
 * (it clears hist_len, not the K/V buffers themselves). */

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen; } SReq;
#define SRV_QMAX 16
static SReq g_q[SRV_QMAX]; static int g_qn = 0;
static const ColiServeWireProfile olmoe_wire = {
    .max_header_bytes = 511,
    .max_payload_bytes = 1u << 22,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 0,
};

/* read one control line (+ payload for SUBMIT). cur_id: request in flight;
 * returns 1 if that request was cancelled, 0 otherwise, -1 on input EOF. */
static int serve_read_cmd(FILE *in, FILE *out, const char *cur_id) {
    ColiServeCommand command;
    ColiServeReadResult result = coli_serve_read_command(in, &olmoe_wire, &command);
    if (result == COLI_SERVE_READ_EOF || result == COLI_SERVE_READ_BAD_FRAME) return -1;
    if (result == COLI_SERVE_READ_NOMEM) {
        coli_serve_write_error(out, command.id, "out of memory");
        return -1;
    }
    if (result == COLI_SERVE_READ_BAD_REQUEST &&
        command.kind == COLI_SERVE_COMMAND_SUBMIT) {
        coli_serve_write_error(out, command.id, "bad submit header");
        return -1;
    }
    if (result != COLI_SERVE_READ_OK) return 0;
    if (command.kind == COLI_SERVE_COMMAND_CANCEL) {
        int cancelled = cur_id && !strcmp(command.id, cur_id);
        coli_serve_command_dispose(&command);
        return cancelled;
    }
    if (command.kind == COLI_SERVE_COMMAND_SUBMIT) {
        if (g_qn < SRV_QMAX) {
            SReq *q = &g_q[g_qn++];
            snprintf(q->id, sizeof(q->id), "%s", command.id);
            q->max_tok = command.max_tokens;
            q->temp = command.temperature;
            q->top_p = command.top_p;
            q->payload = (char *)coli_serve_command_take_payload(&command);
            q->plen = (int)command.payload_bytes;
        } else {
            coli_serve_write_error(out, command.id, "queue full");
        }
    }
    coli_serve_command_dispose(&command);
    return 0;
}

static int serve_one(Model *m, Tok *T, SReq *q, int ctx_cap) {
    Cfg *c = &m->c;
    int cap = q->plen + 16;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { coli_serve_write_error(stdout, q->id, "empty prompt"); free(ids); return 0; }
    if (np + q->max_tok > ctx_cap) {
        char message[128];
        snprintf(message, sizeof(message), "context exceeds CTX (%d + %d > %d)",
                 np, q->max_tok, ctx_cap);
        coli_serve_write_error(stdout, q->id, message); free(ids); return 0;
    }
    g_temp = q->temp; g_nuc = q->top_p;
    double t0 = now_s();
    uint64_t h0 = m->hits, m0 = m->miss;
    float *logit = step(m, ids, np, 0);
    int hist_len = np, gen = 0, limited = 1, cancelled = 0;
    char buf[512];
    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        int nt = pick_tok(logit, c->vocab, -1);
        free(logit); logit = NULL;
        if (is_stop(nt)) { limited = 0; break; }
        int nb = tok_decode(T, &nt, 1, buf, sizeof(buf)-1);
        coli_serve_write_data(stdout, q->id, buf, (size_t)nb);
        gen++; hist_len++;
        while (coli_stdin_readable()) {
            int r = serve_read_cmd(stdin, stdout, q->id);
            if (r < 0) { free(ids); return -1; }
            if (r > 0) { cancelled = 1; limited = 0; }
        }
        /* Unlike run_chat(), we do not step() the final token just to populate
         * its KV slot: nothing in serve mode reads past this request's own
         * reply, so a discarded logit here costs nothing. */
        if (cancelled || s == q->max_tok - 1 || hist_len >= ctx_cap) break;
        logit = step(m, &nt, 1, hist_len - 1);
    }
    free(logit);
    double dt = now_s() - t0;
    double tot = (double)(m->hits - h0 + m->miss - m0);
    ColiServeDone done = {
        .completion_tokens = gen,
        .tokens_per_second = dt > 0 ? gen/dt : 0.0,
        .cache_hit_percent = tot ? 100.0*(m->hits-h0)/tot : 0.0,
        .rss_gb = rss_gb(),
        .prompt_tokens = np,
        .length_limited = limited,
    };
    coli_serve_write_done(stdout, q->id, &done);
    /* PROF: per-turn phase timings for the dashboard. olmoe.c does not split
     * its wall time into fill/expert/shared/attn phases the way glm.c and
     * inkling.c do, so this reports total time only; a real phase breakdown
     * is future work, not a protocol requirement. */
    printf("PROF %.3f %d %d 0.0 0.0 0.0 0.0 0.0 %d\n", dt, np, gen, gen + 1);
    fflush(stdout);
    free(ids);
    return 0;
}

/* dashboard HWINFO/TIERS/EMAP: same lines the other serve-capable engines
 * emit for the web dashboard's hardware panel and Brain page. olmoe.c is
 * CPU-only (no CUDA/Metal backend), so the GPU fields are always empty, and
 * every layer is a MoE layer (no dense/sparse split like GLM-5.2), so the
 * tier scan below runs over all n_layers unconditionally. */
static void serve_hwinfo(Model *m) {
    (void)m;
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
    printf("HWINFO %d %.1f %.1f 0 0.0 %s|\n", cores, rt, ra, cpu[0] ? cpu : "unknown");
    fflush(stdout);
}

static void serve_tiers_emap(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts;
    int filled = 0;
    for (int i = 0; i < c->n_layers; i++) filled += m->cache[i].n;
    int64_t I = c->inter, D = c->hidden;
    /* per-expert resident bytes: int8 gate/up/down + one f32 scale per row */
    int64_t slotb = 3*I*D + (2*I+D)*4;
    printf("TIERS 0 %d %d 0.00 %.2f\n", filled, c->n_layers*E - filled, filled*(double)slotb/1e9);
    /* EMAP: 1 byte/expert hex — tier(2b: 0=disk 1=RAM)<<6 | heat(6b: log2 usage) */
    char *hex = malloc((size_t)c->n_layers*E*2 + 1); int w = 0;
    for (int i = 0; i < c->n_layers; i++) {
        LCache *lc = &m->cache[i];
        for (int e = 0; e < E; e++) {
            int tier = 0;
            for (int z = 0; z < lc->n; z++) if (lc->slots[z].eid == e) { tier = 1; break; }
            uint32_t u = m->freq[i] ? m->freq[i][e] : 0;
            int heat = 0; while (u) { heat++; u >>= 1; } if (heat > 63) heat = 63;
            int b = (tier << 6) | heat;
            hex[w++] = "0123456789abcdef"[b >> 4];
            hex[w++] = "0123456789abcdef"[b & 15];
        }
    }
    hex[w] = 0;
    printf("EMAP %d %d %s\n", c->n_layers, E, hex);
    fflush(stdout); free(hex);
}

static void serve_loop(Model *m, Tok *T, int ctx_cap) {
    coli_serve_stdio_init();
    int tok_eos = tok_id_of(T, "|||IP_ADDRESS|||");
    stops_arm_tok(&m->c, tok_eos, T);
    coli_serve_write_ready(stdout, rss_gb());
    serve_hwinfo(m);
    serve_tiers_emap(m);
    for (;;) {
        while (!g_qn) if (serve_read_cmd(stdin, stdout, NULL) < 0) return;
        SReq q = g_q[0];
        memmove(g_q, g_q + 1, (size_t)(--g_qn) * sizeof(SReq));
        int fatal = serve_one(m, T, &q, ctx_cap);
        free(q.payload);
        if (fatal < 0) return;
    }
}

/* ---------- lettura ref.json ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

#ifndef OLMOE_NO_MAIN
int main(int argc, char **argv) {
    coli_omp_tune_threads("olmoe");   /* squadra sui core fisici, niente spin-wait: vedi omp_tune.h */
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    g_wide  = getenv("WIDE")  ? atoi(getenv("WIDE"))  : 1;
    g_pilot_evict_guard = getenv("PILOT_EVICT_GUARD") ? atoi(getenv("PILOT_EVICT_GUARD")) : 1;
    g_expert_drop = getenv("EXPERT_DROP") ? atoi(getenv("EXPERT_DROP")) : 0;
    g_fused3     = getenv("FUSED3") ? atoi(getenv("FUSED3")) : 0;
    if (g_wide < 1) g_wide = 1;
    if (g_wide > 4) g_wide = 4;
    int hot_n  = getenv("HOT")   ? atoi(getenv("HOT"))   : 0;
    int cap    = argc > 1 ? atoi(argv[1]) : 16;
    int bits   = argc > 2 ? atoi(argv[2]) : 8;
    if (bits < 2 || bits > 8) {
        fprintf(stderr, "quant_bits must be 2..8 (got %d)\n", bits);
        return 1;
    }

    /* SERVE=1: openai_server.py drives the engine over stdin/stdout (READY
     * handshake, SUBMIT/CANCEL, DATA/DONE/PROF frames, HWINFO/TIERS/EMAP for
     * the dashboard) — same protocol as colibri.c/inkling.c/kimi_k3.c. v1:
     * one request at a time, full re-prefill every turn (see serve_one()). */
    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        int ctx_cap = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
        if (ctx_cap < 1 || ctx_cap > 4096) {   /* attention()'s sc[4096] score buffer hard-caps this */
            fprintf(stderr, "CTX must be 1..4096 (got %d)\n", ctx_cap);
            return 1;
        }
        Model m; model_init(&m, snap, cap, bits);
        m.max_t = ctx_cap;
        m.K = calloc(m.c.n_layers, sizeof(float*)); m.V = calloc(m.c.n_layers, sizeof(float*));
        for (int i = 0; i < m.c.n_layers; i++) {
            m.K[i] = falloc((int64_t)m.c.n_heads * m.max_t * m.c.head_dim);
            m.V[i] = falloc((int64_t)m.c.n_heads * m.max_t * m.c.head_dim);
        }
        Tok T;
        char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
        tok_load(&T, tokpath);
        serve_loop(&m, &T, ctx_cap);
        { const char *up = getenv("COLI_USAGE");
          if (up && *up) rt_save(up, 0); }
        return 0;
    }

    if (getenv("CHAT")) {   /* interactive mode: bypasses the ref.json harness entirely */
        /* #509 convention, same as the GLM engine: COLI_TEMP is the primary channel;
         * TEMP stays a legacy alias ONLY when it is fully numeric — on Windows and under
         * ROCm stacks %TEMP% names a directory, and atof("C:\...") == 0.0 would silently
         * force greedy decoding for every olmoe chat on those hosts. */
        if (getenv("COLI_TEMP")) g_temp = (float)atof(getenv("COLI_TEMP"));
        else if (getenv("TEMP") && *getenv("TEMP")) {
            char *tend; double tv = strtod(getenv("TEMP"), &tend);
            if (tend != getenv("TEMP") && *tend == '\0') g_temp = (float)tv;
        }
        g_nuc  = getenv("NUCLEUS") ? (float)atof(getenv("NUCLEUS")) : g_nuc;
        int ctx_cap = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
        if (ctx_cap < 1 || ctx_cap > 4096) {   /* attention()'s sc[4096] score buffer hard-caps this */
            fprintf(stderr, "CTX must be 1..4096 (got %d)\n", ctx_cap);
            return 1;
        }
        Model m; model_init(&m, snap, cap, bits);
        printf("resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());
        Tok T;
        char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
        tok_load(&T, tokpath);
        run_chat(&m, &T, ctx_cap);
        { const char *up = getenv("COLI_USAGE");
          if (up && *up) rt_save(up, 0); }
        return 0;
    }

    const char *refpath = argc > 3 ? argv[3] : "ref.json";

    float smooth = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    float conf   = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;

    printf("== Streaming C engine v2.2 | cache=%d/layer bits=%d pilot=%d wide=%d guard=%d hot=%d smooth=%.2f conf=%.2f fused3=%d ==\n",
           cap, bits, g_pilot, g_wide, g_pilot_evict_guard, hot_n, smooth, conf, g_fused3);

    FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if (fread(buf,1,n,f)!=(size_t)n) {} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull; int *prompt = read_int_array(ref,"prompt_ids",&np); int *full = read_int_array(ref,"full_ids",&nfull);
    int n_new = nfull - np;

    Model m; model_init(&m, snap, cap, bits);
    printf("resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());

    if (getenv("PPL") && atoi(getenv("PPL")) == 1) {   /* loss-meter mode: teacher-forced NLL */
        double nll; double t = now_s();
        int scored = tf_nll(&m, full, nfull, np, &nll);
        double dt = now_s() - t;
        double tot = m.hits + m.miss;
        printf("TF-NLL: %.4f nats/token over %d tokens  |  ppl = %.2f\n", nll, scored, exp(nll));
        printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss);
        printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", scored/dt, dt, scored, rss_gb());
        free(buf); free(arena);
        return 0;      /* PPL is a measurement run: no rt_save on purpose, so a loss
                        * sweep cannot fold its own tokens into the persisted ranking */
    }

    int *out = malloc((np + n_new) * sizeof(int));
    double t = now_s();
    generate(&m, prompt, np, n_new, out);
    double dt = now_s() - t;

    int match = 0;
    printf("\nReference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double tot = m.hits + m.miss;
    printf("\nPEAK RSS: %.2f GB\n", rss_gb());
    printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);


    // Persistent Hot Pinning: save dynamic pinning if newly created
    if (m.hot_pinned) {
        char pinpath[512];
        snprintf(pinpath, sizeof(pinpath), "%s/hot_pinned.bin", snap);
        FILE *pinf_chk = fopen(pinpath, "rb");
        if (!pinf_chk) {
            FILE *pinf_save = fopen(pinpath, "wb");
            if (pinf_save) {
                size_t expected_size = (size_t)m.c.n_layers * m.c.n_experts;
                fwrite(m.is_pinned, 1, expected_size, pinf_save);
                fclose(pinf_save);
                printf("[HOT] Saved persistent pinning to %s\n", pinpath);
            }
        } else {
            fclose(pinf_chk);
        }
    }

    { const char *up = getenv("COLI_USAGE");
      if (up && *up) rt_save(up, 0); }              /* same bytes as every other engine */
    printf("Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    /* One line, every engine, one format: `coli tune` sweeps scheduling knobs and
     * needs tokens-and-elapsed to compare candidates. Before this only colibri
     * emitted a parseable throughput line (REPLAY decode), so the tuner was
     * GLM-only and bannered the right model while launching the wrong engine
     * (#898). Printed to stdout, which is what autotune captures.
     * Tokens and seconds, not tok/s: the ratio is derived by the caller at full
     * precision (#852 -- two decimals of tok/s is one significant digit at the
     * rates this engine runs at). */
    printf("TUNE decode: %d tokens in %.3fs\n", n_new, dt);
    free(buf); free(arena);
    return 0;
}
#endif /* OLMOE_NO_MAIN */

#ifdef COLI_SEGMENT_ADAPTER
/* ---------- engine-owned Segment adapter ------------------------------ */

typedef struct {
    Model model;
    uint32_t layer_begin, layer_end, context_tokens;
    pthread_mutex_t run_lock;
} OlmoeSegmentEngine;

typedef struct {
    OlmoeSegmentEngine *engine;
    float **K, **V;
    uint32_t context_tokens, position;
} OlmoeSegmentSession;

static void olmoe_segment_model_destroy(OlmoeSegmentEngine *engine) {
    if (!engine) return;
    Model *model = &engine->model;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        Layer *weights = &model->L[layer];
        free(weights->in_ln); free(weights->post_ln);
        free(weights->q); free(weights->k); free(weights->v); free(weights->o);
        free(weights->qn); free(weights->kn); free(weights->gate);
        LCache *cache = &model->cache[layer];
        for (int slot = 0; slot < cache->n; slot++) {
            free(cache->slots[slot].g);
            free(cache->slots[slot].gs);
        }
        free(cache->slot_by_expert);
        free(cache->slots);
    }
    free(model->last_access); free(model->is_queued); free(model->is_pinned);
    free(model->freq);
    free(model->momentum_logits);
    free(model->cache); free(model->L);
    st_destroy(&model->S);
}

static int olmoe_segment_engine_open(
    void **engine_impl, ColiSegmentCapabilities *capabilities,
    const ColiSegmentEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid OLMoE Segment open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_SEGMENT_CAP_CPU))
        return coli_segment_adapter_error(error, error_size,
                                           "OLMoE Segment supports CPU only");
    if (options->context_tokens > 4096)
        return coli_segment_adapter_error(error, error_size,
                                           "OLMoE Segment context exceeds 4096");

    Cfg config;
    memset(&config, 0, sizeof(config));
    load_cfg(&config, options->model_dir);
    if (options->layer_end > (uint32_t)config.n_layers)
        return coli_segment_adapter_error(error, error_size,
                                           "OLMoE Segment range exceeds model");
    int range_layers = (int)(options->layer_end - options->layer_begin);
    int cap = 16;
    if (options->memory_limit_bytes) {
        uint64_t weights = (uint64_t)config.hidden * config.inter * 3u;
        uint64_t scales = (uint64_t)(config.inter * 2 + config.hidden) *
                          sizeof(float);
        uint64_t per_slot = weights + scales;
        uint64_t slots = per_slot && range_layers > 0
            ? options->memory_limit_bytes / per_slot / (uint64_t)range_layers
            : 0;
        cap = slots > (uint64_t)config.n_experts ? config.n_experts : (int)slots;
        if (cap < 1) cap = 1;
    }

    OlmoeSegmentEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory opening OLMoE Segment");
    engine->layer_begin = options->layer_begin;
    engine->layer_end = options->layer_end;
    engine->context_tokens = options->context_tokens;
    if (pthread_mutex_init(&engine->run_lock, NULL)) {
        free(engine);
        return coli_segment_adapter_error(error, error_size,
                                           "cannot initialize OLMoE Segment lock");
    }
    model_init_range(&engine->model, options->model_dir, cap, 8,
                     (int)options->layer_begin, (int)options->layer_end, 0, 0);

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_SNAPSHOT |
                          COLI_SEGMENT_CAP_RANGE_NATIVE |
                          COLI_SEGMENT_CAP_MULTI_SESSION |
                          COLI_SEGMENT_CAP_CPU;
    coli_segment_capability_string(capabilities->engine_id,
                                   sizeof(capabilities->engine_id), "olmoe");
    coli_segment_capability_string(capabilities->state_schema,
                                   sizeof(capabilities->state_schema),
                                   "olmoe/kv-f32-v1");
    coli_segment_capability_string(capabilities->numeric_class,
                                   sizeof(capabilities->numeric_class),
                                   "olmoe/f32-int8/cpu-v1");
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = (uint32_t)config.hidden;
    capabilities->max_batch_rows = 128;
    capabilities->max_context_tokens = 4096;
    capabilities->num_layers = (uint32_t)config.n_layers;
    *engine_impl = engine;
    return 0;
}

static void olmoe_segment_engine_destroy(void *engine_impl) {
    OlmoeSegmentEngine *engine = (OlmoeSegmentEngine *)engine_impl;
    if (!engine) return;
    olmoe_segment_model_destroy(engine);
    pthread_mutex_destroy(&engine->run_lock);
    free(engine);
}

static int olmoe_segment_session_create(
    void *engine_impl, void **session_impl,
    const ColiSegmentSessionOptions *options, char *error, size_t error_size) {
    OlmoeSegmentEngine *engine = (OlmoeSegmentEngine *)engine_impl;
    if (!engine || !session_impl || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid OLMoE Segment session");
    *session_impl = NULL;
    OlmoeSegmentSession *session = calloc(1, sizeof(*session));
    if (!session)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory creating OLMoE session");
    session->engine = engine;
    session->context_tokens = options->context_tokens;
    int layers = engine->model.c.n_layers;
    session->K = calloc((size_t)layers, sizeof(*session->K));
    session->V = calloc((size_t)layers, sizeof(*session->V));
    if (!session->K || !session->V) goto oom;
    size_t cells;
    if (coli_segment_size_mul((size_t)engine->model.c.n_heads,
                              options->context_tokens, &cells) ||
        coli_segment_size_mul(cells, (size_t)engine->model.c.head_dim,
                              &cells)) goto oom;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        session->K[layer] = calloc(cells, sizeof(float));
        session->V[layer] = calloc(cells, sizeof(float));
        if (!session->K[layer] || !session->V[layer]) goto oom;
    }
    *session_impl = session;
    return 0;

oom:
    if (session->K && session->V)
        for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
             layer++) {
            free(session->K[layer]); free(session->V[layer]);
        }
    free(session->K); free(session->V); free(session);
    return coli_segment_adapter_error(error, error_size,
                                       "out of memory allocating OLMoE KV");
}

static void olmoe_segment_session_destroy(void *session_impl) {
    OlmoeSegmentSession *session = (OlmoeSegmentSession *)session_impl;
    if (!session) return;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++) {
        free(session->K[layer]); free(session->V[layer]);
    }
    free(session->K); free(session->V); free(session);
}

static int olmoe_segment_session_run(void *session_impl,
                                     const ColiSegmentRunRequest *request,
                                     char *error, size_t error_size) {
    OlmoeSegmentSession *session = (OlmoeSegmentSession *)session_impl;
    if (!session || !request || request->position != session->position)
        return coli_segment_adapter_error(
            error, error_size, "OLMoE Segment requires contiguous positions");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return coli_segment_adapter_error(error, error_size,
                                           "OLMoE Segment run cancelled");
    OlmoeSegmentEngine *engine = session->engine;
    if (request->output != request->input)
        memcpy(request->output, request->input, request->input_bytes);
    pthread_mutex_lock(&engine->run_lock);
    Model *model = &engine->model;
    model->K = session->K; model->V = session->V;
    model->max_t = (int)session->context_tokens;
    model->kv_len = (int)session->position;
    layers_forward_range(model, (float *)request->output, (int)request->rows,
                         (int)request->position, (int)engine->layer_begin,
                         (int)engine->layer_end, 0);
    model->K = NULL; model->V = NULL; model->max_t = 0; model->kv_len = 0;
    pthread_mutex_unlock(&engine->run_lock);
    session->position += request->rows;
    return 0;
}

static int olmoe_segment_payload_size(const OlmoeSegmentSession *session,
                                      uint32_t position, size_t *bytes) {
    size_t cells = (size_t)(session->engine->layer_end -
                            session->engine->layer_begin);
    if (coli_segment_size_mul(cells, 2, &cells) ||
        coli_segment_size_mul(cells,
                              (size_t)session->engine->model.c.n_heads,
                              &cells) ||
        coli_segment_size_mul(cells, position, &cells) ||
        coli_segment_size_mul(cells,
                              (size_t)session->engine->model.c.head_dim,
                              &cells) ||
        coli_segment_size_mul(cells, sizeof(float), bytes)) return -1;
    return 0;
}

static uint64_t olmoe_segment_state_hash(const OlmoeSegmentSession *session) {
    uint64_t hash = COLI_SEGMENT_HASH_INIT;
    size_t row_bytes = (size_t)session->position *
                       session->engine->model.c.head_dim * sizeof(float);
    int heads = session->engine->model.c.n_heads;
    size_t stride = (size_t)session->context_tokens *
                    session->engine->model.c.head_dim;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++)
        for (int kv = 0; kv < 2; kv++) {
            float *state = kv ? session->V[layer] : session->K[layer];
            for (int head = 0; head < heads; head++)
                hash = coli_segment_hash_update(hash, state + head * stride,
                                                row_bytes);
        }
    return hash;
}

static int olmoe_segment_session_snapshot(
    void *session_impl, ColiSegmentWriteFn write_fn, void *write_user_data,
    char *error, size_t error_size) {
    OlmoeSegmentSession *session = (OlmoeSegmentSession *)session_impl;
    size_t payload_bytes;
    if (!session || olmoe_segment_payload_size(session, session->position,
                                               &payload_bytes))
        return coli_segment_adapter_error(error, error_size,
                                           "OLMoE snapshot size overflow");
    ColiSegmentSnapshotHeader header;
    coli_segment_snapshot_header_init(
        &header, "olmoe", session->engine->layer_begin,
        session->engine->layer_end, session->context_tokens, session->position,
        payload_bytes, olmoe_segment_state_hash(session));
    if (coli_segment_stream_write(write_fn, write_user_data, &header,
                                  sizeof(header), error, error_size)) return -1;
    size_t row_bytes = (size_t)session->position *
                       session->engine->model.c.head_dim * sizeof(float);
    int heads = session->engine->model.c.n_heads;
    size_t stride = (size_t)session->context_tokens *
                    session->engine->model.c.head_dim;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++)
        for (int kv = 0; kv < 2; kv++) {
            float *state = kv ? session->V[layer] : session->K[layer];
            for (int head = 0; head < heads; head++)
                if (coli_segment_stream_write(
                        write_fn, write_user_data, state + head * stride,
                        row_bytes, error, error_size)) return -1;
        }
    return 0;
}

static int olmoe_segment_session_restore(
    void *session_impl, ColiSegmentReadFn read_fn, void *read_user_data,
    char *error, size_t error_size) {
    OlmoeSegmentSession *session = (OlmoeSegmentSession *)session_impl;
    ColiSegmentSnapshotHeader header;
    if (!session || coli_segment_stream_read(read_fn, read_user_data, &header,
                                             sizeof(header), error, error_size))
        return -1;
    size_t payload_bytes;
    if (olmoe_segment_payload_size(session, header.position, &payload_bytes) ||
        coli_segment_snapshot_header_valid(
            &header, "olmoe", session->engine->layer_begin,
            session->engine->layer_end, session->context_tokens, payload_bytes,
            error, error_size)) return -1;
    unsigned char *payload = payload_bytes ? malloc(payload_bytes) : NULL;
    if (payload_bytes && !payload)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory restoring OLMoE KV");
    if (coli_segment_stream_read(read_fn, read_user_data, payload, payload_bytes,
                                 error, error_size)) {
        free(payload);
        return -1;
    }
    if (coli_segment_hash_update(COLI_SEGMENT_HASH_INIT, payload,
                                 payload_bytes) != header.payload_hash) {
        free(payload);
        return coli_segment_adapter_error(error, error_size,
                                           "OLMoE snapshot checksum mismatch");
    }
    size_t row_bytes = (size_t)header.position *
                       session->engine->model.c.head_dim * sizeof(float);
    int heads = session->engine->model.c.n_heads;
    size_t stride = (size_t)session->context_tokens *
                    session->engine->model.c.head_dim;
    unsigned char *cursor = payload;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++)
        for (int kv = 0; kv < 2; kv++) {
            float *state = kv ? session->V[layer] : session->K[layer];
            for (int head = 0; head < heads; head++) {
                memcpy(state + head * stride, cursor, row_bytes);
                cursor += row_bytes;
            }
        }
    session->position = header.position;
    free(payload);
    return 0;
}

static const ColiSegmentAdapter olmoe_segment_adapter = {
    sizeof(ColiSegmentAdapter), COLI_SEGMENT_ABI_VERSION, "olmoe",
    olmoe_segment_engine_open, olmoe_segment_engine_destroy,
    olmoe_segment_session_create, olmoe_segment_session_destroy,
    olmoe_segment_session_run, olmoe_segment_session_snapshot,
    olmoe_segment_session_restore, {0}
};

int coli_olmoe_segment_adapter_register(void) {
    return coli_segment_adapter_register(&olmoe_segment_adapter);
}
#endif /* COLI_SEGMENT_ADAPTER */
