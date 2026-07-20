/* Motore di inferenza OLMoE in C puro, con EXPERT-STREAMING dal disco.
 * Porting del motore Python (engine.py). Obiettivo Stadio A: produrre gli STESSI
 * token id del riferimento (ref.json) -> valida il core prima di scalare a GLM-5.2.
 *
 * Densa (embed, attn, router, norme, lm_head) residente in RAM (float32).
 * Expert letti dal disco on-demand via pread+fadvise(DONTNEED), cache LRU per-layer.
 * Matmul multi-thread con OpenMP (niente BLAS).
 *
 * ENV VARS:
 *   PILOT=0/1/2/3 : 0=no prefetch, 1=1-layer lookahead, 2=2-layer, 3=3-layer lookahead
 *   HOT=N         : pin top-N hot experts per layer permanently (never evict)
 *   WARMUP=N      : tokens before hot pinning activates (default 5)
 *   WIDE=N        : prefetch top-K*N candidates (default 1, try 2 or 3)
 *   SMOOTH=F      : EMA coefficient for routing momentum (default 0.3, range 0.0-0.95)
 *   CONF_LIMIT=F  : cumulative gate probability threshold for prefetch cutoff (default 0.92)
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
typedef struct { Slot *slots; int n, cap; } LCache;

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
    uint32_t *freq;
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
} Model;

static pthread_mutex_t g_pilot_mx = PTHREAD_MUTEX_INITIALIZER;
static struct { int l, e; } pilot_q[4096];
static volatile unsigned pilot_r = 0, pilot_w = 0;
static Model *pilot_m = NULL;
static int g_pilot = 0;
static int g_wide  = 1;  /* IMPROVEMENT 4: top-K * g_wide candidates prefetched */

static void pilot_prefetch(Model *m, int lnext, const float *x, int S);
static void *pilot_worker(void *arg);
static void ensure_pilot_worker_started(Model *m);
static void slot_ensure_allocated(Model *m, Slot *s);

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

/* y[S,O] = x[S,I] @ W^T,  W e' [O,I] row-major */
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

/* y[1,O] = x[1,I] @ W^T con W quantizzato: q[O,I] int8 + scala per riga.
 * W[o,i] ~= q[o,i]*scale[o]  ->  y[o] = scale[o] * sum_i x[i]*q[o,i].
 * Su ARM: attivazione quantizzata Q8_0 (scala per blocco di 16) + dot int8
 * NEON (sdot dove c'e' dotprod) — stessa famiglia IDOT di glm.c, IDOT=0 per
 * la via scalare byte-esatta. Misurato 2.7x end-to-end su M5. */
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
#endif
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
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
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

/* quantizza un weight f32 [O,I] -> int8 q[O,I] + scala[O], simmetrica per riga.
 * Replica quant_dequant() del Python: scale = amax(|w|, riga)/qmax, q = round(w/scale). */
static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1;     /* 8->127, 4->7, 2->1 */
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
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    c->hidden    = (int)json_get(r,"hidden_size")->num;
    c->n_layers  = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads   = (int)json_get(r,"num_attention_heads")->num;
    c->n_kv_heads= (int)json_get(r,"num_key_value_heads")->num;
    c->n_experts = (int)json_get(r,"num_experts")->num;
    c->topk      = (int)json_get(r,"num_experts_per_tok")->num;
    c->inter     = (int)json_get(r,"intermediate_size")->num;
    c->vocab     = (int)json_get(r,"vocab_size")->num;
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

static void model_init(Model *m, const char *snap, int cap, int bits) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    m->embed      = load_t(m, "model.embed_tokens.weight");
    m->lm_head    = load_t(m, "lm_head.weight");
    m->final_norm = load_t(m, "model.norm.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
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
    for (int i = 0; i < c->n_layers; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc(cap, sizeof(Slot));
    }
    /* IMPROVEMENT 2: frequency heatmap for hot expert pinning */
    m->freq = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint32_t));
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
    float cl = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;
    if (cl < 0.1f) cl = 0.1f; if (cl > 1.0f) cl = 1.0f;
    m->pilot_conf_limit = cl;
    m->dense_load_s = now_s() - t0;

    // Persistent Hot Pinning: try to load hot_pinned.bin
    char pinpath[512];
    snprintf(pinpath, sizeof(pinpath), "%s/hot_pinned.bin", snap);
    FILE *pinf = fopen(pinpath, "rb");
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
    st_read_raw(&m->S, nm, s->g, 1);
    st_read_f32(&m->S, qsnm, s->gs, 0);  /* scales are F32; use typed reader for dtype safety */
}

/* ---------- cache expert: ritorna i pesi quantizzati (q+scale) da cache o disco ---------- */
static void expert_get(Model *m, int layer, int eid, Slot **out) {
    LCache *lc = &m->cache[layer];
    pthread_mutex_lock(&g_pilot_mx);
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        m->hits++; lc->slots[i].used = ++m->clock; *out = &lc->slots[i];
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
        if (lru < 0) lru = 0; /* absolute last resort: all in-flight, evict slot 0 */
        s = &lc->slots[lru];
        s->pinned = 0;
    }
    s->eid = -1;
    s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);

    load_expert_merged(m, layer, eid, s);

    pthread_mutex_lock(&g_pilot_mx);
    s->eid = eid;
    s->pinned = m->is_pinned[layer * c->n_experts + eid];
    s->used = ++m->clock;
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
        uint32_t *freq_l = m->freq + (int64_t)l * c->n_experts;
        
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

            LCache *lc = &m->cache[l];
            int found = 0;
            pthread_mutex_lock(&g_pilot_mx);
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid == eid) { lc->slots[i].pinned = 1; found = 1; break; }
            }
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
            idx[kk] = best; val[kk] = bv;
        }
        if (c->norm_topk) { float sm=0; for(int kk=0;kk<K;kk++) sm+=val[kk]; for(int kk=0;kk<K;kk++) val[kk]/=sm; }
        /* IMPROVEMENT 2: update activation heatmap (before pinning activates) */
        if (!m->hot_pinned && m->freq) {
            uint32_t *freq_l = m->freq + (int64_t)layer * E;
            for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) freq_l[idx[kk]]++;
        }
        const float *xs = x + (int64_t)s*D;
        for (int kk = 0; kk < K; kk++) {
            Slot *e; expert_get(m, layer, idx[kk], &e);
            matmul_q(g, xs, e->g, e->gs, D, I);     /* gate_proj [I,D] */
            matmul_q(u, xs, e->u, e->us, D, I);     /* up_proj   [I,D] */
            for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
            matmul_q(hh, g, e->d, e->ds, I, D);     /* down_proj [D,I] */
            float w = val[kk];
            float *os = out + (int64_t)s*D;
            for (int d = 0; d < D; d++) os[d] += w * hh[d];
        }
    }
    free(logits); free(g); free(u); free(hh);
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
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        /* IMPROVEMENT 1: PILOT=1 -> 1-layer lookahead */
        if (g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        moe(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];

        /* PREDICTION IMPROVEMENT C (Residual gate trick):
         * PILOT=2 -> prefetch layer i+2 using completed state x (containing MoE residual). */
        if (g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
        if (g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
        
    }
    /* count actual tokens processed (S>1 during prefill) */
    m->token_count += S; m->freq_token_count += S;
    if (!m->hot_pinned && m->hot_n > 0 && m->freq_token_count >= m->warmup_tokens)
        pin_hot_experts(m);
    m->kv_len = pos_base + S;
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, last, m->lm_head, 1, D, c->vocab);
    free(x); free(nrm); free(tmp); free(last);
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
    for (int i = 0; i < lc->n; i++) {
        if (lc->slots[i].eid == eid) {
            m->is_queued[layer * c->n_experts + eid] = 0;
            pthread_mutex_unlock(&g_pilot_mx);
            return;
        }
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
        s = &lc->slots[lru]; s->pinned = 0;
    }
    s->eid = -1; s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);

    load_expert_merged(m, layer, eid, s);

    pthread_mutex_lock(&g_pilot_mx);
    s->eid = eid;
    s->pinned = m->is_pinned[layer * c->n_experts + eid];
    s->used = ++m->clock;
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
            LCache *lc = &m->cache[lnext];
            for (int z = 0; z < lc->n; z++) {
                if (lc->slots[z].eid == eid) { found = 1; break; }
            }
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

/* ---------- lettura ref.json ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    g_wide  = getenv("WIDE")  ? atoi(getenv("WIDE"))  : 1;
    if (g_wide < 1) g_wide = 1;
    if (g_wide > 4) g_wide = 4;
    int hot_n  = getenv("HOT")   ? atoi(getenv("HOT"))   : 0;
    int cap    = argc > 1 ? atoi(argv[1]) : 16;
    int bits   = argc > 2 ? atoi(argv[2]) : 8;
    if (bits < 2 || bits > 8) {
        fprintf(stderr, "quant_bits must be 2..8 (got %d)\n", bits);
        return 1;
    }
    const char *refpath = argc > 3 ? argv[3] : "ref.json";

    float smooth = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    float conf   = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;

    printf("== Streaming C engine v2.2 | cache=%d/layer bits=%d pilot=%d wide=%d hot=%d smooth=%.2f conf=%.2f ==\n",
           cap, bits, g_pilot, g_wide, hot_n, smooth, conf);

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
        return 0;
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

    printf("Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    free(buf); free(arena);
    return 0;
}
