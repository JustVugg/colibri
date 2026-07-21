/* Motore GLM-5.2 (architettura glm_moe_dsa) in C puro.
 * Stadio B: replica fedele del forward di transformers (modeling_glm_moe_dsa.py):
 *   - attenzione MLA (q/kv-LoRA, RoPE interleaved parziale)
 *   - router sigmoid + noaux_tc (n_group=1) con routed_scaling_factor
 *   - shared expert + expert routed in streaming dal disco (per-expert)
 *   - primi first_k_dense_replace layer densi
 * Il DSA indexer e' un NO-OP per seq <= index_topk (seleziona tutte le key): qui si usa
 * attenzione causale densa -> output identico all'oracolo su prompt corti.
 *
 * QUANTIZZAZIONE: gli expert (streaming) e la parte DENSA residente (attenzione, lm_head,
 * embed, mlp densa, shared expert) sono tenuti in int8 per-riga + scala (dequant-on-use).
 * E' cio' che fa entrare GLM-5.2 nei 15 GB: ~17B param residenti a int4 ~= 8.7 GB.
 * Norme/router/bias restano f32 (piccoli e sensibili).
 *
 * Validazione: stessi token id di ref_glm.json (oracolo transformers, c/tools/make_glm_oracle.py).
 *   build: make glm   run: SNAP=./glm_tiny ./glm <cap> <expert_bits> <dense_bits>
 *   TF=1 -> teacher-forcing (valida il prefill su tutta la sequenza)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <pthread.h>                              /* thread I/O del PILOTA */
#include <stdatomic.h>                            /* PIPE ready-flags/job queue + PILOT_REAL cross-layer handshake */
#include <sched.h>                                /* sched_yield: PIPE spin / PILOT barrier */
#include <unistd.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/select.h>                             /* select() serve-loop polling (#68); not on native MinGW */
#endif
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/mman.h>                             /* mlock: inchioda le pagine in RAM / wire pages into RAM */
#ifdef __linux__
#include <sys/syscall.h>                          /* COLI_NUMA: mbind degli slab expert / expert-slab interleave */
#endif
#include <sys/stat.h>                             /* fstat per mmap degli shard (COLI_MMAP) */
#include <signal.h>                               /* SIGINT = stop morbido del turno in serve mode */
#endif
#ifdef __linux__
#include <sys/vfs.h>                              /* statfs: real fs-type check for the 9p warning (below) */
#endif
#if defined(_WIN32) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>                                /* hwinfo_emit: CPU brand string senza /proc */
#endif
#include "st.h"
#ifdef __linux__
#include "uring.h"
#endif
#include "tok.h"
#include "tier.h"
#include "grammar.h"                              /* metodo F: draft grammaticali (#48) */
#include "schema_gbnf.h"                          /* SCHEMA=: JSON-Schema -> GBNF for method F */
#include "decode_batch.h"
#ifdef _OPENMP
#include <omp.h>                                  /* scratch per-thread nell'attention */
#else
static inline int omp_get_max_threads(void){ return 1; }
static inline int omp_get_thread_num(void){ return 0; }
#endif
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif
#ifdef COLI_METAL
#include "backend_metal.h"
#include <omp.h>
static int g_metal_enabled;
static int g_metal_gemm_min=16;   /* COLI_METAL_GEMM_MIN: min rows to send a matmul_qt GEMM to GPU */
/* routing precalcolata dalla GPU (layer CB): moe() la usa e salta la FASE A */
static const int *g_pre_idx; static const float *g_pre_w; static const int *g_pre_keff;
static const float *g_pre_sh;   /* output dello shared expert gia' calcolato su GPU */
#endif
#ifdef __APPLE__
#include <mach/mach.h>                            /* host_statistics64: MemAvailable di macOS */
#endif

typedef struct {
    int hidden, n_layers, n_heads, n_experts, topk, moe_inter, dense_inter;
    int first_dense, q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head, n_shared, vocab;
    int n_group, topk_group, norm_topk;
    int stop_ids[8], n_stop;                     /* eos_token_id dal config (GLM-5.2 ne ha 3!) */
    int index_topk, index_nh, index_hd;          /* DSA lightning indexer */
    int8_t idx_type[128];                        /* per layer: 1=full (calcola), 0=shared (riusa) */
    float eps, theta, attn_scale, routed_scale;
} Cfg;

/* tensore [O,I] in uno di tre formati:
 *   fmt=0 F32   -> qf
 *   fmt=1 INT8  -> q8 (1 byte/param) + scala per riga
 *   fmt=2 INT4  -> q4 (2 valori per byte, impacchettati) + scala per riga
 * INT4 e' cio' che fa stare la densa residente nei 15 GB (0.5 byte/param). */
/* fmt: 0 F32, 1 INT8, 2 INT4 (2/byte), 3 INT2 (4/byte). q4 ospita sia int4 che int2 packed. */
typedef struct {
    int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I, gs;  /* gs=group size (0=per-row, 128=grouped) */
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;
#endif
    int cuda_eligible, cuda_failed, cuda_device;  /* resident tensor, never a reused expert slot */
} QT;
static int64_t qt_bytes(const QT *t){    /* byte residenti del tensore */
    int64_t n=(int64_t)t->O*t->I;
    if(t->fmt==0) return n*4;
    if(t->fmt==1) return n + (int64_t)t->O*4;
    if(t->fmt==3) return (int64_t)t->O*((t->I+3)/4) + (int64_t)t->O*4;
    if(t->fmt==4){ /* int4 grouped: packed nibbles + O*ceil(I/gs) scales */
        int ng=(t->I+t->gs-1)/t->gs;
        return (int64_t)t->O*((t->I+1)/2) + (int64_t)t->O*ng*4; }
    return (int64_t)t->O*((t->I+1)/2) + (int64_t)t->O*4;  /* fmt=2 int4 per-row */
}

typedef struct {
    float *in_ln, *post_ln;
    /* MLA (densa, quantizzata) */
    QT q_a, q_b, kv_a, kv_b, o; float *q_a_ln, *kv_a_ln;
#ifdef COLI_CUDA
    ColiCudaTensor *kv_b_shard[COLI_CUDA_MAX_DEVICES];
    int shard_h0[COLI_CUDA_MAX_DEVICES],shard_hn[COLI_CUDA_MAX_DEVICES],n_kv_b_shard;
    int shared_w4a16_failed;
#endif
    int sparse;
    /* dense mlp (sparse==0) */
    QT gate_proj, up_proj, down_proj;
    /* moe (sparse==1) */
    float *router, *router_bias;                 /* router f32 (sensibile) */
    QT sh_gate, sh_up, sh_down;                  /* shared expert */
} Layer;

/* slot di un expert: pesi quantizzati + scale. Nel container pre-quantizzato g/u/d sono
 * VISTE dentro `slab` (una sola pread coalescente); nel fallback hanno buffer propri.
 * slab_cap/fslab_cap: capienza allocata — gli slot ws[] sono riusati TRA layer e gli
 * expert non hanno tutti la stessa taglia (layer MTP int8 = 2x i layer int4). */
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used; } ESlot;

typedef struct {
    float **Lc, **Rc, **Ic;
    int *kv_start, max_t;
    int disk_nrec;
    char disk_path[2048];
    FILE *disk_fp;       /* kept-open handle: fopen once, fwrite per turn, fclose at exit (#4) */
    uint8_t *disk_buf;   /* staging buffer: one contiguous record per position (#1) */
    int64_t disk_buf_cap;
} KVState;

typedef struct {
    KVState *kv;
    int token, pos;
} DecodeRow;

typedef struct {
    Cfg c; shards S;
    int ebits, dbits;                            /* bit expert / bit densa */
    QT embed, lm_head; float *final_norm;
    Layer *L;
    /* KV-cache MLA COMPRESSA: per token si tiene solo il latente normato [kv_lora] e
     * k_rot [qk_rope] (576 vs 32768 valori/token). k_nope e value si ricostruiscono al
     * volo con kv_b. E' cio' che rende gestibile il contesto su 15 GB (64 teste, no GQA). */
    float **Lc, **Rc; int max_t;                 /* alias della KVState attiva */
    int *kv_start;                               /* prima pos valida nella KV del layer (MTP: parziale) */
    KVState *kv;
    ESlot **ecache; int *ecn; int ecap;          /* LRU expert per-layer */
    float **kv_dev_L, **kv_dev_R; int *kv_dev_valid; /* ombra KV su device (decode) */
    ESlot ws[64];                                /* working set del layer corrente (load paralleli) */
    ESlot **pin; int *npin;                      /* HOT-STORE: expert pinnati in RAM (mai evicted) */
    uint32_t **eusage;                           /* contatori persistenti (per STATS/PIN) */
    uint32_t **eheat;                            /* calore recente per promotion/demotion live */
    uint32_t **elast, eaccess_clock;              /* recency per LFRU session-local */
    /* DISK-CLASS: PRIVATE recency state, read only by expert_classify(). Private --
     * not the real elast/eaccess_clock -- kept fully separate so DISK-CLASS's bookkeeping
     * can never read from or write into stock eviction state: every DISK-CLASS write lives
     * inside its own need_classify/dc_on gate, so "byte-identical with PROF=0" is provable
     * by construction instead of by argument. (Historical note: when this was first written,
     * the Metal pre-routed FASE A path (g_pre_idx) never bumped the real elast/eaccess_clock
     * -- on Metal decode the real clock froze at end of prefill, so REPIN's LRU tie-breaker
     * ran on stale recency for the rest of the run. That was an upstream defect; it has since
     * been reported and fixed (#417, cfcc742) -- FASE A now bumps the real clock too. The
     * private clock is retained anyway: separation from stock state is the stronger property,
     * independent of whether the real clock is correct.) elast_dc/eaccess_clock_dc tick in
     * BOTH FASE A paths, under the same need_classify gate, at the same rate the real clock
     * ticks on the CPU path (one per selected (position,expert)) -- so the
     * COLI_DISKCLASS_WINDOW window keeps its meaning in every mode. elast_pre snapshots
     * elast_dc just BEFORE this call's own bump (see the touched[] guard in FASE A) --
     * classifying against the live array would read the bump routing just made a few lines
     * above the load that needed it, so a giant cold prefill burst would score every expert
     * "just accessed" and get called warm. Recency alone (not eheat's access COUNT): a count
     * never decays, so an expert hot early in a long session would keep reading "warm" long
     * after it dropped out of the working set. Same shape/allocation as elast; NULL for dense
     * layers. */
    uint32_t **elast_dc, **elast_pre, eaccess_clock_dc;
    /* DSA lightning indexer (attivo solo se i pesi out-idx-* sono presenti) */
    int has_dsa;
    QT *ix_wq, *ix_wk, *ix_wp;                   /* per layer FULL: wq_b, wk, weights_proj */
    float **ix_knw, **ix_knb;                    /* k_norm (LayerNorm, eps 1e-6) */
    float **Ic;                                  /* alias KVState: cache indexer [max_t*hd] */
    int *dsa_sel, *dsa_nsel; int dsa_scap;       /* selezione per posizione del batch corrente */
    /* testa MTP (layer n_layers, stile DeepSeek-V3): draft nativi ad alta acceptance */
    int has_mtp; Layer mtpL; QT eh_proj;
    float *enorm, *hnorm, *mtp_norm;
    float *hlast, *h_all;                        /* hidden pre-norm: ultima pos / tutte le pos batch */
    uint64_t mtp_prop, mtp_acc;                  /* statistica acceptance */
    int **eroute; int *enr;                      /* metodo C: routing dell'ULTIMO token per layer */
    uint64_t eclock, hits, miss, ereq;
    uint64_t hit_pin, hit_ecache;                /* split di hits per tier (#336): pin vs LRU ecache */
    uint64_t gpu_expert_calls; int gpu_expert_count; int64_t gpu_expert_bytes;
    uint64_t n_fw, n_emit;                       /* metodo E: forward di decode / token emessi */
    uint64_t route_slots, route_swaps;            /* CACHE_ROUTE: slots chosen / substituted vs true top-K */
    uint64_t route_agree_hit, route_agree_tot;    /* ROUTE_AGREE: |chosen ∩ true top-K| / K */
    double route_kl_sum; uint64_t route_kl_n;     /* mean KL(true||chosen) on gate mass */
    double t_ewait, t_emm, t_ecpu, t_egpu, t_route, t_p2p, t_attn, t_kvb, t_head;
    uint64_t n_p2p;                              /* P0 execution profile: tier split + residual hops */
    uint64_t cpu_expert_rows; int64_t cpu_expert_bytes;
                                                 /* profiling: dove va il tempo (wall del
                                                  * thread di compute; il servizio disco
                                                  * overlappato vive in g_edisk_ns) */
    double t_aproj,t_acore,t_aout;                     /* attention breakdown */
    int64_t resident_bytes;
    /* DISK_SPLIT=1: split dei DISK LOAD (miss LRU -> expert_load) per contesto e per tipo
     * di layer. ld_ctx: 0=main/verify/prefill, 1=dentro mtp_draft, 2=dentro mtp_absorb. */
    int ld_ctx;
    uint64_t miss_draft, miss_absorb;            /* miss in moe() per contesto */
    uint64_t ld_mtp, ld_main;                    /* expert_load per tipo layer (MTP int8 vs main int4) */
    uint64_t bytes_mtp, bytes_main;              /* byte letti da disco per tipo layer */
} Model;

#include "quant.h"
static int g_no_fused_pair=0;
static int g_spec_pin=1;
static int g_spec_live=0;
static inline int spec_pinned(void){ return g_spec_pin && g_spec_live; }

static void matmul_qt_ex(float *y, const float *x, QT *w, int S, int allow_idot);
static void matmul_qt(float *y, const float *x, QT *w, int S){ matmul_qt_ex(y,x,w,S,1); }

static void expert_gate_up(float *g,float *u,const float *x,QT *wg,QT *wu,int S){
    if(!g_no_fused_pair&&!spec_pinned()&&S==1&&wg->fmt==2&&wu->fmt==2&&wg->I==wu->I&&wg->O==wu->O)
        matmul_i4_pair(g,u,x,wg->q4,wg->s,wu->q4,wu->s,wg->I,wg->O);
    else { matmul_qt(g,x,wg,S); matmul_qt(u,x,wu,S); }
}

static int g_repin;
static uint64_t g_last_repin;
#ifdef COLI_CUDA
static int g_cuda_enabled;
static double g_cuda_expert_gb;
static int g_cuda_expert_auto;
static int g_cuda_dense;
static int g_cuda_release_host;
static int g_cuda_devices[COLI_CUDA_MAX_DEVICES], g_cuda_ndev, g_cuda_rr;
static int64_t g_cuda_dense_projected[COLI_CUDA_MAX_DEVICES];
static void qt_cuda_reset(QT *t){
    if(t->cuda){ coli_cuda_tensor_free(t->cuda); t->cuda=NULL; }
    t->cuda_failed=0;
}
static int qt_cuda_upload(QT *t){
    const void *weights = t->fmt==0 ? (const void*)t->qf
                        : t->fmt==1 ? (const void*)t->q8 : (const void*)t->q4;
    return coli_cuda_tensor_upload(&t->cuda,weights,t->s,t->fmt,t->I,t->O,t->cuda_device);
}
static int qt_cuda_update(QT *t){
    const void *weights=t->fmt==0?(const void*)t->qf:
                        t->fmt==1?(const void*)t->q8:(const void*)t->q4;
    return coli_cuda_tensor_update(t->cuda,weights,t->s);
}
static void cuda_stats_print(void){
    size_t n=0,b=0; coli_cuda_stats(-1,&n,&b);
    fprintf(stderr,"[CUDA] resident set: %zu tensors, %.2f GB VRAM\n",n,b/1e9);
    if(g_cuda_ndev>1) for(int i=0;i<g_cuda_ndev;i++){
        coli_cuda_stats(g_cuda_devices[i],&n,&b);
        fprintf(stderr,"[CUDA]   device %d: %zu tensors, %.2f GB\n",g_cuda_devices[i],n,b/1e9);
    }
    uint64_t calls=0,experts=0,rows=0; double h2d=0,kernel=0,d2h=0;
    coli_cuda_group_stats(&calls,&experts,&rows,&h2d,&kernel,&d2h);
    if(calls) fprintf(stderr,"[CUDA] expert groups: %llu call, %llu expert, %llu righe "
        "(%.2f expert/call)%s\n",(unsigned long long)calls,(unsigned long long)experts,
        (unsigned long long)rows,(double)experts/calls,
        getenv("COLI_CUDA_PROFILE")?"; timing sotto":"");
    if(calls&&getenv("COLI_CUDA_PROFILE")) fprintf(stderr,
        "[CUDA] expert groups timing: H2D %.1f ms | kernel %.1f ms | D2H %.1f ms\n",h2d,kernel,d2h);
}
static int parse_cuda_devices(const char *list, int *out){
    if(!list||!*list) return 0;
    int n=0; const char *p=list;
    while(*p){
        char *end=NULL; long v=strtol(p,&end,10);
        if(end==p||v<0||v>INT_MAX||n>=COLI_CUDA_MAX_DEVICES) return 0;
        for(int i=0;i<n;i++) if(out[i]==(int)v) return 0;
        out[n++]=(int)v; p=end;
        while(*p==' '||*p=='\t') p++;
        if(!*p) break;
        if(*p++!=',') return 0;
        while(*p==' '||*p=='\t') p++;
        if(!*p) return 0;
    }
    return n;
}
#endif
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);   /* macOS: ru_maxrss in BYTE */
#else
    return r.ru_maxrss/(1024.0*1024.0);          /* Linux: in KB */
#endif
}
/* ---- PROF=1: opt-in performance profile ----------------------------------
 * Records per-forward decode latency and expert-file bytes fetched, then
 * reports percentiles, I/O totals, phase shares and a tuning verdict next to
 * the existing PROFILE line. Additive only: with PROF unset the output of
 * every mode stays byte-identical. */
static int g_prof=0;
static _Atomic int64_t g_prof_io;                /* bytes pread()/faulted from expert files */
/* Disk service: wall time inside expert_load on whichever thread runs the read
 * (PIPE I/O workers, OMP loaders, the speculative pilot). It overlaps compute,
 * so it is NOT a wall-time phase — the stall the compute thread actually felt
 * is m->t_ewait. Thread-seconds, so it can exceed wall time under parallel
 * reads; wait << service means overlap/parallelism is hiding the reads,
 * wait ~ service means the loads block the compute thread. */
static _Atomic int64_t g_edisk_ns;
static double edisk_s(void){ return atomic_load_explicit(&g_edisk_ns,memory_order_relaxed)*1e-9; }
/* DISK-CLASS (PROF=1): per-load cold/warm classification against the engine's own
 * recency state. Instrumentation only -- it never changes which fd serves a read (see
 * expert_classify() and its call site in expert_load_impl; the fd choice expression is
 * untouched by this feature). COLI_DISKCLASS_WINDOW is the recency window in ticks of the
 * PRIVATE clock (m->eaccess_clock_dc -- NOT the real eaccess_clock; see elast_dc in Model
 * for why DISK-CLASS keeps its own clock instead of reading the real one): one tick per
 * selected (position,expert) in FASE A while classification is active, the same per-token
 * rate the real clock has on the CPU path, so the window's meaning is unchanged. At or
 * under the window = warm; 0 (default, unset) derives it from topk*n_layers*8 once the
 * model config is known (main(), right after model_init) -- roughly "seen in the last ~8
 * tokens", generous on purpose (conservative-toward-warm: a load the page cache could have
 * served that gets labeled cold overstates the cold class, the bucket this line exists to
 * size). */
static uint32_t g_direct_heat_ticks=0;
static int g_direct_heat_explicit=0;    /* 1 if COLI_DISKCLASS_WINDOW was set (skip the auto-derive) */
#define DC_COLD 0
#define DC_WARM 1
static _Atomic uint64_t g_dc_n[2];              /* [DC_COLD]/[DC_WARM]: loads classified */
static _Atomic int64_t  g_dc_bytes[2];          /* bytes read (weights + scales, matches g_prof_io) */
static _Atomic int64_t  g_dc_ns[2];             /* wall ns spent reading (thread-seconds, like g_edisk_ns) */
static _Atomic uint64_t g_dc_direct_n[2];       /* subset of the above ACTUALLY served by the uncached fd */
/* Busy-wall per class + combined: how much WALL time had >=1 classified load of the
 * class in flight (thread-seconds / busy-wall = average concurrency; bytes / busy-wall
 * = aggregate GB/s the disk actually delivered for that class -- the quantity the
 * thread-second numbers alone can't answer: N slow overlapped reads can beat N fast
 * serial ones in aggregate, and only wall-denominated rates see it). Transition scheme:
 * 0->1 records a start, 1->0 accumulates (now - start). One dedicated mutex serializes
 * the transition bookkeeping -- two short lock/unlock pairs per load against ms-scale
 * reads; a CAS scheme would save nothing measurable and be harder to audit
 * (correctness over cleverness). Only COMPLETED intervals are in the accumulators: an
 * interval still open at report time is not counted (bounded by one read's duration --
 * noise at report granularity). */
static pthread_mutex_t g_dc_wall_mx=PTHREAD_MUTEX_INITIALIZER;
static int g_dc_inflight[2], g_dc_inflight_all;              /* guarded by g_dc_wall_mx */
static double g_dc_wall_open[2], g_dc_wall_open_all;         /* start of the open interval */
static int64_t g_dc_wall_ns[2], g_dc_wall_all_ns;            /* completed busy-wall ns */
static void dc_wall_enter(int cls, double now){
    pthread_mutex_lock(&g_dc_wall_mx);
    if(g_dc_inflight[cls]++==0) g_dc_wall_open[cls]=now;
    if(g_dc_inflight_all++==0)  g_dc_wall_open_all=now;
    pthread_mutex_unlock(&g_dc_wall_mx);
}
static void dc_wall_exit(int cls, double now){
    pthread_mutex_lock(&g_dc_wall_mx);
    if(--g_dc_inflight[cls]==0) g_dc_wall_ns[cls]+=(int64_t)((now-g_dc_wall_open[cls])*1e9);
    if(--g_dc_inflight_all==0)  g_dc_wall_all_ns +=(int64_t)((now-g_dc_wall_open_all)*1e9);
    pthread_mutex_unlock(&g_dc_wall_mx);
}
static int dc_needed(void);                                  /* fwd: defined with the classifier (needs g_prof) */
static void dc_wall_read(int64_t out[2], int64_t *all){      /* mutex-consistent snapshot for the report */
    if(!dc_needed()){ out[0]=out[1]=0; *all=0; return; }     /* off for the whole process => accumulators are
                                                              * provably zero (only dc_wall_exit writes them,
                                                              * only under dc_on): skip the lock, zero work.
                                                              * Needed because prof_base runs unconditionally
                                                              * at some call sites ("cheap enough to always"). */
    pthread_mutex_lock(&g_dc_wall_mx);
    out[0]=g_dc_wall_ns[0]; out[1]=g_dc_wall_ns[1]; *all=g_dc_wall_all_ns;
    pthread_mutex_unlock(&g_dc_wall_mx);
}
#define PROF_LAT_CAP 32768
static double g_prof_lat[PROF_LAT_CAP];          /* per-forward decode wall clock (ring) */
static uint64_t g_prof_nlat;                     /* forwards recorded (monotonic) */
static void prof_lat(double s){ g_prof_lat[g_prof_nlat++ % PROF_LAT_CAP]=s; }
/* snapshot for windowed reports (serve mode: one report per turn) */
typedef struct {
    double edisk,ewait,emm,ecpu,egpu,route,p2p,attn,head;
    int64_t io,cpu_bytes; uint64_t hits,miss,ereq,n_fw,n_emit,nlat,n_p2p,cpu_rows;
    uint64_t hit_pin,hit_ecache;
    uint64_t dc_n[2], dc_direct_n[2]; int64_t dc_bytes[2], dc_ns[2]; /* DISK-CLASS */
    int64_t dc_wall_ns[2], dc_wall_all_ns;       /* busy-wall (per class + combined) */
} ProfBase;
static void prof_base(Model *m, ProfBase *b){
    b->edisk=edisk_s(); b->ewait=m->t_ewait; b->emm=m->t_emm;
    b->ecpu=m->t_ecpu; b->egpu=m->t_egpu; b->route=m->t_route; b->p2p=m->t_p2p;
    b->attn=m->t_attn; b->head=m->t_head;
    b->io=atomic_load_explicit(&g_prof_io,memory_order_relaxed);
    b->hits=m->hits; b->miss=m->miss; b->ereq=m->ereq;
    b->hit_pin=m->hit_pin; b->hit_ecache=m->hit_ecache;
    b->n_fw=m->n_fw; b->n_emit=m->n_emit; b->nlat=g_prof_nlat; b->n_p2p=m->n_p2p;
    b->cpu_bytes=m->cpu_expert_bytes;b->cpu_rows=m->cpu_expert_rows;
    for(int i=0;i<2;i++){
        b->dc_n[i]=atomic_load_explicit(&g_dc_n[i],memory_order_relaxed);
        b->dc_bytes[i]=atomic_load_explicit(&g_dc_bytes[i],memory_order_relaxed);
        b->dc_ns[i]=atomic_load_explicit(&g_dc_ns[i],memory_order_relaxed);
        b->dc_direct_n[i]=atomic_load_explicit(&g_dc_direct_n[i],memory_order_relaxed);
    }
    dc_wall_read(b->dc_wall_ns,&b->dc_wall_all_ns);
}

static float *falloc(int64_t n){
    /* guardia anti-wrap (report PR #25): n assurdo da file modello ostili non deve
     * diventare una malloc piccola. Niente calloc: il memset nel percorso caldo costa. */
    if(n<0 || (uint64_t)n > SIZE_MAX/sizeof(float)){ fprintf(stderr,"falloc: n=%lld is out of range\n",(long long)n); exit(1); }
    float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }


static void matmul_qt_ex(float *y, const float *x, QT *w, int S, int allow_idot){
#ifdef COLI_METAL
    if(g_metal_enabled && S>=g_metal_gemm_min && !spec_pinned() && (w->fmt==1||w->fmt==2) && !omp_in_parallel()){
        const void *wp = w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_metal_gemm(y,x,wp,w->s,w->fmt,S,w->I,w->O)) return;
    }
#endif
#ifdef COLI_CUDA
    if(g_cuda_enabled && w->cuda_eligible && !w->cuda_failed && !omp_in_parallel()){
        const void *weights = w->fmt==0 ? (const void*)w->qf
                            : w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_cuda_matmul(&w->cuda,y,x,weights,w->s,w->fmt,S,w->I,w->O,w->cuda_device)) return;
        w->cuda_failed=1;
        fprintf(stderr,"[CUDA] tensor [%d,%d] on device %d disabled after an error; falling back to CPU\n",
            w->O,w->I,w->cuda_device);
    }
#endif
    if(w->fmt==0){ matmul(y,x,w->qf,S,w->I,w->O); return; }
    if(w->fmt==4){ matmul_i4_grouped(y,x,w->q4,w->s,S,w->I,w->O,w->gs); return; }
    if(allow_idot && g_idot && (w->fmt==1 || (w->fmt==2 && (spec_pinned() ? g_i4s<=1 : S>=g_i4s)))){
        int I=w->I; int8_t *xq; float *sx;
        if(S<0 || I<0 || (size_t)S>SIZE_MAX/(size_t)(I?I:1)){ fprintf(stderr,"matmul_qt: shape overflow\n"); exit(1); }
        quant_scratch((size_t)S*I,(size_t)S,&xq,&sx);
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        if(w->fmt==1) matmul_q_idot(y,xq,sx,w->q8,w->s,S,I,w->O);
        else matmul_i4_idot(y,xq,sx,w->q4,w->s,S,I,w->O);
        return;
    }
    if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==3) matmul_i2(y,x,w->q4,w->s,S,w->I,w->O);
    else matmul_i4(y,x,w->q4,w->s,S,w->I,w->O);
}

static int g_nopack=0;   /* NOPACK=1 -> tiene i valori <=4bit in contenitore int8 (per validare il packing) */
static int g_drop=0;     /* DROP=1 -> scarta le pagine expart dopo l'uso. Default 0: le lascia in
                          * page-cache (buff/cache, NON RSS) come L2 gratuito -> sfrutta lo
                          * sbilanciamento del routing MoE (pochi expert "caldi" riusati). */
static int g_prefetch=0; /* PREFETCH=1 -> riabilita il WILLNEED cross-layer (metodo C). Default
                          * OFF: i load VERI in parallelo lo hanno reso superfluo, e sotto
                          * pressione di memoria il readahead speculativo veniva rievictato. */
static int g_direct=0;   /* DIRECT=1 -> O_DIRECT sugli slab expert. Default OFF: su questo host
                          * (VHDX su NVMe DRAM-less, latenza serializzata ~60ms/req) il buffered
                          * liscio e' risultato il migliore; su NVMe veri DIRECT=1 rende di piu'. */
static float g_temp=-1;  /* TEMP: temperatura di sampling sui TOKEN. <0 = auto (1.0 in chat/testo,
                          * 0=greedy in validazione). 0 = greedy puro. */
static float g_nuc=0.95f;/* NUCLEUS: top-p sul vocabolario (default dal generation_config GLM-5.2) */
static int g_topk=0;     /* TOPK=n -> usa n expert/token invece di config (ricerca: meno disco) */
static float g_topp=0;   /* TOPP=p (0..1) -> top-p adattivo: tieni gli expert fino a peso cumulato p */
static int g_expert_budget=0; /* EXPERT_BUDGET=N -> cap distinct experts loaded per layer across the
                               * batch-union. Reduces disk I/O on cold/low-RAM hosts by dropping the
                               * lowest-gate-weight experts from the cross-position union. MoE-Spec
                               * (arXiv 2602.16052): top-32 of 64 capture 93% routing weight. */
static int64_t g_budget_dropped=0; /* total experts dropped by EXPERT_BUDGET across all layers */
/* CACHE_ROUTE (paper 2412.00099 max-rank): opt-in only. Keep true top-J always;
 * fill remaining slots preferring pin∪LRU experts ranked within top-M (or mass ROUTE_P). */
static int g_cache_route=0;
static int g_route_j=2;      /* ROUTE_J: sacred top ranks (always take, even uncached) */
static int g_route_m=12;     /* ROUTE_M: max-rank window for cache-preferring fill */
static float g_route_p=0;    /* ROUTE_P: if >0, choose M from cumulative router mass instead */
static float g_route_alpha=1.f; /* ROUTE_ALPHA: scale gate mass of CACHE_ROUTE substitutes before renorm (1=off) */
static int g_route_agree=0;  /* ROUTE_AGREE=1: footer overlap% + mean KL vs true top-K */
static int expert_is_resident(Model *m, int layer, int eid); /* pin∪LRU; defined near pilot */
static int g_spec=1;     /* metodo C: SPEC=0 disabilita il prefetch speculativo cross-layer */
static int g_draft=0;    /* metodo E: DRAFT=n token auto-speculati per forward via n-gram lookup
                          * (0=off). LOSSLESS: verifica = output identico al greedy. Default OFF:
                          * misurato sul run reale (2026-07-03) acceptance ~5% -> ogni draft
                          * rifiutato paga comunque i suoi expert dal disco = ~3x piu' lento.
                          * Opt-in (DRAFT=4) per testi ripetitivi dove l'acceptance e' alta. */
/* metodo F (#48): GRAMMAR=<file.gbnf> -> terza sorgente di draft, la grammatica stessa.
 * Nei workload a output vincolato (JSON/NDJSON, function calling) i byte FORZATI dalla
 * grammatica (chiavi, punteggiatura, valori enum) sono draft gratuiti ad acceptance ~1:
 * nessuna testa, nessuna lookup table, e si aggancia anche dove la testa MTP int4 non
 * parte (#8). MAI un vincolo sul sampling: solo proposte, la verifica batch-union
 * decide — grammatica sbagliata = draft rifiutati, output identico.
 * GRAMMAR_DRAFT=n (default 24) limita i token forzati per forward. */
static Grammar g_gram; static GrState g_gst;
static Tok *g_gr_T=NULL;
static int g_gr_on=0;     /* grammatica caricata e walker vivo */
static int g_gr_armed=0;  /* lazy: parte dal primo byte ammesso dalla radice (salta i preamboli) */
static int g_gr_max=24;
static uint64_t g_gr_prop=0, g_gr_acc=0;
static FILE *g_route_fp=NULL; /* ROUTE_TRACE=<path>: dump per-position top-K routing (ids:gates)
                               * per layer — offline co-activation / coupling analysis. Zero
                               * effect on computation; measurement only. */
static int g_route_call=0;
/* COUPLE=<.coli_pairs>: coupling-scored cross-layer prefetch. The routing of layer L
 * strongly constrains the routing of L+1/L+2 (measured: median co-activation lift 1.8x
 * over independence, p99 40x, and the structure TRANSFERS across workloads — it is a
 * property of the model, not the session). An offline table (tools/route_pairs.py,
 * built from ROUTE_TRACE dumps) maps (layer, expert) -> top co-activated experts of the
 * next layer(s); after FASE A routing we score candidates by summing counts over the
 * position's routed set and enqueue the top COUPLE_K non-resident ones into the SAME
 * pilot ring (worker, residency re-check, safety invariants unchanged). Unlike PILOT,
 * no router matmul is needed — prediction is a table lookup on ids the layer just
 * produced. Hints only: a wrong prediction costs bandwidth, never output. */
#define CP_M 16
static int g_couple=0, g_couple_k=8, g_couple_d=1;
static int16_t *cp_pred=NULL;    /* [(L*2+(dL-1))*E + e]*CP_M + j -> target id (-1 none) */
static float   *cp_cnt=NULL;
static long g_cp_enq=0;
static void couple_prefetch(Model *m, int layer, const int *idx, int Ke);
static int g_looka=0;    /* LOOKA=1: misura (solo contatori, zero effetti) quanto il routing MoE
                          * e' predicibile IN ANTICIPO — la domanda che decide se un prefetch
                          * pilotato dal router puo' riempire i tempi morti del disco.
                          * [0] token precedente, stesso layer (cio' che usa gia' SPEC/PREFETCH)
                          * [1] ingresso del layer -> routing dello STESSO layer (salta l'attention)
                          * [2] post-attention del layer L -> routing di L+1 (un residuo MoE e
                          *     un'attention di anticipo: il punto dove il prefetch avrebbe
                          *     un intero giro di disco per lavorare in ombra). */
static int64_t la_hit[4], la_tot[4];  /* [0]=prev, [1]=skip-attn, [2]=PILOT, [3]=two-step */
static int la_pred[3][130][16]; static signed char la_val[3][130];
static int g_pilot=0;    /* PILOT=1: prefetch pilotato dal router (vedi pilot_prefetch) */
static int g_pilot_k=8;  /* PILOT_K=k: prefetcha solo le prime k predizioni per posizione */
static int g_disk_split=0; /* DISK_SPLIT=1: contatori che spezzano i DISK LOAD (miss LRU) in
                          * draft MTP / absorb / verify-main e in layer MTP (int8) vs main
                          * (int4), con i byte letti. Default OFF: a flag spento gli atomic
                          * non vengono MAI toccati (zero overhead), le righe extra di stats
                          * non vengono stampate. Solo misura: nessun effetto sull'output. */

#include "sample.h"
#include "kv_persist.h"
#include "telemetry.h"

/* Aligned allocator for dense QT weights/scales: under METAL, page-align + register so the
 * GPU reads them zero-copy (no upload duplicate). Plain malloc otherwise. */
/* ---- COLI_NUMA=1 (#82): interleave the expert slabs across NUMA nodes ----
 * On multi-socket hosts first-touch parks nearly the whole pin+LRU on the loader
 * thread's node (measured: node0 766MB free / node1 idle), and every far-socket
 * core then streams weights over the interconnect. Interleaving ONLY the expert
 * slabs recruits all memory controllers: +7%/-14% expert-matmul on 2 sockets,
 * +40% on a 4-socket (#82). Blanket `numactl --interleave=all` is NOT equivalent:
 * it also interleaves the CUDA pinned staging buffers and cost a 4-socket GPU host
 * 10x (#82) — hence per-region mbind here and nothing else. Raw syscall, no libnuma
 * dependency; MPOL_MF_MOVE migrates pages of reused heap chunks too. Linux-only,
 * silent no-op elsewhere or on single-node hosts. */
#ifdef __linux__
static int g_numa_nodes=0;      /* only touched under __linux__; off-Linux NUMA is a no-op */
#endif
static void numa_slab_bind(void *p, size_t n){
#ifdef __linux__
    if(g_numa_nodes<2 || !p || !n) return;
    unsigned long mask=(1UL<<g_numa_nodes)-1;
    uintptr_t a=(uintptr_t)p & ~(uintptr_t)4095;
    size_t len=(((uintptr_t)p+n+4095) & ~(uintptr_t)4095) - a;
    syscall(SYS_mbind,a,len,3/*MPOL_INTERLEAVE*/,&mask,
            (unsigned long)(g_numa_nodes+1),(unsigned)2/*MPOL_MF_MOVE*/);
#else
    (void)p;(void)n;
#endif
}
static void numa_init(void){
#ifdef __linux__
    if(!getenv("COLI_NUMA")||!atoi(getenv("COLI_NUMA"))) return;
    for(int i=0;i<64;i++){ char pth[64]; snprintf(pth,sizeof(pth),"/sys/devices/system/node/node%d",i);
        struct stat st; if(stat(pth,&st)) break; g_numa_nodes=i+1; }
    if(g_numa_nodes>=2) fprintf(stderr,"[NUMA] expert slabs interleaved across %d nodes\n",g_numa_nodes);
    else fprintf(stderr,"[NUMA] single node: COLI_NUMA ignored\n");
#endif
}

static void *qalloc(size_t n){
#ifdef COLI_METAL
    if(g_metal_enabled){ void *p; size_t r=(n+16383)&~(size_t)16383;
        if(posix_memalign(&p,16384,r)){fprintf(stderr,"OOM qalloc\n");exit(1);}
        coli_metal_register(p,r); return p; }
#endif
    void *p=malloc(n);
    if(n>=(size_t)1<<20) numa_slab_bind(p,n);      /* resident dense weights too (#82: attention/shared stream from RAM every token) */
    return p;
}
static float *qsalloc(int O){ return (float*)qalloc((size_t)O*sizeof(float)); }
static int g_pilot_real=0;/* PILOT_REAL=1: il pilota fa LOAD VERI cross-layer dentro ecache[L+1]
                          * (non il semplice WILLNEED). Implica PILOT=1. Default OFF: hint-only. */
static int g_pilot_two=0; /* PILOT_TWO=1: two-step prefetch — before running L+1's router,
                          * approximate MoE(L) using only the shared expert (resident, no disk)
                          * and add it to the state. Trades 3 small matmuls for +2.3% recall. */
/* Handshake main<->pilota per il load-vero cross-layer. Invariante di sicurezza in DUE parti:
 *  1) Percorso MATMUL (moe): il pilota scrive SOLO ecache[layer] con layer > g_cur_moe_layer;
 *     il matmul in moe() legge SOLO ecache[layer]==g_cur_moe_layer, e la barriera a inizio moe()
 *     aspetta l'eventuale load in volo su QUEL layer. Quindi NESSUNO slot mezzo-caricato viene
 *     mai matmul-ato: il matmul e il pilota non toccano mai lo stesso layer contemporaneamente.
 *  2) Percorso SCAN (pilot_prefetch, anch'esso sul MAIN): la scansione di residenza gira sul
 *     layer FUTURO (lnext = layer corrente + 1), esattamente il layer che il pilota sta scrivendo
 *     -> QUI i due thread toccano davvero la stessa ecache. Percio' quella scansione prende
 *     g_pilot_mx (lo stesso lock del worker): letture e pubblicazione degli slot sono serializzate,
 *     niente torn read di ecn[]/eid. Il pilota non altera MAI il valore di un expert, solo QUALE
 *     expert e' residente: con un load andato a buon fine l'output resta byte-identico all'OFF. */
static pthread_mutex_t g_pilot_mx=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pilot_cv=PTHREAD_COND_INITIALIZER;
static _Atomic int g_cur_moe_layer=-1;   /* massimo layer moe in cui il MAIN e' entrato (per forward) */
static int g_pilot_inflight[256];        /* protected by g_pilot_mx; URING can load a layer concurrently */
static _Atomic long g_pilot_loads=0;     /* load cross-layer VERI completati (banda spesa) */
static _Atomic long g_pilot_drops=0;     /* predizioni scartate perche' il main possiede gia' il layer */
/* sceglie il formato da `bits`: >=16 f32, 5..8 int8, <=4 int4-packed */
static void qt_alloc(QT *t, int O, int I, int bits){
    t->O=O; t->I=I; t->qf=NULL; t->q8=NULL; t->q4=NULL; t->s=NULL;
    if(bits>=16){ t->fmt=0; t->qf=falloc((int64_t)O*I); }
    else if(bits>=5 || g_nopack){ t->fmt=1; t->q8=qalloc((int64_t)O*I); t->s=qsalloc(O); }
    else if(bits>=3){ t->fmt=2; t->q4=qalloc((int64_t)O*((I+1)/2)); t->s=qsalloc(O); }
    else { t->fmt=3; t->q4=qalloc((int64_t)O*((I+3)/4)); t->s=qsalloc(O); }
}
static void qt_fill(QT *t, const float *w, int bits){
    if(t->fmt==0) memcpy(t->qf, w, (int64_t)t->O*t->I*sizeof(float));
    else if(t->fmt==1) quantize_rows(w, t->q8, t->s, t->O, t->I, bits);
    else if(t->fmt==3) pack_int2(w, t->q4, t->s, t->O, t->I, bits);
    else pack_int4(w, t->q4, t->s, t->O, t->I, bits);
}

static void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps); for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}
/* LayerNorm classica (media+varianza, weight+bias) — usata dal k_norm dell'indexer DSA */
static void layernorm(float *v, const float *w, const float *b, int n, float eps){
    double mu=0; for(int i=0;i<n;i++) mu+=v[i]; mu/=n;
    double var=0; for(int i=0;i<n;i++){ double d=v[i]-mu; var+=d*d; } var/=n;
    float r=1.f/sqrtf((float)var+eps);
    for(int i=0;i<n;i++) v[i]=((float)(v[i]-mu))*r*w[i]+b[i];
}
static void softmax(float *x,int n){ float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++) x[i]/=s; }
static inline float sigmoidf(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf(float x){ return x/(1.f+expf(-x)); }

/* RoPE interleaved su un vettore di dimensione qk_rope a posizione pos */
static void rope_interleave(float *v, int pos, const Cfg *c){
    int half = c->qk_rope/2;
    /* Validate against the fixed buffers (in[256], cache cs/sn[128] -> qk_rope<=256).
     * Abort cleanly instead of smashing the stack. (GLM-5.2 qk_rope=64.) (#183) */
    if(c->qk_rope > 256){ fprintf(stderr,"qk_rope=%d exceeds rope_interleave buffer (256)\n",c->qk_rope); exit(1); }
    typedef struct { int pos,qk,valid; float theta,cs[128],sn[128]; } RopeCache;   /* (#80) */
    static _Thread_local RopeCache cache;
    float in[256]; memcpy(in,v,c->qk_rope*sizeof(float));
    if(!cache.valid||cache.pos!=pos||cache.qk!=c->qk_rope||cache.theta!=c->theta){
        for(int j=0;j<half;j++){
            float inv=powf(c->theta,-2.0f*j/c->qk_rope),ang=pos*inv;
            cache.cs[j]=cosf(ang); cache.sn[j]=sinf(ang);
        }
        cache.pos=pos; cache.qk=c->qk_rope; cache.theta=c->theta; cache.valid=1;
    }
    for(int j=0;j<half;j++){
        float cs=cache.cs[j],sn=cache.sn[j];
        float a=in[2*j], b=in[2*j+1];
        v[j]      = a*cs - b*sn;
        v[half+j] = b*cs + a*sn;
    }
}

/* ---------- config ---------- */
static jval* cfg_root(const char *snap, char **arena){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    /* SEC: config.json arriva dalla dir modello non fidata. Limita la dimensione
     * (un file ostile enorme = OOM al load) e controlla la malloc: senza il NULL
     * check, b[got]=0 su malloc fallita era un NULL-deref. */
    if(n<0 || n>(256L<<20)){ fprintf(stderr,"%s: size %ld out of range (0..256 MB)\n",p,n); exit(1); }
    char *b=malloc((size_t)n+1); if(!b){ fprintf(stderr,"OOM reading %s (%ld bytes)\n",p,n); exit(1); }
    size_t got=fread(b,1,(size_t)n,f); b[got]=0; fclose(f);
    if((long)got!=n) fprintf(stderr,"warning: short read on %s (%ld of %ld)\n",p,(long)got,n);
    return json_parse(b,arena);
}
static int gi(jval*r,const char*k){ jval*v=json_get(r,k); return v?(int)v->num:0; }
static void load_cfg(Cfg *c, const char *snap){
    char *ar=NULL; jval *r=cfg_root(snap,&ar);
    c->hidden=gi(r,"hidden_size"); c->n_layers=gi(r,"num_hidden_layers");
    c->n_heads=gi(r,"num_attention_heads"); c->n_experts=gi(r,"n_routed_experts");
    c->topk=gi(r,"num_experts_per_tok"); c->moe_inter=gi(r,"moe_intermediate_size");
    c->dense_inter=gi(r,"intermediate_size"); c->first_dense=gi(r,"first_k_dense_replace");
    c->q_lora=gi(r,"q_lora_rank"); c->kv_lora=gi(r,"kv_lora_rank");
    c->qk_nope=gi(r,"qk_nope_head_dim"); c->qk_rope=gi(r,"qk_rope_head_dim");
    c->v_head=gi(r,"v_head_dim"); c->n_shared=gi(r,"n_shared_experts"); c->vocab=gi(r,"vocab_size");
    c->n_group=gi(r,"n_group"); c->topk_group=gi(r,"topk_group");
    jval *nt=json_get(r,"norm_topk_prob"); c->norm_topk=(nt&&nt->t==J_BOOL)?nt->boolean:0;
    jval *ep=json_get(r,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-5f;
    jval *rs=json_get(r,"routed_scaling_factor"); c->routed_scale=rs?(float)rs->num:1.f;
    jval *rp=json_get(r,"rope_parameters"); jval *th=rp?json_get(rp,"rope_theta"):NULL;
    c->theta = th?(float)th->num:10000.f;
    /* token di stop: GLM-5.2 ne ha TRE (endoftext, user, observation). Fermarsi solo sul
     * primo = generare spazzatura invisibile dopo la fine del turno (5-10x token sprecati). */
    c->n_stop=0;
    jval *eo=json_get(r,"eos_token_id");
    if(eo){ if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
            else if(eo->t==J_ARR) for(int i=0;i<eo->len && c->n_stop<8;i++)
                c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num; }
    /* generation_config.json e' il file AUTOREVOLE per la generazione secondo HuggingFace:
     * config.json ne porta spesso una copia legacy o parziale. Un tool di conversione che
     * rigenera un config.json ridotto lascia il motore fermo su MENO stop del dovuto, e i
     * token di controllo che restano finiscono stampati in chat come testo (woolcoxm, #298:
     * "the stop token being printed to chat", verificato sui token id). Unione dei due:
     * uno stop in piu' non fa danno, uno in meno si' -- e chi converte i pesi non siamo noi.
     * EN: generation_config.json is HF's authority for generation; config.json often carries
     * a partial legacy copy. Union both -- an extra stop is harmless, a missing one is not. */
    { char gp[2100]; snprintf(gp,sizeof(gp),"%s/generation_config.json",snap);
      FILE *gf=fopen(gp,"rb");                  /* assente = nessun problema: e' opzionale */
      if(gf){
        fseek(gf,0,SEEK_END); long gn=ftell(gf); fseek(gf,0,SEEK_SET);
        char *gb = (gn>0 && gn<=(256L<<20)) ? malloc((size_t)gn+1) : NULL;   /* SEC: cap + NULL check */
        if(gb){
            size_t gg=fread(gb,1,(size_t)gn,gf); gb[gg]=0;
            char *ga=NULL; jval *gr=json_parse(gb,&ga);
            jval *ge=gr?json_get(gr,"eos_token_id"):NULL;
            if(ge){
                int add[8], na=0;
                if(ge->t==J_NUM) add[na++]=(int)ge->num;
                else if(ge->t==J_ARR) for(int i=0;i<ge->len && na<8;i++) add[na++]=(int)ge->kids[i]->num;
                for(int i=0;i<na && c->n_stop<8;i++){
                    int dup=0; for(int j=0;j<c->n_stop;j++) if(c->stop_ids[j]==add[i]) dup=1;
                    if(!dup) c->stop_ids[c->n_stop++]=add[i];
                }
            }
            free(ga); free(gb);
        }
        fclose(gf);
      } }
    /* DSA lightning indexer: parametri + tipo per-layer (lista esplicita o formula freq/offset) */
    c->index_topk=gi(r,"index_topk"); c->index_nh=gi(r,"index_n_heads"); c->index_hd=gi(r,"index_head_dim");
    { jval *it=json_get(r,"indexer_types");
      int freq=gi(r,"index_topk_freq"); if(freq<1) freq=1;
      jval *of=json_get(r,"index_skip_topk_offset"); int off=of?(int)of->num:2;
      for(int i=0;i<c->n_layers && i<128;i++){
          if(it && it->t==J_ARR && i<it->len && it->kids[i]->str)
              c->idx_type[i] = !strcmp(it->kids[i]->str,"full");
          else { int v=i-off+1; if(v<0) v=0; c->idx_type[i] = (v%freq)==0; }
      } }
    c->qk_head=c->qk_nope+c->qk_rope;
    c->attn_scale = 1.f / sqrtf((float)c->qk_head);
    if(c->n_group!=1){ fprintf(stderr,"this engine requires n_group=1 (GLM-5.2)\n"); exit(1); }
    /* VALIDAZIONE (report PR #25): il config.json arriva da mirror non fidati — dimensioni
     * ostili non devono superare questo punto. Un solo choke point protegge ogni alloc a valle. */
    #define CKR(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){ \
        fprintf(stderr,"config: %s=%d is outside [%d,%d]\n",name,(int)(v),(int)(lo),(int)(hi)); exit(1); }
    CKR("hidden_size",c->hidden,1,1<<20)         CKR("num_hidden_layers",c->n_layers,1,128)
    CKR("num_attention_heads",c->n_heads,1,1024) CKR("n_routed_experts",c->n_experts,1,4096)
    CKR("num_experts_per_tok",c->topk,1,64)      CKR("moe_intermediate_size",c->moe_inter,1,1<<20)
    CKR("intermediate_size",c->dense_inter,1,1<<24) CKR("first_k_dense_replace",c->first_dense,0,c->n_layers)
    CKR("q_lora_rank",c->q_lora,0,1<<20)         CKR("kv_lora_rank",c->kv_lora,1,1<<20)
    CKR("qk_nope_head_dim",c->qk_nope,1,1<<16)   CKR("qk_rope_head_dim",c->qk_rope,1,1<<16)
    CKR("v_head_dim",c->v_head,1,1<<16)          CKR("n_shared_experts",c->n_shared,0,64)
    CKR("vocab_size",c->vocab,1,1<<24)           CKR("index_topk",c->index_topk,0,1<<20)
    CKR("index_n_heads",c->index_nh,0,1024)      CKR("index_head_dim",c->index_hd,0,1<<16)
    #undef CKR
    free(ar);
}

/* Derive the fmt=4 group size from the scale-array byte count. A grouped-int4
 * tensor stores ceil(I/gs) f32 scales per output row, so:
 *     ns_bytes == O * ceil(I/gs) * 4   =>   gs == I * 4 / (ns_bytes/O - ... )
 * We probe candidate group sizes (must be a multiple of 16, the AVX2 vector
 * width the grouped kernel requires) from finest to coarsest and return the
 * first whose predicted scale-array size matches ns_bytes. Returns 0 if no
 * candidate fits (then it's plain per-row int4, fmt=2, not grouped).
 * Data-driven: g64/g128/g256 all just work; adding a size means listing it. */
static int detect_group_size(int O, int I, int64_t ns){
    if(O<=0 || ns<=(int64_t)O*4 || I<=0) return 0;   /* not grouped */
    /* ns/O is the per-row scale bytes; groups = (ns/O)/4; gs = ceil(I/groups).
     * Probe from small gs (finest granularity) upward so the most granular
     * match wins — that's what we want, since finer groups are unambiguous. */
    static const int cands[]={16,32,48,64,96,128,192,256};
    for(int ci=0; ci<(int)(sizeof(cands)/sizeof(cands[0])); ci++){
        int gs=cands[ci];
        if(gs>I) break;
        int ng=(I+gs-1)/gs;
        if(ns==(int64_t)O*ng*4) return gs;
    }
    return 0;
}

/* SEC: risolve e VALIDA il formato quantizzato di un tensore [O,I] letto da un
 * container non fidato (mirror). L'inferenza precedente (`?1:?2:3`) cadeva su
 * int2 per QUALSIASI conteggio byte non riconosciuto: un peso troppo corto
 * diventava un int2 valido e il matmul leggeva oltre il buffer (O*I nibble a
 * 4/byte). Qui i byte del peso devono corrispondere a un layout noto e i byte
 * della scala alla cardinalita' attesa (O per-row, O*ng per-gruppo) — altrimenti
 * si termina invece di sforare. Ritorna fmt (1/2/3/4) e scrive *gs. */
static int qt_resolve_fmt(const char *name, int O, int I, int64_t nb, int64_t ns, int *gs){
    int64_t exp_i8=(int64_t)O*I, exp_i4=(int64_t)O*((I+1)/2), exp_i2=(int64_t)O*((I+3)/4);
    int fmt = (nb==exp_i8)?1 : (nb==exp_i4)?2 : (nb==exp_i2)?3 : 0;
    if(!fmt){
        fprintf(stderr,"%s: quantized weight is %lld bytes — no int8/int4/int2 layout for [%d,%d], refusing (untrusted container)\n",
                name,(long long)nb,O,I); exit(1); }
    *gs=0;
    if(fmt==2){ int g=detect_group_size(O,I,ns); if(g>0){ fmt=4; *gs=g; } }
    int64_t exp_scale = (fmt==4)? (int64_t)O*((I+*gs-1)/(*gs)) : (int64_t)O;   /* in FLOAT */
    if(ns != exp_scale*4){
        fprintf(stderr,"%s: scale array is %lld bytes — expected %lld for [%d,%d] fmt=%d, refusing (untrusted container)\n",
                name,(long long)ns,(long long)(exp_scale*4),O,I,fmt); exit(1); }
    return fmt;
}
/* costruisce un QT [O,I] dal disco in `t` (buffer riusabili tra chiamate).
 *  - se esiste `name.qs`: pesi GIA' quantizzati nel container (U8 qdata + F32 scala) -> letti diretti
 *  - altrimenti: tensore pieno (f32/bf16) -> quantizzato a runtime a `bits` (oracolo tiny / pesi pieni)
 * drop=1 -> fadvise DONTNEED (streaming expert). */
static void qt_from_disk(Model *m, const char *name, int O, int I, int bits, int drop, QT *t){
    char sn[300]; snprintf(sn,sizeof(sn),"%s.qs",name);
    if(st_has(&m->S,sn)){
        int64_t nb=st_nbytes(&m->S,name);
        int64_t ns=st_nbytes(&m->S,sn);   /* scale bytes (F32) */
        /* fmt=4 int4-grouped: byte int4 ma scala > O*4 — gs deriva dalla scala.
         * qt_resolve_fmt valida entrambi i conteggi contro [O,I] e termina se
         * non fidati (SEC). */
        int gs=0;
        int fmt = qt_resolve_fmt(name,O,I,nb,ns,&gs);
        if(fmt==1){ if(t->fmt!=1||!t->q8){ t->fmt=1; t->O=O; t->I=I; t->gs=0; t->q8=qalloc(nb); t->s=qsalloc(O); } st_read_raw(&m->S,name,t->q8,drop); }
        else if(fmt==4){ int ng=(I+gs-1)/gs;
            if(t->fmt!=4||!t->q4){ t->fmt=4; t->O=O; t->I=I; t->gs=gs; t->q4=qalloc(nb); t->s=falloc((int64_t)O*ng); }
            st_read_raw(&m->S,name,t->q4,drop); }
        else      { if(t->fmt!=fmt||!t->q4){ t->fmt=fmt; t->O=O; t->I=I; t->gs=0; t->q4=qalloc(nb); t->s=qsalloc(O); } st_read_raw(&m->S,name,t->q4,drop); }
        st_read_f32(&m->S,sn,t->s,drop);
    } else {
        if(!t->qf && !t->q8 && !t->q4) qt_alloc(t,O,I,bits);
        if(t->fmt==0) st_read_f32(&m->S,name,t->qf,drop);
        else { float *tmp=falloc((int64_t)O*I); st_read_f32(&m->S,name,tmp,drop); qt_fill(t,tmp,bits); free(tmp); }
    }
}
static QT qt_load(Model *m, const char *name, int O, int I, int bits){
    QT t; memset(&t,0,sizeof(t)); qt_from_disk(m,name,O,I,bits,0,&t);
#ifdef COLI_CUDA
    if(g_cuda_enabled&&g_cuda_dense){
        t.cuda_eligible=1;
        int slot=g_cuda_rr++%g_cuda_ndev; t.cuda_device=g_cuda_devices[slot];
        g_cuda_dense_projected[slot]+=qt_bytes(&t);
    }
#endif
    return t;
}
static float *ld(Model *m, const char *name){   /* tensore 1D f32 residente (norme/bias) */
    int64_t n=st_numel(&m->S,name); if(n<0){fprintf(stderr,"missing %s\n",name);exit(1);}
    float *p=(float*)qalloc((size_t)n*sizeof(float));   /* registrato per la GPU sotto METAL */
    st_read_f32(&m->S,name,p,0); return p;
}
#ifdef COLI_CUDA
static void qt_cuda_colocate(QT *dst,const QT *src){
    if(!g_cuda_enabled||!g_cuda_dense||!dst->cuda_eligible||!src->cuda_eligible||
       dst->cuda_device==src->cuda_device)return;
    int old=-1,now=-1;for(int i=0;i<g_cuda_ndev;i++){
        if(g_cuda_devices[i]==dst->cuda_device)old=i;if(g_cuda_devices[i]==src->cuda_device)now=i;
    }
    if(old>=0)g_cuda_dense_projected[old]-=qt_bytes(dst);
    if(now>=0)g_cuda_dense_projected[now]+=qt_bytes(dst);
    dst->cuda_device=src->cuda_device;
}
static void layer_cuda_shard_kvb(Layer *l,int H,int Q,int V){
    if(!g_cuda_enabled||!g_cuda_dense||g_cuda_ndev<2||l->kv_b.fmt==0)return;
    int rb=l->kv_b.fmt==1?l->kv_b.I:(l->kv_b.fmt==2?(l->kv_b.I+1)/2:(l->kv_b.I+3)/4);
    const uint8_t *weights=l->kv_b.fmt==1?(const uint8_t*)l->kv_b.q8:l->kv_b.q4;
    for(int d=0,h0=0;d<g_cuda_ndev;d++){
        int hn=H/g_cuda_ndev+(d<H%g_cuda_ndev),rows=hn*(Q+V);
        const void *part=weights+(int64_t)h0*(Q+V)*rb;
        const float *scale=l->kv_b.s+(int64_t)h0*(Q+V);
        if(!coli_cuda_tensor_upload(&l->kv_b_shard[d],part,scale,l->kv_b.fmt,l->kv_b.I,rows,g_cuda_devices[d]))return;
        l->shard_h0[d]=h0;l->shard_hn[d]=hn;l->n_kv_b_shard++;h0+=hn;
    }
    int old=-1;for(int i=0;i<g_cuda_ndev;i++)if(g_cuda_devices[i]==l->kv_b.cuda_device)old=i;
    if(old>=0)g_cuda_dense_projected[old]-=qt_bytes(&l->kv_b);
    l->kv_b.cuda_eligible=0;
}
#endif

static void model_init(Model *m, const char *snap, int cap, int ebits, int dbits){
    memset(m,0,sizeof(*m)); m->ebits=ebits; m->dbits=dbits;
    load_cfg(&m->c,snap); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[256]; int H=c->n_heads, D=c->hidden;
    /* embed e lm_head sono il confine I/O: tenerli ad alta precisione (come i quant dynamic
     * reali). A bf16 ~1.9GB su GLM reale: trascurabile. dbits>=8 -> qui f32; piu' basso -> dbits. */
    int io_bits = dbits>=8 ? 16 : dbits;
    m->embed   = qt_load(m,"model.embed_tokens.weight", c->vocab, D, io_bits);
    m->lm_head = qt_load(m,"lm_head.weight", c->vocab, D, io_bits);
    m->final_norm = ld(m,"model.norm.weight");
    m->L=calloc(c->n_layers,sizeof(Layer));
    int NR=c->n_layers+1;                        /* +1: riga del layer MTP */
    m->ecap=cap; m->ecache=calloc(NR,sizeof(ESlot*)); m->ecn=calloc(NR,sizeof(int));
    m->kv_dev_L=calloc(NR,sizeof(float*)); m->kv_dev_R=calloc(NR,sizeof(float*));
    m->kv_dev_valid=calloc(NR,sizeof(int));
    m->eroute=calloc(NR,sizeof(int*)); m->enr=calloc(NR,sizeof(int));
    m->pin=calloc(NR,sizeof(ESlot*)); m->npin=calloc(NR,sizeof(int));
    m->eusage=calloc(NR,sizeof(uint32_t*)); m->eheat=calloc(NR,sizeof(uint32_t*));
    m->elast=calloc(NR,sizeof(uint32_t*));
    m->elast_dc=calloc(NR,sizeof(uint32_t*)); m->elast_pre=calloc(NR,sizeof(uint32_t*));
    m->kv=calloc(1,sizeof(KVState));
    m->kv_start=m->kv->kv_start=calloc(NR,sizeof(int));
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        #define P(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
        l->in_ln=ld(m,P("input_layernorm.weight"));
        l->post_ln=ld(m,P("post_attention_layernorm.weight"));
        l->q_a   = qt_load(m,P("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
        l->q_a_ln= ld(m,P("self_attn.q_a_layernorm.weight"));
        l->q_b   = qt_load(m,P("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
        l->kv_a  = qt_load(m,P("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
        l->kv_a_ln= ld(m,P("self_attn.kv_a_layernorm.weight"));
        l->kv_b  = qt_load(m,P("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
        l->o     = qt_load(m,P("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
#ifdef COLI_CUDA
        qt_cuda_colocate(&l->o,&l->kv_b);
        qt_cuda_colocate(&l->q_a,&l->kv_b);   /* PIPE: intera catena attention sulla */
        qt_cuda_colocate(&l->q_b,&l->kv_b);   /* stessa scheda / whole attention chain */
        qt_cuda_colocate(&l->kv_a,&l->kv_b);  /* on the layer home device */
        if(getenv("COLI_CUDA_ATTN_SHARD")&&atoi(getenv("COLI_CUDA_ATTN_SHARD")))
            layer_cuda_shard_kvb(l,H,c->qk_nope,c->v_head);
#endif
        l->sparse = (i >= c->first_dense);
        if(!l->sparse){
            l->gate_proj = qt_load(m,P("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
            l->up_proj   = qt_load(m,P("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
            l->down_proj = qt_load(m,P("mlp.down_proj.weight"), D, c->dense_inter, dbits);
        } else {
            l->router=ld(m,P("mlp.gate.weight"));
            l->router_bias=ld(m,P("mlp.gate.e_score_correction_bias"));
            int sI=c->moe_inter*c->n_shared;
            l->sh_gate = qt_load(m,P("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,P("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,P("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
#ifdef COLI_CUDA
            qt_cuda_colocate(&l->sh_gate,&l->kv_b);  /* PIPE2: shared chain on the layer home device */
            qt_cuda_colocate(&l->sh_up,&l->sh_gate);
            qt_cuda_colocate(&l->sh_down,&l->sh_gate);
#endif
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));      /* metodo C: ultimo routing del layer */
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast_dc[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast_pre[i]=calloc(c->n_experts,sizeof(uint32_t));
        }
        #undef P
    }
    /* testa MTP (layer n_layers): presente solo se convertita con --mtp */
    {
        /* MTP attiva SOLO se il set e' COMPLETO (i tensori vivono su 3 shard: durante la
         * conversione parziale ne esiste solo una parte). MTP=0 la disabilita comunque. */
        const char *req[]={"eh_proj.weight","enorm.weight","hnorm.weight","shared_head.norm.weight",
            "input_layernorm.weight","post_attention_layernorm.weight",
            "self_attn.q_a_proj.weight","self_attn.q_b_proj.weight","self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight","self_attn.o_proj.weight","mlp.gate.weight",
            "mlp.shared_experts.gate_proj.weight","mlp.shared_experts.down_proj.weight",
            "mlp.experts.0.gate_proj.weight"};
        char mn[256]; m->has_mtp=1;
        for(unsigned q=0;q<sizeof(req)/sizeof(req[0]);q++){
            snprintf(mn,sizeof(mn),"model.layers.%d.%s",c->n_layers,req[q]);
            if(!st_has(&m->S,mn)){ m->has_mtp=0; break; }
        }
        /* probe the LAST expert by index, not a fixed 255: REAP-pruned
         * checkpoints have n_routed_experts < 256 and the MTP set stays complete,
         * so a hardcoded expert.255 would spuriously report has_mtp=0 on them. */
        snprintf(mn,sizeof(mn),"model.layers.%d.mlp.experts.%d.down_proj.weight",c->n_layers,c->n_experts-1);
        if(!st_has(&m->S,mn)) m->has_mtp=0;
        if(getenv("MTP") && atoi(getenv("MTP"))==0) m->has_mtp=0;
        if(m->has_mtp){
            int i=c->n_layers; Layer *l=&m->mtpL;
            #define PM(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
            l->in_ln=ld(m,PM("input_layernorm.weight"));
            l->post_ln=ld(m,PM("post_attention_layernorm.weight"));
            l->q_a   = qt_load(m,PM("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
            l->q_a_ln= ld(m,PM("self_attn.q_a_layernorm.weight"));
            l->q_b   = qt_load(m,PM("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
            l->kv_a  = qt_load(m,PM("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
            l->kv_a_ln= ld(m,PM("self_attn.kv_a_layernorm.weight"));
            l->kv_b  = qt_load(m,PM("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
            l->o     = qt_load(m,PM("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
            l->sparse=1;
            l->router=ld(m,PM("mlp.gate.weight"));
            l->router_bias=ld(m,PM("mlp.gate.e_score_correction_bias"));
            int sI=c->moe_inter*c->n_shared;
            l->sh_gate = qt_load(m,PM("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,PM("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,PM("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
            m->eh_proj = qt_load(m,PM("eh_proj.weight"), D, 2*D, dbits);
            m->enorm=ld(m,PM("enorm.weight")); m->hnorm=ld(m,PM("hnorm.weight"));
            m->mtp_norm=ld(m,PM("shared_head.norm.weight"));
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast_dc[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast_pre[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->kv_start[i]=-1;                    /* KV MTP: parte dalla prima posizione di decode */
            #undef PM
        }
    }
    /* DSA lightning indexer: attivo SOLO se i pesi (conversione --indexer) ci sono per
     * TUTTI i layer full. Auto-rilevamento come per MTP: niente flag, niente passi extra. */
    {
        m->has_dsa = (c->index_topk>0 && c->index_nh>0 && c->index_hd>0 && c->index_hd<=256);
        char inm[300];
        for(int i=0;i<c->n_layers && m->has_dsa;i++){
            if(!c->idx_type[i]) continue;
            snprintf(inm,sizeof(inm),"model.layers.%d.self_attn.indexer.wq_b.weight",i);
            if(!st_has(&m->S,inm)) m->has_dsa=0;
        }
        if(getenv("DSA") && atoi(getenv("DSA"))==0) m->has_dsa=0;
        if(m->has_dsa){
            m->ix_wq=calloc(c->n_layers,sizeof(QT)); m->ix_wk=calloc(c->n_layers,sizeof(QT));
            m->ix_wp=calloc(c->n_layers,sizeof(QT));
            m->ix_knw=calloc(c->n_layers,sizeof(float*)); m->ix_knb=calloc(c->n_layers,sizeof(float*));
            for(int i=0;i<c->n_layers;i++){
                if(!c->idx_type[i]) continue;
                #define PI(s) (snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.indexer." s,i),nm)
                m->ix_wq[i]=qt_load(m,PI("wq_b.weight"), c->index_nh*c->index_hd, c->q_lora, dbits);
                m->ix_wk[i]=qt_load(m,PI("wk.weight"), c->index_hd, D, dbits);
                m->ix_wp[i]=qt_load(m,PI("weights_proj.weight"), c->index_nh, D, dbits);
                m->ix_knw[i]=ld(m,PI("k_norm.weight")); m->ix_knb[i]=ld(m,PI("k_norm.bias"));
                #undef PI
            }
            fprintf(stderr,"[DSA] indexer active: top-%d sparse attention beyond %d context tokens\n",
                c->index_topk, c->index_topk);
        }
    }
    m->hlast=falloc(D); m->h_all=falloc((int64_t)512*D);

    /* byte della parte DENSA residente (embed+lm_head+attn+mlp densa+shared+norme) */
    int64_t rb=qt_bytes(&m->embed)+qt_bytes(&m->lm_head);
    for(int i=0;i<c->n_layers;i++){ Layer *l=&m->L[i];
        rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
        if(!l->sparse) rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
        else rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down);
    }
    if(m->has_mtp){ Layer *l=&m->mtpL;
        rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
        rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down)+qt_bytes(&m->eh_proj);
    }
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        rb+=qt_bytes(&m->ix_wq[i])+qt_bytes(&m->ix_wk[i])+qt_bytes(&m->ix_wp[i]);
    m->resident_bytes=rb;
}

/* embed: dequantizza la riga del token (scala per-riga) in x[hidden] */
static void embed_row(Model *m, int tok, float *x){
    int D=m->c.hidden; QT *e=&m->embed;
    if(e->fmt==0){ memcpy(x, e->qf+(int64_t)tok*D, D*sizeof(float)); return; }
    if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)tok*D; float s=e->s[tok];
        for(int i=0;i<D;i++) x[i]=(float)q[i]*s; return; }
    if(e->fmt==2){ const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); float s=e->s[tok];   /* int4 */
        for(int i=0;i<D;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
            if(i+1<D) x[i+1]=(float)((int)(byte>>4)-8)*s; }
        return; }
    const uint8_t *q=e->q4+(int64_t)tok*((D+3)/4); float s=e->s[tok];   /* int2 */
    for(int i=0;i<D;i++){ uint8_t byte=q[i>>2]; int sh=(i&3)*2; x[i]=(float)((int)((byte>>sh)&3)-2)*s; }
}

/* COLI_MMAP=1: gli expert diventano VISTE dentro mmap dei file safetensors (niente pread,
 * niente slab, niente copia: la page cache del kernel E' la cache). Le mappe sono
 * registrate con Metal (newBufferWithBytesNoCopy su pagine file-backed, come llama.cpp),
 * quindi la GPU legge gli stessi byte. Fallback allo slab path su disallineamento. */
static int g_mmap=0;
static struct { int fd; void *base; size_t len; } g_maps[512]; static int g_nmaps;
static pthread_mutex_t g_map_mtx = PTHREAD_MUTEX_INITIALIZER;   /* expert_load e' OMP-parallel */
/* forward decls: mem_should_wire/mem_wire live near pin_wire() further down, but
 * qt_wire_mmap() (also further down, used by pin_wire()'s COLI_MMAP path) needs
 * them declared before its own definition. Real mlock-ing of mmap'd pinned
 * experts happens there, not in expert_load() -- see qt_wire_mmap() for why. */
static int mem_should_wire(void);
static int mem_wire(void *addr, size_t len);
static void qt_unwire_mmap(QT *t);   /* def. presso pin_wire / defined near pin_wire */
static int64_t g_mmap_wired=0; static long g_mmap_wire_failed=0;
static void *map_of_fd(int fd){
    pthread_mutex_lock(&g_map_mtx);
    for(int i=0;i<g_nmaps;i++) if(g_maps[i].fd==fd){ void *b=g_maps[i].base; pthread_mutex_unlock(&g_map_mtx); return b; }
    void *base=NULL;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    struct stat st;
    if(g_nmaps<512 && fstat(fd,&st)==0){
        size_t len=((size_t)st.st_size+16383)&~(size_t)16383;
        void *p=mmap(NULL,len,PROT_READ,MAP_SHARED,fd,0);
        if(p!=MAP_FAILED){
            base=p; g_maps[g_nmaps].fd=fd; g_maps[g_nmaps].base=p; g_maps[g_nmaps].len=len; g_nmaps++;
#ifdef COLI_METAL
            if(g_metal_enabled) coli_metal_register(p,len);
#endif
        }
    }
#endif
    pthread_mutex_unlock(&g_map_mtx);
    return base;
}

/* carica un expert nello slot. Container pre-quantizzato: le 3 matrici sono contigue nel
 * file -> UNA pread coalescente da ~19 MB dentro `slab` (+ le scale in fslab); i QT sono
 * viste dentro lo slab (zero copie). Fallback per modelli non quantizzati (oracolo tiny).
 * THREAD-SAFE su slot distinti (pread posizionale, st_find read-only). */
/* Load one expert's weights into slot `s`. Returns 0 on success, -1 on failure.
 * fatal=1 (all main / on-demand / REPIN / pin callers): preserve the original
 * exit-on-error contract byte-for-byte — any missing tensor, OOM, short read or
 * pread error aborts the process. fatal=0 (speculative pilot only): the same
 * errors instead abandon the load and return -1 without touching s->eid, so a
 * mispredicted cross-layer prefetch can never kill the server. */
/* pread completo: gestisce le short-read (POSIX le ammette su file regolari
 * sotto pressione di memoria) e le EINTR, e riporta un errore ONESTO. perror
 * stampava "Success" quando pread ritorna un conteggio corto invece di -1
 * (errno resta 0 dalla syscall precedente) -> messaggio fuorviante nel path
 * score/bench (#236). Ritorna 0 = ok, -1 = errore reale o EOF. */
static int pread_full(int fd, void *buf, int64_t n, int64_t off, const char *tag){
    char *p=buf; int64_t got=0;
    while(got<n){
        ssize_t r=pread(fd, p+got, (size_t)(n-got), off+got);
        if(r<0){ if(errno==EINTR) continue;
#ifdef _WIN32
            fprintf(stderr,"%s: %s (off %lld, %lld/%lld bytes, WinErr=%lu)\n",tag,strerror(errno),
                    (long long)off,(long long)got,(long long)n,(unsigned long)compat_pread_lasterr);
#else
            fprintf(stderr,"%s: %s (off %lld, %lld/%lld bytes)\n",tag,strerror(errno),
                    (long long)off,(long long)got,(long long)n);
#endif
            return -1; }
        if(r==0){ fprintf(stderr,"%s: short read at EOF (off %lld, %lld/%lld bytes) — truncated shard?\n",
                    tag,(long long)off,(long long)got,(long long)n); return -1; }
        got+=r;
    }
    return 0;
}
/* DISK-CLASS: is classification active right now? Single point of truth shared by
 * moe()'s pre-bump snapshot (FASE A, below) and expert_load_impl's classify-and-read --
 * they must agree, or expert_load_impl reads a snapshot moe() never bothered to write.
 * PROF=1 is the only trigger: this feature is measurement-only, the verdict never picks
 * an fd (kept as one function anyway so a future policy consumer cannot drift out of
 * sync with the snapshot writer by construction). */
static int dc_needed(void){ return g_prof; }
/* DISK-CLASS: cold/warm verdict for ONE demand load, read from the pre-bump snapshot
 * (Model's elast_pre) so routing's OWN bump for THIS call can't contaminate the read --
 * prefill is one giant moe() call where every newly-seen expert gets its `last` bumped
 * a few lines above the load that made it "new"; classifying off the live, post-bump
 * array would call the whole cold burst warm. Ages against the PRIVATE clock
 * (eaccess_clock_dc, see its declaration in Model), NEVER the real eaccess_clock: kept
 * separate by design (see elast_dc in Model for why DISK-CLASS keeps its own clock
 * instead of reading the real one -- before #417/cfcc742 the real one also froze on the
 * Metal pre-routed decode path, which would have made every prefill-touched expert warm
 * forever and everything else cold forever; that's fixed upstream now, but DISK-CLASS
 * still doesn't read the real clock, for the isolation property, not the freeze).
 * Conservative-toward-warm on the one genuinely ambiguous input (missing snapshot --
 * defensive only, elast_pre is allocated everywhere elast is); a demonstrable first-ever
 * access (last_pre==0) is not a judgment call, it stays cold regardless of that bias. */
static int expert_classify(Model *m, int layer, int eid){
    if(!m->elast_pre || !m->elast_pre[layer]) return DC_WARM;  /* no snapshot: label as the safe class */
    uint32_t last_pre=m->elast_pre[layer][eid];
    if(last_pre==0) return DC_COLD;                             /* never touched before this call: certain cold */
    uint32_t age=m->eaccess_clock_dc-last_pre;                  /* ticks since last access, PRE this call's bump */
    return age>g_direct_heat_ticks ? DC_COLD : DC_WARM;         /* '>' not '>=': ties lean warm */
}
static int expert_load_impl(Model *m, int layer, int eid, ESlot *s, int fatal, int demand){
#ifdef COLI_CUDA
    /* A live REPIN may reuse a GPU-enabled pinned slot for a different expert.
     * Keep its tier assignment, but invalidate the old device weights. */
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
    char nm[3][288]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[300]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(!st_has(&m->S,qn)){                       /* fallback: tensori pieni, quantizza a runtime.
                                                  * Reachable ONLY for unquantized models (no .qs);
                                                  * GLM always has .qs, so the pilot never hits it. */
        qt_from_disk(m,nm[0],I,D,b,g_drop,&s->g);
        qt_from_disk(m,nm[1],I,D,b,g_drop,&s->u);
        qt_from_disk(m,nm[2],D,I,b,g_drop,&s->d);
        atomic_fetch_add_explicit(&g_prof_io,
            st_nbytes(&m->S,nm[0])+st_nbytes(&m->S,nm[1])+st_nbytes(&m->S,nm[2]),memory_order_relaxed);
        s->eid=eid; return 0;
    }
    st_tensor *tw[3], *tq[3];
    for(int k=0;k<3;k++){
        tw[k]=st_find(&m->S,nm[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm[k]); tq[k]=st_find(&m->S,qn);
        if(!tw[k]||!tq[k]){ fprintf(stderr,"missing %s\n",nm[k]); if(fatal) exit(1); return -1; }
    }
    if(g_disk_split){ /* split load/byte per tipo layer; atomici: expert_load gira anche su OMP/pipe/pilot */
        int64_t tb=0; for(int k=0;k<3;k++) tb+=tw[k]->nbytes+tq[k]->nbytes;
        if(layer==c->n_layers){ __atomic_add_fetch(&m->ld_mtp,1,__ATOMIC_RELAXED);
                                __atomic_add_fetch(&m->bytes_mtp,(uint64_t)tb,__ATOMIC_RELAXED); }
        else                  { __atomic_add_fetch(&m->ld_main,1,__ATOMIC_RELAXED);
                                __atomic_add_fetch(&m->bytes_main,(uint64_t)tb,__ATOMIC_RELAXED); }
    }
    if(g_mmap){
        void *bw[3],*bq[3]; int okm=1;
        for(int k=0;k<3;k++){
            bw[k]=map_of_fd(tw[k]->fd); bq[k]=map_of_fd(tq[k]->fd);
            if(!bw[k]||!bq[k]||((tw[k]->off)&3)||((tq[k]->off)&3)) okm=0;
        }
        if(okm){
            QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
            for(int k=0;k<3;k++){
                int64_t nb=tw[k]->nbytes;
                int gs=0;
                int fmt=qt_resolve_fmt(tw[k]->name,OO[k],II[k],nb,tq[k]->nbytes,&gs);
                qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->gs=gs; qt[k]->qf=NULL;
                qt[k]->q8=(int8_t*)((char*)bw[k]+tw[k]->off); qt[k]->q4=(uint8_t*)((char*)bw[k]+tw[k]->off);
                qt[k]->s=(float*)((char*)bq[k]+tq[k]->off);
            }
            /* CPU pre-touch: fault the pages in HERE (cheap, parallel, overlapped with the
             * resident-experts GPU submit) so the GPU never demand-faults file-backed pages
             * (measured catastrophic). madvise starts async readahead, the touch guarantees
             * residency. This is pread's I/O without the copy and without the slab. */
            for(int k=0;k<3;k++){
                char *p=(char*)bw[k]+tw[k]->off; size_t n=(size_t)tw[k]->nbytes;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
                madvise((void*)((uintptr_t)p & ~16383UL), n+16384, MADV_WILLNEED);
#endif
                volatile char acc=0;
                for(size_t i=0;i<n;i+=4096) acc+=p[i];
                acc+=p[n-1]; (void)acc;
                char *q=(char*)bq[k]+tq[k]->off; size_t nq=(size_t)tq[k]->nbytes;
                for(size_t i=0;i<nq;i+=4096) acc+=q[i];
                /* mlock deliberately NOT done here: this fires for every expert_load call,
                 * including the transient VRAM-staging pass in pin_load (host copy loaded,
                 * uploaded to GPU, then "released" via expert_host_release -- which only
                 * knows how to munlock s->slab, always NULL under mmap, so wiring here would
                 * leak locked pages for every GPU-tier expert). See pin_wire() below: it wires
                 * the final resident set only, after GPU release has already nulled out the
                 * pointers for anything that isn't genuinely RAM-tier. */
                atomic_fetch_add_explicit(&g_prof_io,(int64_t)(n+nq),memory_order_relaxed);
            }
            s->eid=eid; return 0;
        }
    }
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    /* rialloca se lo slot (riusato tra layer) e' troppo piccolo per QUESTO expert:
     * pread oltre la mappatura = short-read o CORRUZIONE silenziosa dei vicini */
    if(!s->slab || wtot+8192 > s->slab_cap){
#ifdef COLI_METAL
        /* page-align + zero-copy wrap: the GPU reads this slab in place (unified memory) */
        if(s->slab && g_metal_enabled) coli_metal_unregister(s->slab);
        compat_aligned_free(s->slab);
        size_t need=((size_t)wtot+8192+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->slab,16384,need)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=need;
        if(g_metal_enabled) coli_metal_register(s->slab,need);
#else
        compat_aligned_free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,wtot+8192)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=wtot+8192;
        numa_slab_bind(s->slab,(size_t)s->slab_cap);
#endif
    }
    if(!s->fslab || ftot > s->fslab_cap){
#ifdef COLI_METAL
        /* page-align + register: the GPU reads the scales in place (unified memory).
         * Honours `fatal` exactly like the CPU arm below — a speculative pilot load
         * that hits OOM must unwind into a clean hidden slot, never exit(). */
        if(s->fslab && g_metal_enabled) coli_metal_unregister(s->fslab);
        free(s->fslab);
        size_t fb=(((size_t)ftot*sizeof(float))+16383)&~(size_t)16383;
        if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
           posix_memalign((void**)&s->fslab,16384,fb)){
            fprintf(stderr,"OOM fslab\n"); if(fatal) exit(1);
            compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0;  /* clean, hidden slot (eid stays -1) */
            s->fslab=NULL; s->fslab_cap=0; return -1;
        }
        s->fslab_cap=ftot;
        if(g_metal_enabled) coli_metal_register(s->fslab,fb);
#else
        free(s->fslab);
        if(fatal){ s->fslab=falloc(ftot); }          /* main path: byte-identical exit-on-OOM */
        else {                                        /* speculative pilot: checked alloc, never exit() */
            /* replicate falloc's anti-wrap guard + malloc (no zeroing/alignment) */
            if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
               !(s->fslab=malloc((size_t)ftot*sizeof(float)))){
                fprintf(stderr,"OOM fslab\n");
                compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0; /* leave a clean, hidden slot (eid stays -1) */
                s->fslab=NULL; s->fslab_cap=0; return -1;
            }
        }
        s->fslab_cap=ftot;
        numa_slab_bind(s->fslab,(size_t)ftot*sizeof(float));
#endif
    }
    /* DISK-CLASS: classify before the reads; computed unconditionally at dc_on sites so
     * the timer (dc_t0) brackets exactly the read work, matching what the GB/s in the
     * DISK-CLASS line describes. dc_on gates ALL of it off demand=0 call sites
     * (pilot/repin/pin -- never classified, see the call sites) and off PROF=0 runs
     * (dc_needed()) -- zero cost, zero behavior change there. The fd choice below is
     * NOT influenced by the verdict: this is measurement only. */
    int dc_on = demand && dc_needed();
    int dc_cls = dc_on ? expert_classify(m,layer,eid) : DC_WARM;
    double dc_t0 = dc_on ? now_s() : 0;
    if(dc_on) dc_wall_enter(dc_cls,dc_t0);        /* busy-wall open; EVERY exit path below must pair it */
    int ord[3]={0,1,2};                          /* ordina per offset nel file */
    for(int a=0;a<3;a++) for(int bb=a+1;bb<3;bb++) if(tw[ord[bb]]->off<tw[ord[a]]->off){ int t=ord[a]; ord[a]=ord[bb]; ord[bb]=t; }
    int contig = tw[ord[0]]->fd==tw[ord[1]]->fd && tw[ord[1]]->fd==tw[ord[2]]->fd
              && tw[ord[0]]->off+tw[ord[0]]->nbytes==tw[ord[1]]->off
              && tw[ord[1]]->off+tw[ord[1]]->nbytes==tw[ord[2]]->off;
    int64_t pos[3]; int done=0, dc_direct=0;
    if(contig){
        int64_t off0=tw[ord[0]]->off;
        int dfd = g_direct ? st_direct_fd(&m->S, tw[ord[0]]->fd) : -1;
        if(dfd>=0){                              /* O_DIRECT: offset/len allineati a 4K */
            int64_t base=off0 & ~4095LL, need=(off0-base)+wtot;
            int64_t len=(need+4095)&~4095LL;
            ssize_t r=pread(dfd, s->slab, len, base);
            if(r>=need){
                pos[ord[0]]=off0-base; pos[ord[1]]=pos[ord[0]]+tw[ord[0]]->nbytes;
                pos[ord[2]]=pos[ord[1]]+tw[ord[1]]->nbytes; done=1; dc_direct=1;
            }
        }
        if(!done){                               /* fallback bufferizzato */
            if(pread_full(tw[ord[0]]->fd, s->slab, wtot, off0, "pread expert")){ if(fatal) exit(1);
                if(dc_on) dc_wall_exit(dc_cls,now_s());   /* pair the enter on the non-fatal unwind */
                return -1; }
            pos[ord[0]]=0; pos[ord[1]]=tw[ord[0]]->nbytes; pos[ord[2]]=tw[ord[0]]->nbytes+tw[ord[1]]->nbytes; done=1;
        }
    }
    if(!done){                                   /* non contigui: 3 pread bufferizzate */
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a];
            if(pread_full(tw[k]->fd, s->slab+o, tw[k]->nbytes, tw[k]->off, "pread expert")){ if(fatal) exit(1);
                if(dc_on) dc_wall_exit(dc_cls,now_s());   /* pair the enter on the non-fatal unwind */
                return -1; }
            pos[k]=o; o+=tw[k]->nbytes; }
    }
    float *fp[3]; int64_t fo=0;                  /* scale (piccole) */
    for(int k=0;k<3;k++){
        if(pread_full(tq[k]->fd, (char*)(s->fslab+fo), tq[k]->nbytes, tq[k]->off, "pread qs")){ if(fatal) exit(1);
            if(dc_on) dc_wall_exit(dc_cls,now_s());       /* pair the enter on the non-fatal unwind */
            return -1; }
        fp[k]=s->fslab+fo; fo+=tq[k]->nbytes/4; }
    atomic_fetch_add_explicit(&g_prof_io,wtot+fo*4,memory_order_relaxed);
    if(dc_on){                                    /* DISK-CLASS accounting, see dc_needed() */
        double dc_t1=now_s();                     /* one clock read for thread-ns AND the wall exit */
        int64_t bytes=wtot+fo*4;
        atomic_fetch_add_explicit(&g_dc_n[dc_cls],1,memory_order_relaxed);
        atomic_fetch_add_explicit(&g_dc_bytes[dc_cls],bytes,memory_order_relaxed);
        atomic_fetch_add_explicit(&g_dc_ns[dc_cls],(int64_t)((dc_t1-dc_t0)*1e9),memory_order_relaxed);
        dc_wall_exit(dc_cls,dc_t1);
        if(dc_direct)                             /* which fd ACTUALLY served this class */
            atomic_fetch_add_explicit(&g_dc_direct_n[dc_cls],1,memory_order_relaxed);
    }
    if(g_drop){                                  /* scarta subito le pagine: evita che la page
                                                  * cache in pressione strangoli il throughput */
        posix_fadvise(tw[ord[0]]->fd, tw[ord[0]]->off, wtot, POSIX_FADV_DONTNEED);
        for(int k=0;k<3;k++) posix_fadvise(tq[k]->fd, tq[k]->off, tq[k]->nbytes, POSIX_FADV_DONTNEED);
    }
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
    for(int k=0;k<3;k++){
        int64_t nb=tw[k]->nbytes;
        int gs=0;
        int fmt=qt_resolve_fmt(tw[k]->name,OO[k],II[k],nb,tq[k]->nbytes,&gs);
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->gs=gs; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+pos[k]); qt[k]->q4=s->slab+pos[k]; qt[k]->s=fp[k];
    }
    s->eid=eid; return 0;
}
/* Every expert read goes through here: time the whole load (pread/fault +
 * bookkeeping) on the thread that runs it, into the disk-service counter. */
static int expert_load(Model *m, int layer, int eid, ESlot *s, int fatal, int demand){
    /* `demand` marks a routing-driven demand-load (moe()'s PIPE/OMP miss path, where the
     * pre-bump elast_pre snapshot moe() just wrote is valid) -- pass 0 from anywhere else
     * (pilot speculative loads, repin, startup PIN loading): those never run through THIS
     * call's own FASE A, so the snapshot either doesn't apply or was never written for
     * them, and DISK-CLASS deliberately leaves them unclassified -- see expert_classify()'s
     * call site. */
    double t0=now_s();
    int rc=expert_load_impl(m,layer,eid,s,fatal,demand);
    atomic_fetch_add_explicit(&g_edisk_ns,(int64_t)((now_s()-t0)*1e9),memory_order_relaxed);
    return rc;
}

#ifdef __linux__
/* io_uring expert batches.  One owner prepares all reads for a block, submits
 * them in one syscall, and reaps CQEs on demand.  The kernel, rather than a set
 * of blocking pthreads, owns the I/O concurrency. */
#define URING_LOAD_MAX 64
#define URING_REQ_MAX  512
typedef struct {
    int load, expect;
} UringRead;
typedef struct {
    Model *m; ESlot *s; int layer,eid,fatal;
    st_tensor *tw[3],*tq[3]; int64_t pos[3];
    int pending,done,finalized,error;
} UringLoad;
typedef struct {
    ColiUring ring;
    UringLoad load[URING_LOAD_MAX];
    UringRead req[URING_REQ_MAX];
    int nload,nreq,started;
} UringBatch;
static UringBatch g_ub_pipe, g_ub_pilot;

static int uring_batch_init(UringBatch *b){
    if(b->started) return 0;
    if(coli_uring_init(&b->ring,URING_REQ_MAX)) return -1;
    b->started=1; return 0;
}
static void uring_batch_reset(UringBatch *b){
    b->nload=0; b->nreq=0;
}
static int uring_load_error(UringLoad *l,int err,const char *what){
    l->error=err?err:EIO; l->done=1;
    if(l->fatal){ errno=l->error; perror(what); exit(1); }
    return -1;
}
static int uring_add_read(UringBatch *b,int li,int fd,void *buf,size_t len,
                          int64_t off,size_t expect){
    if(b->nreq>=URING_REQ_MAX || expect>INT_MAX){ errno=E2BIG; return -1; }
    int ri=b->nreq++;
    b->req[ri]=(UringRead){li,(int)expect};
    if(coli_uring_prep_read(&b->ring,fd,buf,len,off,(uint64_t)ri+1)) return -1;
    b->load[li].pending++;
    return 0;
}
/* Returns the load index. URING is intentionally a quantized streaming path;
 * unsupported layouts fail instead of silently dropping back to pread. */
static int uring_load_add(UringBatch *b,Model *m,int layer,int eid,ESlot *s,int fatal){
    if(b->nload>=URING_LOAD_MAX){ errno=E2BIG; return -1; }
    int li=b->nload++;
    UringLoad *l=&b->load[li]; memset(l,0,sizeof(*l));
    l->m=m; l->s=s; l->layer=layer; l->eid=eid; l->fatal=fatal;
    char nm[3][288],qn[300]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(g_mmap || !st_has(&m->S,qn))
        return uring_load_error(l,ENOTSUP,"URING requires quantized expert tensors"),li;
#ifdef COLI_CUDA
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
    for(int k=0;k<3;k++){
        l->tw[k]=st_find(&m->S,nm[k]);
        size_t n=strnlen(nm[k],sizeof(nm[k]));
        if(n+3>=sizeof(qn)) return uring_load_error(l,ENAMETOOLONG,"io_uring expert metadata"),li;
        memcpy(qn,nm[k],n); memcpy(qn+n,".qs",4); l->tq[k]=st_find(&m->S,qn);
        if(!l->tw[k]||!l->tq[k]) return uring_load_error(l,ENOENT,"io_uring expert metadata"),li;
    }
    int64_t wtot=l->tw[0]->nbytes+l->tw[1]->nbytes+l->tw[2]->nbytes;
    int64_t ftot=(l->tq[0]->nbytes+l->tq[1]->nbytes+l->tq[2]->nbytes)/4;
    if(wtot<=0 || ftot<=0) return uring_load_error(l,EINVAL,"io_uring expert size"),li;
    if(!s->slab || wtot+8192>s->slab_cap){
#ifdef COLI_METAL
        if(s->slab&&g_metal_enabled) coli_metal_unregister(s->slab);
        compat_aligned_free(s->slab);
        size_t need=((size_t)wtot+8192+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->slab,16384,need)){
            s->slab=NULL; s->slab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert slab"),li; }
        s->slab_cap=need; if(g_metal_enabled) coli_metal_register(s->slab,need);
#else
        compat_aligned_free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,(size_t)wtot+8192)){
            s->slab=NULL; s->slab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert slab"),li; }
        s->slab_cap=wtot+8192;
#endif
    }
    if(!s->fslab || ftot>s->fslab_cap){
#ifdef COLI_METAL
        if(s->fslab&&g_metal_enabled) coli_metal_unregister(s->fslab);
        free(s->fslab); size_t fb=(((size_t)ftot*sizeof(float))+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->fslab,16384,fb)){
            s->fslab=NULL; s->fslab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert scales"),li; }
        s->fslab_cap=ftot; if(g_metal_enabled) coli_metal_register(s->fslab,fb);
#else
        free(s->fslab); s->fslab=malloc((size_t)ftot*sizeof(float));
        if(!s->fslab){ s->fslab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert scales"),li; }
        s->fslab_cap=ftot;
#endif
    }
    int ord[3]={0,1,2};
    for(int a=0;a<3;a++) for(int z=a+1;z<3;z++) if(l->tw[ord[z]]->off<l->tw[ord[a]]->off){int t=ord[a];ord[a]=ord[z];ord[z]=t;}
    int contig=l->tw[ord[0]]->fd==l->tw[ord[1]]->fd && l->tw[ord[1]]->fd==l->tw[ord[2]]->fd
        && l->tw[ord[0]]->off+l->tw[ord[0]]->nbytes==l->tw[ord[1]]->off
        && l->tw[ord[1]]->off+l->tw[ord[1]]->nbytes==l->tw[ord[2]]->off;
    if(contig){
        int64_t off0=l->tw[ord[0]]->off;
        int dfd=g_direct?st_direct_fd(&m->S,l->tw[ord[0]]->fd):-1;
        if(dfd>=0){
            int64_t base=off0&~4095LL,need=(off0-base)+wtot,len=(need+4095)&~4095LL;
            l->pos[ord[0]]=off0-base; l->pos[ord[1]]=l->pos[ord[0]]+l->tw[ord[0]]->nbytes;
            l->pos[ord[2]]=l->pos[ord[1]]+l->tw[ord[1]]->nbytes;
            if(uring_add_read(b,li,dfd,s->slab,(size_t)len,base,(size_t)need))
                return uring_load_error(l,errno,"io_uring direct expert read"),li;
        }else{
            l->pos[ord[0]]=0; l->pos[ord[1]]=l->tw[ord[0]]->nbytes;
            l->pos[ord[2]]=l->pos[ord[1]]+l->tw[ord[1]]->nbytes;
            if(uring_add_read(b,li,l->tw[ord[0]]->fd,s->slab,(size_t)wtot,off0,(size_t)wtot))
                return uring_load_error(l,errno,"io_uring expert read"),li;
        }
    }else{
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a]; l->pos[k]=o;
            if(uring_add_read(b,li,l->tw[k]->fd,s->slab+o,(size_t)l->tw[k]->nbytes,l->tw[k]->off,(size_t)l->tw[k]->nbytes))
                return uring_load_error(l,errno,"io_uring expert read"),li;
            o+=l->tw[k]->nbytes;
        }
    }
    int64_t fo=0;
    for(int k=0;k<3;k++){
        if(uring_add_read(b,li,l->tq[k]->fd,s->fslab+fo,(size_t)l->tq[k]->nbytes,l->tq[k]->off,(size_t)l->tq[k]->nbytes))
            return uring_load_error(l,errno,"io_uring expert scale read"),li;
        fo+=l->tq[k]->nbytes/4;
    }
    return li;
}
static void uring_reap(UringBatch *b){
    struct io_uring_cqe cqe;
    while(coli_uring_peek(&b->ring,&cqe)){
        if(!cqe.user_data || cqe.user_data>(uint64_t)b->nreq) continue;
        UringRead *r=&b->req[cqe.user_data-1]; UringLoad *l=&b->load[r->load];
        if(cqe.res<r->expect && !l->error) l->error=cqe.res<0?-cqe.res:EIO;
        if(l->pending>0) l->pending--;
        if(l->pending==0) l->done=1;
    }
}
static int uring_submit_batch(UringBatch *b){
    if(coli_uring_enter(&b->ring,0)<0) return -1;
    uring_reap(b); return 0;
}
static int uring_wait_load(UringBatch *b,int li){
    UringLoad *l=&b->load[li];
    while(!l->done){
        uring_reap(b); if(l->done) break;
        if(coli_uring_enter(&b->ring,1)<0) return uring_load_error(l,errno,"io_uring wait");
    }
    return l->error?-1:0;
}
static int uring_finalize_load(UringBatch *b,int li,int publish_eid){
    UringLoad *l=&b->load[li]; ESlot *s=l->s;
    if(l->finalized) return 0;
    if(uring_wait_load(b,li)<0){ errno=l->error; if(l->fatal){perror("io_uring expert completion");exit(1);} return -1; }
    if(g_drop){
        int ord0=0; for(int k=1;k<3;k++) if(l->tw[k]->off<l->tw[ord0]->off) ord0=k;
        int64_t wtot=l->tw[0]->nbytes+l->tw[1]->nbytes+l->tw[2]->nbytes;
        posix_fadvise(l->tw[ord0]->fd,l->tw[ord0]->off,wtot,POSIX_FADV_DONTNEED);
        for(int k=0;k<3;k++) posix_fadvise(l->tq[k]->fd,l->tq[k]->off,l->tq[k]->nbytes,POSIX_FADV_DONTNEED);
    }
    Cfg *c=&l->m->c; int I=c->moe_inter,D=c->hidden; float *fp[3]; int64_t fo=0;
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D},II[3]={D,D,I};
    for(int k=0;k<3;k++){
        fp[k]=s->fslab+fo; fo+=l->tq[k]->nbytes/4;
        int64_t nb=l->tw[k]->nbytes;
        int fmt=(nb==(int64_t)OO[k]*II[k])?1:(nb==(int64_t)OO[k]*((II[k]+1)/2))?2:3;
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+l->pos[k]); qt[k]->q4=s->slab+l->pos[k]; qt[k]->s=fp[k];
    }
    if(publish_eid) s->eid=l->eid;
    l->finalized=1; return 0;
}
static int uring_wait_all(UringBatch *b){
    for(int i=0;i<b->nload;i++) if(uring_wait_load(b,i)<0) return -1;
    return 0;
}
#endif

/* ============================ PIPE: load ‖ matmul ============================
 * Overlap NVMe expert-weight loads with expert matmul. A small persistent pool
 * of I/O worker pthreads runs the misses' pread (expert_load) into distinct
 * ws[] slabs and sets a per-slot `ready` flag; the MAIN thread walks the block's
 * experts in order, waiting on ready[q] only for the expert it needs right now,
 * and does all matmul_qt on itself (matmul_qt parallelises internally via OpenMP
 * and checks !omp_in_parallel() for GPU dispatch — so it must stay off the omp
 * team and off these I/O threads).
 *
 * Cross-generation safety is provided by a single generation-tagged, lock-free
 * cursor `cur = (gen<<8) | index`. The main thread is the sole writer of `gen`
 * (monotonic bump, so no ABA); workers grab jobs by CAS-advancing the low 8-bit
 * index. THE INVARIANT: a worker reads eids[i]/layer only AFTER its winning CAS,
 * and that CAS's comparand carries the generation — so if `cur`'s gen advanced
 * (a new batch was published), the CAS fails and the worker re-reads, seeing the
 * new generation. A straggler preempted anywhere (wake gap, post-cursor) can
 * therefore NEVER grab a wrong-generation job or read torn batch state: its
 * first act is a gen-checked CAS. dispatch publishes all batch state with
 * relaxed stores and then RELEASE-stores `cur`; each worker ACQUIRE-loads `cur`,
 * so the ready[] reset + eids[]/njobs/layer are visible before any worker acts.
 * The per-expert pipe_wait(ready[q]) in the matmul loop makes every grabbed job
 * complete before the block ends, so no grab outlives its generation — which is
 * why the old `active` counter AND the end-of-block drain barrier are gone (both
 * were redundant with those per-slot waits + the gen-tagged cursor). The mutex/
 * condvar exist ONLY to park/wake idle workers, never for correctness. Gated
 * behind PIPE=1; OFF => the original blocking-load + serial-matmul path runs
 * byte-identically. */
static int g_pipe=0;      /* PIPE=1: async expert-load pipeline. Default ON for Windows
                           * (parsed in main: getenv("PIPE")?:1 on _WIN32, :0 elsewhere).
                           * Keeps expert pread off the forward-pass thread so loads overlap
                           * the matmul. PIPE=0 opts back into the blocking serial path. */
static int g_pipe_nw=8;   /* PIPE_WORKERS=n: I/O worker threads (disk-parallel reads) */
static int g_uring=0;     /* URING=1: Linux io_uring load/completion backend; implies PIPE */
typedef struct {
    _Atomic uint64_t cur;                         /* (gen<<8)|index; gen main-only, index 0..njobs (≤64) */
    _Atomic int njobs;                            /* current batch job count */
    _Atomic int eids[64];                         /* current batch expert ids */
    _Atomic int layer;                            /* current batch layer */
    _Atomic int ready[64];                        /* per-slot load-done flag */
    pthread_mutex_t mx; pthread_cond_t cv;        /* ONLY for parking/waking idle workers */
    Model *m;
    pthread_t th[16]; int nw; int started;
} PipePool;
static PipePool g_pp;

static void *pipe_worker(void *arg){
    (void)arg; PipePool *p=&g_pp; uint64_t seen=0;
    for(;;){
        pthread_mutex_lock(&p->mx);
        while((atomic_load_explicit(&p->cur,memory_order_relaxed)>>8)==seen)
            pthread_cond_wait(&p->cv,&p->mx);
        pthread_mutex_unlock(&p->mx);
        for(;;){
            uint64_t c=atomic_load_explicit(&p->cur,memory_order_acquire);
            seen=c>>8;
            uint32_t i=(uint32_t)(c & 0xFF);
            if(i >= (uint32_t)atomic_load_explicit(&p->njobs,memory_order_relaxed))
                break;                                /* batch drained → re-park */
            if(atomic_compare_exchange_weak_explicit(&p->cur,&c,c+1,
                    memory_order_acq_rel,memory_order_relaxed)){
                int L  =atomic_load_explicit(&p->layer,memory_order_relaxed);
                int eid=atomic_load_explicit(&p->eids[i],memory_order_relaxed); /* AFTER winning CAS */
                expert_load(p->m,L,eid,&p->m->ws[i],1,1);  /* needed-now load: fatal on I/O error (matches serial path); demand=1: this IS moe()'s own miss path */
                atomic_store_explicit(&p->ready[i],1,memory_order_release);
            }
            /* CAS failed → another worker advanced index (or gen advanced): re-loop */
        }
    }
    return NULL;
}
static void pipe_init(Model *m){
    if(g_pp.started) return;
#ifdef __linux__
    if(g_uring){
        if(uring_batch_init(&g_ub_pipe)){ perror("URING=1 io_uring_setup"); exit(1); }
        g_pp.m=m; g_pp.started=1; return;
    }
#endif
    g_pp.m=m; g_pp.nw=g_pipe_nw; if(g_pp.nw>16) g_pp.nw=16; if(g_pp.nw<1) g_pp.nw=1;
    atomic_store(&g_pp.cur,0); atomic_store(&g_pp.njobs,0);
    pthread_mutex_init(&g_pp.mx,NULL); pthread_cond_init(&g_pp.cv,NULL);
    for(int i=0;i<g_pp.nw;i++) pthread_create(&g_pp.th[i],NULL,pipe_worker,NULL);
    g_pp.started=1;
}
/* enqueue `njobs` loads (slots ws[0..njobs)); returns immediately, workers run ahead.
 * Order is load-bearing: write all batch state RELAXED, then RELEASE-store cur to
 * publish it, then wake parked workers. */
static void pipe_dispatch(Model *m,int layer,const int *eids,int njobs){
#ifdef __linux__
    if(g_uring){
        uring_batch_reset(&g_ub_pipe);
        for(int q=0;q<njobs;q++){
            int li=uring_load_add(&g_ub_pipe,m,layer,eids[q],&m->ws[q],1);
            if(li!=q){ fprintf(stderr,"URING: expert batch overflow\n"); exit(1); }
        }
        if(uring_submit_batch(&g_ub_pipe)){ perror("URING: submit"); exit(1); }
        return;
    }
#endif
    g_pp.m=m;
    atomic_store_explicit(&g_pp.njobs,njobs,memory_order_relaxed);
    atomic_store_explicit(&g_pp.layer,layer,memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.eids[q],eids[q],memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.ready[q],0,memory_order_relaxed); /* reset BEFORE publish */
    uint64_t g=(atomic_load_explicit(&g_pp.cur,memory_order_relaxed)>>8)+1;
    atomic_store_explicit(&g_pp.cur,(g<<8),memory_order_release);                          /* PUBLISH */
    pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv); pthread_mutex_unlock(&g_pp.mx);
}
static inline void pipe_wait(int q){
#ifdef __linux__
    if(g_uring){
        if(uring_finalize_load(&g_ub_pipe,q,1)){ perror("URING: expert load"); exit(1); }
        return;
    }
#endif
    while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) sched_yield();
}

#ifdef COLI_CUDA
static void expert_host_release(Model *m, ESlot *s){
    if(!s->slab&&!s->fslab) return;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    if(s->slab) munlock(s->slab,(size_t)s->slab_cap);
    if(s->fslab) munlock(s->fslab,(size_t)s->fslab_cap*sizeof(float));
#elif defined(_WIN32)
    if(s->slab) compat_munlock(s->slab,(size_t)s->slab_cap);
    if(s->fslab) compat_munlock(s->fslab,(size_t)s->fslab_cap*sizeof(float));
#endif
    int64_t bytes=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
    /* slab is posix_memalign'd: on Windows that is _aligned_malloc, and plain
     * free() corrupts the CRT heap (0xC0000374) — same bug the compat.h audit
     * fixed at the original expert_load site. fslab is plain malloc/falloc
     * on the CPU path, so its free() stays plain (Metal path frees it before
     * re-alloc and never reaches here with an aligned fslab on _WIN32). */
    compat_aligned_free(s->slab); free(s->fslab); s->slab=NULL; s->fslab=NULL; s->slab_cap=s->fslab_cap=0;
    QT *q[3]={&s->g,&s->u,&s->d};
    for(int k=0;k<3;k++){ q[k]->qf=NULL; q[k]->q8=NULL; q[k]->q4=NULL; q[k]->s=NULL; }
    m->resident_bytes-=bytes; if(m->resident_bytes<0) m->resident_bytes=0;
}
static void expert_host_ensure(Model *m, int layer, ESlot *s){
    if(!s->slab) expert_load(m,layer,s->eid,s,1,0);   /* re-materializing a GPU-resident expert's host copy, not a routing miss: demand=0 */
}
#endif

/* prefetch asincrono dei pesi di un expert (e delle sue scale .qs): avvia il readahead
 * cosi' le letture sincrone successive trovano la page-cache calda. */
static void expert_prefetch(Model *m, int layer, int eid){
    char nm[300];
    const char *suf[3]={"gate_proj.weight","up_proj.weight","down_proj.weight"};
    for(int k=0;k<3;k++){
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s",layer,eid,suf[k]); st_prefetch(&m->S,nm);
        char qs[320]; snprintf(qs,sizeof(qs),"%s.qs",nm); st_prefetch(&m->S,qs);
    }
}

/* ---- helper per l'ABSORPTION: accesso per-riga ai QT quantizzati ---- */
/* acc[0..I) += coef * W[row,:] (dequant al volo) */
static void qt_addrow(const QT *t, int row, float coef, float *acc){
    int I=t->I;
    if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=coef*w[i]; return; }
    /* fmt=4 PRIMA del calcolo di c: s[] e' [O,ng] per-gruppo, s[row] sarebbe la scala
     * sbagliata. Senza questo ramo il fall-through int2 decodificava i nibble int4 come
     * coppie di valori a 2 bit — lo stesso bug di #298 sui kernel absorb CUDA, lato CPU.
     * EN: fmt=4 BEFORE computing c: s[] is [O,ng] per-group, s[row] would be the wrong
     * scale. Without this branch the int2 fall-through decoded int4 nibbles as pairs of
     * 2-bit values — the same bug #298 fixed in the CUDA absorb kernels, CPU side. */
    if(t->fmt==4){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
        int gs=t->gs, ng=(I+gs-1)/gs; const float *scl=t->s+(int64_t)row*ng;
        for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1];
            acc[i]  +=coef*scl[i/gs]    *((int)(b&0xF)-8);
            acc[i+1]+=coef*scl[(i+1)/gs]*((int)(b>>4)-8); }
        if(I&1){ uint8_t b=w[I>>1]; acc[I-1]+=coef*scl[(I-1)/gs]*((int)(b&0xF)-8); } return; }
    float c=coef*t->s[row];
    if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=c*(float)w[i]; return; }
    if(t->fmt==2){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
        for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc[i]+=c*((int)(b&0xF)-8); acc[i+1]+=c*((int)(b>>4)-8); }
        if(I&1){ uint8_t b=w[I>>1]; acc[I-1]+=c*((int)(b&0xF)-8); } return; }
    const uint8_t *w=t->q4+(int64_t)row*((I+3)/4);
    for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; acc[i]+=c*((int)((b>>((i&3)*2))&3)-2); }
}
/* y[0..n) = W[r0+j,:]·x  (matvec su una FETTA di righe del QT) */
static void qt_matvec_rows(const QT *t, int r0, int n, const float *x, float *y){
    int I=t->I;
    for(int j=0;j<n;j++){ int row=r0+j; double a=0;
        if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) a+=(double)w[i]*x[i]; }
        else if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; float s=t->s[row];
            float acc=0; for(int i=0;i<I;i++) acc+=(float)w[i]*x[i]; a=acc*s; }
        else if(t->fmt==2){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2); float s=t->s[row]; float acc=0;
            for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
            if(I&1){ uint8_t b=w[I>>1]; acc+=((int)(b&0xF)-8)*x[I-1]; } a=acc*s; }
        else if(t->fmt==4){ /* per-gruppo, come matmul_i4_grouped / per-group, as matmul_i4_grouped */
            const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
            int gs=t->gs, ng=(I+gs-1)/gs; const float *scl=t->s+(int64_t)row*ng;
            for(int g=0; g*gs<I; g++){ int base=g*gs, end=base+gs>I?I:base+gs; float acc=0;
                for(int i=base;i<end;i++){ uint8_t b=w[i>>1];
                    acc+=(float)((i&1)?((int)(b>>4)-8):((int)(b&0xF)-8))*x[i]; }
                a+=(double)acc*scl[g]; } }
        else { const uint8_t *w=t->q4+(int64_t)row*((I+3)/4); float s=t->s[row]; float acc=0;
            for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; acc+=((int)((b>>((i&3)*2))&3)-2)*x[i]; } a=acc*s; }
        y[j]=(float)a;
    }
}
static int g_absorb=-1;
#ifdef COLI_CUDA
static int g_cuda_pipe=0;   /* COLI_CUDA_PIPE=1: prefill attention chain resident on the layer home device */
#endif   /* ABSORB: -1 auto (decode S<=4), 0 mai, 1 sempre (test) */
static int g_dsa_force=0; /* DSA_FORCE=1: selezione sempre attiva (test: top-min(k,T)=denso) */
static int cmp_fdesc(const void *a,const void *b){
    float x=*(const float*)a, y=*(const float*)b; return x<y?1:x>y?-1:0; }

/* PARTIAL SELECT (quickselect, Hoare partition, DESCending). After this call the k
 * LARGEST elements of a[0..n) are in a[0..k) in unspecified order; the (k+1)-th and
 * beyond are untouched-or-smaller. O(n) average, O(n^2) pathological (mitigated by
 * median-of-three below) — and unlike a full qsort it never orders more than needed.
 *
 * Why this exists (#356): the DSA top-keep in attention_rows previously full-qsorted
 * all nk context scores (O(nk log nk)) per layer per token just to read ONE value --
 * the keep-th largest (the threshold). quickselect finds that pivot in O(nk) average,
 * and the position-order scans that build dst[] are unchanged, so the kept set is
 * bit-identical. Mirrors the sampling-side fix in #335 (heap partial-select there).
 *
 * NOT a stable partition: callers must derive the threshold and then re-scan the
 * ORIGINAL array (the DSA code does exactly this) rather than reading a[0..k). */
static void partial_select_desc(float *a, int n, int k){
    if(k<=0) return;
    if(k>=n) return;                 /* nothing to partition: all kept */
    int lo=0, hi=n-1;
    while(lo<hi){
        /* median-of-three pivot to dodge the O(n^2) path on sorted/reverse input */
        int mid=lo+((hi-lo)>>1);
        if(a[mid]>a[lo]){ float t=a[lo]; a[lo]=a[mid]; a[mid]=t; }
        if(a[hi]>a[lo]){  float t=a[lo]; a[lo]=a[hi];  a[hi]=t;  }
        if(a[mid]>a[hi]){ float t=a[hi]; a[hi]=a[mid]; a[mid]=t; }
        float piv=a[hi];
        int i=lo, j=hi;
        for(;;){
            while(a[i]>piv) i++;     /* desc: large values go left */
            while(j>lo && a[j]<piv) j--;
            if(i>=j) break;
            float t=a[i]; a[i]=a[j]; a[j]=t; i++; if(i>j) break; j--;
        }
        /* partition point: a[lo..i) are all >= piv, a[i..hi] are all <= piv */
        if(k<=i-1) hi=i-1;          /* the k-th largest is in the left partition */
        else       lo=i;            /* it's in the right partition */
    }
}

/* attenzione MLA con KV-cache compressa, su token nuovi x[S,hidden], pos_base = pos del primo */
/* kvs/pos describe a ragged decode batch: each row may belong to a different
 * sequence.  NULL keeps the original contiguous, currently-bound KV path. */
#ifdef COLI_CUDA
/* Ombra KV su device per il DECODE: righe [0,upto) valide sulla scheda di kv_b.
 * L'host resta canonico; l'ombra si riallinea in blocco quando resta indietro e
 * viene invalidata da kv_bind / dalla riscrittura di righe gia' specchiate. */
static int kv_dev_sync(Model *m, Layer *l, int layer, int upto){
    Cfg *c=&m->c; int kvl=c->kv_lora, R=c->qk_rope, dev=l->kv_b.cuda_device;
    if(upto>m->max_t) return 0;
    if(!m->kv_dev_L[layer]){
        m->kv_dev_L[layer]=(float*)coli_cuda_pipe_alloc(dev,(size_t)m->max_t*kvl*4);
        m->kv_dev_R[layer]=(float*)coli_cuda_pipe_alloc(dev,(size_t)m->max_t*R*4);
        m->kv_dev_valid[layer]=0;
        if(!m->kv_dev_L[layer]||!m->kv_dev_R[layer]) return 0;
    }
    int v=m->kv_dev_valid[layer];
    if(v<upto){
        if(!coli_cuda_pipe_upload(dev,m->kv_dev_L[layer]+(size_t)v*kvl,
            coli_kv_row(m->kv->Lc[layer],v,kvl),(size_t)(upto-v)*kvl*4)||
           !coli_cuda_pipe_upload(dev,m->kv_dev_R[layer]+(size_t)v*R,
            coli_kv_row(m->kv->Rc[layer],v,R),(size_t)(upto-v)*R*4)) return 0;
        m->kv_dev_valid[layer]=upto;
    }
    return 1;
}

/* Inc.1a — catena attention residente sul device del layer / attention chain
 * resident on the layer home device. Proiezioni q/kv, norme, RoPE, batch
 * attention e o_proj girano sulla scheda di kv_b; scaricano solo out [S,D],
 * i nuovi record KV [S,kvl+R] e nulla altro. Ritorna 0 su qualsiasi errore:
 * il chiamante riesegue il percorso CPU (idempotente). */
static int attn_pipe_prefill(Model *m, Layer *l, int layer, const float *x, int x_is_dev,
                             int S, int pos_base, float *out, float *out_dev){
    Cfg *c=&m->c; int H=c->n_heads, D=c->hidden, qh=c->qk_head;
    int kvl=c->kv_lora, R=c->qk_rope, ql=c->q_lora;
    int dev=l->kv_b.cuda_device;
    if(l->q_a.cuda_device!=dev||l->q_b.cuda_device!=dev||
       l->kv_a.cuda_device!=dev||l->o.cuda_device!=dev) return 0;
    int st0=m->kv_start[layer], T=pos_base+S-st0, old=pos_base-st0;
    if(T<S||T>8192) return 0;
    double t0=now_s();
    size_t xb=(size_t)S*D*4, qrb=(size_t)S*ql*4, qb=(size_t)S*H*qh*4;
    size_t cb=(size_t)S*(kvl+R)*4, lb=(size_t)T*kvl*4, rb=(size_t)T*R*4;
    float *chost=NULL; int ok=0;
    /* scratch persistenti (slot fissi per device): zero churn di cudaMalloc */
    float *xd =x_is_dev?(float*)x:coli_cuda_pipe_scratch(dev,0,xb);
    float *qrd=coli_cuda_pipe_scratch(dev,1,qrb);
    float *qd =coli_cuda_pipe_scratch(dev,2,qb),  *cd =coli_cuda_pipe_scratch(dev,3,cb);
    float *ld_=coli_cuda_pipe_scratch(dev,4,lb),  *rd =coli_cuda_pipe_scratch(dev,5,rb);
    float *w1 =coli_cuda_pipe_scratch(dev,6,(size_t)ql*4);
    float *w2 =coli_cuda_pipe_scratch(dev,7,(size_t)kvl*4);
    chost=(float*)malloc(cb);
    if(!xd||!qrd||!qd||!cd||!ld_||!rd||!w1||!w2||!chost) goto done;
    if((!x_is_dev&&!coli_cuda_pipe_upload(dev,xd,x,xb))||
       !coli_cuda_pipe_upload(dev,w1,l->q_a_ln,(size_t)ql*4)||
       !coli_cuda_pipe_upload(dev,w2,l->kv_a_ln,(size_t)kvl*4)) goto done;
    /* proiezioni + norme + rope, tutto sul device */
    if(!coli_cuda_pipe_gemm(l->q_a.cuda,qrd,xd,S)) goto done;
    if(!coli_cuda_pipe_rmsnorm(dev,qrd,qrd,w1,S,ql,c->eps)) goto done;
    if(!coli_cuda_pipe_gemm(l->q_b.cuda,qd,qrd,S)) goto done;
    if(!coli_cuda_pipe_rope_base(dev,qd,pos_base,S*H,qh,c->qk_nope,R,H,c->theta)) goto done;
    if(!coli_cuda_pipe_gemm(l->kv_a.cuda,cd,xd,S)) goto done;
    if(!coli_cuda_pipe_rmsnorm_s(dev,cd,cd,w2,S,kvl,c->eps,kvl+R,kvl+R)) goto done;
    if(!coli_cuda_pipe_rope_base(dev,cd,pos_base,S,kvl+R,kvl,R,1,c->theta)) goto done;
    /* cache latente [T,kvl] + rot [T,R] contigue: righe vecchie da host, nuove da cd */
    if(old>0){
        if(!coli_cuda_pipe_upload(dev,ld_,coli_kv_row(m->Lc[layer],st0,kvl),(size_t)old*kvl*4)||
           !coli_cuda_pipe_upload(dev,rd,coli_kv_row(m->Rc[layer],st0,R),(size_t)old*R*4)) goto done;
    }
    if(!coli_cuda_pipe_copy2d(dev,ld_+(size_t)old*kvl,kvl,cd,kvl+R,kvl,S)) goto done;
    if(!coli_cuda_pipe_copy2d(dev,rd+(size_t)old*R,R,cd+kvl,kvl+R,R,S)) goto done;
    /* KV host resta canonica: scarica i record nuovi (gia' normati+ropati) */
    if(!coli_cuda_pipe_download(dev,cd,chost,cb)) goto done;
    for(int s=0;s<S;s++){
        memcpy(coli_kv_row(m->Lc[layer],pos_base+s,kvl),chost+(size_t)s*(kvl+R),kvl*4);
        memcpy(coli_kv_row(m->Rc[layer],pos_base+s,R),chost+(size_t)s*(kvl+R)+kvl,R*4);
    }
    if(m->kv_dev_valid[layer]>pos_base) m->kv_dev_valid[layer]=pos_base;
    m->t_aproj+=now_s()-t0; t0=now_s();
#ifdef COLI_CUDA
    /* Negativo (2026-07-13): P2P a stella dal device di casa serializza ~95MB/layer
     * sul suo link PCIe — attention 26->41-44s. Resta opt-in per topologie NVLink. */
    if(out_dev && l->n_kv_b_shard>1 &&
       getenv("COLI_CUDA_PIPE_SHARD") && atoi(getenv("COLI_CUDA_PIPE_SHARD"))){
        /* head-shard nel pipeline: q gia' sul device di casa. Per ogni scheda:
         * slice di q (repack strided->contiguo), broadcast latent+rope via P2P,
         * score parallelo sui rispettivi head, ctx slice riportata a casa e
         * ricomposta, poi o_proj residente. */
        int n=l->n_kv_b_shard, vh=c->v_head;
        size_t ctxb=(size_t)S*H*vh*4;
        size_t stage_one=(size_t)S*H*(size_t)(c->qk_head>vh?c->qk_head:vh)*4;
        float *ctx_full=coli_cuda_pipe_scratch(dev,16,ctxb);
        float *stage=coli_cuda_pipe_scratch(dev,17,stage_one*n);
        int ok_sh=(ctx_full&&stage)?1:0;
        if(ok_sh){
            #pragma omp parallel for schedule(static) reduction(&:ok_sh)
            for(int d2=0;d2<n;d2++){
                int hn=l->shard_hn[d2], h0=l->shard_h0[d2];
                int sdev=coli_cuda_tensor_device(l->kv_b_shard[d2]);
                float *st=stage+(size_t)d2*(stage_one/4);
                size_t qsb=(size_t)S*hn*qh*4, csb=(size_t)S*hn*vh*4;
                float *qs_r=coli_cuda_pipe_scratch(sdev,18,qsb);
                float *ld_r=coli_cuda_pipe_scratch(sdev,19,(size_t)T*kvl*4);
                float *rr_r=coli_cuda_pipe_scratch(sdev,20,(size_t)T*R*4);
                float *cx_r=coli_cuda_pipe_scratch(sdev,21,csb);
                int okd=qs_r&&ld_r&&rr_r&&cx_r;
                /* slice di q: [S,H,qh] -> [S,hn,qh] contigua sul device di casa */
                okd=okd&&coli_cuda_pipe_copy2d(dev,st,hn*qh,qd+(size_t)h0*qh,H*qh,hn*qh,S);
                okd=okd&&coli_cuda_pipe_peer_copy(sdev,qs_r,dev,st,qsb);
                okd=okd&&coli_cuda_pipe_peer_copy(sdev,ld_r,dev,ld_,(size_t)T*kvl*4);
                okd=okd&&coli_cuda_pipe_peer_copy(sdev,rr_r,dev,rd,(size_t)T*R*4);
                okd=okd&&coli_cuda_attention_absorb_batch_dev(l->kv_b_shard[d2],cx_r,qs_r,ld_r,rr_r,
                        S,hn,c->qk_nope,R,vh,kvl,T,c->attn_scale);
                okd=okd&&coli_cuda_pipe_peer_copy(dev,st,sdev,cx_r,csb);
                okd=okd&&coli_cuda_pipe_copy2d(dev,ctx_full+(size_t)h0*vh,H*vh,st,hn*vh,hn*vh,S);
                ok_sh&=okd;
            }
        }
        if(ok_sh){
            ok=coli_cuda_pipe_gemm(l->o.cuda,out_dev,ctx_full,S)&&coli_cuda_pipe_sync(dev);
        } else ok=0;
        if(!ok)
            ok=coli_cuda_attention_project_batch_dev_out(l->kv_b.cuda,l->o.cuda,out_dev,qd,ld_,rd,
                S,H,c->qk_nope,R,c->v_head,kvl,T,c->attn_scale);
    } else
#endif
    ok=out_dev?coli_cuda_attention_project_batch_dev_out(l->kv_b.cuda,l->o.cuda,out_dev,qd,ld_,rd,
            S,H,c->qk_nope,R,c->v_head,kvl,T,c->attn_scale)
              :coli_cuda_attention_project_batch_dev(l->kv_b.cuda,l->o.cuda,out,qd,ld_,rd,
            S,H,c->qk_nope,R,c->v_head,kvl,T,c->attn_scale);
    m->t_acore+=now_s()-t0;
done:
    free(chost);                              /* gli scratch device restano al contesto */
    return ok;
}
#endif

static void attention_rows(Model *m, Layer *l, int layer, float *x, int S, int pos_base,
                           KVState *const *kvs, const int *positions, float *out){
    Cfg *c=&m->c; int H=c->n_heads, D=c->hidden, qh=c->qk_head, vh=c->v_head;
    int kvb_dim=H*(c->qk_nope+vh), Tk=pos_base+S;
    double ta0=now_s();
#ifdef COLI_METAL
    /* Fused decode attention on GPU: whole layer in one command buffer (keeps the GPU hot).
     * S<=4 absorption path with st0==0, DSA selection inactive, and GLM-5.2 int4 dims.
     * RAGGED GUARD (!kvs): the kernel takes ONE Lc/Rc pair and ONE pos_base — it assumes
     * row s is token pos_base+s of the SAME sequence. The batched mux decode
     * (step_decode_batch) passes per-row kvs[]/positions[] with pos_base=0, so the kernel
     * would rope every row at position 0 and attend over a 1-token window of the wrong
     * cache -> greedy decode hits EOS at token 2 (mux answers truncated to 1 token).
     * Ragged rows take the CPU absorb path below, which reads kvs[s]/positions[s]. */
    if(g_metal_enabled && !kvs && S<=4 && (g_absorb==1||(g_absorb<0&&S<=4)) && m->kv_start[layer]==0
       && D==6144 && H==64 && c->q_lora==2048 && c->kv_lora==512 && c->qk_nope==192
       && c->qk_rope==64 && vh==256 && l->kv_b.fmt==2){
        int sel_active = m->has_dsa && layer<c->n_layers && c->idx_type[layer] && (pos_base+S) > c->index_topk;
        if(!sel_active){
            if(m->has_dsa && layer<c->n_layers && c->idx_type[layer]){   /* index keys for future selection */
                for(int s=0;s<S;s++){ int pos=pos_base+s; float *kd=m->Ic[layer]+(int64_t)pos*c->index_hd;
                    matmul_qt(kd, x+(int64_t)s*D, &m->ix_wk[layer], 1);
                    layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], c->index_hd, 1e-6f);
                    rope_interleave(kd, pos, c); }
            }
            #define WP_(q) ((q).fmt==1?(const void*)(q).q8:(const void*)(q).q4)
            int ok = coli_metal_attn_decode(x,
                WP_(l->q_a), l->q_a.s, l->q_a.fmt, l->q_a_ln,
                WP_(l->q_b), l->q_b.s, l->q_b.fmt,
                WP_(l->kv_a), l->kv_a.s, l->kv_a.fmt, l->kv_a_ln,
                WP_(l->kv_b), l->kv_b.s, l->kv_b.fmt,
                WP_(l->o), l->o.s, l->o.fmt,
                m->Lc[layer], m->Rc[layer], S, pos_base, m->kv_start[layer], c->eps, c->theta, c->attn_scale, out);
            #undef WP_
            if(ok){ m->t_attn += now_s()-ta0; return; }
        }
    }
#endif
    float *ctx=falloc((int64_t)S*H*vh);
    float *Q=falloc((int64_t)S*H*qh);                  /* query (roped) dei token nuovi */
    int cw=c->kv_lora+c->qk_rope;
    float *QR=falloc((int64_t)S*c->q_lora), *comp=falloc((int64_t)S*cw);
    /* 1) query roped + latente normato e k_rot roped -> in cache.
     * QR tiene il residuo q_a per TUTTE le posizioni: serve anche all'indexer DSA.
     *
     * BATCH-ROWS: le tre proiezioni girano su tutte le S righe in un colpo solo, come gia' fa
     * o_proj (matmul_qt(...,S) sotto) e come fa moe() con la batch-union. Una riga per volta
     * il peso veniva ri-letto per OGNI token; a S righe si legge una volta sola.
     * matmul_qt_ex(...,0): restano sul kernel int4 ESATTO. Con l'IDOT (che il gate S>=g_i4s
     * abiliterebbe da solo appena S>1) il prefill sarebbe molto piu' veloce ma la qualita'
     * cala: -5040.33 -> -5158.68 di log-lik su 1023 token (~+12% perplexity). Il batch da
     * solo e' bit-identical all'originale; il kernel no. Vedi issue.
     * EN: batch the three projections over all S rows, like o_proj below and moe()'s
     * batch-union. matmul_qt_ex(...,0) keeps them on the EXACT int4 kernel: letting S>1 pull
     * them into IDOT is much faster but costs ~12% perplexity (measured). Batching alone is
     * bit-identical to upstream; the kernel switch is not. */
    int pipe_done=0;
#ifdef COLI_CUDA
    if(g_cuda_pipe&&!kvs&&S>=8&&layer<c->n_layers&&g_cuda_enabled&&c->kv_lora<=512&&
       !(m->has_dsa&&pos_base+S>c->index_topk)&&
       l->q_a.cuda_eligible&&l->q_b.cuda_eligible&&l->kv_a.cuda_eligible&&
       l->kv_b.cuda_eligible&&l->o.cuda_eligible&&
       qt_cuda_upload(&l->q_a)&&qt_cuda_upload(&l->q_b)&&qt_cuda_upload(&l->kv_a)&&
       qt_cuda_upload(&l->kv_b)&&qt_cuda_upload(&l->o))
        pipe_done=attn_pipe_prefill(m,l,layer,x,0,S,pos_base,out,NULL);
#endif
    if(!pipe_done){
        matmul_qt_ex(QR, x, &l->q_a, S, 0);
        for(int s=0;s<S;s++){ float *qr=QR+(int64_t)s*c->q_lora;
            rmsnorm(qr, qr, l->q_a_ln, c->q_lora, c->eps); }         /* q_b legge il residuo NORMATO */
        matmul_qt_ex(Q, QR, &l->q_b, S, 0);
        matmul_qt_ex(comp, x, &l->kv_a, S, 0);
    }
    if(!pipe_done) for(int s=0;s<S;s++){
        KVState *ks=kvs?kvs[s]:m->kv;
        int pos=positions?positions[s]:pos_base+s;
        float *qfull=Q+(int64_t)s*H*qh;
        for(int h=0;h<H;h++) rope_interleave(qfull+(int64_t)h*qh+c->qk_nope, pos, c);
        const float *cs=comp+(int64_t)s*cw;
        float *Ldst=coli_kv_row(ks->Lc[layer],pos,c->kv_lora);
        float *Rdst=coli_kv_row(ks->Rc[layer],pos,c->qk_rope);
#ifdef COLI_CUDA
        if(ks==m->kv&&m->kv_dev_valid&&layer<=c->n_layers&&m->kv_dev_valid[layer]>pos)
            m->kv_dev_valid[layer]=pos;              /* riga riscritta: l'ombra si accorcia */
#endif
        memcpy(Ldst, cs, c->kv_lora*sizeof(float));
        rmsnorm(Ldst, Ldst, l->kv_a_ln, c->kv_lora, c->eps);     /* latente normato */
        memcpy(Rdst, cs+c->kv_lora, c->qk_rope*sizeof(float));
        rope_interleave(Rdst, pos, c);                            /* k_rot roped, condiviso fra teste */
    }
    /* ---- DSA lightning indexer ----
     * Layer FULL: k_idx dei token nuovi in cache + selezione top-k per query (riusata
     * dai layer SHARED successivi). Selezione attiva solo con contesto > index_topk
     * (o DSA_FORCE=1 per il test: selezionare TUTTO deve dare l'output denso esatto). */
    const int *dsel=NULL, *dnsel=NULL; int dtopk=0;
    if(m->has_dsa && layer<c->n_layers && ((!kvs && m->kv_start[layer]==0) || kvs)){
        int nh=c->index_nh, hd=c->index_hd; dtopk=c->index_topk;
        if(c->idx_type[layer]){
            /* BATCH-ROWS, come le proiezioni di attenzione sopra: ix_wk (D x index_hd) veniva
             * ri-letto per OGNI token. matmul_qt_ex(...,0) lo tiene sul kernel int4 ESATTO:
             * il batch da solo supererebbe il gate S>=g_i4s e cambierebbe la quantizzazione
             * delle attivazioni. Cosi' l'output resta bit-identical.
             * EN: batch ix_wk over all S rows like the attention projections; allow_idot=0
             * keeps it on the exact int4 kernel so the result stays bit-identical. */
            float *KD=falloc((int64_t)S*hd);
            matmul_qt_ex(KD, x, &m->ix_wk[layer], S, 0);
            for(int s=0;s<S;s++){
                KVState *ks=kvs?kvs[s]:m->kv;
                int pos=positions?positions[s]:pos_base+s;
                float *kd=coli_kv_row(ks->Ic[layer],pos,hd);
                memcpy(kd, KD+(int64_t)s*hd, (size_t)hd*sizeof(float));
                layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], hd, 1e-6f);
                rope_interleave(kd, pos, c);                 /* primi qk_rope dim, interleaved */
            }
            free(KD);
            if((int64_t)S*dtopk > m->dsa_scap){
                free(m->dsa_sel); free(m->dsa_nsel);
                m->dsa_scap=(int64_t)S*dtopk;
                m->dsa_sel=malloc((size_t)m->dsa_scap*sizeof(int));
                m->dsa_nsel=malloc((size_t)S*sizeof(int));
            }
            #pragma omp parallel for schedule(dynamic,1)
            for(int s=0;s<S;s++){
                KVState *ks=kvs?kvs[s]:m->kv;
                int pos=positions?positions[s]:pos_base+s, nk=pos+1;
                if(ks->kv_start[layer]!=0){ m->dsa_nsel[s]=0; continue; }
                if(nk<=dtopk && !g_dsa_force){ m->dsa_nsel[s]=0; continue; }
                int keep = nk<dtopk ? nk : dtopk;
                float *qi=falloc((int64_t)nh*hd);
                matmul_qt(qi, QR+(int64_t)s*c->q_lora, &m->ix_wq[layer], 1);
                for(int h=0;h<nh;h++) rope_interleave(qi+(int64_t)h*hd, pos, c);
                float *w32=falloc(nh);
                matmul_qt(w32, x+(int64_t)s*D, &m->ix_wp[layer], 1);
                float wsc=1.f/sqrtf((float)nh), rs=1.f/sqrtf((float)hd);
                float *isc=falloc(nk);
                for(int t=0;t<nk;t++){
                    const float *kt=coli_kv_row(ks->Ic[layer],t,hd);
                    float a=0;
                    for(int h=0;h<nh;h++){ const float *qhp=qi+(int64_t)h*hd;
                        float d0=0; for(int i=0;i<hd;i++) d0+=qhp[i]*kt[i];
                        d0*=rs; if(d0>0) a+=w32[h]*d0;       /* ReLU sullo score, poi peso */
                    }
                    isc[t]=a*wsc;
                }
                /* top-keep: threshold via PARTIAL SELECT (#356), poi scan in ordine di posizione.
                 * Era un qsort completo su nk (O(nk log nk)); quickselect estrae solo il
                 * keep-esimo valore piu' grande in O(nk) medio. La soglia (= min del blocco
                 * dei keep maggiori) e' identica a tmp[keep-1] del vecchio qsort, quindi i
                 * due scan qui sotto costruiscono dst[] bit-identical. */
                float *tmp=falloc(nk); memcpy(tmp,isc,nk*sizeof(float));
                partial_select_desc(tmp,nk,keep);
                float thr=tmp[0]; for(int t=1;t<keep;t++) if(tmp[t]<thr) thr=tmp[t];
                int *dst=m->dsa_sel+(int64_t)s*dtopk, nd=0;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]>thr) dst[nd++]=t;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]==thr) dst[nd++]=t;
                m->dsa_nsel[s]=nd;
                free(qi); free(w32); free(isc); free(tmp);
            }
        }
        if(m->dsa_nsel){ dsel=m->dsa_sel; dnsel=m->dsa_nsel; }
    }
    /* WEIGHT ABSORPTION (DeepSeek): per S piccoli (decode/verifica MTP) NON si ricostruisce
     * k/v per ogni token del contesto. Per linearita':
     *   q·k_nope_t = (W_K^hT q_nope)·L_t      ctx^h = W_V^h (Σ_t a_t L_t)
     * costo per step ~O(T·kv_lora) invece di O(T·H·(nope+vh)) del matmul kvb_all. */
    if(pipe_done){
        free(ctx); free(Q); free(QR); free(comp);
        m->t_attn += now_s()-ta0;
        return;
    }
    int cuda_absorb=0;
#ifdef COLI_CUDA
    cuda_absorb=layer<c->n_layers&&!kvs&&g_cuda_enabled&&getenv("COLI_CUDA_ATTN")&&
                atoi(getenv("COLI_CUDA_ATTN"))&&c->kv_lora<=512;
#endif
    int absorb = kvs || g_absorb==1 || (g_absorb<0 && S<=4) || cuda_absorb;
    if(absorb && c->kv_lora<=512){
        m->t_aproj+=now_s()-ta0; double tac=now_s();
        int kvl=c->kv_lora, r0v=c->qk_nope;      /* offset righe V dentro il blocco di testa */
        /* Punteggi per-thread sul HEAP. Il cap DEVE essere il massimo nt effettivo del
         * batch, non Tk+1: Tk=pos_base+S vale solo quando pos==pos_base+s. Il percorso
         * batched (step_decode_batch da run_serve_mux) passa positions[] e kv_start
         * per-slot, quindi nt=pos+1-st0 puo' superare Tk+1 -> heap-buffer-overflow su
         * sc[jj]. Si conta esattamente come il loop sotto. */
        int64_t sc_cap = 1;
        for(int s=0;s<S;s++){
            KVState *ks=kvs?kvs[s]:m->kv;
            int pos=positions?positions[s]:pos_base+s;
            int st0=ks->kv_start[layer];
            int ns=(dnsel && dnsel[s]>0)?dnsel[s]:0;      /* DSA: top-k, altrimenti range pieno */
            int64_t nt = ns ? (int64_t)ns : (int64_t)pos+1-st0;
            if(nt>sc_cap) sc_cap=nt;
        }
        float *sc_all = falloc((int64_t)omp_get_max_threads()*sc_cap);
        int cuda_core=0,cuda_projected=0;
#ifdef COLI_CUDA
        if(kvs&&g_cuda_enabled&&getenv("COLI_CUDA_ATTN")&&atoi(getenv("COLI_CUDA_ATTN"))&&
           !dnsel&&l->kv_b.cuda_eligible&&l->o.cuda_eligible&&
           qt_cuda_upload(&l->kv_b)&&qt_cuda_upload(&l->o)){
            const float **rl=malloc((size_t)S*sizeof(*rl)),**rr=malloc((size_t)S*sizeof(*rr));
            const void **rk=malloc((size_t)S*sizeof(*rk));
            int *rn=malloc((size_t)S*sizeof(*rn)); int mt=0;
            if(rk&&rl&&rr&&rn){
                for(int s=0;s<S;s++){
                    int pos=positions[s],st0=kvs[s]->kv_start[layer]; rn[s]=pos+1-st0;
                    rk[s]=kvs[s];
                    rl[s]=coli_kv_row(kvs[s]->Lc[layer],st0,kvl);
                    rr[s]=coli_kv_row(kvs[s]->Rc[layer],st0,c->qk_rope);
                    if(rn[s]>mt)mt=rn[s];
                }
                cuda_core=cuda_projected=coli_cuda_attention_project_ragged(l->kv_b.cuda,l->o.cuda,
                    out,Q,rk,rl,rr,rn,S,H,c->qk_nope,c->qk_rope,vh,kvl,mt,c->attn_scale);
            }
            free(rk);free(rl);free(rr);free(rn);
        } else if(cuda_absorb&&l->n_kv_b_shard>1){
            int n=l->n_kv_b_shard,st0=m->kv_start[layer],nt=pos_base+S-st0,ok=1;
            float *qs=falloc((int64_t)S*H*qh),*cs=falloc((int64_t)S*H*vh);
            for(int d=0;d<n;d++)for(int s=0;s<S;s++)memcpy(
                qs+(int64_t)l->shard_h0[d]*S*qh+(int64_t)s*l->shard_hn[d]*qh,
                Q+((int64_t)s*H+l->shard_h0[d])*qh,(size_t)l->shard_hn[d]*qh*sizeof(float));
            #pragma omp parallel for schedule(static) reduction(&:ok)
            for(int d=0;d<n;d++)ok&=coli_cuda_attention_absorb_batch(l->kv_b_shard[d],
                cs+(int64_t)l->shard_h0[d]*S*vh,qs+(int64_t)l->shard_h0[d]*S*qh,
                coli_kv_row(m->Lc[layer],st0,kvl),coli_kv_row(m->Rc[layer],st0,c->qk_rope),
                S,l->shard_hn[d],c->qk_nope,c->qk_rope,vh,kvl,nt,c->attn_scale);
            if(ok)for(int d=0;d<n;d++)for(int s=0;s<S;s++)memcpy(
                ctx+((int64_t)s*H+l->shard_h0[d])*vh,
                cs+(int64_t)l->shard_h0[d]*S*vh+(int64_t)s*l->shard_hn[d]*vh,
                (size_t)l->shard_hn[d]*vh*sizeof(float));
            free(qs);free(cs);cuda_core=ok;
        } else if(cuda_absorb&&l->kv_b.cuda_eligible&&l->o.cuda_eligible&&
           qt_cuda_upload(&l->kv_b)&&qt_cuda_upload(&l->o)){
            int st0=m->kv_start[layer],nt=pos_base+S-st0;
            cuda_core=cuda_projected=coli_cuda_attention_project_batch(l->kv_b.cuda,l->o.cuda,out,Q,
                coli_kv_row(m->Lc[layer],st0,kvl),coli_kv_row(m->Rc[layer],st0,c->qk_rope),
                S,H,c->qk_nope,c->qk_rope,vh,kvl,nt,c->attn_scale);
        } else if(S<=4&&g_cuda_enabled&&getenv("COLI_CUDA_ATTN")&&atoi(getenv("COLI_CUDA_ATTN"))&&
           l->kv_b.cuda_eligible&&qt_cuda_upload(&l->kv_b)){
            cuda_core=1;
            for(int s=0;s<S&&cuda_core;s++){
                KVState *ks=kvs?kvs[s]:m->kv;int pos=positions?positions[s]:pos_base+s;
                int st0=ks->kv_start[layer],nt=pos+1-st0;
                if(dnsel&&dnsel[s]>0){cuda_core=0;break;}
                cuda_core=0;
                if(g_cuda_pipe&&ks==m->kv&&layer<c->n_layers&&kv_dev_sync(m,l,layer,pos+1))
                    cuda_core=coli_cuda_attention_absorb_kvdev(l->kv_b.cuda,ctx+(int64_t)s*H*vh,
                        Q+(int64_t)s*H*qh,m->kv_dev_L[layer]+(size_t)st0*kvl,
                        m->kv_dev_R[layer]+(size_t)st0*c->qk_rope,H,c->qk_nope,c->qk_rope,
                        vh,kvl,nt,c->attn_scale);
                if(!cuda_core)
                    cuda_core=coli_cuda_attention_absorb(l->kv_b.cuda,ctx+(int64_t)s*H*vh,
                        Q+(int64_t)s*H*qh,coli_kv_row(ks->Lc[layer],st0,kvl),
                        coli_kv_row(ks->Rc[layer],st0,c->qk_rope),H,c->qk_nope,c->qk_rope,
                        vh,kvl,nt,c->attn_scale);
            }
        }
#endif
        if(!cuda_core){
        #pragma omp parallel for collapse(2) schedule(static)
        for(int s=0;s<S;s++) for(int h=0;h<H;h++){
            KVState *ks=kvs?kvs[s]:m->kv;
            int pos=positions?positions[s]:pos_base+s;
            const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;
            const float *qr=qp+c->qk_nope;
            int rbase=h*(c->qk_nope+vh);
            float qabs[512]; memset(qabs,0,kvl*sizeof(float));
            for(int d=0;d<c->qk_nope;d++) qt_addrow(&l->kv_b, rbase+d, qp[d], qabs);
            float *sc = sc_all + (int64_t)omp_get_thread_num()*sc_cap;
            int st0=ks->kv_start[layer];
            int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;    /* DSA: lista top-k o range pieno */
            const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
            int nt = ns ? ns : pos+1-st0;
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=coli_kv_row(ks->Lc[layer],t,kvl);
                const float *kr=coli_kv_row(ks->Rc[layer],t,c->qk_rope);
                float a=0; for(int i=0;i<kvl;i++) a+=qabs[i]*Lt[i];
                for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
                sc[jj]=a*c->attn_scale;
            }
            softmax(sc,nt);
            float clat[512]; memset(clat,0,kvl*sizeof(float));
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=coli_kv_row(ks->Lc[layer],t,kvl);
                float a=sc[jj]; for(int i=0;i<kvl;i++) clat[i]+=a*Lt[i]; }
            qt_matvec_rows(&l->kv_b, rbase+r0v, vh, clat, ctx+((int64_t)s*H+h)*vh);
        }
        }
        m->t_acore+=now_s()-tac; double tao=now_s();
        if(!cuda_projected){matmul_qt(out, ctx, &l->o, S);} m->t_aout+=now_s()-tao;
        free(ctx); free(Q); free(QR); free(comp); free(sc_all);
        m->t_attn += now_s()-ta0;
        return;
    }
    /* 2) ricostruzione di k_nope+value per TUTTI i token 0..Tk-1 (un solo matmul su kv_b) */
    m->t_aproj+=now_s()-ta0; double tk0=now_s();
    int stL=m->kv_start[layer];
    float *kvb_all=falloc((int64_t)Tk*kvb_dim);
    matmul_qt(kvb_all+(int64_t)stL*kvb_dim, m->Lc[layer]+(int64_t)stL*c->kv_lora, &l->kv_b, Tk-stL);
    m->t_kvb += now_s()-tk0;
    /* 3) attenzione causale: score = q_pass·k_nope + q_rot·k_rot
     * (punteggi sul heap, per-thread: vedi il commento nel ramo absorb) */
    int64_t sc_cap = Tk - stL;
    float *sc_all = falloc((int64_t)omp_get_max_threads()*sc_cap);
    double tac=now_s();
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s;
        const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;          /* [qk_nope | qk_rope] */
        const float *qr=qp+c->qk_nope;
        float *sc = sc_all + (int64_t)omp_get_thread_num()*sc_cap;
        int st0=m->kv_start[layer];
        int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;        /* DSA: lista top-k o range pieno */
        const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
        int nt = ns ? ns : pos+1-st0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            const float *kn=kvb_all+(int64_t)t*kvb_dim+(int64_t)h*(c->qk_nope+vh);
            const float *kr=m->Rc[layer]+(int64_t)t*c->qk_rope;
            float a=0; for(int d=0;d<c->qk_nope;d++) a+=qp[d]*kn[d];
            for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
            sc[jj]=a*c->attn_scale;
        }
        softmax(sc,nt);
        float *cx=ctx+((int64_t)s*H+h)*vh; for(int d=0;d<vh;d++) cx[d]=0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            const float *vv=kvb_all+(int64_t)t*kvb_dim+(int64_t)h*(c->qk_nope+vh)+c->qk_nope;
            float a=sc[jj]; for(int d=0;d<vh;d++) cx[d]+=a*vv[d]; }
    }
    m->t_acore+=now_s()-tac; double tao=now_s();
    matmul_qt(out, ctx, &l->o, S); m->t_aout+=now_s()-tao;
    free(ctx); free(Q); free(QR); free(comp); free(kvb_all); free(sc_all);
    m->t_attn += now_s()-ta0;
}

static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    attention_rows(m,l,layer,x,S,pos_base,NULL,NULL,out);
}

/* MoE GLM su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, + shared expert).
 * BATCH-UNION: per S>1 (prefill, verifica MTP) ogni expert UNICO del batch viene caricato
 * una volta sola e moltiplicato per tutte le posizioni che lo usano (pesi letti 1 volta);
 * lo shared expert e' un unico matmul a S righe. Per posizione l'accumulo resta
 * nell'ordine (routed nel loro ordine di union, poi shared). */
/* pin ∪ LRU residency probe (used by CACHE_ROUTE max-rank fill). */
static int expert_is_resident(Model *m, int layer, int eid){
    ESlot *P=m->pin[layer];
    for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid) return 1;
    ESlot *Sl=m->ecache[layer];
    for(int z=0;z<m->ecn[layer];z++) if(Sl[z].eid==eid) return 1;
    return 0;
}

static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out, int with_shared){
    if(g_pilot_real){   /* barriera cross-layer: prendi possesso di QUESTO layer e aspetta
                         * l'eventuale load-pilota in volo sullo stesso layer (dopodiche' il
                         * worker droppa ogni nuovo load <= layer -> ecache[layer] e' stabile
                         * per tutto il resolve/matmul/promozione qui sotto). */
        pthread_mutex_lock(&g_pilot_mx);
        atomic_store_explicit(&g_cur_moe_layer,layer,memory_order_release);
        while(layer>=0 && layer<256 && g_pilot_inflight[layer]>0)
            pthread_cond_wait(&g_pilot_cv,&g_pilot_mx);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    /* DISK-CLASS: does THIS call need the pre-bump recency snapshot? Must agree with
     * dc_needed() in expert_load_impl -- that's what reads what this writes. touched[]
     * makes the write once-per-call: an expert routed by more than one position in a big
     * batch (prefill's S) must snapshot the state from BEFORE this call started, not from
     * an earlier position's bump within the SAME call (which would reintroduce the
     * same-call contamination one position later). Unconditional VLA like the FASE B
     * `seen[E]` below -- E is small, cost is noise. */
    int need_classify = dc_needed();
    unsigned char touched[E]; if(need_classify) memset(touched,0,(size_t)E);
    float *choice=falloc(E);
    int sI=c->moe_inter*c->n_shared;
    /* Rank buffer for CACHE_ROUTE max-rank selection (up to all E experts). */
    int *rank_buf=NULL; float *rank_w=NULL;
    int do_cache_route = g_cache_route && E>0 && K>0;
    int rank_cap = do_cache_route ? (g_route_m>K?g_route_m:K) : 0;
    if(rank_cap>E) rank_cap=E;
    if(do_cache_route){
        rank_buf=malloc((size_t)rank_cap*sizeof(int));
        rank_w=malloc((size_t)rank_cap*sizeof(float));
        if(!rank_buf||!rank_w){ free(rank_buf); free(rank_w); rank_buf=NULL; rank_w=NULL; do_cache_route=0; }
    }
    /* ---- FASE A: routing di tutte le S posizioni ---- */
    double route_t0=g_prof?now_s():0;
    int *idxs=malloc((size_t)S*K*sizeof(int)); float *ws=malloc((size_t)S*K*sizeof(float));
    int *keff=malloc(S*sizeof(int));
    /* router in UN matmul batch: stessa matematica, via le S chiamate S=1 */
    float *logits_all=falloc((int64_t)S*E);
    int pre_routed=0; (void)pre_routed;
#ifdef COLI_METAL
    if(g_pre_idx){                               /* routing gia' calcolata dal layer CB (GPU) */
        memcpy(idxs,g_pre_idx,(size_t)S*K*sizeof(int));
        memcpy(ws,g_pre_w,(size_t)S*K*sizeof(float));
        memcpy(keff,g_pre_keff,(size_t)S*sizeof(int));
        for(int s=0;s<S;s++){
            m->ereq+=keff[s];
            for(int kk=0;kk<keff[s];kk++){
                m->eusage[layer][idxs[(int64_t)s*K+kk]]++;
                ehit_mark(m,layer,idxs[(int64_t)s*K+kk]);
                if(need_classify){                /* DISK-CLASS private recency -- snapshot BEFORE this call's own bump,
                                                   * then tick. This path also bumps the REAL elast/eaccess_clock a few
                                                   * lines below (#417/cfcc742 fixed the once-missing bump on Metal
                                                   * decode) -- DISK-CLASS's own clock stays independent regardless of
                                                   * that fix, see elast_dc in Model. */
                    int e=idxs[(int64_t)s*K+kk];
                    if(!touched[e]){ m->elast_pre[layer][e]=m->elast_dc[layer][e]; touched[e]=1; }
                    m->elast_dc[layer][e]=++m->eaccess_clock_dc;
                }
                if(m->eheat[layer][idxs[(int64_t)s*K+kk]]<UINT32_MAX) m->eheat[layer][idxs[(int64_t)s*K+kk]]++;
                /* #417: la scorciatoia GPU-prerouted deve far avanzare l'orologio di recency
                 * come il percorso router completo (riga ~3055), altrimenti elast/eaccess_clock
                 * si congelano a fine prefill e il tie-breaker LFRU di REPIN gira su punteggi
                 * stantii durante il decode su Metal. */
                m->elast[layer][idxs[(int64_t)s*K+kk]]=++m->eaccess_clock;
            }
            for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
        }
        pre_routed=1;
    }
#endif
    if(!pre_routed) matmul(logits_all, x, l->router, S, D, E);
    if(!pre_routed)
    for(int s=0;s<S;s++){
        float *logit=logits_all+(int64_t)s*E;
        for(int e=0;e<E;e++){ logit[e]=sigmoidf(logit[e]); choice[e]=logit[e]+l->router_bias[e]; }
        int *idx=idxs+(int64_t)s*K; float *w=ws+(int64_t)s*K;
        int Ksel = g_topk>0 ? (g_topk<K?g_topk:K) : K;
        if(do_cache_route){
            /* Full ranking of top rank_cap experts by choice (bias-augmented). */
            int Mwin=rank_cap;
            if(g_route_p>0.f && g_route_p<1.f){
                /* Cumulative-mass variant: grow M until mass covers ROUTE_P. */
                int Mmax=g_route_m>Ksel*4?g_route_m:Ksel*4; if(Mmax>E) Mmax=E; if(Mmax>rank_cap) Mmax=rank_cap;
                for(int kk=0;kk<Mmax;kk++){ int best=-1; float bv=-1e30f;
                    for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(rank_buf[j]==e){tk=1;break;}
                        if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
                    rank_buf[kk]=best; rank_w[kk]=logit[best];
                }
                float tot=1e-20f; for(int kk=0;kk<Mmax;kk++) tot+=rank_w[kk]>0?rank_w[kk]:0;
                float cum=0; Mwin=Ksel;
                for(int kk=0;kk<Mmax;kk++){ cum+=rank_w[kk]>0?rank_w[kk]:0;
                    if(cum>=g_route_p*tot){ Mwin=kk+1; break; } Mwin=kk+1; }
                if(Mwin<Ksel) Mwin=Ksel;
            } else {
                for(int kk=0;kk<Mwin;kk++){ int best=-1; float bv=-1e30f;
                    for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(rank_buf[j]==e){tk=1;break;}
                        if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
                    rank_buf[kk]=best; rank_w[kk]=logit[best];
                }
            }
            int J=g_route_j; if(J<0) J=0; if(J>Ksel) J=Ksel;
            int chosen=0;
            /* Always take true top-J (even if uncached). */
            for(int kk=0;kk<J && chosen<Ksel;kk++){
                idx[chosen]=rank_buf[kk]; w[chosen]=rank_w[kk]; chosen++;
            }
            /* Remaining slots: prefer resident experts within top-Mwin. */
            for(int r=J;r<Mwin && chosen<Ksel;r++){
                int e=rank_buf[r]; int already=0;
                for(int j=0;j<chosen;j++) if(idx[j]==e){already=1;break;}
                if(already) continue;
                if(expert_is_resident(m,layer,e)){
                    idx[chosen]=e; w[chosen]=rank_w[r]; chosen++;
                }
            }
            /* Fill remainder from true ranking order. */
            for(int r=0;r<Mwin && chosen<Ksel;r++){
                int e=rank_buf[r]; int already=0;
                for(int j=0;j<chosen;j++) if(idx[j]==e){already=1;break;}
                if(already) continue;
                idx[chosen]=e; w[chosen]=rank_w[r]; chosen++;
            }
            /* Swap accounting vs true top-Ksel (rank_buf[0..Ksel)). */
            m->route_slots+=(uint64_t)Ksel;
            for(int kk=0;kk<Ksel;kk++){
                int e=idx[kk], in_true=0;
                for(int t=0;t<Ksel;t++) if(rank_buf[t]==e){in_true=1;break;}
                if(!in_true) m->route_swaps++;
            }
            /* Pad if somehow short (shouldn't happen). */
            while(chosen<Ksel){ idx[chosen]=rank_buf[chosen]; w[chosen]=rank_w[chosen]; chosen++; }
            /* ROUTE_ALPHA: down-weight substituted experts' gate mass before renorm. */
            if(g_route_alpha>0.f && g_route_alpha<1.f){
                for(int kk=0;kk<Ksel;kk++){
                    int e=idx[kk], in_true=0;
                    for(int t=0;t<Ksel;t++) if(rank_buf[t]==e){in_true=1;break;}
                    if(!in_true) w[kk]*=g_route_alpha;
                }
            }
            /* ROUTE_AGREE: overlap + KL(true top-K mass || chosen mass). */
            if(g_route_agree || g_cache_route){
                int ov=0;
                for(int kk=0;kk<Ksel;kk++){
                    for(int t=0;t<Ksel;t++) if(idx[kk]==rank_buf[t]){ ov++; break; }
                }
                m->route_agree_hit+=(uint64_t)ov;
                m->route_agree_tot+=(uint64_t)Ksel;
                float tsum=1e-20f, csum=1e-20f;
                for(int t=0;t<Ksel;t++) tsum+=rank_w[t]>0?rank_w[t]:0;
                for(int kk=0;kk<Ksel;kk++) csum+=w[kk]>0?w[kk]:0;
                double kl=0;
                for(int t=0;t<Ksel;t++){
                    double pt=(rank_w[t]>0?rank_w[t]:0)/tsum;
                    if(pt<=0) continue;
                    double pc=1e-12;
                    for(int kk=0;kk<Ksel;kk++) if(idx[kk]==rank_buf[t]){
                        pc=(w[kk]>0?w[kk]:0)/csum; break; }
                    kl+=pt*log(pt/pc);
                }
                m->route_kl_sum+=kl; m->route_kl_n++;
            }
        } else {
            for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
                for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                    if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
                idx[kk]=best; w[kk]=logit[best];
            }
            if(g_route_agree){
                m->route_agree_hit+=(uint64_t)Ksel;
                m->route_agree_tot+=(uint64_t)Ksel;
                m->route_kl_sum+=0; m->route_kl_n++;
            }
        }
        int Ke=Ksel;
        if(g_topp>0 && g_topp<1.f){
            for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
                while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
            float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
            float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=g_topp*tot){ Ke=kk+1; break; } }
        }
        keff[s]=Ke; m->ereq+=Ke;
        for(int kk=0;kk<Ke;kk++){
            m->eusage[layer][idx[kk]]++;
            ehit_mark(m,layer,idx[kk]);
            if(need_classify){                    /* DISK-CLASS private recency -- snapshot BEFORE this call's own bump,
                                                   * then tick (same rate as the real clock below: one per (s,kk)) */
                if(!touched[idx[kk]]){ m->elast_pre[layer][idx[kk]]=m->elast_dc[layer][idx[kk]]; touched[idx[kk]]=1; }
                m->elast_dc[layer][idx[kk]]=++m->eaccess_clock_dc;
            }
            if(m->eheat[layer][idx[kk]]<UINT32_MAX) m->eheat[layer][idx[kk]]++;
            m->elast[layer][idx[kk]]=++m->eaccess_clock;
        }
        if(c->norm_topk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
        for(int kk=0;kk<Ke;kk++) w[kk]*=c->routed_scale;
        if(g_route_fp){                       /* ROUTE_TRACE: one line per (position, layer) */
            fprintf(g_route_fp,"%d %d %d",g_route_call,s,layer);
            for(int kk=0;kk<Ke;kk++) fprintf(g_route_fp," %d:%.4f",idx[kk],w[kk]);
            fputc('\n',g_route_fp);
        }
        for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
    }
    free(rank_buf); free(rank_w);
    if(g_prof)m->t_route+=now_s()-route_t0;
    if(g_route_fp) g_route_call++;
    if(g_couple && cp_pred && S<=8)
        for(int s2=0;s2<S;s2++) couple_prefetch(m,layer,idxs+(int64_t)s2*K,keff[s2]);
    if(g_looka && S==1 && layer<c->n_layers){
        int Ke=keff[0];
        if(m->enr[layer]>0){                       /* [0] vs routing del token precedente */
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<m->enr[layer];z++)
                if(m->eroute[layer][z]==idxs[kk]){ la_hit[0]++; break; }
            la_tot[0]+=Ke;
        }
        for(int kind=0;kind<3;kind++) if(la_val[kind][layer]){   /* score all prediction kinds */
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<K;z++)
                if(la_pred[kind][layer][z]==idxs[kk]){ la_hit[1+kind]++; break; }
            la_tot[1+kind]+=Ke; la_val[kind][layer]=0;
        }
    }
    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];
    /* ---- FASE B: union degli expert del batch ---- */
    int *uniq=malloc((size_t)E*sizeof(int)); int nu=0;
    unsigned char seen[E]; memset(seen,0,(size_t)E);
    for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){
        int e=idxs[(int64_t)s*K+kk];
        if(!seen[e]){ seen[e]=1; uniq[nu++]=e; }
    }
    /* EXPERT_BUDGET: cap distinct experts per layer to reduce disk I/O on cold/low-RAM
     * hosts. MISS-AWARE: always keep cache hits (pin/LRU — they're free, no disk I/O),
     * only drop from misses. From the misses, keep the highest-aggregate-gate-weight
     * ones up to the budget; drop the rest from idxs[] so they're never loaded.
     * (MoE-Spec arXiv 2602.16052: top-32 of 64 capture 93% routing weight.)
     * Complementary to TOPP (per-position) — this trims cross-position.
     * DECODE-ONLY (S<=4, incl. MTP verify): during prefill S=prompt_len the batch
     * union nu is 30-100+ experts and capping to 4-8 drops 80-90% of them, each with
     * non-trivial gate weight -> corrupted prefill hidden state -> wrong KV cache ->
     * repetitive garbage decode. The budget is only safe token-by-token, where the
     * prefill KV cache is already correct. (woolcoxm, #292.) */
    if(g_expert_budget>0 && S<=4 && nu>g_expert_budget){
        /* compute aggregate gate weight per unique expert */
        float *wsum=falloc(nu); for(int j=0;j<nu;j++) wsum[j]=0;
        for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){
            int e=idxs[(int64_t)s*K+kk];
            for(int j=0;j<nu;j++) if(uniq[j]==e){ wsum[j]+=ws[(int64_t)s*K+kk]; break; }
        }
        /* residency pre-scan: which experts are already in pin or ecache (hits)? */
        unsigned char *is_hit=calloc(nu,1); int nhits=0;
        for(int j=0;j<nu;j++){ int eid=uniq[j];
            int found=0;
            ESlot *P=m->pin[layer];
            for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ found=1; break; }
            if(!found){ ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
                for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ found=1; break; } }
            if(found){ is_hit[j]=1; nhits++; }
        }
        /* budget for misses = total budget - hits already kept (min 0) */
        int miss_budget = g_expert_budget - nhits; if(miss_budget<0) miss_budget=0;
        /* mark which unique experts to keep (1) or drop (0): keep all hits, fill rest
         * with top-weight misses up to miss_budget */
        unsigned char *keep=calloc(nu,1); int nkeep=0;
        for(int j=0;j<nu;j++) if(is_hit[j]){ keep[j]=1; nkeep++; }
        for(int rank=0;rank<miss_budget;rank++){
            int best=-1; float bv=-1e30f;
            for(int j=0;j<nu;j++) if(!keep[j] && wsum[j]>bv){ bv=wsum[j]; best=j; }
            if(best<0) break; keep[best]=1; nkeep++;
        }
        /* build a lookup: for each expert id, is it kept? (reuse seen[]) */
        memset(seen,0,(size_t)E);
        for(int j=0;j<nu;j++) if(keep[j]) seen[uniq[j]]=1;
        int dropped=nu-nkeep; g_budget_dropped+=dropped;
        /* remove dropped experts from each position's routing list */
        for(int s=0;s<S;s++){
            int w=0;
            for(int kk=0;kk<keff[s];kk++){
                int e=idxs[(int64_t)s*K+kk];
                if(seen[e]){ idxs[(int64_t)s*K+w]=e; ws[(int64_t)s*K+w]=ws[(int64_t)s*K+kk]; w++; }
            }
            if(w<keff[s]){
                keff[s]=w;
                /* renormalize remaining weights per position */
                if(c->norm_topk && w>0){
                    float sm=0; for(int kk=0;kk<w;kk++) sm+=ws[(int64_t)s*K+kk]; sm+=1e-20f;
                    for(int kk=0;kk<w;kk++) ws[(int64_t)s*K+kk]/=sm;
                    for(int kk=0;kk<w;kk++) ws[(int64_t)s*K+kk]*=c->routed_scale;
                }
            }
        }
        /* compact uniq[] to kept experts only */
        int nu2=0;
        for(int j=0;j<nu;j++) if(keep[j]) uniq[nu2++]=uniq[j];
        nu=nu2;
        free(wsum); free(is_hit); free(keep);
    }
    /* ---- FASE C/D: risolvi (pin/cache/disco) e calcola, a blocchi di 64 unici ---- */
    float *xg=falloc((int64_t)S*D), *gg=falloc((int64_t)S*I), *uu=falloc((int64_t)S*I), *hh=falloc((int64_t)S*D);
    int *rows=malloc(S*sizeof(int)); float *rw=malloc(S*sizeof(float));
#ifdef COLI_CUDA
    /* PIPE Inc.1b: il batch-union del prefill passa dai gruppi GPU — prima di
     * questo, 9343 expert in VRAM restavano INUTILIZZATI durante il prefill
     * (misurato: 81s di expert-matmul tutto su CPU, GPU groups 21ms totali). */
    int group_enabled = S<=64 || (g_cuda_pipe && S<=4096);
    float *group_x=group_enabled?falloc((int64_t)S*K*D):NULL;
    float *group_y=group_enabled?falloc((int64_t)S*K*D):NULL;
    int *group_row=group_enabled?malloc((size_t)64*S*sizeof(int)):NULL;
    float *group_weight=group_enabled?malloc((size_t)64*S*sizeof(float)):NULL;
#endif
    int shared_on_gpu=0; (void)shared_on_gpu;   /* set by the Metal path when Phase E was fused */
    for(int base=0;base<nu;base+=64){
        int nb = nu-base<64 ? nu-base : 64;
        ESlot *use[64]; int missk[64]; int qof[64]; int nmiss=0;
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; use[j]=NULL; qof[j]=-1;
            ESlot *P=m->pin[layer];
            for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ m->hits++; m->hit_pin++; use[j]=&P[z]; break; }
            if(!use[j]){ ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
                for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ m->hits++; m->hit_ecache++; Sl[z].used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED); use[j]=&Sl[z]; break; } }
            if(!use[j]){ qof[j]=nmiss; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++;
                if(g_disk_split){ if(m->ld_ctx==1) m->miss_draft++; else if(m->ld_ctx==2) m->miss_absorb++; } }
        }
        int metal_done=0;
#ifdef COLI_METAL
        /* GPU/disk OVERLAP: submit the RESIDENT experts (pin/LRU hits, + shared expert on
         * the first block) to the GPU BEFORE loading the missed experts from disk, so the
         * preads run while the GPU computes; the missed subset follows in a second submit.
         * Per-subset CPU fallback on unresolved slab / bad fmt / GPU fault. */
        int is_miss[64]={0}; ColiMetalMoeHandle *mh=NULL;
        int cpu_res=1, cpu_miss=1, mh_shared=0, nbb=0, Rtot=0, mfmt=-1, sh_in=0;
        const void *MG[65],*MU[65],*MD[65]; const float *MGS[65],*MUS[65],*MDS[65];
        int xoffb[65],nrb[65];
        float *mxg=NULL; int *mrows=NULL; float *mrw=NULL;
        /* subset builder: experts with is_miss==WANTMISS (+ shared expert when TRY_SH) */
        #define MB_BUILD(WANTMISS, TRY_SH) do{ \
            nbb=0; Rtot=0; mfmt=-1; sh_in=0; \
            for(int j=0;j<nb;j++){ if(is_miss[j]!=(WANTMISS)) continue; \
                int eid=uniq[base+j]; ESlot *e=use[j]; int cnt=0; \
                for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++) \
                    if(idxs[(int64_t)s*K+kk]==eid){ cnt++; break; } \
                if(!cnt) continue; \
                if(mfmt<0) mfmt=e->g.fmt; \
                MG[nbb]=e->g.fmt==1?(const void*)e->g.q8:(const void*)e->g.q4; \
                MU[nbb]=e->u.fmt==1?(const void*)e->u.q8:(const void*)e->u.q4; \
                MD[nbb]=e->d.fmt==1?(const void*)e->d.q8:(const void*)e->d.q4; \
                MGS[nbb]=e->g.s; MUS[nbb]=e->u.s; MDS[nbb]=e->d.s; \
                xoffb[nbb]=Rtot; nrb[nbb]=cnt; Rtot+=cnt; nbb++; \
            } \
            if(TRY_SH){ int shf = mfmt<0 ? l->sh_gate.fmt : mfmt; \
                if(c->n_shared==1 && sI==I && l->sh_gate.fmt==shf && l->sh_up.fmt==shf && l->sh_down.fmt==shf){ \
                    if(mfmt<0) mfmt=shf; \
                    MG[nbb]=shf==1?(const void*)l->sh_gate.q8:(const void*)l->sh_gate.q4; \
                    MU[nbb]=shf==1?(const void*)l->sh_up.q8  :(const void*)l->sh_up.q4; \
                    MD[nbb]=shf==1?(const void*)l->sh_down.q8:(const void*)l->sh_down.q4; \
                    MGS[nbb]=l->sh_gate.s; MUS[nbb]=l->sh_up.s; MDS[nbb]=l->sh_down.s; \
                    xoffb[nbb]=Rtot; nrb[nbb]=S; Rtot+=S; nbb++; sh_in=1; } } \
            int p=0; \
            for(int j=0;j<nb;j++){ if(is_miss[j]!=(WANTMISS)) continue; int eid=uniq[base+j]; \
                for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++) \
                    if(idxs[(int64_t)s*K+kk]==eid){ \
                        memcpy(mxg+(int64_t)p*D, x+(int64_t)s*D, D*sizeof(float)); \
                        mrows[p]=s; mrw[p]=ws[(int64_t)s*K+kk]; p++; break; } } \
            if(sh_in) for(int s=0;s<S;s++){ \
                memcpy(mxg+(int64_t)p*D, x+(int64_t)s*D, D*sizeof(float)); \
                mrows[p]=s; mrw[p]=1.0f; p++; } \
        }while(0)
        if(g_metal_enabled){
            for(int q=0;q<nmiss;q++) is_miss[missk[q]]=1;
            mxg=falloc((int64_t)(nb+1)*S*D);
            mrows=malloc((size_t)(nb+1)*S*sizeof(int)); mrw=malloc((size_t)(nb+1)*S*sizeof(float));
            MB_BUILD(0, base==0 && !g_pre_sh);
            if(nbb>0){
                double t0=now_s();
                mh=coli_metal_moe_block_begin(nbb,D,I,mfmt,MG,MU,MD,MGS,MUS,MDS,mxg,xoffb,nrb,mrows,mrw);
                m->t_emm += now_s()-t0;
                if(mh){ cpu_res=0; mh_shared=sh_in; }
            } else cpu_res=0;
        }
#endif
        /* Expert loads run HERE, after the resident-experts GPU submit above: under METAL the
         * preads overlap the GPU compute (that submit is async). With METAL off the submit block
         * is a no-op / compiled out, so this sits exactly where dev put it and CPU behaviour is
         * unchanged. */
        if(nmiss){
            if(g_pipe){                            /* PIPE: launch loads async, matmul overlaps them */
                if(!g_pp.started) pipe_init(m);
                double t0=now_s();
                int eids[64]; for(int q=0;q<nmiss;q++) eids[q]=uniq[base+missk[q]];
                pipe_dispatch(m,layer,eids,nmiss);
                m->t_ewait += now_s()-t0;           /* dispatch only; the reads overlap matmul and
                                                     * are timed as service inside expert_load */
            } else { double t0=now_s();             /* ORIGINALE: blocking parallel load */
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++) expert_load(m,layer,uniq[base+missk[q]],&m->ws[q],1,1);   /* demand=1: this IS the miss path */
                m->t_ewait += now_s()-t0; }         /* compute thread blocked for the whole load */
        }
        /* I/O ASINCRONO: readahead (WILLNEED) del blocco SUCCESSIVO mentre calcoliamo
         * questo — il kernel legge in background, le pread dopo trovano cache calda */
        if(base+64<nu){
            int nb2 = nu-(base+64)<64 ? nu-(base+64) : 64;
            for(int j=0;j<nb2;j++){ int eid=uniq[base+64+j]; int found=0;
                ESlot *P=m->pin[layer];
                for(int z=0;z<m->npin[layer] && !found;z++) if(P[z].eid==eid) found=1;
                ESlot *Sl=m->ecache[layer];
                for(int z=0;z<m->ecn[layer] && !found;z++) if(Sl[z].eid==eid) found=1;
                if(!found) expert_prefetch(m,layer,eid);
            }
        }
#ifdef COLI_CUDA
        ESlot *group_e[64]; int group_n[64]; int ngroup=0;
#endif
#ifdef COLI_METAL
        if(g_metal_enabled){
            /* PIPE drain. Two reasons this barrier is mandatory here, and not optional:
             *  1) MB_BUILD(1) hands the missed experts' slabs straight to the GPU — a slot still
             *     being pread by an I/O worker would be matmul-ed half-loaded.
             *  2) PIPE's only drain barrier is the per-expert pipe_wait() in the CPU matmul loop
             *     below, which metal_done SKIPS ENTIRELY. Without this, a still-writing worker
             *     would race the end-of-block LRU swap that recycles ws[].
             * pipe_wait() is an idempotent spin on ready[q], so the per-expert waits below stay
             * correct (and free) when a subset falls back to the CPU. */
            if(g_pipe && nmiss){ double tw=now_s();
                for(int q=0;q<nmiss;q++) pipe_wait(q);
                m->t_ewait += now_s()-tw; }
            MB_BUILD(1, 0);                                   /* missed experts, now loaded */
            if(nbb>0){
                double t0=now_s();
                if(coli_metal_moe_block(nbb,D,I,mfmt,MG,MU,MD,MGS,MUS,MDS,mxg,xoffb,nrb,mrows,mrw,out,S)) cpu_miss=0;
                m->t_emm += now_s()-t0;
            } else cpu_miss=0;
            if(mh){ double t0=now_s();
                if(coli_metal_moe_block_end(mh,out)){ if(mh_shared) shared_on_gpu=1; }
                else cpu_res=1;
                m->t_emm += now_s()-t0; mh=NULL; }
            metal_done = (!cpu_res && !cpu_miss);
            free(mxg); free(mrows); free(mrw);
        }
        #undef MB_BUILD
#endif
        if(!metal_done)
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
            /* Drain this miss's async load BEFORE the nr==0 early-exit below: every
             * dispatched slot must be waited before the end-of-block LRU swap can reuse
             * its ws[] slab, so correctness does not depend on the nr>=1 routing invariant.
             * Stays ABOVE the METAL skip: a subset that fell back to the CPU still needs its
             * slot drained here, and under METAL the block-level drain above already ran (this
             * spin is then a no-op). */
            if(g_pipe && qof[j]>=0){ double tw=now_s(); pipe_wait(qof[j]); m->t_ewait += now_s()-tw; }
#ifdef COLI_METAL
            /* skip the subsets already computed on GPU */
            if(g_metal_enabled && ((is_miss[j] && !cpu_miss) || (!is_miss[j] && !cpu_res))) continue;
#endif
            int nr=0;                                 /* righe (posizioni) che usano questo expert */
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
#ifdef COLI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
            if(group_enabled && g_cuda_enabled && e->g.cuda_eligible && e->u.cuda_eligible && e->d.cuda_eligible &&
               !omp_in_parallel()){
                group_e[ngroup]=e; group_n[ngroup]=nr;
                for(int r=0;r<nr;r++){ group_row[(int64_t)ngroup*S+r]=rows[r]; group_weight[(int64_t)ngroup*S+r]=rw[r]; }
                ngroup++; continue;
            }
#endif
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
#ifdef COLI_CUDA
            if(!group_enabled && g_cuda_enabled && e->g.cuda_eligible && e->u.cuda_eligible &&
               e->d.cuda_eligible && !omp_in_parallel() &&
               coli_cuda_expert_mlp(e->g.cuda,e->u.cuda,e->d.cuda,hh,xg,nr)){
                for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D,wgt=rw[r],*hr=hh+(int64_t)r*D;
                    for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
                double dt=now_s()-t0;m->t_emm+=dt;if(g_prof)m->t_egpu+=dt;continue;
            }
            if(!e->slab) expert_host_ensure(m,layer,e);
#endif
            expert_gate_up(gg,uu,xg,&e->g,&e->u,nr);
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
            matmul_qt(hh, gg, &e->d, nr);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            double dt=now_s()-t0;m->t_emm+=dt;if(g_prof){m->t_ecpu+=dt;
                m->cpu_expert_bytes+=qt_bytes(&e->g)+qt_bytes(&e->u)+qt_bytes(&e->d);
                m->cpu_expert_rows+=(uint64_t)nr;}
        }
#ifdef COLI_CUDA
        ColiCudaTensor *dev_g[COLI_CUDA_MAX_DEVICES][64],*dev_u[COLI_CUDA_MAX_DEVICES][64];
        ColiCudaTensor *dev_d[COLI_CUDA_MAX_DEVICES][64];
        int dev_rows[COLI_CUDA_MAX_DEVICES][64],dev_which[COLI_CUDA_MAX_DEVICES][64];
        int dev_nc[COLI_CUDA_MAX_DEVICES]={0},dev_total[COLI_CUDA_MAX_DEVICES]={0};
        int dev_off[COLI_CUDA_MAX_DEVICES]={0},dev_ok[COLI_CUDA_MAX_DEVICES]={0};
        double dev_time[COLI_CUDA_MAX_DEVICES]={0};
        for(int di=0;di<g_cuda_ndev;di++) for(int q=0;q<ngroup;q++)
            if(group_e[q]->g.cuda_device==g_cuda_devices[di]) dev_total[di]+=group_n[q];
        for(int di=1;di<g_cuda_ndev;di++) dev_off[di]=dev_off[di-1]+dev_total[di-1];
        for(int di=0;di<g_cuda_ndev;di++){
            int cursor=0,device=g_cuda_devices[di];
            for(int q=0;q<ngroup;q++) if(group_e[q]->g.cuda_device==device){
                int nc=dev_nc[di]++; ESlot *e=group_e[q];
                dev_g[di][nc]=e->g.cuda; dev_u[di][nc]=e->u.cuda; dev_d[di][nc]=e->d.cuda;
                dev_rows[di][nc]=group_n[q]; dev_which[di][nc]=q;
                for(int r=0;r<group_n[q];r++) memcpy(group_x+(int64_t)(dev_off[di]+cursor+r)*D,
                    x+(int64_t)group_row[(int64_t)q*S+r]*D,D*sizeof(float));
                cursor+=group_n[q];
            }
        }
        double tg=now_s();
        #pragma omp parallel for if(g_cuda_ndev>1) schedule(static)
        for(int di=0;di<g_cuda_ndev;di++) if(dev_nc[di]){
            double td=g_prof?now_s():0;
            dev_ok[di]=coli_cuda_expert_group(dev_g[di],dev_u[di],dev_d[di],dev_rows[di],dev_nc[di],
                group_y+(int64_t)dev_off[di]*D,group_x+(int64_t)dev_off[di]*D);
            if(g_prof)dev_time[di]=now_s()-td;
        }
        for(int di=0;di<g_cuda_ndev;di++){
            int off=dev_off[di];
            for(int q=0;q<dev_nc[di];q++){
                int gi=dev_which[di][q],nr=group_n[gi]; ESlot *e=group_e[gi];
                if(!dev_ok[di]){
                    for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D,x+(int64_t)group_row[(int64_t)gi*S+r]*D,D*sizeof(float));
                    double tc=g_prof?now_s():0;
                    if(!coli_cuda_expert_mlp(e->g.cuda,e->u.cuda,e->d.cuda,hh,xg,nr)){
                        expert_host_ensure(m,layer,e);
                        expert_gate_up(gg,uu,xg,&e->g,&e->u,nr);
                        for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
                        matmul_qt(hh,gg,&e->d,nr);
                        if(g_prof){m->cpu_expert_bytes+=qt_bytes(&e->g)+qt_bytes(&e->u)+qt_bytes(&e->d);
                            m->cpu_expert_rows+=(uint64_t)nr;}
                    }
                    if(g_prof)m->t_ecpu+=now_s()-tc;
                }
                float *src=dev_ok[di]?group_y+(int64_t)off*D:hh;
                for(int r=0;r<nr;r++){ float *os=out+(int64_t)group_row[(int64_t)gi*S+r]*D,wgt=group_weight[(int64_t)gi*S+r];
                    for(int d=0;d<D;d++) os[d]+=wgt*src[(int64_t)r*D+d]; }
                off+=nr;
            }
        }
        if(g_prof){double mx=0;for(int di=0;di<g_cuda_ndev;di++)if(dev_time[di]>mx)mx=dev_time[di];m->t_egpu+=mx;}
        m->t_emm+=now_s()-tg;
#endif
        /* No drain barrier: the per-expert pipe_wait(qof[j]) above (issued for every
         * dispatched miss slot, before the nr==0 skip) already waited on all ws[] loads
         * for this block, so they are complete before the LRU swap — and the gen-tagged
         * cursor keeps any still-spinning worker off a wrong-generation slot. */
        { ESlot *Sl=m->ecache[layer]; int *nn=&m->ecn[layer];   /* promozione LRU (swap buffer) */
          int promo = nmiss<m->ecap ? nmiss : m->ecap;
          for(int a=0;a<promo;a++){ int q=nmiss-1-a; ESlot *dst;
              if(*nn<m->ecap) dst=&Sl[(*nn)++];
              else { int lru=0; for(int z=1;z<*nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; dst=&Sl[lru]; }
              ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp; dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED); }
        }
    }
    /* ---- FASE E: shared expert (PIPE2: gia' sul device; Metal CB: gia' sommata) ---- */
    if(!with_shared) goto shared_done;
    {
    float *sg=NULL,*su=NULL;int shared_cuda=0;
#ifdef COLI_METAL
    if(g_pre_sh){ for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=g_pre_sh[z]; shared_on_gpu=1; }
    if(shared_on_gpu) shared_cuda=2;             /* gia' sommato in out: salta calcolo e add */
#endif
#ifdef COLI_CUDA
    int shared_min=getenv("COLI_CUDA_SHARED_W4A16_MIN_ROWS")?
        atoi(getenv("COLI_CUDA_SHARED_W4A16_MIN_ROWS")):32;
    if(shared_min<16)shared_min=16;
    if(shared_cuda==0&&S>=shared_min&&!l->shared_w4a16_failed&&!omp_in_parallel()&&g_cuda_enabled&&
       l->sh_gate.fmt==2&&l->sh_up.fmt==2&&l->sh_down.fmt==2&&
       getenv("COLI_CUDA_SHARED_W4A16")&&atoi(getenv("COLI_CUDA_SHARED_W4A16"))&&
       qt_cuda_upload(&l->sh_gate)&&qt_cuda_upload(&l->sh_up)&&qt_cuda_upload(&l->sh_down)){
        shared_cuda=coli_cuda_shared_mlp_w4a16(l->sh_gate.cuda,l->sh_up.cuda,
                                               l->sh_down.cuda,hh,x,S);
        if(!shared_cuda)l->shared_w4a16_failed=1;
    }
#endif
    if(!shared_cuda){
        sg=falloc((int64_t)S*sI);su=falloc((int64_t)S*sI);
        matmul_qt(sg, x, &l->sh_gate, S);
        matmul_qt(su, x, &l->sh_up,   S);
        for(int64_t z=0;z<(int64_t)S*sI;z++) sg[z]=siluf(sg[z])*su[z];
        matmul_qt(hh, sg, &l->sh_down, S);
    }
    if(shared_cuda!=2) for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=hh[z];
    free(sg); free(su);
    }
shared_done:
    free(logits_all); free(choice); free(idxs); free(ws); free(keff); free(uniq);
    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw);
#ifdef COLI_CUDA
    free(group_x);free(group_y);
    free(group_row); free(group_weight);
#endif
}

static void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out){
    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);
    matmul_qt(g, x, &l->gate_proj, S);
    matmul_qt(u, x, &l->up_proj,   S);
    for(int64_t i=0;i<(int64_t)S*I;i++) g[i]=siluf(g[i])*u[i];
    matmul_qt(out, g, &l->down_proj, S);
    free(g); free(u);
}

/* LOOKA: predice il top-K del router del layer `target` dallo stato h (residual stream),
 * usando la STESSA pipeline del routing vero (post_ln -> router -> sigmoid+bias, top-K).
 * kind 0 = stesso layer saltando l'attention
 * kind 1 = layer successivo (PILOT: stale state, 75.8% recall)
 * kind 2 = two-step: approximate L's shared expert output, add to state, THEN predict L+1.
 *   The shared expert is resident (part of dense model), so this adds 3 small matmuls
 *   but no disk I/O. The corrected state includes the dominant part of MoE(L) that the
 *   stale PILOT prediction is missing. */
static void la_predict(Model *m, int target, const float *h, int kind){
    Cfg *c=&m->c; Layer *l=&m->L[target]; int D=c->hidden, E=c->n_experts, K=c->topk;
    float *nrm=falloc(D), *ch=falloc(E);

    if(kind==2){
        /* Two-step: h is L's post-attention state (pre-MoE). We want to predict L+1's
         * routing. The real L+1 router sees h + MoE(L). We approximate MoE(L) by
         * computing ONLY the shared expert (resident, no disk) on the post_ln-normalized
         * state, then add it to h before running L+1's router.
         *
         * target = L+1, so the layer we need the shared expert from is L = target-1. */
        int src_layer = target - 1;
        if(src_layer < 0 || src_layer >= c->n_layers || !m->L[src_layer].sparse
           || c->n_shared <= 0 || c->moe_inter <= 0){
            la_val[2][target] = 0; free(nrm); free(ch); return;
        }
        Layer *sl = &m->L[src_layer];
        int sI = c->moe_inter * c->n_shared;
        float *snrm = falloc(D), *sg = falloc(sI), *su = falloc(sI);
        float *sout = falloc(D), *hc = falloc(D);
        rmsnorm(snrm, h, sl->post_ln, D, c->eps);
        matmul_qt(sg, snrm, &sl->sh_gate, 1);
        matmul_qt(su, snrm, &sl->sh_up,   1);
        for(int i=0;i<sI;i++) sg[i] = siluf(sg[i]) * su[i];
        matmul_qt(sout, sg, &sl->sh_down, 1);
        for(int i=0;i<D;i++) hc[i] = h[i] + sout[i];
        rmsnorm(nrm, hc, l->post_ln, D, c->eps);
        free(snrm); free(sg); free(su); free(sout); free(hc);
        matmul(ch, nrm, l->router, 1, D, E);
        for(int e=0;e<E;e++) ch[e] = sigmoidf(ch[e]) + l->router_bias[e];
        int *pred = la_pred[2][target];
        for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(pred[j]==e){tk=1;break;}
                if(!tk && ch[e]>bv){bv=ch[e];best=e;} }
            pred[kk]=best; }
        la_val[2][target]=1;
        free(nrm); free(ch);
        return;
    }

    /* Baseline kinds 0 and 1: pure router on the given state */
    rmsnorm(nrm,h,l->post_ln,D,c->eps);
    matmul(ch,nrm,l->router,1,D,E);
    for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
    int *pred=la_pred[kind][target];
    for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(pred[j]==e){tk=1;break;}
            if(!tk && ch[e]>bv){bv=ch[e];best=e;} }
        pred[kk]=best; }
    la_val[kind][target]=1;
    free(nrm); free(ch);
}

/* PILOTA: prefetch guidato dal router. Predice il top-K del layer L+1 dallo stato
 * post-attention di L (recall misurato 71.6% su GLM-5.2, vs 41.3% del token precedente)
 * e lancia il WILLNEED degli expert mancanti MENTRE il MoE di L legge i suoi: il disco
 * lavora nei tempi morti del calcolo invece di aspettare il routing vero. Con MTP attiva
 * predice per TUTTE le posizioni del draft: la speculazione pilota anche l'I/O.
 * PILOT_K limita alle prime k predizioni (la testa del ranking e' piu' affidabile
 * della coda: meno banda sprecata sulle predizioni sbagliate).
 *
 * I WILLNEED partono da un THREAD I/O dedicato: con la coda disco satura la submit
 * del fadvise BLOCCA (~0.5ms x 169k chiamate = +92s/48 token, misurato) — inline
 * il pilota costava piu' di quanto rendesse. Ring lock-free 1P/1C; pieno = scarta
 * (un hint perso non e' un errore). */
static struct { int l,e; } pilot_q[4096];
static volatile unsigned pilot_w=0, pilot_r=0;
static Model *pilot_m=NULL;
/* PILOT_REAL: load VERO dell'expert predetto dentro la LRU del layer FUTURO. Vedi
 * l'invariante di sicurezza accanto a g_pilot_real. Il pread (lento) gira FUORI dal lock;
 * il lock protegge solo la scelta/pubblicazione dello slot e l'handshake col main. */
static void pilot_realload(Model *m, int layer, int eid){
    pthread_mutex_lock(&g_pilot_mx);
    if(layer <= atomic_load_explicit(&g_cur_moe_layer,memory_order_acquire)){
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
        pthread_mutex_unlock(&g_pilot_mx); return;      /* il main possiede gia' questo layer */
    }
    ESlot *P=m->pin[layer];                             /* gia' residente (pin o ecache)? skip */
    for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ pthread_mutex_unlock(&g_pilot_mx); return; }
    ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
    for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ pthread_mutex_unlock(&g_pilot_mx); return; }
    int slot,isnew;                                     /* cresci se c'e' posto, altrimenti LRU */
    if(nn<m->ecap){ slot=nn; isnew=1; }
    else { int lru=0; for(int z=1;z<nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; slot=lru; isnew=0; }
    ESlot *dst=&Sl[slot];
    dst->eid=-1;                                        /* nascondi dagli scan-hint mentre carica */
    g_pilot_inflight[layer]++;
    pthread_mutex_unlock(&g_pilot_mx);

    int rc=expert_load(m,layer,eid,dst,0,0);            /* pread VERO — fuori dal lock, sovrapposto al compute; fatal=0: un errore su una speculazione NON deve uccidere il server; demand=0: speculative, never classified */

    pthread_mutex_lock(&g_pilot_mx);
    if(rc==0){
        dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED);
        if(isnew) m->ecn[layer]=slot+1;                 /* pubblica lo slot SOLO ora che eid e' valido */
        atomic_fetch_add_explicit(&g_pilot_loads,1,memory_order_relaxed);
    } else {
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed); /* load fallito: slot resta nascosto (eid=-1), mai pubblicato */
    }
    g_pilot_inflight[layer]--;
    pthread_cond_broadcast(&g_pilot_cv);
    pthread_mutex_unlock(&g_pilot_mx);
    if(rc!=0)                                            /* mai swallow silenzioso: logga (una riga) e prosegui */
        fprintf(stderr,"[PILOT] load speculativo abbandonato: layer %d expert %d (I/O error/short read) — nessun impatto sull'output\n",layer,eid);
}
#ifdef __linux__
typedef struct { int layer,eid,li; ESlot *dst; } PilotUringDone;
static void pilot_uring_batch(Model *m){
    PilotUringDone done[URING_LOAD_MAX]; int nd=0;
    uring_batch_reset(&g_ub_pilot);
    unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
    unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
    while(r!=w && nd<URING_LOAD_MAX){
        int layer=pilot_q[r&4095].l,eid=pilot_q[r&4095].e; r++;
        if(layer<0 || layer>=256){ atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed); continue; }
        pthread_mutex_lock(&g_pilot_mx);
        if(layer<=atomic_load_explicit(&g_cur_moe_layer,memory_order_acquire)){
            atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
            pthread_mutex_unlock(&g_pilot_mx); continue;
        }
        int found=0; ESlot *P=m->pin[layer];
        for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){found=1;break;}
        ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
        for(int z=0;z<nn && !found;z++) if(Sl[z].eid==eid || Sl[z].eid==-(eid+2)) found=1;
        if(found){ pthread_mutex_unlock(&g_pilot_mx); continue; }
        int slot;
        if(nn<m->ecap){ slot=nn; m->ecn[layer]=nn+1; }
        else{
            slot=-1;
            for(int z=0;z<nn;z++){
                if(Sl[z].eid==-1){ slot=z; break; }
                if(Sl[z].eid< -1) continue;          /* URING reservation in flight */
                if(slot<0 || Sl[z].used<Sl[slot].used) slot=z;
            }
        }
        if(slot<0){ atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed); pthread_mutex_unlock(&g_pilot_mx); continue; }
        ESlot *dst=&Sl[slot];
        dst->eid=-(eid+2);                         /* visible reservation; never considered resident/evictable */
        g_pilot_inflight[layer]++;
        pthread_mutex_unlock(&g_pilot_mx);

        int li=uring_load_add(&g_ub_pilot,m,layer,eid,dst,0);
        if(li<0){
            pthread_mutex_lock(&g_pilot_mx); dst->eid=-1; g_pilot_inflight[layer]--;
            pthread_cond_broadcast(&g_pilot_cv); pthread_mutex_unlock(&g_pilot_mx);
            atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed); continue;
        }
        done[nd++]=(PilotUringDone){layer,eid,li,dst};
    }
    __atomic_store_n(&pilot_r,r,__ATOMIC_RELEASE);
    if(!nd) return;
    if(uring_submit_batch(&g_ub_pilot)<0){
        int err=errno;
        for(int i=0;i<g_ub_pilot.nload;i++){
            g_ub_pilot.load[i].error=err; g_ub_pilot.load[i].done=1;
        }
    }
    for(int i=0;i<nd;i++){
        PilotUringDone *d=&done[i];
        int rc=uring_finalize_load(&g_ub_pilot,d->li,0);
        pthread_mutex_lock(&g_pilot_mx);
        if(rc==0){
            d->dst->eid=d->eid;
            d->dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED);
            atomic_fetch_add_explicit(&g_pilot_loads,1,memory_order_relaxed);
        }else{
            d->dst->eid=-1;
            atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
        }
        g_pilot_inflight[d->layer]--;
        pthread_cond_broadcast(&g_pilot_cv);
        pthread_mutex_unlock(&g_pilot_mx);
        if(rc) fprintf(stderr,"[PILOT/URING] load speculativo abbandonato: layer %d expert %d: %s\n",
                       d->layer,d->eid,strerror(g_ub_pilot.load[d->li].error));
    }
}
#endif
static void *pilot_worker(void *arg){
    (void)arg;
    for(;;){
        unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
        unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
        if(r==w){ usleep(200); continue; }
        if(g_pilot_real){
#ifdef __linux__
            if(g_uring){ pilot_uring_batch(pilot_m); continue; }
#endif
            pilot_realload(pilot_m, pilot_q[r&4095].l, pilot_q[r&4095].e);
        }
        else             expert_prefetch(pilot_m, pilot_q[r&4095].l, pilot_q[r&4095].e);
        __atomic_store_n(&pilot_r,r+1,__ATOMIC_RELEASE);
    }
    return NULL;
}
/* parse .coli_pairs (see tools/route_pairs.py): "COLIPAIRS 1 <n>" then
 * "<L> <dL> <e> f:c f:c ..." lines. Needs c->n_experts/n_layers -> called post-init. */
static void couple_load(Model *m, const char *path){
    Cfg *c=&m->c; int E=c->n_experts, NL=c->n_layers;
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[COUPLE] cannot open %s\n",path); return; }
    char magic[16]; int ver=0; long n=0;
    if(fscanf(f,"%15s %d %ld",magic,&ver,&n)!=3 || strcmp(magic,"COLIPAIRS") || ver!=1){
        fprintf(stderr,"[COUPLE] %s: bad header\n",path); fclose(f); return; }
    size_t cells=(size_t)NL*2*E*CP_M;
    cp_pred=malloc(cells*sizeof(int16_t)); cp_cnt=calloc(cells,sizeof(float));
    if(!cp_pred||!cp_cnt){ fprintf(stderr,"[COUPLE] OOM\n"); free(cp_pred); free(cp_cnt); cp_pred=NULL; fclose(f); return; }
    for(size_t i=0;i<cells;i++) cp_pred[i]=-1;
    long used=0;
    char *ln=NULL; size_t lcap=0;
    while(getline(&ln,&lcap,f)>0){          /* line-based: a malformed line cannot eat the next */
        char *p=ln; int L,dL,e; int nc=0;
        if(sscanf(p,"%d %d %d%n",&L,&dL,&e,&nc)!=3) continue;
        p+=nc;
        if(L<0||L>=NL||(dL!=1&&dL!=2)||e<0||e>=E) continue;
        size_t base=((size_t)(L*2+(dL-1))*E+e)*CP_M;
        int j=0;
        while(j<CP_M){
            int fe; float fc;
            if(sscanf(p," %d:%f%n",&fe,&fc,&nc)!=2) break;
            p+=nc;
            if(fe>=0&&fe<E){ cp_pred[base+j]=(int16_t)fe; cp_cnt[base+j]=fc; j++; }
        }
        if(j) used++;
    }
    free(ln);
    fclose(f);
    g_couple=1;
    fprintf(stderr,"[COUPLE] %s: %ld conditioning entries, K=%d depth=%d\n",path,used,g_couple_k,g_couple_d);
}
/* score + enqueue: called from moe() after FASE A with the position's routed set */
static void couple_prefetch(Model *m, int layer, const int *idx, int Ke){
    Cfg *c=&m->c; int E=c->n_experts;
    if(E>512) return;
    if(!pilot_m){ pilot_m=m; pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
    for(int dL=1; dL<=g_couple_d; dL++){
        int lt=layer+dL;
        if(lt>=c->n_layers || !m->L[lt].sparse) continue;
        float sc[512]; memset(sc,0,(size_t)E*sizeof(float));
        for(int kk=0;kk<Ke;kk++){
            size_t base=((size_t)(layer*2+(dL-1))*E+idx[kk])*CP_M;
            for(int j=0;j<CP_M && cp_pred[base+j]>=0;j++) sc[cp_pred[base+j]]+=cp_cnt[base+j];
        }
        for(int kk=0;kk<g_couple_k;kk++){
            int best=-1; float bv=0;
            for(int e=0;e<E;e++) if(sc[e]>bv){bv=sc[e];best=e;}
            if(best<0) break;
            sc[best]=0;
            int found=0;                            /* residency scan, same locking as pilot */
            pthread_mutex_lock(&g_pilot_mx);
            ESlot *P=m->pin[lt];
            for(int z=0;z<m->npin[lt] && !found;z++) if(P[z].eid==best) found=1;
            ESlot *Sl=m->ecache[lt];
            for(int z=0;z<m->ecn[lt] && !found;z++)
                if(Sl[z].eid==best || Sl[z].eid==-(best+2)) found=1;
            pthread_mutex_unlock(&g_pilot_mx);
            if(!found){
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    pilot_q[w&4095].l=lt; pilot_q[w&4095].e=best;
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                    g_cp_enq++;
                }
            }
        }
    }
}
static void pilot_prefetch(Model *m, int lnext, const float *x, int S){
    Cfg *c=&m->c; Layer *l=&m->L[lnext]; int D=c->hidden, E=c->n_experts;
    int K = g_pilot_k<c->topk ? g_pilot_k : c->topk;
    if(!pilot_m){ pilot_m=m; pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
    float *nrm=falloc(D), *ch=falloc(E);
    /* Two-step workspace (allocated once, reused across positions) */
    float *snrm=NULL, *sg=NULL, *su=NULL, *sout=NULL, *hc=NULL;
    int src_layer = lnext - 1;
    int sI = 0;
    int can_two = g_pilot_two && src_layer>=0 && src_layer<c->n_layers
                  && m->L[src_layer].sparse && c->n_shared>0 && c->moe_inter>0;
    if(can_two){
        sI = c->moe_inter * c->n_shared;
        snrm=falloc(D); sg=falloc(sI); su=falloc(sI); sout=falloc(D); hc=falloc(D);
    }
    for(int s=0;s<S;s++){
        const float *xs = x+(int64_t)s*D;
        if(can_two){
            /* Two-step: approximate MoE(src_layer) via shared expert only (resident, no disk),
             * then run lnext's router on the corrected state. */
            Layer *sl = &m->L[src_layer];
            rmsnorm(snrm, xs, sl->post_ln, D, c->eps);
            matmul_qt(sg, snrm, &sl->sh_gate, 1);
            matmul_qt(su, snrm, &sl->sh_up,   1);
            for(int i=0;i<sI;i++) sg[i] = siluf(sg[i]) * su[i];
            matmul_qt(sout, sg, &sl->sh_down, 1);
            for(int i=0;i<D;i++) hc[i] = xs[i] + sout[i];
            rmsnorm(nrm, hc, l->post_ln, D, c->eps);
        } else {
            rmsnorm(nrm, xs, l->post_ln, D, c->eps);
        }
        matmul(ch, nrm, l->router, 1, D, E);
        for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
        for(int kk=0;kk<K;kk++){
            int best=0; for(int e=1;e<E;e++) if(ch[e]>ch[best]) best=e;
            ch[best]=-2e30f;
            /* Residency scan of the FUTURE layer lnext under g_pilot_mx: with
             * PILOT_REAL=1 the pilot worker mutates ecache[lnext]/ecn[lnext]
             * concurrently, so read them under the same lock (Option A). Decide
             * under the lock, then enqueue AFTER unlocking — the pilot_q ring is
             * lock-free (pilot_w/pilot_r atomics, not g_pilot_mx) so there is no
             * re-entrant double-lock, and the worker re-checks residency under the
             * lock anyway, making a racing redundant enqueue harmless. */
            int found=0;
            pthread_mutex_lock(&g_pilot_mx);
            ESlot *P=m->pin[lnext];
            for(int z=0;z<m->npin[lnext] && !found;z++) if(P[z].eid==best) found=1;
            ESlot *Sl=m->ecache[lnext];
            for(int z=0;z<m->ecn[lnext] && !found;z++)
                if(Sl[z].eid==best || Sl[z].eid==-(best+2)) found=1;
            pthread_mutex_unlock(&g_pilot_mx);
            if(!found){
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    pilot_q[w&4095].l=lnext; pilot_q[w&4095].e=best;
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                }
            }
        }
    }
    free(nrm); free(ch);
    if(can_two){ free(snrm); free(sg); free(su); free(sout); free(hc); }
}

/* forward di UN layer (usato dai 78 principali e dal layer MTP) */
#ifdef COLI_CUDA
/* Inc.2a — intero layer SPARSO residente sul device del layer. x_dev entra e resta;
 * lasciano il device solo: nrm post-attention (router + expert CPU + gather dei
 * gruppi), i nuovi record KV, e la nrm pre-attention sui layer con indexer DSA.
 * Ritorna 0 su errore: il chiamante ripristina lo snapshot e rifa' il layer su CPU. */
static int pipe_layer_sparse(Model *m, Layer *l, int li, float *x_dev, int S, int pos_base,
                             float *nrm_host, float *out_host){
    Cfg *c=&m->c; int D=c->hidden, dev=l->kv_b.cuda_device;
    int sI=c->moe_inter*c->n_shared;
    size_t xb=(size_t)S*D*4;
    if(!l->sh_gate.cuda_eligible||!l->sh_up.cuda_eligible||!l->sh_down.cuda_eligible||
       !qt_cuda_upload(&l->sh_gate)||!qt_cuda_upload(&l->sh_up)||!qt_cuda_upload(&l->sh_down)||
       l->sh_gate.cuda_device!=dev||l->sh_up.cuda_device!=dev||l->sh_down.cuda_device!=dev) return 0;
    float *w_in =coli_cuda_pipe_scratch(dev,8,(size_t)D*4);
    float *w_post=coli_cuda_pipe_scratch(dev,9,(size_t)D*4);
    float *nrm_d=coli_cuda_pipe_scratch(dev,10,xb);
    float *y_d  =coli_cuda_pipe_scratch(dev,11,xb);
    float *sg_d =coli_cuda_pipe_scratch(dev,12,(size_t)S*sI*4);
    float *su_d =coli_cuda_pipe_scratch(dev,13,(size_t)S*sI*4);
    float *snap =coli_cuda_pipe_scratch(dev,14,xb);
    if(!w_in||!w_post||!nrm_d||!y_d||!sg_d||!su_d||!snap) return 0;
    if(!coli_cuda_pipe_peer_copy(dev,snap,dev,x_dev,xb)) return 0;   /* snapshot per il fallback */
    if(!coli_cuda_pipe_upload(dev,w_in,l->in_ln,(size_t)D*4)||
       !coli_cuda_pipe_upload(dev,w_post,l->post_ln,(size_t)D*4)) return 0;
    double ta=now_s();
    if(!coli_cuda_pipe_rmsnorm(dev,nrm_d,x_dev,w_in,S,D,c->eps)) return 0;
    /* DSA: i layer con indexer FULL cachano k_idx dalla nrm pre-attention (CPU, piccolo) */
    if(m->has_dsa && li<c->n_layers && m->kv_start[li]==0 && c->idx_type[li]){
        if(!coli_cuda_pipe_download(dev,nrm_d,nrm_host,xb)) return 0;
        int nh=c->index_nh, hd=c->index_hd; (void)nh;
        for(int s=0;s<S;s++){
            int pos=pos_base+s;
            float *kd=coli_kv_row(m->kv->Ic[li],pos,hd);
            matmul_qt(kd, nrm_host+(int64_t)s*D, &m->ix_wk[li], 1);
            layernorm(kd, m->ix_knw[li], m->ix_knb[li], hd, 1e-6f);
            rope_interleave(kd, pos, c);
        }
    }
    if(!attn_pipe_prefill(m,l,li,nrm_d,1,S,pos_base,NULL,y_d)) return 0;
    if(!coli_cuda_pipe_add(dev,x_dev,y_d,(size_t)S*D)) return 0;         /* prima mutazione */
    if(!coli_cuda_pipe_rmsnorm(dev,nrm_d,x_dev,w_post,S,D,c->eps)) return 0;
    if(!coli_cuda_pipe_download(dev,nrm_d,nrm_host,xb)) return 0;
    m->t_attn+=now_s()-ta;
    /* OVERLAP: issue the shared expert on the GPU BEFORE moe() runs on the CPU.
     * The shared expert reads nrm_d (valid after the download above) and writes its
     * residual into x_dev (async). While the GPU computes this, the CPU enters moe()
     * for routing + expert disk loads + matmul — ~50ms of work that previously left
     * the GPU idle. The shared expert (~0.5ms) finishes early in that window.
     *
     * After moe(), the routed-expert result is uploaded (sync pipe_upload) and added
     * to x_dev (async). Both residual adds (shared + routed) are ordered on the same
     * stream — the next layer's pipe_rmsnorm reads x_dev after both complete.
     *
     * No pipe_sync at the end: the next layer's pipe_download (sync cudaMemcpy)
     * provides the implicit sync point. The fallback path (caller downloads x_dev)
     * also uses pipe_download which syncs. This lets GPU work chain across layers
     * without a per-layer stall.
     *
     * Profiling: moe() self-times its own t_emm (routed expert matmul). We time only
     * the GPU work that moe() does NOT cover: the shared-expert dispatch and the
     * routed-expert upload+add. Previously a single outer span wrapped everything
     * including moe(), double-counting the routed-expert time and driving the
     * profile's "other" bucket negative (#292). */
    double te=now_s();
    if(!coli_cuda_pipe_gemm(l->sh_gate.cuda,sg_d,nrm_d,S)) return 0;
    if(!coli_cuda_pipe_gemm(l->sh_up.cuda,su_d,nrm_d,S)) return 0;
    if(!coli_cuda_pipe_silu_mul(dev,sg_d,su_d,(size_t)S*sI)) return 0;
    if(!coli_cuda_pipe_gemm(l->sh_down.cuda,y_d,sg_d,S)) return 0;
    if(!coli_cuda_pipe_add(dev,x_dev,y_d,(size_t)S*D)) return 0;  /* shared residual (async) */
    m->t_emm += now_s()-te;                                       /* shared-expert GPU dispatch only */
    /* expert routed su CPU/gruppi GPU come oggi (shared saltata: la fa il device) */
    moe(m,l,li,nrm_host,S,out_host,0);                            /* self-times its own t_emm */
    te=now_s();
    if(!coli_cuda_pipe_upload(dev,y_d,out_host,xb)) return 0;     /* sync: waits for moe */
    if(!coli_cuda_pipe_add(dev,x_dev,y_d,(size_t)S*D)) return 0;  /* routed residual (async) */
    m->t_emm += now_s()-te;                                       /* routed-expert upload + add only */
    return 1;
}
#endif

static void layer_forward_rows(Model *m, Layer *l, int li, float *x, int S, int pos_base,
                               KVState *const *kvs, const int *positions, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_spec && g_prefetch && l->sparse && m->enr[li]>0)
        for(int z=0;z<m->enr[li];z++) expert_prefetch(m,li,m->eroute[li][z]);
    if(g_looka && S==1 && li<c->n_layers && l->sparse) la_predict(m,li,x,0);
#ifdef COLI_METAL
    /* FULL-LAYER CB: in_ln + attention + residuo + post_ln + shared expert + router/top-K
     * in un solo submit GPU; la CPU legge il routing e fa solo resolve/disk/expert-CB.
     * Fallback: qualsiasi condizione mancante -> percorso CPU intero qui sotto.
     * !kvs: ragged mux rows (per-row KV/position) are not expressible in this kernel's
     * single Lc/Rc + pos_base contract — see the matching guard in attention_rows. */
    if(g_metal_enabled && !kvs && S<=4 && li<c->n_layers && l->sparse
       && (g_absorb==1||(g_absorb<0&&S<=4)) && m->kv_start[li]==0
       && D==6144 && c->n_heads==64 && c->q_lora==2048 && c->kv_lora==512
       && c->qk_nope==192 && c->qk_rope==64 && c->v_head==256 && l->kv_b.fmt==2
       && c->n_experts==256 && c->topk==8 && c->n_shared==1 && c->moe_inter==2048){
        int sel_active = m->has_dsa && c->idx_type[li] && (pos_base+S) > c->index_topk;
        if(!sel_active){
            static float *linrm,*lnrm,*lsh,*lw; static int *lidx,*lkeff;
            if(!linrm){ linrm=falloc(4*(int64_t)D); lnrm=falloc(4*(int64_t)D); lsh=falloc(4*(int64_t)D);
                        lidx=malloc(4*8*sizeof(int)); lw=malloc(4*8*sizeof(float)); lkeff=malloc(4*sizeof(int)); }
            int Ksel = g_topk>0 ? (g_topk<8?g_topk:8) : 8;
            float tp = (g_topp>0 && g_topp<1.f) ? g_topp : 0.f;
            double ta0=now_s();
            #define WP_(q) ((q).fmt==1?(const void*)(q).q8:(const void*)(q).q4)
            int ok = coli_metal_layer_decode(x, l->in_ln, l->post_ln,
                WP_(l->q_a), l->q_a.s, l->q_a.fmt, l->q_a_ln,
                WP_(l->q_b), l->q_b.s, l->q_b.fmt,
                WP_(l->kv_a), l->kv_a.s, l->kv_a.fmt, l->kv_a_ln,
                WP_(l->kv_b), l->kv_b.s, l->kv_b.fmt,
                WP_(l->o), l->o.s, l->o.fmt,
                WP_(l->sh_gate), l->sh_gate.s, l->sh_gate.fmt,
                WP_(l->sh_up),   l->sh_up.s,   l->sh_up.fmt,
                WP_(l->sh_down), l->sh_down.s, l->sh_down.fmt,
                l->router, l->router_bias,
                c->n_experts, c->topk, Ksel, tp, c->norm_topk, c->routed_scale,
                m->Lc[li], m->Rc[li], S, pos_base, m->kv_start[li],
                c->eps, c->theta, c->attn_scale,
                linrm, lnrm, lsh, lidx, lw, lkeff);
            #undef WP_
            if(ok){
                m->t_attn += now_s()-ta0;
                if(m->has_dsa && c->idx_type[li]){            /* index key per selezioni future */
                    for(int s=0;s<S;s++){ int pos=pos_base+s;
                        float *kd=m->Ic[li]+(int64_t)pos*c->index_hd;
                        matmul_qt(kd, linrm+(int64_t)s*D, &m->ix_wk[li], 1);
                        layernorm(kd, m->ix_knw[li], m->ix_knb[li], c->index_hd, 1e-6f);
                        rope_interleave(kd, pos, c);
                    }
                }
                if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
                if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse){
                    la_predict(m,li+1,x,1);
                    la_predict(m,li+1,x,2);
                }
                g_pre_idx=lidx; g_pre_w=lw; g_pre_keff=lkeff; g_pre_sh=lsh;
                moe(m,l,li,lnrm,S,tmp,1);
                g_pre_idx=NULL; g_pre_w=NULL; g_pre_keff=NULL; g_pre_sh=NULL;
                for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
                return;
            }
        }
    }
#endif
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
    attention_rows(m,l,li,nrm,S,pos_base,kvs,positions,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
    if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse){
        la_predict(m,li+1,x,1);  /* baseline: stale-state PILOT */
        la_predict(m,li+1,x,2);  /* two-step: shared-expert-corrected prediction */
    }
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->post_ln, D, c->eps);
    if(l->sparse) moe(m,l,li,nrm,S,tmp,1); else dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
}
static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    layer_forward_rows(m,l,li,x,S,pos_base,NULL,NULL,nrm,tmp);
}
static void layers_forward_rows(Model *m, float *x, int S, int pos_base,
                                KVState *const *kvs, const int *positions){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_pilot_real){   /* nuovo forward: il possesso-layer riparte da -1 (i layer si rifanno da 0) */
        pthread_mutex_lock(&g_pilot_mx);
        atomic_store_explicit(&g_cur_moe_layer,-1,memory_order_release);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
#ifdef COLI_CUDA
    /* PIPE2 (Inc.2a): il residuo resta sul device del layer, saltando tra le schede
     * ai confini di layer. x host diventa STALE finche' la residenza e' attiva.
     *
     * S threshold is device-count-dependent (#273): on a single GPU the resident
     * stream wins at S=1 (evicts the CPU round-trips that dominate small-batch
     * decode — +49% on a 5070 Ti). With layers sharded across multiple GPUs each
     * resident forward crosses P2P per layer group, and at one token per forward
     * those hops don't amortize — A/B on 6x5090 showed S=1 is a wash there. So:
     * single-GPU engages at S=1, multi-GPU keeps the original S>=8 prefill gate.
     * COLI_CUDA_PIPE_S_MIN overrides for anyone who wants to measure. */
    float *x_dev=NULL; int x_dev_on=-1;
    size_t xb=(size_t)S*(size_t)D*4;
    int pipe_s_min = getenv("COLI_CUDA_PIPE_S_MIN") ? atoi(getenv("COLI_CUDA_PIPE_S_MIN"))
                                                     : (g_cuda_ndev<=1 ? 1 : 8);
    int pipe2 = g_cuda_pipe>=2 && !kvs && S>=pipe_s_min && g_cuda_enabled && c->kv_lora<=512 &&
                !(m->has_dsa && pos_base+S>c->index_topk);
#endif
    for(int i=0;i<c->n_layers;i++){
        /* progresso su stderr per i batch grossi (prefill): il primo byte di risposta
         * puo' arrivare dopo MINUTI di streaming — al buio sembra un blocco. */
        if(S>=8 && (i%4==0 || i==c->n_layers-1))
            fprintf(stderr,"[prefill] layer %d/%d · %d token\n", i+1, c->n_layers, S);
#ifdef COLI_CUDA
        Layer *l=&m->L[i];
        if(pipe2 && l->sparse && i<c->n_layers &&
           l->q_a.cuda_eligible&&l->q_b.cuda_eligible&&l->kv_a.cuda_eligible&&
           l->kv_b.cuda_eligible&&l->o.cuda_eligible&&
           qt_cuda_upload(&l->q_a)&&qt_cuda_upload(&l->q_b)&&qt_cuda_upload(&l->kv_a)&&
           qt_cuda_upload(&l->kv_b)&&qt_cuda_upload(&l->o)&&
           l->q_a.cuda_device==l->kv_b.cuda_device&&l->q_b.cuda_device==l->kv_b.cuda_device&&
           l->kv_a.cuda_device==l->kv_b.cuda_device&&l->o.cuda_device==l->kv_b.cuda_device){
            int dev=l->kv_b.cuda_device, ok=1;
            float *dst=coli_cuda_pipe_scratch(dev,15,xb);
            if(dst){
                if(x_dev_on<0) ok=coli_cuda_pipe_upload(dev,dst,x,xb);
                else if(x_dev_on!=dev){
                    double tp=g_prof?now_s():0;
                    ok=coli_cuda_pipe_peer_copy(dev,dst,x_dev_on,x_dev,xb);
                    if(g_prof){m->t_p2p+=now_s()-tp;m->n_p2p++;}
                }
                else dst=x_dev;
                if(ok){
                    x_dev=dst; x_dev_on=dev;
                    if(pipe_layer_sparse(m,l,i,x_dev,S,pos_base,nrm,tmp)) continue;
                    /* fallback: snapshot -> host, layer rifatto sul percorso CPU */
                    coli_cuda_pipe_peer_copy(dev,x_dev,dev,coli_cuda_pipe_scratch(dev,14,xb),xb);
                    coli_cuda_pipe_download(dev,x_dev,x,xb);
                    x_dev_on=-1;
                }else x_dev_on=-1;
            }
        } else if(x_dev_on>=0){                 /* layer fuori pipe: il residuo torna a casa */
            coli_cuda_pipe_download(x_dev_on,x_dev,x,xb);
            x_dev_on=-1;
        }
#endif
        layer_forward_rows(m,&m->L[i],i,x,S,pos_base,kvs,positions,nrm,tmp);
    }
#ifdef COLI_CUDA
    if(x_dev_on>=0) coli_cuda_pipe_download(x_dev_on,x_dev,x,xb);
#endif
    free(nrm); free(tmp);
}
static void layers_forward(Model *m, float *x, int S, int pos_base){
    layers_forward_rows(m,x,S,pos_base,NULL,NULL);
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    KVState *k=m->kv;
#ifdef COLI_CUDA
    if(m->kv_dev_L) for(int i=0;i<c->n_layers+1;i++){    /* dimensioni cambiate: ombra da rifare */
        if(m->kv_dev_L[i]){ coli_cuda_pipe_free(m->L[i<c->n_layers?i:0].kv_b.cuda_device,m->kv_dev_L[i]); m->kv_dev_L[i]=NULL; }
        if(m->kv_dev_R[i]){ coli_cuda_pipe_free(m->L[i<c->n_layers?i:0].kv_b.cuda_device,m->kv_dev_R[i]); m->kv_dev_R[i]=NULL; }
        m->kv_dev_valid[i]=0;
    }
#endif
    if(k->Lc){ for(int i=0;i<c->n_layers+1;i++){
#ifdef COLI_METAL
        if(g_metal_enabled){ coli_metal_unregister(k->Lc[i]); coli_metal_unregister(k->Rc[i]); }
#endif
        free(k->Lc[i]); free(k->Rc[i]); } free(k->Lc); free(k->Rc); }
    if(k->Ic){ for(int i=0;i<c->n_layers;i++) free(k->Ic[i]); free(k->Ic); k->Ic=NULL; }
    if(m->has_dsa){
        k->Ic=calloc(c->n_layers,sizeof(float*));
        for(int i=0;i<c->n_layers;i++) if(c->idx_type[i]) k->Ic[i]=falloc((int64_t)max_t*c->index_hd);
    }
    k->max_t=max_t;
    int NR=c->n_layers+1;                        /* riga extra: KV del layer MTP */
    k->Lc=calloc(NR,sizeof(float*)); k->Rc=calloc(NR,sizeof(float*));
    for(int i=0;i<NR;i++){ k->Lc[i]=falloc((int64_t)max_t*c->kv_lora);
        k->Rc[i]=falloc((int64_t)max_t*c->qk_rope);
#ifdef COLI_METAL
        /* page-align + register Lc/Rc for zero-copy GPU attention. falloc isn't 16K-aligned,
         * so re-allocate aligned and register the exact byte length. */
        if(g_metal_enabled){
            size_t lb=(((size_t)max_t*c->kv_lora*sizeof(float))+16383)&~(size_t)16383;
            size_t rb=(((size_t)max_t*c->qk_rope*sizeof(float))+16383)&~(size_t)16383;
            free(k->Lc[i]); free(k->Rc[i]); void *lp,*rp;
            if(posix_memalign(&lp,16384,lb)||posix_memalign(&rp,16384,rb)){fprintf(stderr,"OOM kv\n");exit(1);}
            k->Lc[i]=lp; k->Rc[i]=rp;
            coli_metal_register(k->Lc[i],lb); coli_metal_register(k->Rc[i],rb);
        }
#endif
    }
    m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic; m->max_t=k->max_t; m->kv_start=k->kv_start;
}

static void kv_bind(Model *m, KVState *k){
    if(m->kv!=k && m->kv_dev_valid)                 /* ombra legata al KVState corrente */
        for(int i=0;i<m->c.n_layers+1;i++) m->kv_dev_valid[i]=0;
    m->kv=k; m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic;
    m->max_t=k->max_t; m->kv_start=k->kv_start;
}

static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base);
static float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    if(m->has_mtp && S>=2 && g_draft>0) mtp_absorb(m, ids+1, x, S-1, pos_base);
    float *last=falloc(D); rmsnorm(last, x+(int64_t)(S-1)*D, m->final_norm, D, c->eps);
    double th0=now_s();
    float *logit=falloc(c->vocab); matmul_qt(logit,last,&m->lm_head,1);
    m->t_head += now_s()-th0;
    free(x); free(last); return logit;
}

/* come step(), ma ritorna i logits di TUTTE le S posizioni [S,vocab] (per la verifica spec) */
static float *step_all(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->h_all) memcpy(m->h_all, x, (int64_t)S*D*sizeof(float));   /* hidden di TUTTE le pos (S<=512) */
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    float *lo=falloc((int64_t)S*c->vocab), *row=falloc(D);
    for(int s=0;s<S;s++){ rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo+(int64_t)s*c->vocab, row, &m->lm_head, 1); }
    free(x); free(row); return lo;
}

/* One decode token from each independent sequence, evaluated as a single MoE
 * batch.  Prefill and speculative batches retain their contiguous-KV path. */
static float *step_decode_batch(Model *m, const DecodeRow *rows, int S){
    Cfg *c=&m->c; int D=c->hidden;
    /* Ragged KV currently uses MLA absorption; the stack kernel is sized to 512. */
    if(!rows || S<1 || S>512 || c->kv_lora>512) return NULL;
    KVState *kvs[512]; int positions[512];
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++){
        if(!rows[s].kv || !rows[s].kv->Lc || !rows[s].kv->Rc || !rows[s].kv->kv_start ||
           rows[s].token<0 || rows[s].token>=c->vocab ||
           rows[s].pos<0 || rows[s].pos>=rows[s].kv->max_t){
            free(x); return NULL;
        }
        for(int l=0;l<c->n_layers;l++){
            if(!rows[s].kv->Lc[l] || !rows[s].kv->Rc[l] ||
               rows[s].kv->kv_start[l]<0 || rows[s].kv->kv_start[l]>rows[s].pos ||
               (m->has_dsa && c->idx_type[l] &&
                (!rows[s].kv->Ic || !rows[s].kv->Ic[l]))){ free(x); return NULL; }
        }
        for(int p=0;p<s;p++) if(rows[p].kv==rows[s].kv){ free(x); return NULL; }
        kvs[s]=rows[s].kv; positions[s]=rows[s].pos;
        embed_row(m,rows[s].token,x+(int64_t)s*D);
    }
    layers_forward_rows(m,x,S,0,kvs,positions);
    float *norm=falloc((int64_t)S*D);
    for(int s=0;s<S;s++)
        rmsnorm(norm+(int64_t)s*D,x+(int64_t)s*D,m->final_norm,D,c->eps);
    double th0=now_s();
    float *logit=falloc((int64_t)S*c->vocab);
    matmul_qt(logit,norm,&m->lm_head,S);
    m->t_head+=now_s()-th0;
    free(x); free(norm);
    return logit;
}

/* METODO E — prompt-lookup: cerca l'occorrenza piu' recente dell'ultimo bigramma nel
 * contesto e propone i token che la seguirono. Zero pesi extra, zero costo: e' solo
 * un'ipotesi che il modello verifichera'. */
static int ngram_draft(const int *ids, int len, int G, int *draft){
    if(len<4 || G<1) return 0;
    int a=ids[len-2], b=ids[len-1];
    for(int i=len-3;i>=1;i--)
        if(ids[i-1]==a && ids[i]==b){
            int n=0; for(int j=i+1;j<len && n<G;j++) draft[n++]=ids[j];
            return n;
        }
    return 0;
}

/* METODO MTP: propone fino a G draft con la testa multi-token nativa di GLM-5.2.
 * Input: next_tok (appena emesso, posizione kv) e hlast (hidden pre-norm della pos kv-1).
 * Catena DeepSeek-V3: h' = Layer78( eh_proj[ enorm(emb(tok)) ; hnorm(h) ] ),
 * draft = argmax(lm_head(shared_head.norm(h'))). La KV del layer MTP vive alla riga n_layers
 * ed e' valida da kv_start (niente prefill: finestra di solo-decode, basta per il draft). */
static int mtp_argmax(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}
static int mtp_draft(Model *m, int next_tok, int kv, int G, int *draft){
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    int p=kv-1; if(p<0||G<1) return 0;
    if(m->kv_start[li]<0 || m->kv_start[li]>p) m->kv_start[li]=p;
    float *x=falloc(D), *cat=falloc(2*D), *hx=falloc(D), *nrm=falloc(D), *tmp=falloc(D);
    float *row=falloc(D), *logit=falloc(c->vocab), *h=falloc(D);
    memcpy(h, m->hlast, D*sizeof(float));
    int tok=next_tok, n=0;
    m->ld_ctx=1;                                 /* DISK_SPLIT: i load da qui sono draft-path */
    int prenorm = getenv("MTP_PRENORM")!=NULL;
    for(int g=0; g<G; g++){
        int pos=p+g; if(pos+2>=m->max_t) break;
        embed_row(m, tok, x);
        rmsnorm(x, x, m->enorm, D, c->eps);
        if(g==0 && !prenorm) rmsnorm(h, h, m->final_norm, D, c->eps);  /* h vero: post model.norm */
        rmsnorm(h, h, m->hnorm, D, c->eps);
        if(getenv("MTP_SWAP")){ memcpy(cat, h, D*sizeof(float)); memcpy(cat+D, x, D*sizeof(float)); }
        else { memcpy(cat, x, D*sizeof(float)); memcpy(cat+D, h, D*sizeof(float)); }
        matmul_qt(hx, cat, &m->eh_proj, 1);
        double n_eh=0; for(int d=0;d<D;d++) n_eh+=hx[d]*hx[d];
        int dbg = getenv("MTP_DEBUG") && atoi(getenv("MTP_DEBUG"))>=2;
        int t_pre=-1;
        if(dbg){ rmsnorm(row, hx, m->mtp_norm, D, c->eps); matmul_qt(logit, row, &m->lm_head, 1);
                 t_pre=mtp_argmax(logit, c->vocab); }
        layer_forward(m, &m->mtpL, li, hx, 1, pos, nrm, tmp);
        double n_post=0; for(int d=0;d<D;d++) n_post+=hx[d]*hx[d];
        rmsnorm(row, hx, m->mtp_norm, D, c->eps);
        matmul_qt(logit, row, &m->lm_head, 1);
        int t2=mtp_argmax(logit, c->vocab);
        if(dbg) fprintf(stderr,"[mtp2] pos=%d in_tok=%d ||eh||=%.1f ||post||=%.1f pre_blk=%d post_blk=%d\n",
                        pos, tok, sqrt(n_eh), sqrt(n_post), t_pre, t2);
        draft[n++]=t2; tok=t2; memcpy(h, hx, D*sizeof(float));
    }
    m->ld_ctx=0;
    free(x); free(cat); free(hx); free(nrm); free(tmp); free(row); free(logit); free(h);
    return n;
}
/* assorbe nella KV della testa MTP le coppie VERIFICATE (emb(token@pos+1), h_vero@pos):
 * next_ids[i] = token alla posizione pos_base+i+1; x[i] = hidden VERO a pos_base+i.
 * Un solo passaggio batch del layer MTP (il batch-union rende economici gli expert). */
static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base){
    if(!m->has_mtp || S<1) return;
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    if(m->kv_start[li]<0 || m->kv_start[li]>pos_base) m->kv_start[li]=pos_base;
    float *hx=falloc((int64_t)S*D), *cat=falloc(2*D), *e=falloc(D), *hn=falloc(D), *hf=falloc(D);
    int prenorm = getenv("MTP_PRENORM")!=NULL;
    for(int i=0;i<S;i++){
        embed_row(m,next_ids[i],e);
        rmsnorm(e,e,m->enorm,D,c->eps);
        if(prenorm) rmsnorm(hn,x+(int64_t)i*D,m->hnorm,D,c->eps);
        else { rmsnorm(hf,x+(int64_t)i*D,m->final_norm,D,c->eps);   /* vLLM: h POST model.norm */
               rmsnorm(hn,hf,m->hnorm,D,c->eps); }
        if(getenv("MTP_SWAP")){ memcpy(cat,hn,D*sizeof(float)); memcpy(cat+D,e,D*sizeof(float)); }
        else { memcpy(cat,e,D*sizeof(float)); memcpy(cat+D,hn,D*sizeof(float)); }
        matmul_qt(hx+(int64_t)i*D, cat, &m->eh_proj, 1);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    m->ld_ctx=2;                                 /* DISK_SPLIT: load del layer MTP in absorb */
    layer_forward(m,&m->mtpL,li,hx,S,pos_base,nrm,tmp);
    m->ld_ctx=0;
    free(hx); free(cat); free(e); free(hn); free(hf); free(nrm); free(tmp);
}


static void repin_pass_limit(Model *m,int limit);
static void repin_pass(Model *m){ repin_pass_limit(m,16); }

/* ---- METODO F: draft grammaticale (#48) ----
 * gr_feed consuma i byte di ogni token EMESSO e tiene il walker in sync con l'output;
 * grammar_draft propone lo span FORZATO successivo (un solo byte legale per posizione)
 * gia' tokenizzato. Il confine di tokenizzazione non e' garantito coincidere con quello
 * del modello: la verifica assorbe la differenza (al peggio l'ultimo draft e' rifiutato). */
static void grammar_setup(Tok *T){
    /* GRAMMAR=<file.gbnf> takes precedence; SCHEMA=<file.json> compiles a JSON-Schema
     * to GBNF (schema_gbnf.h) for the same draft source. Both fail soft: the engine
     * runs without a grammar and output is unchanged. */
    const char *gf=getenv("GRAMMAR");
    const char *sf=(gf&&*gf)?NULL:getenv("SCHEMA");
    if((!gf||!*gf)&&(!sf||!*sf)) return;
    const char *path=(gf&&*gf)?gf:sf;
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[GRAMMAR] cannot open %s\n",path); return; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *txt=malloc((size_t)n+1);
    if(!txt || fread(txt,1,(size_t)n,f)!=(size_t)n){
        fprintf(stderr,"[GRAMMAR] failed to read %s\n",path); fclose(f); free(txt); return; }
    fclose(f); txt[n]=0;
    if(sf){ /* schema -> GBNF, then the same gr_parse as the GRAMMAR path */
        char serr[160];
        char *gbnf=schema_to_gbnf(txt,serr,sizeof serr);
        free(txt);
        if(!gbnf){ fprintf(stderr,"[SCHEMA] %s: %s (running without grammar)\n",sf,serr); return; }
        txt=gbnf;
    }
    if(gr_parse(&g_gram,txt)){ fprintf(stderr,"[GRAMMAR] %s: %s\n",path,g_gram.err); free(txt); return; }
    free(txt);
    gr_state_init(&g_gst,&g_gram);
    if(!g_gst.alive){ fprintf(stderr,"[GRAMMAR] %s: grammar cannot be evaluated (left recursion?)\n",path); return; }
    if(getenv("GRAMMAR_DRAFT")) g_gr_max=atoi(getenv("GRAMMAR_DRAFT"));
    if(g_gr_max<1) g_gr_max=1;
    if(g_gr_max>48) g_gr_max=48;
    g_gr_T=T; g_gr_on=1;
    fprintf(stderr,"[GRAMMAR] %s: %d rules, forced span capped at %d tokens/forward\n",path,g_gram.n,g_gr_max);
}
/* stato pulito all'inizio di ogni RISPOSTA (non tra i \x02MORE, che continuano) */
static void grammar_reset(void){
    if(!g_gr_on) return;
    gr_state_init(&g_gst,&g_gram); g_gr_armed=0;
    if(!g_gst.alive) g_gr_on=0;
}
/* consuma i byte di un token emesso. Preambolo (prima dell'arming): ignorato.
 * Desync dopo l'arming: si riarma in attesa del prossimo inizio valido — al peggio
 * i draft vengono rifiutati dalla verifica, l'output non cambia MAI. */
static void gr_feed(int t){
    if(!g_gr_on||!g_gr_T) return;
    char b[64]; int n=tok_decode(g_gr_T,&t,1,b,63);
    for(int i=0;i<n;i++){
        int r=gr_accept(&g_gst,(unsigned char)b[i]);
        if(r==1){ g_gr_armed=1; continue; }
        if(r<0){ g_gr_on=0; return; }                 /* walker spento: fine dei draft */
        if(!g_gr_armed) continue;                     /* preambolo: aspetta l'inizio */
        gr_state_init(&g_gst,&g_gram); g_gr_armed=0;  /* desync: riparti dalla radice */
        if(!g_gst.alive){ g_gr_on=0; return; }
        if(gr_accept(&g_gst,(unsigned char)b[i])==1) g_gr_armed=1;
    }
}
/* propone lo span forzato come token (max cap); 0 se la grammatica dirama qui */
static int grammar_draft(int *draft, int cap){
    if(!g_gr_on||!g_gr_armed||!g_gr_T||cap<1) return 0;
    if(g_gr_prop>=32 && g_gr_acc*2<g_gr_prop){        /* guardia adattiva, come per MTP:
        acceptance sotto il 50% = tokenizzazione fuori asse, meglio spegnersi */
        g_gr_on=0;
        fprintf(stderr,"[GRAMMAR] %.0f%% acceptance after %llu proposals: grammar drafts disabled\n",
            100.0*g_gr_acc/g_gr_prop,(unsigned long long)g_gr_prop);
        return 0;
    }
    char fb[512]; int nb=gr_forced(&g_gst,fb,(int)sizeof fb-1);
    if(nb<=0) return 0;
    int g=tok_encode(g_gr_T,fb,nb,draft,cap);
    return g>0?g:0;
}

/* ---- SAMPLING (temperatura + nucleus) con verifica speculativa LOSSLESS ----
 * Il draft (MTP/n-gram) e' DETERMINISTICO (argmax della testa): q = massa puntuale.
 * Rejection sampling di Leviathan: accetta il draft x_d con prob p(x_d); al rifiuto
 * ricampiona da p con x_d azzerato e rinormalizzato. La distribuzione risultante e'
 * ESATTAMENTE p: la speculazione resta invisibile all'output anche col sampling. */

/* decode greedy con SELF-SPECULATION n-gram: LOSSLESS (output identico al greedy puro).
 * Ogni forward verifica fino a g_draft token proposti dal contesto: i token accettati
 * costano UNA sola passata sui pesi -> disco e banda RAM ammortizzati su piu' token.
 * all: storia token (capacita' >= kv+n_new+g_draft+2), kv = token gia' in KV.
 * logit = logits della posizione kv-1 (dal prefill); viene liberato qui.
 * emit(tok,ud) per ogni token emesso. Ritorna i token emessi; *kv_out = nuova kv. */
/* STOP MORBIDO (serve/chat): SIGINT chiude il turno CORRENTE per la stessa via
 * del tetto NGEN (stats, usage_save, KV append, sentinella END tutti normali)
 * invece di uccidere il motore; :more puo' continuare la risposta interrotta.
 * Il flag e' armato solo nei serve-loop (intr_install): nei run one-shot e in
 * validazione SIGINT resta il default (morte immediata). Solo POSIX: su
 * Windows il comportamento di Ctrl-C non cambia.
 * EN: soft stop (serve/chat): SIGINT ends the CURRENT turn through the same
 * path as the NGEN cap — stats/usage/KV/END sentinel all normal — instead of
 * killing the engine; :more can continue the interrupted answer. Armed only
 * in the serve loops; one-shot runs keep default SIGINT. POSIX only. */
static volatile sig_atomic_t g_intr=0;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
static void intr_sig(int s){ (void)s; g_intr=1; }
static void intr_install(void){
    struct sigaction sa; memset(&sa,0,sizeof(sa));
    sa.sa_handler=intr_sig; sigemptyset(&sa.sa_mask);
    sa.sa_flags=SA_RESTART;              /* getline/pread non devono vedere EINTR */
    sigaction(SIGINT,&sa,NULL);
}
#else
static void intr_install(void){}
#endif
static int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit,
                       void (*emit)(int,void*), void *ud, int *kv_out){
    Cfg *c=&m->c; int V=c->vocab; int emitted=0, done=0;
    int draft[64]; if(g_draft>63) g_draft=63;
    int carry_ban=-1;                    /* token rifiutato dalla verifica: escluso dal resample */
    /* #163: draft del modello attivi -> pin della famiglia di kernel per draft+verifica.
     * EN: model drafts live -> pin the kernel family for draft+verify forwards. */
    g_spec_live = (g_draft>0);
    if(spec_pinned() && m->has_mtp){ static int once=0; if(!once){ once=1;
        fprintf(stderr,"[SPEC_PIN] draft+verify pinned to the S=1 kernel family: int4=%s int8=%s (#163; SPEC_PIN=0 for A/B)\n",
            (g_idot&&g_i4s<=1)?"idot":"exact", g_idot?"idot":"exact"); } }
    /* guardia MTP morbida (#163): finestra di 24 proposte, pausa e ri-arma invece del
     * latch permanente — una regressione transitoria non spegne MTP per tutta la sessione.
     * EN: soft MTP guard (#163): 24-proposal window, pause and re-arm instead of the
     * permanent latch — a transient collapse no longer kills MTP for the whole session. */
    enum { GUARD_PAUSE_TOKENS = 256 };
    uint64_t gd_prop0=m->mtp_prop, gd_acc0=m->mtp_acc; int gd_pause=0;
    while(emitted<n_new && !done && !g_intr){   /* g_intr: stessa uscita del tetto n_new */
        int next=pick_tok(logit,V,carry_ban); carry_ban=-1; free(logit); logit=NULL;
        if((eos>=0 && next==eos) || is_stop(next)) break;
        emit(next,ud); all[kv]=next; emitted++; m->n_emit++;
        gr_feed(next);                                  /* il walker segue l'output emesso */
        if(emitted>=n_new) break;                       /* l'ultimo token non serve forwardarlo */
        int g = 0, gsrc = 0;                            /* sorgente: 1=grammatica 2=MTP/n-gram */
        if(g_gr_on){                                    /* metodo F: prima la grammatica — dove
                                                         * forza, l'acceptance e' ~1 (#48) */
            g=grammar_draft(draft,g_gr_max);
            if(g>0) gsrc=1;
        }
        if(!g && g_draft>0 && m->has_mtp){
            /* pausa adattiva: draft che non vengono mai accettati = solo tassa disco,
             * ma il vecchio g_draft=0 era permanente. EN: adaptive pause; the old
             * g_draft=0 latch was permanent. */
            if(gd_pause>0){ gd_pause--; if(!gd_pause){ gd_prop0=m->mtp_prop; gd_acc0=m->mtp_acc; } }
            else if(m->mtp_prop-gd_prop0>=24 && (m->mtp_acc-gd_acc0)*10 < m->mtp_prop-gd_prop0){
                fprintf(stderr,"[MTP] %.0f%% acceptance over the last %llu proposals: drafts paused for %d tokens\n",
                    100.0*(m->mtp_acc-gd_acc0)/(m->mtp_prop-gd_prop0),
                    (unsigned long long)(m->mtp_prop-gd_prop0), (int)GUARD_PAUSE_TOKENS);
                gd_pause=GUARD_PAUSE_TOKENS;
            }
        }
        if(!g && g_draft>0 && !(m->has_mtp && gd_pause>0)){
            if(m->has_mtp){ g=mtp_draft(m,next,kv,g_draft,draft); m->mtp_prop+=g; if(g)gsrc=2; }
            else { g=ngram_draft(all,kv+1,g_draft,draft); if(g)gsrc=2; }
        }
        if(g>n_new-emitted) g=n_new-emitted;
        if(kv+1+g+1>m->max_t) g=m->max_t-kv-2;
        if(g<0) g=0;
        if(gsrc==1) g_gr_prop+=(uint64_t)g;
        int S=1+g; int batch[64]; batch[0]=next; memcpy(batch+1,draft,g*sizeof(int));
        double tf0=g_prof?now_s():0;
        float *lo=step_all(m,batch,S,kv); m->n_fw++;
        if(g_prof) prof_lat(now_s()-tf0);
        int k=0;                                        /* verifica: accetta finche' coincide */
        if(g>0 && getenv("MTP_DEBUG")){ int veri=argmax_v(lo,V);
            fprintf(stderr,"[mtpdbg] draft0=%d verified=%d %s\n", draft[0], veri, draft[0]==veri?"HIT":"miss"); }
        while(k<g && emitted<n_new){
            int accept;
            if(g_temp<=0) accept = (argmax_v(lo+(int64_t)k*V,V)==draft[k]);
            else { dist_build(lo+(int64_t)k*V,V);          /* rejection sampling: p(draft) */
                   accept = (rndu() < g_pbuf[draft[k]]); }
            if(!accept){ if(g_temp>0) carry_ban=draft[k]; break; }
            if((eos>=0 && draft[k]==eos) || is_stop(draft[k])){ done=1; break; }
            emit(draft[k],ud); all[kv+1+k]=draft[k]; emitted++; m->n_emit++;
            gr_feed(draft[k]); k++;
        }
        if(gsrc==1) g_gr_acc+=(uint64_t)k;
        else if(gsrc==2 && m->has_mtp) m->mtp_acc+=k;
        if(m->has_mtp && k>=1) mtp_absorb(m, all+kv+1, m->h_all, k, kv);   /* KV MTP in sync coi verificati */
        /* hlast deve corrispondere all'ultima posizione ACCETTATA (kv+k), non a fine batch */
        if(m->h_all && k<S-1) memcpy(m->hlast, m->h_all+(int64_t)k*m->c.hidden, m->c.hidden*sizeof(float));
        kv += 1+k;                                      /* KV oltre kv e' stantia: verra' sovrascritta */
        logit=falloc(V); memcpy(logit, lo+(int64_t)k*V, V*sizeof(float)); free(lo);
        repin_pass(m);                                  /* safe point: all device work is synchronized */
    }
    g_spec_live = 0;                     /* prefill/decode successivi: gate normali / next prefill: normal gates */
    if(logit) free(logit);
    if(kv_out) *kv_out=kv;
    return emitted;
}

/* emit callback: accumula in un array (validazione) */
typedef struct { int *dst; int n; } EmitStore;
static void emit_store(int t, void *ud){ EmitStore *e=(EmitStore*)ud; e->dst[e->n++]=t; }
/* emit callback: detokenizza e stampa in streaming (chat/run), con heartbeat */
typedef struct { Tok *T; Model *m; double t0; int count; int quiet; } EmitStream;
static void emit_stream(int t, void *ud){
    EmitStream *e=(EmitStream*)ud; char dec[64];
    int dn=tok_decode(e->T,&t,1,dec,63); dec[dn]=0; fputs(dec,stdout); fflush(stdout);
    if(!e->quiet && ++e->count%16==0){ double tt=e->m->hits+e->m->miss;
        if(g_cache_route && e->m->route_slots){
            double swap=100.0*e->m->route_swaps/e->m->route_slots;
            fprintf(stderr,"\n[t=%d  RSS %.2f GB  hit %.0f%%  swap %.0f%%  %.2f tok/s  %.2f tok/fw]\n", e->count,
                rss_gb(), tt?100.0*e->m->hits/tt:0.0, swap, e->count/(now_s()-e->t0),
                e->m->n_fw?(double)e->m->n_emit/e->m->n_fw:1.0);
        } else {
            fprintf(stderr,"\n[t=%d  RSS %.2f GB  hit %.0f%%  %.2f tok/s  %.2f tok/fw]\n", e->count,
                rss_gb(), tt?100.0*e->m->hits/tt:0.0, e->count/(now_s()-e->t0),
                e->m->n_fw?(double)e->m->n_emit/e->m->n_fw:1.0);
        }
    }
}

/* teacher-forcing: un solo forward su ids[S], argmax per posizione in pred[S] */
static void forward_all(Model *m, const int *ids, int S, int *pred){
    Cfg *c=&m->c; int D=c->hidden;
    kv_alloc(m,S);
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,0);
    float *lo=falloc(c->vocab);
    float *row=falloc(D);
    for(int s=0;s<S;s++){
        rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);   /* heap row (#183) */
        matmul_qt(lo, row, &m->lm_head, 1);
        int best=0; float bv=lo[0]; for(int i=1;i<c->vocab;i++) if(lo[i]>bv){bv=lo[i];best=i;}
        pred[s]=best;
    }
    free(x); free(lo); free(row);
}

/* log-prob (log-softmax) del token target dato il vettore di logit; *am=1 se e' l'argmax */
/* modalita' SCORING per i benchmark (stile lm-eval, log-likelihood):
 * input: file con righe "<ctxlen> <contlen> <id0> .. <id_{T-1}>"  (T=ctxlen+contlen)
 * output: riga "<logprob_continuazione> <contlen> <greedy 0/1>" per richiesta.
 * Un solo forward per richiesta (teacher-forcing): niente generazione -> fattibile a bassa velocita'. */
static void run_score(Model *m, const char *snap, const char *path){
    Cfg *c=&m->c; int D=c->hidden;
    /* prefisso GLM (#108): il modello vede [gMASK]<sop> in testa a OGNI sequenza di training —
     * scorare stream nudi e' out-of-distribution e deprime/distorce i punteggi. Se il config
     * dice glm* gli id dei due token vengono chiesti al tokenizer.json dello snapshot (per
     * GLM-5.2: 154822,154824 — mai fidarsi di costanti cablate, il vocabolario cambia tra
     * release) e anteposti al CONTESTO delle richieste che non li hanno gia'; chi arriva GIA'
     * prefissato (eval_glm.py post-#194) passa INTATTO. SCORE_PREFIX=0 -> comportamento nudo. */
    int pfx[2]={-1,-1}, pfx_on=0;
    if(!getenv("SCORE_PREFIX")||atoi(getenv("SCORE_PREFIX"))){
        char *ar=NULL; jval *r=cfg_root(snap,&ar);
        jval *mt=json_get(r,"model_type");
        if(mt_is_glm(mt?mt->str:NULL)){
            char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
            Tok T; tok_load(&T,tkp);
            pfx[0]=tok_id_of(&T,"[gMASK]"); pfx[1]=tok_id_of(&T,"<sop>");
            if(pfx[0]>=0&&pfx[1]>=0){ pfx_on=1;
                fprintf(stderr,"[SCORE] GLM snapshot: prepending [gMASK]<sop> (ids %d,%d) to unprefixed requests — disable with SCORE_PREFIX=0\n",pfx[0],pfx[1]);
            } else fprintf(stderr,"[SCORE] GLM config but tokenizer has no [gMASK]/<sop>: prefix OFF\n");
        }
        free(ar);
    }
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    int maxT=1; { char *ln=NULL; size_t cp=0;
        while(getline(&ln,&cp,f)>0){ int a,b; if(sscanf(ln,"%d %d",&a,&b)==2 && a+b>maxT) maxT=a+b; }
        free(ln); }
    if(pfx_on) maxT+=2;   /* le richieste senza prefisso crescono di 2 token */
    kv_alloc(m,maxT);
    float *x=falloc((int64_t)maxT*D), *lo=falloc(c->vocab), *row=falloc(D);
    int *ids=malloc(maxT*sizeof(int));
    rewind(f); char *ln=NULL; size_t cp=0; int nreq=0; double t0=now_s();
    while(getline(&ln,&cp,f)>0){
        char *p=ln; int ctxlen=strtol(p,&p,10), contlen=strtol(p,&p,10), T=ctxlen+contlen;
        if(T<=0||ctxlen<1){ printf("0 0 0\n"); fflush(stdout); continue; }
        for(int i=0;i<T;i++) ids[i]=strtol(p,&p,10);
        if(pfx_on && !(T>=2 && ids[0]==pfx[0] && ids[1]==pfx[1])){   /* gia' prefissato -> intatto */
            memmove(ids+2,ids,(size_t)T*sizeof(int));
            ids[0]=pfx[0]; ids[1]=pfx[1]; ctxlen+=2; T+=2;
        }
        for(int s=0;s<T;s++) embed_row(m, ids[s], x+(int64_t)s*D);
        layers_forward(m,x,T,0);
        double lp=0; int greedy=1;
        for(int pos=ctxlen-1; pos<T-1; pos++){
            rmsnorm(row, x+(int64_t)pos*D, m->final_norm, D, c->eps);
            matmul_qt(lo,row,&m->lm_head,1);
            int am; lp += logprob_target(lo,c->vocab,ids[pos+1],&am); if(!am) greedy=0;
        }
        printf("%.6f %d %d\n", lp, contlen, greedy); fflush(stdout);
        if(++nreq%5==0) fprintf(stderr,"[score %d req | %.1fs | RSS %.2f GB | hit %.0f%%]\n",
            nreq, now_s()-t0, rss_gb(), (m->hits+m->miss)?100.0*m->hits/(m->hits+m->miss):0.0);
    }
    free(ln); free(ids); free(x); free(lo); free(row); fclose(f);
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out){
    kv_alloc(m,np+n_new+g_draft+2);
    for(int i=0;i<np;i++) out[i]=prompt[i];
    float *logit=step(m,prompt,np,0);
    EmitStore es={out+np,0};
    spec_decode(m,out,np,n_new,-1,logit,emit_store,&es,NULL);
}

static void profile_print(Model *m, double elapsed){
    double accounted=m->t_ewait+m->t_emm+m->t_attn+m->t_head;
    printf("PROFILE: expert-disk %.3fs service / %.3fs wait | expert-matmul %.3fs | attention %.3fs "
           "(including kvb %.3fs) | lm_head %.3fs | other %.3fs\n",
        edisk_s(),m->t_ewait,m->t_emm,m->t_attn,m->t_kvb,m->t_head,elapsed-accounted);
    printf("ATTENTION: projection/RoPE %.3fs | score-softmax-value %.3fs | output projection %.3fs\n",
        m->t_aproj,m->t_acore,m->t_aout);
    if(g_prof)printf("P0-EXEC: routed CPU %.3fs / %.2f GB/s (%llu row) | routed GPU critical %.3fs | router %.3fs | residual P2P %.3fs / %llu hop | orchestration %.3fs\n",
        m->t_ecpu,m->t_ecpu>0?m->cpu_expert_bytes/1e9/m->t_ecpu:0.0,
        (unsigned long long)m->cpu_expert_rows,m->t_egpu,m->t_route,m->t_p2p,(unsigned long long)m->n_p2p,
        elapsed-m->t_ewait-m->t_emm-m->t_attn-m->t_head-m->t_route-m->t_p2p>0?
        elapsed-m->t_ewait-m->t_emm-m->t_attn-m->t_head-m->t_route-m->t_p2p:0);
#ifdef COLI_METAL
    if(g_metal_enabled){ uint64_t ok=0,fb=0,ex=0; double su=0,gp=0,sc=0;
        coli_metal_moe_counts(&ok,&fb,&ex); coli_metal_moe_times(&su,&gp,&sc);
        { uint64_t aok=0; double aw=0,ak=0; coli_metal_attn_counts(&aok,&aw,&ak);
          if(aok){ double ks=0,gs=0; coli_metal_attn_lat(&ks,&gs);
          printf("METAL-ATTN: layer GPU %llu | gpu-wall %.2fs (kernel %.2fs | cpu-sched %.2fs gpu-sched %.2fs)\n",(unsigned long long)aok,aw,ak,ks,gs); } }
        printf("METAL: blocchi GPU %llu | fallback CPU %llu | expert su GPU %llu | setup %.2fs gpu-wall %.2fs (kernel %.2fs) scatter %.2fs\n",
               (unsigned long long)ok,(unsigned long long)fb,(unsigned long long)ex,su,gp,coli_metal_moe_kernel_time(),sc); }
#endif
}

static void profile_reset(Model *m){
    m->t_ewait=m->t_emm=m->t_attn=m->t_kvb=m->t_head=0;
    m->t_ecpu=m->t_egpu=m->t_route=m->t_p2p=0;m->n_p2p=0;
    m->cpu_expert_bytes=0;m->cpu_expert_rows=0;
    m->t_aproj=m->t_acore=m->t_aout=0;
    atomic_store_explicit(&g_edisk_ns,0,memory_order_relaxed);
}

/* PROF=1 report: forward-latency percentiles, expert I/O totals, phase shares
 * of wall time, and a plain-language verdict naming the knob most likely to
 * move tok/s on THIS machine with THIS config. `b` marks the window start
 * (serve mode reports per turn; batch modes snapshot right after reset). */
static int prof_cmp_d(const void *a,const void *b){
    double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }
static void prof_report(Model *m, const ProfBase *b, double elapsed, int tokens, FILE *f){
    Cfg *c=&m->c; if(elapsed<1e-9) elapsed=1e-9;
    uint64_t nw=g_prof_nlat-b->nlat; if(nw>PROF_LAT_CAP) nw=PROF_LAT_CAP;  /* ring keeps the tail */
    uint64_t nfw=m->n_fw-b->n_fw, nem=m->n_emit-b->n_emit;
    if(nw){
        double *v=malloc((size_t)nw*sizeof(double));
        if(v){
            for(uint64_t i=0;i<nw;i++) v[i]=g_prof_lat[(g_prof_nlat-nw+i)%PROF_LAT_CAP];
            qsort(v,(size_t)nw,sizeof(double),prof_cmp_d);
            double p50=v[(nw-1)/2], p90=v[(uint64_t)((nw-1)*0.90)], p99=v[(uint64_t)((nw-1)*0.99)], mx=v[nw-1];
            fprintf(f,"[PROF] decode forwards: %llu | latency p50 %.1f ms | p90 %.1f ms | p99 %.1f ms | max %.1f ms | %.2f tok/forward\n",
                (unsigned long long)nfw,p50*1e3,p90*1e3,p99*1e3,mx*1e3,nfw?(double)nem/nfw:0.0);
            if(nw>=32 && p99>3*p50)
                fprintf(f,"[PROF] tail: p99 is %.1fx p50 — the slow forwards are cold-cache expert loads; "
                          "a warm-up turn or a pinned hot-store (PIN / AUTOPIN history) shrinks them\n",p99/p50);
            free(v);
        }
    }
    int64_t io=atomic_load_explicit(&g_prof_io,memory_order_relaxed)-b->io;
    uint64_t dh=m->hits-b->hits, dm=m->miss-b->miss, dq=m->ereq-b->ereq;
    double hitp=(dh+dm)?100.0*dh/(dh+dm):100.0;
    double eb=(double)expert_bytes_probe(m,m->ebits);
    int pinned=0,lru=0;
    for(int i=0;i<=c->n_layers;i++){ if(m->npin)pinned+=m->npin[i]; if(m->ecn)lru+=m->ecn[i]; }
    double io_w=m->t_ewait-b->ewait;    /* stall the compute thread felt */
    double io_svc=edisk_s()-b->edisk;   /* read service on the loading threads (overlaps compute) */
    uint64_t dhp=m->hit_pin-b->hit_pin, dhe=m->hit_ecache-b->hit_ecache;   /* split #336 */
    fprintf(f,"[PROF] expert I/O: %.3f GB fetched (%.1f MB/token, %.2f GB/s over the run%s) | "
              "hit %.1f%% (%llu pin + %llu lru / %llu load) | %.1f loads/token | %.1fs read service / %.1fs felt wait\n",
        io/1e9, tokens>0?io/1e6/tokens:0.0, io/1e9/elapsed,
        g_mmap?"; COLI_MMAP=1: page cache may serve part":"",
        hitp,(unsigned long long)dhp,(unsigned long long)dhe,(unsigned long long)dm, tokens>0?(double)dq/tokens:0.0,
        io_svc,io_w);
    /* DISK-CLASS: per-load cold/warm classification vs. which fd ACTUALLY served it.
     * Three per-class rates, labeled to keep the units unambiguous (ambiguous units
     * mislead -- measured lesson): GB/s-thread = bytes / thread-seconds (per-read
     * service rate, same convention as the read-service line above); GB/s-wall =
     * bytes / busy-wall (aggregate rate the disk actually delivered while >=1 load of
     * the class was in flight); avg-conc = thread-seconds / busy-wall (mean overlap
     * depth). disk-busy = combined busy-wall (either class in flight) as a share of
     * the profile window. */
    {
        uint64_t dcn[2], dcdn[2]; int64_t dcbytes[2], dcns[2], dcwall[2], wnow[2], dcwall_all, wall_now;
        dc_wall_read(wnow,&wall_now);
        for(int i=0;i<2;i++){
            dcn[i]=atomic_load_explicit(&g_dc_n[i],memory_order_relaxed)-b->dc_n[i];
            dcbytes[i]=atomic_load_explicit(&g_dc_bytes[i],memory_order_relaxed)-b->dc_bytes[i];
            dcns[i]=atomic_load_explicit(&g_dc_ns[i],memory_order_relaxed)-b->dc_ns[i];
            dcdn[i]=atomic_load_explicit(&g_dc_direct_n[i],memory_order_relaxed)-b->dc_direct_n[i];
            dcwall[i]=wnow[i]-b->dc_wall_ns[i];
        }
        dcwall_all=wall_now-b->dc_wall_all_ns;
        if(dcn[DC_COLD]+dcn[DC_WARM]){
            double ct=dcns[DC_COLD]>0?(double)dcbytes[DC_COLD]/dcns[DC_COLD]:0.0;      /* bytes/ns = GB/s */
            double wt=dcns[DC_WARM]>0?(double)dcbytes[DC_WARM]/dcns[DC_WARM]:0.0;
            double cw=dcwall[DC_COLD]>0?(double)dcbytes[DC_COLD]/dcwall[DC_COLD]:0.0;
            double ww=dcwall[DC_WARM]>0?(double)dcbytes[DC_WARM]/dcwall[DC_WARM]:0.0;
            double cc=dcwall[DC_COLD]>0?(double)dcns[DC_COLD]/dcwall[DC_COLD]:0.0;
            double wc=dcwall[DC_WARM]>0?(double)dcns[DC_WARM]/dcwall[DC_WARM]:0.0;
            fprintf(f,"[PROF] DISK-CLASS (recency split): cold %llu x %.2f GB @ %.2f GB/s-thread, wall %.1fs @ %.2f GB/s-wall, avg-conc %.1f (direct %llu/%llu) | "
                      "warm %llu x %.2f GB @ %.2f GB/s-thread, wall %.1fs @ %.2f GB/s-wall, avg-conc %.1f (direct %llu/%llu) | "
                      "disk-busy %.1fs (%.0f%% of window)\n",
                (unsigned long long)dcn[DC_COLD],dcbytes[DC_COLD]/1e9,ct,dcwall[DC_COLD]/1e9,cw,cc,
                (unsigned long long)dcdn[DC_COLD],(unsigned long long)dcn[DC_COLD],
                (unsigned long long)dcn[DC_WARM],dcbytes[DC_WARM]/1e9,wt,dcwall[DC_WARM]/1e9,ww,wc,
                (unsigned long long)dcdn[DC_WARM],(unsigned long long)dcn[DC_WARM],
                dcwall_all/1e9,100.0*(dcwall_all/1e9)/elapsed);
        }
    }
    fprintf(f,"[PROF] resident experts: %d pinned (%.1f GB) + %d in LRU (%.1f GB, cap %d/layer)\n",
        pinned,pinned*eb/1e9,lru,lru*eb/1e9,m->ecap);
    double emm=m->t_emm-b->emm, ecpu=m->t_ecpu-b->ecpu, egpu=m->t_egpu-b->egpu;
    double route=m->t_route-b->route,p2p=m->t_p2p-b->p2p;
    uint64_t np2p=m->n_p2p-b->n_p2p;
    int64_t cpu_bytes=m->cpu_expert_bytes-b->cpu_bytes;
    uint64_t cpu_rows=m->cpu_expert_rows-b->cpu_rows;
    double attn=m->t_attn-b->attn, head=m->t_head-b->head;
    double other=elapsed-io_w-emm-attn-head-route-p2p; if(other<0) other=0;
    double f_io=io_w/elapsed, f_emm=emm/elapsed, f_attn=attn/elapsed;
    fprintf(f,"[PROF] time shares: expert-I/O %.0f%% | expert-matmul %.0f%% | attention %.0f%% | lm_head %.0f%% | other %.0f%%\n",
        100*f_io,100*f_emm,100*f_attn,100*head/elapsed,100*other/elapsed);
    double slow=ecpu>egpu?ecpu:egpu,fast=ecpu<egpu?ecpu:egpu;
    fprintf(f,"[PROF] P0 execution: routed CPU %.3fs / %.2f GB/s (%llu row) | routed GPU critical %.3fs | tier straggler %.2fx | "
              "router %.3fs | residual P2P %.3fs (%llu hop, %.3f ms/hop) | orchestration %.3fs\n",
        ecpu,ecpu>0?cpu_bytes/1e9/ecpu:0.0,(unsigned long long)cpu_rows,
        egpu,fast>1e-9?slow/fast:0.0,route,p2p,(unsigned long long)np2p,
        np2p?p2p*1e3/np2p:0.0,other);
    if(f_io>=0.30){
        fprintf(f,"[PROF] verdict: I/O-bound — %.0f%% of the time waits on expert reads (hit %.0f%%).",100*f_io,hitp);
        if(hitp<90) fprintf(f," More cache is the lever: raise RAM_GB (or add RAM).");
        else fprintf(f," The cache is already warm — the routed working set streams from disk; a faster disk or a bigger pinned tier (PIN_GB) is the lever.");
        if(!g_pipe) fprintf(f," Try PIPE=1 (overlap reads with matmul).");
        if(!g_direct) fprintf(f," On NVMe try DIRECT=1.");
        fprintf(f,"\n");
    } else if(f_emm>=0.40){
        fprintf(f,"[PROF] verdict: compute-bound in expert matmuls (%.0f%%) — more cores/threads help; keep IDOT=1, or move hot experts to a GPU tier (COLI_CUDA / COLI_METAL).%s\n",
            100*f_emm, g_mmap?" Note: with COLI_MMAP=1 page-fault I/O is accounted inside matmul.":"");
    } else if(f_attn>=0.35){
        fprintf(f,"[PROF] verdict: attention-bound (%.0f%%) — context length is the cost (DSA %s). A lower CTX helps if the workload allows.\n",
            100*f_attn, m->has_dsa?"on":"not available for this model");
    } else {
        fprintf(f,"[PROF] verdict: balanced — no phase dominates (I/O %.0f%%, matmul %.0f%%, attention %.0f%%); this config is a reasonable fit for this machine.\n",
            100*f_io,100*f_emm,100*f_attn);
    }
}

/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ. */
static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\n"); return; }
    kv_alloc(m,nfull+2);
    float *logit=step(m,full,np-1,0); free(logit);
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0; m->hit_pin=m->hit_ecache=0;
    profile_reset(m);
    ProfBase pb; prof_base(m,&pb);
    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){
        double tf0=g_prof?now_s():0;
        logit=step(m,full+i,1,i); free(logit); steps++;
        if(g_prof){ prof_lat(now_s()-tf0); m->n_fw++; m->n_emit++; }
    }
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
    if(g_prof) prof_report(m,&pb,dt,steps,stdout);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
}

/* generazione reale: tokenizza PROMPT, prefill + decode greedy con stop su EOS,
 * detokenizza e stampa il testo in streaming. */
static void run_text(Model *m, const char *snap, const char *prompt, int ngen){
    Cfg *c=&m->c; char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm_tok(&m->c, eos, &T);
    grammar_setup(&T);                   /* metodo F: GRAMMAR=file.gbnf (#48) */
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int cap=(int)strlen(prompt)+16; int *pids=malloc(cap*sizeof(int));
    int np=tok_encode(&T,prompt,(int)strlen(prompt),pids,cap);
    if(np<1){ fprintf(stderr,"prompt is empty after tokenization\n"); return; }
    printf("prompt: %d tokens | generating up to %d (EOS stop=%d) | n-gram draft=%d\n", np, ngen, eos, g_draft);
    fputs(prompt,stdout); fflush(stdout);
    kv_alloc(m, np+ngen+g_draft+2);
    int *all=malloc((np+ngen+g_draft+2)*sizeof(int)); memcpy(all,pids,np*sizeof(int));
    double prefill_t=now_s();
    float *logit=step(m,pids,np,0);
    if(g_repin>0){
        m->n_emit=(uint64_t)g_repin;
        int limit=32;
#ifdef COLI_CUDA
        if(m->gpu_expert_count) limit=m->c.n_layers;
#endif
        repin_pass_limit(m,limit);                  /* prompt routing seeds every GPU layer */
    }
    prefill_t=now_s()-prefill_t;
    printf("PROFILO PREFILL (%.2fs):\n",prefill_t); profile_print(m,prefill_t);
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0; m->hit_pin=m->hit_ecache=0;
    m->n_emit=m->n_fw=0;
    g_last_repin=0;
    profile_reset(m);
    ProfBase pb; prof_base(m,&pb);
    double t=now_s();
    EmitStream es={&T,m,t,0,0};
    grammar_reset();
    int produced=spec_decode(m,all,np,ngen,eos,logit,emit_stream,&es,NULL);
    double dt=now_s()-t;
    double tot=m->hits+m->miss;
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    printf("\n---\nprefill %d tokens in %.2fs | decode %d tokens in %.2fs (%.2f tok/s) | "
           "expert hit rate %.1f%% (pin %.1f%% + lru %.1f%%) | RSS %.2f GB",       /* split #336: quale tier serve gli hit */
        np,prefill_t,produced,dt,produced/dt,tot?100.0*m->hits/tot:0.0,
        tot?100.0*m->hit_pin/tot:0.0, tot?100.0*m->hit_ecache/tot:0.0, rss_gb());
    if(g_cache_route && m->route_slots)
        printf(" | swap %.1f%% (%llu/%llu)",
            100.0*m->route_swaps/m->route_slots,
            (unsigned long long)m->route_swaps,(unsigned long long)m->route_slots);
    if(m->route_agree_tot)
        printf(" | route_agree %.1f%% | route_kl %.4f",
            100.0*m->route_agree_hit/m->route_agree_tot,
            m->route_kl_n?m->route_kl_sum/(double)m->route_kl_n:0.0);
    printf("\n");
    printf("experts loaded/token: %.1f (per-layer %.2f across %d; baseline topk=%d) | TOPK=%d TOPP=%.2f",
        produced?(double)m->ereq/produced:0.0, (produced&&nsp)?(double)m->ereq/produced/nsp:0.0, nsp, c->topk, g_topk, g_topp);
    if(g_cache_route) printf(" | CACHE_ROUTE J=%d M=%d P=%.2f alpha=%.2f", g_route_j, g_route_m, g_route_p, g_route_alpha);
    if(g_expert_budget) printf(" | EXPERT_BUDGET=%d (dropped %lld experts, ~%.1f GB I/O saved)", g_expert_budget, (long long)g_budget_dropped, g_budget_dropped*18.9e6/1e9);
    printf("\n");
    printf("speculation: %.2f tokens/forward (%llu forwards per %llu tokens) | MTP acceptance %.0f%% (%llu/%llu)\n",
        m->n_fw?(double)m->n_emit/m->n_fw:1.0, (unsigned long long)m->n_fw, (unsigned long long)m->n_emit,
        m->mtp_prop?100.0*m->mtp_acc/m->mtp_prop:0.0, (unsigned long long)m->mtp_acc, (unsigned long long)m->mtp_prop);
    if(g_cp_enq) printf("couple: %ld cross-layer prefetch hints enqueued\n", g_cp_enq);
    if(g_gr_prop) printf("grammar: %.0f%% acceptance (%llu/%llu forced drafts)\n",
        100.0*g_gr_acc/g_gr_prop, (unsigned long long)g_gr_acc, (unsigned long long)g_gr_prop);
    if(g_disk_split) printf("disk-load split: draft %llu + absorb %llu + verify/main %llu misses | "
           "MTP-layer %llu loads %.2f GB | main-layers %llu loads %.2f GB (MTP %.1f%% of bytes)\n",
        (unsigned long long)m->miss_draft, (unsigned long long)m->miss_absorb,
        (unsigned long long)(m->miss - m->miss_draft - m->miss_absorb),
        (unsigned long long)m->ld_mtp, m->bytes_mtp/1e9,
        (unsigned long long)m->ld_main, m->bytes_main/1e9,
        (m->bytes_mtp+m->bytes_main)?100.0*m->bytes_mtp/(m->bytes_mtp+m->bytes_main):0.0);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    profile_print(m,dt);
    if(g_prof) prof_report(m,&pb,dt,produced,stdout);
    if(g_pilot_real) printf("PILOT_REAL: %ld load cross-layer completati, %ld scartati (main gia' sul layer) | PILOT_K=%d\n",
        (long)atomic_load_explicit(&g_pilot_loads,memory_order_relaxed),
        (long)atomic_load_explicit(&g_pilot_drops,memory_order_relaxed), g_pilot_k);
    if(g_pilot_two) printf("PILOT_TWO: two-step shared-expert-corrected prefetch active (3 extra matmuls/prediction)\n");
    if(g_looka){
        const char *nm[4]={"previous token (=SPEC prefetch)","layer input, skip attention","next layer (PILOT, stale)","next layer (two-step, shared-expert)"};
        printf("LOOKAHEAD routing — recall of true experts in predicted top-8:\n");
        for(int i=0;i<4;i++) printf("  %-42s %5.1f%%  (%lld/%lld)\n", nm[i],
            la_tot[i]?100.0*la_hit[i]/la_tot[i]:0.0, (long long)la_hit[i], (long long)la_tot[i]);
    }
    /* TOKENS=1: dump the generated token ids (newline-separated) to stderr,
     * for exact A/B comparison across decode paths (e.g. resident vs CPU).
     * The ids are all[np .. np+produced-1]. */
    if(getenv("TOKENS") && atoi(getenv("TOKENS"))){
        fprintf(stderr,"[TOKENS] %d generated:",produced);
        for(int i=np;i<np+produced;i++) fprintf(stderr," %d",all[i]);
        fprintf(stderr,"\n");
    }
    free(pids); free(all);
    usage_save(m);
}

/* modalita' SERVE (per la CLI 'coli'): carica il modello UNA volta, poi CHAT conversazionale.
 * KV-cache PERSISTENTE tra i turni: la storia resta in cache, si fa il prefill solo dei
 * token NUOVI -> il modello RICORDA la conversazione e non ri-processa il passato (lossless,
 * piu' umano, piu' veloce). Template chat GLM con token speciali (CHAT_TEMPLATE=0 -> grezzo).
 * Protocollo: "\x01\x01" "READY" "\x01\x01\n" dopo il load; risposta in streaming; "\x01\x01" "END" "\x01\x01\n" a fine turno.
 * ":reset" (riga "\x02RESET") azzera la memoria. EOF -> esce. */
/* ---- RFC: RE-PIN A CALDO / LIVE RE-PIN (opt-in, REPIN=n, default OFF) ----
 * Upstream fa AUTOPIN allo START (dalla storia .coli_usage). Questo aggiunge un re-pin
 * TRA I TURNI: nel punto sicuro dopo la risposta scambia i pin peggiori con i non-pinnati
 * piu' caldi, cosi' l'hot-store insegue il carico VIVO senza un profilo a parte. Isteresi
 * 25% (+4) contro il ping-pong; max 4 scambi/passata (~20 MB di disco l'uno). Una heat
 * map separata decade a ogni passata: la storia persistente .coli_usage resta intatta.
 * EN: upstream AUTOPINs at START (from .coli_usage). This adds a between-turns re-pin: at
 * the safe point after the reply, swap the worst pins for the hottest unpinned, so the
 * hot-store tracks the LIVE workload without a separate profile. 25% (+4) hysteresis vs
 * ping-pong; max 4 swaps/pass (~20 MB disk each). A separate decaying heat map keeps
 * persistent .coli_usage intact while adapting to the current workload. */
typedef struct { long gain; int l, slot, eid, gpu_swap; } RepinCand;
static int repin_pick(Model *m, RepinCand *out, int maxc){
    Cfg *c=&m->c; int nb=0;
    for(int l=0;l<c->n_layers;l++){
        if(!m->npin || m->npin[l]<1 || !m->eheat[l]) continue;
#ifdef COLI_CUDA
        int cold=-1,hot=-1;
        for(int z=0;z<m->npin[l];z++){
            ESlot *s=&m->pin[l][z]; uint32_t heat=m->eheat[l][s->eid];
            if(s->g.cuda_eligible){
                if(cold<0||heat<m->eheat[l][m->pin[l][cold].eid]) cold=z;
            }else if(hot<0||heat>m->eheat[l][m->pin[l][hot].eid]) hot=z;
        }
        if(cold>=0&&hot>=0){
            uint32_t ch=m->eheat[l][m->pin[l][cold].eid],hh=m->eheat[l][m->pin[l][hot].eid];
            if(hh>ch+1){
                RepinCand v={(long)hh-(long)ch,l,cold,m->pin[l][hot].eid,1};
                if(nb<maxc) out[nb++]=v;
                else { int w=0; for(int b=1;b<maxc;b++) if(out[b].gain<out[w].gain)w=b;
                       if(v.gain>out[w].gain)out[w]=v; }
                continue;
            }
        }
#endif
        ESlot *P=m->pin[l]; int ids[4096], zp, eu; long g;
        int np=m->npin[l]; if(np>4096) np=4096;
        for(int z=0;z<np;z++) ids[z]=P[z].eid;
        if(!tier_pick_lfru(m->eheat[l],m->elast[l],m->eaccess_clock,
                           c->n_experts,ids,np,&zp,&eu,&g)) continue;
        if(nb<maxc){ out[nb]=(RepinCand){g,l,zp,eu,0}; nb++; }
        else { int w=0; for(int b=1;b<maxc;b++) if(out[b].gain<out[w].gain) w=b;
               if(g>out[w].gain) out[w]=(RepinCand){g,l,zp,eu,0}; }
    }
    return nb;
}
/* ---- RSS GUARD (#403) -----------------------------------------------------
 * La proiezione di cap_for_ram e' una STIMA: sul GB10 (#403) le generazioni
 * lunghe l'hanno sforata di ~40 GB (proiettato 74.4, reale 115.6 -> 3 kill del
 * kernel). La run D dell'issue prova che un cap piu' basso CONTIENE la crescita:
 * questa guardia lo fa da sola, sull'RSS MISURATO invece che sul proiettato.
 * Al safe point (stessa sede di repin: nessun moe in volo), ogni ~16 token
 * emessi: se l'RSS supera il budget, svuota gli slot LRU meno usati e abbassa
 * ecap perche' non ricrescano. Gli slab sono >128KB (mmap'd da glibc): la free
 * restituisce le pagine al kernel subito, quindi l'RSS scende davvero.
 * Lo slot NON viene compattato via: resta al suo posto con eid=-1/used=0 (primo
 * candidato al riuso), perche' con PILOT_REAL il worker tiene puntatori dentro
 * ecache[] durante i suoi pread e uno spostamento li invaliderebbe; per lo
 * stesso motivo gli slot eid<0 (riservati/in caricamento) non si toccano e la
 * selezione avviene sotto g_pilot_mx. resident_bytes resta invariato: gli slot
 * LRU non sono mai contati li' (solo pin e densa).
 * EN: evict = free the slab in place (eid=-1, used=0, never compact: PILOT_REAL
 * EN: holds pointers into ecache[] across its preads), skip eid<0 reservations,
 * EN: select under g_pilot_mx. RSS_GUARD_GB=<gb> forces an explicit ceiling. */
static double g_ram_budget_gb=0;              /* budget risolto, scritto da cap_for_ram */
static uint64_t g_rssg_last=0;
static void rss_guard(Model *m){
    double lim = getenv("RSS_GUARD_GB") ? atof(getenv("RSS_GUARD_GB")) : g_ram_budget_gb;
    if(lim<=0) return;
    if(m->n_emit - g_rssg_last < 16) return;
    g_rssg_last = m->n_emit;
    double rss=rss_gb();
    if(rss <= lim*1.02+0.3) return;                       /* tolleranza: 2% + 300MB */
    Cfg *c=&m->c;
    int64_t need=(int64_t)((rss-lim)*1e9), freed=0; int dropped=0;
    for(int pass=0; pass<8 && freed<need; pass++){
        for(int l=0; l<=c->n_layers && freed<need; l++){
            if(!m->ecache || !m->ecache[l]) continue;
            pthread_mutex_lock(&g_pilot_mx);
            int nn=m->ecn[l], lru=-1;
            for(int z=0;z<nn;z++){                        /* solo slot pubblicati e con slab */
                ESlot *cand=&m->ecache[l][z];
                if(cand->eid<0 || !cand->slab) continue;
                if(lru<0 || cand->used<m->ecache[l][lru].used) lru=z;
            }
            if(lru<0){ pthread_mutex_unlock(&g_pilot_mx); continue; }
            ESlot *s=&m->ecache[l][lru];
            s->eid=-1;                                    /* nascosto: nessun hit/evict altrui */
            int64_t sb=s->slab_cap + s->fslab_cap*4;
#ifdef COLI_METAL
            if(s->slab && g_metal_enabled) coli_metal_unregister(s->slab);
#endif
            /* La free resta SOTTO g_pilot_mx. Sbloccando prima, lo slot e' visibile come
             * {eid=-1, slab ancora valido}: il pilota (pilot_realload) riusa per primo gli
             * slot eid==-1, quindi puo' prenderlo e fare pread dentro lo slab MENTRE lo
             * liberiamo -> use-after-free / double-free. Slab valido e slot riusabile
             * devono restare mutuamente esclusivi finche' il puntatore non e' NULL. */
            compat_aligned_free(s->slab); free(s->fslab);
            s->slab=NULL; s->fslab=NULL; s->slab_cap=s->fslab_cap=0;
            QT *q[3]={&s->g,&s->u,&s->d};
            for(int k=0;k<3;k++){ q[k]->qf=NULL; q[k]->q8=NULL; q[k]->q4=NULL; q[k]->s=NULL; }
            s->used=0;                                    /* primo candidato al riuso */
            pthread_mutex_unlock(&g_pilot_mx);
            freed += sb; dropped++;
        }
        if(m->ecap>2) m->ecap--;                           /* il tetto scende: niente ricrescita */
    }
    if(dropped)
        fprintf(stderr,"[RAM-GUARD] RSS %.1f GB over the %.1f GB budget (#403): "
                       "dropped %d cached experts, cap -> %d\n", rss, lim, dropped, m->ecap);
}
static void repin_pass_limit(Model *m,int limit){
    rss_guard(m);                     /* #403: il budget si fa rispettare sull'RSS MISURATO */
    if(g_repin<=0) return;
    if(m->n_emit - g_last_repin < (uint64_t)g_repin) return;
    g_last_repin = m->n_emit;
    double pass_t0=now_s(); int gpu_swaps=0;
    RepinCand cd[130];
    if(limit<1) limit=1; if(limit>130) limit=130;
    int nb=repin_pick(m,cd,limit);
#ifdef COLI_CUDA
    /* Cold GPU slots have no host backing. Restore all demoted experts in
     * parallel first; serial 20 MB reads made a 32-slot adaptation pass cost
     * ~0.7 s on the six-GPU host. */
    #pragma omp parallel for schedule(dynamic,1)
    for(int b=0;b<nb;b++) if(cd[b].gpu_swap){
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        expert_host_ensure(m,cd[b].l,s);
    }
    for(int b=0;b<nb;b++) if(cd[b].gpu_swap){
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        m->resident_bytes+=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
    }
#endif
    for(int b=0;b<nb;b++){
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        int old=s->eid;
        uint32_t old_heat=m->eheat[cd[b].l][old], new_heat=m->eheat[cd[b].l][cd[b].eid];
#ifdef COLI_CUDA
        if(cd[b].gpu_swap){
            ESlot *hot=NULL;
            for(int z=0;z<m->npin[cd[b].l];z++)
                if(m->pin[cd[b].l][z].eid==cd[b].eid){hot=&m->pin[cd[b].l][z];break;}
            if(!hot||hot->g.cuda_eligible) continue;
            double t0=now_s();
            QT *cq[3]={&s->g,&s->u,&s->d},*hq[3]={&hot->g,&hot->u,&hot->d};
            int ok=1;
            for(int k=0;k<3;k++){
                hq[k]->cuda=cq[k]->cuda; cq[k]->cuda=NULL;
                hq[k]->cuda_device=cq[k]->cuda_device;
                hq[k]->cuda_eligible=1; cq[k]->cuda_eligible=0;
                if(!qt_cuda_update(hq[k])) ok=0;
            }
            if(!ok){ fprintf(stderr,"[REPIN] refresh VRAM fallito\n"); exit(1); }
            /* promoted expert now computes from VRAM: drop its host mlock
             * (mmap path; no-op otherwise) or every swap leaks locked pages */
            qt_unwire_mmap(&hot->g); qt_unwire_mmap(&hot->u); qt_unwire_mmap(&hot->d);
            if(g_cuda_release_host) expert_host_release(m,hot);
            gpu_swaps++;
            if(getenv("REPIN_VERBOSE")) fprintf(stderr,
                "[REPIN] VRAM layer %d: esce/out %d (heat=%u) <- entra/in %d "
                "(heat=%u) in %.0f ms\n",cd[b].l,old,old_heat,cd[b].eid,new_heat,(now_s()-t0)*1e3);
            continue;
        }
        int gpu=s->g.cuda_eligible;
        int64_t old_gpu=gpu ? (int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                             +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                             +(int64_t)coli_cuda_tensor_bytes(s->d.cuda) : 0;
#endif
        double t0=now_s();
        expert_load(m,cd[b].l,cd[b].eid,s,1,0);     /* disk -> RAM, same resident slot; demand=0: repin, never classified */
        const char *tier="RAM";
#ifdef COLI_CUDA
        if(gpu){                                  /* refresh the same VRAM slot now, not lazily */
            if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                int64_t now_gpu=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                m->gpu_expert_bytes+=now_gpu-old_gpu; tier="VRAM";
                if(g_cuda_release_host) expert_host_release(m,s);
            } else {
                qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                m->gpu_expert_count--; m->gpu_expert_bytes-=old_gpu;
                fprintf(stderr,"[REPIN] VRAM upload failed; slot downgraded to RAM\n");
            }
        }
#endif
        fprintf(stderr,"[REPIN] %s layer %d: evict %d (heat=%u) <- admit %d (heat=%u) in %.0f ms\n",
            tier,cd[b].l,old,old_heat,cd[b].eid,new_heat,(now_s()-t0)*1e3);
    }
    if(gpu_swaps) fprintf(stderr,"[REPIN] VRAM: %d expert scambiati/swapped in %.0f ms\n",
        gpu_swaps,(now_s()-pass_t0)*1e3);
    for(int l=0;l<m->c.n_layers;l++) if(m->eheat[l]) tier_decay(m->eheat[l],m->c.n_experts);
}
/* ---- KV SU DISCO: la conversazione si riapre CALDA (KVSAVE=0 disattiva) ----
 * Il re-prefill di una chat riaperta costa ore su questo disco; la KV compressa MLA
 * costa ~182 KB/token. File <SNAP>/.coli_kv append-only: header (magic + dimensioni +
 * nrec) e un record per posizione [tok i32][Lc+Rc dei 78 layer][Ic DSA]. A fine turno
 * si appendono SOLO le posizioni nuove e si riscrive nrec per ultimo: un crash a meta'
 * append lascia nrec vecchio = file coerente. La riga KV del layer MTP non si salva:
 * al resume kv_start=-1 e la finestra di draft riparte da sola. */

typedef struct { KVState kv; int *hist, len, first; } ServeCtx;
static double kv_pool_bytes(Model *m, int max_ctx);

static void serve_ctx_init(Model *m, ServeCtx *s, const char *snap, int slot, int maxctx){
    s->kv.kv_start=calloc(m->c.n_layers+1,sizeof(int));
    if(m->has_mtp) s->kv.kv_start[m->c.n_layers]=-1;
    kv_bind(m,&s->kv); kv_alloc(m,maxctx);
    s->hist=malloc(maxctx*sizeof(int));
    if(!s->hist){ fprintf(stderr,"OOM serve_ctx_init hist\n"); exit(1); }
    s->first=1;
    if(slot==0) snprintf(s->kv.disk_path,sizeof(s->kv.disk_path),"%s/.coli_kv",snap);
    else snprintf(s->kv.disk_path,sizeof(s->kv.disk_path),"%s/.coli_kv.%d",snap,slot);
    s->len=kv_disk_load(m,s->hist,maxctx); if(s->len>0) s->first=0;
}

static void serve_ctx_free(Model *m, ServeCtx *s){
    KVState *k=&s->kv; int NR=m->c.n_layers+1;
    if(k->disk_fp){ fclose(k->disk_fp); k->disk_fp=NULL; }
    free(k->disk_buf); k->disk_buf=NULL;
    if(k->Lc) for(int i=0;i<NR;i++){ free(k->Lc[i]); free(k->Rc[i]); }
    if(k->Ic) for(int i=0;i<m->c.n_layers;i++) free(k->Ic[i]);
    free(k->Lc); free(k->Rc); free(k->Ic); free(k->kv_start); free(s->hist);
}

typedef struct {
    int active, pending, emitted, maximum, prompt_tokens, length_limited;
    unsigned long long id;
    float temp, top_p;
    double started;
    uint64_t hits0, miss0;
    ProfBase pb;                         /* phase-time window start (same convention as hits0):
                                            feeds the PROF protocol line and the PROF=1 report */
} ServeReq;

static void mux_data(Tok *T, unsigned long long id, int token){
    char out[256]; int n=tok_decode(T,&token,1,out,sizeof(out));
    printf("DATA %llu %d\n",id,n); if(n>0) fwrite(out,1,(size_t)n,stdout); putchar('\n');
    fflush(stdout);
}

static void mux_done(Model *m, ServeCtx *sc, ServeReq *r){
    double dt=now_s()-r->started; if(dt<1e-6) dt=1e-6;
    double dh=(double)(m->hits-r->hits0), dm=(double)(m->miss-r->miss0);
    hwinfo_emit(m);
    usage_save(m);                       /* la cache che impara non deve aspettare l'uscita */
    tiers_emit(m);
    emap_emit(m);
    hits_emit(m);
    /* PROF: per-turn phase timings for the dashboard profiling page —
     * "PROF <wall_s> <prompt> <completion> <edisk> <ewait> <emm> <attn> <head> <n_fw>".
     * edisk = disk service (expert_load wall on the reading threads, overlaps
     * compute); ewait = the stall the compute thread felt — only ewait belongs
     * in a wall-time breakdown. With KV_SLOTS>1 concurrent slots share the
     * batched forwards, so the shares describe the whole engine over the
     * window, not the single request (same convention as the STAT hit% below). */
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n",dt,
           r->prompt_tokens,r->emitted,
           edisk_s()-r->pb.edisk,m->t_ewait-r->pb.ewait,m->t_emm-r->pb.emm,
           m->t_attn-r->pb.attn,m->t_head-r->pb.head,
           (unsigned long long)(m->n_fw-r->pb.n_fw));
    printf("DONE %llu STAT %d %.2f %.1f %.2f %d %d\n",r->id,r->emitted,
           r->emitted/dt,(dh+dm)>0?100.0*dh/(dh+dm):0.0,rss_gb(),
           r->prompt_tokens,r->length_limited);
    fflush(stdout); kv_bind(m,&sc->kv); kv_disk_append(m,sc->hist,sc->len);
    /* PROF window = this request's lifetime; with KV_SLOTS>1 concurrent slots
     * share the batched forwards, so the shares describe the engine, not the
     * single request (same convention as the STAT hit%% above). */
    if(g_prof) prof_report(m,&r->pb,dt,r->emitted,stderr);
    r->active=0;
}

/* Read and prefill one request. Returns -1 on EOF, 0 for a rejected frame and
 * 1 for an accepted request. Prefill deliberately remains serial: continuous
 * batching starts at decode, where every active slot contributes one row. */
static int mux_submit(Model *m, Tok *T, ServeCtx *ctx, ServeReq *req, int nctx,
                      int maxctx, int eos){
    char *line=NULL; size_t cap=0; ssize_t nr=getline(&line,&cap,stdin);
    if(nr<0){ free(line); return -1; }
    if(nr && line[nr-1]=='\n') line[--nr]=0;
    if(!strncmp(line,"CANCEL ",7)){
        unsigned long long id=0; char tail;
        if(sscanf(line+7,"%llu %c",&id,&tail)!=1 || id==0){
            printf("ERROR 0 BAD_REQUEST\n"); fflush(stdout); free(line); return 0;
        }
        for(int i=0;i<nctx;i++) if(req[i].active && req[i].id==id){
            req[i].active=0; kv_bind(m,&ctx[i].kv);
            kv_disk_append(m,ctx[i].hist,ctx[i].len);
            printf("ERROR %llu CANCELLED\n",id); fflush(stdout); free(line); return 0;
        }
        printf("ERROR %llu NOT_FOUND\n",id); fflush(stdout); free(line); return 0;
    }
    ColiSubmit sub; int valid=coli_submit_parse(line,&sub);
    if(!valid){ printf("ERROR 0 BAD_REQUEST\n"); fflush(stdout); free(line); return 0; }
    char *raw=malloc((size_t)sub.bytes+1);
    if(!raw){ fprintf(stderr,"OOM multiplex payload\n"); exit(1); }
    if(fread(raw,1,(size_t)sub.bytes,stdin)!=(size_t)sub.bytes){ free(raw); free(line); return -1; }
    int delim=fgetc(stdin);
    if(delim!='\n'){
        printf("ERROR %llu BAD_FRAME\n",sub.id); fflush(stdout);
        free(raw); free(line); return -1;
    }
    raw[sub.bytes]=0;
    if(sub.slot>=nctx || memchr(raw,0,(size_t)sub.bytes)){
        printf("ERROR %llu BAD_REQUEST\n",sub.id); fflush(stdout); free(raw); free(line); return 0;
    }
    if(req[sub.slot].active){
        printf("ERROR %llu SLOT_BUSY\n",sub.id); fflush(stdout); free(raw); free(line); return 0;
    }
    for(int i=0;i<nctx;i++) if(req[i].active && req[i].id==sub.id){
        printf("ERROR %llu DUPLICATE_ID\n",sub.id); fflush(stdout); free(raw); free(line); return 0;
    }
    ServeCtx *sc=&ctx[sub.slot]; kv_bind(m,&sc->kv);
    int *tmp=malloc(maxctx*sizeof(int));
    if(!tmp){ fprintf(stderr,"OOM mux_submit tmp\n"); free(raw); free(line); exit(1); }
    int nt=tok_encode(T,raw,(int)sub.bytes,tmp,maxctx-2);
    free(raw); free(line);
    if(nt<1){ free(tmp); printf("ERROR %llu EMPTY_PROMPT\n",sub.id); fflush(stdout); return 0; }
    int prefix=0; while(prefix<sc->len && prefix<nt && sc->hist[prefix]==tmp[prefix]) prefix++;
    if(prefix<sc->len){ sc->len=prefix; if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
        kv_disk_truncate(m,sc->len); }
    int add=nt-sc->len;
    if(add>0) memcpy(sc->hist+sc->len,tmp+sc->len,(size_t)add*sizeof(int));
    fprintf(stderr,"[API] KV slot %d prefix %d/%d token, prefill %d\n",sub.slot,sc->len,nt,add);
    free(tmp);
    float *logit = add>0 ? step(m,sc->hist+sc->len,add,sc->len)
                         : step(m,sc->hist+sc->len-1,1,sc->len-1);
    sc->len+=add; sc->first=0;
    ServeReq *r=&req[sub.slot]; memset(r,0,sizeof(*r));
    r->id=sub.id; r->maximum=sub.max_tokens; r->temp=sub.temperature; r->top_p=sub.top_p;
    r->prompt_tokens=nt; r->started=now_s(); r->hits0=m->hits; r->miss0=m->miss;
    prof_base(m,&r->pb);                 /* a few loads: cheap enough to always track */
    int room=maxctx-sc->len-1; if(r->maximum>room){r->maximum=room; r->length_limited=1;}
    g_temp=r->temp; g_nuc=r->top_p;
    int next=pick_tok(logit,m->c.vocab,-1); free(logit);
    if(r->maximum<=0 || next==eos || is_stop(next)){ mux_done(m,sc,r); return 1; }
    r->pending=next; r->emitted=1; r->active=1; sc->hist[sc->len]=next; m->n_emit++;
    mux_data(T,r->id,next);
    if(r->emitted>=r->maximum) mux_done(m,sc,r);
    return 1;
}

static void run_serve_mux(Model *m, const char *snap){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp); int eos=tok_id_of(&T,"<|endoftext|>"); stops_arm_tok(&m->c,eos,&T);
    g_draft=0; /* one scheduler owns every forward; MTP/speculation is not ragged-safe */
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int nctx=getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
    if(nctx<1||nctx>512){fprintf(stderr,"KV_SLOTS must be between 1 and 512\n");exit(2);}
    g_kvsave=getenv("KVSAVE")?atoi(getenv("KVSAVE")):1;
    KVState *initial=m->kv; free(initial->kv_start); free(initial);
    ServeCtx *ctx=calloc(nctx,sizeof(*ctx)); ServeReq *req=calloc(nctx,sizeof(*req));
    for(int i=0;i<nctx;i++) serve_ctx_init(m,&ctx[i],snap,i,maxctx);
#ifdef _WIN32
    /* Same byte-exact protocol as run_serve: in TEXT mode the CRT collapses CRLF in
     * fread() payloads (waits forever for the missing bytes) and expands LF on the
     * way out (corrupting the READY/STAT sentinels). BINARY on both ends. (#195) */
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    setvbuf(stdout, NULL, _IONBF, 0);
#endif
    setvbuf(stdin,NULL,_IONBF,0);
    intr_install();                      /* Ctrl-C = chiudi i turni in volo, non il processo */
    printf("\x01\x01READY\x01\x01\nSTAT 0 0.00 0.0 %.2f\n",rss_gb()); fflush(stdout);
    hwinfo_emit(m);
    tiers_emit(m);
    emap_emit(m);
    int eof=0;
    for(;;){
        if(g_intr){ g_intr=0;            /* stop morbido: ogni request attiva finisce ORA per la
                                          * via normale di mux_done (DONE+stats+KV coerenti) */
            for(int i=0;i<nctx;i++) if(req[i].active) mux_done(m,&ctx[i],&req[i]);
        }
        int active=0; for(int i=0;i<nctx;i++) active+=req[i].active;
        /* Poll stdin for available input without blocking. On POSIX this is
         * select(); on Windows, select() on a pipe handle routes to winsock
         * and always returns -1 (SOCKET_ERROR), so the batch loop could never
         * accept a request (#139). PeekNamedPipe on the stdin OS handle is
         * the Windows equivalent: it reports bytes available without reading. */
        int ready=0;
        if(!eof){
#if defined(__APPLE__) || defined(__linux__) ||	defined(__FreeBSD__)
            fd_set rfds; FD_ZERO(&rfds); FD_SET(STDIN_FILENO,&rfds);
            struct timeval tv={0,0}, *ptv=active?&tv:NULL;
            ready=select(STDIN_FILENO+1,&rfds,NULL,NULL,ptv);
            if(ready>0 && FD_ISSET(STDIN_FILENO,&rfds))
#elif defined(_WIN32)
            HANDLE ih=(HANDLE)_get_osfhandle(_fileno(stdin));
            DWORD avail=0;
            /* Anonymous pipes are NOT waitable objects: WaitForSingleObject on them is
             * undefined (always-signaled or WAIT_FAILED), and PeekNamedPipe fails on
             * file/console handles — the old gate never dispatched (#195). New rule:
             * idle -> block in getline() inside mux_submit (same semantics as the
             * POSIX select(NULL)); active -> poll the pipe with PeekNamedPipe, and on
             * non-pipe stdin just defer submits until the batch finishes. */
            if(eof) ready=0;
            else if(!active) ready=1;
            else ready=(PeekNamedPipe(ih,NULL,0,NULL,&avail,NULL) && avail>0)?1:0;
            if(ready)
#endif
                if(mux_submit(m,&T,ctx,req,nctx,maxctx,eos)<0) eof=1;
        }
        active=0; for(int i=0;i<nctx;i++) active+=req[i].active;
        if(!active){ if(eof) break; continue; }
        DecodeRow rows[512]; int slots[512], S=0;
        for(int i=0;i<nctx;i++) if(req[i].active){
            rows[S]=(DecodeRow){&ctx[i].kv,req[i].pending,ctx[i].len}; slots[S++]=i;
        }
        double tf0=g_prof?now_s():0;
        float *lo=step_decode_batch(m,rows,S); if(!lo){fprintf(stderr,"decode batch failed\n");break;}
        m->n_fw++;
        if(g_prof) prof_lat(now_s()-tf0);
        for(int s=0;s<S;s++){
            int i=slots[s]; ServeCtx *sc=&ctx[i]; ServeReq *r=&req[i];
            sc->len++; g_temp=r->temp; g_nuc=r->top_p;
            int next=pick_tok(lo+(int64_t)s*m->c.vocab,m->c.vocab,-1);
            if(next==eos || is_stop(next)){mux_done(m,sc,r);continue;}
            r->pending=next; sc->hist[sc->len]=next; r->emitted++; m->n_emit++;
            mux_data(&T,r->id,next);
            if(r->emitted>=r->maximum) mux_done(m,sc,r);
        }
        free(lo);
    }
    usage_save(m);
    for(int i=0;i<nctx;i++) serve_ctx_free(m,&ctx[i]); free(ctx); free(req);
    m->kv=NULL; m->Lc=m->Rc=m->Ic=NULL; m->kv_start=NULL; m->max_t=0;
}

static void run_serve(Model *m, const char *snap){
    /* Serve mode speaks a byte protocol over BOTH stdout and stdin:
     *   stdout: \x01\x01READY\x01\x01\n, STAT lines, \x01\x01END\x01\x01\n
     *   stdin:  text lines plus \x02RESET / \x02MORE control bytes.
     * 'coli' matches the sentinels with endswith() and a "^STAT ..." regex,
     * so they must arrive byte-exact (LF, no CR). On Windows the CRT opens
     * both handles in TEXT mode: stdout translates '\n'->'\r\n' (so the READY
     * sentinel never matches and chat hangs at ~10 GB resident), and stdin
     * translates '\r\n'->'\n' and rejects writes of raw bytes with EINVAL,
     * breaking the control protocol. Put BOTH handles in BINARY mode so the
     * protocol bytes are exact in both directions. No-op on Linux/macOS. */
#ifdef _WIN32
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    setvbuf(stdout, NULL, _IONBF, 0);
#endif
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm_tok(&m->c, eos, &T);
    grammar_setup(&T);                   /* metodo F: GRAMMAR=file.gbnf (#48) */
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):256;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int templ=getenv("CHAT_TEMPLATE")?atoi(getenv("CHAT_TEMPLATE")):1;
    g_kvsave = getenv("KVSAVE")?atoi(getenv("KVSAVE")):1;
    int nctx=getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
    if(nctx<1||nctx>16){ fprintf(stderr,"KV_SLOTS must be between 1 and 16\n"); exit(2); }
    KVState *initial=m->kv; free(initial->kv_start); free(initial);
    ServeCtx *ctx=calloc(nctx,sizeof(ServeCtx));
    for(int i=0;i<nctx;i++) serve_ctx_init(m,&ctx[i],snap,i,maxctx);
    int active=0; ServeCtx *sc=&ctx[0]; kv_bind(m,&sc->kv);
    fprintf(stderr,"[KV] context slots: %d x %d tokens, projected pool %.2f GB\n",
        nctx,maxctx,kv_pool_bytes(m,maxctx)/1e9);
    #define hist  (sc->hist)
    #define len   (sc->len)
    #define first (sc->first)
    char *line=NULL; size_t cap=0; ssize_t nr; char *buf=malloc(1<<16);
    intr_install();                      /* Ctrl-C = fine turno, non fine processo */
    printf("\x01\x01" "READY" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout);
    tiers_emit(m);
    while((nr=getline(&line,&cap,stdin))>0){
        g_intr=0;                        /* interruzioni arrivate tra i turni: stantie */
        if(nr>0 && line[nr-1]=='\n') line[--nr]=0;
        if(!strcmp(line,"\x02RESET")){ len=0; first=1; if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
            kv_disk_reset(m);
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        if(!strcmp(line,"\x02MORE")){                /* continua la risposta troncata da NGEN:
            la storia e' gia' in KV, basta ri-forwardare l'ULTIMO token per riavere i logits */
            if(len<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
            int cur=ngen; if(len+cur+g_draft+2>=maxctx) cur=maxctx-len-g_draft-2;
            uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
            ProfBase pb; if(g_prof) prof_base(m,&pb);
            float *logit=step(m,hist+len-1,1,len-1);
            EmitStream es={&T,m,now_s(),0,1};
            int prod=0;
            if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len);
            else free(logit);
            double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
            double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
            printf("\n\x01\x01" "END" "\x01\x01\n");
            printf("STAT %d %.2f %.1f %.2f\n", prod, prod/tdt, (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb());
            fflush(stdout);
            if(g_prof) prof_report(m,&pb,tdt,prod,stderr);   /* per-turn window; stdout is the framed protocol */
            kv_disk_append(m,hist,len); repin_pass(m); continue; }   /* RFC: re-pin a caldo tra i turni / live re-pin between turns */
        if(nr<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        /* API mode: an exact, length-prefixed prompt. Unlike the interactive
         * line protocol this accepts newlines. The tokenized prompt is matched
         * against hist so the common KV prefix survives stateless HTTP turns.
         * Per-request generation controls follow the byte count:
         *   \x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]\n<prompt>\n */
        char *raw=NULL, *input=line;
        int input_n=(int)nr, raw_mode=0, req_ngen=ngen, prompt_tokens=0;
        float base_temp=g_temp, base_nuc=g_nuc;
        if(!strncmp(line,"\x02PROMPT ",8)){
            unsigned long long nb=0; double rt=0, rp=0; int slot=0;
            int nf=sscanf(line+8,"%llu %d %lf %lf %d",&nb,&req_ngen,&rt,&rp,&slot);
            if(nf<4 || nb>(16u<<20) || req_ngen<1 || rt<0 || rt>2 || rp<=0 || rp>1 ||
               slot<0 || slot>=nctx){
                printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;
            }
            active=slot; sc=&ctx[active]; kv_bind(m,&sc->kv);
            raw=malloc((size_t)nb+1); if(!raw){fprintf(stderr,"OOM raw prompt\n");exit(1);}
            if(fread(raw,1,(size_t)nb,stdin)!=(size_t)nb){free(raw);break;}
            int delim=fgetc(stdin); if(delim!='\n' && delim!=EOF) ungetc(delim,stdin);
            if(memchr(raw,0,(size_t)nb)){free(raw); printf("\x01\x01" "END" "\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;}
            raw[nb]=0; input=raw; input_n=(int)nb; raw_mode=1;
            if(req_ngen>ngen) req_ngen=ngen;
            g_temp=(float)rt; g_nuc=(float)rp;
        } else { active=0; sc=&ctx[0]; kv_bind(m,&sc->kv); }
        int bl=0, k=0;                           /* costruisce/tokenizza il turno */
        /* template UFFICIALE GLM-5.2 (chat_template.jinja): niente \n dopo i ruoli, e dopo
         * <|assistant|> serve SEMPRE il blocco think — <think></think> lo DISATTIVA (nothink):
         * col template sbagliato il modello farfuglia e non emette mai lo stop. THINK=1 lo abilita. */
        const char *tk = getenv("THINK")&&atoi(getenv("THINK"))? "<think>" : "<think></think>";
        if(raw_mode){
            int *tmp=malloc(maxctx*sizeof(int)); if(!tmp){fprintf(stderr,"OOM raw tokens\n");exit(1);}
            prompt_tokens=tok_encode(&T,input,input_n,tmp,maxctx-8-g_draft);
            int old_len=len, prefix=0;
            while(prefix<old_len && prefix<prompt_tokens && hist[prefix]==tmp[prefix]) prefix++;
            if(prefix<old_len){
                len=prefix;
                if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
                kv_disk_truncate(m,len);           /* il prossimo append sovrascrive solo la coda */
            }
            k=prompt_tokens-len;
            if(k>0) memcpy(hist+len,tmp+len,k*sizeof(int));
            fprintf(stderr,"[API] KV slot %d prefix %d/%d token, prefill %d\n",
                active,len,prompt_tokens,k);
            free(tmp);
        } else {
            if(templ){ if(first) bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop>");
                       bl+=snprintf(buf+bl,(1<<16)-bl,"<|user|>%s<|assistant|>%s",input,tk); }
            else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",input);
            k=tok_encode(&T,buf,bl,hist+len,maxctx-len); prompt_tokens=k;
            if(len+k+8+g_draft>=maxctx){ len=0; first=1; kv_disk_reset(m);
                bl=0; if(templ){ bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop><|user|>%s<|assistant|>%s",input,tk); }
                else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",input);
                k=tok_encode(&T,buf,bl,hist,maxctx); if(k>maxctx-8-g_draft) k=maxctx-8-g_draft;
                prompt_tokens=k;
            }
        }
        if(prompt_tokens<1){ free(raw); g_temp=base_temp; g_nuc=base_nuc;
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb()); fflush(stdout); continue; }
        first=0;
        int cur=req_ngen; if(len+k+cur+g_draft+2>=maxctx) cur=maxctx-len-k-g_draft-2;
        uint64_t h0=m->hits, ms0=m->miss;
        uint64_t rs0=m->route_slots, rw0=m->route_swaps;
        uint64_t agh0=m->route_agree_hit, agt0=m->route_agree_tot;
        uint64_t kln0=m->route_kl_n; double kls0=m->route_kl_sum;
        double tt0=now_s();
        ProfBase pb; if(g_prof) prof_base(m,&pb);
        float *logit;
        if(k>0){ logit=step(m,hist+len,k,len); len+=k; }
        else logit=step(m,hist+len-1,1,len-1);   /* prompt identico/prefisso: rigenera i logits */
        EmitStream es={&T,m,now_s(),0,1};
        int prod=0;
        grammar_reset();                         /* nuova risposta = nuovo documento (MORE invece continua) */
        if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len);
        else free(logit);
        double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
        double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
        uint64_t rslots=m->route_slots-rs0, rswaps=m->route_swaps-rw0;
        double swap_pct=rslots?100.0*rswaps/rslots:0.0;
        uint64_t ag_hit=m->route_agree_hit-agh0, ag_tot=m->route_agree_tot-agt0;
        uint64_t kl_n=m->route_kl_n-kln0; double kl_sum=m->route_kl_sum-kls0;
        double agree_pct=ag_tot?100.0*ag_hit/ag_tot:100.0;
        double kl_mean=kl_n?kl_sum/(double)kl_n:0.0;
        printf("%s\x01\x01" "END" "\x01\x01\n",raw_mode?"":"\n");
        printf("STAT %d %.2f %.1f %.2f %d %d", prod, prod/tdt,
            (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb(), prompt_tokens, prod>=cur);
        if(g_cache_route || rslots || ag_tot)
            printf(" swap_pct=%.1f route_swaps=%llu route_slots=%llu"
                   " route_agree=%.1f route_kl=%.4f",
                swap_pct,(unsigned long long)rswaps,(unsigned long long)rslots,
                agree_pct,kl_mean);
        printf("\n");
        fflush(stdout);
        if(g_prof) prof_report(m,&pb,tdt,prod,stderr);   /* per-turn window; stdout is the framed protocol */
        free(raw); g_temp=base_temp; g_nuc=base_nuc;
        usage_save(m);                   /* la cache che impara: storia aggiornata a ogni turno */
        kv_disk_append(m,hist,len);      /* KV su disco: il prossimo avvio riparte da qui */
        repin_pass(m);                   /* safe request boundary: adapt session-local hot tier */
    }
    free(line); free(buf);
    usage_save(m);
    #undef hist
    #undef len
    #undef first
    for(int i=0;i<nctx;i++) serve_ctx_free(m,&ctx[i]);
    free(ctx); m->kv=NULL; m->Lc=m->Rc=m->Ic=NULL; m->kv_start=NULL; m->max_t=0;
}

static int *read_arr(jval*o,const char*k,int*n){
    jval*a=json_get(o,k);
    if(!a){ *n=0; return NULL; }
    int*r=malloc(a->len*sizeof(int));
    if(!r){ fprintf(stderr,"OOM read_arr\n"); exit(1); }
    for(int i=0;i<a->len;i++) r[i]=(int)a->kids[i]->num; *n=a->len; return r; }

/* telemetry, stats, usage persistence — moved to telemetry.h */

/* HOT-STORE ("il redis del colibri'"): carica in RAM, UNA VOLTA e per sempre, i top expert
 * per frequenza d'uso misurata (file STATS di un run precedente), entro un budget in GB.
 * Ogni hit evita una lettura dal disco lento. */
/* MLOCK: inchioda in RAM fisica gli expert pinnati cosi' il compressore di memoria di
 * macOS non li comprime/evacua (visto: RSS reale < residente previsto -> "hit" lenti).
 * -1 = auto (ON su macOS dove serve e RLIMIT_MEMLOCK e' permissivo; OFF altrove, dove
 * il limite e' spesso minuscolo e va alzato a mano), 0 = off, 1 = force.
 * EN: MLOCK: wire pinned experts into physical RAM so macOS's memory compressor can't
 * compress/evict them (we saw actual RSS < intended resident -> slow "hits"). -1 = auto
 * (ON on macOS where it matters and RLIMIT_MEMLOCK is permissive; OFF elsewhere, where the
 * limit is often tiny and must be raised by hand), 0 = off, 1 = force. */
static int g_mlock=-1;
static int mem_should_wire(void){
    if(g_mlock>=0) return g_mlock;
#if defined(__APPLE__)
    return 1;                                     /* macOS: default ON */
#else
    return 0;                                     /* Linux/altri: opt-in via MLOCK=1 / opt-in */
#endif
}
/* Inchioda [addr,addr+len) in RAM fisica. No-op fuori da POSIX (Windows ecc.).
 * EN: wire [addr,addr+len) into physical RAM. No-op off POSIX (Windows, etc.). */
static int mem_wire(void *addr, size_t len){
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    return mlock(addr, len);
#elif defined(_WIN32)
    return compat_mlock(addr, len);   /* VirtualLock + working-set growth */
#else
    (void)addr; (void)len; return 0;
#endif
}
/* Inchioda tutti gli slab degli expert pinnati (pesi + scale). Non fatale se fallisce.
 * EN: wire all pinned-expert slabs (weights + scales). Non-fatal on failure. */
/* mlock a single mmap'd QT's weight + scale ranges. Skips VRAM-tier QTs
 * (cuda_eligible): their compute runs from device memory, so wiring the host
 * mmap range would pin ~137 GB of never-touched file pages. NOTE the q8/q4
 * NULL check alone is NOT enough here: expert_host_release() early-returns
 * for mmap experts (no slab) without nulling the host pointers, so GPU-tier
 * slots keep live-looking q8/q4 forever -- that was the bug that wired 363 GB
 * instead of 231 GB and starved the kernel into page-cache thrashing.
 * wired/failed are accumulated into the caller's counters. */
/* undo qt_wire_mmap for one QT: used when a REPIN gpu_swap promotes a wired
 * RAM-tier expert into VRAM -- without this every promotion leaks its locked
 * host range and the dead-weight lock re-grows over a long session. */
static void qt_unwire_mmap(QT *t){
    if(!g_mmap || !mem_should_wire()) return;
    if(!t->q8 && !t->q4) return;
    int64_t scale_b=(int64_t)t->O*4;
    int64_t weight_b=qt_bytes(t)-scale_b;
    void *wp=t->q8?(void*)t->q8:(void*)t->q4;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    if(weight_b>0 && !munlock(wp,(size_t)weight_b)) g_mmap_wired-=weight_b;
    if(t->s && scale_b>0 && !munlock(t->s,(size_t)scale_b)) g_mmap_wired-=scale_b;
#elif defined(_WIN32)
    if(weight_b>0 && !compat_munlock(wp,(size_t)weight_b)) g_mmap_wired-=weight_b;
    if(t->s && scale_b>0 && !compat_munlock(t->s,(size_t)scale_b)) g_mmap_wired-=scale_b;
#endif
}
static void qt_wire_mmap(QT *t, int64_t *wired, long *failed){
    if(!t->q8 && !t->q4) return;
    if(t->cuda_eligible) return;   /* resident in VRAM; host range is dead weight */
    int64_t scale_b=(int64_t)t->O*4;
    int64_t weight_b=qt_bytes(t)-scale_b;
    void *wp=t->q8?(void*)t->q8:(void*)t->q4;
    if(weight_b>0){ if(mem_wire(wp,(size_t)weight_b)==0) *wired+=weight_b; else (*failed)++; }
    if(t->s && scale_b>0){ if(mem_wire(t->s,(size_t)scale_b)==0) *wired+=scale_b; else (*failed)++; }
}
static void pin_wire(Model *m){
    if(!mem_should_wire()) return;
    if(g_mmap){
        /* Wire the FINAL resident set only, after pin_load's GPU-upload pass
         * has already run -- qt_wire_mmap() skips cuda_eligible (VRAM-tier)
         * slots, so only the genuinely RAM-tier experts get locked. */
        Cfg *c=&m->c; double t0=now_s();
        for(int i=0;i<c->n_layers;i++) for(int z=0;z<m->npin[i];z++){
            ESlot *s=&m->pin[i][z];
            qt_wire_mmap(&s->g,&g_mmap_wired,&g_mmap_wire_failed);
            qt_wire_mmap(&s->u,&g_mmap_wired,&g_mmap_wire_failed);
            qt_wire_mmap(&s->d,&g_mmap_wired,&g_mmap_wire_failed);
        }
        fprintf(stderr,"[PIN] mlock (mmap): %.1f GB wired in physical RAM%s in %.0fs\n",
            g_mmap_wired/1e9, g_mmap_wire_failed?" (some allocations failed -- raise: ulimit -l unlimited)":"", now_s()-t0);
        return;
    }
    Cfg *c=&m->c; double t0=now_s(); int64_t wired=0; long failed=0;
    for(int i=0;i<c->n_layers;i++) for(int z=0;z<m->npin[i];z++){
        ESlot *s=&m->pin[i][z];
        if(s->slab){  if(mem_wire(s->slab, s->slab_cap)==0) wired+=s->slab_cap; else failed++; }
        if(s->fslab){ size_t fl=(size_t)s->fslab_cap*sizeof(float);
                      if(mem_wire(s->fslab, fl)==0) wired+=fl; else failed++; }
    }
    if(failed)
        fprintf(stderr,"[PIN] mlock: %.1f GB wired, %ld allocations failed "
            "(raise the limit: ulimit -l unlimited) in %.0fs\n", wired/1e9, failed, now_s()-t0);
    else
        fprintf(stderr,"[PIN] mlock: %.1f GB wired in physical RAM "
            "(no compression) in %.0fs\n", wired/1e9, now_s()-t0);
}

typedef struct { int l,e; uint32_t c; } PinRec;
static int pin_rec_cmp(const void *a,const void *b){
    const PinRec *x=a,*y=b; return x->c<y->c?1:x->c>y->c?-1:0;
}
static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx);  /* def. sotto */
static void pin_load(Model *m, const char *statspath, double gb){
    FILE *f=fopen(statspath,"r"); if(!f){ perror(statspath); return; }
    Cfg *c=&m->c; int cap=(c->n_layers+1)*c->n_experts;
    PinRec *r=malloc((size_t)cap*sizeof(PinRec)); int n=0;
    unsigned char *seen=calloc((size_t)(c->n_layers+1)*c->n_experts,1);
    int l,e; uint32_t cnt;
    while(n<cap && fscanf(f,"%d %d %u",&l,&e,&cnt)==3){
        int ok = l>=0 && e>=0 && e<c->n_experts &&
                 ((l<c->n_layers && m->L[l].sparse) || (l==c->n_layers && m->has_mtp));
        int64_t key=(int64_t)l*c->n_experts+e;
        if(ok&&!seen[key]){ r[n++]=(PinRec){l,e,cnt}; seen[key]=1; }
    }
    fclose(f);
    int fill=getenv("PIN_FILL")?atoi(getenv("PIN_FILL")):0;
#ifdef COLI_CUDA
    if(!getenv("PIN_FILL")&&g_cuda_release_host) fill=1;
#endif
    if(fill) for(int li=0;li<=c->n_layers;li++){
        int sparse=(li<c->n_layers&&m->L[li].sparse)||(li==c->n_layers&&m->has_mtp);
        if(sparse) for(int ei=0;ei<c->n_experts;ei++) if(!seen[(int64_t)li*c->n_experts+ei])
            r[n++]=(PinRec){li,ei,0};
    }
    free(seen);
    qsort(r,(size_t)n,sizeof(*r),pin_rec_cmp);
    int64_t eb=expert_bytes_probe(m,m->ebits);
    /* PIN_GB=all (#80): NON "tutti" alla lettera. Pinnare l'intero set ignora il
     * budget --ram e fa OOM-kill del kernel a meta' generazione (#229: host 92 GB
     * ucciso con --ram 78, anon-rss 89 GB). Clampa a quanti expert entrano nel
     * budget RAM, come AUTOPIN; il pin aggiorna resident_bytes, quindi cap_for_ram
     * dopo restringe la LRU di conseguenza (nessun doppio conteggio). */
    int npin;
    if(gb<0){
        double ram_env=getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
        int est_ctx=getenv("CTX")?atoi(getenv("CTX")):4096;   /* stesso default del call site */
        double avail=expert_avail(m,ram_env,m->ebits,est_ctx);
        npin=avail>0?(int)(avail/eb):0;
    } else npin=(int)(gb*1e9/eb);
#ifdef COLI_CUDA
    /* The VRAM budget must be known BEFORE npin is finalized: with
     * CUDA_RELEASE_HOST the VRAM-ranked prefix's host slabs are freed right
     * after upload, so those slots must NOT consume the RAM pin budget.
     * Before this, a 6-GPU host lost its top ~9k ranked experts from the RAM
     * count and pinned only the leftovers (measured: 9,280 VRAM + 1,721 RAM
     * on a box whose RAM fits ~11k — the cold tail then paid disk forever). */
    double remaining[COLI_CUDA_MAX_DEVICES]={0}, placed_b[COLI_CUDA_MAX_DEVICES]={0};
    int placed_n[COLI_CUDA_MAX_DEVICES]={0}, gpu_prefix=0, prefix_est=0;
    double budget=g_cuda_expert_gb*1e9, safe_total=0;
    if(g_cuda_enabled&&(g_cuda_expert_gb>0||g_cuda_expert_auto)) for(int i=0;i<g_cuda_ndev;i++){
        size_t free_b=0,total_b=0;
        if(coli_cuda_mem_info(g_cuda_devices[i],&free_b,&total_b)){
            remaining[i]=(double)free_b-(double)g_cuda_dense_projected[i]-2e9;
            if(remaining[i]<0) remaining[i]=0; safe_total+=remaining[i];
        }
    }
    if(g_cuda_expert_auto||budget>safe_total) budget=safe_total;
    if(g_cuda_enabled&&g_cuda_release_host&&budget>0){
        prefix_est=(int)(budget/eb)+g_cuda_ndev;
        npin+=prefix_est;                       /* additive: prefix RAM is returned after upload */
    }
#endif
    if(npin>n) npin=n;
    if(npin<1){ free(r); return; }
    int *cnt_l=calloc(c->n_layers+1,sizeof(int));   /* +1: riga MTP */
    for(int a=0;a<npin;a++) cnt_l[r[a].l]++;
    for(int i=0;i<=c->n_layers;i++) if(cnt_l[i]) m->pin[i]=calloc(cnt_l[i],sizeof(ESlot));
    int *slot_of=malloc((size_t)npin*sizeof(int)), *next=calloc(c->n_layers+1,sizeof(int));
    for(int a=0;a<npin;a++) slot_of[a]=next[r[a].l]++;
    for(int i=0;i<=c->n_layers;i++) m->npin[i]=cnt_l[i];
    double t0=now_s();
#ifdef COLI_CUDA
    if(prefix_est>0){ gpu_prefix=prefix_est; if(gpu_prefix>npin) gpu_prefix=npin; }
#else
    int gpu_prefix=0;
#endif
    /* Load the VRAM-ranked prefix first.  Once uploaded its host backing is
     * released before the disjoint RAM-ranked suffix is allocated. */
    #pragma omp parallel for schedule(dynamic,1)
    for(int a=0;a<(gpu_prefix?gpu_prefix:npin);a++)
        expert_load(m,r[a].l,r[a].e,&m->pin[r[a].l][slot_of[a]],1,0);   /* startup pin load; demand=0, never classified */
    m->resident_bytes+=(int64_t)(gpu_prefix?gpu_prefix:npin)*eb;
#ifdef COLI_CUDA
    if(g_cuda_enabled && budget>0){
        int gpu_limit=gpu_prefix?gpu_prefix:npin;
        for(int a=0;a<gpu_limit && m->gpu_expert_bytes<budget;a++){
            int li=r[a].l;
            { ESlot *s=&m->pin[li][slot_of[a]];
                int64_t need=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
                if(m->gpu_expert_bytes+need>budget) break;
                int tried[COLI_CUDA_MAX_DEVICES]={0}, placed=0;
                for(int attempt=0;attempt<g_cuda_ndev && !placed;attempt++){
                    int best=-1;
                    for(int i=0;i<g_cuda_ndev;i++) if(!tried[i] && remaining[i]>=need &&
                        (best<0||placed_b[i]<placed_b[best])) best=i;
                    if(best<0) break;
                    tried[best]=1;
                    s->g.cuda_device=s->u.cuda_device=s->d.cuda_device=g_cuda_devices[best];
                    s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=1;
                    if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                        int64_t actual=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                                      +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                                      +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                        m->gpu_expert_count++; m->gpu_expert_bytes+=actual;
                        remaining[best]-=actual; placed_b[best]+=actual; placed_n[best]++;
                        if(g_cuda_release_host) expert_host_release(m,s);
                        placed=1;
                    } else {
                        qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                        s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                        remaining[best]=0;             /* device rejected its projected capacity */
                    }
                }
            }
        }
        fprintf(stderr,"[CUDA] hot expert tier: %d/%d experts, VRAM %.2f GB (total budget %.1f GB)\n",
            m->gpu_expert_count,npin,m->gpu_expert_bytes/1e9,g_cuda_expert_gb);
        for(int i=0;i<g_cuda_ndev;i++) fprintf(stderr,"[CUDA]   device %d: %d experts, %.2f GB\n",
            g_cuda_devices[i],placed_n[i],placed_b[i]/1e9);
    }
#endif
    if(gpu_prefix>0&&gpu_prefix<npin){
        #pragma omp parallel for schedule(dynamic,1)
        for(int a=gpu_prefix;a<npin;a++)
            expert_load(m,r[a].l,r[a].e,&m->pin[r[a].l][slot_of[a]],1,0);   /* startup pin load; demand=0, never classified */
        m->resident_bytes+=(int64_t)(npin-gpu_prefix)*eb;
    }
    fprintf(stderr,"[PIN] placement: %d VRAM + %d RAM expert (%.1f GB warm) in %.0fs da %s\n",
        m->gpu_expert_count,npin-m->gpu_expert_count,(npin-m->gpu_expert_count)*eb/1e9,now_s()-t0,statspath);
    pin_wire(m);                                   /* inchioda in RAM (no compressione) / wire in RAM (no compression) */
    free(r); free(cnt_l); free(slot_of); free(next);
}

static double g_mem_avail_boot=0;   /* MemAvailable all'avvio, prima di caricare il modello */
/* RAM disponibile ADESSO (GB): e' il tetto vero, non il totale. Linux: MemAvailable
 * da /proc/meminfo. macOS: pagine free+inactive+purgeable da host_statistics64
 * (stessa semantica: recuperabili senza swap). Senza questo ramo il fallback
 * "assumo 8 GB" castrava la cache expert proprio sulle macchine con piu' RAM. */
static double mem_available_gb(void){
#ifdef __APPLE__
    mach_msg_type_number_t cnt=HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm;
    if(host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&cnt)!=KERN_SUCCESS) return 0;
    return ((double)vm.free_count+(double)vm.inactive_count+(double)vm.purgeable_count)
           * (double)sysconf(_SC_PAGESIZE) / 1e9;
#elif defined(_WIN32)
    double total, avail;
    compat_meminfo(&total, &avail);
    return avail;
#else
    FILE *f=fopen("/proc/meminfo","r"); if(!f) return 0;
    char ln[256]; double kb=0;
    while(fgets(ln,sizeof(ln),f)) if(sscanf(ln,"MemAvailable: %lf",&kb)==1) break;
    fclose(f); return kb/1e6;
#endif
}

static int kv_slot_count(void){
    if(!getenv("SERVE")) return 1;
    return getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
}

static double kv_pool_bytes(Model *m, int max_ctx){
    Cfg *c=&m->c; double one=(double)(c->n_layers+1)*max_ctx*(c->kv_lora+c->qk_rope)*4.0;
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        one+=(double)max_ctx*c->index_hd*4.0;
    int slots=kv_slot_count(); if(slots<1||slots>16) slots=1;
    return one*slots;
}

/* byte disponibili per gli expert (pin + LRU) nel budget — specchio del conto di cap_for_ram */
static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int64_t eb=expert_bytes_probe(m,ebits);
    if(ram_gb<=0){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    double ws_b = (g_expert_budget>0 && g_expert_budget<64) ? (double)(g_expert_budget+4)*(double)eb : 64.0*(double)eb;
    double slack = 1.2e9 + 2.5e9 + ws_b
        + kv_pool_bytes(m,max_ctx)
        + (double)max_ctx*c->n_heads*(c->qk_nope+c->v_head)*4.0;
    return ram_gb*1e9 - (double)m->resident_bytes - slack;
}

/* clampa la cache expert a un budget RAM (GB): cap t.c. residente + cache + slack <= budget.
 * ram_gb<=0 -> budget AUTO = 88% della RAM disponibile adesso (lascia respiro a OS+wrapper:
 * sforare = OOM-kill del kernel a meta' generazione, molto peggio di una cache piu' piccola). */
static void cap_for_ram(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    if(m->has_mtp) nsp+=2;                       /* riga cache MTP: conta ~doppia (expert int8 = 2x int4) */
    int64_t eb=expert_bytes_probe(m,ebits);
    int auto_b = ram_gb<=0;
    if(auto_b){ ram_gb = g_mem_avail_boot*0.88;   /* misurata PRIMA del load: il residente gia'
                                                   * allocato viene sottratto sotto, non due volte */
        if(ram_gb<4){ fprintf(stderr,"[RAM] MemAvailable is unreadable or too low; assuming 8 GB\n"); ram_gb=8; } }
    g_ram_budget_gb = ram_gb;                    /* #403: la RSS-guard usa il budget RISOLTO */
    /* slack ONESTO, non forfettario (l'OOM del 2026-07-04 veniva da qui):
     *  ws[64] slab del working-set (si materializzano TUTTI nel prefill batch-union),
     *  KV cache a max_ctx, kvb_all della ricostruzione k/v in attention,
     *  attivazioni+logits+overhead ~1.2 GB */
    double ws_b  = 64.0*(double)eb;
    /* Under EXPERT_BUDGET, the block-of-64 working set is capped at budget experts
     * per layer — only ws[0..budget-1] are populated, not all 64. The 64×eb reserve
     * overcounts by 16x at budget=4, starving the LRU cache (cap 3 instead of 4).
     * Cap=4 matches budget=4, eliminating LRU thrashing that causes excessive disk
     * re-reads. Clamp ws_b to the actual budget (min 8 for non-budgeted / prefill). */
    if(g_expert_budget>0 && g_expert_budget<64) ws_b = (double)(g_expert_budget+4) * (double)eb;
    double kv_b  = kv_pool_bytes(m,max_ctx);
    double kvb_b = (double)max_ctx*c->n_heads*(c->qk_nope+c->v_head)*4.0;
    /* RISERVA PAGE-CACHE (misurato 2026-07-06 su Linux): strangolarla fa crollare
     * le pread buffered da ~800 a ~180 MB/s — gli ultimi GB di LRU rendono MENO di
     * quanto costino in banda disco persa. 2.5 GB restano SEMPRE al kernel.
     * NOTE: tested removing this under Windows+DIRECT (it should be dead weight when
     * O_DIRECT bypasses the buffer cache). Result: cap went 4->5 but RSS hit 24 GB
     * on a 32 GB machine, causing memory pressure that DROPPED the hit rate (73%->57%)
     * and slowed decode (1.03->0.83 tok/s). The reserve is a legitimate safety margin
     * for OS + CUDA + file metadata, not just buffered pread throughput. Keep it. */
    double pc_b  = 2.5e9;
    double slack = 1.2e9 + pc_b + ws_b + kv_b + kvb_b;
    double avail = ram_gb*1e9 - (double)m->resident_bytes - slack;
    int capmax = (avail>0 && nsp>0) ? (int)(avail/((double)nsp*eb)) : 0;
    int floored = capmax<1;   /* il budget non regge nemmeno UNO slot per layer */
    if(capmax<1) capmax=1;
    /* Il floor a 1 e' una bugia comoda: con avail negativo capmax sarebbe 0, cioe'
     * "non ci sto nel tuo budget". Alzarlo a 1 e proseguire trasforma "non ci sto"
     * in "sforo" -- ed e' esattamente l'OOM-kill a meta' generazione che questa
     * funzione esiste per evitare. Il kernel uccide con SIGKILL: nessun errore,
     * nessun log, il motore muore muto (issue #305). Dirlo, e fermarsi se il picco
     * non entra nemmeno nella RAM realmente disponibile misurata all'avvio. */
    if(floored){
        double peak = (double)m->resident_bytes + (double)capmax*nsp*eb + slack;
        fprintf(stderr,"[RAM_GB=%.1f%s] WARNING: cap=1 is the floor, projected peak %.1f GB is "
            "%.1f GB OVER the budget (resident %.1f GB + reserve %.1f GB).%s\n",
            ram_gb,auto_b?" auto":"",peak/1e9,(peak-ram_gb*1e9)/1e9,
            m->resident_bytes/1e9,slack/1e9,
            getenv("PIN_GB")?" PIN_GB is inflating the resident set: lower it or drop it.":"");
        if(g_mem_avail_boot>0 && peak > g_mem_avail_boot*1e9 &&
           !(getenv("COLI_RAM_OVERCOMMIT") && atoi(getenv("COLI_RAM_OVERCOMMIT")))){
            fprintf(stderr,"[RAM] refusing to start: that peak also exceeds the %.1f GB actually "
                "available on this machine, so the kernel would OOM-kill this run mid-generation.\n"
                "[RAM] lower PIN_GB, lower the context, or raise the RAM budget if the box really has it "
                "(COLI_RAM_OVERCOMMIT=1 overrides this check).\n", g_mem_avail_boot);
            exit(2);
        }
    }
    if(capmax < m->ecap){
        fprintf(stderr,"[RAM_GB=%.1f%s] resident %.1f GB + reserve %.1f GB (ws %.1f, KV %dx%d %.1f, kvb %.1f), "
            "experts %.1f MB x %d layers -> cap lowered %d->%d (projected peak %.1f GB)\n",
            ram_gb,auto_b?" auto":"",m->resident_bytes/1e9,slack/1e9,ws_b/1e9,
            kv_slot_count(),max_ctx,kv_b/1e9,kvb_b/1e9,
            eb/1e6, nsp, m->ecap, capmax,
            (m->resident_bytes + (double)capmax*nsp*eb + slack)/1e9);
        m->ecap=capmax;
    } else {
        /* AUTO-RAISE (issue #12): il budget consente PIU' cache di quella chiesta.
         * Senza questo, una macchina da 128 GB girava con la LRU di una da 16
         * (cap=8 di default in coli): hit 23-28% con decine di GB inutilizzati.
         * Tetto a n_experts: oltre, ogni layer avrebbe slot che non puo' riempire.
         * CAP_RAISE=0 ripristina il comportamento fisso. */
        int raise_on = getenv("CAP_RAISE")?atoi(getenv("CAP_RAISE")):1;
        int newcap = capmax>c->n_experts ? c->n_experts : capmax;
        if(raise_on && newcap>m->ecap){
            for(int i=0;i<=c->n_layers;i++) if(m->ecache[i]){
                m->ecache[i]=realloc(m->ecache[i],(size_t)newcap*sizeof(ESlot));
                memset(m->ecache[i]+m->ecap,0,(size_t)(newcap-m->ecap)*sizeof(ESlot));
            }
            fprintf(stderr,"[RAM_GB=%.1f%s] cap raised %d->%d: budget allows it "
                "(projected peak %.1f GB; set CAP_RAISE=0 to disable)\n",
                ram_gb, auto_b?" auto":"", m->ecap, newcap,
                (m->resident_bytes + (double)newcap*nsp*eb + slack)/1e9);
            m->ecap=newcap;
        } else
            fprintf(stderr,"[RAM_GB=%.1f%s] cap=%d ok (projected peak %.1f GB)\n", ram_gb, auto_b?" auto":"", m->ecap,
                (m->resident_bytes + (double)m->ecap*nsp*eb + slack)/1e9);
    }
}

/* The user's generation prompt. COLI_PROMPT is honored on every platform; a bare
 * PROMPT is honored too, EXCEPT on Windows, where cmd.exe always exports its own
 * PROMPT template (default "$P$G", the thing that draws "C:\...>") into the child's
 * environment. That is a shell UI string, not a prompt: taking it would send the
 * engine into text-generation mode (needing a tokenizer) instead of the oracle
 * self-test, and would "generate" from "$P$G". So on Windows a PROMPT carrying
 * cmd's $-metacodes is ignored; set COLI_PROMPT to pass a real prompt from cmd. */
static const char *coli_user_prompt(void){
    const char *p = getenv_utf8("COLI_PROMPT");
    if(p) return p;
    p = getenv_utf8("PROMPT");
#ifdef _WIN32
    if(p) for(const char *q=p; q[0]; q++)
        if(q[0]=='$' && q[1] && strchr("ABCDEFGHLNPQSTV_+|$", q[1]&~0x20)){ p=NULL; break; }
#endif
    return p;
}

/* PROF=1 startup header: one self-describing block so a saved log answers
 * "what machine, what config" when comparing runs after changing RAM_GB,
 * knobs, or moving to another host. */
static void prof_config(Model *m, double ram_env, int est_ctx){
    Cfg *c=&m->c;
    char cpu[256]; int cores; double rt,ra;
    hw_probe(cpu,sizeof(cpu),&cores,&rt,&ra);
    const char *backend="CPU";
#ifdef COLI_CUDA
    if(g_cuda_enabled) backend="CUDA";
#endif
#ifdef COLI_METAL
    if(g_metal_enabled) backend="Metal";
#endif
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    int rows=nsp+(m->has_mtp?2:0);               /* stessa convenzione di cap_for_ram (MTP int8 = 2x) */
    int pinned=0; for(int i=0;i<=c->n_layers;i++) if(m->npin) pinned+=m->npin[i];
    double eb=(double)expert_bytes_probe(m,m->ebits);
    fprintf(stderr,"[PROF] machine: %s | %d cores (%d omp threads) | RAM %.1f GB total, %.1f GB available | backend %s\n",
        cpu[0]?cpu:"unknown CPU",cores,omp_get_max_threads(),rt,ra,backend);
    fprintf(stderr,"[PROF] config: RAM_GB=%s%.1f CTX=%d | expert cache cap %d/layer (up to %.1f GB) | pinned %d (%.1f GB) | "
        "DRAFT=%d PIPE=%d DIRECT=%d MMAP=%d IDOT=%d DSA=%s PILOT=%d CACHE_ROUTE=%d\n",
        ram_env<=0?"auto ":"",ram_env<=0?g_mem_avail_boot*0.88:ram_env,est_ctx,
        m->ecap,(double)m->ecap*rows*eb/1e9,pinned,pinned*eb/1e9,
        g_draft,g_pipe,g_direct,g_mmap,g_idot,
        (m->has_dsa&&c->index_topk)?"on":"off",g_pilot,g_cache_route);
}

int main(int argc, char **argv){
    /* ---- Permanent OpenMP hot-thread tuning. The per-expert matmul regions are
     * tiny and back-to-back; with the default passive wait policy libgomp parks
     * the worker team between regions and the re-wake latency dominates. Keeping
     * the threads hot (active spin) collapses that overhead — measured matmul
     * time 66.9s -> 20.9s on the Zen5 build, with no change to numerical output.
     *
     * libgomp reads the OMP_ / GOMP_ vars in a CONSTRUCTOR that runs before
     * main(), so setenv() here and continuing would be too late (verified:
     * setenv-in-main is ignored by the already-initialised runtime). Instead, on
     * first entry seed the winning defaults — respecting anything the user
     * already set (overwrite=0) — then re-exec self once so a fresh libgomp
     * constructor picks them up. The COLI_OMP_TUNED sentinel guards the exec so
     * we re-exec at most once. Fully overridable: any explicit OMP_/GOMP_ env the
     * user sets wins (overwrite=0), pre-setting COLI_OMP_TUNED=1 skips the
     * re-exec entirely (runs with whatever policy the environment already has),
     * and COLI_NO_OMP_TUNE=1 is a documented kill-switch that disables the whole
     * re-exec + tuning path (distinct from the internal COLI_OMP_TUNED sentinel).
     *
     * Must remain the FIRST statement in main(): argv is passed verbatim to execv(). */
    if(!getenv("COLI_OMP_TUNED") && !getenv("COLI_NO_OMP_TUNE") &&
       !getenv("COLI_CUDA") && !getenv("COLI_METAL")){
        setenv("OMP_WAIT_POLICY","active",0);  /* keep the team hot across the tiny per-expert matmul regions */
        setenv("GOMP_SPINCOUNT","200000",0);   /* spin briefly, then yield so long disk waits don't burn a core */
        /* LLVM libomp (clang builds: FreeBSD cc, macOS, some Linux setups) does not
         * read GOMP_*: with OMP_WAIT_POLICY=active it sets KMP_BLOCKTIME=infinite,
         * so the idle team SPINS FOREVER once generation ends — a serve-mode engine
         * parked on stdin burns ~100% x nthreads (#341, measured 3000% on FreeBSD).
         * 200 ms of blocktime keeps the team hot across back-to-back expert matmuls
         * and lets it sleep at the prompt. libgomp ignores KMP_*; overwrite=0 keeps
         * the user's own setting authoritative. */
        setenv("KMP_BLOCKTIME","200",0);
        setenv("OMP_PROC_BIND","close",0);     /* pack the team onto adjacent cores for cache locality */
        setenv("OMP_DYNAMIC","FALSE",0);       /* fixed team size: no per-region thread-count churn */
        setenv("COLI_OMP_TUNED","1",1);
#ifdef __linux__
        fprintf(stderr,"[OMP] hot-thread tuning: re-exec once (COLI_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/self/exe", argv);         /* returns only on failure -> fall through and run untuned */
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
#ifdef __FreeBSD__
        fprintf(stderr,"[OMP] hot-thread tuning: re-exec once (COLI_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/curproc/file", argv);         /* returns only on failure -> fall through and run untuned */
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
#ifdef _WIN32
    _setmode(fileno(stdout), O_BINARY);
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if(getenv("I4_ACC512")) g_i4_acc512=atoi(getenv("I4_ACC512"))!=0;
    if(getenv("I4_ACC512_TEST")){
        if(!i4_acc512_selftest()) return 1;
        puts("AVX512 i4 selftest: ok"); return 0;
    }
#endif
    const char *snap=getenv("SNAP"); if(!snap){fprintf(stderr,"SNAP=<dir>\n");return 1;}
    g_nopack = getenv("NOPACK")?1:0;
    g_drop = getenv("DROP")?1:0;
    g_prefetch = getenv("PREFETCH")?atoi(getenv("PREFETCH")):0;
    g_mmap = getenv("COLI_MMAP")?atoi(getenv("COLI_MMAP")):0;
    if(g_mmap) fprintf(stderr,"[MMAP] expert = viste zero-copy nei file (page cache = cache)\n");
    numa_init();                                       /* COLI_NUMA=1: expert-slab interleave (#82) */
    g_topk = getenv("TOPK")?atoi(getenv("TOPK")):0;
    g_topp = getenv("TOPP")?atof(getenv("TOPP")):0;
    /* EXPERT_BUDGET e' sotto quarantena: la finestra operativa e' misurata VUOTA.
     * @bokiko su tre host (#303) e riprodotto qui su un 25 GB / WSL:
     *   - hellaswag 30% a budget=8 contro 90% a budget spento (25% = il caso);
     *   - a budget=4 il decode e' rumore ("The **1...: s2151:");
     *   - accettazione MTP 0%: quali expert sopravvivono al cap dipende dalla
     *     residenza in cache al momento del forward, quindi draft e verify NON
     *     calcolano la stessa funzione -- la stessa invariante che #294 ha appena
     *     stabilito, violata via stato di cache invece che via scelta del kernel;
     *   - 0.13 tok/s contro 0.30 di baseline, con 14.66 expert caricati per layer
     *     contro topk=8: il cap fa piu' I/O di quello che dice di risparmiare, e la
     *     riga "~N GB I/O saved" conta esperti scartati, non byte non letti.
     * Resta compilato e sviluppabile (EXPERT_BUDGET_EXPERIMENTAL=1) perche' l'idea
     * -- MoE-Spec, arXiv 2602.16052 -- non e' sbagliata: e' l'implementazione che
     * finora non ha un punto in cui sia insieme piu' veloce e corretta. Riaccenderlo
     * di default richiede una misura di qualita' accanto a quella di velocita'. */
    g_expert_budget = getenv("EXPERT_BUDGET")?atoi(getenv("EXPERT_BUDGET")):0;
    if(g_expert_budget>0 && !getenv("EXPERT_BUDGET_EXPERIMENTAL")){
        fprintf(stderr,"[EXPERT_BUDGET] ignored: measured empty operating window (issue #303).\n"
            "[EXPERT_BUDGET] every tested setting is either no faster or no longer coherent:\n"
            "[EXPERT_BUDGET]   budget=8 -> hellaswag 30%% (90%% with it off) | budget=4 -> decode is noise\n"
            "[EXPERT_BUDGET]   MTP acceptance 0%% (the cap breaks the draft/verify contract, #294)\n"
            "[EXPERT_BUDGET]   0.13 tok/s vs 0.30 baseline, loading 14.7 experts/layer vs topk=8\n"
            "[EXPERT_BUDGET] set EXPERT_BUDGET_EXPERIMENTAL=1 to run it anyway (expect garbage).\n");
        g_expert_budget=0;
    }
    g_cache_route = getenv("CACHE_ROUTE")?atoi(getenv("CACHE_ROUTE")):0;
    g_route_j = getenv("ROUTE_J")?atoi(getenv("ROUTE_J")):2;
    g_route_m = getenv("ROUTE_M")?atoi(getenv("ROUTE_M")):12;
    g_route_p = getenv("ROUTE_P")?atof(getenv("ROUTE_P")):0;
    g_route_alpha = getenv("ROUTE_ALPHA")?atof(getenv("ROUTE_ALPHA")):1.f;
    g_route_agree = getenv("ROUTE_AGREE")?atoi(getenv("ROUTE_AGREE")):0;
    if(g_route_j<0) g_route_j=0;
    if(g_route_m<1) g_route_m=1;
    if(g_route_m>4096) g_route_m=4096;
    if(g_route_alpha<=0.f) g_route_alpha=1.f;
    if(g_route_alpha>1.f) g_route_alpha=1.f;
    if(g_cache_route)
        fprintf(stderr,"[CACHE_ROUTE] on J=%d M=%d P=%.2f alpha=%.2f (pin∪LRU prefer; never default)\n",
                g_route_j,g_route_m,g_route_p,g_route_alpha);
    if(g_route_agree)
        fprintf(stderr,"[ROUTE_AGREE] telemetry on (overlap%% + mean KL vs true top-K)\n");
    /* Auto-enable agree telemetry when CACHE_ROUTE is on (cheap quality leading indicator). */
    if(g_cache_route && !getenv("ROUTE_AGREE")) g_route_agree=1;
    const char *policy=getenv("COLI_POLICY"); if(!policy) policy="quality";
    int experimental=!strcmp(policy,"experimental-fast");
    if(strcmp(policy,"quality")&&strcmp(policy,"balanced")&&!experimental){
        fprintf(stderr,"COLI_POLICY non valida: quality, balanced o experimental-fast\n"); return 2;
    }
    if(!experimental&&(g_topk>0||g_topp>0)){
        fprintf(stderr,"[policy] --topp/--topk drop low-weight experts (~1.6x fewer reads, small quality cost)\n");
    }
    g_mlock  = getenv("MLOCK")?atoi(getenv("MLOCK")):-1;   /* -1 auto (ON macOS), 0 off, 1 force / auto (ON macOS), 0 off, 1 force */
    g_spec = getenv("SPEC")?atoi(getenv("SPEC")):1;
    g_draft = getenv("DRAFT")?atoi(getenv("DRAFT")):-1;
    g_no_fused_pair = getenv("COLI_NO_FUSED_PAIR")?atoi(getenv("COLI_NO_FUSED_PAIR")):0;   /* -1 = auto: 3 se MTP, 0 senza */
    g_looka = getenv("LOOKA")?atoi(getenv("LOOKA")):0;    /* 1 = misura predicibilita' routing */
    g_pilot = getenv("PILOT")?atoi(getenv("PILOT")):0;    /* 1 = prefetch pilotato dal router */
    g_pilot_real = getenv("PILOT_REAL")?atoi(getenv("PILOT_REAL")):0; /* default OFF: load VERI cross-layer (value-preserving prefetch); PILOT_REAL=1 opta in */
    if(g_pilot_real) g_pilot=1;                           /* PILOT_REAL implica il pilota attivo */
    g_pilot_two = getenv("PILOT_TWO")?atoi(getenv("PILOT_TWO")):0; /* 1 = two-step: shared-expert-corrected router prediction (+2.3% recall, 3 extra matmuls) */
    if(g_pilot_two) g_pilot=1;                            /* PILOT_TWO implies PILOT active */
    /* Default K: hint-only PILOT keeps 8 (WILLNEED hints are free, no eviction).
     * Under PILOT_REAL the speculative loads are REAL and create LRU eviction
     * pressure, so at ~28% mispredict a large K thrashes the cache — default to 6
     * (best-measured this session) unless the user set PILOT_K explicitly. */
    g_pilot_k = getenv("PILOT_K")?atoi(getenv("PILOT_K")):(g_pilot_real?6:8);
    if(g_pilot_k<1) g_pilot_k=1;
    g_disk_split = getenv("DISK_SPLIT")?atoi(getenv("DISK_SPLIT")):0; /* 1 = split dei disk load nelle stats */
    g_pipe = getenv("PIPE")?atoi(getenv("PIPE")):
#ifdef _WIN32
        1                        /* default ON: overlap expert load ‖ matmul (byte-identical; reorders I/O). PIPE=0 opts out */
#else
        0
#endif
        ;
    g_pipe_nw = getenv("PIPE_WORKERS")?atoi(getenv("PIPE_WORKERS")):8; /* I/O worker threads */
    if(g_pipe_nw<1) g_pipe_nw=1;
    g_direct = getenv("DIRECT")?atoi(getenv("DIRECT")):0;
    { const char *dh=getenv("COLI_DISKCLASS_WINDOW");        /* DISK-CLASS recency window, see its declaration */
      if(dh){ g_direct_heat_ticks=(uint32_t)strtoul(dh,NULL,10); g_direct_heat_explicit=1; } }
    g_uring = getenv("URING")?atoi(getenv("URING")):0;
    if(g_uring){
#ifdef __linux__
        if(g_mmap){ fprintf(stderr,"URING=1 is incompatible with COLI_MMAP=1\n"); return 2; }
        g_pipe=1;
        if(uring_batch_init(&g_ub_pipe) || (g_pilot_real&&uring_batch_init(&g_ub_pilot))){
            fprintf(stderr,"URING=1: io_uring_setup failed: %s\n",strerror(errno)); return 2;
        }
        unsigned uw=(unsigned)(g_pipe_nw>64?64:g_pipe_nw);
        if(coli_uring_set_workers(&g_ub_pipe.ring,uw) ||
           (g_pilot_real&&coli_uring_set_workers(&g_ub_pilot.ring,uw)))
            fprintf(stderr,"[URING] warning: cannot set io-wq workers=%u: %s\n",uw,strerror(errno));
        fprintf(stderr,"[URING] queued expert I/O active (depth=%d, workers=%u, %s%s)\n",URING_REQ_MAX,uw,
                g_direct?"DIRECT=1":"buffered",g_pilot_real?", batched PILOT_REAL":"");
        if(!g_direct) fprintf(stderr,"[URING] cold NVMe: DIRECT=1 avoids page-cache copy/readahead bottlenecks\n");
#else
        fprintf(stderr,"URING=1 is supported only on Linux\n"); return 2;
#endif
    }
    g_idot = getenv("IDOT")?atoi(getenv("IDOT")):1;        /* 0 = kernel f32 esatti (A/B) */
    g_spec_pin = getenv("SPEC_PIN")?atoi(getenv("SPEC_PIN")):1; /* #163: 0 = gate S-dipendenti storici / legacy S-dependent gates */
    if(getenv("ROUTE_TRACE")&&*getenv("ROUTE_TRACE")){
        g_route_fp=fopen(getenv("ROUTE_TRACE"),"w");
        if(!g_route_fp) fprintf(stderr,"[ROUTE_TRACE] cannot open %s\n",getenv("ROUTE_TRACE"));
        else fprintf(stderr,"[ROUTE_TRACE] logging routing to %s\n",getenv("ROUTE_TRACE"));
    }
    g_repin = getenv("REPIN")?atoi(getenv("REPIN")):0;     /* RFC: re-pin ogni n token emessi (0=off) / live re-pin every n emitted tokens (0=off) */
    g_absorb = getenv("ABSORB")?atoi(getenv("ABSORB")):-1; /* -1 auto: assorbita per S<=4 */
    g_dsa_force = getenv("DSA_FORCE")?atoi(getenv("DSA_FORCE")):0;
    /* matmul_qt documenta la soglia int4-IDOT come "configurabile con I4S" ma il getenv non
     * c'era: la variabile non aveva alcun effetto. I4S=<n> -> IDOT int4 solo per S>=n.
     * EN: matmul_qt documents the int4 IDOT threshold as "configurable via I4S", but the
     * getenv was missing, so the knob did nothing. I4S=<n> -> int4 IDOT only for S>=n. */
    if(getenv("I4S")) g_i4s=atoi(getenv("I4S"));
    g_temp = getenv("TEMP")?atof(getenv("TEMP")):-1;       /* -1 = auto (1.0 chat/testo, greedy altrove) */
    g_nuc  = getenv("NUCLEUS")?atof(getenv("NUCLEUS")):0.90f;  /* piu' stretto dell'ufficiale 0.95: la coda int4 e' rumore */
    if(getenv("SEED")) g_rng = (uint64_t)atoll(getenv("SEED"))*0x9E3779B97F4A7C15ULL+1;
    else { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); g_rng ^= (uint64_t)ts.tv_nsec<<20 ^ (uint64_t)getpid(); }
    if(g_draft>63) g_draft=63;                             /* -1 = auto, risolto dopo model_init */
    int cap  = argc>1?atoi(argv[1]):64;
    int ebits= argc>2?atoi(argv[2]):8;
    int dbits= argc>3?atoi(argv[3]):ebits;
    int kv_limit=(getenv("SERVE_BATCH")&&atoi(getenv("SERVE_BATCH")))?512:16;
    if(getenv("SERVE") && (kv_slot_count()<1 || kv_slot_count()>kv_limit)){
        fprintf(stderr,"KV_SLOTS must be between 1 and %d\n",kv_limit); return 2;
    }
#ifdef COLI_CUDA
    if(getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))){
        const char *one=getenv("COLI_GPU"), *many=getenv("COLI_GPUS");
        if(one&&many){ fprintf(stderr,"use COLI_GPU or COLI_GPUS, not both\n"); return 2; }
        if(many) g_cuda_ndev=parse_cuda_devices(many,g_cuda_devices);
        else if(one) g_cuda_ndev=parse_cuda_devices(one,g_cuda_devices);
        else { g_cuda_ndev=1; g_cuda_devices[0]=0; }
        if(g_cuda_ndev<1){ fprintf(stderr,"invalid COLI_GPUS: use a list such as 0,1,2\n"); return 2; }
        g_cuda_enabled=coli_cuda_init(g_cuda_devices,g_cuda_ndev);
        if(!g_cuda_enabled){ fprintf(stderr,"[CUDA] requested backend is unavailable\n"); return 2; }
    }
    g_cuda_dense=getenv("CUDA_DENSE")?atoi(getenv("CUDA_DENSE")):0;
    g_cuda_pipe=getenv("COLI_CUDA_PIPE")?atoi(getenv("COLI_CUDA_PIPE")):0;
    const char *cuda_expert=getenv("CUDA_EXPERT_GB");
    g_cuda_expert_auto=cuda_expert&&!strcmp(cuda_expert,"auto");
    g_cuda_expert_gb=cuda_expert&&!g_cuda_expert_auto?atof(cuda_expert):0;
    if(!getenv("REPIN")&&g_cuda_expert_auto&&getenv("PIN_GB")&&
       !strcmp(getenv("PIN_GB"),"all")) g_repin=16;
    g_cuda_release_host=getenv("CUDA_RELEASE_HOST")?atoi(getenv("CUDA_RELEASE_HOST")):(g_cuda_ndev>1);
    if((getenv("COLI_GPU")||getenv("COLI_GPUS"))&&!g_cuda_enabled){ fprintf(stderr,"COLI_GPU(S) requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_dense&&!g_cuda_enabled){ fprintf(stderr,"CUDA_DENSE requires COLI_CUDA=1\n"); return 2; }
    if((g_cuda_expert_gb>0||g_cuda_expert_auto) && !g_cuda_enabled){ fprintf(stderr,"CUDA_EXPERT_GB requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_enabled) fprintf(stderr,"[CUDA] mode: routed experts%s%s\n",
        g_cuda_dense?" + resident dense tensors":" only (resident dense on CPU)",
        g_cuda_release_host?"; VRAM experts without host backing":"");
#else
    if((getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))) ||
       getenv("COLI_GPU") || getenv("COLI_GPUS") ||
       (getenv("CUDA_DENSE") && atoi(getenv("CUDA_DENSE"))) ||
        (getenv("CUDA_EXPERT_GB") &&
        (!strcmp(getenv("CUDA_EXPERT_GB"),"auto")||atof(getenv("CUDA_EXPERT_GB"))>0))){
        fprintf(stderr,"CUDA was requested, but this binary is CPU-only; rebuild with: make CUDA=1\n");
        return 2;
    }
#endif
#ifdef COLI_METAL
    if(getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))){
        g_metal_enabled = coli_metal_init();
        if(!g_metal_enabled){ fprintf(stderr,"[METAL] backend requested but not available\n"); return 2; }
        fprintf(stderr,"[METAL] mode: batched routed experts on GPU (unified-memory zero-copy)\n");
        if(getenv("COLI_METAL_SPIN") && atoi(getenv("COLI_METAL_SPIN"))){ coli_metal_spin_start(); fprintf(stderr,"[METAL] keep-alive spinner ON\n"); }
        if(getenv("COLI_METAL_GEMM_MIN")) g_metal_gemm_min=atoi(getenv("COLI_METAL_GEMM_MIN"));
    }
#else
    if(getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))){
        fprintf(stderr,"METAL was requested, but this binary has no Metal backend; rebuild with: make METAL=1\n");
        return 2;
    }
#endif
    printf("== GLM C engine (glm_moe_dsa), cache=%d experts/layer | experts@%d-bit dense@%d-bit | idot: " IDOT_KERNEL " ==\n", cap, ebits, dbits);
    g_mem_avail_boot = mem_available_gb();
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    if(!g_direct_heat_explicit){                     /* COLI_DISKCLASS_WINDOW default, needs m.c (topk/n_layers) */
        /* CURRENT-STATE CALIBRATION: the "8" multiplier (recency window ~= the last 8
         * tokens' worth of routing) is a measured-config constant, not a derived truth.
         * Coordinates: measured 2026-07, macOS 26.5, M5 Max 128 GB, base caa49f7,
         * GLM-5.2 int4 (topk=8, n_layers=78). On that setup it splits the classes
         * cleanly (decode cold share ~71% of classified bytes, 1061.85/1491.49 GB);
         * other models, page-cache pressures, or disks may want a different window --
         * COLI_DISKCLASS_WINDOW overrides, and the DISK-CLASS line itself is the
         * tuning instrument. */
        uint32_t nl=(uint32_t)(m.c.n_layers>0?m.c.n_layers:1), k=(uint32_t)(m.c.topk>0?m.c.topk:1);
        g_direct_heat_ticks = k*nl*8u;               /* ~last 8 tokens' worth of routing, see the declaration */
        if(!g_direct_heat_ticks) g_direct_heat_ticks=1;
    }
    if(g_draft<0){
#ifdef COLI_CUDA
        /* MTP is disabled under CUDA by default: cold (streaming) experts still
         * run on the CPU, where the S==1 fused-pair kernel and the S>=2 IDOT
         * kernel diverge in FP accumulation order, collapsing draft acceptance
         * (#163). GPU-resident experts have no divergence, but the cold subset
         * always exists on a single 16 GB card. COLI_CUDA_MTP=1 opts in for
         * users who want to test speculation under CUDA — the #163 thread shows
         * acceptance can still reach 30-50% even with the cold-expert mismatch.
         * See #292 for the diagnostic sweep that identified this. */
        int cuda_mtp = getenv("COLI_CUDA_MTP") ? atoi(getenv("COLI_CUDA_MTP")) : 0;
        g_draft = (m.has_mtp && (!g_cuda_enabled || cuda_mtp)) ? 3 : 0;
#else
        g_draft = m.has_mtp ? 3 : 0;
#endif
    }
    if(getenv("DSA_TOPK")) m.c.index_topk=atoi(getenv("DSA_TOPK"));   /* override per test */
    /* Il path MUX (SERVE_BATCH=1, cioe' `coli serve`) forza g_draft=0 sotto —
     * la speculazione non e' ragged-safe nel batch multi-slot. Segnalarlo QUI,
     * altrimenti "MTP active (draft=8)" mentirebbe: il messaggio e' stampato
     * prima della scelta del path (run_serve_mux, sotto), e con DRAFT=8 diceva
     * "active" per poi disabilitarlo in silenzio (#358, LordMZTE). */
    int mux_will_disable_mtp = getenv("SERVE") && getenv("SERVE_BATCH") && atoi(getenv("SERVE_BATCH"));
    int eff_draft = mux_will_disable_mtp ? 0 : g_draft;
    printf("loaded in %.2fs | resident dense: %.2f MB | layers=%d experts=%d | MTP %s (draft=%d)\n",
           now_s()-t0, m.resident_bytes/(1024.0*1024.0), m.c.n_layers, m.c.n_experts,
           m.has_mtp?(mux_will_disable_mtp?"DISABLED (multiplexed serve)":"ACTIVE"):"absent", eff_draft);
    /* anche su stderr: e' il canale che le UI (coli) mostrano all'utente */
    if(mux_will_disable_mtp && m.has_mtp)
        fprintf(stderr,"[MTP] disabled in multiplexed serve (SERVE_BATCH=1): speculation is not "
                       "ragged-safe across KV slots. Single-client interactive use (`coli chat`) keeps MTP.\n");
    else
        fprintf(stderr,"[MTP] %s (draft=%d)\n", m.has_mtp?"active: native speculative decoding":"absent", eff_draft);
#ifdef __linux__
    {   /* Only warn for a GENUINE 9p mount (WSL Windows drives, magic 0x01021997), where
         * fadvise is a no-op. The old check was `snap` starting with "/mnt/", which
         * false-positives on native-Linux ZFS/ext4/xfs/NFS mounts that also live under /mnt. */
        struct statfs sfb;
        if(statfs(snap,&sfb)==0 && (unsigned long)sfb.f_type==0x01021997UL)
            fprintf(stderr,"WARNING: the model is on %s (9p/Windows filesystem; fadvise is ineffective).\n"
                           "         Keep it on a native Linux fs (ext4/xfs/zfs) for memory efficiency and speed.\n", snap);
    }
#endif
    /* HOT-STORE: PIN=<statsfile> [PIN_GB=g] -> top expert per frequenza fissi in RAM.
     * Va PRIMA di cap_for_ram: i pinnati contano nel residente. */
    if(getenv("PIN")){
        const char *pin=getenv("PIN"); char pauto[2100];
        if(!strcmp(pin,"auto")){
            /* PIN=auto: la storia VIVA <SNAP>/.coli_usage (appesa a ogni turno) batte il
             * profilo congelato stats.txt — il pin di ogni riavvio riflette il carico reale
             * accumulato, non il prompt di bootstrap. Fallback stats.txt per una dir vergine;
             * nessuno dei due -> nessun pin (AUTOPIN piu' sotto resta escluso: PIN e' settato).
             * EN: prefer the live usage history over the frozen one-shot profile, so each
             * reload's pin placement follows the accumulated real workload. */
            snprintf(pauto,sizeof(pauto),"%s/.coli_usage",snap);
            FILE *pf=fopen(pauto,"rb"); long psz=0;
            if(pf){ fseek(pf,0,SEEK_END); psz=ftell(pf); fclose(pf); }
            if(psz<=0){ snprintf(pauto,sizeof(pauto),"%s/stats.txt",snap);
                pf=fopen(pauto,"rb"); psz=0;
                if(pf){ fseek(pf,0,SEEK_END); psz=ftell(pf); fclose(pf); } }
            if(psz>0){ pin=pauto; fprintf(stderr,"[PIN] auto: seeding from %s\n",pauto); }
            else { pin=NULL; fprintf(stderr,"[PIN] auto: no .coli_usage or stats.txt in %s yet (no pin this run)\n",snap); }
        }
        if(pin){
            const char *pin_gb=getenv("PIN_GB");
            pin_load(&m,pin,pin_gb&&!strcmp(pin_gb,"all")?-1.0:pin_gb?atof(pin_gb):10.0);   /* PIN_GB=all (#80) */
        }
    }
    if(getenv("COUPLE")&&*getenv("COUPLE")){    /* coupling-scored cross-layer prefetch (#176) */
        g_couple_k=getenv("COUPLE_K")?atoi(getenv("COUPLE_K")):8;
        if(g_couple_k<1)g_couple_k=1; if(g_couple_k>32)g_couple_k=32;
        g_couple_d=getenv("COUPLE_D")?atoi(getenv("COUPLE_D")):1;
        if(g_couple_d<1)g_couple_d=1; if(g_couple_d>2)g_couple_d=2;
        couple_load(&m, getenv("COUPLE"));
    }
    /* CACHE CHE IMPARA: l'uso degli expert si accumula in <SNAP>/.coli_usage tra le sessioni;
     * all'avvio i piu' usati vengono auto-pinnati in RAM (meta' del budget expert: il pin
     * conosce la TUA storia, la LRU si adatta alla sessione). AUTOPIN=0 disattiva. */
    { double ram_env = getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx = getenv("CTX")?atoi(getenv("CTX")):4096;   /* stesso default di run_serve */
      snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage",snap);
      int64_t hist = usage_load(&m,g_usage_path);
      if(hist>0) fprintf(stderr,"[USAGE] expert history: %lld selections (%s)\n",(long long)hist,g_usage_path);
      int autopin = getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
      if(!getenv("PIN") && autopin && hist>=5000){
          /* quota pin proporzionale alla FIDUCIA nella storia: con pochi dati il pin
           * sbaglia expert e ruba slot alla LRU adattiva; a regime (>=200k selezioni,
           * qualche ora di chat) arriva a meta' del budget expert. */
          double conf = (double)hist/200000.0; if(conf>1) conf=1;
          double pin_gb = expert_avail(&m,ram_env,ebits,est_ctx)*0.5*conf/1e9;
          if(pin_gb>=0.5) pin_load(&m, g_usage_path, pin_gb);
      }
      /* SEMPRE: senza clamp la LRU cresce fino a cap*76 layer = decine di GB -> OOM-kill.
       * RAM_GB assente o <=0 = budget automatico da MemAvailable. */
      cap_for_ram(&m, ram_env, ebits, est_ctx);
      g_prof = getenv("PROF")?atoi(getenv("PROF")):0;   /* PROF=1: opt-in performance profile */
      if(g_prof) prof_config(&m, ram_env, est_ctx); }
    const char *stats=getenv("STATS");   /* STATS=<file> -> istogramma uso expert a fine run */

    /* modo scoring per benchmark: SCORE=<requests.txt> -> log-likelihood per riga */
    if(getenv("SCORE")){ run_score(&m, snap, getenv("SCORE")); if(stats) stats_dump(&m,stats); return 0; }

    /* modo serve persistente per la CLI 'coli': SERVE=1 */
    if(getenv("SERVE")){
        if(getenv("SERVE_BATCH") && atoi(getenv("SERVE_BATCH"))) run_serve_mux(&m,snap);
        else run_serve(&m,snap);
        if(stats) stats_dump(&m,stats); return 0;
    }

    /* modo testo reale: PROMPT="..." [NGEN=n] -> tokenizza, genera, detokenizza */
    const char *user_prompt = coli_user_prompt();   /* ignores cmd.exe's PROMPT template (#271) */
    if(user_prompt){
        int ngen=getenv("NGEN")?atoi(getenv("NGEN")):64;
        run_text(&m, snap, user_prompt, ngen);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    /* altrimenti: validazione contro l'oracolo (ref_glm.json) */
    const char *refpath=getenv("REF")?getenv("REF"):"ref_glm.json";
    FILE *f=fopen(refpath,"rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); size_t got=fread(b,1,n,f); b[got]=0; fclose(f);
    if((long)got!=n) fprintf(stderr,"warning: short read on %s (%ld of %ld)\n",refpath,(long)got,n);
    char *ar=NULL; jval *ref=json_parse(b,&ar);
    int np=0,nfull=0; int *prompt=read_arr(ref,"prompt_ids",&np); int *full=read_arr(ref,"full_ids",&nfull);
    if(!prompt||!full||np<1||nfull<np){ fprintf(stderr,"ref file missing prompt_ids/full_ids or empty\n"); return 1; }
    int n_new=nfull-np;
    /* L'oracolo (ref_glm.json in repo) e' del modello TINY: contro il 744B da' 0/20
     * garantito su OGNI piattaforma (prompt-token tiny = spazzatura per il modello vero).
     * Non e' un bug del motore — vedi #76. */
    { int maxid=0; for(int i=0;i<nfull;i++) if(full[i]>maxid) maxid=full[i];
      if(m.c.vocab>1000 && maxid<1000 && !getenv("REF_FORCE")){
        fprintf(stderr,
          "ERROR: no PROMPT given, so this is oracle self-test mode — but ref_glm.json is the TINY\n"
          "       model's oracle (max token %d) and your model's vocab is %d. Nothing to validate here.\n"
          "         Engine self-test:  SNAP=./glm_tiny TF=1 ./glm 64 16 16      (expect 32/32)\n"
          "         Real generation:   PROMPT=\"Hello\" NGEN=32 SNAP=<model> ./glm 64\n"
          "         or:                python coli chat --model <model>\n"
          "         REF_FORCE=1 to run the comparison anyway (meaningless).\n"
          "  --- IT ---\n"
          "  Nessun PROMPT: modo auto-validazione, ma ref_glm.json e' l'oracolo del modello TINY\n"
          "  (token max %d, il tuo vocab e' %d). Usa PROMPT=... per generare davvero (vedi sopra).\n",
          maxid, m.c.vocab, maxid, m.c.vocab);
        return 1;
      } }

    if(getenv("REPLAY")){
        run_replay(&m,full,nfull,np);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    if(getenv("TF")){
        int *tf=read_arr(ref,"tf_pred",&(int){0});
        int *pred=malloc(nfull*sizeof(int)); double tt=now_s();
        forward_all(&m, full, nfull, pred); double tdt=now_s()-tt;
        int ok=0; for(int i=0;i<nfull;i++){
            if(pred[i]==tf[i]) ok++;
            else fprintf(stderr,"[ORACLE] mismatch pos=%d expected=%d got=%d\n",i,tf[i],pred[i]);
        }
        printf("PREFILL (teacher-forcing) C vs oracle: %d/%d positions | %.1f pos/s\n",
            ok,nfull,nfull/tdt);
        if(ok<nfull) fprintf(stderr,
            "[ORACLE] %d/%d mismatches — run: TF=1 DEBUG_LOGITS=1 for top-5 logit dump\n",
            nfull-ok,nfull);
        profile_print(&m,tdt);
#ifdef COLI_CUDA
        if(g_cuda_enabled) cuda_stats_print();
#endif
        return 0;
    }
    int *out=malloc((np+n_new)*sizeof(int));
    ProfBase pb; prof_base(&m,&pb);
    double t=now_s(); generate(&m,prompt,np,n_new,out); double dt=now_s()-t;
    int match=0;
    printf("\nReference (oracle): "); for(int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nGLM C engine      : "); for(int i=np;i<nfull;i++){ printf("%d ", out[i]); if(out[i]==full[i])match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double tot=m.hits+m.miss;
    printf("N-gram speculation (DRAFT=%d): %.2f tokens/forward (%llu forwards per %llu tokens)\n",
        g_draft, m.n_fw?(double)m.n_emit/m.n_fw:1.0, (unsigned long long)m.n_fw, (unsigned long long)m.n_emit);
    printf("Expert cache hit rate: %.1f%% (%llu pin + %llu lru / %llu miss) | RSS: %.2f GB | %.1f tok/s\n",
           tot?100.0*m.hits/tot:0.0, (unsigned long long)m.hit_pin, (unsigned long long)m.hit_ecache,
           (unsigned long long)m.miss, rss_gb(), n_new/dt);
    profile_print(&m,dt);
    if(g_prof) prof_report(&m,&pb,dt,n_new,stdout);
#ifdef COLI_CUDA
    if(m.gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m.gpu_expert_count,m.gpu_expert_bytes/1e9,(unsigned long long)m.gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    if(g_looka){
        const char *nm[4]={"previous token (=SPEC prefetch)","layer input, skip attention","next layer (PILOT, stale)","next layer (two-step, shared-expert)"};
        printf("LOOKAHEAD routing — recall of true experts in predicted top-8:\n");
        for(int i=0;i<4;i++) printf("  %-42s %5.1f%%  (%lld/%lld)\n", nm[i],
            la_tot[i]?100.0*la_hit[i]/la_tot[i]:0.0, (long long)la_hit[i], (long long)la_tot[i]);
    }
    if(stats) stats_dump(&m,stats);
    return 0;
}
