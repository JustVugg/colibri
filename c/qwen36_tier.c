/* qwen36_tier.c -- CUDA VRAM expert tier for the qwen36 engine. See header. */
#ifdef COLI_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "qwen36_tier.h"
#include "backend_cuda.h"
#include "tier.h"

#define QT_MAX_DEV 8
/* Max rows in one issued group = max topk. The stride of a device's input
 * replica block, the width of every per-device row array and the topk clamp
 * must all agree: #1339 was three of these disagreeing, each spelling the
 * same literal 32 separately. Name it once. */
#define QT_MAX_ROWS 32
#define QT_QCAP 48            /* upload queue depth (staging ~1.6 MB/entry) */
#define QT_CHG_CAP 512        /* residency changes the chain reads between two of its forwards */
#define QT_PM_CAP 4096        /* a step's misses the chain may promote */
#define QT_CONV_DECAY 1024    /* ticks between halvings of the conversation's counts */

typedef struct {
    ColiCudaTensor *tg, *tu, *td;
    uint32_t heat;
    uint32_t conv;   /* this conversation's routes (qt_chain_count), halved every QT_CONV_DECAY ticks */
    uint8_t resident, queued, planned;
    /* raw RAM pointers (slots are never evicted when cap==n_experts) -- lets
     * warmstart, lookahead and LFRU swaps run without an engine callback */
    const uint8_t *g4,*u4,*d4; const float *gs,*us,*ds;
    /* the host pool (qt_note_host): g4 and gs page-locked for the device's copies */
    uint8_t hreg;
    int aslot;   /* a GGUF's arena slot on its device, plus one (0: none) */
} QSlot;

static struct {
    int on, nl, ne, D, Ih, topk, ndev;
    int egs; size_t sc_gu, sc_d;   /* expert group size + per-matrix scale counts (gs64) */
    /* Formato dei pesi che il tier spedisce in VRAM: 4 = int4 raggruppato
     * (container gs64), 1 = int8 per-riga. Prima era cablato a 4 in ogni punto,
     * e su un container int8 -- dove s->g4 e' NULL perche' non c'e' nulla da
     * impacchettare -- il tier riservava budget, marcava planned=1 e non
     * promuoveva mai niente, senza dire una parola (#1331). backend_cuda sa
     * gia' leggere fmt=1: mancava solo che glielo offrissimo. */
    int wfmt;
    int dev[QT_MAX_DEV];
    size_t budget[QT_MAX_DEV], used[QT_MAX_DEV];
    size_t exp_bytes;                     /* estimated VRAM bytes per expert */
    QSlot *slot;                          /* [nl*ne] */
    pthread_mutex_t mx;
    pthread_t th;
    int th_stop, waiters;
    /* upload ring with staging copies */
    struct { int layer, eid; uint8_t *w; float *s; int v_layer, v_eid; } q[QT_QCAP];
    int qh, qt_, qn;
    int inflight;   /* enqueued and not yet resident. qn frees the ring slot at
                     * dequeue, BEFORE the backend copies anything, so "queue
                     * empty" says nothing about the last expert; qt_fill_wait
                     * needs this separate completion count (#1360). */
    pthread_cond_t cv;
    /* statistics */
    uint64_t hits[QT_MAX_DEV], miss, uploads, q_full_skips;
    /* issue state of the (single) decode thread */
    int is_cnt[QT_MAX_DEV];
    int is_k[QT_MAX_DEV][QT_MAX_ROWS];
    float *is_x; size_t is_x_floats;      /* QT_MAX_ROWS*D input replicas per device */
    /* M3 */
    int *fill_order; int fill_cur;        /* warmstart order (heat desc) */
    uint8_t *off;                         /* layers without experts (qt_layer_off): never planned */
    int issue_open;                       /* guard: no tensor_free while a group is in flight */
    pthread_cond_t cv_take;               /* signals qt_take done + queue space */
    uint64_t tick, swaps, inplace, pf_hits, pf_notes;
    uint32_t *heat0;                      /* heat table loaded from HEAT_FILE */
    /* re-plan (qt_replan): pending (candidate, victim) slot-index pairs, drained on the decode ticks */
    int *rp_c, *rp_v; int rp_n, rp_i; uint64_t rp_planned, rp_done;
    /* evicted-expert ring for qt_evicted_take (slot indices) */
    int *ev; int ev_h, ev_n;
    /* qt_stats_mark: counters at the prefill/decode boundary */
    uint64_t mk_hits, mk_miss; int mk_on;
    /* the dense chain's routed experts (qt_chain_*): residency changes counted, and its
     * forwards in flight, which no swap may touch the tensors of */
    unsigned epoch; int chain_hold;
    /* the chain's promotions (qt_chain_promote): this step's misses, their draw, how many moved and
     * landed; the residency changes since the chain last read them (qt_chain_changes; chg_over:
     * more than the ring holds, or a change the ring does not describe) */
    int *pm; int pm_n, pm_cap; uint32_t rng; uint64_t promos, promo_done;
    int chg[QT_CHG_CAP]; int chg_n, chg_over, chg_on;
} G;
#define QT_EV_CAP 8192

/* Count parked callers so shutdown can reclaim their shared storage safely. */
static void wait_take_locked(void){
    G.waiters++;
    pthread_cond_wait(&G.cv_take,&G.mx);
    if(--G.waiters==0 && G.th_stop) pthread_cond_broadcast(&G.cv_take);
}

static QSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
/* G.mx held: every change of residency goes through here (qt_chain_epoch) */
static void res_set(QSlot *s, int r){
    if(s->resident==r) return;
    s->resident=(uint8_t)r; G.epoch++;
    if(G.chg_on){ if(G.chg_n<QT_CHG_CAP) G.chg[G.chg_n++]=(int)(s-G.slot); else G.chg_over=1; }
}
/* Expert homes. Default: expert eid lives on device index eid % ndev in every
 * layer (each layer's group joins across the cards). QT_HOME=layer: every
 * expert of layer l lives on one device, chosen per layer range, so a token
 * runs each layer on one card and the cards form a pipeline -- no per-layer
 * join, which on unequal cards is what the slower one paced (measured: take
 * 5.2 ms/token on two cards against 0.2 on one). */
static int home2(int layer, int eid);
static void evict_note_locked(int gi);

/* Copy mode (G_stream): the fp8 experts (qt_init_fp8), and any format when the
 * RAM holds fewer than every expert of a layer (cap < n_experts): its slots are
 * recycled then, exactly the Qwen3.8 case, so the tier copies what it takes
 * while the bytes pass by. Before, qwen36 refused the tier outright below full
 * residency, and a 32 GB box with a 4 GB card ran the CPU alone (#2090). */
static int G_stream;
static uint32_t G_cold_heat[QT_MAX_DEV];   /* copy mode: the coldest resident's heat a device (UINT32_MAX: none), each tick */
/* No expert format the tier takes (the mixed layout, int8 with grouped scales,
 * experts=cpu): the devices still take the dense trunk, which is read on every
 * token. Before, each of these switched the trunk off as well. */
static int G_trunk_only;
static int G_chain_own;   /* qt_chain_own: the arena's experts are the dense chain's to move */
/* Where the engine's int4 bytes come from (qt_source_planar): 0 = the
 * container's signed pairs, 1 = expert_ffn.h's planar blocks, the layout the
 * engine keeps in RAM when its shared kernel computes the CPU's experts. */
static int G_src_planar;
/* A GGUF's experts (qt_init_gguf): the file's rows, padded to words on the device, gate and
 * up of ggml type G_gtg[l] and down of G_gtd[l] in layer l (the types change from layer to
 * layer in published files), fmt 100 + type (cuda_bd.cuh); no scale arrays. G.wfmt QT_GGUF. */
#define QT_GGUF 100
static int *G_gtg, *G_gtd, G_gguf_init, G_gguf_copy;   /* G_gguf_copy: the rows qt_note hands over live only for the call */
static size_t *G_gmb, *G_gmbd;          /* a gate (or up) matrix's bytes and a down's, per layer */
/* A GGUF's experts in one arena a device: slots of G_aslot bytes, an expert's gate, up and
 * down rows (padded to words) at 0, G_aoff_u, G_aoff_d, any layer's types in any slot. A
 * cudaMalloc a matrix rounded each up to the allocator's granularity (Qwen3.6 Q4_K's 576 KiB
 * gate took 1 MiB: 3 MiB an expert for 2.0 of bytes, a third fewer experts held), and a swap
 * into another layer's types freed and allocated, synchronising the device; a slot does
 * neither. Off when the backend lacks coli_cuda_tensor_upload_at or the arena is refused. */
static void *G_arena[QT_MAX_DEV];
static int *G_afree[QT_MAX_DEV], G_anfree[QT_MAX_DEV], G_aslots[QT_MAX_DEV], G_arena_on;
static size_t G_aslot, G_aoff_u, G_aoff_d;
static size_t qt_pad256(size_t b){ return (b + 255) / 256 * 256; }
/* Exact slots (qt_gguf_exact_slots, one device): each expert in the room of its own layer's
 * matrices (G_lsl bytes, up and down at G_lou and G_lod), carved from one allocation of the
 * budget as the warm start fills it, a freed slot kept for the next expert of its size. A slot
 * of the largest layer's left Flint UD-Q3_K_XL's average expert 18 % of it unused, UD-Q4_K_XL's
 * 28 % (2.45 MB slots for 1.91 MB): on an RTX PRO 4000 the card held 8140 of its 12032 experts.
 * The budget counts each expert at its slot (qt_ebytes); a swap pairs slots of one size. The
 * qwen38 chain reads the arena as slots of one size (qt_chain_arena): it does not ask for it. */
static int G_exact_want, G_exact;
static size_t *G_lsl, *G_lou, *G_lod, G_lsl_min, G_abump;
static int *G_lcls, G_ncls;                 /* a layer's size class, of G_ncls */
static size_t **G_cfree; static int *G_cnfree, *G_ccap;   /* a class's freed slots (byte offsets) */
void qt_gguf_exact_slots(int on){ G_exact_want = on; }
static size_t qt_ebytes(int layer){ return G_exact ? G_lsl[layer] : G.exp_bytes; }
static size_t qt_ebytes_min(void){ return G_exact ? G_lsl_min : G.exp_bytes; }
/* a slot for an expert of layer l in the exact arena: a freed one of its size, else carved; -1 none */
static long long qt_slot_take(int l){
    int c = G_lcls[l];
    if (G_cnfree[c] > 0) return (long long)G_cfree[c][--G_cnfree[c]];
    if (G_abump + G_lsl[l] > (size_t)G_aslots[0] * 256u) return -1;
    long long o = (long long)G_abump; G_abump += G_lsl[l];
    return o;
}
static void qt_slot_give(int l, size_t off){
    int c = G_lcls[l];
    if (G_cnfree[c] == G_ccap[c]) {
        int nc = G_ccap[c] ? 2 * G_ccap[c] : 64;
        size_t *f = realloc(G_cfree[c], (size_t)nc * sizeof *f);
        if (!f) return;   /* the slot is lost to the arena, not to correctness */
        G_cfree[c] = f; G_ccap[c] = nc;
    }
    G_cfree[c][G_cnfree[c]++] = off;
}
/* A swap writes the newcomer into the victim's device buffers: a GGUF's layers of other types
 * have buffers of other sizes, and freeing them for a new upload synchronises the device,
 * which a dense chain's hybrid layer (the device waiting for the host's word, the host's next
 * launch waiting for the free) turns into a deadlock. So without the arena (whose slots take
 * any layer's types in place) a GGUF's swaps pair same-typed layers. */
static int swap_ok(int cand_layer, int victim_layer){
    if(G_exact) return G_lsl[cand_layer]==G_lsl[victim_layer];   /* the victim's slot holds the newcomer's matrices */
    return G.wfmt!=QT_GGUF || G_arena_on || (G_gtg[cand_layer]==G_gtg[victim_layer] && G_gtd[cand_layer]==G_gtd[victim_layer]);
}
/* bytes of one gate (or up) matrix and of one down matrix as the staging holds them */
static size_t qt_mb_gu(int layer){ return G.wfmt==QT_GGUF ? G_gmb[layer] : (size_t)G.D*G.Ih/(G.wfmt==4?2:1); }
static size_t qt_mb_d(int layer){ return G.wfmt==QT_GGUF ? G_gmbd[layer] : (size_t)G.D*G.Ih/(G.wfmt==4?2:1); }
void qt_source_planar(int on){ G_src_planar = on ? 1 : 0; }

/* One planar block (64 elements in 32 bytes: byte k = element k low, k+32
 * high, both v+8) as the backend's offset-binary pairs (byte j = element 2j
 * low, 2j+1 high, both v+8): the same 32 bytes in pair order, no sign change.
 * noinline: inlined three times into stage(), GCC 13 at -O3 -march=native took
 * over 2 GB and 150 s to compile this file (0.76 s and 68 MB without). */
__attribute__((noinline))
static void planar_to_pairs(uint8_t *dst, const uint8_t *src, size_t bytes){
    size_t b=0;
#if defined(__AVX2__)
    const __m256i m0f=_mm256_set1_epi16(0x000F), mf0=_mm256_set1_epi16(0x00F0);
    for(; b+32<=bytes; b+=32){
        __m256i v=_mm256_loadu_si256((const __m256i*)(src+b)), v4=_mm256_srli_epi16(v,4);
        __m256i lo=_mm256_or_si256(_mm256_and_si256(v,m0f), _mm256_and_si256(v4,mf0));
        __m256i hi=_mm256_or_si256(_mm256_and_si256(v4,m0f), _mm256_and_si256(_mm256_srli_epi16(v,8),mf0));
        _mm256_storeu_si256((__m256i*)(dst+b), _mm256_permute4x64_epi64(_mm256_packus_epi16(lo,hi),0xD8));
    }
#endif
    for(; b<bytes; b+=32){
        const uint8_t *p=src+b; uint8_t *q=dst+b;
        for(int j=0;j<16;j++){
            q[j]    = (uint8_t)((p[2*j]&0x0F) | (p[2*j+1]<<4));
            q[16+j] = (uint8_t)((p[2*j]>>4)   | (p[2*j+1]&0xF0));
        }
    }
}

/* The same 32-byte planar blocks as the device's signed pairs: XOR 0x88 is the
 * offset-to-signed step the upload otherwise runs on the device after its copy
 * (offset_to_signed_s4), done here because the chain copies straight into the
 * victim's buffers. bytes is whole blocks: a matrix of int4 groups of 64. */
__attribute__((noinline))
static void planar_to_pairs_signed(uint8_t *dst, const uint8_t *src, size_t bytes){
    size_t b=0;
#if defined(__AVX2__)
    const __m256i m0f=_mm256_set1_epi16(0x000F), mf0=_mm256_set1_epi16(0x00F0), x88=_mm256_set1_epi8((char)0x88);
    for(; b+32<=bytes; b+=32){
        __m256i v=_mm256_loadu_si256((const __m256i*)(src+b)), v4=_mm256_srli_epi16(v,4);
        __m256i lo=_mm256_or_si256(_mm256_and_si256(v,m0f), _mm256_and_si256(v4,mf0));
        __m256i hi=_mm256_or_si256(_mm256_and_si256(v4,m0f), _mm256_and_si256(_mm256_srli_epi16(v,8),mf0));
        __m256i pk=_mm256_permute4x64_epi64(_mm256_packus_epi16(lo,hi),0xD8);
        _mm256_storeu_si256((__m256i*)(dst+b), _mm256_xor_si256(pk,x88));
    }
#endif
    for(; b<bytes; b+=32){
        const uint8_t *p=src+b;
        uint8_t tmp[32];
        for(int j=0;j<16;j++){
            tmp[j]    = (uint8_t)(((p[2*j]&0x0F) | (p[2*j+1]<<4)) ^ 0x88);
            tmp[16+j] = (uint8_t)(((p[2*j]>>4)   | (p[2*j+1]&0xF0)) ^ 0x88);
        }
        memcpy(dst+b, tmp, 32);
    }
}

/* Staging: packed int4 (g|u|d) two's-complement -> offset-binary (XOR 0x88,
 * the upload format of backend_cuda fmt=2), or planar -> pairs, + copy the
 * scales (gs|us|ds). */
static void stage(uint8_t *dw, float *dsc, int layer,
                  const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
                  const float *gs,const float *us,const float *ds){
    size_t mb = (size_t)G.D*G.Ih/(G.wfmt==4?2:1);
    if(G.wfmt==QT_GGUF){   /* the file's rows as they are: gate | up | down, no scales */
        size_t mg=qt_mb_gu(layer);
        memcpy(dw, g4, mg); memcpy(dw+mg, u4, mg); memcpy(dw+2*mg, d4, qt_mb_d(layer));
        return;
    }
    if(G.wfmt==4 && G_src_planar){
        planar_to_pairs(dw,      g4, mb);
        planar_to_pairs(dw+mb,   u4, mb);
        planar_to_pairs(dw+2*mb, d4, mb);
    } else if(G.wfmt==1 || G.wfmt==8){
        /* int8: il formato del backend e' gia' quello in RAM, si copia e basta.
         * Niente XOR: quello serve a portare i nibble int4 da complemento a due
         * a binario sfalsato, e su byte interi sarebbe corruzione. */
        memcpy(dw,        g4, mb);
        memcpy(dw+mb,     u4, mb);
        memcpy(dw+2*mb,   d4, mb);
    } else {
    const uint64_t X=0x8888888888888888ull;
    const uint64_t *sg=(const uint64_t*)g4,*su=(const uint64_t*)u4,*sd=(const uint64_t*)d4;
    uint64_t *w0=(uint64_t*)dw,*w1=(uint64_t*)(dw+mb),*w2=(uint64_t*)(dw+2*mb);
    for(size_t i=0;i<mb/8;i++){ w0[i]=sg[i]^X; w1[i]=su[i]^X; w2[i]=sd[i]^X; }
    }
    memcpy(dsc,                 gs, G.sc_gu*sizeof(float));
    memcpy(dsc+G.sc_gu,         us, G.sc_gu*sizeof(float));
    memcpy(dsc+2*G.sc_gu,       ds, G.sc_d *sizeof(float));
}

/* Thread affinity around the tier's own threads (Linux).
 *
 * With OMP_PROC_BIND set, libgomp binds the initial thread to place 0 before
 * main() runs, and a pthread inherits the CPU mask of the thread that creates
 * it. The uploader thread and the CUDA runtime's own threads were therefore
 * jailed on the OpenMP master's core: every staging copy and every driver
 * call competed with the master thread's share of each expert matmul, and
 * the whole team waited for it. Measured on Qwen3.8 (12 threads, one card):
 * the CPU time per remaining expert rose 64 % while the tier was on, eating
 * the whole gain of computing 45-59 % of the experts on the GPU. So the tier
 * widens the calling thread's mask to every online CPU while it creates its
 * thread and initializes CUDA, and restores the caller's mask afterwards.
 * Raw syscalls, no _GNU_SOURCE: this file is also #included by tests after
 * the engine's own headers. */
#ifdef __linux__
#define QT_AFF_WORDS 64                              /* 4096 CPUs */
typedef struct { unsigned long w[QT_AFF_WORDS]; int len; } qt_affmask;
static int qt_aff_get(qt_affmask *m){
    long r=syscall(SYS_sched_getaffinity,0,sizeof m->w,m->w);
    if(r<=0) return 0;
    m->len=(int)r; return 1;
}
static void qt_aff_widen(const qt_affmask *saved){
    if(!saved->len) return;
    long n=sysconf(_SC_NPROCESSORS_ONLN);
    if(n<=1) return;
    qt_affmask all; memset(&all,0,sizeof all);
    for(long i=0;i<n && i<(long)(8*sizeof all.w);i++) all.w[i/(8*sizeof(unsigned long))] |= 1ul<<(i%(8*sizeof(unsigned long)));
    syscall(SYS_sched_setaffinity,0,(size_t)saved->len,all.w);
}
static void qt_aff_restore(const qt_affmask *saved){
    if(saved->len) syscall(SYS_sched_setaffinity,0,(size_t)saved->len,saved->w);
}
static int qt_aff_count_self(void){
    qt_affmask m; if(!qt_aff_get(&m)) return 0;
    int c=0; for(int i=0;i<m.len/(int)sizeof(unsigned long);i++) c+=__builtin_popcountl(m.w[i]);
    return c;
}
#else
typedef struct { int len; } qt_affmask;
static int  qt_aff_get(qt_affmask *m){ m->len=0; return 0; }
static void qt_aff_widen(const qt_affmask *m){ (void)m; }
static void qt_aff_restore(const qt_affmask *m){ (void)m; }
static int  qt_aff_count_self(void){ return 0; }
#endif
static int G_uploader_cpus;   /* CPUs the uploader thread may run on (0 = unknown) */

static void *uploader(void *arg){
    (void)arg;
    G_uploader_cpus=qt_aff_count_self();
    for(;;){
        pthread_mutex_lock(&G.mx);
        while(G.qn==0 && !G.th_stop) pthread_cond_wait(&G.cv,&G.mx);
        if(G.th_stop && G.qn==0){ pthread_mutex_unlock(&G.mx); return NULL; }
        int layer=G.q[G.qh].layer, eid=G.q[G.qh].eid;
        int vl=G.q[G.qh].v_layer, ve=G.q[G.qh].v_eid;
        uint8_t *w=G.q[G.qh].w; float *sc=G.q[G.qh].s;
        G.qh=(G.qh+1)%QT_QCAP; G.qn--;
        pthread_cond_broadcast(&G.cv_take);          /* queue space available */
        /* staging layout: three matrices of mb bytes each; int4 packed = half a
         * byte per element, int8/fp8 = one. Scales follow the format. */
        size_t mb=qt_mb_gu(layer);
        const float *sg=sc, *su, *sd;
        if(G.wfmt==8||G.egs){ su=sc+G.sc_gu; sd=sc+2*G.sc_gu; } else { su=sc+G.Ih; sd=sc+2*G.Ih; }
        int dv = G.dev[home2(layer,eid)];
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok=0, inplace=0, aslot=-1;   /* a GGUF's arena slot: the victim's, or a free one */
        if(G_arena_on){
            if(ve>=0){ QSlot *v=qs(vl,ve); aslot=v->aslot-1; v->aslot=0; }
            else if(G_exact){ long long o=qt_slot_take(layer); aslot = o<0 ? -1 : (int)(o/256); }   /* in 256-byte units */
            else { int hi=home2(layer,eid); if(G_anfree[hi]>0) aslot=G_afree[hi][--G_anfree[hi]]; }
        }
        if(ve>=0){
            /* LFRU swap: touch the victim only when no group and no chain forward is in flight */
            while((G.issue_open||G.chain_hold) && !G.th_stop) wait_take_locked();
            QSlot *v=qs(vl,ve);
            if(G.th_stop && (G.issue_open||G.chain_hold)){
                /* Shutting down with a group still open: qt_take() -- the only
                 * thing that clears issue_open -- will never come. Abandon
                 * this swap instead of freeing a victim tensor the in-flight
                 * group may still reference; qt_lfru_tick_locked already
                 * cleared the victim's resident flag before enqueueing, so
                 * restore it to keep the flag consistent with the tensor it
                 * still holds. */
                res_set(v,1); qs(layer,eid)->queued=0; G.inflight--;
                if(aslot>=0) v->aslot=aslot+1;   /* the victim keeps its slot */
                pthread_cond_broadcast(&G.cv_take);
                pthread_mutex_unlock(&G.mx); free(w); free(sc); continue;
            }
            ColiCudaTensor *a=v->tg,*b=v->tu,*ct=v->td;
            v->tg=v->tu=v->td=NULL;
            pthread_mutex_unlock(&G.mx);
            /* Every expert has the same geometry: write the newcomer into the
             * victim's device buffers instead of cudaFree + cudaMalloc, each
             * of which synchronises the device under the async groups (a
             * re-plan of 1,500 swaps cost 1.3 s of prefill that way). A
             * backend without the entry point, or a failure half-way, falls
             * back to the old free-then-upload. */
            int same=G.wfmt!=QT_GGUF || (G_gtg[vl]==G_gtg[layer] && G_gtd[vl]==G_gtd[layer]);   /* a GGUF's layers differ: the arena re-uploads */
            if(a&&b&&ct && same && coli_cuda_tensor_overwrite(a,w,sg) && coli_cuda_tensor_overwrite(b,w+mb,su)
                        && coli_cuda_tensor_overwrite(ct,w+2*mb,sd)){
                tg=a; tu=b; td=ct; ok=1; inplace=1;
            } else {
                if(a)coli_cuda_tensor_free(a); if(b)coli_cuda_tensor_free(b); if(ct)coli_cuda_tensor_free(ct);
            }
        } else pthread_mutex_unlock(&G.mx);

        if(!inplace){
            if(G.wfmt==QT_GGUF && G_arena_on){
                const size_t ou = G_exact ? G_lou[layer] : G_aoff_u, od = G_exact ? G_lod[layer] : G_aoff_d,
                             sl = G_exact ? G_lsl[layer] : G_aslot;
                char *b=(char *)G_arena[home2(layer,eid)] + (size_t)(aslot<0?0:aslot)*(G_exact ? 256u : G_aslot);
                ok = aslot>=0
                  && coli_cuda_tensor_upload_at(&tg, b,      ou,      w,      100+G_gtg[layer], G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload_at(&tu, b + ou, od - ou, w+mb,   100+G_gtg[layer], G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload_at(&td, b + od, sl - od, w+2*mb, 100+G_gtd[layer], G.Ih, G.D,  dv);
            } else if(G.wfmt==QT_GGUF){
                ok = coli_cuda_tensor_upload(&tg, w,      NULL, 100+G_gtg[layer], G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload(&tu, w+mb,   NULL, 100+G_gtg[layer], G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload(&td, w+2*mb, NULL, 100+G_gtd[layer], G.Ih, G.D,  dv);
            } else if(G.wfmt==8){
                /* e4m3 bytes as they came from the checkpoint, block scales
                 * [ceil(O/128), ceil(I/128)] per matrix -- the layout #817's
                 * kernels and tensor_upload(fmt=8) already agree on */
                ok = coli_cuda_tensor_upload(&tg, w,      sg, 8, G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload(&tu, w+mb,   su, 8, G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload(&td, w+2*mb, sd, 8, G.Ih, G.D,  dv);
            } else if(G.wfmt==1){
                /* int8, scale per riga: qt_init ha gia' rifiutato il caso raggruppato,
                 * che questo formato non sa esprimere. */
                ok = coli_cuda_tensor_upload(&tg, w,      sg, 1, G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload(&tu, w+mb,   su, 1, G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload(&td, w+2*mb, sd, 1, G.Ih, G.D,  dv);
            } else if(G.egs){
                ok = coli_cuda_tensor_upload_g(&tg, w,      sg, 4, G.D,  G.Ih, dv, G.egs)
                  && coli_cuda_tensor_upload_g(&tu, w+mb,   su, 4, G.D,  G.Ih, dv, G.egs)
                  && coli_cuda_tensor_upload_g(&td, w+2*mb, sd, 4, G.Ih, G.D,  dv, G.egs);
            } else {
                ok = coli_cuda_tensor_upload(&tg, w,      sg, 2, G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload(&tu, w+mb,   su, 2, G.D,  G.Ih, dv)
                  && coli_cuda_tensor_upload(&td, w+2*mb, sd, 2, G.Ih, G.D,  dv);
            }
        }
        free(w); free(sc);
        if(!ok){
            if(tg) coli_cuda_tensor_free(tg);
            if(tu) coli_cuda_tensor_free(tu);
            if(td) coli_cuda_tensor_free(td);
        }
        pthread_mutex_lock(&G.mx);
        QSlot *s=qs(layer,eid);
        if(G_arena_on && aslot>=0){
            if(ok) s->aslot=aslot+1;
            else if(G_exact) qt_slot_give(layer, (size_t)aslot*256u);
            else { int hi=home2(layer,eid); G_afree[hi][G_anfree[hi]++]=aslot; }   /* the slot back */
        }
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; res_set(s,1); G.uploads++; if(inplace) G.inplace++; }
        else  { int hd=home2(layer,eid); G.used[hd]-=qt_ebytes(layer);
                G.budget[hd]=G.used[hd];   /* device genuinely full: stop trying */ }
        s->queued=0;
        G.inflight--;
        pthread_cond_broadcast(&G.cv_take);          /* this upload is complete */
        pthread_mutex_unlock(&G.mx);
    }
}

/* R4 role split: lm_head as a resident int8 tensor on its own device.
 * The dense-i8 quantization (engine-side) provides q/sc with the same
 * per-row semantics quant_matmul's fmt=1 applies (y[o] = acc * sc[o]),
 * so CPU and GPU compute the same numbers up to accumulation order. */
static struct { ColiCudaTensor *t; int dev, dev_ok, on, fmt; } G_lmh;   /* fmt: 1 int8 rows, 100 + t a GGUF's blocks */

/* ---- placement table (COLI_PLACE) --------------------------------------- */
/* Parsed lazily on first query and cached: qt_place_of runs per layer during
 * init and must not re-parse the environment 30 times. */
#define QT_PLACE_MAX 8
#define QT_SPLIT_MAX 8
static struct {
    char name[16];
    struct { int dev, count; } seg[QT_SPLIT_MAX];   /* count<=0: all remaining */
    int nseg;
} G_place[QT_PLACE_MAX];
static int G_place_n = 0, G_place_done = 0;

/* one component spec: "cpu" | "<dev>" | "<dev>:<n>+<dev>:<n>..." */
#define QT_PLACE_ALL (-2)   /* only meaningful for "experts" */
static void place_add(const char *name, size_t nlen, const char *spec){
    if(G_place_n >= QT_PLACE_MAX) return;
    if(nlen >= sizeof(G_place[0].name)) nlen = sizeof(G_place[0].name)-1;
    memcpy(G_place[G_place_n].name, name, nlen);
    G_place[G_place_n].name[nlen] = 0;
    int ns = 0;
    for(const char *p = spec; *p && ns < QT_SPLIT_MAX; ){
        while(*p==' ') p++;
        int dev, count = -1;
        if(!strncmp(p,"cpu",3)){ dev = QT_PLACE_CPU; p += 3; }
        else if(!strncmp(p,"all",3)){ dev = QT_PLACE_ALL; p += 3; }   /* experts=all: every COLI_GPUS card, reserved or not */
        else { dev = atoi(p); while(*p && *p!=':' && *p!='+') p++; }
        if(*p==':'){ count = atoi(p+1); p++; while(*p && *p!='+') p++; }
        G_place[G_place_n].seg[ns].dev = dev;
        G_place[G_place_n].seg[ns].count = count;
        ns++;
        if(*p=='+') p++; else break;
    }
    G_place[G_place_n].nseg = ns;
    G_place_n++;
}

static void place_parse(void){
    G_place_done = 1;
    const char *e = getenv("COLI_PLACE");
    if(!e || !*e) return;
    const char *p = e;
    while(*p){
        while(*p==' '||*p==','||*p==';') p++;
        const char *name = p;
        while(*p && *p!='=' && *p!=',' && *p!=';') p++;
        if(*p!='='){ while(*p && *p!=','&&*p!=';') p++; continue; }
        size_t nlen = (size_t)(p - name);
        p++;                                   /* past '=' */
        char spec[64]; size_t si = 0;
        while(*p && *p!=',' && *p!=';' && si < sizeof(spec)-1) spec[si++] = *p++;
        spec[si] = 0;
        place_add(name, nlen, spec);
    }
    fprintf(stderr,"[place] COLI_PLACE=%s\n", e);
}

/* Was this component named at all? Distinguishes "experts=cpu" (an explicit
 * request to disable the tier) from "not mentioned" (keep today's default). */
static int qt_place_named(const char *component){
    if(!G_place_done) place_parse();
    for(int i=0;i<G_place_n;i++) if(!strcmp(G_place[i].name,component)) return 1;
    return 0;
}

/* ---- DeltaNet input projections ----------------------------------------- */
/* One fused qkv++z tensor per DeltaNet layer. Indexed by model layer index,
 * so the array is n_layers wide and the attention slots stay empty. */
#define QT_DN_MAX_LAYERS 128
static struct { ColiCudaTensor *t; int dev, on, fmt; } G_dnp[QT_DN_MAX_LAYERS];
static int G_home_layer;                          /* QT_HOME=layer */
static uint8_t G_layer_dev[QT_DN_MAX_LAYERS];     /* device INDEX per layer in that mode */
static int home(int eid){ return eid % G.ndev; }   /* the expert-index mapping (tests read it too) */
static int home2(int layer, int eid){ return G_home_layer ? G_layer_dev[layer] : home(eid); }

/* ---- automatic placement (COLI_PLACE unset or "auto") ------------------ */
/* The hand-written list above is a measurement tool. Nobody running a 6 GB
 * card should have to work out that 1.2 GB of dense trunk is worth more than
 * 800 experts (#1040); the engine knows every size involved and decides.
 *
 * Rule: bytes saved on the memory bus per token, per byte of VRAM spent.
 *   dense component  -> read on EVERY token: value 1.0 per byte.
 *   routed expert    -> read with the probability p_e that a token routes to
 *                       it (heat share when a HEAT_FILE exists, topk/n_experts
 *                       otherwise), and the CPU fallback reads the int8 slot,
 *                       which is twice the bytes an int4 expert occupies in
 *                       VRAM: value 2*p_e per byte (1*p_e on an int8 container).
 * A trunk item goes to the device with the most room if its value beats the
 * value of the coldest experts it would push out of that device -- the
 * experts at the tail of the heat order that still fit today. Without heat
 * that tail is worth 2*topk/n_experts per byte (0.06 on the 35B) and the
 * trunk always wins; with heat, a card whose marginal expert is routed on
 * more than every second token keeps its experts. That is the R4
 * measurement: on two near-full 8 GB cards, moving all projections onto one
 * card cost 0.7 GB of hot experts and lost 13 ms of savings again.
 *
 * The engine OFFERS the trunk before qt_init (qt_trunk_offer: component,
 * layer, bytes -- sizes only, pointers come later as before); the decision
 * lands in the same table qt_place_of() reads, so nothing downstream
 * changes. Placed bytes are subtracted from that device's expert budget,
 * which the hand-written list never did (the 0.7 GB above was the
 * discovery). COLI_PLACE=off keeps today's behaviour: nothing placed. */
#define QT_OFFER_MAX 1024
static struct { char name[16]; int layer; size_t bytes; int dev; } G_offer[QT_OFFER_MAX];
static int G_offer_n;
static int G_auto_on;                                  /* auto placement decided */
static int G_auto_lmh = QT_PLACE_CPU;
static int G_auto_dnp[QT_DN_MAX_LAYERS];               /* per layer, or QT_PLACE_CPU */
static size_t G_trunk_bytes[QT_MAX_DEV];               /* placed trunk per device index */

int qt_place_of(const char *component, int layer){
    if(G_auto_on){
        /* the decision lives on the offer: any component name an engine
         * offered can be asked for (lmhead/dnproj for qwen36, the trunk
         * matrices of qwen38 by their own names) */
        for(int o = 0; o < G_offer_n; o++)
            if(G_offer[o].layer == layer && !strcmp(G_offer[o].name, component)) return G_offer[o].dev;
        return QT_PLACE_CPU;               /* experts follow COLI_GPUS */
    }
    if(!G_place_done) place_parse();
    for(int i = 0; i < G_place_n; i++){
        if(strcmp(G_place[i].name, component)) continue;
        int seen = 0;
        for(int s = 0; s < G_place[i].nseg; s++){
            int cnt = G_place[i].seg[s].count;
            if(cnt <= 0) return G_place[i].seg[s].dev;      /* rest of the layers */
            if(layer < seen + cnt) return G_place[i].seg[s].dev;
            seen += cnt;
        }
        return QT_PLACE_CPU;             /* past the last segment: stay on CPU */
    }
    return QT_PLACE_CPU;
}


void qt_trunk_offer(const char *component, int layer, size_t bytes){
    if(!component || !bytes || G_offer_n >= QT_OFFER_MAX) return;
    if(layer < 0 || layer >= QT_DN_MAX_LAYERS) return;
    snprintf(G_offer[G_offer_n].name, sizeof G_offer[0].name, "%s", component);
    G_offer[G_offer_n].layer = layer; G_offer[G_offer_n].bytes = bytes; G_offer[G_offer_n].dev = QT_PLACE_CPU;
    G_offer_n++;
}

static int auto_mode(void){
    const char *e = getenv("COLI_PLACE");
    return !e || !*e || !strcmp(e, "auto");
}

/* p_e of the k coldest experts that still fit on device index di, summed as
 * bytes-per-token they save; heat from the HEAT_FILE table when present. */
static int cmp_double_desc(const void *a, const void *b){
    double x=*(const double*)a, y=*(const double*)b; return x<y ? 1 : x>y ? -1 : 0;
}
static int auto_dnp_count(int dev){
    int c = 0; for(int l = 0; l < QT_DN_MAX_LAYERS; l++) if(G_auto_dnp[l] == dev) c++; return c;
}

static double auto_displaced_value(int di, size_t room, int k, size_t exp_bytes,
                                   int nl, int ne, int topk, const uint32_t *heat0,
                                   double *p_marginal_out){
    size_t homed = 0;
    for(int l = 0; l < nl; l++) for(int e = 0; e < ne; e++) if(home2(l, e) == di) homed++;
    size_t fit = room / exp_bytes;
    if(fit >= homed || G_trunk_only){ *p_marginal_out = 0; return 0; }   /* room to spare: displaces nothing */
    if(k <= 0){ *p_marginal_out = 0; return 0; }
    /* the CPU reads the int8 slot, or the int4 one when the engine keeps them planar */
    double cpu_factor = (G.wfmt == 1 || G_src_planar) ? 1.0 : 2.0;
    if(!heat0){
        double p = (double)topk / ne;
        *p_marginal_out = p;
        return (double)k * cpu_factor * p * (double)exp_bytes;
    }
    /* heat share per expert on this device, sorted descending; the marginal
     * ones sit at ranks [fit-k, fit) */
    size_t n = homed; double *p = malloc(n * sizeof *p); size_t m = 0;
    for(int l = 0; l < nl; l++){
        double sum = 0;
        for(int e = 0; e < ne; e++) sum += (double)heat0[(size_t)l*ne + e];
        for(int e = 0; e < ne; e++){
            if(home2(l, e) != di) continue;
            double pe = sum > 0 ? (double)topk * heat0[(size_t)l*ne + e] / sum : (double)topk / ne;
            p[m++] = pe > 1.0 ? 1.0 : pe;
        }
    }
    qsort(p, m, sizeof *p, cmp_double_desc);
    double value = 0, pm = 0;
    for(size_t r = (fit > (size_t)k ? fit - k : 0); r < fit && r < m; r++){ value += cpu_factor * p[r] * (double)exp_bytes; pm = p[r]; }
    free(p);
    *p_marginal_out = pm;
    return value;
}

/* Whether the trunk placement is the automatic one (COLI_PLACE unset or
 * "auto"): the only decision the engine's startup probe may overturn. A
 * hand-written list, or "off", is the user's word and stands. */
int qt_place_is_auto(void){ return G_auto_on && auto_mode(); }

/* Undo the automatic trunk placement: every offer back to the CPU, lm_head and
 * the DeltaNet projections included, and the bytes it had taken back into each
 * device's expert budget. The engine calls this BEFORE any trunk upload, when
 * its startup probe measured the GPU GEMV slower than the CPU's: on four Tesla
 * M10 every placed component lost, lm_head 68.8 ms against 41.7 on the CPU,
 * decode 2.68 against 3.56 tok/s (#1652). Nothing to undo when the placement
 * was not automatic. */
void qt_trunk_withdraw(const char *why){
    if(!qt_place_is_auto()) return;
    size_t back = 0;
    for(int o = 0; o < G_offer_n; o++) G_offer[o].dev = QT_PLACE_CPU;
    G_auto_lmh = QT_PLACE_CPU; G_lmh.dev_ok = 0;
    for(int l = 0; l < QT_DN_MAX_LAYERS; l++) G_auto_dnp[l] = QT_PLACE_CPU;
    for(int i = 0; i < G.ndev; i++){ back += G_trunk_bytes[i]; G.budget[i] += G_trunk_bytes[i]; G_trunk_bytes[i] = 0; }
    fprintf(stderr,"[place] trunk stays on the CPU (%s): %.2f GB of VRAM back to the experts\n",
            why ? why : "withdrawn", back/1073741824.0);
}

/* The experts follow the trunk back to the CPU (#2211). The probe that withdrew the trunk
 * times the same int8 GEMV the tier runs for a resident expert, a round trip over PCIe
 * included, and the tier's share of a layer waits for it: on a GTX 1650 Max-Q (0.900 ms
 * against 0.306 on the CPU) the tier with a tenth of the experts resident lost decode
 * (9.2 against 10.9 tok/s) and prefill (14 against 46 tok/s). Only an automatic allowance
 * is withdrawn: CUDA_EXPERT_GB as a number is the user's word. The engine calls this
 * before its warm start, so nothing is resident yet; the tier stays up for nothing else
 * and every expert call answers "not here". */
/* qt_reserve: device bytes the engine places after the tier sized itself (qwen36_tier.h) */
static size_t G_reserve;
void qt_reserve(size_t bytes){ G_reserve = bytes; }
/* qt_reserve_short: bytes kept out of a device's expert budget only when it holds fewer than
 * `experts` of them (a chain's staging of the experts the RAM keeps, which with every expert on
 * the card would hold memory for nothing) */
static size_t G_reserve_short, G_short_experts;
void qt_reserve_short(size_t bytes, size_t experts){ G_reserve_short = bytes; G_short_experts = experts; }
void qt_experts_withdraw(const char *why){
    const char *bg = getenv("CUDA_EXPERT_GB");
    if(!G.on || G_trunk_only || (bg && strcmp(bg,"auto") && atof(bg) > 0)) return;
    G_trunk_only = 1;
    for(int i = 0; i < G.ndev; i++) G.budget[i] = 0;
    for(int i = 0; i < QT_MAX_DEV; i++){   /* a GGUF's arena: nothing is resident yet (before the warm start) */
        if(G_arena[i]) coli_cuda_pipe_free(G.dev[i], G_arena[i]);
        G_arena[i] = NULL; free(G_afree[i]); G_afree[i] = NULL; G_aslots[i] = G_anfree[i] = 0;
    }
    G_arena_on = 0;
    fprintf(stderr,"[qtier] the experts stay on the CPU too (%s; CUDA_EXPERT_GB=<GB> keeps them on the device)\n",
            why ? why : "withdrawn");
}

static void auto_place(int nl, int ne, int topk, const size_t *capacity, const uint32_t *heat0){
    size_t room[QT_MAX_DEV];
    for(int i = 0; i < G.ndev; i++){ room[i] = capacity[i]; G_trunk_bytes[i] = 0; }
    for(int l = 0; l < QT_DN_MAX_LAYERS; l++) G_auto_dnp[l] = QT_PLACE_CPU;
    G_auto_lmh = QT_PLACE_CPU;
    int placed = 0, kept = 0;
    /* lmhead first (one call per token, latency-tolerant), then the
     * projections in layer order */
    for(int o = 0; o < G_offer_n; o++) G_offer[o].dev = QT_PLACE_CPU;
    /* lmhead first (one call per token, latency-tolerant), then every other
     * offered component in offer order -- the engine offers in layer order,
     * so a partial placement is a prefix of the layers (#1361: whole
     * layers, never a slice of the stream) */
    for(int pass = 0; pass < 2; pass++)
        for(int o = 0; o < G_offer_n; o++){
            int is_lmh = !strcmp(G_offer[o].name, "lmhead");
            if((pass == 0) != is_lmh) continue;
            size_t bytes = G_offer[o].bytes;
            int di = 0;
            for(int i = 1; i < G.ndev; i++) if(room[i] > room[di]) di = i;
            /* pipeline homes: a layer's trunk goes where its experts live, lm_head
             * to the last layer's card -- the token then changes card once */
            if(G_home_layer) di = is_lmh ? G_layer_dev[nl-1] : G_layer_dev[G_offer[o].layer < QT_DN_MAX_LAYERS ? G_offer[o].layer : nl-1];
            /* a layer's out_proj goes where its in_proj went when that card has
             * the room: with both on one device the whole DeltaNet layer can run
             * there (qt_dn_gpu_init), which is worth more than balancing bytes */
            if(!strcmp(G_offer[o].name, "dnout") && G_offer[o].layer < QT_DN_MAX_LAYERS && G_auto_dnp[G_offer[o].layer] != QT_PLACE_CPU)
                for(int i = 0; i < G.ndev; i++) if(G.dev[i] == G_auto_dnp[G_offer[o].layer] && room[i] >= bytes) di = i;
            if(room[di] < bytes){ kept++; continue; }
            int k = (int)((bytes + G.exp_bytes - 1) / G.exp_bytes);
            double pm = 0;
            double lose = auto_displaced_value(di, room[di], k, G.exp_bytes, nl, ne, topk, heat0, &pm);
            if((double)bytes < lose){
                fprintf(stderr,"[place] auto: %s layer %d stays on CPU -- %.1f MB would displace %d experts "
                               "worth %.1f MB/token on dev %d (p_marginal %.3f)\n",
                        G_offer[o].name, G_offer[o].layer, bytes/1048576.0, k, lose/1048576.0, G.dev[di], pm);
                kept++; continue;
            }
            G_offer[o].dev = G.dev[di];
            if(is_lmh) G_auto_lmh = G.dev[di];
            else if(!strcmp(G_offer[o].name, "dnproj")) G_auto_dnp[G_offer[o].layer] = G.dev[di];
            room[di] -= bytes; G_trunk_bytes[di] += bytes; placed++;
        }
    G_auto_on = 1;
    for(int i = 0; i < G.ndev; i++)
        fprintf(stderr,"[place] auto: dev %d holds %.1f MB of trunk (lmhead%s, %d dnproj layers), %.2f GB left for experts\n",
                G.dev[i], G_trunk_bytes[i]/1048576.0, G_auto_lmh == G.dev[i] ? " yes" : " no",
                auto_dnp_count(G.dev[i]), room[i]/1073741824.0);
    (void)placed; (void)kept;
}


/* ---- copy ("streaming") mode: fp8 (Qwen3.8), and qwen36 below full residency --
 * With every expert in RAM (cap == n_experts) qwen36 lets the tier retain raw
 * pointers into slots that are never recycled. A model whose experts do not
 * fit in RAM (Qwen3.8: 24 576 x 4.7 MiB; Qwen3.6 on a 32 GB box) streams them
 * through an LRU whose slots ARE recycled, so a retained pointer would dangle
 * by the next token (G_stream). In this mode the tier owns what it uploads: the
 * bytes are copied into the staging buffer inside the qt_note call, while the
 * engine's slot is still live, and the pointers are dropped right after. A
 * promotion can therefore only happen when the bytes pass by -- the LFRU
 * decision moves from the periodic tick into qt_note, which asks: is this
 * expert, now in hand, hotter than the coldest resident on its device? */
static int G_fp8_stream;
static const float *G_fp8_lut;
static int G_upload_sync;             /* QT_UPLOAD_SYNC=1: qt_issue waits for in-flight uploads first (tests) */

int qt_init_fp8(int nl, int ne, int D, int Ih, int cap, int topk, const float *e4m3_lut){
    if(G.on) return 0;
    G_fp8_stream = 1; G_fp8_lut = e4m3_lut;
    int ok = qt_init(nl, ne, D, Ih, cap, topk, 0, 0);
    if(!ok) G_fp8_stream = 0;
    return ok;
}

/* each device's arena: as many slots as its budget holds and its homed experts need */
static int qt_arena_up(void){
    if(G_exact && G.ndev != 1) G_exact = 0;
    if(G_exact){   /* one allocation of the budget (or of every homed expert's slot, if less), carved as it fills */
        size_t all=0;
        for(int l=0;l<G.nl;l++) all += (size_t)G.ne*G_lsl[l];
        size_t bytes = G.budget[0] < all ? G.budget[0] : all;
        bytes = bytes/256*256;
        G_aslots[0]=(int)(bytes/256); G_anfree[0]=0; G_abump=0;
        if(bytes && !(G_arena[0]=coli_cuda_pipe_alloc(G.dev[0], bytes))){ G_aslots[0]=0; G_exact=0; return 0; }
        fprintf(stderr,"[qtier] dev %d: the GGUF experts' arena, %.2f GB in slots of their layers' sizes (%d sizes, %.2f to %.2f MB; "
                       "the largest layer's for every expert would hold %.0f%% fewer)\n", G.dev[0], bytes/1073741824.0, G_ncls,
                G_lsl_min/1048576.0, G_aslot/1048576.0, 100.0*(1.0 - (double)all/G.ne/G.nl/(double)G_aslot));
        G_arena_on=1;
        return 1;
    }
    for(int i=0;i<G.ndev;i++){
        size_t homed=0;
        for(int l=0;l<G.nl;l++) for(int e=0;e<G.ne;e++) if(home2(l,e)==i) homed++;
        size_t n=G.budget[i]/G_aslot; if(n>homed) n=homed;
        G_aslots[i]=(int)n; G_anfree[i]=0;
        if(!n) continue;
        G_arena[i]=coli_cuda_pipe_alloc(G.dev[i], n*G_aslot);
        G_afree[i]=malloc(n*sizeof(int));
        if(!G_arena[i] || !G_afree[i]){
            for(int j=0;j<=i;j++){ if(G_arena[j]) coli_cuda_pipe_free(G.dev[j],G_arena[j]); G_arena[j]=NULL; free(G_afree[j]); G_afree[j]=NULL; G_aslots[j]=G_anfree[j]=0; }
            return 0;
        }
        for(size_t k=0;k<n;k++) G_afree[i][k]=(int)(n-1-k);
        G_anfree[i]=(int)n;
        fprintf(stderr,"[qtier] dev %d: the GGUF experts' arena, %zu slots of %.2f MB (%.2f GB)\n",
                G.dev[i], n, G_aslot/1048576.0, n*G_aslot/1073741824.0);
    }
    G_arena_on=1;
    return 1;
}
int qt_init_gguf(int nl, int ne, int D, int Ih, int cap, int topk, const int *tg, const int *td,
                 const size_t *gu_bytes, const size_t *d_bytes, int copy){
    if(G.on || nl<1 || !tg || !td || !gu_bytes || !d_bytes) return 0;
    G_gguf_copy = copy;
    free(G_gtg); free(G_gtd); free(G_gmb); free(G_gmbd);
    G_gtg=malloc((size_t)nl*sizeof(int)); G_gtd=malloc((size_t)nl*sizeof(int));
    G_gmb=malloc((size_t)nl*sizeof(size_t)); G_gmbd=malloc((size_t)nl*sizeof(size_t));
    if(!G_gtg||!G_gtd||!G_gmb||!G_gmbd) return 0;
    for(int l=0;l<nl;l++){
        int bg=coli_cuda_block_elems(100+tg[l]), bd=coli_cuda_block_elems(100+td[l]);
        if(bg<=0 || bd<=0 || D%bg || Ih%bd) return 0;
        G_gtg[l]=tg[l]; G_gtd[l]=td[l];
        G_gmb[l]=gu_bytes[l]; G_gmbd[l]=d_bytes[l];
    }
    G_gguf_init=1;
    int ok=qt_init(nl,ne,D,Ih,cap,topk,0,1);
    G_gguf_init=0;
    return ok;
}

/* VRAM an allocation of `bytes` really occupies (cudaMalloc granularity,
 * see the exp_bytes comment in qt_init). */
static size_t dev_alloc_footprint(size_t bytes){
    /* measured with cudaMemGetInfo over 256 allocations each (driver 5xx):
     *   400 B, 3 KiB, 4 KiB -> 8 KiB      10 KiB -> 16 KiB     16..64 KiB -> exact
     *   96 KiB -> 104 KiB   384 KiB -> 416 KiB   768 KiB -> 1 MiB   1 MiB -> 1 MiB
     *   1.5 MiB -> 2 MiB    3 MiB -> 4 MiB
     * i.e. above 1 MiB multiples of 2 MiB, above 512 KiB one 1 MiB page, and
     * below that roughly the size plus a sixteenth, in 8 KiB steps, 8 KiB
     * minimum. The small-size rule is a fit, slightly conservative. */
    const size_t KiB = 1024u, MiB = 1048576u;
    if(bytes > MiB) return (bytes + 2*MiB - 1) / (2*MiB) * (2*MiB);
    if(bytes > 512*KiB) return MiB;
    size_t b = bytes + bytes/16;
    if(b < 8*KiB) b = 8*KiB;
    return (b + 8*KiB - 1) / (8*KiB) * (8*KiB);
}

int qt_init(int nl, int ne, int D, int Ih, int cap, int topk, int expert_gs,
            int expert_is_int4){
    if(G.on) return 0;
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(topk>QT_MAX_ROWS){ fprintf(stderr,"[qtier] topk>%d unsupported\n",QT_MAX_ROWS); return 0; }
    /* cap > n_experts holds every expert too (a slot per expert, never recycled):
     * it used to be refused along with the smaller caps */
    G_stream = G_fp8_stream || cap < ne || (G_gguf_init && G_gguf_copy);
    G_trunk_only = expert_is_int4 < 0;
    if(G_stream && !G_fp8_stream && !G_trunk_only)
        fprintf(stderr,"[qtier] RAM holds %d of %d experts a layer: the tier copies each expert it takes as it passes\n",cap,ne);
    memset(&G,0,sizeof G);
    { const char *e=getenv("QT_UPLOAD_SYNC"); G_upload_sync=e&&*e&&*e!='0'; }
    G.nl=nl; G.ne=ne; G.D=D; G.Ih=Ih; G.topk=topk;
    /* Placement state is re-derived per init: the device fold-in below reads
     * COLI_PLACE before the automatic placement has decided anything, and a
     * parse latched from an earlier init (tests start the tier many times)
     * would otherwise stand in for the current environment. */
    G_place_done = 0; G_place_n = 0; G_auto_on = 0;

    /* devices: COLI_GPUS="0,1" (default: first two visible devices).
     * COLI_GPU is the singular the planner writes for a one-device plan
     * (resource_plan.py) and colibri.c reads; accept it here as well, or a
     * `coli chat --gpu 1` lands on every visible device. */
    const char *gl=getenv("COLI_GPUS");
    if(!gl || !*gl) gl=getenv("COLI_GPU");
    if (gl && *gl) {
        char buf[128]; snprintf(buf,sizeof buf,"%s",gl);
        for(char *t=strtok(buf,","); t && G.ndev<QT_MAX_DEV; t=strtok(NULL,","))
            G.dev[G.ndev++]=atoi(t);
    } else {
        int available=coli_cuda_available_device_count();
        int want=available<2?available:2;
        for(int i=0;i<want && i<QT_MAX_DEV;i++) G.dev[G.ndev++]=i;
        fprintf(stderr,"[qtier] COLI_GPUS unset: selecting %d visible device(s)\n",G.ndev);
    }
    /* A device named only in COLI_PLACE still needs a CUDA context before
     * anything can be uploaded to it. Fold those in here rather than making
     * the caller repeat every device in COLI_GPUS as well -- forgetting that
     * would silently drop a component back to the CPU mid-A/B. */
    {
        static const char *comps[] = {"lmhead","dnproj","dnout","attnproj","shexp"};
        for(size_t ci=0; ci<sizeof comps/sizeof *comps; ci++)
            for(int l=0; l<nl && G.ndev<QT_MAX_DEV; l++){
                int d=qt_place_of(comps[ci],l);
                if(d==QT_PLACE_CPU) continue;
                int seen=0; for(int i=0;i<G.ndev;i++) if(G.dev[i]==d) seen=1;
                if(!seen){ G.dev[G.ndev++]=d;
                    fprintf(stderr,"[place] dev %d aus COLI_PLACE zur CUDA-Init ergaenzt\n",d); }
            }
    }
    if(G.ndev<1){ fprintf(stderr,"[qtier] no visible CUDA devices -> CPU path\n"); return 0; }
    qt_affmask aff; qt_aff_get(&aff); qt_aff_widen(&aff);   /* CUDA's threads are born here */
    int cuda_ok_=coli_cuda_init(G.dev,G.ndev);
    qt_aff_restore(&aff);
    if(!cuda_ok_){ fprintf(stderr,"[qtier] coli_cuda_init failed -> CPU path\n"); return 0; }
    int have=coli_cuda_device_count();
    if(have<G.ndev){ G.ndev=have; }
    if(G.ndev<1){ fprintf(stderr,"[qtier] no CUDA devices -> CPU path\n"); return 0; }
    if(G_fp8_stream){
        if(!G_fp8_lut || !coli_cuda_fp8_set_lut(G_fp8_lut)){
            fprintf(stderr,"[qtier] fmt=8 decode table not published -> CPU path\n");
            return 0;
        }
    }

    /* Weight format and bytes per expert come first now: the automatic
     * placement below needs them to price the experts a trunk item displaces. */
    G.wfmt = G_gguf_init ? QT_GGUF : G_fp8_stream ? 8 : (expert_is_int4 > 0 ? 4 : 1);
    if(G.wfmt==1 && expert_gs>0 && !G_trunk_only){
        fprintf(stderr,"[qtier] int8 experts with grouped scales (gs=%d) cannot be "
                       "expressed on the GPU (fmt=1 is per-row only)\n", expert_gs);
        G_trunk_only = 1;
    }
    if(G.wfmt!=4) G_src_planar = 0;   /* planar is an int4 layout */
    G.egs = expert_gs;
    if(G.wfmt==QT_GGUF){ G.sc_gu = G.sc_d = 0; }   /* the scales are in the blocks */
    else if(G.wfmt==8){
        /* one f32 scale per 128x128 block of [O,I]: gate/up are [Ih,D], down is
         * [D,Ih] -- the same count either way, kept as two fields for symmetry */
        size_t nbD=(size_t)(D+127)/128, nbI=(size_t)(Ih+127)/128;
        G.sc_gu = nbI*nbD; G.sc_d = nbD*nbI;
    } else {
        G.sc_gu = expert_gs ? (size_t)Ih * ((D + expert_gs - 1)/expert_gs) : (size_t)Ih;
        G.sc_d  = expert_gs ? (size_t)D  * ((Ih + expert_gs - 1)/expert_gs) : (size_t)D;
    }
    /* Charge what the device allocator takes, not what the bytes measure:
     * cudaMalloc rounds an allocation above 1 MiB up to a multiple of 2 MiB,
     * one above 512 KiB up to 1 MiB, and small ones to 8 KiB steps
     * (dev_alloc_footprint has the measured table).
     * An expert is three weight allocations plus three scale allocations.
     * Charged by payload, the fp8 Qwen3.8 expert (3 x 1.56 MiB) looked like
     * 4.69 MiB and took 6.03 MiB: the budget filled the card to the last
     * megabyte and the uploader ran into "tensor allocation: out of memory"
     * before its stop-trying fallback shrank the budget. Now the planned count
     * is the resident count. The 22-28 % the granularity costs is real; only
     * pooling experts into one arena per device would win it back (open). */
    size_t mat_bytes = G.wfmt==4 ? (size_t)D*Ih/2 : (size_t)D*Ih;
    G.exp_bytes = 3*dev_alloc_footprint(mat_bytes)
                + 2*dev_alloc_footprint(G.sc_gu*sizeof(float))
                + dev_alloc_footprint(G.sc_d*sizeof(float));
    if(G.wfmt==QT_GGUF){   /* the biggest layer's (a slot takes any layer's expert); padded rows, no scales */
        G.exp_bytes = 0; size_t mg=0, md=0;
        for(int l=0;l<nl;l++){
            size_t pg=(G_gmb[l]+3*(size_t)Ih)/4*4, pd=(G_gmbd[l]+3*(size_t)D)/4*4;
            size_t e = 2*dev_alloc_footprint(pg) + dev_alloc_footprint(pd);
            if(e>G.exp_bytes) G.exp_bytes=e;
            if(pg>mg) mg=pg; if(pd>md) md=pd;
        }
        /* the arena's slot (qt_arena_up below): the matrices back to back, 256-byte aligned */
        G_aoff_u=qt_pad256(mg); G_aoff_d=2*G_aoff_u; G_aslot=G_aoff_d+qt_pad256(md);
        G_arena_on=0;
        G_exact=0;
        if(G_exact_want){   /* each layer's own slot, its sizes in classes */
            free(G_lsl); free(G_lou); free(G_lod); free(G_lcls);
            G_lsl=calloc((size_t)nl,sizeof(size_t)); G_lou=calloc((size_t)nl,sizeof(size_t)); G_lod=calloc((size_t)nl,sizeof(size_t));
            G_lcls=calloc((size_t)nl,sizeof(int));
            size_t *cl=calloc((size_t)nl,sizeof(size_t));
            if(G_lsl && G_lou && G_lod && G_lcls && cl){
                G_ncls=0; G_lsl_min=0;
                for(int l=0;l<nl;l++){
                    size_t pg=(G_gmb[l]+3*(size_t)Ih)/4*4, pd=(G_gmbd[l]+3*(size_t)D)/4*4;
                    G_lou[l]=qt_pad256(pg); G_lod[l]=2*G_lou[l]; G_lsl[l]=G_lod[l]+qt_pad256(pd);
                    if(!G_lsl_min || G_lsl[l]<G_lsl_min) G_lsl_min=G_lsl[l];
                    int c=0; while(c<G_ncls && cl[c]!=G_lsl[l]) c++;
                    if(c==G_ncls) cl[G_ncls++]=G_lsl[l];
                    G_lcls[l]=c;
                }
                G_cfree=calloc((size_t)G_ncls,sizeof *G_cfree); G_cnfree=calloc((size_t)G_ncls,sizeof(int)); G_ccap=calloc((size_t)G_ncls,sizeof(int));
                G_exact = G_cfree && G_cnfree && G_ccap;
            }
            free(cl);
        }
        { const char *ae=getenv("QT_GGUF_ARENA"); if(!(ae && *ae=='0')) G.exp_bytes=G_aslot; }   /* 0: a cudaMalloc a matrix (tests) */
    }

    /* Per-device allowance for tier + trunk: CUDA_EXPERT_GB when numeric,
     * else free minus 1 GB headroom and what the engine reserved (qt_reserve). The heat table is loaded here too (it
     * used to be loaded after the budgets) because the placer prices
     * experts by heat. */
    size_t capacity[QT_MAX_DEV]; int capdev[QT_MAX_DEV]; int ncap = G.ndev;
    const char *bg=getenv("CUDA_EXPERT_GB");
    for(int i=0;i<G.ndev;i++){
        size_t freeb=0,totb=0; coli_cuda_mem_info(G.dev[i],&freeb,&totb);
        capdev[i] = G.dev[i];
        capacity[i] = (bg && strcmp(bg,"auto") && atof(bg)>0)
                   ? (size_t)(atof(bg)*1024.0*1024.0*1024.0)
                   : (freeb>(1ull<<30)+G_reserve ? freeb-(1ull<<30)-G_reserve : 0);
        fprintf(stderr,"[qtier] dev %d: %.1f GB free, allowance %.1f GB\n",
                G.dev[i], freeb/1073741824.0, capacity[i]/1073741824.0);
    }
    G.slot=calloc((size_t)nl*ne,sizeof(QSlot));
    if(!G.slot) return 0;
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"rb");
        if(f){
            uint32_t hdr[3]={0,0,0};
            if(fread(hdr,4,3,f)==3 && hdr[0]==0x51544831u && hdr[1]==(uint32_t)nl && hdr[2]==(uint32_t)ne){
                G.heat0=malloc((size_t)nl*ne*4);
                if(G.heat0 && fread(G.heat0,4,(size_t)nl*ne,f)==(size_t)nl*ne){
                    for(size_t i=0;i<(size_t)nl*ne;i++) G.slot[i].heat=G.heat0[i]>>1; /* decay */
                    fprintf(stderr,"[qtier] HEAT_FILE loaded: %s\n",hf);
                } else { free(G.heat0); G.heat0=NULL; }
            }
            fclose(f);
        }
    }
    if(auto_mode()){
    /* QT_HOME=layer: expert homes per layer range, proportional to the cards'
     * allowances unless QT_LAYER_SPLIT=<n> says how many layers the first card
     * takes (two cards). Layers on one card form a pipeline stage. */
    G_home_layer = 0;
    { const char *hm = getenv("QT_HOME");
      if(hm && !strcmp(hm, "layer") && G.ndev > 1 && nl <= QT_DN_MAX_LAYERS){
          G_home_layer = 1;
          /* Weight of a card = its allowance divided by the time it needs for a
           * dense GEMV: a slower card takes fewer layers, or the token waits
           * on it for every one of them (3070 + Quadro RTX 4000: 0.16 against
           * 0.26 ms for 16 MB). Probed here with a 64 MB int8 matrix, ten
           * GEMVs, best of three rounds; a backend that answers in no time
           * (the fake) leaves the weights at the allowances. */
          double w[QT_MAX_DEV], tprobe[QT_MAX_DEV]; int probed = 1;
          { const int PI = 4096, PO = 16384;                 /* 64 MB int8 */
            int8_t *pw = malloc((size_t)PI * PO); float *psc = malloc((size_t)PO * sizeof(float));
            float *px = malloc((size_t)PI * sizeof(float)), *py = malloc((size_t)PO * sizeof(float));
            if(pw && psc && px && py){
                for(size_t i = 0; i < (size_t)PI * PO; i++) pw[i] = (int8_t)(i * 7 % 13) - 6;
                for(int o = 0; o < PO; o++) psc[o] = 0.01f;
                for(int i = 0; i < PI; i++) px[i] = (float)((i * 37) % 101) / 50.f - 1.f;
                for(int i = 0; i < G.ndev; i++){
                    ColiCudaTensor *pt = NULL; tprobe[i] = 0;
                    if(!coli_cuda_matmul(&pt, py, px, pw, psc, 1, 1, PI, PO, G.dev[i], 0)){ probed = 0; break; }
                    double best = 1e30;
                    for(int r = 0; r < 3; r++){
                        for(int j = 0; j < 3; j++) coli_cuda_matmul(&pt, py, px, NULL, NULL, 1, 1, PI, PO, G.dev[i], 0);
                        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
                        for(int j = 0; j < 10; j++) coli_cuda_matmul(&pt, py, px, NULL, NULL, 1, 1, PI, PO, G.dev[i], 0);
                        clock_gettime(CLOCK_MONOTONIC, &t1);
                        double dt = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) / 10;
                        if(dt < best) best = dt;
                    }
                    tprobe[i] = best;
                    coli_cuda_tensor_free(pt);
                    if(best < 20e-6) probed = 0;               /* no real GEMV happened: keep the allowances */
                }
            } else probed = 0;
            free(pw); free(psc); free(px); free(py); }
          double tot = 0;
          for(int i = 0; i < G.ndev; i++){ w[i] = (double)capacity[i] / (probed ? tprobe[i] : 1.0); tot += w[i]; }
          const char *sp = getenv("QT_LAYER_SPLIT"); int k = (sp && *sp) ? atoi(sp) : -1;
          for(int l = 0; l < nl; l++){
              int di = G.ndev - 1;
              if(k >= 0 && k <= nl && G.ndev == 2) di = l < k ? 0 : 1;
              else { double f = (l + 0.5) / nl, acc = 0; for(int i = 0; i < G.ndev; i++){ acc += tot > 0 ? w[i] / tot : 1.0 / G.ndev; if(f < acc){ di = i; break; } } }
              G_layer_dev[l] = (uint8_t)di;
          }
          if(probed){
              int slow = 0; for(int i = 1; i < G.ndev; i++) if(tprobe[i] > tprobe[slow]) slow = i;
              int fast = 0; for(int i = 1; i < G.ndev; i++) if(tprobe[i] < tprobe[fast]) fast = i;
              fprintf(stderr,"[qtier] homes: dense GEMV probe (64 MB int8):");
              for(int i = 0; i < G.ndev; i++) fprintf(stderr," dev %d %.3f ms", G.dev[i], tprobe[i]*1e3);
              if(slow != fast) fprintf(stderr," -> dev %d is %.0f %% slower per dense byte; every layer it holds pays that", G.dev[slow], (tprobe[slow]/tprobe[fast]-1)*100);
              fprintf(stderr,"%s\n", k >= 0 ? " (QT_LAYER_SPLIT set: split taken from it)" : "");
          }
          for(int i = 0; i < G.ndev; i++){
              int lo = -1, hi = -1, n = 0;
              for(int l = 0; l < nl; l++) if(G_layer_dev[l] == i){ if(lo < 0) lo = l; hi = l; n++; }
              fprintf(stderr,"[qtier] homes: layers %d-%d (%d) with their experts and trunk on dev %d\n", lo, hi, n, G.dev[i]);
          }
      } }
        if(G_offer_n) auto_place(nl, ne, topk, capacity, G.heat0);
        else { G_auto_on = 1; G_auto_lmh = QT_PLACE_CPU; for(int l=0;l<QT_DN_MAX_LAYERS;l++) G_auto_dnp[l]=QT_PLACE_CPU; }
    } else {
        const char *e = getenv("COLI_PLACE");
        if(e && !strcmp(e, "off")){ G_auto_on = 1; G_auto_lmh = QT_PLACE_CPU; for(int l=0;l<QT_DN_MAX_LAYERS;l++) G_auto_dnp[l]=QT_PLACE_CPU; }
    }

    /* R4 role split: the lm_head device (COLI_LMHEAD_GPU) is initialized above
     * but removed from the expert PLACEMENT list. Zeroing its budget instead
     * is not enough: home() still hashes experts onto it, and those can never
     * be placed — measured as a hit-rate collapse 89.9% -> 49.5% (only the
     * half of the hot set homed to the remaining device got resident). Its
     * take() would also pace every layer (Quadro: 12.8 ms/token vs 3070 2.4),
     * while lm_head is one latency-tolerant call per token. If it is the ONLY
     * device, experts stay on it — a role split needs two cards. */
    {
        int reserved[QT_MAX_DEV], nres=0;
        /* lm_head: COLI_LMHEAD_GPU stays honoured, COLI_PLACE wins when both
         * are set (it is the newer, general form). */
        const char *lhx=getenv("COLI_LMHEAD_GPU");
        int ld=qt_place_of("lmhead",0);
        if(ld==QT_PLACE_CPU && lhx && *lhx) ld=atoi(lhx);
        if(ld!=QT_PLACE_CPU){
            int present=0; for(int i=0;i<G.ndev;i++) if(G.dev[i]==ld) present=1;
            if(present){ G_lmh.dev=ld; G_lmh.dev_ok=1; reserved[nres++]=ld; }
            else fprintf(stderr,"[qtier] lm_head-Device %d nicht verfuegbar -> CPU\n",ld);
        }
        /* every other component's devices, deduplicated */
        static const char *comps[] = {"dnproj","dnout","attnproj","shexp"};
        for(size_t ci=0; ci<sizeof comps/sizeof *comps; ci++)
            for(int l=0; l<nl; l++){
                int d=qt_place_of(comps[ci],l);
                if(d==QT_PLACE_CPU) continue;
                int seen=0; for(int r=0;r<nres;r++) if(reserved[r]==d) seen=1;
                if(!seen && nres<QT_MAX_DEV) reserved[nres++]=d;
            }
        /* experts=<dev> pins the tier to one card, experts=cpu turns it off.
         * Explicit beats inference: with BOTH cards reserved for other
         * components, the fallback below would hand the experts back to both
         * -- including the slow card, whose take() paces every layer (the
         * measured reason asymmetric expert placement lost). */
        int ed=qt_place_of("experts",0);
        if(G_home_layer){
            /* layer homes: every card keeps the experts of its own layers,
             * whatever a hand-written list reserved on it */
        } else if(ed==QT_PLACE_ALL){
            /* experts=all: every COLI_GPUS card keeps its experts, reserved or
             * not -- the form for "trunk on the fast card, experts on both" */
            fprintf(stderr,"[place] experts=all: the experts stay on all %d cards, the reserved ones included\n",G.ndev);
        } else if(ed!=QT_PLACE_CPU){
            int present=0; for(int i=0;i<G.ndev;i++) if(G.dev[i]==ed) present=1;
            if(present){
                G.dev[0]=ed; G.ndev=1;
                fprintf(stderr,"[place] Experten auf Device %d festgelegt\n",ed);
            } else fprintf(stderr,"[place] experts=%d nicht verfuegbar -> COLI_GPUS bleibt\n",ed);
        } else if(!G_auto_on && G_place_n && qt_place_named("experts")){
            fprintf(stderr,"[place] experts=cpu: no experts in VRAM, the placed trunk stays\n");
            G_trunk_only = 1;
        } else if(nres && !G_auto_on){
            int w=0;
            for(int i=0;i<G.ndev;i++){
                int res=0; for(int r=0;r<nres;r++) if(G.dev[i]==reserved[r]) res=1;
                if(!res) G.dev[w++]=G.dev[i];
            }
            /* w==0: the reserved devices are the only ones -- experts stay on
             * them, exactly as the single-card lm_head case did. */
            if(w>0 && w<G.ndev){
                G.ndev=w;
                fprintf(stderr,"[qtier] %d Device(s) reserviert: aus der Experten-Platzierung genommen\n",nres);
            }
        }
    }

    /* Expert budget per device: the allowance minus the trunk that landed
     * there. The hand-written list gets the same subtraction now: with
     * COLI_PLACE="dnproj=0" the 0.7 GB of projections used to come out of the
     * expert cache unannounced (the R4 measurement). Devices may have been
     * dropped from the expert list by the role split above; match by ordinal. */
    for(int i=0;i<G.ndev;i++){
        size_t trunk = 0;
        for(int o=0;o<G_offer_n;o++)
            if(qt_place_of(G_offer[o].name, G_offer[o].layer)==G.dev[i]) trunk += G_offer[o].bytes;
        size_t cap_i = 0;
        for(int j=0;j<ncap;j++) if(capdev[j]==G.dev[i]) cap_i = capacity[j];
        G_trunk_bytes[i] = trunk;                 /* by expert-device index, for qt_stats */
        G.budget[i] = G_trunk_only ? 0 : cap_i > trunk ? cap_i - trunk : 0;
        /* with exact slots the experts' count at their layers' average slot */
        size_t per = G.exp_bytes;
        if(G_exact){ size_t all=0; for(int l=0;l<G.nl;l++) all+=G_lsl[l]; per = all/(size_t)G.nl; }
        if(G_reserve_short && G.ndev == 1 && G.budget[i] && per && G.budget[i]/per < G_short_experts){
            G.budget[i] = G.budget[i] > G_reserve_short ? G.budget[i] - G_reserve_short : 0;
            fprintf(stderr,"[qtier] dev %d: %.2f GB kept for a prompt chunk's experts from the host pool (fewer than all %zu fit)\n",
                    G.dev[i], G_reserve_short/1073741824.0, G_short_experts);
        }
        fprintf(stderr,"[qtier] dev %d: budget %.2f GB for experts (~%zu experts)%s\n",
                G.dev[i], G.budget[i]/1073741824.0, G.budget[i]/per,
                trunk ? " after trunk" : "");
    }
    if(G.wfmt==QT_GGUF && G.exp_bytes==G_aslot && !G_trunk_only && !qt_arena_up()){
        /* no arena: an expert costs what three cudaMallocs take */
        G.exp_bytes=0;
        for(int l=0;l<nl;l++){
            size_t e = 2*dev_alloc_footprint((G_gmb[l]+3*(size_t)Ih)/4*4) + dev_alloc_footprint((G_gmbd[l]+3*(size_t)D)/4*4);
            if(e>G.exp_bytes) G.exp_bytes=e;
        }
        fprintf(stderr,"[qtier] the GGUF experts' arena was refused: a device allocation an expert matrix (%.2f MB an expert)\n",
                G.exp_bytes/1048576.0);
    }
    /* qt_issue strides each device's block by QT_MAX_ROWS*D floats (its max
     * row count), not 8*D: a device other than 0 with a full 32-row issue
     * used to run past its own slice and off the end of this allocation
     * (#1339). The stride and G.is_k's row capacity are the same constant. */
    G.is_x_floats=(size_t)G.ndev*QT_MAX_ROWS*D;
    G.is_x=malloc(G.is_x_floats*sizeof(float));
    if(!G.is_x) goto fail_storage;
    if(pthread_mutex_init(&G.mx,NULL)) goto fail_storage;
    if(pthread_cond_init(&G.cv,NULL)) goto fail_mutex;
    if(pthread_cond_init(&G.cv_take,NULL)) goto fail_cv;
    qt_aff_get(&aff); qt_aff_widen(&aff);                  /* the uploader inherits this mask */
    int th_ok=pthread_create(&G.th,NULL,uploader,NULL)==0;
    qt_aff_restore(&aff);
    if(!th_ok) goto fail_cv_take;
    G.on=1;
    if(G_trunk_only)
        fprintf(stderr,"[qtier] CUDA: the dense trunk only, the experts of this container stay on the CPU\n");
    else
        fprintf(stderr,"[qtier] CUDA VRAM expert tier active: %d device(s), %.2f MB/expert%s\n",
                G.ndev, G.exp_bytes/1048576.0, G_stream ? ", copied as they pass" : "");
    /* The launcher and coli doctor recognise a Windows CUDA_DLL build by this
     * literal in the binary (they cannot read an import table for a DLL
     * loaded at run time); without it a working GPU build of this engine read
     * as CPU-only and --gpu was refused (#1533). */
    fprintf(stderr,"[CUDA] mode: routed experts (qwen36 VRAM tier)\n");
    return 1;

fail_cv_take:
    pthread_cond_destroy(&G.cv_take);
fail_cv:
    pthread_cond_destroy(&G.cv);
fail_mutex:
    pthread_mutex_destroy(&G.mx);
fail_storage:
    free(G.is_x); G.is_x=NULL; G.is_x_floats=0;
    free(G.heat0); G.heat0=NULL;
    free(G.slot); G.slot=NULL;
    G_lmh.dev_ok=0;
    /* CUDA contexts may also serve standalone dense projections. */
    return 0;
}

int qt_ready(void){ return G.on; }
int qt_streaming(void){ return G.on && G_stream; }
int qt_experts_on(void){
    if(!G.on || G_trunk_only) return 0;
    for(int i=0;i<G.ndev;i++) if(G.budget[i]>=G.exp_bytes) return 1;
    return 0;
}

int qt_lmhead_init_fmt(const void *w, const float *sc, int fmt, int I, int O){
    if(!G_lmh.dev_ok||!G.on||!w||(fmt<100&&!sc)) return 0;
    int dev=G_lmh.dev;
    if(!coli_cuda_tensor_upload(&G_lmh.t,w,sc,fmt,I,O,dev)){
        fprintf(stderr,"[lmh] lm_head upload failed -> stays on CPU\n");
        return 0;
    }
    G_lmh.dev=dev; G_lmh.on=1; G_lmh.fmt=fmt;
    fprintf(stderr,"[lmh] lm_head [%d x %d] %s resident on CUDA dev %d (%.2f GB)\n",
            O,I,fmt>=100?"the GGUF's blocks":"int8",dev,coli_cuda_tensor_bytes(G_lmh.t)/1073741824.0);
    return 1;
}
int qt_lmhead_init(const int8_t *q, const float *sc, int I, int O){ return qt_lmhead_init_fmt(q, sc, 1, I, O); }

int qt_dnproj_init_fmt(int layer, const void *w, const float *sc, int fmt,
                       int I, int O, int device){
    if(layer < 0 || layer >= QT_DN_MAX_LAYERS) return 0;
    if(device == QT_PLACE_CPU || !w || (fmt < 100 && !sc)) return 0;
    /* No G.on requirement: the projections are independent of the expert tier,
     * so they can be measured on a card that holds no experts at all. */
    if(!coli_cuda_tensor_upload(&G_dnp[layer].t, w, sc, fmt, I, O, device)){
        fprintf(stderr,"[dnp] layer %d upload failed -> stays on CPU\n", layer);
        return 0;
    }
    G_dnp[layer].dev = device; G_dnp[layer].on = 1; G_dnp[layer].fmt = fmt;
    return 1;
}
int qt_dnproj_init(int layer, const int8_t *q, const float *sc,
                   int I, int O, int device){ return qt_dnproj_init_fmt(layer, q, sc, 1, I, O, device); }

/* ---- generic resident dense matrices (Qwen3.8 trunk) -----------------------
 * Same mechanism as lmhead/dnproj -- an int8 per-row tensor resident on one
 * device, one GEMV per call -- but addressed by a handle the engine keeps in
 * its weight, so any matrix of any layer can live on the GPU without the
 * tier learning its name. The engine offers sizes through qt_trunk_offer(),
 * asks qt_place_of() where each went, and hands the quantized bytes here. */
#define QT_DENSE_MAX 1024
static struct { ColiCudaTensor *t; int dev, on, fmt; size_t bytes; } G_dense[QT_DENSE_MAX];
static int G_dense_n;
int qt_dense_init_fmt(const void *w, const float *sc, int fmt, int I, int O, int device){
    if(device == QT_PLACE_CPU || !w || (fmt < 100 && !sc) || I <= 0 || O <= 0) return -1;
    if(G_dense_n >= QT_DENSE_MAX) return -1;
    int h = G_dense_n;
    if(!coli_cuda_tensor_upload(&G_dense[h].t, w, sc, fmt, I, O, device)){
        fprintf(stderr,"[dense] upload [%d x %d] to dev %d failed -> stays on CPU\n", O, I, device);
        return -1;
    }
    G_dense[h].dev = device; G_dense[h].on = 1; G_dense[h].fmt = fmt; G_dense[h].bytes = coli_cuda_tensor_bytes(G_dense[h].t);
    G_dense_n++;
    return h;
}
int qt_dense_init(const int8_t *q, const float *sc, int I, int O, int device){ return qt_dense_init_fmt(q, sc, 1, I, O, device); }
int qt_dense_matmul_batch(int h, float *y, const float *x, int S, int I, int O){
    if(h < 0 || h >= G_dense_n || !G_dense[h].on || S <= 0) return 0;
    if(coli_cuda_matmul(&G_dense[h].t, y, x, NULL, NULL, G_dense[h].fmt, S, I, O, G_dense[h].dev, 0)) return 1;
    fprintf(stderr,"[dense] handle %d GPU matmul failed; CPU from here on\n", h);
    G_dense[h].on = 0;
    return 0;
}
int qt_dense_matmul(int h, float *y, const float *x, int I, int O){
    return qt_dense_matmul_batch(h, y, x, 1, I, O);
}
ColiCudaTensor *qt_dense_tensor(int h, int *dev){
    if(h < 0 || h >= G_dense_n || !G_dense[h].on || !G_dense[h].t) return NULL;
    if(dev) *dev = G_dense[h].dev;
    return G_dense[h].t;
}
ColiCudaTensor *qt_dnproj_tensor(int layer, int *dev){
    if(layer < 0 || layer >= QT_DN_MAX_LAYERS || !G_dnp[layer].on || !G_dnp[layer].t) return NULL;
    if(dev) *dev = G_dnp[layer].dev;
    return G_dnp[layer].t;
}
ColiCudaTensor *qt_lmhead_tensor(int *dev){
    if(!G_lmh.on || !G_lmh.t) return NULL;
    if(dev) *dev = G_lmh.dev;
    return G_lmh.t;
}
int qt_dense_count(void){ return G_dense_n; }

int qt_dnproj_ready(int layer){
    return layer >= 0 && layer < QT_DN_MAX_LAYERS && G_dnp[layer].on;
}
int qt_dnproj_matmul_batch(int layer, float *y, const float *x, int S, int I, int O){
    if(!qt_dnproj_ready(layer) || S <= 0) return 0;
    if(coli_cuda_matmul(&G_dnp[layer].t,y,x,NULL,NULL,G_dnp[layer].fmt,S,I,O,G_dnp[layer].dev,0))
        return 1;
    fprintf(stderr,"[dnp] layer %d GPU matmul failed; CPU from here on\n", layer);
    G_dnp[layer].on = 0;
    return 0;
}

/* ---- the DeltaNet layer on the device (see qwen36_tier.h) ---------------- */
static struct { ColiCudaDn *d; int dev, on; ColiCudaTensor *proj, *projz, *outp; } G_dn[QT_DN_MAX_LAYERS];
int qt_dn_gpu_ready(int layer){ return layer >= 0 && layer < QT_DN_MAX_LAYERS && G_dn[layer].on; }
int qt_dn_gpu_init(int layer, int vh, int vk, int kdim, int vdim, int conv_dim, int convk, int hidden,
                   const float *conv_w, const float *norm_w, float eps, int dnout_handle_plus1){
    if(layer < 0 || layer >= QT_DN_MAX_LAYERS) return 0;
    if(!qt_dnproj_ready(layer)) return 0;                              /* the in_proj must be on the card */
    int h = dnout_handle_plus1 - 1;
    if(h < 0 || h >= G_dense_n || !G_dense[h].on) return 0;             /* and the out_proj */
    if(G_dense[h].dev != G_dnp[layer].dev) return 0;                    /* on the same card */
    ColiCudaDn *d = coli_cuda_dn_create(G_dnp[layer].dev, vh, vk, kdim, vdim, conv_dim, convk, hidden, conv_w, norm_w, eps, 0);
    if(!d) return 0;                                                    /* older backend or no memory: CPU path stands */
    G_dn[layer].d = d; G_dn[layer].dev = G_dnp[layer].dev; G_dn[layer].on = 1;
    G_dn[layer].proj = G_dnp[layer].t; G_dn[layer].projz = NULL; G_dn[layer].outp = G_dense[h].t;
    return 1;
}
/* The same layer from three dense handles (qwen38: dnqkv, dnz, dnout as the
 * engine's own trunk items), all on one device; gate_sigmoid selects the
 * gated norm's gate (sigmoid(z) for Qwen3.8, silu(z) for Qwen3.6). */
int qt_dn_gpu_init_dense(int layer, int vh, int vk, int kdim, int vdim, int conv_dim, int convk, int hidden,
                         const float *conv_w, const float *norm_w, float eps, int gate_sigmoid,
                         int proj_handle_plus1, int projz_handle_plus1, int dnout_handle_plus1){
    if(layer < 0 || layer >= QT_DN_MAX_LAYERS) return 0;
    int hp = proj_handle_plus1 - 1, hz = projz_handle_plus1 - 1, ho = dnout_handle_plus1 - 1;
    if(hp < 0 || hp >= G_dense_n || !G_dense[hp].on) return 0;
    if(hz < 0 || hz >= G_dense_n || !G_dense[hz].on) return 0;
    if(ho < 0 || ho >= G_dense_n || !G_dense[ho].on) return 0;
    if(G_dense[hp].dev != G_dense[hz].dev || G_dense[hp].dev != G_dense[ho].dev) return 0;   /* one card */
    ColiCudaDn *d = coli_cuda_dn_create(G_dense[hp].dev, vh, vk, kdim, vdim, conv_dim, convk, hidden, conv_w, norm_w, eps, gate_sigmoid);
    if(!d) return 0;
    G_dn[layer].d = d; G_dn[layer].dev = G_dense[hp].dev; G_dn[layer].on = 1;
    G_dn[layer].proj = G_dense[hp].t; G_dn[layer].projz = G_dense[hz].t; G_dn[layer].outp = G_dense[ho].t;
    return 1;
}
int qt_dn_gpu_set_state(int layer, const float *ring, const float *rec){
    return qt_dn_gpu_ready(layer) && coli_cuda_dn_set_state(G_dn[layer].d, ring, rec);
}
int qt_dn_gpu_get_state(int layer, float *ring, float *rec){
    /* also after a failed step (on == 0): the object still holds the state */
    return layer >= 0 && layer < QT_DN_MAX_LAYERS && G_dn[layer].d && coli_cuda_dn_get_state(G_dn[layer].d, ring, rec);
}
int qt_dn_gpu_step(int layer, const float *x, float *out, const float *egh, const float *beta){
    if(!qt_dn_gpu_ready(layer)) return 0;
    if(coli_cuda_dn_step(G_dn[layer].d, G_dn[layer].proj, G_dn[layer].projz, G_dn[layer].outp, x, out, egh, beta)) return 1;
    fprintf(stderr,"[dn] layer %d GPU step failed; CPU from here on\n", layer);
    G_dn[layer].on = 0;
    return 0;
}
static void dn_gpu_free_all(void){
    for(int l=0;l<QT_DN_MAX_LAYERS;l++){ if(G_dn[l].d) coli_cuda_dn_free(G_dn[l].d); G_dn[l].d=NULL; G_dn[l].on=0; }
}

int qt_dnproj_matmul(int layer, float *y, const float *x, int I, int O){
    return qt_dnproj_matmul_batch(layer, y, x, 1, I, O);
}

int qt_lmhead_matmul(float *y, const float *x, int I, int O){
    if(!G_lmh.on) return 0;
    /* cached-tensor path: upload params are ignored once *t exists */
    if(coli_cuda_matmul(&G_lmh.t,y,x,NULL,NULL,G_lmh.fmt,1,I,O,G_lmh.dev,0)) return 1;
    fprintf(stderr,"[lmh] GPU matmul failed; falling back to CPU from here on\n");
    G_lmh.on=0;
    return 0;
}

/* Is (layer,eid) currently VRAM-resident? (used to free RAM-side int8 copies) */
/* ---- the dense chain's routed experts (qwen36_cuda_chain.h, qwen38_cuda_chain.h) ----
 * The chain reads the residents' tensors by address from a table it builds when the epoch
 * moved, and holds the tier for each forward that reads that table: no swap writes into a
 * resident's buffers while one is in flight (an eviction only clears the flag at once,
 * which moves the epoch; its buffers wait for the hold to drop). The books qt_issue keeps
 * for the experts it serves, qt_chain_count keeps for the chain's. */
static void qt_lfru_tick_locked(void);
/* ---- the host pool: the RAM cache the device copies a prompt chunk's experts from -------------
 * A prompt chunk routes nearly every expert of a layer, and the CPU's step costs per row while a
 * copy over the bus costs per expert: from a few dozen rows up, copying a layer's non-resident
 * experts into VRAM (the chain's staging, one layer at a time, the copy of the next overlapping
 * the layer before) and running them there beats the host's step. The engine's RAM cache serves
 * the copies as it is, page-locked here (a copy at the bus's full rate, no second copy in RAM).
 * A decode step's slots stay the CPU's. On when the RAM holds every expert (no slot is ever
 * recycled), one device, and the RAM copy is a layout the device's GEMMs read: expert_ffn.h's
 * planar int4 per 64 (coli_cuda_planar_rows), or a GGUF's rows as the file has them (gate, up
 * and down back to back, no scales; the engine holds them so when this tier takes a GGUF's
 * experts). COLI_CUDA_HOST_POOL=0 off. */
static int G_host_on = -1;
static unsigned long long G_host_n, G_host_bytes;
static int host_pool_on(void){
    if(G_host_on >= 0) return G_host_on;
    const char *e = getenv("COLI_CUDA_HOST_POOL");
    G_host_on = !(e && *e == '0') && G.on && !G_stream && !G_trunk_only && G.ndev == 1 &&
                ((G_src_planar && G.wfmt == 4 && G.egs == 64 && coli_cuda_planar_rows(G.dev[0], G.D, G.Ih)) || G.wfmt == QT_GGUF);
    return G_host_on;
}
/* a host pool expert's bytes: its matrices (three of mb, a GGUF's gate, up and down) and scales */
static size_t host_w_bytes(int layer){ return G.wfmt == QT_GGUF ? 2*G_gmb[layer] + G_gmbd[layer] : 3*((size_t)G.D*G.Ih/2); }
static size_t host_s_bytes(void){ return G.wfmt == QT_GGUF ? 0 : (2*G.sc_gu + G.sc_d)*sizeof(float); }
void qt_note_host(int layer,int eid,
                  const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
                  const float *gs,const float *us,const float *ds){
    const int gg = G.wfmt == QT_GGUF;
    if(!G.on || !g4 || (!gs && !gg) || layer<0 || layer>=G.nl || eid<0 || eid>=G.ne) return;
    size_t mb = gg ? G_gmb[layer] : (size_t)G.D*G.Ih/2, wb = host_w_bytes(layer), sb = host_s_bytes();
    /* one block each: the three matrices and the three scale arrays, as the engine's slot holds them */
    if(u4 != g4 + mb || d4 != g4 + 2*mb || (!gg && (us != gs + G.sc_gu || ds != gs + 2*G.sc_gu))) return;
    pthread_mutex_lock(&G.mx);
    QSlot *s = qs(layer,eid);
    int skip = !host_pool_on() || s->hreg || G.th_stop;   /* decided once, under the lock (the warm start's threads) */
    if(!skip){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gg?NULL:gs; s->us=gg?NULL:us; s->ds=gg?NULL:ds; }
    pthread_mutex_unlock(&G.mx);
    if(skip) return;
    if(!coli_cuda_host_register((void *)g4, wb)) return;
    if(sb && !coli_cuda_host_register((void *)gs, sb)){ coli_cuda_host_unregister((void *)g4); return; }
    pthread_mutex_lock(&G.mx);
    s->hreg = 1; G.epoch++; if(G.chg_on) G.chg_over=1;
    G_host_n++; G_host_bytes += wb + sb;
    pthread_mutex_unlock(&G.mx);
}
int qt_chain_host(int layer,int eid,const void **w,size_t *wb,const float **sc,size_t *sb){
    if(!G.on || G_host_on <= 0 || layer<0 || layer>=G.nl || eid<0 || eid>=G.ne) return 0;
    pthread_mutex_lock(&G.mx);
    QSlot *s = qs(layer,eid);
    int r = s->hreg;
    if(r){
        *w = s->g4; *wb = host_w_bytes(layer);
        *sc = s->gs; *sb = host_s_bytes();
    }
    pthread_mutex_unlock(&G.mx);
    return r;
}
int qt_chain_host_count(void){ return G.on && G_host_on > 0 ? (int)G_host_n : 0; }
/* The pool follows the residents. The warm start page-locks the experts it leaves in RAM; the
 * chain's promotions and the tier's LFRU swaps then trade experts between the card and the RAM, and
 * the pool kept its first set: an expert that left the card was not in it, so a prompt chunk's
 * layer holding one went to the host's step whole (q36cc_stage_table), and one that came in kept
 * its RAM page-locked for nothing. Before a large prompt chunk (Q36CC_FOLLOW_ROWS) the chain calls
 * this: the non-residents join the pool, the residents leave it; the CUDA calls outside the lock,
 * as in qt_note_host.
 * 1 when the pool changed; *joined and *left say how many. */
int qt_chain_host_follow(int *joined,int *left){
    *joined=*left=0;
    if(!G.on || G_host_on <= 0) return 0;
    size_t n = (size_t)G.nl*G.ne, mb = (size_t)G.D*G.Ih/2, sb = (2*G.sc_gu + G.sc_d)*sizeof(float);
    int *mv = malloc(n*sizeof(int)), nj = 0, nl = 0;   /* the joins from the front, the leaves from the back */
    if(!mv) return 0;
    pthread_mutex_lock(&G.mx);
    for(size_t i=0; i<n && !G.th_stop; i++){
        QSlot *s = &G.slot[i];
        if(!s->resident && !s->hreg && s->g4 && s->gs &&
           s->u4 == s->g4 + mb && s->d4 == s->g4 + 2*mb && s->us == s->gs + G.sc_gu && s->ds == s->gs + 2*G.sc_gu)
            mv[nj++] = (int)i;
        else if(s->resident && s->hreg){   /* out of qt_chain_host now, released below */
            s->hreg = 0; G_host_n--; G_host_bytes -= 3*mb + sb; mv[n-1-nl++] = (int)i;
        }
    }
    pthread_mutex_unlock(&G.mx);
    /* the slots' RAM stays where it is: the pool is on only when the RAM holds every expert */
    for(int k=0; k<nl; k++){ QSlot *s = &G.slot[mv[n-1-k]]; coli_cuda_host_unregister((void *)s->g4); coli_cuda_host_unregister((void *)s->gs); }
    int in = 0;
    for(int k=0; k<nj; k++){
        QSlot *s = &G.slot[mv[k]];
        if(!coli_cuda_host_register((void *)s->g4, 3*mb)) continue;
        if(!coli_cuda_host_register((void *)s->gs, sb)){ coli_cuda_host_unregister((void *)s->g4); continue; }
        pthread_mutex_lock(&G.mx);
        int still = !s->resident && !s->hreg && !G.th_stop;
        if(still){ s->hreg = 1; G_host_n++; G_host_bytes += 3*mb + sb; }
        pthread_mutex_unlock(&G.mx);
        if(still) in++;
        else { coli_cuda_host_unregister((void *)s->g4); coli_cuda_host_unregister((void *)s->gs); }
    }
    free(mv);
    *joined = in; *left = nl;
    return in + nl > 0;
}

unsigned qt_chain_epoch(void){
    if(!G.on) return 0;
    pthread_mutex_lock(&G.mx);
    unsigned e=G.epoch;
    pthread_mutex_unlock(&G.mx);
    return e;
}
int qt_chain_devices(void){ return G.on ? G.ndev : 0; }
int qt_chain_history(void){ return G.on && G.heat0; }
int qt_chain_resident(int layer,int eid,int device,ColiCudaTensor **g,ColiCudaTensor **u,ColiCudaTensor **d){
    if(!G.on || G_trunk_only || G_chain_own || layer<0 || layer>=G.nl || eid<0 || eid>=G.ne) return 0;
    pthread_mutex_lock(&G.mx);
    QSlot *s=qs(layer,eid);
    int r=s->resident && s->tg && s->tu && s->td && G.dev[home2(layer,eid)]==device;
    if(r){ *g=s->tg; *u=s->tu; *d=s->td; }
    pthread_mutex_unlock(&G.mx);
    return r;
}
void qt_chain_hold(int on){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    if(on) G.chain_hold++;
    else if(G.chain_hold>0 && --G.chain_hold==0) pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
}
void qt_chain_count(int layer,const int *eids,int K){
    if(!G.on || layer<0 || layer>=G.nl) return;
    pthread_mutex_lock(&G.mx);
    if(G.th_stop){ pthread_mutex_unlock(&G.mx); return; }
    if(layer==0) qt_lfru_tick_locked();
    for(int k=0;k<K;k++){
        if(eids[k]<0 || eids[k]>=G.ne) continue;
        QSlot *s=qs(layer,eids[k]);
        if(s->conv<0xFFFFFFFFu) s->conv++;
        if(s->resident){ if(s->heat<0xFFFFFFFFu) s->heat++; G.hits[home2(layer,eids[k])]++; }
        else {
            G.miss++;
            if(!G.pm){ G.pm=malloc(QT_PM_CAP*sizeof(int)); G.pm_cap=G.pm?QT_PM_CAP:0; }
            if(G.pm_n<G.pm_cap && !s->queued && s->g4) G.pm[G.pm_n++]=(int)(s-G.slot);
        }
    }
    pthread_mutex_unlock(&G.mx);
}
/* The chain's promotions, after a decode-sized step: up to max of the step's misses, the most
 * worth the most first (pm_score: the heat, which carries the history and this conversation,
 * and this conversation's routes once more), each into the buffers of the coldest of up to 64
 * residents drawn at random on its device (none queued, swap_ok), where the newcomer clears
 * the tier's admission margin (tier_should_promote). The victim computes on the CPU from now
 * on (its flag cleared, the epoch moved, its int8 copy rebuilt by tier_rebuild_evicted before
 * the next forward where QWEN_EXPERT_KERNEL=0 needs one, planar int4 keeping its RAM blocks)
 * and both slots are marked queued until qt_chain_promote_done: the chain
 * copies the newcomer's RAM bytes into the victim's device buffers itself, on its copy stream
 * (the tier's uploader would take the device's lock from another thread while the chain's
 * frame waits for the host: a deadlock inside the driver, seen on an RTX 3070). The bytes go
 * into the chain's staging slot through qt_chain_promote_stage: as the RAM holds them for
 * int8 and pair-ordered int4 rows, or converted from planar blocks into signed pairs.
 * Returns how many pairs out holds. max 0 (the chain's brake held, or its rate is 0) drops the
 * step's misses: the list is a step's, the next step's begins empty. */
/* an expert's worth: its heat (the history's routes, decayed, plus this conversation's: the
 * residents' from qt_chain_count, the misses' from qt_note) and this conversation's routes
 * once more, so that a conversation away from the history overtakes within tens of tokens */
static inline uint64_t pm_score(const QSlot *s){ return (uint64_t)s->heat + s->conv; }
static int cmp_pm(const void *a,const void *b){          /* score desc, conv desc, index asc: a total order */
    int ia=*(const int*)a, ib=*(const int*)b;
    uint64_t sa=pm_score(&G.slot[ia]), sb=pm_score(&G.slot[ib]);
    if(sa!=sb) return sa<sb ? 1 : -1;
    uint32_t ca=G.slot[ia].conv, cb=G.slot[ib].conv;
    if(ca!=cb) return ca<cb ? 1 : -1;
    return ia<ib ? -1 : ia>ib ? 1 : 0;
}
static int G_promo_min=-1;   /* COLI_CUDA_CHAIN_PROMOTE_MIN: routes in this conversation a miss needs before it is promoted (a one-off is not worth a swap) */
int qt_chain_promote(int max,QtPromo *out){
    if(!G.on) return 0;
    if(max<=0 || !out){ pthread_mutex_lock(&G.mx); G.pm_n=0; pthread_mutex_unlock(&G.mx); return 0; }
    if(G_chain_own || G_trunk_only || G_stream) return 0;
    if(G_promo_min<0){ const char *e=getenv("COLI_CUDA_CHAIN_PROMOTE_MIN"); G_promo_min = e && *e ? atoi(e) : 2; if(G_promo_min<1) G_promo_min=1; }
    if(G.wfmt==QT_GGUF) { G.pm_n=0; return 0; }
    pthread_mutex_lock(&G.mx);
    int done=0;
    if(G.pm_n>0 && !G.th_stop){
        size_t n=(size_t)G.nl*G.ne, mb=(size_t)G.D*G.Ih/(G.wfmt==4?2:1);
        if(G.wfmt==2) mb=(size_t)G.D*G.Ih/2;
        if(!G.rng) G.rng=0x9e3779b9u;
        qsort(G.pm,(size_t)G.pm_n,sizeof(int),cmp_pm);
        for(int a=0;a<G.pm_n && done<max;a++){
            int ci=G.pm[a]; QSlot *cs=&G.slot[ci];
            if(cs->conv<(uint32_t)G_promo_min) continue;       /* a one-off is not worth a swap */
            if(cs->resident||cs->queued||!cs->g4) continue;   /* taken by an earlier entry (a verify routes a slot twice), or landed */
            int cl=ci/G.ne, di=home2(cl,ci%G.ne);
            int v=-1, found=0; uint64_t vsc=0;
            for(int t=0;t<1024 && found<64;t++){
                G.rng=G.rng*1664525u+1013904223u;
                int r=(int)((G.rng>>8)%(unsigned)n);
                QSlot *s=&G.slot[r];
                if(!s->resident||s->queued||!s->tg||!s->tu||!s->td||home2(r/G.ne,r%G.ne)!=di||!swap_ok(cl,r/G.ne)) continue;
                found++;
                if(v<0 || pm_score(s)<vsc){ v=r; vsc=pm_score(s); }
            }
            /* the tier's admission margin (tier.h): a quarter and four over the victim, no ping-pong */
            if(v<0 || !tier_should_promote((uint32_t)(pm_score(cs)>0xFFFFFFFFu?0xFFFFFFFFu:pm_score(cs)), (uint32_t)(vsc>0xFFFFFFFFu?0xFFFFFFFFu:vsc))) continue;
            QSlot *vs=&G.slot[v];
            res_set(vs,0); evict_note_locked(v);               /* CPU fallback from now on, from its RAM copy (see above) */
            vs->queued=1; cs->queued=1;                        /* neither moves until qt_chain_promote_done */
            QtPromo *o=&out[done++];
            o->ci=ci; o->vi=v; o->g=cs->g4; o->u=cs->u4; o->d=cs->d4; o->gs=cs->gs; o->us=cs->us; o->ds=cs->ds;
            o->tg=vs->tg; o->tu=vs->tu; o->td=vs->td;
            o->mb=mb; o->sb_gu=G.sc_gu*sizeof(float); o->sb_d=G.sc_d*sizeof(float);
            G.swaps++; G.promos++;
        }
    }
    G.pm_n=0;
    pthread_mutex_unlock(&G.mx);
    return done;
}
/* Fill a page-locked staging slot with the newcomer's three matrices (3 * p->mb bytes)
 * followed by its three scale arrays (2 * p->sb_gu + p->sb_d bytes), in the layout the
 * victim's device tensors hold: planar int4 blocks become signed pairs, int8 and pair-ordered
 * int4 rows are copied as they are. */
void qt_chain_promote_stage(void *dst,const QtPromo *p){
    if(!dst || !p) return;
    uint8_t *dw=(uint8_t*)dst;
    if(G.wfmt==4 && G_src_planar){
        planar_to_pairs_signed(dw,         p->g, p->mb);
        planar_to_pairs_signed(dw+p->mb,   p->u, p->mb);
        planar_to_pairs_signed(dw+2*p->mb, p->d, p->mb);
    } else {
        memcpy(dw,         p->g, p->mb);
        memcpy(dw+p->mb,   p->u, p->mb);
        memcpy(dw+2*p->mb, p->d, p->mb);
    }
    char *sc=(char*)dst + 3*p->mb;
    memcpy(sc,            p->gs, p->sb_gu);
    memcpy(sc+p->sb_gu,   p->us, p->sb_gu);
    memcpy(sc+2*p->sb_gu, p->ds, p->sb_d);
}
/* the copy of a pair qt_chain_promote gave landed (ok) or failed: the newcomer takes the
 * victim's tensors and is resident; on a failure the victim's buffers hold neither expert
 * and are left to the tier's exit (nobody's from now on) */
void qt_chain_promote_done(int ci,int vi,int ok){
    if(!G.on || ci<0 || vi<0 || (size_t)ci>=(size_t)G.nl*G.ne || (size_t)vi>=(size_t)G.nl*G.ne) return;
    pthread_mutex_lock(&G.mx);
    QSlot *cs=&G.slot[ci], *vs=&G.slot[vi];
    if(ok){ cs->tg=vs->tg; cs->tu=vs->tu; cs->td=vs->td; vs->tg=vs->tu=vs->td=NULL; res_set(cs,1); G.uploads++; G.inplace++; G.promo_done++; }
    else  { vs->tg=vs->tu=vs->td=NULL; }
    cs->queued=0; vs->queued=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
}
/* the residency changes since the last call (slot indices layer*ne+eid, at most max of them);
 * *overflow 1: more than fit, or one the list does not describe -- read every entry again.
 * The first call turns the recording on: before it the chain reads every entry anyway. */
int qt_chain_changes(int *idx,int max,int *overflow){
    if(!G.on){ *overflow=0; return 0; }
    pthread_mutex_lock(&G.mx);
    int n=G.chg_n<max?G.chg_n:max;
    if(n>0) memcpy(idx,G.chg,(size_t)n*sizeof(int));
    *overflow = !G.chg_on || G.chg_over || G.chg_n>max;
    G.chg_on=1; G.chg_n=0; G.chg_over=0;
    pthread_mutex_unlock(&G.mx);
    return n;
}
uint64_t qt_chain_promoted(void){ return G.on ? G.promo_done : 0; }
/* The arena of a GGUF's experts on the tier's one device, for a chain that takes it over
 * (qt_chain_own): its base, a slot's bytes, the up and down rows' offsets in a slot (gate at
 * 0), its slots; 0 without an arena. qt_chain_slot: the slot expert eid of layer holds, -1
 * none. qt_chain_own: from now on the chain moves experts in and out of the slots itself;
 * the tier places, swaps and computes none (qt_note*, qt_issue and qt_chain_resident answer
 * nothing), its uploads in flight done first. */
int qt_chain_arena(void **base,size_t *slot,size_t *off_u,size_t *off_d,int *nslots){
    if(!G.on || G.ndev!=1 || G.wfmt!=QT_GGUF || !G_arena_on || !G_arena[0] || G_exact) return 0;   /* slots of one size only */
    *base=G_arena[0]; *slot=G_aslot; *off_u=G_aoff_u; *off_d=G_aoff_d; *nslots=G_aslots[0];
    return 1;
}
int qt_chain_slot(int layer,int eid){
    if(!G.on || !G_arena_on || layer<0 || layer>=G.nl || eid<0 || eid>=G.ne) return -1;
    pthread_mutex_lock(&G.mx);
    QSlot *s=qs(layer,eid);
    int r=s->resident && s->aslot>0 ? s->aslot-1 : -1;
    pthread_mutex_unlock(&G.mx);
    return r;
}
void qt_chain_own(void){
    if(!G.on) return;
    qt_fill_wait();
    pthread_mutex_lock(&G.mx);
    G_chain_own=1;
    pthread_mutex_unlock(&G.mx);
}
int qt_chain_format(int *fmt,int *gs){
    if(!G.on) return 0;
    if(fmt) *fmt=G.wfmt==QT_GGUF?QT_GGUF:G.wfmt==8?8:G.wfmt==1?1:G.egs?4:2;
    if(gs) *gs=G.egs;
    return 1;
}
int qt_chain_layer_format(int layer,int *fmt,int *fmt_d){
    if(!G.on || layer<0 || layer>=G.nl) return 0;
    int f; qt_chain_format(&f,NULL);
    *fmt = G.wfmt==QT_GGUF ? 100+G_gtg[layer] : f;
    *fmt_d = G.wfmt==QT_GGUF ? 100+G_gtd[layer] : f;
    return 1;
}

int qt_is_resident(int layer,int eid){
    if(!G.on) return 0;
    pthread_mutex_lock(&G.mx);
    int r = qs(layer,eid)->resident;
    pthread_mutex_unlock(&G.mx);
    return r;
}

/* internal, G.mx held: enqueue one upload. victim=-1: plain upload (budget is
 * reserved here); victim>=0: LFRU swap (budget neutral). */
static int enqueue_locked(int layer,int eid,int v_layer,int v_eid,int reserved){
    QSlot *s=qs(layer,eid);
    /* Nothing is accepted once shutdown has been requested: a waiter woken by
     * the shutdown broadcast (qt_note_block / qt_note_planned on a full queue)
     * would otherwise enqueue into a queue the uploader may already have left,
     * and that entry stays queued=1 with its staging buffers forever. */
    if(G.th_stop) return 0;
    if(s->resident||s->queued||!s->g4) return 0;
    if(G.qn>=QT_QCAP){ G.q_full_skips++; return 0; }
    int hd=home2(layer,eid);
    if(!reserved && v_eid<0 && G.used[hd]+qt_ebytes(layer)>G.budget[hd]) return 0;
    /* staging: three matrices (int8/fp8 a byte an element, a GGUF's the file's rows), their scales */
    uint8_t *w=malloc(2*qt_mb_gu(layer)+qt_mb_d(layer)); float *sc=malloc((2*G.sc_gu+G.sc_d+1)*sizeof(float));
    if(!w||!sc){ free(w); free(sc); return 0; }
    if(!reserved && v_eid<0) G.used[hd]+=qt_ebytes(layer);
    s->queued=1;
    stage(w,sc,layer,s->g4,s->u4,s->d4,s->gs,s->us,s->ds);
    G.q[G.qt_].layer=layer; G.q[G.qt_].eid=eid; G.q[G.qt_].w=w; G.q[G.qt_].s=sc;
    G.q[G.qt_].v_layer=v_layer; G.q[G.qt_].v_eid=v_eid;
    G.qt_=(G.qt_+1)%QT_QCAP; G.qn++; G.inflight++;
    pthread_cond_signal(&G.cv);
    return 1;
}

/* streaming mode: the bytes in hand are valid only during this call, so set
 * the pointers for the enqueue (which stages a copy under the lock) and drop
 * them again before returning. Nothing downstream may read them later. */
static void stream_point(QSlot *s,const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
                         const float *gs,const float *us,const float *ds){
    s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds;
}
static void stream_forget(QSlot *s){ s->g4=s->u4=s->d4=NULL; s->gs=s->us=s->ds=NULL; }

/* The LFRU decision at the moment the bytes pass by: if this expert is not
 * resident and its device has no room, evict the coldest resident there when
 * the admission rule says the newcomer is worth it. Budget-neutral swap. */
static void stream_promote_locked(int layer,int eid){
    QSlot *s=qs(layer,eid);
    if(s->resident||s->queued) return;
    int hd=home2(layer,eid);
    if(G.used[hd]+qt_ebytes(layer)<=G.budget[hd]){ enqueue_locked(layer,eid,-1,-1,0); return; }
    size_t n=(size_t)G.nl*G.ne; int cold=-1; uint32_t ch=0;
    for(size_t i=0;i<n;i++){
        QSlot *c=&G.slot[i];
        if(home2((int)(i/G.ne),(int)(i%G.ne))!=hd || !c->resident || c->queued || !swap_ok(layer,(int)(i/G.ne))) continue;
        if(cold<0||c->heat<ch){ cold=(int)i; ch=c->heat; }
    }
    if(cold<0 || !tier_should_promote(s->heat,ch)) return;
    QSlot *v=&G.slot[cold];
    res_set(v,0);
    if(enqueue_locked(layer,eid,cold/G.ne,cold%G.ne,0)) G.swaps++;
    else res_set(v,1);
}

void qt_note(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4 || G_trunk_only || G_chain_own) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(G_stream){
        if(s->heat<0xFFFFFFFFu) s->heat++;
        stream_point(s,g4,u4,d4,gs,us,ds);
        stream_promote_locked(layer,eid);
        stream_forget(s);
        pthread_mutex_unlock(&G.mx);
        return;
    }
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    if(s->heat<0xFFFFFFFFu) s->heat++;
    enqueue_locked(layer,eid,-1,-1,0);
    pthread_mutex_unlock(&G.mx);
}

/* The warm start's order from an engine's expert history (route_trace.h's counts, counts[l][e]
 * for nl layers of ne experts): qt_plan_fill hands out the most used first, as HEAT_FILE's. */
void qt_heat_order(uint32_t *const *counts, int nl, int ne){
    if(!G.on || !counts || nl!=G.nl || ne!=G.ne) return;
    pthread_mutex_lock(&G.mx);
    if(!G.heat0) G.heat0=malloc((size_t)nl*ne*4);
    if(G.heat0){
        for(int l=0;l<nl;l++) for(int e=0;e<ne;e++) G.heat0[(size_t)l*ne+e]=counts[l]?counts[l][e]:0;
        free(G.fill_order); G.fill_order=NULL;   /* sorted again at the next plan */
    }
    pthread_mutex_unlock(&G.mx);
}
/* Copy mode: whether a qt_note of this expert now would copy it (not resident nor queued,
 * and room on its device or a resident colder than it, as of the last tick), so an engine
 * converts the bytes only then; qt_note_heat counts a pass of one it does not hand over. */
int qt_note_wanted(int layer,int eid){
    if(!G.on || G_trunk_only || G_chain_own || layer<0 || layer>=G.nl || eid<0 || eid>=G.ne) return 0;
    if(!G_stream) return 1;
    QSlot *s=qs(layer,eid);
    if(s->resident || s->queued) return 0;
    int hd=home2(layer,eid);
    if(G.used[hd]+qt_ebytes(layer)<=G.budget[hd]) return 1;
    return G_cold_heat[hd]!=0xFFFFFFFFu && tier_should_promote(s->heat+1,G_cold_heat[hd]);
}
void qt_note_heat(int layer,int eid){
    if(!G.on || layer<0 || layer>=G.nl || eid<0 || eid>=G.ne) return;
    QSlot *s=qs(layer,eid);
    if(s->heat<0xFFFFFFFFu) __atomic_add_fetch(&s->heat,1u,__ATOMIC_RELAXED);
}

/* blocking variant for the warmstart (waits for queue space). */
void qt_note_block(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4 || G_trunk_only || G_chain_own) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(G_stream) stream_point(s,g4,u4,d4,gs,us,ds);
    else if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    while(G.qn>=QT_QCAP && !G.th_stop) wait_take_locked();
    enqueue_locked(layer,eid,-1,-1,0);
    if(G_stream) stream_forget(s);
    pthread_mutex_unlock(&G.mx);
}

/* warmstart order -- heat descending (HEAT_FILE, or an engine's qt_heat_order), else expert
 * by expert across the layers: expert 0 of every layer, then expert 1, ... Index order
 * filled whole layers (a T4: layers 0-24 and part of 25 on the card, 26-39 none), and a
 * prompt chunk then copied each of those layers' 256 experts over the bus while it waited,
 * after 25 layers with the bus idle; spread, every layer copies its share beside the work of
 * the layer before. Under G.mx; 0: no memory. */
static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    uint32_t ha=g_sort_heat[*(const int*)a], hb=g_sort_heat[*(const int*)b];
    return ha<hb ? 1 : ha>hb ? -1 : 0;
}
static int fill_order_make(void){
    size_t n=(size_t)G.nl*G.ne;
    G.fill_order=malloc(n*sizeof(int));
    if(!G.fill_order) return 0;
    /* no heat: the i-th pick is layer i%nl, expert i/nl; with heat: index order, sorted below */
    for(size_t i=0;i<n;i++) G.fill_order[i]=G.heat0 ? (int)i : (int)((i%G.nl)*G.ne+i/G.nl);
    if(G.heat0){ g_sort_heat=G.heat0; qsort(G.fill_order,n,sizeof(int),cmp_heat_desc); }
    G.fill_cur=0;
    return 1;
}
/* Returns 0 once all budgets are full or the list is exhausted. */
int qt_fill_next(int *layer,int *eid){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order && !fill_order_make()){ pthread_mutex_unlock(&G.mx); return 0; }
    while((size_t)G.fill_cur<n){
        int gi=G.fill_order[G.fill_cur];
        int l=gi/G.ne, e=gi%G.ne, hd=home2(l,e);
        QSlot *s=qs(l,e);
        int full=1; for(int i=0;i<G.ndev;i++) if(G.used[i]+qt_ebytes_min()<=G.budget[i]) full=0;
        if(full){ pthread_mutex_unlock(&G.mx); return 0; }
        G.fill_cur++;
        if(s->resident||s->queued) continue;
        if(G.used[hd]+qt_ebytes(l)>G.budget[hd]) continue;   /* dieses Device voll */
        *layer=l; *eid=e;
        pthread_mutex_unlock(&G.mx);
        return 1;
    }
    pthread_mutex_unlock(&G.mx);
    return 0;
}

/* Plan the whole warmstart set in one pass -- same heat order and budget
 * reservation as qt_fill_next, but without loading. The experts are then
 * loaded by any number of threads and handed over via qt_note_planned. */
/* Layer `layer` routes nothing (a Flint dense layer): no expert of it is ever planned. */
void qt_layer_off(int layer){
    if(!G.on || layer<0 || layer>=G.nl) return;
    pthread_mutex_lock(&G.mx);
    if(!G.off) G.off=calloc((size_t)G.nl,1);
    if(G.off) G.off[layer]=1;
    free(G.fill_order); G.fill_order=NULL;   /* planned again without it */
    pthread_mutex_unlock(&G.mx);
}
int qt_plan_fill(int *layers,int *eids,int max){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    int cnt=0;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order && !fill_order_make()){ pthread_mutex_unlock(&G.mx); return 0; }
    while((size_t)G.fill_cur<n && cnt<max){
        int full=1; for(int i=0;i<G.ndev;i++) if(G.used[i]+qt_ebytes_min()<=G.budget[i]) full=0;
        if(full) break;
        int gi=G.fill_order[G.fill_cur++];
        int l=gi/G.ne, e=gi%G.ne, hd=home2(l,e);
        if(G.off && G.off[l]) continue;   /* a layer with no experts (Flint's dense ones) */
        QSlot *s=qs(l,e);
        if(s->resident||s->queued||s->planned) continue;
        if(G.used[hd]+qt_ebytes(l)>G.budget[hd]) continue;
        G.used[hd]+=qt_ebytes(l);         /* reserve */
        s->planned=1;
        layers[cnt]=l; eids[cnt]=e; cnt++;
    }
    pthread_mutex_unlock(&G.mx);
    return cnt;
}

/* Thread-safe (callable from multiple loader threads): stage + enqueue one
 * expert reserved by qt_plan_fill; blocks only while the queue is full. */
/* g/u/d: i pesi COME STANNO IN RAM -- int4 impacchettati su un container gs64,
 * int8 su un container int8. Il formato lo decide qt_init dal container, e da
 * li' in poi staging e upload lo seguono. */
void qt_note_planned(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || G_chain_own) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!g4){
        /* The loader had nothing to hand over. qt_plan_fill reserved budget
         * and set planned=1 for this expert; returning here without undoing
         * both keeps the bytes out of the budget for the life of the process
         * and "if(resident||queued||planned) continue" never reconsiders the
         * expert. #1331 was this leak for every expert of an int8 container. */
        if(s->planned){ G.used[home2(layer,eid)]-=qt_ebytes(layer); s->planned=0; }
        pthread_mutex_unlock(&G.mx);
        return;
    }
    if(G_stream) stream_point(s,g4,u4,d4,gs,us,ds);
    else if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    while(G.qn>=QT_QCAP && !G.th_stop) wait_take_locked();
    if(!enqueue_locked(layer,eid,-1,-1,1)){
        /* not enqueueable (e.g. already resident): return the reservation */
        if(s->planned) G.used[home2(layer,eid)]-=qt_ebytes(layer);
    }
    s->planned=0;
    if(G_stream) stream_forget(s);
    pthread_mutex_unlock(&G.mx);
}

/* Blocks until every enqueued upload has COMPLETED (end of warmstart): the
 * engine frees the RAM int8 copies of the planned experts right after this
 * returns, so "dequeued" is not enough -- the uploader drops qn before it
 * calls the backend, and the last expert would still be queued=1 (#1360).
 * Must not be called with an expert group open: an LFRU swap parks the
 * uploader on issue_open until qt_take() clears it. */
void qt_fill_wait(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    while(G.inflight>0 && !G.th_stop) wait_take_locked();
    pthread_mutex_unlock(&G.mx);
}

/* Adaptive swap check (every 16 ticks = tokens): per device, coldest resident
 * vs hottest non-resident. Decay every 1024 ticks so an old workload cannot
 * permanently own the tier; admission uses the shared tier.h contract. */
/* G.mx held: remember an evicted slot for qt_evicted_take. */
static void evict_note_locked(int gi){
    if(!G.ev){ G.ev=malloc(QT_EV_CAP*sizeof(int)); if(!G.ev) return; G.ev_h=0; G.ev_n=0; }
    if(G.ev_n==QT_EV_CAP) return;                     /* full: the miss path rebuilds lazily */
    G.ev[(G.ev_h+G.ev_n)%QT_EV_CAP]=gi; G.ev_n++;
}
int qt_evicted_take(int *layers,int *eids,int max){
    if(!G.on||max<=0) return 0;
    pthread_mutex_lock(&G.mx);
    int n=0;
    while(n<max && G.ev_n>0){
        int gi=G.ev[G.ev_h]; G.ev_h=(G.ev_h+1)%QT_EV_CAP; G.ev_n--;
        layers[n]=gi/G.ne; eids[n]=gi%G.ne; n++;
    }
    pthread_mutex_unlock(&G.mx);
    return n;
}

/* G.mx held: start up to `max` pending re-plan swaps, as many as the queue
 * takes. Flags are re-checked -- the LFRU tick or an earlier drain may have
 * moved a slot since the plan was made. */
static void replan_drain_locked(int max){
    while(max-- > 0 && G.rp_i<G.rp_n && G.qn<QT_QCAP && !G.th_stop){
        int ci=G.rp_c[G.rp_i], vi=G.rp_v[G.rp_i]; G.rp_i++;
        QSlot *cs=&G.slot[ci], *vs=&G.slot[vi];
        if(cs->resident||cs->queued||!cs->g4) continue;
        if(!vs->resident||vs->queued) continue;
        res_set(vs,0);                                    /* CPU fallback from now on */
        if(enqueue_locked(ci/G.ne,ci%G.ne,vi/G.ne,vi%G.ne,0)){ G.swaps++; G.rp_done++; evict_note_locked(vi); }
        else { res_set(vs,1); G.rp_i--; break; }          /* no staging memory: retry later */
    }
    if(G.rp_i==G.rp_n){ free(G.rp_c); free(G.rp_v); G.rp_c=G.rp_v=NULL; G.rp_n=G.rp_i=0; }
}
#define QT_REPLAN_PER_TICK 4   /* a decode tick starts at most this many pending swaps: each swap is a
                                * cudaFree + cudaMalloc, which synchronise the device under the async groups */

static const uint32_t *g_rp_cnt; static int g_rp_base;   /* counts indexed by slot - base */
/* Both orders are total (the slot index breaks the last tie): qsort is not
 * stable, and glibc's and msvcrt's disagree on ties, so without it the same
 * counts would pick different victims on Linux and Windows. */
static int cmp_rp_cand(const void *a,const void *b){          /* count desc, index asc */
    int ia=*(const int*)a, ib=*(const int*)b;
    uint32_t fa=g_rp_cnt[ia-g_rp_base], fb=g_rp_cnt[ib-g_rp_base];
    if(fa!=fb) return fa<fb ? 1 : -1;
    return ia<ib ? -1 : ia>ib ? 1 : 0;
}
static int cmp_rp_vict(const void *a,const void *b){          /* count asc, heat asc, index asc */
    int ia=*(const int*)a, ib=*(const int*)b;
    uint32_t fa=g_rp_cnt[ia-g_rp_base], fb=g_rp_cnt[ib-g_rp_base];
    if(fa!=fb) return fa<fb ? -1 : 1;
    uint32_t ha=G.slot[ia].heat, hb=G.slot[ib].heat;
    if(ha!=hb) return ha<hb ? -1 : 1;
    return ia<ib ? -1 : ia>ib ? 1 : 0;
}
int qt_replan(int layer,const uint32_t *counts,int max_swaps){
    if(!G.on||!counts||G_stream||max_swaps<=0||layer>=G.nl) return 0;
    size_t lo = layer>=0 ? (size_t)layer*G.ne : 0, hi = layer>=0 ? lo+G.ne : (size_t)G.nl*G.ne;
    size_t n=hi-lo;
    int *cand=malloc(n*sizeof(int)), *vict=malloc(n*sizeof(int));
    if(!cand||!vict){ free(cand); free(vict); return 0; }
    pthread_mutex_lock(&G.mx);
    g_rp_cnt=counts; g_rp_base=(int)lo;
    int np=0, seen=0;
    int *pc=realloc(G.rp_c,(size_t)(G.rp_n+(int)n)*sizeof(int)), *pv=realloc(G.rp_v,(size_t)(G.rp_n+(int)n)*sizeof(int));
    if(!pc||!pv){ free(pc?pc:G.rp_c); free(pv?pv:G.rp_v); G.rp_c=G.rp_v=NULL; G.rp_n=G.rp_i=0;
                  pthread_mutex_unlock(&G.mx); free(cand); free(vict); return 0; }
    G.rp_c=pc; G.rp_v=pv;
    for(int di=0;di<G.ndev && np<max_swaps;di++){
        int nc=0, nv=0;
        for(size_t i=lo;i<hi;i++){
            QSlot *s=&G.slot[i]; int e=(int)(i%G.ne);
            if(home2((int)(i/G.ne),e)!=di || s->queued) continue;
            if(s->resident) vict[nv++]=(int)i;
            else if(s->g4 && counts[i-lo]>0) cand[nc++]=(int)i;
        }
        seen+=nc;
        qsort(cand,nc,sizeof(int),cmp_rp_cand);
        qsort(vict,nv,sizeof(int),cmp_rp_vict);
        if(G.wfmt!=QT_GGUF){
            for(int k=0;k<nc && k<nv && np<max_swaps;k++){
                if(counts[cand[k]-lo]<=counts[vict[k]-lo]) break;    /* the rest is not worth a swap */
                G.rp_c[G.rp_n+np]=cand[k]; G.rp_v[G.rp_n+np]=vict[k]; np++;
            }
        } else {
            /* a GGUF's: each candidate, hottest first, with the coldest unused victim of its
             * layer's types (swap_ok); a used victim is marked -1 */
            for(int k=0;k<nc && np<max_swaps;k++){
                int v=-1;
                for(int j=0;j<nv;j++) if(vict[j]>=0 && swap_ok(cand[k]/G.ne,vict[j]/G.ne)){ v=j; break; }
                if(v<0 || counts[cand[k]-lo]<=counts[vict[v]-lo]) continue;
                G.rp_c[G.rp_n+np]=cand[k]; G.rp_v[G.rp_n+np]=vict[v]; np++; vict[v]=-1;
            }
        }
    }
    G.rp_n+=np; G.rp_planned+=np;
    replan_drain_locked(QT_QCAP);                         /* start what the queue takes */
    pthread_mutex_unlock(&G.mx);
    free(cand); free(vict);
    if(layer<0) fprintf(stderr,"[qtier] replan: %d swaps planned from %d routed non-residents\n",np,seen);
    return np;
}

void qt_stats_mark(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    G.mk_hits=0; for(int i=0;i<G.ndev;i++) G.mk_hits+=G.hits[i];
    G.mk_miss=G.miss; G.mk_on=1;
    pthread_mutex_unlock(&G.mx);
}

static void qt_lfru_tick_locked(void){
    if(G_chain_own) return;   /* the chain's to move */
    size_t n=(size_t)G.nl*G.ne;
    G.tick++;
    if(G_stream){   /* copy mode: qt_note_wanted's bar */
        for(int di=0;di<G.ndev;di++) G_cold_heat[di]=0xFFFFFFFFu;
        for(size_t i=0;i<n;i++){
            QSlot *s=&G.slot[i]; int di=home2((int)(i/G.ne),(int)(i%G.ne));
            if(s->resident && !s->queued && s->heat<G_cold_heat[di]) G_cold_heat[di]=s->heat;
        }
    }
    if(!(G.tick%1024))
        for(size_t i=0;i<n;i++) G.slot[i].heat=tier_decay_value(G.slot[i].heat);
    if(!(G.tick%QT_CONV_DECAY))
        for(size_t i=0;i<n;i++) G.slot[i].conv>>=1;
    replan_drain_locked(QT_REPLAN_PER_TICK);              /* pending re-plan pairs, a few per token */
    if(G.tick%16) return;
    for(int di=0;di<G.ndev;di++){
        int cold=-1, hot=-1; uint32_t ch=0, hh=0;
        for(size_t i=0;i<n;i++){
            QSlot *s=&G.slot[i];
            int e=(int)(i%G.ne);
            if(home2((int)(i/G.ne),e)!=di) continue;
            if(!s->resident && !s->queued && s->g4){ if(hot<0||s->heat>hh){ hot=(int)i; hh=s->heat; } }
        }
        if(hot<0) continue;
        for(size_t i=0;i<n;i++){   /* the coldest resident the hottest may replace (swap_ok) */
            QSlot *s=&G.slot[i];
            int e=(int)(i%G.ne);
            if(home2((int)(i/G.ne),e)!=di || !swap_ok(hot/G.ne,(int)(i/G.ne))) continue;
            if(s->resident && !s->queued){ if(cold<0||s->heat<ch){ cold=(int)i; ch=s->heat; } }
        }
        if(cold<0) continue;
        if(!tier_should_promote(hh,ch)) continue;
        QSlot *v=&G.slot[cold];
        res_set(v,0);                                     /* CPU fallback from now on */
        if(enqueue_locked(hot/G.ne,hot%G.ne,cold/G.ne,cold%G.ne,0)){ G.swaps++; evict_note_locked(cold); }
        else res_set(v,1);                                /* queue full: revert */
    }
}

uint32_t qt_issue(int layer,const int *eids,int K,const float *x){
    if(!G.on||K>QT_MAX_ROWS||G_chain_own) return 0;
    uint32_t mask=0;
    ColiCudaTensor *tg[QT_MAX_DEV][QT_MAX_ROWS],*tu[QT_MAX_DEV][QT_MAX_ROWS],*td[QT_MAX_DEV][QT_MAX_ROWS];
    static int rows[QT_MAX_ROWS]={0};
    if(!rows[0]) for(int i=0;i<QT_MAX_ROWS;i++) rows[i]=1;
    for(int i=0;i<G.ndev;i++) G.is_cnt[i]=0;

    pthread_mutex_lock(&G.mx);
    /* QT_UPLOAD_SYNC=1: everything enqueued so far is resident before this
     * group is formed. Costs the upload/compute overlap, so it is for tests
     * and diagnostics: the fake-backend engine test asserts on residency and
     * hits after eight tokens, and on a two-vCPU runner the uploader thread
     * did not get scheduled once before the run was over (0 uploads, 0 hits,
     * six entries still queued). No group is open here, so the wait cannot
     * meet a swap parked on issue_open. */
    if(G_upload_sync) while(G.inflight>0 && !G.th_stop) wait_take_locked();
    if(G.th_stop){ pthread_mutex_unlock(&G.mx); return 0; }
    if(layer==0) qt_lfru_tick_locked();
    G.issue_open=1;
    for(int k=0;k<K;k++){
        QSlot *s=qs(layer,eids[k]);
        if(s->resident){
            /* a hit is routed heat too: the engine notes (qt_note) only what the
             * CPU computes, so without this a resident in copy mode never warmed
             * again and the coldest-resident test let any newcomer evict it */
            if(s->heat<0xFFFFFFFFu) s->heat++;
            int di=home2(layer,eids[k]); int c=G.is_cnt[di];
            tg[di][c]=s->tg; tu[di][c]=s->tu; td[di][c]=s->td;
            G.is_k[di][c]=k; G.is_cnt[di]=c+1;
            mask|=1u<<k; G.hits[di]++;
        } else G.miss++;
    }
    pthread_mutex_unlock(&G.mx);

    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        float *xr=G.is_x + (size_t)di*QT_MAX_ROWS*G.D;     /* per-device input block */
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        if(!coli_cuda_expert_group_issue(tg[di],tu[di],td[di],rows,c,xr)){
            /* issue failed -> hand these k back to the CPU */
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        }
    }
    return mask;
}

int qt_take(uint32_t mask,const float *val,int K,float *out){ return qt_take_scaled(mask,val,K,out,NULL,0.f); }
int qt_take_scaled(uint32_t mask,const float *val,int K,float *out,const float *s,float eps){
    (void)K;
    if(!G.on) return mask==0;
    const float *result[QT_MAX_DEV]={0};
    int ok=1;
    /* Drain every device before deciding whether this layer is usable. */
    if(mask) for(int di=0;di<G.ndev;di++){
        if(!G.is_cnt[di]) continue;
        result[di]=coli_cuda_expert_group_take(G.dev[di]);
        if(!result[di]){
            fprintf(stderr,"[qtier] dev %d: expert group result unavailable\n",G.dev[di]);
            ok=0;
        }
    }
    /* Do not publish a partial contribution when any device failed. */
    for(int di=0;di<G.ndev;di++){
        if(ok && result[di]) for(int j=0;j<G.is_cnt[di];j++){
            float w=val[G.is_k[di][j]];
            const float *row=result[di]+(size_t)j*G.D;
            if(s){   /* Muse Spark: w * rmsnorm(row * s) */
                double ms=0; for(int d=0;d<G.D;d++){ double v=(double)row[d]*s[d]; ms+=v*v; }
                const float r=1.f/sqrtf((float)(ms/G.D)+eps);
                for(int d=0;d<G.D;d++) out[d]+=w*(row[d]*s[d]*r);
                continue;
            }
            for(int d=0;d<G.D;d++) out[d]+=w*row[d];
        }
        G.is_cnt[di]=0;
    }
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
    return ok;
}

void qt_stats(void){
    if(!G.on) return;
    uint64_t hits=0; size_t res=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++) res += G.slot[i].resident;
    fprintf(stderr,"[qtier] resident %zu/%d experts | uploads %llu | miss(CPU) %llu | q_skips %llu\n",
            res, G.nl*G.ne, (unsigned long long)G.uploads,
            (unsigned long long)G.miss, (unsigned long long)G.q_full_skips);
    for(int i=0;i<G.ndev;i++){
        size_t tc=0,tb=0; coli_cuda_stats(G.dev[i],&tc,&tb);
        hits+=G.hits[i];
        /* tb counts every tensor on the device, trunk included; say how much of
         * it is trunk so "used > budget" does not read like an overrun. */
        fprintf(stderr,"[qtier]   dev %d: hits %llu | %zu tensors, %.2f GB VRAM used (%.2f GB trunk + experts, budget %.2f GB)\n",
                G.dev[i], (unsigned long long)G.hits[i], tc, tb/1073741824.0,
                G_trunk_bytes[i]/1073741824.0, G.budget[i]/1073741824.0);
    }
    double tot=(double)(hits+G.miss);
    fprintf(stderr,"[qtier] VRAM hit rate: %.1f %% | swaps %llu (%llu in place)\n",
            tot>0? 100.0*hits/tot : 0.0, (unsigned long long)G.swaps, (unsigned long long)G.inplace);
    if(G.mk_on){
        uint64_t dh=hits-G.mk_hits, dm=G.miss-G.mk_miss;
        fprintf(stderr,"[qtier] decode VRAM hit rate: %.1f %% (%llu hits, %llu misses after the prefill)\n",
                dh+dm? 100.0*dh/(dh+dm) : 0.0, (unsigned long long)dh, (unsigned long long)dm);
    }
    if(G.rp_planned)
        fprintf(stderr,"[qtier] replan: %llu swaps planned, %llu done, %d still pending\n",
                (unsigned long long)G.rp_planned, (unsigned long long)G.rp_done, G.rp_n-G.rp_i);
    { uint64_t calls=0,ex=0,rows=0; double h2d=0,kms=0,d2h=0;
      coli_cuda_group_stats(&calls,&ex,&rows,&h2d,&kms,&d2h);
      if(calls) fprintf(stderr,"[qtier] group_stats: %llu calls, %llu experts | h2d %.0f ms, kernel %.0f ms, d2h %.0f ms\n",
              (unsigned long long)calls,(unsigned long long)ex,h2d,kms,d2h); }
}

static void dense_free_all(void){
    dn_gpu_free_all();
    for(int h = 0; h < G_dense_n; h++){ if(G_dense[h].t) coli_cuda_tensor_free(G_dense[h].t); G_dense[h].t = NULL; G_dense[h].on = 0; }
    G_dense_n = 0;
    if(G_lmh.t) coli_cuda_tensor_free(G_lmh.t);
    memset(&G_lmh,0,sizeof G_lmh);
    for(int l=0;l<QT_DN_MAX_LAYERS;l++)
        if(G_dnp[l].t) coli_cuda_tensor_free(G_dnp[l].t);
    memset(G_dnp,0,sizeof G_dnp);
}
void qt_shutdown(void){
    dense_free_all();
    if(!G.on) return;
    free(G.rp_c); free(G.rp_v); G.rp_c=G.rp_v=NULL; G.rp_n=G.rp_i=0;
    free(G.ev); G.ev=NULL; G.ev_n=0;
    free(G.pm); G.pm=NULL; G.pm_n=G.pm_cap=0; G.chg_n=G.chg_over=G.chg_on=0;
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"wb");
        if(f){
            uint32_t hdr[3]={0x51544831u,(uint32_t)G.nl,(uint32_t)G.ne};
            fwrite(hdr,4,3,f);
            for(size_t i=0;i<(size_t)G.nl*G.ne;i++) fwrite(&G.slot[i].heat,4,1,f);
            fclose(f);
            fprintf(stderr,"[qtier] HEAT_FILE saved: %s\n",hf);
        }
    }
    /* Wake cv_take too: the uploader's LFRU victim wait (and qt_note_block /
     * qt_note_planned / qt_fill_wait, all waiting on the same condvar) would
     * otherwise never notice th_stop and pthread_join below would hang (#1340). */
    pthread_mutex_lock(&G.mx); G.th_stop=1; pthread_cond_signal(&G.cv); pthread_cond_broadcast(&G.cv_take); pthread_mutex_unlock(&G.mx);
    pthread_join(G.th,NULL);
    pthread_mutex_lock(&G.mx);
    while(G.waiters) pthread_cond_wait(&G.cv_take,&G.mx);
    pthread_mutex_unlock(&G.mx);
    /* The uploader is stopped, but a decode group may still own the tensors. */
    for(int di=0;di<G.ndev;di++)
        if(G.is_cnt[di]) coli_cuda_expert_group_take(G.dev[di]);
    if(G_host_n) fprintf(stderr,"[qtier] host pool: %llu experts (%.2f GB of the RAM cache) page-locked for the device's copies\n",
                         G_host_n, G_host_bytes/1073741824.0);
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++){
        QSlot *s=&G.slot[i];
        if(s->hreg){ coli_cuda_host_unregister((void *)s->g4); coli_cuda_host_unregister((void *)s->gs); s->hreg=0; }
        if(s->tg) coli_cuda_tensor_free(s->tg);
        if(s->tu) coli_cuda_tensor_free(s->tu);
        if(s->td) coli_cuda_tensor_free(s->td);
        s->aslot=0;
    }
    for(int i=0;i<QT_MAX_DEV;i++){   /* the arenas, once no tensor points into them */
        if(G_arena[i]) coli_cuda_pipe_free(G.dev[i],G_arena[i]);
        G_arena[i]=NULL; free(G_afree[i]); G_afree[i]=NULL; G_aslots[i]=G_anfree[i]=0;
    }
    G_arena_on=0; G_chain_own=0;
    if(G_cfree) for(int c=0;c<G_ncls;c++) free(G_cfree[c]);
    free(G_cfree); G_cfree=NULL; free(G_cnfree); G_cnfree=NULL; free(G_ccap); G_ccap=NULL;
    free(G_lsl); G_lsl=NULL; free(G_lou); G_lou=NULL; free(G_lod); G_lod=NULL; free(G_lcls); G_lcls=NULL;
    G_exact=0; G_ncls=0; G_abump=0;
    free(G.slot); G.slot=NULL;
    free(G.is_x); G.is_x=NULL; G.is_x_floats=0;
    free(G.fill_order); G.fill_order=NULL;
    free(G.off); G.off=NULL;
    free(G.heat0); G.heat0=NULL;
    pthread_cond_destroy(&G.cv_take);
    pthread_cond_destroy(&G.cv);
    pthread_mutex_destroy(&G.mx);
    G.issue_open=0;
    memset(G.is_cnt,0,sizeof G.is_cnt);
    G.on=0;
    G_fp8_stream=0; G_stream=0; G_trunk_only=0; G_src_planar=0;
    G_host_on=-1; G_host_n=G_host_bytes=0;
    coli_cuda_shutdown();
}

#endif /* COLI_CUDA */
