/* qwen36_tier.c -- CUDA VRAM expert tier for the qwen36 engine. See header. */
#ifdef COLI_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#endif
#include "qwen36_tier.h"
#include "backend_cuda.h"
#include "tier.h"

#define QT_MAX_DEV 8
#define QT_QCAP 48            /* upload queue depth (staging ~1.6 MB/entry) */

typedef struct {
    ColiCudaTensor *tg, *tu, *td;
    uint32_t heat;
    uint8_t resident, queued, planned;
    /* raw RAM pointers (slots are never evicted when cap==n_experts) -- lets
     * warmstart, lookahead and LFRU swaps run without an engine callback */
    const uint8_t *g4,*u4,*d4; const float *gs,*us,*ds;
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
    int th_stop;
    /* upload ring with staging copies */
    struct { int layer, eid; uint8_t *w; float *s; int v_layer, v_eid; } q[QT_QCAP];
    int qh, qt_, qn;
    pthread_cond_t cv;
    /* statistics */
    uint64_t hits[QT_MAX_DEV], miss, uploads, q_full_skips;
    /* issue state of the (single) decode thread */
    int is_cnt[QT_MAX_DEV];
    int is_k[QT_MAX_DEV][32];
    float *is_x; size_t is_x_floats;      /* 32*D input replicas per device */
    /* M3 */
    int *fill_order; int fill_cur;        /* warmstart order (heat desc) */
    int issue_open;                       /* guard: no tensor_free while a group is in flight */
    pthread_cond_t cv_take;               /* signals qt_take done + queue space */
    uint64_t tick, swaps, pf_hits, pf_notes;
    uint32_t *heat0;                      /* heat table loaded from HEAT_FILE */
} G;

static QSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
static int home(int eid){ return eid % G.ndev; }

/* Staging: packed int4 (g|u|d) two's-complement -> offset-binary (XOR 0x88,
 * the upload format of backend_cuda fmt=2) + copy the scales (gs|us|ds). */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
                  const float *gs,const float *us,const float *ds){
    size_t mb = (size_t)G.D*G.Ih/(G.wfmt==4?2:1);
    if(G.wfmt==1 || G.wfmt==8){
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
        if(ve>=0){
            /* LFRU swap: free the victim only when no group is in flight */
            while(G.issue_open && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
            QSlot *v=qs(vl,ve);
            if(G.th_stop && G.issue_open){
                /* Shutting down with a group still open: qt_take() -- the only
                 * thing that clears issue_open -- will never come. Abandon
                 * this swap instead of freeing a victim tensor the in-flight
                 * group may still reference; qt_lfru_tick_locked already
                 * cleared the victim's resident flag before enqueueing, so
                 * restore it to keep the flag consistent with the tensor it
                 * still holds. */
                v->resident=1; qs(layer,eid)->queued=0;
                pthread_mutex_unlock(&G.mx); free(w); free(sc); continue;
            }
            ColiCudaTensor *a=v->tg,*b=v->tu,*ct=v->td;
            v->tg=v->tu=v->td=NULL;
            pthread_mutex_unlock(&G.mx);
            if(a)coli_cuda_tensor_free(a); if(b)coli_cuda_tensor_free(b); if(ct)coli_cuda_tensor_free(ct);
        } else pthread_mutex_unlock(&G.mx);

        int dv = G.dev[home(eid)];
        /* passo fra le tre matrici nello staging: int4 impacchettato = mezzo
         * byte per elemento, int8 = uno. */
        size_t mb=(size_t)G.D*G.Ih/(G.wfmt==4?2:1);
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok;
        if(G.wfmt==8){
            /* e4m3 bytes as they came from the checkpoint, block scales
             * [ceil(O/128), ceil(I/128)] per matrix -- the layout #817's
             * kernels and tensor_upload(fmt=8) already agree on */
            ok = coli_cuda_tensor_upload(&tg, w,      sc,            8, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+mb,   sc+G.sc_gu,    8, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*mb, sc+2*G.sc_gu,  8, G.Ih, G.D,  dv);
        } else if(G.wfmt==1){
            /* int8, scale per riga: qt_init ha gia' rifiutato il caso raggruppato,
             * che questo formato non sa esprimere. */
            ok = coli_cuda_tensor_upload(&tg, w,      sc,          1, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+mb,   sc+G.Ih,     1, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*mb, sc+2*G.Ih,   1, G.Ih, G.D,  dv);
        } else if(G.egs){
            ok = coli_cuda_tensor_upload_g(&tg, w,      sc,             4, G.D,  G.Ih, dv, G.egs)
              && coli_cuda_tensor_upload_g(&tu, w+mb,   sc+G.sc_gu,     4, G.D,  G.Ih, dv, G.egs)
              && coli_cuda_tensor_upload_g(&td, w+2*mb, sc+2*G.sc_gu,   4, G.Ih, G.D,  dv, G.egs);
        } else {
            ok = coli_cuda_tensor_upload(&tg, w,      sc,          2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+mb,   sc+G.Ih,     2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*mb, sc+2*G.Ih,   2, G.Ih, G.D,  dv);
        }
        free(w); free(sc);
        pthread_mutex_lock(&G.mx);
        QSlot *s=qs(layer,eid);
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
        else  { int hd=home(eid); G.used[hd]-=G.exp_bytes;
                G.budget[hd]=G.used[hd];   /* device genuinely full: stop trying */ }
        s->queued=0;
        pthread_mutex_unlock(&G.mx);
    }
}

/* R4 role split: lm_head as a resident int8 tensor on its own device.
 * The dense-i8 quantization (engine-side) provides q/sc with the same
 * per-row semantics quant_matmul's fmt=1 applies (y[o] = acc * sc[o]),
 * so CPU and GPU compute the same numbers up to accumulation order. */
static struct { ColiCudaTensor *t; int dev, dev_ok, on; } G_lmh;

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
static struct { ColiCudaTensor *t; int dev, on; } G_dnp[QT_DN_MAX_LAYERS];

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
static struct { char name[16]; int layer; size_t bytes; } G_offer[QT_OFFER_MAX];
static int G_offer_n;
static int G_auto_on;                                  /* auto placement decided */
static int G_auto_lmh = QT_PLACE_CPU;
static int G_auto_dnp[QT_DN_MAX_LAYERS];               /* per layer, or QT_PLACE_CPU */
static size_t G_trunk_bytes[QT_MAX_DEV];               /* placed trunk per device index */

int qt_place_of(const char *component, int layer){
    if(G_auto_on){
        if(!strcmp(component, "lmhead")) return G_auto_lmh;
        if(!strcmp(component, "dnproj"))
            return (layer >= 0 && layer < QT_DN_MAX_LAYERS) ? G_auto_dnp[layer] : QT_PLACE_CPU;
        return QT_PLACE_CPU;               /* experts follow COLI_GPUS; dnout/attnproj not yet placed */
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
    G_offer[G_offer_n].layer = layer; G_offer[G_offer_n].bytes = bytes;
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
    for(int e = 0; e < ne; e++) if(e % G.ndev == di) homed++;
    homed *= (size_t)nl;
    size_t fit = room / exp_bytes;
    if(fit >= homed){ *p_marginal_out = 0; return 0; }   /* room to spare: displaces nothing */
    if(k <= 0){ *p_marginal_out = 0; return 0; }
    double cpu_factor = (G.wfmt == 1) ? 1.0 : 2.0;      /* CPU reads the int8 slot */
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
            if(e % G.ndev != di) continue;
            double pe = sum > 0 ? (double)topk * heat0[(size_t)l*ne + e] / sum : (double)topk / ne;
            p[m++] = pe > 1.0 ? 1.0 : pe;
        }
    }
    qsort(p, m, sizeof *p, cmp_double_desc);
    double value = 0, pm = 0; int cnt = 0;
    for(size_t r = (fit > (size_t)k ? fit - k : 0); r < fit && r < m; r++){ value += cpu_factor * p[r] * (double)exp_bytes; pm = p[r]; cnt++; }
    free(p);
    *p_marginal_out = pm;
    return value;
}

static void auto_place(int nl, int ne, int topk, const size_t *capacity, const uint32_t *heat0){
    size_t room[QT_MAX_DEV];
    for(int i = 0; i < G.ndev; i++){ room[i] = capacity[i]; G_trunk_bytes[i] = 0; }
    for(int l = 0; l < QT_DN_MAX_LAYERS; l++) G_auto_dnp[l] = QT_PLACE_CPU;
    G_auto_lmh = QT_PLACE_CPU;
    int placed = 0, kept = 0;
    /* lmhead first (one call per token, latency-tolerant), then the
     * projections in layer order */
    for(int pass = 0; pass < 2; pass++)
        for(int o = 0; o < G_offer_n; o++){
            int is_lmh = !strcmp(G_offer[o].name, "lmhead");
            if((pass == 0) != is_lmh) continue;
            if(!is_lmh && strcmp(G_offer[o].name, "dnproj")) continue;   /* v1: these two */
            size_t bytes = G_offer[o].bytes;
            int di = 0;
            for(int i = 1; i < G.ndev; i++) if(room[i] > room[di]) di = i;
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
            if(is_lmh) G_auto_lmh = G.dev[di]; else G_auto_dnp[G_offer[o].layer] = G.dev[di];
            room[di] -= bytes; G_trunk_bytes[di] += bytes; placed++;
        }
    G_auto_on = 1;
    for(int i = 0; i < G.ndev; i++)
        fprintf(stderr,"[place] auto: dev %d holds %.1f MB of trunk (lmhead%s, %d dnproj layers), %.2f GB left for experts\n",
                G.dev[i], G_trunk_bytes[i]/1048576.0, G_auto_lmh == G.dev[i] ? " yes" : " no",
                auto_dnp_count(G.dev[i]), room[i]/1073741824.0);
    (void)placed; (void)kept;
}


/* ---- fp8 streaming mode (Qwen3.8) -----------------------------------------
 * qwen36 keeps every expert in RAM and lets the tier retain raw pointers into
 * slots that are never recycled; that is what `cap == n_experts` guards. A
 * model whose experts do not fit in RAM (Qwen3.8: 24 576 x 4.7 MiB) streams
 * them through an LRU whose slots ARE recycled, so a retained pointer would
 * dangle by the next token. In this mode the tier owns what it uploads: the
 * bytes are copied into the staging buffer inside the qt_note call, while the
 * engine's slot is still live, and the pointers are dropped right after. A
 * promotion can therefore only happen when the bytes pass by -- the LFRU
 * decision moves from the periodic tick into qt_note, which asks: is this
 * expert, now in hand, hotter than the coldest resident on its device? */
static int G_fp8_stream;
static const float *G_fp8_lut;

int qt_init_fp8(int nl, int ne, int D, int Ih, int cap, int topk, const float *e4m3_lut){
    G_fp8_stream = 1; G_fp8_lut = e4m3_lut;
    int ok = qt_init(nl, ne, D, Ih, cap, topk, 0, 0);
    if(!ok) G_fp8_stream = 0;
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
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(cap != ne && !G_fp8_stream){
        fprintf(stderr,"[qtier] cap=%d != n_experts=%d -> tier disabled (needs full RAM residency)\n",cap,ne);
        return 0;
    }
    if(topk>32){ fprintf(stderr,"[qtier] topk>32 unsupported\n"); return 0; }
    memset(&G,0,sizeof G);
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
        static const char *comps[] = {"lmhead","dnproj","dnout","attnproj"};
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
    G.wfmt = G_fp8_stream ? 8 : (expert_is_int4 ? 4 : 1);
    if(G.wfmt==1 && expert_gs>0){
        fprintf(stderr,"[qtier] int8 experts with grouped scales (gs=%d) cannot be "
                       "expressed on the GPU (fmt=1 is per-row only) -> CPU path\n", expert_gs);
        return 0;
    }
    G.egs = expert_gs;
    if(G.wfmt==8){
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
    size_t scl_bytes = (2*G.sc_gu+G.sc_d)/3*sizeof(float);
    G.exp_bytes = 3*dev_alloc_footprint(mat_bytes) + 3*dev_alloc_footprint(scl_bytes); /* + allocation slack */

    /* Per-device allowance for tier + trunk: CUDA_EXPERT_GB when numeric,
     * else free minus 1 GB headroom. The heat table is loaded here too (it
     * used to be loaded after the budgets) because the placer prices
     * experts by heat. */
    size_t capacity[QT_MAX_DEV]; int capdev[QT_MAX_DEV]; int ncap = G.ndev;
    const char *bg=getenv("CUDA_EXPERT_GB");
    for(int i=0;i<G.ndev;i++){
        size_t freeb=0,totb=0; coli_cuda_mem_info(G.dev[i],&freeb,&totb);
        capdev[i] = G.dev[i];
        capacity[i] = (bg && strcmp(bg,"auto") && atof(bg)>0)
                   ? (size_t)(atof(bg)*1024.0*1024.0*1024.0)
                   : (freeb>(1ull<<30) ? freeb-(1ull<<30) : 0);
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
        static const char *comps[] = {"dnproj","dnout","attnproj"};
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
        if(ed!=QT_PLACE_CPU){
            int present=0; for(int i=0;i<G.ndev;i++) if(G.dev[i]==ed) present=1;
            if(present){
                G.dev[0]=ed; G.ndev=1;
                fprintf(stderr,"[place] Experten auf Device %d festgelegt\n",ed);
            } else fprintf(stderr,"[place] experts=%d nicht verfuegbar -> COLI_GPUS bleibt\n",ed);
        } else if(!G_auto_on && G_place_n && qt_place_named("experts")){
            fprintf(stderr,"[place] experts=cpu -> VRAM-Tier aus\n");
            return 0;
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
        G.budget[i] = cap_i > trunk ? cap_i - trunk : 0;
        fprintf(stderr,"[qtier] dev %d: budget %.2f GB for experts (~%zu experts)%s\n",
                G.dev[i], G.budget[i]/1073741824.0, G.budget[i]/G.exp_bytes,
                trunk ? " after trunk" : "");
    }
    G.is_x_floats=(size_t)G.ndev*32*D;
    G.is_x=malloc(G.is_x_floats*sizeof(float));
    if(!G.is_x) return 0;
    pthread_mutex_init(&G.mx,NULL); pthread_cond_init(&G.cv,NULL); pthread_cond_init(&G.cv_take,NULL);
    qt_aff_get(&aff); qt_aff_widen(&aff);                  /* the uploader inherits this mask */
    int th_ok=pthread_create(&G.th,NULL,uploader,NULL)==0;
    qt_aff_restore(&aff);
    if(!th_ok) return 0;
    G.on=1;
    fprintf(stderr,"[qtier] CUDA VRAM expert tier active: %d device(s), %.2f MB/expert\n",
            G.ndev, G.exp_bytes/1048576.0);
    return 1;
}

int qt_ready(void){ return G.on; }

int qt_lmhead_init(const int8_t *q, const float *sc, int I, int O){
    if(!G_lmh.dev_ok||!G.on||!q||!sc) return 0;
    int dev=G_lmh.dev;
    if(!coli_cuda_tensor_upload(&G_lmh.t,q,sc,1,I,O,dev)){
        fprintf(stderr,"[lmh] lm_head upload failed -> stays on CPU\n");
        return 0;
    }
    G_lmh.dev=dev; G_lmh.on=1;
    fprintf(stderr,"[lmh] lm_head [%d x %d] int8 resident on CUDA dev %d (%.2f GB)\n",
            O,I,dev,(double)O*I/1073741824.0);
    return 1;
}

int qt_dnproj_init(int layer, const int8_t *q, const float *sc,
                   int I, int O, int device){
    if(layer < 0 || layer >= QT_DN_MAX_LAYERS) return 0;
    if(device == QT_PLACE_CPU || !q || !sc) return 0;
    /* No G.on requirement: the projections are independent of the expert tier,
     * so they can be measured on a card that holds no experts at all. */
    if(!coli_cuda_tensor_upload(&G_dnp[layer].t, q, sc, 1, I, O, device)){
        fprintf(stderr,"[dnp] layer %d upload failed -> stays on CPU\n", layer);
        return 0;
    }
    G_dnp[layer].dev = device; G_dnp[layer].on = 1;
    return 1;
}

int qt_dnproj_matmul(int layer, float *y, const float *x, int I, int O){
    if(layer < 0 || layer >= QT_DN_MAX_LAYERS || !G_dnp[layer].on) return 0;
    if(coli_cuda_matmul(&G_dnp[layer].t,y,x,NULL,NULL,1,1,I,O,G_dnp[layer].dev,0))
        return 1;
    fprintf(stderr,"[dnp] layer %d GPU matmul failed; CPU from here on\n", layer);
    G_dnp[layer].on = 0;
    return 0;
}

int qt_lmhead_matmul(float *y, const float *x, int I, int O){
    if(!G_lmh.on) return 0;
    /* cached-tensor path: upload params are ignored once *t exists */
    if(coli_cuda_matmul(&G_lmh.t,y,x,NULL,NULL,1,1,I,O,G_lmh.dev,0)) return 1;
    fprintf(stderr,"[lmh] GPU matmul failed; falling back to CPU from here on\n");
    G_lmh.on=0;
    return 0;
}

/* Is (layer,eid) currently VRAM-resident? (used to free RAM-side int8 copies) */
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
    int hd=home(eid);
    if(!reserved && v_eid<0 && G.used[hd]+G.exp_bytes>G.budget[hd]) return 0;
    size_t mb=(size_t)G.D*G.Ih/(G.wfmt==4?2:1);   /* buffer di staging: int8/fp8 = 1 byte/elemento */
    uint8_t *w=malloc(3*mb); float *sc=malloc((2*G.sc_gu+G.sc_d)*sizeof(float));
    if(!w||!sc){ free(w); free(sc); return 0; }
    if(!reserved && v_eid<0) G.used[hd]+=G.exp_bytes;
    s->queued=1;
    stage(w,sc,s->g4,s->u4,s->d4,s->gs,s->us,s->ds);
    G.q[G.qt_].layer=layer; G.q[G.qt_].eid=eid; G.q[G.qt_].w=w; G.q[G.qt_].s=sc;
    G.q[G.qt_].v_layer=v_layer; G.q[G.qt_].v_eid=v_eid;
    G.qt_=(G.qt_+1)%QT_QCAP; G.qn++;
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
    int hd=home(eid);
    if(G.used[hd]+G.exp_bytes<=G.budget[hd]){ enqueue_locked(layer,eid,-1,-1,0); return; }
    size_t n=(size_t)G.nl*G.ne; int cold=-1; uint32_t ch=0;
    for(size_t i=0;i<n;i++){
        QSlot *c=&G.slot[i];
        if(home((int)(i%G.ne))!=hd || !c->resident || c->queued) continue;
        if(cold<0||c->heat<ch){ cold=(int)i; ch=c->heat; }
    }
    if(cold<0 || !tier_should_promote(s->heat,ch)) return;
    QSlot *v=&G.slot[cold];
    v->resident=0;
    if(enqueue_locked(layer,eid,cold/G.ne,cold%G.ne,0)) G.swaps++;
    else v->resident=1;
}

void qt_note(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(G_fp8_stream){
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

/* blocking variant for the warmstart (waits for queue space). */
void qt_note_block(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(G_fp8_stream) stream_point(s,g4,u4,d4,gs,us,ds);
    else if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    enqueue_locked(layer,eid,-1,-1,0);
    if(G_fp8_stream) stream_forget(s);
    pthread_mutex_unlock(&G.mx);
}

/* warmstart order -- heat descending (HEAT_FILE) or natural order.
 * Returns 0 once all budgets are full or the list is exhausted. */
static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    uint32_t ha=g_sort_heat[*(const int*)a], hb=g_sort_heat[*(const int*)b];
    return ha<hb ? 1 : ha>hb ? -1 : 0;
}
int qt_fill_next(int *layer,int *eid){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order){
        G.fill_order=malloc(n*sizeof(int));
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        if(G.heat0){ g_sort_heat=G.heat0; qsort(G.fill_order,n,sizeof(int),cmp_heat_desc); }
        G.fill_cur=0;
    }
    while((size_t)G.fill_cur<n){
        int gi=G.fill_order[G.fill_cur];
        int l=gi/G.ne, e=gi%G.ne, hd=home(e);
        QSlot *s=qs(l,e);
        int full=1; for(int i=0;i<G.ndev;i++) if(G.used[i]+G.exp_bytes<=G.budget[i]) full=0;
        if(full){ pthread_mutex_unlock(&G.mx); return 0; }
        G.fill_cur++;
        if(s->resident||s->queued) continue;
        if(G.used[hd]+G.exp_bytes>G.budget[hd]) continue;   /* dieses Device voll */
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
int qt_plan_fill(int *layers,int *eids,int max){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    int cnt=0;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order){
        G.fill_order=malloc(n*sizeof(int));
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        if(G.heat0){ g_sort_heat=G.heat0; qsort(G.fill_order,n,sizeof(int),cmp_heat_desc); }
        G.fill_cur=0;
    }
    while((size_t)G.fill_cur<n && cnt<max){
        int full=1; for(int i=0;i<G.ndev;i++) if(G.used[i]+G.exp_bytes<=G.budget[i]) full=0;
        if(full) break;
        int gi=G.fill_order[G.fill_cur++];
        int l=gi/G.ne, e=gi%G.ne, hd=home(e);
        QSlot *s=qs(l,e);
        if(s->resident||s->queued||s->planned) continue;
        if(G.used[hd]+G.exp_bytes>G.budget[hd]) continue;
        G.used[hd]+=G.exp_bytes;          /* reserve */
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
    if(!G.on) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!g4){
        /* The loader had nothing to hand over. qt_plan_fill reserved budget
         * and set planned=1 for this expert; returning here without undoing
         * both keeps the bytes out of the budget for the life of the process
         * and "if(resident||queued||planned) continue" never reconsiders the
         * expert. #1331 was this leak for every expert of an int8 container. */
        if(s->planned){ G.used[home(eid)]-=G.exp_bytes; s->planned=0; }
        pthread_mutex_unlock(&G.mx);
        return;
    }
    if(G_fp8_stream) stream_point(s,g4,u4,d4,gs,us,ds);
    else if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    if(!enqueue_locked(layer,eid,-1,-1,1)){
        /* not enqueueable (e.g. already resident): return the reservation */
        if(s->planned) G.used[home(eid)]-=G.exp_bytes;
    }
    s->planned=0;
    if(G_fp8_stream) stream_forget(s);
    pthread_mutex_unlock(&G.mx);
}

/* waits until the upload queue is drained (end of warmstart). */
void qt_fill_wait(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    while(G.qn>0 && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    pthread_mutex_unlock(&G.mx);
}

/* Adaptive swap check (every 16 ticks = tokens): per device, coldest resident
 * vs hottest non-resident. Decay every 1024 ticks so an old workload cannot
 * permanently own the tier; admission uses the shared tier.h contract. */
static void qt_lfru_tick_locked(void){
    size_t n=(size_t)G.nl*G.ne;
    G.tick++;
    if(!(G.tick%1024))
        for(size_t i=0;i<n;i++) G.slot[i].heat=tier_decay_value(G.slot[i].heat);
    if(G.tick%16) return;
    for(int di=0;di<G.ndev;di++){
        int cold=-1, hot=-1; uint32_t ch=0, hh=0;
        for(size_t i=0;i<n;i++){
            QSlot *s=&G.slot[i];
            int e=(int)(i%G.ne);
            if(home(e)!=di) continue;
            if(s->resident && !s->queued){ if(cold<0||s->heat<ch){ cold=(int)i; ch=s->heat; } }
            else if(!s->resident && !s->queued && s->g4){ if(hot<0||s->heat>hh){ hot=(int)i; hh=s->heat; } }
        }
        if(cold<0||hot<0) continue;
        if(!tier_should_promote(hh,ch)) continue;
        QSlot *v=&G.slot[cold];
        v->resident=0;                                    /* CPU fallback from now on */
        if(enqueue_locked(hot/G.ne,hot%G.ne,cold/G.ne,cold%G.ne,0)) G.swaps++;
        else v->resident=1;                               /* queue full: revert */
    }
}

uint32_t qt_issue(int layer,const int *eids,int K,const float *x){
    if(!G.on||K>32) return 0;
    uint32_t mask=0;
    ColiCudaTensor *tg[QT_MAX_DEV][32],*tu[QT_MAX_DEV][32],*td[QT_MAX_DEV][32];
    static int rows[32]={0};
    if(!rows[0]) for(int i=0;i<32;i++) rows[i]=1;
    for(int i=0;i<G.ndev;i++) G.is_cnt[i]=0;

    pthread_mutex_lock(&G.mx);
    if(layer==0) qt_lfru_tick_locked();
    G.issue_open=1;
    for(int k=0;k<K;k++){
        QSlot *s=qs(layer,eids[k]);
        if(s->resident){
            int di=home(eids[k]); int c=G.is_cnt[di];
            tg[di][c]=s->tg; tu[di][c]=s->tu; td[di][c]=s->td;
            G.is_k[di][c]=k; G.is_cnt[di]=c+1;
            mask|=1u<<k; G.hits[di]++;
        } else G.miss++;
    }
    pthread_mutex_unlock(&G.mx);

    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        float *xr=G.is_x + (size_t)di*32*G.D;              /* per-device input block */
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        if(!coli_cuda_expert_group_issue(tg[di],tu[di],td[di],rows,c,xr)){
            /* issue failed -> hand these k back to the CPU */
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        }
    }
    return mask;
}

void qt_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    if(!G.on) return;
    if(mask) for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        const float *y=coli_cuda_expert_group_take(G.dev[di]);
        if(!y) continue;
        for(int j=0;j<c;j++){
            float w=val[G.is_k[di][j]];
            const float *row=y+(size_t)j*G.D;
            for(int d=0;d<G.D;d++) out[d]+=w*row[d];
        }
        G.is_cnt[di]=0;
    }
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
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
    fprintf(stderr,"[qtier] VRAM hit rate: %.1f %% | LFRU swaps %llu\n",
            tot>0? 100.0*hits/tot : 0.0, (unsigned long long)G.swaps);
    { uint64_t calls=0,ex=0,rows=0; double h2d=0,kms=0,d2h=0;
      coli_cuda_group_stats(&calls,&ex,&rows,&h2d,&kms,&d2h);
      if(calls) fprintf(stderr,"[qtier] group_stats: %llu calls, %llu experts | h2d %.0f ms, kernel %.0f ms, d2h %.0f ms\n",
              (unsigned long long)calls,(unsigned long long)ex,h2d,kms,d2h); }
}

void qt_shutdown(void){
    if(!G.on) return;
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
    G.on=0;
    G_fp8_stream=0;
    coli_cuda_shutdown();
}

#endif /* COLI_CUDA */
