/* qwen36_tier.c -- CUDA VRAM expert tier for the qwen36 engine. See header. */
#ifdef COLI_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
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
    int issue_open;                       /* guard: no tensor_free while a group is in flight */
    int blocking_calls;                    /* note/fill callers that shutdown must wake and join */
    pthread_cond_t cv_take;               /* signals qt_take done + queue space */
    uint64_t tick, swaps, pf_hits, pf_notes;
    uint32_t *heat0;                      /* heat table loaded from HEAT_FILE */
} G;

#ifdef QT_TEST_HOOKS
/* Test-only gate, called on the wake and BEFORE the waiter leaves G.waiters.
 * The drain in qt_shutdown is only observable if a test can hold an awakened
 * caller inside the window, and once woken its whole remaining path is a
 * mutex hand-back -- there is no seam to widen it from the outside. Compiled
 * out of production objects, like COLI_V4_TEST_HOOKS in deepseek_v4.c; the
 * hook must return with G.mx held. */
void (*qt_test_take_wake_hook)(void);
#define QT_TAKE_WAKE_HOOK() (qt_test_take_wake_hook ? qt_test_take_wake_hook() : (void)0)
/* Test-only seams either side of the G.waiters drain in qt_shutdown, both
 * with G.mx held. The claim under test -- teardown must not begin while a
 * woken caller is still inside qt_fill_wait -- is a NEGATIVE one, and a
 * timeout can never prove it: delay the shutdown thread past the poll and a
 * deleted drain still reports success (#1564). So both ends of the drain
 * report themselves instead.
 *
 * WAIT fires from inside `while(G.waiters)`, which the loop only reaches when
 * it has found a caller to wait for: a shutdown thread that never runs it has
 * deleted the drain, and its silence is the signal a test can act on.
 * LEAVE fires after the drain and before any teardown, with G.mx held, so a
 * test reads G.waiters on the shutdown thread itself -- no wall clock
 * anywhere, and no scheduling delay can make a teardown that never waited
 * look like one that did.
 *
 * Compiled out of production objects like the hook above; each must return
 * with G.mx held. */
void (*qt_test_drain_wait_hook)(void);
void (*qt_test_drain_leave_hook)(void);
#define QT_DRAIN_WAIT_HOOK() (qt_test_drain_wait_hook ? qt_test_drain_wait_hook() : (void)0)
#define QT_DRAIN_LEAVE_HOOK() (qt_test_drain_leave_hook ? qt_test_drain_leave_hook() : (void)0)

/* Test-only, on the teardown thread, and both ends of the exclusive section
 * below, so "the teardown waited" and "the teardown freed" are read where they
 * are decided instead of being inferred from a wall clock.
 *
 * BLOCK fires on the teardown thread at the moment it is about to WAIT for the
 * callers still inside the tier, with qg_mx held. A teardown that has nothing
 * to wait for -- which is exactly the reader defect, where entering the tier is
 * not counted at all -- never fires it, and that silence is the signal.
 * FREED fires on the same thread once the storage is gone.
 *
 * READER_PRE fires inside a qt_* entry point that has already been admitted
 * and has not touched the storage yet: the window an `if(!G.on)` test followed
 * by a lock leaves open. It runs with the gate HELD and G.mx NOT held, and it
 * must not touch G. */
void (*qt_test_gate_block_hook)(void);
void (*qt_test_gate_freed_hook)(void);
void (*qt_test_reader_pre_hook)(void);
/* Test-only, inside qt_lmhead_init, on the last line before the lifecycle
 * admission -- the window a pre-gate read of G_lmh.dev_ok leaves open, and the
 * one seam wide enough to walk a caller through a shutdown and a re-init and
 * land it in a generation that never asked for what it is about to install.
 * Runs with no lock held and must touch no state. */
void (*qt_test_lmhead_pre_gate_hook)(void);
#define QT_GATE_BLOCK_HOOK() (qt_test_gate_block_hook ? qt_test_gate_block_hook() : (void)0)
#define QT_GATE_FREED_HOOK() (qt_test_gate_freed_hook ? qt_test_gate_freed_hook() : (void)0)
#define QT_READER_PRE_HOOK() (qt_test_reader_pre_hook ? qt_test_reader_pre_hook() : (void)0)
#define QT_LMHEAD_PRE_GATE_HOOK() (qt_test_lmhead_pre_gate_hook ? qt_test_lmhead_pre_gate_hook() : (void)0)
#else
#define QT_TAKE_WAKE_HOOK() ((void)0)
#define QT_DRAIN_WAIT_HOOK() ((void)0)
#define QT_DRAIN_LEAVE_HOOK() ((void)0)
#define QT_GATE_BLOCK_HOOK() ((void)0)
#define QT_GATE_FREED_HOOK() ((void)0)
#define QT_READER_PRE_HOOK() ((void)0)
#define QT_LMHEAD_PRE_GATE_HOOK() ((void)0)
#endif

/* ====================== tier lifecycle gate ===============================
 *
 * ONE mechanism, and deliberately NOT part of G: qt_init memsets G and
 * qt_shutdown destroys G.mx, so any claim kept inside G is state the two of
 * them can wipe out from under a running teardown -- which is precisely how the
 * once-only latch this replaces was handed straight back by a re-init that
 * landed in the teardown window (#1564).
 *
 *      QT_DEAD  ->  QT_LIVE  ->  QT_TEARING_DOWN  ->  QT_DEAD
 *
 * A caller that touches the tier enters through qt_gate_enter() and leaves
 * through qt_gate_leave(); what it adds to qg_live is what makes the
 * check-and-use atomic with respect to the teardown:
 *
 *   - a caller ALREADY inside is counted, so qt_shutdown cannot free the
 *     storage under its feet: the exclusive section waits for qg_live to reach
 *     0, and it is inside that section -- never one line earlier -- that
 *     G.slot is freed and G.mx, G.cv and G.cv_take are destroyed;
 *   - a caller arriving AFTER the teardown has started is refused at the door,
 *     because the state is no longer QT_LIVE, and it never reaches the storage.
 *
 * Those are the two things `if(!G.on) return;` before
 * `pthread_mutex_lock(&G.mx)` could not do. The flag was read with nothing
 * held, so a caller descheduled between the test and the lock resumed into a
 * destroyed mutex and a NULL slot array (#1564: SIGSEGV at
 * `int r = qs(layer,eid)->resident`). The same shape stood in twelve sibling
 * entry points, so the invariant lives HERE and every sibling calls these two
 * helpers: there is no per-function guard to get right, and none to get wrong
 * once. Clearing the flag at the TOP of the teardown is what made the same
 * window a re-init could walk into -- see qt_shutdown.
 *
 * Readers REFUSE while an exclusive section is queued rather than queue behind
 * it: a decode thread must not park behind a teardown or a model load, and
 * refusing needs no lock the holder is keeping. Refusal is also what makes the
 * queue drainable -- qg_want_x is raised BEFORE the wait, so no late caller can
 * re-inflate qg_live behind the teardown's back. That deadlock is also why
 * qt_shutdown is split in two and does not hold the gate for its whole run: the
 * callers parked on G.cv_take hold a gate slot of their own and can only be
 * woken by the teardown, so a teardown waiting for them while holding the gate
 * would be waiting for a wake-up only it could send.
 *
 * Self-contained on purpose. A pthread_rwlock would want
 * PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP, which sits behind _GNU_SOURCE,
 * and this translation unit is #included by tests that pull in <stdio.h> before
 * it -- far too late to define that here. */
enum { QT_DEAD = 0, QT_LIVE = 1, QT_TEARING_DOWN = 2 };
static pthread_mutex_t qg_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  qg_cv = PTHREAD_COND_INITIALIZER;
static int qg_live;      /* admitted callers inside, not yet out              */
static int qg_want_x;    /* an exclusive section is queued or held            */
static int qg_x;         /* exclusive section held                           */
static int qg_state = QT_DEAD;

/* qg_state is read by qt_ready() off the decode path with no lock held, so it
 * goes through __atomic everywhere rather than only there: mixing atomic and
 * plain accesses to one object is not something to leave to chance. */
static int qg_get(void){ return __atomic_load_n(&qg_state,__ATOMIC_ACQUIRE); }
static void qg_set(int s){ __atomic_store_n(&qg_state,s,__ATOMIC_RELEASE); }

/* Admit one caller for the whole of its body, or refuse it. On 1 the gate is
 * held, and the tier storage cannot be freed or rebuilt underneath it. */
static int qt_gate_enter(void){
    pthread_mutex_lock(&qg_mx);
    if(qg_get()!=QT_LIVE){ pthread_mutex_unlock(&qg_mx); return 0; }
    qg_live++;
    pthread_mutex_unlock(&qg_mx);
    return 1;
}
static void qt_gate_leave(void){
    pthread_mutex_lock(&qg_mx);
    if(--qg_live==0) pthread_cond_broadcast(&qg_cv);
    pthread_mutex_unlock(&qg_mx);
}
/* The dense-trunk variant (qt_dnproj_*): the projections work with the expert
 * tier never started, so QT_DEAD admits too. What it refuses is what matters --
 * a teardown in progress, or an exclusive section queued or held -- so the
 * tensors qt_shutdown frees are counted exactly like the expert slots and
 * cannot be freed under a matmul the backend has already picked up. */
static int qt_gate_enter_trunk(void){
    pthread_mutex_lock(&qg_mx);
    if(qg_get()==QT_TEARING_DOWN || qg_x || qg_want_x){ pthread_mutex_unlock(&qg_mx); return 0; }
    qg_live++;
    pthread_mutex_unlock(&qg_mx);
    return 1;
}

/* Exclusive: no caller inside, and none can arrive. 0 if somebody else already
 * has it or is queued for it -- the loser refuses rather than waits, which is
 * what keeps a second init (or an init against a running teardown) from
 * queueing behind work it has no business joining.
 *
 * Admitted only FROM the lifecycle state the caller is entitled to leave --
 * QT_DEAD for qt_init, QT_TEARING_DOWN for the teardown's second half -- and
 * that test is made here, under qg_mx, BEFORE qg_want_x is raised. Testing it
 * after the wait let an init that arrived during TEARING_DOWN take the queue
 * while a counted caller was still inside; the teardown then found qg_want_x
 * taken, its own exclusive entry failed, and qt_shutdown returned with the
 * state stuck at TEARING_DOWN and every tensor still allocated (#1564). The
 * state cannot move while the queue is held: DEAD and TEARING_DOWN are each
 * left only by the holder of this section. */
static int qt_gate_xenter(int from){
    pthread_mutex_lock(&qg_mx);
    if(qg_get()!=from || qg_x || qg_want_x){ pthread_mutex_unlock(&qg_mx); return 0; }
    qg_want_x=1;
    while(qg_live){ QT_GATE_BLOCK_HOOK(); pthread_cond_wait(&qg_cv,&qg_mx); }
    qg_want_x=0; qg_x=1;
    pthread_mutex_unlock(&qg_mx);
    return 1;
}
static void qt_gate_xleave(void){
    pthread_mutex_lock(&qg_mx);
    qg_x=0; pthread_cond_broadcast(&qg_cv);
    pthread_mutex_unlock(&qg_mx);
}

/* Trunk-only storage ownership, counted here and not in G for the reason the
 * lifecycle state is: qt_init memsets G, and an init must not be able to erase
 * the record of storage a teardown still has to free.
 *
 * G_dnp[] is live with the expert tier NEVER started -- the dense projections
 * are admitted by qt_gate_enter_trunk() at QT_DEAD -- so the state alone does
 * not say whether anything is still resident. Nothing counted that, and a
 * qt_shutdown that found the state already DEAD returned at once, so the
 * projections were never freed: qt_dnproj_matmul kept computing on them
 * through the trunk gate, and a later qt_dnproj_init of the SAME SHAPE was
 * answered by the backend's cached path (backend_cuda.cu returns 1 from a
 * non-NULL *tensor whose fmt/I/O/device/gs match, WITHOUT copying) -- a new
 * generation running on the previous generation's weights, with nothing
 * reporting it. A DEAD fast path may only return when there is nothing left
 * to free -- and "nothing left" is a question about WORK, not about tensors:
 * see the count's accessors, and the fast path in qt_shutdown, which is where
 * the two are read as a pair.
 *
 * G_lmh needs no counter of its own: it is only ever created under the full
 * gate, so half two of every teardown that can see it also frees it. This is
 * the trunk's "the expert tier may never start" case, and nothing else. */
static int qg_trunk;      /* live G_dnp[].t tensors; NOT protected by qg_mx  */

/* __atomic here for the same reason qg_state uses it above, and for one more.
 * A trunk caller holds its qg_live slot across the ENTRY and the LEAVE only:
 * the upload, and the increment that follows it, both run with qg_mx
 * released. Two inits admitted together are therefore inside that stretch at
 * the same time, and a read-modify-write on a plain int is a lost update, not
 * a slow one -- which is why the add is a fetch_add and not a load and a
 * store. The read is the load the teardown's fast path needs to take alongside
 * qg_live, and the store is how a teardown clears the count under the
 * exclusive section that has just freed the tensors. */
static int qg_trunk_get(void){ return __atomic_load_n(&qg_trunk,__ATOMIC_ACQUIRE); }
static void qg_trunk_set(int n){ __atomic_store_n(&qg_trunk,n,__ATOMIC_RELEASE); }
static void qg_trunk_add(int n){ (void)__atomic_fetch_add(&qg_trunk,n,__ATOMIC_RELAXED); }

/* Count parked callers so shutdown can reclaim their shared storage safely. */
static void wait_take_locked(void){
    G.waiters++;
    pthread_cond_wait(&G.cv_take,&G.mx);
    QT_TAKE_WAKE_HOOK();
    if(--G.waiters==0 && G.th_stop) pthread_cond_broadcast(&G.cv_take);
}

static QSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
static int home(int eid){ return eid % G.ndev; }

/* Set by qt_init_stream_int4, read by stage(): the stream tier's caller
 * (glm53) packs its RAM nibbles offset-binary already — the exact layout
 * coli_cuda_tensor_upload_g converts (offset_to_signed_s4, XOR 0x88) before
 * the kernels sign-decode. qwen36's warmstart tier instead feeds
 * two's-complement container bytes, which DO need the XOR to reach the
 * upload format. */
static int G_int4_stream;

static size_t expert_matrix_bytes(void){
    size_t n=(size_t)G.D*G.Ih;
    return G.wfmt==0 ? n*sizeof(float) : G.wfmt==4 ? n/2 : n;
}

/* Staging: copy the packed int4 (g|u|d) bytes and the scales (gs|us|ds).
 * Two's-complement RAM (qwen36 int4 containers) is XORed into the
 * offset-binary upload format of backend_cuda fmt=2/4; offset-binary RAM
 * (glm53 stream, whose CPU matmul_i4_grouped decodes nibble-8) must pass
 * verbatim or the upload's own conversion double-flips the sign bits. */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
                  const float *gs,const float *us,const float *ds){
    size_t mb = expert_matrix_bytes();
    if(G.wfmt==0 || G.wfmt==1 || G.wfmt==8 || G_int4_stream){
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
    if(dsc){
        memcpy(dsc,                 gs, G.sc_gu*sizeof(float));
        memcpy(dsc+G.sc_gu,         us, G.sc_gu*sizeof(float));
        memcpy(dsc+2*G.sc_gu,       ds, G.sc_d *sizeof(float));
    }
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
                v->resident=1; qs(layer,eid)->queued=0; G.inflight--;
                pthread_cond_broadcast(&G.cv_take);
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
        size_t mb=expert_matrix_bytes();
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok;
        if(G.wfmt==0){
            ok = coli_cuda_tensor_upload(&tg, w,      NULL, 0, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+mb,   NULL, 0, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*mb, NULL, 0, G.Ih, G.D,  dv);
        } else if(G.wfmt==8){
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
        if(!ok){
            if(tg) coli_cuda_tensor_free(tg);
            if(tu) coli_cuda_tensor_free(tu);
            if(td) coli_cuda_tensor_free(td);
        }
        pthread_mutex_lock(&G.mx);
        QSlot *s=qs(layer,eid);
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
        else  { int hd=home(eid); G.used[hd]-=G.exp_bytes;
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
    double value = 0, pm = 0;
    for(size_t r = (fit > (size_t)k ? fit - k : 0); r < fit && r < m; r++){ value += cpu_factor * p[r] * (double)exp_bytes; pm = p[r]; }
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
static int G_fp8_stream;   /* G_int4_stream declared above stage() */
static float G_stream_swiglu_limit;
static const float *G_fp8_lut;

static int qt_init_mode(int nl, int ne, int D, int Ih, int cap, int topk, int expert_gs,
                        int expert_is_int4, int fp8, const float *lut, int int4, float limit);
int qt_init_fp8(int nl, int ne, int D, int Ih, int cap, int topk, const float *e4m3_lut){
    return qt_init_mode(nl, ne, D, Ih, cap, topk, 0, 0, 1, e4m3_lut, 0, 0.0f);
}
int qt_init_stream_int4(int nl, int ne, int D, int Ih, int cap, int topk, float limit){
    if(nl<1||ne<1||D<1||Ih<1||cap<1||cap>ne||topk<1||!isfinite(limit)||limit<=0) return 0;
    return qt_init_mode(nl, ne, D, Ih, cap, topk, 64, 1, 0, NULL, 1, limit);
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

/* The only other place a generation is built, and where the lifecycle is
 * entered. The state test and the whole rebuild are ONE exclusive step: the
 * gate is held from the test to the end of the build, so no reader can be
 * inside, no teardown can be running, and no second init can start.
 *
 * Admit only from QT_DEAD -- "never built" or "the last one is fully torn
 * down". That single test closes both halves of #1564:
 *   - a second init while the first is live used to memset G over a running
 *     uploader: the mutex and both condvars it is parked on, its thread handle
 *     (a second uploader starts, the first is orphaned and never joined), the
 *     slot array and the host allocations behind it;
 *   - an init racing a TEARDOWN used to be ACCEPTED, because the teardown
 *     cleared G.on at its TOP. For the whole slow window -- the drain, the
 *     pthread_join, the backend drain, the frees -- `if(G.on) return 0;` read
 *     "never started", so the init memset over the running teardown, replaced
 *     the mutex the teardown was about to park on and re-armed the once-only
 *     claim: SIGSEGV inside pthread_join, exit 139 (#1564). Nothing is re-armed
 *     here any more; the state is outside G and only qt_shutdown and this
 *     function ever move it.
 *
 * A refused init leaves the running tier exactly as it was: the guard runs
 * before the memset AND before the mode globals (stream/fp8 flag, LUT, SwiGLU
 * limit) are written, so there is nothing to undo. The wrappers used to set
 * the mode first and clear it on refusal, which switched a live streaming tier
 * from the clamped to the plain path (#1564). */
static int qt_init_body(int nl, int ne, int D, int Ih, int cap, int topk,
                        int expert_gs, int expert_is_int4);

static int qt_init_mode(int nl, int ne, int D, int Ih, int cap, int topk, int expert_gs,
                        int expert_is_int4, int fp8, const float *lut, int int4, float limit){
    if(!qt_gate_xenter(QT_DEAD)) return 0;
    G_fp8_stream=fp8; G_fp8_lut=lut; G_int4_stream=int4; G_stream_swiglu_limit=limit;
    int r=qt_init_body(nl,ne,D,Ih,cap,topk,expert_gs,expert_is_int4);
    if(r) qg_set(QT_LIVE);
    else G_fp8_stream=G_int4_stream=0;   /* still DEAD: no live tier to disturb */
    qt_gate_xleave();
    return r;
}
int qt_init(int nl, int ne, int D, int Ih, int cap, int topk, int expert_gs,
            int expert_is_int4){
    return qt_init_mode(nl,ne,D,Ih,cap,topk,expert_gs,expert_is_int4,0,NULL,0,0.0f);
}

static int qt_init_body(int nl, int ne, int D, int Ih, int cap, int topk,
            int expert_gs, int expert_is_int4){
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(cap != ne && !G_fp8_stream && !G_int4_stream){
        fprintf(stderr,"[qtier] cap=%d != n_experts=%d -> tier disabled (needs full RAM residency)\n",cap,ne);
        return 0;
    }
    if(topk>QT_MAX_ROWS){ fprintf(stderr,"[qtier] topk>%d unsupported\n",QT_MAX_ROWS); return 0; }
    /* G is wiped here and nowhere else, from inside the exclusive section the
     * wrapper above holds, so the slot array, the mutex, both condvars, the
     * thread handle and the host allocations are only ever rebuilt by a caller
     * already admitted against the lifecycle. */
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
    /* expert_is_int4=2 extends the old boolean ABI with unquantized f32. */
    G.wfmt = G_fp8_stream ? 8 : expert_is_int4==2 ? 0 : (expert_is_int4 ? 4 : 1);
    if(G.wfmt==1 && expert_gs>0){
        fprintf(stderr,"[qtier] int8 experts with grouped scales (gs=%d) cannot be "
                       "expressed on the GPU (fmt=1 is per-row only) -> CPU path\n", expert_gs);
        return 0;
    }
    G.egs = expert_gs;
    if(G.wfmt==0){
        G.sc_gu=G.sc_d=0;
    } else if(G.wfmt==8){
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
    size_t mat_bytes = expert_matrix_bytes();
    size_t scl_bytes = (2*G.sc_gu+G.sc_d)/3*sizeof(float);
    G.exp_bytes = 3*dev_alloc_footprint(mat_bytes)
                + (G.wfmt ? 3*dev_alloc_footprint(scl_bytes) : 0); /* + allocation slack */

    /* Per-device allowance for tier + trunk: CUDA_EXPERT_GB when numeric,
     * else free minus 1 GB headroom. The heat table is loaded here too (it
     * used to be loaded after the budgets) because the placer prices
     * experts by heat. */
    size_t capacity[QT_MAX_DEV]; int capdev[QT_MAX_DEV]; int ncap = G.ndev;
    const char *bg=getenv("CUDA_EXPERT_GB");
    for(int i=0;i<G.ndev;i++){
        size_t freeb=0,totb=0; coli_cuda_mem_info(G.dev[i],&freeb,&totb);
        capdev[i] = G.dev[i];
        size_t headroom = G_int4_stream ? (3ull << 29) : (1ull << 30);
        capacity[i] = (bg && strcmp(bg,"auto") && atof(bg)>0)
                   ? (size_t)(atof(bg)*1024.0*1024.0*1024.0)
                   : (freeb>headroom ? freeb-headroom : 0);
        fprintf(stderr,"[qtier] dev %d: %.1f GB free, allowance %.1f GB\n",
                G.dev[i], freeb/1073741824.0, capacity[i]/1073741824.0);
    }
    /* ne == 0 is dense: no expert slots, G.slot stays NULL (G was wiped above)
     * and every nl*ne walk over it is empty. calloc(0) may return NULL, which
     * used to disable the whole tier as if the host were out of memory. */
    if(ne>0){
        G.slot=calloc((size_t)nl*ne,sizeof(QSlot));
        if(!G.slot) return 0;
    }
    const char *hf=getenv("HEAT_FILE");
    if(hf && !G_int4_stream){
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
    /* qt_issue strides each device's block by QT_MAX_ROWS*D floats (its max
     * row count), not 8*D: a device other than 0 with a full 32-row issue
     * used to run past its own slice and off the end of this allocation
     * (#1339). The stride and G.is_k's row capacity are the same constant. */
    G.is_x_floats=(size_t)G.ndev*QT_MAX_ROWS*D;
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

int qt_ready(void){ return qg_get()==QT_LIVE; }

int qt_lmhead_init(const int8_t *q, const float *sc, int I, int O){
    /* G_lmh is read ONLY after the lifecycle admits this caller, dev_ok
     * included (#1564, review of e6c70eb7, HIGH B). It used to be read on the
     * line above the admission, with nothing held: a caller descheduled in
     * that window crossed a shutdown and a re-init, was then admitted by the
     * gate -- which asks only whether a generation is live -- and installed
     * the PREVIOUS generation's weights, scales and device into a generation
     * configured with COLI_PLACE=off, which had decided this lm_head belongs
     * on the CPU. The upload succeeds, so nothing reports it and the new
     * generation answers tokens with the old model's weights.
     *
     * The init path has no NULL rejection to save it the way
     * qt_lmhead_matmul has: the backend refuses a NULL cached tensor and that
     * call falls back to the CPU, but here the call CREATES the tensor, so
     * there is nothing left to refuse. Same guard, two outcomes -- so the
     * state is read where the lifecycle is, not per function. The caller's own
     * arguments are not lifecycle state and are still checked up front. */
    if(!q||!sc) return 0;
    QT_LMHEAD_PRE_GATE_HOOK();
    if(!qt_gate_enter()) return 0;
    int ok=0;
    if(G_lmh.dev_ok){
        int dev=G_lmh.dev;
        if(!coli_cuda_tensor_upload(&G_lmh.t,q,sc,1,I,O,dev)){
            fprintf(stderr,"[lmh] lm_head upload failed -> stays on CPU\n");
        } else {
            G_lmh.dev=dev; G_lmh.on=1;
            fprintf(stderr,"[lmh] lm_head [%d x %d] int8 resident on CUDA dev %d (%.2f GB)\n",
                    O,I,dev,(double)O*I/1073741824.0);
            ok=1;
        }
    }
    qt_gate_leave();
    return ok;
}

int qt_dnproj_init(int layer, const int8_t *q, const float *sc,
                   int I, int O, int device){
    if(layer < 0 || layer >= QT_DN_MAX_LAYERS) return 0;
    if(device == QT_PLACE_CPU || !q || !sc) return 0;
    /* No expert-tier requirement: the projections can be measured on a card
     * that holds no experts at all, hence the trunk gate. But qt_shutdown frees
     * G_dnp[].t, so they are counted like every other caller. */
    if(!qt_gate_enter_trunk()) return 0;
    /* Was this slot already owned? A repeated init of a resident layer is
     * answered by the backend's cached path and allocates nothing, so the
     * ownership count may only move when the tensor itself appeared. */
    int had = G_dnp[layer].t != NULL;
    int ok = coli_cuda_tensor_upload(&G_dnp[layer].t, q, sc, 1, I, O, device);
    if(!ok) fprintf(stderr,"[dnp] layer %d upload failed -> stays on CPU\n", layer);
    else { G_dnp[layer].dev = device; G_dnp[layer].on = 1; if(!had) qg_trunk_add(1); }
    qt_gate_leave();
    return ok;
}

int qt_dnproj_matmul(int layer, float *y, const float *x, int I, int O){
    if(layer < 0 || layer >= QT_DN_MAX_LAYERS) return 0;
    /* Refused (a teardown is running) means the CPU path for this call. */
    if(!qt_gate_enter_trunk()) return 0;
    int ok = 0;
    if(G_dnp[layer].on){
        ok = coli_cuda_matmul(&G_dnp[layer].t,y,x,NULL,NULL,1,1,I,O,G_dnp[layer].dev,0);
        if(!ok){
            fprintf(stderr,"[dnp] layer %d GPU matmul failed; CPU from here on\n", layer);
            G_dnp[layer].on = 0;
        }
    }
    qt_gate_leave();
    return ok;
}

int qt_lmhead_matmul(float *y, const float *x, int I, int O){
    if(!G_lmh.on || !qt_gate_enter()) return 0;
    /* cached-tensor path: upload params are ignored once *t exists */
    int ok = coli_cuda_matmul(&G_lmh.t,y,x,NULL,NULL,1,1,I,O,G_lmh.dev,0);
    if(!ok){
        fprintf(stderr,"[lmh] GPU matmul failed; falling back to CPU from here on\n");
        G_lmh.on=0;
    }
    qt_gate_leave();
    return ok;
}

/* Is (layer,eid) currently VRAM-resident? (used to free RAM-side int8 copies) */
int qt_is_resident(int layer,int eid){
    if(!qt_gate_enter()) return 0;
    QT_READER_PRE_HOOK();
    pthread_mutex_lock(&G.mx);
    int r = qs(layer,eid)->resident;
    pthread_mutex_unlock(&G.mx);
    qt_gate_leave();
    return r;
}

size_t qt_resident_count(void){
    if(!qt_gate_enter()) return 0;
    pthread_mutex_lock(&G.mx);
    size_t n=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++) n+=G.slot[i].resident;
    pthread_mutex_unlock(&G.mx);
    qt_gate_leave();
    return n;
}

size_t qt_resident_bytes(void){
    if(!qt_gate_enter()) return 0;
    pthread_mutex_lock(&G.mx);
    size_t n=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++) n+=G.slot[i].resident;
    size_t bytes=n*G.exp_bytes;
    pthread_mutex_unlock(&G.mx);
    qt_gate_leave();
    return bytes;
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
    size_t mb=expert_matrix_bytes();
    uint8_t *w=malloc(3*mb);
    float *sc=G.wfmt ? malloc((2*G.sc_gu+G.sc_d)*sizeof(float)) : NULL;
    if(!w||(G.wfmt&&!sc)){ free(w); free(sc); return 0; }
    if(!reserved && v_eid<0) G.used[hd]+=G.exp_bytes;
    s->queued=1;
    stage(w,sc,s->g4,s->u4,s->d4,s->gs,s->us,s->ds);
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
    if(!g4 || !qt_gate_enter()) return;
    QSlot *s=qs(layer,eid);        /* safe without G.mx: the gate pins the array */
    pthread_mutex_lock(&G.mx);
    if(G_fp8_stream || G_int4_stream){
        if(s->heat<0xFFFFFFFFu) s->heat++;
        stream_point(s,g4,u4,d4,gs,us,ds);
        stream_promote_locked(layer,eid);
        stream_forget(s);
        pthread_mutex_unlock(&G.mx);
        qt_gate_leave();
        return;
    }
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    if(s->heat<0xFFFFFFFFu) s->heat++;
    enqueue_locked(layer,eid,-1,-1,0);
    pthread_mutex_unlock(&G.mx);
    qt_gate_leave();
}

/* blocking variant for the warmstart (waits for queue space). */
void qt_note_block(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!g4 || !qt_gate_enter()) return;
    QSlot *s=qs(layer,eid);        /* safe without G.mx: the gate pins the array */
    pthread_mutex_lock(&G.mx);
    G.blocking_calls++;
    if(G_fp8_stream) stream_point(s,g4,u4,d4,gs,us,ds);
    else if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    enqueue_locked(layer,eid,-1,-1,0);
    if(G_fp8_stream) stream_forget(s);
    G.blocking_calls--;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
    qt_gate_leave();
}

/* warmstart order -- heat descending (HEAT_FILE) or natural order.
 * Returns 0 once all budgets are full or the list is exhausted. */
static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    uint32_t ha=g_sort_heat[*(const int*)a], hb=g_sort_heat[*(const int*)b];
    return ha<hb ? 1 : ha>hb ? -1 : 0;
}
int qt_fill_next(int *layer,int *eid){
    if(!qt_gate_enter()) return 0;
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
        if(full){ pthread_mutex_unlock(&G.mx); qt_gate_leave(); return 0; }
        G.fill_cur++;
        if(s->resident||s->queued) continue;
        if(G.used[hd]+G.exp_bytes>G.budget[hd]) continue;   /* dieses Device voll */
        *layer=l; *eid=e;
        pthread_mutex_unlock(&G.mx);
        qt_gate_leave();
        return 1;
    }
    pthread_mutex_unlock(&G.mx);
    qt_gate_leave();
    return 0;
}

/* Plan the whole warmstart set in one pass -- same heat order and budget
 * reservation as qt_fill_next, but without loading. The experts are then
 * loaded by any number of threads and handed over via qt_note_planned. */
int qt_plan_fill(int *layers,int *eids,int max){
    if(!qt_gate_enter()) return 0;
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
    qt_gate_leave();
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
    if(!qt_gate_enter()) return;
    QSlot *s=qs(layer,eid);        /* safe without G.mx: the gate pins the array */
    pthread_mutex_lock(&G.mx);
    G.blocking_calls++;
    if(!g4){
        /* The loader had nothing to hand over. qt_plan_fill reserved budget
         * and set planned=1 for this expert; returning here without undoing
         * both keeps the bytes out of the budget for the life of the process
         * and "if(resident||queued||planned) continue" never reconsiders the
         * expert. #1331 was this leak for every expert of an int8 container. */
        if(s->planned){ G.used[home(eid)]-=G.exp_bytes; s->planned=0; }
        G.blocking_calls--;
        pthread_cond_broadcast(&G.cv_take);
        pthread_mutex_unlock(&G.mx);
        qt_gate_leave();
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
    G.blocking_calls--;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
    qt_gate_leave();
}

/* Blocks until every enqueued upload has COMPLETED (end of warmstart): the
 * engine frees the RAM int8 copies of the planned experts right after this
 * returns, so "dequeued" is not enough -- the uploader drops qn before it
 * calls the backend, and the last expert would still be queued=1 (#1360).
 * Must not be called with an expert group open: an LFRU swap parks the
 * uploader on issue_open until qt_take() clears it. */
void qt_fill_wait(void){
    if(!qt_gate_enter()) return;
    pthread_mutex_lock(&G.mx);
    while(G.inflight>0 && !G.th_stop) wait_take_locked();
    pthread_mutex_unlock(&G.mx);
    qt_gate_leave();
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
    if(K>QT_MAX_ROWS || !qt_gate_enter()) return 0;
    uint32_t mask=0;
    ColiCudaTensor *tg[QT_MAX_DEV][QT_MAX_ROWS],*tu[QT_MAX_DEV][QT_MAX_ROWS],*td[QT_MAX_DEV][QT_MAX_ROWS];
    static int rows[QT_MAX_ROWS]={0};
    if(!rows[0]) for(int i=0;i<QT_MAX_ROWS;i++) rows[i]=1;
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
        float *xr=G.is_x + (size_t)di*QT_MAX_ROWS*G.D;     /* per-device input block */
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        int ok = G_int4_stream
               ? coli_cuda_expert_group_issue_clamped(tg[di],tu[di],td[di],rows,c,xr,G_stream_swiglu_limit)
               : coli_cuda_expert_group_issue(tg[di],tu[di],td[di],rows,c,xr);
        if(!ok){
            /* issue failed -> hand these k back to the CPU */
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        }
    }
    qt_gate_leave();
    return mask;
}

void qt_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    if(!qt_gate_enter()) return;
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
    qt_gate_leave();
}

void qt_stats(void){
    if(!qt_gate_enter()) return;
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
    qt_gate_leave();
}

/* The teardown is a lifecycle, and the two halves below are load-bearing:
 *
 *      QT_LIVE  --(first qt_shutdown)-->  QT_TEARING_DOWN  -->  QT_DEAD
 *
 * A concurrent qt_shutdown does not win the claim and walk away: it WAITS for
 * the teardown to finish and only then returns, so the shutdown is
 * synchronous -- when qt_shutdown returns to ANY caller, the storage is
 * already freed and G.mx is already destroyed. Returning straight away is
 * exactly how a loser came to observe a half-torn-down tier (#1564), and it is
 * why the once-only claim this replaces was never a lifecycle: a bit says
 * "somebody started", not "everybody finished".
 *
 * HALF ONE runs with the gate OPEN. It stops admitting callers -- the state is
 * no longer QT_LIVE, so every sibling qt_* entry point refuses from here on --
 * and it wakes the ones already inside. It cannot hold the gate while it does
 * that: the callers it has to wake are parked on G.cv_take holding a gate slot
 * of their own, so a teardown that held the gate while waiting for them would
 * be waiting for a wake-up only it could send. That is the deadlock in "one
 * exclusive gate for the whole teardown", and it is why the gate is taken
 * second rather than first.
 *
 * HALF TWO takes the gate EXCLUSIVELY. Only there is it true that no caller is
 * inside and that no caller can arrive, and only there are the tensors freed
 * and G.mx/G.cv/G.cv_take destroyed.
 *
 * G.on stays 1 for all of half one. Clearing it at the TOP is what turned this
 * window into a hole: `if(G.on) return 0;` then read "never started" for the
 * whole drain/join/free stretch, ACCEPTED a re-init, and the memset replaced
 * the mutex the teardown was about to park on -- SIGSEGV inside pthread_join,
 * exit 139 (#1564). G.on is now cleared in half two, immediately before the
 * destroy, and nothing consults it for admission any more: the gate is the
 * authority and the flag only reports what the lifecycle has already decided. */
void qt_shutdown(void){
    /* ---- who tears down, and who waits ---- */
    pthread_mutex_lock(&qg_mx);
    if(qg_get()!=QT_LIVE){
        /* A teardown is already running, or one already finished: this caller
         * never enters the teardown body at all, however late it is.
         *
         * Wait on "no longer TEARING_DOWN", not on "== QT_DEAD": a fresh
         * qt_init may have moved the state on to QT_LIVE by the time this
         * thread is scheduled, and waiting for QT_DEAD would then never be
         * satisfied. Reaching QT_LIVE is itself proof the teardown is over --
         * the only ways out of TEARING_DOWN are QT_DEAD at the end of this
         * function and QT_LIVE from qt_init, and qt_init starts only from
         * QT_DEAD. This thread touches no state on the way out, so it is
         * correct for it to return either way. */
        while(qg_get()==QT_TEARING_DOWN) pthread_cond_wait(&qg_cv,&qg_mx);
        /* Losing the claim is the normal end of a shutdown -- but "already
         * DEAD" is not proof that the storage is gone. The dense projections
         * are admitted at QT_DEAD, so a tier that was never started can still
         * own live G_dnp[].t, and returning here used to leave them resident:
         * callable after the shutdown, and a same-shaped re-init would inherit
         * the cached weights instead of copying its own. So the DEAD fast path
         * may only return when there is nothing left to free -- and a state
         * that has since moved back to QT_LIVE owns its own teardown, whose
         * half two frees this as well, so leave that one alone. That state is
         * the only question answerable from out here, and it is re-read under
         * this same lock rather than carried out of the loop above. */
        if(qg_get()==QT_LIVE){ pthread_mutex_unlock(&qg_mx); return; }
        /* "Nothing left to free" is a question about WORK, and a count of
         * TENSORS cannot answer it from here. qg_trunk moves when an upload
         * RETURNS, while qt_gate_enter_trunk has been counting the caller in
         * qg_live since before the upload started -- so for the whole of every
         * upload there is a trunk caller inside the gate that this line, read
         * with nothing but the mutex, cannot see. Deciding here therefore let
         * the shutdown return with a qt_dnproj_init still parked in its upload,
         * which then finished and left a live G_dnp[].t behind a completed
         * teardown -- HIGH A again, one interleaving later (#1564, review of
         * 6917c3f6).
         *
         * So the pair is read together, both under qg_mx, and both are needed:
         *
         *   qg_live == 0   no trunk caller is between admission and leave, so
         *                  there is no increment in flight that can appear
         *                  after this read, and nothing is on its way to make a
         *                  tensor resident;
         *   qg_trunk == 0  and nothing is resident now.
         *
         * Either one alone is the defect: the count alone misses a caller in
         * flight (above), qg_live alone misses a tensor whose caller has
         * already left. Together they are the whole of "nothing left to do",
         * and they are a consistent pair because qg_mx is what orders the
         * admission against the read. At QT_DEAD, qg_live counts trunk callers
         * and nothing else -- qt_gate_enter admits only at QT_LIVE -- so it is
         * exactly the count of in-flight trunk work.
         *
         * Anything else is somebody's own work and is NOT decided here: a
         * resident tensor, or a trunk caller in flight, both fall through to
         * the exclusive section below, which is the only place the answer is
         * true, because it is the only place no caller is inside and none can
         * arrive. The frees stay driven by the storage itself rather than by
         * the count that was consulted to get here, so the count can never be
         * the thing that decides what is freed. */
        int live = qg_live, trunk = qg_trunk_get();
        pthread_mutex_unlock(&qg_mx);
        if(!trunk && !live) return;
        /* Waiting is the fix, not the exclusive section: a caller in flight has
         * already been admitted, so refusing to wait for it is what let it
         * escape. qt_gate_xenter raises qg_want_x first, so a trunk caller that
         * arrives from here on is refused at the door instead of joining a
         * teardown in progress. */
        if(!qt_gate_xenter(QT_DEAD)) return;
        for(int i=0;i<QT_DN_MAX_LAYERS;i++)
            if(G_dnp[i].t) coli_cuda_tensor_free(G_dnp[i].t);
        memset(G_dnp,0,sizeof G_dnp);
        qg_trunk_set(0);
        qg_set(QT_DEAD);
        qt_gate_xleave();
        return;
    }
    qg_set(QT_TEARING_DOWN);
    pthread_mutex_unlock(&qg_mx);
    const char *hf=getenv("HEAT_FILE");
    if(hf && !G_int4_stream){
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
     * otherwise never notice th_stop and pthread_join below would hang (#1340).
     * These are the inside callers the teardown has to let go before it may
     * take the gate, which is why this half runs without it. */
    pthread_mutex_lock(&G.mx);
    G.th_stop=1; pthread_cond_signal(&G.cv); pthread_cond_broadcast(&G.cv_take);
    while(G.blocking_calls) pthread_cond_wait(&G.cv_take,&G.mx);
    while(G.waiters){ QT_DRAIN_WAIT_HOOK(); pthread_cond_wait(&G.cv_take,&G.mx); }
    QT_DRAIN_LEAVE_HOOK();
    pthread_mutex_unlock(&G.mx);
    /* ---- half two: exclusive. Nothing is inside, and nothing can arrive. ----
     * Every free and every destroy below is inside this section, so the storage
     * a counted caller is still using cannot be pulled from under it, and a
     * caller that arrives late is refused at the door instead of being let in
     * against a state it never checked. */
    if(!qt_gate_xenter(QT_TEARING_DOWN)) return;   /* unreachable: qt_init cannot
                                                     * queue from TEARING_DOWN */
    pthread_join(G.th,NULL);
    /* The backend queues expert kernels and the output download on its device
     * streams. Drain every outstanding group before releasing weights those
     * kernels may still reference. */
    if(G.issue_open)
        for(int i=0;i<G.ndev;i++)
            if(G.is_cnt[i]) (void)coli_cuda_expert_group_take(G.dev[i]);
    G.issue_open=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++){
        coli_cuda_tensor_free(G.slot[i].tg);
        coli_cuda_tensor_free(G.slot[i].tu);
        coli_cuda_tensor_free(G.slot[i].td);
    }
    if(G_lmh.t) coli_cuda_tensor_free(G_lmh.t);
    for(int i=0;i<QT_DN_MAX_LAYERS;i++)
        if(G_dnp[i].t) coli_cuda_tensor_free(G_dnp[i].t);
    free(G.fill_order); free(G.heat0); free(G.is_x); free(G.slot);
    G.fill_order=NULL; G.heat0=NULL; G.is_x=NULL; G.slot=NULL;
    memset(&G_lmh,0,sizeof G_lmh); memset(G_dnp,0,sizeof G_dnp);
    qg_trunk_set(0);  /* same invariant as the trunk-only teardown: the count is
                       * what the DEAD fast path reads beside qg_live, so it may
                       * not outlive the tensors it counts -- a stale positive
                       * sends every later redundant shutdown through an
                       * exclusive section that has nothing left to free. */
    /* G.on goes last, and only here. It stayed 1 for the whole of half one on
     * purpose: a caller in that window must not be able to read "torn down"
     * and walk into storage that was still live -- and, symmetrically, a
     * qt_init in that window must be refused by the state rather than accepted
     * by a flag that is only cleared once the teardown is nearly finished. */
    G.on=0;
    pthread_cond_destroy(&G.cv_take); pthread_cond_destroy(&G.cv); pthread_mutex_destroy(&G.mx);
    G_fp8_stream=G_int4_stream=0;
    coli_cuda_shutdown();
    QT_GATE_FREED_HOOK();
    /* QT_DEAD, still exclusive: a qt_init cannot slip in between the frees and
     * this line, and the losers parked in the claim above are released by the
     * broadcast in qt_gate_xleave(). */
    qg_set(QT_DEAD);
    qt_gate_xleave();
}

#endif /* COLI_CUDA */
