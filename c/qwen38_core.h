#include "qwen36_tier.h"   /* CUDA VRAM expert tier, shared with qwen36; CPU-only builds get the inline no-op stubs */
#include "vk_tier.h"       /* Vulkan routed-expert tier (COLI_VULKAN=1); builds without VK=1 get inline no-op stubs */
/* Native Qwen3.8-Flash-Next text core.
 *
 * This header is included once by qwen38.c after its tokenizer and protocol
 * helpers.  It intentionally consumes the official HF safetensors layout:
 * the multimodal wrapper's `model.language_model` namespace and the standalone
 * text model's `model` namespace are both accepted.  Vision tensors are read
 * by the vision tower only, and the MTP head's only under Q38_MTP=1
 * (q38_mtp_attach).
 */
#ifndef COLI_QWEN38_CORE_H
#define COLI_QWEN38_CORE_H
#include "kv_prefix.h"
#include "spec_draft.h"   /* prompt-lookup drafts and the gate that decides when drafting pays */
#include "expert_ffn.h"   /* the int4-g64 routed experts (experts-int4g64/ sidecar) run through its f32 kernel */
#include <pthread.h>   /* q38_ehit_mark publishes the lazy HITS table under a lock */
#ifdef COLI_VULKAN
#include "backend_vulkan.h" /* COLI_VULKAN=1: the int8 trunk on a Vulkan device */
static int g_vk_ready = 0;
static int g_vk_dense = 1;  /* COLI_VK_DENSE=0: the trunk stays on the CPU, the expert tier alone uses the device */
#endif

#define Q38_MAX_LAYERS 512
#define Q38_MAX_EXPERTS 1024
#define Q38_MAX_TOPK 256
#define Q38_MAX_PLE_PARTS 512
#define Q38_PREFILL_BATCH_ROWS 32
/* A speculative verify's rows: the picked token and up to five drafts (three from the
 * MTP head, five from prompt lookup). A rejection after row k restores the state the
 * verify copied after that row: one copy slot per row but the last. */
#define Q38_SPEC_ROWS 6
#define Q38_SPEC_SNAPS (Q38_SPEC_ROWS - 1)
#define Q38_PREFILL_WORKSPACE_BYTES (64u << 20)

typedef struct {
    int hidden, layers, vocab, max_positions, eos_id;
    float eps, theta;
    int hc_count, hc_rank, hc_width;
    int q_heads, kv_heads, head_dim, rotary_dim;
    int idx_qheads, idx_kheads, idx_dim, idx_budget, idx_ratio;
    int experts, topk, inter, shared_inter, norm_topk;
    int dn_kheads, dn_vheads, dn_kdim, dn_vdim, dn_convk, dn_conv_dim;
    int ple_layer, ple_dim, ple_convk, ngram_size, heads_per_ngram;
    int ngram_heads, ngram_head_dim, ngram_parts;
    uint8_t *is_attn;
    /* MTP head: mtp_num_hidden_layers (0 = the checkpoint has none, -1 = one
     * this engine does not run) and the RoPE base of its attention (config
     * mtp.rope_theta, else the model's) */
    int mtp_layers;
    float mtp_theta;
    /* vision: 0 = checkpoint di solo testo, o torre non caricata */
    int image_token;
    int vis_depth, vis_hidden, vis_heads, vis_inter, vis_patch;
    int vis_merge, vis_temporal, vis_in_ch, vis_out_hidden, vis_num_pos;
} Cfg;

typedef enum {
    Q38_WEIGHT_NONE = 0,
    Q38_WEIGHT_F32,
    Q38_WEIGHT_BF16,
    Q38_WEIGHT_FP8,
    Q38_WEIGHT_INT4G64             /* routed experts from the experts-int4g64/ sidecar */
} Q38WeightKind;

/* int4-g64 rows: ceil(cols/2) bytes of codes and one f32 scale per 64 inputs */
static inline int64_t q38_int4_row_bytes(int cols){ return ((int64_t)cols+1)/2; }
static inline int64_t q38_int4_groups(int cols){ return ((int64_t)cols+XF_BLOCK-1)/XF_BLOCK; }

typedef struct {
    void *data;
    float *scales;                 /* block-FP8: per 128x128 block; int4-g64: per row and 64 inputs */
    int rows, cols;
    int64_t elements, scale_count;
    Q38WeightKind kind;
    unsigned owns_data:1, owns_scales:1;
    int gpu;                       /* 0 = CPU; else 1 + tier handle of an int8 copy resident in VRAM (decode, S == 1) */
    int8_t *q8; float *q8sc;       /* the trunk's int8 rows on the CPU (default; Q38_TRUNK_CPU_INT8=0 keeps BF16): the same rows the GPU holds, met by an int8 activation in idot.h */
    void *vk; int vk_off;          /* COLI_VULKAN=1: device copy of q8, or of the BF16/F32 rows (uploaded at the first matmul), vk_off = upload failed, stays on the CPU */
    int vk_res;                    /* COLI_VULKAN=1: a resident matrix (q38_load_weight), never an expert slot that is refilled in place */
    char *vk_name;                 /* COLI_VULKAN=1: its tensor's name, to read it back from disk (q38_dho_reload) */
    int vk_fmt, vk_gone;           /* dense weights on the device only (COLI_VK_DENSE_HOST): the device copy's format, and 1 while no host copy is kept */
} Q38Weight;

typedef struct { float *norm; Q38Weight down, up, inject; } GatedResidual;

/* Persistent wall-clock counters.  They live on Model rather than in process
 * globals so Segment sessions and future multi-model serving cannot leak phase
 * time into one another.  Some categories intentionally overlap: architecture
 * phases (DeltaNet/QSA/PLE) contain resident matmuls, while the matmul counter
 * answers the orthogonal question "how much time is in dense kernels?". */
typedef enum {
    Q38_TM_EXPERT_READ = 0,
    Q38_TM_FP8_EXPAND,
    Q38_TM_ROUTED_EXPERT,
    Q38_TM_SHARED_EXPERT,
    Q38_TM_DENSE_MATMUL,
    Q38_TM_DELTANET,
    Q38_TM_QSA_INDEX,
    Q38_TM_QSA_ATTENTION,
    Q38_TM_PLE,
    Q38_TM_LM_HEAD,
    Q38_TM_COUNT
} Q38Timer;

typedef struct {
    double seconds[Q38_TM_COUNT];
    uint64_t forwards;
} Q38Timers;

typedef struct {
    GatedResidual attn_gr, mlp_gr;
    Q38Weight router, sh_g, sh_u, sh_d;
    float *sh_gate;
    Q38Weight q, k, v, o;
    float *qn, *kn;
    Q38Weight idx_qk;
    float *idx_qn, *idx_kn;
    Q38Weight dn_qkv, dn_z, dn_b, dn_a, dn_out;
    float *dn_conv;
    float *dn_dtbias, *dn_alog, *dn_norm;
    Q38Weight ple_key, ple_value;
    float *ple_norm_key, *ple_norm_query;
    float *ple_norm_conv, *ple_conv;
} Layer;

typedef struct {
    int eid;
    Q38Weight gate, up, down;
    uint64_t used;
    void *fp8_slab;
    int64_t fp8_slab_bytes;
    void *int4_slab;               /* one experts-int4g64/ record: codes, then scales */
    int64_t int4_slab_bytes;
} Slot;
typedef struct { Slot *slots; int *by_expert, n, cap; } LCache;

typedef struct {
    float *values;                 /* [expert][gate,up,down][block] */
    int64_t scale_count;
    int ready;                     /* 0 unknown, 1 resident, -1 incompatible */
} Q38ExpertScaleCache;

/* The routed experts as int4-g64 (q38_expert_int4_attach): where every
 * expert's record sits in the sidecar's layer files, and its geometry. */
typedef struct {
    shards S;                      /* <snap>/experts-int4g64/, one file per layer */
    int *fd;                       /* [layer] the file holding that layer's records */
    int64_t *off;                  /* [layer*experts + expert] where the record starts */
    int64_t code_bytes[3], scale_count[3];   /* gate, up, down */
    int64_t scale_off, record_bytes;
} Q38Int4Experts;

typedef enum {
    Q38_EXPERT_BATCH_FALLBACK_NONE = 0,
    Q38_EXPERT_BATCH_FALLBACK_DISABLED,
    Q38_EXPERT_BATCH_FALLBACK_CACHE_CAPACITY,
    Q38_EXPERT_BATCH_FALLBACK_SCALE_BANK,
    Q38_EXPERT_BATCH_FALLBACK_DUPLICATE,
    Q38_EXPERT_BATCH_FALLBACK_LAYOUT,
} Q38ExpertBatchFallback;

typedef struct {
    Cfg c;
    shards S;
    char prefix[32];
    Q38Weight embed, lm_head;
    GatedResidual final_gr;
    Layer *L;
    LCache *cache;
    uint8_t **ehit;                    /* experts routed this turn, for HITS (dashboard Brain) */
    Q38ExpertScaleCache *expert_scales;
    Q38Int4Experts *x4;                /* routed experts from experts-int4g64/, NULL = the snapshot's own */
    uint64_t clock, hits, miss;
    uint64_t expert_weight_reads, expert_scale_reads, expert_pair_reads;
    uint64_t expert_prefetch_ranges, expert_parallel_batches;
    uint64_t expert_scale_bytes;
    float **DN_rec, **DN_conv;
    float **K, **V, **IK;
    int kv_len, kv_cap, max_t;
    kv_prefix kvp; /* token identity of the live attention rows, not a snapshot */
    st_tensor *ple_parts[Q38_MAX_PLE_PARTS];
    char ple_part_names[Q38_MAX_PLE_PARTS][320];
    int64_t ple_part_start[Q38_MAX_PLE_PARTS + 1];
    int ple_part_count;
    float ple_weight_scale;
    int64_t ple_multipliers[3], ple_head_vocab[64], ple_head_offset[64];
    int64_t *ple_history;
    float *PLE_conv_state;
    int ple_history_len;
    int range_begin, range_end;
    int native_fp8, native_bf16, expert_prefetch, expert_parallel_reads;
    Q38ExpertBatchFallback expert_batch_fallback;
    int prefill_batch;
    uint64_t resident_weight_bytes;
    int trunk_table_built;         /* q38_trunk_offer_all ran for this load (the table is process-wide, the model is not) */
    double dense_load_s;
    /* vision. `vis_map` mappa la posizione ASSOLUTA nella sequenza alla riga di
     * `vis_rows`, oppure -1. Assoluta e non relativa al chunk: il prefill arriva
     * a pezzi, e un indice relativo darebbe l'immagine sbagliata al secondo
     * pezzo senza che niente protesti. */
    /* PLE prefetto: le righe gia' lette per il chunk in corso, o NULL. */
    float *ple_pref; int ple_pref_rows;
    Q38Vision vis;
    int vis_ready;
    float *vis_rows;
    int *vis_map, vis_map_len, vis_rows_n;
    Q38Timers timers;
    /* The MTP head (q38_mtp_attach, Q38_MTP=1). Its decoder layer is layer
     * index c.layers wherever an index names storage: L[], cache[],
     * expert_scales[], K/V/IK[] and the route-trace row. mtp_len counts the
     * head's KV rows that hold a verified pair (the streams at position p
     * with the token at p+1); mtp_pend keeps the streams of the rows after
     * them whose next token has not reached the head yet (one, or the rows of
     * a verify that stood: up to Q38_SPEC_ROWS, every one but the last with
     * its next token in mtp_pend_tok). */
    int mtp, mtp_wiring;
    char mtp_prefix[32];
    float *mtp_norm_emb, *mtp_norm_hid;
    Q38Weight mtp_fc_emb, mtp_fc_hid;
    GatedResidual mtp_mixer;
    int mtp_len, mtp_pend_n, mtp_pend_tok[Q38_SPEC_ROWS];
    float *mtp_pend;                  /* Q38_SPEC_ROWS rows of hc_width */
    /* A verify forward copies the DeltaNet and PLE state as it stands after
     * each of its first snap_rows rows, row r into slot r (0 = no copy), so a
     * draft rejected after row r+1 rolls back by swapping slot r in instead
     * of running the forward again. snap_slots: the slots allocated
     * (q38_spec_alloc). */
    int snap_rows, snap_slots;
    float **snap_rec[Q38_SPEC_SNAPS], **snap_conv[Q38_SPEC_SNAPS], *snap_ple[Q38_SPEC_SNAPS];
    int64_t snap_ple_history[Q38_SPEC_SNAPS][2];
    int snap_ple_history_len[Q38_SPEC_SNAPS];
#ifdef COLI_VULKAN
    void *vkchain;                 /* the dense chain's device state (qwen38_chain.h), NULL until it runs */
#endif
} Model;

/* Layer rows of storage: the c.layers decoder layers, plus the MTP head's at
 * index c.layers once it is attached. model_init_range sizes L[], cache[]
 * and expert_scales[] for that extra row up front; a loop over the rows
 * stops here so a model without the head never touches it. */
static inline int q38_layer_rows(const Model *m){ return m->c.layers+(m->mtp?1:0); }

static float *g_last_logit;
static int g_capture_last_logit;
/* Set for the length of an MTP verify forward (q38_spec_step): its rows must
 * get the bits a decode step gives them, see q38_weight_matmul and q38_moe. */
static int g_q38_rowwise;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static inline void q38_tm_add(Model *m,Q38Timer timer,double started) {
    m->timers.seconds[timer] += now_s() - started;
}

static Q38Timers q38_tm_delta(const Q38Timers *after,const Q38Timers *before) {
    Q38Timers delta={0};
    for(int i=0;i<Q38_TM_COUNT;i++)
        delta.seconds[i]=after->seconds[i]-before->seconds[i];
    delta.forwards=after->forwards-before->forwards;
    return delta;
}

#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1073741824.0; }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1048576.0; }
#endif

static float *falloc(int64_t n) {
    if (n < 0 || (uint64_t)n > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "invalid float allocation: %lld\n", (long long)n); exit(1);
    }
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p && n) { fprintf(stderr, "OOM allocating %lld floats\n", (long long)n); exit(1); }
    return p;
}

/* W is row-major [O,I], y=x@W^T. */
static void q38_matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0.f;
            for (int i = 0; i < I; i++) a += xs[i] * w[i];
            y[(int64_t)s * O + o] = a;
        }
    }
}

static void q38_weight_free(Q38Weight *weight) {
    if(!weight)return;
    if(weight->owns_data)free(weight->data);
    if(weight->owns_scales)free(weight->scales);
    free(weight->q8); free(weight->q8sc);
#ifdef COLI_VULKAN
    if(weight->vk)coli_vk_tensor_free((ColiVkTensor*)weight->vk);
    free(weight->vk_name);
#endif
    memset(weight,0,sizeof(*weight));
}

static void q38_weight_reserve(Q38Weight *weight,Q38WeightKind kind,int rows,int cols) {
    if(rows<=0||cols<=0||(int64_t)rows>INT64_MAX/cols){
        fprintf(stderr,"invalid weight geometry [%d,%d]\n",rows,cols);exit(1);
    }
    int64_t elements=(int64_t)rows*cols;
    int64_t scales=kind==Q38_WEIGHT_FP8?fp8_nblk(rows)*fp8_nblk(cols):0;
    if(weight->kind==kind&&weight->rows==rows&&weight->cols==cols&&
       weight->owns_data&&weight->data&&
       (!scales||(weight->owns_scales&&weight->scales)))return;
    q38_weight_free(weight);
    size_t element_size=kind==Q38_WEIGHT_F32?sizeof(float):
                        kind==Q38_WEIGHT_BF16?sizeof(uint16_t):
                        kind==Q38_WEIGHT_FP8?sizeof(uint8_t):0;
    if(!element_size||(uint64_t)elements>SIZE_MAX/element_size||
       (scales&&(uint64_t)scales>SIZE_MAX/sizeof(float))){
        fprintf(stderr,"unsupported or oversized weight geometry [%d,%d] kind=%d\n",
                rows,cols,(int)kind);exit(1);
    }
    weight->data=malloc((size_t)elements*element_size);
    if(scales)weight->scales=(float*)malloc((size_t)scales*sizeof(float));
    if(!weight->data||(scales&&!weight->scales)){
        fprintf(stderr,"OOM allocating weight [%d,%d] kind=%d\n",rows,cols,(int)kind);exit(1);
    }
    weight->rows=rows;weight->cols=cols;weight->elements=elements;
    weight->scale_count=scales;weight->kind=kind;
    weight->owns_data=1;weight->owns_scales=scales?1u:0u;
}

static uint64_t q38_weight_bytes(const Q38Weight *weight) {
    if(weight->kind==Q38_WEIGHT_INT4G64)
        return (uint64_t)weight->rows*(uint64_t)q38_int4_row_bytes(weight->cols)+
               (uint64_t)weight->scale_count*sizeof(float);
    uint64_t element_size=weight->kind==Q38_WEIGHT_F32?sizeof(float):
                          weight->kind==Q38_WEIGHT_BF16?sizeof(uint16_t):
                          weight->kind==Q38_WEIGHT_FP8?sizeof(uint8_t):0;
    return (uint64_t)weight->elements*element_size+
           (uint64_t)weight->scale_count*sizeof(float);
}

/* Native BF16 storage with FP32 activations and accumulation.  This deliberately
 * does not round activations to BF16 or use BF16 dot-product instructions: it is
 * the storage-equivalent form of the existing st_read_f32 reference. */
static void q38_matmul_bf16(float *y,const float *x,const uint16_t *W,
                            int S,int I,int O) {
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;float a=0.f;
            for(int i=0;i<I;i++)a+=xs[i]*bf16_to_f32(w[i]);
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* The routed experts as the checkpoint ships them: e4m3 bytes with one f32
 * scale per 128x128 block. quant.h's matmul_fp8 decodes every byte through a
 * 256-entry table, one gather per weight; this kernel decodes eight bytes at a
 * time in registers (quant.h e4m3_decode8: a shift and one multiply, NaNs
 * kept) and multiplies them with FMA. The block scale still applies once per
 * block and the blocks still add in double, so the result differs from the
 * scalar kernel only by the float summation order inside a block. For a
 * batch of rows (prefill) the block is decoded once and held while every row
 * runs through it: the matrix streams past once, not once per row.
 * Q38_FP8_KERNEL=scalar restores the table kernel (bisecting a difference). */
static int q38_fp8_vector_on(void) {
    static int v=-1;
    if(v<0){ const char *e=getenv("Q38_FP8_KERNEL"); v=!(e&&!strcmp(e,"scalar")); }
    return v;
}
#ifdef __AVX2__
#define Q38_FP8_ROWS 8
static void q38_matmul_fp8_vec(float *y,const float *x,const uint8_t *q8,
                               const float *bscale,int S,int I,int O) {
    int64_t nblkI=fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q8+(int64_t)o*I;
        const float *scl=bscale+((int64_t)o/FP8_BLOCK)*nblkI;
        if(S==1){
            /* decode: the weight bytes are the traffic, so decode and multiply in one pass */
            double a=0;
            for(int64_t bi=0;bi*FP8_BLOCK<I;bi++){
                int base=(int)(bi*FP8_BLOCK),blen=I-base<FP8_BLOCK?I-base:FP8_BLOCK,i=0;
                __m256 acc=_mm256_setzero_ps();
                for(;i+8<=blen;i+=8)
                    acc=_mm256_fmadd_ps(e4m3_decode8(w+base+i),_mm256_loadu_ps(x+base+i),acc);
                float part=hsum256(acc);
                for(;i<blen;i++)part+=e4m3_decode(w[base+i])*x[base+i];
                a+=(double)part*scl[bi];
            }
            y[o]=(float)a; continue;
        }
        for(int s0=0;s0<S;s0+=Q38_FP8_ROWS){
            int ns=S-s0<Q38_FP8_ROWS?S-s0:Q38_FP8_ROWS; double a[Q38_FP8_ROWS]={0};
            for(int64_t bi=0;bi*FP8_BLOCK<I;bi++){
                int base=(int)(bi*FP8_BLOCK),blen=I-base<FP8_BLOCK?I-base:FP8_BLOCK,i=0;
                float wf[FP8_BLOCK];
                for(;i+8<=blen;i+=8)_mm256_storeu_ps(wf+i,e4m3_decode8(w+base+i));
                for(;i<blen;i++)wf[i]=e4m3_decode(w[base+i]);
                float sc=scl[bi];
                for(int r=0;r<ns;r++){
                    const float *xs=x+(int64_t)(s0+r)*I+base; int k=0;
                    __m256 acc=_mm256_setzero_ps();
                    for(;k+8<=blen;k+=8)
                        acc=_mm256_fmadd_ps(_mm256_loadu_ps(wf+k),_mm256_loadu_ps(xs+k),acc);
                    float part=hsum256(acc);
                    for(;k<blen;k++)part+=wf[k]*xs[k];
                    a[r]+=(double)part*sc;
                }
            }
            for(int r=0;r<ns;r++)y[(int64_t)(s0+r)*O+o]=(float)a[r];
        }
    }
}
#endif
static void q38_matmul_fp8(float *y,const float *x,const uint8_t *q8,const float *bscale,
                           int S,int I,int O) {
#ifdef __AVX2__
    if(q38_fp8_vector_on()){ q38_matmul_fp8_vec(y,x,q8,bscale,S,I,O); return; }
#endif
    matmul_fp8(y,x,q8,bscale,S,I,O);
}

/* The int4-g64 routed experts (q38_expert_int4_attach) through expert_ffn.h's
 * f32 kernel: every code is exact in f32, one fma per element and one per
 * block for its scale, and the activations stay f32 as on the FP8 path. A row
 * is planar blocks of 64; a width that is not a multiple of 64 (only the tiny
 * fixtures) keeps its tail in pairs and finishes with the last group's scale.
 * Each output row is computed the same way whatever S is, so decode and the
 * prefill batch give the same bits. */
static void q38_matmul_int4g64(float *y,const float *x,const uint8_t *codes,
                               const float *scales,int S,int I,int O) {
    int64_t row_bytes=q38_int4_row_bytes(I),groups=q38_int4_groups(I);
    int body=I/XF_BLOCK*XF_BLOCK;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=codes+(int64_t)o*row_bytes;
        const float *sc=scales+(int64_t)o*groups;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            float a=body?xf_dot_f32(w,sc,xs,body):0.f;
            if(body<I){
                float tail=0.f;
                for(int i=body;i<I;i++)
                    tail+=(float)((int)((w[i>>1]>>((i&1)*4))&15)-8)*xs[i];
                a=fmaf(tail,sc[groups-1],a);
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

#ifdef COLI_VULKAN
/* COLI_VULKAN=1: a resident matrix answers from the Vulkan device, decode and
 * prefill alike, with the weights the CPU would read and in the order
 * q38_weight_matmul picks them: the trunk's int8 rows when there are (q8, one
 * scale per row, fmt 1), else the rows as loaded, BF16 (fmt 11) or F32
 * (fmt 10, Q38_NATIVE_BF16=0), times the f32 activation like
 * q38_matmul_bf16 / q38_matmul. For the int8 rows the activation stays f32
 * on the device where the CPU's integer kernel rounds it to int8; the float
 * rows differ from the CPU only in the order of the sums. Only matrices
 * loaded by q38_load_weight (vk_res) qualify: the routed experts' slots are
 * refilled in place, so a copy cached in them would go stale, and they stay
 * on the CPU in every format. The device copy lives in weight->vk from the
 * first call; a failed upload sets vk_off and the matrix stays on the CPU.
 * The backend has one command buffer: never from a parallel region. */
static int q38_vk_eligible(const Q38Weight *w) {
    return w->q8 || w->vk_gone || (w->vk_res && w->data &&
                     (w->kind==Q38_WEIGHT_BF16 || w->kind==Q38_WEIGHT_F32));
}
/* The format of the weight's device copy: the int8 rows, else the rows as loaded; with
 * the host copy dropped (vk_gone), the one it was uploaded in. */
static int q38_vk_fmt(const Q38Weight *w) {
    return w->vk_gone?w->vk_fmt:w->q8?1:w->kind==Q38_WEIGHT_BF16?11:10;
}
static unsigned g_q38_vk_placed[3];   /* uploads by format: int8 rows, bf16, f32 */
/* A partial dense chain (qwen38_chain.h, q38c_start): the first N layers' matrices are on
 * the device, the other layers and the head stay on the CPU, so nothing that is not on
 * the device already goes up any more (docs/vulkan.md, "A partial chain"). */
static int g_q38_vk_noup;
static void q38_dho_reload(Q38Weight *w);   /* below, beside the trunk's int8 rows */
/* The weight's device copy, uploaded on the first call; NULL when it is not eligible
 * or the upload failed (vk_off). The dense chain (qwen38_chain.h) reads the same one. */
static ColiVkTensor *q38_vk_tensor(const Q38Weight *weight) {
    if(!weight||weight->vk_off||!q38_vk_eligible(weight))return NULL;
    Q38Weight *w=(Q38Weight*)weight;   /* vk is a cache in a weight the forward pass treats as read-only */
    ColiVkTensor **t=(ColiVkTensor**)&w->vk;
    if(*t)return *t;
    if(w->vk_gone||g_q38_vk_noup)return NULL;
    int fmt=q38_vk_fmt(w);
    const void *wq=w->q8?(const void*)w->q8:(const void*)w->data;
    const float *sc=w->q8?w->q8sc:NULL;   /* fmt 10/11: no scales */
    if(!coli_vk_tensor_ensure(t,wq,sc,fmt,w->cols,w->rows,0)){w->vk_off=1;return NULL;}
    g_q38_vk_placed[fmt==1?0:fmt==11?1:2]++;
    return *t;
}
static int q38_vk_matmul(float *y,const float *x,const Q38Weight *weight,int S,int I,int O) {
#ifdef _OPENMP
    if(omp_in_parallel())return 0;
#endif
    if(weight->vk_off||S<1||S>65535)return 0;
    Q38Weight *w=(Q38Weight*)weight;
    int fmt=q38_vk_fmt(w);
    const void *wq=w->q8?(const void*)w->q8:(const void*)w->data;
    const float *sc=w->q8?w->q8sc:NULL;   /* fmt 10/11: no scales */
    ColiVkTensor *t=q38_vk_tensor(w);
    if(!t)return 0;
    return coli_vk_matmul(&t,y,x,wq,sc,fmt,S,I,O,0);
}
/* One line at the end of a run or a serve turn: how many matmuls the device
 * really answered, so a test can tell a used path from an initialised one. */
static void q38_vk_report(void) {
    if(!g_vk_ready)return;
    size_t bytes=0,tensors=0;
    coli_vk_mem_info(&bytes,&tensors);
    fprintf(stderr,"[VK] qwen38: %llu matmuls on the GPU (%zu matrices resident, %.1f MiB; placed int8 %u, bf16 %u, f32 %u)\n",
            coli_vk_matmul_calls(),tensors,bytes/1048576.0,
            g_q38_vk_placed[0],g_q38_vk_placed[1],g_q38_vk_placed[2]);
}
#endif
static void q38_weight_matmul(float *y,const float *x,const Q38Weight *weight,
                              int S,int I,int O) {
    /* Every CPU kernel below computes a row the same way whatever S is, so an
     * MTP verify forward (S=2) reproduces two decode steps bit for bit. A
     * device answers S == 1 and S > 1 by different routes (the VRAM trunk
     * only takes decode rows; a Vulkan batch is its own dispatch), so under
     * a verify such a matrix runs its rows one at a time, as decode does. */
    if(S>1&&g_q38_rowwise&&weight&&
       (weight->gpu
#ifdef COLI_VULKAN
        ||(g_vk_ready&&g_vk_dense&&q38_vk_eligible(weight)&&(weight->vk||!g_q38_vk_noup))
#endif
       )){
        for(int s=0;s<S;s++)q38_weight_matmul(y+(int64_t)s*O,x+(int64_t)s*I,weight,1,I,O);
        return;
    }
    /* A matrix the tier placed in VRAM (q38_trunk_place) answers a decode
     * GEMV from there; prefill rows and any failure take the CPU path below,
     * so the BF16 copy stays the reference for everything but S == 1. */
    if(S==1&&weight&&weight->gpu&&weight->rows==O&&weight->cols==I&&
       qt_dense_matmul(weight->gpu-1,y,x,I,O))return;
#ifdef COLI_VULKAN
    if(g_vk_ready&&g_vk_dense&&weight&&weight->rows==O&&weight->cols==I&&q38_vk_eligible(weight)&&
       q38_vk_matmul(y,x,weight,S,I,O))return;
    /* the CPU needs a matrix the device holds alone (a lost device): read it back */
    if(weight&&weight->vk_gone)q38_dho_reload((Q38Weight*)weight);
#endif
    if(weight&&weight->q8&&weight->rows==O&&weight->cols==I){
        /* the trunk's int8 rows (the same the GPU holds) meet an int8
         * activation in the integer kernel: x quantized once per row with one
         * scale, then maddubs / vpdpbusd dot products (idot.h). Decode and
         * prefill take the same path, so GPU or not the trunk quantization is
         * the only thing that separates the output from the BF16 run. */
        int8_t *xq=(int8_t*)malloc((size_t)S*I); float *sx=(float*)malloc((size_t)S*sizeof(float));
        if(!xq||!sx){fprintf(stderr,"OOM activation quantization\n");exit(1);}
        for(int s=0;s<S;s++)sx[s]=dense_act_i8(x+(int64_t)s*I,I,xq+(int64_t)s*I,NULL);
        matmul_q_idot(y,xq,sx,weight->q8,weight->q8sc,S,I,O);
        free(xq);free(sx);
        return;
    }
    if(!weight||weight->rows!=O||weight->cols!=I||!weight->data){
        fprintf(stderr,"invalid matmul weight: have [%d,%d] kind=%d, need [%d,%d]\n",
                weight?weight->rows:0,weight?weight->cols:0,
                weight?(int)weight->kind:0,O,I);exit(1);
    }
    if(weight->kind==Q38_WEIGHT_F32)
        q38_matmul(y,x,(const float*)weight->data,S,I,O);
    else if(weight->kind==Q38_WEIGHT_BF16)
        q38_matmul_bf16(y,x,(const uint16_t*)weight->data,S,I,O);
    else if(weight->kind==Q38_WEIGHT_FP8&&weight->scales)
        q38_matmul_fp8(y,x,(const uint8_t*)weight->data,weight->scales,S,I,O);
    else if(weight->kind==Q38_WEIGHT_INT4G64&&weight->scales)
        q38_matmul_int4g64(y,x,(const uint8_t*)weight->data,weight->scales,S,I,O);
    else {fprintf(stderr,"unsupported matmul weight kind %d\n",(int)weight->kind);exit(1);}
}

static void q38_weight_row(const Q38Weight *weight,int row,float *out) {
    if(!weight||row<0||row>=weight->rows||!weight->data){
        fprintf(stderr,"invalid weight row %d\n",row);exit(1);
    }
    if(weight->kind==Q38_WEIGHT_F32)
        memcpy(out,(const float*)weight->data+(int64_t)row*weight->cols,
               (size_t)weight->cols*sizeof(float));
    else if(weight->kind==Q38_WEIGHT_BF16){
        const uint16_t *src=(const uint16_t*)weight->data+(int64_t)row*weight->cols;
        for(int i=0;i<weight->cols;i++)out[i]=bf16_to_f32(src[i]);
    } else {fprintf(stderr,"weight kind %d has no dense row view\n",(int)weight->kind);exit(1);}
}

static void q38_dense_matmul(Model *m,float *y,const float *x,const Q38Weight *weight,
                             int S,int I,int O) {
    double started=now_s();
    q38_weight_matmul(y,x,weight,S,I,O);
    q38_tm_add(m,Q38_TM_DENSE_MATMUL,started);
}

static inline float q38_sigmoid(float x) {
    if (x >= 0.f) { float z=expf(-x); return 1.f/(1.f+z); }
    float z=expf(x); return z/(1.f+z);
}
static inline float q38_silu(float x) { return x * q38_sigmoid(x); }
static inline float q38_softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

/* Qwen4-Exp RMSNorms are zero-centered: the learned scale is 1+weight. */
static void q38_rms0(float *out, const float *x, const float *w, int n, float eps) {
    double ss=0.0; for (int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for (int i=0;i<n;i++) out[i]=x[i]*r*(1.f+w[i]);
}

/* DeltaNet's RMSNormGated is inherited from Qwen3-Next and is not zero-centered. */
static void q38_rmsg(float *out,const float *x,const float *gate,const float *w,
                     int n,float eps,int sigmoid_gate) {
    double ss=0.0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++) out[i]=x[i]*r*w[i]*(sigmoid_gate?q38_sigmoid(gate[i]):q38_silu(gate[i]));
}

static void q38_rope(float *x, int dim, int rotary_dim, int pos, float theta) {
    int half=rotary_dim/2;
    for(int i=0;i<half;i++) {
        float ang=(float)pos/powf(theta,(float)(2*i)/rotary_dim);
        float co=cosf(ang), si=sinf(ang), a=x[i], b=x[i+half];
        x[i]=a*co-b*si; x[i+half]=b*co+a*si;
    }
    (void)dim;
}

static jval *q38_obj(jval *o,const char *key) {
    jval *v=json_get(o,key);
    if(v&&v->t!=J_OBJ){fprintf(stderr,"config.json: %s must be an object\n",key);exit(1);}
    return v;
}
static double q38_num(jval *o,const char *key,double def,int required) {
    jval *v=json_get(o,key);
    if(v&&v->t==J_NUM) return v->num;
    if(v){fprintf(stderr,"config.json: %s must be numeric\n",key);exit(1);}
    if(required){fprintf(stderr,"config.json: missing numeric %s\n",key);exit(1);} return def;
}
/* JSON numbers are doubles in the small parser.  Never narrow an unchecked
 * value to an int: besides accepting fractional dimensions, a huge finite
 * value can become implementation-defined before q38_validate_cfg sees it. */
static int q38_num_int(jval *o,const char *key,double def,int required,
                       int min_value,int max_value) {
    double value=q38_num(o,key,def,required);
    if(!isfinite(value)||floor(value)!=value||value<(double)min_value||
       value>(double)max_value){
        fprintf(stderr,"config.json: %s must be an integer in [%d,%d]\n",
                key,min_value,max_value);exit(1);
    }
    return (int)value;
}
static int q38_derived_product(const char *name,int left,int right) {
    if(left<0||right<0||(uint64_t)left*(uint64_t)right>(uint64_t)INT_MAX){
        fprintf(stderr,"[qwen38 config] derived %s overflows int\n",name);exit(1);
    }
    return left*right;
}
static int q38_derived_sum(const char *name,int left,int right) {
    if(left<0||right<0||left>INT_MAX-right){
        fprintf(stderr,"[qwen38 config] derived %s overflows int\n",name);exit(1);
    }
    return left+right;
}
static int q38_bool(jval *o,const char *key,int def) {
    jval *v=json_get(o,key);
    if(v&&v->t!=J_BOOL){fprintf(stderr,"config.json: %s must be boolean\n",key);exit(1);}
    return v?v->boolean:def;
}
static const char *q38_string(jval *o,const char *key,const char *def) {
    jval *v=json_get(o,key);
    if(v&&v->t!=J_STR){fprintf(stderr,"config.json: %s must be a string\n",key);exit(1);}
    return v?v->str:def;
}
static void q38_require_string(jval *o,const char *key,const char *expected) {
    const char *value=q38_string(o,key,expected);
    if(strcmp(value,expected)){
        fprintf(stderr,"config.json: unsupported %s=%s (expected %s)\n",
                key,value,expected);exit(1);
    }
}
static void q38_require_present_string(jval *o,const char *key,
                                        const char *expected) {
    jval *value=json_get(o,key);
    if(!value||value->t!=J_STR||strcmp(value->str,expected)){
        fprintf(stderr,"config.json: %s must be explicitly set to %s\n",
                key,expected);exit(1);
    }
}
static void q38_require_bool(jval *o,const char *key,int expected) {
    int value=q38_bool(o,key,expected);
    if(value!=expected){
        fprintf(stderr,"config.json: unsupported %s=%s (expected %s)\n",key,
                value?"true":"false",expected?"true":"false");exit(1);
    }
}

static void q38_load_cfg(Cfg *c,const char *snap) {
    memset(c,0,sizeof(*c));
    char path[2048]; snprintf(path,sizeof path,"%s/config.json",snap);
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);} fseek(f,0,SEEK_END);
    long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0||n>(256L<<20)){fprintf(stderr,"invalid config size\n");exit(1);}
    char *buf=(char*)malloc((size_t)n+1),*arena=NULL;
    if(!buf||fread(buf,1,(size_t)n,f)!=(size_t)n){fprintf(stderr,"cannot read %s\n",path);exit(1);}
    fclose(f); buf[n]=0; jval *root=json_parse(buf,&arena);
    if(!root||root->t!=J_OBJ){fprintf(stderr,"invalid %s\n",path);exit(1);}
    jval *tc=q38_obj(root,"text_config"); if(!tc) tc=root;
    const char *mt=jstr(tc,"model_type");
    if(!mt||strcmp(mt,"qwen4_exp_text")){fprintf(stderr,"unsupported text model_type: %s\n",mt?mt:"(missing)");exit(1);}
    q38_require_string(tc,"hidden_act","silu");
    /* Upstream's omitted/None default resolves to hidden_act (SiLU), whereas
     * this implementation deliberately uses the released sigmoid gate. */
    q38_require_present_string(tc,"output_gate_type","sigmoid");
    q38_require_bool(tc,"attention_bias",0);
    q38_require_bool(tc,"tie_word_embeddings",0);
    c->hidden=q38_num_int(tc,"hidden_size",0,1,0,INT_MAX);
    c->layers=q38_num_int(tc,"num_hidden_layers",0,1,0,Q38_MAX_LAYERS);
    c->vocab=q38_num_int(tc,"vocab_size",0,1,0,INT_MAX);
    c->max_positions=q38_num_int(tc,"max_position_embeddings",0,1,0,INT_MAX);
    c->eos_id=q38_num_int(tc,"eos_token_id",-1,1,INT_MIN,INT_MAX);
    double eps=q38_num(tc,"rms_norm_eps",1e-6,0);
    if(!isfinite(eps)||eps<=0.0||eps>FLT_MAX){fprintf(stderr,"invalid rms_norm_eps\n");exit(1);}
    c->eps=(float)eps;
    jval *rp=q38_obj(tc,"rope_parameters");
    if(rp)q38_require_string(rp,"rope_type","default");
    double theta=rp?q38_num(rp,"rope_theta",10000,0):q38_num(tc,"rope_theta",10000,0);
    if(!isfinite(theta)||theta<=0.0||theta>FLT_MAX){fprintf(stderr,"invalid rope_theta\n");exit(1);}
    c->theta=(float)theta;
    c->hc_count=q38_num_int(tc,"hc_count",4,0,0,INT_MAX);
    c->hc_rank=q38_num_int(tc,"hc_lowrank",320,0,0,INT_MAX);
    c->hc_width=q38_derived_product("hc_width",c->hc_count,c->hidden);
    c->q_heads=q38_num_int(tc,"num_attention_heads",0,1,0,INT_MAX);
    c->kv_heads=q38_num_int(tc,"num_key_value_heads",0,1,0,INT_MAX);
    c->head_dim=q38_num_int(tc,"head_dim",0,1,0,INT_MAX);
    double partial=rp?q38_num(rp,"partial_rotary_factor",q38_num(tc,"partial_rotary_factor",1,0),0):q38_num(tc,"partial_rotary_factor",1,0);
    double rotary_product=(double)c->head_dim*partial;
    if(!isfinite(partial)||!isfinite(rotary_product)||rotary_product<0||
       rotary_product>(double)INT_MAX){
        fprintf(stderr,"[qwen38 config] derived rotary_dim overflows int\n");exit(1);
    }
    /* Upstream Qwen4ExpTextConfig derives this with Python int(), i.e. truncation
     * toward zero rather than rounding to the nearest dimension. */
    c->rotary_dim=(int)rotary_product;
    c->idx_qheads=q38_num_int(tc,"indexer_n_heads",0,1,0,INT_MAX);
    c->idx_kheads=q38_num_int(tc,"indexer_kv_heads",0,1,0,INT_MAX);
    c->idx_dim=q38_num_int(tc,"indexer_head_dim",0,1,0,INT_MAX);
    c->idx_budget=q38_num_int(tc,"indexer_budget",0,1,0,INT_MAX);
    c->idx_ratio=q38_num_int(tc,"indexer_compress_ratio",0,1,0,INT_MAX);
    c->experts=q38_num_int(tc,"num_experts",0,1,0,INT_MAX);
    c->topk=q38_num_int(tc,"num_experts_per_tok",0,1,0,INT_MAX);
    c->inter=q38_num_int(tc,"moe_intermediate_size",0,1,0,INT_MAX);
    c->shared_inter=q38_num_int(tc,"shared_expert_intermediate_size",0,1,0,INT_MAX);
    c->norm_topk=q38_bool(tc,"norm_topk_prob",1);
    c->dn_kheads=q38_num_int(tc,"linear_num_key_heads",0,1,0,INT_MAX);
    c->dn_vheads=q38_num_int(tc,"linear_num_value_heads",0,1,0,INT_MAX);
    c->dn_kdim=q38_num_int(tc,"linear_key_head_dim",0,1,0,INT_MAX);
    c->dn_vdim=q38_num_int(tc,"linear_value_head_dim",0,1,0,INT_MAX);
    c->dn_convk=q38_num_int(tc,"linear_conv_kernel_dim",0,1,0,INT_MAX);
    int dn_qk=q38_derived_product("deltanet_qk",c->dn_kheads,c->dn_kdim);
    dn_qk=q38_derived_product("deltanet_qk_twice",2,dn_qk);
    c->dn_conv_dim=q38_derived_sum("dn_conv_dim",dn_qk,
                                    q38_derived_product("deltanet_v",c->dn_vheads,c->dn_vdim));
    c->ple_dim=q38_num_int(tc,"ple_embed_dim",c->hidden,0,0,INT_MAX);
    c->ple_convk=q38_num_int(tc,"ple_conv_kernel_size",4,0,0,INT_MAX);
    c->ngram_size=q38_num_int(tc,"ngram_size",3,0,0,INT_MAX);
    c->heads_per_ngram=q38_num_int(tc,"heads_per_ngram",8,0,0,INT_MAX);
    c->ngram_heads=(c->ngram_size>0)?q38_derived_product("ngram_heads",c->ngram_size-1,c->heads_per_ngram):0;
    if(c->ngram_heads<=0){fprintf(stderr,"[qwen38 config] derived ngram_heads invalid\n");exit(1);}
    c->ngram_head_dim=c->ple_dim/c->ngram_heads;
    c->ngram_parts=q38_num_int(tc,"split_ngram_parts",1,0,0,INT_MAX);
    c->ple_layer=-1;
    jval *pl=json_get(tc,"ple_layer_ids");
    if(pl&&pl->t==J_ARR&&pl->len){
        if(pl->len!=1||pl->kids[0]->t!=J_NUM){fprintf(stderr,"only one PLE layer is supported\n");exit(1);}
        double raw=pl->kids[0]->num;
        if(!isfinite(raw)||floor(raw)!=raw||raw<1||raw>(double)INT_MAX){
            fprintf(stderr,"config.json: ple_layer_ids[0] must be a positive integer\n");exit(1);
        }
        int one_based=(int)raw;
        c->ple_layer=one_based-1;
    }
    c->is_attn=(uint8_t*)calloc((size_t)c->layers,1);
    jval *lt=json_get(tc,"layer_types");
    if(!lt||lt->t!=J_ARR||lt->len!=c->layers){fprintf(stderr,"config layer_types must have %d entries\n",c->layers);exit(1);}
    for(int i=0;i<c->layers;i++){
        const char *s=lt->kids[i]->t==J_STR?lt->kids[i]->str:NULL;
        if(!s){fprintf(stderr,"invalid layer_types[%d]\n",i);exit(1);}
        if(!strcmp(s,"linear_attention")) c->is_attn[i]=0;
        else if(!strcmp(s,"full_attention")||!strcmp(s,"qwen_sparse_attention")) c->is_attn[i]=1;
        else {fprintf(stderr,"unsupported layer type %s\n",s);exit(1);}
    }
    /* The MTP head is described next to the text model (or at the root of a
     * multimodal config). Only read here; q38_mtp_attach decides whether the
     * engine can run it, so a config this engine cannot draft with still
     * loads for plain decoding. */
    {
        jval *at=json_get(tc,"mtp_num_hidden_layers")?tc:root;
        c->mtp_layers=q38_num_int(at,"mtp_num_hidden_layers",0,0,0,64);
        c->mtp_theta=c->theta;
        jval *mc=q38_obj(tc,"mtp"); if(!mc) mc=q38_obj(root,"mtp");
        if(mc){
            double mt=q38_num(mc,"rope_theta",theta,0);
            if(!isfinite(mt)||mt<=0.0||mt>FLT_MAX){fprintf(stderr,"invalid mtp.rope_theta\n");exit(1);}
            c->mtp_theta=(float)mt;
            jval *types=json_get(mc,"layer_types");
            if(types&&types->t==J_ARR)
                for(int i=0;i<types->len;i++){
                    const char *s=types->kids[i]->t==J_STR?types->kids[i]->str:"";
                    /* the head's one layer is an attention layer; anything else is a head this engine does not have */
                    if(strcmp(s,"full_attention")&&strcmp(s,"qwen_sparse_attention")) c->mtp_layers=-1;
                }
        }
        jval *dedicated=json_get(tc,"mtp_use_dedicated_embeddings"); if(!dedicated) dedicated=json_get(root,"mtp_use_dedicated_embeddings");
        if(dedicated&&dedicated->t==J_BOOL&&dedicated->boolean) c->mtp_layers=-1;   /* mtp.embed_tokens: not this head */
    }
    /* Vision: opzionale. Un checkpoint di solo testo non ha vision_config, e in
     * quel caso la torre resta spenta invece di rifiutare il modello. La
     * geometria viene dal file, non da costanti qui: una torre di misura diversa
     * deve fallire dicendo cosa non torna, non leggere pesi della misura
     * sbagliata. */
    {
        jval *vc = q38_obj(root, "vision_config");
        if (vc) {
            c->vis_depth      = q38_num_int(vc,"depth",0,1,1,1024);
            c->vis_hidden     = q38_num_int(vc,"hidden_size",0,1,1,65536);
            c->vis_heads      = q38_num_int(vc,"num_heads",0,1,1,1024);
            c->vis_inter      = q38_num_int(vc,"intermediate_size",0,1,1,262144);
            c->vis_patch      = q38_num_int(vc,"patch_size",16,0,1,512);
            c->vis_merge      = q38_num_int(vc,"spatial_merge_size",2,0,1,8);
            c->vis_temporal   = q38_num_int(vc,"temporal_patch_size",2,0,1,8);
            c->vis_in_ch      = q38_num_int(vc,"in_channels",3,0,1,8);
            c->vis_out_hidden = q38_num_int(vc,"out_hidden_size",c->hidden,0,1,65536);
            c->vis_num_pos    = q38_num_int(vc,"num_position_embeddings",0,1,1,1<<20);
            c->image_token    = q38_num_int(root,"image_token_id",-1,0,0,INT_MAX);
        }
    }
    json_free(root); free(buf); free(arena);
}

#define Q38_NEED(x,...) do{if(!(x)){fprintf(stderr,"[qwen38 config] ");fprintf(stderr,__VA_ARGS__);fprintf(stderr," -- refusing\n");exit(1);}}while(0)
static void q38_validate_cfg(const Cfg *c) {
    Q38_NEED(c->hidden>0&&c->hidden<=65536,"hidden_size=%d",c->hidden);
    if (c->vis_depth) {
        /* La torre proietta direttamente nello spazio del testo: se le due
         * dimensioni non coincidono i token immagine finirebbero nel posto
         * giusto con i numeri sbagliati, che e' peggio di un rifiuto. */
        Q38_NEED(c->vis_out_hidden==c->hidden,
                 "vision out_hidden_size=%d but text hidden_size=%d",
                 c->vis_out_hidden,c->hidden);
        Q38_NEED(c->vis_hidden%c->vis_heads==0,
                 "vision hidden_size=%d not divisible by num_heads=%d",
                 c->vis_hidden,c->vis_heads);
        Q38_NEED(c->image_token>=0&&c->image_token<c->vocab,
                 "image_token_id=%d outside vocabulary",c->image_token);
    }
    Q38_NEED(c->layers>0&&c->layers<=Q38_MAX_LAYERS,"layers=%d",c->layers);
    Q38_NEED(c->vocab>0&&c->max_positions>0&&
             c->max_positions<=QWEN38_ATTN_MAX_CTX,"vocab/context invalid");
    Q38_NEED(isfinite(c->eps)&&c->eps>0.f&&isfinite(c->theta)&&c->theta>0.f,"RoPE/norm constants invalid");
    Q38_NEED(c->eos_id>=0&&c->eos_id<c->vocab,"eos=%d",c->eos_id);
    Q38_NEED(c->hc_count>1&&c->hc_count<=16&&c->hc_rank>0&&
             c->hc_rank<=65536&&c->hc_width>0,"gated residual dimensions invalid");
    Q38_NEED(c->q_heads>0&&c->kv_heads>0&&c->q_heads%c->kv_heads==0,"attention heads invalid");
    Q38_NEED(c->head_dim>0&&c->rotary_dim>0&&!(c->rotary_dim&1)&&c->rotary_dim<=c->head_dim,"RoPE dimensions invalid");
    Q38_NEED(c->head_dim<=INT_MAX/2&&c->q_heads<=INT_MAX/(2*c->head_dim)&&
             c->kv_heads<=INT_MAX/c->head_dim,"attention projection dimensions overflow");
    Q38_NEED(c->idx_qheads>0&&c->idx_kheads==1&&c->idx_dim>=c->rotary_dim,"indexer dimensions invalid");
    Q38_NEED(c->idx_ratio>0&&c->idx_budget>0&&c->idx_budget%c->idx_ratio==0&&
             c->idx_budget<=INT_MAX-c->idx_ratio+1,"indexer budget invalid");
    Q38_NEED(c->idx_qheads<INT_MAX&&c->idx_qheads<=INT_MAX/c->idx_dim-c->idx_kheads,
             "indexer projection dimensions overflow");
    Q38_NEED(c->experts>0&&c->experts<=Q38_MAX_EXPERTS,"experts=%d",c->experts);
    Q38_NEED(c->topk>0&&c->topk<=Q38_MAX_TOPK&&c->topk<=c->experts,"topk=%d",c->topk);
    Q38_NEED(c->inter<=INT_MAX/2,"expert projection dimensions overflow");
    Q38_NEED(c->inter>0&&c->shared_inter>0,"MoE widths invalid");
    Q38_NEED(c->dn_vheads>0&&c->dn_kheads>0&&c->dn_vheads%c->dn_kheads==0,"DeltaNet heads invalid");
    Q38_NEED(c->dn_kdim>0&&c->dn_vdim>0&&c->dn_vdim<=512&&c->dn_convk>=2&&c->dn_conv_dim>0&&
             c->dn_vdim<=INT_MAX/c->dn_vheads&&c->dn_kdim<=INT_MAX/c->dn_vdim,
             "DeltaNet dimensions invalid");
    Q38_NEED((uint64_t)c->dn_vheads<=SIZE_MAX/sizeof(float)/(uint64_t)c->dn_kdim/(uint64_t)c->dn_vdim&&
             (uint64_t)c->dn_conv_dim<=SIZE_MAX/sizeof(float)/(uint64_t)(c->dn_convk-1),
             "DeltaNet state dimensions overflow");
    Q38_NEED(c->ngram_size==3&&c->ngram_heads>0&&c->ngram_heads<=64&&
             c->ple_dim>0&&c->ple_dim%c->ngram_heads==0&&
             c->ngram_head_dim>0&&c->ngram_head_dim<=512&&
             c->ple_convk>=2&&c->ple_convk<=INT_MAX/c->ngram_size,
             "PLE dimensions invalid");
    Q38_NEED((uint64_t)c->hc_width<=SIZE_MAX/sizeof(float)/(uint64_t)(c->ple_convk-1)/(uint64_t)c->ngram_size,
             "PLE state dimensions overflow");
    Q38_NEED(c->ngram_parts>0&&c->ngram_parts<=Q38_MAX_PLE_PARTS,"PLE parts=%d",c->ngram_parts);
    Q38_NEED(c->ple_layer>=0&&c->ple_layer<c->layers,"PLE layer=%d",c->ple_layer);
}

static float *q38_load_tensor(Model *m,const char *name,int64_t want) {
    st_tensor *t=st_find(&m->S,name);
    if(!t){fprintf(stderr,"missing %s\n",name);exit(1);}
    if(t->numel!=want){fprintf(stderr,"%s: %lld elements, expected %lld\n",name,(long long)t->numel,(long long)want);exit(1);}
    float *p=falloc(want); st_read_f32(&m->S,name,p,1); return p;
}

static int q38_env_bool(const char *name,int default_value) {
    const char *value=getenv(name);if(!value||!*value)return default_value;
    if(value[0]=='0'&&!value[1])return 0;
    if(value[0]=='1'&&!value[1])return 1;
    fprintf(stderr,"%s must be exactly 0 or 1\n",name);exit(1);
}

static Q38Weight q38_load_weight(Model *m,const char *name,int rows,int cols) {
    st_tensor *tensor=st_find(&m->S,name);Q38Weight weight={0};
    if(!tensor){fprintf(stderr,"missing %s\n",name);exit(1);}
    if(tensor->rank!=2||tensor->shape[0]!=rows||tensor->shape[1]!=cols||
       tensor->numel!=(int64_t)rows*cols){
        fprintf(stderr,"%s: invalid matrix shape, expected [%d,%d]\n",name,rows,cols);exit(1);
    }
    if(tensor->dtype==0&&m->native_bf16){
        q38_weight_reserve(&weight,Q38_WEIGHT_BF16,rows,cols);
        if(tensor->nbytes!=weight.elements*(int64_t)sizeof(uint16_t)){
            fprintf(stderr,"%s: invalid BF16 byte count\n",name);exit(1);
        }
        st_read_raw_cap(&m->S,name,weight.data,tensor->nbytes,1);
    } else {
        if(tensor->dtype<0||tensor->dtype>2){
            fprintf(stderr,"%s: unsupported resident dtype %s\n",name,st_dtype_name(tensor->dtype));exit(1);
        }
        q38_weight_reserve(&weight,Q38_WEIGHT_F32,rows,cols);
        st_read_f32(&m->S,name,(float*)weight.data,1);
    }
    m->resident_weight_bytes+=q38_weight_bytes(&weight);
#ifdef COLI_VULKAN
    weight.vk_res=1;   /* resident for the life of the model: its rows may live on the device */
    weight.vk_name=strdup(name);   /* to read it back if the device ends up holding it alone */
    if(!weight.vk_name){fprintf(stderr,"OOM weight name\n");exit(1);}
#endif
    return weight;
}

/* A layer's tensor name. Index c.layers is the MTP head's decoder layer,
 * which the checkpoint keeps as <mtp>.layers.0. */
static void q38_name(Model *m,char *out,size_t cap,int layer,const char *suffix) {
    if(layer==m->c.layers&&m->mtp_prefix[0]) snprintf(out,cap,"%s.layers.0.%s",m->mtp_prefix,suffix);
    else snprintf(out,cap,"%s.layers.%d.%s",m->prefix,layer,suffix);
}

static void q38_load_gr_at(Model *m,GatedResidual *g,const char *base,int inject) {
    Cfg *c=&m->c; char nm[384];
    snprintf(nm,sizeof nm,"%s.hc_norm.weight",base); g->norm=q38_load_tensor(m,nm,c->hc_width);
    snprintf(nm,sizeof nm,"%s.input_mix_weight_down.weight",base); g->down=q38_load_weight(m,nm,c->hc_rank,c->hc_width);
    snprintf(nm,sizeof nm,"%s.input_mix_weight_up.weight",base); g->up=q38_load_weight(m,nm,c->hc_width,c->hc_rank);
    if(inject){snprintf(nm,sizeof nm,"%s.block_inject_weight.weight",base);g->inject=q38_load_weight(m,nm,c->hc_count,c->hc_width);}
}

static void q38_load_gr(Model *m,GatedResidual *g,int layer,const char *kind,int inject) {
    char base[320];
    if(layer>=0) q38_name(m,base,sizeof base,layer,kind); else snprintf(base,sizeof base,"%s.hyper_connection_mixer",m->prefix);
    q38_load_gr_at(m,g,base,inject);
}

static void q38_load_ple(Model *m,Layer *l) {
    Cfg *c=&m->c; int i=c->ple_layer; char nm[320];
    q38_name(m,nm,sizeof nm,i,"ple.key_proj.weight");l->ple_key=q38_load_weight(m,nm,c->hc_width,c->ple_dim);
    q38_name(m,nm,sizeof nm,i,"ple.value_proj.weight");l->ple_value=q38_load_weight(m,nm,c->hidden,c->ple_dim);
    #define PL(field,suf,n) q38_name(m,nm,sizeof nm,i,"ple." suf); l->field=q38_load_tensor(m,nm,(n))
    PL(ple_norm_key,"norm_key.weight",c->hc_width);PL(ple_norm_query,"norm_query.weight",c->hc_width);
    PL(ple_norm_conv,"norm_conv.weight",c->hc_width);PL(ple_conv,"conv1d.weight",(int64_t)c->hc_width*c->ple_convk);
    #undef PL
    const char *bufs[]={"layer_multipliers","ngram_heads_vocab_sizes","ngram_heads_offsets"};
    int64_t *dsts[]={m->ple_multipliers,m->ple_head_vocab,m->ple_head_offset};
    int counts[]={c->ngram_size,c->ngram_heads,c->ngram_heads};
    for(int b=0;b<3;b++){
        q38_name(m,nm,sizeof nm,i,"ple.ple_embedding."); strncat(nm,bufs[b],sizeof(nm)-strlen(nm)-1);
        st_tensor *t=st_find(&m->S,nm); if(!t||t->dtype!=6||t->numel!=counts[b]){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        st_read_raw_cap(&m->S,nm,dsts[b],(int64_t)counts[b]*8,1);
    }
    for(int h=0;h<c->ngram_heads;h++)
        Q38_NEED(m->ple_head_vocab[h]>0&&m->ple_head_offset[h]>=0&&
                 m->ple_head_vocab[h]<=INT64_MAX-m->ple_head_offset[h],
                 "invalid PLE head range %d",h);
    m->ple_part_count=0; m->ple_part_start[0]=0;
    q38_name(m,nm,sizeof nm,i,"ple.ple_embedding.ngram_embedding.weight");
    if(st_has(&m->S,nm)){
        st_tensor *t=st_find(&m->S,nm);
        if(!t||t->rank!=2||t->shape[0]<=0||t->shape[1]!=c->ngram_head_dim){fprintf(stderr,"invalid PLE table %s\n",nm);exit(1);}
        snprintf(m->ple_part_names[0],sizeof m->ple_part_names[0],"%s",nm);
        m->ple_parts[0]=t; m->ple_part_count=1;
        m->ple_part_start[1]=m->ple_parts[0]->shape[0];
    } else {
        for(int p=0;p<c->ngram_parts;p++){
            q38_name(m,nm,sizeof nm,i,"ple.ple_embedding.ngram_embedding.");
            size_t z=strlen(nm); snprintf(nm+z,sizeof(nm)-z,"shard_%d.weight",p);
            st_tensor *t=st_find(&m->S,nm); if(!t){fprintf(stderr,"missing PLE shard %s\n",nm);exit(1);}
            if(t->rank!=2||t->shape[0]<=0||t->shape[1]!=c->ngram_head_dim||
               m->ple_part_start[p]>INT64_MAX-t->shape[0]){fprintf(stderr,"invalid PLE shard %s\n",nm);exit(1);}
            snprintf(m->ple_part_names[p],sizeof m->ple_part_names[p],"%s",nm);
            m->ple_parts[p]=t; m->ple_part_start[p+1]=m->ple_part_start[p]+t->shape[0];
            m->ple_part_count++;
        }
    }
    q38_name(m,nm,sizeof nm,i,"ple.ple_embedding.ngram_embedding.weight_scale");
    m->ple_weight_scale=1.f;
    if(st_has(&m->S,nm)){
        st_tensor *t=st_find(&m->S,nm); if(t->numel!=1){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        st_read_f32(&m->S,nm,&m->ple_weight_scale,1);
    }
    int64_t need=0; for(int h=0;h<c->ngram_heads;h++) if(m->ple_head_offset[h]+m->ple_head_vocab[h]>need) need=m->ple_head_offset[h]+m->ple_head_vocab[h];
    Q38_NEED(m->ple_part_start[m->ple_part_count]>=need,"PLE table rows %lld < required %lld",(long long)m->ple_part_start[m->ple_part_count],(long long)need);
}

static void q38_alloc_state(Model *m) {
    Cfg *c=&m->c;
    m->DN_rec=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->DN_conv=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->K=(float**)calloc((size_t)c->layers+1,sizeof(float*));    /* +1: the MTP head's row */
    m->V=(float**)calloc((size_t)c->layers+1,sizeof(float*));
    m->IK=(float**)calloc((size_t)c->layers+1,sizeof(float*));
    if(!m->DN_rec||!m->DN_conv||!m->K||!m->V||!m->IK){fprintf(stderr,"OOM model state metadata\n");exit(1);}
    for(int i=m->range_begin;i<m->range_end;i++) if(!c->is_attn[i]) {
        m->DN_rec[i]=(float*)calloc((size_t)c->dn_vheads*c->dn_kdim*c->dn_vdim,sizeof(float));
        m->DN_conv[i]=(float*)calloc((size_t)c->dn_conv_dim*(c->dn_convk-1),sizeof(float));
        if(!m->DN_rec[i]||!m->DN_conv[i]){fprintf(stderr,"OOM DeltaNet state\n");exit(1);}
    }
    if(c->ple_layer>=m->range_begin&&c->ple_layer<m->range_end){
        m->ple_history=(int64_t*)calloc(2,sizeof(int64_t));
        m->PLE_conv_state=(float*)calloc((size_t)c->hc_width*(c->ple_convk-1)*c->ngram_size,sizeof(float));
        if(!m->ple_history||!m->PLE_conv_state){fprintf(stderr,"OOM PLE state\n");exit(1);}
    }
}

/* ---- vision ------------------------------------------------------------- */

/* La torre e' in F32 residente: 27 blocchi da 1152 sono ~0.6 GB, che accanto ai
 * 9.2 GiB dei pesi densi non cambia la classe di macchina. Gli esperti restano
 * su disco; la torre no, perche' si usa una volta per immagine e non per token. */
static const float *q38_vis_tensor(Model *m,const char *suffix,int64_t expect) {
    char nm[512];
    snprintf(nm,sizeof nm,"%s.visual.%s",m->prefix,suffix);
    if(!st_has(&m->S,nm)){
        snprintf(nm,sizeof nm,"model.visual.%s",suffix);
        if(!st_has(&m->S,nm)){fprintf(stderr,"vision tensor missing: %s\n",suffix);exit(1);}
    }
    st_tensor *t=st_find(&m->S,nm);
    if(expect>0&&t->numel!=expect){
        fprintf(stderr,"vision tensor %s has %lld values, expected %lld -- refusing\n",
                nm,(long long)t->numel,(long long)expect);exit(1);
    }
    float *buf=(float*)malloc((size_t)t->numel*sizeof(float));
    if(!buf){fprintf(stderr,"OOM loading %s\n",nm);exit(1);}
    st_read_f32(&m->S,nm,buf,t->numel);
    m->resident_weight_bytes+=(uint64_t)t->numel*sizeof(float);
    return buf;
}

static void q38_vis_linear(Model *m,Q38Linear *l,const char *stem,int out,int in) {
    char nm[512];
    snprintf(nm,sizeof nm,"%s.weight",stem); l->w=q38_vis_tensor(m,nm,(int64_t)out*in);
    snprintf(nm,sizeof nm,"%s.bias",stem);   l->b=q38_vis_tensor(m,nm,out);
    l->out=out; l->in=in;
}

static void q38_vis_norm(Model *m,Q38Norm *n,const char *stem,int width) {
    char nm[512];
    snprintf(nm,sizeof nm,"%s.weight",stem); n->w=q38_vis_tensor(m,nm,width);
    snprintf(nm,sizeof nm,"%s.bias",stem);   n->b=q38_vis_tensor(m,nm,width);
}

static void q38_load_vision(Model *m) {
    Cfg *c=&m->c;
    if(!c->vis_depth) return;
    char probe[512];
    snprintf(probe,sizeof probe,"%s.visual.pos_embed.weight",m->prefix);
    if(!st_has(&m->S,probe)&&!st_has(&m->S,"model.visual.pos_embed.weight")){
        /* config multimodale ma pesi assenti: e' un export solo-testo di un
         * checkpoint multimodale. Spegnere la torre e dirlo e' meglio che
         * rifiutare un modello che per il testo funziona benissimo. */
        fprintf(stderr,"[qwen38] vision_config present but no visual weights; text only\n");
        c->vis_depth=0; return;
    }
    Q38Vision *v=&m->vis;
    memset(v,0,sizeof *v);
    v->depth=c->vis_depth; v->hidden=c->vis_hidden; v->heads=c->vis_heads;
    v->head_dim=c->vis_hidden/c->vis_heads; v->inter=c->vis_inter;
    v->patch=c->vis_patch; v->merge=c->vis_merge; v->temporal=c->vis_temporal;
    v->in_ch=c->vis_in_ch; v->out_hidden=c->vis_out_hidden;
    v->num_pos=c->vis_num_pos; v->side=(int)(sqrt((double)c->vis_num_pos)+0.5);
    v->eps=1e-6f;
    if(v->side*v->side!=v->num_pos){
        fprintf(stderr,"[qwen38] num_position_embeddings=%d is not a square grid -- refusing\n",
                v->num_pos);exit(1);
    }
    int features=v->in_ch*v->temporal*v->patch*v->patch;
    q38_vis_linear(m,&v->patch_embed,"patch_embed.proj",v->hidden,features);
    v->pos_embed=q38_vis_tensor(m,"pos_embed.weight",(int64_t)v->num_pos*v->hidden);
    v->blocks=(Q38VBlock*)calloc((size_t)v->depth,sizeof(Q38VBlock));
    if(!v->blocks){fprintf(stderr,"OOM vision blocks\n");exit(1);}
    for(int i=0;i<v->depth;i++){
        char stem[160];
        snprintf(stem,sizeof stem,"blocks.%d.norm1",i);          q38_vis_norm(m,&v->blocks[i].norm1,stem,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.norm2",i);          q38_vis_norm(m,&v->blocks[i].norm2,stem,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.attn.qkv",i);       q38_vis_linear(m,&v->blocks[i].qkv,stem,3*v->hidden,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.attn.proj",i);      q38_vis_linear(m,&v->blocks[i].proj,stem,v->hidden,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.mlp.linear_fc1",i); q38_vis_linear(m,&v->blocks[i].fc1,stem,v->inter,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.mlp.linear_fc2",i); q38_vis_linear(m,&v->blocks[i].fc2,stem,v->hidden,v->inter);
    }
    int wide=v->hidden*v->merge*v->merge;
    q38_vis_norm(m,&v->merger_norm,"merger.norm",v->hidden);
    q38_vis_linear(m,&v->merger_fc1,"merger.linear_fc1",wide,wide);
    q38_vis_linear(m,&v->merger_fc2,"merger.linear_fc2",v->out_hidden,wide);
    m->vis_ready=1;
    fprintf(stderr,"[qwen38] vision tower: %d blocks, hidden %d, %d heads, patch %d, merge %d\n",
            v->depth,v->hidden,v->heads,v->patch,v->merge);
}

/* Esegue la torre su un'immagine gia' preprocessata e prepara la mappa
 * posizione->riga. `ids`/`n` sono i token del prompt: le righe vengono
 * assegnate ai token immagine nell'ordine in cui compaiono. */
static int q38_vision_attach(Model *m,const float *patches,int grid_h,int grid_w,
                             const int *ids,int n) {
    Cfg *c=&m->c;
    if(!m->vis_ready) return -1;
    int tokens=(grid_h*grid_w)/(c->vis_merge*c->vis_merge);
    int slots=0;
    for(int i=0;i<n;i++) if(ids[i]==c->image_token) slots++;
    if(slots!=tokens){
        /* Il numero di segnaposto nel prompt DEVE essere quello che la griglia
         * produce. Se non lo e', il template e il preprocessing hanno visto due
         * immagini diverse, e proseguire vorrebbe dire mettere i vettori giusti
         * nelle posizioni sbagliate. */
        fprintf(stderr,"[qwen38] prompt has %d image placeholders but the grid gives %d tokens\n",
                slots,tokens);
        return -1;
    }
    free(m->vis_rows); free(m->vis_map);
    m->vis_rows=(float*)calloc((size_t)tokens*c->hidden,sizeof(float));
    m->vis_map=(int*)malloc((size_t)n*sizeof(int));
    if(!m->vis_rows||!m->vis_map){free(m->vis_rows);free(m->vis_map);
        m->vis_rows=NULL;m->vis_map=NULL;return -1;}
    if(q38_vision_forward(&m->vis,patches,grid_h,grid_w,m->vis_rows)!=tokens){
        free(m->vis_rows);free(m->vis_map);m->vis_rows=NULL;m->vis_map=NULL;return -1;}
    int next=0;
    for(int i=0;i<n;i++) m->vis_map[i]=(ids[i]==c->image_token)?next++:-1;
    m->vis_map_len=n; m->vis_rows_n=tokens;
    return tokens;
}

static void q38_vision_detach(Model *m) {
    free(m->vis_rows); free(m->vis_map);
    m->vis_rows=NULL; m->vis_map=NULL; m->vis_map_len=0; m->vis_rows_n=0;
}

/* One decoder layer's resident tensors and its expert cache. Index c.layers
 * is the MTP head's layer (q38_name gives it its checkpoint names): an
 * attention layer like the model's QSA layers, with the same tensors. */
static void q38_load_layer(Model *m,int i,int cap) {
    Cfg *c=&m->c; char nm[320];
    int attn=i<c->layers?c->is_attn[i]:1;
    Layer *l=&m->L[i]; q38_load_gr(m,&l->attn_gr,i,"attn_hyper_connection",1); q38_load_gr(m,&l->mlp_gr,i,"mlp_hyper_connection",1);
    #define WLD(field,suf,o,in) q38_name(m,nm,sizeof nm,i,suf); l->field=q38_load_weight(m,nm,(o),(in))
    #define VLD(field,suf,n) q38_name(m,nm,sizeof nm,i,suf); l->field=q38_load_tensor(m,nm,(n))
    WLD(router,"mlp.gate.weight",c->experts,c->hidden);
    WLD(sh_g,"mlp.shared_expert.gate_proj.weight",c->shared_inter,c->hidden);
    WLD(sh_u,"mlp.shared_expert.up_proj.weight",c->shared_inter,c->hidden);
    WLD(sh_d,"mlp.shared_expert.down_proj.weight",c->hidden,c->shared_inter);
    VLD(sh_gate,"mlp.shared_expert_gate.weight",c->hidden);
    if(attn){
        WLD(q,"self_attn.q_proj.weight",c->q_heads*c->head_dim*2,c->hidden);
        WLD(k,"self_attn.k_proj.weight",c->kv_heads*c->head_dim,c->hidden);
        WLD(v,"self_attn.v_proj.weight",c->kv_heads*c->head_dim,c->hidden);
        WLD(o,"self_attn.o_proj.weight",c->hidden,c->q_heads*c->head_dim);
        VLD(qn,"self_attn.q_norm.weight",c->head_dim);VLD(kn,"self_attn.k_norm.weight",c->head_dim);
        WLD(idx_qk,"self_attn.indexer.index_qk_proj.weight",(c->idx_qheads+c->idx_kheads)*c->idx_dim,c->hidden);
        VLD(idx_qn,"self_attn.indexer.q_layernorm.weight",c->idx_dim);VLD(idx_kn,"self_attn.indexer.k_layernorm.weight",c->idx_dim);
    } else {
        int vd=c->dn_vheads*c->dn_vdim;
        WLD(dn_qkv,"linear_attn.in_proj_qkv.weight",c->dn_conv_dim,c->hidden);
        WLD(dn_z,"linear_attn.in_proj_z.weight",vd,c->hidden);
        WLD(dn_b,"linear_attn.in_proj_b.weight",c->dn_vheads,c->hidden);
        WLD(dn_a,"linear_attn.in_proj_a.weight",c->dn_vheads,c->hidden);
        VLD(dn_conv,"linear_attn.conv1d.weight",(int64_t)c->dn_conv_dim*c->dn_convk);
        VLD(dn_dtbias,"linear_attn.dt_bias",c->dn_vheads);VLD(dn_alog,"linear_attn.A_log",c->dn_vheads);
        VLD(dn_norm,"linear_attn.norm.weight",c->dn_vdim);
        WLD(dn_out,"linear_attn.out_proj.weight",c->hidden,vd);
    }
    #undef WLD
    #undef VLD
    LCache *lc=&m->cache[i]; lc->cap=cap; lc->slots=(Slot*)calloc((size_t)cap,sizeof(Slot)); lc->by_expert=(int*)malloc((size_t)c->experts*sizeof(int));
    if(!lc->slots||!lc->by_expert){fprintf(stderr,"OOM expert cache\n");exit(1);} for(int e=0;e<c->experts;e++)lc->by_expert[e]=-1;
}

static void model_init_range(Model *m,const char *snap,int cap,int bits,
                             int layer_begin,int layer_end,int load_boundaries,
                             int allocate_state) {
    (void)bits; memset(m,0,sizeof(*m)); double t0=now_s();
    m->native_fp8=q38_env_bool("Q38_NATIVE_FP8",1);
    m->native_bf16=q38_env_bool("Q38_NATIVE_BF16",1);
    m->expert_prefetch=q38_env_bool("Q38_EXPERT_PREFETCH",1);
    m->expert_parallel_reads=q38_env_bool("Q38_EXPERT_PARALLEL_READS",1);
    m->prefill_batch=q38_env_bool("Q38_PREFILL_BATCH",1);
    q38_load_cfg(&m->c,snap); q38_validate_cfg(&m->c); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[320];
    if(st_has(&m->S,"model.language_model.embed_tokens.weight")) snprintf(m->prefix,sizeof m->prefix,"model.language_model");
    else if(st_has(&m->S,"model.embed_tokens.weight")) snprintf(m->prefix,sizeof m->prefix,"model");
    else {fprintf(stderr,"checkpoint has no Qwen4-Exp text embedding\n");exit(1);}
    if(layer_end==0) layer_end=c->layers;
    if(layer_begin<0||layer_begin>=layer_end||layer_end>c->layers){fprintf(stderr,"invalid Qwen3.8 layer range [%d,%d)\n",layer_begin,layer_end);exit(1);}
    m->range_begin=layer_begin; m->range_end=layer_end;
    if(load_boundaries){
        snprintf(nm,sizeof nm,"%s.embed_tokens.weight",m->prefix); m->embed=q38_load_weight(m,nm,c->vocab,c->hidden);
        m->lm_head=q38_load_weight(m,"lm_head.weight",c->vocab,c->hidden);
        q38_load_gr(m,&m->final_gr,-1,NULL,0);
    }
    /* +1 row each: the MTP head's layer, filled only by q38_mtp_attach */
    m->L=(Layer*)calloc((size_t)c->layers+1,sizeof(Layer));
    m->cache=(LCache*)calloc((size_t)c->layers+1,sizeof(LCache));
    m->expert_scales=(Q38ExpertScaleCache*)calloc((size_t)c->layers+1,
                                                  sizeof(*m->expert_scales));
    if(!m->L||!m->cache||!m->expert_scales){fprintf(stderr,"OOM model metadata\n");exit(1);}
    for(int i=layer_begin;i<layer_end;i++) q38_load_layer(m,i,cap);
    if(c->ple_layer>=layer_begin&&c->ple_layer<layer_end) q38_load_ple(m,&m->L[c->ple_layer]);
    /* La torre solo quando il motore possiede la sequenza intera: uno shard che
     * ospita solo alcuni layer non ha da fare niente con le immagini, e
     * caricarla la' sarebbe mezzo giga per nulla. */
    if(load_boundaries&&q38_env_bool("Q38_VISION",1)) q38_load_vision(m);
    if(allocate_state) q38_alloc_state(m);
    m->dense_load_s=now_s()-t0;
    fprintf(stderr,"[qwen38] native text weights: prefix=%s, %d layers, PLE=%d, cache=%d/layer, "
                   "FP8=%s, BF16=%s (resident matrices %.2f GiB)\n",m->prefix,c->layers,c->ple_layer,cap,
                   m->native_fp8?"native":"expanded-f32",m->native_bf16?"native":"expanded-f32",
                   m->resident_weight_bytes/1073741824.0);
}

static void model_init(Model *m,const char *snap,int cap,int bits) {
    model_init_range(m,snap,cap,bits,0,0,1,1);
}

static void q38_gr_read(Model *m,const GatedResidual *g,const float *hyper,
                        int S,float *mixed,float *inject) {
    const Cfg *c=&m->c;
    int H=c->hidden,W=c->hc_width,C=c->hc_count,R=c->hc_rank;
    if(C<=0||H<=0||W!=C*H){fprintf(stderr,"invalid gated-residual width\n");exit(1);}
    float *norm=falloc((int64_t)S*W),*low=falloc((int64_t)S*R),*mix=falloc((int64_t)S*W);
    /* Makes the initialized-input invariant visible to aggressive interprocedural
     * warning analysis; every element is overwritten by q38_rms0 below. */
    memset(norm,0,(size_t)S*W*sizeof(float));
    for(int s=0;s<S;s++) for(int b=0;b<C;b++)
        q38_rms0(norm+(int64_t)s*W+(int64_t)b*H,hyper+(int64_t)s*W+(int64_t)b*H,g->norm+(int64_t)b*H,H,c->eps);
    q38_dense_matmul(m,low,norm,&g->down,S,W,R);
    for(int64_t z=0;z<(int64_t)S*R;z++) low[z]=q38_silu(low[z]/C);
    q38_dense_matmul(m,mix,low,&g->up,S,R,W);
    for(int s=0;s<S;s++) for(int d=0;d<H;d++){
        float v=0.f;
        for(int b=0;b<C;b++) v+=q38_sigmoid(mix[(int64_t)s*W+(int64_t)b*H+d])*norm[(int64_t)s*W+(int64_t)b*H+d];
        mixed[(int64_t)s*H+d]=v/C;
    }
    if(inject){
        q38_dense_matmul(m,inject,norm,&g->inject,S,W,C);
        for(int64_t z=0;z<(int64_t)S*C;z++) inject[z]=2.f*q38_sigmoid(inject[z]/C);
    }
    free(norm);free(low);free(mix);
}

static void q38_gr_apply(const Cfg *c,float *hyper,const float *block,const float *inject,int S) {
    int H=c->hidden,C=c->hc_count,W=c->hc_width;
    for(int s=0;s<S;s++)for(int b=0;b<C;b++){
        float a=inject[(int64_t)s*C+b];
        for(int d=0;d<H;d++)hyper[(int64_t)s*W+(int64_t)b*H+d]+=a*block[(int64_t)s*H+d];
    }
}

typedef struct {
    st_tensor *tensor;
    int expert, projection;
} Q38ScaleDesc;

static int q38_scale_desc_cmp(const void *left,const void *right) {
    const Q38ScaleDesc *a=(const Q38ScaleDesc*)left;
    const Q38ScaleDesc *b=(const Q38ScaleDesc*)right;
    if(a->tensor->off<b->tensor->off)return -1;
    if(a->tensor->off>b->tensor->off)return 1;
    return 0;
}

static int q38_scale_group_span(Q38ScaleDesc *desc,int count,int *fd,
                                int64_t *begin,int64_t *nbytes) {
    if(!desc||count<1||!fd||!begin||!nbytes)return -1;
    qsort(desc,(size_t)count,sizeof(*desc),q38_scale_desc_cmp);
    int group_fd=desc[0].tensor->fd;
    int64_t first=desc[0].tensor->off,cursor=first;
    for(int i=0;i<count;i++){
        st_tensor *tensor=desc[i].tensor;
        if(tensor->fd!=group_fd||tensor->off!=cursor||tensor->nbytes<=0||
           tensor->nbytes>INT64_MAX-cursor)return -1;
        cursor+=tensor->nbytes;
    }
    *fd=group_fd;*begin=first;*nbytes=cursor-first;return 0;
}

static void q38_decode_scale_tensor(float *out,const unsigned char *raw,
                                     const st_tensor *tensor) {
    if(tensor->dtype==2)memcpy(out,raw,(size_t)tensor->nbytes);
    else for(int64_t i=0;i<tensor->numel;i++){
        uint16_t half;memcpy(&half,raw+(size_t)i*sizeof(half),sizeof(half));
        out[i]=tensor->dtype==0?bf16_to_f32(half):f16_to_f32(half);
    }
}

/* The official checkpoint stores every layer's gate/up scale sidecars as one
 * compact range and every down sidecar as another.  Normalize both ranges to
 * an expert-indexed F32 bank once, scattering by the numeric expert id rather
 * than the checkpoint's lexical tensor order (0, 1, 10, ...).  Variants that
 * do not satisfy the compact-range invariant simply keep the established
 * per-matrix loader. */
static int q38_prepare_expert_scale_bank(Model *m,int layer) {
    Cfg *c=&m->c;
    if(!m->native_fp8||layer<0||layer>=q38_layer_rows(m))return 0;
    if(!m->expert_scales){
        m->expert_scales=(Q38ExpertScaleCache*)calloc((size_t)c->layers+1,
                                                       sizeof(*m->expert_scales));
        if(!m->expert_scales){fprintf(stderr,"OOM expert scale metadata\n");exit(1);}
    }
    Q38ExpertScaleCache *cache=&m->expert_scales[layer];
    if(cache->ready)return cache->ready>0;
    int64_t scale_count=fp8_nblk(c->inter)*fp8_nblk(c->hidden);
    if(scale_count<1||(uint64_t)c->experts>SIZE_MAX/(3u*(uint64_t)scale_count*sizeof(float))){
        fprintf(stderr,"invalid expert scale-bank geometry\n");exit(1);
    }
    Q38ScaleDesc *gate_up=(Q38ScaleDesc*)malloc((size_t)c->experts*2*sizeof(*gate_up));
    Q38ScaleDesc *down=(Q38ScaleDesc*)malloc((size_t)c->experts*sizeof(*down));
    if(!gate_up||!down){fprintf(stderr,"OOM expert scale descriptors\n");exit(1);}
    int ngu=0,ndown=0;char suffix[192],name[320];
    const char *projection[3]={"gate_proj","up_proj","down_proj"};
    for(int expert=0;expert<c->experts;expert++)for(int kind=0;kind<3;kind++){
        int rows=kind==2?c->hidden:c->inter;
        int cols=kind==2?c->inter:c->hidden;
        int length=snprintf(suffix,sizeof suffix,
            "mlp.experts.%d.%s.weight_scale_inv",expert,projection[kind]);
        if(length<0||(size_t)length>=sizeof suffix)goto incompatible;
        q38_name(m,name,sizeof name,layer,suffix);
        st_tensor *tensor=st_find(&m->S,name);
        int64_t block_rows=fp8_nblk(rows),block_cols=fp8_nblk(cols);
        if(!tensor||tensor->dtype<0||tensor->dtype>2||tensor->rank!=2||
           tensor->shape[0]!=block_rows||tensor->shape[1]!=block_cols||
           tensor->numel!=scale_count||
           tensor->numel>INT64_MAX/st_dtype_esz(tensor->dtype)||
           tensor->nbytes!=tensor->numel*st_dtype_esz(tensor->dtype))
            goto incompatible;
        Q38ScaleDesc item={tensor,expert,kind};
        if(kind<2)gate_up[ngu++]=item;else down[ndown++]=item;
    }
    int gu_fd,down_fd;int64_t gu_begin,down_begin,gu_bytes,down_bytes;
    if(q38_scale_group_span(gate_up,ngu,&gu_fd,&gu_begin,&gu_bytes)||
       q38_scale_group_span(down,ndown,&down_fd,&down_begin,&down_bytes)||
       (uint64_t)gu_bytes>SIZE_MAX||(uint64_t)down_bytes>SIZE_MAX)
        goto incompatible;
    float *values=(float*)malloc((size_t)c->experts*3*(size_t)scale_count*sizeof(float));
    unsigned char *gu_raw=(unsigned char*)malloc((size_t)gu_bytes);
    unsigned char *down_raw=(unsigned char*)malloc((size_t)down_bytes);
    if(!values||!gu_raw||!down_raw){fprintf(stderr,"OOM resident expert scales\n");exit(1);}
    double started=now_s();
    st_read_range_raw_cap(&m->S,gu_fd,gu_begin,gu_bytes,gu_raw,gu_bytes,1,
                          "pread Qwen3.8 gate/up scales");
    st_read_range_raw_cap(&m->S,down_fd,down_begin,down_bytes,down_raw,down_bytes,1,
                          "pread Qwen3.8 down scales");
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);m->expert_scale_reads+=2;
    for(int i=0;i<ngu;i++){
        Q38ScaleDesc *item=&gate_up[i];
        float *dst=values+((int64_t)item->expert*3+item->projection)*scale_count;
        q38_decode_scale_tensor(dst,gu_raw+(item->tensor->off-gu_begin),item->tensor);
    }
    for(int i=0;i<ndown;i++){
        Q38ScaleDesc *item=&down[i];
        float *dst=values+((int64_t)item->expert*3+item->projection)*scale_count;
        q38_decode_scale_tensor(dst,down_raw+(item->tensor->off-down_begin),item->tensor);
    }
    free(gu_raw);free(down_raw);free(gate_up);free(down);
    cache->values=values;cache->scale_count=scale_count;cache->ready=1;
    m->expert_scale_bytes+=(uint64_t)c->experts*3*(uint64_t)scale_count*sizeof(float);
    return 1;
incompatible:
    free(gate_up);free(down);cache->ready=-1;return 0;
}

static void q38_bind_borrowed_fp8(Q38Weight *weight,void *data,float *scales,
                                  int rows,int cols) {
    q38_weight_free(weight);
    weight->data=data;weight->scales=scales;weight->rows=rows;weight->cols=cols;
    weight->elements=(int64_t)rows*cols;
    weight->scale_count=fp8_nblk(rows)*fp8_nblk(cols);
    weight->kind=Q38_WEIGHT_FP8;
}

static void q38_bind_fp8_slot(Slot *slot,float *scales,int scale_count,
                              int hidden,int intermediate) {
    int64_t matrix_bytes=(int64_t)hidden*intermediate;
    if(matrix_bytes<1||matrix_bytes>INT64_MAX/3||
       (uint64_t)(matrix_bytes*3)>SIZE_MAX||
       scale_count!=fp8_nblk(hidden)*fp8_nblk(intermediate)){
        fprintf(stderr,"invalid native FP8 expert slab geometry\n");exit(1);
    }
    q38_weight_free(&slot->gate);q38_weight_free(&slot->up);
    q38_weight_free(&slot->down);
    int64_t slab_bytes=matrix_bytes*3;
    if(!slot->fp8_slab||slot->fp8_slab_bytes!=slab_bytes){
        void *replacement=realloc(slot->fp8_slab,(size_t)slab_bytes);
        if(!replacement){fprintf(stderr,"OOM native FP8 expert slab\n");exit(1);}
        slot->fp8_slab=replacement;slot->fp8_slab_bytes=slab_bytes;
    }
    unsigned char *raw=(unsigned char*)slot->fp8_slab;
    q38_bind_borrowed_fp8(&slot->gate,raw,scales,intermediate,hidden);
    q38_bind_borrowed_fp8(&slot->up,raw+matrix_bytes,scales+scale_count,
                          intermediate,hidden);
    q38_bind_borrowed_fp8(&slot->down,raw+2*matrix_bytes,scales+2*scale_count,
                          hidden,intermediate);
}

static int q38_native_fp8_expert_tensors(Model *m,int layer,int expert,
                                         st_tensor *weight[3]) {
    if(!m->native_fp8)return 0;
    Cfg *c=&m->c;const char *projection[3]={"gate_proj","up_proj","down_proj"};
    int rows[3]={c->inter,c->inter,c->hidden};
    int cols[3]={c->hidden,c->hidden,c->inter};
    char suffix[192],name[320];
    for(int kind=0;kind<3;kind++){
        int length=snprintf(suffix,sizeof suffix,"mlp.experts.%d.%s.weight",
                            expert,projection[kind]);
        if(length<0||(size_t)length>=sizeof suffix)return 0;
        q38_name(m,name,sizeof name,layer,suffix);weight[kind]=st_find(&m->S,name);
        if(!weight[kind]||weight[kind]->dtype!=4||weight[kind]->rank!=2||
           weight[kind]->shape[0]!=rows[kind]||weight[kind]->shape[1]!=cols[kind]||
           weight[kind]->nbytes!=(int64_t)rows[kind]*cols[kind])return 0;
    }
    if(weight[0]->fd!=weight[1]->fd||
       weight[0]->off>INT64_MAX-weight[0]->nbytes||
       weight[1]->off!=weight[0]->off+weight[0]->nbytes)return 0;
    return 1;
}

static void q38_load_native_fp8_ranges(Model *m,int layer,int expert,Slot *slot,
                                       st_tensor *weight[3]) {
    Cfg *c=&m->c;
    Q38ExpertScaleCache *cache=&m->expert_scales[layer];
    float *scales=cache->values+(int64_t)expert*3*cache->scale_count;
    /* Se i tre intervalli sono mappati (COLI_MAP_EXPERTS=1), lo slot li PUNTA
     * invece di copiarli: niente slab, niente 14 MB per miss. */
    {
        const uint8_t *pg=(const uint8_t*)st_map_shard_range(weight[0]->fd,weight[0]->off,weight[0]->nbytes);
        const uint8_t *pu=(const uint8_t*)st_map_shard_range(weight[1]->fd,weight[1]->off,weight[1]->nbytes);
        const uint8_t *pd=(const uint8_t*)st_map_shard_range(weight[2]->fd,weight[2]->off,weight[2]->nbytes);
        if(pg&&pu&&pd){
            int sc=(int)cache->scale_count;
            q38_bind_borrowed_fp8(&slot->gate,(void*)pg,scales,c->inter,c->hidden);
            q38_bind_borrowed_fp8(&slot->up,(void*)pu,scales+sc,c->inter,c->hidden);
            q38_bind_borrowed_fp8(&slot->down,(void*)pd,scales+2*sc,c->hidden,c->inter);
            if(slot->fp8_slab){free(slot->fp8_slab);slot->fp8_slab=NULL;slot->fp8_slab_bytes=0;}
            return;
        }
    }
    q38_bind_fp8_slot(slot,scales,(int)cache->scale_count,c->hidden,c->inter);
    int64_t pair_bytes=weight[0]->nbytes+weight[1]->nbytes;
    unsigned char *raw=(unsigned char*)slot->fp8_slab;
    st_read_range_raw_cap(&m->S,weight[0]->fd,weight[0]->off,pair_bytes,
                          raw,pair_bytes,1,"pread Qwen3.8 gate/up expert");
    st_read_range_raw_cap(&m->S,weight[2]->fd,weight[2]->off,weight[2]->nbytes,
                          raw+pair_bytes,slot->fp8_slab_bytes-pair_bytes,1,
                          "pread Qwen3.8 down expert");
}

static int q38_try_load_native_fp8_expert(Model *m,int layer,int expert,Slot *slot) {
    st_tensor *weight[3];
    if(!q38_native_fp8_expert_tensors(m,layer,expert,weight)||
       !q38_prepare_expert_scale_bank(m,layer))return 0;
    double started=now_s();
    q38_load_native_fp8_ranges(m,layer,expert,slot,weight);
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);
    m->expert_weight_reads+=2;m->expert_pair_reads++;
    return 1;
}

/* Whether a layer's routed experts come from the int4-g64 sidecar. It holds
 * the model's layers only: the MTP head's layer (index c.layers) keeps the
 * snapshot's FP8 experts with a sidecar or without one. */
static inline int q38_layer_int4(const Model *m,int layer) {
    return m->x4&&layer>=0&&layer<m->c.layers;
}

/* Advise the kernel of the experts a route is about to miss: the int4
 * sidecar's one record each, or the native FP8 pair and down ranges. */
static void q38_prefetch_experts(Model *m,int layer,const int *experts,int count) {
    if(!m->expert_prefetch||!experts||count<1)return;
    LCache *cache=&m->cache[layer];
    if(q38_layer_int4(m,layer)){
        for(int index=0;index<count;index++){
            int expert=experts[index];
            if(expert<0||expert>=m->c.experts||cache->by_expert[expert]>=0)continue;
            posix_fadvise(m->x4->fd[layer],m->x4->off[(int64_t)layer*m->c.experts+expert],
                          m->x4->record_bytes,POSIX_FADV_WILLNEED);
            m->expert_prefetch_ranges++;
        }
        return;
    }
    if(!m->native_fp8||!q38_prepare_expert_scale_bank(m,layer))return;
    for(int index=0;index<count;index++){
        int expert=experts[index];st_tensor *weight[3];
        if(expert<0||expert>=m->c.experts||cache->by_expert[expert]>=0||
           !q38_native_fp8_expert_tensors(m,layer,expert,weight))continue;
        posix_fadvise(weight[0]->fd,weight[0]->off,
                      weight[0]->nbytes+weight[1]->nbytes,POSIX_FADV_WILLNEED);
        posix_fadvise(weight[2]->fd,weight[2]->off,weight[2]->nbytes,
                      POSIX_FADV_WILLNEED);
        m->expert_prefetch_ranges+=2;
    }
}

static void q38_load_fp8_expert_weight(Model *m,const char *wn,const char *sn,
                                        Q38Weight *out,int O,int I) {
    st_tensor *w=st_find(&m->S,wn),*sc=st_find(&m->S,sn);
    int nb_o=(O+127)/128,nb_i=(I+127)/128;
    if(!w||w->dtype!=4||w->rank!=2||w->shape[0]!=O||w->shape[1]!=I||
       w->nbytes!=(int64_t)O*I||!sc||sc->dtype<0||sc->dtype>2||
       sc->rank!=2||sc->shape[0]!=nb_o||sc->shape[1]!=nb_i||
       sc->numel!=(int64_t)nb_o*nb_i){
        fprintf(stderr,"invalid block-FP8 expert matrix %s / %s\n",wn,sn);exit(1);
    }
    if(m->native_fp8){
        q38_weight_reserve(out,Q38_WEIGHT_FP8,O,I);
        double started=now_s();
        st_read_raw_cap(&m->S,wn,out->data,w->nbytes,1);
        st_read_f32(&m->S,sn,out->scales,1);
        q38_tm_add(m,Q38_TM_EXPERT_READ,started);
        m->expert_weight_reads++;m->expert_scale_reads++;
        return;
    }
    q38_weight_reserve(out,Q38_WEIGHT_F32,O,I);
    uint8_t *raw=(uint8_t*)malloc((size_t)w->nbytes);float *scale=falloc(sc->numel);
    if(!raw){fprintf(stderr,"OOM FP8 expert staging\n");exit(1);}
    double started=now_s();
    st_read_raw_cap(&m->S,wn,raw,w->nbytes,1);st_read_f32(&m->S,sn,scale,1);
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);
    m->expert_weight_reads++;m->expert_scale_reads++;started=now_s();
    float *decoded=(float*)out->data;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++)for(int i=0;i<I;i++)
        decoded[(int64_t)o*I+i]=e4m3_decode(raw[(int64_t)o*I+i])*
                                  scale[(o/128)*nb_i+i/128];
    q38_tm_add(m,Q38_TM_FP8_EXPAND,started);
    free(raw);free(scale);
}

static void q38_load_expert_weight(Model *m,const char *name,Q38Weight *out,
                                    int O,int I) {
    st_tensor *tensor=st_find(&m->S,name);
    if(!tensor||tensor->rank!=2||tensor->shape[0]!=O||tensor->shape[1]!=I||
       tensor->numel!=(int64_t)O*I||tensor->dtype<0||tensor->dtype>2){
        fprintf(stderr,"invalid expert matrix %s\n",name);exit(1);
    }
    Q38WeightKind kind=m->native_bf16&&tensor->dtype==0?Q38_WEIGHT_BF16:Q38_WEIGHT_F32;
    q38_weight_reserve(out,kind,O,I);double started=now_s();
    if(kind==Q38_WEIGHT_BF16)
        st_read_raw_cap(&m->S,name,out->data,tensor->nbytes,1);
    else st_read_f32(&m->S,name,(float*)out->data,1);
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);m->expert_weight_reads++;
}

static void q38_load_expert_slice(Model *m,const char *name,const st_tensor *tensor,
                                   int64_t element_offset,Q38Weight *out,int O,int I) {
    if(!tensor||tensor->dtype<0||tensor->dtype>2||element_offset<0||
       element_offset>tensor->numel-(int64_t)O*I){
        fprintf(stderr,"invalid expert slice %s\n",name);exit(1);
    }
    Q38WeightKind kind=m->native_bf16&&tensor->dtype==0?Q38_WEIGHT_BF16:Q38_WEIGHT_F32;
    q38_weight_reserve(out,kind,O,I);double started=now_s();
    if(kind==Q38_WEIGHT_BF16){
        int64_t bytes=(int64_t)O*I*(int64_t)sizeof(uint16_t);
        st_read_slice_raw_cap(&m->S,name,element_offset*(int64_t)sizeof(uint16_t),
                              bytes,out->data,bytes,1);
    } else st_read_slice_f32(&m->S,name,element_offset,(int64_t)O*I,(float*)out->data,1);
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);m->expert_weight_reads++;
}

/* ---- routed experts from the int4-g64 sidecar -------------------------------
 * tools/convert_qwen38_experts_int4.py writes <snap>/experts-int4g64/: one
 * safetensors file per layer and an index.json that says what is in them.
 * The release's FP8 shards are only read by it and stay usable as they are.
 * Every expert is one contiguous record, so a miss is one read:
 *
 *   gate codes [I][H/2] | up codes [I][H/2] | down codes [H][I/2] |
 *   gate scales [I][H/64] | up scales [I][H/64] | down scales [H][I/64]
 *
 * named like the release's tensors (...experts.E.gate_proj.weight, and the
 * same name + ".qs" for its f32 scales). The codes are expert_ffn.h's planar
 * int4 (a row is blocks of 64, byte k holding element k in its low nibble
 * and element k+32 in its high one, both as v+8; a width that is not a
 * multiple of 64 keeps its tail in pairs, as idot.h's planarize_i4 leaves
 * it), one scale per 64 inputs, quantized as tools/convert_qwen36.py --ebits
 * 4 --gs 64 does. On the release an expert is 2.76 MB, 56% of its 4.92 MB
 * in FP8: the same RAM holds 1.78x the experts and a miss reads 56% of the
 * bytes.
 * Attached by the CLI/serve engine only: the Segment and Edge adapters keep
 * the snapshot's experts, the representation their numeric class names. */
#define Q38_INT4_DIR "experts-int4g64"
#define Q38_INT4_FORMAT "colibri.qwen38.experts-int4g64"
#define Q38_INT4_VERSION 1

/* Q38_EXPERT_INT4: unset = the sidecar when there is one, 0 = never (the
 * snapshot's FP8), 1 = refuse to start without it (a benchmark that must not
 * quietly measure FP8). */
static int q38_expert_int4_wanted(void) {
    const char *value=getenv("Q38_EXPERT_INT4");
    if(!value||!*value)return -1;
    if(value[0]=='0'&&!value[1])return 0;
    if(value[0]=='1'&&!value[1])return 1;
    fprintf(stderr,"Q38_EXPERT_INT4 must be exactly 0 or 1\n");exit(1);
}

static void q38_int4_refuse(const char *dir,const char *why) {
    fprintf(stderr,"[qwen38] %s: %s -- refusing (Q38_EXPERT_INT4=0 ignores the sidecar)\n",dir,why);
    exit(1);
}

static int64_t q38_json_int(jval *object,const char *key) {
    jval *value=json_get(object,key);
    if(!value||value->t!=J_NUM||!isfinite(value->num)||floor(value->num)!=value->num||
       value->num<0||value->num>(double)INT64_MAX/2)return -1;
    return (int64_t)value->num;
}

/* The six tensors of one expert must be the record the index describes:
 * right dtype and shape, one file, back to back in record order, at an
 * offset that keeps the f32 scales aligned in a mapping. */
static int q38_int4_record_at(Model *m,Q38Int4Experts *x,int layer,int expert,
                              int *fd,int64_t *start) {
    Cfg *c=&m->c;const char *projection[3]={"gate_proj","up_proj","down_proj"};
    int rows[3]={c->inter,c->inter,c->hidden},cols[3]={c->hidden,c->hidden,c->inter};
    char suffix[192],name[320];int64_t at=0;
    for(int part=0;part<6;part++){
        int k=part%3,is_scale=part>=3;
        int length=snprintf(suffix,sizeof suffix,"mlp.experts.%d.%s.weight%s",
                            expert,projection[k],is_scale?".qs":"");
        if(length<0||(size_t)length>=sizeof suffix)return -1;
        q38_name(m,name,sizeof name,layer,suffix);
        st_tensor *t=st_find(&x->S,name);
        int64_t width=is_scale?q38_int4_groups(cols[k]):q38_int4_row_bytes(cols[k]);
        int64_t bytes=is_scale?x->scale_count[k]*(int64_t)sizeof(float):x->code_bytes[k];
        if(!t||t->dtype!=(is_scale?2:3)||t->rank!=2||t->shape[0]!=rows[k]||
           t->shape[1]!=width||t->nbytes!=bytes)return -1;
        if(!part){
            if(t->off%(int64_t)sizeof(float))return -1;
            *fd=t->fd;*start=t->off;
        }else if(t->fd!=*fd||t->off!=*start+at)return -1;
        at+=bytes;
    }
    return 0;
}

/* Read <snap>/experts-int4g64/index.json and, if it describes a complete
 * conversion of this model, index every record of the loaded layers. A
 * sidecar for another geometry or format, or whose files disagree with the
 * index, is refused rather than ignored; one still being written is skipped
 * with a line saying so (or refused under Q38_EXPERT_INT4=1). */
static void q38_expert_int4_attach(Model *m,const char *snap) {
    int wanted=q38_expert_int4_wanted();
    if(!wanted)return;
    Cfg *c=&m->c;char dir[2100],path[2200];
    snprintf(dir,sizeof dir,"%s/%s",snap,Q38_INT4_DIR);
    snprintf(path,sizeof path,"%s/index.json",dir);
    FILE *f=fopen(path,"rb");
    if(!f){
        if(wanted>0){
            fprintf(stderr,"Q38_EXPERT_INT4=1 but %s cannot be opened "
                           "(tools/convert_qwen38_experts_int4.py writes it)\n",path);exit(1);
        }
        return;
    }
    char *text=NULL,*arena=NULL;long size=-1;
    if(!fseek(f,0,SEEK_END))size=ftell(f);
    if(size<0||size>(16L<<20)||fseek(f,0,SEEK_SET)||!(text=(char*)malloc((size_t)size+1))||
       fread(text,1,(size_t)size,f)!=(size_t)size){
        fclose(f);q38_int4_refuse(dir,"index.json cannot be read");
    }
    fclose(f);text[size]=0;
    jval *root=json_parse(text,&arena);
    if(!root||root->t!=J_OBJ)q38_int4_refuse(dir,"index.json is not a JSON object");
    const char *format=jstr(root,"format"),*codes=jstr(root,"codes"),*scales=jstr(root,"scales");
    if(!format||strcmp(format,Q38_INT4_FORMAT)||q38_json_int(root,"version")!=Q38_INT4_VERSION||
       !codes||strcmp(codes,"planar64-v8")||!scales||strcmp(scales,"f32-per-64")||
       q38_json_int(root,"group_size")!=XF_BLOCK)
        q38_int4_refuse(dir,"index.json is not format " Q38_INT4_FORMAT " version 1, int4 codes v+8 "
                            "in planar blocks of 64 with f32 scales per 64");
    Q38Int4Experts *x=(Q38Int4Experts*)calloc(1,sizeof(*x));
    if(!x){fprintf(stderr,"OOM int4 expert index\n");exit(1);}
    int rows[3]={c->inter,c->inter,c->hidden},cols[3]={c->hidden,c->hidden,c->inter};
    for(int k=0;k<3;k++){
        x->code_bytes[k]=(int64_t)rows[k]*q38_int4_row_bytes(cols[k]);
        x->scale_count[k]=(int64_t)rows[k]*q38_int4_groups(cols[k]);
        x->scale_off+=x->code_bytes[k];
    }
    x->record_bytes=x->scale_off+
        (x->scale_count[0]+x->scale_count[1]+x->scale_count[2])*(int64_t)sizeof(float);
    if(q38_json_int(root,"layers")!=c->layers||q38_json_int(root,"experts")!=c->experts||
       q38_json_int(root,"hidden_size")!=c->hidden||
       q38_json_int(root,"moe_intermediate_size")!=c->inter||
       q38_json_int(root,"record_bytes")!=x->record_bytes||x->scale_off%(int64_t)sizeof(float))
        q38_int4_refuse(dir,"index.json was written for another geometry than config.json's");
    jval *complete=json_get(root,"complete"),*done=json_get(root,"done");
    if(!complete||complete->t!=J_BOOL||!complete->boolean){
        int layers_done=done&&done->t==J_ARR?done->len:0;
        if(wanted>0){
            fprintf(stderr,"[qwen38] %s: conversion incomplete (%d of %d layers); "
                           "Q38_EXPERT_INT4=1 refuses to fall back to FP8\n",dir,layers_done,c->layers);exit(1);
        }
        fprintf(stderr,"[qwen38] %s: conversion incomplete (%d of %d layers); "
                       "the routed experts stay as the snapshot has them\n",dir,layers_done,c->layers);
        json_free(root);free(arena);free(text);free(x);
        return;
    }
    json_free(root);free(arena);free(text);
    st_init(&x->S,dir);
    x->fd=(int*)malloc((size_t)c->layers*sizeof(int));
    x->off=(int64_t*)malloc((size_t)c->layers*(size_t)c->experts*sizeof(int64_t));
    if(!x->fd||!x->off){fprintf(stderr,"OOM int4 expert index\n");exit(1);}
    for(int layer=0;layer<c->layers;layer++){
        x->fd[layer]=-1;
        for(int expert=0;expert<c->experts;expert++)x->off[(int64_t)layer*c->experts+expert]=-1;
    }
    for(int layer=m->range_begin;layer<m->range_end;layer++)
        for(int expert=0;expert<c->experts;expert++){
            int fd=-1;int64_t start=-1;
            if(q38_int4_record_at(m,x,layer,expert,&fd,&start)||
               (x->fd[layer]>=0&&fd!=x->fd[layer])){
                char why[160];
                snprintf(why,sizeof why,"layer %d expert %d is missing or is not one int4-g64 "
                         "record of %lld bytes",layer,expert,(long long)x->record_bytes);
                q38_int4_refuse(dir,why);
            }
            x->fd[layer]=fd;x->off[(int64_t)layer*c->experts+expert]=start;
        }
    m->x4=x;
}

/* Point a slot's three matrices into one record (a slab or a mapping). */
static void q38_bind_int4_record(Model *m,Slot *slot,uint8_t *record) {
    Cfg *c=&m->c;const Q38Int4Experts *x=m->x4;
    int rows[3]={c->inter,c->inter,c->hidden},cols[3]={c->hidden,c->hidden,c->inter};
    Q38Weight *dst[3]={&slot->gate,&slot->up,&slot->down};
    uint8_t *codes=record;float *scales=(float*)(record+x->scale_off);
    for(int k=0;k<3;k++){
        Q38Weight *weight=dst[k];q38_weight_free(weight);
        weight->data=codes;weight->scales=scales;weight->rows=rows[k];weight->cols=cols[k];
        weight->elements=(int64_t)rows[k]*cols[k];weight->scale_count=x->scale_count[k];
        weight->kind=Q38_WEIGHT_INT4G64;
        codes+=x->code_bytes[k];scales+=x->scale_count[k];
    }
}

/* One read per miss, into the slot's own slab; with COLI_MAP_EXPERTS=1 the
 * slot points into the mapped file instead, as the FP8 path does. Safe from
 * the parallel batch loader: every job owns a distinct slot. */
static void q38_load_int4_record(Model *m,int layer,int expert,Slot *slot) {
    Q38Int4Experts *x=m->x4;
    int fd=x->fd[layer];int64_t off=x->off[(int64_t)layer*m->c.experts+expert];
    const uint8_t *mapped=(const uint8_t*)st_map_shard_range(fd,off,x->record_bytes);
    if(mapped){
        q38_bind_int4_record(m,slot,(uint8_t*)mapped);
        free(slot->int4_slab);slot->int4_slab=NULL;slot->int4_slab_bytes=0;
        return;
    }
    if(!slot->int4_slab||slot->int4_slab_bytes!=x->record_bytes){
        void *replacement=realloc(slot->int4_slab,(size_t)x->record_bytes);
        if(!replacement){fprintf(stderr,"OOM int4 expert record\n");exit(1);}
        slot->int4_slab=replacement;slot->int4_slab_bytes=x->record_bytes;
    }
    q38_bind_int4_record(m,slot,(uint8_t*)slot->int4_slab);
    st_read_range_raw_cap(&x->S,fd,off,x->record_bytes,slot->int4_slab,
                          slot->int4_slab_bytes,1,"pread Qwen3.8 int4 expert");
}

static void q38_load_expert(Model *m,int layer,int eid,Slot *s) {
    Cfg *c=&m->c; int H=c->hidden,I=c->inter; char nm[320],sn[340];
    if(q38_layer_int4(m,layer)){
        double started=now_s();
        q38_load_int4_record(m,layer,eid,s);
        q38_tm_add(m,Q38_TM_EXPERT_READ,started);m->expert_weight_reads++;
        return;
    }
    q38_name(m,nm,sizeof nm,layer,"mlp.experts.gate_up_proj");
    if(st_has(&m->S,nm)){
        st_tensor *t=st_find(&m->S,nm);
        if(t->rank!=3||t->shape[0]!=c->experts||t->shape[1]!=2*I||t->shape[2]!=H){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        int64_t base=(int64_t)eid*2*I*H;
        q38_load_expert_slice(m,nm,t,base,&s->gate,I,H);
        q38_load_expert_slice(m,nm,t,base+(int64_t)I*H,&s->up,I,H);
        q38_name(m,nm,sizeof nm,layer,"mlp.experts.down_proj"); t=st_find(&m->S,nm);
        if(!t||t->rank!=3||t->shape[0]!=c->experts||t->shape[1]!=H||t->shape[2]!=I){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        q38_load_expert_slice(m,nm,t,(int64_t)eid*H*I,&s->down,H,I);
        return;
    }
    if(q38_try_load_native_fp8_expert(m,layer,eid,s))return;
    const char *kind[3]={"gate_proj","up_proj","down_proj"};Q38Weight *dst[3]={&s->gate,&s->up,&s->down};
    int os[3]={I,I,H},is[3]={H,H,I};
    for(int k=0;k<3;k++){
        char suf[192]; snprintf(suf,sizeof suf,"mlp.experts.%d.%s.weight",eid,kind[k]);q38_name(m,nm,sizeof nm,layer,suf);
        st_tensor *t=st_find(&m->S,nm); if(!t){fprintf(stderr,"missing %s\n",nm);exit(1);}
        if(t->dtype==4){snprintf(sn,sizeof sn,"%s_scale_inv",nm);q38_load_fp8_expert_weight(m,nm,sn,dst[k],os[k],is[k]);}
        else q38_load_expert_weight(m,nm,dst[k],os[k],is[k]);
    }
}

/* One byte per expert: routed in this turn or not. The dashboard's Brain tab
 * reads it as the HITS bitmap after every turn (serve_hits in qwen38.c),
 * cleared there. Marked on both lookup paths, single and batched. */
static pthread_mutex_t g_q38_ehit_mx=PTHREAD_MUTEX_INITIALIZER;
static void q38_ehit_mark(Model *m,int layer,int eid) {
    const Cfg *c=&m->c;
    /* The first touch can come from a parallel region (qwen36: the tier
     * warmstart's omp loop calls expert_get from twelve threads at once):
     * one thread published m->ehit while it was still filling the rows and
     * a sibling dereferenced m->ehit[layer] == NULL -- SIGSEGV in about one
     * run in twelve on a CUDA warmstart. Build the table privately, publish
     * it once under a lock (double-checked), and read it with acquire order. */
    uint8_t **ehit=__atomic_load_n(&m->ehit,__ATOMIC_ACQUIRE);
    if(!ehit){
        pthread_mutex_lock(&g_q38_ehit_mx);
        ehit=m->ehit;
        if(!ehit){
            ehit=(uint8_t**)calloc((size_t)c->layers,sizeof(uint8_t*));
            for(int i=0;i<c->layers;i++)ehit[i]=(uint8_t*)calloc((size_t)c->experts,1);
            __atomic_store_n(&m->ehit,ehit,__ATOMIC_RELEASE);
        }
        pthread_mutex_unlock(&g_q38_ehit_mx);
    }
    if(layer>=0&&layer<c->layers&&eid>=0&&eid<c->experts)ehit[layer][eid]=1;
}
static Slot *q38_expert_get(Model *m,int layer,int eid) {
    q38_ehit_mark(m,layer,eid);
    LCache *lc=&m->cache[layer]; int si=lc->by_expert[eid];
    if(si>=0){m->hits++;lc->slots[si].used=++m->clock;return &lc->slots[si];}
    m->miss++; Slot *s;
    if(lc->n<lc->cap){s=&lc->slots[lc->n++];s->eid=-1;}
    else {
        int victim=0;for(int i=1;i<lc->n;i++)if(lc->slots[i].used<lc->slots[victim].used)victim=i;
        s=&lc->slots[victim];if(s->eid>=0)lc->by_expert[s->eid]=-1;
    }
    s->eid=-1;q38_load_expert(m,layer,eid,s);s->eid=eid;s->used=++m->clock;lc->by_expert[eid]=(int)(s-lc->slots);return s;
}

typedef struct {
    int expert;
    Slot *slot;
    st_tensor *weight[3];
} Q38ExpertLoadJob;

/* Reserve an entire routed demand set on the main thread, fill distinct slots
 * in parallel, then publish every cache index on the main thread.  Demand-set
 * residents are protected from victim selection, so no worker can overwrite a
 * slot another selected expert will consume.  Smaller caches and heterogeneous
 * layouts retain the serial LRU path. */
/* Says once per model why the parallel read path is not taken, so a slow
 * prefill on a small cache or a converted container is not a mystery. */
static int q38_expert_batch_fallback(Model *m,Q38ExpertBatchFallback reason,
                                     int layer,int first,int second) {
    if(m->expert_batch_fallback!=Q38_EXPERT_BATCH_FALLBACK_NONE)return 0;
    m->expert_batch_fallback=reason;
    fprintf(stderr,"[qwen38 expert I/O] parallel reads unavailable: ");
    switch(reason){
    case Q38_EXPERT_BATCH_FALLBACK_DISABLED:
        fprintf(stderr,"Q38_EXPERT_PARALLEL_READS=0");break;
    case Q38_EXPERT_BATCH_FALLBACK_CACHE_CAPACITY:
        fprintf(stderr,"cache holds %d experts/layer but the route needs %d; "
                       "lower --ctx/Q38_MAXT or raise --ram",first,second);break;
    case Q38_EXPERT_BATCH_FALLBACK_SCALE_BANK:
        fprintf(stderr,"layer %d has no compatible resident FP8 scale bank",layer);break;
    case Q38_EXPERT_BATCH_FALLBACK_DUPLICATE:
        fprintf(stderr,"layer %d route repeats expert %d",layer,first);break;
    case Q38_EXPERT_BATCH_FALLBACK_LAYOUT:
        fprintf(stderr,"layer %d expert %d is not native block-FP8",layer,first);break;
    default:
        fprintf(stderr,"unknown reason");break;
    }
    fprintf(stderr,"; using serial expert reads\n");
    return 0;
}

static int q38_expert_get_batch(Model *m,int layer,const int *experts,int count,
                                Slot **selected) {
    if(!experts||!selected||count<2)return 0;
    if(!m->expert_parallel_reads)
        return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_DISABLED,layer,0,0);
    LCache *cache=&m->cache[layer];
    if(count>cache->cap)
        return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_CACHE_CAPACITY,
                                         layer,cache->cap,count);
    int x4=q38_layer_int4(m,layer);
    if(!x4&&!q38_prepare_expert_scale_bank(m,layer))
        return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_SCALE_BANK,layer,0,0);
    /* The demand set is no longer bounded by the decode top-k: the MoE prefill
     * hands over the whole chunk union (up to the cache cap) so its loads run
     * one OMP wave instead of serial groups of Q38_MAX_TOPK.  Load grouping
     * never touches FP order: routed outputs are written per assignment and
     * the per-position expert sum follows the router order, so a bigger wave
     * only changes WHICH slots serve the reads, not the arithmetic. */
    Q38ExpertLoadJob *jobs=malloc((size_t)count*sizeof(*jobs));
    if(!jobs)return 0;
    for(int index=0;index<count;index++){
        int expert=experts[index];
        if(expert<0||expert>=m->c.experts){free(jobs);return 0;}
        q38_ehit_mark(m,layer,expert);
        for(int previous=0;previous<index;previous++)
            if(experts[previous]==expert){
                free(jobs);
                return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_DUPLICATE,
                                                 layer,expert,0);
            }
        int slot_index=cache->by_expert[expert];
        if(slot_index>=0){
            if(slot_index>=cache->n||cache->slots[slot_index].eid!=expert){free(jobs);return 0;}
            continue;
        }
        st_tensor *weight[3];
        if(!x4&&!q38_native_fp8_expert_tensors(m,layer,expert,weight)){
            free(jobs);
            return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_LAYOUT,
                                             layer,expert,0);
        }
    }
    unsigned char *protected_slots=(unsigned char*)calloc((size_t)cache->cap,1);
    if(!protected_slots){fprintf(stderr,"OOM expert batch reservations\n");exit(1);}
    for(int index=0;index<count;index++){
        int slot_index=cache->by_expert[experts[index]];
        if(slot_index>=0)protected_slots[slot_index]=1;
    }
    int job_count=0;
    for(int index=0;index<count;index++){
        int expert=experts[index],slot_index=cache->by_expert[expert];Slot *slot;
        if(slot_index>=0){
            slot=&cache->slots[slot_index];m->hits++;
        }else{
            m->miss++;
            if(cache->n<cache->cap){
                slot=&cache->slots[cache->n++];slot->eid=-1;
            }else{
                int victim=-1;
                for(int candidate=0;candidate<cache->n;candidate++)
                    if(!protected_slots[candidate]&&
                       (victim<0||cache->slots[candidate].used<cache->slots[victim].used))
                        victim=candidate;
                if(victim<0){
                    fprintf(stderr,"Qwen3.8 expert demand set has no reservable cache slot\n");
                    exit(1);
                }
                slot=&cache->slots[victim];
                if(slot->eid>=0)cache->by_expert[slot->eid]=-1;
            }
            slot->eid=-1;jobs[job_count].expert=expert;jobs[job_count].slot=slot;
            if(!x4&&!q38_native_fp8_expert_tensors(m,layer,expert,jobs[job_count].weight)){
                fprintf(stderr,"Qwen3.8 expert layout changed during batch reservation\n");exit(1);
            }
            job_count++;
        }
        slot->used=++m->clock;selected[index]=slot;
        protected_slots[slot-cache->slots]=1;
    }
    free(protected_slots);
    if(job_count){
        int workers=job_count;
#ifdef _OPENMP
        int thread_limit=omp_get_max_threads();if(workers>thread_limit)workers=thread_limit;
#else
        (void)workers;   /* only the pragma below reads it, and that is gone without OpenMP */
#endif
        double started=now_s();
        #pragma omp parallel for schedule(static) num_threads(workers) if(job_count>1)
        for(int job=0;job<job_count;job++){
            if(x4)q38_load_int4_record(m,layer,jobs[job].expert,jobs[job].slot);
            else q38_load_native_fp8_ranges(m,layer,jobs[job].expert,jobs[job].slot,
                                            jobs[job].weight);
        }
        q38_tm_add(m,Q38_TM_EXPERT_READ,started);
        m->expert_weight_reads+=(uint64_t)job_count*(x4?1:2);
        if(!x4)m->expert_pair_reads+=(uint64_t)job_count;
        if(job_count>1)m->expert_parallel_batches++;
        for(int job=0;job<job_count;job++){
            Slot *slot=jobs[job].slot;slot->eid=jobs[job].expert;
            cache->by_expert[jobs[job].expert]=(int)(slot-cache->slots);
        }
    }
    free(jobs);
    return 1;
}

static void q38_ple_row(Model *m,int64_t row,float *out) {
    Cfg *c=&m->c; int p=0;
    while(p+1<m->ple_part_count&&row>=m->ple_part_start[p+1])p++;
    if(p>=m->ple_part_count||row<m->ple_part_start[p]){fprintf(stderr,"PLE row out of range: %lld\n",(long long)row);exit(1);}
    int64_t local=row-m->ple_part_start[p]; st_tensor *t=m->ple_parts[p]; const char *nm=m->ple_part_names[p];
    Q38_NEED(c->ngram_head_dim>0&&local>=0&&local<=INT64_MAX/c->ngram_head_dim,
             "PLE row byte offset overflows");
    if(t->dtype==4){
        uint8_t raw[512]; Q38_NEED(c->ngram_head_dim<=(int)sizeof raw,"PLE row too wide");
        st_read_slice_raw_cap(&m->S,nm,local*c->ngram_head_dim,c->ngram_head_dim,raw,sizeof raw,1);
        for(int d=0;d<c->ngram_head_dim;d++)out[d]=e4m3_decode(raw[d])*m->ple_weight_scale;
    } else st_read_slice_f32(&m->S,nm,local*c->ngram_head_dim,c->ngram_head_dim,out,1);
}

static int64_t q38_hash_row(Model *m,int head,int ngram,int64_t cur,int64_t p1,int64_t p2) {
    uint64_t x=(uint64_t)cur*(uint64_t)m->ple_multipliers[0];
    x^=(uint64_t)p1*(uint64_t)m->ple_multipliers[1];
    if(ngram==3)x^=(uint64_t)p2*(uint64_t)m->ple_multipliers[2];
    int64_t sx=(int64_t)x,mod=m->ple_head_vocab[head],r=sx%mod;if(r<0)r+=mod;
    return m->ple_head_offset[head]+r;
}


/* Anticipa le letture PLE all'inizio del forward invece di emetterle inline al
 * layer che le usa.
 *
 * Perche' si puo' fare, e perche' NON si puo' fare con gli esperti: gli indici
 * delle righe PLE sono funzione pura degli id dei token e della finestra di due
 * token che li precede -- q38_hash_row non guarda nessuno stato nascosto. Sono
 * quindi noti PRIMA che parta un solo calcolo. Gli esperti no: quelli dipendono
 * dal router del layer precedente, e per questo la loro finestra di anticipo e'
 * di un layer e non di tutto il forward.
 *
 * Il PLE cade sul layer 2 di 48, quindi fra l'emissione e l'uso ci sono due
 * layer interi di calcolo -- con il loro streaming di esperti, che su questo
 * motore e' la parte lenta. E' abbondantemente il tempo di far arrivare 16
 * righe da 160 byte.
 *
 * Sono anche letture PARALLELE invece che in fila: prima erano 16 pread seriali
 * a queue depth 1 per token, che in prefill diventano lunghezza-del-prompt per
 * 16, una alla volta.
 *
 * Riordino puro: stessi byte, letti prima. I token non cambiano, e il test lo
 * pretende invece di darlo per scontato. */
static void q38_ple_prefetch(Model *m,const int *ids,int S) {
    Cfg *c=&m->c;
    free(m->ple_pref); m->ple_pref=NULL; m->ple_pref_rows=0;
    if(c->ple_layer<0||S<1||c->ngram_heads<1||c->ngram_head_dim<1) return;
    if(!q38_env_bool("Q38_PLE_PREFETCH",1)) return;

    int64_t per_row=c->ngram_head_dim, per_pos=(int64_t)c->ngram_heads*per_row;
    if(S>INT_MAX/c->ngram_heads) return;
    float *buffer=(float*)malloc((size_t)S*per_pos*sizeof(float));
    int64_t *rows=(int64_t*)malloc((size_t)S*c->ngram_heads*sizeof(int64_t));
    if(!buffer||!rows){free(buffer);free(rows);return;}   /* niente prefetch: si legge inline */

    /* La finestra di due token viene SIMULATA, non mutata: q38_ple la aggiorna
     * per conto suo mentre gira, e toccarla qui la farebbe avanzare due volte. */
    int64_t history[2]={m->ple_history[0],m->ple_history[1]};
    int history_len=m->ple_history_len;
    for(int s=0;s<S;s++){
        int64_t p1=history_len>=1?history[history_len-1]:c->eos_id;
        int64_t p2=history_len>=2?history[history_len-2]:c->eos_id;
        for(int h=0;h<c->ngram_heads;h++){
            int ng=h<c->heads_per_ngram?2:3;
            rows[(int64_t)s*c->ngram_heads+h]=q38_hash_row(m,h,ng,ids[s],p1,p2);
        }
        if(ids[s]==c->eos_id) history_len=0;
        else if(history_len==0){history[0]=ids[s];history_len=1;}
        else if(history_len==1){history[1]=ids[s];history_len=2;}
        else {history[0]=history[1];history[1]=ids[s];}
    }

    int64_t total=(int64_t)S*c->ngram_heads;
    volatile int failed=0;
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<total;i++){
        if(failed) continue;
        q38_ple_row(m,rows[i],buffer+i*per_row);
    }
    free(rows);
    if(failed){free(buffer);return;}
    m->ple_pref=buffer; m->ple_pref_rows=S;
}

/* The key and value projections read only the token's n-gram embedding, so they
 * run once for a block of rows: one S-row call (a device GEMM) instead of one
 * GEMV per token. The embedding lookups walk the n-gram history and the
 * convolution its ring, both in token order as before; every CPU kernel computes
 * each row on its own, so a block gives the bits of the per-token calls. A
 * projection the CUDA tier holds answers one row at a time (the tier serves
 * S == 1, prefill rows would fall to the CPU), so with one the block is a row,
 * as with Q38_PREFILL_BATCH=0. */
static int q38_bounded_prefill_rows(int requested,uint64_t fixed,uint64_t per_row);
static void q38_ple(Model *m,const int *ids,int S,const float *hyper,float *out) {
    double phase_started=now_s();
    Cfg *c=&m->c; Layer *l=&m->L[c->ple_layer]; int H=c->hidden,C=c->hc_count,W=c->hc_width,E=c->ple_dim;
    int B=!m->prefill_batch||l->ple_key.gpu||l->ple_value.gpu?1:
          q38_bounded_prefill_rows(S,0,((uint64_t)E+(uint64_t)W+(uint64_t)H)*sizeof(float));
    float *embs=falloc((int64_t)B*E),*keysb=falloc((int64_t)B*W),*valueb=falloc((int64_t)B*H);
    float *kn=falloc(W),*qn=falloc(W),*gated=falloc(W),*norm=falloc(W);
    int state_len=(c->ple_convk-1)*c->ngram_size; float *ring=m->PLE_conv_state;
    for(int base=0;base<S;base+=B){
        int rows=S-base<B?S-base:B;
        for(int r=0;r<rows;r++){
            int s=base+r; float *emb=embs+(int64_t)r*E;
            int64_t p1=m->ple_history_len>=1?m->ple_history[m->ple_history_len-1]:c->eos_id;
            int64_t p2=m->ple_history_len>=2?m->ple_history[m->ple_history_len-2]:c->eos_id;
            if(m->ple_pref&&s<m->ple_pref_rows){
                /* gia' in memoria: le ha portate q38_ple_prefetch mentre i primi
                 * layer calcolavano */
                memcpy(emb,m->ple_pref+(int64_t)s*c->ngram_heads*c->ngram_head_dim,
                       (size_t)c->ngram_heads*c->ngram_head_dim*sizeof(float));
            } else for(int h=0;h<c->ngram_heads;h++){
                int ng=h<c->heads_per_ngram?2:3; int64_t row=q38_hash_row(m,h,ng,ids[s],p1,p2);
                q38_ple_row(m,row,emb+(int64_t)h*c->ngram_head_dim);
            }
            if(ids[s]==c->eos_id)m->ple_history_len=0;
            else if(m->ple_history_len==0){m->ple_history[0]=ids[s];m->ple_history_len=1;}
            else if(m->ple_history_len==1){m->ple_history[1]=ids[s];m->ple_history_len=2;}
            else {m->ple_history[0]=m->ple_history[1];m->ple_history[1]=ids[s];}
            if(s<m->snap_rows){   /* a verify: the history after this row (the ring below) */
                memcpy(m->snap_ple_history[s],m->ple_history,sizeof(m->snap_ple_history[s]));
                m->snap_ple_history_len[s]=m->ple_history_len;
            }
        }
        q38_dense_matmul(m,keysb,embs,&l->ple_key,rows,E,W);q38_dense_matmul(m,valueb,embs,&l->ple_value,rows,E,H);
        for(int r=0;r<rows;r++){
            int s=base+r; const float *keys=keysb+(int64_t)r*W,*value=valueb+(int64_t)r*H;
            for(int b=0;b<C;b++){
                q38_rms0(kn+(int64_t)b*H,keys+(int64_t)b*H,l->ple_norm_key+(int64_t)b*H,H,c->eps);
                q38_rms0(qn+(int64_t)b*H,hyper+(int64_t)s*W+(int64_t)b*H,l->ple_norm_query+(int64_t)b*H,H,c->eps);
                float dot=0.f;for(int d=0;d<H;d++)dot+=kn[(int64_t)b*H+d]*qn[(int64_t)b*H+d];dot/=sqrtf((float)H);
                float shaped=copysignf(sqrtf(fmaxf(fabsf(dot),1e-6f)),dot),g=q38_sigmoid(shaped);
                for(int d=0;d<H;d++)gated[(int64_t)b*H+d]=g*value[d];
                q38_rms0(norm+(int64_t)b*H,gated+(int64_t)b*H,l->ple_norm_conv+(int64_t)b*H,H,c->eps);
            }
            for(int d=0;d<W;d++){
                float a=l->ple_conv[(int64_t)d*c->ple_convk+c->ple_convk-1]*norm[d];
                for(int k=0;k<c->ple_convk-1;k++)a+=l->ple_conv[(int64_t)d*c->ple_convk+k]*ring[(int64_t)d*state_len+k*c->ngram_size];
                out[(int64_t)s*W+d]=gated[d]+q38_silu(a);
                float *rg=ring+(int64_t)d*state_len;for(int k=0;k<state_len-1;k++)rg[k]=rg[k+1];rg[state_len-1]=norm[d];
            }
            if(s<m->snap_rows)   /* a verify: as q38_deltanet */
                memcpy(m->snap_ple[s],ring,(size_t)c->hc_width*state_len*sizeof(float));
        }
    }
    free(embs);free(keysb);free(valueb);free(kn);free(qn);free(gated);free(norm);
    q38_tm_add(m,Q38_TM_PLE,phase_started);
}

/* The chunk ceiling and the workspace budget used to be compile-time only.  An
 * isolated prefill measurement (274 tokens, one forward) showed that the
 * ceiling -- not the budget -- is what binds: 32 rows cut the prompt into nine
 * chunks, each chunk touches ~91 distinct experts, so a loaded expert serves
 * ~3.3 rows.  That drags 4.69 MiB of FP8 weights in for three rows of
 * activations, which is decode-grade arithmetic intensity inside a path that is
 * supposed to be batched, and it shows: 1.29 TFLOP in 48.8 s is 26.5 GFLOP/s,
 * a few percent of what the cores can do.  Both values are therefore runtime
 * knobs now.  Widening the chunk cannot change any result -- boundaries alter
 * neither routing nor accumulation order -- so this is a pure A/B. */
static int q38_env_positive_int(const char *name,int default_value,
                                int max_value) {
    const char *value=getenv(name);
    if(!value||!*value)return default_value;
    char *end=NULL;long parsed=strtol(value,&end,10);
    if(end==value||*end||parsed<1||parsed>(long)max_value){
        fprintf(stderr,"%s must be an integer in 1..%d\n",name,max_value);
        exit(1);
    }
    return (int)parsed;
}

static int q38_prefill_batch_rows(void) {
    static int cached=0;
    if(!cached)
        cached=q38_env_positive_int("Q38_PREFILL_BATCH_ROWS",
                                    Q38_PREFILL_BATCH_ROWS,1<<20);
    return cached;
}

/* Expressed in MiB because the byte count is the thing a human gets wrong. */
static uint64_t q38_prefill_workspace_bytes(void) {
    static uint64_t cached=0;
    if(!cached)
        cached=(uint64_t)q38_env_positive_int(
                   "Q38_PREFILL_WORKSPACE_MIB",
                   (int)(Q38_PREFILL_WORKSPACE_BYTES>>20),4096)<<20;
    return cached;
}

/* Choose a context-independent prefill chunk whose private workspace fits the
 * common target.  Callers provide exact fixed and per-row byte counts; even a
 * hostile-but-valid geometry gets one row rather than an unbounded allocation. */
static int q38_bounded_prefill_rows(int requested,uint64_t fixed,
                                    uint64_t per_row) {
    int ceiling=q38_prefill_batch_rows();
    uint64_t budget=q38_prefill_workspace_bytes();
    int rows=requested<ceiling?requested:ceiling;
    if(rows<1)return 1;
    for(;rows>1;rows--)
        if(per_row<=UINT64_MAX/(uint64_t)rows&&
           fixed<=UINT64_MAX-per_row*(uint64_t)rows&&
           fixed+per_row*(uint64_t)rows<=budget)
            return rows;
    return 1;
}

/* Batch the four resident DeltaNet input projections and the output projection
 * in bounded chunks.  The convolution and recurrent update remain strictly
 * token-causal inside each chunk, so chunk boundaries cannot change state or
 * floating-point order. */
static void q38_deltanet(Model *m,Layer *l,int layer,const float *x,int S,
                         float *out) {
    double phase_started=now_s();
    Cfg *c=&m->c;
    int H=c->hidden,VH=c->dn_vheads,KH=c->dn_kheads;
    int KD=c->dn_kdim,VD=c->dn_vdim,CD=c->dn_conv_dim,CK=c->dn_convk;
    int V=VH*VD,K=KH*KD,rep=VH/KH;
    uint64_t row_floats=(uint64_t)CD+2u*(uint64_t)V+2u*(uint64_t)VH;
    uint64_t fixed_floats=(uint64_t)CD+2u*(uint64_t)VH*(uint64_t)KD+
                          (uint64_t)V;
    uint64_t row_bytes=row_floats>UINT64_MAX/sizeof(float)?
                       UINT64_MAX:row_floats*sizeof(float);
    uint64_t fixed_bytes=fixed_floats>UINT64_MAX/sizeof(float)?
                         UINT64_MAX:fixed_floats*sizeof(float);
    int rows_capacity=m->prefill_batch?
                      q38_bounded_prefill_rows(S,fixed_bytes,row_bytes):1;

    float *qkv=falloc((int64_t)rows_capacity*CD);
    float *z=falloc((int64_t)rows_capacity*V);
    float *bb=falloc((int64_t)rows_capacity*VH);
    float *aa=falloc((int64_t)rows_capacity*VH);
    float *norm=falloc((int64_t)rows_capacity*V);
    float *conv=falloc(CD);
    float *q=falloc((int64_t)VH*KD),*k=falloc((int64_t)VH*KD);
    float *core=falloc(V);
    float *rec=m->DN_rec[layer],*ring=m->DN_conv[layer];

    for(int base=0;base<S;) {
        int rows=S-base<rows_capacity?S-base:rows_capacity;
        const float *chunk=x+(int64_t)base*H;
        q38_dense_matmul(m,qkv,chunk,&l->dn_qkv,rows,H,CD);
        q38_dense_matmul(m,z,chunk,&l->dn_z,rows,H,V);
        q38_dense_matmul(m,bb,chunk,&l->dn_b,rows,H,VH);
        q38_dense_matmul(m,aa,chunk,&l->dn_a,rows,H,VH);

        for(int s=0;s<rows;s++) {
            float *qkv_row=qkv+(int64_t)s*CD;
            float *z_row=z+(int64_t)s*V;
            float *b_row=bb+(int64_t)s*VH;
            float *a_row=aa+(int64_t)s*VH;
            for(int d=0;d<CD;d++) {
                float value=l->dn_conv[(int64_t)d*CK+CK-1]*qkv_row[d];
                float *history=ring+(int64_t)d*(CK-1);
                for(int tap=0;tap<CK-1;tap++)
                    value+=l->dn_conv[(int64_t)d*CK+tap]*history[tap];
                conv[d]=q38_silu(value);
                for(int tap=0;tap<CK-2;tap++)history[tap]=history[tap+1];
                history[CK-2]=qkv_row[d];
            }
            const float *qi=conv,*ki=conv+K,*vi=conv+2*K;
            for(int h=0;h<VH;h++) {
                float *qh=q+(int64_t)h*KD,*kh=k+(int64_t)h*KD;
                memcpy(qh,qi+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
                memcpy(kh,ki+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
                double qsum=1e-6,ksum=1e-6;
                for(int d=0;d<KD;d++) {
                    qsum+=(double)qh[d]*qh[d];
                    ksum+=(double)kh[d]*kh[d];
                }
                float qscale=1.f/sqrtf((float)qsum)/sqrtf((float)KD);
                float kscale=1.f/sqrtf((float)ksum);
                for(int d=0;d<KD;d++){qh[d]*=qscale;kh[d]*=kscale;}
            }
            #pragma omp parallel for schedule(static)
            for(int h=0;h<VH;h++) {
                float *state=rec+(int64_t)h*KD*VD;
                const float *qh=q+(int64_t)h*KD;
                const float *kh=k+(int64_t)h*KD;
                const float *vh=vi+(int64_t)h*VD;
                float alpha=expf(-expf(l->dn_alog[h])*
                                 q38_softplus(a_row[h]+l->dn_dtbias[h]));
                float beta=q38_sigmoid(b_row[h]);
                float delta[512];
                int64_t state_cells=(int64_t)KD*VD;
                for(int64_t cell=0;cell<state_cells;cell++)state[cell]*=alpha;
                for(int value=0;value<VD;value++) {
                    float previous=0.f;
                    for(int d=0;d<KD;d++)
                        previous+=kh[d]*state[(int64_t)d*VD+value];
                    delta[value]=(vh[value]-previous)*beta;
                }
                for(int d=0;d<KD;d++)for(int value=0;value<VD;value++)
                    state[(int64_t)d*VD+value]+=kh[d]*delta[value];
                for(int value=0;value<VD;value++) {
                    float current=0.f;
                    for(int d=0;d<KD;d++)
                        current+=qh[d]*state[(int64_t)d*VD+value];
                    core[(int64_t)h*VD+value]=current;
                }
            }
            float *norm_row=norm+(int64_t)s*V;
            for(int h=0;h<VH;h++)
                q38_rmsg(norm_row+(int64_t)h*VD,core+(int64_t)h*VD,
                         z_row+(int64_t)h*VD,l->dn_norm,VD,c->eps,1);
            /* a verify: the state this row leaves, for the rollback of a
             * draft rejected after it (q38_spec_rollback) */
            if(base+s<m->snap_rows){
                memcpy(m->snap_rec[base+s][layer],rec,(size_t)VH*KD*VD*sizeof(float));
                memcpy(m->snap_conv[base+s][layer],ring,(size_t)CD*(CK-1)*sizeof(float));
            }
        }
        q38_dense_matmul(m,out+(int64_t)base*H,norm,&l->dn_out,
                         rows,V,H);
        base+=rows;
    }
    free(qkv);free(z);free(bb);free(aa);free(norm);free(conv);
    free(q);free(k);free(core);
    q38_tm_add(m,Q38_TM_DELTANET,phase_started);
}

typedef struct { float score; int block; } Q38Block;
static int q38_block_desc(const void *aa,const void *bb){
    const Q38Block *a=(const Q38Block*)aa,*b=(const Q38Block*)bb;
    if(a->score>b->score)return -1;if(a->score<b->score)return 1;return a->block-b->block;
}

static void q38_attention(Model *m,Layer *l,int layer,const float *x,int S,int pos_base,float *out) {
    Cfg *c=&m->c;int H=c->hidden,QH=c->q_heads,KVH=c->kv_heads,D=c->head_dim;
    float theta=layer<c->layers?c->theta:c->mtp_theta;   /* the MTP head has its own RoPE base */
    int IQ=c->idx_qheads,ID=c->idx_dim,R=c->idx_ratio,maxsel=c->idx_budget+R-1;
    float *qp=falloc((int64_t)S*QH*2*D),*kp=falloc((int64_t)S*KVH*D),*vp=falloc((int64_t)S*KVH*D);
    float *ip=falloc((int64_t)S*(IQ+c->idx_kheads)*ID);
    q38_dense_matmul(m,qp,x,&l->q,S,H,QH*2*D);q38_dense_matmul(m,kp,x,&l->k,S,H,KVH*D);q38_dense_matmul(m,vp,x,&l->v,S,H,KVH*D);
    q38_dense_matmul(m,ip,x,&l->idx_qk,S,H,(IQ+c->idx_kheads)*ID);
    /* Cause before parallelism: the K/V/IK writes are disjoint per position
     * (each s writes only its own row) and must be complete before the
     * ranking, which reads the whole IK[0..pos] prefix.  Guarded on S>1 so
     * decode keeps the serial path it has today. */
    #pragma omp parallel for schedule(static) if(S>1)
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        for(int h=0;h<KVH;h++){
            float *kh=kp+(int64_t)s*KVH*D+(int64_t)h*D;q38_rms0(kh,kh,l->kn,D,c->eps);q38_rope(kh,D,c->rotary_dim,pos,theta);
            memcpy(m->K[layer]+((int64_t)h*m->kv_cap+pos)*D,kh,(size_t)D*sizeof(float));
            memcpy(m->V[layer]+((int64_t)h*m->kv_cap+pos)*D,vp+(int64_t)s*KVH*D+(int64_t)h*D,(size_t)D*sizeof(float));
        }
        memcpy(m->IK[layer]+(int64_t)pos*ID,ip+(int64_t)s*(IQ+1)*ID+(int64_t)IQ*ID,(size_t)ID*sizeof(float));
    }
    float *heads=falloc((int64_t)S*QH*D);
    /* Ranking and attention are independent per position: no shared writes
     * (heads is row-disjoint, the scratch is per-thread) and no FP order
     * changes inside a position, so the result is bit-identical to the
     * serial path. The scheduling is dynamic because the ranking cost grows
     * with the position (the IK prefix to read is O(pos)). */
    double index_dt=0,attn_dt=0;
    /* Wall, not aggregate CPU: the reduction below sums per-thread seconds, so
     * at prefill these two phases would report ~20x what the clock saw while
     * every other phase reports wall -- on a 3006-token prompt the phase sum
     * came to 510 s against a 268 s TTFT, the parts outweighing the whole.
     * Take the clock across the whole team and split it by CPU share.  At
     * decode S==1 the loop is serial, cpu_total equals the wall, and the
     * rescale below is an exact no-op. */
    double qsa_wall_started=now_s();
    #pragma omp parallel for schedule(dynamic,8) reduction(+:index_dt,attn_dt) if(S>1)
    for(int s=0;s<S;s++){
        int pos=pos_base+s,visible=pos+1,blocks=visible/R,tail=blocks*R;
        double phase_started=now_s();
        float *qidx=falloc((int64_t)IQ*ID),*pool=falloc(ID);
        int *selected=(int*)malloc((size_t)maxsel*sizeof(int));
        if(!selected){fprintf(stderr,"OOM QSA selection\n");exit(1);}
        for(int h=0;h<IQ;h++){float *qh=qidx+(int64_t)h*ID;memcpy(qh,ip+(int64_t)s*(IQ+1)*ID+(int64_t)h*ID,(size_t)ID*sizeof(float));q38_rms0(qh,qh,l->idx_qn,ID,c->eps);q38_rope(qh,ID,c->rotary_dim,pos,theta);}
        int take=blocks<c->idx_budget/R?blocks:c->idx_budget/R,nsel=0;
        Q38Block *rank=blocks?(Q38Block*)malloc((size_t)blocks*sizeof(Q38Block)):NULL;
        if(blocks&&!rank){fprintf(stderr,"OOM QSA block ranking\n");exit(1);}
        for(int b=0;b<blocks;b++){
            memset(pool,0,(size_t)ID*sizeof(float));for(int r=0;r<R;r++){const float *raw=m->IK[layer]+(int64_t)(b*R+r)*ID;for(int d=0;d<ID;d++)pool[d]+=raw[d]/R;}
            q38_rms0(pool,pool,l->idx_kn,ID,c->eps);q38_rope(pool,ID,c->rotary_dim,b*R,theta);
            float score=0.f;for(int h=0;h<IQ;h++){float a=0.f;for(int d=0;d<ID;d++)a+=qidx[(int64_t)h*ID+d]*pool[d];if(a>0.f)score+=a;}rank[b]=(Q38Block){score/sqrtf((float)ID),b};
        }
        if(blocks)qsort(rank,(size_t)blocks,sizeof(Q38Block),q38_block_desc);
        for(int z=0;z<take;z++)for(int r=0;r<R;r++)selected[nsel++]=rank[z].block*R+r;
        for(int t=tail;t<visible;t++)selected[nsel++]=t;free(rank);
        index_dt+=now_s()-phase_started; phase_started=now_s();
        for(int h=0;h<QH;h++){
            float *qraw=qp+(int64_t)s*QH*2*D+(int64_t)h*2*D;
            float *qh=falloc(D);memcpy(qh,qraw,(size_t)D*sizeof(float));q38_rms0(qh,qh,l->qn,D,c->eps);q38_rope(qh,D,c->rotary_dim,pos,theta);
            float *score=falloc(nsel);float mx=-INFINITY;
            int khidx=h/(QH/KVH);for(int j=0;j<nsel;j++){const float *kh=m->K[layer]+((int64_t)khidx*m->kv_cap+selected[j])*D;float a=0.f;for(int d=0;d<D;d++)a+=qh[d]*kh[d];a/=sqrtf((float)D);score[j]=a;if(a>mx)mx=a;}
            float den=0.f;for(int j=0;j<nsel;j++){score[j]=expf(score[j]-mx);den+=score[j];}
            float *oh=heads+(int64_t)s*QH*D+(int64_t)h*D;memset(oh,0,(size_t)D*sizeof(float));
            for(int j=0;j<nsel;j++){float a=score[j]/den;const float *vh=m->V[layer]+((int64_t)khidx*m->kv_cap+selected[j])*D;for(int d=0;d<D;d++)oh[d]+=a*vh[d];}
            for(int d=0;d<D;d++)oh[d]*=q38_sigmoid(qraw[D+d]);free(qh);free(score);
        }
        attn_dt+=now_s()-phase_started;
        free(qidx);free(pool);free(selected);
    }
    double qsa_wall=now_s()-qsa_wall_started,cpu_total=index_dt+attn_dt;
    if(cpu_total>0.0){
        index_dt=qsa_wall*(index_dt/cpu_total);
        attn_dt =qsa_wall*(attn_dt /cpu_total);
    }
    m->timers.seconds[Q38_TM_QSA_INDEX]+=index_dt;
    m->timers.seconds[Q38_TM_QSA_ATTENTION]+=attn_dt;
    q38_dense_matmul(m,out,heads,&l->o,S,QH*D,H);
    free(qp);free(kp);free(vp);free(ip);free(heads);
}

/* The single-row path is intentionally kept separate from prefill.  Decode is
 * the latency-sensitive steady state and should not pay for route tables or a
 * prompt-sized workspace. */
/* ---- CUDA VRAM expert tier (qwen36_tier.c, fp8 streaming mode) ------------
 * The tier keeps its own copies of hot experts in VRAM; the RAM LRU below is
 * untouched by it. qt_issue() takes the resident experts of a token's route
 * off the CPU, qt_take() adds their outputs back, and every expert the CPU
 * did compute is reported with qt_note() so the tier can promote it. The
 * tier copies what it needs during qt_note; the slot may be recycled by the
 * next token. Only native FP8 slots are reported: gate.data at the slab start
 * is what q38_bind_fp8_slot() produces, an expanded slot points elsewhere. */
static void q38_tier_note(int layer,int eid,const Slot *ex) {
    if(!qt_ready()||!ex->fp8_slab||ex->gate.data!=ex->fp8_slab)return;
    qt_note(layer,eid,(const uint8_t*)ex->gate.data,(const uint8_t*)ex->up.data,
            (const uint8_t*)ex->down.data,ex->gate.scales,ex->up.scales,ex->down.scales);
}

/* ---- dense trunk on the GPU (R7b, stage 1) ---------------------------------
 * The BF16 trunk is the largest fixed cost of a decode token here (about a
 * third), and it is bandwidth-bound. Every matrix of at least 1 MiB is
 * offered to the tier's placer by name and layer before qt_init; whatever
 * the placer accepts is quantized to int8 per row at start (scale = max|w| /
 * 127, the qwen36 dnproj/lmhead format) and uploaded once; the BF16 copy
 * stays for prefill and as fallback. Q38_TRUNK_GPU=0 keeps the trunk on the
 * CPU (parity runs against the BF16 reference). */
typedef struct { Q38Weight *w; char name[16]; int layer, cpu_only; } Q38TrunkItem;
static Q38TrunkItem *g_trunk; static int g_trunk_n, g_trunk_cap, g_trunk_offer_gpu;
static long g_trunk_min_kb; static const char *g_trunk_skip;   /* read once per load in q38_trunk_offer_all */
static void q38_trunk_add(Q38Weight *w,const char *name,int layer) {
    if(!w||!w->data||(w->kind!=Q38_WEIGHT_BF16&&w->kind!=Q38_WEIGHT_F32))return;
    size_t bytes=(size_t)w->rows*w->cols+(size_t)w->rows*sizeof(float);
    /* Q38_TRUNK_MIN_KB (default 1024): a round trip costs more than a tiny
     * GEMV saves, and a tiny matrix in BF16 costs nothing on the CPU either;
     * Q38_TRUNK_SKIP=name,name: leave those components in BF16 on the CPU
     * (bisecting a numeric difference, or a component that does not pay) */
    long min_kb=g_trunk_min_kb; const char *skip=g_trunk_skip;
    if(bytes<(size_t)min_kb*1024)return;
    if(skip&&*skip){
        size_t n=strlen(name); const char *s=skip;
        while(*s){ const char *c=strchr(s,','); size_t l=c?(size_t)(c-s):strlen(s);
                   if(l==n&&!strncmp(s,name,n))return; if(!c)break; s=c+1; }
    }
    if(g_trunk_n==g_trunk_cap){
        g_trunk_cap=g_trunk_cap?2*g_trunk_cap:256;
        g_trunk=(Q38TrunkItem*)realloc(g_trunk,(size_t)g_trunk_cap*sizeof(*g_trunk));
        if(!g_trunk){fprintf(stderr,"OOM trunk table\n");exit(1);}
    }
    Q38TrunkItem *it=&g_trunk[g_trunk_n++]; it->w=w; it->layer=layer; it->cpu_only=0;
    snprintf(it->name,sizeof it->name,"%s",name);
    if(g_trunk_offer_gpu)qt_trunk_offer(it->name,layer,bytes);
}
static int q38_trunk_enabled(void) {
    const char *e=getenv("Q38_TRUNK_GPU"); return !(e&&e[0]=='0'&&!e[1]);
}
/* the trunk table, built once: lm_head first (the placer takes it first),
 * then the layers in order so a partial placement is a prefix of the layers.
 * The same table feeds the CPU's int8 rows; the placer is told about the
 * matrices only when the GPU trunk is enabled (Q38_TRUNK_GPU). */
static void q38_trunk_offer_all(Model *m) {
    g_trunk_n=0; m->trunk_table_built=1;          /* rebuilt per load: a test opens several models in one process */
    g_trunk_offer_gpu=q38_trunk_enabled();
    { const char *e=getenv("Q38_TRUNK_MIN_KB"); g_trunk_min_kb=e?atol(e):1024; g_trunk_skip=getenv("Q38_TRUNK_SKIP"); }
    Cfg *c=&m->c;
    q38_trunk_add(&m->lm_head,"lmhead",0);
    for(int l=0;l<c->layers;l++){
        Layer *L=&m->L[l];
        if(c->is_attn[l]){
            q38_trunk_add(&L->q,"attnq",l); q38_trunk_add(&L->k,"attnk",l);
            q38_trunk_add(&L->v,"attnv",l); q38_trunk_add(&L->o,"attno",l);
            q38_trunk_add(&L->idx_qk,"qsaidx",l);
        } else {
            q38_trunk_add(&L->dn_qkv,"dnqkv",l); q38_trunk_add(&L->dn_z,"dnz",l);
            q38_trunk_add(&L->dn_out,"dnout",l);
        }
        q38_trunk_add(&L->attn_gr.down,"hcad",l); q38_trunk_add(&L->attn_gr.up,"hcau",l);
        q38_trunk_add(&L->attn_gr.inject,"hcai",l);
        q38_trunk_add(&L->mlp_gr.down,"hcmd",l); q38_trunk_add(&L->mlp_gr.up,"hcmu",l);
        q38_trunk_add(&L->mlp_gr.inject,"hcmi",l);
        q38_trunk_add(&L->sh_g,"shg",l); q38_trunk_add(&L->sh_u,"shu",l); q38_trunk_add(&L->sh_d,"shd",l);
        q38_trunk_add(&L->router,"router",l);
    }
    if(m->mtp){
        /* The MTP head's matrices: int8 on the CPU like the rest, but never
         * offered to the placer, which plans the model's layers only, and
         * never placed by a COLI_PLACE spec that names their component. */
        int offer=g_trunk_offer_gpu,first=g_trunk_n; g_trunk_offer_gpu=0;
        int l=c->layers; Layer *L=&m->L[l];
        q38_trunk_add(&L->q,"attnq",l); q38_trunk_add(&L->k,"attnk",l);
        q38_trunk_add(&L->v,"attnv",l); q38_trunk_add(&L->o,"attno",l);
        q38_trunk_add(&L->idx_qk,"qsaidx",l);
        q38_trunk_add(&L->attn_gr.down,"hcad",l); q38_trunk_add(&L->attn_gr.up,"hcau",l);
        q38_trunk_add(&L->attn_gr.inject,"hcai",l);
        q38_trunk_add(&L->mlp_gr.down,"hcmd",l); q38_trunk_add(&L->mlp_gr.up,"hcmu",l);
        q38_trunk_add(&L->mlp_gr.inject,"hcmi",l);
        q38_trunk_add(&L->sh_g,"shg",l); q38_trunk_add(&L->sh_u,"shu",l); q38_trunk_add(&L->sh_d,"shd",l);
        q38_trunk_add(&L->router,"router",l);
        q38_trunk_add(&m->mtp_fc_emb,"mtpfce",l); q38_trunk_add(&m->mtp_fc_hid,"mtpfch",l);
        q38_trunk_add(&m->mtp_mixer.down,"mtpmixd",l); q38_trunk_add(&m->mtp_mixer.up,"mtpmixu",l);
        for(int i=first;i<g_trunk_n;i++)g_trunk[i].cpu_only=1;
        g_trunk_offer_gpu=offer;
    }
}
/* int8 per row, scale = max|w| / 127 (the qwen36 dnproj/lmhead format) */
static void q38_trunk_quantize(const Q38Weight *w,int8_t **qp,float **scp) {
    int O=w->rows,I=w->cols;
    int8_t *q=(int8_t*)malloc((size_t)O*I); float *sc=(float*)malloc((size_t)O*sizeof(float));
    if(!q||!sc){fprintf(stderr,"OOM trunk quantization\n");exit(1);}
    #pragma omp parallel for schedule(static)
    for(int r=0;r<O;r++){
        float row[8192]; float *src=row; float *heap=NULL;
        if(I>8192){heap=(float*)malloc((size_t)I*sizeof(float)); src=heap;}
        q38_weight_row(w,r,src);
        float mx=0.f; for(int k=0;k<I;k++){float a=fabsf(src[k]); if(a>mx)mx=a;}
        float s=mx>0.f?mx/127.f:1.f, inv=1.f/s; sc[r]=s;
        int8_t *dst=q+(size_t)r*I;
        for(int k=0;k<I;k++){int v=(int)lrintf(src[k]*inv); if(v>127)v=127; if(v<-127)v=-127; dst[k]=(int8_t)v;}
        free(heap);
    }
    *qp=q; *scp=sc;
}
/* The trunk on the CPU: int8 rows with one scale per row, and the BF16 copy
 * released. The trunk is read whole on every token (3.6 G weights on the
 * released checkpoint, more than the ten routed experts), so its bytes are
 * the decode's floor: int8 halves them and the integer kernel keeps up with
 * the memory. Default on; Q38_TRUNK_CPU_INT8=0 keeps the BF16 rows and the
 * f32 kernel, the numeric reference. A matrix the tier already quantized for
 * the GPU keeps those same rows here (prefill rows run on the CPU). */
static int q38_trunk_cpu_int8_wanted(void) {
    const char *e=getenv("Q38_TRUNK_CPU_INT8"); return !(e&&e[0]=='0'&&!e[1]);
}
static void q38_trunk_cpu_int8(Model *m) {
    if(!q38_trunk_cpu_int8_wanted())return;
    if(!m->trunk_table_built)q38_trunk_offer_all(m);   /* the tier may have built it already */
    double t0=now_s(); size_t bytes=0,released=0; int n=0;
    for(int i=0;i<g_trunk_n;i++){
        Q38Weight *w=g_trunk[i].w;
        if(!w->q8)q38_trunk_quantize(w,&w->q8,&w->q8sc);
        bytes+=(size_t)w->rows*w->cols+(size_t)w->rows*sizeof(float); n++;
        if(w->owns_data&&w->data){
            /* every path that reads this matrix now goes through q8 */
            uint64_t was=q38_weight_bytes(w);
            free(w->data); w->data=NULL; w->owns_data=0; released+=was;
            m->resident_weight_bytes-=was; m->resident_weight_bytes+=(size_t)w->rows*w->cols+(size_t)w->rows*sizeof(float);
        }
    }
    fprintf(stderr,"[qwen38] trunk: %d matrices int8 on the CPU (%.2f GiB, %.2f GiB of BF16 released) in %.1fs; Q38_TRUNK_CPU_INT8=0 keeps BF16\n",
            n,bytes/1073741824.0,released/1073741824.0,now_s()-t0);
}
#ifdef COLI_VULKAN
/* ---- dense weights on the device only (COLI_VK_DENSE_HOST) ----------------------
 * With the trunk on the device (the chain, or COLI_VK_DENSE), every resident matrix is
 * uploaded at start and its host copy (the int8 rows, or the BF16/F32 rows of a matrix
 * below Q38_TRUNK_MIN_KB) is given back. The CPU then never multiplies by it, except
 * after a lost device: q38_weight_matmul reads it back from the checkpoint first
 * (q38_load_weight, then the int8 rows exactly as q38_trunk_cpu_int8 made them) and the
 * copy stays from there on. What keeps its host copy: the embedding (its rows are
 * gathered on the CPU), the vision tower (CPU only), norms and vectors. */
static Model *g_q38_dho_model;
static pthread_mutex_t g_q38_dho_mx=PTHREAD_MUTEX_INITIALIZER;
static size_t q38_dho_host_bytes(const Q38Weight *w) {
    if(w->q8)return (size_t)w->rows*w->cols+(size_t)w->rows*sizeof(float);
    return w->data?(size_t)q38_weight_bytes(w):0;
}
static void q38_dho_reload(Q38Weight *w) {
    pthread_mutex_lock(&g_q38_dho_mx);
    if(w->vk_gone){
        Model *m=g_q38_dho_model;
        if(!m||!w->vk_name){fprintf(stderr,"[VK] qwen38: a dense matrix the device held alone cannot be read back\n");exit(1);}
        size_t keep=m->resident_weight_bytes;
        Q38Weight nw=q38_load_weight(m,w->vk_name,w->rows,w->cols);
        m->resident_weight_bytes=keep;
        if(w->vk_fmt==1){
            q38_trunk_quantize(&nw,&w->q8,&w->q8sc);
            free(nw.data);
        } else { w->data=nw.data; w->owns_data=1; w->kind=nw.kind; w->elements=nw.elements; }
        free(nw.vk_name);
        w->vk_gone=0;
        coli_vk_dense_host_reloaded(q38_dho_host_bytes(w));
    }
    pthread_mutex_unlock(&g_q38_dho_mx);
}
static void q38_dho_drop(Q38Weight *w,size_t *bytes,int *n) {
    if(!w->vk_res||w->vk_gone||!q38_vk_eligible(w)||!w->vk_name||!q38_vk_tensor(w))return;
    size_t b=q38_dho_host_bytes(w);
    w->vk_fmt=q38_vk_fmt(w);
    if(w->q8){free(w->q8);free(w->q8sc);w->q8=NULL;w->q8sc=NULL;}
    if(w->owns_data&&w->data){free(w->data);w->data=NULL;w->owns_data=0;}
    w->vk_gone=1;
    coli_vk_dense_host_dropped(b);
    *bytes+=b; (*n)++;
}
/* Every resident matrix the device can take, in a fixed order: the head, each layer's
 * (the MTP head's layer included), the PLE projections, the MTP head's own. A partial
 * chain (qwen38_chain.h) bounds it: the first `layers` layers only, and what is not a
 * layer's (the head, the MTP head) only with `head`. */
static void q38_dho_each(Model *m,void (*f)(Q38Weight *,size_t *,int *),size_t *bytes,int *n,int layers,int head) {
    Cfg *c=&m->c;
    if(head){f(&m->lm_head,bytes,n); f(&m->final_gr.down,bytes,n); f(&m->final_gr.up,bytes,n);}
    for(int i=0;i<=c->layers;i++){
        if(i==c->layers&&(!m->mtp||!head))break;
        if(i<c->layers&&i>=layers)continue;
        Layer *l=&m->L[i];
        Q38Weight *ws[]={&l->attn_gr.down,&l->attn_gr.up,&l->attn_gr.inject,&l->mlp_gr.down,&l->mlp_gr.up,
                         &l->mlp_gr.inject,&l->router,&l->sh_g,&l->sh_u,&l->sh_d,&l->q,&l->k,&l->v,&l->o,
                         &l->idx_qk,&l->dn_qkv,&l->dn_z,&l->dn_b,&l->dn_a,&l->dn_out,&l->ple_key,&l->ple_value};
        for(size_t k=0;k<sizeof ws/sizeof ws[0];k++)f(ws[k],bytes,n);
    }
    if(m->mtp&&head){f(&m->mtp_fc_emb,bytes,n);f(&m->mtp_fc_hid,bytes,n);f(&m->mtp_mixer.down,bytes,n);f(&m->mtp_mixer.up,bytes,n);}
}
static void q38_dho_count(Q38Weight *w,size_t *bytes,int *n) {
    if(w->vk_res&&!w->vk_off&&q38_vk_eligible(w)){*bytes+=q38_dho_host_bytes(w);(*n)++;}
}
#endif

/* after qt_init: quantize and upload what the placer accepted */
static void q38_trunk_place_all(Model *m) {
    (void)m;
    int placed=0,offered=g_trunk_n; size_t placed_bytes=0; double t0=now_s();
    for(int i=0;i<g_trunk_n;i++){
        Q38TrunkItem *it=&g_trunk[i]; Q38Weight *w=it->w;
        if(it->cpu_only){offered--;continue;}
        int dev=qt_place_of(it->name,it->layer);
        if(dev==QT_PLACE_CPU)continue;
        int O=w->rows,I=w->cols;
        if(!w->q8)q38_trunk_quantize(w,&w->q8,&w->q8sc);   /* kept: the CPU answers prefill rows from the same bytes */
        const int8_t *q=w->q8; const float *sc=w->q8sc;
        int h=qt_dense_init(q,sc,I,O,dev);
        if(h>=0&&getenv("Q38_TRUNK_SELFTEST")){
            /* DIAG: GPU int8 GEMV against the same int8 matrix on the CPU */
            float *x=(float*)malloc((size_t)I*sizeof(float)),*yg=(float*)malloc((size_t)O*sizeof(float)),*yc=(float*)malloc((size_t)O*sizeof(float));
            for(int k=0;k<I;k++)x[k]=sinf(0.37f*k)+0.1f*(k%7);
            for(int r=0;r<O;r++){double a=0; for(int k=0;k<I;k++)a+=(double)q[(size_t)r*I+k]*x[k]; yc[r]=(float)(a*sc[r]);}
            int ok=qt_dense_matmul(h,yg,x,I,O); double num=0,den=0; int worst=0;
            for(int r=0;r<O;r++){double d=yg[r]-yc[r]; num+=d*d; den+=(double)yc[r]*yc[r]; if(fabs(d)>fabs(yg[worst]-yc[worst]))worst=r;}
            fprintf(stderr,"[selftest] %-7s L%-2d [O=%d I=%d] ok=%d rel.err %.2e worst row %d gpu %.5g cpu %.5g\n",it->name,it->layer,O,I,ok,den>0?sqrt(num/den):-1.0,worst,yg[worst],yc[worst]);
            free(x);free(yg);free(yc);
        }
        if(h>=0){ w->gpu=h+1; placed++; placed_bytes+=(size_t)O*I; }
    }
    if(g_trunk_n)
        fprintf(stderr,"[qtier] qwen38 trunk: %d of %d offered matrices resident as int8 (%.2f GiB) in %.1fs; the rest answers from the CPU\n",
                placed,offered,placed_bytes/1073741824.0,now_s()-t0);
}

/* Start the tier after the model is loaded. COLI_CUDA=1 turns it on (the
 * tier reads that itself); it needs every layer's experts in native FP8 with
 * the block-scale bank resident, because the GPU kernels consume exactly the
 * checkpoint layout (e4m3 bytes + [ceil(O/128), ceil(I/128)] scales). */
static void q38_tier_start(Model *m,int cap) {
    const char *on=getenv("COLI_CUDA");
    if(!on||on[0]!='1'||on[1])return;
    Cfg *c=&m->c;
    if(m->x4){
        /* The tier streams e4m3 bytes (fmt 8) and copies them at qt_note; its
         * int4 mode wants every expert resident in RAM. Neither is this. */
        fprintf(stderr,"[qtier] qwen38: the expert tier streams native FP8 experts only; the int4-g64 "
                       "experts stay on the CPU and the tier stays off (Q38_EXPERT_INT4=0 gives it the FP8 ones)\n");
        return;
    }
    if(!m->native_fp8){
        fprintf(stderr,"[qtier] qwen38: expert tier needs native FP8 experts (Q38_NATIVE_FP8=1); staying on the CPU\n");
        return;
    }
    for(int layer=0;layer<c->layers;layer++)
        if(!q38_prepare_expert_scale_bank(m,layer)){
            fprintf(stderr,"[qtier] qwen38: layer %d experts are not native FP8; staying on the CPU\n",layer);
            return;
        }
    q38_trunk_offer_all(m);                        /* sizes only; the placer decides in qt_init */
    if(qt_init_fp8(c->layers,c->experts,c->hidden,c->inter,cap,c->topk,E4M3_LUT)){
        atexit(qt_shutdown);
        fprintf(stderr,"[qtier] qwen38: fp8 expert tier on (RAM LRU %d/layer stays; experts stream to VRAM as they get hot)\n",cap);
        q38_trunk_place_all(m);
    }
}

/* ---- Vulkan routed-expert tier (vk_tier.c) -------------------------------------
 * The tier computes the resident experts of a step on the device while the CPU
 * computes the rest; every expert's output then joins the row in rank order, the
 * device's and the CPU's alike, so the sum's order is the CPU run's whatever was
 * resident. The model's layers only: the MTP head's layer keeps its FP8 experts on
 * the CPU (the sidecar's int4 covers the model's layers). A slot's bytes are what
 * the tier reads when it promotes the expert: codes and scales of each matrix. */
static VktExpertSrc q38_vk_src(const Slot *ex) {
    VktExpertSrc s={ex->gate.data,ex->up.data,ex->down.data,ex->gate.scales,ex->up.scales,ex->down.scales};
    return s;
}
static void q38_expert_row(Model *m,int layer,int eid,const float *xs,float *eg,float *eu,float *eh,float *eo) {
    Cfg *c=&m->c;int H=c->hidden,I=c->inter;
    Slot *ex=q38_expert_get(m,layer,eid);double started=now_s();
    q38_weight_matmul(eg,xs,&ex->gate,1,H,I);q38_weight_matmul(eu,xs,&ex->up,1,H,I);
    for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_weight_matmul(eo,eh,&ex->down,1,I,H);
    q38_tm_add(m,Q38_TM_ROUTED_EXPERT,started);
}

/* logits_in: the router's raw rows (S x E) when the caller has them, else NULL and the
 * router runs here; routed_only = 1: `out` gets the routed experts' rank-order sum
 * without the shared expert (the Vulkan dense chain runs the router and the shared
 * expert on the device and adds them there). q38_moe passes NULL, 0: unchanged. */
static void q38_moe_decode_ex(Model *m,Layer *l,int layer,const float *x,int S,float *out,
                              const float *logits_in,int routed_only) {
    Cfg *c=&m->c;int H=c->hidden,E=c->experts,K=c->topk,I=c->inter,SI=c->shared_inter;
    float *logits=falloc(E),*sg=falloc(SI),*su=falloc(SI),*sh=falloc(SI),*shared=falloc(H);
    float *eg=falloc(I),*eu=falloc(I),*eh=falloc(I),*eo=falloc(H);
    int vk=vkt_ready()&&layer<c->layers;      /* the Vulkan tier: outputs buffered per rank */
    float *ebuf=vk?falloc((int64_t)K*H):NULL;
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*H;float *ys=out+(int64_t)s*H;memset(ys,0,(size_t)H*sizeof(float));
        if(logits_in)memcpy(logits,logits_in+(int64_t)s*E,(size_t)E*sizeof(float));
        else q38_dense_matmul(m,logits,xs,&l->router,1,H,E);
        float mx=logits[0];for(int e=1;e<E;e++)if(logits[e]>mx)mx=logits[e];
        double all=0;for(int e=0;e<E;e++){logits[e]=expf(logits[e]-mx);all+=logits[e];}
        int idx[Q38_MAX_TOPK];float val[Q38_MAX_TOPK];
        for(int z=0;z<K;z++){int best=-1;float bv=-1.f;for(int e=0;e<E;e++){int used=0;for(int j=0;j<z;j++)if(idx[j]==e)used=1;if(!used&&logits[e]>bv){bv=logits[e];best=e;}}idx[z]=rt_router_pick(best,z,E,layer);val[z]=logits[idx[z]];}
        double top=0;for(int z=0;z<K;z++)top+=val[z];double den=c->norm_topk?top:all;
        float route_gates[Q38_MAX_TOPK];
        for(int z=0;z<K;z++) route_gates[z]=(float)(val[z]/den);
        rt_route(layer,s,idx,route_gates,K); /* shared counts + post-normalization trace */
        /* Resident experts run on the GPU from here on (asynchronously); the
         * CPU only loads and computes the rest, in route order. The tier
         * knows the model's layers only: the MTP head's stays on the CPU. */
        int tiered=layer<c->layers;
        uint8_t taken[Q38_MAX_TOPK]={0};
        int ndev=vk?vkt_issue(layer,xs,1,K,idx,taken):0;
        for(int z=0;z<K;z++)if(taken[z])q38_ehit_mark(m,layer,idx[z]);   /* routed this turn, wherever it ran */
        uint32_t qmask=tiered?qt_issue(layer,idx,K,xs):0;
        int cpu_idx[Q38_MAX_TOPK],cpu_rank[Q38_MAX_TOPK],cpu_n=0;
        for(int z=0;z<K;z++)if(!((qmask>>z)&1u)&&!taken[z]){cpu_idx[cpu_n]=idx[z];cpu_rank[cpu_n]=z;cpu_n++;}
        q38_prefetch_experts(m,layer,cpu_idx,cpu_n);
        double phase_started=now_s();
        float gate=0.f;
        if(!routed_only){
        q38_weight_matmul(sg,xs,&l->sh_g,1,H,SI);q38_weight_matmul(su,xs,&l->sh_u,1,H,SI);
        for(int j=0;j<SI;j++)sh[j]=q38_silu(sg[j])*su[j];q38_weight_matmul(shared,sh,&l->sh_d,1,SI,H);
        for(int d=0;d<H;d++)gate+=xs[d]*l->sh_gate[d];gate=q38_sigmoid(gate);
        q38_tm_add(m,Q38_TM_SHARED_EXPERT,phase_started);
        }
        Slot *selected[Q38_MAX_TOPK];
        int loaded_batch=q38_expert_get_batch(m,layer,cpu_idx,cpu_n,selected);
        for(int i=0;i<cpu_n;i++){
            int z=cpu_rank[i];
            Slot *ex=loaded_batch?selected[i]:q38_expert_get(m,layer,cpu_idx[i]);phase_started=now_s();
            q38_weight_matmul(eg,xs,&ex->gate,1,H,I);q38_weight_matmul(eu,xs,&ex->up,1,H,I);
            for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_weight_matmul(eo,eh,&ex->down,1,I,H);
            if(vk){
                memcpy(ebuf+(int64_t)z*H,eo,(size_t)H*sizeof(float));
                VktExpertSrc src=q38_vk_src(ex); vkt_note(layer,cpu_idx[i],&src);
            }
            else for(int d=0;d<H;d++)ys[d]+=route_gates[z]*eo[d];
            q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
            if(tiered)q38_tier_note(layer,cpu_idx[i],ex);
        }
        if(vk){
            /* the device's experts, then every rank in order; a batch that failed is
             * recomputed here (the tier has turned itself off) */
            const float *dev[Q38_MAX_TOPK];
            if(ndev&&!vkt_join(dev))
                for(int z=0;z<K;z++)if(taken[z]){
                    q38_expert_row(m,layer,idx[z],xs,eg,eu,eh,ebuf+(int64_t)z*H);taken[z]=0;
                }
            for(int z=0;z<K;z++){
                const float *e=taken[z]?dev[z]:ebuf+(int64_t)z*H;
                for(int d=0;d<H;d++)ys[d]+=route_gates[z]*e[d];
            }
        }
        /* GPU experts land after the CPU ones: same values, one more group in
         * the float sum (that is the only ordering difference to a CPU run). */
        if(tiered&&!qt_take(qmask,route_gates,K,ys)){
            fprintf(stderr,"qwen38: CUDA expert collection failed at layer %d; stopping inference\n",layer);
            exit(1);
        }
        if(!routed_only)for(int d=0;d<H;d++)ys[d]+=gate*shared[d];
    }
    rt_trace_end();
    free(logits);free(sg);free(su);free(sh);free(shared);free(eg);free(eu);free(eh);free(eo);free(ebuf);
}
static void q38_moe_decode(Model *m,Layer *l,int layer,const float *x,int S,float *out) {
    q38_moe_decode_ex(m,l,layer,x,S,out,NULL,0);
}

typedef struct {
    int expert;
    float gate;
} Q38RouteAssignment;

/* Pick a prefill size from a fixed workspace budget.  The number of rows is
 * deliberately bounded independently of the context length: a long prompt
 * therefore reuses the same route, expert and matmul buffers one chunk at a
 * time.  A single assignment needs input, gate/up activations and output;
 * these are the only buffers that scale with the number of routed rows. */
static int q38_moe_prefill_rows(const Cfg *c,int requested) {
    int64_t H=c->hidden,I=c->inter,E=c->experts,K=c->topk,SI=c->shared_inter;
    uint64_t per_assignment=(uint64_t)(2LL*((int64_t)H+I))*sizeof(float)+
                             sizeof(Q38RouteAssignment)+2*sizeof(int);
    uint64_t per_row=(uint64_t)E*sizeof(float)+
                     (uint64_t)(3LL*(int64_t)SI+H+1)*sizeof(float);
    uint64_t fixed=(uint64_t)E*(4*sizeof(int)+2*sizeof(Slot*));
    if(fixed<=UINT64_MAX-sizeof(int))fixed+=sizeof(int); /* group_offsets[E] */
    else fixed=UINT64_MAX;
    uint64_t assignment_row=(uint64_t)K>UINT64_MAX/per_assignment?
                            UINT64_MAX:(uint64_t)K*per_assignment;
    uint64_t total_row=per_row>UINT64_MAX-assignment_row?
                       UINT64_MAX:per_row+assignment_row;
    return q38_bounded_prefill_rows(requested,fixed,total_row);
}

/* With the Vulkan tier: n of an expert's assignments (positions pos[] in the
 * grouped order, assignment ids in assign[]) gathered, run as one batch and their
 * outputs scattered to their places in routed_out, where the reduction reads them.
 * Row for row the same arithmetic as the grouped loop below. */
static void q38_prefill_expert_rows(Model *m,Slot *expert,const float *xc,int K,const int *assign,
                                    const int *pos,int n,float *in,float *gate,float *up,float *routed_out) {
    if(n<1)return;
    Cfg *c=&m->c;int H=c->hidden,I=c->inter;
    for(int a=0;a<n;a++)memcpy(in+(int64_t)a*H,xc+(int64_t)(assign[pos[a]]/K)*H,(size_t)H*sizeof(float));
    double started=now_s();
    q38_weight_matmul(gate,in,&expert->gate,n,H,I);
    q38_weight_matmul(up,in,&expert->up,n,H,I);
    for(int a=0;a<n;a++)for(int j=0;j<I;j++)
        gate[(int64_t)a*I+j]=q38_silu(gate[(int64_t)a*I+j])*up[(int64_t)a*I+j];
    float *o=in;   /* the inputs are consumed: their rows hold the outputs, H wide each */
    q38_weight_matmul(o,gate,&expert->down,n,I,H);
    for(int a=0;a<n;a++)memcpy(routed_out+(int64_t)pos[a]*H,o+(int64_t)a*H,(size_t)H*sizeof(float));
    q38_tm_add(m,Q38_TM_ROUTED_EXPERT,started);
}

/* Prefill MoE: route a bounded row chunk first, then execute each distinct
 * expert's assignments as one batched SwiGLU.  Grouping is an I/O optimization
 * only.  Expert outputs are placed back in assignment order and the final
 * weighted reduction still visits rank 0..top-k-1 for every row, preserving the
 * decode path's floating-point accumulation order. */
static void q38_moe_prefill(Model *m,Layer *l,int layer,const float *x,
                            int S,float *out,const float *logits_in,int routed_only) {
    Cfg *c=&m->c;
    int H=c->hidden,E=c->experts,K=c->topk,I=c->inter,SI=c->shared_inter;
    int rows_capacity=q38_moe_prefill_rows(c,S);
    /* the Vulkan tier streams a prompt chunk's cold experts (vk_tier.c): it takes the
     * whole chunk in one step, so each expert's rows meet in one GEMM */
    if(vkt_ready()&&layer<c->layers)rows_capacity=vkt_step_rows(S,rows_capacity);
    int64_t max_assign=(int64_t)rows_capacity*K;

    Q38RouteAssignment *routes=(Q38RouteAssignment*)malloc(
        (size_t)max_assign*sizeof(*routes));
    int *assignments=(int*)malloc((size_t)max_assign*sizeof(*assignments));
    int *assignment_positions=(int*)malloc((size_t)max_assign*sizeof(*assignment_positions));
    int *group_counts=(int*)calloc((size_t)E,sizeof(*group_counts));
    int *group_offsets=(int*)malloc((size_t)(E+1)*sizeof(*group_offsets));
    int *group_cursor=(int*)malloc((size_t)E*sizeof(*group_cursor));
    int *unique=(int*)malloc((size_t)E*sizeof(*unique));
    Slot **batch_slots=(Slot**)calloc((size_t)E,sizeof(*batch_slots));
    if(!routes||!assignments||!assignment_positions||!group_counts||
       !group_offsets||!group_cursor||!unique||!batch_slots){
        fprintf(stderr,"OOM Qwen3.8 MoE prefill metadata\n");exit(1);
    }

    /* the Vulkan tier: per assignment, whether the device takes it and its output */
    int vk=vkt_ready()&&layer<c->layers;
    int *vk_idx=vk?(int*)malloc((size_t)max_assign*sizeof(int)):NULL;
    uint8_t *vk_taken=vk?(uint8_t*)calloc((size_t)max_assign,1):NULL;
    const float **vk_dev=vk?(const float**)malloc((size_t)max_assign*sizeof(*vk_dev)):NULL;
    int *vk_cpu=vk?(int*)malloc((size_t)E*sizeof(int)):NULL;
    int *vk_sub=vk?(int*)malloc((size_t)max_assign*sizeof(int)):NULL;
    if(vk&&(!vk_idx||!vk_taken||!vk_dev||!vk_cpu||!vk_sub)){fprintf(stderr,"OOM Qwen3.8 MoE prefill metadata\n");exit(1);}

    float *logits=falloc((int64_t)rows_capacity*E);
    float *shared_gate=falloc(rows_capacity);
    float *shared_g=falloc((int64_t)rows_capacity*SI);
    float *shared_u=falloc((int64_t)rows_capacity*SI);
    float *shared_hidden=falloc((int64_t)rows_capacity*SI);
    float *shared_out=falloc((int64_t)rows_capacity*H);
    float *expert_input=falloc(max_assign*H);
    float *expert_gate=falloc(max_assign*I);
    float *expert_up=falloc(max_assign*I);
    float *routed_out=falloc(max_assign*H);

    for(int base=0;base<S;) {
        int rows=S-base<rows_capacity?S-base:rows_capacity;
        int assignment_count=rows*K;
        memset(group_counts,0,(size_t)E*sizeof(*group_counts));

        /* Route the complete chunk with one resident router matmul.  Selection
         * intentionally mirrors q38_moe_decode, including rt_router_pick's
         * deterministic fallback for invalid logits. */
        if(logits_in)memcpy(logits,logits_in+(int64_t)base*E,(size_t)rows*E*sizeof(float));
        else q38_dense_matmul(m,logits,x+(int64_t)base*H,&l->router,rows,H,E);
        for(int s=0;s<rows;s++) {
            float *probabilities=logits+(int64_t)s*E;
            float maximum=probabilities[0];
            for(int e=1;e<E;e++)if(probabilities[e]>maximum)maximum=probabilities[e];
            double total=0.0;
            for(int e=0;e<E;e++) {
                probabilities[e]=expf(probabilities[e]-maximum);
                total+=probabilities[e];
            }
            int selected[Q38_MAX_TOPK];
            float selected_probability[Q38_MAX_TOPK];
            for(int rank=0;rank<K;rank++) {
                int best=-1;float best_value=-1.f;
                for(int e=0;e<E;e++) {
                    int used=0;
                    for(int previous=0;previous<rank;previous++)
                        if(selected[previous]==e){used=1;break;}
                    if(!used&&probabilities[e]>best_value){
                        best_value=probabilities[e];best=e;
                    }
                }
                selected[rank]=rt_router_pick(best,rank,E,layer);
                selected_probability[rank]=probabilities[selected[rank]];
            }
            double top=0.0;
            for(int rank=0;rank<K;rank++)top+=selected_probability[rank];
            double denominator=c->norm_topk?top:total;
            float gates[Q38_MAX_TOPK];
            for(int rank=0;rank<K;rank++) {
                gates[rank]=(float)(selected_probability[rank]/denominator);
                int assignment=s*K+rank;
                routes[assignment]=(Q38RouteAssignment){selected[rank],gates[rank]};
                group_counts[selected[rank]]++;
            }
            rt_route(layer,base+s,selected,gates,K);
        }

        /* Prefixing by expert makes every group contiguous while the inverse
         * map lets the final reduction recover the original row/rank order. */
        group_offsets[0]=0;
        int unique_count=0;
        for(int e=0;e<E;e++) {
            group_offsets[e+1]=group_offsets[e]+group_counts[e];
            if(group_counts[e])unique[unique_count++]=e;
        }
        memset(group_cursor,0,(size_t)E*sizeof(*group_cursor));
        for(int assignment=0;assignment<assignment_count;assignment++) {
            int expert=routes[assignment].expert;
            int position=group_offsets[expert]+group_cursor[expert]++;
            assignments[position]=assignment;
            assignment_positions[assignment]=position;
        }

        /* The Vulkan tier takes the chunk's assignments of its resident experts now,
         * and computes them while the shared expert and the other experts run here.
         * cpu_unique: the experts with an assignment the device did not take (all of
         * them without the tier). */
        int ndev=0;
        int *cpu_unique=unique,cpu_count=unique_count;
        if(vk){
            for(int a=0;a<assignment_count;a++)vk_idx[a]=routes[a].expert;
            ndev=vkt_issue(layer,x+(int64_t)base*H,rows,K,vk_idx,vk_taken);
            for(int a=0;a<assignment_count;a++)if(vk_taken[a])q38_ehit_mark(m,layer,vk_idx[a]);
            cpu_count=0;
            for(int u=0;u<unique_count;u++){
                int e=unique[u],left=0;
                for(int a=0;a<group_counts[e];a++)left+=!vk_taken[assignments[group_offsets[e]+a]];
                if(left)vk_cpu[cpu_count++]=e;
            }
            cpu_unique=vk_cpu;
        }

        /* One advice range per distinct expert is enough for this chunk. */
        q38_prefetch_experts(m,layer,cpu_unique,cpu_count);

        /* Shared expert work is independent across rows and remains resident;
         * batching it here also keeps its cost out of the routed groups. */
        double phase_started=now_s();
        if(!routed_only){
        q38_weight_matmul(shared_g,x+(int64_t)base*H,&l->sh_g,rows,H,SI);
        q38_weight_matmul(shared_u,x+(int64_t)base*H,&l->sh_u,rows,H,SI);
        for(int s=0;s<rows;s++)
            for(int j=0;j<SI;j++)
                shared_hidden[(int64_t)s*SI+j]=
                    q38_silu(shared_g[(int64_t)s*SI+j])*
                    shared_u[(int64_t)s*SI+j];
        q38_weight_matmul(shared_out,shared_hidden,&l->sh_d,rows,SI,H);
        for(int s=0;s<rows;s++) {
            const float *xs=x+(int64_t)(base+s)*H;
            float gate=0.f;
            for(int d=0;d<H;d++)gate+=xs[d]*l->sh_gate[d];
            shared_gate[s]=q38_sigmoid(gate);
        }
        q38_tm_add(m,Q38_TM_SHARED_EXPERT,phase_started);
        }

        /* A demand-set batch is particularly effective for native FP8: reserve
         * all slots before workers read the two coalesced ranges.  Other native
         * layouts use the same grouped matmul loop with the ordinary bounded
         * LRU loader, so no format loses correctness or caching. */
        memset(out+(int64_t)base*H,0,(size_t)rows*H*sizeof(float));
        /* A prompt chunk can route to more distinct experts than the retained
         * cache can hold.  Load and consume cache-sized groups instead of
         * declining the complete parallel demand set: every expert is still
         * loaded once for this chunk, and a later group may safely reuse its
         * slots because the preceding outputs already live in routed_out. */
        int load_limit=m->cache[layer].cap;
        if(load_limit<1)load_limit=1;
        for(int unique_base=0;unique_base<cpu_count;) {
            int load_count=cpu_count-unique_base;
            if(load_count>load_limit)load_count=load_limit;
            int loaded_batch=load_count>=2&&q38_expert_get_batch(
                m,layer,cpu_unique+unique_base,load_count,batch_slots);
            for(int offset=0;offset<load_count;offset++) {
                int e=cpu_unique[unique_base+offset];
                int count=group_counts[e];
                Slot *expert=loaded_batch?batch_slots[offset]:
                                          q38_expert_get(m,layer,e);
                int first=group_offsets[e];
                if(vk){
                    /* the assignments the device left (all of them, unless its
                     * batch filled up mid-expert), computed and scattered */
                    int n=0;
                    for(int a=0;a<count;a++)if(!vk_taken[assignments[first+a]])vk_sub[n++]=first+a;
                    q38_prefill_expert_rows(m,expert,x+(int64_t)base*H,K,assignments,vk_sub,n,
                                            expert_input,expert_gate,expert_up,routed_out);
                    VktExpertSrc src=q38_vk_src(expert); vkt_note(layer,e,&src);
                    continue;
                }
                for(int a=0;a<count;a++) {
                    int assignment=assignments[first+a];
                    int row=assignment/K;
                    memcpy(expert_input+(int64_t)a*H,
                           x+(int64_t)(base+row)*H,(size_t)H*sizeof(float));
                }
                phase_started=now_s();
                q38_weight_matmul(expert_gate,expert_input,&expert->gate,
                                  count,H,I);
                q38_weight_matmul(expert_up,expert_input,&expert->up,count,H,I);
                for(int a=0;a<count;a++)for(int j=0;j<I;j++)
                    expert_gate[(int64_t)a*I+j]=
                        q38_silu(expert_gate[(int64_t)a*I+j])*
                        expert_up[(int64_t)a*I+j];
                q38_weight_matmul(routed_out+(int64_t)first*H,expert_gate,
                                  &expert->down,count,I,H);
                q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
            }
            unique_base+=load_count;
        }

        if(ndev&&!vkt_join(vk_dev)){
            /* the batch failed (the tier has turned itself off): its experts here */
            for(int u=0;u<unique_count;u++){
                int e=unique[u],first=group_offsets[e],n=0;
                for(int a=0;a<group_counts[e];a++)if(vk_taken[assignments[first+a]])vk_sub[n++]=first+a;
                if(!n)continue;
                q38_prefill_expert_rows(m,q38_expert_get(m,layer,e),x+(int64_t)base*H,K,assignments,vk_sub,n,
                                        expert_input,expert_gate,expert_up,routed_out);
                for(int a=0;a<n;a++)vk_taken[assignments[vk_sub[a]]]=0;
            }
        }

        /* The grouped execution above is intentionally not the reduction
         * order.  Replaying the original top-k sequence gives the same result
         * as the single-row implementation for F32, BF16 and block-FP8. */
        for(int s=0;s<rows;s++) {
            float *ys=out+(int64_t)(base+s)*H;
            for(int rank=0;rank<K;rank++) {
                int assignment=s*K+rank;
                const float *expert_output=vk&&vk_taken[assignment]?vk_dev[assignment]:routed_out+
                    (int64_t)assignment_positions[assignment]*H;
                float gate=routes[assignment].gate;
                for(int d=0;d<H;d++)ys[d]+=gate*expert_output[d];
            }
            const float *shared=shared_out+(int64_t)s*H;
            if(!routed_only)for(int d=0;d<H;d++)ys[d]+=shared_gate[s]*shared[d];
        }
        base+=rows;
    }
    rt_trace_end();
    free(logits);free(shared_gate);free(shared_g);free(shared_u);
    free(shared_hidden);free(shared_out);free(expert_input);free(expert_gate);
    free(expert_up);free(routed_out);free(routes);free(assignments);
    free(assignment_positions);free(group_counts);free(group_offsets);
    free(group_cursor);free(unique);free(batch_slots);
    free(vk_idx);free(vk_taken);free(vk_dev);free(vk_cpu);free(vk_sub);
}

static void q38_moe_ex(Model *m,Layer *l,int layer,const float *x,int S,float *out,
                       const float *logits_in,int routed_only) {
    /* An MTP verify on the CUDA tier routes row by row as decode does, so
     * the tier's resident experts serve it (the batched path is CPU only). */
    if(S<=1||!m->prefill_batch||(g_q38_rowwise&&qt_ready()&&layer<m->c.layers))
        q38_moe_decode_ex(m,l,layer,x,S,out,logits_in,routed_only);
    else q38_moe_prefill(m,l,layer,x,S,out,logits_in,routed_only);
}
static void q38_moe(Model *m,Layer *l,int layer,const float *x,int S,float *out) {
    q38_moe_ex(m,l,layer,x,S,out,NULL,0);
}
#ifdef COLI_VULKAN
#include "qwen38_chain.h"  /* COLI_VK_CHAIN: every layer's dense chain on the device */
#endif

static void reset_recurrent(Model *m) {
    kv_prefix_clear(&m->kvp);
    Cfg *c=&m->c;
    for(int i=0;i<c->layers;i++)if(!c->is_attn[i]){
        memset(m->DN_rec[i],0,(size_t)c->dn_vheads*c->dn_kdim*c->dn_vdim*sizeof(float));
        memset(m->DN_conv[i],0,(size_t)c->dn_conv_dim*(c->dn_convk-1)*sizeof(float));
    }
    memset(m->PLE_conv_state,0,(size_t)c->hc_width*(c->ple_convk-1)*c->ngram_size*sizeof(float));m->ple_history_len=0;
    m->mtp_len=0;m->mtp_pend_n=0;   /* the MTP head's rows go with the model's */
#ifdef COLI_VULKAN
    q38c_host_wrote(m,1);           /* zeros: the dense chain fills its copy with zeros */
#endif
}

static void ensure_kv(Model *m) {
    Cfg *c=&m->c;if(m->max_t<=m->kv_cap&&m->K)return;
    int rows=q38_layer_rows(m);
    if(m->K){for(int i=0;i<rows;i++){free(m->K[i]);free(m->V[i]);free(m->IK[i]);}free(m->K);free(m->V);free(m->IK);}
    /* +1: the MTP head's attention rows at index c.layers */
    m->K=(float**)calloc((size_t)c->layers+1,sizeof(float*));m->V=(float**)calloc((size_t)c->layers+1,sizeof(float*));m->IK=(float**)calloc((size_t)c->layers+1,sizeof(float*));
    for(int i=0;i<rows;i++)if(i==c->layers||c->is_attn[i]){
        m->K[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->V[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->IK[i]=falloc((int64_t)m->max_t*c->idx_dim);
    }
    m->kv_cap=m->max_t;
    kv_prefix_alloc(&m->kvp,m->kv_cap); /* ensure_kv discards the old rows */
    m->mtp_len=0;m->mtp_pend_n=0;
}

/* One decoder layer over the four hyper-connection streams: PLE where the
 * model has it, the token mixer, then the MoE, each read from the streams
 * and written back through its GatedResidual. step(), the Segment range and
 * the MTP head (index c.layers, an attention layer) all run this. */
static void q38_layer_forward(Model *m,int i,float *hyper,const int *ids,int S,int pos_base,
                              float *mixed,float *inject,float *block) {
    Cfg *c=&m->c; Layer *l=&m->L[i]; int W=c->hc_width;
    if(i==c->ple_layer){
        float *ple=falloc((int64_t)S*W); q38_ple(m,ids,S,hyper,ple);
        for(int64_t z=0;z<(int64_t)S*W;z++) hyper[z]+=ple[z];
        free(ple);
        /* consumate: il chunk successivo ha altri token, e riusare queste
         * righe darebbe gli embedding del chunk precedente combaciando in
         * silenzio invece di dare errore. */
        free(m->ple_pref); m->ple_pref=NULL; m->ple_pref_rows=0;
    }
    q38_gr_read(m,&l->attn_gr,hyper,S,mixed,inject);
    if(i>=c->layers||c->is_attn[i]) q38_attention(m,l,i,mixed,S,pos_base,block);
    else q38_deltanet(m,l,i,mixed,S,block);
    q38_gr_apply(c,hyper,block,inject,S);
    q38_gr_read(m,&l->mlp_gr,hyper,S,mixed,inject);
    q38_moe(m,l,i,mixed,S,block);
    q38_gr_apply(c,hyper,block,inject,S);
}

/* Run only the requested native layer interval over hyper-residual activations.
 * Segment callers supply the four-stream boundary state directly; unlike step,
 * this path deliberately does not gather embeddings or apply the final mixer. */
static void q38_layers_forward_range(Model *m,float *hyper,const int *ids,
                                     int S,int pos_base,int layer_begin,
                                     int layer_end) {
    Cfg *c=&m->c; int H=c->hidden,C=c->hc_count;
    float *mixed=falloc((int64_t)S*H),*inject=falloc((int64_t)S*C),*block=falloc((int64_t)S*H);
    for(int i=layer_begin;i<layer_end;i++)
        q38_layer_forward(m,i,hyper,ids,S,pos_base,mixed,inject,block);
    free(mixed); free(inject); free(block);
}

/* Canale logprobs (modalita jev). Qui la fotografia dello stato NON si
 * aggiunge: questo motore ce l'ha gia, si chiama Q38PrefixCache e salva stato
 * ricorrente, id e logit finali dopo ogni prompt. Manca solo la lettura, cioe
 * un passaggio di lm_head per posizione invece che solo sull'ultima, e il
 * predittore del primo token fresco, che sono i logit gia salvati. */
static int    g_echo_k = 0;
static const char *g_echo_id = NULL;
static const float *g_echo_pin_logit = NULL;   /* logit dell'ultima posizione riusata */

static void q38_echo(const char *id, int pos, int token, const float *lo, int V, int k){
    char tail[1024]; coli_logprob_tail(tail, sizeof tail, lo, V, token, k);
    unsigned char *piece=NULL; int n=0;
    if(decode_id_alloc(token,&piece,&n)) { piece=NULL; n=0; }
    if(n<0) n=0;
    printf("ECHO %s %d %d%s\n", id, n, pos, tail);
    if(n>0) fwrite(piece,1,(size_t)n,stdout);
    fputc('\n', stdout); fflush(stdout);
    free(piece);
}

/* The input row of the token at absolute position abs_pos: its embedding,
 * or the vision tower's row when an image covers that position. */
static void q38_embed_row(Model *m,int id,int abs_pos,float *out) {
    int vis_row=-1;
    if(m->vis_map&&abs_pos>=0&&abs_pos<m->vis_map_len) vis_row=m->vis_map[abs_pos];
    if(vis_row>=0&&vis_row<m->vis_rows_n)
        memcpy(out,m->vis_rows+(int64_t)vis_row*m->c.hidden,(size_t)m->c.hidden*sizeof(float));
    else q38_weight_row(&m->embed,id,out);
}

/* The forward behind step(): ids[0..S) at pos_base through every layer, the
 * final mixer and lm_head. The logits of the last `nlogits` rows come back
 * (step() asks for one, an MTP verify for two), and with `streams` every
 * row's four streams as the final mixer read them are handed over instead of
 * freed: that is the state the MTP head reads. */
static float *q38_forward(Model *m,const int *ids,int S,int pos_base,int nlogits,float **streams) {
    Cfg *c=&m->c;int H=c->hidden,W=c->hc_width,C=c->hc_count;
    m->timers.forwards++;
    float *hyper=falloc((int64_t)S*W);
    /* Le righe PLE partono ADESSO, non al layer 2 dove servono: sono note dagli
     * id dei token soltanto, e i due layer che le precedono danno il tempo di
     * farle arrivare dal disco. */
    q38_ple_prefetch(m,ids,S);
    for(int s=0;s<S;s++){
        if(ids[s]<0||ids[s]>=c->vocab){fprintf(stderr,"token id %d outside vocabulary\n",ids[s]);exit(1);}
        float *e=hyper+(int64_t)s*W;
        q38_embed_row(m,ids[s],pos_base+s,e);
        for(int b=1;b<C;b++)memcpy(e+(int64_t)b*H,e,(size_t)H*sizeof(float));
    }
    float *mixed=falloc((int64_t)S*H),*inject=falloc((int64_t)S*C),*block=falloc((int64_t)S*H);
    int first=0;   /* the layers a partial dense chain ran on the device (COLI_VULKAN) */
#ifdef COLI_VULKAN
    /* COLI_VK_CHAIN: the layers, the final mixer and lm_head on the device; the streams
     * come back for the MTP head, every row's mixed for the prefill read-out. A partial
     * chain runs its first N layers there and hands the streams back after layer N-1:
     * the CPU runs the other layers and the head from them, below. */
    float *chain_logit=NULL;
    if(g_vk_chain){
        int echo=g_echo_k>0&&g_echo_id&&S>1;
        chain_logit=falloc((int64_t)nlogits*c->vocab);
        int took=q38c_forward(m,ids,S,pos_base,nlogits,hyper,streams!=NULL,echo?mixed:NULL,chain_logit);
        if(took==2){   /* the first N layers: the PLE rows were the chain's when its layer is one of them */
            free(chain_logit); chain_logit=NULL;
            first=q38c_layers(m);
            if(c->ple_layer<first){free(m->ple_pref); m->ple_pref=NULL; m->ple_pref_rows=0;}
        } else if(took){
            free(m->ple_pref); m->ple_pref=NULL; m->ple_pref_rows=0;   /* consumed, as q38_layer_forward does */
        } else {
            free(chain_logit); chain_logit=NULL;
            q38c_cpu_step(m,pos_base);
            /* A completed chunk may already have returned its final streams in
             * hyper. Replay the whole forward from embeddings after a later
             * chunk fails, including rows the device had finished. */
            for(int s=0;s<S;s++){
                float *e=hyper+(int64_t)s*W;
                q38_embed_row(m,ids[s],pos_base+s,e);
                for(int b=1;b<C;b++)memcpy(e+(int64_t)b*H,e,(size_t)H*sizeof(float));
            }
        }
    }
    if(!chain_logit){
#endif
    for(int i=first;i<c->layers;i++)
        q38_layer_forward(m,i,hyper,ids,S,pos_base,mixed,inject,block);
    q38_gr_read(m,&m->final_gr,hyper,S,mixed,NULL);
#ifdef COLI_VULKAN
    }
#endif
    m->kv_len=pos_base+S;
    /* Rewinding and writing a shorter branch invalidates its old tail. */
    if(m->kvp.len>pos_base)m->kvp.len=pos_base;
    kv_prefix_record(&m->kvp,ids,pos_base,S);
    if(m->vis_map && m->vis_rows_n>0)kv_prefix_taint(&m->kvp);
    float *logit;
#ifdef COLI_VULKAN
    if(chain_logit)logit=chain_logit; else
#endif
    logit=falloc((int64_t)nlogits*c->vocab);
    double phase_started=now_s();
    /* Lettura del prefill: la posizione p predice il token p+1. Il primo token
     * fresco lo predice la fotografia del prefisso, quando c'e. Pagata solo da
     * chi ha chiesto il canale. */
    if(g_echo_k>0&&g_echo_id&&S>0){
        if(g_echo_pin_logit)
            q38_echo(g_echo_id,pos_base,ids[0],g_echo_pin_logit,c->vocab,g_echo_k);
        float *elog=logit;
#ifdef COLI_VULKAN
        if(chain_logit)elog=falloc(c->vocab);   /* logit already holds the chain's last rows */
#endif
        for(int p=0;p+1<S;p++){
            q38_weight_matmul(elog,mixed+(int64_t)p*H,&m->lm_head,1,H,c->vocab);
            q38_echo(g_echo_id,pos_base+p+1,ids[p+1],elog,c->vocab,g_echo_k);
        }
        if(elog!=logit)free(elog);
    }
#ifdef COLI_VULKAN
    if(!chain_logit)
#endif
    q38_weight_matmul(logit,mixed+(int64_t)(S-nlogits)*H,&m->lm_head,nlogits,H,c->vocab);
    q38_tm_add(m,Q38_TM_LM_HEAD,phase_started);
    if(streams)*streams=hyper; else free(hyper);
    free(mixed);free(inject);free(block);return logit;
}

/* ---- the MTP head (Q38_MTP=1) ---------------------------------------------
 * The checkpoint carries one more decoder layer under mtp.*, trained to read
 * the model's state at position p together with the token at p+1 and to say
 * what comes at p+2. Everything it computes is a block this engine already
 * runs for the model's own layers and checks against the oracle: the decoder
 * layer is q38_layer_forward at index c.layers (an attention layer:
 * q38_attention with its own K/V/indexer rows and RoPE base, then q38_moe
 * with its own router, shared expert and routed experts), its four streams
 * leave through a GatedResidual of its own, mtp.hyper_connection_mixer, read
 * by q38_gr_read exactly as q38_forward reads final_gr, and the shared
 * lm_head gives the logits. The token at p+1 enters through the model's
 * embedding (q38_embed_row); the head has none of its own.
 *
 * What the tensors leave open is how the two inputs become the four streams
 * that layer reads. The model's streams at p (the 4 x hidden that the final
 * mixer reads) meet pre_fc_norm_hidden [4*hidden] and fc_hidden
 * [hidden, hidden]; the token's embedding meets pre_fc_norm_embedding
 * [hidden] and fc_embedding [hidden, hidden]. Two square projections add (a
 * single fc over the concatenation [e; h] is the same sum split in two), and
 * the sum is a stream: fc_hidden maps each of the four streams and
 * fc_embedding(norm(e)) is added to all four, as q38_forward puts a token's
 * embedding in all four streams. What the shapes do not say is how
 * pre_fc_norm_hidden [4*hidden] groups its input; Q38_MTP_WIRING picks:
 *   b  (default) a norm per stream: q38_rms0 over each hidden-wide stream
 *      with its slice of the weight, the grouping hc_norm has in
 *      q38_gr_read;
 *   a  one norm over the whole 4*hidden vector, the way
 *      pre_fc_norm_embedding normalizes its input.
 * The output is the same whichever is chosen (the verify decides every
 * token); only how often a draft is right differs, and only the trained
 * weights can say which reading they were trained with. Measured on the
 * release (docs/qwen38.md), b accepts 94-96% of the drafts and a 85-92%.
 * A third reading, the normalized streams averaged into one vector before a
 * single fc_hidden, accepted 63-78% and is gone. Each norm is zero-centered
 * (q38_rms0) like every Qwen4-Exp text RMSNorm but DeltaNet's gated one;
 * q38_mtp_attach prints the mean of each norm weight, which on the release
 * are -0.76, -0.33 and 3.79 (embedding, hidden, mixer): far from the 1 a
 * plain norm's weights start from. The pair (streams at p, token at
 * p+1) is the head's row p: its K/V row and its RoPE position are p.
 * Counting it at p+1 instead would change no score beyond rounding, since
 * RoPE and the indexer's block scores depend on distances only.
 *
 * The head's KV rows hold verified pairs only, so it reads a row of the
 * model's streams once the token after that row is known: the last row of a
 * forward waits in mtp_pend, a draft runs the pending rows with the token
 * the caller just picked, and the rows of a verify wait until the caller's
 * next pick settles which of them stand. */
enum { Q38_MTP_WHOLE_NORM='a', Q38_MTP_STREAM_NORM='b' };

/* Q38_MTP_FORCE (tests): 'r' rejects every draft, a right one too, so the
 * rollback runs on every token; 'a' drafts the reference's next token
 * (ref.json, g_q38_mtp_oracle) so every draft is accepted; 'm' alternates a
 * reference draft with a wrong one. The head still runs in every mode. */
static const int *g_q38_mtp_oracle; static int g_q38_mtp_oracle_n;
static FILE *g_q38_mtp_dump;   /* Q38_MTP_DUMP=<file>: per draft, the head's row, its token and its logits (tests) */

/* The four streams the head's layer starts from, for S pairs: row s pairs
 * the model's streams at pos_base+s with the token next[s] at pos_base+s+1. */
static void q38_mtp_input(Model *m,const float *streams,const int *next,int S,int pos_base,float *hyper) {
    Cfg *c=&m->c; int H=c->hidden,W=c->hc_width,C=c->hc_count;
    float *en=falloc((int64_t)S*H),*fe=falloc((int64_t)S*H),*row=falloc(H),*hn=falloc((int64_t)S*W);
    for(int s=0;s<S;s++){
        q38_embed_row(m,next[s],pos_base+s+1,row);
        q38_rms0(en+(int64_t)s*H,row,m->mtp_norm_emb,H,c->eps);
    }
    q38_dense_matmul(m,fe,en,&m->mtp_fc_emb,S,H,H);
    for(int s=0;s<S;s++){
        const float *x=streams+(int64_t)s*W; float *y=hn+(int64_t)s*W;
        if(m->mtp_wiring==Q38_MTP_WHOLE_NORM) q38_rms0(y,x,m->mtp_norm_hid,W,c->eps);
        else for(int b=0;b<C;b++)
            q38_rms0(y+(int64_t)b*H,x+(int64_t)b*H,m->mtp_norm_hid+(int64_t)b*H,H,c->eps);
    }
    /* fc_hidden on every stream: S rows of four streams are S*C rows of hidden */
    float *fh=falloc((int64_t)S*W);
    q38_dense_matmul(m,fh,hn,&m->mtp_fc_hid,S*C,H,H);
    for(int s=0;s<S;s++)for(int b=0;b<C;b++)for(int d=0;d<H;d++)
        hyper[(int64_t)s*W+(int64_t)b*H+d]=fh[(int64_t)s*W+(int64_t)b*H+d]+fe[(int64_t)s*H+d];
    free(fh);free(en);free(fe);free(row);free(hn);
}

/* The head over S pairs at rows pos_base..pos_base+S-1: their K/V/indexer
 * rows enter its attention and, with `logit`, the last row's logits come
 * back through the head's mixer and lm_head; with `hidden`, the last row's
 * four streams as the head's layer leaves them (what a deeper draft reads in
 * place of the model's). Rows go in bounded chunks so a long prompt's pairs
 * do not need a prompt-sized workspace; the head's attention is causal, so
 * chunking changes no result. */
static void q38_mtp_rows(Model *m,const float *streams,const int *next,int S,int pos_base,float *logit,
                         float *hidden) {
    Cfg *c=&m->c; int H=c->hidden,W=c->hc_width,C=c->hc_count;
    int cap=q38_bounded_prefill_rows(S,0,(uint64_t)(3*W+2*H+C)*sizeof(float));
    float *hyper=falloc((int64_t)cap*W),*mixed=falloc((int64_t)cap*H);
    float *inject=falloc((int64_t)cap*C),*block=falloc((int64_t)cap*H);
    for(int base=0;base<S;base+=cap){
        int rows=S-base<cap?S-base:cap;
        q38_mtp_input(m,streams+(int64_t)base*W,next+base,rows,pos_base+base,hyper);
        q38_layer_forward(m,c->layers,hyper,next+base,rows,pos_base+base,mixed,inject,block);
        m->mtp_len=pos_base+base+rows;
        if(hidden&&base+rows==S)memcpy(hidden,hyper+(int64_t)(rows-1)*W,(size_t)W*sizeof(float));
        if(logit&&base+rows==S){
            double started=now_s();
            q38_gr_read(m,&m->mtp_mixer,hyper+(int64_t)(rows-1)*W,1,mixed,NULL);
            q38_weight_matmul(logit,mixed,&m->lm_head,1,H,c->vocab);
            q38_tm_add(m,Q38_TM_LM_HEAD,started);
        }
    }
    free(hyper);free(mixed);free(inject);free(block);
}

/* The pending rows meet their next token: the head reads them (each but the
 * last with the token already known in mtp_pend_tok, the last with `tok`). */
static void q38_mtp_take_pending(Model *m,int tok,float *logit,float *hidden) {
    int next[Q38_SPEC_ROWS],n=m->mtp_pend_n;
    for(int j=0;j+1<n;j++)next[j]=m->mtp_pend_tok[j];
    next[n-1]=tok;
    q38_mtp_rows(m,m->mtp_pend,next,n,m->mtp_len,logit,hidden);
    m->mtp_pend_n=0;
}

/* After a forward fed ids[0..S) at pos_base (step()): the pairs it completes
 * go into the head -- the pending rows with ids[0], then rows 0..S-2 with
 * ids[1..S-1] -- and row S-1 waits for the token after it. A forward that
 * does not continue the head's rows (a rewind no restore accompanied) stops
 * the head until the next reset or restore: it cannot draft past rows it
 * never read. */
static void q38_mtp_feed(Model *m,const int *ids,int S,int pos_base,const float *streams) {
    int W=m->c.hc_width;
    if(m->mtp_len+m->mtp_pend_n>pos_base){
        m->mtp_pend_n=0;
        if(m->mtp_len>pos_base-1)m->mtp_len=pos_base>0?pos_base-1:0;
    }
    if(m->mtp_pend_n&&m->mtp_len+m->mtp_pend_n==pos_base)q38_mtp_take_pending(m,ids[0],NULL,NULL);
    if(m->mtp_len!=pos_base){m->mtp_pend_n=0;return;}
    if(S>1)q38_mtp_rows(m,streams,ids+1,S-1,pos_base,NULL,NULL);
    memcpy(m->mtp_pend,streams+(int64_t)(S-1)*W,(size_t)W*sizeof(float));
    m->mtp_pend_n=1;
}

static int q38_argmax(const float *lo,int V) {
    int best=0;float value=lo[0];
    for(int i=1;i<V;i++)if(lo[i]>value){value=lo[i];best=i;}
    return best;
}

/* The head's draft for the token after `tok`, which is about to be fed at
 * `pos`: the pending rows take `tok`, and the last one's logits name the
 * token at pos+1; `hidden` gets that row's streams. -1 when the head has not
 * read up to pos. */
static int q38_mtp_draft(Model *m,int tok,int pos,float *hidden) {
    if(!m->mtp_pend_n||m->mtp_len+m->mtp_pend_n!=pos)return -1;
    float *logit=falloc(m->c.vocab);
    q38_mtp_take_pending(m,tok,logit,hidden);
    int draft=q38_argmax(logit,m->c.vocab);
    if(g_q38_mtp_dump){
        int32_t head[2]={pos-1,tok};
        if(fwrite(head,sizeof head,1,g_q38_mtp_dump)!=1||
           fwrite(logit,sizeof(float),(size_t)m->c.vocab,g_q38_mtp_dump)!=(size_t)m->c.vocab){
            fprintf(stderr,"Q38_MTP_DUMP: write failed\n");exit(1);
        }
    }
    free(logit);
    return draft;
}

/* A deeper draft: the head runs one more row past its settled ones, on its own
 * streams of the row before (`hidden`, standing in for the model's, which only
 * the verify computes) with the draft just proposed, at head row `row`. Its
 * K/V/indexer row there is a tail: the settled pair the head reads at that row
 * once the verify has run overwrites it before anything reads past it.
 * mtp_len stays where the settled rows end; `hidden` becomes this row's
 * streams, for the draft after. */
static int q38_mtp_draft_more(Model *m,float *hidden,int tok,int row) {
    Cfg *c=&m->c; int H=c->hidden,W=c->hc_width,C=c->hc_count;
    float *hyper=falloc(W),*mixed=falloc(H),*inject=falloc(C),*block=falloc(H),*logit=falloc(c->vocab);
    q38_mtp_input(m,hidden,&tok,1,row,hyper);
    q38_layer_forward(m,c->layers,hyper,&tok,1,row,mixed,inject,block);
    memcpy(hidden,hyper,(size_t)W*sizeof(float));
    double started=now_s();
    q38_gr_read(m,&m->mtp_mixer,hyper,1,mixed,NULL);
    q38_weight_matmul(logit,mixed,&m->lm_head,1,H,c->vocab);
    q38_tm_add(m,Q38_TM_LM_HEAD,started);
    int draft=q38_argmax(logit,c->vocab);
    free(hyper);free(mixed);free(inject);free(block);free(logit);
    return draft;
}

static float *step(Model *m,const int *ids,int S,int pos_base) {
    float *streams=NULL;
    float *logit=q38_forward(m,ids,S,pos_base,1,m->mtp?&streams:NULL);
    if(streams){q38_mtp_feed(m,ids,S,pos_base,streams);free(streams);}
    return logit;
}

/* ---- speculative decoding: MTP drafts and prompt lookup -------------------
 * colibri.c's DRAFT loop on a model with recurrent state. The caller asks for
 * the logits that follow the token it just picked (q38_spec_step); a draft
 * source proposes up to five tokens after it -- the MTP head (Q38_MTP=1) up to
 * three, the first from the model's streams and each next one from the head's
 * own streams of the row before (q38_mtp_draft_more), or prompt lookup
 * (COLI_LOOKUP=1) the tokens that followed the last n-gram where it occurred
 * before (spec_draft.h) -- and one forward over the token and its k drafts
 * (the verify, S = k+1 rows) returns the first row's logits as step() would
 * and keeps the others. The caller's next picks settle the drafts one at a
 * time: a pick equal to the next draft is answered with that draft's row and
 * no forward; the first pick that differs undoes the rows after the ones that
 * stood. The DeltaNet and PLE state go back to the copy the verify took after
 * its last standing row (q38_deltanet and q38_ple copy it after each of the
 * first snap_rows rows, row r into slot r; the rollback swaps pointers),
 * kv_len and the kv_prefix record go back to it, and the attention and
 * indexer rows past it are a stale tail the next forward overwrites, as every
 * rewind here treats them. The head's own rows never need undoing: it reads
 * settled rows only, and the rows its deeper drafts write past them are such
 * a tail too.
 *
 * Every verify row is computed the way a decode step computes it (the CPU
 * kernels give a row the same bits whatever S is; device matrices run row by
 * row, see q38_weight_matmul), so the logits the caller sees, and its picks,
 * are those of plain decoding, greedy or sampled: no draft reaches the
 * sampler, and a sample that equals one is simply an accepted draft. The one
 * exception is the CUDA expert tier, whose float order already depends on
 * which experts are resident when, with drafts or without.
 *
 * How many drafts: Q38_MTP_DRAFTS (1..3, default 2) fixes the head's; 0 lets
 * the gate pick 0..3 from the measured acceptance per draft position and the
 * measured cost of a verify by its rows (spec_draft.h). Lookup drafts are
 * always gated (COLI_SPEC_GATE=0 drafts every proposal in full: tests). With
 * both sources a verify carries the proposal whose expected tokens per unit of
 * time is higher, the longer one on a tie. */
typedef struct {
    int on, force, force_row;    /* MTP drafting; Q38_MTP_FORCE's mode letter (or 0) and its row */
    int depth;                   /* Q38_MTP_DRAFTS: drafts per verify, 0 = the gate picks 1..3 */
    int lookup, lookup_max;      /* COLI_LOOKUP: prompt-lookup drafts, up to COLI_LOOKUP_DRAFTS */
    int lk_force, lk_force_row;  /* COLI_LOOKUP_FORCE (tests): the oracle's tokens as the proposal */
    int *hist, hist_n, hist_cap; /* the tokens fed so far and the one about to be: what lookup searches */
    SpecGate gate;
    /* a verify whose rows past the first await the caller's picks: rows
     * 0..ahead_n-1 at ahead_pos.., ids[] their tokens, ahead_i the next row to
     * hand out (its draft ids[ahead_i] must be the caller's pick) */
    int ahead_n, ahead_i, ahead_pos, ahead_src, head_ok;
    int ids[Q38_SPEC_ROWS];
    float *ahead_logit, *ahead_streams;
    uint64_t drafts, accepted, forwards, tokens;   /* the MTP head's drafts (the report's counts) */
    uint64_t verifies;                             /* the MTP head's verifies */
    uint64_t lk_drafts, lk_accepted, lk_verifies;  /* prompt lookup's */
    uint64_t depth_prop[Q38_SPEC_ROWS], depth_hit[Q38_SPEC_ROWS];   /* MTP drafts by position */
    int ended_ahead;             /* the generation ended with a verify's rows unconsumed (undone) */
} Q38Spec;

/* Copy slots for a verify of up to rows+1 rows: allocated the first time they are
 * needed, kept for the model's life. 0 = out of memory (the caller drafts less). */
static int q38_spec_alloc(Model *m,int rows) {
    Cfg *c=&m->c;
    if(rows>Q38_SPEC_SNAPS)rows=Q38_SPEC_SNAPS;
    for(int s=m->snap_slots;s<rows;s++){
        m->snap_rec[s]=(float**)calloc((size_t)c->layers,sizeof(float*));
        m->snap_conv[s]=(float**)calloc((size_t)c->layers,sizeof(float*));
        if(!m->snap_rec[s]||!m->snap_conv[s])goto refused;
        for(int i=0;i<c->layers;i++)if(!c->is_attn[i]){
            m->snap_rec[s][i]=(float*)malloc((size_t)c->dn_vheads*c->dn_kdim*c->dn_vdim*sizeof(float));
            m->snap_conv[s][i]=(float*)malloc((size_t)c->dn_conv_dim*(c->dn_convk-1)*sizeof(float));
            if(!m->snap_rec[s][i]||!m->snap_conv[s][i])goto refused;
        }
        if(m->PLE_conv_state){
            m->snap_ple[s]=(float*)malloc((size_t)c->hc_width*(c->ple_convk-1)*c->ngram_size*sizeof(float));
            if(!m->snap_ple[s])goto refused;
        }
        m->snap_slots=s+1;
        continue;
refused:
        /* The next token may try again. Keep complete slots, but release the
         * unfinished one so retries neither lose its pointers nor consume
         * the memory the plain decode step needs. */
        for(int i=0;i<c->layers;i++){
            free(m->snap_rec[s]?m->snap_rec[s][i]:NULL);
            free(m->snap_conv[s]?m->snap_conv[s][i]:NULL);
        }
        free(m->snap_rec[s]);free(m->snap_conv[s]);free(m->snap_ple[s]);
        m->snap_rec[s]=m->snap_conv[s]=NULL;m->snap_ple[s]=NULL;
        return 0;
    }
    return 1;
}

/* Back to the state after the verify's first `keep` rows, `len` positions fed. */
static void q38_spec_rollback(Model *m,int len,int keep) {
    Cfg *c=&m->c; int slot=keep-1;
#ifdef COLI_VULKAN
    q38c_rollback(m,slot,len);   /* the dense chain's own copies: its device buffers swap too */
#endif
    for(int i=0;i<c->layers;i++)if(!c->is_attn[i]){
        float *t=m->DN_rec[i];m->DN_rec[i]=m->snap_rec[slot][i];m->snap_rec[slot][i]=t;
        t=m->DN_conv[i];m->DN_conv[i]=m->snap_conv[slot][i];m->snap_conv[slot][i]=t;
    }
    if(m->snap_ple[slot]){
        float *t=m->PLE_conv_state;m->PLE_conv_state=m->snap_ple[slot];m->snap_ple[slot]=t;
        memcpy(m->ple_history,m->snap_ple_history[slot],sizeof(m->snap_ple_history[slot]));
        m->ple_history_len=m->snap_ple_history_len[slot];
    }
    m->kv_len=len;
    if(m->kvp.len>len)m->kvp.len=len;
}

/* The verify's first `keep` rows stand (rejected: the next one did not): undo
 * the rest, and hand the standing rows' streams to the head, each but the last
 * with the token that followed it. */
static void q38_spec_settle(Model *m,Q38Spec *sp,int keep,int rejected) {
    int n=sp->ahead_n,W=m->c.hc_width;
    if(keep<n)q38_spec_rollback(m,sp->ahead_pos+keep,keep);
    if(sp->head_ok){
        memcpy(m->mtp_pend,sp->ahead_streams,(size_t)keep*W*sizeof(float));
        m->mtp_pend_n=keep;
        for(int j=0;j+1<keep;j++)m->mtp_pend_tok[j]=sp->ids[j+1];
    }
    spec_gate_result(&sp->gate,sp->ahead_src,keep-1+(rejected?1:0),keep-1);
    free(sp->ahead_logit);free(sp->ahead_streams);
    sp->ahead_logit=sp->ahead_streams=NULL;sp->ahead_n=sp->ahead_i=0;
}

/* The token the forced modes put at draft row j (1-based) of a verify at pos, or
 * -1 to keep the source's: the oracle's token there (ref.json), or a wrong one at
 * the row the mode rejects. */
static int q38_spec_forced(int mode,int row_mode,int verify,int k,int j,int pos,int V) {
    int truth=pos+j<g_q38_mtp_oracle_n?g_q38_mtp_oracle[pos+j]:-1;
    if(truth<0||!mode||mode=='r')return -1;
    int wrong=0;
    if(mode=='m')wrong=(verify&1)?1:0;               /* alternate verifies: the first draft wrong */
    else if(mode=='w')wrong=row_mode;                /* row N: drafts before it right, N wrong */
    else if(mode=='c')wrong=verify%(k+1)+1;          /* cycle the rejected row, k+1 = none */
    return j==wrong?(truth+1)%V:truth;
}

static void q38_spec_hist_push(Q38Spec *sp,int tok,int pos) {
    if(!sp->lookup||pos<0)return;
    if(pos>=sp->hist_cap){
        int cap=sp->hist_cap?sp->hist_cap:256; while(cap<=pos)cap*=2;
        int *h=(int*)realloc(sp->hist,(size_t)cap*sizeof(int));
        if(!h){sp->lookup=0;return;}
        sp->hist=h;sp->hist_cap=cap;
    }
    if(pos>sp->hist_n){sp->lookup=0;return;}   /* a gap: the caller's history is not ours */
    sp->hist[pos]=tok;sp->hist_n=pos+1;
}

/* The logits that follow `tok`, fed at `pos`, exactly as step(m,&tok,1,pos)
 * gives them. `more` is how many tokens the caller may still want after
 * `tok`: a verify of k drafts is worth its rows only when k+1 more are wanted.
 * The returned buffer is the caller's (it may hold more rows past the first
 * vocab floats). */
static float *q38_spec_step(Model *m,Q38Spec *sp,int tok,int pos,int more) {
    Cfg *c=&m->c; int V=c->vocab;
    sp->tokens++;
    q38_spec_hist_push(sp,tok,pos);
    if(sp->ahead_n){
        int i=sp->ahead_i;
        if(tok==sp->ids[i]&&pos==sp->ahead_pos+i&&sp->force!='r'){
            /* draft i stands: its row's logits answer, with no forward */
            float *logit=falloc(V);
            memcpy(logit,sp->ahead_logit+(int64_t)(i-1)*V,(size_t)V*sizeof(float));
            if(sp->ahead_src==SPEC_SRC_MTP){sp->accepted++;sp->depth_hit[i-1]++;}
            else sp->lk_accepted++;
            if(++sp->ahead_i==sp->ahead_n)q38_spec_settle(m,sp,sp->ahead_n,0);
            return logit;
        }
        q38_spec_settle(m,sp,i,1);
    }
    /* how many drafts this verify can carry: rows the caller still wants, room in the cache */
    int room=more-1;
    if(room>Q38_SPEC_ROWS-1)room=Q38_SPEC_ROWS-1;
    if(pos+room>=m->kv_cap)room=m->kv_cap-1-pos;
    int d[Q38_SPEC_ROWS],k=0,src=SPEC_SRC_MTP;
    double t_verify=0.0;   /* when the verify's cost began (0: at its forward) */
    int head_ready=sp->on&&m->mtp_pend_n&&m->mtp_len+m->mtp_pend_n==pos;
    if(room>0&&(head_ready||sp->lookup)){
        /* the lookup's proposal, gated; the forced mode proposes the oracle's tokens */
        int kl=0,dl[Q38_SPEC_ROWS];double vl=0.0;
        if(sp->lookup){
            int want=room<sp->lookup_max?room:sp->lookup_max,n=0;
            if(sp->lk_force)
                while(n<want){
                    int t=q38_spec_forced(sp->lk_force,sp->lk_force_row,(int)sp->lk_verifies,want,n+1,pos,V);
                    if(t<0)break;
                    dl[n++]=t;
                }
            else n=spec_lookup(sp->hist,sp->hist_n,2,4,want,dl);
            if(n>0)kl=spec_gate_pick(&sp->gate,SPEC_SRC_LOOKUP,n,&vl);
        }
        /* the head's depth: fixed, or the gate's pick */
        int km=0;double vm=0.0;
        if(head_ready){
            int maxm=room<3?room:3;
            if(sp->depth){km=sp->depth<maxm?sp->depth:maxm;vm=spec_gate_value(&sp->gate,SPEC_SRC_MTP,km,NULL);}
            else km=spec_gate_pick(&sp->gate,SPEC_SRC_MTP,maxm,&vm);
        }
        if(kl>0&&(km==0||sp->lk_force||vl>vm||(vl==vm&&kl>km))){k=kl;src=SPEC_SRC_LOOKUP;memcpy(d,dl,(size_t)k*sizeof(int));}
        else if(km>0)k=km;
        if(k>0&&!q38_spec_alloc(m,k))k=0;   /* no memory for the copies: a plain step */
        if(k>0&&src==SPEC_SRC_MTP){
            float *hidden=falloc(c->hc_width);
            t_verify=now_s();   /* the head's first draft is the head pass a plain step makes too */
            d[0]=q38_mtp_draft(m,tok,pos,hidden);
            double t_first=now_s();
            for(int j=1;j<k;j++){
                double t0=now_s();
                d[j]=q38_mtp_draft_more(m,hidden,d[j-1],pos+j-1);
                spec_gate_draft_cost(&sp->gate,SPEC_SRC_MTP,now_s()-t0);
            }
            free(hidden);
            t_verify+=now_s()-t_first;   /* the deeper drafts are the gate's d, not the verify's */
            if(sp->force)for(int j=1;j<=k;j++){
                int t=q38_spec_forced(sp->force,sp->force_row,(int)sp->verifies,k,j,pos,V);
                if(t>=0)d[j-1]=t;
            }
        }
    }
    if(!k){
        sp->forwards++;
        double t0=now_s();
        float *logit=step(m,&tok,1,pos);
        spec_gate_forward(&sp->gate,1,now_s()-t0);
        return logit;
    }
    /* the head reads its pending rows before the verify's: the MTP draft did, a
     * lookup verify has them take `tok` here */
    if(!t_verify)t_verify=now_s();
    if(m->mtp&&src!=SPEC_SRC_MTP&&m->mtp_pend_n&&m->mtp_len+m->mtp_pend_n==pos)
        q38_mtp_take_pending(m,tok,NULL,NULL);
    int S=k+1;
    sp->ids[0]=tok;memcpy(sp->ids+1,d,(size_t)k*sizeof(int));
    float *streams=NULL;
    m->snap_rows=k;g_q38_rowwise=1;
    float *logit=q38_forward(m,sp->ids,S,pos,S,m->mtp?&streams:NULL);
    spec_gate_forward(&sp->gate,S,now_s()-t_verify);
    m->snap_rows=0;g_q38_rowwise=0;
    sp->forwards++;
    if(src==SPEC_SRC_MTP){
        sp->drafts+=(uint64_t)k;sp->verifies++;
        for(int j=0;j<k;j++)sp->depth_prop[j]++;
    } else {sp->lk_drafts+=(uint64_t)k;sp->lk_verifies++;}
    sp->ahead_n=S;sp->ahead_i=1;sp->ahead_pos=pos;sp->ahead_src=src;
    sp->head_ok=m->mtp&&m->mtp_len==pos&&streams;
    sp->ahead_logit=falloc((int64_t)k*V);
    memcpy(sp->ahead_logit,logit+V,(size_t)k*V*sizeof(float));
    sp->ahead_streams=streams;
    return logit;
}

/* End of a generation: verify rows the caller never consumed are undone, so
 * the state is the one plain decoding leaves after the same tokens. */
static void q38_spec_end(Model *m,Q38Spec *sp) {
    if(sp->ahead_n){sp->ended_ahead=1;q38_spec_settle(m,sp,sp->ahead_i,0);}
}

static void q38_spec_free(Q38Spec *sp) {
    free(sp->hist);sp->hist=NULL;sp->hist_n=sp->hist_cap=0;
}

static float q38_mean(const float *x,int n) {
    double sum=0.0; for(int i=0;i<n;i++) sum+=x[i];
    return n?(float)(sum/n):0.f;
}

/* The checkpoint's MTP head drafts by default (Q38_MTP unset), when the checkpoint has
 * one this engine runs and the whole model is loaded; otherwise plain decoding, with a
 * line. Q38_MTP=0 leaves it unread. The CLI and serve engine attach it after the model
 * (the Segment and Edge adapters never do). Q38_MTP=1 asks for it: a checkpoint without
 * a head this engine runs is then refused rather than decoded without one, so a
 * benchmark cannot quietly measure plain decoding. The head's
 * routed experts stay the snapshot's (FP8 on the release): the int4-g64
 * sidecar holds the model's layers, and the head's one layer, read on at
 * most two rows per draft, is a small part of the expert traffic. */
static void q38_mtp_attach(Model *m,int cap) {
    /* On by default when the checkpoint has the head and the whole model is loaded
     * (docs/speculative.md: +16 to +20% measured, the output plain decoding's); unset,
     * a checkpoint without it decodes as before with a line. Q38_MTP=1 asks for it and
     * refuses a checkpoint that cannot; Q38_MTP=0 turns it off. */
    const char *asked=getenv("Q38_MTP");
    int forced=asked&&*asked;
    if(!q38_env_bool("Q38_MTP",1))return;
    Cfg *c=&m->c;int H=c->hidden,W=c->hc_width;
    if(!forced&&(m->range_begin!=0||m->range_end!=c->layers||!m->lm_head.rows||c->mtp_layers!=1)){
        fprintf(stderr,"[qwen38] speculative decoding with the MTP head: off (%s; Q38_MTP=0 silences this)\n",
                c->mtp_layers==0?"the checkpoint has no MTP head":c->mtp_layers!=1?"an MTP head this engine does not run":
                "it needs the whole model loaded");
        return;
    }
    if(m->range_begin!=0||m->range_end!=c->layers||!m->lm_head.rows){
        fprintf(stderr,"Q38_MTP=1 needs the whole model loaded -- refusing\n");exit(1);
    }
    if(c->mtp_layers!=1){
        if(c->mtp_layers==0)
            fprintf(stderr,"Q38_MTP=1 but config.json has no MTP head (mtp_num_hidden_layers) -- refusing\n");
        else if(c->mtp_layers<0)
            fprintf(stderr,"Q38_MTP=1 but the config's MTP head is not one attention layer over the "
                           "model's embedding -- refusing\n");
        else fprintf(stderr,"Q38_MTP=1 but the MTP head has %d layers; this engine runs one -- refusing\n",
                     c->mtp_layers);
        exit(1);
    }
    const char *wiring=getenv("Q38_MTP_WIRING");
    m->mtp_wiring=wiring&&*wiring?wiring[0]:Q38_MTP_STREAM_NORM;
    if((wiring&&*wiring&&wiring[1])||(m->mtp_wiring!=Q38_MTP_WHOLE_NORM&&m->mtp_wiring!=Q38_MTP_STREAM_NORM)){
        fprintf(stderr,"Q38_MTP_WIRING must be b (the default) or a (qwen38_core.h, the MTP head)\n");exit(1);
    }
    /* the release keeps the head at the top level, next to model.* */
    char probe[96];const char *found=NULL;
    const char *bases[3]={"mtp","model.mtp",NULL};
    char nested[48];snprintf(nested,sizeof nested,"%s.mtp",m->prefix);bases[2]=nested;
    for(int i=0;i<3&&!found;i++){
        snprintf(probe,sizeof probe,"%s.fc_embedding.weight",bases[i]);
        if(st_has(&m->S,probe))found=bases[i];
    }
    if(!found&&!forced){   /* the config names a head the weights do not carry (a stripped container) */
        fprintf(stderr,"[qwen38] speculative decoding with the MTP head: off (the checkpoint has no "
                       "mtp.fc_embedding.weight; Q38_MTP=0 silences this)\n");
        return;
    }
    if(!found){fprintf(stderr,"Q38_MTP=1 but the checkpoint has no mtp.fc_embedding.weight -- refusing\n");exit(1);}
    snprintf(m->mtp_prefix,sizeof m->mtp_prefix,"%s",found);
    uint64_t resident_before=m->resident_weight_bytes;
    char nm[384],base[320];
    snprintf(nm,sizeof nm,"%s.pre_fc_norm_embedding.weight",found); m->mtp_norm_emb=q38_load_tensor(m,nm,H);
    snprintf(nm,sizeof nm,"%s.pre_fc_norm_hidden.weight",found); m->mtp_norm_hid=q38_load_tensor(m,nm,W);
    snprintf(nm,sizeof nm,"%s.fc_embedding.weight",found); m->mtp_fc_emb=q38_load_weight(m,nm,H,H);
    snprintf(nm,sizeof nm,"%s.fc_hidden.weight",found); m->mtp_fc_hid=q38_load_weight(m,nm,H,H);
    snprintf(base,sizeof base,"%s.hyper_connection_mixer",found); q38_load_gr_at(m,&m->mtp_mixer,base,0);
    m->mtp=1;                                 /* from here on index c.layers is the head's layer */
    /* Q38_MTP_CAP: the head's expert cache, by default the cap every layer
     * has. Its experts are FP8 even beside the int4 sidecar, so on a cap
     * sized to fill RAM with int4 experts it costs more than a model layer. */
    int mtp_cap=q38_env_positive_int("Q38_MTP_CAP",cap,c->experts);
    q38_load_layer(m,c->layers,mtp_cap);
    m->mtp_pend=falloc((int64_t)Q38_SPEC_ROWS*W);
    /* the verify's first rollback copy (q38_spec_rollback); deeper verifies add theirs */
    if(!q38_spec_alloc(m,1)){fprintf(stderr,"OOM MTP rollback state\n");exit(1);}
    snprintf(nm,sizeof nm,"%s.layers.0.mlp.experts.0.gate_proj.weight",found);
    st_tensor *expert=st_find(&m->S,nm);
    double expert_bytes=expert?3.0*(double)expert->nbytes:0.0;   /* gate, up and down are the same size */
    const char *how=m->mtp_wiring==Q38_MTP_STREAM_NORM?"a norm per stream, fc_hidden per stream":
                    "one norm over the four streams, fc_hidden per stream (diagnostic; b is the default)";
    fprintf(stderr,"[qwen38] MTP head %s.*: wiring %c (%s), RoPE base %g, %.2f MiB resident; "
                   "routed experts %s from the snapshot%s, cache %d (Q38_MTP_CAP) x %.2f MiB = %.2f GiB full\n",
            found,m->mtp_wiring,how,(double)c->mtp_theta,
            (m->resident_weight_bytes-resident_before)/1048576.0,
            expert?st_dtype_name(expert->dtype):"(fused)",m->x4?" (the int4-g64 sidecar covers the model's layers)":"",
            mtp_cap,expert_bytes/1048576.0,expert_bytes*mtp_cap/1073741824.0);
    fprintf(stderr,"[qwen38] MTP norm weights, mean: embedding %.4f, hidden %.4f, mixer %.4f "
                   "(q38_rms0 scales by 1+w; a plain norm's weights would sit near 1)\n",
            q38_mean(m->mtp_norm_emb,H),q38_mean(m->mtp_norm_hid,W),q38_mean(m->mtp_mixer.norm,W));
}

static int q38_tm_enabled(void) {
    const char *enabled=getenv("COLI_TIMERS");
    return enabled&&enabled[0]=='1'&&enabled[1]=='\0';
}

static void q38_tm_report_bank(const Q38Timers *timers,const char *scope) {
    if(!q38_tm_enabled())return;
    static const char *names[Q38_TM_COUNT]={
        "expert-read","fp8-expand","routed-expert","shared-expert",
        "resident-mm","deltanet","qsa-index","qsa-attn","ple","lm-head"
    };
    double per=timers->forwards?1000.0/timers->forwards:0.0;
    fprintf(stderr,"[qwen38 timers] %s: %llu forwards\n",scope,
            (unsigned long long)timers->forwards);
    for(int i=0;i<Q38_TM_COUNT;i++)
        fprintf(stderr,"[qwen38 timers]   %-14s %9.3f s  %9.3f ms/forward\n",
                names[i],timers->seconds[i],timers->seconds[i]*per);
    fprintf(stderr,"[qwen38 timers] architecture phases overlap resident-mm; "
                   "expert-read is disk service and fp8-expand is synchronous miss work\n");
}

/* Snapshot of the timer bank taken when the prompt's forward has finished
 * (generate() sets it), so a "decode" bank can be reported apart from the
 * prefill. Without it the per-forward figures of a long prompt are prefill
 * work divided by decode forwards, which reads like a decode budget and is
 * not one. */
static Q38Timers g_tm_prefill_snapshot; static int g_tm_have_snapshot;
static void q38_tm_snapshot_prefill(const Model *m) {
    g_tm_prefill_snapshot=m->timers; g_tm_have_snapshot=1;
}
static void tm_report(const Model *m) {
    q38_tm_report_bank(&m->timers,"total");
    if(g_tm_have_snapshot&&q38_tm_enabled()){
        Q38Timers decode=m->timers;
        for(int i=0;i<Q38_TM_COUNT;i++)decode.seconds[i]-=g_tm_prefill_snapshot.seconds[i];
        decode.forwards-=g_tm_prefill_snapshot.forwards;
        q38_tm_report_bank(&g_tm_prefill_snapshot,"prefill");
        q38_tm_report_bank(&decode,"decode");
    }
    if(q38_tm_enabled())
        fprintf(stderr,"[qwen38 expert I/O] weight-ranges=%llu scale-ranges=%llu "
                       "coalesced-gate-up=%llu prefetched=%llu parallel-batches=%llu "
                       "resident-scales=%.2f MiB\n",
                (unsigned long long)m->expert_weight_reads,
                (unsigned long long)m->expert_scale_reads,
                (unsigned long long)m->expert_pair_reads,
                (unsigned long long)m->expert_prefetch_ranges,
                (unsigned long long)m->expert_parallel_batches,
                m->expert_scale_bytes/1048576.0);
}

static void q38_layer_free(Layer *l) {
    if(!l) return;
    free(l->attn_gr.norm);q38_weight_free(&l->attn_gr.down);q38_weight_free(&l->attn_gr.up);q38_weight_free(&l->attn_gr.inject);
    free(l->mlp_gr.norm);q38_weight_free(&l->mlp_gr.down);q38_weight_free(&l->mlp_gr.up);q38_weight_free(&l->mlp_gr.inject);
    q38_weight_free(&l->router);q38_weight_free(&l->sh_g);q38_weight_free(&l->sh_u);q38_weight_free(&l->sh_d);free(l->sh_gate);
    q38_weight_free(&l->q);q38_weight_free(&l->k);q38_weight_free(&l->v);q38_weight_free(&l->o);free(l->qn);free(l->kn);
    q38_weight_free(&l->idx_qk);free(l->idx_qn);free(l->idx_kn);
    q38_weight_free(&l->dn_qkv);q38_weight_free(&l->dn_z);q38_weight_free(&l->dn_b);q38_weight_free(&l->dn_a);free(l->dn_conv);
    free(l->dn_dtbias);free(l->dn_alog);free(l->dn_norm);q38_weight_free(&l->dn_out);
    q38_weight_free(&l->ple_key);q38_weight_free(&l->ple_value);free(l->ple_norm_key);free(l->ple_norm_query);
    free(l->ple_norm_conv); free(l->ple_conv);
    memset(l,0,sizeof(*l));
}

static void q38_model_free(Model *m) {
    if(!m) return;
    kv_prefix_free(&m->kvp);
    int rows=q38_layer_rows(m);
    for(int i=0;i<rows;i++) {
        if(m->L)q38_layer_free(&m->L[i]);
        if(m->cache) {
            if(m->cache[i].slots) {
                for(int s=0;s<m->cache[i].n;s++) {
                    q38_weight_free(&m->cache[i].slots[s].gate);
                    q38_weight_free(&m->cache[i].slots[s].up);
                    q38_weight_free(&m->cache[i].slots[s].down);
                    free(m->cache[i].slots[s].fp8_slab);
                    free(m->cache[i].slots[s].int4_slab);
                }
            }
            free(m->cache[i].slots); free(m->cache[i].by_expert);
        }
        if(m->expert_scales)free(m->expert_scales[i].values);
        if(i<m->c.layers){free(m->DN_rec ? m->DN_rec[i] : NULL); free(m->DN_conv ? m->DN_conv[i] : NULL);}
        if(i<m->c.layers)for(int sl=0;sl<m->snap_slots;sl++){
            free(m->snap_rec[sl] ? m->snap_rec[sl][i] : NULL); free(m->snap_conv[sl] ? m->snap_conv[sl][i] : NULL);
        }
        free(m->K ? m->K[i] : NULL); free(m->V ? m->V[i] : NULL); free(m->IK ? m->IK[i] : NULL);
    }
    for(int sl=0;sl<Q38_SPEC_SNAPS;sl++){ free(m->snap_rec[sl]); free(m->snap_conv[sl]); free(m->snap_ple[sl]); }
    free(m->mtp_pend);
    free(m->mtp_norm_emb); free(m->mtp_norm_hid);
    q38_weight_free(&m->mtp_fc_emb); q38_weight_free(&m->mtp_fc_hid);
    free(m->mtp_mixer.norm); q38_weight_free(&m->mtp_mixer.down); q38_weight_free(&m->mtp_mixer.up);
    if(m->x4){st_destroy(&m->x4->S);free(m->x4->fd);free(m->x4->off);free(m->x4);}
    free(m->L); free(m->cache); free(m->expert_scales); free(m->DN_rec); free(m->DN_conv); free(m->K); free(m->V); free(m->IK);
    q38_weight_free(&m->embed);q38_weight_free(&m->lm_head);
    free(m->final_gr.norm);q38_weight_free(&m->final_gr.down);q38_weight_free(&m->final_gr.up);q38_weight_free(&m->final_gr.inject);
    free(m->ple_history); free(m->PLE_conv_state); free(m->c.is_attn); st_destroy(&m->S);
    memset(m,0,sizeof(*m));
}

#endif /* COLI_QWEN38_CORE_H */
