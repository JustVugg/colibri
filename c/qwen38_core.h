/* Native Qwen3.8-Flash-Next text core.
 *
 * This header is included once by qwen38.c after its tokenizer and protocol
 * helpers.  It intentionally consumes the official HF safetensors layout:
 * the multimodal wrapper's `model.language_model` namespace and the standalone
 * text model's `model` namespace are both accepted.  Vision and MTP tensors are
 * indexed by st.h but are never read.
 */
#ifndef COLI_QWEN38_CORE_H
#define COLI_QWEN38_CORE_H

#define Q38_MAX_LAYERS 512
#define Q38_MAX_EXPERTS 1024
#define Q38_MAX_TOPK 256
#define Q38_MAX_PLE_PARTS 512
#define Q38_PREFILL_BATCH_ROWS 32
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
} Cfg;

typedef enum {
    Q38_WEIGHT_NONE = 0,
    Q38_WEIGHT_F32,
    Q38_WEIGHT_BF16,
    Q38_WEIGHT_FP8
} Q38WeightKind;

typedef struct {
    void *data;
    float *scales;                 /* block-FP8 only */
    int rows, cols;
    int64_t elements, scale_count;
    Q38WeightKind kind;
    unsigned owns_data:1, owns_scales:1;
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
} Slot;
typedef struct { Slot *slots; int *by_expert, n, cap; } LCache;

typedef struct {
    float *values;                 /* [expert][gate,up,down][block] */
    int64_t scale_count;
    int ready;                     /* 0 unknown, 1 resident, -1 incompatible */
} Q38ExpertScaleCache;

typedef struct {
    Cfg c;
    shards S;
    char prefix[32];
    Q38Weight embed, lm_head;
    GatedResidual final_gr;
    Layer *L;
    LCache *cache;
    Q38ExpertScaleCache *expert_scales;
    uint64_t clock, hits, miss;
    uint64_t expert_weight_reads, expert_scale_reads, expert_pair_reads;
    uint64_t expert_prefetch_ranges, expert_parallel_batches;
    uint64_t expert_scale_bytes;
    float **DN_rec, **DN_conv;
    float **K, **V, **IK;
    int kv_len, kv_cap, max_t;
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
    int prefill_batch;
    uint64_t resident_weight_bytes;
    double dense_load_s;
    Q38Timers timers;
} Model;

static float *g_last_logit;
static int g_capture_last_logit;

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

static void q38_weight_matmul(float *y,const float *x,const Q38Weight *weight,
                              int S,int I,int O) {
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
        matmul_fp8(y,x,(const uint8_t*)weight->data,weight->scales,S,I,O);
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
    json_free(root); free(buf); free(arena);
}

#define Q38_NEED(x,...) do{if(!(x)){fprintf(stderr,"[qwen38 config] ");fprintf(stderr,__VA_ARGS__);fprintf(stderr," -- refusing\n");exit(1);}}while(0)
static void q38_validate_cfg(const Cfg *c) {
    Q38_NEED(c->hidden>0&&c->hidden<=65536,"hidden_size=%d",c->hidden);
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
    return weight;
}

static void q38_name(Model *m,char *out,size_t cap,int layer,const char *suffix) {
    snprintf(out,cap,"%s.layers.%d.%s",m->prefix,layer,suffix);
}

static void q38_load_gr(Model *m,GatedResidual *g,int layer,const char *kind,int inject) {
    Cfg *c=&m->c; char nm[320],base[180];
    if(layer>=0) snprintf(base,sizeof base,"layers.%d.%s",layer,kind); else snprintf(base,sizeof base,"hyper_connection_mixer");
    snprintf(nm,sizeof nm,"%s.%s.hc_norm.weight",m->prefix,base); g->norm=q38_load_tensor(m,nm,c->hc_width);
    snprintf(nm,sizeof nm,"%s.%s.input_mix_weight_down.weight",m->prefix,base); g->down=q38_load_weight(m,nm,c->hc_rank,c->hc_width);
    snprintf(nm,sizeof nm,"%s.%s.input_mix_weight_up.weight",m->prefix,base); g->up=q38_load_weight(m,nm,c->hc_width,c->hc_rank);
    if(inject){snprintf(nm,sizeof nm,"%s.%s.block_inject_weight.weight",m->prefix,base);g->inject=q38_load_weight(m,nm,c->hc_count,c->hc_width);}
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
    m->K=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->V=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->IK=(float**)calloc((size_t)c->layers,sizeof(float*));
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
    m->L=(Layer*)calloc((size_t)c->layers,sizeof(Layer));
    m->cache=(LCache*)calloc((size_t)c->layers,sizeof(LCache));
    m->expert_scales=(Q38ExpertScaleCache*)calloc((size_t)c->layers,
                                                  sizeof(*m->expert_scales));
    if(!m->L||!m->cache||!m->expert_scales){fprintf(stderr,"OOM model metadata\n");exit(1);}
    for(int i=layer_begin;i<layer_end;i++){
        Layer *l=&m->L[i]; q38_load_gr(m,&l->attn_gr,i,"attn_hyper_connection",1); q38_load_gr(m,&l->mlp_gr,i,"mlp_hyper_connection",1);
        #define WLD(field,suf,o,in) q38_name(m,nm,sizeof nm,i,suf); l->field=q38_load_weight(m,nm,(o),(in))
        #define VLD(field,suf,n) q38_name(m,nm,sizeof nm,i,suf); l->field=q38_load_tensor(m,nm,(n))
        WLD(router,"mlp.gate.weight",c->experts,c->hidden);
        WLD(sh_g,"mlp.shared_expert.gate_proj.weight",c->shared_inter,c->hidden);
        WLD(sh_u,"mlp.shared_expert.up_proj.weight",c->shared_inter,c->hidden);
        WLD(sh_d,"mlp.shared_expert.down_proj.weight",c->hidden,c->shared_inter);
        VLD(sh_gate,"mlp.shared_expert_gate.weight",c->hidden);
        if(c->is_attn[i]){
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
    if(c->ple_layer>=layer_begin&&c->ple_layer<layer_end) q38_load_ple(m,&m->L[c->ple_layer]);
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
    if(!m->native_fp8||layer<0||layer>=c->layers)return 0;
    if(!m->expert_scales){
        m->expert_scales=(Q38ExpertScaleCache*)calloc((size_t)c->layers,
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

static void q38_prefetch_native_fp8_experts(Model *m,int layer,
                                            const int *experts,int count) {
    if(!m->expert_prefetch||!m->native_fp8||!experts||count<1||
       !q38_prepare_expert_scale_bank(m,layer))return;
    LCache *cache=&m->cache[layer];
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

static void q38_load_expert(Model *m,int layer,int eid,Slot *s) {
    Cfg *c=&m->c; int H=c->hidden,I=c->inter; char nm[320],sn[340];
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

static Slot *q38_expert_get(Model *m,int layer,int eid) {
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
static int q38_expert_get_batch(Model *m,int layer,const int *experts,int count,
                                Slot **selected) {
    if(!m->expert_parallel_reads||!experts||!selected||count<2||
       count>Q38_MAX_TOPK)return 0;
    LCache *cache=&m->cache[layer];
    if(cache->cap<count||!q38_prepare_expert_scale_bank(m,layer))return 0;
    Q38ExpertLoadJob jobs[Q38_MAX_TOPK];
    for(int index=0;index<count;index++){
        int expert=experts[index];
        if(expert<0||expert>=m->c.experts)return 0;
        for(int previous=0;previous<index;previous++)
            if(experts[previous]==expert)return 0;
        int slot_index=cache->by_expert[expert];
        if(slot_index>=0){
            if(slot_index>=cache->n||cache->slots[slot_index].eid!=expert)return 0;
            continue;
        }
        st_tensor *weight[3];
        if(!q38_native_fp8_expert_tensors(m,layer,expert,weight))return 0;
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
            if(!q38_native_fp8_expert_tensors(m,layer,expert,jobs[job_count].weight)){
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
#endif
        double started=now_s();
        #pragma omp parallel for schedule(static) num_threads(workers) if(job_count>1)
        for(int job=0;job<job_count;job++)
            q38_load_native_fp8_ranges(m,layer,jobs[job].expert,jobs[job].slot,
                                        jobs[job].weight);
        q38_tm_add(m,Q38_TM_EXPERT_READ,started);
        m->expert_weight_reads+=(uint64_t)job_count*2;
        m->expert_pair_reads+=(uint64_t)job_count;
        if(job_count>1)m->expert_parallel_batches++;
        for(int job=0;job<job_count;job++){
            Slot *slot=jobs[job].slot;slot->eid=jobs[job].expert;
            cache->by_expert[jobs[job].expert]=(int)(slot-cache->slots);
        }
    }
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

static void q38_ple(Model *m,const int *ids,int S,const float *hyper,float *out) {
    double phase_started=now_s();
    Cfg *c=&m->c; Layer *l=&m->L[c->ple_layer]; int H=c->hidden,C=c->hc_count,W=c->hc_width,E=c->ple_dim;
    float *emb=falloc(E),*keys=falloc(W),*value=falloc(H),*kn=falloc(W),*qn=falloc(W),*gated=falloc(W),*norm=falloc(W);
    int state_len=(c->ple_convk-1)*c->ngram_size; float *ring=m->PLE_conv_state;
    for(int s=0;s<S;s++){
        int64_t p1=m->ple_history_len>=1?m->ple_history[m->ple_history_len-1]:c->eos_id;
        int64_t p2=m->ple_history_len>=2?m->ple_history[m->ple_history_len-2]:c->eos_id;
        for(int h=0;h<c->ngram_heads;h++){
            int ng=h<c->heads_per_ngram?2:3; int64_t row=q38_hash_row(m,h,ng,ids[s],p1,p2);
            q38_ple_row(m,row,emb+(int64_t)h*c->ngram_head_dim);
        }
        q38_dense_matmul(m,keys,emb,&l->ple_key,1,E,W);q38_dense_matmul(m,value,emb,&l->ple_value,1,E,H);
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
            float *r=ring+(int64_t)d*state_len;for(int k=0;k<state_len-1;k++)r[k]=r[k+1];r[state_len-1]=norm[d];
        }
        if(ids[s]==c->eos_id)m->ple_history_len=0;
        else if(m->ple_history_len==0){m->ple_history[0]=ids[s];m->ple_history_len=1;}
        else if(m->ple_history_len==1){m->ple_history[1]=ids[s];m->ple_history_len=2;}
        else {m->ple_history[0]=m->ple_history[1];m->ple_history[1]=ids[s];}
    }
    free(emb);free(keys);free(value);free(kn);free(qn);free(gated);free(norm);
    q38_tm_add(m,Q38_TM_PLE,phase_started);
}

/* Choose a context-independent prefill chunk whose private workspace fits the
 * common target.  Callers provide exact fixed and per-row byte counts; even a
 * hostile-but-valid geometry gets one row rather than an unbounded allocation. */
static int q38_bounded_prefill_rows(int requested,uint64_t fixed,
                                    uint64_t per_row) {
    int rows=requested<Q38_PREFILL_BATCH_ROWS?requested:Q38_PREFILL_BATCH_ROWS;
    if(rows<1)return 1;
    for(;rows>1;rows--)
        if(per_row<=UINT64_MAX/(uint64_t)rows&&
           fixed<=UINT64_MAX-per_row*(uint64_t)rows&&
           fixed+per_row*(uint64_t)rows<=Q38_PREFILL_WORKSPACE_BYTES)
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
    int IQ=c->idx_qheads,ID=c->idx_dim,R=c->idx_ratio,maxsel=c->idx_budget+R-1;
    float *qp=falloc((int64_t)S*QH*2*D),*kp=falloc((int64_t)S*KVH*D),*vp=falloc((int64_t)S*KVH*D);
    float *ip=falloc((int64_t)S*(IQ+c->idx_kheads)*ID);
    q38_dense_matmul(m,qp,x,&l->q,S,H,QH*2*D);q38_dense_matmul(m,kp,x,&l->k,S,H,KVH*D);q38_dense_matmul(m,vp,x,&l->v,S,H,KVH*D);
    q38_dense_matmul(m,ip,x,&l->idx_qk,S,H,(IQ+c->idx_kheads)*ID);
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        for(int h=0;h<KVH;h++){
            float *kh=kp+(int64_t)s*KVH*D+(int64_t)h*D;q38_rms0(kh,kh,l->kn,D,c->eps);q38_rope(kh,D,c->rotary_dim,pos,c->theta);
            memcpy(m->K[layer]+((int64_t)h*m->kv_cap+pos)*D,kh,(size_t)D*sizeof(float));
            memcpy(m->V[layer]+((int64_t)h*m->kv_cap+pos)*D,vp+(int64_t)s*KVH*D+(int64_t)h*D,(size_t)D*sizeof(float));
        }
        memcpy(m->IK[layer]+(int64_t)pos*ID,ip+(int64_t)s*(IQ+1)*ID+(int64_t)IQ*ID,(size_t)ID*sizeof(float));
    }
    float *heads=falloc((int64_t)S*QH*D),*qidx=falloc((int64_t)IQ*ID),*pool=falloc(ID);
    int *selected=(int*)malloc((size_t)maxsel*sizeof(int));
    if(!selected){fprintf(stderr,"OOM QSA selection\n");exit(1);}
    for(int s=0;s<S;s++){
        int pos=pos_base+s,visible=pos+1,blocks=visible/R,tail=blocks*R;
        double phase_started=now_s();
        for(int h=0;h<IQ;h++){float *qh=qidx+(int64_t)h*ID;memcpy(qh,ip+(int64_t)s*(IQ+1)*ID+(int64_t)h*ID,(size_t)ID*sizeof(float));q38_rms0(qh,qh,l->idx_qn,ID,c->eps);q38_rope(qh,ID,c->rotary_dim,pos,c->theta);}
        int take=blocks<c->idx_budget/R?blocks:c->idx_budget/R,nsel=0;
        Q38Block *rank=blocks?(Q38Block*)malloc((size_t)blocks*sizeof(Q38Block)):NULL;
        if(blocks&&!rank){fprintf(stderr,"OOM QSA block ranking\n");exit(1);}
        for(int b=0;b<blocks;b++){
            memset(pool,0,(size_t)ID*sizeof(float));for(int r=0;r<R;r++){const float *raw=m->IK[layer]+(int64_t)(b*R+r)*ID;for(int d=0;d<ID;d++)pool[d]+=raw[d]/R;}
            q38_rms0(pool,pool,l->idx_kn,ID,c->eps);q38_rope(pool,ID,c->rotary_dim,b*R,c->theta);
            float score=0.f;for(int h=0;h<IQ;h++){float a=0.f;for(int d=0;d<ID;d++)a+=qidx[(int64_t)h*ID+d]*pool[d];if(a>0.f)score+=a;}rank[b]=(Q38Block){score/sqrtf((float)ID),b};
        }
        if(blocks)qsort(rank,(size_t)blocks,sizeof(Q38Block),q38_block_desc);
        for(int z=0;z<take;z++)for(int r=0;r<R;r++)selected[nsel++]=rank[z].block*R+r;
        for(int t=tail;t<visible;t++)selected[nsel++]=t;free(rank);
        q38_tm_add(m,Q38_TM_QSA_INDEX,phase_started); phase_started=now_s();
        for(int h=0;h<QH;h++){
            float *qraw=qp+(int64_t)s*QH*2*D+(int64_t)h*2*D;
            float *qh=falloc(D);memcpy(qh,qraw,(size_t)D*sizeof(float));q38_rms0(qh,qh,l->qn,D,c->eps);q38_rope(qh,D,c->rotary_dim,pos,c->theta);
            float *score=falloc(nsel);float mx=-INFINITY;
            int khidx=h/(QH/KVH);for(int j=0;j<nsel;j++){const float *kh=m->K[layer]+((int64_t)khidx*m->kv_cap+selected[j])*D;float a=0.f;for(int d=0;d<D;d++)a+=qh[d]*kh[d];a/=sqrtf((float)D);score[j]=a;if(a>mx)mx=a;}
            float den=0.f;for(int j=0;j<nsel;j++){score[j]=expf(score[j]-mx);den+=score[j];}
            float *oh=heads+(int64_t)s*QH*D+(int64_t)h*D;memset(oh,0,(size_t)D*sizeof(float));
            for(int j=0;j<nsel;j++){float a=score[j]/den;const float *vh=m->V[layer]+((int64_t)khidx*m->kv_cap+selected[j])*D;for(int d=0;d<D;d++)oh[d]+=a*vh[d];}
            for(int d=0;d<D;d++)oh[d]*=q38_sigmoid(qraw[D+d]);free(qh);free(score);
        }
        q38_tm_add(m,Q38_TM_QSA_ATTENTION,phase_started);
    }
    q38_dense_matmul(m,out,heads,&l->o,S,QH*D,H);
    free(qp);free(kp);free(vp);free(ip);free(heads);free(qidx);free(pool);free(selected);
}

/* The single-row path is intentionally kept separate from prefill.  Decode is
 * the latency-sensitive steady state and should not pay for route tables or a
 * prompt-sized workspace. */
static void q38_moe_decode(Model *m,Layer *l,int layer,const float *x,int S,float *out) {
    Cfg *c=&m->c;int H=c->hidden,E=c->experts,K=c->topk,I=c->inter,SI=c->shared_inter;
    float *logits=falloc(E),*sg=falloc(SI),*su=falloc(SI),*sh=falloc(SI),*shared=falloc(H);
    float *eg=falloc(I),*eu=falloc(I),*eh=falloc(I),*eo=falloc(H);
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*H;float *ys=out+(int64_t)s*H;memset(ys,0,(size_t)H*sizeof(float));
        q38_dense_matmul(m,logits,xs,&l->router,1,H,E);float mx=logits[0];for(int e=1;e<E;e++)if(logits[e]>mx)mx=logits[e];
        double all=0;for(int e=0;e<E;e++){logits[e]=expf(logits[e]-mx);all+=logits[e];}
        int idx[Q38_MAX_TOPK];float val[Q38_MAX_TOPK];
        for(int z=0;z<K;z++){int best=-1;float bv=-1.f;for(int e=0;e<E;e++){int used=0;for(int j=0;j<z;j++)if(idx[j]==e)used=1;if(!used&&logits[e]>bv){bv=logits[e];best=e;}}idx[z]=rt_router_pick(best,z,E,layer);val[z]=logits[idx[z]];}
        double top=0;for(int z=0;z<K;z++)top+=val[z];double den=c->norm_topk?top:all;
        float route_gates[Q38_MAX_TOPK];
        for(int z=0;z<K;z++) route_gates[z]=(float)(val[z]/den);
        rt_route(layer,s,idx,route_gates,K); /* shared counts + post-normalization trace */
        q38_prefetch_native_fp8_experts(m,layer,idx,K);
        double phase_started=now_s();
        q38_weight_matmul(sg,xs,&l->sh_g,1,H,SI);q38_weight_matmul(su,xs,&l->sh_u,1,H,SI);
        for(int j=0;j<SI;j++)sh[j]=q38_silu(sg[j])*su[j];q38_weight_matmul(shared,sh,&l->sh_d,1,SI,H);
        float gate=0.f;for(int d=0;d<H;d++)gate+=xs[d]*l->sh_gate[d];gate=q38_sigmoid(gate);
        q38_tm_add(m,Q38_TM_SHARED_EXPERT,phase_started);
        Slot *selected[Q38_MAX_TOPK];
        int loaded_batch=q38_expert_get_batch(m,layer,idx,K,selected);
        for(int z=0;z<K;z++){
            Slot *ex=loaded_batch?selected[z]:q38_expert_get(m,layer,idx[z]);phase_started=now_s();
            q38_weight_matmul(eg,xs,&ex->gate,1,H,I);q38_weight_matmul(eu,xs,&ex->up,1,H,I);
            for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_weight_matmul(eo,eh,&ex->down,1,I,H);
            for(int d=0;d<H;d++)ys[d]+=route_gates[z]*eo[d];
            q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
        }
        for(int d=0;d<H;d++)ys[d]+=gate*shared[d];
    }
    rt_trace_end();
    free(logits);free(sg);free(su);free(sh);free(shared);free(eg);free(eu);free(eh);free(eo);
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

/* Prefill MoE: route a bounded row chunk first, then execute each distinct
 * expert's assignments as one batched SwiGLU.  Grouping is an I/O optimization
 * only.  Expert outputs are placed back in assignment order and the final
 * weighted reduction still visits rank 0..top-k-1 for every row, preserving the
 * decode path's floating-point accumulation order. */
static void q38_moe_prefill(Model *m,Layer *l,int layer,const float *x,
                            int S,float *out) {
    Cfg *c=&m->c;
    int H=c->hidden,E=c->experts,K=c->topk,I=c->inter,SI=c->shared_inter;
    int rows_capacity=q38_moe_prefill_rows(c,S);
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
        q38_dense_matmul(m,logits,x+(int64_t)base*H,&l->router,rows,H,E);
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

        /* One advice range per distinct expert is enough for this chunk. */
        q38_prefetch_native_fp8_experts(m,layer,unique,unique_count);

        /* Shared expert work is independent across rows and remains resident;
         * batching it here also keeps its cost out of the routed groups. */
        double phase_started=now_s();
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
        if(load_limit>Q38_MAX_TOPK)load_limit=Q38_MAX_TOPK;
        if(load_limit<1)load_limit=1;
        for(int unique_base=0;unique_base<unique_count;) {
            int load_count=unique_count-unique_base;
            if(load_count>load_limit)load_count=load_limit;
            int loaded_batch=load_count>=2&&q38_expert_get_batch(
                m,layer,unique+unique_base,load_count,batch_slots);
            for(int offset=0;offset<load_count;offset++) {
                int e=unique[unique_base+offset];
                int count=group_counts[e];
                Slot *expert=loaded_batch?batch_slots[offset]:
                                          q38_expert_get(m,layer,e);
                int first=group_offsets[e];
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

        /* The grouped execution above is intentionally not the reduction
         * order.  Replaying the original top-k sequence gives the same result
         * as the single-row implementation for F32, BF16 and block-FP8. */
        for(int s=0;s<rows;s++) {
            float *ys=out+(int64_t)(base+s)*H;
            for(int rank=0;rank<K;rank++) {
                int assignment=s*K+rank;
                const float *expert_output=routed_out+
                    (int64_t)assignment_positions[assignment]*H;
                float gate=routes[assignment].gate;
                for(int d=0;d<H;d++)ys[d]+=gate*expert_output[d];
            }
            const float *shared=shared_out+(int64_t)s*H;
            for(int d=0;d<H;d++)ys[d]+=shared_gate[s]*shared[d];
        }
        base+=rows;
    }
    rt_trace_end();
    free(logits);free(shared_gate);free(shared_g);free(shared_u);
    free(shared_hidden);free(shared_out);free(expert_input);free(expert_gate);
    free(expert_up);free(routed_out);free(routes);free(assignments);
    free(assignment_positions);free(group_counts);free(group_offsets);
    free(group_cursor);free(unique);free(batch_slots);
}

static void q38_moe(Model *m,Layer *l,int layer,const float *x,int S,float *out) {
    if(S<=1||!m->prefill_batch)q38_moe_decode(m,l,layer,x,S,out);
    else q38_moe_prefill(m,l,layer,x,S,out);
}

static void reset_recurrent(Model *m) {
    Cfg *c=&m->c;
    for(int i=0;i<c->layers;i++)if(!c->is_attn[i]){
        memset(m->DN_rec[i],0,(size_t)c->dn_vheads*c->dn_kdim*c->dn_vdim*sizeof(float));
        memset(m->DN_conv[i],0,(size_t)c->dn_conv_dim*(c->dn_convk-1)*sizeof(float));
    }
    memset(m->PLE_conv_state,0,(size_t)c->hc_width*(c->ple_convk-1)*c->ngram_size*sizeof(float));m->ple_history_len=0;
}

static void ensure_kv(Model *m) {
    Cfg *c=&m->c;if(m->max_t<=m->kv_cap&&m->K)return;
    if(m->K){for(int i=0;i<c->layers;i++){free(m->K[i]);free(m->V[i]);free(m->IK[i]);}free(m->K);free(m->V);free(m->IK);}
    m->K=(float**)calloc((size_t)c->layers,sizeof(float*));m->V=(float**)calloc((size_t)c->layers,sizeof(float*));m->IK=(float**)calloc((size_t)c->layers,sizeof(float*));
    for(int i=0;i<c->layers;i++)if(c->is_attn[i]){
        m->K[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->V[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->IK[i]=falloc((int64_t)m->max_t*c->idx_dim);
    }
    m->kv_cap=m->max_t;
}

/* Run only the requested native layer interval over hyper-residual activations.
 * Segment callers supply the four-stream boundary state directly; unlike step,
 * this path deliberately does not gather embeddings or apply the final mixer. */
static void q38_layers_forward_range(Model *m,float *hyper,const int *ids,
                                     int S,int pos_base,int layer_begin,
                                     int layer_end) {
    Cfg *c=&m->c; int H=c->hidden,W=c->hc_width,C=c->hc_count;
    float *mixed=falloc((int64_t)S*H),*inject=falloc((int64_t)S*C),*block=falloc((int64_t)S*H);
    for(int i=layer_begin;i<layer_end;i++){
        Layer *l=&m->L[i];
        if(i==c->ple_layer){
            float *ple=falloc((int64_t)S*W); q38_ple(m,ids,S,hyper,ple);
            for(int64_t z=0;z<(int64_t)S*W;z++) hyper[z]+=ple[z];
            free(ple);
        }
        q38_gr_read(m,&l->attn_gr,hyper,S,mixed,inject);
        if(c->is_attn[i]) q38_attention(m,l,i,mixed,S,pos_base,block);
        else q38_deltanet(m,l,i,mixed,S,block);
        q38_gr_apply(c,hyper,block,inject,S);
        q38_gr_read(m,&l->mlp_gr,hyper,S,mixed,inject);
        q38_moe(m,l,i,mixed,S,block);
        q38_gr_apply(c,hyper,block,inject,S);
    }
    free(mixed); free(inject); free(block);
}

static float *step(Model *m,const int *ids,int S,int pos_base) {
    Cfg *c=&m->c;int H=c->hidden,W=c->hc_width,C=c->hc_count;
    m->timers.forwards++;
    float *hyper=falloc((int64_t)S*W);
    for(int s=0;s<S;s++){
        if(ids[s]<0||ids[s]>=c->vocab){fprintf(stderr,"token id %d outside vocabulary\n",ids[s]);exit(1);}
        float *e=hyper+(int64_t)s*W;q38_weight_row(&m->embed,ids[s],e);
        for(int b=1;b<C;b++)memcpy(e+(int64_t)b*H,e,(size_t)H*sizeof(float));
    }
    float *mixed=falloc((int64_t)S*H),*inject=falloc((int64_t)S*C),*block=falloc((int64_t)S*H);
    for(int i=0;i<c->layers;i++){
        Layer *l=&m->L[i];
        if(i==c->ple_layer){float *ple=falloc((int64_t)S*W);q38_ple(m,ids,S,hyper,ple);for(int64_t z=0;z<(int64_t)S*W;z++)hyper[z]+=ple[z];free(ple);}
        q38_gr_read(m,&l->attn_gr,hyper,S,mixed,inject);
        if(c->is_attn[i])q38_attention(m,l,i,mixed,S,pos_base,block);else q38_deltanet(m,l,i,mixed,S,block);
        q38_gr_apply(c,hyper,block,inject,S);
        q38_gr_read(m,&l->mlp_gr,hyper,S,mixed,inject);q38_moe(m,l,i,mixed,S,block);q38_gr_apply(c,hyper,block,inject,S);
    }
    q38_gr_read(m,&m->final_gr,hyper,S,mixed,NULL);m->kv_len=pos_base+S;
    float *logit=falloc(c->vocab);double phase_started=now_s();
    q38_weight_matmul(logit,mixed+(int64_t)(S-1)*H,&m->lm_head,1,H,c->vocab);
    q38_tm_add(m,Q38_TM_LM_HEAD,phase_started);
    free(hyper);free(mixed);free(inject);free(block);return logit;
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

static void tm_report(const Model *m) {
    q38_tm_report_bank(&m->timers,"total");
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
    for(int i=0;i<m->c.layers;i++) {
        if(m->L)q38_layer_free(&m->L[i]);
        if(m->cache) {
            if(m->cache[i].slots) {
                for(int s=0;s<m->cache[i].n;s++) {
                    q38_weight_free(&m->cache[i].slots[s].gate);
                    q38_weight_free(&m->cache[i].slots[s].up);
                    q38_weight_free(&m->cache[i].slots[s].down);
                    free(m->cache[i].slots[s].fp8_slab);
                }
            }
            free(m->cache[i].slots); free(m->cache[i].by_expert);
        }
        if(m->expert_scales)free(m->expert_scales[i].values);
        free(m->DN_rec ? m->DN_rec[i] : NULL); free(m->DN_conv ? m->DN_conv[i] : NULL);
        free(m->K ? m->K[i] : NULL); free(m->V ? m->V[i] : NULL); free(m->IK ? m->IK[i] : NULL);
    }
    free(m->L); free(m->cache); free(m->expert_scales); free(m->DN_rec); free(m->DN_conv); free(m->K); free(m->V); free(m->IK);
    q38_weight_free(&m->embed);q38_weight_free(&m->lm_head);
    free(m->final_gr.norm);q38_weight_free(&m->final_gr.down);q38_weight_free(&m->final_gr.up);q38_weight_free(&m->final_gr.inject);
    free(m->ple_history); free(m->PLE_conv_state); free(m->c.is_attn); st_destroy(&m->S);
    memset(m,0,sizeof(*m));
}

#endif /* COLI_QWEN38_CORE_H */
