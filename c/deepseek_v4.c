/* DeepSeek-V4 inference engine -- sibling of glm.c and kimi_k3.c.
 *
 * The first milestone deliberately stops after strict checkpoint discovery.
 * DeepSeek-V4 is not DeepSeek-V3 with renamed tensors: mHC, Sparse MLA and
 * native MXFP4 experts all change the forward pass.  Refuse incompatible
 * configs here instead of producing plausible-looking garbage later.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "json.h"
#include "st.h"
#include "tok.h"
#include "dsv4_mhc.h"
#include "dsv4_quant.h"
#include "quant.h"
#ifdef COLI_CUDA
#include "backend_cuda_dsv4.h"
#endif

#define DSV4_MAX_CONTEXT 2048

typedef struct {
    int hidden, layers, vocab;
    int heads, head_dim, kv_heads;
    int q_lora, o_lora, qk_rope;
    int index_heads, index_head_dim, index_topk, hash_layers;
    int experts, topk, moe_inter, shared_experts;
    int hc_mult, hc_iters;
    int dspark_block, dspark_rank, dspark_noise;
    int window, bos, eos;
    float eps, hc_eps, routed_scale, swiglu_limit;
    float rope_theta, rope_factor, beta_fast, beta_slow;
    int rope_original;
    float compress_theta;
    int compress[256];
} Dsv4Cfg;

static void die(const char *msg);
static double prof_mhc,prof_attn,prof_route,prof_routed,prof_shared,prof_head;
static double wall_time(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void profile_print(void){fprintf(stderr,"[DSV4 PROFILE] mHC %.3fs attention %.3fs router %.3fs routed %.3fs shared %.3fs head %.3fs\n",
    prof_mhc,prof_attn,prof_route,prof_routed,prof_shared,prof_head);}

typedef struct {
    float *kv, *compressed, *comp_value, *comp_score;
    int *compressed_len;
    int len, max_compressed;
#ifdef COLI_CUDA
    Dsv4CudaKvCache **gpu_kv,**gpu_peer_kv;
    Dsv4CudaExpertSet **gpu_experts;
    Dsv4CudaGraph **gpu_ffn_graph;
    Dsv4CudaGraph *gpu_segment_graph[16];
    Dsv4CudaActivation **gpu_residual,**gpu_next,**gpu_mhc,**gpu_input,**gpu_block;
    Dsv4CudaActivation **gpu_peer_residual,**gpu_peer_next,**gpu_peer_mhc,**gpu_peer_input,**gpu_peer_block;
    int gpu_max_tokens,gpu_layers;
#endif
} Dsv4State;

static Dsv4State state_init(const Dsv4Cfg *c,int max_tokens){
    Dsv4State st={0};st.max_compressed=(max_tokens+3)/4;
    st.kv=calloc((size_t)c->layers*c->window*c->head_dim,sizeof(float));
    st.compressed=calloc((size_t)c->layers*st.max_compressed*c->head_dim,sizeof(float));
    st.comp_value=calloc((size_t)c->layers*2*c->window*c->head_dim,sizeof(float));
    st.comp_score=malloc((size_t)c->layers*2*c->window*c->head_dim*sizeof(float));
    st.compressed_len=calloc((size_t)c->layers,sizeof(int));
    if(!st.kv||!st.compressed||!st.comp_value||!st.comp_score||!st.compressed_len)die("out of memory allocating KV state");
    long n=(long)c->layers*2*c->window*c->head_dim;for(long i=0;i<n;i++)st.comp_score[i]=-INFINITY;
#ifdef COLI_CUDA
    st.gpu_kv=calloc((size_t)c->layers,sizeof(*st.gpu_kv));
    st.gpu_peer_kv=calloc((size_t)c->layers,sizeof(*st.gpu_peer_kv));
    st.gpu_experts=calloc((size_t)c->layers,sizeof(*st.gpu_experts));
    st.gpu_ffn_graph=calloc((size_t)c->layers,sizeof(*st.gpu_ffn_graph));
    st.gpu_residual=calloc((size_t)c->layers,sizeof(*st.gpu_residual));st.gpu_next=calloc((size_t)c->layers,sizeof(*st.gpu_next));
    st.gpu_mhc=calloc((size_t)c->layers,sizeof(*st.gpu_mhc));st.gpu_input=calloc((size_t)c->layers,sizeof(*st.gpu_input));
    st.gpu_block=calloc((size_t)c->layers,sizeof(*st.gpu_block));st.gpu_peer_residual=calloc((size_t)c->layers,sizeof(*st.gpu_peer_residual));
    st.gpu_peer_next=calloc((size_t)c->layers,sizeof(*st.gpu_peer_next));st.gpu_peer_mhc=calloc((size_t)c->layers,sizeof(*st.gpu_peer_mhc));
    st.gpu_peer_input=calloc((size_t)c->layers,sizeof(*st.gpu_peer_input));st.gpu_peer_block=calloc((size_t)c->layers,sizeof(*st.gpu_peer_block));st.gpu_max_tokens=max_tokens;st.gpu_layers=c->layers;
    if(!st.gpu_kv||!st.gpu_peer_kv||!st.gpu_experts||!st.gpu_ffn_graph||!st.gpu_residual||!st.gpu_next||!st.gpu_mhc||!st.gpu_input||!st.gpu_block||!st.gpu_peer_residual||!st.gpu_peer_next||!st.gpu_peer_mhc||!st.gpu_peer_input||!st.gpu_peer_block)die("out of memory allocating GPU layer state");
#endif
    return st;
}

static void state_free(Dsv4State *st){
#ifdef COLI_CUDA
    if(st->gpu_kv){for(int l=0;l<st->gpu_layers;l++)dsv4_cuda_kv_free(st->gpu_kv[l]);free(st->gpu_kv);}
    if(st->gpu_peer_kv){for(int l=0;l<st->gpu_layers;l++)dsv4_cuda_kv_free(st->gpu_peer_kv[l]);free(st->gpu_peer_kv);}
    free(st->gpu_experts);
    if(st->gpu_ffn_graph){for(int l=0;l<st->gpu_layers;l++)dsv4_cuda_graph_free(st->gpu_ffn_graph[l]);free(st->gpu_ffn_graph);}
    for(int d=0;d<16;d++)dsv4_cuda_graph_free(st->gpu_segment_graph[d]);
    Dsv4CudaActivation ***arrays[]={&st->gpu_residual,&st->gpu_next,&st->gpu_mhc,&st->gpu_input,&st->gpu_block,&st->gpu_peer_residual,&st->gpu_peer_next,&st->gpu_peer_mhc,&st->gpu_peer_input,&st->gpu_peer_block};
    for(size_t a=0;a<sizeof(arrays)/sizeof(arrays[0]);a++)if(*arrays[a]){for(int l=0;l<st->gpu_layers;l++)dsv4_cuda_activation_free((*arrays[a])[l]);free(*arrays[a]);}
#endif
    free(st->kv);free(st->compressed);free(st->comp_value);free(st->comp_score);free(st->compressed_len);memset(st,0,sizeof(*st));
}

static void die(const char *msg){ fprintf(stderr,"deepseek_v4: %s\n",msg); exit(1); }

#ifdef COLI_CUDA
typedef struct { char *name;Dsv4CudaTensor *t;int device; } GpuWeight;
typedef struct { char *name;uint8_t *w,*scale; } CpuWeight;
typedef struct {
    Dsv4CudaTensor *attn_fn,*attn_scale,*attn_base;
    Dsv4CudaTensor *ffn_fn,*ffn_scale,*ffn_base,*ffn_norm,*gate,*bias;
} TpLayerWeights;
static GpuWeight *gw_cache;static int gw_n,gw_cap,gpu_dev[16],gpu_ndev;
static int gpu_layer_count;
static Dsv4CudaExpertSet *ge_cache[256],*ge_cache_peer[256];
static Dsv4CudaAttentionWeights tp_attention[256][2];
static Dsv4CudaTensor *batch_qkv[256];
static TpLayerWeights tp_layer[256];
static CpuWeight *cw_cache;static int cw_n,cw_cap;static long long cw_bytes;
static int gpu_ep2(void){const char*p=getenv("DSV4_CUDA_EP2");return p&&atoi(p)&&gpu_ndev==6;}
static int gpu_replicated_tp2(void){const char*p=getenv("DSV4_CUDA_REPLICATED_TP2");return gpu_ep2()&&p&&atoi(p);}
static int gpu_stage(int layer){return gpu_ep2()?2*(int)((long long)layer*3/gpu_layer_count):(int)((long long)layer*gpu_ndev/gpu_layer_count);}
static int gpu_device_for(const char *name){int l=-1,e=-1;if(sscanf(name,"layers.%d.ffn.experts.%d.",&l,&e)==2){
        if(gpu_ep2())return gpu_dev[gpu_stage(l)+(e>=128)];
        const char*p=getenv("DSV4_CUDA_EXPERT_PARALLEL");if(p&&atoi(p))return gpu_dev[e%gpu_ndev];}
    if(gpu_ep2()&&sscanf(name,"layers.%d.",&l)==1)return gpu_dev[gpu_stage(l)];
    if(sscanf(name,"layers.%d.",&l)==1){const char*p=getenv("DSV4_CUDA_PIPELINE_LAYOUT");int pipeline=p?atoi(p):1;return gpu_dev[pipeline?((long long)l*gpu_ndev/gpu_layer_count):l%gpu_ndev];}
    int tail=!strncmp(name,"head.",5)||!strncmp(name,"hc_head_",8)||!strcmp(name,"norm.weight");
    return gpu_dev[tail?(gpu_ep2()?gpu_ndev-2:gpu_ndev-1):0];}
static Dsv4CudaTensor **gpu_slot(const char *name){for(int i=0;i<gw_n;i++)if(!strcmp(gw_cache[i].name,name))return &gw_cache[i].t;
    if(gw_n==gw_cap){gw_cap=gw_cap?gw_cap*2:256;gw_cache=realloc(gw_cache,(size_t)gw_cap*sizeof(*gw_cache));if(!gw_cache)die("out of memory indexing GPU weights");}
    gw_cache[gw_n]=(GpuWeight){strdup(name),NULL,gpu_device_for(name)};return &gw_cache[gw_n++].t;}
static Dsv4CudaTensor **gpu_slot_on(const char *name,int device){char key[320];snprintf(key,sizeof(key),"%s@cuda%d",name,device);
    for(int i=0;i<gw_n;i++)if(!strcmp(gw_cache[i].name,key))return &gw_cache[i].t;
    if(gw_n==gw_cap){gw_cap=gw_cap?gw_cap*2:256;gw_cache=realloc(gw_cache,(size_t)gw_cap*sizeof(*gw_cache));if(!gw_cache)die("out of memory indexing GPU weights");}
    gw_cache[gw_n]=(GpuWeight){strdup(key),NULL,device};return &gw_cache[gw_n++].t;}
static int gpu_boot(void){const char *p=getenv("COLI_GPUS");if(!p||!*p)p="0";while(*p&&gpu_ndev<16){char *e;long d=strtol(p,&e,10);if(e==p||d<0||d>255)die("invalid COLI_GPUS");gpu_dev[gpu_ndev++]=(int)d;p=*e==','?e+1:e;if(*e&&*e!=',')die("invalid COLI_GPUS");}return dsv4_cuda_init(gpu_dev,gpu_ndev);}
static CpuWeight *cpu_slot(const char *name){for(int i=0;i<cw_n;i++)if(!strcmp(cw_cache[i].name,name))return cw_cache+i;
    if(cw_n==cw_cap){cw_cap=cw_cap?cw_cap*2:256;cw_cache=realloc(cw_cache,(size_t)cw_cap*sizeof(*cw_cache));if(!cw_cache)die("out of memory indexing CPU weights");}
    cw_cache[cw_n]=(CpuWeight){strdup(name),NULL,NULL};return cw_cache+cw_n++;}
static void gpu_shutdown(void){for(int l=0;l<gpu_layer_count;l++){dsv4_cuda_expert_set_free(ge_cache[l]);dsv4_cuda_expert_set_free(ge_cache_peer[l]);}for(int i=0;i<gw_n;i++){dsv4_cuda_tensor_free(gw_cache[i].t);free(gw_cache[i].name);}free(gw_cache);
    for(int i=0;i<cw_n;i++){free(cw_cache[i].name);free(cw_cache[i].w);free(cw_cache[i].scale);}free(cw_cache);dsv4_cuda_shutdown();}
static void gpu_report(void){long long bytes[16]={0};for(int i=0;i<gw_n;i++)for(int d=0;d<gpu_ndev;d++)if(gw_cache[i].device==gpu_dev[d])bytes[d]+=dsv4_cuda_tensor_bytes(gw_cache[i].t);
    for(int d=0;d<gpu_ndev;d++)fprintf(stderr,"[DSV4 CUDA] device %d resident %.2f GiB\n",gpu_dev[d],bytes[d]/1073741824.0);}
#endif

static char *read_file(const char *path){
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"deepseek_v4: %s: %s\n",path,strerror(errno)); exit(1); }
    if(fseek(f,0,SEEK_END)) die("cannot seek config.json");
    long n=ftell(f);
    if(n<2 || n>(64L<<20)) die("invalid config.json size");
    rewind(f);
    char *buf=malloc((size_t)n+1);
    if(!buf) die("out of memory reading config.json");
    if(fread(buf,1,(size_t)n,f)!=(size_t)n) die("short read from config.json");
    fclose(f); buf[n]=0; return buf;
}

static double number(jval *root, const char *key){
    jval *v=json_get(root,key);
    if(!v || v->t!=J_NUM){
        fprintf(stderr,"deepseek_v4: config.json missing numeric %s\n",key); exit(1);
    }
    return v->num;
}

static const char *string(jval *root, const char *key){
    jval *v=json_get(root,key);
    if(!v || v->t!=J_STR){
        fprintf(stderr,"deepseek_v4: config.json missing string %s\n",key); exit(1);
    }
    return v->str;
}

static Dsv4Cfg load_config(const char *model_dir){
    char path[4096];
    if(snprintf(path,sizeof(path),"%s/config.json",model_dir)>=(int)sizeof(path))
        die("model path is too long");
    char *text=read_file(path), *arena=NULL;
    jval *root=json_parse(text,&arena);
    if(strcmp(string(root,"model_type"),"deepseek_v4")) die("model_type is not deepseek_v4");
    if(strcmp(string(root,"expert_dtype"),"fp4")) die("only the native FP4 checkpoint is supported");

    Dsv4Cfg c={0};
    c.hidden=(int)number(root,"hidden_size");
    c.layers=(int)number(root,"num_hidden_layers");
    c.vocab=(int)number(root,"vocab_size");
    c.heads=(int)number(root,"num_attention_heads");
    c.head_dim=(int)number(root,"head_dim");
    c.kv_heads=(int)number(root,"num_key_value_heads");
    c.q_lora=(int)number(root,"q_lora_rank");
    c.o_lora=(int)number(root,"o_lora_rank");
    c.qk_rope=(int)number(root,"qk_rope_head_dim");
    c.index_heads=(int)number(root,"index_n_heads");
    c.index_head_dim=(int)number(root,"index_head_dim");
    c.index_topk=(int)number(root,"index_topk");
    c.hash_layers=(int)number(root,"num_hash_layers");
    c.experts=(int)number(root,"n_routed_experts");
    c.topk=(int)number(root,"num_experts_per_tok");
    c.moe_inter=(int)number(root,"moe_intermediate_size");
    c.shared_experts=(int)number(root,"n_shared_experts");
    c.hc_mult=(int)number(root,"hc_mult");
    c.hc_iters=(int)number(root,"hc_sinkhorn_iters");
    c.dspark_block=(int)number(root,"dspark_block_size");
    c.dspark_rank=(int)number(root,"dspark_markov_rank");
    c.dspark_noise=(int)number(root,"dspark_noise_token_id");
    c.eps=(float)number(root,"rms_norm_eps");
    c.hc_eps=(float)number(root,"hc_eps");
    c.routed_scale=(float)number(root,"routed_scaling_factor");
    c.swiglu_limit=(float)number(root,"swiglu_limit");
    c.window=(int)number(root,"sliding_window");
    c.rope_theta=(float)number(root,"rope_theta");
    c.compress_theta=(float)number(root,"compress_rope_theta");
    c.bos=(int)number(root,"bos_token_id");c.eos=(int)number(root,"eos_token_id");
    jval *rs=json_get(root,"rope_scaling");if(!rs||rs->t!=J_OBJ)die("missing rope_scaling");
    c.rope_factor=(float)number(rs,"factor");c.beta_fast=(float)number(rs,"beta_fast");
    c.beta_slow=(float)number(rs,"beta_slow");c.rope_original=(int)number(rs,"original_max_position_embeddings");
    jval *cr=json_get(root,"compress_ratios");if(!cr||cr->t!=J_ARR||cr->len<c.layers)die("invalid compress_ratios");
    for(int i=0;i<c.layers;i++)c.compress[i]=(int)cr->kids[i]->num;
    free(text); (void)arena;

    if(c.hidden<1 || c.hidden>65536 || c.layers<1 || c.layers>256 ||
       c.vocab<1 || c.vocab>(1<<22) || c.heads<1 || c.heads>512 ||
       c.head_dim<1 || c.head_dim>1024 || c.kv_heads<1 || c.kv_heads>c.heads ||
       c.q_lora<1 || c.o_lora<1 || c.qk_rope<1 || c.qk_rope>c.head_dim ||
       c.index_heads<1 || c.index_head_dim<1 || c.index_topk<1 ||
       c.experts<1 || c.experts>4096 || c.topk<1 || c.topk>c.experts ||
       c.moe_inter<32 || c.moe_inter%32 || c.shared_experts<0 ||
       c.hc_mult<1 || c.hc_mult>16 || c.hc_iters<1 || c.hc_iters>128 ||
       c.dspark_block<1 || c.dspark_rank<1 || c.dspark_noise<0)
        die("configuration dimensions are out of range");
    return c;
}

static void require_tensor(shards *s, const char *name, long *count, long long *bytes){
    st_tensor *t=st_find(s,name);
    if(!t) st_die_missing(s,name);
    if(t->numel<1 || t->nbytes<1){
        fprintf(stderr,"deepseek_v4: tensor %s is empty\n",name); exit(1);
    }
    (*count)++; *bytes+=t->nbytes;
}

static void require_layer_tensor(shards *s, int layer, const char *suffix,
                                 long *count, long long *bytes){
    char name[256];
    snprintf(name,sizeof(name),"layers.%d.%s",layer,suffix);
    require_tensor(s,name,count,bytes);
}

static void validate_checkpoint(shards *s, const char *model_dir, const Dsv4Cfg *c,
                                long *required, long long *required_bytes,
                                int *indexed){
    static const char *common[]={
        "hc_attn_base","hc_ffn_base","hc_attn_fn","hc_attn_scale",
        "hc_ffn_fn","hc_ffn_scale","attn.attn_sink",
        "attn.wq_a.weight","attn.wq_a.scale","attn.wq_b.weight","attn.wq_b.scale",
        "attn.q_norm.weight","attn.wo_a.weight","attn.wo_a.scale",
        "attn.wkv.weight","attn.wkv.scale","attn.kv_norm.weight",
        "attn.wo_b.weight","attn.wo_b.scale","attn_norm.weight","ffn_norm.weight",
        "ffn.shared_experts.w1.weight","ffn.shared_experts.w1.scale",
        "ffn.shared_experts.w3.weight","ffn.shared_experts.w3.scale",
        "ffn.shared_experts.w2.weight","ffn.shared_experts.w2.scale",
        "ffn.gate.weight"
    };
    static const char *compressor[]={
        "attn.compressor.ape","attn.compressor.wkv.weight",
        "attn.compressor.wgate.weight","attn.compressor.norm.weight"
    };
    static const char *indexer[]={
        "attn.indexer.wq_b.weight","attn.indexer.wq_b.scale",
        "attn.indexer.compressor.ape","attn.indexer.compressor.wkv.weight",
        "attn.indexer.compressor.wgate.weight","attn.indexer.compressor.norm.weight",
        "attn.indexer.weights_proj.weight"
    };
    static const char *root[]={
        "embed.weight","norm.weight","head.weight",
        "hc_head_base","hc_head_fn","hc_head_scale"
    };

    st_init(s,model_dir);
    *indexed=s->n; *required=0; *required_bytes=0;
    for(size_t i=0;i<sizeof(root)/sizeof(*root);i++)
        require_tensor(s,root[i],required,required_bytes);
    for(int l=0;l<c->layers;l++){
        for(size_t i=0;i<sizeof(common)/sizeof(*common);i++)
            require_layer_tensor(s,l,common[i],required,required_bytes);
        if(l<c->hash_layers)
            require_layer_tensor(s,l,"ffn.gate.tid2eid",required,required_bytes);
        else
            require_layer_tensor(s,l,"ffn.gate.bias",required,required_bytes);
        if(c->compress[l])
            for(size_t i=0;i<sizeof(compressor)/sizeof(*compressor);i++)
                require_layer_tensor(s,l,compressor[i],required,required_bytes);
        if(l>=c->hash_layers-1 && !(l&1))
            for(size_t i=0;i<sizeof(indexer)/sizeof(*indexer);i++)
                require_layer_tensor(s,l,indexer[i],required,required_bytes);
        for(int e=0;e<c->experts;e++) for(int w=1;w<=3;w++){
            char suffix[192];
            snprintf(suffix,sizeof(suffix),"ffn.experts.%d.w%d.weight",e,w);
            require_layer_tensor(s,l,suffix,required,required_bytes);
            snprintf(suffix,sizeof(suffix),"ffn.experts.%d.w%d.scale",e,w);
            require_layer_tensor(s,l,suffix,required,required_bytes);
        }
    }
}

static float *load_f32(shards *s,const char *name,long long want){
    st_tensor *t=st_find(s,name); if(!t) st_die_missing(s,name);
    if(want>=0&&t->numel!=want){
        fprintf(stderr,"deepseek_v4: %s numel %lld != %lld\n",name,(long long)t->numel,want); exit(1); }
    float *p=malloc((size_t)t->numel*sizeof(float)); if(!p) die("out of memory loading tensor");
    st_read_f32(s,name,p,0); return p;
}

static uint8_t *load_raw(shards *s,const char *name,long long want){
    st_tensor *t=st_find(s,name); if(!t) st_die_missing(s,name);
    if(want>=0&&t->nbytes!=want){
        fprintf(stderr,"deepseek_v4: %s bytes %lld != %lld\n",name,(long long)t->nbytes,want); exit(1); }
    uint8_t *p=malloc((size_t)t->nbytes); if(!p) die("out of memory loading raw tensor");
    st_read_raw(s,name,p,0); return p;
}

static void rmsnorm(float *y,const float *x,const float *w,int n,float eps){
    double ss=0; for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];
    float inv=1.f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++)y[i]=dsv4_bf16(x[i]*inv*w[i]);
}

static float rope_freq(const Dsv4Cfg *c,int pair,float base){
    int dim=c->qk_rope;float f=1.f/powf(base,(float)(2*pair)/dim);
    float low=floorf(dim*logf(c->rope_original/(c->beta_fast*2.f*(float)M_PI))/(2.f*logf(base)));
    float high=ceilf(dim*logf(c->rope_original/(c->beta_slow*2.f*(float)M_PI))/(2.f*logf(base)));
    if(low<0)low=0;if(high>dim-1)high=dim-1;
    float ramp=(pair-low)/(high-low);if(ramp<0)ramp=0;if(ramp>1)ramp=1;
    float smooth=1.f-ramp;return f/c->rope_factor*(1.f-smooth)+f*smooth;
}

static void rope_base(float *x,const Dsv4Cfg *c,int pos,int inverse,float base){
    int off=c->head_dim-c->qk_rope;
    for(int p=0;p<c->qk_rope/2;p++){float a=pos*rope_freq(c,p,base)*(inverse?-1.f:1.f);
        float cs=cosf(a),sn=sinf(a),u=x[off+2*p],v=x[off+2*p+1];
        x[off+2*p]=dsv4_bf16(u*cs-v*sn);x[off+2*p+1]=dsv4_bf16(u*sn+v*cs);}
}

static void rope(float *x,const Dsv4Cfg *c,int pos,int inverse){rope_base(x,c,pos,inverse,c->rope_theta);}

static void fp8_layer(shards *s,const char *stem,const float *x,int O,int I,float *y){
    char wn[256],sn[256]; snprintf(wn,sizeof(wn),"%s.weight",stem); snprintf(sn,sizeof(sn),"%s.scale",stem);
#ifdef COLI_CUDA
    Dsv4CudaTensor **slot=gpu_slot(wn);float *qx=malloc((size_t)I*sizeof(float));if(!qx)die("out of memory quantizing activation");
    memcpy(qx,x,(size_t)I*sizeof(float));dsv4_fp8_simulate(qx,I,128);
    if(!*slot){uint8_t *w=load_raw(s,wn,(long long)O*I),*sc=load_raw(s,sn,(long long)((O+127)/128)*((I+127)/128));
        int dev=gpu_device_for(wn);if(!dsv4_cuda_upload_fp8(slot,w,sc,O,I,dev))die("CUDA FP8 upload failed");free(w);free(sc);}
    if(!dsv4_cuda_matvec(*slot,y,qx))die("CUDA FP8 matvec failed");free(qx);
    { static int checks;if(getenv("DSV4_CUDA_CHECK")&&checks++<20){uint8_t*w=load_raw(s,wn,(long long)O*I),*sc=load_raw(s,sn,(long long)((O+127)/128)*((I+127)/128));
        float *ref=malloc((size_t)O*sizeof(float)),*cx=malloc((size_t)I*sizeof(float));memcpy(cx,x,(size_t)I*sizeof(float));dsv4_fp8_simulate(cx,I,128);dsv4_fp8_matvec(ref,cx,w,sc,O,I);
        float mx=0,mean=0;for(int i=0;i<O;i++){float d=fabsf(ref[i]-y[i]);if(d>mx)mx=d;mean+=d;}fprintf(stderr,"[DSV4 CUDA CHECK FP8] %s max=%g mean=%g\n",stem,mx,mean/O);free(w);free(sc);free(ref);free(cx);} }
#else
    uint8_t *w=load_raw(s,wn,(long long)O*I);
    uint8_t *sc=load_raw(s,sn,(long long)((O+127)/128)*((I+127)/128));
    float *qx=malloc((size_t)I*sizeof(float));if(!qx)die("out of memory quantizing activation");
    memcpy(qx,x,(size_t)I*sizeof(float));dsv4_fp8_simulate(qx,I,128);
    dsv4_fp8_matvec(y,qx,w,sc,O,I); free(qx);free(w); free(sc);
#endif
    for(int i=0;i<O;i++)y[i]=dsv4_bf16(y[i]);
}

static void bf16_layer(shards *s,const char *name,const float *x,int O,int I,float *y){
#ifdef COLI_CUDA
    Dsv4CudaTensor **slot=gpu_slot(name);if(!*slot){uint8_t *w=load_raw(s,name,(long long)O*I*2);int dev=gpu_device_for(name);
        if(!dsv4_cuda_upload_bf16(slot,(uint16_t*)w,O,I,dev))die("CUDA BF16 upload failed");free(w);}
    if(!dsv4_cuda_matvec(*slot,y,x))die("CUDA BF16 matvec failed");
#else
    float *w=load_f32(s,name,(long long)O*I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){float sum=0;for(int i=0;i<I;i++)sum+=x[i]*w[(long)o*I+i];y[o]=sum;}
    free(w);
#endif
}

static void compressor_step(shards *s,const Dsv4Cfg *c,int layer,int pos,const float *x,Dsv4State *state){
    int ratio=c->compress[layer];if(!state||!ratio)return;
    int H=c->hidden,D=c->head_dim,overlap=ratio==4,coff=1+overlap,O=coff*D;
    char name[256];float *v=malloc((size_t)O*sizeof(float)),*score=malloc((size_t)O*sizeof(float));
    if(!v||!score)die("out of memory in KV compressor");
    snprintf(name,sizeof(name),"layers.%d.attn.compressor.wkv.weight",layer);bf16_layer(s,name,x,O,H,v);
    snprintf(name,sizeof(name),"layers.%d.attn.compressor.wgate.weight",layer);bf16_layer(s,name,x,O,H,score);
    snprintf(name,sizeof(name),"layers.%d.attn.compressor.ape",layer);float *ape=load_f32(s,name,(long long)ratio*O);
    long base=(long)layer*2*c->window*D;int slot=(overlap?ratio:0)+pos%ratio;
    memcpy(state->comp_value+base+(long)slot*O,v,(size_t)O*sizeof(float));
    for(int d=0;d<O;d++)state->comp_score[base+(long)slot*O+d]=score[d]+ape[(long)(pos%ratio)*O+d];
    if((pos+1)%ratio==0){
        int items=overlap?2*ratio:ratio;float *out=malloc((size_t)D*sizeof(float));if(!out)die("out of memory pooling KV");
        for(int d=0;d<D;d++){float mx=-INFINITY;
            for(int j=0;j<items;j++){int col=overlap&&j>=ratio?D+d:d;float z=state->comp_score[base+(long)j*O+col];if(z>mx)mx=z;}
            float den=0,sum=0;for(int j=0;j<items;j++){int col=overlap&&j>=ratio?D+d:d;
                float a=expf(state->comp_score[base+(long)j*O+col]-mx);den+=a;sum+=a*state->comp_value[base+(long)j*O+col];}
            out[d]=sum/den;
        }
        snprintf(name,sizeof(name),"layers.%d.attn.compressor.norm.weight",layer);float *nw=load_f32(s,name,D);
        rmsnorm(out,out,nw,D,c->eps);rope_base(out,c,pos+1-ratio,0,c->compress_theta);dsv4_fp8_simulate(out,D-c->qk_rope,64);
        int ci=state->compressed_len[layer]++;if(ci>=state->max_compressed)die("compressed KV cache is full");
        memcpy(state->compressed+((long)layer*state->max_compressed+ci)*D,out,(size_t)D*sizeof(float));
        if(overlap){memcpy(state->comp_value+base,state->comp_value+base+(long)ratio*O,(size_t)ratio*O*sizeof(float));
            memcpy(state->comp_score+base,state->comp_score+base+(long)ratio*O,(size_t)ratio*O*sizeof(float));}
        free(nw);free(out);
    }
    free(ape);free(v);free(score);
}

static void mhc_pre(shards *s,const Dsv4Cfg *c,int layer,const char *kind,
                    const float *res,float *in,float *post,float *comb){
    int M=c->hc_mult,H=c->hidden,N=2*M+M*M,MH=M*H; char name[256];
    snprintf(name,sizeof(name),"layers.%d.hc_%s_fn",layer,kind);
    float *fn=load_f32(s,name,(long long)N*MH);
    snprintf(name,sizeof(name),"layers.%d.hc_%s_scale",layer,kind);
    float *scale=load_f32(s,name,3);
    snprintf(name,sizeof(name),"layers.%d.hc_%s_base",layer,kind);
    float *base=load_f32(s,name,N);
    dsv4_mhc_pre(res,fn,scale,base,M,H,c->eps,c->hc_eps,c->hc_eps,2.f,c->hc_iters,post,comb,in);
    free(fn);free(scale);free(base);
}

static float *initial_residual(shards *s,const Dsv4Cfg *c,int token){
    int M=c->hc_mult,H=c->hidden; float *embed=malloc((size_t)H*sizeof(float));
    float *res=malloc((size_t)M*H*sizeof(float)); if(!embed||!res)die("out of memory loading embedding");
    st_read_slice_f32(s,"embed.weight",(long long)token*H,H,embed,0);
    for(int i=0;i<M;i++)memcpy(res+(long)i*H,embed,(size_t)H*sizeof(float));
    free(embed); return res;
}

static void trace_first_mhc(shards *s,const Dsv4Cfg *c,int token,const char *path){
    if(token<0||token>=c->vocab) die("trace token id is outside vocabulary");
    int M=c->hc_mult,H=c->hidden;
    float *in=calloc((size_t)H,sizeof(float)),*post=malloc((size_t)M*sizeof(float));
    float *comb=malloc((size_t)M*M*sizeof(float));
    if(!in||!post||!comb) die("out of memory tracing mHC");
    float *res=initial_residual(s,c,token); mhc_pre(s,c,0,"attn",res,in,post,comb); free(res);
    FILE *f=fopen(path,"wb"); if(!f){ perror(path); exit(1); }
    fwrite(in,sizeof(float),(size_t)H,f); fwrite(post,sizeof(float),(size_t)M,f);
    fwrite(comb,sizeof(float),(size_t)M*M,f); fclose(f);
    printf("  mhc_trace=%s token=%d input0=%.9g post0=%.9g comb0=%.9g\n",
           path,token,in[0],post[0],comb[0]);
    free(in);free(post);free(comb);
}

#ifdef COLI_CUDA
static float *attention_first_cuda(const Dsv4Cfg *c,int layer,const float *in){
    char n[256];
#define AT(name, suffix) snprintf(n,sizeof(n),"layers.%d.%s",layer,suffix);Dsv4CudaTensor *name=*gpu_slot(n)
    AT(an,"attn_norm.weight");AT(qa,"attn.wq_a.weight");AT(qn,"attn.q_norm.weight");
    AT(qb,"attn.wq_b.weight");AT(kv,"attn.wkv.weight");AT(kvn,"attn.kv_norm.weight");
    AT(sink,"attn.attn_sink");AT(wa,"attn.wo_a.weight");AT(wb,"attn.wo_b.weight");
#undef AT
    int dev=gpu_device_for(n),H=c->hidden;Dsv4CudaActivation *x=dsv4_cuda_activation_create(dev,H),*y=dsv4_cuda_activation_create(dev,H);float*out=malloc((size_t)H*sizeof(float));
    if(!x||!y||!out||!dsv4_cuda_activation_upload(x,in,H)||!dsv4_cuda_attention_first(x,an,qa,qn,qb,kv,kvn,sink,wa,wb,c->heads,c->head_dim,c->qk_rope,8,c->eps,y)||!dsv4_cuda_activation_download(out,y,H)||!dsv4_cuda_activation_sync(y))die("CUDA first-position attention failed");
    dsv4_cuda_activation_free(x);dsv4_cuda_activation_free(y);return out;
}

static int attention_window_cuda_activation(const Dsv4Cfg *c,int layer,int pos,Dsv4State *state,
                                             const Dsv4CudaActivation *x,Dsv4CudaActivation *y){
    char n[256];
#define AT(name, suffix) snprintf(n,sizeof(n),"layers.%d.%s",layer,suffix);Dsv4CudaTensor *name=*gpu_slot(n)
    AT(an,"attn_norm.weight");AT(qa,"attn.wq_a.weight");AT(qn,"attn.q_norm.weight");AT(qb,"attn.wq_b.weight");
    AT(kv,"attn.wkv.weight");AT(kvn,"attn.kv_norm.weight");AT(sink,"attn.attn_sink");AT(wa,"attn.wo_a.weight");AT(wb,"attn.wo_b.weight");
#undef AT
    int ratio=c->compress[layer];Dsv4CudaTensor *cwkv=NULL,*cwg=NULL,*ape=NULL,*cnorm=NULL;if(ratio){snprintf(n,sizeof(n),"layers.%d.attn.compressor.wkv.weight",layer);cwkv=*gpu_slot(n);snprintf(n,sizeof(n),"layers.%d.attn.compressor.wgate.weight",layer);cwg=*gpu_slot(n);snprintf(n,sizeof(n),"layers.%d.attn.compressor.ape",layer);ape=*gpu_slot(n);snprintf(n,sizeof(n),"layers.%d.attn.compressor.norm.weight",layer);cnorm=*gpu_slot(n);}
    int dev=gpu_device_for(n),pairs=c->qk_rope/2;if(!state->gpu_kv[layer]){size_t rn=(size_t)state->gpu_max_tokens*pairs;float *cs=malloc(rn*sizeof(float)),*sn=malloc(rn*sizeof(float)),*cc=malloc(rn*sizeof(float)),*ss=malloc(rn*sizeof(float));if(!cs||!sn||!cc||!ss)die("out of memory preparing GPU RoPE");for(int p=0;p<state->gpu_max_tokens;p++)for(int k=0;k<pairs;k++){float a=p*rope_freq(c,k,c->rope_theta),b=p*rope_freq(c,k,c->compress_theta);cs[(long)p*pairs+k]=cosf(a);sn[(long)p*pairs+k]=sinf(a);cc[(long)p*pairs+k]=cosf(b);ss[(long)p*pairs+k]=sinf(b);}state->gpu_kv[layer]=dsv4_cuda_kv_create(dev,c->window,c->head_dim,state->gpu_max_tokens,pairs,cs,sn,cc,ss);free(cs);free(sn);free(cc);free(ss);if(!state->gpu_kv[layer])die("CUDA KV cache allocation failed");}
    if(gpu_ep2())return dsv4_cuda_attention_window_tp2(x,state->gpu_peer_input[layer],&tp_attention[layer][0],&tp_attention[layer][1],
        ratio,c->heads,c->head_dim,c->qk_rope,8,pos,c->eps,state->gpu_kv[layer],state->gpu_peer_kv[layer],y,state->gpu_peer_block[layer]);
    return dsv4_cuda_attention_window(x,an,qa,qn,qb,kv,kvn,sink,wa,wb,cwkv,cwg,ape,cnorm,ratio,c->heads,c->head_dim,c->qk_rope,8,pos,c->eps,state->gpu_kv[layer],y);
}

static void gpu_state_prepare(const Dsv4Cfg *c,Dsv4State *state){
    int H=c->hidden,M=c->hc_mult,MH=M*H,S=2*M+M*M,pairs=c->qk_rope/2;
    size_t rn=(size_t)state->gpu_max_tokens*pairs;float *cs=malloc(rn*sizeof(float)),*sn=malloc(rn*sizeof(float)),*cc=malloc(rn*sizeof(float)),*ss=malloc(rn*sizeof(float));
    if(!cs||!sn||!cc||!ss)die("out of memory preparing GPU state");
    for(int p=0;p<state->gpu_max_tokens;p++)for(int k=0;k<pairs;k++){float a=p*rope_freq(c,k,c->rope_theta),b=p*rope_freq(c,k,c->compress_theta);cs[(long)p*pairs+k]=cosf(a);sn[(long)p*pairs+k]=sinf(a);cc[(long)p*pairs+k]=cosf(b);ss[(long)p*pairs+k]=sinf(b);}
    for(int l=0;l<c->layers;l++){
        char n[64];snprintf(n,sizeof(n),"layers.%d.attn_norm.weight",l);int dev=gpu_device_for(n);
        state->gpu_residual[l]=dsv4_cuda_activation_create(dev,MH);state->gpu_next[l]=dsv4_cuda_activation_create(dev,MH);
        state->gpu_mhc[l]=dsv4_cuda_activation_create(dev,S);state->gpu_input[l]=dsv4_cuda_activation_create(dev,H);state->gpu_block[l]=dsv4_cuda_activation_create(dev,H);
        state->gpu_kv[l]=dsv4_cuda_kv_create(dev,c->window,c->head_dim,state->gpu_max_tokens,pairs,cs,sn,cc,ss);
        if(gpu_ep2()){int peer=gpu_dev[gpu_stage(l)+1];state->gpu_peer_residual[l]=dsv4_cuda_activation_create(peer,MH);state->gpu_peer_next[l]=dsv4_cuda_activation_create(peer,MH);
            state->gpu_peer_mhc[l]=dsv4_cuda_activation_create(peer,S);state->gpu_peer_input[l]=dsv4_cuda_activation_create(peer,H);state->gpu_peer_block[l]=dsv4_cuda_activation_create(peer,H);
            state->gpu_peer_kv[l]=dsv4_cuda_kv_create(peer,c->window,c->head_dim,state->gpu_max_tokens,pairs,cs,sn,cc,ss);}
        if(!state->gpu_residual[l]||!state->gpu_next[l]||!state->gpu_mhc[l]||!state->gpu_input[l]||!state->gpu_block[l]||!state->gpu_kv[l]||
           (gpu_ep2()&&(!state->gpu_peer_residual[l]||!state->gpu_peer_next[l]||!state->gpu_peer_mhc[l]||!state->gpu_peer_input[l]||!state->gpu_peer_block[l]||!state->gpu_peer_kv[l])))die("CUDA state allocation failed");
    }
    free(cs);free(sn);free(cc);free(ss);
}

static float *attention_window_cuda(const Dsv4Cfg *c,int layer,int pos,Dsv4State *state,const float *in){
    char n[256];snprintf(n,sizeof(n),"layers.%d.attn_norm.weight",layer);int dev=gpu_device_for(n),H=c->hidden;
    Dsv4CudaActivation *x=dsv4_cuda_activation_create(dev,H),*y=dsv4_cuda_activation_create(dev,H);float*out=malloc((size_t)H*sizeof(float));
    if(!x||!y||!out||!dsv4_cuda_activation_upload(x,in,H)||!attention_window_cuda_activation(c,layer,pos,state,x,y)||
       !dsv4_cuda_activation_download(out,y,H)||!dsv4_cuda_activation_sync(y))die("CUDA window attention failed");
    dsv4_cuda_activation_free(x);dsv4_cuda_activation_free(y);return out;
}
#endif

static float *attention_step(shards *s,const Dsv4Cfg *c,int layer,int pos,Dsv4State *state,
                             const float *in,const char *path){
#ifdef COLI_CUDA
    const char *gpu_window=getenv("DSV4_CUDA_ATTENTION_WINDOW");
    if(state&&!path&&gpu_window&&atoi(gpu_window))return attention_window_cuda(c,layer,pos,state,in);
    const char *gpu_first=getenv("DSV4_CUDA_ATTENTION_FIRST");
    if(!state&&pos==0&&!path&&gpu_first&&atoi(gpu_first))return attention_first_cuda(c,layer,in);
    const char *gpu_check=getenv("DSV4_CUDA_ATTENTION_CHECK");float *gpu_check_out=NULL;
    if(!state&&pos==0&&!path&&gpu_check&&atoi(gpu_check))gpu_check_out=attention_first_cuda(c,layer,in);
#endif
    int H=c->hidden,HD=c->head_dim,NH=c->heads,G=8,R=c->o_lora;
    char name[256]; snprintf(name,sizeof(name),"layers.%d.attn_norm.weight",layer);
    float *nw=load_f32(s,name,H),*x=malloc((size_t)H*sizeof(float));
    float *kv=malloc((size_t)HD*sizeof(float)),*oa_in=malloc((size_t)(NH/G)*HD*sizeof(float));
    float *oa=malloc((size_t)G*R*sizeof(float)),*out=malloc((size_t)H*sizeof(float));
    if(!x||!kv||!oa_in||!oa||!out)die("out of memory tracing attention");
    rmsnorm(x,in,nw,H,c->eps);
    compressor_step(s,c,layer,pos,x,state);
    float *qx=malloc((size_t)H*sizeof(float));if(!qx)die("out of memory tracing quantized input");
    memcpy(qx,x,(size_t)H*sizeof(float));dsv4_fp8_simulate(qx,H,128);
    float *qr=malloc((size_t)c->q_lora*sizeof(float)),*q=malloc((size_t)NH*HD*sizeof(float));
    snprintf(name,sizeof(name),"layers.%d.attn.q_norm.weight",layer);float *qn=load_f32(s,name,c->q_lora);
    int fused_qkv=0;
#ifdef COLI_CUDA
    {char an[256],bn[256],kn[256],nn[256];snprintf(an,sizeof(an),"layers.%d.attn.wq_a.weight",layer);snprintf(bn,sizeof(bn),"layers.%d.attn.wq_b.weight",layer);
     snprintf(kn,sizeof(kn),"layers.%d.attn.wkv.weight",layer);snprintf(nn,sizeof(nn),"layers.%d.attn.q_norm.weight",layer);
     Dsv4CudaTensor *ta=*gpu_slot(an),*tb=*gpu_slot(bn),*tk=*gpu_slot(kn),*tn=*gpu_slot(nn);
     if(ta&&tb&&tk&&tn)fused_qkv=dsv4_cuda_qkv(ta,tn,tb,tk,c->eps,q,kv,qx);}
#endif
    if(!fused_qkv){snprintf(name,sizeof(name),"layers.%d.attn.wq_a",layer);fp8_layer(s,name,x,c->q_lora,H,qr);
        rmsnorm(qr,qr,qn,c->q_lora,c->eps);
        snprintf(name,sizeof(name),"layers.%d.attn.wq_b",layer);fp8_layer(s,name,qr,NH*HD,c->q_lora,q);}
    for(int h=0;h<NH;h++){
        double ss=0;for(int d=0;d<HD;d++)ss+=(double)q[(long)h*HD+d]*q[(long)h*HD+d];
        float inv=1.f/sqrtf((float)(ss/HD)+c->eps);
        for(int d=0;d<HD;d++)q[(long)h*HD+d]=dsv4_bf16(q[(long)h*HD+d]*inv);
        rope(q+(long)h*HD,c,pos,0);
    }
    if(!fused_qkv){snprintf(name,sizeof(name),"layers.%d.attn.wkv",layer); fp8_layer(s,name,x,HD,H,kv);}
    snprintf(name,sizeof(name),"layers.%d.attn.kv_norm.weight",layer); float *kvn=load_f32(s,name,HD);
    rmsnorm(kv,kv,kvn,HD,c->eps);
    rope(kv,c,pos,0);
    dsv4_fp8_simulate(kv,HD-c->qk_rope,64);
    float *cache=state?state->kv+(long)layer*c->window*HD:NULL;
    if(cache){memcpy(cache+(long)(pos%c->window)*HD,kv,(size_t)HD*sizeof(float));if(state->len<pos+1)state->len=pos+1;}
    snprintf(name,sizeof(name),"layers.%d.attn.attn_sink",layer);float *sink=load_f32(s,name,NH);
    int fused_wo=0;
    /* wo_a has one independent [R,8*HD] matrix per output group. */
    {
#ifdef COLI_CUDA
      snprintf(name,sizeof(name),"layers.%d.attn.wo_a.weight",layer);Dsv4CudaTensor **slot=gpu_slot(name);
      if(!*slot){uint8_t*w=load_raw(s,name,(long long)G*R*(NH/G)*HD);char sn[256];snprintf(sn,sizeof(sn),"layers.%d.attn.wo_a.scale",layer);
        uint8_t*sc=load_raw(s,sn,(long long)(G*R/128)*(((NH/G)*HD)/128));int dev=gpu_device_for(name);
        if(!dsv4_cuda_upload_fp8_bf16(slot,w,sc,G*R,(NH/G)*HD,dev))die("CUDA grouped FP8 upload failed");free(w);free(sc);}
      for(int g=0;g<G;g++){
          for(int h=0;h<NH/G;h++){int head=g*(NH/G)+h,start=pos-c->window+1;if(start<0)start=0;
              float mx=sink[head];for(int t=start;t<=pos;t++){float score=0,*v=cache?cache+(long)(t%c->window)*HD:kv;for(int d=0;d<HD;d++)score+=q[(long)head*HD+d]*v[d];score/=sqrtf((float)HD);if(score>mx)mx=score;}
              int cn=state&&state->compressed_len?state->compressed_len[layer]:0;for(int ci=0;ci<cn;ci++){float score=0,*v=state->compressed+((long)layer*state->max_compressed+ci)*HD;for(int d=0;d<HD;d++)score+=q[(long)head*HD+d]*v[d];score/=sqrtf((float)HD);if(score>mx)mx=score;}
              float den=expf(sink[head]-mx);memset(oa_in+(long)h*HD,0,(size_t)HD*sizeof(float));for(int t=start;t<=pos;t++){float score=0,*v=cache?cache+(long)(t%c->window)*HD:kv;for(int d=0;d<HD;d++)score+=q[(long)head*HD+d]*v[d];score/=sqrtf((float)HD);float a=expf(score-mx);den+=a;for(int d=0;d<HD;d++)oa_in[(long)h*HD+d]+=a*v[d];}
              for(int ci=0;ci<cn;ci++){float score=0,*v=state->compressed+((long)layer*state->max_compressed+ci)*HD;for(int d=0;d<HD;d++)score+=q[(long)head*HD+d]*v[d];score/=sqrtf((float)HD);float a=expf(score-mx);den+=a;for(int d=0;d<HD;d++)oa_in[(long)h*HD+d]+=a*v[d];}
              for(int d=0;d<HD;d++)oa_in[(long)h*HD+d]=dsv4_bf16(oa_in[(long)h*HD+d]/den);rope(oa_in+(long)h*HD,c,pos,1);
          }
          memcpy(q+(long)g*(NH/G)*HD,oa_in,(size_t)(NH/G)*HD*sizeof(float));
      }
      if(!path){char bn[256];snprintf(bn,sizeof(bn),"layers.%d.attn.wo_b.weight",layer);Dsv4CudaTensor *wb=*gpu_slot(bn);
          if(wb)fused_wo=dsv4_cuda_wo(*slot,wb,G,out,q);}
      if(!fused_wo&&!dsv4_cuda_matvec_grouped(*slot,oa,q,G))die("CUDA grouped FP8 matvec failed");
#else
      snprintf(name,sizeof(name),"layers.%d.attn.wo_a.weight",layer); uint8_t *w=load_raw(s,name,(long long)G*R*(NH/G)*HD);
      snprintf(name,sizeof(name),"layers.%d.attn.wo_a.scale",layer); uint8_t *sc=load_raw(s,name,(long long)(G*R/128)*(((NH/G)*HD)/128));
      for(int g=0;g<G;g++){
          for(int h=0;h<NH/G;h++){int head=g*(NH/G)+h,start=pos-c->window+1;if(start<0)start=0;
              float mx=sink[head];for(int t=start;t<=pos;t++){float score=0,*v=cache?cache+(long)(t%c->window)*HD:kv;
                  for(int d=0;d<HD;d++)score+=q[(long)head*HD+d]*v[d];score/=sqrtf((float)HD);if(score>mx)mx=score;}
              int cn=state&&state->compressed_len?state->compressed_len[layer]:0;
              for(int ci=0;ci<cn;ci++){float score=0,*v=state->compressed+((long)layer*state->max_compressed+ci)*HD;
                  for(int d=0;d<HD;d++)score+=q[(long)head*HD+d]*v[d];score/=sqrtf((float)HD);if(score>mx)mx=score;}
              float den=expf(sink[head]-mx);memset(oa_in+(long)h*HD,0,(size_t)HD*sizeof(float));
              for(int t=start;t<=pos;t++){float score=0,*v=cache?cache+(long)(t%c->window)*HD:kv;
                  for(int d=0;d<HD;d++)score+=q[(long)head*HD+d]*v[d];score/=sqrtf((float)HD);
                  float a=expf(score-mx);den+=a;for(int d=0;d<HD;d++)oa_in[(long)h*HD+d]+=a*v[d];}
              for(int ci=0;ci<cn;ci++){float score=0,*v=state->compressed+((long)layer*state->max_compressed+ci)*HD;
                  for(int d=0;d<HD;d++)score+=q[(long)head*HD+d]*v[d];score/=sqrtf((float)HD);
                  float a=expf(score-mx);den+=a;for(int d=0;d<HD;d++)oa_in[(long)h*HD+d]+=a*v[d];}
              for(int d=0;d<HD;d++)oa_in[(long)h*HD+d]=dsv4_bf16(oa_in[(long)h*HD+d]/den);
              rope(oa_in+(long)h*HD,c,pos,1);
          }
          dsv4_fp8_matvec_bf16w(oa+(long)g*R,oa_in,
              w+(long)g*R*(NH/G)*HD,sc+(long)g*(R/128)*(((NH/G)*HD)/128),R,(NH/G)*HD);
      }
      free(w);free(sc);
#endif
      if(!fused_wo)for(int i=0;i<G*R;i++)oa[i]=dsv4_bf16(oa[i]);
    }
    if(!fused_wo){snprintf(name,sizeof(name),"layers.%d.attn.wo_b",layer); fp8_layer(s,name,oa,H,G*R,out);}
    if(path){ FILE *f=fopen(path,"wb");if(!f){perror(path);exit(1);}
        fwrite(in,sizeof(float),(size_t)H,f);fwrite(x,sizeof(float),(size_t)H,f);fwrite(qx,sizeof(float),(size_t)H,f);
        fwrite(kv,sizeof(float),(size_t)HD,f);fwrite(oa,sizeof(float),(size_t)G*R,f);
        fwrite(out,sizeof(float),(size_t)H,f);fclose(f);
        printf("  attn_trace=%s layer=%d pos=%d out0=%.9g\n",path,layer,pos,out[0]); }
    #ifdef COLI_CUDA
    if(gpu_check_out){float mx=0,mean=0;int diff=0;for(int i=0;i<H;i++){float d=fabsf(out[i]-gpu_check_out[i]);mx=fmaxf(mx,d);mean+=d;diff+=d!=0;}fprintf(stderr,"[DSV4 CUDA ATTN CHECK] layer %d different=%d max=%g mean=%g\n",layer,diff,mx,mean/H);free(gpu_check_out);}
    #endif
    free(nw);free(x);free(qx);free(qr);free(q);free(qn);free(kv);free(kvn);free(sink);free(oa_in);free(oa);
    return out;
}

static void fp4_layer(shards *s,const char *stem,const float *x,int O,int I,float *y){
    char wn[256],sn[256];snprintf(wn,sizeof(wn),"%s.weight",stem);snprintf(sn,sizeof(sn),"%s.scale",stem);
#ifdef COLI_CUDA
    if(cw_n){CpuWeight *t=cpu_slot(wn);if(!t->w)die("CPU expert was not preloaded");float *qx=malloc((size_t)I*sizeof(float));
        if(!qx)die("out of memory quantizing CPU FP4 input");memcpy(qx,x,(size_t)I*sizeof(float));dsv4_fp8_simulate(qx,I,128);
        if(getenv("COLI_DSV4_CPU_IDOT")&&atoi(getenv("COLI_DSV4_CPU_IDOT")))matmul_mxfp4_i8(y,qx,t->w,t->scale,1,I,O);
        else matmul_mxfp4(y,qx,t->w,t->scale,1,I,O);free(qx);for(int i=0;i<O;i++)y[i]=dsv4_bf16(y[i]);return;}
    Dsv4CudaTensor **slot=gpu_slot(wn);float *qx=malloc((size_t)I*sizeof(float));if(!qx)die("out of memory quantizing FP4 input");
    memcpy(qx,x,(size_t)I*sizeof(float));dsv4_fp8_simulate(qx,I,128);
    if(!*slot){uint8_t*w=load_raw(s,wn,(long long)O*(I/2)),*sc=load_raw(s,sn,(long long)O*(I/32));int dev=gpu_device_for(wn);
        if(!dsv4_cuda_upload_fp4(slot,w,sc,O,I,dev))die("CUDA FP4 upload failed");free(w);free(sc);}
    if(!dsv4_cuda_matvec(*slot,y,qx))die("CUDA FP4 matvec failed");free(qx);
    { static int checks; if(getenv("DSV4_CUDA_CHECK")&&checks++<3){uint8_t*w=load_raw(s,wn,(long long)O*(I/2)),*sc=load_raw(s,sn,(long long)O*(I/32));
        float *ref=malloc((size_t)O*sizeof(float)),*cx=malloc((size_t)I*sizeof(float));memcpy(cx,x,(size_t)I*sizeof(float));dsv4_fp8_simulate(cx,I,128);matmul_mxfp4(ref,cx,w,sc,1,I,O);
        float mx=0,mean=0;for(int i=0;i<O;i++){float d=fabsf(ref[i]-y[i]);if(d>mx)mx=d;mean+=d;}fprintf(stderr,"[DSV4 CUDA CHECK] %s max=%g mean=%g\n",stem,mx,mean/O);free(w);free(sc);free(ref);free(cx);} }
#else
    uint8_t *w=load_raw(s,wn,(long long)O*(I/2));uint8_t *sc=load_raw(s,sn,(long long)O*(I/32));
    float *qx=malloc((size_t)I*sizeof(float));if(!qx)die("out of memory quantizing FP4 input");
    memcpy(qx,x,(size_t)I*sizeof(float));dsv4_fp8_simulate(qx,I,128);
    matmul_mxfp4(y,qx,w,sc,1,I,O);
    for(int i=0;i<O;i++)y[i]=dsv4_bf16(y[i]);
    free(qx);free(w);free(sc);
#endif
    for(int i=0;i<O;i++)y[i]=dsv4_bf16(y[i]);
}

static void expert_fp4(shards *s,int layer,int expert,const float *x,float weight,
                       int H,int I,float limit,float *out){
    char stem[256];float *g=malloc((size_t)I*sizeof(float)),*u=malloc((size_t)I*sizeof(float));
    float *a=malloc((size_t)I*sizeof(float));if(!g||!u||!a)die("out of memory in FP4 expert");
    snprintf(stem,sizeof(stem),"layers.%d.ffn.experts.%d.w1",layer,expert);fp4_layer(s,stem,x,I,H,g);
    snprintf(stem,sizeof(stem),"layers.%d.ffn.experts.%d.w3",layer,expert);fp4_layer(s,stem,x,I,H,u);
    for(int i=0;i<I;i++){float gg=fminf(g[i],limit),uu=fmaxf(-limit,fminf(u[i],limit));
        a[i]=dsv4_bf16(weight*(gg/(1.f+expf(-gg)))*uu);}
    snprintf(stem,sizeof(stem),"layers.%d.ffn.experts.%d.w2",layer,expert);fp4_layer(s,stem,a,H,I,out);
    free(g);free(u);free(a);
}

#ifdef COLI_CUDA
static inline float mxfp4_row_exact(const float*x,const uint8_t*w,const uint8_t*scl,int I){int ng=I/32;
#ifdef __AVX2__
    const __m128i lut2=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12),m4=_mm_set1_epi8(0x0F);__m256 acc=_mm256_setzero_ps();
    for(int g=0;g<ng;g++){__m128i by=_mm_loadu_si128((const __m128i*)(w+g*16));__m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_shuffle_epi8(lut2,_mm_unpacklo_epi8(lo,hi)),n1=_mm_shuffle_epi8(lut2,_mm_unpackhi_epi8(lo,hi));const float*xg=x+g*32;
        __m256 ga=_mm256_mul_ps(_mm256_loadu_ps(xg),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n0)));
        ga=_mm256_fmadd_ps(_mm256_loadu_ps(xg+8),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n0,8))),ga);
        ga=_mm256_fmadd_ps(_mm256_loadu_ps(xg+16),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n1)),ga);
        ga=_mm256_fmadd_ps(_mm256_loadu_ps(xg+24),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n1,8))),ga);
        acc=_mm256_fmadd_ps(ga,_mm256_set1_ps(mx4_scale(scl[g])*.5f),acc);}return hsum256(acc);
#else
    float sum=0;for(int g=0;g<ng;g++){float part=0;for(int i=0;i<32;i+=2){uint8_t q=w[g*16+i/2];part+=x[g*32+i]*mx4_lut[q&15]+x[g*32+i+1]*mx4_lut[q>>4];}sum+=part*mx4_scale(scl[g]);}return sum;
#endif
}
static void mxfp4_quant_i8(const float*x,int8_t*q,float*scale,int I){for(int g=0;g<I/32;g++){const float*xg=x+g*32;float am=0;for(int i=0;i<32;i++){float a=fabsf(xg[i]);if(a>am)am=a;}float sc=am/127.f;if(sc<1e-20f)sc=1e-20f;scale[g]=sc;float inv=1.f/sc;for(int i=0;i<32;i++){int v=(int)lrintf(xg[i]*inv);if(v>127)v=127;if(v<-127)v=-127;q[g*32+i]=(int8_t)v;}}}
static inline float mxfp4_row_i8(const int8_t*x,const float*xscale,const uint8_t*w,const uint8_t*scl,int I){int ng=I/32;
#ifdef __AVX2__
    const __m128i lut2=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12),m4=_mm_set1_epi8(0x0F);const __m256i ones=_mm256_set1_epi16(1);__m256 acc=_mm256_setzero_ps();
    for(int g=0;g<ng;g++){__m128i by=_mm_loadu_si128((const __m128i*)(w+g*16));__m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m256i wv=_mm256_set_m128i(_mm_shuffle_epi8(lut2,_mm_unpackhi_epi8(lo,hi)),_mm_shuffle_epi8(lut2,_mm_unpacklo_epi8(lo,hi)));__m256i xv=_mm256_loadu_si256((const __m256i*)(x+g*32));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p,ones)),_mm256_set1_ps(mx4_scale(scl[g])*.5f*xscale[g]),acc);}return hsum256(acc);
#else
    float sum=0;static const int8_t lut[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};for(int g=0;g<ng;g++){int v=0;for(int i=0;i<32;i+=2){uint8_t p=w[g*16+i/2];v+=lut[p&15]*x[g*32+i]+lut[p>>4]*x[g*32+i+1];}sum+=v*(mx4_scale(scl[g])*.5f*xscale[g]);}return sum;
#endif
}
static int expert_group_cpu(int layer,const int*ids,const float*weights,int count,const float*x,int H,int I,float limit,float*out){
    if(!cw_n||count<1||count>16||H%32||I%32)return 0;CpuWeight *g[16],*u[16],*d[16];char name[256];for(int e=0;e<count;e++){
        snprintf(name,sizeof(name),"layers.%d.ffn.experts.%d.w1.weight",layer,ids[e]);g[e]=cpu_slot(name);
        snprintf(name,sizeof(name),"layers.%d.ffn.experts.%d.w3.weight",layer,ids[e]);u[e]=cpu_slot(name);
        snprintf(name,sizeof(name),"layers.%d.ffn.experts.%d.w2.weight",layer,ids[e]);d[e]=cpu_slot(name);if(!g[e]->w||!u[e]->w||!d[e]->w)return 0;}
    int idot=getenv("COLI_DSV4_CPU_IDOT")&&atoi(getenv("COLI_DSV4_CPU_IDOT"));float*qx=malloc((size_t)H*sizeof(float)),*gate=malloc((size_t)count*I*sizeof(float)),*up=malloc((size_t)count*I*sizeof(float)),*act=malloc((size_t)count*I*sizeof(float)),*part=malloc((size_t)count*H*sizeof(float));
    int8_t*q8=idot?malloc((size_t)(H+count*I)):NULL;float*qsc=idot?malloc((size_t)(H/32+count*(I/32))*sizeof(float)):NULL;
    if(!qx||!gate||!up||!act||!part||(idot&&(!q8||!qsc)))die("out of memory in grouped CPU experts");memcpy(qx,x,(size_t)H*sizeof(float));dsv4_fp8_simulate(qx,H,128);if(idot)mxfp4_quant_i8(qx,q8,qsc,H);
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(int z=0;z<2*count*I;z++){int side=z/(count*I),p=z%(count*I),e=p/I,o=p%I;CpuWeight*t=side?u[e]:g[e];float*v=side?up:gate;
            v[p]=dsv4_bf16(idot?mxfp4_row_i8(q8,qsc,t->w+(long long)o*(H/2),t->scale+(long long)o*(H/32),H):mxfp4_row_exact(qx,t->w+(long long)o*(H/2),t->scale+(long long)o*(H/32),H));}
        #pragma omp for schedule(static)
        for(int p=0;p<count*I;p++){int e=p/I;float gg=fminf(gate[p],limit),uu=fmaxf(-limit,fminf(up[p],limit));act[p]=dsv4_bf16(weights[e]*(gg/(1.f+expf(-gg)))*uu);}
        #pragma omp for schedule(static)
        for(int e=0;e<count;e++){dsv4_fp8_simulate(act+(long long)e*I,I,128);if(idot)mxfp4_quant_i8(act+(long long)e*I,q8+H+(long long)e*I,qsc+H/32+(long long)e*(I/32),I);}
        #pragma omp for schedule(static)
        for(int p=0;p<count*H;p++){int e=p/H,o=p%H;CpuWeight*t=d[e];part[p]=dsv4_bf16(idot?mxfp4_row_i8(q8+H+(long long)e*I,qsc+H/32+(long long)e*(I/32),t->w+(long long)o*(I/2),t->scale+(long long)o*(I/32),I):mxfp4_row_exact(act+(long long)e*I,t->w+(long long)o*(I/2),t->scale+(long long)o*(I/32),I));}
    }
    memset(out,0,(size_t)H*sizeof(float));for(int e=0;e<count;e++)for(int h=0;h<H;h++)out[h]+=part[(long long)e*H+h];free(qx);free(q8);free(qsc);free(gate);free(up);free(act);free(part);return 1;
}
static int expert_group_fp4(int layer,const int *ids,const float *weights,int count,
                            const float *x,int H,int I,float limit,float *out){
    if(cw_n)return expert_group_cpu(layer,ids,weights,count,x,H,I,limit,out);
    if(count<1||count>16)return 0;Dsv4CudaTensor *g[16],*u[16],*d[16];char name[256];
    for(int k=0;k<count;k++){
        snprintf(name,sizeof(name),"layers.%d.ffn.experts.%d.w1.weight",layer,ids[k]);g[k]=*gpu_slot(name);
        snprintf(name,sizeof(name),"layers.%d.ffn.experts.%d.w3.weight",layer,ids[k]);u[k]=*gpu_slot(name);
        snprintf(name,sizeof(name),"layers.%d.ffn.experts.%d.w2.weight",layer,ids[k]);d[k]=*gpu_slot(name);
        if(!g[k]||!u[k]||!d[k])return 0;
    }
    float *qx=malloc((size_t)H*sizeof(float));if(!qx)die("out of memory quantizing grouped Expert input");memcpy(qx,x,(size_t)H*sizeof(float));dsv4_fp8_simulate(qx,H,128);
    const char*ep=getenv("DSV4_CUDA_EXPERT_PARALLEL");if(ep&&atoi(ep)&&gpu_ndev>1){Dsv4CudaTensor *gg[16][16]={},*uu[16][16]={},*dd[16][16]={};float ww[16][16]={};int n[16]={},okv[16]={};
        for(int k=0;k<count;k++){int di=ids[k]%gpu_ndev,p=n[di]++;gg[di][p]=g[k];uu[di][p]=u[k];dd[di][p]=d[k];ww[di][p]=weights[k];}
        float*parts=calloc((size_t)gpu_ndev*H,sizeof(float));if(!parts)die("out of memory collecting expert-parallel outputs");
        #pragma omp parallel for num_threads(16) schedule(static)
        for(int di=0;di<gpu_ndev;di++)if(n[di])okv[di]=dsv4_cuda_expert_group(gg[di],uu[di],dd[di],ww[di],n[di],limit,parts+(long long)di*H,qx);else okv[di]=1;
        int ok=1;for(int di=0;di<gpu_ndev;di++)ok&=okv[di];if(ok){memset(out,0,(size_t)H*sizeof(float));for(int di=0;di<gpu_ndev;di++)for(int h=0;h<H;h++)out[h]+=parts[(long long)di*H+h];}free(parts);free(qx);return ok;}
    int ok=dsv4_cuda_expert_group(g,u,d,weights,count,limit,out,qx);free(qx);return ok;
}
static int moe_group_cuda(int layer,const int *ids,const float *weights,int count,const float*x,int H,float limit,float*out){
    if(cw_n||count<1||count>16)return 0;Dsv4CudaTensor *g[16],*u[16],*d[16];char name[256];for(int k=0;k<count;k++){
        snprintf(name,sizeof(name),"layers.%d.ffn.experts.%d.w1.weight",layer,ids[k]);g[k]=*gpu_slot(name);
        snprintf(name,sizeof(name),"layers.%d.ffn.experts.%d.w3.weight",layer,ids[k]);u[k]=*gpu_slot(name);
        snprintf(name,sizeof(name),"layers.%d.ffn.experts.%d.w2.weight",layer,ids[k]);d[k]=*gpu_slot(name);if(!g[k]||!u[k]||!d[k])return 0;}
    snprintf(name,sizeof(name),"layers.%d.ffn.shared_experts.w1.weight",layer);Dsv4CudaTensor*sg=*gpu_slot(name);
    snprintf(name,sizeof(name),"layers.%d.ffn.shared_experts.w3.weight",layer);Dsv4CudaTensor*su=*gpu_slot(name);
    snprintf(name,sizeof(name),"layers.%d.ffn.shared_experts.w2.weight",layer);Dsv4CudaTensor*sd=*gpu_slot(name);if(!sg||!su||!sd)return 0;
    float*qx=malloc((size_t)H*sizeof(float));if(!qx)die("out of memory quantizing fused MoE input");memcpy(qx,x,(size_t)H*sizeof(float));dsv4_fp8_simulate(qx,H,128);
    int ok=dsv4_cuda_moe(g,u,d,weights,count,sg,su,sd,limit,out,qx);free(qx);return ok;
}
#endif

static void expert_fp8(shards *s,const char *prefix,const float *x,int H,int I,float limit,float *out){
#ifdef COLI_CUDA
    char gn[256],un[256],dn[256];snprintf(gn,sizeof(gn),"%s.w1.weight",prefix);snprintf(un,sizeof(un),"%s.w3.weight",prefix);snprintf(dn,sizeof(dn),"%s.w2.weight",prefix);
    Dsv4CudaTensor *gt=*gpu_slot(gn),*ut=*gpu_slot(un),*dt=*gpu_slot(dn);if(gt&&ut&&dt){float *qx=malloc((size_t)H*sizeof(float));if(!qx)die("out of memory quantizing shared Expert input");
        memcpy(qx,x,(size_t)H*sizeof(float));dsv4_fp8_simulate(qx,H,128);int ok=dsv4_cuda_expert_fp8(gt,ut,dt,limit,out,qx);free(qx);if(ok)return;}
#endif
    char stem[256];float *g=malloc((size_t)I*sizeof(float)),*u=malloc((size_t)I*sizeof(float));
    float *a=malloc((size_t)I*sizeof(float));if(!g||!u||!a)die("out of memory in shared expert");
    snprintf(stem,sizeof(stem),"%s.w1",prefix);fp8_layer(s,stem,x,I,H,g);
    snprintf(stem,sizeof(stem),"%s.w3",prefix);fp8_layer(s,stem,x,I,H,u);
    for(int i=0;i<I;i++){float gg=fminf(g[i],limit),uu=fmaxf(-limit,fminf(u[i],limit));
        a[i]=dsv4_bf16((gg/(1.f+expf(-gg)))*uu);}
    snprintf(stem,sizeof(stem),"%s.w2",prefix);fp8_layer(s,stem,a,H,I,out);
    free(g);free(u);free(a);
}

static void select_experts(shards *s,const Dsv4Cfg *c,int layer,int token,const float *x,
                           int *ids,float *weights){
    int H=c->hidden,E=c->experts,K=c->topk; char name[256];
#ifdef COLI_CUDA
    const char *gpu_router=getenv("DSV4_CUDA_ROUTER");if(gpu_router&&atoi(gpu_router)){
        int fixed[6],*fp=NULL;if(layer<c->hash_layers){snprintf(name,sizeof(name),"layers.%d.ffn.gate.tid2eid",layer);uint8_t*map=load_raw(s,name,(long long)c->vocab*K*8);for(int k=0;k<K;k++){int64_t v;memcpy(&v,map+((long long)token*K+k)*8,8);fixed[k]=(int)v;}free(map);fp=fixed;}
        snprintf(name,sizeof(name),"layers.%d.ffn.gate.weight",layer);Dsv4CudaTensor*gate=*gpu_slot(name),*bias=NULL;if(layer>=c->hash_layers){snprintf(name,sizeof(name),"layers.%d.ffn.gate.bias",layer);bias=*gpu_slot(name);}int dev=gpu_device_for(name);Dsv4CudaActivation*a=dsv4_cuda_activation_create(dev,H);if(!a||!dsv4_cuda_activation_upload(a,x,H)||!dsv4_cuda_route(a,gate,bias,fp,c->routed_scale,ids,weights))die("CUDA router failed");dsv4_cuda_activation_free(a);return;}
#endif
    snprintf(name,sizeof(name),"layers.%d.ffn.gate.weight",layer);
    float *gw=load_f32(s,name,(long long)E*H),*scores=malloc((size_t)E*sizeof(float));
    if(!scores)die("out of memory routing experts");
    for(int e=0;e<E;e++){float v=0;for(int h=0;h<H;h++)v+=x[h]*gw[(long)e*H+h];
        scores[e]=sqrtf(log1pf(expf(-fabsf(v)))+fmaxf(v,0.f));}
    if(layer<c->hash_layers){
        snprintf(name,sizeof(name),"layers.%d.ffn.gate.tid2eid",layer);
        uint8_t *map=load_raw(s,name,(long long)c->vocab*K*8);
        for(int k=0;k<K;k++){int64_t v;memcpy(&v,map+((long long)token*K+k)*8,8);ids[k]=(int)v;}
        free(map);
    }else{
        snprintf(name,sizeof(name),"layers.%d.ffn.gate.bias",layer);
        float *bias=load_f32(s,name,E); unsigned char *used=calloc((size_t)E,1);
        if(!used)die("out of memory selecting experts");
        for(int k=0;k<K;k++){int best=-1;float bv=-INFINITY;
            for(int e=0;e<E;e++)if(!used[e]&&scores[e]+bias[e]>bv){bv=scores[e]+bias[e];best=e;}
            ids[k]=best;used[best]=1;
        }
        free(used);free(bias);
    }
    float sum=0;for(int k=0;k<K;k++)sum+=scores[ids[k]];
    for(int k=0;k<K;k++)weights[k]=scores[ids[k]]/sum*c->routed_scale;
    free(scores);free(gw);
}

#ifdef COLI_CUDA
static int run_layer_cuda(shards *s,const Dsv4Cfg *c,int layer,int token,int pos,Dsv4State *state,float *host_res,
                          int copy_input,int download_output){
    int M=c->hc_mult,H=c->hidden,MH=M*H,S=2*M+M*M;char n[256];
    snprintf(n,sizeof(n),"layers.%d.attn_norm.weight",layer);int dev=gpu_device_for(n);
    if(!state->gpu_residual[layer]){
        state->gpu_residual[layer]=dsv4_cuda_activation_create(dev,MH);state->gpu_next[layer]=dsv4_cuda_activation_create(dev,MH);
        state->gpu_mhc[layer]=dsv4_cuda_activation_create(dev,S);state->gpu_input[layer]=dsv4_cuda_activation_create(dev,H);
        state->gpu_block[layer]=dsv4_cuda_activation_create(dev,H);
        if(!state->gpu_residual[layer]||!state->gpu_next[layer]||!state->gpu_mhc[layer]||!state->gpu_input[layer]||!state->gpu_block[layer])die("CUDA layer activation allocation failed");
    }
    if(gpu_ep2()&&!state->gpu_peer_block[layer]){state->gpu_peer_block[layer]=dsv4_cuda_activation_create(gpu_dev[gpu_stage(layer)+1],H);if(!state->gpu_peer_block[layer])die("CUDA EP2 peer activation allocation failed");}
    Dsv4CudaActivation *res=state->gpu_residual[layer],*next=state->gpu_next[layer],*mhc=state->gpu_mhc[layer];
    Dsv4CudaActivation *x=state->gpu_input[layer],*block=state->gpu_block[layer];
    if(copy_input&&(layer? !dsv4_cuda_activation_copy(res,state->gpu_next[layer-1],MH):!dsv4_cuda_activation_upload(res,host_res,MH)))die("CUDA residual transfer failed");
    Dsv4CudaActivation *pres=NULL,*pnext=NULL,*pmhc=NULL,*px=NULL,*pblock=NULL;TpLayerWeights *tp=NULL;
    if(gpu_replicated_tp2()){pres=state->gpu_peer_residual[layer];pnext=state->gpu_peer_next[layer];pmhc=state->gpu_peer_mhc[layer];px=state->gpu_peer_input[layer];pblock=state->gpu_peer_block[layer];tp=&tp_layer[layer];
        int same=layer&&gpu_stage(layer)==gpu_stage(layer-1);const Dsv4CudaActivation *psrc=same?state->gpu_peer_next[layer-1]:(copy_input?res:NULL);
        if(psrc&&!dsv4_cuda_activation_copy(pres,psrc,MH))die("CUDA peer residual transfer failed");}
#define WT(var,suffix) snprintf(n,sizeof(n),"layers.%d.%s",layer,suffix);Dsv4CudaTensor *var=*gpu_slot(n)
    WT(fn,"hc_attn_fn");WT(scale,"hc_attn_scale");WT(base,"hc_attn_base");
    WT(ffn_fn,"hc_ffn_fn");WT(ffn_scale,"hc_ffn_scale");WT(ffn_base,"hc_ffn_base");WT(ffn_norm,"ffn_norm.weight");
    if(!dsv4_cuda_mhc_pre(res,fn,scale,base,M,H,c->eps,c->hc_eps,c->hc_eps,2.f,c->hc_iters,mhc,x)||
       (tp&&!dsv4_cuda_mhc_pre(pres,tp->attn_fn,tp->attn_scale,tp->attn_base,M,H,c->eps,c->hc_eps,c->hc_eps,2.f,c->hc_iters,pmhc,px))||
       !attention_window_cuda_activation(c,layer,pos,state,x,block)||
       !dsv4_cuda_mhc_post_pre_norm(block,res,mhc,M,H,next,ffn_fn,ffn_scale,ffn_base,ffn_norm,c->eps,c->hc_eps,c->hc_eps,2.f,c->hc_iters,c->eps,x)||
       (tp&&!dsv4_cuda_mhc_post_pre_norm(pblock,pres,pmhc,M,H,pnext,tp->ffn_fn,tp->ffn_scale,tp->ffn_base,tp->ffn_norm,c->eps,c->hc_eps,c->hc_eps,2.f,c->hc_iters,c->eps,px))||
       !dsv4_cuda_activation_copy(res,next,MH)||(tp&&!dsv4_cuda_activation_copy(pres,pnext,MH)))die("CUDA attention block failed");
    WT(gate,"ffn.gate.weight");Dsv4CudaTensor *bias=NULL;
    if(layer>=c->hash_layers){snprintf(n,sizeof(n),"layers.%d.ffn.gate.bias",layer);bias=*gpu_slot(n);}
    state->gpu_experts[layer]=ge_cache[layer];if(!state->gpu_experts[layer])die("CUDA expert table is unavailable");
    int graph=!(getenv("DSV4_CUDA_SEGMENT_GRAPH")&&atoi(getenv("DSV4_CUDA_SEGMENT_GRAPH")))&&
              getenv("DSV4_CUDA_FFN_GRAPH")&&atoi(getenv("DSV4_CUDA_FFN_GRAPH"))&&layer>=c->hash_layers;
    if(state->gpu_ffn_graph[layer]){
        if(!dsv4_cuda_graph_launch(state->gpu_ffn_graph[layer]))die("CUDA FFN graph launch failed");
    }else{
        int capture=graph&&pos>0;if(capture&&!dsv4_cuda_graph_begin(dev))die("CUDA FFN graph capture failed");
        int moe_ok=gpu_ep2()?dsv4_cuda_route_moe_ep2(x,gate,bias,px,tp?tp->gate:NULL,tp?tp->bias:NULL,token,c->routed_scale,state->gpu_experts[layer],ge_cache_peer[layer],c->swiglu_limit,block,state->gpu_peer_block[layer]):
                             dsv4_cuda_route_moe(x,gate,bias,token,c->routed_scale,state->gpu_experts[layer],c->swiglu_limit,block);
        if(!moe_ok||
           !dsv4_cuda_mhc_post(block,res,mhc,M,H,next)||(tp&&!dsv4_cuda_mhc_post(pblock,pres,pmhc,M,H,pnext)))die("CUDA FFN block failed");
        if(capture){state->gpu_ffn_graph[layer]=dsv4_cuda_graph_end(dev);if(!state->gpu_ffn_graph[layer]||!dsv4_cuda_graph_launch(state->gpu_ffn_graph[layer]))die("CUDA FFN graph finalize failed");}
    }
#undef WT
    const char *limit_env=getenv("DSV4_CUDA_LAYER_LIMIT");int limit=limit_env?atoi(limit_env):c->layers;
    if(download_output&&(layer==c->layers-1||layer+1==limit)&&(!dsv4_cuda_activation_download(host_res,next,MH)||!dsv4_cuda_activation_sync(next)))die("CUDA final residual download failed");
    return 1;
}

static Dsv4CudaActivation *run_cuda_segments(shards *s,const Dsv4Cfg *c,int token,int pos,Dsv4State *state,float *res){
    int M=c->hc_mult,MH=M*c->hidden;
    int trace=getenv("DSV4_CUDA_STAGE_TRACE")&&atoi(getenv("DSV4_CUDA_STAGE_TRACE"))&&pos>1;
    for(int first=0;first<c->layers;){
        double started=trace?wall_time():0;
        char n[64];snprintf(n,sizeof(n),"layers.%d.attn_norm.weight",first);int dev=gpu_device_for(n),last=first+1;
        while(last<c->layers){snprintf(n,sizeof(n),"layers.%d.attn_norm.weight",last);if(gpu_device_for(n)!=dev)break;last++;}
        if(first){
            if(!dsv4_cuda_activation_copy(state->gpu_residual[first],state->gpu_next[first-1],MH))die("CUDA segment transfer failed");
        }else if(!dsv4_cuda_activation_upload(state->gpu_residual[0],res,MH))die("CUDA initial residual upload failed");
        if(gpu_replicated_tp2()&&!dsv4_cuda_activation_copy(state->gpu_peer_residual[first],state->gpu_residual[first],MH))die("CUDA TP2 stage transfer failed");
        int slot=0;while(slot<gpu_ndev&&gpu_dev[slot]!=dev)slot++;
        if(slot==gpu_ndev)die("CUDA segment device is unavailable");
        int peer_slot=gpu_ep2()?slot+1:-1;
        if(state->gpu_segment_graph[slot]){
            if(gpu_ep2()&&(!state->gpu_segment_graph[peer_slot]||!dsv4_cuda_graph_launch(state->gpu_segment_graph[peer_slot])))die("CUDA EP2 peer graph launch failed");
            if(!dsv4_cuda_graph_launch(state->gpu_segment_graph[slot]))die("CUDA segment graph launch failed");
        }else{
            int capture=pos>0;if(capture&&(!dsv4_cuda_graph_begin(dev)||(gpu_ep2()&&!dsv4_cuda_graph_begin(gpu_dev[peer_slot]))))die("CUDA segment graph capture failed");
            for(int l=first;l<last;l++)run_layer_cuda(s,c,l,token,pos,state,res,l!=first,0);
            if(capture){
                if(gpu_ep2()){if(!dsv4_cuda_graph_end_pair(dev,gpu_dev[peer_slot],&state->gpu_segment_graph[slot],&state->gpu_segment_graph[peer_slot]))die("CUDA EP2 graph finalize failed");}
                else state->gpu_segment_graph[slot]=dsv4_cuda_graph_end(dev);
                if(!state->gpu_segment_graph[slot]||(gpu_ep2()&&!dsv4_cuda_graph_launch(state->gpu_segment_graph[peer_slot]))||!dsv4_cuda_graph_launch(state->gpu_segment_graph[slot]))die("CUDA segment graph finalize failed");
            }
        }
        if(trace){
            if(!dsv4_cuda_activation_sync(state->gpu_next[last-1]))die("CUDA stage trace sync failed");
            fprintf(stderr,"[DSV4 stage] pos=%d layers=%d..%d device=%d elapsed=%.6fs\n",
                    pos,first,last-1,dev,wall_time()-started);
        }
        first=last;
    }
    return state->gpu_next[c->layers-1];
}
#endif

static float *run_layer(shards *s,const Dsv4Cfg *c,int layer,int token,int pos,Dsv4State *state,
                        float *res,const char *path){
#ifdef COLI_CUDA
    const char *gpu_layer=getenv("DSV4_CUDA_LAYER"),*limit_env=getenv("DSV4_CUDA_LAYER_LIMIT");int limit=limit_env?atoi(limit_env):c->layers;
    if(state&&!path&&gpu_layer&&atoi(gpu_layer)&&layer<limit){run_layer_cuda(s,c,layer,token,pos,state,res,1,1);return res;}
#endif
    int M=c->hc_mult,H=c->hidden,I=c->moe_inter,N=2*M+M*M,MH=M*H;
    (void)N; float post[16],comb[256],*x=calloc((size_t)H,sizeof(float));
    float *next=malloc((size_t)MH*sizeof(float)); if(!x||!next)die("out of memory running layer");
    double pt=wall_time();mhc_pre(s,c,layer,"attn",res,x,post,comb);prof_mhc+=wall_time()-pt;
    pt=wall_time();float *attn=attention_step(s,c,layer,pos,state,x,NULL);prof_attn+=wall_time()-pt;
    dsv4_mhc_post(attn,res,post,comb,M,H,next);memcpy(res,next,(size_t)MH*sizeof(float));
    pt=wall_time();memset(x,0,(size_t)H*sizeof(float));mhc_pre(s,c,layer,"ffn",res,x,post,comb);prof_mhc+=wall_time()-pt;
    char name[256];snprintf(name,sizeof(name),"layers.%d.ffn_norm.weight",layer);
    float *nw=load_f32(s,name,H);rmsnorm(x,x,nw,H,c->eps);
    int ids[16];float rw[16];pt=wall_time();select_experts(s,c,layer,token,x,ids,rw);prof_route+=wall_time()-pt;
    float *moe=calloc((size_t)H,sizeof(float)),*tmp=malloc((size_t)H*sizeof(float));
    if(!moe||!tmp)die("out of memory running MoE");
    int grouped=0,combined=0;pt=wall_time();
#ifdef COLI_CUDA
    combined=moe_group_cuda(layer,ids,rw,c->topk,x,H,c->swiglu_limit,moe);
    if(!combined)grouped=expert_group_fp4(layer,ids,rw,c->topk,x,H,I,c->swiglu_limit,moe);
#endif
    if(!combined&&!grouped)for(int k=0;k<c->topk;k++){expert_fp4(s,layer,ids[k],x,rw[k],H,I,c->swiglu_limit,tmp);
        for(int h=0;h<H;h++)moe[h]+=tmp[h];}
    prof_routed+=wall_time()-pt;pt=wall_time();
    if(!combined){snprintf(name,sizeof(name),"layers.%d.ffn.shared_experts",layer);
        expert_fp8(s,name,x,H,I,c->swiglu_limit,tmp);}
    prof_shared+=wall_time()-pt;
    if(!combined)for(int h=0;h<H;h++)moe[h]=dsv4_bf16(moe[h]+tmp[h]);
    dsv4_mhc_post(moe,res,post,comb,M,H,next);memcpy(res,next,(size_t)MH*sizeof(float));
    if(path){FILE *f=fopen(path,"wb");if(!f){perror(path);exit(1);}fwrite(res,sizeof(float),(size_t)MH,f);fclose(f);
        printf("  block_trace=%s layer=%d token=%d out0=%.9g experts=",path,layer,token,res[0]);for(int k=0;k<c->topk;k++)printf("%s%d",k?",":"",ids[k]);putchar('\n');}
    free(attn);free(x);free(nw);free(moe);free(tmp);free(next);
    return res;
}

static float *first_block(shards *s,const Dsv4Cfg *c,int token,const char *path){
    float *res=initial_residual(s,c,token);return run_layer(s,c,0,token,0,NULL,res,path);
}

static float *first_attention(shards *s,const Dsv4Cfg *c,int token,const char *path){
    int H=c->hidden;float post[16],comb[256],*in=calloc((size_t)H,sizeof(float));
    float *res=initial_residual(s,c,token);if(!in)die("out of memory tracing attention");
    mhc_pre(s,c,0,"attn",res,in,post,comb);float *out=attention_step(s,c,0,0,NULL,in,path);
    free(res);free(in);return out;
}

static void trace_attention_two(shards *s,const Dsv4Cfg *c,int first,int second,const char *path){
    if(first<0||first>=c->vocab||second<0||second>=c->vocab)die("trace token id is outside vocabulary");
    int H=c->hidden;float post[16],comb[256],*in=calloc((size_t)H,sizeof(float));
    Dsv4State state=state_init(c,2);if(!in)die("out of memory tracing cached attention");
    float *res=initial_residual(s,c,first);mhc_pre(s,c,0,"attn",res,in,post,comb);
    float *out=attention_step(s,c,0,0,&state,in,NULL);free(out);free(res);
    memset(in,0,(size_t)H*sizeof(float));res=initial_residual(s,c,second);
    mhc_pre(s,c,0,"attn",res,in,post,comb);out=attention_step(s,c,0,1,&state,in,path);
    free(out);free(res);free(in);state_free(&state);
}

static void trace_compressor(shards *s,const Dsv4Cfg *c,int layer,int token,const char *path){
    if(layer<0||layer>=c->layers||!c->compress[layer])die("trace layer has no compressor");
    int H=c->hidden,ratio=c->compress[layer],steps=ratio==4?2*ratio:ratio;Dsv4State state=state_init(c,steps);
    float post[16],comb[256],*in=calloc((size_t)H,sizeof(float)),*res=initial_residual(s,c,token);
    char name[256];snprintf(name,sizeof(name),"layers.%d.attn_norm.weight",layer);float *nw=load_f32(s,name,H);
    if(!in)die("out of memory tracing compressor");
    for(int pos=0;pos<steps;pos++){memset(in,0,(size_t)H*sizeof(float));mhc_pre(s,c,layer,"attn",res,in,post,comb);
        rmsnorm(in,in,nw,H,c->eps);compressor_step(s,c,layer,pos,in,&state);}
    float *out=state.compressed+((long)layer*state.max_compressed+state.compressed_len[layer]-1)*c->head_dim;
    FILE *f=fopen(path,"wb");if(!f){perror(path);exit(1);}fwrite(out,sizeof(float),(size_t)c->head_dim,f);fclose(f);
    printf("  compressor_trace=%s layer=%d ratio=%d emission=%d out0=%.9g\n",path,layer,ratio,state.compressed_len[layer],out[0]);
    free(nw);free(in);free(res);state_free(&state);
}

static void final_hidden(shards *s,const Dsv4Cfg *c,const float *res,float *x){
    int M=c->hc_mult,H=c->hidden,MH=M*H;float *fn=load_f32(s,"hc_head_fn",(long long)M*MH);
    float *scale=load_f32(s,"hc_head_scale",1),*base=load_f32(s,"hc_head_base",M);
    double ss=0;for(int i=0;i<MH;i++)ss+=(double)res[i]*res[i];
    float inv=1.f/sqrtf((float)(ss/MH)+c->eps);
    memset(x,0,(size_t)H*sizeof(float));
    for(int m=0;m<M;m++){float mix=0;for(int i=0;i<MH;i++)mix+=res[i]*fn[(long)m*MH+i];
        float pre=1.f/(1.f+expf(-(mix*inv*scale[0]+base[m])))+c->hc_eps;
        for(int h=0;h<H;h++)x[h]+=pre*res[(long)m*H+h];}
    for(int h=0;h<H;h++)x[h]=dsv4_bf16(x[h]);
    float *nw=load_f32(s,"norm.weight",H);rmsnorm(x,x,nw,H,c->eps);
    free(fn);free(scale);free(base);free(nw);
}

static int head_argmax(shards *s,const Dsv4Cfg *c,const float *x,float *best_logit){
#ifdef COLI_CUDA
    int H=c->hidden;Dsv4CudaTensor **slot=gpu_slot("head.weight");if(!*slot){uint8_t*w=load_raw(s,"head.weight",(long long)c->vocab*H*2);int dev=gpu_device_for("head.weight");
        if(!dsv4_cuda_upload_bf16(slot,(uint16_t*)w,c->vocab,H,dev))die("CUDA output head upload failed");free(w);}
    int fast_id;if(dsv4_cuda_head_argmax(*slot,x,&fast_id,best_logit))return fast_id;
    float *logits=malloc((size_t)c->vocab*sizeof(float));if(!logits)die("out of memory reading logits");
    if(!dsv4_cuda_matvec(*slot,logits,x))die("CUDA output head failed");int best=0;for(int v=1;v<c->vocab;v++)if(logits[v]>logits[best])best=v;
    *best_logit=logits[best];free(logits);return best;
#else
    enum { CHUNK=256 };int H=c->hidden,best=-1;float bv=-INFINITY;
    float *w=malloc((size_t)CHUNK*H*sizeof(float)),logits[CHUNK];if(!w)die("out of memory reading output head");
    for(int v=0;v<c->vocab;v+=CHUNK){int n=c->vocab-v;if(n>CHUNK)n=CHUNK;
        st_read_slice_f32(s,"head.weight",(long long)v*H,(long long)n*H,w,0);
        #pragma omp parallel for
        for(int r=0;r<n;r++){float sum=0;for(int h=0;h<H;h++)sum+=x[h]*w[(long)r*H+h];logits[r]=sum;}
        for(int r=0;r<n;r++)if(logits[r]>bv){bv=logits[r];best=v+r;}
    }
    free(w);*best_logit=bv;return best;
#endif
}

#ifdef COLI_CUDA
static int head_argmax_gpu(const Dsv4Cfg*c,Dsv4CudaActivation*res,float*best_logit){
    Dsv4CudaTensor *fn=*gpu_slot("hc_head_fn"),*scale=*gpu_slot("hc_head_scale"),*base=*gpu_slot("hc_head_base");
    Dsv4CudaTensor *norm=*gpu_slot("norm.weight"),*head=*gpu_slot("head.weight");int id;
    if(!dsv4_cuda_final_argmax(res,fn,scale,base,norm,head,c->hc_mult,c->hidden,c->eps,c->hc_eps,&id,best_logit))die("CUDA final head failed");return id;
}
#endif

#ifdef COLI_CUDA
static void preload_fp8(shards *s,const char *stem,int O,int I,int bf16_round){
    char wn[256],sn[256];snprintf(wn,sizeof(wn),"%s.weight",stem);snprintf(sn,sizeof(sn),"%s.scale",stem);
    Dsv4CudaTensor **slot=gpu_slot(wn);if(*slot)return;
    uint8_t *w=load_raw(s,wn,(long long)O*I),*sc=load_raw(s,sn,(long long)((O+127)/128)*((I+127)/128));
    int ok=bf16_round?dsv4_cuda_upload_fp8_bf16(slot,w,sc,O,I,gpu_device_for(wn)):dsv4_cuda_upload_fp8(slot,w,sc,O,I,gpu_device_for(wn));
    free(w);free(sc);if(!ok)die("CUDA FP8 preload failed");
}

static Dsv4CudaTensor *preload_fp8_rows_on(shards *s,const char *stem,int O,int I,int first,int rows,int device,int bf16_round){
    char wn[256],sn[256],key[288];snprintf(wn,sizeof(wn),"%s.weight",stem);snprintf(sn,sizeof(sn),"%s.scale",stem);
    snprintf(key,sizeof(key),"%s.rows.%d.%d",wn,first,rows);Dsv4CudaTensor **slot=gpu_slot_on(key,device);if(*slot)return *slot;
    uint8_t *all=load_raw(s,wn,(long long)O*I),*asc=load_raw(s,sn,(long long)(O/128)*(I/128));
    uint8_t *w=malloc((size_t)rows*I),*sc=malloc((size_t)(rows/128)*(I/128));if(!w||!sc)die("out of memory sharding FP8 rows");
    memcpy(w,all+(long long)first*I,(size_t)rows*I);memcpy(sc,asc+(long long)(first/128)*(I/128),(size_t)(rows/128)*(I/128));
    int ok=bf16_round?dsv4_cuda_upload_fp8_bf16(slot,w,sc,rows,I,device):dsv4_cuda_upload_fp8(slot,w,sc,rows,I,device);
    free(all);free(asc);free(w);free(sc);if(!ok)die("CUDA row-sharded FP8 preload failed");return *slot;
}

static Dsv4CudaTensor *preload_fp8_cols_on(shards *s,const char *stem,int O,int I,int first,int cols,int device){
    char wn[256],sn[256],key[288];snprintf(wn,sizeof(wn),"%s.weight",stem);snprintf(sn,sizeof(sn),"%s.scale",stem);
    snprintf(key,sizeof(key),"%s.cols.%d.%d",wn,first,cols);Dsv4CudaTensor **slot=gpu_slot_on(key,device);if(*slot)return *slot;
    uint8_t *all=load_raw(s,wn,(long long)O*I),*asc=load_raw(s,sn,(long long)(O/128)*(I/128));
    uint8_t *w=malloc((size_t)O*cols),*sc=malloc((size_t)(O/128)*(cols/128));if(!w||!sc)die("out of memory sharding FP8 columns");
    for(int o=0;o<O;o++)memcpy(w+(long long)o*cols,all+(long long)o*I+first,(size_t)cols);
    for(int ob=0;ob<O/128;ob++)memcpy(sc+(long long)ob*(cols/128),asc+(long long)ob*(I/128)+first/128,(size_t)(cols/128));
    int ok=dsv4_cuda_upload_fp8(slot,w,sc,O,cols,device);free(all);free(asc);free(w);free(sc);
    if(!ok)die("CUDA column-sharded FP8 preload failed");return *slot;
}

static Dsv4CudaTensor *preload_fp8_copy_on(shards *s,const char *stem,int O,int I,int device,int bf16_round){
    char wn[256],sn[256];snprintf(wn,sizeof(wn),"%s.weight",stem);snprintf(sn,sizeof(sn),"%s.scale",stem);Dsv4CudaTensor **slot=gpu_slot_on(wn,device);if(*slot)return *slot;
    uint8_t *w=load_raw(s,wn,(long long)O*I),*sc=load_raw(s,sn,(long long)(O/128)*(I/128));
    int ok=bf16_round?dsv4_cuda_upload_fp8_bf16(slot,w,sc,O,I,device):dsv4_cuda_upload_fp8(slot,w,sc,O,I,device);
    free(w);free(sc);if(!ok)die("CUDA replicated FP8 preload failed");return *slot;
}

static Dsv4CudaTensor *preload_fp8_pair_on(shards *s,const char *first,int first_rows,const char *second,
                                           int second_rows,int cols,int device){
    char aw[256],as[256],bw[256],bs[256],key[576];
    snprintf(aw,sizeof(aw),"%s.weight",first);snprintf(as,sizeof(as),"%s.scale",first);
    snprintf(bw,sizeof(bw),"%s.weight",second);snprintf(bs,sizeof(bs),"%s.scale",second);
    snprintf(key,sizeof(key),"%s+%s",aw,bw);Dsv4CudaTensor **slot=gpu_slot_on(key,device);if(*slot)return *slot;
    int rows=first_rows+second_rows,sc_cols=cols/128;
    uint8_t *w=malloc((size_t)rows*cols),*sc=malloc((size_t)(rows/128)*sc_cols);
    if(!w||!sc)die("out of memory fusing attention projections");
    uint8_t *a=load_raw(s,aw,(long long)first_rows*cols),*sa=load_raw(s,as,(long long)(first_rows/128)*sc_cols);
    uint8_t *b=load_raw(s,bw,(long long)second_rows*cols),*sb=load_raw(s,bs,(long long)(second_rows/128)*sc_cols);
    memcpy(w,a,(size_t)first_rows*cols);memcpy(w+(long long)first_rows*cols,b,(size_t)second_rows*cols);
    memcpy(sc,sa,(size_t)(first_rows/128)*sc_cols);memcpy(sc+(long long)(first_rows/128)*sc_cols,sb,(size_t)(second_rows/128)*sc_cols);
    int ok=dsv4_cuda_upload_fp8(slot,w,sc,rows,cols,device);
    free(a);free(sa);free(b);free(sb);free(w);free(sc);if(!ok)die("CUDA fused attention preload failed");return *slot;
}

static Dsv4CudaTensor *preload_raw_on(shards *s,const char *name,int O,int I,int device,int fmt){
    Dsv4CudaTensor **slot=gpu_slot_on(name,device);if(*slot)return *slot;int ok=0;
    if(fmt==32){float *w=load_f32(s,name,(long long)O*I);ok=dsv4_cuda_upload_f32(slot,w,O,I,device);free(w);}
    else{uint8_t *w=load_raw(s,name,(long long)O*I*2);ok=dsv4_cuda_upload_bf16(slot,(uint16_t*)w,O,I,device);free(w);}
    if(!ok)die("CUDA replicated raw preload failed");return *slot;
}

static Dsv4CudaTensor *preload_f32_rows_on(shards *s,const char *name,int O,int first,int rows,int device){
    char key[288];snprintf(key,sizeof(key),"%s.rows.%d.%d",name,first,rows);Dsv4CudaTensor **slot=gpu_slot_on(key,device);if(*slot)return *slot;
    float *all=load_f32(s,name,O),*w=malloc((size_t)rows*sizeof(float));if(!w)die("out of memory sharding F32 rows");
    memcpy(w,all+first,(size_t)rows*sizeof(float));int ok=dsv4_cuda_upload_f32(slot,w,rows,1,device);free(all);free(w);
    if(!ok)die("CUDA row-sharded F32 preload failed");return *slot;
}

static void preload_fp4(shards *s,const char *stem,int O,int I){
    char wn[256],sn[256];snprintf(wn,sizeof(wn),"%s.weight",stem);snprintf(sn,sizeof(sn),"%s.scale",stem);
    Dsv4CudaTensor **slot=gpu_slot(wn);if(*slot)return;
    uint8_t *w=load_raw(s,wn,(long long)O*(I/2)),*sc=load_raw(s,sn,(long long)O*(I/32));
    int ok=dsv4_cuda_upload_fp4(slot,w,sc,O,I,gpu_device_for(wn));free(w);free(sc);if(!ok)die("CUDA FP4 preload failed");
}

static void preload_fp4_cpu(shards *s,const char *stem,int O,int I){
    char wn[256],sn[256];snprintf(wn,sizeof(wn),"%s.weight",stem);snprintf(sn,sizeof(sn),"%s.scale",stem);CpuWeight *t=cpu_slot(wn);if(t->w)return;
    t->w=load_raw(s,wn,(long long)O*(I/2));t->scale=load_raw(s,sn,(long long)O*(I/32));cw_bytes+=(long long)O*(I/2)+(long long)O*(I/32);
}

static void preload_bf16(shards *s,const char *name,int O,int I){
    Dsv4CudaTensor **slot=gpu_slot(name);if(*slot)return;uint8_t *w=load_raw(s,name,(long long)O*I*2);
    int ok=dsv4_cuda_upload_bf16(slot,(uint16_t*)w,O,I,gpu_device_for(name));free(w);if(!ok)die("CUDA BF16 preload failed");
}

static void preload_f32(shards *s,const char *name,int O,int I){
    Dsv4CudaTensor **slot=gpu_slot(name);if(*slot)return;float *w=load_f32(s,name,(long long)O*I);
    int ok=dsv4_cuda_upload_f32(slot,w,O,I,gpu_device_for(name));free(w);if(!ok)die("CUDA F32 preload failed");
}

static void gpu_preload(shards *s,const Dsv4Cfg *c){
    struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);char n[256];int H=c->hidden,I=c->moe_inter,G=8;
    int hybrid=getenv("COLI_DSV4_HYBRID")&&atoi(getenv("COLI_DSV4_HYBRID"));
    int ep2=gpu_ep2();if(getenv("DSV4_CUDA_EP2")&&atoi(getenv("DSV4_CUDA_EP2"))&&!ep2)die("DSV4_CUDA_EP2 requires exactly 6 GPUs");
    int moe_tp2=ep2&&getenv("DSV4_CUDA_MOE_TP2")&&atoi(getenv("DSV4_CUDA_MOE_TP2"));
    int shared_tp=moe_tp2||(ep2&&getenv("DSV4_CUDA_SHARED_TP2")&&atoi(getenv("DSV4_CUDA_SHARED_TP2")));
    fprintf(stderr,"[DSV4 CUDA] preloading dense matrices to VRAM, routed experts to %s\n",hybrid?"RAM":"VRAM");
    for(int l=0;l<c->layers;l++){
        int M=c->hc_mult,N=2*M+M*M,MH=M*H;
        snprintf(n,sizeof(n),"layers.%d.hc_attn_fn",l);preload_f32(s,n,N,MH);
        snprintf(n,sizeof(n),"layers.%d.hc_attn_scale",l);preload_f32(s,n,3,1);
        snprintf(n,sizeof(n),"layers.%d.hc_attn_base",l);preload_f32(s,n,N,1);
        snprintf(n,sizeof(n),"layers.%d.hc_ffn_fn",l);preload_f32(s,n,N,MH);
        snprintf(n,sizeof(n),"layers.%d.hc_ffn_scale",l);preload_f32(s,n,3,1);
        snprintf(n,sizeof(n),"layers.%d.hc_ffn_base",l);preload_f32(s,n,N,1);
        snprintf(n,sizeof(n),"layers.%d.attn_norm.weight",l);preload_f32(s,n,H,1);
        snprintf(n,sizeof(n),"layers.%d.attn.kv_norm.weight",l);preload_f32(s,n,c->head_dim,1);
        snprintf(n,sizeof(n),"layers.%d.attn.attn_sink",l);preload_f32(s,n,c->heads,1);
        snprintf(n,sizeof(n),"layers.%d.ffn_norm.weight",l);preload_f32(s,n,H,1);
        snprintf(n,sizeof(n),"layers.%d.ffn.gate.weight",l);preload_f32(s,n,c->experts,H);
        if(l>=c->hash_layers){snprintf(n,sizeof(n),"layers.%d.ffn.gate.bias",l);preload_f32(s,n,c->experts,1);}
        if(ep2){int peer=gpu_dev[gpu_stage(l)+1];TpLayerWeights *tp=&tp_layer[l];
            snprintf(n,sizeof(n),"layers.%d.hc_attn_fn",l);tp->attn_fn=preload_raw_on(s,n,N,MH,peer,32);
            snprintf(n,sizeof(n),"layers.%d.hc_attn_scale",l);tp->attn_scale=preload_raw_on(s,n,3,1,peer,32);
            snprintf(n,sizeof(n),"layers.%d.hc_attn_base",l);tp->attn_base=preload_raw_on(s,n,N,1,peer,32);
            snprintf(n,sizeof(n),"layers.%d.hc_ffn_fn",l);tp->ffn_fn=preload_raw_on(s,n,N,MH,peer,32);
            snprintf(n,sizeof(n),"layers.%d.hc_ffn_scale",l);tp->ffn_scale=preload_raw_on(s,n,3,1,peer,32);
            snprintf(n,sizeof(n),"layers.%d.hc_ffn_base",l);tp->ffn_base=preload_raw_on(s,n,N,1,peer,32);
            snprintf(n,sizeof(n),"layers.%d.ffn_norm.weight",l);tp->ffn_norm=preload_raw_on(s,n,H,1,peer,32);
            snprintf(n,sizeof(n),"layers.%d.ffn.gate.weight",l);tp->gate=preload_raw_on(s,n,c->experts,H,peer,32);
            if(l>=c->hash_layers){snprintf(n,sizeof(n),"layers.%d.ffn.gate.bias",l);tp->bias=preload_raw_on(s,n,c->experts,1,peer,32);}}
        snprintf(n,sizeof(n),"layers.%d.attn.wq_a",l);preload_fp8(s,n,c->q_lora,H,0);
        snprintf(n,sizeof(n),"layers.%d.attn.q_norm.weight",l);preload_bf16(s,n,1,c->q_lora);
        snprintf(n,sizeof(n),"layers.%d.attn.wq_b",l);preload_fp8(s,n,c->heads*c->head_dim,c->q_lora,0);
        snprintf(n,sizeof(n),"layers.%d.attn.wkv",l);preload_fp8(s,n,c->head_dim,H,0);
#ifdef COLI_DSV4_DEEPGEMM
        if(!ep2){char qa[256],kv[256];snprintf(qa,sizeof(qa),"layers.%d.attn.wq_a",l);snprintf(kv,sizeof(kv),"layers.%d.attn.wkv",l);batch_qkv[l]=preload_fp8_pair_on(s,qa,c->q_lora,kv,c->head_dim,H,gpu_device_for(n));}
#endif
        snprintf(n,sizeof(n),"layers.%d.attn.wo_a",l);preload_fp8(s,n,G*c->o_lora,(c->heads/G)*c->head_dim,1);
        snprintf(n,sizeof(n),"layers.%d.attn.wo_b",l);preload_fp8(s,n,H,G*c->o_lora,0);
        if(ep2){int primary=gpu_dev[gpu_stage(l)],peer=gpu_dev[gpu_stage(l)+1],lh=c->heads/2,lg=G/2,Q=lh*c->head_dim,WR=lg*c->o_lora;
            for(int rank=0;rank<2;rank++){int d=rank?peer:primary;Dsv4CudaAttentionWeights *tp=&tp_attention[l][rank];
                snprintf(n,sizeof(n),"layers.%d.attn_norm.weight",l);tp->attn_norm=rank?preload_raw_on(s,n,H,1,d,32):*gpu_slot(n);
                snprintf(n,sizeof(n),"layers.%d.attn.wq_a",l);if(rank)tp->q_a=preload_fp8_copy_on(s,n,c->q_lora,H,d,0);else{char wn[256];snprintf(wn,sizeof(wn),"%s.weight",n);tp->q_a=*gpu_slot(wn);}
                snprintf(n,sizeof(n),"layers.%d.attn.q_norm.weight",l);tp->q_norm=rank?preload_raw_on(s,n,1,c->q_lora,d,16):*gpu_slot(n);
                snprintf(n,sizeof(n),"layers.%d.attn.wq_b",l);tp->q_b=preload_fp8_rows_on(s,n,c->heads*c->head_dim,c->q_lora,rank*Q,Q,d,0);
                snprintf(n,sizeof(n),"layers.%d.attn.wkv",l);if(rank)tp->wkv=preload_fp8_copy_on(s,n,c->head_dim,H,d,0);else{char wn[256];snprintf(wn,sizeof(wn),"%s.weight",n);tp->wkv=*gpu_slot(wn);}
#ifdef COLI_DSV4_DEEPGEMM
                {char qa[256],kv[256];snprintf(qa,sizeof(qa),"layers.%d.attn.wq_a",l);snprintf(kv,sizeof(kv),"layers.%d.attn.wkv",l);
                 tp->qkv=preload_fp8_pair_on(s,qa,c->q_lora,kv,c->head_dim,H,d);}
#endif
                snprintf(n,sizeof(n),"layers.%d.attn.kv_norm.weight",l);tp->kv_norm=rank?preload_raw_on(s,n,c->head_dim,1,d,32):*gpu_slot(n);
                snprintf(n,sizeof(n),"layers.%d.attn.attn_sink",l);tp->sink=preload_f32_rows_on(s,n,c->heads,rank*lh,lh,d);
                snprintf(n,sizeof(n),"layers.%d.attn.wo_a",l);tp->wo_a=preload_fp8_rows_on(s,n,G*c->o_lora,(c->heads/G)*c->head_dim,rank*WR,WR,d,1);
                snprintf(n,sizeof(n),"layers.%d.attn.wo_b",l);tp->wo_b=preload_fp8_cols_on(s,n,H,G*c->o_lora,rank*WR,WR,d);
            }}
        if(c->compress[l]){int O=(1+(c->compress[l]==4))*c->head_dim;
            snprintf(n,sizeof(n),"layers.%d.attn.compressor.wkv.weight",l);preload_bf16(s,n,O,H);
            snprintf(n,sizeof(n),"layers.%d.attn.compressor.wgate.weight",l);preload_bf16(s,n,O,H);
            snprintf(n,sizeof(n),"layers.%d.attn.compressor.ape",l);preload_f32(s,n,c->compress[l]*O,1);
            snprintf(n,sizeof(n),"layers.%d.attn.compressor.norm.weight",l);preload_f32(s,n,c->head_dim,1);
            if(ep2){for(int rank=0;rank<2;rank++){int d=gpu_dev[gpu_stage(l)+rank];Dsv4CudaAttentionWeights *tp=&tp_attention[l][rank];
                snprintf(n,sizeof(n),"layers.%d.attn.compressor.wkv.weight",l);tp->compress_wkv=rank?preload_raw_on(s,n,O,H,d,16):*gpu_slot(n);
                snprintf(n,sizeof(n),"layers.%d.attn.compressor.wgate.weight",l);tp->compress_wgate=rank?preload_raw_on(s,n,O,H,d,16):*gpu_slot(n);
                snprintf(n,sizeof(n),"layers.%d.attn.compressor.ape",l);tp->compress_ape=rank?preload_raw_on(s,n,c->compress[l]*O,1,d,32):*gpu_slot(n);
                snprintf(n,sizeof(n),"layers.%d.attn.compressor.norm.weight",l);tp->compress_norm=rank?preload_raw_on(s,n,c->head_dim,1,d,32):*gpu_slot(n);
            }}}
        if(!shared_tp){snprintf(n,sizeof(n),"layers.%d.ffn.shared_experts.w1",l);preload_fp8(s,n,I,H,0);
            snprintf(n,sizeof(n),"layers.%d.ffn.shared_experts.w3",l);preload_fp8(s,n,I,H,0);
            snprintf(n,sizeof(n),"layers.%d.ffn.shared_experts.w2",l);preload_fp8(s,n,H,I,0);}
        Dsv4CudaExpertSet *bank=NULL,*peer=NULL;
        if(!hybrid){
            Dsv4CudaTensor *sg=NULL,*su=NULL,*sd=NULL,*psg=NULL,*psu=NULL,*psd=NULL;
            if(shared_tp){int J=I/2,d0=gpu_dev[gpu_stage(l)],d1=gpu_dev[gpu_stage(l)+1];
                snprintf(n,sizeof(n),"layers.%d.ffn.shared_experts.w1",l);sg=preload_fp8_rows_on(s,n,I,H,0,J,d0,0);psg=preload_fp8_rows_on(s,n,I,H,J,J,d1,0);
                snprintf(n,sizeof(n),"layers.%d.ffn.shared_experts.w3",l);su=preload_fp8_rows_on(s,n,I,H,0,J,d0,0);psu=preload_fp8_rows_on(s,n,I,H,J,J,d1,0);
                snprintf(n,sizeof(n),"layers.%d.ffn.shared_experts.w2",l);sd=preload_fp8_cols_on(s,n,H,I,0,J,d0);psd=preload_fp8_cols_on(s,n,H,I,J,J,d1);
            }else{snprintf(n,sizeof(n),"layers.%d.ffn.shared_experts.w1.weight",l);sg=*gpu_slot(n);
                snprintf(n,sizeof(n),"layers.%d.ffn.shared_experts.w3.weight",l);su=*gpu_slot(n);
                snprintf(n,sizeof(n),"layers.%d.ffn.shared_experts.w2.weight",l);sd=*gpu_slot(n);}
            snprintf(n,sizeof(n),"layers.%d.ffn.experts.0.w1.weight",l);
            int expert_i=moe_tp2?I/2:I,d0=ep2?gpu_dev[gpu_stage(l)]:gpu_device_for(n),d1=ep2?gpu_dev[gpu_stage(l)+1]:-1;
            bank=dsv4_cuda_expert_bank_create(moe_tp2?c->experts:(ep2?c->experts/2:c->experts),H,expert_i,d0,sg,su,sd);
            if(!bank)die("CUDA contiguous expert bank allocation failed");
            if(ep2){snprintf(n,sizeof(n),"layers.%d.ffn.experts.%d.w1.weight",l,c->experts/2);
                peer=dsv4_cuda_expert_bank_create(moe_tp2?c->experts:c->experts/2,H,expert_i,d1,psg,psu,psd);
                if(!peer)die("CUDA peer expert bank allocation failed");}
        }
        for(int e=0;e<c->experts;e++){
            if(hybrid){
                snprintf(n,sizeof(n),"layers.%d.ffn.experts.%d.w1",l,e);preload_fp4_cpu(s,n,I,H);
                snprintf(n,sizeof(n),"layers.%d.ffn.experts.%d.w3",l,e);preload_fp4_cpu(s,n,I,H);
                snprintf(n,sizeof(n),"layers.%d.ffn.experts.%d.w2",l,e);preload_fp4_cpu(s,n,H,I);
            }else{
                char wn[3][256],sn[3][256];uint8_t*w[3],*sc[3];
                int O[3]={I,I,H},K[3]={H,H,I};Dsv4CudaTensor **slot[3];
                for(int j=0;j<3;j++){
                    int projection=j==0?1:j==1?3:2;
                    snprintf(wn[j],sizeof(wn[j]),"layers.%d.ffn.experts.%d.w%d.weight",l,e,projection);
                    snprintf(sn[j],sizeof(sn[j]),"layers.%d.ffn.experts.%d.w%d.scale",l,e,projection);
                    slot[j]=gpu_slot(wn[j]);w[j]=load_raw(s,wn[j],(long long)O[j]*(K[j]/2));sc[j]=load_raw(s,sn[j],(long long)O[j]*(K[j]/32));
                }
                int ok;if(moe_tp2)ok=dsv4_cuda_expert_bank_upload_tp2(bank,e,0,w[0],sc[0],w[1],sc[1],w[2],sc[2])&&
                                      dsv4_cuda_expert_bank_upload_tp2(peer,e,1,w[0],sc[0],w[1],sc[1],w[2],sc[2]);
                else{Dsv4CudaExpertSet *target=ep2&&e>=c->experts/2?peer:bank;int local=ep2?e%(c->experts/2):e;
                    ok=dsv4_cuda_expert_bank_upload(target,local,w[0],sc[0],w[1],sc[1],w[2],sc[2],slot[0],slot[1],slot[2]);}
                for(int j=0;j<3;j++){free(w[j]);free(sc[j]);}if(!ok)die("CUDA contiguous expert upload failed");
            }
        }
        if(!hybrid){ge_cache[l]=bank;ge_cache_peer[l]=peer;
            if(l<c->hash_layers){snprintf(n,sizeof(n),"layers.%d.ffn.gate.tid2eid",l);uint8_t*map=load_raw(s,n,(long long)c->vocab*c->topk*8);
                int ok=dsv4_cuda_expert_set_upload_hash(ge_cache[l],(int64_t*)map,c->vocab,c->topk);
                if(ep2)ok=ok&&dsv4_cuda_expert_set_upload_hash(ge_cache_peer[l],(int64_t*)map,c->vocab,c->topk);free(map);if(!ok)die("CUDA hash router preload failed");}}
        fprintf(stderr,"[DSV4 CUDA] preloaded layer %d/%d\r",l+1,c->layers);
    }
    preload_f32(s,"hc_head_fn",c->hc_mult,c->hc_mult*H);preload_f32(s,"hc_head_scale",1,1);preload_f32(s,"hc_head_base",c->hc_mult,1);preload_f32(s,"norm.weight",H,1);preload_bf16(s,"head.weight",c->vocab,H);clock_gettime(CLOCK_MONOTONIC,&b);
    fprintf(stderr,"\n[DSV4 CUDA] preload complete in %.1fs, %d GPU matrices, %d CPU matrices (%.2f GiB)\n",b.tv_sec-a.tv_sec+(b.tv_nsec-a.tv_nsec)*1e-9,gw_n,cw_n,cw_bytes/1073741824.0);gpu_report();
}
#endif

#ifdef COLI_CUDA
typedef struct {Dsv4CudaActivation *res,*next,*mhc,*input,*block,*context;} PrefillGpu;
static void prefill_gpu_free(PrefillGpu*p){dsv4_cuda_activation_free(p->context);dsv4_cuda_activation_free(p->block);dsv4_cuda_activation_free(p->input);dsv4_cuda_activation_free(p->mhc);dsv4_cuda_activation_free(p->next);dsv4_cuda_activation_free(p->res);memset(p,0,sizeof(*p));}
static int prefill_batch_cuda(shards*s,const Dsv4Cfg*c,Dsv4State*state,const int*tokens,int count,int start,float*logit){
    if(count<1||count>64||gpu_ep2()||!batch_qkv[0])return -1;
    enum {G=8};int H=c->hidden,MH=c->hc_mult*H;long long RH=(long long)count*MH,XH=(long long)count*H,CH=(long long)count*c->heads*c->head_dim;
    float*host=malloc((size_t)RH*sizeof(float));if(!host)die("out of memory preparing batched embeddings");
    for(int t=0;t<count;t++){float*r=initial_residual(s,c,tokens[t]);memcpy(host+(long long)t*MH,r,(size_t)MH*sizeof(float));free(r);}
    PrefillGpu stage[16]={0};Dsv4CudaActivation*current=NULL;int current_dev=-1;
    for(int l=0;l<c->layers;l++){
        char n[256];snprintf(n,sizeof(n),"layers.%d.attn_norm.weight",l);int dev=gpu_device_for(n),slot=0;while(slot<gpu_ndev&&gpu_dev[slot]!=dev)slot++;if(slot==gpu_ndev)die("batched prefill device is unavailable");PrefillGpu*p=stage+slot;
        if(!p->res){p->res=dsv4_cuda_activation_create(dev,RH);p->next=dsv4_cuda_activation_create(dev,RH);p->mhc=dsv4_cuda_activation_create(dev,(long long)count*20);p->input=dsv4_cuda_activation_create(dev,XH);p->block=dsv4_cuda_activation_create(dev,XH);p->context=dsv4_cuda_activation_create(dev,CH);if(!p->res||!p->next||!p->mhc||!p->input||!p->block||!p->context)die("batched prefill activation allocation failed");}
        if(!current){if(!dsv4_cuda_activation_upload(p->res,host,RH))die("batched embedding upload failed");current=p->res;current_dev=dev;}
        else if(current_dev!=dev){if(!dsv4_cuda_activation_copy(p->res,current,RH))die("batched pipeline transfer failed");current=p->res;current_dev=dev;}
        else if(current!=p->res){Dsv4CudaActivation*tmp=p->res;p->res=current;current=tmp;}
#define BWT(var,suffix) snprintf(n,sizeof(n),"layers.%d.%s",l,suffix);Dsv4CudaTensor*var=*gpu_slot(n)
        BWT(afn,"hc_attn_fn");BWT(ascale,"hc_attn_scale");BWT(abase,"hc_attn_base");BWT(an,"attn_norm.weight");BWT(qn,"attn.q_norm.weight");BWT(qb,"attn.wq_b.weight");BWT(kvn,"attn.kv_norm.weight");BWT(sink,"attn.attn_sink");BWT(wa,"attn.wo_a.weight");BWT(wb,"attn.wo_b.weight");
        BWT(ffn_fn,"hc_ffn_fn");BWT(ffn_scale,"hc_ffn_scale");BWT(ffn_base,"hc_ffn_base");BWT(ffn_norm,"ffn_norm.weight");BWT(gate,"ffn.gate.weight");Dsv4CudaTensor*bias=NULL;if(l>=c->hash_layers){snprintf(n,sizeof(n),"layers.%d.ffn.gate.bias",l);bias=*gpu_slot(n);}
        if(!dsv4_cuda_mhc_pre_batch(current,afn,ascale,abase,count,H,p->mhc,p->input)||!dsv4_cuda_attention_sparse_batch(p->input,an,batch_qkv[l],qn,qb,kvn,sink,c->heads,c->head_dim,start,count,c->eps,state->gpu_kv[l],p->context)||!dsv4_cuda_attention_output_batch(p->context,wa,wb,G,count,p->block)||!dsv4_cuda_mhc_post_pre_norm_batch(p->block,current,p->mhc,count,H,p->next,ffn_fn,ffn_scale,ffn_base,ffn_norm,p->input)||!dsv4_cuda_route_moe_batch(p->input,gate,bias,tokens,count,c->routed_scale,ge_cache[l],c->swiglu_limit,p->block)||!dsv4_cuda_mhc_post_batch(p->block,p->next,p->mhc,count,H,p->res))die("batched prefill layer failed");
        current=p->res;
#undef BWT
    }
    if(!dsv4_cuda_activation_copy_range(state->gpu_next[c->layers-1],0,current,(long long)(count-1)*MH,MH))die("batched prefill final hidden copy failed");int next=head_argmax_gpu(c,state->gpu_next[c->layers-1],logit);for(int d=0;d<16;d++)prefill_gpu_free(stage+d);free(host);return next;
}
#endif

static int model_step(shards *s,const Dsv4Cfg *c,Dsv4State *state,int token,int pos,float *logit){
#ifdef COLI_CUDA
    if(state&&getenv("DSV4_CUDA_LAYER")&&atoi(getenv("DSV4_CUDA_LAYER")))for(int d=0;d<gpu_ndev;d++)if(!dsv4_cuda_decode_state_set(gpu_dev[d],token,pos))die("CUDA decode state update failed");
#endif
#ifdef COLI_CUDA
    int segment_graph=state&&getenv("DSV4_CUDA_LAYER")&&atoi(getenv("DSV4_CUDA_LAYER"))&&
                      getenv("DSV4_CUDA_SEGMENT_GRAPH")&&atoi(getenv("DSV4_CUDA_SEGMENT_GRAPH"));
    if(segment_graph&&pos>0){float *res=initial_residual(s,c,token);Dsv4CudaActivation*out=run_cuda_segments(s,c,token,pos,state,res);free(res);double pt=wall_time();int next=head_argmax_gpu(c,out,logit);prof_head+=wall_time()-pt;return next;}
    else
#endif
    {float *res=initial_residual(s,c,token);
    for(int l=0;l<c->layers;l++)run_layer(s,c,l,token,pos,state,res,NULL);
    float *x=malloc((size_t)c->hidden*sizeof(float));if(!x)die("out of memory producing logits");
    final_hidden(s,c,res,x);double pt=wall_time();int next=head_argmax(s,c,x,logit);prof_head+=wall_time()-pt;free(x);free(res);return next;}
}

static int first_token(shards *s,const Dsv4Cfg *c,int token){
    if(token<0||token>=c->vocab)die("input token id is outside vocabulary");
    struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);float logit;int next=model_step(s,c,NULL,token,0,&logit);
    clock_gettime(CLOCK_MONOTONIC,&b);double sec=b.tv_sec-a.tv_sec+(b.tv_nsec-a.tv_nsec)*1e-9;
    printf("  first_token input=%d output=%d logit=%.9g elapsed=%.3fs\n",token,next,logit,sec);
    profile_print();
    return next;
}

static void bench_resident(shards *s,const Dsv4Cfg *c,int token){
    if(token<0||token>=c->vocab)die("input token id is outside vocabulary");
    double elapsed[4];float logit=0;int next=0;
    for(int i=0;i<4;i++){
        struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);
        next=model_step(s,c,NULL,token,0,&logit);
        clock_gettime(CLOCK_MONOTONIC,&b);
        elapsed[i]=b.tv_sec-a.tv_sec+(b.tv_nsec-a.tv_nsec)*1e-9;
    }
    double warm=elapsed[1]+elapsed[2]+elapsed[3];
    printf("  resident input=%d output=%d logit=%.9g cold=%.3fs warm=%.3fs warm_speed=%.4f tok/s\n",
           token,next,logit,elapsed[0],warm/3.0,3.0/warm);
}

static void bench_decode_two(shards*s,const Dsv4Cfg*c,int first,int second){
    if(first<0||first>=c->vocab||second<0||second>=c->vocab)die("decode token id is outside vocabulary");Dsv4State state=state_init(c,2);
#ifdef COLI_CUDA
    gpu_state_prepare(c,&state);
#endif
    float l0,l1;double a=wall_time();int o0=model_step(s,c,&state,first,0,&l0);double b=wall_time();int o1=model_step(s,c,&state,second,1,&l1);double z=wall_time();printf("  decode2 inputs=%d,%d outputs=%d,%d logits=%.9g,%.9g first=%.3fs second=%.3fs second_speed=%.4f tok/s\n",first,second,o0,o1,l0,l1,b-a,z-b,1.0/(z-b));state_free(&state);
}

static int prefill_prompt(shards*s,const Dsv4Cfg*c,Dsv4State*state,const int*ids,int n,float*logit){
#ifdef COLI_CUDA
    const char*batch=getenv("DSV4_CUDA_BATCH_PREFILL");if(batch&&atoi(batch)&&n<=c->window&&getenv("DSV4_CUDA_LAYER")&&atoi(getenv("DSV4_CUDA_LAYER"))&&!gpu_ep2()){
        int next=0;for(int p=0;p<n;p+=64){int count=n-p<64?n-p:64;next=prefill_batch_cuda(s,c,state,ids+p,count,p,logit);if(next<0)die("batched prefill is unavailable");}return next;
    }
#endif
    int next=0;for(int p=0;p<n;p++)next=model_step(s,c,state,ids[p],p,logit);return next;
}

static void generate(shards *s,const Dsv4Cfg *c,const char *model_dir,const char *arg){
    char *end;long max=strtol(arg,&end,10);if(end==arg||*end!=':'||max<1||max>128)die("--generate expects MAX:TEXT");
    char path[4096];snprintf(path,sizeof(path),"%s/tokenizer.json",model_dir);Tok tok;tok_load(&tok,path);
    int ids[DSV4_MAX_CONTEXT],n=tok_encode(&tok,end+1,(int)strlen(end+1),ids,DSV4_MAX_CONTEXT);if(n<1)ids[n++]=c->bos;
    if(n+(int)max>DSV4_MAX_CONTEXT)die("generator is currently limited to 2048 total tokens");
    Dsv4State state=state_init(c,n+(int)max);
#ifdef COLI_CUDA
    gpu_state_prepare(c,&state);
#endif
    struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);
    float logit=0;int next=prefill_prompt(s,c,&state,ids,n,&logit);
    clock_gettime(CLOCK_MONOTONIC,&b);double pre=b.tv_sec-a.tv_sec+(b.tv_nsec-a.tv_nsec)*1e-9;
    int out[128],made=0;clock_gettime(CLOCK_MONOTONIC,&a);
#ifdef COLI_CUDA
    int profile=getenv("DSV4_CUDA_PROFILE")&&atoi(getenv("DSV4_CUDA_PROFILE"));if(profile)dsv4_cuda_profiler_start();
#endif
    while(made<max){out[made++]=next;if(next==c->eos||made>=max)break;next=model_step(s,c,&state,next,n+made-1,&logit);}
#ifdef COLI_CUDA
    if(profile)dsv4_cuda_profiler_stop();
#endif
    clock_gettime(CLOCK_MONOTONIC,&b);double dec=b.tv_sec-a.tv_sec+(b.tv_nsec-a.tv_nsec)*1e-9;
    char text[16384];int z=tok_decode(&tok,out,made,text,sizeof(text));if(z<0)z=0;text[z]=0;
    printf("  generation=%s\n  generation_ids=",text);for(int i=0;i<made;i++)printf("%s%d",i?",":"",out[i]);
    printf("\n  prompt_tokens=%d ttft=%.3fs output_tokens=%d decode=%.4f tok/s total=%.4f tok/s\n",n,pre,made,made>1?(made-1)/dec:0.f,made/(pre+dec));
    profile_print();
    state_free(&state);
}

static void generate_token(shards*s,const Dsv4Cfg*c,const char*arg){
    char*colon,*end;long max=strtol(arg,&colon,10);if(colon==arg||*colon!=':'||max<1||max>128)die("--generate-token expects MAX:TOKEN");
    long token=strtol(colon+1,&end,10);if(end==colon+1||*end||token<0||token>=c->vocab)die("--generate-token token is outside vocabulary");
    Dsv4State state=state_init(c,(int)max+1);
#ifdef COLI_CUDA
    gpu_state_prepare(c,&state);
#endif
    float logit=0;double a=wall_time();int next=model_step(s,c,&state,(int)token,0,&logit);double pre=wall_time()-a;
    int out[128],made=0;a=wall_time();while(made<max){out[made++]=next;if(next==c->eos||made>=max)break;next=model_step(s,c,&state,next,made,&logit);}double dec=wall_time()-a;
    printf("  generation_ids=");for(int i=0;i<made;i++)printf("%s%d",i?",":"",out[i]);
    printf("\n  prompt_token=%ld ttft=%.3fs output_tokens=%d decode=%.4f tok/s total=%.4f tok/s\n",token,pre,made,made>1?(made-1)/dec:0.f,made/(pre+dec));
    profile_print();state_free(&state);
}

static void serve_data(const char *id,const char *data,int n){
    printf("DATA %s %d\n",id,n);if(n)fwrite(data,1,(size_t)n,stdout);putchar('\n');fflush(stdout);
}

static void serve_request(shards *s,const Dsv4Cfg *c,const char *model_dir,
                          const char *id,const char *prompt,int max_tokens){
    char path[4096];snprintf(path,sizeof(path),"%s/tokenizer.json",model_dir);Tok tok;tok_load(&tok,path);
    int *ids=malloc((DSV4_MAX_CONTEXT+1)*sizeof(*ids));
    int *out=malloc((DSV4_MAX_CONTEXT+1)*sizeof(*out));
    char *text=malloc((size_t)DSV4_MAX_CONTEXT*8+1);
    if(!ids||!out||!text)die("out of memory serving request");
    int n=tok_encode(&tok,prompt,(int)strlen(prompt),ids,DSV4_MAX_CONTEXT+1);
    if(n<1)ids[n++]=c->bos;
    if(n>DSV4_MAX_CONTEXT){
        fprintf(stderr,"[dsv4] prompt exceeds %d-token context\n",DSV4_MAX_CONTEXT);
        printf("DONE %s STAT 0 0.000 0.0 0.00 %d 1\n",id,n);fflush(stdout);goto done;
    }
    int budget=DSV4_MAX_CONTEXT-n;if(max_tokens>budget)max_tokens=budget;
    if(max_tokens<1){printf("DONE %s STAT 0 0.000 0.0 0.00 %d 1\n",id,n);fflush(stdout);goto done;}
    Dsv4State state=state_init(c,n+max_tokens);
#ifdef COLI_CUDA
    gpu_state_prepare(c,&state);
#endif
    float logit=0;int next=0;double a=wall_time();
    next=prefill_prompt(s,c,&state,ids,n,&logit);
    double decode_start=wall_time();int made=0,shown=0;
    while(made<max_tokens){
        if(next==c->eos)break;
        out[made++]=next;
        int z=tok_decode(&tok,out,made,text,(size_t)DSV4_MAX_CONTEXT*8);
        if(z>shown){serve_data(id,text+shown,z-shown);shown=z;}
        next=model_step(s,c,&state,next,n+made-1,&logit);
    }
    double end=wall_time(),dec=end-decode_start;
    printf("DONE %s STAT %d %.3f 0.0 0.00 %d %d\n",id,made,
           dec>0?made/dec:0.0,n,made>=max_tokens);fflush(stdout);
    (void)a;state_free(&state);
done:
    free(text);free(out);free(ids);
}

static void serve_loop(shards *s,const Dsv4Cfg *c,const char *model_dir){
    fputs("\x01\x01READY\x01\x01\nSTAT 0 0.000 0.0 0.00\n",stdout);fflush(stdout);
    char line[4096];
    while(fgets(line,sizeof(line),stdin)){
        char cmd[16],id[128];int slot=0,bytes=0,max_tokens=0;float temp=0,top_p=1;
        if(sscanf(line,"%15s %127s %d %d %d %f %f",cmd,id,&slot,&bytes,&max_tokens,&temp,&top_p)<5)continue;
        if(strcmp(cmd,"SUBMIT")||bytes<0)continue;
        char *prompt=malloc((size_t)bytes+1);if(!prompt)die("out of memory reading request");
        if(fread(prompt,1,(size_t)bytes,stdin)!=(size_t)bytes){free(prompt);break;}
        prompt[bytes]=0;int ch=fgetc(stdin);(void)ch;(void)slot;(void)temp;(void)top_p;
        serve_request(s,c,model_dir,id,prompt,max_tokens);free(prompt);
    }
}

int main(int argc, char **argv){
    if(argc!=2 && argc!=4){
        fprintf(stderr,"usage: %s MODEL_DIR [--tokenize TEXT | --generate MAX:TEXT | --generate-token MAX:TOKEN | --first-token TOKEN | --bench-resident TOKEN | --bench-decode2 TOKEN,TOKEN | --trace-mhc TOKEN:OUT | --trace-attn TOKEN:OUT | --trace-attn2 TOKEN,TOKEN:OUT | --trace-compressor L,TOKEN:OUT | --trace-block TOKEN:OUT]\n",argv[0]); return 2; }
    Dsv4Cfg c=load_config(argv[1]);
    long required; long long required_bytes; int indexed;
    shards tensors;
    validate_checkpoint(&tensors,argv[1],&c,&required,&required_bytes,&indexed);
#ifdef COLI_CUDA
    gpu_layer_count=c.layers;
    if(!gpu_boot())die("CUDA initialization failed");
#endif
    printf("DeepSeek-V4 checkpoint\n");
    printf("  layers=%d hidden=%d vocab=%d mHC=%dx%d\n",c.layers,c.hidden,c.vocab,c.hc_mult,c.hc_iters);
    printf("  attention=%dx%d kv_heads=%d q_lora=%d o_lora=%d rope=%d\n",
           c.heads,c.head_dim,c.kv_heads,c.q_lora,c.o_lora,c.qk_rope);
    printf("  indexer=%dx%d topk=%d hash_layers=%d\n",
           c.index_heads,c.index_head_dim,c.index_topk,c.hash_layers);
    printf("  moe=%d experts top-%d inter=%d shared=%d scale=%.3g\n",
           c.experts,c.topk,c.moe_inter,c.shared_experts,c.routed_scale);
    printf("  dspark=block-%d rank-%d noise-token-%d\n",
           c.dspark_block,c.dspark_rank,c.dspark_noise);
    printf("  tensors=%d indexed, %ld required (%.2f GiB)\n",
           indexed,required,required_bytes/(1024.0*1024.0*1024.0));
#ifdef COLI_CUDA
    int serving=argc==2&&getenv("SERVE")&&getenv("SERVE")[0]=='1';
    if(serving||(argc==4&&strcmp(argv[2],"--tokenize")))gpu_preload(&tensors,&c);
#endif
    if(argc==2&&getenv("SERVE")&&getenv("SERVE")[0]=='1')serve_loop(&tensors,&c,argv[1]);
    if(argc==4){
        if(!strcmp(argv[2],"--tokenize")){
            char path[4096]; snprintf(path,sizeof(path),"%s/tokenizer.json",argv[1]);
            Tok tok; tok_load(&tok,path);
            int ids[4096], n=tok_encode(&tok,argv[3],(int)strlen(argv[3]),ids,4096);
            printf("  token_ids=");
            for(int i=0;i<n;i++) printf("%s%d",i?",":"",ids[i]);
            putchar('\n');
        } else if(!strcmp(argv[2],"--first-token")){
            char *end;long token=strtol(argv[3],&end,10);if(end==argv[3]||*end)die("--first-token expects TOKEN");
            first_token(&tensors,&c,(int)token);
        } else if(!strcmp(argv[2],"--bench-resident")){
            char *end;long token=strtol(argv[3],&end,10);if(end==argv[3]||*end)die("--bench-resident expects TOKEN");
            bench_resident(&tensors,&c,(int)token);
        } else if(!strcmp(argv[2],"--bench-decode2")){
            char *comma,*end;long first=strtol(argv[3],&comma,10);if(comma==argv[3]||*comma!=',')die("--bench-decode2 expects TOKEN,TOKEN");long second=strtol(comma+1,&end,10);if(end==comma+1||*end)die("--bench-decode2 expects TOKEN,TOKEN");bench_decode_two(&tensors,&c,(int)first,(int)second);
        } else if(!strcmp(argv[2],"--generate")){
            generate(&tensors,&c,argv[1],argv[3]);
        } else if(!strcmp(argv[2],"--generate-token")){
            generate_token(&tensors,&c,argv[3]);
        } else if(!strcmp(argv[2],"--trace-mhc")){
            char *end; long token=strtol(argv[3],&end,10);
            if(end==argv[3]||*end!=':') die("--trace-mhc expects TOKEN:OUT");
            trace_first_mhc(&tensors,&c,(int)token,end+1);
        } else if(!strcmp(argv[2],"--trace-attn")){
            char *end; long token=strtol(argv[3],&end,10);
            if(end==argv[3]||*end!=':')die("--trace-attn expects TOKEN:OUT");
            float *out=first_attention(&tensors,&c,(int)token,end+1);free(out);
        } else if(!strcmp(argv[2],"--trace-attn2")){
            char *comma,*colon;long a=strtol(argv[3],&comma,10);if(comma==argv[3]||*comma!=',')die("--trace-attn2 expects TOKEN,TOKEN:OUT");
            long b=strtol(comma+1,&colon,10);if(colon==comma+1||*colon!=':')die("--trace-attn2 expects TOKEN,TOKEN:OUT");
            trace_attention_two(&tensors,&c,(int)a,(int)b,colon+1);
        } else if(!strcmp(argv[2],"--trace-compressor")){
            char *comma,*colon;long l=strtol(argv[3],&comma,10);if(comma==argv[3]||*comma!=',')die("--trace-compressor expects L,TOKEN:OUT");
            long token=strtol(comma+1,&colon,10);if(colon==comma+1||*colon!=':')die("--trace-compressor expects L,TOKEN:OUT");
            trace_compressor(&tensors,&c,(int)l,(int)token,colon+1);
        } else if(!strcmp(argv[2],"--trace-block")){
            char *end;long token=strtol(argv[3],&end,10);if(end==argv[3]||*end!=':')die("--trace-block expects TOKEN:OUT");
            float *out=first_block(&tensors,&c,(int)token,end+1);free(out);
        } else die("unknown option");
    }
    if(argc==2)printf("checkpoint index accepted\n");
#ifdef COLI_CUDA
    gpu_shutdown();
#endif
    return 0;
}
