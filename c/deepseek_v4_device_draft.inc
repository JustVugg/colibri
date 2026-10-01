#if defined(COLI_V4_GPU_TIER) && !defined(_WIN32)
#include "backend_cuda_dsv4.h"
typedef struct {
    Dsv4CudaTargetWorkspace *work;
    Dsv4CudaTargetLayer *layer;
    Dsv4CudaExpertSet *experts;
    Dsv4CudaTensor *owned[24];
    int count,device;
} V4DeviceDraftStage;
typedef struct {V4DeviceDraftStage stages[3];} V4DeviceDraft;
static void coli_v4_device_draft_free(ColiV4Engine *engine) {
    V4DeviceDraft *d=engine->gpu.device_draft;if(!d)return;
    for(int i=0;i<3;i++) {
        V4DeviceDraftStage *s=&d->stages[i];
        dsv4_cuda_target_workspace_free(s->work);dsv4_cuda_target_layer_free(s->layer);
        dsv4_cuda_expert_set_free(s->experts);
        for(int j=0;j<s->count;j++)dsv4_cuda_tensor_free(s->owned[j]);
    }
    free(d);engine->gpu.device_draft=NULL;
}
static Dsv4CudaTensor *v4_dd_float(V4DeviceDraftStage *s,const float *data,int rows,int cols) {
    if(s->count==24)return NULL;
    Dsv4CudaTensor **t=&s->owned[s->count++];
    return dsv4_cuda_upload_f32(t,data,rows,cols,s->device)?*t:NULL;
}
static Dsv4CudaTensor *v4_dd_fp8(V4DeviceDraftStage *s,const uint8_t *packed,const float *scales,int rows,int cols,int grouped) {
    if(s->count==24)return NULL;
    size_t blocks=(size_t)(rows/128)*(cols/128);
    uint8_t *encoded=malloc(blocks),*unpacked=NULL;
    if(!encoded)return NULL;
    for(size_t i=0;i<blocks;i++)encoded[i]=(uint8_t)(ilogbf(scales[i])+127);
    const uint8_t *data=packed;
#ifdef __AVX2__
    unpacked=malloc((size_t)rows*cols);
    if(!unpacked){free(encoded);return NULL;}
    for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)
        unpacked[(size_t)r*cols+c]=packed[((size_t)(r/8)*cols+c)*8+r%8];
    data=unpacked;
#endif
    Dsv4CudaTensor **t=&s->owned[s->count++];
    int good=grouped?dsv4_cuda_upload_fp8_bf16(t,data,encoded,rows,cols,s->device)
                    :dsv4_cuda_upload_fp8(t,data,encoded,rows,cols,s->device);
    free(unpacked);free(encoded);return good?*t:NULL;
}
static int v4_dd_prepare(ColiV4Engine *engine,V4DeviceDraftStage *s,V4DSparkStage *stage,
    const ColiSafetensorsIndex *index,const ColiDeepSeekV4Config *c,void *mirrors) {
    s->device=coli_v4_gpu_dspark_resident_device(mirrors,stage->stage);
    if(s->device<0 || c->hc_mult!=4 || c->hidden_size!=4096 || c->o_lora_rank!=1024 ||
        c->q_lora_rank!=1024 || c->num_attention_heads!=64 || c->head_dim!=512 ||
        c->qk_rope_head_dim!=64 || c->o_groups!=8 || c->n_routed_experts!=256 || c->num_experts_per_tok!=6)return -1;
    int budget=256;
    for(int e=0;e<256;e++) {
        ColiExpertView view;
        if(!v4_ds_expert(index,stage,e,&view,&budget,mirrors))return -1;
    }
    Dsv4CudaTargetWeights w={0};
#define FP8(field,name,rows,cols,grouped) w.field=v4_dd_fp8(s,stage->name##_w,stage->name##_s,rows,cols,grouped)
    FP8(qa,wq_a,1024,4096,0);FP8(qb,wq_b,32768,1024,0);FP8(kv,wkv,512,4096,0);
    FP8(wa,wo_a,8192,4096,1);FP8(wb,wo_b,4096,8192,0);
#undef FP8
    Dsv4CudaTensor *sg=v4_dd_fp8(s,stage->sh1_w,stage->sh1_s,c->moe_intermediate_size,4096,0);
    Dsv4CudaTensor *su=v4_dd_fp8(s,stage->sh3_w,stage->sh3_s,c->moe_intermediate_size,4096,0);
    Dsv4CudaTensor *sd=v4_dd_fp8(s,stage->sh2_w,stage->sh2_s,4096,c->moe_intermediate_size,0);
#define F(field,name,rows,cols) w.field=v4_dd_float(s,stage->name.data,rows,cols)
    F(hc_fn[0],hc_attn_fn,24,16384);F(hc_scale[0],hc_attn_scale,3,1);F(hc_base[0],hc_attn_base,24,1);
    F(hc_fn[1],hc_ffn_fn,24,16384);F(hc_scale[1],hc_ffn_scale,3,1);F(hc_base[1],hc_ffn_base,24,1);
    F(norm[0],attn_norm,4096,1);F(norm[1],ffn_norm,4096,1);
    F(qnorm,q_norm,1024,1);F(kvnorm,kv_norm,512,1);F(sink,attn_sink,64,1);
    F(gate,gate_w,256,4096);F(bias,gate_b,256,1);
#undef F
    for(int i=0;i<s->count;i++)if(!s->owned[i])return -1;
    s->experts=coli_v4_gpu_dspark_resident_set(mirrors,stage->stage,sg,su,sd);
    if(!s->experts)return -1;
    w.experts=s->experts;w.hidden=4096;w.rank=1024;w.heads=64;w.dim=512;w.rope=64;
    w.groups=8;w.window=128;w.capacity=2048;w.draft=1;
    w.eps=c->rms_norm_eps;w.routed_scale=c->routed_scaling_factor;w.swiglu_limit=c->swiglu_limit;
    float *co=malloc(2048*32*4),*si=malloc(2048*32*4);
    int good=co && si && !coli_v4_rope_precompute_range(co,si,64,0,2048,0,c->rope_theta,c->rope_factor,c->rope_beta_fast,c->rope_beta_slow);
    if(good)s->layer=dsv4_cuda_target_layer_create(&w,co,si);
    free(co);free(si);
    s->work=dsv4_cuda_target_workspace_create(s->device,5);
    if(!s->layer || !s->work)return -1;
    fprintf(stderr,"[MTP] device stage=%d device=%d resident_experts=256\n",stage->stage,s->device);
    return 0;
}
static int v4_ds_device_stage(ColiV4Engine *engine,float *state,V4DSparkStage *stage,int64_t position,
    const ColiSafetensorsIndex *index,const ColiDeepSeekV4Config *config,void *mirrors) {
    if(!engine->gpu.device_draft)engine->gpu.device_draft=calloc(1,sizeof(V4DeviceDraft));
    engine->gpu.device_draft_free=coli_v4_device_draft_free;
    V4DeviceDraft *d=engine->gpu.device_draft;if(!d)return -1;
    V4DeviceDraftStage *s=&d->stages[stage->stage];
    if(!s->layer && v4_dd_prepare(engine,s,stage,index,config,mirrors))goto fail;
    int tokens[5]={0};
    if(position<0 || position>2043 ||
        !dsv4_cuda_target_input(s->work,state,tokens,(int)position,5) ||
        !dsv4_cuda_target_past(s->layer,stage->kv,stage->kv_abs) ||
        !dsv4_cuda_target_stage(s->work,&s->layer,1,3,5,1) ||
        !dsv4_cuda_target_output(s->work,state,5))goto fail;
    return 0;
fail:
    fprintf(stderr,"[MTP] device stage failed stage=%d\n",stage->stage);
    coli_v4_device_draft_free(engine);return -1;
}
#endif
