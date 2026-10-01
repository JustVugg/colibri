/* Engine-owned prepared target. This path never falls back after device state
 * has advanced. Unsupported requests are rejected before their first layer. */
#ifndef _WIN32
typedef struct {
    Dsv4CudaTargetLayer *layer;
    Dsv4CudaExpertSet *experts;
    Dsv4CudaTensor *owned[5];
} V4DeviceLayer;
typedef struct {
    V4DeviceLayer layers[COLI_V4_RESIDENT_MAX_LAYERS];
    Dsv4CudaTargetWorkspace *work[16];
    const void *owner;
    int cursor, capacity;
} V4DeviceTarget;

static void v4_device_target_free(ColiV4Engine *engine) {
    V4DeviceTarget *target = engine->gpu.device_target;
    if (!target) return;
    for (int i = 0; i < 16; i++) dsv4_cuda_target_workspace_free(target->work[i]);
    for (int i = 0; i < COLI_V4_RESIDENT_MAX_LAYERS; i++) {
        V4DeviceLayer *l = &target->layers[i];
        dsv4_cuda_target_layer_free(l->layer);
        dsv4_cuda_expert_set_free(l->experts);
        for (int j = 0; j < 5; j++) dsv4_cuda_tensor_free(l->owned[j]);
    }
    free(target); engine->gpu.device_target = NULL;
}

static int v4_device_layer_prepare(ColiV4Engine *engine, V4DeviceTarget *target,
    const ColiDeepSeekV4LayerWeights *lw, int owner) {
    int id = lw->plan.layer, device = engine->gpu.devices[owner];
    V4DeviceLayer *l = &target->layers[id];
    if (l->layer) return 0;
    const ColiDeepSeekV4Config *c = &engine->config;
    V4GpuExpertMirrorCache *cache = v4_gpu_expert_cache(engine->experts, id);
    if (!cache || cache->experts_per_layer != 256) return -1;
    Dsv4CudaTargetWeights w = {0};
#define MIRROR(key) ((Dsv4CudaTensor *)coli_v4_layer_gpu(lw, key))
    w.qa=MIRROR("attn.wq_a"); w.qb=MIRROR("attn.wq_b"); w.kv=MIRROR("attn.wkv");
    w.wa=MIRROR("attn.wo_a"); w.wb=MIRROR("attn.wo_b");
    w.gate=MIRROR("ffn.gate"); w.bias=MIRROR("ffn.gate.bias");
    w.comp_kv=MIRROR("attn.compressor.wkv"); w.comp_gate=MIRROR("attn.compressor.wgate");
    const char *branch[2] = {"attn", "ffn"};
    for (int b=0;b<2;b++) {
        char key[64]; snprintf(key,sizeof(key),"hc_%s_fn",branch[b]); w.hc_fn[b]=MIRROR(key);
        snprintf(key,sizeof(key),"hc_%s_scale",branch[b]); w.hc_scale[b]=MIRROR(key);
        snprintf(key,sizeof(key),"hc_%s_base",branch[b]); w.hc_base[b]=MIRROR(key);
        snprintf(key,sizeof(key),"%s_norm",branch[b]); w.norm[b]=MIRROR(key);
        if (!w.hc_fn[b] || !w.hc_scale[b] || !w.hc_base[b] || !w.norm[b]) return -1;
    }
    ColiDeepSeekV4LayerWeights *mutable_weights = &engine->dense_resident.layers[id];
    long long bytes=0;
    w.qnorm=l->owned[0]=v4_gpu_upload_norm_f32(mutable_weights,device,"attn.q_norm",c->q_lora_rank,&bytes);
    w.kvnorm=l->owned[1]=v4_gpu_upload_norm_f32(mutable_weights,device,"attn.kv_norm",c->head_dim,&bytes);
    w.sink=l->owned[2]=v4_gpu_upload_f32_tensor(mutable_weights,device,"attn.attn_sink",c->num_attention_heads,1,&bytes);
    int ratio=lw->plan.compression_ratio, width=(ratio==4?2:1)*c->head_dim;
    if (ratio) {
        w.comp_norm=l->owned[3]=v4_gpu_upload_norm_f32(mutable_weights,device,"attn.compressor.norm",c->head_dim,&bytes);
        w.ape=l->owned[4]=v4_gpu_upload_f32_tensor(mutable_weights,device,"attn.compressor.ape",ratio,width,&bytes);
        if (!w.comp_kv || !w.comp_gate || !w.comp_norm || !w.ape) return -1;
    }
    if (!w.qa || !w.qb || !w.kv || !w.wa || !w.wb || !w.gate || !w.qnorm || !w.kvnorm || !w.sink) return -1;
    Dsv4CudaTensor *g[256], *u[256], *d[256];
    pthread_mutex_lock(&cache->mutex);
    int good=1;
    for (int e=0;e<256;e++) {
        V4GpuExpertMirror *entry=v4_gpu_expert_find(cache,id,e);
        if (!entry) {good=0; break;}
        g[e]=entry->gate; u[e]=entry->up; d[e]=entry->down;
    }
    if (good) l->experts=dsv4_cuda_expert_set_create(g,u,d,256,MIRROR("ffn.shared_experts.w1"),
        MIRROR("ffn.shared_experts.w3"),MIRROR("ffn.shared_experts.w2"));
    pthread_mutex_unlock(&cache->mutex);
    if (!l->experts) return -1;
    if (lw->plan.uses_hash_router) {
        char key[96]; snprintf(key,sizeof(key),"layers.%d.ffn.gate.tid2eid",id);
        const int64_t *map=coli_v4_layer_data(lw,key,NULL);
        if (!map || !dsv4_cuda_expert_set_upload_hash(l->experts,map,c->vocab_size,6)) return -1;
    } else if (!w.bias) return -1;
    w.experts=l->experts; w.hidden=c->hidden_size; w.rank=c->q_lora_rank;
    w.heads=c->num_attention_heads; w.dim=c->head_dim; w.rope=c->qk_rope_head_dim;
    w.groups=c->o_groups; w.window=c->sliding_window; w.ratio=ratio; w.capacity=target->capacity;
    w.eps=c->rms_norm_eps; w.routed_scale=c->routed_scaling_factor; w.swiglu_limit=c->swiglu_limit;
    size_t count=(size_t)target->capacity*c->qk_rope_head_dim/2;
    float *co=malloc(count*sizeof(float)), *si=malloc(count*sizeof(float));
    good=co && si && !coli_v4_rope_precompute_range(co,si,c->qk_rope_head_dim,0,target->capacity,
        ratio?c->original_max_position_embeddings:0,ratio?c->compress_rope_theta:c->rope_theta,
        c->rope_factor,c->rope_beta_fast,c->rope_beta_slow);
    if (good) l->layer=dsv4_cuda_target_layer_create(&w,co,si);
    free(co); free(si);
    return l->layer?0:-1;
#undef MIRROR
}

int coli_v4_gpu_device_target(ColiV4Engine *engine, const void *owner, float *output,
    float *taps, const float *input, const int *tokens, int start, int batch,
    ColiV4SessionAbortFn abort_fn, void *abort_context, char *error, size_t error_size) {
    if (!engine || !engine->gpu.enabled || !owner || !output || !input || !tokens ||
        batch<1 || batch>128 || start<0 || engine->config.hc_mult!=4) return -1;
    V4DeviceTarget *s=engine->gpu.device_target;
    if (!s) {
        s=calloc(1,sizeof(*s)); if (!s) return -1;
        s->capacity=engine->config.index_topk*4;
        if (s->capacity>2048) s->capacity=2048;
        engine->gpu.device_target=s;
    }
    if (start>s->capacity-batch || (start && (s->owner!=owner || start>s->cursor))) {
        snprintf(error,error_size,"device target requires a valid retained prefix within %d tokens",s->capacity); return -1;
    }
    for (int t=0;t<batch;t++) if (tokens[t]<0 || tokens[t]>=engine->config.vocab_size) return -1;
    s->owner=owner; s->cursor=start; /* Never expose a partially written suffix. */
    int previous=-1, total=engine->config.num_hidden_layers;
    const char *graph_env=getenv("V4_DEVICE_GRAPH");
    int graph=graph_env && atoi(graph_env)!=0;
    for (int first=0;first<total;) {
        if (abort_fn && abort_fn(abort_context)) {snprintf(error,error_size,"CANCELLED");goto fail;}
        int slot=coli_v4_gpu_layer_owner(first,engine->gpu.device_count,total), end=first+1;
        while (end<total && coli_v4_gpu_layer_owner(end,engine->gpu.device_count,total)==slot) ++end;
        if (!s->work[slot]) s->work[slot]=dsv4_cuda_target_workspace_create(engine->gpu.devices[slot],128);
        if (!s->work[slot]) goto fail;
        Dsv4CudaTargetLayer *layers[COLI_V4_RESIDENT_MAX_LAYERS];
        for (int id=first;id<end;id++) {
            if (!s->layers[id].layer) {
                ColiDeepSeekV4LayerWeights lw;
                if (coli_v4_layer_load(engine,&lw,&engine->config,engine->target_index,id,error,error_size)) goto fail;
                int rc=v4_device_layer_prepare(engine,s,&lw,slot);
                coli_v4_layer_free(engine,&lw);
                if (rc) goto fail;
            }
            layers[id-first]=s->layers[id].layer;
        }
        if (previous<0) {
            if (!dsv4_cuda_target_input(s->work[slot],input,tokens,start,batch)) goto fail;
        } else if (!dsv4_cuda_target_transfer(s->work[slot],s->work[previous],batch)) goto fail;
        if (!dsv4_cuda_target_stage(s->work[slot],layers,end-first,total-3-first,batch,graph)) goto fail;
        int tap_first=first>total-3?first-(total-3):0;
        int tap_end=end>total-3?end-(total-3):0;
        if (taps && tap_end>tap_first && !dsv4_cuda_target_taps(s->work[slot],taps,batch,tap_first,tap_end-tap_first)) goto fail;
        previous=slot; first=end;
    }
    if (!dsv4_cuda_target_output(s->work[previous],output,batch)) goto fail;
    s->cursor=start+batch; return 0;
fail:
    v4_device_target_free(engine);
    if (error && error_size && !*error) snprintf(error,error_size,"device target failed; session state is invalid");
    return -1;
}
int coli_v4_gpu_device_target_retain(ColiV4Engine *engine, const void *owner, int position) {
    V4DeviceTarget *s=engine?engine->gpu.device_target:NULL;
    if (!s || s->owner!=owner || position<0 || position>s->cursor) return -1;
    s->cursor=position; return 0;
}
int coli_v4_gpu_draft_head(ColiV4Engine *engine,const float *input,int batch,int anchor,
    const uint16_t *markov1,const uint16_t *markov2,const float *weights,
    int rank,int *ids,float *confidence) {
    if (!engine || !engine->gpu.enabled || !engine->gpu.head) return -1;
    if (!engine->gpu.draft_head) engine->gpu.draft_head=dsv4_cuda_draft_head_create(
        engine->gpu.head,markov1,markov2,weights,rank);
    return dsv4_cuda_draft_head(engine->gpu.draft_head,input,batch,anchor,ids,confidence)?0:-1;
}
int coli_v4_gpu_dspark_resident_device(void *mirrors,int stage) {
    V4GpuExpertMirrorCache *c=mirrors;
    while(c && !(stage>=c->first_layer && stage<c->end_layer))c=c->next;
    return c && c->capacity>=256 && c->end_layer-c->first_layer==1?c->device:-1;
}
void *coli_v4_gpu_dspark_resident_set(void *mirrors,int stage,void *gate,void *up,void *down) {
    V4GpuExpertMirrorCache *c=mirrors;
    while(c && !(stage>=c->first_layer && stage<c->end_layer))c=c->next;
    if(!c || coli_v4_gpu_dspark_resident_device(mirrors,stage)<0)return NULL;
    Dsv4CudaTensor *g[256],*u[256],*d[256];int good=1;
    pthread_mutex_lock(&c->mutex);
    for(int e=0;e<256;e++) {
        V4GpuExpertMirror *entry=v4_gpu_expert_find(c,stage,e);
        if(!entry){good=0;break;}g[e]=entry->gate;u[e]=entry->up;d[e]=entry->down;
    }
    void *set=good?dsv4_cuda_expert_set_create(g,u,d,256,gate,up,down):NULL;
    pthread_mutex_unlock(&c->mutex);return set;
}
#else
static void v4_device_target_free(ColiV4Engine *engine) { (void)engine; }
int coli_v4_gpu_device_target(ColiV4Engine *engine, const void *owner, float *output,
    float *taps, const float *input, const int *tokens, int start, int batch,
    ColiV4SessionAbortFn abort_fn, void *abort_context, char *error, size_t error_size) {
    snprintf(error,error_size,"device target is not available in the Windows loader"); return -1;
}
int coli_v4_gpu_device_target_retain(ColiV4Engine *engine, const void *owner, int position) {return -1;}
int coli_v4_gpu_draft_head(ColiV4Engine *engine,const float *input,int batch,int anchor,
    const uint16_t *markov1,const uint16_t *markov2,const float *weights,
    int rank,int *ids,float *confidence) {return -1;}
#endif
