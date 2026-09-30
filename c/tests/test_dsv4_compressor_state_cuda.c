/* Compile as C with the production math/native-quant objects and CUDA backend. */
#define COLI_V4_UNIT_COMPRESSOR
#ifndef COLI_V4_GPU_TIER
#define COLI_V4_GPU_TIER
#endif
#include "../deepseek_v4.c"
#include "../backend_cuda_dsv4.h"
#include <assert.h>

static int use_gpu, fail_after_values;
const void *coli_v4_layer_data(const ColiDeepSeekV4LayerWeights *w,const char *name,
                               const ColiDeepSeekV4TensorSpec **spec) {
    (void)spec;
    for(size_t i=0;i<w->plan.tensor_count;i++)
        if(!strcmp(name,w->plan.tensors[i].name))return w->data[i];
    return NULL;
}
int coli_v4_gpu_compressor_project(const ColiDeepSeekV4LayerWeights *w,const char *prefix,
                                    float *v,float *g,const float *x) {
    (void)prefix;
    if(!use_gpu)return -1;
    if(fail_after_values){v[0]=123.f;g[0]=456.f;return -1;}
    return dsv4_cuda_compressor_project(w->gpu[0],w->gpu[1],x,v,g)?0:-1;
}
static void check(int device,int ratio,int dim,int rotate) {
    const int hidden=256,projection=(ratio==4?2:1)*dim;
    ColiDeepSeekV4LayerWeights w={0};
    w.plan.compression_ratio=ratio;w.plan.tensor_count=4;
    const char *keys[]={"wkv.weight","wgate.weight","ape","norm.weight"};
    const char *prefix=rotate?"attn.indexer.compressor":"attn.compressor";
    for(int i=0;i<4;i++)snprintf(w.plan.tensors[i].name,COLI_V4_MAX_TENSOR_NAME,
                                "layers.0.%s.%s",prefix,keys[i]);
    w.data[0]=malloc((size_t)projection*hidden*2);
    w.data[1]=malloc((size_t)projection*hidden*2);
    w.data[2]=malloc((size_t)ratio*projection*4);
    w.data[3]=malloc(dim*2);
    for(int i=0;i<projection*hidden;i++) {
        ((uint16_t*)w.data[0])[i]=0x3a00+i%384;
        ((uint16_t*)w.data[1])[i]=0x3900+i%513;
    }
    for(int i=0;i<ratio*projection;i++)((float*)w.data[2])[i]=(i%17-8)*.01f;
    for(int i=0;i<dim;i++)((uint16_t*)w.data[3])[i]=0x3f80;
    assert(dsv4_cuda_upload_compressor((Dsv4CudaTensor**)&w.gpu[0],w.data[0],projection,hidden,device));
    assert(dsv4_cuda_upload_compressor((Dsv4CudaTensor**)&w.gpu[1],w.data[1],projection,hidden,device));
    ColiDeepSeekV4Config config={0};
    config.hidden_size=hidden;config.head_dim=dim;config.qk_rope_head_dim=64;
    config.rms_norm_eps=1e-6f;config.original_max_position_embeddings=4096;
    config.compress_rope_theta=10000;config.rope_factor=1;
    config.rope_beta_fast=32;config.rope_beta_slow=1;
    ColiDeepSeekV4CompressorOptions options={prefix,dim,rotate};
    ColiDeepSeekV4CompressorState *cpu=NULL,*gpu=NULL;
    char error[256]={0};
    assert(!coli_v4_compressor_create_with_options(&cpu,&w,&config,&options,error,sizeof(error)));
    assert(!coli_v4_compressor_create_with_options(&gpu,&w,&config,&options,error,sizeof(error)));
    float input[hidden],a[dim],b[dim];
    for(int pass=0;pass<2;pass++) {
        coli_v4_compressor_reset(cpu);coli_v4_compressor_reset(gpu);
        for(int pos=0;pos<2*ratio+3;pos++) {
            for(int i=0;i<hidden;i++)input[i]=(i%31-15)*.019f+(pos%9)*.013f;
            int ca=0,cb=0;
            use_gpu=0;
            assert(!coli_v4_compressor_step(cpu,a,&ca,input,pos,error,sizeof(error)));
            use_gpu=1;fail_after_values=pass&&pos%3==0;
            assert(!coli_v4_compressor_step(gpu,b,&cb,input,pos,error,sizeof(error)));
            assert(ca==cb && ca==((pos+1)%ratio==0));
            assert(!ca || !memcmp(a,b,dim*4));
            size_t bytes=(size_t)cpu->state_rows*projection*4;
            assert(!memcmp(cpu->kv_state,gpu->kv_state,bytes));
            assert(!memcmp(cpu->score_state,gpu->score_state,bytes));
        }
    }
    coli_v4_compressor_destroy(cpu);coli_v4_compressor_destroy(gpu);
    dsv4_cuda_tensor_free(w.gpu[0]);dsv4_cuda_tensor_free(w.gpu[1]);
    for(int i=0;i<4;i++)free(w.data[i]);
    printf("device=%d ratio=%d dim=%d indexer=%d state/output/reset/partial-failure exact\n",device,ratio,dim,rotate);
}
int main(void) {
    int devices[]={0,5};assert(dsv4_cuda_init(devices,2));
    for(int i=0;i<2;i++) {check(devices[i],4,512,0);check(devices[i],128,512,0);check(devices[i],4,128,1);}
    dsv4_cuda_shutdown();
}
