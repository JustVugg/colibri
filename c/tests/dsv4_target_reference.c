/* Production CPU compressor is the oracle, with projected inputs supplied. */
#define COLI_V4_UNIT_COMPRESSOR
#include "../deepseek_v4.c"
#include <assert.h>
const void *coli_v4_layer_data(const ColiDeepSeekV4LayerWeights *w,const char *name,
                               const ColiDeepSeekV4TensorSpec **spec) {
    (void)spec;
    for(size_t i=0;i<w->plan.tensor_count;i++)
        if(!strcmp(name,w->plan.tensors[i].name))return w->data[i];
    return NULL;
}
void target_reference(int ratio,int rows,const float *kv,const float *gate,
    const float *ape,float *co,float *si,float *out) {
    ColiDeepSeekV4Config c={0};
    c.hidden_size=4096;c.head_dim=512;c.qk_rope_head_dim=64;
    c.rms_norm_eps=1e-6f;c.original_max_position_embeddings=4096;
    c.compress_rope_theta=10000;c.rope_factor=1;c.rope_beta_fast=32;c.rope_beta_slow=1;
    ColiDeepSeekV4LayerWeights w={0};uint16_t norm[512];
    for(int i=0;i<512;i++)norm[i]=0x3f80;
    w.plan.compression_ratio=ratio;w.plan.tensor_count=2;
    strcpy(w.plan.tensors[0].name,"layers.0.attn.compressor.ape");w.data[0]=(void*)ape;
    strcpy(w.plan.tensors[1].name,"layers.0.attn.compressor.norm.weight");w.data[1]=norm;
    ColiDeepSeekV4CompressorState *state=NULL;char error[256]={0};
    assert(!coli_v4_compressor_create(&state,&w,&c,error,sizeof(error)));
    assert(!coli_v4_rope_precompute_range(co,si,64,0,rows,4096,10000,1,32,1));
    int width=ratio==4?1024:512;float row[512];
    for(int p=0;p<rows;p++) {
        int produced=0;
        assert(!coli_v4_compressor_advance(state,row,&produced,kv+(size_t)p*width,
            gate+(size_t)p*width,p,error,sizeof(error)));
        if(produced)memcpy(out+(size_t)(p/ratio)*512,row,sizeof(row));
    }
    coli_v4_compressor_destroy(state);
}
