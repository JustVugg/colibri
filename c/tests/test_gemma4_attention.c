#define _CRT_SECURE_NO_WARNINGS

#include "../gemma4_model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE "gemma4-attention-test.gguf"

static void w8(FILE *f, uint8_t v) { fwrite(&v, 1, 1, f); }
static void w32(FILE *f, uint32_t v) {
    uint8_t b[4]={(uint8_t)v,(uint8_t)(v>>8),(uint8_t)(v>>16),(uint8_t)(v>>24)};
    fwrite(b,1,4,f);
}
static void w64(FILE *f, uint64_t v) { w32(f,(uint32_t)v); w32(f,(uint32_t)(v>>32)); }
static void wf(FILE *f, float v) { uint32_t b; memcpy(&b,&v,4); w32(f,b); }
static void ws(FILE *f, const char *s) { size_t n=strlen(s); w64(f,n); fwrite(s,1,n,f); }
static void mu(FILE *f,const char *k,uint32_t v){ws(f,k);w32(f,4);w32(f,v);}
static void mf(FILE *f,const char *k,float v){ws(f,k);w32(f,6);wf(f,v);}
static void tensor(FILE *f,const char *name,uint32_t type,uint32_t nd,
                   uint64_t d0,uint64_t d1,uint64_t off){
    ws(f,name);w32(f,nd);w64(f,d0);if(nd==2)w64(f,d1);w32(f,type);w64(f,off);
}
static void q4_identity_rows(FILE *f, uint32_t rows, uint32_t columns) {
    uint32_t row, block, i;
    for (row=0; row<rows; ++row) {
        for(block=0;block<columns/32;++block){
            uint8_t q[16];
            uint16_t half_one=0x3c00;
            uint32_t target=row%columns;
            memset(q,0x88,sizeof(q));
            if(target/32==block){
                target%=32;
                if(target<16)q[target]=(uint8_t)((q[target]&0xf0)|9);
                else q[target%16]=(uint8_t)((q[target%16]&0x0f)|0x90);
            }
            w8(f,(uint8_t)half_one);w8(f,(uint8_t)(half_one>>8));
            for(i=0;i<16;++i)w8(f,q[i]);
        }
    }
}

static int make_fixture(void) {
    FILE *f=fopen(FIXTURE,"wb");
    long p;
    uint32_t i;
    if(!f)return -1;
    fwrite("GGUF",1,4,f);w32(f,3);w64(f,21);w64(f,17);
    ws(f,"general.architecture");w32(f,8);ws(f,"gemma4");
    mu(f,"general.alignment",32);mu(f,"gemma4.block_count",1);
    mu(f,"gemma4.embedding_length",32);mu(f,"gemma4.expert_count",2);
    mu(f,"gemma4.expert_used_count",1);mu(f,"gemma4.expert_feed_forward_length",32);
    mu(f,"gemma4.attention.sliding_window",16);mu(f,"gemma4.attention.head_count",2);
    mu(f,"gemma4.attention.key_length",64);mu(f,"gemma4.attention.key_length_swa",32);
    mu(f,"gemma4.attention.value_length",64);mu(f,"gemma4.attention.value_length_swa",32);
    mf(f,"gemma4.attention.layer_norm_rms_epsilon",1e-6F);
    ws(f,"gemma4.attention.head_count_kv");w32(f,9);w32(f,5);w64(f,1);w32(f,1);
    ws(f,"gemma4.attention.sliding_window_pattern");w32(f,9);w32(f,7);w64(f,1);w8(f,1);
    mf(f,"gemma4.rope.freq_base_swa",10000.0F);
    tensor(f,"token_embd.weight",COLI_GGML_TYPE_F32,2,32,32,0);
    tensor(f,"blk.0.attn_q.weight",COLI_GGML_TYPE_Q4_0,2,32,64,4096);
    tensor(f,"blk.0.attn_k.weight",COLI_GGML_TYPE_Q4_0,2,32,32,5248);
    tensor(f,"blk.0.attn_v.weight",COLI_GGML_TYPE_Q4_0,2,32,32,5824);
    tensor(f,"blk.0.attn_norm.weight",COLI_GGML_TYPE_F32,1,32,0,6400);
    tensor(f,"blk.0.attn_q_norm.weight",COLI_GGML_TYPE_F32,1,32,0,6528);
    tensor(f,"blk.0.attn_k_norm.weight",COLI_GGML_TYPE_F32,1,32,0,6656);
    tensor(f,"blk.0.attn_output.weight",COLI_GGML_TYPE_Q4_0,2,64,32,6784);
    tensor(f,"blk.0.ffn_gate.weight",COLI_GGML_TYPE_Q4_0,2,32,32,7936);
    tensor(f,"blk.0.ffn_up.weight",COLI_GGML_TYPE_Q4_0,2,32,32,8512);
    tensor(f,"blk.0.ffn_down.weight",COLI_GGML_TYPE_Q4_0,2,32,32,9088);
    tensor(f,"blk.0.ffn_gate_inp.scale",COLI_GGML_TYPE_F32,1,32,0,9664);
    tensor(f,"blk.0.ffn_gate_inp.weight",COLI_GGML_TYPE_F32,2,32,2,9792);
    tensor(f,"blk.0.ffn_down_exps.scale",COLI_GGML_TYPE_F32,1,2,0,10048);
    tensor(f,"blk.0.ffn_norm.weight",COLI_GGML_TYPE_F32,1,32,0,10080);
    tensor(f,"blk.0.post_attention_norm.weight",COLI_GGML_TYPE_F32,1,32,0,10208);
    tensor(f,"blk.0.post_ffw_norm.weight",COLI_GGML_TYPE_F32,1,32,0,10336);
    tensor(f,"blk.0.post_ffw_norm_1.weight",COLI_GGML_TYPE_F32,1,32,0,10464);
    tensor(f,"blk.0.post_ffw_norm_2.weight",COLI_GGML_TYPE_F32,1,32,0,10592);
    tensor(f,"blk.0.pre_ffw_norm_2.weight",COLI_GGML_TYPE_F32,1,32,0,10720);
    tensor(f,"blk.0.layer_output_scale.weight",COLI_GGML_TYPE_F32,1,1,0,10848);
    p=ftell(f);if(p<0){fclose(f);return -1;} while(p%32){w8(f,0);++p;}
    for(i=0;i<32*32;++i)wf(f,0.0F);
    q4_identity_rows(f,64,32);q4_identity_rows(f,32,32);q4_identity_rows(f,32,32);
    for(i=0;i<32*3;++i)wf(f,1.0F);
    q4_identity_rows(f,32,64);
    q4_identity_rows(f,32,32);q4_identity_rows(f,32,32);q4_identity_rows(f,32,32);
    for(i=0;i<32;++i)wf(f,1.0F);
    for(i=0;i<64;++i)wf(f,0.0F);
    wf(f,1.0F);wf(f,1.0F);
    for(i=0;i<6;++i)wf(f,0.0F);
    for(i=0;i<32*6;++i)wf(f,1.0F);
    wf(f,1.0F);
    return fclose(f)==0?0:-1;
}

typedef struct { int prepared, released; } fake_experts;

static int fake_prepare(void *context, uint32_t layer,
                        const uint32_t *ids, uint32_t count) {
    fake_experts *fake=(fake_experts *)context;
    if(layer!=0||count!=1||!ids||ids[0]!=0)return -1;
    ++fake->prepared;return 0;
}
static int fake_run(void *context, uint32_t layer, const uint32_t *ids,
                    const float *weights, uint32_t count,
                    const float *input, float *output) {
    uint32_t i;
    (void)context;(void)input;
    if(layer!=0||count!=1||!ids||ids[0]!=0||!weights||
       fabsf(weights[0]-1.0F)>1e-6F||!output)return -1;
    for(i=0;i<32;++i)output[i]=0.0F;
    return 0;
}
static void fake_release(void *context, uint32_t layer) {
    fake_experts *fake=(fake_experts *)context;
    if(layer==0)++fake->released;
}

static int test_decoder_layer(const coli_gemma4_gguf *gguf,
                              const float *input) {
    coli_gemma4_decoder_layer decoder;
    coli_gemma4_kv_cache actual_cache, reference_cache;
    fake_experts fake={0,0};
    coli_expert_backend backend={
        .ctx=&fake,
        .prepare_layer=fake_prepare,
        .run_experts=fake_run,
        .release_layer=fake_release
    };
    float actual[32],attention_output[32],after_attention[32];
    float dense_input[32],dense_output[32],dense_normalized[32];
    float expected[32];
    uint32_t i;
    int status=-1;
    memset(&decoder,0,sizeof(decoder));
    memset(&actual_cache,0,sizeof(actual_cache));
    memset(&reference_cache,0,sizeof(reference_cache));
    if(coli_gemma4_decoder_layer_open(&decoder,gguf,0)!=0){
        fprintf(stderr,"decoder open: %s\n",
                coli_gemma4_decoder_layer_last_error(&decoder));goto cleanup;
    }
    if(decoder.dense_mlp.intermediate_width!=32||
       coli_gemma4_kv_cache_init(&actual_cache,&decoder.attention,2)!=0||
       coli_gemma4_kv_cache_init(&reference_cache,&decoder.attention,2)!=0||
       coli_gemma4_decoder_layer_step(&decoder,&actual_cache,&backend,0,
                                      input,actual)!=0||
       coli_gemma4_attention_step(&decoder.attention,&reference_cache,0,input,
                                  attention_output)!=0||
       coli_gemma4_rmsnorm(attention_output,decoder.post_attention_norm,32,
                           gguf->rms_epsilon,after_attention)!=0)
        goto cleanup;
    for(i=0;i<32;++i)after_attention[i]+=input[i];
    if(coli_gemma4_rmsnorm(after_attention,decoder.ffn_norm,32,
                           gguf->rms_epsilon,dense_input)!=0||
       coli_gemma4_dense_mlp_run(&decoder.dense_mlp,dense_input,dense_output)!=0||
       coli_gemma4_rmsnorm(dense_output,decoder.post_ffw_norm_1,32,
                           gguf->rms_epsilon,dense_normalized)!=0||
       coli_gemma4_rmsnorm(dense_normalized,decoder.post_ffw_norm,32,
                           gguf->rms_epsilon,expected)!=0)
        goto cleanup;
    for(i=0;i<32;++i){
        expected[i]=(after_attention[i]+expected[i])*decoder.layer_output_scale;
        if(fabsf(actual[i]-expected[i])>1e-5F){
            fprintf(stderr,"decoder ordering mismatch at %u\n",i);goto cleanup;
        }
    }
    if(fake.prepared!=1||fake.released!=1){
        fprintf(stderr,"expert backend lifecycle mismatch\n");goto cleanup;
    }
    status=0;
cleanup:
    coli_gemma4_kv_cache_close(&actual_cache);
    coli_gemma4_kv_cache_close(&reference_cache);
    coli_gemma4_decoder_layer_close(&decoder);
    return status;
}

int main(void) {
    coli_gemma4_gguf gguf;
    coli_gemma4_attention attention;
    coli_gemma4_kv_cache cache;
    float input[32],q[64],k[32],v[32],output[32];
    float sequence_input[64],sequence_output[64],sequence_expected[32];
    float q0[64],q1[64],k0[32],k1[32],v0[32],v1[32],context[64];
    float ss=0.0F,inv1,ss2=0.0F,inv2;
    float before_q0, before_q16, expected_q0, expected_q16;
    const float *cached_key, *cached_value;
    uint32_t i;
    int status=1;
    remove(FIXTURE);
    for(i=0;i<32;++i){input[i]=(float)(i+1);ss+=input[i]*input[i];}
    if(make_fixture()!=0){fprintf(stderr,"fixture write failed\n");return 1;}
    if(coli_gemma4_gguf_open(&gguf,FIXTURE)!=0){
        fprintf(stderr,"GGUF open: %s\n",coli_gemma4_gguf_last_error(&gguf));goto cleanup;
    }
    if(coli_gemma4_attention_open(&attention,&gguf,0)!=0){
        fprintf(stderr,"attention open: %s\n",coli_gemma4_attention_last_error(&attention));goto close_gguf;
    }
    if(attention.query_heads!=2||attention.kv_heads!=1||attention.head_dim!=32||
       !attention.sliding||attention.key_equals_value||
       coli_gemma4_attention_project(&attention,input,q,k,v)!=0){
        fprintf(stderr,"attention geometry or projection failed\n");goto close_attention;
    }
    inv1=1.0F/sqrtf(ss/32.0F+1e-6F);
    for(i=0;i<32;++i){float x=input[i]*inv1;ss2+=x*x;}
    inv2=1.0F/sqrtf(ss2/32.0F+1e-6F);
    for(i=0;i<32;++i){
        float expected=input[i]*inv1*inv2;
        if(fabsf(q[i]-expected)>1e-5F||fabsf(q[32+i]-expected)>1e-5F||
           fabsf(k[i]-expected)>1e-5F||fabsf(v[i]-expected)>1e-5F){
            fprintf(stderr,"attention value mismatch at %u\n",i);goto close_attention;
        }
    }
    before_q0=q[0];before_q16=q[16];
    expected_q0=before_q0*cosf(1.0F)-before_q16*sinf(1.0F);
    expected_q16=before_q16*cosf(1.0F)+before_q0*sinf(1.0F);
    if(coli_gemma4_attention_apply_rope(&attention,1,q,k)!=0||
       fabsf(q[0]-expected_q0)>1e-6F||fabsf(q[16]-expected_q16)>1e-6F){
        fprintf(stderr,"RoPE mismatch\n");goto close_attention;
    }
    memset(&cache,0,sizeof(cache));
    if(coli_gemma4_kv_cache_init(&cache,&attention,20)!=0||cache.capacity!=16||
       coli_gemma4_kv_cache_store(&cache,0,k,v)!=0||
       coli_gemma4_kv_cache_find(&cache,0,&cached_key,&cached_value)!=0||
       cached_key[0]!=k[0]||cached_value[0]!=v[0]||
       coli_gemma4_kv_cache_store(&cache,16,k,v)!=0||
       coli_gemma4_kv_cache_find(&cache,0,&cached_key,&cached_value)==0||
       coli_gemma4_kv_cache_find(&cache,16,&cached_key,&cached_value)!=0){
        fprintf(stderr,"sliding KV cache mismatch\n");
        coli_gemma4_kv_cache_close(&cache);goto close_attention;
    }
    coli_gemma4_kv_cache_close(&cache);
    memset(&cache,0,sizeof(cache));
    if(coli_gemma4_kv_cache_init(&cache,&attention,2)!=0||
       coli_gemma4_attention_step(&attention,&cache,0,input,output)!=0){
        fprintf(stderr,"causal attention step failed\n");
        coli_gemma4_kv_cache_close(&cache);goto close_attention;
    }
    for(i=0;i<32;++i){
        float expected=input[i]*inv1*inv2;
        if(fabsf(output[i]-expected)>1e-5F){
            fprintf(stderr,"causal attention output mismatch at %u\n",i);
            coli_gemma4_kv_cache_close(&cache);goto close_attention;
        }
    }
    coli_gemma4_kv_cache_close(&cache);
    memcpy(sequence_input,input,sizeof(input));
    for(i=0;i<32;++i)sequence_input[32+i]=(float)(32-i);
    memset(context,0,sizeof(context));
    if(coli_gemma4_attention_project(&attention,sequence_input,q0,k0,v0)!=0||
       coli_gemma4_attention_project(&attention,sequence_input+32,q1,k1,v1)!=0||
       coli_gemma4_attention_apply_rope(&attention,0,q0,k0)!=0||
       coli_gemma4_attention_apply_rope(&attention,1,q1,k1)!=0)
        goto close_attention;
    for(i=0;i<2;++i){
        uint32_t d;
        float score0=0.0F,score1=0.0F,maximum,e0,e1,denominator;
        const float *qh=q0+(size_t)i*32;
        float *ch=context+(size_t)i*32;
        for(d=0;d<32;++d){score0+=qh[d]*k0[d];score1+=qh[d]*k1[d];}
        maximum=score0>score1?score0:score1;
        e0=expf(score0-maximum);e1=expf(score1-maximum);denominator=e0+e1;
        for(d=0;d<32;++d)ch[d]=(e0*v0[d]+e1*v1[d])/denominator;
    }
    if(coli_gemma4_matrix_matvec(&attention.output,context,sequence_expected)!=0||
       coli_gemma4_kv_cache_init(&cache,&attention,2)!=0||
       coli_gemma4_attention_noncausal(&attention,&cache,0,sequence_input,2,
                                       sequence_output)!=0){
        fprintf(stderr,"non-causal image attention failed\n");
        coli_gemma4_kv_cache_close(&cache);goto close_attention;
    }
    for(i=0;i<32;++i)if(fabsf(sequence_output[i]-sequence_expected[i])>1e-5F){
        fprintf(stderr,"non-causal image attention mismatch at %u\n",i);
        coli_gemma4_kv_cache_close(&cache);goto close_attention;
    }
    coli_gemma4_kv_cache_close(&cache);
    if(test_decoder_layer(&gguf,input)!=0)goto close_attention;
    puts("Gemma attention and decoder-layer ordering passed");status=0;
close_attention: coli_gemma4_attention_close(&attention);
close_gguf: coli_gemma4_gguf_close(&gguf);
cleanup: remove(FIXTURE);return status;
}
