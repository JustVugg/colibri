/* Qwen3.8 hybrid prompt-prefix state: exact identity, extension, and reset.
 * The fabricated model exercises the same snapshot/restore helpers used by
 * SERVE without loading a checkpoint or depending on a tokenizer. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#define QWEN38_TEST_SERVE
#include "../qwen38.c"

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x);return 1; \
} } while(0)

static void fill_state(Model *m,float value){
    for(int i=0;i<m->c.layers;i++)if(!m->c.is_attn[i]){
        for(size_t j=0;j<g_q38_prefix.rec_cells;j++)m->DN_rec[i][j]=value;
        for(size_t j=0;j<g_q38_prefix.conv_cells;j++)m->DN_conv[i][j]=value+1.f;
    }
    for(size_t j=0;j<g_q38_prefix.ple_cells;j++)m->PLE_conv_state[j]=value+2.f;
    m->ple_history_len=2;m->ple_history[0]=41;m->ple_history[1]=42;
}

static int check_state(const Model *m,float value){
    for(int i=0;i<m->c.layers;i++)if(!m->c.is_attn[i]){
        for(size_t j=0;j<g_q38_prefix.rec_cells;j++)if(m->DN_rec[i][j]!=value)return 0;
        for(size_t j=0;j<g_q38_prefix.conv_cells;j++)if(m->DN_conv[i][j]!=value+1.f)return 0;
    }
    for(size_t j=0;j<g_q38_prefix.ple_cells;j++)if(m->PLE_conv_state[j]!=value+2.f)return 0;
    return m->ple_history_len==2&&m->ple_history[0]==41&&m->ple_history[1]==42;
}

static void free_fake(Model *m){
    q38_prefix_cache_release(m);
    for(int i=0;i<m->c.layers;i++){
        free(m->DN_rec[i]);free(m->DN_conv[i]);
        free(m->K[i]);free(m->V[i]);free(m->IK[i]);
    }
    free(m->DN_rec);free(m->DN_conv);free(m->K);free(m->V);free(m->IK);
    free(m->PLE_conv_state);free(m->ple_history);free(m->c.is_attn);
}

int main(void){
    Model m={0};
    m.c.layers=2;m.c.vocab=8;m.c.dn_vheads=1;m.c.dn_kdim=2;m.c.dn_vdim=2;
    m.c.dn_conv_dim=2;m.c.dn_convk=3;m.c.ple_layer=1;m.c.hc_width=2;
    m.c.ple_convk=3;m.c.ngram_size=1;m.c.is_attn=calloc(2,1);CHECK(m.c.is_attn);
    m.c.is_attn[1]=1;
    m.DN_rec=calloc(2,sizeof(float*));m.DN_conv=calloc(2,sizeof(float*));
    m.K=calloc(2,sizeof(float*));m.V=calloc(2,sizeof(float*));m.IK=calloc(2,sizeof(float*));
    CHECK(m.DN_rec&&m.DN_conv&&m.K&&m.V&&m.IK);
    m.DN_rec[0]=calloc(4,sizeof(float));m.DN_conv[0]=calloc(4,sizeof(float));
    m.PLE_conv_state=calloc(4,sizeof(float));m.ple_history=calloc(2,sizeof(int64_t));
    CHECK(m.DN_rec[0]&&m.DN_conv[0]&&m.PLE_conv_state&&m.ple_history);
    CHECK(q38_prefix_cache_layout(&m));

    /* Serve allocates its planned QSA ceiling once.  Reuse must preserve the
     * position-indexed prefix without a second allocation or growth peak. */
    m.c.kv_heads=1;m.c.head_dim=2;m.c.idx_dim=2;
    CHECK(q38_serve_ensure_kv(&m,6));CHECK(m.kv_cap==6);m.kv_len=3;
    float *initial_k=m.K[1],*initial_v=m.V[1],*initial_ik=m.IK[1];
    for(int i=0;i<6;i++){m.K[1][i]=(float)i;m.V[1][i]=(float)(10+i);m.IK[1][i]=(float)(20+i);}

    const int first[]={1,2,3};float first_logits[8];
    for(int i=0;i<8;i++)first_logits[i]=(float)(100+i);
    fill_state(&m,7.f);CHECK(q38_prefix_cache_save(&m,first,3,first_logits));
    fill_state(&m,99.f);m.kv_len=3;
    const int extension[]={1,2,3,4};
    CHECK(q38_prefix_restore(&m,extension,4)==3);CHECK(check_state(&m,7.f));
    CHECK(q38_serve_ensure_kv(&m,6));CHECK(m.kv_cap==6&&m.kv_len==3);
    CHECK(m.K[1]==initial_k&&m.V[1]==initial_v&&m.IK[1]==initial_ik);
    CHECK(!q38_serve_ensure_kv(&m,7));
    for(int i=0;i<6;i++){CHECK(m.K[1][i]==(float)i);CHECK(m.V[1][i]==(float)(10+i));CHECK(m.IK[1][i]==(float)(20+i));}

    /* An exact prompt restores the recurrent state and its saved logits. */
    fill_state(&m,55.f);CHECK(q38_prefix_restore(&m,first,3)==3);CHECK(check_state(&m,7.f));
    fill_state(&m,56.f);CHECK(q38_prefix_restore(&m,extension,4)==3);CHECK(check_state(&m,7.f));
    CHECK(q38_prefix_cache_save(&m,extension,4,first_logits));
    fill_state(&m,88.f);CHECK(q38_prefix_restore(&m,extension,4)==4);CHECK(check_state(&m,7.f));
    const float *cached=q38_prefix_cached_logits(&m);CHECK(cached&&cached[0]==100.f&&cached[7]==107.f);

    /* A mismatch is never restored; the serve miss path invalidates then
     * resets the live recurrent state before feeding the new prompt. */
    const int shorter[]={1,2,3};CHECK(q38_prefix_restore(&m,shorter,3)==0);
    const int mismatch[]={1,2,4};CHECK(q38_prefix_restore(&m,mismatch,3)==0);
    q38_prefix_cache_invalidate();reset_recurrent(&m);m.kv_len=0;
    CHECK(!g_q38_prefix.valid&&m.kv_len==0&&m.ple_history_len==0);
    for(size_t i=0;i<g_q38_prefix.rec_cells;i++)CHECK(m.DN_rec[0][i]==0.f);
    for(size_t i=0;i<g_q38_prefix.conv_cells;i++)CHECK(m.DN_conv[0][i]==0.f);
    for(size_t i=0;i<g_q38_prefix.ple_cells;i++)CHECK(m.PLE_conv_state[i]==0.f);
    free_fake(&m);
    puts("qwen38 prefix: single QSA allocation, exact/extension reuse, reset: ok");
    return 0;
}
