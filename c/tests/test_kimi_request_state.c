/* Kimi serve request-state sequencing regression (#855).
 *
 * A first request has no reusable prefix. The broken path allocated Lc/Rc,
 * observed that miss, reset the model, and entered MLA prefill with both arrays
 * NULL. Exercise the production helper with tiny fabricated layers: first
 * request, growing reuse, and a divergent request that must reset then allocate. */
#define main kimi_k3_main_unused
#include "../kimi_k3.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if(!(cond)){ \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while(0)

static void init_model(Model *m){
    memset(m,0,sizeof(*m));
    m->c.n_layers=2; m->c.kv_lora=3; m->c.qk_rope=2;
    m->c.kda_heads=1; m->c.kda_hd=1; m->c.kda_proj=1; m->c.conv_k=1;
    m->L=calloc((size_t)m->c.n_layers,sizeof(Layer));
    m->L[0].kda=1;                                  /* recurrent row: no Lc/Rc */
    m->kstate=calloc((size_t)m->c.n_layers,sizeof(float*));
    m->cwq=calloc((size_t)m->c.n_layers,sizeof(float*));
    m->cwk=calloc((size_t)m->c.n_layers,sizeof(float*));
    m->cwv=calloc((size_t)m->c.n_layers,sizeof(float*));
    m->kstate[0]=calloc(1,sizeof(float));
    m->cwq[0]=calloc(1,sizeof(float)); m->cwk[0]=calloc(1,sizeof(float));
    m->cwv[0]=calloc(1,sizeof(float));
}

static void free_model(Model *m){
    model_state_reset(m); kv_prefix_free(&m->kvp);
    free(m->kstate[0]); free(m->cwq[0]); free(m->cwk[0]); free(m->cwv[0]);
    free(m->kstate); free(m->cwq); free(m->cwk); free(m->cwv); free(m->L);
}

int main(void){
    Model m; init_model(&m);
    const int first[]={1,2,3};
    int reuse=prepare_request_state(&m,first,3,8);
    CHECK(reuse==0,"first request reused %d tokens",reuse);
    CHECK(m.Lc&&m.Rc,"first request left cache pointer arrays NULL");
    CHECK(m.Lc[1]&&m.Rc[1],"first request left MLA cache rows NULL");
    CHECK(m.max_t==8&&m.kvp.cap==8,"first allocation max_t=%d prefix cap=%d",m.max_t,m.kvp.cap);

    kv_prefix_record(&m.kvp,first,0,3);
    float *old_l=m.Lc[1], *old_r=m.Rc[1];
    for(int i=0;i<9;i++) old_l[i]=(float)(i+1);
    for(int i=0;i<6;i++) old_r[i]=(float)(20+i);
    const int extended[]={1,2,3,4};
    reuse=prepare_request_state(&m,extended,4,12);
    CHECK(reuse==3,"extended request reused %d tokens, want 3",reuse);
    CHECK(m.Lc&&m.Rc&&m.Lc[1]&&m.Rc[1],"growing reuse lost MLA cache");
    CHECK(m.Lc[1]!=old_l&&m.Rc[1]!=old_r,"grow did not replace undersized storage");
    for(int i=0;i<9;i++) CHECK(m.Lc[1][i]==(float)(i+1),"Lc prefix changed at %d",i);
    for(int i=0;i<6;i++) CHECK(m.Rc[1][i]==(float)(20+i),"Rc prefix changed at %d",i);

    const int divergent[]={1,9,3,4};
    reuse=prepare_request_state(&m,divergent,4,10);
    CHECK(reuse==0,"divergent request reused %d tokens",reuse);
    CHECK(m.Lc&&m.Rc&&m.Lc[1]&&m.Rc[1],"divergent reset was not followed by allocation");
    CHECK(m.max_t==10&&m.kvp.cap==10&&m.kvp.len==0,
          "fresh state max_t=%d cap=%d len=%d",m.max_t,m.kvp.cap,m.kvp.len);

    kv_prefix_record(&m.kvp,divergent,0,4);
    m.kstate[0][0]=7.f; m.cwq[0][0]=8.f; m.cwk[0][0]=9.f; m.cwv[0][0]=10.f;
    k3_cancel_unpublished_state(&m);
    CHECK(!m.Lc&&!m.Rc&&m.max_t==0&&m.kvp.len==0,
          "cancelled prefill retained unpublished KV/prefix state");
    CHECK(m.kstate[0][0]==0.f&&m.cwq[0][0]==0.f&&
          m.cwk[0][0]==0.f&&m.cwv[0][0]==0.f,
          "cancelled prefill retained unpublished recurrent state");

    reuse=prepare_request_state(&m,first,3,8);
    CHECK(reuse==0&&m.Lc&&m.Rc&&m.Lc[1]&&m.Rc[1],
          "request after cancellation did not cold-rebuild cleanly");

    free_model(&m);
    if(failures){ fprintf(stderr,"kimi request state: %d failure(s)\n",failures); return 1; }
    puts("kimi request state: ok");
    return 0;
}
