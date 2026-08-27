/* kv_alloc must survive re-allocation on the same KVState: every free path is
 * guarded by if(k->Lc) precisely so callers (context resize, slot re-init) can
 * call it again. A stale duplicate free block frees every Lc[i]/Rc[i] and both
 * arrays twice on the second call -> allocator abort. No model file needed:
 * the CPU path of kv_alloc only reads c->n_layers/kv_lora/qk_rope. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

int main(void){
    static Model m;
    m.c.n_layers=2; m.c.kv_lora=8; m.c.qk_rope=4;
    m.kv=calloc(1,sizeof(KVState));
    kv_alloc(&m,16);
    for(int i=0;i<m.c.n_layers+1;i++){ m.Lc[i][0]=1.0f; m.Rc[i][0]=1.0f; }
    kv_alloc(&m,32);                       /* the re-allocation path under test */
    for(int i=0;i<m.c.n_layers+1;i++){
        m.Lc[i][(int64_t)32*m.c.kv_lora-1]=2.0f;
        m.Rc[i][(int64_t)32*m.c.qk_rope-1]=2.0f;
    }
    /* KV8: the f32 -> fp8 transition frees the f32 rows and allocates the byte
     * cache + per-row scales; a second KV8 kv_alloc re-runs the fp8 free path;
     * switching back must survive too (the KV8 arrays are freed and nulled). */
    g_kv8=1;
    kv_alloc(&m,16);
    for(int i=0;i<m.c.n_layers+1;i++){
        if(m.Lc[i]||m.Rc[i]){ fprintf(stderr,"KV8: f32 rows must stay NULL\n"); return 1; }
        m.Lc8[i][0]=0x7e; m.Rc8[i][0]=0x7e; m.Lsc[i][0]=1.f; m.Rsc[i][0]=1.f;
    }
    kv_alloc(&m,64);                       /* fp8 re-allocation */
    for(int i=0;i<m.c.n_layers+1;i++){
        m.Lc8[i][(int64_t)64*m.c.kv_lora-1]=1;
        m.Rc8[i][(int64_t)64*m.c.qk_rope-1]=1;
        m.Lsc[i][63]=2.f; m.Rsc[i][63]=2.f;
    }
    g_kv8=0;
    kv_alloc(&m,16);                       /* fp8 -> f32 transition */
    if(m.kv->Lc8||m.kv->Rc8||m.kv->Lsc||m.kv->Rsc){
        fprintf(stderr,"KV8 arrays must be freed and nulled on the way back\n"); return 1; }
    for(int i=0;i<m.c.n_layers+1;i++){ m.Lc[i][0]=3.0f; m.Rc[i][0]=3.0f; }
    printf("OK kv_alloc re-allocation (f32 + KV8 fp8)\n");
    return 0;
}
