/* kv_alloc must survive re-allocation on the same KVState: every free path is
 * guarded by if(k->Lc) precisely so callers (context resize, slot re-init) can
 * call it again. A stale duplicate free block frees every Lc[i]/Rc[i] and both
 * arrays twice on the second call -> allocator abort. No model file needed:
 * the CPU path of kv_alloc only reads c->n_layers/kv_lora/qk_rope. */
#define main coli_glm_main_unused
#include "../glm.c"
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
    printf("OK kv_alloc re-allocation\n");
    return 0;
}
