/* Exercise production OLMoE adapter teardown with resident synthetic weights.
 * No checkpoint/tokenizer is required; ASan/LSan checks repeated ownership cycles. */
#define OLMOE_NO_MAIN 1
#define COLI_SEGMENT_ADAPTER 1
#include "../olmoe.c"
#include <assert.h>

static OlmoeSegmentEngine *fixture_engine(int begin) {
    OlmoeSegmentEngine *engine = calloc(1, sizeof(*engine));
    assert(engine);
    engine->layer_begin=begin; engine->layer_end=begin+2; engine->context_tokens=8;
    assert(!pthread_mutex_init(&engine->run_lock, NULL));
    Model *m=&engine->model;
    m->c.n_layers=4; m->c.n_experts=2; m->c.hidden=8; m->c.inter=4;
    m->c.n_heads=2; m->c.head_dim=4;
    m->L=calloc(4,sizeof(*m->L));
    m->cache=calloc(4,sizeof(*m->cache));
    assert(m->L && m->cache);
    for (int l=begin; l<begin+2; l++) {
        LCache *c=&m->cache[l];
        c->n=c->cap=2;
        c->slots=calloc(2,sizeof(*c->slots));
        c->slot_by_expert=malloc(2*sizeof(int));
        assert(c->slots && c->slot_by_expert);
        for (int e=0; e<2; e++) {
            Slot *s=&c->slots[e];
            s->eid=e; c->slot_by_expert[e]=e;
            s->g=calloc(96,1); s->gs=calloc(16,sizeof(float));
            assert(s->g && s->gs);
            s->u=s->g+32; s->d=s->u+32;
            s->us=s->gs+4; s->ds=s->us+4;
        }
    }
    return engine;
}

int main(void) {
    for (int cycle=0; cycle<32; cycle++) {
        int begin=cycle%2;
        OlmoeSegmentEngine *engine=fixture_engine(begin);
        ColiSegmentSessionOptions options={.struct_size=sizeof(options), .context_tokens=8};
        void *a=NULL, *b=NULL;
        char error[128];
        assert(!olmoe_segment_session_create(engine,&a,&options,error,sizeof(error)));
        assert(!olmoe_segment_session_create(engine,&b,&options,error,sizeof(error)));
        OlmoeSegmentSession *first=a, *second=b;
        assert(first->K[begin] != second->K[begin] && first->V[begin+1] != second->V[begin+1]);
        first->K[begin][0]=3;
        assert(second->K[begin][0]==0);
        ColiExpertStore store=olmoe_expert_store(&engine->model);
        for (int l=begin; cycle%4 && l<begin+2; l++) {
            ColiExpertView view;
            assert(!coli_expert_lookup(&store,(ColiExpertKey){l,1},&view));
            assert(engine->model.ehit[l][1]==1);
            assert(engine->model.cache[l].slots[1].busy==1);
            coli_expert_release(&store,&view);
            assert(engine->model.cache[l].slots[1].busy==0);
        }
        olmoe_segment_session_destroy(a);
        assert(second->K[begin][0]==0);
        olmoe_segment_session_destroy(b);
        olmoe_segment_engine_destroy(engine);
    }
    puts("olmoe Segment lifecycle: 32 session/store/engine cycles passed");
    return 0;
}
