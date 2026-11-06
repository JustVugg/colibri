/* Borrowed OLMoE store: metadata, leases, failure, and the existing CPU math. */
#define OLMOE_TESTING 1
#define main coli_olmoe_main_unused
#include "../olmoe.c"
#undef main
#include <assert.h>

enum { TD = 32, TI = 64 };

static void check_view(const ColiTensorView *v, const void *data, int rows, int cols,
                       int ggml, int type) {
    assert(v->data == data && v->rows == rows && v->columns == cols);
    assert(v->format == (ggml ? COLI_TENSOR_GGML : COLI_TENSOR_Q8_ROW));
    assert(v->data_bytes == (size_t)rows * (ggml ? (size_t)gq_row_bytes(type, cols) : (size_t)cols));
    if (ggml) {
        assert(v->block_columns == (uint32_t)type);
        assert(v->scales == NULL && v->scale_bytes == 0 && v->scale_format == COLI_SCALE_NONE);
    } else {
        assert(v->scales && v->scale_format == COLI_SCALE_F32);
        assert(v->scale_bytes == rows * sizeof(float));
    }
}

static void bench_store(Model *m, ColiExpertStore *store) {
    const int count = 200000;
    double raw_best=1e30, view_best=1e30;
    volatile uintptr_t witness=0;
    for (int trial=0; trial<5; trial++) {
        double start=now_s();
        for (int i=0; i<count; i++) {
            Slot *slot; expert_get(m,0,0,&slot);
            witness ^= (uintptr_t)slot->d;
            expert_put(slot);
        }
        double raw=now_s()-start;
        start=now_s();
        for (int i=0; i<count; i++) {
            ColiExpertView view;
            assert(!coli_expert_lookup(store,(ColiExpertKey){0,0},&view));
            witness ^= (uintptr_t)view.down.data;
            coli_expert_release(store,&view);
        }
        double shared=now_s()-start;
        if (raw<raw_best) raw_best=raw;
        if (shared<view_best) view_best=shared;
    }
    printf("resident lookup/release ns: slot=%.1f ExpertStore=%.1f (best of 5, %d iterations, witness=%llu)\n",
           raw_best*1e9/count, view_best*1e9/count, count, (unsigned long long)witness);
}

static void check_store(int ggml, int packed) {
    Model m = {0};
    Slot slot = {.eid=0};
    int index = 0;
    LCache cache = {.slots=&slot, .slot_by_expert=&index, .n=1, .cap=1};
    m.c.hidden=TD; m.c.inter=TI; m.c.n_layers=1; m.c.n_experts=1;
    m.cache=&cache; m.gx=ggml;
    int rows[] = {TI, TI, TD}, cols[] = {TD, TD, TI};
    int8_t **weights[] = {&slot.g, &slot.u, &slot.d};
    float **scales[] = {&slot.gs, &slot.us, &slot.ds};
    for (int k=0; k<3; k++) {
        int type = packed ? GQ_Q8_0_R16 : GQ_Q8_0;
        slot.gt[k] = type;
        size_t bytes = (size_t)rows[k] * (ggml ? (size_t)gq_row_bytes(type, cols[k]) : (size_t)cols[k]);
        *weights[k] = malloc(bytes);
        assert(*weights[k]);
        if (ggml) {
            size_t raw_bytes = (size_t)rows[k] * (size_t)gq_row_bytes(GQ_Q8_0, cols[k]);
            unsigned char *raw = calloc(1, raw_bytes);
            assert(raw);
            for (size_t b=0; b<raw_bytes; b+=34) {
                raw[b]=0; raw[b+1]=0x24; /* fp16 scale = 1/64 */
                for (int j=0; j<32; j++) raw[b+2+j]=(unsigned char)((int)((b+j+k*7)%17)-8);
            }
            if (packed) gq_r16_pack(type, (uint8_t *)*weights[k], raw, rows[k], cols[k]);
            else memcpy(*weights[k], raw, bytes);
            free(raw);
        } else {
            *scales[k] = malloc(rows[k]*sizeof(float));
            assert(*scales[k]);
            for (int j=0; j<rows[k]; j++) (*scales[k])[j]=0.015625f;
            for (size_t j=0; j<bytes; j++) (*weights[k])[j]=(int8_t)((int)((j+k*7)%17)-8);
        }
    }
    ColiExpertStore store = olmoe_expert_store(&m);
    ColiExpertView v, second, empty = {0};
    assert(coli_expert_lookup(&store, (ColiExpertKey){0,0}, &v)==0);
    assert(coli_expert_lookup(&store, (ColiExpertKey){0,0}, &second)==0);
    assert(slot.busy==2 && m.hits==2 && !m.miss);
    check_view(&v.gate, slot.g, TI, TD, ggml, slot.gt[0]);
    check_view(&v.up, slot.u, TI, TD, ggml, slot.gt[1]);
    check_view(&v.down, slot.d, TD, TI, ggml, slot.gt[2]);
    coli_expert_release(&store, &second);
    assert(slot.busy==1 && !memcmp(&second, &empty, sizeof(empty)));
    coli_expert_release(&store, &second);
    assert(slot.busy==1);

    float x[TD], g[TI], u[TI], actual[TD], rg[TI], ru[TI], expected[TD];
    for (int i=0; i<TD; i++) x[i]=(float)(i-16)/32;
    moe_expert_row(&m, &v, x, g, u, actual);
    if (ggml) {
        ColiTensorView gate = {.format=COLI_TENSOR_GGML, .data=slot.g, .rows=TI, .columns=TD, .block_columns=(uint32_t)slot.gt[0]};
        ColiTensorView up = gate;
        up.data=slot.u;
        ColiTensorView down = {.format=COLI_TENSOR_GGML, .data=slot.d, .rows=TD, .columns=TI, .block_columns=(uint32_t)slot.gt[2]};
        assert(!coli_ggml_matmul2(rg, &gate, ru, &up, x, 1));
        for (int i=0; i<TI; i++) { float z=rg[i]; rg[i]=(z/(1.f+expf(-z)))*ru[i]; }
        assert(!coli_ggml_matmul(expected, &down, rg, 1));
    } else {
        matmul_q(rg, x, slot.g, slot.gs, TD, TI);
        matmul_q(ru, x, slot.u, slot.us, TD, TI);
        for (int i=0; i<TI; i++) { float z=rg[i]; rg[i]=(z/(1.f+expf(-z)))*ru[i]; }
        matmul_q(expected, rg, slot.d, slot.ds, TI, TD);
    }
    assert(!memcmp(actual, expected, sizeof(actual)));
    coli_expert_release(&store, &v);
    assert(!slot.busy && !memcmp(&v, &empty, sizeof(empty)));
    const ColiExpertKey invalid[] = {{-1,0},{1,0},{0,-1},{0,1}};
    for (size_t i=0; i<sizeof(invalid)/sizeof(*invalid); i++) {
        memset(&v, 0xa5, sizeof(v)); v.lease=NULL;
        assert(coli_expert_lookup(&store, invalid[i], &v)!=0);
        assert(!memcmp(&v, &empty, sizeof(v)) && !slot.busy && m.hits==2);
    }
    assert(coli_expert_lookup(&store, (ColiExpertKey){0,0}, NULL)!=0);
    if (!ggml && getenv("OLMOE_STORE_BENCH")) bench_store(&m, &store);
    for (int k=0; k<3; k++) { free(*weights[k]); free(*scales[k]); }
    if (m.ehit) { free(m.ehit[0]); free(m.ehit); }
}

int main(void) {
    check_store(0,0);
    check_store(1,0);
    check_store(1,1);
    puts("olmoe ExpertStore: row-int8, GGML and R16 views/leases/math passed");
    return 0;
}
