/* Model-free exactness gate for Qwen's dense-int8 prefill kernel.  The old
 * path calls matmul_q once per prompt row; the new path shares every int8
 * weight decode between two rows.  Compare raw float bytes across even/odd
 * row counts, vector tails, and both sides of the OpenMP threshold. */
#include <stddef.h>
#include <stdlib.h>
static void *qwen_test_malloc(size_t n);
static void qwen_test_free(void *p);
#define malloc qwen_test_malloc
#define free qwen_test_free
#define COLI_QWEN_BATCH_TEST 1
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#undef malloc
#undef free

#include <sys/stat.h>

static void *recycle_addr;
static size_t recycle_size;
static int recycle_active;
static void *qwen_test_malloc(size_t n) {
    if(recycle_active&&n==recycle_size&&recycle_addr){
        void *p=recycle_addr;recycle_addr=NULL;return p;
    }
    return malloc(n);
}
static void qwen_test_free(void *p) {
    if(recycle_active&&p&&p==recycle_addr)return;
    if(recycle_active&&p&&recycle_addr==NULL){
        recycle_addr=p;return;
    }
    free(p);
}

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

static float input_value(int64_t i, int salt) {
    int v = (int)((i * 37 + salt * 19) % 251);
    return (float)(v - 125) / (float)(97 + salt);
}

static void one_shape(int S, int I, int O) {
    float *x = falloc((int64_t)S*I);
    int8_t *q = malloc((size_t)O*I);
    float *sc = falloc(O);
    float *ref = falloc((int64_t)S*O), *got = falloc((int64_t)S*O);
    CHECK(q != NULL, "weight allocation failed");
    if (!q) exit(2);
    for (int64_t i=0;i<(int64_t)S*I;i++) x[i]=input_value(i,1);
    for (int64_t i=0;i<(int64_t)O*I;i++) q[i]=(int8_t)((i*29+7)%255-127);
    for (int o=0;o<O;o++) sc[o]=0.001f*(float)(1+(o*11)%31);

    for (int s=0;s<S;s++)
        matmul_q(ref+(int64_t)s*O,x+(int64_t)s*I,q,sc,I,O);
    matmul_q_batch(got,x,q,sc,S,I,O);
    if (memcmp(ref,got,(size_t)S*O*sizeof(float))) {
        int shown=0, different=0; float worst=0.f;
        for (int64_t i=0;i<(int64_t)S*O;i++) if (ref[i]!=got[i]) {
            float d=fabsf(ref[i]-got[i]); if(d>worst)worst=d;
            if(shown++<4)fprintf(stderr,"diff[%lld] ref=%a batch=%a delta=%g\n",
                                 (long long)i,ref[i],got[i],d);
            different++;
        }
        CHECK(0,"S=%d I=%d O=%d differs at %d values, worst=%g",
              S,I,O,different,worst);
    }
    printf("qwen dense batch exact: S=%d I=%d O=%d\n",S,I,O);
    free(x);free(q);free(sc);free(ref);free(got);
}

static void env_set(const char *name,const char *value) {
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static void env_unset(const char *name) {
#ifdef _WIN32
    _putenv_s(name,"");
#else
    unsetenv(name);
#endif
}

static void clear_dense_weight(DenseWeight *w) {
    free(w->f32);free(w->q);free(w->sc);memset(w,0,sizeof(*w));
}

static void dense_quant_convention_case(void) {
    enum { I=5,O=3 };
    DenseWeight w={0};
    static const float src[O*I]={
        -2.f,-1.f,0.f,1.f,2.f,
        0.f,0.f,0.f,0.f,0.f,
        -.5f,-.25f,.125f,.25f,.5f
    };
    static const int8_t want_q[O*I]={
        -127,-64,0,64,127,
        0,0,0,0,0,
        -127,-64,32,64,127
    };
    env_set("COLI_KEEP_F32","1");
    w.f32=falloc(O*I);memcpy(w.f32,src,sizeof(src));
    CHECK(dense_weight_quantize(&w,I,O),"quantization convention tensor failed");
    CHECK(!memcmp(w.q,want_q,sizeof(want_q)),"dense per-row signed int8 convention changed");
    CHECK(w.sc[0]==2.f/127.f&&w.sc[1]==1.f&&w.sc[2]==.5f/127.f,
          "dense per-row f32 scale convention changed");
    env_unset("COLI_KEEP_F32");clear_dense_weight(&w);
    puts("qwen dense quantization: signed int8 + one exact f32 scale per row");
}

static void qwen_gs64_loader_case(void) {
    enum { D=64,I=64,N=3*D*I,PACKED=N/2,NS=3*D };
    static int8_t unpacked[N];static uint8_t packed[PACKED];static float scales[NS];
    for(int i=0;i<N;i++)unpacked[i]=(int8_t)((i*11+3)%16-8);
    for(int i=0;i<PACKED;i++)
        packed[i]=(uint8_t)(((uint8_t)unpacked[2*i]&15)|(((uint8_t)unpacked[2*i+1]&15)<<4));
    for(int i=0;i<NS;i++)scales[i]=(float)(i+1)/1024.f;

    const char *dir="tests/tmp_qwen36_gs64_snap";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir,0755);
#endif
    char path[256];snprintf(path,sizeof(path),"%s/model.safetensors",dir);
    char hdr[1024];int nsbytes=(int)sizeof(scales);
    int hl=snprintf(hdr,sizeof(hdr),
        "{\"model.layers.0.mlp.experts.0.merged_weight\":{\"dtype\":\"U8\",\"shape\":[%d],\"data_offsets\":[0,%d]},"
        "\"model.layers.0.mlp.experts.0.qs\":{\"dtype\":\"F32\",\"shape\":[%d],\"data_offsets\":[%d,%d]}}",
        PACKED,PACKED,NS,PACKED,PACKED+nsbytes);
    FILE *f=fopen(path,"wb");
    CHECK(f!=NULL,"cannot create Qwen gs64 safetensors fixture");
    if(!f)return;
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f);fwrite(hdr,1,(size_t)hl,f);
    fwrite(packed,1,sizeof(packed),f);fwrite(scales,1,sizeof(scales),f);fclose(f);

    Model m;memset(&m,0,sizeof(m));m.c.hidden=D;m.c.inter=I;m.c.expert_gs=64;
    int active_of[1]={0};m.active_of=active_of;st_init(&m.S,dir);
    Slot slot;memset(&slot,0,sizeof(slot));slot_ensure_allocated(&m,&slot);
    load_expert_merged(&m,0,0,&slot);
    CHECK(slot.is_int4==1,"packed-int4 Qwen expert was not detected by stored size");
    CHECK(!memcmp(slot.g,unpacked,sizeof(unpacked)),"Qwen signed-nibble unpack convention changed");
    CHECK(!memcmp(slot.gs,scales,sizeof(scales)),"Qwen gs64 f32 scales changed during load");
    CHECK(slot.g4==NULL&&slot.u4==NULL&&slot.d4==NULL,
          "dependency-free CPU load retained an unnecessary packed expert copy");

    float x[D],got[I],want[I];for(int i=0;i<D;i++)x[i]=(float)(i-31)/64.f;
    g_expert_gs=64;matmul_qe(got,x,slot.g,slot.gs,D,I);
    for(int o=0;o<I;o++){
        double acc=0.;for(int i=0;i<D;i++)acc+=(double)x[i]*(double)unpacked[o*D+i]*(double)scales[o];
        want[o]=(float)acc;
    }
    for(int o=0;o<I;o++)
        CHECK(fabsf(got[o]-want[o])<=1e-6f*(1.f+fabsf(want[o])),
              "Qwen gs64 routed forward mismatch at row %d: got=%g want=%g",o,got[o],want[o]);
    g_expert_gs=0;

    free(slot.g);free(slot.gs);
    for(int i=0;i<m.S.nfd;i++)close(m.S.fds[i]);
    unlink(path);rmdir(dir);
    puts("qwen routed expert: packed signed int4 + f32 gs64 scales forward exact");
}

static void incremental_release_case(void) {
    enum { I=16,O=4 };
    DenseWeight first={0},second={0},kept={0};
    float x[I],got[O],want[O];
    env_unset("COLI_KEEP_F32");

    first.f32=falloc((int64_t)I*O);
    for(int i=0;i<I*O;i++)first.f32[i]=input_value(i,8);
    void *first_addr=first.f32;
    recycle_addr=first.f32;recycle_size=sizeof(float)*I*O;recycle_active=1;
    CHECK(dense_weight_quantize(&first,I,O),"first tensor did not quantize");
    CHECK(first.f32==NULL,"first f32 tensor survived its quantization step");
    CHECK(first.q!=NULL&&first.sc!=NULL,"first tensor has no int8 representation");

    /* Allocate the next source only after the first has been released. The
     * two DenseWeight objects must still dispatch to their own int8 data even
     * if malloc recycles the just-freed f32 address. */
    second.f32=falloc((int64_t)I*O);
    CHECK(second.f32==first_addr,
          "test allocator did not hand the exact freed source address to the next tensor");
    CHECK(recycle_addr==NULL,"test allocator did not force exact source-address reuse");
    for(int i=0;i<I*O;i++)second.f32[i]=input_value(i,9);
    recycle_active=0;
    CHECK(dense_weight_quantize(&second,I,O),"second tensor did not quantize");
    CHECK(second.f32==NULL,"second f32 tensor survived its quantization step");
    for(int i=0;i<I;i++)x[i]=input_value(i,10);
    matmul_d(got,x,&first,1,I,O);matmul_q(want,x,first.q,first.sc,I,O);
    CHECK(!memcmp(got,want,sizeof(got)),"first tensor lost its quantized dispatch");
    matmul_d(got,x,&second,1,I,O);matmul_q(want,x,second.q,second.sc,I,O);
    CHECK(!memcmp(got,want,sizeof(got)),"second tensor lost its quantized dispatch");

    env_set("COLI_KEEP_F32","1");
    kept.f32=falloc((int64_t)I*O);
    for(int i=0;i<I*O;i++)kept.f32[i]=input_value(i,11);
    CHECK(dense_weight_quantize(&kept,I,O),"debug tensor did not quantize");
    CHECK(kept.f32!=NULL,"COLI_KEEP_F32 did not retain the source tensor");
    env_unset("COLI_KEEP_F32");

    clear_dense_weight(&first);clear_dense_weight(&second);clear_dense_weight(&kept);
    puts("qwen dense load: each f32 source released before the next tensor");
}

static void shared_case(const char *format,int quantized) {
    enum { S=12,D=64,I=32 };
    Model m;memset(&m,0,sizeof(m));m.c.hidden=D;m.c.shared_inter=I;
    Layer l;memset(&l,0,sizeof(l));
    l.sh_g.f32=falloc((int64_t)I*D);l.sh_u.f32=falloc((int64_t)I*D);
    l.sh_d.f32=falloc((int64_t)D*I);l.sh_gate=falloc(D);
    for(int64_t i=0;i<(int64_t)I*D;i++){l.sh_g.f32[i]=input_value(i,2);l.sh_u.f32[i]=input_value(i,3);}
    for(int64_t i=0;i<(int64_t)D*I;i++)l.sh_d.f32[i]=input_value(i,4);
    for(int i=0;i<D;i++)l.sh_gate[i]=input_value(i,5);
    if(quantized){dense_weight_quantize(&l.sh_g,D,I);dense_weight_quantize(&l.sh_u,D,I);dense_weight_quantize(&l.sh_d,I,D);}
    float *x=falloc((int64_t)S*D),*seed=falloc((int64_t)S*D);
    float *ref=falloc((int64_t)S*D),*got=falloc((int64_t)S*D);
    float *g=falloc(I),*u=falloc(I),*hh=falloc(D);
    for(int64_t i=0;i<(int64_t)S*D;i++){x[i]=input_value(i,6);seed[i]=input_value(i,7);}

    memcpy(ref,seed,(size_t)S*D*sizeof(float));env_set("QWEN_SHARED_BATCH","0");
    env_set("QWEN_DENSE_BATCH","0");g_qwen_matmul_d_calls=0;
    qwen_shared_experts_cpu(&m,&l,x,S,ref,g,u,hh);
    CHECK(g_qwen_matmul_d_calls==(uint64_t)S*3,"%s scalar calls=%llu expected=%d",format,
          (unsigned long long)g_qwen_matmul_d_calls,S*3);

    memcpy(got,seed,(size_t)S*D*sizeof(float));env_unset("QWEN_SHARED_BATCH");
    env_unset("QWEN_DENSE_BATCH");g_qwen_matmul_d_calls=0;
    qwen_shared_experts_cpu(&m,&l,x,S,got,g,u,hh);
    CHECK(!memcmp(ref,got,(size_t)S*D*sizeof(float)),"%s shared batch is not scalar bit-exact",format);
    CHECK(g_qwen_matmul_d_calls==3,"%s batch calls=%llu expected=3",format,
          (unsigned long long)g_qwen_matmul_d_calls);

    memcpy(got,seed,(size_t)D*sizeof(float));g_qwen_matmul_d_calls=0;
    qwen_shared_experts_cpu(&m,&l,x,1,got,g,u,hh);
    CHECK(!memcmp(ref,got,(size_t)D*sizeof(float)),"%s decode row changed",format);
    CHECK(g_qwen_matmul_d_calls==3,"%s decode calls=%llu expected=3",format,
          (unsigned long long)g_qwen_matmul_d_calls);
    printf("qwen shared batch exact: format=%s S=%d calls=%d -> 3\n",format,S,S*3);

    clear_dense_weight(&l.sh_g);clear_dense_weight(&l.sh_u);clear_dense_weight(&l.sh_d);free(l.sh_gate);
    free(x);free(seed);free(ref);free(got);free(g);free(u);free(hh);
}

int main(void) {
    /* This gate owns the dense-int8 mode regardless of the caller's shell. */
    env_unset("COLI_DENSE_I8");
    incremental_release_case();
    dense_quant_convention_case();
    qwen_gs64_loader_case();
    one_shape(1, 17, 13);       /* decode-shaped fallback */
    one_shape(2, 32, 31);       /* one vector block, one row pair */
    one_shape(4, 64, 73);       /* even prompt, serial OpenMP clause */
    one_shape(5, 67, 259);      /* odd prompt + scalar tail + parallel clause */
    shared_case("f32",0);
    shared_case("int8",1);
    env_unset("QWEN_SHARED_BATCH");env_unset("QWEN_DENSE_BATCH");
    if(failures){fprintf(stderr,"qwen dense batch: %d failure(s)\n",failures);return 1;}
    puts("qwen dense batch: ok");return 0;
}
