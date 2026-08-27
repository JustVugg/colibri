/* Model-free exactness gate for Qwen's dense-int8 prefill kernel.  The old
 * path calls matmul_q once per prompt row; the new path shares every int8
 * weight decode between two rows.  Compare raw float bytes across even/odd
 * row counts, vector tails, and both sides of the OpenMP threshold. */
#define COLI_QWEN_BATCH_TEST 1
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

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

static void clear_qdw(void) {
    for(int i=0;i<g_qdw_n;i++){free(g_qdw[i].q);free(g_qdw[i].sc);}
    g_qdw_n=0;
}

static void shared_case(const char *format,int quantized) {
    enum { S=12,D=64,I=32 };
    Model m;memset(&m,0,sizeof(m));m.c.hidden=D;m.c.shared_inter=I;
    Layer l;memset(&l,0,sizeof(l));
    l.sh_g=falloc((int64_t)I*D);l.sh_u=falloc((int64_t)I*D);
    l.sh_d=falloc((int64_t)D*I);l.sh_gate=falloc(D);
    for(int64_t i=0;i<(int64_t)I*D;i++){l.sh_g[i]=input_value(i,2);l.sh_u[i]=input_value(i,3);}
    for(int64_t i=0;i<(int64_t)D*I;i++)l.sh_d[i]=input_value(i,4);
    for(int i=0;i<D;i++)l.sh_gate[i]=input_value(i,5);
    if(quantized){qdw_register(l.sh_g,D,I);qdw_register(l.sh_u,D,I);qdw_register(l.sh_d,I,D);}
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

    clear_qdw();free(l.sh_g);free(l.sh_u);free(l.sh_d);free(l.sh_gate);
    free(x);free(seed);free(ref);free(got);free(g);free(u);free(hh);
}

int main(void) {
    /* This gate owns the dense-int8 mode regardless of the caller's shell. */
    env_unset("COLI_DENSE_I8");
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
