/* Model-free oracle for Inkling shared-expert prefill batching.  It drives the
 * production helper in all resident formats (f32 oracle, bf16 checkpoint,
 * int4-g64 sidecar), compares against the old scalar path bit for bit, and
 * proves that the number of weight traversals is per chunk rather than per
 * prompt position. */
#define COLI_INKLING_SHARED_BATCH_TEST 1
#define main inkling_main_unused
#include "../inkling.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

/* compat.h maps setenv() to SetEnvironmentVariableA on Windows, but getenv()
 * reads the CRT environment copy.  Use _putenv_s there so the production
 * getenv("INK_SHARED_BATCH") sees the mode under test. */
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

static float value(int64_t i, int salt) {
    int v = (int)((i * 37 + salt * 17) % 101);
    return (float)(v - 50) / (float)(71 + salt);
}
static void fill(float *p, int64_t n, int salt) {
    for (int64_t i=0;i<n;i++) p[i]=value(i,salt);
}
static uint16_t bf16(float x) {
    union { float f; uint32_t u; } v={x}; return (uint16_t)(v.u>>16);
}
static void fill_h(uint16_t *p, int64_t n, int salt) {
    for (int64_t i=0;i<n;i++) p[i]=bf16(value(i,salt));
}
static void init_q4(Wt *w, int rows, int cols, int salt) {
    int rb=(cols+1)/2,ng=(cols+63)/64;
    w->qbits=4;w->gs=64;w->qn=(int64_t)rows*rb;
    w->q4=malloc((size_t)w->qn);w->qs=malloc((size_t)rows*ng*sizeof(float));
    CHECK(w->q4&&w->qs,"q4 allocation failed");if(!w->q4||!w->qs)exit(2);
    for(int64_t i=0;i<w->qn;i++){
        int lo=(int)((i*5+salt)%16),hi=(int)((i*11+salt*3)%16);
        w->q4[i]=(uint8_t)(lo|(hi<<4));
    }
    for(int64_t i=0;i<(int64_t)rows*ng;i++)w->qs[i]=0.01f*(float)(1+(i+salt)%7);
}
static void free_w(Wt *w){free(w->f);free(w->h);free(w->q4);free(w->qs);}

static void compare_paths(const char *format,Model *m,Layer *l,int S,
                          const float *x,const float *wgt,const float *seed){
    Cfg *c=&m->c;int D=c->hidden,I=c->moe_inter,ns=c->n_shared;
    size_t out_bytes=(size_t)S*D*sizeof(float);
    float *scalar=falloc((int64_t)S*D),*out=falloc((int64_t)S*D);
    float *g=falloc((int64_t)2*I),*u=g+I,*hh=falloc(D);
    memcpy(scalar,seed,out_bytes);env_set("INK_SHARED_BATCH","0");
    g_matmul_w_calls=0;shared_experts_cpu(m,l,x,S,scalar,wgt,g,u,hh);
    CHECK(g_matmul_w_calls==(uint64_t)S*ns*3,
          "%s scalar calls=%llu expected=%d",format,
          (unsigned long long)g_matmul_w_calls,S*ns*3);

    const char *modes[]={NULL,"3"};
    for(int z=0;z<2;z++){
        memcpy(out,seed,out_bytes);
        if(modes[z])env_set("INK_SHARED_BATCH",modes[z]);else env_unset("INK_SHARED_BATCH");
        g_matmul_w_calls=0;shared_experts_cpu(m,l,x,S,out,wgt,g,u,hh);
        int chunks=modes[z]?(S+2)/3:1,expected=chunks*ns*3;
        CHECK(!memcmp(out,scalar,out_bytes),"%s mode=%s is not scalar bit-exact",
              format,modes[z]?modes[z]:"default");
        CHECK(g_matmul_w_calls==(uint64_t)expected,
              "%s mode=%s calls=%llu expected=%d",format,modes[z]?modes[z]:"default",
              (unsigned long long)g_matmul_w_calls,expected);
        printf("inkling shared: format=%s S=%d mode=%s calls=%d -> %llu\n",
               format,S,modes[z]?modes[z]:"default",S*ns*3,
               (unsigned long long)g_matmul_w_calls);
    }

    /* Decode must remain the original one-row GEMV path. */
    memcpy(out,seed,(size_t)D*sizeof(float));env_unset("INK_SHARED_BATCH");
    g_matmul_w_calls=0;shared_experts_cpu(m,l,x,1,out,wgt,g,u,hh);
    CHECK(!memcmp(out,scalar,(size_t)D*sizeof(float)),"%s decode row changed",format);
    CHECK(g_matmul_w_calls==(uint64_t)ns*3,"%s decode used batch shape",format);
    free(g);free(hh);free(scalar);free(out);
}

static void run_format(const char *format,int kind,int S,int D,int I){
    enum{K=2,NS=2};Model m;memset(&m,0,sizeof(m));
    m.c.hidden=D;m.c.moe_inter=I;m.c.topk=K;m.c.n_shared=NS;
    Layer l;memset(&l,0,sizeof(l));int64_t gi=(int64_t)NS*I*D,di=(int64_t)NS*D*I;
    if(kind==0){
        l.sh_g.f=falloc(gi);l.sh_u.f=falloc(gi);l.sh_d.f=falloc(di);
        fill(l.sh_g.f,gi,1);fill(l.sh_u.f,gi,2);fill(l.sh_d.f,di,3);
    }else if(kind==1){
        l.sh_g.h=malloc((size_t)gi*2);l.sh_u.h=malloc((size_t)gi*2);l.sh_d.h=malloc((size_t)di*2);
        CHECK(l.sh_g.h&&l.sh_u.h&&l.sh_d.h,"bf16 allocation failed");
        if(!l.sh_g.h||!l.sh_u.h||!l.sh_d.h)exit(2);
        fill_h(l.sh_g.h,gi,1);fill_h(l.sh_u.h,gi,2);fill_h(l.sh_d.h,di,3);
    }else{
        init_q4(&l.sh_g,NS*I,D,1);init_q4(&l.sh_u,NS*I,D,2);init_q4(&l.sh_d,NS*D,I,3);
    }
    float *x=falloc((int64_t)S*D),*w=falloc((int64_t)S*(K+NS)),*seed=falloc((int64_t)S*D);
    fill(x,(int64_t)S*D,4);fill(w,(int64_t)S*(K+NS),5);fill(seed,(int64_t)S*D,6);
    compare_paths(format,&m,&l,S,x,w,seed);
    free_w(&l.sh_g);free_w(&l.sh_u);free_w(&l.sh_d);free(x);free(w);free(seed);
}

int main(void){
    run_format("f32",0,12,17,13);
    run_format("bf16",1,12,32,16);
    run_format("int4-g64",2,12,64,64);
    env_unset("INK_SHARED_BATCH");
    if(failures){fprintf(stderr,"inkling shared batch: %d failure(s)\n",failures);return 1;}
    puts("inkling shared batch: ok");return 0;
}
