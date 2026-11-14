/* qwen38's routed experts as MXFP4 (the experts-mxfp4/ sidecar written by
 * tools/convert_qwen38_experts_mxfp4.py from AMD Quark's checkpoint).
 *
 * Pinned with no model:
 *  - q38_weight_matmul on an MXFP4 weight (quant.h matmul_mxfp4) equals the
 *    dequantized reference (e2m1 times 2^(s-127), summed in double) to float
 *    rounding, on the release's two expert shapes, the tiny fixture's and
 *    widths with a short last group of 32, for one row and for a batch;
 *  - every row of a batch is the bits of that row alone, so decode and the
 *    prefill batch cannot drift apart;
 *  - the record of the release (H 2560, I 640) is 2,611,200 bytes, 2,613,248
 *    on its 4096-byte stride, with the scales of gate, down and up first.
 * With a converted snapshot on argv (make qwen38-tiny-mxfp4-check):
 *  - the sidecar attaches under Q38_EXPERT_LOWBIT=mxfp4, and every expert the
 *    single LRU path and the parallel batch path load holds exactly the bytes
 *    the sidecar stores under that expert's tensor names, read by name;
 *  - its values are this snapshot's experts to the bit: the fixture's FP8
 *    shards hold the MXFP4 values exactly, so the engine's own f32 expansion
 *    of them (Q38_NATIVE_FP8=0) equals every dequantized element, which pins
 *    which matrix is gate, up and down. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

static int fails;
static void ck(int ok,const char *what){ if(ok){printf("  ok   %s\n",what);return;} printf("  FAIL %s\n",what); fails++; }

static unsigned g_seed=20261114;
static unsigned rnd_u(void){ g_seed=g_seed*1103515245u+12345u; return g_seed>>8; }
static float rnd(void){ return (float)(rnd_u()&0xFFFF)/32768.f-1.f; }

/* element i of a row, without the engine's helpers */
static double value_at(const uint8_t *codes,const uint8_t *scales,int i){
    static const double e2m1[8]={0,.5,1,1.5,2,3,4,6};
    int n=(codes[i>>1]>>((i&1)*4))&15;
    return (n&8?-1:1)*ldexp(e2m1[n&7],(int)scales[i/32]-127);
}

/* y = W x in double, and the sum of |terms| the float error is measured against */
static void reference(double *y,double *mag,const uint8_t *codes,const uint8_t *scales,
                      const float *x,int S,int I,int O){
    int64_t rb=(I+1)/2,ng=(I+31)/32;
    for(int s=0;s<S;s++)for(int o=0;o<O;o++){
        double a=0,m=0;
        for(int i=0;i<I;i++){
            double t=value_at(codes+o*rb,scales+o*ng,i)*(double)x[(int64_t)s*I+i];
            a+=t;m+=fabs(t);
        }
        y[(int64_t)s*O+o]=a;mag[(int64_t)s*O+o]=m;
    }
}

static int within(const float *y,const double *ref,const double *mag,int n,int I){
    double tol=4.0*(sqrt((double)I)+8.0)*FLT_EPSILON;
    for(int k=0;k<n;k++)
        if(!(fabs((double)y[k]-ref[k])<=tol*mag[k]+1e-30))return 0;
    return 1;
}

static void kernel_case(int O,int I,int S){
    int64_t rb=(I+1)/2,ng=(I+31)/32;
    uint8_t *codes=malloc((size_t)(O*rb)),*scales=malloc((size_t)(O*ng));
    float *x=malloc((size_t)S*I*sizeof(float)),*y=malloc((size_t)S*O*sizeof(float)),*one=malloc((size_t)O*sizeof(float));
    double *ref=malloc((size_t)S*O*sizeof(double)),*mag=malloc((size_t)S*O*sizeof(double));
    for(int64_t k=0;k<O*rb;k++)codes[k]=(uint8_t)rnd_u();      /* every nibble, -0 and an odd width's pad too */
    for(int64_t k=0;k<O*ng;k++)scales[k]=(uint8_t)(118+rnd_u()%13);
    for(int64_t k=0;k<(int64_t)S*I;k++)x[k]=rnd()*3.f;
    Q38Weight w={0};
    w.data=codes;w.x2q=scales;w.rows=O;w.cols=I;w.elements=(int64_t)O*I;
    w.scale_count=O*ng;w.kind=Q38_WEIGHT_MXFP4;
    q38_weight_matmul(y,x,&w,S,I,O);
    reference(ref,mag,codes,scales,x,S,I,O);
    char what[160];
    snprintf(what,sizeof what,"[%d x %d], %d row(s): equals the dequantized reference to float rounding",O,I,S);
    ck(within(y,ref,mag,S*O,I),what);
    int same=1;
    for(int s=0;s<S&&same;s++){
        q38_weight_matmul(one,x+(int64_t)s*I,&w,1,I,O);
        same=!memcmp(one,y+(int64_t)s*O,(size_t)O*sizeof(float));
    }
    snprintf(what,sizeof what,"[%d x %d]: each batch row is the bits of that row alone",O,I);
    if(S>1)ck(same,what);
    snprintf(what,sizeof what,"[%d x %d]: q38_weight_bytes counts codes and scales",O,I);
    ck(q38_weight_bytes(&w)==(uint64_t)(O*rb+O*ng),what);
    free(codes);free(scales);free(x);free(y);free(one);free(ref);free(mag);
}

static void record_case(void){
    Q38Int4Experts x;memset(&x,0,sizeof x);
    q38_mx4_sizes(&x,2560,640);
    int64_t code_off[3],scale_off[3];
    q38_mx4_offsets(&x,code_off,scale_off);
    ck(x.record_bytes==2611200&&(x.record_bytes+4095)/4096*4096==2613248,
       "the release's record: 2,611,200 bytes, 2,613,248 on its 4096-byte stride");
    /* gate, up, down indices; the record holds gate, down, up scales, then their codes */
    ck(scale_off[0]==0&&scale_off[2]==640*80&&scale_off[1]==640*80+2560*20&&
       code_off[0]==2*640*80+2560*20&&code_off[2]==code_off[0]+640*1280&&code_off[1]==code_off[2]+2560*320&&
       code_off[1]+x.code_bytes[1]==x.record_bytes,
       "record order: scales of gate, down, up, then their e2m1 codes");
    q38_mx4_sizes(&x,32,8);   /* the tiny fixture: one short group in the down rows */
    ck(x.record_bytes==8*1*2+32*1+8*16*2+32*4,"a width under 32 keeps one scale per row");
}

/* the bytes the sidecar stores under this expert's names, read by name */
static uint8_t *stored(Model *m,int layer,int expert,int k,int scale,int64_t *bytes){
    const char *projection[3]={"gate_proj","up_proj","down_proj"};
    char suffix[192],name[320];
    snprintf(suffix,sizeof suffix,"mlp.experts.%d.%s.weight%s",expert,projection[k],scale?"_scale":"");
    q38_name(m,name,sizeof name,layer,suffix);
    st_tensor *t=st_find(&m->x4->S,name);
    if(!t)return NULL;
    uint8_t *out=malloc((size_t)t->nbytes);
    st_read_raw(&m->x4->S,name,out,0);*bytes=t->nbytes;
    return out;
}

/* a loaded slot holds the stored bytes, and multiplies like them */
static int slot_matches(Model *m,const Slot *slot,int layer,int expert){
    const Q38Weight *w[3]={&slot->gate,&slot->up,&slot->down};
    int ok=slot->eid==expert;
    for(int k=0;k<3&&ok;k++){
        int64_t nc=0,ns=0;
        uint8_t *codes=stored(m,layer,expert,k,0,&nc),*scales=stored(m,layer,expert,k,1,&ns);
        int O=w[k]->rows,I=w[k]->cols,S=3;
        ok=codes&&scales&&w[k]->kind==Q38_WEIGHT_MXFP4&&
           O==(k<2?m->c.inter:m->c.hidden)&&I==(k<2?m->c.hidden:m->c.inter)&&
           nc==(int64_t)O*((I+1)/2)&&ns==(int64_t)O*((I+31)/32)&&
           !memcmp(w[k]->data,codes,(size_t)nc)&&!memcmp(w[k]->x2q,scales,(size_t)ns);
        if(ok){
            float *x=malloc((size_t)S*I*sizeof(float)),*y=malloc((size_t)S*O*sizeof(float));
            double *ref=malloc((size_t)S*O*sizeof(double)),*mag=malloc((size_t)S*O*sizeof(double));
            for(int64_t q=0;q<(int64_t)S*I;q++)x[q]=rnd();
            q38_weight_matmul(y,x,w[k],S,I,O);
            reference(ref,mag,codes,scales,x,S,I,O);
            ok=within(y,ref,mag,S*O,I);
            free(x);free(y);free(ref);free(mag);
        }
        free(codes);free(scales);
    }
    return ok;
}

/* every dequantized element of the sidecar's expert equals the engine's f32 expansion of the FP8 one */
static int values_equal(Model *mx,const Slot *f32,int layer,int expert){
    const Q38Weight *w[3]={&f32->gate,&f32->up,&f32->down};
    int ok=1;
    for(int k=0;k<3&&ok;k++){
        int64_t nc=0,ns=0;
        uint8_t *codes=stored(mx,layer,expert,k,0,&nc),*scales=stored(mx,layer,expert,k,1,&ns);
        int O=w[k]->rows,I=w[k]->cols;int64_t rb=(I+1)/2,ng=(I+31)/32;
        ok=codes&&scales&&w[k]->kind==Q38_WEIGHT_F32;
        for(int o=0;o<O&&ok;o++)for(int i=0;i<I&&ok;i++)
            ok=(float)value_at(codes+o*rb,scales+o*ng,i)==((const float*)w[k]->data)[(int64_t)o*I+i];
        free(codes);free(scales);
    }
    return ok;
}

static void fixture_checks(const char *snap){
    printf("the sidecar of %s\n",snap);
    setenv("Q38_EXPERT_LOWBIT","mxfp4",1);
    Model m;model_init(&m,snap,1,8);q38_expert_int4_attach(&m,snap);
    ck(m.x4&&m.x4->lb&&m.x4->lb->mx,"the sidecar attaches under Q38_EXPERT_LOWBIT=mxfp4");
    if(!m.x4){q38_model_free(&m);return;}
    int L=m.c.layers,E=m.c.experts,ok=1;
    for(int layer=0;layer<L;layer++)for(int e=0;e<E;e++)
        ok&=slot_matches(&m,q38_expert_get(&m,layer,e),layer,e);
    ck(ok,"single LRU path (cap 1, every expert reloaded): slot bytes are the stored bytes, matmuls equal their reference");
    ck(m.expert_weight_reads==(uint64_t)L*E&&m.miss==m.expert_weight_reads,"one read per miss");
    q38_model_free(&m);

    Model b;model_init(&b,snap,E,8);q38_expert_int4_attach(&b,snap);
    int *all=malloc((size_t)E*sizeof(int));
    Slot **sel=malloc((size_t)E*sizeof(Slot*));
    for(int e=0;e<E;e++)all[e]=E-1-e;           /* not in id order */
    ok=1;int batched=1;
    for(int layer=0;layer<L;layer++){
        batched&=q38_expert_get_batch(&b,layer,all,E,sel);
        for(int e=0;e<E&&batched;e++)ok&=slot_matches(&b,sel[e],layer,all[e]);
    }
    ck(batched&&b.expert_parallel_batches==(uint64_t)L,"parallel batch path taken for every layer");
    ck(ok,"batch path: slot bytes are the stored bytes, matmuls equal their reference");

    /* the same experts from the snapshot's FP8 shards, expanded to f32 by the engine */
    unsetenv("Q38_EXPERT_LOWBIT");
    setenv("Q38_EXPERT_INT4","0",1);setenv("Q38_NATIVE_FP8","0",1);setenv("Q38_NATIVE_BF16","0",1);
    Model f;model_init(&f,snap,1,8);q38_expert_int4_attach(&f,snap);
    ck(f.x4==NULL,"without Q38_EXPERT_LOWBIT=mxfp4 the snapshot's experts stay");
    ok=1;
    for(int layer=0;layer<L;layer++)for(int e=0;e<E;e++)ok&=values_equal(&b,q38_expert_get(&f,layer,e),layer,e);
    ck(ok,"every MXFP4 value is the snapshot's FP8 value to the bit (gate, up and down where they belong)");
    unsetenv("Q38_EXPERT_INT4");unsetenv("Q38_NATIVE_FP8");unsetenv("Q38_NATIVE_BF16");
    q38_model_free(&f);free(all);free(sel);q38_model_free(&b);
}

int main(int argc,char **argv){
    printf("the MXFP4 expert matmul\n");
    kernel_case(640,2560,1);      /* gate/up of the release */
    kernel_case(640,2560,5);
    kernel_case(2560,640,1);      /* down */
    kernel_case(2560,640,9);
    kernel_case(8,32,3);          /* the tiny fixture: gate/up one group, down a short one */
    kernel_case(32,8,3);
    kernel_case(7,70,4);          /* two groups and a short third */
    kernel_case(5,99,2);          /* an odd width: the last byte's high nibble is padding */
    record_case();
    if(argc>1)fixture_checks(argv[1]);
    else printf("(no snapshot on argv: the sidecar checks run from make qwen38-tiny-mxfp4-check)\n");
    if(fails){printf("test_qwen38_mxfp4_experts: %d failure(s)\n",fails);return 1;}
    printf("test_qwen38_mxfp4_experts: ok\n");
    return 0;
}
