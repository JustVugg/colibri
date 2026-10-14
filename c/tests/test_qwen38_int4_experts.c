/* qwen38's routed experts as int4-g64 (the experts-int4g64/ sidecar written by
 * tools/convert_qwen38_experts_int4.py).
 *
 * Pinned with no model:
 *  - q38_weight_matmul on an int4-g64 weight equals the dequantized reference
 *    ((code - 8) times its group's scale, summed in double) to float rounding,
 *    on the release's two expert shapes and on widths whose tail is stored in
 *    pairs, for one row (decode) and for a batch (prefill);
 *  - every row of a batch is the same bits as that row alone, so decode and
 *    the prefill batch cannot drift apart, and a row of whole blocks is the
 *    same bits as expert_ffn.h's scalar reference, its kernel's contract.
 * With a converted snapshot on argv (make qwen38-tiny-int4-check):
 *  - the sidecar attaches, and every expert loaded through the single LRU path
 *    and through the parallel batch path holds exactly the bytes the sidecar
 *    stores under that expert's tensor names, read by name;
 *  - its matmuls equal the reference built from those bytes;
 *  - the prefetch advises one range per expert the cache does not hold;
 *  - the int4 values are this snapshot's experts: against the engine's own
 *    expanded FP8 their relative L2 error is what int4-g64 costs (2-20%). */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

static int fails;
static void ck(int ok,const char *what){ if(ok){printf("  ok   %s\n",what);return;} printf("  FAIL %s\n",what); fails++; }

static unsigned g_seed=20261002;
static unsigned rnd_u(void){ g_seed=g_seed*1103515245u+12345u; return g_seed>>8; }
static float rnd(void){ return (float)(rnd_u()&0xFFFF)/32768.f-1.f; }

/* element i of an int4-g64 row, decoded without the engine's helpers */
static int code_at(const uint8_t *row,int cols,int i){
    int body=cols/64*64;
    if(i<body){ const uint8_t *blk=row+(i/64)*32; int k=i%64; return k<32?(blk[k]&15):(blk[k-32]>>4); }
    return (i&1)?(row[i>>1]>>4):(row[i>>1]&15);
}

/* y = W x in double, and the sum of |terms| the float error is measured against */
static void reference(double *y,double *mag,const uint8_t *codes,const float *scales,
                      const float *x,int S,int I,int O){
    int64_t rb=(I+1)/2,ng=(I+63)/64;
    for(int s=0;s<S;s++)for(int o=0;o<O;o++){
        double a=0,m=0;
        for(int i=0;i<I;i++){
            double t=(double)(code_at(codes+o*rb,I,i)-8)*(double)scales[o*ng+i/64]*(double)x[(int64_t)s*I+i];
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
    int64_t rb=(I+1)/2,ng=(I+63)/64;
    uint8_t *codes=malloc((size_t)(O*rb));float *scales=malloc((size_t)(O*ng)*sizeof(float));
    float *x=malloc((size_t)S*I*sizeof(float)),*y=malloc((size_t)S*O*sizeof(float)),*one=malloc((size_t)O*sizeof(float));
    double *ref=malloc((size_t)S*O*sizeof(double)),*mag=malloc((size_t)S*O*sizeof(double));
    for(int64_t k=0;k<O*rb;k++)codes[k]=(uint8_t)rnd_u();      /* every nibble, the odd tail's pad too */
    for(int64_t k=0;k<O*ng;k++)scales[k]=0.001f+0.05f*(rnd()+1.f);
    for(int64_t k=0;k<(int64_t)S*I;k++)x[k]=rnd()*3.f;
    Q38Weight w={0};
    w.data=codes;w.scales=scales;w.rows=O;w.cols=I;w.elements=(int64_t)O*I;
    w.scale_count=O*ng;w.kind=Q38_WEIGHT_INT4G64;
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
    if(I%64==0){
        int exact=1;
        for(int o=0;o<O&&exact;o++)
            exact=y[o]==xf_dot_f32_ref(codes+o*rb,scales+o*ng,x,I);
        snprintf(what,sizeof what,"[%d x %d]: bit-identical to expert_ffn.h's scalar reference",O,I);
        ck(exact,what);
    }
    free(codes);free(scales);free(x);free(y);free(one);free(ref);free(mag);
}

/* the bytes the sidecar stores under this expert's names, read by name */
static uint8_t *stored(Model *m,int layer,int expert,int k,int scale,int64_t *bytes){
    const char *projection[3]={"gate_proj","up_proj","down_proj"};
    char suffix[192],name[320];
    snprintf(suffix,sizeof suffix,"mlp.experts.%d.%s.weight%s",expert,projection[k],scale?".qs":"");
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
        ok=codes&&scales&&w[k]->kind==Q38_WEIGHT_INT4G64&&
           O==(k<2?m->c.inter:m->c.hidden)&&I==(k<2?m->c.hidden:m->c.inter)&&
           nc==O*q38_int4_row_bytes(I)&&ns==(int64_t)O*q38_int4_groups(I)*(int64_t)sizeof(float)&&
           !memcmp(w[k]->data,codes,(size_t)nc)&&!memcmp(w[k]->scales,scales,(size_t)ns);
        if(ok){
            float *x=malloc((size_t)S*I*sizeof(float)),*y=malloc((size_t)S*O*sizeof(float));
            double *ref=malloc((size_t)S*O*sizeof(double)),*mag=malloc((size_t)S*O*sizeof(double));
            for(int64_t q=0;q<(int64_t)S*I;q++)x[q]=rnd();
            q38_weight_matmul(y,x,w[k],S,I,O);
            reference(ref,mag,codes,(const float*)scales,x,S,I,O);
            ok=within(y,ref,mag,S*O,I);
            free(x);free(y);free(ref);free(mag);
        }
        free(codes);free(scales);
    }
    return ok;
}

static double dequant_gap(Model *m,const Slot *fp8,int layer,int expert,int k){
    const Q38Weight *w=k==0?&fp8->gate:k==1?&fp8->up:&fp8->down;
    int64_t nc=0,ns=0;
    uint8_t *codes=stored(m,layer,expert,k,0,&nc);float *scales=(float*)stored(m,layer,expert,k,1,&ns);
    int O=w->rows,I=w->cols;int64_t rb=(I+1)/2,ng=(I+63)/64;
    double num=0,den=0;
    for(int o=0;o<O;o++)for(int i=0;i<I;i++){
        double source=((const float*)w->data)[(int64_t)o*I+i];
        double v=(double)(code_at(codes+o*rb,I,i)-8)*scales[o*ng+i/64];
        num+=(v-source)*(v-source);den+=source*source;
    }
    free(codes);free(scales);
    return den>0?sqrt(num/den):0;
}

static void fixture_checks(const char *snap){
    printf("the sidecar of %s\n",snap);
    setenv("Q38_EXPERT_INT4","1",1);            /* refuse to fall back to FP8 */
    Model m;model_init(&m,snap,1,8);q38_expert_int4_attach(&m,snap);
    ck(m.x4!=NULL,"the sidecar attaches");
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
    q38_prefetch_experts(&b,0,all,E);
    ck(b.expert_prefetch_ranges==(uint64_t)E,"prefetch advises one range per expert not in the cache");
    ok=1;int batched=1;
    for(int layer=0;layer<L;layer++){
        batched&=q38_expert_get_batch(&b,layer,all,E,sel);
        for(int e=0;e<E&&batched;e++)ok&=slot_matches(&b,sel[e],layer,all[e]);
    }
    ck(batched&&b.expert_parallel_batches==(uint64_t)L,"parallel batch path taken for every layer");
    ck(ok,"batch path: slot bytes are the stored bytes, matmuls equal their reference");
    uint64_t prefetched=b.expert_prefetch_ranges;
    q38_prefetch_experts(&b,0,all,E);
    ck(b.expert_prefetch_ranges==prefetched,"nothing is advised once the experts are cached");

    /* the same experts from the snapshot itself, expanded to f32 by the engine */
    setenv("Q38_EXPERT_INT4","0",1);setenv("Q38_NATIVE_FP8","0",1);setenv("Q38_NATIVE_BF16","0",1);
    Model f;model_init(&f,snap,1,8);q38_expert_int4_attach(&f,snap);
    ck(f.x4==NULL,"Q38_EXPERT_INT4=0 leaves the snapshot's experts");
    double worst=0,sum=0;int n=0,kind_ok=1;
    for(int layer=0;layer<L;layer++)for(int e=0;e<E;e++){
        Slot *slot=q38_expert_get(&f,layer,e);
        kind_ok&=slot->gate.kind==Q38_WEIGHT_F32;
        for(int k=0;k<3;k++){double g=dequant_gap(&b,slot,layer,e,k);sum+=g;n++;if(g>worst)worst=g;}
    }
    printf("  int4-g64 against the engine's expanded experts: relative L2 mean %.4f, worst %.4f over %d matrices\n",
           n?sum/n:0,worst,n);
    ck(kind_ok&&n&&sum/n>0.02&&worst<0.2,"the int4 values are this snapshot's experts, within int4-g64's error");
    unsetenv("Q38_EXPERT_INT4");unsetenv("Q38_NATIVE_FP8");unsetenv("Q38_NATIVE_BF16");
    q38_model_free(&f);free(all);free(sel);q38_model_free(&b);
}

int main(int argc,char **argv){
    printf("the int4-g64 matmul\n");
    kernel_case(640,2560,1);      /* gate/up of the release */
    kernel_case(640,2560,5);
    kernel_case(2560,640,1);      /* down */
    kernel_case(2560,640,9);
    kernel_case(8,32,3);          /* the tiny fixture: one short group */
    kernel_case(32,8,3);
    kernel_case(7,130,4);         /* two blocks and a two-element tail */
    kernel_case(5,99,2);          /* an odd tail: the last byte's high nibble is padding */
    if(argc>1)fixture_checks(argv[1]);
    else printf("(no snapshot on argv: the sidecar checks run from make qwen38-tiny-int4-check)\n");
    if(fails){printf("test_qwen38_int4_experts: %d failure(s)\n",fails);return 1;}
    printf("test_qwen38_int4_experts: ok\n");
    return 0;
}
