/* qwen38's trunk on the CPU as int8 rows met by an int8 activation in the
 * integer kernel (idot.h), and the vector FP8 kernel for the routed experts.
 *
 * Pinned, with no model:
 *  - the int8 trunk is the default; Q38_TRUNK_CPU_INT8=0 is the only way off;
 *  - a BF16 matrix quantized as the loader does answers q38_weight_matmul from
 *    its int8 rows within what an int8 row and an int8 activation allow, for one
 *    row (decode) and for a batch (prefill), and keeps answering after the BF16
 *    copy is released, with the same bytes;
 *  - the vector FP8 kernel agrees with quant.h's table kernel to float
 *    summation order on a matrix with a partial last block and an odd number of
 *    rows, for one row and for a batch, and Q38_FP8_KERNEL=scalar routes the
 *    dispatch to the table kernel byte for byte. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

static int fails;
static void ck(int ok,const char *what){ if(ok){printf("  ok   %s\n",what);return;} printf("  FAIL %s\n",what); fails++; }

static unsigned g_seed=777;
static float rnd(void){ g_seed=g_seed*1103515245u+12345u; return ((g_seed>>8)&0xFFFF)/32768.f-1.f; }
static double rel_gap(const float *a,const float *b,int n){
    double worst=0,scale=1e-6;
    for(int i=0;i<n;i++){ double d=fabs((double)a[i]-b[i]); if(d>worst)worst=d; if(fabs((double)b[i])>scale)scale=fabs((double)b[i]); }
    return worst/scale;
}
static uint16_t f32_to_bf16(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }

int main(void){
    printf("the flag\n");
    unsetenv("Q38_TRUNK_CPU_INT8");
    ck(q38_trunk_cpu_int8_wanted()==1,"int8 trunk on by default");
    setenv("Q38_TRUNK_CPU_INT8","0",1);
    ck(q38_trunk_cpu_int8_wanted()==0,"Q38_TRUNK_CPU_INT8=0 keeps BF16");
    setenv("Q38_TRUNK_CPU_INT8","1",1);
    ck(q38_trunk_cpu_int8_wanted()==1,"=1 still means on");
    unsetenv("Q38_TRUNK_CPU_INT8");

    printf("the int8 trunk\n");
    {
        enum { I=320, O=96, S=5 };
        Q38Weight w={0}; q38_weight_reserve(&w,Q38_WEIGHT_BF16,O,I);
        uint16_t *wb=(uint16_t*)w.data;
        for(int64_t k=0;k<(int64_t)O*I;k++)wb[k]=f32_to_bf16(rnd());
        float *x=malloc((size_t)S*I*sizeof(float)),*ref=malloc((size_t)S*O*sizeof(float));
        float *y=malloc((size_t)S*O*sizeof(float)),*y2=malloc((size_t)S*O*sizeof(float));
        for(int64_t k=0;k<(int64_t)S*I;k++)x[k]=rnd()*2.f;
        q38_matmul_bf16(ref,x,wb,S,I,O);
        q38_weight_matmul(y,x,&w,S,I,O);
        ck(!memcmp(y,ref,(size_t)S*O*sizeof(float)),"without int8 rows the BF16 kernel answers, byte for byte");
        q38_trunk_quantize(&w,&w.q8,&w.q8sc);
        q38_weight_matmul(y,x,&w,S,I,O);
        ck(rel_gap(y,ref,S*O)<3e-2,"int8 rows x int8 activation within 3% of the BF16 reference, 5 rows");
        q38_weight_matmul(y2,x,&w,1,I,O);
        ck(rel_gap(y2,ref,O)<3e-2,"same for one row");
        /* the loader releases the BF16 copy once the rows exist */
        free(w.data); w.data=NULL; w.owns_data=0;
        q38_weight_matmul(y2,x,&w,S,I,O);
        ck(!memcmp(y,y2,(size_t)S*O*sizeof(float)),"after the BF16 copy is released the int8 rows answer with the same bytes");
        free(w.q8); free(w.q8sc); free(x); free(ref); free(y); free(y2);
    }

    printf("the vector FP8 kernel\n");
    {
        enum { I=200, O=7, S=5 };          /* a partial last block, an odd row count */
        int nbo=(O+127)/128, nbi=(I+127)/128;
        uint8_t *q=malloc((size_t)O*I); float *sc=malloc((size_t)nbo*nbi*sizeof(float));
        for(int64_t k=0;k<(int64_t)O*I;k++){ unsigned b=(unsigned)((rnd()+1.f)*127.99f); if((b&0x7F)==0x7F)b&=0x7E; q[k]=(uint8_t)b; }
        for(int k=0;k<nbo*nbi;k++)sc[k]=0.5f+(rnd()+1.f)*0.75f;
        float *x=malloc((size_t)S*I*sizeof(float)),*ref=malloc((size_t)S*O*sizeof(float)),*y=malloc((size_t)S*O*sizeof(float));
        for(int64_t k=0;k<(int64_t)S*I;k++)x[k]=rnd();
        matmul_fp8(ref,x,q,sc,S,I,O);
        setenv("Q38_FP8_KERNEL","scalar",1);
        q38_matmul_fp8(y,x,q,sc,S,I,O);
        ck(!memcmp(y,ref,(size_t)S*O*sizeof(float)),"Q38_FP8_KERNEL=scalar: the table kernel, byte for byte");
#ifdef __AVX2__
        q38_matmul_fp8_vec(y,x,q,sc,S,I,O);
        ck(rel_gap(y,ref,S*O)<1e-5,"vector kernel within float summation order of the table kernel, 5 rows");
        q38_matmul_fp8_vec(y,x,q,sc,1,I,O);
        ck(rel_gap(y,ref,O)<1e-5,"same for one row (the fused decode-and-multiply path)");
        float one[O]; memcpy(one,y,sizeof one);
        q38_matmul_fp8_vec(y,x,q,sc,2,I,O);
        ck(!memcmp(one,y,sizeof one),"the first of two rows equals the one-row result to the byte");
#else
        printf("  skip vector kernel: not an AVX2 build\n");
#endif
        free(q); free(sc); free(x); free(ref); free(y);
    }
    if(fails){ printf("test_qwen38_idot: %d failure(s)\n",fails); return 1; }
    printf("OK test_qwen38_idot: int8 trunk and vector FP8 kernel\n");
    return 0;
}
