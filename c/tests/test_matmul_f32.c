/* f32 matmul exactness gate (#442).
 *
 * On FMA targets quant.h's matmul computes each output as one fmaf chain over
 * i in order, eight rows at a time. The row blocking must not change a single
 * bit: this compares it with memcmp against the one-row fmaf chain, over shapes
 * that exercise the eight-row blocks, the leftover rows and short rows. Without
 * FMA matmul keeps the plain `a+=x*w` loop, and the reference follows it.
 *
 * Build: make -C c tests/test_matmul_f32 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "../quant.h"

static uint64_t rng = 0x2545F4914F6CDD1Dull;
static float rnd_f(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (float)(rng>>40)/(float)(1<<24)*2.f-1.f; }

static void ref_matmul(float *y, const float *x, const float *W, int S, int I, int O){
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0;
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
            for(int i=0;i<I;i++) a=fmaf(xs[i],w[i],a);
#else
            for(int i=0;i<I;i++) a+=xs[i]*w[i];
#endif
            y[(int64_t)s*O+o]=a; } }
}

int main(void){
    static const int shapes[][3]={ {1,1,1}, {1,5,8}, {1,7,13}, {3,33,17}, {2,1027,41},
                                   {1,2048,64}, {4,2048,64}, {5,6144,256}, {1,6144,257} };
    int fails=0;
    for (size_t k=0;k<sizeof shapes/sizeof shapes[0];k++){
        int S=shapes[k][0], I=shapes[k][1], O=shapes[k][2];
        float *x=malloc(sizeof(float)*(size_t)S*I), *W=malloc(sizeof(float)*(size_t)I*O);
        float *y=malloc(sizeof(float)*(size_t)S*O), *r=malloc(sizeof(float)*(size_t)S*O);
        for (size_t i=0;i<(size_t)S*I;i++) x[i]=rnd_f();
        for (size_t i=0;i<(size_t)I*O;i++) W[i]=rnd_f()*0.05f;
        matmul(y,x,W,S,I,O); ref_matmul(r,x,W,S,I,O);
        int bad=0; for (int i=0;i<S*O;i++) bad+=memcmp(&y[i],&r[i],sizeof(float))!=0;
        printf("%s  matmul S=%d I=%d O=%d: %d/%d outputs differ from the reference\n",
               bad?"FAIL":"PASS", S, I, O, bad, S*O);
        fails+=bad!=0;
        free(x); free(W); free(y); free(r);
    }
    printf("\n%s\n", fails ? "FAILED" : "ALL PASS");
    return fails?1:0;
}
