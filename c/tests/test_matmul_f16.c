/* Half-precision weight matmul exactness gate.
 *
 * matmul_f16w() / matmul_bf16w() widen an IEEE-half / BF16 weight to f32 (exact) and run the same fixed lanes,
 * the same fused multiply-add and the same halving tree as matmul() in matmul_f32.h. So it
 * must produce exactly the bits matmul() produces on the widened matrix. This compares them
 * with memcmp over shapes that exercise full 32-lane blocks, a partial last block, rows
 * shorter than one block and a ragged length that is not a multiple of 8, plus the
 * OLMoE attention / lm_head geometry.
 *
 * Without F16C matmul_f16w is not defined and only the BF16 half of this test runs.
 *
 * Build: make -C c tests/test_matmul_f16 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "../matmul_f16.h"

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float rnd_f(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (float)(rng>>40)/(float)(1<<24)*2.f-1.f; }

int main(void){
    static const int shapes[][3]={ {1,1,1}, {1,5,8}, {1,8,3}, {1,31,3}, {1,32,3}, {1,33,5}, {3,33,17}, {2,1027,41},
                                   {1,2048,64}, {4,2048,64}, {1,2048,2048}, {1,2048,1021}, {5,6144,256} };
    int fails=0;
    for (size_t k=0;k<sizeof shapes/sizeof shapes[0];k++){
        int S=shapes[k][0], I=shapes[k][1], O=shapes[k][2];
        float *x=malloc((size_t)S*I*sizeof(float)), *Wf=malloc((size_t)O*I*sizeof(float));
        uint16_t *Wh=malloc((size_t)O*I*sizeof(uint16_t));
        float *y1=malloc((size_t)S*O*sizeof(float)), *y2=malloc((size_t)S*O*sizeof(float));
        if (!x||!Wf||!Wh||!y1||!y2){ fprintf(stderr,"oom\n"); return 1; }
        for (int64_t i=0;i<(int64_t)S*I;i++) x[i]=rnd_f();
#ifdef COLI_MATMUL_F16
        for (int64_t i=0;i<(int64_t)O*I;i++){ Wh[i]=_cvtss_sh(rnd_f()*0.1f,0); Wf[i]=_cvtsh_ss(Wh[i]); }   /* Wf = widened Wh */
        matmul(y1,x,Wf,S,I,O);
        matmul_f16w(y2,x,Wh,S,I,O);
        int same=memcmp(y1,y2,(size_t)S*O*sizeof(float))==0;
        printf("%s  f16  S=%d I=%d O=%d\n", same?"PASS":"FAIL", S, I, O);
        if (!same) fails++;
#endif
        for (int64_t i=0;i<(int64_t)O*I;i++){                                        /* BF16: the top half of an f32 */
            float f=rnd_f()*0.1f; uint32_t u; memcpy(&u,&f,sizeof u);
            Wh[i]=(uint16_t)(u>>16); Wf[i]=bf16w_to_f32(Wh[i]);
        }
        matmul(y1,x,Wf,S,I,O);
        matmul_bf16w(y2,x,Wh,S,I,O);
        int same_bf=memcmp(y1,y2,(size_t)S*O*sizeof(float))==0;
        printf("%s  bf16 S=%d I=%d O=%d\n", same_bf?"PASS":"FAIL", S, I, O);
        if (!same_bf) fails++;
        free(x); free(Wf); free(Wh); free(y1); free(y2);
    }
    printf("\n%s\n", fails?"FAILED":"ALL PASS");
    return fails?1:0;
}
