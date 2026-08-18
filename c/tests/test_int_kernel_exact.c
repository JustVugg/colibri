/* Integer-kernel exactness gate (#1081).
 *
 * The olmoe (#1044) / qwen36 (#712) / inkling (#1080) class of bug was an ISA
 * branch of an INTEGER kernel that was not equivalent to the reference path —
 * invisible to every x86-only oracle. Integer kernels have no rounding excuse:
 * against a pure-C reference on the same inputs they must match BIT FOR BIT,
 * on every ISA. This test builds that reference and asserts it for:
 *
 *   - dot_i4i8            (pair-layout int4 x int8 — AVX2/VNNI/AVX-512/NEON)
 *   - planarize_i4        (pair -> plane repack must preserve logical weights)
 *   - dot_i4p_u           (K1 plane-layout unsigned dot, - 8*sum(x) identity)
 *   - matmul_i4p_idot     (must equal matmul_i4_idot bitwise, same scales)
 *
 * Compile with the NATIVE arch so the SIMD branches under test actually run:
 *   x86: gcc -O3 -march=native -fopenmp -I. tests/test_int_kernel_exact.c
 *   arm: gcc -O3 -mcpu=native  -fopenmp -I. tests/test_int_kernel_exact.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "../quant.h"

static uint64_t rng = 0x2545F4914F6CDD1Dull;
static uint32_t rnd(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
/* CONTRATTO ATTIVAZIONI: qrow_i8 clampa a [-127,127] — il -128 non entra mai
 * nei kernel, e il trucco abs/sign dei rami VNNI non saprebbe rappresentarlo
 * (-(-128) resta -128 in int8). Il generatore deve rispettare il contratto.
 * EN: activations are clamped to [-127,127] by qrow_i8; -128 never reaches the
 * kernels, and the abs/sign VNNI trick could not represent it anyway. */
static int8_t rnd_x(void){ int v=(int)(rnd()%255)-127; return (int8_t)v; }

/* pure-C reference: signed pair-layout int4 dot (the format's definition) */
static int64_t ref_dot_pair(const uint8_t *w4, const int8_t *x, int I){
    int64_t s=0;
    for(int i=0;i<I;i++){
        uint8_t byte=w4[i>>1];
        int v=(int)((i&1)?(byte>>4):(byte&0xF))-8;
        s+=(int64_t)v*x[i];
    }
    return s;
}
static int64_t ref_sum(const int8_t *x, int I){
    int64_t s=0; for(int i=0;i<I;i++) s+=x[i]; return s;
}

int main(void){
    int sizes[]={32,33,64,96,100,2048,6100,6144};
    int fails=0, checks=0;
    for(size_t t=0;t<sizeof(sizes)/sizeof(*sizes);t++){
        int I=sizes[t], rb=(I+1)/2;
        for(int rep=0;rep<8;rep++){
            uint8_t *wp=malloc(rb); for(int i=0;i<rb;i++) wp[i]=(uint8_t)rnd();
            int8_t  *x =malloc(I);  for(int i=0;i<I;i++)  x[i]=rnd_x();
            int64_t ref=ref_dot_pair(wp,x,I);

            /* 1) pair-layout kernel vs reference */
            checks++;
            if((int64_t)dot_i4i8(wp,x,I)!=ref){
                fails++; fprintf(stderr,"FAIL dot_i4i8 I=%d rep=%d\n",I,rep); }

            /* 2) planarize preserves the logical weights; 3) plane dot identity */
            uint8_t *wl=malloc(rb); memcpy(wl,wp,rb); planarize_i4_row(wl,I);
            checks++;
            int64_t got=(int64_t)dot_i4p_u(wl,x,I)-8*ref_sum(x,I);
            if(got!=ref){
                fails++; fprintf(stderr,"FAIL dot_i4p_u I=%d rep=%d (got %lld want %lld)\n",
                                 I,rep,(long long)got,(long long)ref); }
            free(wp); free(x); free(wl);
        }
    }
    /* 4) matmul-level: planar IDOT must equal pair IDOT bitwise */
    {
        int O=512, I=2048, rb=(I+1)/2, S=6;   /* K2: 4 righe di tile + 2 di resto */
        uint8_t *qp=malloc((size_t)O*rb); for(size_t i=0;i<(size_t)O*rb;i++) qp[i]=(uint8_t)rnd();
        uint8_t *ql=malloc((size_t)O*rb); memcpy(ql,qp,(size_t)O*rb); planarize_i4(ql,O,I);
        float *sc=malloc(O*sizeof(float)); for(int o=0;o<O;o++) sc[o]=0.001f+(rnd()%997)*1e-6f;
        int8_t *xq=malloc((size_t)S*I); for(size_t i=0;i<(size_t)S*I;i++) xq[i]=rnd_x();
        float *sx=malloc(S*sizeof(float)); int32_t *xs=malloc(S*sizeof(int32_t));
        for(int s=0;s<S;s++){ sx[s]=0.01f+(rnd()%97)*1e-4f;
            int32_t a=0; for(int i=0;i<I;i++) a+=xq[(size_t)s*I+i]; xs[s]=a; }
        float *ya=malloc((size_t)S*O*sizeof(float)), *yb=malloc((size_t)S*O*sizeof(float));
        matmul_i4_idot(ya,xq,sx,qp,sc,S,I,O);
        matmul_i4p_idot(yb,xq,sx,xs,ql,sc,S,I,O);
        for(size_t i=0;i<(size_t)S*O;i++){ checks++;
            if(memcmp(&ya[i],&yb[i],4)){ fails++;
                fprintf(stderr,"FAIL matmul_i4p_idot @%zu\n",i); if(fails>8) break; } }
        free(qp);free(ql);free(sc);free(xq);free(sx);free(xs);free(ya);free(yb);
    }
#if defined(__AVX2__) && !(defined(__AVX512F__) && defined(__AVX512BW__))
    /* f32 planare vs f32 a coppie: bit-uguali DOVE il gate accende il planare.
     * Il gate del motore (planar_on) esclude ARM (niente ramo NEON in
     * matmul_i4p) E i build AVX-512F (matmul_i4 li' puo' prendere il ramo
     * dot_i4f_avx512, ordine di accumulo diverso) — il test DEVE rispecchiare
     * il gate alla lettera, o asserisce un claim che il motore non fa.
     * Trovato dal runner-lottery: la PR e' passata su un runner AVX2, il push
     * su dev e' finito su un runner AVX-512-VNNI ed e' fallito qui (#1081).
     * EN: the test must mirror planar_on() exactly; an AVX-512 runner runs
     * matmul_i4's 512-bit arm while the engine keeps planar OFF there. */
    {
        int O=256, I=6100, rb=(I+1)/2, S=3;   /* coda I%64!=0 inclusa */
        uint8_t *qp=malloc((size_t)O*rb); for(size_t i=0;i<(size_t)O*rb;i++) qp[i]=(uint8_t)rnd();
        uint8_t *ql=malloc((size_t)O*rb); memcpy(ql,qp,(size_t)O*rb); planarize_i4(ql,O,I);
        float *sc=malloc(O*sizeof(float)); for(int o=0;o<O;o++) sc[o]=0.001f+(rnd()%997)*1e-6f;
        float *x=malloc((size_t)S*I*sizeof(float));
        for(size_t i=0;i<(size_t)S*I;i++) x[i]=((int32_t)(rnd()&0xFFFF)-0x8000)/8192.0f;
        float *ya=malloc((size_t)S*O*sizeof(float)), *yb=malloc((size_t)S*O*sizeof(float));
        matmul_i4(ya,x,qp,sc,S,I,O);
        matmul_i4p(yb,x,ql,sc,S,I,O);
        for(size_t i=0;i<(size_t)S*O;i++){ checks++;
            if(memcmp(&ya[i],&yb[i],4)){ fails++;
                fprintf(stderr,"FAIL matmul_i4p f32 @%zu\n",i); if(fails>8) break; } }
        free(qp);free(ql);free(sc);free(x);free(ya);free(yb);
    }
#endif
    printf("int-kernel exactness: %d checks, %d failures (%s / " IDOT_KERNEL ")\n",
           checks,fails,fails?"FAIL":"PASS");
    return fails?1:0;
}
