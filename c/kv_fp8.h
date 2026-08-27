#ifndef COLIBRI_KV_FP8_H
#define COLIBRI_KV_FP8_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/* FP8 e4m3 (OCP "fn" variant: no infinities, max ±448, NaN = S.1111.111) per la
 * KV-cache latente MLA. Il latente compresso e' sia key sia value nell'attention
 * assorbita, quindi l'errore di quantizzazione entra negli score E nel context —
 * lo stesso regime della pratica FP8-KV di DeepSeek-V3. Scala per-riga (amax/448,
 * f32, Lc e Rc separate): dentro una riga la dinamica e' piccola, tra token no.
 * EN: e4m3 storage for the MLA latent KV rows, per-row f32 amax scale. Decode is
 * lut[byte]*scale; encode is bit-math + round-to-nearest-even, no libraries. */

static float coli_fp8_lut[256];

static void coli_fp8_lut_init(void){
    if(coli_fp8_lut[1]!=0.f) return;                 /* idempotente */
    for(int b=0;b<256;b++){
        int E=(b>>3)&0xF, M=b&7;
        float v = E ? ldexpf(1.f+(float)M/8.f, E-7)  /* normale: (1+M/8)*2^(E-7) */
                    : ldexpf((float)M, -9);          /* denormale: M/8 * 2^-6 */
        if(E==15 && M==7) v=0.f;                     /* codice NaN: mai scritto; inerte in lettura */
        coli_fp8_lut[b] = (b&0x80) ? -v : v;
    }
}

/* float -> e4m3, round-to-nearest-even, saturazione a ±448 (come __nv_cvt SATFINITE).
 * NaN -> 0: una riga f32 con NaN avvelenerebbe comunque tutto; 0 e' inerte nei dot. */
static inline uint8_t coli_fp8_enc(float f){
    union { float f; uint32_t u; } v; v.f=f;
    uint8_t s=(uint8_t)((v.u>>24)&0x80u);
    float a=fabsf(f);
    if(!(a<=448.f)) return (a!=a) ? 0 : (uint8_t)(s|0x7e);   /* NaN -> 0, inf/overflow -> ±448 */
    if(a<0x1p-6f){                                   /* griglia denormale: passo 2^-9 */
        int k=(int)rintf(a*0x1p9f);                  /* RNE su [0..8] */
        return (uint8_t)(s|(uint8_t)k);              /* k==8 -> 0x08 = primo normale 2^-6 */
    }
    int e; frexpf(a,&e);                             /* a = m*2^e, m in [0.5,1) */
    int E=e+6;                                       /* esponente biased e4m3 (1..15) */
    int k=(int)rintf(ldexpf(a,3-(e-1)));             /* mantissa RNE in [8,16] */
    if(k==16){ k=8; E++; }                           /* overflow di mantissa -> esponente su */
    if(E>15||(E==15&&k>14)) return (uint8_t)(s|0x7e);
    return (uint8_t)(s|(uint8_t)(E<<3)|(uint8_t)(k-8));
}

/* quantizza una riga latente: scala amax/448 per-riga, ritorna la scala.
 * La riga si decodifica come coli_fp8_lut[b]*scale. Riga tutta zero, subnormale
 * (448/amax andrebbe a +inf), tutta NaN o con ±inf: byte 0 e scala 1, niente 1/0.
 * NB: un NaN MISTO a valori finiti non alza amax (fabsf(NaN)>amax e' falso) —
 * la riga si quantizza normalmente e il NaN diventa byte 0 via coli_fp8_enc. */
static inline float coli_kv8_quant_row(const float *src, uint8_t *dst, int n){
    float amax=0;
    for(int i=0;i<n;i++){ float a=fabsf(src[i]); if(a>amax) amax=a; }
    if(!(amax>1e-35f) || amax>3.4e38f){ memset(dst,0,(size_t)n); return 1.f; }
    float inv=448.f/amax;
    for(int i=0;i<n;i++) dst[i]=coli_fp8_enc(src[i]*inv);
    return amax/448.f;
}

/* dequantizza una riga (staging f32 per i consumatori matmul) */
static inline void coli_kv8_dequant_row(const uint8_t *src, float scale, float *dst, int n){
    for(int i=0;i<n;i++) dst[i]=coli_fp8_lut[src[i]]*scale;
}

/* KV8_GS: grouped-scale variants (FlashMLA keeps one f32 scale per 128 latent
 * elements; one amax per 512-dim row lets a single outlier dilate the grid for
 * the whole row). gs==0 -> the per-row functions above. The row's scales live
 * consecutively: ceil(n/gs) floats per row. */
static inline int coli_kv8_nscale(int n, int gs){ return gs>0 ? (n+gs-1)/gs : 1; }
static inline void coli_kv8_quant_row_gs(const float *src, uint8_t *dst, float *sc, int n, int gs){
    if(gs<=0){ sc[0]=coli_kv8_quant_row(src,dst,n); return; }
    for(int g=0,k=0;g<n;g+=gs,k++){
        int m=n-g<gs?n-g:gs;
        sc[k]=coli_kv8_quant_row(src+g,dst+g,m);
    }
}
static inline void coli_kv8_dequant_row_gs(const uint8_t *src, const float *sc, float *dst, int n, int gs){
    if(gs<=0){ coli_kv8_dequant_row(src,sc[0],dst,n); return; }
    for(int g=0,k=0;g<n;g+=gs,k++){
        int m=n-g<gs?n-g:gs;
        coli_kv8_dequant_row(src+g,sc[k],dst+g,m);
    }
}

#endif
