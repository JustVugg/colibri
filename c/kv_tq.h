#ifndef COLIBRI_KV_TQ_H
#define COLIBRI_KV_TQ_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/* PolarQuant (Han/Kacham/Mirrokni/Zandieh/Karbasi, arXiv:2502.02617 — la base di
 * TurboQuant): quantizzazione KV data-oblivious in due parti, senza calibrazione.
 *   1) precondizionamento con una ROTAZIONE casuale (qui Hadamard randomizzato:
 *      sign-flip deterministico D + FWHT normalizzato H, ortonormale e auto-inverso)
 *      — rende le coordinate ~gaussiane, quindi la griglia degli angoli e' nota a
 *      priori e non serve normalizzare per riga;
 *   2) TRASFORMATA POLARE ricorsiva a coppie: il vettore ruotato diventa (raggio,
 *      n-1 angoli). Il raggio E' la norma L2 della riga (la rotazione la conserva),
 *      quindi riusa la scala f32 per-riga che KV8 ha gia' (Lsc/Rsc). Solo gli angoli
 *      finiscono nei byte: livello 1 uniforme su (-pi,pi], livelli >=2 concentrati
 *      su pi/4 (griglia centrata, mezza-ampiezza analitica).
 *
 * EN: two-part data-oblivious KV quantization, no calibration. (1) randomized
 * Hadamard rotation (deterministic sign flips + normalized self-inverse FWHT) makes
 * coordinates near-Gaussian so the angle grid is distribution-free; (2) recursive
 * pairwise polar transform turns the rotated row into (radius, n-1 angles). The
 * radius is the row's L2 norm (rotation-invariant) and reuses the per-row f32 scale
 * that KV8 already stores; only the angles are packed into bytes. Phase 1 is
 * PolarQuant only — the QJL 1-bit residual (arXiv:2406.03482) is a later phase.
 *
 * n MUST be a power of two (kv_lora=512, qk_rope=64 both qualify) and <= COLI_TQ_MAXN.
 * Decode is coli_tq_dequant_row -> f32 staging, same shape as coli_kv8_dequant_row. */

#ifndef COLI_TQ_SEED
#define COLI_TQ_SEED 0x9E3779B9u                  /* fixed rotation seed (golden ratio) */
#endif
#define COLI_TQ_MAXN 2048                          /* max row width (kv_lora/qk_rope <= this) */

/* ---- randomized Hadamard rotation ---------------------------------------- */

/* splitmix32: deterministic bit-mix, gives the reproducible +-1 sign vector D. */
static inline uint32_t coli_tq_hash(uint32_t x){
    x += 0x9E3779B9u;
    x = (x ^ (x>>16)) * 0x85EBCA6Bu;
    x = (x ^ (x>>13)) * 0xC2B2AE35u;
    return x ^ (x>>16);
}
static inline float coli_tq_sign(int i, uint32_t seed){
    return (coli_tq_hash(seed ^ (uint32_t)i) & 1u) ? -1.f : 1.f;
}

/* Normalized fast Walsh-Hadamard transform, in place. H_n has +-1 entries and
 * H_n^2 = n*I, so the normalized H/sqrt(n) is symmetric, orthogonal AND its own
 * inverse: applying this twice returns the input (the "FWHT self-inverse" gate). */
static inline void coli_tq_fwht(float *a, int n){
    for(int len=1; len<n; len<<=1)
        for(int i=0;i<n;i+=len<<1)
            for(int j=i;j<i+len;j++){
                float u=a[j], v=a[j+len];
                a[j]=u+v; a[j+len]=u-v;
            }
    float inv=1.f/sqrtf((float)n);
    for(int i=0;i<n;i++) a[i]*=inv;
}

/* y = R x = H (D x). R is orthonormal, so ||y|| == ||x||. */
static inline void coli_tq_rotate(const float *x, float *y, int n, uint32_t seed){
    for(int i=0;i<n;i++) y[i]=x[i]*coli_tq_sign(i,seed);
    coli_tq_fwht(y,n);
}
/* x = R^-1 y = D (H y), in place (H symmetric orthogonal, D^-1 = D). */
static inline void coli_tq_unrotate(float *y, int n, uint32_t seed){
    coli_tq_fwht(y,n);
    for(int i=0;i<n;i++) y[i]*=coli_tq_sign(i,seed);
}

/* ---- angle grids --------------------------------------------------------- */
/* Bit split: level-1 angles (n/2 of them) get b1=bits; every deeper level gets
 * b2=bits-1 (>=1). Level-1 angles are uniform on (-pi,pi] after the rotation;
 * deeper-level angles are atan2 of two nonnegative sub-radii, in [0,pi/2] and
 * sharply concentrated at pi/4 (variance ~ 1/(4s), s = subtree half-size), so a
 * uniform grid would waste codes — we center a narrow uniform grid on pi/4. */
static inline int coli_tq_b1(int bits){ return bits; }
static inline int coli_tq_b2(int bits){ return bits>1 ? bits-1 : 1; }

#ifndef COLI_TQ_PI
#define COLI_TQ_PI 3.14159265358979323846f
#endif

/* level-1: uniform bins over (-pi,pi] */
static inline uint32_t coli_tq_enc1(float ang, int b){
    float u = (ang + COLI_TQ_PI) * (0.5f/COLI_TQ_PI);      /* -> [0,1) */
    int m = (1<<b)-1;
    int q = (int)floorf(u*(float)(1<<b));
    if(q<0) q=0; if(q>m) q=m;
    return (uint32_t)q;
}
static inline float coli_tq_dec1(uint32_t q, int b){
    return -COLI_TQ_PI + ((float)q+0.5f)*(2.f*COLI_TQ_PI)/(float)(1<<b);
}

/* deeper levels: half-width of the grid centered on pi/4. level>=2, s=2^(level-1);
 * ~3 std = 3/(2*sqrt(s)), capped at pi/4 so the window stays inside [0,pi/2]. */
static inline float coli_tq_hw(int level){
    float s = (float)(1u << (level-1));
    float w = 1.5f/sqrtf(s);
    float cap = 0.25f*COLI_TQ_PI;
    return w<cap ? w : cap;
}
static inline uint32_t coli_tq_enc2(float ang, int b, int level){
    float w = coli_tq_hw(level);
    float lo = 0.25f*COLI_TQ_PI - w;                       /* window [lo, lo+2w] */
    float u = (ang - lo) / (2.f*w);
    int m = (1<<b)-1;
    int q = (int)floorf(u*(float)(1<<b));
    if(q<0) q=0; if(q>m) q=m;
    return (uint32_t)q;
}
static inline float coli_tq_dec2(uint32_t q, int b, int level){
    float w = coli_tq_hw(level);
    float lo = 0.25f*COLI_TQ_PI - w;
    return lo + ((float)q+0.5f)*(2.f*w)/(float)(1<<b);
}

/* ---- bit-cursor packing (codes are <=6 bits, so a uint32 accumulator is ample) */
static inline void coli_tq_wbits(uint8_t *buf, int *pos, uint32_t val, int w){
    for(int k=0;k<w;k++){
        int bit=(*pos)+k;
        if((val>>k)&1u) buf[bit>>3] |= (uint8_t)(1u<<(bit&7));
    }
    *pos += w;
}
static inline uint32_t coli_tq_rbits(const uint8_t *buf, int *pos, int w){
    uint32_t v=0;
    for(int k=0;k<w;k++){
        int bit=(*pos)+k;
        if((buf[bit>>3]>>(bit&7))&1u) v |= (1u<<k);
    }
    *pos += w;
    return v;
}

/* Packed byte count for one row: (n/2) level-1 angles + (n/2 - 1) deeper angles. */
static inline int coli_tq_row_bytes(int n, int bits){
    long total = (long)(n/2)*coli_tq_b1(bits) + (long)(n/2 - 1)*coli_tq_b2(bits);
    return (int)((total + 7) / 8);
}

/* ---- encode / decode ----------------------------------------------------- */

/* Quantize one latent row. Returns the radius (row L2 norm) to store in the
 * per-row scale slot; packs the n-1 quantized angles into dst (dst must hold
 * coli_tq_row_bytes(n,bits) bytes). A zero / subnormal / non-finite row is inert:
 * radius 0, all-zero codes, and coli_tq_dequant_row reproduces the zero vector.
 * (norm NaN/Inf both fail the finite guard, so a poisoned row never reaches FWHT.) */
static inline float coli_tq_quant_row(const float *src, uint8_t *dst, int n, int bits){
    int rb = coli_tq_row_bytes(n,bits);
    memset(dst,0,(size_t)rb);
    if(n<2 || n>COLI_TQ_MAXN || (n&(n-1))) return 0.f;   /* n must be a power of two */

    double ss=0; for(int i=0;i<n;i++) ss += (double)src[i]*(double)src[i];
    float norm = (float)sqrt(ss);
    if(!(norm>1e-35f) || norm>3.4e38f) return 0.f;        /* zero / NaN / Inf -> inert */

    float r[COLI_TQ_MAXN];
    coli_tq_rotate(src, r, n, COLI_TQ_SEED);              /* r starts as the rotated row */

    int b1=coli_tq_b1(bits), b2=coli_tq_b2(bits), pos=0;
    /* Reduce radii pair-by-pair, one level at a time; write each level's angles.
     * r[j] is overwritten with the parent radius after its pair is consumed —
     * safe because index j (<= the pair it reads) is never read again this level. */
    for(int level=1, rlen=n; rlen>1; level++){
        int half=rlen/2, b=(level==1)?b1:b2;
        for(int j=0;j<half;j++){
            float a=r[2*j], c=r[2*j+1];
            float ang = atan2f(c,a);
            uint32_t code = (level==1) ? coli_tq_enc1(ang,b) : coli_tq_enc2(ang,b,level);
            coli_tq_wbits(dst,&pos,code,b);
            r[j] = hypotf(a,c);
        }
        rlen=half;
    }
    return norm;
}

/* Reconstruct one row from its packed angles + radius (the stored scale). Inverse
 * of coli_tq_quant_row: dequantize angles, rebuild the rotated vector top-down
 * through the polar tree (each split preserves the parent radius exactly), then
 * apply R^-1. radius<=0 or non-finite -> the zero vector. */
static inline void coli_tq_dequant_row(const uint8_t *src, float radius, float *dst, int n, int bits){
    if(n<2 || n>COLI_TQ_MAXN || (n&(n-1))){ memset(dst,0,(size_t)(n>0?n:0)*sizeof(float)); return; }  /* power-of-two only */
    if(!(radius>0.f) || radius>3.4e38f){ for(int i=0;i<n;i++) dst[i]=0.f; return; }

    int b1=coli_tq_b1(bits), b2=coli_tq_b2(bits), pos=0;
    /* Read angles in the SAME order they were written (level 1 first), dequantizing
     * into a per-level-offset flat array. Level l has n/2^l angles at offset
     * off[l] = n - n/2^(l-1); total n-1 entries. */
    float ang[COLI_TQ_MAXN];
    for(int level=1, len=n/2, off=0; len>=1; level++){
        int b=(level==1)?b1:b2;
        for(int j=0;j<len;j++){
            uint32_t code = coli_tq_rbits(src,&pos,b);
            ang[off+j] = (level==1) ? coli_tq_dec1(code,b) : coli_tq_dec2(code,b,level);
        }
        off += len;
        if(len==1) break;
        len >>= 1;
    }

    /* Rebuild top-down. cur holds r^(l) (clen radii); its angles are the level with
     * exactly clen entries, which starts at offset n-2*clen (level l, clen=n/2^l ->
     * off = n - n/2^(l-1) = n - 2*clen). Each split preserves the parent radius
     * (cos^2+sin^2=1), so ||cur|| stays == radius all the way down to y (clen==n). */
    float cur[COLI_TQ_MAXN], nxt[COLI_TQ_MAXN];
    cur[0]=radius;
    for(int clen=1; clen<n; clen*=2){
        int off = n - 2*clen;
        for(int j=0;j<clen;j++){
            float aa = ang[off+j];
            nxt[2*j]   = cur[j]*cosf(aa);
            nxt[2*j+1] = cur[j]*sinf(aa);
        }
        memcpy(cur,nxt,(size_t)(2*clen)*sizeof(float));
    }
    memcpy(dst,cur,(size_t)n*sizeof(float));
    coli_tq_unrotate(dst,n,COLI_TQ_SEED);
}

/* ---- rotated int4 codec (codec 1, the competitive 4-bit tier) ---------------
 * Same randomized-Hadamard rotation as PolarQuant, but the gaussianized coordinates are
 * quantized with a fixed 16-level Lloyd-Max codebook (data-oblivious: N(0,1)-optimal, scaled
 * per row by std = ||x||/sqrt(n)). Full reconstruction of the row, so it fixes BOTH the MLA
 * key and the value (QJL residual only corrects scores) — ~0.10 rel L2 at 4 bit vs polar's
 * ~0.15. Radius (row L2 norm) rides the per-row scale slot; 2 nibbles/byte, n even. */
static const float coli_q4_lev[16]={ -2.733f,-2.069f,-1.618f,-1.256f,-0.942f,-0.657f,-0.388f,-0.128f,
                                      0.128f, 0.388f, 0.657f, 0.942f, 1.256f, 1.618f, 2.069f, 2.733f };
static inline int coli_q4_row_bytes(int n){ return n/2; }
static inline int coli_q4_enc(float v){ int best=0; float bd=1e30f;
    for(int k=0;k<16;k++){ float d=fabsf(v-coli_q4_lev[k]); if(d<bd){bd=d;best=k;} } return best; }
static inline float coli_q4_quant_row(const float *src, uint8_t *dst, int n){
    int rb=coli_q4_row_bytes(n); memset(dst,0,(size_t)rb);
    if(n<2 || n>COLI_TQ_MAXN || (n&(n-1))) return 0.f;
    double ss=0; for(int i=0;i<n;i++) ss+=(double)src[i]*(double)src[i];
    float norm=(float)sqrt(ss);
    if(!(norm>1e-35f) || norm>3.4e38f) return 0.f;                 /* zero / NaN / Inf -> inert */
    float y[COLI_TQ_MAXN]; coli_tq_rotate(src,y,n,COLI_TQ_SEED);
    float invstd=sqrtf((float)n)/norm;                            /* y*invstd ~ N(0,1) */
    for(int i=0;i<n;i++){ int c=coli_q4_enc(y[i]*invstd); dst[i>>1]|=(uint8_t)(c<<((i&1)*4)); }
    return norm;
}
static inline void coli_q4_dequant_row(const uint8_t *src, float radius, float *dst, int n){
    if(n<2 || n>COLI_TQ_MAXN || (n&(n-1))){ memset(dst,0,(size_t)(n>0?n:0)*sizeof(float)); return; }
    if(!(radius>0.f) || radius>3.4e38f){ for(int i=0;i<n;i++) dst[i]=0.f; return; }
    float std=radius/sqrtf((float)n);
    for(int i=0;i<n;i++){ int c=(src[i>>1]>>((i&1)*4))&0xF; dst[i]=coli_q4_lev[c]*std; }
    coli_tq_unrotate(dst,n,COLI_TQ_SEED);
}

/* ---- codec dispatch: 0 = PolarQuant (paper-faithful, variable bits), 1 = rotated int4 ---- */
static inline int   coli_kvq_row_bytes(int n,int bits,int codec){ return codec? coli_q4_row_bytes(n) : coli_tq_row_bytes(n,bits); }
static inline float coli_kvq_quant_row(const float*s,uint8_t*d,int n,int bits,int codec){ return codec? coli_q4_quant_row(s,d,n) : coli_tq_quant_row(s,d,n,bits); }
static inline void  coli_kvq_dequant_row(const uint8_t*s,float sc,float*d,int n,int bits,int codec){ if(codec) coli_q4_dequant_row(s,sc,d,n); else coli_tq_dequant_row(s,sc,d,n,bits); }

#endif
