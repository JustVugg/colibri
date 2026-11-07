/* Exactness test for the integer-dot kernels: dot_i8i8 and dot_i4i8 must return
 * EXACTLY the same value as a plain-C reference, whatever SIMD path was compiled
 * in (avx512-vnni / avx2 / ssse3 / neon / vsx / scalar). Integer arithmetic has no
 * rounding, so any mismatch is a kernel bug, not noise.
 *
 * Covers: odd sizes (scalar tail), sizes below one vector, the w=-128 edge
 * (sign-trick kernels must treat |−128| as 128 unsigned, not saturate to 127),
 * and random data at qrow_i8's contract (|x| <= 127, w full int8 range). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static uint32_t rng_state=0x12345678u;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }

static int32_t ref_i8i8(const int8_t *w, const int8_t *x, int I){
    int64_t s=0; for(int i=0;i<I;i++) s+=(int32_t)w[i]*x[i]; return (int32_t)s;
}
static int32_t ref_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int64_t s=0;
    for(int i=0;i<I;i++){ uint8_t b=w4[i>>1]; int v=(i&1)?((int)(b>>4)-8):((int)(b&0xF)-8); s+=v*x[i]; }
    return (int32_t)s;
}

/* Driver-level exactness: matmul_qt_ex on the IDOT path (allow_idot=1) must match
 * a plain-C reference bit-for-bit. This exercises the SMMLA 2x2-tile drivers on the
 * ARCH=native (i8mm) build and the SDOT drivers on the default build. The integer
 * dot is exact and qrow_i8 is the shared quantizer, so any float bit mismatch is a
 * driver bug (lane map, tiling, tail, or scale order), not rounding. */
static void fill_qt(QT *w, int fmt, int O, int I){
    memset(w,0,sizeof *w);
    w->fmt=fmt; w->O=O; w->I=I;
    w->s=malloc((size_t)O*sizeof(float));
    for(int o=0;o<O;o++) w->s[o]=0.001f+(float)(xr()%1000)*1e-6f;
    if(fmt==1){
        w->q8=malloc((size_t)O*I);
        for(int64_t i=0;i<(int64_t)O*I;i++) w->q8[i]=(int8_t)((int)(xr()%256)-128);
    }else{
        size_t nb=(size_t)O*((I+1)/2);
        w->q4=malloc(nb);
        for(size_t i=0;i<nb;i++) w->q4[i]=(uint8_t)(xr()&0xFF);
    }
}
static int check_driver(int fmt,int O,int I,int S){
    QT w; fill_qt(&w,fmt,O,I); int rb=(I+1)/2;
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y=malloc((size_t)S*O*sizeof(float));
    float *yref=malloc((size_t)S*O*sizeof(float));
    int8_t *xqr=malloc((size_t)S*I);
    float *sxr=malloc((size_t)S*sizeof(float));
    for(int64_t i=0;i<(int64_t)S*I;i++) x[i]=((float)(xr()%4001)-2000.f)/500.f;
    matmul_qt_ex(y,x,&w,S,1);
    for(int s=0;s<S;s++) sxr[s]=qrow_i8(x+(int64_t)s*I, xqr+(int64_t)s*I, I);
    for(int o=0;o<O;o++) for(int s=0;s<S;s++){
        int32_t d=fmt==1 ? ref_i8i8(w.q8+(int64_t)o*I, xqr+(int64_t)s*I, I)
                         : ref_i4i8(w.q4+(int64_t)o*rb, xqr+(int64_t)s*I, I);
        yref[(int64_t)s*O+o]=(float)d*w.s[o]*sxr[s];
    }
    int rc=0;
    for(int64_t i=0;i<(int64_t)S*O;i++)
        if(memcmp(&y[i],&yref[i],sizeof(float))!=0){
            fprintf(stderr,"FAIL driver fmt=%d O=%d I=%d S=%d idx=%lld: %.9g != %.9g\n",
                    fmt,O,I,S,(long long)i,(double)y[i],(double)yref[i]); rc=1; break;
        }
    free(w.s); free(w.q8); free(w.q4); free(x); free(y); free(yref); free(xqr); free(sxr);
    return rc;
}

/* K1b's gate (planar_on, qt_planarize): an x86 CPU build with SSSE3 planarizes a gs64 fmt=4
 * tensor under IDOT_GS=1, a pre-AVX2 one (Sandy/Ivy Bridge, test_idot_ssse3) as well, and
 * matmul_qt then gives the pure-C reference's bits: the integer dot of each group, fmaf with
 * its scale, times the row's scale. fmt=2 is planarized only where matmul_i4p's f32 arm is,
 * an AVX2 build that is not AVX-512. */
static int check_k1b_gate(void){
    const int O=37, I=320, S=3, gs=64, ng=I/gs, rb=I/2;
#if (defined(__x86_64__)||defined(_M_X64)) && defined(__SSSE3__) && \
    !defined(COLI_CUDA) && !defined(COLI_METAL) && !defined(COLI_VULKAN)
    const char *pe=getenv("PLANAR"), *xe=getenv("XEXP");
    const int want4=!(pe&&*pe=='0') && !(xe&&*xe=='1');
#else
    const int want4=0;
#endif
#if defined(__AVX2__) && !(defined(__AVX512F__)&&defined(__AVX512BW__))
    const int want2=want4;
#else
    const int want2=0;
#endif
    QT w; memset(&w,0,sizeof w);
    w.fmt=4; w.O=O; w.I=I; w.gs=gs;
    w.q4=malloc((size_t)O*rb); w.s=malloc((size_t)O*ng*sizeof(float));
    uint8_t *pairs=malloc((size_t)O*rb);
    for(size_t i=0;i<(size_t)O*rb;i++) pairs[i]=w.q4[i]=(uint8_t)(xr()&0xFF);
    for(int i=0;i<O*ng;i++) w.s[i]=0.0005f+(float)(xr()%911)*1e-6f;
    qt_planarize(&w);
    QT w2; fill_qt(&w2,2,O,I);
    qt_planarize(&w2);
    int rc=0;
    if(w.planar!=want4 || w2.planar!=want2){
        fprintf(stderr,"FAIL K1b gate: fmt=4 gs64 planar %d (want %d), fmt=2 planar %d (want %d)\n",
                w.planar,want4,w2.planar,want2);
        rc=1;
    }
    if(!rc && w.planar){
        float *x=malloc((size_t)S*I*sizeof(float)), *y=malloc((size_t)S*O*sizeof(float)), *yref=malloc((size_t)S*O*sizeof(float));
        int8_t *xq=malloc((size_t)S*I); float *sx=malloc((size_t)S*sizeof(float));
        for(int64_t i=0;i<(int64_t)S*I;i++) x[i]=((float)(xr()%4001)-2000.f)/500.f;
        matmul_qt_ex(y,x,&w,S,1);
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        for(int o=0;o<O;o++) for(int s=0;s<S;s++){
            float a=0;
            for(int g=0;g<ng;g++){
                int32_t d=0, xs=0;
                for(int i=g*gs;i<(g+1)*gs;i++){
                    uint8_t b=pairs[(int64_t)o*rb+(i>>1)];
                    d+=(int32_t)((i&1)?(b>>4):(b&0xF))*xq[(int64_t)s*I+i]; xs+=xq[(int64_t)s*I+i];
                }
                a=fmaf((float)(d-8*xs),w.s[(int64_t)o*ng+g],a);
            }
            yref[(int64_t)s*O+o]=a*sx[s];
        }
        for(int64_t i=0;i<(int64_t)S*O;i++)
            if(memcmp(&y[i],&yref[i],sizeof(float))!=0){
                fprintf(stderr,"FAIL K1b planar fmt=4 gs64 idx=%lld: %.9g != %.9g\n",(long long)i,(double)y[i],(double)yref[i]);
                rc=1; break;
            }
        free(x); free(y); free(yref); free(xq); free(sx);
    }
    if(!rc) printf("K1b gate (%s): fmt=4 gs64 %s, fmt=2 %s\n", IDOT_KERNEL,
                   w.planar?"planar, the reference's bits":"pairs", w2.planar?"planar":"pairs");
    free(w.q4); free(w.s); free(pairs); free(w2.s); free(w2.q8); free(w2.q4);
    return rc;
}

int main(void){
    setenv("IDOT_GS","1",1);   /* K1b's opt-in, before the first matmul reads it (check_k1b_gate) */
    static const int sizes[]={1,2,15,16,17,31,32,33,63,64,65,100,127,128,1408,4096,4097};
    static int8_t w[8192], x[8192]; static uint8_t w4[4096];
    for(unsigned t=0;t<sizeof(sizes)/sizeof(sizes[0]);t++){
        int I=sizes[t];
        for(int rep=0;rep<64;rep++){
            for(int i=0;i<I;i++) x[i]=(int8_t)((int)(xr()%255)-127);      /* [-127,127]: contratto di qrow_i8 */
            for(int i=0;i<I;i++) w[i]=(int8_t)((int)(xr()%256)-128);      /* [-128,127]: range pieno */
            if(rep==0) for(int i=0;i<I;i++) w[i]=-128;                    /* caso limite del trucco del segno */
            if(rep==1) for(int i=0;i<I;i++){ w[i]=127; x[i]=(int8_t)(i&1?-127:127); }
            for(int i=0;i<(I+1)/2;i++) w4[i]=(uint8_t)(xr()&0xFF);
            int32_t got=dot_i8i8(w,x,I), want=ref_i8i8(w,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i8i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
            got=dot_i4i8(w4,x,I); want=ref_i4i8(w4,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i4i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
        }
    }
    printf("idot kernel exactness (%s): ok\n", IDOT_KERNEL);

    static const int Os[]={1,2,3,64,65};
    static const int Is[]={16,17,100,1408};
    static const int Ss[]={2,3,4,5,8};
    for(int rep=0;rep<4;rep++)
     for(unsigned a=0;a<sizeof Os/sizeof Os[0];a++)
      for(unsigned b=0;b<sizeof Is/sizeof Is[0];b++)
       for(unsigned c=0;c<sizeof Ss/sizeof Ss[0];c++)
        for(int fmt=1;fmt<=2;fmt++)
         if(check_driver(fmt,Os[a],Is[b],Ss[c])) return 1;
    printf("idot driver exactness (%s): ok\n", IDOT_KERNEL);
    return check_k1b_gate();
}
