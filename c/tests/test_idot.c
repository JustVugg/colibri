/* Exactness test for the integer-dot kernels: dot_i8i8 and dot_i4i8 must return
 * EXACTLY the same value as a plain-C reference, whatever SIMD path was compiled
 * in (avx512-vnni / avx2 / neon / vsx / scalar). Integer arithmetic has no
 * rounding, so any mismatch is a kernel bug, not noise.
 *
 * Covers: odd sizes (scalar tail), sizes below one vector, the w=-128 edge
 * (sign-trick kernels must treat |−128| as 128 unsigned, not saturate to 127),
 * and random data at qrow_i8's contract (|x| <= 127, w full int8 range). */
#define main coli_glm_main_unused
#include "../glm.c"
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

int main(void){
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
    return 0;
}
