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

int main(void){
    static const int sizes[]={1,2,15,16,17,31,32,33,63,64,65,100,127,128,1408,4096,4097};
    static int8_t w[8192], w1[8192], x[8192], x1[8192];
    static uint8_t w4[4096], w41[4096];
    int have_i8mm=i8mm_enabled();
    for(unsigned t=0;t<sizeof(sizes)/sizeof(sizes[0]);t++){
        int I=sizes[t];
        for(int rep=0;rep<64;rep++){
            for(int i=0;i<I;i++) x[i]=(int8_t)((int)(xr()%255)-127);      /* [-127,127]: contratto di qrow_i8 */
            for(int i=0;i<I;i++) w[i]=(int8_t)((int)(xr()%256)-128);      /* [-128,127]: range pieno */
            for(int i=0;i<I;i++){ x1[i]=(int8_t)((int)(xr()%255)-127); w1[i]=(int8_t)((int)(xr()%256)-128); }
            if(rep==0) for(int i=0;i<I;i++) w[i]=-128;                    /* caso limite del trucco del segno */
            if(rep==1) for(int i=0;i<I;i++){ w[i]=127; x[i]=(int8_t)(i&1?-127:127); }
            for(int i=0;i<(I+1)/2;i++){ w4[i]=(uint8_t)(xr()&0xFF); w41[i]=(uint8_t)(xr()&0xFF); }
            int32_t got=dot_i8i8(w,x,I), want=ref_i8i8(w,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i8i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
            got=dot_i4i8(w4,x,I); want=ref_i4i8(w4,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i4i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
#if COLI_HAVE_I8MM
            if(have_i8mm){
                int32_t d[4], r[4]={ref_i8i8(w,x,I),ref_i8i8(w,x1,I),ref_i8i8(w1,x,I),ref_i8i8(w1,x1,I)};
                dot_i8i8_2x2_i8mm_entry(d,w,w1,x,x1,I);
                for(int j=0;j<4;j++) if(d[j]!=r[j]){
                    fprintf(stderr,"FAIL i8mm int8 I=%d rep=%d lane=%d: %d != %d\n",I,rep,j,d[j],r[j]); return 1; }
                r[0]=ref_i4i8(w4,x,I); r[1]=ref_i4i8(w4,x1,I);
                r[2]=ref_i4i8(w41,x,I); r[3]=ref_i4i8(w41,x1,I);
                dot_i4i8_2x2_i8mm_entry(d,w4,w41,x,x1,I);
                for(int j=0;j<4;j++) if(d[j]!=r[j]){
                    fprintf(stderr,"FAIL i8mm int4 I=%d rep=%d lane=%d: %d != %d\n",I,rep,j,d[j],r[j]); return 1; }
            }
#endif
        }
    }

#if COLI_HAVE_I8MM
    if(have_i8mm){
        static int8_t qm8[5*8192], xm[3*8192]; static uint8_t qm4[5*4096];
        float sc[5]={0.25f,0.5f,1.f,1.5f,2.f}, sx[3]={0.125f,0.75f,1.25f};
        float ref8[15],got8[15],ref4[15],got4[15];
        static const int msizes[]={8,15,32,127,1408,4097,6144};
        for(unsigned t=0;t<sizeof(msizes)/sizeof(msizes[0]);t++){
            int I=msizes[t],rb=(I+1)/2;
            for(int i=0;i<3*I;i++) xm[i]=(int8_t)((int)(xr()%255)-127);
            for(int i=0;i<5*I;i++) qm8[i]=(int8_t)((int)(xr()%256)-128);
            for(int i=0;i<5*rb;i++) qm4[i]=(uint8_t)xr();
            for(int S=1;S<=3;S++) for(int O=2;O<=5;O++){
                g_i8mm=0; matmul_q_idot(ref8,xm,sx,qm8,sc,S,I,O);
                g_i8mm=1; matmul_q_idot(got8,xm,sx,qm8,sc,S,I,O);
                g_i8mm=0; matmul_i4_idot(ref4,xm,sx,qm4,sc,S,I,O);
                g_i8mm=1; matmul_i4_idot(got4,xm,sx,qm4,sc,S,I,O);
                for(int i=0;i<S*O;i++) if(got8[i]!=ref8[i] || got4[i]!=ref4[i]){
                    fprintf(stderr,"FAIL i8mm matmul I=%d S=%d O=%d at %d\n",I,S,O,i); return 1; }
            }
        }
    }
#endif
    printf("idot kernel exactness (%s%s): ok\n", IDOT_KERNEL,have_i8mm?"+i8mm":"");
    return 0;
}
