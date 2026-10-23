/* Microbenchmark: per-row vs multi-row K1b grouped planar IDOT (quant.h).
 * NOT a unit test -- test_int_kernel_exact.c proves correctness/bit-equality.
 *
 * This measures the multi-row claim behind the fmt=4 tile work: the per-row
 * kernel re-loads and re-masks every weight block once per activation row, so
 * batched calls (prefill batch-union rows, the serve mux's decode batch) pay
 * S times the weight traffic. The 1x4 tile (and AMX on Sapphire Rapids) pays
 * it once per tile. The OLD kernel below is a verbatim copy of the pre-tile
 * body; the NEW one is the real dispatcher, so on an AMX host this also
 * benches the tile-unit path (AMX_S_MIN gates it; force with AMX_S_MIN=2).
 *
 * Run:  make tests/bench_i4p_gidot ARCH=native && ./tests/bench_i4p_gidot
 *       (not in TEST_BINS -- not a gate) */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include <stdint.h>
#include <string.h>

static uint32_t rs=0x2545F491u;
static uint32_t xr(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }

/* ---- OLD kernel: verbatim copy of the pre-tile per-row body ---- */
static void gidot_old(float *y, const int8_t *xq, const float *sx,
                      const int32_t *xsg, const uint8_t *q4,
                      const float *scale, int S, int I, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs, bpg=gs/64;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const int8_t *xr2=xq+(int64_t)s*I;
            const int32_t *xg=xsg+(int64_t)s*ng;
            float a=0; int g=0;
            for(; (g+1)*gs<=I; g++){
                int32_t d=0;
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    const uint8_t *blk=w+(base>>1);
                    const int8_t *xb=xr2+base;
#if defined(coli_dpbusd256)
                    const __m256i m4=_mm256_set1_epi8(0x0F);
                    __m256i bb=_mm256_loadu_si256((const __m256i*)blk);
                    __m256i acc=_mm256_setzero_si256();
                    acc=coli_dpbusd256(acc,_mm256_and_si256(bb,m4),
                                       _mm256_loadu_si256((const __m256i*)xb));
                    acc=coli_dpbusd256(acc,_mm256_and_si256(_mm256_srli_epi16(bb,4),m4),
                                       _mm256_loadu_si256((const __m256i*)(xb+32)));
                    d+=hsum256_i32(acc);
#elif defined(__AVX2__)
                    const __m256i m4=_mm256_set1_epi8(0x0F);
                    const __m256i ones=_mm256_set1_epi16(1);
                    __m256i bb=_mm256_loadu_si256((const __m256i*)blk);
                    __m256i p0=_mm256_maddubs_epi16(_mm256_and_si256(bb,m4),
                                                    _mm256_loadu_si256((const __m256i*)xb));
                    __m256i p1=_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(bb,4),m4),
                                                    _mm256_loadu_si256((const __m256i*)(xb+32)));
                    __m256i acc=_mm256_add_epi32(_mm256_madd_epi16(p0,ones),
                                                 _mm256_madd_epi16(p1,ones));
                    d+=hsum256_i32(acc);
#else
                    for(int k=0;k<32;k++){
                        d+=(int32_t)(blk[k]&0xF)*xb[k];
                        d+=(int32_t)(blk[k]>>4)*xb[k+32];
                    }
#endif
                }
                a=fmaf((float)(d-8*xg[g]),scl[g],a);
            }
            if(g*gs<I){
                int32_t d=0;
                for(int i=g*gs;i<I;i++){
                    uint8_t byte=w[i>>1];
                    d+=(int32_t)((i&1)?(byte>>4):(byte&0xF))*xr2[i];
                }
                a=fmaf((float)(d-8*xg[g]),scl[g],a);
            }
            y[(int64_t)s*O+o]=a*sx[s];
        }
    }
}

#define O_DIM 2048
#define I_DIM 2048
#define GS 64
#define REPS 400
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }

int main(void){
    int rb=(I_DIM+1)/2, ng=(I_DIM+GS-1)/GS;
    int Ss[]={1,4,8,16};
    uint8_t *q4=malloc((size_t)O_DIM*rb);
    float *sc=malloc((size_t)O_DIM*ng*sizeof(float));
    for(size_t i=0;i<(size_t)O_DIM*rb;i++) q4[i]=(uint8_t)(xr()&0xFF);
    planarize_i4(q4,O_DIM,I_DIM);
    for(size_t i=0;i<(size_t)O_DIM*ng;i++) sc[i]=0.0005f+(xr()%911)*1e-6f;

    int Smax=16;
    int8_t *xq=malloc((size_t)Smax*I_DIM);
    float *sx=malloc(Smax*sizeof(float));
    int32_t *xsg=malloc((size_t)Smax*ng*sizeof(int32_t));
    for(size_t i=0;i<(size_t)Smax*I_DIM;i++){ int v=(int)(xr()%255)-127; xq[i]=(int8_t)v; }
    for(int s=0;s<Smax;s++){ sx[s]=0.01f+(xr()%89)*1e-4f;
        for(int g=0;g<ng;g++){ int32_t a=0;
            for(int i=g*GS;i<(g+1)*GS;i++) a+=xq[(size_t)s*I_DIM+i];
            xsg[(size_t)s*ng+g]=a; } }
    float *ya=malloc((size_t)Smax*O_DIM*sizeof(float));
    float *yb=malloc((size_t)Smax*O_DIM*sizeof(float));
    static double t[REPS];

    printf("bench_i4p_gidot: O=%d I=%d gs=%d, %s, weights %.1f MB\n",
           O_DIM,I_DIM,GS,IDOT_KERNEL,(double)O_DIM*rb/1e6);
    for(size_t k=0;k<sizeof(Ss)/sizeof(*Ss);k++){
        int S=Ss[k];
        gidot_old(ya,xq,sx,xsg,q4,sc,S,I_DIM,O_DIM,GS);
        matmul_i4p_grouped_idot(yb,xq,sx,xsg,q4,sc,S,I_DIM,O_DIM,GS);
        if(memcmp(ya,yb,(size_t)S*O_DIM*sizeof(float))){ fprintf(stderr,"MISMATCH S=%d\n",S); return 1; }
        double med[2];
        for(int which=0;which<2;which++){
            for(int wu=0;wu<20;wu++) (which?matmul_i4p_grouped_idot:gidot_old)(yb,xq,sx,xsg,q4,sc,S,I_DIM,O_DIM,GS);
            for(int r=0;r<REPS;r++){
                double t0=now_s();
                (which?matmul_i4p_grouped_idot:gidot_old)(yb,xq,sx,xsg,q4,sc,S,I_DIM,O_DIM,GS);
                t[r]=now_s()-t0;
            }
            qsort(t,REPS,sizeof(double),cmp_d);
            med[which]=t[REPS/2];
        }
        double wb=(double)O_DIM*rb;   /* weight bytes touched once per call */
        printf("S=%2d  old %8.1f us (%6.2f GB/s-w)   new %8.1f us (%6.2f GB/s-w)   new/old %.2fx\n",
               S, med[0]*1e6, wb/med[0]/1e9, med[1]*1e6, wb/med[1]/1e9, med[0]/med[1]);
    }
    return 0;
}
