/* Exactness test for the grouped-int4 kernel (fmt=4, one f32 scale per `gs`
 * elements along I) against a plain-C reference that dequantizes and multiplies
 * in double.
 *
 * Why this test exists: matmul_i4_grouped is the REFERENCE the CUDA fmt=4 path
 * (#298) is expected to reproduce, and it had no test of its own. Debugging a
 * backend against an unverified oracle means two moving targets. Anyone porting
 * fmt=4 to a new backend can now diff against a kernel that is known exact here.
 *
 * Covers: I a clean multiple of gs, I with a partial last group (the `glen`
 * clamp), odd I (the nibble tail), gs larger than I (single group), and the
 * nibble edges 0 and 15 (which decode to -8 and +7 — an offset encoding, NOT
 * two's complement; getting this backwards is silent and looks like noise).
 *
 * FP note: the kernel sums each group in f32 (AVX2 accumulator + scalar tail)
 * while the reference sums in double, so we compare against a relative epsilon
 * rather than bit-exactly. The tolerance is tight enough that a wrong scale
 * index, a wrong group boundary or a swapped nibble cannot hide under it —
 * those are O(1) relative errors, not O(1e-6). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static uint32_t rng_state=0xC0FFEEu;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }
static float frand(void){ return (float)((int)(xr()%2001)-1000)/1000.0f; }

/* Reference: dequantize nibble -> (v-8)*scale[group], accumulate in double.
 * Deliberately the dumbest possible expression of the format. */
static void ref_grouped(double *y, double *mag, const float *x, const uint8_t *q4,
                        const float *scale, int S, int I, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; double a=0, m=0;
            for(int i=0;i<I;i++){
                uint8_t byte=w[i>>1];
                int nib=(i&1)?(int)(byte>>4):(int)(byte&0xF);
                double term=(double)xs[i] * (double)(nib-8) * (double)scl[i/gs];
                a += term; m += fabs(term);
            }
            y[(int64_t)s*O+o]=a;
            /* Sum of |terms|: the scale the f32 rounding error actually lives on.
             * Comparing against |result| instead would flag pure cancellation --
             * a dot product of signed terms can land near zero, and then a 1e-6
             * absolute error reads as a 1e-3 relative one. That is the accumulator's
             * precision, not a kernel defect. A wrong scale index or group boundary
             * shifts the result by a fraction OF THE TERMS, so it is caught here. */
            mag[(int64_t)s*O+o]=m;
        }
    }
}

static int check(const char *name, int S, int I, int O, int gs, int fill_edges){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    uint8_t *q4=malloc((size_t)O*rb);
    float *scale=malloc((size_t)O*ng*sizeof(float));
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y=malloc((size_t)S*O*sizeof(float));
    double *yr=malloc((size_t)S*O*sizeof(double));
    double *ym=malloc((size_t)S*O*sizeof(double));
    if(!q4||!scale||!x||!y||!yr||!ym){ fprintf(stderr,"%s: OOM\n",name); return 1; }

    for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(xr()&0xFF);
    if(fill_edges){
        /* nibble extremes: 0x0F -> +7, 0x00 -> -8. A two's-complement misread
         * turns 15 into -1 instead of +7 and the error is data-dependent noise. */
        for(size_t i=0;i<(size_t)O*rb && i<64;i++) q4[i]=(i&1)?0x00:0xFF;
    }
    /* scales span a few orders of magnitude: a wrong group index shows up big */
    for(int i=0;i<O*ng;i++) scale[i]=(0.001f+(float)(xr()%1000)/1000.0f)*((xr()&1)?1.f:-1.f);
    for(int i=0;i<S*I;i++) x[i]=frand();

    matmul_i4_grouped(y,x,q4,scale,S,I,O,gs);
    ref_grouped(yr,ym,x,q4,scale,S,I,O,gs);

    int bad=0; double worst=0;
    for(int i=0;i<S*O;i++){
        double d=fabs((double)y[i]-yr[i]);
        double rel = ym[i]>1e-30 ? d/ym[i] : d;   /* error relative to the summed magnitude */
        if(rel>worst) worst=rel;
        if(rel>1e-6){
            if(bad<3) fprintf(stderr,"%s: [%d] got %.9g want %.9g (|terms| %.3g, rel %.3g)\n",
                              name,i,(double)y[i],yr[i],ym[i],rel);
            bad++;
        }
    }
    free(q4);free(scale);free(x);free(y);free(yr);free(ym);
    if(bad){ fprintf(stderr,"%s: FAIL (%d/%d mismatched, worst rel %.3g)\n",name,bad,S*O,worst); return 1; }
    printf("  %-42s ok (S=%d I=%d O=%d gs=%d ng=%d, worst rel %.2g)\n",name,S,I,O,gs,ng,worst);
    return 0;
}

/* matmul_i4_grouped_pair (fused gate+up, #298) reads x once instead of twice.
 * Checked two ways, because "identical" is only true where it can be:
 *
 *  - Correctness, always: both outputs must match the double reference within
 *    the same magnitude-relative epsilon as the unfused kernel.
 *  - Bit-exactness, only when I % gs == 0: then every group is covered by the
 *    AVX2 body, whose accumulation order is identical to the unfused kernel, so
 *    the results agree to the last bit. This is the shape the real g64
 *    checkpoints have (I = 2048 / 6144, gs = 64), i.e. the production path.
 *
 * With a PARTIAL last group the group tail falls to scalar code, and the
 * compiler is free to contract/reassociate the fused body differently from the
 * single-matrix one. The results then differ by ~1e-7 -- rounding, not logic
 * (which of gate/up "differs" is arbitrary, the tell that it is FP luck).
 * Demanding bit-exactness there would report a compiler artifact as a bug. */
#ifdef COLI_HAVE_GROUPED_PAIR
static int check_pair(const char *name, int S, int I, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    uint8_t *qg=malloc((size_t)O*rb), *qu=malloc((size_t)O*rb);
    float *sg=malloc((size_t)O*ng*sizeof(float)), *su=malloc((size_t)O*ng*sizeof(float));
    float *x=malloc((size_t)S*I*sizeof(float));
    float *yg=malloc((size_t)S*O*sizeof(float)), *yu=malloc((size_t)S*O*sizeof(float));
    float *rg=malloc((size_t)S*O*sizeof(float)), *ru=malloc((size_t)S*O*sizeof(float));
    double *dg=malloc((size_t)S*O*sizeof(double)), *du=malloc((size_t)S*O*sizeof(double));
    double *mg=malloc((size_t)S*O*sizeof(double)), *mu=malloc((size_t)S*O*sizeof(double));
    if(!qg||!qu||!sg||!su||!x||!yg||!yu||!rg||!ru||!dg||!du||!mg||!mu){ fprintf(stderr,"%s: OOM\n",name); return 1; }

    for(size_t i=0;i<(size_t)O*rb;i++){ qg[i]=(uint8_t)(xr()&0xFF); qu[i]=(uint8_t)(xr()&0xFF); }
    for(int i=0;i<O*ng;i++){ sg[i]=frand(); su[i]=frand(); }
    for(int i=0;i<S*I;i++) x[i]=frand();

    matmul_i4_grouped_pair(yg,yu,x,qg,sg,qu,su,S,I,O,gs);
    matmul_i4_grouped(rg,x,qg,sg,S,I,O,gs);
    matmul_i4_grouped(ru,x,qu,su,S,I,O,gs);
    ref_grouped(dg,mg,x,qg,sg,S,I,O,gs);
    ref_grouped(du,mu,x,qu,su,S,I,O,gs);

    int bad=0, exact=1; double worst=0;
    for(int i=0;i<S*O;i++){
        double eg = mg[i]>1e-30 ? fabs((double)yg[i]-dg[i])/mg[i] : fabs((double)yg[i]-dg[i]);
        double eu = mu[i]>1e-30 ? fabs((double)yu[i]-du[i])/mu[i] : fabs((double)yu[i]-du[i]);
        if(eg>worst) worst=eg;
        if(eu>worst) worst=eu;
        if(eg>1e-6||eu>1e-6){
            if(bad<3) fprintf(stderr,"%s: [%d] gate %.9g/%.9g up %.9g/%.9g (rel %.3g/%.3g)\n",
                              name,i,(double)yg[i],dg[i],(double)yu[i],du[i],eg,eu);
            bad++;
        }
        if(yg[i]!=rg[i]||yu[i]!=ru[i]) exact=0;
    }
    /* Aligned shapes run entirely through the AVX2 body: same order as unfused,
     * so bit-exactness is a real invariant there and worth asserting. */
    if(I%gs==0 && !exact){
        fprintf(stderr,"%s: FAIL fused != unfused bitwise on an ALIGNED shape "
                       "(no scalar tail runs here; the orders must match)\n",name);
        bad++;
    }
    free(qg);free(qu);free(sg);free(su);free(x);free(yg);free(yu);free(rg);free(ru);
    free(dg);free(du);free(mg);free(mu);
    if(bad){ fprintf(stderr,"%s: FAIL (%d mismatched)\n",name,bad); return 1; }
    printf("  %-42s ok (S=%d I=%d O=%d gs=%d, worst rel %.2g%s)\n",name,S,I,O,gs,worst,
           I%gs==0?", bit-exact vs unfused":"");
    return 0;
}
#endif

int main(void){
    int fail=0;
    printf("test_i4_grouped: matmul_i4_grouped vs plain-C dequant reference\n");

    /* the shape the g64 checkpoints actually use */
    fail|=check("gs=64, I multiple of gs",            2, 512, 8, 64, 0);
    fail|=check("gs=64, single row single token",     1, 128, 1, 64, 0);
    fail|=check("gs=64, nibble edges (0x00/0xFF)",    1, 256, 4, 64, 1);

    /* partial last group: glen clamp, the classic off-by-one */
    fail|=check("gs=64, partial last group (I=200)",  2, 200, 4, 64, 0);
    fail|=check("gs=64, I just over a group (I=65)",  1,  65, 3, 64, 0);
    fail|=check("gs=64, I one under a group (I=63)",  1,  63, 3, 64, 0);

    /* odd I: the scalar nibble tail (i+1 == I) */
    fail|=check("gs=64, odd I (I=201)",               2, 201, 4, 64, 0);
    fail|=check("gs=16, odd I (I=33)",                1,  33, 2, 16, 0);

    /* gs > I: everything in one group */
    fail|=check("gs=128 > I=64 (single group)",       1,  64, 4, 128, 0);

    /* the other documented group size */
    fail|=check("gs=128, I multiple of gs",           2, 512, 4, 128, 0);

    /* batch: S>1 exercises the per-s inner loop against a shared scale row */
    fail|=check("gs=64, batch S=8",                   8, 320, 6, 64, 0);

#ifdef COLI_HAVE_GROUPED_PAIR
    printf("test_i4_grouped: matmul_i4_grouped_pair (fused gate+up) vs two separate calls\n");
    fail|=check_pair("pair: gs=64, I multiple of gs",  2, 512, 8, 64);
    fail|=check_pair("pair: gs=64, partial last group",2, 200, 4, 64);
    fail|=check_pair("pair: gs=64, odd I (I=201)",     2, 201, 4, 64);
    fail|=check_pair("pair: gs=64, decode S=1",        1, 320, 6, 64);
    fail|=check_pair("pair: gs=128, I=512",            2, 512, 4, 128);
#endif

    if(fail){ printf("test_i4_grouped: FAIL\n"); return 1; }
    printf("test_i4_grouped: ok\n");
    return 0;
}
