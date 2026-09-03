/* qt_addrow()/qt_matvec_rows() (colibri.c) -- per-row absorb helpers used only by the
 * kv_b MLA-absorption CPU path (l->kv_b, see attention_rows/decode). FIX ROUND 2, engine
 * defect (clean-room conformance trial finding, mutation-proven): both functions dispatch
 * fmt 0/4/5 explicitly, then fall through assuming a PER-ROW scale (t->s[row]) followed by
 * fmt=1/2/3 (qt_addrow) or fmt=0/1/2/3/4/5 (qt_matvec_rows, via an if/else-if chain ending
 * in a bare `else`) -- nothing stopped fmt=6 (E8/IQ3, t->s is a FIXED 4-byte tag, not O
 * floats) from reaching that fall-through, where it silently misreads the real E8/IQ3
 * lattice bytes as int2-packed data (same bug SHAPE as #298's CUDA absorb-kernel fix, which
 * is why this file's own fmt=4/5 branches exist -- fmt=6 was simply missed). fmt=6 has no
 * decoder in either function and still refuses loudly (exit(1), naming the function and the
 * fmt), matching qt_resolve_fmt's own "refuse rather than misread" discipline.
 *
 * fmt=8 (fp8-e4m3-b128) USED to hit this same fall-through and SIGSEGV (t->s[row] overread,
 * then a NULL t->q4 dereference -- see git history for the pre-fix account) but now has its
 * own explicit branch in both functions (the absorb-decode commit this file accompanies):
 * t->s holds per-128x128-block scales, t->q8 holds raw e4m3 bytes (t->q4 stays NULL), decoded
 * through the same e4m3 LUT / block-scale geometry (quant.h's e4m3_decode/fp8_nblk/FP8_BLOCK)
 * that matmul_fp8 already exercises on the dense/expert path. This file now (1) proves fmt=8
 * produces correct, tolerance-bounded output through BOTH functions -- an exact-dequant check
 * against an independent per-element reference (qt_dequant_row_ref's new fmt==8 branch,
 * disjoint code shape from qt_addrow/qt_matvec_rows' block-batched loops) plus a direct parity
 * check against matmul_fp8 (the proven non-absorb fp8 reference) on a kv_b-shaped tensor; (2)
 * proves the refusal still fires for fmt=6 through both functions (fork+pipe+waitpid, this
 * suite's established house pattern for exit(1)-terminated paths -- see
 * tests/test_fp8_load.c's expect_refuse/expect_stamp_refuse); (3) proves every other format
 * both functions handle (0/1/2/3/4/5) produces byte-identical results against the same
 * independent reference dequantizer, so a real regression in either the guard's placement or
 * the untouched per-fmt math would show up here, not just tautologically re-run the same code.
 * Reachability note (not a scope excuse, just context): both functions serve ONLY the kv_b
 * absorb path (attention_rows/decode call sites), and the repo's own repack tool mints
 * exactly the container these arms decode -- tools/repack_fp8_passthrough.py emits kv_b_proj
 * (kind "kvb", in its RESIDENT_KINDS/STAMPABLE_KINDS sets) as byte-preserved fmt=8 with a
 * stamped scale sidecar, and attention_rows' `int absorb = kvs || ...` makes absorb
 * unbypassable on the batched serving path. These arms are the decode support that
 * container needs. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <math.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static uint64_t rng = 0xA11CE5EEDF00Dull;
static uint8_t rndbyte(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint8_t)(rng & 0xFF); }
static float rndsmallf(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((int64_t)(rng & 0xFFF) - 0x800) / (float)0x800; }   /* [-1,1)-ish, small magnitude */
/* fmt=8 fixtures must avoid the two NaN byte patterns (0x7F/0xFF, see quant.h's E4M3_LUT
 * comment) -- a NaN weight is a real, policy-accepted outcome in production, but it would
 * make this file's relative-error comparisons meaningless (NaN != NaN), same reasoning
 * tests/test_fp8_load.c's rndbyte_nonan documents. */
static uint8_t rndbyte_nonan(void){
    for(;;){ uint8_t b=rndbyte(); if(b!=0x7F && b!=0xFF) return b; }
}

/* ---- independent reference dequantizer: W[row,:] as floats, one element per loop
 * iteration -- deliberately NOT the same code shape as qt_addrow/qt_matvec_rows (those
 * unroll pairs for fmt=2/3, split low/high planes for fmt=5) so this genuinely
 * cross-checks the production math, not just the production code running twice. ---- */
static void qt_dequant_row_ref_fmt8(const QT *t, int row, float *out);
static void qt_dequant_row_ref(const QT *t, int row, float *out){
    int I=t->I;
    if(t->fmt==8){ qt_dequant_row_ref_fmt8(t,row,out); return; }
    if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) out[i]=w[i]; return; }
    if(t->fmt==4){
        const uint8_t *w=t->q4+(int64_t)row*((I+1)/2); int gs=t->gs, ng=(I+gs-1)/gs;
        const float *scl=t->s+(int64_t)row*ng;
        for(int i=0;i<I;i++){ uint8_t b=w[i>>1]; int nib=(i&1)?(b>>4):(b&0xF);
            out[i]=((int)nib-8)*scl[i/gs]; }
        return; }
    if(t->fmt==5){
        const uint8_t *w=t->q4+(int64_t)row*i3_rowbytes(I);
        const float *sr=t->s+(int64_t)row*i3_groups(I);
        for(int i=0;i<I;i++){ int64_t g=i/I3_GROUP; const uint8_t *lo=w+g*I3_GBYTES, *hi=lo+16;
            int k=i-(int)(g*I3_GROUP);
            unsigned u=((lo[k>>2]>>((k&3)*2))&3)|(((hi[k>>3]>>(k&7))&1)<<2);
            out[i]=((int)u-4)*sr[g]; }
        return; }
    float s=t->s[row];
    if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; for(int i=0;i<I;i++) out[i]=(float)w[i]*s; return; }
    if(t->fmt==2){
        const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
        for(int i=0;i<I;i++){ uint8_t b=w[i>>1]; int nib=(i&1)?(b>>4):(b&0xF); out[i]=((int)nib-8)*s; }
        return; }
    /* fmt==3 */
    { const uint8_t *w=t->q4+(int64_t)row*((I+3)/4);
      for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; int v=(b>>((i&3)*2))&3; out[i]=((int)v-2)*s; } }
}
/* fmt=8 (fp8-e4m3-b128) independent reference: a flat per-ELEMENT loop, deliberately NOT
 * the block-batched-accumulation shape qt_addrow/qt_matvec_rows and matmul_fp8 all share --
 * this recomputes each element's block index and looks up its scale independently, on every
 * iteration, rather than walking block-by-block. Reuses quant.h's e4m3_decode/fp8_nblk/
 * FP8_BLOCK constants deliberately: those ARE the declared format geometry under test here
 * (the same LUT/geometry every fmt=8 consumer in the tree shares), not implementation
 * detail this reference should reinvent. */
static void qt_dequant_row_ref_fmt8(const QT *t, int row, float *out){
    int I=t->I;
    int64_t nblkI=fp8_nblk(I), blkO=(int64_t)row/FP8_BLOCK;
    const float *scl=t->s+blkO*nblkI;
    for(int i=0;i<I;i++){
        int64_t bi=(int64_t)i/FP8_BLOCK;
        out[i]=e4m3_decode((uint8_t)t->q8[(int64_t)row*I+i])*scl[bi];
    }
}

/* ---- fixture builders: one per format, deterministic pseudo-random payload ---- */
static void fill_fmt0(QT *t, int O, int I){
    t->fmt=0; t->O=O; t->I=I; t->gs=0;
    t->qf=(float*)malloc((size_t)O*I*sizeof(float));
    for(int i=0;i<O*I;i++) t->qf[i]=rndsmallf();
}
static void fill_fmt1(QT *t, int O, int I){
    t->fmt=1; t->O=O; t->I=I; t->gs=0;
    t->q8=(int8_t*)malloc((size_t)O*I);
    t->s=(float*)malloc((size_t)O*sizeof(float));
    for(int i=0;i<O*I;i++) t->q8[i]=(int8_t)(rndbyte()-128);
    for(int i=0;i<O;i++) t->s[i]=0.01f+0.001f*(float)i;
}
static void fill_fmt2(QT *t, int O, int I){
    t->fmt=2; t->O=O; t->I=I; t->gs=0;
    int rb=(I+1)/2;
    t->q4=(uint8_t*)malloc((size_t)O*rb);
    t->s=(float*)malloc((size_t)O*sizeof(float));
    for(int i=0;i<O*rb;i++) t->q4[i]=rndbyte();
    for(int i=0;i<O;i++) t->s[i]=0.02f+0.001f*(float)i;
}
static void fill_fmt3(QT *t, int O, int I){
    t->fmt=3; t->O=O; t->I=I; t->gs=0;
    int rb=(I+3)/4;
    t->q4=(uint8_t*)malloc((size_t)O*rb);
    t->s=(float*)malloc((size_t)O*sizeof(float));
    for(int i=0;i<O*rb;i++) t->q4[i]=rndbyte();
    for(int i=0;i<O;i++) t->s[i]=0.03f+0.001f*(float)i;
}
static void fill_fmt4(QT *t, int O, int I, int gs){
    t->fmt=4; t->O=O; t->I=I; t->gs=gs;
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    t->q4=(uint8_t*)malloc((size_t)O*rb);
    t->s=(float*)malloc((size_t)O*ng*sizeof(float));
    for(int i=0;i<O*rb;i++) t->q4[i]=rndbyte();
    for(int i=0;i<O*ng;i++) t->s[i]=0.015f+0.0007f*(float)i;
}
static void fill_fmt5(QT *t, int O, int I){
    t->fmt=5; t->O=O; t->I=I; t->gs=0;
    int64_t ng=i3_groups(I), rb=i3_rowbytes(I);
    t->q4=(uint8_t*)malloc((size_t)O*rb);
    t->s=(float*)malloc((size_t)(O*ng)*sizeof(float));
    for(int i=0;i<O*rb;i++) t->q4[i]=rndbyte();
    for(int i=0;i<O*ng;i++) t->s[i]=0.025f+0.0003f*(float)i;
}
/* kv_b-shaped fmt=8 fixture: I is left caller-chosen so callers can pick both the
 * real contraction dimension (kv_lora_rank=512, GLM-5.2's MLA latent width) and
 * small tail-covering shapes. */
static void fill_fmt8(QT *t, int O, int I){
    t->fmt=8; t->O=O; t->I=I; t->gs=0;
    int64_t nblkO=fp8_nblk(O), nblkI=fp8_nblk(I), nblk=nblkO*nblkI;
    t->q8=(int8_t*)malloc((size_t)O*I);
    t->s=(float*)malloc((size_t)nblk*sizeof(float));
    for(int64_t i=0;i<(int64_t)O*I;i++) t->q8[i]=(int8_t)rndbyte_nonan();
    for(int64_t i=0;i<nblk;i++) t->s[i]=0.01f+0.001f*(float)i;
}
static void free_qt(QT *t){ free(t->qf); free(t->q8); free(t->q4); free(t->s); memset(t,0,sizeof *t); }

/* ---- byte-identity: qt_addrow / qt_matvec_rows vs the independent reference, every
 * format both functions still legitimately handle ---- */
/* Epsilon, not bit-exact: qt_addrow precomputes c=coef*scale ONCE, then c*w[i]
 * ((coef*scale)*w[i]); the reference below multiplies coef*(scale*w[i]) --
 * mathematically identical, but floating-point multiplication isn't
 * associative, so the two can differ in the last ULP for some inputs. This
 * is expected float-reassociation noise, not a regression -- confirmed by
 * inspecting actual failures at bit-exact comparison (all last-digit-of-
 * mantissa only, no shape/format-decode errors) before relaxing to epsilon,
 * matching this suite's own established practice for reduction-order-
 * sensitive checks (e.g. the metal-test suite's worst_rel comparisons). */
static void check_addrow_identity(QT *t, const char *tag){
    int I=t->I; float *ref=(float*)malloc((size_t)I*sizeof(float));
    float *acc=(float*)malloc((size_t)I*sizeof(float));
    float coef=1.7f;   /* != 1, so a coef-scaling bug can't hide */
    for(int row=0; row<t->O; row++){
        qt_dequant_row_ref(t,row,ref);
        memset(acc,0,(size_t)I*sizeof(float));
        qt_addrow(t,row,coef,acc);
        for(int i=0;i<I;i++){
            float want=coef*ref[i];
            float ae=fabsf(acc[i]-want);
            float rel = fabsf(want)>1e-6f ? ae/fabsf(want) : ae;
            if(rel > 1e-5f){
                printf("FAIL %s: qt_addrow row=%d i=%d got=%.9g want=%.9g rel=%.3g\n",tag,row,i,(double)acc[i],(double)want,(double)rel);
                fails++;
            }
        }
    }
    free(ref); free(acc);
}
static void check_matvec_identity(QT *t, const char *tag){
    int I=t->I; float *ref=(float*)malloc((size_t)I*sizeof(float));
    float *x=(float*)malloc((size_t)I*sizeof(float));
    for(int i=0;i<I;i++) x[i]=rndsmallf();
    for(int row=0; row<t->O; row++){
        qt_dequant_row_ref(t,row,ref);
        double want=0; for(int i=0;i<I;i++) want+=(double)ref[i]*x[i];
        float y=0.f;
        qt_matvec_rows(t,row,1,x,&y);
        double relerr = fabs(want)>1e-6 ? fabs((double)y-want)/fabs(want) : fabs((double)y-want);
        if(relerr > 1e-4){
            printf("FAIL %s: qt_matvec_rows row=%d got=%.9g want=%.9g relerr=%.3g\n",tag,row,(double)y,want,relerr);
            fails++;
        }
    }
    free(ref); free(x);
}

static void test_byte_identity_all_formats(void){
    QT t;
    memset(&t,0,sizeof t); fill_fmt0(&t,4,17);   check_addrow_identity(&t,"fmt=0"); check_matvec_identity(&t,"fmt=0"); free_qt(&t);
    memset(&t,0,sizeof t); fill_fmt1(&t,4,17);   check_addrow_identity(&t,"fmt=1"); check_matvec_identity(&t,"fmt=1"); free_qt(&t);
    memset(&t,0,sizeof t); fill_fmt2(&t,4,17);   check_addrow_identity(&t,"fmt=2"); check_matvec_identity(&t,"fmt=2"); free_qt(&t);
    memset(&t,0,sizeof t); fill_fmt3(&t,4,17);   check_addrow_identity(&t,"fmt=3"); check_matvec_identity(&t,"fmt=3"); free_qt(&t);
    memset(&t,0,sizeof t); fill_fmt4(&t,4,40,16);check_addrow_identity(&t,"fmt=4"); check_matvec_identity(&t,"fmt=4"); free_qt(&t);
    memset(&t,0,sizeof t); fill_fmt5(&t,4,130);  check_addrow_identity(&t,"fmt=5"); check_matvec_identity(&t,"fmt=5"); free_qt(&t);
    /* fmt=8: small single-block-both-dims shape (mirrors the other formats' O=4-ish
     * defaults) plus a kv_b-shaped multi-block shape (O=192 -> nblkO=2 with a partial
     * tail block; I=512=kv_lora_rank -> nblkI=4, exact blocks) so both the partial-row-
     * block and the multi-column-block paths through the new branches are exercised. */
    memset(&t,0,sizeof t); fill_fmt8(&t,6,17);    check_addrow_identity(&t,"fmt=8 (single block)"); check_matvec_identity(&t,"fmt=8 (single block)"); free_qt(&t);
    memset(&t,0,sizeof t); fill_fmt8(&t,192,512); check_addrow_identity(&t,"fmt=8 (kv_b-shaped, multi-block)"); check_matvec_identity(&t,"fmt=8 (kv_b-shaped, multi-block)"); free_qt(&t);
    /* nblkI>=2 WITH a partial COLUMN tail: I=200 -> nblkI=2 with a 72-wide tail block,
     * O=130 -> nblkO=2 with a 2-row tail. The kv_b-shaped case above has I=512 (exact
     * column blocks), so the per-row scale STRIDE (nblkI) and the bi block-scale index
     * were only ever exercised at shapes where flooring/misdeviating them is invisible.
     * Mutation-checked: corrupting the block-scale index math (nblkI=I/FP8_BLOCK, the
     * floor) passes every fmt=8 case above and fails exactly this one. */
    memset(&t,0,sizeof t); fill_fmt8(&t,130,200); check_addrow_identity(&t,"fmt=8 (partial column tail)"); check_matvec_identity(&t,"fmt=8 (partial column tail)"); free_qt(&t);
}

/* ---- fmt=8 absorb-path vs the proven non-absorb matmul_fp8 reference (quant.h) ----
 * qt_matvec_rows(t,row,1,x,&y) and matmul_fp8's per-output-row computation are the SAME
 * mathematical quantity (y[row] = sum_i x[i]*dequant(W[row,i])) computed with the exact
 * same block-scale geometry and the exact same accumulation order (float-accumulate within
 * a 128-wide block, double-accumulate the per-block partials) -- both were written to
 * mirror that convention deliberately (see this PR's qt_matvec_rows fmt=8 branch comment
 * in colibri.c). Compiled in the same translation unit with the same flags, so the two are
 * expected to agree tightly; the tolerance below is a guard against incidental FP-contraction
 * differences between the two call sites, not evidence of a real algorithmic mismatch. */
static void test_fmt8_matmul_fp8_parity(void){
    enum { O=192, I=512 };   /* kv_b-shaped: I=kv_lora_rank=512, O spans a partial row-block */
    QT t; memset(&t,0,sizeof t); fill_fmt8(&t,O,I);
    float *x=(float*)malloc((size_t)I*sizeof(float));
    for(int i=0;i<I;i++) x[i]=rndsmallf();
    float *yref=(float*)malloc((size_t)O*sizeof(float));
    matmul_fp8(yref,x,(const uint8_t*)t.q8,t.s,1,I,O);
    for(int row=0; row<O; row++){
        float y=0.f;
        qt_matvec_rows(&t,row,1,x,&y);
        float want=yref[row];
        float ae=fabsf(y-want);
        float rel = fabsf(want)>1e-6f ? ae/fabsf(want) : ae;
        if(rel > 1e-5f){
            printf("FAIL fmt=8 matmul_fp8 parity: row=%d got=%.9g want=%.9g rel=%.3g\n",
                   row,(double)y,(double)want,(double)rel);
            fails++;
        }
    }
    free(x); free(yref); free_qt(&t);
}

/* ---- fmt=8 NaN propagation (policy pin -- see quant.h's "NaN POLICY" note): a NaN
 * weight byte (0x7F/0xFF) decodes to a real IEEE NaN and is left to PROPAGATE, relying
 * on the tested downstream sampler net (tests/test_logit_nan.c), never scrubbed at the
 * weight level. The identity checks above deliberately exclude NaN bytes
 * (rndbyte_nonan) -- and would pass NaN lanes silently anyway (NaN > eps compares
 * false) -- so this pins the absorb-path behavior explicitly against the independent
 * reference: qt_addrow poisons exactly the accumulator lanes whose reference dequant is
 * NaN (per-element accumulate, both NaN byte codes), qt_matvec_rows poisons the whole
 * dot product of any row containing one, and clean rows/lanes of the same tensor stay
 * NaN-free and tolerance-identical to the reference. ---- */
static void test_fmt8_nan_propagation(void){
    enum { O=130, I=200 };   /* same tail-covering shape as the identity case above */
    QT t; memset(&t,0,sizeof t); fill_fmt8(&t,O,I);
    /* one NaN of EACH byte code, in different row-blocks and different column blocks;
     * (129,199) lands in the tail row-block x tail column-block corner */
    const int nan_row[2]={3,129}, nan_col[2]={5,199};
    t.q8[(int64_t)nan_row[0]*I+nan_col[0]]=(int8_t)0x7F;
    t.q8[(int64_t)nan_row[1]*I+nan_col[1]]=(int8_t)0xFF;
    float *ref=(float*)malloc((size_t)I*sizeof(float));
    float *acc=(float*)malloc((size_t)I*sizeof(float));
    float *x=(float*)malloc((size_t)I*sizeof(float));
    for(int i=0;i<I;i++) x[i]=rndsmallf();
    float coef=1.7f;   /* != 1, same reasoning as check_addrow_identity */
    for(int r=0;r<2;r++){
        int row=nan_row[r];
        qt_dequant_row_ref(&t,row,ref);
        CHECK(isnan(ref[nan_col[r]]));   /* the reference itself says: this lane is NaN */
        memset(acc,0,(size_t)I*sizeof(float));
        qt_addrow(&t,row,coef,acc);
        for(int i=0;i<I;i++){
            if(isnan(ref[i])){
                if(!isnan(acc[i])){ printf("FAIL fmt=8 NaN: qt_addrow row=%d i=%d expected NaN, got %.9g\n",row,i,(double)acc[i]); fails++; }
            } else if(isnan(acc[i])){
                printf("FAIL fmt=8 NaN: qt_addrow row=%d i=%d poisoned a clean lane\n",row,i); fails++;
            } else {
                float want=coef*ref[i];
                float ae=fabsf(acc[i]-want);
                float rel=fabsf(want)>1e-6f?ae/fabsf(want):ae;
                if(rel>1e-5f){ printf("FAIL fmt=8 NaN: qt_addrow row=%d i=%d clean lane got=%.9g want=%.9g\n",row,i,(double)acc[i],(double)want); fails++; }
            }
        }
        float y=0.f; qt_matvec_rows(&t,row,1,x,&y);
        CHECK(isnan(y));   /* dot product over a NaN-bearing row is NaN, like the reference sum */
    }
    /* a clean row of the SAME tensor stays NaN-free through both functions */
    { int row=4; float y=0.f;
      qt_dequant_row_ref(&t,row,ref);
      for(int i=0;i<I;i++) CHECK(!isnan(ref[i]));
      memset(acc,0,(size_t)I*sizeof(float)); qt_addrow(&t,row,coef,acc);
      for(int i=0;i<I;i++) CHECK(!isnan(acc[i]));
      qt_matvec_rows(&t,row,1,x,&y); CHECK(!isnan(y)); }
    free(ref); free(acc); free(x); free_qt(&t);
}

/* ---- refusal: fork+pipe+waitpid, this suite's house pattern (see test_fp8_load.c's
 * expect_refuse/expect_stamp_refuse) -- must exit(1) with a "refus"-containing message,
 * never reach the caller's continuation, never crash with a signal. ---- */
/* fmt=8 no longer belongs in this refusal suite -- it has its own decoder now (see the
 * byte-identity and matmul_fp8-parity tests above). fmt=6 (E8/IQ3) has no absorb decoder
 * and must keep refusing loudly through both functions. */
typedef void (*absorb_fn)(void);
static void call_addrow_fmt6(void){
    QT t; memset(&t,0,sizeof t);
    enum { O=4, I=98 };
    static uint8_t q4[O*98]; static float s[1];
    for(int i=0;i<O*98;i++) q4[i]=rndbyte();
    s[0]=0.01f;
    t.fmt=6; t.O=O; t.I=I; t.gs=0; t.q4=q4; t.s=s;
    float acc[I]; memset(acc,0,sizeof acc);
    qt_addrow(&t,0,1.f,acc);
}
static void call_matvec_fmt6(void){
    QT t; memset(&t,0,sizeof t);
    enum { O=4, I=98 };
    static uint8_t q4[O*98]; static float s[1];
    for(int i=0;i<O*98;i++) q4[i]=rndbyte();
    s[0]=0.01f;
    t.fmt=6; t.O=O; t.I=I; t.gs=0; t.q4=q4; t.s=s;
    static float x[I]; for(int i=0;i<I;i++) x[i]=rndsmallf();
    float y=0.f;
    qt_matvec_rows(&t,0,1,x,&y);
}

static int expect_refuse_call(absorb_fn fn, const char *tag){
#ifndef _WIN32
    int pipefd[2]; if(pipe(pipefd)!=0) return 0;
    pid_t pid = fork();
    if(pid < 0) return 0;
    if(pid == 0){
        dup2(pipefd[1],2); close(pipefd[0]); close(pipefd[1]);
        fn();
        _exit(42);   /* reaching here (surviving the call without exit(1)) is the bug */
    }
    close(pipefd[1]);
    char err[1024]={0}; size_t eoff=0; ssize_t n; /* drain to EOF: a single read() can return SHORT on Linux pipes (glibc unbuffered stderr arrives in chunks) -- truncated the refusal message past the marker, CI-caught */ while(eoff<sizeof(err)-1 && (n=read(pipefd[0],err+eoff,sizeof(err)-1-eoff))>0) eoff+=(size_t)n;
    close(pipefd[0]);
    int status=0; waitpid(pid,&status,0);
    if(WIFSIGNALED(status)){
        printf("FAIL %s: crashed with signal %d instead of refusing (exit(1)) -- this IS the "
               "engine defect this test exists to catch\n", tag, WTERMSIG(status));
        return 0;
    }
    int ok = WIFEXITED(status) && WEXITSTATUS(status)==1;
    if(!ok){
        printf("FAIL %s: expected exit(1) refusal, got status=%d, stderr=%.200s\n", tag, status, err);
        return 0;
    }
    if(!strstr(err,"refus")){
        printf("FAIL %s: exited(1) but message lacked a refusal explanation: %.200s\n", tag, err);
        return 0;
    }
    return 1;
#else
    printf("skipped on Windows (no fork): %s\n", tag);
    (void)fn;
    return 1;
#endif
}

static void test_refusals(void){
    CHECK(expect_refuse_call(call_addrow_fmt6,  "qt_addrow refuses fmt=6"));
    CHECK(expect_refuse_call(call_matvec_fmt6,  "qt_matvec_rows refuses fmt=6"));
}

int main(void){
    test_byte_identity_all_formats();
    test_fmt8_matmul_fp8_parity();
    test_fmt8_nan_propagation();
    test_refusals();
    if(fails){ printf("qt_addrow/qt_matvec_rows tests: %d FAILED\n", fails); return 1; }
    printf("qt_addrow/qt_matvec_rows tests: ok\n");
    return 0;
}
