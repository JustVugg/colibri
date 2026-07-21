/* DSA top-keep partial-select: the quickselect rewrite (#356) must produce a
 * BIT-IDENTICAL kept-position set to the old full-vocab qsort, for every score
 * shape the attention indexer can see.
 *
 * Why this test exists (#356): attention_rows() selects the top-`keep` context
 * keys (index_topk=2048 on GLM-5.2) to attend to. It previously did this by
 * full-qsorting all `nk` scores (O(nk log nk)) and reading tmp[keep-1] as the
 * threshold. It now does a partial_select_desc (quickselect, O(nk) average) and
 * takes the threshold as the min of the selected top-keep block. The contract is
 * subtle but STRONGER than test_topp's:
 *
 *   The two position-order scans that build dst[] --
 *     for t: if isc[t] > thr  -> keep     (strictly above threshold)
 *     for t: if isc[t] == thr -> keep     (ties, in position order)
 *   -- are UNCHANGED by the rewrite. So if the threshold value is identical,
 *   the kept-position set is identical element-by-element (not just as a
 *   multiset, which is all the unstable sampling heap in #335 could promise).
 *
 * Strategy: drive the REAL partial_select_desc (via the include-glm.c pattern)
 * and replicate the production threshold derivation + scans, then compare the
 * resulting dst[] against an INDEPENDENT reference that re-implements the OLD
 * algorithm (full qsort + tmp[keep-1] threshold) on a private buffer. The kept
 * sets must be element-wise equal on every shape, including tie plateaus where
 * the boundary membership is decided by the position scan.
 *
 * We also directly unit-test partial_select_desc's partition invariant: after
 * the call, max(a[keep..n)) <= min(a[0..keep)) -- i.e. the keep largest really
 * did land in the prefix. This catches a broken quickselect even before the
 * end-to-end comparison.
 *
 * In-memory only (no scratch files), so it builds clean on the Windows MinGW CI
 * job without the unmerged compat shim. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <math.h>

static int g_nfails = 0;

#define FAIL(fmt, ...) do { \
    fprintf(stderr, "  FAIL [%s nk=%d keep=%d shape=%s]: " fmt "\n", \
            label, nk, keep, shape_name, ##__VA_ARGS__); \
    g_nfails++; \
    return; \
} while (0)

/* ---- independent reference: the OLD algorithm (full qsort + tmp[keep-1]) ---- */
/* qsort comparator matching the production cmp_fdesc exactly (desc, unstable). */
static int cmp_ref_desc(const void *a, const void *b){
    float x=*(const float*)a, y=*(const float*)b; return x<y?1:x>y?-1:0; }

/* Reproduce the OLD glm.c:2589-2596 exactly: copy, qsort desc, threshold =
 * tmp[keep-1], then the two position-order scans into dst[]. Returns nd. */
static int keep_old(const float *isc, int nk, int keep, int *dst){
    float *tmp=malloc((size_t)nk*sizeof(float));
    memcpy(tmp,isc,(size_t)nk*sizeof(float));
    qsort(tmp,(size_t)nk,sizeof(float),cmp_ref_desc);
    float thr=tmp[keep-1];
    int nd=0;
    for(int t=0;t<nk && nd<keep;t++) if(isc[t]>thr) dst[nd++]=t;
    for(int t=0;t<nk && nd<keep;t++) if(isc[t]==thr) dst[nd++]=t;
    free(tmp);
    return nd;
}

/* Reproduce the NEW glm.c path: partial_select desc, threshold = min of the
 * selected block, same two scans. Uses the REAL partial_select_desc from glm.c. */
static int keep_new(const float *isc, int nk, int keep, int *dst){
    float *tmp=malloc((size_t)nk*sizeof(float));
    memcpy(tmp,isc,(size_t)nk*sizeof(float));
    partial_select_desc(tmp,nk,keep);
    float thr=tmp[0]; for(int t=1;t<keep;t++) if(tmp[t]<thr) thr=tmp[t];
    int nd=0;
    for(int t=0;t<nk && nd<keep;t++) if(isc[t]>thr) dst[nd++]=t;
    for(int t=0;t<nk && nd<keep;t++) if(isc[t]==thr) dst[nd++]=t;
    free(tmp);
    return nd;
}

/* ---- direct unit test of the partition invariant ---- */
static void check_partition(const char *label, const float *isc, int nk, int keep,
                            const char *shape_name){
    float *tmp=malloc((size_t)nk*sizeof(float));
    memcpy(tmp,isc,(size_t)nk*sizeof(float));
    partial_select_desc(tmp,nk,keep);
    /* invariant: every element of tmp[0..keep) is >= every element of tmp[keep..n).
     * (>=, not >: equal values may sit on either side of the partition boundary,
     * which is fine -- the threshold is the MIN of the prefix, and the position
     * scan handles ties.) */
    float top_min=INFINITY, tail_max=-INFINITY;
    for(int i=0;i<keep;i++)        if(tmp[i]<top_min) top_min=tmp[i];
    for(int i=keep;i<nk;i++)       if(tmp[i]>tail_max) tail_max=tmp[i];
    if(!(top_min >= tail_max))
        FAIL("partition invariant violated: top_min=%.9g < tail_max=%.9g", top_min, tail_max);
    free(tmp);
}

/* ---- end-to-end: old vs new kept-set must be element-wise identical ---- */
static void check_case(const char *label, int nk, int keep, const char *shape_name,
                       const float *isc){
    int *da=malloc((size_t)nk*sizeof(int));
    int *db=malloc((size_t)nk*sizeof(int));
    int na=keep_old(isc,nk,keep,da);
    int nb=keep_new(isc,nk,keep,db);

    /* 1. both keep exactly `keep` positions (the contract: keep the top-keep by
     *    count). A count mismatch is a real bug, not a tie artifact. */
    if(na!=keep) FAIL("old kept %d, expected %d (old path is the reference)", na, keep);
    if(nb!=keep) FAIL("new kept %d, expected %d", nb, keep);
    if(na!=nb)   FAIL("keep-count mismatch: old=%d new=%d", na, nb);

    /* 2. element-wise identical dst[]. This is the strong contract: because the
     *    threshold is derived identically and the position-order scans are byte-
     *    for-byte the same, the kept SET and its ORDER must match exactly. (This
     *    is what makes #356 cleaner than #335, which was multiset-only.) */
    int first_diff=-1;
    for(int i=0;i<na;i++){ if(da[i]!=db[i]){ first_diff=i; break; } }
    if(first_diff>=0)
        FAIL("kept-set differs at index %d: old dst[%d]=%d new dst[%d]=%d",
             first_diff, first_diff, da[first_diff], first_diff, db[first_diff]);

    /* 3. also check the partition invariant directly (catches a subtly broken
     *    quickselect even if the threshold happened to come out right). */
    check_partition(label,isc,nk,keep,shape_name);

    free(da); free(db);
    printf("  ok [nk=%d keep=%d shape=%s]\n", nk, keep, shape_name);
}
#undef FAIL

/* deterministic xorshift32 RNG (matches the test_i4_grouped.c / test_topp.c convention) */
static uint32_t rng_state = 0x12345678u;
static uint32_t xr(void){ rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5; return rng_state; }
static double frand(void){ return (xr() >> 8) * (1.0 / 16777216.0); }   /* [0,1) */

/* fill scores for a given shape. Shapes stress the threshold boundary and the
 * quickselect's median-of-three pivot differently. */
static void fill_shape(float *isc, int nk, int shape){
    switch(shape){
    case 0: /* uniform random distinct (no ties): the clean contract case */
        for(int i=0;i<nk;i++) isc[i]=(float)(frand()*1000.0); break;
    case 1: /* peaked: a few hot, long distinct tail (realistic attention shape) */
        for(int i=0;i<nk;i++) isc[i]=(float)(-1.0 - frand()*4.0);
        isc[0]=3.f; if(nk>3) isc[nk/3]=1.f; if(nk>2) isc[nk/2]=0.5f; break;
    case 2: /* strictly decreasing geometric (no ties): sorted input -- worst case
             * for a naive quickselect; median-of-three must handle it */
        for(int i=0;i<nk;i++) isc[i]=(float)(-0.001*(double)i); break;
    case 3: /* strictly increasing (reverse-sorted): the other quickselect worst case */
        for(int i=0;i<nk;i++) isc[i]=(float)(0.001*(double)i); break;
    case 4: /* plateau ties: blocks of equal value -> boundary membership decided
             * entirely by the position scan (exercises the ==thr path) */
        for(int i=0;i<nk;i++) isc[i]=(float)(-(double)(i/7)); break;
    case 5: /* all-equal: every value identical -> degenerate threshold, all kept
             * via the ==thr scan; quickselect must not infinite-loop or corrupt */
        for(int i=0;i<nk;i++) isc[i]=5.f; break;
    }
}

int main(void){
    /* Sizes around the real index_topk=2048 boundary, plus small cases that
     * exercise the k>=n / k==1 / k==n edges. */
    int nks[]  = {1, 2, 8, 64, 2049, 4097, 8193};
    int keeps[] = {1, 8, 256, 1024, 2048};
    int n_shapes = 6;

    int cases = 0;
    for(size_t ni=0; ni<sizeof(nks)/sizeof(nks[0]); ni++){
        int nk=nks[ni];
        float *isc=malloc((size_t)nk*sizeof(float));
        for(int shape=0; shape<n_shapes; shape++){
            /* skip shapes that write out of bounds on tiny nk (fill_shape guards
             * the hot-spots with nk>n, but skip the plateau/geometric edge if nk<7) */
            fill_shape(isc,nk,shape);
            for(size_t ki=0; ki<sizeof(keeps)/sizeof(keeps[0]); ki++){
                int keep=keeps[ki];
                if(keep>nk) continue;     /* keep<=nk invariant of the production code */
                if(keep<=0) continue;
                char label[40]; snprintf(label,sizeof(label),"nk[%zu]/keep[%zu]/shape[%d]",ni,ki,shape);
                const char *sn=(const char*[]){"random","peaked","decreasing","increasing","plateau","all-equal"}[shape];
                check_case(label,nk,keep,sn,isc);
                cases++;
            }
        }
        free(isc);
    }

    /* edge: keep == nk (nothing to partition; both paths keep everything) */
    {
        int nk=100, keep=100; float isc[100];
        for(int i=0;i<nk;i++) isc[i]=(float)frand();
        int *db=malloc(sizeof(int)*nk); int nb=keep_new(isc,nk,keep,db);
        if(nb!=nk){ fprintf(stderr,"  FAIL [keep==nk]: kept %d expected %d\n",nb,nk); g_nfails++; }
        else printf("  ok [keep==nk nk=%d]\n",nk);
        free(db); cases++;
    }
    /* edge: keep == 1 (threshold = the single max; quickselect must find it) */
    {
        int nk=500, keep=1; float isc[500];
        for(int i=0;i<nk;i++) isc[i]=(float)frand();
        int da[1],db[1]; int na=keep_old(isc,nk,keep,da), nb=keep_new(isc,nk,keep,db);
        if(na!=1||nb!=1||da[0]!=db[0]){
            fprintf(stderr,"  FAIL [keep==1]: old={%d (n=%d)} new={%d (n=%d)}\n",da[0],na,db[0],nb); g_nfails++; }
        else printf("  ok [keep==1 argmax=%d]\n",db[0]); cases++;
    }
    /* edge: all-equal scores, keep in the middle -> every kept slot is a tie;
     * the position scan must pick positions 0..keep-1 deterministically */
    {
        int nk=1000, keep=500; float isc[1000];
        for(int i=0;i<nk;i++) isc[i]=3.14f;
        int *db=malloc(sizeof(int)*nk); int nb=keep_new(isc,nk,keep,db);
        int bad=0; for(int i=0;i<nb;i++) if(db[i]!=i) bad=1;
        if(nb!=keep||bad){ fprintf(stderr,"  FAIL [all-equal keep=%d]: nb=%d bad=%d\n",keep,nb,bad); g_nfails++; }
        else printf("  ok [all-equal keep=%d -> positions 0..%d]\n",keep,keep-1); cases++;
        free(db);
    }

    printf("\ntest_dsa_select: %d cases run, %d failure(s)\n", cases, g_nfails);
    if(g_nfails){ printf("test_dsa_select: FAIL\n"); return 1; }
    printf("test_dsa_select: ok\n");
    return 0;
}
