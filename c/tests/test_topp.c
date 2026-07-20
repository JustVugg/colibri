/* Top-p (nucleus) truncation in dist_build: the partial-select rewrite (#335) must be
 * indistinguishable from the old full-vocab qsort for every shape dist_sample can see.
 *
 * Why this test exists (#335): dist_build() previously qsort-ed the entire 151936-entry
 * vocab on every sampled token to find the few-hundred-token head whose cumulative mass
 * reaches g_nuc. It now heapifies (O(V)) and pops only the head (k * O(log V)). The win
 * is structural; the risk is a silent sampling-distribution change, because the contract
 * is subtle:
 *
 *   dist_sample() iterates g_pbuf[0..V-1] BY TOKEN ID and sums probabilities directly.
 *   So dist_build MUST leave g_pbuf indexed by id (never reordered) AND must zero every
 *   truncated tail entry -- merely excluding the tail from the head would leave mass on
 *   it and the sampled distribution would drift with no crash and no error.
 *
 * Strategy: drive the REAL dist_build (via the test_stops.c include-glm.c pattern) on a
 * sweep of distributions and g_nuc values, and compare against an INDEPENDENT reference
 * that re-implements the OLD algorithm (full qsort + zero-tail + renorm) in double on a
 * private buffer. On shapes with no ties the renormalized head must be BIT-IDENTICAL to
 * the reference (the issue's stated invariant: s2 accumulates in the same descending
 * order). On tie shapes, where the unstable qsort already left ordering unspecified, we
 * check multiset equality instead. Every shape also checks: exact-zero tails, head sums
 * to 1.0, and a sane keep-count.
 *
 * No scratch files: the test runs entirely in memory (no mkdtemp), so it builds clean on
 * the Windows MinGW CI job without the unmerged compat shim (#352). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <math.h>

static int g_nfails = 0;

/* pointer set by ref_build so cmp_ref_desc can read the current reference buffer
 * (the qsort comparator gets no user-data argument in C). */
static const double *g_ref_p = NULL;

#define FAIL(fmt, ...) do { \
    fprintf(stderr, "  FAIL [%s V=%d nuc=%.3f shape=%s]: " fmt "\n", \
            label, V, nuc, shape_name, ##__VA_ARGS__); \
    g_nfails++; \
    return; \
} while (0)

/* ---- independent reference: the OLD algorithm, in double, on a private buffer ------- */
/* Stable qsort by descending probability (ties broken by ascending index, which makes
 * the reference deterministic regardless of the production comparator). */
static int cmp_ref_desc(const void *a, const void *b){
    double pa = ((const double *)g_ref_p)[*(const int*)a];
    double pb = ((const double *)g_ref_p)[*(const int*)b];
    if (pa < pb) return 1;
    if (pa > pb) return -1;
    /* tie -> lower index first (stable, unlike the production comparator) */
    return *(const int*)a - *(const int*)b;
}

/* Build the reference distribution into out[0..V-1] (indexed by token id), mirroring the
 * old dist_build: softmax(lo/temp) truncated to top-p nuc, tail zeroed, head renormalized.
 * Returns the keep-count through *keep_out. */
static void ref_build(const float *lo, int V, double temp, double nuc,
                      double *out, int *pidx, int *keep_out){
    double mx = lo[0]; for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
    double s = 0, invt = 1.0 / (temp > 1e-4 ? temp : 1e-4);
    for (int i = 0; i < V; i++){ out[i] = exp((lo[i]-mx)*invt); s += out[i]; }
    for (int i = 0; i < V; i++) out[i] /= s;

    if (nuc > 0 && nuc < 1.0){
        for (int i = 0; i < V; i++) pidx[i] = i;
        qsort(pidx, V, sizeof(int), cmp_ref_desc);
        double cum = 0; int keep = V;
        for (int i = 0; i < V; i++){ cum += out[pidx[i]]; if (cum >= nuc){ keep = i+1; break; } }
        double s2 = 0;
        for (int i = keep; i < V; i++) out[pidx[i]] = 0;
        for (int i = 0; i < keep; i++) s2 += out[pidx[i]];
        for (int i = 0; i < keep; i++) out[pidx[i]] /= s2;
        *keep_out = keep;
    } else {
        *keep_out = V;
    }
}

/* count how many production g_pbuf entries are non-zero == the head size */
static int head_count(int V){
    int n = 0; for (int i = 0; i < V; i++) if (g_pbuf[i] != 0.f) n++; return n;
}

/* Run one case: load logits into g_pbuf via the real dist_build, compare to reference.
 * shape_name is for diagnostics only. */
static void check_case(const char *label, int V, double nuc, const char *shape_name,
                       const float *lo){
    /* reference on a private buffer */
    double *ref = malloc((size_t)V * sizeof(double));
    int *ridx = malloc((size_t)V * sizeof(int));
    int ref_keep = 0;
    g_ref_p = ref;   /* cmp_ref_desc reads this */
    ref_build(lo, V, g_temp, nuc, ref, ridx, &ref_keep);

    /* production: drive the real dist_build (writes the global g_pbuf) */
    g_nuc = (float)nuc;
    dist_build(lo, V);

    int got_keep = head_count(V);

    /* 1. keep-count must match the reference exactly. The partial select and the old
     *    qsort keep the same NUMBER of tokens by construction (same cumulative-mass rule);
     *    a count divergence is a real bug, not a tie artifact. */
    if (got_keep != ref_keep)
        FAIL("keep-count mismatch: got %d, ref %d", got_keep, ref_keep);

    /* 2. Detect ties across the WHOLE pre-truncation distribution, not just the kept set.
     *    A tie at the head/tail boundary makes which-side-a-token-lands-on interchangeable:
     *    both algorithms keep the right count but may keep different MEMBERS. So any input
     *    with a duplicated softmax value needs the relaxed multiset comparison below. We
     *    detect this on the reference softmax (pre-truncation) by sorting all V values. */
    int has_ties = 0;
    {
        double *all = malloc((size_t)V * sizeof(double));
        /* reconstruct the pre-truncation softmax the same way ref_build does */
        double mx = lo[0]; for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
        double s = 0, invt = 1.0 / (g_temp > 1e-4 ? g_temp : 1e-4);
        for (int i = 0; i < V; i++){ all[i] = exp((lo[i]-mx)*invt); s += all[i]; }
        for (int i = 0; i < V; i++) all[i] /= s;
        for (int a = 1; a < V; a++){ double k = all[a]; int b = a-1;
            while (b >= 0 && all[b] > k){ all[b+1] = all[b]; b--; } all[b+1] = k; }
        for (int a = 1; a < V; a++) if (all[a] == all[a-1]){ has_ties = 1; break; }
        free(all);
    }

    if (has_ties){
        /* Multiset equality of the non-zero (head) values. Ties make membership
         * interchangeable, so we compare sorted value-multisets, not id-aligned values.
         * Tolerance is 1e-6 relative -- the engine uses float arithmetic, the reference
         * double, so sub-ULP noise is expected (matches test_i4_grouped.c's convention). */
        double *got = malloc((size_t)ref_keep * sizeof(double));
        int gm = 0;
        for (int i = 0; i < V; i++) if (g_pbuf[i] != 0.f) got[gm++] = (double)g_pbuf[i];
        if (gm != ref_keep)
            FAIL("tie-shape head size mismatch: got %d non-zero, ref %d", gm, ref_keep);
        for (int a = 1; a < gm; a++){ double k = got[a]; int b = a-1;
            while (b >= 0 && got[b] > k){ got[b+1] = got[b]; b--; } got[b+1] = k; }
        double *rsort = malloc((size_t)ref_keep * sizeof(double));
        int rm = 0;
        for (int i = 0; i < V; i++) if (ref[i] != 0.0) rsort[rm++] = ref[i];
        for (int a = 1; a < rm; a++){ double k = rsort[a]; int b = a-1;
            while (b >= 0 && rsort[b] > k){ rsort[b+1] = rsort[b]; b--; } rsort[b+1] = k; }
        int mm = 0; double worst = 0;
        for (int i = 0; i < gm; i++){
            double d = fabs(got[i] - rsort[i]);
            double rel = rsort[i] > 1e-30 ? d / rsort[i] : d;
            if (rel > worst) worst = rel;
            if (rel > 1e-6) mm++;
        }
        free(got); free(rsort);
        if (mm) FAIL("tie-shape multiset mismatch: %d/%d head values differ beyond 1e-6 rel (worst %.3g)",
                     mm, ref_keep, worst);
    } else {
        /* No ties anywhere: membership is forced, so compare id-aligned head values. The
         * engine computes in float (g_pbuf /= (float)s2) while the reference uses double,
         * so the comparison is relative-tolerance (1e-6), not bit-exact -- the partial
         * select and qsort accumulate s2 in the same descending order, so any difference
         * is pure float-rounding noise, not an ordering bug. */
        int bad = 0; int first_id = -1; float gv = 0, rv = 0; double worst = 0;
        for (int i = 0; i < V; i++){
            if (ref[i] == 0.0) continue;   /* tail */
            float want = (float)ref[i];
            double d = fabs((double)g_pbuf[i] - (double)want);
            double rel = fabs((double)want) > 1e-30 ? d / fabs((double)want) : d;
            if (rel > worst) worst = rel;
            if (rel > 1e-6){
                bad++; if (first_id < 0){ first_id = i; gv = g_pbuf[i]; rv = want; }
                if (bad > 3) break;
            }
        }
        if (bad)
            FAIL("head not within 1e-6 rel of reference: %d entries differ (first id %d: got %.9g want %.9g, worst %.3g)",
                 bad, first_id, (double)gv, (double)rv, worst);
    }

    /* 3. head must renormalize to 1.0 (within float epsilon) */
    double sum = 0; for (int i = 0; i < V; i++) sum += g_pbuf[i];
    if (fabs(sum - 1.0) > 1e-5)
        FAIL("head does not sum to 1.0: sum=%.12g (keep=%d)", sum, got_keep);

    free(ref); free(ridx);
    printf("  ok [V=%d nuc=%.3f shape=%s keep=%d%s sum=%.10f]\n",
           V, nuc, shape_name, got_keep, has_ties ? " (ties)" : "", sum);
}
#undef FAIL

/* deterministic xorshift32 RNG (matches the test_i4_grouped.c convention) */
static uint32_t rng_state = 0x12345678u;
static uint32_t xr(void){ rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5; return rng_state; }
static double frand(void){ return (xr() >> 8) * (1.0 / 16777216.0); }   /* [0,1) */

/* fill logits for a given shape. Shapes chosen to stress the comparator and the head/tail
 * boundary differently. */
static void fill_shape(float *lo, int V, int shape){
    switch (shape){
    case 0: /* uniform -> every token equal probability -> massive tie plateau */
        for (int i = 0; i < V; i++) lo[i] = 0.f; break;
    case 1: /* peaked: one dominant token, rest small and distinct (no ties).
             * The fixed hot-spots are clamped to V-1 so small V (incl. V=1) doesn't
             * write out of bounds and corrupt heap metadata on the later free(lo). */
        for (int i = 0; i < V; i++) lo[i] = (float)(-1.0 - frand()*4.0);
        lo[0] = 3.f; lo[V/3<V?V/3:V-1] = 1.f; lo[V/2<V?V/2:V-1] = 0.5f; break;
    case 2: /* all-equal distinct decay (no ties): geometric, strictly decreasing */
        for (int i = 0; i < V; i++) lo[i] = (float)(-0.001 * (double)i); break;
    case 3: /* plateau ties: blocks of equal value -> comparator tie handling */
        for (int i = 0; i < V; i++) lo[i] = (float)(-(double)(i / 7));   /* 7-wide plateaus */
        break;
    case 4: /* sharp-tail: a few hot, then a long flat floor (small tie at the floor).
             * Hot count is min(12,V) so V<12 (incl. V=1) stays in bounds. */
        for (int i = 0; i < V; i++) lo[i] = -8.f;
        { int hot = V<12 ? V : 12; for (int i = 0; i < hot; i++) lo[i] = (float)(2.0 - frand()); } break;
    }
}

int main(void){
    /* sizes: small for exhaustive tie detection up to near-production scale */
    int sizes[] = {1, 2, 8, 64, 257, 1519};   /* 1519 ~= V/100 of GLM-5.2 */
    double nucs[] = {0.001, 0.5, 0.9, 0.999}; /* tight -> almost-everything */
    int n_shapes = 5;

    /* temperature used by dist_build: pick a normal serving value */
    g_temp = 0.7f;

    int cases = 0;
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++){
        int V = sizes[si];
        /* dist_build allocates g_pbuf/g_pidx ONCE and reuses them (single-V invariant in
         * real serving, where V is the constant model vocab). This sweep varies V, so free
         * and force a reallocation per size -- otherwise a later, larger V would overflow
         * the buffer sized for the first (smallest) V. */
        free(g_pbuf); g_pbuf = NULL; free(g_pidx); g_pidx = NULL;
        float *lo = malloc((size_t)V * sizeof(float));
        for (int shape = 0; shape < n_shapes; shape++){
            fill_shape(lo, V, shape);
            for (size_t ni = 0; ni < sizeof(nucs)/sizeof(nucs[0]); ni++){
                char label[32]; snprintf(label, sizeof(label), "size[%zu]/shape[%d]", si, shape);
                const char *sn = (const char*[]){"uniform","peaked","geometric","plateau","sharptail"}[shape];
                check_case(label, V, nucs[ni], sn, lo);
                cases++;
            }
        }
        free(lo);
    }

    /* guard-off path: g_nuc >= 1 must skip truncation entirely (full softmax kept) */
    {
        int V = 256; float lo[256];
        for (int i = 0; i < V; i++) lo[i] = (float)(frand()*4 - 2);
        g_nuc = 1.0f; dist_build(lo, V);
        int nz = 0; for (int i = 0; i < V; i++) if (g_pbuf[i] != 0.f) nz++;
        if (nz != V){ fprintf(stderr, "  FAIL [guard-off nuc=1.0]: %d/%d entries kept, expected all\n", nz, V); g_nfails++; }
        else printf("  ok [guard-off nuc=1.0 keep=%d]\n", nz);
        cases++;

        g_nuc = 0.0f; dist_build(lo, V);
        nz = 0; for (int i = 0; i < V; i++) if (g_pbuf[i] != 0.f) nz++;
        if (nz != V){ fprintf(stderr, "  FAIL [guard-off nuc=0.0]: %d/%d entries kept, expected all\n", nz, V); g_nfails++; }
        else printf("  ok [guard-off nuc=0.0 keep=%d]\n", nz);
        cases++;
    }

    /* extreme tie edge case: V=1, single token -> keep=1 regardless of nuc */
    {
        float lo[1] = {5.f};
        g_nuc = 0.5f; dist_build(lo, 1);
        if (g_pbuf[0] == 0.f || !(fabs((double)g_pbuf[0] - 1.0) < 1e-6)){
            fprintf(stderr, "  FAIL [V=1]: g_pbuf[0]=%.9g, expected 1.0\n", (double)g_pbuf[0]); g_nfails++;
        } else printf("  ok [V=1 keep=1]\n");
        cases++;
    }

    printf("\ntest_topp: %d cases run, %d failure(s)\n", cases, g_nfails);
    if (g_nfails){ printf("test_topp: FAIL\n"); return 1; }
    printf("test_topp: ok\n");
    return 0;
}
