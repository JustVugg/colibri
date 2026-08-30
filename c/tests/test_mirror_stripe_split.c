/* test_mirror_stripe_split.c -- mir_stripe_plan() contract.
 *
 * The striped expert read joins every stripe, so one read costs whatever the
 * SLOWEST leg costs. Before this the chunk size was len/nsf regardless of the
 * per-drive bandwidth the engine had already probed, so a drive the engine
 * knew was 10x slower still got an equal third of every 19 MB expert.
 *
 * Measured on 2x NVMe + 1x SATA, 19 MB expert, O_DIRECT, one thread per leg:
 *
 *     2 legs, equal      join waits  2.0 ms
 *     3 legs, equal      join waits 12.6 ms     <- adding a slow drive
 *     3 legs, weighted   join waits  1.9 ms
 *
 * End to end that was 0.900 -> 0.666 -> 0.888 tok/s (n=5 per arm, GLM-5.2 int4,
 * RTX 3090 host, interleaved on an idle machine).
 *
 * These are pure-arithmetic checks: no fds, no threads, no model. The plan
 * reads only its arguments plus g_mir_cut, so a test drives it by setting that.
 */
#define COLI_STRIPE_SPLIT_TEST 1
#define main coli_glm_main_unused   /* same trick as test_cap_precedence.c */
#include "../colibri.c"
#undef main

static int failures = 0;

static void ck(int cond, const char *what, const char *detail){
    printf("%-4s %-58s %s\n", cond?"ok":"FAIL", what, detail?detail:"");
    if(!cond) failures++;
}

/* g_mir_cut holds CUMULATIVE cuts of 256; replica r's weight is
 * cut[r] - cut[r-1]. Set it from plain per-replica weights. */
static void set_weights(const int *wt, int n){
    int acc=0, tot=0;
    for(int i=0;i<n;i++) tot+=wt[i];
    for(int i=0;i<n;i++){
        acc += (int)((256.0*wt[i])/tot + 0.5);
        if(acc>256) acc=256;
        g_mir_cut[i]=acc;
    }
    g_mir_cut[n-1]=256;                  /* last replica absorbs rounding */
    g_mir_nrep=n;
}

int main(void){
    const int64_t LEN = 19*1024*1024;    /* a GLM-5.2 int4 expert */
    int64_t off[MIR_REPS], sz[MIR_REPS];
    char d[192];

    /* 1. THE LOAD-BEARING ONE: a slow leg must get a SMALL share.
     *    This is the whole bug. Equal thirds of 19 MB is 6.33 MB each; the
     *    measured 10x-slower drive must come out far below that. */
    {
        int wt[3] = {51, 45, 4};         /* 5.69 / 5.03 / 0.44 GB/s */
        int srep[3] = {0,1,2};
        set_weights(wt, 3);
        mir_stripe_plan(LEN, 3, srep, 0, off, sz);
        int64_t equal = LEN/3;
        snprintf(d,sizeof d,"slow leg %lld B vs equal-split %lld B",
                 (long long)sz[2], (long long)equal);
        /* Both bounds are deliberately far from `equal`. A `> equal` check on
         * the fast leg PASSES on the old equal-split code, by 2,731 bytes of
         * 4K rounding - verified by reverting the plan and watching it stay
         * green while its sibling went red. An assertion that survives the bug
         * is not testing for the bug. */
        ck(sz[2] < equal/2,       "a 4 pct leg gets under half an equal third", d);
        ck(sz[0] > equal*13/10,   "a 51 pct leg gets over 1.3x an equal third", d);
    }

    /* 2. Sizes must sum to exactly len and offsets must be contiguous.
     *    A gap or an overlap here is silent data corruption, not a slow read. */
    {
        int wt[3] = {51, 45, 4};
        int srep[3] = {0,1,2};
        set_weights(wt, 3);
        for(int rep=0; rep<3; rep++){
            mir_stripe_plan(LEN, 3, srep, rep, off, sz);
            int64_t total=0; int contiguous=1;
            for(int i=0;i<3;i++){
                if(off[i]!=total) contiguous=0;
                total+=sz[i];
            }
            snprintf(d,sizeof d,"rep=%d sum=%lld want=%lld",rep,(long long)total,(long long)LEN);
            ck(total==LEN,   "stripe sizes sum to exactly len", d);
            ck(contiguous,   "stripe offsets are contiguous from 0", d);
        }
    }

    /* 3. Equal weights must still produce an equal split. Without this the
     *    suite would pass on a plan that always favours replica 0, and every
     *    matched-drive user would silently get a worse split than before. */
    {
        int wt[2] = {50, 50};
        int srep[2] = {0,1};
        set_weights(wt, 2);
        mir_stripe_plan(LEN, 2, srep, 0, off, sz);
        int64_t diff = sz[0]>sz[1] ? sz[0]-sz[1] : sz[1]-sz[0];
        snprintf(d,sizeof d,"%lld vs %lld, diff %lld B",
                 (long long)sz[0],(long long)sz[1],(long long)diff);
        ck(diff <= 8192, "equal weights still split ~evenly", d);
    }

    /* 4. Chunk 0 lands on the ROUTED replica. The caller relies on this so the
     *    hash still spreads first-chunk load; losing it would silently point
     *    every first chunk at replica 0. */
    {
        int wt[3] = {40, 30, 30};
        int srep[3] = {0,1,2};
        set_weights(wt, 3);
        int64_t s0[MIR_REPS], o0[MIR_REPS];
        mir_stripe_plan(LEN, 3, srep, 0, o0, s0);
        mir_stripe_plan(LEN, 3, srep, 1, off, sz);
        snprintf(d,sizeof d,"rep0 first=%lld  rep1 first=%lld",
                 (long long)s0[0],(long long)sz[0]);
        ck(s0[0] != sz[0], "rotating rep changes which weight chunk 0 takes", d);
    }

    /* 5. No leg may exceed len, and none may be negative. Guards the
     *    remainder branch, which is the one place a bad weight could produce
     *    a wild size. */
    {
        int wt[3] = {1, 1, 254};         /* deliberately lopsided */
        int srep[3] = {0,1,2};
        set_weights(wt, 3);
        mir_stripe_plan(LEN, 3, srep, 0, off, sz);
        int sane=1;
        for(int i=0;i<3;i++) if(sz[i]<0 || sz[i]>LEN) sane=0;
        snprintf(d,sizeof d,"%lld / %lld / %lld",
                 (long long)sz[0],(long long)sz[1],(long long)sz[2]);
        ck(sane, "no stripe is negative or larger than len", d);
    }

    printf(failures ? "\nmirror stripe split: %d FAILURE(S)\n"
                    : "\nmirror stripe split: ok\n", failures);
    return failures ? 1 : 0;
}
