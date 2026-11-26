/* mirror_policy.h: the split of expert reads over COLI_MODEL_MIRROR copies.
 *
 *   - COLI_DISK_WEIGHTS: exactly one positive number per drive, else refused.
 *   - The hash's cuts: each drive keeps a share, the last cut is 256, and for
 *     ordinary weights they are colibri.c's (its algorithm written again below).
 *   - The hash: colibri.c's expert_route, and the primary when the policy is off.
 *   - The round-robin: each drive gets its share of the period. In every run of
 *     consecutive slots (any batch, any start) a drive is less than one read
 *     from its share with equal weights, and less than two with other weights.
 *   - The cursor continues from batch to batch.
 *   - mirror_setup on real files: weights, the model directory, a missing copy.
 * Fixture dirs are created in the current working directory and removed on exit. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../st.h"
#include "../mirror_policy.h"

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#define RMDIR(p) _rmdir(p)
#define SETENV(k, v) _putenv_s(k, v)
#define UNSETENV(k) _putenv_s(k, "")
#else
#include <sys/stat.h>
#include <unistd.h>
#define MKDIR(p) mkdir(p, 0777)
#define RMDIR(p) rmdir(p)
#define SETENV(k, v) setenv(k, v, 1)
#define UNSETENV(k) unsetenv(k)
#endif

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#define DIR_A "tmp_mpol_a"   /* primary */
#define DIR_B "tmp_mpol_b"   /* identical copy */
#define DIR_C "tmp_mpol_c"   /* second identical copy */
#define DIR_E "tmp_mpol_e"   /* empty: no shard */

/* colibri.c's mirror_setup cuts, written again: the reference for ordinary weights */
static void colibri_cuts(const double *wt, int nrep, int *cut) {
    double W = 0; for (int r = 0; r < nrep; r++) W += wt[r];
    int acc = 0; double cum = 0;
    for (int r = 0; r < nrep; r++) {
        cum += wt[r];
        int c = (int)(256.0 * cum / W + 0.5);
        if (c <= acc) c = acc + 1;
        if (c > 256) c = 256;
        cut[r] = c; acc = c;
    }
    cut[nrep - 1] = 256;
}

/* colibri.c's expert_route, written again */
static int colibri_route(const int *cut, int layer, int eid) {
    uint32_t h = (uint32_t)layer * 2654435761u ^ (uint32_t)eid * 0x9E3779B9u;
    h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
    int hv = (int)(h & 255), r = 0;
    while (hv >= cut[r]) r++;
    return r;
}

static int test_parse(void) {
    double w[MIRROR_REPS + 1];
    CHECK(mirror_parse_weights("9,3", 2, w) && w[0] == 9 && w[1] == 3);
    CHECK(mirror_parse_weights("1,1,1", 3, w) && w[2] == 1);
    CHECK(mirror_parse_weights(" 1.5, 0.5 ", 2, w) && w[0] == 1.5 && w[1] == 0.5);
    CHECK(mirror_parse_weights("4 6", 2, w) && w[1] == 6);           /* a space separates too */
    CHECK(mirror_parse_weights("2,1,", 2, w));                        /* a separator at the end */
    const char *bad[] = {"", "1", "1,1,1", "1,0", "0,1", "-1,1", "1,x", "9GB,3",
                         "nan,1", "inf,1", "1e10,1", ",", "1;2"};
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++)
        CHECK(!mirror_parse_weights(bad[i], 2, w));
    CHECK(!mirror_parse_weights(NULL, 2, w));
    CHECK(!mirror_parse_weights("1,1", 0, w));
    CHECK(!mirror_parse_weights("1,1,1,1,1,1", MIRROR_REPS + 1, w));
    w[2] = -7;                                   /* never written past nrep */
    CHECK(!mirror_parse_weights("1,2,3", 2, w) && w[2] == -7);
    return 0;
}

static int test_cuts(void) {
    const double sets[][MIRROR_REPS] = {
        {1, 1}, {1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1, 1}, {9, 3}, {9, 3, 3},
        {5.69, 5.03, 0.44}, {4.0, 6.3, 5.1}, {1, 2, 3, 4, 5}, {100, 1},
    };
    const int nsets[] = {2, 3, 4, 5, 2, 3, 3, 3, 5, 2};
    for (size_t s = 0; s < sizeof nsets / sizeof nsets[0]; s++) {
        int cut[MIRROR_REPS], ref[MIRROR_REPS], n = nsets[s];
        mirror_cuts(sets[s], n, MIRROR_HASH, cut);
        colibri_cuts(sets[s], n, ref);
        for (int r = 0; r < n; r++) CHECK(cut[r] == ref[r]);
        CHECK(cut[n - 1] == MIRROR_HASH);
    }
    int cut[MIRROR_REPS];
    mirror_cuts((const double[]){1, 1, 1}, 3, MIRROR_HASH, cut);
    CHECK(cut[0] == 85 && cut[1] == 171 && cut[2] == 256);
    mirror_cuts((const double[]){1, 1, 1}, 3, MIRROR_ROUND, cut);
    CHECK(cut[0] == 80 && cut[1] == 160 && cut[2] == 240);
    /* a drive 1e6 times slower still keeps one value (colibri.c's cuts give it none) */
    mirror_cuts((const double[]){1e6, 1, 1}, 3, MIRROR_HASH, cut);
    CHECK(cut[0] == 254 && cut[1] == 255 && cut[2] == 256);
    mirror_cuts((const double[]){1, 1e6, 1e6, 1e6, 1e6}, 5, MIRROR_ROUND, cut);
    CHECK(cut[0] >= 1);
    for (int r = 1; r < 5; r++) CHECK(cut[r] > cut[r - 1]);
    return 0;
}

static void policy_from(MirrorPolicy *p, const double *w, int n) {
    int round_cut[MIRROR_REPS];
    memset(p, 0, sizeof *p);
    p->nrep = n;
    mirror_cuts(w, n, MIRROR_HASH, p->cut);
    mirror_cuts(w, n, MIRROR_ROUND, round_cut);
    mirror_order(round_cut, n, p->order);
}

static int test_hash(void) {
    MirrorPolicy off;
    memset(&off, 0, sizeof off);
    CHECK(mirror_hash(&off, 3, 7) == 0 && !mirror_on(&off));
    const double w[] = {5.69, 5.03, 0.44};
    MirrorPolicy p;
    policy_from(&p, w, 3);
    CHECK(mirror_on(&p));
    int seen[3] = {0};
    for (int layer = 0; layer < 93; layer++)
        for (int eid = 0; eid < 896; eid++) {
            int r = mirror_hash(&p, layer, eid);
            CHECK(r == colibri_route(p.cut, layer, eid));
            seen[r]++;
        }
    CHECK(seen[0] > 0 && seen[1] > 0 && seen[2] > 0);
    return 0;
}

/* every run of 1 to MIRROR_ROUND consecutive slots, from every start */
static int window_ok(const MirrorPolicy *p, const double *w, int n, double bound) {
    int round_cut[MIRROR_REPS], share[MIRROR_REPS];
    mirror_cuts(w, n, MIRROR_ROUND, round_cut);
    for (int r = 0; r < n; r++) share[r] = round_cut[r] - (r ? round_cut[r - 1] : 0);
    int total[MIRROR_REPS] = {0};
    for (int i = 0; i < MIRROR_ROUND; i++) total[p->order[i]]++;
    for (int r = 0; r < n; r++) if (total[r] != share[r]) return 0;
    for (int start = 0; start < MIRROR_ROUND; start++) {
        int count[MIRROR_REPS] = {0};
        for (int len = 1; len <= MIRROR_ROUND; len++) {
            count[mirror_pick(p, (uint64_t)start, len - 1)]++;
            for (int r = 0; r < n; r++)
                if (fabs(count[r] - (double)len * share[r] / MIRROR_ROUND) >= bound) return 0;
        }
    }
    return 1;
}

static int test_round_robin(void) {
    MirrorPolicy p;
    for (int n = 2; n <= MIRROR_REPS; n++) {
        double w[MIRROR_REPS];
        for (int r = 0; r < n; r++) w[r] = 1;
        policy_from(&p, w, n);
        for (int i = 0; i < MIRROR_ROUND; i++) CHECK(p.order[i] == i % n);   /* no break at the end */
        CHECK(window_ok(&p, w, n, 1.0));
    }
    const double sets[][MIRROR_REPS] = {{1, 3}, {9, 3, 3}, {4.0, 6.3, 5.1}, {5.69, 5.03, 0.44},
                                        {1, 2, 3, 4, 5}, {1e6, 1, 1}};
    const int nsets[] = {2, 3, 3, 3, 5, 3};
    for (size_t s = 0; s < sizeof nsets / sizeof nsets[0]; s++) {
        policy_from(&p, sets[s], nsets[s]);
        CHECK(window_ok(&p, sets[s], nsets[s], 2.0));
    }
    /* the cursor continues: batches of 13, 5, 16 and 1 reads take the order's slots 0..34 */
    policy_from(&p, (const double[]){1, 1, 1}, 3);
    const int sizes[] = {13, 5, 16, 1};
    int slot = 0;
    for (int b = 0; b < 4; b++) {
        uint64_t first = mirror_take(&p, sizes[b]);
        CHECK(first == (uint64_t)slot);
        for (int q = 0; q < sizes[b]; q++, slot++) CHECK(mirror_pick(&p, first, q) == p.order[slot % MIRROR_ROUND]);
    }
    CHECK(mirror_take(&p, 0) == 0);              /* an empty batch takes no slot ... */
    CHECK(mirror_take(&p, 2) == (uint64_t)slot); /* ... and the next one continues */
    MirrorPolicy off;
    memset(&off, 0, sizeof off);
    CHECK(mirror_take(&off, 9) == 0 && mirror_pick(&off, 5, 3) == 0);
    return 0;
}

static int test_count(void) {
    MirrorPolicy p;
    policy_from(&p, (const double[]){1, 1}, 2);
    mirror_count(&p, 0, 100);
    mirror_count(&p, 1, 50);
    mirror_count(&p, 1, 50);
    mirror_count(&p, 2, 999);                    /* not a drive: not counted */
    CHECK(p.reads[0] == 1 && p.bytes[0] == 100 && p.reads[1] == 2 && p.bytes[1] == 100);
    MirrorPolicy off;
    memset(&off, 0, sizeof off);
    mirror_count(&off, 0, 100);
    CHECK(off.reads[0] == 0);
    return 0;
}

static int write_model(const char *dir) {
    char hdr[128], path[256];
    int hl = snprintf(hdr, sizeof(hdr), "{\"t0\":{\"dtype\":\"F32\",\"shape\":[8],\"data_offsets\":[0,32]}}");
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    uint64_t n = (uint64_t)hl;
    fwrite(&n, 8, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    float data[8] = {1, -2, 3.5f, 0, 42, -0.5f, 7, 8};
    fwrite(data, 4, 8, f);
    fclose(f);
    return 0;
}

static void cleanup(void) {
    const char *dirs[] = {DIR_A, DIR_B, DIR_C, DIR_E};
    for (int i = 0; i < 4; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/model.safetensors", dirs[i]);
        remove(path);
        RMDIR(dirs[i]);
    }
}

static int test_setup(void) {
    cleanup();
    CHECK(MKDIR(DIR_A) == 0 && MKDIR(DIR_B) == 0 && MKDIR(DIR_C) == 0 && MKDIR(DIR_E) == 0);
    CHECK(write_model(DIR_A) == 0 && write_model(DIR_B) == 0 && write_model(DIR_C) == 0);
    shards S;
    st_init(&S, DIR_A);
    st_tensor *t = st_find(&S, "t0");
    CHECK(t != NULL);
    MirrorPolicy p;

    mirror_setup(&p, &S, NULL, DIR_A, "[test]");                /* no list: off */
    CHECK(p.nrep == 1 && S.nrep == 0);

    UNSETENV("COLI_DISK_WEIGHTS");
    mirror_setup(&p, &S, DIR_E, DIR_A, "[test]");               /* no shard in the copy: off */
    CHECK(p.nrep == 1 && mirror_hash(&p, 1, 2) == 0);

    SETENV("COLI_DISK_WEIGHTS", "1,3");
    mirror_setup(&p, &S, DIR_A ";" DIR_E "; " DIR_B " ", DIR_A, "[test]");   /* the model dir: ignored */
    CHECK(p.nrep == 2 && S.nrep == 1 && p.cut[0] == 64 && p.cut[1] == 256);
    CHECK(st_fd_rep(&S, t->fd, 1) >= 0);

    SETENV("COLI_DISK_WEIGHTS", "1,1,1");
    mirror_setup(&p, &S, DIR_B "," DIR_C, DIR_A, "[test]");
    CHECK(p.nrep == 3 && S.nrep == 2);
    for (int i = 0; i < 9; i++) CHECK(p.order[i] == i % 3);

    /* weights that do not fit: the probe (the shard is too small to measure), then even */
    SETENV("COLI_DISK_WEIGHTS", "1,1");
    mirror_setup(&p, &S, DIR_B "," DIR_C, DIR_A, "[test]");
    CHECK(p.nrep == 3 && p.cut[0] == 85 && p.cut[1] == 171);
    CHECK(mirror_probe(&S, 1, "[test]") == 0);

    UNSETENV("COLI_DISK_WEIGHTS");
    cleanup();
    return 0;
}

int main(void) {
    if (test_parse() || test_cuts() || test_hash() || test_round_robin() || test_count() || test_setup()) {
        cleanup();
        return 1;
    }
    puts("mirror policy tests: ok");
    return 0;
}
