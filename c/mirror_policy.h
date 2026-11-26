/* mirror_policy.h: which drive serves an expert read when COLI_MODEL_MIRROR
 * lists copies of the model on other drives (#812).
 *
 * st.h holds the replica plumbing. st_mirror_add registers a copy, and
 * st_fd_rep / st_direct_fd_rep give the descriptor of the same file on
 * replica `rep` (0 = the primary). This header holds the policy on top of it:
 * the directory list, the weight of each drive (COLI_DISK_WEIGHTS, or a probe
 * at startup), and two ways to pick a drive for a read.
 *
 *   mirror_hash  One fixed drive per (layer, expert), the hash colibri.c's
 *                expert_route uses. Use it for buffered reads: a readahead
 *                hint and the read after it must go to the same file, and the
 *                page cache must not keep one expert twice.
 *   mirror_take, mirror_pick
 *                A weighted round-robin over the reads of one batch. The cursor
 *                continues from batch to batch, so no drive gets the extra read
 *                of every batch. Use it for O_DIRECT reads that the engine
 *                waits for as a group: each drive gets its share of every batch,
 *                so one drive does not hold the batch up. With equal weights a
 *                drive is less than one read from its share in any batch, with
 *                other weights less than two (tests/test_mirror_policy.c).
 *
 * The policy never changes the bytes of a read. st_mirror_add accepts a copy
 * only when its size and header are equal to the primary's, so the tokens do
 * not depend on the drive. An engine that does not call mirror_setup has a
 * zeroed policy, and every function below then gives the primary.
 *
 * The functions without shards (mirror_parse_weights, mirror_cuts,
 * mirror_order, mirror_hash, mirror_take, mirror_pick) are pure, so
 * tests/test_mirror_policy.c checks them without files or threads. */
#ifndef MIRROR_POLICY_H
#define MIRROR_POLICY_H

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "st.h"

#define MIRROR_REPS  (1 + ST_MAX_MIR)   /* drives, the primary counted */
#define MIRROR_HASH  256   /* the hash's values: 8 bits, as colibri.c's expert_route */
#define MIRROR_ROUND 240   /* round-robin period, a multiple of every drive count from 2 to 6 */

typedef struct {
    int nrep;                           /* drives, the primary counted. Below 2 = off */
    int cut[MIRROR_REPS];               /* drive r takes hash values [cut[r-1], cut[r]) */
    unsigned char order[MIRROR_ROUND];  /* the drive of each round-robin slot */
    _Atomic uint64_t next;              /* the next free round-robin slot */
    _Atomic int64_t bytes[MIRROR_REPS]; /* bytes read from each drive */
    _Atomic int64_t reads[MIRROR_REPS]; /* reads from each drive */
} MirrorPolicy;

/* COLI_DISK_WEIGHTS: one relative weight per drive, the primary first
 * ("9,3", "1,1,1"). A comma or a space separates two weights. Each weight
 * must be a number above 0 and not above 1e9, with nothing after it. Returns
 * 1 and fills w[0..nrep-1] when there are exactly nrep weights, else 0. */
static int mirror_parse_weights(const char *spec, int nrep, double *w) {
    if (!spec || nrep < 1 || nrep > MIRROR_REPS) return 0;
    int n = 0;
    const char *p = spec;
    for (;;) {
        while (*p == ',' || *p == ' ') p++;
        if (!*p) break;
        char *end = NULL;
        double v = strtod(p, &end);
        if (end == p || !(v > 0) || v > 1e9 || n >= nrep) return 0;
        if (*end && *end != ',' && *end != ' ') return 0;   /* "9GB" is not a weight */
        w[n++] = v;
        p = end;
    }
    return n == nrep;
}

/* The cumulative cuts of `slots` values in proportion to the weights: drive
 * r takes the values [cut[r-1], cut[r]). Each drive keeps at least one value,
 * and the last cut is `slots`. With slots = MIRROR_HASH and weights that do
 * not differ by more than about 250 times, the cuts are the same as the ones
 * colibri.c's mirror_setup computes. */
static void mirror_cuts(const double *w, int nrep, int slots, int *cut) {
    double total = 0, sum = 0;
    for (int r = 0; r < nrep; r++) total += w[r];
    int prev = 0;
    for (int r = 0; r < nrep; r++) {
        sum += w[r];
        int c = r == nrep - 1 ? slots : (int)(slots * sum / total + 0.5);
        if (c <= prev) c = prev + 1;                       /* a drive with no share */
        if (c > slots - (nrep - 1 - r))                    /* leave one for each later drive */
            c = slots - (nrep - 1 - r);
        cut[r] = prev = c;
    }
}

/* The round-robin order from cuts of MIRROR_ROUND: drive r gets
 * cut[r] - cut[r-1] of the slots. Each slot goes to the drive that is the most
 * behind its share, which is the smallest (count + 1) / share. A tie goes to
 * the lower drive. With equal weights the order is 0, 1, 2, 0, 1, 2 and so on,
 * without a break at the end of the period. */
static void mirror_order(const int *cut, int nrep, unsigned char *order) {
    int share[MIRROR_REPS], count[MIRROR_REPS] = {0};
    for (int r = 0; r < nrep; r++) share[r] = cut[r] - (r ? cut[r - 1] : 0);
    for (int i = 0; i < MIRROR_ROUND; i++) {
        int best = 0;
        for (int r = 1; r < nrep; r++)
            if ((int64_t)(count[r] + 1) * share[best] < (int64_t)(count[best] + 1) * share[r])
                best = r;
        order[i] = (unsigned char)best;
        count[best]++;
    }
}

static inline int mirror_on(const MirrorPolicy *p) { return p->nrep > 1; }

static double mirror_now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* The fixed drive of one expert. Every read of an expert and every hint
 * before it go to this drive. */
static inline int mirror_hash(const MirrorPolicy *p, int layer, int eid) {
    if (p->nrep < 2) return 0;
    uint32_t h = (uint32_t)layer * 2654435761u ^ (uint32_t)eid * 0x9E3779B9u;
    h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
    int hv = (int)(h & (MIRROR_HASH - 1)), r = 0;
    while (r < p->nrep - 1 && hv >= p->cut[r]) r++;
    return r;
}

/* Reserves n consecutive round-robin slots for a batch of n reads and returns
 * the first. Read q of the batch then goes to mirror_pick(p, first, q). The
 * cursor has 64 bits, so it does not wrap in practice. */
static inline uint64_t mirror_take(MirrorPolicy *p, int n) {
    if (p->nrep < 2 || n <= 0) return 0;
    return atomic_fetch_add_explicit(&p->next, (uint64_t)n, memory_order_relaxed);
}
static inline int mirror_pick(const MirrorPolicy *p, uint64_t first, int q) {
    if (p->nrep < 2) return 0;
    return p->order[(first + (uint64_t)q) % MIRROR_ROUND];
}

/* Counts one read of `bytes` from drive rep, for mirror_report. */
static inline void mirror_count(MirrorPolicy *p, int rep, int64_t bytes) {
    if (p->nrep < 2 || rep < 0 || rep >= p->nrep) return;
    atomic_fetch_add_explicit(&p->bytes[rep], bytes, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->reads[rep], 1, memory_order_relaxed);
}

/* One line with the reads and the bytes of each drive, for example
 * "[K3 MIRROR] reads: primary 312 (5.47 GB) | mirror1 311 (5.45 GB)". */
static void mirror_report(MirrorPolicy *p, const char *tag) {
    if (p->nrep < 2) return;
    fprintf(stderr, "%s reads:", tag);
    for (int r = 0; r < p->nrep; r++) {
        long long n = (long long)atomic_load_explicit(&p->reads[r], memory_order_relaxed);
        double gb = (double)atomic_load_explicit(&p->bytes[r], memory_order_relaxed) / 1e9;
        if (r) fprintf(stderr, " | mirror%d %lld (%.2f GB)", r, n, gb);
        else fprintf(stderr, " primary %lld (%.2f GB)", n, gb);
    }
    fprintf(stderr, "\n");
}

/* The read rate of one drive in GB/s, with the access pattern of an expert
 * engine: 8 parallel reads of 19 MiB at fixed places in the largest shard,
 * through the O_DIRECT twin when there is one (a buffered probe measures the
 * page cache). The same probe as colibri.c's mirror_probe_bw. Returns 0 when
 * the drive has no shard large enough or no byte arrives. */
static double mirror_probe(shards *S, int rep, const char *tag) {
    const int64_t blk = 19ll << 20;
    const int nb = 8;
    int big = -1;
    int64_t bsz = 0;
    for (int i = 0; i < S->nfd; i++) {
        if (st_fd_rep(S, S->fds[i], rep) < 0) continue;
        if (S->sizes[i] > bsz) { bsz = S->sizes[i]; big = i; }
    }
    if (big < 0 || bsz < blk * (nb + 1)) return 0;
    int dfd = st_direct_fd_rep(S, S->fds[big], rep);
    int fd = dfd >= 0 ? dfd : st_fd_rep(S, S->fds[big], rep);
    if (dfd < 0)
        fprintf(stderr, "%s no O_DIRECT on drive %d: the probe can read the page cache. "
                        "Set COLI_DISK_WEIGHTS for an accurate split\n", tag, rep);
    double t0 = mirror_now_s();
    int64_t tot = 0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:tot)
    for (int i = 0; i < nb; i++) {
        void *buf;
        if (!posix_memalign(&buf, 4096, (size_t)blk)) {
            int64_t off = (((bsz - blk) / nb) * i) & ~4095ll;   /* fixed, spread over the file */
            ssize_t got = pread(fd, buf, (size_t)blk, off);
            if (got > 0) tot += got;
            compat_aligned_free(buf);
        }
    }
    double dt = mirror_now_s() - t0;
    return dt > 0 && tot > 0 ? (double)tot / 1e9 / dt : 0;
}

/* Registers each directory of `dirs` (separated by ';' or ',') as a copy of
 * the shards in S. The function skips, with a message, a directory equal to
 * `primary`, a directory past ST_MAX_MIR, and a directory with no file equal
 * to the primary's. Returns the number of drives, the primary counted. */
static int mirror_add_dirs(shards *S, const char *dirs, const char *primary, const char *tag) {
    char list[4096];
    snprintf(list, sizeof list, "%s", dirs);
    int nrep = 1;
    for (char *p = list, *next; p; p = next) {
        next = strpbrk(p, ";,");
        if (next) *next++ = 0;
        while (*p == ' ') p++;
        for (size_t n = strlen(p); n && p[n - 1] == ' ';) p[--n] = 0;
        if (!*p) continue;
        if (primary && !strcmp(p, primary)) {
            fprintf(stderr, "%s %s is the model directory itself: ignored\n", tag, p);
        } else if (nrep >= MIRROR_REPS) {
            fprintf(stderr, "%s %s: more than %d mirrors: ignored\n", tag, p, ST_MAX_MIR);
        } else {
            int files = st_mirror_add(S, p);
            if (files > 0) {
                fprintf(stderr, "%s %s: %d of %d shards (drive %d)\n", tag, p, files, S->nfd, nrep);
                nrep++;
            } else {
                fprintf(stderr, "%s %s: no shard equal to the primary's: skipped\n", tag, p);
            }
        }
    }
    return nrep;
}

/* Sets up the policy for an engine. `dirs` is the engine's directory list
 * (COLI_MODEL_MIRROR, or a legacy alias the engine reads), `primary` the
 * model directory, `tag` the prefix of each log line. Call it after the shard
 * index is built and before any expert read. Without a usable copy the
 * policy stays off and every read goes to the primary. The weights come from
 * COLI_DISK_WEIGHTS. If that is unset or not valid, the probe measures each
 * drive, and if the probe fails, the split is even. */
static void mirror_setup(MirrorPolicy *p, shards *S, const char *dirs, const char *primary,
                         const char *tag) {
    memset(p, 0, sizeof *p);
    p->nrep = 1;
    p->cut[0] = MIRROR_HASH;
    if (!dirs || !*dirs) return;
    st_mirror_reset(S);
    int nrep = mirror_add_dirs(S, dirs, primary, tag);
    if (nrep < 2) {
        fprintf(stderr, "%s no usable mirror: all reads go to the primary drive\n", tag);
        return;
    }
    double w[MIRROR_REPS];
    const char *how = "COLI_DISK_WEIGHTS";
    const char *spec = getenv("COLI_DISK_WEIGHTS");
    if (!(spec && *spec && mirror_parse_weights(spec, nrep, w))) {
        if (spec && *spec)
            fprintf(stderr, "%s COLI_DISK_WEIGHTS='%s' is not %d positive weights, the primary "
                            "first: the probe sets the split\n", tag, spec, nrep);
        int ok = 1;
        for (int r = 0; r < nrep; r++) { w[r] = mirror_probe(S, r, tag); if (!(w[r] > 0)) ok = 0; }
        if (ok) {
            how = "startup probe";
            fprintf(stderr, "%s probe:", tag);
            for (int r = 0; r < nrep; r++) fprintf(stderr, "%s drive %d %.2f GB/s", r ? " |" : "", r, w[r]);
            fprintf(stderr, "\n");
        } else {
            how = "even: the probe gave no rate";
            for (int r = 0; r < nrep; r++) w[r] = 1;
        }
    }
    int round_cut[MIRROR_REPS];
    mirror_cuts(w, nrep, MIRROR_HASH, p->cut);
    mirror_cuts(w, nrep, MIRROR_ROUND, round_cut);
    mirror_order(round_cut, nrep, p->order);
    p->nrep = nrep;
    fprintf(stderr, "%s %d drives | read split", tag, nrep);
    for (int r = 0; r < nrep; r++)
        fprintf(stderr, "%s %.0f%%", r ? " /" : "", 100.0 * (round_cut[r] - (r ? round_cut[r - 1] : 0)) / MIRROR_ROUND);
    fprintf(stderr, " (%s)\n", how);
}

#endif /* MIRROR_POLICY_H */
