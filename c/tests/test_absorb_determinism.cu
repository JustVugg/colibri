/* The batched and ragged MLA absorb kernels must reproduce their own output.
 *
 * This is NOT a claim that the engine is run-to-run reproducible -- it is not,
 * and docs/METAL-M5MAX-PERF-REPORT.md:54 says so: floating-point
 * non-associativity in parallel reductions makes decode output vary, which is
 * expected and benign.  These two kernels are a narrower case.  Each block
 * reduces a fixed number of elements in a fixed tree order with no atomics and
 * no cross-block combining, so one launch with one set of inputs performs one
 * fixed sequence of floating-point operations.  Two such launches that disagree
 * have not reassociated anything; they have read different values, which means
 * a thread saw shared memory another thread had not finished with.
 *
 * The specific hazard: both kernels reduce twice through one shared array
 * `red[]`.  Every thread reads red[0] for the softmax maximum, then every
 * thread overwrites red[tid] with its partial exponential sum.  With no barrier
 * between, thread 0's write can land before another warp has read the maximum,
 * and that warp exponentiates against a partial SUM where a MAX belongs.  The
 * project's Metal backend already carries the matching barrier in the matching
 * place -- c/backend_metal.mm:325, `float mx=red[0]; threadgroup_barrier(...)`.
 *
 * Two things have to line up for it to reach the output, and only one of them
 * is grid oversubscription.  With S rows the launch is dim3(H,S) blocks against
 * a fixed number of SMs, and once blocks are queued rather than resident their
 * warps interleave far more freely; that is the coarse lever, and this test
 * pulls it in both halves.  But interleaving alone is not enough.  Whether a
 * corrupted read changes ctx[] depends sharply on nt, the token count the block
 * reduces, because thread 0 is the racing writer and its own loop decides how
 * far ahead it gets:
 *
 *   - nt <= 32: only warp 0 has any t to exponentiate, and thread 0 is in it.
 *     Writer and victims advance together, so nothing is exposed.
 *   - 32 < nt <= blockDim.x (256): thread 0's exponential loop is a SINGLE
 *     iteration, so it reaches `red[tid]=local` while later warps are still at
 *     `float mx=red[0]`.  Threads with tid < nt then exponentiate against a
 *     partial sum, and the wrong value survives into the output.  This is the
 *     band the defect lives in.
 *   - nt > blockDim.x: thread 0 must make two or more passes before it writes,
 *     which usually gives every warp time to take its copy of the maximum
 *     first.  That protection is partial, and it decays with nt instead of
 *     switching on at 256: measured, cells at nt 384 and nt 400 each fired on
 *     29 of 29 launches in 1 of 3 runs and on none in the other two.  Above the
 *     band corruption is intermittent from run to run rather than absent, and
 *     it thins out gradually across roughly nt 257..512.
 *
 * The sweep below is built around that band rather than in ignorance of it.  At
 * S=128 the batch cells T=256..320 put every block's nt at or below 320 with
 * most of it inside 33..256, and they differ on essentially every launch; the
 * T=513 cell puts every nt at 386..513, above blockDim.x, and is comparatively
 * quiet -- it differs on 0..3 launches of 29, that 3 being the maximum seen
 * over 30 runs of a probe dedicated to that cell, and it has not once fired on
 * all 29.  Expect roughly 4 of the 5 batch cells to fire on an unpatched
 * kernel, not 5.  T=513 is kept on purpose as a negative control: a fixture
 * change that made the whole sweep fire would mean the harness had moved, not
 * the kernel.  Being a control it is a statement about degree, not about
 * immunity -- the graded fall-off above is why the stated bound is 0..3 of 29
 * and not 0.
 *
 * A bounded band is not the same as a corner case.  33..256 tokens of context
 * is an ordinary operating point -- a short prompt, an early decode step, the
 * first page of a KV cache -- so the affected region sits squarely inside
 * normal use.  That is what makes this worth a barrier rather than a footnote.
 *
 * What this oracle structurally cannot see: it compares launches against each
 * other, so a defect that is wrong the same way on every launch -- one output
 * element pinned to a constant, say -- reproduces perfectly and passes here
 * (measured: rc=0 over 6 runs against exactly that mutation).  Absolute
 * correctness is covered by test_backend_cuda and test_ragged_attention, which
 * run earlier in the same `make cuda-test` recipe; this test only adds the
 * run-to-run axis and should not be read as covering more.
 *
 * The deterministic companion check, when the toolkit is present, is
 *     compute-sanitizer --tool=racecheck ./absorb_determinism_test
 * which reports the hazard on every run of an unpatched kernel and none on a
 * fixed one.  Set COLI_ABSORB_DET_ITERS to change the launch count per cell; it
 * is clamped to 2..1000 so that a stray value cannot hang the recipe, and a
 * value that is not a decimal integer (surrounding whitespace aside) is
 * reported on stderr before the default is used.
 */
#include "../backend_cuda.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef COLI_KV_PAGE_TOKENS
#define COLI_KV_PAGE_TOKENS 64
#endif

static const int H = 8, Q = 8, R = 4, V = 8, K = 64;
static const int ROWS = H * (Q + V), D = H * V, O = 5;
static const int SB = 128;   /* batch rows */
static const int SR = 128;   /* ragged rows */

/* Corruption is reported against the output tensor's own dynamic range.  The
 * absolute delta is not a property of the defect -- it scales with whatever the
 * inputs happen to be -- so an absolute number here would say more about the
 * fixture than about the bug. */
struct Verdict { int differing; double worst_pct; };

static double range_of(const std::vector<float> &v) {
    double lo = v[0], hi = v[0];
    for (size_t i = 1; i < v.size(); i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
    return hi - lo;
}

static bool all_finite(const std::vector<float> &v) {
    for (size_t i = 0; i < v.size(); i++) if (!std::isfinite(v[i])) return false;
    return true;
}

/* A determinism oracle is blind to a kernel that has stopped producing output:
 * an all-zero result reproduces perfectly, and every delta measured against it
 * is a delta of nothing.  Require the reference launch to carry real dynamic
 * range before treating its reproducibility as evidence of anything.  This also
 * keeps range_of()'s value safe as score()'s divisor.
 *
 * Non-finite output is rejected on its own message rather than folded into the
 * degenerate-range one: an all-NaN launch also yields a NaN range, and calling
 * that a narrow dynamic range sends whoever is debugging it looking for a flat
 * output when what they have is an arithmetic blow-up. */
static bool live_range(const std::vector<float> &v, double range, const char *what) {
    if (!all_finite(v)) {
        std::fprintf(stderr, "FAIL: %s reference output contains non-finite values "
                     "(NaN or infinity); the kernel produced no usable numbers, so a "
                     "determinism check against it would be vacuous\n", what);
        return false;
    }
    if (range > 1e-6) return true;
    std::fprintf(stderr, "FAIL: %s reference output is degenerate (dynamic range %.3g); "
                 "a determinism check against it would be vacuous\n", what, range);
    return false;
}

static void score(const std::vector<float> &first, const std::vector<float> &got,
                  double range, Verdict *v) {
    if (std::memcmp(first.data(), got.data(), first.size() * sizeof(float)) == 0) return;
    v->differing++;
    double worst = 0;
    for (size_t i = 0; i < first.size(); i++) {
        double d = std::fabs((double)got[i] - (double)first[i]);
        if (d > worst) worst = d;
    }
    double pct = 100.0 * worst / range;
    if (pct > v->worst_pct) v->worst_pct = pct;
}

int main(void) {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) return 77;

    /* Bounded at BOTH ends: this loop count is the whole runtime of the test,
     * and the test runs inside `make cuda-test`.  An unbounded value does not
     * fail loudly, it hangs the recipe.  strtol rather than atoi because atoi
     * on an out-of-range value is undefined -- and measured, an overflowing
     * count hangs exactly like an in-range huge one, so it is not benign. */
    const char *env = std::getenv("COLI_ABSORB_DET_ITERS");
    long want = 30;
    if (env && *env) {   /* an empty value means the same as unset, and is not reported */
        char *end = nullptr;
        errno = 0;
        long got = std::strtol(env, &end, 10);
        bool ok = (end != env) && (errno != ERANGE);
        /* strtol already skips leading whitespace, so trailing whitespace is
         * accepted as well rather than letting " 50" and "50 " disagree for a
         * reason the caller cannot see from the outside. */
        if (ok) {
            while (*end && std::isspace((unsigned char)*end)) end++;
            ok = (*end == '\0');
        }
        if (ok) {
            want = got;
        } else {
            /* Falling back in silence is how a typo turns into a shorter run
             * that still reports PASS.  Name the variable and the value. */
            std::fprintf(stderr, "WARN: COLI_ABSORB_DET_ITERS=\"%s\" is not a decimal "
                         "integer in range; using the default of %ld launches per cell\n",
                         env, want);
        }
    }
    if (want < 2) want = 2;
    if (want > 1000) want = 1000;
    int iters = (int)want;

    std::vector<float> w((size_t)ROWS * K), p((size_t)O * D);
    std::vector<float> q((size_t)(SB > SR ? SB : SR) * H * (Q + R));
    for (size_t i = 0; i < w.size(); i++) w[i] = ((int)(i % 11) - 5) * .07f;
    for (size_t i = 0; i < p.size(); i++) p[i] = ((int)(i % 7) - 3) * .09f;
    for (size_t i = 0; i < q.size(); i++) q[i] = ((int)(i % 13) - 6) * .05f;

    ColiCudaTensor *tw = nullptr, *tp = nullptr;
    if (!coli_cuda_tensor_upload(&tw, w.data(), nullptr, 0, K, ROWS, dev) ||
        !coli_cuda_tensor_upload(&tp, p.data(), nullptr, 0, D, O, dev)) {
        std::fprintf(stderr, "FAIL: coli_cuda_tensor_upload failed on device %d "
                     "(absorb weights %dx%d, projection %dx%d)\n", dev, ROWS, K, O, D);
        return 1;
    }

    int rc = 0;

    /* ---- batch: nt = T-S+s+1, so each launch spans an SB-token window ---- */
    static const int sweep[] = { 256, 272, 288, 320, 513 };
    static const int NCELL = (int)(sizeof(sweep) / sizeof(sweep[0]));
    Verdict batch = { 0, 0.0 };
    for (int c = 0; c < NCELL; c++) {
        int T = sweep[c];
        std::vector<float> lat((size_t)T * K), rope((size_t)T * R);
        for (size_t i = 0; i < lat.size(); i++) lat[i] = ((int)(i % 9) - 4) * .08f;
        for (size_t i = 0; i < rope.size(); i++) rope[i] = ((int)(i % 5) - 2) * .06f;
        std::vector<float> first((size_t)SB * H * V), got((size_t)SB * H * V);
        if (!coli_cuda_attention_absorb_batch(tw, first.data(), q.data(), lat.data(),
                                              rope.data(), SB, H, Q, R, V, K, T, .2f)) {
            std::fprintf(stderr, "FAIL: coli_cuda_attention_absorb_batch reference launch "
                         "failed at T=%d\n", T);
            rc = 2; goto done;
        }
        {
            double range = range_of(first);
            if (!live_range(first, range, "batch")) { rc = 5; goto done; }
            for (int it = 1; it < iters; it++) {
                if (!coli_cuda_attention_absorb_batch(tw, got.data(), q.data(), lat.data(),
                                                      rope.data(), SB, H, Q, R, V, K, T, .2f)) {
                    std::fprintf(stderr, "FAIL: coli_cuda_attention_absorb_batch failed at "
                                 "T=%d iteration %d\n", T, it);
                    rc = 2; goto done;
                }
                score(first, got, range, &batch);
            }
        }
    }

    /* ---- ragged: nt = lengths[s], uneven across the blocks of one launch ---- */
    {
        std::vector<int> lengths(SR);
        for (int s = 0; s < SR; s++) lengths[s] = 129 + s;      /* 129 .. 256 */
        const int T = 129 + SR - 1;
        const int PAGES = (T + COLI_KV_PAGE_TOKENS - 1) / COLI_KV_PAGE_TOKENS;
        const int CAP = PAGES * COLI_KV_PAGE_TOKENS;
        std::vector<std::vector<float> > l(SR), r(SR);
        std::vector<const float *> lp(SR), rp(SR);
        std::vector<const void *> keys(SR);
        for (int s = 0; s < SR; s++) {
            l[s].resize((size_t)CAP * K);
            r[s].resize((size_t)CAP * R);
            for (size_t i = 0; i < l[s].size(); i++) l[s][i] = ((int)((i + s * 3) % 9) - 4) * .08f;
            for (size_t i = 0; i < r[s].size(); i++) r[s][i] = ((int)((i + s) % 5) - 2) * .06f;
            lp[s] = l[s].data(); rp[s] = r[s].data(); keys[s] = &l[s];
        }
        std::vector<float> first((size_t)SR * O), got((size_t)SR * O);
        Verdict ragged = { 0, 0.0 };
        if (!coli_cuda_attention_project_ragged(tw, tp, first.data(), q.data(), keys.data(),
                                                lp.data(), rp.data(), lengths.data(),
                                                SR, H, Q, R, V, K, T, .2f)) {
            std::fprintf(stderr, "FAIL: coli_cuda_attention_project_ragged reference launch "
                         "failed at T=%d\n", T);
            rc = 3; goto done;
        }
        {
            double range = range_of(first);
            if (!live_range(first, range, "ragged")) { rc = 5; goto done; }
            for (int it = 1; it < iters; it++) {
                if (!coli_cuda_attention_project_ragged(tw, tp, got.data(), q.data(), keys.data(),
                                                        lp.data(), rp.data(), lengths.data(),
                                                        SR, H, Q, R, V, K, T, .2f)) {
                    std::fprintf(stderr, "FAIL: coli_cuda_attention_project_ragged failed at "
                                 "T=%d iteration %d\n", T, it);
                    rc = 3; goto done;
                }
                score(first, got, range, &ragged);
            }
        }
        std::printf("absorb_determinism iters_per_cell=%d S_batch=%d S_ragged=%d "
                    "batch_differing=%d/%d batch_worst_pct_of_range=%.3g "
                    "ragged_differing=%d/%d ragged_worst_pct_of_range=%.3g\n",
                    iters, SB, SR,
                    batch.differing, NCELL * (iters - 1), batch.worst_pct,
                    ragged.differing, iters - 1, ragged.worst_pct);
        if (batch.differing || ragged.differing) {
            /* The counts above are the evidence for the line below, so flush
             * them before crossing streams; piped to one file, a block-buffered
             * stdout would otherwise land after this. */
            std::fflush(stdout);
            std::fprintf(stderr, "FAIL: identical inputs produced different output; a thread "
                         "read shared red[] while another was still writing it\n");
            rc = 4;
        }
    }

done:
    coli_cuda_tensor_free(tw);
    coli_cuda_tensor_free(tp);
    coli_cuda_shutdown();
    return rc;
}
