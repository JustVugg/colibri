/* test_sycl_xmx.cpp — coli_sycl_xmx.cpp's GEMM family on a device, against a CPU reference.
 *
 * Synthetic weights in the tier's formats (fmt 1 int8 rows, fmt 2 int4 rows, fmt 4 int4 per 64
 * with the nibbles signed as the device holds them, and the experts' planar int4) with f32
 * activations; the dense GEMM, the gate/up GLU pair, the int8-activation variant (a_s8) and the
 * routed experts (phase 0 and 1 over a CPU-made sort) run on the selected device and are
 * compared element by element with a double-precision reference that rounds the activations
 * to f16 once and stages the weights as the kernels do (int4 per gs: the product with the group's
 * scale rounded to f16): the tolerance is then 1e-3 of the row's largest magnitude, judged a
 * row at a time (row 0 carries the f16-saturation sentinel, 65504 x w, orders above the other
 * rows: one tolerance over the whole output would clear every other row of anything), the
 * f32 summation order's room (a wrong nibble, scale or row index is off by whole units). The
 * distance from a pure f32 reference (no f16 rounding) is reported, not judged. Every shape
 * also prints its time, GB/s (weights, scales, activations, outputs moved once) and TFLOP/s.
 *
 * Device: --device N or COLI_GPUS=N (the runtime's GPU order; --list prints it with the profile
 * and PCI address of each), the first GPU otherwise. --small: the tiny shapes only (run that
 * first on a card that has not run this before: a hung kernel wedges a card on this host).
 * --no-xmx: the plain tile (has_xmx forced 0), --no-s8 / --no-moe / --no-glu skip a family,
 * --reps N the timed repetitions (5). COLI_SYCL_XMX_TM=16 is read by the unit under test.
 * Exits 1 on any mismatch, 2 when no device is found.
 *
 * Build (the basekit container, AOT for pvc and bmg; JIT with -fsycl-targets=spir64):
 *   icpx -fsycl -std=c++17 -O2 -fp-model=precise -fsycl-targets=spir64_gen \
 *        -Xsycl-target-backend=spir64_gen "-device pvc,bmg" -I.. -I../.. \
 *        tests/test_sycl_xmx.cpp coli_sycl_xmx.cpp coli_sycl_profile.cpp coli_sycl_profile_rules.cpp \
 *        coli_sycl_stubs.cpp -o test_sycl_xmx
 * (from c/sycl; the -I.. for coli_sycl.h, the -I../.. for cuda_chain.h). */
#include "../coli_sycl.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Opts { int device = -1; bool list = false, small = false, no_xmx = false, s8 = true, moe = true, glu = true, moe_bench = false; int reps = 5; };

/* a small deterministic generator */
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed ? seed : 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float uniform() { return (next() >> 8) * (1.f / 16777216.f); }          /* [0, 1) */
    float sym() { return uniform() * 2.f - 1.f; }                            /* (-1, 1) */
    int ranged(int lo, int hi) { return lo + (int)(next() % (uint32_t)(hi - lo + 1)); }
};

/* what the kernels do to an activation: f16, held to the f16 range */
float h16(float f) {
    if (f != f) return f;
    f = std::fmin(std::fmax(f, -65504.f), 65504.f);
    return (float)sycl::half(f);
}

/* ---- synthetic weights in the device's layouts ------------------------------------------ */
struct Weights {
    int fmt, gs, K, N, planar;
    std::vector<uint8_t> w;     /* the codes, the device's bytes */
    std::vector<float> sc;      /* fmt 1, 2: N; fmt 4: N x ng */
    std::vector<int8_t> code;   /* the plain codes [N][K], for the reference */
    std::vector<float> stg, ext;   /* [N][K]: the weight as the kernel stages it in f16 (fmt 4: the product
                                      with the group's scale, rounded; fmt 1, 2: the code, the row's scale
                                      after the sum), and the plain value code x scale (the f32 reference) */
    int ng() const { return fmt == 4 ? (K + gs - 1) / gs : 1; }
    float staged(int n, int k) const { return stg[(size_t)n * K + k]; }
    float row_scale(int n) const { return fmt == 4 ? 1.f : sc[n]; }
    double exact(int n, int k) const { return ext[(size_t)n * K + k]; }
    void fill_ref() {
        stg.resize(code.size()); ext.resize(code.size());
        for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) {
            const size_t i = (size_t)n * K + k; const float c = (float)code[i];
            if (fmt == 4) { const float s = sc[(size_t)n * ng() + k / gs]; stg[i] = (float)sycl::half(c * s); ext[i] = c * s; }
            else { stg[i] = c; ext[i] = c * sc[n]; }
        }
    }
};
Weights make_weights(int fmt, int gs, int K, int N, int planar, uint32_t seed) {
    Weights W; W.fmt = fmt; W.gs = gs; W.K = K; W.N = N; W.planar = planar;
    Rng r(seed);
    W.code.resize((size_t)N * K);
    for (size_t i = 0; i < W.code.size(); i++) W.code[i] = (int8_t)(fmt == 1 ? r.ranged(-127, 127) : r.ranged(-8, 7));
    if (fmt == 4) { W.sc.resize((size_t)N * W.ng()); for (float &s : W.sc) s = 0.002f + 0.03f * r.uniform(); }
    else { W.sc.resize(N); for (float &s : W.sc) s = (fmt == 1 ? 0.0005f : 0.01f) + 0.02f * r.uniform(); }
    if (fmt == 1) { W.w.resize((size_t)N * K); for (size_t i = 0; i < W.w.size(); i++) W.w[i] = (uint8_t)W.code[i]; }
    else if (!planar) {   /* int4 pairs: weight 2j low nibble, 2j + 1 high, signed (offset_to_signed_s4's result) */
        W.w.assign((size_t)N * K / 2, 0);
        for (int n = 0; n < N; n++) for (int k = 0; k < K; k += 2) {
            unsigned lo = (unsigned)W.code[(size_t)n * K + k] & 15u, hi = (unsigned)W.code[(size_t)n * K + k + 1] & 15u;
            W.w[((size_t)n * K + k) / 2] = (uint8_t)(lo | (hi << 4));
        }
    } else {   /* planar: in each 64 weights' 32 bytes, byte j holds weight j low and j + 32 high, both v + 8 */
        W.w.assign((size_t)N * K / 2, 0);
        for (int n = 0; n < N; n++) for (int b = 0; b < K / 64; b++) for (int j = 0; j < 32; j++) {
            unsigned lo = (unsigned)(W.code[(size_t)n * K + b * 64 + j] + 8), hi = (unsigned)(W.code[(size_t)n * K + b * 64 + 32 + j] + 8);
            W.w[(size_t)n * K / 2 + b * 32 + j] = (uint8_t)(lo | (hi << 4));
        }
    }
    W.fill_ref();
    return W;
}

/* ---- the device side ------------------------------------------------------------------ */
template <typename T> T *to_dev(sycl::queue &q, const std::vector<T> &v) {
    T *p = sycl::malloc_device<T>(v.size() ? v.size() : 1, q);
    if (v.size()) q.memcpy(p, v.data(), v.size() * sizeof(T)).wait();
    return p;
}
struct DevWeights { uint8_t *w; float *sc; };
DevWeights upload(sycl::queue &q, const Weights &W) { return DevWeights{to_dev(q, W.w), to_dev(q, W.sc)}; }

/* ---- the comparison ----------------------------------------------------------------------- */
struct Cmp { double max_abs = 0, max_ref = 0, max_abs_f32 = 0; long bad = 0, n = 0; };
void cmp_one(Cmp &c, float got, double ref, double ref32, double tol_abs) {
    double e = std::fabs((double)got - ref);
    c.n++;
    if (!(e <= tol_abs) || got != got) c.bad++;
    if (e > c.max_abs) c.max_abs = e;
    if (std::fabs(ref) > c.max_ref) c.max_ref = std::fabs(ref);
    double e32 = std::fabs((double)got - ref32);
    if (e32 > c.max_abs_f32) c.max_abs_f32 = e32;
}
const double REL_TOL = 1e-3;   /* of the output's largest magnitude: the f32 summation order's room */

int fails = 0;
void report(const char *what, const Cmp &c, double ms, double bytes, double flops) {
    bool ok = c.bad == 0;
    if (!ok) fails++;
    printf("  %s %-56s max abs %.3g (%.2g of max %.4g) vs f16-staged ref; vs f32 ref %.3g; %d bad of %ld",
           ok ? "ok  " : "FAIL", what, c.max_abs, c.max_ref > 0 ? c.max_abs / c.max_ref : 0.0, c.max_ref, c.max_abs_f32, (int)c.bad, c.n);
    if (ms > 0) printf(" | %.3f ms %.1f GB/s %.2f TFLOP/s", ms, bytes / ms * 1e-6, flops / ms * 1e-9);
    printf("\n");
}

/* ---- the dense GEMM and the GLU pair ---------------------------------------------------------- */
/* C = A x W^T (or the GLU of two), checked on `check_rows` rows (every row when M is small: the
 * reference is a double loop on the host) */
void test_dense(sycl::queue &q, const coli_sycl_profile &p, const Opts &o, int fmt, int gs, int M, int K, int N, bool glu, bool s8, uint32_t seed) {
    Weights Wg = make_weights(fmt, gs, K, N, 0, seed), Wu = glu ? make_weights(fmt, gs, K, N, 0, seed + 77) : Weights{};
    Rng r(seed + 5);
    std::vector<float> A((size_t)M * K);
    for (float &a : A) a = r.sym() * 4.f;
    A[0] = 70000.f;                                  /* a value past f16's range: held to 65504 */
    DevWeights dg = upload(q, Wg), du = glu ? upload(q, Wu) : DevWeights{nullptr, nullptr};
    float *dA = to_dev(q, A), *dC = sycl::malloc_device<float>((size_t)M * N, q);
    q.memset(dC, 0, (size_t)M * N * sizeof(float)).wait();
    coli_sycl_gemm g; memset(&g, 0, sizeof g);
    g.C = dC; g.ldc = N; g.A = dA; g.lda = K; g.a_f16 = 0; g.a_s8 = s8 ? 1 : 0;
    g.W = dg.w; g.scales = dg.sc; g.fmt = fmt; g.gs = gs; g.M = M; g.K = K; g.N = N;
    char what[128];
    snprintf(what, sizeof what, "%s fmt %d%s M %d K %d N %d%s", glu ? "glu  " : "dense", fmt, fmt == 4 ? " g64" : "", M, K, N, s8 ? " s8" : "");
    if (!coli_sycl_xmx_gemm_ok(p, fmt, gs, K, N)) { printf("  skip %s: _ok declines\n", what); return; }
    int took = glu ? coli_sycl_xmx_gemm_glu(p, q, g, du.w, du.sc) : coli_sycl_xmx_gemm(p, q, g);
    if (!took) { printf("  FAIL %s: not taken\n", what); fails++; return; }
    q.wait();
    std::vector<float> C((size_t)M * N);
    q.memcpy(C.data(), dC, C.size() * sizeof(float)).wait();
    /* the reference: the activations as the kernel staged them */
    const bool s8_taken = s8 && (fmt == 1 || fmt == 2) && p.has_xmx && (p.xmx_shapes & COLI_SYCL_XMX_INT8);
    std::vector<float> Ah((size_t)M * K); std::vector<float> a_scale(M, 1.f);
    for (int m = 0; m < M; m++) {
        if (s8_taken) {
            float mx = 0.f; for (int k = 0; k < K; k++) mx = std::fmax(mx, std::fabs(A[(size_t)m * K + k]));
            a_scale[m] = mx > 0.f ? mx / 127.f : 1.f;
            for (int k = 0; k < K; k++) { float v = std::rint(A[(size_t)m * K + k] / a_scale[m]); Ah[(size_t)m * K + k] = std::fmin(std::fmax(v, -127.f), 127.f); }
        } else for (int k = 0; k < K; k++) Ah[(size_t)m * K + k] = h16(A[(size_t)m * K + k]);
    }
    const int check_rows = M <= 96 ? M : 64;
    std::vector<double> row_max(check_rows, 0.0);   /* a row's largest magnitude: its tolerance (row 0 the sentinel's) */
    std::vector<double> ref((size_t)check_rows * N), ref32((size_t)check_rows * N); std::vector<int> rows(check_rows);
    for (int i = 0; i < check_rows; i++) rows[i] = check_rows == M ? i : (int)((long)i * (M - 1) / (check_rows - 1));
    for (int i = 0; i < check_rows; i++) {
        const int m = rows[i];
        for (int n = 0; n < N; n++) {
            double sg = 0, su = 0, sg32 = 0, su32 = 0;
            for (int k = 0; k < K; k++) {
                double a = Ah[(size_t)m * K + k], a32 = A[(size_t)m * K + k];
                sg += a * Wg.staged(n, k); sg32 += a32 * Wg.exact(n, k);
                if (glu) { su += a * Wu.staged(n, k); su32 += a32 * Wu.exact(n, k); }
            }
            sg *= Wg.row_scale(n) * a_scale[m]; su *= glu ? Wu.row_scale(n) * a_scale[m] : 1.0;
            double v = glu ? (sg / (1.0 + std::exp(-sg))) * su : sg, v32 = glu ? (sg32 / (1.0 + std::exp(-sg32))) * su32 : sg32;
            ref[(size_t)i * N + n] = v; ref32[(size_t)i * N + n] = v32;
            if (std::fabs(v) > row_max[i]) row_max[i] = std::fabs(v);
        }
    }
    Cmp c;
    for (int i = 0; i < check_rows; i++) for (int n = 0; n < N; n++)
        cmp_one(c, C[(size_t)rows[i] * N + n], ref[(size_t)i * N + n], ref32[(size_t)i * N + n], REL_TOL * row_max[i] + 1e-6);
    /* the time: the call again, reps times, the queue drained between them */
    double ms = 0;
    if (o.reps > 0) {
        (void)(glu ? coli_sycl_xmx_gemm_glu(p, q, g, du.w, du.sc) : coli_sycl_xmx_gemm(p, q, g)); q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < o.reps; i++) (void)(glu ? coli_sycl_xmx_gemm_glu(p, q, g, du.w, du.sc) : coli_sycl_xmx_gemm(p, q, g));
        q.wait();
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / o.reps;
    }
    const double wbytes = (double)(Wg.w.size() + Wg.sc.size() * 4) * (glu ? 2 : 1);
    report(what, c, ms, wbytes + (double)M * K * 4 + (double)M * N * 4, 2.0 * M * N * K * (glu ? 2 : 1));
    sycl::free(dg.w, q); sycl::free(dg.sc, q); if (glu) { sycl::free(du.w, q); sycl::free(du.sc, q); }
    sycl::free(dA, q); sycl::free(dC, q);
}

/* ---- the routed experts ---------------------------------------------------------------- */
/* E experts of gate/up [I][D] and down [D][I] (fmt 4 g64, pair or planar layout), S rows of D,
 * Kp picks a row (a few -1), the sort as moe_sort_* would leave it, phase 0 then 1 */
void test_moe(sycl::queue &q, const coli_sycl_profile &p, const Opts &o, int E, int S, int Kp, int D, int I, int planar, uint32_t seed) {
    char what[128];
    snprintf(what, sizeof what, "moe E %d S %d K %d D %d I %d%s", E, S, Kp, D, I, planar ? " planar" : "");
    Rng r(seed);
    std::vector<Weights> Wg, Wu, Wd;
    std::vector<CcExpert> tab(E);
    std::vector<void *> dev_ptrs;
    for (int e = 0; e < E; e++) {
        Wg.push_back(make_weights(4, 64, D, I, planar, seed + 3 * e)); Wu.push_back(make_weights(4, 64, D, I, planar, seed + 3 * e + 1));
        Wd.push_back(make_weights(4, 64, I, D, planar, seed + 3 * e + 2));
        DevWeights g = upload(q, Wg[e]), u = upload(q, Wu[e]), d = upload(q, Wd[e]);
        tab[e] = CcExpert{g.w, u.w, d.w, g.sc, u.sc, d.sc, planar, 0};
        dev_ptrs.insert(dev_ptrs.end(), {g.w, g.sc, u.w, u.sc, d.w, d.sc});
    }
    const int A = S * Kp;
    std::vector<float> x((size_t)S * D); for (float &v : x) v = r.sym() * 3.f;
    std::vector<int> ids((size_t)A);
    for (int a = 0; a < A; a++) ids[a] = r.uniform() < 0.05f ? -1 : r.ranged(0, E - 1);
    /* the sort (moe_sort_count_k / scan_k / place_k): counts, bases, tiles of 64, the order */
    const int T = (A + 63) / 64 + (E < A ? E : A), TR = 64;
    std::vector<int> cnt(E, 0), base(E + 1, 0), te(T, 0), tr(T, 0), order(A, -1), posof(A, -1), cur(E, 0);
    for (int a = 0; a < A; a++) if (ids[a] >= 0) cnt[ids[a]]++;
    int ntile = 0;
    for (int e = 0; e < E; e++) {
        base[e + 1] = base[e] + cnt[e];
        for (int rr = 0; rr < cnt[e] && ntile < T; rr += TR) { te[ntile] = e; tr[ntile] = rr; ntile++; }
    }
    for (int a = 0; a < A; a++) if (ids[a] >= 0) { int pos = base[ids[a]] + cur[ids[a]]++; order[pos] = a; posof[a] = pos; }
    const int placed = base[E];
    CcMoe m; memset(&m, 0, sizeof m);
    m.S = S; m.K = Kp; m.D = D; m.I = I; m.fmt = 4; m.gs = 64; m.act = 0; m.parts = 1; m.E = E; m.planar = planar;
    m.x_off = 0; m.x_row = D; m.t_off = 0;
    if (!coli_sycl_xmx_moe_ok(p, m)) { printf("  skip %s: _ok declines\n", what); return; }
    CcExpert *dtab = to_dev(q, tab);
    float *dx = to_dev(q, x);
    std::vector<int> ntile_v(1, ntile);
    coli_sycl_moe_sort sort; sort.order = to_dev(q, order); sort.base = to_dev(q, base); sort.te = to_dev(q, te); sort.tr = to_dev(q, tr);
    sort.ntile = to_dev(q, ntile_v); sort.T = T; sort.TR = TR;
    const size_t hid_floats = (size_t)A * (I + D);
    float *dhid = sycl::malloc_device<float>(hid_floats, q);
    q.memset(dhid, 0, hid_floats * sizeof(float)).wait();
    if (!coli_sycl_xmx_moe(p, q, dtab, m, sort, dx, dhid, 0)) { printf("  FAIL %s: phase 0 not taken\n", what); fails++; return; }
    if (!coli_sycl_xmx_moe(p, q, dtab, m, sort, dx, dhid, 1)) { printf("  FAIL %s: phase 1 not taken\n", what); fails++; return; }
    q.wait();
    std::vector<float> hid(hid_floats);
    q.memcpy(hid.data(), dhid, hid_floats * sizeof(float)).wait();
    /* the reference over the placed slots: phase 0 from x's f16 rows, phase 1 from the device's
     * phase 0 rows rounded to f16 (the kernel's input; its own rounding, so phase 1 is judged
     * on its own arithmetic) */
    Cmp c0, c1; double max0 = 0, max1 = 0;
    const int checked = o.moe_bench && placed > 64 ? 64 : placed;
    std::vector<int> positions(checked);
    for (int j = 0; j < checked; j++) positions[j] = checked == placed ? j : (int)((long)j * (placed - 1) / (checked - 1));
    if (checked != placed) printf("  reference: %d evenly spaced slots of %d, including first and last\n", checked, placed);
    std::vector<double> h0((size_t)checked * I), h0_32((size_t)checked * I), h1((size_t)checked * D), h1_32((size_t)checked * D);
    for (int j = 0; j < checked; j++) {
        const int pos = positions[j];
        const int a = order[pos], e = ids[a], s = a / Kp;
        for (int n = 0; n < I; n++) {
            double sg = 0, su = 0, sg32 = 0, su32 = 0;
            for (int k = 0; k < D; k++) {
                double av = h16(x[(size_t)s * D + k]), a32 = x[(size_t)s * D + k];
                sg += av * Wg[e].staged(n, k); su += av * Wu[e].staged(n, k);
                sg32 += a32 * Wg[e].exact(n, k); su32 += a32 * Wu[e].exact(n, k);
            }
            double v = (sg / (1.0 + std::exp(-sg))) * su;
            h0[(size_t)j * I + n] = v; h0_32[(size_t)j * I + n] = (sg32 / (1.0 + std::exp(-sg32))) * su32;
            if (std::fabs(v) > max0) max0 = std::fabs(v);
        }
    }
    for (int j = 0; j < checked; j++) for (int n = 0; n < I; n++)
        cmp_one(c0, hid[(size_t)positions[j] * I + n], h0[(size_t)j * I + n], h0_32[(size_t)j * I + n], REL_TOL * max0 + 1e-6);
    for (int j = 0; j < checked; j++) {
        const int pos = positions[j];
        const int a = order[pos], e = ids[a];
        for (int n = 0; n < D; n++) {
            double sd = 0, sd32 = 0;
            for (int k = 0; k < I; k++) {
                sd += (double)h16(hid[(size_t)pos * I + k]) * Wd[e].staged(n, k);
                sd32 += h0_32[(size_t)j * I + k] * Wd[e].exact(n, k);
            }
            h1[(size_t)j * D + n] = sd; h1_32[(size_t)j * D + n] = sd32;
            if (std::fabs(sd) > max1) max1 = std::fabs(sd);
        }
    }
    for (int j = 0; j < checked; j++) for (int n = 0; n < D; n++)
        cmp_one(c1, hid[(size_t)A * I + (size_t)positions[j] * D + n], h1[(size_t)j * D + n], h1_32[(size_t)j * D + n], REL_TOL * max1 + 1e-6);
    double ms0 = 0, ms1 = 0;
    if (o.reps > 0) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < o.reps; i++) coli_sycl_xmx_moe(p, q, dtab, m, sort, dx, dhid, 0);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < o.reps; i++) coli_sycl_xmx_moe(p, q, dtab, m, sort, dx, dhid, 1);
        q.wait();
        auto t2 = std::chrono::steady_clock::now();
        ms0 = std::chrono::duration<double, std::milli>(t1 - t0).count() / o.reps;
        ms1 = std::chrono::duration<double, std::milli>(t2 - t1).count() / o.reps;
    }
    /* bytes: every live expert's matrices once, the rows in and out */
    int live = 0; for (int e = 0; e < E; e++) live += cnt[e] > 0;
    const double gu_bytes = (double)live * 2 * (Wg[0].w.size() + Wg[0].sc.size() * 4) + (double)placed * D * 4 + (double)placed * I * 4;
    const double dn_bytes = (double)live * (Wd[0].w.size() + Wd[0].sc.size() * 4) + (double)placed * (I + D) * 4;
    char w0[160], w1[160];
    snprintf(w0, sizeof w0, "%s gate/up (%d slots, %d tiles)", what, placed, ntile);
    snprintf(w1, sizeof w1, "%s down", what);
    report(w0, c0, ms0, gu_bytes, 4.0 * placed * I * D);
    report(w1, c1, ms1, dn_bytes, 2.0 * placed * I * D);
    for (void *ptr : dev_ptrs) sycl::free(ptr, q);
    sycl::free(dtab, q); sycl::free(dx, q); sycl::free(dhid, q);
    sycl::free((void *)sort.order, q); sycl::free((void *)sort.base, q); sycl::free((void *)sort.te, q); sycl::free((void *)sort.tr, q);
    sycl::free((void *)sort.ntile, q);
}

} // namespace

int main(int argc, char **argv) {
    Opts o;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--list") o.list = true;
        else if (s == "--small") o.small = true;
        else if (s == "--no-xmx") o.no_xmx = true;
        else if (s == "--no-s8") o.s8 = false;
        else if (s == "--no-moe") o.moe = false;
        else if (s == "--no-glu") o.glu = false;
        else if (s == "--moe-bench") o.moe_bench = true;
        else if (s == "--device" && i + 1 < argc) o.device = atoi(argv[++i]);
        else if (s == "--reps" && i + 1 < argc) o.reps = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--list] [--small] [--no-xmx] [--no-s8] [--no-moe] [--no-glu] [--moe-bench] [--device N] [--reps N]\n", argv[0]); return 1; }
    }
    if (o.device < 0) { const char *e = getenv("COLI_GPUS"); if (e && *e) o.device = atoi(e); }
    std::vector<sycl::device> gpus;
    try { gpus = sycl::device::get_devices(sycl::info::device_type::gpu); } catch (const sycl::exception &e) { fprintf(stderr, "no GPU: %s\n", e.what()); }
    if (gpus.empty()) { fprintf(stderr, "no GPU device found\n"); return 2; }
    if (o.list) {
        for (size_t i = 0; i < gpus.size(); i++) {
            coli_sycl_profile p = coli_sycl_profile_detect(gpus[i]);
            coli_sycl_profile_print(stdout, (int)i, p);
        }
        return 0;
    }
    if (o.device < 0) o.device = 0;
    if (o.device >= (int)gpus.size()) { fprintf(stderr, "device %d: only %zu GPUs\n", o.device, gpus.size()); return 2; }
    sycl::queue q(gpus[o.device], sycl::property::queue::in_order());
    coli_sycl_profile p = coli_sycl_profile_detect(q.get_device());
    if (o.no_xmx) p.has_xmx = 0;
    printf("device %d: %s (%s), profile %s, sg default %d, slm %zu KiB, xmx %s (shapes 0x%x)%s\n", o.device, p.device_name, p.pci_address,
           p.name, p.sg_default, p.slm_bytes / 1024, p.has_xmx ? "on" : "off", p.xmx_shapes, o.no_xmx ? " [--no-xmx: the plain tile]" : "");
    const char *tm = getenv("COLI_SYCL_XMX_TM");
    if (tm && *tm) printf("COLI_SYCL_XMX_TM=%s\n", tm);
    /* said plainly when the plain tile runs instead of the matrix engines, so a pass here never
     * reads as the engines having run */
    if (!o.no_xmx && !coli_sycl_xmx_supported(p, COLI_SYCL_DT_F16))
        printf("WARNING: the profile has no f16 8x16x16 shape (or COLI_SYCL_XMX=0): every case below runs the plain f32 tile, not the matrix engines\n");

    if (o.moe_bench) {
        printf("Qwen3.6 grouped MoE geometry, uniform synthetic routes; 64 sampled reference slots\n");
        test_moe(q, p, o, 256, 256, 8, 2048, 512, 1, 51);
        test_moe(q, p, o, 256, 2048, 8, 2048, 512, 1, 52);
        printf("%s\n", fails ? "FAILED" : "PASSED");
        return fails ? 1 : 0;
    }

    /* the tiny shapes first: a tail of rows, a tail of outputs, one row, every format */
    printf("small shapes\n");
    test_dense(q, p, o, 1, 0, 16, 64, 16, false, false, 11);
    test_dense(q, p, o, 2, 0, 5, 128, 80, false, false, 12);
    test_dense(q, p, o, 4, 64, 1, 64, 64, false, false, 13);
    test_dense(q, p, o, 4, 64, 100, 256, 200, false, false, 14);
    test_dense(q, p, o, 4, 32, 33, 192, 48, false, false, 15);
    if (o.glu) test_dense(q, p, o, 4, 64, 20, 128, 96, true, false, 16);
    if (o.glu) test_dense(q, p, o, 1, 0, 70, 192, 130, true, false, 17);
    if (o.s8) { test_dense(q, p, o, 1, 0, 17, 128, 48, false, true, 18); test_dense(q, p, o, 2, 0, 65, 256, 72, true, true, 19); }
    if (o.moe) { test_moe(q, p, o, 4, 24, 2, 128, 192, 0, 21); test_moe(q, p, o, 3, 40, 2, 192, 128, 1, 22); }
    if (!o.small) {
        printf("prompt shapes\n");
        test_dense(q, p, o, 1, 0, 64, 2048, 2048, false, false, 31);
        test_dense(q, p, o, 1, 0, 256, 2048, 8192, false, false, 32);
        test_dense(q, p, o, 1, 0, 512, 4096, 4096, false, false, 33);
        test_dense(q, p, o, 2, 0, 256, 2048, 2048, false, false, 34);
        test_dense(q, p, o, 4, 64, 512, 2048, 512, false, false, 35);
        if (o.glu) test_dense(q, p, o, 4, 64, 512, 2048, 512, true, false, 36);
        if (o.glu) test_dense(q, p, o, 1, 0, 256, 2048, 512, true, false, 37);
        if (o.s8) { test_dense(q, p, o, 1, 0, 256, 2048, 8192, false, true, 38); test_dense(q, p, o, 1, 0, 512, 4096, 4096, false, true, 39); }
        if (o.moe) { test_moe(q, p, o, 16, 128, 4, 1024, 512, 0, 41); test_moe(q, p, o, 16, 128, 4, 1024, 512, 1, 42); }
    }
    printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
