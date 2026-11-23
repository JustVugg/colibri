/* COLI_CUDA_SK_PART: gemv_i8_w4's split partial sums in a buffer a stream (gemv_sk_part) must give
 * the bits the stream-ordered allocator gives (cudaMallocAsync + cudaFreeAsync a call).
 *
 * The switch is read once into a static (gemv_sk_part_on), so each setting runs in its own process:
 * the parent re-runs this binary with COLI_CUDA_SK_PART=0 and =1 (COLI_CUDA_GEMV_W4 and
 * COLI_CUDA_GEMV_SK on), each child writes every y it computed to a file, and the parent compares
 * the two files byte for byte. Each child also checks its y against a double-precision reference,
 * so two equal wrong answers do not pass.
 *
 * What a child runs, all through coli_cuda_tensor_gemm_async on int8 rows (fmt 1):
 *   1. shapes whose outputs are too few to fill the multiprocessors twice, so the inputs split
 *      (checked: gemv_i8_sk_splits > 1), and one that does not split, at 1 to 8 rows;
 *   2. the same call on nine more streams: the eight buffers are taken by the first eight
 *      streams, the rest go to the allocator, every stream's y the same bits;
 *   3. a call that needs more than its stream's buffer. The buffer is 8 MiB or the first call's
 *      need, and a split call's partials are at most kz x 8 rows x O floats with kz x O below
 *      128 x the multiprocessors, so no real shape gets there on a card of fewer than 2048 of
 *      them; the test lowers the recorded capacity (this file includes backend_cuda.cu) and
 *      checks the call takes the allocator, leaves the buffer where it was and gives the same y;
 *   4. the call captured in a CUDA graph and replayed twice: the same y as the eager call (with
 *      the switch on the graph keeps the buffer's address; off, it holds the allocator's nodes);
 *   5. a stream's first call made while the stream is captured: no buffer then (a cudaMalloc would
 *      end the capture), the allocator's nodes, the eager call's bits; the next eager call makes it.
 * With the switch on the child also checks that every buffer stays at its first address; off,
 * that no buffer was made.
 *
 * CUDA only: HIP keeps its own buffer a stream, grown when needed, whatever the switch says. */

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../backend_cuda.cu"

#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
int main(void) {
    std::printf("gemv_sk_part test: CUDA only (HIP's buffer a stream is not behind COLI_CUDA_SK_PART), skipped\n");
    return 0;
}
#else

#ifdef _WIN32
static int sk_setenv(const char *n, const char *v) { return _putenv_s(n, v); }
#else
#include <sys/wait.h>
static int sk_setenv(const char *n, const char *v) { return setenv(n, v, 1); }
#endif

#define SK_CHILD_FLAG "--sk-part-child"

static uint32_t sk_rng = 0x2468ace1u;
static uint32_t sk_next(void) { sk_rng = sk_rng * 1664525u + 1013904223u; return sk_rng; }

struct SkShape { int I, O; int want_split; };

static int sk_fail = 0;
#define SK_CHECK(c, ...) do { if (!(c)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); sk_fail = 1; } } while (0)

/* y = x . w^T * scale, in double, against the device's floats: |error| within 1e-4 of the sum of
 * the terms' magnitudes (float sums over up to 12288 inputs). */
static void sk_check_ref(const char *what, const float *y, const float *x, const int8_t *w, const float *sc, int I, int O, int S) {
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double a = 0, m = 0;
        for (int i = 0; i < I; i++) { double t = (double)x[(size_t)s * I + i] * w[(size_t)o * I + i]; a += t; m += std::fabs(t); }
        a *= sc[o]; m *= std::fabs(sc[o]);
        double d = std::fabs((double)y[(size_t)s * O + o] - a);
        if (d > 1e-4 * m + 1e-30) {
            std::printf("FAIL: %s row %d output %d: %.9g, reference %.9g\n", what, s, o, (double)y[(size_t)s * O + o], a);
            sk_fail = 1; return;
        }
    }
}

struct SkMat {
    int I, O; std::vector<int8_t> w; std::vector<float> sc; ColiCudaTensor *t = nullptr;
};

static int sk_make(SkMat &m, int I, int O, int dev) {
    m.I = I; m.O = O; m.w.resize((size_t)I * O); m.sc.resize(O);
    for (auto &v : m.w) v = (int8_t)((int)(sk_next() >> 24) - 128);
    for (auto &v : m.sc) v = 1e-3f * (1.f + (float)(sk_next() >> 20) / 4096.f);
    return coli_cuda_tensor_upload(&m.t, m.w.data(), m.sc.data(), 1, I, O, dev);
}

/* One call on stream st: y copied back, appended to out, checked against the reference. */
static int sk_run(const char *what, SkMat &m, const float *x_dev, const std::vector<float> &xh, float *y_dev, int S,
                  cudaStream_t st, std::vector<float> &out) {
    if (cudaMemsetAsync(y_dev, 0xff, (size_t)S * m.O * sizeof(float), st) != cudaSuccess) return 0;
    if (!coli_cuda_tensor_gemm_async(m.t, y_dev, x_dev, S, st)) { std::printf("FAIL: %s: gemm_async refused\n", what); sk_fail = 1; return 0; }
    std::vector<float> y((size_t)S * m.O);
    if (cudaMemcpyAsync(y.data(), y_dev, y.size() * sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess) { std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(cudaGetLastError())); sk_fail = 1; return 0; }
    sk_check_ref(what, y.data(), xh.data(), m.w.data(), m.sc.data(), m.I, m.O, S);
    out.insert(out.end(), y.begin(), y.end());
    return 1;
}

static int sk_slot(DeviceContext *ctx, cudaStream_t st) {
    for (int i = 0; i < 8; i++) if (ctx->skp[i].p && ctx->skp[i].st == st) return i;
    return -1;
}
static int sk_slots(DeviceContext *ctx) { int n = 0; for (int i = 0; i < 8; i++) n += ctx->skp[i].p != nullptr; return n; }

static int sk_child(const char *path) {
    const int on = gemv_sk_part_on();
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { std::printf("FAIL: coli_cuda_init\n"); return 2; }
    DeviceContext *ctx = find_ctx(dev);
    if (!ctx || !select_ctx(ctx)) { std::printf("FAIL: no device context\n"); return 2; }
    std::printf("  [child SK_PART=%d] %d multiprocessors\n", on, ctx->sms);

    /* splits while ceil(O / 32) < 2 x the multiprocessors; the last shape's outputs fill the card */
    const int sms2 = 2 * ctx->sms;
    const int o_big = ((sms2 * 32 + 1023) / 1024 + 1) * 1024;
    const SkShape shapes[] = { {10240, 320, 1}, {10240, 333, 1}, {12288, 256, 1}, {8192, 512, 1}, {6144, 2048, sms2 > 64}, {5120, o_big, 0} };
    const int rows[] = {1, 2, 3, 4, 5, 8};
    const int IMAX = 12288, OMAX = o_big;

    std::vector<float> xh((size_t)8 * IMAX);
    for (auto &v : xh) v = (float)((int)(sk_next() >> 16) - 32768) / 32768.f;
    float *x_dev = nullptr, *y_dev = nullptr;
    if (cudaMalloc(&x_dev, xh.size() * sizeof(float)) != cudaSuccess || cudaMalloc(&y_dev, (size_t)8 * OMAX * sizeof(float)) != cudaSuccess) return 2;

    std::vector<float> out;
    cudaStream_t s0;
    if (cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking) != cudaSuccess) return 2;
    void *first = nullptr;

    /* 1. the shapes, 1 to 8 rows */
    SkMat hc;   /* 10240 x 320: Qwen3.8's hyper-connection read, kept for the cases below */
    for (const SkShape &sh : shapes) {
        SkMat m;
        if (!sk_make(m, sh.I, sh.O, dev)) { std::printf("FAIL: upload %d x %d\n", sh.I, sh.O); return 2; }
        const unsigned bw = (unsigned)((sh.O + GEMV_I8_ROWS * GEMV_I8_OPW - 1) / (GEMV_I8_ROWS * GEMV_I8_OPW));
        const int ks = gemv_i8_sk_splits(ctx, sh.I, bw);
        SK_CHECK((ks > 1) == (sh.want_split != 0), "%d x %d: %d splits, expected %s", sh.I, sh.O, ks,
                 sh.want_split ? "more than one" : "one");
        std::vector<float> xs(xh.begin(), xh.begin() + (size_t)8 * sh.I);
        if (cudaMemcpy(x_dev, xs.data(), xs.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
        for (int S : rows) {
            char what[96]; std::snprintf(what, sizeof what, "%d x %d, %d rows, %d splits", sh.I, sh.O, S, ks);
            sk_run(what, m, x_dev, xs, y_dev, S, s0, out);
            if (on && ks > 1) {
                int k = sk_slot(ctx, s0);
                SK_CHECK(k >= 0, "%s: no buffer for the stream", what);
                if (k >= 0) {
                    if (!first) first = ctx->skp[k].p;
                    SK_CHECK(ctx->skp[k].p == first, "%s: the stream's buffer moved", what);
                    SK_CHECK(ctx->skp[k].cap >= ((size_t)8 << 20), "%s: buffer of %zu bytes", what, ctx->skp[k].cap);
                }
            }
        }
        if (sh.I == 10240 && sh.O == 320) hc = m; else coli_cuda_tensor_free(m.t);
    }
    std::vector<float> xs(xh.begin(), xh.begin() + (size_t)8 * hc.I);
    if (cudaMemcpy(x_dev, xs.data(), xs.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 2;

    /* 2. nine more streams: seven buffers left, then the allocator; every stream the same bits */
    cudaStream_t ss[9];
    std::vector<float> ref; sk_run("10240 x 320, 4 rows, first stream", hc, x_dev, xs, y_dev, 4, s0, ref);
    out.insert(out.end(), ref.begin(), ref.end());
    for (int i = 0; i < 9; i++) {
        if (cudaStreamCreateWithFlags(&ss[i], cudaStreamNonBlocking) != cudaSuccess) return 2;
        std::vector<float> y; char what[64]; std::snprintf(what, sizeof what, "10240 x 320, 4 rows, stream %d", i + 1);
        sk_run(what, hc, x_dev, xs, y_dev, 4, ss[i], y);
        SK_CHECK(y.size() == ref.size() && std::memcmp(y.data(), ref.data(), y.size() * sizeof(float)) == 0, "%s: not the first stream's bits", what);
        out.insert(out.end(), y.begin(), y.end());
    }
    SK_CHECK(sk_slots(ctx) == (on ? 8 : 0), "%d buffers after ten streams, expected %d", sk_slots(ctx), on ? 8 : 0);
    if (on) SK_CHECK(sk_slot(ctx, ss[7]) < 0 && sk_slot(ctx, ss[8]) < 0, "the ninth and tenth streams have buffers");

    /* 3. a call that needs more than its stream's buffer: the allocator, the buffer left as it was */
    {
        int k = sk_slot(ctx, s0); size_t cap = 0; void *p = nullptr;
        if (on) { SK_CHECK(k >= 0, "no buffer for the first stream"); if (k >= 0) { p = ctx->skp[k].p; cap = ctx->skp[k].cap; ctx->skp[k].cap = 64; } }
        /* a byte pattern in the buffer's first 64 KiB, which the call must leave alone */
        const size_t probe = 64 << 10;
        if (p && cudaMemsetAsync(p, 0x5a, probe, s0) != cudaSuccess) return 2;
        std::vector<float> y; sk_run("10240 x 320, 8 rows, past the buffer", hc, x_dev, xs, y_dev, 8, s0, y);
        if (on && k >= 0) {
            SK_CHECK(ctx->skp[k].p == p && ctx->skp[k].cap == 64, "the buffer was replaced by a call that did not fit");
            std::vector<unsigned char> b(probe);
            if (cudaMemcpy(b.data(), p, probe, cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
            size_t i = 0; while (i < probe && b[i] == 0x5a) i++;
            SK_CHECK(i == probe, "a call that did not fit wrote into the buffer (byte %zu)", i);
            ctx->skp[k].cap = cap;
        }
        out.insert(out.end(), y.begin(), y.end());
        std::vector<float> y2; sk_run("10240 x 320, 8 rows, in the buffer", hc, x_dev, xs, y_dev, 8, s0, y2);
        SK_CHECK(y == y2, "8 rows: the allocator's call and the buffer's call differ");
    }

    /* 4. captured in a graph and replayed twice: the eager call's bits */
    {
        cudaStream_t gs = ss[0];   /* has its buffer with the switch on */
        std::vector<float> eager; sk_run("10240 x 320, 3 rows, eager", hc, x_dev, xs, y_dev, 3, gs, eager);
        void *p = on ? ctx->skp[sk_slot(ctx, gs)].p : nullptr;
        cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
        int ok = cudaStreamBeginCapture(gs, cudaStreamCaptureModeGlobal) == cudaSuccess;
        int launched = ok && coli_cuda_tensor_gemm_async(hc.t, y_dev, x_dev, 3, gs);
        ok = cudaStreamEndCapture(gs, &g) == cudaSuccess && ok && launched;
        ok = ok && cudaGraphInstantiate(&ge, g, 0) == cudaSuccess;
        SK_CHECK(ok, "graph capture of the call: %s", cudaGetErrorString(cudaGetLastError()));
        for (int rep = 0; ok && rep < 2; rep++) {
            std::vector<float> y(eager.size());
            ok = cudaMemsetAsync(y_dev, 0xff, y.size() * sizeof(float), gs) == cudaSuccess &&
                 cudaGraphLaunch(ge, gs) == cudaSuccess &&
                 cudaMemcpyAsync(y.data(), y_dev, y.size() * sizeof(float), cudaMemcpyDeviceToHost, gs) == cudaSuccess &&
                 cudaStreamSynchronize(gs) == cudaSuccess;
            SK_CHECK(ok, "graph replay %d: %s", rep, cudaGetErrorString(cudaGetLastError()));
            SK_CHECK(!ok || y == eager, "graph replay %d: not the eager call's bits", rep);
            out.insert(out.end(), y.begin(), y.end());
        }
        if (on) SK_CHECK(ctx->skp[sk_slot(ctx, gs)].p == p, "the stream's buffer moved under the graph");
        if (ge) cudaGraphExecDestroy(ge);
        if (g) cudaGraphDestroy(g);

        /* 5. the first call on a stream made while that stream is captured (thread-local mode, as the
         * engine's captures): no buffer may be made then (a cudaMalloc would end the capture), the
         * call takes the allocator's nodes; the next eager call makes the buffer. One slot is
         * emptied first so the stream could get one. */
        if (on) {
            int k = sk_slot(ctx, ss[6]);
            SK_CHECK(k >= 0, "the seventh stream has no buffer");
            if (k >= 0) { cudaDeviceSynchronize(); cudaFree(ctx->skp[k].p); ctx->skp[k].p = nullptr; ctx->skp[k].cap = 0; ctx->skp[k].st = nullptr; }
        }
        cudaStream_t cs;
        if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) return 2;
        g = nullptr; ge = nullptr;
        ok = cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        launched = ok && coli_cuda_tensor_gemm_async(hc.t, y_dev, x_dev, 3, cs);
        ok = cudaStreamEndCapture(cs, &g) == cudaSuccess && ok && launched;
        ok = ok && cudaGraphInstantiate(&ge, g, 0) == cudaSuccess;
        SK_CHECK(ok, "graph capture of a stream's first call: %s", cudaGetErrorString(cudaGetLastError()));
        (void)cudaGetLastError();
        SK_CHECK(sk_slot(ctx, cs) < 0, "a buffer was made while the stream was captured");
        std::vector<float> y(eager.size());
        int rok = ok && cudaMemsetAsync(y_dev, 0xff, y.size() * sizeof(float), cs) == cudaSuccess &&
                  cudaGraphLaunch(ge, cs) == cudaSuccess &&
                  cudaMemcpyAsync(y.data(), y_dev, y.size() * sizeof(float), cudaMemcpyDeviceToHost, cs) == cudaSuccess &&
                  cudaStreamSynchronize(cs) == cudaSuccess;
        SK_CHECK(rok, "replay of the graph of a stream's first call: %s", cudaGetErrorString(cudaGetLastError()));
        SK_CHECK(!rok || y == eager, "the graph of a stream's first call: not the eager call's bits");
        out.insert(out.end(), y.begin(), y.end());
        if (ge) cudaGraphExecDestroy(ge);
        if (g) cudaGraphDestroy(g);
        std::vector<float> y3; sk_run("10240 x 320, 3 rows, after the capture", hc, x_dev, xs, y_dev, 3, cs, y3);
        SK_CHECK(y3 == eager, "the call after the capture: not the eager call's bits");
        if (on) SK_CHECK(sk_slot(ctx, cs) >= 0, "no buffer for the stream after an eager call");
        cudaStreamDestroy(cs);
    }

    coli_cuda_tensor_free(hc.t);
    for (int i = 0; i < 9; i++) cudaStreamDestroy(ss[i]);
    cudaStreamDestroy(s0);
    cudaFree(x_dev); cudaFree(y_dev);

    FILE *f = std::fopen(path, "wb");
    if (!f || std::fwrite(out.data(), sizeof(float), out.size(), f) != out.size()) { std::printf("FAIL: write %s\n", path); if (f) std::fclose(f); return 2; }
    std::fclose(f);
    std::printf("  [child SK_PART=%d] %zu outputs, %s\n", on, out.size(), sk_fail ? "FAILED" : "ok");
    return sk_fail ? 1 : 0;
}

static int sk_spawn(const char *self, const char *val, const char *path) {
    if (sk_setenv("COLI_CUDA_SK_PART", val)) return -1;
    char cmd[2048];
#ifdef _WIN32
    std::snprintf(cmd, sizeof cmd, "\"\"%s\" %s \"%s\"\"", self, SK_CHILD_FLAG, path);   /* cmd.exe strips the outer quotes */
#else
    std::snprintf(cmd, sizeof cmd, "'%s' %s '%s'", self, SK_CHILD_FLAG, path);
#endif
    std::fflush(stdout);
    int rc = std::system(cmd);
#ifndef _WIN32
    if (rc != -1 && WIFEXITED(rc)) rc = WEXITSTATUS(rc); else rc = -1;
#endif
    return rc;
}

static int sk_read(const char *path, std::vector<unsigned char> &b) {
    FILE *f = std::fopen(path, "rb"); if (!f) return 0;
    unsigned char buf[65536]; size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
    std::fclose(f); return 1;
}

int main(int argc, char **argv) {
    if (argc == 3 && !std::strcmp(argv[1], SK_CHILD_FLAG)) return sk_child(argv[2]);
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("FAIL: no CUDA device\n"); return 1; }
    if (sk_setenv("COLI_CUDA_GEMV_W4", "1") || sk_setenv("COLI_CUDA_GEMV_SK", "1")) return 1;
    const char *p0 = "gemv_sk_part_off.bin", *p1 = "gemv_sk_part_on.bin";
    int r0 = sk_spawn(argv[0], "0", p0), r1 = sk_spawn(argv[0], "1", p1);
    std::vector<unsigned char> b0, b1;
    int got = sk_read(p0, b0) && sk_read(p1, b1);
    std::remove(p0); std::remove(p1);
    if (r0 != 0 || r1 != 0 || !got) { std::printf("FAIL: children exited %d (SK_PART=0) and %d (SK_PART=1)\n", r0, r1); return 1; }
    if (b0.size() != b1.size() || b0.empty() || std::memcmp(b0.data(), b1.data(), b0.size())) {
        size_t i = 0; while (i < b0.size() && i < b1.size() && b0[i] == b1[i]) i++;
        std::printf("FAIL: COLI_CUDA_SK_PART=1 changes the bits (%zu and %zu bytes, first difference at float %zu)\n", b0.size(), b1.size(), i / 4);
        return 1;
    }
    std::printf("gemv_sk_part: COLI_CUDA_SK_PART=0 and =1 give the same %zu floats, bit for bit\n", b0.size() / 4);
    return 0;
}
#endif
