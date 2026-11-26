/* v41_amx.h -- V41_AMX: deepseek_v41's prompt-pass matrix products (rows >= V41_AMX_MIN, default 16) on the CPU's
 * AMX-BF16 tiles (Sapphire Rapids and later). Off in the code; deepseek_v41's platform defaults set V41_AMX=2 on a CPU
 * with AMX-BF16 (an explicit V41_AMX, or COLI_PLATFORM_DEFAULTS=0, wins). NOT the AVX2 path's bits: the products are summed in the
 * tile unit's order, and the activations go in as bf16 -- so a run with it is judged by a quality gate, not by
 * identity.
 *   V41_AMX=1  activations rounded to bf16 (round to nearest even): ~3 significant digits an activation.
 *   V41_AMX=2  activations split x = hi + lo, both bf16, two tile products: ~16 bits of mantissa, near float32.
 * The weights are exact in bf16 with their scales folded in: an e4m3 value times its ue8m0 tile scale, an e2m1 value
 * times its group's 2^(e-127), a bf16 weight as it is (powers of two only move the exponent, which bf16 shares with
 * float32). Decode-sized calls (rows below the threshold) and calls from inside an OpenMP region keep the AVX2 path.
 * Layout (v2): the weights are the A operand -- 32 output rows a thread decodes row-major into bf16 (vectorized), two
 * A tiles of 16 rows x 32 columns read straight from that slab -- and the activations the B operand, put once a call
 * into the VNNI pairing TDPBF16PS reads (16 pairs of columns x 16 rows x 2 bf16), shared by every output. C tiles 0..3
 * are 2 output tiles x 2 row tiles of 16x16 float32; A tiles 4,5; B tiles 6,7.
 */
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>
#define V41_AMX_HAVE 1
#ifdef _OPENMP
#include <omp.h>
#define V41_AMX_NTH() omp_get_max_threads()
#define V41_AMX_INPAR() omp_in_parallel()
#else   /* a build without OpenMP: one thread, never inside a team */
#define V41_AMX_NTH() 1
#define V41_AMX_INPAR() 0
#endif

/* decode(ctx, o0, O, I, dst): output rows o0..o0+31 (zeros past O) as bf16, row-major, I apart */
typedef void (*v41_amx_decode_fn)(const void *ctx, int o0, int O, int I, uint16_t *dst);
typedef struct { uint8_t palette, start_row; uint8_t rsv[14]; uint16_t colsb[16]; uint8_t rows[16]; } V41TileCfg;

static int g_amx_mode = -1, g_amx_min = 16;
static unsigned long long g_amx_calls, g_amx_rows;
static double g_amx_s, g_amx_flop;

static void v41_amx_report(void) {
    if (g_amx_mode > 0)
        fprintf(stderr, "[v41] V41_AMX=%d: %llu products, %llu rows, %.2f s, %.1f GFLOP/s\n", g_amx_mode, g_amx_calls,
                g_amx_rows, g_amx_s, g_amx_s > 0 ? g_amx_flop / g_amx_s / 1e9 : 0.0);
}
static int v41_amx_on(void) {
    if (g_amx_mode >= 0) return g_amx_mode;
    const char *e = getenv("V41_AMX");
    int m = e ? atoi(e) : 0;
    const char *mn = getenv("V41_AMX_MIN");
    if (mn && atoi(mn) > 0) g_amx_min = atoi(mn);
    if (m > 0) {
        unsigned a = 0, b = 0, c = 0, d = 0;
        /* CPUID.7.0: EDX bit 24 AMX-TILE, bit 22 AMX-BF16; EBX bit 16 AVX512F */
        int ok = __get_cpuid_count(7, 0, &a, &b, &c, &d) && (d & (1u << 24)) && (d & (1u << 22)) && (b & (1u << 16));
        /* the tile data state needs the kernel's permission, process-wide (ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) */
        if (ok) ok = syscall(SYS_arch_prctl, 0x1023, 18) == 0;
        if (!ok) { fprintf(stderr, "[v41] V41_AMX=%d: no AMX-BF16 here, or the kernel refused the tile state: off\n", m); m = 0; }
        else {
            if (m > 2) m = 2;
            fprintf(stderr, "[v41] V41_AMX=%d: prompt-pass products on AMX-BF16 tiles (rows >= %d; activations %s)\n", m,
                    g_amx_min, m == 2 ? "split hi+lo bf16" : "bf16");
            static int registered; if (!registered) { registered = 1; atexit(v41_amx_report); }
        }
    }
    g_amx_mode = m;
    return m;
}
static inline uint16_t v41_f2bf(float f) {   /* round to nearest even; a NaN stays a (quiet) NaN */
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline uint16_t v41_f2bf_exact(float f) {   /* a value bf16 holds exactly: the top half */
    uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16);
}
static inline float v41_bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

__attribute__((target("amx-tile"), noinline)) static void v41_amx_cfg(void) {
    V41TileCfg cfg; memset(&cfg, 0, sizeof cfg);
    cfg.palette = 1;
    for (int t = 0; t < 8; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
    _tile_loadconfig(&cfg);
}
__attribute__((target("amx-tile"), noinline)) static void v41_amx_release(void) { _tile_release(); }

/* output slab ws (32 rows of I bf16) times the row groups [tb, te) (multiples of 16) of the packed activations */
__attribute__((target("amx-tile,amx-bf16"), noinline))
static void v41_amx_task(float *y, int ystride, const uint16_t *xh, const uint16_t *xl, int S, int I, int O,
                         const uint16_t *ws, int o0, int tb, int te) {
    float c[4][256] __attribute__((aligned(64)));   /* [output tile * 2 + row tile][16 outputs][16 rows] */
    const long as = (long)I * 2;
    const int kbs = I / 32;
    const int two_o = o0 + 16 < O;
    for (int t0 = tb; t0 < te; t0 += 32) {
        const int two = t0 + 16 < te;
        const size_t g0 = (size_t)(t0 >> 4) * kbs * 512, g1 = g0 + (size_t)kbs * 512;
        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
        for (int kb = 0; kb < kbs; kb++) {
            _tile_loadd(4, ws + kb * 32, as);
            if (two_o) _tile_loadd(5, ws + (size_t)16 * I + kb * 32, as);
            _tile_loadd(6, xh + g0 + (size_t)kb * 512, 64);
            _tile_dpbf16ps(0, 4, 6);
            if (two_o) _tile_dpbf16ps(2, 5, 6);
            if (two) {
                _tile_loadd(7, xh + g1 + (size_t)kb * 512, 64);
                _tile_dpbf16ps(1, 4, 7);
                if (two_o) _tile_dpbf16ps(3, 5, 7);
            }
            if (xl) {
                _tile_loadd(6, xl + g0 + (size_t)kb * 512, 64);
                _tile_dpbf16ps(0, 4, 6);
                if (two_o) _tile_dpbf16ps(2, 5, 6);
                if (two) {
                    _tile_loadd(7, xl + g1 + (size_t)kb * 512, 64);
                    _tile_dpbf16ps(1, 4, 7);
                    if (two_o) _tile_dpbf16ps(3, 5, 7);
                }
            }
        }
        _tile_stored(0, c[0], 64);
        if (two) _tile_stored(1, c[1], 64);
        if (two_o) { _tile_stored(2, c[2], 64); if (two) _tile_stored(3, c[3], 64); }
        for (int ot = 0; ot <= two_o; ot++)
            for (int rt = 0; rt <= two; rt++) {
                const float *cc = c[ot * 2 + rt];
                const int ob = o0 + ot * 16, n = O - ob < 16 ? O - ob : 16;
                for (int rr = 0; rr < 16; rr++) {
                    const int r = t0 + rt * 16 + rr;
                    if (r >= S) break;
                    float *yr = y + (size_t)r * ystride + ob;
                    for (int k = 0; k < n; k++) yr[k] = cc[k * 16 + rr];
                }
            }
    }
}

/* y[r][o] = sum_i W(o,i) x[r][i] for S rows; decode(ctx, o0, O, I, dst) gives 32 output rows as bf16. Returns 0
 * (nothing written) where the call stays the AVX2 path's. */
static int v41_amx_gemm(float *y, int ystride, const float *x, int xstride, int S, int I, int O,
                        v41_amx_decode_fn decode, const void *ctx) {
    const int mode = v41_amx_on();
    if (!mode || S < g_amx_min || I < 32 || (I & 31) || O < 1 || V41_AMX_INPAR()) return 0;
    const double t0 = now_s();
    const int Sp = (S + 15) & ~15, kbs = I / 32, ngr = Sp / 16;
    const size_t xn = (size_t)Sp * I;   /* the packed activations: [row group][kb][16 pairs][16 rows][2] */
    uint16_t *xh = aligned_alloc(64, (xn * 2 + 63) & ~(size_t)63);
    uint16_t *xl = mode == 2 ? aligned_alloc(64, (xn * 2 + 63) & ~(size_t)63) : NULL;
    if (!xh || (mode == 2 && !xl)) { free(xh); free(xl); return 0; }
    #pragma omp parallel for schedule(static)
    for (int gk = 0; gk < ngr * kbs; gk++) {
        const int g = gk / kbs, kb = gk % kbs;
        uint16_t *h = xh + ((size_t)g * kbs + kb) * 512, *l = xl ? xl + ((size_t)g * kbs + kb) * 512 : NULL;
        for (int rr = 0; rr < 16; rr++) {
            const int r = g * 16 + rr;
            const float *xr = r < S ? x + (size_t)r * xstride + kb * 32 : NULL;
            for (int k = 0; k < 32; k++) {
                const size_t at = (size_t)(k >> 1) * 32 + rr * 2 + (k & 1);
                if (!xr) { h[at] = 0; if (l) l[at] = 0; continue; }
                const uint16_t b = v41_f2bf(xr[k]);
                h[at] = b;
                if (l) { const float hv = v41_bf2f(b); l[at] = isfinite(hv) ? v41_f2bf(xr[k] - hv) : 0; }
            }
        }
    }
    const int nob = (O + 31) / 32, nth = V41_AMX_NTH();
    int nrg = 1;   /* row groups (multiples of 32 rows) when the output slabs alone would leave threads idle */
    while (nob * nrg < 2 * nth && (nrg + 1) * 32 <= Sp) nrg++;
    const int rows_per = ((Sp + nrg - 1) / nrg + 31) & ~31;
    const int ntask = nob * nrg;
    int failed = 0;
    #pragma omp parallel
    {
        uint16_t *ws = aligned_alloc(64, (size_t)I * 64);
        if (!ws) {
            #pragma omp atomic write
            failed = 1;
        }
        #pragma omp barrier
        if (!failed) {
            v41_amx_cfg();
            int last = -1;
            #pragma omp for schedule(static)
            for (int t = 0; t < ntask; t++) {
                const int ob = t / nrg, rg = t % nrg;
                const int rb = rg * rows_per;
                int re = rb + rows_per; if (re > Sp) re = Sp;
                if (rb >= re) continue;
                if (ob != last) { decode(ctx, ob * 32, O, I, ws); last = ob; }
                v41_amx_task(y, ystride, xh, xl, S, I, O, ws, ob * 32, rb, re);
            }
            v41_amx_release();
        }
        free(ws);
    }
    free(xh); free(xl);
    if (failed) return 0;
    g_amx_calls++; g_amx_rows += (unsigned long long)S;
    g_amx_flop += 2.0 * S * (double)I * O * (mode == 2 ? 2 : 1);
    g_amx_s += now_s() - t0;
    return 1;
}

/* V41_AMX_WARM (default on where V41_AMX is): the process's first AMX use as a ramp, before the engine is ready and
 * with nothing else in flight. v41_amx_on is lazy, so without this the first prompt pass is at once the request to the
 * kernel for the tile state, every thread's first tile instruction (the kernel grows each thread's state there) and
 * every core's step to its AMX power level, in the second of the card's first uploads: the instant the host's CPU
 * errors of 2026-11-24/25 fell on (the logs of three of the four end at the V41_AMX line). Here the same tile task runs
 * on private scratch on 1, 2, 4, .. V41_AMX_WARM_STEP (8) threads, then STEP more a step up to all of them,
 * V41_AMX_WARM_MS (150) each with V41_AMX_WARM_PAUSE_MS (100) between; a flushed line a step, so a log shows how far
 * it came. No result is used. V41_AMX_WARM=0: off (the first prompt pass is the first use, as before). */
static void v41_amx_warm(void) {
    const char *e = getenv("V41_AMX_WARM");
    if (e && *e == '0') return;
    const int mode = v41_amx_on();
    if (!mode) return;
    const char *ms_e = getenv("V41_AMX_WARM_MS"), *pz_e = getenv("V41_AMX_WARM_PAUSE_MS"), *st_e = getenv("V41_AMX_WARM_STEP");
    int ms = ms_e && atoi(ms_e) > 0 ? atoi(ms_e) : 150, pz = pz_e && *pz_e ? atoi(pz_e) : 100, step = st_e && atoi(st_e) > 0 ? atoi(st_e) : 8;
    if (ms > 5000) ms = 5000;
    if (pz < 0) pz = 0;
    if (pz > 5000) pz = 5000;
    enum { WI = 1024, WS = 32 };   /* a pass: 32 outputs x 1024 inputs x 32 rows */
    const int nth = V41_AMX_NTH();
    const size_t xn = (size_t)WS * WI;
    uint16_t *xh = aligned_alloc(64, xn * 2), *xl = mode == 2 ? aligned_alloc(64, xn * 2) : NULL;
    if (!xh || (mode == 2 && !xl)) { free(xh); free(xl); return; }
    for (size_t i = 0; i < xn; i++) {
        xh[i] = v41_f2bf((float)((int)(i % 13) - 6) * 0.125f);
        if (xl) xl[i] = v41_f2bf((float)((int)(i % 7) - 3) * 0x1p-12f);
    }
    const double t0 = now_s();
    fprintf(stderr, "[v41] AMX warm-up (V41_AMX_WARM): the tile products on 1 to %d threads, %d ms a step, %d ms between, "
            "before the engine is ready\n", nth, ms, pz);
    fflush(stderr);
    for (int k = 1;;) {
        double flop = 0;
        int used = 0;
        const double ts = now_s();
        #pragma omp parallel num_threads(k) reduction(+:flop) reduction(+:used)
        {
            uint16_t *ws = aligned_alloc(64, (size_t)WI * 64);
            float *y = aligned_alloc(64, (size_t)WS * 32 * sizeof(float));
            if (ws && y) {
                for (int i = 0; i < WI * 32; i++) ws[i] = v41_f2bf((float)((i % 11) - 5) * 0.25f);
                v41_amx_cfg();
                do {
                    v41_amx_task(y, 32, xh, xl, WS, WI, 32, ws, 0, 0, WS);
                    flop += 2.0 * WS * WI * 32 * (mode == 2 ? 2 : 1);
                } while ((now_s() - ts) * 1e3 < ms);
                v41_amx_release();
                used = 1;
            }
            free(ws); free(y);
        }
        const double dt = now_s() - ts;
        fprintf(stderr, "[v41] AMX warm-up: %d of %d threads, %.0f ms, %.1f GFLOP/s\n", used, nth, dt * 1e3,
                dt > 0 ? flop / dt * 1e-9 : 0.0);
        fflush(stderr);
        if (k >= nth) break;
        k = k < step ? k * 2 : k + step;
        if (k > nth) k = nth;
        if (pz) usleep((useconds_t)pz * 1000);
    }
    fprintf(stderr, "[v41] AMX warm-up done in %.2f s\n", now_s() - t0);
    fflush(stderr);
    free(xh); free(xl);
}

/* vectorized decoders' pieces (AVX-512F/BW): 16 floats that bf16 holds exactly -> their top halves */
__attribute__((target("avx512f,avx512bw"))) static inline void v41_amx_st16(uint16_t *d, __m512 v) {
    _mm256_storeu_si256((__m256i *)d, _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(v), 16)));
}
/* 32 e2m1 values (16 bytes, low nibble first) times f -> bf16; f = the group's scale (exact) */
__attribute__((target("avx512f,avx512bw"))) static inline void v41_amx_mx4_32(uint16_t *d, const uint8_t *q, float f) {
    const __m128i lut2 = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), m4 = _mm_set1_epi8(0x0F);
    const __m128i by = _mm_loadu_si128((const __m128i *)q);
    const __m128i lo = _mm_and_si128(by, m4), hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
    const __m128i n0 = _mm_shuffle_epi8(lut2, _mm_unpacklo_epi8(lo, hi)), n1 = _mm_shuffle_epi8(lut2, _mm_unpackhi_epi8(lo, hi));
    const __m512 fv = _mm512_set1_ps(f * 0.5f);   /* the doubled values' halving rides the scale, as matmul_mxfp4 */
    v41_amx_st16(d, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(n0)), fv));
    v41_amx_st16(d + 16, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(n1)), fv));
}
/* 16 e4m3 bytes times f -> bf16 (quant.h's decode: bits placed, times 2^120; f folds 2^120 and the tile scale) */
__attribute__((target("avx512f,avx512bw"))) static inline void v41_amx_e4_16(uint16_t *d, const uint8_t *q, float f) {
    const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)q));
    const __m512i bits = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x80)), 24),
                                         _mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x7f)), 20));
    v41_amx_st16(d, _mm512_mul_ps(_mm512_castsi512_ps(bits), _mm512_set1_ps(f)));
}
/* The decoders: 32 output rows as bf16, row-major (the A operand), zeros past O; I a multiple of 32. */
typedef struct { const uint8_t *q, *s; } V41AmxFp8;     /* e4m3 [O][I], one ue8m0 a 32x32 tile [(O+31)/32][I/32] */
typedef struct { const uint8_t *q4, *e8s; } V41Mx4;     /* e2m1 pairs [O][I/2] (low nibble first), ue8m0 [O][I/32] */
typedef struct { const uint16_t *w; } V41AmxBf16;       /* bf16 [O][I] */
static inline float v41_amx_e8(uint8_t e) { uint32_t u = (uint32_t)e << 23; float f; memcpy(&f, &u, 4); return f; }
__attribute__((target("avx512f,avx512bw")))
static void v41_amx_dec_fp8(const void *ctx, int o0, int O, int I, uint16_t *dst) {
    const V41AmxFp8 *w = (const V41AmxFp8 *)ctx;
    const int tiles_i = I / 32;
    for (int n = 0; n < 32; n++) {
        const int o = o0 + n;
        uint16_t *d = dst + (size_t)n * I;
        if (o >= O) { memset(d, 0, (size_t)I * 2); continue; }
        const uint8_t *row = w->q + (size_t)o * I, *sc = w->s + (size_t)(o / 32) * tiles_i;
        for (int t = 0; t < tiles_i; t++) {
            const float f = 0x1p120f * v41_amx_e8(sc[t]);
            v41_amx_e4_16(d + t * 32, row + t * 32, f);
            v41_amx_e4_16(d + t * 32 + 16, row + t * 32 + 16, f);
        }
    }
}
__attribute__((target("avx512f,avx512bw")))
static void v41_amx_dec_mxfp4(const void *ctx, int o0, int O, int I, uint16_t *dst) {
    const V41Mx4 *w = (const V41Mx4 *)ctx;
    const int rb = I / 2, ng = I / 32;
    for (int n = 0; n < 32; n++) {
        const int o = o0 + n;
        uint16_t *d = dst + (size_t)n * I;
        if (o >= O) { memset(d, 0, (size_t)I * 2); continue; }
        const uint8_t *row = w->q4 + (size_t)o * rb, *sc = w->e8s + (size_t)o * ng;
        for (int g = 0; g < ng; g++) v41_amx_mx4_32(d + g * 32, row + g * 16, v41_amx_e8(sc[g]));
    }
}
static void v41_amx_dec_bf16(const void *ctx, int o0, int O, int I, uint16_t *dst) {
    const V41AmxBf16 *w = (const V41AmxBf16 *)ctx;
    for (int n = 0; n < 32; n++) {
        const int o = o0 + n;
        if (o < O) memcpy(dst + (size_t)n * I, w->w + (size_t)o * I, (size_t)I * 2);
        else memset(dst + (size_t)n * I, 0, (size_t)I * 2);
    }
}
#else
#define V41_AMX_HAVE 0
static inline void v41_amx_warm(void) {}
#endif
