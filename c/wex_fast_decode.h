#ifndef COLI_WEX_FAST_DECODE_H
#define COLI_WEX_FAST_DECODE_H
/* Fast CPU decoder for WNR3 records. No format change: same bytes in, same bytes out, same accept/reject
 * as the reference wex_nr2_decode (wex_neural_rans.h). From Wex-WRN2's wex4/research/fast-cpu-decode/
 * (RESULTS.md there has the measurements and the validation), with its kernels 0 and 1; kernel 2 (AVX2 and
 * SVE gathers, slower than kernel 1 on the CPUs measured there) is not carried.
 *
 *   int wex_fast_decode(view, layout, model, scales_out, weights_out);            tables built per call
 *   wex_fast_model_init(&tables, model);                                          once per model (224 KiB)
 *   int wex_fast_decode_with(&tables, view, layout, model, scales_out, weights_out, &options);
 *   const char *wex_fast_kernels(void);      "AVX2", "NEON" or "scalar": what kernel 1 runs on, on this CPU
 *
 * Contract (the reference's): `view` comes from wex_nr3_parse with this `layout`, `model` from
 * wex_nr2_model_open. Returns 0 or -1; outputs may be partially written on -1. Views that are not WNR3 go to
 * the reference decoder unchanged.
 *
 * What is different from the reference:
 *  (a) streams are independent, so they can run in parallel (OpenMP, options.threads): the u16 lengths are
 *      prefix-summed once into at most WEXF_MAX_TASKS task offsets, no allocation;
 *  (b) symbol search without branches. Weights (kernel 1): each (matrix, srel, maximum, previous) context has
 *      a 128-byte table entry {upper[16], start[16], freq[16], next[16]}; one AVX2 compare of the slot with
 *      upper[] + tzcnt gives the symbol (NEON: two 8-lane compares narrowed to a 64-bit mask + ctz), next[]
 *      gives the following context (no arithmetic on previous / maximum). Scales: a 16384-entry slot ->
 *      symbol table per matrix;
 *  (c) N streams are decoded in one loop (kernel 1) so the long per-symbol dependency chain of one stream
 *      overlaps with the others';
 *  (d) fuse = 1 computes the MurmurHash3 of each stream and undoes the ceil form inside the decode loop.
 *
 * Renormalisation is branchless and reads 4 bytes ahead; a stream that runs past its end is noticed at the
 * next 32-symbol boundary instead of at the byte. This is equivalent to the reference check: the read
 * pointer only grows, so "needs a byte at end" in the reference <=> "pointer beyond end" here, and both then
 * fail. Streams closer than WEXF_SLACK bytes to the end of the weight payload take the checked per-byte
 * loop, so nothing is ever read at or beyond that end. The state stays in [2^23, 2^31) for any input bytes
 * (valid model: every frequency >= 1), hence at most 2 renormalisation bytes per symbol; the code is
 * memory-safe (3 bytes per symbol at most) even if that did not hold.
 *
 * Kernels and CPUs. x86-64 with GCC or clang: the AVX2 kernel 1 (AVX2, BMI1, BMI2, LZCNT) is compiled in
 * every build, with a target attribute when the build itself does not target AVX2, and runs only when CPUID
 * says the CPU (and the OS, for the YMM state) has all four; otherwise kernel 0 runs. Little-endian AArch64:
 * the NEON kernel 1 (NEON is part of the base instruction set). Any other target, or -DWEX_FAST_SCALAR: kernel
 * 0 only (the reference loop per stream, still parallel across streams). WEXF_SIMD is 1 (AVX2 compiled in),
 * 2 (NEON) or 0. */
#include <stdlib.h>
#include <string.h>
#include "wex_format.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#if !defined(WEX_FAST_SCALAR) && (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#include <cpuid.h>
#define WEXF_SIMD 1
#if defined(__AVX2__) && defined(__BMI__) && defined(__BMI2__) && defined(__LZCNT__)
#define WEXF_TARGET
#else
#define WEXF_TARGET __attribute__((target("avx2,bmi,bmi2,lzcnt")))
#endif
#elif !defined(WEX_FAST_SCALAR) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__BYTE_ORDER__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#include <arm_neon.h>
#define WEXF_SIMD 2
#define WEXF_TARGET
#else
#define WEXF_SIMD 0
#define WEXF_TARGET
#endif

#define WEXF_GROUP_MAX 8u       /* streams of one call of kernel 1 */
#define WEXF_MAX_TASKS 256u
#define WEXF_SLACK 104u         /* > 3 bytes x 32 symbols + a 4-byte read */
#define WEXF_ENTRY 128u         /* bytes per weight context: u16 upper[16], start[16], freq[16], next[16] */
#define WEXF_SREL_BYTES (65u * WEXF_ENTRY)   /* entries maximum * 8 + previous (0..63) and 64 = block start */
#define WEXF_CTX_BYTES (3u * 7u * WEXF_SREL_BYTES)
#define WEXF_INLINE static inline __attribute__((always_inline))

/* Streams interleaved by kernel 1 in wex_fast_decode (the default entry point), and in place of a `lanes` value
 * that kernel 1 does not have: -DWEX_FAST_LANES=N, N one of 1, 2, 3, 4, 6, 8; default 4. The decoded bytes and
 * the verdict are the same for every N (all six are compiled in every build, the options pick one at run
 * time); only the speed differs. */
#ifndef WEX_FAST_LANES
#define WEX_FAST_LANES 4
#endif
#if WEX_FAST_LANES != 1 && WEX_FAST_LANES != 2 && WEX_FAST_LANES != 3 && WEX_FAST_LANES != 4 && \
    WEX_FAST_LANES != 6 && WEX_FAST_LANES != 8
#error "WEX_FAST_LANES must be 1, 2, 3, 4, 6 or 8"
#endif

typedef struct {
    int threads;   /* OpenMP threads, 0 = the runtime's default; ignored without OpenMP */
    int kernel;    /* 0 checked scalar loop, 1 AVX2 or NEON search + interleaved streams */
    int lanes;     /* kernel 1: streams per loop (1, 2, 3, 4, 6, 8) */
    int fuse;      /* kernel 1: hash and ceil inverse inside the decode loop */
} WexFastOptions;

/* Decode tables of one model (kernel 1). Read-only after wex_fast_model_init, shareable between threads.
 * The entries start 64-byte aligned inside `raw`; a byte copy of the struct still decodes correctly (unaligned). */
typedef struct {
    uint32_t id, skew;                         /* model id, offset of the first entry in raw */
    unsigned char lut[3u * WEX_NR2_TOTAL];     /* slot -> scale symbol */
    unsigned char raw[WEXF_CTX_BYTES + 64u];
} WexFastModel;

typedef struct {
    const WexNr2Record *v;
    const WexNr2Layout *layout;
    const WexNr2Model *model;
    unsigned char *const *scales, *const *weights;
    const unsigned char *hard_end;   /* end of the weight payload */
    const WexFastModel *fm;          /* NULL: kernel 0 only */
    WexFastOptions o;
    uint32_t cz, ceil;
} WexfJob;

typedef struct {
    const unsigned char *begin;
    uint32_t length, index, matrix, first, count;   /* index = record-wide stream number (hashed) */
} WexfStream;

/* 1 when kernel 1 can run on this CPU. x86-64: CPUID leaf 7 (AVX2, BMI1, BMI2), leaf 0x80000001 (LZCNT), and
 * the OS saves the YMM state (OSXSAVE, XCR0 bits 1-2); asked once per translation unit. */
static inline int wexf_cpu_simd(void) {
#if WEXF_SIMD == 1
    static int known;   /* 0 not asked yet, 1 yes, -1 no */
    int v = __atomic_load_n(&known, __ATOMIC_RELAXED);
    if (!v) {
        unsigned int a, b, c, d, lo = 0, hi = 0;
        int ok = __get_cpuid(1u, &a, &b, &c, &d) && (c & (1u << 27)) && (c & (1u << 28));   /* OSXSAVE, AVX */
        if (ok) {
            __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0u));
            ok = (lo & 6u) == 6u;
        }
        ok = ok && __get_cpuid_count(7u, 0u, &a, &b, &c, &d) && (b & (1u << 5)) && (b & (1u << 3)) && (b & (1u << 8));
        ok = ok && __get_cpuid(0x80000001u, &a, &b, &c, &d) && (c & (1u << 5));
        (void)hi;
        v = ok ? 1 : -1;
        __atomic_store_n(&known, v, __ATOMIC_RELAXED);
    }
    return v > 0;
#elif WEXF_SIMD == 2
    return 1;
#else
    return 0;
#endif
}

static inline const char *wex_fast_kernels(void) {
    if (!wexf_cpu_simd()) return "scalar";
    return WEXF_SIMD == 1 ? "AVX2" : "NEON";
}

WEXF_INLINE void wexf_ident(const WexfJob *J, uint32_t index, uint32_t ident[4]) {
    ident[0] = J->v->model_id; ident[1] = J->v->owner; ident[2] = index; ident[3] = J->v->word24;
}

/* Ceil inverse of the blocks of one stream (reference tables). */
static inline int wexf_ceil_blocks(unsigned char *q, unsigned char *scales, uint32_t blocks) {
    uint32_t b, k;
    for (b = 0; b < blocks; b++, q += 16) {
        if (wex_nr3_block_max(q, NULL) != 5u) continue;
        if (!scales[b]) return -1;
        for (k = 0; k < 16u; k++)
            q[k] = (unsigned char)((q[k] & 0x88u) | wex_nr3_double[q[k] & 7u] | wex_nr3_double[(q[k] >> 4) & 7u] << 4);
        scales[b]--;
    }
    return 0;
}

/* Kernel 0: one stream with the reference's loop and its per-byte end check. Portable C. */
static inline int wexf_stream_checked(const WexfJob *J, const WexfStream *s, int weights) {
    const WexNr2Model *model = J->model;
    const uint32_t m = s->matrix;
    const unsigned char *ptr, *end;
    uint32_t state, j, symbol, ident[4];
    if (wex_nr2_stream_open(s->begin, s->length, &ptr, &end, &state)) return -1;
    wexf_ident(J, s->index, ident);
    if (!weights) {
        const unsigned char *cdf = wex_nr2_scale_cdf(model, m);
        unsigned char *out = J->scales[m] + s->first;
        for (j = 0; j < s->count; j++) {
            if (wex_nr2_decode_symbol(cdf, 256u, &state, &ptr, end, &symbol)) return -1;
            out[j] = (unsigned char)symbol;
        }
        return ptr != end || state != wex_nr3_stream_x0(ident, out, s->count) ? -1 : 0;
    } else {
        unsigned char *out = J->weights[m] + s->first / 2u, *sc = J->scales[m] + s->first / 32u;
        uint32_t srel = 0, previous = 8u, maximum = 8u;
        for (j = 0; j < s->count; j++) {
            if (!(j % 32u)) {
                srel = wex_nr2_srel(sc[j / 32u], J->v->median[m]);
                previous = 8u; maximum = 8u;
            }
            if (wex_nr2_decode_symbol(wex_nr2_weight_cdf(model, m, wex_nr2_context(srel, previous, maximum)),
                                      16u, &state, &ptr, end, &symbol) || (J->cz && symbol == 8u)) return -1;
            if (!(j & 1u)) out[j / 2u] = (unsigned char)symbol;
            else out[j / 2u] |= (unsigned char)(symbol << 4);
            previous = symbol & 7u;
            if (maximum == 8u || previous > maximum) maximum = previous;
        }
        if (ptr != end || state != wex_nr3_stream_x0(ident, out, s->count / 2u)) return -1;
        return J->ceil >> m & 1u ? wexf_ceil_blocks(out, sc, s->count / 32u) : 0;
    }
}

/* Tables of one model for kernel 1. Plain C: built on any CPU, used only where kernel 1 runs. */
static inline void wex_fast_model_init(WexFastModel *f, const WexNr2Model *model) {
    uint32_t m, srel, e, symbol;
    f->id = model->id;
    f->skew = (uint32_t)(0u - (uintptr_t)f->raw) & 63u;
    memset(f->lut, 0, sizeof(f->lut));
    for (m = 0; m < 3u; m++)
        for (symbol = 0; symbol < 256u; symbol++) {
            const unsigned char *cdf = wex_nr2_scale_cdf(model, m) + symbol * 2u;
            const uint32_t lo = wex_nr_u16(cdf), hi = wex_nr_u16(cdf + 2u);
            if (lo < hi && hi <= WEX_NR2_TOTAL) memset(f->lut + (size_t)m * WEX_NR2_TOTAL + lo, (int)symbol, hi - lo);
        }
    for (m = 0; m < 3u; m++)
        for (srel = 0; srel < 7u; srel++)
            for (e = 0; e < 65u; e++) {
                const uint32_t previous = e == 64u ? 8u : e & 7u, maximum = e == 64u ? 8u : e >> 3;
                const unsigned char *cdf = wex_nr2_weight_cdf(model, m, wex_nr2_context(srel, previous, maximum));
                uint16_t entry[64];
                for (symbol = 0; symbol < 16u; symbol++) {
                    const uint32_t a = symbol & 7u, mx = e == 64u || a > maximum ? a : maximum;
                    entry[symbol] = (uint16_t)wex_nr_u16(cdf + symbol * 2u + 2u);
                    entry[16u + symbol] = (uint16_t)wex_nr_u16(cdf + symbol * 2u);
                    entry[32u + symbol] = (uint16_t)(entry[symbol] - entry[16u + symbol]);
                    entry[48u + symbol] = (uint16_t)((mx * 8u + a) * (WEXF_ENTRY / 8u));   /* next entry, in 8-byte units */
                }
                entry[15] = 0x7fff;   /* cdf[16] = 16384 on a valid model: the search always stops by symbol 15 */
                memcpy(f->raw + f->skew + ((size_t)(m * 7u + srel) * 65u + e) * WEXF_ENTRY, entry, WEXF_ENTRY);
            }
}

#if WEXF_SIMD
/* Branchless byte renormalisation with the next bytes at p: 0, 1 or 2 are used on a valid model (never more
 * than 3); *spare = 3 - bytes used. */
WEXF_INLINE WEXF_TARGET uint32_t wexf_renorm(uint32_t x, const unsigned char *p, uint32_t *spare) {
#if WEXF_SIMD == 2
    const uint32_t keep = (0u - (x ? (uint32_t)__builtin_clz(x) : 32u)) & 24u;   /* one CLZ instruction */
#else
    const uint32_t keep = (0u - _lzcnt_u32(x)) & 24u;   /* leading zeros 1..8 -> 24, 9..16 -> 16, 17..24 -> 8 */
#endif
    uint32_t next;
    uint64_t wide;
    memcpy(&next, p, 4);
    *spare = keep >> 3;
    wide = ((uint64_t)x << 32 | __builtin_bswap32(next)) >> 8;
    return (uint32_t)(wide >> keep);
}

/* D(a) = a + min(a, 2) on the magnitudes of 16 nibbles that are all <= 5; signs kept. */
WEXF_INLINE WEXF_TARGET uint64_t wexf_double16(uint64_t w) {
    const uint64_t one = UINT64_C(0x1111111111111111), mag = w & 7u * one;
    const uint64_t ge2 = (mag >> 1 | mag >> 2) & one;
    return (w & 8u * one) | (mag + (ge2 << 1) + (mag & one & ~ge2));
}
/* Non-zero iff some nibble of w is 8. */
WEXF_INLINE WEXF_TARGET uint64_t wexf_has8(uint64_t w) {
    const uint64_t one = UINT64_C(0x1111111111111111), t = w ^ 8u * one;
    return (t - one) & ~t & 8u * one;
}
WEXF_INLINE WEXF_TARGET int wexf_ceil_fused(unsigned char *q, unsigned char *scale) {
    uint64_t w[2];
    if (!*scale) return -1;
    memcpy(w, q, 16);
    w[0] = wexf_double16(w[0]); w[1] = wexf_double16(w[1]);
    memcpy(q, w, 16);
    --*scale;
    return 0;
}

/* Kernel 1, weights: N streams of one matrix with the same symbol count (a multiple of 32), one loop. */
WEXF_INLINE WEXF_TARGET int wexf_weights_n(const WexfJob *J, const WexfStream *s, const uint32_t N, const int fuse) {
    const uint32_t m = s[0].matrix, blocks = s[0].count / 32u, median = J->v->median[m], ceil = J->ceil >> m & 1u;
    const unsigned char *table = J->fm->raw + J->fm->skew + (size_t)m * (7u * WEXF_SREL_BYTES);
#if WEXF_SIMD == 1
    const uint64_t nibbles = UINT64_C(0x1e1e1e1e1e1e1e1e);   /* 2 x symbol in each byte of s2 */
#endif
    const unsigned char *p, *e[8], *base[8], *ctx[8];
    unsigned char *out[8], *sc[8], s2[8][16];
    uintptr_t q[8];   /* read position of a stream minus 3 x (symbols done in the current half block) */
    uint32_t x[8], h[8], ident[4], k, b, j, half;
    uint64_t bad = 0;
    for (k = 0; k < N; k++) {
        if (wex_nr2_stream_open(s[k].begin, s[k].length, &p, &e[k], &x[k])) return -1;
        q[k] = (uintptr_t)p;
        out[k] = J->weights[m] + s[k].first / 2u; sc[k] = J->scales[m] + s[k].first / 32u;
        wexf_ident(J, s[k].index, ident);
        h[k] = wex_nr3_ident_hash(ident);
    }
    for (b = 0; b < blocks; b++) {
        for (k = 0; k < N; k++) {
            if (q[k] > (uintptr_t)e[k]) return -1;
            base[k] = table + (size_t)wex_nr2_srel(sc[k][b], median) * WEXF_SREL_BYTES;
            ctx[k] = base[k] + 64u * WEXF_ENTRY;   /* block start: previous = 8, maximum = 8 */
        }
        for (half = 0; half < 2u; half++) {
            for (j = 0; j < 16u; j++)
                for (k = 0; k < N; k++) {
                    const uint32_t slot = x[k] & (WEX_NR2_TOTAL - 1u);
#if WEXF_SIMD == 2
                    /* the same signed compare as AVX2 on the 16 upper[] values, in two halves; one byte of each
                     * 16-bit lane is kept (uzp1), then 4 bits of each byte (shrn): bit 4 x symbol is the lowest
                     * set bit of the mask, which is never 0 because upper[15] = 0x7fff > slot */
                    const int16x8_t slots = vdupq_n_s16((int16_t)slot);
                    const uint8x16_t gt_lo = vreinterpretq_u8_u16(vcgtq_s16(vreinterpretq_s16_u8(vld1q_u8(ctx[k])), slots));
                    const uint8x16_t gt_hi = vreinterpretq_u8_u16(vcgtq_s16(vreinterpretq_s16_u8(vld1q_u8(ctx[k] + 16)), slots));
                    const uint64_t found = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vuzp1q_u8(gt_lo, gt_hi)), 4)), 0);
                    const uint32_t t = (uint32_t)__builtin_ctzll(found) >> 1;
#else
                    const __m256i upper = _mm256_loadu_si256((const __m256i *)(const void *)ctx[k]);
                    const uint32_t t = _tzcnt_u32((uint32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi16(upper, _mm256_set1_epi16((short)slot))));
#endif
                    uint16_t start, freq, next;   /* t = 2 x symbol <= 30 */
                    uint32_t spare;
                    memcpy(&start, ctx[k] + 32u + t, 2); memcpy(&freq, ctx[k] + 64u + t, 2); memcpy(&next, ctx[k] + 96u + t, 2);
                    s2[k][j] = (unsigned char)t;
                    x[k] = wexf_renorm(freq * (x[k] >> WEX_NR2_PRECISION) + (slot - start), (const unsigned char *)(q[k] + 3u * j), &spare);
                    q[k] -= spare;
                    ctx[k] = base[k] + 8u * (size_t)next;
                }
            for (k = 0; k < N; k++) {
#if WEXF_SIMD == 2
                /* a symbol per byte, then the two symbols of each 16-bit lane into its low byte: a + (a >> 4) */
                const uint16x8_t pair = vreinterpretq_u16_u8(vshrq_n_u8(vld1q_u8(s2[k]), 1));
                const uint64_t acc = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vsraq_n_u16(pair, pair, 4))), 0);
#else
                uint64_t lo, hi, acc;
                memcpy(&lo, s2[k], 8); memcpy(&hi, s2[k] + 8, 8);
                acc = _pext_u64(lo, nibbles) | _pext_u64(hi, nibbles) << 32;
#endif
                memcpy(out[k] + (size_t)b * 16u + half * 8u, &acc, 8);
                bad |= wexf_has8(acc);
                if (fuse) h[k] = wex_nr3_h32_block(wex_nr3_h32_block(h[k], (uint32_t)acc), (uint32_t)(acc >> 32));
                q[k] += 48u;
            }
        }
        if (fuse && ceil)
            for (k = 0; k < N; k++)   /* entry index = maximum * 8 + previous */
                if ((size_t)(ctx[k] - base[k]) / (8u * WEXF_ENTRY) == 5u && wexf_ceil_fused(out[k] + (size_t)b * 16u, sc[k] + b)) return -1;
    }
    if (bad && J->cz) return -1;
    for (k = 0; k < N; k++) {
        uint32_t x0;
        if (fuse) x0 = WEX_NR_L + (wex_nr3_h32_final(h[k], 16u + blocks * 16u) & WEX_NR3_HASH_MASK);
        else { wexf_ident(J, s[k].index, ident); x0 = wex_nr3_stream_x0(ident, out[k], blocks * 16u); }
        if (q[k] != (uintptr_t)e[k] || x[k] != x0) return -1;
        if (!fuse && ceil && wexf_ceil_blocks(out[k], sc[k], blocks)) return -1;
    }
    return 0;
}

/* Kernel 1, scales: N streams of one matrix with the same count; symbol = table[slot]. */
WEXF_INLINE WEXF_TARGET int wexf_scales_n(const WexfJob *J, const WexfStream *s, const uint32_t N) {
    const uint32_t m = s[0].matrix, count = s[0].count;
    const unsigned char *cdf = wex_nr2_scale_cdf(J->model, m), *lut = J->fm->lut + (size_t)m * WEX_NR2_TOTAL;
    const unsigned char *p[8], *e[8];
    unsigned char *out[8];
    uint32_t x[8], ident[4], k, j;
    for (k = 0; k < N; k++) {
        if (wex_nr2_stream_open(s[k].begin, s[k].length, &p[k], &e[k], &x[k])) return -1;
        out[k] = J->scales[m] + s[k].first;
    }
    for (j = 0; j < count; j++) {
        if (!(j & 31u))
            for (k = 0; k < N; k++) if (p[k] > e[k]) return -1;
        for (k = 0; k < N; k++) {
            const uint32_t slot = x[k] & (WEX_NR2_TOTAL - 1u), symbol = lut[slot];
            uint32_t sf, spare;   /* sf = cdf[symbol] | cdf[symbol + 1] << 16 */
            memcpy(&sf, cdf + 2u * symbol, 4);
            x[k] = wexf_renorm(((sf >> 16) - (sf & 0xffffu)) * (x[k] >> WEX_NR2_PRECISION) + (slot - (sf & 0xffffu)), p[k], &spare);
            p[k] += 3u - spare;
            out[k][j] = (unsigned char)symbol;
        }
    }
    for (k = 0; k < N; k++) {
        wexf_ident(J, s[k].index, ident);
        if (p[k] != e[k] || x[k] != wex_nr3_stream_x0(ident, out[k], count)) return -1;
    }
    return 0;
}

/* Out-of-line instances: the group size and the fuse flag are compile-time constants inside each. */
static WEXF_TARGET int wexf_group(const WexfJob *J, const WexfStream *s, int weights, uint32_t n) {
    const int fuse = J->o.fuse;
    if (!weights) switch (n) {
        case 1: return wexf_scales_n(J, s, 1);
        case 2: return wexf_scales_n(J, s, 2);
        case 3: return wexf_scales_n(J, s, 3);
        case 4: return wexf_scales_n(J, s, 4);
        case 6: return wexf_scales_n(J, s, 6);
        default: return wexf_scales_n(J, s, 8);
    }
    switch (n * 2u + (uint32_t)fuse) {
        case 2: return wexf_weights_n(J, s, 1, 0);
        case 3: return wexf_weights_n(J, s, 1, 1);
        case 4: return wexf_weights_n(J, s, 2, 0);
        case 5: return wexf_weights_n(J, s, 2, 1);
        case 6: return wexf_weights_n(J, s, 3, 0);
        case 7: return wexf_weights_n(J, s, 3, 1);
        case 8: return wexf_weights_n(J, s, 4, 0);
        case 9: return wexf_weights_n(J, s, 4, 1);
        case 12: return wexf_weights_n(J, s, 6, 0);
        case 13: return wexf_weights_n(J, s, 6, 1);
        case 16: return wexf_weights_n(J, s, 8, 0);
        default: return wexf_weights_n(J, s, 8, 1);
    }
}
#endif

/* Streams [s0, s1) of a section (all scale streams or all weight streams); stream s0 starts at `at`. */
static int wexf_run(const WexfJob *J, int weights, uint32_t s0, uint32_t s1, const unsigned char *at) {
    const uint32_t *total = weights ? J->layout->symbol_count : J->layout->scale_count;
    const uint32_t chunk = weights ? J->v->weight_chunk : J->v->scale_chunk;
    const uint32_t index0 = weights ? J->v->scale_streams : 0u;
    const uint32_t n0 = wex_nr2_streams(total[0], chunk), n1 = n0 + wex_nr2_streams(total[1], chunk);
    const uint32_t want = J->o.kernel == 1 ? (uint32_t)J->o.lanes : 1u;
    while (s0 < s1) {
        WexfStream g[WEXF_GROUP_MAX] = {{0}};
        const unsigned char *q = at;
        uint32_t n, fast = 0;
        for (n = 0; n < want && s0 + n < s1; n++) {
            const uint32_t t = s0 + n, m = t < n0 ? 0u : t < n1 ? 1u : 2u, local = t - (m == 0u ? 0u : m == 1u ? n0 : n1);
            WexfStream *s = g + n;
            s->begin = q; s->index = index0 + t; s->matrix = m; s->first = local * chunk;
            s->length = wex_nr_u16(J->v->lengths + (size_t)s->index * 2u);
            s->count = total[m] - s->first < chunk ? total[m] - s->first : chunk;
            if ((size_t)(J->hard_end - q) < s->length) return -1;   /* cannot happen on a parsed record */
            fast = (size_t)(J->hard_end - q) >= (size_t)s->length + WEXF_SLACK && !(weights && s->count % 32u);
            if (!fast || m != g[0].matrix || s->count != g[0].count) break;
            q += s->length;
        }
#if WEXF_SIMD
        if (J->o.kernel && n == want) {
            if (wexf_group(J, g, weights, n)) return -1;
            s0 += n; at = q;
            continue;
        }
        if (J->o.kernel && n) {   /* stream s0 alone is fast */
            if (wexf_group(J, g, weights, 1u)) return -1;
            s0++; at += g[0].length;
            continue;
        }
#endif
        (void)fast;
        if (wexf_stream_checked(J, g, weights)) return -1;
        s0++; at += g[0].length;
    }
    return 0;
}

static int wexf_section(const WexfJob *J, int weights) {
    const uint32_t *total = weights ? J->layout->symbol_count : J->layout->scale_count;
    const uint32_t chunk = weights ? J->v->weight_chunk : J->v->scale_chunk;
    const uint32_t index0 = weights ? J->v->scale_streams : 0u;
    const uint64_t streams = (uint64_t)wex_nr2_streams(total[0], chunk) + wex_nr2_streams(total[1], chunk) +
                             wex_nr2_streams(total[2], chunk);
    /* tasks of `per` streams: a multiple of 48 so that full groups of any lane count fit, at most WEXF_MAX_TASKS */
    const uint64_t per = ((streams + WEXF_MAX_TASKS - 1u) / WEXF_MAX_TASKS + 47u) / 48u * 48u;
    const int tasks = per ? (int)((streams + per - 1u) / per) : 0;
    const unsigned char *payload = weights ? J->v->weight_payload : J->v->scale_payload;
    uint64_t offset[WEXF_MAX_TASKS], at = 0, s = 0;
    int bad = 0, t;
#ifdef _OPENMP
    const int threads = J->o.threads > 0 ? J->o.threads : omp_get_max_threads();
#endif
    for (t = 0; t < tasks; t++) {   /* prefix sum of the u16 lengths, one offset per task */
        const uint64_t stop = s + per < streams ? s + per : streams;
        offset[t] = at;
        for (; s < stop; s++) at += wex_nr_u16(J->v->lengths + (size_t)(index0 + s) * 2u);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(|:bad) num_threads(threads) if (threads > 1 && tasks > 1)
#endif
    for (t = 0; t < tasks; t++) {
        const uint64_t s0 = (uint64_t)t * per, s1 = s0 + per < streams ? s0 + per : streams;
        bad |= wexf_run(J, weights, (uint32_t)s0, (uint32_t)s1, payload + offset[t]) != 0;
    }
    return bad ? -1 : 0;
}

/* tables: from wex_fast_model_init for this model, or NULL (kernel 0). Kernel 1 runs only where
 * wex_fast_kernels() is not "scalar"; elsewhere a request for it runs kernel 0. */
static inline int wex_fast_decode_with(const WexFastModel *tables, const WexNr2Record *v, const WexNr2Layout *layout,
                                       const WexNr2Model *model, unsigned char *const scales_out[3],
                                       unsigned char *const weights_out[3], const WexFastOptions *options) {
    WexfJob J;
    if (v->magic != WEX_NR3_MAGIC) return wex_nr2_decode(v, layout, model, scales_out, weights_out);
    if (v->model_id != model->id) return -1;
    J.v = v; J.layout = layout; J.model = model; J.scales = scales_out; J.weights = weights_out;
    J.hard_end = v->weight_payload + v->weight_payload_bytes;
    J.o = *options;
    J.cz = v->flags & WEX_NR3_FLAG_CZ; J.ceil = v->flags & WEX_NR3_FLAG_CEIL_MASK;
    J.fm = tables;
    if (J.o.kernel != 0) J.o.kernel = 1;
    if (!tables || tables->id != model->id || !wexf_cpu_simd()) J.o.kernel = 0;
    if (J.o.kernel == 1 && J.o.lanes != 1 && J.o.lanes != 2 && J.o.lanes != 3 && J.o.lanes != 4 && J.o.lanes != 6 &&
        J.o.lanes != 8) J.o.lanes = WEX_FAST_LANES;
    J.o.fuse = J.o.fuse != 0;
    if (wexf_section(&J, 0)) return -1;
    return wexf_section(&J, 1);
}

/* Same, with the tables built for this call (heap; the checked kernel if there is no memory). */
static inline int wex_fast_decode_opt(const WexNr2Record *v, const WexNr2Layout *layout, const WexNr2Model *model,
                                      unsigned char *const scales_out[3], unsigned char *const weights_out[3],
                                      const WexFastOptions *options) {
    WexFastModel *tables = NULL;
    int status;
    if (options->kernel && v->magic == WEX_NR3_MAGIC && v->model_id == model->id && wexf_cpu_simd()) {
        tables = (WexFastModel *)malloc(sizeof(*tables));
        if (tables) wex_fast_model_init(tables, model);
    }
    status = wex_fast_decode_with(tables, v, layout, model, scales_out, weights_out, options);
    free(tables);
    return status;
}

/* Default: the fastest configuration measured in Wex-WRN2: kernel 1, WEX_FAST_LANES streams, fused. */
static inline int wex_fast_decode(const WexNr2Record *v, const WexNr2Layout *layout, const WexNr2Model *model,
                                  unsigned char *const scales_out[3], unsigned char *const weights_out[3]) {
    static const WexFastOptions best = {0, 1, WEX_FAST_LANES, 1};
    return wex_fast_decode_opt(v, layout, model, scales_out, weights_out, &best);
}

/* One record into its slab, as wex_decode_nr2_record (wex_format.h) does it with the reference decoder: the
 * same checks in the same order, the record parsed at slab + wex_record_read_offset(entry), decoded into
 * scratch and copied to the slab only on 0, so on -1 the slab is as it was. A WNR3 record goes to
 * wex_fast_decode_with (tables, options as there); any other record to wex_decode_nr2_record. The WNR3 owner
 * check is the caller's (wex_nr3_owner_ok), as with the reference. */
static inline int wex_fast_decode_record(const WexEntry *entry, unsigned char *slab, size_t capacity,
                                         const WexNr2Model *model, const WexFastModel *tables,
                                         const WexFastOptions *options) {
    uint64_t raw, aligned_raw;
    size_t read_offset;
    WexNr2Layout layout;
    WexNr2Record view;
    unsigned char *decoded, *scales[3], *weights[3];
    unsigned int m;
    int result;
    if (!entry || entry->flags != WEX_ENTRY_NEURAL_RANS3) return wex_decode_nr2_record(entry, slab, capacity, model);
    if (!slab || !model || !options || wex_entry_raw_sizes(entry, &raw, &aligned_raw) || aligned_raw > capacity)
        return -1;
    read_offset = wex_record_read_offset(entry);
    if (!read_offset || read_offset > capacity || entry->total_bytes > capacity - read_offset) return -1;
    wex_nr2_layout(entry, &layout);
    if (wex_nrx_parse(slab + read_offset, entry->total_bytes, &layout, model->id, wex_entry_nr_magic(entry), &view))
        return -1;
    decoded = (unsigned char *)malloc((size_t)raw);
    if (!decoded) return -1;
    for (m = 0; m < 3u; m++) {
        scales[m] = decoded + entry->scale_offsets[m];
        weights[m] = decoded + entry->weight_offsets[m];
    }
    result = wex_fast_decode_with(tables, &view, &layout, model, scales, weights, options);
    if (!result) memcpy(slab, decoded, (size_t)raw);
    free(decoded);
    return result;
}
#endif
