#ifndef COLIBRI_SSE41_KERNELS_H
#define COLIBRI_SSE41_KERNELS_H
/*
 * sse41_kernels.h — shared SSE 4.1 primitives for Colibri engines.
 *
 * The engines (c/deepseek_v4.c, c/colibri.c, c/kimi_k3.c, c/olmoe.c, c/inkling.c)
 * have an `#if defined(__AVX2__)` dispatch for fast paths and fall through to scalar
 * on pre-Haswell hardware (Sandy Bridge: AVX 1.0, no FMA, no AVX-2). This header
 * provides the missing middle tier: 128-bit SIMD primitives, FMA-free.
 *
 * Why a separate header (not just inline in each .c):
 * - 109 AVX2 sites total across 5 engines need patching. Copy-pasting the
 *   128-bit intrinsics 109 times is a typo factory. A single macro definition
 *   is the difference between correct and wrong-on-100-sites.
 * - The single most critical shared piece is the FMA-emulation macro: on
 *   Sandy Bridge, _mm_mul_ps + _mm_add_ps has double rounding vs. hardware
 *   FMA's single rounding, so the output is NOT bit-identical to AVX2 (1-2 ULP
 *   difference). A typo in a single copy is a silent correctness bug.
 *
 * This is the minimum needed for the SSE 4.1 fallback. More primitives can be
 * added as additional engines are patched.
 */
#if defined(__SSE2__)

/*
 * COLIBRI_FMA: emulate FMA on non-FMA hardware.
 *
 * On FMA hardware: maps to _mm_fmadd_ps (single rounding, 1 instruction).
 * On Sandy Bridge (no FMA): separate mul+add, double rounding, 2 instructions.
 *
 * On Sandy Bridge this is NOT bit-identical to AVX2/FMA output — typically
 * within 1-2 ULP. Test tolerance must accommodate this.
 */
#if defined(__FMA__)
#  define COLIBRI_FMA(a, b, c)  _mm_fmadd_ps((a), (b), (c))
#else
#  define COLIBRI_FMA(a, b, c)  _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#endif

/*
 * SSE 4.1 (or lower) load/store helpers. Sandy Bridge has these natively.
 * _mm_load_ps is aligned; _mm_loadu_ps is unaligned. For 128-bit (16-byte)
 * data, aligned loads are faster but UB on misaligned pointers. Default to
 * unaligned: buffers from malloc / numa_slab_bind have no 16-byte guarantee.
 * Aligned loads can be added as a profiled follow-up.
 */
static inline __m128 colibri_sse41_loadu_ps(const float *p) { return _mm_loadu_ps(p); }
static inline void  colibri_sse41_storeu_ps(float *p, __m128 v) { _mm_storeu_ps(p, v); }

/*
 * Min/max (SSE 4.1 native). Identical to AVX2, just narrower width.
 */
static inline __m128 colibri_sse41_min_ps(__m128 a, __m128 b) { return _mm_min_ps(a, b); }
static inline __m128 colibri_sse41_max_ps(__m128 a, __m128 b) { return _mm_max_ps(a, b); }

/*
 * Prefetch (SSE 1+, always available). Identical to AVX2/FMA path.
 */
static inline void colibri_sse41_prefetch(const void *p) { _mm_prefetch(p, _MM_HINT_T0); }

#endif /* __SSE2__ */
#endif /* COLIBRI_SSE41_KERNELS_H */
