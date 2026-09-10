/* olmoe's dot_i8_16 (c/olmoe.c, ARM NEON / AVX2 / SSE4.1 variants): must be
 * bit-for-bit identical to a plain scalar int8 dot product.
 *
 * Unlike qwen36's float GEMV SSE4.1 tiers (gsgemv.h/qgemv.h, tolerance-tested
 * because float addition is not associative), dot_i8_16 is pure integer
 * arithmetic -- sign-extend to int16, madd, horizontal sum, no rounding at
 * any step -- so every variant's own comment in olmoe.c claims exactness
 * ("Bit-for-bit identical to the AVX2 version above"). That claim previously
 * had no automated test backing it in this diff; this is it.
 *
 * Does NOT test matmul_q's IDOT path (that's tests/test_olmoe_matmul_q.c's
 * job, and deliberately non-exact -- activation quantization -- see that
 * file's header comment for why the two must not be conflated).
 *
 * Build: make -C c tests/test_olmoe_dot_i8_16
 *        make -C c tests/test_olmoe_dot_i8_16_sse41   (forces the SSE4.1 body)
 */
#define OLMOE_TESTING 1
#define main coli_olmoe_main_unused
#include "../olmoe.c"
#undef main

#include <stdio.h>
#include <stdlib.h>

static int32_t dot_i8_16_scalar_ref(const int8_t *a, const int8_t *b) {
    int32_t acc = 0;
    for (int i = 0; i < 16; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

int main(void) {
#if !defined(HAVE_FAST_DOT_I8)
    printf("test_olmoe_dot_i8_16: no fast dot_i8_16 compiled in this build "
           "(scalar-only -- nothing to test here)\n");
    return 0;
#else
    int fails = 0;
    const int N = 10000;

    /* dot_i8_16's SSE4.1 body loads a full 16-byte SSE register from a+8/b+8
     * and only consumes the low 8 bytes -- exactly like production callers'
     * larger buffers (e.g. matmul_q's `xi[4096]`), a 16-element array here
     * would make that load read 8 bytes past the end. Pad so the same call
     * pattern stays in-bounds for THIS test without touching dot_i8_16 itself. */
    srand(12345);
    for (int t = 0; t < N; t++) {
        int8_t a[24] = {0}, b[24] = {0};
        for (int i = 0; i < 16; i++) {
            a[i] = (int8_t)(rand() % 256 - 128);
            b[i] = (int8_t)(rand() % 256 - 128);
        }
        int32_t got = dot_i8_16(a, b);
        int32_t want = dot_i8_16_scalar_ref(a, b);
        if (got != want) {
            if (fails < 10) fprintf(stderr, "MISMATCH random t=%d: got %d want %d\n", t, got, want);
            fails++;
        }
    }

    /* Edge cases: all-zero, and the saturation-adjacent extremes (-128 has no
     * positive counterpart in int8, the one place a sign-extend bug would
     * show up first). */
    int8_t zero[24] = {0}, minv[24] = {0}, maxv[24] = {0};
    for (int i = 0; i < 16; i++) { minv[i] = -128; maxv[i] = 127; }
    struct { const char *name; const int8_t *a; const int8_t *b; } edge[] = {
        { "zero.zero", zero, zero },
        { "min.min",   minv, minv },
        { "min.max",   minv, maxv },
        { "max.max",   maxv, maxv },
    };
    for (size_t e = 0; e < sizeof(edge) / sizeof(edge[0]); e++) {
        int32_t got = dot_i8_16(edge[e].a, edge[e].b);
        int32_t want = dot_i8_16_scalar_ref(edge[e].a, edge[e].b);
        if (got != want) {
            fprintf(stderr, "MISMATCH edge %s: got %d want %d\n", edge[e].name, got, want);
            fails++;
        }
    }

    if (fails) {
        fprintf(stderr, "test_olmoe_dot_i8_16: %d failures\n", fails);
        return 1;
    }
    printf("test_olmoe_dot_i8_16: ALL PASS (0 failures), %d random pairs + 4 edge cases, bit-exact\n", N);
    return 0;
#endif
}
