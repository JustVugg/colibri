/* Exhaustive bit-exact gate for st.h's bf16_to_f32_bulk / f16_to_f32_bulk
 * (AVX2/SSE4.1/scalar tiers). BF16->F32 and F16->F32 are lossless IEEE-754
 * widenings: for every one of the 65536 possible input bit patterns -- every
 * normal and subnormal exponent, +-0, +-inf, and every NaN payload -- there
 * is exactly one correct output, so there is no tolerance to allow: the
 * vectorized tiers must produce bit-identical floats to the scalar reference
 * (bf16_to_f32/f16_to_f32) for the entire input space, not just a sample.
 *
 * Runs the full 65536-value sweep (which lands on an exact multiple of both
 * the AVX2 8-wide and SSE4.1 4-wide vector width) and a set of odd-length,
 * odd-offset slices of the same data so the scalar remainder tail in each
 * tier's loop is also exercised, not just the vectorized body. */
#include "../st.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

static int32_t bits_of(float f) { int32_t u; memcpy(&u, &f, 4); return u; }

static void check_range(const uint16_t *src, int64_t n, const char *label) {
    float *bulk_bf = malloc((size_t)n * sizeof(float));
    float *bulk_f16 = malloc((size_t)n * sizeof(float));
    if (!bulk_bf || !bulk_f16) { fprintf(stderr, "OOM\n"); exit(2); }
    bf16_to_f32_bulk(src, bulk_bf, n);
    f16_to_f32_bulk(src, bulk_f16, n);
    int64_t bf_diffs = 0, f16_diffs = 0;
    int64_t bf_first = -1, f16_first = -1;
    for (int64_t i = 0; i < n; i++) {
        float ref_bf = bf16_to_f32(src[i]);
        float ref_f16 = f16_to_f32(src[i]);
        if (bits_of(ref_bf) != bits_of(bulk_bf[i])) { bf_diffs++; if (bf_first < 0) bf_first = i; }
        if (bits_of(ref_f16) != bits_of(bulk_f16[i])) { f16_diffs++; if (f16_first < 0) f16_first = i; }
    }
    if (bf_diffs) fprintf(stderr, "  bf16 first mismatch idx=%lld h=0x%04x scalar=0x%08x bulk=0x%08x\n",
        (long long)bf_first, src[bf_first], bits_of(bf16_to_f32(src[bf_first])), bits_of(bulk_bf[bf_first]));
    if (f16_diffs) fprintf(stderr, "  f16 first mismatch idx=%lld h=0x%04x scalar=0x%08x bulk=0x%08x\n",
        (long long)f16_first, src[f16_first], bits_of(f16_to_f32(src[f16_first])), bits_of(bulk_f16[f16_first]));
    CHECK(bf_diffs == 0, "%s: bf16_to_f32_bulk %lld/%lld mismatches", label, (long long)bf_diffs, (long long)n);
    CHECK(f16_diffs == 0, "%s: f16_to_f32_bulk %lld/%lld mismatches", label, (long long)f16_diffs, (long long)n);
    free(bulk_bf); free(bulk_f16);
}

int main(void) {
    /* every possible 16-bit pattern, once each -- covers +-0, every normal
     * and subnormal exponent, +-inf, and every NaN payload for both formats */
    uint16_t *all = malloc(65536 * sizeof(uint16_t));
    if (!all) { fprintf(stderr, "OOM\n"); return 2; }
    for (int32_t h = 0; h <= 0xFFFF; h++) all[h] = (uint16_t)h;

    check_range(all, 65536, "full sweep");
    printf("st f16/bf16 simd: exhaustive 65536/65536 patterns exact\n");

    /* odd offsets/lengths so the scalar remainder tail (n%8 for AVX2, n%4 for
     * SSE4.1) actually runs, on real (not synthetic-zero) data */
    static const int64_t offs[] = {0, 1, 2, 3, 5, 7, 9, 13, 15, 16, 17, 100, 12345};
    static const int64_t lens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 65423, 65535};
    for (size_t oi = 0; oi < sizeof(offs)/sizeof(offs[0]); oi++) {
        for (size_t li = 0; li < sizeof(lens)/sizeof(lens[0]); li++) {
            int64_t off = offs[oi], len = lens[li];
            if (off + len > 65536) continue;
            char label[64]; snprintf(label, sizeof label, "off=%lld len=%lld", (long long)off, (long long)len);
            check_range(all + off, len, label);
        }
    }
    printf("st f16/bf16 simd: tail-remainder slices ok\n");

    free(all);
    if (failures) { fprintf(stderr, "st f16/bf16 simd: %d failure(s)\n", failures); return 1; }
    puts("st f16/bf16 simd: ok");
    return 0;
}
