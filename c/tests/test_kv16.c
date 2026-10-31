/* kv16.h: the float16 rounding the KV cache uses, its integer reference (the Vulkan shader's
 * and the CUDA kernel's arithmetic, chain_ew.comp and cuda_chain.cu op 11) against the
 * compiler's IEEE conversion, and the integer path's conversions both ways.
 *   make tests/test_kv16 && ./tests/test_kv16 [step]     (the _Float16 path; step 1: every float)
 *   make tests/test_kv16_integer && ./tests/test_kv16_integer   (-DKV16_INTEGER) */
#include "../kv16.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float flt(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static int check(uint32_t u, long *bad) {
    float x = flt(u);
    uint32_t want = bits(kv16_to(kv16_from(x)));            /* the conversion the CPU's cache does */
    uint32_t got = kv16_round_bits(bits(kv16_clamp(x)));    /* the devices' arithmetic */
    if (isnan(x)) return isnan(flt(want)) && isnan(flt(got)) ? 0 : (++*bad, 1);
    if (want != got) {
        if (*bad < 8) fprintf(stderr, "x=%a (0x%08x): conversion 0x%08x, integer 0x%08x\n", x, u, want, got);
        ++*bad; return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    long bad = 0, n = 0;
    uint32_t step = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 0) : 4099;   /* 1: every float */
    for (uint64_t u = 0; u <= 0xffffffffull; u += step, n++) check((uint32_t)u, &bad);
    /* the edges: around 2^-25, 2^-24, 2^-14 and 65504, each sign, ties included */
    const float edge[] = {0.f, 2.9802322e-8f, 5.9604645e-8f, 8.940697e-8f, 6.097555e-5f, 6.1035156e-5f, 6.1094761e-5f,
                          1.f, 1.00048828125f, 1.000732421875f, 65504.f, 65519.f, 65520.f, 1e30f, INFINITY};
    for (unsigned i = 0; i < sizeof edge / sizeof *edge; i++)
        for (int d = -2; d <= 2; d++) for (int s = 0; s < 2; s++, n++) check((bits(edge[i]) + d) ^ (s ? 0x80000000u : 0), &bad);
    /* the array loops (eight at a time where the host has F16C) against the scalar conversion,
     * value by value: every float the step samples, 64 a call (a tail of 3 after each 61) */
    {
        float xs[64], rr[64], un[64]; uint16_t pk[64];
        int k = 0, len = 61;
        for (uint64_t u = 0; u <= 0xffffffffull; u += step | 1u) {
            xs[k++] = flt((uint32_t)u);
            if (k < len) continue;
            memcpy(rr, xs, sizeof rr); kv16_round(rr, len); kv16_pack(pk, xs, len); kv16_unpack(un, pk, len);
            for (int j = 0; j < len; j++, n++) {
                uint16_t h = kv16_from(xs[j]); float w = kv16_to(h);
                int nan = isnan(w), ok = nan ? isnan(rr[j]) && isnan(un[j]) : bits(rr[j]) == bits(w) && pk[j] == h && bits(un[j]) == bits(w);
                if (!ok) { if (bad < 8) fprintf(stderr, "array loops, x=%a: round 0x%08x pack 0x%04x unpack 0x%08x, scalar 0x%04x\n",
                                                xs[j], bits(rr[j]), pk[j], bits(un[j]), h); ++bad; }
            }
            k = 0; len = len == 61 ? 64 : 61;
        }
    }
    /* the stored half widens back to what was rounded, and clamping never makes an infinity */
    for (uint32_t h = 0; h < 0x10000u; h++) {
        float f = kv16_to((uint16_t)h);
        if (isnan(f)) continue;
        if (!isinf(f) && kv16_from(f) != h && !(h == 0x8000u && kv16_from(f) == 0x8000u)) { ++bad; fprintf(stderr, "half 0x%04x does not come back\n", h); }
    }
    if (isinf(kv16_to(kv16_from(1e30f))) || kv16_to(kv16_from(-1e30f)) != -65504.f) { ++bad; fprintf(stderr, "clamp\n"); }
    /* attention's two loops over a float16 row: the bits of the float loops over the floats it holds */
    uint32_t r = 12345u;
    for (int I = 1; I <= 300; I += (I < 70 ? 1 : 37)) {
        float x[300], wf[300], y[300], z[300];
        uint16_t w[300];
        for (int k = 0; k < I; k++) {
            r = r * 1664525u + 1013904223u; x[k] = (float)((int)(r >> 8) - (1 << 23)) / (1 << 21);
            r = r * 1664525u + 1013904223u; w[k] = kv16_from((float)((int)(r >> 8) - (1 << 23)) / (1 << 20));
            wf[k] = kv16_to(w[k]); y[k] = z[k] = x[k] * 0.5f;
        }
        if (bits(kv16_dot_lanes(x, w, I)) != bits(dot_f32_lanes(x, wf, I))) { ++bad; fprintf(stderr, "dot, %d elements\n", I); }
        float a = x[0] * 0.75f;
        kv16_axpy(y, a, w, I);
        /* the float row's loop as -O2 and up compile it, fused (at -O1 gcc leaves it unfused) */
        for (int k = 0; k < I; k++) z[k] = MATMUL_F32_MADD(z[k], a, wf[k]);
        if (memcmp(y, z, (size_t)I * sizeof(float))) { ++bad; fprintf(stderr, "axpy, %d elements\n", I); }
        n += 2;
    }
    printf("kv16: %ld values, %ld differ (%s path%s)\n", n, bad,
#if defined(__FLT16_MAX__) && !defined(KV16_INTEGER)
           "_Float16",
#else
           "integer",
#endif
#ifdef KV16_F16C
           ", F16C loops"
#else
           ", scalar loops"
#endif
#ifdef KV16_CVT8
           ", F16C conversions"
#endif
           );
    return bad != 0;
}
