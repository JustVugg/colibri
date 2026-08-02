/* test_ue8m0 — the UE8M0 block-scale decode, and the dtype table it arrived with.
 *
 * UE8M0 is an unsigned power-of-two exponent with no mantissa: the value is
 * 2^(v-127), and 0xff is NaN. Native fp8 checkpoints (DeepSeek-V4, and anything
 * with quantization_config.scale_fmt == "ue8m0") write block scales this way
 * instead of as f32.
 *
 * The whole domain is 256 values, so this checks all of them rather than
 * sampling. That is not thoroughness for its own sake: the first implementation
 * used the obvious bit trick, `(uint32_t)v << 23`, which is exact for
 * v in [1,254] and silently wrong at v == 0 -- the all-zero bit pattern is EXACT
 * ZERO in IEEE 754, not 2^-127. A weight block carrying that scale would have
 * been zeroed rather than made almost-zero. Sampling would not have found it;
 * the boundary is the whole bug.
 *
 * Also pins st_dtype_esz, because that function replaced three copies of a
 * `dtype==2 ? 4 : 2` ternary. That ternary was correct while only four dtypes
 * existed and would have claimed 2 bytes for an 8-byte I64 the moment a fifth
 * appeared. A wrong element size is an out-of-bounds read, not a wrong number.
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../st.h"

static int fails = 0;

static void check(int cond, const char *what)
{
    if (!cond) { printf("  FAIL %s\n", what); fails++; }
}

int main(void)
{
    /* --- the whole UE8M0 domain --- */
    int bad = 0;
    for (int v = 0; v < 256; v++) {
        float got = ue8m0_to_f32((uint8_t)v);
        if (v == 0xff) {
            if (!isnan(got)) { printf("  FAIL 0xff must be NaN, got %g\n", got); bad++; }
            continue;
        }
        double want = ldexp(1.0, v - 127);
        /* exact equality is the right test: every value is a power of two and
         * therefore representable, so "close enough" would hide a real error. */
        if ((double)got != want) {
            if (bad < 5) printf("  FAIL v=%d: got %g, want %g\n", v, got, want);
            bad++;
        }
    }
    check(bad == 0, "all 256 UE8M0 values decode exactly");
    if (bad == 0) printf("  ok   all 256 UE8M0 values decode exactly\n");

    /* --- the boundary that the bit trick got wrong --- */
    check(ue8m0_to_f32(0) != 0.0f, "v=0 is 2^-127, NOT zero (the bit-trick bug)");
    check((double)ue8m0_to_f32(0) == ldexp(1.0, -127), "v=0 == 2^-127 exactly");
    printf("  ok   v=0 -> %g (not zero)\n", (double)ue8m0_to_f32(0));

    /* --- the values a reader is most likely to hit --- */
    check(ue8m0_to_f32(127) == 1.0f,   "v=127 -> 1.0");
    check(ue8m0_to_f32(128) == 2.0f,   "v=128 -> 2.0");
    check(ue8m0_to_f32(126) == 0.5f,   "v=126 -> 0.5");
    /* the two scales actually observed in DeepSeek-V4's attention tensors */
    check((double)ue8m0_to_f32(115) == ldexp(1.0, -12), "v=115 -> 2^-12 (real checkpoint value)");
    check((double)ue8m0_to_f32(116) == ldexp(1.0, -11), "v=116 -> 2^-11 (real checkpoint value)");
    printf("  ok   1.0 / 2.0 / 0.5 and the two scales seen in a real checkpoint\n");

    /* --- element sizes: a wrong one here is an OOB read, not a wrong number --- */
    check(st_dtype_esz(0) == 2, "BF16 is 2 bytes");
    check(st_dtype_esz(1) == 2, "F16 is 2 bytes");
    check(st_dtype_esz(2) == 4, "F32 is 4 bytes");
    check(st_dtype_esz(3) == 1, "U8/I8 is 1 byte");
    check(st_dtype_esz(4) == 1, "F8_E4M3 is 1 byte");
    check(st_dtype_esz(5) == 1, "F8_E8M0 is 1 byte");
    check(st_dtype_esz(6) == 8, "I64 is 8 bytes");
    printf("  ok   element sizes for all seven dtype codes\n");

    /* --- the codes themselves must not move: containers on disk depend on the
     *     reader agreeing with what wrote them --- */
    check(st_dtype_code("BF16") == 0 && st_dtype_code("F16") == 1 &&
          st_dtype_code("F32")  == 2 && st_dtype_code("U8")  == 3 &&
          st_dtype_code("I8")   == 3, "existing dtype codes are unchanged");
    check(st_dtype_code("F8_E4M3") == 4 && st_dtype_code("float8_e4m3fn") == 4,
          "F8_E4M3 and its safetensors spelling both map to 4");
    check(st_dtype_code("F8_E8M0") == 5, "F8_E8M0 maps to 5");
    check(st_dtype_code("I64") == 6 && st_dtype_code("U64") == 6, "I64/U64 map to 6");
    printf("  ok   dtype codes: 0-3 unchanged, 4/5/6 added\n");

    printf("test_ue8m0: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
