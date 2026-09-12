/* The vector e4m3 decoder must agree with the table, for every byte.
 *
 * quant.h decodes e4m3 with a 256-entry lookup; e4m3_decode8 does it with two
 * shifts and a multiply, because a lookup per byte is what made V4.1's dense
 * attention 41% of a turn. The two must not diverge anywhere, and "anywhere"
 * is 256 values, so this checks all of them rather than a sample.
 *
 * Two cases are the reason this test exists rather than a comment:
 *   - the subnormals, 0x01 to 0x07, which the trick produces by scaling a
 *     subnormal float into the normal range. A build with flush-to-zero, or a
 *     -ffast-math that implies it, turns them into zero and nothing else
 *     notices: they are the smallest weights in the model.
 *   - 0x7F and 0xFF, e4m3's NaNs, which the arithmetic would make into +-480.
 *     A corrupt weight that becomes a plausible one is worse than one that
 *     poisons the output, so those are blended back and asserted here.
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "../quant.h"

int main(void) {
#ifndef __AVX2__
    printf("SKIP: built without AVX2, the scalar table is the only decoder\n");
    return 0;
#else
    uint8_t bytes[256];
    for (int i = 0; i < 256; i++) bytes[i] = (uint8_t)i;
    int bad = 0, subnormals = 0, nans = 0;
    for (int base = 0; base < 256; base += 8) {
        float got[8];
        _mm256_storeu_ps(got, e4m3_decode8(bytes + base));
        for (int k = 0; k < 8; k++) {
            uint8_t b = bytes[base + k];
            float want = e4m3_decode(b);
            int both_nan = isnan(got[k]) && isnan(want);
            if (isnan(want)) nans++;
            if ((b & 0x7F) && !(b & 0x78)) subnormals++;
            if (!both_nan && memcmp(&got[k], &want, sizeof(float))) {
                if (bad < 8)
                    printf("  FAIL 0x%02X: vector %.9g, table %.9g\n", b, got[k], want);
                bad++;
            }
        }
    }
    printf("e4m3: %d/256 bytes identical (%d subnormals, %d NaNs)\n",
           256 - bad, subnormals, nans);
    if (subnormals != 14 || nans != 2) {
        printf("FAIL: the encoding is not what this test was written against\n");
        return 1;
    }
    printf("%s\n", bad ? "FAILED" : "PASS");
    return bad ? 1 : 0;
#endif
}
