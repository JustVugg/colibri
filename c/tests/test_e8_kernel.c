/* fmt=6 (E8/IQ3 lattice) CPU kernel oracle — #452 ladder step 3.
 *
 * tools/iq3_pack.py is the format's reference implementation; this checks that
 * matmul_e8 reads the same bytes to the same numbers. The fixture is generated
 * by that codec (tests/fixtures/e8_case.bin, written by tools/make_e8_fixture.py)
 * and carries, for one [O,I] tensor: the packed bytes, the codec's own dequant of
 * them, an activation row, and the reference y = dequant @ x computed in float64.
 *
 * Two properties are checked:
 *   1. dequant agreement — expanding each sub-block must reproduce the codec's
 *      floats bit-for-bit (same grid, same parity rule, same scale arithmetic)
 *   2. matmul agreement — y within float rounding of the float64 reference
 *
 * Build: cc -O2 -fopenmp tests/test_e8_kernel.c e8_grid.c -o tests/test_e8_kernel -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../quant.h"

static void *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s — run tools/make_e8_fixture.py\n", path);
        exit(77);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)sz);
    if (fread(p, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "short read\n");
        exit(1);
    }
    fclose(f);
    *n = (size_t)sz;
    return p;
}

int main(void) {
    size_t nbytes;
    uint8_t *raw = slurp("tests/fixtures/e8_case.bin", &nbytes);
    /* header: int32 O, I, then packed[O*rowbytes], deq[O*I], x[I], yref[O] */
    int32_t O, I;
    memcpy(&O, raw, 4);
    memcpy(&I, raw + 4, 4);
    size_t rb = (size_t)e8_rowbytes(I);
    const uint8_t *packed = raw + 8;
    const float *deq = (const float *)(packed + (size_t)O * rb);
    const float *x = deq + (size_t)O * I;
    const float *yref = x + I;

    /* 1. dequant agreement, sub-block by sub-block */
    int mism = 0;
    float w[E8_SUB];
    for (int o = 0; o < O && mism < 5; o++) {
        const uint8_t *row = packed + (size_t)o * rb;
        for (int64_t b = 0; b < e8_blocks(I); b++) {
            const uint8_t *blk = row + b * E8_BBYTES;
            uint16_t dh;
            memcpy(&dh, blk + 96, 2);
            float d = e8_fp16_to_f32(dh);
            for (int ib = 0; ib < E8_QK / E8_SUB; ib++) {
                int off = (int)(b * E8_QK) + ib * E8_SUB;
                if (off >= I) break;
                e8_expand_sub(blk, ib, d, w);
                int n = I - off < E8_SUB ? I - off : E8_SUB;
                for (int k = 0; k < n; k++) {
                    float got = w[k], want = deq[(size_t)o * I + off + k];
                    if (fabsf(got - want) > 1e-6f * (fabsf(want) + 1e-6f)) {
                        if (mism < 5)
                            fprintf(stderr, "dequant mismatch o=%d i=%d: got %.9g want %.9g\n", o, off + k, got, want);
                        mism++;
                    }
                }
            }
        }
    }
    if (mism) {
        printf("FAIL: %d dequant mismatches\n", mism);
        return 1;
    }

    /* 2. matmul agreement */
    float *y = malloc((size_t)O * sizeof(float));
    matmul_e8(y, x, packed, NULL, 1, I, O);
    double worst = 0;
    for (int o = 0; o < O; o++) {
        double d = fabs((double)y[o] - (double)yref[o]);
        double rel = d / (fabs((double)yref[o]) + 1e-6);
        if (rel > worst) worst = rel;
    }
    printf("e8 kernel oracle: O=%d I=%d, dequant exact, matmul worst rel %.2e\n", O, I, worst);
    if (worst > 1e-5) {
        printf("FAIL\n");
        return 1;
    }

    /* 3. rotation agreement (converter step, #452): the fixture carries a
     * rotated-weights section — packed W@Q from the Python side, the raw
     * activation, Python's Q^T x, and the float64 reference product. The C
     * side must reproduce the rotation (regenerated signs + block tiling +
     * FWHT) and then the matmul, or every rotated container decodes wrong. */
    const uint8_t *sec = (const uint8_t *)(yref + O);
    if ((size_t)(sec - raw) < nbytes) {
        int32_t O2, I2;
        memcpy(&O2, sec, 4);
        memcpy(&I2, sec + 4, 4);
        size_t rb2 = (size_t)e8_rowbytes(I2);
        const uint8_t *packed2 = sec + 8;
        const float *x2 = (const float *)(packed2 + (size_t)O2 * rb2);
        const float *xrot2 = x2 + I2;
        const float *y2ref = xrot2 + I2;
        float *xr = malloc((size_t)I2 * sizeof(float));
        memcpy(xr, x2, (size_t)I2 * sizeof(float));
        e8_rot_rows(xr, 1, I2);
        double wrot = 0;
        for (int i = 0; i < I2; i++) {
            double d = fabs((double)xr[i] - (double)xrot2[i]);
            double rel = d / (fabs((double)xrot2[i]) + 1e-6);
            if (rel > wrot) wrot = rel;
        }
        float *y2 = malloc((size_t)O2 * sizeof(float));
        matmul_e8(y2, xr, packed2, NULL, 1, I2, O2);
        double wmm = 0;
        for (int o = 0; o < O2; o++) {
            double d = fabs((double)y2[o] - (double)y2ref[o]);
            double rel = d / (fabs((double)y2ref[o]) + 1e-6);
            if (rel > wmm) wmm = rel;
        }
        printf("e8 rotation oracle: O2=%d I2=%d, rot worst rel %.2e, matmul worst rel %.2e\n", O2, I2, wrot, wmm);
        if (wrot > 1e-5 || wmm > 1e-5) {
            printf("FAIL\n");
            return 1;
        }
        free(xr);
        free(y2);
    } else {
        fprintf(stderr, "fixture has no rotated section — regenerate with tools/make_e8_fixture.py\n");
        return 1;
    }
    printf("OK\n");
    free(y);
    free(raw);
    return 0;
}
