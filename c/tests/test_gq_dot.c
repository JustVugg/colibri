/* The integer matmul against the same product in double precision, with the
 * activation ALREADY rounded to int8: this isolates the kernel (unpack, the
 * AVX2 dot, the min term, the per-group scales) from the activation's own
 * rounding. Reads: int32 type, int32 O, int32 I, int32 S, W bytes, S*I floats.
 * Prints the max error relative to sum|w*x| over all outputs. */
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../ggml_quants.h"
int main(void) {
#ifdef _WIN32   /* the blocks and floats go through stdin/stdout: no CRLF translation */
    _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY);
#endif
    int32_t t, O, I, S;
    if (fread(&t, 4, 1, stdin) != 1 || fread(&O, 4, 1, stdin) != 1 || fread(&I, 4, 1, stdin) != 1 || fread(&S, 4, 1, stdin) != 1) return 2;
    int64_t rb = gq_row_bytes(t, I); if (rb < 0) return 3;
    uint8_t *W = malloc((size_t)rb * O); float *x = malloc((size_t)S * I * 4), *y = malloc((size_t)S * O * 4);
    if (fread(W, 1, (size_t)rb * O, stdin) != (size_t)rb * O || fread(x, 4, (size_t)S * I, stdin) != (size_t)S * I) return 4;
    if (gq_matmul(y, x, W, t, S, I, O)) return 5;
    int8_t *xq = malloc((size_t)I); float *dx = malloc((size_t)(I / 32) * 4); int32_t *xs = malloc((size_t)(I / 16) * 4);
    float *wf = malloc((size_t)I * 4); double worst = 0;
    for (int s = 0; s < S; s++) {
        gq_quant_act(x + (int64_t)s * I, I, xq, dx, xs);
        for (int o = 0; o < O; o++) {
            gq_dequant_row(t, W + (int64_t)o * rb, wf, I);
            double ref = 0, mag = 0;
            for (int i = 0; i < I; i++) { double xv = (double)xq[i] * dx[i / 32]; ref += (double)wf[i] * xv; mag += fabs((double)wf[i] * xv); }
            double e = fabs((double)y[(int64_t)s * O + o] - ref) / (mag > 0 ? mag : 1);
            if (e > worst) worst = e;
        }
    }
    printf("%.3e\n", worst);
    return 0;
}
