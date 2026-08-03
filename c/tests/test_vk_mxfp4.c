/* fmt=7 (MXFP4) Vulkan matmul vs the CPU reference kernel (quant.h).
 * Random e2m1 nibbles + ue8m0 group exponents, K3 expert dims.
 * Skips (exit 0) when no Vulkan device is available. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../quant.h"
#include "../backend_vulkan.h"

int main(void) {
    if (!coli_vk_init("shaders/qmatmul.spv") || !coli_vk_available()) {
        fprintf(stderr, "vk-mxfp4: no Vulkan device — skipped\n");
        return 0;
    }
    srand(7);
    int S = 2, I = 3584, O = 3072; /* K3 expert w1/w3 shape */
    int rb = (I + 1) / 2, ng = (I + 31) / 32;
    uint8_t *q4 = malloc((size_t)O * rb), *e8 = malloc((size_t)O * ng);
    float *x = malloc((size_t)S * I * sizeof(float));
    float *yc = calloc((size_t)S * O, sizeof(float)), *yg = calloc((size_t)S * O, sizeof(float));
    float *sc = malloc((size_t)O * ng * sizeof(float));
    for (size_t i = 0; i < (size_t)O * rb; i++) q4[i] = (uint8_t)rand();
    for (size_t i = 0; i < (size_t)O * ng; i++) {
        e8[i] = (uint8_t)(120 + rand() % 12);
        sc[i] = mx4_scale(e8[i]);
    }
    for (int i = 0; i < S * I; i++) x[i] = (float)(rand() % 2001 - 1000) / 500.f;

    matmul_mxfp4(yc, x, q4, e8, S, I, O);

    ColiVkTensor *t = NULL;
    if (!coli_vk_matmul(&t, yg, x, q4, sc, 7, S, I, O, 32)) {
        fprintf(stderr, "vk-mxfp4: FAIL coli_vk_matmul fmt=7 refused\n");
        return 1;
    }
    double num = 0, den = 0, mx = 0;
    for (int i = 0; i < S * O; i++) {
        double d = yg[i] - yc[i];
        num += d * d;
        den += (double)yc[i] * yc[i];
        double r = fabs(d) / (fabs(yc[i]) + 1e-6);
        if (r > mx) mx = r;
    }
    double rel = sqrt(num / (den + 1e-30));
    printf("vk-mxfp4: rel_l2 %.3e  max_rel %.3e  (S=%d I=%d O=%d)\n", rel, mx, S, I, O);
    if (rel > 1e-5) {
        fprintf(stderr, "vk-mxfp4: FAIL rel_l2 %.3e\n", rel);
        return 1;
    }
    printf("vk-mxfp4: OK\n");
    coli_vk_shutdown();
    return 0;
}
