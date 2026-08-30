/* Il nostro KDA contro i vettori generati da transformers 5.16.1. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../delta_attention.h"
#include "glm53_kda_case.h"   /* generato da tools/make_glm53_tiny.py */

int main(void) {
    const int H = GLM53_KDA_HEADS, D = GLM53_KDA_DIM, K = GLM53_KDA_KERNEL, S = GLM53_KDA_STEPS;
    const int width = H * D;
    float *state = calloc((size_t)H * D * D, sizeof(float));
    float *window = calloc((size_t)3 * width * K, sizeof(float));
    float *scratch = malloc((size_t)coli_kda_scratch_floats(H, D, D) * sizeof(float));
    float *out = malloc((size_t)S * width * sizeof(float));
    for (int s = 0; s < S; s++)
        if (coli_kda_step(out + (size_t)s * width, state, window,
                          glm53_kda_qkv + (size_t)s * 3 * width, glm53_kda_conv,
                          glm53_kda_decay + (size_t)s * width,
                          glm53_kda_beta + (size_t)s * H,
                          H, D, D, K, 1e-6f, scratch)) {
            printf("coli_kda_step ha fallito\n"); return 1;
        }
    float worst = 0.f; int at = 0;
    for (int i = 0; i < S * width; i++) {
        float d = fabsf(out[i] - glm53_kda_output[i]);
        if (d > worst) { worst = d; at = i; }
    }
    printf("KDA: %d passi x %d teste x %d, max|delta| = %.3e (indice %d: %.9g vs %.9g)\n",
           S, H, D, worst, at, out[at], glm53_kda_output[at]);
    if (worst > 2e-5f) { printf("FALLITO\n"); return 1; }
    printf("PASS KDA: combacia con l'oracolo transformers\n");
    return 0;
}
