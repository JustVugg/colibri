/* Compare scalaire et NEON sur les dimensions REELLES d'un expert GLM-5.2 :
 * moe_intermediate_size=2048 lignes de sortie, hidden_size=6144 en entree.
 * S=1 correspond au decodage (un token), S=64 a un lot de prefill.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../quant.h"

static uint32_t graine = 987654321u;
static uint32_t suivant(void){
    graine ^= graine << 13; graine ^= graine >> 17; graine ^= graine << 5;
    return graine;
}

static double maintenant(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Reference scalaire : le chemin qu'emprunte reellement aarch64 aujourd'hui. */
static void matmul_e8_scalaire(float *y, const float *x, const uint8_t *q,
                               int S, int I, int O){
    int64_t nb = e8_blocks(I), rb = e8_rowbytes(I);
    #pragma omp parallel for schedule(static)
    for(int o = 0; o < O; o++){
        const uint8_t *wrow = q + (int64_t)o * rb;
        for(int s = 0; s < S; s++){
            const float *xs = x + (int64_t)s * I;
            float acc = 0;
            for(int64_t b = 0; b < nb; b++){
                const uint8_t *blk = wrow + b * E8_BBYTES;
                uint16_t dh; memcpy(&dh, blk + 96, 2);
                float d = e8_fp16_to_f32(dh);
                int base = (int)(b * E8_QK);
                for(int ib = 0; ib < E8_QK / E8_SUB; ib++){
                    int off = base + ib * E8_SUB;
                    if(off >= I) break;
                    float w[E8_SUB];
                    e8_expand_sub(blk, ib, d, w);
                    int n = I - off < E8_SUB ? I - off : E8_SUB;
                    float a = 0;
                    for(int k = 0; k < n; k++) a += xs[off + k] * w[k];
                    acc += a;
                }
            }
            y[(int64_t)s * O + o] = acc;
        }
    }
}

static void mesure(int S, int I, int O, int repet){
    int64_t rb = e8_rowbytes(I);
    uint8_t *q = malloc((size_t)O * rb);
    float *x = malloc(sizeof(float) * S * I);
    float *ys = malloc(sizeof(float) * S * O);
    float *yn = malloc(sizeof(float) * S * O);
    for(int64_t i = 0; i < (int64_t)O * rb; i++) q[i] = (uint8_t)(suivant() & 0xFF);
    for(int i = 0; i < S * I; i++) x[i] = ((float)(suivant() % 2000) - 1000.0f) / 1000.0f;

    matmul_e8_scalaire(ys, x, q, S, I, O);          /* rechauffe les caches */
    double t0 = maintenant();
    for(int r = 0; r < repet; r++) matmul_e8_scalaire(ys, x, q, S, I, O);
    double t_sc = (maintenant() - t0) / repet;

    matmul_e8_neon(yn, x, q, NULL, S, I, O);
    t0 = maintenant();
    for(int r = 0; r < repet; r++) matmul_e8_neon(yn, x, q, NULL, S, I, O);
    double t_ne = (maintenant() - t0) / repet;

    double pire = 0;
    for(int i = 0; i < S * O; i++){
        double a = ys[i], b = yn[i];
        double den = fabs(a) > 1e-6 ? fabs(a) : 1e-6;
        double e = fabs(a - b) / den; if(e > pire) pire = e;
    }

    printf("  S=%-3d I=%d O=%d | scalaire %8.2f ms | NEON %8.2f ms | gain %5.2fx | ecart %.1e\n",
           S, I, O, t_sc * 1e3, t_ne * 1e3, t_sc / t_ne, pire);
    free(q); free(x); free(ys); free(yn);
}

int main(void){
    printf("  dimensions d'un expert GLM-5.2 (hidden 6144, moe_intermediate 2048)\n");
    mesure(1,  6144, 2048, 5);    /* decodage : un token */
    mesure(8,  6144, 2048, 3);    /* petit lot */
    mesure(64, 6144, 2048, 2);    /* lot de prefill */
    return 0;
}
