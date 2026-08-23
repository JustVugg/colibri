/* Lequel des deux chemins est le plus proche de la verite ?
 *
 * Mon premier test prenait le scalaire pour reference. C'est un oracle
 * douteux : il accumule 6144 termes dans UN flottant simple precision, alors
 * que NEON repartit sur 8 accumulateurs -- ce qui reduit normalement l'erreur
 * d'arrondi. L'ecart mesure entre les deux ne dit donc pas qui se trompe.
 *
 * On tranche avec une reference en double precision, qui elle fait autorite.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../quant.h"

static uint32_t graine = 24681357u;
static uint32_t suivant(void){
    graine ^= graine << 13; graine ^= graine >> 17; graine ^= graine << 5;
    return graine;
}

/* Verite terrain : meme deballage, accumulation en double. */
static void matmul_e8_double(double *y, const float *x, const uint8_t *q,
                             int S, int I, int O){
    int64_t nb = e8_blocks(I), rb = e8_rowbytes(I);
    for(int o = 0; o < O; o++){
        const uint8_t *wrow = q + (int64_t)o * rb;
        for(int s = 0; s < S; s++){
            const float *xs = x + (int64_t)s * I;
            double acc = 0;
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
                    for(int k = 0; k < n; k++) acc += (double)xs[off + k] * (double)w[k];
                }
            }
            y[(int64_t)s * O + o] = acc;
        }
    }
}

static void matmul_e8_scalaire(float *y, const float *x, const uint8_t *q,
                               int S, int I, int O){
    int64_t nb = e8_blocks(I), rb = e8_rowbytes(I);
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
                    float w[E8_SUB]; e8_expand_sub(blk, ib, d, w);
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

static double ecart_max(const double *ref, const float *v, int n){
    double pire = 0;
    for(int i = 0; i < n; i++){
        double den = fabs(ref[i]) > 1e-6 ? fabs(ref[i]) : 1e-6;
        double e = fabs(ref[i] - (double)v[i]) / den;
        if(e > pire) pire = e;
    }
    return pire;
}

int main(void){
    const int S = 64, I = 6144, O = 256;
    int64_t rb = e8_rowbytes(I);
    uint8_t *q = malloc((size_t)O * rb);
    float *x = malloc(sizeof(float) * S * I);
    float *ys = malloc(sizeof(float) * S * O);
    float *yn = malloc(sizeof(float) * S * O);
    double *yd = malloc(sizeof(double) * S * O);

    for(int64_t i = 0; i < (int64_t)O * rb; i++) q[i] = (uint8_t)(suivant() & 0xFF);
    for(int i = 0; i < S * I; i++) x[i] = ((float)(suivant() % 2000) - 1000.0f) / 1000.0f;

    matmul_e8_double(yd, x, q, S, I, O);
    matmul_e8_scalaire(ys, x, q, S, I, O);
    matmul_e8_neon(yn, x, q, NULL, S, I, O);

    double e_sc = ecart_max(yd, ys, S * O);
    double e_ne = ecart_max(yd, yn, S * O);

    printf("  reference : double precision, S=%d I=%d O=%d\n", S, I, O);
    printf("  ecart scalaire : %.3e\n", e_sc);
    printf("  ecart NEON     : %.3e\n", e_ne);
    if(e_ne <= e_sc){
        printf("  OK : NEON est AUSSI PRECIS OU MEILLEUR que le scalaire\n");
        return 0;
    }
    printf("  ECHEC : NEON degrade la precision (%.2fx pire)\n", e_ne / e_sc);
    return 1;
}
