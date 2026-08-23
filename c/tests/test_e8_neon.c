/* Le chemin NEON de matmul_e8 doit rendre EXACTEMENT le meme resultat que le
 * chemin scalaire de reference.
 *
 * La reference n'est pas une reimplementation : c'est `e8_expand_sub`, la
 * fonction que le moteur utilise deja pour deballer un sous-bloc, suivie d'un
 * produit scalaire naif. Si le noyau vectorise diverge d'elle, il change les
 * poids du modele -- exactement ce que la doctrine du projet interdit
 * ("never silently changes model precision").
 *
 * Tolerance : l'ordre d'accumulation differe entre scalaire et vectorise, donc
 * l'egalite bit-a-bit n'est pas attendue. On exige une erreur relative sous
 * 1e-5, soit bien en dessous du pas de quantification du format.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../quant.h"

/* Generateur deterministe : un echec doit etre rejouable a l'identique. */
static uint32_t graine = 12345u;
static uint32_t suivant(void){
    graine ^= graine << 13; graine ^= graine >> 17; graine ^= graine << 5;
    return graine;
}

/* Produit de reference, deballage par e8_expand_sub puis produit scalaire. */
static void matmul_e8_reference(float *y, const float *x, const uint8_t *q,
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
                    float w[E8_SUB];
                    e8_expand_sub(blk, ib, d, w);
                    int n = I - off < E8_SUB ? I - off : E8_SUB;
                    for(int k = 0; k < n; k++) acc += xs[off + k] * w[k];
                }
            }
            y[(int64_t)s * O + o] = acc;
        }
    }
}

int main(void){
    const int S = 4, I = 512, O = 8;          /* I multiple de E8_QK */
    int64_t rb = e8_rowbytes(I);

    uint8_t *q = malloc((size_t)O * rb);
    float *x = malloc(sizeof(float) * S * I);
    float *y_neon = malloc(sizeof(float) * S * O);
    float *y_ref  = malloc(sizeof(float) * S * O);
    if(!q || !x || !y_neon || !y_ref){ fprintf(stderr, "allocation\n"); return 2; }

    for(int64_t i = 0; i < (int64_t)O * rb; i++) q[i] = (uint8_t)(suivant() & 0xFF);
    for(int i = 0; i < S * I; i++)
        x[i] = ((float)(suivant() % 2000) - 1000.0f) / 1000.0f;

    matmul_e8_reference(y_ref, x, q, S, I, O);
    matmul_e8_neon(y_neon, x, q, NULL, S, I, O);

    double pire = 0.0;
    int i_pire = -1;
    for(int i = 0; i < S * O; i++){
        double a = y_ref[i], b = y_neon[i];
        double denom = fabs(a) > 1e-6 ? fabs(a) : 1e-6;
        double err = fabs(a - b) / denom;
        if(err > pire){ pire = err; i_pire = i; }
    }

    printf("  erreur relative max : %.3e (indice %d)\n", pire, i_pire);
    if(pire > 1e-5){
        printf("  ECHEC : reference %.9g, neon %.9g\n", y_ref[i_pire], y_neon[i_pire]);
        return 1;
    }
    printf("  OK : le chemin NEON est conforme au scalaire\n");
    free(q); free(x); free(y_neon); free(y_ref);
    return 0;
}
