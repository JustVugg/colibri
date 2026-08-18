/* Zero-copie CUDA sur memoire unifiee : le GPU doit pouvoir calculer un expert
 * DEPUIS le tampon hote ou le streaming vient de l ecrire, sans recopie.
 *
 * C est la symetrie de ce que fait deja la branche Metal
 * (newBufferWithBytesNoCopy + registre de slabs). Le test verifie les deux
 * proprietes qui comptent :
 *   1. le resultat est celui du chemin CPU (a l epsilon pres) ;
 *   2. le tenseur ne POSSEDE pas ses poids -- sinon il y a eu copie, et la
 *      liberation detruirait le tampon de streaming du moteur.
 *
 * Sur un GPU discret, prop.integrated vaut 0 : le test s abstient plutot que
 * d echouer, car le zero-copie n y a pas de sens.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../quant.h"
#include "../backend_cuda.h"

static uint32_t graine = 4242u;
static uint32_t suivant(void){
    graine ^= graine << 13; graine ^= graine >> 17; graine ^= graine << 5;
    return graine;
}

int main(void){
    int devs[1] = {0};
    if(!coli_cuda_init(devs, 1)){ printf("  (pas de CUDA) - test ignore\n"); return 0; }
    /* fmt=6 decode contre le livre de codes E8 : sans lui les noyaux rendent
     * du bruit (ou zero). Le moteur le publie au demarrage, pas ce test. */
    if(!coli_cuda_e8_set_grid(e8_grid)){ printf("  ECHEC : publication du livre de codes E8\n"); return 1; }
    if(!coli_cuda_device_integrated(0)){
        printf("  (GPU discret : le zero-copie ne s applique pas) — test ignore\n");
        return 0;
    }

    const int S = 4, I = 6144, O = 256;
    int64_t rb = e8_rowbytes(I);
    size_t taille = ((size_t)O * rb + 16383) & ~(size_t)16383;   /* aligne page, comme Metal */

    void *slab = NULL;
    if(posix_memalign(&slab, 16384, taille)){ fprintf(stderr, "OOM\n"); return 2; }
    for(size_t i = 0; i < (size_t)O * rb; i++) ((uint8_t*)slab)[i] = (uint8_t)(suivant() & 0xFF);

    float *x = malloc(sizeof(float) * S * I);
    float *y_gpu = malloc(sizeof(float) * S * O);
    float *y_cpu = malloc(sizeof(float) * S * O);
    for(int i = 0; i < S * I; i++)
        x[i] = ((float)(suivant() % 2000) - 1000.0f) / 1000.0f;

    /* AUCUN enregistrement : sur GB10 (pageableMemoryAccess=1) le pointeur
     * hote est deja adressable par le GPU. C est precisement ce qu on teste. */
    printf("  pageable sans enregistrement : %d\n", coli_cuda_device_pageable(0));

    ColiCudaTensor *t = NULL;
    if(!coli_cuda_tensor_wrap(&t, slab, NULL, 6 /*fmt E8*/, I, O, 0 /*device*/, 0 /*gs*/)){
        printf("  ECHEC : impossible d envelopper le slab en tenseur\n"); return 1;
    }
    if(coli_cuda_tensor_owns_weights(t)){
        printf("  ECHEC : le tenseur POSSEDE ses poids — il y a eu copie\n"); return 1;
    }

    if(!coli_cuda_matmul(&t, y_gpu, x, slab, NULL, 6, S, I, O, 0, 0)){
        printf("  ECHEC : le calcul CUDA a refuse\n"); return 1;
    }
    matmul_e8_neon(y_cpu, x, (const uint8_t*)slab, NULL, S, I, O);

    double pire = 0; int ip = -1;
    for(int i = 0; i < S * O; i++){
        double a = y_cpu[i], b = y_gpu[i];
        double den = fabs(a) > 1e-6 ? fabs(a) : 1e-6;
        double e = fabs(a - b) / den;
        if(e > pire){ pire = e; ip = i; }
    }
    printf("  ecart GPU/CPU max : %.3e (indice %d)\n", pire, ip);

    coli_cuda_tensor_free(t);
    /* Le slab doit AVOIR SURVECU a la liberation du tenseur : c est la memoire
     * du moteur, pas celle du backend. On la relit pour le prouver. */
    volatile uint8_t sonde = ((uint8_t*)slab)[16];
    (void)sonde;
    free(slab);

    if(pire > 1e-3){ printf("  ECHEC : resultat GPU divergent\n"); return 1; }
    printf("  OK : le GPU a calcule depuis le tampon hote, sans copie\n");
    return 0;
}
