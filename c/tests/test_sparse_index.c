/* Indexer k-pool e attenzione sparsa contro i vettori transformers. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../sparse_index.h"
#include "glm53_indexer_case.h"
#include "glm53_sparse_attention_case.h"

int main(void) {
    int ok = 1;
    /* --- selezione dei blocchi --- */
    const int SQ=GLM53_INDEX_SEQUENCE, H=GLM53_INDEX_HEADS, D=GLM53_INDEX_DIM;
    const int P=GLM53_INDEX_POOL, TK=GLM53_INDEX_TOPK, W=GLM53_INDEX_WIDTH;
    int *sel = malloc((size_t)SQ*W*sizeof(int));
    if (coli_sparse_index_select(sel, glm53_index_queries, glm53_index_keys,
                                 glm53_index_gates, glm53_index_weights, glm53_index_ape,
                                 glm53_index_valid, SQ, H, D, P, TK, 1)) {
        printf("select ha fallito\n"); return 1; }
    int wrong = -1;
    for (int i = 0; i < SQ*W; i++)
        if (sel[i] != glm53_index_expected[i]) { wrong = i; break; }
    if (wrong < 0) printf("indexer k-pool: %d query x %d slot, indici IDENTICI\n", SQ, W);
    else { printf("indexer: differenza allo slot %d (%d vs %d atteso)\n",
                  wrong, sel[wrong], glm53_index_expected[wrong]); ok = 0;
           for (int q=0;q<SQ;q++){ printf("  q%d nostro:",q);
             for(int j=0;j<W;j++) printf(" %3d",sel[q*W+j]);
             printf("  atteso:"); for(int j=0;j<W;j++) printf(" %3d",glm53_index_expected[q*W+j]);
             printf("\n"); } }

    /* --- attenzione sui blocchi selezionati --- */
    const int S2=GLM53_SA_SEQUENCE, H2=GLM53_SA_HEADS, KD=GLM53_SA_KEY_DIM, VD=GLM53_SA_VALUE_DIM;
    float *out = malloc((size_t)S2*H2*VD*sizeof(float));
    if (coli_sparse_attention(out, glm53_sa_queries, glm53_sa_keys, glm53_sa_values,
                              glm53_sa_indices, S2, GLM53_SA_WIDTH, H2, KD, VD)) {
        printf("attention ha fallito\n"); return 1; }
    float worst=0.f; int at=0;
    for (int i=0;i<S2*H2*VD;i++){ float d=fabsf(out[i]-glm53_sa_expected[i]);
        if(d>worst){worst=d;at=i;} }
    printf("attenzione sparsa: max|delta| = %.3e (indice %d: %.9g vs %.9g)\n",
           worst, at, out[at], glm53_sa_expected[at]);
    if (worst > 2e-5f) ok = 0;
    printf("\n%s\n", ok ? "PASS: entrambi combaciano con l'oracolo" : "FALLITO");
    return ok ? 0 : 1;
}
