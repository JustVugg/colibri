/* Un esperto del checkpoint convertito, letto come lo legge il motore.
 *
 * Il contenitore int4 gs64 e' piatto: nibble impacchettati due per byte e una
 * scala ogni 64 colonne, senza forma scritta nel file. Fra il converter che lo
 * scrive e il motore che lo legge ci sono l'ordine dei nibble, il verso dei
 * gruppi e il significato delle scale, e nessuna di queste cose fallisce in
 * modo rumoroso: si sbaglia e si continua a calcolare.
 *
 * Questo confronta un matvec vero con quello che ne fa numpy sugli stessi
 * byte. Serve un checkpoint gia' convertito, quindi non gira in CI: si lancia
 * a mano dopo una conversione, ed e' il primo controllo da fare quando il
 * modello vero risponde in modo strano.
 *
 * USO:
 *   gcc -O2 -std=gnu11 -Ic -o check_container c/tools/check_glm53_container.c -lm
 *   ./check_container DIR NOME_TENSORE RIGHE COLONNE
 *
 * Il riferimento numpy si ottiene dequantizzando lo stesso tensore con lo
 * stesso vettore di prova, che e' deterministico: x[i] = ((i*37 %% 199)-99)/100.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "json.h"
#include "st.h"
#include "quant.h"
int main(int c, char **v) {
    shards S; st_init(&S, v[1]);
    const char *name = v[2];
    int rows = atoi(v[3]), cols = atoi(v[4]);
    char qn[600]; snprintf(qn, sizeof(qn), "%s.qs", name);
    st_tensor *t = st_find(&S, name), *q = st_find(&S, qn);
    if (!t || !q) { printf("mancante\n"); return 1; }
    if (t->nbytes != (int64_t)rows*cols/2 || q->numel != (int64_t)rows*cols/64) {
        printf("GEOMETRIA NON TORNA: %lld byte, %lld scale\n",
               (long long)t->nbytes, (long long)q->numel); return 1; }
    uint8_t *packed = malloc((size_t)t->nbytes);
    float *scale = malloc((size_t)q->numel * sizeof(float));
    st_read_raw(&S, name, packed, 0);
    st_read_f32_cap(&S, qn, scale, q->numel, 0);
    float *x = malloc((size_t)cols * sizeof(float));
    for (int i = 0; i < cols; i++) x[i] = (float)((i * 37 % 199) - 99) / 100.0f;
    float *y = malloc((size_t)rows * sizeof(float));
    matmul_i4_grouped(y, x, packed, scale, 1, cols, rows, 64);
    printf("y[0..4] = %.9g %.9g %.9g %.9g %.9g\n", y[0], y[1], y[2], y[3], y[4]);
    double s = 0; for (int i = 0; i < rows; i++) s += y[i];
    printf("somma %.9g\n", s);
    return 0;
}
