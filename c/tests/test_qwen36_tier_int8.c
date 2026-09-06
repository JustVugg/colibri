/* Il tier VRAM deve promuovere anche gli esperti int8 (#1331).
 *
 * Il difetto: qt_plan_fill riservava budget e metteva planned=1 senza guardare
 * il formato, mentre la promozione esigeva i puntatori int4. Su un container
 * int8 quei puntatori sono NULL perche' non c'e' niente da impacchettare, quindi
 * il budget restava riservato, il flag alzato -- e "if(resident||queued||planned)
 * continue" garantiva che quell'esperto non venisse piu' riconsiderato. Zero
 * esperti in VRAM per tutta la vita del processo, nessun messaggio, e un tier
 * che dall'esterno sembrava acceso e freddo.
 *
 * Il backend qui e' finto e REGISTRA cio' che riceve: e' l'unico modo di provare
 * su CI senza GPU la cosa che conta davvero, cioe' che qualcosa arrivi davvero
 * in VRAM e nel formato giusto. Un test che si fermasse a "qt_init ritorna 1"
 * sarebbe passato anche col difetto. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- backend CUDA finto: le firme vengono da backend_cuda.h, che il tier
   include per conto suo; il corpo (condiviso con gli altri test del tier) e'
   in qwen36_fake_cuda.h. ---- */
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

/* Aspetta che l'uploader asincrono abbia svuotato la coda, senza dormire a caso. */
static void drain(void) {
    for (int i = 0; i < 500 && fake_uploads == 0; i++) {
        struct timespec ts = {0, 2000000};          /* 2 ms */
        nanosleep(&ts, NULL);
    }
}

int main(void) {
    enum { NL = 2, NE = 4, D = 64, IH = 32, TOPK = 2 };
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);

    /* ---- 1. container int8: il tier deve accendersi e promuovere ---- */
    if (!qt_init(NL, NE, D, IH, NE, TOPK, 0 /* per-row */, 0 /* int8 */)) {
        printf("  FAIL: il tier rifiuta un container int8 con scale per riga\n");
        return 1;
    }
    check(G.wfmt == 1, "int8 deve usare fmt=1");
    /* charged at the allocator's granularity: three int8 matrices of D*IH
     * bytes each plus three per-row scale tables */
    check(G.exp_bytes == 3 * dev_alloc_footprint((size_t)D * IH)
                       + 3 * dev_alloc_footprint((2 * G.sc_gu + G.sc_d) / 3 * sizeof(float)),
          "il budget int8 deve contare un byte per elemento, non mezzo");

    int layers[8], eids[8];
    int planned = qt_plan_fill(layers, eids, 8);
    check(planned > 0, "qt_plan_fill non ha pianificato nessun esperto");

    /* pesi finti: valori riconoscibili, per vedere se arrivano intatti */
    size_t mb = (size_t)D * IH;
    unsigned char *g = malloc(mb), *u = malloc(mb), *d = malloc(mb);
    for (size_t i = 0; i < mb; i++) { g[i] = (unsigned char)(i & 0x7f); u[i] = 0x11; d[i] = 0x22; }
    float *sc = calloc(2 * G.sc_gu + G.sc_d, sizeof(float));

    qt_note_planned(layers[0], eids[0], g, u, d, sc, sc + G.sc_gu, sc + 2 * G.sc_gu);
    drain();

    /* Il cuore del test: con il difetto, qui fake_uploads restava 0. */
    check(fake_uploads > 0,
          "nessun esperto e' arrivato in VRAM: e' esattamente #1331, il tier "
          "riserva budget e non promuove niente");
    check(last_fmt == 1, "il backend deve ricevere fmt=1 per un container int8");
    check(last_bytes == mb, "int8: un byte per elemento, senza impacchettamento");
    /* i byte devono arrivare COME SONO: lo XOR 0x88 serve ai nibble int4 e su
     * byte interi sarebbe corruzione silenziosa */
    int intact = 1;
    for (size_t i = 0; i < captured_len; i++)
        if (captured[i] != (unsigned char)(i & 0x7f)) { intact = 0; break; }
    check(intact, "i pesi int8 sono stati alterati durante lo staging (XOR di troppo?)");

    qt_shutdown();

    /* ---- 2. int8 + scale raggruppate: il kernel non sa esprimerle ---- */
    fake_uploads = 0;
    check(qt_init(NL, NE, D, IH, NE, TOPK, 64 /* grouped */, 0 /* int8 */) == 0,
          "int8 con scale raggruppate va rifiutato PRIMA di riservare budget, "
          "non promosso con numeri sbagliati");

    /* ---- 3. int4 non deve cambiare comportamento ---- */
    if (!qt_init(NL, NE, D, IH, NE, TOPK, 64, 1 /* int4 */)) {
        printf("  FAIL: regressione, il tier non parte piu' su int4-gs64\n");
        return 1;
    }
    check(G.wfmt == 4, "int4 deve restare fmt=4");
    check(G.exp_bytes == 3 * dev_alloc_footprint((size_t)D * IH / 2)
                       + 3 * dev_alloc_footprint((2 * G.sc_gu + G.sc_d) / 3 * sizeof(float)),
          "int4 impacchettato deve contare mezzo byte per elemento");
    qt_shutdown();

    free(g); free(u); free(d); free(sc);
    if (fails) { printf("test_qwen36_tier_int8: %d fallimenti\n", fails); return 1; }
    printf("test_qwen36_tier_int8: ok\n");
    return 0;
}
