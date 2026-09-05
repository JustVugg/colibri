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
   include per conto suo; qui si definisce solo il corpo. ---- */
#include <stdint.h>
#include <time.h>

/* MinGW non ha setenv: il tier legge la sua configurazione dall'ambiente, e il
 * test deve poterla impostare anche su Windows. */
#if defined(_WIN32)
#include <stdlib.h>
static int test_setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
#define setenv test_setenv
#endif
#include "../backend_cuda.h"

struct ColiCudaTensor { int fmt, I, O, device, gs; const void *w; };

static int fake_uploads;
static int last_fmt = -1;
static size_t last_bytes;
static unsigned char captured[4096];
static size_t captured_len;

static int upload_common(ColiCudaTensor **t, const void *w, int fmt,
                         int I, int O, int device, int gs) {
    ColiCudaTensor *n = (ColiCudaTensor *)calloc(1, sizeof *n);
    n->fmt = fmt; n->I = I; n->O = O; n->device = device; n->gs = gs; n->w = w;
    *t = n;
    fake_uploads++;
    last_fmt = fmt;
    last_bytes = (size_t)I * O / (fmt == 1 ? 1 : 2);
    if (fake_uploads == 1) {
        captured_len = last_bytes < sizeof captured ? last_bytes : sizeof captured;
        memcpy(captured, w, captured_len);
    }
    return 1;
}
int coli_cuda_tensor_upload(ColiCudaTensor **t, const void *w, const float *s,
                            int fmt, int I, int O, int device) {
    (void)s; return upload_common(t, w, fmt, I, O, device, 0);
}
int coli_cuda_tensor_upload_g(ColiCudaTensor **t, const void *w, const float *s,
                              int fmt, int I, int O, int device, int gs) {
    (void)s; return upload_common(t, w, fmt, I, O, device, gs);
}
void coli_cuda_tensor_free(ColiCudaTensor *t) { free(t); }
int coli_cuda_available_device_count(void) { return 1; }
int coli_cuda_device_count(void) { return 1; }
int coli_cuda_init(const int *d, int n) { (void)d; (void)n; return 1; }
void coli_cuda_shutdown(void) {}
int coli_cuda_mem_info(int device, size_t *freeb, size_t *total) {
    (void)device;
    *freeb = 2ull << 30; *total = 4ull << 30;      /* 2 GiB liberi */
    return 1;
}
int coli_cuda_expert_group_issue(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                                 ColiCudaTensor *const *d, const int *rows,
                                 int count, const float *x) {
    (void)g; (void)u; (void)d; (void)rows; (void)count; (void)x; return 0;
}
const float *coli_cuda_expert_group_take(int device) { (void)device; return NULL; }
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d, double *kernel, double *d2h) {
    if (calls) *calls = 0; if (experts) *experts = 0; if (rows) *rows = 0;
    if (h2d) *h2d = 0; if (kernel) *kernel = 0; if (d2h) *d2h = 0;
}
void coli_cuda_stats(int device, size_t *count, size_t *bytes) {
    (void)device; if (count) *count = 0; if (bytes) *bytes = 0;
}

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
    check(G.exp_bytes > 3ull * D * IH,
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

    /* ---- 1b. il tier CONSERVA il puntatore: chi glielo passa non puo' liberarlo
       ----------------------------------------------------------------------
       #1341: la mia prima versione consegnava al tier i pesi int8 e la riga
       dopo li liberava. Il test guardava che i byte ARRIVASSERO, non che
       restassero validi, quindi non se n'e' accorto. Qui si pretende che il
       tier stia ancora puntando alla memoria viva del chiamante -- se la
       ritiene, il chiamante deve tenerla in vita, ed e' un contratto che va
       scritto invece che sperato. */
    {
        QSlot *slot = qs(layers[0], eids[0]);
        check(slot->g4 == g,
              "il tier deve ritenere il puntatore che gli e' stato dato: chi lo "
              "libera dopo averlo consegnato crea un use-after-free (#1341)");
        check(slot->u4 == u && slot->d4 == d,
              "anche up/down devono restare quelli consegnati");
    }

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
    check(G.exp_bytes < 3ull * D * IH + 4096 + (2 * G.sc_gu + G.sc_d) * sizeof(float),
          "int4 impacchettato deve contare mezzo byte per elemento");
    qt_shutdown();

    free(g); free(u); free(d); free(sc);
    if (fails) { printf("test_qwen36_tier_int8: %d fallimenti\n", fails); return 1; }
    printf("test_qwen36_tier_int8: ok\n");
    return 0;
}
