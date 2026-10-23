/* pin_pool.h — piu fotografie dello stato, non una sola.
 *
 * IL PROBLEMA. Una fotografia sola (SUBMIT pin=1) copre UN prefisso. Chi
 * punteggia un insieme chiuso ne ha due annidati: le istruzioni, condivise da
 * mille richieste, e istruzioni+domanda, condivise dalle alternative di quella
 * richiesta. Con una sola si e' costretti a scegliere: o si tengono le
 * istruzioni e ogni alternativa fa rileggere la domanda da capo, o si tiene la
 * domanda e le istruzioni si rileggono a ogni item.
 *
 * Misurato su qwen36 (22 GB), 4 item, istruzioni da 121 token, 3 opzioni:
 * con una fotografia sola jev macina 124 token per item invece di 46, cioe'
 * tre volte la domanda, e passa da 126,6 s a item contro i 60,1 della
 * generazione. Non e' un limite del metodo, e' questo pezzo che manca.
 *
 * LA FORMA. Un pugno di scatti; si rimette SEMPRE il piu' profondo che sia un
 * prefisso stretto del prompt nuovo. Nessuna chiave nuova sul filo: il client
 * manda pin=1 dove gli serve un punto di ritorno, e il motore li tiene tutti
 * finche' c'e' posto. kimi_k3.c fa gia' cosi' con i suoi checkpoint (di li'
 * viene l'idea di tenere il piu' profondo valido); qui la parte che non
 * dipende dal motore sta in un posto solo.
 *
 * COSA C'E' DENTRO E COSA NO. Qui stanno gli id, i logit finali e la
 * contabilita' degli slot. Lo stato ricorrente NO: e' diverso in ogni motore
 * (DeltaNet, SSM, KDA, convoluzioni corte) e alcuni non ne hanno affatto. Ogni
 * motore se lo attacca a `state` e se lo copia da se'.
 *
 * QUELLO CHE QUESTO FILE NON PUO' SAPERE. Che le righe K/V di quelle posizioni
 * siano ancora vive. Non entrano nella fotografia, restano dove sono, e i
 * banchi possono essere stati ributtati nel frattempo. Chi rimette uno scatto
 * DEVE chiederlo a kv_prefix_holds() prima di fidarsi. */
#ifndef COLI_PIN_POOL_H
#define COLI_PIN_POOL_H

#include <stdlib.h>
#include <string.h>

#ifndef COLI_PIN_SLOTS_MAX
#define COLI_PIN_SLOTS_MAX 8
#endif

typedef struct {
    int   *ids;      /* copia propria degli id che lo scatto copre */
    int    len;      /* quante posizioni */
    int    cap;      /* capienza di ids */
    float *logit;    /* logit dell'ultima posizione coperta: predice il primo token fresco */
    void  *state;    /* blob del motore (stato ricorrente), opaco qui */
    unsigned long long used;   /* marca LRU */
} ColiPin;

typedef struct {
    ColiPin slot[COLI_PIN_SLOTS_MAX];
    int n;                       /* slot in uso, 0 = spento */
    int vocab;
    unsigned long long clock;
} ColiPinPool;

/* Quanti scatti tenere. Default 4: bastano istruzioni + domanda con margine.
 * Su un motore con stato ricorrente grosso ogni scatto costa (qwen36: decine
 * di MB), quindi si puo' abbassare senza toccare il codice. */
static inline int coli_pin_slots_wanted(void) {
    const char *e = getenv("COLI_PIN_SLOTS");
    int n = e ? atoi(e) : 4;
    if (n < 0) n = 0;
    if (n > COLI_PIN_SLOTS_MAX) n = COLI_PIN_SLOTS_MAX;
    return n;
}

static inline void coli_pin_pool_init(ColiPinPool *p, int vocab) {
    if (!p || p->n) return;
    memset(p, 0, sizeof(*p));
    p->n = coli_pin_slots_wanted();
    p->vocab = vocab;
}

/* Lo scatto piu' profondo che sia un prefisso STRETTO di questo prompt.
 *
 * Stretto perche' serve almeno un token nuovo: un prompt uguale allo scatto
 * vorrebbe dire riavvolgere fino a se stesso e non lascerebbe niente da
 * macinare, quindi nessun logit da cui ripartire. Torna l'indice o -1. */
static inline int coli_pin_best(const ColiPinPool *p, const int *ids, int n) {
    if (!p || !ids || n < 1) return -1;
    int best = -1;
    for (int s = 0; s < p->n; s++) {
        const ColiPin *k = &p->slot[s];
        if (k->len < 1 || !k->ids || k->len >= n) continue;
        if (memcmp(k->ids, ids, (size_t)k->len * sizeof(int)) != 0) continue;
        if (best < 0 || k->len > p->slot[best].len) best = s;
    }
    return best;
}

/* Dove mettere lo scatto nuovo: uno slot libero, oppure lo stesso prefisso se
 * c'e' gia' (rinfrescarlo, non duplicarlo), oppure il meno usato di recente. */
static inline int coli_pin_victim(ColiPinPool *p, const int *ids, int n) {
    if (!p || p->n < 1) return -1;
    for (int s = 0; s < p->n; s++)
        if (p->slot[s].len == n && p->slot[s].ids &&
            memcmp(p->slot[s].ids, ids, (size_t)n * sizeof(int)) == 0) return s;
    for (int s = 0; s < p->n; s++) if (p->slot[s].len < 1) return s;
    int lru = 0;
    for (int s = 1; s < p->n; s++) if (p->slot[s].used < p->slot[lru].used) lru = s;
    return lru;
}

/* Scrive id e logit nello slot. Torna lo slot, o NULL se non c'e' memoria: la
 * fotografia e' un'ottimizzazione e non deve MAI essere il motivo di un
 * errore, quindi chi chiama tira dritto. `state` non viene toccato: se lo
 * slot ne portava uno, il chiamante lo riusa o lo libera lui. */
static inline ColiPin *coli_pin_store(ColiPinPool *p, const int *ids, int n,
                                      const float *logit) {
    if (!p || p->n < 1 || !ids || n < 1) return NULL;
    int s = coli_pin_victim(p, ids, n);
    if (s < 0) return NULL;
    ColiPin *k = &p->slot[s];
    if (k->cap < n) {
        int *grown = (int *)realloc(k->ids, (size_t)n * sizeof(int));
        if (!grown) { k->len = 0; return NULL; }
        k->ids = grown; k->cap = n;
    }
    if (logit && p->vocab > 0) {
        if (!k->logit) k->logit = (float *)malloc((size_t)p->vocab * sizeof(float));
        if (!k->logit) { k->len = 0; return NULL; }
        memcpy(k->logit, logit, (size_t)p->vocab * sizeof(float));
    }
    memcpy(k->ids, ids, (size_t)n * sizeof(int));
    k->len = n;
    k->used = ++p->clock;
    return k;
}

static inline void coli_pin_touch(ColiPinPool *p, int s) {
    if (p && s >= 0 && s < p->n) p->slot[s].used = ++p->clock;
}

/* Butta ogni scatto. Si chiama quando lo stato che descrivono non esiste piu'
 * (banchi K/V riallocati, sessione chiusa): uno scatto orfano rimesso a posto
 * risponderebbe da posizioni che non ci sono, e lo farebbe in silenzio. */
static inline void coli_pin_pool_clear(ColiPinPool *p, void (*free_state)(void *)) {
    if (!p) return;
    for (int s = 0; s < p->n; s++) {
        if (free_state && p->slot[s].state) free_state(p->slot[s].state);
        p->slot[s].state = NULL;
        p->slot[s].len = 0;
    }
}

#endif /* COLI_PIN_POOL_H */
