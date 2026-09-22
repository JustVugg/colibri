/* pin_pool.h: the nested-snapshot bookkeeping, without an engine.
 *
 * The pool is what makes closed-set scoring cheap: the shared instructions are
 * photographed once, the instructions-plus-question a second time, and each
 * option then costs only its own tokens. Everything that can go wrong here is
 * silent -- a snapshot restored at the wrong length answers from a state that
 * is not the one it claims, and the reply still reads plausibly. So the rules
 * are pinned down by a test that needs no model. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../pin_pool.h"
#include "../kv_prefix.h"

static void set_slots(int n) {
    static char buf[32];
    snprintf(buf, sizeof buf, "%d", n);
#ifdef _WIN32
    _putenv_s("COLI_PIN_SLOTS", buf);
#else
    setenv("COLI_PIN_SLOTS", buf, 1);
#endif
}

int main(void) {
    const int ids[8] = {10, 11, 12, 13, 14, 15, 16, 17};
    float logit[4] = {1.f, 2.f, 3.f, 4.f};

    /* --- the deepest snapshot wins ------------------------------------- */
    set_slots(4);
    ColiPinPool p; memset(&p, 0, sizeof p);
    coli_pin_pool_init(&p, 4);
    assert(p.n == 4);
    assert(coli_pin_store(&p, ids, 2, logit));
    assert(coli_pin_store(&p, ids, 5, logit));
    int best = coli_pin_best(&p, ids, 8);
    assert(best >= 0 && p.slot[best].len == 5);

    /* --- a strict prefix only: equal length leaves nothing to prefill --- */
    best = coli_pin_best(&p, ids, 5);
    assert(best >= 0 && p.slot[best].len == 2);

    /* --- a prompt that diverges falls back to the shallower snapshot ---- */
    const int other[8] = {10, 11, 99, 99, 99, 99, 99, 99};
    best = coli_pin_best(&p, other, 8);
    assert(best >= 0 && p.slot[best].len == 2);

    /* --- diverging at the first token matches nothing ------------------- */
    const int alien[4] = {99, 98, 97, 96};
    assert(coli_pin_best(&p, alien, 4) < 0);

    /* --- the logits travel with the snapshot ---------------------------- */
    best = coli_pin_best(&p, ids, 8);
    assert(p.slot[best].logit && p.slot[best].logit[3] == 4.f);

    /* --- storing the same prefix refreshes, it does not duplicate ------- */
    int before = 0;
    for (int s = 0; s < p.n; s++) if (p.slot[s].len > 0) before++;
    assert(coli_pin_store(&p, ids, 5, logit));
    int after = 0;
    for (int s = 0; s < p.n; s++) if (p.slot[s].len > 0) after++;
    assert(after == before);

    /* --- with one slot the newer snapshot replaces the older ------------ */
    ColiPinPool one; memset(&one, 0, sizeof one);
    set_slots(1);
    coli_pin_pool_init(&one, 4);
    assert(one.n == 1);
    assert(coli_pin_store(&one, ids, 2, logit));
    assert(coli_pin_store(&one, ids, 5, logit));
    best = coli_pin_best(&one, ids, 8);
    assert(best >= 0 && one.slot[best].len == 5);
    assert(coli_pin_best(&one, ids, 4) < 0);   /* the 2-long one is gone */

    /* --- zero slots: the pool is off and nothing is ever stored --------- */
    ColiPinPool off; memset(&off, 0, sizeof off);
    set_slots(0);
    coli_pin_pool_init(&off, 4);
    assert(off.n == 0);
    assert(coli_pin_store(&off, ids, 2, logit) == NULL);
    assert(coli_pin_best(&off, ids, 8) < 0);

    /* --- clearing drops every snapshot, and frees engine state ---------- */
    set_slots(4);
    ColiPinPool q; memset(&q, 0, sizeof q);
    coli_pin_pool_init(&q, 4);
    ColiPin *k = coli_pin_store(&q, ids, 3, logit);
    assert(k);
    k->state = malloc(16);                     /* what an engine would attach */
    assert(k->state);
    coli_pin_pool_clear(&q, free);
    assert(coli_pin_best(&q, ids, 8) < 0);
    assert(q.slot[0].state == NULL);

    /* --- kv_prefix_holds: the rows a snapshot needs must still be live -- */
    kv_prefix record; memset(&record, 0, sizeof record);
    assert(kv_prefix_alloc(&record, 8));
    kv_prefix_record(&record, ids, 0, 5);
    assert(kv_prefix_holds(&record, ids, 5));
    assert(kv_prefix_holds(&record, ids, 3));
    assert(!kv_prefix_holds(&record, ids, 6));   /* claims more than is held */
    assert(!kv_prefix_holds(&record, other, 5)); /* same length, other tokens */
    kv_prefix_clear(&record);
    assert(!kv_prefix_holds(&record, ids, 1));   /* banks dropped: nothing holds */
    kv_prefix_free(&record);

    puts("pin_pool: deepest match, strict prefix, refresh, eviction, liveness: ok");
    return 0;
}
