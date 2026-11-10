/* prefix_store.h: what a later request finds, and what it gets back.
 *
 * A synthetic engine: two caches by position (2 heads of 8-byte rows, 3 heads of 4-byte rows),
 * each row a function of every token up to its position, a compressed one (a row for 4 positions,
 * written once its group is read, as DeepSeek V4.1's compressed KV), and a "recurrence" that is the hash
 * of every token read. A conversation's prompt is read and kept with checkpoints at the starts
 * of its messages; a new conversation on the same system prompt takes the system prompt's
 * checkpoint, its rows the ones a fresh read writes and the state the fresh one's. Whole blocks
 * count alone only for an engine without state. A request keeps a token to read. The budget and
 * the free RAM's floor evict the oldest entries nothing follows, never a block a checkpoint
 * stands on; a layout that changes empties the store; a group that does not divide the block keeps
 * nothing. The checkpoints' places: the first message start past what the slot held, the last but
 * the reply's. */
#include <stdio.h>
#include <stdint.h>
#include "../prefix_store.h"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

enum { CAP = 4096, B = 16, PER = 4, NP = 3 };
static uint8_t c0[2][CAP][8], c1[3][CAP][4], c2[CAP / PER][8];
static ColiPsPart parts[NP];

static uint64_t prefix_hash(const int *ids, int n) { uint64_t h = 1469598103934665603ull; for (int i = 0; i < n; i++) h = coli_ps_mix(h, (uint64_t)ids[i]); return h; }
/* the engine reads ids[from, n): row t from the tokens [0, t] */
static void read_rows(const int *ids, int from, int n) {
    for (int t = from; t < n; t++) {
        uint64_t h = prefix_hash(ids, t + 1);
        for (int k = 0; k < 2; k++) for (int j = 0; j < 8; j++) c0[k][t][j] = (uint8_t)(h >> ((j + k) % 8 * 8));
        for (int k = 0; k < 3; k++) for (int j = 0; j < 4; j++) c1[k][t][j] = (uint8_t)(h >> ((j + 3 * k) % 8 * 8)) ^ 0x5a;
        if ((t + 1) % PER == 0) for (int j = 0; j < 8; j++) c2[t / PER][j] = (uint8_t)(h >> (j * 8)) ^ 0xa5;   /* a group closed */
    }
}
static int rows_are_fresh(const int *ids, int n) {
    static uint8_t a0[2][CAP][8], a1[3][CAP][4], a2[CAP / PER][8];
    memcpy(a0, c0, sizeof a0); memcpy(a1, c1, sizeof a1); memcpy(a2, c2, sizeof a2);
    read_rows(ids, 0, n);
    int same = 1;
    for (int k = 0; k < 2 && same; k++) same = !memcmp(a0[k], c0[k], (size_t)n * 8);
    for (int k = 0; k < 3 && same; k++) same = !memcmp(a1[k], c1[k], (size_t)n * 4);
    if (same) same = !memcmp(a2, c2, (size_t)(n / PER) * 8);
    memcpy(c0, a0, sizeof a0); memcpy(c1, a1, sizeof a1); memcpy(c2, a2, sizeof a2);
    return same;
}
static int frees;
static void state_free(void *p) { frees++; free(p); }
static void *state_of(const int *ids, int n) { uint64_t *s = malloc(sizeof *s); *s = prefix_hash(ids, n); return s; }

/* a prompt: the system prompt (sys tokens, the same for every conversation), then a message */
static int mk(int *ids, int sys, int msg, int seed) {
    int n = 0;
    for (int i = 0; i < sys; i++) ids[n++] = 1000 + (i * 7) % 97;
    ids[n++] = 1;   /* a message starts */
    for (int i = 0; i < msg; i++) ids[n++] = 2000 + (i * 13 + seed * 31) % 89;
    return n;
}

int main(void) {
    parts[0] = (ColiPsPart){c0, 8, CAP, 2, sizeof c0[0], 1};
    parts[1] = (ColiPsPart){c1, 4, CAP, 3, sizeof c1[0], 1};
    parts[2] = (ColiPsPart){c2, 8, CAP / PER, 1, sizeof c2, PER};
    ColiPrefixStore s;
    coli_ps_init(&s, B, (size_t)64 << 20);
    int A[600], Bq[600], C[600];
    const int sys = 200;
    int na = mk(A, sys, 150, 1), nb = mk(Bq, sys, 90, 2), nc = mk(C, sys, 40, 3);

    /* conversation A read in two pieces, a checkpoint at the start of its message and one at its end */
    read_rows(A, 0, sys);
    check(coli_ps_save(&s, A, sys, parts, NP, state_of(A, sys), 8, state_free, -1), "A's checkpoint at the system prompt's end kept");
    read_rows(A, sys, na);
    check(coli_ps_save(&s, A, na, parts, NP, state_of(A, na), 8, state_free, -1), "A's checkpoint at its end kept");
    check(coli_ps_live_blocks(&s) == na / B && coli_ps_live_ckpts(&s) == 2, "A's whole blocks once each, two checkpoints");

    /* B: a new conversation on the same system prompt */
    memset(c0, 0, sizeof c0); memset(c1, 0, sizeof c1); memset(c2, 0, sizeof c2);   /* the slot held something else since */
    ColiPsHit h;
    int P = coli_ps_lookup(&s, Bq, nb, nb - 1, 1, &h);
    check(P == sys && h.ck >= 0, "B finds the system prompt's checkpoint");
    void *st = coli_ps_load(&s, &h, parts, NP);
    check(st && *(uint64_t *)st == prefix_hash(Bq, sys), "B's state is the one a fresh read of the system prompt leaves");
    check(rows_are_fresh(Bq, sys), "B's rows of the system prompt are a fresh read's");
    coli_ps_hit_free(&h);
    read_rows(Bq, sys, nb);   /* B reads only its message */
    check(rows_are_fresh(Bq, nb), "B then reads only its message: every row a fresh read's");

    /* A again, its whole prompt: the end checkpoint, one token kept to read */
    P = coli_ps_lookup(&s, A, na, na - 1, 1, &h);
    check(P == sys, "A's prompt again: the end checkpoint is its whole prompt, past what may be taken; the system prompt's");
    coli_ps_hit_free(&h);
    int A2[700]; memcpy(A2, A, (size_t)na * sizeof(int)); int na2 = na; A2[na2++] = 1; for (int i = 0; i < 30; i++) A2[na2++] = 3000 + i;
    P = coli_ps_lookup(&s, A2, na2, na2 - 1, 1, &h);
    check(P == na, "A's next turn takes A's end checkpoint");
    memset(c0, 0, sizeof c0); memset(c1, 0, sizeof c1); memset(c2, 0, sizeof c2);
    st = coli_ps_load(&s, &h, parts, NP);
    check(st && *(uint64_t *)st == prefix_hash(A, na) && rows_are_fresh(A2, na), "A's rows and state back whole");
    coli_ps_hit_free(&h);

    /* a load that keeps what the parts hold already: the whole blocks below `keep` stay as they are */
    P = coli_ps_lookup(&s, A2, na2, na2 - 1, 1, &h);
    memset(c0, 0x77, sizeof c0); memset(c1, 0x77, sizeof c1); memset(c2, 0x77, sizeof c2);
    st = coli_ps_load_from(&s, &h, parts, NP, 2 * B + 5);
    {
        const int kept = coli_ps_kept(&s, &h, 2 * B + 5);
        int stay = kept == 2 * B && c0[0][0][0] == 0x77 && c0[1][kept - 1][7] == 0x77 && c1[2][kept - 1][3] == 0x77 && c2[0][0] == 0x77;
        read_rows(A2, 0, kept);   /* what the slot held there: the same prefix's rows */
        check(stay && st && rows_are_fresh(A2, na), "a load from a kept prefix: its blocks untouched, the rest back");
    }
    coli_ps_hit_free(&h);

    /* an engine without state takes whole blocks alone */
    P = coli_ps_lookup(&s, C, nc, nc - 1, 0, &h);
    check(P == sys && h.ck >= 0, "no state: C still takes the checkpoint where it reaches further than blocks");
    coli_ps_hit_free(&h);
    int D[600]; int nd = mk(D, sys - 5, 60, 4);   /* parts from A within the system prompt */
    P = coli_ps_lookup(&s, D, nd, nd - 1, 0, &h);
    check(P == (sys - 5) / B * B && h.ck < 0, "no state: D takes the whole blocks it shares");
    coli_ps_hit_free(&h);
    P = coli_ps_lookup(&s, D, nd, nd - 1, 1, &h);
    check(P == 0, "with state, blocks alone are not enough");
    coli_ps_hit_free(&h);

    /* the budget: room for a few blocks only; the oldest leaves go, never what a checkpoint stands on */
    ColiPrefixStore t;
    const size_t bb = (size_t)B * (2 * 8 + 3 * 4) + (size_t)(B / PER) * 8 + (size_t)B * sizeof(int);
    coli_ps_init(&t, B, bb * 20);
    read_rows(A, 0, na);
    check(coli_ps_save(&t, A, sys, parts, NP, state_of(A, sys), 8, state_free, -1), "small store: A's system checkpoint");
    int kept = coli_ps_live_blocks(&t);
    int E[900]; int ne = 0; for (int i = 0; i < 640; i++) E[ne++] = 5000 + i;   /* another prompt, unrelated */
    read_rows(E, 0, ne);
    int fr = frees;
    coli_ps_save(&t, E, ne, parts, NP, state_of(E, ne), 8, state_free, -1);
    check(t.bytes <= t.budget, "within the budget");
    check(frees > fr || coli_ps_live_ckpts(&t) <= 2, "what could not stay was freed");
    /* whatever is left is consistent: every live checkpoint's blocks are live and chained */
    int ok = 1;
    for (int i = 0; i < t.nck; i++) {
        ColiPsCkpt *c = &t.ck[i];
        if (!c->live) continue;
        int b = c->base, d = c->depth;
        while (b >= 0) { if (!t.blk[b].live) ok = 0; b = t.blk[b].parent; d--; }
        if (d != 0) ok = 0;
    }
    for (int i = 0; i < t.nblk; i++) if (t.blk[i].live && t.blk[i].parent >= 0 && !t.blk[t.blk[i].parent].live) ok = 0;
    check(ok, "no live checkpoint or block stands on one evicted");
    (void)kept;
    /* the free RAM's floor: with no RAM to spare nothing new stays */
    ColiPrefixStore u;
    coli_ps_init(&u, B, (size_t)64 << 20);
    fr = frees;
    check(!coli_ps_save(&u, A, sys, parts, NP, state_of(A, sys), 8, state_free, 100.0) && frees == fr + 1 && u.bytes == 0,
          "the free RAM below the floor: nothing kept, the state freed");
    check(coli_ps_save(&u, A, sys, parts, NP, state_of(A, sys), 8, state_free, 64000.0), "with RAM to spare it is kept");

    /* a layout that changes empties the store */
    ColiPsPart other[1] = {{c0, 4, CAP, 2, sizeof c0[0], 1}};
    coli_ps_save(&u, A, B, other, 1, NULL, 0, NULL, -1);
    check(coli_ps_live_ckpts(&u) == 0 && coli_ps_live_blocks(&u) == 1, "another layout: the old entries gone, the new block kept");

    /* the same prefix saved again replaces its checkpoint */
    ColiPrefixStore v;
    coli_ps_init(&v, B, (size_t)64 << 20);
    read_rows(A, 0, sys);
    coli_ps_save(&v, A, sys, parts, NP, state_of(A, sys), 8, state_free, -1);
    fr = frees;
    coli_ps_save(&v, A, sys, parts, NP, state_of(A, sys), 8, state_free, -1);
    check(coli_ps_live_ckpts(&v) == 1 && frees == fr + 1, "the same prefix again: one checkpoint, the old state freed");

    /* a compressed part whose group does not divide the block: nothing kept, nothing taken */
    ColiPrefixStore w;
    coli_ps_init(&w, B, (size_t)64 << 20);
    ColiPsPart odd[2] = {parts[0], {c2, 8, CAP / 5, 1, sizeof c2, 5}};
    fr = frees;
    check(!coli_ps_save(&w, A, sys, odd, 2, state_of(A, sys), 8, state_free, -1) && frees == fr + 1 && w.bytes == 0,
          "a group of 5 in blocks of 16: nothing kept, the state freed");
    check(!coli_ps_layout(&w, odd, 2) && coli_ps_layout(&w, parts, NP), "the layout refuses it, takes the block's divisors");

    /* a checkpoint whose tail ends inside a group: the group's rows are the whole groups before it */
    ColiPrefixStore x;
    coli_ps_init(&x, B, (size_t)64 << 20);
    read_rows(A, 0, na);
    const int P7 = sys + 7;   /* 207: 12 blocks, a tail of 15, 3 whole groups of it */
    check(coli_ps_save(&x, A, P7, parts, NP, state_of(A, P7), 8, state_free, -1), "a checkpoint inside a group kept");
    memset(c0, 0, sizeof c0); memset(c1, 0, sizeof c1); memset(c2, 0, sizeof c2);
    int A3[700]; memcpy(A3, A, (size_t)na * sizeof(int));
    P = coli_ps_lookup(&x, A3, na, na - 1, 1, &h);
    st = coli_ps_load(&x, &h, parts, NP);
    check(P == P7 && st && rows_are_fresh(A3, P7 / PER * PER) && !c2[P7 / PER][0], "its rows back up to its last whole group, the filling one not");
    coli_ps_hit_free(&h);

    /* where checkpoints go: the first message start a block past what the slot held, and the last
     * one but the reply's own */
    int M[400]; int nm = 0, opens[2] = {1, 7};
    for (int i = 0; i < 40; i++) M[nm++] = 100 + i;                       /* the system prompt */
    M[nm++] = 1; for (int i = 0; i < 30; i++) M[nm++] = 200 + i;          /* a user message at 40 */
    M[nm++] = 7; for (int i = 0; i < 30; i++) M[nm++] = 300 + i;          /* the answer at 71 */
    M[nm++] = 1; for (int i = 0; i < 30; i++) M[nm++] = 400 + i;          /* the next message at 102 */
    M[nm++] = 7; M[nm++] = 500;                                           /* the reply's opening at 133 */
    int mk_[4], nk = coli_ps_marks(M, nm, 0, B, opens, 2, mk_);
    check(nk == 2 && mk_[0] == 40 && mk_[1] == 102, "marks: the first message start and the last one, not the reply's opening");
    nk = coli_ps_marks(M, nm, 80, B, opens, 2, mk_);
    check(nk == 1 && mk_[0] == 102, "marks from 80: the next start a block on, once");
    nk = coli_ps_marks(M, nm, 90, B, opens, 2, mk_);
    check(nk == 0, "marks from 90: the next start is less than a block on, none");
    nk = coli_ps_marks(M, nm, 0, B, opens, 0, mk_);
    check(nk == 0, "marks: no token opens a message, none");
    /* a short last question (a few tokens before the reply's opening) still marks where it starts */
    int Y[200]; int ny = 0;
    for (int i = 0; i < 40; i++) Y[ny++] = 100 + i;
    Y[ny++] = 1;                                  /* the question at 40, five tokens */
    for (int i = 0; i < 5; i++) Y[ny++] = 200 + i;
    Y[ny++] = 7; Y[ny++] = 500; Y[ny++] = 501;    /* the reply's opening at 46 */
    nk = coli_ps_marks(Y, ny, 0, B, opens, 2, mk_);
    check(nk == 1 && mk_[0] == 40, "marks: a short last question's start, not the reply's opening");

    /* a message's end marks the next one's start (XTML: the token after <|end_of_msg|>) */
    int X[400]; int nx = 0, eom = 9;
    for (int i = 0; i < 40; i++) X[nx++] = 100 + i;   /* the system message, ended at 40 */
    X[nx++] = eom;
    for (int i = 0; i < 30; i++) X[nx++] = 200 + i;   /* a user message from 41, ended at 71 */
    X[nx++] = eom;
    for (int i = 0; i < 30; i++) X[nx++] = 300 + i;   /* the next one from 72, ended at 102 */
    X[nx++] = eom;
    X[nx++] = 400; X[nx++] = 401;                                         /* the reply's opening from 103 */
    nk = coli_ps_marks_ex(X, nx, 0, B, &eom, 1, 1, mk_);
    check(nk == 2 && mk_[0] == 41 && mk_[1] == 72, "marks after the token that closes a message");

    coli_ps_clear(&s); coli_ps_clear(&t); coli_ps_clear(&u); coli_ps_clear(&v); coli_ps_clear(&w); coli_ps_clear(&x);
    if (fails) { printf("test_prefix_store: %d failure(s)\n", fails); return 1; }
    printf("OK test_prefix_store: checkpoints at a message's start serve a new conversation on the same prompt, rows (a compressed cache's whole groups) and state a fresh read's; blocks alone without state; the budget and the RAM floor evict leaves only; checkpoints where messages start\n");
    return 0;
}
