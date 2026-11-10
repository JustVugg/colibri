/* prefix_store.h -- the prompts a server has read, kept for any later request that starts with them.
 *
 * THE PROBLEM. An engine holds the state of the conversation in its slot: a request that continues
 * it reads only what is new (kv_prefix.h), a short request between two turns leaves it parked
 * (park.h), a retry resumes from where the last prompt ended (the prompt-end snapshot). An agent
 * works differently: every task is a new conversation that repeats the same system prompt and
 * tools, thousands of tokens, then a question of its own. On an engine whose layers carry a
 * recurrence (Qwen3.6's DeltaNet, a convolution, a sliding window's ring) the state the slot holds
 * cannot be wound back to where the new conversation leaves the old one, so the new one reads its
 * whole prompt again: 9,736 shared tokens of a 9,757-token prompt read in 2.9 s on an RTX PRO 4000
 * with the whole model on the card, minutes where the experts stream. With two slots the gateway
 * may also send a conversation back to the other slot, and its next turn reads everything.
 *
 * THE IDEA. Keep what was read, by content, for every request: a prompt's rows by position in
 * blocks of B tokens, each block named by its tokens and the block before it (a chain: a block is
 * the same block only after the same prefix), shared by every prompt that starts with those tokens;
 * and checkpoints, the engine's state at a position (the recurrence, the rings' rows, what is not
 * rows by position) with the rows past the last whole block. A request takes the deepest
 * checkpoint its prompt starts with (an engine with nothing but rows by position may also take
 * whole blocks alone), puts the rows and the state back and reads only what follows. The engine
 * decides where checkpoints go: where conversations part (the start of a message), and where a
 * turn ends.
 *
 * WHAT THIS FILE KNOWS AND WHAT IT DOES NOT. The tokens, the blocks' rows, the bookkeeping, the
 * budget. Not what the state is: the engine hands an opaque piece with its size and a function
 * that frees it, and puts it back itself. Not where the device keeps its copies: the engine brings
 * them to the host before a save and tells its devices after a load that the host's rows and state
 * are theirs again. A store holds one layout (the parts' row sizes and heads): a layout that
 * changes (another KV element size) empties it.
 *
 * MEMORY. A budget in bytes, and the RAM the system has free: an entry that would leave less than
 * COLI_PS_RAM_FLOOR_MB free goes only if older ones make the room. The least recently used goes
 * first, a checkpoint or a block nothing follows (a block other blocks or checkpoints follow stays
 * while they do). */
#ifndef COLI_PREFIX_STORE_H
#define COLI_PREFIX_STORE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef COLI_PS_RAM_FLOOR_MB
#define COLI_PS_RAM_FLOOR_MB 1536
#endif

/* Rows of a cache by position: row t of head h at base + h * head_stride + t * row, t < rows. */
typedef struct { void *base; size_t row; int rows, heads; size_t head_stride; } ColiPsPart;

typedef struct {
    int parent;                 /* the block before (-1: the prompt's first) */
    uint64_t key;               /* the chain's name: the parent's and these tokens' */
    int *tok;                   /* the block's B tokens */
    void *rows;                 /* each part's rows of the B positions, part after part, head after head */
    size_t bytes;
    int refs;                   /* blocks and checkpoints that follow it */
    uint64_t used;
    int live;
} ColiPsBlock;

typedef struct {
    int base;                   /* the last whole block before it (-1: none) */
    int depth;                  /* whole blocks before it */
    int tail;                   /* its tokens past them (0 .. B - 1) */
    int *ttok;                  /* those tokens */
    void *trows;                /* each part's rows of those positions */
    void *state; size_t state_bytes; void (*state_free)(void *);
    size_t bytes;
    uint64_t used;
    int live;
} ColiPsCkpt;

typedef struct {
    int on, B;
    uint64_t sig;               /* the layout: parts' row sizes and heads */
    size_t pos_bytes;           /* a position's rows over every part */
    ColiPsBlock *blk; int nblk, cblk;
    ColiPsCkpt *ck; int nck, cck;
    int *ht; int htcap, htused;   /* block key -> index + 1 (0 empty, -1 removed) */
    size_t bytes, budget;
    uint64_t tick;
    unsigned long long lookups, hits, hit_tokens, saves, blocks_new, evicted;
} ColiPrefixStore;

/* What a lookup found: P positions (0: nothing), the blocks on the way (depth of them) and the
 * checkpoint (-1: blocks alone). */
typedef struct { int P, depth, ck; int *path; } ColiPsHit;

static inline uint64_t coli_ps_mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 29;
    return h;
}
static inline uint64_t coli_ps_key(uint64_t parent_key, const int *tok, int n) {
    uint64_t h = coli_ps_mix(parent_key, (uint64_t)n);
    for (int i = 0; i < n; i++) h = coli_ps_mix(h, (uint64_t)(uint32_t)tok[i]);
    return h ? h : 1;
}
static inline uint64_t coli_ps_root_key(void) { return 0x51ed270b0f7a5e3dull; }

static inline void coli_ps_init(ColiPrefixStore *s, int B, size_t budget) {
    memset(s, 0, sizeof *s);
    s->B = B < 16 ? 16 : B;
    s->budget = budget;
    s->on = budget > 0;
}

static inline void coli_ps_ht_put(ColiPrefixStore *s, uint64_t key, int idx);
static inline void coli_ps_ht_rebuild(ColiPrefixStore *s, int cap) {
    free(s->ht);
    s->ht = (int *)calloc((size_t)cap, sizeof(int));
    s->htcap = s->ht ? cap : 0; s->htused = 0;
    for (int i = 0; s->ht && i < s->nblk; i++) if (s->blk[i].live) coli_ps_ht_put(s, s->blk[i].key, i);
}
static inline void coli_ps_ht_put(ColiPrefixStore *s, uint64_t key, int idx) {
    if (!s->ht || (s->htused + 1) * 2 > s->htcap) { coli_ps_ht_rebuild(s, s->htcap ? s->htcap * 2 : 256); if (!s->ht) return; }
    size_t m = (size_t)s->htcap - 1, i = (size_t)key & m;
    while (s->ht[i] > 0) i = (i + 1) & m;
    if (s->ht[i] == 0) s->htused++;
    s->ht[i] = idx + 1;
}
/* the live block named key after parent with these tokens; -1 none */
static inline int coli_ps_find(const ColiPrefixStore *s, int parent, uint64_t key, const int *tok) {
    if (!s->ht) return -1;
    size_t m = (size_t)s->htcap - 1, i = (size_t)key & m;
    for (size_t k = 0; k < (size_t)s->htcap && s->ht[i] != 0; k++, i = (i + 1) & m) {
        if (s->ht[i] < 0) continue;
        const ColiPsBlock *b = &s->blk[s->ht[i] - 1];
        if (b->live && b->key == key && b->parent == parent && !memcmp(b->tok, tok, (size_t)s->B * sizeof(int))) return s->ht[i] - 1;
    }
    return -1;
}
static inline void coli_ps_ht_del(ColiPrefixStore *s, uint64_t key, int idx) {
    if (!s->ht) return;
    size_t m = (size_t)s->htcap - 1, i = (size_t)key & m;
    for (size_t k = 0; k < (size_t)s->htcap && s->ht[i] != 0; k++, i = (i + 1) & m)
        if (s->ht[i] == idx + 1) { s->ht[i] = -1; return; }
}

static inline void coli_ps_block_drop(ColiPrefixStore *s, int i) {
    ColiPsBlock *b = &s->blk[i];
    if (!b->live) return;
    coli_ps_ht_del(s, b->key, i);
    if (b->parent >= 0) s->blk[b->parent].refs--;
    s->bytes -= b->bytes;
    free(b->tok); free(b->rows);
    memset(b, 0, sizeof *b);
}
static inline void coli_ps_ckpt_drop(ColiPrefixStore *s, int i) {
    ColiPsCkpt *c = &s->ck[i];
    if (!c->live) return;
    if (c->base >= 0) s->blk[c->base].refs--;
    s->bytes -= c->bytes;
    if (c->state && c->state_free) c->state_free(c->state);
    free(c->ttok); free(c->trows);
    memset(c, 0, sizeof *c);
}
static inline void coli_ps_clear(ColiPrefixStore *s) {
    for (int i = 0; i < s->nck; i++) coli_ps_ckpt_drop(s, i);
    for (int i = 0; i < s->nblk; i++) { ColiPsBlock *b = &s->blk[i]; if (b->live) { free(b->tok); free(b->rows); } }
    free(s->blk); free(s->ck); free(s->ht);
    s->blk = NULL; s->ck = NULL; s->ht = NULL;
    s->nblk = s->cblk = s->nck = s->cck = s->htcap = s->htused = 0;
    s->bytes = 0; s->sig = 0; s->pos_bytes = 0;
}

/* the layout of the parts; a store kept for another one is emptied first */
static inline void coli_ps_layout(ColiPrefixStore *s, const ColiPsPart *parts, int nparts) {
    uint64_t h = coli_ps_mix(coli_ps_root_key(), (uint64_t)nparts);
    size_t pb = 0;
    for (int p = 0; p < nparts; p++) {
        h = coli_ps_mix(h, (uint64_t)parts[p].row);
        h = coli_ps_mix(h, (uint64_t)parts[p].heads);
        pb += parts[p].row * (size_t)parts[p].heads;
    }
    if (s->sig && s->sig != h) coli_ps_clear(s);
    s->sig = h; s->pos_bytes = pb;
}

/* rows of positions [t0, t0 + n) of every part into buf (out) or from it (!out) */
static inline void coli_ps_rows(const ColiPsPart *parts, int nparts, int t0, int n, char *buf, int out) {
    for (int p = 0; p < nparts; p++) {
        const ColiPsPart *q = &parts[p];
        const size_t len = (size_t)n * q->row;
        for (int h = 0; h < q->heads; h++) {
            char *a = (char *)q->base + (size_t)h * q->head_stride + (size_t)t0 * q->row;
            if (out) memcpy(buf, a, len); else memcpy(a, buf, len);
            buf += len;
        }
    }
}
static inline int coli_ps_parts_hold(const ColiPsPart *parts, int nparts, int n) {
    for (int p = 0; p < nparts; p++) if (!parts[p].base || parts[p].rows < n) return 0;
    return 1;
}

/* The oldest entry nothing follows goes; 0 when there is none. */
static inline int coli_ps_evict_one(ColiPrefixStore *s) {
    int bi = -1, ci = -1; uint64_t bu = UINT64_MAX, cu = UINT64_MAX;
    for (int i = 0; i < s->nck; i++) if (s->ck[i].live && s->ck[i].used < cu) { cu = s->ck[i].used; ci = i; }
    for (int i = 0; i < s->nblk; i++) if (s->blk[i].live && !s->blk[i].refs && s->blk[i].used < bu) { bu = s->blk[i].used; bi = i; }
    if (ci < 0 && bi < 0) return 0;
    if (ci >= 0 && (bi < 0 || cu <= bu)) coli_ps_ckpt_drop(s, ci); else coli_ps_block_drop(s, bi);
    s->evicted++;
    return 1;
}
/* room for `need` more bytes: the budget, and the free RAM's floor (avail_mb < 0: not known) */
static inline int coli_ps_room(ColiPrefixStore *s, size_t need, double avail_mb) {
    const double floor_mb = COLI_PS_RAM_FLOOR_MB;
    for (;;) {
        int over = s->bytes + need > s->budget;
        if (!over && avail_mb >= 0 && avail_mb - (double)need / 1048576.0 < floor_mb) over = 2;
        if (!over) return 1;
        size_t before = s->bytes;
        if (!coli_ps_evict_one(s)) return 0;
        if (over == 2) avail_mb += (double)(before - s->bytes) / 1048576.0;
    }
}

static inline void coli_ps_hit_free(ColiPsHit *h) { free(h->path); memset(h, 0, sizeof *h); h->ck = -1; }

/* The deepest prefix of ids[0, n) the store holds, at most `most` positions (a request keeps one
 * token to read at least). with_state: a checkpoint is needed (the engine has state beyond rows);
 * else whole blocks alone count too. P in h (0: nothing). */
static inline int coli_ps_lookup(ColiPrefixStore *s, const int *ids, int n, int most, int with_state, ColiPsHit *h) {
    memset(h, 0, sizeof *h); h->ck = -1;
    if (!s->on || most < 1 || n < 1) return 0;
    if (most > n) most = n;
    s->lookups++;
    const int B = s->B, maxd = most / B;
    h->path = (int *)malloc((size_t)(maxd + 1) * sizeof(int));
    if (!h->path) return 0;
    uint64_t key = coli_ps_root_key(); int cur = -1, d = 0;
    while (d < maxd) {
        uint64_t k = coli_ps_key(key, ids + (size_t)d * B, B);
        int b = coli_ps_find(s, cur, k, ids + (size_t)d * B);
        if (b < 0) break;
        h->path[d++] = b; cur = b; key = k;
    }
    int bestP = with_state ? 0 : d * B, bestck = -1, bestd = with_state ? 0 : d;
    for (int i = 0; i < s->nck; i++) {
        const ColiPsCkpt *c = &s->ck[i];
        if (!c->live || c->depth > d) continue;
        if (c->depth > 0 ? h->path[c->depth - 1] != c->base : c->base != -1) continue;
        const int P = c->depth * B + c->tail;
        if (P > most || P <= bestP) continue;
        if (c->tail && memcmp(c->ttok, ids + (size_t)c->depth * B, (size_t)c->tail * sizeof(int))) continue;
        bestP = P; bestck = i; bestd = c->depth;
    }
    h->P = bestP; h->ck = bestck; h->depth = bestd;
    if (bestP > 0) {
        s->tick++;
        for (int k = 0; k < bestd; k++) s->blk[h->path[k]].used = s->tick;
        if (bestck >= 0) s->ck[bestck].used = s->tick;
    }
    return bestP;
}

/* The rows of a hit into the parts (positions [0, P)); its checkpoint's state, NULL for blocks
 * alone. The parts must hold P rows. */
static inline void *coli_ps_load(ColiPrefixStore *s, const ColiPsHit *h, const ColiPsPart *parts, int nparts) {
    for (int k = 0; k < h->depth; k++) coli_ps_rows(parts, nparts, k * s->B, s->B, (char *)s->blk[h->path[k]].rows, 0);
    s->hits++; s->hit_tokens += (unsigned long long)h->P;
    if (h->ck < 0) return NULL;
    const ColiPsCkpt *c = &s->ck[h->ck];
    if (c->tail) coli_ps_rows(parts, nparts, c->depth * s->B, c->tail, (char *)c->trows, 0);
    return c->state;
}

/* Keep ids[0, P) with the parts' rows of those positions and, when state is not NULL, a checkpoint
 * at P (the store owns state from here on, and frees it if it cannot keep it). Whole blocks the
 * store has already are not copied again; a checkpoint at the same prefix is replaced. 1 when kept;
 * avail_mb: the system's free RAM in MiB (< 0: not known). */
static inline int coli_ps_save(ColiPrefixStore *s, const int *ids, int P, const ColiPsPart *parts, int nparts,
                               void *state, size_t state_bytes, void (*state_free)(void *), double avail_mb) {
    if (!s->on || P < 1 || !coli_ps_parts_hold(parts, nparts, P)) { if (state && state_free) state_free(state); return 0; }
    coli_ps_layout(s, parts, nparts);
    const int B = s->B, nd = P / B, tail = P - nd * B;
    const size_t bb = (size_t)B * s->pos_bytes;
    uint64_t key = coli_ps_root_key(); int cur = -1, ok = 1;
    s->tick++;
    for (int d = 0; d < nd && ok; d++) {
        const int *tk = ids + (size_t)d * B;
        uint64_t k = coli_ps_key(key, tk, B);
        int b = coli_ps_find(s, cur, k, tk);
        if (b < 0) {
            if (cur >= 0) s->blk[cur].refs++;   /* held while the new block is made: the room may evict leaves */
            ok = coli_ps_room(s, bb + (size_t)B * sizeof(int), avail_mb);
            if (cur >= 0) s->blk[cur].refs--;
            if (!ok) break;
            for (b = 0; b < s->nblk && s->blk[b].live; b++) {}
            if (b == s->nblk) {
                if (s->nblk == s->cblk) {
                    int nc = s->cblk ? 2 * s->cblk : 64;
                    ColiPsBlock *g = (ColiPsBlock *)realloc(s->blk, (size_t)nc * sizeof *g);
                    if (!g) { ok = 0; break; }
                    memset(g + s->cblk, 0, (size_t)(nc - s->cblk) * sizeof *g);
                    s->blk = g; s->cblk = nc;
                }
                s->nblk++;
            }
            ColiPsBlock *x = &s->blk[b];
            x->tok = (int *)malloc((size_t)B * sizeof(int)); x->rows = malloc(bb);
            if (!x->tok || !x->rows) { free(x->tok); free(x->rows); memset(x, 0, sizeof *x); ok = 0; break; }
            memcpy(x->tok, tk, (size_t)B * sizeof(int));
            coli_ps_rows(parts, nparts, d * B, B, (char *)x->rows, 1);
            x->parent = cur; x->key = k; x->bytes = bb + (size_t)B * sizeof(int); x->live = 1;
            if (cur >= 0) s->blk[cur].refs++;
            s->bytes += x->bytes; s->blocks_new++;
            coli_ps_ht_put(s, k, b);
        }
        s->blk[b].used = s->tick;
        cur = b; key = k;
    }
    if (!ok || !state) { if (state && state_free) state_free(state); return ok && !state; }
    /* the checkpoint: one at the same prefix goes first */
    for (int i = 0; i < s->nck; i++) {
        ColiPsCkpt *c = &s->ck[i];
        if (c->live && c->base == cur && c->depth == nd && c->tail == tail &&
            (!tail || !memcmp(c->ttok, ids + (size_t)nd * B, (size_t)tail * sizeof(int)))) coli_ps_ckpt_drop(s, i);
    }
    const size_t tb = (size_t)tail * s->pos_bytes, need = tb + (size_t)tail * sizeof(int) + state_bytes;
    if (cur >= 0) s->blk[cur].refs++;
    ok = coli_ps_room(s, need, avail_mb);
    if (cur >= 0) s->blk[cur].refs--;
    if (!ok) { if (state_free) state_free(state); return 0; }
    int c; for (c = 0; c < s->nck && s->ck[c].live; c++) {}
    if (c == s->nck) {
        if (s->nck == s->cck) {
            int nc = s->cck ? 2 * s->cck : 32;
            ColiPsCkpt *g = (ColiPsCkpt *)realloc(s->ck, (size_t)nc * sizeof *g);
            if (!g) { if (state_free) state_free(state); return 0; }
            memset(g + s->cck, 0, (size_t)(nc - s->cck) * sizeof *g);
            s->ck = g; s->cck = nc;
        }
        s->nck++;
    }
    ColiPsCkpt *x = &s->ck[c];
    x->ttok = tail ? (int *)malloc((size_t)tail * sizeof(int)) : NULL;
    x->trows = tail ? malloc(tb) : NULL;
    if (tail && (!x->ttok || !x->trows)) { free(x->ttok); free(x->trows); memset(x, 0, sizeof *x); if (state_free) state_free(state); return 0; }
    if (tail) { memcpy(x->ttok, ids + (size_t)nd * B, (size_t)tail * sizeof(int)); coli_ps_rows(parts, nparts, nd * B, tail, (char *)x->trows, 1); }
    x->base = cur; x->depth = nd; x->tail = tail;
    x->state = state; x->state_bytes = state_bytes; x->state_free = state_free;
    x->bytes = need; x->used = s->tick; x->live = 1;
    if (cur >= 0) s->blk[cur].refs++;
    s->bytes += need; s->saves++;
    return 1;
}

static inline int coli_ps_live_blocks(const ColiPrefixStore *s) { int n = 0; for (int i = 0; i < s->nblk; i++) n += s->blk[i].live; return n; }
static inline int coli_ps_live_ckpts(const ColiPrefixStore *s) { int n = 0; for (int i = 0; i < s->nck; i++) n += s->ck[i].live; return n; }

#endif /* COLI_PREFIX_STORE_H */
