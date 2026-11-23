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
 * COLI_PS_RAM_FLOOR_MB free goes only if older ones make the room (the floor holds across a whole
 * save, every entry it adds counted). The least recently used goes first, a checkpoint or a block
 * nothing follows (a block other blocks or checkpoints follow stays while they do); between
 * checkpoints used at once the deeper. A chain's shallowest checkpoint (where its system prompt
 * ends: what every new conversation on it takes) is used whenever the chain is, so it goes last
 * while its chain is in use. A save whose chain and state the budget cannot hold is refused before
 * it evicts anything, and a save that fails takes back the blocks it added. */
#ifndef COLI_PREFIX_STORE_H
#define COLI_PREFIX_STORE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef COLI_PS_RAM_FLOOR_MB
#define COLI_PS_RAM_FLOOR_MB 1536
#endif

/* Rows of a cache by position: row t of head h at base + h * head_stride + t * row, t < rows. A
 * compressed cache keeps a row for `per` positions (0 or 1: a row a position): row j once the
 * positions [j * per, (j + 1) * per) are read, the group still filling being the engine's state;
 * the block's tokens must be a multiple of it. */
typedef struct { void *base; size_t row; int rows, heads; size_t head_stride; int per; } ColiPsPart;

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
    size_t blk_bytes;           /* a block's rows over every part */
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
    s->bytes = 0; s->sig = 0; s->blk_bytes = 0;
}

static inline int coli_ps_per(const ColiPsPart *q) { return q->per > 1 ? q->per : 1; }
/* the bytes of every part's rows for n positions from a block's start (a compressed part's whole
 * groups) */
static inline size_t coli_ps_span(const ColiPsPart *parts, int nparts, int n) {
    size_t b = 0;
    for (int p = 0; p < nparts; p++) b += (size_t)(n / coli_ps_per(&parts[p])) * parts[p].row * (size_t)parts[p].heads;
    return b;
}
/* What a checkpoint at P costs the store with its whole chain: every position's rows and token, and
 * the state. A budget under it can never keep that checkpoint. */
static inline size_t coli_ps_cost(const ColiPsPart *parts, int nparts, int P, size_t state_bytes) {
    return coli_ps_span(parts, nparts, P) + (size_t)P * sizeof(int) + state_bytes;
}
/* The layout of the parts; a store kept for another one is emptied first. 0 when a part's group
 * does not divide the block: then nothing is kept or taken. */
static inline int coli_ps_layout(ColiPrefixStore *s, const ColiPsPart *parts, int nparts) {
    uint64_t h = coli_ps_mix(coli_ps_root_key(), (uint64_t)nparts);
    for (int p = 0; p < nparts; p++) {
        if (s->B % coli_ps_per(&parts[p])) return 0;
        h = coli_ps_mix(h, (uint64_t)parts[p].row);
        h = coli_ps_mix(h, (uint64_t)parts[p].heads);
        h = coli_ps_mix(h, (uint64_t)coli_ps_per(&parts[p]));
    }
    if (s->sig && s->sig != h) coli_ps_clear(s);
    s->sig = h; s->blk_bytes = coli_ps_span(parts, nparts, s->B);
    return 1;
}

/* rows of positions [t0, t0 + n) of every part into buf (out) or from it (!out); t0 a block's
 * start, so a compressed part's rows are its groups [t0 / per, (t0 + n) / per) */
static inline void coli_ps_rows(const ColiPsPart *parts, int nparts, int t0, int n, char *buf, int out) {
    for (int p = 0; p < nparts; p++) {
        const ColiPsPart *q = &parts[p];
        const int per = coli_ps_per(q);
        const size_t len = (size_t)(n / per) * q->row;
        if (!len) continue;
        for (int h = 0; h < q->heads; h++) {
            char *a = (char *)q->base + (size_t)h * q->head_stride + (size_t)(t0 / per) * q->row;
            if (out) memcpy(buf, a, len); else memcpy(a, buf, len);
            buf += len;
        }
    }
}
static inline int coli_ps_parts_hold(const ColiPsPart *parts, int nparts, int n) {
    for (int p = 0; p < nparts; p++) if (!parts[p].base || parts[p].rows < n / coli_ps_per(&parts[p])) return 0;
    return 1;
}

/* The oldest entry nothing follows goes (of checkpoints used at once, the deeper); 0 when there is none. */
static inline int coli_ps_evict_one(ColiPrefixStore *s) {
    int bi = -1, ci = -1, cp = -1; uint64_t bu = UINT64_MAX, cu = UINT64_MAX;
    for (int i = 0; i < s->nck; i++) {
        const ColiPsCkpt *c = &s->ck[i];
        const int P = c->depth * s->B + c->tail;
        if (c->live && (c->used < cu || (c->used == cu && P > cp))) { cu = c->used; ci = i; cp = P; }
    }
    for (int i = 0; i < s->nblk; i++) if (s->blk[i].live && !s->blk[i].refs && s->blk[i].used < bu) { bu = s->blk[i].used; bi = i; }
    if (ci < 0 && bi < 0) return 0;
    if (ci >= 0 && (bi < 0 || cu <= bu)) coli_ps_ckpt_drop(s, ci); else coli_ps_block_drop(s, bi);
    s->evicted++;
    return 1;
}
/* room for `need` more bytes: the budget, and the free RAM's floor (*avail_mb < 0: not known), which
 * the bytes evicted raise and the `need` granted lowers, so it holds across a whole save */
static inline int coli_ps_room(ColiPrefixStore *s, size_t need, double *avail_mb) {
    const double floor_mb = COLI_PS_RAM_FLOOR_MB;
    for (;;) {
        int over = s->bytes + need > s->budget;
        if (!over && *avail_mb >= 0 && *avail_mb - (double)need / 1048576.0 < floor_mb) over = 2;
        if (!over) { if (*avail_mb >= 0) *avail_mb -= (double)need / 1048576.0; return 1; }
        size_t before = s->bytes;
        if (!coli_ps_evict_one(s)) return 0;
        if (*avail_mb >= 0) *avail_mb += (double)(before - s->bytes) / 1048576.0;
    }
}
/* The shallowest checkpoint at most P on the chain that ends at block cur (d blocks) with ids'
 * tokens, used now: where a system prompt ends, every conversation on it parts there, so it stays
 * while its chain is in use, past the deeper checkpoints of older turns. */
static inline void coli_ps_touch_root(ColiPrefixStore *s, int cur, int d, const int *ids, int P) {
    int root = -1, rootP = P + 1;
    for (int i = 0; i < s->nck; i++) {
        const ColiPsCkpt *c = &s->ck[i];
        const int cp = c->depth * s->B + c->tail;
        if (!c->live || c->depth > d || cp >= rootP) continue;
        int b = cur;
        for (int k = d; k > c->depth; k--) b = s->blk[b].parent;   /* the chain's block at depth c->depth - 1 */
        if (b != c->base || (c->tail && memcmp(c->ttok, ids + (size_t)c->depth * s->B, (size_t)c->tail * sizeof(int)))) continue;
        root = i; rootP = cp;
    }
    if (root >= 0) s->ck[root].used = s->tick;
}

/* Fewer positions than this past what the slot holds are not worth a load (the rows of every
 * position come back): 64, or a block when blocks are smaller. */
static inline int coli_ps_min_gain(const ColiPrefixStore *s) { return s->B < 64 ? s->B : 64; }

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
        coli_ps_touch_root(s, bestd > 0 ? h->path[bestd - 1] : -1, bestd, ids, bestP);
    }
    return bestP;
}

/* The rows of a hit into the parts (positions [0, P)), but the whole blocks below `keep`: the
 * engine says the parts hold those positions' rows already (the slot's record shares them with the
 * prompt). Its checkpoint's state, NULL for blocks alone. The parts must hold P rows. */
static inline void *coli_ps_load_from(ColiPrefixStore *s, const ColiPsHit *h, const ColiPsPart *parts, int nparts, int keep) {
    for (int k = keep > 0 ? keep / s->B : 0; k < h->depth; k++)
        coli_ps_rows(parts, nparts, k * s->B, s->B, (char *)s->blk[h->path[k]].rows, 0);
    s->hits++; s->hit_tokens += (unsigned long long)h->P;
    if (h->ck < 0) return NULL;
    const ColiPsCkpt *c = &s->ck[h->ck];
    if (c->tail) coli_ps_rows(parts, nparts, c->depth * s->B, c->tail, (char *)c->trows, 0);
    return c->state;
}
static inline void *coli_ps_load(ColiPrefixStore *s, const ColiPsHit *h, const ColiPsPart *parts, int nparts) {
    return coli_ps_load_from(s, h, parts, nparts, 0);
}
/* the positions a load from `keep` left in place: its whole blocks */
static inline int coli_ps_kept(const ColiPrefixStore *s, const ColiPsHit *h, int keep) {
    int k = keep > 0 ? keep / s->B : 0;
    return (k < h->depth ? k : h->depth) * s->B;
}

/* A save that failed takes back the n blocks it added, the last of the chain that ends at cur:
 * without their checkpoint they are only RAM. */
static inline void coli_ps_unwind(ColiPrefixStore *s, int cur, int n) {
    while (n-- > 0 && cur >= 0) { const int p = s->blk[cur].parent; coli_ps_block_drop(s, cur); cur = p; }
}

/* Keep ids[0, P) with the parts' rows of those positions and, when state is not NULL, a checkpoint
 * at P (the store owns state from here on, and frees it if it cannot keep it). Whole blocks the
 * store has already are not copied again; a checkpoint at the same prefix is replaced. 1 when kept;
 * avail_mb: the system's free RAM in MiB as the engine calls (< 0: not known); every entry the save
 * adds, the state included, is counted against it. A chain and state over the budget are refused
 * before anything is evicted; a checkpoint that cannot be kept takes the blocks its save added with
 * it. */
static inline int coli_ps_save(ColiPrefixStore *s, const int *ids, int P, const ColiPsPart *parts, int nparts,
                               void *state, size_t state_bytes, void (*state_free)(void *), double avail_mb) {
    if (!s->on || P < 1 || !coli_ps_parts_hold(parts, nparts, P) || !coli_ps_layout(s, parts, nparts) ||
        coli_ps_cost(parts, nparts, state ? P : P / s->B * s->B, state ? state_bytes : 0) > s->budget) {
        if (state && state_free) state_free(state);
        return 0;
    }
    const int B = s->B, nd = P / B, tail = P - nd * B;
    const size_t bb = s->blk_bytes;
    uint64_t key = coli_ps_root_key(); int cur = -1, ok = 1, d = 0, added = 0;
    s->tick++;
    for (; d < nd; d++) {   /* the blocks the store has */
        const int *tk = ids + (size_t)d * B;
        uint64_t k = coli_ps_key(key, tk, B);
        int b = coli_ps_find(s, cur, k, tk);
        if (b < 0) break;
        s->blk[b].used = s->tick;
        cur = b; key = k;
    }
    coli_ps_touch_root(s, cur, d, ids, P);   /* before anything is evicted for this save */
    for (; d < nd; d++) {   /* the rest, new: past a new block nothing is stored */
        const int *tk = ids + (size_t)d * B;
        uint64_t k = coli_ps_key(key, tk, B);
        if (cur >= 0) s->blk[cur].refs++;   /* held while the new block is made: the room may evict leaves */
        ok = coli_ps_room(s, bb + (size_t)B * sizeof(int), &avail_mb);
        if (cur >= 0) s->blk[cur].refs--;
        if (!ok) break;
        int b;
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
        x->parent = cur; x->key = k; x->bytes = bb + (size_t)B * sizeof(int); x->live = 1; x->used = s->tick;
        if (cur >= 0) s->blk[cur].refs++;
        s->bytes += x->bytes; s->blocks_new++; added++;
        coli_ps_ht_put(s, k, b);
        cur = b; key = k;
    }
    if (!ok || !state) {
        if (state) { coli_ps_unwind(s, cur, added); if (state_free) state_free(state); }
        return ok && !state;
    }
    /* the checkpoint: one at the same prefix goes first */
    for (int i = 0; i < s->nck; i++) {
        ColiPsCkpt *c = &s->ck[i];
        if (c->live && c->base == cur && c->depth == nd && c->tail == tail &&
            (!tail || !memcmp(c->ttok, ids + (size_t)nd * B, (size_t)tail * sizeof(int)))) coli_ps_ckpt_drop(s, i);
    }
    const size_t tb = coli_ps_span(parts, nparts, tail), need = tb + (size_t)tail * sizeof(int) + state_bytes;
    if (cur >= 0) s->blk[cur].refs++;
    ok = coli_ps_room(s, need, &avail_mb);
    if (cur >= 0) s->blk[cur].refs--;
    if (!ok) goto fail;
    int c; for (c = 0; c < s->nck && s->ck[c].live; c++) {}
    if (c == s->nck) {
        if (s->nck == s->cck) {
            int nc = s->cck ? 2 * s->cck : 32;
            ColiPsCkpt *g = (ColiPsCkpt *)realloc(s->ck, (size_t)nc * sizeof *g);
            if (!g) goto fail;
            memset(g + s->cck, 0, (size_t)(nc - s->cck) * sizeof *g);
            s->ck = g; s->cck = nc;
        }
        s->nck++;
    }
    ColiPsCkpt *x = &s->ck[c];
    x->ttok = tail ? (int *)malloc((size_t)tail * sizeof(int)) : NULL;
    x->trows = tail ? malloc(tb) : NULL;
    if (tail && (!x->ttok || !x->trows)) { free(x->ttok); free(x->trows); memset(x, 0, sizeof *x); goto fail; }
    if (tail) { memcpy(x->ttok, ids + (size_t)nd * B, (size_t)tail * sizeof(int)); coli_ps_rows(parts, nparts, nd * B, tail, (char *)x->trows, 1); }
    x->base = cur; x->depth = nd; x->tail = tail;
    x->state = state; x->state_bytes = state_bytes; x->state_free = state_free;
    x->bytes = need; x->used = s->tick; x->live = 1;
    if (cur >= 0) s->blk[cur].refs++;
    s->bytes += need; s->saves++;
    return 1;
fail:
    coli_ps_unwind(s, cur, added);
    if (state_free) state_free(state);
    return 0;
}

static inline int coli_ps_live_blocks(const ColiPrefixStore *s) { int n = 0; for (int i = 0; i < s->nblk; i++) n += s->blk[i].live; return n; }
static inline int coli_ps_live_ckpts(const ColiPrefixStore *s) { int n = 0; for (int i = 0; i < s->nck; i++) n += s->ck[i].live; return n; }

/* The store as every engine sets it up, at its first request: COLI_PREFIX_STORE=0 off,
 * COLI_PREFIX_STORE_MB its budget (unset: a quarter of the RAM free now, avail_mb, at most 4096
 * MiB), COLI_PREFIX_STORE_BLOCK the tokens a block (256). off: the engine keeps it off (its own
 * prefix reuse turned off). */
static inline void coli_ps_init_env(ColiPrefixStore *s, int off, double avail_mb) {
    const char *e = getenv("COLI_PREFIX_STORE"), *mb = getenv("COLI_PREFIX_STORE_MB"), *bk = getenv("COLI_PREFIX_STORE_BLOCK");
    size_t budget = 0;
    if (!(e && *e == '0') && !off) {
        double want = avail_mb > 0 ? avail_mb / 4 : 1024.0;
        if (want > 4096.0) want = 4096.0;
        budget = mb && *mb ? (size_t)atoll(mb) << 20 : (size_t)want << 20;
    }
    int B = bk && atoi(bk) >= 16 ? atoi(bk) : 256;
    coli_ps_init(s, B, budget);
    if (s->on)
        fprintf(stderr, "[PREFIX] store: up to %zu MiB of the prompts read, kept by content for any later request "
                        "(blocks of %d tokens; COLI_PREFIX_STORE=0 off)\n", budget >> 20, B);
}

/* Where a prompt read from `from` keeps checkpoints: where its messages start (a token of opens[]
 * opens one; coli_ps_marks_ex with shift 1: the position after a token of toks[], which closes the
 * message before), the reply's opening aside (the last start, a few tokens from the end: a short
 * question just before it is still a start), the first such past `from` (after a new conversation's system prompt, where the next
 * one on the same prompt parts from it) and the last but the reply's (where the last message
 * starts: a retry, an edited or a returning turn parts there), each at least a block on. */
static inline int coli_ps_marks_ex(const int *ids, int np, int from, int B, const int *toks, int ntok, int shift, int *marks) {
    int first = -1, last = -1, prev = -1, top = -1;
    for (int p = from + 1; p < np; p++) {   /* the prompt's last message start */
        int open = 0;
        for (int k = 0; k < ntok && !open && p - shift >= 0; k++) open = ids[p - shift] == toks[k];
        if (open) top = p;
    }
    for (int p = from + 1; p < np; p++) {
        int open = 0;
        for (int k = 0; k < ntok && !open && p - shift >= 0; k++) open = ids[p - shift] == toks[k];
        if (!open) continue;
        if (p == top && np - p <= 16) break;   /* the reply's own opening: nothing past it to keep */
        if (first < 0 && p - from >= B) first = p;
        prev = p;
    }
    last = prev;
    int n = 0;
    if (first > 0) marks[n++] = first;
    if (last > 0 && last - (first > 0 ? first : from) >= B) marks[n++] = last;
    return n;
}
static inline int coli_ps_marks(const int *ids, int np, int from, int B, const int *opens, int nopen, int *marks) {
    return coli_ps_marks_ex(ids, np, from, B, opens, nopen, 0, marks);
}

static inline void coli_ps_report(const ColiPrefixStore *s) {
    if (!s->on || !s->lookups) return;
    fprintf(stderr, "[PREFIX] store: %llu lookups, %llu from the store (%llu prompt tokens not read again); %.1f MiB in %d blocks "
                    "and %d checkpoints, %llu evicted\n", s->lookups, s->hits, s->hit_tokens,
            s->bytes / 1048576.0, coli_ps_live_blocks(s), coli_ps_live_ckpts(s), s->evicted);
}

#endif /* COLI_PREFIX_STORE_H */
