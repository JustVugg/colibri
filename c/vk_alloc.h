/* vk_alloc.h -- the offset sub-allocator behind backend_vulkan.c's weight memory.
 *
 * Why: a weight tensor is a VkBuffer bound at an offset inside a few big
 * VkDeviceMemory blocks (one memory object per tensor makes every submit pay for
 * thousands of referenced allocations, measured in backend_vulkan.c). The first
 * version handed those offsets out with a bump pointer and never took one back,
 * so a tier that evicts could not reuse a byte: MiMo's tier never evicts for that
 * reason. This keeps the blocks and gives the space back:
 *
 *   - every block keeps its free extents sorted by offset; a free coalesces with
 *     both neighbours, so a block whose tensors are all freed is one extent again;
 *   - an allocation takes the best fit over all blocks (the smallest extent that
 *     holds it after alignment), which for a tier of same-sized experts means an
 *     eviction's hole is exactly the next upload's place;
 *   - the pool has a byte limit (the budget): a new block that would cross it is
 *     refused, which is how a tier learns that it is full;
 *   - a block that becomes empty is handed back to the caller to release (one empty
 *     block is kept, so an evict/upload cycle at the boundary does not churn); its
 *     index is reused by the next block, and an index never moves while its block
 *     lives, so a range handed out stays valid until it is freed.
 *
 * Pure bookkeeping: no Vulkan, no locking, no allocation of device memory. The
 * caller owns the blocks' memory (vka_pool_add_block takes an opaque pointer) and
 * the lock. tests/test_vk_alloc.c drives it on the CPU. */
#ifndef COLI_VK_ALLOC_H
#define COLI_VK_ALLOC_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t off, len; } VkaExt;

typedef struct {
    uint64_t cap;              /* bytes in the block */
    uint64_t used;             /* bytes handed out, alignment padding included */
    int live;                  /* allocations not freed yet */
    VkaExt *fx; int nfx, cfx;  /* free extents, sorted by offset, never adjacent */
    void *user;                /* the caller's memory object */
    int present;               /* 0 = a released block's index, free for the next block */
} VkaBlock;

typedef struct {
    VkaBlock *b; int nb, cb;
    uint64_t block_bytes;      /* capacity of a new block (a larger request gets its own size) */
    uint64_t limit;            /* most bytes the blocks may hold together; 0 = no limit */
    uint64_t total;            /* sum of the blocks' capacities */
    uint64_t peak_used;
    uint64_t allocs, frees, refusals;   /* lifetime counters */
} VkaPool;

/* One allocation: which block, where, and how many bytes it holds there (the
 * caller hands all three back to vka_free). */
typedef struct { int block; uint64_t off, len; } VkaRange;

typedef struct {
    int blocks, live;
    uint64_t total, used, free, largest_free, peak_used;
    double frag;               /* 1 - largest free extent / free bytes: 0 = one hole */
} VkaStats;

static inline void vka_pool_init(VkaPool *p, uint64_t block_bytes, uint64_t limit) {
    memset(p, 0, sizeof(*p));
    p->block_bytes = block_bytes; p->limit = limit;
}

static inline int vka_ext_insert(VkaBlock *b, int at, VkaExt e) {
    if (b->nfx == b->cfx) {
        int nc = b->cfx ? 2 * b->cfx : 8;
        VkaExt *n = (VkaExt *)realloc(b->fx, (size_t)nc * sizeof(*n));
        if (!n) return 0;
        b->fx = n; b->cfx = nc;
    }
    memmove(b->fx + at + 1, b->fx + at, (size_t)(b->nfx - at) * sizeof(*b->fx));
    b->fx[at] = e; b->nfx++;
    return 1;
}
static inline void vka_ext_remove(VkaBlock *b, int at) {
    memmove(b->fx + at, b->fx + at + 1, (size_t)(b->nfx - at - 1) * sizeof(*b->fx));
    b->nfx--;
}

/* Would a block of `bytes` stay within the limit? */
static inline int vka_pool_can_grow(const VkaPool *p, uint64_t bytes) {
    return !p->limit || p->total + bytes <= p->limit;
}
/* The capacity a new block for a `size` request gets. */
static inline uint64_t vka_block_size_for(const VkaPool *p, uint64_t size) {
    uint64_t c = p->block_bytes;
    if (size > c) c = (size + 4095) & ~(uint64_t)4095;
    return c;
}

/* Register a block of `cap` bytes whose memory the caller created. Returns its
 * index, or -1 (limit, or out of host memory). A released block's index is reused
 * first; an index never moves while its block is present, so a range stays valid. */
static inline int vka_pool_add_block(VkaPool *p, uint64_t cap, void *user) {
    if (!cap || !vka_pool_can_grow(p, cap)) { p->refusals++; return -1; }
    int k = 0;
    while (k < p->nb && p->b[k].present) k++;
    if (k == p->nb && p->nb == p->cb) {
        int nc = p->cb ? 2 * p->cb : 8;
        VkaBlock *n = (VkaBlock *)realloc(p->b, (size_t)nc * sizeof(*n));
        if (!n) return -1;
        p->b = n; p->cb = nc;
    }
    VkaBlock *b = &p->b[k];
    memset(b, 0, sizeof(*b));
    b->cap = cap; b->user = user; b->present = 1;
    if (!vka_ext_insert(b, 0, (VkaExt){0, cap})) { free(b->fx); b->present = 0; return -1; }
    p->total += cap;
    if (k == p->nb) p->nb++;
    return k;
}

/* Best fit over every block. Returns 1 and fills *r, or 0: nothing fits, the
 * caller may add a block (vka_block_size_for / vka_pool_can_grow) and retry.
 * align must be a power of two. */
static inline int vka_alloc(VkaPool *p, uint64_t size, uint64_t align, VkaRange *r) {
    if (!size) size = 1;
    if (!align) align = 1;
    int bb = -1, bi = -1; uint64_t bwaste = UINT64_MAX, boff = 0;
    for (int k = 0; k < p->nb; k++) {
        VkaBlock *b = &p->b[k];
        if (!b->present || b->cap - b->used < size) continue;
        for (int i = 0; i < b->nfx; i++) {
            VkaExt e = b->fx[i];
            uint64_t a = (e.off + align - 1) & ~(align - 1);
            if (a < e.off || a - e.off > e.len || e.len - (a - e.off) < size) continue;
            uint64_t waste = e.len - size;            /* what the extent keeps around it */
            if (waste < bwaste) { bwaste = waste; bb = k; bi = i; boff = a; }
            if (!waste) break;
        }
        if (bwaste == 0) break;
    }
    if (bb < 0) return 0;
    VkaBlock *b = &p->b[bb];
    VkaExt e = b->fx[bi];
    uint64_t head = boff - e.off, tail = e.len - head - size;
    /* the head pad stays free; the tail stays free; the range is [boff, boff+size) */
    if (head && tail) {
        b->fx[bi].len = head;
        if (!vka_ext_insert(b, bi + 1, (VkaExt){boff + size, tail})) { b->fx[bi] = e; return 0; }
    } else if (head) {
        b->fx[bi].len = head;
    } else if (tail) {
        b->fx[bi].off = boff + size; b->fx[bi].len = tail;
    } else {
        vka_ext_remove(b, bi);
    }
    b->used += size; b->live++;
    r->block = bb; r->off = boff; r->len = size;
    p->allocs++;
    uint64_t u = 0; for (int k = 0; k < p->nb; k++) if (p->b[k].present) u += p->b[k].used;
    if (u > p->peak_used) p->peak_used = u;
    return 1;
}

/* Give a range back. Returns 1 when its block became empty and is not the only
 * empty block of the pool: the caller should release that block's memory and
 * then call vka_pool_drop_block. */
static inline int vka_free(VkaPool *p, VkaRange r) {
    if (r.block < 0 || r.block >= p->nb || !r.len || !p->b[r.block].present) return 0;
    VkaBlock *b = &p->b[r.block];
    int i = 0;
    while (i < b->nfx && b->fx[i].off < r.off) i++;
    int merge_prev = i > 0 && b->fx[i - 1].off + b->fx[i - 1].len == r.off;
    int merge_next = i < b->nfx && r.off + r.len == b->fx[i].off;
    if (merge_prev && merge_next) {
        b->fx[i - 1].len += r.len + b->fx[i].len;
        vka_ext_remove(b, i);
    } else if (merge_prev) {
        b->fx[i - 1].len += r.len;
    } else if (merge_next) {
        b->fx[i].off = r.off; b->fx[i].len += r.len;
    } else if (!vka_ext_insert(b, i, (VkaExt){r.off, r.len})) {
        return 0;   /* out of host memory: the range leaks, the books stay consistent */
    }
    b->used -= r.len; b->live--;
    p->frees++;
    if (b->live) return 0;
    int empties = 0;
    for (int k = 0; k < p->nb; k++) if (p->b[k].present && !p->b[k].live) empties++;
    return empties > 1;
}

/* Forget an empty block after the caller released its memory. Returns 0 when the
 * block is not present or still holds allocations. */
static inline int vka_pool_drop_block(VkaPool *p, int k) {
    if (k < 0 || k >= p->nb || !p->b[k].present || p->b[k].live) return 0;
    p->total -= p->b[k].cap;
    free(p->b[k].fx);
    memset(&p->b[k], 0, sizeof(p->b[k]));
    while (p->nb && !p->b[p->nb - 1].present) p->nb--;
    return 1;
}

static inline void vka_stats(const VkaPool *p, VkaStats *s) {
    memset(s, 0, sizeof(*s));
    s->total = p->total; s->peak_used = p->peak_used;
    for (int k = 0; k < p->nb; k++) {
        const VkaBlock *b = &p->b[k];
        if (!b->present) continue;
        s->blocks++; s->used += b->used; s->live += b->live;
        for (int i = 0; i < b->nfx; i++) {
            s->free += b->fx[i].len;
            if (b->fx[i].len > s->largest_free) s->largest_free = b->fx[i].len;
        }
    }
    s->frag = s->free ? 1.0 - (double)s->largest_free / (double)s->free : 0.0;
}

static inline void vka_pool_destroy(VkaPool *p) {
    for (int k = 0; k < p->nb; k++) if (p->b[k].present) free(p->b[k].fx);
    free(p->b);
    memset(p, 0, sizeof(*p));
}

#endif
