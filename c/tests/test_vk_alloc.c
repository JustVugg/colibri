/* The Vulkan weight sub-allocator's bookkeeping (vk_alloc.h), on the CPU: no
 * device needed, so it runs in the ordinary test suite. The same core decides
 * every offset backend_vulkan.c hands a weight tensor; the device-side cases
 * (real VkDeviceMemory blocks, tensors freed and reused) are in the VK_TEST
 * harness. Checked here:
 *   - alloc/free/reuse: a freed range is the next same-sized allocation's place;
 *   - coalescing: freeing everything leaves one extent per block;
 *   - alignment: every offset is aligned, every range inside its block;
 *   - the budget: a block that would cross the limit is refused and counted;
 *   - empty blocks: the second empty block is handed back, its index reused;
 *   - stress: thousands of random alloc/free against a byte map that must never
 *     see two live ranges overlap, with the statistics checked at the end. */
#include <stdio.h>
#include <stdlib.h>
#include "../vk_alloc.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* grow-and-retry, as backend_vulkan.c does it */
static int get(VkaPool *p, uint64_t size, uint64_t align, VkaRange *r) {
    if (vka_alloc(p, size, align, r)) return 1;
    uint64_t cap = vka_block_size_for(p, size);
    if (vka_pool_add_block(p, cap, NULL) < 0) return 0;
    return vka_alloc(p, size, align, r);
}
static void put(VkaPool *p, VkaRange r) {
    if (vka_free(p, r)) {
        /* release one empty block that is not the last one kept */
        for (int k = 0; k < p->nb; k++)
            if (p->b[k].present && !p->b[k].live && k != r.block) { vka_pool_drop_block(p, k); break; }
    }
}

static void basic(void) {
    VkaPool p; vka_pool_init(&p, 1 << 20, 0);
    VkaRange a, b, c;
    CHECK(get(&p, 1000, 256, &a) && a.block == 0 && a.off == 0, "first range at 0");
    CHECK(get(&p, 1000, 256, &b) && b.off == 1024, "second range aligned to 256: %llu", (unsigned long long)b.off);
    CHECK(get(&p, 1000, 256, &c) && c.off == 2048, "third range: %llu", (unsigned long long)c.off);
    put(&p, b);
    VkaRange d;
    CHECK(get(&p, 1000, 256, &d) && d.block == b.block && d.off == b.off, "the freed hole is reused exactly");
    put(&p, a); put(&p, c); put(&p, d);
    VkaStats s; vka_stats(&p, &s);
    CHECK(s.live == 0 && s.used == 0 && p.b[0].nfx == 1 && p.b[0].fx[0].len == p.b[0].cap,
          "all freed: one extent (%d extents, used %llu)", p.b[0].nfx, (unsigned long long)s.used);
    CHECK(s.frag == 0.0, "no fragmentation when empty");
    vka_pool_destroy(&p);
}

static void budget(void) {
    VkaPool p; vka_pool_init(&p, 4096, 3 * 4096);
    VkaRange r[8]; int n = 0;
    while (n < 8 && get(&p, 4096, 64, &r[n])) n++;
    CHECK(n == 3, "a limit of three blocks holds three block-sized ranges, got %d", n);
    CHECK(p.refusals >= 1, "the fourth block was refused and counted");
    put(&p, r[1]);
    VkaRange x;
    CHECK(get(&p, 4096, 64, &x) && x.block == r[1].block, "space freed under the limit is usable again");
    /* a request larger than a block gets its own block, still under the limit */
    VkaPool q; vka_pool_init(&q, 4096, 16384);
    VkaRange big;
    CHECK(get(&q, 10000, 64, &big) && q.total == 12288, "large request: its own block (%llu)", (unsigned long long)q.total);
    VkaRange big2;
    CHECK(!get(&q, 10000, 64, &big2), "a second large block would cross the limit");
    vka_pool_destroy(&p); vka_pool_destroy(&q);
}

static void empty_blocks(void) {
    VkaPool p; vka_pool_init(&p, 4096, 0);
    VkaRange r[3];
    for (int i = 0; i < 3; i++) CHECK(get(&p, 4096, 1, &r[i]) && r[i].block == i, "block %d", i);
    CHECK(vka_free(&p, r[0]) == 0, "the first empty block is kept");
    CHECK(vka_free(&p, r[1]) == 1, "the second empty block is handed back");
    CHECK(vka_pool_drop_block(&p, 1), "drop the handed-back block");
    VkaStats s; vka_stats(&p, &s);
    CHECK(s.blocks == 2 && p.total == 8192, "two blocks left (%d, %llu)", s.blocks, (unsigned long long)p.total);
    CHECK(r[2].block == 2 && p.b[2].present && p.b[2].live == 1, "the live block kept its index");
    VkaRange n;
    CHECK(vka_alloc(&p, 4096, 1, &n) && n.block == 0, "the kept empty block serves the next range");
    VkaRange m;
    CHECK(get(&p, 4096, 1, &m) && m.block == 1, "a new block reuses the dropped index (%d)", m.block);
    vka_pool_destroy(&p);
}

static void stress(void) {
    enum { CAP = 1 << 16, N = 600 };
    VkaPool p; vka_pool_init(&p, CAP, 6 * CAP);
    static unsigned char owner[6][CAP];            /* 0 = free, else slot + 1 (mod 255) */
    VkaRange live[N]; int on[N] = {0};
    unsigned seed = 12345;
    int refused = 0;
    for (int it = 0; it < 40000; it++) {
        seed = seed * 1103515245u + 12345u;
        int i = (int)((seed >> 8) % N);
        if (on[i]) {
            for (uint64_t o = live[i].off; o < live[i].off + live[i].len; o++) owner[live[i].block][o] = 0;
            put(&p, live[i]); on[i] = 0;
            continue;
        }
        seed = seed * 1103515245u + 12345u;
        uint64_t size = 1 + (seed >> 8) % 3000, align = 1ull << ((seed >> 20) % 9);
        VkaRange r;
        if (!get(&p, size, align, &r)) { refused++; continue; }
        CHECK(r.off % align == 0, "alignment %llu at %llu", (unsigned long long)align, (unsigned long long)r.off);
        CHECK(r.block >= 0 && r.block < 6 && r.off + r.len <= p.b[r.block].cap, "inside the block");
        for (uint64_t o = r.off; o < r.off + r.len; o++) {
            if (owner[r.block][o]) { CHECK(0, "overlap at block %d offset %llu", r.block, (unsigned long long)o); break; }
            owner[r.block][o] = (unsigned char)(1 + i % 254);
        }
        live[i] = r; on[i] = 1;
    }
    VkaStats s; vka_stats(&p, &s);
    uint64_t used = 0; int nl = 0;
    for (int i = 0; i < N; i++) if (on[i]) { used += live[i].len; nl++; }
    CHECK(s.used == used && s.live == nl, "stats: used %llu vs %llu, live %d vs %d",
          (unsigned long long)s.used, (unsigned long long)used, s.live, nl);
    CHECK(s.used + s.free == s.total, "used + free = total");
    CHECK(p.total <= p.limit, "never above the limit");
    for (int i = 0; i < N; i++) if (on[i]) put(&p, live[i]);
    vka_stats(&p, &s);
    CHECK(s.used == 0 && s.live == 0 && s.blocks <= 1 + 1, "all freed: at most the kept empty blocks (%d)", s.blocks);
    for (int k = 0; k < p.nb; k++)
        if (p.b[k].present) CHECK(p.b[k].nfx == 1, "block %d coalesced to one extent (%d)", k, p.b[k].nfx);
    printf("stress: %llu allocs, %llu frees, %d refused by the limit, peak %llu of %llu bytes\n",
           (unsigned long long)p.allocs, (unsigned long long)p.frees, refused,
           (unsigned long long)p.peak_used, (unsigned long long)p.limit);
    vka_pool_destroy(&p);
}

int main(void) {
    basic();
    budget();
    empty_blocks();
    stress();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
