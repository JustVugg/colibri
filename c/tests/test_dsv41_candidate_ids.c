/* The candidate source layer's selection, as the re-encoding it now ships: a row of block
 * ids (rows x candidate_topk int32) instead of a dense mask (rows x the reach bytes), which
 * is what made a 262144-token prompt allocate 64 GiB. The mask was the union of the chosen
 * blocks' ranges clipped at the reach, so reading the blocks out of it and painting them
 * back is the identity -- independent of how the blocks were chosen. (A row a chain's device
 * wrote is clipped at the *chunk's* reach, not the layer's, so its ids repaint the block
 * straddling that reach out past it: clipped_row below pins that geometry.) This gate holds
 * that identity, and holds it against a *frozen* copy of the pre-change writer: with the writer
 * itself kept (the selection must stay in one place) a harness that compares the live
 * writer with the live reconstruction compares the engine with itself and cannot see a
 * selection change at all (a flipped tie-break passes it). Hence ref_candidate_blocks below:
 * the 15f810f0 text, verbatim, as the oracle.
 *
 * The invariants, asserted here:
 *  P1 every one of the topk slots is written, -1 past the kept blocks (the ids are poisoned
 *     before the extraction, and a reader that trusts a stale slot masks differently);
 *  P2 the paint bound is min(compress_len, candidate_width), asserted where the reader
 *     computes it (candidate_paint_width) as well as where it is used: an id past the bound
 *     writes nothing, and [w, width) keeps its poison;
 *  P3 positions past the bound may differ and no reader looks at them (indexer_run sets
 *     score[j] = -INFINITY before the mask runs), so [0, w) is compared against the source's
 *     row and [w, width) against the poison -- an over-paint is a CHECK, not a heap abort;
 *  P5 the ids are block ids in the source's own coordinates.
 * The memory side (v41_candidate_bytes) is asserted at the end: no width enters it, and the
 * topk is the caller's, so an allocator and the copy of its array cannot disagree. */
#define main dsv41_main_unused
#include "../deepseek_v41.c"
#undef main

/* PRE-CHANGE ENGINE CODE, THE FROZEN ORACLE: candidate_blocks() as 15f810f0 has it.
 * Do not edit to match the live writer -- its whole purpose is to differ if that changes. */
static void ref_candidate_blocks(const Cfg *c, const float *score, int width, int lens, uint8_t *keep) {
    int block = c->candidate_block_size;
    int blocks = (width + block - 1) / block;
    float *best = xmalloc((size_t)blocks * sizeof(float), "block scores");
    for (int b = 0; b < blocks; b++) {
        float top = -INFINITY;
        for (int i = b * block; i < (b + 1) * block && i < width; i++)
            if (score[i] > top) top = score[i];
        best[b] = top;
    }
    int last = (lens - 1) / block;
    if (last >= 0 && last < blocks) best[last] = INFINITY;   /* the newest block is pinned in */
    int wanted = c->candidate_topk_blocks < blocks ? c->candidate_topk_blocks : blocks;
    memset(keep, 0, (size_t)width);
    for (int pick = 0; pick < wanted; pick++) {
        int chosen = -1;
        for (int b = 0; b < blocks; b++)
            if (best[b] > -INFINITY && (chosen < 0 || best[b] > best[chosen])) chosen = b;
        if (chosen < 0) break;
        for (int i = chosen * block; i < (chosen + 1) * block && i < width; i++) keep[i] = 1;
        best[chosen] = -INFINITY;
    }
    free(best);
}

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)
#define POISON 0x7f   /* 0x7f7f7f7f as an id: out of range, so an unwritten slot cannot pass */

/* Four score rows as indexer_run hands them over: coarse values (ties everywhere), all
 * zero, all equal, and the tail's -INFINITY past the row's reach `lens`. */
static void score_row(int which, int width, int lens, float *score) {
    for (int i = 0; i < width; i++) {
        if (which == 0) score[i] = (float)((i * 7 + 3) % 5);
        else if (which == 1) score[i] = 0.0f;
        else if (which == 2) score[i] = 1.0f;
        else score[i] = i < lens ? (float)((i * 5 + 1) % 3) : -INFINITY;
    }
}

static int pinned_tie_vector(void) {
    /* tests/test_dsv41_ref_topk.py:53 -- four blocks of two, all scores 0, the newest block
     * pinned: the ties go to the lowest blocks, so 1,1,0,0,0,0,1,1 */
    Cfg c; memset(&c, 0, sizeof c);
    c.candidate_block_size = 2; c.candidate_topk_blocks = 2;
    float score[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint8_t keep[8];
    ref_candidate_blocks(&c, score, 8, 8, keep);
    const char *want = "1,1,0,0,0,0,1,1";
    char got[32]; int at = 0;
    for (int i = 0; i < 8; i++) at += snprintf(got + at, sizeof got - (size_t)at, "%s%d", i ? "," : "", keep[i]);
    if (strcmp(got, want)) {
        fprintf(stderr, "pinned tie vector: %s, the reference pins %s\n", got, want);
        return 1;
    }
    uint8_t live[8];
    candidate_blocks(&c, score, 8, 8, live);
    CHECK(memcmp(live, keep, 8) == 0);
    return 0;
}

static int paint_width(void) {
    /* P2 where the reader computes it: a `w` this harness chose cannot see the reader's own
     * bound being widened, and a widened one memsets and paints past its compress_len-byte
     * scratch. */
    static const int table[][3] = {{0, 0, 0}, {0, 5, 0}, {5, 0, 0}, {5, 9, 5}, {9, 5, 5}, {9, 9, 9}, {7, 7, 7}};
    for (size_t i = 0; i < sizeof table / sizeof *table; i++)
        CHECK(candidate_paint_width(table[i][0], table[i][1]) == table[i][2]);
    return 0;
}

static int clipped_row(void) {
    /* A row the chain converted: the device painted blocks 0 and 1 of a 16-position row and
     * the conversion clips it at the chunk's reach wc = 12 (deepseek_v41_chain.h:1454), so
     * block 1 is set on [8,12) and 0 on [12,16); the ids are extracted over wh = 16 and the
     * reader repaints at w = 16. This is the one geometry in which a painted block's *last*
     * in-scan position is 0: an extraction that asked "is the last position set" (`hit =`
     * for `hit |=`) would drop block 1 and unmask [8,12), which every reader consults. */
    enum { width = 16, block = 8, wc = 12, slots = 4 };
    uint8_t row[width], back[width];
    int32_t ids[slots];
    memset(row, 1, (size_t)wc);
    memset(row + wc, 0, (size_t)(width - wc));
    int kept = candidate_row_ids(row, block, width, slots, ids);
    CHECK(kept == 2 && ids[0] == 0 && ids[1] == 1 && ids[2] == -1 && ids[3] == -1);
    memset(back, 0xa5, sizeof back);
    candidate_keep_row(ids, slots, block, width, back);
    CHECK(memcmp(back, row, (size_t)wc) == 0);     /* [0, wc): the clipped row, exactly */
    for (int j = wc; j < width; j++)               /* [wc, width): block 1's tail, past the reach */
        CHECK(back[j] == 1);
    return 0;
}

static int candidate_bytes(void) {
    /* 262144 rows x 2048 slots x 4 B: the reach does not appear, so neither does n^2 */
    CHECK(v41_candidate_bytes(2048, 262144) == 2147483648u);
    CHECK(v41_candidate_bytes(2048, 2 * 1000) == 2 * v41_candidate_bytes(2048, 1000));   /* linear in rows */
    CHECK(v41_candidate_bytes(0, 7) == 7 * (size_t)sizeof(int32_t));   /* topk 0: one slot a row */
    CHECK(v41_candidate_bytes(3, 1) == 3 * sizeof(int32_t));           /* and nothing about a width */
    /* the topk is the caller's: a copy sized for `stored` slots is a copy of `stored` slots,
     * whatever the config says, so the park copy cannot out- or undersize the live array */
    CHECK(v41_candidate_bytes(2, 5) == 5 * 2 * sizeof(int32_t));
    CHECK(v41_candidate_bytes(2, 5) != v41_candidate_bytes(3, 5));
    return 0;
}

int main(void) {
    static const int widths[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 64, 100, 257, 1024};
    static const int blocksz[] = {1, 2, 4, 8, 16};
    static const int topks[] = {0, 1, 2, 3, 8, 64, 2048};
    long cases = 0, padding = 0;
    if (pinned_tie_vector()) return 1;
    if (paint_width()) return 1;
    if (clipped_row()) return 1;
    for (size_t wi = 0; wi < sizeof widths / sizeof *widths; wi++) {
        int width = widths[wi], nblocks = 0;
        float *score = malloc((size_t)width * sizeof(float));
        uint8_t *live = malloc((size_t)width), *ref = malloc((size_t)width), *back = malloc((size_t)width);
        if (!score || !live || !ref || !back) return 1;
        for (size_t bi = 0; bi < sizeof blocksz / sizeof *blocksz; bi++) {
            int block = blocksz[bi];
            nblocks = (width + block - 1) / block;
            for (size_t ti = 0; ti < sizeof topks / sizeof *topks; ti++) {
                Cfg c; memset(&c, 0, sizeof c);
                c.candidate_block_size = block; c.candidate_topk_blocks = topks[ti];
                int slots = topks[ti] > 0 ? topks[ti] : 1;   /* what v41_candidate_bytes sizes */
                int32_t *ids = malloc(v41_candidate_bytes(slots, 1));
                if (!ids) return 1;
                for (int li = 0; li < 6; li++) {
                    int lens = (int[]){0, 1, width / 2, width - 1, width, width + 3}[(size_t)li];
                    for (int which = 0; which < 4; which++) {
                        cases++;
                        score_row(which, width, lens, score);
                        memset(live, 0, (size_t)width); memset(ref, 0, (size_t)width);
                        candidate_blocks(&c, score, width, lens, live);
                        ref_candidate_blocks(&c, score, width, lens, ref);
                        /* 1. the live writer is the frozen writer: a tie-break or a pin
                         *    change shows up here and nowhere else in this file */
                        CHECK(memcmp(live, ref, (size_t)width) == 0);
                        /* 2. the ids are the frozen row's blocks, every slot written (P1) */
                        memset(ids, POISON, v41_candidate_bytes(slots, 1));
                        int kept = candidate_row_ids(ref, block, width, slots, ids);
                        CHECK(kept <= slots);
                        for (int k = 0; k < slots; k++) {
                            if (k < kept) CHECK(ids[k] >= 0 && ids[k] < nblocks);
                            else CHECK(ids[k] == -1);
                        }
                        /* 3. and they paint the frozen row back on [0, w) for every reach
                         *    the reader can have, w = min(compress_len, candidate_width) */
                        for (int cl = 0; cl <= width; cl++) {
                            int w = candidate_paint_width(cl, width);
                            memset(back, 0xa5, (size_t)width);
                            candidate_keep_row(ids, slots, block, w, back);
                            CHECK(memcmp(back, ref, (size_t)w) == 0);
                            for (int j = w; j < width; j++)   /* P2/P3: past `w` is untouched */
                                CHECK(back[j] == 0xa5);
                        }
                        /* the -1 padding is load-bearing: a stale slot holding the last
                         * block's id would mask a row differently, which is why the
                         * extraction is asserted to have written all of them */
                        if (kept < slots && nblocks > 0) {
                            ids[kept] = nblocks - 1;
                            memset(back, 0xa5, (size_t)width);
                            candidate_keep_row(ids, slots, block, width, back);
                            padding += memcmp(back, ref, (size_t)width) != 0;
                            ids[kept] = -1;
                        }
                    }
                }
                free(ids);
            }
        }
        free(score); free(live); free(ref); free(back);
    }
    if (candidate_bytes()) return 1;
    CHECK(cases == 11760);   /* 14 widths x 5 block sizes x 7 topks x 6 lens x 4 score rows */
    CHECK(padding > 0);      /* the stale-slot case is exercised, not merely counted */
    printf("dsv41 candidate ids: %ld cases, %ld padding-sensitive; writer == the frozen 15f810f0 "
           "writer, the ids repaint it on [0,w), bytes(262144 rows, topk 2048) = 2 GiB\n", cases, padding);
    return 0;
}
