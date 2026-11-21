/* Q38_PREFILL_GPU_PERCARD's placement helpers (qwen38_pg.h): the balance with a worker a card gives each card an equal
 * part of the cards' share, its experts with the most rows first, and the rest to the CPU; the workers' ranges cover the
 * grouped list a run of equal cards each, with the rows before them. */
#include <stdio.h>
#include <string.h>
#include "../qwen38_pg.h"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

int main(void) {
    int order[16];
    {   /* two cards; want 60 rows: 30 a card, biggest first */
        int home[8]   = { 0,  1,  0,  1, -1,  0,  1,  0 };
        int counts[8] = { 20, 25, 15, 10, 50, 12,  9,  1 };
        q38_pg_balance_percard(home, counts, 8, order, 60.0);
        /* card 0 by rows: 20 (u0), 15 (u2), 12 (u5), 1 (u7): 20 < 30 takes u2 -> 35 >= 30, the rest to the CPU */
        check(home[0] == 0 && home[2] == 0 && home[5] == -1 && home[7] == -1, "card 0: 20 + 15 rows, then its part is full");
        /* card 1 by rows: 25 (u1), 10 (u3), 9 (u6): 25 < 30 takes u3 -> 35, u6 to the CPU */
        check(home[1] == 1 && home[3] == 1 && home[6] == -1, "card 1: 25 + 10 rows, then its part is full");
        check(home[4] == -1, "an expert not resident stays the CPU's");
    }
    {   /* a card with fewer rows than its part keeps them all; the rest of its part goes to the other card */
        int home[5]   = { 0,  1,  0,  0,  1 };
        int counts[5] = { 30,  6, 20, 15,  4 };   /* card 0: 65 rows, card 1: 10 */
        q38_pg_balance_percard(home, counts, 5, order, 60.0);
        /* parts: card 1 min(10, 30) = 10, card 0 the remaining 50: 30 + 20 = 50 -> the 15-row expert to the CPU */
        check(home[1] == 1 && home[4] == 1, "the small card takes all of its 10 rows");
        check(home[0] == 0 && home[2] == 0 && home[3] == -1, "the other card takes the rest of the share (50 rows)");
        check(q38_pg_cards(home, 5) == 2, "two cards still hold experts");
    }
    {   /* one card present: it gets the whole share */
        int home[4]   = { 1, -1,  1,  1 };
        int counts[4] = { 5,  9,  4,  3 };
        q38_pg_balance_percard(home, counts, 4, order, 8.0);
        check(home[0] == 1 && home[2] == 1 && home[3] == -1, "one card: 5 + 4 rows of a share of 8");
    }
    {   /* a share of 0: everything to the CPU; a card past the limit: the CPU's */
        int home[3] = { 0, Q38_PG_MAX_CARDS, 1 }, counts[3] = { 3, 3, 3 };
        q38_pg_balance_percard(home, counts, 3, order, 0.0);
        check(home[0] == -1 && home[1] == -1 && home[2] == -1, "share 0 and an out-of-range card: all the CPU's");
    }
    {   /* ties keep the list's order (stable), as the single-worker balance */
        int home[3] = { 0, 0, 0 }, counts[3] = { 4, 4, 4 };
        q38_pg_balance_percard(home, counts, 3, order, 4.0);
        check(order[0] == 0 && order[1] == 1 && order[2] == 2, "equal rows: the list's order");
        check(home[0] == 0 && home[1] == -1 && home[2] == -1, "the first of equal experts fills the part");
    }
    {   /* ranges: a run of equal cards each, with the rows before it */
        int devs[6] = { 0, 0, 0, 1, 1, 3 }, counts[6] = { 3, 1, 2, 5, 1, 7 };
        int start[8], len[8]; size_t rb[8];
        int nr = q38_pg_ranges(devs, counts, 6, start, len, rb, 8);
        check(nr == 3, "three cards, three ranges");
        check(start[0] == 0 && len[0] == 3 && rb[0] == 0, "card 0: entries 0-2 from row 0");
        check(start[1] == 3 && len[1] == 2 && rb[1] == 6, "card 1: entries 3-4 from row 6");
        check(start[2] == 5 && len[2] == 1 && rb[2] == 12, "card 3: entry 5 from row 12");
        nr = q38_pg_ranges(devs, counts, 6, start, len, rb, 2);
        check(nr == 2, "at most max ranges: the rest get no worker");
        check(q38_pg_ranges(devs, counts, 0, start, len, rb, 8) == 0, "no entries, no range");
    }
    if (fails) { printf("test_qwen38_pg_percard: %d failure(s)\n", fails); return 1; }
    printf("test_qwen38_pg_percard: ok\n");
    return 0;
}
