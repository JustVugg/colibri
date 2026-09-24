/* Regression gate for wiring slot_ensure_int8() onto #1271's unpack_int4_to_int8
 * (the same branchless AVX2/NEON/scalar kernel load_expert_merged already uses
 * for the container's int4 read) instead of its own separate scalar
 * nibble-unpack loop. Integer nibble extraction + sign-extend is bit-exact by
 * construction (no reduction, no rounding), so this compares the NEW
 * slot_ensure_int8 (calling unpack_int4_to_int8) against a copy of the OLD
 * scalar loop it replaced -- not just "it compiles", a real regression check
 * that the two nibble-unpack formulas produce identical bytes. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

/* Verbatim copy of slot_ensure_int8's pre-#1271 scalar loop (the code this
 * change replaces), used here only as the regression oracle. */
static void old_unpack(int8_t *dst, const uint8_t *p, int64_t len) {
    for (int64_t i = 0; i < len; i += 2) {
        uint8_t b = p[i >> 1];
        int8_t lo = (int8_t)(b & 0xF); if (lo & 8) lo -= 16;
        int8_t hi = (int8_t)((b >> 4) & 0xF); if (hi & 8) hi -= 16;
        dst[i] = lo; dst[i + 1] = hi;
    }
}

static uint8_t *random_packed(int64_t nbytes, uint32_t seed) {
    uint8_t *p = malloc((size_t)nbytes);
    uint32_t x = seed ? seed : 1;
    for (int64_t i = 0; i < nbytes; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;   /* xorshift32 */
        p[i] = (uint8_t)x;
    }
    return p;
}

/* inter/hidden chosen so ng=inter*hidden and nd=hidden*inter land on both
 * sides of the AVX2 16-byte (32-element) vector boundary across the three
 * segments (g/u sized ng, d sized nd). ng/nd must stay even: nibble-packing
 * always yields ceil(n/2) bytes, and real containers guarantee even
 * inter*hidden (SIMD/GS64-block-aligned dims), so old_unpack -- a verbatim
 * copy of the pre-#1271 scalar loop, kept unmodified on purpose -- was never
 * written to handle an odd total (its last iteration writes dst[i+1]
 * unconditionally) and shouldn't be, either: that shape cannot occur. */
static void one_case(int inter, int hidden, const char *label) {
    Model m; memset(&m, 0, sizeof(m));
    m.c.inter = inter; m.c.hidden = hidden;
    int64_t ng = (int64_t)inter * hidden, nd = (int64_t)hidden * inter;
    CHECK(ng % 2 == 0, "%s: ng must be even (nibble-packed, see one_case)", label);
    if (ng % 2) return;

    Slot s; memset(&s, 0, sizeof(s));
    s.g4 = random_packed(ng / 2, 1001);
    s.u4 = random_packed(ng / 2, 2002);
    s.d4 = random_packed(nd / 2, 3003);

    slot_ensure_int8(&m, &s);
    CHECK(s.g != NULL, "%s: slot_ensure_int8 did not populate s->g", label);
    if (!s.g) return;

    int8_t *exp_g = malloc((size_t)ng), *exp_u = malloc((size_t)ng), *exp_d = malloc((size_t)nd);
    old_unpack(exp_g, s.g4, ng);
    old_unpack(exp_u, s.u4, ng);
    old_unpack(exp_d, s.d4, nd);

    CHECK(!memcmp(s.g, exp_g, (size_t)ng), "%s: gate segment differs from old scalar loop", label);
    CHECK(!memcmp(s.u, exp_u, (size_t)ng), "%s: up segment differs from old scalar loop", label);
    CHECK(!memcmp(s.d, exp_d, (size_t)nd), "%s: down segment differs from old scalar loop", label);
    printf("qwen36 slot_ensure_int8 exact: %s inter=%d hidden=%d (ng=%lld nd=%lld)\n",
           label, inter, hidden, (long long)ng, (long long)nd);

    free(exp_g); free(exp_u); free(exp_d);
    free(s.g); /* s.u/s.d point into the same block as s.g (slot_ensure_int8's `w`) */
    free(s.g4); free(s.u4); free(s.d4);
}

int main(void) {
    one_case(4, 8, "tiny, well under one vector");             /* ng=32, nd=32 */
    one_case(16, 16, "exact AVX2 vector boundary");             /* ng=256, nd=256, /16=16 (mult of 16-byte block) */
    one_case(17, 16, "vector body + scalar tail");              /* ng=272 -> nb=136, not mult of 16 */
    one_case(37, 42, "odd, real-expert-shaped");                /* ng=1554, nd=1554 (even: see one_case) */
    one_case(2048, 512, "large, real Qwen3.6 expert scale");    /* ng=1048576, nd=1048576 */

    /* re-entrancy guard: slot_ensure_int8 must no-op when s->g is already set
     * (the "rematerialize only if evicted" contract) -- unrelated to which
     * unpack kernel runs, but worth pinning since this function was touched */
    {
        Model m; memset(&m, 0, sizeof(m)); m.c.inter = 4; m.c.hidden = 8;
        Slot s; memset(&s, 0, sizeof(s));
        s.g4 = random_packed(16, 42); s.u4 = random_packed(16, 43); s.d4 = random_packed(16, 44);
        slot_ensure_int8(&m, &s);
        int8_t *g_before = s.g;
        slot_ensure_int8(&m, &s);   /* should be a no-op: s->g already non-NULL */
        CHECK(s.g == g_before, "slot_ensure_int8 re-ran when s->g was already set");
        free(s.g); free(s.g4); free(s.u4); free(s.d4);
    }

    if (failures) { fprintf(stderr, "qwen36 slot_ensure_int8: %d failure(s)\n", failures); return 1; }
    puts("qwen36 slot_ensure_int8: ok");
    return 0;
}
