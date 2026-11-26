/* Qwen3.8 PLE rows read for a prefill chunk (q38_ple_prefetch, Q38_PLE_COALESCE): every
 * distinct row read once, adjacent rows of one part file in one read, never across two
 * parts; each (token, head) gets the same floats as its own one-row read
 * (Q38_PLE_COALESCE=0, the path of before) and as a decode of the file's bytes done here.
 * A fabricated model over a two-part table written to a temporary directory, F8_E4M3
 * with a weight scale and BF16; small head vocabularies so rows repeat and touch. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"

static int failures;
#define CHECK(x) do { if(!(x)){ fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x);failures++; } } while(0)

enum { DIM = 8, HEADS = 4, PART0 = 40, PART1 = 60, ROWS = PART0 + PART1 };
static uint8_t table[ROWS][DIM * 2];   /* the bytes as written: one byte (F8) or two (BF16) a value */

static void write_table(const char *dir, int bf16) {
    char path[1200], header[512];
    int esz = bf16 ? 2 : 1;
    uint32_t x = 2463534242u;
    for (int r = 0; r < ROWS; r++)
        for (int b = 0; b < DIM * esz; b++) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            uint8_t v = (uint8_t)x;
            if (!bf16 && (v & 0x7f) == 0x7f) v ^= 1;                    /* no e4m3 NaN */
            if (bf16 && (b & 1)) v = (uint8_t)(0x3c + (v & 3));        /* bf16 high byte: a modest float */
            table[r][b] = v;
        }
    const char *dt = bf16 ? "BF16" : "F8_E4M3";
    size_t b0 = (size_t)PART0 * DIM * esz, b1 = (size_t)PART1 * DIM * esz;
    int n = snprintf(header, sizeof header,
                     "{\"p0\":{\"dtype\":\"%s\",\"shape\":[%d,%d],\"data_offsets\":[0,%zu]},"
                     "\"p1\":{\"dtype\":\"%s\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}}",
                     dt, PART0, DIM, b0, dt, PART1, DIM, b0, b0 + b1);
    while (n % 8) header[n++] = ' ';
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb");
    uint64_t hl = (uint64_t)n;
    CHECK(f && fwrite(&hl, 8, 1, f) == 1 && fwrite(header, 1, (size_t)n, f) == (size_t)n);
    for (int r = 0; r < ROWS; r++) CHECK(fwrite(table[r], 1, (size_t)DIM * esz, f) == (size_t)DIM * esz);
    CHECK(fclose(f) == 0);
}

static float reference(int64_t row, int d, int bf16, float scale) {
    if (!bf16) return e4m3_decode(table[row][d]) * scale;
    uint32_t bits = (uint32_t)(table[row][2 * d] | table[row][2 * d + 1] << 8) << 16;
    float v; memcpy(&v, &bits, 4); return v;
}

static int cmp64(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b; return (x > y) - (x < y);
}

static void run(int bf16) {
    char dir[] = "test_qwen38_ple_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    write_table(dir, bf16);
    Model *m = (Model *)calloc(1, sizeof *m);
    st_init(&m->S, dir);
    Cfg *c = &m->c;
    c->ple_layer = 0; c->ngram_heads = HEADS; c->ngram_head_dim = DIM; c->heads_per_ngram = 2;
    c->eos_id = 1;
    m->ple_multipliers[0] = 3; m->ple_multipliers[1] = 7; m->ple_multipliers[2] = 11;
    static const int64_t vocab[HEADS] = {23, 29, 24, 24}, offset[HEADS] = {0, 23, 52, 76};
    for (int h = 0; h < HEADS; h++) { m->ple_head_vocab[h] = vocab[h]; m->ple_head_offset[h] = offset[h]; }
    m->ple_parts[0] = st_find(&m->S, "p0"); m->ple_parts[1] = st_find(&m->S, "p1");
    CHECK(m->ple_parts[0] && m->ple_parts[1]);
    snprintf(m->ple_part_names[0], sizeof m->ple_part_names[0], "p0");
    snprintf(m->ple_part_names[1], sizeof m->ple_part_names[1], "p1");
    m->ple_part_start[0] = 0; m->ple_part_start[1] = PART0; m->ple_part_start[2] = ROWS;
    m->ple_part_count = 2;
    m->ple_weight_scale = bf16 ? 1.f : 0.375f;
    m->ple_history = (int64_t *)calloc(2, sizeof(int64_t));
    m->ple_history[0] = 4; m->ple_history[1] = 9; m->ple_history_len = 2;

    enum { S = 40 };
    int ids[S];
    for (int s = 0; s < S; s++) ids[s] = (s * 7 + s / 5) % 13;   /* repeats, and an eos (1) or two */
    size_t floats = (size_t)S * HEADS * DIM;

    setenv("Q38_PLE_COALESCE", "0", 1);
    m->ple_reads = m->ple_rows_wanted = 0;
    q38_ple_prefetch(m, ids, S);
    CHECK(m->ple_pref && m->ple_pref_rows == S);
    float *before = (float *)malloc(floats * sizeof(float));
    memcpy(before, m->ple_pref, floats * sizeof(float));
    uint64_t reads_before = m->ple_reads;
    CHECK(reads_before == (uint64_t)S * HEADS && m->ple_rows_wanted == (uint64_t)S * HEADS);

    setenv("Q38_PLE_COALESCE", "1", 1);
    m->ple_reads = m->ple_rows_wanted = 0;
    q38_ple_prefetch(m, ids, S);
    CHECK(m->ple_pref && m->ple_pref_rows == S);
    CHECK(!memcmp(before, m->ple_pref, floats * sizeof(float)));   /* the bits of before */
    uint64_t reads_after = m->ple_reads;

    /* the rows as q38_ple_prefetch derives them, here from q38_hash_row directly */
    int64_t rows[S * HEADS], sorted[S * HEADS];
    int64_t hist[2] = {4, 9}; int hl = 2;
    for (int s = 0; s < S; s++) {
        int64_t p1 = hl >= 1 ? hist[hl - 1] : c->eos_id, p2 = hl >= 2 ? hist[hl - 2] : c->eos_id;
        for (int h = 0; h < HEADS; h++)
            rows[s * HEADS + h] = q38_hash_row(m, h, h < c->heads_per_ngram ? 2 : 3, ids[s], p1, p2);
        if (ids[s] == c->eos_id) hl = 0;
        else if (hl == 0) { hist[0] = ids[s]; hl = 1; }
        else if (hl == 1) { hist[1] = ids[s]; hl = 2; }
        else { hist[0] = hist[1]; hist[1] = ids[s]; }
    }
    int bad = 0;
    for (int i = 0; i < S * HEADS; i++)
        for (int d = 0; d < DIM; d++) {
            float want = reference(rows[i], d, bf16, m->ple_weight_scale);
            if (memcmp(&want, &m->ple_pref[(size_t)i * DIM + d], 4)) bad++;
        }
    CHECK(bad == 0);
    /* reads: one per run of consecutive distinct rows inside one part */
    memcpy(sorted, rows, sizeof rows);
    qsort(sorted, S * HEADS, sizeof(int64_t), cmp64);
    int distinct = 0, runs = 0;
    for (int i = 0; i < S * HEADS; i++) {
        if (i && sorted[i] == sorted[i - 1]) continue;
        int fresh = !distinct || sorted[i] != sorted[i - 1] + 1 || sorted[i] == PART0;
        runs += fresh; distinct++;
    }
    CHECK(reads_after == (uint64_t)runs && m->ple_rows_wanted == (uint64_t)S * HEADS);
    CHECK(distinct < S * HEADS && runs < distinct);   /* the fixture does repeat and touch */
    printf("  %s: %d (token, head) rows, %d distinct, reads %llu before, %llu after\n",
           bf16 ? "BF16" : "F8_E4M3 x scale", S * HEADS, distinct,
           (unsigned long long)reads_before, (unsigned long long)reads_after);

    /* a run stops at a part's end: rows 39 | 40 41 (39 twice), and 5 on its own -> 3 reads */
    int64_t pick[5] = {PART0 - 1, PART0, PART0 + 1, PART0 - 1, 5};
    float got[5 * DIM];
    m->ple_reads = 0;
    CHECK(q38_ple_coalesced(m, pick, 5, got) && m->ple_reads == 3);
    for (int i = 0; i < 5; i++)
        for (int d = 0; d < DIM; d++) {
            float want = reference(pick[i], d, bf16, m->ple_weight_scale);
            CHECK(!memcmp(&want, &got[i * DIM + d], 4));
        }

    /* the inline path (no prefetch) reads the same floats, one row at a time */
    float one[DIM];
    q38_ple_row(m, PART0 - 1, one);
    for (int d = 0; d < DIM; d++) {
        float want = reference(PART0 - 1, d, bf16, m->ple_weight_scale);
        CHECK(!memcmp(&want, &one[d], 4));
    }
    free(before); free(m->ple_pref); free(m->ple_history);
    st_destroy(&m->S);
    char path[1200]; snprintf(path, sizeof path, "%s/model.safetensors", dir);
    CHECK(remove(path) == 0); CHECK(rmdir(dir) == 0);
    free(m);
}

int main(void) {
    run(0);
    run(1);
    unsetenv("Q38_PLE_COALESCE");
    if (failures) { fprintf(stderr, "qwen38 PLE reads: %d failures\n", failures); return 1; }
    puts("qwen38 PLE reads: deduplicated and coalesced within a part, the bits of one read a row: ok");
    return 0;
}
