/* V41_ENGRAM_STAGE: engram_stage_rows reads the rows of a block of positions that the
 * engram cache does not hold on reader threads, before the gather looks them up. The
 * gather must then return the same bits as when it reads each missing row itself, find
 * every staged row in the cache, and the stager must read each missing row once. With a
 * cache too small for a block it reads at most half the cache, and the gather reads the
 * rest itself, still the same bits.
 *
 * One table of 420 rows, written as a shard beside its sidecar: no model, no GPU. Each
 * row's bytes and scale differ from its neighbours', so a row put in the wrong slot or
 * read from the wrong offset changes the gathered floats. */
#define main dsv41_main_unused
#include "../deepseek_v41.c"
#undef main

static int g_fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("FAIL: %s\n", what); g_fails++; }
}

#define ROWS 420          /* offsets 0, 101, 204, 311 plus the last prime, 109 */
#define POSITIONS 200

static void write_fixture(const char *dir) {
    char path[600], hdr[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    const long long w = (long long)ROWS * 32, s = ROWS;
    int hl = snprintf(hdr, sizeof(hdr),
        "{\"layers.1.engram.embed.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[%d,32],"
        "\"data_offsets\":[0,%lld]},"
        "\"layers.1.engram.embed.scale\":{\"dtype\":\"U8\",\"shape\":[%d,1],"
        "\"data_offsets\":[%lld,%lld]}}", ROWS, w, ROWS, w, w + s);
    uint64_t hlen = (uint64_t)hl;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    for (int r = 0; r < ROWS; r++)
        for (int i = 0; i < 32; i++) fputc((r * 29 + i * 7) % 0x7e, f);   /* never 0x7f, e4m3's NaN */
    for (int r = 0; r < ROWS; r++) fputc(120 + r % 8, f);
    fclose(f);
    snprintf(path, sizeof(path), "%s/dsv41_engram.json", dir);
    f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fputs("{\"max_ngram_size\":3,\"n_heads\":2,\"head_dim\":32,\"pad_id\":0,\"layer_ids\":[1],"
          "\"primes\":[[[101,103],[107,109]]],\"offsets\":[[0,101,204,311]],"
          "\"multipliers\":[[\"3\",\"5\",\"7\"]],\"token_map\":[", f);
    for (int i = 0; i < 50; i++) fprintf(f, "%s%d", i ? "," : "", i);
    fputs("]}", f);
    fclose(f);
}

static int g_tokens[POSITIONS];

/* the sidecar, its one table with a cache of `cache_rows`, and the prompt's history */
static void open_engram(Engram *e, shards *S, const char *dir, int cache_rows) {
    memset(e, 0, sizeof(*e));
    engram_load_sidecar(e, dir);
    st_init(S, dir);
    engram_table_open(&e->table[0], S, e->layer_of[0], e->head_dim, cache_rows);
    engram_push(e, g_tokens, POSITIONS, NULL);
}

static void close_engram(Engram *e, shards *S) {
    free(e->table[0].key); free(e->table[0].used); free(e->table[0].value);
    free(e->token_map); free(e->history);
    st_destroy(S);
}

/* every position's rows gathered as engram_run gathers them, in blocks of `block`
 * positions, each staged first on `threads` readers (0: not staged); returns the calls */
static int gather(Engram *e, int block, int threads, float *out) {
    int64_t ids[V41_MAX_NGRAM * V41_MAX_EHEADS];
    const int width = e->cols * e->head_dim;
    int calls = 0;
    for (int p0 = 0; p0 < POSITIONS; p0 += block) {
        int n = POSITIONS - p0 < block ? POSITIONS - p0 : block;
        if (threads) { engram_stage_rows(e, 0, p0, n, threads); calls++; }
        for (int r = 0; r < n; r++) {
            engram_hash(e, 0, p0 + r, ids);
            for (int col = 0; col < e->cols; col++)
                memcpy(out + (size_t)(p0 + r) * width + (size_t)col * e->head_dim,
                       engram_row(&e->table[0], ids[col], e->head_dim),
                       (size_t)e->head_dim * sizeof(float));
        }
    }
    return calls;
}

int main(void) {
    char dir[] = "test_dsv41_engram_stage_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    write_fixture(dir);
    uint32_t x = 12345;
    for (int i = 0; i < POSITIONS; i++) { x = x * 1103515245u + 12345u; g_tokens[i] = (int)(x >> 16) % 50; }

    Engram e; shards S;
    open_engram(&e, &S, dir, 1024);
    const size_t floats = (size_t)POSITIONS * e.cols * e.head_dim;
    float *ref = malloc(floats * sizeof(float)), *got = malloc(floats * sizeof(float));
    if (!ref || !got) { printf("FAIL: out of memory\n"); return 1; }
    gather(&e, POSITIONS, 0, ref);
    /* a cache wider than the table evicts nothing: every miss is a distinct row */
    const uint64_t distinct = e.table[0].misses;
    check(distinct > 100 && e.table[0].hits > 0, "the prompt reads over 100 distinct rows and repeats some");
    close_engram(&e, &S);

    static const struct { int block, threads; const char *what; } cases[] = {
        {POSITIONS, 8, "the whole prompt on 8 readers"},
        {32, 3, "blocks of 32 positions (engram_run's chunk) on 3 readers"},
        {1, 8, "one position at a time (a decode step) on 8 readers"},
        {7, 1, "blocks of 7 on 1 reader"},
    };
    char label[200];
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        open_engram(&e, &S, dir, 1024);
        memset(got, 0xff, floats * sizeof(float));
        gather(&e, cases[c].block, cases[c].threads, got);
        snprintf(label, sizeof(label), "%s: the gather's bits", cases[c].what);
        check(!memcmp(got, ref, floats * sizeof(float)), label);
        snprintf(label, sizeof(label), "%s: the gather finds every row staged", cases[c].what);
        check(e.table[0].misses == 0, label);
        snprintf(label, sizeof(label), "%s: each missing row read once", cases[c].what);
        check(e.table[0].staged == distinct, label);
        close_engram(&e, &S);
    }

    /* a cache of 16 rows: at most 8 staged a block, the gather reads the rest itself */
    open_engram(&e, &S, dir, 16);
    memset(got, 0xff, floats * sizeof(float));
    int calls = gather(&e, 32, 4, got);
    check(!memcmp(got, ref, floats * sizeof(float)), "a 16-row cache: the gather's bits");
    check(e.table[0].staged > 0 && e.table[0].staged <= (uint64_t)calls * 8,
          "a 16-row cache: at most half the cache staged a block");
    check(e.table[0].misses > 0, "a 16-row cache: the gather reads the rows not staged");
    close_engram(&e, &S);

    /* the switch: engram_advise_rows stages on V41_ENGRAM_STAGE_THREADS readers */
    setenv("V41_ENGRAM_STAGE", "1", 1);
    setenv("V41_ENGRAM_STAGE_THREADS", "2", 1);
    check(engram_stage_threads() == 2, "V41_ENGRAM_STAGE=1 with 2 threads");
    open_engram(&e, &S, dir, 1024);
    engram_advise_rows(&e, 0, 0, POSITIONS);
    check(e.table[0].staged == distinct, "V41_ENGRAM_STAGE=1: engram_advise_rows reads the block's rows");
    close_engram(&e, &S);

    free(ref); free(got);
    char path[600];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir); unlink(path);
    snprintf(path, sizeof(path), "%s/dsv41_engram.json", dir); unlink(path);
    rmdir(dir);
    if (g_fails) { printf("%d check(s) failed\n", g_fails); return 1; }
    printf("dsv41 engram stage: staged rows give the gather the same bits, each missing row "
           "read once, at most half the cache a block -- ok\n");
    return 0;
}
