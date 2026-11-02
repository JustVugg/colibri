/* deepseek_v41.c reads three things from the snapshot directory that are as
 * attacker-chosen as the weights themselves, and all trusted the file for a size
 * nothing had checked:
 *
 *  - wf_load() allocated `n` floats and let st_read_f32 write as many as the TENSOR
 *    declares, comparing the count only after the copy. A norm.weight with more
 *    elements than the config implies overran the heap with file bytes, and the
 *    check then reported a write it could no longer undo.
 *  - engram_load_sidecar() walked primes/offsets/multipliers by the length of
 *    layer_ids -- a different, equally file-chosen number -- and dereferenced
 *    whatever sat past the end of a shorter array, or a NULL kids pointer when
 *    the key was not an array at all.
 *  - engram_row() read row `id` of a table at id * head_dim in the shard. The id is
 *    a hash bucket plus a sidecar offset, head_dim is the sidecar's, the row count
 *    and the scale count are the shards': none was compared with another, so an id
 *    past the table read whatever the file holds there.
 *
 * All refusals exit(1) in place, so those cases run in a forked child with stderr
 * captured (the idiom of tests/test_dup_name_refusal.c). Under `make test-asan` the
 * unfixed loader is a heap-buffer-overflow report instead of the refusal message
 * these checks ask for; in a plain build it is the wrong message, or a crash. */
#define main dsv41_main_unused
#include "../deepseek_v41.c"
#undef main
#ifndef _WIN32
#include <sys/wait.h>
#endif

static int g_fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("FAIL: %s\n", what); g_fails++; }
}

#ifndef _WIN32
static char g_dir[512];
static int64_t g_want;

/* one F32 tensor holding 1, 2, ..., n */
static void write_f32_shard(const char *dir, const char *name, int n) {
    char path[600], hdr[256];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    int hl = snprintf(hdr, sizeof(hdr),
        "{\"%s\":{\"dtype\":\"F32\",\"shape\":[%d],\"data_offsets\":[0,%d]}}", name, n, n * 4);
    uint64_t hlen = (uint64_t)hl;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    for (int i = 0; i < n; i++) { float v = (float)(i + 1); fwrite(&v, 4, 1, f); }
    fclose(f);
}

static void write_sidecar(const char *dir, const char *json) {
    char path[600];
    snprintf(path, sizeof(path), "%s/dsv41_engram.json", dir);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fputs(json, f);
    fclose(f);
}

static void run_forked(void (*fn)(void), int *exit_code, char *err, size_t err_sz) {
    int pipefd[2];
    if (pipe(pipefd) != 0) { *exit_code = -1; err[0] = 0; return; }
    fflush(stdout);  /* else a child that exit()s flushes the parent's FAIL lines again */
    pid_t pid = fork();
    if (pid < 0) { *exit_code = -1; err[0] = 0; return; }
    if (pid == 0) {
        dup2(pipefd[1], 2); close(pipefd[0]); close(pipefd[1]);
        fn();
        _exit(42);  /* reaching here means fn() did NOT refuse */
    }
    close(pipefd[1]);
    size_t off = 0; ssize_t n;
    while (off < err_sz - 1 && (n = read(pipefd[0], err + off, err_sz - 1 - off)) > 0) off += (size_t)n;
    err[off] = 0;
    close(pipefd[0]);
    int status = 0; waitpid(pid, &status, 0);
    *exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static void child_wf_load(void) {
    shards S; st_init(&S, g_dir);
    WF w; wf_load(&S, &w, "norm.weight", g_want);
}

static Engram g_engram;
static void child_engram(void) {
    memset(&g_engram, 0, sizeof(g_engram));
    engram_load_sidecar(&g_engram, g_dir);
}

static void test_wf_load(void) {
    char dir[] = "test_dsv41_untrusted_wf_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); check(0, "wf_load: mkdtemp"); return; }
    snprintf(g_dir, sizeof(g_dir), "%s", dir);
    int code; char err[8192];

    /* longer than the config says: refused before a single float is written */
    write_f32_shard(dir, "norm.weight", 64);
    g_want = 4;
    run_forked(child_wf_load, &code, err, sizeof(err));
    check(code == 1, "wf_load: 64-element tensor into 4 floats exits(1)");
    check(strstr(err, "exceeds destination capacity") != NULL,
          "wf_load: refused by the capacity check, before the copy");
    check(strstr(err, "AddressSanitizer") == NULL, "wf_load: no heap overflow on the way out");
    if (g_fails) printf("--- child stderr ---\n%s\n", err);

    /* shorter than the config says: still refused, by the count check after the read */
    write_f32_shard(dir, "norm.weight", 2);
    run_forked(child_wf_load, &code, err, sizeof(err));
    check(code == 1, "wf_load: 2-element tensor for 4 floats exits(1)");
    check(strstr(err, "expected 4 floats") != NULL, "wf_load: short tensor names the expected count");

    /* exact: loads, and the values are the file's */
    write_f32_shard(dir, "norm.weight", 4);
    shards S; st_init(&S, dir);
    WF w; wf_load(&S, &w, "norm.weight", 4);
    check(w.n == 4 && w.w[0] == 1.f && w.w[3] == 4.f, "wf_load: exact tensor loads its values");
    free(w.w);

    char path[600];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir); unlink(path);
    rmdir(dir);
}

/* two tables; every per-table array has two entries */
#define ENGRAM_OK \
    "{\"max_ngram_size\":3,\"n_heads\":2,\"head_dim\":32,\"pad_id\":0,\"layer_ids\":[1,2]," \
    "\"primes\":[[[5,7],[11,13]],[[17,19],[23,29]]]," \
    "\"offsets\":[[0,5,12,23],[0,17,36,59]]," \
    "\"multipliers\":[[\"3\",\"5\",\"7\"],[\"11\",\"13\",\"17\"]]," \
    "\"token_map\":[0,1,2]}"

static void expect_engram_refused(const char *dir, const char *json, const char *key, const char *what) {
    int code; char err[8192];
    write_sidecar(dir, json);
    run_forked(child_engram, &code, err, sizeof(err));
    char label[160];
    snprintf(label, sizeof(label), "engram: %s exits(1)", what);
    check(code == 1, label);
    snprintf(label, sizeof(label), "engram: %s is refused by name", what);
    int named = strstr(err, "[engram]") != NULL && strstr(err, key) != NULL;
    check(named, label);
    if (code != 1 || !named) printf("--- child stderr (exit %d) ---\n%s\n", code, err);
}

static void test_engram_sidecar(void) {
    char dir[] = "test_dsv41_untrusted_engram_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); check(0, "engram: mkdtemp"); return; }
    snprintf(g_dir, sizeof(g_dir), "%s", dir);

    write_sidecar(dir, ENGRAM_OK);
    child_engram();
    check(g_engram.active == 1 && g_engram.n_layers == 2, "engram: well-formed sidecar loads both tables");
    check(g_engram.primes[1][1][1] == 29 && g_engram.offsets[1][3] == 59 &&
          g_engram.multipliers[1][2] == 17 && g_engram.token_map_len == 3,
          "engram: well-formed sidecar keeps its values");
    free(g_engram.token_map);

    expect_engram_refused(dir,
        "{\"max_ngram_size\":3,\"n_heads\":2,\"head_dim\":32,\"layer_ids\":[1,2],"
        "\"primes\":[[[5,7],[11,13]]],\"offsets\":[[0,5,12,23],[0,17,36,59]],"
        "\"multipliers\":[[\"3\",\"5\",\"7\"],[\"11\",\"13\",\"17\"]],\"token_map\":[0]}",
        "primes", "primes shorter than layer_ids");
    expect_engram_refused(dir,
        "{\"max_ngram_size\":3,\"n_heads\":2,\"head_dim\":32,\"layer_ids\":[1,2],"
        "\"primes\":[[[5,7],[11,13]],[[17,19],[23,29]]],\"offsets\":7,"
        "\"multipliers\":[[\"3\",\"5\",\"7\"],[\"11\",\"13\",\"17\"]],\"token_map\":[0]}",
        "offsets", "offsets that is a number");
    /* an object has kids and a len, so only the type check stands between it and
     * a table read out of a map */
    expect_engram_refused(dir,
        "{\"max_ngram_size\":3,\"n_heads\":2,\"head_dim\":32,\"layer_ids\":[1,2],"
        "\"primes\":[[[5,7],[11,13]],[[17,19],[23,29]]],"
        "\"offsets\":{\"a\":[0,5,12,23],\"b\":[0,17,36,59]},"
        "\"multipliers\":[[\"3\",\"5\",\"7\"],[\"11\",\"13\",\"17\"]],\"token_map\":[0]}",
        "offsets", "offsets that is an object");
    expect_engram_refused(dir,
        "{\"max_ngram_size\":3,\"n_heads\":2,\"head_dim\":32,\"layer_ids\":[1,2],"
        "\"primes\":[[[5,7],[11,13]],[[17,19],[23,29]]],\"offsets\":[[0,5,12,23],[0,17,36,59]],"
        "\"multipliers\":[[\"3\",\"5\",\"7\"]],\"token_map\":[0]}",
        "multipliers", "multipliers shorter than layer_ids");

    char path[600];
    snprintf(path, sizeof(path), "%s/dsv41_engram.json", dir); unlink(path);
    rmdir(dir);
}

static uint8_t engram_byte(int row) { return (uint8_t)(0x30 + row % 8); }
static uint8_t engram_scale(int row) { return (uint8_t)(127 + row % 3); }

/* layers.1.engram.embed.weight as `rows` x `row_bytes` e4m3 bytes and its scale as
 * `scale_rows` x `scale_cols` ue8m0 bytes, then a 64-byte tail: a read past either
 * tensor lands in file data, as it would in a real shard, not at EOF. */
static void write_engram_shard(const char *dir, int rows, int row_bytes, int scale_rows,
                               int scale_cols) {
    char path[600], hdr[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    long long w = (long long)rows * row_bytes, s = (long long)scale_rows * scale_cols;
    int hl = snprintf(hdr, sizeof(hdr),
        "{\"layers.1.engram.embed.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[%d,%d],"
        "\"data_offsets\":[0,%lld]},"
        "\"layers.1.engram.embed.scale\":{\"dtype\":\"U8\",\"shape\":[%d,%d],"
        "\"data_offsets\":[%lld,%lld]},"
        "\"tail\":{\"dtype\":\"U8\",\"shape\":[64],\"data_offsets\":[%lld,%lld]}}",
        rows, row_bytes, w, scale_rows, scale_cols, w, w + s, w + s, w + s + 64);
    uint64_t hlen = (uint64_t)hl;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < row_bytes; i++) fputc(engram_byte(r), f);
    for (int r = 0; r < scale_rows; r++)
        for (int g = 0; g < scale_cols; g++) fputc(engram_scale(r), f);
    for (int i = 0; i < 64; i++) fputc(0x41, f);
    fclose(f);
}

static int g_head_dim;
static int64_t g_id;

/* layer 1's table opened with g_head_dim, then row g_id looked up */
static void child_engram_row(void) {
    shards S; st_init(&S, g_dir);
    EngramTable t; engram_table_open(&t, &S, 1, g_head_dim, 16);
    engram_row(&t, g_id, g_head_dim);
}

/* the sidecar's hash over a short history, every column looked up in table 0 */
static void child_engram_lookup(void) {
    memset(&g_engram, 0, sizeof(g_engram));
    engram_load_sidecar(&g_engram, g_dir);
    shards S; st_init(&S, g_dir);
    engram_table_open(&g_engram.table[0], &S, g_engram.layer_of[0], g_engram.head_dim, 16);
    static const int ids[] = {1, 2, 0, 2, 1};
    engram_push(&g_engram, ids, 5, NULL);
    int64_t rows[V41_MAX_NGRAM * V41_MAX_EHEADS];
    for (int pos = 0; pos < 5; pos++) {
        engram_hash(&g_engram, 0, pos, rows);
        for (int col = 0; col < g_engram.cols; col++)
            engram_row(&g_engram.table[0], rows[col], g_engram.head_dim);
    }
}

static void expect_table_refused(void (*fn)(void), const char *key, const char *what) {
    int code; char err[8192];
    run_forked(fn, &code, err, sizeof(err));
    char label[160];
    snprintf(label, sizeof(label), "engram table: %s exits(1)", what);
    check(code == 1, label);
    snprintf(label, sizeof(label), "engram table: %s is refused by name", what);
    int named = strstr(err, "[engram]") != NULL && strstr(err, key) != NULL;
    check(named, label);
    if (code != 1 || !named) printf("--- child stderr (exit %d) ---\n%s\n", code, err);
}

static void test_engram_table(void) {
    char dir[] = "test_dsv41_untrusted_table_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); check(0, "engram table: mkdtemp"); return; }
    snprintf(g_dir, sizeof(g_dir), "%s", dir);
    int code; char err[8192];

    /* ENGRAM_OK's table 0 has 5 + 7 + 11 + 13 = 36 rows: every id its hash makes fits */
    write_engram_shard(dir, 36, 32, 36, 1);
    write_sidecar(dir, ENGRAM_OK);
    run_forked(child_engram_lookup, &code, err, sizeof(err));
    check(code == 42, "engram table: a sidecar that describes the table looks every n-gram up");
    if (code != 42) printf("--- child stderr (exit %d) ---\n%s\n", code, err);

    shards S; st_init(&S, dir);
    EngramTable t; engram_table_open(&t, &S, 1, 32, 16);
    check(t.rows == 36, "engram table: 36 rows of 32 bytes");
    for (int row = 0; row < 36; row += 35) {
        const float *v = engram_row(&t, row, 32);
        float want = e4m3_decode(engram_byte(row)) * ue8m0(engram_scale(row));
        check(v[0] == want && v[31] == want, "engram table: first and last rows read their own bytes");
    }
    free(t.key); free(t.used); free(t.value);

    /* the last column starts where the table ends: a sidecar for another table */
    write_sidecar(dir,
        "{\"max_ngram_size\":3,\"n_heads\":2,\"head_dim\":32,\"pad_id\":0,\"layer_ids\":[1,2],"
        "\"primes\":[[[5,7],[11,13]],[[17,19],[23,29]]],"
        "\"offsets\":[[0,5,12,36],[0,17,36,59]],"
        "\"multipliers\":[[\"3\",\"5\",\"7\"],[\"11\",\"13\",\"17\"]],"
        "\"token_map\":[0,1,2]}");
    expect_table_refused(child_engram_lookup, "outside the table", "a sidecar hashing past the table");

    /* the ids themselves, at both ends */
    g_head_dim = 32;
    g_id = 36;
    expect_table_refused(child_engram_row, "outside the table", "row 36 of 36");
    g_id = -1;
    expect_table_refused(child_engram_row, "outside the table", "row -1, the empty-slot key");
    g_id = INT64_MAX / 16;
    expect_table_refused(child_engram_row, "outside the table", "a row whose offset overflows");

    /* the table, its scales and head_dim disagree */
    g_id = 35;
    write_engram_shard(dir, 36, 32, 35, 1);
    expect_table_refused(child_engram_row, "scale bytes", "a scale tensor one row short");
    write_engram_shard(dir, 36, 32, 37, 1);
    expect_table_refused(child_engram_row, "scale bytes", "a scale tensor one row long");
    write_engram_shard(dir, 1, 36 * 32 + 1, 36, 1);
    expect_table_refused(child_engram_row, "scale bytes", "a table that is not whole rows");
    write_engram_shard(dir, 36, 32, 36, 1);
    g_head_dim = 0; g_id = 0;
    expect_table_refused(child_engram_row, "multiple of 32", "head_dim 0");
    write_engram_shard(dir, 4, 48, 4, 1);
    g_head_dim = 48; g_id = 3;
    expect_table_refused(child_engram_row, "multiple of 32", "head_dim 48");

    char path[600];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir); unlink(path);
    snprintf(path, sizeof(path), "%s/dsv41_engram.json", dir); unlink(path);
    rmdir(dir);
}
#endif

int main(void) {
#ifndef _WIN32
    test_wf_load();
    test_engram_sidecar();
    test_engram_table();
#else
    printf("dsv41 untrusted load: skipped on Windows (no fork)\n");
#endif
    if (g_fails) { printf("%d check(s) failed\n", g_fails); return 1; }
    printf("dsv41 untrusted load: wf_load capacity + engram sidecar shape + engram table "
           "bounds -- ok\n");
    return 0;
}
