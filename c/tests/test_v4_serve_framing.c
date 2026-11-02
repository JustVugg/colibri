/* Freeze DeepSeek V4's serving wire before adopting serve_codec.h. */
#define COLI_V4_UNIT_GENERATE_STATS
#define COLI_V4_SKIP_GENERATE_MAIN
#include "../deepseek_v4.c"

#include <assert.h>

static void binary_stream(FILE *stream)
{
#ifdef _WIN32
    assert(_setmode(_fileno(stream), _O_BINARY) != -1);
#else
    (void)stream;
#endif
}

static FILE *bytes_input(const void *data, size_t size)
{
    FILE *stream = tmpfile();
    assert(stream);
    binary_stream(stream);
    assert(fwrite(data, 1, size, stream) == size);
    rewind(stream);
    return stream;
}

static size_t read_output(FILE *stream, unsigned char *output, size_t capacity)
{
    assert(fflush(stream) == 0);
    rewind(stream);
    return fread(output, 1, capacity, stream);
}

static void test_submit_shapes_and_controls(void)
{
    static const char base[] = "SUBMIT base 0 5 4 0.7 0.95\nab\ncd\n";
    FILE *input = bytes_input(base, sizeof(base) - 1);
    FILE *output = tmpfile(); assert(output); binary_stream(output);
    V4ServeRequest request = {0};
    assert(v4_serve_read_request(input, output, &request, NULL) == 2);
    assert(strcmp(request.id, "base") == 0);
    assert(request.prompt_bytes == 5 && request.extension_bytes == 0);
    assert(request.prefix_bytes == 0 && memcmp(request.prompt, "ab\ncd", 5) == 0);
    assert(fgetc(input) == EOF);
    free(request.prompt); fclose(input); fclose(output);

    static const char extended[] =
        "SUBMIT ext 0 3 4 0 1 4 2\nabcWXYZ\n";
    input = bytes_input(extended, sizeof(extended) - 1);
    output = tmpfile(); assert(output); binary_stream(output);
    memset(&request, 0, sizeof(request));
    assert(v4_serve_read_request(input, output, &request, NULL) == 2);
    assert(request.prompt_bytes == 3 && request.extension_bytes == 4);
    assert(request.prefix_bytes == 2 && memcmp(request.prompt, "abc", 3) == 0);
    assert(memcmp(request.prompt + request.prompt_bytes + 1, "WXYZ", 4) == 0);
    assert(fgetc(input) == EOF);
    free(request.prompt); fclose(input); fclose(output);

    static const char stop[] = "STOP live\n";
    input = bytes_input(stop, sizeof(stop) - 1);
    output = tmpfile(); assert(output); binary_stream(output);
    assert(v4_serve_read_request(input, output, &request, "live") == 1);
    fclose(input); fclose(output);

    static const char cancel[] = "CANCEL live\n";
    input = bytes_input(cancel, sizeof(cancel) - 1);
    output = tmpfile(); assert(output); binary_stream(output);
    assert(v4_serve_read_request(input, output, &request, "other") == 0);
    fclose(input); fclose(output);
}

static void test_errors_and_writer_bytes(void)
{
    static const char bad[] = "SUBMIT bad 1 0 4 0 1\n\n";
    FILE *input = bytes_input(bad, sizeof(bad) - 1);
    FILE *output = tmpfile(); assert(output); binary_stream(output);
    V4ServeRequest request = {0};
    assert(v4_serve_read_request(input, output, &request, NULL) == -2);
    unsigned char actual[512];
    size_t count = read_output(output, actual, sizeof(actual));
    static const char bad_want[] = "ERROR bad bad submit header\n";
    assert(count == sizeof(bad_want) - 1);
    assert(memcmp(actual, bad_want, count) == 0);
    fclose(input); fclose(output);

    output = tmpfile(); assert(output); binary_stream(output);
    v4_serve_data(output, "req-7", "A\n\0B", 4);
    v4_serve_error(output, "req-7", "bad\r\nframe");
    v4_serve_done(output, "req-7", 3, 1.25, 50.0, 1.25, 7, 1, 5);
    static const unsigned char want[] =
        "DATA req-7 4\nA\n\0B\n"
        "ERROR req-7 bad  frame\n"
        "DONE req-7 STAT 3 1.250 50.0 1.25 7 1 5\n";
    count = read_output(output, actual, sizeof(actual));
    assert(count == sizeof(want) - 1 && memcmp(actual, want, count) == 0);
    fclose(output);
}

static void test_extension_rejection_precedes_accept(void)
{
    V4ServeRequest request = {
        .prompt = "abc",
        .prompt_bytes = 3,
        .max_tokens = 4,
        .extension_bytes = 4,
    };
    snprintf(request.id, sizeof(request.id), "ext");
    request.temperature = 0.0f;
    request.top_p = 1.0f;

    FILE *output = tmpfile(); assert(output); binary_stream(output);
    int saved = dup(fileno(stdout)); assert(saved >= 0);
    assert(fflush(stdout) == 0);
    assert(dup2(fileno(output), fileno(stdout)) >= 0);
    assert(v4_serve_one(NULL, NULL, &request) == 0);
    assert(fflush(stdout) == 0);
    assert(dup2(saved, fileno(stdout)) >= 0); close(saved);

    unsigned char actual[128];
    size_t count = read_output(output, actual, sizeof(actual));
    static const char want[] = "ERROR ext unsupported request extension\n";
    assert(count == sizeof(want) - 1 && memcmp(actual, want, count) == 0);
    fclose(output);
}

/* ---- KV cold tier v2 (P1.1): hermetic disk-catalog gates T1-T7.
 *
 * Same trick as the framing suites above: this TU IS the
 * COLI_V4_UNIT_GENERATE_STATS unit, so the tier's statics are reachable. A
 * synthetic ratio-0 session (window ring only, no compressor/indexer, so a
 * layer's snapshot is exactly its 20-byte header + the ring) against a
 * scratch directory exercises the on-disk contract without a model:
 * store/scan/restore round-trip, the COLIV4C3 header, torn-file and
 * checksum rejection, GC caps, the legacy-import policy and per-session
 * plan state. The bodies carry no #ifdef: the implementation's per-OS scan
 * arms make the round-trip gates run on the Windows leg too (gate T8). */
static void test_setenv(const char *name, const char *value)
{
#ifdef _WIN32
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%s=%s", name, value);
    _putenv(buffer);
#else
    setenv(name, value, 1);
#endif
}

static char *test_mkdtemp(char *buffer)
{
    const char *base;
#ifdef _WIN32
    base = getenv("TEMP");
    if (!base || !*base) base = ".";
#else
    base = getenv("TMPDIR");
    if (!base || !*base) base = "/tmp";
#endif
    snprintf(buffer, 256, "%s/ckpt_test_XXXXXX", base);
#ifdef _WIN32
    if (_mktemp(buffer) == NULL) return NULL;
    if (_mkdir(buffer)) return NULL;
    return buffer;
#else
    return mkdtemp(buffer);
#endif
}

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <fcntl.h>
#define test_rmdir(path) _rmdir(path)
#else
#define test_rmdir(path) rmdir(path)
#endif

static void test_file_truncate(const char *path, long bytes)
{
#ifdef _WIN32
    int fd = _open(path, _O_WRONLY);
    assert(fd >= 0);
    assert(_chsize(fd, bytes) == 0);
    _close(fd);
#else
    assert(truncate(path, bytes) == 0);
#endif
}

static size_t test_file_size(const char *path)
{
    struct stat st;
    assert(stat(path, &st) == 0);
    return (size_t)st.st_size;
}

static int test_file_exists(const char *path)
{
    FILE *stream = fopen(path, "rb");
    if (!stream) return 0;
    fclose(stream);
    return 1;
}

/* The synthetic session every disk-tier gate shares: one ratio-0 layer, so
 * a snapshot is exactly its 20-byte header + the window ring and the
 * prepare sweep never runs (fed is recorded before every restore). */
#define CKPT_TEST_RING_FLOATS 32   /* sliding_window 8 x head_dim 4 */
#define CKPT_TEST_SNAPSHOT_BYTES (20 + CKPT_TEST_RING_FLOATS * 4)

static void ckpt_fake_session(ColiV4Session *session,
                              ColiV4Engine *engine, int *ids)
{
    memset(session, 0, sizeof(*session));
    memset(engine, 0, sizeof(*engine));
    engine->runtime.context_tokens = 4096;
    session->engine = engine;
    session->prompt_ids = ids;
    session->config.hidden_size = 16;
    session->config.hc_mult = 1;
    session->config.num_hidden_layers = 1;
    session->config.sliding_window = 8;
    session->config.head_dim = 4;
    session->config.index_head_dim = 4;
    session->config.compress_ratio_count = 1;
    session->config.compress_ratios[0] = 0;   /* ratio-0: ring only */
    assert(kv_prefix_alloc(&session->fed, 4096));
    ColiDeepSeekV4WindowAttentionState **attention =
        calloc(1, sizeof(*attention));
    assert(attention);
    assert(coli_v4_window_attention_create(&attention[0],
                                           &session->config) == 0);
    session->attention = attention;
}

static void ckpt_fake_session_free(ColiV4Session *session)
{
    coli_v4_window_attention_destroy(session->attention[0]);
    free(session->attention);
    kv_prefix_free(&session->fed);
    free(session->ckpt_prev_ids);
    session->ckpt_prev_ids = NULL;
    session->ckpt_prev_len = 0;
}

/* One gate = one scratch directory (under the system temp, so a failed run
 * leaves nothing in the repo) with a known fingerprint, empty catalog, scan
 * suppressed (tests trigger it explicitly). Restores v4_ckpt_dir on exit. */
static void ckpt_scratch_enter(char *saved_dir, size_t size, char *scratch)
{
    snprintf(saved_dir, size, "%s", v4_ckpt_dir);
    v4_ckpt_fingerprint = 0x600dfee7u;
    v4_ckpt_fingerprint_v1 = 0x600dfee7u;
    v4_ckpt_disk_scanned = 1;
    v4_ckpt_catalog_clear();
    assert(test_mkdtemp(scratch));
    snprintf(v4_ckpt_dir, sizeof(v4_ckpt_dir), "%s", scratch);
}

static void ckpt_scratch_exit(const char *saved_dir, const char *scratch)
{
    assert(test_rmdir(scratch) == 0);
    snprintf(v4_ckpt_dir, sizeof(v4_ckpt_dir), "%s", saved_dir);
    v4_ckpt_disk_scanned = 1;
    v4_ckpt_catalog_clear();
}

/* Hand-crafted ratio-0 snapshot payload (the type is opaque in this unit,
 * the on-disk format is plain: int32[5] then the ring). */
static void ckpt_craft_payload(uint8_t *payload)
{
    int32_t head[5] = {8, 4, 0, 0, 0};
    memcpy(payload, head, sizeof(head));
    float ring[CKPT_TEST_RING_FLOATS];
    for (int i = 0; i < CKPT_TEST_RING_FLOATS; i++)
        ring[i] = (float)(i * 0.25f) - 4.0f;
    memcpy(payload + 20, ring, sizeof(ring));
}

/* Rewrite an entry's file: same header and ids, given payload, CRC patched. */
static void ckpt_rewrite_payload(const char *path, const int *ids, int len,
                                 const uint8_t *payload, size_t payload_bytes)
{
    FILE *stream = fopen(path, "wb");
    assert(stream);
    static const char magic[9] = "COLIV4C3";
    uint32_t fp = v4_ckpt_fingerprint;
    int32_t head[6] = {3, 1, len, 1, (int32_t)payload_bytes,
                       (int32_t)v4_ckpt_crc32c(0, payload, payload_bytes)};
    assert(fwrite(magic, 1, 8, stream) == 8);
    assert(fwrite(&fp, 4, 1, stream) == 1);
    assert(fwrite(head, 4, 6, stream) == 6);
    assert(fwrite(ids, 4, len, stream) == (size_t)len);
    assert(fwrite(payload, 1, payload_bytes, stream) == payload_bytes);
    assert(fclose(stream) == 0);
}

/* T1 round-trip + T2 restart rescan + T5 checksum flip. */
static void test_cold_tier_roundtrip(void)
{
    int minimum = v4_ckpt_min_tokens();
    int len = minimum + 6, total = len + 30;
    int *ids = malloc((size_t)total * sizeof(int));
    assert(ids);
    for (int i = 0; i < total; i++) ids[i] = 5000 + i;

    ColiV4Engine engine;
    ColiV4Session session;
    ckpt_fake_session(&session, &engine, ids);
    char saved_dir[1024], scratch[300];
    ckpt_scratch_enter(saved_dir, sizeof(saved_dir), scratch);

    /* STORE: header + ids byte-exact, no temp left behind. */
    v4_ckpt_store(&session, len, 1);
    assert(v4_ckpt_cat_n == 1 && v4_ckpt_cat[0].len == len);
    assert(v4_ckpt_cat[0].kind == 1 && v4_ckpt_cat[0].bytes > 0);
    assert(v4_ckpt_cat[0].file[0] && v4_ckpt_cat[0].verified == 1);
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", v4_ckpt_dir,
             v4_ckpt_cat[0].file);
    FILE *stream = fopen(path, "rb");
    assert(stream);
    char magic[8];
    uint32_t fp = 0;
    int32_t head[6];
    assert(fread(magic, 1, 8, stream) == 8 && !memcmp(magic, "COLIV4C3", 8));
    assert(fread(&fp, 4, 1, stream) == 1 && fp == v4_ckpt_fingerprint);
    assert(fread(head, 4, 6, stream) == 6);
    assert(head[0] == 3 && head[1] == 1 && head[2] == len && head[3] == 1);
    assert(head[4] == CKPT_TEST_SNAPSHOT_BYTES);
    assert(head[5] != 0);   /* crc of a non-empty payload */
    int *ids_back = malloc((size_t)len * sizeof(int));
    assert(ids_back);
    assert(fread(ids_back, 4, len, stream) == (size_t)len);
    assert(!memcmp(ids_back, ids, (size_t)len * sizeof(int)));
    free(ids_back);
    fclose(stream);
    assert(test_file_size(path) == 36 + (size_t)len * 4 + CKPT_TEST_SNAPSHOT_BYTES);
    assert(v4_ckpt_cat[0].bytes == test_file_size(path));
    char tmp[1280];
    snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path, (int)v4_ckpt_pid());
    assert(!test_file_exists(tmp));

    /* DATA ROUND-TRIP: replace the payload with hand-crafted bytes (crc in
     * the header patched to match), rescan like a restart, restore -- the
     * state's own serialization must return those exact bytes. */
    uint8_t payload[CKPT_TEST_SNAPSHOT_BYTES];
    ckpt_craft_payload(payload);
    ckpt_rewrite_payload(path, ids, len, payload, sizeof(payload));

    /* T2: restart rescan rediscovers the row; a second scan is a no-op. */
    v4_ckpt_catalog_clear();
    v4_ckpt_disk_scanned = 0;
    v4_ckpt_disk_scan();
    assert(v4_ckpt_cat_n == 1 && v4_ckpt_cat[0].len == len);
    assert(v4_ckpt_cat[0].verified == 0);   /* payload unchecked until use */
    v4_ckpt_disk_scanned = 0;
    v4_ckpt_disk_scan();
    assert(v4_ckpt_cat_n == 1);

    /* Match rules: a fork AT the boundary extends it (restore); a prompt
     * at or inside the boundary does not (no rewind). */
    assert(v4_ckpt_best_match(ids, total) == len);
    assert(v4_ckpt_best_match(ids, len) == 0);
    assert(v4_ckpt_have(ids, len));
    kv_prefix_record(&session.fed, ids, 0, len);   /* fed: no prepare sweep */
    assert(v4_ckpt_restore(&session, total) == len);
    FILE *serialized = tmpfile();
    assert(serialized);
    ColiV4AttentionSnapshot *snapshot = NULL;
    assert(coli_v4_attention_snapshot_create(session.attention[0],
                                             &snapshot) == 0);
    assert(coli_v4_attention_snapshot_write(snapshot, serialized) == 0);
    coli_v4_attention_snapshot_destroy(snapshot);
    assert(fflush(serialized) == 0 && fseek(serialized, 0, SEEK_SET) == 0);
    uint8_t back[sizeof(payload)];
    assert(fread(back, 1, sizeof(back), serialized) == sizeof(back));
    assert(!memcmp(back, payload, sizeof(payload)));
    fclose(serialized);
    assert(v4_ckpt_cat[0].verified == 1);   /* verify ran once, lazily */

    /* T5: one flipped payload byte -> verify fails on the next restore,
     * the row is dropped and the file removed. */
    stream = fopen(path, "r+b");
    assert(stream);
    assert(fseek(stream, 36 + (long)len * 4 + 5, SEEK_SET) == 0);
    int flipped = fgetc(stream);
    assert(fseek(stream, -1, SEEK_CUR) == 0);
    assert(fputc(flipped ^ 0x40, stream) != EOF);
    assert(fclose(stream) == 0);
    v4_ckpt_cat[0].verified = 0;
    kv_prefix_record(&session.fed, ids, 0, len);
    assert(v4_ckpt_restore(&session, total) == 0);
    assert(v4_ckpt_cat_n == 0);
    assert(!test_file_exists(path));

    ckpt_fake_session_free(&session);
    free(ids);
    ckpt_scratch_exit(saved_dir, scratch);
}

/* T3: torn at the header / inside the ids -> the scan rejects and removes
 * (our magic, our fingerprint, regenerable); a payload torn so the sizes
 * still agree passes the scan (header + ids only) and only restore catches
 * it: drop the row, remove the file. */
static void test_cold_tier_torn_files(void)
{
    int minimum = v4_ckpt_min_tokens();
    int len = minimum + 4;
    int *ids = malloc((size_t)len * sizeof(int));
    assert(ids);
    for (int i = 0; i < len; i++) ids[i] = 3100 + i;

    ColiV4Engine engine;
    ColiV4Session session;
    ckpt_fake_session(&session, &engine, ids);
    char saved_dir[1024], scratch[300];
    ckpt_scratch_enter(saved_dir, sizeof(saved_dir), scratch);

    /* torn inside the header: the header-sized read fails. */
    v4_ckpt_store(&session, len, 1);
    assert(v4_ckpt_cat_n == 1);
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", v4_ckpt_dir,
             v4_ckpt_cat[0].file);
    size_t whole = test_file_size(path);
    test_file_truncate(path, (long)whole - 30);
    v4_ckpt_catalog_clear();
    v4_ckpt_disk_scanned = 0;
    v4_ckpt_disk_scan();
    assert(v4_ckpt_cat_n == 0);
    assert(!test_file_exists(path));

    /* torn inside the ids: header ok, ids short. */
    v4_ckpt_store(&session, len, 1);
    assert(v4_ckpt_cat_n == 1);
    whole = test_file_size(path);
    test_file_truncate(path, (long)(36 + (size_t)len * 4 - 8));
    v4_ckpt_catalog_clear();
    v4_ckpt_disk_scanned = 0;
    v4_ckpt_disk_scan();
    assert(v4_ckpt_cat_n == 0);
    assert(!test_file_exists(path));

    /* payload torn but size-consistent: the scan accepts the row, restore
     * must drop + remove it (the snapshot stream comes up short). */
    v4_ckpt_store(&session, len, 1);
    assert(v4_ckpt_cat_n == 1);
    uint8_t short_payload[4] = {1, 2, 3, 4};
    ckpt_rewrite_payload(path, ids, len, short_payload, sizeof(short_payload));
    v4_ckpt_catalog_clear();
    v4_ckpt_disk_scanned = 0;
    v4_ckpt_disk_scan();
    assert(v4_ckpt_cat_n == 1);   /* the scan reads header + ids only */
    assert(v4_ckpt_cat[0].bytes == 36 + (uint64_t)len * 4 + 4);
    kv_prefix_record(&session.fed, ids, 0, len);
    assert(v4_ckpt_restore(&session, len + 8) == 0);
    assert(v4_ckpt_cat_n == 0);
    assert(!test_file_exists(path));

    ckpt_fake_session_free(&session);
    free(ids);
    ckpt_scratch_exit(saved_dir, scratch);
}

/* T4: GC caps. Prompt-end cap evicts the LRU prompt-end WITH its file; the
 * byte budget stops at pinned (kind=0) rows; the ids-RAM cap evicts without
 * any payload bytes; beyond the pin cap the LRU prefix goes too. */
static void test_cold_tier_gc(void)
{
    int ids[128], other[128], third[128];
    for (int i = 0; i < 128; i++) {
        ids[i] = 7000 + i;
        other[i] = 8000 + i;
        third[i] = 9000 + i;
    }
    v4_ckpt_catalog_clear();
    int saved_cap = v4_ckpt_prompt_end_cap_cache;
    int saved_read = v4_ckpt_disk_budget_read;
    uint64_t saved_budget = v4_ckpt_disk_budget_cache;

    /* prompt-end cap 1: the LRU prompt-end row is evicted with its file. */
    v4_ckpt_prompt_end_cap_cache = 1;
    v4_ckpt_disk_budget_read = 1;
    v4_ckpt_disk_budget_cache = 0;   /* unlimited payload bytes */
    assert(v4_ckpt_catalog_add(ids, 100, 1, 1ull << 20, "kv_gc_old.bin") >= 0);
    assert(v4_ckpt_catalog_add(other, 100, 1, 1ull << 20, "kv_gc_new.bin") >= 0);
    v4_ckpt_gc();
    assert(v4_ckpt_cat_n == 1);
    assert(v4_ckpt_cat[0].len == 100 && v4_ckpt_cat[0].kind == 1);
    assert(memcmp(v4_ckpt_cat[0].ids, other, 100 * sizeof(int)) == 0);

    /* byte budget: a pinned prefix survives an overrun alone... */
    v4_ckpt_disk_budget_cache = 1;   /* 1 byte: everything is over */
    assert(v4_ckpt_catalog_add(ids, 100, 0, 1ull << 30, "kv_gc_pin.bin") >= 0);
    v4_ckpt_gc();
    assert(v4_ckpt_cat_n == 1 && v4_ckpt_cat[0].kind == 0);
    /* ...until prefixes exceed the pin cap (4 here: no mux slots): the LRU
     * prefix becomes evictable. */
    assert(v4_ckpt_catalog_add(third, 100, 0, 1ull << 10, "kv_gc_p2.bin") >= 0);
    assert(v4_ckpt_catalog_add(ids + 8, 100, 0, 1ull << 10, "kv_gc_p3.bin") >= 0);
    assert(v4_ckpt_catalog_add(other + 8, 100, 0, 1ull << 10, "kv_gc_p4.bin") >= 0);
    assert(v4_ckpt_catalog_add(third + 8, 100, 0, 1ull << 10, "kv_gc_p5.bin") >= 0);
    v4_ckpt_gc();
    assert(v4_ckpt_cat_n == 4);   /* pin cap: the LRU 5th prefix went */

    /* ids-RAM cap: zero payload bytes, but two 100-id rows (800 B of ids)
     * overrun budget/8 = 8 B -> the ids cap alone evicts them. */
    v4_ckpt_catalog_clear();
    v4_ckpt_prompt_end_cap_cache = -1;   /* default cap: not the trigger */
    v4_ckpt_disk_budget_cache = 64;
    assert(v4_ckpt_catalog_add(ids, 100, 1, 0, "kv_gc_i1.bin") >= 0);
    assert(v4_ckpt_catalog_add(other, 100, 1, 0, "kv_gc_i2.bin") >= 0);
    v4_ckpt_gc();
    assert(v4_ckpt_cat_n == 0);

    v4_ckpt_catalog_clear();
    v4_ckpt_prompt_end_cap_cache = saved_cap;
    v4_ckpt_disk_budget_read = saved_read;
    v4_ckpt_disk_budget_cache = saved_budget;
}

/* T6: migration policy. v1 ckpt_* files are untouched and unmatched by
 * default; V4_PREFIX_CKPT_LEGACY_IMPORT=1 imports them byte-preserved
 * (payload stream-copied verbatim, ids from the v1 header) and removes the
 * v1 file only after the v2 rename; a foreign COLIV4C2 file always
 * survives, unmatched. */
static void test_cold_tier_migration(void)
{
    int minimum = v4_ckpt_min_tokens();
    int len = minimum + 5;
    int *ids = malloc((size_t)len * sizeof(int));
    assert(ids);
    for (int i = 0; i < len; i++) ids[i] = 4100 + i;

    ColiV4Engine engine;
    ColiV4Session session;
    ckpt_fake_session(&session, &engine, ids);
    char saved_dir[1024], scratch[300];
    ckpt_scratch_enter(saved_dir, sizeof(saved_dir), scratch);

    /* A v1 slot-mirror file: COLIV4CK + v1 fp + {1,kind,len,layers} + ids
     * + payload. The payload is the same crafted snapshot v3 would write. */
    uint8_t payload[CKPT_TEST_SNAPSHOT_BYTES];
    ckpt_craft_payload(payload);
    char v1path[1200];
    snprintf(v1path, sizeof(v1path), "%s/ckpt_%08x_%d.bin",
             v4_ckpt_dir, v4_ckpt_fingerprint_v1, 0);
    FILE *stream = fopen(v1path, "wb");
    assert(stream);
    static const char v1magic[9] = "COLIV4CK";
    uint32_t v1fp = v4_ckpt_fingerprint_v1;
    int32_t v1head[4] = {1, 1, len, 1};
    assert(fwrite(v1magic, 1, 8, stream) == 8);
    assert(fwrite(&v1fp, 4, 1, stream) == 1);
    assert(fwrite(v1head, 4, 4, stream) == 4);
    assert(fwrite(ids, 4, len, stream) == (size_t)len);
    assert(fwrite(payload, 1, sizeof(payload), stream) == sizeof(payload));
    assert(fclose(stream) == 0);

    /* A foreign fork file (COLIV4C2): must survive every scan, unmatched. */
    char c2path[1200];
    snprintf(c2path, sizeof(c2path), "%s/kv_%08x_deadbeefdeadbeef.bin",
             v4_ckpt_dir, v4_ckpt_fingerprint);
    stream = fopen(c2path, "wb");
    assert(stream);
    static const char c2magic[9] = "COLIV4C2";
    uint32_t fp = v4_ckpt_fingerprint;
    int32_t c2head[4] = {2, 1, 8, 1};
    int eight[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    assert(fwrite(c2magic, 1, 8, stream) == 8);
    assert(fwrite(&fp, 4, 1, stream) == 1);
    assert(fwrite(c2head, 4, 4, stream) == 4);
    assert(fwrite(eight, 4, 8, stream) == 8);
    assert(fclose(stream) == 0);

    /* Default: legacy left in place, foreign left in place, no matches. */
    v4_ckpt_disk_scanned = 0;
    v4_ckpt_disk_scan();
    assert(v4_ckpt_cat_n == 0);
    assert(test_file_exists(v1path));
    assert(test_file_exists(c2path));
    assert(!v4_ckpt_have(ids, len));
    assert(v4_ckpt_best_match(ids, len + 8) == 0);

    /* Opt-in import: v3 twin appears with the payload byte-preserved, the
     * v1 file disappears, the row restores. */
    v4_ckpt_legacy_import = 1;
    v4_ckpt_disk_scanned = 0;
    v4_ckpt_disk_scan();
    assert(v4_ckpt_cat_n == 1 && v4_ckpt_cat[0].len == len);
    assert(v4_ckpt_cat[0].kind == 1);
    assert(!test_file_exists(v1path));   /* only after the rename */
    assert(test_file_exists(c2path));    /* foreign still untouched */
    char v3path[1200];
    snprintf(v3path, sizeof(v3path), "%s/%s", v4_ckpt_dir,
             v4_ckpt_cat[0].file);
    stream = fopen(v3path, "rb");
    assert(stream);
    uint8_t payload_back[sizeof(payload)];
    assert(fseek(stream, 36 + (long)len * 4, SEEK_SET) == 0);
    assert(fread(payload_back, 1, sizeof(payload_back), stream) ==
           sizeof(payload_back));
    fclose(stream);
    assert(!memcmp(payload_back, payload, sizeof(payload)));  /* byte-identical */
    kv_prefix_record(&session.fed, ids, 0, len);
    assert(v4_ckpt_restore(&session, len + 8) == len);

    ckpt_fake_session_free(&session);
    free(ids);
    assert(remove(v3path) == 0);
    assert(remove(c2path) == 0);
    v4_ckpt_legacy_import = -1;
    ckpt_scratch_exit(saved_dir, scratch);
}

/* T7: per-session plan state under KV_SLOTS. The LCP probe reads THIS
 * session's previous prompt (a global would compare another slot's), while
 * the catalog itself is process-global: a row captured for one session is
 * visible to another session's best_match. */
static void test_cold_tier_kv_slots(void)
{
    int ids[128], other[128];
    for (int i = 0; i < 128; i++) {
        ids[i] = 6100 + i;
        other[i] = 7100 + i;
    }

    ColiV4Session a, b;
    ColiV4Engine engine;
    ckpt_fake_session(&a, &engine, ids);
    ckpt_fake_session(&b, &engine, other);

    int minimum = v4_ckpt_min_tokens();
    int count = minimum + 40;

    /* Session A saw prompt `ids`: its probe finds the LCP of a shared
     * prefix against ITS OWN record. */
    int plan = v4_ckpt_plan(&a, ids, count);
    assert(plan == 0);   /* first prompt: nothing to compare against */
    assert(a.ckpt_prev_ids && a.ckpt_prev_len == count);
    assert(!b.ckpt_prev_ids);   /* B's lineage state is untouched */

    /* A divergent prompt sharing the first `share` tokens: A's probe sees
     * the LCP, B's does not (no record). */
    int share = minimum + 8;
    int mixed[192];
    memcpy(mixed, ids, (size_t)share * sizeof(int));
    for (int i = share; i < count; i++) mixed[i] = 9100 + i;
    assert(v4_ckpt_plan_probe(&a, mixed, count) == share);
    assert(v4_ckpt_plan_probe(&b, mixed, count) == 0);

    /* B's own plan records against B's state only. */
    assert(v4_ckpt_plan(&b, other, count) == 0);
    assert(b.ckpt_prev_len == count && a.ckpt_prev_len == count);
    assert(memcmp(a.ckpt_prev_ids, b.ckpt_prev_ids,
                  (size_t)count * sizeof(int)) != 0);
    assert(v4_ckpt_plan_probe(&b, other, count) == 0);
    /* `count - 8` is the upper bound: an LCP longer than count-8 is
     * rejected (tiny-tail rule). */
    assert(v4_ckpt_plan_probe(&a, ids, count) == 0);

    /* The catalog is process-global: a row captured through A's session is
     * matched for B. */
    v4_ckpt_catalog_clear();
    assert(v4_ckpt_catalog_add(ids, share, 1, 4096, "kv_slots.bin") >= 0);
    assert(v4_ckpt_best_match(mixed, count) == share);   /* any session */
    assert(v4_ckpt_have(ids, share));
    v4_ckpt_catalog_clear();

    ckpt_fake_session_free(&a);
    ckpt_fake_session_free(&b);
}

int main(void)
{
    /* The env knobs cache on first use; set them before anything in this
     * binary evaluates them. DISK=2 keeps every store on the disk path
     * (prompt-ends included); the memory backend is exercised end to end by
     * the tiny-fixture suites with V4_PREFIX_CKPT_DISK=0. */
    test_setenv("V4_PREFIX_CKPT_MIN", "64");
    test_setenv("V4_PREFIX_CKPT_DISK", "2");
    test_submit_shapes_and_controls();
    test_errors_and_writer_bytes();
    test_extension_rejection_precedes_accept();
    test_cold_tier_roundtrip();   /* T1 + T2 + T5 */
    test_cold_tier_torn_files();  /* T3 */
    test_cold_tier_gc();          /* T4 */
    test_cold_tier_migration();   /* T6 */
    test_cold_tier_kv_slots();    /* T7 */
    puts("v4 serve framing baseline: ok");
    return 0;
}
