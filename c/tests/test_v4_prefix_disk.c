/* Self-contained CPU unit test for content-keyed DeepSeek V4 disk prefix checkpoints:
 *   1. Restart restore is bit-identical (including 0 RAM slots and tier identity).
 *   2. Flipped byte / truncated file is safely dropped and falls back cleanly;
 *      a SHA-valid file that fails restore is kept on disk.
 *   3. LRU trim removes only its own ck2_* files, sweeps old ckpt_* files,
 *      keeps unrelated foreign files, and respects V4_PREFIX_CKPT_DISK_MB. */
#define COLI_V4_UNIT_GENERATE_STATS
#define COLI_V4_SKIP_GENERATE_MAIN
#include "../deepseek_v4.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    TEST_LAYERS = 2,
    TEST_WINDOW = 8,
    TEST_HEAD_DIM = 16,
    TEST_COMPRESSED = 6,
    TEST_PROMPT_CAP = 64
};

static int g_checks;
static int g_failures;
static char g_ckpt_dir[1024];

static void check(int cond, const char *desc) {
    g_checks++;
    if (!cond) {
        g_failures++;
        fprintf(stderr, "FAIL: %s\n", desc);
    }
}

static void set_env_str(const char *name, const char *value) {
    if (value) setenv(name, value, 1);
    else unsetenv(name);
}

/* Populate an attention state with deterministic non-zero floats via the public
 * snapshot wire format so we never depend on private struct layouts. */
static int populate_attention_state(ColiDeepSeekV4WindowAttentionState *st, int seed) {
    FILE *f = tmpfile();
    if (!f) return -1;
    int32_t head[5] = {TEST_WINDOW, TEST_HEAD_DIM, TEST_COMPRESSED, 0, 0};
    float kv[TEST_WINDOW * TEST_HEAD_DIM];
    float comp[TEST_COMPRESSED * TEST_HEAD_DIM];
    for (int i = 0; i < TEST_WINDOW * TEST_HEAD_DIM; i++)
        kv[i] = (float)((seed * 31 + i * 7) % 97) * 0.125f - 3.0f;
    for (int i = 0; i < TEST_COMPRESSED * TEST_HEAD_DIM; i++)
        comp[i] = (float)((seed * 17 + i * 13) % 89) * 0.0625f - 2.0f;
    int ok = fwrite(head, sizeof(head), 1, f) == 1 &&
             fwrite(kv, sizeof(float), TEST_WINDOW * TEST_HEAD_DIM, f) ==
                 (size_t)(TEST_WINDOW * TEST_HEAD_DIM) &&
             fwrite(comp, sizeof(float), TEST_COMPRESSED * TEST_HEAD_DIM, f) ==
                 (size_t)(TEST_COMPRESSED * TEST_HEAD_DIM);
    rewind(f);
    ColiV4AttentionSnapshot *snap = NULL;
    ok = ok && !coli_v4_attention_snapshot_read(f, &snap) &&
         !coli_v4_attention_snapshot_restore(st, snap);
    coli_v4_attention_snapshot_destroy(snap);
    fclose(f);
    return ok ? 0 : -1;
}

static unsigned char *serialize_state(const ColiDeepSeekV4WindowAttentionState *st, size_t *out_len) {
    ColiV4AttentionSnapshot *snap = NULL;
    if (coli_v4_attention_snapshot_create(st, &snap)) return NULL;
    FILE *f = tmpfile();
    if (!f) { coli_v4_attention_snapshot_destroy(snap); return NULL; }
    int ok = !coli_v4_attention_snapshot_write(snap, f) && !fflush(f);
    coli_v4_attention_snapshot_destroy(snap);
    if (!ok) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    unsigned char *buf = malloc((size_t)sz);
    rewind(f);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

static int states_bit_identical(const ColiDeepSeekV4WindowAttentionState *a,
                                const ColiDeepSeekV4WindowAttentionState *b) {
    size_t la = 0, lb = 0;
    unsigned char *ba = serialize_state(a, &la);
    unsigned char *bb = serialize_state(b, &lb);
    int same = ba && bb && la == lb && memcmp(ba, bb, la) == 0;
    free(ba);
    free(bb);
    return same;
}

typedef struct {
    ColiSafetensorsIndex index;
    ColiV4Engine engine;
    ColiV4Session session;
    ColiDeepSeekV4WindowAttentionState *attentions[TEST_LAYERS];
    int prompt_ids[TEST_PROMPT_CAP];
} FakeFixture;

static FakeFixture *fixture_new(int seed, int zeroed) {
    FakeFixture *fx = calloc(1, sizeof(*fx));
    if (!fx) return NULL;
    fx->index.nfd = 2;
    fx->index.paths[0] = "/tmp/model/model-00001-of-00002.safetensors";
    fx->index.paths[1] = "/tmp/model/model-00002-of-00002.safetensors";
    fx->index.sizes[0] = 123456;
    fx->index.sizes[1] = 654321;

    fx->engine.owned_target_model_dir = "/tmp/model";
    fx->engine.target_index = &fx->index;
    fx->engine.gpu.enabled = 0;

    fx->session.engine = &fx->engine;
    fx->session.config.num_hidden_layers = TEST_LAYERS;
    fx->session.config.hidden_size = 64;
    fx->session.config.head_dim = TEST_HEAD_DIM;
    fx->session.config.sliding_window = TEST_WINDOW;
    fx->session.config.index_head_dim = 16;
    fx->session.config.index_topk = 4;
    fx->session.config.index_n_heads = 2;
    fx->session.config.hc_mult = 2;
    fx->session.attention = fx->attentions;
    fx->session.prompt_ids = fx->prompt_ids;

    for (int i = 0; i < TEST_PROMPT_CAP; i++)
        fx->prompt_ids[i] = 100 + i;
    for (int l = 0; l < TEST_LAYERS; l++) {
        check(!coli_v4_window_attention_create(&fx->attentions[l], &fx->session.config),
              "fixture: window_attention_create succeeded");
        if (!zeroed && fx->attentions[l])
            check(!populate_attention_state(fx->attentions[l], seed + l * 1000),
                  "fixture: populated attention state");
    }
    return fx;
}

static void fixture_free(FakeFixture *fx) {
    if (!fx) return;
    for (int l = 0; l < TEST_LAYERS; l++) {
        coli_v4_window_attention_destroy(fx->attentions[l]);
        fx->attentions[l] = NULL;
    }
    free(fx);
}

static void clear_slots(void) {
    for (int i = 0; i < V4_CKPT_MAX_SLOTS; i++)
        v4_ckpt_slot_free(&v4_ckpt_slots[i]);
    v4_ckpt_clock = 0;
}

static int dir_clear_all(const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) return 0;
    struct dirent *entry;
    char path[1200];
    int removed = 0;
    while ((entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
        if (!remove(path)) removed++;
    }
    closedir(dir);
    return removed;
}

static int count_ck2_files(const char *dir_path, int64_t *total_bytes) {
    DIR *dir = opendir(dir_path);
    int count = 0;
    if (total_bytes) *total_bytes = 0;
    if (!dir) return 0;
    struct dirent *entry;
    char path[1200];
    while ((entry = readdir(dir))) {
        if (v4_ckpt_engine_name(entry->d_name) != 1) continue;
        snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
        struct stat st;
        if (stat(path, &st) || !S_ISREG(st.st_mode)) continue;
        if (total_bytes) *total_bytes += (int64_t)st.st_size;
        count++;
    }
    closedir(dir);
    return count;
}

static int find_file_with_len(const char *dir_path, int len, char *out, size_t size) {
    DIR *dir = opendir(dir_path);
    if (!dir) return 0;
    int found = 0;
    struct dirent *entry;
    char path[1200];
    while (!found && (entry = readdir(dir))) {
        size_t n = strlen(entry->d_name);
        if (strncmp(entry->d_name, "ck2_", 4) || n < 4 || strcmp(entry->d_name + n - 4, ".bin"))
            continue;
        snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
        FILE *f = fopen(path, "rb");
        int32_t head[4] = {0};
        found = f && !fseek(f, 16, SEEK_SET) && fread(head, sizeof(head), 1, f) == 1 &&
                head[2] == len;
        if (f) fclose(f);
        if (found && out && size) snprintf(out, size, "%s", path);
    }
    closedir(dir);
    return found;
}

static int file_exists(const char *dir_path, const char *name) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", dir_path, name);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static int create_named_file(const char *dir_path, const char *name) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", dir_path, name);
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fputs("not a checkpoint payload", f);
    return fclose(f) == 0;
}

static int set_mtime(const char *path, time_t t) {
    struct utimbuf tb;
    tb.actime = t;
    tb.modtime = t;
    return utime(path, &tb);
}

/* how: 0 = flip 1 byte mid-file, 1 = truncate to half, 2 = corrupt layer window_size with valid SHA */
static int damage_file(const char *path, int len, int how) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size <= 64 + 4L * len) { fclose(f); return -1; }
    unsigned char *buf = malloc((size_t)size);
    fseek(f, 0, SEEK_SET);
    int ok = buf && fread(buf, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    if (ok && how == 0) {
        buf[size / 2] ^= 0x55;
    } else if (ok && how == 2) {
        /* Layer 0 snapshot header is 5 int32s right after V4_CKPT_IDS_AT + 4*len:
         * [window_size, head_dim, compressed_count, has_comp, has_idx]. Flip window_size. */
        buf[V4_CKPT_IDS_AT + 4 * len] ^= 1;
        EvidenceSha256 sha;
        evidence_sha256_init(&sha);
        evidence_sha256_update(&sha, buf, (size_t)(size - 32));
        evidence_sha256_final(&sha, buf + size - 32);
    }
    if (ok) {
        f = fopen(path, "wb");
        size_t keep = how == 1 ? (size_t)(size / 2) : (size_t)size;
        ok = f && fwrite(buf, 1, keep, f) == keep;
        if (f) ok = !fclose(f) && ok;
    }
    free(buf);
    return ok ? 0 : -1;
}

/* ---- 1. Restart restore is bit-identical ---- */
static void test_restart_bit_identical(void) {
    dir_clear_all(g_ckpt_dir);
    clear_slots();
    v4_ckpt_disk_forget();

    set_env_str("V4_PREFIX_CKPT_DIR", g_ckpt_dir);
    set_env_str("V4_PREFIX_CKPT_SLOTS", "0");
    set_env_str("V4_PREFIX_CKPT_DISK", "2");
    set_env_str("V4_PREFIX_CKPT_DISK_MB", "4096");
    set_env_str("COLI_VULKAN", "0");

    FakeFixture *src32 = fixture_new(11, 0);
    FakeFixture *src16 = fixture_new(22, 0);

    v4_ckpt_store(&src32->session, 32, 1);
    v4_ckpt_store(&src16->session, 16, 0);

    check(count_ck2_files(g_ckpt_dir, NULL) == 2,
          "restart: 0-slot store wrote two ck2 files");

    /* Simulate process/engine restart */
    v4_ckpt_disk_forget();
    clear_slots();

    FakeFixture *dst = fixture_new(0, 1);
    check(v4_ckpt_disk_on(&dst->session) != 0, "restart: disk enabled on new engine");

    char path[1200] = {0};
    int found = v4_ckpt_disk_find(&dst->session, 0, 40, path, sizeof(path));
    check(found == 32, "restart: disk_find selects longest matching prefix (32)");
    check(v4_ckpt_disk_restore(&dst->session, path, found) == 0,
          "restart: disk_restore succeeds");
    int identical = 1;
    for (int l = 0; l < TEST_LAYERS; l++)
        identical = identical && states_bit_identical(src32->attentions[l], dst->attentions[l]);
    check(identical, "restart: restored attention state is bit-identical across all layers");

    /* Different tier identity must not restore this file */
    dst->engine.gpu.enabled = 1;
    v4_ckpt_disk_forget();
    check(v4_ckpt_disk_on(&dst->session) != 0, "restart: disk enabled on gpu tier");
    check(v4_ckpt_disk_find(&dst->session, 0, 40, path, sizeof(path)) == 0,
          "restart: different tier identity does not match CPU checkpoint");
    dst->engine.gpu.enabled = 0;

    /* File outlives an evicted RAM slot (V4_PREFIX_CKPT_SLOTS=1) */
    dir_clear_all(g_ckpt_dir);
    clear_slots();
    v4_ckpt_disk_forget();
    set_env_str("V4_PREFIX_CKPT_SLOTS", "1");
    v4_ckpt_store(&src16->session, 16, 1);
    FakeFixture *other = fixture_new(33, 0);
    for (int i = 0; i < TEST_PROMPT_CAP; i++) other->prompt_ids[i] = 900 + i;
    v4_ckpt_store(&other->session, 20, 1);   /* evicts src16 from the 1 RAM slot */
    check(!v4_ckpt_have(src16->prompt_ids, 16), "restart: RAM slot was evicted");

    FakeFixture *dst2 = fixture_new(0, 1);
    found = v4_ckpt_disk_find(&dst2->session, 0, 40, path, sizeof(path));
    check(found == 16 && v4_ckpt_disk_restore(&dst2->session, path, found) == 0,
          "restart: evicted slot's file still restores from disk");
    identical = 1;
    for (int l = 0; l < TEST_LAYERS; l++)
        identical = identical && states_bit_identical(src16->attentions[l], dst2->attentions[l]);
    check(identical, "restart: evicted slot's disk restore is bit-identical");

    fixture_free(src32);
    fixture_free(src16);
    fixture_free(other);
    fixture_free(dst);
    fixture_free(dst2);
}

/* ---- 2. Flipped byte / truncated file dropped, clean fallback ---- */
static void test_corruption_and_fallback(void) {
    dir_clear_all(g_ckpt_dir);
    clear_slots();
    v4_ckpt_disk_forget();

    set_env_str("V4_PREFIX_CKPT_DIR", g_ckpt_dir);
    set_env_str("V4_PREFIX_CKPT_SLOTS", "0");
    set_env_str("V4_PREFIX_CKPT_DISK", "2");
    set_env_str("V4_PREFIX_CKPT_DISK_MB", "4096");

    FakeFixture *src24 = fixture_new(41, 0);
    FakeFixture *src12 = fixture_new(52, 0);
    FakeFixture *src20 = fixture_new(63, 0);
    FakeFixture *dst = fixture_new(0, 1);

    v4_ckpt_store(&src24->session, 24, 1);
    v4_ckpt_store(&src12->session, 12, 1);

    /* 2a. Flip 1 byte mid-file in len=24: dropped, falls back to len=12 */
    char p24[1200] = {0}, p12[1200] = {0}, path[1200] = {0};
    check(find_file_with_len(g_ckpt_dir, 24, p24, sizeof(p24)),
          "corrupt: len=24 file exists before flip");
    check(damage_file(p24, 24, 0) == 0, "corrupt: flipped 1 byte in len=24");

    int found = v4_ckpt_disk_find(&dst->session, 0, 40, path, sizeof(path));
    check(found == 24, "corrupt: disk_find sees len=24 before SHA check");
    check(v4_ckpt_disk_restore(&dst->session, path, found) == 1,
          "corrupt: flipped-byte file refused with rc=1");
    check(!find_file_with_len(g_ckpt_dir, 24, NULL, 0),
          "corrupt: flipped-byte file deleted from disk");

    found = v4_ckpt_disk_find(&dst->session, 0, found, path, sizeof(path));
    check(found == 12, "corrupt: fallback finds shorter len=12");
    check(v4_ckpt_disk_restore(&dst->session, path, found) == 0,
          "corrupt: shorter len=12 restores cleanly");
    int identical = 1;
    for (int l = 0; l < TEST_LAYERS; l++)
        identical = identical && states_bit_identical(src12->attentions[l], dst->attentions[l]);
    check(identical, "corrupt: fallback len=12 state is bit-identical");

    /* 2b. Truncate len=12 to half: dropped, falls back to 0 (full prefill) */
    check(find_file_with_len(g_ckpt_dir, 12, p12, sizeof(p12)),
          "corrupt: len=12 file exists before truncation");
    check(damage_file(p12, 12, 1) == 0, "corrupt: truncated len=12 to half");

    found = v4_ckpt_disk_find(&dst->session, 0, 40, path, sizeof(path));
    check(found == 12, "corrupt: disk_find sees truncated len=12 header");
    check(v4_ckpt_disk_restore(&dst->session, path, found) == 1,
          "corrupt: truncated file refused with rc=1");
    check(!find_file_with_len(g_ckpt_dir, 12, NULL, 0),
          "corrupt: truncated file deleted from disk");
    check(v4_ckpt_disk_find(&dst->session, 0, 40, path, sizeof(path)) == 0,
          "corrupt: no remaining files, clean fallback to full prefill");

    /* 2c. Valid SHA-256 trailer but session-side restore fails: file is KEPT */
    v4_ckpt_store(&src20->session, 20, 1);
    char p20[1200] = {0};
    check(find_file_with_len(g_ckpt_dir, 20, p20, sizeof(p20)),
          "corrupt: len=20 file exists before SHA-valid mismatch");
    check(damage_file(p20, 20, 2) == 0,
          "corrupt: mutated snapshot window_size behind valid SHA-256");
    check(v4_ckpt_disk_restore(&dst->session, p20, 20) == -1,
          "corrupt: SHA-valid restore failure returns -1");
    check(find_file_with_len(g_ckpt_dir, 20, NULL, 0),
          "corrupt: SHA-valid file that failed restore is kept on disk");

    fixture_free(src24);
    fixture_free(src12);
    fixture_free(src20);
    fixture_free(dst);
}

/* ---- 3. LRU trim respects MB cap and touches only engine's own files ---- */
static void test_lru_trim_and_own_names(void) {
    dir_clear_all(g_ckpt_dir);
    clear_slots();
    v4_ckpt_disk_forget();

    set_env_str("V4_PREFIX_CKPT_DIR", g_ckpt_dir);
    set_env_str("V4_PREFIX_CKPT_SLOTS", "0");
    set_env_str("V4_PREFIX_CKPT_DISK", "2");
    set_env_str("V4_PREFIX_CKPT_DISK_MB", "4096");

    FakeFixture *fx = fixture_new(77, 0);

    /* Measure one checkpoint file */
    v4_ckpt_store(&fx->session, 30, 1);
    int64_t one_bytes = 0;
    check(count_ck2_files(g_ckpt_dir, &one_bytes) == 1 && one_bytes > 0,
          "trim: baseline single checkpoint written");

    /* Cap at 2.5x one file => at most 2 files fit */
    char cap_buf[64];
    snprintf(cap_buf, sizeof(cap_buf), "%.9f", 2.5 * (double)one_bytes / 1048576.0);
    set_env_str("V4_PREFIX_CKPT_DISK_MB", cap_buf);

    time_t base_t = time(NULL) - 100;
    char p30[1200] = {0}, p29[1200] = {0}, p28[1200] = {0};
    find_file_with_len(g_ckpt_dir, 30, p30, sizeof(p30));
    set_mtime(p30, base_t);

    v4_ckpt_store(&fx->session, 29, 1);
    find_file_with_len(g_ckpt_dir, 29, p29, sizeof(p29));
    set_mtime(p29, base_t + 10);

    v4_ckpt_store(&fx->session, 28, 1);
    find_file_with_len(g_ckpt_dir, 28, p28, sizeof(p28));
    set_mtime(p28, base_t + 20);

    v4_ckpt_store(&fx->session, 27, 1);

    int64_t total_bytes = 0;
    int kept_ck2 = count_ck2_files(g_ckpt_dir, &total_bytes);
    check(kept_ck2 == 2 && (double)total_bytes <= 2.5 * (double)one_bytes,
          "trim: V4_PREFIX_CKPT_DISK_MB kept 2 files under cap");
    check(find_file_with_len(g_ckpt_dir, 27, NULL, 0) &&
              find_file_with_len(g_ckpt_dir, 28, NULL, 0) &&
              !find_file_with_len(g_ckpt_dir, 30, NULL, 0) &&
              !find_file_with_len(g_ckpt_dir, 29, NULL, 0),
          "trim: oldest ck2 files (30, 29) evicted, newest (28, 27) kept");

    /* Foreign files must survive even under an aggressive cap; old ckpt_* swept */
    static const char *const foreign[] = {
        "ckpt_epoch3.pt",
        "ckpt_hits.log",
        "ck2_notes.txt",
        "ckpt_0badf00d_0.bin.bak"
    };
    enum { N_FOREIGN = (int)(sizeof(foreign) / sizeof(foreign[0])) };
    for (int i = 0; i < N_FOREIGN; i++)
        check(create_named_file(g_ckpt_dir, foreign[i]), "trim: created foreign file");
    check(create_named_file(g_ckpt_dir, "ckpt_0badf00d_0.bin"),
          "trim: created legacy slot-indexed ckpt file");

    set_env_str("V4_PREFIX_CKPT_DISK_MB", "0.000001");
    v4_ckpt_store(&fx->session, 26, 1);

    int foreign_kept = 0;
    for (int i = 0; i < N_FOREIGN; i++)
        foreign_kept += file_exists(g_ckpt_dir, foreign[i]);
    check(foreign_kept == N_FOREIGN,
          "trim: all unrelated foreign files preserved");
    check(!file_exists(g_ckpt_dir, "ckpt_0badf00d_0.bin"),
          "trim: legacy ckpt_<8hex>_<n>.bin swept");
    check(find_file_with_len(g_ckpt_dir, 26, NULL, 0) &&
              !find_file_with_len(g_ckpt_dir, 27, NULL, 0) &&
              !find_file_with_len(g_ckpt_dir, 28, NULL, 0),
          "trim: older ck2 files evicted, newly written len=26 kept");

    fixture_free(fx);
}

int main(void) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || !*tmp) tmp = getenv("TEMP");
    if (!tmp || !*tmp) tmp = "/tmp";
    snprintf(g_ckpt_dir, sizeof(g_ckpt_dir), "%s/coli_v4_prefix_disk_test_%d",
             tmp, (int)v4_ckpt_pid());
    v4_ckpt_mkdir(g_ckpt_dir);

    test_restart_bit_identical();
    test_corruption_and_fallback();
    test_lru_trim_and_own_names();

    dir_clear_all(g_ckpt_dir);
    rmdir(g_ckpt_dir);
    clear_slots();

    printf("test_v4_prefix_disk: %d/%d checks %s\n",
           g_checks - g_failures, g_checks, g_failures ? "FAIL" : "PASS");
    return g_failures ? 1 : 0;
}
