/* DeepSeek V4's expert store reading its routed experts from a WEX4 container
 * (COLI_WEX_PATH, deepseek_v4.c v4_wex_*), CPU only, no model: a synthetic
 * checkpoint of 2 layers x 4 experts (40 x 2048 FP4 matrices, 130,560 bytes a
 * record, not a whole number of pages) and its container, written the way the
 * in-place upgrade of a WEX1 file writes one (tests/wex4_fixture.h), with one
 * expert left out of it.
 *
 * Every lookup must give the checkpoint's bytes, whichever source served it.
 * The container serves every record it holds; a pinned root that is the file's
 * keeps it, another root, a root that is not 64 hex digits, a damaged header
 * page or index, a truncated copy, a WEX1 file, another grid or a record of
 * other shapes leave the store on the safetensors; a record changed on disk is
 * served from the safetensors once its first-read SHA-256 fails (on with the
 * pin, or COLI_WEX_VERIFY=1) and only then; nonzero padding is never served.
 *
 * Then the same container with WNR3 records (tests/wnr3_fixture.h), plain and
 * canonical zero: every record it holds decoded to the checkpoint's bytes; a
 * record changed on disk refused by its stream check with the hash off and by
 * its hash with it on, a record under another expert's name refused by its
 * owner, both then served from the safetensors; a changed model blob, or a
 * canonical-zero container under DSV4_CUDA_MOE_MASKED=1, not used. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"
#include "../deepseek_v4_internal.h"
#include "wex4_fixture.h"
#include "wnr3_fixture.h"

int coli_v4_test_wex(ColiExpertStore *store, uint64_t counters[5]);

enum { LAYERS = 2, EXPERTS = 4, ROWS = 40, SCALE_COLS = 64, WEIGHT_COLS = 1024 };
#define CELLS (LAYERS * EXPERTS)
#define SCALE_M ((size_t)ROWS * SCALE_COLS)
#define WEIGHT_M ((size_t)ROWS * WEIGHT_COLS)
#define RAW (3 * (SCALE_M + WEIGHT_M))            /* 130,560 bytes */
#define STORED ((size_t)WEX_ALIGN_UP(RAW, WEX_ALIGN))
#define ABSENT 5                                  /* layer 1 expert 1: not in the container */
/* coli_v4_expert_store_read_private (a GPU build's) reads layer 0 expert 3 once more */
#if defined(COLI_VULKAN) || defined(COLI_CUDA)
#define PRIVATE 1
#else
#define PRIVATE 0
#endif
#define HELD (CELLS - 1 + PRIVATE)                /* the container's reads when it serves every record */

static unsigned char source[CELLS][RAW];
static char directory[] = "colibri-v4-wex4-XXXXXX", st_path[512], wex_path[512];
static int failures;

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

static void write_file(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    assert(f && fwrite(data, 1, bytes, f) == bytes && fclose(f) == 0);
}

/* model.safetensors: per expert its w1, w2, w3 scales then its w1, w2, w3 weights */
static void write_safetensors(void) {
    static const char *names[3] = {"w1", "w2", "w3"};
    char header[16384];
    size_t used = (size_t)snprintf(header, sizeof header, "{");
    for (int c = 0; c < CELLS; c++)
        for (int m = 0; m < 3; m++) {
            size_t s = (size_t)c * RAW + m * SCALE_M, w = (size_t)c * RAW + 3 * SCALE_M + m * WEIGHT_M;
            used += (size_t)snprintf(header + used, sizeof header - used,
                "%s\"layers.%d.ffn.experts.%d.%s.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]},"
                "\"layers.%d.ffn.experts.%d.%s.weight\":{\"dtype\":\"I8\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                used > 1 ? "," : "", c / EXPERTS, c % EXPERTS, names[m], ROWS, SCALE_COLS, s, s + SCALE_M,
                c / EXPERTS, c % EXPERTS, names[m], ROWS, WEIGHT_COLS, w, w + WEIGHT_M);
            assert(used < sizeof header);
        }
    header[used++] = '}';
    uint64_t length = used;
    FILE *f = fopen(st_path, "wb");
    assert(f && fwrite(&length, 8, 1, f) == 1 && fwrite(header, 1, used, f) == used);
    assert(fwrite(source, 1, sizeof source, f) == sizeof source && fclose(f) == 0);
}

/* A WEX1 file of the checkpoint's experts (all but ABSENT), records behind the table in cell
 * order. rows != ROWS writes entries of other shapes and the same sizes. */
static unsigned char *wex1(int layers, int rows, size_t *bytes, size_t offset[CELLS]) {
    size_t cells = (size_t)layers * EXPERTS, table = (size_t)WEX_ALIGN_UP(cells * sizeof(WexEntry), WEX_ALIGN);
    size_t total = WEX_ALIGN + table, at = total;
    for (size_t c = 0; c < cells; c++) if (c != ABSENT) total += STORED;
    unsigned char *out = calloc(1, total);
    assert(out);
    WexHeader *h = (WexHeader *)out;
    h->magic = WEX_MAGIC_V1; h->version = 1; h->num_layers = (uint32_t)layers; h->num_experts = EXPERTS;
    h->aligned_expert_bytes = STORED; h->table_offset = WEX_ALIGN;
    WexEntry *e = (WexEntry *)(out + WEX_ALIGN);
    for (size_t c = 0; c < cells; c++) {
        offset[c] = 0;
        if (c == ABSENT) continue;
        const uint32_t scale_cols = (uint32_t)(SCALE_M / rows);
        e[c].file_offset = at; e[c].total_bytes = (uint32_t)STORED;
        e[c].scale_bytes = (uint32_t)(3 * SCALE_M); e[c].weight_bytes = (uint32_t)(3 * WEIGHT_M);
        for (int m = 0; m < 3; m++) {
            e[c].scale_shapes[m][0] = e[c].weight_shapes[m][0] = (uint16_t)rows;
            e[c].scale_shapes[m][1] = (uint16_t)scale_cols; e[c].weight_shapes[m][1] = (uint16_t)(scale_cols * 16u);
            e[c].scale_offsets[m] = (uint32_t)(m * SCALE_M);
            e[c].weight_offsets[m] = (uint32_t)(3 * SCALE_M + m * WEIGHT_M);
        }
        memcpy(out + at, source[c], RAW);
        offset[c] = at;
        at += STORED;
    }
    *bytes = total;
    return out;
}

static int same_matrix(const ColiTensorView *v, size_t cell, int m) {
    return v->data_bytes == WEIGHT_M && v->scale_bytes == SCALE_M &&
           !memcmp(v->data, source[cell] + 3 * SCALE_M + m * WEIGHT_M, WEIGHT_M) &&
           !memcmp(v->scales, source[cell] + m * SCALE_M, SCALE_M);
}

/* Open the store, look every expert up, compare with the checkpoint; counters: reads,
 * bytes, fallbacks, refused of the container (all 0 without one). Returns whether the
 * store had a container. */
static int run(uint64_t counters[5]) {
    char error[256] = {0};
    ColiDeepSeekV4ExpertStoreOptions options = {directory, LAYERS, EXPERTS, (uint64_t)CELLS * 2 * RAW, -1, 0, 1, 0};
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof error)) {
        fprintf(stderr, "store open: %s\n", error);
        failures++;
        return -1;
    }
    for (int pass = 0; pass < 2; pass++)   /* the second pass: cache hits, and refused records again */
        for (size_t c = 0; c < CELLS; c++) {
            ColiExpertKey key = {(int)(c / EXPERTS), (int)(c % EXPERTS)};
            ColiExpertView view;
            memset(&view, 0, sizeof view);
            if (coli_expert_lookup(store, key, &view)) { fprintf(stderr, "lookup %zu failed\n", c); failures++; continue; }
            if (!same_matrix(&view.gate, c, 0) || !same_matrix(&view.down, c, 1) || !same_matrix(&view.up, c, 2)) {
                fprintf(stderr, "layer %d expert %d: not the checkpoint's bytes\n", key.layer, key.expert);
                failures++;
            }
            coli_expert_release(store, &view);
        }
#if defined(COLI_VULKAN) || defined(COLI_CUDA)
    {   /* the private read (a buffer of record + slack, as the GPU unit allocates it) */
        unsigned char *buffer = NULL;
        ColiExpertView view;
        assert(!posix_memalign((void **)&buffer, 4096, RAW + COLI_V4_EXPERT_SLAB_SLACK));
        CHECK(!coli_v4_expert_store_read_private(store, (ColiExpertKey){0, 3}, buffer, &view));
        CHECK(same_matrix(&view.gate, 3, 0) && same_matrix(&view.down, 3, 1) && same_matrix(&view.up, 3, 2));
        compat_aligned_free(buffer);
    }
#endif
    ColiExpertKey keys[CELLS];
    for (size_t c = 0; c < CELLS; c++) keys[c] = (ColiExpertKey){(int)(c / EXPERTS), (int)(c % EXPERTS)};
    (void)store->ops->prefetch(store, keys, CELLS);   /* advice only: must not fail or read */
    int active = coli_v4_test_wex(store, counters);
    store->ops->destroy(store);
    return active;
}

/* decoded: how many of the reads were WNR3 records decoded (0 for a RAW container) */
static void expect5(const char *what, int active, uint64_t reads, uint64_t fallbacks, uint64_t refused,
                    uint64_t decoded) {
    uint64_t c[5];
    int got = run(c);
    if (got != active || c[0] != reads || c[2] != fallbacks || c[3] != refused || c[4] != decoded) {
        fprintf(stderr, "FAIL %s: container %d (want %d), reads %llu (want %llu), fallbacks %llu (want %llu), "
                        "refused %llu (want %llu), decoded %llu (want %llu)\n", what, got, active,
                (unsigned long long)c[0], (unsigned long long)reads, (unsigned long long)c[2],
                (unsigned long long)fallbacks, (unsigned long long)c[3], (unsigned long long)refused,
                (unsigned long long)c[4], (unsigned long long)decoded);
        failures++;
    } else printf("ok   %s\n", what);
}
#define expect(what, active, reads, fallbacks, refused) expect5(what, active, reads, fallbacks, refused, 0)

int main(void) {
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    unsetenv("COLI_WEX_PATH"); unsetenv("COLI_WEX_ROOT"); unsetenv("COLI_WEX_VERIFY"); unsetenv("COLI_MODEL_MIRROR");
    setbuf(stdout, NULL);
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(st_path, sizeof st_path, "%s/model.safetensors", directory);
    snprintf(wex_path, sizeof wex_path, "%s/experts.wex4", directory);
    /* FP4 codes mostly small, as an expert's are (so that WNR3 records are smaller than RAW
     * ones), and no -0: the canonical-zero container then decodes to the same bytes */
    static const unsigned char magnitude[16] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7};
    uint32_t seed = 20261116u;
    for (size_t c = 0; c < CELLS; c++)
        for (size_t j = 0; j < RAW; j++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            unsigned lo = magnitude[seed & 15u], hi = magnitude[seed >> 4 & 15u];
            lo |= lo ? (seed >> 8 & 8u) : 0u; hi |= hi ? (seed >> 9 & 8u) : 0u;
            source[c][j] = (unsigned char)(j < 3 * SCALE_M ? 118u + seed % 8u : lo | hi << 4);
        }
    write_safetensors();

    size_t old_bytes, bytes, offset[CELLS], other_bytes;
    char root[65], other[65];
    unsigned char *old = wex1(LAYERS, ROWS, &old_bytes, offset);
    unsigned char *w4 = wex4_fixture_from_wex1(old, old_bytes, &bytes, root);
    uint64_t index_offset;
    memcpy(&index_offset, w4 + 32, 8);
    /* Counted reads: the 7 records the container holds, once each (the second pass hits the
     * cache), plus the private read of layer 0 expert 3 in a GPU build: HELD. */

    expect("no COLI_WEX_PATH: the safetensors, no container", 0, 0, 0, 0);
    write_file(wex_path, w4, bytes);
    setenv("COLI_WEX_PATH", wex_path, 1);
    expect("WEX4 container: the 7 records it holds read from it, the 8th from the safetensors", 1, HELD, 0, 0);
    setenv("COLI_WEX_ROOT", root, 1);
    expect("its root pinned (first-read SHA-256 on): the same", 1, HELD, 0, 0);

    w4[offset[2] + 1000] ^= 0x40u;                              /* a byte of record 2 */
    write_file(wex_path, w4, bytes);
    expect("record 2 changed on disk, pinned: refused at its first read, served from the safetensors",
           1, HELD - 1, 0, 1);
    unsetenv("COLI_WEX_ROOT");
    setenv("COLI_WEX_VERIFY", "1", 1);
    expect("the same with COLI_WEX_VERIFY=1 and no pin", 1, HELD - 1, 0, 1);
    unsetenv("COLI_WEX_VERIFY");
    w4[offset[2] + 1000] ^= 0x40u;

    w4[offset[3] + RAW + 7] = 1u;                               /* record 3's zero padding */
    write_file(wex_path, w4, bytes);
    expect("nonzero padding after record 3: never served (verify off)", 1, HELD - 1 - PRIVATE, 1 + PRIVATE, 0);
    w4[offset[3] + RAW + 7] = 0u;

    write_file(wex_path, w4, bytes);
    setenv("COLI_WEX_ROOT", "00", 1);
    expect("COLI_WEX_ROOT not 64 hex digits: no container", 0, 0, 0, 0);
    memset(other, 'a', 64); other[64] = 0;
    setenv("COLI_WEX_ROOT", other, 1);
    expect("another root pinned: no container", 0, 0, 0, 0);
    unsetenv("COLI_WEX_ROOT");

    w4[index_offset + 70] ^= 1u;
    write_file(wex_path, w4, bytes);
    expect("a bit of the index changed: no container", 0, 0, 0, 0);
    w4[index_offset + 70] ^= 1u;
    w4[100] ^= 1u;
    write_file(wex_path, w4, bytes);
    expect("a bit of the header page changed: no container", 0, 0, 0, 0);
    w4[100] ^= 1u;
    write_file(wex_path, w4, bytes - WEX_ALIGN);
    expect("a truncated copy: no container", 0, 0, 0, 0);
    write_file(wex_path, old, old_bytes);
    expect("the WEX1 form: no container", 0, 0, 0, 0);

    unsigned char *one = wex1(1, ROWS, &other_bytes, offset), *w4one = wex4_fixture_from_wex1(one, other_bytes, &bytes, other);
    write_file(wex_path, w4one, bytes);
    expect("a container of another grid (1 x 4): no container", 0, 0, 0, 0);
    free(one); free(w4one);
    unsigned char *shaped = wex1(LAYERS, ROWS * 2, &other_bytes, offset),
                  *w4shaped = wex4_fixture_from_wex1(shaped, other_bytes, &bytes, other);
    write_file(wex_path, w4shaped, bytes);
    expect("records of other shapes and the same sizes: no container", 0, 0, 0, 0);
    free(shaped); free(w4shaped);

    snprintf(wex_path, sizeof wex_path, "%s/missing.wex4", directory);
    setenv("COLI_WEX_PATH", wex_path, 1);
    expect("COLI_WEX_PATH names no file: no container", 0, 0, 0, 0);
    unsetenv("COLI_WEX_PATH");

    /* WNR3 records (codec 5), decoded on the CPU */
    snprintf(wex_path, sizeof wex_path, "%s/experts.wex4", directory);
    setenv("COLI_WEX_PATH", wex_path, 1);
    for (int cz = 0; cz < 2; cz++) {
        size_t nr3_bytes, at[CELLS];
        char nroot[65], line[200];
        unsigned char *nr3 = wnr3f_from_wex4(w4, bytes, cz, &nr3_bytes, nroot, at);
        CHECK(nr3 != NULL);
        if (!nr3) break;
        const char *form = cz ? "canonical-zero WNR3" : "WNR3";
        write_file(wex_path, nr3, nr3_bytes);
        snprintf(line, sizeof line, "%s container: the 7 records decoded from it to the checkpoint's bytes, the 8th "
                                    "from the safetensors", form);
        expect5(line, 1, HELD, 0, 0, HELD);
        setenv("COLI_WEX_ROOT", nroot, 1);
        snprintf(line, sizeof line, "%s, its root pinned (first-read SHA-256 on): the same", form);
        expect5(line, 1, HELD, 0, 0, HELD);
        unsetenv("COLI_WEX_ROOT");
        if (cz) {
            setenv("DSV4_CUDA_MOE_MASKED", "1", 1);
            expect("canonical-zero WNR3 with DSV4_CUDA_MOE_MASKED=1: no container", 0, 0, 0, 0);
            unsetenv("DSV4_CUDA_MOE_MASKED");
            free(nr3);
            continue;
        }
        nr3[at[2] + 3000] ^= 0x10u;                             /* a byte of record 2's streams */
        write_file(wex_path, nr3, nr3_bytes);
        expect5("WNR3 record 2 changed on disk, no hash check: refused by its stream check, from the safetensors",
                1, HELD - 1, 0, 1, HELD - 1);
        setenv("COLI_WEX_ROOT", nroot, 1);
        expect5("the same, pinned: refused by its SHA-256 at its first read", 1, HELD - 1, 0, 1, HELD - 1);
        unsetenv("COLI_WEX_ROOT");
        nr3[at[2] + 3000] ^= 0x10u;
        char renamed_root[65];
        unsigned char *renamed = malloc(nr3_bytes);
        assert(renamed);
        memcpy(renamed, nr3, nr3_bytes);
        CHECK(!wnr3f_rename(renamed, nr3_bytes, 0u << 16 | 2u, 0u << 16 | 3u, source[2], renamed_root));   /* says it is expert 3 */
        write_file(wex_path, renamed, nr3_bytes);
        setenv("COLI_WEX_ROOT", renamed_root, 1);
        expect5("a whole WNR3 record under another expert's name (its hash in the index matches): refused, from the "
                "safetensors", 1, HELD - 1, 0, 1, HELD - 1);
        unsetenv("COLI_WEX_ROOT");
        free(renamed);
        nr3[WEX4_PAGE + 100] ^= 1u;                             /* a byte of the WNM2 model blob */
        write_file(wex_path, nr3, nr3_bytes);
        expect("the WNM2 model blob changed: no container", 0, 0, 0, 0);
        nr3[WEX4_PAGE + 100] ^= 1u;
        free(nr3);
    }
    unsetenv("COLI_WEX_PATH");
    remove(wex_path); remove(st_path);
    {
        char usage[600];
        snprintf(usage, sizeof usage, "%s/.coli_usage", directory);
        remove(usage);
    }
    rmdir(directory);
    free(old); free(w4);
    if (failures) { fprintf(stderr, "test_v4_wex4: %d failures\n", failures); return 1; }
    puts("test_v4_wex4: the WEX4 container serves the records it holds with the checkpoint's bytes; a bad, foreign "
         "or unpinned file is not used; a changed record falls back once its hash fails");
    return 0;
}
