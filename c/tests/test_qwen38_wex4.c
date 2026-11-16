/* qwen38's MXFP4 routed experts from a WEX4 container (COLI_WEX_PATH, qwen38_core.h
 * q38_wex4_attach), no model: a synthetic grid of 2 layers x 3 experts, hidden 64 and
 * moe_intermediate 32 (records of 3,264 bytes, not a whole page), written as the in-place
 * upgrade of a WEX1 file writes it (tests/wex4_fixture.h) from records laid out as
 * experts-mxfp4/ lays them out (gate, down, up scales, then their e2m1 codes).
 *
 * In process: the container attaches without Q38_EXPERT_LOWBIT and with =mxfp4; every
 * expert the engine loads holds the record's bytes in the right matrix; a pinned root that
 * is the file's, and COLI_WEX_VERIFY=1, pass on an intact file and mark every record
 * checked. In a child process each (st.h maps a file once per process and descriptor, so
 * the mapped cases cannot share one): pinned and mapped (COLI_MAP_EXPERTS=1) passes; the
 * engine stops with exit 1 on another low-bit format asked for, a WEX1 file, a damaged
 * header page or index, a truncated copy, another grid, a missing expert, a record of
 * other shapes, another root, a root that is not 64 hex digits, and a record changed on
 * disk under the pin, read and mapped; with no check asked, the changed record is served
 * as it is (exit 4: the child sees the changed byte). Without COLI_WEX_PATH nothing
 * attaches. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */
#include "wex4_fixture.h"
#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#endif

enum { LAYERS = 2, EXPERTS = 3, H = 64, I = 32 };
#define CELLS (LAYERS * EXPERTS)
/* gate [I][H], down [H][I], up [I][H]: scale bytes per matrix, then code bytes */
#define SG ((size_t)I * (H / 32))
#define SD ((size_t)H * (I / 32))
#define CG ((size_t)I * (H / 2))
#define CD ((size_t)H * (I / 2))
#define RAW (2 * SG + SD + 2 * CG + CD)          /* 3,264 bytes */
#define STORED ((size_t)WEX_ALIGN_UP(RAW, WEX_ALIGN))

static unsigned char source[CELLS][RAW];
static char directory[] = "colibri-q38-wex4-XXXXXX", wex_path[600];
static int fails;
static void ck(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); fails += !ok; }

/* A WEX1 file of the grid's records (cell `skip` left out when >= 0), behind the table in cell
 * order; other_shapes halves every matrix's rows and doubles its columns (same sizes). */
static unsigned char *wex1(int layers, int skip, int other_shapes, size_t *bytes, size_t offset[CELLS]) {
    const size_t cells = (size_t)layers * EXPERTS, table = (size_t)WEX_ALIGN_UP(cells * sizeof(WexEntry), WEX_ALIGN);
    size_t total = WEX_ALIGN + table, at = total;
    for (size_t c = 0; c < cells; c++) if ((int)c != skip) total += STORED;
    unsigned char *out = calloc(1, total);
    assert(out);
    WexHeader *hd = (WexHeader *)out;
    hd->magic = WEX_MAGIC_V1; hd->version = 1; hd->num_layers = (uint32_t)layers; hd->num_experts = EXPERTS;
    hd->aligned_expert_bytes = STORED; hd->table_offset = WEX_ALIGN;
    WexEntry *e = (WexEntry *)(out + WEX_ALIGN);
    const uint32_t rows[3] = {I, H, I}, scols[3] = {H / 32, I / 32, H / 32};   /* w1 gate, w2 down, w3 up */
    for (size_t c = 0; c < cells; c++) {
        offset[c] = 0;
        if ((int)c == skip) continue;
        e[c].file_offset = at; e[c].total_bytes = (uint32_t)STORED;
        uint32_t s = 0, w = 0;
        for (int m = 0; m < 3; m++) {
            uint32_t r = other_shapes ? rows[m] / 2 : rows[m], sc = other_shapes ? scols[m] * 2 : scols[m];
            e[c].scale_shapes[m][0] = e[c].weight_shapes[m][0] = (uint16_t)r;
            e[c].scale_shapes[m][1] = (uint16_t)sc; e[c].weight_shapes[m][1] = (uint16_t)(sc * 16u);
            e[c].scale_offsets[m] = s; s += r * sc;
        }
        e[c].scale_bytes = s;
        for (int m = 0; m < 3; m++) { e[c].weight_offsets[m] = s + w; w += e[c].weight_shapes[m][0] * e[c].weight_shapes[m][1]; }
        e[c].weight_bytes = w;
        assert(s + w == RAW);
        memcpy(out + at, source[c], RAW);
        offset[c] = at;
        at += STORED;
    }
    *bytes = total;
    return out;
}

static void put(const unsigned char *data, size_t bytes) {
    FILE *f = fopen(wex_path, "wb");
    assert(f && fwrite(data, 1, bytes, f) == bytes && fclose(f) == 0);
}

static void grid(Model *m) {
    memset(m, 0, sizeof *m);
    m->c.layers = LAYERS; m->c.experts = EXPERTS; m->c.hidden = H; m->c.inter = I;
    m->range_begin = 0; m->range_end = LAYERS;
}

/* every expert loaded through q38_load_int4_record holds its record in the right matrix */
static int load_all(Model *m) {
    int ok = m->x4 != NULL;
    for (int c = 0; c < CELLS && ok; c++) {
        Slot slot;
        memset(&slot, 0, sizeof slot);
        q38_load_int4_record(m, c / EXPERTS, c % EXPERTS, &slot);
        const unsigned char *r = source[c];
        const size_t code = 2 * SG + SD;
        ok = slot.gate.kind == Q38_WEIGHT_MXFP4 && slot.up.kind == Q38_WEIGHT_MXFP4 && slot.down.kind == Q38_WEIGHT_MXFP4 &&
             slot.gate.rows == I && slot.gate.cols == H && slot.down.rows == H && slot.down.cols == I &&
             !memcmp(slot.gate.x2q, r, SG) && !memcmp(slot.down.x2q, r + SG, SD) && !memcmp(slot.up.x2q, r + SG + SD, SG) &&
             !memcmp(slot.gate.data, r + code, CG) && !memcmp(slot.down.data, r + code + CG, CD) &&
             !memcmp(slot.up.data, r + code + CG + CD, CG);
        free(slot.int4_slab);
    }
    return ok;
}

static void drop(Model *m) {   /* the test's Model holds only the grid and m->x4 */
    if (m->x4) {
        st_destroy(&m->x4->S); free(m->x4->fd); free(m->x4->off);
        free(m->x4->wex); free(m->x4->wex_sha); free(m->x4->wex_seen); free(m->x4);
    }
    m->x4 = NULL;
}

/* attach and load every expert: 0 when every record is right (and, with the hash check on,
 * every one checked), 3 when nothing attached, 4 when a record's bytes are not the source's */
static int attach_and_load(void) {
    Model m;
    grid(&m);
    q38_expert_int4_attach(&m, directory);
    if (!m.x4) return 3;
    int rc = load_all(&m) ? 0 : 4;
    for (int c = 0; !rc && m.x4->wex_sha && c < CELLS; c++) if (m.x4->wex_seen[c] != 1) rc = 5;
    drop(&m);
    return rc;
}

static int spawn(const char *self) {
    fflush(stdout); fflush(stderr);
#ifdef _WIN32
    return (int)_spawnl(_P_WAIT, self, self, "child", NULL);
#else
    pid_t pid = fork();
    if (pid == 0) { execl(self, self, "child", (char *)NULL); _exit(127); }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

static void child_case(const char *self, int want, const char *what) {
    char line[300];
    int rc = spawn(self);
    snprintf(line, sizeof line, "%s (exit %d)", what, rc);
    ck(rc == want, line);
}

static void fill_source(void) {
    uint32_t seed = 20261116u;
    for (int c = 0; c < CELLS; c++)
        for (size_t j = 0; j < RAW; j++) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; source[c][j] = (unsigned char)seed; }
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "child")) {   /* one case, in its own process */
        const char *dir = getenv("Q38_WEX4_TEST_DIR");
        if (!dir) return 2;
        snprintf(directory, sizeof directory, "%s", dir);
        fill_source();
        return attach_and_load();
    }
    setbuf(stdout, NULL);
    unsetenv("COLI_WEX_PATH"); unsetenv("COLI_WEX_ROOT"); unsetenv("COLI_WEX_VERIFY"); unsetenv("Q38_EXPERT_LOWBIT");
    unsetenv("Q38_EXPERT_INT2"); unsetenv("Q38_EXPERT_INT4"); unsetenv("COLI_MAP_EXPERTS"); unsetenv("COLI_MODEL_MIRROR");
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(wex_path, sizeof wex_path, "%s/experts.wex4", directory);
    setenv("Q38_WEX4_TEST_DIR", directory, 1);
    fill_source();

    size_t old_bytes, w4_bytes, bytes, other_bytes, offset[CELLS], scratch[CELLS];
    char root[65], other[65];
    unsigned char *old = wex1(LAYERS, -1, 0, &old_bytes, offset), *w4 = wex4_fixture_from_wex1(old, old_bytes, &w4_bytes, root);
    uint64_t index_offset;
    memcpy(&index_offset, w4 + 32, 8);
    put(w4, w4_bytes);

    printf("in process\n");
    Model m;
    grid(&m); q38_expert_int4_attach(&m, directory);
    ck(m.x4 == NULL, "without COLI_WEX_PATH (and no sidecar) nothing attaches");
    setenv("COLI_WEX_PATH", wex_path, 1);
    grid(&m); q38_expert_int4_attach(&m, directory);
    ck(m.x4 && m.x4->lb && m.x4->lb->mx && m.x4->record_bytes == (int64_t)RAW && !m.x4->wex_sha,
       "COLI_WEX_PATH alone: the container attaches as the MXFP4 experts, records of 3,264 bytes, no hash check");
    ck(load_all(&m), "every expert loaded holds its record: gate, down and up scales and codes where they belong");
    drop(&m);
    setenv("Q38_EXPERT_LOWBIT", "mxfp4", 1);
    ck(attach_and_load() == 0, "with Q38_EXPERT_LOWBIT=mxfp4: the same");
    unsetenv("Q38_EXPERT_LOWBIT");
    setenv("COLI_WEX_ROOT", root, 1);
    grid(&m); q38_expert_int4_attach(&m, directory);
    ck(m.x4 && m.x4->wex_sha, "its root pinned: the hash check is on");
    drop(&m);
    ck(attach_and_load() == 0, "pinned: every record loads and passes its hash at its first read");
    unsetenv("COLI_WEX_ROOT");
    setenv("COLI_WEX_VERIFY", "1", 1);
    ck(attach_and_load() == 0, "COLI_WEX_VERIFY=1 without a pin: the same");
    unsetenv("COLI_WEX_VERIFY");

    printf("one child process each (the engine stops with exit 1 on a refusal)\n");
    setenv("COLI_WEX_ROOT", root, 1); setenv("COLI_MAP_EXPERTS", "1", 1);
    child_case(argv[0], 0, "pinned and mapped (COLI_MAP_EXPERTS=1): every mapped record checked and right");
    unsetenv("COLI_MAP_EXPERTS"); unsetenv("COLI_WEX_ROOT");
    setenv("Q38_EXPERT_LOWBIT", "int2g32", 1);
    child_case(argv[0], 1, "Q38_EXPERT_LOWBIT=int2g32 with a container: refused");
    unsetenv("Q38_EXPERT_LOWBIT");
    put(old, old_bytes);
    child_case(argv[0], 1, "the WEX1 form: refused");
    w4[100] ^= 1u; put(w4, w4_bytes); w4[100] ^= 1u;
    child_case(argv[0], 1, "a bit of the header page changed: refused");
    w4[index_offset + 70] ^= 1u; put(w4, w4_bytes); w4[index_offset + 70] ^= 1u;
    child_case(argv[0], 1, "a bit of the index changed: refused");
    put(w4, w4_bytes - WEX_ALIGN);
    child_case(argv[0], 1, "a truncated copy: refused");
    unsigned char *one = wex1(1, -1, 0, &other_bytes, scratch), *w4x = wex4_fixture_from_wex1(one, other_bytes, &bytes, other);
    put(w4x, bytes); free(one); free(w4x);
    child_case(argv[0], 1, "a container of another grid (1 x 3): refused");
    one = wex1(LAYERS, 4, 0, &other_bytes, scratch); w4x = wex4_fixture_from_wex1(one, other_bytes, &bytes, other);
    put(w4x, bytes); free(one); free(w4x);
    child_case(argv[0], 1, "layer 1 expert 1 missing from the container: refused");
    one = wex1(LAYERS, -1, 1, &other_bytes, scratch); w4x = wex4_fixture_from_wex1(one, other_bytes, &bytes, other);
    put(w4x, bytes); free(one); free(w4x);
    child_case(argv[0], 1, "records of other shapes and the same size: refused");
    put(w4, w4_bytes);
    memset(other, 'b', 64); other[64] = 0;
    setenv("COLI_WEX_ROOT", other, 1);
    child_case(argv[0], 1, "another root pinned: refused");
    setenv("COLI_WEX_ROOT", "12ab", 1);
    child_case(argv[0], 1, "COLI_WEX_ROOT not 64 hex digits: refused");
    setenv("COLI_WEX_ROOT", root, 1);
    w4[offset[4] + 100] ^= 0x10u; put(w4, w4_bytes);
    child_case(argv[0], 1, "layer 1 expert 1 changed on disk, its root pinned: refused at its first read");
    setenv("COLI_MAP_EXPERTS", "1", 1);
    child_case(argv[0], 1, "the same, mapped: refused at its first use");
    unsetenv("COLI_MAP_EXPERTS"); unsetenv("COLI_WEX_ROOT");
    child_case(argv[0], 4, "the changed record with no pin and no COLI_WEX_VERIFY: served as it is (no check was asked)");
    w4[offset[4] + 100] ^= 0x10u;

    unsetenv("COLI_WEX_PATH");
    remove(wex_path); rmdir(directory);
    free(old); free(w4);
    if (fails) { printf("test_qwen38_wex4: %d failure(s)\n", fails); return 1; }
    printf("test_qwen38_wex4: ok\n");
    return 0;
}
