/* glm53's MXFP4 routed experts, from the shards of an AMD Quark checkpoint and from a WEX4
 * container (COLI_WEX_PATH, glm53.c g53_wex4_attach), no model: a synthetic grid of 3 layers (layer
 * 0 dense, without experts) x 3 experts, hidden 256 and moe_intermediate 64 (records of 26,112
 * bytes, not a whole number of pages). The shards hold each expert as the checkpoint does
 * (`gate_proj.weight` U8 e2m1 codes, `gate_proj.weight_scale` U8 ue8m0, the same for up and down);
 * the container holds the same bytes as one record per expert in the slab order of Wex-WRN2's
 * `pack --names qwen38` (gate, down, up scales, then their codes), RAW as the in-place upgrade of a
 * WEX1 file writes it (tests/wex4_fixture.h), and WNR3 (tests/wnr3_fixture.h), plain and canonical
 * zero.
 *
 * In process: the MXFP4 form is found from the shards; every expert read from them holds its
 * bytes in the right matrix (fmt 7), and mv on it equals a scalar dequantization; with
 * COLI_WEX_PATH every expert read from the container holds the same bytes and gives the same
 * products bit for bit, RAW and WNR3; a pinned root and COLI_WEX_VERIFY=1 mark every record
 * checked; with the fake CUDA helper (argv[1], COLI_WEX_CUDA_LIB) the WNR3 records are decoded by
 * it and counted as its decodes. In a child process each: the engine stops with exit 1 on an int4
 * checkpoint asked to take a container, a WEX1 file, a damaged header page or index, another grid,
 * a missing expert, a record of other shapes, another root, a root that is not 64 hex digits, a
 * record changed on disk under the pin (RAW and WNR3), a WNR3 record changed with no pin (its
 * stream check), a whole record under another expert's name and a changed model blob; with no
 * check asked a changed RAW record is served as it is (exit 4). */
#define _GNU_SOURCE
#define GLM53_NO_MAIN
#include "../glm53.c"
#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */
#include "wex4_fixture.h"
#include "wnr3_fixture.h"
#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#endif

enum { LAYERS = 3, FIRST = 1, EXPERTS = 3, H = 256, I = 64 };
#define CELLS (LAYERS * EXPERTS)
/* the slab: gate [I][H], down [H][I], up [I][H]: scale bytes per matrix, then code bytes */
#define SG ((size_t)I * (H / 32))
#define SD ((size_t)H * (I / 32))
#define CG ((size_t)I * (H / 2))
#define CD ((size_t)H * (I / 2))
#define RAW (2 * SG + SD + 2 * CG + CD)          /* 26,112 bytes */
#define STORED ((size_t)WEX_ALIGN_UP(RAW, WEX_ALIGN))

static unsigned char source[CELLS][RAW];
static char directory[] = "colibri-g53-wex4-XXXXXX", wex_path[600], st_path[600];
static int fails;
static void ck(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); fails += !ok; }

static int sparse(int c) { return c / EXPERTS >= FIRST; }

/* scales near 2^-7, e2m1 codes mostly small and no -0, as in test_qwen38_wex4 */
static void fill_source(void) {
    static const unsigned char magnitude[16] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7};
    uint32_t seed = 20261116u;
    for (int c = 0; c < CELLS; c++)
        for (size_t j = 0; j < RAW; j++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            unsigned lo = magnitude[seed & 15u], hi = magnitude[seed >> 4 & 15u];
            lo |= lo ? (seed >> 8 & 8u) : 0u; hi |= hi ? (seed >> 9 & 8u) : 0u;
            source[c][j] = (unsigned char)(j < 2 * SG + SD ? 118u + seed % 8u : lo | hi << 4);
        }
}

/* The checkpoint's form: one safetensors file with the six U8 tensors of every sparse-layer expert. */
static void write_shards(void) {
    static const char *mat[3] = {"gate", "down", "up"};
    const size_t sbytes[3] = {SG, SD, SG}, cbytes[3] = {CG, CD, CG};
    const int rows[3] = {I, H, I}, cols[3] = {H, I, H};
    size_t soff[3] = {0, SG, SG + SD}, coff[3] = {2 * SG + SD, 2 * SG + SD + CG, 2 * SG + SD + CG + CD};
    char *json = malloc(1 << 16);
    size_t n = 0, data = 0;
    assert(json);
    n += (size_t)sprintf(json + n, "{");
    for (int c = 0; c < CELLS; c++) {
        if (!sparse(c)) continue;
        for (int k = 0; k < 3; k++) {
            n += (size_t)sprintf(json + n, "%s\"model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight\":"
                                 "{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                                 n > 1 ? "," : "", c / EXPERTS, c % EXPERTS, mat[k], rows[k], cols[k] / 2, data, data + cbytes[k]);
            data += cbytes[k];
            n += (size_t)sprintf(json + n, ",\"model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight_scale\":"
                                 "{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                                 c / EXPERTS, c % EXPERTS, mat[k], rows[k], cols[k] / 32, data, data + sbytes[k]);
            data += sbytes[k];
        }
    }
    n += (size_t)sprintf(json + n, "}");
    while (n % 8) json[n++] = ' ';
    FILE *f = fopen(st_path, "wb");
    uint64_t len = n;
    assert(f && fwrite(&len, 8, 1, f) == 1 && fwrite(json, 1, n, f) == n);
    for (int c = 0; c < CELLS; c++) {
        if (!sparse(c)) continue;
        for (int k = 0; k < 3; k++) {
            assert(fwrite(source[c] + coff[k], 1, cbytes[k], f) == cbytes[k]);
            assert(fwrite(source[c] + soff[k], 1, sbytes[k], f) == sbytes[k]);
        }
    }
    assert(fclose(f) == 0);
    free(json);
}

/* A WEX1 file of the sparse cells' records (cell `skip` left out when >= 0), behind the table in cell
 * order; other_shapes halves every matrix's rows and doubles its columns (same sizes). */
static unsigned char *wex1(int layers, int skip, int other_shapes, size_t *bytes, size_t offset[CELLS]) {
    const size_t cells = (size_t)layers * EXPERTS, table = (size_t)WEX_ALIGN_UP(cells * sizeof(WexEntry), WEX_ALIGN);
    size_t total = WEX_ALIGN + table, at = total;
    for (size_t c = 0; c < cells; c++) if ((int)c != skip && sparse((int)c)) total += STORED;
    unsigned char *out = calloc(1, total);
    assert(out);
    WexHeader *hd = (WexHeader *)out;
    hd->magic = WEX_MAGIC_V1; hd->version = 1; hd->num_layers = (uint32_t)layers; hd->num_experts = EXPERTS;
    hd->aligned_expert_bytes = STORED; hd->table_offset = WEX_ALIGN;
    WexEntry *e = (WexEntry *)(out + WEX_ALIGN);
    const uint32_t rows[3] = {I, H, I}, scols[3] = {H / 32, I / 32, H / 32};   /* gate, down, up */
    for (size_t c = 0; c < cells; c++) {
        offset[c] = 0;
        if ((int)c == skip || !sparse((int)c)) continue;
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

/* the grid, the shards, the MXFP4 form, the table of the shards' experts and (COLI_WEX_PATH) the container */
static void open_model(GModel *m) {
    memset(m, 0, sizeof *m);
    m->c.n_layers = LAYERS; m->c.n_experts = EXPERTS; m->c.hidden = H; m->c.moe_inter = I; m->c.first_dense = FIRST;
    m->layer_begin = 0; m->layer_end = LAYERS;
    st_init(&m->S, directory);
    snprintf(m->prefix, sizeof m->prefix, "model.language_model.");
    char first[200];
    snprintf(first, sizeof first, "%slayers.%d.mlp.experts.0.gate_proj.weight", m->prefix, FIRST);
    st_tensor *probe = st_find(&m->S, first);
    assert(probe);
    m->streaming = g53_expert_form(m, probe, first);
    expert_geometry(m);
    expert_table_init(m);
    if (getenv("G53_TEST_INT4") && getenv("COLI_WEX_PATH")) m->mx = 0;   /* as an int4 container's experts would be */
    g53_wex_maybe_attach(m);
}

static void close_model(GModel *m) {
    if (m->wex) {
        st_destroy(&m->wex->S); free(m->wex->off); free(m->wex->sha); free(m->wex->seen); free(m->wex->nr3);
        free(m->wex->path); free(m->wex);
    }
    st_destroy(&m->S); free(m->eref);
    m->wex = NULL; m->eref = NULL;
}

/* f32 products of the three matrices with a fixed input, and the scalar dequantized gate row 0 */
static void products(const Mat *g, const Mat *u, const Mat *d, float *out) {
    float x[H], y[I];
    for (int i = 0; i < H; i++) x[i] = (float)((i * 37) % 17 - 8) / 16.0f;
    mv(out, g, x); mv(out + I, u, x);
    for (int i = 0; i < I; i++) y[i] = out[i] * 0.5f - out[I + i] * 0.25f;
    mv(out + 2 * I, d, y);
}

/* every sparse expert read through expert_read holds its record's bytes in the right matrix; the
 * products (into prod[cell]) and a scalar check of gate row 0 */
static int load_all(GModel *m, float prod[CELLS][2 * I + H]) {
    int ok = 1;
    for (int c = 0; c < CELLS && ok; c++) {
        if (!sparse(c)) continue;
        Slot slot;
        memset(&slot, 0, sizeof slot);
        expert_read(m, c / EXPERTS, c % EXPERTS, &slot);
        Mat g, u, d;
        expert_mats(m, c / EXPERTS, &slot, &g, &u, &d);
        const unsigned char *r = source[c];
        const size_t code = 2 * SG + SD;
        ok = g.fmt == G53_MX && u.fmt == G53_MX && d.fmt == G53_MX && g.rows == I && g.columns == H && d.rows == H &&
             d.columns == I && !memcmp(g.e8, r, SG) && !memcmp(d.e8, r + SG, SD) && !memcmp(u.e8, r + SG + SD, SG) &&
             !memcmp(g.q4, r + code, CG) && !memcmp(d.q4, r + code + CG, CD) && !memcmp(u.q4, r + code + CG + CD, CG);
        products(&g, &u, &d, prod[c]);
        double ref = 0;
        float x[H];
        for (int i = 0; i < H; i++) x[i] = (float)((i * 37) % 17 - 8) / 16.0f;
        for (int col = 0; col < H; col++) {
            const uint8_t b = g.q4[col / 2];
            ref += (double)x[col] * mx4_lut[col & 1 ? b >> 4 : b & 15] * mx4_scale(g.e8[col / 32]);
        }
        ok = ok && fabs(ref - prod[c][0]) <= 1e-4 * (1.0 + fabs(ref));
        g53_slot_free(&slot);
    }
    return ok;
}

static float shard_prod[CELLS][2 * I + H];

/* attach and load every expert: 0 when every record is right (and, with the hash check on, every
 * one checked), 3 when nothing attached, 4 when a record's bytes are not the source's, 5 a record
 * not hashed, 6 a WNR3 record not decoded, 7 the products differ from the shards' */
static int attach_and_load(void) {
    static float prod[CELLS][2 * I + H];
    GModel m;
    open_model(&m);
    if (!m.wex) { close_model(&m); return 3; }
    int rc = load_all(&m, prod) ? 0 : 4;
    for (int c = 0; !rc && m.wex->sha && c < CELLS; c++) if (sparse(c) && m.wex->seen[c] != 1) rc = 5;
    if (!rc && m.wex->nr3 && m.wex->decoded != (uint64_t)(CELLS - EXPERTS * FIRST)) rc = 6;
    /* COLI_WEX_CUDA_LIB (a child: the helper is looked for once per process): every one by the helper */
    if (!rc && m.wex->nr3 && getenv("COLI_WEX_CUDA_LIB") && coli_wex_cuda_decode_count() != m.wex->decoded) rc = 9;
    if (!rc && memcmp(prod + EXPERTS * FIRST, shard_prod + EXPERTS * FIRST, sizeof prod - sizeof prod[0] * EXPERTS * FIRST)) rc = 7;
    close_model(&m);
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

static int shards_products(void) {   /* the shards' own experts: products into shard_prod */
    char saved[600] = "";
    const char *path = getenv("COLI_WEX_PATH");
    if (path) { snprintf(saved, sizeof saved, "%s", path); unsetenv("COLI_WEX_PATH"); }
    GModel m;
    open_model(&m);
    int ok = !m.wex && m.mx && m.streaming && load_all(&m, shard_prod);
    close_model(&m);
    if (*saved) setenv("COLI_WEX_PATH", saved, 1);
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "child")) {   /* one case, in its own process */
        const char *dir = getenv("G53_WEX4_TEST_DIR");
        if (!dir) return 2;
        snprintf(directory, sizeof directory, "%s", dir);
        fill_source();
        if (!shards_products()) return 8;
        return attach_and_load();
    }
    setbuf(stdout, NULL);
    unsetenv("COLI_WEX_PATH"); unsetenv("COLI_WEX_ROOT"); unsetenv("COLI_WEX_VERIFY"); unsetenv("COLI_WEX_CUDA_LIB");
    unsetenv("COLI_MODEL_MIRROR"); unsetenv("G53_TEST_INT4");
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(wex_path, sizeof wex_path, "%s/experts.wex4", directory);
    snprintf(st_path, sizeof st_path, "%s/model.safetensors", directory);
    setenv("G53_WEX4_TEST_DIR", directory, 1);
    fill_source();
    write_shards();

    printf("the shards (an AMD Quark MXFP4 checkpoint's experts)\n");
    ck(shards_products(), "MXFP4 found; every expert read from the shards holds its bytes as fmt 7, mv equals the scalar dequantization");

    size_t old_bytes, w4_bytes, bytes, other_bytes, offset[CELLS], scratch[CELLS];
    char root[65], other[65];
    unsigned char *old = wex1(LAYERS, -1, 0, &old_bytes, offset), *w4 = wex4_fixture_from_wex1(old, old_bytes, &w4_bytes, root);
    uint64_t index_offset;
    memcpy(&index_offset, w4 + 32, 8);
    put(w4, w4_bytes);

    printf("RAW records, in process\n");
    GModel m;
    setenv("COLI_WEX_PATH", wex_path, 1);
    open_model(&m);
    ck(m.wex && !m.wex->sha && !m.wex->nr3, "COLI_WEX_PATH: the container attaches, no hash check, no WNR3");
    close_model(&m);
    ck(attach_and_load() == 0, "every expert read from the container holds the shards' bytes, the products bit for bit the same");
    setenv("COLI_WEX_ROOT", root, 1);
    ck(attach_and_load() == 0, "its root pinned: every record passes its hash at its first read");
    unsetenv("COLI_WEX_ROOT");
    setenv("COLI_WEX_VERIFY", "1", 1);
    ck(attach_and_load() == 0, "COLI_WEX_VERIFY=1 without a pin: the same");
    unsetenv("COLI_WEX_VERIFY");

    printf("one child process each (the engine stops with exit 1 on a refusal)\n");
    setenv("G53_TEST_INT4", "1", 1);
    child_case(argv[0], 1, "an int4 checkpoint's experts and a container: refused");
    unsetenv("G53_TEST_INT4");
    put(old, old_bytes);
    child_case(argv[0], 1, "the WEX1 form: refused");
    w4[100] ^= 1u; put(w4, w4_bytes); w4[100] ^= 1u;
    child_case(argv[0], 1, "a bit of the header page changed: refused");
    w4[index_offset + 70] ^= 1u; put(w4, w4_bytes); w4[index_offset + 70] ^= 1u;
    child_case(argv[0], 1, "a bit of the index changed: refused");
    unsigned char *one = wex1(2, -1, 0, &other_bytes, scratch), *w4x = wex4_fixture_from_wex1(one, other_bytes, &bytes, other);
    put(w4x, bytes); free(one); free(w4x);
    child_case(argv[0], 1, "a container of another grid (2 x 3): refused");
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
    unsetenv("COLI_WEX_ROOT");
    child_case(argv[0], 4, "the changed record with no pin and no COLI_WEX_VERIFY: served as it is (no check was asked)");
    w4[offset[4] + 100] ^= 0x10u;

    printf("WNR3 records (codec 5)\n");
    size_t nr3_bytes, at[CELLS], cz_bytes;
    char nroot[65], zroot[65], renamed_root[65];
    unsigned char *nr3 = wnr3f_from_wex4(w4, w4_bytes, 0, &nr3_bytes, nroot, at);
    unsigned char *cz = wnr3f_from_wex4(w4, w4_bytes, 1, &cz_bytes, zroot, NULL);
    ck(nr3 && cz, "the WNR3 and canonical-zero forms of the container written (every record smaller than RAW)");
    if (!nr3 || !cz) return 1;
    put(nr3, nr3_bytes);
    open_model(&m);
    ck(m.wex && m.wex->nr3, "the WNR3 container attaches");
    close_model(&m);
    ck(attach_and_load() == 0, "every expert decoded from it holds the shards' bytes, the products bit for bit the same");
    setenv("COLI_WEX_ROOT", nroot, 1);
    ck(attach_and_load() == 0, "pinned: every WNR3 record passes its hash at its first read and decodes");
    unsetenv("COLI_WEX_ROOT");
#ifdef _WIN32
    const char *helper = argc > 1 ? argv[1] : "tests/fake_wex_cuda.dll";
#else
    const char *helper = argc > 1 ? argv[1] : "tests/fake_wex_cuda.so";
#endif
#ifdef COLI_WEX_CUDA_LOADER
    FILE *probe = fopen(helper, "rb");
    if (probe) fclose(probe);
#else
    FILE *probe = NULL;   /* wex_decode.c loads the helper on Windows and Linux only */
#endif
    if (probe) {   /* the fake CUDA helper (tests/fake_wex_cuda): every WNR3 record decoded by it */
        setenv("COLI_WEX_CUDA_LIB", helper, 1);
        child_case(argv[0], 0, "COLI_WEX_CUDA_LIB (the fake helper): every WNR3 record decoded by the helper, the same bytes");
        setenv("FAKE_WEX_HOST_FAIL", "1", 1);
        child_case(argv[0], 9, "the helper refuses every record (status 1): each decoded on the CPU, the same bytes");
        unsetenv("FAKE_WEX_HOST_FAIL");
        unsetenv("COLI_WEX_CUDA_LIB");
    } else
#ifdef COLI_WEX_CUDA_LOADER
        printf("  skip COLI_WEX_CUDA_LIB: no %s (make tests/fake_wex_cuda first)\n", helper);
#else
        printf("  skip COLI_WEX_CUDA_LIB: the helper is loaded on Windows and Linux only (%s)\n", helper);
#endif
    put(cz, cz_bytes);
    ck(attach_and_load() == 0, "the canonical-zero WNR3 container: the same records");
    nr3[at[4] + 5000] ^= 0x10u; put(nr3, nr3_bytes);
    child_case(argv[0], 1, "WNR3 record of layer 1 expert 1 changed on disk, no hash check: its stream check stops the engine");
    setenv("COLI_WEX_ROOT", nroot, 1);
    child_case(argv[0], 1, "the same, pinned: its SHA-256 stops the engine at its first read");
    unsetenv("COLI_WEX_ROOT");
    nr3[at[4] + 5000] ^= 0x10u;
    unsigned char *renamed = malloc(nr3_bytes);
    assert(renamed);
    memcpy(renamed, nr3, nr3_bytes);
    ck(!wnr3f_rename(renamed, nr3_bytes, 1u << 16 | 1u, 1u << 16 | 2u, source[4], renamed_root),
       "layer 1 expert 1's record written again under expert 2's name, the index resealed");
    put(renamed, nr3_bytes);
    setenv("COLI_WEX_ROOT", renamed_root, 1);
    child_case(argv[0], 1, "that whole record, its hash right, pinned: its owner stops the engine");
    unsetenv("COLI_WEX_ROOT");
    free(renamed);
    nr3[WEX4_PAGE + 100] ^= 1u; put(nr3, nr3_bytes); nr3[WEX4_PAGE + 100] ^= 1u;
    child_case(argv[0], 1, "a bit of the WNM2 model blob changed: refused");
    free(nr3); free(cz);

    unsetenv("COLI_WEX_PATH");
    remove(wex_path); remove(st_path); rmdir(directory);
    free(old); free(w4);
    if (fails) { printf("test_glm53_wex4: %d failure(s)\n", fails); return 1; }
    printf("test_glm53_wex4: ok\n");
    return 0;
}
