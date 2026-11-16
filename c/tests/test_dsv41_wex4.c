/* deepseek_v41's backbone experts from a WEX4 container (COLI_WEX_PATH, v41_wex_*), no model:
 * a synthetic checkpoint of 2 layers x 3 experts (dim 64, moe_intermediate 32: six parts of
 * 1,024 and 64 bytes, a record of 3,264 bytes, not a whole page) and its container, written
 * the way the in-place upgrade of a WEX1 file writes one (tests/wex4_fixture.h), with one
 * expert left out.
 *
 * Every expert read through expert_slot_at, one at a time and as a batch (expert_slots_at),
 * must hold the checkpoint's bytes in w1/w3/w2 and s1/s3/s2, whichever source served it.
 * The container serves the records it holds, pinned or not; a record changed on disk is read
 * again from the shards once its first-read SHA-256 fails (pinned, or COLI_WEX_VERIFY=1) and
 * stays on them; another root, a bad root, a damaged header page or index, a truncated copy,
 * the WEX1 form, another grid, records of other shapes and a missing file leave every expert
 * on the shards. */
#define main dsv41_main_unused
#include "../deepseek_v41.c"
#undef main
#include "wex4_fixture.h"

enum { LAYERS = 2, EXPERTS = 3, DIM = 64, INTER = 32 };
#define CELLS (LAYERS * EXPERTS)
#define S13 ((size_t)INTER * (DIM / 32))      /* w1 / w3 scales */
#define S2 ((size_t)DIM * (INTER / 32))       /* w2 scales */
#define W13 ((size_t)INTER * (DIM / 2))
#define W2 ((size_t)DIM * (INTER / 2))
#define RAW (2 * S13 + S2 + 2 * W13 + W2)    /* the record: s1 s2 s3 w1 w2 w3 */
#define STORED ((size_t)WEX_ALIGN_UP(RAW, WEX_ALIGN))
#define ABSENT 4                             /* layer 1 expert 1: not in the container */

static unsigned char source[CELLS][RAW];
static char directory[] = "colibri-v41-wex4-XXXXXX", st_path[600], wex_path[600];
static int fails;

/* where each matrix sits in a record (and so in the safetensors data, laid out the same) */
static size_t at_scale(int m) { return m == 0 ? 0 : m == 1 ? S13 : S13 + S2; }          /* w1, w2, w3 */
static size_t at_weight(int m) { return 2 * S13 + S2 + (m == 0 ? 0 : m == 1 ? W13 : W13 + W2); }

static void write_file(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, 1, bytes, f) != bytes || fclose(f)) { perror(path); exit(1); }
}

static void write_safetensors(void) {
    static const char *names[3] = {"w1", "w2", "w3"};
    char header[16384];
    size_t used = (size_t)snprintf(header, sizeof header, "{");
    for (int c = 0; c < CELLS; c++)
        for (int m = 0; m < 3; m++) {
            const int rows = m == 1 ? DIM : INTER, k = m == 1 ? INTER : DIM;
            const size_t s = (size_t)c * RAW + at_scale(m), w = (size_t)c * RAW + at_weight(m);
            used += (size_t)snprintf(header + used, sizeof header - used,
                "%s\"layers.%d.ffn.experts.%d.%s.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]},"
                "\"layers.%d.ffn.experts.%d.%s.weight\":{\"dtype\":\"I8\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                used > 1 ? "," : "", c / EXPERTS, c % EXPERTS, names[m], rows, k / 32, s, s + (size_t)rows * (k / 32),
                c / EXPERTS, c % EXPERTS, names[m], rows, k / 2, w, w + (size_t)rows * (k / 2));
        }
    header[used++] = '}';
    uint64_t length = used;
    FILE *f = fopen(st_path, "wb");
    if (!f || fwrite(&length, 8, 1, f) != 1 || fwrite(header, 1, used, f) != used ||
        fwrite(source, 1, sizeof source, f) != sizeof source || fclose(f)) { perror(st_path); exit(1); }
}

/* A WEX1 file of the records (all but ABSENT); other_shapes: the same sizes, other shapes. */
static unsigned char *wex1(int layers, int other_shapes, size_t *bytes, size_t offset[CELLS]) {
    const size_t cells = (size_t)layers * EXPERTS, table = (size_t)WEX_ALIGN_UP(cells * sizeof(WexEntry), WEX_ALIGN);
    size_t total = WEX_ALIGN + table, at = total;
    for (size_t c = 0; c < cells; c++) if (c != ABSENT) total += STORED;
    unsigned char *out = calloc(1, total);
    if (!out) exit(1);
    WexHeader *h = (WexHeader *)out;
    h->magic = WEX_MAGIC_V1; h->version = 1; h->num_layers = (uint32_t)layers; h->num_experts = EXPERTS;
    h->aligned_expert_bytes = STORED; h->table_offset = WEX_ALIGN;
    WexEntry *e = (WexEntry *)(out + WEX_ALIGN);
    for (size_t c = 0; c < cells; c++) {
        offset[c] = 0;
        if (c == ABSENT) continue;
        e[c].file_offset = at; e[c].total_bytes = (uint32_t)STORED;
        e[c].scale_bytes = (uint32_t)(2 * S13 + S2); e[c].weight_bytes = (uint32_t)(2 * W13 + W2);
        for (int m = 0; m < 3; m++) {
            uint32_t rows = m == 1 ? DIM : INTER, sc = (m == 1 ? INTER : DIM) / 32;
            if (other_shapes) { rows /= 2; sc *= 2; }
            e[c].scale_shapes[m][0] = e[c].weight_shapes[m][0] = (uint16_t)rows;
            e[c].scale_shapes[m][1] = (uint16_t)sc; e[c].weight_shapes[m][1] = (uint16_t)(sc * 16u);
            e[c].scale_offsets[m] = (uint32_t)at_scale(m);
            e[c].weight_offsets[m] = (uint32_t)at_weight(m);
        }
        memcpy(out + at, source[c], RAW);
        offset[c] = at;
        at += STORED;
    }
    *bytes = total;
    return out;
}

static int slot_ok(const Slot *s, int c) {
    const unsigned char *r = source[c];
    return s->eid == c % EXPERTS &&
           !memcmp(s->s1, r + at_scale(0), S13) && !memcmp(s->s2, r + at_scale(1), S2) && !memcmp(s->s3, r + at_scale(2), S13) &&
           !memcmp(s->w1, r + at_weight(0), W13) && !memcmp(s->w2, r + at_weight(1), W2) && !memcmp(s->w3, r + at_weight(2), W13);
}

/* Load the model's experts, one at a time and as a batch; every slot checked. Returns whether
 * a container was open, and its reads and refusals. */
static int run(uint64_t *reads, uint64_t *refused) {
    Model m;
    memset(&m, 0, sizeof m);
    m.c.dim = DIM; m.c.moe_inter = INTER; m.c.n_layers = LAYERS; m.c.n_routed = EXPERTS;
    st_init(&m.S, directory);
    v41_wex_open(&m);
    int ok = 1;
    for (int layer = 0; layer < LAYERS; layer++) {
        LCache one, all;
        cache_init(&m, &one, 1, 0);           /* one slot: every expert read again */
        for (int pass = 0; pass < 2; pass++)
            for (int e = 0; e < EXPERTS; e++) ok &= slot_ok(expert_slot_at(&m, &one, "layers", layer, e), layer * EXPERTS + e);
        cache_init(&m, &all, EXPERTS, 0);     /* the batch: every miss in one expert_fetch */
        const int chosen[EXPERTS] = {2, 0, 1};
        Slot *out[EXPERTS];
        expert_slots_at(&m, &all, "layers", layer, chosen, EXPERTS, out);
        for (int k = 0; k < EXPERTS; k++) ok &= slot_ok(out[k], layer * EXPERTS + chosen[k]);
        LCache *caches[2] = {&one, &all};
        for (int q = 0; q < 2; q++) {
            for (int i = 0; i < caches[q]->cap; i++)
                for (int p = 0; p < V41_EXPERT_TENSORS; p++) compat_aligned_free(caches[q]->slot[i].base[p]);
            free(caches[q]->slot);
        }
    }
    if (!ok) { printf("  FAIL a slot does not hold the checkpoint's bytes\n"); fails++; }
    const int active = m.wex != NULL;
    *reads = active ? m.wex->reads : 0;
    *refused = active ? m.wex->refused : 0;
    v41_wex_free(m.wex);
    st_destroy(&m.S);
    for (int i = 0; i < LAYERS; i++) free(m.ehit ? m.ehit[i] : NULL);
    free(m.ehit);
    return active;
}

static void expect(const char *what, int active, uint64_t reads, uint64_t refused) {
    uint64_t r, f;
    const int got = run(&r, &f);
    const int ok = got == active && r == reads && f == refused;
    printf("  %s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) printf(" (container %d, reads %llu, refused %llu; want %d, %llu, %llu)", got, (unsigned long long)r,
                    (unsigned long long)f, active, (unsigned long long)reads, (unsigned long long)refused);
    printf("\n");
    fails += !ok;
}

int main(void) {
    setbuf(stdout, NULL);
    unsetenv("COLI_WEX_PATH"); unsetenv("COLI_WEX_ROOT"); unsetenv("COLI_WEX_VERIFY"); unsetenv("COLI_MODEL_MIRROR");
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(st_path, sizeof st_path, "%s/model.safetensors", directory);
    snprintf(wex_path, sizeof wex_path, "%s/experts.wex4", directory);   /* not *.safetensors: st_init skips it */
    uint32_t seed = 20261116u;
    for (int c = 0; c < CELLS; c++)
        for (size_t j = 0; j < RAW; j++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            source[c][j] = (unsigned char)(j < 2 * S13 + S2 ? 118u + seed % 8u : seed);
        }
    write_safetensors();
    size_t old_bytes, w4_bytes, bytes, other_bytes, offset[CELLS], scratch[CELLS];
    char root[65], other[65];
    unsigned char *old = wex1(LAYERS, 0, &old_bytes, offset), *w4 = wex4_fixture_from_wex1(old, old_bytes, &w4_bytes, root);
    uint64_t index_offset;
    memcpy(&index_offset, w4 + 32, 8);
    /* Reads counted per expert served from the container: per layer 2 passes x 3 one at a
     * time and a batch of 3, 9 reads, minus those of the absent expert (layer 1 expert 1:
     * 3 in layer 1): 18 - 3 = 15. */
    const uint64_t held = 2 * 9 - 3;

    expect("no COLI_WEX_PATH: the shards, no container", 0, 0, 0);
    write_file(wex_path, w4, w4_bytes);
    setenv("COLI_WEX_PATH", wex_path, 1);
    expect("WEX4 container: every expert it holds read from it, the one it lacks from the shards", 1, held, 0);
    setenv("COLI_WEX_ROOT", root, 1);
    expect("its root pinned (first-read SHA-256 on): the same", 1, held, 0);
    w4[offset[2] + 100] ^= 0x20u; write_file(wex_path, w4, w4_bytes);
    expect("layer 0 expert 2 changed on disk, pinned: refused at its first read, from the shards from then on",
           1, held - 3, 1);
    unsetenv("COLI_WEX_ROOT"); setenv("COLI_WEX_VERIFY", "1", 1);
    expect("the same with COLI_WEX_VERIFY=1 and no pin", 1, held - 3, 1);
    unsetenv("COLI_WEX_VERIFY");
    w4[offset[2] + 100] ^= 0x20u; write_file(wex_path, w4, w4_bytes);

    setenv("COLI_WEX_ROOT", "0123", 1);
    expect("COLI_WEX_ROOT not 64 hex digits: no container", 0, 0, 0);
    memset(other, 'c', 64); other[64] = 0;
    setenv("COLI_WEX_ROOT", other, 1);
    expect("another root pinned: no container", 0, 0, 0);
    unsetenv("COLI_WEX_ROOT");
    w4[index_offset + 70] ^= 1u; write_file(wex_path, w4, w4_bytes); w4[index_offset + 70] ^= 1u;
    expect("a bit of the index changed: no container", 0, 0, 0);
    w4[100] ^= 1u; write_file(wex_path, w4, w4_bytes); w4[100] ^= 1u;
    expect("a bit of the header page changed: no container", 0, 0, 0);
    write_file(wex_path, w4, w4_bytes - WEX_ALIGN);
    expect("a truncated copy: no container", 0, 0, 0);
    write_file(wex_path, old, old_bytes);
    expect("the WEX1 form: no container", 0, 0, 0);
    unsigned char *one = wex1(1, 0, &other_bytes, scratch), *w4x = wex4_fixture_from_wex1(one, other_bytes, &bytes, other);
    write_file(wex_path, w4x, bytes); free(one); free(w4x);
    expect("a container of another grid (1 x 3): no container", 0, 0, 0);
    one = wex1(LAYERS, 1, &other_bytes, scratch); w4x = wex4_fixture_from_wex1(one, other_bytes, &bytes, other);
    write_file(wex_path, w4x, bytes); free(one); free(w4x);
    expect("records of other shapes and the same sizes: no container", 0, 0, 0);
    snprintf(wex_path, sizeof wex_path, "%s/missing.wex4", directory);
    setenv("COLI_WEX_PATH", wex_path, 1);
    expect("COLI_WEX_PATH names no file: no container", 0, 0, 0);
    unsetenv("COLI_WEX_PATH");

    snprintf(wex_path, sizeof wex_path, "%s/experts.wex4", directory);
    remove(wex_path); remove(st_path); rmdir(directory);
    free(old); free(w4);
    if (fails) { printf("test_dsv41_wex4: %d failure(s)\n", fails); return 1; }
    printf("test_dsv41_wex4: ok\n");
    return 0;
}
