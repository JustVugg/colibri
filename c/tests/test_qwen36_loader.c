/* CPU-only merged int4 loader contract: signed values and row/group scales,
 * in unpacked-int8 and planar-int4 slots. A second expert changes every byte;
 * a second model is larger. The latter is a cross-model/process-lifetime test,
 * not growth during ordinary fixed-model inference. No allocation/perf claim.
 * The oracle describes logical elements; it calls no unpack/repack helpers.
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: loader check failed: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

/* Block, row, matrix and expert all affect values, including the two planar
 * halves. Avoid a period-16 pattern whose halves would be indistinguishable. */
static int value_at(int matrix, int row, int col, int expert) {
    return (row * 3 + col * 5 + col / 16 + matrix * 7 + expert * 9) % 16 - 8;
}
static float scale_at(int matrix, int row, int group, int expert) {
    return (float)(1 + matrix * 1000 + row * 8 + group + expert * 4000) / 16.0f;
}

/* Write a minimal safetensors file directly, with two experts under active
 * layer 0; loading logical layer 1 also checks active-index translation. */
static int fixture(const char *path, int h, int in) {
    char header[2048]; int pos = 0;
    size_t weights = (size_t)3 * h * in / 2;
    size_t scales = (size_t)3 * h * in / 64;
    CHECK(sizeof(float) == 4);
    pos += snprintf(header + pos, sizeof(header) - (size_t)pos, "{");
    for (int e = 0; e < 2; e++) {
        size_t start = (size_t)e * (weights + scales * 4);
        int n = snprintf(header + pos, sizeof(header) - (size_t)pos,
            "%s\"model.layers.0.mlp.experts.%d.merged_weight\":{\"dtype\":\"U8\",\"shape\":[%zu],\"data_offsets\":[%zu,%zu]},"
            "\"model.layers.0.mlp.experts.%d.qs\":{\"dtype\":\"F32\",\"shape\":[%zu],\"data_offsets\":[%zu,%zu]}",
            e ? "," : "", e, weights, start, start + weights,
            e, scales, start + weights, start + weights + scales * 4);
        CHECK(n > 0 && (size_t)n < sizeof(header) - (size_t)pos); pos += n;
    }
    CHECK((size_t)pos + 1 < sizeof(header)); header[pos++] = '}';
    FILE *f = fopen(path, "wb"); CHECK(f != NULL);
    /* Both safetensors header length and F32 data are little endian. */
    for (int b = 0; b < 8; b++) CHECK(fputc((int)(((uint64_t)pos >> (8 * b)) & 255), f) != EOF);
    CHECK(fwrite(header, 1, (size_t)pos, f) == (size_t)pos);
    for (int e = 0; e < 2; e++) {
        for (int m = 0; m < 3; m++) {
            int rows = m == 2 ? h : in, cols = m == 2 ? in : h;
            for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c += 2) {
                unsigned lo = (unsigned)(value_at(m, r, c, e) & 15);
                unsigned hi = (unsigned)(value_at(m, r, c + 1, e) & 15);
                CHECK(fputc((int)(lo | hi << 4), f) != EOF);
            }
        }
        for (int m = 0; m < 3; m++) {
            int rows = m == 2 ? h : in, cols = m == 2 ? in : h;
            for (int r = 0; r < rows; r++) for (int g = 0; g < cols / 64; g++) {
                float v = scale_at(m, r, g, e); uint32_t bits; memcpy(&bits, &v, 4);
                for (int b = 0; b < 4; b++) CHECK(fputc((int)((bits >> (8 * b)) & 255), f) != EOF);
            }
        }
    }
    CHECK(fclose(f) == 0); return 0;
}

static int exercise(const char *dir, int h, int in, int planar) {
    Model model = {0}; Slot slot = {0}; int active[2] = {-1, 0};
    model.c.hidden = h; model.c.inter = in; model.c.expert_gs = 64;
    model.active_of = active;
    st_init(&model.S, dir);
    CHECK(st_find(&model.S, "model.layers.0.mlp.experts.0.merged_weight") != NULL);
    size_t ng = (size_t)h * in, ns = ng / 64;
    if (planar) { slot.pw = malloc(3 * ng / 2); CHECK(slot.pw != NULL); memset(slot.pw, 0xA5, 3 * ng / 2); }
    else { slot.g = malloc(3 * ng); CHECK(slot.g != NULL); memset(slot.g, 0x55, 3 * ng); slot.u = slot.g + ng; slot.d = slot.u + ng; }
    slot.gs = malloc(3 * ns * sizeof(float)); CHECK(slot.gs != NULL);
    slot.us = slot.gs + ns; slot.ds = slot.us + ns;
    /* Same slot, same sizes, changed bytes AND scales; then revisit expert 0. */
    const int order[] = {0, 1, 0};
    for (int step = 0; step < 3; step++) {
        int e = order[step]; load_expert_merged(&model, 1, e, &slot);
        CHECK(slot.is_int4 == 1);
        for (int m = 0; m < 3; m++) {
            int rows = m == 2 ? h : in, cols = m == 2 ? in : h;
            for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
                int got;
                if (planar) {
                    /* Each row has blocks of 64 logical values: first half in
                     * low nibbles, second half in high nibbles, unsigned v+8. */
                    size_t address = (size_t)m * ng / 2 + (size_t)r * cols / 2
                                   + (size_t)(c / 64) * 32 + (size_t)(c % 32);
                    unsigned byte = slot.pw[address];
                    got = (int)((byte >> (c % 64 < 32 ? 0 : 4)) & 15) - 8;
                } else got = slot.g[(size_t)m * ng + (size_t)r * cols + (size_t)c];
                if (got != value_at(m, r, c, e)) {
                    fprintf(stderr, "value mismatch planar=%d h=%d in=%d step=%d matrix=%d row=%d col=%d got=%d expected=%d\n",
                            planar, h, in, step, m, r, c, got, value_at(m, r, c, e)); return 1;
                }
            }
            for (int r = 0; r < rows; r++) for (int g = 0; g < cols / 64; g++)
                CHECK(slot.gs[(size_t)m * ns + (size_t)r * (cols / 64) + (size_t)g] == scale_at(m, r, g, e));
        }
    }
    free(slot.pw); free(slot.g); free(slot.gs); free(slot.g4); free(slot.u4); free(slot.d4);
    /* Close this fixture's descriptors so Windows permits deletion. Parsing
     * metadata has no public teardown API; no leak-freedom claim is made. */
    for (int i = 0; i < model.S.nfd; i++) {
        CHECK(close(model.S.fds[i]) == 0);
        if (model.S.dfds[i] >= 0) CHECK(close(model.S.dfds[i]) == 0);
    }
    printf("loader ok: %s hidden=%d inter=%d experts=0,1,0\n", planar ? "planar-int4" : "unpacked-int8", h, in);
    return 0;
}

int main(void) {
    const char *tmp = getenv("TMPDIR"); if (!tmp || !*tmp) tmp = ".";
    char dir[1024], path[1100];
    int n = snprintf(dir, sizeof(dir), "%s/qwen36-loader-XXXXXX", tmp);
    CHECK(n > 0 && (size_t)n < sizeof(dir)); CHECK(mkdtemp(dir) != NULL);
    n = snprintf(path, sizeof(path), "%s/fixture.safetensors", dir);
    CHECK(n > 0 && (size_t)n < sizeof(path));
    CHECK(fixture(path, 64, 64) == 0);
    CHECK(exercise(dir, 64, 64, 0) == 0); CHECK(exercise(dir, 64, 64, 1) == 0);
    CHECK(fixture(path, 128, 192) == 0);
    CHECK(exercise(dir, 128, 192, 0) == 0); CHECK(exercise(dir, 128, 192, 1) == 0);
    CHECK(remove(path) == 0);
#ifdef _WIN32
    CHECK(_rmdir(dir) == 0);
#else
    CHECK(rmdir(dir) == 0);
#endif
    puts("merged int4 loader tests: ok"); return 0;
}
