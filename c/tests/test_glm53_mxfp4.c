/* glm53 on an AMD Quark MXFP4 checkpoint's experts (amd/GLM-5.3-Flash-Quark-MXFP4), no model: a
 * synthetic grid of 3 layers (layer 0 dense, without experts) x 3 experts, hidden 256 and
 * moe_intermediate 64, written as that checkpoint writes them (`gate_proj.weight` U8 e2m1 codes
 * [rows][columns/2], `gate_proj.weight_scale` U8 ue8m0 [rows][columns/32], the same for up and
 * down), plus one MXFP4 shared expert.
 *
 * The MXFP4 form is found from the first expert's `weight_scale`; the experts stream (slots of
 * 26,112 bytes, six pieces); every expert read holds its bytes as three fmt 7 matrices, and mv,
 * mm and mm_rows on them equal a scalar dequantization; the shared expert loads through load_mat
 * as a dense matrix of the MXFP4 values (GLM53_BITS=32: exactly). */
#define _GNU_SOURCE
#define GLM53_NO_MAIN
#include "../glm53.c"
#include "../compat.h"   /* setenv: MinGW has none */
#include <assert.h>

enum { LAYERS = 3, FIRST = 1, EXPERTS = 3, H = 256, I = 64 };
#define CELLS (LAYERS * EXPERTS)
static unsigned char codes[CELLS + 1][3][H * I / 2], exps[CELLS + 1][3][H * I / 32];   /* gate, up, down; last: shared */
static char directory[] = "colibri-g53-mx-XXXXXX", st_path[600];
static int fails;
static void ck(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); fails += !ok; }

static void fill(void) {
    uint32_t seed = 20261116u;
    for (int c = 0; c <= CELLS; c++)
        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < H * I / 2; j++) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; codes[c][k][j] = (unsigned char)seed; }
            for (int j = 0; j < H * I / 32; j++) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; exps[c][k][j] = (unsigned char)(120u + seed % 8u); }
        }
}

static void write_shards(void) {
    static const char *mat[3] = {"gate", "up", "down"};
    const int rows[3] = {I, I, H}, cols[3] = {H, H, I};
    char *json = malloc(1 << 16);
    size_t n = 0, data = 0;
    assert(json);
    n += (size_t)sprintf(json + n, "{");
    for (int c = 0; c <= CELLS; c++) {
        if (c < CELLS && c / EXPERTS < FIRST) continue;
        for (int k = 0; k < 3; k++) {
            char base[160];
            if (c < CELLS) snprintf(base, sizeof base, "model.language_model.layers.%d.mlp.experts.%d.%s_proj", c / EXPERTS, c % EXPERTS, mat[k]);
            else snprintf(base, sizeof base, "model.language_model.layers.1.mlp.shared_experts.%s_proj", mat[k]);
            n += (size_t)sprintf(json + n, "%s\"%s.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                                 n > 1 ? "," : "", base, rows[k], cols[k] / 2, data, data + (size_t)H * I / 2);
            data += (size_t)H * I / 2;
            n += (size_t)sprintf(json + n, ",\"%s.weight_scale\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                                 base, rows[k], cols[k] / 32, data, data + (size_t)H * I / 32);
            data += (size_t)H * I / 32;
        }
    }
    n += (size_t)sprintf(json + n, "}");
    while (n % 8) json[n++] = ' ';
    FILE *f = fopen(st_path, "wb");
    uint64_t len = n;
    assert(f && fwrite(&len, 8, 1, f) == 1 && fwrite(json, 1, n, f) == n);
    for (int c = 0; c <= CELLS; c++) {
        if (c < CELLS && c / EXPERTS < FIRST) continue;
        for (int k = 0; k < 3; k++) {
            assert(fwrite(codes[c][k], 1, (size_t)H * I / 2, f) == (size_t)H * I / 2);
            assert(fwrite(exps[c][k], 1, (size_t)H * I / 32, f) == (size_t)H * I / 32);
        }
    }
    assert(fclose(f) == 0);
    free(json);
}

static float deq(const unsigned char *q, const unsigned char *e, int columns, int r, int col) {
    const uint8_t b = q[(size_t)r * (columns / 2) + col / 2];
    return mx4_lut[col & 1 ? b >> 4 : b & 15] * mx4_scale(e[(size_t)r * (columns / 32) + col / 32]);
}

/* the matrix's products against the scalar dequantization: mv, mm of 3 rows, mm_rows of a block */
static int products_ok(const Mat *w, const unsigned char *q, const unsigned char *e) {
    enum { S = 3 };
    const int R = w->rows, C = w->columns;
    float x[S * H], y[S * H], z[S * H], ref;
    for (int i = 0; i < S * C; i++) x[i] = (float)((i * 37) % 17 - 8) / 16.0f;
    int ok = 1;
    mv(y, w, x);
    for (int r = 0; r < R && ok; r++) {
        double a = 0;
        for (int col = 0; col < C; col++) a += (double)x[col] * deq(q, e, C, r, col);
        ref = (float)a;
        ok = fabsf(ref - y[r]) <= 1e-4f * (1.0f + fabsf(ref));
    }
    mm(z, w, x, S);
    ok = ok && !memcmp(z, y, (size_t)R * sizeof(float));   /* mm's row 0 is mv's, bit for bit */
    mm_rows(y, w, x, S, 8, 16);
    for (int s = 0; s < S && ok; s++) ok = !memcmp(y + s * 16, z + (size_t)s * R + 8, 16 * sizeof(float));
    return ok;
}

int main(void) {
    setbuf(stdout, NULL);
    unsetenv("COLI_WEX_PATH");
    setenv("GLM53_BITS", "32", 1);
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(st_path, sizeof st_path, "%s/model.safetensors", directory);
    fill();
    write_shards();

    GModel m;
    memset(&m, 0, sizeof m);
    m.c.n_layers = LAYERS; m.c.n_experts = EXPERTS; m.c.hidden = H; m.c.moe_inter = I; m.c.first_dense = FIRST; m.c.n_shared = 1;
    m.layer_begin = 0; m.layer_end = LAYERS;
    st_init(&m.S, directory);
    snprintf(m.prefix, sizeof m.prefix, "model.language_model.");
    char first[200];
    snprintf(first, sizeof first, "%slayers.%d.mlp.experts.0.gate_proj.weight", m.prefix, FIRST);
    st_tensor *probe = st_find(&m.S, first);
    assert(probe);
    ck(g53_expert_form(&m, probe, first) == 1 && m.mx == 1 && !m.gx, "the weight_scale beside the first expert: MXFP4, streamed");
    m.streaming = 1;
    expert_geometry(&m);
    ck(m.e_slot == (int64_t)(3 * (H * I / 2 + H * I / 32)) && m.e_len[1] == H * I / 32 && m.e_len[0] == H * I / 2,
       "slots of 26,112 bytes: per matrix the codes, then a ue8m0 byte per 32 values");
    expert_table_init(&m);
    int ok = 1;
    for (int c = 0; c < CELLS && ok; c++) {
        if (c / EXPERTS < FIRST) continue;
        Slot slot;
        memset(&slot, 0, sizeof slot);
        expert_read(&m, c / EXPERTS, c % EXPERTS, &slot);
        Mat g, u, d;
        expert_mats(&m, c / EXPERTS, &slot, &g, &u, &d);
        ok = g.fmt == G53_MX && u.fmt == G53_MX && d.fmt == G53_MX && g.rows == I && g.columns == H && d.rows == H && d.columns == I &&
             !memcmp(g.q4, codes[c][0], H * I / 2) && !memcmp(g.e8, exps[c][0], H * I / 32) &&
             !memcmp(u.q4, codes[c][1], H * I / 2) && !memcmp(u.e8, exps[c][1], H * I / 32) &&
             !memcmp(d.q4, codes[c][2], H * I / 2) && !memcmp(d.e8, exps[c][2], H * I / 32) &&
             products_ok(&g, codes[c][0], exps[c][0]) && products_ok(&u, codes[c][1], exps[c][1]) &&
             products_ok(&d, codes[c][2], exps[c][2]);
        g53_slot_free(&slot);
    }
    ck(ok, "every expert read holds its codes and scales as three fmt 7 matrices; mv, mm and mm_rows equal the dequantization");
    Mat shared = load_mat(&m, I, H, "%slayers.1.mlp.shared_experts.gate_proj.weight", m.prefix);
    ok = shared.fmt == 0 && shared.rows == I && shared.columns == H;
    for (int r = 0; r < I && ok; r++)
        for (int col = 0; col < H && ok; col++) ok = shared.f[(size_t)r * H + col] == deq(codes[CELLS][0], exps[CELLS][0], H, r, col);
    ck(ok, "an MXFP4 shared expert through load_mat: the MXFP4 values, decoded at load (GLM53_BITS=32)");
    setenv("GLM53_BITS", "4", 1);
    st_destroy(&m.S); free(m.eref);
    remove(st_path); rmdir(directory);
    if (fails) { printf("test_glm53_mxfp4: %d failure(s)\n", fails); return 1; }
    printf("test_glm53_mxfp4: ok\n");
    return 0;
}
