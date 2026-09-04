/* Dense-bridge gates for qwen36 against an MLX affine model.safetensors (the
 * dense half of a Swiftlet qpack container).  What must hold:
 *
 *   1. name resolution -- the engine speaks unprefixed names
 *      (`model.layers.N...`); a multimodal container stores the text stack
 *      under `language_model.`.  Both spellings must load, and the plain
 *      spelling must win when it exists (converted snapshots untouched).
 *   2. U32 indexing -- st_init must index the packed-U32 weight tensors a
 *      quantized container carries (it used to exit(1) at the first one)
 *      with the correct shape/rank, while st_read_f32 still refuses them.
 *   3. affine expansion -- a `.weight` U32 tensor with `.scales`/`.biases`
 *      siblings expands through the checked affine contract, with bits
 *      (Q4/Q8) and group size DERIVED from the packed shape against the
 *      config-implied element count, and the expanded values equal to
 *      scale*q + bias for the authored bit patterns.
 *   4. refusals -- a want that the packed shape cannot tile, an orphan U32
 *      weight without siblings, and a plain tensor of the wrong size all
 *      exit non-zero instead of returning plausible rows (fork gates,
 *      POSIX only).
 *
 * With argv, the binary becomes the round-trip harness for the Python tool
 * test (tests/test_make_qwen36_qpack_snap.py):
 *     test_qwen36_dense_affine <dir> <tensor> <want> <expected.f32>
 * loads <tensor> through the engine's own load_t_n and compares against a
 * reference computed INDEPENDENTLY in Python -- the cross-implementation
 * evidence that the C expansion agrees with the format's producer side.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include <sys/stat.h>
#ifndef _WIN32
#include <sys/wait.h>
#endif

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                           \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

/* ---- fixture: a minimal safetensors file with affine triples ------------- */

typedef struct {
    const char *name, *dtype, *shape;
    const void *bytes;
    size_t nbytes;
} FixtureTensor;

static void write_fixture(const char *path, const FixtureTensor *tensors, int n) {
    char header[4096];
    size_t hlen = 0, off = 0;
    hlen += (size_t)snprintf(header + hlen, sizeof(header) - hlen, "{");
    for (int i = 0; i < n; i++) {
        hlen += (size_t)snprintf(header + hlen, sizeof(header) - hlen,
            "%s\"%s\":{\"dtype\":\"%s\",\"shape\":%s,"
            "\"data_offsets\":[%zu,%zu]}",
            i ? "," : "", tensors[i].name, tensors[i].dtype, tensors[i].shape,
            off, off + tensors[i].nbytes);
        off += tensors[i].nbytes;
    }
    hlen += (size_t)snprintf(header + hlen, sizeof(header) - hlen, "}");
    CHECK(hlen < sizeof(header));
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    uint64_t h64 = (uint64_t)hlen;
    CHECK(fwrite(&h64, 8, 1, f) == 1);
    CHECK(fwrite(header, 1, hlen, f) == hlen);
    for (int i = 0; i < n; i++)
        CHECK(tensors[i].nbytes == 0 ||
              fwrite(tensors[i].bytes, 1, tensors[i].nbytes, f) == tensors[i].nbytes);
    CHECK(fclose(f) == 0);
}

static uint16_t bf16_bits(float value) {   /* fixture values are bf16-exact */
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(text, 1, strlen(text), f) == strlen(text));
    CHECK(fclose(f) == 0);
}

/* Q4 [2,32] logical, gs=16: per (row,group) scale 0.5*(1+idx), bias
 * 0.25*idx - 1.0 (all bf16-exact). */
static const uint32_t q4_words[2][4] = {
    { 0x76543210u, 0xFEDCBA98u, 0x00000000u, 0x11111111u },
    { 0xAAAAAAAAu, 0x55555555u, 0x0F0F0F0Fu, 0xF0F0F0F0u },
};
static float q4_scale(int r, int g) { return 0.5f * (float)(1 + r * 2 + g); }
static float q4_bias(int r, int g)  { return 0.25f * (float)(r * 2 + g) - 1.0f; }

/* Q8 [3,8] logical, gs=8: one group per row. */
static const uint32_t q8_words[3][2] = {
    { 0x03020100u, 0x07060504u },
    { 0xFF800001u, 0x00000000u },
    { 0x10204080u, 0x01010101u },
};
static float q8_scale(int r) { return 1.0f + 0.5f * (float)r; }
static float q8_bias(int r)  { return -0.5f * (float)r; }

static const float plain_bf16_vals[4] = { 1.0f, -2.0f, 0.5f, 4.0f };
static const float plain_f32_vals[3] = { 0.25f, -0.5f, 8.0f };

static void build_fixture_dir(const char *dir) {
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0700);
#endif
    static uint16_t q4_scales[4], q4_biases[4], q8_scales[3], q8_biases[3];
    static uint16_t plain_bf16[4];
    for (int r = 0; r < 2; r++) for (int g = 0; g < 2; g++) {
        q4_scales[r * 2 + g] = bf16_bits(q4_scale(r, g));
        q4_biases[r * 2 + g] = bf16_bits(q4_bias(r, g));
    }
    for (int r = 0; r < 3; r++) {
        q8_scales[r] = bf16_bits(q8_scale(r));
        q8_biases[r] = bf16_bits(q8_bias(r));
    }
    for (int i = 0; i < 4; i++) plain_bf16[i] = bf16_bits(plain_bf16_vals[i]);
    static const uint32_t orphan_word[1] = { 0x12345678u };
    const FixtureTensor tensors[] = {
        { "language_model.model.aff_q4.weight", "U32", "[2,4]",
          q4_words, sizeof(q4_words) },
        { "language_model.model.aff_q4.scales", "BF16", "[2,2]",
          q4_scales, sizeof(q4_scales) },
        { "language_model.model.aff_q4.biases", "BF16", "[2,2]",
          q4_biases, sizeof(q4_biases) },
        { "language_model.model.aff_q8.weight", "U32", "[3,2]",
          q8_words, sizeof(q8_words) },
        { "language_model.model.aff_q8.scales", "BF16", "[3,1]",
          q8_scales, sizeof(q8_scales) },
        { "language_model.model.aff_q8.biases", "BF16", "[3,1]",
          q8_biases, sizeof(q8_biases) },
        { "language_model.model.plain.weight", "BF16", "[4]",
          plain_bf16, sizeof(plain_bf16) },
        { "model.unprefixed.weight", "F32", "[3]",
          plain_f32_vals, sizeof(plain_f32_vals) },
        { "language_model.model.orphan.weight", "U32", "[1,1]",
          orphan_word, sizeof(orphan_word) },
    };
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    write_fixture(path, tensors, (int)(sizeof(tensors) / sizeof(tensors[0])));
}

/* ---- gates --------------------------------------------------------------- */

static Model g_dm;

static void test_u32_indexed_with_shape(void) {
    st_tensor *t = st_find(&g_dm.S, "language_model.model.aff_q4.weight");
    CHECK(t != NULL);
    CHECK(t->dtype == 7);
    CHECK(t->rank == 2 && t->shape[0] == 2 && t->shape[1] == 4);
    CHECK(t->numel == 8 && t->nbytes == 32);
}

static void test_q4_expansion(void) {
    float *w = load_t_n(&g_dm, "model.aff_q4.weight", 2 * 32);
    for (int r = 0; r < 2; r++) for (int i = 0; i < 32; i++) {
        int g = i / 16;
        float q = (float)((q4_words[r][i / 8] >> (4 * (i % 8))) & 0xFu);
        CHECK(w[r * 32 + i] == q4_scale(r, g) * q + q4_bias(r, g));
    }
    free(w);
}

static void test_q8_expansion(void) {
    float *w = load_t_n(&g_dm, "model.aff_q8.weight", 3 * 8);
    for (int r = 0; r < 3; r++) for (int i = 0; i < 8; i++) {
        float q = (float)((q8_words[r][i / 4] >> (8 * (i % 4))) & 0xFFu);
        CHECK(w[r * 8 + i] == q8_scale(r) * q + q8_bias(r));
    }
    free(w);
}

/* MLX norm-weight dialect: with zero_centered_norms=0 (what the snap-view
 * tool emits for MLX-derived containers) load_norm_n must undo the
 * materialised +1 so rmsnorm_row's (1 + w) sees zero-centered weights again;
 * with the default HF dialect it must not touch a value. */
static void test_norm_unshift(void) {
    g_dm.c.zero_centered_norms = 1;
    float *w = load_norm_n(&g_dm, "model.plain.weight", 4);
    for (int i = 0; i < 4; i++) CHECK(w[i] == plain_bf16_vals[i]);
    free(w);
    g_dm.c.zero_centered_norms = 0;
    w = load_norm_n(&g_dm, "model.plain.weight", 4);
    for (int i = 0; i < 4; i++) CHECK(w[i] == plain_bf16_vals[i] - 1.0f);
    free(w);
    g_dm.c.zero_centered_norms = 1;
}

/* qwen36_meta.json carries the dialect flag; load_meta must parse it. */
static void test_meta_flag(void) {
    write_file("tests/tmp_dense_affine/qwen36_meta.json",
               "{\"zero_centered_norms\": false}");
    Cfg c;
    memset(&c, 0, sizeof(c));
    c.n_layers = 1;
    c.is_attn = calloc(1, 1);
    c.zero_centered_norms = 1;
    load_meta(&c, "tests/tmp_dense_affine");
    CHECK(c.zero_centered_norms == 0);
    free(c.is_attn);
}

static void test_prefix_resolution(void) {
    CHECK(dense_has(&g_dm, "model.plain.weight"));
    CHECK(dense_has(&g_dm, "model.unprefixed.weight"));
    CHECK(!dense_has(&g_dm, "model.absent.weight"));
    float *p = load_t_n(&g_dm, "model.plain.weight", 4);
    for (int i = 0; i < 4; i++) CHECK(p[i] == plain_bf16_vals[i]);
    free(p);
    /* the plain spelling wins when it exists: this fixture only has the
     * unprefixed one, and it must load without any prefix probing */
    float *u = load_t_n(&g_dm, "model.unprefixed.weight", 3);
    for (int i = 0; i < 3; i++) CHECK(u[i] == plain_f32_vals[i]);
    free(u);
}

#ifndef _WIN32
static void expect_load_death(const char *name, int64_t want) {
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stderr)) { /* keep the noise then */ }
        float *p = load_t_n(&g_dm, name, want);
        (void)p;
        _exit(0);   /* reaching here means the refusal did not happen */
    }
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) != 0);
}

static void test_refusals(void) {
    expect_load_death("model.aff_q4.weight", 2 * 32 + 1); /* untileable want */
    expect_load_death("model.aff_q4.weight", 2 * 12);     /* neither Q4 nor Q8 */
    expect_load_death("model.orphan.weight", 8);          /* missing siblings */
    expect_load_death("model.plain.weight", 5);           /* plain wrong size */
    expect_load_death("model.absent.weight", 4);          /* missing tensor */
}
#endif

/* ---- Python round-trip harness ------------------------------------------ */

static int compare_mode(const char *dir, const char *tensor, int64_t want,
                        const char *expected_path) {
    st_init(&g_dm.S, dir);
    float *got = load_t_n(&g_dm, tensor, want);
    FILE *f = fopen(expected_path, "rb");
    if (!f) { perror(expected_path); return 1; }
    float *expected = falloc(want);
    if (fread(expected, sizeof(float), (size_t)want, f) != (size_t)want) {
        fprintf(stderr, "%s: short read\n", expected_path); return 1;
    }
    fclose(f);
    double max_diff = 0.0;
    for (int64_t i = 0; i < want; i++) {
        double diff = fabs((double)got[i] - (double)expected[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-5 * (1.0 + fabs((double)expected[i]))) {
            fprintf(stderr, "%s[%lld]: got %.9g expected %.9g\n",
                    tensor, (long long)i, got[i], expected[i]);
            return 1;
        }
    }
    printf("round-trip %s: %lld elements, max |diff| %.3g\n",
           tensor, (long long)want, max_diff);
    free(got); free(expected);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 5)
        return compare_mode(argv[1], argv[2], atoll(argv[3]), argv[4]);
    if (argc != 1) {
        fprintf(stderr, "usage: %s [<dir> <tensor> <want> <expected.f32>]\n",
                argv[0]);
        return 2;
    }
    const char *dir = "tests/tmp_dense_affine";
    build_fixture_dir(dir);
    st_init(&g_dm.S, dir);
    test_u32_indexed_with_shape();
    test_q4_expansion();
    test_q8_expansion();
    test_prefix_resolution();
    test_norm_unshift();
    test_meta_flag();
#ifndef _WIN32
    test_refusals();
#endif
    printf("test_qwen36_dense_affine: OK\n");
    return 0;
}
