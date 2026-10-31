/* canonical.qc (qwen36_dense_file.h) without a model: synthetic BF16, F16 and
 * F32 source matrices go through qwen36-canonical's own writer and back
 * through the engine's reader (qwen_dense_read), whatever the size of the
 * blocks the writer converts at a time; every kind of wrong file is refused
 * with its reason. It also pins the quantizer: qw_quantize must give the bytes
 * of QDF_QUANTIZER's definition below, so a change to qw_quantize fails here
 * until QDF_QUANTIZER (and this definition) changes with it. */
#define QWEN36_CANONICAL_NO_MAIN
#include "../tools/qwen36_canonical.c"

static int failures, checks;
#define CHECK(c, ...) do { checks++; if (!(c)) { failures++; fprintf(stderr, "FAIL line %d: ", __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

enum { N = 3 };
static const QdfSpec specs[N] = {
    {"model.layers.0.mlp.gate.weight", 64, 8},             /* BF16 */
    {"model.layers.0.self_attn.q_proj.weight", 64, 33},    /* F16, an odd row count */
    {"lm_head.weight", 128, 80},                            /* F32 */
};
static const char *const paths[] = {"qwen36_dense_file_test.qc", "qwen36_dense_file_test-blocks.qc",
                                    "qwen36_dense_file_test-prefix.qc", "qwen36_dense_file_test-bad.qc"};
static st_tensor tensors[N];
static char prefixed[N][QDF_NAME + 16];
static shards S;
static FILE *src;

static void cleanup(void) {
    for (size_t i = 0; i < sizeof paths / sizeof paths[0]; i++) remove(paths[i]);
}

/* QDF_QUANTIZER "rows-amax127-lrintf-v1": per row, scale = max|w| / 127 (1 for
 * a row below 1e-12), values lrintf-rounded and clamped to +-127. */
static void quantize_rows(int8_t *q, float *sc, const float *f, int I, int O) {
    for (int o = 0; o < O; o++) {
        const float *r = f + (size_t)o * I; float am = 0.f;
        for (int i = 0; i < I; i++) { float a = fabsf(r[i]); if (a > am) am = a; }
        float s = am > 1e-12f ? am / 127.f : 1.f, inv = 1.f / s; sc[o] = s;
        for (int i = 0; i < I; i++) { int v = (int)lrintf(r[i] * inv); if (v > 127) v = 127; if (v < -127) v = -127; q[(size_t)o * I + i] = (int8_t)v; }
    }
}

/* A value of matrix m: a mix of signs and magnitudes, a zero row (row 0 of
 * matrix 1: scale 1) and a row of exact halves (row 3 of matrix 2: max 127, so
 * lrintf's ties to even decide every value). */
static float value(int m, size_t i) {
    size_t I = (size_t)specs[m].I, row = i / I, col = i % I;
    if (m == 1 && row == 0) return 0.f;
    if (m == 2 && row == 3) return col == 0 ? 127.f : (float)((int)col % 9 - 4) + 0.5f;
    return (float)((int)((i * 37 + (size_t)m * 11) % 255) - 127) / 50.f * (float)(1 + row % 5);
}

/* The source matrices in one file, each in its dtype, as a checkpoint holds them. */
static void make_source(void) {
    src = tmpfile();
    if (!src) exit(2);
    unsigned char prefix[40] = {0};   /* the matrices start at an odd offset */
    if (fwrite(prefix, 1, 37, src) != 37) exit(2);
    int64_t at = 37;
    for (int m = 0; m < N; m++) {
        size_t n = (size_t)specs[m].I * specs[m].O, esz = m == 2 ? 4 : 2;
        for (size_t i = 0; i < n; i++) {
            float f = value(m, i); uint32_t bits; memcpy(&bits, &f, 4);
            uint16_t h = m == 0 ? (uint16_t)(bits >> 16) : f32_to_f16_bits(f);
            if ((esz == 4 ? fwrite(&f, 4, 1, src) : fwrite(&h, 2, 1, src)) != 1) exit(2);
        }
        st_tensor *t = &tensors[m];
        memset(t, 0, sizeof *t);
        t->name = (char *)specs[m].name; t->fd = fileno(src); t->off = at; t->dtype = m;
        t->rank = 2; t->shape[0] = specs[m].O; t->shape[1] = specs[m].I;
        t->numel = (int64_t)n; t->nbytes = (int64_t)(n * esz);
        at += t->nbytes;
    }
    if (fflush(src)) exit(2);
    memset(&S, 0, sizeof S);
    S.t = tensors; S.n = N;
}

static unsigned char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); unsigned char *b = NULL; long k = -1;
    if (f && !fseek(f, 0, SEEK_END) && (k = ftell(f)) >= 0 && !fseek(f, 0, SEEK_SET) && (b = malloc((size_t)k + 1)) &&
        fread(b, 1, (size_t)k, f) == (size_t)k) { fclose(f); *n = (size_t)k; return b; }
    if (f) fclose(f);
    free(b); exit(2);
}
static void spit(const char *path, const unsigned char *b, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(b, 1, n, f) != n || fclose(f)) exit(2);
}

static void expect_refused(const char *path, uint64_t fingerprint, const char *what, const char *reason) {
    QdfFile f;
    char err[512] = "";
    CHECK(!qdf_open(&f, path, specs, N, fingerprint, err, sizeof err) && !f.base, "%s accepted", what);
    CHECK(strstr(err, reason) != NULL, "%s: reason \"%s\", expected \"%s\"", what, err, reason);
}

int main(void) {
    atexit(cleanup);
    make_source();
    static Model m;
    char err[512] = "";

    /* the quantizer: qw_quantize gives QDF_QUANTIZER's bytes on every source matrix */
    CHECK(!strcmp(QDF_QUANTIZER, "rows-amax127-lrintf-v1"), "QDF_QUANTIZER is %s: update quantize_rows", QDF_QUANTIZER);
    float *ref[N];
    for (int i = 0; i < N; i++) {
        int I = specs[i].I, O = specs[i].O;
        ref[i] = malloc((size_t)I * O * 4);
        if (!ref[i]) exit(2);
        st_read_f32(&S, specs[i].name, ref[i], 0);
        int8_t *q = malloc((size_t)I * O); float *sc = malloc((size_t)O * 4);
        QW w; memset(&w, 0, sizeof w);
        qw_quantize(ref[i], I, O, NULL, &w);
        quantize_rows(q, sc, ref[i], I, O);
        CHECK(!memcmp(q, w.q, (size_t)I * O) && !memcmp(sc, w.sc, (size_t)O * 4),
              "qw_quantize is not %s on matrix %d: change QDF_QUANTIZER", QDF_QUANTIZER, i);
        qw_free(&w); free(q); free(sc);
    }

    /* the tool's writer, in one block per matrix and in blocks of a few rows: the same bytes */
    uint64_t size = 0, size2 = 0;
    CHECK(qwen36_canonical_write(paths[0], &S, specs, N, (size_t)64 << 20, &size), "write failed");
    CHECK(qwen36_canonical_write(paths[1], &S, specs, N, 3 * 128 * 4, &size2), "write in small blocks failed");
    size_t n0 = 0, n1 = 0;
    unsigned char *good = slurp(paths[0], &n0), *blocks = slurp(paths[1], &n1);
    CHECK(n0 == size && n1 == size2 && n0 == n1 && !memcmp(good, blocks, n0), "the block size changes the file");
    free(blocks);

    /* the engine's reader: the file's matrices are qw_quantize's, row for row and scale for scale */
    uint64_t fingerprint = qwen_dense_fingerprint(&S, specs, N);
    CHECK(fingerprint != 0, "no fingerprint");
    CHECK(qdf_open(&m.dense_file, paths[0], specs, N, fingerprint, err, sizeof err), "valid file refused: %s", err);
    for (int i = 0; m.dense_file.base && i < N; i++) {
        int I = specs[i].I, O = specs[i].O;
        QW got, want; memset(&got, 0, sizeof got); memset(&want, 0, sizeof want);
        CHECK(qwen_dense_read(&m, specs[i].name, I, O, &got), "matrix %d not read", i);
        qw_quantize(ref[i], I, O, NULL, &want);
        CHECK(got.q && got.sc && !memcmp(got.q, want.q, (size_t)I * O) && !memcmp(got.sc, want.sc, (size_t)O * 4),
              "matrix %d: the bytes read differ from the conversion", i);
        const int8_t *wq = NULL; const float *wsc = NULL;
        CHECK(qdf_find(&m.dense_file, specs[i].name, I, O, &wq, &wsc) &&
              ((const unsigned char *)wq - m.dense_file.base) % 16384 == 0 && ((uintptr_t)wsc % 4) == 0,
              "matrix %d misaligned", i);
        qw_free(&got); qw_free(&want);
    }
    const int8_t *wq; const float *wsc;
    CHECK(!qdf_find(&m.dense_file, specs[0].name, specs[0].I + 1, specs[0].O, &wq, &wsc), "wrong shape found");
    CHECK(!qdf_find(&m.dense_file, "model.layers.9.mlp.gate.weight", 64, 8, &wq, &wsc), "unknown name found");
    qdf_close(&m.dense_file);

    /* a checkpoint that stores the matrices under the language_model. prefix: the same file */
    for (int i = 0; i < N; i++) {
        snprintf(prefixed[i], sizeof prefixed[i], "language_model.%s", specs[i].name);
        tensors[i].name = prefixed[i];
    }
    CHECK(qwen36_canonical_write(paths[2], &S, specs, N, (size_t)64 << 20, NULL), "write from prefixed names failed");
    size_t np = 0; unsigned char *pre = slurp(paths[2], &np);
    CHECK(np == n0 && !memcmp(pre, good, n0) && qwen_dense_fingerprint(&S, specs, N) == fingerprint,
          "prefixed names give another file");
    free(pre);
    for (int i = 0; i < N; i++) tensors[i].name = (char *)specs[i].name;

    /* the fingerprint: a byte of another checkpoint at either end of a source matrix */
    for (int end = 0; end < 2; end++) {
        st_tensor *t = &tensors[1];
        int64_t at = end ? t->off + t->nbytes - 5 : t->off + 3;
        unsigned char b = 0, flip;
        if (fseek(src, (long)at, SEEK_SET) || fread(&b, 1, 1, src) != 1) exit(2);
        flip = (unsigned char)(b ^ 0x10);
        if (fseek(src, (long)at, SEEK_SET) || fwrite(&flip, 1, 1, src) != 1 || fflush(src)) exit(2);
        uint64_t other = qwen_dense_fingerprint(&S, specs, N);
        CHECK(other != fingerprint, "a changed %s of a source matrix keeps the fingerprint", end ? "end" : "start");
        expect_refused(paths[0], other, "other weights", "other weights");
        if (fseek(src, (long)at, SEEK_SET) || fwrite(&b, 1, 1, src) != 1 || fflush(src)) exit(2);
    }
    CHECK(qwen_dense_fingerprint(&S, specs, N) == fingerprint, "fingerprint not restored");
    CHECK(!qdf_open(&m.dense_file, paths[0], specs, N - 1, fingerprint, err, sizeof err) &&
          strstr(err, "matrices of this model"), "another matrix list accepted: %s", err);

    /* damaged copies of the file */
    QdfEntry e[N];
    qdf_layout(specs, N, e);
    struct { const char *what, *reason; size_t at; unsigned char x; float scale; } bad[] = {
        {"foreign file", "not a canonical.qc", 0, 'X', 0},
        {"other version", "another version", 8, 9, 0},
        {"changed table", "matrix table", sizeof(QdfHeader) + 128 + 104, 1, 0},
        {"NaN scale", "invalid scale", (size_t)e[1].sc + 4 * 7, 0, NAN},
        {"zero scale", "invalid scale", (size_t)e[2].sc, 0, 0.f},
    };
    for (size_t k = 0; k < sizeof bad / sizeof bad[0]; k++) {
        unsigned char *b = malloc(n0);
        if (!b) exit(2);
        memcpy(b, good, n0);
        if (bad[k].x) b[bad[k].at] = k == 2 ? (unsigned char)(b[bad[k].at] ^ bad[k].x) : bad[k].x;
        else memcpy(b + bad[k].at, &bad[k].scale, 4);
        spit(paths[3], b, n0);
        expect_refused(paths[3], fingerprint, bad[k].what, bad[k].reason);
        free(b);
    }
    spit(paths[3], good, n0 / 2);
    expect_refused(paths[3], fingerprint, "truncated file", "matrices of this model");
    remove(paths[3]);
    expect_refused(paths[3], fingerprint, "missing file", paths[3]);

    free(good);
    for (int i = 0; i < N; i++) free(ref[i]);
    printf("qwen36 dense file: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
