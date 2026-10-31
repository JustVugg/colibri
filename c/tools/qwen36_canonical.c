/* qwen36-canonical: converts the INT8 dense matrices of a Qwen3.6 model once
 * and writes them as canonical.qc (format in qwen36_dense_file.h). The engine
 * then reads the file instead of converting the matrices at every start.
 *
 *   qwen36-canonical <model-dir> [<output>]     default: <model-dir>/canonical.qc
 *
 * The conversion is the engine's own (qw_quantize), so the file holds the
 * bytes of a start without it. It is written as <output>.tmp, flushed to the
 * disk and renamed at the end; an interrupted file is refused by the engine
 * anyway (its size does not match). The dense matrices must be BF16, F16 or
 * F32 (a qpack container's affine matrices are not converted). */
#define main qwen36_engine_main
#include "../qwen36.c"
#undef main

/* One matrix: its rows read from the checkpoint `block` bytes of f32 at a time
 * and quantized by the engine's qw_quantize. Rows are independent, so a block
 * at a time gives the bytes of the whole matrix, without its f32 copy in memory. */
static int convert(FILE *out, uint64_t *at, shards *S, const QdfSpec *s, const QdfEntry *e, size_t block) {
    const st_tensor *t = qwen_dense_source(S, s->name);
    if (!t) { fprintf(stderr, "%s: not in the checkpoint\n", s->name); return 0; }
    if (t->dtype < 0 || t->dtype > 2) {
        fprintf(stderr, "%s: stored as dtype %d; qwen36-canonical converts BF16, F16 and F32 matrices\n", s->name, t->dtype);
        return 0;
    }
    if (t->rank != 2 || t->shape[0] != s->O || t->shape[1] != s->I) {
        fprintf(stderr, "%s: not a %d x %d matrix\n", s->name, s->O, s->I);
        return 0;
    }
    size_t I = (size_t)s->I, O = (size_t)s->O, esz = (size_t)st_dtype_esz(t->dtype);
    size_t rows = block / (I * 4);
    if (rows < 1) rows = 1;
    if (rows > O) rows = O;
    float *f = malloc(rows * I * 4), *sc = malloc(O * 4);
    void *raw = malloc(rows * I * esz);
    int ok = f && sc && raw && qdf_pad(out, at, e->q);
    for (size_t row = 0; ok && row < O; row += rows) {
        size_t n = O - row < rows ? O - row : rows;
        st_pread_full(t->fd, raw, (int64_t)(n * I * esz), t->off + (int64_t)(row * I * esz), s->name);
        for (size_t k = 0; k < n * I; k++)
            f[k] = t->dtype == 2 ? ((const float *)raw)[k]
                 : t->dtype == 1 ? f16_to_f32(((const uint16_t *)raw)[k]) : bf16_to_f32(((const uint16_t *)raw)[k]);
        QW w;
        memset(&w, 0, sizeof w);
        qw_quantize(f, s->I, (int)n, NULL, &w);
        memcpy(sc + row, w.sc, n * 4);
        ok = fwrite(w.q, 1, n * I, out) == n * I;
        *at += n * I;
        qw_free(&w);
    }
    ok = ok && qdf_pad(out, at, e->sc) && fwrite(sc, 4, O, out) == O;
    *at += 4 * O;
    if (!ok) fprintf(stderr, "%s: conversion or write failed\n", s->name);
    free(f); free(sc); free(raw);
    return ok;
}

/* canonical.qc for the matrices spec[0..n) of S: written to path through
 * path.tmp, `block` bytes of f32 rows converted at a time; *size gets the
 * file's size. 0 on failure, with the reason on stderr. */
static int qwen36_canonical_write(const char *path, shards *S, const QdfSpec *spec, int n, size_t block, uint64_t *size) {
    char tmp[2100];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    for (int i = 0; i < n; i++)
        if (!qwen_dense_source(S, spec[i].name)) { fprintf(stderr, "%s: not in the checkpoint\n", spec[i].name); return 0; }
    QdfEntry *e = calloc(n > 0 ? (size_t)n : 1, sizeof *e);
    uint64_t bytes = e && n > 0 ? qdf_layout(spec, n, e) : 0, fingerprint = qwen_dense_fingerprint(S, spec, n);
    if (!bytes || !fingerprint) {
        fprintf(stderr, "%s: cannot lay out the dense matrices (no matrix, a name too long or an empty one)\n", path);
        free(e);
        return 0;
    }
    FILE *out = fopen(tmp, "wb");
    if (!out) { fprintf(stderr, "%s: %s\n", tmp, strerror(errno)); free(e); return 0; }
    uint64_t at = sizeof(QdfHeader) + (uint64_t)n * sizeof(QdfEntry);
    int ok = qdf_write_header(out, e, n, fingerprint, bytes);
    for (int i = 0; ok && i < n; i++) ok = convert(out, &at, S, &spec[i], &e[i], block);
    ok = ok && qdf_pad(out, &at, bytes) && !fflush(out);
#ifdef _WIN32
    ok = ok && !_commit(_fileno(out));
#else
    ok = ok && !fsync(fileno(out));
#endif
    ok = !fclose(out) && ok && !rename(tmp, path);
    if (!ok) { fprintf(stderr, "writing %s failed: %s\n", path, strerror(errno)); remove(tmp); }
    free(e);
    if (ok && size) *size = bytes;
    return ok;
}

#ifndef QWEN36_CANONICAL_NO_MAIN   /* tests/test_qwen36_dense_file.c includes the tool without its main */
int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <model-dir> [<output>]   (default <model-dir>/canonical.qc)\n", argv[0]);
        return 2;
    }
    char path[2048];
    if (argc > 2) snprintf(path, sizeof path, "%s", argv[2]);
    else snprintf(path, sizeof path, "%s/canonical.qc", argv[1]);
    Cfg c;
    memset(&c, 0, sizeof c);
    load_cfg(&c, argv[1]);
    int layers = c.n_layers;
    load_meta(&c, argv[1]);
    validate_cfg(&c, layers);
    shards S;
    st_init(&S, argv[1]);
    QwenDenseList l;
    if (!qwen_dense_list(&c, &l)) return 1;
    double started = now_s();
    uint64_t bytes = 0;
    if (!qwen36_canonical_write(path, &S, l.spec, l.count, (size_t)64 << 20, &bytes)) return 1;
    printf("%s: %d INT8 matrices, %llu bytes, %.1f s\n", path, l.count, (unsigned long long)bytes, now_s() - started);
    qwen_dense_list_free(&l);
    return 0;
}
#endif
