/* qwen36-canonical: converts the INT8 dense matrices of a Qwen3.6 model once
 * and writes them as canonical.qc (format in qwen36_dense_file.h). The engine
 * then reads the file instead of converting the matrices at every start.
 *
 *   qwen36-canonical <model-dir> [<output>]     default: <model-dir>/canonical.qc
 *   qwen36-canonical --experts <model-dir> [<output>]  default: <model-dir>/experts.qc
 *
 * --experts can come anywhere; any other argument that starts with '-' is
 * refused rather than taken for a path. The conversion is the engine's own
 * (qw_quantize), so canonical.qc holds the bytes of a start without it. It is
 * written as <output>.tmp, flushed to the disk and renamed at the end; an
 * interrupted run leaves that temporary, which the next run overwrites, and
 * the engine never reads it. experts.qc is written as <output>.tmp.XXXXXX,
 * removed if a signal stops the run (see qwen36_expert_convert.h). The dense
 * matrices must be BF16, F16 or F32 (a qpack container's affine matrices are
 * not converted). */
#define main qwen36_engine_main
#include "../qwen36.c"
#undef main
#include "../qwen36_expert_convert.h"

static int convert_experts_main(const char *model, const char *output) {
    char path[2048], error[2300];
    int n = output ? snprintf(path, sizeof path, "%s", output)
                   : snprintf(path, sizeof path, "%s/experts.qc", model);
    if (n < 0 || (size_t)n >= sizeof path) { fprintf(stderr, "output path is too long\n"); return 2; }
    Cfg config = {0};
    load_cfg(&config, model); int layers = config.n_layers;
    load_meta(&config, model); validate_cfg(&config, layers);
    QefGeometry geometry = {(uint32_t)config.n_layers, (uint32_t)config.n_experts,
        (uint32_t)config.hidden, (uint32_t)config.inter, (uint32_t)config.expert_gs};
    QefHeader layout;
    if (!qef_layout(&geometry, 1, &layout, error, sizeof error)) {
        fprintf(stderr, "%s\n", error); free(config.is_attn); return 1;
    }
    shards source; st_init(&source, model);
    double started = now_s();
    int ok = qec_convert(&source, &geometry, path, error, sizeof error);
    st_destroy(&source); free(config.is_attn);
    if (!ok) { fprintf(stderr, "%s: %s\n", path, error); return 1; }
    struct stat info;
    uint64_t bytes = stat(path, &info) ? 0 : (uint64_t)info.st_size;
    /* The layout from the written header: with records padded to the same
     * stride, both layouts can have the same size. */
    QefHeader written; FILE *in = fopen(path, "rb");
    int header = in && fread(&written, sizeof written, 1, in) == 1;
    if (in) fclose(in);
    printf("%s: %llu planar INT4 experts, %llu bytes, scales %s, %.1f s; no requantization\n", path,
        (unsigned long long)layout.count, (unsigned long long)bytes,
        !header ? "(header unreadable)" : written.layout == QEF_LAYOUT_CODES ? "as 16-bit codes" : "F32",
        now_s() - started);
    return 0;
}

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
    const char *arg[2] = {NULL, NULL};
    int experts = 0, n = 0, bad = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--experts")) experts = 1;
        else if (argv[i][0] == '-' || n == 2) bad = 1;
        else arg[n++] = argv[i];
    }
    if (bad || !n) {
        fprintf(stderr, "usage: %s <model-dir> [<output>]   (default <model-dir>/canonical.qc)\n", argv[0]);
        fprintf(stderr, "       %s --experts <model-dir> [<output>]   (default <model-dir>/experts.qc)\n", argv[0]);
        return 2;
    }
    if (experts) return convert_experts_main(arg[0], arg[1]);
    char path[2048];
    if (arg[1]) snprintf(path, sizeof path, "%s", arg[1]);
    else snprintf(path, sizeof path, "%s/canonical.qc", arg[0]);
    Cfg c;
    memset(&c, 0, sizeof c);
    load_cfg(&c, arg[0]);
    int layers = c.n_layers;
    load_meta(&c, arg[0]);
    validate_cfg(&c, layers);
    shards S;
    st_init(&S, arg[0]);
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
