/* Dumps what gguf.h reads, for tests/test_gguf_c.py:
 *   test_gguf_c FILE            -> "OK", kv lines, tensor lines (with a float checksum)
 *   test_gguf_c FILE TENSOR     -> the tensor as float32 on stdout
 *   test_gguf_c --join DIR PATH -> the path an engine opens for gguf.json's PATH in DIR
 *   test_gguf_c --concurrent FILE TENSOR -> "OK" if 8 threads reading random slices of the
 *                               tensor at once get the bytes of one serial read */
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include "../gguf.h"

int main(int argc, char **argv) {
#ifdef _WIN32   /* the blocks and floats go through stdin/stdout: no CRLF translation */
    _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc < 2) return 2;
    if (argc == 4 && !strcmp(argv[1], "--join")) {
        char out[4096]; gguf_path_join(out, sizeof out, argv[2], argv[3]);
        printf("%s\n", out); return 0;
    }
    gguf_ctx g;
    if (argc == 4 && !strcmp(argv[1], "--concurrent")) {
        if (gguf_open(&g, argv[2])) { printf("ERR %s\n", g.err); return 1; }
        const gguf_tensor *t = gguf_find(&g, argv[3]);
        if (!t || t->nbytes < 1) { fprintf(stderr, "no tensor %s\n", argv[3]); return 3; }
        uint8_t *ref = malloc((size_t)t->nbytes);
        if (!ref || gguf_read(&g, t, ref, 0, t->nbytes)) { fprintf(stderr, "read failed\n"); return 4; }
        int bad = 0;
        #pragma omp parallel for num_threads(8) schedule(static, 1) reduction(+:bad)
        for (int i = 0; i < 4096; i++) {
            uint32_t x = 0x9E3779B9u * (uint32_t)(i + 1); x ^= x >> 15; x *= 0x85EBCA6Bu; x ^= x >> 13;
            int64_t off = (int64_t)(x % (uint32_t)t->nbytes), n = 1 + (int64_t)((x >> 7) % 97u);
            if (n > t->nbytes - off) n = t->nbytes - off;
            uint8_t buf[97];
            bad += gguf_read(&g, t, buf, off, n) || memcmp(buf, ref + off, (size_t)n);
        }
        printf(bad ? "MISMATCH %d\n" : "OK\n", bad); free(ref); gguf_close(&g); return bad != 0;
    }
    if (gguf_open(&g, argv[1])) { printf("ERR %s\n", g.err); return 1; }
    if (argc == 3) {
        const gguf_tensor *t = gguf_find(&g, argv[2]);
        if (!t) { fprintf(stderr, "no tensor %s\n", argv[2]); return 3; }
        int64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
        float *f = malloc((size_t)n * 4);
        if (!f || gguf_read_f32(&g, t, f)) { fprintf(stderr, "read failed\n"); return 4; }
        fwrite(f, 4, (size_t)n, stdout); free(f); gguf_close(&g); return 0;
    }
    printf("OK files=%d kv=%d tensors=%d\n", g.n_files, g.n_kv, g.n_tensors);
    for (int i = 0; i < g.n_kv; i++) {
        gguf_kv *k = &g.kv[i];
        printf("KV %s %d ", k->key, k->type);
        if (k->type == GGUF_T_STRING) printf("s:%s", k->s);
        else if (k->type == GGUF_T_ARRAY) {
            printf("a:%d:%llu", k->atype, (unsigned long long)k->n);
            if (k->atype == GGUF_T_STRING && k->n) printf(":%s:%s", k->strs[0], k->strs[k->n - 1]);
        } else if (k->type == GGUF_T_F32 || k->type == GGUF_T_F64) printf("f:%.9g", k->v.f);
        else printf("i:%lld", (long long)gguf_get_int(&g, k->key, -1));
        printf("\n");
    }
    for (int i = 0; i < g.n_tensors; i++) {
        gguf_tensor *t = &g.t[i];
        printf("T %s %s %lld %lld %lld %lld part=%d bytes=%lld\n", t->name, gq_type_name(t->type),
               (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], t->file, (long long)t->nbytes);
    }
    gguf_close(&g);
    return 0;
}
