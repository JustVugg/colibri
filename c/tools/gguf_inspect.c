/* gguf_inspect -- dump a GGUF's metadata, tensor table and per-type byte budget.
 * Read-only (pread): never touches the file's tensor bytes and never writes.
 * Tool for the IQ2_XXS Metal dispatch validation run; not wired into the Makefile.
 */
#include "../gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gguf_inspect <model.gguf>\n"); return 1; }
    gguf_ctx g;
    if (gguf_open(&g, argv[1])) { fprintf(stderr, "GGUF: %s\n", g.err); return 1; }
    printf("files=%d tensors=%d\n", g.n_files, g.n_tensors);
    for (int p = 0; p < g.n_files; p++) {
        struct stat st; long long sz = -1;
        if (!fstat(g.fds[p], &st)) sz = (long long)st.st_size;
        printf("file[%d] %s bytes=%.3fGiB\n", p, g.paths[p], sz / (double)(1LL<<30));
    }
    printf("\n-- metadata --\n");
    for (int i = 0; i < g.n_kv; i++) {
        gguf_kv *k = &g.kv[i];
        if (k->type == GGUF_T_STRING)
            printf("%s = \"%s\"\n", k->key, k->s);
        else if (k->type == GGUF_T_ARRAY) {
            const char *et =
                k->atype==GGUF_T_STRING ? "string" :
                k->atype==GGUF_T_F32 ? "f32" : k->atype==GGUF_T_I32 ? "i32" :
                k->atype==GGUF_T_F64 ? "f64" : k->atype==GGUF_T_I64 ? "i64" :
                k->atype==GGUF_T_U64 ? "u64" : k->atype==GGUF_T_I16 ? "i16" : "other";
            printf("%s = array<%s>[%llu]", k->key, et, (unsigned long long)k->n);
            if (k->atype == GGUF_T_STRING) {
                for (uint64_t j = 0; j < k->n && j < 32; j++) printf("%s\"%s\"", j ? ", " : " ", k->strs[j]);
                if (k->n > 32) printf(", ...");
            } else if (k->atype == GGUF_T_F32) {
                const float *a = (const float *)k->arr;
                for (uint64_t j = 0; j < k->n && j < 64; j++) printf("%s%g", j ? ", " : " ", a[j]);
                if (k->n > 64) printf(", ...");
            } else {
                for (uint64_t j = 0; j < k->n && j < 32; j++) printf("%s%lld", j ? ", " : " ", (long long)((int64_t *)k->arr)[j]);
                if (k->n > 32) printf(", ...");
            }
            printf("\n");
        } else {
            if (k->type == GGUF_T_F32) printf("%s = %g\n", k->key, k->v.f);
            else printf("%s = %lld\n", k->key, (long long)k->v.i);
        }
    }
    /* per-type histogram */
    printf("\n-- tensor type histogram --\n");
    int hist[GQ_TYPE_MAX]; memset(hist, 0, sizeof hist);
    long long bytes[GQ_TYPE_MAX]; memset(bytes, 0, sizeof bytes);
    for (int i = 0; i < g.n_tensors; i++) {
        if (g.t[i].type < 0 || g.t[i].type >= GQ_TYPE_MAX) continue;
        hist[g.t[i].type]++;
        bytes[g.t[i].type] += g.t[i].nbytes;
    }
    for (int t = 0; t < GQ_TYPE_MAX; t++)
        if (hist[t])
            printf("%-8s x%-5d %12.3f GiB\n", gq_type_name(t), hist[t], bytes[t] / (double)(1LL<<30));
    long long total = 0;
    for (int t = 0; t < GQ_TYPE_MAX; t++) total += bytes[t];
    printf("%-8s   %d tensors  %12.3f GiB\n", "TOTAL", g.n_tensors, total / (double)(1LL<<30));

    printf("\n-- tensors (name dims bytes type) --\n");
    for (int i = 0; i < g.n_tensors; i++) {
        gguf_tensor *t = &g.t[i];
        printf("%-60s [%6lld %6lld %3lld %2lld] %11lld  %s\n",
               t->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
               (long long)t->nbytes, gq_type_name(t->type));
    }
    gguf_close(&g);
    return 0;
}
