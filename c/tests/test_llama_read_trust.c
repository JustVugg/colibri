/* llama.c's load_t must check a tensor's declared element count against the
 * count the config implies BEFORE it allocates or reads. The forward pass reads
 * every weight with CONFIG dims: a short tensor sized from its own header was
 * read past its end at the first rmsnorm_row() (heap overflow under ASan), and
 * an oversized one would be silently truncated.
 * Case 0: right-sized -- loads, and the forward's own consumer reads it cleanly.
 * Case 1: one element where the config implies D -- refused by name.
 * Case 2: 2*D elements where the config implies D -- refused by name. */
#define main llama_main_unused
#include "../llama.c"
#undef main
#include "st_fixture.h"

enum { D = 64 };
static const int64_t sizes[3] = { D, 1, 2 * D };

static int child(int c, const char *dir){
    FxT t[1] = { {"w", "F32", sizes[c], sizes[c] * 4} };
    if (fx_write(dir, t, 1)) return 90;
    static Model m; memset(&m, 0, sizeof m);
    st_init(&m.S, dir);
    float *w = load_t(&m, "w", D), x[D], out[D];
    for (int i = 0; i < D; i++) x[i] = 1.f;
    rmsnorm_row(out, x, w, D, 1e-6f);   /* the forward pass's own consumer, at config D */
    fprintf(stderr, "loaded; out[0] %g\n", out[0]);
    free(w);
    return 0;
}

int main(int argc, char **argv){
    if (argc == 4 && !strcmp(argv[1], "--child")) return child(atoi(argv[2]), argv[3]);
    int fails = 0;
    for (int c = 0; c < 3; c++) {
        char dir[] = "test_llama_read_trust_XXXXXX", log[16384];
        if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
        int got = fx_child(argv[0], c, dir, log, sizeof log);
        int ok = c ? fx_refused(got, log, "w") : got == 0 && !strstr(log, "Sanitizer");
        if (!ok) { fails++; printf("FAIL: case %d (%lld elements, config %d): exit %d, want %s\n"
                                   "--- child stderr ---\n%s\n", c, (long long)sizes[c], D, got,
                                   c ? "a refusal naming w" : "a clean load", log); }
        fx_cleanup(dir);
    }
    if (fails) { printf("test_llama_read_trust: %d failure(s)\n", fails); return 1; }
    printf("OK test_llama_read_trust: right-sized tensor loads, short and oversized tensors refused before any config-sized read\n");
    return 0;
}
