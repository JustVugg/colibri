/* colibri's ld() sizes its buffer from the tensor's declared count, while the forward pass
 * reads it with CONFIG dims. A tensor shorter than the config says loaded
 * cleanly and was read past its end at the first rmsnorm() (run under ASan to see
 * the overflow). Case 0 is a right-sized tensor and must load and be consumed
 * cleanly; case 1 is one value where the config implies D and must be refused
 * by name at load. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include "st_fixture.h"
enum { D = 64 };
static int child(int c, const char *dir){
    FxT t[1] = { {"w", "F32", c ? 1 : D, (c ? 1 : D) * 4} };
    if (fx_write(dir, t, 1)) return 90;
    static Model m; memset(&m, 0, sizeof m);
    st_init(&m.S, dir);
    float *w = ld(&m, "w", D), x[D], out[D];
    for (int i = 0; i < D; i++) x[i] = 1.f;
    rmsnorm(out, x, w, D, 1e-6f);   /* the forward pass's own consumer, at config D */
    fprintf(stderr, "loaded; out[0] %g\n", out[0]);
    free(w);
    return 0;
}
int main(int argc, char **argv){
    if (argc == 4 && !strcmp(argv[1], "--child")) return child(atoi(argv[2]), argv[3]);
    int fails = 0;
    for (int c = 0; c < 2; c++) {
        char dir[] = "test_colibri_read_trust_XXXXXX", log[16384];
        if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
        int got = fx_child(argv[0], c, dir, log, sizeof log);
        int ok = c ? fx_refused(got, log, "w") : got == 0 && !strstr(log, "Sanitizer");
        if (!ok) { fails++; printf("FAIL: case %d: exit %d, want %s\n--- child stderr ---\n%s\n",
                                   c, got, c ? "a refusal naming w" : "a clean load", log); }
        fx_cleanup(dir);
    }
    if (fails) { printf("test_colibri_read_trust: %d failure(s)\n", fails); return 1; }
    printf("OK test_colibri_read_trust: right-sized tensor loads, short tensor refused before a config-sized read\n");
    return 0;
}
