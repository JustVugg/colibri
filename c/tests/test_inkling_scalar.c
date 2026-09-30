/* load_scalar reads one float into a stack variable. The capacity bound stops a
 * tensor declaring MORE than one value; a tensor declaring NONE wrote nothing and
 * the uninitialised stack float came back as the scale. It must be refused. */
#define main inkling_main_unused
#include "../inkling.c"
#undef main
#include "st_fixture.h"
static int child(int c, const char *dir){
    FxT t[1] = { {"s", "F32", c ? 0 : 1, c ? 0 : 4} };
    if (fx_write(dir, t, 1)) return 90;
    static Model m; memset(&m, 0, sizeof m);
    st_init(&m.S, dir);
    fprintf(stderr, "loaded; scale %g\n", load_scalar(&m, "s", 1.0f));
    return 0;
}
int main(int argc, char **argv){
    if (argc == 4 && !strcmp(argv[1], "--child")) return child(atoi(argv[2]), argv[3]);
    int fails = 0;
    for (int c = 0; c < 2; c++) {
        char dir[] = "test_inkling_scalar_XXXXXX", log[16384];
        if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
        int got = fx_child(argv[0], c, dir, log, sizeof log);
        int ok = c ? fx_refused(got, log, "s") : got == 0;
        if (!ok) { fails++; printf("FAIL: case %d: exit %d, want %s\n--- child stderr ---\n%s\n",
                                   c, got, c ? "a refusal" : "a load", log); }
        fx_cleanup(dir);
    }
    if (fails) { printf("test_inkling_scalar: %d failure(s)\n", fails); return 1; }
    printf("OK test_inkling_scalar: one-value scalar loads, zero-value scalar refused\n");
    return 0;
}
