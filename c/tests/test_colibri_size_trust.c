/* Loader destinations sized from one count and filled from the tensor's own
 * declared count. Before this fix each case below went wrong silently (run under
 * ASan to see the overflow cases); cases 0 and 2 must now be refused by name and
 * case 1 must load cleanly into a buffer of the right size:
 *   0. routed expert whose U8 .qs sidecars are not whole floats: fslab is
 *      sum(bytes)/4 floats, the reads landed past it, validation came after;
 *   1. a grouped-int4 QT reused for a tensor with a finer group size kept the
 *      coarse scale buffer and read the finer scale count into it;
 *   2. a full-precision tensor with fewer than O*I values left the tail of the
 *      destination uninitialised and loaded anyway. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include "st_fixture.h"
enum { EI = 8, ED = 8, GO = 2, GI = 128 };
static int child(int c, const char *dir){
    static Model m; memset(&m, 0, sizeof m);
    if (c == 0) {
        FxT t[6]; const char *suf[3] = {"gate_proj", "up_proj", "down_proj"};
        static char nm[6][96];
        for (int k = 0; k < 3; k++) {
            snprintf(nm[2*k], 96, "model.layers.0.mlp.experts.0.%s.weight", suf[k]);
            snprintf(nm[2*k+1], 96, "model.layers.0.mlp.experts.0.%s.weight.qs", suf[k]);
            t[2*k] = (FxT){nm[2*k], "U8", EI * ED, EI * ED};
            t[2*k+1] = (FxT){nm[2*k+1], "U8", 5, 5};
        }
        if (fx_write(dir, t, 6)) return 90;
        st_init(&m.S, dir);
        m.c.hidden = ED; m.c.moe_inter = EI; m.c.n_layers = 1; m.ebits = 8;
        ESlot s; memset(&s, 0, sizeof s); s.eid = -1;
        expert_load_impl(&m, 0, 0, &s, 1, 0);
        return 0;
    }
    if (c == 1) {
        FxT t[4] = { {"a", "U8", GO * GI / 2, GO * GI / 2}, {"a.qs", "F32", GO * 2, GO * 2 * 4},
                     {"b", "U8", GO * GI / 2, GO * GI / 2}, {"b.qs", "F32", GO * 8, GO * 8 * 4} };
        if (fx_write(dir, t, 4)) return 90;
        st_init(&m.S, dir);
        QT q; memset(&q, 0, sizeof q);
        qt_from_disk(&m, "a", GO, GI, 4, 0, &q);    /* gs 64: 2 scales per row */
        qt_from_disk(&m, "b", GO, GI, 4, 0, &q);    /* gs 16: 8 scales per row */
        return q.gs == 16 ? 0 : 94;
    }
    FxT t[1] = { {"w", "F32", GO * GI - 1, (GO * GI - 1) * 4} };
    if (fx_write(dir, t, 1)) return 90;
    st_init(&m.S, dir);
    QT q; memset(&q, 0, sizeof q);
    qt_from_disk(&m, "w", GO, GI, 16, 0, &q);
    fprintf(stderr, "loaded; last value %g (file holds none for it)\n", q.qf[GO * GI - 1]);
    return 0;
}
int main(int argc, char **argv){
    if (argc == 4 && !strcmp(argv[1], "--child")) return child(atoi(argv[2]), argv[3]);
    const char *name[3] = {"model.layers.0.mlp.experts.0.gate_proj.weight", "b", "w"};
    int fails = 0;
    for (int c = 0; c < 3; c++) {
        char dir[] = "test_colibri_size_trust_XXXXXX", log[16384];
        if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
        int got = fx_child(argv[0], c, dir, log, sizeof log);
        int ok = c == 1 ? got == 0 && !strstr(log, "Sanitizer") : fx_refused(got, log, name[c]);
        if (!ok) {
            fails++; printf("FAIL: case %d: exit %d, want %s %s\n--- child stderr ---\n%s\n",
                            c, got, c == 1 ? "a clean load of" : "a refusal naming", name[c], log);
        }
        fx_cleanup(dir);
    }
    if (fails) { printf("test_colibri_size_trust: %d failure(s)\n", fails); return 1; }
    printf("OK test_colibri_size_trust: mis-sized expert scales, reused int4 group size reallocated, short f32 refused\n");
    return 0;
}
