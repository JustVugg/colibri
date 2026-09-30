/* DeepSeek-V4.1 loader destinations sized from the config and filled from the
 * tensor's own declared count. Each case must be refused by name; before this
 * fix none was (run under ASan to see case 0):
 *   0. a small f32 tensor (norm/bias) declaring more values than the config:
 *      wf_load wrote them all, then compared the count;
 *   1. an fp8 scale sidecar shorter than its tile count: the missing scales
 *      stayed uninitialised;
 *   2. a routed-expert tensor shorter than its slot: the slot tail kept whatever
 *      it held before. */
#define main v41_main_unused
#include "../deepseek_v41.c"
#undef main
#include "st_fixture.h"
enum { WO = 256, WI = 256, TILES = (WO / FP8_TILE) * (WI / FP8_TILE), DIM = 64, INTER = 32 };
static int child(int c, const char *dir){
    static shards S;
    if (c == 0) {
        FxT t[1] = { {"n.weight", "F32", 256, 256 * 4} };
        if (fx_write(dir, t, 1)) return 90;
        st_init(&S, dir);
        WF w; wf_load(&S, &w, "n.weight", 4);
        return 0;
    }
    if (c == 1) {
        FxT t[2] = { {"x.weight", "U8", WO * WI, WO * WI}, {"x.scale", "U8", TILES - 1, TILES - 1} };
        if (fx_write(dir, t, 2)) return 90;
        st_init(&S, dir);
        W8 w; w8_load(&S, &w, "x.weight", WO, WI);
        fprintf(stderr, "loaded; scale[%d] = 0x%02x (file holds none for it)\n", TILES - 1, w.s[TILES - 1]);
        return 0;
    }
    static char nm[6][96]; FxT t[6];
    const char *suf[3] = {"w1", "w3", "w2"};
    for (int k = 0; k < 3; k++) {
        int64_t wb = (int64_t)INTER * DIM / 2, sb = k < 2 ? (int64_t)INTER * (DIM / 32) : (int64_t)DIM * (INTER / 32);
        snprintf(nm[2*k], 96, "layers.0.ffn.experts.0.%s.weight", suf[k]);
        snprintf(nm[2*k+1], 96, "layers.0.ffn.experts.0.%s.scale", suf[k]);
        t[2*k] = (FxT){nm[2*k], "U8", wb, wb};
        t[2*k+1] = (FxT){nm[2*k+1], "U8", k == 2 ? sb - 1 : sb, k == 2 ? sb - 1 : sb};
    }
    if (fx_write(dir, t, 6)) return 90;
    static Model m; memset(&m, 0, sizeof m);
    st_init(&m.S, dir);
    m.c.dim = DIM; m.c.moe_inter = INTER;
    Slot s; memset(&s, 0, sizeof s);
    s.w1 = malloc(INTER * DIM / 2); s.w3 = malloc(INTER * DIM / 2); s.w2 = malloc(DIM * INTER / 2);
    s.s1 = malloc(INTER * (DIM / 32)); s.s3 = malloc(INTER * (DIM / 32)); s.s2 = malloc(DIM * (INTER / 32));
    ExpertRead list[V41_EXPERT_TENSORS];
    int n = expert_read_list(&m, "layers", 0, 0, &s, list);
    for (int i = 0; i < n; i++) st_read_raw_cap(&m.S, list[i].name, list[i].dest, list[i].size, 0);
    int last = DIM * (INTER / 32) - 1;
    fprintf(stderr, "loaded; s2[%d] = 0x%02x (file holds none for it)\n", last, s.s2[last]);
    return 0;
}
int main(int argc, char **argv){
    if (argc == 4 && !strcmp(argv[1], "--child")) return child(atoi(argv[2]), argv[3]);
    const char *name[3] = {"n.weight", "x.scale", "layers.0.ffn.experts.0.w2.scale"};
    int fails = 0;
    for (int c = 0; c < 3; c++) {
        char dir[] = "test_v41_size_trust_XXXXXX", log[16384];
        if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
        int got = fx_child(argv[0], c, dir, log, sizeof log);
        if (!fx_refused(got, log, name[c]) && !(c == 0 && got == 1 && strstr(log, "n.weight") &&
                                               strstr(log, "exceeds destination capacity") &&
                                               !strstr(log, "Sanitizer"))) {
            fails++; printf("FAIL: case %d: exit %d, want a refusal naming %s\n--- child stderr ---\n%s\n",
                            c, got, name[c], log);
        }
        fx_cleanup(dir);
    }
    if (fails) { printf("test_v41_size_trust: %d failure(s)\n", fails); return 1; }
    printf("OK test_v41_size_trust: oversized norm, short fp8 scales and short expert tensors refused\n");
    return 0;
}
