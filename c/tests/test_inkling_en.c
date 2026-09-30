/* inkling dense load path: a container shorter than the config's geometry must be
 * refused at matmul_w, not read past its end. Reverting the guard turns case 1 into a
 * heap-buffer-overflow under ASan, so the check is not decorative. */
#define main inkling_main_unused
#include "../inkling.c"
#undef main

static int child(int refuse) {
    float y[4] = {0}, x[4] = {0};
    int64_t en = refuse ? 8 : 16;         /* the caller always needs 16 */
    Wt w = {0};
    w.en = en;
    w.f = (float*)calloc((size_t)en, sizeof(float));
    matmul_w(y, x, w, 1, 4, 4);
    return 0;
}

/* Best-effort recursive removal: the child leaves a temp dir behind if its
 * matmul_w check fires exit(1) inside the call, and on a hung child the
 * parent must still reclaim its scratchpad. The earlier guard skipped
 * cleanup on early return, which is how 70k+ of these ended up under c/. */
static void rm_rf(const char *path) {
    if (!path || !*path) return;
    char cmd[32768];
    snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", path);
    int r = system(cmd);
    (void)r;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--child")) return child(atoi(argv[2]));

    int fails = 0;
    char dir[] = "test_inkling_en_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    char path[16384], cmd[65536];

    /* Resolve argv[0] to an absolute path so the recursive re-exec does not
     * rely on $PATH. */
    char self[4096] = {0};
    if (argv[0] && realpath(argv[0], self) && self[0] == '/') {
        /* absolute path */
    } else if (argv[0] && strchr(argv[0], '/')) {
        snprintf(self, sizeof self, "%s", argv[0]);
    } else {
        snprintf(self, sizeof self, "./tests/test_inkling_en");
    }

    for (int c = 0; c < 2; c++) {
        snprintf(path, sizeof path, "%s/log%d.txt", dir, c);
        snprintf(cmd, sizeof cmd, "'%s' --child %d >'%s' 2>&1", self, c, path);
        int got = system(cmd);
        int rc = WIFEXITED(got) ? WEXITSTATUS(got) : -1;
        char log[16384] = {0};
        FILE *f = fopen(path, "r");
        if (f) { size_t k = fread(log, 1, sizeof log - 1, f); log[k] = 0; fclose(f); }
        if (c == 0) {
            /* case 0: window holds exactly O*I elements -> matmul_w returns, no
             * "geometria incoerente" message, child exits 0. */
            if (rc != 0 || strstr(log, "geometria incoerente")) { fails++;
                printf("FAIL: right-sized window was not accepted (rc=%d)\n--- stderr ---\n%s\n", rc, log); }
        } else {
            /* case 1: window holds half of what the caller needs -> matmul_w
             * detects the geometry mismatch, prints the diagnostic, exits 1. */
            if (rc == 0 || !strstr(log, "geometria incoerente")) { fails++;
                printf("FAIL: short window was not refused (rc=%d)\n--- stderr ---\n%s\n", rc, log); }
        }
    }
    /* Always remove the scratch dir, success or failure. */
    rm_rf(dir);

    if (fails) { printf("test_inkling_en: %d failure(s)\n", fails); return 1; }
    puts("OK test_inkling_en: right-sized window loads, short window refused before a config-sized read");
    return 0;
}
