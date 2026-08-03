/* fp8 passthrough END-TO-END loader harness.
 *
 * Neither test_fp8_load.c (hand-built wire-format C fixtures -- proves the
 * LOADER's own disambiguation logic in isolation, but the container bytes
 * are hand-authored, not tool-produced) nor test_fp8_repack.py (proves the
 * WRITER's Python-side output shape -- never invokes the C loader at all)
 * exercises the REAL round trip: a container the REAL
 * tools/repack_fp8_passthrough.py actually produced, fed into the REAL
 * st_init/qt_from_disk. This binary is the C-side half of that round trip;
 * tests/test_fp8_e2e_repack_load.py drives it end to end (builds a synthetic
 * FP8 checkpoint via tools/glm_fp8_emit.py, repacks it with the real tool
 * via subprocess -- the actual CLI, not a reimplementation -- then invokes
 * this binary against the real output directory).
 *
 * Usage: test_fp8_e2e_loader <container_dir> [name O I]...
 * For every (name,O,I) triple, calls the REAL qt_from_disk (the identical
 * function every model load uses) and asserts:
 *   (a) fmt==8 resolved (native fp8-e4m3-passthrough -- byte-arithmetic
 *       inference AND, since this tool stamps its output, metadata-stamp
 *       agreement, both exercised for real here, neither mocked);
 *   (b) the weight (q8) and scale (s) buffers are non-NULL;
 *   (c) every dequantized value (e4m3_decode(byte) * block scale) is finite
 *       -- catches a decode-table or block-index bug a pure byte-count
 *       check wouldn't. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <dir> [name O I]...\n", argv[0]);
        return 2;
    }
    if ((argc - 2) % 3 != 0) {
        fprintf(stderr, "args after <dir> must come in (name,O,I) triples (got %d)\n", argc - 2);
        return 2;
    }
    const char *dir = argv[1];
    int ntensors = (argc - 2) / 3;
    if (ntensors == 0) {
        fprintf(stderr, "no (name,O,I) triples given -- nothing to check\n");
        return 2;
    }

    static Model gm;
    memset(&gm, 0, sizeof gm);
    st_init(&gm.S, dir);

    int fails = 0;
    for (int i = 0; i < ntensors; i++) {
        const char *name = argv[2 + i * 3];
        int O = atoi(argv[3 + i * 3]);
        int I = atoi(argv[4 + i * 3]);

        QT t;
        memset(&t, 0, sizeof t);
        qt_from_disk(&gm, name, O, I, 8, 0, &t); /* THE REAL LOADER PATH -- not mocked */

        if (t.fmt != 8) {
            printf("FAIL %s: fmt=%d, expected 8 (native fp8-e4m3 passthrough)\n", name, t.fmt);
            fails++;
            continue;
        }
        if (!t.q8 || !t.s) {
            printf("FAIL %s: fmt=8 but q8=%p s=%p (expected both non-NULL)\n", name, (void *)t.q8, (void *)t.s);
            fails++;
            continue;
        }
        int64_t nblkI = fp8_nblk(I);
        int64_t bad = -1;
        for (int64_t o = 0; o < O && bad < 0; o++) {
            int64_t blkO = o / FP8_BLOCK;
            const float *scl = t.s + blkO * nblkI;
            for (int64_t ii = 0; ii < I; ii++) {
                int64_t bi = ii / FP8_BLOCK;
                float v = e4m3_decode(t.q8[o * (int64_t)I + ii]) * scl[bi];
                if (!isfinite(v)) {
                    bad = o * (int64_t)I + ii;
                    break;
                }
            }
        }
        if (bad >= 0) {
            printf("FAIL %s: non-finite dequantized value at flat index %lld\n", name, (long long)bad);
            fails++;
            continue;
        }
        printf("ok %s: fmt=8 O=%d I=%d, weights+scale loaded through the real loader, all-finite\n", name, O, I);
    }
    if (fails) {
        printf("fp8 e2e repack->load: %d/%d tensor(s) FAILED\n", fails, ntensors);
        return 1;
    }
    printf("fp8 e2e repack->load: ok (%d tensor(s))\n", ntensors);
    return 0;
}
