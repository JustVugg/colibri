/* Generic resident dense matrices and the placer that feeds them, on the
 * fake backend: any component name an engine offers gets a decision per
 * offer (not only lmhead/dnproj), qt_place_of answers by name and layer,
 * qt_dense_init uploads int8 per-row (fmt 1) and hands back a handle, a
 * failing matmul falls back to the CPU once and stays there, and shutdown
 * releases the tensors. No GPU, no toolkit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

enum { NL = 2, NE = 8, D = 64, IH = 32, TOPK = 2 };

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1);
    setenv("COLI_PLACE", "", 1);                    /* "" == unset == auto; unsetenv has no UCRT64 shim */
    fake_ndev = 1; fake_uploads = 0;

    /* room for 8 experts plus a little; each offer is 1 expert worth of bytes */
    size_t exp_bytes = 3 * dev_alloc_footprint((size_t)D * IH / 2) + 3 * dev_alloc_footprint((2 * IH + D) / 3 * sizeof(float));
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(8 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);

    printf(" 1. offers by name, decision per offer\n");
    qt_trunk_offer("lmhead", 0, exp_bytes);
    qt_trunk_offer("dnqkv", 0, exp_bytes); qt_trunk_offer("dnz", 0, exp_bytes);
    qt_trunk_offer("dnqkv", 1, exp_bytes); qt_trunk_offer("dnz", 1, exp_bytes);
    qt_trunk_offer("hcmu", 1, exp_bytes);
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts (int4 mode, cap == n_experts)");
    check(qt_place_of("lmhead", 0) == 0, "lmhead placed on the one device");
    check(qt_place_of("dnqkv", 0) == 0 && qt_place_of("dnz", 1) == 0, "arbitrary names are placed by name and layer");
    check(qt_place_of("hcmu", 1) == 0, "a name the tier never heard of before is placed like any other");
    check(qt_place_of("dnqkv", 5) == QT_PLACE_CPU, "an unoffered layer stays on the CPU");
    check(qt_place_of("nothing", 0) == QT_PLACE_CPU, "an unoffered name stays on the CPU");
    pthread_mutex_lock(&G.mx);
    check(G.budget[0] < 8 * exp_bytes, "placed trunk bytes come out of the expert budget");
    pthread_mutex_unlock(&G.mx);

    printf(" 2. dense init uploads int8 per row and returns a handle\n");
    enum { O = 48, I = 96 };
    static int8_t q[O * I]; static float sc[O];
    for (int i = 0; i < O * I; i++) q[i] = (int8_t)(i % 251 - 125);
    for (int r = 0; r < O; r++) sc[r] = 0.01f * (r + 1);
    int before = fake_uploads;
    int h = qt_dense_init(q, sc, I, O, qt_place_of("dnqkv", 0));
    check(h >= 0, "handle returned for a placed matrix");
    check(fake_uploads == before + 1, "one upload per matrix");
    check(last_fmt == 1 && last_bytes == (size_t)I * O, "fmt 1 (int8 per row), one byte per element");
    check(qt_dense_init(q, sc, I, O, QT_PLACE_CPU) == -1, "a CPU placement gets no handle");
    check(qt_dense_init(NULL, sc, I, O, 0) == -1, "no bytes, no handle");
    int h2 = qt_dense_init(q, sc, I, O, 0);
    check(h2 == h + 1 && qt_dense_count() == 2, "handles count up");

    printf(" 3. matmul goes to the backend by handle; unknown handles are refused\n");
    float x[I], y[O]; for (int i = 0; i < I; i++) x[i] = 1.0f;
    int mm = fake_matmuls;
    check(qt_dense_matmul(h, y, x, I, O) == 1 && fake_matmuls == mm + 1, "a placed handle answers through coli_cuda_matmul");
    check(qt_dense_matmul(h2, y, x, I, O) == 1 && fake_matmuls == mm + 2, "so does the second handle");
    check(qt_dense_matmul(99, y, x, I, O) == 0 && qt_dense_matmul(-1, y, x, I, O) == 0 && fake_matmuls == mm + 2, "unknown handles are refused without a backend call");

    printf(" 4. shutdown releases the matrices\n");
    qt_shutdown();
    check(qt_dense_count() == 0, "dense handles released at shutdown");

    if (fails) { printf("test_qwen36_tier_dense: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_dense: ok\n");
    return 0;
}
