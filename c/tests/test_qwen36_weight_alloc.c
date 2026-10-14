/* Vulkan's page-aligned q/h rows must use the matching release on Windows.
 * Track the allocation family even on Linux, where free would hide a mismatch. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include "../compat.h"

static void *aligned_rows[32];
static int nrows, aligned_calls, wrong_free;
static int weight_memalign(void **p, size_t al, size_t n) {
    int rc = posix_memalign(p, al, n);
    if (!rc) { if (nrows == 32) abort(); aligned_rows[nrows++] = *p; aligned_calls++; }
    return rc;
}
static int remove_row(void *p) {
    for (int i = 0; i < nrows; i++) if (aligned_rows[i] == p) {
        aligned_rows[i] = aligned_rows[--nrows]; return 1;
    }
    return 0;
}
static void weight_aligned_free(void *p) {
    if (p && !remove_row(p)) abort();
    compat_aligned_free(p);
}
static void weight_free(void *p) {
    if (p && remove_row(p)) { wrong_free++; compat_aligned_free(p); }
    else free(p);
}
#undef posix_memalign
#undef compat_aligned_free
#define posix_memalign weight_memalign
#define compat_aligned_free weight_aligned_free
#define free weight_free
#define QWEN36_NO_MAIN
#include "../qwen36.c"
#undef free

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
#ifdef COLI_VULKAN
/* Combine explicit import with host removal. The freed pages must never remain a
 * Vulkan tensor's backing storage, and aligned int8/f16 rows use the right release. */
static int check_device_only(void) {
    setenv("COLI_VK_DENSE_HOST", "0", 1);
    g_vk_ready = coli_vk_init_env_tier("qwen36", 0);
    CHECK(g_vk_ready);
    CHECK(coli_vk_dense_host_decide("qwen36", 1, 4096));
    g_vk_import = 1;
    for (int half = 0; half < 2; half++) {
        QW w = {0}; w.I = 64; w.O = 16;
        w.vk_name = strdup("test.dense.weight");
        float x[64], y[16], want[16];
        for (int i = 0; i < 64; i++) x[i] = 1.f;
        if (half) {
            w.h = q36_walloc(64 * 16 * sizeof(uint16_t));
            CHECK(w.h);
            for (int i = 0; i < 64 * 16; i++) w.h[i] = 0x3c00; /* 1.0 */
            for (int o = 0; o < 16; o++) want[o] = 64.f;
        } else {
            w.q = q36_walloc(64 * 16); w.sc = malloc(16 * sizeof(float));
            CHECK(w.q && w.sc);
            for (int o = 0; o < 16; o++) {
                w.sc[o] = 0.125f; want[o] = 0.f;
                for (int i = 0; i < 64; i++) {
                    w.q[o * 64 + i] = (i + o) % 7 - 3;
                    want[o] += w.q[o * 64 + i] * w.sc[o];
                }
            }
        }
        size_t bytes = 0; int n = 0;
        q36_dho_drop(&w, &bytes, &n);
        CHECK(n == 1 && bytes > 0 && w.vk_gone && !w.vk_imported);
        CHECK(!w.q && !w.h && !wrong_free && !nrows);
        CHECK(vk_dense_matmul(y, x, &w, 1, 64, 16));
        for (int o = 0; o < 16; o++) CHECK(fabsf(y[o] - want[o]) < 1e-4f);
        qw_free(&w);
    }
    coli_vk_shutdown();
    return 0;
}
#endif
int main(void) {
    QW w = {0};
    w.q = q36_walloc(64);
    w.h = q36_walloc(128);
    CHECK(w.q && w.h);
    qw_free(&w);
    CHECK(!wrong_free && !nrows && !w.q && !w.h);
    /* The int8 intermediate is released as soon as int4 packing succeeds. */
    setenv("COLI_DENSE_BITS", "4", 1);
    unsetenv("COLI_DENSE_INT4");
    unsetenv("COLI_DENSE_KEEP_I8");
    unsetenv("COLI_CUDA");
    float values[64 * 16];
    for (int i = 0; i < 64 * 16; i++) values[i] = (float)(i % 19 - 9) / 9.f;
    qw_quantize(values, 64, 16, "model.layers.0.self_attn.q_proj.weight", &w);
    CHECK(w.q4 && !w.q);
    qw_free(&w);
    CHECK(!wrong_free && !nrows);
#ifdef COLI_VULKAN
    const char *e = getenv("COLI_VULKAN");
    CHECK(aligned_calls == ((e && atoi(e)) ? 3 : 0));
    if (e && atoi(e)) CHECK(!check_device_only());
#else
    CHECK(aligned_calls == 0);
#endif
    puts("qwen36 weight allocation: matching release and independent device storage after host removal");
    return 0;
}
