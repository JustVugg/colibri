/* Exercise the real initialization paths on two logical devices. Lavapipe is
 * sufficient: the existing probe override emulates a restricted mapped window.
 * Include the backend to inspect uploader state without adding a public API. */
#include "../backend_vulkan.c"
#include <assert.h>

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    static const struct { const char *probe, *forced; int staged; } cases[] = {
        {"1", NULL, 1}, {"100000000", NULL, 0},
        {"1", "0", 0}, {"100000000", "1", 1}
    };
    unsetenv("COLI_VK_HOST_VISIBLE_CAP_MB");
    unsetenv("COLI_VK_WINDOW_ASK_MB");
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        /* The shared probe still selects the primary's path correctly. */
        setenv("COLI_VK_WINDOW_PROBE_MB", cases[i].probe, 1);
        if (cases[i].forced) setenv("COLI_VK_STAGED", cases[i].forced, 1);
        else unsetenv("COLI_VK_STAGED");
        assert(coli_vk_init(spv));
        assert(g_up[0].on == cases[i].staged);
        assert(!g_up[0].later);
        coli_vk_shutdown();
        /* The primary stays mapped; only the secondary sees the probe answer. */
        setenv("COLI_VK_STAGED", "0", 1);
        assert(coli_vk_init(spv));
        setenv("COLI_VK_WINDOW_PROBE_MB", cases[i].probe, 1);
        if (cases[i].forced) setenv("COLI_VK_STAGED", cases[i].forced, 1);
        else unsetenv("COLI_VK_STAGED");
        assert(coli_vk_init_dev2(spv, 0));
        assert(!g_up[0].on);
        assert(g_up[1].on == cases[i].staged);
        assert(!g_up[1].later);
        if (cases[i].staged) {
            assert(g_up[1].dev == G2.dev && g_up[1].shared);
        } else {
            assert(!g_up[1].dev && !g_up[1].shared && !g_up[1].nfams);
        }
        ColiVkTensor *t = NULL;
        uint8_t *rows; size_t stride; float *scales;
        assert(coli_vk_tier_tensor_dev(1, &t, 1, 32, 4, 0, &rows, &stride, &scales));
        memset(rows, 1, stride * 4);
        for (int j = 0; j < 4; j++) scales[j] = 1.f;
        assert(coli_vk_tensor_commit(&t, 1));
        coli_vk_tensor_free(t);
        coli_vk_shutdown();
    }
    puts("dev2 mapped-window selection and uploads: PASS");
    return 0;
}
