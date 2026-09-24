/* Exercise the actual engine dispatch without a GPU, including an older DLL. */
#define main kimi_main_unused
#include "../kimi_k3.c"
#undef main

static int fused_ok, legacy_ok, fused_calls, legacy_calls;
int coli_cuda_init(const int *devices, int n) { return 1; }
int coli_cuda_expert_mxfp4(float *y, const float *x,
        const unsigned char *gw, const unsigned char *gs,
        const unsigned char *uw, const unsigned char *us,
        const unsigned char *dw, const unsigned char *ds,
        int S, int D, int I, float b1, float b2) {
    fused_calls++;
    for (int i = 0; i < D; i++) y[i] = fused_ok ? 2.f : -999.f;
    return fused_ok;
}
int coli_cuda_matmul_mxfp4(float *y, const float *x, const unsigned char *q,
        const unsigned char *sc, int S, int I, int O) {
    legacy_calls++;
    for (int i = 0; i < O; i++) y[i] = 3.f;
    return legacy_ok;
}
int main(void) {
    Model m = {0}; m.c.latent = 3; m.c.moe_inter = 5;
    m.c.situ_b1 = 2.f; m.c.situ_b2 = 3.f;
    uint8_t w[16] = {0}, sc[5] = {0};
    float x[3] = {1,2,3}, out[3] = {4,4,4}, gate[5], up[5], hz[3];
    fused_ok = 1;
    if (!cuda_expert_apply(&m,w,sc,w,sc,w,sc,x,0.5f,out,gate,up,hz) ||
        fused_calls != 1 || legacy_calls || out[0] != 5.f || out[2] != 5.f) return 1;
    fused_ok = 0; legacy_ok = 1;
    if (!cuda_expert_apply(&m,w,sc,w,sc,w,sc,x,0.5f,out,gate,up,hz) ||
        legacy_calls != 3 || out[0] != 6.5f || out[2] != 6.5f) return 1;
    legacy_ok = 0;
    if (cuda_expert_apply(&m,w,sc,w,sc,w,sc,x,0.5f,out,gate,up,hz) ||
        out[0] != 6.5f || out[2] != 6.5f) return 1;
    puts("ok Kimi CUDA fused dispatch, legacy fallback, failed expert does not accumulate");
    return 0;
}
