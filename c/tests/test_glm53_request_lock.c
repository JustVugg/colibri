#define GLM53_NO_MAIN
#include "../glm53.c"

int main(void) {
    ColiGlm53GpuOps no_gpu = {0};
    char err[128] = {0};

    coli_glm53_gpu_backend_init(&g_glm53_gpu_backend);
    if (!coli_glm53_gpu_backend_select(&g_glm53_gpu_backend,
                                       COLI_GLM53_GPU_MODE_CPU,
                                       no_gpu, -1,
                                       COLI_GLM53_GPU_REQUIRED_CAPS,
                                       err, sizeof(err))) return 1;

    if (!glm53_request_begin()) return 1;
    if (coli_glm53_gpu_backend_select(&g_glm53_gpu_backend,
                                      COLI_GLM53_GPU_MODE_CPU,
                                      no_gpu, -1,
                                      COLI_GLM53_GPU_REQUIRED_CAPS,
                                      err, sizeof(err))) return 1;
    glm53_request_end();
    coli_glm53_gpu_backend_destroy(&g_glm53_gpu_backend, no_gpu);
    return 0;
}
