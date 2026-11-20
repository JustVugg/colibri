#ifndef COLI_SYCL_FP8_HOST_SILU_H
#define COLI_SYCL_FP8_HOST_SILU_H
#include <math.h>

/* Host equivalent of q38_silu(g)*u: preserve the float operations and expf
 * used by the CPU expert, including its stable negative-input branch. */
static inline float coli_fp8_host_silu_mul(float g, float u) {
    float z=expf(g>=0.f?-g:g);
    float sig=g>=0.f?1.f/(1.f+z):z/(1.f+z);
    float hidden=g*sig;
    return hidden*u;
}
#endif
