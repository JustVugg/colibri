/* CPU grouped-int4 (fmt=4) reference for tests/test_i4g_cuda.cu.
 *
 * Same reason as tests/mxfp4_ref.c: quant.h declares a `static _Thread_local`,
 * which nvcc's C++ front end rejects, so the reference is compiled as C and
 * exposed as one symbol. The point is to compare against the code the engine
 * actually runs, not a C++-friendly copy that can drift from it. */
#include "../quant.h"

void i4g_ref(float *y, const float *x, const unsigned char *q4,
             const float *scale, int S, int I, int O, int gs) {
    matmul_i4_grouped(y, x, q4, scale, S, I, O, gs);
}

/* int8 per-row (fmt=1), the format K3's MLA projections use at K3_MLA_BITS=8. */
void i8_ref(float *y, const float *x, const signed char *q8, const float *scale,
            int S, int I, int O) {
    matmul_q(y, x, q8, scale, S, I, O);
}
