/* CPU MXFP4 reference for tests/test_mxfp4_cuda.cu.
 *
 * quant.h is C: it declares `static _Thread_local QScratch g_qscratch`, which
 * nvcc's C++ front end rejects. Compiling the reference here as C and exposing
 * one symbol keeps the test comparing against the SAME code the engine runs,
 * rather than a C++-friendly re-implementation that could drift from it. */
#include "../quant.h"

void mxfp4_ref(float *y, const float *x, const unsigned char *q4,
               const unsigned char *e8s, int S, int I, int O) {
    matmul_mxfp4(y, x, q4, e8s, S, I, O);
}
