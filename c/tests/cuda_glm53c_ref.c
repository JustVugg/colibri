/* The CPU's references for tests/test_cuda_glm53c.cu, which nvcc compiles as C++ (these headers
 * are C): delta_attention.h's KDA step. */
#include "../delta_attention.h"
int glm53c_ref_kda_scratch(int H, int KD, int VD) { return coli_kda_scratch_floats(H, KD, VD); }
int glm53c_ref_kda_step(float *out, float *state, float *window, const float *qkv, const float *conv_w, const float *gate,
                        const float *beta, int H, int KD, int VD, int K, float neps, float *scratch) {
    return coli_kda_step(out, state, window, qkv, conv_w, gate, beta, H, KD, VD, K, neps, scratch);
}
