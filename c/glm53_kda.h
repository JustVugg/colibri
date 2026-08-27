#ifndef COLIBRI_GLM53_KDA_H
#define COLIBRI_GLM53_KDA_H

/* One recurrent GLM-5.3 KDA step after the projection/forget-gate matmuls.
 * qkv is [3, heads, dim], conv_weight/window are [3, heads, dim, kernel],
 * log_decay is [heads, dim], beta is [heads], and state is
 * [heads, dim, dim]. All buffers are fp32; state and window are updated. */
int coli_glm53_kda_step(float *output, float *state, float *conv_window,
                        const float *qkv, const float *conv_weight,
                        const float *log_decay, const float *beta,
                        int heads, int dim, int kernel);

#endif
