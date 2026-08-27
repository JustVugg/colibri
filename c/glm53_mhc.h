#ifndef COLIBRI_GLM53_MHC_H
#define COLIBRI_GLM53_MHC_H

int coli_glm53_mhc_pre(float *collapsed, float *post, float *comb,
                       const float *streams, const float *function,
                       const float scale[3], const float *base,
                       int copies, int dimension, int iterations,
                       float norm_eps, float hc_eps);
int coli_glm53_mhc_post(float *output, const float *branch,
                        const float *streams, const float *post,
                        const float *comb, int copies, int dimension);

#endif
