/* The device target and draft head (V4_DEVICE_TARGET, V4_MTP_GPU_HEAD): Linux
 * only, so not part of the Windows coli_cuda_dsv4*.dll ABI (dsv4.def and
 * backend_loader_dsv4.c). Included from backend_cuda_dsv4.h. */
#ifndef COLIBRI_BACKEND_CUDA_DSV4_TARGET_H
#define COLIBRI_BACKEND_CUDA_DSV4_TARGET_H
/* Prepared device target. Weights are borrowed; workspace and history are owned.
 * Every operation is stream ordered. No host activation transfer inside a layer. */
Dsv4CudaTargetWorkspace *dsv4_cuda_target_workspace_create(int device, int max_rows);
void dsv4_cuda_target_workspace_free(Dsv4CudaTargetWorkspace *workspace);
int dsv4_cuda_target_input(Dsv4CudaTargetWorkspace *workspace, const float *input,
                          const int *tokens, int start, int rows);
int dsv4_cuda_target_transfer(Dsv4CudaTargetWorkspace *dst, Dsv4CudaTargetWorkspace *src, int rows);
int dsv4_cuda_target_output(Dsv4CudaTargetWorkspace *workspace, float *output, int rows);
Dsv4CudaTargetLayer *dsv4_cuda_target_layer_create(const Dsv4CudaTargetWeights *weights,
                                                const float *cosines, const float *sines);
void dsv4_cuda_target_layer_free(Dsv4CudaTargetLayer *layer);
int dsv4_cuda_target_past(Dsv4CudaTargetLayer *layer,const float *past,const int64_t *positions);
int dsv4_cuda_target_layer_forward(Dsv4CudaTargetLayer *layer, Dsv4CudaTargetWorkspace *workspace, int rows);
int dsv4_cuda_target_stage(Dsv4CudaTargetWorkspace *workspace, Dsv4CudaTargetLayer *const *layers,
    int count, int first_tap, int rows, int graph);
int dsv4_cuda_target_taps(Dsv4CudaTargetWorkspace *workspace, float *output, int rows, int first, int count);
Dsv4CudaDraftHead *dsv4_cuda_draft_head_create(Dsv4CudaTensor *head,
    const uint16_t *markov1,const uint16_t *markov2,const float *confidence,int rank);
void dsv4_cuda_draft_head_free(Dsv4CudaDraftHead *head);
int dsv4_cuda_draft_head(Dsv4CudaDraftHead *head,const float *hidden,int rows,int anchor,int *ids,float *confidence);
#endif
