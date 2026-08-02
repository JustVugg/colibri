#ifndef COLIBRI_BACKEND_CUDA_DSV4_FLASHINFER_H
#define COLIBRI_BACKEND_CUDA_DSV4_FLASHINFER_H
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
int dsv4_flashinfer_grouped(const void *descriptors,int count,int O,int I,int device,cudaStream_t stream);
void dsv4_flashinfer_shutdown(void);
#ifdef __cplusplus
}
#endif
#endif
