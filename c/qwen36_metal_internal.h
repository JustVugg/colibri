/* qwen36_metal.m's device and command stream, for other Objective-C Metal
 * layers (qwen38_metal.m) that dispatch their own kernels into it. Not for C. */
#ifndef QWEN36_METAL_INTERNAL_H
#define QWEN36_METAL_INTERNAL_H
#ifndef __OBJC__
#error "qwen36_metal_internal.h is Objective-C only; C code uses qwen36_gpu.h"
#endif
#import <Metal/Metal.h>
#include <stdint.h>
#include "qwen36_gpu.h"

/* The device q36_gpu_init chose; nil before it. */
id<MTLDevice> q36_metal_device(void);
/* The encoder of the current q36 batch (opened if there is none), with pso set:
 * the same batch, ordering and concurrency rules as q36's own dispatches, so
 * work encoded here runs between the q36 dispatches encoded before and after. */
id<MTLComputeCommandEncoder> q36_metal_encoder(id<MTLComputePipelineState> pso);
/* t's buffer, at t's offset, at buffer index `index`. */
void q36_metal_bind(id<MTLComputeCommandEncoder> e, q36_tensor *t, int index);
/* 1 when t is non-NULL and holds at least `bytes`. */
int q36_metal_fits(q36_tensor *t, uint64_t bytes);

#endif
