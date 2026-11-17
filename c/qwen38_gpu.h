/* Metal kernels for the Qwen3.8 engine, as plain C.
 *
 * They live in their own library (metal/qwen38.metal, built into
 * qwen38_metal.o) but run on qwen36's device and in qwen36's command batch:
 * q36_tensor holds their data, a q38 dispatch joins the encoder q36 dispatches
 * use, in the order they are called, and q36_gpu_flush/q36_gpu_sync commit and
 * wait for both. Every wrapper returns 1 when its work is enqueued and 0 when
 * it declines (a bad shape or size; nothing is enqueued). No wrapper syncs. */
#ifndef QWEN38_GPU_H
#define QWEN38_GPU_H
#include <stddef.h>
#include "qwen36_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif

/* 1 ready, 0 not: runs q36_gpu_init() if needed, then compiles the qwen38
 * library on the same device (32-lane SIMD groups required). */
int q38_gpu_init(void);

/* y[i] = x[i] for n floats. Plumbing test only. */
int q38_copy(q36_tensor *y, q36_tensor *x, size_t n);

#ifdef __cplusplus
}
#endif
#endif
