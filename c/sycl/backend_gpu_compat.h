/* backend_gpu_compat.h — the SYCL units' copy of c/backend_gpu_compat.h.
 *
 * The original is one GPU backend source for two vendors: compiled by nvcc it passes through to
 * the CUDA runtime (and mma.h, the tensor-core paths), by hipcc it maps that surface onto HIP. The
 * units here (backend_cuda.dp.cpp, cuda_chain.dp.cpp, cu_xb.dp.cpp) are dpct's migration of those
 * sources to SYCL: the CUDA runtime's surface is gone from them, the tensor-core kernels with it
 * (the matrix engines' GEMMs of coli_sycl.h take their place), and programmatic dependent launch
 * (griddepcontrol, sm_90) has no Intel form. What is left for this header to say:
 *   - no WMMA: the host gates that tested COLI_GPU_HAS_WMMA fall back as under COLI_HIP_NO_WMMA
 *     (the block and warp kernels), and the migrated units compiled from the COLI_GPU_PORTABLE_ONLY
 *     arm of the originals (MIGRATION.md) keep those gates, now constant;
 *   - COLI_PDL_WAIT() is nothing: a plain launch on an in-order queue orders the kernels;
 *   - the dpct helpers every unit includes, for the migrated names that remain (dim3, dp4a,
 *     byte_level_permute, atomic_fetch_add).
 * COLI_GPU_PORTABLE_ONLY, the originals' gate the migration ran under, is harmless here, defined
 * or not. */
#ifndef COLIBRI_BACKEND_GPU_COMPAT_H
#define COLIBRI_BACKEND_GPU_COMPAT_H

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#define COLI_GPU_HAS_WMMA        0
#define COLI_GPU_HAS_S4_WMMA     0
#define COLI_PDL_WAIT()          ((void)0)

#endif /* COLIBRI_BACKEND_GPU_COMPAT_H */
