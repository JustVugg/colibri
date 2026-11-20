/* backend_sycl.h -- what the SYCL backend's units share (backend_cuda.dp.cpp, cuda_bd.dp.hpp,
 * cu_xb.dp.cpp): the error protocol over SYCL's exceptions, the sub-group shuffles the migrated
 * warp kernels take in place of CUDA's width-32 warp shuffles, and the launch of a kernel body
 * written over SG lanes at the device's sub-group size (coli_sycl.h's dispatch, 16 and 32).
 * Included after dpct's headers, before any kernel. */
#ifndef COLIBRI_BACKEND_SYCL_H
#define COLIBRI_BACKEND_SYCL_H

#include <sycl/sycl.hpp>
#include <atomic>
#include <cstdio>
#include "coli_sycl.h"

/* ---- errors ----------------------------------------------------------------------------
 * A SYCL call that fails throws where a CUDA call returned a code. COLI_SYCL_TRY(expr) runs one in
 * a try block and gives the code the units' cuda_ok takes (0 success, 1 a failure, logged with
 * the exception's text); every entry point is a function-try-block (dpct's) whose handler calls
 * sycl_caught, which logs and returns the failure the ABI documents. Either sets the unit's lost
 * flag, and so does an asynchronous error the runtime delivers to a queue's handler instead of
 * throwing (coli_sycl_lost, read by fault_injected): the backend's fault_injected then refuses
 * every compute entry, as it does under COLI_GPU_FAIL_AFTER, and the engine keeps its CPU paths. */
static std::atomic<int> g_sycl_lost{0};   /* this unit saw a SYCL exception (atomic: the FP8 finish threads catch too) */
static inline int sycl_caught(const sycl::exception &e, const char *where) {
    std::fprintf(stderr, "[CUDA] %s: SYCL exception: %s\n", where, e.what());
    g_sycl_lost = 1;
    return 0;
}
#define COLI_SYCL_TRY(expr) \
    ([&]() { try { expr; return 0; } catch (sycl::exception const &e) { sycl_caught(e, "call"); return 1; } }())

/* cudaMalloc / cudaHostAlloc: sycl::malloc_device and malloc_host answer a request the device or the
 * host cannot meet with NULL (the specification's failure form; the Level Zero adapter throws nothing
 * for it, as test_backend_cuda's deliberate 16 TB ask showed on the first card, MIGRATION.md section
 * 13.6), where cudaMalloc returned cudaErrorMemoryAllocation. COLI_SYCL_ALLOC(lhs, expr, n) runs the
 * allocation `lhs = expr` as COLI_SYCL_TRY does and gives COLI_SYCL_NOMEM for a NULL where n (the
 * bytes or elements asked) is not zero, and for an exception whose code is errc::memory_allocation:
 * cuda_ok then says "out of memory" in cudaGetErrorString's words, and nothing is lost, as nothing
 * was for the .cu (a copy into the NULL would have thrown, and that exception marked the backend lost:
 * every later entry refused, which the test's "must not poison later launches" check caught). Any
 * other exception is sycl_caught's, as before. A request of zero bytes may answer NULL and is not a
 * failure, as cudaMalloc(0) was not. */
#define COLI_SYCL_NOMEM 2
static inline int sycl_nomem(const sycl::exception &e) {
    if (e.code() == sycl::errc::memory_allocation) return COLI_SYCL_NOMEM;
    sycl_caught(e, "call");
    return 1;
}
#define COLI_SYCL_ALLOC(lhs, expr, n) \
    ([&]() { try { lhs = expr; return (n) && !(lhs) ? COLI_SYCL_NOMEM : 0; } \
             catch (sycl::exception const &e) { return sycl_nomem(e); } }())

/* ---- the shuffles ----------------------------------------------------------------------
 * __shfl_down_sync / __shfl_xor_sync over a full mask became sub-group operations over the
 * kernel's SG lanes (the required sub-group size of the launch below): a reduction tree of
 * log2(SG) steps where the CUDA code took 5 over 32 lanes. Every call site is in converged flow
 * (a sub-group's early returns are uniform: the row it owns is past the end for all its lanes).
 * A lane shifted past the end gets an unspecified value, which only a lane that is never read
 * sees (the trees read lane 0). */
template <typename T> static inline T shfl_down(T v, int o) {
    return sycl::shift_group_left(sycl::ext::oneapi::this_work_item::get_sub_group(), v, (unsigned)o);
}
template <typename T> static inline T shfl_xor(T v, int o) {
    return sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), v, (unsigned)o);
}

/* ---- the launch of a warp-32-style kernel -------------------------------------------------
 * COLI_SG_LAUNCH(q, sg, grid, block, slm_bytes, kernel<..., SG>(args..., dpct_local)) instantiates
 * the call at SG 16 and 32 (coli_sycl_launch_sg) and runs the one `sg` names (the profile's
 * sg_default, ctx->sg): the body's SG is a constexpr, so the kernel's lane arithmetic and its
 * shuffle trees are compiled for exactly that width. `grid` is in blocks (a dpct::dim3 or a
 * sycl::range<3>), `block` the threads a block (256 in every kernel here), `slm_bytes` the dynamic
 * local memory the body reads as dpct_local (0 none). The arguments must be locals or copies (a
 * pointer into a host struct would be read on the device). */
#define COLI_SG_LAUNCH(q, sg, grid, block, slm_bytes, ...)                                                  \
    coli_sycl_fprof_ev(coli_sycl_launch_sg((q), (sg), (grid), sycl::range<3>(1, 1, (block)), (slm_bytes),   \
                        [=](sycl::nd_item<3>, auto sg_tag, uint8_t *dpct_local) {                           \
                            constexpr int SG = decltype(sg_tag)::value;                                     \
                            (void)dpct_local;                                                               \
                            __VA_ARGS__;                                                                    \
                        }))

#endif /* COLIBRI_BACKEND_SYCL_H */
