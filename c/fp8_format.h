/* fmt=8 (fp8-e4m3-b128) block geometry -- the single definition site for the
 * 128x128 scale-block edge, shared by the CPU side (quant.h: matmul_fp8,
 * e4m3 dequant plumbing; colibri.c: qt_addrow/qt_matvec_rows, qt_from_disk)
 * and the CUDA backend's three converted sites (backend_cuda.cu:
 * absorb_scale, quant_matmul's fmt=8 branch, the upload-time ng/scale_count
 * computation); the f8-warp and f8-group kernels there keep their own
 * 128 / >>7 literals, pinned against drift by backend_cuda.cu's
 * static_assert(FP8_BLOCK == 128). Before this header the two backends
 * agreed by IDENTICAL LITERALS restated in each file, so an edit to one side
 * could not break the other side's build -- only a full-scale CPU-vs-CUDA
 * parity run would have noticed. Kept deliberately tiny (no LUTs, no
 * functions with OpenMP pragmas, no intrinsics) so the CUDA translation unit
 * can include it without dragging in quant.h.
 *
 * The in-repo FP8 tests are NOT independent of this constant: the reference
 * decoders in test_fp8_passthrough.c, test_fp8_load.c, test_fp8_e2e_loader.c,
 * test_qwen38_native_weights.c, test_qt_addrow.c and test_shard_kvb_refuse.c
 * all index with FP8_BLOCK/fp8_nblk via quant.h, so they move in lockstep
 * with an edit here (test_backend_metal.mm's ref_fp8_nblk and
 * test_backend_cuda.cu's fmt=8 reference keep their own literals, on
 * purpose). An edit to FP8_BLOCK is a format change, not a tunable -- those
 * tests cannot catch a wrong edit on their own. */
#ifndef COLI_FP8_FORMAT_H
#define COLI_FP8_FORMAT_H

#include <stdint.h>

#define FP8_BLOCK 128

/* Blocks covering n elements: ceil(n/FP8_BLOCK). Host-side helper (device
 * code uses the FP8_BLOCK macro arithmetic directly -- this is not decorated
 * for device compilation on purpose, to keep the header plain C). */
static inline int64_t fp8_nblk(int n){ return ((int64_t)n + FP8_BLOCK - 1) / FP8_BLOCK; }

#endif /* COLI_FP8_FORMAT_H */
