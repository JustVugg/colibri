/* fmt=8 (fp8-e4m3-b128) block geometry -- the single definition site for the
 * 128x128 scale-block edge. Every consumer reaches these symbols through
 * quant.h, which includes this header where the definitions used to live:
 * matmul_fp8 and the e4m3 dequant plumbing (quant.h), the qt_addrow /
 * qt_matvec_rows / qt_from_disk paths (colibri.c), and the native-FP8
 * reduction (qwen38.c / qwen38_core.h). Kept deliberately tiny (no LUTs, no
 * functions with OpenMP pragmas, no intrinsics) so a translation unit that
 * cannot include quant.h itself can still name the block edge instead of
 * restating 128 / >>7 as literals.
 *
 * The in-repo FP8 tests are NOT independent of this constant: the reference
 * decoders in test_fp8_passthrough.c, test_fp8_load.c, test_fp8_e2e_loader.c,
 * test_qwen38_native_weights.c and test_qt_addrow.c all index with
 * FP8_BLOCK/fp8_nblk via quant.h, so they move in lockstep with an edit here
 * (only test_backend_metal.mm's ref_fp8_nblk keeps its own literal, on purpose).
 * An edit to FP8_BLOCK is a format change, not a tunable -- those tests
 * cannot catch a wrong edit on their own. */
#ifndef COLI_FP8_FORMAT_H
#define COLI_FP8_FORMAT_H

#include <stdint.h>

#define FP8_BLOCK 128

/* Blocks covering n elements: ceil(n/FP8_BLOCK). Plain C on purpose --
 * nothing in this header may need decoration or a special front end, so it
 * stays includable from any translation unit. */
static inline int64_t fp8_nblk(int n){ return ((int64_t)n + FP8_BLOCK - 1) / FP8_BLOCK; }

#endif /* COLI_FP8_FORMAT_H */
