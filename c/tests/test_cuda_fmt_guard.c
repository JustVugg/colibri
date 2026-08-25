/* The CUDA generic-decoder format guard, unit-tested WITHOUT a GPU or a CUDA
 * toolchain.
 *
 * THE DEFECT THIS PINS. backend_cuda.cu's weight_at() decoded fmt 0/1/2/4
 * explicitly and then FELL THROUGH to a 2-bit decode for everything else, with
 * no guard. So a tensor in any other format handed to the absorb or
 * grouped-expert kernels came back as plausible numbers derived from reading
 * its bytes two bits at a time. The CPU functions doing the same job on the
 * same tensor -- qt_addrow and qt_matvec_rows in colibri.c -- both exit(1)
 * naming the function and the fmt. Two backends, identical unsupported input,
 * one refusing and one fabricating.
 *
 * weight_at now decodes fmt=3 in an explicit branch and __trap()s on the
 * fall-through, and the host-side gate that keeps unsupported formats off the
 * device reads its admissible set from the SAME predicate rather than
 * restating it (`fmt <= 4`, which also admitted negative fmt). That predicate,
 * coli_cuda_weight_at_supported, is an ordinary C function in backend_cuda.h,
 * so its truth table is checkable on any host -- the same arrangement
 * colibri.c uses for metal_fused_fmt_ok ("Defined unconditionally (not #ifdef
 * COLI_METAL) so the CPU test build can unit-test the truth table").
 *
 * WHAT THIS TEST CANNOT DO, stated rather than implied: it does not execute
 * weight_at, because that is device code. It pins the predicate the host gates
 * consult and the boundary of the set weight_at's explicit branches cover. The
 * device-side __trap() firing on a real launch needs CUDA hardware and is
 * verified there, not here.
 *
 * ENUMERATED, NOT SAMPLED: every fmt value a container can carry (0..8 --
 * qt_resolve_fmt's own documented return set, colibri.c) is asserted by name,
 * plus the out-of-range values a corrupt or hostile descriptor could present. */
#include "../backend_cuda.h"

#include <stdio.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

int main(void) {
    /* Decodable by weight_at's explicit branches. */
    CHECK(coli_cuda_weight_at_supported(0));   /* f32           */
    CHECK(coli_cuda_weight_at_supported(1));   /* int8-row      */
    CHECK(coli_cuda_weight_at_supported(2));   /* int4 nibbles  */
    CHECK(coli_cuda_weight_at_supported(3));   /* int2          */
    CHECK(coli_cuda_weight_at_supported(4));   /* grouped int4  */

    /* NOT decodable. Each of these has a real in-tree meaning, and each used to
     * be read as int2 by the fall-through. fmt=5 and fmt=6 are the ones that
     * could already reach a CUDA tensor; fmt=8 is the one this branch is about;
     * fmt=7 (MXFP4) has its own quant_matmul branch and never routes here. */
    CHECK(!coli_cuda_weight_at_supported(5));  /* int3-g64      */
    CHECK(!coli_cuda_weight_at_supported(6));  /* E8/IQ3        */
    CHECK(!coli_cuda_weight_at_supported(7));  /* MXFP4         */
    CHECK(!coli_cuda_weight_at_supported(8));  /* fp8-e4m3-b128 */

    /* Out of range in both directions. The negative cases are the ones the
     * previous `fmt <= 4` host gate admitted: a descriptor whose fmt field is
     * garbage passed that inequality and then fell through weight_at into the
     * int2 decode. */
    CHECK(!coli_cuda_weight_at_supported(-1));
    CHECK(!coli_cuda_weight_at_supported(-1000));
    CHECK(!coli_cuda_weight_at_supported(9));
    CHECK(!coli_cuda_weight_at_supported(100));   /* PRIVATE ORDINAL BLOCK base */
    CHECK(!coli_cuda_weight_at_supported(1 << 30));

    /* The set is exactly {0,1,2,3,4} and nothing else in a wide sweep -- so a
     * later edit that widens the predicate has to change this line too. */
    for (int fmt = -2048; fmt <= 2048; fmt++) {
        int expect = (fmt >= 0 && fmt <= 4);
        if (!!coli_cuda_weight_at_supported(fmt) != expect) {
            printf("FAIL fmt=%d: supported=%d, expected %d\n",
                   fmt, coli_cuda_weight_at_supported(fmt), expect);
            fails++;
        }
    }

    if (fails) { printf("cuda fmt guard tests: %d FAILED\n", fails); return 1; }
    printf("cuda fmt guard tests: ok\n");
    return 0;
}
