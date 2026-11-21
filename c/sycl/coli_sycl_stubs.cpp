/* coli_sycl_stubs.cpp — weak "unsupported" definitions of coli_sycl.h's XMX GEMM family
 * (section 4), so that each SYCL unit links before the real implementations exist. Every
 * definition is weak: the unit that implements a function (coli_sycl_xmx.cpp) replaces it at
 * link time without touching this file. The stubs decline everything: the GEMMs return 0 and
 * the guards 0, which is what the callers' fallbacks expect (the warp and block kernels take
 * the prompt rows). Sections 1 and 2 (the profile, the devices, the arena) are implemented by
 * coli_sycl_profile.cpp and coli_sycl_profile_rules.cpp and have no stub any more. */
#include "coli_sycl.h"

#define COLI_SYCL_WEAK __attribute__((weak))

COLI_SYCL_WEAK int coli_sycl_xmx_gemm(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_gemm &g) {
    (void)p; (void)q; (void)g; return 0;
}
COLI_SYCL_WEAK int coli_sycl_xmx_gemm_glu(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_gemm &gate,
                                          const void *Wu, const float *scales_u) {
    (void)p; (void)q; (void)gate; (void)Wu; (void)scales_u; return 0;
}
COLI_SYCL_WEAK int coli_sycl_xmx_gemm_ok(const coli_sycl_profile &p, int fmt, int gs, int K, int N) {
    (void)p; (void)fmt; (void)gs; (void)K; (void)N; return 0;
}
COLI_SYCL_WEAK int coli_sycl_xmx_moe(const coli_sycl_profile &p, sycl::queue &q, const CcExpert *tab, const CcMoe &m,
                                     const coli_sycl_moe_sort &sort, const float *x, float *hid, int phase) {
    (void)p; (void)q; (void)tab; (void)m; (void)sort; (void)x; (void)hid; (void)phase; return 0;
}
COLI_SYCL_WEAK int coli_sycl_xmx_moe_ok(const coli_sycl_profile &p, const CcMoe &m) { (void)p; (void)m; return 0; }
COLI_SYCL_WEAK int coli_sycl_xmx_attn(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_attn &a) {
    (void)p; (void)q; (void)a; return 0;
}
COLI_SYCL_WEAK int coli_sycl_xmx_attn_ok(const coli_sycl_profile &p, int hd, int G, int kv_f16, int sel) {
    (void)p; (void)hd; (void)G; (void)kv_f16; (void)sel; return 0;
}
