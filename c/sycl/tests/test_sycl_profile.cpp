/* The profile's decisions (coli_sycl_profile.h, coli_sycl_profile_rules.cpp) over fake
 * property sets: a Data Center GPU Max 1100 (pvc), an Arc B580 (bmg_g21), a Lunar Lake iGPU
 * (lnl), an Alchemist card (dg2: matrix engines at sub-group 8, which the kernels cannot take)
 * and an Intel GPU of an unknown generation; the defaults, the environment overrides, the
 * names, the shape flags. Host code only: no device, no SYCL runtime in the process (the unit
 * under test has no SYCL in it), so it runs anywhere a C++17 compiler is.
 *   make -C c sycl-profile-test   (or: g++ -std=c++17 -Isycl sycl/tests/test_sycl_profile.cpp
 *                                  sycl/coli_sycl_profile_rules.cpp -o test_sycl_profile) */
#include "../coli_sycl_profile.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

static int fails = 0, checks = 0;
#define CHECK(cond) do { checks++; if (!(cond)) { fails++; printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_STR(a, b) do { checks++; if (strcmp((a), (b))) { fails++; printf("  FAIL %s:%d: %s = \"%s\", want \"%s\"\n", __FILE__, __LINE__, #a, (a), (b)); } } while (0)

static const coli_sycl_env no_env = { nullptr, nullptr, nullptr };

/* the combinations the runtime lists on pvc and xe2, in its own encoding (read out of libsycl's
 * tables: each dimension fixed or ranged, {max_m 8, m 0, n 16, k K} is M up to 8 with N and K
 * fixed; the compound f16 16x16x16 is a row of three fixed sizes), and dg2's rows, the same with
 * N fixed at 8 (sub-group 8) */
static const coli_sycl_combo xmx16[] = {
    { 8, 0, 0, 0, 16, 16, COLI_SYCL_DT_BF16, COLI_SYCL_DT_BF16, COLI_SYCL_DT_F32 },
    { 8, 0, 0, 0, 16, 16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F32 },
    { 0, 0, 0, 16, 16, 16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F32 },
    { 8, 0, 0, 0, 16, 32, COLI_SYCL_DT_INT8, COLI_SYCL_DT_INT8, COLI_SYCL_DT_S32 },
    { 8, 0, 0, 0, 16, 32, COLI_SYCL_DT_OTHER, COLI_SYCL_DT_OTHER, COLI_SYCL_DT_S32 },   /* u8: no row */
    { 8, 0, 0, 0, 16, 8, COLI_SYCL_DT_TF32, COLI_SYCL_DT_TF32, COLI_SYCL_DT_F32 },
};
static const coli_sycl_combo xmx8[] = {
    { 8, 0, 0, 0, 8, 16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F32 },
    { 8, 0, 0, 0, 8, 32, COLI_SYCL_DT_INT8, COLI_SYCL_DT_INT8, COLI_SYCL_DT_S32 },
};
/* the same shapes as all-fixed rows (an encoding the runtime does not use, which the rule must
 * still read right) */
static const coli_sycl_combo xmx16_fixed[] = {
    { 0, 0, 0, 8, 16, 16, COLI_SYCL_DT_BF16, COLI_SYCL_DT_BF16, COLI_SYCL_DT_F32 },
    { 0, 0, 0, 8, 16, 16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F32 },
    { 0, 0, 0, 16, 16, 16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F32 },
    { 0, 0, 0, 8, 16, 32, COLI_SYCL_DT_INT8, COLI_SYCL_DT_INT8, COLI_SYCL_DT_S32 },
    { 0, 0, 0, 8, 16, 8, COLI_SYCL_DT_TF32, COLI_SYCL_DT_TF32, COLI_SYCL_DT_F32 },
};

static coli_sycl_props props(coli_sycl_arch arch, const char *name, const int *sg, int nsg, int eus, size_t slm,
                             double gb, int integrated, unsigned shapes) {
    coli_sycl_props pr;
    memset(&pr, 0, sizeof pr);
    pr.arch = arch;
    snprintf(pr.device_name, sizeof pr.device_name, "%s", name);
    for (int i = 0; i < nsg; i++) pr.sg_sizes[pr.sg_count++] = sg[i];
    pr.eu_count = eus; pr.max_compute_units = eus; pr.hw_threads_per_eu = 8;
    pr.slm_bytes = slm; pr.max_wg_size = 1024; pr.global_mem_bytes = (size_t)(gb * 1e9);
    pr.host_unified_memory = integrated;
    pr.xmx_shapes = shapes;
    return pr;
}

int main(void) {
    const int sg16_32[] = { 16, 32 }, sg8_16_32[] = { 8, 16, 32 }, sg32[] = { 32 }, sg16[] = { 16 };
    const unsigned s16 = coli_sycl_xmx_shape_flags(xmx16, 6), s8 = coli_sycl_xmx_shape_flags(xmx8, 2);

    printf("shape flags\n");
    CHECK(s16 == (COLI_SYCL_XMX_F16 | COLI_SYCL_XMX_F16_16 | COLI_SYCL_XMX_BF16 | COLI_SYCL_XMX_INT8 | COLI_SYCL_XMX_TF32));
    CHECK(s8 == 0);   /* 8x8 tiles at sub-group 8: not the kernels' shape */
    CHECK(coli_sycl_xmx_shape_flags(xmx16_fixed, 5) == s16);   /* all-fixed rows read the same */
    {   /* a ranged dimension takes every size up to its maximum, a fixed one its size alone */
        const coli_sycl_combo ranged[] = { { 8, 16, 32, 0, 0, 0, COLI_SYCL_DT_INT8, COLI_SYCL_DT_INT8, COLI_SYCL_DT_S32 },
                                           { 8, 16, 16, 0, 0, 0, COLI_SYCL_DT_F16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F32 } };
        CHECK(coli_sycl_xmx_shape_flags(ranged, 2) == (COLI_SYCL_XMX_INT8 | COLI_SYCL_XMX_F16));
        CHECK(coli_sycl_xmx_shape_flags(ranged, 0) == 0);
        const coli_sycl_combo m4[] = { { 4, 0, 0, 0, 16, 16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F32 } };   /* M up to 4: not 8 */
        CHECK(coli_sycl_xmx_shape_flags(m4, 1) == 0);
        const coli_sycl_combo k16[] = { { 8, 0, 0, 0, 16, 16, COLI_SYCL_DT_INT8, COLI_SYCL_DT_INT8, COLI_SYCL_DT_S32 } };   /* s8 with K 16: not 32 */
        CHECK(coli_sycl_xmx_shape_flags(k16, 1) == 0);
        const coli_sycl_combo big[] = { { 0, 0, 0, 32, 64, 16, COLI_SYCL_DT_BF16, COLI_SYCL_DT_BF16, COLI_SYCL_DT_F32 } };   /* pvc's 32x64x16: no row */
        CHECK(coli_sycl_xmx_shape_flags(big, 1) == 0);
        const coli_sycl_combo none[] = { { 0, 0, 0, 0, 0, 0, COLI_SYCL_DT_F16, COLI_SYCL_DT_F16, COLI_SYCL_DT_F32 } };
        CHECK(coli_sycl_xmx_shape_flags(none, 1) == 0);
    }

    coli_sycl_props pvc = props(COLI_SYCL_ARCH_PVC, "Intel(R) Data Center GPU Max 1100", sg16_32, 2, 448, 128 * 1024, 48.0, 0, s16);
    snprintf(pvc.pci_address, sizeof pvc.pci_address, "0000:29:00.0"); pvc.pci_device_id = 0x0bda;
    coli_sycl_props bmg = props(COLI_SYCL_ARCH_BMG, "Intel(R) Arc(TM) B580 Graphics", sg16_32, 2, 160, 128 * 1024, 12.0, 0, s16);
    coli_sycl_props lnl = props(COLI_SYCL_ARCH_LNL, "Intel(R) Arc(TM) 140V GPU", sg16_32, 2, 64, 64 * 1024, 16.0, 1, s16);
    coli_sycl_props dg2 = props(COLI_SYCL_ARCH_DG2, "Intel(R) Arc(TM) A770 Graphics", sg8_16_32, 3, 512, 64 * 1024, 16.0, 0, s8);
    coli_sycl_props unknown = props(COLI_SYCL_ARCH_OTHER, "Intel(R) Graphics [0xffff]", sg16_32, 2, 128, 64 * 1024, 8.0, 1, 0);

    printf("defaults\n");
    {
        coli_sycl_profile p = coli_sycl_profile_derive(pvc, no_env, nullptr);
        CHECK(p.arch == COLI_SYCL_ARCH_PVC && p.gen == COLI_SYCL_GEN_PVC);
        CHECK(p.sg_default == 32);
        CHECK_STR(p.name, "pvc-sg32");
        CHECK(p.has_xmx == 1 && p.xmx_shapes == s16);
        CHECK(coli_sycl_xmx_supported(p, COLI_SYCL_DT_F16) && coli_sycl_xmx_supported(p, COLI_SYCL_DT_INT8));
        CHECK(coli_sycl_xmx_supported(p, COLI_SYCL_DT_BF16) && coli_sycl_xmx_supported(p, COLI_SYCL_DT_TF32));
        CHECK(!coli_sycl_xmx_supported(p, COLI_SYCL_DT_F32));
        CHECK(p.is_integrated == 0 && p.eu_count == 448 && p.hw_threads_per_eu == 8);
        CHECK(p.slm_bytes == 128 * 1024 && p.max_wg_size == 1024);
        CHECK(p.sg_count == 2 && p.sg_sizes[0] == 16 && p.sg_sizes[1] == 32);
        CHECK(coli_sycl_sg_offered(p, 16) && coli_sycl_sg_offered(p, 32) && !coli_sycl_sg_offered(p, 8));
        CHECK_STR(p.pci_address, "0000:29:00.0");
        CHECK(p.pci_device_id == 0x0bda);
        CHECK_STR(p.device_name, "Intel(R) Data Center GPU Max 1100");
    }
    {
        coli_sycl_profile p = coli_sycl_profile_derive(bmg, no_env, nullptr);
        CHECK(p.arch == COLI_SYCL_ARCH_BMG && p.gen == COLI_SYCL_GEN_XE2);
        CHECK(p.sg_default == 16);
        CHECK_STR(p.name, "bmg-sg16");
        CHECK(p.has_xmx == 1 && p.is_integrated == 0);
    }
    {
        coli_sycl_profile p = coli_sycl_profile_derive(lnl, no_env, nullptr);
        CHECK(p.arch == COLI_SYCL_ARCH_LNL && p.gen == COLI_SYCL_GEN_XE2);
        CHECK(p.sg_default == 16 && p.has_xmx == 1 && p.is_integrated == 1);
        CHECK_STR(p.name, "lnl-sg16");
    }
    {   /* dg2: a matrix engine the kernels cannot use (8x8 tiles at sub-group 8) stays off */
        coli_sycl_profile p = coli_sycl_profile_derive(dg2, no_env, nullptr);
        CHECK(p.arch == COLI_SYCL_ARCH_DG2 && p.gen == COLI_SYCL_GEN_XE);
        CHECK(p.sg_default == 16 && p.has_xmx == 0 && p.xmx_shapes == 0);
        CHECK_STR(p.name, "dg2-sg16");
        CHECK(!coli_sycl_xmx_supported(p, COLI_SYCL_DT_F16));
    }
    {   /* an unknown generation: the generic profile from the values, never a failure */
        coli_sycl_profile p = coli_sycl_profile_derive(unknown, no_env, nullptr);
        CHECK(p.arch == COLI_SYCL_ARCH_OTHER && p.gen == COLI_SYCL_GEN_UNKNOWN);
        CHECK(p.sg_default == 16 && p.has_xmx == 0 && p.is_integrated == 1);
        CHECK_STR(p.name, "generic-sg16");
        /* with shapes listed the matrix engines go on by the query alone, never by the name */
        coli_sycl_props u2 = unknown; u2.xmx_shapes = s16;
        coli_sycl_profile q = coli_sycl_profile_derive(u2, no_env, nullptr);
        CHECK(q.has_xmx == 1 && coli_sycl_xmx_supported(q, COLI_SYCL_DT_F16));
    }
    {   /* sub-group choices on what the device offers */
        coli_sycl_props only32 = props(COLI_SYCL_ARCH_BMG, "x", sg32, 1, 0, 0, 0, 0, s16);
        only32.max_wg_size = 0;   /* every query of this one failed */
        coli_sycl_profile p = coli_sycl_profile_derive(only32, no_env, nullptr);
        CHECK(p.sg_default == 32);
        CHECK(p.has_xmx == 0);   /* joint_matrix needs 16 */
        CHECK(p.slm_bytes == 64 * 1024 && p.max_wg_size == 256 && p.eu_count == 1);   /* the stated defaults */
        coli_sycl_props only16 = props(COLI_SYCL_ARCH_PVC, "x", sg16, 1, 1, 0, 0, 0, 0);
        CHECK(coli_sycl_profile_derive(only16, no_env, nullptr).sg_default == 16);   /* pvc without 32: 16 */
        coli_sycl_props none = props(COLI_SYCL_ARCH_OTHER, "x", sg16, 0, 1, 0, 0, 0, 0);
        CHECK(coli_sycl_profile_derive(none, no_env, nullptr).sg_default == 16);     /* no sizes: 16 */
        const int sg8[] = { 8 };
        coli_sycl_props eight = props(COLI_SYCL_ARCH_OTHER, "x", sg8, 1, 1, 0, 0, 0, 0);
        CHECK(coli_sycl_profile_derive(eight, no_env, nullptr).sg_default == 16);    /* neither 16 nor 32: 16, the kernels' instance (said) */
        coli_sycl_props cpu = props(COLI_SYCL_ARCH_NONE, "cpu", sg8_16_32, 3, 8, 32 * 1024, 64.0, 1, 0);
        coli_sycl_profile c = coli_sycl_profile_derive(cpu, no_env, nullptr);
        CHECK(c.arch == COLI_SYCL_ARCH_NONE && c.sg_default == 16 && c.has_xmx == 0);
        CHECK_STR(c.name, "none-sg16");
    }

    printf("overrides\n");
    {
        coli_sycl_env e = { "16", nullptr, nullptr };
        coli_sycl_profile p = coli_sycl_profile_derive(pvc, e, nullptr);
        CHECK(p.sg_default == 16);
        CHECK_STR(p.name, "pvc-sg16");
        coli_sycl_env e32 = { "32", nullptr, nullptr };
        CHECK(coli_sycl_profile_derive(bmg, e32, nullptr).sg_default == 32);
        coli_sycl_props only16 = props(COLI_SYCL_ARCH_BMG, "x", sg16, 1, 1, 0, 0, 0, 0);
        CHECK(coli_sycl_profile_derive(only16, e32, nullptr).sg_default == 16);   /* not offered: kept */
        coli_sycl_env e24 = { "24", nullptr, nullptr };
        CHECK(coli_sycl_profile_derive(pvc, e24, nullptr).sg_default == 32);      /* nonsense: kept */
        coli_sycl_env empty = { "", "", "" };
        CHECK(coli_sycl_profile_derive(pvc, empty, nullptr).sg_default == 32);    /* set to nothing: unset */
    }
    {
        coli_sycl_env off = { nullptr, "0", nullptr };
        coli_sycl_profile p = coli_sycl_profile_derive(pvc, off, nullptr);
        CHECK(p.has_xmx == 0 && p.xmx_shapes == s16);   /* the shapes stay listed, the use is off */
        CHECK(!coli_sycl_xmx_supported(p, COLI_SYCL_DT_F16));
        coli_sycl_env on = { nullptr, "1", nullptr };
        CHECK(coli_sycl_profile_derive(dg2, on, nullptr).has_xmx == 0);   /* no shape: stays off */
        CHECK(coli_sycl_profile_derive(pvc, on, nullptr).has_xmx == 1);
        coli_sycl_props only32 = props(COLI_SYCL_ARCH_BMG, "x", sg32, 1, 1, 0, 0, 0, s16);
        CHECK(coli_sycl_profile_derive(only32, on, nullptr).has_xmx == 1);   /* forced on over the sg-16 rule */
        coli_sycl_env two = { nullptr, "2", nullptr };
        CHECK(coli_sycl_profile_derive(pvc, two, nullptr).has_xmx == 1);     /* nonsense: kept */
        /* the engines are on only with the kernels' f16 8x16x16 shape: a device listing other
         * shapes stays off, and =1 cannot turn it on */
        coli_sycl_props bfonly = props(COLI_SYCL_ARCH_BMG, "x", sg16_32, 2, 1, 0, 0, 0, COLI_SYCL_XMX_BF16 | COLI_SYCL_XMX_INT8);
        CHECK(coli_sycl_profile_derive(bfonly, no_env, nullptr).has_xmx == 0);
        CHECK(coli_sycl_profile_derive(bfonly, on, nullptr).has_xmx == 0);
        CHECK(!coli_sycl_xmx_supported(coli_sycl_profile_derive(bfonly, no_env, nullptr), COLI_SYCL_DT_INT8));
    }
    {
        coli_sycl_env as_pvc = { nullptr, nullptr, "pvc" };
        coli_sycl_profile p = coli_sycl_profile_derive(bmg, as_pvc, nullptr);
        CHECK(p.arch == COLI_SYCL_ARCH_PVC && p.gen == COLI_SYCL_GEN_PVC && p.sg_default == 32);
        CHECK_STR(p.name, "pvc-sg32");
        CHECK(p.eu_count == 160);   /* the measured values stay the device's */
        coli_sycl_env as_generic = { nullptr, nullptr, "generic" };
        coli_sycl_profile g = coli_sycl_profile_derive(pvc, as_generic, nullptr);
        CHECK(g.arch == COLI_SYCL_ARCH_OTHER && g.gen == COLI_SYCL_GEN_UNKNOWN && g.sg_default == 16);
        CHECK_STR(g.name, "generic-sg16");
        CHECK(g.has_xmx == 1);      /* still from the query */
        coli_sycl_env bogus = { nullptr, nullptr, "hopper" };
        coli_sycl_profile b = coli_sycl_profile_derive(pvc, bogus, nullptr);
        CHECK(b.arch == COLI_SYCL_ARCH_PVC && b.sg_default == 32);   /* unknown name: ignored */
        coli_sycl_env suffixed = { nullptr, nullptr, "bmg-sg32" };
        coli_sycl_profile s = coli_sycl_profile_derive(pvc, suffixed, nullptr);
        CHECK(s.arch == COLI_SYCL_ARCH_BMG && s.sg_default == 32);
        CHECK_STR(s.name, "bmg-sg32");
        coli_sycl_env suffixed16 = { nullptr, nullptr, "pvc-sg16" };
        CHECK(coli_sycl_profile_derive(pvc, suffixed16, nullptr).sg_default == 16);
        coli_sycl_env both = { "32", nullptr, "pvc-sg16" };
        CHECK(coli_sycl_profile_derive(pvc, both, nullptr).sg_default == 32);   /* COLI_SYCL_SG wins */
        coli_sycl_env bad_suffix = { nullptr, nullptr, "pvc-sg64" };
        CHECK(coli_sycl_profile_derive(pvc, bad_suffix, nullptr).sg_default == 32);
    }
    {   /* the process environment is read through coli_sycl_env_read */
        setenv("COLI_SYCL_SG", "16", 1); setenv("COLI_SYCL_XMX", "0", 1); setenv("COLI_SYCL_PROFILE", "generic", 1);
        coli_sycl_env e = coli_sycl_env_read();
        CHECK(e.sg && !strcmp(e.sg, "16") && e.xmx && !strcmp(e.xmx, "0") && e.profile && !strcmp(e.profile, "generic"));
        coli_sycl_profile p = coli_sycl_profile_derive(pvc, e, nullptr);
        CHECK(p.sg_default == 16 && p.has_xmx == 0 && p.arch == COLI_SYCL_ARCH_OTHER);
        CHECK_STR(p.name, "generic-sg16");
        unsetenv("COLI_SYCL_SG"); unsetenv("COLI_SYCL_XMX"); unsetenv("COLI_SYCL_PROFILE");
        coli_sycl_env n = coli_sycl_env_read();
        CHECK(!n.sg && !n.xmx && !n.profile);
    }
    {   /* the log: an override the device cannot honour is reported, a clean run says nothing */
        char buf[512] = { 0 };
        FILE *f = tmpfile();
        CHECK(f != nullptr);
        if (f) {
            coli_sycl_env e = { "32", "1", "hopper" };
            coli_sycl_props only16 = props(COLI_SYCL_ARCH_DG2, "x", sg16, 1, 1, 64 * 1024, 0, 0, 0);
            coli_sycl_profile_derive(only16, e, f);
            fflush(f); rewind(f);
            size_t n = fread(buf, 1, sizeof buf - 1, f); buf[n] = 0;
            CHECK(strstr(buf, "COLI_SYCL_SG=32") && strstr(buf, "COLI_SYCL_XMX=1") && strstr(buf, "COLI_SYCL_PROFILE=hopper"));
            int lines = 0; for (size_t i = 0; i < n; i++) lines += buf[i] == '\n';
            CHECK(lines == 3);
            rewind(f);   /* a device offering neither 16 nor 32 is told what runs */
            const int sg8[] = { 8 };
            coli_sycl_props eight = props(COLI_SYCL_ARCH_OTHER, "x", sg8, 1, 1, 0, 0, 0, 0);
            coli_sycl_profile_derive(eight, no_env, f);
            fflush(f); rewind(f);
            n = fread(buf, 1, sizeof buf - 1, f); buf[n] = 0;
            CHECK(strstr(buf, "neither sub-group 16 nor 32") != nullptr);
            rewind(f);
            coli_sycl_profile_derive(pvc, no_env, f);
            fflush(f);
            CHECK(ftell(f) == 0);
            fclose(f);
        }
    }

    printf("names\n");
    {
        const coli_sycl_arch all[] = { COLI_SYCL_ARCH_PVC, COLI_SYCL_ARCH_BMG, COLI_SYCL_ARCH_LNL, COLI_SYCL_ARCH_PTL,
                                       COLI_SYCL_ARCH_ARL, COLI_SYCL_ARCH_MTL, COLI_SYCL_ARCH_DG2, COLI_SYCL_ARCH_WCL,
                                       COLI_SYCL_ARCH_XELP, COLI_SYCL_ARCH_GEN9, COLI_SYCL_ARCH_OTHER };
        for (coli_sycl_arch a : all) CHECK(coli_sycl_arch_from_name(coli_sycl_arch_name(a)) == a);
        CHECK(coli_sycl_arch_from_name("none") == COLI_SYCL_ARCH_NONE && coli_sycl_arch_from_name("") == COLI_SYCL_ARCH_NONE);
        CHECK(coli_sycl_arch_from_name(nullptr) == COLI_SYCL_ARCH_NONE);
        CHECK_STR(coli_sycl_arch_name(COLI_SYCL_ARCH_NONE), "none");
        CHECK_STR(coli_sycl_arch_name(COLI_SYCL_ARCH_OTHER), "generic");
        CHECK(coli_sycl_gen_of(COLI_SYCL_ARCH_PTL) == COLI_SYCL_GEN_XE3 && coli_sycl_gen_of(COLI_SYCL_ARCH_WCL) == COLI_SYCL_GEN_XE3);
        CHECK(coli_sycl_gen_of(COLI_SYCL_ARCH_MTL) == COLI_SYCL_GEN_XE && coli_sycl_gen_of(COLI_SYCL_ARCH_ARL) == COLI_SYCL_GEN_XE);
        CHECK(coli_sycl_gen_of(COLI_SYCL_ARCH_XELP) == COLI_SYCL_GEN_LEGACY && coli_sycl_gen_of(COLI_SYCL_ARCH_GEN9) == COLI_SYCL_GEN_LEGACY);
        CHECK(coli_sycl_gen_of(COLI_SYCL_ARCH_NONE) == COLI_SYCL_GEN_UNKNOWN);
        CHECK_STR(coli_sycl_gen_name(COLI_SYCL_GEN_XE2), "xe2");
        CHECK_STR(coli_sycl_gen_name(COLI_SYCL_GEN_UNKNOWN), "unknown");
    }
    {   /* the printed form, as COLI_SYCL_LIST=1 shows it */
        coli_sycl_profile p = coli_sycl_profile_derive(pvc, no_env, nullptr);
        coli_sycl_profile_print(stdout, 0, p);
        char buf[1024] = { 0 };
        FILE *f = tmpfile();
        if (f) {
            coli_sycl_profile_print(f, 3, p);
            fflush(f); rewind(f);
            size_t n = fread(buf, 1, sizeof buf - 1, f); buf[n] = 0; fclose(f);
            CHECK(strstr(buf, "device 3: Intel(R) Data Center GPU Max 1100") && strstr(buf, "profile pvc-sg32") &&
                  strstr(buf, "sub-groups: 16 32 (kernels at 32)") && strstr(buf, "448 EUs x 8 threads, SLM 128 KiB") &&
                  strstr(buf, "xmx on: f16-8x16x16 f16-16x16x16 bf16-8x16x16 s8-8x16x32 tf32-8x16x8"));
        }
    }

    printf("tuned defaults\n");
    {   /* the measured rows apply on pvc only; every other architecture, an unknown switch and
         * COLI_PLATFORM_DEFAULTS=0 keep the code's own default */
        const coli_sycl_profile p_pvc = coli_sycl_profile_derive(pvc, no_env, nullptr);
        const coli_sycl_profile others[] = { coli_sycl_profile_derive(bmg, no_env, nullptr),
                                             coli_sycl_profile_derive(lnl, no_env, nullptr),
                                             coli_sycl_profile_derive(dg2, no_env, nullptr),
                                             coli_sycl_profile_derive(unknown, no_env, nullptr) };
        const struct { const char *knob; int generic, pvc; } rows[] = {
            { "COLI_CUDA_ATTN_GROUPED_PLAIN", 0, 32 }, { "COLI_CUDA_ATTN_GROUPED_VEC", 0, 2 },
            { "COLI_CUDA_DNREC_PROMPT_SG", 0, 16 },    { "COLI_CUDA_GEMV_XPAD", 1, 2 },
            { "COLI_CUDA_MUX_BATCH", 0, 15 },          { "COLI_TRUNK_PROBE", 1, 0 },
            { "Q38_SYCL_HYBRID", 0, 1 },              { "COLI_CUDA_EXPERT_ARENA", 0, 2 },
            { "COLI_EXPERT_PAGECACHE", 0, 1 },         { "COLI_CUDA_DENSE_CACHE", 0, 1 },
        };
        for (const auto &r : rows) {
            CHECK(coli_sycl_tuned_default(p_pvc, nullptr, r.knob, r.generic) == r.pvc);
            CHECK(coli_sycl_tuned_default(p_pvc, "1", r.knob, r.generic) == r.pvc);    /* any value but 0 leaves them on */
            CHECK(coli_sycl_tuned_default(p_pvc, "", r.knob, r.generic) == r.pvc);
            CHECK(coli_sycl_tuned_default(p_pvc, "0", r.knob, r.generic) == r.generic);
            for (const auto &o : others) CHECK(coli_sycl_tuned_default(o, nullptr, r.knob, r.generic) == r.generic);
        }
        CHECK(coli_sycl_tuned_default(p_pvc, nullptr, "COLI_SOMETHING_ELSE", 7) == 7);
        CHECK(coli_sycl_tuned_default(p_pvc, nullptr, nullptr, 5) == 5);
        /* COLI_SYCL_PROFILE=pvc on another part takes pvc's arch tag, and so pvc's rows */
        const coli_sycl_env as_pvc = { nullptr, nullptr, "pvc" };
        const coli_sycl_profile bmg_as_pvc = coli_sycl_profile_derive(bmg, as_pvc, nullptr);
        CHECK(bmg_as_pvc.arch == COLI_SYCL_ARCH_PVC);
        CHECK(coli_sycl_tuned_default(bmg_as_pvc, nullptr, "COLI_CUDA_MUX_BATCH", 0) == 15);
    }

    printf("%s: %d checks, %d failed\n", fails ? "FAILED" : "PASSED", checks, fails);
    return fails != 0;
}
