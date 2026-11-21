/* coli_sycl.h's sub-group dispatch on a device: one body, instantiated at sub-group 16 and 32
 * (coli_sycl_launch_sg), a warp-per-row sum over the sub-group's lanes with the shuffle width
 * of the instance, checked against the host for both sizes the device offers. Needs a device:
 * run by hand in the runtime container; compiled (and AOT-linked for pvc and bmg) without one,
 * which is what the build check uses it for. */
#include "../coli_sycl.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

template <int SG>
static void rows_sum(sycl::nd_item<3> it, const float *x, float *y, int n, uint8_t *slm) {
    const int tid = (int)it.get_local_id(2), lane = tid & (SG - 1), warp = tid / SG;
    const int row = (int)it.get_group(2) * (256 / SG) + warp;   /* 256 / SG warps a block */
    /* each warp stages ITS row in its own 256-float slice of the SLM (the first version staged
     * one shared slice from 256 / SG different rows: every sum was a mix of rows) */
    float *xs = reinterpret_cast<float *>(slm) + warp * 256;
    for (int i = lane; i < 256; i += SG) xs[i] = row < n ? x[row * 256 + i] : 0.f;
    auto sg = it.get_sub_group();
    sycl::group_barrier(sg);
    float v = 0.f;
    for (int i = lane; i < 256; i += SG) v += xs[i];
    for (int o = SG / 2; o > 0; o >>= 1) v += sycl::shift_group_left(sg, v, o);
    if (lane == 0 && row < n) y[row] = v;
}

int main() {
    const int n = 64;
    std::vector<float> hx(n * 256), hy(n), want(n, 0.f);
    unsigned s = 7;
    for (int i = 0; i < n * 256; i++) { s = s * 1664525u + 1013904223u; hx[i] = ((s >> 8) & 0xff) / 255.f; want[i / 256] += hx[i]; }
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    coli_sycl_profile p = coli_sycl_profile_detect(q.get_device());
    float *x = sycl::malloc_device<float>(n * 256, q), *y = sycl::malloc_device<float>(n, q);
    q.memcpy(x, hx.data(), hx.size() * sizeof(float)).wait();
    int fails = 0;
    for (int sg : {16, 32}) {
        if (!coli_sycl_sg_offered(p, sg) && p.sg_count) { printf("  skip sub-group %d (not offered)\n", sg); continue; }
        q.memset(y, 0, n * sizeof(float)).wait();
        coli_sycl_launch_sg(q, sg, sycl::range<3>(1, 1, (n + 256 / sg - 1) / (256 / sg)), sycl::range<3>(1, 1, 256), (256 / 16) * 256 * sizeof(float),   /* a slice per warp, sized for SG 16 */
                            [=](sycl::nd_item<3> it, auto tag, uint8_t *slm) { rows_sum<decltype(tag)::value>(it, x, y, n, slm); }).wait();
        q.memcpy(hy.data(), y, n * sizeof(float)).wait();
        int bad = 0;
        for (int i = 0; i < n; i++) if (std::abs(hy[i] - want[i]) > 1e-3f * want[i]) bad++;
        printf("  %s sub-group %d: %d of %d rows wrong\n", bad ? "FAIL" : "ok  ", sg, bad, n);
        fails += bad != 0;
    }
    sycl::free(x, q); sycl::free(y, q);
    printf("%s (%s, sg default %d)\n", fails ? "FAILED" : "PASSED", p.device_name, p.sg_default);
    return fails != 0;
}
