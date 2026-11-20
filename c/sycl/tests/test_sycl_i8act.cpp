/* coli_cuda_matmul_i8act (Qwen3.8's Q38_TRUNK_GPU_I8): a resident fmt=1 tensor (int8 rows, a float scale a row) times
 * int8 activations quantized on the host with a scale a row must give, byte for byte, the CPU's matmul_q_idot:
 * y[s][o] = ((float)dot_int32 * scale[o]) * sx[s]. Shapes of the Qwen3.8 trunk (2560 in, 8192/2560/12288 out) and odd
 * ones; 1 to 8 activation rows; extreme values (+-127 weights, -128..127 activations) to reach the int32 range; a
 * refused call for a non-int8 tensor and for I not a multiple of 4. */
#include "../../backend_cuda.h"
#include "sycl_test_args.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL I8ACT: %s\n", what); std::exit(1); }
}
static uint32_t state = 0x1234abcdu;
static uint32_t rnd() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }

static void reference(float *y, const int8_t *xq, const float *sx, const int8_t *w, const float *sc, int S, int I, int O) {
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            int32_t d = 0;
            for (int i = 0; i < I; i++) d += (int32_t)w[(size_t)o * I + i] * (int32_t)xq[(size_t)s * I + i];
            const float sco = sc[o];
            y[(size_t)s * O + o] = (float)d * sco * sx[s];   /* matmul_q_idot's expression */
        }
}

static void shape(int dev, int I, int O, int S, bool extreme) {
    std::vector<int8_t> w((size_t)O * I), xq((size_t)S * I);
    std::vector<float> sc(O), sx(S), y((size_t)S * O), want((size_t)S * O);
    for (auto &v : w) v = extreme ? (int8_t)((rnd() & 1) ? 127 : -127) : (int8_t)((int)(rnd() % 255) - 127);
    for (auto &v : xq) v = extreme ? (int8_t)((rnd() & 1) ? 127 : -128) : (int8_t)((int)(rnd() % 256) - 128);
    if (extreme && I > 100000) {   /* all positive: every dot near 127*127*I (2.11e9 at 131 068), past 2^24 */
        for (auto &v : w) v = 127;
        for (size_t k = 0; k < xq.size(); k++) xq[k] = (int8_t)(127 - (int)(k % 3));
    }
    for (auto &v : sc) v = (float)(rnd() % 100000 + 1) * 1.37e-7f;
    for (auto &v : sx) v = (float)(rnd() % 100000 + 1) * 2.11e-6f;
    ColiCudaTensor *t = nullptr;
    require(coli_cuda_tensor_upload(&t, w.data(), sc.data(), 1, I, O, dev), "upload");
    require(coli_cuda_matmul_i8act(t, y.data(), (const signed char *)xq.data(), sx.data(), S, I, O), "i8act call");
    reference(want.data(), xq.data(), sx.data(), w.data(), sc.data(), S, I, O);
    size_t diff = 0;
    for (size_t k = 0; k < y.size(); k++) diff += std::memcmp(&y[k], &want[k], sizeof(float)) != 0;
    std::printf("I8ACT I=%d O=%d S=%d%s: %zu of %zu outputs differ\n", I, O, S, extreme ? " extreme" : "", diff, y.size());
    std::fflush(stdout);
    require(diff == 0, "byte-identical to the CPU formula");
    coli_cuda_tensor_free(t);
}

int main(int argc, char **argv) {
    int dev[1]; int n = sycl_test_devices(argc, argv, dev, 1); if (n < 0) return 0;
    require(coli_cuda_init(dev, n), "init");
    shape(dev[0], 128, 64, 1, false);
    shape(dev[0], 2560, 8192, 1, false);    /* Qwen3.8 DeltaNet qkv-like */
    shape(dev[0], 2560, 2560, 4, false);
    shape(dev[0], 8192, 2560, 1, false);    /* a down-like row */
    shape(dev[0], 2560, 12288, 8, false);
    shape(dev[0], 132, 37, 3, false);       /* odd output count */
    shape(dev[0], 2560, 257, 1, true);      /* extreme values */
    shape(dev[0], 131068, 4, 1, true);      /* near the int32 limit of a row */
    {   /* refusals */
        std::vector<float> wf(64 * 128, 0.f), scf(64, 1.f), yy(64);
        std::vector<int8_t> xq(130, 1); std::vector<float> sx(1, 1.f);
        ColiCudaTensor *tf = nullptr;
        require(coli_cuda_tensor_upload(&tf, wf.data(), nullptr, 0, 128, 64, dev[0]), "f32 upload");
        require(!coli_cuda_matmul_i8act(tf, yy.data(), (const signed char *)xq.data(), sx.data(), 1, 128, 64), "a float tensor is refused");
        coli_cuda_tensor_free(tf);
        std::vector<int8_t> w8(64 * 130, 1); std::vector<float> sc8(64, 1.f);
        ColiCudaTensor *t8 = nullptr;
        require(coli_cuda_tensor_upload(&t8, w8.data(), sc8.data(), 1, 130, 64, dev[0]), "int8 upload, I=130");
        require(!coli_cuda_matmul_i8act(t8, yy.data(), (const signed char *)xq.data(), sx.data(), 1, 130, 64), "I not a multiple of 4 is refused");
        coli_cuda_tensor_free(t8);
        std::vector<int8_t> wb((size_t)2 * 131072, 1); std::vector<float> scb(2, 1.f), yb(2);
        std::vector<int8_t> xb(131072, 1);
        ColiCudaTensor *tb = nullptr;
        require(coli_cuda_tensor_upload(&tb, wb.data(), scb.data(), 1, 131072, 2, dev[0]), "int8 upload, I=131072");
        require(!coli_cuda_matmul_i8act(tb, yb.data(), (const signed char *)xb.data(), sx.data(), 1, 131072, 2), "I past the int32 bound is refused");
        coli_cuda_tensor_free(tb);
    }
    coli_cuda_shutdown();
    std::puts("I8ACT SUITE PASS");
    return 0;
}
