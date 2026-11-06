/* coli_cuda_expert_group_issue past 8 rows (the qwen36 tier's group of one device for
 * a token, up to QT_MAX_ROWS = 32 rows). For int8, int4 per row, int4 in groups of 16, 32,
 * 64 and 128, fp8, and every GGUF block type with an integer step (bdi_has), 32 experts
 * of one row each are issued three ways and must give the same bits (E8, fmt 6, takes its
 * own per-expert kernels, which do not depend on the row count, and is not covered here):
 *   A  one 32-row issue (refused before this change: the tier then ran them on the CPU);
 *   B  the same rows as four 8-row issues (what the old cap admitted);
 *   C  coli_cuda_expert_group_pinned(pin 1), the synchronous call on the decode family.
 * Each output row depends only on its own expert and input row, so any difference is a
 * launch or numeric-family change in the 32-row path. Real kernels, random weights.
 * Exit 0: all formats accepted and bit-exact. 1: a refusal or a mismatch. 77: no GPU. */
#include "../backend_cuda.h"
#include "../blockdec_host.h"
#include "../fp8_format.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

enum { NE = 32, D = 256, H = 256, SPLIT = 8 };

static uint32_t rng_state = 0x2545F491u;
static uint32_t rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }
static float frand(float lo, float hi) { return lo + (hi - lo) * ((rnd() >> 8) * (1.0f / 16777216.0f)); }

struct FmtCase { int fmt; int gs; const char *name; };

static size_t weight_bytes(int fmt, int I, int O) {
    if (fmt >= 100) return (size_t)(I / BD_BLK[fmt - 100]) * BD_BYTES[fmt - 100] * (size_t)O;   /* the GGUF's blocks */
    if (fmt == 2 || fmt == 4) return (size_t)((I + 1) / 2) * (size_t)O;
    return (size_t)I * (size_t)O;                                /* fmt 1 and 8: a byte per weight */
}
static size_t scale_count(const FmtCase &f, int I, int O) {
    if (f.fmt >= 100) return 0;                                  /* scales live in the blocks */
    if (f.fmt == 8) return (size_t)((O + 127) / 128) * (size_t)((I + 127) / 128);
    if (f.fmt == 4) return (size_t)O * (size_t)(I / f.gs);
    return (size_t)O;                                            /* fmt 1 and 2: a scale per row */
}
static void fill_tensor(const FmtCase &f, int I, int O, std::vector<uint8_t> &w, std::vector<float> &s) {
    w.resize(weight_bytes(f.fmt, I, O));
    for (auto &b : w) b = (uint8_t)rnd();
    /* GGUF blocks: every block size here is even, so each fp16 field is one aligned
     * 16-bit word. Clearing bits 13 and 14 of every word keeps any fp16 below 2^-7
     * and never inf or NaN; the other words stay valid random quants. */
    if (f.fmt >= 100)
        for (size_t b = 0; b + 1 < w.size(); b += 2) w[b + 1] &= (uint8_t)~0x60u;
    s.resize(scale_count(f, I, O));
    for (auto &v : s) v = frand(0.002f, 0.03f);
}
static int upload(ColiCudaTensor **t, const FmtCase &f, const std::vector<uint8_t> &w, const std::vector<float> &s, int I, int O) {
    const float *sp = s.empty() ? nullptr : s.data();
    if (f.fmt == 4) return coli_cuda_tensor_upload_g(t, w.data(), sp, 4, I, O, 0, f.gs);
    return coli_cuda_tensor_upload(t, w.data(), sp, f.fmt, I, O, 0);
}

static int run_case(const FmtCase &f) {
    std::vector<ColiCudaTensor *> tg(NE), tu(NE), td(NE);
    for (int e = 0; e < NE; e++) {
        std::vector<uint8_t> w; std::vector<float> s;
        fill_tensor(f, D, H, w, s); if (!upload(&tg[e], f, w, s, D, H)) { std::printf("fmt=%d upload failed\n", f.fmt); return 1; }
        fill_tensor(f, D, H, w, s); if (!upload(&tu[e], f, w, s, D, H)) { std::printf("fmt=%d upload failed\n", f.fmt); return 1; }
        fill_tensor(f, H, D, w, s); if (!upload(&td[e], f, w, s, H, D)) { std::printf("fmt=%d upload failed\n", f.fmt); return 1; }
    }
    std::vector<float> x((size_t)NE * D);
    for (auto &v : x) v = frand(-1.f, 1.f);
    std::vector<int> rows(NE, 1);

    std::vector<float> ya((size_t)NE * D, 0.f), yb((size_t)NE * D, 0.f), yc((size_t)NE * D, 0.f);
    int a = coli_cuda_expert_group_issue(tg.data(), tu.data(), td.data(), rows.data(), NE, x.data());
    if (a) std::memcpy(ya.data(), coli_cuda_expert_group_take(0), ya.size() * sizeof(float));
    int b = 1;
    for (int g = 0; g < NE / SPLIT && b; g++) {
        b = coli_cuda_expert_group_issue(tg.data() + g * SPLIT, tu.data() + g * SPLIT, td.data() + g * SPLIT,
                                         rows.data(), SPLIT, x.data() + (size_t)g * SPLIT * D);
        if (b) std::memcpy(yb.data() + (size_t)g * SPLIT * D, coli_cuda_expert_group_take(0), (size_t)SPLIT * D * sizeof(float));
    }
    int c = coli_cuda_expert_group_pinned(tg.data(), tu.data(), td.data(), rows.data(), NE, yc.data(), x.data(), 1);

    size_t dab = 0, dcb = 0; float maxv = 0.f;
    for (size_t i = 0; i < yb.size(); i++) {
        if (std::memcmp(&ya[i], &yb[i], sizeof(float))) dab++;
        if (std::memcmp(&yc[i], &yb[i], sizeof(float))) dcb++;
        float m = std::fabs(yb[i]); if (m > maxv) maxv = m;
    }
    int exact = a && b && c && dab == 0 && dcb == 0 && maxv > 0.f;
    std::printf("fmt=%-3d %-13s issue32=%s split8x4=%s pinned32=%s  issue32 vs split: %zu differ, pinned vs split: %zu differ  %s\n",
                f.fmt, f.name, a ? "ok" : "REFUSED", b ? "ok" : "REFUSED", c ? "ok" : "REFUSED", dab, dcb,
                exact ? "BIT-EXACT" : "FAIL");
    for (int e = 0; e < NE; e++) { coli_cuda_tensor_free(tg[e]); coli_cuda_tensor_free(tu[e]); coli_cuda_tensor_free(td[e]); }
    return exact ? 0 : 1;
}

int main(void) {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { std::printf("expert group rows: no CUDA device, skipped\n"); return 77; }
    float lut[256];
    for (int i = 0; i < 256; i++) lut[i] = (float)(i - 128) / 64.0f;
    if (!coli_cuda_fp8_set_lut(lut)) { std::printf("fp8 LUT publish failed\n"); coli_cuda_shutdown(); return 1; }
    const FmtCase cases[] = {
        {1, 0, "int8-row"}, {2, 0, "int4-nibble"}, {4, 16, "int4-g16"}, {4, 32, "int4-g32"},
        {4, 64, "int4-g64"}, {4, 128, "int4-g128"},
        {8, 0, "fp8-e4m3"},
        {108, 0, "GGUF-Q8_0"}, {110, 0, "GGUF-Q2_K"}, {111, 0, "GGUF-Q3_K"}, {112, 0, "GGUF-Q4_K"},
        {113, 0, "GGUF-Q5_K"}, {114, 0, "GGUF-Q6_K"}, {116, 0, "GGUF-IQ2_XXS"}, {117, 0, "GGUF-IQ2_XS"},
        {118, 0, "GGUF-IQ3_XXS"}, {119, 0, "GGUF-IQ1_S"}, {120, 0, "GGUF-IQ4_NL"}, {121, 0, "GGUF-IQ3_S"},
        {122, 0, "GGUF-IQ2_S"}, {123, 0, "GGUF-IQ4_XS"}, {129, 0, "GGUF-IQ1_M"}, {142, 0, "GGUF-type42"},
    };   /* the GGUF rows are bdi_has (cuda_bd.cuh): 8, 10-14, 16-23, 29, 42 */
    int failed = 0;
    for (const FmtCase &f : cases) failed += run_case(f);
    coli_cuda_shutdown();
    std::printf("expert group 32 rows: %s (%d of %d formats not bit-exact)\n", failed ? "FAIL" : "all bit-exact",
                failed, (int)(sizeof cases / sizeof cases[0]));
    return failed ? 1 : 0;
}
