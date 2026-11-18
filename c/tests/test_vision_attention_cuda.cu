/* coli_cuda_vision_attention (the Qwen vision tower's attention on the card, qwen38_vision.h)
 * against the attention in double precision, and against itself: softmax(q.k * scale) over v for
 * every row of a qkv laid out [n][3][heads][D], for the head widths the kernel takes, a row count
 * below one tile, one between tiles and one over a block of queries, and a width it refuses (the
 * CPU's then, out untouched). Two launches give the same bits: no atomics, the keys in order. */
#include "../backend_cuda.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static unsigned rng_state = 20261117u;
static float rnd() { rng_state = rng_state * 1664525u + 1013904223u; return ((rng_state >> 8) / 16777216.0f) * 2.f - 1.f; }

static int check(int n, int heads, int D) {
    const int H = heads * D;
    std::vector<float> qkv((size_t)n * 3 * H), out((size_t)n * H, -7.f), again((size_t)n * H);
    for (auto &v : qkv) v = rnd() * 2.f;
    const float scale = 1.f / sqrtf((float)D);
    if (!coli_cuda_vision_attention(0, out.data(), qkv.data(), n, heads, D, scale)) { printf("FAIL n=%d D=%d: refused\n", n, D); return 1; }
    if (!coli_cuda_vision_attention(0, again.data(), qkv.data(), n, heads, D, scale)) { printf("FAIL n=%d D=%d: refused again\n", n, D); return 1; }
    if (memcmp(out.data(), again.data(), out.size() * sizeof(float))) { printf("FAIL n=%d D=%d: two launches differ\n", n, D); return 1; }
    double worst = 0;
    std::vector<double> s(n);
    for (int h = 0; h < heads; h++)
        for (int i = 0; i < n; i++) {
            const float *q = &qkv[(size_t)i * 3 * H + (size_t)h * D];
            double best = -1e300, sum = 0;
            for (int j = 0; j < n; j++) {
                const float *k = &qkv[(size_t)j * 3 * H + H + (size_t)h * D];
                double a = 0;
                for (int d = 0; d < D; d++) a += (double)q[d] * k[d];
                s[j] = a * scale;
                if (s[j] > best) best = s[j];
            }
            for (int j = 0; j < n; j++) { s[j] = exp(s[j] - best); sum += s[j]; }
            for (int d = 0; d < D; d++) {
                double ref = 0, mag = 0;
                for (int j = 0; j < n; j++) {
                    const double v = qkv[(size_t)j * 3 * H + 2 * H + (size_t)h * D + d];
                    ref += s[j] / sum * v;
                    mag += s[j] / sum * fabs(v);
                }
                const double e = fabs(out[(size_t)i * H + (size_t)h * D + d] - ref) / (mag > 0 ? mag : 1);
                if (e > worst) worst = e;
            }
        }
    printf("%s n=%4d heads=%d D=%3d: worst error %.2e of sum|p v|\n", worst < 1e-5 ? "ok  " : "FAIL", n, heads, D, worst);
    return worst < 1e-5 ? 0 : 1;
}

int main() {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) return 77;
    int fails = 0;
    const int widths[] = {64, 72, 80, 96, 128};
    for (int w : widths) fails += check(37, 3, w);
    fails += check(1, 2, 72);
    fails += check(300, 4, 72);     /* more than a block of queries (128), keys over many tiles */
    fails += check(1000, 2, 72);
    float q[3 * 48] = {0}, o[48];
    for (int i = 0; i < 48; i++) o[i] = 5.f;
    if (coli_cuda_vision_attention(0, o, q, 1, 1, 48, 1.f) || o[0] != 5.f) { printf("FAIL: D=48 taken, or out written\n"); fails++; }
    else printf("ok   D=48 refused, out untouched\n");
    printf(fails ? "vision attention: %d FAILED\n" : "vision attention: all passed\n", fails);
    return fails ? 1 : 0;
}
