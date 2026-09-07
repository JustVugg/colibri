/* test_exact_dot.c — properties of exact_dot.h (the opt-in exact verify mode's kernel):
 *  1. order independence: 200 random permutations of a cancellation-heavy f32 dot are bit-identical;
 *  2. exactness: 1e30 + 1 - 1e30 style sums come out exactly, where float accumulation loses them;
 *  3. agreement: when no rounding is possible (small integers), exact == naive;
 *  4. (w*s)*x path: order independent and equal to the f32 path on exactly-representable data;
 *  5. specials: NaN / inf classification matches float;
 *  6. cost: ns per element vs a plain float loop (printed, not asserted). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../exact_dot.h"

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float rndf(void){ uint32_t u = rnd(); int e = 100 + (int)(u % 55); return ((u & 1) ? -1.f : 1.f) * ldexpf((float)((u >> 8) & 0xFFFF) + 1, e - 127 - 16); }
static uint32_t fbits(float f){ uint32_t u; memcpy(&u, &f, 4); return u; }
static float naive(const float *x, const float *y, int n){ float a = 0; for(int i = 0; i < n; i++) a += x[i] * y[i]; return a; }
static int fails = 0;
#define CHECK(c, msg) do{ if(!(c)){ printf("  [FAIL] %s\n", msg); fails++; } else printf("  [PASS] %s\n", msg); }while(0)

int main(void){
    enum { N = 4096 };
    static float x[N], y[N]; static int idx[N];
    for(int i = 0; i < N; i++){ x[i] = rndf(); y[i] = rndf(); }
    for(int i = 0; i < N; i += 2){ x[i + 1] = -x[i]; y[i + 1] = y[i] * 1.0000001f; }   /* near-cancelling pairs */
    float ref = exd_dot_ff(x, y, N); int same = 1;
    for(int p = 0; p < 200 && same; p++){
        for(int i = 0; i < N; i++) idx[i] = i;
        for(int i = N - 1; i > 0; i--){ int j = (int)(rnd() % (uint32_t)(i + 1)); int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        exd_acc a; exd_init(&a); for(int i = 0; i < N; i++) exd_add_ff(&a, x[idx[i]], y[idx[i]]);
        if(fbits(exd_finish(&a)) != fbits(ref)) same = 0;
    }
    CHECK(same, "order independence: 200 permutations bit-identical (cancellation-heavy)");
    { float a[3] = {1e30f, 1.f, -1e30f}, b[3] = {1.f, 1.f, 1.f};
      CHECK(exd_dot_ff(a, b, 3) == 1.0f && naive(a, b, 3) == 0.0f, "exactness: 1e30 + 1 - 1e30 == 1 (float loop gives 0)"); }
    { static float xi[N], yi[N]; for(int i = 0; i < N; i++){ xi[i] = (float)((int)(rnd() % 9) - 4); yi[i] = (float)((int)(rnd() % 9) - 4); }
      CHECK(fbits(exd_dot_ff(xi, yi, N)) == fbits(naive(xi, yi, N)), "agreement with naive when no rounding can occur"); }
    { static int w[N]; static float s[N], xv[N]; float refw; exd_acc a; exd_init(&a);
      for(int i = 0; i < N; i++){ w[i] = (int)(rnd() % 16) - 8; s[i] = ldexpf(1.f + (float)(rnd() % 7) / 8.f, -(int)(rnd() % 6)); xv[i] = rndf(); exd_add_wsx(&a, w[i], s[i], xv[i]); }
      refw = exd_finish(&a); int ok = 1;
      for(int p = 0; p < 50 && ok; p++){ for(int i = 0; i < N; i++) idx[i] = i;
        for(int i = N - 1; i > 0; i--){ int j = (int)(rnd() % (uint32_t)(i + 1)); int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        exd_acc b; exd_init(&b); for(int i = 0; i < N; i++) exd_add_wsx(&b, w[idx[i]], s[idx[i]], xv[idx[i]]); if(fbits(exd_finish(&b)) != fbits(refw)) ok = 0; }
      CHECK(ok, "(w*s)*x path: 50 permutations bit-identical");
      exd_acc c2; exd_init(&c2); for(int i = 0; i < N; i++) exd_add_ff(&c2, (float)w[i] * s[i], xv[i]);   /* w*s exact for these s */
      CHECK(fbits(exd_finish(&c2)) == fbits(refw), "(w*s)*x == exact f32 path when w*s is representable"); }
    { float a[2] = {1.f, INFINITY}, b[2] = {1.f, 2.f}; CHECK(isinf(exd_dot_ff(a, b, 2)) && exd_dot_ff(a, b, 2) > 0, "inf propagates with sign");
      float c[2] = {INFINITY, 0.f}, d[2] = {0.f, 1.f}; CHECK(isnan(exd_dot_ff(c, d, 2)), "inf*0 -> nan");
      float e[2] = {NAN, 1.f}; CHECK(isnan(exd_dot_ff(e, b, 2)), "nan propagates"); }
    { struct timespec t0, t1; volatile float sink = 0; int reps = 200;
      clock_gettime(CLOCK_MONOTONIC, &t0); for(int r = 0; r < reps; r++) sink += naive(x, y, N); clock_gettime(CLOCK_MONOTONIC, &t1);
      double tn = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / (double)reps / N;
      clock_gettime(CLOCK_MONOTONIC, &t0); for(int r = 0; r < reps; r++) sink += exd_dot_ff(x, y, N); clock_gettime(CLOCK_MONOTONIC, &t1);
      double te = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / (double)reps / N;
      printf("  [COST] naive float loop %.2f ns/elem, exact %.2f ns/elem (%.1fx), n=%d\n", tn, te, te / tn, N); (void)sink; }
    printf("%s (%d failures)\n", fails ? "FAIL" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
