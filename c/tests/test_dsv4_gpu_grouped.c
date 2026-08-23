#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "backend_cuda_dsv4.h"

static float e4m3_decode(unsigned char v) {
    int sign = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float val;
    if (!e) val = ldexpf((float)m, -9);
    else if (e == 15) val = m == 7 ? NAN : ldexpf(1.0f + m / 8.0f, 8);
    else val = ldexpf(1.0f + m / 8.0f, e - 7);
    return sign ? -val : val;
}
static unsigned char e4m3_encode(float x) {
    float best = 0, bd = 1e30f;
    unsigned char bc = 0;
    for (int v = 0; v < 255; v++) {
        float z = e4m3_decode((unsigned char)v);
        if (isnan(z)) continue;
        float d = fabsf(z - x);
        if (d < bd || (d == bd && !(v & 1) && (bc & 1))) { bd = d; bc = (unsigned char)v; }
    }
    return bc;
}
static float e8m0(unsigned char v) {
    return v == 255 ? NAN : ldexpf(1.0f, (int)v - 127);
}

int main(void) {
    int device = 0;
    if (!dsv4_cuda_init(&device, 1)) { fprintf(stderr, "no device\n"); return 2; }
    const int groups = 4, o_rank = 128, group_width = 256;
    const int O = groups * o_rank, I = group_width;
    unsigned char *w = malloc((size_t)O * I);
    unsigned char *scales = malloc((size_t)(O / 128) * (I / 128));
    float *x = malloc((size_t)groups * I * sizeof(float));
    float *cpu = malloc((size_t)O * sizeof(float));
    float *gpu = malloc((size_t)O * sizeof(float));
    if (!w || !scales || !x || !cpu || !gpu) return 6;
    for (int i = 0; i < O * I; i++)
        w[i] = e4m3_encode((float)((i % 97) - 48) / 64.0f);
    for (int b = 0; b < (O / 128) * (I / 128); b++)
        scales[b] = (unsigned char)((b % 7) - 3 + 127);
    for (int i = 0; i < groups * I; i++)
        x[i] = (float)((i % 61) - 30) / 16.0f;
    int scale_rows_per_group = (o_rank + 127) / 128;
    int scale_columns = (group_width + 127) / 128;
    for (int g = 0; g < groups; g++)
        for (int o = 0; o < o_rank; o++) {
            float s = 0;
            for (int i = 0; i < I; i++) {
                int b128 = (size_t)g * scale_rows_per_group * scale_columns +
                           (o / 128) * scale_columns + i / 128;
                float wv = e4m3_decode(w[((size_t)g * o_rank + o) * I + i]) *
                           e8m0(scales[b128]);
                s += x[(size_t)g * I + i] * wv;
            }
            cpu[(size_t)g * o_rank + o] = s;
        }
    Dsv4CudaTensor *t = NULL;
    if (!dsv4_cuda_upload_fp8(&t, w, scales, O, I, device)) {
        fprintf(stderr, "upload fail\n"); return 3;
    }
    if (!dsv4_cuda_matvec_grouped(t, gpu, x, groups)) {
        fprintf(stderr, "grouped fail\n"); return 4;
    }
    double worst = 0;
    for (int i = 0; i < O; i++) {
        double d = fabsf(gpu[i] - cpu[i]);
        if (d > worst) worst = d;
        if (fabsf(gpu[i] - cpu[i]) > 1e-2f) {
            fprintf(stderr, "mismatch at %d gpu=%f cpu=%f\n", i, gpu[i], cpu[i]);
            return 5;
        }
    }
    printf("grouped matvec: OK worst=%.2e (O=%d I=%d groups=%d)\n",
           worst, O, I, groups);
    dsv4_cuda_tensor_free(t);
    dsv4_cuda_shutdown();
    return 0;
}
