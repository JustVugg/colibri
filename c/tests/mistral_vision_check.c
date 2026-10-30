/* Mistral-Large-4's tower (mistral_vision.h) on raw patches, for tools/mistral_vision_ref.py:
 *   mistral_vision_check MMPROJ PATCHES.f32 GRID_H GRID_W OUT.f32 OUT_HIDDEN EPS */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../mistral_vision.h"
int main(int argc, char **argv) {
    if (argc != 8) { fprintf(stderr, "usage: mistral_vision_check MMPROJ PATCHES.f32 GRID_H GRID_W OUT.f32 OUT_HIDDEN EPS\n"); return 2; }
    int gh = atoi(argv[3]), gw = atoi(argv[4]), oh = atoi(argv[6]);
    MistralVision v;
    if (mistral_vision_load(&v, argv[1], oh, (float)atof(argv[7]))) return 1;
    size_t pd = (size_t)v.in_ch * v.patch * v.patch, n = (size_t)gh * gw * pd;
    float *p = malloc(n * sizeof(float));
    FILE *f = fopen(argv[2], "rb");
    if (!p || !f || fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "patches: %zu floats expected\n", n); return 1; }
    fclose(f);
    int rows = (gh / v.merge) * (gw / v.merge);
    float *out = malloc((size_t)rows * oh * sizeof(float));
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    int got = mistral_vision_forward(&v, p, gh, gw, out);
    clock_gettime(CLOCK_MONOTONIC, &b);
    if (got != rows) { fprintf(stderr, "forward: %d rows, %d expected\n", got, rows); return 1; }
    f = fopen(argv[5], "wb"); fwrite(out, sizeof(float), (size_t)rows * oh, f); fclose(f);
    printf("%d rows of %d in %.2f s\n", rows, oh, (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9);
    return 0;
}
