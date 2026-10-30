#include "../v41_topk.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); \
} } while (0)

static int reference(const float *scores, int width, int limit, int *out) {
    if (limit > width) limit = width;
    int count = 0;
    for (int k = 0; k < limit; k++) {
        int best = -1;
        for (int j = 0; j < width; j++) {
            if (scores[j] == -INFINITY) continue;
            int seen = 0;
            for (int u = 0; u < count; u++)
                if (out[u] == j) { seen = 1; break; }
            if (seen) continue;
            if (best < 0 || scores[j] > scores[best]) best = j;
        }
        if (best < 0) break;
        out[count++] = best;
    }
    for (int a = 0; a < count; a++)
        for (int b = a + 1; b < count; b++)
            if (out[b] < out[a]) { int t = out[a]; out[a] = out[b]; out[b] = t; }
    for (int i = count; i < limit; i++) out[i] = -1;
    return count;
}

static uint32_t rng_state = 0x61c88647u;
static uint32_t next_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static void compare(const float *scores, int width, int limit) {
    int *expected = calloc((size_t)width + 1, sizeof(int));
    int *actual = calloc((size_t)width + 1, sizeof(int));
    uint8_t *selected = calloc((size_t)width + 1, 1);
    CHECK(expected && actual && selected);
    int ne = reference(scores, width, limit, expected);
    int na = v41_topk_positions(scores, width, limit, actual, selected);
    CHECK(ne == na);
    for (int i = 0; i < (limit < width ? limit : width); i++)
        CHECK(expected[i] == actual[i]);
    free(expected); free(actual); free(selected);
}

int main(void) {
    float scores[1220];
    for (int width = 1; width <= 80; width++) {
        for (int repetition = 0; repetition < 8; repetition++) {
            for (int i = 0; i < width; i++) {
                uint32_t value = next_u32();
                scores[i] = (value % 19 == 0) ? -INFINITY
                           : (float)((int)(value % 31) - 15) * 0.25f;
            }
            for (int limit = 0; limit <= width + 2; limit++)
                compare(scores, width, limit);
        }
    }

    /* NaNs retain the historical left-to-right selection behavior. */
    scores[0] = NAN; scores[1] = 9.0f; scores[2] = -INFINITY;
    compare(scores, 3, 2);
    const float special[] = {INFINITY, -INFINITY, 0.0f, -0.0f, INFINITY, NAN, -3.0f};
    for (int width=0;width<=7;width++)
        for (int limit=0;limit<=width+2;limit++) compare(special,width,limit);
    for (int i=0;i<80;i++) scores[i]=-INFINITY;
    compare(scores,80,80);
    for (int i=0;i<80;i++) scores[i]=(i&1)?-0.0f:0.0f;
    compare(scores,80,37);
    int expected[512], actual[512];
    uint8_t selected[1220];
    const int widths[] = {610, 1220};
    for (int shape = 0; shape < 2; shape++) {
        int width = widths[shape];
        for (int i = 0; i < width; i++)
            scores[i] = (float)((int)(next_u32() % 100003) - 50001) * 0.0001f;
        int ne = reference(scores, width, 512, expected);
        int na = v41_topk_positions(scores, width, 512, actual, selected);
        CHECK(ne == na);
        for (int i = 0; i < 512; i++) CHECK(expected[i] == actual[i]);
        printf("v41 top-k exact selection: PASS width=%d limit=512\n", width);
    }
    return 0;
}
