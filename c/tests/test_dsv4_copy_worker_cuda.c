/* V4_COPY_WORKER ordering and exactness: an fp4 mirror refilled through the
 * backend copy worker (dsv4_cuda_tensor_refill_fp4_async) must give the same
 * expert output, bit for bit, as the synchronous refill of the same bytes.
 * Sequence on one mirror slot:
 *   refill E0 sync -> group(E0) (reads the slot) -> post E1 async -> group
 *   expected: group(E1) computed after a synchronous refill of E1
 * The post must wait for the group that read E0 (the copy stream waits on
 * last_read), and the group after the post must wait for the copies (copy_gate).
 * Host slabs of E0 and E1 stay alive until dsv4_cuda_copy_drain returns 1. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

enum { H = 512, I = 256 };

typedef struct { uint8_t *w[3], *s[3]; int O[3], In[3]; } Expert;

static int expert_make(Expert *e, unsigned seed) {
    /* gate, up: O=I, In=H; down: O=H, In=I (fp4 weights, UE8M0 scales) */
    e->O[0] = e->O[1] = I; e->In[0] = e->In[1] = H;
    e->O[2] = H; e->In[2] = I;
    for (int k = 0; k < 3; k++) {
        size_t wb = (size_t)e->O[k] * e->In[k] / 2, sb = (size_t)e->O[k] * e->In[k] / 32;
        e->w[k] = malloc(wb); e->s[k] = malloc(sb);
        if (!e->w[k] || !e->s[k]) return 0;
        for (size_t i = 0; i < wb; i++) e->w[k][i] = (uint8_t)(rand() ^ (int)seed);
        for (size_t i = 0; i < sb; i++) e->s[k][i] = (uint8_t)(118 + rand() % 12);
    }
    return 1;
}

static int refill(Dsv4CudaTensor **m, const Expert *e, int sync) {
    for (int k = 0; k < 3; k++)
        if (!dsv4_cuda_tensor_refill_fp4(m[k], e->w[k], e->s[k], e->O[k], e->In[k], sync)) return 0;
    return 1;
}

static int post(Dsv4CudaTensor **m, const Expert *e) {
    for (int k = 0; k < 3; k++)
        if (dsv4_cuda_tensor_refill_fp4_async(m[k], e->w[k], e->s[k], e->O[k], e->In[k], 0) != 1) return 0;
    return 1;
}

static int group(Dsv4CudaTensor **m, const float *x, float *y) {
    Dsv4CudaTensor *g[1] = { m[0] }, *u[1] = { m[1] }, *d[1] = { m[2] };
    const float w[1] = { 1.0f };
    return dsv4_cuda_expert_group(g, u, d, w, 1, 7.0f, y, x);
}

static int same_bits(const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) return 0;
        if (!isfinite(a[i])) return 0;
    }
    return 1;
}

int main(void) {
    int dev = 0;
    if (!dsv4_cuda_init(&dev, 1)) return 1;
    if (!dsv4_cuda_copy_async_available()) { puts("copy worker: SKIP (backend has no copy worker)"); return 0; }
    srand(7);
    Expert e0, e1;
    if (!expert_make(&e0, 1) || !expert_make(&e1, 2)) return 2;

    Dsv4CudaTensor *m[3] = { NULL, NULL, NULL };
    for (int k = 0; k < 3; k++)
        if (!dsv4_cuda_upload_fp4(&m[k], e0.w[k], e0.s[k], e0.O[k], e0.In[k], dev)) return 3;

    float *x = malloc(H * sizeof(float)), *y_ref = malloc(H * sizeof(float));
    float *y_e0 = malloc(H * sizeof(float)), *y_async = malloc(H * sizeof(float));
    if (!x || !y_ref || !y_e0 || !y_async) return 4;
    for (int i = 0; i < H; i++) x[i] = (float)((i % 13) - 6) / 8.0f;

    /* reference: E1 through the synchronous path */
    if (!refill(m, &e1, 1) || !group(m, x, y_ref)) return 5;
    /* slot holds E0 again, and a launch reads it (the post must wait for this read) */
    if (!refill(m, &e0, 1) || !group(m, x, y_e0)) return 6;
    /* post E1 on the copy worker, then launch: copy_gate orders the launch after the copies */
    if (!post(m, &e1) || !group(m, x, y_async)) return 7;
    if (!dsv4_cuda_copy_drain(dev)) return 8;

    int ok_exact = same_bits(y_async, y_ref, H);
    int sensitive = !same_bits(y_e0, y_ref, H);   /* the test must see E0 and E1 differ */
    printf("copy worker: async refill vs sync refill bit-exact %s; E0 and E1 outputs differ %s\n",
           ok_exact ? "OK" : "FAIL", sensitive ? "OK" : "FAIL");
    return (ok_exact && sensitive) ? 0 : 9;
}
