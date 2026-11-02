/* argmax_first.h -- the first index of the largest of V floats, as the scan
 *     b = 0; for (i = 1; i < V; i++) if (x[i] > x[b]) b = i;
 * picks it: a NaN never wins, a NaN at 0 keeps 0, +0 and -0 tie. That scan reloads x[b] at
 * every step and no compiler vectorizes it: one comparison a nanosecond, 0.21 ms of Qwen3.6's
 * 248,320 logits a greedy token on an RTX PRO 4000 server's cores, 2% of its decode. Here the
 * largest value first, in sixteen lanes of maxima (x > m ? x : m, a vector max), then its first
 * place, sixteen values at a time: 48 us, the same index. */
#ifndef COLI_ARGMAX_FIRST_H
#define COLI_ARGMAX_FIRST_H
static inline int argmax_first(const float *x, int V) {
    if (V < 1 || x[0] != x[0]) return 0;
    float m[16];
    for (int k = 0; k < 16; k++) m[k] = x[0];
    int i = 0;
    for (; i + 16 <= V; i += 16)
        for (int k = 0; k < 16; k++) m[k] = x[i + k] > m[k] ? x[i + k] : m[k];
    float mx = m[0];
    for (int k = 1; k < 16; k++) mx = m[k] > mx ? m[k] : mx;
    for (; i < V; i++) mx = x[i] > mx ? x[i] : mx;
    for (i = 0; i + 16 <= V; i += 16) {
        int any = 0;
        for (int k = 0; k < 16; k++) any |= x[i + k] == mx;
        if (any) break;
    }
    for (; i < V; i++) if (x[i] == mx) return i;
    return 0;
}
#endif
