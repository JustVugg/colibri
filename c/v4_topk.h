/* v4_topk.h: exact top-k for the DeepSeek V4 sparse-attention indexer.
 *
 * The indexer ranks its candidates with qsort and the comparator "score
 * descending, then index ascending". Every candidate has its own index, so for
 * scores without NaN that comparator is a strict total order and the sorted
 * array is unique: any correct sort gives the same indices in the same order.
 * v4_topk_desc gets there with a stable LSD radix sort on a 32-bit key that
 * orders floats descending. The input enters in index order and the sort is
 * stable, so equal scores stay in ascending index order, as the comparator
 * wants. -0 maps onto +0 first: the comparator sees them as equal, the raw bit
 * patterns would not.
 *
 * A NaN makes the comparator inconsistent (NaN is neither larger nor smaller,
 * so qsort's result depends on its algorithm). v4_topk_desc refuses those rows
 * with -1 and the caller keeps its qsort, so the selection stays the old one.
 *
 * Cost: four counting passes over count keys (a pass is skipped when every key
 * has the same byte there) instead of count*log2(count) indirect comparator
 * calls. */
#ifndef V4_TOPK_H
#define V4_TOPK_H

#include <stdint.h>
#include <string.h>

/* Larger key = ranked later: descending score. */
static inline uint32_t v4_topk_key(float score) {
    uint32_t bits;
    memcpy(&bits, &score, sizeof(bits));
    if (bits == 0x80000000u) bits = 0;                 /* -0 ties with +0 */
    bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);   /* ascending */
    return ~bits;
}

/* Writes min(count, k) candidate indices to out, best first. work holds
 * 2*count uint64_t. Returns the number written, or -1 (nothing written) when a
 * score is NaN. */
static inline int v4_topk_desc(int *out, const float *scores, int count, int k,
                               uint64_t *work) {
    if (count <= 0 || k <= 0) return 0;
    uint64_t *a = work, *b = work + count;
    for (int i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, &scores[i], sizeof(bits));
        if ((bits & 0x7fffffffu) > 0x7f800000u) return -1;
        a[i] = ((uint64_t)v4_topk_key(scores[i]) << 32) | (uint32_t)i;
    }
    for (int shift = 32; shift < 64; shift += 8) {
        uint32_t histogram[256] = {0};
        for (int i = 0; i < count; i++) histogram[(a[i] >> shift) & 255]++;
        if (histogram[(a[0] >> shift) & 255] == (uint32_t)count) continue;
        uint32_t position = 0;
        for (int digit = 0; digit < 256; digit++) {
            uint32_t n = histogram[digit];
            histogram[digit] = position;
            position += n;
        }
        for (int i = 0; i < count; i++) b[histogram[(a[i] >> shift) & 255]++] = a[i];
        uint64_t *swap = a; a = b; b = swap;
    }
    int n = count < k ? count : k;
    for (int i = 0; i < n; i++) out[i] = (int)(uint32_t)a[i];
    return n;
}

#endif /* V4_TOPK_H */
