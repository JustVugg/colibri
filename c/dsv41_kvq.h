#ifndef DSV41_KVQ_H
#define DSV41_KVQ_H
/* The packed cache's codec (COLI_VK_KV_FP32 unset): the DeepSeek V4.1 device KV mirrors in the checkpoint's
 * quantized formats, as DeepInfer's cache defines them (model_executor/layers/
 * tilelang_ops/cache_common.py) and quantizes them (asc/kv_quant.py, compute_scale):
 *
 *   win   e4m3 + a ue8m0 scale per 32, [data | scales] a token: 512 B + 16 B = 528 B/token
 *   ckv   e2m1 + an e4m3 scale per 16, [data | scales] a token: 256 B + 32 B = 288 B/token
 *   ikey  e2m1 + a ue8m0 scale per 32, block-wide sections (a block = 32 tokens: 2048 B of
 *         data then 128 B of scales, a token's 4 scale bytes one packed int32): 68 B/token
 *
 * The scale of a group is the ceil power of two covering amax/quant_max for ue8m0
 * (exponent = ((bits(amax/quant_max) - 1) >> 23) + 1, decoded 2^(b-127) exactly), or
 * e4m3-RNE of min(clamped_amax/6, 448) whose DECODED value scales the group (the fp4
 * main cache); where that scale would leave the group's scaled amax at the e2m1 top tie
 * or below it, the packer steps it down a code (kvq_pack_ckv's note) so a dequantized
 * row re-packs to itself -- the one place colibri's ckv bytes differ from DI's, which
 * never re-packs a row. The amax margins floor the scale away from zero (FP8 1e-4,
 * FP4 6*2^-126, the e4m3-scale path 2^-9 * 6). Values are rounded to nearest, ties to
 * the even code (RNE),
 * the e4m3 codes as dsv4_quant.h's dsv4_e4m3_round picks them. The host keeps its state
 * fp32 holding the DEQUANTIZED rows (a pull's write-back and a re-upload's source), so the
 * reference here is what the device's pack (chain_dsv4.comp mode 13) must reproduce bit
 * for bit and its unpack (mode 14) reads back. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "dsv4_quant.h"

#define KVQ_WIN_BYTES 528     /* a window row: 512 e4m3 data bytes then 16 ue8m0 scales */
#define KVQ_CKV_BYTES 288     /* a compressed row: 256 e2m1 data bytes then 32 e4m3 scales */
#define KVQ_IKEY_BYTES 68     /* an index-key row: 64 data bytes + 4 scale bytes, block-wide */
#define KVQ_IKEY_BLOCK 32     /* the index-key block: 32 rows of data then 32 rows of scales */
/* an index-key mirror of cap rows: the blocks are whole, so a partial last block's
 * scales still sit at its block's 2048-byte mark -- 68 B a row on average, the buffer
 * rounded up to the block (cap * 68 alone would cut a partial block's scales off) */
static inline size_t kvq_ikey_mirror_bytes(size_t cap) {
    return ((cap + KVQ_IKEY_BLOCK - 1) / KVQ_IKEY_BLOCK) * KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES;
}

static const float KVQ_FP8_MARGIN = 1e-4f;             /* DI's FP8_AMAX_MARGIN */
static const float KVQ_FP4_MARGIN = 6.0f * 1.17549435e-38f;   /* 6 * 2^-126, DI's FP4_AMAX_MARGIN */
static const float KVQ_E4M3SF_MIN = 1.953125e-3f * 6;  /* 2^-9 * 6, DI's e4m3-scale clamp floor */

static inline uint8_t kvq_e4m3_code(float x) {   /* RNE, dsv4_e4m3_round's code (ties to even);
    a value whose nearest code is a zero keeps its sign, as the device's packer does */
    if (x == 0.0f) return signbit(x) ? 0x80 : 0x00;
    int best = 0; float bd = INFINITY;
    for (int v = 0; v < 255; v++) {
        float q = dsv4_e4m3((uint8_t)v);
        float d = fabsf(q - x);
        if (d < bd || (d == bd && !(v & 1) && (best & 1))) { bd = d; best = v; }
    }
    if ((best & 0x7f) == 0) return signbit(x) ? 0x80 : 0x00;
    return (uint8_t)best;
}
static inline uint8_t kvq_e2m1_code(float x) {   /* |x| clamped by the ladder's last arm; ties to the even code */
    float a = fabsf(x);
    uint8_t s = signbit(x) ? 8 : 0;
    uint8_t c;
    if (a <= 0.25f) c = 0;              /* ties at .25/1.25/2.5/5 -> 0/1.0/2.0/4.0, at .75/1.75/3.5 -> 1.0/2.0/4.0 */
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.0f) c = 6;
    else c = 7;
    return (uint8_t)(s | c);
}
static inline float kvq_e2m1(uint8_t nib) {
    static const float g[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float v = g[nib & 7];
    return (nib & 8) ? -v : v;
}
static inline float kvq_ue8m0(uint8_t b) { return ldexpf(1.0f, (int)b - 127); }   /* 2^(b-127), st.h's ue8m0_to_f32 */
static inline uint8_t kvq_ue8m0_scale(float amax, float qmax) {   /* the ceil-pow2 exponent byte */
    uint32_t u; float r = amax * (1.0f / qmax);   /* a multiply: the device's 1/x is not to be trusted */
    memcpy(&u, &r, 4);
    return (uint8_t)(((u - 1u) >> 23) + 1u);
}
/* the exact RNE reciprocal of dsv4_e4m3(sf), as bit math -- DI's reciprocal-LUT trick
 * (asc/kv_quant.py: RNE(8/(8+m)) with an exponent adjustment is exactly RNE(1/sf)),
 * because a GPU's 1/x may be a fast reciprocal a couple of ulps low and the codec needs
 * the host's and the device's bits to agree. A subnormal sf (m * 2^-9) reciprocals to
 * 512/m; both tables are exact. */
static inline float kvq_e4m3_inv(uint8_t sf) {
    static const uint32_t rn[8] = { 0x3f800000u, 0x3f638e39u, 0x3f4ccccdu, 0x3f3a2e8cu, 0x3f2aaaabu, 0x3f1d89d9u, 0x3f124925u, 0x3f088889u };
    static const uint32_t rs[8] = { 0x3f800000u, 0x44000000u, 0x43800000u, 0x432aaaabu, 0x43000000u, 0x42cccccdu, 0x42aaaaabu, 0x42924925u };
    int e = (sf >> 3) & 15, m = sf & 7;
    uint32_t u = (e ? rn[m] + (uint32_t)(7 - e) * 0x800000u : rs[m]) | ((uint32_t)(sf & 0x80) << 24);   /* (7-e)<<23, the sign */
    float f; memcpy(&f, &u, 4);
    return f;
}
static inline float kvq_ue8m0_inv(uint8_t b) {   /* 2^(127-b), exact: DI's (254 - b) << 23 */
    uint32_t u = (uint32_t)(254 - b) << 23;
    float f; memcpy(&f, &u, 4);
    return f;
}
static inline float kvq_amax(const float *x, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    return mx;
}

/* one window row: x[0..511] -> 528 B (512 e4m3 bytes, then a ue8m0 byte a group of 32) */
static inline void kvq_pack_win(uint8_t dst[KVQ_WIN_BYTES], const float *x) {
    for (int g = 0; g < 16; g++) {
        const float *p = x + g * 32;
        float mx = kvq_amax(p, 32);
        if (mx < KVQ_FP8_MARGIN) mx = KVQ_FP8_MARGIN;
        uint8_t sf = kvq_ue8m0_scale(mx, 448.0f);
        float inv = kvq_ue8m0_inv(sf);
        for (int i = 0; i < 32; i++) dst[g * 32 + i] = kvq_e4m3_code(p[i] * inv);
        dst[512 + g] = sf;
    }
}
/* one compressed row: x[0..511] -> 288 B (256 e2m1 bytes, then an e4m3 byte a group of 16) */
static inline void kvq_pack_ckv(uint8_t dst[KVQ_CKV_BYTES], const float *x) {
    for (int g = 0; g < 32; g++) {
        const float *p = x + g * 16;
        float cl = kvq_amax(p, 16);
        if (cl < KVQ_E4M3SF_MIN) cl = KVQ_E4M3SF_MIN;
        float sff = cl * (1.0f / 6.0f);
        if (sff > 448.0f) sff = 448.0f;
        uint8_t sf = kvq_e4m3_code(sff);
        float inv = kvq_e4m3_inv(sf);   /* the group quantizes with the scale as stored */
        if (cl * inv <= 5.0f) {   /* the near-dead band's demote: with the scaled amax at
         * the e2m1 top tie or under it (a group amax in [9*2^-9, 10*2^-9]: sf 0x02, the
         * amax element 4.5-5.0 -> grid 4 -> 8*2^-9), the pull's dequantize-and-re-pack
         * cycle would re-pick sf 0x01 and hop the group's large elements once (8*2^-9 ->
         * 6*2^-9) -- a pack(unpack(pack)) value fixpoint needs the smaller scale up front.
         * Only a subnormal sf >= 0x02 can trip it (a normal sf keeps the scaled amax
         * above 5.6, and 0x01's groups sit at >= 6 by the clamp floor), and the demoted
         * scale puts the scaled amax at 9-10, the 6-grid: the re-pack re-picks it. The
         * first-pack cost in the band is the amax decoding to 6*2^-9, 33-40% low against
         * 11-20% had sf 0x02 kept -- near-dead groups only, and DI never re-packs, so
         * outside the band the bytes are DI's */
            sf--;
            inv = kvq_e4m3_inv(sf);
        }
        for (int i = 0; i < 8; i++)   /* 16 elements, two nibbles a byte */
            dst[g * 8 + i] = (uint8_t)(kvq_e2m1_code(p[2 * i] * inv) | (kvq_e2m1_code(p[2 * i + 1] * inv) << 4));
        dst[256 + g] = sf;
    }
}
/* one index-key row: x[0..127] into token t of a block-wide-sections mirror of cap rows */
static inline void kvq_pack_ikey(uint8_t *mirror, int t, const float *x) {
    size_t blk = (size_t)(t / KVQ_IKEY_BLOCK) * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES);
    uint8_t *db = mirror + blk + (size_t)(t % KVQ_IKEY_BLOCK) * 64, *sb = mirror + blk + 2048 + (size_t)(t % KVQ_IKEY_BLOCK) * 4;
    for (int g = 0; g < 4; g++) {
        const float *p = x + g * 32;
        float mx = kvq_amax(p, 32);
        if (mx < KVQ_FP4_MARGIN) mx = KVQ_FP4_MARGIN;
        uint8_t sf = kvq_ue8m0_scale(mx, 6.0f);
        float inv = kvq_ue8m0_inv(sf);
        for (int i = 0; i < 16; i++)
            db[g * 16 + i] = (uint8_t)(kvq_e2m1_code(p[2 * i] * inv) | (kvq_e2m1_code(p[2 * i + 1] * inv) << 4));
        sb[g] = sf;
    }
}
static inline void kvq_unpack_win(float *x, const uint8_t *src) {
    for (int i = 0; i < 512; i++) x[i] = dsv4_e4m3(src[i]) * kvq_ue8m0(src[512 + (i >> 5)]);
}
static inline void kvq_unpack_ckv(float *x, const uint8_t *src) {
    for (int i = 0; i < 512; i++) {
        uint8_t b = src[i >> 1];
        x[i] = kvq_e2m1((i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 15)) * dsv4_e4m3(src[256 + (i >> 4)]);
    }
}
static inline void kvq_unpack_ikey(float *x, const uint8_t *mirror, int t) {
    const uint8_t *db = mirror + (size_t)(t / KVQ_IKEY_BLOCK) * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES) + (size_t)(t % KVQ_IKEY_BLOCK) * 64;
    const uint8_t *sb = mirror + (size_t)(t / KVQ_IKEY_BLOCK) * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES) + 2048 + (size_t)(t % KVQ_IKEY_BLOCK) * 4;
    for (int i = 0; i < 128; i++) {
        uint8_t b = db[i >> 1];
        x[i] = kvq_e2m1((i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 15)) * kvq_ue8m0(sb[i >> 5]);
    }
}

/* The host's own block rules (the packed cache, the host arrays held in these
 * formats): a mirror's blocks are whole and pack from byte 0, so the whole blocks
 * covering rows [0, rows) are exactly the linear prefix kvq_ikey_mirror_bytes(rows)
 * long -- a park, a reset or a zero-fill that rounds its row count up to whole blocks
 * may copy that prefix as bytes, and NOTHING else may go linear: a row's data and its
 * scales sit 2048 B apart inside its block, so a prefix cut at a row boundary strands
 * the tail rows' scales. These two move ONE row between a mirror and the 68-byte
 * linear staging a boundary writes through (a pull's write-back, a raw upload). */
static inline void kvq_ikey_row_gather(uint8_t row[KVQ_IKEY_BYTES], const uint8_t *mirror, int t) {
    size_t blk = (size_t)(t / KVQ_IKEY_BLOCK) * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES);
    memcpy(row, mirror + blk + (size_t)(t % KVQ_IKEY_BLOCK) * 64, 64);
    memcpy(row + 64, mirror + blk + 2048 + (size_t)(t % KVQ_IKEY_BLOCK) * 4, 4);
}
static inline void kvq_ikey_row_scatter(uint8_t *mirror, int t, const uint8_t row[KVQ_IKEY_BYTES]) {
    size_t blk = (size_t)(t / KVQ_IKEY_BLOCK) * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES);
    memcpy(mirror + blk + (size_t)(t % KVQ_IKEY_BLOCK) * 64, row, 64);
    memcpy(mirror + blk + 2048 + (size_t)(t % KVQ_IKEY_BLOCK) * 4, row + 64, 4);
}
#endif
