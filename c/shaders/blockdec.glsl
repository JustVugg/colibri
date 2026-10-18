// blockdec.glsl: one 32-weight unit of a block-quantized row, dotted with x, for the
// GEMV shaders (qmatmul.comp, qmatmul_gate_up.comp). Formats (the push constant fmt):
//   100 + t   a ggml block type t, the row as the GGUF file stores it (ggml_quants.h's
//             numbering and block layouts; every decoder below is gq_ublk's, so a weight
//             is sw[g] * w - mw[g] over groups of 16 as the CPU's integer form has it)
//   200 int2-g32, 201 int2-g16, 202 int3-g16: colibri's low-bit experts (int2_g32.h,
//             lowbit_g16.h) laid out on the device as codes | group scales | f32 row scale
//   203 vq2 (vq2_g32.h): codes (two index bytes per 8 weights) | 4-bit scales per 32 |
//             f32 row scale, the codebooks in the table buffer at BDT_VQ2
// The including shader defines, before #include:
//   BD_P      a prefix for the function names (one instance per weight buffer)
//   BD_W(i)   uint word i of that buffer
//   BD_X(i)   the activation at element i of the current row
//   BD_G(i)   uint word i of the table buffer (blockdec_table.h), which the backend binds
//             in the scales' place for a block format (they have none)
// and calls BD_P##unit(fmt, row byte offset, unit index, row width) for unit u
// (elements 32u..32u+31). Every block size here is a multiple of 32. No constant array
// is indexed at run time: tables come from the buffer, the low-bit levels and powers of
// three from bytes packed in a word.
#ifndef BD_P
#error "define BD_P, BD_W, BD_X and BD_G before including blockdec.glsl"
#endif
#include "blockdec_table.h"

#ifndef BD_CAT
#define BD_CAT2(a, b) a##b
#define BD_CAT(a, b) BD_CAT2(a, b)
#endif
#define BDN(n) BD_CAT(BD_P, n)

uint BDN(u8)(uint off) { return (BD_W(off >> 2u) >> ((off & 3u) << 3u)) & 0xffu; }
// four bytes from any offset: one word when aligned, two otherwise (blocks of 34, 66, 210
// bytes leave their fields unaligned); every decoder below reads its codes a word at a time
uint BDN(u32)(uint off) {
    uint a = off >> 2u, sh = (off & 3u) << 3u;
    uint lo = BD_W(a);
    return sh == 0u ? lo : (lo >> sh) | (BD_W(a + 1u) << (32u - sh));
}
uint BDN(u16)(uint off) { return BDN(u32)(off) & 0xffffu; }
float BDN(f16)(uint off) { return unpackHalf2x16(BDN(u16)(off)).x; }
int BDN(s8)(uint off) { int v = int(BDN(u8)(off)); return v >= 128 ? v - 256 : v; }

#ifndef BD_HELPERS
#define BD_HELPERS
uint bd_byte(uvec2 v, uint j) { return ((j < 4u ? v.x : v.y) >> ((j & 3u) * 8u)) & 0xffu; }
int bd_sbyte(uvec2 v, uint j) { int b = int(bd_byte(v, j)); return b >= 128 ? b - 256 : b; }
float bd_e8m0_half(uint x) { return uintBitsToFloat(x < 2u ? (0x00200000u << x) : ((x - 1u) << 23)); }
float bd_ue4m3_half(uint x) {
    if (x == 0u || x == 0x7fu) return 0.0;
    uint e = (x >> 3) & 15u, m = x & 7u;
    return e == 0u ? ldexp(float(m), -10) : ldexp(float(8u + m), int(e) - 11);
}
int bd_sb(uint word, uint k) { return bitfieldExtract(int(word), int(8u * k), 8); }   // signed byte k
#endif
uvec2 BDN(g2)(uint base, uint i) { return uvec2(BD_G(base + 2u * i), BD_G(base + 2u * i + 1u)); }

// The common GGUF types straight from the words they read: the same weights and the same
// sums of groups of 16 as the general path below (sw * w - mw), without its array of 32
// codes. 0 when the format has no such path (then the general one runs).
bool BDN(fast_dot)(int fmt, uint rb, uint u, out float r) {
    uint xb = u * 32u;
    if (fmt == 108) {   // Q8_0: the unit is the block (34 bytes)
        uint off = rb + u * 34u;
        float a = 0.0;
        for (uint q4 = 0u; q4 < 8u; q4++) {
            uint wd = BDN(u32)(off + 2u + 4u * q4), i = xb + 4u * q4;
            a += float(bd_sb(wd, 0u)) * BD_X(i) + float(bd_sb(wd, 1u)) * BD_X(i + 1u) +
                 float(bd_sb(wd, 2u)) * BD_X(i + 2u) + float(bd_sb(wd, 3u)) * BD_X(i + 3u);
        }
        r = BDN(f16)(off) * a;
        return true;
    }
    if (fmt == 112 || fmt == 113) {   // Q4_K, Q5_K: a group of 32 with a 6-bit scale and min
        uint e0 = xb & 255u, off = rb + (xb >> 8) * (fmt == 112 ? 144u : 176u), g = e0 >> 5, sc = off + 4u;
        vec2 dm = unpackHalf2x16(BDN(u32)(off));
        uint sv, mv;
        if (g < 4u) { sv = BDN(u8)(sc + g) & 63u; mv = BDN(u8)(sc + g + 4u) & 63u; }
        else {
            sv = (BDN(u8)(sc + g + 4u) & 15u) | ((BDN(u8)(sc + g - 4u) >> 2) & 48u);
            mv = (BDN(u8)(sc + g + 4u) >> 4) | ((BDN(u8)(sc + g) >> 2) & 48u);
        }
        uint q = off + (fmt == 112 ? 16u : 48u) + (g >> 1) * 32u, sh = (g & 1u) * 4u;
        float a = 0.0, xs = 0.0;
        for (uint q4 = 0u; q4 < 8u; q4++) {
            uint wd = BDN(u32)(q + 4u * q4) >> sh, hd = fmt == 113 ? BDN(u32)(off + 16u + 4u * q4) >> g : 0u;
            for (uint k = 0u; k < 4u; k++) {
                float x = BD_X(xb + 4u * q4 + k);
                uint v = (wd >> (8u * k)) & 15u;
                if (fmt == 113) v |= ((hd >> (8u * k)) & 1u) << 4;
                a += float(v) * x; xs += x;
            }
        }
        r = dm.x * float(sv) * a - dm.y * float(mv) * xs;
        return true;
    }
    if (fmt == 114) {   // Q6_K: two groups of 16, each its signed 8-bit scale
        uint e0 = xb & 255u, off = rb + (xb >> 8) * 210u, c = e0 >> 7, r0 = e0 & 127u, nib = r0 >> 6, sh = r0 >> 5;
        uint ql = off + c * 64u + (r0 & 63u), qh = off + 128u + c * 32u, g0 = e0 >> 4;
        float a = 0.0, b = 0.0;
        for (uint q4 = 0u; q4 < 8u; q4++) {
            uint lw = BDN(u32)(ql + 4u * q4) >> (4u * nib), hw = BDN(u32)(qh + 4u * q4) >> (2u * sh);
            float acc = 0.0;
            for (uint k = 0u; k < 4u; k++) {
                int v = int(((lw >> (8u * k)) & 15u) | (((hw >> (8u * k)) & 3u) << 4)) - 32;
                acc += float(v) * BD_X(xb + 4u * q4 + k);
            }
            if (q4 < 4u) a += acc; else b += acc;
        }
        float d = BDN(f16)(off + 208u);
        r = d * (float(BDN(s8)(off + 192u + g0)) * a + float(BDN(s8)(off + 192u + g0 + 1u)) * b);
        return true;
    }
    return false;
}

// sum over the unit's 32 weights of weight * x; I is the row's width (low-bit formats)
float BDN(unit)(int fmt, uint rb, uint u, uint I) {
    { float r; if (BDN(fast_dot)(fmt, rb, u, r)) return r; }
    int wv[32];   /* not w: the including shader's weight buffer may be named w (BD_W) */
    float sA = 0.0, mA = 0.0, sB = 0.0, mB = 0.0;   // scale and min of elements 0..15 and 16..31
    uint xb = u * 32u;
    if (fmt == 203) {
        // vq2: the unit is four vectors a0 b0 .. a3 b3, each A[a] + B[b] (8 signed bytes a row)
        uint cbytes = I / 4u, ub = rb + u * 8u;
        float d = uintBitsToFloat(BDN(u32)(rb + cbytes + I / 64u));
        sA = sB = d * float((BDN(u8)(rb + cbytes + (u >> 1u)) >> (4u * (u & 1u))) & 15u);
        for (uint v = 0u; v < 4u; v++) {
            uvec2 ga = BDN(g2)(BDT_VQ2, BDN(u8)(ub + 2u * v)), gb = BDN(g2)(BDT_VQ2 + 512u, BDN(u8)(ub + 2u * v + 1u));
            for (uint j = 0u; j < 8u; j++) wv[8u * v + j] = bd_sbyte(ga, j) + bd_sbyte(gb, j);
        }
    } else if (fmt >= 200) {
        // colibri low-bit: codes (blocks of 128: 32 bytes of low bits, then 16 of high
        // bits at 3 bits), the group scales (I/32 bytes), the f32 row scale
        int bits = fmt == 202 ? 3 : 2;
        uint blk = u >> 2u, h = u & 3u;
        uint cbytes = I / 128u * (bits == 3 ? 48u : 32u);
        uint boff = rb + blk * (bits == 3 ? 48u : 32u);
        float d = uintBitsToFloat(BDN(u32)(rb + cbytes + I / 32u));   /* through the helper: wv shadows nothing */
        uint qb = BDN(u8)(rb + cbytes + u);
        if (fmt == 200) { sA = sB = d * float(qb); }
        else { sA = d * float(qb & 15u); sB = d * float(qb >> 4); }
        for (uint j = 0u; j < 32u; j++) {
            uint c = (BDN(u8)(boff + j) >> (2u * h)) & 3u;
            if (bits == 3) { uint k = h * 32u + j; c |= ((BDN(u8)(boff + 32u + (k & 15u)) >> (k >> 4u)) & 1u) << 2u; }
            // levels -26 -16 -9 -3 3 9 16 26 and -10 -3 3 10 as signed bytes
            wv[j] = bits == 3 ? bd_sb(c < 4u ? 0xfdf7f0e6u : 0x1a100903u, c & 3u) : bd_sb(0x0a03fdf6u, c);
        }
    } else {
        int t = fmt - 100;
        uint B = BD_G(BDT_BLK + uint(t)), T = BD_G(BDT_BYTES + uint(t));
        uint e0 = xb % B, off = rb + (xb / B) * T;
        uint g0 = e0 / 16u;   // the unit's first group of 16 within the block
        if (t == 2 || t == 3 || t == 6 || t == 7) {        // Q4_0, Q4_1, Q5_0, Q5_1
            bool mn = t == 3 || t == 7;
            float d = BDN(f16)(off), m = mn ? BDN(f16)(off + 2u) : 0.0;
            uint qh = t == 6 ? BDN(u32)(off + 2u) : t == 7 ? BDN(u32)(off + 4u) : 0u;
            uint qs = off + (t == 2 ? 2u : t == 3 ? 4u : t == 6 ? 6u : 8u);
            int z = t == 2 ? 8 : t == 6 ? 16 : 0;
            for (uint q4 = 0u; q4 < 4u; q4++) {
                uint wd = BDN(u32)(qs + 4u * q4);
                for (uint k = 0u; k < 4u; k++) {
                    uint j = 4u * q4 + k, b = (wd >> (8u * k)) & 255u;
                    int lo = int(b & 15u), hi = int(b >> 4);
                    if (t >= 6) { lo |= int((qh >> j) & 1u) << 4; hi |= int((qh >> (j + 16u)) & 1u) << 4; }
                    wv[j] = lo - z; wv[j + 16u] = hi - z;
                }
            }
            sA = sB = d; mA = mB = -m;
        } else if (t == 8) {                                 // Q8_0
            float d = BDN(f16)(off);
            for (uint q4 = 0u; q4 < 8u; q4++) {
                uint wd = BDN(u32)(off + 2u + 4u * q4);
                for (uint k = 0u; k < 4u; k++) wv[4u * q4 + k] = bd_sb(wd, k);
            }
            sA = sB = d;
        } else if (t == 20 || t == 39) {                     // IQ4_NL, MXFP4
            float d = t == 20 ? BDN(f16)(off) : bd_e8m0_half(BDN(u8)(off));
            uint qs = off + (t == 20 ? 2u : 1u), kv = t == 20 ? BDT_IQ4NL : BDT_MXFP4;
            for (uint q4 = 0u; q4 < 4u; q4++) {
                uint wd = BDN(u32)(qs + 4u * q4);
                for (uint k = 0u; k < 4u; k++) {
                    uint j = 4u * q4 + k, b = (wd >> (8u * k)) & 255u;
                    wv[j] = int(BD_G(kv + (b & 15u))); wv[j + 16u] = int(BD_G(kv + (b >> 4)));
                }
            }
            sA = sB = d;
        } else if (t == 42) {                                // Q2_0: element 4k+r in bits 2r of byte k
            float d = BDN(f16)(off);
            for (uint j = 0u; j < 32u; j++) { uint e = e0 + j; wv[j] = int((BDN(u8)(off + 2u + (e >> 2)) >> ((e & 3u) * 2u)) & 3u) - 1; }
            sA = sB = d;
        } else if (t == 41) {                                // Q1_0
            float d = BDN(f16)(off);
            for (uint j = 0u; j < 32u; j++) { uint e = e0 + j; wv[j] = ((BDN(u8)(off + 2u + (e >> 3)) >> (e & 7u)) & 1u) != 0u ? 1 : -1; }
            sA = sB = d;
        } else if (t == 40) {                                // NVFP4: 4 sub-blocks of 16
            uint qs = off + 4u;
            for (uint hh = 0u; hh < 2u; hh++) {
                uint sb = g0 + hh;
                float s = bd_ue4m3_half(BDN(u8)(off + sb));
                if (hh == 0u) sA = s; else sB = s;
                for (uint j = 0u; j < 8u; j++) {
                    uint b = BDN(u8)(qs + 8u * sb + j);
                    wv[16u * hh + j] = int(BD_G(BDT_MXFP4 + (b & 15u))); wv[16u * hh + 8u + j] = int(BD_G(BDT_MXFP4 + (b >> 4)));
                }
            }
        } else if (t == 10) {                                // Q2_K
            float d = BDN(f16)(off + 80u), dmin = BDN(f16)(off + 82u);
            {   // the unit's 32 weights: bytes 32c..32c+31 of the codes, bit pair sh of each
                uint c = e0 >> 7, sh = (e0 >> 5) & 3u, q = off + 16u + c * 32u;
                for (uint q4 = 0u; q4 < 8u; q4++) {
                    uint wd = BDN(u32)(q + 4u * q4);
                    for (uint k = 0u; k < 4u; k++) wv[4u * q4 + k] = int((wd >> (8u * k + 2u * sh)) & 3u);
                }
            }
            uint a = BDN(u8)(off + g0), b = BDN(u8)(off + g0 + 1u);
            sA = d * float(a & 15u); mA = dmin * float(a >> 4); sB = d * float(b & 15u); mB = dmin * float(b >> 4);
        } else if (t == 11) {                                // Q3_K
            float d = BDN(f16)(off + 108u);
            for (uint hh = 0u; hh < 2u; hh++) {
                uint k = g0 + hh;
                uint l = k < 8u ? (BDN(u8)(off + 96u + k) & 15u) : (BDN(u8)(off + 96u + k - 8u) >> 4);
                uint h = (BDN(u8)(off + 96u + 8u + (k & 3u)) >> (2u * (k >> 2))) & 3u;
                float s = d * float(int(l | (h << 4)) - 32);
                if (hh == 0u) sA = s; else sB = s;
            }
            for (uint j = 0u; j < 32u; j++) {
                uint i = e0 + j, c = i >> 7, sh = (i >> 5) & 3u, jj = i & 31u;
                int ql = int((BDN(u8)(off + 32u + c * 32u + jj) >> (2u * sh)) & 3u);
                int hb = int((BDN(u8)(off + (i & 31u)) >> (i >> 5)) & 1u);
                wv[j] = ql - ((hb ^ 1) << 2);
            }
        } else if (t == 12 || t == 13) {                     // Q4_K, Q5_K: groups of 32 with 6-bit scale and min
            float d = BDN(f16)(off), dmin = BDN(f16)(off + 2u);
            uint g = e0 >> 5, sc = off + 4u, qs = off + (t == 12 ? 16u : 48u);
            uint s, m;
            if (g < 4u) { s = BDN(u8)(sc + g) & 63u; m = BDN(u8)(sc + g + 4u) & 63u; }
            else {
                s = (BDN(u8)(sc + g + 4u) & 15u) | ((BDN(u8)(sc + g - 4u) >> 2) & 48u);
                m = (BDN(u8)(sc + g + 4u) >> 4) | ((BDN(u8)(sc + g) >> 2) & 48u);
            }
            sA = sB = d * float(s); mA = mB = dmin * float(m);
            uint q = qs + (g >> 1) * 32u, sh = (g & 1u) * 4u;
            for (uint q4 = 0u; q4 < 8u; q4++) {
                uint wd = BDN(u32)(q + 4u * q4), hd = t == 13 ? BDN(u32)(off + 16u + 4u * q4) : 0u;
                for (uint k = 0u; k < 4u; k++) {
                    int v = int((wd >> (8u * k + sh)) & 15u);
                    if (t == 13) v |= int((hd >> (8u * k + g)) & 1u) << 4;
                    wv[4u * q4 + k] = v;
                }
            }
        } else if (t == 14) {                                // Q6_K
            float d = BDN(f16)(off + 208u);
            {   // the unit is 32 consecutive weights of one 32-run: low nibbles at bytes
                // c*64 + (r & 63), high bit pairs at 128 + c*32 + (r & 31)
                uint c = e0 >> 7, r0 = e0 & 127u, nib = r0 >> 6, sh = r0 >> 5;
                uint ql = off + c * 64u + (r0 & 63u), qh = off + 128u + c * 32u;
                for (uint q4 = 0u; q4 < 8u; q4++) {
                    uint lw = BDN(u32)(ql + 4u * q4), hw = BDN(u32)(qh + 4u * q4);
                    for (uint k = 0u; k < 4u; k++) {
                        int lo = int((lw >> (8u * k + 4u * nib)) & 15u), hi = int((hw >> (8u * k + 2u * sh)) & 3u);
                        wv[4u * q4 + k] = (lo | (hi << 4)) - 32;
                    }
                }
            }
            sA = d * float(BDN(s8)(off + 192u + g0)); sB = d * float(BDN(s8)(off + 192u + g0 + 1u));
        } else if (t == 16) {                                // IQ2_XXS
            float d = BDN(f16)(off);
            uint ib = e0 >> 5, qs = off + 2u + 8u * ib;
            uint a0 = BDN(u32)(qs), a1 = BDN(u32)(qs + 4u);
            sA = sB = d * (0.5 + float(a1 >> 28)) * 0.25;
            for (uint l = 0u; l < 4u; l++) {
                uvec2 gr = BDN(g2)(BDT_IQ2XXS, (a0 >> (8u * l)) & 255u); uint sg = BD_G(BDT_KSIGNS + ((a1 >> (7u * l)) & 127u));
                for (uint j = 0u; j < 8u; j++) { int v = int(bd_byte(gr, j)); wv[l * 8u + j] = ((sg >> j) & 1u) != 0u ? -v : v; }
            }
        } else if (t == 17 || t == 22) {                     // IQ2_XS, IQ2_S
            float d = BDN(f16)(off);
            uint sc = off + (t == 17 ? 66u : 74u);
            for (uint hh = 0u; hh < 2u; hh++) {
                uint si = g0 + hh;
                float s = d * (0.5 + float((BDN(u8)(sc + (si >> 1)) >> (4u * (si & 1u))) & 15u)) * 0.25;
                if (hh == 0u) sA = s; else sB = s;
            }
            for (uint l = 0u; l < 4u; l++) {
                uint k = (e0 >> 3) + l;   // the 8-weight chunk within the block
                uvec2 gr; uint sg;
                if (t == 17) { uint q = BDN(u16)(off + 2u + 2u * k); gr = BDN(g2)(BDT_IQ2XS, q & 511u); sg = BD_G(BDT_KSIGNS + (q >> 9)); }
                else {
                    uint idx = BDN(u8)(off + 2u + k) | (((BDN(u8)(off + 66u + (k >> 2)) >> (2u * (k & 3u))) & 3u) << 8);
                    gr = BDN(g2)(BDT_IQ2S, idx); sg = BDN(u8)(off + 34u + k);
                }
                for (uint j = 0u; j < 8u; j++) { int v = int(bd_byte(gr, j)); wv[l * 8u + j] = ((sg >> j) & 1u) != 0u ? -v : v; }
            }
        } else if (t == 18) {                                // IQ3_XXS
            float d = BDN(f16)(off);
            uint ib = e0 >> 5, qs = off + 2u;
            uint a = BDN(u32)(off + 66u + 4u * ib);
            sA = sB = d * (0.5 + float(a >> 28)) * 0.5;
            for (uint l = 0u; l < 4u; l++) {
                uint sg = BD_G(BDT_KSIGNS + ((a >> (7u * l)) & 127u));
                uint g1 = BD_G(BDT_IQ3XXS + BDN(u8)(qs + 8u * ib + 2u * l)), g2 = BD_G(BDT_IQ3XXS + BDN(u8)(qs + 8u * ib + 2u * l + 1u));
                for (uint j = 0u; j < 4u; j++) {
                    int v1 = int((g1 >> (8u * j)) & 255u), v2 = int((g2 >> (8u * j)) & 255u);
                    wv[l * 8u + j] = ((sg >> j) & 1u) != 0u ? -v1 : v1;
                    wv[l * 8u + j + 4u] = ((sg >> (j + 4u)) & 1u) != 0u ? -v2 : v2;
                }
            }
        } else if (t == 21) {                                // IQ3_S
            float d = BDN(f16)(off);
            uint ib = e0 >> 5;
            sA = sB = d * float(1u + 2u * ((BDN(u8)(off + 106u + (ib >> 1)) >> (4u * (ib & 1u))) & 15u));
            for (uint l = 0u; l < 8u; l++) {
                uint k = (e0 >> 2) + l;   // the 4-weight chunk within the block
                uint idx = BDN(u8)(off + 2u + k) | (((BDN(u8)(off + 66u + (k >> 3)) >> (k & 7u)) & 1u) << 8);
                uint gr = BD_G(BDT_IQ3S + idx);
                for (uint j = 0u; j < 4u; j++) {
                    uint e = k * 4u + j; int v = int((gr >> (8u * j)) & 255u);
                    wv[l * 4u + j] = ((BDN(u8)(off + 74u + (e >> 3)) >> (e & 7u)) & 1u) != 0u ? -v : v;
                }
            }
        } else if (t == 19) {                                // IQ1_S
            float d = BDN(f16)(off);
            uint ib = e0 >> 5;
            uint qh = BDN(u16)(off + 34u + 2u * ib);
            float dl = d * float(2u * ((qh >> 12) & 7u) + 1u);
            int dsign = (qh & 0x8000u) != 0u ? -1 : 1;
            sA = sB = dl * 0.125;
            for (uint l = 0u; l < 4u; l++) {
                uvec2 gr = BDN(g2)(BDT_IQ1S, BDN(u8)(off + 2u + 4u * ib + l) | (((qh >> (3u * l)) & 7u) << 8));
                for (uint j = 0u; j < 8u; j++) wv[l * 8u + j] = 8 * bd_sbyte(gr, j) + dsign;
            }
        } else if (t == 29) {                                // IQ1_M
            uint s0 = BDN(u16)(off + 48u), s1 = BDN(u16)(off + 50u), s2 = BDN(u16)(off + 52u), s3 = BDN(u16)(off + 54u);
            float d = unpackHalf2x16((s0 >> 12) | ((s1 >> 8) & 0xf0u) | ((s2 >> 4) & 0xf00u) | (s3 & 0xf000u)).x;
            for (uint hh = 0u; hh < 2u; hh++) {
                uint s16 = g0 + hh, q = s16 >> 2, sw = q == 0u ? s0 : q == 1u ? s1 : q == 2u ? s2 : s3;
                float s = d * float(2u * ((sw >> (3u * (s16 & 3u))) & 7u) + 1u) * 0.125;
                if (hh == 0u) sA = s; else sB = s;
            }
            for (uint l = 0u; l < 4u; l++) {
                uint k = (e0 >> 3) + l;
                uint h = (BDN(u8)(off + 32u + (k >> 1)) >> (4u * (k & 1u))) & 15u;
                uvec2 gr = BDN(g2)(BDT_IQ1S, BDN(u8)(off + k) | ((h & 7u) << 8));
                int dsign = (h & 8u) != 0u ? -1 : 1;
                for (uint j = 0u; j < 8u; j++) wv[l * 8u + j] = 8 * bd_sbyte(gr, j) + dsign;
            }
        } else if (t == 23) {                                // IQ4_XS
            float d = BDN(f16)(off);
            uint sh = BDN(u16)(off + 2u), ib = e0 >> 5;
            uint ls = ((BDN(u8)(off + 4u + (ib >> 1)) >> (4u * (ib & 1u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4);
            sA = sB = d * float(int(ls) - 32);
            uint q = off + 8u + 16u * ib;
            for (uint j = 0u; j < 16u; j++) { uint b = BDN(u8)(q + j); wv[j] = int(BD_G(BDT_IQ4NL + (b & 15u))); wv[j + 16u] = int(BD_G(BDT_IQ4NL + (b >> 4))); }
        } else if (t == 35) {                                // TQ2_0
            float d = BDN(f16)(off + 64u);
            for (uint j = 0u; j < 32u; j++) {
                uint i = e0 + j, c = i >> 7, s = (i >> 5) & 3u, jj = i & 31u;
                wv[j] = int((BDN(u8)(off + c * 32u + jj) >> (2u * s)) & 3u) - 1;
            }
            sA = sB = d;
        } else if (t == 34) {                                // TQ1_0: base-3 digits
            float d = BDN(f16)(off + 52u);
            for (uint j = 0u; j < 32u; j++) {
                uint e = e0 + j, byte, n;
                if (e < 160u) { n = e >> 5; byte = BDN(u8)(off + (e & 31u)); }
                else if (e < 240u) { uint f = e - 160u; n = f >> 4; byte = BDN(u8)(off + 32u + (f & 15u)); }
                else { uint f = e - 240u; n = f >> 2; byte = BDN(u8)(off + 48u + (f & 3u)); }
                uint v = (byte * (n < 4u ? uint(bd_sb(0x1b090301u, n)) : 81u)) & 255u;   // 3^n
                wv[j] = int((v * 3u) >> 8) - 1;
            }
            sA = sB = d;
        }
    }
    float a = 0.0, b = 0.0, xa = 0.0, xb2 = 0.0;
    for (uint j = 0u; j < 16u; j++) {
        float x0 = BD_X(xb + j), x1 = BD_X(xb + 16u + j);
        a += float(wv[j]) * x0; b += float(wv[j + 16u]) * x1;
        xa += x0; xb2 += x1;
    }
    return sA * a - mA * xa + sB * b - mB * xb2;
}
#undef BDN
