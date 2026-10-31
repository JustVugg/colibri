// bdstep_i8.glsl: bdstep.glsl's 32-weight steps in the CPU's integer form (ggml_quants.h,
// gq_unpack_row and gq_dot_i8): the weights as int8, a scale sw and a min mw per 16, the
// input row quantized to int8 in blocks of 32 (gq_quant_act_ref: d = amax / 127, codes
// rounded to nearest even and clamped to +-127, their sum per 16), and each group's
//     dx[g / 2] * (sw[g] * sum_{i in g} w8[i] * xq[i] - mw[g] * xs[g])
// with the inner sums in integers (packed 4 x 8 dot products). The same integers and
// scales as the CPU's, so the two differ only in the order of the float sums.
// The including shader includes bdstep.glsl first (its loads, tables and helpers) and
// defines, for BSI_NR rows of x (a step's weights decoded once for all of them: a
// speculative verify's rows), for row r:
//   BSI_X(r, i)   int word i of the row's codes (elements 4i..4i+3)
//   BSI_DX(r, b)  the scale of block b (elements 32b..32b+31)
//   BSI_XS(r, g)  the sum of the codes of group g (elements 16g..16g+15)
//   BSI_ACC(r, v) adds the step's float v to the row's sum
#ifndef BDSTEP_I8_GLSL
#define BDSTEP_I8_GLSL
#extension GL_EXT_integer_dot_product : require

// a / b rounded to nearest even, as the CPU's float division rounds it: Vulkan allows a
// division 2.5 ulp off and an fma that is not fused (Lavapipe's), and an ulp of 127 / amax
// moves a code at a tie. The quotient's 25 bits floor(ma 2^24 / mb) of the mantissas from
// the device's division (2.5 ulp: a few units, well inside the 16 steps), made exact by
// the remainder (small, so its low 32 bits are all of it), the remainder for the sticky
// bit. A bit-by-bit division for a device further off cost the GEMV's prologue a fifth
// of IQ2_S's one-row GEMV (29.8 -> 35.1 us on an RTX PRO 4000) without ever running.
// For a, b or a quotient not positive and normal, the device's division
float bsi_div(float a, float b) {
    uint ua = floatBitsToUint(a), ub = floatBitsToUint(b);
    int e = int(ua >> 23) - int(ub >> 23);
    uint ma = (ua & 0x7FFFFFu) | 0x800000u, mb = (ub & 0x7FFFFFu) | 0x800000u;
    if (ma < mb) { ma <<= 1; e--; }
    if ((ua >> 23) - 1u > 253u || (ub >> 23) - 1u > 253u || e < -126 || e > 127) return a / b;
    uint qv = uint(float(ma) / float(mb) * 16777216.0);
    int r = int((ma << 24) - qv * mb);
    for (int i = 0; i < 16 && r < 0; i++) { qv--; r += int(mb); }
    for (int i = 0; i < 16 && r >= int(mb); i++) { qv++; r -= int(mb); }
    uint m = qv >> 1;
    if ((qv & 1u) != 0u && (r != 0 || (m & 1u) != 0u)) m++;
    if (m > 0xFFFFFFu) { m >>= 1; e++; }
    if (e > 127) return a / b;
    return uintBitsToFloat((uint(e + 127) << 23) | (m & 0x7FFFFFu));
}
float bsi_id(float amax) { return amax > 0.0 ? bsi_div(127.0, amax) : 0.0; }
bool bsi_has(int t) { return t == 8 || t == 10 || t == 11 || t == 12 || t == 13 || t == 14 || t == 16 || t == 17 || t == 18 || t == 19 || t == 20 ||
                             t == 21 || t == 22 || t == 23 || t == 29 || t == 42; }
int bsi_dot(uint w, uint r, uint xi) { return dotPacked4x8EXT(int(w), BSI_X(r, xi)); }
// 0xFF in each byte whose bit (of the low four) is set: those bytes are negated
uint bsi_smask(uint b4) { return ((b4 * 0x00204081u) & 0x01010101u) * 0xFFu; }
uint bsi_neg(uint g, uint b4) { uint m = bsi_smask(b4 & 15u); return (g ^ m) + (m & 0x01010101u); }
// four kvalues (IQ4) of the nibbles at bit sh of each byte of w, as packed int8
uint bsi_kv4(uint w, int sh) {
    return (uint(int(bs_kvt[bitfieldExtract(w, sh, 4)])) & 255u) | ((uint(int(bs_kvt[bitfieldExtract(w, sh + 8, 4)])) & 255u) << 8) |
           ((uint(int(bs_kvt[bitfieldExtract(w, sh + 16, 4)])) & 255u) << 16) | ((uint(int(bs_kvt[bitfieldExtract(w, sh + 24, 4)])) & 255u) << 24);
}

// step q (weights 32q..32q+31 of the row at byte rb, ggml type t), for each row of x
#define BSI_ROWS [[unroll]] for (uint r = 0u; r < uint(BSI_NR); r++)
void bsi_step(int t, uint rb, uint q) {
    if (t == 12 || t == 13) {   // Q4_K / Q5_K: chunk c (64 weights), half h: weights 64c + 16h + 0..15 (low nibbles)
        uint b = q >> 3, c = (q >> 1) & 3u, h = q & 1u;   // and 64c + 32 + 16h + 0..15 (high), groups of their own
        uint base = rb + b * (t == 12 ? 144u : 176u);
        uvec4 hd = BS_W4(base >> 4);
        vec2 dm = unpackHalf2x16(hd.x);
        uvec2 lo = bs_k4sm(hd, 2u * c), hi = bs_k4sm(hd, 2u * c + 1u);
        uvec4 qs = BS_W4((base + (t == 12 ? 16u : 48u) + 32u * c + 16u * h) >> 4);
        uvec4 qh = t == 13 ? BS_W4((base + 16u + 16u * h) >> 4) : uvec4(0u);
        uint e = b * 64u + c * 16u + h * 4u;   // the low run's first word of codes
        uint wl[4], wh[4];
        [[unroll]] for (int k = 0; k < 4; k++) {
            wl[k] = qs[k] & 0x0F0F0F0Fu; wh[k] = (qs[k] >> 4) & 0x0F0F0F0Fu;
            if (t == 13) { uint hb = qh[k] >> (2u * c); wl[k] |= (hb & 0x01010101u) << 4; wh[k] |= ((hb >> 1) & 0x01010101u) << 4; }
        }
        uint gl = e >> 2, gh = gl + 2u;   // their groups of 16 (and blocks of 32: g / 2)
        BSI_ROWS {
            int sl = 0, sh = 0;
            [[unroll]] for (int k = 0; k < 4; k++) { sl += bsi_dot(wl[k], r, e + uint(k)); sh += bsi_dot(wh[k], r, e + 8u + uint(k)); }
            BSI_ACC(r, BSI_DX(r, gl >> 1) * (dm.x * float(lo.x) * float(sl) - dm.y * float(lo.y) * float(BSI_XS(r, gl))) +
                       BSI_DX(r, gh >> 1) * (dm.x * float(hi.x) * float(sh) - dm.y * float(hi.y) * float(BSI_XS(r, gh))));
        }
        return;
    }
    if (t == 14) {   // Q6_K: (6-bit code) - 32, the codes' dot less 32 times the sum of x
        uint b = q >> 3, j = q & 7u, n = j >> 2, m = j & 3u;
        uint base = rb + b * 210u, qlo = base + 64u * n + 32u * (m & 1u), qhi = base + 128u + 32u * n;
        uint s2 = bs_ldu(base + 192u + 8u * n + 2u * m);
        float d = bs_f16at(base + 208u);
        uint sh = (m >> 1) * 4u, hs = 2u * m;
        uint w[8];
        [[unroll]] for (int k = 0; k < 8; k++) {
            uint lw = bs_ldu(qlo + 4u * uint(k)), hw = bs_ldu(qhi + 4u * uint(k));
            w[k] = ((lw >> sh) & 0x0F0F0F0Fu) | (((hw >> hs) & 0x03030303u) << 4);
        }
        BSI_ROWS {
            int s0 = 0, s1 = 0;
            [[unroll]] for (int k = 0; k < 8; k++) { int v = bsi_dot(w[k], r, q * 8u + uint(k)); if (k < 4) s0 += v; else s1 += v; }
            s0 -= 32 * BSI_XS(r, 2u * q); s1 -= 32 * BSI_XS(r, 2u * q + 1u);
            BSI_ACC(r, BSI_DX(r, q) * (d * bs_sbyte(s2, 0) * float(s0) + d * bs_sbyte(s2, 1) * float(s1)));
        }
        return;
    }
    if (t == 8) {   // Q8_0: the step is the block
        uint base = rb + q * 34u;
        uint w[8];
        [[unroll]] for (int k = 0; k < 8; k++) w[k] = bs_ldu(base + 2u + 4u * uint(k));
        float d = bs_f16at(base);
        BSI_ROWS {
            int s = 0;
            [[unroll]] for (int k = 0; k < 8; k++) s += bsi_dot(w[k], r, q * 8u + uint(k));
            BSI_ACC(r, BSI_DX(r, q) * d * float(s));
        }
        return;
    }
    if (t == 20 || t == 23) {   // IQ4_NL (the step is the block) / IQ4_XS (8 steps a block): kvalues of the nibbles
        float d; uint qa;
        if (t == 20) { uint base = rb + q * 18u; d = bs_f16at(base); qa = base + 2u; }
        else {
            uint b = q >> 3, ib = q & 7u, base = rb + b * 136u;
            uint w0 = BS_W1(base >> 2), w2 = BS_W1((base >> 2) + 1u);
            uint ls = ((w2 >> (8u * (ib >> 1) + 4u * (ib & 1u))) & 15u) | (((w0 >> (16u + 2u * ib)) & 3u) << 4);
            d = unpackHalf2x16(w0).x * float(int(ls) - 32); qa = base + 8u + 16u * ib;
        }
        uint wl[4], wh[4];
        [[unroll]] for (int k = 0; k < 4; k++) {
            uint wd = t == 20 ? bs_ldu(qa + 4u * uint(k)) : BS_W1((qa >> 2) + uint(k));
            wl[k] = bsi_kv4(wd, 0); wh[k] = bsi_kv4(wd, 4);
        }
        BSI_ROWS {
            int s = 0;
            [[unroll]] for (int k = 0; k < 4; k++) s += bsi_dot(wl[k], r, q * 8u + uint(k)) + bsi_dot(wh[k], r, q * 8u + 4u + uint(k));
            BSI_ACC(r, BSI_DX(r, q) * d * float(s));
        }
        return;
    }
    if (t == 21) {   // IQ3_S: 8 chunks of 4 weights, a grid row each, its signs
        uint b = q >> 3, ib = q & 7u, base = rb + b * 110u;
        uint i0 = bs_ldu(base + 2u + 8u * ib), i1 = bs_ldu(base + 6u + 8u * ib);
        uint hb = bs_u8at(base + 66u + ib), sg = bs_ldu(base + 74u + 4u * ib);
        uint sc = (bs_u8at(base + 106u + (ib >> 1)) >> (4u * (ib & 1u))) & 15u;
        uint w[8];
        [[unroll]] for (int l = 0; l < 8; l++) {
            uint idx = (((l < 4 ? i0 : i1) >> (8u * uint(l & 3))) & 255u) | (((hb >> uint(l)) & 1u) << 8);
            w[l] = bsi_neg(bs_grid[idx], sg >> (4u * uint(l)));
        }
        float d = bs_f16at(base);
        BSI_ROWS {
            int s = 0;
            [[unroll]] for (int l = 0; l < 8; l++) s += bsi_dot(w[l], r, q * 8u + uint(l));
            BSI_ACC(r, BSI_DX(r, q) * d * float(1u + 2u * sc) * float(s));
        }
        return;
    }
    if (t == 22) {   // IQ2_S: 4 chunks of 8 weights, a 10-bit grid index each; a scale per 16
        uint b = q >> 3, ib = q & 7u, base = rb + b * 82u;
        uint ix = bs_ldu(base + 2u + 4u * ib), sg = bs_ldu(base + 34u + 4u * ib);
        uint hb = bs_u8at(base + 66u + ib), scb = bs_u8at(base + 74u + ib);
        float d = bs_f16at(base);
        uint w[8];
        [[unroll]] for (int l = 0; l < 4; l++) {
            uint idx = ((ix >> (8u * uint(l))) & 255u) | (((hb >> (2u * uint(l))) & 3u) << 8);
            uint s8 = sg >> (8u * uint(l));
            w[2 * l] = bsi_neg(bs_grid[2u * idx], s8); w[2 * l + 1] = bsi_neg(bs_grid[2u * idx + 1u], s8 >> 4);
        }
        BSI_ROWS {
            int s0 = 0, s1 = 0;
            [[unroll]] for (int l = 0; l < 4; l++) {
                int v = bsi_dot(w[2 * l], r, q * 8u + 2u * uint(l)) + bsi_dot(w[2 * l + 1], r, q * 8u + 2u * uint(l) + 1u);
                if (l < 2) s0 += v; else s1 += v;
            }
            BSI_ACC(r, BSI_DX(r, q) * (d * (0.5 + float(scb & 15u)) * 0.25 * float(s0) + d * (0.5 + float(scb >> 4)) * 0.25 * float(s1)));
        }
        return;
    }
    if (t == 17) {   // IQ2_XS: 4 entries of a 9-bit grid index and a 7-bit sign index (ksigns); a scale per 16
        uint b = q >> 3, ib = q & 7u, base = rb + b * 74u;
        uint i0 = bs_ldu(base + 2u + 8u * ib), i1 = bs_ldu(base + 6u + 8u * ib), scb = bs_u8at(base + 66u + ib);
        float d = bs_f16at(base);
        uint w[8];
        [[unroll]] for (int l = 0; l < 4; l++) {
            uint e = ((l < 2 ? i0 : i1) >> (16u * uint(l & 1))) & 0xFFFFu, idx = e & 511u, s8 = bs_ks[e >> 9];
            w[2 * l] = bsi_neg(bs_grid[2u * idx], s8); w[2 * l + 1] = bsi_neg(bs_grid[2u * idx + 1u], s8 >> 4);
        }
        BSI_ROWS {
            int s0 = 0, s1 = 0;
            [[unroll]] for (int l = 0; l < 4; l++) {
                int v = bsi_dot(w[2 * l], r, q * 8u + 2u * uint(l)) + bsi_dot(w[2 * l + 1], r, q * 8u + 2u * uint(l) + 1u);
                if (l < 2) s0 += v; else s1 += v;
            }
            BSI_ACC(r, BSI_DX(r, q) * (d * (0.5 + float(scb & 15u)) * 0.25 * float(s0) + d * (0.5 + float(scb >> 4)) * 0.25 * float(s1)));
        }
        return;
    }
    if (t == 18) {   // IQ3_XXS: 8 grid rows of 4, their signs (ksigns) and a 4-bit scale
        uint b = q >> 3, ib = q & 7u, base = rb + b * 98u;
        uint g0 = bs_ldu(base + 2u + 8u * ib), g1 = bs_ldu(base + 6u + 8u * ib), a = bs_ldu(base + 66u + 4u * ib);
        uint w[8];
        [[unroll]] for (int l = 0; l < 4; l++) {
            uint gw = l < 2 ? g0 : g1, sh = 16u * uint(l & 1), s8 = bs_ks[(a >> (7u * uint(l))) & 127u];
            w[2 * l] = bsi_neg(bs_grid[(gw >> sh) & 255u], s8); w[2 * l + 1] = bsi_neg(bs_grid[(gw >> (sh + 8u)) & 255u], s8 >> 4);
        }
        float d = bs_f16at(base);
        BSI_ROWS {
            int s = 0;
            [[unroll]] for (int l = 0; l < 8; l++) s += bsi_dot(w[l], r, q * 8u + uint(l));
            BSI_ACC(r, BSI_DX(r, q) * d * (0.5 + float(a >> 28)) * 0.5 * float(s));
        }
        return;
    }
    if (t == 10) {   // Q2_K: 2-bit codes, a scale and a min per 16
        uint b = q >> 3, s = q & 7u, base = rb + b * 84u, sh = 2u * (s & 3u), qa = base + 16u + 32u * (s >> 2);
        uint s0 = bs_u8at(base + 2u * s), s1 = bs_u8at(base + 2u * s + 1u);
        vec2 dm = unpackHalf2x16(bs_ldu(base + 80u));
        uint w[8];
        [[unroll]] for (int k = 0; k < 8; k++) w[k] = (bs_ldu(qa + 4u * uint(k)) >> sh) & 0x03030303u;
        BSI_ROWS {
            int a0 = 0, a1 = 0;
            [[unroll]] for (int k = 0; k < 8; k++) { int v = bsi_dot(w[k], r, q * 8u + uint(k)); if (k < 4) a0 += v; else a1 += v; }
            BSI_ACC(r, BSI_DX(r, q) * (dm.x * float(s0 & 15u) * float(a0) - dm.y * float(s0 >> 4) * float(BSI_XS(r, 2u * q)) +
                                       dm.x * float(s1 & 15u) * float(a1) - dm.y * float(s1 >> 4) * float(BSI_XS(r, 2u * q + 1u))));
        }
        return;
    }
    if (t == 11) {   // Q3_K: codes 0..7 (the 2 low bits, hmask's bit as 4), less 4 times the sum of x; a scale less 32 per 16
        uint b = q >> 3, s = q & 7u, base = rb + b * 110u, sh = 2u * (s & 3u), qa = base + 32u + 32u * (s >> 2);
        float d = bs_f16at(base + 108u), sw0, sw1;
        [[unroll]] for (int h = 0; h < 2; h++) {
            uint k = 2u * s + uint(h);
            uint l = k < 8u ? (bs_u8at(base + 96u + k) & 15u) : (bs_u8at(base + 88u + k) >> 4);
            uint hh = (bs_u8at(base + 104u + (k & 3u)) >> (2u * (k >> 2))) & 3u;
            float v = d * float(int(l | (hh << 4)) - 32);
            if (h == 0) sw0 = v; else sw1 = v;
        }
        uint w[8];
        [[unroll]] for (int k = 0; k < 8; k++)
            w[k] = ((bs_ldu(qa + 4u * uint(k)) >> sh) & 0x03030303u) | (((bs_ldu(base + 4u * uint(k)) >> s) & 0x01010101u) << 2);
        BSI_ROWS {
            int a0 = 0, a1 = 0;
            [[unroll]] for (int k = 0; k < 8; k++) { int v = bsi_dot(w[k], r, q * 8u + uint(k)); if (k < 4) a0 += v; else a1 += v; }
            a0 -= 4 * BSI_XS(r, 2u * q); a1 -= 4 * BSI_XS(r, 2u * q + 1u);
            BSI_ACC(r, BSI_DX(r, q) * (sw0 * float(a0) + sw1 * float(a1)));
        }
        return;
    }
    if (t == 19 || t == 29) {   // IQ1_S / IQ1_M: 8 * (the grid rows' dot) + the delta's sign times the sum of x
        uint b = q >> 3, ib = q & 7u, base = rb + b * (t == 19 ? 50u : 56u);
        uint qs, qh; float sw0, sw1;
        if (t == 19) {
            qs = bs_ldu(base + 2u + 4u * ib); qh = bs_u16at(base + 34u + 2u * ib);
            sw0 = sw1 = bs_f16at(base) * float(2u * ((qh >> 12) & 7u) + 1u) * 0.125;
        } else {
            qs = bs_ldu(base + 4u * ib); qh = bs_u16at(base + 32u + 2u * ib);
            uint s01 = bs_ldu(base + 48u), s23 = bs_ldu(base + 52u);
            float d = unpackHalf2x16(((s01 >> 12) & 0xFu) | ((s01 >> 24) & 0xF0u) | ((s23 >> 4) & 0xF00u) | ((s23 >> 16) & 0xF000u)).x;
            uint scw = (ib >> 1) == 0u ? s01 : (ib >> 1) == 1u ? s01 >> 16 : (ib >> 1) == 2u ? s23 : s23 >> 16;
            sw0 = d * float(2u * ((scw >> (3u * ((2u * ib) & 3u))) & 7u) + 1u) * 0.125;
            sw1 = d * float(2u * ((scw >> (3u * ((2u * ib + 1u) & 3u))) & 7u) + 1u) * 0.125;
        }
        uint w[8]; int sg[4];
        [[unroll]] for (int l = 0; l < 4; l++) {
            uint idx, h;
            if (t == 19) { idx = ((qs >> (8u * uint(l))) & 255u) | (((qh >> (3u * uint(l))) & 7u) << 8); sg[l] = (qh & 0x8000u) != 0u ? -1 : 1; }
            else { h = (qh >> (4u * uint(l))) & 15u; idx = ((qs >> (8u * uint(l))) & 255u) | ((h & 7u) << 8); sg[l] = (h & 8u) != 0u ? -1 : 1; }
            w[2 * l] = BS_G(BDT_IQ1S + 2u * idx); w[2 * l + 1] = BS_G(BDT_IQ1S + 2u * idx + 1u);
        }
        BSI_ROWS {
            int a0 = 0, a1 = 0;
            [[unroll]] for (int l = 0; l < 4; l++) {
                uint xi = q * 8u + 2u * uint(l);
                int g = bsi_dot(w[2 * l], r, xi) + bsi_dot(w[2 * l + 1], r, xi + 1u);
                int xs8 = bsi_dot(0x01010101u, r, xi) + bsi_dot(0x01010101u, r, xi + 1u);   // the 8 codes' sum
                int v = 8 * g + sg[l] * xs8;
                if (l < 2) a0 += v; else a1 += v;
            }
            BSI_ACC(r, BSI_DX(r, q) * (sw0 * float(a0) + sw1 * float(a1)));
        }
        return;
    }
    if (t == 16) {   // IQ2_XXS: 4 grid indices, their signs (ksigns) and a 4-bit scale
        uint b = q >> 3, ib = q & 7u, base = rb + b * 66u;
        uint i0 = bs_ldu(base + 2u + 8u * ib), i1 = bs_ldu(base + 6u + 8u * ib);
        uint w[8];
        [[unroll]] for (int l = 0; l < 4; l++) {
            uint idx = (i0 >> (8u * uint(l))) & 255u, s8 = bs_ks[(i1 >> (7u * uint(l))) & 127u];
            w[2 * l] = bsi_neg(bs_grid[2u * idx], s8); w[2 * l + 1] = bsi_neg(bs_grid[2u * idx + 1u], s8 >> 4);
        }
        float d = bs_f16at(base);
        BSI_ROWS {
            int s = 0;
            [[unroll]] for (int l = 0; l < 8; l++) s += bsi_dot(w[l], r, q * 8u + uint(l));
            BSI_ACC(r, BSI_DX(r, q) * d * (0.5 + float(i1 >> 28)) * 0.25 * float(s));
        }
        return;
    }
    if (t == 42) {   // Q2_0: codes 0..3, w = code - 1: the codes' dot less the sum of x
        uint b = q >> 1, h = q & 1u, base = rb + b * 18u;
        uint c0 = bs_ldu(base + 2u + 8u * h), c1 = bs_ldu(base + 6u + 8u * h);
        uint w[8];
        [[unroll]] for (int k = 0; k < 2; k++) {
            uint wd = k == 0 ? c0 : c1;
            [[unroll]] for (int j = 0; j < 4; j++) {
                uint by = wd >> (8u * uint(j));
                w[4 * k + j] = (by & 3u) | ((by << 6) & 0x300u) | ((by << 12) & 0x30000u) | ((by << 18) & 0x3000000u);
            }
        }
        float d = bs_f16at(base);
        BSI_ROWS {
            int s = -BSI_XS(r, 2u * q) - BSI_XS(r, 2u * q + 1u);
            [[unroll]] for (int k = 0; k < 8; k++) s += bsi_dot(w[k], r, q * 8u + uint(k));
            BSI_ACC(r, BSI_DX(r, q) * d * float(s));
        }
        return;
    }
}
#undef BSI_ROWS

#endif
