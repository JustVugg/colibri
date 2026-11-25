/* test_f8_mma_idx: the host-checkable half of the fmt=8 tensor-core expert path
 * (COLI_CUDA_F8_MMA, f8_mma.h + cuda_f8_mma.cuh), on the CPU, no GPU.
 *
 *   1. e4m3 pair widening: f8m_pair_bits read as bf16 and scaled by 2^120 (what the
 *      kernel's fma.rn.bf16x2 computes, exact here: a power-of-two scale of a value
 *      that stays normal) equals quant.h's E4M3_LUT -- the CPU reference -- for all
 *      254 finite codes, both halves, to the bit; the two NaN codes give +-480 (the
 *      documented policy difference from the LUT kernels).
 *   2. A model of one block of f8m_group: the kernel's own stage loops fill emulated
 *      shared tiles, every lane's fragments are read through f8m_wsh/f8m_xsh/comp exactly
 *      as the kernel reads them, mma.m16n8k16 is applied with the PTX ISA's fragment
 *      layout (A row-major m16k16, B col-major k16n8, C m16n8), and the accumulators land
 *      through f8m_c_row/f8m_c_col. Integer-valued operands and power-of-two scales make
 *      every sum exact, so the block must equal the direct dot product to the bit:
 *      a wrong k permutation, a swapped a1/a2, a wrong epilogue column fails it.
 *      Covers M=1 and M=2, RT=1,2,4, a partial last row tile, two scale blocks.
 *   3. Shared-memory bank conflicts of the 16-byte fragment loads (per 8-lane phase):
 *      none, for both tiles.
 *   4. The stage loops copy every 16-byte chunk of a stage exactly once.
 *   5. Static shared memory of every launched instance <= 48 KiB.
 *   6. The dispatch gate (f8m_eligible), the env parse, the row-tile choice.
 *
 * Build: make tests/test_f8_mma_idx */
#include "../quant.h"
#include "../f8_mma.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static float bf16f(uint32_t h) { uint32_t u = (h & 0xFFFFu) << 16; float f; memcpy(&f, &u, 4); return f; }
static uint32_t f32bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* the kernel's fp8pair on the CPU: half `hi` of the bf16x2 result, as float */
static float widen(uint32_t v, int hi) {
    uint32_t b = f8m_pair_bits(v);
    float r = bf16f(hi ? b >> 16 : b) * ldexpf(1.f, 120);
    CHECK((f32bits(r) & 0xFFFFu) == 0, "widened value %a is not a bf16 (fma.rn.bf16x2 would round)", r);
    return r;
}

static void test_widen(void) {
    int bad = 0;
    for (int b = 0; b < 256; b++)
        for (int hi = 0; hi < 2; hi++) {
            uint32_t v = hi ? (uint32_t)b << 8 | 0x38u : (uint32_t)b | 0x3800u;   /* the other byte: 1.0 */
            float got = widen(v, hi), other = widen(v, !hi);
            if (other != 1.f) bad++;                   /* the neighbour code does not leak */
            if ((b & 0x7F) == 0x7F) { if (got != ((b & 0x80) ? -480.f : 480.f)) bad++; }
            else if (f32bits(got) != f32bits(E4M3_LUT[b])) bad++;
        }
    CHECK(!bad, "e4m3 pair widening: %d mismatches against E4M3_LUT", bad);
    printf("e4m3 pair widening vs E4M3_LUT: 254 finite codes x 2 halves bitwise, NaN codes -> +-480: %s\n",
           bad ? "FAIL" : "ok");
}

/* ---- block model ------------------------------------------------------------------ */
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
/* e4m3 codes of small integers: 0, +-1, +-2, +-3, +-4 (exact in bf16 and in the sums) */
static uint8_t rnd_code(void) { static const uint8_t c[9] = {0x00, 0x38, 0xB8, 0x40, 0xC0, 0x44, 0xC4, 0x48, 0xC8}; return c[rnd() % 9]; }
static uint16_t bf16_of_int(int v) { float f = (float)v; return (uint16_t)(f32bits(f) >> 16); }

/* one m16n8k16 with the PTX fragment layouts; a[lane][4], b[lane][2] are bf16x2 words */
static void mma_model(float c[32][4], const uint32_t a[32][4], const uint32_t b[32][2]) {
    float A[16][16], B[16][8];
    for (int lane = 0; lane < 32; lane++) {
        int gq = lane >> 2, t = lane & 3;
        for (int e = 0; e < 2; e++) {
            A[gq][2 * t + e] = bf16f(a[lane][0] >> 16 * e);
            A[gq + 8][2 * t + e] = bf16f(a[lane][1] >> 16 * e);
            A[gq][2 * t + 8 + e] = bf16f(a[lane][2] >> 16 * e);
            A[gq + 8][2 * t + 8 + e] = bf16f(a[lane][3] >> 16 * e);
            B[2 * t + e][gq] = bf16f(b[lane][0] >> 16 * e);
            B[2 * t + 8 + e][gq] = bf16f(b[lane][1] >> 16 * e);
        }
    }
    for (int lane = 0; lane < 32; lane++) {
        int gq = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; i++) {
            int row = gq + 8 * (i >> 1), col = 2 * t + (i & 1);
            double s = 0;
            for (int k = 0; k < 16; k++) s += (double)A[row][k] * B[k][col];
            c[lane][i] += (float)s;
        }
    }
}

static uint32_t word_of(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }

/* f8m_group<M,RT> for one block (column block cb, row block rb of one expert), on the
 * CPU. W[m]: [O][K] codes, S[m]: scales, X: bf16 [rows][K]. Writes out[row][col] for the
 * block's valid rows: M=1 the f32 accumulator, M=2 the gate accumulator and up's in out2. */
static void block_model(int M, int RT, const uint8_t *W[2], const float *S[2], const uint16_t *X, int rows,
                        int K, int O, int cb, int rb, float *out, float *out2) {
    const int BM = 16 * RT, nb0 = cb * F8M_COLS, r0 = rb * BM, nrows = rows - r0 < BM ? rows - r0 : BM;
    static uint8_t sw[2][F8M_COLS * F8M_KSTEP];
    static uint16_t sx[64 * F8M_XSTRIDE];
    static float acc[F8M_WARPS][2][4][4][32][4], part[F8M_WARPS][2][4][4][32][4];
    memset(acc, 0, sizeof acc);
    for (int s = 0; s < K / F8M_KSTEP; s++) {
        const int k0 = s * F8M_KSTEP;
        /* the kernel's stage loops, every thread of the block */
        for (int tid = 0; tid < F8M_WARPS * 32; tid++) {
            for (int i = tid; i < M * F8M_COLS * 4; i += F8M_WARPS * 32) {
                int m = i / (F8M_COLS * 4), row = (i >> 2) & (F8M_COLS - 1), ch = i & 3;
                memcpy(&sw[m][f8m_wsh(row, ch)], W[m] + (size_t)(nb0 + row) * K + k0 + 16 * ch, 16);
            }
            for (int i = tid; i < BM * 8; i += F8M_WARPS * 32) {
                int row = i >> 3, ch = i & 7, in = row < nrows;
                if (in) memcpy(&sx[row * F8M_XSTRIDE + 8 * ch], X + (size_t)(r0 + row) * K + k0 + 8 * ch, 16);
                else memset(&sx[row * F8M_XSTRIDE + 8 * ch], 0, 16);
            }
        }
        if (!(s & 1)) memset(part, 0, sizeof part);
        for (int w = 0; w < F8M_WARPS; w++)
            for (int qq = 0; qq < 2; qq++)
                for (int h = 0; h < 2; h++) {
                    int q = 2 * qq + h;
                    for (int m = 0; m < M; m++)
                        for (int j = 0; j < 4; j++)
                            for (int r = 0; r < RT; r++) {
                                uint32_t a[32][4], b[32][2];
                                for (int lane = 0; lane < 32; lane++) {
                                    int gq = lane >> 2, t = lane & 3;
                                    const uint8_t *wv = &sw[m][f8m_wsh(32 * w + 8 * j + gq, t)];
                                    uint32_t word = word_of(wv + 4 * q);           /* comp(wv, q) */
                                    /* b0 / b1: the code bits; the model widens them below */
                                    uint32_t lo = f8m_pair_bits(word), hi = f8m_pair_bits(word >> 16);
                                    float v[4] = {bf16f(lo) * ldexpf(1.f, 120), bf16f(lo >> 16) * ldexpf(1.f, 120),
                                                  bf16f(hi) * ldexpf(1.f, 120), bf16f(hi >> 16) * ldexpf(1.f, 120)};
                                    b[lane][0] = f32bits(v[0]) >> 16 | (f32bits(v[1]) & 0xFFFF0000u);
                                    b[lane][1] = f32bits(v[2]) >> 16 | (f32bits(v[3]) & 0xFFFF0000u);
                                    const uint16_t *xa = &sx[f8m_xsh(16 * r + gq, t, qq)];
                                    const uint16_t *xc = &sx[f8m_xsh(16 * r + gq + 8, t, qq)];
                                    /* uint4 words .x .y .z .w = elements (0,1) (2,3) (4,5) (6,7) */
                                    a[lane][0] = xa[4 * h] | (uint32_t)xa[4 * h + 1] << 16;
                                    a[lane][1] = xc[4 * h] | (uint32_t)xc[4 * h + 1] << 16;
                                    a[lane][2] = xa[4 * h + 2] | (uint32_t)xa[4 * h + 3] << 16;
                                    a[lane][3] = xc[4 * h + 2] | (uint32_t)xc[4 * h + 3] << 16;
                                }
                                mma_model(part[w][m][r][j], a, b);
                            }
                }
        if (s & 1)
            for (int w = 0; w < F8M_WARPS; w++)
                for (int m = 0; m < M; m++) {
                    float sc = S[m][f8m_scale_at(nb0, s >> 1, K)];
                    for (int r = 0; r < RT; r++)
                        for (int j = 0; j < 4; j++)
                            for (int lane = 0; lane < 32; lane++)
                                for (int i = 0; i < 4; i++)
                                    acc[w][m][r][j][lane][i] = fmaf(part[w][m][r][j][lane][i], sc, acc[w][m][r][j][lane][i]);
                }
    }
    for (int w = 0; w < F8M_WARPS; w++)
        for (int r = 0; r < RT; r++)
            for (int j = 0; j < 4; j++)
                for (int lane = 0; lane < 32; lane++)
                    for (int i = 0; i < 4; i++) {
                        int gq = lane >> 2, t = lane & 3, row = 16 * r + f8m_c_row(gq, i);
                        if (row >= nrows) continue;
                        int col = nb0 + f8m_c_col(w, j, t, i);
                        out[(size_t)(r0 + row) * O + col] = acc[w][0][r][j][lane][i];
                        if (M == 2) out2[(size_t)(r0 + row) * O + col] = acc[w][1][r][j][lane][i];
                    }
}

static int test_block(int M, int RT, int rows, int K, int O) {
    uint8_t *W[2]; float *S[2]; const int KB = K / 128, OB = O / 128;
    for (int m = 0; m < 2; m++) {
        W[m] = malloc((size_t)O * K); S[m] = malloc((size_t)OB * KB * sizeof(float));
        for (size_t i = 0; i < (size_t)O * K; i++) W[m][i] = rnd_code();
        for (int i = 0; i < OB * KB; i++) S[m][i] = ldexpf(1.f, (int)(rnd() % 7) - 3);
    }
    uint16_t *X = malloc((size_t)rows * K * 2);
    int *Xi = malloc((size_t)rows * K * sizeof(int));
    for (size_t i = 0; i < (size_t)rows * K; i++) { Xi[i] = (int)(rnd() % 9) - 4; X[i] = bf16_of_int(Xi[i]); }
    float *out = calloc((size_t)rows * O, 4), *out2 = calloc((size_t)rows * O, 4);
    const uint8_t *Wc[2] = {W[0], W[1]}; const float *Sc[2] = {S[0], S[1]};
    int nrb = (rows + 16 * RT - 1) / (16 * RT);
    for (int cb = 0; cb < OB; cb++)
        for (int rb = 0; rb < nrb; rb++) block_model(M, RT, Wc, Sc, X, rows, K, O, cb, rb, out, out2);
    int bad = 0;
    for (int m = 0; m < M; m++)
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < O; o++) {
                double want = 0;
                for (int kb = 0; kb < KB; kb++) {
                    double p = 0;
                    for (int k = kb * 128; k < kb * 128 + 128; k++) p += (double)E4M3_LUT[W[m][(size_t)o * K + k]] * Xi[(size_t)r * K + k];
                    want += p * S[m][f8m_scale_at(o, kb, K)];
                }
                float got = (m ? out2 : out)[(size_t)r * O + o];
                if ((double)got != want) { if (bad < 4) printf("  M=%d RT=%d row %d col %d m %d: got %g want %g\n", M, RT, r, o, m, got, want); bad++; }
            }
    printf("block model M=%d RT=%d rows=%d K=%d O=%d: %d mismatches against the direct dot\n", M, RT, rows, K, O, bad);
    CHECK(!bad, "block model M=%d RT=%d", M, RT);
    for (int m = 0; m < 2; m++) { free(W[m]); free(S[m]); }
    free(X); free(Xi); free(out); free(out2);
    return bad;
}

/* ---- banks, coverage, smem ----------------------------------------------------- */
/* 16-byte loads are served 8 lanes at a time; within a phase no two lanes may touch the
 * same bank at different addresses. byte(lane) gives the load's shared byte offset. */
static int phase_conflicts(int (*byte)(int lane, int a, int b), int a, int b) {
    int bad = 0;
    for (int ph = 0; ph < 4; ph++) {
        int owner[32]; for (int i = 0; i < 32; i++) owner[i] = -1;
        for (int l = 8 * ph; l < 8 * ph + 8; l++) {
            int o = byte(l, a, b);
            for (int w = 0; w < 4; w++) {
                int bank = (o / 4 + w) % 32, addr = o / 4 + w;
                if (owner[bank] >= 0 && owner[bank] != addr) bad++;
                owner[bank] = addr;
            }
        }
    }
    return bad;
}
static int wbyte(int lane, int w, int j) { return f8m_wsh(32 * w + 8 * j + (lane >> 2), lane & 3); }
static int xbyte_lo(int lane, int r, int qq) { return 2 * f8m_xsh(16 * r + (lane >> 2), lane & 3, qq); }
static int xbyte_hi(int lane, int r, int qq) { return 2 * f8m_xsh(16 * r + (lane >> 2) + 8, lane & 3, qq); }

static void test_banks(void) {
    int bad = 0;
    for (int w = 0; w < F8M_WARPS; w++) for (int j = 0; j < 4; j++) bad += phase_conflicts(wbyte, w, j);
    for (int r = 0; r < 4; r++) for (int qq = 0; qq < 2; qq++)
        bad += phase_conflicts(xbyte_lo, r, qq) + phase_conflicts(xbyte_hi, r, qq);
    CHECK(!bad, "shared-memory bank conflicts: %d", bad);
    for (int r = 0; r < 64; r++) CHECK((2 * f8m_xsh(r, 0, 0)) % 16 == 0, "x row %d not 16-B aligned for cp.async", r);
    printf("16-byte fragment loads: %d bank conflicts (weights and activations)\n", bad);
}

static void test_coverage(void) {
    for (int M = 1; M <= 2; M++) {
        static int seen[2 * F8M_COLS * F8M_KSTEP];
        memset(seen, 0, sizeof seen);
        for (int tid = 0; tid < F8M_WARPS * 32; tid++)
            for (int i = tid; i < M * F8M_COLS * 4; i += F8M_WARPS * 32) {
                int m = i / (F8M_COLS * 4), row = (i >> 2) & (F8M_COLS - 1), ch = i & 3;
                seen[m * F8M_COLS * F8M_KSTEP + f8m_wsh(row, ch)]++;
            }
        int bad = 0;
        for (int i = 0; i < M * F8M_COLS * F8M_KSTEP; i += 16) bad += seen[i] != 1;
        CHECK(!bad, "weight stage M=%d: %d chunks not copied exactly once", M, bad);
    }
    for (int RT = 1; RT <= 4; RT *= 2) {
        static int seen[64 * F8M_XSTRIDE];
        memset(seen, 0, sizeof seen);
        for (int tid = 0; tid < F8M_WARPS * 32; tid++)
            for (int i = tid; i < 16 * RT * 8; i += F8M_WARPS * 32) seen[(i >> 3) * F8M_XSTRIDE + 8 * (i & 7)]++;
        int bad = 0;
        for (int r = 0; r < 16 * RT; r++) for (int c = 0; c < 8; c++) bad += seen[r * F8M_XSTRIDE + 8 * c] != 1;
        CHECK(!bad, "x stage RT=%d: %d chunks not copied exactly once", RT, bad);
    }
    /* the instances f8_mma_launch launches: <2,1> <2,2> <1,1> <1,2> <1,4> */
    static const int inst[5][2] = {{2, 1}, {2, 2}, {1, 1}, {1, 2}, {1, 4}};
    for (int i = 0; i < 5; i++) {
        int M = inst[i][0], RT = inst[i][1];
        int smem = 2 * M * F8M_COLS * F8M_KSTEP + 2 * 16 * RT * F8M_XSTRIDE * 2;
        printf("f8m_group<%d,%d>: %d B static shared\n", M, RT, smem);
        CHECK(smem <= 48 * 1024, "f8m_group<%d,%d> needs %d B of static shared memory", M, RT, smem);
    }
}

static void test_gate(void) {
    CHECK(f8m_parse(NULL, 0) == 0 && f8m_parse("", 7) == 7 && f8m_parse("1", 0) == 1 && f8m_parse("1x", 0) == 0 &&
          f8m_parse("16", 8) == 16, "env parse");
    /* on, min, cc, D, I, max_rows, pin, aligned */
    CHECK(f8m_eligible(1, 8, 120, 2048, 512, 8, 0, 1), "the plain case");
    CHECK(!f8m_eligible(0, 8, 120, 2048, 512, 64, 0, 1), "default off");
    CHECK(!f8m_eligible(2, 8, 120, 2048, 512, 64, 0, 1), "only =1 turns it on");
    CHECK(!f8m_eligible(1, 8, 120, 2048, 512, 7, 0, 1), "below the row threshold");
    CHECK(f8m_eligible(1, 0, 120, 2048, 512, 1, 0, 1), "threshold 0 means 1");
    CHECK(!f8m_eligible(1, 8, 75, 2048, 512, 64, 0, 1), "pre-Ampere");
    CHECK(f8m_eligible(1, 8, 80, 2048, 512, 64, 0, 1), "Ampere");
    CHECK(!f8m_eligible(1, 8, 120, 2048, 512, 64, 1, 1), "pinned (a verify keeps decode bits)");
    CHECK(!f8m_eligible(1, 8, 120, 2048, 512, 64, 0, 0), "unaligned weights");
    CHECK(!f8m_eligible(1, 8, 120, 2048 + 64, 512, 64, 0, 1), "D not whole scale blocks");
    CHECK(!f8m_eligible(1, 8, 120, 2048, 300, 64, 0, 1), "I not whole scale blocks");
    CHECK(!f8m_eligible(1, 8, 120, 0, 512, 64, 0, 1), "empty");
    for (int M = 1; M <= 2; M++)
        for (int r = 1; r <= 4096; r++) {
            int rt = f8m_row_tiles(M, r), nb = f8m_row_blocks(M, r);
            CHECK(rt == 1 || rt == 2 || (rt == 4 && M == 1), "row tiles %d for M=%d", rt, M);
            CHECK(nb * 16 * rt >= r && (nb - 1) * 16 * rt < r, "row blocks M=%d rows=%d", M, r);
        }
    CHECK(f8m_scale_at(0, 3, 2048) == 3 && f8m_scale_at(128, 0, 2048) == 16 && f8m_scale_at(255, 15, 2048) == 31 &&
          f8m_scale_at(256, 1, 300) == 7, "scale geometry: [ceil(O/128)][ceil(K/128)]");
    printf("gate, env parse, row tiles, scale geometry: checked\n");
}

int main(void) {
    test_widen();
    test_banks();
    test_coverage();
    test_gate();
    test_block(1, 1, 13, 256, 128);    /* one partial row tile, two scale blocks */
    test_block(1, 2, 32, 256, 256);
    test_block(1, 4, 70, 256, 128);    /* a second row block with 6 rows */
    test_block(2, 1, 16, 256, 128);
    test_block(2, 2, 40, 128, 256);    /* 32 + 8 */
    if (fails) { printf("test_f8_mma_idx: %d FAILED\n", fails); return 1; }
    printf("test_f8_mma_idx: all passed\n");
    return 0;
}
