/* The int4 sets of 16 (idot.h i4r16_*, matmul_i4p_gidot_r16) against the K1b grouped kernel
 * they stand in for on COLI_DENSE_BITS=4's trunk (i4p_gidot_rows): the same bits for the row
 * counts a decode, a verify and a prompt give, on random planar matrices with random scales of
 * both signs and activations at qrow_i8's contract; a packed row reads back as the plain row.
 * Built with the flags of the build (x86-64-v3: AVX2, FMA); `make i4r16-check` also builds and
 * runs it for AVX-512 VNNI and AVX-VNNI, on a CPU that has them. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static uint32_t rs = 0x9E3779B9u;
static uint32_t xr(void){ rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

#if defined(IDOT_I4R16)
static int check(int O, int I, int S){
    const int rb = I / 2, ng = I / 64;
    uint8_t *q4 = malloc((size_t)O * rb), *q4p = malloc((size_t)O * rb), *tmp = malloc((size_t)16 * rb), *row = malloc((size_t)rb);
    float *sg = malloc((size_t)O * ng * sizeof(float)), *sgp = malloc((size_t)O * ng * sizeof(float));
    float *tsc = malloc((size_t)16 * ng * sizeof(float)), *rsc = malloc((size_t)ng * sizeof(float));
    float *x = malloc((size_t)S * I * sizeof(float)), *sx = malloc((size_t)S * sizeof(float));
    float *y = malloc((size_t)S * O * sizeof(float)), *yref = malloc((size_t)S * O * sizeof(float));
    int8_t *xq = malloc((size_t)S * I); int32_t *xsg = malloc((size_t)S * ng * sizeof(int32_t));
    for(size_t i = 0; i < (size_t)O * rb; i++) q4[i] = (uint8_t)(xr() & 0xFF);
    for(size_t i = 0; i < (size_t)O * ng; i++) sg[i] = ((float)(xr() % 20001) - 10000.f) * 1e-7f;
    for(size_t i = 0; i < (size_t)S * I; i++) x[i] = ((float)(xr() % 4001) - 2000.f) / 500.f;
    for(int s = 0; s < S; s++){
        sx[s] = qrow_i8(x + (size_t)s * I, xq + (size_t)s * I, I);
        for(int g = 0; g < ng; g++){ int32_t t = 0; for(int i = 0; i < 64; i++) t += xq[(size_t)s * I + g * 64 + i]; xsg[(size_t)s * ng + g] = t; }
    }
    i4p_gidot_rows(yref, xq, sx, xsg, q4, sg, S, I, O, 64, 0, O);
    memcpy(q4p, q4, (size_t)O * rb); memcpy(sgp, sg, (size_t)O * ng * sizeof(float));
    for(int t = 0; t < O / 16; t++) i4r16_pack_set(q4p + (size_t)t * 16 * rb, sgp + (size_t)t * 16 * ng, I, tmp, tsc);
    matmul_i4p_gidot_r16(y, xq, sx, xsg, q4p, sgp, S, I, O);
    int rc = 0;
    for(size_t i = 0; i < (size_t)S * O; i++)
        if(memcmp(&y[i], &yref[i], sizeof(float))){
            fprintf(stderr, "FAIL i4r16 O=%d I=%d S=%d row %zu out %zu: %.9g != %.9g\n", O, I, S, i / O, i % O, (double)y[i], (double)yref[i]);
            rc = 1; break;
        }
    for(int id = 0; id < O && !rc; id += 7){
        i4r16_row(row, rsc, q4p, sgp, id, I);
        if(memcmp(row, q4 + (size_t)id * rb, (size_t)rb) || memcmp(rsc, sg + (size_t)id * ng, (size_t)ng * sizeof(float))){
            fprintf(stderr, "FAIL i4r16 O=%d I=%d: packed row %d does not read back\n", O, I, id); rc = 1;
        }
    }
    free(q4); free(q4p); free(tmp); free(row); free(sg); free(sgp); free(tsc); free(rsc);
    free(x); free(sx); free(y); free(yref); free(xq); free(xsg);
    return rc;
}
#endif

int main(void){
#if !defined(IDOT_I4R16)
    puts("test_i4r16: the kernel needs AVX2 and FMA, not in this build: nothing to check");
    return 0;
#else
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    const char *path = "AVX-512 VNNI";
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const char *path = "AVX-512 VNNI, 256 bits";
#elif defined(__AVXVNNI__)
    const char *path = "AVX-VNNI";
#else
    const char *path = "AVX2";
#endif
    static const int shapes[][2] = {{16, 64}, {48, 192}, {32, 5120}, {160, 1088}, {64, 17408}};
    static const int rows[] = {1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 64};
    int fails = 0, n = 0;
    for(size_t a = 0; a < sizeof shapes / sizeof *shapes; a++)
        for(size_t b = 0; b < sizeof rows / sizeof *rows; b++){
            if(shapes[a][1] > 5120 && rows[b] > 9) continue;   /* the widest rows: a decode's and a verify's */
            fails += check(shapes[a][0], shapes[a][1], rows[b]); n++;
        }
    printf("test_i4r16 (%s): %d of %d shapes and row counts with the K1b kernel's bits\n", path, n - fails, n);
    if(fails) return 1;
    puts("PASS");
    return 0;
#endif
}
