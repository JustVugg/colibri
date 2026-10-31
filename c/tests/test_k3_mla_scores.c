/* The gated MLA's CPU kernels against their arithmetic spelled out: k3_mla_scores (one
 * head) and k3_mla_scores_g (up to K3_MLA_G heads) give every position the products of
 * q and its cached row rounded, then added in the row's order from zero, then times the
 * scale; k3_rowdot8_i8 gives each of 8 int8 rows its widened weights times x rounded,
 * added in order, then times the row's scale. Positions 1 to 40 (whole and partial
 * blocks of 16), Kimi K3's widths (512 + 64) and the tiny model's (32 + 8), and none for
 * the rope part. Exits 77 without AVX2 (the kernels' fast paths are AVX2).
 *   make tests/test_k3_mla_scores.exe && tests/test_k3_mla_scores.exe */
#define main coli_k3_main_unused
#include "../kimi_k3.c"
#undef main

#include <stdio.h>

static uint32_t rng = 777;
static float rnd(void){ rng = rng * 1664525u + 1013904223u; return ((float)(rng >> 9) / 4194304.0f - 1.0f) * 3.0f; }
/* a product that is rounded where it is computed */
static float prod(float a, float b){ volatile float p = a * b; return p; }

int main(void){
#if !defined(__AVX2__)
    printf("no AVX2: skipped\n");
    return 77;
#else
    int bad = 0, checked = 0;
    struct { int kvl, qr; } w[] = { {512, 64}, {32, 8}, {24, 0} };
    for(size_t c = 0; c < sizeof w / sizeof w[0]; c++){
        int kvl = w[c].kvl, qr = w[c].qr, T = 40;
        float *L = malloc((size_t)T * kvl * sizeof(float)), *R = malloc((size_t)T * (qr ? qr : 1) * sizeof(float));
        float *qa = malloc((size_t)K3_MLA_G * kvl * sizeof(float)), *qr6 = malloc((size_t)K3_MLA_G * (qr ? qr : 1) * sizeof(float));
        for(int i = 0; i < T * kvl; i++) L[i] = rnd();
        for(int i = 0; i < T * qr; i++) R[i] = rnd();
        for(int i = 0; i < K3_MLA_G * kvl; i++) qa[i] = rnd();
        for(int i = 0; i < K3_MLA_G * qr; i++) qr6[i] = rnd();
        const float *qrp[K3_MLA_G];
        for(int g = 0; g < K3_MLA_G; g++) qrp[g] = qr6 + (size_t)g * qr;
        float scale = 0.0722f;
        for(int nt = 1; nt <= T; nt++){
            float *one = malloc((size_t)nt * sizeof(float)), *grp = malloc((size_t)K3_MLA_G * nt * sizeof(float));
            for(int nh = 1; nh <= K3_MLA_G; nh++){
                k3_mla_scores_g(grp, qa, qrp, nh, L, R, nt, kvl, qr, scale);
                for(int g = 0; g < nh; g++){
                    k3_mla_scores(one, qa + (size_t)g * kvl, qrp[g], L, R, nt, kvl, qr, scale);
                    for(int t = 0; t < nt; t++){
                        float s = 0;
                        for(int i = 0; i < kvl; i++) s = s + prod(qa[(size_t)g * kvl + i], L[(size_t)t * kvl + i]);
                        for(int i = 0; i < qr; i++) s = s + prod(qrp[g][i], R[(size_t)t * qr + i]);
                        s = s * scale;
                        if(memcmp(&s, &one[t], 4) || memcmp(&s, &grp[(size_t)g * nt + t], 4)){
                            if(bad < 5) fprintf(stderr, "kvl %d qr %d nt %d nh %d head %d position %d: %.9g %.9g %.9g\n",
                                                kvl, qr, nt, nh, g, t, s, one[t], grp[(size_t)g * nt + t]);
                            bad++;
                        }
                    }
                }
                checked++;
            }
            free(one); free(grp);
        }
        free(L); free(R); free(qa); free(qr6);
    }
    /* k3_rowdot8_i8 against the scalar loop it replaces, spelled out */
    for(int I = 8; I <= 520; I += 128){
        W m; memset(&m, 0, sizeof m);
        m.fmt = 1; m.I = I; m.O = 16;
        m.q8 = malloc((size_t)16 * I); m.s = malloc(16 * sizeof(float));
        for(int i = 0; i < 16 * I; i++) m.q8[i] = (int8_t)((int)(rnd() * 42.0f));
        for(int o = 0; o < 16; o++) m.s[o] = fabsf(rnd()) * 0.01f + 1e-4f;
        float *x = malloc((size_t)I * sizeof(float)), y[8];
        for(int i = 0; i < I; i++) x[i] = rnd();
        for(int r0 = 0; r0 <= 8; r0 += 8){
            k3_rowdot8_i8(&m, r0, x, y);
            for(int j = 0; j < 8; j++){
                float a = 0;
                for(int i = 0; i < I; i++) a = a + prod(x[i], (float)m.q8[(size_t)(r0 + j) * I + i]);
                a = a * m.s[r0 + j];
                if(memcmp(&a, &y[j], 4)){
                    if(bad < 5) fprintf(stderr, "rowdot I %d row %d: %.9g %.9g\n", I, r0 + j, a, y[j]);
                    bad++;
                }
            }
            checked++;
        }
        free(m.q8); free(m.s); free(x);
    }
    printf("%s: %d combinations, %d values differ\n", bad ? "FAIL" : "ok", checked, bad);
    return bad ? 1 : 0;
#endif
}
