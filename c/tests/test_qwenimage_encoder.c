/* Compare the C encoder with Diffusers AutoencoderKLQwenImage21.
 * tools/make_qwenimage_encoder_tiny.py FIXTURE; test_qwenimage_encoder FIXTURE
 * or: test_qwenimage_encoder --vae VAE_DIR --ref REFERENCE_DIR */
#include "../qwenimage_encoder.h"

/* Run these dependency-free checks through make check. Pass a fixture to
 * also compare the full encoder with the pinned Diffusers reference. */
static int unit_tests(void) {
    int fails = 0;
    float x[4 * 6 * 2], w[3 * 9 * 2], bias[3] = {.5f, -.5f, 1.f}, y[2 * 3 * 3];
    for (int i = 0; i < 48; i++) x[i] = (float)(i - 17) / 8;
    for (int i = 0; i < 54; i++) w[i] = (float)(i % 11 - 5) / 16;
    QivConv conv = {.cin = 2, .cout = 3, .k = 3, .w = w, .b = bias, .m = {.fmt = QI_F32, .N = 3, .K = 18, .w = w}};
    for (int bands = 0; bands < 2; bands++) {
        qiv_col_floats = bands ? 1 : (size_t)16 << 20;
        fails += qie_downsample(&conv, x, 4, 6, y) != 0;
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 3; c++)
                for (int o = 0; o < 3; o++) {
                    float expected = bias[o];
                    /* Independent direct convolution, no im2col; checks the asymmetric
                     * bottom/right pad as well as channel and kernel-axis ordering. */
                    for (int iy = 2 * r; iy < 2 * r + 3 && iy < 4; iy++)
                        for (int ix = 2 * c; ix < 2 * c + 3 && ix < 6; ix++)
                            for (int i = 0; i < 2; i++)
                                expected +=
                                    x[(iy * 6 + ix) * 2 + i] * w[((o * 3 + ix - 2 * c) * 3 + iy - 2 * r) * 2 + i];
                    fails += fabsf(y[(r * 3 + c) * 3 + o] - expected) > 1e-6f;
                }
    }
    /* Channel expansion makes a temporal padding bug visible: one output
     * group contains only zeros, the other the current frame's four pixels. */
    const float square[] = {1, 3, 5, 7};
    float avg[2] = {10, 20};
    QieBlock block = {.cin = 1, .cout = 2, .down = 1, .ft = 2};
    qie_avg_add(&block, square, 2, 2, avg);
    fails += avg[0] != 10 || avg[1] != 24;
    block.cout = 1;
    avg[0] = 10;
    qie_avg_add(&block, square, 2, 2, avg);
    fails += avg[0] != 12; /* zero frame counts in the divisor of eight */
    block.ft = 1;
    avg[0] = 10;
    qie_avg_add(&block, square, 2, 2, avg);
    fails += avg[0] != 14;
    block.down = 0;
    block.cin = 2;
    block.cout = 1;
    avg[0] = 10;
    qie_avg_add(&block, square, 1, 1, avg);
    fails += avg[0] != 12;
    QiVaeEncoder v = {.in_ch = 4, .scale = 16, .conv_in = {.cout = 8}};
    float sentinel = 42;
    uint8_t pixel = 0;
    fails += qie_encode(NULL, &pixel, 16, 16, 4, &sentinel) != -1;
    fails += qie_encode(&v, NULL, 16, 16, 4, &sentinel) != -1;
    fails += qie_encode(&v, &pixel, 0, 16, 4, &sentinel) != -1;
    fails += qie_encode(&v, &pixel, 16, 15, 4, &sentinel) != -1;
    fails += qie_encode(&v, &pixel, 65536, 65536, 4, &sentinel) != -1;
    fails += qie_encode(&v, &pixel, 16, 16, 2, &sentinel) != -1;
    fails += qie_encode_f32(&v, &sentinel, -16, 16, &sentinel) != -1;
    fails += sentinel != 42;
    qiv_col_floats = (size_t)16 << 20;
    printf("encoder unit tests: %d failures\n", fails);
    return fails ? 1 : 0;
}

static int compare(const float *actual, const float *expected, size_t n, const char *name) {
    double worst = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) return 1;
        double err = fabs((double)actual[i] - expected[i]);
        if (err > worst) worst = err;
    }
    printf("%s: max abs error %.9g\n", name, worst);
    return worst > 1e-4;
}

static void *read_tensor(shards *s, const char *name, int bytes, int64_t n) {
    st_tensor *t = st_find(s, name);
    if (!t || t->numel != n || t->nbytes != n * bytes) return NULL;
    void *p = malloc((size_t)n * bytes);
    if (p) st_read_raw(s, name, p, 0);
    return p;
}

int main(int argc, char **argv) {
    if (argc == 1) return unit_tests();
    if (argc == 3 && !strcmp(argv[1], "--reject")) {
        QiVaeEncoder *v = qie_load(argv[2]);
        if (!v) return 0;
        qie_free(v);
        fprintf(stderr, "invalid checkpoint accepted\n");
        return 1;
    }
    char vae_path[4096], ref_path[4096];
    const char *vp, *rp;
    if (argc == 2) {
        snprintf(vae_path, sizeof vae_path, "%s/vae", argv[1]);
        snprintf(ref_path, sizeof ref_path, "%s/encoder_ref", argv[1]);
        vp = vae_path;
        rp = ref_path;
    } else if (argc == 5 && !strcmp(argv[1], "--vae") && !strcmp(argv[3], "--ref")) {
        vp = argv[2];
        rp = argv[4];
    } else {
        fprintf(stderr, "usage: %s FIXTURE | --vae DIR --ref DIR\n", argv[0]);
        return 2;
    }
    QiVaeEncoder *v = qie_load(vp);
    if (!v) return 1;
    shards s;
    st_init(&s, rp);
    int fails = 0, cases = 0;
    for (int k = 0;; k++) {
        char name[80];
        snprintf(name, sizeof name, "case%d.pixels", k);
        st_tensor *t = st_find(&s, name);
        if (!t) break;
        if (t->rank != 3) {
            fails++;
            break;
        }
        int H = (int)t->shape[0], W = (int)t->shape[1], channels = (int)t->shape[2];
        size_t n = (size_t)(H / v->scale) * (W / v->scale) * v->z_dim;
        uint8_t *pixels = read_tensor(&s, name, 1, (int64_t)H * W * channels);
        snprintf(name, sizeof name, "case%d.input", k);
        float *input = read_tensor(&s, name, 4, (int64_t)H * W * v->in_ch);
        snprintf(name, sizeof name, "case%d.z", k);
        float *expected = read_tensor(&s, name, 4, (int64_t)n);
        snprintf(name, sizeof name, "case%d.mode", k);
        float *mode = read_tensor(&s, name, 4, (int64_t)n);
        float *z = malloc(n * sizeof(float));
        if (!pixels || !input || !expected || !mode || !z) {
            fails++;
            goto done;
        }
        for (int bands = 0; bands < 2; bands++) {
            qiv_col_floats = bands ? 1 : (size_t)16 << 20;
            qiv_attn_floats = bands ? 1 : (size_t)4 << 20;
            printf("case%d: %dx%d, %d channels, bands=%d\n", k, H, W, channels, bands);
            if (qie_encode(v, pixels, H, W, channels, z)) {
                fails++;
                continue;
            }
            fails += compare(z, expected, n, "bytes -> normalized mode");
            if (qie_encode_f32(v, input, H, W, z)) {
                fails++;
                continue;
            }
            fails += compare(z, expected, n, "normalized pixels -> normalized mode");
            for (size_t i = 0; i < n; i++) z[i] = z[i] * v->std[i % v->z_dim] + v->mean[i % v->z_dim];
            fails += compare(z, mode, n, "posterior mode");
        }
        z[0] = 42;
        fails += qie_encode(v, pixels, 0, W, channels, z) != -1;
        fails += qie_encode(v, pixels, H - 1, W, channels, z) != -1;
        fails += qie_encode(v, pixels, H, W - 1, channels, z) != -1;
        fails += qie_encode(v, pixels, H, W, 2, z) != -1;
        fails += qie_encode(v, NULL, H, W, channels, z) != -1;
        fails += qie_encode(v, pixels, H, W, channels, NULL) != -1;
        fails += z[0] != 42;
        cases++;
    done:
        free(pixels);
        free(input);
        free(expected);
        free(mode);
        free(z);
    }
    st_destroy(&s);
    qie_free(v);
    if (!cases) fails++;
    printf("%s: %d cases, %d failures\n", fails ? "FAILED" : "PASSED", cases, fails);
    return fails ? 1 : 0;
}
