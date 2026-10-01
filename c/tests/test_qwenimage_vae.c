/* The Qwen-Image-2.1 VAE decoder (qwenimage_vae.h) against diffusers.
 *
 *   test_qwenimage_vae FIXTURE             FIXTURE/vae + FIXTURE/vae_ref (make_qwenimage_vae_tiny.py),
 *                                          each case twice: default bands, then one-row bands (--bands)
 *   test_qwenimage_vae --vae DIR --ref DIR [--tol T] [--bands] [--ppm PREFIX]
 *   test_qwenimage_vae --vae DIR --bench HxW [--ppm PREFIX]      random latent: time and peak RSS
 *   test_qwenimage_vae --vae DIR --latent STDIR NAME HxW [--expect OUT RGBA] [--ppm PREFIX]
 *                                          decode one [h*w][z] tensor, e.g. a pipeline's final latents,
 *                                          and compare with tensors OUT and RGBA of the same file
 *
 * A reference case is case{k}.z [h][w][z] (normalized latents), case{k}.out
 * [4][16h][16w] (vae.decode after the clamp) and case{k}.rgba [16h][16w][4]
 * (VaeImageProcessor.postprocess). Reported per case: max abs error of the
 * float output and how many uint8 values differ. The float output of two f32
 * implementations with different summation orders can land on opposite sides
 * of a rounding boundary, so a handful of off-by-one bytes is expected; an
 * off-by-two is not. */
#include "../qwenimage_vae.h"
#include <time.h>

static long rss_kb(const char *key){
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long v = -1; size_t n = strlen(key);
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, n)) { v = atol(line + n); break; }
    fclose(f);
    return v;
}

/* reset VmHWM so the next reading is the peak of what follows (Linux >= 4.0) */
static void reset_peak(void){
    FILE *f = fopen("/proc/self/clear_refs", "w");
    if (f) { fputs("5", f); fclose(f); }
}

static void write_ppm(const char *prefix, int k, const uint8_t *rgba, int H, int W){
    char path[1024]; snprintf(path, sizeof path, "%s%d.ppm", prefix, k);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (size_t p = 0; p < (size_t)H * W; p++) fwrite(rgba + p * 4, 1, 3, f);
    fclose(f);
    printf("  wrote %s\n", path);
}

static int threads(void){
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* decode z [h][w][Z]; prints time and peak; returns the buffers through out pointers */
static int run(QiVae *v, const float *z, int h, int w, uint8_t **rgba_o, float **f_o, double *secs){
    const size_t HW = (size_t)h * w * v->scale * v->scale;
    uint8_t *rgba = malloc(HW * 4);
    float *f = malloc(HW * 4 * sizeof(float));
    if (!rgba || !f) { fprintf(stderr, "OOM\n"); return -1; }
    long before = rss_kb("VmRSS:");
    reset_peak();
    double t0 = qiv_now();
    int rc = qiv_decode(v, z, h, w, rgba, f);
    *secs = qiv_now() - t0;
    long peak = rss_kb("VmHWM:");
    printf("  decode %dx%d -> %dx%d: %.2f s at %d threads, RSS before %.0f MB, peak during decode %.0f MB (+%.0f)\n",
           h, w, h * v->scale, w * v->scale, *secs, threads(), before / 1024.0, peak / 1024.0,
           (peak - before) / 1024.0);
    *rgba_o = rgba; *f_o = f;
    return rc;
}

/* decode z and compare with the reference float output ro and bytes rr; returns 1 if ok */
static int check(QiVae *v, const float *z, int h, int w, const float *ro, const uint8_t *rr,
                 double tol, const char *ppm, int k){
    const size_t HW = (size_t)h * w * v->scale * v->scale;
    uint8_t *rgba; float *f; double secs;
    if (run(v, z, h, w, &rgba, &f, &secs)) return 0;
    double maxerr = 0, sumerr = 0;
    for (size_t i = 0; i < HW * v->out_ch; i++) {
        double e = fabs((double)f[i] - ro[i]);
        sumerr += e; if (e > maxerr) maxerr = e;
    }
    size_t ndiff = 0; int maxd = 0;
    for (size_t i = 0; i < HW * 4; i++) {
        int d = abs((int)rgba[i] - (int)rr[i]);
        if (d) { ndiff++; if (d > maxd) maxd = d; }
    }
    int ok = maxerr <= tol && maxd <= 1 && ndiff * 1000 <= HW * 4;
    printf("  %s float max|err| %.3g (mean %.3g, tol %.1g); uint8: %zu of %zu differ, max diff %d\n",
           ok ? "ok  " : "FAIL", maxerr, sumerr / (HW * v->out_ch), tol, ndiff, HW * 4, maxd);
    if (ppm) write_ppm(ppm, k, rgba, h * v->scale, w * v->scale);
    free(rgba); free(f);
    return ok;
}

static float *read_f32(shards *S, const char *name, int64_t numel){
    st_tensor *t = st_find(S, name);
    if (!t || t->numel != numel) { fprintf(stderr, "%s: missing or not %lld elements\n", name, (long long)numel); return NULL; }
    float *p = malloc(sizeof(float) * numel);
    st_read_f32(S, name, p, 0);
    return p;
}

static uint8_t *read_u8(shards *S, const char *name, int64_t n){
    st_tensor *t = st_find(S, name);
    if (!t || t->nbytes != n) { fprintf(stderr, "%s: missing or not %lld bytes\n", name, (long long)n); return NULL; }
    uint8_t *p = malloc(n);
    st_read_raw(S, name, p, 0);
    return p;
}

int main(int argc, char **argv){
    const char *vae_dir = NULL, *ref_dir = NULL, *bench = NULL, *ppm = NULL;
    const char *lat_dir = NULL, *lat_name = NULL, *lat_hw = NULL, *exp_out = NULL, *exp_rgba = NULL;
    double tol = 1e-3;
    int bands = 0;
    char vbuf[4096], rbuf[4096];
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--vae") && i + 1 < argc) vae_dir = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref_dir = argv[++i];
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = argv[++i];
        else if (!strcmp(argv[i], "--ppm") && i + 1 < argc) ppm = argv[++i];
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--bands")) bands = 1;
        else if (!strcmp(argv[i], "--latent") && i + 3 < argc) { lat_dir = argv[++i]; lat_name = argv[++i]; lat_hw = argv[++i]; }
        else if (!strcmp(argv[i], "--expect") && i + 2 < argc) { exp_out = argv[++i]; exp_rgba = argv[++i]; }
        else if (argv[i][0] != '-' && !vae_dir) {
            snprintf(vbuf, sizeof vbuf, "%s/vae", argv[i]); snprintf(rbuf, sizeof rbuf, "%s/vae_ref", argv[i]);
            vae_dir = vbuf; ref_dir = rbuf; bands = 1;
        } else { fprintf(stderr, "usage: see the comment at the top of %s\n", __FILE__); return 2; }
    }
    if (!vae_dir || (!ref_dir && !bench && !lat_dir)) { fprintf(stderr, "usage: see the comment at the top of %s\n", __FILE__); return 2; }

    double t0 = qiv_now();
    QiVae *v = qiv_load(vae_dir);
    if (!v) return 1;
    printf("loaded %s in %.2f s: z_dim %d, %d up blocks, x%d, %.1f MB of weights, RSS %.0f MB\n",
           vae_dir, qiv_now() - t0, v->z_dim, v->nblk, v->scale, v->wbytes / 1e6, rss_kb("VmRSS:") / 1024.0);

    int fails = 0;
    if (bench || lat_dir) {
        int h = 0, w = 0;
        if (sscanf(bench ? bench : lat_hw, "%dx%d", &h, &w) != 2 || h <= 0 || w <= 0) { fprintf(stderr, "bad HxW\n"); return 2; }
        const int64_t nz = (int64_t)h * w * v->z_dim, HW = (int64_t)h * w * v->scale * v->scale;
        float *z = NULL, *ro = NULL;
        uint8_t *rr = NULL;
        shards L;
        if (bench) {
            z = malloc(sizeof(float) * nz);
            uint64_t s = 42;
            for (int64_t i = 0; i < nz; i += 2) {   /* Box-Muller: N(0, 1), the scale of the DiT's output */
                s = s * 6364136223846793005ull + 1442695040888963407ull; double u1 = ((s >> 11) + 1.0) / 9007199254740993.0;
                s = s * 6364136223846793005ull + 1442695040888963407ull; double u2 = (s >> 11) / 9007199254740992.0;
                double r = sqrt(-2 * log(u1));
                z[i] = (float)(r * cos(2 * M_PI * u2));
                if (i + 1 < nz) z[i + 1] = (float)(r * sin(2 * M_PI * u2));
            }
        } else {
            st_init(&L, lat_dir);
            if (!(z = read_f32(&L, lat_name, nz))) return 1;
            if (exp_out && (!(ro = read_f32(&L, exp_out, HW * v->out_ch)) || !(rr = read_u8(&L, exp_rgba, HW * 4)))) return 1;
            st_destroy(&L);
        }
        printf("latent %dx%d%s%s\n", h, w, lat_name ? " from " : " (random)", lat_name ? lat_name : "");
        if (ro) fails += !check(v, z, h, w, ro, rr, tol, ppm, 0);
        else {
            uint8_t *rgba; float *f; double secs;
            if (run(v, z, h, w, &rgba, &f, &secs)) return 1;
            if (ppm) write_ppm(ppm, 0, rgba, h * v->scale, w * v->scale);
            free(rgba); free(f);
        }
        free(z); free(ro); free(rr);
    } else {
        shards R; st_init(&R, ref_dir);
        for (int k = 0;; k++) {
            char nz[64], no[64], nr[64];
            snprintf(nz, sizeof nz, "case%d.z", k); snprintf(no, sizeof no, "case%d.out", k); snprintf(nr, sizeof nr, "case%d.rgba", k);
            st_tensor *tz = st_find(&R, nz);
            if (!tz) { if (k == 0) { fprintf(stderr, "no case0.z in %s\n", ref_dir); fails++; } break; }
            if (tz->rank != 3 || tz->shape[2] != v->z_dim) { fprintf(stderr, "%s: malformed case\n", nz); fails++; break; }
            int h = (int)tz->shape[0], w = (int)tz->shape[1];
            int64_t HW = (int64_t)h * w * v->scale * v->scale;
            float *z = read_f32(&R, nz, tz->numel), *ro = read_f32(&R, no, HW * v->out_ch);
            uint8_t *rr = read_u8(&R, nr, HW * 4);
            if (!z || !ro || !rr) { fails++; break; }
            for (int pass = 0; pass < (bands ? 2 : 1); pass++) {
                /* pass 1: one output row per band and 16-query attention blocks,
                 * so a tiny model crosses every band edge the real one does */
                qiv_col_floats = pass ? 1 : (size_t)16 << 20;
                qiv_attn_floats = pass ? 1 : (size_t)4 << 20;
                printf("case%d: latent %dx%d%s\n", k, h, w, pass ? ", one-row bands" : "");
                fails += !check(v, z, h, w, ro, rr, tol, pass ? NULL : ppm, k);
            }
            free(z); free(ro); free(rr);
        }
        st_destroy(&R);
    }
    qiv_free(v);
    printf(fails ? "FAILED\n" : "PASSED\n");
    return fails ? 1 : 0;
}
