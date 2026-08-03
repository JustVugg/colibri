/* Loader-seam test for fmt=5: writes a real .safetensors file containing an
 * int3-g64 tensor (U8 payload + per-GROUP .qs) next to an int4 control tensor
 * (per-row .qs), indexes it with st_init, loads both through qt_from_disk, and
 * checks the byte-count/.qs-size format inference picks fmt=5 vs fmt=2 correctly
 * and the loaded weights dequantize identically to pack_int3_g64's output. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                                        \
            fails++;                                                                                                   \
        }                                                                                                              \
    } while (0)

static uint64_t rng = 0xA5A5A5A55A5A5A5Aull;
static float rndf(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return ((int64_t)(rng & 0xFFFFF) - 0x80000) / (float)0x80000;
}

static void deq4(const QT *t, float *dq) {
    for (int o = 0; o < t->O; o++)
        for (int i = 0; i < t->I; i++) {
            if (t->fmt == 5) {
                int64_t g = i / I3_GROUP;
                const uint8_t *lo = t->q4 + (int64_t)o * i3_rowbytes(t->I) + g * I3_GBYTES, *hi = lo + 16;
                int k = i % I3_GROUP;
                unsigned u = ((lo[k >> 2] >> ((k & 3) * 2)) & 3) | (((hi[k >> 3] >> (k & 7)) & 1) << 2);
                dq[(int64_t)o * t->I + i] = (float)((int)u - 4) * t->s[(int64_t)o * i3_groups(t->I) + g];
            } else { /* fmt2 */
                uint8_t b = t->q4[(int64_t)o * ((t->I + 1) / 2) + (i >> 1)];
                int v = (i & 1) ? ((int)(b >> 4) - 8) : ((int)(b & 0xF) - 8);
                dq[(int64_t)o * t->I + i] = (float)v * t->s[o];
            }
        }
}

int main(void) {
    enum {
        O = 5,
        I = 320
    }; /* 5 groups per row; I > 256 so the per-row int4
        * control stays fmt=2 (detect_group_size probes
        * gs up to 256: any smaller I would make O row
        * scales match a legitimate 1-group layout) */
    int64_t ng = i3_groups(I), rb = i3_rowbytes(I);
    static float w[O * I];
    for (int i = 0; i < O * I; i++) w[i] = rndf() * 0.05f;
    w[7] = 1.9f;

    static uint8_t q3[O * (I / 64) * 24];
    static float s3[O * (I / 64)];
    pack_int3_g64(w, q3, s3, O, I);
    static uint8_t q4b[O * ((I + 1) / 2)];
    static float s4[O];
    pack_int4(w, q4b, s4, O, I, 4);

    /* write a minimal single-shard safetensors file */
    const char *dir = "tests/tmp_int3_snap";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir, 0755);
#endif
    char path[256];
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    int64_t nb3 = (int64_t)O * rb, ns3 = (int64_t)O * ng * 4, nb4 = (int64_t)O * ((I + 1) / 2), ns4 = (int64_t)O * 4;
    char hdr[1024];
    int hl = snprintf(hdr, sizeof hdr,
                      "{\"w3\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[0,%lld]},"
                      "\"w3.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
                      "\"w4\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
                      "\"w4.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}}",
                      (long long)nb3, (long long)nb3, (long long)(O * ng), (long long)nb3, (long long)(nb3 + ns3),
                      (long long)nb4, (long long)(nb3 + ns3), (long long)(nb3 + ns3 + nb4), (long long)O,
                      (long long)(nb3 + ns3 + nb4), (long long)(nb3 + ns3 + nb4 + ns4));
    FILE *f = fopen(path, "wb");
    if (!f) {
        printf("FAIL: cannot create %s (run from c/, like tools/run_tests.py does)\n", path);
        return 1;
    }
    uint64_t hlen = (uint64_t)hl;
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, hl, f);
    fwrite(q3, 1, (size_t)nb3, f);
    fwrite(s3, 1, (size_t)ns3, f);
    fwrite(q4b, 1, (size_t)nb4, f);
    fwrite(s4, 1, (size_t)ns4, f);
    fclose(f);

    static Model gm; /* only gm.S is used by qt_from_disk */
    st_init(&gm.S, dir);

    QT t3;
    memset(&t3, 0, sizeof t3);
    qt_from_disk(&gm, "w3", O, I, 8, 0, &t3);
    CHECK(t3.fmt == 5);
    static float dq_load[O * I], dq_ref[O * I];
    deq4(&t3, dq_load);
    QT tr = {.fmt = 5, .q4 = q3, .s = s3, .O = O, .I = I};
    deq4(&tr, dq_ref);
    CHECK(memcmp(dq_load, dq_ref, sizeof dq_ref) == 0);

    QT t4;
    memset(&t4, 0, sizeof t4);
    qt_from_disk(&gm, "w4", O, I, 8, 0, &t4);
    CHECK(t4.fmt == 2); /* control: row-scale int4 still detected */
    deq4(&t4, dq_load);
    QT tr4 = {.fmt = 2, .q4 = q4b, .s = s4, .O = O, .I = I};
    deq4(&tr4, dq_ref);
    CHECK(memcmp(dq_load, dq_ref, sizeof dq_ref) == 0);

    unlink(path);
    rmdir(dir);
    if (fails) {
        printf("int3 loader tests: %d FAILED\n", fails);
        return 1;
    }
    printf("int3 loader tests: ok\n");
    return 0;
}
