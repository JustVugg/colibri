/* Qwen3.8 Q38_GROUP_OVERLAP: verify opt-in default=0, slot protection in
 * q38_expert_batch_reserve, and bit-identical grouped prefill / grouped MoE
 * outputs with Q38_GROUP_OVERLAP=1 vs 0 across cold, warm, and edge-cap runs. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../qwen38.c"
#include "../compat.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
    return 1; \
} } while (0)

enum {
    FIX_H = 32,
    FIX_I = 32,
    FIX_SI = 32,
    FIX_E = 12,
    FIX_K = 2,
    FIX_TENSORS = FIX_E * 6
};

typedef struct {
    char name[192];
    const char *dtype;
    int rows, cols;
    int64_t begin, end;
    unsigned char raw[FIX_H * FIX_I];
    size_t raw_bytes;
    float scale;
    int is_scale;
} FixTensor;

static int write_fp8_fixture(const char *directory) {
    FixTensor *tensors = (FixTensor *)calloc(FIX_TENSORS, sizeof(*tensors));
    if (!tensors) return -1;
    const char *proj[3] = {"gate_proj", "up_proj", "down_proj"};
    int count = 0;
    int64_t offset = 0;
    /* Official packing layout expected by q38_prepare_expert_scale_bank:
     * 1) contiguous gate+up scales, 2) contiguous down scales, 3) expert weights. */
    for (int e = 0; e < FIX_E; e++) {
        for (int p = 0; p < 2; p++) {
            FixTensor *sc = &tensors[count++];
            snprintf(sc->name, sizeof sc->name,
                     "model.layers.0.mlp.experts.%d.%s.weight_scale_inv", e, proj[p]);
            sc->dtype = "F32";
            sc->rows = 1;
            sc->cols = 1;
            sc->is_scale = 1;
            sc->scale = 0.03125f * (float)(1 + ((e * 3 + p) % 7));
            sc->begin = offset;
            offset += (int64_t)sizeof(float);
            sc->end = offset;
        }
    }
    for (int e = 0; e < FIX_E; e++) {
        FixTensor *sc = &tensors[count++];
        snprintf(sc->name, sizeof sc->name,
                 "model.layers.0.mlp.experts.%d.%s.weight_scale_inv", e, proj[2]);
        sc->dtype = "F32";
        sc->rows = 1;
        sc->cols = 1;
        sc->is_scale = 1;
        sc->scale = 0.03125f * (float)(1 + ((e * 3 + 2) % 7));
        sc->begin = offset;
        offset += (int64_t)sizeof(float);
        sc->end = offset;
    }
    for (int e = 0; e < FIX_E; e++) {
        for (int p = 0; p < 3; p++) {
            int rows = p == 2 ? FIX_H : FIX_I;
            int cols = p == 2 ? FIX_I : FIX_H;
            FixTensor *wt = &tensors[count++];
            snprintf(wt->name, sizeof wt->name,
                     "model.layers.0.mlp.experts.%d.%s.weight", e, proj[p]);
            wt->dtype = "F8_E4M3";
            wt->rows = rows;
            wt->cols = cols;
            wt->begin = offset;
            wt->raw_bytes = (size_t)rows * (size_t)cols;
            for (size_t i = 0; i < wt->raw_bytes; i++) {
                /* Deterministic valid E4M3 codes (sign bit + non-NaN exponent/mantissa) */
                unsigned sign = ((e * 13 + p * 7 + (int)i) & 1) ? 0x80u : 0x00u;
                unsigned mag = 0x20u + (unsigned)((e * 17 + p * 11 + (int)i * 5) % 0x38);
                wt->raw[i] = (unsigned char)(sign | mag);
            }
            offset += (int64_t)wt->raw_bytes;
            wt->end = offset;
        }
    }

    char *header = (char *)malloc(65536);
    if (!header) { free(tensors); return -1; }
    size_t used = 0;
    header[used++] = '{';
    for (int i = 0; i < count; i++) {
        FixTensor *t = &tensors[i];
        int n = snprintf(header + used, 65536 - used,
                         "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]}",
                         i ? "," : "", t->name, t->dtype, t->rows, t->cols,
                         (long long)t->begin, (long long)t->end);
        if (n < 0 || used + (size_t)n >= 65536 - 4) { free(header); free(tensors); return -1; }
        used += (size_t)n;
    }
    header[used++] = '}';
    header[used] = 0;

    char path[512];
    snprintf(path, sizeof path, "%s/model.safetensors", directory);
    FILE *f = fopen(path, "wb");
    if (!f) { free(header); free(tensors); return -1; }
    uint64_t hlen = (uint64_t)used;
    int failed = fwrite(&hlen, sizeof hlen, 1, f) != 1 ||
                 fwrite(header, 1, used, f) != used;
    for (int i = 0; !failed && i < count; i++) {
        FixTensor *t = &tensors[i];
        if (t->is_scale) failed = fwrite(&t->scale, sizeof t->scale, 1, f) != 1;
        else failed = fwrite(t->raw, 1, t->raw_bytes, f) != t->raw_bytes;
    }
    if (fclose(f)) failed = 1;
    free(header);
    free(tensors);
    return failed ? -1 : 0;
}

static int init_fixture_model(Model *m, const char *dir, int cap, int overlap) {
    memset(m, 0, sizeof(*m));
    m->c.hidden = FIX_H;
    m->c.inter = FIX_I;
    m->c.shared_inter = FIX_SI;
    m->c.experts = FIX_E;
    m->c.topk = FIX_K;
    m->c.layers = 1;
    m->c.norm_topk = 1;
    m->range_begin = 0;
    m->range_end = 1;
    m->native_fp8 = 1;
    m->native_bf16 = 1;
    m->expert_prefetch = 1;
    m->expert_parallel_reads = 1;
    m->prefill_batch = 1;
    m->group_overlap = overlap;
    snprintf(m->prefix, sizeof m->prefix, "model");
    st_init(&m->S, dir);
    m->cache = (LCache *)calloc(1, sizeof(*m->cache));
    if (!m->cache) { st_destroy(&m->S); return -1; }
    LCache *lc = &m->cache[0];
    lc->cap = cap;
    lc->slots = (Slot *)calloc((size_t)cap, sizeof(*lc->slots));
    lc->by_expert = (int *)malloc((size_t)FIX_E * sizeof(*lc->by_expert));
    if (!lc->slots || !lc->by_expert) {
        free(lc->slots); free(lc->by_expert); free(m->cache);
        st_destroy(&m->S); memset(m, 0, sizeof(*m)); return -1;
    }
    for (int e = 0; e < FIX_E; e++) lc->by_expert[e] = -1;

    Layer *l = (Layer *)calloc(1, sizeof(*l));
    if (!l) return -1;
    q38_weight_reserve(&l->router, Q38_WEIGHT_F32, FIX_E, FIX_H);
    q38_weight_reserve(&l->sh_g, Q38_WEIGHT_F32, FIX_SI, FIX_H);
    q38_weight_reserve(&l->sh_u, Q38_WEIGHT_F32, FIX_SI, FIX_H);
    q38_weight_reserve(&l->sh_d, Q38_WEIGHT_F32, FIX_H, FIX_SI);
    float *rg = (float *)l->router.data;
    float *sg = (float *)l->sh_g.data;
    float *su = (float *)l->sh_u.data;
    float *sd = (float *)l->sh_d.data;
    for (int i = 0; i < FIX_E * FIX_H; i++) rg[i] = 0.05f * (float)(((i * 7) % 11) - 5);
    for (int i = 0; i < FIX_SI * FIX_H; i++) {
        sg[i] = 0.03f * (float)(((i * 5) % 9) - 4);
        su[i] = 0.03f * (float)(((i * 3) % 9) - 4);
        sd[i] = 0.03f * (float)(((i * 11) % 9) - 4);
    }
    l->sh_gate = (float *)malloc((size_t)FIX_H * sizeof(float));
    if (!l->sh_gate) return -1;
    for (int i = 0; i < FIX_H; i++) l->sh_gate[i] = 0.04f * (float)((i % 7) - 3);
    m->L = l;
    return 0;
}

static void destroy_fixture_model(Model *m) {
    if (m->L) {
        q38_weight_free(&m->L[0].router);
        q38_weight_free(&m->L[0].sh_g);
        q38_weight_free(&m->L[0].sh_u);
        q38_weight_free(&m->L[0].sh_d);
        free(m->L[0].sh_gate);
        free(m->L);
    }
    if (m->cache) {
        for (int s = 0; s < m->cache[0].n; s++) {
            q38_weight_free(&m->cache[0].slots[s].gate);
            q38_weight_free(&m->cache[0].slots[s].up);
            q38_weight_free(&m->cache[0].slots[s].down);
            free(m->cache[0].slots[s].fp8_slab);
        }
        free(m->cache[0].slots);
        free(m->cache[0].by_expert);
    }
    if (m->expert_scales) free(m->expert_scales[0].values);
    free(m->expert_scales);
    free(m->cache);
    st_destroy(&m->S);
    memset(m, 0, sizeof(*m));
}

static void fill_inputs_and_logits(float *x, float *logits, int S) {
    for (int s = 0; s < S; s++) {
        for (int d = 0; d < FIX_H; d++)
            x[(size_t)s * FIX_H + (size_t)d] = 0.1f * (float)(((s * 13 + d * 5) % 15) - 7);
        for (int e = 0; e < FIX_E; e++)
            logits[(size_t)s * FIX_E + (size_t)e] = -4.0f;
        /* Route every row to two distinct experts, covering all FIX_E experts across S >= 6 rows */
        int e0 = (s * 2) % FIX_E;
        int e1 = (s * 2 + 1) % FIX_E;
        logits[(size_t)s * FIX_E + (size_t)e0] = 3.0f + 0.1f * (float)(s % 5);
        logits[(size_t)s * FIX_E + (size_t)e1] = 2.0f + 0.1f * (float)(s % 3);
    }
}

static int check_finite_nonzero(const float *buf, int n) {
    int nz = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(buf[i])) return 0;
        if (buf[i] != 0.0f) nz = 1;
    }
    return nz;
}

static int test_default_off(void) {
    unsetenv("Q38_GROUP_OVERLAP");
    CHECK(q38_env_bool("Q38_GROUP_OVERLAP", 0) == 0);
    setenv("Q38_GROUP_OVERLAP", "1", 1);
    CHECK(q38_env_bool("Q38_GROUP_OVERLAP", 0) == 1);
    setenv("Q38_GROUP_OVERLAP", "0", 1);
    CHECK(q38_env_bool("Q38_GROUP_OVERLAP", 0) == 0);
    unsetenv("Q38_GROUP_OVERLAP");
    return 0;
}

static int test_batch_reserve_keep(const char *dir) {
    Model m;
    CHECK(init_fixture_model(&m, dir, 6, 1) == 0);

    int wave1[3] = {0, 1, 2};
    Slot *s1[3] = {0};
    CHECK(q38_expert_get_batch(&m, 0, wave1, 3, s1) == 1);

    int wave2[3] = {3, 4, 5};
    Slot *s2[3] = {0};
    Q38ExpertLoadJob *jobs = NULL;
    int jn = 0;
    CHECK(q38_expert_batch_reserve(&m, 0, wave2, 3, s2, s1, 3, &jobs, &jn) == 1);
    CHECK(jn == 3 && jobs != NULL);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            CHECK(s1[i] != s2[j]);

    CHECK(q38_loader_start(&m, 0, jobs, jn, 0) == 1);
    q38_loader_join(&m);
    q38_expert_batch_publish(&m, 0, jobs, jn);
    free(jobs);

    for (int i = 0; i < 3; i++) {
        CHECK(m.cache[0].by_expert[wave1[i]] == (int)(s1[i] - m.cache[0].slots));
        CHECK(s1[i]->eid == wave1[i]);
        CHECK(m.cache[0].by_expert[wave2[i]] == (int)(s2[i] - m.cache[0].slots));
        CHECK(s2[i]->eid == wave2[i]);
    }

    /* count + keep_n > cache->cap must fall back cleanly */
    int too_big[4] = {6, 7, 8, 9};
    Slot *sb[4] = {0};
    jobs = NULL;
    jn = 0;
    CHECK(q38_expert_batch_reserve(&m, 0, too_big, 4, sb, s1, 3, &jobs, &jn) == 0);
    CHECK(jobs == NULL && jn == 0);

    destroy_fixture_model(&m);
    return 0;
}

static int test_prefill_parity_cap(const char *dir, int cap, int expect_overlap) {
    const int S = 16;
    float *x = (float *)calloc((size_t)S * FIX_H, sizeof(float));
    float *logits = (float *)calloc((size_t)S * FIX_E, sizeof(float));
    float *out0 = (float *)calloc((size_t)S * FIX_H, sizeof(float));
    float *out1 = (float *)calloc((size_t)S * FIX_H, sizeof(float));
    float *out0_warm = (float *)calloc((size_t)S * FIX_H, sizeof(float));
    float *out1_warm = (float *)calloc((size_t)S * FIX_H, sizeof(float));
    CHECK(x && logits && out0 && out1 && out0_warm && out1_warm);
    fill_inputs_and_logits(x, logits, S);

    Model m0, m1;
    CHECK(init_fixture_model(&m0, dir, cap, 0) == 0);
    CHECK(init_fixture_model(&m1, dir, cap, 1) == 0);

    /* Cold pass */
    q38_moe_prefill(&m0, &m0.L[0], 0, x, S, out0, logits, 0);
    q38_moe_prefill(&m1, &m1.L[0], 0, x, S, out1, logits, 0);
    CHECK(check_finite_nonzero(out0, S * FIX_H));
    CHECK(m0.group_overlap_waves == 0);
    if (expect_overlap) CHECK(m1.group_overlap_waves > 0);
    else CHECK(m1.group_overlap_waves == 0);
    CHECK(memcmp(out0, out1, (size_t)S * FIX_H * sizeof(float)) == 0);

    /* Second pass with partially warm cache */
    q38_moe_prefill(&m0, &m0.L[0], 0, x, S, out0_warm, logits, 0);
    q38_moe_prefill(&m1, &m1.L[0], 0, x, S, out1_warm, logits, 0);
    CHECK(memcmp(out0, out0_warm, (size_t)S * FIX_H * sizeof(float)) == 0);
    CHECK(memcmp(out0_warm, out1_warm, (size_t)S * FIX_H * sizeof(float)) == 0);

    destroy_fixture_model(&m0);
    destroy_fixture_model(&m1);
    free(x); free(logits); free(out0); free(out1); free(out0_warm); free(out1_warm);
    return 0;
}

static int test_moe_ex_parity(const char *dir) {
    const int S = 16;
    float *x = (float *)calloc((size_t)S * FIX_H, sizeof(float));
    float *logits = (float *)calloc((size_t)S * FIX_E, sizeof(float));
    float *out0 = (float *)calloc((size_t)S * FIX_H, sizeof(float));
    float *out1 = (float *)calloc((size_t)S * FIX_H, sizeof(float));
    CHECK(x && logits && out0 && out1);
    fill_inputs_and_logits(x, logits, S);

    Model m0, m1;
    CHECK(init_fixture_model(&m0, dir, 6, 0) == 0);
    CHECK(init_fixture_model(&m1, dir, 6, 1) == 0);

    q38_moe_ex(&m0, &m0.L[0], 0, x, S, out0, logits, 0);
    q38_moe_ex(&m1, &m1.L[0], 0, x, S, out1, logits, 0);
    CHECK(check_finite_nonzero(out0, S * FIX_H));
    CHECK(m0.group_overlap_waves == 0);
    CHECK(m1.group_overlap_waves > 0);
    CHECK(memcmp(out0, out1, (size_t)S * FIX_H * sizeof(float)) == 0);

    destroy_fixture_model(&m0);
    destroy_fixture_model(&m1);
    free(x); free(logits); free(out0); free(out1);
    return 0;
}

int main(void) {
    if (test_default_off()) return 1;

    char dir[256];
    snprintf(dir, sizeof dir, "q38_group_overlap_fixture_%d", (int)getpid());
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0700);
#endif
    if (write_fp8_fixture(dir)) {
        fprintf(stderr, "failed to write FP8 fixture in %s\n", dir);
        return 1;
    }

    int rc = 0;
    if (!rc) rc = test_batch_reserve_keep(dir);
    if (!rc) rc = test_prefill_parity_cap(dir, 6, 1);
    if (!rc) rc = test_prefill_parity_cap(dir, 4, 1);
    if (!rc) rc = test_prefill_parity_cap(dir, 3, 0);
    if (!rc) rc = test_moe_ex_parity(dir);

    char st_path[320];
    snprintf(st_path, sizeof st_path, "%s/model.safetensors", dir);
    remove(st_path);
    rmdir(dir);

    if (!rc) printf("test_qwen38_group_overlap: PASS\n");
    return rc;
}
