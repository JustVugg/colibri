#define GLM53_NO_MAIN
#include "../glm53.c"

#include <inttypes.h>

#define QUALITY_GENERATED 4
#define QUALITY_PROMPTS 3

typedef struct {
    double rel_l2, cosine;
    int finite;
} Metric;

typedef struct {
    char *text;
    int *ids;
    int count;
    float *logits;
    double cpu_nll, gpu_nll;
    int generated[QUALITY_GENERATED];
    int gpu_generated[QUALITY_GENERATED];
    int generated_count;
    uint64_t logits_hash, kda_hash, mla_hash;
    float *site[5];
    float *kda_matrix, *kda_window;
    float *mla_latent, *mla_keys, *mla_gates;
    Metric logits_metric, site_metric[5], kda_matrix_metric;
    Metric kda_window_metric, mla_latent_metric, mla_keys_metric;
    Metric mla_gates_metric;
    int gpu_deterministic, generated_agree;
} QualityRun;

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static Metric compare_values(const float *got, const float *want, size_t n) {
    Metric m = {0.0, 0.0, 1};
    double diff2 = 0.0, got2 = 0.0, want2 = 0.0, dot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(got[i]) || !isfinite(want[i])) m.finite = 0;
        double g = got[i], w = want[i], d = g - w;
        diff2 += d * d;
        got2 += g * g;
        want2 += w * w;
        dot += g * w;
    }
    m.rel_l2 = sqrt(diff2 / (want2 + 1e-30));
    m.cosine = dot / sqrt((got2 + 1e-30) * (want2 + 1e-30));
    return m;
}

static double sequence_nll(
    const float *logits, const int *ids, int count, int vocab) {
    double total = 0.0;
    for (int t = 0; t + 1 < count; ++t) {
        const float *row = logits + (size_t)t * vocab;
        float maximum = row[0];
        for (int v = 1; v < vocab; ++v)
            if (row[v] > maximum) maximum = row[v];
        double sum = 0.0;
        for (int v = 0; v < vocab; ++v)
            sum += exp((double)row[v] - maximum);
        total += log(sum) + maximum - row[ids[t + 1]];
    }
    return count > 1 ? total / (count - 1) : 0.0;
}

static int representative_layers(
    const GModel *m, int *kda, int *mla, int *dense, int *moe,
    int *kda_index, int *mla_index) {
    *kda = *mla = *dense = *moe = *kda_index = *mla_index = -1;
    int ikda = 0, imla = 0;
    for (int i = 0; i < m->c.n_layers; ++i) {
        if (m->c.is_full[i]) {
            if (*mla < 0) { *mla = i; *mla_index = imla; }
            imla++;
        } else {
            if (*kda < 0) { *kda = i; *kda_index = ikda; }
            ikda++;
        }
        if (i < m->c.first_dense && *dense < 0) *dense = i;
        if (i >= m->c.first_dense && *moe < 0) *moe = i;
    }
    return *kda >= 0 && *mla >= 0 && *dense >= 0 && *moe >= 0;
}

static int capture_states(
    GModel *m, GSession *s, QualityRun *run, int keep_values,
    int kda_layer, int mla_layer, int kda_index, int mla_index) {
    const Cfg *c = &m->c;
    const size_t km = (size_t)c->kda_heads * c->kda_hd * c->kda_hd;
    const size_t kw = (size_t)3 * c->kda_proj * c->conv_k;
    const size_t ml = (size_t)run->count * c->kv_lora;
    const size_t mi = (size_t)run->count * c->index_hd;
    float *matrix = (float *)malloc(km * sizeof(float));
    float *window = (float *)malloc(kw * sizeof(float));
    float *latent = (float *)malloc(ml * sizeof(float));
    float *keys = (float *)malloc(mi * sizeof(float));
    float *gates = (float *)malloc(mi * sizeof(float));
    if (!matrix || !window || !latent || !keys || !gates) return 0;
    uint64_t kh = UINT64_C(1469598103934665603);
    uint64_t mh = UINT64_C(1469598103934665603);
#ifdef COLI_CUDA
    if (s->gpu) {
        for (int i = 0, ki = 0, mi_idx = 0; i < c->n_layers; ++i) {
            if (c->is_full[i]) {
                const size_t lf = (size_t)run->count * c->kv_lora;
                const size_t ix = (size_t)run->count * c->index_hd;
                float *l = i == mla_layer ? latent :
                    (float *)malloc(lf * sizeof(float));
                float *k = i == mla_layer ? keys :
                    (float *)malloc(ix * sizeof(float));
                float *g = i == mla_layer ? gates :
                    (float *)malloc(ix * sizeof(float));
                if (!l || !k || !g ||
                    !coli_glm53_gpu_session_mla_state_download(
                        s->gpu, mi_idx, l, lf, k, ix, g, ix))
                    return 0;
                mh = hash_bytes(mh, l, lf * sizeof(float));
                mh = hash_bytes(mh, k, ix * sizeof(float));
                mh = hash_bytes(mh, g, ix * sizeof(float));
                if (i != mla_layer) { free(l); free(k); free(g); }
                mi_idx++;
            } else {
                float *a = i == kda_layer ? matrix :
                    (float *)malloc(km * sizeof(float));
                float *w = i == kda_layer ? window :
                    (float *)malloc(kw * sizeof(float));
                if (!a || !w ||
                    !coli_glm53_gpu_session_kda_state_download(
                        s->gpu, ki, a, km, w, kw))
                    return 0;
                kh = hash_bytes(kh, a, km * sizeof(float));
                kh = hash_bytes(kh, w, kw * sizeof(float));
                if (i != kda_layer) { free(a); free(w); }
                ki++;
            }
        }
    } else
#endif
    {
        for (int i = 0; i < c->n_layers; ++i) {
            GLayerState *st = &s->layer[i];
            if (c->is_full[i]) {
                size_t lf = (size_t)run->count * c->kv_lora;
                size_t ix = (size_t)run->count * c->index_hd;
                mh = hash_bytes(mh, st->latent, lf * sizeof(float));
                mh = hash_bytes(mh, st->ikeys, ix * sizeof(float));
                mh = hash_bytes(mh, st->igates, ix * sizeof(float));
                if (i == mla_layer) {
                    memcpy(latent, st->latent, lf * sizeof(float));
                    memcpy(keys, st->ikeys, ix * sizeof(float));
                    memcpy(gates, st->igates, ix * sizeof(float));
                }
            } else {
                kh = hash_bytes(kh, st->kda_state, km * sizeof(float));
                kh = hash_bytes(kh, st->kda_window, kw * sizeof(float));
                if (i == kda_layer) {
                    memcpy(matrix, st->kda_state, km * sizeof(float));
                    memcpy(window, st->kda_window, kw * sizeof(float));
                }
            }
        }
    }
    (void)kda_index; (void)mla_index;
    run->kda_hash = kh;
    run->mla_hash = mh;
    if (keep_values) {
        run->kda_matrix = matrix; run->kda_window = window;
        run->mla_latent = latent; run->mla_keys = keys;
        run->mla_gates = gates;
    } else {
        free(matrix); free(window); free(latent); free(keys); free(gates);
    }
    return 1;
}

static int execute_run(
    GModel *m, QualityRun *run, int keep_logits, int keep_states,
    int kda_layer, int mla_layer, int dense_layer, int moe_layer,
    int kda_index, int mla_index) {
    const int d = m->c.hidden;
    Glm53QualityCapture cpu = {
        kda_layer, mla_layer, dense_layer, moe_layer,
        run->site[0], run->site[1], run->site[2], run->site[3], run->site[4]
    };
    GSession *session = session_open(m, run->count + QUALITY_GENERATED + 1);
#ifdef COLI_CUDA
    if (session->gpu) {
        ColiGlm53GpuQualityCapture gpu = {
            kda_layer, mla_layer, dense_layer, moe_layer,
            run->site[0], run->site[1], run->site[2], run->site[3],
            run->site[4], (size_t)d
        };
        if (!coli_glm53_gpu_session_set_quality_capture(session->gpu, &gpu))
            return 0;
    }
#endif
    g_glm53_quality_capture = session->layer ? &cpu : NULL;
    if (!glm53_request_begin()) return 0;
    float *logits = glm53_request_dispatch(
        m, &session, run->count + QUALITY_GENERATED + 1,
        run->ids, run->count, NULL, 0, 1, 1);
    if (!logits) return 0;
    run->gpu_nll =
        sequence_nll(logits, run->ids, run->count, m->c.vocab);
    run->logits_hash = hash_bytes(
        UINT64_C(1469598103934665603), logits,
        (size_t)run->count * m->c.vocab * sizeof(float));
    if (!capture_states(
            m, session, run, keep_states, kda_layer, mla_layer,
            kda_index, mla_index))
        return 0;
#ifdef COLI_CUDA
    if (session->gpu)
        (void)coli_glm53_gpu_session_set_quality_capture(session->gpu, NULL);
#endif
    g_glm53_quality_capture = NULL;
    if (keep_logits) {
        size_t bytes =
            (size_t)run->count * m->c.vocab * sizeof(float);
        run->logits = (float *)malloc(bytes);
        if (!run->logits) return 0;
        memcpy(run->logits, logits, bytes);
    }
    int rows = run->count;
    for (int step = 0; step < QUALITY_GENERATED; ++step) {
        int next = argmax(
            logits + (size_t)(rows - 1) * m->c.vocab, m->c.vocab);
        run->generated[run->generated_count++] = next;
        free(logits);
        logits = glm53_request_dispatch(
            m, &session, run->count + QUALITY_GENERATED + 1,
            &next, 1, NULL, 0, 0, 0);
        if (!logits) return 0;
        run->logits_hash = hash_bytes(
            run->logits_hash, logits,
            (size_t)m->c.vocab * sizeof(float));
        rows = 1;
    }
    free(logits);
    glm53_request_end();
    session_close(m, session);
    g_glm53_quality_capture = NULL;
    return 1;
}

static void print_metric(FILE *f, const char *name, Metric m) {
    fprintf(f,
        "\"%s\":{\"relative_l2\":%.12g,\"cosine\":%.12g,\"finite\":%s}",
        name, m.rel_l2, m.cosine, m.finite ? "true" : "false");
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL OUTPUT.json\n", argv[0]);
        return 2;
    }
    static const char *texts[QUALITY_PROMPTS] = {
        "The key insight about mixture of experts is",
        "Write a Python function to implement binary search",
        "Solve x squared minus 5x plus 6 equals zero"
    };
    const char *model_path = argv[1], *output_path = argv[2];
    char tokenizer_path[1024];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", model_path);
    Tok tokenizer;
    tok_load(&tokenizer, tokenizer_path);
    QualityRun cpu[QUALITY_PROMPTS] = {0};
    for (int p = 0; p < QUALITY_PROMPTS; ++p) {
        cpu[p].text = (char *)texts[p];
        cpu[p].ids = (int *)malloc(256 * sizeof(int));
        cpu[p].count = tok_encode(
            &tokenizer, texts[p], (int)strlen(texts[p]), cpu[p].ids, 256);
        for (int s = 0; s < 5; ++s)
            cpu[p].site[s] = (float *)malloc(4096 * sizeof(float));
    }

    setenv("GLM53_BACKEND", "cpu", 1);
    GModel cm = {0};
    model_load(&cm, model_path);
    int kl, ml, dl, ol, ki, mi;
    if (!representative_layers(&cm, &kl, &ml, &dl, &ol, &ki, &mi))
        return 1;
    for (int p = 0; p < QUALITY_PROMPTS; ++p) {
        if (!execute_run(
                &cm, &cpu[p], 1, 1, kl, ml, dl, ol, ki, mi))
            return 1;
        cpu[p].cpu_nll = cpu[p].gpu_nll;
    }
    model_release(&cm);

    setenv("GLM53_BACKEND", "gpu", 1);
    GModel gm = {0};
    model_load(&gm, model_path);
    ColiGpuTelemetry telemetry_before = {0}, telemetry_after = {0};
    coli_gpu_context_telemetry(
        coli_glm53_gpu_backend_context(&g_glm53_gpu_backend),
        &telemetry_before);
    int gates = 1;
    double cpu_mean = 0.0, gpu_mean = 0.0;
    for (int p = 0; p < QUALITY_PROMPTS; ++p) {
        QualityRun first = {0}, second = {0};
        first.text = second.text = cpu[p].text;
        first.ids = second.ids = cpu[p].ids;
        first.count = second.count = cpu[p].count;
        for (int s = 0; s < 5; ++s) {
            first.site[s] = (float *)malloc((size_t)gm.c.hidden * sizeof(float));
            second.site[s] = (float *)malloc((size_t)gm.c.hidden * sizeof(float));
        }
        if (!execute_run(&gm, &first, 1, 1, kl, ml, dl, ol, ki, mi) ||
            !execute_run(&gm, &second, 1, 0, kl, ml, dl, ol, ki, mi))
            return 1;
        cpu[p].logits_metric = compare_values(
            first.logits, cpu[p].logits,
            (size_t)cpu[p].count * gm.c.vocab);
        for (int s = 0; s < 5; ++s)
            cpu[p].site_metric[s] =
                compare_values(first.site[s], cpu[p].site[s], gm.c.hidden);
        const size_t km = (size_t)gm.c.kda_heads * gm.c.kda_hd * gm.c.kda_hd;
        const size_t kw = (size_t)3 * gm.c.kda_proj * gm.c.conv_k;
        const size_t lf = (size_t)cpu[p].count * gm.c.kv_lora;
        const size_t ix = (size_t)cpu[p].count * gm.c.index_hd;
        cpu[p].kda_matrix_metric =
            compare_values(first.kda_matrix, cpu[p].kda_matrix, km);
        cpu[p].kda_window_metric =
            compare_values(first.kda_window, cpu[p].kda_window, kw);
        cpu[p].mla_latent_metric =
            compare_values(first.mla_latent, cpu[p].mla_latent, lf);
        cpu[p].mla_keys_metric =
            compare_values(first.mla_keys, cpu[p].mla_keys, ix);
        cpu[p].mla_gates_metric =
            compare_values(first.mla_gates, cpu[p].mla_gates, ix);
        cpu[p].gpu_deterministic =
            first.logits_hash == second.logits_hash &&
            first.kda_hash == second.kda_hash &&
            first.mla_hash == second.mla_hash &&
            first.generated_count == second.generated_count &&
            !memcmp(first.generated, second.generated,
                    sizeof(first.generated));
        cpu[p].generated_agree =
            first.generated_count == cpu[p].generated_count &&
            !memcmp(first.generated, cpu[p].generated,
                    sizeof(first.generated));
        cpu[p].kda_hash = first.kda_hash;
        cpu[p].mla_hash = first.mla_hash;
        cpu[p].logits_hash = first.logits_hash;
        cpu[p].gpu_nll = first.gpu_nll;
        memcpy(cpu[p].gpu_generated, first.generated,
               sizeof(cpu[p].gpu_generated));
        cpu_mean += cpu[p].cpu_nll;
        gpu_mean += first.gpu_nll;
        Metric all[] = {
            cpu[p].logits_metric, cpu[p].site_metric[0],
            cpu[p].site_metric[1], cpu[p].site_metric[2],
            cpu[p].site_metric[3], cpu[p].site_metric[4],
            cpu[p].kda_matrix_metric, cpu[p].kda_window_metric,
            cpu[p].mla_latent_metric, cpu[p].mla_keys_metric,
            cpu[p].mla_gates_metric
        };
        for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i)
            gates = gates && all[i].finite &&
                all[i].rel_l2 <= 1e-3 && all[i].cosine >= 0.9999;
        gates = gates && cpu[p].gpu_deterministic;
        free(first.logits); free(second.logits);
        free(first.kda_matrix); free(first.kda_window);
        free(first.mla_latent); free(first.mla_keys); free(first.mla_gates);
        for (int s = 0; s < 5; ++s) {
            free(first.site[s]); free(second.site[s]);
        }
    }
    cpu_mean /= QUALITY_PROMPTS;
    gpu_mean /= QUALITY_PROMPTS;
    double nll_change = fabs(gpu_mean - cpu_mean) / cpu_mean;
    gates = gates && nll_change <= 0.005;
    coli_gpu_context_telemetry(
        coli_glm53_gpu_backend_context(&g_glm53_gpu_backend),
        &telemetry_after);

    FILE *f = fopen(output_path, "wb");
    if (!f) return 1;
    fprintf(f,
        "{\"schema\":\"glm53-real-quality-v1\",\"model\":\"%s\","
        "\"git_commit\":\"ea28811+task7\",\"gpu\":\"AMD Instinct MI350P\","
        "\"representative_layers\":{\"kda\":%d,\"mla_dsa\":%d,"
        "\"dense_ffn\":%d,\"moe\":%d},\"prompts\":[",
        model_path, kl, ml, dl, ol);
    for (int p = 0; p < QUALITY_PROMPTS; ++p) {
        if (p) fputc(',', f);
        fprintf(f, "{\"text\":\"%s\",\"ids\":[", cpu[p].text);
        for (int i = 0; i < cpu[p].count; ++i)
            fprintf(f, "%s%d", i ? "," : "", cpu[p].ids[i]);
        fprintf(f, "],\"cpu_nll\":%.12g,\"gpu_nll\":%.12g,"
                   "\"generated_cpu\":[",
                cpu[p].cpu_nll, cpu[p].gpu_nll);
        for (int i = 0; i < cpu[p].generated_count; ++i)
            fprintf(f, "%s%d", i ? "," : "", cpu[p].generated[i]);
        fprintf(f, "],\"generated_gpu\":[");
        for (int i = 0; i < QUALITY_GENERATED; ++i)
            fprintf(f, "%s%d", i ? "," : "", cpu[p].gpu_generated[i]);
        fprintf(f, "],\"generated_agreement\":%s,"
                   "\"gpu_deterministic\":%s,"
                   "\"hashes\":{\"logits\":\"%016" PRIx64 "\","
                   "\"kda\":\"%016" PRIx64 "\",\"mla\":\"%016" PRIx64 "\"},"
                   "\"metrics\":{",
                cpu[p].generated_agree ? "true" : "false",
                cpu[p].gpu_deterministic ? "true" : "false",
                cpu[p].logits_hash, cpu[p].kda_hash, cpu[p].mla_hash);
        print_metric(f, "full_logits", cpu[p].logits_metric); fputc(',', f);
        static const char *names[5] = {
            "kda_site", "mla_dsa_site", "dense_ffn_site",
            "routed_shared_moe_site", "final_norm_lm_head_input"
        };
        for (int s = 0; s < 5; ++s) {
            print_metric(f, names[s], cpu[p].site_metric[s]); fputc(',', f);
        }
        print_metric(f, "kda_state", cpu[p].kda_matrix_metric); fputc(',', f);
        print_metric(f, "kda_window", cpu[p].kda_window_metric); fputc(',', f);
        print_metric(f, "mla_latent", cpu[p].mla_latent_metric); fputc(',', f);
        print_metric(f, "mla_index_keys", cpu[p].mla_keys_metric); fputc(',', f);
        print_metric(f, "mla_index_gates", cpu[p].mla_gates_metric);
        fprintf(f, "}}");
    }
    fprintf(f,
        "],\"mean_cpu_nll\":%.12g,\"mean_gpu_nll\":%.12g,"
        "\"relative_mean_nll_change\":%.12g,"
        "\"telemetry\":{\"device_allocations\":%" PRIu64 ","
        "\"host_activation_h2d_copies\":%" PRIu64 ","
        "\"host_activation_d2h_copies\":%" PRIu64 ","
        "\"expert_upload_bytes\":%" PRIu64 "},\"gates_pass\":%s}\n",
        cpu_mean, gpu_mean, nll_change,
        telemetry_after.device_allocations - telemetry_before.device_allocations,
        telemetry_after.host_activation_h2d_copies -
            telemetry_before.host_activation_h2d_copies,
        telemetry_after.host_activation_d2h_copies -
            telemetry_before.host_activation_d2h_copies,
        telemetry_after.expert_upload_bytes -
            telemetry_before.expert_upload_bytes,
        gates ? "true" : "false");
    fclose(f);
    model_release(&gm);
    tok_free(&tokenizer);
    return gates ? 0 : 1;
}
