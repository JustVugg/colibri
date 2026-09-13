#define GLM53_NO_MAIN
#include "../glm53.c"

typedef struct {
    int context_create;
    int context_probe;
    int context_destroy;
    int model_create;
    int model_destroy;
    int session_create;
    int session_destroy;
    int forward;
} StartupCounts;

static StartupCounts counts;

static Mat fixture_f32(int rows, int columns) {
    Mat mat = {0};
    mat.fmt = 0;
    mat.rows = rows;
    mat.columns = columns;
    mat.f = (float *)calloc((size_t)rows * columns, sizeof(float));
    return mat;
}

static Mat fixture_g4(int rows, int columns) {
    Mat mat = {0};
    mat.fmt = 4;
    mat.rows = rows;
    mat.columns = columns;
    mat.gs = 64;
    mat.q4 = (uint8_t *)calloc(
        (size_t)rows * (size_t)(columns + 1) / 2, 1);
    mat.s = (float *)malloc((size_t)rows * sizeof(float));
    if (mat.s)
        for (int row = 0; row < rows; ++row)
            ((float *)mat.s)[row] = 0.1f;
    return mat;
}

static int fixture_model(GModel *m) {
    memset(m, 0, sizeof(*m));
    Cfg *c = &m->c;
    c->hidden = 2;
    c->n_layers = 1;
    c->vocab = 2;
    c->first_dense = 0;
    c->dense_inter = 2;
    c->kda_heads = 1;
    c->kda_hd = 2;
    c->kda_proj = 2;
    c->conv_k = 1;
    c->gate_lb = -5.0f;
    c->n_experts = 2;
    c->topk = 1;
    c->moe_inter = 2;
    c->routed_scale = 0.75f;
    c->swiglu_limit = 1.5f;
    c->hc_mult = 4;
    c->hc_iters = 20;
    c->eps = 1e-6f;
    c->hc_eps = 1e-6f;
    m->layer_begin = 0;
    m->layer_end = 1;
    m->embed = (float *)malloc(4 * sizeof(float));
    m->final_norm = (float *)malloc(2 * sizeof(float));
    m->head = fixture_f32(2, 2);
    m->layer = (GLayer *)calloc(1, sizeof(*m->layer));
    if (!m->embed || !m->final_norm || !m->head.f || !m->layer)
        return 0;
    ((float *)m->embed)[0] = 0.4f;
    ((float *)m->embed)[1] = -0.7f;
    ((float *)m->embed)[2] = 0.25f;
    ((float *)m->embed)[3] = 0.75f;
    ((float *)m->final_norm)[0] = 1.0f;
    ((float *)m->final_norm)[1] = 1.0f;
    ((float *)m->head.f)[0] = 1.0f;
    ((float *)m->head.f)[3] = 1.0f;

    GLayer *layer = &m->layer[0];
    const int mix_count = (2 + c->hc_mult) * c->hc_mult;
    const int fn_count = mix_count * c->hc_mult * c->hidden;
    layer->in_ln = (float *)malloc(2 * sizeof(float));
    layer->post_ln = (float *)malloc(2 * sizeof(float));
    layer->hc_attn_fn = (float *)calloc((size_t)fn_count, sizeof(float));
    layer->hc_attn_base =
        (float *)calloc((size_t)mix_count, sizeof(float));
    layer->hc_attn_scale = (float *)calloc(3, sizeof(float));
    layer->hc_ffn_fn = (float *)calloc((size_t)fn_count, sizeof(float));
    layer->hc_ffn_base =
        (float *)calloc((size_t)mix_count, sizeof(float));
    layer->hc_ffn_scale = (float *)calloc(3, sizeof(float));
    if (!layer->in_ln || !layer->post_ln || !layer->hc_attn_fn ||
        !layer->hc_attn_base || !layer->hc_attn_scale ||
        !layer->hc_ffn_fn || !layer->hc_ffn_base ||
        !layer->hc_ffn_scale)
        return 0;
    ((float *)layer->in_ln)[0] = ((float *)layer->in_ln)[1] = 1.0f;
    ((float *)layer->post_ln)[0] =
        ((float *)layer->post_ln)[1] = 1.0f;

    layer->kq = fixture_f32(2, 2);
    layer->kk = fixture_f32(2, 2);
    layer->kv = fixture_f32(2, 2);
    layer->ko = fixture_f32(2, 2);
    layer->kga = fixture_f32(2, 2);
    layer->kgb = fixture_f32(2, 2);
    layer->kfa = fixture_f32(2, 2);
    layer->kfb = fixture_f32(2, 2);
    layer->kb = fixture_f32(1, 2);
    layer->conv = (float *)calloc(6, sizeof(float));
    layer->dt = (float *)calloc(2, sizeof(float));
    layer->alog = (float *)calloc(1, sizeof(float));
    layer->onorm = (float *)malloc(2 * sizeof(float));
    layer->router = (float *)malloc(4 * sizeof(float));
    layer->rg = fixture_f32(2, 2);
    layer->ru = fixture_f32(2, 2);
    layer->rd = fixture_f32(2, 2);
    layer->eg = (Mat *)malloc(2 * sizeof(Mat));
    layer->eu = (Mat *)malloc(2 * sizeof(Mat));
    layer->ed = (Mat *)malloc(2 * sizeof(Mat));
    if (!layer->kq.f || !layer->kk.f || !layer->kv.f || !layer->ko.f ||
        !layer->kga.f || !layer->kgb.f || !layer->kfa.f ||
        !layer->kfb.f || !layer->kb.f || !layer->conv || !layer->dt ||
        !layer->alog || !layer->onorm || !layer->router || !layer->rg.f ||
        !layer->ru.f || !layer->rd.f || !layer->eg || !layer->eu ||
        !layer->ed)
        return 0;
    ((float *)layer->onorm)[0] =
        ((float *)layer->onorm)[1] = 1.0f;
    ((float *)layer->router)[0] = 1.0f;
    ((float *)layer->router)[1] = -1.0f;
    ((float *)layer->router)[2] = -1.0f;
    ((float *)layer->router)[3] = 1.0f;
    for (int expert = 0; expert < 2; ++expert) {
        layer->eg[expert] = fixture_g4(2, 2);
        layer->eu[expert] = fixture_g4(2, 2);
        layer->ed[expert] = fixture_g4(2, 2);
        if (!layer->eg[expert].q4 || !layer->eg[expert].s ||
            !layer->eu[expert].q4 || !layer->eu[expert].s ||
            !layer->ed[expert].q4 || !layer->ed[expert].s)
            return 0;
    }
    return 1;
}

static void fixture_release(GModel *m) {
    m->gpu_model = NULL;
    m->has_io = 0;
    model_release(m);
}

static int counted_context_create(ColiGpuContext **out, int device) {
    counts.context_create++;
    return coli_gpu_context_create(out, device);
}

static int counted_context_probe(
    ColiGpuContext *ctx, uint64_t required_caps) {
    counts.context_probe++;
    return glm53_gpu_complete_probe(ctx, required_caps);
}

static void counted_context_destroy(ColiGpuContext *ctx) {
    if (ctx) counts.context_destroy++;
    coli_gpu_context_destroy(ctx);
}

static int counted_model_create(
    GModel *m, ColiGpuContext *ctx, int max_rows, int max_context) {
    counts.model_create++;
    return glm53_gpu_model_from_loaded(m, ctx, max_rows, max_context);
}

static void counted_model_destroy(ColiGlm53GpuModel *model) {
    if (model) counts.model_destroy++;
    coli_glm53_gpu_model_destroy(model);
}

static int counted_session_create(
    ColiGlm53GpuSession **out, ColiGlm53GpuModel *model,
    int max_context) {
    counts.session_create++;
    return coli_glm53_gpu_session_create(out, model, max_context);
}

static void counted_session_destroy(ColiGlm53GpuSession *session) {
    if (session) counts.session_destroy++;
    coli_glm53_gpu_session_destroy(session);
}

static int counted_forward(
    ColiGlm53GpuSession *session, const int *tokens, int rows,
    float *logits) {
    counts.forward++;
    return coli_glm53_gpu_forward(session, tokens, rows, logits);
}

static Glm53GpuStartupOps counted_ops(void) {
    Glm53GpuStartupOps ops = glm53_gpu_startup_ops();
    ops.backend.context_create = counted_context_create;
    ops.backend.context_probe = counted_context_probe;
    ops.backend.context_destroy = counted_context_destroy;
    ops.model_from_loaded = counted_model_create;
    ops.model_destroy = counted_model_destroy;
    ops.session_create = counted_session_create;
    ops.session_destroy = counted_session_destroy;
    ops.forward = counted_forward;
    return ops;
}

static int startup_failure_counts_ok(
    ColiGpuFaultPoint fault, const StartupCounts *got) {
    int model_owned =
        fault != COLI_GPU_FAULT_MODEL_ALLOCATION &&
        fault != COLI_GPU_FAULT_STREAM_SYNC;
    int session_attempt =
        fault != COLI_GPU_FAULT_MODEL_ALLOCATION &&
        fault != COLI_GPU_FAULT_STREAM_SYNC;
    int session_owned =
        session_attempt && fault != COLI_GPU_FAULT_SESSION_ALLOCATION;
    int forward_attempt = session_owned;
    return got->context_create == 1 && got->context_probe == 1 &&
           got->context_destroy == 1 && got->model_create == 1 &&
           got->model_destroy == model_owned &&
           got->session_create == session_attempt &&
           got->session_destroy == session_owned &&
           got->forward == forward_attempt;
}

static int test_gpu_cache_capacity_accounts_for_double_banks(void) {
    const int sparse_layers = 42;
    const int64_t slot_bytes = 14155776;
    const size_t free_bytes = 154602045440ull;
    const int logical_host_slots = 162;
    const int cap = glm53_gpu_cache_capacity(
        logical_host_slots, sparse_layers, slot_bytes, free_bytes);
    const size_t physical =
        (size_t)cap * (size_t)sparse_layers * (size_t)slot_bytes * 2u;
    GModel model = {0};
    model.c.n_layers = 45;
    model.c.first_dense = 3;
    model.e_slot = slot_bytes;
    model.streaming = 1;
    model.ecache = (LCache *)calloc(45, sizeof(*model.ecache));
    if (!model.ecache) return 0;
    for (int layer = 3; layer < 45; ++layer) {
        model.ecache[layer].cap = logical_host_slots;
        model.ecache[layer].s =
            (Slot *)calloc(logical_host_slots, sizeof(Slot));
        if (!model.ecache[layer].s) return 0;
    }
    int resized = glm53_gpu_apply_cache_capacity(&model, free_bytes);
    int all_match = 1;
    for (int layer = 3; layer < 45; ++layer) {
        all_match = all_match && model.ecache[layer].cap == cap;
        free(model.ecache[layer].s);
    }
    free(model.ecache);
    return cap == 97 && resized && all_match &&
           physical <= free_bytes - free_bytes / 4u &&
           cap < logical_host_slots;
}

static int test_post_loader_startup_faults(void) {
    static const ColiGpuFaultPoint faults[] = {
        COLI_GPU_FAULT_MODEL_ALLOCATION,
        COLI_GPU_FAULT_SESSION_ALLOCATION,
        COLI_GPU_FAULT_TENSOR_UPLOAD,
        COLI_GPU_FAULT_EXPERT_PUBLICATION,
        COLI_GPU_FAULT_KERNEL_LAUNCH,
        COLI_GPU_FAULT_DEVICE_STATUS,
        COLI_GPU_FAULT_STREAM_SYNC
    };
    for (size_t at = 0; at < sizeof(faults) / sizeof(faults[0]); ++at) {
        for (int forced = 0; forced < 2; ++forced) {
            GModel model;
            if (!fixture_model(&model)) return 0;
            memset(&counts, 0, sizeof(counts));
            coli_glm53_gpu_backend_init(&g_glm53_gpu_backend);
            Glm53GpuStartupOps ops = counted_ops();
            char err[160] = {0};
            int result = glm53_gpu_startup_loaded(
                &model,
                forced ? COLI_GLM53_GPU_MODE_GPU :
                         COLI_GLM53_GPU_MODE_AUTO,
                &ops, 0, 2, 4, faults[at], 0, err, sizeof(err));
            int ok =
                (forced ? !result : result) &&
                coli_glm53_gpu_backend_selected(&g_glm53_gpu_backend) ==
                    (forced ? COLI_GLM53_BACKEND_UNSELECTED :
                              COLI_GLM53_BACKEND_CPU) &&
                !model.gpu_model &&
                startup_failure_counts_ok(faults[at], &counts);
            StartupCounts before_destroy = counts;
            coli_glm53_gpu_backend_destroy(
                &g_glm53_gpu_backend, ops.backend);
            ok = ok &&
                 !memcmp(&counts, &before_destroy, sizeof(counts));
            fixture_release(&model);
            if (!ok) {
                fprintf(stderr,
                        "FAIL: post-loader startup fault %d mode %s "
                        "counts=%d/%d/%d %d/%d %d/%d/%d err=%s\n",
                        (int)faults[at], forced ? "gpu" : "auto",
                        counts.context_create, counts.context_probe,
                        counts.context_destroy, counts.model_create,
                        counts.model_destroy, counts.session_create,
                        counts.session_destroy, counts.forward, err);
                return 0;
            }
        }
    }
    return 1;
}

static int test_dispatcher_gpu_failure_then_real_cpu(void) {
    static const ColiGpuFaultPoint faults[] = {
        COLI_GPU_FAULT_SESSION_ALLOCATION,
        COLI_GPU_FAULT_TENSOR_UPLOAD,
        COLI_GPU_FAULT_EXPERT_PUBLICATION
    };
    for (size_t at = 0; at < sizeof(faults) / sizeof(faults[0]); ++at) {
        GModel model;
        if (!fixture_model(&model)) return 0;
        memset(&counts, 0, sizeof(counts));
        coli_glm53_gpu_backend_init(&g_glm53_gpu_backend);
        Glm53GpuStartupOps ops = counted_ops();
        char err[160] = {0};
        if (!glm53_gpu_startup_loaded(
                &model, COLI_GLM53_GPU_MODE_GPU, &ops, 0, 2, 4,
                COLI_GPU_FAULT_NONE, 0, err, sizeof(err))) {
            fixture_release(&model);
            return 0;
        }
        ColiGpuContext *ctx =
            coli_glm53_gpu_backend_context(&g_glm53_gpu_backend);
        const int token = 1;
        GSession *session = NULL;
        int ok = glm53_request_begin() &&
                 coli_gpu_context_inject_fault(ctx, faults[at], 0);
        float *failed = glm53_request_dispatch(
            &model, &session, 4, &token, 1, NULL, 0, 1, 0);
        ok = ok && !failed && !session &&
             !coli_gpu_context_healthy(ctx) &&
             coli_glm53_gpu_backend_selected(&g_glm53_gpu_backend) ==
                 COLI_GLM53_BACKEND_CPU &&
             g_glm53_gpu_backend.active_requests == 1;
        free(failed);
        glm53_request_end();

        ColiGpuTelemetry before = {0}, after = {0};
        coli_gpu_context_telemetry(ctx, &before);
        ok = ok && glm53_request_begin();
        float *cpu_logits = glm53_request_dispatch(
            &model, &session, 4, &token, 1, NULL, 0, 1, 0);
        coli_gpu_context_telemetry(ctx, &after);
        const float observed0 = cpu_logits ? cpu_logits[0] : NAN;
        const float observed1 = cpu_logits ? cpu_logits[1] : NAN;
        ok = ok && cpu_logits && session && session->layer &&
             !session->gpu && isfinite(cpu_logits[0]) &&
             isfinite(cpu_logits[1]) &&
             fabsf(cpu_logits[0]) + fabsf(cpu_logits[1]) > 0.1f &&
             !memcmp(&before, &after, sizeof(before)) &&
             coli_glm53_gpu_backend_selected(&g_glm53_gpu_backend) ==
                 COLI_GLM53_BACKEND_CPU;
        free(cpu_logits);
        session_close(&model, session);
        glm53_request_end();
        counted_model_destroy(model.gpu_model);
        model.gpu_model = NULL;
        coli_glm53_gpu_backend_destroy(
            &g_glm53_gpu_backend, ops.backend);
        fixture_release(&model);
        if (!ok) {
            fprintf(stderr,
                    "FAIL: dispatcher fault %d did not run next request "
                    "on real CPU path (logits %.9g %.9g)\n",
                    (int)faults[at], observed0, observed1);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    if (!test_gpu_cache_capacity_accounts_for_double_banks() ||
        !test_post_loader_startup_faults() ||
        !test_dispatcher_gpu_failure_then_real_cpu())
        return 1;
    puts("glm53 production request/startup integration: PASS");
    return 0;
}
