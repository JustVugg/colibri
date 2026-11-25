/* Q38_KV=f16 (qwen38_core.h's q38_kv16): host K/V bank in float16 (uint16_t).
 *
 * On the tiny fixtures (make qwen38-tiny-generate qwen38-tiny-fp8-generate):
 *   - Q38_KV unset: float32 for argv, float16 when SERVE=1; Q38_KV=f16 / f32 overrides;
 *   - float16 halves the host K and V banks (index keys stay float32): verified on the
 *     fixture's bank size, q38_state_later, q38_segment_session_state_bytes, and reported
 *     for the release geometry;
 *   - bit-identity of q38_dot8_h / q38_dot8_h_x4 and kv16_axpy(_x2/_x4) against the
 *     float32-stored f16-rounded kernels (dev's previous behavior);
 *   - a greedy run of NGEN tokens in float32, then the same tokens teacher-forced in
 *     float16: max |dlogit| <= 1e-3 and top-1 >= 95%;
 *   - verify rows match decode rows: prompt lookup with rejections (COLI_LOOKUP_FORCE=mixed)
 *     gives every step the plain float16 decode's bits, and a prompt read whole gives the
 *     bits of a prompt read a token at a time (both float32 and float16);
 *   - the CUDA chain on the fake card (tests/qwen36_fake_cuda.h): a backend without
 *     CC_EW_F16 keeps the chain off under Q38_KV=f16; with CC_EW_F16 (float32 mirrors),
 *     the host's float16 rows widened are bit-identical to the mirror before and after a
 *     rewind/upload, and the chain's float16 logits stay within tolerance of its float32. */
#define main qwen38_main_unused
#include "../qwen38.c"
#undef main

#ifdef _WIN32
#undef unsetenv
#define unsetenv(name) _putenv_s(name, "")
#endif

#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int t_fails;
static void tk(int ok, const char *what) {
    if (ok) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    t_fails++;
}

#define KV16_MAX_ABS 1e-3
#define KV16_TOP1_MIN 0.95

#define NP 16
#define NGEN 96
static const int g_prompt[NP] = {1, 3, 4, 5, 6, 9, 17, 33, 12, 40, 7, 21, 58, 3, 4, 5};

static void kv_mode(Model *m, int kv) {
    g_q38_kv16 = kv;
    m->kv_cap = 0;
    ensure_kv(m);
}
static int argmax(const float *x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

static void run(Model *m, const int *force, int n, float *L, int *tok, Q38Spec *sp) {
    int V = m->c.vocab;
    reset_recurrent(m); m->kv_len = 0;
    float *logit = step(m, g_prompt, NP, 0);
    for (int s = 0; s < n; s++) {
        memcpy(L + (size_t)s * V, logit, (size_t)V * sizeof(float));
        int t = force ? force[s] : argmax(logit, V);
        tok[s] = t; free(logit);
        if (s == n - 1) break;
        logit = sp ? q38_spec_step(m, sp, t, NP + s, n - 1 - s) : step(m, &t, 1, NP + s);
    }
    if (sp) q38_spec_end(m, sp);
}

static void diff(const float *A, const float *B, int n, int V, double *maxd, int *top1) {
    *maxd = 0; *top1 = 0;
    for (int s = 0; s < n; s++) {
        const float *a = A + (size_t)s * V, *b = B + (size_t)s * V;
        for (int i = 0; i < V; i++) { double d = fabs((double)a[i] - b[i]); if (d > *maxd) *maxd = d; }
        *top1 += argmax(a, V) == argmax(b, V);
    }
}

static int agree(const int *a, const int *b, int n) {
    int k = 0;
    while (k < n && a[k] == b[k]) k++;
    return k;
}

static size_t kv_bank_bytes(const Model *m) {
    const Cfg *c = &m->c; size_t b = 0;
    for (int i = 0; i < q38_layer_rows(m); i++)
        if (i == c->layers || c->is_attn[i])
            b += (size_t)m->kv_cap * (size_t)2 * c->kv_heads * c->head_dim * q38_kv_es();
    return b;
}

/* Kernel-level bit-identity against dev's f32-stored f16-rounded attention kernels. */
static void t_kernels_bit_identical(void) {
    printf(" kernel bit-identity (q38_dot8_h[_x4], kv16_axpy[_x2/_x4] vs f16-rounded f32)\n");
    const int D = 256;
    float q[4 * 256], kf[256], vf[256];
    uint16_t kh[256], vh[256];
    for (int i = 0; i < 4 * D; i++) q[i] = sinf((float)(i + 1) * 0.137f) * 1.7f;
    for (int i = 0; i < D; i++) {
        kf[i] = cosf((float)(i + 3) * 0.219f) * 2.3f;
        vf[i] = sinf((float)(i + 7) * 0.311f) * 1.9f;
    }
    kv16_pack(kh, kf, D);
    kv16_pack(vh, vf, D);
    kv16_round(kf, D);
    kv16_round(vf, D);

    float d_ref[4], d_h[4];
    q38_dot8_x4(d_ref, q, D, kf, D);
    q38_dot8_h_x4(d_h, q, D, kh, D);
    int dot_ok = memcmp(d_ref, d_h, sizeof d_ref) == 0;
    for (int g = 0; g < 4; g++) {
        float r1 = q38_dot8(q + g * D, kf, D);
        float h1 = q38_dot8_h(q + g * D, kh, D);
        if (memcmp(&r1, &h1, sizeof(float)) != 0) dot_ok = 0;
    }
    tk(dot_ok, "q38_dot8_h and q38_dot8_h_x4 bit-identical to q38_dot8[_x4] over f16-rounded f32");

    float o_ref[4][256], o_h[4][256];
    float w[4] = {0.17f, 0.29f, 0.41f, 0.13f};
    for (int g = 0; g < 4; g++) {
        for (int i = 0; i < D; i++) o_ref[g][i] = o_h[g][i] = cosf((float)(g * D + i) * 0.05f);
        for (int i = 0; i < D; i++) o_ref[g][i] = MATMUL_F32_MADD(o_ref[g][i], w[g], vf[i]);
    }
    kv16_axpy_x4(o_h[0], o_h[1], o_h[2], o_h[3], w[0], w[1], w[2], w[3], vh, D);
    tk(memcmp(o_ref, o_h, sizeof o_ref) == 0, "kv16_axpy_x4 bit-identical to fused f32 axpy over f16-rounded f32");

    for (int g = 0; g < 2; g++) {
        for (int i = 0; i < D; i++) o_ref[g][i] = o_h[g][i] = sinf((float)(g * D + i) * 0.07f);
        for (int i = 0; i < D; i++) o_ref[g][i] = MATMUL_F32_MADD(o_ref[g][i], w[g], vf[i]);
    }
    kv16_axpy_x2(o_h[0], o_h[1], w[0], w[1], vh, D);
    tk(memcmp(o_ref[0], o_h[0], (size_t)D * sizeof(float)) == 0 &&
       memcmp(o_ref[1], o_h[1], (size_t)D * sizeof(float)) == 0,
       "kv16_axpy_x2 bit-identical to fused f32 axpy over f16-rounded f32");
}

static void t_cpu(void) {
    printf(" the CPU path on qwen38_tiny_fp8\n");
    unsetenv("Q38_KV"); unsetenv("SERVE"); g_q38_kv16 = -1;
    tk(!q38_kv16() && q38_kv_es() == 4, "Q38_KV unset (argv): K and V float32 (4 B/elem)");
    setenv("SERVE", "1", 1); g_q38_kv16 = -1;
    tk(q38_kv16() && q38_kv_es() == 2, "Q38_KV unset (SERVE=1): K and V float16 (2 B/elem)");
    setenv("Q38_KV", "f32", 1); g_q38_kv16 = -1;
    tk(!q38_kv16() && q38_kv_es() == 4, "Q38_KV=f32 overrides SERVE=1");
    unsetenv("SERVE"); setenv("Q38_KV", "f16", 1); g_q38_kv16 = -1;
    tk(q38_kv16() && q38_kv_es() == 2, "Q38_KV=f16: K and V float16 (2 B/elem)");
    unsetenv("Q38_KV"); g_q38_kv16 = -1;

    Model m; model_init(&m, "./qwen38_tiny_fp8", 4, 8);
    const Cfg *c = &m.c; int V = c->vocab, n = NGEN;
    m.max_t = NP + NGEN;
    float *L32 = falloc((int64_t)n * V), *L16 = falloc((int64_t)n * V), *Ls = falloc((int64_t)n * V);
    int T32[NGEN], T16[NGEN], Ts[NGEN], full[NP + NGEN];

    /* host K/V bank bytes and state accounting */
    kv_mode(&m, 0);
    size_t bank32 = kv_bank_bytes(&m), later32 = q38_state_later(&m);
    uint64_t seg32 = 0, seg16 = 0;
    q38_segment_session_state_bytes(c, 0, (uint32_t)c->layers, (uint32_t)m.max_t, &seg32);
    kv_mode(&m, 1);
    size_t bank16 = kv_bank_bytes(&m), later16 = q38_state_later(&m);
    q38_segment_session_state_bytes(c, 0, (uint32_t)c->layers, (uint32_t)m.max_t, &seg16);
    printf("  kv16: fixture K/V bank f32=%zu B -> f16=%zu B; q38_state_later %zu -> %zu B\n",
           bank32, bank16, later32, later16);
    tk(bank16 * 2 == bank32 && bank16 > 0, "host K/V bank at max_t: exactly half of float32");
    tk(later32 - later16 == bank32 - bank16, "q38_state_later reflects halved K/V bank");
    tk(seg32 - seg16 == bank32 - bank16, "q38_segment_session_state_bytes reflects halved K/V bank");

    /* float32 greedy reference vs float16 teacher-forced and greedy */
    kv_mode(&m, 0);
    run(&m, NULL, n, L32, T32, NULL);
    kv_mode(&m, 1);
    double maxd; int top1;
    run(&m, T32, n, L16, T16, NULL);
    diff(L32, L16, n, V, &maxd, &top1);
    run(&m, NULL, n, L16, T16, NULL);
    int same = agree(T32, T16, n);
    printf("  kv16: f16 teacher-forced on %d tokens: max |dlogit| %.3g, top-1 %d/%d; greedy: first %d/%d agree\n",
           n, maxd, top1, n, same, n);
    char msg[200];
    snprintf(msg, sizeof msg, "f16 vs f32: max |dlogit| %.3g <= %g, top-1 %d/%d >= %.0f%%",
             maxd, KV16_MAX_ABS, top1, n, 100 * KV16_TOP1_MIN);
    tk(maxd <= KV16_MAX_ABS && top1 >= KV16_TOP1_MIN * n, msg);

    /* verify rows against decode rows in float16 */
    memcpy(full, g_prompt, sizeof g_prompt); memcpy(full + NP, T16, sizeof T16);
    g_q38_mtp_oracle = full; g_q38_mtp_oracle_n = NP + n;
    setenv("COLI_LOOKUP", "1", 1); setenv("COLI_LOOKUP_FORCE", "mixed", 1); setenv("COLI_SPEC_GATE", "0", 1);
    Q38Spec sp = q38_spec_begin(&m, g_prompt, NP);
    run(&m, NULL, n, Ls, Ts, &sp);
    unsetenv("COLI_LOOKUP"); unsetenv("COLI_LOOKUP_FORCE"); unsetenv("COLI_SPEC_GATE");
    snprintf(msg, sizeof msg, "f16: %llu verifies (%llu of %llu drafts accepted): every step plain decode's bits",
             (unsigned long long)sp.lk_verifies, (unsigned long long)sp.lk_accepted, (unsigned long long)sp.lk_drafts);
    tk(sp.lk_verifies > 4 && sp.lk_accepted > 0 && sp.lk_accepted < sp.lk_drafts &&
       !memcmp(Ts, T16, sizeof T16) && !memcmp(Ls, L16, (size_t)n * V * sizeof(float)), msg);
    q38_spec_free(&sp);
    g_q38_mtp_oracle = NULL; g_q38_mtp_oracle_n = 0;

    /* prompt whole vs token-at-a-time */
    for (int k = 0; k < 2; k++) {
        kv_mode(&m, k);
        reset_recurrent(&m); m.kv_len = 0;
        float *whole = step(&m, g_prompt, NP, 0), *one = NULL;
        reset_recurrent(&m); m.kv_len = 0;
        for (int s = 0; s < NP; s++) { free(one); one = step(&m, g_prompt + s, 1, s); }
        snprintf(msg, sizeof msg, "%s: prompt read whole gives bits of token-at-a-time", k ? "float16" : "float32");
        tk(!memcmp(whole, one, (size_t)V * sizeof(float)), msg);
        free(whole); free(one);
    }

    free(L32); free(L16); free(Ls);
    g_q38_kv16 = -1;
    q38_model_free(&m);
}

/* ---- CUDA chain on the fake card ---- */
static float t_f16(float x) { return kv16_to(kv16_from(x)); }
static int t_kv16_on(void) { return 1; }

static void engine_env(void) {
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("NOSTREAM", "1", 1);
    setenv("USAGE_SAVE", "0", 1);
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("QT_UPLOAD_SYNC", "1", 1);
    setenv("HEAT_FILE", "", 1);
    setenv("Q38_TRUNK_GPU", "1", 1);
    setenv("Q38_TRUNK_MIN_KB", "0", 1);
    setenv("COLI_PLACE", "auto", 1);
    setenv("CUDA_EXPERT_GB", "0.5", 1);
    setenv("Q38_MTP", "0", 1);
    setenv("Q38_EXPERT_INT4", "0", 1);
    fake_dense_compute = 1;
}

static int mirror_is_host(Model *m, Q38CChain *ch, int upto) {
    const Cfg *c = &m->c; int D = c->head_dim, ID = c->idx_dim, bad = 0;
    for (int i = 0; i < c->layers; i++) {
        if (!c->is_attn[i]) continue;
        for (int h = 0; h < c->kv_heads; h++)
            for (int t = 0; t < upto; t++) {
                float k[256], v[256];
                q38_kv_get(q38_kv16(), k, m->K[i], ((int64_t)h * m->kv_cap + t) * D, D);
                q38_kv_get(q38_kv16(), v, m->V[i], ((int64_t)h * m->kv_cap + t) * D, D);
                bad += memcmp(k, ch->kc[i]->d + ((size_t)h * ch->cap + t) * D, (size_t)D * 4) != 0;
                bad += memcmp(v, ch->vc[i]->d + ((size_t)h * ch->cap + t) * D, (size_t)D * 4) != 0;
            }
        for (int t = 0; t < upto; t++)
            bad += memcmp(m->IK[i] + (size_t)t * ID, ch->ik[i]->d + (size_t)t * ID, (size_t)ID * 4) != 0;
    }
    return bad == 0;
}

static void t_chain(void) {
    printf(" the CUDA chain on the fake card, qwen38_tiny_fp8\n");
    engine_env();
    const int n = 24;
    Model m; model_init(&m, "./qwen38_tiny_fp8", 8, 8);
    q38_tier_start(&m, 8);
    int V = m.c.vocab;
    m.max_t = NP + n;
    float *Lp = falloc((int64_t)n * V), *Lc = falloc((int64_t)n * V);
    float *Lp32 = falloc((int64_t)n * V), *Lc32 = falloc((int64_t)n * V);
    int Tp[NGEN], Tc[NGEN];

    kv_mode(&m, 0); run(&m, NULL, n, Lp32, Tp, NULL);
    kv_mode(&m, 1); run(&m, NULL, n, Lp, Tp, NULL);

    setenv("COLI_CUDA_CHAIN", "1", 1);
    tk(q38cc_setup(&m, 1) == NULL && (!m.cchain || !((Q38CChain *)m.cchain)->ok),
       "backend without CC_EW_F16: chain stays off under float16 host cache");
    m.cchain = NULL;

    size_t size0 = fake_chain_table.size;
    fake_chain_table.size = offsetof(ColiCudaChainOps, kv16) + sizeof fake_chain_table.kv16;
    fake_chain_table.kv16 = t_kv16_on; fake_chain_f16 = t_f16;
    Q38CChain *ch = q38cc_setup(&m, 1);
    g_cuda_chain = ch != NULL;
    tk(g_cuda_chain, "chain enabled with CC_EW_F16");
    if (g_cuda_chain) {
        kv_mode(&m, 0); run(&m, NULL, n, Lc32, Tc, NULL);
        kv_mode(&m, 1);
        int f0 = fake_chain_frames, r0 = fake_chain_f16_ops;
        run(&m, NULL, n, Lc, Tc, NULL);
        tk(fake_chain_frames > f0 + n && fake_chain_f16_ops > r0,
           "f16: steps ran through chain with CC_EW_F16 rounding");
        int upto = NP + n - 1;
        tk(m.kv_len == upto && mirror_is_host(&m, ch, upto),
           "host f16 K/V rows widened match mirror bits (and f32 IK)");
        q38cc_rollback(&m, -1, 0);
        int t = Tc[n - 1];
        free(step(&m, &t, 1, upto));
        tk(mirror_is_host(&m, ch, upto + 1),
           "mirror re-uploaded from host f16 cache after rewind: identical bits");
        double d32, d16; int a32, a16;
        diff(Lp32, Lc32, n, V, &d32, &a32);
        diff(Lp, Lc, n, V, &d16, &a16);
        printf("  kv16: chain vs per-matrix (%d steps): f32 max |dlogit| %.3g (%d/%d), f16 %.3g (%d/%d)\n",
               n, d32, a32, n, d16, a16, n);
        diff(Lc32, Lc, n, V, &d16, &a16);
        printf("  kv16: chain f16 vs f32: max |dlogit| %.3g, top-1 %d/%d\n", d16, a16, n);
        char msg[200];
        snprintf(msg, sizeof msg, "chain f16 within %g of chain f32 (%.3g), all top-1 kept", KV16_MAX_ABS, d16);
        tk(d16 <= KV16_MAX_ABS && a16 == n, msg);
    }
    g_cuda_chain = 0;
    fake_chain_table.size = size0; fake_chain_table.kv16 = NULL; fake_chain_f16 = NULL;
    free(Lp); free(Lc); free(Lp32); free(Lc32);
    g_q38_kv16 = -1;
    qt_shutdown();
    q38_model_free(&m);
}

int main(void) {
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("Q38_MTP", "0", 1);
    t_kernels_bit_identical();
    t_cpu();
    t_chain();
    if (t_fails) { printf("test_qwen38_kv16: %d failure(s)\n", t_fails); return 1; }
    printf("OK test_qwen38_kv16: host float16 K/V bank bit-identical to f16-rounded kernels, half RAM/pos\n");
    return 0;
}
