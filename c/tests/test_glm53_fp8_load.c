/* Self-contained CPU unit test for GLM-5.3 native FP8 e4m3 checkpoint loading.
 *
 * Covers:
 *   1. glm53_load_fp8_f32() multi-tile (non-multiple-of-128) dequantization
 *      against e4m3_decode(byte) * scale[r/128][c/128], plus refusal paths
 *      (missing _scale_inv, non-F32 scale, wrong scale count, non-2D weight).
 *   2. End-to-end tiny GLM-5.3 fixture (1 KDA+dense layer, 1 DSA/MLA+MoE layer)
 *      loaded as F32 reference vs native F8_E4M3 + _scale_inv, verifying exact
 *      teacher-forcing tokens, greedy decode tokens, and bit-exact logits. */
#define GLM53_NO_MAIN
#include "../glm53.c"

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#define test_mkdir(p) _mkdir(p)
#define test_rmdir(p) _rmdir(p)
#define test_pid()    ((long)_getpid())
#else
#include <sys/stat.h>
#include <unistd.h>
#define test_mkdir(p) mkdir((p), 0700)
#define test_rmdir(p) rmdir(p)
#define test_pid()    ((long)getpid())
#endif

static int g_checks = 0;
static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }
    g_checks++;
}

/* --- Minimal safetensors writer (F32 / F8_E4M3) --- */
typedef struct {
    char    *name;
    char     dtype[16];
    int      rank;
    int64_t  shape[4];
    uint8_t *data;
    size_t   nbytes;
} StEntry;

typedef struct {
    StEntry *items;
    int      n;
    int      cap;
} StBuilder;

static void sb_init(StBuilder *b) { memset(b, 0, sizeof(*b)); }

static void sb_free(StBuilder *b) {
    for (int i = 0; i < b->n; i++) { free(b->items[i].name); free(b->items[i].data); }
    free(b->items);
    memset(b, 0, sizeof(*b));
}

static void sb_add(StBuilder *b, const char *name, const char *dtype,
                   int rank, const int64_t *shape, const void *data, size_t nbytes) {
    if (b->n == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 64;
        b->items = realloc(b->items, (size_t)b->cap * sizeof(*b->items));
        check(b->items != NULL, "StBuilder realloc");
    }
    StEntry *e = &b->items[b->n++];
    memset(e, 0, sizeof(*e));
    e->name = strdup(name);
    check(e->name != NULL, "StEntry strdup");
    snprintf(e->dtype, sizeof(e->dtype), "%s", dtype);
    e->rank = rank;
    for (int i = 0; i < rank; i++) e->shape[i] = shape[i];
    e->data = malloc(nbytes ? nbytes : 1);
    check(e->data != NULL, "StEntry malloc");
    if (nbytes) memcpy(e->data, data, nbytes);
    e->nbytes = nbytes;
}

static void sb_write(const StBuilder *b, const char *path) {
    size_t hcap = 4096, hlen = 0;
    char *hdr = malloc(hcap);
    check(hdr != NULL, "header malloc");
    hdr[hlen++] = '{';
    uint64_t off = 0;
    for (int i = 0; i < b->n; i++) {
        const StEntry *e = &b->items[i];
        char entry[1024];
        int p = snprintf(entry, sizeof(entry), "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                         i ? "," : "", e->name, e->dtype);
        for (int d = 0; d < e->rank; d++)
            p += snprintf(entry + p, sizeof(entry) - (size_t)p, "%s%lld",
                          d ? "," : "", (long long)e->shape[d]);
        uint64_t end = off + (uint64_t)e->nbytes;
        p += snprintf(entry + p, sizeof(entry) - (size_t)p,
                      "],\"data_offsets\":[%llu,%llu]}",
                      (unsigned long long)off, (unsigned long long)end);
        off = end;
        while (hlen + (size_t)p + 2 >= hcap) {
            hcap *= 2;
            hdr = realloc(hdr, hcap);
            check(hdr != NULL, "header realloc");
        }
        memcpy(hdr + hlen, entry, (size_t)p);
        hlen += (size_t)p;
    }
    hdr[hlen++] = '}';
    FILE *f = fopen(path, "wb");
    check(f != NULL, "open safetensors for write");
    uint64_t hlen64 = (uint64_t)hlen;
    check(fwrite(&hlen64, 8, 1, f) == 1, "write header len");
    check(fwrite(hdr, 1, hlen, f) == hlen, "write header json");
    for (int i = 0; i < b->n; i++) {
        if (b->items[i].nbytes)
            check(fwrite(b->items[i].data, 1, b->items[i].nbytes, f) == b->items[i].nbytes,
                  "write tensor data");
    }
    fclose(f);
    free(hdr);
}

/* --- Deterministic PRNG & valid FP8 e4m3 generator --- */
static uint32_t rng_next(uint32_t *st) {
    *st = (*st) * 1664525u + 1013904223u;
    return *st;
}

static float rng_f32(uint32_t *st, float lo, float hi) {
    float u = (float)(rng_next(st) >> 8) * (1.0f / 16777216.0f);
    return lo + u * (hi - lo);
}

/* Generates a valid finite FP8 e4m3 byte (excludes 0x7f and 0xff NaNs, keeps magnitude modest). */
static uint8_t rng_e4m3(uint32_t *st) {
    uint32_t v = rng_next(st);
    uint8_t sign = (uint8_t)((v & 1u) << 7);
    uint8_t exp  = (uint8_t)(2u + ((v >> 1) % 7u));  /* exp 2..8: finite, non-extreme */
    uint8_t mant = (uint8_t)((v >> 8) & 7u);
    return (uint8_t)(sign | (exp << 3) | mant);
}

static void make_temp_dir(char *out, size_t cap, const char *tag) {
    const char *base = getenv("TMPDIR");
#ifdef _WIN32
    if (!base || !*base) base = getenv("TEMP");
    if (!base || !*base) base = getenv("TMP");
    if (!base || !*base) base = ".";
#else
    if (!base || !*base) base = "/tmp";
#endif
    snprintf(out, cap, "%s/%s_%ld", base, tag, test_pid());
    test_mkdir(out);
}

static void remove_fixture_dir(const char *dir) {
    char p[512];
    snprintf(p, sizeof(p), "%s/config.json", dir);       remove(p);
    snprintf(p, sizeof(p), "%s/model.safetensors", dir); remove(p);
    test_rmdir(dir);
}

/* --- Part 1: Direct glm53_load_fp8_f32 unit & refusal tests --- */
static void test_fp8_dequant_direct(void) {
    char dir[512], st_path[600];
    make_temp_dir(dir, sizeof(dir), "glm53_fp8_unit");
    snprintf(st_path, sizeof(st_path), "%s/model.safetensors", dir);

    const int64_t R = 130, C = 135; /* spans 2 x 2 tiles of 128x128 */
    const int64_t nr = fp8_nblk((int)R), nc = fp8_nblk((int)C);
    check(nr == 2 && nc == 2, "130x135 spans 2x2 FP8 tiles");

    uint8_t *w = malloc((size_t)(R * C));
    check(w != NULL, "alloc w");
    uint32_t rng = 0x12345678u;
    for (int64_t i = 0; i < R * C; i++) w[i] = rng_e4m3(&rng);
    float scales[4] = { 0.25f, 0.75f, 1.5f, 2.25f };

    StBuilder sb;
    sb_init(&sb);
    {
        int64_t sh[2] = { R, C }, ssh[2] = { nr, nc };
        sb_add(&sb, "ok.weight", "F8_E4M3", 2, sh, w, (size_t)(R * C));
        sb_add(&sb, "ok.weight_scale_inv", "F32", 2, ssh, scales, sizeof(scales));

        /* Error case 1: missing scale */
        sb_add(&sb, "no_scale.weight", "F8_E4M3", 2, sh, w, (size_t)(R * C));

        /* Error case 2: wrong scale dtype (F8_E4M3 instead of F32) */
        uint8_t bad_sc[4] = { 0x38, 0x38, 0x38, 0x38 };
        sb_add(&sb, "bad_dtype.weight", "F8_E4M3", 2, sh, w, (size_t)(R * C));
        sb_add(&sb, "bad_dtype.weight_scale_inv", "F8_E4M3", 2, ssh, bad_sc, sizeof(bad_sc));

        /* Error case 3: wrong scale count (1 instead of 4) */
        int64_t ssh1[2] = { 1, 1 };
        sb_add(&sb, "bad_cnt.weight", "F8_E4M3", 2, sh, w, (size_t)(R * C));
        sb_add(&sb, "bad_cnt.weight_scale_inv", "F32", 2, ssh1, scales, sizeof(float));

        /* Error case 4: wrong rank (1D) */
        int64_t sh1[1] = { R * C };
        sb_add(&sb, "bad_rank.weight", "F8_E4M3", 1, sh1, w, (size_t)(R * C));
        sb_add(&sb, "bad_rank.weight_scale_inv", "F32", 2, ssh, scales, sizeof(scales));
    }
    sb_write(&sb, st_path);
    sb_free(&sb);

    GModel m;
    memset(&m, 0, sizeof(m));
    st_init(&m.S, dir);

    st_tensor *t_ok = st_find(&m.S, "ok.weight");
    check(t_ok && t_ok->dtype == 4, "F8_E4M3 parsed as dtype=4");
    float *got = glm53_load_fp8_f32(&m, "ok.weight", t_ok);
    check(got != NULL, "glm53_load_fp8_f32 succeeded on 130x135");
    int mismatches = 0;
    for (int64_t r = 0; r < R; r++) {
        for (int64_t c = 0; c < C; c++) {
            float ref = e4m3_decode(w[r * C + c]) * scales[(r / FP8_BLOCK) * nc + (c / FP8_BLOCK)];
            if (got[r * C + c] != ref) mismatches++;
        }
    }
    check(mismatches == 0, "130x135 multi-tile FP8 dequant is bit-exact");
    free(got);

    check(glm53_load_fp8_f32(&m, "no_scale.weight", st_find(&m.S, "no_scale.weight")) == NULL,
          "missing _scale_inv rejected");
    check(glm53_load_fp8_f32(&m, "bad_dtype.weight", st_find(&m.S, "bad_dtype.weight")) == NULL,
          "non-F32 _scale_inv rejected");
    check(glm53_load_fp8_f32(&m, "bad_cnt.weight", st_find(&m.S, "bad_cnt.weight")) == NULL,
          "mismatched _scale_inv count rejected");
    check(glm53_load_fp8_f32(&m, "bad_rank.weight", st_find(&m.S, "bad_rank.weight")) == NULL,
          "non-2D FP8 weight rejected");

    st_destroy(&m.S);
    free(w);
    remove_fixture_dir(dir);
}

/* --- Part 2: End-to-end tiny GLM-5.3 F32 vs FP8 fixture comparison --- */
static void add_f32_tensor(StBuilder *ref, StBuilder *fp8, const char *name,
                           int rank, const int64_t *shape, uint32_t *rng, float lo, float hi) {
    int64_t n = 1;
    for (int i = 0; i < rank; i++) n *= shape[i];
    float *buf = malloc((size_t)n * sizeof(float));
    check(buf != NULL, "alloc f32 tensor");
    for (int64_t i = 0; i < n; i++) buf[i] = rng_f32(rng, lo, hi);
    sb_add(ref, name, "F32", rank, shape, buf, (size_t)n * sizeof(float));
    sb_add(fp8, name, "F32", rank, shape, buf, (size_t)n * sizeof(float));
    free(buf);
}

static void add_mat_pair(StBuilder *ref, StBuilder *fp8, const char *name,
                         int64_t rows, int64_t cols, uint32_t *rng, int as_fp8) {
    int64_t sh[2] = { rows, cols };
    if (!as_fp8) {
        add_f32_tensor(ref, fp8, name, 2, sh, rng, -0.05f, 0.05f);
        return;
    }
    int64_t nr = fp8_nblk((int)rows), nc = fp8_nblk((int)cols);
    uint8_t *w = malloc((size_t)(rows * cols));
    float *sc = malloc((size_t)(nr * nc) * sizeof(float));
    float *dq = malloc((size_t)(rows * cols) * sizeof(float));
    check(w && sc && dq, "alloc fp8 mat pair");
    for (int64_t i = 0; i < nr * nc; i++) sc[i] = rng_f32(rng, 0.005f, 0.02f);
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            uint8_t b = rng_e4m3(rng);
            w[r * cols + c] = b;
            dq[r * cols + c] = e4m3_decode(b) * sc[(r / FP8_BLOCK) * nc + (c / FP8_BLOCK)];
        }
    }
    sb_add(ref, name, "F32", 2, sh, dq, (size_t)(rows * cols) * sizeof(float));
    sb_add(fp8, name, "F8_E4M3", 2, sh, w, (size_t)(rows * cols));
    char sc_name[544];
    snprintf(sc_name, sizeof(sc_name), "%s_scale_inv", name);
    int64_t ssh[2] = { nr, nc };
    sb_add(fp8, sc_name, "F32", 2, ssh, sc, (size_t)(nr * nc) * sizeof(float));
    free(w); free(sc); free(dq);
}

static void write_tiny_config(const char *path, int fp8_cfg) {
    FILE *f = fopen(path, "wb");
    check(f != NULL, "open config.json");
    fprintf(f,
        "{\n"
        "  \"architectures\": [\"Glm5NextForConditionalGeneration\"],\n"
        "  \"model_type\": \"glm5_next\",\n"
        "  \"torch_dtype\": \"float32\",\n"
        "  \"text_config\": {\n"
        "    \"model_type\": \"glm5_next_text\",\n"
        "    \"vocab_size\": 32,\n"
        "    \"hidden_size\": 128,\n"
        "    \"intermediate_size\": 128,\n"
        "    \"moe_intermediate_size\": 128,\n"
        "    \"num_hidden_layers\": 2,\n"
        "    \"num_attention_heads\": 2,\n"
        "    \"num_key_value_heads\": 2,\n"
        "    \"n_shared_experts\": 1,\n"
        "    \"n_routed_experts\": 2,\n"
        "    \"num_experts_per_tok\": 1,\n"
        "    \"kv_lora_rank\": 64,\n"
        "    \"q_lora_rank\": 64,\n"
        "    \"qk_rope_head_dim\": 0,\n"
        "    \"qk_nope_head_dim\": 32,\n"
        "    \"v_head_dim\": 32,\n"
        "    \"max_position_embeddings\": 64,\n"
        "    \"layer_types\": [\"linear_attention\", \"deepseek_sparse_attention\"],\n"
        "    \"mlp_layer_types\": [\"dense\", \"sparse\"],\n"
        "    \"indexer_types\": [\"full\", \"full\"],\n"
        "    \"index_topk\": 4,\n"
        "    \"index_kpool\": 2,\n"
        "    \"index_head_dim\": 32,\n"
        "    \"index_n_heads\": 2,\n"
        "    \"linear_head_dim\": 32,\n"
        "    \"linear_num_heads\": 2,\n"
        "    \"linear_conv_kernel_dim\": 4,\n"
        "    \"linear_attn_config\": {\n"
        "      \"num_heads\": 2,\n"
        "      \"head_dim\": 32,\n"
        "      \"short_conv_kernel_size\": 4,\n"
        "      \"gate_lower_bound\": -5.0,\n"
        "      \"kda_layers\": [0]\n"
        "    },\n"
        "    \"hc_mult\": 2,\n"
        "    \"hc_sinkhorn_iters\": 3,\n"
        "    \"hc_eps\": 1e-6,\n"
        "    \"rms_norm_eps\": 1e-5,\n"
        "    \"routed_scaling_factor\": 2.5,\n"
        "    \"swiglu_limit\": 10.0,\n"
        "    \"num_nextn_predict_layers\": 0,\n"
        "    \"bos_token_id\": 0,\n"
        "    \"eos_token_id\": 1,\n"
        "    \"tie_word_embeddings\": false\n"
        "  }%s\n"
        "}\n",
        fp8_cfg
            ? ",\n  \"quantization_config\": {\"quant_method\": \"fp8\", \"fmt\": \"e4m3\", \"activation_scheme\": \"dynamic\", \"weight_block_size\": [128, 128]}"
            : "");
    fclose(f);
}

static int argmax_f32(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

static void test_tiny_e2e_fp8_exact(void) {
    char ref_dir[512], fp8_dir[512], p[600];
    make_temp_dir(ref_dir, sizeof(ref_dir), "glm53_ref_e2e");
    make_temp_dir(fp8_dir, sizeof(fp8_dir), "glm53_fp8_e2e");

    snprintf(p, sizeof(p), "%s/config.json", ref_dir); write_tiny_config(p, 0);
    snprintf(p, sizeof(p), "%s/config.json", fp8_dir); write_tiny_config(p, 1);

    const char *P = "model.language_model.";
    const int64_t V = 32, D = 128, I = 128, M = 128, H = 2, QK = 32, VH = 32;
    const int64_t KVL = 64, QL = 64, IH = 2, IHD = 32, KP = 2;
    const int64_t KH = 2, KHD = 32, KPROJ = 64, CK = 4, HC = 2, HCM = (2 + HC) * HC;
    uint32_t rng = 0x5EED1243u;

    StBuilder ref, fp8;
    sb_init(&ref);
    sb_init(&fp8);

    char nm[256];
    {
        int64_t sh_e[2] = { V, D }, sh_d[1] = { D };
        snprintf(nm, sizeof(nm), "%sembed_tokens.weight", P);
        add_f32_tensor(&ref, &fp8, nm, 2, sh_e, &rng, -0.2f, 0.2f);
        snprintf(nm, sizeof(nm), "%snorm.weight", P);
        add_f32_tensor(&ref, &fp8, nm, 1, sh_d, &rng, 0.8f, 1.2f);
        add_mat_pair(&ref, &fp8, "lm_head.weight", V, D, &rng, 1);
    }

    for (int l = 0; l < 2; l++) {
        int64_t sh_d[1] = { D };
        int64_t sh_hfn[2] = { HCM, HC * D }, sh_hb[1] = { HCM }, sh_hs[1] = { 3 };
        snprintf(nm, sizeof(nm), "%slayers.%d.input_layernorm.weight", P, l);
        add_f32_tensor(&ref, &fp8, nm, 1, sh_d, &rng, 0.8f, 1.2f);
        snprintf(nm, sizeof(nm), "%slayers.%d.post_attention_layernorm.weight", P, l);
        add_f32_tensor(&ref, &fp8, nm, 1, sh_d, &rng, 0.8f, 1.2f);
        snprintf(nm, sizeof(nm), "%slayers.%d.hc_attn_fn", P, l);
        add_f32_tensor(&ref, &fp8, nm, 2, sh_hfn, &rng, -0.05f, 0.05f);
        snprintf(nm, sizeof(nm), "%slayers.%d.hc_attn_base", P, l);
        add_f32_tensor(&ref, &fp8, nm, 1, sh_hb, &rng, 0.1f, 0.5f);
        snprintf(nm, sizeof(nm), "%slayers.%d.hc_attn_scale", P, l);
        add_f32_tensor(&ref, &fp8, nm, 1, sh_hs, &rng, 0.05f, 0.2f);
        snprintf(nm, sizeof(nm), "%slayers.%d.hc_ffn_fn", P, l);
        add_f32_tensor(&ref, &fp8, nm, 2, sh_hfn, &rng, -0.05f, 0.05f);
        snprintf(nm, sizeof(nm), "%slayers.%d.hc_ffn_base", P, l);
        add_f32_tensor(&ref, &fp8, nm, 1, sh_hb, &rng, 0.1f, 0.5f);
        snprintf(nm, sizeof(nm), "%slayers.%d.hc_ffn_scale", P, l);
        add_f32_tensor(&ref, &fp8, nm, 1, sh_hs, &rng, 0.05f, 0.2f);

        if (l == 0) {
            /* KDA linear attention + dense MLP */
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.q_proj.weight", P); add_mat_pair(&ref, &fp8, nm, KPROJ, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.k_proj.weight", P); add_mat_pair(&ref, &fp8, nm, KPROJ, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.v_proj.weight", P); add_mat_pair(&ref, &fp8, nm, KPROJ, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.o_proj.weight", P); add_mat_pair(&ref, &fp8, nm, D, KPROJ, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.g_a_proj.weight", P); add_mat_pair(&ref, &fp8, nm, KHD, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.g_b_proj.weight", P); add_mat_pair(&ref, &fp8, nm, KPROJ, KHD, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.f_a_proj.weight", P); add_mat_pair(&ref, &fp8, nm, KHD, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.f_b_proj.weight", P); add_mat_pair(&ref, &fp8, nm, KPROJ, KHD, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.b_proj.weight", P); add_mat_pair(&ref, &fp8, nm, KH, D, &rng, 1);
            int64_t sh_kp[1] = { KPROJ }, sh_kh[1] = { KH }, sh_khd[1] = { KHD }, sh_cv[2] = { KPROJ, CK };
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.dt_bias", P); add_f32_tensor(&ref, &fp8, nm, 1, sh_kp, &rng, -0.1f, 0.1f);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.A_log", P); add_f32_tensor(&ref, &fp8, nm, 1, sh_kh, &rng, -0.5f, 0.5f);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.o_norm.weight", P); add_f32_tensor(&ref, &fp8, nm, 1, sh_khd, &rng, 0.8f, 1.2f);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.q_conv1d.weight", P); add_f32_tensor(&ref, &fp8, nm, 2, sh_cv, &rng, -0.1f, 0.1f);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.k_conv1d.weight", P); add_f32_tensor(&ref, &fp8, nm, 2, sh_cv, &rng, -0.1f, 0.1f);
            snprintf(nm, sizeof(nm), "%slayers.0.self_attn.v_conv1d.weight", P); add_f32_tensor(&ref, &fp8, nm, 2, sh_cv, &rng, -0.1f, 0.1f);
            snprintf(nm, sizeof(nm), "%slayers.0.mlp.gate_proj.weight", P); add_mat_pair(&ref, &fp8, nm, I, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.mlp.up_proj.weight", P); add_mat_pair(&ref, &fp8, nm, I, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.0.mlp.down_proj.weight", P); add_mat_pair(&ref, &fp8, nm, D, I, &rng, 1);
        } else {
            /* DSA/MLA full attention + MoE (shared + 2 routed experts, all FP8) */
            int64_t sh_ql[1] = { QL }, sh_kvl[1] = { KVL }, sh_ihd[1] = { IHD };
            int64_t sh_kvb[2] = { H * (QK + VH), KVL }, sh_ape[2] = { KP, IHD };
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.q_a_proj.weight", P); add_mat_pair(&ref, &fp8, nm, QL, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.q_a_layernorm.weight", P); add_f32_tensor(&ref, &fp8, nm, 1, sh_ql, &rng, 0.8f, 1.2f);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.q_b_proj.weight", P); add_mat_pair(&ref, &fp8, nm, H * QK, QL, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.kv_a_proj_with_mqa.weight", P); add_mat_pair(&ref, &fp8, nm, KVL, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.kv_a_layernorm.weight", P); add_f32_tensor(&ref, &fp8, nm, 1, sh_kvl, &rng, 0.8f, 1.2f);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.kv_b_proj.weight", P); add_f32_tensor(&ref, &fp8, nm, 2, sh_kvb, &rng, -0.05f, 0.05f);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.o_proj.weight", P); add_mat_pair(&ref, &fp8, nm, D, H * VH, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.indexer.wq_b.weight", P); add_mat_pair(&ref, &fp8, nm, IH * IHD, QL, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.indexer.wk.weight", P); add_mat_pair(&ref, &fp8, nm, IHD, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.indexer.weights_proj.weight", P); add_mat_pair(&ref, &fp8, nm, IH, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.indexer.k_norm.weight", P); add_f32_tensor(&ref, &fp8, nm, 1, sh_ihd, &rng, 0.8f, 1.2f);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.indexer.k_norm.bias", P); add_f32_tensor(&ref, &fp8, nm, 1, sh_ihd, &rng, -0.05f, 0.05f);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.indexer.index_kpool_compress_ape", P); add_f32_tensor(&ref, &fp8, nm, 2, sh_ape, &rng, -0.05f, 0.05f);
            snprintf(nm, sizeof(nm), "%slayers.1.self_attn.indexer.index_kpool_compress_gate", P); add_mat_pair(&ref, &fp8, nm, IHD, D, &rng, 1);

            int64_t sh_r[2] = { 2, D }, sh_rb[1] = { 2 };
            snprintf(nm, sizeof(nm), "%slayers.1.mlp.gate.weight", P); add_f32_tensor(&ref, &fp8, nm, 2, sh_r, &rng, -0.1f, 0.1f);
            snprintf(nm, sizeof(nm), "%slayers.1.mlp.gate.e_score_correction_bias", P); add_f32_tensor(&ref, &fp8, nm, 1, sh_rb, &rng, -0.05f, 0.05f);
            snprintf(nm, sizeof(nm), "%slayers.1.mlp.shared_experts.gate_proj.weight", P); add_mat_pair(&ref, &fp8, nm, M, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.1.mlp.shared_experts.up_proj.weight", P); add_mat_pair(&ref, &fp8, nm, M, D, &rng, 1);
            snprintf(nm, sizeof(nm), "%slayers.1.mlp.shared_experts.down_proj.weight", P); add_mat_pair(&ref, &fp8, nm, D, M, &rng, 1);
            for (int e = 0; e < 2; e++) {
                snprintf(nm, sizeof(nm), "%slayers.1.mlp.experts.%d.gate_proj.weight", P, e); add_mat_pair(&ref, &fp8, nm, M, D, &rng, 1);
                snprintf(nm, sizeof(nm), "%slayers.1.mlp.experts.%d.up_proj.weight", P, e); add_mat_pair(&ref, &fp8, nm, M, D, &rng, 1);
                snprintf(nm, sizeof(nm), "%slayers.1.mlp.experts.%d.down_proj.weight", P, e); add_mat_pair(&ref, &fp8, nm, D, M, &rng, 1);
            }
        }
    }

    snprintf(p, sizeof(p), "%s/model.safetensors", ref_dir); sb_write(&ref, p);
    snprintf(p, sizeof(p), "%s/model.safetensors", fp8_dir); sb_write(&fp8, p);
    sb_free(&ref);
    sb_free(&fp8);

    GModel m_ref, m_fp8;
    memset(&m_ref, 0, sizeof(m_ref));
    memset(&m_fp8, 0, sizeof(m_fp8));
    model_load(&m_ref, ref_dir);
    model_load(&m_fp8, fp8_dir);
    check(m_fp8.streaming == 0, "FP8 routed experts load resident via load_mat");

    GSession *s_ref = session_open(&m_ref, 32);
    GSession *s_fp8 = session_open(&m_fp8, 32);

    const int prompt[4] = { 1, 5, 12, 3 };
    float *l_ref = forward_span(&m_ref, s_ref, prompt, 4, NULL, 0);
    float *l_fp8 = forward_span(&m_fp8, s_fp8, prompt, 4, NULL, 0);
    check(l_ref != NULL && l_fp8 != NULL, "prefill forward_span succeeded");

    float worst = 0.0f;
    for (int t = 0; t < 4; t++) {
        const float *rrow = l_ref + (size_t)t * V;
        const float *frow = l_fp8 + (size_t)t * V;
        for (int v = 0; v < (int)V; v++) {
            check(isfinite(rrow[v]) && isfinite(frow[v]), "prefill logits finite");
            float d = fabsf(rrow[v] - frow[v]);
            if (d > worst) worst = d;
        }
        check(argmax_f32(rrow, (int)V) == argmax_f32(frow, (int)V),
              "teacher-forcing token matches F32 reference");
    }
    check(worst == 0.0f, "prefill logits bit-exact between F32 and FP8 e4m3");

    int next_tok = argmax_f32(l_fp8 + 3 * V, (int)V);
    free(l_ref);
    free(l_fp8);

    for (int step = 0; step < 4; step++) {
        l_ref = forward_span(&m_ref, s_ref, &next_tok, 1, NULL, 0);
        l_fp8 = forward_span(&m_fp8, s_fp8, &next_tok, 1, NULL, 0);
        check(l_ref != NULL && l_fp8 != NULL, "decode forward_span succeeded");
        for (int v = 0; v < (int)V; v++) {
            check(isfinite(l_ref[v]) && isfinite(l_fp8[v]), "decode logits finite");
            float d = fabsf(l_ref[v] - l_fp8[v]);
            if (d > worst) worst = d;
        }
        int tok_ref = argmax_f32(l_ref, (int)V);
        int tok_fp8 = argmax_f32(l_fp8, (int)V);
        check(tok_ref == tok_fp8, "greedy decode token matches F32 reference");
        next_tok = tok_fp8;
        free(l_ref);
        free(l_fp8);
    }
    check(worst == 0.0f, "greedy decode logits bit-exact between F32 and FP8 e4m3");

    session_close(&m_ref, s_ref);
    session_close(&m_fp8, s_fp8);
    model_release(&m_ref);
    model_release(&m_fp8);
    remove_fixture_dir(ref_dir);
    remove_fixture_dir(fp8_dir);
}

int main(void) {
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
#ifdef _WIN32
    _putenv_s("GLM53_BITS", "32");
#else
    setenv("GLM53_BITS", "32", 1);
#endif
    test_fp8_dequant_direct();
    test_tiny_e2e_fp8_exact();
    printf("glm53 FP8 e4m3 load: %d checks passed\n", g_checks);
    return 0;
}
