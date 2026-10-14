/* mimo_vision.h -- the MiMo ViT (681M on the release), images only.
 *
 * Included by mimo.c after the Model and DW helpers. Mirrors
 * MiMoVisionTransformer in modeling_mimo_v2.py, with the two points where the
 * release and that module disagree resolved as the release (and vLLM's
 * mimo_v2_omni.py) has them:
 *   - the merger stores no biases: its two Linears and its LayerNorm bias are
 *     zero (vLLM builds them with bias=False; a bias present in a container is
 *     still used);
 *   - after a run of column-order blocks that reaches the end of the tower, the
 *     tokens go back to row order before the merger (vLLM does; the release's
 *     last block is a full-attention one, so the two readings never differ on it).
 *
 * The tower, per block:  x += proj(attn(rms(x))),  x += mlp(rms(x))
 *   patches     [N, 3*T*16*16] in 2x2-merge-block order (the Qwen2-VL processor)
 *   2D RoPE     head_dim/2 dims: h then w frequencies (theta 1e4), cat(e, e),
 *               rotate-half over the whole head
 *   attention   GQA; full over the image in fullatt_block_indexes, else a 1D
 *               window |i - j| <= visual_token_window_size in the CURRENT token
 *               order, plus a sink logit added to key 0 of that order
 *   order       vit_window_attn_types: -1 full, 0 row order, 1 column order of
 *               the merge units (entering a type-1 run permutes the tokens and
 *               their rotary angles, leaving it permutes them back)
 *   merger      LayerNorm, the 2x2 units concatenated, Linear, exact GELU, Linear
 */

typedef struct {
    float *norm1, *norm2, *sinks;
    DW qkv, proj, gate, up, down;
    float *qkv_b, *proj_b, *gate_b, *up_b, *down_b;
    int full, type;
} VBlock;

struct Vision {
    int depth, d, inter, heads, kvh, hd, out, patch, temporal, merge, in_ch, window;
    float eps;
    DW embed;                   /* [d, in_ch*T*P*P] */
    VBlock *b;
    float *ln_w, *ln_b, *fc1_b, *fc2_b;
    DW fc1, fc2;
    int patch_in;
};

static void vis_mat(DW *d, shards *S, const char *name, int O, int I) {
    /* the tower stays BF16 (or f32 for the oracle): int8 is a dense-trunk option */
    int saved = g_dense_bits;
    g_dense_bits = g_dense_bits == 32 ? 32 : 0;
    dw_load_bf16(d, S, name, O, I);
    g_dense_bits = saved;
}

static float *vis_vec_opt(shards *S, const char *name, int n) {
    return st_has(S, name) ? load_vec(S, name, n) : NULL;
}

static int vjson_int(jval *o, const char *key, int fallback) {
    jval *v = o ? json_get(o, key) : NULL;
    return (v && v->t == J_NUM) ? (int)v->num : fallback;
}

static Vision *vision_load(Model *m, const char *dir) {
    if (!st_has(&m->S, "visual.patch_embed.proj.weight")) return NULL;
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    char *text = read_text(path, NULL);
    if (!text) return NULL;
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    jval *vc = root ? json_get(root, "vision_config") : NULL;
    if (!vc || vc->t != J_OBJ) {
        fprintf(stderr, "[mimo] visual.* tensors but no vision_config: text only\n");
        json_free(root); free(arena); free(text);
        return NULL;
    }
    Vision *v = xcalloc(1, sizeof(Vision), "vision");
    v->depth = vjson_int(vc, "depth", 0);
    v->d = vjson_int(vc, "hidden_size", 0);
    v->inter = vjson_int(vc, "intermediate_size", 0);
    v->heads = vjson_int(vc, "num_heads", 0);
    v->kvh = vjson_int(vc, "num_key_value_heads", v->heads);
    v->hd = vjson_int(vc, "qk_channels", 64);
    v->out = vjson_int(vc, "out_hidden_size", 0);
    v->patch = vjson_int(vc, "patch_size", 16);
    v->temporal = vjson_int(vc, "temporal_patch_size", 2);
    v->merge = vjson_int(vc, "spatial_merge_size", 2);
    v->in_ch = vjson_int(vc, "in_chans", 3);
    v->window = vjson_int(vc, "visual_token_window_size", -1);
    v->eps = (float)jnum(vc, "rms_norm_eps", 1e-6);
    int use_sink = jbool(vc, "use_sink", 0);
    if (v->depth < 1 || v->depth > 256 || v->d < 8 || v->d > 16384 || v->inter < 1 || v->inter > 65536 ||
        v->heads < 1 || v->kvh < 1 || v->heads % v->kvh || v->hd < 4 || v->hd % 4 || v->hd > 1024 ||
        v->out != m->c.hidden || v->merge != 2 || v->patch < 1 || v->patch > 64 ||
        v->temporal < 1 || v->temporal > 4 || v->in_ch != 3) {
        fprintf(stderr, "[mimo] vision_config out of range (or out_hidden_size != hidden_size) -- refusing\n");
        exit(1);
    }
    v->patch_in = v->in_ch * v->temporal * v->patch * v->patch;
    v->b = xcalloc((size_t)v->depth, sizeof(VBlock), "vision blocks");
    jval *full = json_get(vc, "fullatt_block_indexes"), *types = json_get(vc, "vit_window_attn_types");
    for (int i = 0; i < v->depth; i++) v->b[i].type = -1;
    if (types && types->t == J_ARR) {
        if (types->len != v->depth) { fprintf(stderr, "[mimo] vit_window_attn_types must list every block\n"); exit(1); }
        for (int i = 0; i < v->depth; i++) v->b[i].type = (int)types->kids[i]->num;
    }
    if (full && full->t == J_ARR)
        for (int i = 0; i < full->len; i++) {
            int b = (int)full->kids[i]->num;
            if (b >= 0 && b < v->depth) v->b[b].full = 1;
        }
    json_free(root); free(arena); free(text);

    shards *S = &m->S;
    st_tensor *t = st_find(S, "visual.patch_embed.proj.weight");
    if (t->dtype != 0 || t->numel != (int64_t)v->d * v->patch_in || t->nbytes != t->numel * 2) {
        fprintf(stderr, "[mimo] visual.patch_embed.proj.weight is not BF16 [%d, %d] -- refusing\n", v->d, v->patch_in);
        exit(1);
    }
    /* Conv3d [d, C, T, P, P] with stride = kernel is a Linear over the flattened
     * patch, in the processor's C, T, row, col order */
    v->embed.O = v->d; v->embed.I = v->patch_in;
    if (g_dense_bits == 32) {
        float *w = xmalloc((size_t)v->d * v->patch_in * sizeof(float), "patch embed");
        st_read_f32(S, "visual.patch_embed.proj.weight", w, 1);
        v->embed.fmt = DW_F32; v->embed.w = w;
    } else {
        uint16_t *b16 = xmalloc((size_t)v->d * v->patch_in * sizeof(uint16_t), "patch embed");
        st_read_raw_cap(S, "visual.patch_embed.proj.weight", b16, (int64_t)v->d * v->patch_in * 2, 1);
        v->embed.fmt = DW_BF16; v->embed.w = b16;
    }
    int qkv_rows = (v->heads + 2 * v->kvh) * v->hd;
    for (int i = 0; i < v->depth; i++) {
        VBlock *b = &v->b[i];
        char nm[512];
#define VN(fmt_) (snprintf(nm, sizeof(nm), "visual.blocks.%d." fmt_, i), nm)
        b->norm1 = load_vec(S, VN("norm1.weight"), v->d);
        b->norm2 = load_vec(S, VN("norm2.weight"), v->d);
        vis_mat(&b->qkv, S, VN("attn.qkv.weight"), qkv_rows, v->d);
        b->qkv_b = load_vec(S, VN("attn.qkv.bias"), qkv_rows);
        vis_mat(&b->proj, S, VN("attn.proj.weight"), v->d, v->heads * v->hd);
        b->proj_b = load_vec(S, VN("attn.proj.bias"), v->d);
        vis_mat(&b->gate, S, VN("mlp.gate_proj.weight"), v->inter, v->d);
        b->gate_b = load_vec(S, VN("mlp.gate_proj.bias"), v->inter);
        vis_mat(&b->up, S, VN("mlp.up_proj.weight"), v->inter, v->d);
        b->up_b = load_vec(S, VN("mlp.up_proj.bias"), v->inter);
        vis_mat(&b->down, S, VN("mlp.down_proj.weight"), v->d, v->inter);
        b->down_b = load_vec(S, VN("mlp.down_proj.bias"), v->d);
        if (use_sink && !b->full) b->sinks = load_vec(S, VN("attn.sinks"), v->heads);
#undef VN
    }
    int merged = v->d * v->merge * v->merge;
    v->ln_w = load_vec(S, "visual.merger.ln_q.weight", v->d);
    v->ln_b = vis_vec_opt(S, "visual.merger.ln_q.bias", v->d);
    vis_mat(&v->fc1, S, "visual.merger.mlp.0.weight", merged, merged);
    v->fc1_b = vis_vec_opt(S, "visual.merger.mlp.0.bias", merged);
    vis_mat(&v->fc2, S, "visual.merger.mlp.2.weight", v->out, merged);
    v->fc2_b = vis_vec_opt(S, "visual.merger.mlp.2.bias", v->out);
    return v;
}

static void add_bias(float *y, const float *b, int n, int dim) {
    if (!b) return;
    for (int t = 0; t < n; t++) for (int i = 0; i < dim; i++) y[(size_t)t * dim + i] += b[i];
}

/* new[unit k] = old[unit idx[k]], units of 4 tokens (the 2x2 merge) */
static void permute_units(float *x, const int *idx, int units, int width) {
    size_t unit = (size_t)4 * width;
    float *tmp = xmalloc((size_t)units * unit * sizeof(float), "vision permute");
    for (int k = 0; k < units; k++) memcpy(tmp + (size_t)k * unit, x + (size_t)idx[k] * unit, unit * sizeof(float));
    memcpy(x, tmp, (size_t)units * unit * sizeof(float));
    free(tmp);
}

static void vis_rope(float *x, int nh, int hd, const float *ang) {
    int half = hd / 2;
    for (int h = 0; h < nh; h++) {
        float *v = x + (size_t)h * hd;
        for (int i = 0; i < half; i++) {
            float cs = cosf(ang[i]), sn = sinf(ang[i]);
            float a = v[i], b = v[i + half];
            v[i] = a * cs - b * sn;
            v[i + half] = b * cs + a * sn;
        }
    }
}

static float *vision_run(Model *m, const float *patches, int gh, int gw, int *n_rows) {
    Vision *v = g_vision;
    (void)m;
    *n_rows = 0;
    if (!v || gh < 2 || gw < 2 || gh % 2 || gw % 2) return NULL;
    double t0 = now_s();
    int N = gh * gw, d = v->d, hd = v->hd, nh = v->heads, kvh = v->kvh, group = nh / kvh;
    int units = N / 4, uh = gh / 2, uw = gw / 2;
    float *x = xmalloc((size_t)N * d * sizeof(float), "vision tokens");
    dw_matmul(x, patches, N, &v->embed);
    /* rotary angles per token, in the processor's block-major order */
    int rdim = hd / 2, nf = rdim / 2;
    float *ang = xmalloc((size_t)N * rdim * sizeof(float), "vision angles");
    float *inv = xmalloc((size_t)nf * sizeof(float), "vision inv_freq");
    for (int j = 0; j < nf; j++) inv[j] = 1.0f / powf(10000.0f, (float)(2 * j) / (float)rdim);
    for (int n = 0; n < N; n++) {
        int u = n / 4, r = (n % 4) / 2, cc = n % 2;
        int hp = (u / uw) * 2 + r, wp = (u % uw) * 2 + cc;
        for (int j = 0; j < nf; j++) {
            ang[(size_t)n * rdim + j] = (float)hp * inv[j];
            ang[(size_t)n * rdim + nf + j] = (float)wp * inv[j];
        }
    }
    free(inv);
    /* column order of the merge units and its inverse */
    int *col = xmalloc((size_t)units * sizeof(int), "vision col order");
    int *rev = xmalloc((size_t)units * sizeof(int), "vision row order");
    for (int k = 0; k < units; k++) col[k] = (k % uh) * uw + k / uh;
    for (int k = 0; k < units; k++) rev[col[k]] = k;
    float *ang_col = xmalloc((size_t)N * rdim * sizeof(float), "vision col angles");
    memcpy(ang_col, ang, (size_t)N * rdim * sizeof(float));
    permute_units(ang_col, col, units, rdim);

    int qd = nh * hd, kd = kvh * hd, rows = qd + 2 * kd;
    float *xn = xmalloc((size_t)N * d * sizeof(float), "vision normed");
    float *qkv = xmalloc((size_t)N * rows * sizeof(float), "vision qkv");
    float *ctx = xmalloc((size_t)N * qd * sizeof(float), "vision attention");
    float *tmp = xmalloc((size_t)N * d * sizeof(float), "vision block out");
    float *g = xmalloc((size_t)N * v->inter * sizeof(float), "vision mlp gate");
    float *u = xmalloc((size_t)N * v->inter * sizeof(float), "vision mlp up");
    float scale = 1.0f / sqrtf((float)hd);
    int prev = -2;
    for (int bi = 0; bi < v->depth; bi++) {
        VBlock *b = &v->b[bi];
        if (b->type == 1 && prev != 1) permute_units(x, col, units, d);
        if (b->type != 1 && prev == 1) permute_units(x, rev, units, d);
        prev = b->type;
        const float *A = b->type == 1 ? ang_col : ang;
        for (int n = 0; n < N; n++) rmsnorm(xn + (size_t)n * d, x + (size_t)n * d, b->norm1, d, v->eps);
        dw_matmul(qkv, xn, N, &b->qkv);
        add_bias(qkv, b->qkv_b, N, rows);
        for (int n = 0; n < N; n++) {
            vis_rope(qkv + (size_t)n * rows, nh, hd, A + (size_t)n * rdim);
            vis_rope(qkv + (size_t)n * rows + qd, kvh, hd, A + (size_t)n * rdim);
        }
        int window = b->full ? -1 : v->window;
        #pragma omp parallel
        {
            float *score = xmalloc((size_t)N * sizeof(float), "vision scores");
            #pragma omp for collapse(2) schedule(static)
            for (int i = 0; i < N; i++) {
                for (int h = 0; h < nh; h++) {
                    int kh = h / group;
                    int lo = 0, hi = N - 1;
                    if (window >= 0) { lo = i - window > 0 ? i - window : 0; hi = i + window < N - 1 ? i + window : N - 1; }
                    const float *q = qkv + (size_t)i * rows + (size_t)h * hd;
                    float best = -INFINITY;
                    for (int j = lo; j <= hi; j++) {
                        const float *kr = qkv + (size_t)j * rows + qd + (size_t)kh * hd;
                        float s = 0;
                        for (int e = 0; e < hd; e++) s += q[e] * kr[e];
                        s *= scale;
                        if (j == 0 && b->sinks) s += b->sinks[h];
                        score[j - lo] = s;
                        if (s > best) best = s;
                    }
                    float den = 0;
                    for (int j = lo; j <= hi; j++) { score[j - lo] = expf(score[j - lo] - best); den += score[j - lo]; }
                    float *o = ctx + (size_t)i * qd + (size_t)h * hd;
                    for (int e = 0; e < hd; e++) o[e] = 0;
                    for (int j = lo; j <= hi; j++) {
                        const float *vr = qkv + (size_t)j * rows + qd + kd + (size_t)kh * hd;
                        float w = score[j - lo] / den;
                        for (int e = 0; e < hd; e++) o[e] += w * vr[e];
                    }
                }
            }
            free(score);
        }
        dw_matmul(tmp, ctx, N, &b->proj);
        add_bias(tmp, b->proj_b, N, d);
        for (size_t i = 0; i < (size_t)N * d; i++) x[i] += tmp[i];
        for (int n = 0; n < N; n++) rmsnorm(xn + (size_t)n * d, x + (size_t)n * d, b->norm2, d, v->eps);
        dw_matmul(g, xn, N, &b->gate);
        add_bias(g, b->gate_b, N, v->inter);
        dw_matmul(u, xn, N, &b->up);
        add_bias(u, b->up_b, N, v->inter);
        for (size_t i = 0; i < (size_t)N * v->inter; i++) g[i] = silu(g[i]) * u[i];
        dw_matmul(tmp, g, N, &b->down);
        add_bias(tmp, b->down_b, N, d);
        for (size_t i = 0; i < (size_t)N * d; i++) x[i] += tmp[i];
    }
    if (prev == 1) permute_units(x, rev, units, d);
    /* merger: LayerNorm per token, 2x2 units concatenated, Linear, GELU, Linear */
    for (int n = 0; n < N; n++) {
        float *r = x + (size_t)n * d;
        float mean = 0, var = 0;
        for (int i = 0; i < d; i++) mean += r[i];
        mean /= d;
        for (int i = 0; i < d; i++) { float z = r[i] - mean; var += z * z; }
        var /= d;
        float inv_std = 1.0f / sqrtf(var + 1e-6f);
        for (int i = 0; i < d; i++) r[i] = (r[i] - mean) * inv_std * v->ln_w[i] + (v->ln_b ? v->ln_b[i] : 0.0f);
    }
    int merged = 4 * d;
    float *h1 = xmalloc((size_t)units * merged * sizeof(float), "vision merger");
    dw_matmul(h1, x, units, &v->fc1);
    add_bias(h1, v->fc1_b, units, merged);
    for (size_t i = 0; i < (size_t)units * merged; i++) h1[i] = 0.5f * h1[i] * (1.0f + erff(h1[i] * 0.70710678118654752f));
    float *rows_out = xmalloc((size_t)units * v->out * sizeof(float), "image rows");
    dw_matmul(rows_out, h1, units, &v->fc2);
    add_bias(rows_out, v->fc2_b, units, v->out);
    free(h1); free(x); free(ang); free(ang_col); free(col); free(rev);
    free(xn); free(qkv); free(ctx); free(tmp); free(g); free(u);
    *n_rows = units;
    /* MIMO_VISION_DUMP=<file>: the rows the tower produced, f32 [units, hidden],
     * for tools/mimo_real_check.py to hold against Xiaomi's ViT */
    const char *dump = getenv("MIMO_VISION_DUMP");
    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (f) { fwrite(rows_out, sizeof(float), (size_t)units * v->out, f); fclose(f); }
    }
    if (env_int("MIMO_STATS", 0))
        fprintf(stderr, "[mimo] image %dx%d patches -> %d rows in %.2fs\n", gh, gw, units, now_s() - t0);
    return rows_out;
}
