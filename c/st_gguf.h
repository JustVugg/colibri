/* st_gguf.h -- a GGUF model as a safetensors index (st.h's shards).
 *
 * An engine that reads its checkpoint through st.h finds every tensor by the name the
 * original checkpoint gives it. For a folder tools/gguf_import.py wrote (gguf.json names the
 * GGUF), st_init_gguf opens the GGUF in place and lists its tensors under those names: the
 * inverse of llama.cpp's converter for the architecture (st_gguf_maps below). A fused expert
 * tensor [experts][rows][cols] becomes one entry per expert, each its contiguous slice of the
 * file, as the checkpoint has them. Shapes are row-major as safetensors gives them (ggml's ne
 * reversed). Dtypes: F32, F16, BF16, I32, I8 keep their st.h codes; a block-quantized tensor
 * (Q8_0, Q4_K, IQ2_XS, MXFP4, ...) gets ST_GGML_BASE + its ggml type, which the float readers
 * refuse and the engine computes with ggml_quants.h. No byte is copied or converted.
 *
 * The colibri container gguf_import --mode copy writes (colibri.layout expert-packed-v1: each
 * layer's routed experts in one tensor, an expert's gate | up | down in a 4 KiB-aligned slot)
 * reads the same way: every expert's three matrices are entries at their place in the slot.
 *
 * Opt-in: an engine defines ST_GGUF before including st.h; without it st.h is unchanged.
 * A GGUF tensor without a counterpart in the map keeps its GGUF name. */
#ifndef ST_GGUF_H
#define ST_GGUF_H

#include "gguf.h"
#include <limits.h>

#define ST_GGML_BASE 32
static inline int st_is_ggml(int dtype) { return dtype >= ST_GGML_BASE; }
static inline int st_ggml_type(int dtype) { return dtype - ST_GGML_BASE; }
/* the ggml type of any dtype the ggml_quants.h kernels compute (the float ones too), or -1 */
static inline int st_ggml_type_any(int dtype) {
    if (dtype >= ST_GGML_BASE) return dtype - ST_GGML_BASE;
    return dtype == 2 ? GQ_F32 : dtype == 1 ? GQ_F16 : dtype == 0 ? GQ_BF16 : -1;
}

typedef struct { const char *gg, *hf; } st_gg_name;
typedef struct {
    const char *arch;
    const char *layer_fmt;          /* the checkpoint's layer prefix, "%d" = the layer */
    const st_gg_name *globals;      /* whole names */
    const st_gg_name *layer;        /* after "blk.N." -> after the layer prefix; a "%d" in hf = per expert */
} st_gg_map;

/* DeepSeek V4 (deepseek4): conversion/deepseek.py, DeepseekV4Model._map_dsv4_tensor_name. */
static const st_gg_name st_gg_dsv4_glob[] = {
    {"token_embd.weight", "embed.weight"}, {"output_norm.weight", "norm.weight"},
    {"output.weight", "head.weight"}, {"output_hc_fn.weight", "hc_head_fn"},
    {"output_hc_base.weight", "hc_head_base"}, {"output_hc_scale.weight", "hc_head_scale"},
    {NULL, NULL}};
static const st_gg_name st_gg_dsv4_layer[] = {
    {"hc_attn_fn.weight", "hc_attn_fn"}, {"hc_attn_base.weight", "hc_attn_base"},
    {"hc_attn_scale.weight", "hc_attn_scale"}, {"hc_ffn_fn.weight", "hc_ffn_fn"},
    {"hc_ffn_base.weight", "hc_ffn_base"}, {"hc_ffn_scale.weight", "hc_ffn_scale"},
    {"attn_sinks.weight", "attn.attn_sink"},
    {"attn_q_a.weight", "attn.wq_a.weight"}, {"attn_q_b.weight", "attn.wq_b.weight"},
    {"attn_q_a_norm.weight", "attn.q_norm.weight"},
    {"attn_kv.weight", "attn.wkv.weight"}, {"attn_kv_a_norm.weight", "attn.kv_norm.weight"},
    {"attn_output_a.weight", "attn.wo_a.weight"}, {"attn_output_b.weight", "attn.wo_b.weight"},
    {"attn_compressor_ape.weight", "attn.compressor.ape"},
    {"attn_compressor_kv.weight", "attn.compressor.wkv.weight"},
    {"attn_compressor_gate.weight", "attn.compressor.wgate.weight"},
    {"attn_compressor_norm.weight", "attn.compressor.norm.weight"},
    {"indexer.attn_q_b.weight", "attn.indexer.wq_b.weight"},
    {"indexer.proj.weight", "attn.indexer.weights_proj.weight"},
    {"indexer_compressor_ape.weight", "attn.indexer.compressor.ape"},
    {"indexer_compressor_kv.weight", "attn.indexer.compressor.wkv.weight"},
    {"indexer_compressor_gate.weight", "attn.indexer.compressor.wgate.weight"},
    {"indexer_compressor_norm.weight", "attn.indexer.compressor.norm.weight"},
    {"attn_norm.weight", "attn_norm.weight"}, {"ffn_norm.weight", "ffn_norm.weight"},
    {"ffn_gate_inp.weight", "ffn.gate.weight"}, {"exp_probs_b.bias", "ffn.gate.bias"},
    {"ffn_gate_tid2eid.weight", "ffn.gate.tid2eid"},
    {"ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight"},
    {"ffn_down_shexp.weight", "ffn.shared_experts.w2.weight"},
    {"ffn_up_shexp.weight", "ffn.shared_experts.w3.weight"},
    {"ffn_gate_exps.weight", "ffn.experts.%d.w1.weight"},
    {"ffn_down_exps.weight", "ffn.experts.%d.w2.weight"},
    {"ffn_up_exps.weight", "ffn.experts.%d.w3.weight"},
    {NULL, NULL}};

/* Inkling (inkling): conversion/inkling.py of llama.cpp's PR (TML checkpoint names), here
 * under the names of transformers' port, which inkling.c reads. The port keeps gate and up
 * of a routed expert in one tensor (gate_up_proj, gate rows then up rows); the GGUF has them
 * apart, so they come per expert as mlp.experts.<e>.gate_proj / up_proj / down_proj.
 * ffn_gscale is a dense layer's mlp.global_scale and a sparse one's mlp.gate.global_scale:
 * both names (one tensor, the layer's kind decides which is read). */
static const st_gg_name st_gg_ink_glob[] = {
    {"token_embd.weight", "model.embed_tokens.weight"}, {"token_embd_norm.weight", "model.embed_norm.weight"},
    {"output_norm.weight", "model.norm.weight"}, {"output.weight", "lm_head.weight"},
    {NULL, NULL}};
static const st_gg_name st_gg_ink_layer[] = {
    {"attn_norm.weight", "input_layernorm.weight"}, {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"attn_q.weight", "self_attn.q_proj.weight"}, {"attn_k.weight", "self_attn.k_proj.weight"},
    {"attn_v.weight", "self_attn.v_proj.weight"}, {"attn_output.weight", "self_attn.o_proj.weight"},
    {"attn_r.weight", "self_attn.r_proj.weight"},
    {"attn_q_norm.weight", "self_attn.q_norm.weight"}, {"attn_k_norm.weight", "self_attn.k_norm.weight"},
    {"attn_rel_proj.weight", "self_attn.rel_logits_proj.proj"},
    {"shortconv_k.weight", "self_attn.k_sconv.conv1d.weight"}, {"shortconv_v.weight", "self_attn.v_sconv.conv1d.weight"},
    {"shortconv_attn.weight", "attn_sconv.conv1d.weight"}, {"shortconv_mlp.weight", "mlp_sconv.conv1d.weight"},
    {"ffn_gate.weight", "mlp.gate_proj.weight"}, {"ffn_up.weight", "mlp.up_proj.weight"},
    {"ffn_down.weight", "mlp.down_proj.weight"},
    {"ffn_gscale.weight", "mlp.global_scale|mlp.gate.global_scale"},
    {"ffn_gate_inp.weight", "mlp.gate.weight"}, {"exp_probs_b.bias", "mlp.gate.e_score_correction_bias"},
    {"ffn_gate_shexp.weight", "mlp.shared_experts.gate_proj"}, {"ffn_up_shexp.weight", "mlp.shared_experts.up_proj"},
    {"ffn_down_shexp.weight", "mlp.shared_experts.down_proj"},
    {"ffn_gate_exps.weight", "mlp.experts.%d.gate_proj"}, {"ffn_up_exps.weight", "mlp.experts.%d.up_proj"},
    {"ffn_down_exps.weight", "mlp.experts.%d.down_proj"},
    {NULL, NULL}};

/* GLM-5.3-Flash (glm5-next): conversion/glm.py Glm5NextModel. MLA's kv_b comes split the way
 * glm53.c absorbs it (k_b = W_k^T per head, v_b = W_v); KDA's A_log comes as ssm_a = -exp(A_log);
 * glm53.c reads both forms. */
static const st_gg_name st_gg_g5n_glob[] = {
    {"token_embd.weight", "model.language_model.embed_tokens.weight"},
    {"output_norm.weight", "model.language_model.norm.weight"}, {"output.weight", "lm_head.weight"},
    {NULL, NULL}};
static const st_gg_name st_gg_g5n_layer[] = {
    {"attn_norm.weight", "input_layernorm.weight"}, {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"hc_attn_fn.weight", "hc_attn_fn"}, {"hc_attn_base.weight", "hc_attn_base"}, {"hc_attn_scale.weight", "hc_attn_scale"},
    {"hc_ffn_fn.weight", "hc_ffn_fn"}, {"hc_ffn_base.weight", "hc_ffn_base"}, {"hc_ffn_scale.weight", "hc_ffn_scale"},
    {"attn_q.weight", "self_attn.q_proj.weight"}, {"attn_k.weight", "self_attn.k_proj.weight"},
    {"attn_v.weight", "self_attn.v_proj.weight"}, {"attn_output.weight", "self_attn.o_proj.weight"},
    {"ssm_g_a.weight", "self_attn.g_a_proj.weight"}, {"ssm_g_b.weight", "self_attn.g_b_proj.weight"},
    {"ssm_f_a.weight", "self_attn.f_a_proj.weight"}, {"ssm_f_b.weight", "self_attn.f_b_proj.weight"},
    {"ssm_beta.weight", "self_attn.b_proj.weight"},
    {"ssm_conv1d_q.weight", "self_attn.q_conv1d.weight"}, {"ssm_conv1d_k.weight", "self_attn.k_conv1d.weight"},
    {"ssm_conv1d_v.weight", "self_attn.v_conv1d.weight"},
    {"ssm_dt.bias", "self_attn.dt_bias"}, {"ssm_a", "self_attn.ssm_a"}, {"ssm_norm.weight", "self_attn.o_norm.weight"},
    {"attn_q_a.weight", "self_attn.q_a_proj.weight"}, {"attn_q_a_norm.weight", "self_attn.q_a_layernorm.weight"},
    {"attn_q_b.weight", "self_attn.q_b_proj.weight"}, {"attn_kv_a_mqa.weight", "self_attn.kv_a_proj_with_mqa.weight"},
    {"attn_kv_a_norm.weight", "self_attn.kv_a_layernorm.weight"},
    {"attn_k_b.weight", "self_attn.k_b_proj.weight"}, {"attn_v_b.weight", "self_attn.v_b_proj.weight"},
    {"indexer.attn_q_b.weight", "self_attn.indexer.wq_b.weight"}, {"indexer.attn_k.weight", "self_attn.indexer.wk.weight"},
    {"indexer.proj.weight", "self_attn.indexer.weights_proj.weight"},
    {"indexer.k_norm.weight", "self_attn.indexer.k_norm.weight"}, {"indexer.k_norm.bias", "self_attn.indexer.k_norm.bias"},
    {"indexer_compressor_ape.weight", "self_attn.indexer.index_kpool_compress_ape"},
    {"indexer_compressor_gate.weight", "self_attn.indexer.index_kpool_compress_gate"},
    {"ffn_gate.weight", "mlp.gate_proj.weight"}, {"ffn_up.weight", "mlp.up_proj.weight"}, {"ffn_down.weight", "mlp.down_proj.weight"},
    {"ffn_gate_inp.weight", "mlp.gate.weight"}, {"exp_probs_b.bias", "mlp.gate.e_score_correction_bias"},
    {"ffn_gate_shexp.weight", "mlp.shared_experts.gate_proj.weight"}, {"ffn_up_shexp.weight", "mlp.shared_experts.up_proj.weight"},
    {"ffn_down_shexp.weight", "mlp.shared_experts.down_proj.weight"},
    {"ffn_gate_exps.weight", "mlp.experts.%d.gate_proj.weight"}, {"ffn_up_exps.weight", "mlp.experts.%d.up_proj.weight"},
    {"ffn_down_exps.weight", "mlp.experts.%d.down_proj.weight"},
    {"nextn.eh_proj.weight", "eh_proj.weight"}, {"nextn.enorm.weight", "enorm.weight"}, {"nextn.hnorm.weight", "hnorm.weight"},
    {"nextn.shared_head_norm.weight", "shared_head.norm.weight"},
    {NULL, NULL}};

/* GLM-5.2 / GLM-5.3 (glm-dsa): conversion/glm.py GlmMoeDsaModel, DeepSeek V2's layout. MLA's
 * kv_b comes split, k_b = W_k transposed per head and v_b = W_v, under their own names; colibri.c
 * puts kv_b_proj back from them. The NextN block is the last layer, as in the checkpoint. */
static const st_gg_name st_gg_gdsa_glob[] = {
    {"token_embd.weight", "model.embed_tokens.weight"}, {"output_norm.weight", "model.norm.weight"},
    {"output.weight", "lm_head.weight"},
    {NULL, NULL}};
static const st_gg_name st_gg_gdsa_layer[] = {
    {"attn_norm.weight", "input_layernorm.weight"}, {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"attn_q_a.weight", "self_attn.q_a_proj.weight"}, {"attn_q_a_norm.weight", "self_attn.q_a_layernorm.weight"},
    {"attn_q_b.weight", "self_attn.q_b_proj.weight"}, {"attn_kv_a_mqa.weight", "self_attn.kv_a_proj_with_mqa.weight"},
    {"attn_kv_a_norm.weight", "self_attn.kv_a_layernorm.weight"},
    {"attn_k_b.weight", "self_attn.k_b_proj.weight"}, {"attn_v_b.weight", "self_attn.v_b_proj.weight"},
    {"attn_output.weight", "self_attn.o_proj.weight"},
    {"indexer.attn_q_b.weight", "self_attn.indexer.wq_b.weight"}, {"indexer.attn_k.weight", "self_attn.indexer.wk.weight"},
    {"indexer.proj.weight", "self_attn.indexer.weights_proj.weight"},
    {"indexer.k_norm.weight", "self_attn.indexer.k_norm.weight"}, {"indexer.k_norm.bias", "self_attn.indexer.k_norm.bias"},
    {"ffn_gate.weight", "mlp.gate_proj.weight"}, {"ffn_up.weight", "mlp.up_proj.weight"}, {"ffn_down.weight", "mlp.down_proj.weight"},
    {"ffn_gate_inp.weight", "mlp.gate.weight"}, {"exp_probs_b.bias", "mlp.gate.e_score_correction_bias"},
    {"ffn_gate_shexp.weight", "mlp.shared_experts.gate_proj.weight"}, {"ffn_up_shexp.weight", "mlp.shared_experts.up_proj.weight"},
    {"ffn_down_shexp.weight", "mlp.shared_experts.down_proj.weight"},
    {"ffn_gate_exps.weight", "mlp.experts.%d.gate_proj.weight"}, {"ffn_up_exps.weight", "mlp.experts.%d.up_proj.weight"},
    {"ffn_down_exps.weight", "mlp.experts.%d.down_proj.weight"},
    {"nextn.eh_proj.weight", "eh_proj.weight"}, {"nextn.enorm.weight", "enorm.weight"}, {"nextn.hnorm.weight", "hnorm.weight"},
    {"nextn.shared_head_norm.weight", "shared_head.norm.weight"},
    {NULL, NULL}};

/* MiMo-V2 (mimo2): conversion/mimo.py MimoV2Model. qkv_proj comes as one [Q|K|V] (the release
 * pre-shards it; the converter regroups it), the dense matrices dequantized; the NextN blocks
 * past the model's layers keep their GGUF names (mimo.c does not read them). */
static const st_gg_name st_gg_mimo_glob[] = {
    {"token_embd.weight", "model.embed_tokens.weight"}, {"output_norm.weight", "model.norm.weight"},
    {"output.weight", "lm_head.weight"},
    {NULL, NULL}};
static const st_gg_name st_gg_mimo_layer[] = {
    {"attn_norm.weight", "input_layernorm.weight"}, {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"attn_qkv.weight", "self_attn.qkv_proj.weight"}, {"attn_output.weight", "self_attn.o_proj.weight"},
    {"attn_sinks.weight", "self_attn.attention_sink_bias"},
    {"ffn_gate.weight", "mlp.gate_proj.weight"}, {"ffn_up.weight", "mlp.up_proj.weight"}, {"ffn_down.weight", "mlp.down_proj.weight"},
    {"ffn_gate_inp.weight", "mlp.gate.weight"}, {"exp_probs_b.bias", "mlp.gate.e_score_correction_bias"},
    {"ffn_gate_exps.weight", "mlp.experts.%d.gate_proj.weight"}, {"ffn_up_exps.weight", "mlp.experts.%d.up_proj.weight"},
    {"ffn_down_exps.weight", "mlp.experts.%d.down_proj.weight"},
    {NULL, NULL}};

static const st_gg_map st_gguf_maps[] = {
    {"mimo2", "model.layers.%d.", st_gg_mimo_glob, st_gg_mimo_layer},
    {"glm-dsa", "model.layers.%d.", st_gg_gdsa_glob, st_gg_gdsa_layer},
    {"glm5-next", "model.language_model.layers.%d.", st_gg_g5n_glob, st_gg_g5n_layer},
    {"deepseek4", "layers.%d.", st_gg_dsv4_glob, st_gg_dsv4_layer},
    {"inkling", "model.layers.%d.", st_gg_ink_glob, st_gg_ink_layer},
};

static const st_gg_map *st_gguf_map(const char *arch) {
    for (size_t i = 0; arch && i < sizeof st_gguf_maps / sizeof *st_gguf_maps; i++)
        if (!strcmp(st_gguf_maps[i].arch, arch)) return &st_gguf_maps[i];
    return NULL;
}

/* The GGUF metadata of the model st_init_gguf opened (architecture, tokenizer, ...): the
 * engine reads its config from here or from the config.json the importer wrote. */
static gguf_ctx st_gguf_meta;
static int st_gguf_active;

static int st_gguf_dtype(int type) {
    switch (type) {
        case GQ_F32: return 2;
        case GQ_F16: return 1;
        case GQ_BF16: return 0;
        default: break;
    }
    if (type == 26) return 7;   /* I32 (the hash router's table) */
    if (type == 24) return 3;   /* I8 */
    if (type == 27) return 6;   /* I64 */
    return ST_GGML_BASE + type;
}

static st_tensor *st_gguf_add(shards *S, const char *name) {
    if (S->n == S->cap) {
        S->cap *= 2;
        st_tensor *t = (st_tensor *)realloc(S->t, (size_t)S->cap * sizeof(st_tensor));
        if (!t) coli_load_fail(COLI_LOAD_FAIL_NOMEM, "out of memory indexing the GGUF");
        S->t = t;
    }
    st_tensor *t = &S->t[S->n++];
    memset(t, 0, sizeof *t);
    t->name = strdup(name);
    return t;
}

/* The colibri container's layout keys: per layer the three matrices' ggml types and bytes, the
 * slot's stride and each matrix's [cols, rows]; NULL when the GGUF is not one. */
typedef struct { const uint32_t *types; const uint64_t *bytes, *stride, *dims; int layers; } st_gg_packed;
static int st_gguf_packed(const gguf_ctx *G, st_gg_packed *p) {
    const char *layout = gguf_get_str(G, "colibri.layout");
    if (!layout) return 0;
    if (strcmp(layout, "expert-packed-v1")) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF: colibri layout '%s' is not one this build reads", layout);
    const gguf_kv *ty = gguf_get(G, "colibri.expert_types"), *by = gguf_get(G, "colibri.expert_bytes"),
                  *st = gguf_get(G, "colibri.expert_stride"), *di = gguf_get(G, "colibri.expert_dims");
    if (!di) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF: a colibri container without colibri.expert_dims (written before "
                                                   "this engine read containers): import the GGUF again");
    if (!ty || !by || !st || ty->type != GGUF_T_ARRAY || by->type != GGUF_T_ARRAY || st->type != GGUF_T_ARRAY ||
        di->type != GGUF_T_ARRAY || ty->atype != GGUF_T_U32 || by->atype != GGUF_T_U64 || st->atype != GGUF_T_U64 ||
        di->atype != GGUF_T_U64 || ty->n != 3 * st->n || by->n != 3 * st->n || di->n != 6 * st->n)
        coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF: the colibri container's expert layout keys disagree");
    *p = (st_gg_packed){ (const uint32_t *)ty->arr, (const uint64_t *)by->arr, (const uint64_t *)st->arr,
                         (const uint64_t *)di->arr, (int)st->n };
    return 1;
}

/* 1 when snap_dir is a gguf_import folder and S now indexes its GGUF, 0 when it is not one.
 * A broken GGUF or one of an architecture without a map is fatal (the folder says GGUF). */
static int st_init_gguf(shards *S, const char *snap_dir) {
    char gj[4096]; snprintf(gj, sizeof gj, "%s/gguf.json", snap_dir);
    FILE *f = fopen(gj, "rb");
    if (!f) return 0;
    char buf[16384]; size_t nb = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[nb] = 0;
    char *arena = NULL; jval *root = json_parse(buf, &arena);
    jval *gv = root ? json_get(root, "gguf") : NULL;
    const char *g = gv && gv->t == J_STR ? gv->str : NULL;
    if (!g || !*g) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "%s: no \"gguf\" path", gj);
    char path[4096];
    if (g[0] == '/') snprintf(path, sizeof path, "%s", g); else snprintf(path, sizeof path, "%s/%s", snap_dir, g);
    free(arena);
    if (getenv("GGUF_CHECK")) gguf_header_only = 1;
    gguf_ctx *G = &st_gguf_meta;
    if (gguf_open(G, path)) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF: %s", G->err);
    const char *arch = gguf_get_str(G, "general.architecture");
    const st_gg_map *map = st_gguf_map(arch);
    if (!map) coli_load_fail(COLI_LOAD_FAIL_UNSUPPORTED, "%s: GGUF architecture '%s' has no tensor map in st_gguf.h", path, arch ? arch : "?");
    int fds[ST_MAX_SHARDS];
    if (G->n_files > ST_MAX_SHARDS) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "%s: %d parts", path, G->n_files);
    for (int p = 0; p < G->n_files; p++) fds[p] = st_open_fd(S, G->paths[p]);
    S->cap = G->n_tensors + 16; S->t = (st_tensor *)calloc((size_t)S->cap, sizeof(st_tensor));
    if (!S->t) coli_load_fail(COLI_LOAD_FAIL_NOMEM, "out of memory indexing the GGUF");
    int experts = 0, renamed = 0, kept = 0;
    char nm[512];
    st_gg_packed pk = {0};
    const int packed = st_gguf_packed(G, &pk);
    for (int i = 0; i < G->n_tensors; i++) {
        const gguf_tensor *t = &G->t[i];
        const char *hf = NULL; int L = -1, off = 0;
        int pl = -1, pn = 0;
        if (packed && sscanf(t->name, "blk.%d.ffn_exps_colibri%n", &pl, &pn) == 1 && !t->name[pn]) {
            /* a layer's packed experts: each kind's matrices at their place in every slot */
            static const char *kinds[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
            if (pl < 0 || pl >= pk.layers || t->n_dims != 2 || (uint64_t)t->ne[0] != pk.stride[pl])
                coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF %s: disagrees with the container's layout", t->name);
            uint64_t inner = 0;
            for (int k = 0; k < 3; k++) {
                const char *kh = NULL;
                for (const st_gg_name *e = map->layer; e->gg && !kh; e++) if (!strcmp(kinds[k], e->gg)) kh = e->hf;
                if (!kh || !strstr(kh, "%d"))
                    coli_load_fail(COLI_LOAD_FAIL_FORMAT, "%s: this engine reads %s's routed experts as whole tensors, not "
                                                          "a colibri container: import with --mode link", path, arch);
                const uint64_t b = pk.bytes[3 * pl + k], cols = pk.dims[6 * pl + 2 * k], rows = pk.dims[6 * pl + 2 * k + 1];
                const int64_t rb = gq_row_bytes((int)pk.types[3 * pl + k], (int64_t)cols);
                if (rb < 0 || (uint64_t)rb * rows != b || inner + b > pk.stride[pl])
                    coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF %s: matrix %d disagrees with the container's layout", t->name, k);
                for (int64_t x = 0; x < t->ne[1]; x++) {
                    char rest[256], pre[64];
                    snprintf(rest, sizeof rest, kh, (int)x); snprintf(pre, sizeof pre, map->layer_fmt, pl);
                    snprintf(nm, sizeof nm, "%s%s", pre, rest);
                    st_tensor *s = st_gguf_add(S, nm);
                    s->fd = fds[t->file]; s->off = (int64_t)(t->off + (uint64_t)x * pk.stride[pl] + inner); s->nbytes = (int64_t)b;
                    s->dtype = st_gguf_dtype((int)pk.types[3 * pl + k]);
                    s->rank = 2; s->shape[0] = (int64_t)rows; s->shape[1] = (int64_t)cols; s->numel = (int64_t)(rows * cols);
                }
                inner += b;
                experts++;
            }
            continue;
        }
        for (const st_gg_name *e = map->globals; e->gg && !hf; e++) if (!strcmp(t->name, e->gg)) hf = e->hf;
        if (!hf && sscanf(t->name, "blk.%d.%n", &L, &off) == 1 && L >= 0 && off)
            for (const st_gg_name *e = map->layer; e->gg && !hf; e++) if (!strcmp(t->name + off, e->gg)) hf = e->hf;
        char alias[256];   /* "a|b": the tensor under each name (st_gg_ink_layer's ffn_gscale) */
        const char *rest_names = NULL;
        if (hf && strchr(hf, '|')) {
            snprintf(alias, sizeof alias, "%s", hf);
            char *bar = strchr(alias, '|'); *bar = 0;
            hf = alias; rest_names = bar + 1;
        }
again:;
        const int per_expert = hf && strstr(hf, "%d") != NULL;
        const int dims = per_expert ? 2 : t->n_dims;
        if (per_expert && (t->n_dims != 3 || t->ne[2] > INT_MAX))
            coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF %s: expert count does not fit the tensor index", t->name);
        const int E = per_expert ? (int)t->ne[2] : 1;
        if (per_expert && (t->n_dims != 3 || t->nbytes % E)) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF %s: not [experts][rows][cols]", t->name);
        const int64_t slice = t->nbytes / E;
        for (int x = 0; x < E; x++) {
            if (!hf) snprintf(nm, sizeof nm, "%s", t->name);
            else {
                char rest[256];
                if (per_expert) snprintf(rest, sizeof rest, hf, x); else snprintf(rest, sizeof rest, "%s", hf);
                if (L >= 0) { char pre[64]; snprintf(pre, sizeof pre, map->layer_fmt, L); snprintf(nm, sizeof nm, "%s%s", pre, rest); }
                else snprintf(nm, sizeof nm, "%s", rest);
            }
            st_tensor *s = st_gguf_add(S, nm);
            s->fd = fds[t->file]; s->off = (int64_t)t->off + slice * x; s->nbytes = slice;
            s->dtype = st_gguf_dtype(t->type);
            s->rank = dims; s->numel = 1;
            for (int d = 0; d < dims; d++) { s->shape[d] = t->ne[dims - 1 - d]; s->numel *= s->shape[d]; }
        }
        if (per_expert) experts++; else if (hf) renamed++; else kept++;
        if (rest_names) {
            char *bar = strchr((char *)rest_names, '|');
            if (bar) *bar = 0;
            hf = rest_names; rest_names = bar ? bar + 1 : NULL;
            goto again;
        }
    }
    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = (int *)malloc((size_t)S->hcap * sizeof(int));
    if (!S->hidx) coli_load_fail(COLI_LOAD_FAIL_NOMEM, "out of memory indexing the GGUF");
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        if (st_find(S, S->t[i].name)) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF: two tensors map to %s", S->t[i].name);
        uint64_t h = st_hash(S->t[i].name) & (uint64_t)(S->hcap - 1);
        while (S->hidx[h] >= 0) h = (h + 1) & (uint64_t)(S->hcap - 1);
        S->hidx[h] = i;
    }
    st_gguf_active = 1;
    fprintf(stderr, "[gguf] %s: %s, %d part(s), %d tensors under the checkpoint's names (%d renamed, %d %s per expert, "
                    "%d kept as named), read in place\n",
            path, arch, G->n_files, S->n, renamed, experts,
            packed ? "matrices of the container's expert slots" : "fused expert tensors split", kept);
    return 1;
}

#endif /* ST_GGUF_H */
