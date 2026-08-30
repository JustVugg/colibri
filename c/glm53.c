/* GLM-5.3-Flash inference engine in pure C — sibling of kimi_k3.c / colibri.c /
 * deepseek_v4.c / inkling.c / qwen36.c / olmoe.c, sharing st.h / json.h / tok.h /
 * quant.h / compat.h.
 *
 * The name says GLM, but the skeleton is Kimi K3's, not GLM-5.2's: a hybrid
 * stack of linear-attention layers punctuated by full attention, over a
 * streamed MoE. That is why this file starts from kimi_k3.c and borrows the
 * indexer from colibri.c and the hyper-connections from deepseek_v4.c, instead
 * of extending the GLM engine.
 *
 * Architecture, read off the released checkpoint (321.34 B parameters measured
 * from the shard headers; the card says 320 B total / 18 B active):
 *
 *   - 45 text layers plus one MTP layer (index 45), hidden 4096, vocab 154880.
 *   - Hybrid attention, pattern (KDA KDA KDA FULL) repeating: 34 KDA linear
 *     layers and 11 DeepSeek-sparse-attention layers at 3, 7, ... 43, plus the
 *     MTP layer's own full-attention block. config.layer_types is explicit and
 *     is what this engine follows; linear_attn_config.kda_layers agrees.
 *   - MLA with kv_lora 512, q_lora 1536, qk_nope 256 and **qk_rope 0**: the
 *     full-attention layers are NoPE, exactly like K3. Position lives in the
 *     KDA decay and short convolutions, not in a rotation.
 *   - DSA lightning indexer on every full-attention layer, with k-pooling:
 *     keys are grouped into pools of index_kpool=4, the pools are scored
 *     instead of the tokens, the top index_topk/kpool pools are expanded back
 *     into token indices and the incomplete tail pool is always appended
 *     (index_kpool_always_select_tail). Two tensors carry it that GLM-5.2's
 *     indexer does not have: index_kpool_compress_ape and _compress_gate.
 *   - Manifold-Constrained Hyper-Connections (mHC, hc_mult 4, 20 Sinkhorn
 *     iterations) replace the plain residual at both sites of every layer —
 *     the same mHC DeepSeek V4 uses, down to the config keys, so the split
 *     into pre/post/comb and the Sinkhorn projection are shared code.
 *   - MoE: 288 routed experts (top-8) plus 1 shared expert per layer from
 *     layer 3 on, moe_intermediate 2048, sigmoid scoring with noaux_tc and
 *     e_score_correction_bias, routed_scaling_factor 2.5. The first three
 *     layers are dense (intermediate 12288).
 *   - Natively multimodal: a 24-block ViT (hidden 1024, patch 14, 448 px,
 *     spatial merge 2) whose patches are projected to 4096 and substituted at
 *     the image-token positions of the text stream. Text-only prompts never
 *     touch it.
 *
 * KDA recurrence — the same Kimi Delta Attention kimi_k3.c already reproduces
 * token-exact against the vendor, verified line by line against
 * transformers' recurrent_kimi_delta_attention:
 *     q,k,v = SiLU(ShortConv4(W{q,k,v} x));  q,k L2-normalized, q *= d^-0.5
 *     z  = W_fb(W_fa x) + dt_bias
 *     gk = gmin * sigmoid(exp(A_log[h]) * z),  gmin = gate_lower_bound = -5
 *     S  = (I - beta k k^T) Diag(exp(gk)) S + beta k v^T,  beta = sigmoid(W_b x)
 *     o  = S^T q;  out = W_o [ sigmoid(W_gb(W_ga x)) * RMSNorm_head(o) ]
 * One difference from K3: the output gate is LOW-RANK here (g_a_proj into
 * head_dim, then g_b_proj back out), where K3 has a single full g_proj.
 *
 * Container: tools/convert_glm53.py writes routed experts as int4 group-scaled
 * gs64 (`name` U8 + `name.qs` F32, fmt=4 — the same container GLM-5.2 uses,
 * measured cosine 0.994 against the fp8 source on real weights) and everything
 * else as BF16, quantized at LOAD TIME here. That split is deliberate: the
 * non-expert weights are 3% of the bytes, so keeping them exact on disk costs
 * ~14 GB and means retuning dense precision never requires re-downloading the
 * checkpoint.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>

#include "json.h"
#include "st.h"
#include "quant.h"
#include "tok.h"
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
static int g_vk_ready = 0;
#endif
#include "compat.h"
#include <time.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include "hyper_connections.h"   /* mHC, condiviso con deepseek_v4.c */

/* ---------- config ----------
 * Nested like Kimi K3's: the root carries the vision wrapper and `text_config`
 * carries the language model. A text-only export therefore has the text keys at
 * the root, and both shapes are accepted. */
typedef struct {
    /* text */
    int hidden, n_layers, vocab, first_dense, dense_inter;
    int n_heads, q_lora, kv_lora, qk_nope, qk_rope, v_head, qk_head;
    int n_experts, topk, moe_inter, n_shared;
    float routed_scale, eps, swiglu_limit;
    int n_mtp;                       /* num_nextn_predict_layers (1) */
    /* KDA */
    int kda_heads, kda_hd, kda_proj, conv_k;
    float gate_lb;
    /* DSA indexer with k-pooling */
    int index_topk, index_nh, index_hd, index_kpool, index_kpool_tail;
    /* mHC */
    int hc_mult, hc_iters;
    float hc_eps;
    /* per-layer kind: 1 = full attention (MLA + indexer), 0 = KDA */
    unsigned char is_full[128];
    /* vision (0 = text-only checkpoint) */
    int vis_layers, vis_hidden, vis_heads, vis_inter, vis_patch, vis_temporal;
    int vis_merge, vis_out_hidden, vis_proj_inter, vis_image_size, vis_in_ch;
    float vis_swiglu_limit, vis_eps;
    int image_token, image_start_token, image_end_token;
    int video_token, video_start_token, video_end_token;
} Cfg;

static double req_num(jval *object, const char *key) {
    jval *value = json_get(object, key);
    if (!value || value->t != J_NUM) {
        fprintf(stderr, "config.json: missing or non-numeric \"%s\"\n", key);
        exit(1);
    }
    return value->num;
}

static double opt_num(jval *object, const char *key, double fallback) {
    jval *value = object ? json_get(object, key) : NULL;
    return (value && value->t == J_NUM) ? value->num : fallback;
}

static int opt_bool(jval *object, const char *key, int fallback) {
    jval *value = object ? json_get(object, key) : NULL;
    if (!value) return fallback;
    if (value->t == J_BOOL) return value->boolean;
    if (value->t == J_NUM) return value->num != 0.0;
    return fallback;
}

/* layer_types is the authority on which layers are full attention. The
 * linear_attn_config.{kda_layers,full_attn_layers} lists say the same thing;
 * disagreeing checkpoints are refused rather than guessed at, because picking
 * the wrong kind for one layer produces plausible-looking garbage. */
static void load_layer_kinds(Cfg *c, jval *text) {
    memset(c->is_full, 0, sizeof(c->is_full));
    jval *types = json_get(text, "layer_types");
    if (!types || types->t != J_ARR || types->len != c->n_layers) {
        fprintf(stderr, "config.json: layer_types must list %d entries\n", c->n_layers);
        exit(1);
    }
    int full = 0;
    for (int i = 0; i < types->len; i++) {
        jval *entry = types->kids[i];
        if (!entry || entry->t != J_STR) {
            fprintf(stderr, "config.json: layer_types[%d] is not a string\n", i);
            exit(1);
        }
        if (strstr(entry->str, "linear")) {
            c->is_full[i] = 0;
        } else if (strstr(entry->str, "attention") || strstr(entry->str, "full")) {
            c->is_full[i] = 1;
            full++;
        } else {
            fprintf(stderr, "config.json: unknown layer type \"%s\" at %d\n", entry->str, i);
            exit(1);
        }
    }
    jval *linear = json_get(text, "linear_attn_config");
    jval *full_list = linear ? json_get(linear, "full_attn_layers") : NULL;
    if (full_list && full_list->t == J_ARR) {
        if (full_list->len != full) {
            fprintf(stderr, "config.json: full_attn_layers lists %d layers, "
                            "layer_types marks %d\n", full_list->len, full);
            exit(1);
        }
        for (int i = 0; i < full_list->len; i++) {
            int index = (int)full_list->kids[i]->num;
            if (index < 0 || index >= c->n_layers || !c->is_full[index]) {
                fprintf(stderr, "config.json: full_attn_layers disagrees with "
                                "layer_types at %d\n", index);
                exit(1);
            }
        }
    }
    /* The MTP block is a full-attention layer that lives past num_hidden_layers
     * and is not described by layer_types. */
    if (c->n_layers < (int)sizeof(c->is_full)) c->is_full[c->n_layers] = 1;
}

static void load_vision(Cfg *c, jval *root) {
    jval *vision = json_get(root, "vision_config");
    if (!vision || vision->t != J_OBJ) { c->vis_layers = 0; return; }
    c->vis_layers      = (int)req_num(vision, "depth");
    c->vis_hidden      = (int)req_num(vision, "hidden_size");
    c->vis_heads       = (int)req_num(vision, "num_heads");
    c->vis_inter       = (int)req_num(vision, "intermediate_size");
    c->vis_patch       = (int)req_num(vision, "patch_size");
    c->vis_temporal    = (int)opt_num(vision, "temporal_patch_size", 2);
    c->vis_merge       = (int)opt_num(vision, "spatial_merge_size", 2);
    c->vis_out_hidden  = (int)opt_num(vision, "out_hidden_size", c->hidden);
    c->vis_proj_inter  = (int)opt_num(vision, "projection_intermediate_size", 0);
    c->vis_image_size  = (int)opt_num(vision, "image_size", 448);
    c->vis_in_ch       = (int)opt_num(vision, "in_channels", 3);
    c->vis_swiglu_limit= (float)opt_num(vision, "swiglu_limit", 10.0);
    c->vis_eps         = (float)opt_num(vision, "rms_norm_eps", 1e-5);
    c->image_token       = (int)opt_num(root, "image_token_id", -1);
    c->image_start_token = (int)opt_num(root, "image_start_token_id", -1);
    c->image_end_token   = (int)opt_num(root, "image_end_token_id", -1);
    c->video_token       = (int)opt_num(root, "video_token_id", -1);
    c->video_start_token = (int)opt_num(root, "video_start_token_id", -1);
    c->video_end_token   = (int)opt_num(root, "video_end_token_id", -1);
    if (c->vis_layers < 1 || c->vis_layers > 128 || c->vis_hidden < 1 ||
        c->vis_heads < 1 || c->vis_hidden % c->vis_heads ||
        c->vis_patch < 1 || c->vis_merge < 1 || c->vis_out_hidden != c->hidden) {
        fprintf(stderr, "config.json: vision_config out of range "
                        "(out_hidden_size must equal the text hidden size)\n");
        exit(1);
    }
}

static void load_cfg(Cfg *c, const char *snap) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    char *buffer = NULL;
    {
        FILE *stream = fopen(path, "rb");
        if (!stream) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
        if (fseek(stream, 0, SEEK_END)) { fprintf(stderr, "%s: not seekable\n", path); exit(1); }
        long length = ftell(stream);
        if (length < 2 || fseek(stream, 0, SEEK_SET)) {
            fprintf(stderr, "%s: unusable\n", path); exit(1); }
        buffer = malloc((size_t)length + 1);
        if (!buffer) { fprintf(stderr, "OOM reading config\n"); exit(1); }
        if (fread(buffer, 1, (size_t)length, stream) != (size_t)length) {
            fprintf(stderr, "%s: short read\n", path); exit(1); }
        buffer[length] = 0;
        fclose(stream);
    }
    char *arena = NULL;
    jval *root = json_parse(buffer, &arena);
    if (!root || root->t != J_OBJ) { fprintf(stderr, "%s: not a JSON object\n", path); exit(1); }
    memset(c, 0, sizeof(*c));

    jval *tc = json_get(root, "text_config");
    if (!tc || tc->t != J_OBJ) tc = root;      /* text-only export */

    c->hidden      = (int)req_num(tc, "hidden_size");
    c->n_layers    = (int)req_num(tc, "num_hidden_layers");
    c->vocab       = (int)req_num(tc, "vocab_size");
    /* Il checkpoint reale dichiara first_k_dense_replace; la fixture tiny usa
     * mlp_layer_types. Accettiamo entrambi invece di imporne uno: sono la
     * stessa informazione detta in due modi, e rifiutare la fixture
     * significherebbe non poter provare il motore senza 194 GB. */
    {
        jval *fkd = json_get(tc, "first_k_dense_replace");
        if (fkd && fkd->t == J_NUM) {
            c->first_dense = (int)fkd->num;
        } else {
            jval *kinds = json_get(tc, "mlp_layer_types");
            if (!kinds || kinds->t != J_ARR) {
                fprintf(stderr, "config.json: serve first_k_dense_replace "
                                "oppure mlp_layer_types\n");
                exit(1);
            }
            c->first_dense = kinds->len;
            for (int i = 0; i < kinds->len; i++)
                if (kinds->kids[i]->t == J_STR && strstr(kinds->kids[i]->str, "sparse")) {
                    c->first_dense = i;
                    break;
                }
        }
    }
    c->dense_inter = (int)req_num(tc, "intermediate_size");
    c->n_heads     = (int)req_num(tc, "num_attention_heads");
    c->q_lora      = (int)req_num(tc, "q_lora_rank");
    c->kv_lora     = (int)req_num(tc, "kv_lora_rank");
    c->qk_nope     = (int)req_num(tc, "qk_nope_head_dim");
    c->qk_rope     = (int)opt_num(tc, "qk_rope_head_dim", 0);
    c->v_head      = (int)req_num(tc, "v_head_dim");
    c->n_experts   = (int)req_num(tc, "n_routed_experts");
    c->topk        = (int)req_num(tc, "num_experts_per_tok");
    c->moe_inter   = (int)req_num(tc, "moe_intermediate_size");
    c->n_shared    = (int)opt_num(tc, "n_shared_experts", 1);
    c->routed_scale= (float)opt_num(tc, "routed_scaling_factor", 1.0);
    c->eps         = (float)opt_num(tc, "rms_norm_eps", 1e-6);
    c->swiglu_limit= (float)opt_num(tc, "swiglu_limit", 0.0);
    c->n_mtp       = (int)opt_num(tc, "num_nextn_predict_layers", 0);
    c->qk_head     = c->qk_nope + c->qk_rope;

    jval *linear = json_get(tc, "linear_attn_config");
    if (!linear || linear->t != J_OBJ) {
        fprintf(stderr, "config.json: missing linear_attn_config\n"); exit(1);
    }
    c->kda_heads = (int)req_num(linear, "num_heads");
    c->kda_hd    = (int)req_num(linear, "head_dim");
    c->conv_k    = (int)req_num(linear, "short_conv_kernel_size");
    c->gate_lb   = (float)opt_num(linear, "gate_lower_bound", -5.0);
    c->kda_proj  = c->kda_heads * c->kda_hd;

    c->index_topk       = (int)opt_num(tc, "index_topk", 0);
    c->index_nh         = (int)opt_num(tc, "index_n_heads", 0);
    c->index_hd         = (int)opt_num(tc, "index_head_dim", 0);
    c->index_kpool      = (int)opt_num(tc, "index_kpool", 1);
    c->index_kpool_tail = opt_bool(tc, "index_kpool_always_select_tail", 0);

    c->hc_mult  = (int)opt_num(tc, "hc_mult", 1);
    c->hc_iters = (int)opt_num(tc, "hc_sinkhorn_iters", 0);
    c->hc_eps   = (float)opt_num(tc, "hc_eps", 1e-6);

    load_layer_kinds(c, tc);
    load_vision(c, root);

    if (c->hidden < 1 || c->hidden > 65536 ||
        c->n_layers < 1 || c->n_layers > 120 ||
        c->vocab < 1 || c->vocab > (1 << 22) ||
        c->n_experts < 1 || c->n_experts > 4096 ||
        c->topk < 1 || c->topk > 64 || c->topk > c->n_experts ||
        c->kda_proj < 1 || c->kda_proj > (1 << 20) ||
        c->conv_k < 1 || c->conv_k > 8 ||
        c->moe_inter % 32 || c->kda_hd > 512 || c->kv_lora > 4096 ||
        c->first_dense < 0 || c->first_dense > c->n_layers ||
        c->index_kpool < 1 || c->index_kpool > 64 ||
        c->hc_mult < 1 || c->hc_mult > 8) {
        fprintf(stderr, "config.json: dimension out of range\n"); exit(1);
    }
    /* qk_rope must be zero: a rotary GLM-5.3 would need position handling this
     * engine deliberately does not have, and silently ignoring the rotation
     * would produce a model that answers fluently and wrongly. */
    if (c->qk_rope != 0) {
        fprintf(stderr, "config.json: qk_rope_head_dim=%d, but this engine "
                        "implements the NoPE full-attention of GLM-5.3\n", c->qk_rope);
        exit(1);
    }
    free(arena);
    free(buffer);
}

static void cfg_report(const Cfg *c) {
    int full = 0, kda = 0;
    for (int i = 0; i < c->n_layers; i++) { if (c->is_full[i]) full++; else kda++; }
    double expert_params = (double)c->n_experts * 3.0 * c->hidden * c->moe_inter *
                           (c->n_layers - c->first_dense + (c->n_mtp ? 1 : 0));
    fprintf(stderr,
        "GLM-5.3-Flash: %d layers (%d KDA + %d full) + %d MTP, hidden %d, vocab %d\n"
        "  MoE      : %d routed (top-%d) + %d shared, inter %d, from layer %d, scale %.2f\n"
        "  KDA      : %d heads x %d, conv %d, gate floor %.1f\n"
        "  MLA      : q_lora %d, kv_lora %d, qk %d (nope, NoPE), v %d, %d heads\n"
        "  indexer  : top-%d, %d heads x %d, kpool %d%s\n"
        "  mHC      : mult %d, %d Sinkhorn iterations\n"
        "  vision   : %s\n"
        "  routed experts: %.1f B parameters (%.0f GB at int4-g64)\n",
        c->n_layers, kda, full, c->n_mtp, c->hidden, c->vocab,
        c->n_experts, c->topk, c->n_shared, c->moe_inter, c->first_dense, c->routed_scale,
        c->kda_heads, c->kda_hd, c->conv_k, (double)c->gate_lb,
        c->q_lora, c->kv_lora, c->qk_nope, c->v_head, c->n_heads,
        c->index_topk, c->index_nh, c->index_hd, c->index_kpool,
        c->index_kpool_tail ? " (+tail)" : "",
        c->hc_mult, c->hc_iters,
        c->vis_layers ? "yes" : "text-only checkpoint",
        expert_params / 1e9, expert_params * 0.5625 / 1e9);
    if (c->vis_layers)
        fprintf(stderr,
        "             %d blocks x %d, %d heads, patch %d, %dpx, merge %d -> %d\n",
        c->vis_layers, c->vis_hidden, c->vis_heads, c->vis_patch,
        c->vis_image_size, c->vis_merge, c->vis_out_hidden);
}

#ifdef GLM53_CFG_MAIN_UNUSED
/* Config-only entry point: `glm53_cfg <model_dir>` parses and reports, so the
 * parser can be checked against a real checkpoint before any weight exists. */
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    Cfg c;
    load_cfg(&c, argv[1]);
    cfg_report(&c);
    return 0;
}
#endif

/* ---------- RAM-resident weight ----------
 * Same shape as kimi_k3.c's: either f32, int8 per-row, or int4 group-scaled.
 * The container written by tools/convert_glm53.py is U8 + `.qs` F32, which is
 * exactly what K3's loader already reads, so the two engines share a format
 * rather than each inventing one. */
typedef struct {
    int fmt;                              /* 0 = f32, 8 = int8 per-row, 4 = int4-g64 */
    float *f;
    int8_t *q8;
    uint8_t *q4;
    float *s;                             /* scales: [O] per-row, or [O*ngroups] */
    int O, I, gs;
} W;

/* ---------- layer structures ----------
 * Mirrors kimi_k3.c's shapes, with the GLM-5.3 differences called out where
 * they bite: the KDA output gate is low-rank here, the full-attention layers
 * are NOT gated (K3's are), and every layer carries two mHC sites instead of a
 * plain residual. */
typedef struct {                          /* KDA (linear attention) layer */
    W q, k, v, o;                         /* [proj x hidden] x3, [hidden x proj] */
    W ga, gb;                             /* low-rank output gate: hidden->hd->proj */
    float *conv_q, *conv_k, *conv_v;      /* [proj*conv_k] depthwise taps */
    float *fa, *fb;                       /* decay low-rank: [hd,hidden], [proj,hd] */
    float *bp;                            /* beta projection [heads,hidden] */
    float *dt, *A, *onw;                  /* dt_bias[proj], exp(A_log)[heads], o_norm[hd] */
} Kda;

typedef struct {                          /* MLA + DSA indexer (full attention) */
    W qa, qb, kva, kvb, o;
    float *qa_ln, *kva_ln;
    W wq, wk, wp;                         /* indexer: wq_b, wk, weights_proj */
    float *knw, *knb;                     /* indexer key LayerNorm (weight + bias) */
    float *kpool_ape;                     /* [kpool, index_hd] pool position bias */
    W kpool_gate;                         /* [index_hd, hidden] compression gate */
} Mla;

typedef struct {                          /* MoE (routed streamed + shared resident) */
    float *router, *rbias;                /* [E,hidden] f32, [E] correction bias */
    W sh_gate, sh_up, sh_down;
} Moe;

typedef struct {
    int full;                             /* 1 = MLA + indexer, 0 = KDA */
    int dense;                            /* 1 = plain MLP (layers < first_dense) */
    Kda a;
    Mla m;
    Moe moe;
    W d_gate, d_up, d_down;               /* dense layers only */
    float *in_ln, *post_ln;
    /* mHC, two sites per layer. fn is [(2+H)*H, H*hidden], base [(2+H)*H],
     * scale [3]; the split into pre/post/comb and the Sinkhorn projection are
     * the same as DeepSeek V4's. Absent on the MTP layer. */
    float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    float *hc_ffn_fn,  *hc_ffn_base,  *hc_ffn_scale;
    /* MTP layer only */
    W mtp_eh;
    float *mtp_enorm, *mtp_hnorm, *mtp_head_norm;
} Layer;

/* ---------- tensor names ----------
 * One place builds every name the engine asks for. The checkpoint prefixes the
 * language model with `model.language_model.` because the root is the vision
 * wrapper; a text-only export drops it. Both are probed, and the choice is made
 * once from a tensor that must exist either way. */
typedef struct {
    const char *prefix;                   /* "model.language_model." or "model." */
    const char *visual;                   /* "model.visual." */
} Names;

#define GLM53_NAME(dst, fmt, ...) snprintf((dst), sizeof(dst), (fmt), __VA_ARGS__)

/* Emits, in load order, every tensor this engine will look for. `sink` is
 * called with (name, required); a NULL model just prints them, which is how
 * the mapping gets checked against a real checkpoint index before any weight
 * has been downloaded. */
static void glm53_walk_tensors(const Cfg *c, const Names *n,
                               void (*sink)(void *, const char *, int),
                               void *user) {
    char name[512];
#define EMIT(required, fmt, ...) do { \
        GLM53_NAME(name, fmt, __VA_ARGS__); sink(user, name, (required)); } while (0)
    EMIT(1, "%sembed_tokens.weight", n->prefix);
    EMIT(1, "%snorm.weight", n->prefix);
    sink(user, "lm_head.weight", 1);

    int last = c->n_layers + (c->n_mtp ? 1 : 0);
    for (int i = 0; i < last; i++) {
        int mtp = i >= c->n_layers;
        int full = c->is_full[i];
        EMIT(1, "%slayers.%d.input_layernorm.weight", n->prefix, i);
        EMIT(1, "%slayers.%d.post_attention_layernorm.weight", n->prefix, i);
        if (!mtp) {                        /* mHC lives on the 45 real layers */
            EMIT(1, "%slayers.%d.hc_attn_fn", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_attn_base", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_attn_scale", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_ffn_fn", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_ffn_base", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_ffn_scale", n->prefix, i);
        } else {
            EMIT(1, "%slayers.%d.eh_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.enorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.hnorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.shared_head.norm.weight", n->prefix, i);
        }
        if (full) {
            EMIT(1, "%slayers.%d.self_attn.q_a_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.q_a_layernorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.q_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.kv_a_proj_with_mqa.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.kv_a_layernorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.kv_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.o_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.wq_b.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.wk.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.weights_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.k_norm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.k_norm.bias", n->prefix, i);
            if (c->index_kpool > 1) {
                EMIT(1, "%slayers.%d.self_attn.indexer.index_kpool_compress_ape", n->prefix, i);
                EMIT(1, "%slayers.%d.self_attn.indexer.index_kpool_compress_gate", n->prefix, i);
            }
        } else {
            EMIT(1, "%slayers.%d.self_attn.q_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.k_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.v_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.o_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.g_a_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.g_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.q_conv1d.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.k_conv1d.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.v_conv1d.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.f_a_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.f_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.dt_bias", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.A_log", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.o_norm.weight", n->prefix, i);
        }
        if (i < c->first_dense) {
            EMIT(1, "%slayers.%d.mlp.gate_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.up_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.down_proj.weight", n->prefix, i);
        } else {
            EMIT(1, "%slayers.%d.mlp.gate.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.gate.e_score_correction_bias", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.shared_experts.gate_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.shared_experts.up_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.shared_experts.down_proj.weight", n->prefix, i);
            /* Routed experts are streamed, not resident: named here so the
             * inventory can prove they are all present and correctly shaped. */
            for (int e = 0; e < c->n_experts; e++) {
                EMIT(1, "%slayers.%d.mlp.experts.%d.gate_proj.weight", n->prefix, i, e);
                EMIT(1, "%slayers.%d.mlp.experts.%d.up_proj.weight", n->prefix, i, e);
                EMIT(1, "%slayers.%d.mlp.experts.%d.down_proj.weight", n->prefix, i, e);
            }
        }
    }
    if (c->vis_layers) {
        EMIT(1, "%spatch_embed.proj.weight", n->visual);
        EMIT(1, "%spatch_embed.proj.bias", n->visual);
        EMIT(1, "%spost_layernorm.weight", n->visual);
        EMIT(1, "%sdownsample.weight", n->visual);
        EMIT(1, "%sdownsample.bias", n->visual);
        EMIT(1, "%smerger.proj.weight", n->visual);
        EMIT(1, "%smerger.post_projection_norm.weight", n->visual);
        EMIT(1, "%smerger.post_projection_norm.bias", n->visual);
        EMIT(1, "%smerger.gate_proj.weight", n->visual);
        EMIT(1, "%smerger.up_proj.weight", n->visual);
        EMIT(1, "%smerger.down_proj.weight", n->visual);
        for (int b = 0; b < c->vis_layers; b++) {
            EMIT(1, "%sblocks.%d.norm1.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.norm2.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.qkv.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.qkv.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.proj.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.q_norm.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.k_norm.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.gate_proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.gate_proj.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.up_proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.up_proj.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.down_proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.down_proj.bias", n->visual, b);
        }
    }
#undef EMIT
}

#ifdef GLM53_INVENTORY_MAIN_UNUSED
/* Inventory entry point: prints every tensor the engine would load, in load
 * order. Checked against a real checkpoint's index offline, so a naming
 * mistake surfaces before 180 GB have been converted rather than after. */
static void print_name(void *user, const char *name, int required) {
    (void)user; printf("%s\t%d\n", name, required);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    Cfg c;
    load_cfg(&c, argv[1]);
    Names n = { "model.language_model.", "model.visual." };
    glm53_walk_tensors(&c, &n, print_name, NULL);
    return 0;
}
#endif

/* ================= forward =================
 * Il modello caricato in RAM come f32. Lo streaming degli esperti e la
 * quantizzazione a load time arrivano col modello vero; qui la priorita' e'
 * che i token siano quelli giusti, provati contro l'oracolo. */
#include "delta_attention.h"
#include "sparse_index.h"
#include "vision_tower.h"

/* Una matrice residente, nel formato in cui conviene tenerla.
 *
 * Il checkpoint porta i densi in BF16 e gli esperti gia' in int4 gs64. Tenere
 * i densi in f32 vuol dire 39 GB di RAM per il 3% dei parametri, quindi qui si
 * quantizzano al volo secondo GLM53_BITS. Il formato 4 e' lo stesso
 * contenitore che scrive tools/convert_glm53.py e che GLM-5.2 usa gia':
 * `nome` U8 con due valori per byte, `nome.qs` F32 con una scala ogni 64
 * colonne. */
typedef struct {
    int fmt;                              /* 0 = f32, 1 = int8 per riga, 4 = int4 gs64 */
    const float *f;
    const int8_t *q8;
    const uint8_t *q4;
    const float *s;
    int rows, columns, gs;
    void *vk;                             /* ColiVkTensor*, caricata alla prima uso */
} Mat;

typedef struct {
    /* comune */
    const float *in_ln, *post_ln;
    const float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    const float *hc_ffn_fn,  *hc_ffn_base,  *hc_ffn_scale;
    /* KDA */
    Mat kq, kk, kv, ko, kga, kgb, kfa, kfb, kb;
    const float *conv, *dt, *alog, *onorm;
    /* MLA + indexer */
    Mat qa, qb, kva, kvb_kt, kvb_v, o, iwq, iwk, iwp, ikpg;
    const float *qa_ln, *kva_ln, *ik_nw, *ik_nb, *ikpa;
    /* FFN */
    Mat dg, du, dd;                       /* denso */
    Mat rg, ru, rd;                       /* router / shared: gate,up,down */
    const float *router, *rbias;
    Mat *eg, *eu, *ed;                    /* esperti routed */
} GLayer;

/* Quello che una conversazione si porta dietro fra un token e il successivo.
 *
 * Senza, generare il token n costa un passaggio su tutti gli n precedenti, e
 * siccome ogni token attraversa 8 esperti per layer, il conto e' n volte il
 * lavoro che serve. Con la cache il passo costa un token.
 *
 * I layer KDA hanno una ricorrenza: lo stato e la finestra della convoluzione
 * sono gia' tutto quello che serve, e non crescono col contesto. I layer DSA
 * invece devono ricordare chiavi e valori di ogni posizione, piu' le chiavi e i
 * gate dell'indexer, perche' una query nuova puo' guardare ovunque dietro di
 * se'. */
typedef struct {
    float *kda_state;                     /* [teste * k * v] */
    float *kda_window;                    /* [3 * proiezione * kernel] */
    float *latent;                        /* [cap][kv_lora]: MLA assorbita */
    float *ikeys, *igates;                /* [cap][dim indexer] */
} GLayerState;

typedef struct {
    GLayerState *layer;
    float *kda_scratch;
    int filled;                           /* posizioni gia' in cache */
    int cap;
} GSession;

typedef struct {
    Cfg c;
    shards S;
    const float *embed, *final_norm;
    Mat head;
    GLayer *layer;
    char prefix[64];
    /* Quali layer questo motore possiede davvero. Un segment ne carica un
     * pezzo, e caricare il resto vorrebbe dire tenere in RAM i pesi che sta
     * macinando un'altra macchina. */
    int layer_begin, layer_end;
    int has_io;                           /* embedding e testa: solo agli estremi */
    /* esperti: o residenti (checkpoint f32) o in streaming (container int4) */
    int streaming;
    struct ERef *eref;
    struct LCache *ecache;
    int64_t e_len[6], e_at[6], e_slot;
    uint64_t clock, ebytes;
    long hits, miss;
    /* torre vision: presente solo se il checkpoint la porta */
    int has_vision;
    ColiVisionTower vision;
    ColiVisionBlock *vblocks;
} GModel;

static const float *load_f32(GModel *m, const char *fmt, ...) {
    char name[512];
    va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "manca il tensore %s\n", name); exit(1); }
    float *buffer = malloc((size_t)t->numel * sizeof(float));
    if (!buffer) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
    st_read_f32_cap(&m->S, name, buffer, t->numel, 0);
    return buffer;
}

/* Bit dei pesi densi residenti: 4, 8 o 32. Il default e' 4 perche' e' quello
 * che fa stare il modello su una macchina normale; chi vuole la precisione
 * piena la chiede. Gli esperti non passano di qui: arrivano dal disco gia'
 * quantizzati e restano com'erano. */
static int glm53_dense_bits(void) {
    static int cached = 0;
    if (cached) return cached;
    const char *setting = getenv("GLM53_BITS");
    cached = setting ? atoi(setting) : 4;
    if (cached != 4 && cached != 8 && cached != 32) {
        fprintf(stderr, "GLM53_BITS=%s: valori ammessi 4, 8, 32\n", setting);
        exit(1);
    }
    return cached;
}

/* Quantizza [rows, columns] f32 in int4 con una scala ogni `gs` colonne.
 * Stessa aritmetica di quant_int4_grouped() in tools/convert_glm53.py, cosi'
 * un peso quantizzato qui e uno quantizzato dal converter sono lo stesso peso. */
static void quantize_i4_grouped(const float *w, uint8_t *q4, float *scale,
                                int rows, int columns, int gs) {
    const int groups = (columns + gs - 1) / gs;
    const int packed = (columns + 1) / 2;
    for (int r = 0; r < rows; r++) {
        const float *row = w + (size_t)r * columns;
        uint8_t *dst = q4 + (size_t)r * packed;
        memset(dst, 0, (size_t)packed);
        for (int g = 0; g < groups; g++) {
            const int start = g * gs;
            const int stop = start + gs < columns ? start + gs : columns;
            float amax = 0.0f;
            for (int c = start; c < stop; c++) {
                const float value = fabsf(row[c]);
                if (value > amax) amax = value;
            }
            float step = amax / 7.0f;
            if (step < 1e-8f) step = 1e-8f;
            scale[(size_t)r * groups + g] = step;
            for (int c = start; c < stop; c++) {
                int level = (int)lrintf(row[c] / step);
                if (level < -8) level = -8;
                if (level > 7) level = 7;
                const uint8_t nibble = (uint8_t)(level + 8);
                if (c & 1) dst[c >> 1] |= (uint8_t)(nibble << 4);
                else       dst[c >> 1] |= nibble;
            }
        }
    }
}

/* Un blocco f32 gia' in memoria, portato alla precisione chiesta. Possiede il
 * buffer: o lo tiene com'e' o lo libera dopo averlo quantizzato. */
static Mat quantize_loaded(float *buffer, int rows, int columns) {
    Mat mat; memset(&mat, 0, sizeof(mat));
    mat.rows = rows; mat.columns = columns;
    const int bits = glm53_dense_bits();
    if (bits == 32) { mat.fmt = 0; mat.f = buffer; return mat; }
    if (bits == 4 && columns % 64 == 0) {
        const int groups = columns / 64;
        uint8_t *packed = malloc((size_t)rows * ((columns + 1) / 2));
        float *step = malloc((size_t)rows * groups * sizeof(float));
        if (!packed || !step) { fprintf(stderr, "OOM quantizzando %dx%d\n", rows, columns); exit(1); }
        quantize_i4_grouped(buffer, packed, step, rows, columns, 64);
        free(buffer);
        mat.fmt = 4; mat.q4 = packed; mat.s = step; mat.gs = 64;
        return mat;
    }
    /* int8 per riga: e' anche il ripiego quando le colonne non sono multiple
     * di 64, che capita sulle proiezioni piccole dell'indexer. */
    int8_t *level = malloc((size_t)rows * columns);
    float *step = malloc((size_t)rows * sizeof(float));
    if (!level || !step) { fprintf(stderr, "OOM quantizzando %dx%d\n", rows, columns); exit(1); }
    quantize_rows(buffer, level, step, rows, columns, 8);
    free(buffer);
    mat.fmt = 1; mat.q8 = level; mat.s = step;
    return mat;
}

/* Le due forme assorbite di kv_b_proj.
 *
 * MLA comprime chiavi e valori in un latente da kv_lora (512 qui) e li
 * riespande con kv_b_proj. Tenere in cache le chiavi espanse costa
 * teste*qk_nope*2 float per posizione, cioe' 1,39 MB a token su questo
 * modello, undici GB a ottomila token: piu' della macchina.
 *
 * Il latente invece costa 512 float, quarantatre volte meno, e non serve
 * riespanderlo se si piegano le proiezioni nei due estremi. Per una testa h,
 * con W_k e W_v le due meta' del blocco di kv_b_proj:
 *
 *     punteggio_j = q . (W_k c_j) = (W_k^T q) . c_j
 *     uscita      = somma_j a_j (W_v c_j) = W_v (somma_j a_j c_j)
 *
 * A sinistra si moltiplica per ogni posizione in cache, a destra una volta per
 * query. E' un'uguaglianza, non un'approssimazione: cambia solo l'ordine dei
 * prodotti, e con esso quanta memoria serve.
 *
 * Le due matrici si costruiscono qui, da kv_b_proj in f32, e poi passano dallo
 * stesso quantizzatore di tutto il resto. */
static void absorb_kvb(GModel *m, GLayer *l, const char *name) {
    const Cfg *c = &m->c;
    const int H = c->n_heads, QK = c->qk_nope, V = c->v_head, L = c->kv_lora;
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "manca %s\n", name); exit(1); }
    if (t->numel != (int64_t)H * (QK + V) * L) {
        fprintf(stderr, "%s: %lld valori, attesi %lld per %d teste\n", name,
                (long long)t->numel, (long long)H * (QK + V) * L, H);
        exit(1);
    }
    float *whole = malloc((size_t)t->numel * sizeof(float));
    float *kt = malloc((size_t)H * L * QK * sizeof(float));
    float *vv = malloc((size_t)H * V * L * sizeof(float));
    if (!whole || !kt || !vv) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
    st_read_f32_cap(&m->S, name, whole, t->numel, 1);

    for (int h = 0; h < H; h++) {
        const float *block = whole + (size_t)h * (QK + V) * L;
        /* W_k^T: [L, QK], da W_k che e' [QK, L] */
        for (int d = 0; d < L; d++)
            for (int i = 0; i < QK; i++)
                kt[((size_t)h * L + d) * QK + i] = block[(size_t)i * L + d];
        /* W_v: [V, L], gia' nel verso giusto, e' una copia di righe */
        memcpy(vv + (size_t)h * V * L, block + (size_t)QK * L,
               (size_t)V * L * sizeof(float));
    }
    free(whole);
    l->kvb_kt = quantize_loaded(kt, H * L, QK);
    l->kvb_v = quantize_loaded(vv, H * V, L);
}

static Mat load_mat(GModel *m, const char *fmt, ...) {
    char name[512];
    va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "manca la matrice %s\n", name); exit(1); }
    Mat mat; memset(&mat, 0, sizeof(mat));

    /* Gia' quantizzato nel checkpoint: si prende com'e', senza passare per
     * f32. Un giro in f32 costerebbe il picco di RAM che stiamo evitando, e
     * riquantizzare quello che e' gia' quantizzato perde bit per niente. */
    if (t->dtype == 3) {                  /* U8/I8 in st.h: il container int4 */
        char scales[544];
        snprintf(scales, sizeof(scales), "%s.qs", name);
        st_tensor *qs = st_find(&m->S, scales);
        if (!qs) {
            fprintf(stderr, "%s e' int4 ma manca %s\n", name, scales);
            exit(1);
        }
        /* Il contenitore e' piatto: 4.194.304 byte di nibble e 131.072 scale,
         * senza righe ne' colonne scritte da nessuna parte. Va benissimo per
         * gli esperti, che passano dallo streaming e prendono la forma dalla
         * config (moe_inter x hidden, e il down al contrario). Qui invece la
         * forma servirebbe e non c'e': tirarla a indovinare da un solo numero
         * vorrebbe dire calcolare su una matrice trasposta senza accorgersene.
         *
         * Con il converter di oggi questo caso non si presenta, perche' tutto
         * cio' che non e' esperto resta BF16 e la precisione la sceglie
         * GLM53_BITS a load time. Se un domani si quantizzasse anche il resto,
         * la strada e' quella di kimi_k3: la forma la passa il chiamante. */
        const int64_t values = qs->numel * 64;
        if (t->nbytes * 2 != values) {
            fprintf(stderr, "%s: %lld byte e %lld scale non sono un int4 gs64\n",
                    name, (long long)t->nbytes, (long long)qs->numel);
            exit(1);
        }
        if (t->rank != 2) {
            fprintf(stderr, "%s: contenitore int4 piatto fuori dagli esperti; "
                            "la forma non e' nel file e non si indovina\n", name);
            exit(1);
        }
        mat.rows = (int)t->shape[0];
        mat.columns = (int)(values / t->shape[0]);
        if (mat.columns % 64) {
            fprintf(stderr, "%s: %d colonne non sono multiple di 64\n", name, mat.columns);
            exit(1);
        }
        uint8_t *packed = malloc((size_t)t->nbytes);
        float *step = malloc((size_t)qs->numel * sizeof(float));
        if (!packed || !step) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
        st_read_raw(&m->S, name, packed, 1);
        st_read_f32_cap(&m->S, scales, step, qs->numel, 1);
        mat.fmt = 4; mat.q4 = packed; mat.s = step; mat.gs = 64;
        return mat;
    }

    if (t->rank != 2) { fprintf(stderr, "%s: rank %d, attesa 2\n", name, t->rank); exit(1); }
    float *buffer = malloc((size_t)t->numel * sizeof(float));
    if (!buffer) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
    st_read_f32_cap(&m->S, name, buffer, t->numel, 1);
    mat.rows = (int)t->shape[0];
    mat.columns = (int)t->shape[1];

    mat = quantize_loaded(buffer, mat.rows, mat.columns);
    return mat;
}

/* Come mv ma su un blocco di righe contigue: serve alle matrici che tengono
 * una testa dopo l'altra in un unico tensore. */
static void mv_rows(float *out, const Mat *w, const float *x, int row0, int rows) {
    switch (w->fmt) {
    case 4: {
        const int packed = (w->columns + 1) / 2, groups = w->columns / w->gs;
        matmul_i4_grouped(out, x, w->q4 + (size_t)row0 * packed,
                          w->s + (size_t)row0 * groups, 1, w->columns, rows, w->gs);
        break;
    }
    case 1:
        matmul_q(out, x, w->q8 + (size_t)row0 * w->columns, w->s + row0,
                 1, w->columns, rows);
        break;
    default:
        matmul(out, x, w->f + (size_t)row0 * w->columns, 1, w->columns, rows);
        break;
    }
}

/* out[rows] = W x, con W in layout [rows, columns] come transformers.
 *
 * Con COLI_VK=1 le matrici RESIDENTI passano dal backend Vulkan: si caricano
 * sul device alla prima chiamata e restano li'. Gli esperti no, e non e' una
 * dimenticanza: arrivano dal disco a ogni uso, quindi caricarli costerebbe
 * quanto leggerli e il device non ci guadagnerebbe niente. Quelli vogliono un
 * livello residente in VRAM, che e' un'altra cosa.
 *
 * Il campo `vk` e' una cache dentro a una matrice che il resto del codice
 * tratta come sola lettura: da qui il cast, che riguarda solo lui. */
static void mv(float *out, const Mat *w, const float *x) {
#ifdef COLI_VULKAN
    if (g_vk_ready && (w->fmt == 1 || w->fmt == 4)) {
        Mat *mutable_w = (Mat *)w;
        if (coli_vk_matmul((ColiVkTensor **)&mutable_w->vk, out, x,
                           w->fmt == 4 ? (const void *)w->q4 : (const void *)w->q8,
                           w->s, w->fmt, 1, w->columns, w->rows, w->gs))
            return;
    }
#endif
    switch (w->fmt) {
    case 4: matmul_i4_grouped(out, x, w->q4, w->s, 1, w->columns, w->rows, w->gs); break;
    case 1: matmul_q(out, x, w->q8, w->s, 1, w->columns, w->rows); break;
    default: matmul(out, x, w->f, 1, w->columns, w->rows); break;
    }
}

static void rms(float *out, const float *x, const float *w, int n, float eps) {
    float square = 0.0f;
    for (int i = 0; i < n; i++) square += x[i] * x[i];
    float inverse = 1.0f / sqrtf(square / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inverse * w[i];
}

static void layer_norm(float *out, const float *x, const float *w, const float *b,
                       int n, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float variance = 0.0f;
    for (int i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
    variance /= n;
    float inverse = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inverse * w[i] + (b ? b[i] : 0.0f);
}

static float sigmoidf_(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}
static float siluf_(float x) { return x / (1.0f + expf(-x)); }

/* SwiGLU clampata: il gate ha solo il tetto, up e' limitato da entrambi i lati.
 * Vale sia per l'MLP denso che per gli esperti -- il testo di GLM-5.3 NON usa
 * la SiLU semplice, ed e' un errore che darebbe un modello che parla bene e
 * sbaglia. */
static void swiglu_clamped(float *gate, const float *up, int n, float limit) {
    for (int i = 0; i < n; i++) {
        float g = gate[i] > limit ? limit : gate[i];
        float u = up[i] < -limit ? -limit : (up[i] > limit ? limit : up[i]);
        gate[i] = siluf_(g) * u;
    }
}

static void mlp3(float *out, const float *x, const Mat *g, const Mat *u, const Mat *d,
                 float limit, float *sg, float *su) {
    mv(sg, g, x); mv(su, u, x);
    swiglu_clamped(sg, su, g->rows, limit);
    mv(out, d, sg);
}

/* ---------- KDA: proiezioni, gate, ricorrenza ---------- */
static void kda_layer(const Cfg *c, const GLayer *l, const float *x, int tokens,
                      float *out, float *state, float *window, float *scratch) {
    const int P = c->kda_proj, H = c->kda_heads, D = c->kda_hd;
    float *qkv = malloc((size_t)3 * P * sizeof(float));
    float *gate = malloc((size_t)P * sizeof(float));
    float *decay = malloc((size_t)P * sizeof(float));
    float *beta = malloc((size_t)H * sizeof(float));
    float *low = malloc((size_t)D * sizeof(float));
    float *core = malloc((size_t)P * sizeof(float));
    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * c->hidden;
        mv(qkv, &l->kq, row);
        mv(qkv + P, &l->kk, row);
        mv(qkv + 2 * P, &l->kv, row);
        /* decadimento: gate_lower_bound * sigmoid(exp(A_log[h]) * (W_fb W_fa x + dt_bias)) */
        mv(low, &l->kfa, row);
        mv(decay, &l->kfb, low);
        for (int h = 0; h < H; h++)
            for (int d = 0; d < D; d++) {
                int i = h * D + d;
                decay[i] = c->gate_lb * sigmoidf_(expf(l->alog[h]) * (decay[i] + l->dt[i]));
            }
        mv(beta, &l->kb, row);
        for (int h = 0; h < H; h++) beta[h] = sigmoidf_(beta[h]);
        coli_kda_step(core, state, window, qkv, l->conv, decay, beta,
                      H, D, D, c->conv_k, 1e-6f, scratch);
        /* uscita: RMSNorm per testa, pesata da o_norm, moltiplicata dal gate
         * low-rank, poi la proiezione di uscita. */
        mv(low, &l->kga, row);
        mv(gate, &l->kgb, low);
        float *normed = qkv;                         /* riuso: 3P >= P */
        for (int h = 0; h < H; h++) {
            const float *src = core + (size_t)h * D;
            float *dst = normed + (size_t)h * D;
            float square = 0.0f;
            for (int d = 0; d < D; d++) square += src[d] * src[d];
            float inverse = 1.0f / sqrtf(square / D + c->eps);
            for (int d = 0; d < D; d++)
                dst[d] = src[d] * inverse * l->onorm[d] * sigmoidf_(gate[(size_t)h * D + d]);
        }
        mv(out + (size_t)t * c->hidden, &l->ko, normed);
    }
    free(core); free(low); free(beta); free(decay); free(gate); free(qkv);
}

/* ---------- MLA + indexer con k-pool ---------- */
static void mla_layer(const Cfg *c, const GLayer *l, const float *x, int tokens,
                      float *out, GLayerState *st, int base) {
    const int H = c->n_heads, QK = c->qk_nope, V = c->v_head;
    const int IH = c->index_nh, ID = c->index_hd;
    const int seen = base + tokens;
    float *qa = malloc((size_t)tokens * c->q_lora * sizeof(float));
    const int L = c->kv_lora;
    float *queries = malloc((size_t)tokens * H * QK * sizeof(float));
    float *absorbed = malloc((size_t)tokens * H * L * sizeof(float));
    float *latent = st->latent;           /* tutto il prefisso, non solo i nuovi */
    float *iq = malloc((size_t)tokens * IH * ID * sizeof(float));
    float *ik = st->ikeys;
    float *gates = st->igates;
    float *head_w = malloc((size_t)tokens * IH * sizeof(float));
    unsigned char *valid = malloc((size_t)seen);
    memset(valid, 1, (size_t)seen);

    for (int t = 0; t < tokens; t++) {
        const int at = base + t;          /* posizione assoluta nella cache */
        const float *row = x + (size_t)t * c->hidden;
        float *qn = qa + (size_t)t * c->q_lora;
        mv(qn, &l->qa, row);
        rms(qn, qn, l->qa_ln, c->q_lora, c->eps);
        mv(queries + (size_t)t * H * QK, &l->qb, qn);
        float *here = latent + (size_t)at * L;
        mv(here, &l->kva, row);
        rms(here, here, l->kva_ln, L, c->eps);
        /* la query entra nello spazio del latente una volta per testa, invece
         * che il latente nello spazio della query una volta per posizione */
        for (int h = 0; h < H; h++)
            mv_rows(absorbed + ((size_t)t * H + h) * L, &l->kvb_kt,
                    queries + ((size_t)t * H + h) * QK, h * L, L);
        /* indexer: le query vengono dal q_a normalizzato, le chiavi dall'hidden
         * con LayerNorm (con bias), e i pesi per testa sono scalati da IH^-0.5 */
        mv(iq + (size_t)t * IH * ID, &l->iwq, qn);
        float *kraw = ik + (size_t)at * ID;
        mv(kraw, &l->iwk, row);
        layer_norm(kraw, kraw, l->ik_nw, l->ik_nb, ID, 1e-5f);
        mv(gates + (size_t)at * ID, &l->ikpg, row);
        mv(head_w + (size_t)t * IH, &l->iwp, row);
        for (int h = 0; h < IH; h++) head_w[(size_t)t * IH + h] /= sqrtf((float)IH);
    }

    const int width = coli_sparse_index_width(c->index_topk, c->index_kpool, c->index_kpool_tail);
    int *selected = malloc((size_t)tokens * width * sizeof(int));
    if (coli_sparse_index_select_range(selected, iq, ik, gates, head_w, l->ikpa, valid,
                                       seen, IH, ID, c->index_kpool, c->index_topk,
                                       c->index_kpool_tail, base, seen)) {
        fprintf(stderr, "selezione indexer fallita\n"); exit(1);
    }
    /* GLM53_DUMP_INDEX=1 stampa le righe scelte dall'indexer: e' il primo
     * posto da guardare quando il motore diverge solo su certe lunghezze. */
    if (getenv("GLM53_DUMP_INDEX")) {
        for (int t = 0; t < tokens; t++) {
            fprintf(stderr, "index q=%d ->", base + t);
            for (int i = 0; i < width; i++) fprintf(stderr, " %d", selected[(size_t)t * width + i]);
            fprintf(stderr, "\n");
        }
    }
    /* Attenzione nello spazio del latente. La scala resta 1/sqrt(qk_nope):
     * il prodotto e' lo stesso numero di prima, calcolato in un altro ordine. */
    float *context = malloc((size_t)H * V * sizeof(float));
    float *pooled = malloc((size_t)L * sizeof(float));
    float *score = malloc((size_t)width * sizeof(float));
    const float scale = 1.0f / sqrtf((float)QK);
    for (int t = 0; t < tokens; t++) {
        const int *chosen = selected + (size_t)t * width;
        for (int h = 0; h < H; h++) {
            const float *q = absorbed + ((size_t)t * H + h) * L;
            float top = -INFINITY;
            int used = 0;
            for (int i = 0; i < width; i++) {
                const int at = chosen[i];
                if (at < 0 || at >= seen) continue;
                const float *c_j = latent + (size_t)at * L;
                float dot = 0.0f;
                for (int d = 0; d < L; d++) dot += q[d] * c_j[d];
                score[used] = dot * scale;
                if (score[used] > top) top = score[used];
                used++;
            }
            float *result = context + (size_t)h * V;
            memset(result, 0, (size_t)V * sizeof(float));
            if (!used) continue;
            double total = 0.0;
            for (int i = 0; i < used; i++) { score[i] = expf(score[i] - top); total += score[i]; }
            memset(pooled, 0, (size_t)L * sizeof(float));
            int seen_slot = 0;
            for (int i = 0; i < width; i++) {
                const int at = chosen[i];
                if (at < 0 || at >= seen) continue;
                const float weight = (float)(score[seen_slot++] / total);
                const float *c_j = latent + (size_t)at * L;
                for (int d = 0; d < L; d++) pooled[d] += weight * c_j[d];
            }
            mv_rows(result, &l->kvb_v, pooled, h * V, V);
        }
        mv(out + (size_t)t * c->hidden, &l->o, context);
    }
    free(score); free(pooled);

    free(context); free(selected); free(valid); free(head_w);
    free(iq); free(absorbed); free(queries); free(qa);
}

/* ---------- FFN: denso oppure MoE ---------- */
/* ================= streaming degli esperti =================
 *
 * 288 esperti per 42 layer sparsi, 14,2 MB l'uno in int4 gs64: 171 GB, che
 * stanno su disco e non in RAM. Per ogni token se ne toccano 8 per layer, e la
 * stessa manciata torna spesso, quindi ogni layer tiene una cache LRU di slot
 * e il resto arriva con una lettura quando serve.
 *
 * Uno slot e' un blocco unico che contiene i sei pezzi dell'esperto nell'ordine
 * in cui li vuole il calcolo. Le Mat che ne escono puntano dentro allo slot,
 * cosi' il codice del MoE non sa da dove arrivi il peso e resta quello di
 * prima. */
/* Slot per layer chiesti dalla riga di comando; 0 = decide il budget misurato. */
static int g_cap_override = 0;

#define GLM53_EXPERT_PIECES 6

typedef struct ERef {
    int fd[GLM53_EXPERT_PIECES];
    int64_t off[GLM53_EXPERT_PIECES];
    int contig;                           /* i sei pezzi sono consecutivi in un file */
} ERef;

typedef struct { int eid; uint8_t *base; uint64_t used; } Slot;
typedef struct LCache { Slot *s; int n, cap; } LCache;

/* Lunghezze e posizioni dei sei pezzi dentro allo slot. Gate e up sono
 * [moe_inter, hidden], down e' [hidden, moe_inter]: stessi byte, forme
 * scambiate. */
static void expert_geometry(GModel *m) {
    const int64_t hidden = m->c.hidden, inter = m->c.moe_inter;
    const int64_t packed = inter * hidden / 2;
    const int64_t scales = inter * hidden / 64 * (int64_t)sizeof(float);
    const int64_t length[GLM53_EXPERT_PIECES] = { packed, scales, packed, scales, packed, scales };
    int64_t at = 0;
    for (int p = 0; p < GLM53_EXPERT_PIECES; p++) {
        m->e_len[p] = length[p];
        m->e_at[p] = at;
        at += length[p];
    }
    m->e_slot = at;
}

/* Indirizzi su disco di ogni esperto. Le lunghezze dichiarate dal file devono
 * combaciare con la geometria: un checkpoint che dice altro non e' questo
 * modello, e leggerlo lo stesso vorrebbe dire calcolare su byte a caso. */
static void expert_table_init(GModel *m) {
    const Cfg *c = &m->c;
    static const char *piece[GLM53_EXPERT_PIECES] = {
        "gate_proj.weight", "gate_proj.weight.qs",
        "up_proj.weight",   "up_proj.weight.qs",
        "down_proj.weight", "down_proj.weight.qs",
    };
    m->eref = calloc((size_t)c->n_layers * c->n_experts, sizeof(*m->eref));
    if (!m->eref) { fprintf(stderr, "OOM sulla tabella degli esperti\n"); exit(1); }

    const int from = c->first_dense > m->layer_begin ? c->first_dense : m->layer_begin;
    for (int i = from; i < m->layer_end; i++) {
        for (int e = 0; e < c->n_experts; e++) {
            ERef *ref = &m->eref[(size_t)i * c->n_experts + e];
            for (int p = 0; p < GLM53_EXPERT_PIECES; p++) {
                char name[512];
                snprintf(name, sizeof(name), "%slayers.%d.mlp.experts.%d.%s",
                         m->prefix, i, e, piece[p]);
                st_tensor *t = st_find(&m->S, name);
                if (!t) { fprintf(stderr, "manca %s\n", name); exit(1); }
                if (t->nbytes != m->e_len[p]) {
                    fprintf(stderr, "%s: %lld byte, attesi %lld\n", name,
                            (long long)t->nbytes, (long long)m->e_len[p]);
                    exit(1);
                }
                ref->fd[p] = t->fd;
                ref->off[p] = t->off;
            }
            /* Sei letture per esperto costano sei code invece di una. Vale la
             * pena accorgersi quando il file le ha gia' messe in fila. */
            ref->contig = 1;
            for (int p = 1; p < GLM53_EXPERT_PIECES; p++)
                if (ref->fd[p] != ref->fd[0] ||
                    ref->off[p] != ref->off[p - 1] + m->e_len[p - 1]) { ref->contig = 0; break; }
        }
    }
}

/* Quanti slot per layer: il budget diviso i layer sparsi. Il pavimento e' 1 e
 * non topk, perche' un pavimento a topk impegnerebbe topk*layer slot comunque,
 * cioe' molti GB, a dispetto del budget chiesto. */
/* Quanta RAM il sistema dice di poter dare adesso. MemAvailable e non MemFree:
 * la seconda ignora la page cache riutilizzabile e farebbe stimare molto meno
 * di quello che c'e'. */
static double memory_available_gb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    double gb = 0.0;
    while (fgets(line, sizeof(line), f)) {
        long kb;
        if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) { gb = kb / 1048576.0; break; }
    }
    fclose(f);
    return gb;
}

static void expert_cache_init(GModel *m) {
    const Cfg *c = &m->c;
    const char *setting = getenv("GLM53_EXPERT_GB");
    /* Il default si misura: quello che resta libero dopo i pesi residenti,
     * meno un margine per lo stato della conversazione, i temporanei del
     * prefill e il resto del sistema. Un numero fisso sbaglia in entrambi i
     * versi -- su una macchina piccola va in OOM, su una grande lascia RAM
     * inutilizzata mentre il disco fa tutto il lavoro, che e' esattamente
     * quello che e' successo alla prima esecuzione vera. */
    double budget;
    if (setting) budget = atof(setting);
    else {
        const double free_now = memory_available_gb();
        budget = free_now - 3.0;
        if (budget < 1.0) budget = 1.0;
        if (getenv("GLM53_VERBOSE"))
            fprintf(stderr, "budget esperti: %.1f GB (%.1f disponibili, 3 di margine)\n",
                    budget, free_now);
    }
    const int from = c->first_dense > m->layer_begin ? c->first_dense : m->layer_begin;
    int sparse = m->layer_end - from;
    if (sparse < 0) sparse = 0;
    int cap = (int)((budget * 1e9) / ((double)m->e_slot * (sparse > 0 ? sparse : 1)));
    if (g_cap_override > 0) cap = g_cap_override;      /* scelta esplicita: vince */
    if (cap < 1) cap = 1;
    if (cap > c->n_experts) cap = c->n_experts;

    m->ecache = calloc((size_t)c->n_layers, sizeof(*m->ecache));
    if (!m->ecache) { fprintf(stderr, "OOM sulla cache degli esperti\n"); exit(1); }
    for (int i = from; i < m->layer_end; i++) {
        LCache *cache = &m->ecache[i];
        cache->cap = cap;
        cache->s = calloc((size_t)cap, sizeof(*cache->s));
        if (!cache->s) { fprintf(stderr, "OOM sugli slot del layer %d\n", i); exit(1); }
        for (int j = 0; j < cap; j++) cache->s[j].eid = -1;
    }
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "esperti: slot da %.1f MB, %d per layer su %d layer sparsi "
                        "(%.1f GB residenti)\n",
                m->e_slot / 1e6, cap, sparse, (double)cap * sparse * m->e_slot / 1e9);
}

static Slot *slot_find(GModel *m, int layer, int eid) {
    LCache *cache = &m->ecache[layer];
    for (int j = 0; j < cache->n; j++)
        if (cache->s[j].eid == eid) {
            cache->s[j].used = ++m->clock;
            m->hits++;
            return &cache->s[j];
        }
    return NULL;
}

static void expert_read(GModel *m, int layer, int eid, Slot *slot) {
    const ERef *ref = &m->eref[(size_t)layer * m->c.n_experts + eid];
    if (!slot->base) {
        slot->base = malloc((size_t)m->e_slot);
        if (!slot->base) { fprintf(stderr, "OOM su uno slot esperto\n"); exit(1); }
    }
    if (ref->contig) {
        st_pread_full(ref->fd[0], slot->base, m->e_slot, ref->off[0], "expert");
    } else {
        for (int p = 0; p < GLM53_EXPERT_PIECES; p++)
            st_pread_full(ref->fd[p], slot->base + m->e_at[p], m->e_len[p],
                          ref->off[p], "expert piece");
    }
    slot->eid = eid;
    /* expert_read gira dentro a un ciclo parallelo: i contatori sono condivisi
     * e senza questo sarebbero una corsa, cioe' numeri sbagliati proprio nel
     * posto in cui si va a guardare per capire se il riuso funziona. */
#ifdef _OPENMP
#pragma omp atomic
#endif
    m->miss++;
#ifdef _OPENMP
#pragma omp atomic
#endif
    m->ebytes += (uint64_t)m->e_slot;
}

/* Lo slot dell'esperto chiesto, letto se non c'e'. La vittima e' quella usata
 * meno di recente. */
static Slot *expert_slot(GModel *m, int layer, int eid) {
    Slot *slot = slot_find(m, layer, eid);
    if (slot) return slot;
    LCache *cache = &m->ecache[layer];
    if (cache->n < cache->cap) slot = &cache->s[cache->n++];
    else {
        int lru = 0;
        for (int j = 1; j < cache->n; j++)
            if (cache->s[j].used < cache->s[lru].used) lru = j;
        slot = &cache->s[lru];
    }
    expert_read(m, layer, eid, slot);
    slot->used = ++m->clock;
    return slot;
}

/* Le tre matrici di un esperto, che puntano dentro al suo slot. */
static void expert_mats(const GModel *m, const Slot *slot, Mat *gate, Mat *up, Mat *down) {
    const int hidden = m->c.hidden, inter = m->c.moe_inter;
    const Mat shape[3] = {
        { 4, NULL, NULL, slot->base + m->e_at[0], (const float *)(slot->base + m->e_at[1]),
          inter, hidden, 64 },
        { 4, NULL, NULL, slot->base + m->e_at[2], (const float *)(slot->base + m->e_at[3]),
          inter, hidden, 64 },
        { 4, NULL, NULL, slot->base + m->e_at[4], (const float *)(slot->base + m->e_at[5]),
          hidden, inter, 64 },
    };
    *gate = shape[0]; *up = shape[1]; *down = shape[2];
}

/* Il MoE, in due tempi.
 *
 * Prima si decide: per ogni token del blocco quali esperti servono e con che
 * peso. Poi si legge: l'UNIONE di quegli esperti, in parallelo, perche' una
 * lettura da 14,2 MB e' tempo in cui il disco lavora e la CPU no, e farne una
 * per volta e' stato il collo della prima esecuzione vera (2976 letture in
 * fila). Poi si calcola.
 *
 * Fare l'unione paga due volte: le letture vanno insieme, e un esperto che
 * serve a piu' token del blocco si legge una volta sola. */
static void ffn_layer(GModel *m, const GLayer *l, int index, const float *x,
                      int tokens, float *out) {
    const Cfg *c = &m->c;
    const int wide = c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter;

    if (index < c->first_dense) {                 /* layer denso: nessun router */
        float *sg = malloc((size_t)wide * sizeof(float));
        float *su = malloc((size_t)wide * sizeof(float));
        for (int t = 0; t < tokens; t++)
            mlp3(out + (size_t)t * c->hidden, x + (size_t)t * c->hidden,
                 &l->dg, &l->du, &l->dd, c->swiglu_limit, sg, su);
        free(su); free(sg);
        return;
    }

    const int topk = c->topk;
    int *chosen = malloc((size_t)tokens * topk * sizeof(int));
    float *weight = malloc((size_t)tokens * topk * sizeof(float));
    float *score = malloc((size_t)c->n_experts * sizeof(float));
    if (!chosen || !weight || !score) { fprintf(stderr, "OOM nel router\n"); exit(1); }

    /* --- primo tempo: il router, per ogni token --- */
    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * c->hidden;
        for (int e = 0; e < c->n_experts; e++) {
            const float *w = l->router + (size_t)e * c->hidden;
            float sum = 0.0f;
            for (int d = 0; d < c->hidden; d++) sum += w[d] * row[d];
            score[e] = sigmoidf_(sum);
        }
        /* la selezione usa score+bias, il PESO usa lo score puro: la
         * distinzione e' sottile e sbagliarla cambia quali esperti contano
         * quanto. */
        int *mine = chosen + (size_t)t * topk;
        float *mine_w = weight + (size_t)t * topk;
        float total = 0.0f;
        for (int k = 0; k < topk; k++) {
            int best = -1; float value = -INFINITY;
            for (int e = 0; e < c->n_experts; e++) {
                int used = 0;
                for (int j = 0; j < k; j++) if (mine[j] == e) { used = 1; break; }
                float choice = score[e] + (l->rbias ? l->rbias[e] : 0.0f);
                if (!used && choice > value) { value = choice; best = e; }
            }
            mine[k] = best;
            mine_w[k] = score[best];
            total += mine_w[k];
        }
        for (int k = 0; k < topk; k++)
            mine_w[k] = mine_w[k] / (total + 1e-20f) * c->routed_scale;
    }
    free(score);

    /* --- secondo e terzo tempo, a blocchi che stanno in cache ---
     *
     * L'unione di un blocco di prefill puo' superare gli slot che il layer
     * possiede: con 13 slot e sette token si arriva a 56 esperti distinti. Se
     * si assegnassero comunque, due esperti finirebbero sullo stesso slot e il
     * secondo sovrascriverebbe il primo mentre il primo e' ancora in uso, il
     * che non da' errore, da' numeri sbagliati. Quindi si lavora a blocchi
     * grandi al piu' quanto la cache: si legge il blocco in parallelo, si
     * applica a tutti i token, si passa al prossimo. */
    float *sg = malloc((size_t)wide * sizeof(float));
    float *su = malloc((size_t)wide * sizeof(float));
    float *tmp = malloc((size_t)c->hidden * sizeof(float));
    if (!sg || !su || !tmp) { fprintf(stderr, "OOM nel MoE\n"); exit(1); }

    /* l'esperto condiviso e' sempre attivo e non passa dalla cache */
    for (int t = 0; t < tokens; t++)
        mlp3(out + (size_t)t * c->hidden, x + (size_t)t * c->hidden,
             &l->rg, &l->ru, &l->rd, c->swiglu_limit, sg, su);

    if (!m->streaming) {
        for (int t = 0; t < tokens; t++)
            for (int k = 0; k < topk; k++) {
                const int eid = chosen[(size_t)t * topk + k];
                mlp3(tmp, x + (size_t)t * c->hidden, &l->eg[eid], &l->eu[eid], &l->ed[eid],
                     c->swiglu_limit, sg, su);
                const float scale = weight[(size_t)t * topk + k];
                float *dst = out + (size_t)t * c->hidden;
                for (int d = 0; d < c->hidden; d++) dst[d] += scale * tmp[d];
            }
        free(tmp); free(su); free(sg); free(weight); free(chosen);
        return;
    }

    /* unione dei distinti, nell'ordine in cui compaiono */
    int *union_ids = malloc((size_t)tokens * topk * sizeof(int));
    int n_union = 0;
    if (!union_ids) { fprintf(stderr, "OOM sull'unione\n"); exit(1); }
    for (int i = 0; i < tokens * topk; i++) {
        int seen = 0;
        for (int j = 0; j < n_union; j++) if (union_ids[j] == chosen[i]) { seen = 1; break; }
        if (!seen) union_ids[n_union++] = chosen[i];
    }

    LCache *cache = &m->ecache[index];
    const int block = cache->cap;
    int *slot_of = malloc((size_t)block * sizeof(int));
    int *to_read = malloc((size_t)block * sizeof(int));
    if (!slot_of || !to_read) { fprintf(stderr, "OOM sugli slot\n"); exit(1); }

    for (int base = 0; base < n_union; base += block) {
        const int here = base + block <= n_union ? block : n_union - base;
        int reads = 0;
        for (int i = 0; i < here; i++) {
            const int eid = union_ids[base + i];
            Slot *hit = slot_find(m, index, eid);
            if (hit) { slot_of[i] = (int)(hit - cache->s); continue; }
            Slot *victim;
            if (cache->n < cache->cap) victim = &cache->s[cache->n++];
            else {
                int lru = 0;
                for (int j = 1; j < cache->n; j++)
                    if (cache->s[j].used < cache->s[lru].used) lru = j;
                victim = &cache->s[lru];
            }
            /* prenotato subito: cosi' la scelta successiva non lo ripesca */
            victim->used = ++m->clock;
            victim->eid = -1;
            slot_of[i] = (int)(victim - cache->s);
            to_read[reads++] = i;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int r = 0; r < reads; r++) {
            const int i = to_read[r];
            expert_read(m, index, union_ids[base + i], &cache->s[slot_of[i]]);
        }

        /* Un esperto per volta, e per ognuno tutti i token che lo hanno
         * scelto. Nell'ordine opposto i suoi 12,6 MB di pesi verrebbero
         * ripercorsi da capo per ogni token, e a questa taglia la banda di
         * memoria e' quanto costa davvero il calcolo. */
        for (int i = 0; i < here; i++) {
            const int eid = union_ids[base + i];
            Slot *slot = &cache->s[slot_of[i]];
            slot->used = ++m->clock;
            Mat gate, up, down;
            expert_mats(m, slot, &gate, &up, &down);
            for (int t = 0; t < tokens; t++) {
                float scale = 0.0f;
                for (int k = 0; k < topk; k++)
                    if (chosen[(size_t)t * topk + k] == eid) {
                        scale = weight[(size_t)t * topk + k];
                        break;
                    }
                if (scale == 0.0f) continue;          /* non lo ha scelto */
                mlp3(tmp, x + (size_t)t * c->hidden, &gate, &up, &down,
                     c->swiglu_limit, sg, su);
                float *dst = out + (size_t)t * c->hidden;
                for (int d = 0; d < c->hidden; d++) dst[d] += scale * tmp[d];
            }
        }
    }
    free(to_read); free(slot_of); free(union_ids);
    free(tmp); free(su); free(sg); free(weight); free(chosen);
}

/* ---------- caricamento ---------- */
static void vision_load(GModel *m);
static void model_load_range(GModel *m, const char *dir, int b, int e, int io);
static void expert_geometry(GModel *m);
static void expert_table_init(GModel *m);
static void expert_cache_init(GModel *m);

static void model_load_range(GModel *m, const char *dir, int layer_begin,
                             int layer_end, int load_io) {
    load_cfg(&m->c, dir);
    st_init(&m->S, dir);
    /* Il checkpoint reale annida il modello testuale sotto il wrapper vision;
     * un export solo-testo no. Si sceglie una volta, da un tensore che deve
     * esistere in entrambe le forme. */
    snprintf(m->prefix, sizeof(m->prefix), "model.language_model.");
    char probe[256];
    snprintf(probe, sizeof(probe), "%sembed_tokens.weight", m->prefix);
    if (!st_find(&m->S, probe)) snprintf(m->prefix, sizeof(m->prefix), "model.");
    const char *P = m->prefix;

    if (layer_end < 0 || layer_end > m->c.n_layers) layer_end = m->c.n_layers;
    if (layer_begin < 0) layer_begin = 0;
    if (layer_begin > layer_end) layer_begin = layer_end;
    m->layer_begin = layer_begin;
    m->layer_end = layer_end;
    m->has_io = load_io;

    if (load_io) {
        m->embed = load_f32(m, "%sembed_tokens.weight", P);
        m->final_norm = load_f32(m, "%snorm.weight", P);
    }
    /* La testa e' l'unica matrice grande fuori dagli esperti: a vocab 154880
     * per hidden 4096 sono 2,5 GB in f32, quindi passa dallo stesso
     * quantizzatore dei densi. Se il checkpoint la lega all'embedding, si
     * riusa quella tabella cosi' com'e'. Un segment che non tiene l'ultimo
     * layer non la carica proprio: non ha logit da produrre. */
    if (load_io) {
        if (st_find(&m->S, "lm_head.weight")) {
            m->head = load_mat(m, "lm_head.weight");
        } else {
            memset(&m->head, 0, sizeof(m->head));
            m->head.f = m->embed;
            m->head.rows = m->c.vocab;
            m->head.columns = m->c.hidden;
        }
    }
    m->layer = calloc((size_t)m->c.n_layers, sizeof(*m->layer));

    /* Streaming o residenti: lo decide il contenitore, non una variabile
     * d'ambiente. Si guarda il primo esperto del primo layer sparso. */
    int probe_layer = m->c.first_dense > layer_begin ? m->c.first_dense : layer_begin;
    if (probe_layer < layer_end) {
        char first[512];
        snprintf(first, sizeof(first), "%slayers.%d.mlp.experts.0.gate_proj.weight",
                 P, probe_layer);
        st_tensor *probe_expert = st_find(&m->S, first);
        if (!probe_expert) { fprintf(stderr, "manca %s\n", first); exit(1); }
        m->streaming = probe_expert->dtype == 3;
    }
    if (m->streaming) {
        expert_geometry(m);
        expert_table_init(m);
    }

    for (int i = layer_begin; i < layer_end; i++) {
        GLayer *l = &m->layer[i];
        l->in_ln = load_f32(m, "%slayers.%d.input_layernorm.weight", P, i);
        l->post_ln = load_f32(m, "%slayers.%d.post_attention_layernorm.weight", P, i);
        l->hc_attn_fn = load_f32(m, "%slayers.%d.hc_attn_fn", P, i);
        l->hc_attn_base = load_f32(m, "%slayers.%d.hc_attn_base", P, i);
        l->hc_attn_scale = load_f32(m, "%slayers.%d.hc_attn_scale", P, i);
        l->hc_ffn_fn = load_f32(m, "%slayers.%d.hc_ffn_fn", P, i);
        l->hc_ffn_base = load_f32(m, "%slayers.%d.hc_ffn_base", P, i);
        l->hc_ffn_scale = load_f32(m, "%slayers.%d.hc_ffn_scale", P, i);
        if (m->c.is_full[i]) {
            l->qa = load_mat(m, "%slayers.%d.self_attn.q_a_proj.weight", P, i);
            l->qa_ln = load_f32(m, "%slayers.%d.self_attn.q_a_layernorm.weight", P, i);
            l->qb = load_mat(m, "%slayers.%d.self_attn.q_b_proj.weight", P, i);
            l->kva = load_mat(m, "%slayers.%d.self_attn.kv_a_proj_with_mqa.weight", P, i);
            l->kva_ln = load_f32(m, "%slayers.%d.self_attn.kv_a_layernorm.weight", P, i);
            {
                char kvb_name[512];
                snprintf(kvb_name, sizeof(kvb_name),
                         "%slayers.%d.self_attn.kv_b_proj.weight", P, i);
                absorb_kvb(m, l, kvb_name);
            }
            l->o = load_mat(m, "%slayers.%d.self_attn.o_proj.weight", P, i);
            l->iwq = load_mat(m, "%slayers.%d.self_attn.indexer.wq_b.weight", P, i);
            l->iwk = load_mat(m, "%slayers.%d.self_attn.indexer.wk.weight", P, i);
            l->iwp = load_mat(m, "%slayers.%d.self_attn.indexer.weights_proj.weight", P, i);
            l->ik_nw = load_f32(m, "%slayers.%d.self_attn.indexer.k_norm.weight", P, i);
            l->ik_nb = load_f32(m, "%slayers.%d.self_attn.indexer.k_norm.bias", P, i);
            if (m->c.index_kpool > 1) {
                l->ikpa = load_f32(m, "%slayers.%d.self_attn.indexer.index_kpool_compress_ape", P, i);
                l->ikpg = load_mat(m, "%slayers.%d.self_attn.indexer.index_kpool_compress_gate", P, i);
            }
        } else {
            l->kq = load_mat(m, "%slayers.%d.self_attn.q_proj.weight", P, i);
            l->kk = load_mat(m, "%slayers.%d.self_attn.k_proj.weight", P, i);
            l->kv = load_mat(m, "%slayers.%d.self_attn.v_proj.weight", P, i);
            l->ko = load_mat(m, "%slayers.%d.self_attn.o_proj.weight", P, i);
            l->kga = load_mat(m, "%slayers.%d.self_attn.g_a_proj.weight", P, i);
            l->kgb = load_mat(m, "%slayers.%d.self_attn.g_b_proj.weight", P, i);
            l->kfa = load_mat(m, "%slayers.%d.self_attn.f_a_proj.weight", P, i);
            l->kfb = load_mat(m, "%slayers.%d.self_attn.f_b_proj.weight", P, i);
            l->kb = load_mat(m, "%slayers.%d.self_attn.b_proj.weight", P, i);
            l->dt = load_f32(m, "%slayers.%d.self_attn.dt_bias", P, i);
            l->alog = load_f32(m, "%slayers.%d.self_attn.A_log", P, i);
            l->onorm = load_f32(m, "%slayers.%d.self_attn.o_norm.weight", P, i);
            /* Il checkpoint tiene q/k/v conv separate; la ricorrenza le vuole
             * concatenate nello stesso ordine di qkv. */
            {
                int width = m->c.kda_proj * m->c.conv_k;
                float *conv = malloc((size_t)3 * width * sizeof(float));
                const char *parts[3] = { "q_conv1d", "k_conv1d", "v_conv1d" };
                for (int p = 0; p < 3; p++) {
                    const float *piece = load_f32(m, "%slayers.%d.self_attn.%s.weight",
                                                  P, i, parts[p]);
                    memcpy(conv + (size_t)p * width, piece, (size_t)width * sizeof(float));
                    free((void *)piece);
                }
                l->conv = conv;
            }
        }
        if (i < m->c.first_dense) {
            l->dg = load_mat(m, "%slayers.%d.mlp.gate_proj.weight", P, i);
            l->du = load_mat(m, "%slayers.%d.mlp.up_proj.weight", P, i);
            l->dd = load_mat(m, "%slayers.%d.mlp.down_proj.weight", P, i);
        } else {
            l->router = load_f32(m, "%slayers.%d.mlp.gate.weight", P, i);
            l->rbias = st_find(&m->S, (snprintf(probe, sizeof(probe),
                        "%slayers.%d.mlp.gate.e_score_correction_bias", P, i), probe))
                       ? load_f32(m, "%s", probe) : NULL;
            l->rg = load_mat(m, "%slayers.%d.mlp.shared_experts.gate_proj.weight", P, i);
            l->ru = load_mat(m, "%slayers.%d.mlp.shared_experts.up_proj.weight", P, i);
            l->rd = load_mat(m, "%slayers.%d.mlp.shared_experts.down_proj.weight", P, i);
            /* Gli esperti quantizzati restano su disco: sono il 97% dei byte
             * e nessuna macchina li tiene in RAM. Un checkpoint f32, come le
             * fixture degli oracoli, e' piccolo e si carica tutto. */
            if (!m->streaming) {
                l->eg = malloc((size_t)m->c.n_experts * sizeof(Mat));
                l->eu = malloc((size_t)m->c.n_experts * sizeof(Mat));
                l->ed = malloc((size_t)m->c.n_experts * sizeof(Mat));
                for (int e = 0; e < m->c.n_experts; e++) {
                    l->eg[e] = load_mat(m, "%slayers.%d.mlp.experts.%d.gate_proj.weight", P, i, e);
                    l->eu[e] = load_mat(m, "%slayers.%d.mlp.experts.%d.up_proj.weight", P, i, e);
                    l->ed[e] = load_mat(m, "%slayers.%d.mlp.experts.%d.down_proj.weight", P, i, e);
                }
            }
        }
    }
    vision_load(m);
#ifdef COLI_VULKAN
    /* Il device si apre dopo i pesi: se non c'e', il motore continua sulla CPU
     * senza dire niente di piu' di una riga, perche' Vulkan qui e' un'opzione
     * e non un requisito. */
    if (getenv("COLI_VULKAN") && atoi(getenv("COLI_VULKAN"))) {
        /* Il backend vuole il file qmatmul.spv e da li' ricava i fratelli.
         * COLI_VK_SHADERS puo' essere il file o la cartella che lo contiene,
         * come nel resto del progetto; senza, si guarda accanto al binario. */
        char spv[1024];
        const char *given = getenv("COLI_VK_SHADERS");
        if (given && strstr(given, ".spv")) snprintf(spv, sizeof(spv), "%s", given);
        else snprintf(spv, sizeof(spv), "%s/qmatmul.spv", given ? given : "shaders");
        g_vk_ready = coli_vk_init(spv) && coli_vk_available();
        fprintf(stderr, g_vk_ready
                ? "Vulkan: attivo sulle matrici residenti\n"
                : "Vulkan: nessun device utilizzabile (%s), resto su CPU\n", spv);
    }
#endif
    /* La cache si dimensiona qui, non prima: quanto si puo' spendere dipende
     * da quanto hanno gia' preso i pesi, e prima del ciclo sui layer non
     * l'avevano ancora preso. */
    if (m->streaming) expert_cache_init(m);
}

/* ---------- vision ----------
 * La torre e' in vision_tower.h e non sa nulla di GLM: qui si riempiono solo i
 * puntatori ai pesi e si traduce la config. Il checkpoint solo-testo non porta
 * `model.visual.*` e allora has_vision resta 0: l'engine funziona identico. */
static void vision_load(GModel *m) {
    const Cfg *c = &m->c;
    m->has_vision = 0;
    if (c->vis_layers <= 0) return;
    if (!st_find(&m->S, "model.visual.patch_embed.proj.weight")) return;
    const char *V = "model.visual.";

    m->vision.config = (ColiVisionConfig){
        .depth = c->vis_layers, .hidden = c->vis_hidden, .heads = c->vis_heads,
        .head_dim = c->vis_hidden / c->vis_heads, .intermediate = c->vis_inter,
        .patch = c->vis_patch, .temporal = c->vis_temporal, .merge = c->vis_merge,
        .in_channels = c->vis_in_ch, .out_hidden = c->vis_out_hidden,
        .proj_intermediate = c->vis_proj_inter ? c->vis_proj_inter : c->vis_inter,
        .eps = c->vis_eps, .swiglu_limit = c->vis_swiglu_limit,
        .rope_theta = 10000.0f,
    };
    m->vision.patch_w = load_f32(m, "%spatch_embed.proj.weight", V);
    m->vision.patch_b = load_f32(m, "%spatch_embed.proj.bias", V);
    m->vision.post_norm = load_f32(m, "%spost_layernorm.weight", V);
    m->vision.down_w = load_f32(m, "%sdownsample.weight", V);
    m->vision.down_b = load_f32(m, "%sdownsample.bias", V);
    m->vision.merger_proj = load_f32(m, "%smerger.proj.weight", V);
    m->vision.merger_norm_w = load_f32(m, "%smerger.post_projection_norm.weight", V);
    m->vision.merger_norm_b = load_f32(m, "%smerger.post_projection_norm.bias", V);
    m->vision.merger_gate = load_f32(m, "%smerger.gate_proj.weight", V);
    m->vision.merger_up = load_f32(m, "%smerger.up_proj.weight", V);
    m->vision.merger_down = load_f32(m, "%smerger.down_proj.weight", V);

    m->vblocks = calloc((size_t)c->vis_layers, sizeof(*m->vblocks));
    if (!m->vblocks) { fprintf(stderr, "OOM sui blocchi vision\n"); exit(1); }
    for (int b = 0; b < c->vis_layers; b++) {
        ColiVisionBlock *vb = &m->vblocks[b];
        vb->norm1 = load_f32(m, "%sblocks.%d.norm1.weight", V, b);
        vb->norm2 = load_f32(m, "%sblocks.%d.norm2.weight", V, b);
        vb->qkv_w = load_f32(m, "%sblocks.%d.attn.qkv.weight", V, b);
        vb->qkv_b = load_f32(m, "%sblocks.%d.attn.qkv.bias", V, b);
        vb->q_norm = load_f32(m, "%sblocks.%d.attn.q_norm.weight", V, b);
        vb->k_norm = load_f32(m, "%sblocks.%d.attn.k_norm.weight", V, b);
        vb->proj_w = load_f32(m, "%sblocks.%d.attn.proj.weight", V, b);
        vb->proj_b = load_f32(m, "%sblocks.%d.attn.proj.bias", V, b);
        vb->gate_w = load_f32(m, "%sblocks.%d.mlp.gate_proj.weight", V, b);
        vb->gate_b = load_f32(m, "%sblocks.%d.mlp.gate_proj.bias", V, b);
        vb->up_w = load_f32(m, "%sblocks.%d.mlp.up_proj.weight", V, b);
        vb->up_b = load_f32(m, "%sblocks.%d.mlp.up_proj.bias", V, b);
        vb->down_w = load_f32(m, "%sblocks.%d.mlp.down_proj.weight", V, b);
        vb->down_b = load_f32(m, "%sblocks.%d.mlp.down_proj.bias", V, b);
    }
    m->vision.blocks = m->vblocks;
    m->has_vision = 1;
}

/* Un'immagine gia' in patch -> embedding pronti per il flusso testuale.
 *
 * `patches` e' [grid_h*grid_w, in_channels*temporal*patch*patch] nell'ordine in
 * cui il processore li produce; la torre restituisce
 * grid_h/merge * grid_w/merge righe da out_hidden. Il chiamante possiede il
 * buffer restituito. */
static float *vision_encode(GModel *m, const float *patches,
                            int grid_h, int grid_w, int *out_tokens) {
    if (!m->has_vision) {
        fprintf(stderr, "questo checkpoint non porta la torre vision\n");
        exit(1);
    }
    const int tokens = coli_vision_output_tokens(&m->vision.config, grid_h, grid_w);
    if (tokens <= 0) {
        fprintf(stderr, "griglia %dx%d non divisibile per merge %d\n",
                grid_h, grid_w, m->vision.config.merge);
        exit(1);
    }
    float *out = malloc((size_t)tokens * m->vision.config.out_hidden * sizeof(float));
    if (!out) { fprintf(stderr, "OOM sugli embedding vision\n"); exit(1); }
    if (coli_vision_forward(out, &m->vision, patches, grid_h, grid_w) != 0) {
        fprintf(stderr, "la torre vision ha rifiutato l'ingresso\n");
        exit(1);
    }
    /* La torre esce a out_hidden; il flusso testuale vuole hidden. Il
     * checkpoint li dichiara uguali (4096) e il merger e' proprio il pezzo che
     * fa combaciare i due, quindi una differenza qui e' una config sbagliata,
     * non qualcosa da rattoppare in silenzio. */
    if (m->vision.config.out_hidden != m->c.hidden) {
        fprintf(stderr, "vision out_hidden %d != hidden %d\n",
                m->vision.config.out_hidden, m->c.hidden);
        exit(1);
    }
    *out_tokens = tokens;
    return out;
}

/* ---------- sessione ---------- */
static GSession *session_open(const GModel *m, int cap) {
    const Cfg *c = &m->c;
    GSession *s = calloc(1, sizeof(*s));
    if (!s) { fprintf(stderr, "OOM sulla sessione\n"); exit(1); }
    s->cap = cap;
    s->layer = calloc((size_t)c->n_layers, sizeof(*s->layer));
    if (!s->layer) { fprintf(stderr, "OOM sugli stati di layer\n"); exit(1); }
    if (c->kda_proj)
        s->kda_scratch = malloc((size_t)coli_kda_scratch_floats(c->kda_heads, c->kda_hd,
                                                                c->kda_hd) * sizeof(float));
    for (int i = 0; i < c->n_layers; i++) {
        GLayerState *st = &s->layer[i];
        if (c->is_full[i]) {
            st->latent = malloc((size_t)cap * c->kv_lora * sizeof(float));
            st->ikeys = malloc((size_t)cap * c->index_hd * sizeof(float));
            st->igates = malloc((size_t)cap * c->index_hd * sizeof(float));
            if (!st->latent || !st->ikeys || !st->igates) {
                fprintf(stderr, "OOM sulla cache del layer %d\n", i); exit(1);
            }
        } else if (c->kda_proj) {
            st->kda_state = calloc((size_t)c->kda_heads * c->kda_hd * c->kda_hd,
                                   sizeof(float));
            st->kda_window = calloc((size_t)3 * c->kda_proj * c->conv_k, sizeof(float));
            if (!st->kda_state || !st->kda_window) {
                fprintf(stderr, "OOM sullo stato KDA del layer %d\n", i); exit(1);
            }
        }
    }
    if (getenv("GLM53_VERBOSE")) {
        int full = 0;
        for (int i = 0; i < c->n_layers; i++) if (c->is_full[i]) full++;
        const double per_token = (double)full * (c->kv_lora + 2 * c->index_hd) * sizeof(float);
        fprintf(stderr, "cache: %.1f KB per token su %d layer DSA "
                        "(%.2f GB a %d posizioni)\n",
                per_token / 1024.0, full, per_token * cap / 1e9, cap);
    }
    return s;
}

static void session_close(const GModel *m, GSession *s) {
    if (!s) return;
    for (int i = 0; i < m->c.n_layers; i++) {
        GLayerState *st = &s->layer[i];
        free(st->latent); free(st->ikeys); free(st->igates);
        free(st->kda_state); free(st->kda_window);
    }
    free(s->kda_scratch);
    free(s->layer);
    free(s);
}

/* I layer [begin, end) su `streams`, che entra e esce come H flussi residui
 * per posizione. E' il pezzo che un segment esegue: prima e dopo ci sono
 * l'embedding e la testa, che stanno agli estremi della catena e non in mezzo.
 *
 * `next` e' il secondo banco, della stessa misura: il passaggio li scambia a
 * ogni sito, quindi alla fine il risultato puo' essere in uno o nell'altro, e
 * la funzione restituisce quale. */
static float *run_layers(GModel *m, GSession *s, float *streams, float *next,
                         int n, int start, int begin, int end) {
    const Cfg *c = &m->c;
    const int H = c->hc_mult, D = c->hidden;
    float *collapsed = malloc((size_t)n * D * sizeof(float));
    float *normed = malloc((size_t)n * D * sizeof(float));
    float *branch = malloc((size_t)n * D * sizeof(float));
    float *post = malloc((size_t)n * H * sizeof(float));
    float *comb = malloc((size_t)n * H * H * sizeof(float));
    if (!collapsed || !normed || !branch || !post || !comb) {
        fprintf(stderr, "OOM nei temporanei del passaggio\n"); exit(1);
    }

    for (int i = begin; i < end; i++) {
        GLayer *l = &m->layer[i];
        for (int site = 0; site < 2; site++) {
            const float *fn = site ? l->hc_ffn_fn : l->hc_attn_fn;
            const float *base = site ? l->hc_ffn_base : l->hc_attn_base;
            const float *scale = site ? l->hc_ffn_scale : l->hc_attn_scale;
            for (int t = 0; t < n; t++)
                coli_hc_pre(collapsed + (size_t)t * D, post + (size_t)t * H,
                            comb + (size_t)t * H * H, streams + (size_t)t * H * D,
                            fn, scale, base, H, D, c->hc_iters, c->eps, c->hc_eps);
            for (int t = 0; t < n; t++)
                rms(normed + (size_t)t * D, collapsed + (size_t)t * D,
                    site ? l->post_ln : l->in_ln, D, c->eps);
            if (!site) {
                GLayerState *st = &s->layer[i];
                /* Lo stato non si azzera a ogni chiamata: e' della
                 * conversazione, e azzerarlo qui vorrebbe dire ricominciare
                 * la ricorrenza a ogni token generato. */
                if (c->is_full[i]) mla_layer(c, l, normed, n, branch, st, start);
                else kda_layer(c, l, normed, n, branch, st->kda_state, st->kda_window,
                               s->kda_scratch);
            } else {
                ffn_layer(m, l, i, normed, n, branch);
            }
            for (int t = 0; t < n; t++)
                coli_hc_post(next + (size_t)t * H * D, branch + (size_t)t * D,
                             streams + (size_t)t * H * D, post + (size_t)t * H,
                             comb + (size_t)t * H * H, H, D);
            float *swap = streams; streams = next; next = swap;
        }
    }
    free(comb); free(post); free(branch); free(normed); free(collapsed);
    return streams;
}

/* Rilascio completo del modello.
 *
 * Una CLI che finisce lascia fare al sistema operativo; un ospite che tiene
 * piu' motori nello stesso processo no, e un motore che non si smonta diventa
 * una perdita per richiesta. Ogni allocazione fatta dal caricamento ha qui il
 * suo rilascio, l'indice dei tensori compreso. */
static void mat_release(Mat *mat) {
    free((void *)mat->f); free((void *)mat->q8);
    free((void *)mat->q4); free((void *)mat->s);
    memset(mat, 0, sizeof(*mat));
}

static void model_release(GModel *m) {
    if (!m) return;
    if (m->layer) {
        for (int i = m->layer_begin; i < m->layer_end; i++) {
            GLayer *l = &m->layer[i];
            Mat *mats[] = { &l->kq, &l->kk, &l->kv, &l->ko, &l->kga, &l->kgb,
                            &l->kfa, &l->kfb, &l->kb, &l->qa, &l->qb, &l->kva,
                            &l->kvb_kt, &l->kvb_v, &l->o, &l->iwq, &l->iwk,
                            &l->iwp, &l->ikpg, &l->dg, &l->du, &l->dd,
                            &l->rg, &l->ru, &l->rd };
            for (size_t k = 0; k < sizeof(mats) / sizeof(*mats); k++) mat_release(mats[k]);
            const float *vectors[] = { l->in_ln, l->post_ln, l->hc_attn_fn,
                                       l->hc_attn_base, l->hc_attn_scale,
                                       l->hc_ffn_fn, l->hc_ffn_base, l->hc_ffn_scale,
                                       l->conv, l->dt, l->alog, l->onorm,
                                       l->qa_ln, l->kva_ln, l->ik_nw, l->ik_nb,
                                       l->ikpa, l->router, l->rbias };
            for (size_t k = 0; k < sizeof(vectors) / sizeof(*vectors); k++)
                free((void *)vectors[k]);
            if (!m->streaming && l->eg) {
                for (int e = 0; e < m->c.n_experts; e++) {
                    mat_release(&l->eg[e]); mat_release(&l->eu[e]); mat_release(&l->ed[e]);
                }
            }
            free(l->eg); free(l->eu); free(l->ed);
        }
        free(m->layer);
    }
    if (m->ecache) {
        for (int i = 0; i < m->c.n_layers; i++) {
            LCache *cache = &m->ecache[i];
            for (int j = 0; j < cache->cap; j++) free(cache->s[j].base);
            free(cache->s);
        }
        free(m->ecache);
    }
    free(m->eref);
    if (m->vblocks) {
        for (int b = 0; b < m->c.vis_layers; b++) {
            ColiVisionBlock *vb = &m->vblocks[b];
            const float *parts[] = { vb->norm1, vb->norm2, vb->qkv_w, vb->qkv_b,
                                     vb->q_norm, vb->k_norm, vb->proj_w, vb->proj_b,
                                     vb->gate_w, vb->gate_b, vb->up_w, vb->up_b,
                                     vb->down_w, vb->down_b };
            for (size_t k = 0; k < sizeof(parts) / sizeof(*parts); k++)
                free((void *)parts[k]);
        }
        free(m->vblocks);
        const float *tower[] = { m->vision.patch_w, m->vision.patch_b,
                                 m->vision.post_norm, m->vision.down_w,
                                 m->vision.down_b, m->vision.merger_proj,
                                 m->vision.merger_norm_w, m->vision.merger_norm_b,
                                 m->vision.merger_gate, m->vision.merger_up,
                                 m->vision.merger_down };
        for (size_t k = 0; k < sizeof(tower) / sizeof(*tower); k++) free((void *)tower[k]);
    }
    /* La testa puo' essere la tabella degli embedding: liberarla due volte
     * sarebbe un doppio free, non un risparmio. */
    if (m->head.f != m->embed) mat_release(&m->head);
    free((void *)m->embed);
    free((void *)m->final_norm);
    st_destroy(&m->S);
    memset(m, 0, sizeof(*m));
}

/* Il caso pieno: tutti i layer, embedding e testa comprese. */
static void model_load(GModel *m, const char *dir) {
    model_load_range(m, dir, 0, -1, 1);
}

/* ---------- il passaggio completo ----------
 *
 * `vision` sono gli embedding usciti dalla torre, `n_vision` quanti sono: uno
 * per ogni token immagine presente in `tokens`, nello stesso ordine. Il
 * processore ha gia' espanso ogni immagine in altrettanti segnaposto
 * `image_token_id`, quindi qui non c'e' nulla da inserire: si sostituisce la
 * riga dell'embedding testuale con quella della torre e le posizioni restano
 * quelle che sono. Prompt di solo testo passano vision=NULL, n_vision=0. */
static float *forward_span(GModel *m, GSession *s, const int *tokens, int n,
                           const float *vision, int n_vision) {
    const Cfg *c = &m->c;
    const int start = s->filled;   /* NON 'base': nel ciclo dei layer e' gia' preso */
    if (start + n > s->cap) {
        fprintf(stderr, "contesto esaurito: %d posizioni su %d\n", start + n, s->cap);
        exit(1);
    }
    const int H = c->hc_mult, D = c->hidden;
    float *streams = malloc((size_t)n * H * D * sizeof(float));
    float *next = malloc((size_t)n * H * D * sizeof(float));
    /* l'embedding entra replicato in ognuno degli H flussi residui; sui token
     * immagine la riga viene dalla torre invece che dalla tabella */
    int consumed = 0;
    for (int t = 0; t < n; t++) {
        const float *row;
        /* Un id fuori dal vocabolario legge oltre la tabella degli embedding.
         * Da CLI sarebbe un errore di battitura; da server e' input di rete. */
        if (tokens[t] < 0 || tokens[t] >= c->vocab) {
            fprintf(stderr, "token %d fuori dal vocabolario (0..%d)\n",
                    tokens[t], c->vocab - 1);
            exit(1);
        }
        if (vision && c->image_token >= 0 && tokens[t] == c->image_token) {
            if (consumed >= n_vision) {
                fprintf(stderr, "il prompt ha piu' token immagine (%d+) degli "
                                "embedding forniti (%d)\n", consumed + 1, n_vision);
                exit(1);
            }
            row = vision + (size_t)consumed++ * D;
        } else {
            row = m->embed + (size_t)tokens[t] * D;
        }
        for (int h = 0; h < H; h++)
            memcpy(streams + ((size_t)t * H + h) * D, row, (size_t)D * sizeof(float));
    }
    /* Avanzare in silenzio con embedding non consumati vorrebbe dire mandare al
     * modello un'immagine diversa da quella che il chiamante crede di aver
     * passato: e' un errore, non un caso limite. */
    if (consumed != n_vision) {
        fprintf(stderr, "%d embedding vision forniti ma solo %d token immagine "
                        "nel prompt\n", n_vision, consumed);
        exit(1);
    }

    streams = run_layers(m, s, streams, next, n, start,
                         m->layer_begin, m->layer_end);

    float *collapsed = malloc((size_t)n * D * sizeof(float));
    float *normed = malloc((size_t)n * D * sizeof(float));
    if (!collapsed || !normed) { fprintf(stderr, "OOM in chiusura\n"); exit(1); }

    /* i flussi si richiudono con una media NON pesata */
    for (int t = 0; t < n; t++)
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int h = 0; h < H; h++) sum += streams[((size_t)t * H + h) * D + d];
            collapsed[(size_t)t * D + d] = sum / H;
        }
    for (int t = 0; t < n; t++)
        rms(normed + (size_t)t * D, collapsed + (size_t)t * D, m->final_norm, D, c->eps);

    float *logits = malloc((size_t)n * c->vocab * sizeof(float));
    for (int t = 0; t < n; t++)
        mv(logits + (size_t)t * c->vocab, &m->head, normed + (size_t)t * D);

    free(normed); free(collapsed);
    free(next); free(streams);
    s->filled = start + n;
    return logits;
}

/* Prefill a pezzi.
 *
 * Un prefill in un colpo solo tiene in memoria i temporanei di tutto il
 * prompt: due banchi di flussi residui per hc_mult, piu' le proiezioni di
 * ogni layer. Su GLM-5.3-Flash sono quasi quattro GB per ottomila token, che
 * e' assurdo per un motore che esiste per stare stretto. A pezzi il lavoro e'
 * lo stesso e i temporanei costano quanto il pezzo.
 *
 * Il pezzo resta grande abbastanza da non buttare via il vantaggio del
 * prefill: gli esperti si leggono da disco una volta per pezzo e per layer,
 * quindi pezzi minuscoli li rileggerebbero in continuazione.
 *
 * Con `keep_all` si tengono i logit di ogni posizione, che serve solo al
 * confronto con l'oracolo; altrimenti si tiene l'ultima riga, che e' l'unica
 * che decide il token successivo. */
static float *forward_prefill(GModel *m, GSession *s, const int *tokens, int n,
                              const float *vision, int n_vision, int keep_all) {
    const Cfg *c = &m->c;
    const char *setting = getenv("GLM53_PREFILL_CHUNK");
    int chunk = setting ? atoi(setting) : 128;
    if (chunk < 1) chunk = 1;
    if (chunk > n) chunk = n;

    float *all = keep_all ? malloc((size_t)n * c->vocab * sizeof(float)) : NULL;
    if (keep_all && !all) { fprintf(stderr, "OOM sui logit del prefill\n"); exit(1); }
    float *last = NULL;
    int used_vision = 0;

    for (int at = 0; at < n; at += chunk) {
        const int here = at + chunk <= n ? chunk : n - at;
        /* Gli embedding dell'immagine vanno divisi come i token: a ogni pezzo
         * quelli dei segnaposto che contiene, altrimenti il conto non torna e
         * il motore si ferma -- che e' quello che deve fare. */
        int mine = 0;
        if (vision && c->image_token >= 0)
            for (int i = 0; i < here; i++)
                if (tokens[at + i] == c->image_token) mine++;
        float *part = forward_span(m, s, tokens + at, here,
                                   vision ? vision + (size_t)used_vision * c->hidden : NULL,
                                   mine);
        used_vision += mine;
        if (keep_all) {
            memcpy(all + (size_t)at * c->vocab, part,
                   (size_t)here * c->vocab * sizeof(float));
            free(part);
        } else {
            free(last);
            last = part;
            if (here > 1) {
                /* si tiene solo l'ultima riga */
                float *tail = malloc((size_t)c->vocab * sizeof(float));
                if (!tail) { fprintf(stderr, "OOM sui logit\n"); exit(1); }
                memcpy(tail, last + (size_t)(here - 1) * c->vocab,
                       (size_t)c->vocab * sizeof(float));
                free(last);
                last = tail;
            }
        }
    }
    if (vision && used_vision != n_vision) {
        fprintf(stderr, "%d embedding vision forniti ma %d segnaposto nel prompt\n",
                n_vision, used_vision);
        exit(1);
    }
    return keep_all ? all : last;
}

/* Il passaggio senza sessione: apre, macina tutto, chiude. E' quello che usano
 * gli oracoli, dove il punto e' il risultato e non il tempo. */
static float *forward(GModel *m, const int *tokens, int n,
                      const float *vision, int n_vision) {
    GSession *s = session_open(m, n);
    float *logits = forward_span(m, s, tokens, n, vision, n_vision);
    session_close(m, s);
    return logits;
}

/* ---------- CLI ----------
 * Due modi, entrambi pensati per essere confrontati con ref.json:
 *   --ids a,b,c            teacher forcing: stampa l'argmax a ogni posizione
 *   --ids a,b,c --greedy N genera N token continuando dal prompt
 * Il tokenizzatore non serve: l'oracolo parla in id. */
static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* Gli id di fine generazione. GLM ne dichiara piu' di uno (fine turno, fine
 * testo, fine blocco strumenti) e fermarsi solo sul primo vuol dire vedere il
 * modello continuare a parlare oltre la sua risposta. */
static int load_stops(const char *dir, int *out, int max) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/generation_config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, f) != (size_t)size) {
        free(text); fclose(f); return 0;
    }
    text[size] = 0; fclose(f);
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    int found = 0;
    if (root && root->t == J_OBJ) {
        jval *eos = json_get(root, "eos_token_id");
        if (eos && eos->t == J_NUM && found < max) out[found++] = (int)eos->num;
        else if (eos && eos->t == J_ARR)
            for (int i = 0; i < eos->len && found < max; i++)
                if (eos->kids[i]->t == J_NUM) out[found++] = (int)eos->kids[i]->num;
    }
    free(arena);
    free(text);
    return found;
}

/* ================= protocollo serve =================
 *
 * Due protocolli sulla stessa pipa, come gli altri motori (docs/serve_protocol.md):
 * il mux con SERVE_BATCH=1, che e' quello che parlano openai_server.py e
 * `coli web`, e quello interattivo con SERVE=1 da solo, che usa `coli chat`.
 * Le righe escono una per write con fflush, e su Windows i due capi vanno messi
 * in binario prima di tutto: la traduzione CRLF del CRT rovina i sentinelli e
 * blocca le letture a byte contati (#195).
 *
 * Le richieste portano un indice di slot KV. Qui viene accettato e non usato:
 * questo motore riprefilla ogni volta invece di riprendere la conversazione da
 * dove era. E' piu' lento e non e' sbagliato, e il giorno che ci sara' una
 * cache il protocollo non cambia. */
static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static double rss_gb(void) {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1e6;             /* ru_maxrss e' in KB anche su Windows */
}

static float g_temp = 0.0f, g_topp = 1.0f;

/* Confronto per qsort. Una funzione annidata sarebbe piu' comoda ma e'
 * un'estensione GCC, e questo file deve compilare anche con MSVC. */
static const float *g_sort_key = NULL;
static int compare_by_probability(const void *a, const void *b) {
    const int x = *(const int *)a, y = *(const int *)b;
    if (g_sort_key[x] < g_sort_key[y]) return 1;
    if (g_sort_key[x] > g_sort_key[y]) return -1;
    return 0;
}

/* Temperatura piu' nucleus. Con temperatura nulla e' argmax e basta. */
static int sample_token(const float *logits, int vocab) {
    if (g_temp <= 1e-4f) return argmax(logits, vocab);
    static float *probability = NULL;
    static int *order = NULL;
    static int room = 0;
    if (room < vocab) {
        probability = realloc(probability, (size_t)vocab * sizeof(float));
        order = realloc(order, (size_t)vocab * sizeof(int));
        room = vocab;
        if (!probability || !order) { fprintf(stderr, "OOM nel campionamento\n"); exit(1); }
    }
    float top = -INFINITY;
    for (int i = 0; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    const float inverse = 1.0f / g_temp;
    for (int i = 0; i < vocab; i++) {
        probability[i] = expf((logits[i] - top) * inverse);
        total += probability[i];
        order[i] = i;
    }
    for (int i = 0; i < vocab; i++) probability[i] = (float)(probability[i] / total);

    /* nucleus: si ordina per probabilita' e si taglia dove la somma supera
     * top_p. Con vocab 154880 l'ordinamento pieno costa, ma succede una volta
     * per token e il token costa incomparabilmente di piu'. */
    g_sort_key = probability;
    qsort(order, (size_t)vocab, sizeof(int), compare_by_probability);
    double cumulative = 0.0;
    int keep = vocab;
    for (int i = 0; i < vocab; i++) {
        cumulative += probability[order[i]];
        if (cumulative >= g_topp) { keep = i + 1; break; }
    }
    double pick = ((double)rand() / ((double)RAND_MAX + 1.0)) * cumulative;
    double walk = 0.0;
    for (int i = 0; i < keep; i++) {
        walk += probability[order[i]];
        if (walk >= pick) return order[i];
    }
    return order[0];
}

/* ---------- slot KV ----------
 *
 * Un turno di chat rende tutta la conversazione da capo, quindi il prompt del
 * secondo turno comincia con quello del primo piu' la risposta che il motore
 * ha appena dato. Tenere la sessione dello slot e ripartire da dove il prefisso
 * smette di combaciare fa pagare solo la parte nuova.
 *
 * Il riuso vale solo in avanti. Lo stato ricorrente dei layer KDA non si
 * riavvolge: la cache DSA si potrebbe troncare, essendo posizionale, ma la
 * ricorrenza no, e fingere il contrario darebbe risposte sbagliate in silenzio.
 * Se il prompt nuovo non estende quello vecchio, la sessione si rifa'. */
#define GLM53_MAX_SLOTS 16

typedef struct {
    GSession *session;
    int *tokens;                          /* la sequenza che lo slot tiene */
    int n, cap;
} KVSlot;

static KVSlot g_slots[GLM53_MAX_SLOTS];
static int g_n_slots = 0;
static int g_slot_context = 0;

static void slots_init(const GModel *m) {
    const char *setting = getenv("KV_SLOTS");
    g_n_slots = setting ? atoi(setting) : 1;
    if (g_n_slots < 1) g_n_slots = 1;
    if (g_n_slots > GLM53_MAX_SLOTS) g_n_slots = GLM53_MAX_SLOTS;
    const char *context = getenv("GLM53_MAXT");
    g_slot_context = context ? atoi(context) : 8192;
    if (g_slot_context < 64) g_slot_context = 64;
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "slot KV: %d da %d posizioni\n", g_n_slots, g_slot_context);
    (void)m;
}

static void slot_reset(const GModel *m, KVSlot *slot) {
    if (slot->session) session_close(m, slot->session);
    slot->session = NULL;
    slot->n = 0;
}

/* Quanti token iniziali lo slot ha gia' in cache e puo' tenere. */
static int slot_shared(const KVSlot *slot, const int *tokens, int n) {
    int shared = 0;
    while (shared < slot->n && shared < n && slot->tokens[shared] == tokens[shared])
        shared++;
    return shared;
}

static void slot_remember(KVSlot *slot, const int *tokens, int n) {
    if (n > slot->cap) {
        slot->tokens = realloc(slot->tokens, (size_t)n * sizeof(int));
        if (!slot->tokens) { fprintf(stderr, "OOM sulla storia dello slot\n"); exit(1); }
        slot->cap = n;
    }
    memcpy(slot->tokens, tokens, (size_t)n * sizeof(int));
    slot->n = n;
}

/* Un'immagine annunciata prima della richiesta a cui appartiene.
 *
 * Il protocollo porta testo; le patch sono binarie e grosse, quindi arrivano
 * con un frame loro (IMAGE) subito prima del SUBMIT con lo stesso id. Il
 * motore la tiene da parte finche' quella richiesta non arriva, e se arriva
 * un'altra immagine prima la vecchia si butta: tenere quella sbagliata
 * vorrebbe dire rispondere sulla foto precedente senza dirlo. */
typedef struct {
    unsigned long long id;
    float *patches;
    int grid_h, grid_w;
} PendingImage;

static PendingImage g_pending = {0, NULL, 0, 0};

static void pending_clear(void) {
    free(g_pending.patches);
    g_pending.patches = NULL;
    g_pending.id = 0;
}

typedef struct {
    unsigned long long id;
    int slot, max_tokens;
    float temp, top_p;
    char *payload;
    int plen;
} ServeReq;

static int g_stop[16], g_nstop = 0;

/* Gli stop, con la regola che gli altri motori hanno imparato a caro prezzo.
 *
 * In modalita' batch resta solo il vero fine testo. I marcatori di ruolo sono
 * un confine che possiede il server Python, e tenerli come stop duri taglia la
 * generazione nel momento in cui il modello apre un blocco strumenti, perche'
 * il rumore dell'argmax a int4 preferisce l'id di uno stop al '<' giusto
 * (#401). Con SERVE=1 e basta, invece, `coli chat` non ha nessun filtro a
 * valle e gli stop gli servono tutti. */
static void arm_stops(const char *dir, Tok *tokenizer, int batched) {
    g_nstop = load_stops(dir, g_stop, 16);
    if (!batched && tokenizer)
        for (int id = 0; id < tokenizer->n_ids && g_nstop < 16; id++) {
            if (!tokenizer->id_special[id]) continue;
            int seen = 0;
            for (int i = 0; i < g_nstop; i++) if (g_stop[i] == id) seen = 1;
            if (!seen) g_stop[g_nstop++] = id;
        }
    fprintf(stderr, "[stop] %d token di stop:", g_nstop);
    for (int i = 0; i < g_nstop; i++) fprintf(stderr, " %d", g_stop[i]);
    fprintf(stderr, "%s\n", batched ? " (modalita' batch: solo fine testo)" : "");
}

static int is_stop(int token) {
    for (int i = 0; i < g_nstop; i++) if (token == g_stop[i]) return 1;
    return 0;
}

static void serve_line(const char *format, ...) {
    va_list args; va_start(args, format);
    vprintf(format, args);
    va_end(args);
    fflush(stdout);
}

/* Un token decodificato verso il server, con la sua lunghezza in byte. */
static void serve_data(unsigned long long id, const char *text, int n) {
    printf("DATA %llu %d\n", id, n);
    fwrite(text, 1, (size_t)n, stdout);
    putchar('\n');
    fflush(stdout);
}

/* Una richiesta intera, o 0 su EOF. Il payload si legge a byte contati, non a
 * righe: puo' contenerne. */
static int serve_read_req(ServeReq *q, char *verb, size_t verb_size) {
    char header[512];
    if (!fgets(header, sizeof(header), stdin)) return 0;
    memset(q, 0, sizeof(*q));
    if (sscanf(header, "%15s", verb) != 1) { verb[0] = 0; return 1; }
    if (!strcmp(verb, "STOP") || !strcmp(verb, "CANCEL")) {
        sscanf(header, "%*s %llu", &q->id);
        return 1;
    }
    if (!strcmp(verb, "IMAGE")) {
        unsigned long long id; int bytes, grid_h, grid_w;
        if (sscanf(header, "IMAGE %llu %d %d %d", &id, &bytes, &grid_h, &grid_w) != 4 ||
            bytes < 0 || bytes > (1 << 28) || grid_h < 1 || grid_w < 1) {
            strcpy(verb, "BAD_FRAME");
            return 1;
        }
        pending_clear();
        float *patches = malloc((size_t)bytes);
        if (!patches || fread(patches, 1, (size_t)bytes, stdin) != (size_t)bytes) {
            free(patches);
            strcpy(verb, "BAD_FRAME");
            return 1;
        }
        (void)fgetc(stdin);                       /* il '\n' di chiusura */
        g_pending.id = id;
        g_pending.patches = patches;
        g_pending.grid_h = grid_h;
        g_pending.grid_w = grid_w;
        q->id = id;
        return 1;
    }
    if (strcmp(verb, "SUBMIT")) return 1;
    if (sscanf(header, "SUBMIT %llu %d %d %d %f %f", &q->id, &q->slot, &q->plen,
               &q->max_tokens, &q->temp, &q->top_p) != 6) {
        strcpy(verb, "BAD_FRAME");
        return 1;
    }
    if (q->plen < 0 || q->plen > (1 << 24)) { strcpy(verb, "BAD_FRAME"); return 1; }
    q->payload = malloc((size_t)q->plen + 1);
    if (!q->payload) { strcpy(verb, "BAD_FRAME"); return 1; }
    if (q->plen && fread(q->payload, 1, (size_t)q->plen, stdin) != (size_t)q->plen) {
        free(q->payload); q->payload = NULL;
        strcpy(verb, "BAD_FRAME");
        return 1;
    }
    q->payload[q->plen] = 0;
    int trailing = fgetc(stdin);
    (void)trailing;                        /* il '\n' di chiusura del frame */
    return 1;
}

/* Genera per una richiesta e chiude col suo DONE. */
static void serve_one(GModel *m, Tok *tokenizer, ServeReq *q) {
    if (!q->plen) { serve_line("ERROR %llu EMPTY_PROMPT\n", q->id); return; }
    g_temp = q->temp; g_topp = q->top_p > 0.0f ? q->top_p : 1.0f;

    if (q->slot < 0 || q->slot >= g_n_slots) {
        serve_line("ERROR %llu BAD_REQUEST\n", q->id);
        return;
    }
    KVSlot *slot = &g_slots[q->slot];
    const int room = g_slot_context;
    int *sequence = malloc((size_t)room * sizeof(int));
    if (!sequence) { serve_line("ERROR %llu BAD_REQUEST\n", q->id); return; }
    int total = tok_encode(tokenizer, q->payload, q->plen, sequence, room);
    const int prompt_tokens = total;
    if (!total) {
        free(sequence);
        serve_line("ERROR %llu EMPTY_PROMPT\n", q->id);
        return;
    }
    if (total >= room) {
        free(sequence);
        serve_line("ERROR %llu BAD_REQUEST\n", q->id);
        return;
    }

    const double started = now_s();
    int emitted = 0, limited = 0;
    const int budget = q->max_tokens > 0 ? q->max_tokens : 256;

    /* Quanto di questo prompt e' gia' nella sessione.
     *
     * Il numero che conta e' `filled`, non quanti token lo slot ricorda: sono
     * diversi, perche' l'ultimo token generato viene emesso al client ma non
     * rimacinato, quindi resta nella storia e non nella cache. Riprendere da un
     * punto che la sessione non ha davvero raggiunto darebbe al modello un
     * contesto sfasato di un token, che e' esattamente il tipo di errore che
     * non fa rumore.
     *
     * Si riusa solo se il prompt nuovo concorda su TUTTE le posizioni in cache
     * e ne ha almeno una in piu': lo stato ricorrente dei layer KDA non si
     * riavvolge, quindi una divergenza a meta' cache obbliga a rifare. */
    const int cached = slot->session ? slot->session->filled : 0;
    int shared = 0;
    if (cached > 0 && cached < total && slot_shared(slot, sequence, total) >= cached)
        shared = cached;
    if (shared <= 0) {
        slot_reset(m, slot);
        slot->session = session_open(m, room);
        shared = 0;
    }
    /* L'immagine annunciata per QUESTA richiesta, se c'e'. Il prefisso in
     * cache non la riguarda: la torre ha gia' dato i suoi embedding quando
     * quei token sono stati macinati, e i segnaposto stanno nel prompt. Se il
     * riuso salta i token immagine, gli embedding da consumare sono quelli
     * delle posizioni nuove, quindi il riuso si annulla quando c'e' un'immagine
     * -- costa un prefill in piu' ed e' l'unica cosa che non puo' sbagliare. */
    float *vision = NULL;
    int n_vision = 0;
    if (g_pending.patches && g_pending.id == q->id) {
        if (!m->has_vision) {
            pending_clear();
            free(sequence);
            serve_line("ERROR %llu BAD_REQUEST\n", q->id);
            return;
        }
        if (shared) {                             /* niente riuso con un'immagine */
            slot_reset(m, slot);
            slot->session = session_open(m, room);
            shared = 0;
        }
        vision = vision_encode(m, g_pending.patches, g_pending.grid_h,
                               g_pending.grid_w, &n_vision);
        pending_clear();
    }

    const int reused = shared;
    float *logits = forward_prefill(m, slot->session, sequence + shared,
                                    total - shared, vision, n_vision, 0);
    GSession *session = slot->session;
    int rows = 1;
    for (int step = 0; step < budget; step++) {
        if (total >= room) { limited = 1; break; }
        int next = sample_token(logits + (size_t)(rows - 1) * m->c.vocab, m->c.vocab);
        free(logits);
        logits = NULL;
        if (is_stop(next)) break;
        sequence[total++] = next;
        emitted++;
        char piece[512];
        int written = tok_decode(tokenizer, &next, 1, piece, sizeof(piece) - 1);
        serve_data(q->id, piece, written);
        if (step + 1 == budget) { limited = 1; break; }
        logits = forward_span(m, session, &next, 1, NULL, 0);
        rows = 1;
    }
    free(logits);
    free(vision);
    /* La sessione resta allo slot per il turno dopo, con la sequenza che ha
     * davvero macinato: prompt piu' quello che ha generato. */
    slot_remember(slot, sequence, total);
    const double elapsed = now_s() - started;
    /* Quanto prefisso lo slot ha risparmiato.
     *
     * Su STDERR, non nel protocollo. La specifica dice che un server ignora le
     * righe che non conosce, ma questo progetto ha scelto il contrario apposta
     * e lo mette per iscritto in un test: una riga sconosciuta uccide il
     * dispatcher, cosi' un motore non puo' parlare a un server che non lo
     * capisce. La regola vera e' quella del test, non quella del documento.
     *
     * E dietro GLM53_VERBOSE, perche' `coli chat` eredita lo stderr del server:
     * senza guardia questa riga compare a schermo dopo ogni risposta, sotto gli
     * occhi di chi voleva solo la risposta. */
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "REUSE %llu %d %d\n", q->id, reused, prompt_tokens);
    serve_line("DONE %llu STAT %d %.2f %.1f %.1f %d %d\n", q->id, emitted,
               elapsed > 0 ? emitted / elapsed : 0.0,
               m->miss + m->hits ? 100.0 * m->hits / (double)(m->hits + m->miss) : 0.0,
               rss_gb(), prompt_tokens, limited);
    free(sequence);
}

static void serve_loop(GModel *m, Tok *tokenizer) {
    coli_serve_binary_mode();
    setvbuf(stdin, NULL, _IONBF, 0);
    slots_init(m);
    serve_line("\x01\x01READY\x01\x01\n");
    serve_line("STAT 0 0.00 0.0 %.1f\n", rss_gb());
    for (;;) {
        ServeReq q; char verb[16];
        if (!serve_read_req(&q, verb, sizeof(verb))) break;   /* EOF: si esce */
        if (!strcmp(verb, "SUBMIT")) {
            serve_one(m, tokenizer, &q);
            free(q.payload);
        } else if (!strcmp(verb, "IMAGE")) {
            /* annunciata: nessuna risposta, la si usa al SUBMIT che segue */
        } else if (!strcmp(verb, "BAD_FRAME")) {
            serve_line("ERROR %llu BAD_FRAME\n", q.id);
        } else if (!strcmp(verb, "CANCEL")) {
            /* Le richieste qui si servono una per volta e a fine giro, quindi
             * un CANCEL arriva sempre per una che non e' piu' in volo. */
            serve_line("ERROR %llu NOT_FOUND\n", q.id);
        }
        /* STOP e le righe che non riconosciamo si ignorano: la regola di
         * compatibilita' del protocollo vale in tutte e due le direzioni. */
    }
}

#ifndef GLM53_NO_MAIN
int main(int argc, char **argv) {
    const char *dir = NULL, *ids = NULL, *patch_file = NULL, *prompt_text = NULL;
    int greedy = 0, show_logits = 0, grid_h = 0, grid_w = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids = argv[++i];
        else if (!strcmp(argv[i], "--greedy") && i + 1 < argc) greedy = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits")) show_logits = 1;
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_text = argv[++i];
        else if (!strcmp(argv[i], "--patches") && i + 1 < argc) patch_file = argv[++i];
        else if (!strcmp(argv[i], "--grid") && i + 1 < argc &&
                 sscanf(argv[++i], "%dx%d", &grid_h, &grid_w) == 2) continue;
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            /* Capienza posizionale: e' cosi' che il gateway lancia ogni motore
             * (openai_server.py, Engine.__init__). Zero vuol dire "decidila
             * tu", che qui e' il budget misurato dalla RAM disponibile. */
            const int cap = atoi(argv[i]);
            if (cap > 0) g_cap_override = cap;
        }
        else { fprintf(stderr, "argomento sconosciuto: %s\n", argv[i]); return 2; }
    }
    /* SERVE=1 e SNAP=<dir>: il motore non e' piu' una CLI ma il capo di una
     * pipa, e il modello arriva dall'ambiente perche' e' cosi' che lo lancia
     * openai_server.py. */
    if (getenv("SERVE")) {
        const char *snap = getenv("SNAP");
        if (!snap) { fprintf(stderr, "SERVE senza SNAP: non so quale modello aprire\n"); return 2; }
        GModel served;
        memset(&served, 0, sizeof(served));
        model_load(&served, snap);
        Tok serve_tok;
        char tokenizer_path[1024];
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", snap);
        tok_load(&serve_tok, tokenizer_path);
        const char *batch = getenv("SERVE_BATCH");
        arm_stops(snap, &serve_tok, batch && atoi(batch));
        serve_loop(&served, &serve_tok);
        tok_free(&serve_tok);
        return 0;
    }

    if (!dir || (!ids && !prompt_text)) {
        fprintf(stderr, "uso: %s --model DIR (--prompt TESTO | --ids a,b,c)\n"
                        "         [--greedy N] [--logits] [--patches FILE.f32 --grid HxW]\n",
                argv[0]);
        return 2;
    }
    if (ids && prompt_text) {
        fprintf(stderr, "--prompt e --ids sono due modi di dire la stessa cosa\n");
        return 2;
    }
    if (!!patch_file != (grid_h > 0 && grid_w > 0)) {
        fprintf(stderr, "--patches e --grid vanno insieme\n");
        return 2;
    }
    int capacity = 1024, count = 0;
    int *tokens = malloc((size_t)capacity * sizeof(int));
    Tok tokenizer; int has_tokenizer = 0;

    if (prompt_text) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
        tok_load(&tokenizer, path);
        has_tokenizer = 1;
        const int room = 1 << 16;
        tokens = realloc(tokens, (size_t)room * sizeof(int));
        capacity = room;
        count = tok_encode(&tokenizer, prompt_text, (int)strlen(prompt_text), tokens, room);
    } else {
        for (const char *p = ids; *p; ) {
            char *end;
            long value = strtol(p, &end, 10);
            if (end == p) break;
            if (count == capacity) tokens = realloc(tokens, (size_t)(capacity *= 2) * sizeof(int));
            tokens[count++] = (int)value;
            p = (*end == ',') ? end + 1 : end;
        }
    }
    if (!count) { fprintf(stderr, "nessun token nel prompt\n"); return 2; }

    GModel model;
    memset(&model, 0, sizeof(model));     /* contatori e puntatori opzionali */
    const double load_start = now_s();
    model_load(&model, dir);
    const double load_seconds = now_s() - load_start;
    if (getenv("GLM53_VERBOSE")) cfg_report(&model.c);

    float *vision = NULL;
    int n_vision = 0;
    if (patch_file) {
        const ColiVisionConfig *vc = &model.vision.config;
        if (!model.has_vision) {
            fprintf(stderr, "--patches ma il checkpoint non porta la torre vision\n");
            return 2;
        }
        const size_t per_patch = (size_t)vc->in_channels * vc->temporal * vc->patch * vc->patch;
        const size_t wanted = (size_t)grid_h * grid_w * per_patch;
        FILE *f = fopen(patch_file, "rb");
        if (!f) { fprintf(stderr, "non apro %s\n", patch_file); return 2; }
        float *patches = malloc(wanted * sizeof(float));
        if (!patches) { fprintf(stderr, "OOM sulle patch\n"); return 2; }
        size_t got = fread(patches, sizeof(float), wanted, f);
        /* Una patch corta darebbe comunque un'uscita, con la coda letta da
         * memoria non inizializzata: meglio fermarsi e dire di quanto. */
        if (got != wanted) {
            fprintf(stderr, "%s: %zu float su %zu attesi per una griglia %dx%d\n",
                    patch_file, got, wanted, grid_h, grid_w);
            return 2;
        }
        fclose(f);
        vision = vision_encode(&model, patches, grid_h, grid_w, &n_vision);
        free(patches);
        printf("vision_tokens %d\n", n_vision);
    }

    /* Una sola sessione per tutta la generazione: il prompt si prefilla una
     * volta e ogni token dopo costa un token, non tutto il prefisso. */
    GSession *session = session_open(&model, count + (greedy > 0 ? greedy : 0) + 1);
    const double prefill_start = now_s();
    float *logits = forward_prefill(&model, session, tokens, count, vision, n_vision, 1);
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "caricamento %.1fs, prefill %d token in %.1fs\n",
                load_seconds, count, now_s() - prefill_start);
    printf("teacher_forcing");
    for (int t = 0; t < count; t++)
        printf(" %d", argmax(logits + (size_t)t * model.c.vocab, model.c.vocab));
    printf("\n");
    if (show_logits) {
        printf("last_logits");
        const float *last = logits + (size_t)(count - 1) * model.c.vocab;
        for (int v = 0; v < model.c.vocab; v++) printf(" %.9g", last[v]);
        printf("\n");
    }
    if (greedy > 0) {
        int stops[8];
        const int n_stops = has_tokenizer ? load_stops(dir, stops, 8) : 0;
        /* Il costo per token misurato qui e non ricavato per sottrazione dal
         * tempo totale: caricamento e prefill costano quanto costano, e
         * confonderli col decode ha gia' fatto sbagliare un confronto. */
        const double decode_start = now_s();
        int produced = 0;
        /* `rows` dice quante righe ha l'ultimo blocco di logit: il prefill ne
         * restituisce una per posizione, un passo incrementale una sola. In
         * entrambi i casi quella che serve e' l'ultima. */
        int rows = count;
        if (!has_tokenizer) printf("greedy");
        for (int step = 0; step < greedy; step++) {
            int next = argmax(logits + (size_t)(rows - 1) * model.c.vocab, model.c.vocab);
            free(logits);
            logits = NULL;
            int done = 0;
            for (int i = 0; i < n_stops; i++) if (next == stops[i]) done = 1;
            if (done) break;
            if (has_tokenizer) {
                char piece[512];
                int written = tok_decode(&tokenizer, &next, 1, piece, sizeof(piece) - 1);
                fwrite(piece, 1, (size_t)written, stdout);
                fflush(stdout);
            } else {
                printf(" %d", next);
            }
            /* Il token nuovo non e' mai un segnaposto immagine: la torre ha
             * gia' dato i suoi embedding durante il prefill. */
            logits = forward_span(&model, session, &next, 1, NULL, 0);
            rows = 1;
            produced++;
        }
        printf("\n");
        const double spent = now_s() - decode_start;
        if (produced)
            printf("decode %d token in %.1fs = %.3f tok/s (%.1f s/token)\n",
                   produced, spent, produced / spent, spent / produced);
    }
    if (has_tokenizer) tok_free(&tokenizer);
    /* Contatori della cache esperti: servono a un test per accorgersi se lo
     * streaming e' stato aggirato invece che esercitato. */
    if (model.streaming)
        printf("experts hits %ld miss %ld bytes %llu\n",
               model.hits, model.miss, (unsigned long long)model.ebytes);
    free(logits);
    session_close(&model, session);
    free(vision);
    free(tokens);
    return 0;
}
#endif /* GLM53_NO_MAIN */

/* ================= segment adapter =================
 *
 * Un segment esegue un INTERVALLO di layer su attivazioni, non un modello
 * intero su token: entrano H flussi residui per posizione, escono gli stessi
 * flussi dopo quei layer. Embedding e testa stanno agli estremi della catena e
 * non in mezzo, quindi qui non si caricano proprio.
 *
 * Lo stato della conversazione appartiene alla sessione, i pesi al motore. Un
 * ospite puo' tenere piu' motori nello stesso processo, quindi qui non ci sono
 * variabili globali e ogni allocazione ha il suo rilascio.
 */
#ifdef COLI_SEGMENT_ADAPTER
#include <pthread.h>
#include "segment_runtime.h"
#include "segment_adapters.h"
#include "segment_adapter_internal.h"

typedef struct {
    GModel model;
    uint32_t layer_begin, layer_end, context_tokens, state_width;
    pthread_mutex_t run_lock;
} Glm53SegmentEngine;

typedef struct {
    Glm53SegmentEngine *engine;
    GSession *session;
    uint32_t context_tokens, position;
} Glm53SegmentSession;

/* I pezzi di stato che uno snapshot deve portarsi dietro, nell'ordine in cui
 * si scrivono. Un layer DSA tiene il latente MLA e le due file dell'indexer,
 * un layer KDA la ricorrenza e la finestra della convoluzione. */
static size_t glm53_segment_spans(const Glm53SegmentEngine *engine,
                                  const Glm53SegmentSession *session,
                                  ColiSegmentStateSpan *spans, size_t capacity) {
    const Cfg *c = &engine->model.c;
    const size_t cap = session->context_tokens;
    size_t count = 0;
    for (uint32_t i = engine->layer_begin; i < engine->layer_end; i++) {
        GLayerState *st = &session->session->layer[i];
        if (c->is_full[i]) {
            if (count + 3 > capacity) return 0;
            spans[count++] = (ColiSegmentStateSpan){ st->latent, cap * (size_t)c->kv_lora * sizeof(float) };
            spans[count++] = (ColiSegmentStateSpan){ st->ikeys, cap * (size_t)c->index_hd * sizeof(float) };
            spans[count++] = (ColiSegmentStateSpan){ st->igates, cap * (size_t)c->index_hd * sizeof(float) };
        } else if (c->kda_proj) {
            if (count + 2 > capacity) return 0;
            spans[count++] = (ColiSegmentStateSpan){
                st->kda_state, (size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float) };
            spans[count++] = (ColiSegmentStateSpan){
                st->kda_window, 3u * (size_t)c->kda_proj * c->conv_k * sizeof(float) };
        }
    }
    return count;
}

#define GLM53_SEGMENT_MAX_SPANS 512

static int glm53_segment_engine_open(void **engine_impl,
                                     ColiSegmentCapabilities *capabilities,
                                     const ColiSegmentEngineOptions *options,
                                     char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options || !options->model_dir)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment needs a model directory");
    Glm53SegmentEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_segment_adapter_error(error, error_size,
                                          "out of memory opening GLM-5.3 Segment");

    /* Solo i layer chiesti: e' quello che promette RANGE_NATIVE, e caricare il
     * resto vorrebbe dire tenere in RAM i pesi che macina un'altra macchina. */
    const int begin = (int)options->layer_begin;
    const int end = options->layer_end ? (int)options->layer_end : -1;
    model_load_range(&engine->model, options->model_dir, begin, end, 0);
    engine->layer_begin = (uint32_t)engine->model.layer_begin;
    engine->layer_end = (uint32_t)engine->model.layer_end;
    engine->context_tokens = options->context_tokens ? options->context_tokens : 4096u;
    engine->state_width = (uint32_t)(engine->model.c.hc_mult * engine->model.c.hidden);
    pthread_mutex_init(&engine->run_lock, NULL);

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = (uint32_t)sizeof(*capabilities);
    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_SNAPSHOT | COLI_SEGMENT_CAP_RANGE_NATIVE |
                          COLI_SEGMENT_CAP_MULTI_SESSION | COLI_SEGMENT_CAP_CPU;
    coli_segment_capability_string(capabilities->engine_id,
                                   sizeof(capabilities->engine_id), "glm53");
    coli_segment_capability_string(capabilities->state_schema,
                                   sizeof(capabilities->state_schema),
                                   "glm53/mla-latent-kda-conv-dsa-f32-v1");
    /* Ogni leva che cambia i numeri deve comparire qui: due ospiti costruiti
     * diversamente si scambierebbero snapshot incompatibili in silenzio. */
    snprintf(capabilities->numeric_class, sizeof(capabilities->numeric_class),
             "glm53/q%d-i4g64-experts/f32/cpu-v1", glm53_dense_bits());
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = engine->state_width;
    capabilities->max_batch_rows = engine->context_tokens;
    capabilities->max_context_tokens = engine->context_tokens;
    /* Il totale del modello, non quanti ne possiede questo segment: il
     * runtime ci confronta layer_end, che e' un indice assoluto. */
    capabilities->num_layers = (uint32_t)engine->model.c.n_layers;
    *engine_impl = engine;
    return 0;
}

static void glm53_segment_engine_destroy(void *engine_impl) {
    Glm53SegmentEngine *engine = (Glm53SegmentEngine *)engine_impl;
    if (!engine) return;
    pthread_mutex_destroy(&engine->run_lock);
    model_release(&engine->model);
    free(engine);
}

static int glm53_segment_session_create(void *engine_impl, void **session_impl,
                                        const ColiSegmentSessionOptions *options,
                                        char *error, size_t error_size) {
    Glm53SegmentEngine *engine = (Glm53SegmentEngine *)engine_impl;
    if (!engine || !session_impl)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment session needs an engine");
    Glm53SegmentSession *session = calloc(1, sizeof(*session));
    if (!session)
        return coli_segment_adapter_error(error, error_size,
                                          "out of memory creating a GLM-5.3 session");
    session->engine = engine;
    session->context_tokens = (options && options->context_tokens)
                              ? options->context_tokens : engine->context_tokens;
    if (session->context_tokens > engine->context_tokens)
        session->context_tokens = engine->context_tokens;
    session->session = session_open(&engine->model, (int)session->context_tokens);
    session->position = 0;
    *session_impl = session;
    return 0;
}

static void glm53_segment_session_destroy(void *session_impl) {
    Glm53SegmentSession *session = (Glm53SegmentSession *)session_impl;
    if (!session) return;
    session_close(&session->engine->model, session->session);
    free(session);
}

static int glm53_segment_session_run(void *session_impl,
                                     const ColiSegmentRunRequest *request,
                                     char *error, size_t error_size) {
    Glm53SegmentSession *session = (Glm53SegmentSession *)session_impl;
    if (!session || !request)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment run needs a request");
    if (request->position != session->position)
        return coli_segment_adapter_error(
            error, error_size, "GLM-5.3 Segment requires contiguous positions");
    if (request->should_cancel && request->should_cancel(request->cancel_user_data))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment run cancelled");
    Glm53SegmentEngine *engine = session->engine;
    const size_t width = engine->state_width;
    size_t cells;
    if (coli_segment_size_mul(request->rows, width, &cells))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 boundary state overflows");
    if (request->input_bytes < cells * sizeof(float) ||
        request->output_bytes < cells * sizeof(float))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment buffers are too small");
    if (session->position + request->rows > session->context_tokens)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment context is full");

    float *streams = malloc(cells * sizeof(float));
    float *next = malloc(cells * sizeof(float));
    if (!streams || !next) {
        free(streams); free(next);
        return coli_segment_adapter_error(error, error_size,
                                          "out of memory running GLM-5.3 Segment");
    }
    memcpy(streams, request->input, cells * sizeof(float));

    pthread_mutex_lock(&engine->run_lock);
    float *result = run_layers(&engine->model, session->session, streams, next,
                               (int)request->rows, (int)session->position,
                               (int)engine->layer_begin, (int)engine->layer_end);
    session->session->filled = (int)(session->position + request->rows);
    pthread_mutex_unlock(&engine->run_lock);

    memcpy(request->output, result, cells * sizeof(float));
    free(streams); free(next);
    session->position += request->rows;
    return 0;
}

static int glm53_segment_session_snapshot(void *session_impl,
                                          ColiSegmentWriteFn write_fn,
                                          void *write_user_data,
                                          char *error, size_t error_size) {
    Glm53SegmentSession *session = (Glm53SegmentSession *)session_impl;
    if (!session || !write_fn)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment snapshot needs a sink");
    ColiSegmentStateSpan spans[GLM53_SEGMENT_MAX_SPANS];
    const size_t count = glm53_segment_spans(session->engine, session, spans,
                                             GLM53_SEGMENT_MAX_SPANS);
    size_t bytes = 0;
    if (coli_segment_spans_size(spans, count, &bytes))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment state overflows");
    ColiSegmentSnapshotHeader header;
    coli_segment_snapshot_header_init(&header, "glm53", session->engine->layer_begin,
                                      session->engine->layer_end,
                                      session->context_tokens, session->position,
                                      bytes,
                                      coli_segment_spans_hash(spans, count));
    if (coli_segment_stream_write(write_fn, write_user_data, &header, sizeof(header),
                                  error, error_size))
        return -1;
    return coli_segment_spans_write(spans, count, write_fn, write_user_data,
                                    error, error_size);
}

static int glm53_segment_session_restore(void *session_impl,
                                         ColiSegmentReadFn read_fn,
                                         void *read_user_data,
                                         char *error, size_t error_size) {
    Glm53SegmentSession *session = (Glm53SegmentSession *)session_impl;
    if (!session || !read_fn)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment restore needs a source");
    ColiSegmentStateSpan spans[GLM53_SEGMENT_MAX_SPANS];
    const size_t count = glm53_segment_spans(session->engine, session, spans,
                                             GLM53_SEGMENT_MAX_SPANS);
    size_t bytes = 0;
    if (coli_segment_spans_size(spans, count, &bytes))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment state overflows");
    ColiSegmentSnapshotHeader header;
    if (coli_segment_stream_read(read_fn, read_user_data, &header, sizeof(header),
                                 error, error_size))
        return -1;
    if (coli_segment_snapshot_header_valid(&header, "glm53",
                                           session->engine->layer_begin,
                                           session->engine->layer_end,
                                           session->context_tokens, bytes,
                                           error, error_size))
        return -1;
    if (coli_segment_spans_restore(spans, count, header.payload_hash, read_fn,
                                   read_user_data, error, error_size))
        return -1;
    session->position = header.position;
    session->session->filled = (int)header.position;
    return 0;
}

static const ColiSegmentAdapter glm53_segment_adapter = {
    sizeof(ColiSegmentAdapter), COLI_SEGMENT_ABI_VERSION, "glm53",
    glm53_segment_engine_open, glm53_segment_engine_destroy,
    glm53_segment_session_create, glm53_segment_session_destroy,
    glm53_segment_session_run, glm53_segment_session_snapshot,
    glm53_segment_session_restore, {0}
};

int coli_glm53_segment_adapter_register(void) {
    return coli_segment_adapter_register(&glm53_segment_adapter);
}
#endif /* COLI_SEGMENT_ADAPTER */

/* ================= edge adapter =================
 *
 * I due capi della catena: da una parte i token diventano stato, dall'altra lo
 * stato torna token. In mezzo ci sono i segment, che di token non sanno nulla.
 *
 * Qui servono l'embedding, la norma finale, la testa e il tokenizzatore, e NON
 * serve nessun layer: si carica esattamente quello.
 */
#ifdef COLI_EDGE_ADAPTER
#include "edge_runtime.h"
#include "edge_adapters.h"
#include "edge_adapter_internal.h"

typedef struct {
    GModel model;
    Tok tokenizer;
    int has_tokenizer;
    uint32_t state_width;
} Glm53EdgeEngine;

static int glm53_edge_engine_open(void **engine_impl,
                                  ColiEdgeCapabilities *capabilities,
                                  const ColiEdgeEngineOptions *options,
                                  char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options || !options->model_dir)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge needs a model directory");
    Glm53EdgeEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory opening GLM-5.3 Edge");
    /* Nessun layer: i capi della catena non ne eseguono. */
    model_load_range(&engine->model, options->model_dir, 0, 0, 1);
    engine->state_width = (uint32_t)(engine->model.c.hc_mult * engine->model.c.hidden);

    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", options->model_dir);
    FILE *probe = fopen(path, "rb");
    if (probe) {
        fclose(probe);
        tok_load(&engine->tokenizer, path);
        engine->has_tokenizer = 1;
    }

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = (uint32_t)sizeof(*capabilities);
    capabilities->abi_version = COLI_EDGE_ABI_VERSION;
    /* Le bandiere devono corrispondere ai puntatori che l'adapter fornisce:
     * il runtime rifiuta chi dichiara meno o piu' di quello che ha. Il
     * tokenizzatore puo' mancare in un checkpoint di sola matematica. */
    capabilities->flags = COLI_EDGE_CAP_CPU | COLI_EDGE_CAP_GREEDY |
                          COLI_EDGE_CAP_LOGITS;
    if (engine->has_tokenizer)
        capabilities->flags |= COLI_EDGE_CAP_TOKENIZE | COLI_EDGE_CAP_DETOKENIZE;
    coli_edge_capability_string(capabilities->engine_id,
                                sizeof(capabilities->engine_id), "glm53");
    coli_edge_capability_string(capabilities->state_schema,
                                sizeof(capabilities->state_schema),
                                "glm53/mla-latent-kda-conv-dsa-f32-v1");
    snprintf(capabilities->numeric_class, sizeof(capabilities->numeric_class),
             "glm53/q%d-i4g64-experts/f32/cpu-v1", glm53_dense_bits());
    coli_edge_capability_string(capabilities->tokenizer_class,
                                sizeof(capabilities->tokenizer_class),
                                engine->has_tokenizer ? "glm53/bpe" : "none");
    capabilities->state_dtype = COLI_EDGE_DTYPE_F32;
    capabilities->state_width = engine->state_width;
    capabilities->vocab_size = (uint32_t)engine->model.c.vocab;
    capabilities->max_batch_rows = 1024;
    capabilities->max_context_tokens = 0;   /* lo decide il chiamante */
    capabilities->num_layers = (uint32_t)engine->model.c.n_layers;
    capabilities->bos_token_id = -1;
    capabilities->eos_token_id = -1;
    *engine_impl = engine;
    return 0;
}

static void glm53_edge_engine_destroy(void *engine_impl) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine) return;
    if (engine->has_tokenizer) tok_free(&engine->tokenizer);
    model_release(&engine->model);
    free(engine);
}

static int glm53_edge_tokenize(void *engine_impl, const char *text,
                               size_t text_bytes, int32_t *token_ids,
                               size_t token_capacity, size_t *token_count,
                               char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !engine->has_tokenizer)
        return coli_edge_adapter_error(error, error_size,
                                       "this GLM-5.3 checkpoint carries no tokenizer");
    if (!text || !token_count || text_bytes > (size_t)INT_MAX)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge tokenize got bad arguments");
    /* Giro di dimensionamento: il chiamante chiede quanti token servono prima
     * di allocare, e passa un buffer vuoto. Va risposto, non rifiutato.
     * Il tetto e' un token per byte, che con un vocabolario a livello di byte
     * e' il caso peggiore vero e non una stima. */
    const int ceiling = (int)text_bytes + 1;
    const int room = (token_ids && token_capacity) ? (int)token_capacity : ceiling;
    int *scratch = malloc((size_t)(room > 0 ? room : 1) * sizeof(int));
    if (!scratch)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory tokenizing");
    const int produced = tok_encode(&engine->tokenizer, text, (int)text_bytes,
                                    scratch, room);
    if (token_ids && token_capacity)
        for (int i = 0; i < produced && (size_t)i < token_capacity; i++)
            token_ids[i] = (int32_t)scratch[i];
    free(scratch);
    *token_count = (size_t)produced;
    return 0;
}

static int glm53_edge_detokenize(void *engine_impl, const int32_t *token_ids,
                                 size_t token_count, char *text,
                                 size_t text_capacity, size_t *text_bytes,
                                 char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !engine->has_tokenizer)
        return coli_edge_adapter_error(error, error_size,
                                       "this GLM-5.3 checkpoint carries no tokenizer");
    if (!token_ids || !text_bytes || token_count > (size_t)INT_MAX)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge detokenize got bad arguments");
    int *scratch = malloc(token_count ? token_count * sizeof(int) : sizeof(int));
    if (!scratch)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory detokenizing");
    for (size_t i = 0; i < token_count; i++) scratch[i] = (int)token_ids[i];
    /* Anche qui il primo giro puo' essere solo per la misura. Un token del
     * vocabolario piu' lungo limita quanto puo' venire fuori. */
    char *sink = NULL;
    size_t room = text_capacity;
    if (!text || !text_capacity) {
        room = token_count * 512u + 1u;
        sink = malloc(room);
        if (!sink) {
            free(scratch);
            return coli_edge_adapter_error(error, error_size,
                                           "out of memory detokenizing");
        }
    }
    const int written = tok_decode(&engine->tokenizer, scratch, (int)token_count,
                                   sink ? sink : text, (int)room);
    free(scratch);
    free(sink);
    *text_bytes = (size_t)written;
    return 0;
}

/* Token -> stato iniziale: l'embedding entra replicato in ognuno degli
 * hc_mult flussi residui, che e' esattamente come parte il passaggio. */
static int glm53_edge_embed(void *engine_impl, const ColiEdgeEmbedRequest *request,
                            char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !request || !request->token_ids || !request->output)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge embed got bad arguments");
    const Cfg *c = &engine->model.c;
    const size_t width = engine->state_width;
    if (request->output_bytes < (size_t)request->rows * width * sizeof(float))
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge embed output is too small");
    float *output = (float *)request->output;
    for (uint32_t row = 0; row < request->rows; row++) {
        const int32_t token = request->token_ids[row];
        if (token < 0 || token >= c->vocab)
            return coli_edge_adapter_error(error, error_size,
                                           "GLM-5.3 token ID is out of range");
        const float *source = engine->model.embed + (size_t)token * c->hidden;
        float *state = output + (size_t)row * width;
        for (int h = 0; h < c->hc_mult; h++)
            memcpy(state + (size_t)h * c->hidden, source,
                   (size_t)c->hidden * sizeof(float));
    }
    return 0;
}

/* Lo stato finale come logit interi, senza scegliere.
 *
 * E' la stessa trasformazione di glm53_edge_select fino all'ultimo passo: chi
 * vuole campionare a modo suo ha bisogno della distribuzione, non del vincitore,
 * e temperatura e generatore sono politica di chi serve, non matematica del
 * modello. Le due funzioni condividono la chiusura proprio perche' non possano
 * scostarsi: un argmax che guardasse numeri diversi da questi darebbe un token
 * che la distribuzione non spiega. */
static int glm53_edge_final(const Glm53EdgeEngine *engine, const float *streams,
                            float *logits, float *collapsed, float *normed) {
    const Cfg *c = &engine->model.c;
    const int H = c->hc_mult, D = c->hidden;
    for (int d = 0; d < D; d++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++) sum += streams[(size_t)h * D + d];
        collapsed[d] = sum / H;
    }
    rms(normed, collapsed, engine->model.final_norm, D, c->eps);
    mv(logits, &engine->model.head, normed);
    return 0;
}

static int glm53_edge_logits(void *engine_impl, const ColiEdgeLogitsRequest *request,
                             char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !request || !request->input || !request->logits)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge logits got bad arguments");
    const Cfg *c = &engine->model.c;
    const size_t width = engine->state_width;
    if (request->input_bytes < (size_t)request->rows * width * sizeof(float) ||
        request->logits_capacity < (size_t)request->rows * (size_t)c->vocab)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge logits buffers are too small");
    float *collapsed = malloc((size_t)c->hidden * sizeof(float));
    float *normed = malloc((size_t)c->hidden * sizeof(float));
    if (!collapsed || !normed) {
        free(collapsed); free(normed);
        return coli_edge_adapter_error(error, error_size, "out of memory for logits");
    }
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(collapsed); free(normed);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM-5.3 Edge logits cancelled");
        }
        glm53_edge_final(engine, input + (size_t)row * width,
                         request->logits + (size_t)row * c->vocab, collapsed, normed);
    }
    free(collapsed); free(normed);
    return 0;
}

/* Stato finale -> token. I flussi si richiudono con una media NON pesata, poi
 * la norma finale e la testa: le stesse tre righe con cui finisce il
 * passaggio, perche' un capo che chiudesse diversamente darebbe token diversi
 * dallo stesso stato. */
static int glm53_edge_select(void *engine_impl, const ColiEdgeSelectRequest *request,
                             char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !request || !request->input || !request->token_ids)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge select got bad arguments");
    const Cfg *c = &engine->model.c;
    const int H = c->hc_mult, D = c->hidden;
    const size_t width = engine->state_width;
    if (request->input_bytes < (size_t)request->rows * width * sizeof(float) ||
        request->token_capacity < request->rows)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge select buffers are too small");
    float *collapsed = malloc((size_t)D * sizeof(float));
    float *normed = malloc((size_t)D * sizeof(float));
    float *logits = malloc((size_t)c->vocab * sizeof(float));
    if (!collapsed || !normed || !logits) {
        free(collapsed); free(normed); free(logits);
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory selecting");
    }
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(collapsed); free(normed); free(logits);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM-5.3 Edge select cancelled");
        }
        glm53_edge_final(engine, input + (size_t)row * width, logits, collapsed, normed);
        const int best = argmax(logits, c->vocab);
        request->token_ids[row] = (int32_t)best;
        if (request->scores && request->score_capacity > row)
            request->scores[row] = logits[best];
    }
    free(collapsed); free(normed); free(logits);
    return 0;
}

static const ColiEdgeAdapter glm53_edge_adapter = {
    sizeof(ColiEdgeAdapter), COLI_EDGE_ABI_VERSION, "glm53",
    glm53_edge_engine_open, glm53_edge_engine_destroy,
    glm53_edge_tokenize, glm53_edge_detokenize,
    glm53_edge_embed, glm53_edge_select, glm53_edge_logits, {0}
};

int coli_glm53_edge_adapter_register(void) {
    return coli_edge_adapter_register(&glm53_edge_adapter);
}
#endif /* COLI_EDGE_ADAPTER */
