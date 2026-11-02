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

/* DeepSeek V4.1 (deepseek41, antirez's format): V4's names plus the engram's and the
 * router's second bias, under the names of DeepSeek's checkpoint, which deepseek_v41.c reads.
 * An engram table comes as one tensor of 264-byte rows (256 e4m3 then 8 e8m0), the
 * checkpoint's embed.weight and embed.scale side by side; the engine reads both forms.
 * The published deepseek41 GGUFs (K-quant and IQ types) have the same names, with
 * every engram table in one per_layer_token_embd of a ggml type and the engram's
 * q_weight * k_weight as one engram_gate. */
static const st_gg_name st_gg_dsv41_glob[] = {
    {"token_embd.weight", "embed.weight"}, {"output_norm.weight", "norm.weight"},
    {"output.weight", "head.weight"},
    {"per_layer_token_embd.weight", "engram.embed.weight"},   /* the published GGUFs': every engram table in one */
    {NULL, NULL}};
static const st_gg_name st_gg_dsv41_layer[] = {
    {"hc_attn_fn.weight", "hc_attn_fn"}, {"hc_attn_base.weight", "hc_attn_base"},
    {"hc_attn_scale.weight", "hc_attn_scale"}, {"hc_ffn_fn.weight", "hc_ffn_fn"},
    {"hc_ffn_base.weight", "hc_ffn_base"}, {"hc_ffn_scale.weight", "hc_ffn_scale"},
    {"attn_sinks.weight", "attn.attn_sink"},
    {"attn_q_a.weight", "attn.wq_a.weight"}, {"attn_q_b.weight", "attn.wq_b.weight"},
    {"attn_q_a_norm.weight", "attn.q_norm.weight"},
    {"attn_kv.weight", "attn.wkv.weight"}, {"attn_kv_a_norm.weight", "attn.kv_norm.weight"},
    {"attn_output_a.weight", "attn.wo_a.weight"}, {"attn_output_b.weight", "attn.wo_b.weight"},
    {"attn_compressor_kv.weight", "attn.compressor.wkv.weight"},
    {"attn_compressor_gate.weight", "attn.compressor.wgate.weight"},
    {"attn_compressor_norm.weight", "attn.compressor.norm.weight"},
    {"indexer.attn_k.weight", "attn.indexer.wk.weight"}, {"indexer.k_norm.weight", "attn.indexer.k_norm.weight"},
    {"indexer.attn_q_b.weight", "attn.indexer.wq_b.weight"},
    {"indexer.proj.weight", "attn.indexer.weights_proj.weight"},
    {"attn_norm.weight", "attn_norm.weight"}, {"ffn_norm.weight", "ffn_norm.weight"},
    {"ffn_gate_inp.weight", "ffn.gate.weight"}, {"exp_probs_b.bias", "ffn.gate.bias"},
    {"exp_probs_b_vl.bias", "ffn.gate.bias_vl"},
    {"ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight"},
    {"ffn_down_shexp.weight", "ffn.shared_experts.w2.weight"},
    {"ffn_up_shexp.weight", "ffn.shared_experts.w3.weight"},
    {"ffn_gate_exps.weight", "ffn.experts.%d.w1.weight"},
    {"ffn_down_exps.weight", "ffn.experts.%d.w2.weight"},
    {"ffn_up_exps.weight", "ffn.experts.%d.w3.weight"},
    {"engram_q_norm.weight", "engram.q_weight"}, {"engram_k_norm.weight", "engram.k_weight"},
    {"engram_kv.weight", "engram.wkv.weight"}, {"engram_embd.weight", "engram.embed.weight"},
    {"engram_gate.weight", "engram.q_weight|engram.gate"},   /* the published GGUFs': q_weight * k_weight */
    {NULL, NULL}};

/* DeepSeek V4.1's DSpark head, which the published files ship as a GGUF of its own (arch
 * dflash, gguf.json "dspark"): its stages are blk.<s> under V4.1's layer names, here
 * mtp.<s>.; its own tensors are the first stage's (fc, enc.output_norm) or the last's,
 * '@' standing for that stage's number. */
static const st_gg_name st_gg_dflash_glob[] = {
    {"fc.weight", "mtp.0.main_proj.weight"}, {"enc.output_norm.weight", "mtp.0.main_norm.weight"},
    {"output_norm.weight", "mtp.@.norm.weight"}, {"conf_proj.weight", "mtp.@.confidence_head.proj.weight"},
    {"markov_w1.weight", "mtp.@.markov_head.embed.weight"}, {"markov_w2.weight", "mtp.@.markov_head.head.weight"},
    {NULL, NULL}};

/* Kimi K3 (kimi-k3): under the names of Moonshot's checkpoint (text side), which kimi_k3.c
 * reads. Three come in another form than the checkpoint's, under names of their own the
 * engine takes in place of the checkpoint's: res_score is res_norm * res_proj (the engine's
 * attn_sw, mlp_sw, out_sw; verified on the released bytes), ssm_a is -exp(A_log), and MLA's
 * kv_b_proj is split per head into k_b (transposed, [heads][kv_lora][qk_nope]) and v_b. */
static const st_gg_name st_gg_k3_glob[] = {
    {"token_embd.weight", "model.embed_tokens.weight"}, {"output_norm.weight", "model.norm.weight"},
    {"output.weight", "lm_head.weight"}, {"output_res_score.weight", "model.output_attn_res_score.weight"},
    {NULL, NULL}};
static const st_gg_name st_gg_k3_layer[] = {
    {"attn_norm.weight", "input_layernorm.weight"}, {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"attn_res_score.weight", "self_attention_res_score.weight"}, {"ffn_res_score.weight", "mlp_res_score.weight"},
    /* KDA */
    {"attn_q.weight", "self_attn.q_proj.weight"}, {"attn_k.weight", "self_attn.k_proj.weight"},
    {"attn_v.weight", "self_attn.v_proj.weight"}, {"ssm_g.weight", "self_attn.g_proj.weight"},
    {"attn_output.weight", "self_attn.o_proj.weight"},
    {"ssm_conv1d_q.weight", "self_attn.q_conv1d.weight"}, {"ssm_conv1d_k.weight", "self_attn.k_conv1d.weight"},
    {"ssm_conv1d_v.weight", "self_attn.v_conv1d.weight"},
    {"ssm_f_a.weight", "self_attn.f_a_proj.weight"}, {"ssm_f_b.weight", "self_attn.f_b_proj.weight"},
    {"ssm_beta.weight", "self_attn.b_proj.weight"}, {"ssm_dt.bias", "self_attn.dt_bias"},
    {"ssm_norm.weight", "self_attn.o_norm.weight"}, {"ssm_a", "self_attn.ssm_a"},
    /* MLA (attn_output and attn_gate as the KDA's o_proj and g_proj) */
    {"attn_q_a.weight", "self_attn.q_a_proj.weight"}, {"attn_q_a_norm.weight", "self_attn.q_a_layernorm.weight"},
    {"attn_q_b.weight", "self_attn.q_b_proj.weight"}, {"attn_kv_a_mqa.weight", "self_attn.kv_a_proj_with_mqa.weight"},
    {"attn_kv_a_norm.weight", "self_attn.kv_a_layernorm.weight"}, {"attn_gate.weight", "self_attn.g_proj.weight"},
    {"attn_k_b.weight", "self_attn.k_b.weight"}, {"attn_v_b.weight", "self_attn.v_b.weight"},
    /* the MoE: the router, the latent, the shared experts, the routed ones (w1 gate, w3 up, w2 down) */
    {"ffn_gate_inp.weight", "block_sparse_moe.gate.weight"}, {"exp_probs_b.bias", "block_sparse_moe.gate.e_score_correction_bias"},
    {"ffn_routed_norm.weight", "block_sparse_moe.routed_expert_norm.weight"},
    {"ffn_routed_down.weight", "block_sparse_moe.routed_expert_down_proj.weight"},
    {"ffn_routed_up.weight", "block_sparse_moe.routed_expert_up_proj.weight"},
    {"ffn_gate_shexp.weight", "block_sparse_moe.shared_experts.gate_proj.weight"},
    {"ffn_up_shexp.weight", "block_sparse_moe.shared_experts.up_proj.weight"},
    {"ffn_down_shexp.weight", "block_sparse_moe.shared_experts.down_proj.weight"},
    {"ffn_gate_exps.weight", "block_sparse_moe.experts.%d.w1.weight"},
    {"ffn_up_exps.weight", "block_sparse_moe.experts.%d.w3.weight"},
    {"ffn_down_exps.weight", "block_sparse_moe.experts.%d.w2.weight"},
    /* the dense first layer */
    {"ffn_gate.weight", "mlp.gate_proj.weight"}, {"ffn_up.weight", "mlp.up_proj.weight"},
    {"ffn_down.weight", "mlp.down_proj.weight"},
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

/* Mistral-Large-4 (mistral-large4): GLM-5.2's names where the tensor is the same (MLA's k_b and
 * v_b split, the indexer, the experts), the query straight from the input (no LoRA), the
 * softmax scalings (SSMax: attn_ssmax, indexer.ssmax), the indexer's input LayerNorm and the
 * rope's frequency factors (exp(-alpha), the learned LeRoPE alpha) under names of their own. A
 * layer reading another's KV latent has no kv_a. */
static const st_gg_name st_gg_ml4_glob[] = {
    {"token_embd.weight", "model.embed_tokens.weight"}, {"output_norm.weight", "model.norm.weight"},
    {"output.weight", "lm_head.weight"}, {"rope_freqs.weight", "model.rotary_emb.freq_factors"},
    {NULL, NULL}};
static const st_gg_name st_gg_ml4_layer[] = {
    {"attn_norm.weight", "input_layernorm.weight"}, {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"attn_q.weight", "self_attn.q_proj.weight"}, {"attn_kv_a_mqa.weight", "self_attn.kv_a_proj_with_mqa.weight"},
    {"attn_kv_a_norm.weight", "self_attn.kv_a_layernorm.weight"},
    {"attn_k_b.weight", "self_attn.k_b_proj.weight"}, {"attn_v_b.weight", "self_attn.v_b_proj.weight"},
    {"attn_output.weight", "self_attn.o_proj.weight"},
    {"attn_ssmax.weight", "self_attn.softmax_scaling.scale"}, {"attn_ssmax.bias", "self_attn.softmax_scaling.bias"},
    {"indexer.attn_q_b.weight", "self_attn.indexer.wq_b.weight"}, {"indexer.attn_k.weight", "self_attn.indexer.wk.weight"},
    {"indexer.proj.weight", "self_attn.indexer.weights_proj.weight"},
    {"indexer.k_norm.weight", "self_attn.indexer.k_norm.weight"}, {"indexer.k_norm.bias", "self_attn.indexer.k_norm.bias"},
    {"indexer.q_norm.weight", "self_attn.indexer.q_input_norm.weight"}, {"indexer.q_norm.bias", "self_attn.indexer.q_input_norm.bias"},
    {"indexer.ssmax.weight", "self_attn.indexer.softmax_scaling.scale"}, {"indexer.ssmax.bias", "self_attn.indexer.softmax_scaling.bias"},
    {"ffn_gate.weight", "mlp.gate_proj.weight"}, {"ffn_up.weight", "mlp.up_proj.weight"}, {"ffn_down.weight", "mlp.down_proj.weight"},
    {"ffn_gate_inp.weight", "mlp.gate.weight"},
    {"ffn_gate_shexp.weight", "mlp.shared_experts.gate_proj.weight"}, {"ffn_up_shexp.weight", "mlp.shared_experts.up_proj.weight"},
    {"ffn_down_shexp.weight", "mlp.shared_experts.down_proj.weight"},
    {"ffn_gate_exps.weight", "mlp.experts.%d.gate_proj.weight"}, {"ffn_up_exps.weight", "mlp.experts.%d.up_proj.weight"},
    {"ffn_down_exps.weight", "mlp.experts.%d.down_proj.weight"},
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

/* OLMoE (olmoe): conversion/olmo.py OlmoeModel, the checkpoint's names; q and k as the
 * checkpoint has them (no rotary permutation). */
static const st_gg_name st_gg_olmoe_glob[] = {
    {"token_embd.weight", "model.embed_tokens.weight"}, {"output_norm.weight", "model.norm.weight"},
    {"output.weight", "lm_head.weight"},
    {NULL, NULL}};
static const st_gg_name st_gg_olmoe_layer[] = {
    {"attn_norm.weight", "input_layernorm.weight"}, {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"attn_q.weight", "self_attn.q_proj.weight"}, {"attn_k.weight", "self_attn.k_proj.weight"},
    {"attn_v.weight", "self_attn.v_proj.weight"}, {"attn_output.weight", "self_attn.o_proj.weight"},
    {"attn_q_norm.weight", "self_attn.q_norm.weight"}, {"attn_k_norm.weight", "self_attn.k_norm.weight"},
    {"ffn_gate_inp.weight", "mlp.gate.weight"},
    {"ffn_gate_exps.weight", "mlp.experts.%d.gate_proj.weight"}, {"ffn_up_exps.weight", "mlp.experts.%d.up_proj.weight"},
    {"ffn_down_exps.weight", "mlp.experts.%d.down_proj.weight"},
    {NULL, NULL}};

static const st_gg_map st_gguf_maps[] = {
    {"olmoe", "model.layers.%d.", st_gg_olmoe_glob, st_gg_olmoe_layer},
    {"mimo2", "model.layers.%d.", st_gg_mimo_glob, st_gg_mimo_layer},
    {"glm-dsa", "model.layers.%d.", st_gg_gdsa_glob, st_gg_gdsa_layer},
    {"mistral-large4", "model.layers.%d.", st_gg_ml4_glob, st_gg_ml4_layer},
    {"glm5-next", "model.language_model.layers.%d.", st_gg_g5n_glob, st_gg_g5n_layer},
    {"deepseek4", "layers.%d.", st_gg_dsv4_glob, st_gg_dsv4_layer},
    {"deepseek41", "layers.%d.", st_gg_dsv41_glob, st_gg_dsv41_layer},
    {"dflash", "mtp.%d.", st_gg_dflash_glob, st_gg_dsv41_layer},
    {"kimi-k3", "model.layers.%d.", st_gg_k3_glob, st_gg_k3_layer},
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
static gguf_ctx st_gguf_head;   /* gguf.json "dspark": a draft head's own GGUF, when there is one */
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

/* One GGUF's tensors into S under the checkpoint's names (map). `last`: the number '@' stands
 * for in a draft head's names (st_gg_dflash_glob). 1 when the file is a colibri container. */
static int st_gguf_index(shards *S, const gguf_ctx *G, const st_gg_map *map, const char *path,
                         const char *arch, int last, int *experts_out, int *renamed_out, int *kept_out) {
    int fds[ST_MAX_SHARDS];
    if (G->n_files > ST_MAX_SHARDS) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "%s: %d parts", path, G->n_files);
    for (int p = 0; p < G->n_files; p++) fds[p] = st_open_fd(S, G->paths[p]);
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
                char *at = strchr(nm, '@');   /* a draft head's last stage */
                if (at) { char tail[256]; snprintf(tail, sizeof tail, "%s", at + 1); snprintf(at, sizeof nm - (size_t)(at - nm), "%d%s", last, tail); }
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
    *experts_out += experts; *renamed_out += renamed; *kept_out += kept;
    return packed;
}

/* 1 when snap_dir is a gguf_import folder and S now indexes its GGUF (and the draft head's,
 * gguf.json "dspark"), 0 when it is not one. A broken GGUF or one of an architecture without
 * a map is fatal (the folder says GGUF). */
static int st_init_gguf(shards *S, const char *snap_dir) {
    char gj[4096]; snprintf(gj, sizeof gj, "%s/gguf.json", snap_dir);
    FILE *f = fopen(gj, "rb");
    if (!f) return 0;
    char buf[16384]; size_t nb = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[nb] = 0;
    char *arena = NULL; jval *root = json_parse(buf, &arena);
    jval *gv = root ? json_get(root, "gguf") : NULL, *hv = root ? json_get(root, "dspark") : NULL;
    const char *g = gv && gv->t == J_STR ? gv->str : NULL;
    if (!g || !*g) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "%s: no \"gguf\" path", gj);
    char path[4096], hpath[4096] = "";
    gguf_path_join(path, sizeof path, snap_dir, g);
    if (hv && hv->t == J_STR && hv->str[0]) gguf_path_join(hpath, sizeof hpath, snap_dir, hv->str);
    free(arena);
    if (getenv("GGUF_CHECK")) gguf_header_only = 1;
    gguf_ctx *G = &st_gguf_meta;
    if (gguf_open(G, path)) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF: %s", G->err);
    const char *arch = gguf_get_str(G, "general.architecture");
    const st_gg_map *map = st_gguf_map(arch);
    if (!map) coli_load_fail(COLI_LOAD_FAIL_UNSUPPORTED, "%s: GGUF architecture '%s' has no tensor map in st_gguf.h", path, arch ? arch : "?");
    S->cap = G->n_tensors + 16; S->t = (st_tensor *)calloc((size_t)S->cap, sizeof(st_tensor));
    if (!S->t) coli_load_fail(COLI_LOAD_FAIL_NOMEM, "out of memory indexing the GGUF");
    int experts = 0, renamed = 0, kept = 0;
    const int packed = st_gguf_index(S, G, map, path, arch, 0, &experts, &renamed, &kept);
    int head = 0;
    if (hpath[0]) {   /* the draft head: its stages and its own tensors beside the model's */
        gguf_ctx *H = &st_gguf_head;
        if (gguf_open(H, hpath)) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF %s: %s", hpath, H->err);
        const char *ha = gguf_get_str(H, "general.architecture");
        const int64_t stages = gguf_get_int(H, "dflash.block_count", 0);
        if (!ha || strcmp(ha, "dflash") || stages < 1 || stages > 16)
            coli_load_fail(COLI_LOAD_FAIL_FORMAT, "%s: not a DSpark head (architecture '%s', %lld stages)", hpath,
                           ha ? ha : "?", (long long)stages);
        const int was = S->n;
        st_gguf_index(S, H, st_gguf_map("dflash"), hpath, ha, (int)stages - 1, &experts, &renamed, &kept);
        head = S->n - was;
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
    if (head) fprintf(stderr, "[gguf] the DSpark head from %s: %d of those tensors\n", hpath, head);
    return 1;
}

#endif /* ST_GGUF_H */
