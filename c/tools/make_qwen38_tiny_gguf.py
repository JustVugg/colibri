#!/usr/bin/env python3
"""Write a single-file qwen4exp GGUF from a Qwen3.8 safetensors tiny fixture.

The GGUF is what llama.cpp's converter would write for the same checkpoint, with
the quantization of the released IQ4_XS file simplified for a tiny model: routed
experts, the trunk (attention, DeltaNet, shared expert, hyper-connection and PLE
matrices, embedding, lm_head) and the PLE table are Q8_0; the indexer projections
BF16; norms, the router, the shared-expert gate and the DeltaNet's small
tensors F32. Tensor names follow qwen38_core.h's q38_gg_name; the layout changes
the engine undoes (see "GGUF mode" there) are applied here:

  * every RMSNorm weight but the DeltaNet's is stored as 1 + w;
  * A_log is stored as A = -exp(A_log);
  * the DeltaNet's value heads are tiled over the key heads (value head h reads
    key head h % KH): the head order of in_proj_qkv's value rows, in_proj_z,
    in_proj_a/b, the convolution's value channels, dt_bias, A and out_proj's
    columns is permuted from the checkpoint's h // rep order;
  * the indexer's fused q|k projection is two tensors, the experts one 3D tensor
    per projection, the PLE shards one table (its hash constants metadata).

Usage: make_qwen38_tiny_gguf.py SRC_DIR OUT.gguf   (SRC_DIR: model.safetensors + config.json)
       make_qwen38_tiny_gguf.py --rebase-ref ENGINE GGUF_DIR   (ref.json from the CPU engine)

The second form rewrites GGUF_DIR/ref.json (a copy of the source's) with what the CPU
engine produces on the GGUF: Q8_0 rounding and the engine's float routing can move
late tokens and the last logits off the PyTorch reference's, and the Metal gates compare
the GPU to the CPU engine, so the CPU's tokens and logits on this very file are the
fixture's reference. The source fixture's ref.json is not touched.
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
    import torch
    from gguf import GGMLQuantizationType as QT, GGUFValueType as VT
    from safetensors import safe_open
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}. Needs gguf, torch and safetensors (local-llm/.venv-fixtures).")

Q8, BF16, F32 = QT.Q8_0, QT.BF16, QT.F32


def rebase_ref(engine, snap):
    import os
    import re
    import subprocess
    snap = Path(snap)
    ref = json.loads((snap / "ref.json").read_text())
    dump = snap / "logits.f32"
    env = dict(os.environ, COLI_METAL="0", OMP_NUM_THREADS="2", SNAP=str(snap), DUMP=str(dump))
    res = subprocess.run([str(engine), "1", "8", str(snap / "ref.json")], env=env, capture_output=True, text=True)
    line = re.search(r"^C engine : ([0-9 ]+)$", res.stdout, re.M)
    if not line or not dump.exists():
        sys.exit(f"the CPU engine gave no tokens or logits on {snap} (exit {res.returncode}):\n{res.stderr[-800:]}")
    gen = [int(t) for t in line.group(1).split()]
    logits = np.fromfile(dump, dtype="<f4")
    dump.unlink()
    if len(gen) != len(ref["generated_ids"]) or len(logits) != len(ref["final_logits"]):
        sys.exit("the engine's output does not fit ref.json's shape")
    old = ref["generated_ids"]
    ref["generated_ids"] = gen
    ref["full_ids"] = ref["prompt_ids"] + gen
    ref["final_logits"] = [float(v) for v in logits]
    ref["model"] = "qwen38_tiny_256_gguf"
    ref["rebased_from_cpu_engine"] = {"source_generated_ids": old}
    (snap / "ref.json").write_text(json.dumps(ref))
    print(f"ref.json: CPU engine tokens {gen} (source reference {old})")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--rebase-ref":
        return rebase_ref(sys.argv[2], sys.argv[3])
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = json.loads((src / "config.json").read_text())
    st = safe_open(str(src / "model.safetensors"), "pt")
    get = lambda k: st.get_tensor(k).to(torch.float32).numpy()

    layers = cfg["num_hidden_layers"]
    kinds = cfg["layer_types"]
    KH, VH = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    KD, VD = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    rep = VH // KH
    head_dim = cfg["head_dim"]
    ngram, per = cfg["ngram_size"], cfg["heads_per_ngram"]
    row = cfg["ple_embed_dim"] // ((ngram - 1) * per)
    rope_n = int(round(cfg["partial_rotary_factor"] * head_dim))
    ratio = cfg["indexer_compress_ratio"]

    # old value head kh*rep + r  ->  tiled position r*KH + kh
    perm = np.empty(VH, dtype=np.int64)   # perm[new] = old
    for kh in range(KH):
        for r in range(rep):
            perm[r * KH + kh] = kh * rep + r

    def tile_rows(a, width):   # a: [VH*width, ...] rows grouped by value head
        return a.reshape(VH, width, *a.shape[1:])[perm].reshape(a.shape)

    def tile_cols(a, width):   # a: [rows, VH*width]
        return a.reshape(a.shape[0], VH, width)[:, perm].reshape(a.shape)

    w = gguf.GGUFWriter(str(out), "qwen4exp")
    p = "qwen4exp."
    w.add_name("qwen38 tiny fixture")
    w.add_uint32(p + "block_count", layers)
    w.add_uint32(p + "context_length", cfg["max_position_embeddings"])
    w.add_uint32(p + "embedding_length", cfg["hidden_size"])
    w.add_uint32(p + "attention.head_count", cfg["num_attention_heads"])
    w.add_uint32(p + "attention.head_count_kv", cfg["num_key_value_heads"])
    w.add_float32(p + "rope.freq_base", float(cfg["rope_parameters"]["rope_theta"]))
    w.add_float32(p + "attention.layer_norm_rms_epsilon", float(cfg["rms_norm_eps"]))
    w.add_uint32(p + "expert_count", cfg["num_experts"])
    w.add_uint32(p + "expert_used_count", cfg["num_experts_per_tok"])
    w.add_uint32(p + "attention.key_length", head_dim)
    w.add_uint32(p + "attention.value_length", head_dim)
    w.add_uint32(p + "expert_feed_forward_length", cfg["moe_intermediate_size"])
    w.add_uint32(p + "expert_shared_feed_forward_length", cfg["shared_expert_intermediate_size"])
    w.add_uint32(p + "ssm.conv_kernel", cfg["linear_conv_kernel_dim"])
    w.add_uint32(p + "ssm.state_size", KD)
    w.add_uint32(p + "ssm.group_count", KH)
    w.add_uint32(p + "ssm.time_step_rank", VH)
    w.add_uint32(p + "ssm.inner_size", VH * VD)
    w.add_uint32(p + "full_attention_interval", 4 if layers % 4 == 0 and kinds[:4] == ["linear_attention"] * 3 + ["full_attention"] else 2)
    w.add_uint32(p + "rope.dimension_count", rope_n)
    w.add_uint32(p + "hyper_connection.count", cfg["hc_count"])
    w.add_uint32(p + "hyper_connection.low_rank", cfg["hc_lowrank"])
    w.add_uint32(p + "attention.indexer.head_count", cfg["indexer_n_heads"])
    w.add_uint32(p + "attention.indexer.key_length", cfg["indexer_head_dim"])
    w.add_uint32(p + "attention.indexer.top_k", cfg["indexer_budget"])
    w.add_array(p + "attention.compress_ratios", [0 if k == "linear_attention" else ratio for k in kinds])
    ple_layers = [i - 1 for i in cfg["ple_layer_ids"]]
    w.add_array(p + "ple.layers", ple_layers)
    w.add_uint32(p + "ple.ngram_size", ngram)
    w.add_uint32(p + "ple.heads_per_ngram", per)
    w.add_uint32(p + "ple.conv_kernel", cfg["ple_conv_kernel_size"])
    w.add_uint32(p + "ple.eos_token_id", cfg["eos_token_id"])
    w.add_uint32(p + "embedding_length_per_layer_input", row)

    # the PLE hash constants (one PLE layer here, as in the release) are 64-bit metadata
    pl = ple_layers[0]
    pe = f"model.layers.{pl}.ple.ple_embedding."
    for key, name in (("layer_multipliers", "layer_multipliers"), ("head_vocab_sizes", "ngram_heads_vocab_sizes"),
                      ("head_offsets", "ngram_heads_offsets")):
        w.add_key_value(p + "ple." + key, [int(v) for v in st.get_tensor(pe + name).tolist()], VT.ARRAY, VT.INT64)

    vocab = cfg["vocab_size"]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen35")
    w.add_token_list([f"t{i}" for i in range(vocab)])
    w.add_token_types([1] * vocab)
    w.add_token_merges([])
    w.add_bos_token_id(cfg.get("bos_token_id", 1))
    w.add_eos_token_id(cfg["eos_token_id"])
    w.add_pad_token_id(cfg.get("pad_token_id", 0))

    def put(name, arr, kind):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if kind == F32:
            w.add_tensor(name, arr, raw_dtype=F32)
        elif kind == BF16:
            w.add_tensor(name, gguf.quants.quantize(arr, BF16), raw_dtype=BF16)
        else:
            w.add_tensor(name, gguf.quants.quantize(arr, Q8), raw_dtype=Q8)

    norm = lambda a: a + 1.0   # f32 add, as the converter does

    put("token_embd.weight", get("model.embed_tokens.weight"), Q8)
    put("output.weight", get("lm_head.weight"), Q8)
    hm = "model.hyper_connection_mixer."
    put("output_hc_norm.weight", norm(get(hm + "hc_norm.weight")), F32)
    put("output_hc_down.weight", get(hm + "input_mix_weight_down.weight"), Q8)
    put("output_hc_up.weight", get(hm + "input_mix_weight_up.weight"), Q8)

    for i in range(layers):
        L, b = f"model.layers.{i}.", f"blk.{i}."
        for a, g in (("attn", "attn"), ("mlp", "ffn")):
            h = f"{L}{a}_hyper_connection."
            put(f"{b}hc_{g}_inject.weight", get(h + "block_inject_weight.weight"), Q8)
            put(f"{b}hc_{g}_norm.weight", norm(get(h + "hc_norm.weight")), F32)
            put(f"{b}hc_{g}_down.weight", get(h + "input_mix_weight_down.weight"), Q8)
            put(f"{b}hc_{g}_up.weight", get(h + "input_mix_weight_up.weight"), Q8)
        if kinds[i] == "linear_attention":
            la = L + "linear_attn."
            K = KH * KD
            qkv = get(la + "in_proj_qkv.weight")
            qkv = np.concatenate([qkv[:2 * K], tile_rows(qkv[2 * K:], VD)])
            conv = get(la + "conv1d.weight")[:, 0, :]
            conv = np.concatenate([conv[:2 * K], tile_rows(conv[2 * K:], VD)])
            put(b + "attn_qkv.weight", qkv, Q8)
            put(b + "attn_gate.weight", tile_rows(get(la + "in_proj_z.weight"), VD), Q8)
            put(b + "ssm_alpha.weight", tile_rows(get(la + "in_proj_a.weight"), 1), Q8)
            put(b + "ssm_beta.weight", tile_rows(get(la + "in_proj_b.weight"), 1), Q8)
            put(b + "ssm_conv1d.weight", conv, F32)
            put(b + "ssm_dt.bias", tile_rows(get(la + "dt_bias"), 1), F32)
            put(b + "ssm_a", -np.exp(tile_rows(get(la + "A_log"), 1)), F32)
            put(b + "ssm_norm.weight", get(la + "norm.weight"), F32)
            put(b + "ssm_out.weight", tile_cols(get(la + "out_proj.weight"), VD), Q8)
        else:
            sa = L + "self_attn."
            put(b + "attn_q.weight", get(sa + "q_proj.weight"), Q8)
            put(b + "attn_k.weight", get(sa + "k_proj.weight"), Q8)
            put(b + "attn_v.weight", get(sa + "v_proj.weight"), Q8)
            put(b + "attn_output.weight", get(sa + "o_proj.weight"), Q8)
            put(b + "attn_q_norm.weight", norm(get(sa + "q_norm.weight")), F32)
            put(b + "attn_k_norm.weight", norm(get(sa + "k_norm.weight")), F32)
            qk = get(sa + "indexer.index_qk_proj.weight")
            nq = cfg["indexer_n_heads"] * cfg["indexer_head_dim"]
            put(b + "indexer.q_proj.weight", qk[:nq], BF16)
            put(b + "indexer.k_proj.weight", qk[nq:], BF16)
            put(b + "indexer.q_norm.weight", norm(get(sa + "indexer.q_layernorm.weight")), F32)
            put(b + "indexer.k_norm.weight", norm(get(sa + "indexer.k_layernorm.weight")), F32)
        mlp = L + "mlp."
        put(b + "ffn_gate_inp.weight", get(mlp + "gate.weight"), F32)
        put(b + "ffn_gate_inp_shexp.weight", get(mlp + "shared_expert_gate.weight").reshape(-1), F32)
        put(b + "ffn_gate_shexp.weight", get(mlp + "shared_expert.gate_proj.weight"), Q8)
        put(b + "ffn_up_shexp.weight", get(mlp + "shared_expert.up_proj.weight"), Q8)
        put(b + "ffn_down_shexp.weight", get(mlp + "shared_expert.down_proj.weight"), Q8)
        for proj, g in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
            ex = np.stack([get(f"{mlp}experts.{e}.{proj}.weight") for e in range(cfg["num_experts"])])
            put(f"{b}ffn_{g}_exps.weight", ex, Q8)
        if i in ple_layers:
            pp = L + "ple."
            put(b + "ple_conv1d.weight", get(pp + "conv1d.weight")[:, 0, :], F32)
            put(b + "ple_key.weight", get(pp + "key_proj.weight"), Q8)
            put(b + "ple_value.weight", get(pp + "value_proj.weight"), Q8)
            put(b + "ple_norm_conv.weight", norm(get(pp + "norm_conv.weight")), F32)
            put(b + "ple_norm_key.weight", norm(get(pp + "norm_key.weight")), F32)
            put(b + "ple_norm_query.weight", norm(get(pp + "norm_query.weight")), F32)
    shards = sorted((k for k in st.keys() if k.startswith(pe + "ngram_embedding.shard_")),
                    key=lambda k: int(k.rsplit("shard_", 1)[1].split(".")[0]))
    put("per_layer_token_embd.weight", np.concatenate([get(k) for k in shards]), Q8)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
