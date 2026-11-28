#!/usr/bin/env python3
"""Generate a deterministic tiny NemotronH_Omni_Reasoning_V3 fixture for CPU/CUDA
and PyTorch oracle verification (Mamba-2 + Attention + LatentMoE NVFP4/FP8 + RADIO Vision)."""
import argparse
import json
import os
import struct
import numpy as np


def _f32_to_bf16(a: np.ndarray) -> np.ndarray:
    u = a.astype(np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return u.astype(np.uint16)


def _e4m3_encode_scalar(v: float) -> int:
    if not np.isfinite(v) or v == 0.0:
        return 0
    sign = 0x80 if v < 0 else 0
    av = min(abs(v), 448.0)
    best_code, best_err = 0, 1e30
    for code in range(127):
        e = (code >> 3) & 0xF
        m = code & 0x7
        val = (m * (2.0 ** -9)) if e == 0 else ((1.0 + m / 8.0) * (2.0 ** (e - 7)))
        err = abs(val - av)
        if err < best_err:
            best_err = err
            best_code = code
    return sign | best_code


_FP4_VALS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _quantize_nvfp4(w: np.ndarray):
    """Quantize [O, I] f32 matrix to NVFP4 (U8 [O, I/2], F8_E4M3 [O, I/16], F32 scale2)."""
    O, I = w.shape
    assert I % 16 == 0
    amax_global = float(np.max(np.abs(w)))
    scale2 = max(amax_global / (6.0 * 16.0), 1e-6)
    w_s = w / scale2
    ng = I // 16
    q_bytes = np.zeros((O, I // 2), dtype=np.uint8)
    s_bytes = np.zeros((O, ng), dtype=np.uint8)
    for r in range(O):
        for g in range(ng):
            blk = w_s[r, g * 16:(g + 1) * 16]
            bmax = float(np.max(np.abs(blk)))
            sc_f = max(bmax / 6.0, 1e-6)
            sc_code = _e4m3_encode_scalar(sc_f)
            e = (sc_code >> 3) & 0xF
            m = sc_code & 0x7
            sc_q = (m * (2.0 ** -9)) if e == 0 else ((1.0 + m / 8.0) * (2.0 ** (e - 7)))
            if sc_q == 0.0:
                sc_q = 2.0 ** -9
                sc_code = 1
            s_bytes[r, g] = sc_code
            normed = blk / sc_q
            nibs = np.zeros(16, dtype=np.uint8)
            for j in range(16):
                sgn = 8 if normed[j] < 0 else 0
                idx = int(np.argmin(np.abs(_FP4_VALS - abs(normed[j]))))
                nibs[j] = sgn | idx
            for j in range(8):
                q_bytes[r, g * 8 + j] = (nibs[2 * j] & 0xF) | ((nibs[2 * j + 1] & 0xF) << 4)
    return q_bytes, s_bytes, np.array(scale2, dtype=np.float32)


def _quantize_fp8(w: np.ndarray):
    amax = float(np.max(np.abs(w)))
    scale = max(amax / 240.0, 1e-6)
    q = np.vectorize(_e4m3_encode_scalar)(w / scale).astype(np.uint8)
    return q, np.array(scale, dtype=np.float32)


def write_safetensors(path: str, tensors: dict):
    header = {}
    chunks = []
    offset = 0
    for name in sorted(tensors):
        dtype, arr = tensors[name]
        raw = arr.tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hbytes) % 8)) % 8
    hbytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        for c in chunks:
            f.write(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--quant", choices=["f32", "mixed"], default="f32")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Tiny dimensions matching full Nemotron-3.5-Super-VL structure
    vocab = 128
    hidden = 64
    n_heads, n_kv_heads, head_dim = 4, 2, 16
    m_heads, m_hd, ssm_n, conv_k, n_groups = 4, 16, 8, 4, 2
    d_inner = m_heads * m_hd
    conv_dim = d_inner + 2 * n_groups * ssm_n
    proj_out = d_inner + conv_dim + m_heads
    n_exp, top_k, moe_inter, moe_lat, sh_inter = 8, 2, 32, 32, 32
    block_types = ["mamba", "moe", "attention", "moe"]

    # RADIO vision tiny config: 2 blocks, dim=32, heads=4, patch=4, max_img=32 (8x8), image=16x16 -> 4x4 -> ps 2x2=4 tokens
    v_depth, v_dim, v_heads, v_patch, v_max = 2, 32, 4, 4, 32
    v_inter = v_dim * 4
    v_cls, v_reg = 3, 7
    proj_hidden = 64

    cfg = {
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "model_type": "nemotron_h_omni",
        "downsample_ratio": 0.5,
        "force_image_size": 16,
        "img_context_token": "<image>",
        "img_context_token_id": 18,
        "patch_size": v_patch,
        "projector_hidden_size": proj_hidden,
        "ps_version": "v2",
        "vit_hidden_size": v_dim,
        "norm_mean": [0.48145466, 0.4578275, 0.40821073],
        "norm_std": [0.26862954, 0.26130258, 0.27577711],
        "llm_config": {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "vocab_size": vocab,
            "hidden_size": hidden,
            "intermediate_size": moe_inter,
            "layers_block_type": block_types,
            "num_hidden_layers": len(block_types),
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv_heads,
            "head_dim": head_dim,
            "mamba_num_heads": m_heads,
            "mamba_head_dim": m_hd,
            "ssm_state_size": ssm_n,
            "conv_kernel": conv_k,
            "n_groups": n_groups,
            "mamba_hidden_act": "silu",
            "use_conv_bias": True,
            "use_bias": False,
            "mlp_bias": False,
            "mlp_hidden_act": "relu2",
            "n_routed_experts": n_exp,
            "num_experts_per_tok": top_k,
            "n_group": 1,
            "topk_group": 1,
            "moe_intermediate_size": moe_inter,
            "moe_latent_size": moe_lat,
            "moe_shared_expert_intermediate_size": sh_inter,
            "n_shared_experts": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "layer_norm_epsilon": 1e-5,
            "bos_token_id": 1,
            "eos_token_id": 11,
            "num_nextn_predict_layers": 1,
            "time_step_limit": [0.0, 1e9],
        },
        "vision_config": {
            "model_type": "radio",
            "num_hidden_layers": v_depth,
            "hidden_size": v_dim,
            "num_attention_heads": v_heads,
            "mlp_ratio": 4.0,
            "patch_size": v_patch,
            "num_channels": 3,
            "max_img_size": v_max,
            "num_cls_tokens": v_cls,
            "num_registers": v_reg,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu",
            "qkv_bias": True,
            "layerscale_value": 1.0,
            "summary_idxs": [0, 1],
        },
    }
    with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    tensors = {}

    def rmat(shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def add_mat(name, w, kind="plain"):
        if args.quant == "mixed" and kind == "fp8":
            q, sc = _quantize_fp8(w)
            tensors[name] = ("F8_E4M3", q)
            tensors[name.replace(".weight", ".weight_scale")] = ("F32", sc)
        elif args.quant == "mixed" and kind == "nvfp4":
            q, sc, sc2 = _quantize_nvfp4(w)
            tensors[name] = ("U8", q)
            tensors[name.replace(".weight", ".weight_scale")] = ("F8_E4M3", sc)
            tensors[name.replace(".weight", ".weight_scale_2")] = ("F32", sc2)
        else:
            tensors[name] = ("F32", w)

    p = "language_model.backbone."
    tensors[p + "embeddings.weight"] = ("F32", rmat((vocab, hidden), 0.1))
    tensors[p + "norm_f.weight"] = ("F32", np.ones(hidden, dtype=np.float32) + rmat((hidden,), 0.05))
    add_mat("language_model.lm_head.weight", rmat((vocab, hidden), 0.08), "fp8")

    for li, bt in enumerate(block_types):
        lp = f"{p}layers.{li}."
        tensors[lp + "norm.weight"] = ("F32", np.ones(hidden, dtype=np.float32) + rmat((hidden,), 0.05))
        if bt == "mamba":
            add_mat(lp + "mixer.in_proj.weight", rmat((proj_out, hidden), 0.05), "fp8")
            add_mat(lp + "mixer.out_proj.weight", rmat((hidden, d_inner), 0.05), "fp8")
            tensors[lp + "mixer.conv1d.weight"] = ("F32", rmat((conv_dim, 1, conv_k), 0.1))
            tensors[lp + "mixer.conv1d.bias"] = ("F32", rmat((conv_dim,), 0.02))
            tensors[lp + "mixer.dt_bias"] = ("F32", np.zeros(m_heads, dtype=np.float32) + rmat((m_heads,), 0.2))
            tensors[lp + "mixer.A_log"] = ("F32", np.log(np.arange(1, m_heads + 1, dtype=np.float32)))
            tensors[lp + "mixer.D"] = ("F32", np.ones(m_heads, dtype=np.float32) + rmat((m_heads,), 0.05))
            tensors[lp + "mixer.norm.weight"] = ("F32", np.ones(d_inner, dtype=np.float32) + rmat((d_inner,), 0.05))
        elif bt == "attention":
            add_mat(lp + "mixer.q_proj.weight", rmat((n_heads * head_dim, hidden), 0.05), "fp8")
            add_mat(lp + "mixer.k_proj.weight", rmat((n_kv_heads * head_dim, hidden), 0.05), "fp8")
            add_mat(lp + "mixer.v_proj.weight", rmat((n_kv_heads * head_dim, hidden), 0.05), "fp8")
            add_mat(lp + "mixer.o_proj.weight", rmat((hidden, n_heads * head_dim), 0.05), "fp8")
        elif bt == "moe":
            tensors[lp + "mixer.gate.weight"] = ("F32", rmat((n_exp, hidden), 0.1))
            tensors[lp + "mixer.gate.e_score_correction_bias"] = ("F32", rmat((n_exp,), 0.05))
            tensors[lp + "mixer.fc1_latent_proj.weight"] = ("F32", rmat((moe_lat, hidden), 0.05))
            tensors[lp + "mixer.fc2_latent_proj.weight"] = ("F32", rmat((hidden, moe_lat), 0.05))
            add_mat(lp + "mixer.shared_experts.up_proj.weight", rmat((sh_inter, hidden), 0.05), "nvfp4")
            add_mat(lp + "mixer.shared_experts.down_proj.weight", rmat((hidden, sh_inter), 0.05), "nvfp4")
            for e in range(n_exp):
                add_mat(f"{lp}mixer.experts.{e}.up_proj.weight", rmat((moe_inter, moe_lat), 0.08), "nvfp4")
                add_mat(f"{lp}mixer.experts.{e}.down_proj.weight", rmat((moe_lat, moe_inter), 0.08), "nvfp4")

    # RADIO vision weights
    vp = "vision_model.radio_model.model."
    tensors[vp + "patch_generator.embedder.weight"] = ("F32", rmat((v_dim, 3 * v_patch * v_patch), 0.05))
    tensors[vp + "patch_generator.pos_embed"] = ("F32", rmat((1, (v_max // v_patch) ** 2, v_dim), 0.02))
    tensors[vp + "patch_generator.cls_token.token"] = ("F32", rmat((v_cls + v_reg, v_dim), 0.02))
    for bi in range(v_depth):
        bp = f"{vp}blocks.{bi}."
        tensors[bp + "norm1.weight"] = ("F32", np.ones(v_dim, dtype=np.float32) + rmat((v_dim,), 0.05))
        tensors[bp + "norm1.bias"] = ("F32", rmat((v_dim,), 0.02))
        tensors[bp + "norm2.weight"] = ("F32", np.ones(v_dim, dtype=np.float32) + rmat((v_dim,), 0.05))
        tensors[bp + "norm2.bias"] = ("F32", rmat((v_dim,), 0.02))
        tensors[bp + "attn.qkv.weight"] = ("F32", rmat((3 * v_dim, v_dim), 0.05))
        tensors[bp + "attn.qkv.bias"] = ("F32", rmat((3 * v_dim,), 0.02))
        tensors[bp + "attn.proj.weight"] = ("F32", rmat((v_dim, v_dim), 0.05))
        tensors[bp + "attn.proj.bias"] = ("F32", rmat((v_dim,), 0.02))
        tensors[bp + "mlp.fc1.weight"] = ("F32", rmat((v_inter, v_dim), 0.05))
        tensors[bp + "mlp.fc1.bias"] = ("F32", rmat((v_inter,), 0.02))
        tensors[bp + "mlp.fc2.weight"] = ("F32", rmat((v_dim, v_inter), 0.05))
        tensors[bp + "mlp.fc2.bias"] = ("F32", rmat((v_dim,), 0.02))
    tensors["vision_projector.vision_final_layernorm.weight"] = ("F32", np.ones(v_dim, dtype=np.float32))
    tensors["vision_projector.vision_final_layernorm.bias"] = ("F32", np.zeros(v_dim, dtype=np.float32))
    tensors["mlp1.0.weight"] = ("F32", np.ones(v_dim * 4, dtype=np.float32) + rmat((v_dim * 4,), 0.05))
    tensors["mlp1.1.weight"] = ("F32", rmat((proj_hidden, v_dim * 4), 0.05))
    tensors["mlp1.3.weight"] = ("F32", rmat((hidden, proj_hidden), 0.05))

    write_safetensors(os.path.join(args.output, "model.safetensors"), tensors)
    print(f"Wrote tiny NemotronH_Omni fixture ({args.quant}, {len(tensors)} tensors) to {args.output}")


if __name__ == "__main__":
    main()
