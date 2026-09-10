#!/usr/bin/env python3
"""Build a tiny DeepSeek-V4.1-Flash-shaped container plus a token-exact reference.

Same layout as the released checkpoint -- the same tensor names, the same dtypes, the
same fp8/fp4 block-scale conventions -- at toy dimensions, so the C engine reads the
real format and every mechanism is exercised:

  hyper-connections (hc_mult copies + Sinkhorn), sliding-window attention with an
  attention sink, compressed KV shared from source layers, the two-level DSA indexer
  with candidate blocks, engram n-gram memory, fp4 routed experts with a shared expert,
  sqrtsoftplus routing with the noaux_tc bias.

The reference (`dsv41_ref.RefModel`) is torch on the CPU: the vendor's own forward pass
goes through tilelang JIT kernels that need a GPU, so it cannot run here and cannot be
the oracle. What this file guarantees is that the C engine reproduces THIS reference
token for token, and that the reference is the vendor's math transcribed with its own
lines quoted where they could drift.

    python3 tools/make_dsv41_tiny.py --out dsv41_tiny --emit-ref dsv41_tiny/ref.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsv41_ref as R  # noqa: E402

SEED = 20260910


def tiny_config() -> dict:
    """Toy shapes, real structure. Every dimension a quantizer touches is a multiple of
    32 because both block layouts (fp8 32x32, fp4 row groups of 32) require it, exactly
    as the released shapes do."""
    return dict(
        vocab_size=256,
        dim=128,
        moe_inter_dim=64,
        n_layers=6,
        n_heads=4,
        head_dim=64,
        rope_head_dim=16,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        norm_eps=1e-20,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        gate_temp=1.0,
        norm_topk_prob=True,
        route_scale=1.5,
        swiglu_limit=10.0,
        window_size=8,
        # one entry per layer: 0 = sliding window only, r = KV compressed r-to-1
        compress_ratios=[0, 2, 2, 1, 1, 0],
        kv_source_layers=[1, 3],
        index_source_layers=[1, 3, 4],
        compress_rope_theta=40000.0,
        original_seq_len=32,
        rope_theta=10000.0,
        rope_factor=16.0,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=4,
        # The two-level top-k is on. The candidate source must sit in the same
        # compress-ratio group as the layers that filter with its mask, or the mask is
        # the wrong width: the released config has it at layer 20, first of the ratio-1
        # group (20..39), with 24/28/32/36 as its consumers. Here layer 3 owns the
        # ratio-1 group and layer 4 filters with it.
        candidate_source_layer=3,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        engram_layer_ids=[1, 4],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_vocab_size=97,
        engram_pad_id=2,
        # 256 positions: a chat turn with an image costs about fifty tokens of
        # template before the span starts, and the gateway refuses a prompt that
        # does not fit. The reference tokens do not depend on it -- the YaRN
        # correction reads original_seq_len, and this only sets the table length.
        max_seq_len=256,
        # the vision tower, at toy dimensions but with the released structure: a
        # full-attention ViT with 2D RoPE, then a ratio x ratio aligner into `dim`
        vision_n_layers=2,
        vision_dim=64,
        vision_n_heads=2,
        vision_inter_dim=96,
        vision_patch_size=4,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        vision_max_n_token=64,
        image_token_id=255,
        # DSpark, the MTP draft head: its own small stages under the mtp.* namespace,
        # reading the attention input of the last layers of the backbone. The released
        # model drafts 5 tokens from 3 stages of 128 experts; here 3 from 2 stages of 4.
        n_mtp_layers=2,
        dspark_block_size=3,
        dspark_noise_token_id=7,
        dspark_target_layer_ids=[3, 4, 5],
        dspark_markov_rank=32,
        dspark_n_routed_experts=4,
        dspark_n_activated_experts=2,
    )


def engram_layout(cfg: dict) -> dict:
    """engram.py EngramLayout + compute_hash_multipliers, resolved to plain arrays.

    Primes and multipliers are computed once here and shipped in the container: the
    vendor derives them from sympy and numpy's PCG64 at load time, and neither belongs
    in a C engine. `prepare_dsv41.py` writes the same file for the real checkpoint.
    """
    import numpy as np
    from sympy import isprime

    seen: set[int] = set()
    primes, offsets, nrows = [], [], []
    for _ in cfg["engram_layer_ids"]:
        per_layer, flat = [], []
        for _ in range(cfg["engram_max_ngram_size"] - 1):
            row, candidate = [], cfg["engram_vocab_size"] - 1
            for _ in range(cfg["engram_n_heads"]):
                candidate += 1
                while not isprime(candidate) or candidate in seen:
                    candidate += 1
                seen.add(candidate)
                row.append(candidate)
            per_layer.append(row)
            flat.extend(row)
        primes.append(per_layer)
        offsets.append(list(np.cumsum([0, *flat[:-1]])))
        nrows.append(int(sum(flat)))
    multipliers = []
    for layer_id in cfg["engram_layer_ids"]:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(
            low=0,
            high=max(1, (np.iinfo(np.int64).max // cfg["engram_compressed_vocab_size"]) // 2),
            size=(cfg["engram_max_ngram_size"],),
            dtype=np.int64,
        )
        multipliers.append([int(v) * 2 + 1 for v in values])
    return dict(primes=primes, offsets=[[int(o) for o in row] for row in offsets],
                multipliers=multipliers, num_embeddings=nrows)


IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def write_tokenizer(out: Path, cfg: dict) -> None:
    """A byte tokenizer for the fixture, with the image placeholder as one atomic id.

    The gateway sends the engine text, not ids: an image span is a run of
    `IMAGE_PLACEHOLDER`, and the engine finds the span by looking for
    `image_token_id`. That only lines up if the placeholder encodes as exactly one
    token, so the fixture's tokenizer has to carry it as an added token at that id --
    a byte vocabulary alone would split it into fifteen and the span would never be
    found. The id it takes over is the last byte's, which no ASCII prompt reaches.
    """
    from make_edge_tiny_tokenizer import byte_symbols

    image_id = cfg["image_token_id"]
    vocab = {symbol: token for token, symbol in enumerate(byte_symbols()[: cfg["vocab_size"]])}
    vocab = {symbol: token for symbol, token in vocab.items() if token != image_id}
    vocab[IMAGE_PLACEHOLDER] = image_id
    payload = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": image_id, "content": IMAGE_PLACEHOLDER, "single_word": False,
                          "lstrip": False, "rstrip": False, "normalized": False, "special": True}],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": []},
    }
    (out / "tokenizer.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                                        encoding="utf-8")


def build(out: Path, cfg: dict, ref_path: Path | None, max_new: int, prompt_len: int):
    torch.manual_seed(SEED)
    out.mkdir(parents=True, exist_ok=True)

    # A synthetic "compressed token map": the real one collapses tokens that normalize
    # alike (" The", "the", "THE"), which needs the tokenizer. Here every id maps to
    # itself modulo a smaller vocab, which exercises the same code path in the engine.
    cfg = dict(cfg)
    cfg["engram_compressed_vocab_size"] = 64
    token_map = [i % cfg["engram_compressed_vocab_size"] for i in range(cfg["vocab_size"])]
    layout = engram_layout(cfg)

    tensors: dict[str, torch.Tensor] = {}
    dequant: dict[str, torch.Tensor] = {}

    def bf16(name: str, *shape, scale: float = 0.02):
        t = torch.randn(*shape) * scale
        tensors[name] = t.to(torch.bfloat16)
        dequant[name] = tensors[name].to(torch.float32)

    def f32(name: str, tensor: torch.Tensor):
        tensors[name] = tensor.to(torch.float32)
        dequant[name] = tensors[name].clone()

    def fp8(name: str, rows: int, cols: int, scale: float = 0.02):
        """A dense fp8 weight the way the checkpoint stores it: e4m3 bytes plus one
        ue8m0 exponent per 32x32 tile."""
        w = torch.randn(rows, cols) * scale
        q, s = R.fp8_quant_blocked(w, 32)
        tensors[name] = q
        tensors[name.replace(".weight", ".scale")] = s
        dequant[name] = R.fp8_dequant_blocked(q, s, 32)

    def fp4(name: str, rows: int, cols: int, scale: float = 0.05):
        w = torch.randn(rows, cols) * scale
        packed, s = R.fp4_quant_rowgroups(w, 32)
        tensors[name] = packed.view(torch.int8)
        tensors[name.replace(".weight", ".scale")] = s
        dequant[name] = R.fp4_dequant_rowgroups(packed, s, cols, 32)

    dim, hc = cfg["dim"], cfg["hc_mult"]
    hd, rd, nh = cfg["head_dim"], cfg["rope_head_dim"], cfg["n_heads"]
    bf16("embed.weight", cfg["vocab_size"], dim, scale=0.05)
    bf16("head.weight", cfg["vocab_size"], dim, scale=0.05)
    bf16("norm.weight", dim, scale=0.0)
    dequant["norm.weight"] = dequant["norm.weight"] + 1.0
    tensors["norm.weight"] = dequant["norm.weight"].to(torch.bfloat16)

    for layer in range(cfg["n_layers"]):
        p = f"layers.{layer}."
        for which in ("attn_norm", "ffn_norm"):
            t = torch.ones(dim) + torch.randn(dim) * 0.01
            tensors[p + which + ".weight"] = t.to(torch.bfloat16)
            dequant[p + which + ".weight"] = tensors[p + which + ".weight"].to(torch.float32)
        # hyper-connections: fp32 in the checkpoint, small so Sinkhorn stays well conditioned
        mix_hc = (2 + hc) * hc
        for which in ("attn", "ffn"):
            f32(p + f"hc_{which}_fn", torch.randn(mix_hc, hc * dim) * 0.01)
            f32(p + f"hc_{which}_base", torch.randn(mix_hc) * 0.1)
            f32(p + f"hc_{which}_scale", torch.ones(3) * 0.5)
        # attention
        f32(p + "attn.attn_sink", torch.randn(nh) * 0.1)
        fp8(p + "attn.wq_a.weight", cfg["q_lora_rank"], dim)
        bf16(p + "attn.q_norm.weight", cfg["q_lora_rank"], scale=0.0)
        dequant[p + "attn.q_norm.weight"] += 1.0
        tensors[p + "attn.q_norm.weight"] = dequant[p + "attn.q_norm.weight"].to(torch.bfloat16)
        fp8(p + "attn.wq_b.weight", nh * hd, cfg["q_lora_rank"])
        fp8(p + "attn.wkv.weight", hd, dim)
        bf16(p + "attn.kv_norm.weight", hd, scale=0.0)
        dequant[p + "attn.kv_norm.weight"] += 1.0
        tensors[p + "attn.kv_norm.weight"] = dequant[p + "attn.kv_norm.weight"].to(torch.bfloat16)
        fp8(p + "attn.wo_a.weight", cfg["o_groups"] * cfg["o_lora_rank"], nh * hd // cfg["o_groups"])
        fp8(p + "attn.wo_b.weight", dim, cfg["o_groups"] * cfg["o_lora_rank"])
        if layer in cfg["kv_source_layers"]:
            # the compressor stays bf16/fp32 in the checkpoint: its pooling runs in fp32
            bf16(p + "attn.compressor.wkv.weight", hd, dim)
            bf16(p + "attn.compressor.norm.weight", hd, scale=0.0)
            dequant[p + "attn.compressor.norm.weight"] += 1.0
            tensors[p + "attn.compressor.norm.weight"] = dequant[p + "attn.compressor.norm.weight"].to(torch.bfloat16)
            if cfg["compress_ratios"][layer] > 1:
                bf16(p + "attn.compressor.wgate.weight", hd, dim)
        if layer in cfg["index_source_layers"]:
            fp8(p + "attn.indexer.wq_b.weight", cfg["index_n_heads"] * cfg["index_head_dim"], cfg["q_lora_rank"])
            bf16(p + "attn.indexer.weights_proj.weight", cfg["index_n_heads"], dim)
            if layer in cfg["kv_source_layers"]:
                bf16(p + "attn.indexer.wk.weight", cfg["index_head_dim"], hd)
                bf16(p + "attn.indexer.k_norm.weight", cfg["index_head_dim"], scale=0.0)
                dequant[p + "attn.indexer.k_norm.weight"] += 1.0
                tensors[p + "attn.indexer.k_norm.weight"] = dequant[p + "attn.indexer.k_norm.weight"].to(torch.bfloat16)
        # MoE
        bf16(p + "ffn.gate.weight", cfg["n_routed_experts"], dim, scale=0.05)
        f32(p + "ffn.gate.bias", torch.randn(cfg["n_routed_experts"]) * 0.1)
        for e in range(cfg["n_routed_experts"]):
            q = p + f"ffn.experts.{e}."
            fp4(q + "w1.weight", cfg["moe_inter_dim"], dim)
            fp4(q + "w3.weight", cfg["moe_inter_dim"], dim)
            fp4(q + "w2.weight", dim, cfg["moe_inter_dim"])
        s = p + "ffn.shared_experts."
        fp8(s + "w1.weight", cfg["moe_inter_dim"], dim)
        fp8(s + "w3.weight", cfg["moe_inter_dim"], dim)
        fp8(s + "w2.weight", dim, cfg["moe_inter_dim"])
        # engram
        if layer in cfg["engram_layer_ids"]:
            which = cfg["engram_layer_ids"].index(layer)
            rows = layout["num_embeddings"][which]
            table = torch.randn(rows, cfg["engram_head_dim"]) * 0.05
            q, sc = R.fp8_quant_rowgroups(table, 32)
            tensors[p + "engram.embed.weight"] = q
            tensors[p + "engram.embed.scale"] = sc
            dequant[p + "engram.embed.weight"] = R.fp8_dequant_rowgroups(q, sc, 32)
            n_hash_cols = (cfg["engram_max_ngram_size"] - 1) * cfg["engram_n_heads"]
            fp8(p + "engram.wkv.weight", dim * (hc + 1), n_hash_cols * cfg["engram_head_dim"])
            f32(p + "engram.q_weight", torch.ones(hc, dim) + torch.randn(hc, dim) * 0.05)
            f32(p + "engram.k_weight", torch.ones(hc, dim) + torch.randn(hc, dim) * 0.05)

    # --- DSpark stages (mtp.*) ------------------------------------------------
    # Same block as the backbone -- hyper-connections, window attention, a routed MoE --
    # with three differences the engine has to honour: no compressor and no indexer (a
    # stage is window-only), a smaller expert set, and the extra heads on the last
    # stage. The token embedding and the output head are the backbone's, tied, which is
    # why convert.py drops mtp.*.embed.weight and mtp.*.head.weight from the checkpoint.
    targets = cfg["dspark_target_layer_ids"]
    rank = cfg["dspark_markov_rank"]
    for stage in range(cfg["n_mtp_layers"]):
        p = f"mtp.{stage}."
        for which in ("attn_norm", "ffn_norm"):
            t = torch.ones(dim) + torch.randn(dim) * 0.01
            tensors[p + which + ".weight"] = t.to(torch.bfloat16)
            dequant[p + which + ".weight"] = tensors[p + which + ".weight"].to(torch.float32)
        mix_hc = (2 + hc) * hc
        for which in ("attn", "ffn"):
            f32(p + f"hc_{which}_fn", torch.randn(mix_hc, hc * dim) * 0.01)
            f32(p + f"hc_{which}_base", torch.randn(mix_hc) * 0.1)
            f32(p + f"hc_{which}_scale", torch.ones(3) * 0.5)
        f32(p + "attn.attn_sink", torch.randn(nh) * 0.1)
        fp8(p + "attn.wq_a.weight", cfg["q_lora_rank"], dim)
        bf16(p + "attn.q_norm.weight", cfg["q_lora_rank"], scale=0.0)
        dequant[p + "attn.q_norm.weight"] += 1.0
        tensors[p + "attn.q_norm.weight"] = dequant[p + "attn.q_norm.weight"].to(torch.bfloat16)
        fp8(p + "attn.wq_b.weight", nh * hd, cfg["q_lora_rank"])
        fp8(p + "attn.wkv.weight", hd, dim)
        bf16(p + "attn.kv_norm.weight", hd, scale=0.0)
        dequant[p + "attn.kv_norm.weight"] += 1.0
        tensors[p + "attn.kv_norm.weight"] = dequant[p + "attn.kv_norm.weight"].to(torch.bfloat16)
        fp8(p + "attn.wo_a.weight", cfg["o_groups"] * cfg["o_lora_rank"], nh * hd // cfg["o_groups"])
        fp8(p + "attn.wo_b.weight", dim, cfg["o_groups"] * cfg["o_lora_rank"])
        bf16(p + "ffn.gate.weight", cfg["dspark_n_routed_experts"], dim, scale=0.05)
        f32(p + "ffn.gate.bias", torch.randn(cfg["dspark_n_routed_experts"]) * 0.1)
        for e in range(cfg["dspark_n_routed_experts"]):
            q = p + f"ffn.experts.{e}."
            fp4(q + "w1.weight", cfg["moe_inter_dim"], dim)
            fp4(q + "w3.weight", cfg["moe_inter_dim"], dim)
            fp4(q + "w2.weight", dim, cfg["moe_inter_dim"])
        sh = p + "ffn.shared_experts."
        fp8(sh + "w1.weight", cfg["moe_inter_dim"], dim)
        fp8(sh + "w3.weight", cfg["moe_inter_dim"], dim)
        fp8(sh + "w2.weight", dim, cfg["moe_inter_dim"])
        if stage == 0:
            fp8(p + "main_proj.weight", dim, dim * len(targets))
            bf16(p + "main_norm.weight", dim, scale=0.0)
            dequant[p + "main_norm.weight"] += 1.0
            tensors[p + "main_norm.weight"] = dequant[p + "main_norm.weight"].to(torch.bfloat16)
        if stage == cfg["n_mtp_layers"] - 1:
            bf16(p + "norm.weight", dim, scale=0.0)
            dequant[p + "norm.weight"] += 1.0
            tensors[p + "norm.weight"] = dequant[p + "norm.weight"].to(torch.bfloat16)
            bf16(p + "markov_head.embed.weight", cfg["vocab_size"], rank, scale=0.02)
            bf16(p + "markov_head.head.weight", cfg["vocab_size"], rank, scale=0.02)
            bf16(p + "confidence_head.proj.weight", 1, dim + rank, scale=0.05)

    # --- vision tower ---------------------------------------------------------
    vdim, vheads, vinter = cfg["vision_dim"], cfg["vision_n_heads"], cfg["vision_inter_dim"]
    patch_in = 3 * cfg["vision_patch_size"] ** 2
    bf16("vision.patch_embed.proj.weight", vdim, patch_in)
    bf16("vision.patch_embed.proj.bias", vdim, scale=0.01)
    for layer in range(cfg["vision_n_layers"]):
        p = f"vision.blocks.{layer}."
        for which in ("norm1", "norm2"):
            t = torch.ones(vdim) + torch.randn(vdim) * 0.01
            tensors[p + which + ".weight"] = t.to(torch.bfloat16)
            dequant[p + which + ".weight"] = tensors[p + which + ".weight"].to(torch.float32)
        bf16(p + "attn.wqkv.weight", 3 * vdim, vdim)
        bf16(p + "attn.wqkv.bias", 3 * vdim, scale=0.01)
        bf16(p + "attn.wo.weight", vdim, vdim)
        bf16(p + "attn.wo.bias", vdim, scale=0.01)
        bf16(p + "mlp.w1.weight", 2 * vinter, vdim)
        bf16(p + "mlp.w2.weight", vdim, vinter)
    t = torch.ones(vdim) + torch.randn(vdim) * 0.01
    tensors["vision.norm.weight"] = t.to(torch.bfloat16)
    dequant["vision.norm.weight"] = tensors["vision.norm.weight"].to(torch.float32)
    ratio = cfg["vision_downsample_ratio"]
    bf16("aligner.w1.weight", dim, vdim * ratio * ratio)
    bf16("aligner.w1.bias", dim, scale=0.01)
    bf16("aligner.w2.weight", dim, dim)
    bf16("aligner.w2.bias", dim, scale=0.01)
    for name in ("image_start", "image_end", "image_newline"):
        bf16(name, dim, scale=0.05)

    save_file(tensors, str(out / "model.safetensors"))

    # config.json in the released shape: everything under text_config, vision absent
    text = {k: v for k, v in cfg.items() if k not in ("max_seq_len",)}
    config = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                                "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
                                "expert_dtype": "fp4"},
        "vision_config": {
            "model_type": "deepseek_v41_vision",
            "num_hidden_layers": cfg["vision_n_layers"],
            "hidden_size": cfg["vision_dim"],
            "num_attention_heads": cfg["vision_n_heads"],
            "intermediate_size": cfg["vision_inter_dim"],
            "patch_size": cfg["vision_patch_size"],
            "rope_theta": cfg["vision_rope_theta"],
            "downsample_ratio": cfg["vision_downsample_ratio"],
            "max_image_tokens": cfg["vision_max_n_token"],
        },
        "image_token_id": cfg["image_token_id"],
        "text_config": dict(text, model_type="deepseek_v41_text",
                            hidden_size=cfg["dim"],
                            num_hidden_layers=cfg["n_layers"],
                            num_attention_heads=cfg["n_heads"],
                            moe_intermediate_size=cfg["moe_inter_dim"],
                            num_experts_per_tok=cfg["n_activated_experts"],
                            max_position_embeddings=cfg["max_seq_len"],
                            engram_num_embeddings=layout["num_embeddings"],
                            engram_compressed_vocab_size=cfg["engram_compressed_vocab_size"],
                            # the released config's spellings for the same fields: the
                            # engine and the planner read either, and a fixture that
                            # only spoke the short names would not exercise the ones a
                            # user's checkpoint actually has
                            sliding_window=cfg["window_size"],
                            kv_source_layer_ids=cfg["kv_source_layers"],
                            index_source_layer_ids=cfg["index_source_layers"],
                            candidate_source_layer_id=cfg["candidate_source_layer"],
                            engram_pad_token_id=cfg["engram_pad_id"]),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    write_tokenizer(out, cfg)

    # the engram sidecar: what the engine cannot derive without sympy and a tokenizer
    (out / "dsv41_engram.json").write_text(json.dumps({
        "max_ngram_size": cfg["engram_max_ngram_size"],
        "n_heads": cfg["engram_n_heads"],
        "head_dim": cfg["engram_head_dim"],
        "pad_id": token_map[cfg["engram_pad_id"]],
        "layer_ids": cfg["engram_layer_ids"],
        "num_embeddings": layout["num_embeddings"],
        "primes": layout["primes"],
        "offsets": layout["offsets"],
        # decimal STRINGS, not JSON numbers: a multiplier is bounded only by int64 /
        # compressed_vocab, which for a small vocabulary is past a double's 53-bit
        # mantissa. A JSON reader that stores numbers as doubles would round it, and a
        # rounded multiplier rehashes the whole table into the wrong rows -- silently,
        # because every id it produces is still a valid row.
        "multipliers": [[str(v) for v in row] for row in layout["multipliers"]],
        "token_map": token_map,
    }))

    if ref_path is None:
        return
    ngram = R.NgramHash(
        token_map=torch.tensor(token_map),
        primes=torch.tensor(layout["primes"]),
        offsets=torch.tensor(layout["offsets"]),
        multipliers=torch.tensor(layout["multipliers"]),
        pad_id=token_map[cfg["engram_pad_id"]],
        max_ngram_size=cfg["engram_max_ngram_size"],
        n_heads=cfg["engram_n_heads"],
    )
    model = R.RefModel(cfg, dequant, ngram)
    generator = torch.Generator().manual_seed(SEED + 1)
    prompt = torch.randint(0, cfg["vocab_size"], (prompt_len,), generator=generator).tolist()
    compressed = [token_map[i] for i in prompt]
    logits = model.forward(prompt, compressed)
    ids, out_ids = list(prompt), []
    # DSpark rides along: the draft head reads what the main forward just produced, so
    # every step records the block it would have proposed. The engine has to reproduce
    # these exactly -- a draft head that drifts costs nothing in correctness (the main
    # model verifies every token) and everything in speed, which is the failure mode
    # that hides.
    spec = {"block_size": cfg["dspark_block_size"], "start_pos": [],
            "drafts": [], "confidence": []}
    nxt = int(logits.argmax())
    model.spec_forward(nxt, model.main_hidden, 0)      # prefill: seeds the stage windows
    for _ in range(max_new):
        out_ids.append(nxt)
        ids.append(nxt)
        start_pos = len(ids) - 1
        logits = model.forward([nxt], [token_map[nxt]])
        nxt = int(logits.argmax())
        draft, confidence = model.spec_forward(nxt, model.main_hidden, start_pos)
        spec["start_pos"].append(start_pos)
        spec["drafts"].append(draft)
        spec["confidence"].append([round(float(v), 5) for v in confidence])
    # the vision tower gets its own reference: a fixed patch grid in, the aligner's
    # rows out. The engine is held to these, so a ViT that drifts fails here rather
    # than as an image that describes itself slightly wrong.
    vision_generator = torch.Generator().manual_seed(SEED + 2)
    n_h, n_w = 4, 5
    patches = torch.randn(n_h * n_w, 3 * cfg["vision_patch_size"] ** 2,
                          generator=vision_generator) * 0.5
    aligned = R.RefVision(cfg, dequant).forward(patches, n_h, n_w)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps({
        "model": "dsv41_tiny", "seed": SEED,
        "prompt_ids": prompt, "output_ids": out_ids,
        "full_ids": prompt + out_ids,
        "spec": spec,
        "vision": {
            "grid_h": n_h, "grid_w": n_w,
            "patches": [round(float(v), 6) for v in patches.reshape(-1)],
            "aligned_rows": aligned.shape[0],
            "aligned": [round(float(v), 5) for v in aligned.reshape(-1)],
        },
    }, indent=2))
    print(f"ref -> {ref_path}")
    print(f"  prompt {prompt[:8]}...  output {out_ids}")
    print(f"  dspark drafts {spec['drafts'][0]} ...")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--emit-ref", type=Path, default=None)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=12)
    args = ap.parse_args()
    cfg = tiny_config()
    build(args.out, cfg, args.emit_ref, args.max_new, args.prompt_len)
    total = sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    print(f"container -> {args.out}  ({total/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
