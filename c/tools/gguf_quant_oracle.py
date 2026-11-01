#!/usr/bin/env python3
"""Token-exact oracle for qwen36 reading a GGUF, at every ggml weight type.

Input: a tiny Qwen3.5-MoE checkpoint (tools/make_qwen36_tiny.py --geometry
gguf-256) and its F32 GGUF made by llama.cpp's own converter. For each type T
a variant GGUF is written in which every weight matrix (embedding, lm_head,
attention, DeltaNet projections, shared and routed experts) is stored as T.
The blocks are synthetic, with scales chosen so the weights have the
magnitude of the checkpoint's, which makes each variant a well-behaved model.

The reference is the transformers model with exactly those weights: each
tensor dequantized by gguf-py (the format's reference implementation) and
mapped back through the inverse of the converter's transforms. The inverse is
itself checked first: applied to the F32 GGUF it must return the checkpoint
bit for bit. The engine then has to reproduce the reference's greedy tokens.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent

BLOCKS = {  # type -> (block elems, block bytes, fp16 offsets); IQ1_M / MXFP4 special
    "Q4_0": (32, 18, [0]), "Q4_1": (32, 20, [0, 2]), "Q5_0": (32, 22, [0]),
    "Q5_1": (32, 24, [0, 2]), "Q8_0": (32, 34, [0]), "Q2_K": (256, 84, [80, 82]),
    "Q3_K": (256, 110, [108]), "Q4_K": (256, 144, [0, 2]), "Q5_K": (256, 176, [0, 2]),
    "Q6_K": (256, 210, [208]), "IQ2_XXS": (256, 66, [0]), "IQ2_XS": (256, 74, [0]),
    "IQ2_S": (256, 82, [0]), "IQ3_XXS": (256, 98, [0]), "IQ3_S": (256, 110, [0]),
    "IQ1_S": (256, 50, [0]), "IQ1_M": (256, 56, []), "IQ4_NL": (32, 18, [0]),
    "IQ4_XS": (256, 136, [0]), "MXFP4": (32, 17, []), "TQ1_0": (256, 54, [52]),
    "TQ2_0": (256, 66, [64]),
    "NVFP4": (64, 36, []), "Q1_0": (128, 18, [0]), "Q2_0": (64, 18, [0]),
}


def dequant(quants, qt, name, raw):
    """gguf-py where it implements the type; Q1_0 / Q2_0 from the ggml C
    reference (dequantize_row_q1_0 / _q2_0), transcribed."""
    raw = raw.reshape(-1, BLOCKS[name][1])
    if name == "Q1_0":
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32)
        bits = (raw[:, 2:18, None] >> np.arange(8, dtype=np.uint8)) & 1
        return np.where(bits.reshape(-1, 128) == 1, d, -d).astype(np.float32).reshape(-1)
    if name == "Q2_0":
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32)
        q = (raw[:, 2:18, None] >> (np.arange(4, dtype=np.uint8) * 2)) & 3
        return ((q.reshape(-1, 64).astype(np.int32) - 1).astype(np.float32) * d).reshape(-1)
    return quants.dequantize(raw.reshape(-1), qt).astype(np.float32).reshape(-1)


def load_gguf_modules(llama_dir):
    sys.path.insert(0, str(Path(llama_dir) / "gguf-py"))
    import gguf  # noqa: E402
    from gguf import quants  # noqa: E402
    return gguf, quants


# ---------------------------------------------------------------- blocks
def set_scale(raw, name, v):
    """Write scale value v into every block's scale field(s)."""
    blck, size, f16 = BLOCKS[name]
    h = np.array([v], dtype=np.float16).view(np.uint8)
    for off in f16:
        raw[:, off:off + 2] = h
    if name == "IQ1_M":
        u = int(np.array([v], dtype=np.float16).view(np.uint16)[0])
        sc = raw[:, 48:56].copy().view(np.uint16).reshape(-1, 4)
        for i, sh in enumerate((0, 4, 8, 12)):
            sc[:, i] = (sc[:, i] & 0x0FFF) | (((u >> sh) & 0xF) << 12)
        raw[:, 48:56] = sc.view(np.uint8).reshape(-1, 8)
    if name == "MXFP4":
        raw[:, 0] = v  # an e8m0 exponent here


def synth_quant(quants, qt, name, shape, target_std, rng):
    """Blocks of type `name` for a tensor of `shape` (last dim contiguous)."""
    blck, size, _ = BLOCKS[name]
    rows = int(np.prod(shape[:-1])); nb = shape[-1] // blck
    raw = rng.integers(0, 256, (rows * nb, size), dtype=np.uint8)
    if name == "Q4_0":
        # nibbles 1..15 (values -7..+7): symmetric, so the lm_head is not biased toward one token
        c = rng.integers(1, 16, (rows * nb, 16, 2), dtype=np.uint8)
        raw[:, 2:18] = c[..., 0] | (c[..., 1] << 4)
    if name == "Q2_0":
        # same bias as TQ2_0 (codes -1..+2): keep them in {-1, 0, +1} so the synthetic model is not pinned to one token
        c = rng.integers(0, 3, (rows * nb, 16, 4), dtype=np.uint8)
        raw[:, 2:18] = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    if name == "TQ2_0":
        # ternary weights: 2-bit codes 0..2 (-1, 0, +1); code 3 never occurs in real files
        c = rng.integers(0, 3, (rows * nb, 64, 4), dtype=np.uint8)
        raw[:, :64] = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    if name in ("Q4_1", "Q5_1", "Q2_K", "Q4_K", "Q5_K"):
        # two scale fields: d multiplies the codes, the second is an offset
        # (Q4_1/Q5_1: +m) or a scaled minimum (K types: -dmin*m). Choose it so
        # the weights are centred: a biased lm_head makes every step pick the
        # same token, and a reference like that proves nothing.
        f0, f1 = BLOCKS[name][2]
        def deq(d, m):
            probe = raw.copy()
            probe[:, f0:f0 + 2] = np.array([d], dtype=np.float16).view(np.uint8)
            probe[:, f1:f1 + 2] = np.array([m], dtype=np.float16).view(np.uint8)
            return quants.dequantize(probe.reshape(-1), qt).astype(np.float64)
        pos = deq(1.0, 0.0); d = target_std / float(np.std(pos))
        if name in ("Q4_1", "Q5_1"):
            m = -d * float(np.mean(pos))
        else:
            neg = deq(0.0, 1.0); m = d * float(np.mean(pos)) / max(1e-9, -float(np.mean(neg)))
        raw[:, f0:f0 + 2] = np.array([d], dtype=np.float16).view(np.uint8)
        raw[:, f1:f1 + 2] = np.array([m], dtype=np.float16).view(np.uint8)
        return raw.reshape(*shape[:-1], nb * size)
    if name == "NVFP4":
        # four UE4M3 block scales per 64: pick one code giving the target magnitude
        codes = np.arange(1, 0x7F, dtype=np.uint8)
        vals = quants.NVFP4.ue4m3_to_fp32(codes)
        probe = raw.copy(); probe[:, 0:4] = 0x38   # 1.0 * 0.5
        s = float(np.std(quants.dequantize(probe.reshape(-1), qt)))
        want = 0.5 * target_std / s if s > 0 else 0.5
        raw[:, 0:4] = codes[int(np.argmin(np.abs(vals - want)))]
        return raw.reshape(*shape[:-1], nb * size)
    if name == "MXFP4":
        probe = raw.copy(); set_scale(probe, name, 128)
        s = float(np.std(quants.dequantize(probe.reshape(-1), qt)))
        e = int(round(128 + np.log2(target_std / s))) if s > 0 else 128
        set_scale(raw, name, max(2, min(250, e)))
    else:
        probe = raw.copy(); set_scale(probe, name, 1.0)
        s = float(np.std(dequant(quants, qt, name, probe)))
        set_scale(raw, name, target_std / s if s > 0 else target_std)
    return raw.reshape(*shape[:-1], nb * size)


# ------------------------------------------------------ converter inverse
def v_unreorder(x, dim, nk, nvk, hd):
    """Inverse of llama.cpp's _reorder_v_heads (tiled -> grouped) along dim."""
    shape = list(x.shape)
    dim = dim % len(shape)
    new = shape[:dim] + [nvk, nk, hd] + shape[dim + 1:]
    x = x.reshape(new)
    perm = list(range(len(new))); perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(np.transpose(x, perm)).reshape(shape)


def to_hf(name, a, cfg):
    """GGUF tensor (numpy, torch order) -> {hf_name: array}. None = not a weight we map."""
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    kd, vd = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    r = nv // nk
    if name == "token_embd.weight": return {"model.embed_tokens.weight": a}
    if name == "output.weight": return {"lm_head.weight": a}
    if not name.startswith("blk."): return None
    _, L, rest = name.split(".", 2)
    p = f"model.layers.{L}."
    simple = {"attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
              "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
              "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
              "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
              "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight"}
    if rest in simple: return {p + simple[rest]: a}
    if rest == "attn_qkv.weight":
        qk = 2 * nk * kd
        v = a[qk:] if r == 1 else v_unreorder(a[qk:], 0, nk, r, vd)
        return {p + "linear_attn.in_proj_qkv.weight": np.concatenate([a[:qk], v], 0)}
    if rest == "attn_gate.weight":
        return {p + "linear_attn.in_proj_z.weight": a if r == 1 else v_unreorder(a, 0, nk, r, vd)}
    if rest == "ssm_out.weight":
        return {p + "linear_attn.out_proj.weight": a if r == 1 else v_unreorder(a, 1, nk, r, vd)}
    if rest in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
        return {p + "mlp.experts." + rest.split("_")[1]: a}   # gate / up / down, [E, out, in]
    return None


QUANTIZABLE = ("token_embd.weight", "output.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
               "attn_output.weight", "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
               "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
               "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


def is_quantizable(name):
    return name in QUANTIZABLE[:2] or (name.startswith("blk.") and name.split(".", 2)[2] in QUANTIZABLE)


def assign_hf(model, updates):
    """Write {hf_name: np.array} into the transformers model (experts fused)."""
    import torch
    sd = dict(model.named_parameters())
    experts = {}
    for k, v in updates.items():
        if ".mlp.experts." in k:
            base, kind = k.rsplit(".", 1)
            experts.setdefault(base, {})[kind] = v
            continue
        t = sd[k]
        assert tuple(t.shape) == v.shape, (k, t.shape, v.shape)
        with torch.no_grad(): t.copy_(torch.from_numpy(np.ascontiguousarray(v)).to(t.dtype))
    for base, kinds in experts.items():
        gu = sd.get(base + ".gate_up_proj"); dn = sd.get(base + ".down_proj")
        assert gu is not None and dn is not None, base
        with torch.no_grad():
            if "gate" in kinds and "up" in kinds:
                gu.copy_(torch.from_numpy(np.concatenate([kinds["gate"], kinds["up"]], 1)).to(gu.dtype))
            if "down" in kinds:
                dn.copy_(torch.from_numpy(kinds["down"]).to(dn.dtype))


def hf_params(model):
    return {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}


# -------------------------------------------------------------- GGUF write
def write_variant(gguf, reader, out_path, replace):
    w = gguf.GGUFWriter(str(out_path), reader.fields["general.architecture"].contents())
    for f in reader.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY and len(f.types) > 1 else None
        w.add_key_value(f.name, f.contents(), vtype, sub_type=sub)
    for name, (qt, raw) in replace.items():
        if qt is None:                          # a new F32 companion (".scale")
            w.add_tensor(name, raw)
    for t in reader.tensors:
        if t.name in replace:
            qt, raw = replace[t.name]
            w.add_tensor(t.name, raw, raw_dtype=qt)
        else:
            data = np.array(t.data)
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                w.add_tensor(t.name, data.astype(np.float32))
            else:
                w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True, help="F32 GGUF of --hf made by convert_hf_to_gguf.py")
    ap.add_argument("--llama", required=True, help="llama.cpp checkout (gguf-py matching the converter)")
    ap.add_argument("--engine", default=str(HERE / "qwen36"))
    ap.add_argument("--types", default=",".join(BLOCKS) + ",F16,BF16")
    ap.add_argument("--mixed", action="store_true", help="also a UD-style mixed variant")
    ap.add_argument("--ud", action="store_true",
                    help="also a variant typed by layer as published files are (gate and up one type in a layer, "
                         "a few layers bigger ones): the expert tier's per-layer forms")
    ap.add_argument("--caps", default="1,8")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--scales", action="store_true",
                    help="also give every quantized matrix a '.scale' companion (per tensor, per expert for the fused experts), "
                         "as NVFP4 files and llama.cpp's generic per-tensor scales do")
    args = ap.parse_args()

    gguf, quants = load_gguf_modules(args.llama)
    QT = gguf.GGMLQuantizationType
    import torch
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    cfg = json.loads((Path(args.hf) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    reader = gguf.GGUFReader(args.gguf)
    base_model = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.float32)
    base_model.eval()
    ref_params = hf_params(base_model)

    # 1. the inverse mapping, checked on the F32 file: must return the checkpoint exactly
    tensors = {t.name: t for t in reader.tensors}
    checked = 0
    for name, t in tensors.items():
        if not is_quantizable(name):
            continue
        a = np.array(t.data, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)])
        for hk, hv in to_hf(name, a, cfg).items():
            if ".mlp.experts." in hk:
                base, kind = hk.rsplit(".", 1)
                if kind in ("gate", "up"):
                    gu = ref_params[base + ".gate_up_proj"]; I = gu.shape[1] // 2
                    want = gu[:, :I] if kind == "gate" else gu[:, I:]
                else:
                    want = ref_params[base + ".down_proj"]
            else:
                want = ref_params[hk]
            if want.shape != hv.shape or not np.array_equal(want.view(np.uint32), hv.astype(np.float32).view(np.uint32)):
                sys.exit(f"inverse mapping WRONG for {name} -> {hk}: shapes {want.shape} vs {hv.shape}")
            checked += 1
    print(f"inverse mapping: {checked} tensors round-trip bit-exact from the F32 GGUF")

    work = Path(args.workdir or tempfile.mkdtemp(prefix="gguf_oracle_"))
    work.mkdir(parents=True, exist_ok=True)
    prompt = [1, 2, 3, 4, 5]
    variants = [(t, None) for t in args.types.split(",") if t]
    if args.mixed:
        variants.append(("MIXED", {"ffn_gate_exps.weight": "Q4_K", "ffn_up_exps.weight": "Q4_K",
                                   "ffn_down_exps.weight": "Q6_K", "token_embd.weight": "Q8_0",
                                   "output.weight": "Q6_K", "*": "Q8_0",
                                   "blk.0.ffn_gate_exps.weight": "IQ2_XXS", "blk.1.ffn_down_exps.weight": "IQ3_S",
                                   "blk.2.ffn_up_exps.weight": "MXFP4", "blk.3.attn_output.weight": "Q5_K"}))
    if args.ud:
        variants.append(("UD", {"ffn_gate_exps.weight": "Q4_K", "ffn_up_exps.weight": "Q4_K",
                                "ffn_down_exps.weight": "Q5_K", "*": "Q8_0",
                                "blk.0.ffn_gate_exps.weight": "Q5_K", "blk.0.ffn_up_exps.weight": "Q5_K",
                                "blk.0.ffn_down_exps.weight": "Q6_K",
                                "blk.1.ffn_gate_exps.weight": "IQ3_XXS", "blk.1.ffn_up_exps.weight": "IQ3_XXS",
                                "blk.1.ffn_down_exps.weight": "IQ4_XS", "blk.5.ffn_down_exps.weight": "Q6_K"}))
    results = []
    if args.scales:
        variants = [(l + "+S", m) for l, m in variants]
    for label, mix in variants:
      for attempt in range(6):
        rng = np.random.default_rng(sum(label.encode()) * 7919 + 17 + 1000003 * attempt)
        replace, updates = {}, {}
        for name, t in tensors.items():
            if not is_quantizable(name):
                continue
            tname = label.replace("+S", "") if mix is None else mix.get(name, mix.get(name.split(".", 2)[-1] if name.startswith("blk.") else name, mix["*"]))
            if mix is None and attempt >= 3 and name in ("token_embd.weight", "output.weight"):
                # a type whose synthetic codes are biased (TQ2_0: -1..2) can pin
                # every step to one token through the lm_head; keep the two
                # vocabulary matrices at Q8_0 and every layer matrix at the type
                tname = "Q8_0"
            shape = [int(x) for x in reversed(t.shape)]
            orig = np.array(t.data, dtype=np.float32).reshape(shape)
            std = float(np.std(orig)) or 0.02
            if tname in ("F16", "BF16"):
                if tname == "F16":
                    raw = orig.astype(np.float16); deq = raw.astype(np.float32)
                    replace[name] = (QT.F16, raw)
                else:
                    u = (orig.view(np.uint32) >> 16).astype(np.uint16)
                    deq = (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
                    replace[name] = (QT.BF16, u.view(np.uint8).reshape(*shape[:-1], -1))
            else:
                qt = QT[tname]
                raw = synth_quant(quants, qt, tname, shape, std, rng)
                replace[name] = (qt, raw)
                deq = dequant(quants, qt, name if False else tname, raw).reshape(shape)
            if label.endswith("+S"):
                experts = name.endswith("_exps.weight")
                sc = rng.uniform(0.5, 2.0, shape[0] if experts else 1).astype(np.float32)
                replace[name[:-len(".weight")] + ".scale"] = (None, sc)
                deq = deq * (sc.reshape(-1, 1, 1) if experts else sc[0])
            updates.update(to_hf(name, deq, cfg))
        vpath = work / f"variant_{label}.gguf"
        write_variant(gguf, reader, vpath, replace)
        model = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.float32); model.eval()
        assign_hf(model, updates)
        with torch.no_grad():
            out = model.generate(torch.tensor([prompt]), max_new_tokens=args.max_new, do_sample=False, use_cache=True)
        ref = {"prompt_ids": prompt, "full_ids": out[0].tolist()}
        rpath = work / f"ref_{label}.json"; rpath.write_text(json.dumps(ref))
        ok = True; detail = []
        for cap in args.caps.split(","):
            p = subprocess.run([args.engine, cap, "8", str(rpath)], env=dict(os.environ, SNAP=str(vpath)),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
            line = [l for l in p.stdout.splitlines() if l.startswith("Matching tokens")]
            m = line[-1] if line else f"rc={p.returncode} (no match line)"
            detail.append(f"cap{cap} {m.split(': ')[-1]}")
            if not line or not m.endswith(f"{args.max_new}/{args.max_new}") or p.returncode:
                ok = False
                (work / f"engine_{label}_cap{cap}.log").write_text(p.stdout)
        distinct = len(set(ref["full_ids"][len(prompt):]))
        if distinct < 4 and attempt < 5:
            print(f"{label:8s} reference degenerate ({distinct} distinct tokens), new seed", flush=True)
            continue
        results.append((label, ok, detail, distinct))
        print(f"{label:8s} {'OK ' if ok else 'FAIL'} {' '.join(detail)}  ({distinct} distinct ref tokens)", flush=True)
        break
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} variants token-exact; work dir {work}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
