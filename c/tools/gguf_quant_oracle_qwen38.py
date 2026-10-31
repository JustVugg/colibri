#!/usr/bin/env python3
"""Token-exact oracle for qwen38 reading a GGUF (llama.cpp's qwen4exp), at every ggml type.

The Qwen3.8 twin of tools/gguf_quant_oracle.py, whose block synthesis and GGUF writer it
reuses. Input: a tiny Qwen3.8 (tools/make_qwen38_tiny.py --geometry gguf-256) and its F32
GGUF made by llama.cpp's own converter. For each type T a variant GGUF stores every weight
matrix as T (the embedding, lm_head, attention, the DeltaNet projections, the
hyper-connections' projections, the indexer's, the PLE's projections and its table where
T's block fits a 64-wide row, the shared and routed experts), with synthetic blocks of
the checkpoint's magnitude.

The reference is the transformers model in float32 with exactly those weights: each
tensor dequantized by gguf-py and mapped back through the inverse of the converter's
transforms (the inverse is checked first: on the F32 GGUF it must return the checkpoint
bit for bit). The engine, reading the variant through a gguf_import.py directory, has to
reproduce the reference's greedy tokens.

The QSA indexer keeps the best indexer_budget/compress_ratio key blocks by a sum of ReLU
scores; with the tiny's two indexer heads a block scores exactly 0 about one time in four,
and torch's topk leaves the order of equal scores undefined. The reference ranks those
blocks as the engine does (higher score first, then the lower block), so a tie is not
taken for a divergence.

The engine runs with GQ_EXACT=1 unless the caller sets it: the integer path rounds the
activations to int8 (about 1e-2 relative), and the tiny's random router separates its
second and third expert by as little as 1e-5 in probability, so the rounding alone picks
other experts and other tokens. That path is checked kernel by kernel against double
precision by tests/test_gq_dot.py; here the question is whether the bridge reads every
format and maps every tensor right, which only the float path answers token for token.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_quant_oracle as base  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def to_hf(name, a, cfg):
    """GGUF tensor (numpy, torch order) -> {hf_name: array}; None for one not mapped here."""
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    kd, vd = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    r = nv // nk
    top = {"token_embd.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight",
           "output_hc_down.weight": "model.hyper_connection_mixer.input_mix_weight_down.weight",
           "output_hc_up.weight": "model.hyper_connection_mixer.input_mix_weight_up.weight"}
    if name in top:
        return {top[name]: a}
    if name == "per_layer_token_embd.weight":
        return {"__ple_table__": a}
    if not name.startswith("blk."):
        return None
    _, L, rest = name.split(".", 2)
    p = f"model.layers.{L}."
    simple = {"attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
              "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
              "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
              "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
              "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
              "hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down.weight",
              "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
              "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down.weight",
              "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
              "ple_key.weight": "ple.key_proj.weight", "ple_value.weight": "ple.value_proj.weight"}
    if rest in simple:
        return {p + simple[rest]: a}
    if rest in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
        return {f"__idx_{rest.split('.')[1][0]}__{L}": a}
    if rest == "attn_qkv.weight":
        qk = 2 * nk * kd
        v = a[qk:] if r == 1 else base.v_unreorder(a[qk:], 0, nk, r, vd)
        return {p + "linear_attn.in_proj_qkv.weight": np.concatenate([a[:qk], v], 0)}
    if rest == "attn_gate.weight":
        return {p + "linear_attn.in_proj_z.weight": a if r == 1 else base.v_unreorder(a, 0, nk, r, vd)}
    if rest == "ssm_out.weight":
        return {p + "linear_attn.out_proj.weight": a if r == 1 else base.v_unreorder(a, 1, nk, r, vd)}
    if rest == "ffn_down_exps.weight":   # transformers fuses the experts: down_proj [E][out][in]
        return {p + "mlp.experts.down_proj": a}
    if rest in ("ffn_gate_exps.weight", "ffn_up_exps.weight"):   # ... and gate_up_proj, gate rows first
        return {f"__exp_{rest.split('_')[1]}__{L}": a}
    return None


QUANTIZABLE = ("token_embd.weight", "output.weight", "output_hc_down.weight", "output_hc_up.weight",
               "per_layer_token_embd.weight")
QUANTIZABLE_BLK = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight",
                   "attn_gate.weight", "ssm_out.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                   "ffn_down_shexp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                   "hc_attn_down.weight", "hc_attn_up.weight", "hc_ffn_down.weight", "hc_ffn_up.weight",
                   "ple_key.weight", "ple_value.weight", "indexer.q_proj.weight", "indexer.k_proj.weight")


def is_quantizable(name):
    return name in QUANTIZABLE or (name.startswith("blk.") and name.split(".", 2)[2] in QUANTIZABLE_BLK)


def finish(updates, hf, cfg):
    """The pieces to_hf leaves apart: the indexer's q and k rows joined, the PLE table as
    the model's one parameter."""
    out = {}
    for k, v in updates.items():
        if k.startswith("__idx_q__"):
            L = k[len("__idx_q__"):]
            out[f"model.layers.{L}.self_attn.indexer.index_qk_proj.weight"] = np.concatenate([v, updates[f"__idx_k__{L}"]], 0)
        elif k.startswith("__idx_k__") or k.startswith("__exp_up__"):
            continue
        elif k.startswith("__exp_gate__"):
            L = k[len("__exp_gate__"):]
            out[f"model.layers.{L}.mlp.experts.gate_up_proj"] = np.concatenate([v, updates[f"__exp_up__{L}"]], 1)
        elif k == "__ple_table__":   # transformers holds the shards as one parameter
            L = cfg["ple_layer_ids"][0] - 1
            out[f"model.layers.{L}.ple.ple_embedding.ngram_embedding.weight"] = v
        else:
            out[k] = v
    return out


def assign_hf(model, updates):
    import torch
    sd = dict(model.named_parameters())
    for k, v in updates.items():
        t = sd[k]
        assert tuple(t.shape) == v.shape, (k, t.shape, v.shape)
        with torch.no_grad():
            t.copy_(torch.from_numpy(np.ascontiguousarray(v)).to(t.dtype))


class indexer_ties_by_block:
    """While active, a 1-D Tensor.topk (the indexer's block ranking; the router's topk is
    2-D) orders equal scores by index, the engine's q38_block_desc."""
    def __enter__(self):
        import torch
        self.orig = orig = torch.Tensor.topk
        def topk(t, k, dim=-1, largest=True, sorted=True):
            if t.dim() != 1 or not largest:
                return orig(t, k, dim, largest, sorted)
            v = t.detach().float().cpu().numpy()
            idx = torch.from_numpy(np.lexsort((np.arange(v.size), -v))[:k].copy()).to(t.device)
            return torch.return_types.topk((t.index_select(0, idx), idx))
        torch.Tensor.topk = topk
        return self
    def __exit__(self, *exc):
        import torch
        torch.Tensor.topk = self.orig


def model_dir(link_template, variant, work, label):
    """A gguf_import.py directory for the variant: the template's config, gguf.json to it."""
    d = work / f"dir_{label}"
    d.mkdir(exist_ok=True)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (link_template / f).exists():
            shutil.copy(link_template / f, d / f)
    (d / "gguf.json").write_text(json.dumps({"format": "colibri-gguf-v1", "gguf": str(variant), "mode": "link",
                                             "arch": "qwen4exp", "parts": 1}))
    return d


def ud_type(name):
    """--ud: the types by layer of a published file (gate and up alike in a layer, the
    commonest pair Q4_K/Q5_K, three layers in other forms), the rest Q8_0"""
    m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
    if not m:
        return "Q8_0"
    l, role = int(m.group(1)), m.group(2)
    if l == 0:
        return "Q6_K" if role == "down" else "Q5_K"
    if l == 1:
        return "IQ4_XS" if role == "down" else "IQ3_XXS"
    if l == 3 and role == "down":
        return "Q6_K"
    return "Q5_K" if role == "down" else "Q4_K"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True, help="F32 GGUF of --hf made by convert_hf_to_gguf.py")
    ap.add_argument("--link", required=True, help="tools/gguf_import.py directory of --gguf (its config.json)")
    ap.add_argument("--llama", required=True)
    ap.add_argument("--engine", default=str(HERE / "qwen38"))
    ap.add_argument("--types", default=",".join(base.BLOCKS) + ",F16,BF16")
    ap.add_argument("--caps", default="1,8")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1, help="the synthetic blocks' seed")
    ap.add_argument("--only", default="", help="comma-separated substrings: quantize only the tensors whose name has one")
    ap.add_argument("--ud", action="store_true",
                    help="also a variant typed by layer as published files are (gate and up one type in a layer, "
                         "a few layers bigger ones): the expert tier's per-layer forms")
    args = ap.parse_args()
    gguf, quants = base.load_gguf_modules(args.llama)
    import torch
    from transformers import AutoModelForCausalLM
    work = Path(args.workdir or tempfile.mkdtemp(prefix="q38oracle_")).resolve(); work.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(args.hf) / "config.json").read_text())
    reader = gguf.GGUFReader(args.gguf)
    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32); model.eval()
    hf = base.hf_params(model)
    # the inverse, checked on the F32 file first: every mapped tensor back to the checkpoint's bits
    ups = {}
    for t in reader.tensors:
        if not is_quantizable(t.name):
            continue
        a = np.array(t.data, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)])
        ups.update(to_hf(t.name, a, cfg) or {})
    ups = finish(ups, hf, cfg)
    bad = [k for k, v in ups.items() if k not in hf or hf[k].shape != v.shape or not np.array_equal(hf[k], v)]
    if bad:
        sys.exit(f"inverse mapping wrong for {len(bad)} tensors: {bad[:6]}")
    print(f"inverse checked: {len(ups)} checkpoint tensors returned bit for bit from the F32 GGUF", flush=True)
    prompt = [1, 3, 4, 5, 6]
    rng = np.random.default_rng(args.seed)
    results = []
    labels = [x for x in args.types.split(",") if x] + (["UD"] if args.ud else [])
    for label in labels:
        if label == "UD":
            qt = None
        elif label in ("F16", "BF16"):
            qt = getattr(gguf.GGMLQuantizationType, label)
        elif label in base.BLOCKS:
            qt = getattr(gguf.GGMLQuantizationType, label, None)
            if qt is None:
                print(f"{label:8s} not in this gguf-py, skipped"); continue
        else:
            sys.exit(f"unknown type {label}")
        replace, updates = {}, {}
        only = [x for x in args.only.split(",") if x]
        for t in reader.tensors:
            if not is_quantizable(t.name) or (only and not any(o in t.name for o in only)):
                continue
            shape = [int(x) for x in reversed(t.shape)]
            src = np.array(t.data, dtype=np.float32).reshape(shape)
            tl = ud_type(t.name) if label == "UD" else label
            tq = getattr(gguf.GGMLQuantizationType, tl) if label == "UD" else qt
            blck = 1 if tl in ("F16", "BF16") else base.BLOCKS[tl][0]
            if shape[-1] % blck:
                continue                       # a row this type does not tile: stays F32
            if tl in ("F16", "BF16"):
                q = quants.quantize(src, tq)
                deq = quants.dequantize(q, tq).astype(np.float32).reshape(shape)
                replace[t.name] = (tq, q)
            else:
                raw = base.synth_quant(quants, tq, tl, shape, float(np.std(src)) or 0.02, rng)
                deq = base.dequant(quants, tq, tl, raw.reshape(-1)).reshape(shape)
                replace[t.name] = (tq, raw)
            updates.update(to_hf(t.name, deq, cfg) or {})
        updates = finish(updates, hf, cfg)
        vpath = work / f"variant_{label}.gguf"
        base.write_variant(gguf, reader, vpath, replace)
        m2 = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32); m2.eval()
        assign_hf(m2, updates)
        with torch.no_grad(), indexer_ties_by_block():
            ids = torch.tensor([prompt]); past = None; cur = ids; out = list(prompt); steps = []
            for _ in range(args.max_new):
                step = m2(input_ids=cur, past_key_values=past, use_cache=True)
                steps.append(step.logits[0, -1].float().numpy().copy())
                tok = int(step.logits[0, -1].argmax()); out.append(tok)
                cur = torch.tensor([[tok]]); past = step.past_key_values
        # final_logits: the engine's numeric oracle, here against a float32 reference
        ref = {"schema_version": 2, "prompt_ids": prompt, "full_ids": out,
               "final_logits": step.logits[0, -1, :].float().tolist()}
        rpath = work / f"ref_{label}.json"; rpath.write_text(json.dumps(ref))
        d = model_dir(Path(args.link), vpath, work, label)
        ok, detail = True, []
        for cap in args.caps.split(","):
            p = subprocess.run([args.engine, cap, "8", str(rpath)], env=dict({"GQ_EXACT": "1"}, **os.environ, SNAP=str(d), OMP_NUM_THREADS="2"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
            line = [l for l in p.stdout.splitlines() if l.startswith("Matching tokens")]
            got = [l for l in p.stdout.splitlines() if l.startswith("C engine")]
            num = [l for l in p.stdout.splitlines() if l.startswith("Numeric oracle")]
            m = line[-1].split(": ")[-1] if line else f"rc={p.returncode} (no match line)"
            cos = num[-1].split()[2] if num else "cosine=?"
            detail.append(f"cap{cap} {m} {cos}")
            if got:                            # the first divergence, in the reference's own logits
                eng = [int(x) for x in got[-1].split(":", 1)[1].split("Text")[0].split()]
                ref_new = out[len(prompt):]
                k = next((i for i, (a, b) in enumerate(zip(ref_new, eng)) if a != b), None)
                if k is not None:
                    lg = steps[k]; top2 = np.sort(lg)[-2:]
                    detail.append(f"[step {k}: ref {ref_new[k]} {lg[ref_new[k]]:.6f}, engine {eng[k]} "
                                  f"{lg[eng[k]]:.6f}, gap {lg[ref_new[k]] - lg[eng[k]]:.2e}, top-2 margin {top2[1] - top2[0]:.2e}]")
            if not line or not m.endswith(f"{args.max_new}/{args.max_new}"):
                ok = False
                (work / f"engine_{label}_cap{cap}.log").write_text(p.stdout)
        distinct = len(set(out[len(prompt):]))
        print(f"{label:8s} {'OK ' if ok else 'FAIL'} {' '.join(detail)}  ({distinct} distinct ref tokens, "
              f"{len(replace)} tensors as {label})", flush=True)
        results.append((label, ok))
    good = sum(ok for _, ok in results)
    print(f"\n{good}/{len(results)} variants token-exact; work dir {work}")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
