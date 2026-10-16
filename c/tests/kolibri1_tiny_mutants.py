#!/usr/bin/env python3
"""Mutated copies of the tiny Kolibri-1 fixture (mutation matrix).

Every mutation targets an expert that the oracle's own routing EXERCISES (ref.json
`exercised_experts`), so a mutation cannot hide in a weight the test inputs never touch.
Each kind names the gate that is designated to catch it:

  loader     refused by `kolibri1 --validate` / at engine start, before any forward
  numeric    loads fine; caught by the operator/layer numerical gate against the reference
             (ref/probe_engine.py --mutants) and by the stored-oracle logit gate of the
             tiny harness; the argmax/token gate is NOT the designated detector (a
             scale tweak need not move any argmax)

Kinds:
  scale_exp_flip         one exponent bit of one gate_proj grid value flipped (finite)   numeric
  scale_nonfinite        that grid value set to NaN                                      loader (non-finite scale)
  swap_down_up           the expert's down_proj and up_proj tensors exchanged (shapes)   loader (shape)
  swap_gate_up           gate_proj and up_proj exchanged (same shape, silu on the wrong half)  numeric
  grid_transposed_attn   q_proj grid header relabelled [2,8] instead of [8,2]            loader (TRANSPOSED)
  grid_transposed_expert the expert's gate_proj grid data transposed (square [2,2])      numeric
"""
import json
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_kolibri1_loader as TL  # noqa: E402  (shard read/write helpers)

KINDS = {
    "scale_exp_flip": "numeric",
    "scale_nonfinite": "loader",
    "swap_down_up": "loader",
    "swap_gate_up": "numeric",
    "grid_transposed_attn": "loader",
    "grid_transposed_expert": "numeric",
}


def pick_expert(ref, layer=2):
    """the most heavily exercised expert of `layer` according to the oracle's routing"""
    used = ref["exercised_experts"][str(layer)]
    e = max(used, key=lambda k: used[k])
    assert used[e] > 0
    return layer, int(e), used[e]


def make_mutant(kind, src, dst, ref):
    """copy fixture `src` to `dst` and apply `kind`; returns a description dict"""
    if Path(dst).exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    layer, e, count = pick_expert(ref)
    base = f"model.layers.{layer}.mlp.experts.{e}."
    info = {"kind": kind, "designated": KINDS[kind], "layer": layer, "expert": e, "expert_uses": count}
    gate_s = base + "gate_proj.weight_scale_inv"

    def poke(fn):
        def f(entry):
            name, meta, blob = entry
            return name, meta, fn(bytearray(blob))
        return f

    if kind in ("scale_exp_flip", "scale_nonfinite"):
        def fn(b):
            (v,) = struct.unpack_from("<f", b, 0)
            if kind == "scale_exp_flip":
                (bits,) = struct.unpack_from("<I", b, 0)
                struct.pack_into("<I", b, 0, bits ^ (1 << 25))      # exponent bit: the value is multiplied or divided by 16
                (w,) = struct.unpack_from("<f", b, 0)
                info.update(scale_before=v, scale_after=w)
            else:
                struct.pack_into("<f", b, 0, float("nan"))
            info["tensor"] = gate_s
            return bytes(b)
        TL.edit(dst, gate_s, poke(fn))
    elif kind in ("swap_down_up", "swap_gate_up"):
        a, b = ("down_proj", "up_proj") if kind == "swap_down_up" else ("gate_proj", "up_proj")
        ta, tb = base + a + ".weight", base + b + ".weight"
        sa, sb = base + a + ".weight_scale_inv", base + b + ".weight_scale_inv"
        # exchange both the weight and its grid between the two names (same shard or not)
        saved = {}
        for t in (ta, tb, sa, sb):
            shard = TL.find_shard(dst, t)
            for n, m, blob in TL.read_shard(shard):
                if n == t:
                    saved[t] = (m, blob)
        for t, o in ((ta, tb), (tb, ta), (sa, sb), (sb, sa)):
            TL.edit(dst, t, lambda e, o=o: (e[0],) + saved[o])
        info["tensors"] = [ta, tb]
    elif kind == "grid_transposed_attn":
        t = f"model.layers.{layer}.self_attn.q_proj.weight_scale_inv"
        TL.edit(dst, t, lambda e: (e[0], {"dtype": e[1]["dtype"], "shape": list(reversed(e[1]["shape"]))}, e[2]))
        info["tensor"] = t
    elif kind == "grid_transposed_expert":
        def fn(b):
            import numpy as np
            m = np.frombuffer(bytes(b), dtype="<f4").reshape(2, 2)
            assert m[0, 1] != m[1, 0]
            info["tensor"] = gate_s
            return np.ascontiguousarray(m.T).tobytes()
        TL.edit(dst, gate_s, poke(fn))
    else:
        raise KeyError(kind)
    return info


def make_nan_head(src, dst, row=7):
    """copy fixture `src` to `dst` with every element of lm_head row `row` set to bf16 NaN"""
    if Path(dst).exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    def fn(entry):
        name, meta, blob = entry
        b = bytearray(blob)
        H = meta["shape"][1]
        for j in range(H):
            struct.pack_into("<H", b, (row * H + j) * 2, 0x7FC0)
        return name, meta, bytes(b)
    TL.edit(dst, "lm_head.weight", fn)
