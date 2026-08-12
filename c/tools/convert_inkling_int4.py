#!/usr/bin/env python3
"""Convert the Thinking Machines Inkling BF16 checkpoint (TML-native tensor
names) into a colibri snapshot readable by inkling.c.

Name mapping follows transformers' checkpoint_conversion_mapping for
inkling_mm_model verbatim (conversion_mapping.py), including the CRITICAL
detail that every fused w13 tensor stores gate/up rows INTERLEAVED
(g0,u0,g1,u1,...): de-interleaving = even rows first, then odd rows.
Quantization math is bit-identical to convert_fp8_to_int4.py / the C engine
(np.rint = lrintf, per-row symmetric scales, low nibble = even column).

Output tensor policy (Stage A):
  - routed experts (95% of the params)  -> int4 (or --xbits) + `.qs` f32 row scales
  - norms, sconv weights, rel-bias banks, router (+bias/scales) -> f32
  - everything else (attn projections, dense/shared MLP, embed, lm_head)
    -> bf16 passthrough (st_read_f32 in the C engine converts on load)
  - model.audio.* / model.visual.* / model.mtp.* -> skipped (text-only)

Modes:
  --indir DIR --outdir DIR [--watch]   convert local shards (resumable per
        shard; --watch polls while `hf download` is still running and exits
        when every shard in model.safetensors.index.json is converted)
  --selftest                           numpy-only unit tests (interleave, int4)
  --selftest-e2e HFDIR OUTBASE         build a fake TML-named checkpoint from a
        tiny HF-format snapshot (tools/make_tiny_inkling.py), then convert it
        with passthrough and int4 experts -> OUTBASE-pass/, OUTBASE-i4/
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------- quantization (identical to convert_fp8_to_int4.py) ----------

def quant_int8_rows(w):
    """w: f32 [O,I] -> (int8 bytes [O,I] viewed u8, f32 scales [O])"""
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    s[s < 1e-8] = 1e-8
    q = np.clip(np.rint(w / s), -128, 127).astype(np.int8)
    return q.view(np.uint8), s[:, 0].astype(np.float32)


def quant_int4_rows(w):
    """w: f32 [O,I], I even -> (packed u8 [O,I/2], f32 scales [O]).
    Low nibble = even column, high nibble = odd column, offset +8."""
    O, I = w.shape
    assert I % 2 == 0
    s = np.abs(w).max(axis=1, keepdims=True) / 7.0
    s[s < 1e-8] = 1e-8
    q = np.clip(np.rint(w / s), -8, 7).astype(np.int32)
    lo = (q[:, 0::2] + 8).astype(np.uint8)
    hi = (q[:, 1::2] + 8).astype(np.uint8)
    return (lo | (hi << 4)), s[:, 0].astype(np.float32)


def deinterleave(t: torch.Tensor, dim: int) -> torch.Tensor:
    """TML fused-w13 layout (g0,u0,g1,u1,...) -> grouped ([gate; up]).
    Matches transformers' Interleave(dim) conversion op."""
    n = t.shape[dim]
    idx = torch.cat([torch.arange(0, n, 2), torch.arange(1, n, 2)])
    return t.index_select(dim, idx)


def interleave(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of deinterleave (used only to fabricate test checkpoints)."""
    n = t.shape[dim]
    idx = torch.empty(n, dtype=torch.long)
    idx[0::2] = torch.arange(0, n // 2)
    idx[1::2] = torch.arange(n // 2, n)
    return t.index_select(dim, idx)


# ---------- name mapping (TML native -> HF names inkling.c loads) ----------

SKIP_PREFIXES = ("model.audio.", "model.visual.", "model.mtp.")

# --keep-audio: the audio tower is one embedding table + one norm (~10 MB bf16
# on Inkling-Small) and inkling.c consumes it under its original model.audio.*
# names, so keeping it costs nothing. Vision and MTP stay skipped.
KEEP_AUDIO = False


def skip_name(name):
    if KEEP_AUDIO and name.startswith("model.audio."):
        return False
    return name.startswith(SKIP_PREFIXES)

RENAMES = [
    (r"^model\.llm\.embed\.weight$",      "model.embed_tokens.weight"),
    (r"^model\.llm\.embed_norm\.weight$", "model.embed_norm.weight"),
    (r"^model\.llm\.norm\.weight$",       "model.norm.weight"),
    (r"^model\.llm\.unembed\.weight$",    "lm_head.weight"),
    (r"^model\.llm\.layers\.",            "model.layers."),
]
SUBS = [
    (r"\.attn\.wq_du\.", ".self_attn.q_proj."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.r_proj."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj."),
    (r"\.attn\.q_norm\.", ".self_attn.q_norm."),
    (r"\.attn\.k_norm\.", ".self_attn.k_norm."),
    (r"\.attn\.k_sconv\.weight$", ".self_attn.k_sconv.conv1d.weight"),
    (r"\.attn\.v_sconv\.weight$", ".self_attn.v_sconv.conv1d.weight"),
    (r"\.attn\.rel_logits_proj\.proj$", ".self_attn.rel_logits_proj.proj"),
    (r"\.attn_sconv\.weight$", ".attn_sconv.conv1d.weight"),
    (r"\.mlp_sconv\.weight$",  ".mlp_sconv.conv1d.weight"),
    (r"\.attn_norm\.weight$",  ".input_layernorm.weight"),
    (r"\.mlp_norm\.weight$",   ".post_attention_layernorm.weight"),
    (r"\.mlp\.gate\.bias$",    ".mlp.gate.e_score_correction_bias"),
    (r"\.mlp\.w2_md\.weight$", ".mlp.down_proj.weight"),
    (r"\.mlp\.experts\.w2_weight$", ".mlp.experts.down_proj"),
    (r"\.mlp\.shared_experts\.shared_w2_weight$", ".mlp.shared_experts.down_proj"),
]
# names that stay f32 in the container (small / numerically sensitive;
# the reference keeps every sconv in fp32)
F32_RE = re.compile(
    r"(norm\.weight$|layernorm\.weight$|rel_logits_proj\.proj$|conv1d\.weight$"
    r"|global_scale$|e_score_correction_bias$|\.mlp\.gate\.weight$)"
)


def map_name(name):
    """Simple renames only (fused w13 tensors are handled by the shard loop).
    Returns the mapped name, or None to skip. model.audio.* (when kept) maps
    to itself — the engine loads those names directly."""
    if skip_name(name):
        return None
    for pat, rep in RENAMES:
        name = re.sub(pat, rep, name)
    for pat, rep in SUBS:
        name = re.sub(pat, rep, name)
    return name


# ---------- per-shard conversion ----------

def quantize_expert_tensor(f, name, xbits, out, out_name):
    """Per-expert de-interleave/quantize of the fused 3D expert tensors.
    Slice reads stay sequential (kind to the HDD); the numpy quantization —
    the CPU bottleneck, ~95% of the checkpoint flows through here — fans out
    on a thread pool (ufuncs release the GIL on large arrays). Waves of 16
    keep peak RAM at ~1.2 GB of pending slices instead of the whole tensor."""
    from concurrent.futures import ThreadPoolExecutor
    sl = f.get_slice(name)
    E, R, C = sl.get_shape()             # w13: [E,2I,D] (interleaved), w2: [E,D,I]
    is_w13 = name.endswith("w13_weight")
    if xbits == 4:
        qs = np.empty((E * R, C // 2), np.uint8)
    else:
        qs = np.empty((E * R, C), np.uint8)
    sc = np.empty(E * R, np.float32)

    def work(e, w):
        if is_w13:
            w = deinterleave(w, 0)       # -> [gate rows; up rows]
        w = w.float().numpy()
        q, s = quant_int4_rows(w) if xbits == 4 else quant_int8_rows(w)
        qs[e * R:(e + 1) * R] = q
        sc[e * R:(e + 1) * R] = s

    with ThreadPoolExecutor(max_workers=12) as ex:
        for base in range(0, E, 16):
            futs = [ex.submit(work, e, sl[e]) for e in range(base, min(base + 16, E))]
            for fu in futs:
                fu.result()
    out[out_name] = torch.from_numpy(qs.reshape(-1))
    out[out_name + ".qs"] = torch.from_numpy(sc)


def convert_shard(path, out, xbits):
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            if skip_name(name):
                continue
            base = map_name(name)
            if name.endswith(".mlp.experts.w13_weight"):
                tgt = base.replace(".w13_weight", ".gate_up_proj")
                if xbits:
                    quantize_expert_tensor(f, name, xbits, out, tgt)
                else:
                    t = f.get_tensor(name)
                    out[tgt] = deinterleave(t, 1).contiguous()
                continue
            if name.endswith(".mlp.experts.w2_weight"):
                tgt = map_name(name)
                if xbits:
                    quantize_expert_tensor(f, name, xbits, out, tgt)
                else:
                    out[tgt] = f.get_tensor(name)
                continue
            if name.endswith(".mlp.shared_experts.shared_w13_weight"):
                t = deinterleave(f.get_tensor(name), 1)      # [ns, 2I, D] -> grouped
                half = t.shape[1] // 2
                pref = base.replace(".shared_w13_weight", "")
                out[pref + ".gate_proj"] = t[:, :half].contiguous()
                out[pref + ".up_proj"] = t[:, half:].contiguous()
                continue
            if name.endswith(".mlp.w13_dn.weight"):
                t = deinterleave(f.get_tensor(name), 0)      # [2I, D] -> grouped
                half = t.shape[0] // 2
                pref = base.replace(".w13_dn.weight", "")
                out[pref + ".gate_proj.weight"] = t[:half].contiguous()
                out[pref + ".up_proj.weight"] = t[half:].contiguous()
                continue
            t = f.get_tensor(name)
            if F32_RE.search(base):
                out[base] = t.float()
            else:
                out[base] = t                                 # bf16 passthrough


AUX_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "chat_template.jinja"]


def acquire_output_lock(outdir):
    lock = open(os.path.join(outdir, ".convert.lock"), "w")
    try:
        import fcntl
    except ImportError:
        try:
            import msvcrt
        except ImportError:
            return lock
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            lock.close()
            sys.exit("ERROR: another converter is already using this output directory.")
    else:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            sys.exit("ERROR: another converter is already using this output directory.")
    return lock


def shard_out_name(src):
    m = re.search(r"model-(\d+)-of-\d+\.safetensors$", os.path.basename(src))
    return f"out-{m.group(1)}.safetensors" if m else \
        "out-" + os.path.basename(src)


def convert_dir(indir, outdir, xbits, watch=False, delete_src=False):
    os.makedirs(outdir, exist_ok=True)
    index_path = os.path.join(indir, "model.safetensors.index.json")
    done_src = set()
    while True:
        shards = sorted(glob.glob(os.path.join(indir, "model-*.safetensors")))
        if not shards:
            single = os.path.join(indir, "model.safetensors")
            if os.path.exists(single):
                shards = [single]
        for sp in shards:
            if sp in done_src:
                continue
            dst = os.path.join(outdir, shard_out_name(sp))
            if os.path.exists(dst):
                done_src.add(sp)
                continue
            t0 = time.time()
            out = {}
            convert_shard(sp, out, xbits)
            if out:
                save_file(out, dst + ".tmp")
                os.replace(dst + ".tmp", dst)   # atomic: partial writes never count as done
            else:                                # shard held only skipped (mm/mtp) tensors
                open(dst, "wb").close()
            done_src.add(sp)
            src_gb = os.path.getsize(sp) / 1e9
            dst_gb = os.path.getsize(dst) / 1e9
            print(f"{os.path.basename(sp)} ({src_gb:.1f}G) -> "
                  f"{os.path.basename(dst)} ({dst_gb:.1f}G) in {time.time()-t0:.0f}s",
                  flush=True)
            if delete_src:
                os.remove(sp)
        # completion check: every model-* shard named in the index is
        # converted. The index also references mtp.safetensors (the separate
        # MTP-head file, skipped entirely) — only count files we convert.
        total = None
        if os.path.exists(index_path):
            wm = json.load(open(index_path))["weight_map"]
            total = sorted(s for s in set(wm.values())
                           if re.match(r"model-\d+-of-\d+\.safetensors$", s))
            if total and all(os.path.exists(os.path.join(outdir, shard_out_name(s))) for s in total):
                break
        elif shards and not watch:
            break
        if not watch:
            break
        time.sleep(60)
    for fn in AUX_FILES:
        src = os.path.join(indir, fn)
        if os.path.exists(src):
            shutil.copy(src, outdir)
    print(f"conversion complete -> {outdir}")


# ---------- selftests ----------

def selftest():
    torch.manual_seed(0)
    # 1) interleave round-trip + semantics
    g = torch.arange(0, 12).reshape(6, 2).float()
    u = torch.arange(100, 112).reshape(6, 2).float()
    fused = interleave(torch.cat([g, u]), 0)
    assert torch.equal(fused[0], g[0]) and torch.equal(fused[1], u[0]), \
        "interleave: row 0 must be gate0, row 1 up0"
    got = deinterleave(fused, 0)
    assert torch.equal(got[:6], g) and torch.equal(got[6:], u)
    # 2) int4 pack/unpack round-trip against the C engine's convention
    w = np.random.randn(16, 32).astype(np.float32)
    q, s = quant_int4_rows(w)
    lo = (q & 0x0F).astype(np.int32) - 8
    hi = ((q >> 4) & 0x0F).astype(np.int32) - 8
    deq = np.empty_like(w)
    deq[:, 0::2] = lo
    deq[:, 1::2] = hi
    deq *= s[:, None]
    ref = np.clip(np.rint(w / s[:, None]), -8, 7) * s[:, None]
    assert np.array_equal(deq, ref), "int4 round-trip mismatch"
    # 3) name mapping spot checks
    cases = {
        "model.llm.embed.weight": "model.embed_tokens.weight",
        "model.llm.unembed.weight": "lm_head.weight",
        "model.llm.layers.3.attn.wq_du.weight": "model.layers.3.self_attn.q_proj.weight",
        "model.llm.layers.3.attn.k_sconv.weight": "model.layers.3.self_attn.k_sconv.conv1d.weight",
        "model.llm.layers.3.attn_sconv.weight": "model.layers.3.attn_sconv.conv1d.weight",
        "model.llm.layers.3.attn_norm.weight": "model.layers.3.input_layernorm.weight",
        "model.llm.layers.3.mlp_norm.weight": "model.layers.3.post_attention_layernorm.weight",
        "model.llm.layers.3.mlp.gate.bias": "model.layers.3.mlp.gate.e_score_correction_bias",
        "model.llm.layers.3.mlp.gate.global_scale": "model.layers.3.mlp.gate.global_scale",
        "model.llm.layers.3.mlp.experts.w2_weight": "model.layers.3.mlp.experts.down_proj",
        "model.llm.layers.0.mlp.w2_md.weight": "model.layers.0.mlp.down_proj.weight",
        "model.mtp.layers.0.input_proj.weight": None,
        "model.audio.encoder.weight": None,
        "model.visual.final_norm.weight": None,
    }
    for src, want in cases.items():
        got = map_name(src)
        assert got == want, f"map_name({src}) = {got}, want {want}"
    print("SELFTEST OK")


def selftest_e2e(hfdir, outbase):
    """Fabricate a TML-named checkpoint from a tiny HF snapshot (inverse
    mapping, w13 re-fused AND re-interleaved), then convert it back. Validates
    the converter + the C engine's container reader end-to-end without the
    real download. Tensors stay f32 so the oracle comparison is exact."""
    src = os.path.join(hfdir, "model.safetensors")
    fake = outbase + "-tml"
    os.makedirs(fake, exist_ok=True)
    inv = {}
    with safe_open(src, framework="pt") as f:
        names = list(f.keys())
        tensors = {n: f.get_tensor(n) for n in names}
    layers = {}
    for n in names:
        m = re.match(r"model\.layers\.(\d+)\.(.*)", n)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = tensors[n]
    for n, t in tensors.items():
        if n == "model.embed_tokens.weight": inv["model.llm.embed.weight"] = t
        elif n == "model.embed_norm.weight": inv["model.llm.embed_norm.weight"] = t
        elif n == "model.norm.weight":       inv["model.llm.norm.weight"] = t
        elif n == "lm_head.weight":          inv["model.llm.unembed.weight"] = t
    for li, lt in layers.items():
        p = f"model.llm.layers.{li}."
        for hf, tml in [("self_attn.q_proj.weight", "attn.wq_du.weight"),
                        ("self_attn.k_proj.weight", "attn.wk_dv.weight"),
                        ("self_attn.v_proj.weight", "attn.wv_dv.weight"),
                        ("self_attn.r_proj.weight", "attn.wr_du.weight"),
                        ("self_attn.o_proj.weight", "attn.wo_ud.weight"),
                        ("self_attn.q_norm.weight", "attn.q_norm.weight"),
                        ("self_attn.k_norm.weight", "attn.k_norm.weight"),
                        ("self_attn.rel_logits_proj.proj", "attn.rel_logits_proj.proj"),
                        ("self_attn.k_sconv.conv1d.weight", "attn.k_sconv.weight"),
                        ("self_attn.v_sconv.conv1d.weight", "attn.v_sconv.weight"),
                        ("attn_sconv.conv1d.weight", "attn_sconv.weight"),
                        ("mlp_sconv.conv1d.weight", "mlp_sconv.weight"),
                        ("input_layernorm.weight", "attn_norm.weight"),
                        ("post_attention_layernorm.weight", "mlp_norm.weight"),
                        ("mlp.gate.weight", "mlp.gate.weight"),
                        ("mlp.gate.e_score_correction_bias", "mlp.gate.bias"),
                        ("mlp.gate.global_scale", "mlp.gate.global_scale"),
                        ("mlp.global_scale", "mlp.global_scale"),
                        ("mlp.down_proj.weight", "mlp.w2_md.weight"),
                        ("mlp.experts.down_proj", "mlp.experts.w2_weight"),
                        ("mlp.shared_experts.down_proj", "mlp.shared_experts.shared_w2_weight")]:
            if hf in lt:
                inv[p + tml] = lt[hf]
        if "mlp.gate_proj.weight" in lt:      # dense: fuse + interleave
            inv[p + "mlp.w13_dn.weight"] = interleave(
                torch.cat([lt["mlp.gate_proj.weight"], lt["mlp.up_proj.weight"]]), 0)
        if "mlp.experts.gate_up_proj" in lt:  # routed: re-interleave dim1
            inv[p + "mlp.experts.w13_weight"] = interleave(lt["mlp.experts.gate_up_proj"], 1)
        if "mlp.shared_experts.gate_proj" in lt:
            inv[p + "mlp.shared_experts.shared_w13_weight"] = interleave(
                torch.cat([lt["mlp.shared_experts.gate_proj"],
                           lt["mlp.shared_experts.up_proj"]], dim=1), 1)
    save_file({k: v.contiguous() for k, v in inv.items()},
              os.path.join(fake, "model-00001-of-00001.safetensors"))
    shutil.copy(os.path.join(hfdir, "config.json"), fake)
    print(f"fake TML checkpoint: {fake} ({len(inv)} tensors)")
    for tag, xbits in [("pass", 0), ("i4", 4)]:
        convert_dir(fake, f"{outbase}-{tag}", xbits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir")
    ap.add_argument("--outdir")
    ap.add_argument("--xbits", type=int, default=4, choices=[0, 4, 8],
                    help="routed-expert bits: 4 (default), 8, or 0 = bf16 passthrough")
    ap.add_argument("--watch", action="store_true",
                    help="poll --indir while the download is still running")
    ap.add_argument("--delete-src", action="store_true")
    ap.add_argument("--keep-audio", action="store_true",
                    help="keep model.audio.* (DMel embedding table + norm, ~10 MB) "
                         "so the engine can take audio input")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-e2e", nargs=2, metavar=("HFDIR", "OUTBASE"))
    a = ap.parse_args()
    global KEEP_AUDIO
    KEEP_AUDIO = a.keep_audio
    if a.selftest:
        selftest(); return
    if a.selftest_e2e:
        selftest_e2e(*a.selftest_e2e); return
    if not a.indir or not a.outdir:
        ap.error("--indir and --outdir required")
    os.makedirs(a.outdir, exist_ok=True)
    lock = acquire_output_lock(a.outdir)
    convert_dir(a.indir, a.outdir, a.xbits, watch=a.watch, delete_src=a.delete_src)


if __name__ == "__main__":
    main()
