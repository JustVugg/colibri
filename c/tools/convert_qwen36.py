#!/usr/bin/env python3
"""Convert Qwen3.6-35B-A3B (or any Qwen3.5/3.6 MoE) HF checkpoint -> colibri container.

Phase-2 converter: it converts ALL layers (GATED-ATTENTION full_attention AND the
Gated DeltaNet linear_attention layers). Every layer also carries its MoE/MLP block.
DeltaNet layers are stored under `model.layers.{i}.linear_attn.*` (in_proj_qkv/z/b/a,
conv1d.weight, dt_bias, A_log, norm.weight, out_proj). The engine implements the
recurrent gated-delta-rule for these layers; in Phase 1 they were skipped as identity.

Design notes (must stay in sync with c/qwen36.c):
  * The colibri container is just a directory of safetensors shards (+ config.json +
    qwen36_meta.json). The engine reads them lazily via st_init, paging expert weights
    from disk -- that is the whole "run a huge model in little RAM" trick.
  * Expert weights are stored per-expert as `model.layers.{a}.mlp.experts.{e}.merged_weight`
    + `.qs` (f32 scales). For --ebits >= 5 the merged_weight is int8 (layout [gate|up|down]);
    for --ebits <= 4 it is TRUE 4-bit packed uint8 (2 elements/byte, half size) -- see
    pack_int4(). The qs layout (per-row f32 scales) is identical either way. This matches
    what c/qwen36.c's load_expert_merged expects: it detects int4 by ON-DISK SIZE (N/2 bytes)
    and unpacks in-place to int8, so the rest of the MoE path is unchanged.
  * Attention + router + shared-expert + norms stay f16 (they are tiny vs experts).
  * Real Qwen3.6 is a vision-language checkpoint: config dims live under `text_config`,
    and weight keys are prefixed `model.language_model.`. Both are handled transparently.
  * The fused expert tensors `mlp.experts.gate_up_proj` / `down_proj` are split per expert
    into the merged_weight layout (gate_up = [gate; up] along dim 0).
  * Head dims are derived from the actual weight shapes (authoritative), not from config
    heuristics, because Qwen3.6's qk split (q_head_dim=512, k/v_head_dim=256, rope over
    256 dims) does not match naive head_dim=hidden/n_heads.

Usage (cloud, e.g. Colab free CPU):
  python tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B --out ./qwen36_i4 --ebits 4 \\
      --low-disk --stream-upload --upload-repo minne100/qwen36-35b-a3b-colibri-i4

Usage (local tiny model for end-to-end testing):
  python tools/convert_qwen36.py --model ../qwen36_tiny --out ../qwen36_tiny_i4 --ebits 8
"""

import argparse, json, math, os, struct, sys
from pathlib import Path

# Windows: force UTF-8 output
if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

try:
    import torch
    from safetensors.numpy import safe_open as safe_open_np
    from safetensors.torch import safe_open as safe_open_pt
    from safetensors.torch import save_file
except ImportError as exc:
    sys.exit(f"Missing dependencies: {exc}. Install: pip install torch safetensors")


def quantize_row(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise symmetric quantization to `bits` (2..8). Storage is int8 regardless of
    bits (engine dequantizes q*scale); for bits<8 values are confined to a smaller range."""
    qmax = (1 << (bits - 1)) - 1
    w_f32 = w.float()
    row_max = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = row_max / qmax
    q = (w_f32 / scales).round().clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scales.squeeze(1)


def pack_int4(q: "torch.Tensor") -> "torch.Tensor":
    """Pack a 1D int-tensor (values already confined to the signed-4-bit range [-8,7])
    into a uint8 blob, 2 values per byte (LOW nibble = element 2k, HIGH nibble = element 2k+1).

    Convention (kept in sync with the c/qwen36.c int4 dequant path, TODO):
      byte k  = (signed4(element[2k+1]) << 4) | (signed4(element[2k]) & 0xF)
    Each nibble is a SIGNED 4-bit integer in [-8,7]; dequant = nibble * scale[row].
    The whole expert merged blob (gate||up||down, flattened) is packed as one contiguous
    uint8 array, so its length is exactly N/2 (N = element count, always even here).
    """
    q = q.to(torch.int16).clamp(-8, 7)
    n = q.numel()
    if n & 1:
        q = torch.cat([q, torch.zeros(1, dtype=torch.int16)])
    lo = (q[0::2] & 0xF).to(torch.uint8)
    hi = ((q[1::2] & 0xF) << 4).to(torch.uint8)
    return (lo | hi).contiguous()


def quantize_row_grouped(w: "torch.Tensor", bits: int, gs: int):
    """Group-wise symmetric quantization: one f32 scale per `gs` input elements
    per row (like GLM's gs64 containers). Returns (q int8 [O,I], scales f32
    [O, ceil(I/gs)] flattened row-major)."""
    qmax = (1 << (bits - 1)) - 1
    O, I = w.shape[0], w.reshape(w.shape[0], -1).shape[1]
    w_f32 = w.reshape(O, -1).float()
    pad = (-I) % gs
    if pad:
        w_f32 = torch.nn.functional.pad(w_f32, (0, pad))
    ng = w_f32.shape[1] // gs
    g = w_f32.view(O, ng, gs)
    scales = g.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / qmax
    q = (g / scales).round().clamp(-qmax - 1, qmax).to(torch.int8).view(O, -1)[:, :I]
    return q, scales.view(O, ng)


def make_merged(gate, up, down, ebits, gs=0):
    gsz = gs   # local `gs` is rebound to the gate scales below -- keep the group size safe
    """gate/up: [inter, H]; down: [H, inter] (torch, any fp).
    -> (merged_weight, qs f32 1D).

    Storage format depends on ebits:
      * ebits >= 5 : merged_weight is int8  (1 byte / element), as before.
      * ebits <= 4 : merged_weight is TRUE 4-bit packed uint8 (2 elements / byte,
                     half the size) -- see pack_int4 for the nibble convention.
    The engine (c/qwen36.c) currently reads ebits>=5 (int8); the int4 path is WIP.
    qs (per-row f32 scales) is identical in both cases.
    """
    def q(t):
        if gsz:
            qt, s = quantize_row_grouped(t, ebits, gsz)   # [O, ng] scales
            return qt, s.reshape(-1)
        qt, s = quantize_row(t.reshape(t.shape[0], -1), ebits)  # rows along dim0
        return qt, s
    gq, gs = q(gate)
    uq, us = q(up)
    dq, ds = q(down)
    mw_i8 = torch.cat([gq.flatten(), uq.flatten(), dq.flatten()]).contiguous()
    if ebits <= 4:
        mw = pack_int4(mw_i8)              # uint8, 2x4-bit per byte -> HALF size (true int4)
    else:
        mw = mw_i8.to(torch.int8)          # int8 storage (ebits 5..8)
    qs = torch.cat([gs, us, ds]).contiguous().float()
    return mw, qs


def _unpack_int4(packed: "torch.Tensor") -> "torch.Tensor":
    """Inverse of pack_int4: uint8 blob -> signed int8 values (HIGH nibble = element 2k+1)."""
    b = packed.to(torch.uint8)
    lo = (b & 0xF).to(torch.int8)
    hi = ((b >> 4) & 0xF).to(torch.int8)
    # sign-extend 4-bit -> 8-bit
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=1).flatten()


def _selftest():
    """Round-trip check for the true-int4 packing used by --ebits<=4."""
    import random
    print("=== int4 pack/unpack selftest ===")
    ok = True
    for ebits in (2, 3, 4):
        # small synthetic expert: gate[inter,H] up[inter,H] down[H,inter]
        inter, H = 8, 16
        g = torch.randn(inter, H) * 3
        u = torch.randn(inter, H) * 3
        d = torch.randn(H, inter) * 3
        mw, qs = make_merged(g, u, d, ebits)
        # serialize to safetensors + reload (exercises the real dtype path)
        import tempfile, os
        td = tempfile.mkdtemp()
        from safetensors.torch import save_file, load_file
        save_file({"merged_weight": mw, "qs": qs}, os.path.join(td, "e0.safetensors"))
        back = load_file(os.path.join(td, "e0.safetensors"))
        mw_r = back["merged_weight"]
        # rebuild the int8 reference exactly as make_merged did (pre-pack)
        def qref(t):
            qt, _ = quantize_row(t.reshape(t.shape[0], -1), ebits)
            return qt.flatten()
        ref = torch.cat([qref(g), qref(u), qref(d)]).contiguous().to(torch.int8)
        if mw_r.dtype != torch.uint8:
            print(f"  ebits={ebits}: FAIL dtype={mw_r.dtype} (expected uint8)")
            ok = False
            continue
        rec = _unpack_int4(mw_r).to(torch.int8)[:ref.numel()]
        maxerr = (rec.int() - ref.int()).abs().max().item()
        int8_bytes = ref.numel()
        int4_bytes = mw_r.numel()
        exact = bool(maxerr == 0)
        halved = bool(int4_bytes * 2 == int8_bytes)
        print(f"  ebits={ebits}: maxerr={maxerr} exact={exact} "
              f"int4_bytes={int4_bytes} int8_bytes={int8_bytes} halved={halved}")
        ok = ok and exact and halved
    # also confirm ebits>=5 still stores int8
    g = torch.randn(8, 16) * 3; u = torch.randn(8, 16) * 3; d = torch.randn(16, 8) * 3
    mw8, _ = make_merged(g, u, d, 8)
    print(f"  ebits=8: dtype={mw8.dtype} (expected int8), bytes={mw8.numel()}")
    ok = ok and (mw8.dtype == torch.int8)
    print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def resolve_prefix(keys):
    """Return the layer-weight prefix so that `prefix + "layers.{i}."` matches the
    actual weight keys.  Handles the common layouts:
      * 'model.'                for `model.layers.0...`            (standard HF Qwen3 MoE)
      * 'model.language_model.' for `model.language_model.layers.0...` (some VL checkpoints)
    """
    for k in keys:
        if k.startswith("model.language_model.layers."):
            return "model.language_model."
    for k in keys:
        if k.startswith("model.layers."):
            return "model."
    return ""


def main():
    ap = argparse.ArgumentParser(description="Convert Qwen3.6 MoE HF checkpoint -> colibri container")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--repo", help="HuggingFace repo ID (will be downloaded)")
    src.add_argument("--model", help="Local HF checkpoint directory")
    ap.add_argument("--out", required=False, help="Output container directory")
    ap.add_argument("--ebits", type=int, default=4, help="Expert quant bits (2..8, default 4)")
    ap.add_argument("--gs", type=int, default=0,
                    help="Group size for expert scales (e.g. 64). 0 = per-row (default). "
                         "Group-scaled containers need engine support (expert_gs in meta).")
    ap.add_argument("--upload-repo", help="Push the finished container to this HF repo ID")
    ap.add_argument("--hf-token", help="HF token (defaults to HF_TOKEN env / cached login)")
    ap.add_argument("--low-disk", action="store_true",
                    help="Download source shards one at a time, delete each after its layers are "
                         "converted (only the generated container is kept).")
    ap.add_argument("--stream-upload", action="store_true",
                    help="Upload each generated shard to --upload-repo immediately, then delete it "
                         "locally. Keeps local disk at ~one shard.")
    ap.add_argument("--no-readme", action="store_true", help="Skip README.md generation")
    ap.add_argument("--selftest", action="store_true",
                    help="Validate the int4 pack/unpack round-trip on synthetic weights, then exit "
                         "(no model conversion). Checks that packing recovers the exact quantized "
                         "integers and that int4 halves the storage vs int8.")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not (args.repo or args.model):
        sys.exit("error: need --repo or --model (or --selftest)")
    if not args.out:
        sys.exit("error: --out is required")

    if not 2 <= args.ebits <= 8:
        sys.exit(f"--ebits must be 2..8 (got {args.ebits})")

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.repo:
        from huggingface_hub import hf_hub_download, HfApi
        api = HfApi(token=token)
    else:
        api = None

    # ---- locate source ----
    if args.repo:
        # download config + index only first (cheap) to learn structure
        cfg_path = hf_hub_download(args.repo, "config.json", token=token)
        try:
            idx_path = hf_hub_download(args.repo, "model.safetensors.index.json", token=token)
        except Exception:
            idx_path = None
        src_dir = None
    else:
        src_dir = Path(args.model)
        if not (src_dir / "config.json").is_file():
            sys.exit(f"config.json missing in {src_dir}")
        cfg_path = str(src_dir / "config.json")
        idx_path = str(src_dir / "model.safetensors.index.json") if (src_dir / "model.safetensors.index.json").is_file() else None

    cfg_full = json.load(open(cfg_path, encoding="utf-8"))
    # VL checkpoints nest dims under text_config
    mcfg = cfg_full.get("text_config", cfg_full)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    # The engine reads FLAT keys from config.json, but VL checkpoints nest them
    # under text_config -> write the flattened text config (the original goes to
    # config.hf.json for reference). Fixes "missing hidden_size" on VL models.
    with open(out / "config.json", "w", encoding="utf-8") as cf:
        json.dump(mcfg, cf, indent=1)
    shutil.copy2(cfg_path, out / "config.hf.json")
    print(f"config -> {out / 'config.json'} (flat; model_type={cfg_full.get('model_type')})")
    # The engine needs tokenizer.json next to the shards (TOK= override exists,
    # but a container should run out of the box).
    try:
        if args.repo:
            tok_path = hf_hub_download(args.repo, "tokenizer.json", token=token)
        else:
            tok_path = str(src_dir / "tokenizer.json")
        if Path(tok_path).is_file():
            shutil.copy2(tok_path, out / "tokenizer.json")
            print(f"tokenizer -> {out / 'tokenizer.json'}")
        else:
            print("WARNING: tokenizer.json not found; the engine will need TOK=<path>")
    except Exception as e:
        print(f"WARNING: could not fetch tokenizer.json ({e}); the engine will need TOK=<path>")

    # ---- build weight map (key -> shard file) ----
    if idx_path:
        wm = json.load(open(idx_path, encoding="utf-8"))["weight_map"]
    else:
        # no index: enumerate local safetensors
        if args.repo:
            sys.exit("remote repo without index.json is unsupported; download the repo first")
        wm = {}
        for sh in sorted(src_dir.glob("*.safetensors")):
            with safe_open_np(str(sh), framework="np") as f:
                for k in f.keys():
                    wm[k] = sh.name

    # ---- detect prefix + layer count from weights ----
    sample_layer_keys = [k for k in wm if ".layers.0." in k or ".layers.0." in k]
    prefix = resolve_prefix(wm.keys())
    # find max layer index present
    import re
    layer_ids = set(int(m) for m in re.findall(r"\." + re.escape(prefix) + r"layers\.(\d+)\.", " ".join(wm.keys())))
    n_layers_weight = (max(layer_ids) + 1) if layer_ids else mcfg.get("num_hidden_layers", 0)

    layer_types = mcfg.get("layer_types")
    if layer_types is None:
        # infer: every 4th layer is full_attention (Qwen3.5/3.6 convention)
        n = mcfg.get("num_hidden_layers", n_layers_weight)
        layer_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(n)]
    full_idx = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
    all_idx = list(range(len(layer_types)))
    print(f"layers: {len(layer_types)} total, {len(full_idx)} full_attention (Phase 2: all {len(all_idx)} stored)")

    # ---- shard cache / eviction ----
    tmp = out / ".src_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    local_shards = {}     # shard name -> local path
    shard_handles = {}    # shard name -> safe_open handle
    shard_layers = {}     # shard name -> set of layer ids needing it (-1 = globals)

    def layer_keys(i):
        p = f"{prefix}layers.{i}."
        return [k for k in wm if k.startswith(p)]

    for i in all_idx:
        for k in layer_keys(i):
            shard_layers.setdefault(wm[k], set()).add(i)
    globals_keys = [k for k in (f"{prefix}embed_tokens.weight",
                                "lm_head.weight", f"{prefix}lm_head.weight",
                                f"{prefix}norm.weight")
                    if k in wm]
    for k in globals_keys:
        shard_layers.setdefault(wm[k], set()).add(-1)

    def ensure_shard(name):
        if name in local_shards:
            return local_shards[name]
        if args.repo:
            p = hf_hub_download(args.repo, name, token=token, local_dir=str(tmp),
                                local_dir_use_symlinks=False)
        else:
            p = str(src_dir / name)
        local_shards[name] = p
        return p

    def get_tensor(name):
        shard = wm[name]
        ensure_shard(shard)
        if shard not in shard_handles:
            # torch backend returns a native torch tensor (bf16 works natively);
            # numpy backend returns a bf16 numpy array that torch.from_numpy rejects.
            shard_handles[shard] = safe_open_pt(local_shards[shard], framework="pt")
        return shard_handles[shard].get_tensor(name).contiguous()

    def evict_layer(i):
        for shard, lays in list(shard_layers.items()):
            lays.discard(i)
            if not lays:
                if args.repo:   # only delete files we downloaded; for local keep them on disk
                    h = shard_handles.pop(shard, None)
                    try:
                        if h is not None:
                            h.__exit__(None, None, None)
                    except Exception:
                        pass
                    p = local_shards.pop(shard, None)
                    if p is not None:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                # For local source we keep the handle open (the file stays on disk and
                # the post-loop meta head-dim derivation re-reads it). Only the remote
                # path pops/closes, since it deletes the downloaded shard.
                shard_layers.pop(shard, None)

    stream_api = None
    if args.stream_upload or args.upload_repo:
        from huggingface_hub import HfApi
        stream_api = HfApi(token=token)
        if args.upload_repo:
            stream_api.create_repo(args.upload_repo, repo_type="model", exist_ok=True, private=False)

    def upload_local(path: Path):
        if stream_api and args.upload_repo:
            stream_api.upload_file(repo_id=args.upload_repo, repo_type="model",
                                   path_in_repo=path.name, path_or_fileobj=str(path))
            print(f"  streamed -> {args.upload_repo}/{path.name}")

    # ---- resume: discover already-uploaded shards so a restart skips them ----
    done_files = set()
    if args.upload_repo and stream_api is not None:
        try:
            done_files = set(stream_api.list_repo_files(args.upload_repo, repo_type="model"))
            print(f"[resume] {len(done_files)} file(s) already on {args.upload_repo}; "
                  f"will skip finished layers.")
        except Exception as e:
            done_files = set()
            print(f"[resume] could not list repo ({e}); running full conversion.")

    # ---- globals shard (embed / lm_head / final norm) as f16 ----
    if "model-globals.safetensors" in done_files:
        print("[globals] already on HF, skip")
    else:
        g_out = {}
        for k in globals_keys:
            arr = get_tensor(k).half()
            newk = k.replace("model.language_model.", "model.")
            g_out[newk] = arr
        gpath = out / "model-globals.safetensors"
        save_file(g_out, str(gpath))
        print(f"[globals] {gpath.name} ({len(g_out)} tensors)")
        if args.stream_upload:
            upload_local(gpath)
            gpath.unlink()

    # ---- per layer: Gated-Attention (full_attention) AND Gated DeltaNet (linear_attention) ----
    # Every layer also carries its MoE/MLP block. DeltaNet layers export linear_attn.* which
    # the generic f16 copy below handles automatically; the engine implements the recurrence.
    active = 0
    for i in all_idx:
        a = active
        active += 1
        out_name = f"model-{a:05d}.safetensors"
        if out_name in done_files:
            print(f"[layer {i} -> active {a}] {out_name} already on HF, skip")
            evict_layer(i)
            continue
        tens = {}
        lk = layer_keys(i)
        # ---- MoE experts: handle BOTH layouts the source may use ----
        #  (a) FUSED:  model.layers.i.mlp.experts.gate_up_proj [E,2*inter,H]
        #              + model.layers.i.mlp.experts.down_proj    [E,H,inter]
        #  (b) SEPARATE (standard HF Qwen3 MoE, incl. the real 35B):
        #              model.layers.i.mlp.experts.{e}.gate_proj [inter,H]
        #              model.layers.i.mlp.experts.{e}.up_proj   [inter,H]
        #              model.layers.i.mlp.experts.{e}.down_proj [H,inter]
        sep = {}  # e -> {'gate':Tensor,'up':Tensor,'down':Tensor}
        for k in lk:
            if re.search(r"\.experts\.gate_up_proj(?:\.weight)?$", k):
                gu = get_tensor(k).float()            # [E, 2*inter, H]
                E, twoI, H = gu.shape
                inter = twoI // 2
                gate = gu[:, :inter, :]
                up = gu[:, inter:, :]
                dk = k.replace("gate_up_proj", "down_proj")
                down = get_tensor(dk).float()        # [E, H, inter]
                for e in range(E):
                    mw, qs = make_merged(gate[e], up[e], down[e], args.ebits, gs=args.gs)
                    tens[f"model.layers.{a}.mlp.experts.{e}.merged_weight"] = mw
                    tens[f"model.layers.{a}.mlp.experts.{e}.qs"] = qs
                continue
            ms = re.search(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)(?:\.weight)?$", k)
            if ms:
                e = int(ms.group(1)); proj = ms.group(2)
                sep.setdefault(e, {})[proj] = get_tensor(k).float()
                continue
        for e in sorted(sep):
            d = sep[e]
            if "gate_proj" in d and "up_proj" in d and "down_proj" in d:
                mw, qs = make_merged(d["gate_proj"], d["up_proj"], d["down_proj"], args.ebits, gs=args.gs)
                tens[f"model.layers.{a}.mlp.experts.{e}.merged_weight"] = mw
                tens[f"model.layers.{a}.mlp.experts.{e}.qs"] = qs
            else:
                print(f"[layer {i}] WARN expert {e} missing some proj "
                      f"(have {sorted(d.keys())}) — skipped")
        for k in lk:
            newk = k.replace(f"{prefix}layers.{i}.", f"model.layers.{a}.")
            if "mlp.experts." in newk:
                continue  # already handled above
            # everything else stays f16
            tens[newk] = get_tensor(k).half()
        out_path = out / f"model-{a:05d}.safetensors"
        save_file(tens, str(out_path))
        print(f"[layer {i} -> active {a}] {out_path.name} ({len(tens)} tensors)")
        if args.stream_upload:
            upload_local(out_path)
            out_path.unlink()
        evict_layer(i)

    # ---- derive authoritative head dims from a real full-attention layer's weights ----
    meta = {
        "model_type": cfg_full.get("model_type"),
        "hidden": int(mcfg["hidden_size"]),
        "n_layers": int(mcfg["num_hidden_layers"]),
        "n_active": len(all_idx),
        "layer_types": layer_types,
        "num_experts": int(mcfg["num_experts"]),
        "topk": int(mcfg["num_experts_per_tok"]),
        "moe_inter": int(mcfg.get("moe_intermediate_size", mcfg.get("intermediate_size", 0) // 2)),
        "shared_inter": int(mcfg.get("shared_expert_intermediate_size", mcfg.get("moe_intermediate_size", 0))),
        "rms_eps": float(mcfg.get("rms_norm_eps", 1e-6)),
        "ebits": args.ebits,
        "scoring_func": mcfg.get("scoring_func", "softmax"),
        "n_group": int(mcfg.get("n_group", 1)),
        "topk_group": int(mcfg.get("topk_group", 1)),
        "norm_topk_prob": bool(mcfg.get("norm_topk_prob", False)),
        "attn_output_gate": bool(mcfg.get("attn_output_gate", False)),
        "rope_theta": float((mcfg.get("rope_parameters") or {}).get("rope_theta", 10000000.0)),
        "mrope_section": (mcfg.get("rope_parameters") or {}).get("mrope_section", [16, 16, 16]),
        "partial_rotary_factor": float(mcfg.get("partial_rotary_factor", 0.25)),
    }
    # derive head dims from first full-attention layer weights (authoritative).
    # Read from SOURCE (not local output) so this works even when layer 0 was
    # skipped during a resume and its local shard was already stream-deleted.
    if full_idx:
        fi = full_idx[0]
        def shp_src(nm):
            k = f"{prefix}layers.{fi}.self_attn.{nm}.weight"
            if k in wm:
                return tuple(get_tensor(k).shape)
            return None
        qp = shp_src("q_proj")
        kp = shp_src("k_proj")
        vp = shp_src("v_proj")
        op = shp_src("o_proj")
        qn = shp_src("q_norm")
        n_q = int(mcfg["num_attention_heads"])
        n_kv = int(mcfg["num_key_value_heads"])
        meta["q_heads"] = n_q
        meta["kv_heads"] = n_kv
        if qp is not None:
            meta["q_head_dim"] = qp[0] // n_q
        if kp is not None:
            meta["k_head_dim"] = kp[0] // n_kv
        if vp is not None:
            meta["v_head_dim"] = vp[0] // n_kv
        if op is not None:
            meta["o_in"] = op[1]
        if qn is not None:
            meta["qk_rope_head_dim"] = qn[0]
        meta["expert_gs"] = args.gs   # 0 = per-row scales; >0 = group size along input dim
        meta["head_dim"] = meta.get("k_head_dim", meta.get("q_head_dim", 256))
        meta["rope_dim"] = meta.get("qk_rope_head_dim", meta["head_dim"] // 4)
    # ---- DeltaNet (linear_attention) dims. From config (authoritative for dn; unlike the
    #      attention qk split, dn dims are explicit in config and never evicted shards). ----
    meta["dn_vheads"] = int(mcfg.get("linear_num_value_heads", mcfg.get("num_value_heads", 0)))
    meta["dn_kheads"] = int(mcfg.get("linear_num_key_heads", mcfg.get("num_key_heads", 0)))
    meta["dn_kdim"] = int(mcfg.get("linear_key_head_dim", mcfg.get("key_head_dim", 0)))
    meta["dn_vdim"] = int(mcfg.get("linear_value_head_dim", mcfg.get("value_head_dim", 0)))
    meta["dn_convk"] = int(mcfg.get("linear_conv_kernel_dim", mcfg.get("conv_kernel_size", 0)))
    meta["dn_conv_dim"] = meta["dn_kheads"] * meta["dn_kdim"] * 2 + meta["dn_vheads"] * meta["dn_vdim"]
    # close remaining source handles now that we've finished reading for meta
    for h in shard_handles.values():
        try:
            h.__exit__(None, None, None)
        except Exception:
            pass
    (out / "qwen36_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[meta] {out / 'qwen36_meta.json'}")

    # ---- README ----
    if not args.no_readme:
        lines = ["---", "tags:", "  - colibri", "  - qwen3.6", "  - qwen3.6-35b-a3b", "  - moe",
                 "library_name: colibri", "---", "",
                 "colibri container for Qwen3.6-35B-A3B (Phase 2: all layers, incl. Gated DeltaNet).",
                 "Every layer (Gated-Attention + Gated DeltaNet linear_attention) carries its",
                 "MoE/MLP block. DeltaNet weights live under model.layers.{i}.linear_attn.* and are",
                 "run by the recurrent gated-delta-rule in the colibri `qwen36` engine.",
                 "", "Engine: https://github.com/JustVugg/colibri (c/qwen36.c)"]
        (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- final upload (non-stream path) + cleanup ----
    if args.upload_repo:
        print(f"\nUploading to HF Hub: {args.upload_repo}")
        for f in sorted(out.glob("*.safetensors")) + [out / "config.json", out / "config.hf.json",
                                                       out / "tokenizer.json",
                                                       out / "qwen36_meta.json", out / "README.md"]:
            if f.is_file():
                stream_api.upload_file(repo_id=args.upload_repo, repo_type="model",
                                       path_in_repo=f.name, path_or_fileobj=str(f))
        print(f"Uploaded. Pull on your laptop with:\n"
              f"  python -c \"from huggingface_hub import snapshot_download; "
              f"print(snapshot_download('{args.upload_repo}'))\"")
    # never delete the generated container
    try:
        if tmp.is_dir():
            import shutil as _sh
            _sh.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass

    print(f"\nDone. Container at: {out}")
    print(f"All {len(all_idx)} layers stored (incl. Gated DeltaNet). n_active={len(all_idx)}.")
    print(f"Run (engine):  SNAP={out} ./qwen36 16 {args.ebits} ref_qwen36.json")


if __name__ == "__main__":
    main()
