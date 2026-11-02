#!/usr/bin/env python3
"""Mistral-Large-4's vision tower (Pixtral) as the mmproj GGUF colibri reads (mistral_vision.h),
from the release's own tensors: a container converted from the FP8 release
(tools/convert_fp8_to_int4.py) takes its pictures from it, as a GGUF folder does from its
linked mmproj.

The release (mistralai/Mistral-Large-4-1T-A52B) keeps the tower in BF16 in Mistral's names
(vision_encoder.*, vision_language_adapter.*, patch_merger.*, pre_mm_projector_norm.*). Its
values are the published mmproj's (mmproj-BF16.gguf of the GGUF release) tensor for tensor,
q and k rows in the same interleaved order: checked on the real files, every layer-0 tensor and
the projector's equal bit for bit. So they are written as they are: the matrices BF16, the
norms, the patch kernel and the merger widened to F32, as the published file has them.

  python3 tools/ml4_mmproj.py --release DIR_OR_REPO --out DIR    (DIR/mmproj.gguf + preprocessor_config.json)

DIR_OR_REPO is a folder with the release's safetensors, config.json and processor_config.json,
or the repo id (only the tower's bytes are fetched, about 3.4 GB). The converter calls stash()
for each shard's tower tensors and finish() at the end.
"""
import argparse
import json
import mmap
import os
import re
import shutil
import struct
import sys
from pathlib import Path

PREFIXES = ("vision_encoder.", "vision_language_adapter.", "patch_merger.", "pre_mm_projector_norm.")
LAYER = re.compile(r"^vision_encoder\.transformer\.layers\.(\d+)\.(.+)$")
LAYER_NAMES = {"attention.wq.weight": "attn_q", "attention.wk.weight": "attn_k", "attention.wv.weight": "attn_v",
               "attention.wo.weight": "attn_out", "attention_norm.weight": "ln1", "ffn_norm.weight": "ln2",
               "feed_forward.w1.weight": "ffn_gate", "feed_forward.w3.weight": "ffn_up",
               "feed_forward.w2.weight": "ffn_down"}
GLOBAL_NAMES = {"vision_encoder.patch_conv.weight": "v.patch_embd.weight", "vision_encoder.ln_pre.weight": "v.pre_ln.weight",
                "pre_mm_projector_norm.weight": "mm.input_norm.weight",
                "patch_merger.merging_layer.weight": "mm.patch_merger.weight",
                "vision_language_adapter.w_in.weight": "mm.1.weight", "vision_language_adapter.w_out.weight": "mm.2.weight"}
WIDE = {"v.patch_embd.weight", "v.pre_ln.weight", "mm.input_norm.weight", "mm.patch_merger.weight"}   # F32 in the published file


def wide(name):
    """F32 in the published file: the norms, the patch kernel and the merger"""
    return name in WIDE or name.endswith((".ln1.weight", ".ln2.weight"))
GG = {"F32": 0, "F16": 1, "BF16": 30}
SIZE = {"F32": 4, "F16": 2, "BF16": 2}
# the release's processor_config.json, for a folder without one
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def is_vision(name):
    return name.startswith(PREFIXES)


def gguf_name(name):
    """A release tower tensor's name in the mmproj, or None"""
    if name in GLOBAL_NAMES:
        return GLOBAL_NAMES[name]
    m = LAYER.match(name)
    if m and m.group(2) in LAYER_NAMES:
        return f"v.blk.{m.group(1)}.{LAYER_NAMES[m.group(2)]}.weight"
    return None


def raw(f, name):
    """(dtype, shape, bytes) of a tensor from a safetensors safe_open handle (torch or numpy)"""
    t = f.get_tensor(name)
    if hasattr(t, "numpy"):   # torch
        import torch
        dt = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}.get(t.dtype)
        if dt is None:
            raise SystemExit(f"{name}: dtype {t.dtype}, the tower is read as BF16, F16 or F32")
        bits = t.contiguous().view(torch.int16 if dt != "F32" else torch.int32).numpy()
        return dt, list(t.shape), bits.tobytes()
    dt = {"float16": "F16", "float32": "F32"}.get(str(t.dtype))
    if dt is None:
        raise SystemExit(f"{name}: dtype {t.dtype}, the tower is read as BF16, F16 or F32")
    return dt, list(t.shape), t.tobytes()


def st_write(path, tensors):
    """a safetensors file of {name: (dtype, shape, bytes)}, the bytes as they are"""
    meta, off = {}, 0
    for name, (dt, shape, data) in tensors.items():
        meta[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    head = json.dumps(meta).encode()
    head += b" " * (-len(head) % 8)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(head)) + head)
        for _dt, _shape, data in tensors.values():
            fh.write(data)
    os.replace(tmp, path)


def st_header(read):
    """a safetensors header through read(start, end): ({name: entry}, data base)"""
    n = struct.unpack("<Q", read(0, 8))[0]
    if n > 100 << 20:
        raise SystemExit(f"a safetensors header of {n} bytes")
    meta = json.loads(read(8, 8 + n))
    meta.pop("__metadata__", None)
    return meta, 8 + n


def st_tower(path):
    """{name: (dtype, shape, bytes)} of the tower tensors of a local safetensors file"""
    out = {}
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        meta, base = st_header(lambda a, b: mm[a:b])
        for name, e in meta.items():
            if is_vision(name):
                a, b = e["data_offsets"]
                out[name] = (e["dtype"], e["shape"], mm[base + a:base + b])
    return out


def stash(dirpath, tag, tensors):
    """the converter: one shard's tower tensors kept under dirpath until finish()"""
    os.makedirs(dirpath, exist_ok=True)
    st_write(os.path.join(dirpath, tag + ".safetensors"), tensors)


def _f32(dt, data):
    import numpy as np
    if dt == "F32":
        return bytes(data)
    a = np.frombuffer(data, dtype=np.uint16)
    if dt == "BF16":
        return (a.astype(np.uint32) << 16).tobytes()
    return a.view(np.float16).astype(np.float32).tobytes()


def _kv_bytes(kv):
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    def val(vt, v):
        if vt == 8:
            return s(v)
        if vt == 7:
            return b"\x01" if v else b"\x00"
        if vt == 9:
            et, items = v
            return struct.pack("<IQ", et, len(items)) + b"".join(val(et, i) for i in items)
        return struct.pack({4: "<I", 6: "<f"}[vt], v)
    return b"".join(s(k) + struct.pack("<I", vt) + val(vt, v) for k, vt, v in kv), s


def write(out, tensors, config, processor=None):
    """the mmproj from {release name: (dtype, shape, bytes)}, the release's config.json (its
    vision_config) and processor_config.json; refuses a tower with a tensor missing"""
    vc = config.get("vision_config") or {}
    depth = int(vc["num_hidden_layers"])
    want = list(GLOBAL_NAMES) + [f"vision_encoder.transformer.layers.{l}.{k}" for l in range(depth) for k in LAYER_NAMES]
    missing = [n for n in want if n not in tensors]
    if missing:
        raise SystemExit(f"the vision tower misses {len(missing)} tensor(s), {missing[0]} first")
    ip = (processor or {}).get("image_processor", {})
    rope = vc.get("rope_parameters") or {}
    proj = int(tensors["vision_language_adapter.w_out.weight"][1][0])
    kv = [("general.architecture", 8, "clip"), ("general.type", 8, "mmproj"),
          ("general.name", 8, "Mistral-Large-4 vision tower"), ("clip.has_vision_encoder", 7, True),
          ("clip.projector_type", 8, "pixtral"), ("clip.vision.projection_dim", 4, proj),
          ("clip.vision.image_size", 4, int(ip.get("size", {}).get("longest_edge", vc.get("image_size", 1540)))),
          ("clip.vision.patch_size", 4, int(vc.get("patch_size", 14))),
          ("clip.vision.embedding_length", 4, int(vc["hidden_size"])),
          ("clip.vision.feed_forward_length", 4, int(vc["intermediate_size"])),
          ("clip.vision.block_count", 4, depth), ("clip.vision.attention.head_count", 4, int(vc["num_attention_heads"])),
          ("clip.vision.image_mean", 9, (6, [float(x) for x in ip.get("image_mean", MEAN)])),
          ("clip.vision.image_std", 9, (6, [float(x) for x in ip.get("image_std", STD)])),
          ("clip.vision.attention.layer_norm_epsilon", 6, float(vc.get("layer_norm_eps", 1e-5))),
          ("clip.rope.freq_base", 6, float(rope.get("rope_theta", vc.get("rope_theta", 10000.0)))),
          ("clip.use_silu", 7, str(vc.get("hidden_act", "silu")) == "silu"),
          ("clip.vision.spatial_merge_size", 4, int(config.get("spatial_merge_size", 2)))]
    rows = []
    for src in want:
        dt, shape, data = tensors[src]
        name = gguf_name(src)
        if wide(name) and dt != "F32":
            data, dt = _f32(dt, data), "F32"
        rows.append((name, GG[dt], list(reversed(shape)), data))
    kvb, s = _kv_bytes(kv)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(rows), len(kv)) + kvb
    align = lambda n: (n + 31) // 32 * 32   # noqa: E731
    off = 0
    for name, gt, dims, data in rows:
        head += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", gt, off)
        off = align(off + len(data))
    head += b"\x00" * (align(len(head)) - len(head))
    tmp = str(out) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(head)
        for _name, _gt, _dims, data in rows:
            fh.write(data)
            fh.write(b"\x00" * (align(len(data)) - len(data)))
    os.replace(tmp, out)
    return len(rows), off


def preprocessor(out_dir, config, processor=None):
    """preprocessor_config.json for tools/mistral_image.py (the keys gguf_import writes)"""
    vc = config.get("vision_config") or {}
    ip = (processor or {}).get("image_processor", {})
    conf = {"patch_size": int(vc.get("patch_size", 14)), "spatial_merge_size": int(config.get("spatial_merge_size", 2)),
            "longest_edge": int(ip.get("size", {}).get("longest_edge", vc.get("image_size", 1540))),
            "image_mean": [float(x) for x in ip.get("image_mean", MEAN)],
            "image_std": [float(x) for x in ip.get("image_std", STD)]}
    (Path(out_dir) / "preprocessor_config.json").write_text(json.dumps(conf, indent=1), encoding="utf-8")


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def finish(outdir, source="RELEASE"):
    """the converter's last step: the stashed tower as outdir/mmproj.gguf (the release's config
    kept as config.hf.json). Silent without a stash (a checkpoint without the tower); a stash
    without every tensor (a resume of a conversion made before the tower was kept) is left for
    this tool's own run, which is printed."""
    parts = Path(outdir) / "_mmproj"
    if not parts.is_dir():
        return False
    config = _json(Path(outdir) / "config.hf.json") or _json(Path(outdir) / "config.json") or {}
    processor = _json(Path(outdir) / "processor_config.json")
    if not config.get("vision_config"):
        print("[ML4] the release's config.json has no vision_config: no mmproj written")
        return False
    tensors = {}
    for p in sorted(parts.glob("*.safetensors")):
        tensors.update(st_tower(p))
    try:
        n, size = write(Path(outdir) / "mmproj.gguf", tensors, config, processor)
    except SystemExit as why:
        print(f"[ML4] {why}: no mmproj written. Run\n"
              f"      python3 tools/ml4_mmproj.py --release {source} --out {outdir}")
        return False
    preprocessor(outdir, config, processor)
    tensors.clear()
    shutil.rmtree(parts, ignore_errors=True)
    print(f"[ML4] vision tower: mmproj.gguf, {n} tensors, {size / 1e9:.2f} GB (pictures through coli serve)")
    return True


def from_folder(src):
    tensors = {}
    for p in sorted(Path(src).glob("*.safetensors")):
        tensors.update(st_tower(p))
    return tensors, _json(Path(src) / "config.json") or {}, _json(Path(src) / "processor_config.json")


def from_repo(repo):
    """only the tower's bytes, by HTTP range, the shards the index names"""
    import urllib.request
    from huggingface_hub import hf_hub_download, hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    headers = build_hf_headers()
    meta = {}
    for fn in ("config.json", "processor_config.json", "consolidated.safetensors.index.json"):
        try:
            meta[fn] = json.loads(Path(hf_hub_download(repo, fn)).read_text(encoding="utf-8"))
        except Exception:   # noqa: BLE001 - a missing file is reported below
            meta[fn] = None
    index = meta["consolidated.safetensors.index.json"]
    if not index:
        raise SystemExit(f"{repo}: no consolidated.safetensors.index.json")
    shards = sorted({f for n, f in index["weight_map"].items() if is_vision(n)})
    tensors = {}
    for shard in shards:
        url = hf_hub_url(repo, shard)

        def read(a, b):
            r = urllib.request.Request(url, headers={**headers, "Range": f"bytes={a}-{b - 1}"})
            with urllib.request.urlopen(r, timeout=120) as resp:
                return resp.read()
        st, base = st_header(read)
        names = [n for n in st if is_vision(n)]
        for i, n in enumerate(names):
            a, b = st[n]["data_offsets"]
            tensors[n] = (st[n]["dtype"], st[n]["shape"], read(base + a, base + b))
            print(f"\r{shard}: {i + 1}/{len(names)} tensors", end="", flush=True)
        print()
    return tensors, meta["config.json"] or {}, meta["processor_config.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, help="the release's folder, or its repo id")
    ap.add_argument("--out", required=True, type=Path, help="a folder: mmproj.gguf and preprocessor_config.json go here")
    a = ap.parse_args()
    if Path(a.release).is_dir():
        tensors, config, processor = from_folder(a.release)
    else:
        tensors, config, processor = from_repo(a.release)
    if not config.get("vision_config"):
        sys.exit(f"{a.release}: config.json has no vision_config")
    a.out.mkdir(parents=True, exist_ok=True)
    n, size = write(a.out / "mmproj.gguf", tensors, config, processor)
    preprocessor(a.out, config, processor)
    print(f"{a.out / 'mmproj.gguf'}: {n} tensors, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
