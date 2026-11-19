#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Repack Qwen3.8-Flash-Next's MXFP4 routed experts (amd/Qwen3.8-Flash-Next-Quark-MXFP4)
into a WEX1 container for qwen38 (COLI_WEX_PATH, qwen38_wex.h).

Every expert becomes one raw record (codec 0): the E8M0 scales of w1, w2, w3, then their
FP4 E2M1 nibbles, w1 = gate_proj, w2 = down_proj, w3 = up_proj -- the Quark tensors' bytes
unchanged (U8 nibbles [rows, cols/2], low nibble = even column; U8 scales [rows, cols/32]),
the slab layout the V4.1 containers and WNR3 use. Only routed experts of the model's layers
go in; the shared expert, the trunk, the PLE and the MTP head stay the FP8 snapshot's.

Before a layer is written, sampled experts are dequantized and compared with the FP8
snapshot's same expert (cosine per matrix): a nibble order, scale axis or name mix-up
shows as a cosine near 0, a wrong expert as a low one. The container is published
atomically and checked with repack_v4_wex.validate_container; a manifest beside it
(<output>.manifest.json) holds the sha256 of every layer's records and of the header
page + table, the inputs' sha256 the downloader verified (<shard>.verified), and the
FP8 check. --verify re-reads a container against its manifest (a mirror copy, a later run).

  python tools/repack_q38_mxfp4_wex.py --quark-dir U:/models/Qwen3.8-Flash-Next-MXFP4-AMD \\
      --fp8-dir U:/models/Qwen3.8-Flash-Next-FP8 --output U:/kaira_wex/q38_mxfp4.wex [--layers 10]
  python tools/repack_q38_mxfp4_wex.py --verify U:/kaira_wex/q38_mxfp4.wex
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

_SPEC = importlib.util.spec_from_file_location("kaira_repack_v4_wex", Path(__file__).with_name("repack_v4_wex.py"))
wex = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wex)

FORMAT = "colibri.qwen38.mxfp4-wex1"
MATRICES = ("gate_proj", "down_proj", "up_proj")   # WEX w1, w2, w3
PREFIXES = ("model.language_model.layers.", "model.layers.")
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3_table():
    v = np.arange(256)
    sign = np.where(v & 0x80, -1.0, 1.0)
    exp, man = (v >> 3) & 15, v & 7
    val = np.where(exp == 0, man / 8.0 * 2.0 ** -6, (1 + man / 8.0) * 2.0 ** (exp - 7.0))
    val = sign * val
    val[(v & 0x7F) == 0x7F] = np.nan
    return val.astype(np.float32)


E4M3 = e4m3_table()


def geometry(config_path):
    with open(config_path, "r", encoding="utf-8") as source:
        config = json.load(source)
    text = config.get("text_config", config)
    keys = ("num_hidden_layers", "num_experts", "hidden_size", "moe_intermediate_size")
    values = [text.get(key) for key in keys]
    if any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError(f"{config_path}: needs positive integers {keys}")
    layers, experts, hidden, inter = values
    if hidden % 32 or inter % 32 or max(hidden, inter) > 65535:
        raise ValueError("WEX1 MXFP4 records need hidden and moe_intermediate_size multiples of 32, at most 65535")
    return layers, experts, hidden, inter


def mxfp4_values(nibbles, scales, rows, cols):
    w = np.frombuffer(nibbles, np.uint8).reshape(rows, cols // 2)
    s = np.frombuffer(scales, np.uint8).reshape(rows, cols // 32)
    v = np.stack([E2M1[w & 15], E2M1[w >> 4]], -1).reshape(rows, cols)
    return v * np.repeat(np.exp2(s.astype(np.float32) - 127), 32, 1)


def fp8_values(weight, scale_inv, rows, cols, scale_dtype):
    w = E4M3[np.frombuffer(weight, np.uint8)].reshape(rows, cols)
    br, bc = -(-rows // 128), -(-cols // 128)
    if scale_dtype == "BF16":
        s = (np.frombuffer(scale_inv, np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif scale_dtype == "F32":
        s = np.frombuffer(scale_inv, np.float32)
    else:
        raise ValueError(f"FP8 scale dtype {scale_dtype} not handled")
    s = s.reshape(br, bc)
    return w * np.repeat(np.repeat(s, 128, 0), 128, 1)[:rows, :cols]


def prefix_of(tensors):
    for prefix in PREFIXES:
        if any(key.startswith(prefix) and ".mlp.experts." in key for key in tensors):
            return prefix
    raise ValueError("no routed expert tensors (model.language_model.layers.N.mlp.experts.E.*) found")


def expert_plan(tensors, prefix, layer, experts, hidden, inter):
    """[(scale, weight)] in WEX order w1, w2, w3 for every expert of one layer, or None if the layer is absent."""
    want = {"gate_proj": (inter, hidden), "up_proj": (inter, hidden), "down_proj": (hidden, inter)}
    base = f"{prefix}{layer}.mlp.experts."
    if not any(key.startswith(base) for key in tensors):
        return None
    plan = []
    for expert in range(experts):
        matrices = []
        for matrix in MATRICES:
            name = f"{base}{expert}.{matrix}"
            weight, scale = tensors.get(name + ".weight"), tensors.get(name + ".weight_scale")
            if weight is None or scale is None:
                raise ValueError(f"missing Quark tensor {name}.weight / .weight_scale (download incomplete?)")
            rows, cols = want[matrix]
            # trust boundary: U8 nibbles, U8/E8M0 scales, exactly the config's geometry
            if weight.dtype != "U8" or scale.dtype not in ("U8", "F8_E8M0", "F8_E8M0FNU"):
                raise ValueError(f"{name}: dtypes {weight.dtype}/{scale.dtype}, want U8 nibbles and U8 scales")
            if tuple(weight.shape) != (rows, cols // 2) or tuple(scale.shape) != (rows, cols // 32):
                raise ValueError(f"{name}: shapes {weight.shape}/{scale.shape}, want [{rows},{cols // 2}]/[{rows},{cols // 32}]")
            matrices.append((scale, weight))
        plan.append(matrices)
    return plan


def fp8_check(handles, fp8_tensors, fp8_prefix, layer, plan, picks, hidden, inter, min_cos):
    dims = {"gate_proj": (inter, hidden), "up_proj": (inter, hidden), "down_proj": (hidden, inter)}
    worst, rows_out = 1.0, []
    for expert in picks:
        for m, matrix in enumerate(MATRICES):
            scale, weight = plan[expert][m]
            rows, cols = dims[matrix]
            q = mxfp4_values(wex._read_tensor(weight, handles), wex._read_tensor(scale, handles), rows, cols)
            name = f"{fp8_prefix}{layer}.mlp.experts.{expert}.{matrix}"
            fw, fs = fp8_tensors.get(name + ".weight"), fp8_tensors.get(name + ".weight_scale_inv")
            if fw is None or fs is None or fw.dtype != "F8_E4M3" or tuple(fw.shape) != (rows, cols):
                raise ValueError(f"FP8 snapshot has no F8_E4M3 {name}.weight [{rows},{cols}] + weight_scale_inv")
            r = fp8_values(wex._read_tensor(fw, handles), wex._read_tensor(fs, handles), rows, cols, fs.dtype)
            cos = float((q * r).sum() / (np.linalg.norm(q) * np.linalg.norm(r) + 1e-30))
            rel = float(np.linalg.norm(q - r) / (np.linalg.norm(r) + 1e-30))
            rows_out.append({"expert": expert, "matrix": matrix, "cos": round(cos, 6), "rel": round(rel, 5)})
            worst = min(worst, cos)
            if not cos >= min_cos:
                raise ValueError(f"layer {layer} expert {expert} {matrix}: MXFP4 vs FP8 cosine {cos:.4f} < {min_cos} "
                                 "(nibble order, scale axis or tensor names wrong?)")
    return {"min_cos": round(worst, 6), "checks": rows_out}


def verified_sha(path):
    marker = Path(str(path) + ".verified")
    return marker.read_text().strip() if marker.is_file() else None


def repack(quark_dir, fp8_dir, output, layers_wanted=None, mirror_dir=None, check_experts=4, min_cos=0.95,
           skip_fp8_check=False):
    quark_dir, fp8_dir, output = Path(quark_dir).resolve(), Path(fp8_dir).resolve(), Path(output).resolve()
    layers, experts, hidden, inter = geometry(fp8_dir / "config.json")
    tensors = wex.index_safetensors(quark_dir)
    prefix = prefix_of(tensors)
    fp8_tensors = fp8_prefix = None
    if not skip_fp8_check:
        fp8_tensors = wex.index_safetensors(fp8_dir)
        fp8_prefix = prefix_of(fp8_tensors)
    selected = range(layers) if layers_wanted is None else sorted(set(layers_wanted))
    if any(not 0 <= layer < layers for layer in selected):
        raise ValueError(f"layers must be in [0, {layers})")
    plans = {}
    for layer in selected:
        plan = expert_plan(tensors, prefix, layer, experts, hidden, inter)
        if plan is None:
            raise ValueError(f"layer {layer}: no Quark expert tensors in {quark_dir} (download incomplete?)")
        plans[layer] = plan
    inputs = {ref.path for plan in plans.values() for expert in plan for pair in expert for ref in pair}
    if output in inputs or output.parent == quark_dir and output.suffix == ".safetensors":
        raise ValueError("output must not replace a model input")
    table_bytes = layers * experts * wex.ENTRY.size
    data_start = wex.WEX_ALIGN + wex.align_up(table_bytes)
    entries = [bytes(wex.ENTRY.size)] * (layers * experts)
    record_bytes = 0
    manifest = {"format": FORMAT, "version": 1, "container": output.name,
                "geometry": {"layers": layers, "experts": experts, "hidden_size": hidden, "moe_intermediate_size": inter},
                "layers": {}, "fp8_check": {}, "inputs": {}}
    offset = data_start
    stream, temporary = wex._new_temp(output)
    handles = {}
    try:
        stream.write(bytes(data_start))
        for layer in selected:
            plan = plans[layer]
            for path in {ref.path for expert in plan for pair in expert for ref in pair}:
                if path not in handles:
                    handles[path] = open(path, "rb")
            if fp8_tensors is not None:
                picks = sorted({round(i * (experts - 1) / max(check_experts - 1, 1)) for i in range(check_experts)})
                for name in (f"{fp8_prefix}{layer}.mlp.experts.{e}.{m}.{s}" for e in picks for m in MATRICES
                             for s in ("weight", "weight_scale_inv")):
                    ref = fp8_tensors.get(name)
                    if ref is not None and ref.path not in handles:
                        handles[ref.path] = open(ref.path, "rb")
                manifest["fp8_check"][str(layer)] = fp8_check(handles, fp8_tensors, fp8_prefix, layer, plan, picks,
                                                              hidden, inter, min_cos)
            digest = hashlib.sha256()
            for expert, matrices in enumerate(plan):
                scale_bytes, weight_bytes, scale_shapes, weight_shapes, scale_offsets, weight_offsets = \
                    wex._entry_metadata(matrices)
                payload = b"".join(wex._read_tensor(s, handles) for s, _ in matrices) + \
                          b"".join(wex._read_tensor(w, handles) for _, w in matrices)
                total = wex.align_up(len(payload))
                record = payload + bytes(total - len(payload))
                stream.write(record)
                digest.update(record)
                entries[layer * experts + expert] = wex.ENTRY.pack(
                    offset, total, scale_bytes, weight_bytes, *scale_shapes, *weight_shapes,
                    *scale_offsets, *weight_offsets, 0)
                record_bytes = max(record_bytes, total)
                offset += total
            manifest["layers"][str(layer)] = {"experts": experts, "sha256": digest.hexdigest()}
            print(f"layer {layer}: {experts} records"
                  + (f", FP8 min cos {manifest['fp8_check'][str(layer)]['min_cos']:.4f}" if fp8_tensors else ""),
                  flush=True)
        head = wex.HEADER.pack(wex.WEX_MAGIC_V1, 1, layers, experts, record_bytes, wex.WEX_ALIGN, 0, 0)
        head_page = head + bytes(wex.WEX_ALIGN - len(head))
        table = b"".join(entries)
        stream.seek(0)
        stream.write(head_page)
        stream.write(table)
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        stream = None
        wex.validate_container(temporary)
        # self-check: the first and last record of every layer read back == the Quark bytes
        with open(temporary, "rb") as source:
            for layer, plan in plans.items():
                for expert in {0, experts - 1}:
                    entry = wex.ENTRY.unpack(entries[layer * experts + expert])
                    source.seek(entry[0])
                    want = b"".join(wex._read_tensor(s, handles) for s, _ in plan[expert]) + \
                        b"".join(wex._read_tensor(w, handles) for _, w in plan[expert])
                    if source.read(entry[2] + entry[3]) != want:
                        raise ValueError(f"layer {layer} expert {expert}: record read back differs from the Quark bytes")
        manifest["file_bytes"] = offset
        manifest["record_bytes"] = record_bytes
        manifest["header_table_sha256"] = hashlib.sha256(head_page + table).hexdigest()
        manifest["inputs"] = {Path(p).name: {"bytes": Path(p).stat().st_size, "sha256": verified_sha(p)}
                              for p in sorted(inputs)}
        os.replace(temporary, output)
        temporary = None
        manifest_path = Path(str(output) + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        if mirror_dir is not None:
            wex._atomic_mirror(output, Path(mirror_dir).resolve() / output.name)
        return manifest
    finally:
        for handle in handles.values():
            handle.close()
        if stream is not None:
            stream.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()


def verify(container, manifest_path=None):
    """Re-hash a container's layers against its manifest; raises on any difference."""
    container = Path(container)
    manifest = json.loads(Path(manifest_path or str(container) + ".manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("not a " + FORMAT + " manifest")
    info = wex.validate_container(container)
    if info["version"] != 1 or info["file_bytes"] != manifest["file_bytes"]:
        raise ValueError("container size or version differs from the manifest")
    experts = info["experts"]
    with container.open("rb") as source:
        head = source.read(wex.WEX_ALIGN + info["layers"] * experts * wex.ENTRY.size)
        if hashlib.sha256(head).hexdigest() != manifest["header_table_sha256"]:
            raise ValueError("header page or table differs from the manifest")
        for layer, want in manifest["layers"].items():
            digest = hashlib.sha256()
            for expert in range(experts):
                entry = info["entries"][int(layer) * experts + expert]
                if entry[22] or not entry[1]:
                    raise ValueError(f"layer {layer} expert {expert}: not a raw record")
                source.seek(entry[0])
                digest.update(source.read(entry[1]))
            if digest.hexdigest() != want["sha256"]:
                raise ValueError(f"layer {layer}: records differ from the manifest")
    return {"container": str(container), "layers": sorted(int(k) for k in manifest["layers"]), "ok": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quark-dir", help="AMD Quark MXFP4 checkpoint (its *.safetensors shards)")
    parser.add_argument("--fp8-dir", help="the FP8 snapshot qwen38 runs (config.json; experts for the check)")
    parser.add_argument("--output", help="container path")
    parser.add_argument("--layers", help="comma-separated layers to write (default all); the others stay absent")
    parser.add_argument("--mirror-dir", help="also publish a byte-identical copy there (COLI_WEX_EXTRA_MIRRORS)")
    parser.add_argument("--check-experts", type=int, default=4, help="experts per layer compared with FP8")
    parser.add_argument("--min-cos", type=float, default=0.95, help="smallest MXFP4 vs FP8 cosine accepted")
    parser.add_argument("--no-fp8-check", action="store_true", help="skip the FP8 comparison (tests only)")
    parser.add_argument("--verify", metavar="CONTAINER", help="re-hash CONTAINER against its manifest and exit")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            print(json.dumps(verify(args.verify)))
            return 0
        if not (args.quark_dir and args.fp8_dir and args.output):
            parser.error("--quark-dir, --fp8-dir and --output are required")
        layers = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else None
        manifest = repack(args.quark_dir, args.fp8_dir, args.output, layers, args.mirror_dir,
                          max(args.check_experts, 1), args.min_cos, args.no_fp8_check)
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(json.dumps({k: manifest[k] for k in ("container", "file_bytes", "record_bytes", "header_table_sha256")}
                     | {"layers": sorted(int(k) for k in manifest["layers"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
