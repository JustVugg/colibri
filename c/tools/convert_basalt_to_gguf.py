#!/usr/bin/env python3
"""Convert a Basalt `.basalt` pack of a qwen35 dense model (Qwen3.8-27B-QUASAR-NVFP4.basalt) into a GGUF
the qwen36 engine reads, streaming (the pack is never loaded whole), stdlib + numpy only.

The `.basalt` container (Basalt v0.4.0, MIT, include/basalt/core/model_format.hpp and docs/model-file.md;
this reader is written from those, the field layout of tools/model_file.py):
  [0, 64 KiB)   a 64-byte header (magic BASALTMF, u32 version 2, u32 n_sections, u64 file_bytes, 40 reserved),
                then n_sections 128-byte entries (name[64] NUL-padded, u64 offset, u64 bytes, sha256[32], 16 reserved)
  sections      each at a multiple of 2 MiB; `meta` is JSON (gguf_kv: the GGUF metadata; tensors: name -> type,
                type_id, shape in GGUF order, bytes, offset inside the `gguf` section), `gguf` the tensors' bytes.

What becomes of each pack tensor:
  NVFP4_PLANES (Basalt's type 45: an E2M1 code plane, then UE4M3 scales in 128-row x 4-group tiles) -> ggml NVFP4
      (type 40, blocks of 64: four UE4M3 bytes, then 32 code bytes, value 16s+j low nibble, 16s+j+8 high). The same
      codes and scale bytes moved, so the values are the pack's bit for bit; `<name>.scale` (F32 [1]) kept: qwen36
      multiplies the product by it (gg_scale).
  F8_E4M3 (Basalt's type 44: one E4M3 code a value, `<name>.scale` F32 [1] or one per row: the BF16 output head of the
      QUASAR pack) -> Q8_0 by default (ggml's reference rounding of code x scale; a value moves by at most half a step
      plus the f16 rounding of the step, as ggml's quantize_row_q8_0_ref), or with --f8 f16 / bf16 those values
      rounded once. The scale is folded in and dropped. qwen36 has no FP8-rows type.
  Q8_0, BF16, F16, F32: as stored.
  mtp.* (the checkpoint's MTP layer): llama.cpp's nextn layer, blk.<n_layers>.* with nextn.eh_proj / enorm / hnorm /
      shared_head_norm, block_count one more (what Q36_MTP=1 reads); --no-mtp drops it.
  dflash.* (DFlash2 block drafter): not a GGUF tensor qwen36 reads (COLI_DFLASH takes a directory with config.json
      and safetensors). --dflash-dir DIR writes that directory: the Q8_0 projections as F32 of their exact values
      (qwen36's df_upload_q8 rounds them back to the same Q8_0 bytes), norms, base kernels and selector as stored.
  vision/*, tokenizer/*: not used. qwen36's vision tower is LFM3's; the tokenizer travels in the GGUF metadata.

Every section read (meta, gguf) is checked against its SHA-256 first (--no-verify skips it).

Usage:
  python3 tools/convert_basalt_to_gguf.py Qwen3.8-27B-QUASAR-NVFP4.basalt OUT.gguf [--f8 q8_0|f16|bf16]
          [--no-mtp] [--dflash-dir DIR] [--no-verify]
then `python3 tools/gguf_import.py OUT.gguf MODEL_DIR`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time

import numpy as np

MAGIC = b"BASALTMF"
VERSION = 2
SECTION_ALIGN = 2 << 20
HEADER_BYTES = 64 << 10
HEADER = struct.Struct("<8sIIQ40s")     # 64 bytes
ENTRY = struct.Struct("<64sQQ32s16s")   # 128 bytes
COPY_CHUNK = 64 << 20
CHUNK = 32 << 20                        # output bytes per converted piece

# Basalt's tensor type ids (tools/convert/checkpoint_qwen35.py TYPE_ID) -> GGUF ones; 44/45 are Basalt's own
T_F32, T_F16, T_Q8_0, T_BF16, T_NVFP4, T_F8, T_PLANES = 0, 1, 8, 30, 40, 44, 45
PASS = {T_F32: 4, T_F16: 2, T_BF16: 2}   # bytes a value
GGUF_ALIGN = 32

# the MTP layer's own tensors -> llama.cpp's nextn names (qwen36.c gg_name)
MTP_NEXTN = {"mtp.fc.weight": "nextn.eh_proj.weight", "mtp.pre_fc_norm_embedding.weight": "nextn.enorm.weight",
             "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm.weight", "mtp.norm.weight": "nextn.shared_head_norm.weight"}

# the drafter's GGUF names -> z-lab's DFlash2 checkpoint names (Basalt's DRAFTER_LAYER / DRAFTER_GLOBAL reversed)
DF_LAYER = {"attn_norm.weight": "input_layernorm.weight", "post_attention_norm.weight": "post_attention_layernorm.weight",
            "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
            "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
            "attn_conv.base": "attention_conv.base_kernel", "attn_conv.weight": "attention_conv.kernel_projection.weight",
            "ffn_conv.base": "mlp_conv.base_kernel", "ffn_conv.weight": "mlp_conv.kernel_projection.weight"}
DF_GLOBAL = {"hidden_norm.weight": "hidden_norm.weight", "norm.weight": "norm.weight",
             "selector.hidden.weight": "candidate_selector.hidden_projection.weight",
             "selector.pred": "candidate_selector.predecessor_codebook",
             "selector.succ": "candidate_selector.successor_codebook"}


class ConvertError(Exception):
    pass


def e4m3_table() -> np.ndarray:
    """E4M3 code -> float32 (bias 7, subnormals, 0x7F / 0xFF NaN)."""
    c = np.arange(256)
    e, m = (c >> 3) & 15, c & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
    v = np.where(c & 0x80, -v, v)
    v[(c & 0x7F) == 0x7F] = np.nan
    return v.astype(np.float32)


E4M3 = e4m3_table()


# ------------------------------------------------------------------ the pack
class Pack:
    def __init__(self, path: str):
        self.path = path
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(HEADER_BYTES)
        if len(head) < HEADER.size:
            raise ConvertError(f"{path}: shorter than a .basalt header")
        magic, version, n, file_bytes, _ = HEADER.unpack_from(head, 0)
        if magic != MAGIC:
            raise ConvertError(f"{path}: not a .basalt model file (magic {magic!r})")
        if version != VERSION:
            raise ConvertError(f"{path}: .basalt version {version}; this converter reads version {VERSION} "
                               "(Basalt's tools/migrations/basalt_v1_to_v2.py upgrades a version 1 file)")
        if n > (HEADER_BYTES - HEADER.size) // ENTRY.size or size < file_bytes:
            raise ConvertError(f"{path}: {n} sections, {size} of {file_bytes} bytes (truncated or corrupt)")
        self.sections: dict[str, tuple[int, int, bytes]] = {}
        for i in range(n):
            name, off, nb, digest, _ = ENTRY.unpack_from(head, HEADER.size + i * ENTRY.size)
            name = name.rstrip(b"\0").decode()
            if off % SECTION_ALIGN or off < HEADER_BYTES or off + nb > file_bytes:
                raise ConvertError(f"{path}: section {name} at {off} (+{nb}) is outside the file or unaligned")
            self.sections[name] = (off, nb, digest)
        for need in ("meta", "gguf"):
            if need not in self.sections:
                raise ConvertError(f"{path}: no `{need}` section (sections: {', '.join(self.sections)})")
        self.f = open(path, "rb")

    def verify(self, name: str) -> None:
        off, nb, digest = self.sections[name]
        h = hashlib.sha256()
        self.f.seek(off)
        left = nb
        while left > 0:
            b = self.f.read(min(COPY_CHUNK, left))
            if not b:
                raise ConvertError(f"{self.path}: section {name} is short")
            h.update(b)
            left -= len(b)
        if h.digest() != digest:
            raise ConvertError(f"{self.path}: section {name} does not match its SHA-256 (corrupt or incomplete download)")

    def read(self, name: str) -> bytes:
        off, nb, _ = self.sections[name]
        self.f.seek(off)
        return self.f.read(nb)

    def tensor_bytes(self, t: dict, start: int, n: int) -> bytes:
        off, nb, _ = self.sections["gguf"]
        if start < 0 or start + n > t["bytes"] or t["offset"] + t["bytes"] > nb:
            raise ConvertError(f"a read past a tensor's end ({start}+{n} of {t['bytes']})")
        self.f.seek(off + t["offset"] + start)
        b = self.f.read(n)
        if len(b) != n:
            raise ConvertError("short read in the gguf section")
        return b


# ------------------------------------------------------------------ conversions
def planes_to_nvfp4(codes: np.ndarray, tiles: np.ndarray, rows: int, K: int) -> np.ndarray:
    """rows (a multiple of 128) of NVFP4_PLANES -> ggml NVFP4 blocks [rows, K/64, 36]. codes: [rows, K/2] (value 2b low
    nibble of byte b); tiles: rows*K/16 scale bytes, 512-byte tiles of 128 rows x 4 groups, row-block major, byte
    (r%32)*16 + ((r%128)/32)*4 + g%4 (Basalt's include/basalt/kernels/nvfp4_planes.hpp scale_index)."""
    nb = K // 64
    sc = tiles.reshape(rows // 128, nb, 32, 4, 4).transpose(0, 3, 2, 1, 4).reshape(rows, nb, 4)
    if (sc & 0x80).any() or (sc == 0x7F).any():
        raise ConvertError("a UE4M3 scale with the sign bit set or NaN")
    v = np.empty((rows, K), np.uint8)
    v[:, 0::2], v[:, 1::2] = codes & 15, codes >> 4
    v = v.reshape(rows, nb, 4, 2, 8)
    out = np.empty((rows, nb, 36), np.uint8)
    out[:, :, :4] = sc
    out[:, :, 4:] = (v[:, :, :, 0, :] | (v[:, :, :, 1, :] << 4)).reshape(rows, nb, 32)
    return out


def q8_0(x: np.ndarray) -> np.ndarray:
    """float32 rows -> Q8_0 blocks (ggml's quantize_row_q8_0_ref: d = amax/127 stored as f16, q = round(x/d))."""
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1) / np.float32(127)
    inv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)
    out = np.empty((b.shape[0], 34), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out


def q8_0_values(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 34)
    return (np.ascontiguousarray(b[:, :2]).view("<f2").astype(np.float32) * b[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)


def bf16_round(x: np.ndarray) -> np.ndarray:
    """float32 -> BF16 bits, to nearest with ties to even (finite inputs)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def f8_values(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """E4M3 codes [rows, K] x their scale (one, or one a row) -> float32, the product as float32 rounds it."""
    if ((codes & 0x7F) == 0x7F).any():
        raise ConvertError("an E4M3 NaN code")
    s = scale.astype(np.float32).reshape(-1, 1) if scale.size > 1 else np.float32(scale.reshape(-1)[0])
    return E4M3[codes] * s


# ------------------------------------------------------------------ GGUF metadata
def kv_bytes(key: str, v) -> bytes:
    def s(x: str) -> bytes:
        b = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def scalar(x):   # (type id, bytes)
        if isinstance(x, bool):
            return 7, struct.pack("<?", x)
        if isinstance(x, int):
            if 0 <= x < 1 << 32:
                return 4, struct.pack("<I", x)
            if -(1 << 31) <= x < 0:
                return 5, struct.pack("<i", x)
            return (10, struct.pack("<Q", x)) if x >= 0 else (11, struct.pack("<q", x))
        if isinstance(x, float):
            return 6, struct.pack("<f", x)
        if isinstance(x, str):
            return 8, s(x)
        raise ConvertError(f"metadata {key}: a value of type {type(x).__name__}")

    out = s(key)
    if isinstance(v, list):
        kinds = {type(x) for x in v}
        if kinds <= {bool}:
            et, body = 7, b"".join(struct.pack("<?", x) for x in v)
        elif kinds <= {int}:   # token types, rope sections, target layers: i32 as llama.cpp writes them
            et, body = 5, struct.pack(f"<{len(v)}i", *v)
        elif kinds <= {int, float}:
            et, body = 6, struct.pack(f"<{len(v)}f", *v)
        elif kinds <= {str}:
            et, body = 8, b"".join(s(x) for x in v)
        else:
            raise ConvertError(f"metadata {key}: an array of mixed types")
        return out + struct.pack("<IIQ", 9, et, len(v)) + body
    t, b = scalar(v)
    return out + struct.pack("<I", t) + b


# ------------------------------------------------------------------ the plan
class Out:
    """One output tensor: its GGUF name, type, shape (GGUF order), size, and the pack tensor it comes from."""

    def __init__(self, name, type_id, ne, nbytes, src, how, scale=None):
        self.name, self.type_id, self.ne, self.nbytes, self.src, self.how, self.scale = name, type_id, ne, nbytes, src, how, scale


def plan(meta: dict, f8: str, mtp: bool):
    kv = dict(meta["gguf_kv"])
    arch = kv.get("general.architecture")
    if arch != "qwen35":
        raise ConvertError(f"architecture {arch!r}: this converter writes qwen35 (the dense Qwen3.5 family, Qwen3.8-27B)")
    T = meta["tensors"]
    A = arch + "."
    n_layers = int(kv[A + "block_count"])
    has_mtp = any(n.startswith("mtp.") for n in T)
    # Basalt keeps block_count at the main layers and nextn_predict_layers 1 beside mtp.*; llama.cpp (and qwen36)
    # counts the nextn layer in block_count
    for n in T:
        m = re.match(r"blk\.(\d+)\.", n)
        if m and int(m.group(1)) >= n_layers:
            raise ConvertError(f"{n}: a layer past block_count {n_layers}")
    keep_mtp = mtp and has_mtp
    kv[A + "nextn_predict_layers"] = 1 if keep_mtp else 0
    kv[A + "block_count"] = n_layers + (1 if keep_mtp else 0)
    for k in [k for k in kv if k.startswith(A + "dflash.")]:
        del kv[k]
    outs, dropped = [], []
    consumed = set()
    for name, t in T.items():
        if name.startswith("dflash."):
            dropped.append(name)
            continue
        if name.startswith("mtp.") and not keep_mtp:
            dropped.append(name)
            continue
        if name in consumed:
            continue
        gname = name
        if name.startswith("mtp."):
            gname = f"blk.{n_layers}." + MTP_NEXTN.get(name, name[4:])
        tid, ne = int(t["type_id"]), [int(d) for d in t["shape"]]
        n = int(np.prod(ne))
        if tid == T_PLANES:
            if len(ne) != 2 or ne[1] % 128 or ne[0] % 64 or int(t["bytes"]) != n // 64 * 36:
                raise ConvertError(f"{name}: NVFP4_PLANES of shape {ne}, {t['bytes']} bytes")
            outs.append(Out(gname, T_NVFP4, ne, n // 64 * 36, name, "planes"))
        elif tid == T_F8:
            sn = name[:-len(".weight")] + ".scale"
            if sn not in T or int(T[sn]["type_id"]) != T_F32 or int(np.prod(T[sn]["shape"])) not in (1, ne[1]):
                raise ConvertError(f"{name}: F8_E4M3 without an F32 scale of 1 or {ne[1]} values")
            if len(ne) != 2 or int(t["bytes"]) != n:
                raise ConvertError(f"{name}: F8_E4M3 of shape {ne}, {t['bytes']} bytes")
            consumed.add(sn)
            if f8 == "q8_0":
                if ne[0] % 32:
                    raise ConvertError(f"{name}: rows of {ne[0]} values cannot be Q8_0 (--f8 f16 or bf16)")
                outs.append(Out(gname, T_Q8_0, ne, n // 32 * 34, name, "f8", sn))
            else:
                outs.append(Out(gname, T_F16 if f8 == "f16" else T_BF16, ne, 2 * n, name, "f8", sn))
        elif tid == T_Q8_0:
            if ne[0] % 32 or int(t["bytes"]) != n // 32 * 34:
                raise ConvertError(f"{name}: Q8_0 of shape {ne}, {t['bytes']} bytes")
            outs.append(Out(gname, tid, ne, n // 32 * 34, name, "copy"))
        elif tid in PASS:
            if int(t["bytes"]) != n * PASS[tid]:
                raise ConvertError(f"{name}: type {t['type']} of shape {ne}, {t['bytes']} bytes")
            outs.append(Out(gname, tid, ne, n * PASS[tid], name, "copy"))
        else:
            raise ConvertError(f"{name}: tensor type {t.get('type')} ({tid}) is not one this converter knows")
    names = [o.name for o in outs]
    if len(set(names)) != len(names):
        raise ConvertError("two pack tensors map to one GGUF name")
    for o in outs:   # a scale beside a tensor that kept its blocks stays a companion only of NVFP4
        if o.name.endswith(".scale") and o.how == "copy":
            w = o.name[:-len(".scale")] + ".weight"
            src = next((x for x in outs if x.name == w), None)
            if src is None or src.type_id != T_NVFP4:
                raise ConvertError(f"{o.name}: a scale whose tensor is not NVFP4 ({w})")
    kv["general.file_type"] = 40   # mostly NVFP4
    kv["colibri.convert.source"] = "basalt " + str(meta.get("name", ""))
    kv["colibri.convert.f8_e4m3"] = {"q8_0": "Q8_0 (ggml reference rounding of code x scale)",
                                   "f16": "F16 of code x scale, rounded once", "bf16": "BF16 of code x scale, rounded once"}[f8]
    return kv, outs, dropped


def chunks(pack: Pack, meta: dict, o: Out):
    """o's output bytes, piece by piece."""
    T = meta["tensors"]
    t = T[o.src]
    if o.how == "copy":
        for s in range(0, o.nbytes, CHUNK):
            yield pack.tensor_bytes(t, s, min(CHUNK, o.nbytes - s))
        return
    K, rows = o.ne[0], o.ne[1]
    if o.how == "planes":
        step = max(128, CHUNK // (K // 64 * 36) // 128 * 128)
        for r in range(0, rows, step):
            n = min(step, rows - r)
            codes = np.frombuffer(pack.tensor_bytes(t, r * K // 2, n * K // 2), np.uint8).reshape(n, K // 2)
            tiles = np.frombuffer(pack.tensor_bytes(t, rows * K // 2 + r * K // 16, n * K // 16), np.uint8)
            yield planes_to_nvfp4(codes, tiles, n, K).tobytes()
        return
    # f8: E4M3 rows x their scale
    st = T[o.scale]
    scale = np.frombuffer(pack.tensor_bytes(st, 0, int(st["bytes"])), "<f4")
    if not (np.isfinite(scale).all() and (scale > 0).all()):
        raise ConvertError(f"{o.scale}: a scale that is not finite and positive")
    step = max(1, CHUNK // (4 * K))
    for r in range(0, rows, step):
        n = min(step, rows - r)
        codes = np.frombuffer(pack.tensor_bytes(t, r * K, n * K), np.uint8).reshape(n, K)
        x = f8_values(codes, scale[r:r + n] if scale.size > 1 else scale)
        if o.type_id == T_Q8_0:
            yield q8_0(x).tobytes()
        elif o.type_id == T_F16:
            if (np.abs(x) > 65504).any():
                raise ConvertError(f"{o.name}: a value past F16's range (--f8 bf16 or q8_0)")
            yield x.astype("<f2").tobytes()
        else:
            yield bf16_round(x).tobytes()


def write_gguf(path: str, pack: Pack, meta: dict, kv: dict, outs: list) -> None:
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(outs), len(kv)))
    for k, v in kv.items():
        head += kv_bytes(k, v)
    off = 0
    for o in outs:
        nm = o.name.encode()
        head += struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(o.ne)) + struct.pack(f"<{len(o.ne)}Q", *o.ne)
        head += struct.pack("<IQ", o.type_id, off)
        off += (o.nbytes + GGUF_ALIGN - 1) // GGUF_ALIGN * GGUF_ALIGN
    head += b"\0" * ((-len(head)) % GGUF_ALIGN)
    partial = path + ".partial"
    t0, done = time.time(), 0
    with open(partial, "wb") as f:
        f.write(head)
        for i, o in enumerate(outs):
            n = 0
            for b in chunks(pack, meta, o):
                f.write(b)
                n += len(b)
            if n != o.nbytes:
                raise ConvertError(f"{o.name}: wrote {n} bytes, expected {o.nbytes}")
            f.write(b"\0" * ((-n) % GGUF_ALIGN))
            done += n
            if o.nbytes >= 256 << 20 or i == len(outs) - 1:
                print(f"  {i + 1}/{len(outs)} tensors, {done / (1 << 30):.2f} GiB, {time.time() - t0:.0f} s", flush=True)
    os.replace(partial, path)


# ------------------------------------------------------------------ the drafter
def write_dflash(d: str, pack: Pack, meta: dict) -> None:
    """The DFlash2 drafter as the directory qwen36's COLI_DFLASH reads (dflash_cuda.h df_load): config.json and
    model.safetensors in z-lab's names."""
    kv, T = meta["gguf_kv"], meta["tensors"]
    A = "qwen35.dflash."
    if A + "block_count" not in kv:
        raise ConvertError("--dflash-dir: the pack holds no DFlash2 drafter")
    H = int(kv["qwen35.embedding_length"])
    V = int(T["token_embd.weight"]["shape"][1])
    nt = len(kv[A + "target_layers"])
    cfg = {"architectures": ["DFlash2DraftModel"], "hidden_size": H, "vocab_size": V,
           "num_hidden_layers": int(kv[A + "block_count"]), "num_attention_heads": int(kv[A + "attention.head_count"]),
           "num_key_value_heads": int(kv[A + "attention.head_count_kv"]), "head_dim": int(kv[A + "attention.key_length"]),
           "intermediate_size": int(kv[A + "feed_forward_length"]), "rms_norm_eps": float(kv[A + "attention.layer_norm_rms_epsilon"]),
           "rope_parameters": {"rope_theta": float(kv[A + "rope.freq_base"])}, "sliding_window": int(kv[A + "sliding_window"]),
           "is_causal": False, "layer_types": ["sliding_attention"] * int(kv[A + "block_count"]),
           "dflash_config": {"block_size": int(kv[A + "block_size"]), "mask_token_id": int(kv[A + "mask_token_id"]),
                             "target_layer_ids": [int(x) for x in kv[A + "target_layers"]],
                             "conv_kernel_size": int(kv[A + "conv_kernel"]), "conv_group_size": int(kv[A + "conv_group"]),
                             "selector_rank": int(kv[A + "selector_rank"]), "selector_top_k": int(kv[A + "selector_top_k"])},
           "converted_from": "the DFlash2 drafter of a Basalt pack: Q8_0 projections as F32 of their exact values"}
    # (HF name, dtype, HF shape, [pack tensors]) in a stable order; fc is the target layers' blocks side by side
    items = []
    fc = [f"dflash.fc.{j}.weight" for j in range(nt)]
    if any(n not in T for n in fc):
        raise ConvertError("--dflash-dir: the drafter's fc blocks are missing")
    items.append(("fc.weight", [H, H * nt], fc))
    for name in T:
        if not name.startswith("dflash.") or name in fc:
            continue
        m = re.fullmatch(r"dflash\.blk\.(\d+)\.(.+)", name)
        hf = f"layers.{m.group(1)}.{DF_LAYER[m.group(2)]}" if m and m.group(2) in DF_LAYER else DF_GLOBAL.get(name[len("dflash."):])
        if hf is None:
            raise ConvertError(f"{name}: a drafter tensor this converter does not know")
        items.append((hf, list(reversed([int(x) for x in T[name]["shape"]])), [name]))
    header, parts, off = {}, [], 0
    for hf, shape, srcs in items:
        tids = {int(T[s]["type_id"]) for s in srcs}
        if len(tids) != 1 or (tid := tids.pop()) not in (T_Q8_0, T_F32, T_BF16):
            raise ConvertError(f"{hf}: drafter tensors of types {tids}")
        dt, esz = {T_Q8_0: ("F32", 4), T_F32: ("F32", 4), T_BF16: ("BF16", 2)}[tid]
        n = int(np.prod(shape)) * esz
        header[hf] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        parts.append((hf, tid, shape, srcs, n))
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    tmp = os.path.join(d, "model.safetensors.partial")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for hf, tid, shape, srcs, n in parts:
            if tid != T_Q8_0:
                b = pack.tensor_bytes(T[srcs[0]], 0, int(T[srcs[0]]["bytes"]))
                f.write(b)
                continue
            rows = shape[0]
            blocks = []
            for s in srcs:   # each block [rows][cols_j]; written side by side, rows in bounded pieces
                cols = int(T[s]["shape"][0])
                blocks.append((s, cols))
            step = max(1, CHUNK // (4 * sum(c for _, c in blocks)))
            for r in range(0, rows, step):
                k = min(step, rows - r)
                row = [q8_0_values(np.frombuffer(pack.tensor_bytes(T[s], r * c // 32 * 34, k * c // 32 * 34), np.uint8)).reshape(k, c)
                       for s, c in blocks]
                f.write(np.ascontiguousarray(np.concatenate(row, axis=1), "<f4").tobytes())
    os.replace(tmp, os.path.join(d, "model.safetensors"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basalt")
    ap.add_argument("out", help="the GGUF to write")
    ap.add_argument("--f8", choices=("q8_0", "f16", "bf16"), default="q8_0",
                    help="F8_E4M3 tensors (the QUASAR pack's output head): Q8_0 (default, 1.06 bytes a value), or F16 / "
                         "BF16 of their values (2 bytes)")
    ap.add_argument("--no-mtp", action="store_true", help="drop the MTP layer (mtp.*) instead of writing it as llama.cpp's nextn layer")
    ap.add_argument("--dflash-dir", help="write the DFlash2 drafter (dflash.*) as the directory COLI_DFLASH reads")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check of the sections read")
    ap.add_argument("--force", action="store_true", help="overwrite OUT")
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and not a.force:
        print(f"{a.out} exists (--force overwrites it)", file=sys.stderr)
        return 1
    try:
        pack = Pack(a.basalt)
        if not a.no_verify:
            t0 = time.time()
            for s in ("meta", "gguf"):
                pack.verify(s)
            print(f"SHA-256 of meta and gguf: ok ({time.time() - t0:.0f} s)", flush=True)
        meta = json.loads(pack.read("meta"))
        if meta.get("format") != "basalt" or int(meta.get("version", 0)) != VERSION:
            raise ConvertError(f"meta: format {meta.get('format')} version {meta.get('version')}")
        kv, outs, dropped = plan(meta, a.f8, not a.no_mtp)
        by = {}
        for o in outs:
            by[o.type_id] = by.get(o.type_id, 0) + o.nbytes
        tn = {T_F32: "F32", T_F16: "F16", T_Q8_0: "Q8_0", T_BF16: "BF16", T_NVFP4: "NVFP4"}
        sizes = ", ".join("%s %.2f GiB" % (tn[k], v / (1 << 30)) for k, v in sorted(by.items()))
        kinds = ", ".join(sorted({d.split(".")[0] for d in dropped})) or "none"
        print(f"{len(outs)} tensors ({sizes}), {len(dropped)} not written ({kinds})", flush=True)
        write_gguf(a.out, pack, meta, kv, outs)
        if a.dflash_dir:
            write_dflash(a.dflash_dir, pack, meta)
            print(f"DFlash2 drafter -> {a.dflash_dir} (COLI_DFLASH={a.dflash_dir})")
        elif any(d.startswith("dflash.") for d in dropped):
            print("the pack's DFlash2 drafter was not written (--dflash-dir DIR writes it for COLI_DFLASH)")
    except ConvertError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"done: {a.out} ({os.path.getsize(a.out) / (1 << 30):.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
