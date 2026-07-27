#!/usr/bin/env python3
"""Repack a Kimi K3 HF snapshot into colibri's ideal ingestion layout.

Designed so the HDD->NVMe copy IS the conversion (one read of the source):

  - routed experts (MXFP4, QAT): byte-identical passthrough, written in
    expert-id order, the six tensors of an expert back-to-back (the engine
    reads one expert = one pread)
  - big BF16 matrices: quantized ON THE WAY with EXACTLY the engine's
    load-time algorithm (int8 per-row absmax/127 or int4-g64 absmax/7,
    round-half-even, same f32 op order), stored as U8 + <name>.qs f32 scale
    sidecar -> the engine skips its 10-15 min load-time quantization and
    cold-starts from the container in ~1-2 min
  - everything small or sensitive (norms, router, conv taps, dt_bias, A_log,
    o_norm, f_a/f_b, b_proj, res_*, embed_tokens, final norm): passthrough
    in the source dtype (BF16/F32) — quantizing these saves ~nothing and the
    engine keeps them f32 in RAM anyway
  - vision_tower / mm_projector (shards 95-96): dropped
  - output: model-XXXXX-of-000094.safetensors (same shard split: one layer
    per shard + shard 94 = embed/head), spec-valid safetensors (no holes),
    regenerated model.safetensors.index.json, config/tokenizer files copied

Resume-safe: each shard is written to .tmp and renamed; existing outputs are
skipped. Rerun after an interruption and it continues.

usage: k3_repack.py <src_dir> <dst_dir> [--bits 8] [--mla-bits N] [--head-bits N]
                    [--shards 1,2,94] [--verify-full]

Classification needs the layer type (o_proj/g_proj exist on BOTH KDA and MLA
layers) — read from config.json linear_attn_config.
"""
import argparse
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

CHUNK_ROWS = 4096          # quantize in row blocks (bounds RAM to ~120 MB)
COPY_BLOCK = 64 << 20      # passthrough copy block


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def bf16_to_f32(buf):
    u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def quant_int8(rows):
    """rows [n,I] f32 -> (int8 bytes, f32 scales [n]); engine w_load exact."""
    am = np.abs(rows).max(axis=1)
    s = (am / np.float32(127.0)).astype(np.float32)
    s = np.maximum(s, np.float32(1e-20))
    q = np.clip(np.rint(rows * (np.float32(1.0) / s)[:, None]), -127, 127).astype(np.int8)
    return q.tobytes(), s


def quant_int4_g64(rows):
    """rows [n,I] f32, I%64==0 -> (packed bytes low-nibble-even biased+8,
    f32 scales [n, I/64]); engine w_load / matmul_i4_grouped exact."""
    n, I = rows.shape
    g = rows.reshape(n, I // 64, 64)
    am = np.abs(g).max(axis=2)
    s = (am / np.float32(7.0)).astype(np.float32)
    s = np.maximum(s, np.float32(1e-20))
    q = np.clip(np.rint(g * (np.float32(1.0) / s)[:, :, None]), -8, 7).astype(np.int32) + 8
    q = q.reshape(n, I)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed.tobytes(), s


class Classifier:
    def __init__(self, src):
        cfg = json.load(open(os.path.join(src, "config.json")))
        tc = cfg.get("text_config", cfg)
        self.kda = set(x - 1 for x in tc["linear_attn_config"]["kda_layers"])

    def __call__(self, name):
        """-> ('skip'|'expert'|'pass'|'q8class', class) where class in
        {'bits','mla','head'} for quantized tensors."""
        n = name
        if n.startswith("vision_tower") or n.startswith("mm_projector"):
            return "skip", None
        if n.startswith("language_model."):
            n = n[len("language_model."):]
        if ".block_sparse_moe.experts." in n:
            return "expert", None
        if n == "lm_head.weight":
            return "quant", "head"
        parts = n.split(".")
        li = int(parts[2]) if len(parts) > 3 and parts[1] == "layers" else -1
        if ".self_attn." in n and n.endswith(".weight"):
            leaf = parts[-2]
            if leaf in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
                return "quant", "mla"
            if leaf in ("q_proj", "k_proj", "v_proj"):
                return "quant", "bits"
            if leaf in ("o_proj", "g_proj"):
                return ("quant", "bits") if li in self.kda else ("quant", "mla")
            return "pass", None            # conv taps, norms, f_a/f_b, b_proj
        for leaf in ("routed_expert_down_proj", "routed_expert_up_proj",
                     "shared_experts.gate_proj", "shared_experts.up_proj",
                     "shared_experts.down_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            if n.endswith(leaf + ".weight"):
                return "quant", "bits"
        return "pass", None                # router, biases, norms, embed, res_*


def expert_sort_key(name):
    # ...experts.<E>.<w1|w2|w3>.weight_<packed|scale>
    p = name.split(".experts.")[1].split(".")
    e = int(p[0])
    mat = {"w1": 0, "w2": 1, "w3": 2}[p[1]]
    kind = 0 if p[2].endswith("packed") else 1
    # engine slot order: w1p w1s w2p w2s w3p w3s
    return (e, mat, kind)


def repack_shard(src_path, dst_path, cls, bits_map, verify_full):
    hdr, data_start = read_header(src_path)
    items = sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
    non_expert, experts = [], []
    for name, v in items:
        kind, qc = cls(name)
        if kind == "skip":
            continue
        (experts if kind == "expert" else non_expert).append((name, v, kind, qc))
    experts.sort(key=lambda t: expert_sort_key(t[0]))
    plan = non_expert + experts

    # pass 1: output sizes -> header with final offsets
    out_entries = []       # (name, dtype, shape, nbytes)  (+ .qs after its tensor)
    for name, v, kind, qc in plan:
        if kind == "quant":
            O, I = v["shape"]
            bits = bits_map[qc]
            if bits <= 4 and I % 64 == 0:
                out_entries.append((name, "U8", [O, I // 2], O * (I // 2)))
                out_entries.append((name + ".qs", "F32", [O, I // 64], O * (I // 64) * 4))
            else:
                out_entries.append((name, "U8", [O, I], O * I))
                out_entries.append((name + ".qs", "F32", [O], O * 4))
        else:
            out_entries.append((name, v["dtype"], v["shape"],
                                v["data_offsets"][1] - v["data_offsets"][0]))
    oh, off = {}, 0
    for name, dt, shape, nb in out_entries:
        oh[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    hjson = json.dumps(oh, separators=(",", ":")).encode()
    pad = (-(8 + len(hjson))) % 8
    hjson += b" " * pad

    tmp = dst_path + ".tmp"
    written = 0
    with open(src_path, "rb") as fin, open(tmp, "wb") as fout:
        fout.write(struct.pack("<Q", len(hjson)) + hjson)
        for name, v, kind, qc in plan:
            a0, b0 = v["data_offsets"]
            fin.seek(data_start + a0)
            if kind == "quant":
                O, I = v["shape"]
                bits = bits_map[qc]
                use4 = bits <= 4 and I % 64 == 0
                qparts, sparts = [], []
                for r0 in range(0, O, CHUNK_ROWS):
                    n = min(CHUNK_ROWS, O - r0)
                    raw = fin.read(n * I * 2)
                    rows = bf16_to_f32(raw).reshape(n, I)
                    qb, s = quant_int4_g64(rows) if use4 else quant_int8(rows)
                    qparts.append(qb)
                    sparts.append(s.astype(np.float32).tobytes())
                for qb in qparts:
                    fout.write(qb)
                for sb in sparts:
                    fout.write(sb)
                written += sum(len(x) for x in qparts) + sum(len(x) for x in sparts)
            else:
                left = b0 - a0
                while left:
                    blk = fin.read(min(COPY_BLOCK, left))
                    if not blk:
                        raise IOError(f"short read in {name}")
                    fout.write(blk)
                    left -= len(blk)
                written += b0 - a0
    # sanity: bytes written == declared
    want = off
    got = os.path.getsize(tmp) - 8 - len(hjson)
    if got != want:
        raise IOError(f"{dst_path}: wrote {got} data bytes, header declares {want}")
    os.replace(tmp, dst_path)

    if verify_full and experts:
        vh, vstart = read_header(dst_path)
        with open(src_path, "rb") as fs, open(dst_path, "rb") as fd:
            for name, v, kind, qc in experts:
                sa, sb = v["data_offsets"]
                da, db = vh[name]["data_offsets"]
                assert sb - sa == db - da, name
                fs.seek(data_start + sa)
                fd.seek(vstart + da)
                left = sb - sa
                while left:
                    n = min(COPY_BLOCK, left)
                    if fs.read(n) != fd.read(n):
                        raise IOError(f"VERIFY FAILED: {name}")
                    left -= n
    return oh, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--mla-bits", type=int, default=None)
    ap.add_argument("--head-bits", type=int, default=None)
    ap.add_argument("--shards", help="comma list of shard numbers (default 1..94)")
    ap.add_argument("--verify-full", action="store_true",
                    help="re-read every expert byte and compare with the source")
    a = ap.parse_args()
    bits_map = {"bits": a.bits,
                "mla": a.mla_bits if a.mla_bits is not None else max(a.bits, 8),
                "head": a.head_bits if a.head_bits is not None else max(a.bits, 8)}
    os.makedirs(a.dst, exist_ok=True)
    cls = Classifier(a.src)
    shard_nums = ([int(x) for x in a.shards.split(",")] if a.shards
                  else list(range(1, 95)))
    index = {}
    t0 = time.time()
    total = 0
    done_any = False
    for i, num in enumerate(shard_nums):
        sp = os.path.join(a.src, f"model-{num:05d}-of-000096.safetensors")
        dname = f"model-{num:05d}-of-000094.safetensors"
        dp = os.path.join(a.dst, dname)
        if not os.path.exists(sp):
            print(f"[{num:2d}] SOURCE MISSING — skipped (rerun when downloaded)", flush=True)
            continue
        if os.path.exists(dp):
            oh, _ = read_header(dp)
            for name in oh:
                index[name] = dname
            print(f"[{num:2d}] exists — skipped (resume)", flush=True)
            continue
        oh, written = repack_shard(sp, dp, cls, bits_map, a.verify_full)
        for name in oh:
            index[name] = dname
        total += written
        done_any = True
        el = time.time() - t0
        print(f"[{num:2d}] {os.path.getsize(dp)/1e9:6.2f} GB out "
              f"({total/1e9:7.1f} GB total, {total/1e9/max(el,1e-9):5.2f} GB/s, {el:6.0f}s)",
              flush=True)
    if done_any or index:
        with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": index}, f)
    for extra in ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json"):
        s = os.path.join(a.src, extra)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(a.dst, extra))
    print(f"done: {total/1e9:.1f} GB written in {time.time()-t0:.0f}s -> {a.dst}")


if __name__ == "__main__":
    main()
