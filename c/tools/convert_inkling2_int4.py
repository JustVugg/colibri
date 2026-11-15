#!/usr/bin/env python3
"""Inkling-2: the official checkpoint's routed experts as int4 g64, for inkling.c.

Sources, read under their official names (model.llm.*):
  thinkingmachines/Inkling-2-Small        NVFP4 experts in layers 0-39, MXFP8 in 40-47
  thinkingmachines/Inkling-2-Small-MXFP8  MXFP8 experts
  a bf16/f16/f32 snapshot                 transformers' save_pretrained (tools/make_tiny_inkling2.py)
Each expert tensor's format is read from its dtype and from the tensors beside it:
  NVFP4  <w> U8 [E, R, C/2], e2m1 codes, the low nibble the even column; <w>_scale F8_E4M3
         [E, R, C/16]; <w>_amax F32 [E]; <w>_outer_scale_denominator F32 [1].
         value = e2m1 * e4m3 * (amax[e] / denominator)
  MXFP8  <w> F8_E4M3 [E, R, C]; <w>_scale U8 [E, R, C/32] (ue8m0). value = e4m3 * 2^(scale - 127)
  plain  <w> BF16/F16/F32 [E, R, C]
w13 holds the gate and up rows interleaved (even rows gate); the output holds the gate rows, then
the up rows.

int4 g64 is expert_ffn.h's "planar unsigned int4, gs=64": per row, blocks of 64 inputs, 32 bytes a
block, byte k holding element k in its low nibble and element k + 32 in its high one, as v + 8
(v in -8..7); one f32 scale per block. A block's scale is the one among absmax/7 * (1, .95, .9, .85,
.8, .75) and absmax/8 * (1, .95, .9) (absmax/8 puts the most negative weight on -8) whose codes land
nearest the block's weights (least squared error): on Inkling-2-Small's expert 0 that is 9.6%
relative error from the NVFP4 and MXFP8 weights against 11.3% for absmax/7 alone; twice as many
candidates gain 0.001 points at twice the time.

Output, one directory that inkling.c reads under the official names:
  layer-NN.safetensors   every tensor of model.llm.layers.NN; the routed experts as
                           mlp.experts.gate_up_proj     U8  [E, 2I, D/2]   gate rows, then up rows
                           mlp.experts.gate_up_proj.qs  F32 [E, 2I, D/64]
                           mlp.experts.down_proj        U8  [E, D, I/2]
                           mlp.experts.down_proj.qs     F32 [E, D, I/64]
                         the rest as they are (bf16 or f32)
  misc-SSSSS.safetensors the tensors outside the layers held by source shard SSSSS (embedding,
                         unembedding, norms, the attention residuals' queries, the vision tower)
  mtp/mtp.safetensors    the drafter, copied (moved with --delete-src; st_init reads no subdirectory)
  config.json            without quantization_config; the tokenizer and template files
An output is written when every shard holding one of its tensors is complete (the misc ones
first, then the layers in order), under a temporary name renamed at the end: an interrupted run
resumes where it stopped. --watch waits for the shards a download is still writing; --delete-src
removes a shard as soon as every output that reads it is written, so the disk holds the output
plus the shards in flight (~16 GB each). An output that would leave less than --min-free-gb (20)
free waits for room (--watch) or stops the run.

Usage:
  convert_inkling2_int4.py --indir SNAPSHOT --outdir OUT [--watch] [--delete-src] [--jobs N]
                           [--min-free-gb G]
  convert_inkling2_int4.py --selftest
  convert_inkling2_int4.py --fabricate SNAPSHOT OUT [--nvfp4-layers N]
      a quantized copy of a plain snapshot as the official checkpoint lays it out: the experts of
      layers below N in NVFP4, the others in MXFP8, the tensors spread over two shards so that a
      layer straddles them (tests of this converter against tools/make_tiny_inkling2.py)
"""
import argparse
import glob
import json
import os
import re
import shutil
import struct
import sys
import time

if __name__ == "__main__":   # run, not imported (a test's process keeps its environment)
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")   # one thread a process: the workers are the parallelism
import numpy as np  # noqa: E402

GS = 64
AUX_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "chat_template.jinja", "processor_config.json",
             "preprocessor_config.json")
DT_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "U8": 1, "I8": 1, "F8_E4M3": 1, "F64": 8, "I32": 4, "I64": 8}
EXPERT_RE = re.compile(r"^(model\.llm\.layers\.(\d+)\.mlp\.experts\.)(w13|w2)_weight$")
AUX_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")


def e4m3_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        e, m = (b >> 3) & 15, b & 7
        v = 2.0 ** (e - 7) * (1 + m / 8) if e else 2.0 ** -6 * (m / 8)
        t[b] = -v if b & 0x80 else v
    t[0x7F] = t[0xFF] = np.nan
    return t


E4M3 = e4m3_table()
FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


# ---------- safetensors, read and written by hand (F8_E4M3 included) ----------

def read_header(path):
    """-> (tensors {name: (dtype, shape, absolute offset, nbytes)}, metadata, expected file size)"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", {})
    base = 8 + n
    out, end = {}, 0
    for k, v in h.items():
        a, b = v["data_offsets"]
        out[k] = (v["dtype"], tuple(v["shape"]), base + a, b - a)
        end = max(end, b)
    return out, meta, base + end


def read_bytes(path, off, n):
    with open(path, "rb") as f:
        f.seek(off)
        b = f.read(n)
    if len(b) != n:
        raise IOError(f"{path}: short read at {off} ({len(b)} of {n} bytes)")
    return b


def plan_file(tensors):
    """tensors: [(name, dtype, shape, nbytes)] -> (header bytes, {name: absolute offset}, size).
    F32 first, then 2-byte types, then bytes, so that every tensor sits on its element size."""
    order = sorted(tensors, key=lambda t: (-DT_SIZE[t[1]], t[0]))
    hdr, off = {"__metadata__": {"format": "pt", "colibri": "inkling2-int4g64"}}, 0
    for name, dt, shape, nb in order:
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
        off += nb
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    base = 8 + len(hb)
    offs, o = {}, base
    for name, dt, shape, nb in order:
        offs[name] = o
        o += nb
    return struct.pack("<Q", len(hb)) + hb, offs, o


# ---------- decoding the sources ----------

def to_f32(raw, dtype, shape):
    if dtype == "F32":
        return np.frombuffer(raw, "<f4").reshape(shape).copy()
    if dtype == "BF16":
        return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, "<f2").astype(np.float32).reshape(shape)
    raise ValueError(f"no float decoding for {dtype}")


def expert_spec(idx, name):
    """idx {name: (path, dtype, shape, off, nbytes)} -> how to decode the expert tensor `name`."""
    path, dt, shape, off, nb = idx[name]
    if len(shape) != 3:
        raise SystemExit(f"{name}: shape {shape}, not [E, R, C]")
    E, R, C = shape
    sc = idx.get(name + "_scale")
    if dt == "U8" and sc and sc[1] == "F8_E4M3" and name + "_amax" in idx:
        C *= 2
        am, de = idx[name + "_amax"], idx.get(name + "_outer_scale_denominator")
        if not de or tuple(sc[2]) != (E, R, C // 16) or am[2] != (E,):
            raise SystemExit(f"{name}: NVFP4 without the scales the format needs")
        return {"fmt": "nvfp4", "shape": (E, R, C), "w": (path, off), "s": (sc[0], sc[3]),
                "amax": (am[0], am[3]), "den": (de[0], de[3])}
    if dt == "F8_E4M3":
        if not sc or sc[1] != "U8" or tuple(sc[2]) != (E, R, C // 32):
            raise SystemExit(f"{name}: MXFP8 without its ue8m0 scales [E, R, C/32]")
        return {"fmt": "mxfp8", "shape": (E, R, C), "w": (path, off), "s": (sc[0], sc[3])}
    if dt in ("BF16", "F16", "F32"):
        return {"fmt": "plain", "shape": (E, R, C), "w": (path, off), "dt": dt}
    raise SystemExit(f"{name}: dtype {dt} is no expert format this converter knows")


def decode_expert(spec, e):
    """One expert of a [E, R, C] tensor as f32 [R, C]."""
    E, R, C = spec["shape"]
    path, off = spec["w"]
    if spec["fmt"] == "nvfp4":
        b = np.frombuffer(read_bytes(path, off + e * R * C // 2, R * C // 2), np.uint8).reshape(R, C // 2)
        s = np.frombuffer(read_bytes(spec["s"][0], spec["s"][1] + e * R * C // 16, R * C // 16), np.uint8)
        amax = np.frombuffer(read_bytes(spec["amax"][0], spec["amax"][1] + 4 * e, 4), "<f4")[0]
        den = np.frombuffer(read_bytes(spec["den"][0], spec["den"][1], 4), "<f4")[0]
        v = np.empty((R, C), np.float32)
        v[:, 0::2] = FP4[b & 15]
        v[:, 1::2] = FP4[b >> 4]
        v = v.reshape(R, C // 16, 16) * E4M3[s].reshape(R, C // 16, 1)   # e2m1 * e4m3: exact
        return (v * np.float32(np.float64(amax) / np.float64(den))).reshape(R, C)
    if spec["fmt"] == "mxfp8":
        b = np.frombuffer(read_bytes(path, off + e * R * C, R * C), np.uint8).reshape(R, C // 32, 32)
        s = np.frombuffer(read_bytes(spec["s"][0], spec["s"][1] + e * R * C // 32, R * C // 32), np.uint8)
        return (E4M3[b] * np.exp2(s.astype(np.float32) - 127).reshape(R, C // 32, 1)).reshape(R, C)
    sz = DT_SIZE[spec["dt"]]
    return to_f32(read_bytes(path, off + e * R * C * sz, R * C * sz), spec["dt"], (R, C))


# ---------- int4 g64 ----------

CHUNK = 1 << 16   # elements a block of rows quantizes at once: its temporaries stay in the cache
CANDIDATES = (tuple((7, np.float32(a)) for a in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)) +
              tuple((8, np.float32(a)) for a in (1.0, 0.95, 0.9)))


def quantize_g64(w, stats=None):
    """w f32 [R, C], C % 64 == 0 -> (codes u8 [R, C] holding v + 8, scales f32 [R, C/64]).
    A block of rows at a time, so that the candidates' passes stay in the cache (on the whole
    matrix they cost ~18 GB of memory traffic an expert). stats, when given, gets the squared
    error and the squared weights added to its [0] and [1]."""
    R, C = w.shape
    G = C // GS
    codes = np.empty((R, C), np.uint8)
    scales = np.empty((R, G), np.float32)
    step = max(1, CHUNK // C)
    t = np.empty((step, G, GS), np.float32)
    u = np.empty((step, G, GS), np.float32)
    bq = np.empty((step, G, GS), np.float32)
    for r0 in range(0, R, step):
        n = min(step, R - r0)
        g = w[r0:r0 + n].reshape(n, G, GS)
        tt, uu, qq = t[:n], u[:n], bq[:n]
        amax = np.maximum(np.abs(g).max(axis=2, keepdims=True), np.float32(1e-12))
        best_e = np.full(amax.shape, np.inf, np.float32)
        best_s = np.empty(amax.shape, np.float32)
        for top, a in CANDIDATES:
            sc = amax * (a / np.float32(top))
            np.divide(g, sc, out=tt)
            np.rint(tt, out=tt)
            np.clip(tt, -8, 7, out=tt)
            np.multiply(tt, sc, out=uu)
            uu -= g
            np.square(uu, out=uu)
            err = uu.sum(axis=2, keepdims=True)
            m = err < best_e
            np.copyto(best_e, err, where=m)
            np.copyto(best_s, sc, where=m)
            np.copyto(qq, tt, where=m)
        qq += 8
        codes[r0:r0 + n] = qq.reshape(n, C)
        scales[r0:r0 + n] = best_s.reshape(n, G)
        if stats is not None:
            stats[0] += float(best_e.sum(dtype=np.float64))
            stats[1] += float(np.square(g).sum(dtype=np.float64))
    return codes, scales


def pack_planar(codes):
    """u8 codes [R, C] (0..15), C % 64 == 0 -> u8 [R, C/2], expert_ffn.h's planar blocks."""
    R, C = codes.shape
    b = codes.reshape(R, C // GS, 2, GS // 2)
    return np.ascontiguousarray((b[:, :, 0, :] | (b[:, :, 1, :] << 4)).reshape(R, C // 2))


def unpack_planar(packed, scales):
    """Inverse of pack_planar times the scales: the f32 weights inkling.c computes with."""
    R, half = packed.shape
    b = packed.reshape(R, half // (GS // 2), GS // 2)
    v = np.concatenate([b & 15, b >> 4], axis=2).astype(np.float32) - 8
    return (v * scales.reshape(R, -1, 1)).reshape(R, half * 2)


def convert_expert(task):
    """(e, w13 spec, w2 spec) -> e, packed/scales of gate_up and down, and the squared error sums."""
    e, s13, s2 = task
    w13 = decode_expert(s13, e)
    w13 = np.concatenate([w13[0::2], w13[1::2]])      # gate rows, then up rows
    w2 = decode_expert(s2, e)
    out, stats = [], [0.0, 0.0]
    for w in (w13, w2):
        codes, sc = quantize_g64(w, stats)
        out += [pack_planar(codes), sc]
    return e, out, stats[0], stats[1]


# ---------- the conversion ----------

class Source:
    def __init__(self, indir):
        self.indir = indir
        ip = os.path.join(indir, "model.safetensors.index.json")
        if os.path.exists(ip):
            self.wmap = json.load(open(ip))["weight_map"]
        else:   # a single-file or unindexed snapshot: its headers are the index
            self.wmap = {}
            for p in sorted(glob.glob(os.path.join(indir, "*.safetensors"))):
                for k in read_header(p)[0]:
                    self.wmap[k] = os.path.basename(p)
        self.shards = sorted(set(self.wmap.values()))
        self.headers = {}

    def ready(self, shard):
        """The shard is complete: its name exists (hf download renames at the end) and its
        size is the one its header promises."""
        if shard in self.headers:
            return True
        p = os.path.join(self.indir, shard)
        if not os.path.exists(p):
            return False
        try:
            t, _, size = read_header(p)
        except (OSError, ValueError, struct.error):
            return False
        if os.path.getsize(p) != size:
            return False
        self.headers[shard] = t
        return True

    def index(self, shards):
        """{name: (path, dtype, shape, off, nbytes)} over the given (ready) shards."""
        idx = {}
        for s in shards:
            for k, (dt, shape, off, nb) in self.headers[s].items():
                idx[k] = (os.path.join(self.indir, s), dt, shape, off, nb)
        return idx


def layer_of(name):
    m = re.match(r"^model\.llm\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def write_file(dst, entries, pool, tag):
    """entries: [(name, dtype, shape, nbytes, source)], source ("copy", path, offset), ("expert",)
    for the tensors the expert tasks fill, or one ("experts", prefix, E, w13 spec, w2 spec) entry
    that holds no tensor and starts those tasks. Writes dst under a temporary name, then renames."""
    hdr, offs, size = plan_file([(n, dt, sh, nb) for n, dt, sh, nb, src in entries if src[0] != "experts"])
    tmp = dst + ".tmp"
    t0 = time.time()
    err = ref = 0.0
    with open(tmp, "wb") as f:
        f.write(hdr)
        f.truncate(size)
        for name, dt, shape, nb, src in entries:
            if src[0] == "copy":
                _, path, off = src
                with open(path, "rb") as g:
                    g.seek(off)
                    left = nb
                    f.seek(offs[name])
                    while left:
                        chunk = g.read(min(left, 64 << 20))
                        if not chunk:
                            raise IOError(f"{path}: short read copying {name}")
                        f.write(chunk)
                        left -= len(chunk)
        experts = [src for _, _, _, _, src in entries if src[0] == "experts"]
        for _, pre, E, s13, s2 in experts:
            names = [pre + "gate_up_proj", pre + "gate_up_proj.qs", pre + "down_proj", pre + "down_proj.qs"]
            tasks = [(e, s13, s2) for e in range(E)]
            it = pool.imap_unordered(convert_expert, tasks) if pool else map(convert_expert, tasks)
            for e, arrs, de, dr in it:
                for nm, a in zip(names, arrs):
                    f.seek(offs[nm] + e * a.nbytes)
                    f.write(a.tobytes())
                err += de
                ref += dr
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    rel = f", int4 g64 {np.sqrt(err / ref) * 100:.2f}% from the source weights" if ref else ""
    print(f"{tag}: {os.path.basename(dst)} {size / 1e9:.2f} GB in {time.time() - t0:.0f}s{rel}", flush=True)


def layer_entries(src, L, shards):
    idx = src.index(shards)
    pre = f"model.llm.layers.{L}."
    names = sorted(k for k in idx if k.startswith(pre))
    entries, specs = [], {}
    for k in names:
        m = EXPERT_RE.match(k)
        if m:
            specs[m.group(3)] = expert_spec(idx, k)
            continue
        if k.endswith(AUX_SUFFIXES) and EXPERT_RE.match(re.sub(r"(_scale|_amax|_outer_scale_denominator)$", "", k)):
            continue
        if k.startswith(pre + "mlp.experts.gate_up_proj") or k.startswith(pre + "mlp.experts.down_proj"):
            raise SystemExit(f"{k}: the source already holds converted experts")
        path, dt, shape, off, nb = idx[k]
        entries.append((k, dt, shape, nb, ("copy", path, off)))
    if specs:
        if set(specs) != {"w13", "w2"}:
            raise SystemExit(f"layer {L}: w13_weight and w2_weight come together ({sorted(specs)})")
        E, R13, D = specs["w13"]["shape"]
        E2, D2, I = specs["w2"]["shape"]
        if E2 != E or D2 != D or R13 != 2 * I:
            raise SystemExit(f"layer {L}: w13 {specs['w13']['shape']} and w2 {specs['w2']['shape']} disagree")
        if D % GS or I % GS:
            raise SystemExit(f"layer {L}: int4 g64 needs hidden ({D}) and expert width ({I}) multiples of {GS}")
        pe = pre + "mlp.experts."
        entries += [(pe + "gate_up_proj", "U8", (E, 2 * I, D // 2), E * 2 * I * D // 2, ("expert",)),
                    (pe + "gate_up_proj.qs", "F32", (E, 2 * I, D // GS), E * 2 * I * D // GS * 4, ("expert",)),
                    (pe + "down_proj", "U8", (E, D, I // 2), E * D * I // 2, ("expert",)),
                    (pe + "down_proj.qs", "F32", (E, D, I // GS), E * D * I // GS * 4, ("expert",)),
                    ("", "", (), 0, ("experts", pe, E, specs["w13"], specs["w2"]))]
        fmts = {specs["w13"]["fmt"], specs["w2"]["fmt"]}
        return entries, "/".join(sorted(fmts))
    return entries, "no experts"


def acquire_lock(outdir):
    lock = open(os.path.join(outdir, ".convert.lock"), "w")
    try:
        import fcntl
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit("ERROR: another converter is already using this output directory.")
    except ImportError:
        try:
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:
            pass
        except OSError:
            sys.exit("ERROR: another converter is already using this output directory.")
    return lock


def convert(indir, outdir, jobs, watch=False, delete_src=False, min_free_gb=20.0):
    os.makedirs(outdir, exist_ok=True)
    lock = acquire_lock(outdir)
    src = Source(indir)
    layers = {}
    misc = {}
    for k, s in src.wmap.items():
        L = layer_of(k)
        if L is not None:
            layers.setdefault(L, set()).add(s)
        elif not k.startswith("model.mtp."):
            misc.setdefault(s, []).append(k)
    outs = {}
    for s in sorted(misc):   # first: small, and their shards go sooner
        m = re.search(r"(\d+)-of-\d+\.safetensors$", s)
        outs[f"misc-{m.group(1) if m else os.path.splitext(s)[0]}.safetensors"] = [s]
    for L in sorted(layers):
        outs[f"layer-{L:02d}.safetensors"] = sorted(layers[L])
    readers = {s: [o for o, ss in outs.items() if s in ss] for s in src.shards}
    print(f"{indir}: {len(src.shards)} shard(s), {len(layers)} layers -> {outdir}", flush=True)

    def remove_done():
        """Each shard every output of which is written: gone, the disk free for the next ones."""
        for s, rd in readers.items():
            p = os.path.join(indir, s)
            if os.path.exists(p) and src.ready(s) and all(os.path.exists(os.path.join(outdir, o)) for o in rd):
                os.remove(p)
                print(f"removed {s}", flush=True)

    pool = None
    if jobs > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn" if os.name == "nt" else "fork").Pool(jobs)
    said = None
    try:
        while True:
            if delete_src:
                remove_done()
            did = short = None
            for o, need in outs.items():   # the first output whose shards are all here
                dst = os.path.join(outdir, o)
                if os.path.exists(dst) or not all(src.ready(s) for s in need):
                    continue
                if o.startswith("layer-"):
                    L = int(re.search(r"\d+", o).group())
                    entries, fmt = layer_entries(src, L, need)
                    tag = f"layer {L} ({fmt})"
                else:
                    idx = src.index(need)
                    entries = [(k, idx[k][1], idx[k][2], idx[k][4], ("copy", idx[k][0], idx[k][3]))
                               for k in sorted(misc[need[0]])]
                    tag = f"{need[0]} (outside the layers)"
                size = plan_file([(n, dt, sh, nb) for n, dt, sh, nb, x in entries if x[0] != "experts"])[2]
                free = shutil.disk_usage(outdir).free
                if free < size + min_free_gb * 1e9:   # the disk keeps its floor
                    short = (o, size, free)
                    continue
                write_file(dst, entries, pool, tag)
                did = o
                break
            if did:
                said = None
                continue
            left = [o for o in outs if not os.path.exists(os.path.join(outdir, o))]
            if not left:
                break
            if short and short != said:
                print(f"waiting for disk: {short[0]} needs {short[1] / 1e9:.1f} GB with "
                      f"{min_free_gb:.0f} GB to spare, {short[2] / 1e9:.1f} GB free", flush=True)
                said = short
            if not watch:
                miss = sorted({s for o in left for s in outs[o] if not src.ready(s)})
                if short and not miss:
                    raise SystemExit(f"not enough disk: {short[0]} needs {short[1] / 1e9:.1f} GB "
                                     f"with {min_free_gb:.0f} GB to spare ({short[2] / 1e9:.1f} GB free)")
                raise SystemExit(f"incomplete source: {len(left)} output(s) wait for {', '.join(miss)}")
            time.sleep(30)
        if delete_src:
            remove_done()
    finally:
        if pool:
            pool.close()
            pool.join()
    mtp = os.path.join(indir, "mtp.safetensors")
    if os.path.exists(mtp):
        os.makedirs(os.path.join(outdir, "mtp"), exist_ok=True)
        dst = os.path.join(outdir, "mtp", "mtp.safetensors")
        if delete_src:   # moved, as the shards go: no second copy on the disk
            shutil.move(mtp, dst)
        elif not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(mtp):
            shutil.copyfile(mtp, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
    for fn in AUX_FILES:
        p = os.path.join(indir, fn)
        if not os.path.exists(p):
            continue
        if fn == "config.json":
            cfg = json.load(open(p))
            cfg.pop("quantization_config", None)
            cfg["colibri_experts"] = "int4-g64"
            with open(os.path.join(outdir, fn), "w") as f:
                json.dump(cfg, f, indent=2)
        else:
            shutil.copy(p, outdir)
    tk = os.path.join(indir, "tiktoken")
    if os.path.isdir(tk):
        shutil.copytree(tk, os.path.join(outdir, "tiktoken"), dirs_exist_ok=True)
    lock.close()
    print(f"conversion complete -> {outdir}", flush=True)


# ---------- fabricated sources (tests) ----------

def e4m3_round(x):
    """f32 -> nearest finite e4m3 byte (saturating at 448)."""
    pos = E4M3[:0x7F]
    a = np.minimum(np.abs(x), 448.0)
    i = np.abs(a[..., None] - pos).argmin(axis=-1).astype(np.uint8)
    return np.where(x < 0, i | 0x80, i).astype(np.uint8)


def encode_nvfp4(w):
    """f32 [E, R, C] -> (U8 [E, R, C/2], e4m3 bytes [E, R, C/16], amax F32 [E], denominator 2688)."""
    E, R, C = w.shape
    den = np.float32(448 * 6)
    amax = np.abs(w).reshape(E, -1).max(axis=1).astype(np.float32)
    gsc = (amax / den).reshape(E, 1, 1, 1)
    blk = w.reshape(E, R, C // 16, 16)
    s = e4m3_round(np.abs(blk).max(axis=3, keepdims=True) / 6 / np.maximum(gsc, 1e-30))
    sv = E4M3[s] * gsc
    x = np.where(sv > 0, blk / np.where(sv > 0, sv, 1), 0)
    mag = np.abs(FP4[:8])
    code = np.abs(np.abs(x)[..., None] - mag).argmin(axis=-1).astype(np.uint8)
    code = np.where((x < 0) & (code > 0), code | 8, code).reshape(E, R, C)
    return (code[:, :, 0::2] | (code[:, :, 1::2] << 4)).astype(np.uint8), s.reshape(E, R, C // 16), amax, den


def encode_mxfp8(w):
    """f32 [E, R, C] -> (e4m3 bytes [E, R, C], ue8m0 bytes [E, R, C/32])."""
    E, R, C = w.shape
    blk = w.reshape(E, R, C // 32, 32)
    m = np.abs(blk).max(axis=3, keepdims=True)
    ex = np.clip(np.ceil(np.log2(np.maximum(m, 1e-30) / 448.0)), -127, 127)
    return e4m3_round(blk / np.exp2(ex)).reshape(E, R, C), (ex + 127).astype(np.uint8).reshape(E, R, C // 32)


def write_safetensors(path, tensors):
    """tensors {name: (dtype, shape, bytes)}"""
    hdr, offs, size = plan_file([(n, dt, sh, len(b)) for n, (dt, sh, b) in tensors.items()])
    with open(path, "wb") as f:
        f.write(hdr)
        for n, (dt, sh, b) in sorted(tensors.items(), key=lambda kv: offs[kv[0]]):
            assert f.tell() == offs[n]
            f.write(b)


def fabricate(snapdir, outdir, nvfp4_layers):
    """A plain snapshot -> the official checkpoint's layout (see the module's docstring)."""
    os.makedirs(outdir, exist_ok=True)
    tens = {}
    for p in sorted(glob.glob(os.path.join(snapdir, "*.safetensors"))):
        for k, (dt, shape, off, nb) in read_header(p)[0].items():
            tens[k] = (dt, shape, read_bytes(p, off, nb))
    out = {}
    for k, (dt, shape, raw) in tens.items():
        m = EXPERT_RE.match(k)
        if not m:
            out[k] = (dt, shape, raw)
            continue
        w = to_f32(raw, dt, shape)
        if int(m.group(2)) < nvfp4_layers:
            q, s, amax, den = encode_nvfp4(w)
            out[k] = ("U8", q.shape, q.tobytes())
            out[k + "_scale"] = ("F8_E4M3", s.shape, s.tobytes())
            out[k + "_amax"] = ("F32", amax.shape, amax.astype("<f4").tobytes())
            out[k + "_outer_scale_denominator"] = ("F32", (1,), np.array([den], "<f4").tobytes())
        else:
            q, s = encode_mxfp8(w)
            out[k] = ("F8_E4M3", q.shape, q.tobytes())
            out[k + "_scale"] = ("U8", s.shape, s.tobytes())
    # two shards: names in order, cut in the middle, so that a layer's tensors straddle them
    names = sorted(out)
    cut = len(names) // 2
    parts = {"model-00001-of-00002.safetensors": names[:cut], "model-00002-of-00002.safetensors": names[cut:]}
    wmap = {}
    for fn, ns in parts.items():
        write_safetensors(os.path.join(outdir, fn), {n: out[n] for n in ns})
        wmap.update({n: fn for n in ns})
    with open(os.path.join(outdir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wmap}, f, indent=1)
    for fn in AUX_FILES:
        p = os.path.join(snapdir, fn)
        if os.path.exists(p):
            shutil.copy(p, outdir)
    print(f"fabricated {outdir}: experts of layers < {nvfp4_layers} NVFP4, the others MXFP8")


# ---------- selftest ----------

def selftest():
    assert E4M3[0x38] == 1.0 and E4M3[0x7E] == 448.0 and E4M3[0x01] == 2.0 ** -9 and E4M3[0xB8] == -1.0
    assert np.isnan(E4M3[0x7F]) and np.isnan(E4M3[0xFF])
    rng = np.random.default_rng(0)
    # planar layout against expert_ffn.h's xf_dot_f32_ref reading of a block
    codes = rng.integers(0, 16, (3, 128)).astype(np.uint8)
    p = pack_planar(codes)
    for r in range(3):
        for i in range(128):
            blk = p[r, (i // 64) * 32:(i // 64) * 32 + 32]
            k = i % 64
            v = (blk[k] & 15) if k < 32 else (blk[k - 32] >> 4)
            assert v == codes[r, i]
    # the scale search is never worse than absmax/7, and its codes and scales decode to its weights
    w = (rng.standard_normal((32, 256)) * 0.02).astype(np.float32)
    w[0, :64] = 0
    c, s = quantize_g64(w)
    g = w.reshape(32, 4, 64)
    s7 = np.maximum(np.abs(g).max(axis=2, keepdims=True), 1e-12) / 7
    e7 = np.square(np.clip(np.rint(g / s7), -8, 7) * s7 - g).sum()
    d = unpack_planar(pack_planar(c), s)
    assert np.square(d - w).sum() <= e7 and np.all(d[0, :64] == 0)
    # NVFP4 and MXFP8: the encodings of tests decode back near the weights; an exact grid exactly
    w = (rng.standard_normal((2, 4, 64)) * 0.05).astype(np.float32)
    q, sc, amax, den = encode_nvfp4(w)
    spec = {"fmt": "nvfp4", "shape": (2, 4, 64)}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "t.bin")
        with open(pth, "wb") as f:
            f.write(q.tobytes()); f.write(sc.tobytes()); f.write(amax.astype("<f4").tobytes())
            f.write(np.array([den], "<f4").tobytes())
        spec.update(w=(pth, 0), s=(pth, q.nbytes), amax=(pth, q.nbytes + sc.nbytes),
                    den=(pth, q.nbytes + sc.nbytes + 8))
        for e in range(2):
            v = decode_expert(spec, e)
            assert np.abs(v - w[e]).max() <= np.abs(w[e]).max() * 0.2, "NVFP4 round trip"
        q8, s8 = encode_mxfp8(w)
        pth8 = os.path.join(td, "m.bin")
        with open(pth8, "wb") as f:
            f.write(q8.tobytes()); f.write(s8.tobytes())
        spec8 = {"fmt": "mxfp8", "shape": (2, 4, 64), "w": (pth8, 0), "s": (pth8, q8.nbytes)}
        for e in range(2):
            v = decode_expert(spec8, e)
            assert np.abs(v - w[e]).max() <= np.abs(w[e]).max() * 0.07, "MXFP8 round trip"
    # w13: even rows gate
    w13 = np.arange(6 * 64, dtype=np.float32).reshape(6, 64)
    gu = np.concatenate([w13[0::2], w13[1::2]])
    assert gu[0, 0] == 0 and gu[1, 0] == 128 and gu[3, 0] == 64
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--indir")
    ap.add_argument("--outdir")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--watch", action="store_true", help="wait for the shards a download is still writing")
    ap.add_argument("--delete-src", action="store_true", help="remove each shard once its outputs are written")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="free disk an output must leave behind it, or the run waits (--watch) or stops")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fabricate", nargs=2, metavar=("SNAPSHOT", "OUT"))
    ap.add_argument("--nvfp4-layers", type=int, default=1)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.fabricate:
        fabricate(*a.fabricate, a.nvfp4_layers)
        return
    if not a.indir or not a.outdir:
        ap.error("--indir and --outdir are required")
    convert(a.indir, a.outdir, max(1, a.jobs), watch=a.watch, delete_src=a.delete_src,
            min_free_gb=a.min_free_gb)


if __name__ == "__main__":
    main()
