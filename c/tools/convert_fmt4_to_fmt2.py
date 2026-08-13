#!/usr/bin/env python3
"""fmt=4 -> fmt=2 converter: GLM-5.2 colibri container, grouped int4 (gs=64) -> per-row int4.

Motivation: the Metal backend rejects fmt=4 at its attention/dispatch entry points
(#585/#587); fmt=2 (per-row int4) is the format with full backend support. Converted
container benchmarked in docs/METAL-M1ULTRA-FMT2-REPORT.md.

Container layout (both formats): every quantized tensor is a FLAT 1-D U8 array `name` (packed
nibbles, offset binary: value = nibble - 8, LOW nibble = even element) plus a FLAT 1-D F32
companion `name.qs`. The engine detects the format purely by byte counts (qt_resolve_fmt,
c/colibri.c:1027): int4 weight bytes = O*I/2; `.qs` of O floats -> fmt=2, O*ceil(I/64) floats
-> fmt=4. So conversion = rewrite each fmt=4 tensor with exactly O per-row scales.

Per-tensor classification by element-count ratio nb/ns (weight bytes / scale floats):
  - 32 exactly              -> fmt=4 g64 (CONVERT; shape-independent discriminator)
  - nb == O*I,   ns == O    -> int8 per-row (copy verbatim: embed_tokens, lm_head)
  - nb == O*I/2, ns == O    -> int4 per-row already (copy verbatim)
  - anything else           -> abort loudly
O and I come from a shape table derived from the tensor name + config.json; byte/scale counts
are asserted against the table before every conversion. F32 tensors (norms, mlp.gate router,
biases) are copied. The MTP shard (out-mtp-*.safetensors, per-row int8) is byte-copied whole.

Conversion math (all dims are multiples of 64):
  1. dequantize  w[o,i] = (nib[o,i] - 8) * qs[o*ng + i//64]   to f32, ng = I/64
  2. re-quantize per-row with the EXACT math of quant_int4 in convert_fp8_to_int4.py
     (np.rint, absmax/7 clamped >= 1e-8, clip -8..7, +8, low nibble = even element)
  3. write flat U8 [O*I/2] + flat F32 .qs of exactly O floats -> engine auto-detects fmt=2

USO:
  python3 -m pip install numpy safetensors
  python3 c/tools/convert_fmt4_to_fmt2.py --selftest
  python3 c/tools/convert_fmt4_to_fmt2.py --indir SRC --outdir DST --dry-run
  python3 c/tools/convert_fmt4_to_fmt2.py --indir SRC --outdir DST [--workers 8]

Resume: a re-run skips output shards that already exist with the expected byte size (the exact
size is computable a priori — safetensors serialization is deterministic, see expected_size()).
Writes go through a .tmp file + os.replace, so an interrupted run never leaves a half shard.
"""
import os, sys, glob, json, re, shutil, struct, argparse
import numpy as np

GS = 64                                    # fmt=4 group size of the source container
META_FILES = ["config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "README.md"]
DTYPE_SIZE = {"U8": 1, "F32": 4}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_fp8_to_int4 import quant_int4, quant_int4_grouped   # exact reference math


def die(msg):
    raise SystemExit(f"ERROR: {msg}")


# ---------- safetensors header (pure python, no mmap of the data) ----------
def read_header(path):
    """Return {name: {"dtype": str, "shape": [int,...]}} — header-only read, fast on NFS."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


# ---------- shape table: name -> (O, I), derived from config.json ----------
def shape_of(name, C):
    H = C["hidden_size"]
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return (C["vocab_size"], H)
    if not (name.startswith("model.layers.") and name.endswith(".weight")):
        return None
    rest = name[len("model.layers."):]
    li, _, rest = rest.partition(".")
    if not li.isdigit():
        return None
    li = int(li)
    qk_dim = C["qk_nope_head_dim"] + C["qk_rope_head_dim"]
    if rest == "self_attn.q_a_proj.weight":
        return (C["q_lora_rank"], H)
    if rest == "self_attn.q_b_proj.weight":
        return (C["num_attention_heads"] * qk_dim, C["q_lora_rank"])
    if rest == "self_attn.kv_a_proj_with_mqa.weight":
        return (C["kv_lora_rank"] + C["qk_rope_head_dim"], H)
    if rest == "self_attn.kv_b_proj.weight":
        return (C["num_attention_heads"] * (C["qk_nope_head_dim"] + C["v_head_dim"]),
                C["kv_lora_rank"])
    if rest == "self_attn.o_proj.weight":
        return (H, C["num_attention_heads"] * C["v_head_dim"])
    if li < C["first_k_dense_replace"]:                    # dense MLP, layers 0..2
        if rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            return (C["intermediate_size"], H)
        if rest == "mlp.down_proj.weight":
            return (H, C["intermediate_size"])
    else:                                                  # MoE: routed + shared experts
        if re.match(r"mlp\.(experts\.\d+|shared_experts)\.(gate_proj|up_proj)\.weight$", rest):
            return (C["moe_intermediate_size"], H)
        if re.match(r"mlp\.(experts\.\d+|shared_experts)\.down_proj\.weight$", rest):
            return (H, C["moe_intermediate_size"])
    return None


# ---------- per-tensor classification ----------
def classify_tensor(name, info, header, C):
    """-> ("convert", O, I) | ("copy",). Aborts loudly on anything unrecognized."""
    dt, sh = info["dtype"], info["shape"]
    qs = header.get(name + ".qs")
    if qs is None:
        if dt == "F32":
            return ("copy",)                               # norms, router, biases
        die(f"{name}: no .qs companion and dtype {dt} (expected F32); refusing")
    if dt != "U8" or len(sh) != 1:
        die(f"{name}: quantized weight must be flat U8, got {dt} {sh}; refusing")
    if qs["dtype"] != "F32" or len(qs["shape"]) != 1:
        die(f"{name}.qs: scales must be flat F32, got {qs['dtype']} {qs['shape']}; refusing")
    nb, ns = sh[0], qs["shape"][0]
    OI = shape_of(name, C)
    if nb == 32 * ns:                                      # fmt=4 g64, shape-independent
        if OI is None:
            die(f"{name}: fmt=4 counts (nb/ns == 32) but no shape-table entry; refusing")
        O, I = OI
        if I % GS:
            die(f"{name}: I={I} is not a multiple of {GS}; refusing")
        if nb != O * I // 2 or ns != O * (I // GS):
            die(f"{name}: shape-table mismatch for {O}x{I}: nb={nb} (want {O*I//2}), "
                f"ns={ns} (want {O*(I//GS)}); refusing")
        return ("convert", O, I)
    if OI is None:
        die(f"{name}: quantized tensor not in shape table (nb={nb}, ns={ns}); refusing")
    O, I = OI
    if nb == O * I and ns == O:
        return ("copy",)                                   # int8 per-row (embed, lm_head)
    if nb == O * I // 2 and ns == O:
        return ("copy",)                                   # int4 per-row already
    die(f"{name}: unrecognized quantized layout nb={nb} ns={ns} for {O}x{I}; refusing")


# ---------- exact output-size prediction (safetensors serialization is deterministic) ----------
def expected_size(items):
    """items: list of (name, dtype, shape). The Rust serializer sorts by (dtype, name) —
    verified empirically: F32 group first, names ascending within a dtype — emits compact
    JSON, and space-pads the header so (8 + header_len) % 8 == 0. Verified byte-for-byte
    against safetensors.numpy.save_file (selftest)."""
    parts, off = [], 0
    for name, dt, sh in sorted(items, key=lambda t: (t[1], t[0])):
        nb = DTYPE_SIZE[dt] * int(np.prod(sh, dtype=np.int64))
        parts.append('%s:{"dtype":"%s","shape":[%s],"data_offsets":[%d,%d]}'
                     % (json.dumps(name), dt, ",".join(map(str, sh)), off, off + nb))
        off += nb
    j = "{" + ",".join(parts) + "}"
    return 8 + len(j) + ((-len(j)) % 8) + off


# ---------- shard plan ----------
def plan_shard(path, C, is_mtp):
    """Header-only: -> dict(converts=[(name,O,I)], copies=[names], out_bytes=int)."""
    h = read_header(path)
    converts, copies, items = [], [], []
    for name, info in h.items():
        if name.endswith(".qs"):
            continue
        if is_mtp:                                         # whole shard byte-copied anyway;
            copies.append(name)                            # classify leniently for the report
            qs = h.get(name + ".qs")
            if qs is not None and info["dtype"] == "U8" and \
                    info["shape"][0] == 32 * qs["shape"][0]:
                die(f"{os.path.basename(path)}:{name}: fmt=4 counts in the MTP shard — "
                    "the byte-copy assumption is broken; refusing")
            items.append((name, info["dtype"], info["shape"]))
            if qs is not None:
                items.append((name + ".qs", "F32", qs["shape"]))
            continue
        cls = classify_tensor(name, info, h, C)
        if cls[0] == "convert":
            _, O, I = cls
            converts.append((name, O, I))
            items.append((name, "U8", [O * I // 2]))
            items.append((name + ".qs", "F32", [O]))
        else:
            copies.append(name)
            items.append((name, info["dtype"], info["shape"]))
            if name + ".qs" in h:                        # int8/int4-per-row copies keep
                items.append((name + ".qs", "F32", h[name + ".qs"]["shape"]))  # their .qs;
    return {"converts": converts, "copies": copies, "out_bytes": expected_size(items)}


# ---------- conversion math ----------
def dequant_fmt4(q, qs, O, I):
    """fmt=4 -> f32 [O,I]: w[o,i] = (nib[o,i] - 8) * qs[o*ng + i//64]. Low nibble = even."""
    ng = I // GS
    qb = q.reshape(O, I // 2)
    w = np.empty((O, I), np.float32)
    w[:, 0::2] = qb & 0x0F                                 # even element = LOW nibble
    w[:, 1::2] = qb >> 4                                   # odd element = HIGH nibble
    w -= 8.0
    w = w.reshape(O, ng, GS)
    w *= qs.reshape(O, ng, 1)                              # in-place broadcast, no big temp
    return w.reshape(O, I)


def convert_fmt4_to_fmt2(q, qs, O, I):
    """-> (flat U8 [O*I/2], flat F32 [O]) — per-row requant with the exact engine math."""
    return quant_int4(dequant_fmt4(q, qs, O, I), 4)


def convert_shard(path, C):
    """Worker: -> out_dict {name: np.ndarray} ready for safetensors.numpy.save_file."""
    from safetensors import safe_open
    h = read_header(path)
    out = {}
    with safe_open(path, framework="numpy") as f:
        for name, info in h.items():
            if name.endswith(".qs"):
                continue
            cls = classify_tensor(name, info, h, C)
            if cls[0] == "convert":
                _, O, I = cls
                q2, s2 = convert_fmt4_to_fmt2(f.get_tensor(name),
                                              f.get_tensor(name + ".qs"), O, I)
                out[name] = q2
                out[name + ".qs"] = s2
            else:
                out[name] = f.get_tensor(name)
                if name + ".qs" in h:
                    out[name + ".qs"] = f.get_tensor(name + ".qs")
    return out


def _convert_one(args):
    i, sp, C = args
    return i, convert_shard(sp, C)


def free_gb(p):
    while not os.path.isdir(p):                            # outdir may not exist yet
        parent = os.path.dirname(os.path.abspath(p))
        if parent == p:
            break
        p = parent
    return shutil.disk_usage(p).free / 1e9


# ---------- selftest ----------
def selftest():
    import tempfile
    print("[selftest] fmt=4 -> fmt=2 math + container-size predictor")
    # 1) round-trip on synthetic fmt=4 data (built with the reference grouped quantizer)
    rng = np.random.default_rng(0)
    O, I = 96, 256
    w = (rng.standard_normal((O, I)) * 0.05).astype(np.float32)
    q4, s4 = quant_int4_grouped(w, 4, GS)
    assert q4.shape == (O * I // 2,) and s4.shape == (O * I // GS,)
    q2, s2 = convert_fmt4_to_fmt2(q4, s4, O, I)
    assert q2.shape == (O * I // 2,), q2.shape
    assert s2.shape == (O,), f".qs must have exactly O floats, got {s2.shape}"
    ref = dequant_fmt4(q4, s4, O, I)                       # dequant of the fmt=4 source
    qb = q2.reshape(O, I // 2)                             # dequant of the fmt=2 output
    deq = np.empty((O, I), np.float32)
    deq[:, 0::2] = (qb & 0x0F).astype(np.int32) - 8
    deq[:, 1::2] = ((qb >> 4) & 0x0F).astype(np.int32) - 8
    deq *= s2[:, None]
    rel = float(np.abs(deq - ref).mean() / (np.abs(ref).mean() + 1e-12))
    print(f"[selftest] dequant round-trip: mean rel err = {rel:.4f} "
          f"({'OK' if rel < 0.10 else 'FAIL'})")
    assert rel < 0.10, f"round-trip rel err {rel:.3f} too high"
    # 2) nibble order through the whole path: even element -> LOW nibble
    w1 = np.zeros((1, 64), np.float32)
    w1[0, 0], w1[0, 1] = 0.9, -0.9
    q4, s4 = quant_int4_grouped(w1, 4, GS)                 # q = [+7, -7, ...] -> byte 0x1F
    assert q4[0] == 0x1F, f"fmt=4 fixture byte {q4[0]:#06x} != 0x1f"
    q2, s2 = convert_fmt4_to_fmt2(q4, s4, 1, 64)
    assert q2[0] == 0x1F, (f"nibble order wrong: byte {q2[0]:#06x} != 0x1f "
                           "(low nibble must be the even element)")
    print("[selftest] nibble order: low nibble = even element OK")
    # 3) expected_size vs real save_file, sweeping all header-pad residues
    from safetensors.numpy import save_file
    d = tempfile.mkdtemp()
    try:
        for k in range(10):                                # hits every pad residue 0..7
            nm = "w" * (k + 1) + ".weight"
            tens = {nm: np.zeros(5 + k, np.uint8), nm + ".qs": np.zeros(2, np.float32),
                    "n" * (k + 1): np.zeros(3, np.float32)}
            p = os.path.join(d, f"t{k}.safetensors")
            save_file(tens, p)
            items = [(n, str(v.dtype).replace("uint8", "U8").replace("float32", "F32"),
                      list(v.shape)) for n, v in tens.items()]
            want, got = expected_size(items), os.path.getsize(p)
            assert want == got, f"expected_size {want} != real {got} (case {k})"
        print("[selftest] expected_size == save_file size on 10/10 cases OK")
    finally:
        shutil.rmtree(d)
    print("[selftest] OK")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir")
    ap.add_argument("--outdir")
    ap.add_argument("--dry-run", action="store_true",
                    help="header-only scan + classification report + size estimates")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="conversion processes (default 8); the main process still writes "
                         "shards in order, so output is byte-identical for any worker count")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.indir or not a.outdir:
        ap.error("--indir and --outdir are required")
    a.outdir = os.path.expanduser(a.outdir)
    if os.path.abspath(a.indir) == os.path.abspath(a.outdir):
        die("--indir and --outdir must differ (in-place conversion would corrupt the source)")
    cfg_path = os.path.join(a.indir, "config.json")
    if not os.path.exists(cfg_path):
        die(f"{cfg_path} not found — the shape table is derived from config.json")
    C = json.load(open(cfg_path))

    shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))   # top level only:
    if not shards:                                                       # _meta/_inflight skipped
        die(f"no *.safetensors in {a.indir}")
    main_shards = [p for p in shards if not os.path.basename(p).startswith("out-mtp-")]
    mtp_shards = [p for p in shards if os.path.basename(p).startswith("out-mtp-")]

    # ---- plan: header-only scan of every shard (validates the shape table everywhere) ----
    plans, total_in, total_out = {}, 0, 0
    n_conv = n_copy = 0
    for sp in shards:
        bn = os.path.basename(sp)
        is_mtp = bn.startswith("out-mtp-")
        pl = plan_shard(sp, C, is_mtp)
        if is_mtp:
            pl["out_bytes"] = os.path.getsize(sp)          # byte copy: same size
        plans[bn] = pl
        total_in += os.path.getsize(sp)
        total_out += pl["out_bytes"]
        n_conv += len(pl["converts"])
        n_copy += len(pl["copies"])
    print(f"[PLAN] {len(shards)} shard(s): {len(main_shards)} main + {len(mtp_shards)} MTP | "
          f"{n_conv} tensor(s) to convert (fmt=4 g64 -> fmt=2 per-row), "
          f"{n_copy} to copy (int8 / F32 / per-row)")
    print(f"[PLAN] input {total_in/1e9:.1f} GB -> output {total_out/1e9:.1f} GB "
          f"(delta {(total_in-total_out)/1e9:.1f} GB of grouped scales)")
    if a.dry_run:
        per_fmt = {}
        for bn, pl in plans.items():
            for name, O, I in pl["converts"]:
                key = re.sub(r"\d+", "*", name)            # aggregate layers/experts
                per_fmt.setdefault(key, [0, O, I])
                per_fmt[key][0] += 1
        print("[DRY-RUN] converted-tensor shapes (count, O x I):")
        for k in sorted(per_fmt):
            n, O, I = per_fmt[k]
            print(f"    {n:6d}  {O:6d} x {I:6d}  {k}")
        print(f"[DRY-RUN] shape table validated against all {len(shards)} shard headers OK")
        print(f"[DRY-RUN] outdir filesystem free: {free_gb(a.outdir):.0f} GB "
              f"(need ~{total_out/1e9:.0f} GB)")
        return

    # ---- preflight ----
    os.makedirs(a.outdir, exist_ok=True)
    remaining = sum(pl["out_bytes"] for bn, pl in plans.items()
                    if not _shard_done(os.path.join(a.outdir, bn), pl["out_bytes"]))
    if free_gb(a.outdir) * 1e9 < remaining:
        die(f"not enough free space on {a.outdir}: {free_gb(a.outdir):.0f} GB free, "
            f"~{remaining/1e9:.0f} GB still needed")
    print(f"[PLAN] resume check: {remaining/1e9:.1f} GB to write, "
          f"{free_gb(a.outdir):.0f} GB free")

    # ---- MTP shard(s): byte copy whole ----
    for sp in mtp_shards:
        bn = os.path.basename(sp)
        dst = os.path.join(a.outdir, bn)
        if _shard_done(dst, plans[bn]["out_bytes"]):
            print(f"[RESUME] {bn} already done, skipped")
            continue
        print(f"[MTP] byte-copying {bn} ({os.path.getsize(sp)/1e9:.1f} GB)", flush=True)
        tmp = dst + ".tmp"
        shutil.copyfile(sp, tmp)
        os.replace(tmp, dst)

    # ---- main shards: workers convert, main process writes in order ----
    from safetensors.numpy import save_file
    import time as _t
    t_start = _t.time()
    fresh = skipped = 0
    _pool = _result_it = None
    if a.workers and a.workers > 1:
        from multiprocessing import Pool
        _pending = [(i, sp, C) for i, sp in enumerate(main_shards)
                    if not _shard_done(os.path.join(a.outdir, os.path.basename(sp)),
                                       plans[os.path.basename(sp)]["out_bytes"])]
        _pool = Pool(a.workers)
        _result_it = iter(_pool.imap(_convert_one, _pending))
    for i, sp in enumerate(main_shards):
        bn = os.path.basename(sp)
        dst = os.path.join(a.outdir, bn)
        pl = plans[bn]
        if _shard_done(dst, pl["out_bytes"]):
            skipped += 1
            continue
        eta = ""
        if fresh:
            per = (_t.time() - t_start) / fresh
            eta = f", ETA {per * (len(main_shards) - i) / 3600:.1f} h"
        print(f"[{i + 1}/{len(main_shards)}] {bn}: {len(pl['converts'])} convert / "
              f"{len(pl['copies'])} copy ({free_gb(a.outdir):.0f} GB free{eta})", flush=True)
        if _result_it is not None:
            _ri, out = next(_result_it)                    # converted by a worker, in order
        else:
            out = convert_shard(sp, C)
        tmp = dst + ".tmp"
        save_file(out, tmp)
        got = os.path.getsize(tmp)
        if got != pl["out_bytes"]:                         # predictor bug must not ship
            os.remove(tmp)
            die(f"{bn}: wrote {got} bytes, plan expected {pl['out_bytes']} — aborting "
                "rather than resuming from a mispredicted size")
        os.replace(tmp, dst)
        fresh += 1
    if _pool is not None:
        _pool.close()
        _pool.join()
    if skipped:
        print(f"[RESUME] {skipped} shard(s) already done in {a.outdir}, skipped")

    # ---- metadata ----
    copied, missing = [], []
    for fn in META_FILES:
        src = os.path.join(a.indir, fn)
        if os.path.exists(src):
            shutil.copy(src, a.outdir)
            copied.append(fn)
        else:
            missing.append(fn)
    print(f"[META] copied from {a.indir}: {', '.join(copied) if copied else 'nothing'}")
    if missing:
        print(f"[META] WARNING: not found in {a.indir}: {', '.join(missing)}"
              + (" — chat/serve need tokenizer.json" if "tokenizer.json" in missing else ""))
    print(f"converted {fresh} shard(s) -> {a.outdir} "
          f"({total_out/1e9:.1f} GB, fmt=2 per-row int4)")


def _shard_done(dst, expected_bytes):
    return os.path.exists(dst) and os.path.getsize(dst) == expected_bytes


if __name__ == "__main__":
    main()
