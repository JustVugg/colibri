"""Repair an existing colibri int4 container whose MTP head was quantized at int4.

WHY THIS EXISTS
  `model.layers.<N>.eh_proj.weight` [D, 2D] multiplies the MTP concat
  [embedding_norm ; hidden_norm], and its two column halves differ in scale by
  ~20-30x per row (embedding-half absmax ~0.05, hidden-half ~1.5 on GLM-5.2).
  Per-row int4 uses ONE scale (= absmax/7) per row, so every embedding-half
  weight lands below half a quantization step and np.rint rounds the ENTIRE
  embedding half to exact zeros (packed bytes 0x88). The MTP head then drafts
  garbage: acceptance ~0% (issue #8 measured 0-4% at int4; 39-59% at int8 —
  which is why `convert_fp8_to_int4.py --mtp` defaults to --ebits 8).

  A container converted (or downloaded) with an int4 MTP head does not need a
  full re-conversion: this script re-downloads ONLY the affected dense tensors
  (~355 MB of HTTP range reads against the FP8 source repo), requantizes them
  at int8 with the converter's exact math, and patches the local shards in
  place. Originals are kept beside as *.bak-int4.

WHAT IT TOUCHES
  The MTP layer's dense tensors only (eh_proj, q_a/q_b/kv_a/kv_b/o_proj,
  shared_experts.*): the ones that stream into RAM once and stay resident.
  Routed experts (model.layers.<N>.mlp.experts.*) are NOT touched — they are
  statistically like the main layers' experts and int4 is acceptable there.
  The engine auto-detects int8 vs int4 by blob size (qt_from_disk), so no
  engine or config change is needed. Cost: ~+133 MB on disk / resident RAM.

USAGE
  python3 tools/repair_mtp_int8.py --snap /path/to/glm52_i4              # repair
  python3 tools/repair_mtp_int8.py --snap /path/to/glm52_i4 --dry-run    # inspect only

  --source-repo defaults to zai-org/GLM-5.2-FP8 (the checkpoint the public
  int4 containers were converted from). Requires numpy and network access;
  no torch, no HF token (public repo, anonymous range reads).
"""
import argparse, glob, json, os, ssl, struct, sys, urllib.request
import numpy as np

# macOS python.org builds ship no CA bundle: use certifi when available (Linux
# system Pythons generally have working system certs and skip this).
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# ---------- HTTP range reads against the source repo ----------
def http_range(url, start, length, tries=5):
    req = urllib.request.Request(url, headers={"User-Agent": "colibri-mtp-repair",
                                               "Range": f"bytes={start}-{start+length-1}"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as r:
                data = r.read()
            if len(data) == length:
                return data
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            if attempt == tries - 1:
                raise RuntimeError(f"range read failed for {url}: {ex}")
    raise RuntimeError(f"short range read for {url}")


class SourceRepo:
    def __init__(self, repo, revision="main"):
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        with urllib.request.urlopen(self.base + "model.safetensors.index.json", timeout=30, context=_SSL_CTX) as r:
            self.wmap = json.loads(r.read())["weight_map"]
        self._hdr = {}

    def _shard_header(self, shard):
        if shard not in self._hdr:
            n = struct.unpack("<Q", http_range(self.base + shard, 0, 8))[0]
            self._hdr[shard] = (json.loads(http_range(self.base + shard, 8, n)), 8 + n)
        return self._hdr[shard]

    def meta(self, name):
        shard = self.wmap.get(name)
        if not shard:
            return None
        hdr, _ = self._shard_header(shard)
        return hdr[name]

    def fetch_f32(self, name):
        """Download one tensor and dequantize to f32 [O, I] (BF16 or FP8+block scales)."""
        shard = self.wmap[name]
        hdr, base = self._shard_header(shard)
        m = hdr[name]
        o0, o1 = m["data_offsets"]
        raw = http_range(self.base + shard, base + o0, o1 - o0)
        if m["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(m["shape"]).astype(np.float32)
        if m["dtype"] == "F8_E4M3":
            b = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
            sign = np.where(b & 0x80, -1.0, 1.0)
            e = (b >> 3) & 0xF
            mant = (b & 7).astype(np.float64)
            v = (sign * np.where(e > 0, (1 + mant / 8) * np.exp2(e.astype(np.float64) - 7),
                                 mant / 8 * np.exp2(-6.0))).reshape(m["shape"])
            sn = name + "_scale_inv"
            sshard = self.wmap[sn]
            shdr, sbase = self._shard_header(sshard)
            sm = shdr[sn]
            so0, so1 = sm["data_offsets"]
            sc = np.frombuffer(http_range(self.base + sshard, sbase + so0, so1 - so0),
                               dtype=np.float32).reshape(sm["shape"])
            O, I = m["shape"]
            scf = np.repeat(np.repeat(sc, 128, axis=0)[:O], 128, axis=1)[:, :I]
            return (v * scf).astype(np.float32)
        raise ValueError(f"{name}: unsupported source dtype {m['dtype']}")


# ---------- quantization: identical to convert_fp8_to_int4.quant_int8 ----------
def quant_int8(w):
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / 127, 1e-8)
    q = np.clip(np.rint(w / s), -128, 127).astype(np.int8)
    return q.reshape(-1).view(np.uint8).copy(), s[:, 0].astype(np.float32)


# ---------- local safetensors IO (no deps; preserves byte-identity of untouched tensors) ----------
def read_shard(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        order = sorted(((k, v) for k, v in hdr.items() if k != "__metadata__"),
                       key=lambda kv: kv[1]["data_offsets"][0])
        out = {}
        for k, v in order:
            fh.seek(base + v["data_offsets"][0])
            out[k] = (v["dtype"], v["shape"], fh.read(v["data_offsets"][1] - v["data_offsets"][0]))
    return out, hdr.get("__metadata__")


def write_shard(path, tensors, meta):
    hdr = {}
    off = 0
    for k, (dt, shape, raw) in tensors.items():
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    if meta:
        hdr["__metadata__"] = meta
    hj = json.dumps(hdr).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)))
        fh.write(hj)
        for _, (_, _, raw) in tensors.items():
            fh.write(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap", required=True, help="local colibri int4 container directory")
    ap.add_argument("--source-repo", default="zai-org/GLM-5.2-FP8")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, touch nothing")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.snap, "config.json")))
    L = cfg["num_hidden_layers"]
    pref = f"model.layers.{L}."
    src = SourceRepo(a.source_repo, a.revision)

    # find the MTP layer's dense int4 tensors across the local shards
    plan = {}   # shard path -> [tensor names to repair]
    already_ok, skipped_experts = [], 0
    for f in sorted(glob.glob(os.path.join(a.snap, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, v in hdr.items():
            if not name.startswith(pref) or name.endswith(".qs") or v.get("dtype") != "U8":
                continue
            if ".mlp.experts." in name:
                skipped_experts += 1
                continue
            m = src.meta(name)
            if m is None:
                print(f"  ?? {name}: not in source repo, skipping")
                continue
            O, I = m["shape"]
            nb = v["data_offsets"][1] - v["data_offsets"][0]
            if nb == O * I:
                already_ok.append(name)
            elif nb == O * ((I + 1) // 2):
                plan.setdefault(f, []).append((name, O, I))
            else:
                print(f"  ?? {name}: unexpected blob size {nb} for shape {O}x{I}, skipping")

    n_fix = sum(len(v) for v in plan.values())
    print(f"MTP layer {L}: {n_fix} dense tensor(s) at per-row int4 to repair, "
          f"{len(already_ok)} already int8, {skipped_experts} routed-expert tensors left as-is")
    if not n_fix:
        print("nothing to do."); return
    if a.dry_run:
        for f, names in plan.items():
            for nm, O, I in names:
                print(f"  would repair {nm} [{O},{I}] in {os.path.basename(f)}")
        return

    for f, names in plan.items():
        print(f"patching {os.path.basename(f)} ({len(names)} tensors)")
        tensors, meta = read_shard(f)
        for nm, O, I in names:
            print(f"  {nm}: fetching source + requantizing at int8...", flush=True)
            w = src.fetch_f32(nm)
            assert w.shape == (O, I), f"{nm}: source shape {w.shape} != container {O}x{I}"
            q, s = quant_int8(w)
            tensors[nm] = ("U8", [len(q)], q.tobytes())
            tensors[nm + ".qs"] = ("F32", [O], s.tobytes())
            # verify first row against the source
            loc = q[:I].view(np.int8).astype(np.float64) * s[0]
            ref = w[0].astype(np.float64)
            cos = float(ref @ loc / (np.linalg.norm(ref) * np.linalg.norm(loc) + 1e-30))
            print(f"    row-0 cosine vs source: {cos:.5f}")
        bak = f + ".bak-int4"
        write_shard(f + ".new", tensors, meta)
        os.replace(f, bak)
        os.replace(f + ".new", f)
        print(f"  saved; original kept as {os.path.basename(bak)}")
    print("done. Re-run with --dry-run to confirm (should report 'already int8').")


if __name__ == "__main__":
    main()
