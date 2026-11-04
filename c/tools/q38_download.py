#!/usr/bin/env python3
"""Download Qwen3.8-Flash-Next-FP8 from Hugging Face and lay it out over several drives.

qwen38 streams two things from the disks while it runs: the routed experts each token
picks (10 of 512 in each of the 48 MoE layers and in the MTP layer that drafts, 4.9 MB
of block FP8 each) and sixteen 160-byte rows of the 51B-parameter PLE n-gram table.
The published checkpoint keeps a layer's 512 experts in two shards (gate and up in one,
down in the next) and the PLE table in 128 tensors of 400 MB, four to a shard, so a
shard placed whole puts all of a layer's reads on one drive while the others wait.
This script places the checkpoint so that every layer reads from every drive at once:

  * every layer's experts are dealt to the drives in id ranges in proportion to the
    drives' read speeds (--drive PATH=WEIGHT): with weights 10.4 and 6.55, experts
    0-313 of every layer on the first drive and 314-511 on the second. An expert's
    gate, up and down always share a drive.
  * the PLE table's 128 parts the same way (0-78 and 79-127 with those weights).
  * everything else -- the resident BF16 weights, the experts' block scales (read once
    at load), config, tokenizer -- stays on the first drive.
  * a shard whose tensors all belong to one drive goes there whole. A shard holding
    tensors for more than one drive is cut: the first drive keeps a file of the
    shard's own name with its tensors, every other drive gets `<shard>.pN.safetensors`
    with its own. No tensor is split or converted, and every part starts its streamed
    tensors on a 4096-byte boundary (the experts' matrices are multiples of 4096 bytes,
    so each one is a whole aligned read). The engine indexes every *.safetensors of
    the folders it is given (the first drive's folder, then the others) and finds
    every tensor by its name (qwen38: the first folder as the model, the others in
    Q38_DIRS).
  * every byte is checked: a shard's SHA-256 is computed over its published bytes
    while it is cut (a mismatch fetches it again), `--verify` recomputes it from the
    parts and the published header kept in `.q38split/`, and `--join` puts the
    published shards back together byte for byte (for other tools) and checks them
    again.

A rerun continues where the last one stopped: downloads resume, finished files are not
fetched again, a cut that was interrupted is redone from the whole shard, and a
published shard already in one of the folders (an earlier `hf download`, a --join) is
laid out without fetching it again. `q38_layout.json` in the first folder records the
revision, the drives and the place of every file.

  python -I tools/q38_download.py --drive "C:\\AI\\Qwen3.8-Flash-Next-FP8=10.4" --drive "D:\\AI\\Models\\Qwen3.8-Flash-Next-FP8=6.55"

The weights are relative read speeds (GB/s of 17.5 MB unbuffered reads, e.g. from
`iobench <file> 16.74 400 3 1`); without one a drive weighs 1. The plan needs every
shard's header first: two ranged requests a shard, before anything big is fetched.
Standard library only; with huggingface_hub and hf_xet installed the transfers go
through them (--backend hf, the default when they import), else through
setup_download.py's resumable HTTP.
"""
import argparse
import http.client
import json
import os
import re
import shutil
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import setup_download as sd  # noqa: E402

REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
LAYOUT = "q38_layout.json"
SIDE = ".q38split"             # the published shards' headers, in the first folder
CHUNK = 64 << 20
ALIGN = 4096
SHARD_RE = re.compile(r"^model-\d+-of-\d+\.safetensors$")
EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.(?:gate|up|down)_proj\.weight$")
PLE_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")
MARK = "q38split.part"         # __metadata__ keys every part of a cut shard carries
SOURCE = "q38split.source"


class Fail(RuntimeError):
    pass


def say(*parts):
    print("[q38-download]", *parts, flush=True)


def gb(n):
    return f"{n / 1e9:,.1f} GB"


def retried(fn, *args):
    """fn(*args), retried for a few seconds on Windows' sharing violations (a scanner
    that still holds a file just written)."""
    for attempt in range(20):
        try:
            return fn(*args)
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    retried(os.replace, path + ".tmp", path)


# ------------------------------------------------------------------ the layout file


class Layout:
    """q38_layout.json: written whole through a temporary file, under a lock (the
    download threads update it)."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                self.data = json.load(handle)
        self.data.setdefault("files", {})

    def file(self, name):
        with self.lock:
            entry = self.data["files"].get(name)
            return dict(entry) if entry else None

    def set_file(self, name, entry):
        with self.lock:
            self.data["files"][name] = entry
            self._write()

    def save(self):
        with self.lock:
            self._write()

    def _write(self):
        text = json.dumps(self.data, indent=1, sort_keys=True) + "\n"
        write_atomic(self.path, text.encode("utf-8"))


# ------------------------------------------------------------------ safetensors pieces


def parse_header(head, what):
    """(the header without __metadata__, the metadata or None) of 8 + n header bytes"""
    if len(head) < 8:
        raise Fail(f"{what}: shorter than a safetensors header")
    (n,) = struct.unpack("<Q", head[:8])
    if len(head) != 8 + n:
        raise Fail(f"{what}: truncated header")
    try:
        header = json.loads(head[8:].decode("utf-8"))
    except ValueError as error:
        raise Fail(f"{what}: the header is not JSON ({error})")
    if not isinstance(header, dict):
        raise Fail(f"{what}: the header is not a JSON object")
    return header, header.pop("__metadata__", None)


def read_header(path):
    """(the 8 + n header bytes, the header without __metadata__, the metadata or None)"""
    with open(path, "rb") as handle:
        head8 = handle.read(8)
        if len(head8) != 8:
            raise Fail(f"{path}: shorter than a safetensors header")
        (n,) = struct.unpack("<Q", head8)
        if n > (64 << 20):
            raise Fail(f"{path}: header of {n} bytes")
        head = head8 + handle.read(n)
    header, meta = parse_header(head, path)
    return head, header, meta


def by_offset(header):
    return sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])


def data_extent(header, what):
    """The end of the tensors' data; they must cover it without a gap or an overlap."""
    at = 0
    for tname, spec in by_offset(header):
        a, b = spec["data_offsets"]
        if a != at or b < a:
            raise Fail(f"{what}: tensor {tname} at [{a}, {b}), the one before ended at {at}")
        at = b
    return at


def header_bytes(header, meta, lead=None):
    """8 + n bytes of a safetensors header, the JSON padded with spaces to a multiple of
    8. With `lead`, the bytes of the part's data before its first streamed tensor, the
    padding also puts that tensor on a 4096-byte boundary when the multiple of 8 allows
    it (`lead` a multiple of 8)."""
    body = {"__metadata__": meta, **header} if meta else dict(header)
    text = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    n = (len(text) + 7) // 8 * 8
    if lead is not None and lead % 8 == 0:
        n += -(8 + n + lead) % ALIGN
    return struct.pack("<Q", n) + text + b" " * (n - len(text))


def part_name(shard, d):
    return shard if d == 0 else shard[:-len(".safetensors")] + f".p{d + 1}.safetensors"


# ------------------------------------------------------------------ where every tensor goes


def bounds(n, weights):
    """Drive d holds items [b[d], b[d + 1]) of n (expert ids, PLE parts), the counts in
    proportion to the weights."""
    total, acc, out = float(sum(weights)), 0.0, [0]
    for w in weights[:-1]:
        acc += w
        out.append(min(n, max(out[-1], int(round(n * acc / total)))))
    out.append(n)
    return out


def streamed(tensor):
    """('expert', id) or ('ple', part) for the tensors read while the model runs, else
    None (read once at load: they stay on the first drive)."""
    match = EXPERT_RE.search(tensor)
    if match:
        return "expert", int(match.group(1))
    match = PLE_RE.search(tensor)
    if match:
        return "ple", int(match.group(1))
    return None


class Placement:
    """The drive of every tensor: the experts' and the PLE parts' id ranges."""

    def __init__(self, weights, headers):
        top = {"expert": -1, "ple": -1}
        self.layers = set()                       # the MoE layers, e.g. "mtp.layers.0.mlp"
        for header in headers:
            for tname in header:
                kind = streamed(tname)
                if kind:
                    top[kind[0]] = max(top[kind[0]], kind[1])
                    if kind[0] == "expert":
                        self.layers.add(tname[:tname.index(".experts.")])
        self.drives = len(weights)
        self.count = {k: v + 1 for k, v in top.items()}
        self.ranges = {k: bounds(n, weights) for k, n in self.count.items()}

    def drive(self, tensor):
        kind = streamed(tensor)
        if not kind:
            return 0
        b = self.ranges[kind[0]]
        return next(d for d in range(len(b) - 1) if b[d] <= kind[1] < b[d + 1])

    def describe(self, d):
        out = []
        for kind, label in (("expert", "experts"), ("ple", "PLE parts")):
            lo, hi = self.ranges[kind][d], self.ranges[kind][d + 1]
            if hi > lo:
                out.append(f"{label} {lo}-{hi - 1}")
        return ", ".join(out) or "no streamed tensors"


def plan_parts(name, header, meta, place):
    """One shard's files: [{drive, file, names, head, size}], one per drive holding any
    of its tensors, each part's tensors in the published order. A single entry is a
    shard placed whole: its file is the published one."""
    items = by_offset(header)
    names = {}
    for tname, _spec in items:
        names.setdefault(place.drive(tname), []).append(tname)
    if len(names) <= 1:
        d = next(iter(names), 0)
        return [{"drive": d, "file": name, "names": [t for t, _ in items], "head": None, "size": None}]
    parts = []
    for d in sorted(names):
        rebased, at, lead = {}, 0, None
        for tname in names[d]:
            a, b = header[tname]["data_offsets"]
            if lead is None and streamed(tname):
                lead = at
            rebased[tname] = {**header[tname], "data_offsets": [at, at + b - a]}
            at += b - a
        pmeta = dict(meta or {})
        pmeta.update({MARK: f"{d + 1}/{place.drives}", SOURCE: name})
        head = header_bytes(rebased, pmeta, lead)
        parts.append({"drive": d, "file": part_name(name, d), "names": names[d], "head": head,
                      "size": len(head) + at})
    return parts


def is_part(path, name=None):
    """A part of a cut shard (of `name`, if given)?"""
    try:
        _, _, meta = read_header(path)
    except (Fail, OSError, ValueError):
        return False
    return bool(meta and MARK in meta and (name is None or meta.get(SOURCE) == name))


# ------------------------------------------------------------------ the downloader


class Downloader:
    def __init__(self, args):
        self.args = args
        self.repo = args.repo
        self.drives = [d for d, _ in args.drive]
        self.weights = [w for _, w in args.drive]
        self.first = self.drives[0]
        for path in self.drives:
            os.makedirs(path, exist_ok=True)
        self.layout = Layout(os.path.join(self.first, LAYOUT))
        self.token = sd.hf_token()
        self.backend = args.backend
        if self.backend == "auto":
            try:
                import huggingface_hub  # noqa: F401
                import hf_xet  # noqa: F401
                self.backend = "hf"
            except ImportError:
                self.backend = "http"
        self.done_bytes = 0
        self.total_bytes = 0
        self.active = {}
        self.errors = []
        self.mx = threading.Lock()

    # ---- the revision

    def revision(self):
        known = self.layout.data.get("revision")
        rev = self.args.revision or known or "main"
        if not re.fullmatch(r"[0-9a-f]{40}", rev):
            rev = self.resolve(rev)
        if known and rev != known:
            if not self.args.revision:
                raise Fail(f"{LAYOUT} records revision {known}; pass --revision to change it")
            if any(f.get("status") == "done" for f in self.layout.data["files"].values()):
                raise Fail(f"{self.first} holds files of revision {known}; another revision needs "
                           "fresh folders")
        return rev

    def resolve(self, rev):
        url = f"{sd.endpoint()}/api/models/{self.repo}/revision/{rev}"
        try:
            with urllib.request.urlopen(sd._request(url, self.token), timeout=60) as response:
                sha = json.loads(response.read().decode("utf-8")).get("sha")
        except Exception as error:  # noqa: BLE001 -- e.g. a hub without the revision API
            say(f"cannot resolve {rev} to a commit ({error}); using it as given")
            return rev
        return sha if sha and re.fullmatch(r"[0-9a-f]{40}", sha) else rev

    # ---- the published headers

    def ranged(self, url, lo, hi):
        """Bytes [lo, hi) of a file on the hub."""
        error = None
        for attempt in range(6):
            try:
                request = sd._request(url, self.token, {"Range": f"bytes={lo}-{hi - 1}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status == 206:
                        data = response.read(hi - lo)
                    else:                     # the whole file: what is needed, the rest dropped
                        data = response.read(hi)[lo:]
                if len(data) == hi - lo:
                    return data
                error = f"{len(data)} bytes instead of {hi - lo}"
            except urllib.error.HTTPError as err:
                err.close()
                if err.code in (401, 403, 404):
                    raise Fail(f"{url}: HTTP {err.code}")
                error = f"HTTP {err.code}"
            except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
                error = err
            time.sleep(min(60, 5 * (attempt + 1)))
        raise Fail(f"{url}: {error}")

    def find_whole(self, name, spec, skip=None):
        """A published copy of the file in one of the folders (not a part of a cut)."""
        for d, folder in enumerate(self.drives):
            path = os.path.join(folder, name)
            if d != skip and os.path.exists(path) and os.path.getsize(path) == spec["size"] \
                    and not is_part(path):
                return path
        return None

    def published_header(self, name, spec):
        """A shard's published header, 8 + n bytes, kept in .q38split/: read from a whole
        copy in one of the folders if there is one, else fetched with ranged requests."""
        side = os.path.join(self.first, SIDE, name + ".hdr")
        head = None
        if os.path.exists(side):
            with open(side, "rb") as handle:
                head = handle.read()
        else:
            whole = self.find_whole(name, spec)
            if whole:
                head, _, _ = read_header(whole)
            else:
                url = sd.file_url(self.repo, self.rev, name)
                head = self.ranged(url, 0, min(spec["size"], 1 << 20))
                (n,) = struct.unpack("<Q", head[:8])
                if 8 + n > spec["size"] or n > (64 << 20):
                    raise Fail(f"{name}: a header of {n} bytes in a file of {spec['size']}")
                if len(head) < 8 + n:
                    head += self.ranged(url, len(head), 8 + n)
                head = head[:8 + n]
        header, meta = parse_header(head, name)
        if len(head) + data_extent(header, name) != spec["size"]:
            raise Fail(f"{name}: the header covers {len(head) + data_extent(header, name)} bytes, "
                       f"the file has {spec['size']}")
        if not os.path.exists(side):
            write_atomic(side, head)
        return head, header, meta

    # ---- one file placed whole

    def fetch(self, name, spec, folder):
        """The published file in `folder`: (path, whether its hash was checked)."""
        dest = sd.safe_join(folder, name)
        if os.path.exists(dest) and os.path.getsize(dest) == spec["size"]:
            return dest, False
        with self.mx:
            self.active[name] = folder
        try:
            if self.backend == "hf":
                from huggingface_hub import hf_hub_download
                for attempt in range(8):
                    try:
                        hf_hub_download(self.repo, name, revision=self.rev, local_dir=folder)
                        break
                    except Exception as error:  # noqa: BLE001 -- network errors of every kind
                        if attempt == 7:
                            raise Fail(f"{name}: {error}")
                        say(f"{name}: {error}; retrying")
                        time.sleep(min(120, 10 * (attempt + 1)))
                size = os.path.getsize(dest) if os.path.exists(dest) else -1
                if size != spec["size"]:
                    raise Fail(f"{name}: {size} bytes, expected {spec['size']}")
                return dest, False
            url = sd.file_url(self.repo, self.rev, name)
            sd.download_file(url, dest, spec, token=self.token, verify=True, retries=8)
            return dest, True
        finally:
            with self.mx:
                self.active.pop(name, None)

    @staticmethod
    def hash_ok(path, spec):
        hasher, expected = sd._hasher(spec)
        if hasher is None:
            return True
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                hasher.update(block)
        return hasher.hexdigest() == expected

    def whole_file(self, name, spec, d, kind):
        folder = self.drives[d]
        dest = sd.safe_join(folder, name)
        entry = self.layout.file(name) or {}
        done = (entry.get("status") == "done" and entry.get("drive", 0) == d and not entry.get("parts")
                and os.path.exists(dest) and os.path.getsize(dest) == spec["size"])
        if not done:
            if os.path.exists(dest) and (os.path.getsize(dest) != spec["size"] or is_part(dest)):
                retried(os.remove, dest)              # a part of an earlier cut of this shard
            if not os.path.exists(dest):
                other = self.find_whole(name, spec, skip=d)
                if other:
                    say(f"{name}: moving it from {os.path.dirname(other)} to {folder}")
                    shutil.move(other, dest)
            for _attempt in range(2):
                path, checked = self.fetch(name, spec, folder)
                if checked or self.args.no_verify or self.hash_ok(path, spec):
                    break
                say(f"{name}: checksum mismatch, fetching it again")
                retried(os.remove, path)
            else:
                raise Fail(f"{name}: checksum mismatch twice")
            if SHARD_RE.match(name):
                self.clear_cut(name)                  # parts an earlier layout left
            self.layout.set_file(name, {"size": spec["size"], "sha256": spec.get("sha256"),
                                        "git_oid": spec.get("git_oid"), "kind": kind, "drive": d,
                                        "status": "done"})
        with self.mx:
            self.done_bytes += spec["size"]

    # ---- a cut shard

    def part_path(self, part):
        return os.path.join(self.drives[part["drive"]], part["file"])

    def part_ok(self, part, name, header):
        """The part on disk holds exactly its planned tensors, each as long as published."""
        path = self.part_path(part)
        try:
            if not os.path.exists(path):
                return False
            head, h, meta = read_header(path)
            if not meta or meta.get(SOURCE) != name or set(h) != set(part["names"]):
                return False
            for tname, spec in h.items():
                pub = header[tname]
                if (spec["dtype"], spec["shape"]) != (pub["dtype"], pub["shape"]) or \
                        spec["data_offsets"][1] - spec["data_offsets"][0] != \
                        pub["data_offsets"][1] - pub["data_offsets"][0]:
                    return False
            return os.path.getsize(path) == len(head) + data_extent(h, path)
        except (Fail, OSError, ValueError, KeyError, TypeError):
            return False

    def clear_cut(self, name):
        """Every trace of a cut of this shard: its parts and their temporary files (a whole
        published copy under the shard's own name stays)."""
        for k in range(len(self.drives)):
            path = os.path.join(self.drives[k], part_name(name, k))
            for p in (path + ".tmp", path):
                if os.path.exists(p) and (p != path or k > 0 or is_part(p, name)):
                    retried(os.remove, p)

    def cut_shard(self, name, spec, plan):
        _head, header, _meta, parts = plan
        entry = self.layout.file(name) or {}
        if all(self.part_ok(p, name, header) for p in parts):
            if entry.get("status") != "done":
                self.record_cut(name, spec, parts, "done")
        else:
            src = os.path.join(self.first, name)
            if not (os.path.exists(src) and os.path.getsize(src) == spec["size"] and not is_part(src)):
                self.clear_cut(name)
                if os.path.exists(src):               # neither a part nor the published file
                    retried(os.remove, src)
                other = self.find_whole(name, spec, skip=0)
                if other:
                    say(f"{name}: moving it from {os.path.dirname(other)} to {self.first}")
                    shutil.move(other, src)
            for _attempt in range(2):
                self.fetch(name, spec, self.first)
                if self.cut(name, spec, parts):
                    break
                say(f"{name}: checksum mismatch, fetching it again")
                retried(os.remove, src)
            else:
                raise Fail(f"{name}: checksum mismatch twice")
        with self.mx:
            self.done_bytes += spec["size"]

    def record_cut(self, name, spec, parts, status):
        self.layout.set_file(name, {"size": spec["size"], "sha256": spec.get("sha256"),
                                    "git_oid": spec.get("git_oid"), "kind": "cut", "status": status,
                                    "parts": [{"drive": p["drive"], "file": p["file"], "size": p["size"],
                                               "tensors": len(p["names"])} for p in parts]})

    def cut(self, name, spec, parts):
        """One pass over the published shard on the first drive: hash every byte and write
        every drive's part; then the first drive's part takes the shard's place. False on
        a checksum mismatch (the parts written so far are removed then)."""
        src = os.path.join(self.first, name)
        head, header, _meta = read_header(src)
        data_end = data_extent(header, src)
        if len(head) + data_end != spec["size"]:
            raise Fail(f"{name}: the header covers {len(head) + data_end} bytes, the file has {spec['size']}")
        owner = {t: i for i, p in enumerate(parts) for t in p["names"]}
        if set(owner) != set(header):
            raise Fail(f"{name}: the shard on disk holds other tensors than its published header")
        runs = []                                 # [lo, hi, part]: tensors of one part back to back
        for tname, s in by_offset(header):
            a, b = s["data_offsets"]
            if runs and runs[-1][2] == owner[tname] and runs[-1][1] == a:
                runs[-1][1] = b
            else:
                runs.append([a, b, owner[tname]])
        hasher, expected = (None, None) if self.args.no_verify else sd._hasher(spec)
        if hasher:
            hasher.update(head)
        tmps = [self.part_path(p) + ".tmp" for p in parts]
        outs = []
        try:
            for p, tmp in zip(parts, tmps):
                outs.append(open(tmp, "wb"))
                outs[-1].write(p["head"])
            with open(src, "rb") as handle:
                handle.seek(len(head))
                at, r = 0, 0
                while at < data_end:
                    block = handle.read(min(CHUNK, data_end - at))
                    if not block:
                        raise Fail(f"{name}: ends at data byte {at} of {data_end}")
                    if hasher:
                        hasher.update(block)
                    view, end = memoryview(block), at + len(block)
                    while r < len(runs) and runs[r][0] < end:
                        lo, hi, i = runs[r]
                        a, b = max(lo, at), min(hi, end)
                        if a < b:
                            outs[i].write(view[a - at:b - at])
                        if hi > end:
                            break
                        r += 1
                    at = end
            ok = not hasher or hasher.hexdigest() == expected
            for out in outs:
                if ok:
                    out.flush()
                    os.fsync(out.fileno())
                out.close()
        except BaseException:
            for out in outs:
                out.close()
            raise
        if not ok:
            for tmp in tmps:
                retried(os.remove, tmp)
            return False
        for p, tmp in zip(parts, tmps):
            if os.path.getsize(tmp) != p["size"]:
                raise Fail(f"{tmp}: {os.path.getsize(tmp)} bytes written, {p['size']} planned")
        side = os.path.join(self.first, SIDE, name + ".hdr")
        kept = None
        if os.path.exists(side):
            with open(side, "rb") as handle:
                kept = handle.read()
        if kept != head:                          # the bytes just hashed are the published ones
            write_atomic(side, head)
        for p, tmp in zip(parts, tmps):
            if p["drive"] != 0:
                retried(os.replace, tmp, self.part_path(p))
        self.record_cut(name, spec, parts, "cutting")
        if parts[0]["drive"] == 0:                # the first drive's part takes the shard's place
            retried(os.replace, tmps[0], src)
        else:
            retried(os.remove, src)
        self.record_cut(name, spec, parts, "done")
        return True

    # ---- the whole run

    def run(self):
        args = self.args
        known = self.layout.data.get("drives")
        if known and self.layout.data.get("placed") and \
                [(k["path"], k["weight"]) for k in known] != list(args.drive):
            raise Fail(f"{LAYOUT} was laid out for " +
                       ", ".join(f"{k['path']}={k['weight']:g}" for k in known) +
                       "; give the same --drive folders and weights, or --join with those first")
        self.rev = self.revision()
        say(f"{self.repo} at {self.rev}, backend {self.backend}, {len(self.drives)} drive(s)")
        if self.backend == "hf":
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
            if args.high_performance:
                os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        files = {f["path"]: f for f in sd.list_repo_files(self.repo, self.rev, token=self.token)}
        self.layout.data.update({"format": 1, "repo": self.repo, "revision": self.rev,
                                 "drives": [{"path": p, "weight": w} for p, w in args.drive]})
        self.layout.save()
        meta = sorted(k for k in files if not SHARD_RE.match(k))
        shards = sorted(k for k in files if SHARD_RE.match(k))
        if not shards:
            raise Fail(f"{self.repo} has no model-*-of-*.safetensors shards")
        for name in meta:                         # the small files first, on the first drive
            self.whole_file(name, files[name], 0, "meta")
        say(f"reading the headers of {len(shards)} shards")
        with ThreadPoolExecutor(max_workers=8) as pool:
            heads = dict(zip(shards, pool.map(lambda s: self.published_header(s, files[s]), shards)))
        place = Placement(self.weights, [h for _, h, _ in heads.values()])
        plans = {s: (*heads[s], plan_parts(s, heads[s][1], heads[s][2], place)) for s in shards}
        self.total_bytes = sum(f["size"] for f in files.values())
        self.print_plan(files, shards, plans, place)
        self.check_space(files, shards, plans)
        if args.dry_run:
            say("dry run: nothing more fetched")
            return 0
        self.layout.data["placed"] = len(self.drives) > 1
        self.layout.save()
        with self.mx:
            self.done_bytes = sum(files[k]["size"] for k in meta)
        stop = threading.Event()
        reporter = threading.Thread(target=self.report, args=(stop,), daemon=True)
        reporter.start()
        pool = ThreadPoolExecutor(max_workers=max(1, args.jobs))
        try:
            jobs = []
            for name in shards:
                parts = plans[name][3]
                if len(parts) > 1:
                    jobs.append(pool.submit(self.guard, self.cut_shard, name, files[name], plans[name]))
                else:
                    jobs.append(pool.submit(self.guard, self.whole_file, name, files[name],
                                            parts[0]["drive"], "whole"))
            pending = set(jobs)
            while pending:                        # short waits: Ctrl+C reaches the main thread
                _done, pending = wait(pending, timeout=1.0, return_when=FIRST_EXCEPTION)
        except KeyboardInterrupt:
            say("interrupted; the same command continues from here")
            os._exit(130)                         # the transfers in flight resume on the next run
        finally:
            stop.set()
        pool.shutdown(wait=True)
        if self.errors:
            for error in self.errors:
                say("FAILED:", error)
            say(f"{len(self.errors)} file(s) failed; the same command continues from here")
            return 1
        self.layout.data["complete"] = True
        self.layout.save()
        self.summary(files)
        return 0

    def guard(self, fn, *a):
        try:
            fn(*a)
        except Exception as error:  # noqa: BLE001 -- one file's failure must not stop the others
            with self.mx:
                self.errors.append(f"{a[0]}: {error}")

    def print_plan(self, files, shards, plans, place):
        cut = [s for s in shards if len(plans[s][3]) > 1]
        whole = [plans[s][3][0]["drive"] for s in shards if len(plans[s][3]) == 1]
        holding = lambda rx: sum(1 for s in shards if any(rx.search(t) for t in plans[s][1]))  # noqa: E731
        say(f"{len(shards)} shards ({gb(sum(files[s]['size'] for s in shards))}) and "
            f"{len(files) - len(shards)} small files")
        say(f"  routed experts: {len(place.layers)} layers x {place.count['expert']} in "
            f"{holding(EXPERT_RE)} shards; PLE table: {place.count['ple']} parts in {holding(PLE_RE)} shards")
        per = self.needs(files, shards, plans, only_left=False)
        total = float(sum(self.weights))
        for d, path in enumerate(self.drives):
            say(f"  drive {d + 1} {path} (weight {self.weights[d]:g}, {100 * self.weights[d] / total:.1f}%): "
                f"{gb(per[d])} -- {place.describe(d)}" + (", the rest" if d == 0 else ""))
        if len(self.drives) > 1:
            say(f"  {len(cut)} shards cut, {len(whole)} placed whole (" +
                ", ".join(f"{whole.count(d)} on drive {d + 1}" for d in range(len(self.drives))) + ")")

    def needs(self, files, shards, plans, only_left):
        """Bytes each drive will hold (only_left: not counting the files already in place)."""
        per = [0.0] * len(self.drives)
        for name, spec in files.items():
            entry = self.layout.file(name) or {}
            if only_left and entry.get("status") == "done":
                continue
            parts = plans[name][3] if name in plans else [{"drive": 0, "size": None}]
            for p in parts:
                per[p["drive"]] += p["size"] if p["size"] is not None else spec["size"]
        return per

    def check_space(self, files, shards, plans):
        need = self.needs(files, shards, plans, only_left=True)
        left = [files[s]["size"] for s in shards if len(plans[s][3]) > 1
                and (self.layout.file(s) or {}).get("status") != "done"]
        need[0] += max(left, default=0) * max(1, self.args.jobs)   # whole shards waiting to be cut
        bad = []
        for d, path in enumerate(self.drives):
            free = shutil.disk_usage(path).free
            want = need[d] + self.args.reserve_gb * 1e9
            say(f"drive {d + 1} {path}: {gb(need[d])} still to write, {gb(free)} free "
                f"({self.args.reserve_gb:g} GB kept free)")
            if want > free:
                bad.append(f"{path} needs {gb(want)}, has {gb(free)}")
        if bad:
            raise Fail("not enough space: " + "; ".join(bad))

    def partial_bytes(self):
        n = 0
        with self.mx:
            folders = dict(self.active)
        for name, folder in folders.items():
            part = sd.safe_join(folder, name) + ".part"
            if os.path.exists(part):
                n += os.path.getsize(part)
        if self.backend == "hf":
            for folder in set(folders.values()):
                cache = os.path.join(folder, ".cache", "huggingface", "download")
                for root, _dirs, names in os.walk(cache):
                    for fname in names:
                        if fname.endswith(".incomplete"):
                            try:
                                n += os.path.getsize(os.path.join(root, fname))
                            except OSError:
                                pass
        return n

    def report(self, stop):
        last_t, last_b = time.time(), None
        while not stop.wait(self.args.report_s):
            done = min(self.total_bytes, self.done_bytes + self.partial_bytes())
            now = time.time()
            rate = (done - last_b) / (now - last_t) if last_b is not None and now > last_t else 0.0
            last_t, last_b = now, done
            eta = f", about {(self.total_bytes - done) / rate / 3600:.1f} h left" if rate > 1e6 else ""
            with self.mx:
                names = ", ".join(sorted(self.active)[:3])
            say(f"{gb(done)} of {gb(self.total_bytes)} ({100 * done / max(1, self.total_bytes):.1f}%), "
                f"{rate / 1e6:,.0f} MB/s{eta}" + (f" | {names}" if names else ""))

    def summary(self, files):
        per = [0] * len(self.drives)
        for name in files:
            entry = self.layout.file(name) or {}
            for p in entry.get("parts") or [{"drive": entry.get("drive", 0), "size": entry.get("size", 0)}]:
                per[p["drive"]] += p["size"]
        say("complete:", ", ".join(f"{p} {gb(b)}" for p, b in zip(self.drives, per)))
        if len(self.drives) > 1:
            say(f"the engine's model folder is {self.first}, with")
            say(f"  Q38_DIRS={';'.join(self.drives[1:])}")


# ------------------------------------------------------------------ verify, join, status


def layout_or_fail(first):
    path = os.path.join(first, LAYOUT)
    if not os.path.exists(path):
        raise Fail(f"no {LAYOUT} in {first}")
    return Layout(path)


def published_bytes(drives, name):
    """A cut shard's published bytes in order: its published header, then every tensor's
    bytes from the part that holds it."""
    with open(os.path.join(drives[0], SIDE, name + ".hdr"), "rb") as handle:
        head = handle.read()
    header, _ = parse_header(head, name)
    where = {}
    for d in range(len(drives)):
        path = os.path.join(drives[d], part_name(name, d))
        if os.path.exists(path) and is_part(path, name):
            phead, h, _ = read_header(path)
            for tname, spec in h.items():
                where[tname] = (path, len(phead) + spec["data_offsets"][0])
    runs = []                                     # [path, offset, bytes]
    for tname, spec in by_offset(header):
        if tname not in where:
            raise Fail(f"{name}: {tname} is in none of its parts")
        path, off = where[tname]
        n = spec["data_offsets"][1] - spec["data_offsets"][0]
        if runs and runs[-1][0] == path and runs[-1][1] + runs[-1][2] == off:
            runs[-1][2] += n
        else:
            runs.append([path, off, n])
    yield head
    handles = {}
    try:
        for path, off, left in runs:
            if path not in handles:
                handles[path] = open(path, "rb")
            handle = handles[path]
            handle.seek(off)
            while left > 0:
                block = handle.read(min(CHUNK, left))
                if not block:
                    raise Fail(f"{path}: ends early")
                left -= len(block)
                yield block
    finally:
        for handle in handles.values():
            handle.close()


def verify(args):
    drives = [d for d, _ in args.drive]
    files = layout_or_fail(drives[0]).data["files"]
    bad = 0
    for name in sorted(files):
        entry = files[name]
        if entry.get("status") != "done":
            say(f"{name}: not finished")
            bad += 1
            continue
        hasher, expected = sd._hasher({"size": entry.get("size"), "sha256": entry.get("sha256"),
                                       "git_oid": entry.get("git_oid")})
        if hasher is None:
            continue
        try:
            if entry.get("parts"):
                for block in published_bytes(drives, name):
                    hasher.update(block)
            else:
                with open(os.path.join(drives[entry.get("drive", 0)], *name.split("/")), "rb") as handle:
                    for block in iter(lambda: handle.read(CHUNK), b""):
                        hasher.update(block)
        except (OSError, Fail, ValueError) as error:
            say(f"{name}: {error}")
            bad += 1
            continue
        ok = hasher.hexdigest() == expected
        bad += not ok
        if not ok or args.verbose:
            say(f"{name}: {'ok' if ok else 'CHECKSUM MISMATCH'}")
    say(f"verify: {len(files) - bad} of {len(files)} files are the published bytes")
    return 1 if bad else 0


def join(args):
    """Every shard back as the published file on the first drive, checked."""
    drives = [d for d, _ in args.drive]
    layout = layout_or_fail(drives[0])
    bad = 0
    for name in sorted(layout.data["files"]):
        entry = layout.file(name)
        dest = os.path.join(drives[0], *name.split("/"))
        if entry.get("status") not in ("done", "joining"):
            say(f"{name}: not finished, left as it is")
            bad += 1
            continue
        if entry.get("parts"):
            if not (entry["status"] == "joining" and os.path.exists(dest)
                    and os.path.getsize(dest) == entry["size"] and not is_part(dest)):
                hasher, expected = sd._hasher({"size": entry["size"], "sha256": entry.get("sha256"),
                                               "git_oid": entry.get("git_oid")})
                tmp = dest + ".join.tmp"
                with open(tmp, "wb") as out:
                    for block in published_bytes(drives, name):
                        if hasher:
                            hasher.update(block)
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
                if (hasher and hasher.hexdigest() != expected) or os.path.getsize(tmp) != entry["size"]:
                    retried(os.remove, tmp)
                    say(f"{name}: its parts do not give the published file; they are kept")
                    bad += 1
                    continue
                layout.set_file(name, {**entry, "status": "joining"})
                retried(os.replace, tmp, dest)
            for d in range(1, len(drives)):
                path = os.path.join(drives[d], part_name(name, d))
                if os.path.exists(path) and is_part(path, name):
                    retried(os.remove, path)
            entry.pop("parts")
            layout.set_file(name, {**entry, "kind": "whole", "status": "done", "drive": 0})
            say(f"{name}: joined")
        elif entry.get("drive", 0) != 0:
            shutil.move(os.path.join(drives[entry["drive"]], *name.split("/")), dest)
            layout.set_file(name, {**entry, "drive": 0})
            say(f"{name}: moved to {drives[0]}")
    if not bad:
        layout.data["placed"] = False
        layout.save()
    say("join: every file is the published one on the first drive" if not bad
        else f"join: {bad} file(s) not joined")
    return 1 if bad else 0


def status(args):
    drives = [d for d, _ in args.drive]
    layout = layout_or_fail(drives[0])
    files = layout.data["files"]
    done = sum(1 for v in files.values() if v.get("status") == "done")
    say(f"{layout.data.get('repo')} at {layout.data.get('revision')}: {done} of {len(files)} files done, "
        f"complete={bool(layout.data.get('complete'))}")
    per = [0] * len(drives)
    for v in files.values():
        if v.get("status") == "done":
            for p in v.get("parts") or [{"drive": v.get("drive", 0), "size": v.get("size", 0)}]:
                if p["drive"] < len(per):
                    per[p["drive"]] += p["size"]
    for d, path in enumerate(drives):
        say(f"  {path}: {gb(per[d])}")
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", action="append", required=True, metavar="FOLDER[=WEIGHT]",
                    help="a folder on one drive and its relative read speed; the first holds the "
                         "metadata and is the engine's model folder")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", help="commit, branch or tag (default: main, pinned to its commit "
                                       "on the first run)")
    ap.add_argument("--jobs", type=int, default=3, help="files fetched at once (default 3)")
    ap.add_argument("--backend", choices=("auto", "hf", "http"), default="auto")
    ap.add_argument("--reserve-gb", type=float, default=20.0, help="free space to leave on every drive")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA-256 checks (not advised)")
    ap.add_argument("--high-performance", action=argparse.BooleanOptionalAction, default=True,
                    help="HF_XET_HIGH_PERFORMANCE for the hf backend")
    ap.add_argument("--report-s", type=float, default=30.0, help="seconds between progress lines")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch the small files and the shards' headers, plan, check the space, stop")
    ap.add_argument("--verify", action="store_true", help="re-hash everything against the published bytes")
    ap.add_argument("--join", action="store_true",
                    help="every file back as published on the first drive, checked")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    drives = []
    for text in args.drive:
        path, sep, weight = text.rpartition("=")
        try:
            w = float(weight) if sep else 1.0
        except ValueError:
            path, w = text, 1.0
        if not sep:
            path = text
        if not w > 0:
            ap.error(f"--drive {text}: the weight must be above 0")
        drives.append((os.path.abspath(path), w))
    if len({os.path.normcase(p) for p, _ in drives}) != len(drives):
        ap.error("the same folder twice")
    args.drive = drives
    try:
        if args.status:
            return status(args)
        if args.verify:
            return verify(args)
        if args.join:
            return join(args)
        return Downloader(args).run()
    except (Fail, sd.DownloadError) as error:
        say("error:", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
