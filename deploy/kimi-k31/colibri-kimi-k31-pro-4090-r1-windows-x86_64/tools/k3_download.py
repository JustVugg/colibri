#!/usr/bin/env python3
"""Download Kimi K3.1 Pro (or Kimi K3) from Hugging Face and lay it out over several drives.

moonshotai/Kimi-K3.1-Pro is Kimi K3 trained further: the same architecture, the same
tensors and the same one-layer-a-shard layout, so everything below holds for both.
--repo moonshotai/Kimi-K3 fetches the original. Each repository needs its own folders:
k3_layout.json records the repo, and a folder of another one is refused.

kimi_k3 reads each token's 16 routed experts of every MoE layer from the disks:
about 25.8 GB a token on the full model, so the drives' read speed is the decode
speed. The checkpoint keeps one layer per shard (model-000NN-of-000096 holds layer
NN-1), and a shard placed whole puts all of a layer's reads on one drive while the
others wait. This script places the checkpoint so that every layer reads from every
drive at once:

  * each MoE layer's shard is cut at an expert boundary into one part per drive,
    the parts' experts in proportion to the drives' read speeds (--drive PATH=WEIGHT).
    The first drive keeps the shard's own file: its header is rewritten in place to
    name only the tensors it still holds (padded to the old length, so no offset
    moves) and the rest is truncated away. Every other drive gets a
    `<shard>.pN.safetensors` with the next run of experts. An expert stays one
    contiguous read, and the engine indexes every *.safetensors of the folders it
    is given (the first drive's folder, then K3_DIRS), so it reads the parts as it
    read the shard. No tensor is converted.
  * what is read once at load time (config, tokenizer, layer 0, the embeddings and
    the head, the vision tower) stays on the first drive.
  * every byte is checked: a shard's SHA-256 is computed over its bytes in the
    published order while it is cut (a mismatch fetches it again), `--verify`
    recomputes it from the parts and the original header kept in `.k3split/`, and
    `--join` puts the published shards back together byte for byte (for another
    engine) and checks them again.

A rerun continues where the last one stopped: downloads resume, finished files are
not fetched again, and a cut that was interrupted is redone from what is on disk.
`k3_layout.json` in the first folder records the revision and where every part is
(and can be rebuilt from the parts' own headers if it is lost).

  python -I tools/k3_download.py --drive "C:\\AI\\Kimi-K3.1-Pro=10.4" --drive "D:\\AI\\Models\\Kimi-K3.1-Pro=6.55"

The weights are relative read speeds (GB/s of 17.5 MB unbuffered reads, e.g. from
`iobench <file> 16.74 400 3 1`); without one a drive weighs 1. Then run the engine
with the first folder as the model and the others in K3_DIRS (the script prints the
line). Standard library only; with huggingface_hub and hf_xet installed the
transfers go through them (--backend hf, the default when they import), else
through setup_download.py's resumable HTTP.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import setup_download as sd  # noqa: E402

REPO = "moonshotai/Kimi-K3.1-Pro"
LAYOUT = "k3_layout.json"
SIDE = ".k3split"              # the original headers of the cut shards, in the first folder
CHUNK = 64 << 20
SHARD_RE = re.compile(r"^model-\d+-of-\d+\.safetensors$")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
MARK = "k3split.part"          # __metadata__ key every part of a cut shard carries


class Fail(RuntimeError):
    pass


def say(*parts):
    print("[k3-download]", *parts, flush=True)


def gb(n):
    return f"{n / 1e9:,.1f} GB"


# ------------------------------------------------------------------ the layout file


class Layout:
    """k3_layout.json: written whole through a temporary file, under a lock (the
    download and cut threads update it)."""

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
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.data, handle, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)


# ------------------------------------------------------------------ safetensors pieces


def read_header(path):
    """(the 8 + n header bytes, the header without __metadata__, the metadata or None)"""
    with open(path, "rb") as handle:
        head8 = handle.read(8)
        if len(head8) != 8:
            raise Fail(f"{path}: shorter than a safetensors header")
        (n,) = struct.unpack("<Q", head8)
        if n > (64 << 20):
            raise Fail(f"{path}: header of {n} bytes")
        raw = handle.read(n)
    if len(raw) != n:
        raise Fail(f"{path}: truncated header")
    header = json.loads(raw.decode("utf-8"))
    meta = header.pop("__metadata__", None)
    return head8 + raw, header, meta


def header_bytes(header, meta, length=None):
    """8 + n bytes of a safetensors header, the JSON padded with spaces to `length`
    (the in-place rewrite) or to a multiple of 8."""
    body = {"__metadata__": meta, **header} if meta else dict(header)
    text = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if length is None:
        length = (len(text) + 7) // 8 * 8
    if len(text) > length:
        raise Fail(f"the rewritten header needs {len(text)} bytes, the shard has {length}")
    return struct.pack("<Q", length) + text + b" " * (length - len(text))


def data_extent(header):
    return max((s["data_offsets"][1] for s in header.values()), default=0)


def expert_ids(header):
    return len({t[:m.end()] for t in header for m in [EXPERT_RE.search(t)] if m})


def plan_cut(header, weights, name):
    """How a shard's data is cut: one {lo, hi, experts, tensors} per drive, its data
    range [lo, hi) relative to the end of the header. Cuts fall between experts only
    (an expert's six tensors stay one read), and the experts go to the drives in
    proportion to `weights`. The tensors that are not experts stay where they lie:
    in the published shards that is the head of the data, the first drive's part."""
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    at = 0
    for tname, spec in items:
        a, b = spec["data_offsets"]
        if a != at or b < a:
            raise Fail(f"{name}: tensor {tname} at [{a}, {b}), the one before ended at {at}")
        at = b
    data_end = at
    groups = []                               # [start, end) of each expert, in file order
    where = {}
    for tname, spec in items:
        match = EXPERT_RE.search(tname)
        if not match:
            continue
        key = tname[:match.end()]
        a, b = spec["data_offsets"]
        if key in where:
            if groups[where[key]][1] != a:
                raise Fail(f"{name}: the tensors of {key} are not stored in one run")
            groups[where[key]][1] = b
        else:
            where[key] = len(groups)
            groups.append([a, b])
    m = len(groups)
    total = float(sum(weights))
    bounds, acc = [0], 0.0
    for w in weights[:-1]:
        acc += w
        bounds.append(min(m, max(bounds[-1], int(round(m * acc / total)))))
    bounds.append(m)

    def start(j):
        return groups[j][0] if j < m else data_end

    parts = []
    last = len(weights) - 1
    for d in range(len(weights)):
        lo = 0 if d == 0 else start(bounds[d])
        hi = data_end if d == last else start(bounds[d + 1])
        tensors = {t: s for t, s in items
                   if lo <= s["data_offsets"][0] < hi or (d == last and s["data_offsets"][0] == data_end)}
        parts.append({"lo": lo, "hi": hi, "experts": bounds[d + 1] - bounds[d], "tensors": tensors})
    return parts, data_end


def part_name(shard, d):
    return shard[:-len(".safetensors")] + f".p{d + 1}.safetensors"


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
        self.split = not args.no_split and len(self.drives) > 1
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
        import urllib.request
        url = f"{sd.endpoint()}/api/models/{self.repo}/revision/{rev}"
        try:
            with urllib.request.urlopen(sd._request(url, self.token), timeout=60) as response:
                sha = json.loads(response.read().decode("utf-8")).get("sha")
        except Exception as error:  # noqa: BLE001 -- e.g. a hub without the revision API
            say(f"cannot resolve {rev} to a commit ({error}); using it as given")
            return rev
        return sha if sha and re.fullmatch(r"[0-9a-f]{40}", sha) else rev

    # ---- placement

    def drive_of(self, name, kind):
        """The drive a whole file goes to (every file but a cut MoE shard)."""
        if kind == "moe" and not self.split:
            return (self.layout.file(name) or {}).get("drive", 0)
        return 0

    def assign_whole(self, moe):
        """--no-split: every MoE shard on one drive, each drive's bytes in proportion to
        its weight, consecutive layers on different drives where the shares allow."""
        load = [0.0] * len(self.drives)
        for name, spec in moe:
            entry = self.layout.file(name) or {}
            if "drive" in entry:
                load[entry["drive"]] += spec["size"]
        prev = None
        for name, spec in moe:
            entry = self.layout.file(name) or {}
            if "drive" not in entry:
                d = min(range(len(self.drives)),
                        key=lambda k: ((load[k] + spec["size"]) / self.weights[k], k == prev))
                load[d] += spec["size"]
                entry = {**entry, "drive": d, "kind": "moe", "size": spec["size"],
                         "sha256": spec.get("sha256"), "status": "pending"}
                self.layout.set_file(name, entry)
            prev = entry["drive"]

    def needs(self, files, kinds, experts_b, only_left):
        """Bytes each drive will hold (only_left: not counting finished files)."""
        per = [0.0] * len(self.drives)
        total = float(sum(self.weights))
        for name, spec in files.items():
            kind = kinds.get(name, "meta")
            entry = self.layout.file(name) or {}
            if only_left and entry.get("status") == "done" and (
                    kind != "moe" or not self.split or entry.get("parts")):
                continue
            if kind == "moe" and self.split:
                eb = min(experts_b.get(name) or 0.92 * spec["size"], spec["size"])
                for d, w in enumerate(self.weights):
                    per[d] += eb * w / total
                per[0] += spec["size"] - eb
            else:
                per[self.drive_of(name, kind)] += spec["size"]
        return per

    def check_space(self, files, kinds, experts_b):
        need = self.needs(files, kinds, experts_b, only_left=True)
        if self.split:     # whole shards wait on the first drive to be cut
            biggest = max((s["size"] for n, s in files.items() if kinds.get(n) == "moe"), default=0)
            need[0] += biggest * max(1, self.args.jobs)
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

    # ---- one file

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
            while True:
                block = handle.read(CHUNK)
                if not block:
                    break
                hasher.update(block)
        return hasher.hexdigest() == expected

    def whole_file(self, name, spec, kind):
        d = self.drive_of(name, kind)
        folder = self.drives[d]
        entry = self.layout.file(name) or {}
        dest = sd.safe_join(folder, name)
        done = (entry.get("status") == "done" and os.path.exists(dest)
                and os.path.getsize(dest) == spec["size"])
        if not done:
            for _attempt in range(2):
                path, checked = self.fetch(name, spec, folder)
                if checked or self.args.no_verify or self.hash_ok(path, spec):
                    break
                say(f"{name}: checksum mismatch, fetching it again")
                os.remove(path)
            else:
                raise Fail(f"{name}: checksum mismatch twice")
            self.layout.set_file(name, {"size": spec["size"], "sha256": spec.get("sha256"),
                                        "git_oid": spec.get("git_oid"), "kind": kind, "drive": d,
                                        "status": "done"})
        with self.mx:
            self.done_bytes += spec["size"]

    # ---- a cut MoE shard

    def part_paths(self, name):
        return [os.path.join(self.drives[0], name)] + [
            os.path.join(self.drives[d], part_name(name, d)) for d in range(1, len(self.drives))]

    def moe_state(self, name, spec):
        """'done', 'cut' (every part written, the first one maybe not rewritten yet),
        'whole' (the published file is on the first drive) or 'absent'."""
        entry = self.layout.file(name) or {}
        paths = self.part_paths(name)
        if not entry.get("parts") and self.adopt(name, spec):
            entry = self.layout.file(name)
        parts = entry.get("parts")
        if entry.get("status") in ("done", "cutting") and parts:
            others = all(os.path.exists(paths[p["drive"]]) and
                         os.path.getsize(paths[p["drive"]]) == p["size"]
                         for p in parts if p["drive"] > 0)
            if others and os.path.exists(paths[0]):
                size = os.path.getsize(paths[0])
                if entry["status"] == "done" and size == parts[0]["size"]:
                    return "done"
                if size in (parts[0]["size"], spec["size"]):
                    return "cut"
        if os.path.exists(paths[0]) and os.path.getsize(paths[0]) == spec["size"]:
            try:
                _, _, meta = read_header(paths[0])
            except (Fail, ValueError):
                meta = None
            if not (meta and MARK in meta):
                return "whole"
        return "absent"

    def adopt(self, name, spec):
        """Rebuild a cut shard's record from the parts' own headers and the original
        header in .k3split (a lost or older k3_layout.json). False if they do not add
        up to the published shard."""
        paths = self.part_paths(name)
        side = os.path.join(self.first, SIDE, name + ".hdr")
        try:
            head0, h0, m0 = read_header(paths[0])
            if not (m0 and MARK in m0) or not os.path.exists(side):
                return False
            with open(side, "rb") as handle:
                orig = handle.read()
            original = json.loads(orig[8:].decode("utf-8"))
            original.pop("__metadata__", None)
            data_end = data_extent(original)
            hi0 = data_extent(h0)
            parts = [{"drive": 0, "file": name, "size": len(head0) + hi0, "experts": expert_ids(h0),
                      "data": [0, hi0]}]
            at = hi0
            for d in range(1, len(paths)):
                if not os.path.exists(paths[d]):
                    continue
                head, h, m = read_header(paths[d])
                if not m or m.get("k3split.source") != name:
                    return False
                lo, n = int(m["k3split.data_start"]), data_extent(h)
                if lo != at or os.path.getsize(paths[d]) != len(head) + n:
                    return False
                parts.append({"drive": d, "file": os.path.basename(paths[d]),
                              "size": len(head) + n, "experts": expert_ids(h), "data": [lo, lo + n]})
                at = lo + n
        except (OSError, ValueError, KeyError, Fail):
            return False
        if at != data_end or len(orig) + data_end != spec["size"]:
            return False
        self.layout.set_file(name, {"size": spec["size"], "sha256": spec.get("sha256"), "kind": "moe",
                                    "status": "cutting", "parts": parts})
        say(f"{name}: its record rebuilt from the parts on disk")
        return True

    def clear_moe(self, name):
        for path in self.part_paths(name):
            for p in (path, path + ".tmp"):
                if os.path.exists(p):
                    os.remove(p)
        side = os.path.join(self.first, SIDE, name + ".hdr")
        if os.path.exists(side):
            os.remove(side)
        entry = self.layout.file(name)
        if entry:
            entry.pop("parts", None)
            self.layout.set_file(name, {**entry, "status": "pending"})

    def moe_shard(self, name, spec):
        state = self.moe_state(name, spec)
        if state == "cut":
            self.finish_cut(name)
        elif state != "done":
            if state == "absent":
                self.clear_moe(name)
                for d in range(1, len(self.drives)):      # a whole copy --no-split left elsewhere
                    other = os.path.join(self.drives[d], name)
                    if os.path.exists(other) and os.path.getsize(other) == spec["size"]:
                        say(f"{name}: moving the whole shard from {self.drives[d]} to {self.first}")
                        shutil.move(other, os.path.join(self.first, name))
                        break
            for _attempt in range(2):
                self.fetch(name, spec, self.first)
                if self.cut(name, spec):
                    break
                say(f"{name}: checksum mismatch, fetching it again")
                self.clear_moe(name)
            else:
                raise Fail(f"{name}: checksum mismatch twice")
        with self.mx:
            self.done_bytes += spec["size"]

    def cut(self, name, spec):
        """One pass over the published shard: hash every byte, copy the other drives'
        ranges into their parts, then make the file the first drive's part. False on a
        checksum mismatch (only the partial parts are removed then)."""
        paths = self.part_paths(name)
        head, header, meta = read_header(paths[0])
        parts, data_end = plan_cut(header, self.weights, name)
        if len(head) + data_end != spec["size"]:
            raise Fail(f"{name}: the header covers {len(head) + data_end} bytes, the file has {spec['size']}")
        hasher = hashlib.sha256() if spec.get("sha256") and not self.args.no_verify else None
        if hasher:
            hasher.update(head)
        outs = {}
        try:
            for d, p in enumerate(parts):
                if d == 0 or p["hi"] <= p["lo"]:
                    continue
                rebased = {t: {**s, "data_offsets": [s["data_offsets"][0] - p["lo"],
                                                     s["data_offsets"][1] - p["lo"]]}
                           for t, s in p["tensors"].items()}
                pmeta = dict(meta or {})
                pmeta.update({MARK: f"{d + 1}/{len(parts)}", "k3split.source": name,
                              "k3split.data_start": str(p["lo"])})
                outs[d] = open(paths[d] + ".tmp", "wb")
                outs[d].write(header_bytes(rebased, pmeta))
            with open(paths[0], "rb") as handle:
                handle.seek(len(head))
                at = 0
                while at < data_end:
                    block = handle.read(min(CHUNK, data_end - at))
                    if not block:
                        raise Fail(f"{name}: ends at data byte {at} of {data_end}")
                    if hasher:
                        hasher.update(block)
                    view, end = memoryview(block), at + len(block)
                    for d, out in outs.items():
                        a, b = max(parts[d]["lo"], at), min(parts[d]["hi"], end)
                        if a < b:
                            out.write(view[a - at:b - at])
                    at = end
            if hasher and hasher.hexdigest() != spec["sha256"].lower():
                for d, out in list(outs.items()):
                    out.close()
                    os.remove(paths[d] + ".tmp")
                outs.clear()
                return False
            for d, out in list(outs.items()):
                out.flush()
                os.fsync(out.fileno())
                out.close()
                os.replace(paths[d] + ".tmp", paths[d])
            outs.clear()
        finally:
            for out in outs.values():
                out.close()
        side = os.path.join(self.first, SIDE)
        os.makedirs(side, exist_ok=True)
        hdr = os.path.join(side, name + ".hdr")
        with open(hdr + ".tmp", "wb") as handle:
            handle.write(head)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(hdr + ".tmp", hdr)
        record = []
        for d, p in enumerate(parts):
            if d == 0:
                record.append({"drive": 0, "file": name, "size": len(head) + p["hi"],
                               "experts": p["experts"], "data": [0, p["hi"]]})
            elif p["hi"] > p["lo"]:
                record.append({"drive": d, "file": os.path.basename(paths[d]),
                               "size": os.path.getsize(paths[d]), "experts": p["experts"],
                               "data": [p["lo"], p["hi"]]})
        self.layout.set_file(name, {"size": spec["size"], "sha256": spec.get("sha256"), "kind": "moe",
                                    "status": "cutting", "parts": record})
        self.finish_cut(name)
        return True

    def finish_cut(self, name):
        """The first drive's part: a header naming only its tensors, the bytes past them gone."""
        entry = self.layout.file(name)
        first = entry["parts"][0]
        path = self.part_paths(name)[0]
        head, header, meta = read_header(path)
        if not (meta and MARK in meta):
            lo, hi = first["data"]
            keep = {t: s for t, s in header.items() if lo <= s["data_offsets"][0] < hi}
            pmeta = dict(meta or {})
            pmeta.update({MARK: f"1/{len(entry['parts'])}", "k3split.source": name,
                          "k3split.data_start": "0"})
            with open(path, "r+b") as handle:
                handle.write(header_bytes(keep, pmeta, length=len(head) - 8))
                handle.flush()
                os.fsync(handle.fileno())
        if os.path.getsize(path) != first["size"]:
            os.truncate(path, first["size"])
        self.layout.set_file(name, {**entry, "status": "done"})

    # ---- the whole run

    def run(self):
        args = self.args
        known_repo = self.layout.data.get("repo")
        if known_repo and known_repo != self.repo:
            raise Fail(f"{self.first} holds {known_repo} ({LAYOUT}), not {self.repo}; "
                       "give this model its own --drive folders")
        known = self.layout.data.get("drives")
        if known and [k["path"] for k in known] != self.drives:
            raise Fail(f"{LAYOUT} was made for {[k['path'] for k in known]}; give the same --drive "
                       "folders in the same order")
        if known and self.layout.data.get("split") and not self.split:
            raise Fail(f"the shards in {self.first} are cut over the drives; --join them first")
        if known and self.split and [k["weight"] for k in known] != self.weights:
            say("the weights differ from the ones the finished shards were cut with; "
                "only the shards cut from here on use the new ones")
        self.rev = self.revision()
        say(f"{self.repo} at {self.rev}, backend {self.backend}, "
            f"{'every layer cut over' if self.split else 'whole shards on'} {len(self.drives)} drive(s)")
        if self.backend == "hf":
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
            if args.high_performance:
                os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        files = {f["path"]: f for f in sd.list_repo_files(self.repo, self.rev, token=self.token)}
        self.layout.data.update({"format": 1, "repo": self.repo, "revision": self.rev, "split": self.split,
                                 "drives": [{"path": p, "weight": w} for p, w in args.drive]})
        self.layout.save()
        meta = sorted(k for k in files if not SHARD_RE.match(k))
        for name in meta:                      # the small files first: the index says what the shards hold
            self.whole_file(name, files[name], "meta")
        index = os.path.join(self.first, "model.safetensors.index.json")
        if not os.path.exists(index):
            raise Fail("the repository has no model.safetensors.index.json")
        kinds, experts_in = classify(index)
        if not args.vision:
            files = {k: v for k, v in files.items() if kinds.get(k) != "vision"}
        shards = sorted(k for k in files if SHARD_RE.match(k))
        one = expert_bytes(os.path.join(self.first, "config.json"))
        experts_b = {s: experts_in.get(s, 0) * one for s in shards} if one else {}
        moe = [(s, files[s]) for s in shards if kinds.get(s) == "moe"]
        if not self.split:
            self.assign_whole(moe)
        self.total_bytes = sum(f["size"] for f in files.values())
        self.print_plan(files, kinds, experts_b)
        self.check_space(files, kinds, experts_b)
        if args.dry_run:
            say("dry run: nothing more fetched")
            return 0
        self.tokenizer()
        with self.mx:
            self.done_bytes = sum(files[k]["size"] for k in meta)
        order = ([s for s in shards if kinds.get(s) == "resident"] + [s for s, _ in moe] +
                 [s for s in shards if kinds.get(s) == "vision"])
        stop = threading.Event()
        reporter = threading.Thread(target=self.report, args=(stop,), daemon=True)
        reporter.start()
        pool = ThreadPoolExecutor(max_workers=max(1, args.jobs))
        try:
            jobs = []
            for name in order:
                if kinds.get(name) == "moe" and self.split:
                    jobs.append(pool.submit(self.guard, self.moe_shard, name, files[name]))
                else:
                    jobs.append(pool.submit(self.guard, self.whole_file, name, files[name],
                                            kinds.get(name, "resident")))
            pending = set(jobs)
            while pending:                     # short waits: Ctrl+C reaches the main thread
                _done, pending = wait(pending, timeout=1.0, return_when=FIRST_EXCEPTION)
        except KeyboardInterrupt:
            say("interrupted; the same command continues from here")
            os._exit(130)                      # the transfers in flight resume on the next run
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

    def tokenizer(self):
        out = os.path.join(self.first, "tokenizer.json")
        if os.path.exists(out) or not os.path.exists(os.path.join(self.first, "tiktoken.model")):
            return
        import subprocess
        say("writing tokenizer.json (tools/k3_tokenizer.py)")
        result = subprocess.run([sys.executable, "-I", os.path.join(HERE, "k3_tokenizer.py"),
                                 self.first, "-o", out], capture_output=True, text=True)
        if result.returncode != 0:
            say("tokenizer.json was not written:", (result.stderr or result.stdout).strip()[-400:])

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

    def print_plan(self, files, kinds, experts_b):
        count = {}
        for name in files:
            kind = kinds.get(name, "meta") if SHARD_RE.match(name) else "meta"
            count[kind] = count.get(kind, 0) + 1
        say(f"{count.get('moe', 0)} MoE layer shards, {count.get('resident', 0)} read at load, "
            f"{count.get('vision', 0)} vision, {count.get('meta', 0)} small files: {gb(self.total_bytes)}")
        per = self.needs(files, kinds, experts_b, only_left=False)
        total = float(sum(self.weights))
        for d, path in enumerate(self.drives):
            share = f", {100 * self.weights[d] / total:.1f}% of every layer's experts" if self.split else ""
            say(f"  drive {d + 1} {path} (weight {self.weights[d]:g}): {gb(per[d])}{share}")

    def summary(self, files):
        per = [0] * len(self.drives)
        for name in files:
            entry = self.layout.file(name) or {}
            for p in entry.get("parts") or [{"drive": entry.get("drive", 0), "size": entry.get("size", 0)}]:
                per[p["drive"]] += p["size"]
        say("complete:", ", ".join(f"{p} {gb(b)}" for p, b in zip(self.drives, per)))
        if len(self.drives) > 1:
            say(f"the engine's model folder is {self.first}, with")
            say(f"  K3_DIRS={';'.join(self.drives[1:])}")


def classify(index_path):
    """{shard: 'moe' | 'resident' | 'vision'} and {shard: its routed experts}"""
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    seen, ids = {}, {}
    for tname, shard in weight_map.items():
        k = seen.setdefault(shard, {"experts": 0, "vision": 0, "other": 0})
        match = EXPERT_RE.search(tname)
        if match:
            k["experts"] += 1
            ids.setdefault(shard, set()).add(tname[:match.end()])
        elif tname.startswith(("vision_tower.", "mm_projector.")):
            k["vision"] += 1
        else:
            k["other"] += 1
    kinds = {s: "moe" if k["experts"] else ("vision" if k["vision"] and not k["other"] else "resident")
             for s, k in seen.items()}
    return kinds, {s: len(v) for s, v in ids.items()}


def expert_bytes(config_path):
    """Bytes of one routed MXFP4 expert (w1, w3: inter x hidden/2 and a scale per 32;
    w2 the transpose), from config.json; 0 if it does not say."""
    try:
        with open(config_path, encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, ValueError):
        return 0
    cfg = cfg.get("text_config", cfg)
    inter = cfg.get("moe_intermediate_size")
    hidden = cfg.get("routed_expert_hidden_size") or cfg.get("hidden_size")
    if not inter or not hidden:
        return 0
    return 3 * (inter * hidden // 2 + inter * hidden // 32)


# ------------------------------------------------------------------ verify, join, status


def layout_or_fail(first):
    path = os.path.join(first, LAYOUT)
    if not os.path.exists(path):
        raise Fail(f"no {LAYOUT} in {first}")
    return Layout(path)


def published_bytes(drives, name, entry):
    """A cut shard's published bytes, in order: its original header, then each part's data."""
    with open(os.path.join(drives[0], SIDE, name + ".hdr"), "rb") as handle:
        yield handle.read()
    for p in entry["parts"]:
        path = os.path.join(drives[p["drive"]], p["file"])
        head, _, _ = read_header(path)
        left = p["data"][1] - p["data"][0]
        with open(path, "rb") as handle:
            handle.seek(len(head))
            while left > 0:
                block = handle.read(min(CHUNK, left))
                if not block:
                    raise Fail(f"{path}: ends early")
                left -= len(block)
                yield block


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
                for block in published_bytes(drives, name, entry):
                    hasher.update(block)
            else:
                with open(os.path.join(drives[entry.get("drive", 0)], *name.split("/")), "rb") as handle:
                    for block in iter(lambda: handle.read(CHUNK), b""):
                        hasher.update(block)
        except (OSError, Fail) as error:
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
    """Every cut shard back as the published file on the first drive, checked."""
    drives = [d for d, _ in args.drive]
    layout = layout_or_fail(drives[0])
    bad = 0
    for name in sorted(layout.data["files"]):
        entry = layout.file(name)
        if not entry.get("parts"):
            continue
        if entry.get("status") not in ("done", "joining"):
            say(f"{name}: not finished, left as it is")
            bad += 1
            continue
        first = os.path.join(drives[0], name)
        part0 = entry["parts"][0]
        _, _, meta = read_header(first)
        cut = bool(meta and MARK in meta)
        if cut and os.path.getsize(first) != part0["size"]:
            os.truncate(first, part0["size"])          # a join interrupted before its header
        layout.set_file(name, {**entry, "status": "joining"})
        if cut:
            with open(first, "r+b") as out:
                out.seek(0, os.SEEK_END)
                for p in entry["parts"][1:]:
                    path = os.path.join(drives[p["drive"]], p["file"])
                    phead, _, _ = read_header(path)
                    with open(path, "rb") as src:
                        src.seek(len(phead))
                        shutil.copyfileobj(src, out, CHUNK)
                out.flush()
                os.fsync(out.fileno())
            with open(os.path.join(drives[0], SIDE, name + ".hdr"), "rb") as src:
                original = src.read()
            with open(first, "r+b") as out:
                out.write(original)
                out.flush()
                os.fsync(out.fileno())
        spec = {"size": entry["size"], "sha256": entry.get("sha256")}
        if not sd.verify_file(first, spec):
            say(f"{name}: the joined file is not the published one; its parts are kept")
            bad += 1
            continue
        for p in entry["parts"][1:]:
            os.remove(os.path.join(drives[p["drive"]], p["file"]))
        os.remove(os.path.join(drives[0], SIDE, name + ".hdr"))
        entry.pop("parts")
        layout.set_file(name, {**entry, "status": "done", "drive": 0})
        say(f"{name}: joined")
    if not bad:
        layout.data["split"] = False
        layout.save()
    say("join: every shard is the published file again" if not bad else f"join: {bad} shard(s) not joined")
    return 1 if bad else 0


def status(args):
    drives = [d for d, _ in args.drive]
    layout = layout_or_fail(drives[0])
    files = layout.data["files"]
    done = sum(1 for v in files.values() if v.get("status") == "done")
    say(f"{layout.data.get('repo')} at {layout.data.get('revision')}: {done} of {len(files)} files done, "
        f"split={layout.data.get('split')}, complete={bool(layout.data.get('complete'))}")
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
    ap.add_argument("--no-split", action="store_true", help="MoE shards whole, spread over the drives")
    ap.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True,
                    help="the vision tower's shards (0.9 GB; the text engine never reads them)")
    ap.add_argument("--reserve-gb", type=float, default=20.0, help="free space to leave on every drive")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA-256 checks (not advised)")
    ap.add_argument("--high-performance", action=argparse.BooleanOptionalAction, default=True,
                    help="HF_XET_HIGH_PERFORMANCE for the hf backend")
    ap.add_argument("--report-s", type=float, default=30.0, help="seconds between progress lines")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch the small files, plan, check the space, stop")
    ap.add_argument("--verify", action="store_true", help="re-hash everything against the published bytes")
    ap.add_argument("--join", action="store_true", help="put the cut shards back together, checked")
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
