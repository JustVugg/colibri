"""tools/q38_download.py against a local fake of the Hugging Face hub.

The fake repository is laid out as the published Qwen3.8-Flash-Next-FP8 is, in
miniature: a layer's experts in two shards (gate and up with their block scales in
one; down, its scales and the next layer's resident tensors in the next), the PLE
table's parts four to a shard with a few beside other tensors, the MTP layer's down
sharing a shard with a layer's, the tensors in safetensors' own order (I64, then
BF16, then F8, by name within each). Nothing touches the network.

The contract under test:
  - every layer's experts go to the drives in id ranges in proportion to their
    weights, an expert's gate, up and down on the same drive, the PLE parts the
    same way, everything else on the first drive; a shard whose tensors all belong
    to one drive lies there whole, the others are cut into one file per drive;
  - the folders hold every published tensor exactly once with its published bytes,
    found the way the engine's st_scan_dir finds them (every *.safetensors of the
    folders, by name); the parts' streamed tensors start on 4096-byte boundaries;
  - --verify recomputes every shard's SHA-256 from the parts, --join gives back
    files equal to the published ones;
  - a rerun fetches nothing; a lost q38_layout.json is rebuilt; a cut interrupted
    after its parts were written is redone from disk; a lost part fetches its shard
    again; a published copy already in a folder is laid out without fetching; other
    weights are refused until a --join, after which they lay it out again offline.
"""
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
C = HERE.parent
sys.path.insert(0, str(C))
sys.path.insert(0, str(C / "tools"))
sys.path.insert(0, str(HERE))
import q38_download  # noqa: E402
import setup_download  # noqa: E402
from fake_hub import FakeHub  # noqa: E402

REPO = "tester/qwen38-tiny-fp8"
DTYPE_ORDER = ["BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16",
               "I32", "U32", "F32", "F64", "I64", "U64"]
ESZ = {"F8_E4M3": 1, "BF16": 2, "I64": 8}
EXPERTS = 8
PLE_PARTS = 20
LM = "model.language_model."


def read_st(path):
    with open(path, "rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(n))
        header.pop("__metadata__", None)
        base = 8 + n
        out = {}
        for name, spec in header.items():
            a, b = spec["data_offsets"]
            handle.seek(base + a)
            out[name] = (spec["dtype"], spec["shape"], handle.read(b - a), base + a)
    return out


def write_st(path, tensors):
    """safetensors as the published shards are written: dtype rank descending, then name."""
    order = sorted(tensors, key=lambda t: (-DTYPE_ORDER.index(tensors[t][0]), t))
    header, at = {}, 0
    for name in order:
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(data)]}
        at += len(data)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(text)) + text)
        for name in order:
            handle.write(tensors[name][2])


def tensor(dtype, *shape):
    n = ESZ[dtype]
    for s in shape:
        n *= s
    return dtype, list(shape), os.urandom(n)


def experts(prefix, which):
    """`which`: 'gate_up' (the matrices and their scales) or 'down'"""
    out = {}
    for e in range(EXPERTS):
        for proj in (("gate_proj", "up_proj") if which == "gate_up" else ("down_proj",)):
            out[f"{prefix}.mlp.experts.{e}.{proj}.weight"] = tensor("F8_E4M3", 16, 8)
            out[f"{prefix}.mlp.experts.{e}.{proj}.weight_scale_inv"] = tensor("BF16", 2, 2)
    return out


def resident(layer):
    return {f"{LM}layers.{layer}.linear_attn.in_proj_qkv.weight": tensor("BF16", 8, 12),
            f"{LM}layers.{layer}.mlp.gate.weight": tensor("BF16", 8, 8),
            f"{LM}layers.{layer}.mlp.shared_expert.down_proj.weight": tensor("BF16", 8, 4)}


def ple(*parts):
    return {f"{LM}layers.1.ple.ple_embedding.ngram_embedding.shard_{k}.weight": tensor("F8_E4M3", 16, 16)
            for k in parts}


def make_repo(root):
    """The miniature checkpoint; returns its shard names."""
    shards = [
        {"model.visual.blocks.0.attn.qkv.weight": tensor("BF16", 8, 8), **resident(0)},
        experts(f"{LM}layers.0", "gate_up"),
        {**experts(f"{LM}layers.0", "down"), **resident(1)},
        experts(f"{LM}layers.1", "gate_up"),
        {**experts(f"{LM}layers.1", "down"), **ple(0, 1),       # the PLE module's tensors beside them,
         f"{LM}layers.1.ple.key_proj.weight": tensor("BF16", 8, 8),   # a 2-byte one among them
         f"{LM}layers.1.ple.ple_embedding.ngram_embedding.weight_scale": tensor("BF16"),
         f"{LM}layers.1.ple.ple_embedding.layer_multipliers": tensor("I64", 3)},
        ple(2, 3, 4, 5), ple(6, 7, 8, 9), ple(10, 11, 12, 13), ple(14, 15, 16, 17),
        {**ple(18, 19), f"{LM}layers.1.ple.ple_embedding.ngram_heads_offsets": tensor("I64", 16),
         "mtp.hyper_connection_mixer.hc_norm.weight": tensor("BF16", 8)},
        experts(f"{LM}layers.2", "gate_up"),
        {**resident(2), **experts("mtp.layers.0", "gate_up")},
        {**experts(f"{LM}layers.2", "down"), **experts("mtp.layers.0", "down"),
         "mtp.layers.0.self_attn.q_proj.weight": tensor("BF16", 8, 8)},
        {"lm_head.weight": tensor("BF16", 64, 16), f"{LM}embed_tokens.weight": tensor("BF16", 64, 16)},
    ]
    names = [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]
    weight_map = {}
    for name, group in zip(names, shards):
        write_st(root / name, group)
        weight_map.update({t: name for t in group})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=1))
    (root / "config.json").write_text(json.dumps({"architectures": ["Qwen4ExpForConditionalGeneration"],
                                                  "text_config": {"num_experts": EXPERTS}}))
    (root / "README.md").write_text("# tiny\n")
    return names


def expected_drive(tname, expert_cut, ple_cut):
    """The drive a tensor belongs on (two drives), or None for one read only at load
    (the first drive)."""
    match = re.search(r"\.experts\.(\d+)\.(gate|up|down)_proj\.weight$", tname)
    if match:
        return int(int(match.group(1)) >= expert_cut)
    match = re.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", tname)
    if match:
        return int(int(match.group(1)) >= ple_cut)
    return None


def scan(folders):
    """{tensor: (dtype, shape, bytes, file offset, path)} the way st_scan_dir indexes the
    folders: every *.safetensors, a basename taken from the first folder holding it, a
    tensor name in two files refused."""
    seen, out = set(), {}
    for folder in folders:
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".safetensors") or name in seen:
                continue
            seen.add(name)
            for tname, value in read_st(Path(folder) / name).items():
                assert tname not in out, f"{tname} in {out[tname][4]} and {name}"
                out[tname] = (*value, Path(folder) / name)
    return out


class PlanTest(unittest.TestCase):
    def test_full_model_ranges(self):
        self.assertEqual(q38_download.bounds(512, [10.4, 6.55]), [0, 314, 512])
        self.assertEqual(q38_download.bounds(128, [10.4, 6.55]), [0, 79, 128])
        self.assertEqual(q38_download.bounds(512, [1, 1, 1]), [0, 171, 341, 512])
        self.assertEqual(q38_download.bounds(512, [1, 1e-9]), [0, 512, 512])

    def test_full_model_shard(self):
        # model-00002 as published: 1,024 BF16 block scales, then the 512 experts' gate
        # and up in name order (0, 1, 10, 100, ...), 1,638,400 bytes each
        header, at = {}, 0
        for e in sorted(range(512), key=str):
            for proj in ("gate_proj", "up_proj"):
                header[f"{LM}layers.0.mlp.experts.{e}.{proj}.weight_scale_inv"] = {
                    "dtype": "BF16", "shape": [5, 20], "data_offsets": [at, at + 200]}
                at += 200
        for e in sorted(range(512), key=str):
            for proj in ("gate_proj", "up_proj"):
                header[f"{LM}layers.0.mlp.experts.{e}.{proj}.weight"] = {
                    "dtype": "F8_E4M3", "shape": [640, 2560], "data_offsets": [at, at + 1638400]}
                at += 1638400
        place = q38_download.Placement([10.4, 6.55], [header])
        parts = q38_download.plan_parts("model-00002-of-00131.safetensors", header, None, place)
        self.assertEqual([p["file"] for p in parts], ["model-00002-of-00131.safetensors",
                                                      "model-00002-of-00131.p2.safetensors"])
        ids = [{int(t.split(".experts.")[1].split(".")[0]) for t in p["names"] if t.endswith(".weight")}
               for p in parts]
        self.assertEqual(ids, [set(range(314)), set(range(314, 512))])
        self.assertEqual(len(parts[0]["names"]), 1024 + 628)     # every scale stays on the first drive
        for p in parts:
            h, meta = q38_download.parse_header(p["head"], p["file"])
            self.assertEqual(meta[q38_download.SOURCE], "model-00002-of-00131.safetensors")
            self.assertEqual(p["size"], len(p["head"]) + q38_download.data_extent(h, p["file"]))
            for tname, spec in h.items():
                if tname.endswith(".weight"):           # every matrix one aligned read
                    self.assertEqual((len(p["head"]) + spec["data_offsets"][0]) % 4096, 0, tname)

    def test_tensor_drives(self):
        place = q38_download.Placement([2, 1], [{f"x.mlp.experts.{e}.down_proj.weight": {} for e in range(8)},
                                                {f"y.ngram_embedding.shard_{k}.weight": {} for k in range(20)}])
        self.assertEqual(place.ranges, {"expert": [0, 5, 8], "ple": [0, 13, 20]})
        self.assertEqual(place.drive("x.mlp.experts.4.up_proj.weight"), 0)
        self.assertEqual(place.drive("x.mlp.experts.5.gate_proj.weight"), 1)
        self.assertEqual(place.drive("x.mlp.experts.5.gate_proj.weight_scale_inv"), 0)
        self.assertEqual(place.drive("y.ngram_embedding.shard_13.weight"), 1)
        self.assertEqual(place.drive("y.ngram_embedding.weight_scale"), 0)
        self.assertEqual(place.drive("lm_head.weight"), 0)


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.src = root / "src"
        self.src.mkdir()
        self.shards = make_repo(self.src)
        self.a, self.b = root / "drive-a" / "q38", root / "drive-b" / "q38"
        self.hub = FakeHub({REPO: str(self.src)}, page_size=4)
        self.hub.__enter__()
        self.addCleanup(self.hub.__exit__)
        for patcher in (mock.patch.dict(os.environ, {"HF_ENDPOINT": self.hub.base}),
                        mock.patch.object(setup_download.time, "sleep"),
                        mock.patch.object(q38_download.time, "sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_tool(self, *extra, drives=None):
        drives = drives or [f"{self.a}=2", f"{self.b}=1"]
        argv = ["--repo", REPO, "--revision", "0" * 40, "--backend", "http", "--jobs", "2",
                "--report-s", "60", "--reserve-gb", "0"]
        for d in drives:
            argv += ["--drive", d]
        return q38_download.main(argv + list(extra))

    def cdn_requests(self):
        return sum(1 for r in self.hub.requests if r["path"].startswith("/cdn/"))

    def layout(self):
        return json.loads((self.a / q38_download.LAYOUT).read_text())

    def published(self):
        out = {}
        for name in self.shards:
            out.update(read_st(self.src / name))
        return out

    def assert_laid_out(self, expert_cut=5, ple_cut=13):
        """Every published tensor once, with its bytes, on the drive its id says."""
        got, pub = scan([self.a, self.b]), self.published()
        self.assertEqual(set(got), set(pub))
        drive = {t: expected_drive(t, expert_cut, ple_cut) for t in got}
        for tname, (dtype, shape, data, off, path) in got.items():
            self.assertEqual((dtype, shape, data), pub[tname][:3], tname)
            self.assertEqual(path.parent, [self.a, self.b][drive[tname] or 0], tname)
            meta = q38_download.read_header(path)[2]
            if drive[tname] is not None and meta and q38_download.MARK in meta \
                    and path.name != self.shards[4]:   # its first part's lead is not a multiple of 8
                first = min(o for t, (_, _, _, o, p) in got.items() if p == path and drive[t] is not None)
                self.assertEqual(first % 4096, 0, path.name)
        return got

    def assert_published(self):
        """--join gives back the published files, byte for byte."""
        self.assertEqual(self.run_tool("--join"), 0)
        for name in self.shards:
            self.assertEqual((self.a / name).read_bytes(), (self.src / name).read_bytes(), name)
        self.assertEqual(list(self.b.glob("*.safetensors")), [])

    def test_layout_verify_join(self):
        self.assertEqual(self.run_tool(), 0)
        layout = self.layout()
        self.assertTrue(layout["complete"])
        kinds = {n: (e["kind"], e.get("drive")) for n, e in layout["files"].items() if n in self.shards}
        whole = {n for n, (k, _) in kinds.items() if k == "whole"}
        self.assertEqual(sorted(self.shards.index(n) + 1 for n in whole), [1, 6, 7, 9, 14])
        self.assertEqual(kinds[self.shards[8]], ("whole", 1))      # PLE 14-17: all past 13
        for name in self.shards:
            if name in whole:
                folder = [self.a, self.b][kinds[name][1]]
                self.assertEqual((folder / name).read_bytes(), (self.src / name).read_bytes(), name)
            else:
                parts = layout["files"][name]["parts"]
                self.assertEqual([p["drive"] for p in parts], [0, 1], name)
                self.assertTrue(q38_download.is_part(self.a / name, name))
                self.assertTrue(q38_download.is_part(self.b / q38_download.part_name(name, 1), name))
        self.assert_laid_out()
        self.assertEqual(self.run_tool("--verify"), 0)
        self.assert_published()
        self.assertEqual(self.run_tool("--verify"), 0)

    def test_rerun_fetches_nothing(self):
        self.assertEqual(self.run_tool(), 0)
        before = self.cdn_requests()
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.cdn_requests(), before)

    def test_lost_layout_is_rebuilt(self):
        self.assertEqual(self.run_tool(), 0)
        (self.a / q38_download.LAYOUT).unlink()
        before = self.cdn_requests()
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.cdn_requests(), before)
        self.assertEqual(self.run_tool("--verify"), 0)

    def test_interrupted_cut_is_redone(self):
        real = q38_download.Downloader.record_cut
        calls = []

        def crash_once(self_, name, spec, parts, status):
            real(self_, name, spec, parts, status)
            if status == "cutting" and not calls:   # the other drives' parts are in place,
                calls.append(name)                  # the first drive still has the whole shard
                raise OSError("power cut")

        with mock.patch.object(q38_download.Downloader, "record_cut", crash_once):
            self.assertEqual(self.run_tool(), 1)
        self.assertEqual(self.layout()["files"][calls[0]]["status"], "cutting")
        self.assertFalse(q38_download.is_part(self.a / calls[0]))
        before = self.cdn_requests()
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.cdn_requests(), before)
        self.assertEqual(self.run_tool("--verify"), 0)
        self.assert_laid_out()

    def test_lost_part_fetches_its_shard(self):
        self.assertEqual(self.run_tool(), 0)
        name = self.shards[2]
        (self.b / q38_download.part_name(name, 1)).unlink()
        before = self.cdn_requests()
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.cdn_requests(), before + 1)
        self.assertEqual(self.run_tool("--verify"), 0)
        self.assert_published()

    def test_corrupt_part_fails_verify(self):
        self.assertEqual(self.run_tool(), 0)
        path = self.b / q38_download.part_name(self.shards[2], 1)
        data = bytearray(path.read_bytes())
        data[-1] ^= 0x40
        path.write_bytes(bytes(data))
        self.assertEqual(self.run_tool("--verify"), 1)

    def test_published_copy_is_laid_out_offline(self):
        self.a.mkdir(parents=True)                # an earlier `hf download` into the first folder
        for path in self.src.iterdir():
            (self.a / path.name).write_bytes(path.read_bytes())
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.cdn_requests(), 0)
        self.assert_laid_out()
        self.assertEqual(self.run_tool("--verify"), 0)

    def test_other_weights_need_a_join(self):
        self.assertEqual(self.run_tool(), 0)
        other = [f"{self.a}=1", f"{self.b}=1"]
        self.assertEqual(self.run_tool(drives=other), 1)
        self.assert_laid_out()                     # untouched
        self.assertEqual(self.run_tool("--join"), 0)
        before = self.cdn_requests()
        self.assertEqual(self.run_tool(drives=other), 0)
        self.assertEqual(self.cdn_requests(), before)
        self.assert_laid_out(expert_cut=4, ple_cut=10)
        self.assertEqual(self.run_tool("--verify", drives=other), 0)

    def test_one_drive_keeps_the_published_files(self):
        self.assertEqual(self.run_tool(drives=[str(self.a)]), 0)
        for name in self.shards:
            self.assertEqual((self.a / name).read_bytes(), (self.src / name).read_bytes(), name)
        self.assertEqual(self.run_tool("--verify", drives=[str(self.a)]), 0)
        before = self.cdn_requests()                # then spread over two without fetching
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.cdn_requests(), before)
        self.assert_laid_out()

    def test_dry_run_fetches_no_shard(self):
        self.assertEqual(self.run_tool("--dry-run"), 0)
        self.assertFalse(any(self.a.glob("*.safetensors")))
        self.assertFalse(any(self.b.glob("*.safetensors")))
        self.assertTrue((self.a / "model.safetensors.index.json").exists())
        self.assertEqual(len(list((self.a / q38_download.SIDE).glob("*.hdr"))), len(self.shards))


if __name__ == "__main__":
    unittest.main()
