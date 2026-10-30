"""tools/k3_download.py against a local fake of the Hugging Face hub.

The fake repository is the tiny Kimi K3 fixture laid out as the published
checkpoint is: one layer per shard, the tensors in safetensors' own order (F32,
then BF16, then U8, by name within each), the experts' six tensors back to back,
an index, the embeddings and the head in a shard of their own, a vision shard.
Nothing touches the network.

The contract under test:
  - every MoE shard is cut between experts, the parts' experts in proportion to
    the drives' weights, an expert's tensors always in one part;
  - the parts are the published bytes: --verify recomputes every shard's SHA-256
    from them, and --join gives back files equal to the published ones;
  - a rerun fetches nothing; a lost k3_layout.json is rebuilt from the parts; a cut
    interrupted after its parts were written is finished from disk; a lost part
    fetches its shard again;
  - the engine reads the cut layout (model folder + K3_DIRS) to the same tokens as
    the fixture (when a kimi_k3 binary is there: K3_BINARY, or c/kimi_k3[.exe]).
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
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
import k3_download  # noqa: E402
import setup_download  # noqa: E402
from fake_hub import FakeHub  # noqa: E402

REPO = "tester/kimi-k3-tiny"
DTYPE_ORDER = ["BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16",
               "I32", "U32", "F32", "F64", "I64", "U64"]


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
            out[name] = (spec["dtype"], spec["shape"], handle.read(b - a))
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


def fixture_dir():
    fixture = C / "kimi_k3_tiny"
    if (fixture / "model.safetensors").exists():
        return fixture
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("numpy is needed to generate the tiny K3 fixture")
    out = Path(tempfile.mkdtemp(prefix="k3tiny-"))
    shutil.copy(fixture / "config.json", out / "config.json")
    shutil.copy(fixture / "ref.json", out / "ref.json")
    subprocess.run([sys.executable, str(C / "tools" / "make_kimi_k3_tiny.py"), "--output", str(out),
                    "--force"], check=True, capture_output=True)
    return out


def make_repo(root, fixture):
    """The tiny fixture as the published checkpoint is laid out."""
    tensors = read_st(fixture / "model.safetensors")
    cfg = json.loads((fixture / "config.json").read_text())
    layers = cfg["num_hidden_layers"]
    shards = layers + 2
    name = lambda i: f"model-{i:05d}-of-{shards:05d}.safetensors"  # noqa: E731
    groups = {i: {} for i in range(1, shards + 1)}
    for tname, value in tensors.items():
        full = "language_model." + tname
        if tname.startswith("model.layers."):
            groups[int(tname.split(".")[2]) + 1][full] = value
        else:
            groups[layers + 1][full] = value
    groups[shards] = {"vision_tower.encoder.final_layernorm.weight": ("F32", [16], os.urandom(64)),
                      "mm_projector.proj.0.weight": ("F32", [4, 4], os.urandom(64))}
    weight_map = {}
    for i, group in groups.items():
        write_st(root / name(i), group)
        weight_map.update({t: name(i) for t in group})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=1))
    (root / "config.json").write_bytes((fixture / "config.json").read_bytes())
    (root / "README.md").write_text("# tiny\n")
    return [name(i) for i in groups]


def engine_binary():
    env = os.environ.get("K3_BINARY")
    if env:
        return env
    for cand in (C / "kimi_k3.exe", C / "kimi_k3"):
        if cand.exists():
            return str(cand)
    return None


def run_engine(binary, model, ids, n, env):
    result = subprocess.run([binary, str(model), "--ids", " ".join(map(str, ids)), "--ngen", str(n)],
                            env={**os.environ, **env}, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise AssertionError(f"engine exited {result.returncode}:\n{result.stderr[-2000:]}")
    return [int(t) for t in result.stdout.split("TUNE")[0].split()]


class PlanCutTest(unittest.TestCase):
    def test_full_model_shares(self):
        # a published MoE shard: 28 dense tensors first, then 896 experts of 17,547,264 bytes
        header, at = {}, 0
        for i in range(28):
            header[f"language_model.model.layers.1.self_attn.t{i:02d}"] = {"dtype": "BF16", "shape": [8],
                                                                           "data_offsets": [at, at + 1000]}
            at += 1000
        sizes = [5505024, 344064, 5505024, 344064, 5505024, 344064]
        parts = ["w1.weight_packed", "w1.weight_scale", "w2.weight_packed", "w2.weight_scale",
                 "w3.weight_packed", "w3.weight_scale"]
        for e in sorted(range(896), key=str):
            for p, size in zip(parts, sizes):
                header[f"language_model.model.layers.1.block_sparse_moe.experts.{e}.{p}"] = {
                    "dtype": "U8", "shape": [size], "data_offsets": [at, at + size]}
                at += size
        cut, end = k3_download.plan_cut(header, [10.4, 6.55], "shard")
        self.assertEqual(end, at)
        self.assertEqual([p["experts"] for p in cut], [550, 346])
        self.assertEqual(cut[0]["lo"], 0)
        self.assertEqual(cut[0]["hi"], cut[1]["lo"])
        self.assertEqual(cut[1]["hi"], end)
        self.assertEqual(len(cut[0]["tensors"]) + len(cut[1]["tensors"]), len(header))
        for p in cut:   # every expert's six tensors in one part
            ids = {}
            for t in p["tensors"]:
                m = k3_download.EXPERT_RE.search(t)
                if m:
                    ids[m.group(1)] = ids.get(m.group(1), 0) + 1
            self.assertTrue(all(v == 6 for v in ids.values()))
        three, _ = k3_download.plan_cut(header, [1, 1, 1], "shard")
        self.assertEqual([p["experts"] for p in three], [299, 298, 299])

    def test_a_drive_without_experts(self):
        header = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                  "x.experts.0.w1.weight_packed": {"dtype": "U8", "shape": [4], "data_offsets": [4, 8]}}
        cut, _ = k3_download.plan_cut(header, [1, 100], "s")
        self.assertEqual([p["experts"] for p in cut], [0, 1])
        self.assertEqual(set(cut[0]["tensors"]), {"a"})


class DownloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixture_dir()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.src = root / "src"
        self.src.mkdir()
        self.shards = make_repo(self.src, self.fixture)
        self.a, self.b = root / "drive-a" / "k3", root / "drive-b" / "k3"
        self.hub = FakeHub({REPO: str(self.src)}, page_size=4)
        self.hub.__enter__()
        self.addCleanup(self.hub.__exit__)
        for patcher in (mock.patch.dict(os.environ, {"HF_ENDPOINT": self.hub.base}),
                        mock.patch.object(setup_download.time, "sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_tool(self, *extra, drives=None):
        drives = drives or [f"{self.a}=2", f"{self.b}=1"]
        argv = ["--repo", REPO, "--revision", "0" * 40, "--backend", "http", "--jobs", "2",
                "--report-s", "60", "--reserve-gb", "0"]
        for d in drives:
            argv += ["--drive", d]
        return k3_download.main(argv + list(extra))

    def cdn_requests(self):
        return sum(1 for r in self.hub.requests if r["path"].startswith("/cdn/"))

    def layout(self):
        return json.loads((self.a / k3_download.LAYOUT).read_text())

    def assert_published(self):
        """--join gives back the published files, byte for byte."""
        self.assertEqual(self.run_tool("--join"), 0)
        for name in self.shards:
            self.assertEqual((self.a / name).read_bytes(), (self.src / name).read_bytes(), name)
        self.assertEqual(list(self.b.glob("*.safetensors")), [])

    def test_cut_verify_engine_join(self):
        self.assertEqual(self.run_tool(), 0)
        layout = self.layout()
        self.assertTrue(layout["complete"])
        moe = [n for n, e in layout["files"].items() if e.get("parts")]
        self.assertEqual(len(moe), 4)                  # layers 2-5 of the tiny model
        for name in moe:
            parts = layout["files"][name]["parts"]
            self.assertEqual([p["experts"] for p in parts], [5, 3])   # 8 experts, weights 2:1
            _, h0, m0 = k3_download.read_header(self.a / name)
            _, h1, m1 = k3_download.read_header(self.b / k3_download.part_name(name, 1))
            self.assertEqual((m0[k3_download.MARK], m1[k3_download.MARK]), ("1/2", "2/2"))
            self.assertFalse(set(h0) & set(h1))
            self.assertEqual(set(h0) | set(h1), set(k3_download.read_header(self.src / name)[1]))
        for name in self.shards:                       # the rest stays whole on the first drive
            if name not in moe:
                self.assertEqual((self.a / name).read_bytes(), (self.src / name).read_bytes())
        self.assertEqual(self.run_tool("--verify"), 0)
        binary = engine_binary()
        if binary:
            ref = json.loads((self.fixture / "ref.json").read_text())
            env = dict(ref["engine_env"], K3_DIRS=str(self.b))
            for case_name, case in ref["cases"].items():
                got = run_engine(binary, self.a, case["prompt_ids"], case["max_new_tokens"], env)
                self.assertEqual(got, case["greedy_new_ids"], case_name)
        self.assert_published()

    def test_rerun_fetches_nothing(self):
        self.assertEqual(self.run_tool(), 0)
        before = self.cdn_requests()
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.cdn_requests(), before)

    def test_lost_layout_is_rebuilt(self):
        self.assertEqual(self.run_tool(), 0)
        (self.a / k3_download.LAYOUT).unlink()
        before = self.cdn_requests()
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.cdn_requests(), before)
        self.assertEqual(self.run_tool("--verify"), 0)

    def test_interrupted_cut_is_finished(self):
        real = k3_download.Downloader.finish_cut
        calls = []

        def crash_once(self_, name):
            if not calls:
                calls.append(name)
                raise OSError("power cut")
            return real(self_, name)

        with mock.patch.object(k3_download.Downloader, "finish_cut", crash_once):
            self.assertEqual(self.run_tool(), 1)
        self.assertEqual(self.layout()["files"][calls[0]]["status"], "cutting")
        before = self.cdn_requests()
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.cdn_requests(), before)
        self.assertEqual(self.run_tool("--verify"), 0)

    def test_lost_part_fetches_its_shard(self):
        self.assertEqual(self.run_tool(), 0)
        name = next(n for n, e in self.layout()["files"].items() if e.get("parts"))
        (self.b / k3_download.part_name(name, 1)).unlink()
        before = self.cdn_requests()
        self.assertEqual(self.run_tool(), 0)
        self.assertGreater(self.cdn_requests(), before)
        self.assertEqual(self.run_tool("--verify"), 0)
        self.assert_published()

    def test_corrupt_part_fails_verify(self):
        self.assertEqual(self.run_tool(), 0)
        name = next(n for n, e in self.layout()["files"].items() if e.get("parts"))
        path = self.b / k3_download.part_name(name, 1)
        data = bytearray(path.read_bytes())
        data[-1] ^= 0x40
        path.write_bytes(bytes(data))
        self.assertEqual(self.run_tool("--verify"), 1)

    def test_whole_shards_then_cut(self):
        self.assertEqual(self.run_tool("--no-split"), 0)
        layout = self.layout()
        on = [e["drive"] for n, e in layout["files"].items() if e.get("kind") == "moe"]
        # four shards of near equal size, weights 2:1: three on A, one on B finish first
        # (A 3/2, B 1/1 shard-times, against 2/2 and 2/1 for two each)
        self.assertEqual(sorted(on), [0, 0, 0, 1])
        self.assertEqual(self.run_tool("--verify"), 0)
        self.assertEqual(self.run_tool(), 0)            # then cut: the shards on drive B move first
        self.assertTrue(all(e.get("parts") for e in self.layout()["files"].values() if e.get("kind") == "moe"))
        self.assertEqual(self.run_tool("--verify"), 0)

    def test_dry_run_fetches_no_shard(self):
        self.assertEqual(self.run_tool("--dry-run"), 0)
        self.assertFalse(any(self.a.glob("*.safetensors")))
        self.assertTrue((self.a / "model.safetensors.index.json").exists())


if __name__ == "__main__":
    unittest.main()
