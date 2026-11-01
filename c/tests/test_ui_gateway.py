"""The chat UI (ui/) and what the gateway gives it.

- A turn stopped while it was still reasoning (no answer text, its reasoning_content set)
  continues inside the open block on Qwen3.6 and Qwen3.8 (CONTINUE_IN_THINK); any other
  family still refuses an empty continued turn.
- /health carries the turns in flight (`live`) and the box's numbers (`system`).
- /experts?since=N returns the decode steps routed after N (the engine's HITL frames).
- ui/server.py serves the page, proxies the API to the gateway, answers /models/ with a 404
  when no models service is configured, and serves the chat's web search (ui/web.py) to the
  page only, opening public addresses only.
- ui/models.py (the Models page) judges a model against the machine it runs on: which ways it
  fits, the estimates, what an integrated GPU may claim; restarts the gateway with its own
  command and only --model changed; and answers only this page.
"""
import collections
import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

C = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C))
import openai_server as gw   # noqa: E402


class ContinueInThinkTest(unittest.TestCase):
    def setUp(self):
        self.saved = gw.ARCH
        self.addCleanup(setattr, gw, "ARCH", self.saved)
        self.msgs = [{"role": "user", "content": "What is 17 * 23?"},
                     {"role": "assistant", "content": "", "reasoning_content": "17 * 20 is 340, "}]

    def test_qwen36_goes_on_inside_the_open_block(self):
        gw.ARCH = "qwen36"
        agp = gw.resolve_generation_prompt(self.msgs, {})
        self.assertIs(agp, gw.CONTINUE_IN_THINK)
        self.assertFalse(agp)   # every other caller reads it as add_generation_prompt=False
        prompt = gw.render_chat_qwen(self.msgs, enable_thinking=True, add_generation_prompt=agp)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n17 * 20 is 340,"), prompt[-80:])
        self.assertTrue(gw.starts_in_reasoning(True, agp))

    def test_qwen38_too(self):
        gw.ARCH = "qwen38"
        agp = gw.resolve_generation_prompt(self.msgs, {})
        self.assertIs(agp, gw.CONTINUE_IN_THINK)
        prompt = gw.render_chat_qwen38(self.msgs, enable_thinking=True, add_generation_prompt=agp)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n17 * 20 is 340,"), prompt[-80:])

    def test_another_family_still_refuses_an_empty_turn(self):
        gw.ARCH = "glm"
        with self.assertRaises(gw.APIError) as caught:
            gw.resolve_generation_prompt(self.msgs, {})
        self.assertEqual(caught.exception.status, 400)


class LiveSnapshotTest(unittest.TestCase):
    def engine(self):
        e = gw.Engine.__new__(gw.Engine)   # the fields live_snapshot reads, no process
        e.live, e.live_lock, e.hits_live_seq = {}, threading.Lock(), 9
        return e

    def test_reading_then_writing(self):
        e, now = self.engine(), time.time()
        e.live["a"] = {"started": now - 2.0, "prompt_tokens": 120, "tokens": 0, "first": None, "last": None,
                       "recent": collections.deque(maxlen=32), "forwards0": 4}
        snap = e.live_snapshot()[0]
        self.assertEqual((snap["state"], snap["prompt_tokens"], snap["tokens"], snap["forwards"]), ("reading", 120, 0, 5))
        stamps = [now - 1.0 + i * 0.1 for i in range(11)]   # 11 tokens over a second
        e.live["a"].update(tokens=11, first=stamps[0], last=stamps[-1], recent=collections.deque(stamps, maxlen=32))
        snap = e.live_snapshot()[0]
        self.assertEqual(snap["state"], "writing")
        self.assertAlmostEqual(snap["tok_s"], 10.0, places=1)
        self.assertAlmostEqual(snap["tok_s_now"], 10.0, places=1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "/proc")
    def test_the_box(self):
        stats = gw._system_stats(None)
        self.assertGreater(stats["ram_total_gb"], 0)
        self.assertIn("ram_avail_gb", stats)
        self.assertGreaterEqual(stats["cores"], 1)


class LiveEngine:
    """what /health and /experts read of an engine"""
    supports_logprobs_echo = supports_tok_ids = False

    def __init__(self):
        self.emap = {"rows": 2, "cols": 8, "map": "00"}
        self.hits, self.hits_seq = "", 0
        self.hits_live = collections.deque([(1, "0001,0003"), (2, "0002"), (3, "0009")], maxlen=256)
        self.hits_live_seq = 3
        self.live, self.live_lock = {}, threading.Lock()

    def live_snapshot(self):
        return gw.Engine.live_snapshot(self)


class GatewayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LiveEngine()
        cls.server = gw.APIServer(("127.0.0.1", 0), cls.engine, "test-model", "secret", 16)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.scheduler.close()
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with urlopen(Request(self.base + path, headers={"Authorization": "Bearer secret"}), timeout=5) as r:
            return json.load(r)

    def test_health_carries_live_and_system(self):
        body = self.get("/health")
        self.assertEqual(body["live"], [])
        self.assertIsInstance(body["system"], dict)

    def test_experts_since_returns_the_later_steps(self):
        body = self.get("/experts?since=1")
        self.assertEqual((body["rows"], body["cols"], body["live_seq"]), (2, 8, 3))
        self.assertEqual(body["steps"], ["0002", "0009"])
        self.assertEqual(self.get("/experts")["live_seq"], 3)


class WebSearchTest(unittest.TestCase):
    """ui/web.py without the network: DuckDuckGo's HTML parsed, its redirect links undone,
    only public addresses opened"""

    def setUp(self):
        sys.path.insert(0, str(C.parent / "ui"))
        import web
        self.web = web

    def test_duckduckgo_results(self):
        page = ('<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&amp;rut=x">'
                'Hummingbirds &amp; flight</a><a class="result__snippet">They <b>hover</b> by a figure eight.</a></div>'
                '<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">An ad</a></div>'
                '<div class="result"><a class="result__a" href="https://example.com/b">Second</a></div>')
        p = self.web._DDG()
        p.feed(page)
        self.assertEqual([(r["title"], self.web._ddg_url(r["url"])) for r in p.rows],
                         [("Hummingbirds & flight", "https://example.org/a"), ("An ad", "https://duckduckgo.com/y.js?ad=1"),
                          ("Second", "https://example.com/b")])
        self.assertEqual(p.rows[0]["snippet"], "They hover by a figure eight.")

    def test_only_public_addresses(self):
        for url in ("http://127.0.0.1:8000/health", "http://10.1.2.3/", "http://[::1]/", "file:///etc/passwd", "ftp://example.com/"):
            with self.assertRaises(ValueError, msg=url):
                self.web._public_url(url)

    def test_an_empty_query(self):
        with self.assertRaises(ValueError):
            self.web.search("   ")


class ModelsServiceTest(unittest.TestCase):
    """ui/models.py without a network, a gateway or a GPU: a machine described by hand"""
    GB = 1_000_000_000

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(C.parent / "ui"))
        import models
        cls.m = models
        cls.tmp = tempfile.TemporaryDirectory()
        models.configure("http://127.0.0.1:1", [cls.tmp.name], state=os.path.join(cls.tmp.name, ".state"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def machine(self, gpus, ram_gb=64, bw=40.0, disk=3.0):
        self.m.MACHINE.clear()
        self.m.MACHINE.update({"ram_total_gb": ram_gb, "disk_free_gb": 500, "gpus": gpus,
                               "bench": {"ram_bw_gbs": bw, "disk_gbs": disk}})
        self.addCleanup(self.m.MACHINE.clear)

    def geo(self, dense=2.0, experts=16.0, k=8, E=128, dense16=0.0):
        G = self.GB
        return {"k": k, "E": E, "layers": 48, "experts": experts * G, "dense_active": dense * G, "resident": dense * G,
                "sizes": {"dense": dense * G, "experts": experts * G, "dense16": dense16 * G}}

    def test_a_dedicated_gpu_adds_its_vram(self):
        self.machine([{"name": "Card", "kind": "discrete", "vendor": "NVIDIA", "vram_gb": 24}], ram_gb=32)
        t = self.m.tiers_for(self.geo(dense=4, experts=40), 0, have=True, family="qwen36")
        self.assertFalse(t["ram"]["fits"])          # 44 GB, 26.8 GB of usable RAM
        self.assertTrue(t["gpu"]["fits"])           # with 24 GB of VRAM
        self.assertIsNone(t["gpu"]["est"])          # measured on first use, never guessed
        self.assertIn("primo uso", t["gpu"]["est_note"])

    def test_an_integrated_gpu_claims_only_what_was_measured(self):
        g, fam = self.geo(), "qwen36"
        self.machine([{"name": "AMD Radeon 780M Graphics", "kind": "integrated", "vendor": "AMD", "vram_gb": 0}])
        t = self.m.tiers_for(g, 0, have=True, family=fam)
        self.assertAlmostEqual(t["gpu"]["est"] / t["ram"]["est"], 9.94 / 6.01, places=1)
        self.machine([{"name": "Intel(R) Iris(R) Xe Graphics", "kind": "integrated", "vendor": "Intel", "vram_gb": 0}])
        t = self.m.tiers_for(g, 0, have=True, family=fam)
        self.assertTrue(t["gpu"]["fits"])
        self.assertIsNone(t["gpu"]["est"])          # another integrated GPU: no ratio borrowed
        t = self.m.tiers_for(g, 0, have=True, family="olmoe")
        self.assertFalse(t["gpu"]["fits"])          # an engine coli setup keeps on the CPU there
        self.assertIn("resta sulla CPU", t["gpu"]["why"])

    def test_a_cpu_only_engine_has_no_gpu_way(self):
        self.machine([{"name": "Card", "kind": "discrete", "vendor": "NVIDIA", "vram_gb": 24}])
        t = self.m.tiers_for(self.geo(), 0, have=True, family="x", accel=False)
        self.assertFalse(t["gpu"]["fits"])

    def test_qwen36_reads_its_16_bit_dense_part_as_int8(self):
        self.machine([])
        g = self.geo(dense=4, experts=16, dense16=4)
        self.assertAlmostEqual(self.m.per_token(g, "qwen36") / self.GB, 2 + 16 * 8 / 128)
        self.assertAlmostEqual(self.m.per_token(g, "olmoe") / self.GB, 4 + 16 * 8 / 128)
        self.assertGreater(self.m.tiers_for(g, 0, have=True, family="qwen36")["ram"]["est"],
                           self.m.tiers_for(g, 0, have=True, family="olmoe")["ram"]["est"])

    def test_streaming_from_the_ssd(self):
        self.machine([], ram_gb=16)
        t = self.m.tiers_for(self.geo(dense=4, experts=60), 0, have=True, family="qwen36")
        self.assertFalse(t["ram"]["fits"])
        self.assertTrue(t["ssd"]["fits"])
        self.assertLess(t["ssd"]["est_lo"], t["ssd"]["est_hi"])

    def test_a_download_that_does_not_fit_the_disk(self):
        self.machine([])
        self.m.MACHINE["disk_free_gb"] = 10
        t = self.m.tiers_for(self.geo(), 30 * self.GB, family="qwen36")
        self.assertFalse(t["disk_ok"])
        self.assertIn("non entra nel disco", t["ram"]["why"])

    def test_tensor_kinds_and_dtypes(self):
        hdr = {"__metadata__": {}, "model.layers.0.mlp.experts.3.down_proj.weight": {"dtype": "U8", "data_offsets": [0, 100]},
               "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "data_offsets": [100, 300]},
               "model.embed_tokens.weight": {"dtype": "BF16", "data_offsets": [300, 700]},
               "mtp.layers.0.weight": {"dtype": "BF16", "data_offsets": [700, 750]}}
        sz = {}
        self.m.st_header_sizes(json.dumps(hdr).encode(), sz)
        self.assertEqual(sz, {"experts": 100, "dense": 200, "dense16": 200, "embed": 400, "mtp": 50})
        path = os.path.join(self.tmp.name, "x.safetensors")
        raw = json.dumps(hdr).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(raw)) + raw)
        sz2 = {}
        self.m.st_local(path, sz2)
        self.assertEqual(sz2, sz)

    def test_a_linked_gguf_that_is_gone(self):
        d = os.path.join(self.tmp.name, "gone.colibri")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "gguf.json"), "w") as f:
            json.dump({"gguf": os.path.join(self.tmp.name, "model-00001-of-00002.gguf")}, f)
        with self.assertRaises(ValueError) as caught:
            self.m.local_geometry(d)
        self.assertIn("manca il GGUF", str(caught.exception))
        self.assertEqual([os.path.basename(p) for p in self.m.gguf_parts("/a/m-00001-of-00002.gguf")],
                         ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"])

    def test_the_gateway_command_with_another_model(self):
        argv = ["python3", "./coli", "serve", "--model", "/m/a", "--port", "8000", "--gpu=none", "--ctx", "65536"]
        self.assertEqual(self.m._argval(argv, "--model"), "/m/a")
        self.assertEqual(self.m._argval(argv, "--gpu"), "none")
        self.assertEqual(self.m._without(argv, ("--model", "--gpu")),
                         ["python3", "./coli", "serve", "--port", "8000", "--ctx", "65536"])
        self.machine([{"name": "Card", "kind": "discrete", "vendor": "NVIDIA", "vram_gb": 24}])
        self.assertEqual(self.m.tier_of(argv, {}), "ram")
        self.assertEqual(self.m.tier_of(["coli", "serve"], {"COLI_CUDA": "1", "COLI_VULKAN": "0"}), "gpu")
        self.assertEqual(self.m.tier_of(["coli", "serve"], {"COLI_VULKAN": "0"}), "ram")
        self.machine([])
        self.assertEqual(self.m.tier_of(["coli", "serve"], {}), "ram")

    def test_a_model_outside_its_folders_is_refused(self):
        code, body = self.m.handle("POST", "/models/use", "", json.dumps({"path": "/etc"}).encode())
        self.assertEqual((code, body["error"]), (400, "cartella del modello non valida"))
        self.assertEqual(self.m.handle("POST", "/models/use", "", b"[1]")[0], 400)
        self.assertEqual(self.m.handle("GET", "/models/nothing", "", b"")[0], 404)

    def test_alone_it_answers_only_this_page(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), self.m.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = "http://127.0.0.1:%d" % srv.server_port
        with urlopen(base + "/models/jobs", timeout=5) as r:
            self.assertEqual(json.load(r), {"jobs": []})
        for req in (Request(base + "/models/jobs", headers={"Host": "evil.example"}),
                    Request(base + "/models/use", data=b"{}", method="POST", headers={"Content-Type": "application/json"})):
            with self.assertRaises(HTTPError) as caught:
                urlopen(req, timeout=5)
            self.addCleanup(caught.exception.close)
            self.assertEqual(caught.exception.code, 403)


class Upstream(BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps({"status": "ok", "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


class UiServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(C.parent / "ui"))
        import server as ui
        import urllib.parse
        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        ui.Handler.backend = urllib.parse.urlparse(f"http://127.0.0.1:{cls.up.server_port}")
        ui.Handler.models_backend = None
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
        ui.Handler.port = cls.srv.server_port
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        for s in (cls.srv, cls.up):
            s.shutdown()
            s.server_close()

    def test_the_page(self):
        with urlopen(self.base + "/", timeout=5) as r:
            self.assertIn(b"<title>Colibr", r.read())

    def test_the_api_goes_to_the_gateway(self):
        with urlopen(self.base + "/health", timeout=5) as r:
            self.assertEqual(json.load(r), {"status": "ok", "path": "/health"})

    def test_no_models_service_is_a_404(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.base + "/models/list", timeout=5)
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 404)

    def test_web_page_refuses_this_machine(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.base + "/web/page?url=" + "http://127.0.0.1:1/", timeout=10)
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)

    def test_web_only_for_the_page_itself(self):
        req = Request(self.base + "/web/status", headers={"Origin": "http://evil.example"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=5)
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 403)

    def test_agents_only_for_the_page_itself(self):
        req = Request(self.base + "/agents/launch", data=b"{}", method="POST",
                      headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=5)
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
