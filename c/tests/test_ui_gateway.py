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
from unittest import mock
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


SYSCTL = "25769803776\n12\n"
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     8620.
Pages active:                                 294317.
Pages inactive:                               293226.
Pages speculative:                               520.
Pages throttled:                                   0.
"""
PS_OUT = "  37.5 2097152\n"
IOREG = ('    | |   "PerformanceStatistics" = {"In use system memory (driver)"=0,"Tiler Utilization %"=0,'
         '"Device Utilization %"=45,"In use system memory"=343597384}\n'
         '    | |   "model" = "Apple M4 Pro"\n')


def fake_run(table):
    def run(cmd, **kw):
        assert kw.get("timeout") == 0.5, kw
        out = table.get(cmd[0])
        if isinstance(out, Exception):
            raise out
        return mock.Mock(stdout=out or "", returncode=0 if out is not None else 1)
    return run


class DarwinStatsTest(unittest.TestCase):
    def setUp(self):
        gw._GPU_DARWIN.update(t=0.0, v=None)
        getattr(gw, "_CPU_DARWIN", {}).clear()
        self.eng = mock.Mock(process=mock.Mock(pid=4242))
        self.table = {"sysctl": SYSCTL, "vm_stat": VM_STAT, "ps": PS_OUT, "ioreg": IOREG}

    def stats(self, table=None):
        with mock.patch.object(sys, "platform", "darwin"), \
             mock.patch.object(gw.subprocess, "run", fake_run(table or self.table)), \
             mock.patch.object(gw.os, "getloadavg", return_value=(2.345, 1.0, 1.0)):
            return gw._system_stats(self.eng)

    def test_darwin_system_stats(self):
        s = self.stats()
        self.assertEqual(s["ram_total_gb"], 24.0)
        self.assertEqual(s["cores"], 12)
        self.assertEqual(s["ram_avail_gb"], round((8620 + 293226 + 520) * 16384 / 2**30, 2))
        self.assertEqual(s["load1"], 2.35)
        self.assertEqual(s["engine_cpu_pct"], 37.5)
        self.assertEqual(s["engine_rss_gb"], 2.0)

    def test_darwin_gpu_stats(self):
        g = self.stats()["gpus"]
        self.assertEqual(len(g), 1)
        self.assertEqual((g[0]["name"], g[0]["util_pct"]), ("Apple M4 Pro", 45))
        self.assertAlmostEqual(g[0]["mem_used_gb"], 0.32, places=2)

    def test_gpu_cached_one_second(self):
        calls = []
        base = fake_run(self.table)
        def counting(cmd, **kw):
            calls.append(cmd[0])
            return base(cmd, **kw)
        with mock.patch.object(sys, "platform", "darwin"), mock.patch.object(gw.subprocess, "run", counting), \
             mock.patch.object(gw.os, "getloadavg", return_value=(1, 1, 1)):
            gw._system_stats(self.eng)
            gw._system_stats(self.eng)
        self.assertEqual(calls.count("ioreg"), 1)

    def test_darwin_stats_degrade(self):
        import subprocess
        bad = {"sysctl": subprocess.TimeoutExpired("sysctl", 0.5), "vm_stat": "garbage\n",
               "ps": "x y\n", "ioreg": "nothing here"}
        s = self.stats(bad)
        for k in ("ram_total_gb", "cores", "ram_avail_gb", "engine_cpu_pct", "engine_rss_gb", "gpus"):
            self.assertNotIn(k, s)
        self.assertEqual(s["load1"], 2.35)
        gw._GPU_DARWIN.update(t=0.0, v=None)
        self.assertEqual(self.stats({"ps": None, "ioreg": OSError("x")}).get("load1"), 2.35)

    def test_ps_time_parse(self):
        self.assertEqual(gw._ps_time_s("0:00.03"), 0.03)
        self.assertEqual(gw._ps_time_s("12:34.50"), 754.5)
        self.assertEqual(gw._ps_time_s("1:02:03.25"), 3723.25)
        self.assertEqual(gw._ps_time_s("2-01:00:00"), 2 * 86400 + 3600)
        self.assertIsNone(gw._ps_time_s("x"))
        self.assertIsNone(gw._ps_time_s(""))

    def test_darwin_cpu_instantaneous(self):
        """%cpu on macOS is a decayed lifetime average; the engine's CPU is the delta of its
        cumulative CPU time over the wall time since the previous sample (first call: %cpu)."""
        clock = [100.0]
        def at(t, cpu_time):
            clock[0] = t
            table = dict(self.table, ps="  37.5 2097152 %s\n" % cpu_time)
            with mock.patch.object(gw.time, "monotonic", lambda: clock[0]):
                return self.stats(table).get("engine_cpu_pct")
        self.assertEqual(at(100.0, "1:00.00"), 37.5)        # no previous sample: ps's %cpu
        self.assertEqual(at(102.0, "1:03.00"), 150.0)       # 3 s of CPU in 2 s: 1.5 cores
        self.assertEqual(at(102.2, "1:03.10"), 150.0)       # < 0.5 s since: the last value
        self.assertEqual(at(104.0, "1:02.00"), 0.0)         # a smaller time (pid reused): never negative
        self.assertEqual(at(106.0, "1:04.00"), 100.0)

    def test_darwin_cpu_without_time_column(self):
        self.assertEqual(self.stats(dict(self.table, ps="  12.5 2097152\n"))["engine_cpu_pct"], 12.5)
        self.assertEqual(self.stats(dict(self.table, ps="  12.5 2097152\n"))["engine_cpu_pct"], 12.5)

    def test_linux_unchanged(self):
        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.object(gw, "_system_stats_darwin", side_effect=AssertionError), \
             mock.patch.object(gw, "_gpu_stats", return_value=None), \
             mock.patch("builtins.open", side_effect=OSError):
            self.assertEqual(gw._system_stats(self.eng), {})


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

    def test_power_needs_the_key(self):
        with urlopen(self.base + "/power", timeout=5) as r:
            self.assertEqual(json.load(r), {"available": False, "reason": "unauthorized"})

    def test_power_from_joules(self):
        with mock.patch.object(gw.power, "reading", return_value={"available": True, "watts": 12.5}):
            self.assertEqual(self.get("/power"), {"available": True, "watts": 12.5})

    def test_health_carries_live_and_system(self):
        body = self.get("/health")
        self.assertEqual(body["live"], [])
        self.assertIsInstance(body["system"], dict)

    def test_experts_since_returns_the_later_steps(self):
        body = self.get("/experts?since=1")
        self.assertEqual((body["rows"], body["cols"], body["live_seq"]), (2, 8, 3))
        self.assertEqual(body["steps"], ["0002", "0009"])
        self.assertEqual(self.get("/experts")["live_seq"], 3)


class FailingEngine(LiveEngine):
    """ACCEPTs (the streaming 200 goes out), writes one piece, then fails as the engine's ERROR does"""

    def __init__(self, error):
        super().__init__()
        self.error = error

    def generate(self, prompt, maximum, temperature, top_p, on_text, *args, on_accept=None, **kwargs):
        if on_accept is not None:
            on_accept({"prompt_tokens": 5})
        on_text("partial")
        raise self.error


class StreamErrorTest(unittest.TestCase):
    """an engine error after the streaming 200 reaches an OpenAI client as a final `data:` chunk
    with the error object (the SDKs raise it, the chat page shows it), never a faked [DONE]"""

    def stream(self, error, path="/v1/chat/completions"):
        server = gw.APIServer(("127.0.0.1", 0), FailingEngine(error), "test-model", "secret", 16)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        body = ({"model": "test-model", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
                if path == "/v1/chat/completions" else {"model": "test-model", "stream": True, "prompt": "hi"})
        req = Request(f"http://127.0.0.1:{server.server_port}{path}", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer secret"})
        with urlopen(req, timeout=10) as r:
            self.assertEqual(r.status, 200)
            raw = r.read().decode()
        chunks = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        return raw, chunks

    def test_gpu_failed_after_accept_ends_with_an_error_chunk(self):
        raw, chunks = self.stream(gw._engine_error(["GPU_FAILED", "the", "GPU", "path", "failed"], "GPU_FAILED"))
        self.assertIn("partial", raw)                 # what was written before stays
        self.assertNotIn("[DONE]", raw)               # the reply is not passed off as complete
        last = json.loads(chunks[-1])
        self.assertEqual((last["error"]["code"], last["error"]["type"]), ("gpu_failed", "server_error"))
        self.assertIn("read again on the CPU", last["error"]["message"])
        self.assertEqual(raw.count("\"error\""), 1)

    def test_any_engine_error_after_accept_ends_with_an_error_chunk(self):
        raw, chunks = self.stream(gw._engine_error(["SOMETHING_ELSE", "went", "wrong"], "SOMETHING_ELSE went wrong"))
        self.assertNotIn("[DONE]", raw)
        self.assertEqual(json.loads(chunks[-1])["error"]["code"], "engine_error")

    def test_a_gateway_fault_after_accept_still_only_closes(self):
        """not an engine ERROR frame (a pipe that broke, #597): no error object, as before"""
        raw, _ = self.stream(RuntimeError("failed to write STOP to the engine (broken pipe)"))
        self.assertIn("partial", raw)
        self.assertNotIn("[DONE]", raw)
        self.assertNotIn("\"error\"", raw)

    def test_completions_stream_too(self):
        raw, chunks = self.stream(gw._engine_error(["GPU_FAILED"], "GPU_FAILED"), "/v1/completions")
        self.assertNotIn("[DONE]", raw)
        self.assertEqual(json.loads(chunks[-1])["error"]["code"], "gpu_failed")

    def test_the_page_shows_an_error_chunk(self):
        """ui/index.html's stream reader throws on a chunk's `error`, which its catch shows as the
        reply's error (a.error, the error styling), instead of ending the reply as complete"""
        page = (C.parent / "ui" / "index.html").read_text()
        self.assertIn("if(ev.error)throw new Error(", page)
        reader = page[page.index("let ev;try{ev=JSON.parse(d)}"):]
        self.assertLess(reader.index("if(ev.error)throw new Error("), reader.index("const ch=ev.choices"))


def dispatched(*frames):
    """an Engine whose dispatcher read these stdout lines, then the engine's exit"""
    import io
    e = gw.Engine.__new__(gw.Engine)
    e.__dict__.update(closed=False, dispatcher_error=None, pending={}, pending_lock=threading.Lock(),
                      live={}, live_lock=threading.Lock(), tiers=None, hwinfo=None, emap=None, hits=None,
                      hits_seq=0, hits_live=collections.deque(maxlen=256), hits_live_seq=0,
                      profile=collections.deque(maxlen=8), profile_seq=0, metal=None, metal_total=None,
                      process=mock.Mock(stdout=io.BytesIO(b"".join(f.encode() + b"\n" for f in frames)),
                                        **{"wait.return_value": None, "poll.return_value": None}))
    e._fail_pending = lambda error: None
    with mock.patch("sys.stderr"):
        e._dispatch_stdout()
    assert str(e.dispatcher_error) == "colibri engine exited unexpectedly", e.dispatcher_error
    return e


FRAME1 = ("METAL chain=1 reason=- forwards=12 declined=0 experts_gpu=1 cache_gb=0.125 slots=64 loads=40 "
          "hits=10 wait_s=0.5 ahead=1 guesses=30 right=20 kv_regrow=0")
FRAME2 = ("METAL chain=1 reason=- forwards=20 declined=1 experts_gpu=1 cache_gb=0.125 slots=64 loads=2 "
          "hits=48 wait_s=0.25 ahead=1 guesses=36 right=30 kv_regrow=1")


class MetalFrameTest(unittest.TestCase):
    """the engine's METAL frame (one per turn, METAL=1 builds) into /health's `metal`"""

    def test_metal_frame_parsed(self):
        e = dispatched(FRAME1, FRAME2)
        self.assertEqual(e.metal, {"chain": 1, "reason": None, "forwards": 20, "declined": 1, "experts_gpu": 1,
                                   "cache_gb": 0.125, "slots": 64, "loads": 2, "hits": 48, "wait_s": 0.25,
                                   "ahead": 1, "guesses": 36, "right": 30, "kv_regrow": 1})
        self.assertEqual(e.metal_total, {"loads": 42, "hits": 58, "wait_s": 0.75, "guesses": 66, "right": 50})
        eng = LiveEngine()
        eng.metal, eng.metal_total = e.metal, e.metal_total
        server = gw.APIServer(("127.0.0.1", 0), eng, "test-model", "secret", 16)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with urlopen(Request(f"http://127.0.0.1:{server.server_port}/health",
                                 headers={"Authorization": "Bearer secret"}), timeout=5) as r:
                body = json.load(r)
        finally:
            server.scheduler.close(); server.shutdown(); server.server_close()
        self.assertEqual(body["metal"], {**e.metal, "totals": e.metal_total})

    def test_metal_frame_off_reason(self):
        e = dispatched("METAL chain=0 reason=COLI_METAL%3D0 forwards=0 declined=0 experts_gpu=0 cache_gb=0.000 "
                       "slots=0 loads=0 hits=0 wait_s=0.0000 ahead=0 guesses=0 right=0 kv_regrow=0")
        self.assertEqual((e.metal["chain"], e.metal["reason"], e.metal["experts_gpu"]), (0, "COLI_METAL=0", 0))

    def test_metal_frame_unknown_keys_ignored(self):
        e = dispatched("METAL chain=1 future=7 reason=- loads=3 hits=1 bare wait_s=0.1 later=x%20y guesses=0 right=0")
        self.assertNotIn("future", e.metal)
        self.assertNotIn("later", e.metal)
        self.assertEqual((e.metal["chain"], e.metal["loads"], e.metal["hits"]), (1, 3, 1))
        self.assertEqual(e.metal_total["loads"], 3)

    def test_gpu_failed_error_then_chain_off(self):
        """a serve's Metal failure: ERROR GPU_FAILED ends that request as a 503 the client can
        read, and the METAL frame after it says chain=0 with the reason"""
        import io
        e = gw.Engine.__new__(gw.Engine)
        pending = gw._Pending()
        e.__dict__.update(closed=False, dispatcher_error=None, pending={"7": pending}, pending_lock=threading.Lock(),
                          live={}, live_lock=threading.Lock(), tiers=None, hwinfo=None, emap=None, hits=None,
                          hits_seq=0, hits_live=collections.deque(maxlen=256), hits_live_seq=0,
                          profile=collections.deque(maxlen=8), profile_seq=0, metal=None, metal_total=None)
        frames = ["ERROR 7 GPU_FAILED the GPU path failed; the conversation will be re-read on the CPU",
                  "METAL chain=0 reason=the%20attention%20of%20layer%203%20at%20position%2040%20did%20not%20run"
                  "%20on%20the%20GPU forwards=9 declined=1 experts_gpu=0 cache_gb=0.000 slots=0 loads=0 hits=0 "
                  "wait_s=0.0000 ahead=0 guesses=0 right=0 kv_regrow=0"]
        e.process = mock.Mock(stdout=io.BytesIO(b"".join(f.encode() + b"\n" for f in frames)),
                              **{"wait.return_value": None, "poll.return_value": None})
        e._fail_pending = lambda error: None
        with mock.patch("sys.stderr"):
            e._dispatch_stdout()
        kind, error = pending.events.get_nowait()
        self.assertEqual(kind, "error")
        self.assertIsInstance(error, gw.APIError)
        self.assertEqual((error.status, error.code), (503, "gpu_failed"))
        self.assertIn("read again on the CPU", error.message)
        self.assertEqual((e.metal["chain"], e.metal["experts_gpu"]), (0, 0))
        self.assertEqual(e.metal["reason"], "the attention of layer 3 at position 40 did not run on the GPU")

    def test_no_metal_frame(self):
        e = dispatched("TIERS 0 4 4 0.0 1.5")
        self.assertIsNone(e.metal)
        self.assertIsNone(e.metal_total)
        eng = LiveEngine()   # no metal attributes at all: /health exactly as before
        server = gw.APIServer(("127.0.0.1", 0), eng, "test-model", "secret", 16)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            def health():
                with urlopen(Request(f"http://127.0.0.1:{server.server_port}/health",
                                     headers={"Authorization": "Bearer secret"}), timeout=5) as r:
                    return json.load(r)
            before = health()
            eng.metal, eng.metal_total = None, None   # an Engine that has seen no frame
            after = health()
        finally:
            server.scheduler.close(); server.shutdown(); server.server_close()
        self.assertNotIn("metal", before)
        for body in (before, after):
            body.pop("system", None)   # the box's numbers move between two reads
        self.assertEqual(before, after)


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
        # coli setup's estimate (model_fit, fitted to measured runs): the dense part and part of
        # the experts on the card, the rest from RAM; a card the page cannot size says so instead
        self.assertGreater(t["gpu"]["est"], 0)
        self.assertIsNone(t["gpu"]["est_note"])

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

    APPLE = [{"name": "Apple M3 Max GPU", "kind": "integrated", "vendor": "Apple", "vram_gb": 0}]

    def test_apple_metal_qwen38_fits_gpu_tier(self):
        self.machine(self.APPLE, ram_gb=64)
        t = self.m.tiers_for(self.geo(dense=4, experts=100), 0, have=True, family="qwen38")
        self.assertFalse(t["ram"]["fits"])
        self.assertTrue(t["gpu"]["fits"])           # dense resident, experts cached or streamed
        self.assertEqual(t["gpu"]["label"], "GPU Metal + SSD")
        self.assertIn("dall'SSD", t["gpu"]["why"])
        self.assertIn("GPU calcola", t["gpu"]["why"])
        self.machine(self.APPLE, ram_gb=8)          # the dense part alone does not fit
        t = self.m.tiers_for(self.geo(dense=10, experts=100), 0, have=True, family="qwen38")
        self.assertFalse(t["gpu"]["fits"])

    def test_apple_other_family_unchanged(self):
        self.machine(self.APPLE, ram_gb=64)
        t = self.m.tiers_for(self.geo(dense=4, experts=100), 0, have=True, family="olmoe")
        self.assertFalse(t["gpu"]["fits"])
        self.assertNotIn("label", t["gpu"])
        self.assertIn("servono", t["gpu"]["why"])

    def test_tier_follows_gateway_metal_state(self):
        self.machine(self.APPLE)
        for health, tier in (({"metal": {"chain": 1}}, "gpu"), ({"metal": {"chain": 0}}, "ram"), ({"status": "ok"}, "ram")):
            with mock.patch.object(self.m, "gateway_proc", return_value=None), \
                 mock.patch.object(self.m, "gw_json", return_value=health):
                self.assertEqual(self.m.gateway_tier(), tier)

    def test_coli_metal_0_is_not_gpu(self):
        self.machine(self.APPLE)
        self.assertEqual(self.m.tier_of(["coli"], {"COLI_METAL": "0"}), "ram")
        self.assertEqual(self.m.tier_of(["coli"], {}), "gpu")
        proc = {"coli": True, "argv": ["coli"], "env": {"COLI_METAL": "0"}, "pid": 1}
        with mock.patch.object(self.m, "gateway_proc", return_value=proc), \
             mock.patch.object(self.m, "gw_json", return_value={"metal": {"chain": 1}}):
            self.assertEqual(self.m.gateway_tier(), "ram")

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

    def test_power_goes_to_the_gateway(self):
        with urlopen(self.base + "/power", timeout=5) as r:
            self.assertEqual(json.load(r), {"status": "ok", "path": "/power"})

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


class FakeJoules:
    """the bit of joules ui/power.py uses: Session(), its devices and snapshot(), NoMeters"""
    class NoMeters(RuntimeError):
        pass

    def __init__(self, watts=None, fail=None):
        self.watts, self.fail = watts or {}, fail
        fake = self

        class Session:
            def __init__(self, interval=0.1):
                if fake.fail:
                    raise fake.fail
                self.devices = [{"name": n, "kind": "gpu" if n.startswith("GPU") else "cpu", "method": "fake"} for n in fake.watts]
                self.discovery = type("D", (), {"simulated": False})()
                self.t0 = time.monotonic()

            def snapshot(self):
                dt = time.monotonic() - self.t0
                j = {n: w * dt for n, w in fake.watts.items()}
                return {"at": time.time(), "devices_j": j, "total_j": sum(j.values())}
        self.Session = Session


class PowerTest(unittest.TestCase):
    def setUp(self):
        import power
        self.p = power
        self.addCleanup(self.reset)
        self.reset()
        self.saved = (sys.modules.get("joules"), power.INTERVAL)
        power.INTERVAL = 0.05

    def reset(self):
        with self.p._lock:
            self.p._state = self.p._session = None
            self.p._history.clear()
        if hasattr(self, "saved"):
            joules, self.p.INTERVAL = self.saved
            if joules is None:
                sys.modules.pop("joules", None)
            else:
                sys.modules["joules"] = joules

    def test_without_joules_the_page_is_told_why(self):
        sys.modules["joules"] = None   # import joules raises ImportError
        r = self.p.reading()
        self.assertFalse(r["available"])
        self.assertIn("pip install landauer-gap", r["reason"])

    def test_no_counters(self):
        fj = FakeJoules()
        fj.fail = fj.NoMeters("RAPL needs root")
        sys.modules["joules"] = fj
        r = self.p.reading()
        self.assertEqual((r["available"], r["reason"]), (False, "no power counters on this machine (RAPL needs root)"))

    def test_watts_per_device_and_in_all(self):
        sys.modules["joules"] = FakeJoules({"GPU 0 NVIDIA H200": 400.0, "CPU package 0": 60.0})
        first = self.p.reading()
        self.assertTrue(first["available"])
        self.assertIsNone(first["watts"])   # no second reading yet
        deadline = time.time() + 3
        while self.p.reading()["watts"] is None and time.time() < deadline:
            time.sleep(0.02)
        r = self.p.reading()
        self.assertEqual([(d["name"], d["kind"]) for d in r["devices"]], [("GPU 0 NVIDIA H200", "gpu"), ("CPU package 0", "cpu")])
        self.assertAlmostEqual(r["devices"][0]["watts"], 400.0, delta=8)
        self.assertAlmostEqual(r["devices"][1]["watts"], 60.0, delta=2)
        self.assertAlmostEqual(r["watts"], 460.0, delta=10)
        self.assertGreater(r["energy_j"], 0)
        r["devices"][0]["watts"] = -1   # the caller gets a copy
        self.assertNotEqual(self.p.reading()["devices"][0]["watts"], -1)

    def test_a_turns_energy(self):
        self.assertIsNone(self.p.energy_since(5.0))   # nothing asked for /power: no meters run
        sys.modules["joules"] = FakeJoules({"GPU 0 NVIDIA H200": 400.0})
        self.p.reading()
        deadline = time.time() + 3
        while len(self.p._history) < 4 and time.time() < deadline:
            time.sleep(0.02)
        self.assertAlmostEqual(self.p.energy_since(0.1), 40.0, delta=4)    # 400 W for 0.1 s, between two readings
        self.assertAlmostEqual(self.p.energy_since(0.0), 0.0, delta=0.5)
        self.assertIsNone(self.p.energy_since(60.0))  # began before the meters started: not known


if __name__ == "__main__":
    unittest.main()
