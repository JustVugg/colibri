"""Two-GPU start-up (DSV4_CUDA_PEER_DEVICE) at source level.

tests/test_dsv4_peer_init_cuda.cu checks the behaviour on real GPUs; this
pins the shape that a CPU-only CI host can see:

- the engine settles the peer pool BEFORE any init (DSV4_HYBRID=1 first, then
  the DSV4_CUDA_PEER_MIRRORS capacity, then dsv4_cuda_peer_probe), so a
  refused peer is never initialised, and the probe refuses a latched-off peer
  and checks the primary before it touches the peer;
- it never calls dsv4_cuda_shutdown() between the failed two-device init and
  the single-device retry (on Windows the loader's shutdown unloads the DLL);
- with the peer unset the engine makes exactly one init,
  dsv4_cuda_init(&device, 1), on the primary alone;
- every failure inside dsv4_cuda_init goes through init_fail, which releases
  what was set up, drops the contexts of every device but the first and
  clears the last error, and shutdown shares that teardown;
- the probe's rebuild hint follows a missing kernel image only.
"""
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def function_body(text, signature, end="\n}\n"):
    start = text.index(signature)
    return text[start:text.index(end, start)]


class V4PeerInitSourceTest(unittest.TestCase):
    def setUp(self):
        self.engine = (HERE / "deepseek_v4.c").read_text(encoding="utf-8")
        self.backend = (HERE / "backend_cuda_dsv4.cu").read_text(encoding="utf-8")

    def open_body(self):
        return function_body(self.engine, "int coli_v4_gpu_engine_open(ColiV4Engine *engine) {")

    def test_the_peer_is_settled_before_any_init(self):
        body = self.open_body()
        hybrid = body.index("coli_v4_hybrid_enabled()")
        capacity = body.index("peer_capacity = v4_peer_mirrors_capacity()")
        probe = body.index("dsv4_cuda_peer_probe(device, peer,")
        pair = body.index("dsv4_cuda_init(devices, 2)")
        self.assertLess(hybrid, capacity)
        self.assertLess(capacity, probe)
        self.assertLess(probe, pair)
        self.assertEqual(body.count("dsv4_cuda_init("), 2)
        self.assertIn("int ndevices = 1;", body)
        self.assertEqual(body.count("ndevices = 2;"), 1)

    def test_no_shutdown_before_the_single_device_retry(self):
        body = self.open_body()
        pair = body.index("dsv4_cuda_init(devices, 2)")
        single = body.index("dsv4_cuda_init(&device, 1)")
        self.assertLess(pair, single)
        self.assertNotIn("dsv4_cuda_shutdown();", body[:single])

    def test_unset_peer_keeps_the_single_init(self):
        body = self.open_body()
        self.assertIn('int peer = peer_setting && *peer_setting ? atoi(peer_setting) : -1;', body)
        self.assertIn("if (peer >= 0) {", body)
        self.assertIn("if (ndevices == 1 && !dsv4_cuda_init(&device, 1)) {", body)

    def test_init_failures_release_and_clear(self):
        init = function_body(self.backend, 'extern "C" int dsv4_cuda_init(const int *dev,int n){',
                             "\n    return 1;}")
        # the only plain refusal is the argument check, before any device work
        self.assertEqual(init.count("return 0;"), 1)
        self.assertTrue(init.startswith('extern "C" int dsv4_cuda_init(const int *dev,int n){if(n<1||n>16)return 0;'))
        self.assertGreaterEqual(init.count("return init_fail("), 5)
        # the slot's device is known before its first resource is created
        self.assertLess(init.index("g[i].id=dev[i];"), init.index("cudaStreamCreateWithFlags"))
        fail = function_body(self.backend, "static int init_fail(", "return 0;}")
        # release, then drop every context but dev[0]'s (the retry's), then
        # clear the last error the release and the resets may have set
        release_at = fail.index("dev_release(touched);")
        reset_at = fail.index("for(int i=1;i<n;i++)if(dev[i]!=dev[0]&&cudaSetDevice(dev[i])==cudaSuccess)cudaDeviceReset();")
        self.assertLess(release_at, reset_at)
        self.assertLess(reset_at, fail.index("cudaGetLastError();"))
        self.assertEqual(init.count("return init_fail(dev,n,"), init.count("return init_fail("))
        self.assertIn('extern "C" void dsv4_cuda_shutdown(void){dev_release(ng);}', self.backend)
        release = function_body(self.backend, "static void dev_release(int n){",
                                "memset(g_peer,0,sizeof g_peer);}")
        for freed in ("cudaFree(g[i].decode_state)", "tc_dev_free(&g[i])",
                      "if(g[i].stream)cudaStreamDestroy(g[i].stream)",
                      "if(g[i].aux)cudaStreamDestroy(g[i].aux)",
                      "memset(&g[i],0,sizeof g[i]);}ng=0;"):
            self.assertIn(freed, release)

    def test_probe_checks_before_any_context(self):
        probe = function_body(self.backend, 'extern "C" int dsv4_cuda_peer_probe(', "\n    return 1;}")
        order = [probe.index(s) for s in (
            "cudaGetDeviceCount(", "peer==device", "if(dev_lost(peer))",
            "cudaGetDeviceProperties(", "dsv4_cuda_backend_arch_ok(d)",
            "cudaDeviceCanAccessPeer(&ab,device,peer)", "cudaDeviceCanAccessPeer(&ba,peer,device)",
            "cudaSetDevice(d)", "cudaFuncGetAttributes(")]
        self.assertEqual(order, sorted(order))
        # architecture and kernel image for both devices, the primary first, so
        # a primary the build cannot run never gets its peer touched
        self.assertEqual(probe.count("for(int k=0;k<2;k++){int d=k?peer:device"), 3)
        self.assertIn('const char *role[2]={"primary","peer"};', probe)
        # the kernel-image check is the only one that makes a context: the
        # peer's is undone on refusal (the primary's is the one its init makes)
        self.assertEqual(len(re.findall(r"cudaSetDevice\(", probe)), 2)
        self.assertIn("if(k&&!ng&&cudaSetDevice(peer)==cudaSuccess)cudaDeviceReset();", probe)
        # the rebuild hint follows only a missing image; any other error (a
        # full or exclusive-process GPU) is reported without it, and before it
        other = probe.index("if(e!=cudaErrorNoKernelImageForDevice&&e!=cudaErrorInvalidDeviceFunction){")
        self.assertLess(probe.index("cudaDeviceReset();"), other)
        self.assertLess(other, probe.index("CUDA_ARCH"))
        self.assertLess(other, probe.index("sm_120a code only"))
        self.assertNotIn("CUDA_ARCH", probe[other:probe.index("return 0;}", other)])


if __name__ == "__main__":
    unittest.main()
