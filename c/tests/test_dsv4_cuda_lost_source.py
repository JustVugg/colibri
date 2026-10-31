"""A sticky CUDA fault latches CUDA off for the whole process (F18).

A kernel fault (launch failure, illegal address, a trap) leaves the CUDA
context unusable, and with peer access between two GPUs a fault on the peer
takes the primary's context with it: the F18 probe (a real __trap() on the
peer in the decode MoE, two RTX 3090s with NVLink) saw device 0's next calls
fail whether device 0's stream or the host waited for the peer. Before this,
the peer was latched off with "decode continues on device 0", which was not
true, and in a real server every later request failed until restart. A real
fault needs a debug build and two GPUs, so this pins the shape at source level
(tests/test_dsv4_peer_moe_cuda.c covers it on hardware with DSV4_CUDA_PEER_FAULT
=n:sticky):

- ok() latches g_cuda_lost on a sticky error, and ctx() then answers no device,
  so every entry point that looks its device up refuses without touching CUDA;
- the entry points that do not go through ctx() check it themselves;
- dsv4_cuda_lost is exported (dsv4.def, the DLL loader) and the engine's mirror
  lookups (peek, attach) and the decode MoE's host fallback read it;
- the peer-latch message only says decode continues on the primary when the
  error is not sticky;
- the attention output's grouped wo_a matvec falls back to the CPU groups when
  the backend refuses (final12 S7b: after the latch, every request failed in
  attention, at that call, until it did).
"""
import re
import unittest
from pathlib import Path

C = Path(__file__).resolve().parent.parent


def find(text, needle, start=0, where=""):
    at = text.find(needle, start)
    if at < 0:
        raise AssertionError(f"{where}: no {needle!r}")
    return at


class Dsv4CudaLostSourceTest(unittest.TestCase):
    def setUp(self):
        self.cu = (C / "backend_cuda_dsv4.cu").read_text(encoding="utf-8")
        self.engine = (C / "deepseek_v4.c").read_text(encoding="utf-8")

    def test_sticky_errors_latch_and_ctx_refuses(self):
        ok = self.cu[find(self.cu, "static int ok(cudaError_t e,const char *s){"):]
        ok = ok[:ok.index("\n")]
        self.assertIn("if(sticky(e))cuda_lost(s,e);", ok)
        ctx = self.cu[find(self.cu, "static Dev *ctx(int d){"):]
        self.assertTrue(ctx.startswith("static Dev *ctx(int d){if(g_cuda_lost)return nullptr;"))
        sticky = self.cu[find(self.cu, "static bool sticky(cudaError_t e){"):]
        sticky = sticky[:sticky.index("default:return false;}}")]
        for code in ("cudaErrorIllegalAddress", "cudaErrorLaunchFailure", "cudaErrorIllegalInstruction",
                     "cudaErrorMisalignedAddress", "cudaErrorAssert"):
            self.assertIn("case %s:" % code, sticky)
        self.assertNotIn("cudaErrorMemoryAllocation", sticky)   # an OOM is recoverable

    def test_entry_points_without_ctx_check_the_latch(self):
        for sig in ('extern "C" long long dsv4_cuda_mem_free_mb(int device){',
                    'extern "C" int dsv4_cuda_init(const int *dev,int n){',
                    'extern "C" Dsv4CudaExpertSet *dsv4_cuda_expert_bank_create('):
            at = find(self.cu, sig, where="backend_cuda_dsv4.cu")
            self.assertIn("g_cuda_lost", self.cu[at:at + 900], sig)
        dev_ok = self.cu[find(self.cu, "static bool dev_ok(int a,int b){"):]
        self.assertTrue(dev_ok.startswith("static bool dev_ok(int a,int b){return !g_cuda_lost&&("))

    def test_exported_and_read_by_the_engine(self):
        self.assertRegex((C / "dsv4.def").read_text(encoding="utf-8"), r"\n\s+dsv4_cuda_lost\n")
        loader = (C / "backend_loader_dsv4.c").read_text(encoding="utf-8")
        self.assertIn('GetProcAddress(g_dsv4.dll, "dsv4_cuda_lost")', loader)
        self.assertIn("int dsv4_cuda_lost(void);", (C / "backend_cuda_dsv4.h").read_text(encoding="utf-8"))
        self.assertIn("if (!store || !view || !store->gpu || dsv4_cuda_lost()) return -1;", self.engine)
        self.assertIn("if (!cache || !view || dsv4_cuda_lost()) return -1;", self.engine)
        self.assertIn("if (!moe_ok && ((peer_up && !coli_v4_gpu_peer_pool_up()) || dsv4_cuda_lost())) {", self.engine)

    def test_latch_message_tells_sticky_apart(self):
        latch = self.cu[find(self.cu, "static int peer_latch("):]
        latch = latch[:latch.index("return alive?-1:0;}")]
        self.assertRegex(latch, r"if\(sticky\(err\)\)\s*fprintf\(stderr,\"\[DSV4 CUDA\] peer device %d failed in the decode MoE \(%s: %s\); latched off\. The fault is sticky")
        self.assertIn("if(sticky(err))cuda_lost(", latch)

    def test_grouped_wo_a_falls_back_to_the_cpu(self):
        gpu = "grouped_on_gpu = wo_a.gpu &&\n            coli_v4_gpu_matvec_grouped(&wo_a, oa, attended, groups) == 0;"
        loop = "for (int group = 0; !grouped_on_gpu && group < groups; group++) {"
        self.assertEqual(self.engine.count("coli_v4_gpu_matvec_grouped(&wo_a, oa, attended, groups)"), 3)
        self.assertEqual(self.engine.count(gpu), 3)
        self.assertEqual(self.engine.count(loop), 3)
        self.assertNotIn("result = coli_v4_gpu_matvec_grouped(&wo_a", self.engine)


if __name__ == "__main__":
    unittest.main()
