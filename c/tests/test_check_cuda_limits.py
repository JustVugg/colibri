"""Unit tests of tools/check_cuda_limits.py and tools/check_cuda_dups.py on
canned compiler output and small sources: no nvcc, no GPU. The compiled gate
with its planted controls is `make check-cuda-limits`."""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
import check_cuda_dups as dups  # noqa: E402
import check_cuda_limits as lim  # noqa: E402

PTXAS = """\
ptxas info    : Function properties for _Z6helperv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Compiling entry function '_Z3fooPf' for 'sm_120'
ptxas info    : Function properties for _Z3fooPf
    80 bytes stack frame, 8 bytes spill stores, 12 bytes spill loads
ptxas info    : Used 64 registers, used 1 barriers, 80 bytes cumulative stack size, 32896 bytes smem
ptxas info    : Compile time = 1.0 ms
ptxas info    : Compiling entry function '_ZL3barv' for 'sm_120'
ptxas info    : Function properties for _ZL3barv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 39 registers, used 0 barriers
"""

ELF = """\
.nv.info._Z3fooPf
\t<0x13>
\tAttribute:\tEIATTR_MAX_THREADS
\tFormat:\tEIFMT_SVAL
\tValue:\t0x400 0x1 0x1 
.nv.info._ZL3barv
\tAttribute:\tEIATTR_REGCOUNT
\tValue:\t0x27
"""


class Parse(unittest.TestCase):
    def test_ptxas(self):
        k = lim.parse_ptxas(PTXAS)
        self.assertEqual(set(k), {"_Z3fooPf", "_ZL3barv"})
        self.assertEqual((k["_Z3fooPf"]["regs"], k["_Z3fooPf"]["smem"], k["_Z3fooPf"]["stack"]), (64, 32896, 80))
        self.assertEqual((k["_Z3fooPf"]["spill_st"], k["_Z3fooPf"]["spill_ld"]), (8, 12))
        self.assertEqual((k["_ZL3barv"]["regs"], k["_ZL3barv"]["smem"], k["_ZL3barv"]["spill_st"]), (39, 0, 0))

    def test_elf_bounds(self):
        self.assertEqual(lim.parse_elf_bounds(ELF), {"_Z3fooPf": 1024})

    def test_base_name(self):
        self.assertEqual(lim.base_name("_Z10cc_dnrec_kILi256EEvPKf"), "cc_dnrec_k")
        self.assertEqual(lim.base_name("_ZL9cc_copy_kPfPKfx"), "cc_copy_k")
        self.assertEqual(lim.base_name("_ZN9deep_gemm4implILi1EEEvv"), "impl")

    def test_reg_alloc(self):
        # 255 registers round to 256 a thread: 256 threads fill the 64 Ki file exactly
        self.assertEqual(lim.reg_alloc(255, 256), 65536)
        self.assertGreater(lim.reg_alloc(65, 1024), 65536)
        self.assertEqual(lim.reg_alloc(64, 1024), 65536)
        self.assertEqual(lim.reg_alloc(33, 1), 40 * 32)  # a partial warp still takes a whole one

    def test_evaluate(self):
        c = {"NW": 4, "THREADS": 256}
        self.assertEqual(lim.evaluate("32 * NW", c), 128)
        self.assertEqual(lim.evaluate("dim3(THREADS)", c), 256)
        self.assertEqual(lim.evaluate("dim3(32, 4, 1)", c), 128)
        self.assertEqual(lim.evaluate("(unsigned)(64 * 1024u)", c), 65536)
        self.assertEqual(lim.evaluate("(n + 255) / 256", c), None)
        self.assertEqual(lim.evaluate("2*sizeof(float)*128", c), 1024)

    def test_constants_ambiguous(self):
        c = lim.constants(["#define A 128\n#define B (A * 2)\nconst int K = 6;\n",
                           "static constexpr int K = 8;\n"])
        self.assertEqual(c["A"], 128)
        self.assertEqual(c["B"], 256)
        self.assertNotIn("K", c)  # two definitions: not trusted


SRC = """\
template <int N> __global__ void __launch_bounds__(128) tk(float *y) {}
__global__ void pk(float *y) {}
void go(float *y, cudaStream_t s) {
    tk<4><<<dim3(8), 256, 0, s>>>(y);   // a comment with <<<1, 1>>> is not a launch
    coli_launch(tk<2, true>, dim3(1), 128, 65536, s, y);
    cudaFuncSetAttribute(tk<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    pk<<<1, 1024, 0>>>(y);
}
"""


class Sources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = pathlib.Path(self.tmp.name) / "k.cu"
        self.f.write_text(SRC)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sites(self):
        sites, optins = lim.launch_sites([self.f])
        got = [(s["kernel"], s["block"], s["dyn"], s["line"]) for s in sites]
        self.assertIn(("tk", "256", "0", 4), got)
        self.assertIn(("tk", "128", "65536", 5), got)
        self.assertIn(("pk", "1024", "0", 7), got)
        self.assertEqual(len(got), 3)
        self.assertEqual([(o["kernel"], o["bytes"]) for o in optins], [("tk", "65536")])

    def test_definitions(self):
        d = lim.kernel_definitions([self.f])
        self.assertEqual(d["tk"][0][1], 1)
        self.assertEqual(d["pk"][0][1], 2)

    def unit(self, regs_pk, spill=(0, 0)):
        k = {"_Z2tkILi4EEvPf": {"name": "_Z2tkILi4EEvPf", "regs": 32, "smem": 0, "stack": 0,
                                "spill_st": spill[0], "spill_ld": spill[1], "bound": 128},
             "_Z2pkPf": {"name": "_Z2pkPf", "regs": regs_pk, "smem": 0, "stack": 0, "spill_st": 0, "spill_ld": 0}}
        return {"unit": "u", "src": str(self.f), "rc": 0, "files": [str(self.f)], "kernels": k}

    def test_judge(self):
        f, _, st = lim.judge(self.unit(64), {}, str)
        kinds = sorted(x[0] for x in f)
        self.assertEqual(kinds, ["bounds"])  # tk<<<256>>> past its 128; the 64 KiB launch has its opt-in
        self.assertEqual(st["resolved"], 3)
        f, _, _ = lim.judge(self.unit(72), {}, str)
        self.assertIn("regfile", [x[0] for x in f])  # 72 regs x 1024 threads

    def test_spill_baseline(self):
        base = {("u", "_Z2tkILi4EEvPf"): (8, 8)}
        f, notes, _ = lim.judge(self.unit(32, (8, 8)), base, str)
        self.assertNotIn("spill", [x[0] for x in f])
        f, _, _ = lim.judge(self.unit(32, (16, 8)), base, str)
        self.assertIn("spill", [x[0] for x in f])
        f, notes, _ = lim.judge(self.unit(32, (0, 0)), base, str)
        self.assertNotIn("spill", [x[0] for x in f])
        self.assertTrue(any("improved" in n for n in notes))
        f, _, _ = lim.judge(self.unit(32, (4, 4)), {}, str)
        self.assertIn("spill", [x[0] for x in f])  # a new spiller

    def test_compile_error(self):
        r = {"unit": "u", "src": str(self.f), "rc": 1, "errors": ["ptxas error   : Entry function 'x' uses too much shared data (0x19000 bytes, 0xc000 max)"]}
        self.assertEqual(lim.judge(r, {}, str)[0][0][0], "smem")
        r["errors"] = ["k.cu(3): error: identifier \"x\" is undefined"]
        self.assertEqual(lim.judge(r, {}, str)[0][0][0], "compile")


class Dups(unittest.TestCase):
    def test_drift_and_shape(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = (pathlib.Path(d) / n for n in ("a.cu", "b.cuh", "c.cu"))
            a.write_text("__device__ __forceinline__ float wsum(float v) {\n"
                         "  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); return v; }\n"
                         "__global__ void k(float *y) { y[0] = wsum(y[0]); }\n")
            b.write_text("static __device__ float wsum(float v) {  // the same, comment and static aside\n"
                         "  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);\n  return v; }\n"
                         "__device__ float warp_total(float a) {\n"
                         "  for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o); return a; }\n")
            c.write_text("__device__ float wsum(float v) {\n"
                         "  for (int o = 16; o; o >>= 1) v += __shfl_down_sync(~0u, v, o); return v; }\n")
            dups.SRC = pathlib.Path(d)
            defs = dups.scan([a, b, c])
            self.assertEqual(sorted(x["name"] for x in defs), ["warp_total", "wsum", "wsum", "wsum"])
            drifted, same, renamed, shaped, families = dups.report(defs, 4)
            self.assertEqual([x[0] for x in drifted], ["wsum"])
            groups = {x["fp"] for x in drifted[0][2]}
            self.assertEqual(len(groups), 2)  # a and b identical, c drifted (shfl_down)
            self.assertEqual(renamed, [])
            self.assertEqual(len(shaped), 1)  # warp_total is wsum with its variable renamed
            self.assertEqual({x["name"] for x in shaped[0]}, {"wsum", "warp_total"})


if __name__ == "__main__":
    unittest.main()
