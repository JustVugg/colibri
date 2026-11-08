"""Exercise the shell harness's actual counters without requiring a GPU."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest


# Windows: shutil.which("bash") can return System32's WSL shim, which has no
# sed/tail. Prefer the bash of the Git for Windows that provides `git`
# (<root>\cmd\git.exe next to <root>\bin\bash.exe); never take a System32 one.
def _find_bash():
    git = shutil.which("git")
    candidates = [str(Path(git).parents[1] / "bin" / "bash.exe") if git else None,
                  shutil.which("bash")]
    return next((c for c in candidates
                 if c and os.path.exists(c) and "system32" not in c.lower()), None)


BASH = _find_bash()
SCRIPT = Path(__file__).with_name("vulkan_engines.sh").read_text()
# Load only these production helpers: sourcing the full script runs a family.
HELPERS = re.search(r"^fail\(\).*?$", SCRIPT, re.M).group() + "\n"
for name in ("vk_count", "need_gpu", "need_int8"):
    HELPERS += re.search(rf"^{name}\(\).*?^\}}", SCRIPT, re.M | re.S).group() + "\n"
# qwen36's per-matrix report (vk_report) and its dense chain's, as the int8 arm reads them.
Q36_REPORT = ("[VK] qwen36: {n} matmuls on the GPU (73 matrices resident, 1.0 MiB; "
              "placed int8 {i8}, int4 0, f32 3)\n")
Q36_CHAIN = ("[VK] qwen36 chain: 16 forwards, 311 frames (2176 ops, 1168 matmuls, "
             "0 tiled GEMM), timings\n")


@unittest.skipUnless(BASH, "the Vulkan harness requires bash")
class VulkanEngineCounts(unittest.TestCase):
    def run_helper(self, log, engine="qwen36", helper="vk_count"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(log)
            return subprocess.run(
                [BASH, "-eu", "-o", "pipefail", "-c",
                 HELPERS + f'{helper} "$1" "$2" regression',
                 "counts-test", engine, path.as_posix()],
                capture_output=True, text=True, check=False,
            )

    def test_legacy_per_matrix_count(self):
        result = self.run_helper("[VK] qwen36: 7 matmuls on the GPU (73 matrices resident)\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "7")

    def test_chain_only_satisfies_gpu_gate(self):
        for engine, count in (("qwen36", 1168), ("olmoe", 168), ("mimo", 96)):
            with self.subTest(engine=engine):
                log = (f"[VK] {engine}: 0 matmuls on the GPU\n"
                       f"[VK] {engine} chain: 6 forwards, 63 frames (360 ops, "
                       f"{count} matmuls, 4 tiled GEMM), timings\n")
                result = self.run_helper(log, engine)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), str(count))
                self.assertEqual(self.run_helper(log, engine, "need_gpu").returncode, 0)

    def test_latest_cumulative_reports_are_added_once(self):
        log = ("[VK] qwen36: 1 matmuls on the GPU\n"
               "[VK] qwen36 chain: 1 forwards, 3 frames (50 ops, 3 matmuls, 1 tiled GEMM)\n"
               "[VK] qwen36: 2 matmuls on the GPU\n"
               "[VK] qwen36 chain: 2 forwards, 6 frames (100 ops, 9 matmuls, 2 tiled GEMM)\n")
        result = self.run_helper(log)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "11")

    def test_unrelated_engine_does_not_satisfy_gpu_gate(self):
        log = ("[VK] olmoe: 999 matmuls on the GPU\n"
               "[VK] olmoe chain: 1 forwards, 3 frames (50 ops, 99 matmuls, 0 tiled GEMM)\n")
        self.assertEqual(self.run_helper(log).stdout.strip(), "0")
        self.assertNotEqual(self.run_helper(log, helper="need_gpu").returncode, 0)

    def test_empty_and_zero_logs_still_fail_gpu_gate(self):
        for log in ("", "[VK] qwen36: 0 matmuls on the GPU\n",
                    "[VK] qwen36 chain: 1 forwards, 3 frames (50 ops, 0 matmuls, 0 tiled GEMM)\n"):
            with self.subTest(log=log):
                self.assertEqual(self.run_helper(log).stdout.strip(), "0")
                self.assertNotEqual(self.run_helper(log, helper="need_gpu").returncode, 0)

    def test_int8_gate_counts_the_chain(self):
        # #1862: with the dense chain on (a discrete GPU's default) the per-matrix count
        # is 0; the chain's matmuls and its int8 uploads must satisfy the gate.
        for log in (Q36_REPORT.format(n=0, i8=70) + Q36_CHAIN,
                    Q36_REPORT.format(n=7, i8=70)):
            with self.subTest(log=log):
                result = self.run_helper(log, helper="need_int8")
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_int8_gate_still_needs_int8_rows_and_matmuls(self):
        for log, why in ((Q36_REPORT.format(n=0, i8=0) + Q36_CHAIN, "no int8 matrix placed"),
                         (Q36_CHAIN, "no int8 matrix placed"),
                         (Q36_REPORT.format(n=0, i8=70), "no matmul ran on the device"),
                         ("", "no matmul ran on the device"),
                         # only the engine's own report line counts, as in vk_count: one that
                         # starts mid-line, or has no matmul count, is not that report
                         ("note: " + Q36_REPORT.format(n=0, i8=70) + Q36_CHAIN,
                          "no int8 matrix placed"),
                         (Q36_REPORT.format(n="", i8=70) + Q36_CHAIN, "no int8 matrix placed")):
            with self.subTest(log=log):
                result = self.run_helper(log, helper="need_int8")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"FAIL: regression: {why}", result.stdout)


if __name__ == "__main__":
    unittest.main()
