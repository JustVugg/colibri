"""Build-contract checks for the real-GPU MXFP4 correctness test."""
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent


def cuda_test_recipe(*variables):
    result = subprocess.run(
        ["make", "-Bn", "cuda-test",
         "TRIPLET=x86_64-unknown-linux-gnu", *variables],
        cwd=HERE, text=True, capture_output=True, check=False, timeout=120)
    return result.stdout + result.stderr


def mxfp4_link_line(recipe):
    return next(
        (line for line in recipe.splitlines()
         if "tests/test_mxfp4_cuda.cu" in line and " -o " in line),
        "")


class CudaTestMakefileTest(unittest.TestCase):
    def test_nvcc_links_the_cpu_oracle_openmp_runtime(self):
        line = mxfp4_link_line(cuda_test_recipe("CUDA=1", "NVCC=nvcc"))
        self.assertTrue(line, "no MXFP4 CUDA link command in dry-run recipe")
        self.assertIn("-Xcompiler=-fopenmp", line)

    def test_hipcc_links_the_cpu_oracle_openmp_runtime(self):
        line = mxfp4_link_line(cuda_test_recipe(
            "HIP=1", "HIP_ARCH=gfx1100", "HIPCC=hipcc"))
        self.assertTrue(line, "no MXFP4 HIP link command in dry-run recipe")
        self.assertIn("-fopenmp", line)
        self.assertNotIn("-Xcompiler=-fopenmp", line)

    def test_setup_openmp_probe_does_not_require_tmp(self):
        setup = (HERE / "setup.sh").read_text(encoding="utf-8")
        self.assertNotIn("/tmp/_omp", setup)
        self.assertIn('OMP_PROBE=".colibri-omp-probe-$$"', setup)
        self.assertIn('rm -f "$OMP_PROBE" "$OMP_PROBE.exe"', setup)


if __name__ == "__main__":
    unittest.main()
