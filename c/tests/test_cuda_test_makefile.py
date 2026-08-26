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


def mxfp4_ref_compile_line(recipe):
    return next(
        (line for line in recipe.splitlines()
         if "tests/mxfp4_ref.c" in line and " -c " in line),
        "")


class CudaTestMakefileTest(unittest.TestCase):
    # These three replace the pair that asserted the OpenMP runtime WAS linked.
    # The CPU oracle is now built without OpenMP (#971), because that single
    # dependency broke `make cuda-test` on a host with no libgomp - the link
    # died on undefined reference to GOMP_parallel, out of mxfp4_ref.o. The
    # oracle's largest case is S=4 I=2048 O=64, so nothing measurable was lost.
    def test_cpu_oracle_is_compiled_without_openmp(self):
        """The load-bearing assertion: re-adding OpenMP here would silently
        reintroduce the link dependency that made the suite unbuildable."""
        line = mxfp4_ref_compile_line(cuda_test_recipe("CUDA=1", "NVCC=nvcc"))
        self.assertTrue(line, "no MXFP4 oracle compile command in dry-run recipe")
        self.assertIn("-fno-openmp", line)

    def test_nvcc_does_not_link_an_openmp_runtime_for_the_oracle(self):
        line = mxfp4_link_line(cuda_test_recipe("CUDA=1", "NVCC=nvcc"))
        self.assertTrue(line, "no MXFP4 CUDA link command in dry-run recipe")
        self.assertNotIn("-fopenmp", line)

    def test_hipcc_does_not_link_an_openmp_runtime_for_the_oracle(self):
        line = mxfp4_link_line(cuda_test_recipe(
            "HIP=1", "HIP_ARCH=gfx1100", "HIPCC=hipcc"))
        self.assertTrue(line, "no MXFP4 HIP link command in dry-run recipe")
        self.assertNotIn("-fopenmp", line)

    def test_setup_openmp_probe_does_not_require_tmp(self):
        setup = (HERE / "setup.sh").read_text(encoding="utf-8")
        self.assertNotIn("/tmp/_omp", setup)
        self.assertIn('OMP_PROBE=".colibri-omp-probe-$$"', setup)
        self.assertIn('rm -f "$OMP_PROBE" "$OMP_PROBE.exe"', setup)


if __name__ == "__main__":
    unittest.main()
