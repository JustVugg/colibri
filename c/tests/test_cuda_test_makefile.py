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


def nvcc_compile_line(recipe):
    """First real nvcc invocation, skipping the `command -v "nvcc"` guard line
    that a naive grep for "nvcc" matches first."""
    return next(
        (line for line in recipe.splitlines()
         if "backend_cuda.cu" in line and " -o " in line and "command -v" not in line),
        "")


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

    def test_allow_unsupported_compiler_is_off_by_default(self):
        """nvcc's own wording for this flag is "may cause compilation failure or
        incorrect run time execution", so it is never a default."""
        line = nvcc_compile_line(cuda_test_recipe("CUDA=1", "NVCC=nvcc"))
        self.assertTrue(line, "no nvcc compile command in dry-run recipe")
        self.assertNotIn("-allow-unsupported-compiler", line)

    def test_allow_unsupported_compiler_is_reachable(self):
        """CUDA 13.1's host_config.h rejects any MSVC newer than VS2022, which
        fails every nvcc target at the first #include with C1189. Before this
        variable the only handle was NVCCFLAGS, and overriding that wholesale
        drops the gencode, the -ccbin and the platform warning form."""
        line = nvcc_compile_line(
            cuda_test_recipe("CUDA=1", "NVCC=nvcc", "NVCC_ALLOW_UNSUPPORTED=1"))
        self.assertTrue(line, "no nvcc compile command in dry-run recipe")
        self.assertIn("-allow-unsupported-compiler", line)

    def test_allow_unsupported_compiler_preserves_the_other_flags(self):
        """The load-bearing one. Appending must not do what a wholesale
        NVCCFLAGS override does, which is silently drop the arch and lose
        -ftz=false -- and -ftz is a correctness flag here, not a tuning knob:
        the fmt=8 kernels are cross-tier parity instruments and a flushed
        subnormal diverges from the CPU reference."""
        line = nvcc_compile_line(
            cuda_test_recipe("CUDA=1", "NVCC=nvcc", "CUDA_ARCH=sm_86",
                             "NVCC_ALLOW_UNSUPPORTED=1"))
        self.assertTrue(line, "no nvcc compile command in dry-run recipe")
        for flag in ("-ftz=false", "-std=c++17", "sm_86"):
            self.assertIn(flag, line, f"{flag} lost when the override is set")

    def test_setup_openmp_probe_does_not_require_tmp(self):
        setup = (HERE / "setup.sh").read_text(encoding="utf-8")
        self.assertNotIn("/tmp/_omp", setup)
        self.assertIn('OMP_PROBE=".colibri-omp-probe-$$"', setup)
        self.assertIn('rm -f "$OMP_PROBE" "$OMP_PROBE.exe"', setup)


if __name__ == "__main__":
    unittest.main()
