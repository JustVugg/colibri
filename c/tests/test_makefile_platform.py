import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
MAKE = shutil.which("make")


@unittest.skipUnless(MAKE, "make is required")
class MakefilePlatformTests(unittest.TestCase):
    def _dry_run(self, target, triplet, **variables):
        args = [
            MAKE,
            "--no-print-directory",
            "-B",
            "-n",
            target,
            f"TRIPLET={triplet}",
        ]
        args.extend(f"{name}={value}" for name, value in variables.items())
        return subprocess.run(
            args,
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_cuda_glibc_compat_opt_out_preserves_other_flags(self):
        default = self._dry_run("backend_cuda.o", "x86_64-unknown-linux-gnu",
                                CUDA=1, CUDA_ARCH="sm_86")
        disabled = self._dry_run("backend_cuda.o", "x86_64-unknown-linux-gnu",
                                 CUDA=1, CUDA_ARCH="sm_86", CUDA_GLIBC_MATH_COMPAT=0)
        before = next(line for line in default.stdout.splitlines()
                      if "-c backend_cuda.cu" in line)
        after = next(line for line in disabled.stdout.splitlines()
                     if "-c backend_cuda.cu" in line)
        # make spells its directory its own way (/d/a/... under MSYS2, where
        # Python has D:/a/...), so the flag is the one naming the header.
        compat = [f for f in before.split()
                  if f.startswith("-Xcompiler=-include,")
                  and f.endswith(f"/{C_DIR.name}/glibc_c23_math_compat.h")]
        self.assertEqual(len(compat), 1, before)
        flag = compat[0]
        self.assertNotIn("glibc_c23_math_compat.h", after)
        self.assertEqual(before.replace(flag, "").split(), after.split())
        stamp = (C_DIR / ".build-config").read_text()
        self.assertIn("CUDA_GLIBC_MATH_COMPAT=0", stamp)

    def test_windows_nt_without_uname_selects_mingw_build(self):
        env = os.environ.copy()
        env["OS"] = "Windows_NT"
        env["PATH"] = ""

        result = subprocess.run(
            [MAKE, "--no-print-directory", "-B", "-n", "colibri"],
            cwd=C_DIR,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("-o colibri.exe", result.stdout)
        self.assertIn("-fopenmp", result.stdout)
        self.assertIn("-static", result.stdout)

    def test_portable_build_uses_target_architecture(self):
        cases = (
            ("x86_64-unknown-linux-gnu", "-march=x86-64-v3"),
            ("aarch64-unknown-linux-gnu", "-march=armv8-a"),
            ("powerpc64le-unknown-linux-gnu", "-mcpu=power8"),
            ("ppc64le-unknown-linux-gnu", "-mcpu=power8"),
        )

        for triplet, expected_flag in cases:
            with self.subTest(triplet=triplet):
                result = self._dry_run("portable", triplet)
                self.assertIn(expected_flag, result.stdout)

    def test_vk_links_no_vulkan_loader(self):
        # The loader is opened at run time (vk_load.h): a VK=1 binary has to
        # start where there is none, as the release archives' do. Linux takes
        # -ldl for dlopen; Windows stays fully -static (LoadLibrary is kernel32).
        cases = (
            ("x86_64-unknown-linux-gnu", "-o colibri ", ["-pthread", "-ldl"]),
            ("x86_64-w64-mingw32", "-o colibri.exe ", ["-static", "-lpsapi"]),
        )

        for triplet, output, tail in cases:
            with self.subTest(triplet=triplet):
                result = self._dry_run("colibri", triplet, VK=1)
                link = next(
                    line for line in result.stdout.splitlines() if output in line
                )
                self.assertEqual(link.split()[-len(tail):], tail)
                self.assertNotIn("-lvulkan", link)
                self.assertNotIn("-Bdynamic", link)

    def test_clang_mingw_triple_builds_for_windows(self):
        # clang names the MinGW target x86_64-w64-windows-gnu, gcc x86_64-w64-mingw32 (#2286)
        result = self._dry_run("colibri", "x86_64-w64-windows-gnu")
        link = next(
            line for line in result.stdout.splitlines() if "-o colibri.exe " in line
        )
        self.assertEqual(link.split()[-2:], ["-static", "-lpsapi"])

    def test_deepseek_v4_windows_link_per_compiler(self):
        # clang links libomp.a: no -pthread (its driver adds winpthreads), no LTO by
        # default (GNU ld lacks LLVM's plugin); gcc's line stays as it was (#2286)
        cases = (
            ("gcc", ["-static", "-pthread", "-flto"]),
            ("clang", ["-lm", "-fopenmp", "-static"]),
        )
        for cc, tail in cases:
            with self.subTest(cc=cc):
                result = subprocess.run(
                    [MAKE, "--no-print-directory", "-B", "-n", "-f",
                     "Makefile.deepseek-v4", "deepseek-v4", "OS=Windows_NT",
                     f"CC={cc}"],
                    cwd=C_DIR, text=True, capture_output=True, check=True,
                )
                link = next(
                    line for line in result.stdout.splitlines()
                    if " -o deepseek_v4.exe " in line
                )
                self.assertEqual(link.split()[0], cc)
                self.assertEqual(link.split()[-3:], tail)

    def test_deepseek_v4_clang_stub_only_when_num_devices_is_unresolved(self):
        # libomp.a's weak omp_get_num_devices stays unresolved with MSYS2's 22.1.7 or
        # binutils 2.46; where it links, no second definition is added (#2286)
        undefined = "libomp.a(kmp_alloc.cpp.obj): undefined reference to `omp_get_num_devices'"
        with tempfile.TemporaryDirectory() as tmp:
            for libomp, error in (("22", undefined), ("23", "")):
                with self.subTest(libomp=libomp):
                    fake = Path(tmp) / f"clang-{libomp}"
                    fake.write_text(
                        "#!/bin/sh\n"
                        '[ "$1" = -dumpmachine ] && { echo x86_64-w64-windows-gnu; exit 0; }\n'
                        f"cat <<'EOF'\n{error}\nEOF\n"
                    )
                    fake.chmod(0o755)
                    result = subprocess.run(
                        [MAKE, "--no-print-directory", "-B", "-n", "-f",
                         "Makefile.deepseek-v4", "backend_loader_dsv4.o",
                         "OS=Windows_NT", f"CC={fake.as_posix()}"],
                        cwd=C_DIR, text=True, capture_output=True, check=True,
                    )
                    compile_line = next(
                        line for line in result.stdout.splitlines()
                        if "-c backend_loader_dsv4.c" in line
                    )
                    self.assertEqual("-DCOLI_OMP_NUM_DEVICES_STUB" in compile_line,
                                     libomp == "22", compile_line)

    def test_darwin_portable_build_does_not_force_x86_architecture(self):
        missing_libomp = "/colibri-test/missing-libomp"
        result = self._dry_run(
            "portable", "arm64-apple-darwin", OMPDIR=missing_libomp
        )

        self.assertIn("clang -O3", result.stdout)
        self.assertNotIn("-mcpu=x86-64-v3", result.stdout)
        self.assertNotIn(missing_libomp, result.stdout)
        self.assertNotIn("-fopenmp", result.stdout)


if __name__ == "__main__":
    unittest.main()
