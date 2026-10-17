"""Probe the selected compiler's include search, without a Vulkan SDK or GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_flow


class VulkanHeaderProbe(unittest.TestCase):
    def setUp(self):
        self.compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not self.compiler:
            self.skipTest("a C compiler is required")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.headers = self.root / "custom SDK" / "include"
        (self.headers / "vulkan").mkdir(parents=True)
        self.header = self.headers / "vulkan" / "vulkan.h"
        self.header.write_text("typedef unsigned int VkFlags;\n")
        (self.root / "Makefile").write_text("# source checkout\n")

    def toolchain(self, compiler=None):
        tools = {"gcc": compiler or self.compiler, "make": "make", "glslc": "glslc"}
        with mock.patch.object(setup_flow.sys, "platform", "linux"), \
                mock.patch.object(setup_flow.shutil, "which", side_effect=tools.get), \
                mock.patch.object(setup_flow, "_nvcc", return_value=None):
            return setup_flow.toolchain(str(self.root))

    def test_compiler_include_environment_is_used(self):
        with mock.patch.dict(os.environ, {"CPATH": str(self.headers)}):
            tc = self.toolchain()
        self.assertTrue(tc["vulkan_headers"])
        self.assertTrue(tc["can_build_vulkan"])

    @unittest.skipUnless(os.name == "posix", "Nix-style executable wrappers use POSIX")
    def test_compiler_wrapper_include_directory_is_used(self):
        wrapper = self.root / "compiler wrapper"
        wrapper.write_text("#!" + sys.executable + "\nimport os, sys\nos.execv(" +
                           repr(self.compiler) + ", [" + repr(self.compiler) +
                           ", '-I', " + repr(str(self.headers)) + "] + sys.argv[1:])\n")
        wrapper.chmod(0o755)
        self.assertTrue(self.toolchain(str(wrapper))["can_build_vulkan"])

    def test_header_that_compiler_rejects_is_not_available(self):
        self.header.write_text("#error broken Vulkan installation\n")
        with mock.patch.dict(os.environ, {"CPATH": str(self.headers)}):
            self.assertFalse(self.toolchain()["vulkan_headers"])

    @unittest.skipUnless(os.name == "posix", "Nix-style executable wrappers use POSIX")
    def test_compiler_without_header_search_paths_rejects_missing_header(self):
        wrapper = self.root / "isolated compiler"
        wrapper.write_text("#!" + sys.executable + "\nimport os, sys\nos.execv(" +
                           repr(self.compiler) + ", [" + repr(self.compiler) +
                           ", '-nostdinc'] + sys.argv[1:])\n")
        wrapper.chmod(0o755)
        with mock.patch.dict(os.environ, {"CPATH": "", "C_INCLUDE_PATH": ""}):
            self.assertFalse(self.toolchain(str(wrapper))["can_build_vulkan"])

    def test_stalled_compiler_does_not_block_setup(self):
        with mock.patch.object(setup_flow.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired(self.compiler, 5)):
            self.assertFalse(self.toolchain()["can_build_vulkan"])

    def test_macos_does_not_probe_or_offer_vulkan(self):
        with mock.patch.object(setup_flow.sys, "platform", "darwin"), \
                mock.patch.object(setup_flow, "_nvcc", return_value=None), \
                mock.patch.object(setup_flow.subprocess, "run") as probe:
            self.assertFalse(setup_flow.toolchain(str(self.root))["can_build_vulkan"])
        probe.assert_not_called()

    def test_unavailable_compiler_does_not_offer_vulkan(self):
        tc = self.toolchain(str(self.root / "missing compiler"))
        self.assertFalse(tc["vulkan_headers"])
        self.assertFalse(tc["can_build_vulkan"])


if __name__ == "__main__":
    unittest.main()
