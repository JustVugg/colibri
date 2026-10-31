"""The one-step setup flow: backend choice, run configuration, engine
resolution, the whole `coli setup` against a fake hub, and start/status/stop.

No network, no real model, no real build: the hub is tests/fake_hub.py, the
hardware report is canned, the engine is a file that names the Vulkan loader,
and the server is a twenty-line stand-in that answers /health like `coli web`
does and writes the same pidfile, so the real `coli stop` can stop it.

Every test that depends on an OS models it with modeled(): it patches the
setup's host_os()/host_machine(), never sys.platform, so a Linux test runs the
same on Windows and a Windows test the same on Linux (engine names, release
assets, package commands, the make invocation).
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import tarfile
import shutil
import tempfile
import threading
import time
import unittest
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
C_DIR = TESTS.parent
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(TESTS))
import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
import setup_flow  # noqa: E402
import setup_hw  # noqa: E402
import family_registry  # noqa: E402
from fake_hub import FakeHub  # noqa: E402
from family_registry import family_by_id  # noqa: E402

REPO = "tester/tiny-model"


def modeled(os_name, machine="x86_64"):
    """Run the setup as if on `os_name` ("linux", "win32", "darwin"). The
    patches apply from the call; the returned stack undoes them on close or
    at the end of a `with`."""
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(setup_flow, "host_os", return_value=os_name))
    stack.enter_context(mock.patch.object(setup_flow, "host_machine", return_value=machine))
    stack.enter_context(mock.patch.object(setup_hw, "host_os", return_value=os_name))
    return stack


def engine_name(os_name):
    return "qwen36.exe" if os_name == "win32" else "qwen36"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def quiet_port_pair(low=20000, high=30000):
    """A free port whose next one is free too, below every kernel's ephemeral range
    (32768 and up on Linux, 49152 and up on Windows and macOS). free_port() lands
    inside that range, where an outgoing connection can take the next port between
    pick_port's probe and the server's bind (seen once in CI as EADDRINUSE)."""
    import random
    def bindable(port):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False
    rng = random.Random()
    for _ in range(200):
        port = rng.randrange(low, high)
        if bindable(port) and bindable(port + 1):
            return port
    raise unittest.SkipTest("no two consecutive free ports below the ephemeral range")


def hw_report(vulkan=None, nvidia=(), icd=None, os_id="ubuntu"):
    return {
        "os": {"platform": "linux", "machine": "x86_64", "wsl": False, "id": os_id,
               "id_like": "debian", "pretty_name": "Ubuntu 24.04"},
        "cpu": {"name": "Test CPU", "features": ["avx2"], "physical_cores": 6, "logical_cores": 12,
                "arch": "x86_64"},
        "memory": {"total": 27 * 10**9, "available": 24 * 10**9},
        "disk": None, "vulkan": {"devices": [vulkan] if vulkan else [], "icd": icd},
        "nvidia": list(nvidia), "windows_video": [],
        "gpu": {"vulkan": vulkan, "vulkan_icd": icd, "nvidia": list(nvidia),
                "has_gpu": bool(vulkan or nvidia)},
    }


IGPU = {"name": "Iris Xe", "type": "integrated", "api_version": "1.2.318",
        "api_version_raw": (1 << 22) | (2 << 12) | 318}
DGPU = {"name": "AMD Radeon RX 7800 XT", "type": "discrete", "api_version": "1.3.296",
        "api_version_raw": (1 << 22) | (3 << 12) | 296}
RTX = {"index": 0, "name": "NVIDIA GeForce RTX 4070", "total_bytes": 12 * 2**30,
       "free_bytes": 11 * 2**30, "driver": "560"}
TC_ALL = {"source_checkout": True, "make": "/usr/bin/make", "cc": "/usr/bin/gcc",
          "glslc": "/usr/bin/glslc", "vulkan_headers": True, "nvcc": "/usr/local/cuda/bin/nvcc",
          "msys2": None, "npm": None, "can_build": True, "can_build_vulkan": True,
          "can_build_cuda": True}


def setup_args(**overrides):
    values = dict(yes=True, json=False, pick=None, ctx=None, model_dir=None, dir=None, backend=None,
                  no_gpu=False, host=None, port=None, no_start=True, background=False,
                  no_browser=True, reconfigure=False, list=False, all=False, via_windows="no",
                  no_verify=False)
    values.update(overrides)
    return argparse.Namespace(**values)


class HomeTestCase(unittest.TestCase):
    """Each test gets its own setup folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = os.path.join(self.tmp.name, "home")
        patcher = mock.patch.dict(os.environ, {"COLI_SETUP_HOME": self.home})
        patcher.start()
        self.addCleanup(patcher.stop)


class BackendChoice(unittest.TestCase):
    def choose(self, hw, family="qwen36", tc=None, requested="auto", platform="linux", here=None):
        """choose_backend with `here` as the engines' directory (default: an empty one,
        so an engine built in this checkout does not decide the answer)."""
        with modeled(platform), tempfile.TemporaryDirectory() as empty, \
                mock.patch.object(setup_flow, "HERE", here or empty):
            return setup_flow.choose_backend(hw, family_by_id(family), tc or TC_ALL, requested)

    def test_no_gpu_is_cpu(self):
        decision = self.choose(hw_report())
        self.assertEqual(decision["backend"], "cpu")
        self.assertIn("no GPU", decision["reason"])

    def test_integrated_vulkan_gpu(self):
        decision = self.choose(hw_report(vulkan=IGPU))
        self.assertEqual(decision["backend"], "vulkan")
        self.assertEqual(decision["gpu"], "Iris Xe")

    def test_nvidia_with_toolkit_takes_cuda(self):
        self.assertEqual(self.choose(hw_report(vulkan=IGPU, nvidia=[RTX]))["backend"], "cuda")
        # The Windows CUDA build is a separate MSVC build: Vulkan there.
        self.assertEqual(self.choose(hw_report(vulkan=IGPU, nvidia=[RTX]), platform="win32")["backend"],
                         "vulkan")

    def test_nvidia_without_toolkit_falls_to_vulkan_and_says_how(self):
        tc = dict(TC_ALL, nvcc=None, can_build_cuda=False)
        decision = self.choose(hw_report(vulkan=IGPU, nvidia=[RTX]), tc=tc)
        self.assertEqual(decision["backend"], "vulkan")
        self.assertEqual(decision["missing"][0][0], "cuda")
        self.assertIn("nvidia-cuda-toolkit", decision["missing"][0][1])

    def test_engine_without_cuda_path_uses_vulkan(self):
        rtx_vk = {"name": RTX["name"], "type": "discrete", "api_version": "1.3.289",
                  "api_version_raw": (1 << 22) | (3 << 12) | 289}
        decision = self.choose(hw_report(vulkan=rtx_vk, nvidia=[RTX]), family="olmoe")
        self.assertEqual(decision["backend"], "vulkan")

    def test_integrated_gpu_only_for_engines_measured_faster_there(self):
        for family in sorted(setup_flow.VULKAN_IGPU_MEASURED):
            with self.subTest(family=family):
                self.assertEqual(self.choose(hw_report(vulkan=IGPU), family=family)["backend"], "vulkan")
        for family in ("mimo", "glm", "inkling", "kimi", "deepseek_v4"):
            with self.subTest(family=family):
                decision = self.choose(hw_report(vulkan=IGPU), family=family)
                self.assertEqual(decision["backend"], "cpu")
                self.assertIn("integrated GPU", decision["reason"])
                self.assertIn("--backend vulkan", decision["reason"])
                self.assertEqual(decision["missing"], [])   # nothing to install: a choice, not a lack

    def test_integrated_gpu_when_vulkan_is_asked_for(self):
        decision = self.choose(hw_report(vulkan=IGPU), family="mimo", requested="vulkan")
        self.assertEqual(decision["backend"], "vulkan")
        self.assertEqual(decision["gpu"], "Iris Xe")

    def test_discrete_gpu_runs_vulkan_for_every_engine(self):
        for family in ("qwen36", "qwen38", "mimo", "glm", "inkling", "kimi", "deepseek_v4"):
            with self.subTest(family=family):
                self.assertEqual(self.choose(hw_report(vulkan=DGPU), family=family)["backend"], "vulkan")

    def test_vulkan_gpu_without_headers_prints_the_package_command(self):
        tc = dict(TC_ALL, vulkan_headers=False, glslc=None, can_build_vulkan=False)
        decision = self.choose(hw_report(vulkan=IGPU), tc=tc)
        self.assertEqual(decision["backend"], "cpu")
        self.assertEqual(decision["missing"],
                         [("vulkan", "sudo apt install libvulkan-dev glslc mesa-vulkan-drivers")])

    def test_cpu_when_asked(self):
        self.assertEqual(self.choose(hw_report(vulkan=IGPU, nvidia=[RTX]), requested="cpu")["backend"],
                         "cpu")

    def test_windows_release_archive_with_a_cpu_engine_points_at_msys2(self):
        # an archive from before 2.0: its engine has no Vulkan and cannot be rebuilt
        tc = dict(TC_ALL, source_checkout=False, make=None, cc=None, can_build=False,
                  can_build_vulkan=False, can_build_cuda=False)
        with tempfile.TemporaryDirectory() as here:
            Path(here, "qwen36.exe").write_bytes(b"MZ plain")
            decision = self.choose(hw_report(vulkan=IGPU), tc=tc, platform="win32", here=here)
        self.assertEqual(decision["backend"], "cpu")
        hint = decision["missing"][0][1]
        self.assertIn("msys2.org", hint)
        self.assertIn("mingw-w64-ucrt-x86_64-gcc", hint)
        self.assertIn("mingw-w64-ucrt-x86_64-shaderc", hint)
        self.assertIn("source checkout", hint)      # a release archive cannot rebuild itself

    def test_a_vulkan_engine_here_needs_no_vulkan_toolchain(self):
        # a release archive since 2.0: Vulkan engines and their shaders, no compiler
        tc = dict(TC_ALL, source_checkout=False, make=None, cc=None, can_build=False,
                  can_build_vulkan=False, can_build_cuda=False)
        for os_name, content in (("linux", b"\x7fELF libvulkan.so.1"), ("win32", b"MZ vulkan-1.dll")):
            with self.subTest(os=os_name), tempfile.TemporaryDirectory() as here:
                Path(here, engine_name(os_name)).write_bytes(content)
                decision = self.choose(hw_report(vulkan=IGPU), tc=tc, platform=os_name, here=here)
                self.assertEqual(decision["backend"], "cpu")        # no shaders beside it
                os.makedirs(os.path.join(here, "shaders"))
                Path(here, "shaders", "qmatmul.spv").write_bytes(b"\x03\x02\x23\x07")
                decision = self.choose(hw_report(vulkan=IGPU), tc=tc, platform=os_name, here=here)
                self.assertEqual(decision["backend"], "vulkan")
                self.assertEqual(decision["missing"], [])

    def windows_cuda_here(self, here, dll=True):
        """The release's windows-x86_64-cuda.zip unpacked: CUDA and Vulkan engines,
        their shaders and coli_cuda.dll."""
        Path(here, "qwen36.exe").write_bytes(b"MZ coli_cuda.dll vulkan-1.dll")
        os.makedirs(os.path.join(here, "shaders"), exist_ok=True)
        Path(here, "shaders", "qmatmul.spv").write_bytes(b"\x03\x02\x23\x07")
        if dll:
            Path(here, "coli_cuda.dll").write_bytes(b"MZ")

    def test_windows_cuda_package_takes_cuda_on_an_ampere_card(self):
        tc = dict(TC_ALL, source_checkout=False, make=None, cc=None, can_build=False,
                  can_build_vulkan=False, can_build_cuda=False)
        rtx_vk = {"name": RTX["name"], "type": "discrete", "api_version": "1.3.289",
                  "api_version_raw": (1 << 22) | (3 << 12) | 289}
        with tempfile.TemporaryDirectory() as here:
            self.windows_cuda_here(here)
            card = dict(RTX, compute_cap="8.9")
            decision = self.choose(hw_report(vulkan=rtx_vk, nvidia=[card]), tc=tc, platform="win32", here=here)
            self.assertEqual(decision["backend"], "cuda")
            self.assertIn("coli_cuda.dll beside the engine", decision["reason"])
            # a card older than the DLL's sm_80: Vulkan, saying why
            old = dict(RTX, name="NVIDIA GeForce RTX 2070", compute_cap="7.5")
            decision = self.choose(hw_report(vulkan=rtx_vk, nvidia=[old]), tc=tc, platform="win32", here=here)
            self.assertEqual(decision["backend"], "vulkan")
            self.assertIn("compute 8.0 and newer, and the NVIDIA GeForce RTX 2070 is compute 7.5",
                          decision["reason"])
            # unless CUDA is asked for by name
            decision = self.choose(hw_report(vulkan=rtx_vk, nvidia=[old]), tc=tc, platform="win32",
                                   here=here, requested="cuda")
            self.assertEqual(decision["backend"], "cuda")
            # an engine without a CUDA path in that archive: Vulkan
            decision = self.choose(hw_report(vulkan=rtx_vk, nvidia=[card]), family="mimo", tc=tc,
                                   platform="win32", here=here)
            self.assertNotEqual(decision["backend"], "cuda")
        with tempfile.TemporaryDirectory() as here:   # the engines without the DLL
            self.windows_cuda_here(here, dll=False)
            decision = self.choose(hw_report(vulkan=rtx_vk, nvidia=[dict(RTX, compute_cap="8.9")]),
                                   tc=tc, platform="win32", here=here)
            self.assertEqual(decision["backend"], "vulkan")
            self.assertEqual(decision["missing"][0][0], "cuda")
            self.assertIn("windows-x86_64-cuda.zip", decision["missing"][0][1])

    def test_windows_without_engines_takes_the_release_cuda_backend(self):
        # #2091: with nothing here, the setup takes the Windows release, whose CUDA
        # backend it fetches too (fetch_release_cuda); before, it took only the Vulkan
        # engines and an RTX 5080 ran Vulkan
        tc = dict(TC_ALL, source_checkout=True, make=None, cc=None, can_build=False,
                  can_build_vulkan=False, can_build_cuda=False)
        card = dict(RTX, name="NVIDIA GeForce RTX 5080", compute_cap="12.0")
        decision = self.choose(hw_report(vulkan=DGPU, nvidia=[card]), family="kimi", tc=tc, platform="win32")
        self.assertEqual(decision["backend"], "cuda")
        self.assertEqual(decision["gpu"], card["name"])
        self.assertIn("the release's CUDA backend (windows-x86_64-cuda.zip)", decision["reason"])
        old = dict(RTX, name="NVIDIA GeForce RTX 2070", compute_cap="7.5")
        decision = self.choose(hw_report(vulkan=DGPU, nvidia=[old]), family="kimi", tc=tc, platform="win32")
        self.assertEqual(decision["backend"], "vulkan")
        self.assertIn("the release's coli_cuda.dll runs on compute 8.0", decision["reason"])
        # an engine the release carries no CUDA build of, or a compiler here: as before
        decision = self.choose(hw_report(vulkan=DGPU, nvidia=[card]), family="mimo", tc=tc, platform="win32")
        self.assertEqual(decision["backend"], "vulkan")
        decision = self.choose(hw_report(vulkan=DGPU, nvidia=[card]), family="kimi", platform="win32")
        self.assertEqual(decision["backend"], "vulkan")
        self.assertEqual(decision["missing"][0][0], "cuda")
        # and on Linux the release has no CUDA backend to fetch
        decision = self.choose(hw_report(vulkan=DGPU, nvidia=[card]), family="kimi", tc=tc)
        self.assertNotEqual(decision["backend"], "cuda")

    def test_no_compiler_counts_on_the_release_vulkan_engine(self):
        # nothing here builds and no engine is here: resolve_engine downloads the
        # release's, which are Vulkan builds on Linux and Windows x86_64
        tc = dict(TC_ALL, make=None, cc=None, can_build=False, can_build_vulkan=False,
                  can_build_cuda=False)
        for os_name in ("linux", "win32"):
            with self.subTest(os=os_name):
                self.assertEqual(self.choose(hw_report(vulkan=DGPU), tc=tc, platform=os_name)["backend"],
                                 "vulkan")
        with modeled("linux", "aarch64"), tempfile.TemporaryDirectory() as empty, \
                mock.patch.object(setup_flow, "HERE", empty):   # no release for it
            decision = setup_flow.choose_backend(hw_report(vulkan=DGPU), family_by_id("qwen36"), tc)
        self.assertEqual(decision["backend"], "cpu")


#: What NVIDIA's own nvcc prints, measured: 11.x and 12.x from the
#: redistributable archives (developer.download.nvidia.com/compute/cuda/redist/
#: cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-<version>-archive.tar.xz), 13.x
#: from NVIDIA's nvidia-cuda-nvcc wheels. The release line of `nvcc --version`, and
#: `nvcc --list-gpu-arch` with its one-per-line output joined by spaces. The
#: order is nvcc's own (13.x does not sort).
NVCC_SAMPLES = {
    "11.4.152": ("Cuda compilation tools, release 11.4, V11.4.152",
                 "compute_35 compute_37 compute_50 compute_52 compute_53 compute_60 compute_61 compute_62 "
                 "compute_70 compute_72 compute_75 compute_80 compute_86 compute_87"),
    "11.8.89": ("Cuda compilation tools, release 11.8, V11.8.89",
                "compute_35 compute_37 compute_50 compute_52 compute_53 compute_60 compute_61 compute_62 "
                "compute_70 compute_72 compute_75 compute_80 compute_86 compute_87 compute_89 compute_90"),
    "12.0.140": ("Cuda compilation tools, release 12.0, V12.0.140",
                 "compute_50 compute_52 compute_53 compute_60 compute_61 compute_62 compute_70 compute_72 "
                 "compute_75 compute_80 compute_86 compute_87 compute_89 compute_90"),
    "12.6.85": ("Cuda compilation tools, release 12.6, V12.6.85",
                "compute_50 compute_52 compute_53 compute_60 compute_61 compute_62 compute_70 compute_72 "
                "compute_75 compute_80 compute_86 compute_87 compute_89 compute_90"),
    "12.8.93": ("Cuda compilation tools, release 12.8, V12.8.93",
                "compute_50 compute_52 compute_53 compute_60 compute_61 compute_62 compute_70 compute_72 "
                "compute_75 compute_80 compute_86 compute_87 compute_89 compute_90 compute_100 compute_101 "
                "compute_120"),
    "12.9.86": ("Cuda compilation tools, release 12.9, V12.9.86",
                "compute_50 compute_52 compute_53 compute_60 compute_61 compute_62 compute_70 compute_72 "
                "compute_75 compute_80 compute_86 compute_87 compute_89 compute_90 compute_100 compute_101 "
                "compute_103 compute_120 compute_121"),
    "13.0.88": ("Cuda compilation tools, release 13.0, V13.0.88",
                "compute_75 compute_80 compute_86 compute_87 compute_88 compute_89 compute_90 compute_100 "
                "compute_110 compute_103 compute_120 compute_121"),
    "13.4.92": ("Cuda compilation tools, release 13.4, V13.4.92",
                "compute_75 compute_80 compute_86 compute_87 compute_88 compute_89 compute_90 compute_100 "
                "compute_110 compute_103 compute_120 compute_121 compute_107"),
}
#: `nvcc --version` in full (13.0.88), and an option nvcc does not know (11.4.152).
NVCC_VERSION_13 = """nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
"""
NVCC_UNKNOWN_OPTION = "nvcc fatal   : Unknown option '--list-gpu-foo'\n"


def nvcc_toolkit(version):
    """The toolchain fields toolchain() fills from that nvcc's real output."""
    release, archs = NVCC_SAMPLES[version]
    return {"cuda_version": "%d.%d" % setup_flow.parse_nvcc_version(release),
            "cuda_archs": setup_flow.parse_nvcc_arch_list(archs.replace(" ", "\n") + "\n")}


V100 = {"index": 0, "name": "Tesla V100-SXM2-16GB", "total_bytes": 16384 * 2**20,
        "free_bytes": 16144 * 2**20, "driver": "580.178.04", "compute_cap": "7.0"}
V100_VK = {"name": "Tesla V100-SXM2-16GB", "type": "discrete", "api_version": "1.4.312",
           "api_version_raw": (1 << 22) | (4 << 12) | 312}
RTX3090 = {"index": 0, "name": "NVIDIA GeForce RTX 3090", "total_bytes": 24576 * 2**20,
           "free_bytes": 23000 * 2**20, "driver": "580.65", "compute_cap": "8.6"}
RTX5080 = {"index": 0, "name": "NVIDIA GeForce RTX 5080", "total_bytes": 16303 * 2**20,
           "free_bytes": 15000 * 2**20, "driver": "580.65", "compute_cap": "12.0"}
GTX970 = {"index": 0, "name": "NVIDIA GeForce GTX 970", "total_bytes": 4096 * 2**20,
          "free_bytes": 3900 * 2**20, "driver": "535.183", "compute_cap": "5.2"}


class CudaToolkit(unittest.TestCase):
    """What nvcc says it builds for, from its real output."""

    def test_list_gpu_arch(self):
        for version, (release, archs) in NVCC_SAMPLES.items():
            with self.subTest(nvcc=version):
                parsed = setup_flow.parse_nvcc_arch_list(archs.replace(" ", "\n") + "\n")
                self.assertEqual(parsed, sorted(int(a.split("_")[1]) for a in archs.split()))
                self.assertEqual(setup_flow.parse_nvcc_version(release),
                                 tuple(int(p) for p in version.split(".")[:2]))
        self.assertNotIn(70, nvcc_toolkit("13.0.88")["cuda_archs"])      # Volta is gone in 13
        self.assertIn(70, nvcc_toolkit("12.9.86")["cuda_archs"])
        self.assertNotIn(120, nvcc_toolkit("12.6.85")["cuda_archs"])     # Blackwell needs 12.8
        self.assertEqual(setup_flow.parse_nvcc_version(NVCC_VERSION_13), (13, 0))

    def test_an_nvcc_without_the_option_lists_nothing(self):
        self.assertIsNone(setup_flow.parse_nvcc_arch_list(NVCC_UNKNOWN_OPTION))
        self.assertIsNone(setup_flow.parse_nvcc_arch_list(""))
        self.assertIsNone(setup_flow.parse_nvcc_version(""))

    def test_the_release_table_matches_what_nvcc_lists(self):
        """Each measured nvcc's oldest and newest architecture is its row of
        CUDA_RELEASES, which answers for an nvcc that cannot list them."""
        for version in NVCC_SAMPLES:
            with self.subTest(nvcc=version):
                archs = nvcc_toolkit(version)["cuda_archs"]
                bounds = setup_flow._release_bounds(tuple(int(p) for p in version.split(".")[:2]))
                self.assertEqual(bounds, (min(archs), max(archs)))

    def test_the_version_answers_when_the_list_does_not(self):
        old = {"cuda_version": "10.2", "cuda_archs": None}
        self.assertTrue(setup_flow.toolkit_builds(old, 70))
        self.assertFalse(setup_flow.toolkit_builds(old, 80))
        self.assertFalse(setup_flow.toolkit_builds({"cuda_version": "13.1", "cuda_archs": None}, 70))
        self.assertIsNone(setup_flow.toolkit_builds({"cuda_version": None, "cuda_archs": None}, 70))
        self.assertIsNone(setup_flow.toolkit_builds({"cuda_version": "8.0", "cuda_archs": None}, 60))
        self.assertIsNone(setup_flow.toolkit_builds(nvcc_toolkit("13.0.88"), None))  # card unknown

    def test_what_to_install(self):
        advice = setup_flow.cuda_release_advice
        self.assertEqual(advice(70, (13, 0)), ("dropped in CUDA 13", "install a CUDA 12.x toolkit"))
        self.assertEqual(advice(61, (13, 2)), ("dropped in CUDA 13", "install a CUDA 12.x toolkit"))
        self.assertEqual(advice(37, (12, 6)), ("dropped in CUDA 12", "install a CUDA 11.x toolkit"))
        self.assertEqual(advice(120, (12, 6)), ("needs CUDA 12.8 or newer", "install CUDA 12.8 or newer"))
        self.assertEqual(advice(121, (12, 8)), ("needs CUDA 12.9 or newer", "install CUDA 12.9 or newer"))
        self.assertIn("compute_200", advice(200, (13, 0))[1])

    def test_cuda_toolkit_runs_nvcc(self):
        release, archs = NVCC_SAMPLES["13.0.88"]
        answers = {"--version": NVCC_VERSION_13, "--list-gpu-arch": archs.replace(" ", "\n") + "\n"}
        with mock.patch.object(setup_hw, "_run", side_effect=lambda cmd, timeout=10: answers[cmd[1]]):
            info = setup_flow.cuda_toolkit("/usr/local/cuda/bin/nvcc")
        self.assertEqual(info, {"cuda_version": "13.0", "cuda_archs": nvcc_toolkit("13.0.88")["cuda_archs"]})
        answers["--list-gpu-arch"] = NVCC_UNKNOWN_OPTION          # an nvcc without the option
        with mock.patch.object(setup_hw, "_run", side_effect=lambda cmd, timeout=10: answers[cmd[1]]):
            self.assertIsNone(setup_flow.cuda_toolkit("nvcc")["cuda_archs"])


class CudaBackendChoice(unittest.TestCase):
    """#1852: a CUDA toolkit is there, but it has to build for the card."""

    def choose(self, nvidia, toolkit, family="qwen36", vulkan=None, requested="auto", tc=None):
        tc = dict(tc or TC_ALL, **(nvcc_toolkit(toolkit) if toolkit else {}))
        with modeled("linux"):
            return setup_flow.choose_backend(hw_report(vulkan=vulkan, nvidia=nvidia),
                                             family_by_id(family), tc, requested)

    def test_v100_with_cuda_13_takes_vulkan_and_says_why(self):
        decision = self.choose([V100], "13.0.88", vulkan=V100_VK)
        self.assertEqual(decision["backend"], "vulkan")
        self.assertEqual(decision["reason"],
                         "the CUDA 13.0 toolkit cannot build for the Tesla V100-SXM2-16GB (compute 7.0, "
                         "dropped in CUDA 13): using Vulkan; install a CUDA 12.x toolkit for the CUDA path")
        self.assertEqual(decision["missing"][0][0], "cuda")
        self.assertIn("CUDA 12.x", decision["missing"][0][1])
        self.assertIn(setup_flow.CUDA_ARCHIVE_URL, decision["missing"][0][1])

    def test_v100_with_cuda_12_takes_cuda(self):
        for toolkit in ("12.0.140", "12.6.85", "12.9.86", "11.8.89"):
            with self.subTest(nvcc=toolkit):
                decision = self.choose([V100], toolkit, vulkan=V100_VK)
                self.assertEqual(decision["backend"], "cuda")
                self.assertEqual(decision["make_args"], [])
                self.assertIn("compute 7.0", decision["reason"])

    def test_rtx_30_with_cuda_13_takes_cuda(self):
        for toolkit in ("13.0.88", "13.4.92"):
            with self.subTest(nvcc=toolkit):
                decision = self.choose([RTX3090], toolkit)
                self.assertEqual(decision["backend"], "cuda")
                self.assertEqual(decision["gpu"], "NVIDIA GeForce RTX 3090")

    def test_a_card_newer_than_the_toolkit(self):
        decision = self.choose([RTX5080], "12.6.85")
        self.assertEqual(decision["backend"], "cpu")            # no Vulkan GPU here
        self.assertIn("needs CUDA 12.8 or newer", decision["reason"])
        self.assertIn("using the CPU", decision["reason"])
        self.assertEqual(decision["missing"][0][0], "cuda")
        self.assertEqual(self.choose([RTX5080], "12.8.93")["backend"], "cuda")

    def test_explicit_cuda_stops_before_building(self):
        with self.assertRaises(setup_flow.SetupError) as caught:
            self.choose([V100], "13.0.88", vulkan=V100_VK, requested="cuda")
        self.assertEqual(str(caught.exception),
                         "the CUDA 13.0 toolkit cannot build for the Tesla V100-SXM2-16GB (compute 7.0, "
                         "dropped in CUDA 13): install a CUDA 12.x toolkit for the CUDA path, or run "
                         "with --backend auto to use Vulkan")
        with self.assertRaises(setup_flow.SetupError) as caught:
            self.choose([V100], "13.0.88", requested="cuda")      # no Vulkan GPU: the CPU
        self.assertIn("--backend auto to use the CPU", str(caught.exception))
        # A toolkit that does build for the card is taken as asked.
        self.assertEqual(self.choose([V100], "12.6.85", requested="cuda")["backend"], "cuda")

    def test_every_card_has_to_be_buildable(self):
        """-arch=native builds for every card nvcc sees and stops at the first
        it cannot: a V100 next to an RTX 3090 stops CUDA 13 too."""
        second = dict(V100, index=1)
        decision = self.choose([RTX3090, second], "13.0.88")
        self.assertNotEqual(decision["backend"], "cuda")
        self.assertIn("Tesla V100", decision["reason"])

    def test_unknown_is_not_a_no(self):
        """No compute capability (an old driver, no CUDA probe) or a toolkit
        that said nothing: CUDA as before; a failed build falls back later."""
        unknown_card = dict(V100, compute_cap=None)
        self.assertEqual(self.choose([unknown_card], "13.0.88")["backend"], "cuda")
        self.assertEqual(self.choose([V100], None)["backend"], "cuda")

    def test_deepseek_v4_needs_compute_6_and_cuda_12(self):
        decision = self.choose([GTX970], "12.6.85", family="deepseek_v4")
        self.assertEqual(decision["backend"], "cpu")
        self.assertIn("DeepSeek V4 Flash's CUDA tier runs on compute 6.0 and newer", decision["reason"])
        self.assertEqual(decision["missing"], [])           # no toolkit would change that
        self.assertEqual(self.choose([GTX970], "12.6.85")["backend"], "cuda")   # backend_cuda.o: no floor
        decision = self.choose([V100], "11.8.89", family="deepseek_v4")
        self.assertNotEqual(decision["backend"], "cuda")
        self.assertIn("needs CUDA 12.0 or newer", decision["reason"])

    def test_deepseek_v4_below_cuda_12_8_builds_without_the_tensor_core_path(self):
        self.assertEqual(self.choose([V100], "12.6.85", family="deepseek_v4")["make_args"], ["NO_TC=1"])
        self.assertEqual(self.choose([RTX5080], "12.8.93", family="deepseek_v4")["make_args"], [])
        self.assertEqual(self.choose([V100], "12.6.85")["make_args"], [])      # qwen36 has no NO_TC

    def test_an_nvcc_without_native_gets_the_card_architecture(self):
        tc = {"cuda_version": "11.5", "cuda_archs": [35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87]}
        with mock.patch.dict(os.environ, {"CUDA_ARCH": ""}):
            decision = self.choose([V100], None, tc=dict(TC_ALL, **tc))
            self.assertEqual(decision["make_args"], ["CUDA_ARCH=sm_70"])
            self.assertEqual(self.choose([V100], "11.8.89")["make_args"], [])
        with mock.patch.dict(os.environ, {"CUDA_ARCH": "sm_70"}):         # set by hand: kept
            self.assertEqual(self.choose([V100], None, tc=dict(TC_ALL, **tc))["make_args"], [])

    def test_glm53_cuda_build_brings_vulkan(self):
        """glm53, deepseek_v41 and mimo hold their routed experts on the CUDA tier; their CUDA
        build has Vulkan too, which the engine runs on unless COLI_VULKAN=0 (its dense chain)."""
        for family in ("glm53", "deepseek_v41", "mimo"):
            with self.subTest(family=family):
                decision = self.choose([RTX3090], "13.0.88", family=family, vulkan=DGPU)
                self.assertEqual(decision["backend"], "cuda")
                with modeled("linux"), mock.patch.dict(os.environ, {"ARCH": "", "CUDA_ARCH": ""}):
                    cmd, _cwd, _env = setup_flow.make_command(family_by_id(family), "cuda", TC_ALL)
                self.assertIn("CUDA=1", cmd)
                self.assertIn("VK=1", cmd)

    def test_make_command_takes_the_decision_arguments(self):
        with modeled("linux"), mock.patch.dict(os.environ, {"ARCH": "", "CUDA_ARCH": ""}):
            cmd, _cwd, _env = setup_flow.make_command(family_by_id("deepseek_v4"), "cuda", TC_ALL,
                                                      ["NO_TC=1"])
            self.assertEqual(cmd[cmd.index("deepseek-v4"):],
                             ["deepseek-v4", "ARCH=native", "CUDA=1", "VK=1", "CUDA_ARCH=native", "NO_TC=1"])
            cmd, _cwd, _env = setup_flow.make_command(family_by_id("qwen36"), "cuda", TC_ALL,
                                                      ["CUDA_ARCH=sm_70"])
            self.assertEqual(cmd[cmd.index("qwen36"):], ["qwen36", "ARCH=native", "CUDA=1", "VK=1", "CUDA_ARCH=sm_70"])
            # without the Vulkan toolchain, or on Windows (its CUDA package has both), CUDA alone
            cmd, _cwd, _env = setup_flow.make_command(family_by_id("qwen36"), "cuda", dict(TC_ALL, can_build_vulkan=False),
                                                      ["CUDA_ARCH=sm_70"])
            self.assertEqual(cmd[cmd.index("qwen36"):], ["qwen36", "ARCH=native", "CUDA=1", "CUDA_ARCH=sm_70"])
            # GLM-5.2 (a GGUF), Kimi K3 and Inkling run on Vulkan where CUDA lacks their work
            for fid in ("glm", "kimi", "inkling"):
                cmd, _cwd, _env = setup_flow.make_command(family_by_id(fid), "cuda", TC_ALL, ["CUDA_ARCH=sm_70"])
                self.assertIn("VK=1", cmd, fid)
                self.assertIn("CUDA=1", cmd, fid)

    def test_a_pending_cuda_rebuild_checks_the_cards_it_was_for(self):
        cfg = {"backend": "cpu", "gpu_pending": ["cuda"], "family": "qwen36",
               "nvidia": [{"name": V100["name"], "compute_cap": "7.0"}]}
        self.assertFalse(setup_flow.gpu_now_buildable(cfg, dict(TC_ALL, **nvcc_toolkit("13.0.88"))))
        self.assertTrue(setup_flow.gpu_now_buildable(cfg, dict(TC_ALL, **nvcc_toolkit("12.6.85"))))
        self.assertFalse(setup_flow.gpu_now_buildable(
            cfg, dict(TC_ALL, nvcc=None, can_build_cuda=False, **nvcc_toolkit("12.6.85"))))


class CudaEngineTable(unittest.TestCase):
    """CUDA_ENGINES says what each engine's CUDA build is and needs; this reads
    the same facts from the Makefiles and the sources."""

    OBJECTS = {"CUDA_OBJ": "backend_cuda.o", "INK_CUDA_OBJ": "backend_cuda_ink.o"}

    def rule(self, artifact):
        makefile = (C_DIR / "Makefile").read_text(encoding="utf-8")
        match = re.search(rf"^{re.escape(artifact)}\$\(EXE\):.*?(?=\n\S|\Z)", makefile, re.M | re.S)
        return match.group(0) if match else ""

    def test_the_cuda_object_each_engine_links(self):
        for family in family_registry.FAMILIES:
            with self.subTest(family=family.id):
                if family.engine_artifact == "deepseek_v4":
                    v4 = (C_DIR / "Makefile.deepseek-v4").read_text(encoding="utf-8")
                    linked = "backend_cuda_dsv4.o" if "V4_WIN_EXTRA_OBJS += backend_cuda_dsv4.o" in v4 else None
                else:
                    rule = self.rule(family.engine_artifact)
                    self.assertTrue(rule, f"no Makefile rule for {family.engine_artifact}")
                    first_line = rule.splitlines()[0]
                    linked = next((obj for var, obj in self.OBJECTS.items()
                                   if f"$({var})" in first_line), None)
                self.assertEqual((setup_flow.CUDA_ENGINES.get(family.id) or {}).get("object"), linked)

    def test_deepseek_v4_needs_what_the_table_says(self):
        v4 = (C_DIR / "Makefile.deepseek-v4").read_text(encoding="utf-8")
        flags = v4[v4.index("V4_NVCCFLAGS ="):].split("\n\n")[0]
        self.assertIn("-std=c++20", flags)                       # nvcc takes it from 12.0
        self.assertEqual(setup_flow.CUDA_ENGINES["deepseek_v4"]["min_toolkit"], (12, 0))
        self.assertIn("NO_TC ?= 0", v4)
        source = (C_DIR / "backend_cuda_dsv4.cu").read_text(encoding="utf-8")
        gate = source[source.index("int dsv4_cuda_backend_arch_ok"):]
        self.assertIn("return prop.major>=6;", gate[:400])
        self.assertEqual(setup_flow.CUDA_ENGINES["deepseek_v4"]["min_compute"], 60)

    def test_backend_cuda_has_no_compute_floor(self):
        """The tensor-core kernels are guarded and fall back; nothing refuses an
        older card, so the table puts no floor on the engines that link it."""
        makefile = (C_DIR / "Makefile").read_text(encoding="utf-8")
        self.assertIn("NVCC_STD ?= c++17", makefile)
        source = (C_DIR / "backend_cuda.cu").read_text(encoding="utf-8")
        self.assertIn("#if __CUDA_ARCH__ >= 700", source)
        self.assertIn("ctx->compute_major<7)return 0", source)
        for family_id, entry in setup_flow.CUDA_ENGINES.items():
            if entry["object"] in ("backend_cuda.o", "backend_cuda_ink.o"):
                self.assertNotIn("min_compute", entry, family_id)
                self.assertNotIn("min_toolkit", entry, family_id)


class PackageHints(unittest.TestCase):
    def test_one_command_per_system(self):
        hint = setup_flow.package_hint
        self.assertEqual(hint(["build", "vulkan"], {"ID": "ubuntu", "ID_LIKE": "debian"}, "linux"),
                         "sudo apt install build-essential libvulkan-dev glslc mesa-vulkan-drivers")
        self.assertEqual(hint("vulkan", {"id": "fedora"}, "linux"),
                         "sudo dnf install vulkan-loader-devel glslc mesa-vulkan-drivers")
        self.assertIn("sudo pacman -S --needed vulkan-icd-loader", hint("vulkan", {"id": "arch"}, "linux"))
        self.assertIn("Vulkan driver", hint("vulkan", {"id": "arch"}, "linux"))
        self.assertIn("developer.nvidia.com", hint("cuda", {"id": "fedora"}, "linux"))
        self.assertIn("xcode-select", hint("build", {}, "darwin"))
        self.assertIn("from your distribution", hint("vulkan", {"id": "gentoo"}, "linux"))


class RunConfiguration(HomeTestCase):
    def test_vulkan_environment(self):
        engine_dir = os.path.join(self.tmp.name, "engines")
        os.makedirs(os.path.join(engine_dir, "shaders"))
        hw = hw_report(vulkan=IGPU, icd="/home/u/dzn_icd.json")
        env = setup_flow.run_environment("vulkan", os.path.join(engine_dir, "qwen36"),
                                         family_by_id("qwen36"), hw)
        self.assertEqual(env["COLI_VULKAN"], "1")
        self.assertEqual(env["COLI_VK_SHADERS"], os.path.join(engine_dir, "shaders"))
        self.assertEqual(env["VK_ICD_FILENAMES"], "/home/u/dzn_icd.json")
        self.assertNotIn("COLI_NO_OMP_TUNE", env)
        glm = setup_flow.run_environment("vulkan", os.path.join(engine_dir, "colibri"),
                                         family_by_id("glm"), hw)
        self.assertEqual(glm["COLI_NO_OMP_TUNE"], "1")
        self.assertEqual(glm["OMP_NUM_THREADS"], "6")     # physical cores, not 12 threads
        self.assertEqual(setup_flow.run_environment("cpu", "x", family_by_id("qwen36"), hw), {})

    def test_cuda_arguments(self):
        args = setup_flow.launcher_args("cuda", "/m", "127.0.0.1", 8000)
        self.assertEqual(args[:2], ["web", "--model"])
        self.assertIn("--auto-tier", args)
        self.assertEqual(args[args.index("--gpu") + 1], "auto")
        self.assertNotIn("--gpu", setup_flow.launcher_args("vulkan", "/m", "127.0.0.1", 8000))

    def test_every_backend_applies_the_plan(self):
        # #2003: a Vulkan or CPU setup recorded no --auto-tier, and the families whose
        # implicit cap is 8 ran 8 expert slots a layer whatever the printed plan said
        for backend in ("cuda", "vulkan", "cpu"):
            for family in ("qwen36", "qwen38", "kimi", "inkling", "deepseek_v4", "deepseek_v41", "mimo"):
                with self.subTest(backend=backend, family=family):
                    self.assertIn("--auto-tier", setup_flow.launcher_args(
                        backend, "/m", "127.0.0.1", 8000, family=family_by_id(family)))
        # the GLM engines and OLMoE size their cache themselves off CUDA, as before
        for family in ("glm", "glm53", "olmoe"):
            with self.subTest(family=family):
                self.assertNotIn("--auto-tier", setup_flow.launcher_args(
                    "vulkan", "/m", "127.0.0.1", 8000, family=family_by_id(family)))
                self.assertIn("--auto-tier", setup_flow.launcher_args(
                    "cuda", "/m", "127.0.0.1", 8000, family=family_by_id(family)))

    def test_ctx_is_recorded_and_checked(self):
        # #2011: `coli setup --ctx N` is kept for `coli start`, and a context the family
        # cannot hold stops the setup before anything is built
        args = setup_flow.launcher_args("vulkan", "/m", "127.0.0.1", 8000, 32768)
        self.assertEqual(args[args.index("--ctx") + 1], "32768")
        self.assertNotIn("--ctx", setup_flow.launcher_args("vulkan", "/m", "127.0.0.1", 8000))
        qwen36 = family_by_id("qwen36")
        setup_flow.check_ctx(0, qwen36)
        setup_flow.check_ctx(qwen36.limits.max_context, qwen36)
        for bad in (-1, qwen36.limits.max_context + 1):
            with self.subTest(ctx=bad), self.assertRaises(setup_flow.SetupError):
                setup_flow.check_ctx(bad, qwen36)

    def test_written_and_read_back(self):
        entry = setup_catalog.by_id("qwen36-35b")
        engine_info = {"launcher_dir": str(C_DIR), "engine": str(C_DIR / "qwen36"),
                       "backend": "vulkan", "source": "built"}
        cfg = setup_flow.make_config(model_dir="/models/qwen36-35b", family=family_by_id("qwen36"),
                                     entry=entry, engine_info=engine_info,
                                     hw=hw_report(vulkan=IGPU), host="0.0.0.0", port=8123)
        setup_flow.save_config(cfg)
        loaded = setup_flow.load_config()
        self.assertEqual(loaded["model"]["repo"], entry.repo)
        self.assertEqual(loaded["launcher"], str(C_DIR / "coli"))
        self.assertEqual(loaded["env"]["COLI_VULKAN"], "1")
        self.assertEqual(loaded["urls"], {"browser": "http://127.0.0.1:8123/",
                                          "openai_base_url": "http://127.0.0.1:8123/v1",
                                          "anthropic_base_url": "http://127.0.0.1:8123"})
        with modeled("linux"):
            self.assertTrue(setup_flow.equivalent_command(loaded).startswith("COLI_VULKAN=1 "))
            self.assertIn("--port 8123", setup_flow.equivalent_command(loaded))
        with modeled("win32"):
            self.assertTrue(setup_flow.equivalent_command(loaded).startswith('set "COLI_VULKAN=1" && '))
        self.assertEqual(setup_flow.configured_port(), 8123)
        # an unknown layout is not trusted
        Path(setup_flow.config_path()).write_text(json.dumps({"version": 99}))
        self.assertIsNone(setup_flow.load_config())
        self.assertEqual(setup_flow.configured_port(), 8000)

    def test_last_turn_speed(self):
        self.assertEqual(setup_flow.last_turn_speed({"turns": [{"wall_s": 4.0, "completion_tokens": 10}]}),
                         2.5)
        self.assertIsNone(setup_flow.last_turn_speed({"turns": []}))


class EngineResolution(HomeTestCase):
    def test_binary_backend(self):
        for content, expected in ((b"\0libvulkan.so.1\0", "vulkan"), (b"VULKAN-1.DLL", "vulkan"),
                                  (b"libcudart.so.12", "cuda"), (b"plain", "cpu"),
                                  (b"coli_cuda.dll vulkan-1.dll", "cuda")):
            path = Path(self.tmp.name, "engine")
            path.write_bytes(b"\x7fELF" + content)
            self.assertEqual(setup_flow.binary_backend(str(path)), expected)
        self.assertEqual(setup_flow.binary_backends(str(path)), {"cuda", "vulkan"})

    def test_an_engine_with_both_backends_is_used_for_vulkan(self):
        # a Windows release engine built with CUDA_DLL=1 and VK=1, no compiler here
        no_build = dict(TC_ALL, source_checkout=False, can_build=False, can_build_vulkan=False,
                        can_build_cuda=False)
        info, build, said = self.resolve_present("win32", lambda path, *a, **k: str(path), tc=no_build,
                                                 content=b"MZ coli_cuda.dll vulkan-1.dll")
        build.assert_not_called()
        self.assertEqual(info["backend"], "vulkan")
        self.assertEqual(info["source"], "present")

    def resolve_present(self, os_name, make, tc=TC_ALL, content=b"libvulkan.so.1"):
        """resolve_engine with an engine already here and `make` standing in for
        build_engine; returns (info, the build mock, what it said)."""
        said = []
        with modeled(os_name), tempfile.TemporaryDirectory() as engines:
            path = Path(engines, engine_name(os_name))
            path.write_bytes(content)
            os.utime(path, ns=(1_000_000_000, 1_000_000_000))
            with mock.patch.object(setup_flow, "HERE", engines), \
                 mock.patch.object(setup_flow, "build_engine",
                                   side_effect=lambda *a, **k: make(path, *a, **k)) as build:
                info = setup_flow.resolve_engine(family_by_id("qwen36"), None,
                                                 {"backend": "vulkan", "missing": []}, tc,
                                                 out=said.append)
        return info, build, "\n".join(said)

    def test_present_engine_is_brought_up_to_date_with_make(self):
        # #1852: an engine with the right backend used to be taken as it was, so after
        # a `git pull` the old engine kept running. make now redoes what changed.
        for os_name in ("linux", "win32"):
            with self.subTest(os=os_name):
                info, build, said = self.resolve_present(os_name, lambda path, *a, **k: str(path))
                build.assert_called_once()
                self.assertEqual(build.call_args.args[1], "vulkan")
                self.assertTrue(build.call_args.kwargs["update"])
                self.assertEqual(info["source"], "present")          # make had nothing to do
                self.assertIn("up to date", said)
                self.assertEqual(os.path.basename(info["engine"]), engine_name(os_name))

    def test_changed_sources_rebuild_the_present_engine(self):
        def make(path, *a, **k):
            path.write_bytes(b"libvulkan.so.1 newer")
            return str(path)
        info, build, said = self.resolve_present("linux", make)
        self.assertEqual(info["source"], "rebuilt")
        self.assertIn("rebuilt: the sources changed since the last build", said)

    def test_a_failed_update_keeps_the_engine_that_was_here(self):
        def make(path, family, backend, tc, out=print, make_args=(), update=False):
            raise setup_flow.BuildError("the qwen36 VULKAN build failed", backend, "/logs/build.log")
        info, build, said = self.resolve_present("linux", make)
        self.assertEqual(info["source"], "present")
        self.assertEqual(info["backend"], "vulkan")
        self.assertIn("could not update it (log: /logs/build.log); keeping the qwen36 engine", said)

    def test_a_failed_update_that_removed_the_engine_is_a_failed_build(self):
        def make(path, family, backend, tc, out=print, make_args=(), update=False):
            path.unlink()
            raise setup_flow.BuildError("the qwen36 VULKAN build failed", backend, "/logs/build.log")
        with self.assertRaises(setup_flow.BuildError):
            self.resolve_present("linux", make)

    def test_no_toolchain_for_the_engine_kind_keeps_it_as_it_is(self):
        no_vulkan = dict(TC_ALL, vulkan_headers=False, glslc=None, can_build_vulkan=False)
        info, build, said = self.resolve_present("linux", lambda path, *a, **k: str(path), tc=no_vulkan)
        build.assert_not_called()
        self.assertEqual(info["source"], "present")

    def test_a_gpu_engine_used_for_the_cpu_is_updated_as_the_gpu_build_it_is(self):
        said = []
        with modeled("linux"), tempfile.TemporaryDirectory() as engines:
            Path(engines, "qwen36").write_bytes(b"libvulkan.so.1")
            with mock.patch.object(setup_flow, "HERE", engines), \
                 mock.patch.object(setup_flow, "build_engine") as build:
                info = setup_flow.resolve_engine(family_by_id("qwen36"), None,
                                                 {"backend": "cpu", "missing": []}, TC_ALL,
                                                 out=said.append)
        self.assertEqual(build.call_args.args[1], "vulkan")      # not rebuilt as a CPU engine
        self.assertEqual(info["backend"], "cpu")

    def test_an_engine_without_the_exe_suffix_is_not_the_windows_engine(self):
        Path(self.tmp.name, "qwen36").write_bytes(b"vulkan-1.dll")
        with modeled("win32"), mock.patch.object(setup_flow, "HERE", self.tmp.name), \
             mock.patch.object(setup_flow, "build_engine", return_value="x") as build:
            setup_flow.resolve_engine(family_by_id("qwen36"), None,
                                      {"backend": "vulkan", "missing": []}, TC_ALL, out=lambda *_: None)
        build.assert_called_once()

    def test_a_cpu_engine_is_rebuilt_for_vulkan(self):
        for os_name in ("linux", "win32"):
            with self.subTest(os=os_name), modeled(os_name), \
                    tempfile.TemporaryDirectory() as engines:
                Path(engines, engine_name(os_name)).write_bytes(b"plain")
                decision = {"backend": "vulkan", "missing": []}
                with mock.patch.object(setup_flow, "HERE", engines), \
                     mock.patch.object(setup_flow, "build_engine", return_value="/built/qwen36") as build:
                    info = setup_flow.resolve_engine(family_by_id("qwen36"), None, decision, TC_ALL,
                                                     out=lambda *_: None)
                build.assert_called_once()
                self.assertEqual(info["source"], "built")

    def test_make_command_through_msys2_on_windows(self):
        tc = dict(TC_ALL, msys2=r"C:\msys64")
        with modeled("win32"), mock.patch.dict(os.environ, {"ARCH": ""}):
            cmd, _cwd, env = setup_flow.make_command(family_by_id("qwen36"), "vulkan", tc)
        self.assertTrue(cmd[0].endswith("bash.exe"))
        self.assertEqual(cmd[1], "-lc")
        self.assertIn("cygpath", cmd[2])
        self.assertEqual(cmd[-3:], ["qwen36", "ARCH=native", "VK=1"])
        self.assertEqual(env["MSYSTEM"], "UCRT64")

    def test_make_command(self):
        with modeled("linux"), mock.patch.dict(os.environ, {"ARCH": ""}):
            cmd, _cwd, _env = setup_flow.make_command(family_by_id("deepseek_v4"), "vulkan", TC_ALL)
            self.assertEqual(cmd[-3:], ["deepseek-v4", "ARCH=native", "VK=1"])
            cmd, _cwd, _env = setup_flow.make_command(family_by_id("qwen36"), "cuda", TC_ALL)
            self.assertIn("CUDA=1", cmd)
            self.assertIn("CUDA_ARCH=native", cmd)


class Releases(HomeTestCase):
    def test_asset_names(self):
        suffix = setup_flow.release_asset_suffix
        self.assertEqual(suffix("linux", "x86_64"), "linux-x86_64.tar.gz")
        self.assertEqual(suffix("win32", "AMD64"), "windows-x86_64.zip")
        self.assertEqual(suffix("darwin", "arm64"), "macos-arm64.tar.gz")
        self.assertIsNone(suffix("linux", "aarch64"))

    def test_find_release_falls_back_to_latest(self):
        seen = []

        def getter(url):
            seen.append(url)
            if url.endswith("/tags/v9.9.9"):
                raise setup_flow.urllib.error.URLError("404")
            return {"tag_name": "v1.12.1", "assets": [{"name": "a.zip", "browser_download_url": "u"}]}

        release = setup_flow.find_release("9.9.9", api="https://api.example/releases", getter=getter)
        self.assertEqual(release, {"tag": "v1.12.1", "assets": {"a.zip": "u"}})
        self.assertEqual(seen, ["https://api.example/releases/tags/v9.9.9",
                                "https://api.example/releases/latest"])

    def test_sha256sums(self):
        sums = setup_flow.parse_sha256sums("a" * 64 + "  colibri-v1-linux.tar.gz\n" +
                                           "B" * 64 + " *other.zip\nnot a line\n")
        self.assertEqual(sums, {"colibri-v1-linux.tar.gz": "a" * 64, "other.zip": "b" * 64})

    def test_archives_cannot_escape(self):
        evil_zip = os.path.join(self.tmp.name, "evil.zip")
        with zipfile.ZipFile(evil_zip, "w") as zf:
            zf.writestr("../escape.txt", "x")
        with self.assertRaises(setup_flow.SetupError):
            setup_flow.extract_archive(evil_zip, os.path.join(self.tmp.name, "out"))
        evil_tar = os.path.join(self.tmp.name, "evil.tar.gz")
        with tarfile.open(evil_tar, "w:gz") as tf:
            info = tarfile.TarInfo("/etc/escape")
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
        with self.assertRaises(setup_flow.SetupError):
            setup_flow.extract_archive(evil_tar, os.path.join(self.tmp.name, "out2"))
        self.assertFalse(Path(self.tmp.name, "escape.txt").exists())

    def serve_release(self, files, folder="release"):
        """A local release: the archive and SHA256SUMS.txt over HTTP."""
        folder = os.path.join(self.tmp.name, folder)
        os.makedirs(folder)
        for name, data in files.items():
            Path(folder, name).write_bytes(data)
        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=folder))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                         daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        return {name: f"{base}/{name}" for name in files}

    #: What the release workflow packs, per OS (.github/workflows/release.yml).
    LAYOUTS = {"linux": ("linux-x86_64.tar.gz", ("coli", "qwen36")),
               "win32": ("windows-x86_64.zip", ("coli", "coli.cmd", "qwen36.exe"))}

    def make_archive(self, os_name, vulkan=False):
        """A release archive; `vulkan`, one since 2.0: Vulkan engines and their shaders."""
        suffix, binaries = self.LAYOUTS[os_name]
        engine = (b"engine vulkan-1.dll" if os_name == "win32" else b"engine libvulkan.so.1") \
            if vulkan else b"engine"
        members = [(name, engine if name.startswith("qwen36") else b"# launcher") for name in binaries]
        members += [("web/dist/index.html", b"<html></html>"), ("tools/k3_tokenizer.py", b"")]
        if vulkan:
            members.append(("shaders/qmatmul.spv", b"\x03\x02\x23\x07"))
        buffer = io.BytesIO()
        if suffix.endswith(".zip"):
            with zipfile.ZipFile(buffer, "w") as zf:
                for name, data in members:
                    zf.writestr(name, data)
        else:
            with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
                for name, data in members:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
        return buffer.getvalue()

    def test_prebuilt_engine_when_there_is_no_compiler(self):
        for os_name in ("linux", "win32"):
            with self.subTest(os=os_name):
                self.check_prebuilt(os_name)

    def test_prebuilt_vulkan_engine_when_there_is_no_compiler(self):
        for os_name in ("linux", "win32"):
            with self.subTest(os=os_name):
                self.check_prebuilt(os_name, vulkan=True)

    def check_prebuilt(self, os_name, vulkan=False):
        tag = "v1.12.1"
        name = f"colibri-{tag}-{self.LAYOUTS[os_name][0]}"
        archive = self.make_archive(os_name, vulkan)
        sums = f"{hashlib.sha256(archive).hexdigest()}  {name}\n".encode()
        urls = self.serve_release({name: archive, "SHA256SUMS.txt": sums}, folder=os_name)
        getter = lambda url: {"tag_name": tag, "assets": [{"name": n, "browser_download_url": u}
                                                          for n, u in urls.items()]}
        tc = dict(TC_ALL, source_checkout=False, can_build=False, can_build_vulkan=False)
        empty = os.path.join(self.tmp.name, f"nothing-here-{os_name}-{vulkan}")
        os.makedirs(empty)
        home = os.path.join(self.tmp.name, f"home-{os_name}-{vulkan}")
        with modeled(os_name), mock.patch.object(setup_flow, "HERE", empty), \
             mock.patch.dict(os.environ, {"COLI_SETUP_HOME": home}), \
             mock.patch.object(setup_flow, "_api_get", side_effect=getter):
            out = []
            info = setup_flow.resolve_engine(family_by_id("qwen36"), setup_catalog.by_id("qwen36-35b"),
                                             {"backend": "vulkan", "missing": []}, tc, out=out.append)
            # an archive from before 2.0 has CPU engines; one since, Vulkan engines
            self.assertEqual(info["backend"], "vulkan" if vulkan else "cpu")
            self.assertEqual("no vulkan build" in "\n".join(out), not vulkan)
            self.assertTrue(info["source"].startswith("release v1.12.1"))
            self.assertTrue(os.path.isfile(info["engine"]))
            self.assertEqual(os.path.basename(info["engine"]), engine_name(os_name))
            self.assertTrue(os.path.isfile(setup_flow.launcher_in(info["launcher_dir"])))
            self.assertTrue(os.path.isfile(os.path.join(info["launcher_dir"], "web", "dist", "index.html")))
            # A model newer than the release is refused with the way forward.
            with self.assertRaises(setup_flow.SetupError) as caught:
                setup_flow.resolve_engine(family_by_id("qwen36"), setup_catalog.by_id("qwen3-coder-30b"),
                                          {"backend": "cpu", "missing": []}, tc, out=out.append)
            self.assertIn("predates", str(caught.exception))

    def cuda_archive(self, tag):
        """The release's windows-x86_64-cuda.zip: the engines again, built to load
        coli_cuda.dll, the DLL and its runtime."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("qwen36.exe", b"engine coli_cuda.dll vulkan-1.dll")
            zf.writestr("coli_cuda.dll", b"MZ")
            zf.writestr("cudart64_12.dll", b"MZ")
            zf.writestr("README-CUDA.txt", f"colibri {tag}: the CUDA backend for Windows")
        return buffer.getvalue()

    def windows_release(self, tag, folder, cuda=True):
        """The Windows release over HTTP, with its CUDA backend or (an older one) without."""
        name = f"colibri-{tag}-windows-x86_64.zip"
        files = {name: self.make_archive("win32", vulkan=True)}
        if cuda:
            files[f"colibri-{tag}-windows-x86_64-cuda.zip"] = self.cuda_archive(tag)
        files["SHA256SUMS.txt"] = "".join(f"{hashlib.sha256(data).hexdigest()}  {n}\n"
                                          for n, data in files.items()).encode()
        urls = self.serve_release(files, folder=folder)
        return lambda url: {"tag_name": tag, "assets": [{"name": n, "browser_download_url": u}
                                                        for n, u in urls.items()]}

    def test_the_windows_release_brings_its_cuda_backend(self):
        # #2091: a CUDA choice on Windows with no engine here takes the release's
        # engines and unpacks its CUDA backend over them
        tc = dict(TC_ALL, source_checkout=False, can_build=False, can_build_vulkan=False)
        for cuda in (True, False):
            with self.subTest(cuda=cuda):
                empty = os.path.join(self.tmp.name, f"nothing-here-{cuda}")
                os.makedirs(empty)
                getter = self.windows_release("v2.1.0", f"release-{cuda}", cuda=cuda)
                with modeled("win32"), mock.patch.object(setup_flow, "HERE", empty), \
                     mock.patch.dict(os.environ, {"COLI_SETUP_HOME": os.path.join(self.tmp.name, f"h-{cuda}")}), \
                     mock.patch.object(setup_flow, "_api_get", side_effect=getter):
                    out = []
                    info = setup_flow.resolve_engine(family_by_id("qwen36"), setup_catalog.by_id("qwen36-35b"),
                                                     {"backend": "cuda", "missing": []}, tc, out=out.append)
                    said = "\n".join(out)
                    if cuda:
                        self.assertEqual(info["backend"], "cuda")
                        self.assertTrue(setup_flow.prebuilt_runs(family_by_id("qwen36"), "cuda",
                                                                 info["launcher_dir"]))
                        self.assertIn("downloading the CUDA backend of v2.1.0", said)
                        self.assertFalse(os.path.exists(info["launcher_dir"] + ".cuda"))
                    else:   # a release from before the backend: its Vulkan engines
                        self.assertEqual(info["backend"], "vulkan")
                        self.assertIn("release v2.1.0 has no windows-x86_64-cuda.zip: using Vulkan", said)

    def test_advertised_release_checksums_cannot_be_skipped(self):
        name = "colibri-v1.12.1-fixture.tar.gz"
        archive = b"release fixture, never extracted or executed"
        digest = hashlib.sha256(archive).hexdigest()
        mode = {"value": "valid"}
        requests = []
        class ReleaseHTTP(SimpleHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                requests.append(self.path)
                status = 200
                extra_length = 0
                if self.path == "/" + name:
                    body = archive
                elif mode["value"] == "503":
                    status, body = 503, b""
                else:
                    text = {"valid": f"{digest}  {name}\n",
                            "missing": f"{digest}  other-archive.tar.gz\n",
                            "malformed": f"not-a-sha256  {name}\n",
                            "truncated-entry": digest[:24],
                            "truncated-http": "short"}[mode["value"]]
                    body = text.encode()
                    extra_length = 30 if mode["value"] == "truncated-http" else 0
                self.send_response(status)
                self.send_header("Content-Length", str(len(body) + extra_length))
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
        server = ThreadingHTTPServer(("127.0.0.1", 0), ReleaseHTTP)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        release = {"tag": "v1.12.1", "assets": {
            name: f"{base}/{name}", "SHA256SUMS.txt": f"{base}/SHA256SUMS.txt"}}
        try:
            for value in ("503", "truncated-http", "truncated-entry", "missing", "malformed"):
                with self.subTest(case=value):
                    mode["value"] = value
                    before = len(requests)
                    folder = os.path.join(self.tmp.name, value)
                    with self.assertRaises(setup_flow.SetupError) as caught:
                        setup_flow._download_asset(release, name, folder)
                    self.assertIn("SHA256SUMS.txt", str(caught.exception))
                    self.assertIn(name, str(caught.exception))
                    self.assertEqual(requests[before:], ["/SHA256SUMS.txt"])
                    self.assertFalse(Path(folder, name).exists())
            for value in ("valid", "legacy"):
                with self.subTest(case=value):
                    mode["value"] = value
                    selected = release if value == "valid" else {
                        "tag": release["tag"], "assets": {name: release["assets"][name]}}
                    before = len(requests)
                    path = setup_flow._download_asset(selected, name, os.path.join(self.tmp.name, value))
                    self.assertEqual(Path(path).read_bytes(), archive)
                    self.assertEqual(requests[before:], (["/SHA256SUMS.txt"] if value == "valid" else [])
                                     + ["/" + name])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_bad_checksum_is_refused(self):
        tag = "v1.12.1"
        name = f"colibri-{tag}-linux-x86_64.tar.gz"
        urls = self.serve_release({name: self.make_archive("linux"),
                                   "SHA256SUMS.txt": f"{'0' * 64}  {name}\n".encode()})
        getter = lambda url: {"tag_name": tag, "assets": [{"name": n, "browser_download_url": u}
                                                          for n, u in urls.items()]}
        with modeled("linux"):
            with self.assertRaises(setup_flow.SetupError) as caught:
                setup_flow.fetch_release_archive("1.12.1", out=lambda *_: None, getter=getter)
        self.assertIn("checksum", str(caught.exception))


class WholeSetup(HomeTestCase):
    """`coli setup --yes` from nothing to a written configuration, then a rerun."""

    def setUp(self):
        super().setUp()
        self.src = os.path.join(self.tmp.name, "repo")
        os.makedirs(self.src)
        Path(self.src, "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe_text"}))
        Path(self.src, "tokenizer.json").write_text("{}")
        self.shard = os.urandom(400_000)
        Path(self.src, "model-00000.safetensors").write_bytes(self.shard)
        self.hub = FakeHub({REPO: self.src})
        self.hub.__enter__()
        self.addCleanup(self.hub.__exit__)
        catalog = os.path.join(self.tmp.name, "catalog.json")
        Path(catalog).write_text(json.dumps([{
            "id": "tiny", "family": "qwen36", "name": "Tiny test model", "repo": REPO,
            "disk_gb": 0.001, "ram_min_gb": 0.1, "ram_good_gb": 0.2, "dense_gb": 0.01,
            "rank": 99, "size_class": "small", "summary": "fixture"}]))
        self.engines = os.path.join(self.tmp.name, "c")
        os.makedirs(os.path.join(self.engines, "shaders"))
        os.makedirs(os.path.join(self.engines, "web", "dist"))
        Path(self.engines, "web", "dist", "index.html").write_text("<html></html>")
        Path(self.engines, "qwen36").write_bytes(b"\x7fELF libvulkan.so.1")
        Path(self.engines, "coli").write_text("# launcher\n")
        self.models = os.path.join(self.tmp.name, "models")
        self.addCleanup(modeled("linux").close)       # patches from here on
        for patcher in (
                mock.patch.dict(os.environ, {"HF_ENDPOINT": self.hub.base,
                                             "COLI_SETUP_CATALOG": catalog}),
                mock.patch.object(setup_flow, "HERE", self.engines),
                mock.patch.object(setup_hw, "detect",
                                  return_value=hw_report(vulkan=IGPU, icd="/home/u/dzn.json")),
                mock.patch.object(setup_hw, "is_wsl", return_value=False),
                mock.patch.object(setup_flow, "toolchain", return_value=TC_ALL),
                mock.patch.object(setup_flow, "plan_summary",
                                  return_value=("dense part 0.0 GB in RAM", [])),
                mock.patch.object(setup_download.time, "sleep"),
                # No colibri runs (whatever this machine has on port 8000), so a
                # rerun checks its engine; make has nothing to do, the engine here
                # stays as it is. A test that builds patches build_engine again inside.
                mock.patch.object(setup_flow, "server_status",
                                  return_value={"state": "stopped", "pid": None}),
                mock.patch.object(setup_flow, "build_engine",
                                  side_effect=lambda family, *a, **k: os.path.join(self.engines, "qwen36"))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_setup(self, **overrides):
        out = io.StringIO()
        ui = setup_flow.UI(False, stream=out)
        code = setup_flow.run(lambda a: setup_flow.cmd_setup(a, ui=ui),
                              setup_args(dir=self.models, **overrides))
        return code, out.getvalue()

    def _assert_same_as_command_preserves_paths(self, model_name, engine_name=None):
        self.models = os.path.join(self.tmp.name, model_name)
        if engine_name:
            new_engine_dir = os.path.join(self.tmp.name, engine_name)
            os.rename(self.engines, new_engine_dir)
            self.engines = new_engine_dir
            patcher = mock.patch.object(setup_flow, "HERE", self.engines)
            patcher.start()
            self.addCleanup(patcher.stop)
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        cfg = setup_flow.load_config()
        command = next(line for line in text.splitlines() if "same as:" in line).split("same as:", 1)[1].strip()
        bindir = Path(self.tmp.name, "bin")
        bindir.mkdir()
        launcher = bindir / "coli"
        launcher.write_text(
            "#!" + sys.executable + "\nimport os, sys, json\n"
            + "keys = " + repr(list(cfg["env"])) + "\n"
            + "print(json.dumps({'args': sys.argv[1:], 'env': {key: os.environ.get(key) for key in keys}}))\n",
            encoding="utf-8",
        )
        launcher.chmod(0o700)
        # Native shell and argv consumer; no engine or model is invoked.
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            env={"PATH": str(bindir) + ":/usr/bin:/bin", "HOME": self.tmp.name},
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        consumed = json.loads(result.stdout)
        self.assertEqual(consumed["args"], cfg["args"])
        self.assertEqual(consumed["env"], cfg["env"])

    @unittest.skipIf(os.name == "nt", "requires a native POSIX shell")
    def test_same_as_command_preserves_unicode_model_directory(self):
        self._assert_same_as_command_preserves_paths("模型 é")

    @unittest.skipIf(os.name == "nt", "requires a native POSIX shell")
    def test_same_as_command_preserves_unicode_shader_directory(self):
        self._assert_same_as_command_preserves_paths("models", "引擎 é")

    @unittest.skipIf(os.name == "nt", "requires a native POSIX shell")
    def test_same_as_command_preserves_ascii_space_directory(self):
        self._assert_same_as_command_preserves_paths("local models")

    def test_install_then_rerun(self):
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        model_dir = os.path.join(self.models, "tiny")
        self.assertEqual(Path(model_dir, "model-00000.safetensors").read_bytes(), self.shard)
        self.assertTrue(setup_download.is_complete(model_dir))
        cfg = setup_flow.load_config()
        self.assertEqual(cfg["status"], "ready")
        self.assertEqual(cfg["backend"], "vulkan")
        self.assertEqual(cfg["env"]["COLI_VULKAN"], "1")
        self.assertEqual(cfg["env"]["VK_ICD_FILENAMES"], "/home/u/dzn.json")
        self.assertEqual(cfg["args"][:3], ["web", "--model", model_dir])
        self.assertEqual(cfg["launcher"], os.path.join(self.engines, "coli"))
        self.assertIn("Saved the run configuration", text)
        self.assertIn("OpenAI base URL:     http://127.0.0.1:8000/v1", text)
        self.assertIn("Anthropic base URL:  http://127.0.0.1:8000", text)
        # Rerun: straight to the start, nothing fetched, nothing built.
        before = len(self.hub.requests)
        with mock.patch.object(setup_flow, "build_engine") as build:
            code, text = self.run_setup()
        self.assertEqual(code, 0, text)
        self.assertIn("Already set up: Tiny test model", text)
        self.assertEqual(len(self.hub.requests), before)
        build.assert_called_once()                       # make, with nothing to redo
        self.assertTrue(build.call_args.kwargs["update"])

    def test_rerun_ctx_zero_restores_automatic_context(self):
        code, text = self.run_setup(pick="tiny", ctx=8192)
        self.assertEqual(code, 0, text)
        self.assertEqual(setup_flow.load_config()["args"][-2:], ["--ctx", "8192"])
        before = len(self.hub.requests)

        code, text = self.run_setup(ctx=0)
        self.assertEqual(code, 0, text)
        cfg = setup_flow.load_config()
        self.assertEqual(cfg["ctx"], 0)
        self.assertNotIn("--ctx", cfg["args"])
        self.assertEqual(len(self.hub.requests), before)

    def test_invalid_ctx_is_rejected_before_engine_or_download(self):
        with mock.patch.object(setup_flow, "resolve_with_fallback",
                               side_effect=AssertionError("engine work started")):
            code, text = self.run_setup(pick="tiny", ctx=-1)
        self.assertEqual(code, 1, text)
        self.assertIsNone(setup_flow.load_config())
        self.assertFalse(self.hub.requests)

    def test_ready_setup_honors_explicit_cpu_backend(self):
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        previous = setup_flow.load_config()
        self.assertEqual(previous["backend"], "vulkan")
        self.assertEqual(previous["env"]["COLI_VULKAN"], "1")
        build, _ = self.failing_build()
        with mock.patch.object(setup_flow, "build_engine", side_effect=build):
            code, text = self.run_setup(backend="cpu")
        self.assertEqual(code, 0, text)
        current = setup_flow.load_config()
        self.assertEqual(current["backend"], "cpu")
        self.assertNotIn("COLI_VULKAN", current["env"])
        self.assertEqual(current["model_dir"], previous["model_dir"])
        self.assertEqual(
            Path(current["model_dir"], "model-00000.safetensors").read_bytes(),
            self.shard,
        )

    def test_ready_setup_honors_no_gpu(self):
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        previous = setup_flow.load_config()
        self.assertEqual(previous["backend"], "vulkan")
        self.assertEqual(previous["env"]["COLI_VULKAN"], "1")
        build, _ = self.failing_build()
        with mock.patch.object(setup_flow, "build_engine", side_effect=build):
            code, text = self.run_setup(no_gpu=True)
        self.assertEqual(code, 0, text)
        current = setup_flow.load_config()
        self.assertEqual(current["backend"], "cpu")
        self.assertNotIn("COLI_VULKAN", current["env"])
        self.assertEqual(current["model_dir"], previous["model_dir"])
        self.assertEqual(
            Path(current["model_dir"], "model-00000.safetensors").read_bytes(),
            self.shard,
        )

    def test_ready_setup_honors_explicit_host(self):
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        previous = setup_flow.load_config()
        self.assertEqual(previous["host"], "127.0.0.1")
        code, text = self.run_setup(host="127.0.0.2")
        self.assertEqual(code, 0, text)
        current = setup_flow.load_config()
        self.assertEqual(current["host"], "127.0.0.2")
        self.assertEqual(current["args"][current["args"].index("--host") + 1], "127.0.0.2")
        self.assertEqual(current["urls"]["browser"], "http://127.0.0.2:8000/")
        self.assertEqual(current["model_dir"], previous["model_dir"])
        self.assertEqual(
            Path(current["model_dir"], "model-00000.safetensors").read_bytes(),
            self.shard,
        )

    def test_ready_setup_honors_explicit_port(self):
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        previous = setup_flow.load_config()
        self.assertEqual(previous["port"], 8000)
        code, text = self.run_setup(port=8123)
        self.assertEqual(code, 0, text)
        current = setup_flow.load_config()
        self.assertEqual(current["port"], 8123)
        self.assertEqual(current["args"][current["args"].index("--port") + 1], "8123")
        self.assertEqual(current["urls"]["browser"], "http://127.0.0.1:8123/")
        self.assertEqual(current["model_dir"], previous["model_dir"])
        self.assertEqual(
            Path(current["model_dir"], "model-00000.safetensors").read_bytes(),
            self.shard,
        )

    def test_a_rerun_after_a_pull_rebuilds_the_engine_it_starts(self):
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        engine = Path(self.engines, "qwen36")
        os.utime(engine, ns=(1_000_000_000, 1_000_000_000))

        def make(family, backend, tc, out=print, make_args=(), update=False):
            self.assertEqual((backend, update), ("vulkan", True))
            engine.write_bytes(b"\x7fELF libvulkan.so.1 with the pulled sources")
            return str(engine)

        with mock.patch.object(setup_flow, "build_engine", side_effect=make):
            code, text = self.run_setup()
        self.assertEqual(code, 0, text)
        self.assertIn("Already set up: Tiny test model", text)
        self.assertIn("rebuilt: the sources changed since the last build", text)
        self.assertIn(b"pulled sources", engine.read_bytes())

    def test_a_fresh_install_gets_the_shipped_expert_profile(self):
        # profiles/<id>.coli_usage is copied into a model that has no history yet,
        # never over one it has, and a model without a profile gets nothing.
        profiles = Path(self.tmp.name, "profiles")
        profiles.mkdir()
        Path(profiles, "tiny.coli_usage").write_text("-1 2 4\n-2 1 7\n0 1 3\n")
        with mock.patch.object(setup_flow, "PROFILES_DIR", str(profiles)):
            code, text = self.run_setup(pick="tiny")
            self.assertEqual(code, 0, text)
            history = Path(self.models, "tiny", ".coli_usage")
            self.assertEqual(history.read_text(), "-1 2 4\n-2 1 7\n0 1 3\n")
            self.assertIn("expert profile: a starting history for Tiny test model", text)
            history.write_text("-1 2 4\n-2 1 7\n1 0 9\n")           # the user's own use since
            code, text = self.run_setup()
            self.assertEqual(code, 0, text)
            self.assertEqual(history.read_text(), "-1 2 4\n-2 1 7\n1 0 9\n")
            self.assertNotIn("expert profile", text)
            history.unlink()                                          # a rerun seeds a model left without one
            code, text = self.run_setup()
            self.assertIn("expert profile", text)
            self.assertTrue(history.is_file())
        history.unlink()
        with mock.patch.object(setup_flow, "PROFILES_DIR", str(Path(self.tmp.name, "none"))):
            code, text = self.run_setup()
        self.assertFalse(history.exists())

    def test_interrupted_download_continues_on_rerun(self):
        real = setup_download.download_repo
        self.hub.cut_after["model-00000.safetensors"] = 150_000
        with mock.patch.object(setup_download, "download_repo",
                               side_effect=lambda *a, **k: real(*a, **dict(k, retries=0))):
            code, text = self.run_setup(pick="tiny", ctx=8192)
        self.assertEqual(code, 1)
        self.assertEqual(setup_flow.load_config()["status"], "pending")
        self.assertEqual(setup_flow.load_config()["ctx"], 8192)
        self.assertEqual(setup_flow.read_state()["phase"], "error")
        part = Path(self.models, "tiny", "model-00000.safetensors.part")
        self.assertEqual(part.stat().st_size, 150_000)
        # Same command again, without naming the model: it resumes the pending one.
        code, text = self.run_setup()
        self.assertEqual(code, 0, text)
        self.assertIn("Resuming the setup of Tiny test model", text)
        self.assertIn("resuming:", text)
        ranges = [r["headers"].get("Range") for r in self.hub.requests
                  if r["path"].endswith("model-00000.safetensors") and "/cdn/" in r["path"]]
        self.assertEqual(ranges[-1], "bytes=150000-")
        self.assertEqual(Path(self.models, "tiny", "model-00000.safetensors").read_bytes(), self.shard)
        self.assertEqual(setup_flow.load_config()["status"], "ready")
        self.assertEqual(setup_flow.load_config()["args"][-2:], ["--ctx", "8192"])

    def test_install_on_windows_names_the_exe_engine(self):
        os.remove(os.path.join(self.engines, "qwen36"))
        Path(self.engines, "qwen36.exe").write_bytes(b"MZ vulkan-1.dll")
        windows = hw_report(vulkan=IGPU)
        windows["os"] = {"platform": "win32", "machine": "amd64", "wsl": False,
                         "pretty_name": "Windows 11"}
        with modeled("win32", machine="amd64"), \
             mock.patch.object(setup_hw, "detect", return_value=windows):
            code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        cfg = setup_flow.load_config()
        self.assertEqual(os.path.basename(cfg["engine"]), "qwen36.exe")
        self.assertEqual(cfg["backend"], "vulkan")
        self.assertNotIn("VK_ICD_FILENAMES", cfg["env"])        # the loader finds Windows drivers itself
        self.assertIn('same as: set "COLI_VULKAN=1" && set "COLI_VK_SHADERS=', text)
        self.assertIn(" && coli web --model ", text)
        self.assertIn("System  Windows 11", text)

    def test_a_missing_engine_is_rebuilt_without_asking_again(self):
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        os.remove(os.path.join(self.engines, "qwen36"))
        shard_fetches = sum("/cdn/" in r["path"] for r in self.hub.requests)

        def rebuild(family, backend, tc, out=print, make_args=(), update=False):
            Path(self.engines, "qwen36").write_bytes(b"libvulkan.so.1")
            return os.path.join(self.engines, "qwen36")

        with mock.patch.object(setup_flow, "build_engine", side_effect=rebuild) as build:
            code, text = self.run_setup()
        self.assertEqual(code, 0, text)
        self.assertIn("Resuming the setup of Tiny test model", text)
        self.assertIn("Download: already complete", text)
        build.assert_called_once()
        self.assertEqual(sum("/cdn/" in r["path"] for r in self.hub.requests), shard_fetches)

    def test_a_rerun_switches_to_the_gpu_once_its_packages_are_there(self):
        no_vulkan = dict(TC_ALL, vulkan_headers=False, glslc=None, can_build_vulkan=False)
        Path(self.engines, "qwen36").write_bytes(b"\x7fELF plain")          # a CPU build
        with mock.patch.object(setup_flow, "toolchain", return_value=no_vulkan):
            code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        self.assertIn("to use the GPU through VULKAN, first run:", text)
        cfg = setup_flow.load_config()
        self.assertEqual((cfg["backend"], cfg["gpu_pending"]), ("cpu", ["vulkan"]))
        # Still no packages: a rerun only starts.
        with mock.patch.object(setup_flow, "toolchain", return_value=no_vulkan):
            code, text = self.run_setup()
        self.assertIn("Already set up", text)

        def rebuild(family, backend, tc, out=print, make_args=(), update=False):
            self.assertEqual(backend, "vulkan")
            Path(self.engines, "qwen36").write_bytes(b"libvulkan.so.1")
            return os.path.join(self.engines, "qwen36")

        with mock.patch.object(setup_flow, "build_engine", side_effect=rebuild):
            code, text = self.run_setup()                  # the packages are here now
        self.assertEqual(code, 0, text)
        self.assertIn("rebuilding the engine for the GPU", text)
        cfg = setup_flow.load_config()
        self.assertEqual((cfg["backend"], cfg["gpu_pending"]), ("vulkan", []))
        self.assertEqual(cfg["env"]["COLI_VULKAN"], "1")

    def failing_build(self, *broken):
        """build_engine as a mock: the backends in `broken` fail the way make
        does (BuildError with their log), the others write a matching engine.
        `calls` lists the builds; updating an engine already here (refresh_engine)
        is make with nothing to do, and not a build."""
        loaders = {"cuda": b"libcudart.so.12", "vulkan": b"libvulkan.so.1", "cpu": b"plain"}
        calls = []

        def build(family, backend, tc, out=print, make_args=(), update=False):
            if update:
                return os.path.join(self.engines, "qwen36")
            calls.append(backend)
            log = setup_flow.log_path(setup_flow.build_log_name(backend))
            if backend in broken:
                raise setup_flow.BuildError(
                    f"the qwen36 {setup_flow.BACKEND_NAME[backend]} build failed (full log: {log}):\n"
                    "    make: *** [Makefile:949: backend_cuda.o] Error 1", backend, log)
            Path(self.engines, "qwen36").write_bytes(b"\x7fELF " + loaders[backend])
            return os.path.join(self.engines, "qwen36")

        return build, calls

    def v100_machine(self):
        """The #1852 machine, with a toolkit that did not say what it builds:
        CUDA is chosen, so the build is what fails."""
        Path(self.engines, "qwen36").write_bytes(b"\x7fELF plain")
        return mock.patch.object(setup_hw, "detect", return_value=hw_report(vulkan=V100_VK, nvidia=[V100]))

    def test_a_failed_cuda_build_falls_back_to_vulkan(self):
        build, calls = self.failing_build("cuda")
        with self.v100_machine(), mock.patch.object(setup_flow, "build_engine", side_effect=build):
            code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        self.assertEqual(calls, ["cuda", "vulkan"])
        cuda_log = setup_flow.log_path("build-cuda")
        self.assertIn(f"the qwen36 CUDA build failed (full log: {cuda_log})", text)
        self.assertIn(f"Engine: qwen36 with VULKAN (the CUDA build failed (full log: {cuda_log}): "
                      "using Vulkan)", text)
        cfg = setup_flow.load_config()
        self.assertEqual((cfg["backend"], cfg["failed_builds"]), ("vulkan", ["cuda"]))
        self.assertEqual(cfg["env"]["COLI_VULKAN"], "1")

    def test_a_failed_vulkan_build_falls_back_to_the_cpu(self):
        build, calls = self.failing_build("cuda", "vulkan")
        with self.v100_machine(), mock.patch.object(setup_flow, "build_engine", side_effect=build):
            code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        self.assertEqual(calls, ["cuda", "vulkan"])        # a plain engine is already here for the CPU
        self.assertIn("the qwen36 Vulkan build failed (full log: ", text)
        self.assertIn("using the CPU", text)
        cfg = setup_flow.load_config()
        self.assertEqual((cfg["backend"], cfg["failed_builds"], cfg["gpu_pending"]),
                         ("cpu", ["cuda", "vulkan"], []))  # nothing to install: no rebuild on rerun

    def test_a_failed_cpu_build_stops(self):
        os.remove(os.path.join(self.engines, "qwen36"))
        build, calls = self.failing_build("cpu")
        with mock.patch.object(setup_flow, "build_engine", side_effect=build):
            code, text = self.run_setup(pick="tiny", backend="cpu")
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["cpu"])
        self.assertEqual(setup_flow.read_state()["phase"], "error")
        self.assertIn("build-cpu.log", setup_flow.read_state()["message"])

    def test_the_fallback_is_offered_when_someone_can_answer(self):
        build, calls = self.failing_build("cuda")
        out = io.StringIO()
        ui = setup_flow.UI(True, stream=out)
        with self.v100_machine(), mock.patch.object(setup_flow, "build_engine", side_effect=build), \
             mock.patch("builtins.input", side_effect=["n"]) as asked:
            code = setup_flow.run(lambda a: setup_flow.cmd_setup(a, ui=ui),
                                  setup_args(dir=self.models, pick="tiny", yes=False))
        self.assertEqual(code, 1, out.getvalue())
        self.assertEqual(calls, ["cuda"])
        self.assertIn("Build for Vulkan instead?", asked.call_args[0][0])
        self.assertIn("build-cuda.log", setup_flow.read_state()["message"])

    def test_a_resumed_setup_does_not_build_the_failed_backend_again(self):
        build, calls = self.failing_build("cuda")
        real = setup_download.download_repo
        self.hub.cut_after["model-00000.safetensors"] = 150_000
        with self.v100_machine(), mock.patch.object(setup_flow, "build_engine", side_effect=build), \
             mock.patch.object(setup_download, "download_repo",
                               side_effect=lambda *a, **k: real(*a, **dict(k, retries=0))):
            code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 1)                              # the download was cut
        self.assertEqual(setup_flow.load_config()["failed_builds"], ["cuda"])
        with self.v100_machine(), mock.patch.object(setup_flow, "build_engine", side_effect=build):
            Path(self.engines, "qwen36").write_bytes(b"\x7fELF libvulkan.so.1")   # what the fallback built
            code, text = self.run_setup()
        self.assertEqual(code, 0, text)
        self.assertEqual(calls, ["cuda", "vulkan"])            # nothing built the second time
        self.assertIn("the CUDA build failed in the last run", text)
        self.assertEqual(setup_flow.load_config()["backend"], "vulkan")

    def test_explicit_cuda_with_a_toolkit_that_cannot_build_stops_before_make(self):
        tc = dict(TC_ALL, **nvcc_toolkit("13.0.88"))
        with self.v100_machine(), mock.patch.object(setup_flow, "toolchain", return_value=tc), \
             mock.patch.object(setup_flow, "build_engine") as build:
            code, text = self.run_setup(pick="tiny", backend="cuda")
        self.assertEqual(code, 1)
        build.assert_not_called()
        self.assertIn("cannot build for the Tesla V100-SXM2-16GB (compute 7.0, dropped in CUDA 13)",
                      setup_flow.read_state()["message"])
        # auto takes Vulkan on the same machine, and says why.
        with self.v100_machine(), mock.patch.object(setup_flow, "toolchain", return_value=tc), \
             mock.patch.object(setup_flow, "build_engine", side_effect=self.failing_build()[0]):
            code, text = self.run_setup(pick="tiny", backend="auto")
        self.assertEqual(code, 0, text)
        self.assertIn("Engine: qwen36 with VULKAN (the CUDA 13.0 toolkit cannot build for the "
                      "Tesla V100-SXM2-16GB (compute 7.0, dropped in CUDA 13): using Vulkan; "
                      "install a CUDA 12.x toolkit for the CUDA path)", text)
        self.assertIn("GPU     Tesla V100-SXM2-16GB (NVIDIA, 17.2 GB VRAM, compute 7.0, "
                      "driver 580.178.04)", text)

    def test_wsl_download_through_windows_when_faster(self):
        entry = setup_catalog.by_id("tiny")
        files = [{"path": "model-00000.safetensors", "size": 400_000}]
        ui = setup_flow.UI(False, stream=io.StringIO())
        with mock.patch.object(setup_download, "windows_curl", return_value="/mnt/c/curl.exe"), \
             mock.patch.object(setup_download, "probe_throughput", return_value=63e3), \
             mock.patch.object(setup_download, "probe_throughput_windows", return_value=3.3e6):
            self.assertEqual(setup_flow._maybe_windows_transport(ui, entry, files, None, "auto"),
                             "/mnt/c/curl.exe")
        with mock.patch.object(setup_download, "windows_curl", return_value="/mnt/c/curl.exe"), \
             mock.patch.object(setup_download, "probe_throughput", return_value=50e6), \
             mock.patch.object(setup_download, "probe_throughput_windows", return_value=40e6):
            self.assertIsNone(setup_flow._maybe_windows_transport(ui, entry, files, None, "auto"))
        with mock.patch.object(setup_download, "windows_curl", return_value=None):
            self.assertIsNone(setup_flow._maybe_windows_transport(ui, entry, files, None, "yes"))

    def test_existing_model_dir_skips_the_download(self):
        existing = os.path.join(self.tmp.name, "mine")
        os.makedirs(existing)
        Path(existing, "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe_text"}))
        Path(existing, "tokenizer.json").write_text("{}")
        code, text = self.run_setup(model_dir=existing, backend="cpu")
        self.assertEqual(code, 0, text)
        cfg = setup_flow.load_config()
        self.assertEqual(cfg["model_dir"], existing)
        self.assertEqual(cfg["backend"], "cpu")
        self.assertEqual(cfg["env"], {})
        self.assertFalse(any("/cdn/" in r["path"] for r in self.hub.requests))

    def test_the_menu(self):
        rows = setup_catalog.recommend(27 * 10**9, 800 * 10**9)
        out = io.StringIO()
        ui = setup_flow.UI(True, stream=out)
        with mock.patch("builtins.input", side_effect=["", "99", "2"]):
            first = setup_flow.choose_model(ui, rows)
            second = setup_flow.choose_model(ui, rows)
        self.assertEqual(first.id, "tiny")                       # Enter takes the recommendation
        self.assertEqual(second.id, [r for r in rows if r["fits"]][1]["entry"].id)
        menu = out.getvalue()
        self.assertIn("[recommended]", menu)
        self.assertIn(setup_catalog.FITS_EXPLAINED, menu)
        self.assertIn("type a number from 1", menu)

    def test_json_mode_prints_the_configuration(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = setup_flow.run(lambda a: setup_flow.cmd_setup(a, ui=setup_flow.UI(False, quiet=True)),
                                  setup_args(dir=self.models, pick="tiny", json=True))
        self.assertEqual(code, 0)
        result = json.loads(out.getvalue())
        self.assertEqual(result["config"]["backend"], "vulkan")
        self.assertEqual(result["config_path"], setup_flow.config_path())
        self.assertNotIn("status", result)                     # --no-start

    def test_no_compiler_and_no_prebuilt_says_what_to_install(self):
        tc = dict(TC_ALL, source_checkout=False, can_build=False, can_build_vulkan=False)
        os.remove(os.path.join(self.engines, "qwen36"))
        with modeled("linux", machine="riscv64"):               # no release archive for it
            with self.assertRaises(setup_flow.SetupError) as caught:
                setup_flow.resolve_engine(family_by_id("qwen36"), None,
                                          {"backend": "cpu", "missing": []}, tc, out=lambda *_: None)
        self.assertIn("no prebuilt engine", str(caught.exception))

    def test_a_processor_without_avx2_is_not_handed_the_prebuilt_engine(self):
        # #1979: the prebuilt engines assume x86-64-v3 and stopped on their first AVX2
        # instruction on a Xeon E3-1230 V2. Without a compiler setup says so, with what
        # to install, instead of fetching an archive that cannot run.
        tc = dict(TC_ALL, source_checkout=False, can_build=False, can_build_vulkan=False)
        os.remove(os.path.join(self.engines, "qwen36"))
        old = {"arch": "x86_64", "name": "Intel Xeon E3-1230 V2", "features": []}
        fetched = setup_flow.SetupError("fetched")
        with modeled("linux"), mock.patch.object(setup_hw, "detect_cpu", return_value=old), \
             mock.patch.object(setup_flow, "fetch_release_archive", side_effect=fetched) as fetch:
            with self.assertRaises(setup_flow.SetupError) as caught:
                setup_flow.resolve_engine(family_by_id("qwen36"), None,
                                          {"backend": "cpu", "missing": []}, tc, out=lambda *_: None)
            self.assertIn("no AVX2", str(caught.exception))
            self.assertIn(setup_flow.package_hint(["build"]), str(caught.exception))
            fetch.assert_not_called()
            # COLI_CPU_CHECK=0 trusts the archive, as the engines do
            with mock.patch.dict(os.environ, {"COLI_CPU_CHECK": "0"}):
                with self.assertRaisesRegex(setup_flow.SetupError, "fetched"):
                    setup_flow.resolve_engine(family_by_id("qwen36"), None,
                                              {"backend": "cpu", "missing": []}, tc, out=lambda *_: None)

    def test_list_json(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = setup_flow.run(setup_flow.cmd_setup, setup_args(list=True, json=True))
        self.assertEqual(code, 0)
        rows = json.loads(out.getvalue()[out.getvalue().index("["):])
        self.assertEqual(rows[0]["id"], "tiny")
        self.assertTrue(rows[0]["recommended"])


FAKE_SERVER = r'''
import http.server, json, os, signal, sys, tempfile
from unittest import mock
port = int(sys.argv[sys.argv.index("--port") + 1])
pidfile = os.path.join(tempfile.gettempdir(), f"coli-serve-{port}.pid")
with open(pidfile, "w") as f:
    f.write(f"{os.getpid()} fake\n")
with open(os.environ["FAKE_ENV_OUT"], "w") as f:
    json.dump({"argv": sys.argv[1:], "COLI_VULKAN": os.environ.get("COLI_VULKAN"),
               "EXTRA": os.environ.get("EXTRA")}, f)
PAGES = {"/health": {"status": "ok", "arch": "qwen36"},
         "/profile": {"seq": 1, "turns": [{"wall_s": 2.0, "completion_tokens": 10}]},
         "/v1/models": {"object": "list", "data": [{"id": "fake-model"}]}}
class H(http.server.BaseHTTPRequestHandler):
    server_version = "colibri"
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps(PAGES.get(self.path, {})).encode()
        self.send_response(200 if self.path in PAGES else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
def bye(*_):
    try: os.unlink(pidfile)
    except OSError: pass
    os._exit(0)
signal.signal(signal.SIGTERM, bye)
# HTTPServer resolves its server_name during bind. This loopback fixture does
# not use DNS, which can stall each startup on macOS CI.
with mock.patch("socket.getfqdn", return_value="localhost"):
    server = http.server.HTTPServer(("127.0.0.1", port), H)
server.serve_forever()
'''


class ServerControl(HomeTestCase):
    """Runs on Windows too: `coli stop` ends the stand-in with TerminateProcess
    there (its SIGTERM handler simply never runs) and removes the pidfile itself."""

    def setUp(self):
        super().setUp()
        self.launcher = os.path.join(self.tmp.name, "fake_coli.py")
        Path(self.launcher).write_text(FAKE_SERVER)
        self.env_out = os.path.join(self.tmp.name, "env.json")
        self.port = free_port()
        self.cfg = setup_flow.save_config({
            "status": "ready", "model": {"name": "Fake"}, "model_dir": self.tmp.name, "family": "qwen36",
            "launcher": self.launcher, "engine": self.launcher, "backend": "vulkan",
            "env": {"COLI_VULKAN": "1", "EXTRA": "from-config"},
            "args": ["web", "--model", self.tmp.name, "--host", "127.0.0.1", "--port", str(self.port)],
            "host": "127.0.0.1", "port": self.port, "urls": setup_flow.urls("127.0.0.1", self.port)})
        patcher = mock.patch.dict(os.environ, {"FAKE_ENV_OUT": self.env_out, "EXTRA": "from-user"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.pids = []              # pids start_server returned: known before the pidfile exists
        self.addCleanup(self.cleanup_server)

    def cleanup_server(self):
        pidfile = setup_flow.serve_pidfile(self.port)
        pids = set(self.pids)
        try:
            pids.add(int(Path(pidfile).read_text().split()[0]))
        except (OSError, ValueError, IndexError):   # IndexError: created, not yet written
            pass
        for pid in pids:
            try:
                os.kill(pid, 15)    # TerminateProcess on Windows: the handler never runs
            except OSError:
                pass
        # On Windows TerminateProcess returns before the process is gone, and a
        # server that still has serve.log open keeps the folder from being removed.
        deadline = time.time() + 10
        while any(setup_flow._pid_alive(pid) for pid in pids) and time.time() < deadline:
            time.sleep(0.05)
        self.pids.clear()
        try:
            os.unlink(pidfile)      # so no stale pidfile outlives the test there
        except OSError:
            pass

    def wait_state(self, cfg, state, timeout=60):
        # A deadline, not a delay: a quick machine returns at once. A loaded CI
        # runner (macOS, where the whole Python suite takes over 8 minutes) can
        # need well over 10 s to start the stand-in server's interpreter.
        deadline = time.time() + timeout
        status = setup_flow.server_status(cfg)
        while status["state"] != state and time.time() < deadline:
            time.sleep(0.1)
            status = setup_flow.server_status(cfg)
        return status

    def test_start_status_stop(self):
        self.assertEqual(setup_flow.server_status(self.cfg)["state"], "stopped")
        status = setup_flow.start_server(self.cfg, background=True, open_browser=False,
                                         out=lambda *_: None, wait=10)
        self.assertIn(status["state"], ("ready", "loading"))
        status = self.wait_state(self.cfg, "ready")
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["model_id"], "fake-model")
        self.assertEqual(status["tokens_per_second"], 5.0)
        seen = json.loads(Path(self.env_out).read_text())
        self.assertEqual(seen["COLI_VULKAN"], "1")
        self.assertEqual(seen["EXTRA"], "from-user")             # the user's own environment wins
        self.assertIn("--no-browser", seen["argv"])
        self.assertIn("--auto-tier", seen["argv"])   # a setup recorded before #2003 applies its plan too
        self.assertTrue(os.path.exists(setup_flow.log_path("serve")))
        # Starting twice does not start a second server.
        again = setup_flow.start_server(self.cfg, background=True, open_browser=False,
                                        out=lambda *_: None)
        self.assertEqual(again["state"], "ready")
        report = setup_flow.status_report()
        self.assertEqual(report["server"]["state"], "ready")
        self.assertTrue(report["setup"]["ready"])
        result = setup_flow.stop_server(self.cfg, out=lambda *_: None)
        self.assertTrue(result["stopped"], result)
        self.assertEqual(self.wait_state(self.cfg, "stopped")["state"], "stopped")

    def test_a_taken_port_moves_to_the_next_free_one(self):
        self.port = quiet_port_pair()      # pick_port tries the next one: keep it out of reach
        args = list(self.cfg["args"])
        args[args.index("--port") + 1] = str(self.port)
        self.cfg = setup_flow.save_config(dict(self.cfg, args=args, port=self.port,
                                               urls=setup_flow.urls("127.0.0.1", self.port)))
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", self.port))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        with mock.patch.object(setup_flow, "_get_json", return_value=(None, None)):
            status = setup_flow.start_server(self.cfg, background=True, open_browser=False,
                                             out=lambda *_: None, wait=0.5)
        self.pids.append(status["pid"])   # after half a second its pidfile may not be written yet
        moved = setup_flow.load_config()
        self.assertNotEqual(moved["port"], self.port)
        self.assertEqual(moved["args"][moved["args"].index("--port") + 1], str(moved["port"]))
        self.port = moved["port"]                                 # cleanup stops this one

    def test_foreign_http_health_does_not_prevent_starting_the_configured_server(self):
        for payload, product in (({"service": "other program"}, "other"),
                                 (["healthy"], "other"),
                                 ({"status": "ok"}, "uvicorn"),
                                 ({"status": "ok"}, "colibri-extra"),
                                 ({"status": "ok"}, "not-colibri")):
            with self.subTest(payload=payload, product=product):
                class ForeignHealth(SimpleHTTPRequestHandler):
                    server_version = product
                    def log_message(self, *_): pass
                    def do_GET(self):
                        body = json.dumps(payload).encode()
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                # Avoid irrelevant reverse DNS during fixture construction;
                # binding, HTTP health checks and server readiness stay real.
                with mock.patch("socket.getfqdn", return_value="localhost"):
                    blocker = ThreadingHTTPServer(("127.0.0.1", 0), ForeignHealth)
                thread = threading.Thread(target=blocker.serve_forever, daemon=True)
                thread.start()
                try:
                    occupied = blocker.server_port
                    args = list(self.cfg["args"])
                    args[args.index("--port") + 1] = str(occupied)
                    cfg = setup_flow.save_config(dict(self.cfg, args=args, port=occupied))
                    initial = setup_flow.server_status(cfg)
                    setup_flow.start_server(cfg, background=True, open_browser=False,
                                            out=lambda *_: None, wait=10)
                    moved = setup_flow.load_config()
                    self.port = moved["port"]
                    self.assertNotEqual(self.port, occupied)
                    self.assertEqual(initial["state"], "stopped")
                    self.assertEqual(self.wait_state(moved, "ready")["state"], "ready")
                    self.cleanup_server()
                finally:
                    blocker.shutdown()
                    blocker.server_close()
                    thread.join(timeout=2)

    def test_finish_reports_selected_port_after_fallback(self):
        for json_mode in (True, False):
            with self.subTest(json=json_mode):
                class OccupiedPort(SimpleHTTPRequestHandler):
                    def log_message(self, *_): pass
                    def do_GET(self):
                        body = b'{"service":"other"}'
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                with mock.patch("socket.getfqdn", return_value="localhost"):
                    server = ThreadingHTTPServer(("127.0.0.1", 0), OccupiedPort)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    occupied = server.server_port
                    cfg = setup_flow.save_config(dict(self.cfg, port=occupied,
                        urls=setup_flow.urls("127.0.0.1", occupied),
                        args=setup_flow._replace_arg(self.cfg["args"], "--port", str(occupied))))
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = setup_flow._finish(setup_flow.UI(False, stream=output), cfg,
                            setup_args(json=json_mode, no_start=False, background=True))
                    moved = setup_flow.load_config()
                    self.port = moved["port"]
                    self.assertEqual(code, 0)
                    self.assertNotEqual(moved["port"], occupied)
                    self.assertEqual(self.wait_state(moved, "ready")["state"], "ready")
                    if json_mode:
                        result = json.loads(output.getvalue())
                        self.assertEqual(result["config"], moved)
                        self.assertEqual(result["status"]["urls"], moved["urls"])
                    self.assertEqual(setup_flow.read_state()["urls"], moved["urls"])
                finally:
                    current = setup_flow.load_config()
                    if current:
                        self.port = current["port"]
                    self.cleanup_server()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

    def test_finish_keeps_config_when_the_port_does_not_change(self):
        for json_mode in (True, False):
            with self.subTest(json=json_mode):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(setup_flow._finish(setup_flow.UI(False, stream=output), self.cfg,
                        setup_args(json=json_mode, no_start=False, background=True)), 0)
                self.assertEqual(setup_flow.load_config(), self.cfg)
                self.assertEqual(setup_flow.read_state()["urls"], self.cfg["urls"])
                if json_mode:
                    self.assertEqual(json.loads(output.getvalue())["config"], self.cfg)

    def test_finish_no_start_does_not_launch_or_write_started_state(self):
        for json_mode in (True, False):
            with self.subTest(json=json_mode):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(setup_flow._finish(setup_flow.UI(False, stream=output), self.cfg,
                        setup_args(json=json_mode, no_start=True, background=True)), 0)
                self.assertFalse(Path(self.env_out).exists())
                self.assertIsNone(setup_flow.read_state())
                self.assertEqual(setup_flow.load_config(), self.cfg)
                if json_mode:
                    result = json.loads(output.getvalue())
                    self.assertEqual(result["config"], self.cfg)
                    self.assertNotIn("status", result)

    def test_old_colibri_product_header_still_marks_the_server_ready(self):
        class OldColibri(SimpleHTTPRequestHandler):
            server_version = "colibri"
            sys_version = "Python/3.8.10"
            def log_message(self, *_): pass
            def do_GET(self):
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), OldColibri)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.assertEqual(setup_flow.server_status(dict(self.cfg, port=server.server_port))["state"],
                             "ready")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_live_pid_without_health_is_still_loading(self):
        pidfile = Path(setup_flow.serve_pidfile(self.port))
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            pidfile.write_text(str(process.pid))
            result = setup_flow.server_status(self.cfg)
            self.assertEqual(result["state"], "loading")
            self.assertEqual(result["pid"], process.pid)
        finally:
            pidfile.unlink(missing_ok=True)
            process.terminate()
            process.wait(timeout=5)

    def test_actual_protected_colibri_health_remains_ready_without_a_pidfile(self):
        from openai_server import APIServer
        server = APIServer(("127.0.0.1", 0), None, "health-control", api_key="fixture-key")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            cfg = dict(self.cfg, port=server.server_port)
            self.assertFalse(os.path.exists(setup_flow.serve_pidfile(server.server_port)))
            code, health = setup_flow._get_json(f"http://127.0.0.1:{server.server_port}/health")
            self.assertEqual((code, health), (200, {"status": "ok"}))
            self.assertEqual(setup_flow.server_status(cfg)["state"], "ready")
        finally:
            server.shutdown()
            server.server_close()
            server.scheduler.close()
            thread.join(timeout=2)


class ForegroundLogs(HomeTestCase):
    """#1852: the server the setup starts in the foreground prints to its terminal,
    and `coli logs` (from another terminal) said only "no serve log yet"."""

    def logs(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = setup_flow.cmd_logs(argparse.Namespace(install=False, lines=100))
        return code, out.getvalue()

    def test_a_foreground_start_leaves_a_line_saying_where_its_output_is(self):
        Path(setup_flow._ensure_log("serve")).write_text("an older background run's lines\n")
        setup_flow.note_foreground_start(4242, ["python3", "coli", "web", "--port", "8000"])
        code, text = self.logs()
        self.assertEqual(code, 0)
        last = text.strip().splitlines()[-2:]
        self.assertIn("in the foreground (pid 4242): python3 coli web --port 8000", last[0])
        self.assertIn("its output goes to the terminal that started it", last[1])
        self.assertIn("`coli start --background` writes it here", last[1])

    def test_no_log_but_a_running_server_says_where_to_look(self):
        setup_flow.save_config({"status": "ready", "port": 8000, "host": "127.0.0.1"})
        with mock.patch.object(setup_flow, "server_status",
                               return_value={"state": "ready", "pid": 18985}):
            code, text = self.logs()
        self.assertEqual(code, 1)
        self.assertIn("no serve log yet", text)
        self.assertIn("colibri is running (ready, pid 18985), started in the foreground: "
                      "its output is in the terminal that started it", text)

    def test_no_log_and_nothing_running_says_only_that(self):
        setup_flow.save_config({"status": "ready", "port": 8000, "host": "127.0.0.1"})
        with mock.patch.object(setup_flow, "server_status", return_value={"state": "stopped", "pid": None}):
            code, text = self.logs()
        self.assertEqual(code, 1)
        self.assertEqual(text.strip().splitlines()[-1].startswith("no serve log yet"), True)
        self.assertNotIn("running", text)


class SplitModel(HomeTestCase):
    """A model split across drives (#2091): shards moved out of model_dir to a
    folder the engine reads through its family's variable (K3_DIRS for Kimi K3)
    are still the model's for the launcher."""

    SPLIT_VARS = ("K3_DIRS", "COLI_MODEL_DIRS", "MIMO_DIRS")

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in self.SPLIT_VARS:
            os.environ.pop(name, None)
        self.model = os.path.join(self.tmp.name, "kimi-k3")
        self.other = os.path.join(self.tmp.name, "second drive")
        os.makedirs(self.model)
        os.makedirs(self.other)
        files = {"config.json": b"{}", "model-00001-of-00002.safetensors": b"a" * 64,
                 "model-00002-of-00002.safetensors": b"b" * 96}
        for name, data in files.items():
            Path(self.model, name).write_bytes(data)
        setup_download._write_manifest(self.model, {
            "repo": "tester/k3", "revision": "main", "complete": True,
            "files": [{"path": name, "size": len(data)} for name, data in files.items()]})
        os.replace(os.path.join(self.model, "model-00002-of-00002.safetensors"),
                   os.path.join(self.other, "model-00002-of-00002.safetensors"))
        engines = os.path.join(self.tmp.name, "c")
        os.makedirs(engines)
        Path(engines, "kimi_k3").write_bytes(b"engine")
        Path(engines, "coli").write_text("# launcher")
        self.cfg = {"status": "ready", "family": "kimi", "model_dir": self.model,
                    "model": {"id": None, "name": "Kimi K3", "repo": "tester/k3"},
                    "engine": os.path.join(engines, "kimi_k3"), "launcher": os.path.join(engines, "coli"),
                    "env": {}, "args": [], "host": "127.0.0.1", "port": 8000, "backend": "cpu",
                    "urls": setup_flow.urls("127.0.0.1", 8000)}

    def test_the_registry_names_each_engine_s_variable(self):
        names = {f: family_by_id(f).split_dirs_env for f in
                 ("kimi", "mimo", "glm", "deepseek_v4", "deepseek_v41", "qwen36", "qwen38")}
        self.assertEqual(names, {"kimi": "K3_DIRS", "mimo": "MIMO_DIRS", "glm": "COLI_MODEL_DIRS",
                                 "deepseek_v4": "COLI_MODEL_DIRS", "deepseek_v41": "COLI_MODEL_DIRS",
                                 "qwen36": "", "qwen38": ""})

    def test_ready_where_the_engine_reads_the_shards(self):
        self.assertFalse(setup_flow.config_ready(self.cfg), "a shard is missing from the model folder")
        os.environ["K3_DIRS"] = self.other
        self.assertTrue(setup_flow.config_ready(self.cfg), "K3_DIRS in the environment")
        del os.environ["K3_DIRS"]
        saved = dict(self.cfg, env={"K3_DIRS": f"/nowhere;{self.other}"})
        self.assertTrue(setup_flow.config_ready(saved), "K3_DIRS in the run configuration")
        os.environ["K3_DIRS"] = "/nowhere"
        self.assertFalse(setup_flow.config_ready(saved), "the environment wins, as in start_server")
        del os.environ["K3_DIRS"]
        os.environ["COLI_MODEL_DIRS"] = self.other
        self.assertFalse(setup_flow.config_ready(self.cfg), "another engine's variable: Kimi K3 does not read it")
        self.assertTrue(setup_flow.config_ready(dict(self.cfg, family="glm")), "the GLM engine reads COLI_MODEL_DIRS")
        self.assertFalse(setup_flow.config_ready(dict(self.cfg, family="qwen36")), "qwen36 reads one folder")

    def test_setup_keeps_the_folders_for_coli_start(self):
        setup_flow.save_config(self.cfg)
        os.environ["K3_DIRS"] = self.other
        out = io.StringIO()
        with modeled("linux"), contextlib.redirect_stdout(out):
            self.assertEqual(setup_flow.cmd_setup(setup_args()), 0)
        said = out.getvalue()
        self.assertIn("Already set up", said)
        self.assertIn(f"shards also read from: {self.other} (K3_DIRS, kept for coli start)", said)
        self.assertEqual(setup_flow.load_config()["env"]["K3_DIRS"], self.other)
        del os.environ["K3_DIRS"]
        self.assertTrue(setup_flow.config_ready(setup_flow.load_config()), "coli start from a shell without K3_DIRS")
        cfg = setup_flow.load_config()
        self.assertIs(setup_flow.remember_split_dirs(cfg, out=lambda *_: None), cfg, "nothing new to keep")


class ConfiguredRefresh(HomeTestCase):
    """refresh_configured: the engine a ready setup starts, checked against the sources."""

    def setUp(self):
        super().setUp()
        self.engines = os.path.join(self.tmp.name, "c")
        os.makedirs(self.engines)
        self.engine = Path(self.engines, "qwen36")
        self.engine.write_bytes(b"\x7fELF libvulkan.so.1")
        for patcher in (modeled("linux"), mock.patch.object(setup_flow, "HERE", self.engines)):
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)
        self.cfg = {"family": "qwen36", "engine": str(self.engine), "port": 8000, "host": "127.0.0.1"}

    def refresh(self, state="stopped", cfg=None):
        with mock.patch.object(setup_flow, "server_status", return_value={"state": state, "pid": None}), \
             mock.patch.object(setup_flow, "build_engine", return_value=str(self.engine)) as build:
            result = setup_flow.refresh_configured(cfg or self.cfg, out=lambda *_: None, tc=TC_ALL)
        return result, build

    def test_a_stopped_server_gets_its_engine_checked(self):
        result, build = self.refresh()
        self.assertEqual(result, "present")
        self.assertEqual(build.call_args.args[1], "vulkan")

    def test_a_running_server_keeps_the_engine_it_started_with(self):
        for state in ("ready", "loading"):
            result, build = self.refresh(state)
            self.assertIsNone(result)
            build.assert_not_called()

    def test_a_prebuilt_engine_from_a_release_is_left_alone(self):
        elsewhere = Path(self.tmp.name, "release", "qwen36")
        elsewhere.parent.mkdir()
        elsewhere.write_bytes(b"plain")
        result, build = self.refresh(cfg=dict(self.cfg, engine=str(elsewhere)))
        self.assertIsNone(result)
        build.assert_not_called()

    def test_a_cuda_engine_is_updated_with_the_variables_for_its_cards(self):
        self.engine.write_bytes(b"\x7fELF libcudart.so.12")
        cfg = dict(self.cfg, nvidia=[{"name": "Tesla V100-SXM2-16GB", "compute_cap": "7.0"}])
        tc = dict(TC_ALL, cuda_version="11.8")
        with mock.patch.object(setup_flow, "server_status", return_value={"state": "stopped", "pid": None}), \
             mock.patch.object(setup_flow, "build_engine", return_value=str(self.engine)) as build:
            setup_flow.refresh_configured(cfg, out=lambda *_: None, tc=tc)
        self.assertEqual(build.call_args.args[1], "cuda")
        self.assertEqual(build.call_args.kwargs["make_args"],
                         setup_flow.cuda_make_args(family_by_id("qwen36"), tc, cfg["nvidia"]))


class ConfiguredCuda(Releases):
    """A saved run configuration that is not on CUDA where the engine runs CUDA now
    (#2091): the setup's own Vulkan choice kept COLI_VULKAN=1 on a CUDA box."""

    CARD = {"name": "NVIDIA GeForce RTX 5080", "compute_cap": "12.0"}

    def configured(self, engines, backend="vulkan", requested="auto", cards=None, family="qwen36"):
        model = os.path.join(self.tmp.name, "model")
        os.makedirs(model, exist_ok=True)
        Path(engines, "coli").write_text("# launcher")
        name = family_by_id(family).engine_artifact + ".exe"
        engine = os.path.join(engines, name)
        info = {"launcher_dir": engines, "engine": engine, "backend": backend, "source": "present"}
        with modeled("win32"):
            cfg = setup_flow.make_config(model_dir=model, family=family_by_id(family), entry=None,
                                         engine_info=info, hw=hw_report(vulkan=DGPU), host="127.0.0.1",
                                         port=8000)
        cfg.update(requested_backend=requested, nvidia=[self.CARD] if cards is None else cards)
        return setup_flow.save_config(cfg)

    def windows_engines(self, folder, dll=True):
        os.makedirs(os.path.join(folder, "shaders"), exist_ok=True)
        Path(folder, "shaders", "qmatmul.spv").write_bytes(b"\x03\x02\x23\x07")
        Path(folder, "qwen36.exe").write_bytes(b"engine coli_cuda.dll vulkan-1.dll" if dll
                                               else b"engine vulkan-1.dll")
        if dll:
            Path(folder, "coli_cuda.dll").write_bytes(b"MZ")

    def rerun(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(setup_flow.cmd_setup(setup_args()), 0)
        return out.getvalue()

    def test_the_cuda_package_unpacked_after_the_setup_is_taken(self):
        engines = os.path.join(self.tmp.name, "colibri-2.1.0")
        self.windows_engines(engines)
        self.configured(engines)
        self.assertEqual(setup_flow.load_config()["env"]["COLI_VULKAN"], "1")
        with modeled("win32"):
            said = self.rerun()
        cfg = setup_flow.load_config()
        self.assertEqual(cfg["backend"], "cuda")
        self.assertEqual(cfg["gpu"], self.CARD["name"])
        self.assertNotIn("COLI_VULKAN", cfg["env"])
        self.assertNotIn("COLI_VK_SHADERS", cfg["env"])
        self.assertEqual(cfg["args"][cfg["args"].index("--gpu") + 1], "auto")
        self.assertEqual(cfg["args"].count("--auto-tier"), 1)
        self.assertIn("CUDA from now on: NVIDIA GeForce RTX 5080 (compute 12.0) with coli_cuda.dll "
                      "beside the engine; the last setup saved Vulkan.", said)
        with modeled("win32"):     # and once it is on CUDA, nothing more to say
            self.assertNotIn("CUDA from now on", self.rerun())

    def test_coli_start_takes_it_too(self):
        engines = os.path.join(self.tmp.name, "colibri-2.1.0")
        self.windows_engines(engines)
        self.configured(engines)
        out = io.StringIO()
        with modeled("win32"), contextlib.redirect_stdout(out), \
             mock.patch.object(setup_flow, "start_server", return_value=0) as start:
            setup_flow.cmd_start(argparse.Namespace(json=False, background=False, no_browser=True))
        self.assertEqual(start.call_args.args[0]["backend"], "cuda")
        self.assertNotIn("COLI_VULKAN", start.call_args.args[0]["env"])
        self.assertEqual(setup_flow.load_config()["backend"], "cuda")
        self.assertIn("CUDA from now on", out.getvalue())

    def test_a_backend_asked_for_or_a_card_the_dll_does_not_run_on_stays(self):
        engines = os.path.join(self.tmp.name, "colibri-2.1.0")
        self.windows_engines(engines)
        for kwargs in ({"requested": "vulkan"}, {"requested": "cpu", "backend": "cpu"},
                       {"cards": [{"name": "NVIDIA GeForce RTX 2070", "compute_cap": "7.5"}]},
                       {"cards": []}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                cfg = self.configured(engines, **kwargs)
                with modeled("win32"):
                    self.assertIsNone(setup_flow.cuda_unused(cfg))
                    self.assertIsNone(setup_flow.release_cuda_missing(cfg))
        without = os.path.join(self.tmp.name, "no-dll")
        self.windows_engines(without, dll=False)
        with modeled("win32"):
            self.assertIsNone(setup_flow.cuda_unused(self.configured(without)))

    def test_a_release_engine_from_an_older_setup_gets_its_cuda_backend(self):
        # the reporter's state: the release's Vulkan engines in runtime/, COLI_VULKAN=1
        runtime = os.path.join(self.home, "runtime", "colibri-v2.1.0")
        self.windows_engines(runtime, dll=False)
        self.configured(runtime)
        getter = self.windows_release("v2.1.0", "release")
        with modeled("win32"), mock.patch.object(setup_flow, "_api_get", side_effect=getter):
            said = self.rerun()
        cfg = setup_flow.load_config()
        self.assertIn("Release v2.1.0 has a CUDA backend for the NVIDIA card: fetching it.", said)
        self.assertEqual(cfg["backend"], "cuda")
        self.assertTrue(os.path.isfile(os.path.join(runtime, "coli_cuda.dll")))
        self.assertNotIn("COLI_VULKAN", cfg["env"])

    def test_a_release_without_a_cuda_backend_is_asked_once(self):
        runtime = os.path.join(self.home, "runtime", "colibri-v2.1.0")
        self.windows_engines(runtime, dll=False)
        self.configured(runtime)
        getter = mock.Mock(side_effect=self.windows_release("v2.1.0", "release", cuda=False))
        with modeled("win32"), mock.patch.object(setup_flow, "_api_get", getter):
            self.assertIn("release v2.1.0 has no windows-x86_64-cuda.zip: keeping Vulkan", self.rerun())
            asked = getter.call_count
            self.assertNotIn("fetching it", self.rerun())
        self.assertEqual(getter.call_count, asked)
        self.assertEqual(setup_flow.load_config()["backend"], "vulkan")

    def test_an_unreachable_release_is_asked_again_and_the_setup_goes_on(self):
        runtime = os.path.join(self.home, "runtime", "colibri-v2.1.0")
        self.windows_engines(runtime, dll=False)
        self.configured(runtime)
        offline = mock.Mock(side_effect=setup_flow.urllib.error.URLError("offline"))
        with modeled("win32"), mock.patch.object(setup_flow, "_api_get", offline):
            said = self.rerun()
            self.assertIn("the next setup tries again", said)
            self.assertIn("Already set up", said)
            self.rerun()
        self.assertGreater(offline.call_count, 2)
        self.assertEqual(setup_flow.load_config()["backend"], "vulkan")


class CommandLine(HomeTestCase):
    def coli(self, *args):
        return subprocess.run([sys.executable, str(C_DIR / "coli"), *args], capture_output=True,
                              text=True, timeout=120, env=dict(os.environ, COLI_COLOR="0"))

    def test_status_and_logs_before_setup(self):
        result = self.coli("status")
        self.assertEqual(result.returncode, 1)
        self.assertIn("not set up yet", result.stdout)
        result = self.coli("status", "--json")
        self.assertEqual(json.loads(result.stdout)["configured"], False)
        self.assertEqual(self.coli("logs").returncode, 1)

    def test_setup_help_lists_the_options(self):
        result = self.coli("setup", "--help")
        self.assertEqual(result.returncode, 0)
        for flag in ("--yes", "--model", "--model-dir", "--dir", "--backend", "--no-gpu",
                     "--background", "--no-start", "--via-windows", "--json", "--ctx"):
            self.assertIn(flag, result.stdout)

    def test_start_before_setup_says_what_to_do(self):
        result = self.coli("start")
        self.assertEqual(result.returncode, 1)
        self.assertIn("coli setup", result.stderr)


class LinuxVulkanHeadersTests(unittest.TestCase):
    """#2122: a Nix shell's vulkan.h sits in the store, outside /usr/include and
    /usr/local/include; the compiler the build uses finds it, so setup asks it."""

    def test_the_compiler_finds_a_header_on_its_include_path(self):
        cc = shutil.which("cc") or shutil.which("gcc")
        if not cc:
            self.skipTest("no C compiler")
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "vulkan"))
            open(os.path.join(tmp, "vulkan", "vulkan.h"), "w").close()
            with mock.patch.dict(os.environ, {"CPATH": tmp}):
                self.assertTrue(setup_flow._cc_finds_vulkan_h(cc))
        self.assertFalse(setup_flow._cc_finds_vulkan_h(None))
        self.assertFalse(setup_flow._cc_finds_vulkan_h("/nonexistent/cc"))

    def test_toolchain_takes_the_compilers_answer(self):
        real_isfile = os.path.isfile
        def isfile(path):   # nothing in the usual directories
            return False if path.endswith(os.path.join("vulkan", "vulkan.h")) else real_isfile(path)
        with mock.patch.object(setup_flow.sys, "platform", "linux"), \
                mock.patch.object(setup_flow.os.path, "isfile", side_effect=isfile), \
                mock.patch.object(setup_flow, "_cc_finds_vulkan_h", return_value=True) as asked:
            tc = setup_flow.toolchain(here=str(C_DIR))
        self.assertTrue(tc["vulkan_headers"])
        asked.assert_called_once()


class WindowsToolchainTests(unittest.TestCase):
    """toolchain() on Windows: a gcc and make on the PATH that cannot build the Vulkan
    backend no longer hide an MSYS2 that can (#1900)."""

    def tree(self, root, files):
        for f in files:
            path = os.path.join(root, *f.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
        return root

    def run_toolchain(self, path_tools, msys2):
        def which(name):
            return path_tools.get(name)
        with mock.patch.object(setup_flow.sys, "platform", "win32"), \
                mock.patch.object(setup_flow.shutil, "which", side_effect=which), \
                mock.patch.object(setup_flow, "find_msys2", return_value=msys2):
            return setup_flow.toolchain(here=str(C_DIR))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = self.tmp.name
        self.msys2 = self.tree(os.path.join(base, "msys64"), [
            "usr/bin/bash.exe", "usr/bin/make.exe", "ucrt64/bin/gcc.exe", "ucrt64/bin/glslc.exe",
            "ucrt64/include/vulkan/vulkan.h", "ucrt64/lib/libvulkan-1.dll.a"])
        scoop = self.tree(os.path.join(base, "scoop"), ["bin/gcc.exe", "bin/make.exe"])
        self.scoop = {"gcc": os.path.join(scoop, "bin", "gcc.exe"), "make": os.path.join(scoop, "bin", "make.exe")}

    def test_msys2_beats_a_path_compiler_without_vulkan(self):
        tc = self.run_toolchain(self.scoop, self.msys2)
        self.assertEqual(tc["msys2"], self.msys2)
        self.assertEqual(tc["cc"], os.path.join(self.msys2, "ucrt64", "bin", "gcc.exe"))
        self.assertTrue(tc["can_build_vulkan"])

    def test_a_path_compiler_with_vulkan_stays(self):
        prefix = self.tree(os.path.join(self.tmp.name, "full"),
                           ["bin/gcc.exe", "bin/make.exe", "bin/glslc.exe", "include/vulkan/vulkan.h"])
        tools = {"gcc": os.path.join(prefix, "bin", "gcc.exe"), "make": os.path.join(prefix, "bin", "make.exe"),
                 "glslc": os.path.join(prefix, "bin", "glslc.exe")}
        tc = self.run_toolchain(tools, self.msys2)
        self.assertIsNone(tc["msys2"])
        self.assertEqual(tc["cc"], tools["gcc"])
        self.assertTrue(tc["can_build_vulkan"])

    def test_an_msys2_without_its_compiler_does_not_replace_a_working_one(self):
        bare = self.tree(os.path.join(self.tmp.name, "bare"), ["usr/bin/bash.exe"])
        tc = self.run_toolchain(self.scoop, bare)
        self.assertIsNone(tc["msys2"])
        self.assertEqual(tc["cc"], self.scoop["gcc"])
        self.assertTrue(tc["can_build"])
        self.assertFalse(tc["can_build_vulkan"])

    def test_msys2_build_packages_include_libgomp(self):
        self.assertIn("mingw-w64-ucrt-x86_64-libgomp", setup_flow.PACKAGES["msys2"][1]["build"])



class PortFreeTest(unittest.TestCase):
    """A port is free when nothing answers on it AND a server could bind it now."""

    def test_a_port_held_without_listening_is_not_free(self):
        # test_foreign_http_health_does_not_prevent_starting_the_configured_server failed
        # on a busy runner (#1977's run of 2026-10-07): the port after the occupied one
        # had no listener, so it passed for free, yet a socket held it and the server
        # could not bind it.
        held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        held.bind(("127.0.0.1", 0))
        port = held.getsockname()[1]
        try:
            self.assertFalse(setup_flow.port_free("127.0.0.1", port))
        finally:
            held.close()

    def test_a_listening_port_is_not_free_and_a_closed_one_is(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            self.assertFalse(setup_flow.port_free("127.0.0.1", port))
        finally:
            server.close()
        self.assertTrue(setup_flow.port_free("127.0.0.1", free_port()))

if __name__ == "__main__":
    unittest.main()
