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
    values = dict(yes=True, json=False, pick=None, model_dir=None, dir=None, backend=None,
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
    def choose(self, hw, family="qwen36", tc=None, requested="auto", platform="linux"):
        with modeled(platform):
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
        decision = self.choose(hw_report(vulkan=rtx_vk, nvidia=[RTX]), family="mimo")
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

    def test_windows_without_msys2_points_at_it(self):
        tc = dict(TC_ALL, source_checkout=False, make=None, cc=None, can_build=False,
                  can_build_vulkan=False, can_build_cuda=False)
        decision = self.choose(hw_report(vulkan=IGPU), tc=tc, platform="win32")
        self.assertEqual(decision["backend"], "cpu")
        hint = decision["missing"][0][1]
        self.assertIn("msys2.org", hint)
        self.assertIn("mingw-w64-ucrt-x86_64-gcc", hint)
        self.assertIn("mingw-w64-ucrt-x86_64-shaderc", hint)
        self.assertIn("source checkout", hint)      # a release archive cannot rebuild itself


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

    def test_glm53_has_no_cuda_path(self):
        decision = self.choose([RTX3090], "13.0.88", family="glm53", vulkan=DGPU)
        self.assertEqual(decision["backend"], "vulkan")

    def test_make_command_takes_the_decision_arguments(self):
        with modeled("linux"), mock.patch.dict(os.environ, {"ARCH": "", "CUDA_ARCH": ""}):
            cmd, _cwd, _env = setup_flow.make_command(family_by_id("deepseek_v4"), "cuda", TC_ALL,
                                                      ["NO_TC=1"])
            self.assertEqual(cmd[cmd.index("deepseek-v4"):],
                             ["deepseek-v4", "ARCH=native", "CUDA=1", "CUDA_ARCH=native", "NO_TC=1"])
            cmd, _cwd, _env = setup_flow.make_command(family_by_id("qwen36"), "cuda", TC_ALL,
                                                      ["CUDA_ARCH=sm_70"])
            self.assertEqual(cmd[cmd.index("qwen36"):], ["qwen36", "ARCH=native", "CUDA=1", "CUDA_ARCH=sm_70"])

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
                                  (b"libcudart.so.12", "cuda"), (b"plain", "cpu")):
            path = Path(self.tmp.name, "engine")
            path.write_bytes(b"\x7fELF" + content)
            self.assertEqual(setup_flow.binary_backend(str(path)), expected)

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

    def make_archive(self, os_name):
        suffix, binaries = self.LAYOUTS[os_name]
        members = [(name, b"engine" if name.startswith("qwen36") else b"# launcher") for name in binaries]
        members += [("web/dist/index.html", b"<html></html>"), ("tools/k3_tokenizer.py", b"")]
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

    def check_prebuilt(self, os_name):
        tag = "v1.12.1"
        name = f"colibri-{tag}-{self.LAYOUTS[os_name][0]}"
        archive = self.make_archive(os_name)
        sums = f"{hashlib.sha256(archive).hexdigest()}  {name}\n".encode()
        urls = self.serve_release({name: archive, "SHA256SUMS.txt": sums}, folder=os_name)
        getter = lambda url: {"tag_name": tag, "assets": [{"name": n, "browser_download_url": u}
                                                          for n, u in urls.items()]}
        tc = dict(TC_ALL, source_checkout=False, can_build=False, can_build_vulkan=False)
        empty = os.path.join(self.tmp.name, f"nothing-here-{os_name}")
        os.makedirs(empty)
        home = os.path.join(self.tmp.name, f"home-{os_name}")
        with modeled(os_name), mock.patch.object(setup_flow, "HERE", empty), \
             mock.patch.dict(os.environ, {"COLI_SETUP_HOME": home}), \
             mock.patch.object(setup_flow, "_api_get", side_effect=getter):
            out = []
            info = setup_flow.resolve_engine(family_by_id("qwen36"), setup_catalog.by_id("qwen36-35b"),
                                             {"backend": "vulkan", "missing": []}, tc, out=out.append)
            self.assertEqual(info["backend"], "cpu")               # prebuilt engines are CPU builds
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

    def test_interrupted_download_continues_on_rerun(self):
        real = setup_download.download_repo
        self.hub.cut_after["model-00000.safetensors"] = 150_000
        with mock.patch.object(setup_download, "download_repo",
                               side_effect=lambda *a, **k: real(*a, **dict(k, retries=0))):
            code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 1)
        self.assertEqual(setup_flow.load_config()["status"], "pending")
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
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
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
            "status": "ready", "model": {"name": "Fake"}, "model_dir": self.tmp.name,
            "launcher": self.launcher, "engine": self.launcher, "backend": "vulkan",
            "env": {"COLI_VULKAN": "1", "EXTRA": "from-config"},
            "args": ["web", "--model", self.tmp.name, "--host", "127.0.0.1", "--port", str(self.port)],
            "host": "127.0.0.1", "port": self.port, "urls": setup_flow.urls("127.0.0.1", self.port)})
        patcher = mock.patch.dict(os.environ, {"FAKE_ENV_OUT": self.env_out, "EXTRA": "from-user"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.cleanup_server)

    def cleanup_server(self):
        pidfile = setup_flow.serve_pidfile(self.port)
        try:
            pid = int(Path(pidfile).read_text().split()[0])
            os.kill(pid, 15)        # TerminateProcess on Windows: the handler never runs
        except (OSError, ValueError):
            pass
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
            setup_flow.start_server(self.cfg, background=True, open_browser=False,
                                    out=lambda *_: None, wait=0.5)
        moved = setup_flow.load_config()
        self.assertNotEqual(moved["port"], self.port)
        self.assertEqual(moved["args"][moved["args"].index("--port") + 1], str(moved["port"]))
        self.port = moved["port"]                                 # cleanup stops this one


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
                     "--background", "--no-start", "--via-windows", "--json"):
            self.assertIn(flag, result.stdout)

    def test_start_before_setup_says_what_to_do(self):
        result = self.coli("start")
        self.assertEqual(result.returncode, 1)
        self.assertIn("coli setup", result.stderr)


if __name__ == "__main__":
    unittest.main()
