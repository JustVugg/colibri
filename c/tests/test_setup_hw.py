"""Hardware detection for the one-step setup, against mocked hardware.

Every probe in setup_hw.py has a pure parser behind it; these tests feed the
parsers the text and the raw structs real machines produce (Linux /proc files,
nvidia-smi, vulkaninfo, the Vulkan memory-properties layout, the Windows
system calls' results) so the detection is checked without the hardware.
"""
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup_hw  # noqa: E402

MEMINFO = """MemTotal:       26673356 kB
MemFree:         1203980 kB
MemAvailable:   23465316 kB
Buffers:          123456 kB
"""

CPUINFO_X86 = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: 13th Gen Intel(R) Core(TM) i7-1355U
flags\t\t: fpu vme sse sse2 avx f16c fma avx2 avx_vnni sha_ni
processor\t: 1
model name\t: 13th Gen Intel(R) Core(TM) i7-1355U
flags\t\t: fpu vme sse sse2 avx f16c fma avx2 avx_vnni sha_ni
"""

CPUINFO_ARM = """processor\t: 0
BogoMIPS\t: 2000.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics asimddp sve i8mm
CPU implementer\t: 0x41
"""

NVIDIA_SMI = """0, NVIDIA GeForce RTX 4070, 12282, 11520, 560.94
1, NVIDIA GB10, [N/A], [N/A], 580.65
garbage line
"""

# The query with compute_cap and pci.bus_id (setup_hw.NVIDIA_FIELDS): a V100 as
# in #1852, an RTX 3090, a GB10 whose memory is unified, and a card the driver
# gives no compute capability for.
NVIDIA_SMI_CC = """0, Tesla V100-SXM2-16GB, 16384, 16144, 580.178.04, 7.0, 00000000:1A:00.0
1, NVIDIA GeForce RTX 3090, 24576, 24253, 580.65, 8.6, 00000000:68:00.0
2, NVIDIA GB10, [N/A], [N/A], 580.65, 12.1, 0000000F:01:00.0
3, Some Card, 8192, 8000, 580.65, [N/A], 00000000:B1:00.0
"""

VULKANINFO_SUMMARY = """==========
VULKANINFO
==========

Vulkan Instance Version: 1.3.275

Devices:
========
GPU0:
\tapiVersion         = 1.3.289
\tdriverVersion      = 24.2.8
\tvendorID           = 0x1002
\tdeviceID           = 0x15bf
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = AMD Radeon 780M (RADV PHOENIX)
\tdriverID           = DRIVER_ID_MESA_RADV
GPU1:
\tapiVersion         = 1.4.318
\tdriverVersion      = 25.2.8
\tvendorID           = 0x10005
\tdeviceID           = 0x0000
\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU
\tdeviceName         = llvmpipe (LLVM 20.1.2, 256 bits)
"""

# Captured from vulkaninfo.exe --summary on Windows 11 with Intel's driver
# (device UUIDs dropped).
VULKANINFO_WINDOWS = """Devices:
========
GPU0:
\tapiVersion         = 1.4.323
\tdriverVersion      = 101.7076
\tvendorID           = 0x8086
\tdeviceID           = 0xa7a1
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = Intel(R) Iris(R) Xe Graphics
\tdriverID           = DRIVER_ID_INTEL_PROPRIETARY_WINDOWS
\tdriverName         = Intel Corporation
\tdriverInfo         = 101.7076
\tconformanceVersion = 1.4.0.0
"""


def memory_properties(heaps):
    """VkPhysicalDeviceMemoryProperties bytes: heaps = [(size, device_local)]."""
    raw = bytearray(520)
    struct.pack_into("<I", raw, 0, 1)                 # one memory type
    struct.pack_into("<II", raw, 4, 1, 0)             # DEVICE_LOCAL, heap 0
    struct.pack_into("<I", raw, 260, len(heaps))
    for index, (size, local) in enumerate(heaps):
        struct.pack_into("<QI", raw, 264 + index * 16, size, 1 if local else 0)
    return bytes(raw)


def budget_struct(budgets, usages):
    raw = bytearray(512)
    for index, (budget, usage) in enumerate(zip(budgets, usages)):
        struct.pack_into("<Q", raw, 16 + index * 8, budget)
        struct.pack_into("<Q", raw, 144 + index * 8, usage)
    return bytes(raw)


def device(name, kind, api="1.3.0"):
    major, minor, patch = (int(p) for p in api.split("."))
    return {"name": name, "type": kind, "api_version": api,
            "api_version_raw": (major << 22) | (minor << 12) | patch,
            "budget_bytes": None, "device_local_bytes": 8 * 2**30}


class MemoryParsing(unittest.TestCase):
    def test_meminfo(self):
        mem = setup_hw.memory_from_meminfo(MEMINFO)
        self.assertEqual(mem["total"], 26673356 * 1024)
        self.assertEqual(mem["available"], 23465316 * 1024)

    def test_meminfo_without_available_is_unknown_not_zero(self):
        mem = setup_hw.memory_from_meminfo("MemTotal: 1000 kB\n")
        self.assertEqual(mem["total"], 1000 * 1024)
        self.assertIsNone(mem["available"])

    def test_windows_status_uses_the_commit_limit_when_smaller(self):
        mem = setup_hw.memory_from_windows_status(32 * 2**30, 20 * 2**30, 12 * 2**30)
        self.assertEqual(mem, {"total": 32 * 2**30, "available": 12 * 2**30})
        mem = setup_hw.memory_from_windows_status(32 * 2**30, 20 * 2**30, 40 * 2**30)
        self.assertEqual(mem["available"], 20 * 2**30)

    def test_darwin(self):
        vm_stat = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                   "Pages free:                               10000.\n"
                   "Pages inactive:                           5000.\n"
                   "Pages speculative:                        1000.\n")
        mem = setup_hw.memory_from_darwin("17179869184\n", vm_stat)
        self.assertEqual(mem["total"], 17179869184)
        self.assertEqual(mem["available"], 16000 * 16384)


class CpuParsing(unittest.TestCase):
    def test_x86(self):
        cpu = setup_hw.cpu_from_cpuinfo(CPUINFO_X86)
        self.assertEqual(cpu["name"], "13th Gen Intel(R) Core(TM) i7-1355U")
        self.assertIn("avx2", cpu["features"])
        self.assertIn("avx_vnni", cpu["features"])
        self.assertNotIn("avx512f", cpu["features"])

    def test_arm(self):
        cpu = setup_hw.cpu_from_cpuinfo(CPUINFO_ARM)
        for flag in ("asimd", "asimddp", "i8mm", "sve"):
            self.assertIn(flag, cpu["features"])

    def test_windows_feature_codes(self):
        present = {40}                                  # AVX2 only
        self.assertEqual(setup_hw.cpu_features_from_windows(lambda code: code in present), ["avx2"])


class SystemParsing(unittest.TestCase):
    def test_wsl_from_kernel_release(self):
        self.assertTrue(setup_hw.is_wsl("6.18.33.2-microsoft-standard-WSL2", environ={}))
        self.assertFalse(setup_hw.is_wsl("6.8.0-45-generic", environ={}))
        self.assertTrue(setup_hw.is_wsl("6.8.0", environ={"WSL_DISTRO_NAME": "Ubuntu"}))

    def test_os_release(self):
        info = setup_hw.os_release('ID=ubuntu\nID_LIKE=debian\nPRETTY_NAME="Ubuntu 24.04 LTS"\n')
        self.assertEqual(info["ID"], "ubuntu")
        self.assertEqual(info["PRETTY_NAME"], "Ubuntu 24.04 LTS")

    def test_windows_drive_from_wsl_is_flagged(self):
        # POSIX rules whatever the host: on Windows os.path would read
        # /mnt/c/... as C:\\mnt\\c\\... and flag nothing.
        self.assertTrue(setup_hw.path_warnings("/mnt/c/Users/me/models", wsl=True))
        self.assertTrue(setup_hw.path_warnings("/mnt/d", wsl=True))
        self.assertTrue(setup_hw.path_warnings("/home/me/../../mnt/e/x", wsl=True))
        self.assertFalse(setup_hw.path_warnings("/home/me/colibri-models", wsl=True))
        self.assertFalse(setup_hw.path_warnings("/mnt/wslg/x", wsl=True))
        self.assertFalse(setup_hw.path_warnings("/mnt/c/Users/me/models", wsl=False))

    def test_a_relative_folder_is_resolved_against_the_wsl_cwd(self):
        with mock.patch.object(setup_hw.os, "getcwd", return_value="/mnt/c/Users/me"):
            self.assertTrue(setup_hw.path_warnings("models", wsl=True))
        with mock.patch.object(setup_hw.os, "getcwd", return_value="/home/me"):
            self.assertFalse(setup_hw.path_warnings("models", wsl=True))

    def test_fixed_drives(self):
        kinds = {"C:\\": 3, "D:\\": 3, "E:\\": 2, "Z:\\": 4}      # fixed, fixed, removable, network
        mask = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 25)
        self.assertEqual(setup_hw.fixed_drives_from_mask(mask, lambda root: kinds.get(root, 0)),
                         ["C:\\", "D:\\"])

    def test_disk_free_walks_up_to_an_existing_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            free = setup_hw.disk_free(os.path.join(tmp, "not", "yet", "there"))
            self.assertIsInstance(free, int)
            self.assertGreater(free, 0)

    def test_windows_video_controllers(self):
        rows = setup_hw.parse_win_video_controllers(
            "Intel(R) Iris(R) Xe Graphics|1073741824\nNVIDIA GeForce RTX 3060|\n\n")
        self.assertEqual(rows[0], {"name": "Intel(R) Iris(R) Xe Graphics",
                                   "adapter_ram_bytes": 1073741824})
        self.assertIsNone(rows[1]["adapter_ram_bytes"])


class NvidiaParsing(unittest.TestCase):
    def test_rows(self):
        gpus = setup_hw.parse_nvidia_smi(NVIDIA_SMI)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 4070")
        self.assertEqual(gpus[0]["total_bytes"], 12282 * 2**20)
        self.assertEqual(gpus[0]["driver"], "560.94")

    def test_unified_memory_is_unknown_not_zero(self):
        gpus = setup_hw.parse_nvidia_smi(NVIDIA_SMI)
        self.assertIsNone(gpus[1]["total_bytes"])
        self.assertIsNone(gpus[1]["free_bytes"])

    def test_empty(self):
        self.assertEqual(setup_hw.parse_nvidia_smi(""), [])

    def test_compute_capability_and_slot(self):
        gpus = setup_hw.parse_nvidia_smi(NVIDIA_SMI_CC)
        self.assertEqual([g["compute_cap"] for g in gpus], ["7.0", "8.6", "12.1", None])
        self.assertEqual(gpus[0]["name"], "Tesla V100-SXM2-16GB")
        self.assertEqual(gpus[0]["total_bytes"], 16384 * 2**20)
        self.assertEqual(gpus[0]["driver"], "580.178.04")
        self.assertEqual(gpus[0]["pci_bus_id"], "00000000:1A:00.0")
        self.assertIsNone(gpus[2]["total_bytes"])                 # GB10: unified memory
        self.assertEqual(setup_hw.compute_cap_sm(gpus[0]["compute_cap"]), 70)
        self.assertEqual(setup_hw.compute_cap_sm("12.0"), 120)
        self.assertEqual(setup_hw.compute_cap_sm("10.3"), 103)
        self.assertIsNone(setup_hw.compute_cap_sm("[N/A]"))
        self.assertIsNone(setup_hw.compute_cap_sm(None))

    def test_the_old_query_has_no_compute_capability(self):
        gpus = setup_hw.parse_nvidia_smi("0, Tesla V100-SXM2-16GB, 16384, 16144, 470.256.02, "
                                         "00000000:1A:00.0\n", setup_hw.NVIDIA_FIELDS_NO_CC)
        self.assertIsNone(gpus[0]["compute_cap"])
        self.assertEqual(gpus[0]["pci_bus_id"], "00000000:1A:00.0")

    def test_the_query_asks_for_the_compute_capability(self):
        self.assertEqual(setup_hw.NVIDIA_SMI_QUERY,
                         ["--query-gpu=index,name,memory.total,memory.free,driver_version,"
                          "compute_cap,pci.bus_id", "--format=csv,noheader,nounits"])
        self.assertNotIn("compute_cap", setup_hw.nvidia_smi_query(setup_hw.NVIDIA_FIELDS_NO_CC)[0])

    def run_detect(self, answers, exe, probe=None):
        """detect_nvidia with nvidia-smi answering `answers` by query, and the
        CUDA driver probe answering `probe`."""
        seen = []

        def run(cmd, timeout=10):
            seen.append(cmd)
            return answers.get(cmd[1], "")

        with mock.patch.object(setup_hw, "nvidia_smi_path", return_value=exe), \
             mock.patch.object(setup_hw, "_run", side_effect=run), \
             mock.patch.object(setup_hw, "_probe_cuda_child", return_value=probe) as child:
            gpus = setup_hw.detect_nvidia()
        return gpus, seen, child

    def test_detect_on_linux_windows_and_wsl(self):
        """The same query through every nvidia-smi the setup finds: the Linux
        one, Windows' nvidia-smi.exe and the one WSL maps in."""
        query = setup_hw.nvidia_smi_query()[0]
        for exe in ("/usr/bin/nvidia-smi", r"C:\Windows\System32\nvidia-smi.exe",
                    "/usr/lib/wsl/lib/nvidia-smi"):
            with self.subTest(exe=exe):
                gpus, seen, child = self.run_detect({query: NVIDIA_SMI_CC.splitlines()[0] + "\n"}, exe)
                self.assertEqual(seen[0][:2], [exe, query])
                self.assertEqual(len(seen), 1)
                self.assertEqual(gpus[0]["compute_cap"], "7.0")
                child.assert_not_called()                         # nvidia-smi answered

    def test_an_old_driver_is_asked_again_and_the_cuda_driver_fills_in(self):
        query = setup_hw.nvidia_smi_query()[0]
        old_query = setup_hw.nvidia_smi_query(setup_hw.NVIDIA_FIELDS_NO_CC)[0]
        # A driver that does not know the field: an error instead of rows (the
        # wording does not matter, no row parses).
        answers = {query: 'Field "compute_cap" is not a valid field to query.\n\n',
                   old_query: ("0, Tesla V100-SXM2-16GB, 16384, 16144, 470.256.02, 00000000:1A:00.0\n"
                               "1, Tesla P40, 22919, 22800, 470.256.02, 00000000:3B:00.0\n")}
        probe = {"driver": True, "devices": [
            {"name": "Tesla P40", "pci_bus_id": "0000:3B:00.0", "compute_cap": "6.1"},
            {"name": "Tesla V100-SXM2-16GB", "pci_bus_id": "0000:1a:00.0", "compute_cap": "7.0"}]}
        gpus, seen, child = self.run_detect(answers, "/usr/bin/nvidia-smi", probe)
        self.assertEqual([cmd[1] for cmd in seen], [query, old_query])
        child.assert_called_once()
        self.assertEqual([(g["name"], g["compute_cap"]) for g in gpus],
                         [("Tesla V100-SXM2-16GB", "7.0"), ("Tesla P40", "6.1")])   # matched by slot
        # No driver to ask: unknown, not a guess.
        gpus, _seen, _child = self.run_detect(answers, "/usr/bin/nvidia-smi", None)
        self.assertEqual([g["compute_cap"] for g in gpus], [None, None])

    def test_pci_slots_in_both_spellings(self):
        self.assertEqual(setup_hw.pci_bus_key("00000000:1A:00.0"), setup_hw.pci_bus_key("0000:1a:00.0"))
        self.assertNotEqual(setup_hw.pci_bus_key("00000000:1A:00.0"), setup_hw.pci_bus_key("0000:1b:00.0"))
        self.assertIsNone(setup_hw.pci_bus_key("[N/A]"))

    def test_the_cuda_driver_probe(self):
        """_cuda_probe_inprocess against a stand-in for libcuda/nvcuda.dll: the
        attribute numbers are the compute capability's (75, 76 in cuda.h)."""
        cards = [("Tesla V100-SXM2-16GB", "0000:1A:00.0", 7, 0), ("NVIDIA GeForce RTX 3090", "0000:68:00.0", 8, 6)]

        class Driver:
            def cuInit(self, flags):
                return 0

            def cuDeviceGetCount(self, count):
                count._obj.value = len(cards)
                return 0

            def cuDeviceGet(self, device, ordinal):
                device._obj.value = ordinal
                return 0

            def cuDeviceGetAttribute(self, value, attribute, device):
                value._obj.value = {75: cards[device.value][2], 76: cards[device.value][3]}[attribute]
                return 0

            def cuDeviceGetPCIBusId(self, buffer, size, device):
                buffer.value = cards[device.value][1].encode()
                return 0

            def cuDeviceGetName(self, buffer, size, device):
                buffer.value = cards[device.value][0].encode()
                return 0

        with mock.patch.object(setup_hw, "_cuda_driver_library", return_value=Driver()):
            probe = setup_hw._cuda_probe_inprocess()
        self.assertEqual(probe["devices"], [
            {"name": "Tesla V100-SXM2-16GB", "pci_bus_id": "0000:1A:00.0", "compute_cap": "7.0"},
            {"name": "NVIDIA GeForce RTX 3090", "pci_bus_id": "0000:68:00.0", "compute_cap": "8.6"}])
        with mock.patch.object(setup_hw, "_cuda_driver_library", return_value=None):
            self.assertEqual(setup_hw._cuda_probe_inprocess(), {"driver": False, "devices": []})

    def test_the_child_probe_reads_the_last_json_line(self):
        done = mock.Mock(stdout='driver noise\n{"driver": true, "devices": [{"compute_cap": "7.0"}]}\n')
        with mock.patch.object(setup_hw.subprocess, "run", return_value=done) as run:
            self.assertEqual(setup_hw._probe_cuda_child()["devices"], [{"compute_cap": "7.0"}])
        self.assertEqual(run.call_args[0][0][-1], "--cuda-probe")
        with mock.patch.object(setup_hw.subprocess, "run", side_effect=OSError("no python")):
            self.assertIsNone(setup_hw._probe_cuda_child())


class VulkanParsing(unittest.TestCase):
    def test_vulkaninfo_summary(self):
        devices = setup_hw.parse_vulkaninfo_summary(VULKANINFO_SUMMARY)
        self.assertEqual([d["name"] for d in devices],
                         ["AMD Radeon 780M (RADV PHOENIX)", "llvmpipe (LLVM 20.1.2, 256 bits)"])
        self.assertEqual(devices[0]["type"], "integrated")
        self.assertEqual(devices[0]["vendor"], "AMD")
        self.assertEqual(devices[1]["type"], "cpu")
        self.assertEqual(devices[0]["api_version"], "1.3.289")
        self.assertIsNone(devices[0]["device_local_bytes"])   # the summary has no memory figures

    def test_vulkaninfo_on_windows(self):
        devices = setup_hw.parse_vulkaninfo_summary(VULKANINFO_WINDOWS)
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["name"], "Intel(R) Iris(R) Xe Graphics")
        self.assertEqual(devices[0]["vendor"], "Intel")
        self.assertTrue(setup_hw.vulkan_usable(devices[0]))

    def test_memory_record_with_budget(self):
        memory = memory_properties([(16 * 2**30, True), (32 * 2**30, False)])
        budget = budget_struct([12 * 2**30, 30 * 2**30], [1 * 2**30, 2 * 2**30])
        record = setup_hw._vk_device_record("Radeon RX 7800 XT", 2, 0x1002, 0x747e,
                                            (1 << 22) | (3 << 12), 1, memory, budget, True)
        self.assertEqual(record["type"], "discrete")
        self.assertEqual(record["vendor"], "AMD")
        self.assertEqual(record["device_local_bytes"], 16 * 2**30)
        self.assertEqual(record["budget_bytes"], 12 * 2**30)    # device-local heaps only
        self.assertTrue(record["memory_budget_ext"])
        self.assertEqual(record["api_version"], "1.3.0")

    def test_memory_record_without_budget(self):
        memory = memory_properties([(17 * 10**9, True)])
        record = setup_hw._vk_device_record("Microsoft Direct3D12 (Intel(R) Iris(R) Xe Graphics)",
                                            1, 0x8086, 0xa7a1, (1 << 22) | (2 << 12), 1,
                                            memory, None, False)
        self.assertEqual(record["type"], "integrated")
        self.assertEqual(record["vendor"], "Intel")
        self.assertIsNone(record["budget_bytes"])
        self.assertEqual(record["device_local_bytes"], 17 * 10**9)

    def test_usable_and_best(self):
        cpu = device("llvmpipe", "cpu")
        igpu = device("Iris Xe", "integrated", "1.2.318")
        dgpu = device("RTX 4070", "discrete")
        old = device("Old card", "discrete", "1.1.0")
        self.assertFalse(setup_hw.vulkan_usable(cpu))
        self.assertTrue(setup_hw.vulkan_usable(igpu))
        self.assertFalse(setup_hw.vulkan_usable(old))            # below the engine's Vulkan 1.2
        self.assertIs(setup_hw.best_vulkan_device([cpu, igpu, dgpu]), dgpu)
        self.assertIs(setup_hw.best_vulkan_device([cpu, igpu]), igpu)
        summary = setup_hw.gpu_summary({"devices": [cpu], "icd": None}, [])
        self.assertIsNone(summary["vulkan"])
        self.assertFalse(summary["has_gpu"])

    def test_user_driver_manifests(self):
        with tempfile.TemporaryDirectory() as home:
            manifest = Path(home, "mesa-dzn", "install", "share", "vulkan", "icd.d",
                            "dzn_icd.x86_64.json")
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            local = Path(home, ".local", "share", "vulkan", "icd.d", "x.json")
            local.parent.mkdir(parents=True)
            local.write_text("{}")
            found = setup_hw.find_user_icds(home)
            self.assertIn(str(manifest), found)
            self.assertIn(str(local), found)

    def test_wsl_falls_back_to_a_user_driver(self):
        """The system drivers see only the CPU rasterizer; the Dozen manifest
        under $HOME sees the GPU, and its path is kept for the run config."""
        cpu_only = {"loader": True, "devices": [device("llvmpipe", "cpu")]}
        dozen = {"loader": True, "devices": [device("Microsoft Direct3D12 (Iris Xe)",
                                                    "integrated", "1.2.318")]}

        def probe(extra_env=None, timeout=25):
            return dozen if extra_env else cpu_only

        env = {k: v for k, v in os.environ.items()
               if k not in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES")}
        with mock.patch.object(setup_hw, "_probe_vulkan_child", side_effect=probe), \
             mock.patch.object(setup_hw, "find_user_icds", return_value=["/home/u/dzn.json"]), \
             mock.patch.object(setup_hw, "host_os", return_value="linux"), \
             mock.patch.dict(os.environ, env, clear=True):
            result = setup_hw.detect_vulkan(wsl=True)
        self.assertEqual(result["icd"], "/home/u/dzn.json")
        self.assertEqual(result["devices"][0]["type"], "integrated")

    def test_no_loader_uses_vulkaninfo(self):
        with mock.patch.object(setup_hw, "_probe_vulkan_child", return_value={"loader": False}), \
             mock.patch.object(setup_hw.shutil, "which", return_value="/usr/bin/vulkaninfo"), \
             mock.patch.object(setup_hw, "_run", return_value=VULKANINFO_SUMMARY), \
             mock.patch.object(setup_hw, "find_user_icds", return_value=[]), \
             mock.patch.object(setup_hw, "host_os", return_value="linux"):
            result = setup_hw.detect_vulkan(wsl=False)
        self.assertEqual(result["source"], "vulkaninfo")
        self.assertEqual(len(result["devices"]), 2)


class Report(unittest.TestCase):
    def test_format_report_names_the_pieces(self):
        report = {
            "os": {"platform": "linux", "pretty_name": "Ubuntu 24.04", "wsl": True},
            "cpu": {"name": "Test CPU", "features": ["avx2"], "physical_cores": 6,
                    "logical_cores": 12},
            "memory": {"total": 27 * 10**9, "available": 24 * 10**9},
            "disk": {"path": "/home/u/colibri-models", "free_bytes": 800 * 10**9, "warnings": []},
            "vulkan": {"devices": [], "icd": None},
            "nvidia": [{"name": "NVIDIA RTX 4070", "total_bytes": 12 * 2**30, "driver": "560"}],
            "windows_video": [],
            "gpu": {"vulkan": device("Iris Xe", "integrated", "1.2.318"),
                    "vulkan_icd": "/home/u/dzn.json",
                    "nvidia": [{"name": "NVIDIA RTX 4070", "total_bytes": 12 * 2**30,
                                "driver": "560"}], "has_gpu": True},
        }
        text = setup_hw.format_report(report)
        for piece in ("Test CPU", "6 cores (12 threads)", "AVX2", "27.0 GB", "800 GB free",
                      "Iris Xe via Vulkan", "integrated, shares RAM", "NVIDIA RTX 4070", "WSL2",
                      "/home/u/dzn.json"):
            self.assertIn(piece, text)

    def test_the_report_names_the_compute_capability(self):
        card = setup_hw.parse_nvidia_smi(NVIDIA_SMI_CC)[0]
        report = {"os": {"platform": "linux", "pretty_name": "Ubuntu 26.04.1 LTS"},
                  "cpu": {"name": "Test CPU", "features": [], "logical_cores": 16},
                  "memory": {"total": 31.6e9, "available": 24.1e9}, "disk": None,
                  "vulkan": {"devices": []}, "nvidia": [card], "windows_video": [],
                  "gpu": setup_hw.gpu_summary({"devices": []}, [card, dict(card, compute_cap=None)])}
        lines = setup_hw.format_report(report).splitlines()
        self.assertIn("  GPU     Tesla V100-SXM2-16GB (NVIDIA, 17.2 GB VRAM, compute 7.0, "
                      "driver 580.178.04)", lines)
        self.assertIn("  GPU     Tesla V100-SXM2-16GB (NVIDIA, 17.2 GB VRAM, driver 580.178.04)", lines)

    def test_detect_without_gpu_probe_is_plain_data(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            report = setup_hw.detect(tmp, probe_gpu=False)
        json.dumps(report)                      # serialisable for the MCP server
        self.assertIn("memory", report)
        self.assertEqual(report["disk"]["path"], os.path.abspath(tmp))


if __name__ == "__main__":
    unittest.main()
