#!/usr/bin/env python3
"""The one-step setup: `coli setup`, `coli start`, `coli status`, `coli logs`.

    detect -> recommend -> build or fetch the engine -> download -> configure -> start

Each step is idempotent, so the answer to "it stopped half way" is always "run
it again": a finished build is not rebuilt, a finished file is not fetched
again, a partial file resumes, and once everything is there a rerun goes
straight to starting the server.

What the setup writes, all in one folder (setup_home(), overridable with
COLI_SETUP_HOME):

    setup.json          the run configuration: model, engine, backend, the
                        environment and the launcher arguments `coli start` uses
    install-state.json  the current phase and download progress (read by
                        `coli status` and the MCP server while a setup runs)
    logs/serve.log      the server's output when started in the background
    logs/install.log    the output of a background install (MCP)
    runtime/            a prebuilt release, when there is no compiler

It installs nothing system-wide. When a package is missing it prints the exact
command for this system and lets the user run it.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.realpath(__file__))
#: This checkout's own launcher. HERE is where engines are looked for and built
#: (tests point it elsewhere); `coli stop` always comes from the real one.
COLI_SCRIPT = os.path.join(HERE, "coli")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
import setup_hw  # noqa: E402
from family_registry import family_by_id  # noqa: E402

GB = 1_000_000_000
CONFIG_VERSION = 1
DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"
RELEASES_API = "https://api.github.com/repos/JustVugg/colibri/releases"


def host_os():
    """The OS whose conventions decide what to build, fetch and call things:
    engine file names, the release asset, the package commands, the make
    invocation. One function, so a test can model Windows on Linux and the
    reverse; everything that calls an OS API checks sys.platform itself."""
    return sys.platform


def host_machine():
    return setup_hw.platform.machine().lower()


def exe_suffix():
    return ".exe" if host_os() == "win32" else ""


class SetupError(RuntimeError):
    """A stop with a reason the user can act on (printed without a traceback)."""


# ---------------------------------------------------------------- where things live


def setup_home():
    explicit = os.environ.get("COLI_SETUP_HOME")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        return os.path.join(base, "colibri")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/colibri")
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "colibri")


def config_path():
    return os.path.join(setup_home(), "setup.json")


def state_path():
    return os.path.join(setup_home(), "install-state.json")


def log_path(name):
    return os.path.join(setup_home(), "logs", f"{name}.log")


def default_models_root():
    return os.path.join(os.path.expanduser("~"), "colibri-models")


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def load_config():
    cfg = _read_json(config_path())
    if cfg and cfg.get("version") == CONFIG_VERSION:
        return cfg
    return None


def save_config(cfg):
    cfg = dict(cfg, version=CONFIG_VERSION, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _write_json(config_path(), cfg)
    return cfg


def write_state(phase, **fields):
    data = {"phase": phase, "pid": os.getpid(), "updated": time.time(), **fields}
    try:
        _write_json(state_path(), data)
    except OSError:
        pass


def read_state():
    return _read_json(state_path())


# ---------------------------------------------------------------- toolchain and package hints


def find_msys2():
    candidates = [os.environ.get("MSYS2_ROOT"), r"C:\msys64",
                  os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "msys64")]
    for root in candidates:
        if root and os.path.isfile(os.path.join(root, "usr", "bin", "bash.exe")):
            return root
    return None


def _nvcc():
    found = shutil.which("nvcc")
    if found:
        return found
    for root in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        if root and os.path.isfile(os.path.join(root, "bin", "nvcc")):
            return os.path.join(root, "bin", "nvcc")
    return None


#: What each CUDA release compiles for, for an nvcc that cannot answer
#: --list-gpu-arch: (first release, oldest compute_XX, newest compute_XX).
#: From 11.4 on, the bounds are the measured `nvcc --list-gpu-arch` of NVIDIA's
#: own nvcc (the redistributable archives for 11.4.152 to 12.9.86, the
#: nvidia-cuda-nvcc wheels for 13.0.88 and 13.4.92; tests/test_setup_flow.py
#: keeps the outputs); the earlier rows are from the toolkit release notes
#: (9.0 added Volta and dropped Fermi, 10.0 Turing, 11.0 Ampere and dropped
#: sm_30/sm_32, 11.1 sm_86). The table is also what says when a card was
#: dropped and which toolkit still builds for it.
CUDA_RELEASES = (
    ((9, 0), 30, 70),
    ((10, 0), 30, 75),
    ((11, 0), 35, 80),
    ((11, 1), 35, 86),
    ((11, 4), 35, 87),
    ((11, 8), 35, 90),
    ((12, 0), 50, 90),
    ((12, 8), 50, 120),
    ((12, 9), 50, 121),
    ((13, 0), 75, 121),
)
#: nvcc takes -arch=native from 11.6 on (measured: 11.5.119 answers "Value
#: 'native' is not defined for option 'gpu-architecture'", 11.6.55 takes it).
#: Below that the build names the card's architecture itself.
CUDA_NATIVE_SINCE = (11, 6)
CUDA_ARCHIVE_URL = "https://developer.nvidia.com/cuda-toolkit-archive"


def parse_nvcc_version(text):
    """(13, 0) from `nvcc --version` ("Cuda compilation tools, release 13.0,
    V13.0.88"); None when there is no release line."""
    match = re.search(r"\brelease (\d+)\.(\d+)", text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_nvcc_arch_list(text):
    """The compute_XX numbers of `nvcc --list-gpu-arch`, sorted ([75, 80, ...]).
    None when it lists none: an nvcc without the option prints an error."""
    found = {int(n) for n in re.findall(r"^\s*compute_(\d+)[a-z]?\s*$", text or "", re.M)}
    return sorted(found) or None


def cuda_toolkit(nvcc):
    """What this nvcc is and builds for: {"cuda_version": "13.0" or None,
    "cuda_archs": [75, 80, ...] or None (then the release table answers)}."""
    version = parse_nvcc_version(setup_hw._run([nvcc, "--version"], timeout=30))
    archs = parse_nvcc_arch_list(setup_hw._run([nvcc, "--list-gpu-arch"], timeout=30))
    return {"cuda_version": f"{version[0]}.{version[1]}" if version else None,
            "cuda_archs": archs}


def _version_tuple(text):
    match = re.match(r"^\s*(\d+)\.(\d+)", str(text or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _release_bounds(version):
    bounds = None
    for first, low, high in CUDA_RELEASES:
        if version >= first:
            bounds = (low, high)
    return bounds


def toolkit_builds(tc, sm):
    """Whether this machine's CUDA toolkit compiles for compute_<sm>: True or
    False, or None when nothing says (the card or the toolkit is unknown)."""
    if sm is None:
        return None
    if tc.get("cuda_archs"):
        return sm in tc["cuda_archs"]
    version = _version_tuple(tc.get("cuda_version"))
    bounds = _release_bounds(version) if version else None
    if bounds is None:
        return None
    return bounds[0] <= sm <= bounds[1]


def _release_name(version):
    return str(version[0]) if version[1] == 0 else f"{version[0]}.{version[1]}"


def cuda_release_advice(sm, version=None):
    """For a card the toolkit here cannot build for: (what happened, what to
    install). ("dropped in CUDA 13", "install a CUDA 12.x toolkit") for a card
    older than the toolkit, ("needs CUDA 12.8 or newer", "install CUDA 12.8 or
    newer") for one newer than it; from CUDA_RELEASES."""
    rows = [index for index, (_first, low, high) in enumerate(CUDA_RELEASES) if low <= sm <= high]
    if not rows:
        return None, f"install a CUDA toolkit whose `nvcc --list-gpu-arch` lists compute_{sm}"
    first = CUDA_RELEASES[rows[0]][0]
    if version is not None and version < first:
        return (f"needs CUDA {first[0]}.{first[1]} or newer",
                f"install CUDA {first[0]}.{first[1]} or newer")
    if rows[-1] + 1 < len(CUDA_RELEASES):
        dropped = CUDA_RELEASES[rows[-1] + 1][0]
        return (f"dropped in CUDA {_release_name(dropped)}",
                f"install a CUDA {CUDA_RELEASES[rows[-1]][0][0]}.x toolkit")
    return None, f"install a CUDA toolkit whose `nvcc --list-gpu-arch` lists compute_{sm}"


def toolchain(here=None):
    """What can be built here, and with what."""
    here = here or HERE
    tc = {"source_checkout": os.path.isfile(os.path.join(here, "Makefile")),
          "make": shutil.which("make"),
          "cc": shutil.which("gcc") or shutil.which("cc") or shutil.which("clang"),
          "glslc": shutil.which("glslc"), "vulkan_headers": False, "nvcc": _nvcc(),
          "msys2": None, "npm": shutil.which("npm")}
    if sys.platform == "win32":
        tc["nvcc"] = None   # the Windows CUDA path is a separate MSVC build (docs/windows.md)
        root = None if (tc["make"] and tc["cc"]) else find_msys2()
        if root:
            ucrt = os.path.join(root, "ucrt64")
            tc["msys2"] = root
            tc["make"] = os.path.join(root, "usr", "bin", "make.exe") \
                if os.path.isfile(os.path.join(root, "usr", "bin", "make.exe")) else None
            tc["cc"] = os.path.join(ucrt, "bin", "gcc.exe") \
                if os.path.isfile(os.path.join(ucrt, "bin", "gcc.exe")) else None
            tc["glslc"] = os.path.join(ucrt, "bin", "glslc.exe") \
                if os.path.isfile(os.path.join(ucrt, "bin", "glslc.exe")) else None
            tc["vulkan_headers"] = (os.path.isfile(os.path.join(ucrt, "include", "vulkan", "vulkan.h"))
                                    and os.path.isfile(os.path.join(ucrt, "lib", "libvulkan-1.dll.a")))
        elif tc["cc"]:
            prefix = os.path.dirname(os.path.dirname(tc["cc"]))
            tc["vulkan_headers"] = os.path.isfile(os.path.join(prefix, "include", "vulkan", "vulkan.h"))
    elif sys.platform == "darwin":
        tc["vulkan_headers"] = False   # macOS: the GPU path is Metal, not offered here
    else:
        tc["vulkan_headers"] = any(os.path.isfile(os.path.join(p, "vulkan", "vulkan.h"))
                                   for p in ("/usr/include", "/usr/local/include"))
    tc["can_build"] = bool(tc["source_checkout"] and tc["make"] and tc["cc"])
    tc["can_build_vulkan"] = bool(tc["can_build"] and tc["glslc"] and tc["vulkan_headers"])
    tc["can_build_cuda"] = bool(tc["can_build"] and tc["nvcc"] and sys.platform.startswith("linux"))
    # A toolkit is not enough: CUDA 13 cannot build for a V100 (#1852).
    # choose_backend asks toolkit_builds() about each card.
    tc["cuda_version"], tc["cuda_archs"] = None, None
    if tc["nvcc"]:
        tc.update(cuda_toolkit(tc["nvcc"]))
    return tc


def distro_kind(os_info):
    os_info = {k.lower(): v for k, v in (os_info or {}).items()}
    ids = " ".join([os_info.get("id", ""), os_info.get("id_like", "")]).lower()
    for kind, needles in (("debian", ("debian", "ubuntu", "mint", "pop")),
                          ("fedora", ("fedora", "rhel", "centos", "rocky", "alma")),
                          ("arch", ("arch", "manjaro", "endeavouros")),
                          ("suse", ("suse", "opensuse"))):
        if any(n in ids.split() or n in ids for n in needles):
            return kind
    return "other"


#: Per system: the install verb and the packages for each need. One command
#: covers several needs ("build" and "vulkan" together), so the user runs one line.
PACKAGES = {
    "debian": ("sudo apt install", {"build": ["build-essential"],
                                    "vulkan": ["libvulkan-dev", "glslc", "mesa-vulkan-drivers"],
                                    "cuda": ["nvidia-cuda-toolkit"], "python": ["python3"]}),
    "fedora": ("sudo dnf install", {"build": ["gcc", "make", "libgomp"],
                                    "vulkan": ["vulkan-loader-devel", "glslc", "mesa-vulkan-drivers"],
                                    "python": ["python3"]}),
    "arch": ("sudo pacman -S --needed", {"build": ["base-devel"],
                                         "vulkan": ["vulkan-icd-loader", "vulkan-headers", "shaderc"],
                                         "cuda": ["cuda"], "python": ["python"]}),
    "suse": ("sudo zypper install", {"build": ["gcc", "make"], "vulkan": ["vulkan-devel", "shaderc"],
                                     "python": ["python3"]}),
    "msys2": ("pacman -S --needed", {"build": ["mingw-w64-ucrt-x86_64-gcc", "make"],
                                     "vulkan": ["mingw-w64-ucrt-x86_64-vulkan-headers",
                                                "mingw-w64-ucrt-x86_64-vulkan-loader",
                                                "mingw-w64-ucrt-x86_64-shaderc"]}),
}
#: What is not one package: said in words, with where to get it.
TEXT_HINTS = {
    "cuda": "install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads",
    "cuda-win32": "the Windows CUDA build needs the CUDA Toolkit and MSVC: see docs/windows.md",
    "build-other": "install gcc (with OpenMP) and make from your distribution",
    "vulkan-other": ("install the Vulkan loader development files, glslc (shaderc) and the "
                     "Mesa Vulkan drivers from your distribution"),
    "vulkan-driver": "plus your GPU's Vulkan driver if it is not installed yet",
    "build-darwin": "xcode-select --install && brew install libomp",
    "vulkan-darwin": "not offered on macOS (the Apple GPU path is Metal, see docs/metal.md)",
    "python": "install Python 3.10 or newer from https://www.python.org/downloads/",
}


def package_hint(kinds, os_info=None, platform_name=None):
    """One line the user can run to get `kinds` (build, vulkan, cuda, python)."""
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    platform_name = platform_name or host_os()
    if platform_name == "darwin":
        return " ; ".join(TEXT_HINTS.get(f"{k}-darwin", TEXT_HINTS.get(k, k)) for k in kinds)
    if platform_name == "win32":
        verb, table = PACKAGES["msys2"]
        pkgs = [p for k in kinds for p in table.get(k, [])]
        words = [TEXT_HINTS["cuda-win32"] for k in kinds if k == "cuda"]
        if pkgs:
            words.insert(0, "install MSYS2 from https://www.msys2.org/ (if it is not there yet), "
                            f"then in its UCRT64 shell run: {verb} {' '.join(pkgs)}")
        return " ; ".join(words)
    kind = distro_kind(os_info if os_info is not None else setup_hw.os_release())
    if kind == "other":
        return " ; ".join(TEXT_HINTS.get(f"{k}-other", TEXT_HINTS.get(k, k)) for k in kinds)
    verb, table = PACKAGES[kind]
    pkgs = [p for k in kinds for p in table.get(k, [])]
    words = [f"{verb} {' '.join(pkgs)}"] if pkgs else []
    words += [TEXT_HINTS[k] for k in kinds if k not in table and k in TEXT_HINTS]
    if "vulkan" in kinds and kind in ("arch", "suse"):
        words.append(TEXT_HINTS["vulkan-driver"])
    return " ; ".join(words)


# ---------------------------------------------------------------- backend

#: The engines whose Vulkan path was measured faster than their CPU run on an
#: integrated GPU (a Radeon 780M, docs/vulkan.md "Measured on a Radeon 780M" and
#: "The chain on a Radeon 780M"): Qwen3.6 writes at 6.0 tok/s on the CPU and 9.9
#: with the GPU, Qwen3.8 Flash Next at 3.5 and 3.8. An integrated GPU has no memory
#: of its own: it reads the same RAM as the CPU, through the same bandwidth, and on
#: OLMoE that made it slower to write than the CPU alone (23 against 17 tok/s). So
#: on an integrated GPU the other engines stay on the CPU unless asked
#: (--backend vulkan). A discrete GPU, with its own VRAM, runs Vulkan for every one.
VULKAN_IGPU_MEASURED = frozenset({"qwen36", "qwen38"})

#: The engines with a CUDA path on Linux, and what their CUDA build needs, read
#: from the code (tests/test_setup_flow.py, CudaEngineTable, checks each entry
#: against the Makefiles and the sources):
#:
#:   backend_cuda.o (qwen36, qwen38, colibri, kimi_k3): nvcc -std=c++17, no
#:     compute floor in the code. The tensor-core kernels are guarded
#:     (__CUDA_ARCH__ >= 700 on the device, compute_major >= 7 where they are
#:     dispatched) and fall back below that; the engine ran on four sm_50 Tesla
#:     M10 (#1652, docs/qwen36-cuda-tier.md).
#:   backend_cuda_ink.o (inkling): plain bf16 GEMV kernels, no floor either.
#:   backend_cuda_dsv4.o (deepseek_v4, Makefile.deepseek-v4): nvcc -std=c++20,
#:     which nvcc takes from 12.0 on (measured: 11.8.89 answers "Value 'c++20'
#:     is not defined for option 'std'"); NO_TC=1 below 12.8, whose cuBLASLt
#:     block-scaling API the opt-in tensor-core path calls (docs/deepseek-v4.md,
#:     "Linux CUDA tier"); and dsv4_cuda_backend_arch_ok turns the tier on from
#:     compute 6.0, so an older card would sit idle while the CPU did the work.
#:
#: glm53 links no CUDA object (its accelerator is Metal) and the remaining
#: engines are CPU-only: none of them is here, so none of them is built with CUDA.
CUDA_ENGINES = {
    "qwen36": {"object": "backend_cuda.o"},
    "qwen38": {"object": "backend_cuda.o"},
    "glm": {"object": "backend_cuda.o"},
    "kimi": {"object": "backend_cuda.o"},
    "inkling": {"object": "backend_cuda_ink.o"},
    "deepseek_v4": {"object": "backend_cuda_dsv4.o", "min_compute": 60,
                    "min_toolkit": (12, 0), "no_tc_below": (12, 8)},
}

#: How a backend is named in a sentence: "the CUDA build", "using the CPU".
BACKEND_NAME = {"cuda": "CUDA", "vulkan": "Vulkan", "cpu": "CPU"}
USING = {"cuda": "CUDA", "vulkan": "Vulkan", "cpu": "the CPU"}


def cuda_problem(cards, family, tc):
    """Why this engine's CUDA build cannot serve these NVIDIA cards, as
    {"why": ..., "fix": ...}, or None when it can.

    Unknown is not a no: a card whose compute capability nothing reported, or
    a toolkit that did not say what it builds, passes, and a build that then
    fails falls back to the next backend (resolve_with_fallback). `fix` is
    what to install; None when no toolkit would help (the card is older than
    the engine's CUDA code runs on). With -arch=native nvcc builds for every
    card it sees and stops at the first it cannot, so every card is checked."""
    engine = CUDA_ENGINES.get(family.id, {})
    version = _version_tuple(tc.get("cuda_version"))
    floor = engine.get("min_compute")
    for card in cards:
        sm = setup_hw.compute_cap_sm(card.get("compute_cap"))
        if floor and sm is not None and sm < floor:
            return {"why": (f"{family.display_name}'s CUDA tier runs on compute "
                            f"{floor // 10}.{floor % 10} and newer, and the {card['name']} "
                            f"is compute {card['compute_cap']}"),
                    "fix": None}
    need = engine.get("min_toolkit")
    if need and version and version < need:
        return {"why": (f"{family.display_name}'s CUDA code needs CUDA {need[0]}.{need[1]} "
                        f"or newer (it is C++20), and the toolkit here is CUDA {tc['cuda_version']}"),
                "fix": f"install CUDA {need[0]}.{need[1]} or newer"}
    toolkit = f"CUDA {tc['cuda_version']}" if tc.get("cuda_version") else "CUDA"
    for card in cards:
        sm = setup_hw.compute_cap_sm(card.get("compute_cap"))
        if toolkit_builds(tc, sm) is False:
            note, fix = cuda_release_advice(sm, version)
            detail = f"compute {card['compute_cap']}" + (f", {note}" if note else "")
            return {"why": f"the {toolkit} toolkit cannot build for the {card['name']} ({detail})",
                    "fix": fix}
    return None


def cuda_make_args(family, tc, cards):
    """The make variables the CUDA build of this engine needs on this toolkit:
    the card's own architecture when nvcc is too old for -arch=native (and
    CUDA_ARCH is not set by hand), NO_TC=1 where the engine needs it."""
    args = []
    version = _version_tuple(tc.get("cuda_version"))
    if version and version < CUDA_NATIVE_SINCE and not os.environ.get("CUDA_ARCH") and cards:
        sm = setup_hw.compute_cap_sm(cards[0].get("compute_cap"))
        if sm:
            args.append(f"CUDA_ARCH=sm_{sm}")
    below = CUDA_ENGINES.get(family.id, {}).get("no_tc_below")
    if below and version and version < below:
        args.append("NO_TC=1")
    return args


def choose_backend(hw, family, tc, requested="auto", exclude=()):
    """Which build to run the model with, and why.

    CUDA first for an NVIDIA card when this engine has a CUDA path (CUDA_ENGINES)
    and the toolkit here builds for every card (cuda_problem); its VRAM expert
    tier is the measured fast path. Else Vulkan when a Vulkan GPU answered and
    the build has its headers and glslc; else the CPU. On an integrated GPU,
    Vulkan only for the engines measured faster there (VULKAN_IGPU_MEASURED)
    unless Vulkan was asked for by name. Every GPU path the machine has but
    cannot build yet comes back in `missing` with what would enable it.

    A toolkit that cannot build for the card is not a build to try: auto says
    why and takes the next backend, and an explicit --backend cuda stops here
    with the same reason instead of failing inside make. `exclude` names the
    backends whose build already failed (resolve_with_fallback)."""
    gpu = hw.get("gpu") or {}
    vk, nvidia = gpu.get("vulkan"), gpu.get("nvidia") or []
    decision = {"backend": "cpu", "reason": "", "missing": [], "gpu": None}
    if requested == "cpu":
        decision["reason"] = "CPU only, as requested"
        return decision
    cuda_engine = family.id in CUDA_ENGINES and host_os().startswith("linux")
    blocked = None
    if nvidia and requested in ("auto", "cuda") and "cuda" not in exclude:
        if cuda_engine and tc.get("can_build_cuda"):
            blocked = cuda_problem(nvidia, family, tc)
            if blocked is None:
                card = nvidia[0]
                detail = f" (compute {card['compute_cap']})" if card.get("compute_cap") else ""
                toolkit = f"CUDA {tc['cuda_version']}" if tc.get("cuda_version") else "CUDA"
                decision.update(backend="cuda", gpu=card["name"],
                                reason=f"NVIDIA {card['name']}{detail} with the {toolkit} toolkit",
                                make_args=cuda_make_args(family, tc, nvidia))
                return decision
            if requested == "cuda":
                instead = choose_backend(hw, family, tc, "auto", exclude=tuple(exclude) + ("cuda",))
                raise SetupError(f"{blocked['why']}: "
                                 + (f"{blocked['fix']} for the CUDA path, or " if blocked["fix"] else "")
                                 + f"run with --backend auto to use {USING[instead['backend']]}")
            if blocked["fix"]:
                decision["missing"].append(("cuda", f"{blocked['fix']} ({CUDA_ARCHIVE_URL})"))
        elif cuda_engine:
            needs = ["cuda"] if tc.get("can_build") else ["build", "cuda"]
            decision["missing"].append(("cuda", _gpu_hint(needs, hw, tc)))
        elif requested == "cuda" and not cuda_engine:
            decision["missing"].append(("cuda", f"{family.display_name} has no CUDA path here; "
                                                "Vulkan or the CPU run it"))

    def because(using, otherwise):
        """The reason, led by why CUDA was passed over when it was."""
        if blocked is None:
            return otherwise
        text = f"{blocked['why']}: using {using}"
        if blocked["fix"]:
            text += f"; {blocked['fix']} for the CUDA path"
        return text

    if (vk and vk.get("type") == "integrated" and requested in ("auto", "cuda")
            and family.id not in VULKAN_IGPU_MEASURED):
        igpu = (f"{vk['name']} is an integrated GPU, which shares the CPU's RAM, "
                f"and {family.display_name} was not measured faster on one: "
                "the engine runs on the CPU (--backend vulkan uses the GPU anyway)")
        decision["reason"] = igpu if blocked is None else f"{because('the CPU', '')}. {igpu}"
        return decision
    if vk and requested in ("auto", "vulkan", "cuda") and "vulkan" not in exclude:
        if tc.get("can_build_vulkan"):
            kind = vk.get("type")
            decision.update(backend="vulkan", gpu=vk["name"],
                            reason=because("Vulkan", f"{vk['name']}, {kind} GPU"))
            return decision
        needs = ["vulkan"] if tc.get("can_build") else ["build", "vulkan"]
        decision["missing"].append(("vulkan", _gpu_hint(needs, hw, tc)))
    if not vk and not nvidia:
        decision["reason"] = "no GPU found: the engine runs on the CPU"
    elif requested == "vulkan" and not vk:
        decision["reason"] = "no Vulkan GPU answered: the engine runs on the CPU"
    else:
        decision["reason"] = because("the CPU",
                                     "the GPU build is not possible yet: the engine runs on the CPU")
    return decision


def _gpu_hint(needs, hw, tc):
    hint = package_hint(needs, hw.get("os"))
    if not tc.get("source_checkout"):
        # A release archive has the engines but no Makefile to rebuild them.
        hint += (" ; then run the setup from a source checkout "
                 "(git clone https://github.com/JustVugg/colibri)")
    return hint


def binary_links(path, needles):
    """Does the binary name one of these libraries? Cheap and portable: the
    dynamic section (ELF), import table (PE) or load commands (Mach-O) all carry
    the library name as plain bytes."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return False
    lowered = data.lower()
    return any(needle.lower() in lowered for needle in needles)


def binary_backend(path):
    if binary_links(path, (b"libcudart", b"coli_cuda.dll", b"libamdhip64")):
        return "cuda"
    if binary_links(path, (b"libvulkan.so", b"vulkan-1.dll", b"libvulkan.1.dylib")):
        return "vulkan"
    return "cpu"


def engine_path(directory, family):
    return os.path.join(directory, family.engine_artifact + exe_suffix())


# ---------------------------------------------------------------- building


def make_command(family, backend, tc, make_args=()):
    """The make invocation for this engine and backend (argv, cwd, env).
    `make_args` are the extra variables the decision asked for
    (cuda_make_args): they come last, and a CUDA_ARCH among them replaces
    the default one."""
    target = family.build_target
    args = [target, f"ARCH={os.environ.get('ARCH') or 'native'}"]
    if backend == "vulkan":
        args.append("VK=1")
    elif backend == "cuda":
        args.append("CUDA=1")
        if not any(arg.startswith("CUDA_ARCH=") for arg in make_args):
            args.append(f"CUDA_ARCH={os.environ.get('CUDA_ARCH') or 'native'}")
    args += list(make_args)
    if host_os() == "win32" and tc.get("msys2"):
        bash = os.path.join(tc["msys2"], "usr", "bin", "bash.exe")
        script = 'cd "$(cygpath -u "$1")" && shift && exec make "$@"'
        env = dict(os.environ, MSYSTEM="UCRT64", CHERE_INVOKING="1", MSYS2_PATH_TYPE="inherit")
        return [bash, "-lc", script, "colibri-build", HERE] + args, HERE, env
    return [tc.get("make") or "make", "-C", HERE] + args, HERE, dict(os.environ)


class BuildError(SetupError):
    """A build that failed: which backend, and where its log is."""

    def __init__(self, message, backend, log):
        super().__init__(message)
        self.backend = backend
        self.log = log


def build_log_name(backend):
    """One log per backend (logs/build-cuda.log, ...): when a failed CUDA build
    falls back to Vulkan, the CUDA log is still there to read."""
    return f"build-{backend}"


def build_engine(family, backend, tc, out=print, make_args=(), update=False):
    """make the engine; `update` for one already here (refresh_engine), where make
    redoes only what changed since it was built."""
    cmd, cwd, env = make_command(family, backend, tc, make_args)
    shown = [a for a in cmd if a == family.build_target or re.match(r"^[A-Z_]+=", a)]
    log_file = log_path(build_log_name(backend))
    if update:
        out(f"  updating: make {' '.join(shown)}  (only what changed since the last build; "
            f"log: {log_file})")
    else:
        out(f"  building: make {' '.join(shown)}  (a few minutes; log: {log_file})")
    write_state("build", engine=family.engine_artifact, backend=backend)
    with open(_ensure_log(build_log_name(backend)), "w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        tail = []
        for line in process.stdout:
            log.write(line)
            tail = (tail + [line.rstrip()])[-25:]
        process.wait()
    path = engine_path(HERE, family)
    if process.returncode != 0 or not os.path.exists(path):
        detail = "\n".join("    " + line for line in tail[-12:])
        raise BuildError(f"the {family.build_target} {BACKEND_NAME[backend]} build "
                         f"failed (full log: {log_file}):\n{detail}", backend, log_file)
    return path


def _ensure_log(name):
    path = log_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------- prebuilt releases


def release_asset_suffix(os_name=None, machine=None):
    os_name = os_name or host_os()
    machine = (machine or host_machine()).lower()
    if os_name.startswith("linux") and machine in ("x86_64", "amd64"):
        return "linux-x86_64.tar.gz"
    if os_name == "darwin" and machine in ("arm64", "aarch64"):
        return "macos-arm64.tar.gz"
    if os_name == "win32" and machine in ("amd64", "x86_64"):
        return "windows-x86_64.zip"
    return None


def _api_get(url, timeout=30):
    request = urllib.request.Request(url, headers={"User-Agent": "colibri-setup",
                                                   "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def find_release(version, api=None, getter=None):
    """The release matching this checkout's version, else the latest one."""
    api = (api or os.environ.get("COLI_RELEASES_API") or RELEASES_API).rstrip("/")
    getter = getter or _api_get
    for url in (f"{api}/tags/v{version}", f"{api}/latest"):
        try:
            data = getter(url)
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("tag_name"):
            return {"tag": data["tag_name"],
                    "assets": {a["name"]: a["browser_download_url"] for a in data.get("assets", [])
                               if isinstance(a, dict) and a.get("name")}}
    return None


def parse_sha256sums(text):
    sums = {}
    for line in (text or "").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(\S+)$", line.strip())
        if match:
            sums[match.group(2)] = match.group(1).lower()
    return sums


def _safe_members(names):
    for name in names:
        parts = name.replace("\\", "/").split("/")
        if name.startswith(("/", "\\")) or ".." in parts or re.match(r"^[A-Za-z]:", name):
            raise SetupError(f"refusing an archive member outside the folder: {name!r}")


def extract_archive(archive, dest, only_prefix=None):
    """Unpack a release archive (.zip or .tar.gz), refusing any member that
    would land outside `dest`. `only_prefix` keeps one subtree (web/dist/)."""
    os.makedirs(dest, exist_ok=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if not only_prefix or n.startswith(only_prefix)]
            _safe_members(names)
            for name in names:
                zf.extract(name, dest)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if (not only_prefix or m.name.lstrip("./").startswith(only_prefix))
                       and (m.isfile() or m.isdir())]
            _safe_members([m.name for m in members])
            extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            for member in members:
                tf.extract(member, dest, **extra)
    if not only_prefix:
        for name in os.listdir(dest):
            path = os.path.join(dest, name)
            if os.path.isfile(path) and os.name == "posix" and "." not in name:
                os.chmod(path, 0o755)


def fetch_release_archive(version, out=print, getter=None):
    """Download (once) and unpack the release for this OS into runtime/.
    Returns (directory, tag)."""
    suffix = release_asset_suffix()
    if suffix is None:
        raise SetupError("no prebuilt release exists for this system; build from source")
    release = find_release(version, getter=getter)
    if release is None:
        raise SetupError("cannot reach the GitHub releases to fetch a prebuilt engine")
    tag = release["tag"]
    name = f"colibri-{tag}-{suffix}"
    url = release["assets"].get(name)
    if not url:
        raise SetupError(f"release {tag} has no {name}")
    runtime = os.path.join(setup_home(), "runtime")
    dest = os.path.join(runtime, f"colibri-{tag}")
    if os.path.isfile(os.path.join(dest, "coli")):
        return dest, tag
    sums = {}
    if release["assets"].get("SHA256SUMS.txt"):
        try:
            request = urllib.request.Request(release["assets"]["SHA256SUMS.txt"],
                                             headers={"User-Agent": "colibri-setup"})
            with urllib.request.urlopen(request, timeout=30) as response:
                sums = parse_sha256sums(response.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError):
            sums = {}
    os.makedirs(runtime, exist_ok=True)
    archive = os.path.join(runtime, name)
    out(f"  downloading the prebuilt engine {tag} ({name})")
    # The asset size is not in the listing we keep; HEAD-less: let the server tell.
    request = urllib.request.Request(url, headers={"User-Agent": "colibri-setup"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as error:
        raise SetupError(f"cannot download {name}: {error}")
    if sums.get(name):
        import hashlib
        if hashlib.sha256(data).hexdigest() != sums[name]:
            raise SetupError(f"{name}: checksum does not match SHA256SUMS.txt")
    with open(archive, "wb") as handle:
        handle.write(data)
    staging = dest + ".partial"
    shutil.rmtree(staging, ignore_errors=True)
    extract_archive(archive, staging)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(staging, dest)
    return dest, tag


def web_dist_dir(launcher_dir):
    for candidate in (os.path.join(launcher_dir, "web", "dist"),
                      os.path.join(os.path.dirname(launcher_dir), "web", "dist")):
        if os.path.isfile(os.path.join(candidate, "index.html")):
            return candidate
    return None


def ensure_web_dist(launcher_dir, version, out=print):
    """The dashboard is a build artifact: a source checkout has none until
    `npm run build`. Take it from the matching release archive instead (it is
    the same app, prebuilt), so the browser has something to open."""
    if web_dist_dir(launcher_dir):
        return True
    try:
        runtime, _tag = fetch_release_archive(version, out=out)
    except SetupError as error:
        out(f"  the web dashboard is not built and could not be fetched ({error});")
        out("  the API still works. To build it: cd web && npm ci && npm run build")
        return False
    source = os.path.join(runtime, "web", "dist")
    if not os.path.isfile(os.path.join(source, "index.html")):
        return False
    root = os.path.dirname(launcher_dir)
    target = os.path.join(root, "web", "dist") if os.path.isdir(os.path.join(root, "web")) \
        else os.path.join(launcher_dir, "web", "dist")
    shutil.copytree(source, target, dirs_exist_ok=True)
    out(f"  web dashboard: {target} (from the release archive)")
    return True


# ---------------------------------------------------------------- the engine


def current_version():
    try:
        from version import __version__
        return __version__
    except ImportError:
        return "0"


def resolve_engine(family, entry, decision, tc, out=print, allow_prebuilt=True):
    """Make sure an engine for this family exists. Returns
    {"launcher_dir", "engine", "backend", "source"}; may lower the backend to
    "cpu" when only a prebuilt (CPU) engine is possible."""
    local = engine_path(HERE, family)
    if os.path.exists(local):
        have = binary_backend(local)
        if decision["backend"] == "cpu" or have == decision["backend"]:
            make_args = decision.get("make_args", ()) if have == decision["backend"] else ()
            source = refresh_engine(family, have, tc, local, out=out, make_args=make_args)
            return {"launcher_dir": HERE, "engine": local, "backend": decision["backend"],
                    "source": source}
        if not tc.get("can_build"):
            out(f"  the {family.engine_artifact} engine here is a {have} build; "
                f"keeping it, since nothing here can rebuild it")
            return {"launcher_dir": HERE, "engine": local, "backend": have if have != "cuda" else "cpu",
                    "source": "present"}
    if tc.get("can_build"):
        path = build_engine(family, decision["backend"], tc, out=out,
                            make_args=decision.get("make_args", ()))
        return {"launcher_dir": HERE, "engine": path, "backend": decision["backend"], "source": "built"}
    if not allow_prebuilt:
        raise SetupError(f"no compiler here to build {family.engine_artifact}: "
                         f"{package_hint(['build'])}")
    version = current_version()
    if release_asset_suffix() is None:
        raise SetupError(f"no compiler here, and no prebuilt engine is published for "
                         f"{host_os()}/{host_machine()}: {package_hint(['build'])}")
    out("  no compiler found: using the prebuilt engine from the GitHub release")
    runtime, tag = fetch_release_archive(version, out=out)
    if entry is not None and setup_catalog.version_tuple(tag) < setup_catalog.version_tuple(entry.prebuilt_since):
        raise SetupError(f"the published release {tag} predates {entry.name}; build from source "
                         f"instead ({package_hint(['build'])}), or pick another model")
    engine = engine_path(runtime, family)
    if not os.path.exists(engine):
        raise SetupError(f"release {tag} has no {family.engine_artifact} engine for this system")
    if decision["backend"] != "cpu":
        out(f"  the prebuilt engine runs on the CPU; the {decision['backend']} build needs a "
            f"source checkout and a compiler: {package_hint(['build', decision['backend']])}")
    return {"launcher_dir": runtime, "engine": engine, "backend": "cpu", "source": f"release {tag}"}


# What the toolchain needs to build each kind of engine (toolchain()).
CAN_BUILD = {"cpu": "can_build", "vulkan": "can_build_vulkan", "cuda": "can_build_cuda"}


def refresh_engine(family, backend, tc, path, out=print, make_args=()):
    """An engine already here, brought up to date with the sources before it is
    used: make redoes what changed since it was built, and nothing when nothing
    did (about 0.02 s). Without this a checkout updated with `git pull` kept its
    old engine, since one with the right backend was found and used as it was,
    and the new code never ran (#1852). It is rebuilt as the kind it is (`backend`,
    from binary_backend), so a Vulkan engine used for the CPU stays a Vulkan one.
    With no toolchain for that kind, or when the rebuild fails, the engine that
    was here stays, the failure with a line. Returns "present" or "rebuilt"."""
    if not tc.get(CAN_BUILD.get(backend, "can_build")):
        return "present"
    before = os.stat(path).st_mtime_ns
    try:
        build_engine(family, backend, tc, out=out, make_args=make_args, update=True)
    except BuildError as error:
        if not os.path.exists(path):
            raise
        out(f"  could not update it (log: {error.log}); "
            f"keeping the {family.engine_artifact} engine that was here")
        return "present"
    if os.stat(path).st_mtime_ns == before:
        out("  up to date")
        return "present"
    out("  rebuilt: the sources changed since the last build")
    return "rebuilt"


def refresh_configured(cfg, out=print, tc=None):
    """The configured engine, brought up to date before a start (refresh_engine).
    A rerun of the setup, or `coli start`, used to start the engine as it was, so
    after a `git pull` the old code kept running (#1852). Only an engine built in
    this checkout: a prebuilt one from a release archive has no sources here. Not
    while a server runs: it keeps the engine it started with. Returns
    refresh_engine's answer, or None when there was nothing to check."""
    try:
        family = family_by_id(cfg.get("family"))
    except Exception:
        return None
    engine = cfg.get("engine") or ""
    if (not os.path.isfile(engine)
            or os.path.abspath(engine) != os.path.abspath(engine_path(HERE, family))):
        return None
    if server_status(cfg)["state"] != "stopped":
        return None
    tc = tc or toolchain()
    have = binary_backend(engine)
    # The CUDA variables the setup built it with: they come from the cards it kept.
    make_args = cuda_make_args(family, tc, cfg.get("nvidia") or []) if have == "cuda" else ()
    return refresh_engine(family, have, tc, engine, out=out, make_args=make_args)


def resolve_with_fallback(ui, family, entry, decision, tc, hw, requested, failed=()):
    """resolve_engine, and when a build fails the next backend that can work
    instead of a stop: CUDA, then Vulkan, then the CPU (choose_backend again,
    without the backends that failed). With someone at the terminal it asks
    first; with --yes it goes ahead. Either way it says which build failed and
    where its log is. A CPU build that fails still stops the setup.

    Returns (engine_info, decision, failed): the decision the engine was built
    for, and the backends whose build failed, which the run configuration
    keeps so that a resumed setup does not build them again."""
    failed = list(failed)
    while True:
        try:
            return resolve_engine(family, entry, decision, tc, out=ui.say), decision, failed
        except BuildError as error:
            if error.backend == "cpu":
                raise
            failed.append(error.backend)
            following = choose_backend(hw, family, tc, requested, exclude=tuple(failed))
            ui.say(f"  {error}")
            if ui.interactive and not ui.confirm(
                    f"  Build for {USING[following['backend']]} instead?", True):
                raise
            following["reason"] = (f"the {BACKEND_NAME[error.backend]} build failed "
                                   f"(full log: {error.log}): using {USING[following['backend']]}")
            decision = following
            ui.say(f"\nEngine: {family.engine_artifact} with {decision['backend'].upper()} "
                   f"({decision['reason']})")
            for kind, hint in decision["missing"]:
                ui.say(f"  to use the GPU through {kind.upper()}, first run:  {hint}")


# ---------------------------------------------------------------- the run configuration


def urls(host, port):
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    base = f"http://{shown}:{port}"
    return {"browser": f"{base}/", "openai_base_url": f"{base}/v1",
            "anthropic_base_url": base}


def run_environment(backend, engine, family, hw):
    """The environment `coli start` adds for the chosen backend."""
    env = {}
    if backend == "vulkan":
        env["COLI_VULKAN"] = "1"
        shaders = os.path.join(os.path.dirname(engine), "shaders")
        if os.path.isdir(shaders):
            env["COLI_VK_SHADERS"] = shaders
        icd = (hw.get("gpu") or {}).get("vulkan_icd")
        if icd:
            env["VK_ICD_FILENAMES"] = icd
            env["VK_DRIVER_FILES"] = icd
        if family.id == "glm":
            # docs/vulkan.md: the GLM engine's OpenMP spin starves the I/O pool
            # under Vulkan. With the tune off the launcher no longer sets the
            # physical-core team, so set it here (#718).
            env["COLI_NO_OMP_TUNE"] = "1"
            cores = (hw.get("cpu") or {}).get("physical_cores")
            if cores:
                env["OMP_NUM_THREADS"] = str(cores)
    return env


def launcher_args(backend, model_dir, host, port):
    args = ["web", "--model", model_dir, "--host", host, "--port", str(port)]
    if backend == "cuda":
        args += ["--gpu", "auto", "--auto-tier"]
    return args


def make_config(*, model_dir, family, entry, engine_info, hw, host, port, status="ready"):
    backend = engine_info["backend"]
    return {
        "status": status,
        "model": {"id": entry.id if entry else None,
                  "name": entry.name if entry else family.display_name,
                  "repo": entry.repo if entry else None,
                  "revision": entry.revision if entry else None},
        "model_dir": model_dir,
        "family": family.id,
        "launcher": launcher_in(engine_info["launcher_dir"]),
        "engine": engine_info["engine"],
        "engine_source": engine_info.get("source"),
        "backend": backend,
        "gpu": engine_info.get("gpu"),
        "env": run_environment(backend, engine_info["engine"], family, hw),
        "args": launcher_args(backend, model_dir, host, port),
        "host": host,
        "port": port,
        "urls": urls(host, port),
    }


def launcher_in(directory):
    """The `coli` that runs engines from `directory`. An installed layout keeps
    the engines in libexec/colibri and the launcher in bin/: then it is the
    launcher this process was started from."""
    candidate = os.path.join(directory, "coli")
    if os.path.isfile(candidate):
        return candidate
    started = os.path.realpath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if os.path.basename(started) == "coli" and os.path.isfile(started):
        return started
    return candidate


def equivalent_command(cfg):
    env = " ".join(f"{k}={_quote(v)}" for k, v in cfg.get("env", {}).items())
    cmd = " ".join(_quote(a) for a in ["coli"] + cfg.get("args", []))
    if host_os() == "win32":
        sets = "".join(f'set "{k}={v}" && ' for k, v in cfg.get("env", {}).items())
        return f"{sets}{cmd}"
    return f"{env} {cmd}".strip()


def _quote(value):
    value = str(value)
    return value if re.match(r"^[A-Za-z0-9_./:=,+-]+$", value) else json.dumps(value)


# ---------------------------------------------------------------- server control


def serve_pidfile(port):
    return os.path.join(tempfile.gettempdir(), f"coli-serve-{port}.pid")


def _pid_alive(pid):
    if not pid:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            k32.GetExitCodeProcess.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            k32.CloseHandle.restype = wintypes.BOOL
            handle = k32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            try:
                code = wintypes.DWORD()
                return bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
            finally:
                k32.CloseHandle(handle)
        except (OSError, AttributeError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _get_json(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as error:
        error.close()
        return error.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return None, None


def port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) != 0


def pick_port(host, preferred, tries=20):
    for port in range(preferred, preferred + tries):
        if port_free(host, port):
            return port
    raise SetupError(f"ports {preferred}-{preferred + tries - 1} are all in use; pass --port")


def last_turn_speed(profile):
    """tok/s of the last finished turn from /profile, prompt time included."""
    turns = (profile or {}).get("turns") or []
    if not turns:
        return None
    turn = turns[-1]
    wall, tokens = turn.get("wall_s") or 0, turn.get("completion_tokens") or 0
    return round(tokens / wall, 2) if wall > 0 and tokens else None


def server_status(cfg):
    port, host = int(cfg["port"]), cfg.get("host", DEFAULT_HOST)
    pid = None
    try:
        with open(serve_pidfile(port), encoding="utf-8") as handle:
            pid = int(handle.read().split()[0])
    except (OSError, ValueError, IndexError):
        pid = None
    alive = _pid_alive(pid)
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    code, health = _get_json(f"http://{shown}:{port}/health")
    if code == 200:
        state = "ready"
    elif alive:
        state = "loading"
    else:
        state = "stopped"
    out = {"state": state, "pid": pid if alive else None, "urls": urls(host, port),
           "model": (cfg.get("model") or {}).get("name"), "model_dir": cfg.get("model_dir"),
           "backend": cfg.get("backend"), "model_id": None, "tokens_per_second": None,
           "last_turn": None}
    if state == "ready":
        _, profile = _get_json(f"http://{shown}:{port}/profile")
        turns = (profile or {}).get("turns") or []
        if turns:
            out["last_turn"] = turns[-1]
            out["tokens_per_second"] = last_turn_speed(profile)
        if health and health.get("arch"):
            out["arch"] = health["arch"]
        _, models = _get_json(f"http://{shown}:{port}/v1/models")
        data = (models or {}).get("data") or []
        if data and isinstance(data[0], dict):
            out["model_id"] = data[0].get("id")
    return out


def start_server(cfg, *, background=False, open_browser=True, out=print, wait=5.0):
    """Start what the configuration describes. Foreground returns the server's
    exit code; background returns the status after `wait` seconds."""
    status = server_status(cfg)
    if status["state"] in ("ready", "loading"):
        out(f"  colibri is already running ({status['state']}) on port {cfg['port']}")
        return status if background else 0
    host = cfg.get("host", DEFAULT_HOST)
    if not port_free(host, int(cfg["port"])):
        new_port = pick_port(host, int(cfg["port"]) + 1)
        out(f"  port {cfg['port']} is taken by another program; using {new_port}")
        cfg = dict(cfg, port=new_port, urls=urls(host, new_port),
                   args=_replace_arg(cfg["args"], "--port", str(new_port)))
        save_config(cfg)
    launcher = cfg["launcher"]
    if not os.path.isfile(launcher):
        raise SetupError(f"the launcher {launcher} is gone; run `coli setup` again")
    cmd = [sys.executable, launcher] + list(cfg["args"])
    if not open_browser and "--no-browser" not in cmd:
        cmd.append("--no-browser")
    env = dict(cfg.get("env") or {})
    env.update(os.environ)          # what the user sets explicitly wins
    env["PYTHONUNBUFFERED"] = "1"
    if not background:
        print_urls(cfg, out)
        out("  stop: press Ctrl+C here (or close this window)\n")
        sys.stdout.flush()      # ours before the server's, when the output is a file
        process = subprocess.Popen(cmd, env=env)
        note_foreground_start(process.pid, cmd)
        try:
            return process.wait()
        except KeyboardInterrupt:
            # The server got the same Ctrl+C and drains its engine; give it the
            # time `coli stop` gives, rather than killing it mid-save.
            try:
                return process.wait(timeout=35)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                process.terminate()
                return 130
    log = _ensure_log("serve")
    handle = open(log, "a", encoding="utf-8", errors="replace")
    handle.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n")
    handle.flush()
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": handle, "stderr": subprocess.STDOUT, "env": env}
    if sys.platform == "win32":
        # A hidden console of its own (CREATE_NO_WINDOW), in a new process group:
        # it outlives this window, and the engine .exe it starts inherits the
        # hidden console instead of opening one (DETACHED_PROCESS would).
        kwargs["creationflags"] = 0x00000200 | 0x08000000
    else:
        kwargs["start_new_session"] = True
    process = spawn_detached(cmd, **kwargs)
    handle.close()
    deadline = time.time() + wait
    while time.time() < deadline and process.poll() is None:
        time.sleep(0.25)
        if server_status(cfg)["state"] == "ready":
            break
    if process.poll() is not None:
        raise SetupError(f"the server exited at once (code {process.returncode}); "
                         f"see {log}:\n" + tail_file(log, 15))
    status = server_status(cfg)
    status.update(state=status["state"] if status["state"] != "stopped" else "loading",
                  pid=process.pid, log=log)
    out(f"  started in the background (pid {process.pid}); "
        + ("ready. " if status["state"] == "ready" else "the model is loading (`coli status` "
           "says when it is ready). ") + f"Log: {log}")
    print_urls(cfg, out)
    return status


def note_foreground_start(pid, cmd):
    """A server in the foreground prints to its terminal, not to serve.log; say so
    in the log, so that `coli logs` (often run from another terminal) points there
    instead of showing an older background run's lines as if they were this one's
    (#1852). Best effort: a log that cannot be written changes nothing else."""
    try:
        with open(_ensure_log("serve"), "a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')} in the foreground "
                         f"(pid {pid}): {' '.join(cmd)}\n"
                         f"--- its output goes to the terminal that started it, not to this log; "
                         f"`coli start --background` writes it here\n")
    except OSError:
        pass


def spawn_detached(cmd, **kwargs):
    """Start a process that outlives this call, and reap it when it exits.

    Without the waiter a long-lived parent (the MCP server) keeps the exited
    child as a zombie: kill(pid, 0) then still succeeds, `coli stop` waits its
    whole drain window for a process that is already gone, and a finished
    install reads as still running."""
    process = subprocess.Popen(cmd, **kwargs)
    threading.Thread(target=process.wait, name=f"reap-{process.pid}", daemon=True).start()
    return process


def _replace_arg(args, flag, value):
    args = list(args)
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
    return args


def print_urls(cfg, out=print):
    link = cfg["urls"]
    out(f"  Browser:             {link['browser']}")
    out(f"  OpenAI base URL:     {link['openai_base_url']}")
    out(f"  Anthropic base URL:  {link['anthropic_base_url']}")


def stop_server(cfg, out=print):
    coli = COLI_SCRIPT
    if not os.path.isfile(coli):
        coli = cfg["launcher"]
    # coli prints UTF-8 (it reconfigures stdout on Windows); the locale codec
    # (cp1252 there) would raise on its box-drawing characters.
    result = subprocess.run([sys.executable, coli, "stop", "--port", str(cfg["port"])],
                            capture_output=True, encoding="utf-8", errors="replace", timeout=120,
                            env=dict(os.environ, COLI_COLOR="0"))
    text = (result.stdout + result.stderr).strip()
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line).strip() for line in text.splitlines()]
    # Keep what `coli stop` reports, not its banner.
    keep = ("stopping", "stopped", "nothing running", "forced", "would stop")
    lines = [line for line in lines if any(word in line for word in keep)]
    for line in lines:
        out(f"  {line}")
    return {"stopped": "nothing running" not in text, "output": "\n".join(lines)}


def tail_file(path, lines=100):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 256 * 1024))
            data = handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-lines:])


# ---------------------------------------------------------------- the interactive flow


class UI:
    def __init__(self, interactive, quiet=False, stream=None):
        self.interactive = interactive
        self.quiet = quiet
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty() if hasattr(self.stream, "isatty") else False
        self._last_progress = 0.0
        self._last_line = ""

    def say(self, text=""):
        if not self.quiet:
            self._end_progress()
            print(text, file=self.stream, flush=True)

    def ask(self, question, default=""):
        if not self.interactive:
            return default
        self._end_progress()
        try:
            answer = input(f"{question} ").strip()
        except EOFError:
            return default
        return answer or default

    def confirm(self, question, default=True):
        if not self.interactive:
            return default
        answer = self.ask(f"{question} [{'Y/n' if default else 'y/N'}]", "")
        if not answer:
            return default
        return answer.lower().startswith("y")

    def progress(self, line, force=False):
        if self.quiet:
            return
        now = time.time()
        if self.tty:
            if force or now - self._last_progress >= 0.25:
                pad = max(0, len(self._last_line) - len(line))
                self.stream.write("\r" + line + " " * pad)
                self.stream.flush()
                self._last_line = line
                self._last_progress = now
        elif force or now - self._last_progress >= 30:
            print(line, file=self.stream, flush=True)
            self._last_progress = now

    def _end_progress(self):
        if self.tty and self._last_line:
            self.stream.write("\n")
            self.stream.flush()
            self._last_line = ""


def human_size(value):
    value = value or 0
    if value >= 100 * GB:
        return f"{value / GB:.0f} GB"
    if value >= GB:
        return f"{value / GB:.1f} GB"
    return f"{value / 1e6:.1f} MB"


def _fmt_eta(seconds):
    if seconds is None or seconds <= 0 or seconds > 30 * 86400:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


MENU_ROWS = 6


def choose_model(ui, rows, wanted=None, show_all=False):
    if wanted:
        for row in rows:
            if row["entry"].id == wanted:
                if not row["fits"]:
                    ui.say(f"  note: {row['entry'].name}: {row['reason']}")
                return row["entry"]
        raise SetupError(f"unknown model {wanted!r}; `coli setup --list` shows the choices")
    fitting = [r for r in rows if r["fits"]]
    if not show_all:
        # A short menu: the recommendation first, then the next smallest
        # downloads. --all lists every model, --model picks any of them.
        fitting, more = fitting[:MENU_ROWS], fitting[MENU_ROWS:]
    else:
        more = []
    if not fitting:
        raise SetupError("no model in the catalog fits this machine:\n" +
                         "\n".join(f"    {r['entry'].name}: {r['reason']}" for r in rows))
    ui.say("Models that fit this machine")
    ui.say(f"  ({setup_catalog.FITS_EXPLAINED})")
    for number, row in enumerate(fitting, 1):
        entry = row["entry"]
        runs = "runs from RAM" if entry.size_class == "small" else "streams from the SSD"
        if entry.modality == "image":
            runs = "makes images"
        tag = "  [recommended]" if row["recommended"] else ""
        ui.say(f"  {number:>2}) {entry.name:<28} {entry.disk_gb:>7.0f} GB   {runs:<20}{tag}".rstrip())
        ui.say(f"      {entry.summary}; {row['reason']}"
               + (f"; {entry.license_note}" if entry.license_note else ""))
    hidden = [r for r in rows if not r["fits"]]
    if more:
        ui.say(f"  ({len(more)} more fit too: `--all` lists them, `--model ID` picks one)")
    if hidden and show_all:
        ui.say("  Not offered here:")
        for row in hidden:
            ui.say(f"      {row['entry'].name}: {row['reason']}")
    elif hidden:
        ui.say(f"  ({len(hidden)} more need more RAM or disk: `coli setup --list` shows why)")
    default = next((i for i, r in enumerate(fitting, 1) if r["recommended"]), 1)
    while True:
        answer = ui.ask(f"Choose a model [Enter = {default}]:", str(default))
        if answer.isdigit() and 1 <= int(answer) <= len(fitting):
            return fitting[int(answer) - 1]["entry"]
        ui.say(f"  type a number from 1 to {len(fitting)}")


def models_root_for(entry, explicit=None):
    """Where the model goes: --dir, else ~/colibri-models; on Windows a fixed
    drive with room when the home drive has none."""
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    root = default_models_root()
    need = entry.disk_gb * GB * 1.02 + 2 * GB if entry else 0
    free = setup_hw.disk_free(root)
    if sys.platform == "win32" and entry and free is not None and free < need:
        for drive in setup_hw.windows_fixed_drives():
            candidate = os.path.join(drive, "colibri-models")
            if (setup_hw.disk_free(drive) or 0) >= need:
                return candidate
    return root


def download_model(ui, entry, model_dir, *, via_windows="auto", verify=True):
    """List, check space, pick the transport, download with progress."""
    token = setup_download.hf_token()
    files = setup_download.list_repo_files(entry.repo, entry.revision, include=entry.include,
                                           exclude=entry.exclude, token=token)
    if not files:
        raise SetupError(f"{entry.repo}: the repository lists no files")
    total = sum(f["size"] for f in files)
    have = setup_download.bytes_present(model_dir, files) if os.path.isdir(model_dir) else 0
    free = setup_hw.disk_free(model_dir)
    if free is not None and free < (total - have) + 1 * GB:
        raise SetupError(f"{model_dir}: {(total - have) / GB:.0f} GB still to download, "
                         f"{free / GB:.0f} GB free. Free some space or pass --dir.")
    curl = None
    # The probe costs about 15 s: worth it only when the download takes longer
    # than that at a good speed.
    if setup_hw.is_wsl() and via_windows != "no" and (via_windows == "yes" or total - have > 2 * GB):
        curl = _maybe_windows_transport(ui, entry, files, token, via_windows)
    if have:
        ui.say(f"  resuming: {human_size(have)} of {human_size(total)} already here")
    started = time.time()
    first = {"done": None}

    def progress(event):
        if first["done"] is None:
            first["done"] = event["done"]
        elapsed = max(time.time() - started, 1e-3)
        rate = (event["done"] - first["done"]) / elapsed
        eta = (event["total"] - event["done"]) / rate if rate > 0 else None
        pct = 100.0 * event["done"] / event["total"] if event["total"] else 100.0
        width = 24
        filled = int(width * pct / 100)
        ui.progress(f"  [{'#' * filled}{'.' * (width - filled)}] {pct:5.1f}%  "
                    f"{human_size(event['done'])}/{human_size(event['total'])}  "
                    f"{setup_download.human_rate(rate)}  {_fmt_eta(eta)} left")
        write_state_throttled("download", model=entry.id, file=event["file"], done=event["done"],
                              total=event["total"], rate=rate, eta_seconds=eta, model_dir=model_dir)

    write_state("download", model=entry.id, done=have, total=total, model_dir=model_dir)
    setup_download.download_repo(entry.repo, entry.revision, model_dir, files, token=token,
                                 progress=progress, curl=curl, verify=verify)
    ui.progress(f"  [{'#' * 24}] 100.0%  {human_size(total)}  done", force=True)
    ui.say("")
    return total


_STATE_LAST = {"t": 0.0}


def write_state_throttled(phase, **fields):
    now = time.time()
    if now - _STATE_LAST["t"] >= 1.0:
        _STATE_LAST["t"] = now
        write_state(phase, **fields)


def _maybe_windows_transport(ui, entry, files, token, via_windows):
    curl = setup_download.windows_curl()
    if not curl:
        if via_windows == "yes":
            ui.say("  curl.exe was not found on the Windows side; downloading from WSL")
        return None
    biggest = max(files, key=lambda f: f["size"])
    url = setup_download.file_url(entry.repo, entry.revision, biggest["path"])
    if via_windows == "yes":
        ui.say("  downloading through Windows (curl.exe), as requested")
        return curl
    ui.say("  WSL: measuring the download speed from Linux and from Windows (about 15 s)...")
    linux = setup_download.probe_throughput(url, token=token)
    windows = setup_download.probe_throughput_windows(url, curl, token=token)
    ui.say(f"    from WSL: {setup_download.human_rate(linux)}   "
           f"from Windows: {setup_download.human_rate(windows)}")
    if windows and (not linux or windows > 2 * linux):
        ui.say("    WSL's own network is much slower here (a known WSL issue; on one machine "
               "63 KB/s against 3.3 MB/s from Windows).")
        if ui.confirm("  Download through Windows (curl.exe) into the same folder?", True):
            return curl
    return None


def run_post_install(entry, launcher_dir, model_dir, ui):
    """The catalog's post-download steps: (tool under tools/, file it creates).
    A step whose file is already there is done."""
    for tool, creates in (entry.post_install if entry else ()):
        if os.path.exists(os.path.join(model_dir, creates)):
            continue
        script = os.path.join(launcher_dir, "tools", f"{tool}.py")
        ui.say(f"  writing {creates} (tools/{tool}.py)")
        result = subprocess.run([sys.executable, script, model_dir], capture_output=True,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0 or not os.path.exists(os.path.join(model_dir, creates)):
            raise SetupError(f"tools/{tool}.py failed: {(result.stderr or result.stdout).strip()[-400:]}")


def plan_summary(model_dir):
    """The real planner on the downloaded files: (one line, warnings)."""
    try:
        from resource_plan import build_plan
        plan = build_plan(model_dir)
    except Exception as error:  # the plan is advice here; `coli plan` reports it fully
        return f"plan not available ({error})", []
    ram = plan["tiers"]["ram"]
    line = (f"dense part {ram['dense_bytes'] / GB:.1f} GB in RAM, expert cache "
            f"{ram['expert_cache_bytes'] / GB:.1f} GB ({ram['cache_slots_per_layer']} per layer), "
            f"projected hit rate {100 * plan.get('projected_hit_rate', 0):.0f}%")
    return line, list(plan.get("warnings") or [])


def describe_existing(cfg):
    model = cfg.get("model") or {}
    return f"{model.get('name') or cfg.get('family')} in {cfg.get('model_dir')} ({cfg.get('backend')})"


def gpu_now_buildable(cfg, tc=None):
    """True when the last setup ran on the CPU for want of a GPU package and
    that package is here now. Looks at the toolchain only: no hardware probe,
    so a plain rerun stays a fast start."""
    pending = cfg.get("gpu_pending") or []
    if not pending or cfg.get("backend") != "cpu":
        return False
    tc = tc or toolchain()
    for kind in pending:
        if not tc.get(f"can_build_{kind}"):
            continue
        if kind == "cuda":
            # A toolkit is here, but it has to build for the cards the last
            # setup found (kept in the configuration: no hardware probe here).
            try:
                family = family_by_id(cfg.get("family"))
            except Exception:
                continue
            if cuda_problem(cfg.get("nvidia") or [], family, tc) is not None:
                continue
        return True
    return False


def config_ready(cfg):
    if not cfg or cfg.get("status") != "ready":
        return False
    if not os.path.isdir(cfg.get("model_dir") or ""):
        return False
    if (cfg.get("model") or {}).get("repo") and not setup_download.is_complete(cfg["model_dir"]):
        return False
    return os.path.isfile(cfg.get("engine") or "") and os.path.isfile(cfg.get("launcher") or "")


def cmd_setup(a, ui=None):
    interactive = sys.stdin.isatty() and not a.yes and not a.json
    ui = ui or UI(interactive, quiet=a.json)
    ui.say("colibri setup\n")
    cfg = load_config()
    picking = bool(a.pick or a.model_dir or a.reconfigure)

    if a.list:
        hw = setup_hw.detect(models_root_for(None, a.dir), probe_gpu=False)
        rows = _recommend_for(hw, a.dir)
        if a.json:
            print(json.dumps([setup_catalog.as_dict(r) for r in rows], indent=2))
        else:
            for row in rows:
                entry = row["entry"]
                mark = "*" if row["recommended"] else " "
                ui.say(f" {mark} {entry.id:<24} {entry.disk_gb:>7.0f} GB  {row['status']:<10} {row['reason']}")
        return 0

    if cfg and not picking and config_ready(cfg):
        if gpu_now_buildable(cfg):
            ui.say("The GPU packages are installed now: rebuilding the engine for the GPU.\n")
        else:
            ui.say(f"Already set up: {describe_existing(cfg)}")
            refresh_configured(cfg, out=ui.say)
            return _finish(ui, cfg, a)
    # A setup that stopped half way (pending), or a finished one whose engine or
    # files went missing (ready, but not runnable): continue with the same choice.
    resume = cfg if (cfg and cfg.get("status") in ("pending", "ready") and not picking) else None

    ui.say("Your machine")
    probe_root = models_root_for(None, a.dir or (resume or {}).get("models_root"))
    hw = setup_hw.detect(probe_root)
    ui.say(setup_hw.format_report(hw))
    ui.say("")
    if (hw["memory"].get("total") or 0) and hw["memory"].get("available") and \
            hw["memory"]["available"] < 0.6 * hw["memory"]["total"]:
        ui.say(f"  note: only {hw['memory']['available'] / GB:.1f} GB of RAM is free right now; "
               "closing other programs gives the model more room.\n")

    entry = None
    resume_dir = None
    if resume and not (resume.get("model") or {}).get("id"):
        resume_dir = resume.get("model_dir")
    if a.model_dir or resume_dir:
        model_dir = os.path.abspath(os.path.expanduser(a.model_dir or resume_dir))
        from family_registry import resolve_model
        try:
            family = resolve_model(model_dir).descriptor
        except Exception as error:
            raise SetupError(f"{model_dir}: not a model colibri recognises ({error})")
    else:
        if resume and resume.get("model", {}).get("id"):
            try:
                entry = setup_catalog.by_id(resume["model"]["id"])
            except KeyError:
                raise SetupError(f"the model of the last setup ({resume['model']['id']}) is not in "
                                 "the catalog any more: run `coli setup --reconfigure`")
            ui.say(f"Resuming the setup of {entry.name}.")
        else:
            rows = _recommend_for(hw, a.dir)
            entry = choose_model(ui, rows, wanted=a.pick, show_all=a.all)
        family = family_by_id(entry.family)
        root = models_root_for(entry, a.dir) if not resume else os.path.dirname(resume["model_dir"])
        model_dir = resume["model_dir"] if resume else os.path.join(root, entry.id)
        for warning in setup_hw.path_warnings(model_dir, hw["os"].get("wsl")):
            ui.say(f"  warning: {warning}")
            if not ui.confirm("  Use this folder anyway?", False if ui.interactive else True):
                raise SetupError("pass --dir with a folder on the Linux disk")
        ui.say(f"\nModel: {entry.name} ({entry.repo}) -> {model_dir}")

    host = a.host or (resume or {}).get("host") or DEFAULT_HOST
    port = int(a.port or (resume or {}).get("port") or DEFAULT_PORT)
    requested = "cpu" if a.no_gpu else (a.backend or (resume or {}).get("requested_backend") or "auto")
    pending = {"status": "pending", "model": {"id": entry.id if entry else None,
                                              "name": entry.name if entry else family.display_name,
                                              "repo": entry.repo if entry else None,
                                              "revision": entry.revision if entry else None},
               "model_dir": model_dir, "models_root": os.path.dirname(model_dir),
               "family": family.id, "host": host, "port": port, "requested_backend": requested}
    if not resume or picking:
        save_config(pending)

    tc = toolchain()
    # A build that failed in the run this one resumes is not tried again; a new
    # choice (another model, --reconfigure, --backend) starts clean.
    failed = [] if (a.backend or a.no_gpu or picking or not resume) else \
        [b for b in (resume.get("failed_builds") or []) if b in ("cuda", "vulkan")]
    decision = choose_backend(hw, family, tc, requested, exclude=tuple(failed))
    if failed:
        decision["reason"] = (f"the {BACKEND_NAME[failed[-1]]} build failed in the last run "
                              f"(log: {log_path(build_log_name(failed[-1]))}): "
                              f"using {USING[decision['backend']]}")
    ui.say(f"\nEngine: {family.engine_artifact} with {decision['backend'].upper()} "
           f"({decision['reason']})")
    for kind, hint in decision["missing"]:
        ui.say(f"  to use the GPU through {kind.upper()}, first run:  {hint}")
    if decision["missing"] and ui.interactive and decision["backend"] == "cpu":
        if not ui.confirm("  Continue on the CPU for now (rerun after installing to switch)?", True):
            raise SetupError("stopped so you can install the GPU packages; rerun afterwards")
    engine_info, decision, failed = resolve_with_fallback(ui, family, entry, decision, tc, hw,
                                                          requested, failed)
    if failed:
        current = load_config()
        if current:
            save_config(dict(current, failed_builds=failed))
    engine_info["gpu"] = decision.get("gpu") if engine_info["backend"] != "cpu" else None
    ui.say(f"  engine ready: {engine_info['engine']} ({engine_info['backend']}, {engine_info['source']})")

    if entry is not None:
        if setup_download.is_complete(model_dir):
            ui.say("\nDownload: already complete")
        else:
            ui.say(f"\nDownload: {entry.repo} ({human_size(entry.disk_gb * GB)}); safe to "
                   "interrupt, rerun to continue")
            download_model(ui, entry, model_dir, via_windows=a.via_windows, verify=not a.no_verify)
        run_post_install(entry, engine_info["launcher_dir"], model_dir, ui)

    if not web_dist_dir(engine_info["launcher_dir"]):
        ensure_web_dist(engine_info["launcher_dir"], current_version(), out=ui.say)

    line, warnings = plan_summary(model_dir)
    ui.say(f"\nPlan: {line}")
    for warning in warnings:
        ui.say(f"  plan warning: {warning}")

    cfg = make_config(model_dir=model_dir, family=family, entry=entry, engine_info=engine_info,
                      hw=hw, host=host, port=port)
    cfg["requested_backend"] = requested
    # A GPU this setup found but could not build for: a rerun checks whether
    # the packages arrived and rebuilds (gpu_now_buildable).
    cfg["gpu_pending"] = ([kind for kind, _hint in decision["missing"]]
                          if engine_info["backend"] == "cpu" and requested == "auto" else [])
    # The cards a pending CUDA build has to serve: a rerun checks the toolkit
    # against them without probing the hardware again.
    cfg["nvidia"] = [{"name": card.get("name"), "compute_cap": card.get("compute_cap")}
                     for card in (hw.get("gpu") or {}).get("nvidia") or []]
    cfg["failed_builds"] = failed
    cfg = save_config(cfg)
    write_state("ready", model=cfg["model"].get("id"), model_dir=model_dir)
    ui.say(f"\nSaved the run configuration: {config_path()}")
    ui.say(f"  same as: {equivalent_command(cfg)}")
    return _finish(ui, cfg, a)


def _recommend_for(hw, explicit_dir):
    root = models_root_for(None, explicit_dir)
    free = setup_hw.disk_free(root)
    downloaded = {}
    for entry in setup_catalog.catalog():
        manifest = setup_download.read_manifest(os.path.join(root, entry.id))
        if manifest:
            downloaded[entry.id] = setup_download.bytes_present(os.path.join(root, entry.id),
                                                                manifest.get("files", []))
    return setup_catalog.recommend(hw["memory"].get("total"), free, os_name=host_os(),
                                   machine=hw["os"].get("machine"), downloaded=downloaded)


def _finish(ui, cfg, a):
    if a.json:
        result = {"config": cfg, "config_path": config_path()}
        if not a.no_start:
            result["status"] = start_server(cfg, background=True, open_browser=not a.no_browser,
                                            out=lambda *_: None)
            write_state("started", urls=cfg["urls"])
        print(json.dumps(result, indent=2))
        return 0
    if a.no_start:
        ui.say("\nReady. Start it with:  coli start   (stop: coli stop)")
        print_urls(cfg, ui.say)
        return 0
    ui.say("\nStarting colibri")
    if a.background:
        start_server(cfg, background=True, open_browser=not a.no_browser, out=ui.say)
        write_state("started", urls=cfg["urls"])
        ui.say("  stop: coli stop")
        return 0
    return start_server(cfg, background=False, open_browser=not a.no_browser, out=ui.say)


def cmd_start(a):
    cfg = load_config()
    if not cfg or not config_ready(cfg):
        raise SetupError("nothing is set up yet (or the setup did not finish): run `coli setup`")
    refresh_configured(cfg, out=(lambda *_: None) if getattr(a, "json", False) else print)
    if getattr(a, "json", False):
        print(json.dumps(start_server(cfg, background=True, open_browser=not a.no_browser,
                                      out=lambda *_: None), indent=2))
        return 0
    if a.background:
        start_server(cfg, background=True, open_browser=not a.no_browser)
        print("  stop: coli stop")
        return 0
    return start_server(cfg, background=False, open_browser=not a.no_browser)


def status_report():
    cfg = load_config()
    state = read_state()
    report = {"configured": bool(cfg), "config_path": config_path(), "setup": None,
              "server": None, "install": None}
    if cfg:
        report["setup"] = {"status": cfg.get("status"), "model": cfg.get("model"),
                           "model_dir": cfg.get("model_dir"), "backend": cfg.get("backend"),
                           "engine": cfg.get("engine"), "env": cfg.get("env"),
                           "ready": config_ready(cfg)}
        if cfg.get("port"):
            cfg.setdefault("host", DEFAULT_HOST)
            report["server"] = server_status(cfg)
    if state:
        state = dict(state)
        state["running"] = (_pid_alive(state.get("pid")) and state.get("phase")
                            not in ("ready", "error", "done", "started", "interrupted"))
        report["install"] = state
    return report


def cmd_status(a):
    report = status_report()
    if a.json:
        print(json.dumps(report, indent=2))
        return 0
    if not report["configured"]:
        print("colibri is not set up yet: run `coli setup`")
        return 1
    setup = report["setup"]
    model = setup.get("model") or {}
    print(f"{'model':<10}{model.get('name')}  ({setup.get('model_dir')})")
    print(f"{'engine':<10}{setup.get('engine')}  [{setup.get('backend')}]")
    install = report.get("install") or {}
    if install.get("running") and install.get("phase") == "download" and install.get("total"):
        pct = 100.0 * install.get("done", 0) / install["total"]
        print(f"{'install':<10}downloading {pct:.1f}%  ({install.get('done', 0) / GB:.1f}/"
              f"{install['total'] / GB:.1f} GB, {setup_download.human_rate(install.get('rate'))})")
    elif install.get("running"):
        print(f"{'install':<10}{install.get('phase')}")
    elif not setup.get("ready"):
        print(f"{'install':<10}unfinished"
              + (f" ({install['message']})" if install.get("message") else "")
              + ": run `coli setup` to continue")
    server = report.get("server") or {}
    print(f"{'server':<10}{server.get('state', 'not configured')}"
          + (f" (pid {server['pid']})" if server.get("pid") else ""))
    if server:
        for key, label in (("browser", "browser"), ("openai_base_url", "openai"),
                           ("anthropic_base_url", "anthropic")):
            print(f"{label:<10}{server['urls'][key]}")
        if server.get("model_id"):
            print(f"{'model id':<10}{server['model_id']}  (the `model` field for API requests)")
        if server.get("tokens_per_second"):
            print(f"{'speed':<10}{server['tokens_per_second']} tok/s on the last answer "
                  "(prompt time included)")
    return 0


def cmd_logs(a):
    name = "install" if a.install else "serve"
    path = log_path(name)
    text = tail_file(path, a.lines)
    if not text:
        print(f"no {name} log yet ({path})")
        cfg = load_config() if name == "serve" else None
        try:
            status = server_status(cfg) if cfg and cfg.get("port") else None
        except (OSError, ValueError, KeyError):
            status = None
        if status and status["state"] != "stopped":
            # Started some other way (`coli serve` in a terminal): its output is there.
            print(f"colibri is running ({status['state']}"
                  + (f", pid {status['pid']}" if status.get("pid") else "")
                  + "), started in the foreground: its output is in the terminal that "
                  "started it; `coli start --background` writes it to this log instead")
        return 1
    print(text)
    return 0


def configured_port(default=DEFAULT_PORT):
    cfg = load_config()
    try:
        return int(cfg["port"]) if cfg and cfg.get("port") else default
    except (TypeError, ValueError):
        return default


def run(handler, a):
    """Entry for coli: SetupError ends the command with its message, no traceback."""
    try:
        return handler(a) or 0
    except SetupError as error:
        write_state("error", message=str(error))
        print(f"\nsetup stopped: {error}", file=sys.stderr)
        return 1
    except setup_download.DownloadError as error:
        write_state("error", message=str(error))
        print(f"\ndownload stopped: {error}\nRun the same command again to continue.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        write_state("interrupted")
        print("\ninterrupted. Run the same command again to continue where it stopped.",
              file=sys.stderr)
        return 130


