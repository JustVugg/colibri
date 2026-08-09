# --- Target detection ---
# Ask the compiler what it actually targets (rofl0r's note on #129): a gcc/clang
# toolchain reports its target triple via `-dumpmachine`, e.g.
#   x86_64-w64-mingw32   x86_64-pc-cygwin   x86_64-unknown-linux-gnu
#   arm64-apple-darwin   powerpc64le-unknown-linux-gnu
# The triple follows the toolchain, not the host shell that `uname` reports, so
# it is correct under a native-Windows shell (no `uname` on PATH) AND when
# cross-compiling (e.g. make CC=x86_64-w64-mingw32-gcc on Linux). CC is chosen
# per-target below, so probe with the user's CC if they set one, else gcc
# (present on Linux and mingw, and a clang shim on macOS).
DETECT_CC := $(if $(filter default,$(origin CC)),gcc,$(CC))
TRIPLET   := $(shell $(DETECT_CC) -dumpmachine 2>/dev/null)

MINGW  := $(findstring mingw,$(TRIPLET))
CYGWIN := $(findstring cygwin,$(TRIPLET))
DARWIN := $(findstring darwin,$(TRIPLET))
LINUX  := $(findstring linux,$(TRIPLET))
IS_WIN := $(MINGW)$(CYGWIN)

# Appended to every platform's flags, so a caller can add to the build without
# replacing it. Overriding CFLAGS= on the command line would drop that
# platform's OpenMP and -march flags, which is how a sanitizer run ends up
# silently unsanitized. Used by test-asan; also the supported way to pass
# your own flag: make colibri EXTRA_CFLAGS=-DFOO
EXTRA_CFLAGS  ?=
EXTRA_LDFLAGS ?=

# Fallbacks for the rare toolchain that does not answer -dumpmachine: keep the
# #129 signal (OS=Windows_NT, set in every Windows shell) for Windows, then
# `uname` for everything else, so no host regresses.
ifeq ($(TRIPLET),)
IS_WIN := $(if $(filter Windows_NT,$(OS)),1,)
ifeq ($(IS_WIN),)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
DARWIN  := $(findstring Darwin,$(UNAME_S))
endif
endif

TARGET_CPU := $(firstword $(subst -, ,$(TRIPLET)))
ifeq ($(TARGET_CPU),)
TARGET_CPU := $(UNAME_M)
endif
X86_64  := $(filter x86_64 amd64,$(TARGET_CPU))
AARCH64 := $(filter aarch64 arm64,$(TARGET_CPU))
PPC64   := $(filter powerpc64% ppc64%,$(TARGET_CPU))

ifneq (,$(DARWIN))
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
# `brew --prefix libomp` can print the formula's prospective path even when it
# is not installed, so verify both artifacts before adding unusable flags.
ifneq ($(and $(OMPDIR),$(wildcard $(OMPDIR)/include/omp.h),$(wildcard $(OMPDIR)/lib/libomp.*)),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp not found: building single-threaded. For multithreading: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function $(EXTRA_CFLAGS)
# Opt-in: ARCH=native appends -mcpu=native (arm64 clang uses -mcpu, not -march),
# which unlocks the i8mm SMMLA int8/int4 dot kernels in colibri.c. ARCH unset ->
# no -mcpu, default build byte-identical. Apple clang knows apple-m4 / native.
# For older X86_64 Macs (for example, Mac Pro 2019) we need to use -march
ifneq ($(ARCH),)
ifneq (,$(X86_64))
CFLAGS += -march=$(ARCH)
else
CFLAGS += -mcpu=$(ARCH)
endif
endif
LDFLAGS = -lm $(OMPL) $(EXTRA_LDFLAGS)
EXE     =
else ifneq ($(IS_WIN),)
# --- Windows 11 x86-64 (MinGW-w64 / MSYS2) ---
# GCC + libgomp + winpthreads: pthread, OpenMP, clock_gettime, opendir/readdir,
# AVX2 intrinsics - tutto gratis, nessun porting.
# ARCH default = x86-64-v3 (portable binary with AVX2). For max speed on THIS
# machine use ARCH=native: on AVX-VNNI CPUs (Intel Alder Lake+, Meteor Lake+)
# it also unlocks the 128-bit VPDPBUSD int8/int4 dot kernel (dot_i8i8/dot_i4i8),
# which the x86-64-v3 baseline does not define. The #ifdef guards in colibri.c mean
# a v3 build simply compiles out the VNNI path - safe on any x86-64.
CC      = gcc
ARCH   ?= x86-64-v3
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function $(EXTRA_CFLAGS)
# -lpsapi: compat.h calls GetProcessMemoryInfo (rss_gb). It's linked via
# #pragma comment(lib,"psapi.lib") for MSVC, but MinGW gcc ignores that pragma
# (warns), so link psapi explicitly or the build fails undefined-reference on
# modern GCC (e.g. 16.x under UCRT). Harmless where the pragma also resolves it.
LDFLAGS = -lm -fopenmp -static -lpsapi $(EXTRA_LDFLAGS)
EXE     = .exe
else
ifneq (,$(PPC64))
# --- Linux PowerPC (POWER8/POWER9/POWER10) ---
# PowerPC GCC uses -mcpu, not -march. ARCH=native works on gcc >= 4.7.
# The AVX2/NEON kernels fall back to the portable scalar C path
# (validated token-exact vs the transformers oracle on a POWER8 S824).
CC      = gcc
ARCH   ?= native
CFLAGS  = -O3 -mcpu=$(ARCH) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function $(EXTRA_CFLAGS)
LDFLAGS = -lm -fopenmp -pthread $(EXTRA_LDFLAGS)
EXE     =
else ifneq (,$(AARCH64))
# --- Linux / *BSD aarch64 (#631) ---
# NEON e' baseline, ma i kernel SDOT/SMMLA int8-int4 vivono dietro
# __ARM_FEATURE_DOTPROD / __ARM_FEATURE_MATMUL_INT8. A gcc that does not know
# this core (gcc 13 vs Cortex-X925 on GB10: #631) degrades native to plain
# armv8-a SILENTLY: the kernels compile out, no warning, the banner says
# "idot: neon" and int4 S=1 decode falls back to f32. So for the default
# ARCH=native, probe the macro; if it is missing, re-add the features the
# silicon itself reports in /proc/cpuinfo -- but keep them only if this
# compiler accepts the modifiers (gcc < 10 has no +i8mm: composing blindly
# would turn a slow build into a broken one).
CC      = gcc
ARCH   ?= native
ifeq ($(ARCH),native)
ARCHFLAG := -mcpu=native
ifeq ($(shell $(CC) -mcpu=native -dM -E - </dev/null 2>/dev/null | grep -c __ARM_FEATURE_MATMUL_INT8),0)
CPUFEAT  := $(shell grep -m1 '^Features' /proc/cpuinfo 2>/dev/null)
ARCH_EXT := $(if $(filter asimddp,$(CPUFEAT)),+dotprod)$(if $(filter i8mm,$(CPUFEAT)),+i8mm)
ifneq ($(ARCH_EXT),)
ifneq ($(shell $(CC) -march=armv8-a$(ARCH_EXT) -dM -E - </dev/null 2>/dev/null | grep -cE '__ARM_FEATURE_DOTPROD|__ARM_FEATURE_MATMUL_INT8'),0)
ARCHFLAG := -march=armv8-a$(ARCH_EXT)
else
$(warning $(CC) does not accept$(subst +, +,$(ARCH_EXT)): int8/int4 idot kernels compile out (#631))
endif
endif
endif
else ifneq (,$(filter armv%,$(ARCH)))
ARCHFLAG := -march=$(ARCH)
else
ARCHFLAG := -mcpu=$(ARCH)
endif
CFLAGS  = -O3 $(ARCHFLAG) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function $(EXTRA_CFLAGS)
LDFLAGS = -lm -fopenmp -pthread $(EXTRA_LDFLAGS)
EXE     =
else
# --- Linux / *BSD x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
# -pthread: Linux lo tira dentro via -fopenmp, i *BSD no e le pthread_* non risolvono (#219).
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function $(EXTRA_CFLAGS)
LDFLAGS = -lm -fopenmp -pthread $(EXTRA_LDFLAGS)
EXE     =
endif
endif

# --- install ---
PREFIX     ?= /usr/local
BINDIR     ?= $(PREFIX)/bin
LIBEXECDIR ?= $(PREFIX)/libexec/colibri
INSTALL    ?= install

# CUDA=1 adds an opt-in backend for resident tensors. The default build remains
# pure C and keeps the original zero-dependency runtime.
#
# Two paths:
#   - Linux/macOS: CUDA=1 links backend_cuda.o directly (cudart via -l).
#   - Windows:      CUDA_DLL=1 builds a standalone coli_cuda.dll (nvcc+MSVC),
#                   then the host glm.exe loads it at runtime via backend_loader.c
#                   (LoadLibrary/GetProcAddress). MinGW gcc cannot compile .cu
#                   (nvcc needs cl.exe), and cross-linking MSVC objects into a
#                   gcc binary is fragile — the DLL split keeps the toolchains
#                   clean. See backend_loader.c and README "cuda-dll" below.
CUDA      ?= 0
CUDA_DLL  ?= 0
COLI_ANS  ?= 0
DIETGPU_ROOT ?=
ifneq ($(IS_WIN),)
# the CUDA installer sets CUDA_PATH system-wide (e.g. C:\Program Files\NVIDIA
# GPU Computing Toolkit\CUDA\v13.2); fall back to it before the POSIX default.
# NVCC defaults to plain `nvcc` from PATH: CUDA_PATH contains spaces, which the
# unquoted recipe checks cannot survive — and the cuda-dll recipe already
# requires an MSVC environment (vcvars64) on PATH, so requiring the CUDA bin
# directory too is symmetric. The installer adds it by default.
CUDA_HOME ?= $(subst \,/,$(CUDA_PATH))
NVCC      ?= nvcc
# Host compiler override for nvcc (e.g. Fedora ships g++16 but CUDA needs 15):
#   make glm CUDA=1 NVCC_CCBIN=g++-15        (opt-in: unset = nvcc's own default)
ifneq ($(NVCC_CCBIN),)
NVCCFLAGS += -ccbin $(NVCC_CCBIN)
endif
else
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
endif
# CUDA target selection. Default `native` builds SASS for the build machine's GPU
# only (fast local dev). A headless CI/Docker build has no GPU, so `-arch=native`
# there detects nothing and the shipped coli_cuda.dll fails at first kernel dispatch.
# For a portable/release DLL that runs on any Ampere..Blackwell card AND JITs on
# newer archs, build with CUDA_ARCH=portable: it emits SASS for sm_80/86/89/90/120
# plus a compute_120 PTX fallback (sm_120 / RTX 50-series needs CUDA >= 12.8).
CUDA_ARCH ?= native
ifeq ($(CUDA_ARCH),portable)
CUDA_GENCODE = -gencode arch=compute_80,code=sm_80 \
               -gencode arch=compute_86,code=sm_86 \
               -gencode arch=compute_89,code=sm_89 \
               -gencode arch=compute_90,code=sm_90 \
               -gencode arch=compute_120,code=sm_120 \
               -gencode arch=compute_120,code=compute_120
else
CUDA_GENCODE = -arch=$(CUDA_ARCH)
endif
ifneq ($(IS_WIN),)
# nvcc's host compiler on Windows is MSVC cl.exe: the GCC-style
# -Xcompiler=-Wall,-Wextra is rejected ("D8021 invalid numeric argument
# '/Wextra'"), which made `make cuda-dll` unbuildable as shipped. -W3 is
# the MSVC warning level (dash form, NOT /W3: MSYS make would mangle the
# slash-form into a filesystem path).
NVCCFLAGS ?= -O3 -std=c++17 $(CUDA_GENCODE) -Xcompiler=-W3
else
NVCCFLAGS ?= -O3 -std=c++17 $(CUDA_GENCODE) -Xcompiler=-Wall,-Wextra
endif
# HIP=1 builds the SAME backend for AMD GPUs via ROCm: backend_cuda.cu is
# compiled unchanged through backend_gpu_compat.h (one source, two vendors,
# like compat.h does for Windows). HIP_ARCH=native targets the GPU in this
# machine; set an explicit arch (e.g. HIP_ARCH=gfx1201) when distributing.
HIP       ?= 0
# HIP_DLL=1 is the AMD sibling of CUDA_DLL=1 (Windows only): MinGW gcc cannot
# compile .cu and hipcc on Windows targets the MSVC ABI, so the backend goes
# into a standalone coli_hip.dll and the host loads it at runtime — exactly the
# split the CUDA path already uses. The host never links amdhip64. Same single
# source and same coli_cuda_* ABI the Linux HIP path reuses, so the host keeps
# -DCOLI_CUDA; -DCOLI_HIP_DLL only records WHICH DLL it will look for.
HIP_DLL   ?= 0
ifneq ($(IS_WIN),)
# One runtime DLL per host binary: the loader resolves a single hardcoded DLL
# name, so a host built for both would silently be a host built for whichever
# arm ran last. Checked FIRST, before the arch and flag guards below, so the
# contradiction is reported instead of whichever downstream check trips.
ifeq ($(CUDA_DLL)$(HIP_DLL),11)
$(error choose CUDA_DLL=1 or HIP_DLL=1, not both)
endif
endif
ROCM_HOME ?= /opt/rocm
ifneq ($(IS_WIN),)
# --- Windows HIP SDK contract (HIP_DLL only; the Linux ROCM_HOME contract
# above is untouched) ---
# NOTHING here may hardcode an install location: HIP_PATH is the variable the
# official Windows HIP SDK installer sets, and a source-built or relocated SDK
# is selected with HIP_SDK_ROOT=<path> (e.g. from `rocm-sdk path --root`).
# Backslashes become forward slashes so the value survives the shell.
#
# Every component is separately overridable because packaged ROCm layouts do
# not all keep runtime, development and device files under one root:
#   HIP_SDK_ROOT         SDK root                  (--hip-path)
#   HIP_INCLUDE_DIR      headers                   (-I)
#   HIP_LIB_DIR          amdhip64.lib import lib   (-L)
#   HIP_DEVICE_LIB_PATH  amdgcn device bitcode     (--rocm-device-lib-path)
HIP_SDK_ROOT        ?= $(subst \,/,$(HIP_PATH))
HIP_BIN_DIR         ?= $(HIP_SDK_ROOT)/bin
HIP_INCLUDE_DIR     ?= $(HIP_SDK_ROOT)/include
HIP_LIB_DIR         ?= $(HIP_SDK_ROOT)/lib
HIP_DEVICE_LIB_PATH ?= $(HIP_SDK_ROOT)/lib/llvm/amdgcn/bitcode
HIPCC               ?= $(HIP_BIN_DIR)/hipcc.exe
else
HIPCC     ?= $(ROCM_HOME)/bin/hipcc
endif
HIP_ARCH  ?= native
# HIP_ARCH accepts a list (space- or comma-separated, e.g. "gfx906 gfx1030")
# to fat-bundle several targets into one binary; COLI_GPUS picks at runtime.
# `native` is resolved to the local GPUs' arch names HERE (not left to hipcc):
# the NO_WMMA_ARCHS filter below must see real gfx names to know whether the
# rocWMMA paths can compile at all. `override` beats command-line assignment.
comma := ,
override HIP_ARCH := $(subst $(comma), ,$(HIP_ARCH))
ifeq ($(strip $(HIP_ARCH)),native)
ifeq ($(HIP),1)
override HIP_ARCH := $(shell $(ROCM_HOME)/bin/rocm_agent_enumerator 2>/dev/null | grep -v gfx000 | sort -u)
ifeq ($(strip $(HIP_ARCH)),)
$(error HIP_ARCH=native but rocm_agent_enumerator found no GPUs; set HIP_ARCH=gfxNNNN explicitly)
endif
endif
endif
# Guards for the Windows HIP path, active ONLY when the hip-dll TARGET is
# requested — never for HIP_DLL=1 alone. HIP_DLL=1 selects the host arm, which
# compiles backend_loader.c and links colibri.exe against no HIP library and no
# SDK header: demanding an SDK there blocks a build that needs nothing from one
# (and HIP_PATH is not visible in every shell, e.g. an MSYS2 login shell).
# The default CPU build must never depend on a HIP path either.
#
# Only the string checks live here. Existence checks (hipcc, headers, import
# library, device bitcode) run in the hip-dll recipe instead. The stock Windows
# HIP SDK location may contain spaces, and $(wildcard) splits its argument on
# them, so a parse-time $(wildcard) check can incorrectly report a valid
# installation as missing. The recipe's shell tests are quoted and handle
# spaces correctly.
ifneq ($(IS_WIN),)
HIP_DLL_TARGET_REQUESTED := $(filter hip-dll,$(MAKECMDGOALS))
ifneq ($(HIP_DLL_TARGET_REQUESTED),)
ifeq ($(strip $(HIP_SDK_ROOT)),)
$(error Windows HIP build: no HIP SDK selected. Set HIP_PATH, or pass HIP_SDK_ROOT=<path> (e.g. from `rocm-sdk path --root`))
endif
# Windows has no rocm_agent_enumerator to resolve `native` against, and the
# enumerator branch above is HIP=1-only, so an unset HIP_ARCH would reach hipcc
# as --offload-arch=native and fail deep inside the compiler. Demand the arch
# here, where the cause is still visible.
ifeq ($(strip $(HIP_ARCH)),native)
$(error Windows HIP build: set an explicit HIP_ARCH=gfxNNNN (rocm_agent_enumerator is not available here), e.g. HIP_ARCH=gfx1151)
endif
ifeq ($(strip $(HIP_ARCH)),)
$(error Windows HIP build: HIP_ARCH is empty; set HIP_ARCH=gfxNNNN, e.g. HIP_ARCH=gfx1151)
endif
endif
endif
ifneq ($(IS_WIN),)
# No -fPIE on Windows: it is there for the Linux PIE link of backend_cuda.o into
# the gcc engine (the readelf guard in ci.yml exists because dropping it made
# that link fail), and it is meaningless for a PE DLL MinGW never links against.
# -Wall/-Wextra DO carry over unchanged — hipcc is clang and takes GCC-style
# warning flags directly. nvcc's -Xcompiler=-W3 workaround above is specific to
# nvcc driving cl.exe as its host compiler and must NOT be copied here.
HIPCCFLAGS ?= -O3 -std=c++17 -x hip $(foreach a,$(HIP_ARCH),--offload-arch=$(a)) -Wall -Wextra
else
HIPCCFLAGS ?= -O3 -std=c++17 -x hip $(foreach a,$(HIP_ARCH),--offload-arch=$(a)) -Wall -Wextra -fPIE
endif
GPUCC      = $(NVCC)
GPUFLAGS   = $(NVCCFLAGS)
GPUCC_NAME = nvcc (set CUDA_HOME or NVCC)
# PYTHON is a HOST tool (clean/test-c/test-python run it on the build machine),
# so key it off the host, NOT $(IS_WIN) — which is derived from the *target*
# triple ($(CC) -dumpmachine). Otherwise a cross build (make CC=x86_64-w64-
# mingw32-gcc ... on Linux) sets IS_WIN and picks `python`, breaking clean/test
# on hosts where only `python3` exists. $(OS)=Windows_NT in every Windows shell
# (the #129 signal) and is empty on Linux/macOS.
ifeq ($(OS),Windows_NT)
PYTHON    ?= python
else
PYTHON    ?= python3
endif
CUDA_OBJ  =

# DeepSeek V4 is a separate x86-64/aarch64 runtime. Portable infrastructure
# tests run everywhere; the engine and token-exact fixture run on x86-64
# Linux/Windows and aarch64 Linux. SIMD fast paths are gated per-feature
# inside the sources, so a plain armv8-a build stays correct.
COLI_V4_SUPPORTED :=
ifneq (,$(X86_64))
ifneq (,$(IS_WIN)$(LINUX))
COLI_V4_SUPPORTED := 1
endif
endif
ifneq (,$(AARCH64))
ifneq (,$(LINUX))
COLI_V4_SUPPORTED := 1
endif
endif

# Gates are exactly the tests that have a build rule further down -- derived from those
# rules, so adding a gate means adding your tests/test_*.c and its rule, and nothing else.
# There is no shared list to conflict on.
#
# This replaced a single hand-written TEST_BINS line. c/Makefile appears in most open PRs,
# and any two that each added a test conflicted on that line by construction; #386 hit it
# twice while being rebased. A first attempt (#731) globbed tests/test_*.c, which was wrong
# in the other direction: it promoted files that deliberately have no rule (a branch's
# work-in-progress test, test_fse, test_tok, test_tok_kimi, test_vk_mxfp4) into gates, and
# they fail to link. Having a rule is the honest definition of "this is a gate".
TEST_RULES  := $(shell sed -n 's|^tests/\(test_[a-z0-9_]*\)\$$(EXE):.*|\1|p' $(firstword $(MAKEFILE_LIST)))
# test_uring is Linux-only. V4 engine tests are appended below only on supported
# x86-64 Linux/Windows and aarch64 Linux hosts; the V4 infrastructure tests have
# unconditional rules and therefore remain portable gates.
TEST_EXCLUDE = test_uring test_deepseek_v4 test_v4_ownership
TEST_BINS    = $(addprefix tests/,$(addsuffix $(EXE),$(filter-out $(TEST_EXCLUDE),$(TEST_RULES))))
ifneq (,$(LINUX))
TEST_BINS += tests/test_uring$(EXE)
endif
ifeq ($(COLI_V4_SUPPORTED),1)
TEST_BINS += tests/test_deepseek_v4$(EXE) tests/test_v4_ownership$(EXE)
endif

# Windows CUDA/HIP DLL path: host links the loader, NOT cudart or amdhip64.
# (CUDA_DLL/HIP_DLL mutual exclusion is enforced at the HIP_DLL declaration.)
ifneq ($(IS_WIN),)
ifeq ($(CUDA_DLL),1)
CFLAGS  += -DCOLI_CUDA
CUDA_OBJ = backend_loader.o
endif
ifeq ($(HIP_DLL),1)
# -DCOLI_CUDA keeps the existing shared GPU ABI (Linux HIP already reuses the
# coli_cuda_* surface); -DCOLI_HIP_DLL is only the build discriminator, so a
# later phase can point the loader at coli_hip.dll. No LDFLAGS change: linking
# amdhip64 into the MinGW host is exactly what the DLL split exists to avoid.
CFLAGS  += -DCOLI_CUDA -DCOLI_HIP_DLL
CUDA_OBJ = backend_loader.o
endif
endif

# Linux CUDA direct-link path (unchanged).
ifeq ($(CUDA),1)
ifneq (,$(DARWIN))
$(error CUDA=1 is supported only on Linux)
endif
ifneq ($(IS_WIN),)
# On Windows use CUDA_DLL=1 (runtime DLL), not CUDA=1 (direct link).
$(error On Windows use: make CUDA_DLL=1 cuda-dll  (see backend_loader.c))
endif
ifeq ($(HIP),1)
$(error choose CUDA=1 or HIP=1, not both)
endif
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
INK_CUDA_OBJ = backend_cuda_ink.o
ifeq ($(COLI_ANS),1)
ifeq ($(strip $(DIETGPU_ROOT)),)
$(error DIETGPU_ROOT is required when COLI_ANS=1)
endif
CFLAGS   += -DCOLI_ANS
GPUFLAGS += -DCOLI_ANS -I$(DIETGPU_ROOT) -I$(DIETGPU_ROOT)/build/third_party/glog -I$(DIETGPU_ROOT)/third_party/glog/src
ANS_LIBS  = -L$(DIETGPU_ROOT)/build/lib -Wl,-rpath,$(DIETGPU_ROOT)/build/lib -lgpu_ans -ldietgpu_utils -lglog
ANS_NVCC_LIBS = -L$(DIETGPU_ROOT)/build/lib -Xlinker -rpath -Xlinker $(DIETGPU_ROOT)/build/lib -lgpu_ans -ldietgpu_utils -lglog
LDFLAGS  += $(ANS_LIBS)
endif
endif

ifeq ($(HIP),1)
ifeq (,$(LINUX))
$(error HIP=1 is supported only on Linux)
endif
CFLAGS    += -DCOLI_CUDA
LDFLAGS   += -L$(ROCM_HOME)/lib -Wl,-rpath,$(ROCM_HOME)/lib -lamdhip64 -lstdc++
CUDA_OBJ   = backend_cuda.o
GPUCC      = $(HIPCC)
GPUFLAGS   = $(HIPCCFLAGS)
GPUCC_NAME = hipcc (set ROCM_HOME or HIPCC)
# rocWMMA needs matrix cores (MFMA gfx908+ / WMMA gfx11xx); on these archs its
# headers static_assert (ROCm/TheRock#1944). Compile the WMMA paths out and
# let backend_gpu_compat.h fall back via COLI_GPU_HAS_WMMA=0. The flag must
# reach both hipcc passes, hence a compile define, not an arch macro. If a
# HIP_ARCH=native build dies inside rocwmma, set the arch explicitly.
NO_WMMA_ARCHS = gfx803 gfx900 gfx902 gfx904 gfx906 gfx90c \
                gfx1010 gfx1011 gfx1012 gfx1013 \
                gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1103
ifneq (,$(filter $(HIP_ARCH),$(NO_WMMA_ARCHS)))
HIPCCFLAGS += -DCOLI_HIP_NO_WMMA
# The define must hold for every arch in the fat binary (host dispatch is
# shared), so one incapable arch disables the WMMA kernels for all of them.
# Loud, because on a WMMA-capable card that is a real capability loss.
ifneq (,$(filter-out $(NO_WMMA_ARCHS),$(HIP_ARCH)))
$(warning mixed HIP_ARCH list: $(filter $(HIP_ARCH),$(NO_WMMA_ARCHS)) cannot run rocWMMA -> tensor-core kernels disabled for ALL targets ($(HIP_ARCH)). Build separate binaries to keep WMMA on the capable card.)
endif
endif
endif

# METAL=1 adds an opt-in Apple-GPU backend (macOS only). The shader is compiled at
# runtime, so no Xcode / offline metal compiler is required. Default build unchanged.
METAL     ?= 0
METAL_OBJ  =
METALXX    = clang++ -x objective-c++ -std=gnu++17 -fobjc-arc -O3
ifeq ($(METAL),1)
ifeq (,$(DARWIN))
$(error METAL=1 is supported only on macOS)
endif
CFLAGS   += -DCOLI_METAL
LDFLAGS  += -framework Metal -framework Foundation -lc++
METAL_OBJ = backend_metal.o
endif

# VK=1 adds an opt-in Vulkan compute backend (any Vulkan 1.2 driver, e.g. Mesa/RADV)
# for resident expert tensors. Independent of CUDA/HIP: works on GPUs ROCm doesn't
# support (Polaris/RX 580); the VRAM-resident int4 expert-group PRIMITIVE measured ~35%
# faster than the ROCm/HIP one on an RX 9070 (per-call incl. readback — end-to-end tok/s
# depends on what else bounds the box; see docs/vulkan.md, benchmarking notes included).
# Needs libvulkan + glslc (shaderc). Default build unchanged (all colibri.c hooks under
# #ifdef COLI_VULKAN). The .comp shaders compile to .spv, loaded at runtime from
# COLI_VK_SHADERS (default: alongside the binary / ./shaders).
VK        ?= 0
VK_OBJ     =
VK_SPV     =
GLSLC     ?= glslc
ifeq ($(VK),1)
CFLAGS   += -DCOLI_VULKAN
LDFLAGS  += -lvulkan
VK_OBJ    = backend_vulkan.o
VK_SPV    = shaders/qmatmul.spv shaders/qmatmul_gate_up.spv shaders/attention_absorb.spv shaders/rmsnorm.spv
endif

all: colibri$(EXE)

.PHONY: deepseek-v4 deepseek-v4-oracle deepseek-v4-tiny-generate \
	deepseek-v4-tiny-check deepseek-v4-clean
ifeq ($(COLI_V4_SUPPORTED),1)
deepseek-v4:
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4

deepseek-v4-clean:
	$(MAKE) -f Makefile.deepseek-v4 deepseek-v4-clean

deepseek-v4-tiny-generate:
	$(PYTHON) tools/make_deepseek_v4_tiny.py \
		--output ./deepseek_v4_tiny --force

deepseek-v4-tiny-check: deepseek-v4-tiny-generate
	$(MAKE) -f Makefile.deepseek-v4 deepseek-v4-clean
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(if $(strip $(ARCH)),$(ARCH),$(PORTABLE_ARCH)) deepseek-v4
	$(PYTHON) tests/test_deepseek_v4_tiny.py \
		--binary ./$(if $(IS_WIN),deepseek_v4.exe,deepseek_v4) \
		--fixture ./deepseek_v4_tiny
	$(PYTHON) tests/test_deepseek_v4_prefix.py \
		--binary $(CURDIR)/$(if $(IS_WIN),deepseek_v4.exe,deepseek_v4) \
		--fixture $(CURDIR)/deepseek_v4_tiny

deepseek-v4-oracle: deepseek-v4
	@test -n "$(MODEL)" || { echo "usage: make deepseek-v4-oracle MODEL=/path/to/checkpoint" >&2; exit 2; }
	$(PYTHON) tools/make_deepseek_v4_oracle.py --model "$(MODEL)" \
		--binary ./$(if $(IS_WIN),deepseek_v4.exe,deepseek_v4) \
		--output tests/deepseek_v4_oracle.json \
		--memory-gb $(or $(MEMORY_GB),32) \
		--prompt "$(or $(ORACLE_PROMPT),The capital of France is)" \
		--validate \
		--teacher-forcing $(or $(ORACLE_TEACHER_FORCING),32) \
		--greedy $(or $(ORACLE_GREEDY),20)
else
deepseek-v4 deepseek-v4-oracle:
	@echo "$@ is supported only on x86-64/aarch64 Linux and Windows/MSYS2" >&2; exit 1
deepseek-v4-clean:
	@:
deepseek-v4-tiny-check:
	@echo "SKIP deepseek-v4-tiny-check: V4 runtime requires x86-64/aarch64 Linux or Windows/MSYS2"
deepseek-v4-tiny-generate:
	@echo "SKIP deepseek-v4-tiny-generate: V4 runtime requires x86-64/aarch64 Linux or Windows/MSYS2"
endif

# phony targets — 'glm' kept for backward compatibility
colibri: colibri$(EXE)
glm: colibri$(EXE)

# Config stamp: make only tracks file timestamps, not flag changes. Without this,
# `make colibri.exe CUDA_DLL=1` after a prior CPU-only build reports "up to date"
# and silently keeps the CPU-only binary (no CUDA loader) — a build that looks like
# it worked but isn't. We record the build-affecting flags in .build-config and
# rewrite it ONLY when they change (evaluated here at parse time, so the file's
# timestamp moves exactly when the config moves). The binary and CUDA/loader objects depend
# on it, so they relink on a config change and stay put otherwise. (#306)
BUILD_CONFIG     := $(CC)|$(CFLAGS)|$(LDFLAGS)|CUDA=$(CUDA)|CUDA_DLL=$(CUDA_DLL)|ARCH=$(ARCH)|CUDA_ARCH=$(CUDA_ARCH)|METAL=$(METAL)|HIP=$(HIP)|HIP_DLL=$(HIP_DLL)|HIP_ARCH=$(HIP_ARCH)|VK=$(VK)
BUILD_CONFIG_OLD := $(shell cat .build-config 2>/dev/null)
ifneq "$(BUILD_CONFIG)" "$(BUILD_CONFIG_OLD)"
# $(file ...) writes via make's own primitive (GNU Make >= 4.0, 2013), NOT by
# spawning a shell. The earlier `$(shell printf '%s' ... > .build-config)`
# required a POSIX `printf` on PATH — present under MSYS2/Git-Bash but NOT under
# cmd.exe, so `make glm.exe` from the VS Native Tools prompt or with scoop MinGW
# (neither ships sh.exe) failed with "'printf' is not recognized" and left the
# stamp stale. $(file ...) works regardless of the shell make falls back to. (#478)
$(file >.build-config,$(BUILD_CONFIG))
endif
.build-config: ;

colibri$(EXE): colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h quant.h sample.h kv_persist.h telemetry.h route_trace.h omp_tune.h $(CUDA_OBJ) $(METAL_OBJ) $(VK_OBJ) $(VK_SPV) .build-config
	$(CC) $(CFLAGS) colibri.c $(CUDA_OBJ) $(METAL_OBJ) $(VK_OBJ) -o colibri$(EXE) $(LDFLAGS)

# Vulkan backend object (plain C + vulkan headers) and its SPIR-V shaders.
backend_vulkan.o: backend_vulkan.c backend_vulkan.h .build-config
	$(CC) $(CFLAGS) -c backend_vulkan.c -o $@
shaders/%.spv: shaders/%.comp
	@command -v $(GLSLC) >/dev/null 2>&1 || { echo "glslc not found: install shaderc (for VK=1)" >&2; exit 1; }
	$(GLSLC) --target-env=vulkan1.2 $< -o $@

# Windows runtime loader object: resolves coli_cuda_* from coli_cuda.dll.
backend_loader.o: backend_loader.c backend_cuda.h compat.h .build-config
	$(CC) $(CFLAGS) -c backend_loader.c -o $@

# Windows CUDA DLL: compile backend_cuda.cu with nvcc (+MSVC cl.exe as host
# compiler, required by nvcc on Windows) into coli_cuda.dll. Run this from a
# shell that has the MSVC environment set (e.g. after vcvars64.bat, or from a
# "x64 Native Tools Command Prompt"). COLI_CUDA_BUILDING_DLL enables
# __declspec(dllexport) so the 15 API symbols are exported.
cuda-dll: backend_cuda.cu backend_cuda.h
	@command -v "$(NVCC)" >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	@command -v cl >/dev/null 2>&1 || { echo "cl.exe (MSVC) not in PATH — run vcvars64.bat first" >&2; exit 1; }
	"$(NVCC)" $(NVCCFLAGS) -shared -DCOLI_CUDA_BUILDING_DLL \
		-L"$(CUDA_HOME)/lib/x64" -lcudart \
		backend_cuda.cu -o coli_cuda.dll

# Windows HIP DLL: the AMD sibling of cuda-dll above. hipcc compiles the SAME
# backend_cuda.cu (through backend_gpu_compat.h) into coli_hip.dll, exporting
# the same coli_cuda_* surface — which is why COLI_CUDA_BUILDING_DLL, and not a
# new HIP-specific export macro, is the correct switch here. No cl.exe guard
# like cuda-dll's: hipcc brings its own clang host pass, but it still needs the
# MSVC linker and Windows SDK, so run this from a vcvars64 shell.
#
# --hip-path pins the SDK: a machine can carry BOTH a driver-installed HIP SDK
# and a source-built one, and clang injects the driver install via -idirafter on
# its own. Without --hip-path the headers came from one tree while -L/-lamdhip64
# and the device bitcode came from the other — the mismatch that made
# <rocwmma/rocwmma.hpp> unfindable even though the selected SDK ships it.
# --hip-path alone resolves both hip/ and rocwmma/ from the selected root;
# -I"$(HIP_INCLUDE_DIR)" is kept because it takes precedence (verified) and is
# the override point for layouts that split headers off the SDK root.
# --rocm-device-lib-path is kept for determinism: clang can otherwise find the
# bitcode relative to its own resource dir, which is a guess, not a contract.
# Every SDK-derived path is quoted so an install under "C:\Program Files\..."
# survives. No path is hardcoded — see the HIP SDK contract above.
#
#   make hip-dll HIP_DLL=1 HIP_SDK_ROOT=<sdk> HIP_ARCH=gfx1151
hip-dll: backend_cuda.cu backend_cuda.h backend_gpu_compat.h
	@test -x "$(HIPCC)" || command -v "$(HIPCC)" >/dev/null 2>&1 || { echo "hipcc not found at \"$(HIPCC)\": set HIP_SDK_ROOT=<path>, HIP_BIN_DIR=<path> or HIPCC=<path>" >&2; exit 1; }
	@test -d "$(HIP_INCLUDE_DIR)" || { echo "HIP include dir not found: \"$(HIP_INCLUDE_DIR)\" — set HIP_INCLUDE_DIR=<path>" >&2; exit 1; }
	@test -f "$(HIP_INCLUDE_DIR)/hip/hip_runtime.h" || { echo "hip/hip_runtime.h missing under \"$(HIP_INCLUDE_DIR)\" — set HIP_INCLUDE_DIR=<path>" >&2; exit 1; }
	@case "$(HIPCCFLAGS)" in *COLI_HIP_NO_WMMA*) ;; *) test -f "$(HIP_INCLUDE_DIR)/rocwmma/rocwmma.hpp" || { echo "rocwmma/rocwmma.hpp missing under \"$(HIP_INCLUDE_DIR)\": install the rocWMMA component, or set HIP_INCLUDE_DIR=<path>" >&2; exit 1; } ;; esac
	@test -d "$(HIP_LIB_DIR)" || { echo "HIP lib dir not found: \"$(HIP_LIB_DIR)\" — set HIP_LIB_DIR=<path>" >&2; exit 1; }
	@test -f "$(HIP_LIB_DIR)/amdhip64.lib" || { echo "amdhip64.lib missing under \"$(HIP_LIB_DIR)\" — set HIP_LIB_DIR=<path>" >&2; exit 1; }
	@test -d "$(HIP_DEVICE_LIB_PATH)" || { echo "device bitcode dir not found: \"$(HIP_DEVICE_LIB_PATH)\" — set HIP_DEVICE_LIB_PATH=<path>" >&2; exit 1; }
	"$(HIPCC)" $(HIPCCFLAGS) -shared -DCOLI_CUDA_BUILDING_DLL \
		--hip-path="$(HIP_SDK_ROOT)" \
		-I"$(HIP_INCLUDE_DIR)" \
		--rocm-device-lib-path="$(HIP_DEVICE_LIB_PATH)" \
		-L"$(HIP_LIB_DIR)" -lamdhip64 \
		backend_cuda.cu -o coli_hip.dll

backend_cuda_ink.o: backend_cuda_ink.cu backend_cuda_ink.h .build-config
	@command -v "$(NVCC)" >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	"$(NVCC)" $(NVCCFLAGS) -c backend_cuda_ink.cu -o $@

backend_cuda.o: backend_cuda.cu backend_cuda.h backend_gpu_compat.h .build-config
	@command -v "$(GPUCC)" >/dev/null 2>&1 || { echo "$(GPUCC_NAME) not found" >&2; exit 1; }
	"$(GPUCC)" $(GPUFLAGS) -c backend_cuda.cu -o $@

backend_metal.o: backend_metal.mm backend_metal.h
	$(METALXX) -c backend_metal.mm -o $@

metal-test: tests/test_backend_metal.mm backend_metal.mm backend_metal.h
	$(METALXX) tests/test_backend_metal.mm backend_metal.mm -framework Metal -framework Foundation -o backend_metal_test
	./backend_metal_test

# Standalone large-batch GEMM correctness sweep — reproduces the long-context prefill
# corruption in seconds (no model). OMP-parallel CPU reference.
gemm-test: tests/test_gemm_largebatch.mm backend_metal.mm backend_metal.h
	$(METALXX) $(OMPC) tests/test_gemm_largebatch.mm backend_metal.mm $(OMPL) -framework Metal -framework Foundation -o gemm_largebatch_test
	./gemm_largebatch_test

# fmt=6 (E8/IQ3) encoder for tools/convert_fp8_to_int4.py. OPTIONAL: iq3_pack.py
# falls back to its numpy path when the library is absent, producing the same
# bytes ~13x slower. Uses the same ARCH as the engine, so an AVX2 host gets the
# SIMD search and anything else compiles the scalar fallback.
ifneq (,$(DARWIN))
IQ3LIB = tools/libiq3.dylib
else ifneq ($(IS_WIN),)
IQ3LIB = tools/iq3.dll
else
IQ3LIB = tools/libiq3.so
endif
iq3: $(IQ3LIB)
$(IQ3LIB): tools/iq3_encode.c
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@ $(LDFLAGS)

# int4-rans256-g0 codec bridge for tools/repack_rans.py + tools/rans_verify.py.
# OPTIONAL: both tools fall back to a pure-Python codec producing the same
# bytes far more slowly when the library is absent (fine for tests, not for a
# real repack). Same ARCH story as the engine: an AVX-512/NEON host gets the
# batched vector decoder, anything else the scalar paths.
ifneq (,$(DARWIN))
RANSLIB = tools/librans_c.dylib
else ifneq ($(IS_WIN),)
RANSLIB = tools/rans_c.dll
else
RANSLIB = tools/librans_c.so
endif
rans: $(RANSLIB)
$(RANSLIB): tools/rans_ctypes.c rans.h
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@ $(LDFLAGS)

cuda-test: backend_cuda.cu backend_cuda.h backend_gpu_compat.h tests/test_backend_cuda.cu tests/test_ragged_attention.cu tests/test_mxfp4_cuda.cu tests/mxfp4_ref.c
	@command -v "$(GPUCC)" >/dev/null 2>&1 || { echo "$(GPUCC_NAME) not found" >&2; exit 1; }
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE) $(ANS_NVCC_LIBS)
	./backend_cuda_test$(EXE)
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_ragged_attention.cu -o ragged_attention_test$(EXE)
	./ragged_attention_test$(EXE)
	# MXFP4 (fmt=7) against the CPU decoder in quant.h. The reference is built
	# as C on purpose: quant.h uses _Thread_local, which nvcc's C++ front end
	# rejects, and re-implementing it for the test would defeat the comparison.
	$(CC) $(CFLAGS) -c tests/mxfp4_ref.c -o tests/mxfp4_ref.o
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_mxfp4_cuda.cu tests/mxfp4_ref.o -o mxfp4_cuda_test$(EXE)
	./mxfp4_cuda_test$(EXE)

# convenience alias: kernel correctness on AMD (same test, hipcc toolchain)
hip-test:
	$(MAKE) cuda-test HIP=1

# CI: compile the backend and its test binary WITHOUT executing them (for
# runners with the toolchain but no GPU). Pass CUDA_ARCH/HIP=1+HIP_ARCH.
gpu-compile: backend_cuda.o
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE) $(ANS_NVCC_LIBS)
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_ragged_attention.cu -o ragged_attention_test$(EXE)

cuda-bench: backend_cuda.cu backend_cuda.h backend_gpu_compat.h tests/bench_tensor_core.cu
	@command -v "$(GPUCC)" >/dev/null 2>&1 || { echo "$(GPUCC_NAME) not found" >&2; exit 1; }
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/bench_tensor_core.cu -o backend_cuda_bench$(EXE) $(ANS_NVCC_LIBS)
	./backend_cuda_bench$(EXE)

# NOCUDA_*: olmoe.c has no COLI_CUDA code at all, so CUDA=1 would otherwise
# hand it a define matching nothing and link a runtime it never calls -- a
# build whose compile line, libraries and exit status all claim "CUDA" while
# the GPU sits idle. Same guard #783 put on kimi_k3, which since gaining an
# MXFP4 expert path no longer needs it. tests/test_makefile_cuda_scope.py
# asserts this shape.
olmoe$(EXE): olmoe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h
	$(CC) $(NOCUDA_CFLAGS) olmoe.c -o olmoe$(EXE) $(NOCUDA_LDFLAGS)

inkling$(EXE): inkling.c st.h json.h compat.h $(INK_CUDA_OBJ) $(METAL_OBJ)
	$(CC) $(CFLAGS) inkling.c $(INK_CUDA_OBJ) $(METAL_OBJ) -o inkling$(EXE) $(LDFLAGS)

# ENGINES WITHOUT A CUDA BACKEND (#783).
# kimi_k3.c contains no COLI_CUDA code at all — its GPU path is Vulkan. But
# CUDA=1 appends -DCOLI_CUDA to the global CFLAGS and the cudart libs to
# LDFLAGS, so `make kimi_k3 CUDA=1` used to compile a define that matches
# nothing, link a runtime it never calls, and emit no warning: every
# observable signal said "CUDA build" and the GPUs stayed idle. Strip the
# flags for these targets and say so once, loudly.
comma := ,
NOCUDA_CFLAGS  = $(filter-out -DCOLI_CUDA -DCOLI_ANS,$(CFLAGS))
NOCUDA_LDFLAGS = $(filter-out -lcudart -lstdc++ -lcuda -L$(CUDA_HOME)/lib64 \
                   -Wl$(comma)-rpath$(comma)$(CUDA_HOME)/lib64,$(LDFLAGS))

kimi_k3$(EXE): kimi_k3.c st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h omp_tune.h route_trace.h backend_cuda.h $(CUDA_OBJ) $(VK_OBJ) $(VK_SPV)
	$(CC) $(CFLAGS) kimi_k3.c $(CUDA_OBJ) $(VK_OBJ) -o kimi_k3$(EXE) $(LDFLAGS)

# Use a baseline that matches the compiler target. macOS already targets a
# portable baseline when ARCH is empty; forcing the x86 value there breaks
# Apple Silicon. Unknown targets use native rather than an invalid x86 flag.
# Intel Macs need -march for vector instructions
ifneq (,$(DARWIN))
PORTABLE_ARCH = $(if $(X86_64),x86-64-v3,)
else ifneq (,$(AARCH64))
PORTABLE_ARCH = armv8-a
else ifneq (,$(PPC64))
PORTABLE_ARCH = power8
else ifneq (,$(X86_64))
PORTABLE_ARCH = x86-64-v3
else
PORTABLE_ARCH = native
endif

portable:
	$(MAKE) colibri$(EXE) ARCH=$(PORTABLE_ARCH)

iobench$(EXE): iobench.c compat.h
	$(CC) $(CFLAGS) iobench.c -o iobench$(EXE) $(LDFLAGS)

tests/test_serve_sentinel$(EXE): tests/test_serve_sentinel.c compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_ue8m0$(EXE): tests/test_ue8m0.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tok_o200k$(EXE): tests/test_tok_o200k.c tok.h tok_unicode.h tok_unicode_o200k.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_k3_ram_budget$(EXE): tests/test_k3_ram_budget.c kimi_k3.c st.h tok.h quant.h compat.h omp_tune.h route_trace.h kv_prefix.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tok_kimi_tiny$(EXE): tests/test_tok_kimi_tiny.c tok.h tok_unicode.h tok_unicode_o200k.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
tests/test_st_pread$(EXE): tests/test_st_pread.c st.h json.h compat.h
	$(CC) $(CFLAGS) -DST_PREAD_CHUNK=7 $< -o $@ $(LDFLAGS)

tests/test_st_shape$(EXE): tests/test_st_shape.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st$(EXE): tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st_mirror$(EXE): tests/test_st_mirror.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier$(EXE): tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_grammar$(EXE): tests/test_grammar.c grammar.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Standalone: drives a faithful miniature of moe()'s routing+accumulate and links
# the SAME abl.h the engine links -- no model/weights needed (the ablation-logic gate).
tests/test_ablate$(EXE): tests/test_ablate.c abl.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_schema_gbnf$(EXE): tests/test_schema_gbnf.c schema_gbnf.h grammar.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_spec_decode_state$(EXE): tests/test_spec_decode_state.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_decode_batch$(EXE): tests/test_decode_batch.c decode_batch.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_idot$(EXE): tests/test_idot.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_i4_grouped$(EXE): tests/test_i4_grouped.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_stops$(EXE): tests/test_stops.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_topp$(EXE): tests/test_topp.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# bench_topp is a microbenchmark (old qsort vs new heap partial-select, #335), NOT a test
# gate -- intentionally absent from TEST_BINS. Build on demand: make tests/bench_topp
tests/bench_topp$(EXE): tests/bench_topp.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_sample_nan$(EXE): tests/test_sample_nan.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_temp_env$(EXE): tests/test_temp_env.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_kv_alloc$(EXE): tests/test_kv_alloc.c colibri.c st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# fmt=6 kernel oracle: needs the generated grid table, and its fixture comes from
# the reference codec (tools/make_e8_fixture.py) — regenerate if the layout moves.
tests/test_e8_kernel$(EXE): tests/test_e8_kernel.c quant.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_rans$(EXE): tests/test_rans.c rans.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Structured-mutation fuzz for the rans record parser/decoder under
# ASan+UBSan — seeded and bounded (a few thousand mutants, well under a
# minute), asserting no sanitizer report and byte identity across every
# compiled decode arm on every accepted record. NOT a test gate (sanitizer
# flags differ from the normal build); build+run on demand:
#   make fuzz-rans
fuzz-rans: tests/fuzz_rans.c rans.h
	$(CC) -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
	  -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
	  tests/fuzz_rans.c -o tests/fuzz_rans -lm
	./tests/fuzz_rans

tests/test_int3$(EXE): tests/test_int3.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_int3_load$(EXE): tests/test_int3_load.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_fp8_passthrough$(EXE): tests/test_fp8_passthrough.c quant.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_fp8_load$(EXE): tests/test_fp8_load.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_qt_addrow$(EXE): tests/test_qt_addrow.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_logit_nan$(EXE): tests/test_logit_nan.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_router_nan$(EXE): tests/test_router_nan.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_i4_acc512$(EXE): tests/test_i4_acc512.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_compat_direct$(EXE): tests/test_compat_direct.c compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_expert_store_ops$(EXE): tests/test_expert_store_ops.c expert_store.h tensor.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_native_quant$(EXE): tests/test_native_quant.c deepseek_v4.c native_quant.h tensor.h quant.h
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_NATIVE_QUANT deepseek_v4.c $< -o $@ $(LDFLAGS)

ifeq ($(COLI_V4_SUPPORTED),1)
include Makefile.deepseek-v4.units
V4_TEST_LINK_OBJS = \
	COLI_V4_UNIT_ST.o \
	COLI_V4_UNIT_CONFIG.o \
	COLI_V4_UNIT_MATH.o \
	COLI_V4_UNIT_SPARSE_ATTENTION.o \
	COLI_V4_UNIT_RESOURCE_PLAN.o \
	COLI_V4_UNIT_PROMPT.o \
	$(addsuffix .o,$(V4_TEST_UNITS)) \
	COLI_V4_UNIT_NATIVE_QUANT.o

tests/test_deepseek_v4$(EXE): tests/test_deepseek_v4.c deepseek_v4.c deepseek_v4.h \
		Makefile.deepseek-v4.units
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4-test-objs \
		COLI_V4_UNIT_ST.o COLI_V4_UNIT_CONFIG.o COLI_V4_UNIT_MATH.o COLI_V4_UNIT_SPARSE_ATTENTION.o \
		COLI_V4_UNIT_RESOURCE_PLAN.o COLI_V4_UNIT_PROMPT.o COLI_V4_UNIT_NATIVE_QUANT.o
	$(CC) $(CFLAGS) $< $(V4_TEST_LINK_OBJS) -o $@ -pthread $(LDFLAGS)

V4_OWN_DIR = build/ownership
V4_OWN_CFLAGS = $(CFLAGS) -DCOLI_V4_TEST_HOOKS
V4_OWNERSHIP_OBJS = \
	$(V4_OWN_DIR)/COLI_V4_UNIT_RUNTIME.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_CONFIG.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_ST.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_NATIVE_QUANT.o

$(V4_OWN_DIR):
	mkdir -p $(V4_OWN_DIR)

$(V4_OWN_DIR)/COLI_V4_UNIT_RUNTIME.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -DCOLI_V4_UNIT_RUNTIME -c deepseek_v4.c -o $@

$(V4_OWN_DIR)/COLI_V4_UNIT_CONFIG.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -DCOLI_V4_UNIT_CONFIG -c deepseek_v4.c -o $@

$(V4_OWN_DIR)/COLI_V4_UNIT_ST.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h st.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -DCOLI_V4_UNIT_ST -c deepseek_v4.c -o $@

$(V4_OWN_DIR)/COLI_V4_UNIT_NATIVE_QUANT.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h quant.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -DCOLI_V4_UNIT_NATIVE_QUANT -c deepseek_v4.c -o $@

tests/test_v4_ownership$(EXE): tests/test_v4_ownership.c $(V4_OWNERSHIP_OBJS)
	$(CC) $(V4_OWN_CFLAGS) $< $(V4_OWNERSHIP_OBJS) -o $@ -pthread $(LDFLAGS)

endif

tests/test_route_trace$(EXE): tests/test_route_trace.c route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_dsa_select$(EXE): tests/test_dsa_select.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_corpus_draft$(EXE): tests/test_corpus_draft.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_cap_precedence$(EXE): tests/test_cap_precedence.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_cap_mixed_width$(EXE): tests/test_cap_mixed_width.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_ssd_probe$(EXE): tests/test_ssd_probe.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# bench_dsa_select is a microbenchmark (old qsort vs new quickselect partial-select, #356),
# NOT a test gate -- intentionally absent from TEST_BINS. Build on demand: make tests/bench_dsa_select
tests/bench_dsa_select$(EXE): tests/bench_dsa_select.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# bench_idot: microbenchmark (single-acc vs independent-acc AVX-VNNI idot), NOT a test gate.
# Build on demand on an AVX-VNNI CPU: make tests/bench_idot ARCH=native
tests/bench_idot$(EXE): tests/bench_idot.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# bench_mla_simd: microbenchmark (scalar vs AVX2/NEON MLA-absorb reductions, #442),
# NOT a test gate. Build on demand: make tests/bench_mla_simd
tests/bench_mla_simd$(EXE): tests/bench_mla_simd.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_uring$(EXE): tests/test_uring.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_pipe_block$(EXE): tests/test_pipe_block.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_pilot_ring$(EXE): tests/test_pilot_ring.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
tests/test_moe_gs_guard$(EXE): tests/test_moe_gs_guard.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_omp_tune$(EXE): tests/test_omp_tune.c omp_tune.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_kvb_notice$(EXE): tests/test_kvb_notice.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	$(PYTHON) tools/run_tests.py $(TEST_BINS)

# test-python builds the rans ctypes bridge first: without it the repack e2e
# test's C-vs-Python byte-identity pin would silently compare Python against
# itself (the test ALSO skips loudly if the library is somehow still absent,
# e.g. when invoked via unittest directly — but a green `make test` must mean
# the pin was actually checked).
test-python: $(RANSLIB)
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test: test-c test-python

# --- Efficiency / regression suite (issue: "test the program for inefficiencies") ---
# The tiny-model assertions live in test_inefficiency.py and run as part of
# test-python (they're discovered by the test_*.py glob). These targets are
# convenience entry points; the opt-in full-model report is NEVER in `make test`.
#
#   make efficiency        tiny-model asserted regression tests (CPU; part of test-python)
#   make efficiency-cuda   the CUDA-path tests (requires a CUDA build — see below)
#   make efficiency-report opt-in full-model 🟢/🔴 diagnostic, never fails CI
#
# CUDA build (Windows): the CUDA tests need a host built with -DCOLI_CUDA plus
# the runtime DLL. Do this FIRST — note CUDA_DLL=1 on BOTH the host and the
# rule below, or `make glm.exe` will rebuild a CPU-only host and overwrite it:
#   make clean && make glm.exe CUDA_DLL=1 && make cuda-dll
# The tests auto-skip with a clear message if the host is CPU-only.
efficiency: test-python
	$(PYTHON) -m unittest tests.test_inefficiency -v

efficiency-cuda:
	$(PYTHON) -m unittest tests.test_inefficiency.TinyCudaEfficiencyTest -v

efficiency-report:
	$(PYTHON) tests/test_efficiency_report.py


# What a parallel region costs before its body runs. Backs the `if(S > 1)`
# clause in the DSA indexer: at decode that region has one iteration, and the
# fork/join is paid anyway. Not a gate -- a measurement, run it when changing
# how the engine parallelises.
#   make -C c bench-omp-grain
# Honours OMP_NUM_THREADS/OMP_WAIT_POLICY, so measure with the settings the
# engine actually runs under (omp_tune.h sizes the team to physical cores).
bench-omp-grain: tests/bench_omp_grain.c
	$(CC) $(CFLAGS) $< -o tests/bench_omp_grain $(LDFLAGS)
	./tests/bench_omp_grain

# --- sanitizers ---------------------------------------------------------
# The whole C test suite under AddressSanitizer + UndefinedBehaviorSanitizer.
# Until this target the only sanitized code in the tree was fuzz-rans, which
# covers rans.h alone -- the engine's 9k lines of pointer arithmetic, mmap/pread
# and worker threads had never been run under ASan at all.
#
# detect_leaks is OFF, deliberately. The engines allocate their tensor index,
# parsed config and weight slabs once and use them until exit; st.h says as much
# ("intentionally leaked ... one-time startup parsing"). LeakSanitizer reports
# every one of those, which would bury the reports that matter -- a
# heap-buffer-overflow or a use-after-free -- under noise nobody intends to fix.
# halt_on_error makes the first UB report fail the run instead of scrolling past.
#
# -O1: ASan needs frame pointers and readable stacks; -O3 costs signal for speed
# that a test run does not need.
ASAN_CFLAGS  = -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
ASAN_LDFLAGS = -fsanitize=address,undefined
ASAN_ENV     = ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

# `make clean` first is not optional: tools/clean.py had to be fixed to remove
# Unix test binaries at all (it globbed only *.exe), and without that step this
# target re-runs binaries built with the NORMAL flags and reports them clean.
test-asan:
	@$(MAKE) clean
	@$(ASAN_ENV) $(MAKE) test-c EXTRA_CFLAGS="$(ASAN_CFLAGS)" EXTRA_LDFLAGS="$(ASAN_LDFLAGS)"
	@echo "test-asan: suite clean under ASan + UBSan"

# Local validation: one portable CPU build and dependency-free tests.
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

install: colibri$(EXE) olmoe$(EXE) $(if $(filter 1,$(COLI_V4_SUPPORTED)),deepseek-v4,)
	$(INSTALL) -d $(DESTDIR)$(BINDIR)
	$(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)
	$(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)/tools
	$(INSTALL) -m 755 coli $(DESTDIR)$(BINDIR)/coli
	$(INSTALL) -m 755 colibri$(EXE) $(DESTDIR)$(LIBEXECDIR)/colibri$(EXE)
	$(INSTALL) -m 755 olmoe$(EXE) $(DESTDIR)$(LIBEXECDIR)/olmoe$(EXE)
	@if [ -f deepseek_v4$(EXE) ]; then \
	  $(INSTALL) -m 755 deepseek_v4$(EXE) $(DESTDIR)$(LIBEXECDIR)/deepseek_v4$(EXE); \
	fi
	$(INSTALL) -m 644 resource_plan.py doctor.py autotune.py openai_server.py version.py $(DESTDIR)$(LIBEXECDIR)/
	$(INSTALL) -m 644 tools/*.py $(DESTDIR)$(LIBEXECDIR)/tools/
	@# The dashboard is an optional build artifact (cd web && npm run build), so install
	@# it only when it exists. It goes NEXT TO openai_server.py, which probes ./web/dist.
	@if [ -f ../web/dist/index.html ]; then \
	  $(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)/web/dist; \
	  cp -R ../web/dist/. $(DESTDIR)$(LIBEXECDIR)/web/dist/; \
	  echo "installed the web dashboard"; \
	else \
	  echo "web/dist absent -- skipping the dashboard (build it with: cd web && npm run build)"; \
	fi

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/coli
	rm -rf $(DESTDIR)$(LIBEXECDIR)

clean:
	$(PYTHON) tools/clean.py

bench: iobench$(EXE)
	@if [ -n "$(ARGS)" ]; then ./iobench$(EXE) $(ARGS); else echo "built iobench$(EXE) — run: ./iobench$(EXE) <file> <MB> <iters> <threads> <direct 0|1>"; fi
.PHONY: all colibri glm deepseek-v4 deepseek-v4-clean deepseek-v4-oracle \
	deepseek-v4-tiny-generate deepseek-v4-tiny-check \
	iq3 rans fuzz-rans cuda-test hip-test gpu-compile cuda-bench cuda-dll hip-dll portable \
	test-c test-python test check clean install uninstall bench

# Own line: see the TEST_RULES note above on shared .PHONY lines.
.PHONY: bench-omp-grain
.PHONY: test-asan

tests/test_kv_prefix$(EXE): tests/test_kv_prefix.c kv_prefix.h
	$(CC) $(CFLAGS) tests/test_kv_prefix.c -o tests/test_kv_prefix$(EXE) $(LDFLAGS)

tests/test_kimi_request_state$(EXE): tests/test_kimi_request_state.c kimi_k3.c kv_prefix.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h omp_tune.h route_trace.h
	$(CC) $(NOCUDA_CFLAGS) tests/test_kimi_request_state.c $(VK_OBJ) -o tests/test_kimi_request_state$(EXE) $(NOCUDA_LDFLAGS)
