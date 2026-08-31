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
# LTO is opt-in (LTO=1), not default: cross-TU inlining can change
# floating-point contraction (FMA formation), and colibri's token-exactness
# oracles are validated against the default build's optimization regime
# (#1044/#1024 showed how FMA + accumulator order changes rounding).
# Enable only after a token-exactness A/B on your own toolchain.
ifeq ($(LTO),1)
CFLAGS  += -flto
LDFLAGS += -flto
endif
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
ifeq ($(LTO),1)
CFLAGS  += -flto
LDFLAGS += -flto
endif
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
ifeq ($(LTO),1)
CFLAGS  += -flto
LDFLAGS += -flto
endif
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
# gcc 11 defines __ARM_FEATURE_DOTPROD for -march=armv8-a+dotprod yet cannot
# emit vdotq_s32 for that base ("target specific option mismatch"): the macro
# probe alone is not enough. Verify by compiling the intrinsic for real, and
# prefer the armv8.2-a base where dotprod is a first-class ISA feature
# (Neoverse-N1 and friends report asimddp but are armv8.2 cores). Fall back to
# armv8-a only if the compiler accepts the modifiers there too; otherwise the
# idot kernels compile out and int4 S=1 decode falls back to f32 (#631, #1104).
PROBE_C := $(shell mktemp /tmp/coli_v4_probe.XXXXXX.c)
PROBE_OK := $(shell echo '#include <arm_neon.h>' > $(PROBE_C); echo 'int p(const signed char*w,const signed char*x,int n){int32x4_t a=vdupq_n_s32(0);for(;n>=16;n-=16)a=vdotq_s32(a,vld1q_s8(w),vld1q_s8(x));return vaddvq_s32(a);}' >> $(PROBE_C); $(CC) -c $(PROBE_C) -o /dev/null -march=armv8.2-a$(ARCH_EXT) 2>/dev/null && echo yes; rm -f $(PROBE_C))
ifeq ($(PROBE_OK),yes)
ARCHFLAG := -march=armv8.2-a$(ARCH_EXT)
else
PROBE_C8 := $(shell mktemp /tmp/coli_v4_probe.XXXXXX.c)
PROBE_OK8 := $(shell echo '#include <arm_neon.h>' > $(PROBE_C8); echo 'int p(const signed char*w,const signed char*x,int n){int32x4_t a=vdupq_n_s32(0);for(;n>=16;n-=16)a=vdotq_s32(a,vld1q_s8(w),vld1q_s8(x));return vaddvq_s32(a);}' >> $(PROBE_C8); $(CC) -c $(PROBE_C8) -o /dev/null -march=armv8-a$(ARCH_EXT) 2>/dev/null && echo yes; rm -f $(PROBE_C8))
ifeq ($(PROBE_OK8),yes)
ARCHFLAG := -march=armv8-a$(ARCH_EXT)
else
$(warning $(CC) does not accept$(subst +, +,$(ARCH_EXT)): int8/int4 idot kernels compile out (#631))
endif
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
ifeq ($(LTO),1)
CFLAGS  += -flto
LDFLAGS += -flto
endif
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
ifeq ($(LTO),1)
CFLAGS  += -flto
LDFLAGS += -flto
endif
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
DEEPGEMM  ?= 0
NCCL      ?= 0
# DeepGEMM sm120 headers for the DeepSeek V4 DeepGEMM flavour: a pinned
# checkout made by tools/fetch_deepgemm.sh on first use (gitignored), see the
# DeepSeek V4 CUDA kernels block and the cuda-dsv4-dg-dll target below.
DEEPGEMM_HOME ?= third_party/deepgemm
DEEPGEMM_PIN ?= 39fb4447a062b418fd08ce17cd308adb28559417
DEEPGEMM_STAMP = $(DEEPGEMM_HOME)/deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh
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
# sm_121 (GB10 Spark) needs CUDA >= 12.9, so its gencode joins only when the
# detected toolkit supports it — the portable floor stays 12.8. The version
# probe fails closed: no or unparsable nvcc means no compute_121 (the
# compute_120 PTX still JITs on sm_121).
CUDA_ARCH ?= native
ifeq ($(CUDA_ARCH),portable)
CUDA_GENCODE = -gencode arch=compute_80,code=sm_80 \
               -gencode arch=compute_86,code=sm_86 \
               -gencode arch=compute_89,code=sm_89 \
               -gencode arch=compute_90,code=sm_90 \
               -gencode arch=compute_120,code=sm_120 \
               -gencode arch=compute_120,code=compute_120
NVCC_RELEASE := $(shell "$(NVCC)" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')
# The empty-check makes the fail-closed promise real: with no parsable
# version the sort below would compare 12.9 against itself and pass.
ifneq ($(strip $(NVCC_RELEASE)),)
ifeq ($(shell printf '%s\n' 12.9 $(NVCC_RELEASE) | sort -V 2>/dev/null | head -1),12.9)
CUDA_GENCODE += -gencode arch=compute_121,code=sm_121
endif
endif
else
CUDA_GENCODE = -arch=$(CUDA_ARCH)
endif
# NVCC_STD is a variable so an optional backend can raise the standard without
# rewriting the whole flag line: DEEPGEMM=1 needs C++20, and replacing
# NVCCFLAGS wholesale to get it would also discard $(CUDA_GENCODE) and any
# -ccbin the user set above.
NVCC_STD ?= c++17
ifneq ($(IS_WIN),)
# nvcc's host compiler on Windows is MSVC cl.exe: the GCC-style
# -Xcompiler=-Wall,-Wextra is rejected ("D8021 invalid numeric argument
# '/Wextra'"), which made `make cuda-dll` unbuildable as shipped. -W3 is
# the MSVC warning level (dash form, NOT /W3: MSYS make would mangle the
# slash-form into a filesystem path).
NVCCFLAGS ?= -O3 -std=$(NVCC_STD) -ftz=false $(CUDA_GENCODE) -Xcompiler=-W3
else
# -ftz=false is nvcc's default, pinned explicitly: the fmt=8 kernels are
# cross-tier parity instruments and a flushed scale*subnormal contribution
# silently diverges from the CPU reference (backend_cuda.cu also #errors on
# fast-math builds, but -ftz alone defines no macro to guard on).
NVCCFLAGS ?= -O3 -std=$(NVCC_STD) -ftz=false $(CUDA_GENCODE) -Xcompiler=-Wall,-Wextra
# glibc >= 2.41 declares the C23 math additions (rsqrt/rsqrtf, sinpi/cospi, ...)
# with __THROW -> noexcept(true), clashing with CUDA's crt/math_functions.h
# under C++17 (noexcept is part of the function type since C++17). Force-include
# a compat header that strips __THROW from the host pass; see
# glibc_c23_math_compat.h. MSVC never sees glibc headers, so Linux-only.
ifneq ($(LINUX),)
NVCCFLAGS += -Xcompiler=-include,$(CURDIR)/glibc_c23_math_compat.h
endif
endif
# CUDA 13.1 host_config.h accepts only Visual Studio 2019-2022 and refuses
# anything newer outright, so on a box with VS Build Tools 18 (MSVC 19.50)
# EVERY nvcc target fails at the first #include with C1189 -- cuda-test,
# cuda-dll, all of it. nvcc offers -allow-unsupported-compiler to override the
# check, but there was no way to get one flag in: NVCCFLAGS is the only handle
# and replacing it wholesale discards $(CUDA_GENCODE), the -ccbin set above and
# the -Xcompiler warning form that Windows specifically needs -- the same
# argument that made NVCC_STD a variable rather than a rewrite.
#
# Defaults to 0: nvcc's own wording is "may cause compilation failure or
# incorrect run time execution", so an unsupported host compiler is the
# builder's decision to make explicitly, not a default this Makefile makes for
# them. Any result produced with it set should say so.
NVCC_ALLOW_UNSUPPORTED ?= 0
ifeq ($(NVCC_ALLOW_UNSUPPORTED),1)
NVCCFLAGS += -allow-unsupported-compiler
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
# Linux/Windows, aarch64 Linux, and arm64 macOS. SIMD fast paths are gated
# per-feature inside the sources, so a plain armv8-a build stays correct.
# macOS portability rides compat.h (posix_fadvise -> F_RDADVISE, O_DIRECT ->
# F_NOCACHE via st.h's direct twins) plus a mach branch for available memory;
# the token-exact tiny fixture is the acceptance gate on the macos runner.
COLI_V4_SUPPORTED :=
ifneq (,$(X86_64))
ifneq (,$(IS_WIN)$(LINUX))
COLI_V4_SUPPORTED := 1
endif
endif
ifneq (,$(AARCH64))
ifneq (,$(LINUX)$(DARWIN))
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
TEST_EXCLUDE = test_uring test_deepseek_v4 test_v4_ownership test_v4_serve_framing \
	test_segment_adapters_registration test_segment_adapters_real \
	test_edge_adapters_registration test_edge_adapters_real
TEST_BINS    = $(addprefix tests/,$(addsuffix $(EXE),$(filter-out $(TEST_EXCLUDE),$(TEST_RULES))))
ifneq (,$(LINUX))
TEST_BINS += tests/test_uring$(EXE)
endif
ifeq ($(COLI_V4_SUPPORTED),1)
TEST_BINS += tests/test_v4_hybrid_policy$(EXE) tests/test_k3_fill_budget$(EXE) tests/test_v4_bank_pair$(EXE)
TEST_BINS += tests/test_deepseek_v4$(EXE) tests/test_v4_ownership$(EXE) \
	tests/test_v4_serve_framing$(EXE) \
	tests/test_segment_adapters_registration$(EXE) \
	tests/test_edge_adapters_registration$(EXE)
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
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lcublasLt -lcublas -lstdc++
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

# --- DeepSeek V4 CUDA kernels -------------------------------------------
# The GPU tier for the DeepSeek V4 engine in Makefile.deepseek-v4. These
# objects are self-contained: backend_cuda_dsv4.* include only their own header
# and the CUDA runtime, so they build and their tests run without the engine.
#
# DEEPGEMM is an opt-in: its kernels require Blackwell (sm_120a) to run and CI
# does not compile it. It gates nothing and blocks nothing; the default build
# (generic kernels, any sm_80+) is the supported configuration and the one to
# report bugs against. The sm120 headers it needs are NOT in this repository:
# tools/fetch_deepgemm.sh checks out the community sm120 port at a pinned
# commit into third_party/deepgemm (gitignored), verifies the pin, applies
# patches-deepgemm-sm120-msvc.patch, and DEEPGEMM=1 / cuda-dsv4-dg-dll invoke
# it on first build (deepgemm-fetch). DEEPGEMM_HOME points at an existing
# checkout instead; DEEPGEMM_PIN bumps the commit. Attribution:
# THIRD_PARTY_NOTICES.md.
DSV4_CUDA_OBJ = backend_cuda_dsv4.o
ifeq ($(DEEPGEMM),1)
DSV4_DEEPGEMM_DEP = $(DEEPGEMM_STAMP)
CFLAGS += -DCOLI_DSV4_DEEPGEMM
# Append, never replace. This was `:=` and discarded the whole inherited flag
# set: $(CUDA_GENCODE), so any CUDA_ARCH the user chose was silently swapped for
# a hardcoded sm_120f -- making DEEPGEMM=1 a Blackwell-only path by accident
# rather than by design -- and the -ccbin above, which is how a host whose
# default g++ is too new for CUDA names a supported host compiler. DeepGEMM
# needs C++20 and the relaxed-constexpr/extended-lambda pair, not its own arch.
override NVCC_STD  := c++20
override NVCCFLAGS += --expt-relaxed-constexpr --expt-extended-lambda -DCOLI_DSV4_DEEPGEMM -I"$(DEEPGEMM_HOME)" -I"$(DEEPGEMM_HOME)/deep_gemm/include" -I"$(DEEPGEMM_HOME)/cutlass/include" -I"$(DEEPGEMM_HOME)/third-party/cutlass/include"
LDFLAGS += -lcuda
endif
ifeq ($(NCCL),1)
CFLAGS += -DCOLI_DSV4_NCCL
override NVCCFLAGS += -DCOLI_DSV4_NCCL
LDFLAGS += -lnccl
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

# Tiny Kimi K3 fixture + vendor oracle.  The generator is numpy-only and
# deterministic; ref.json is committed and comes from Moonshot's own modeling
# code (tools/make_kimi_k3_ref.py, maintainer-run, SHA-256-pinned), so CI
# needs neither torch nor network.  kimi_k3 builds on every platform, so
# unlike the V4 targets this needs no support gate.
.PHONY: kimi-k3-tiny-generate kimi-k3-tiny-check
kimi-k3-tiny-generate:
	$(PYTHON) tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force

kimi-k3-tiny-check: kimi-k3-tiny-generate kimi_k3$(EXE)
	$(PYTHON) tests/test_kimi_k3_tiny.py \
		--binary ./kimi_k3$(EXE) \
		--fixture ./kimi_k3_tiny
	$(PYTHON) tests/test_kimi_k3_ckpt.py \
		--binary ./kimi_k3$(EXE) \
		--fixture ./kimi_k3_tiny

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

# Convenience aliases — 'glm' kept for backward compatibility.
# On POSIX EXE is empty, so `colibri: colibri$(EXE)` becomes a self-dependency
# and marking it phony also forces the real binary to rebuild on every invocation.
# Only create the alias when the platform executable suffix makes it distinct.
ifneq ($(EXE),)
.PHONY: colibri
colibri: colibri$(EXE)
endif
.PHONY: glm
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

colibri$(EXE): colibri.c cli_args.h st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h quant.h sample.h kv_persist.h telemetry.h route_trace.h omp_tune.h kv_fp8.h kv_tq.h $(CUDA_OBJ) $(METAL_OBJ) $(VK_OBJ) $(VK_SPV) .build-config
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


backend_cuda_dsv4.o: backend_cuda_dsv4.cu backend_cuda_dsv4.h .build-config $(DSV4_DEEPGEMM_DEP)
	@command -v "$(NVCC)" >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	"$(NVCC)" $(NVCCFLAGS) -c backend_cuda_dsv4.cu -o $@

# The pinned DeepGEMM sm120 checkout (see the DEEPGEMM block above). The stamp
# is a header that only exists in the sm120 port, so an empty or wrong tree
# refetches; the script itself is a no-op when the tree is already at the pin.
deepgemm-fetch $(DEEPGEMM_STAMP):
	DEEPGEMM_HOME="$(DEEPGEMM_HOME)" tools/fetch_deepgemm.sh "$(DEEPGEMM_PIN)"
.PHONY: deepgemm-fetch

# Windows DeepSeek V4 CUDA DLLs: the sibling of cuda-dll for the DeepSeek V4
# engine. deepseek-v4.exe is built with MinGW gcc and cannot link MSVC/nvcc
# objects, so backend_cuda_dsv4.cu compiles into its own DLL (nvcc + MSVC
# cl.exe as host) whose symbols are exported via dsv4.def; the engine resolves
# them at runtime through backend_loader_dsv4.c, mirroring the
# backend_loader.c / coli_cuda.dll split. Run from a vcvars64 shell, like
# cuda-dll. Two flavours ship side by side and the loader picks at runtime by
# compute capability: coli_cuda_dsv4.dll is the GENERIC build (plain CUDA
# kernels, any sm_80+ card; CUDA_ARCH=portable for a fat binary) and
# coli_cuda_dsv4_dg.dll the DeepGEMM sm_120a build (tensor-core prefill paths;
# needs the sm120 DeepGEMM checkout at DEEPGEMM_HOME, see
# patches-deepgemm-sm120-msvc.patch). Both need -lcuda for the TMA driver API.
cuda-dsv4-dll: backend_cuda_dsv4.cu backend_cuda_dsv4.h dsv4.def
	@command -v "$(NVCC)" >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	@command -v cl >/dev/null 2>&1 || { echo "cl.exe (MSVC) not in PATH — run vcvars64.bat first" >&2; exit 1; }
	"$(NVCC)" $(NVCCFLAGS) --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=/Zc:preprocessor -shared \
		-Xlinker /DEF:dsv4.def \
		-L"$(CUDA_HOME)/lib/x64" -lcublasLt -lcudart -lcuda \
		backend_cuda_dsv4.cu -o coli_cuda_dsv4.dll

# The DeepGEMM sm120 headers come from tools/fetch_deepgemm.sh (pinned
# checkout into $(DEEPGEMM_HOME), see the DEEPGEMM block above); this target
# fetches them on first use.
DEEPGEMM_INC = -I"$(DEEPGEMM_HOME)/deep_gemm/include" -I"$(DEEPGEMM_HOME)/cutlass/include" -I"$(DEEPGEMM_HOME)/third-party/cutlass/include"
cuda-dsv4-dg-dll: backend_cuda_dsv4.cu backend_cuda_dsv4.h dsv4.def $(DEEPGEMM_STAMP)
	@command -v "$(NVCC)" >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	@command -v cl >/dev/null 2>&1 || { echo "cl.exe (MSVC) not in PATH — run vcvars64.bat first" >&2; exit 1; }
	"$(NVCC)" -O3 -std=c++20 -ftz=false --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=/Zc:preprocessor \
		-gencode arch=compute_120a,code=sm_120a -DCOLI_DSV4_DEEPGEMM \
		$(DEEPGEMM_INC) -shared \
		-Xlinker /DEF:dsv4.def \
		-L"$(CUDA_HOME)/lib/x64" -lcublasLt -lcudart -lcuda \
		backend_cuda_dsv4.cu -o coli_cuda_dsv4_dg.dll

# Windows: the decode-path kernels validated through the runtime DLL loader.
# deepseek-v4.exe is MinGW gcc; backend_loader_dsv4.c resolves the MSVC/nvcc
# built coli_cuda_dsv4*.dll and this test proves that ABI boundary.
dsv4-cuda-loader-test: tests/test_dsv4_decode_cuda.c backend_loader_dsv4.c backend_cuda_dsv4.h
	$(CC) $(CFLAGS) tests/test_dsv4_decode_cuda.c backend_loader_dsv4.c -o dsv4_decode_loader_test$(EXE)
	./dsv4_decode_loader_test$(EXE)
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

cuda-test: backend_cuda.cu backend_cuda.h backend_gpu_compat.h tests/test_backend_cuda.cu tests/test_ragged_attention.cu tests/test_absorb_determinism.cu tests/test_mxfp4_cuda.cu tests/mxfp4_ref.c tests/test_fp8_warp_cuda.cu tests/test_fp8_cuda.cu tests/test_weights_owned_cuda.cu tests/test_cuda_fmt_trap_cuda.cu
	@command -v "$(GPUCC)" >/dev/null 2>&1 || { echo "$(GPUCC_NAME) not found" >&2; exit 1; }
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE) $(ANS_NVCC_LIBS)
	./backend_cuda_test$(EXE)
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_ragged_attention.cu -o ragged_attention_test$(EXE)
	./ragged_attention_test$(EXE)
	# weight_at's device-side refusal (the __trap()) on real silicon. The
	# host-side half is tests/test_cuda_fmt_guard.c, which runs in the CPU
	# suite and cannot execute device code. Includes backend_cuda.cu itself:
	# weight_at is file-static, and the trap is unreachable through the public
	# API by design. Re-execs itself per probe (the trap poisons the context).
	"$(GPUCC)" $(GPUFLAGS) tests/test_cuda_fmt_trap_cuda.cu -o cuda_fmt_trap_test$(EXE) $(ANS_NVCC_LIBS)
	./cuda_fmt_trap_test$(EXE)
	# fmt=8 warp kernels (COLI_CUDA_F8_WARP): decode sweep, census + tail
	# shapes, S-invariance, NaN/denormal policy. Includes backend_cuda.cu
	# itself (kernel-level oracle), so only the test file is compiled.
	# Runs BEFORE the MXFP4 test: that one has a pre-existing exponent-255
	# failure on some hosts and would otherwise abort the recipe first.
	"$(GPUCC)" $(GPUFLAGS) tests/test_fp8_warp_cuda.cu -o fp8_warp_test$(EXE) $(ANS_NVCC_LIBS)
	./fp8_warp_test$(EXE)
	# The batch and ragged absorb kernels must reproduce their own output: they
	# reduce in a fixed tree order with no atomics, so two identical launches
	# that disagree have read shared memory another thread was still writing.
	# Needs both an oversubscribed grid and a token count in the affected band
	# (roughly 33..256 per block) to reach the output; the sweep covers both,
	# and one cell sits above the band on purpose. See the test's header.
	# Runs AFTER the fp8 test on purpose: this is the most schedule-sensitive
	# test in the recipe, and it should not cost the broader kernel suites if
	# it trips. It stays BEFORE MXFP4 so it still runs on the hosts where that
	# one is expected to fail.
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_absorb_determinism.cu -o absorb_determinism_test$(EXE)
	./absorb_determinism_test$(EXE)
	# fmt=8 (fp8-e4m3) kernel oracle + public API (LUT gate, dense, expert
	# group, byte accounting). Includes backend_cuda.cu itself, so only the
	# test file is compiled. Also runs BEFORE the MXFP4 test (same reason).
	"$(GPUCC)" $(GPUFLAGS) tests/test_fp8_cuda.cu -o fp8_cuda_test$(EXE) $(ANS_NVCC_LIBS)
	./fp8_cuda_test$(EXE)
	# Upload-failure ownership: a failed H2D weight copy must release the
	# bytes it allocated (weights_owned set by the malloc, not the copy).
	# Includes backend_cuda.cu itself to plant the memcpy failure, so only
	# the test file is compiled.
	"$(GPUCC)" $(GPUFLAGS) tests/test_weights_owned_cuda.cu -o weights_owned_test$(EXE) $(ANS_NVCC_LIBS)
	./weights_owned_test$(EXE)
	# MXFP4 (fmt=7) against the CPU decoder in quant.h. The reference is built
	# as C on purpose: quant.h uses _Thread_local, which nvcc's C++ front end
	# rejects, and re-implementing it for the test would defeat the comparison.
	# Built WITHOUT OpenMP (#971). This is a correctness oracle over matrices of
	# at most S=4 I=2048 O=64, so threading it buys nothing measurable, while the
	# dependency it creates has blocked this target on two hosts: a Linux box
	# whose libgomp is absent (undefined reference to GOMP_parallel, at this very
	# link step) and MSVC, whose OpenMP predates C99 loop declarations in C mode.
	# It was also the only reason the final nvcc/hipcc link here had to drag in
	# the host OpenMP runtime, so GPU_OMP_LINK goes with it.
	# The OpenMP flags are FILTERED OUT rather than overridden with -fno-openmp
	# alone: on macOS they arrive as `-Xclang -fopenmp`, which goes straight to
	# the frontend and is not necessarily undone by a driver-level -fno-openmp.
	# Getting that wrong would leave OpenMP symbols in the object while the link
	# below no longer pulls the runtime, and CI could not catch it because CI
	# never runs cuda-test. -Xclang appears nowhere else in CFLAGS.
	$(CC) $(filter-out -Xclang -fopenmp,$(CFLAGS)) -fno-openmp -c tests/mxfp4_ref.c -o tests/mxfp4_ref.o
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_mxfp4_cuda.cu tests/mxfp4_ref.o -o mxfp4_cuda_test$(EXE)
	./mxfp4_cuda_test$(EXE)


# The DeepSeek V4 kernels on their own: dense, batched attention and routed MoE
# against the CPU reference in each test. Needs a GPU, so CI compiles it via
# gpu-compile rather than running it.
dsv4-cuda-test: tests/test_dsv4_dense_batch_cuda.c tests/test_dsv4_attention_batch_cuda.c tests/test_dsv4_moe_batch_cuda.c backend_cuda_dsv4.h $(DSV4_CUDA_OBJ)
	$(CC) $(CFLAGS) tests/test_dsv4_dense_batch_cuda.c $(DSV4_CUDA_OBJ) -o dsv4_cuda_test$(EXE) $(LDFLAGS)
	./dsv4_cuda_test$(EXE)
	$(CC) $(CFLAGS) tests/test_dsv4_attention_batch_cuda.c $(DSV4_CUDA_OBJ) -o dsv4_attention_cuda_test$(EXE) $(LDFLAGS)
	./dsv4_attention_cuda_test$(EXE)
	$(CC) $(CFLAGS) tests/test_dsv4_moe_batch_cuda.c $(DSV4_CUDA_OBJ) -o dsv4_moe_cuda_test$(EXE) $(LDFLAGS)
	./dsv4_moe_cuda_test$(EXE)

# Pure-CPU unit tests for the two headers the kernels consume. No GPU, no model.
tests/test_dsv4_mhc$(EXE): tests/test_dsv4_mhc.c dsv4_mhc.h
	$(CC) $(CFLAGS) tests/test_dsv4_mhc.c -o tests/test_dsv4_mhc$(EXE) $(LDFLAGS)

tests/test_dsv4_quant$(EXE): tests/test_dsv4_quant.c dsv4_quant.h
	$(CC) $(CFLAGS) tests/test_dsv4_quant.c -o tests/test_dsv4_quant$(EXE) $(LDFLAGS)
# convenience alias: kernel correctness on AMD (same test, hipcc toolchain)
hip-test:
	$(MAKE) cuda-test HIP=1

# CI: compile the backend and its test binary WITHOUT executing them (for
# runners with the toolchain but no GPU). Pass CUDA_ARCH/HIP=1+HIP_ARCH.
gpu-compile: backend_cuda.o
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE) $(ANS_NVCC_LIBS)
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_ragged_attention.cu -o ragged_attention_test$(EXE)
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/test_absorb_determinism.cu -o absorb_determinism_test$(EXE)
	"$(GPUCC)" $(GPUFLAGS) tests/test_fp8_warp_cuda.cu -o fp8_warp_test$(EXE) $(ANS_NVCC_LIBS)
	"$(GPUCC)" $(GPUFLAGS) tests/test_weights_owned_cuda.cu -o weights_owned_test$(EXE) $(ANS_NVCC_LIBS)
	"$(GPUCC)" $(GPUFLAGS) tests/test_cuda_fmt_trap_cuda.cu -o cuda_fmt_trap_test$(EXE) $(ANS_NVCC_LIBS)

cuda-bench: backend_cuda.cu backend_cuda.h backend_gpu_compat.h tests/bench_tensor_core.cu
	@command -v "$(GPUCC)" >/dev/null 2>&1 || { echo "$(GPUCC_NAME) not found" >&2; exit 1; }
	"$(GPUCC)" $(GPUFLAGS) backend_cuda.cu tests/bench_tensor_core.cu -o backend_cuda_bench$(EXE) $(ANS_NVCC_LIBS)
	./backend_cuda_bench$(EXE)

# fmt=8 kernel bench: old vs COLI_CUDA_F8_WARP kernels per decode path, census
# expert shapes, JSON on stdout (kernel time only). The build is a separate
# file target so tools/run_f8_bench.sh can keep stdout pure JSON.
fp8_bench$(EXE): backend_cuda.cu backend_cuda.h backend_gpu_compat.h tests/bench_fp8_cuda.cu
	@command -v "$(GPUCC)" >/dev/null 2>&1 || { echo "$(GPUCC_NAME) not found" >&2; exit 1; }
	"$(GPUCC)" $(GPUFLAGS) tests/bench_fp8_cuda.cu -o fp8_bench$(EXE) $(ANS_NVCC_LIBS)
fp8-bench: fp8_bench$(EXE)
	./fp8_bench$(EXE)

# NOCUDA_*: olmoe.c has no COLI_CUDA code at all, so CUDA=1 would otherwise
# hand it a define matching nothing and link a runtime it never calls -- a
# build whose compile line, libraries and exit status all claim "CUDA" while
# the GPU sits idle. Same guard #783 put on kimi_k3, which since gaining an
# MXFP4 expert path no longer needs it. tests/test_makefile_cuda_scope.py
# asserts this shape.
olmoe$(EXE): olmoe.c cli_args.h st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h serve_codec.h
	$(CC) $(NOCUDA_CFLAGS) olmoe.c -o olmoe$(EXE) $(NOCUDA_LDFLAGS)

# Qwen3.6-35B-A3B engine (hybrid Gated Attention + Gated DeltaNet + streaming
# MoE). With CUDA=1 the optional VRAM expert tier (qwen36_tier.c) is compiled
# in and reuses the shared CUDA backend; without it the engine is CPU-only
# (qwen36_tier.h provides inline stubs).
#
# The flags follow the same switch. Without CUDA=1 this target keeps the
# NOCUDA_* flags it got when the engine had no CUDA at all: compiling
# -DCOLI_CUDA and linking -lcudart for a build that calls neither only made
# `make qwen36` depend on a toolkit it never touches. With CUDA=1 the tier is
# real CUDA and takes the normal flags back.
ifeq ($(CUDA),1)
QWEN36_TIER_SRC = qwen36_tier.c
QWEN36_CFLAGS   = $(CFLAGS)
QWEN36_LDFLAGS  = $(LDFLAGS)
else
QWEN36_TIER_SRC =
QWEN36_CFLAGS   = $(NOCUDA_CFLAGS)
QWEN36_LDFLAGS  = $(NOCUDA_LDFLAGS)
endif
qwen36$(EXE): qwen36.c cli_args.h qwen36_tier.h st.h json.h compat.h $(QWEN36_TIER_SRC) $(CUDA_OBJ)
	$(CC) $(QWEN36_CFLAGS) qwen36.c $(QWEN36_TIER_SRC) $(CUDA_OBJ) -o qwen36$(EXE) $(QWEN36_LDFLAGS)

# Qwen3.8-Flash-Next text-only sibling. This target is intentionally CPU-only;
# qwen38.c owns the direct checkpoint loader and SERVE=1 protocol, with no
# CUDA/Metal tier advertised by the control plane.
qwen38$(EXE): qwen38.c cli_args.h qwen38_core.h qwen38_nfc.h qwen38_nfc_tables.h st.h json.h compat.h quant.h route_trace.h tok.h tok_unicode.h tok_unicode_o200k.h serve_codec.h
	$(CC) $(NOCUDA_CFLAGS) qwen38.c -o qwen38$(EXE) $(NOCUDA_LDFLAGS)

.PHONY: qwen38-tiny-generate qwen38-tiny-check
qwen38-tiny-generate:
	$(PYTHON) tools/make_qwen38_tiny.py --out ./qwen38_tiny

# Il prefetch PLE e' un riordino, non un cambio di calcolo: le stesse righe,
# lette prima e in parallelo. Quindi acceso e spento devono dare gli STESSI
# token, e questo lo pretende invece di fidarsi. Se un giorno divergono, il
# prefetch ha smesso di essere un riordino e va guardato, non spento.
.PHONY: qwen38-ple-prefetch-check
qwen38-ple-prefetch-check: qwen38-tiny-generate qwen38$(EXE)
	@fail=0; for cap in 1 2 4 8; do for pb in 0 1; do for bf in 0 1; do \
		on=$$(Q38_PREFILL_BATCH=$$pb Q38_NATIVE_BF16=$$bf OMP_NUM_THREADS=2 SNAP=./qwen38_tiny ./qwen38$(EXE) $$cap 8 ./qwen38_tiny/ref.json 2>/dev/null | grep '^C engine'); \
		off=$$(Q38_PLE_PREFETCH=0 Q38_PREFILL_BATCH=$$pb Q38_NATIVE_BF16=$$bf OMP_NUM_THREADS=2 SNAP=./qwen38_tiny ./qwen38$(EXE) $$cap 8 ./qwen38_tiny/ref.json 2>/dev/null | grep '^C engine'); \
		if [ -z "$$on" ]; then echo "FAIL cap=$$cap batch=$$pb bf16=$$bf: no output"; fail=1; \
		elif [ "$$on" != "$$off" ]; then echo "FAIL cap=$$cap batch=$$pb bf16=$$bf: prefetch changes the tokens"; fail=1; fi; \
	done; done; done; \
	[ $$fail -eq 0 ] && echo "PLE prefetch: identical tokens on and off, 16 configurations" || exit 1

qwen38-tiny-check: qwen38-tiny-generate qwen38$(EXE)
	@for batch in 0 1; do for bf16 in 0 1; do for cap in 1 4; do \
		Q38_PREFILL_BATCH=$$batch Q38_NATIVE_BF16=$$bf16 OMP_NUM_THREADS=2 SNAP=./qwen38_tiny ./qwen38$(EXE) $$cap 8 ./qwen38_tiny/ref.json || exit $$?; \
	done; \
	done; \
	done

# Context-size gates: KV layout, growth across requests, and the attention
# capacity. Includes qwen36.c directly, so no model file is needed.
# Same tier sources as the engine: the test includes qwen36.c, so with CUDA=1
# it needs qwen36_tier.c and the backend object too (without CUDA, the header's
# inline stubs cover it and both are empty).
tests/test_qwen36_ctx$(EXE): tests/test_qwen36_ctx.c qwen36.c qwen36_tier.h st.h json.h compat.h $(QWEN36_TIER_SRC) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $< $(QWEN36_TIER_SRC) $(CUDA_OBJ) -o $@ $(LDFLAGS)

tests/test_qwen36_dense_batch$(EXE): tests/test_qwen36_dense_batch.c qwen36.c qwen36_tier.h st.h json.h compat.h $(QWEN36_TIER_SRC) $(CUDA_OBJ)
	$(CC) $(QWEN36_CFLAGS) $< $(QWEN36_TIER_SRC) $(CUDA_OBJ) -o $@ $(QWEN36_LDFLAGS)

tests/test_qwen36_json_escape$(EXE): tests/test_qwen36_json_escape.c qwen36.c qwen36_tier.h st.h json.h compat.h $(QWEN36_TIER_SRC) $(CUDA_OBJ)
	$(CC) $(QWEN36_CFLAGS) $< $(QWEN36_TIER_SRC) $(CUDA_OBJ) -o $@ $(QWEN36_LDFLAGS)

# Reproducible local timing evidence; intentionally not a noisy CI perf gate.
tests/bench_qwen36_dense_batch$(EXE): tests/bench_qwen36_dense_batch.c qwen36.c qwen36_tier.h st.h json.h compat.h $(QWEN36_TIER_SRC) $(CUDA_OBJ)
	$(CC) $(QWEN36_CFLAGS) $< $(QWEN36_TIER_SRC) $(CUDA_OBJ) -o $@ $(QWEN36_LDFLAGS)

inkling$(EXE): inkling.c cli_args.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h omp_tune.h route_trace.h kv_prefix.h serve_codec.h $(INK_CUDA_OBJ) $(METAL_OBJ)
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

# GLM-5.3-Flash: CPU puro, esperti in streaming dal contenitore int4 gs64.
# Nessun oggetto CUDA/Vulkan/Metal fra le dipendenze perche' il motore non ne
# ha ancora: quando li avra', questa riga somigliera' a quella di kimi_k3.
glm53$(EXE): glm53.c cli_args.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h hyper_connections.h delta_attention.h sparse_index.h vision_tower.h $(VK_OBJ) $(VK_SPV)
	$(CC) $(CFLAGS) glm53.c $(VK_OBJ) -o glm53$(EXE) $(LDFLAGS)

kimi_k3$(EXE): kimi_k3.c cli_args.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h omp_tune.h route_trace.h kv_prefix.h serve_codec.h backend_cuda.h $(CUDA_OBJ) $(VK_OBJ) $(VK_SPV) $(METAL_OBJ)
	$(CC) $(CFLAGS) kimi_k3.c $(CUDA_OBJ) $(VK_OBJ) $(METAL_OBJ) -o kimi_k3$(EXE) $(LDFLAGS)

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

tests/test_serve_sentinel$(EXE): tests/test_serve_sentinel.c compat.h serve_codec.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_cluster_protocol$(EXE): tests/test_cluster_protocol.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_ue8m0$(EXE): tests/test_ue8m0.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tok_o200k$(EXE): tests/test_tok_o200k.c tok.h tok_unicode.h tok_unicode_o200k.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_k3_ram_budget$(EXE): tests/test_k3_ram_budget.c kimi_k3.c st.h tok.h quant.h compat.h omp_tune.h route_trace.h kv_prefix.h serve_codec.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_k3_mmap$(EXE): tests/test_k3_mmap.c kimi_k3.c st.h tok.h quant.h compat.h omp_tune.h route_trace.h kv_prefix.h serve_codec.h
	$(CC) $(NOCUDA_CFLAGS) $< $(VK_OBJ) -o $@ $(NOCUDA_LDFLAGS)

tests/test_tok_kimi_tiny$(EXE): tests/test_tok_kimi_tiny.c tok.h tok_unicode.h tok_unicode_o200k.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
tests/test_st_pread$(EXE): tests/test_st_pread.c st.h json.h compat.h
	$(CC) $(CFLAGS) -DST_PREAD_CHUNK=7 $< -o $@ $(LDFLAGS)

tests/test_st_slice$(EXE): tests/test_st_slice.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_qwen38_tokenizer$(EXE): tests/test_qwen38_tokenizer.c qwen38.c qwen38_core.h qwen38_nfc.h qwen38_nfc_tables.h edge_runtime.c edge_runtime.h edge_adapters.h edge_adapter_internal.h st.h json.h compat.h quant.h route_trace.h tok_unicode.h tok_unicode_o200k.h
	$(CC) $(NOCUDA_CFLAGS) $< edge_runtime.c -o $@ $(NOCUDA_LDFLAGS)

tests/test_qwen38_config$(EXE): tests/test_qwen38_config.c qwen38.c qwen38_core.h qwen38_nfc.h qwen38_nfc_tables.h st.h json.h compat.h quant.h route_trace.h tok_unicode.h tok_unicode_o200k.h
	$(CC) $(NOCUDA_CFLAGS) $< -o $@ $(NOCUDA_LDFLAGS)

tests/test_qwen38_serve_framing$(EXE): tests/test_qwen38_serve_framing.c qwen38.c qwen38_core.h qwen38_nfc.h qwen38_nfc_tables.h st.h json.h compat.h quant.h route_trace.h serve_codec.h tok_unicode.h tok_unicode_o200k.h
	$(CC) $(NOCUDA_CFLAGS) $< -o $@ $(NOCUDA_LDFLAGS)

tests/test_qwen38_vision$(EXE): tests/test_qwen38_vision.c qwen38_vision.h st.h json.h compat.h
	$(CC) $(CFLAGS) -Wno-unused-function $< -o $@ $(LDFLAGS)

# Genera la fixture della torre e la confronta con l'oracolo upstream. Non serve
# il checkpoint: 240k parametri casuali, meno di un megabyte.
.PHONY: qwen38-vision-check
qwen38-vision-check: tests/test_qwen38_vision$(EXE)
	$(PYTHON) tools/make_qwen38_vision_tiny.py --out ./qwen38_vision_tiny
	./tests/test_qwen38_vision$(EXE) ./qwen38_vision_tiny

# Il percorso completo: frame IMAGE, torre, sostituzione degli embedding,
# generazione. Prova che due immagini diverse danno risposte diverse, che e'
# l'unica domanda a cui una fixture con pesi casuali sa rispondere onestamente.
.PHONY: qwen38-vision-serve-check
qwen38-vision-serve-check: qwen38$(EXE)
	$(PYTHON) tools/make_qwen38_multimodal_tiny.py --out ./qwen38_mm_tiny
	$(PYTHON) tools/make_edge_tiny_tokenizer.py --vocab-size 64 ./qwen38_mm_tiny
	$(PYTHON) tests/test_qwen38_vision_serve.py --binary ./qwen38$(EXE) --fixture ./qwen38_mm_tiny

tests/test_qwen38_prefix$(EXE): tests/test_qwen38_prefix.c qwen38.c qwen38_core.h qwen38_nfc.h qwen38_nfc_tables.h st.h json.h compat.h quant.h route_trace.h serve_codec.h tok_unicode.h tok_unicode_o200k.h
	$(CC) $(NOCUDA_CFLAGS) $< -o $@ $(NOCUDA_LDFLAGS)

tests/test_qwen38_metrics$(EXE): tests/test_qwen38_metrics.c qwen38.c qwen38_core.h qwen38_nfc.h qwen38_nfc_tables.h st.h json.h compat.h quant.h route_trace.h serve_codec.h tok_unicode.h tok_unicode_o200k.h
	$(CC) $(NOCUDA_CFLAGS) $< -o $@ $(NOCUDA_LDFLAGS)

tests/test_qwen38_native_weights$(EXE): tests/test_qwen38_native_weights.c qwen38.c qwen38_core.h qwen38_nfc.h qwen38_nfc_tables.h segment_runtime.c segment_runtime.h segment_adapters.h segment_adapter_internal.h st.h json.h compat.h quant.h route_trace.h serve_codec.h tok_unicode.h tok_unicode_o200k.h
	$(CC) $(NOCUDA_CFLAGS) $< segment_runtime.c -o $@ $(NOCUDA_LDFLAGS)

tests/test_st_map$(EXE): tests/test_st_map.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_dup_name_refusal$(EXE): tests/test_dup_name_refusal.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

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

# RoPE inv_freq precompute must stay byte-identical to the old powf-per-position
# path; drives the real rope_interleave (includes colibri.c) vs a reference.
# -ffp-contract=off: the in-test reference is a DIFFERENT loop body than the
# shipping one, and on FMA targets (x86-64-v3, where -ffp-contract=fast is the
# default) the compiler contracts the two bodies differently — a*cs-b*sn then
# rounds differently between reference and real function even for pre-change
# code, a harness artifact rather than an engine regression. Engine-level
# equivalence under the shipping flags is covered by the differential dump
# (old vs new rope_interleave are byte-identical); this test guards the
# source-level formula equivalence and must not depend on contraction luck.
tests/test_rope_invfreq$(EXE): tests/test_rope_invfreq.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) -ffp-contract=off $< -o $@ $(LDFLAGS)

# schema->GBNF compile cache (#7): grammar_reset must equal a fresh setup, and
# the GrDraft.src ownership must not leak/double-free. Includes colibri.c.
tests/test_grammar_cache$(EXE): tests/test_grammar_cache.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
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

tests/test_serve_codec$(EXE): tests/test_serve_codec.c serve_codec.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_segment_runtime$(EXE): tests/test_segment_runtime.c segment_runtime.c segment_runtime.h
	$(CC) $(CFLAGS) tests/test_segment_runtime.c segment_runtime.c -o $@ $(LDFLAGS)

tests/test_segment_conformance$(EXE): tests/test_segment_conformance.c tests/segment_conformance_fixtures.c tests/segment_conformance_fixtures.h segment_runtime.c segment_runtime.h
	$(CC) $(CFLAGS) tests/test_segment_conformance.c tests/segment_conformance_fixtures.c segment_runtime.c -o $@ $(LDFLAGS)

tests/test_inkling_serve_framing$(EXE): tests/test_inkling_serve_framing.c inkling.c st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h omp_tune.h route_trace.h kv_prefix.h serve_codec.h $(INK_CUDA_OBJ) $(METAL_OBJ)
	$(CC) $(CFLAGS) $< $(INK_CUDA_OBJ) $(METAL_OBJ) -o $@ $(LDFLAGS)

tests/test_inkling_shared_batch$(EXE): tests/test_inkling_shared_batch.c inkling.c st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h omp_tune.h route_trace.h kv_prefix.h serve_codec.h $(INK_CUDA_OBJ) $(METAL_OBJ)
	$(CC) $(CFLAGS) $< $(INK_CUDA_OBJ) $(METAL_OBJ) -o $@ $(LDFLAGS)

# Reproducible local A/B for the shared-expert prefill path.  This is timing
# evidence, not a CI gate: build and run it explicitly on the target CPU.
tests/bench_inkling_shared_batch$(EXE): tests/bench_inkling_shared_batch.c inkling.c st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h omp_tune.h route_trace.h kv_prefix.h serve_codec.h $(INK_CUDA_OBJ) $(METAL_OBJ)

tests/test_inkling_cache_index$(EXE): tests/test_inkling_cache_index.c inkling.c st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h omp_tune.h route_trace.h kv_prefix.h serve_codec.h $(INK_CUDA_OBJ) $(METAL_OBJ)
	$(CC) $(CFLAGS) $< $(INK_CUDA_OBJ) $(METAL_OBJ) -o $@ $(LDFLAGS)

tests/test_kimi_serve_framing$(EXE): tests/test_kimi_serve_framing.c kimi_k3.c kv_prefix.h serve_codec.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h omp_tune.h route_trace.h
	$(CC) $(NOCUDA_CFLAGS) $< $(VK_OBJ) -o $@ $(NOCUDA_LDFLAGS)

tests/test_kimi_cache_index$(EXE): tests/test_kimi_cache_index.c kimi_k3.c kv_prefix.h serve_codec.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h omp_tune.h route_trace.h
	$(CC) $(NOCUDA_CFLAGS) $< $(VK_OBJ) -o $@ $(NOCUDA_LDFLAGS)

tests/test_k3_chat_tools$(EXE): tests/test_k3_chat_tools.c kimi_k3.c kv_prefix.h serve_codec.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h omp_tune.h route_trace.h
	$(CC) $(NOCUDA_CFLAGS) $< $(VK_OBJ) -o $@ $(NOCUDA_LDFLAGS)

# olmoe's matmul_q, not colibri's: compares the IDOT path against FP32
# ACTIVATIONS rather than an integer reference. NOCUDA_* for the same reason
# the olmoe target uses it -- olmoe.c contains no COLI_CUDA code.
tests/test_olmoe_matmul_q$(EXE): tests/test_olmoe_matmul_q.c olmoe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h serve_codec.h
	$(CC) $(NOCUDA_CFLAGS) $< -o $@ $(NOCUDA_LDFLAGS)

tests/test_olmoe_serve_framing$(EXE): tests/test_olmoe_serve_framing.c olmoe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h serve_codec.h
	$(CC) $(NOCUDA_CFLAGS) $< -o $@ $(NOCUDA_LDFLAGS)

tests/test_olmoe_cache_index$(EXE): tests/test_olmoe_cache_index.c olmoe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h serve_codec.h
	$(CC) $(NOCUDA_CFLAGS) $< -o $@ $(NOCUDA_LDFLAGS)

tests/test_qwen36_cache_index$(EXE): tests/test_qwen36_cache_index.c qwen36.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h serve_codec.h qwen36_tier.h $(QWEN36_TIER_SRC) $(CUDA_OBJ)
	$(CC) $(QWEN36_CFLAGS) $< $(QWEN36_TIER_SRC) $(CUDA_OBJ) -o $@ $(QWEN36_LDFLAGS)

tests/test_idot$(EXE): tests/test_idot.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_i4_grouped$(EXE): tests/test_i4_grouped.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_stops$(EXE): tests/test_stops.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_cfg_topk$(EXE): tests/test_cfg_topk.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
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

tests/test_kv_alloc$(EXE): tests/test_kv_alloc.c colibri.c st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h kv_fp8.h kv_tq.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_kv_fp8$(EXE): tests/test_kv_fp8.c kv_fp8.h decode_batch.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_kv_tq$(EXE): tests/test_kv_tq.c kv_tq.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_kv_disk$(EXE): tests/test_kv_disk.c colibri.c st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h kv_fp8.h kv_tq.h
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

# Host-only: backend_cuda.h's format predicate is plain C, so its truth table is
# a portable gate even where there is no CUDA toolchain. Same arrangement as
# metal_fused_fmt_ok's CPU-testable truth table.
tests/test_cuda_fmt_guard$(EXE): tests/test_cuda_fmt_guard.c backend_cuda.h
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

tests/test_edge_runtime$(EXE): tests/test_edge_runtime.c edge_runtime.c edge_runtime.h
	$(CC) $(CFLAGS) tests/test_edge_runtime.c edge_runtime.c -o $@ $(LDFLAGS)

ifeq ($(COLI_V4_SUPPORTED),1)
include Makefile.deepseek-v4.units

# Real engine-owned Segment adapters. These are CPU-baseline embedding
# objects: ordinary Colibri executables do not link them, and GPU capability
# is advertised only after an adapter has a real backend implementation.
SEGMENT_BUILD_DIR = build/segment
SEGMENT_CPU_CFLAGS = $(filter-out -DCOLI_CUDA -DCOLI_ANS -DCOLI_METAL \
	-DCOLI_VULKAN,$(CFLAGS)) -include pthread.h \
	-DCOLI_SEGMENT_ADAPTER -DCOLI_EDGE_ADAPTER
SEGMENT_ENGINE_OBJS = \
	$(SEGMENT_BUILD_DIR)/glm.o \
	$(SEGMENT_BUILD_DIR)/glm53.o \
	$(SEGMENT_BUILD_DIR)/inkling.o \
	$(SEGMENT_BUILD_DIR)/kimi.o \
	$(SEGMENT_BUILD_DIR)/olmoe.o \
	$(SEGMENT_BUILD_DIR)/qwen36.o \
	$(SEGMENT_BUILD_DIR)/qwen38.o
SEGMENT_V4_UNITS = $(filter-out COLI_V4_UNIT_PROMPT \
	COLI_V4_UNIT_GENERATE_STATS,$(V4_TARGET_UNITS)) $(V4_SEGMENT_UNIT)
SEGMENT_V4_OBJS = $(addprefix $(SEGMENT_BUILD_DIR)/,$(addsuffix .o,$(SEGMENT_V4_UNITS)))
SEGMENT_V4_REGISTRY_OBJ = $(SEGMENT_BUILD_DIR)/expert_store_registry.o
SEGMENT_ALL_OBJS = $(SEGMENT_ENGINE_OBJS) $(SEGMENT_V4_OBJS) \
	$(SEGMENT_V4_REGISTRY_OBJ)
SEGMENT_RUNTIME_OBJS = \
	$(SEGMENT_BUILD_DIR)/segment_runtime.o \
	$(SEGMENT_BUILD_DIR)/edge_runtime.o
SEGMENT_RUNTIME_LIB = $(SEGMENT_BUILD_DIR)/libcolibri_segment_edge.a

$(SEGMENT_BUILD_DIR):
	mkdir -p $@

$(SEGMENT_BUILD_DIR)/glm.o: colibri.c segment_runtime.h edge_runtime.h \
		segment_adapters.h edge_adapters.h segment_adapter_internal.h \
		edge_adapter_internal.h edge_tok_internal.h st.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -DCOLIBRI_NO_MAIN -c colibri.c -o $@

$(SEGMENT_BUILD_DIR)/glm53.o: glm53.c segment_runtime.h edge_runtime.h \
		segment_adapters.h edge_adapters.h segment_adapter_internal.h \
		edge_adapter_internal.h st.h quant.h tok.h hyper_connections.h \
		delta_attention.h sparse_index.h vision_tower.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -DGLM53_NO_MAIN -c glm53.c -o $@

$(SEGMENT_BUILD_DIR)/inkling.o: inkling.c segment_runtime.h edge_runtime.h \
		segment_adapters.h edge_adapters.h segment_adapter_internal.h \
		edge_adapter_internal.h edge_tok_internal.h st.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -DINKLING_NO_MAIN -c inkling.c -o $@

$(SEGMENT_BUILD_DIR)/kimi.o: kimi_k3.c segment_runtime.h edge_runtime.h \
		segment_adapters.h edge_adapters.h segment_adapter_internal.h \
		edge_adapter_internal.h edge_tok_internal.h st.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -DKIMI_K3_NO_MAIN -c kimi_k3.c -o $@

$(SEGMENT_BUILD_DIR)/olmoe.o: olmoe.c segment_runtime.h edge_runtime.h \
		segment_adapters.h edge_adapters.h segment_adapter_internal.h \
		edge_adapter_internal.h edge_tok_internal.h st.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -DOLMOE_NO_MAIN -c olmoe.c -o $@

$(SEGMENT_BUILD_DIR)/qwen36.o: qwen36.c segment_runtime.h edge_runtime.h \
		segment_adapters.h edge_adapters.h segment_adapter_internal.h \
		edge_adapter_internal.h st.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -DQWEN36_NO_MAIN -c qwen36.c -o $@

$(SEGMENT_BUILD_DIR)/qwen38.o: qwen38.c qwen38_core.h segment_runtime.h \
		segment_adapters.h segment_adapter_internal.h edge_runtime.h \
		edge_adapters.h edge_adapter_internal.h st.h json.h compat.h quant.h \
		qwen38_nfc.h qwen38_nfc_tables.h route_trace.h tok_unicode.h \
		tok_unicode_o200k.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -DQWEN38_NO_MAIN -c qwen38.c -o $@

$(SEGMENT_V4_OBJS): $(SEGMENT_BUILD_DIR)/%.o: deepseek_v4.c deepseek_v4.h \
		deepseek_v4_internal.h segment_runtime.h segment_adapters.h \
		edge_runtime.h edge_adapters.h segment_adapter_internal.h \
		edge_adapter_internal.h edge_tok_internal.h st.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -D$* -c deepseek_v4.c -o $@

$(SEGMENT_V4_REGISTRY_OBJ): expert_store_registry.c expert_store_registry.h \
		expert_store.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -c expert_store_registry.c -o $@

$(SEGMENT_BUILD_DIR)/segment_runtime.o: segment_runtime.c segment_runtime.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -c segment_runtime.c -o $@

$(SEGMENT_BUILD_DIR)/edge_runtime.o: edge_runtime.c edge_runtime.h | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -c edge_runtime.c -o $@

$(SEGMENT_RUNTIME_LIB): $(SEGMENT_RUNTIME_OBJS) $(SEGMENT_ALL_OBJS)
	$(AR) rcs $@ $^

.PHONY: segment-edge-library
segment-edge-library: $(SEGMENT_RUNTIME_LIB)

tests/test_segment_adapters_registration$(EXE): \
		tests/test_segment_adapters_registration.c segment_runtime.c \
		edge_runtime.c segment_runtime.h edge_runtime.h segment_adapters.h \
		edge_adapters.h $(SEGMENT_ALL_OBJS)
	$(CC) $(SEGMENT_CPU_CFLAGS) tests/test_segment_adapters_registration.c \
		segment_runtime.c edge_runtime.c $(SEGMENT_ALL_OBJS) -o $@ $(NOCUDA_LDFLAGS)

tests/test_segment_adapters_real$(EXE): tests/test_segment_adapters_real.c \
		segment_runtime.c edge_runtime.c segment_runtime.h edge_runtime.h \
		segment_adapters.h edge_adapters.h $(SEGMENT_ALL_OBJS)
	$(CC) $(SEGMENT_CPU_CFLAGS) tests/test_segment_adapters_real.c \
		segment_runtime.c edge_runtime.c $(SEGMENT_ALL_OBJS) -o $@ $(NOCUDA_LDFLAGS)

tests/test_edge_adapters_registration$(EXE): \
		tests/test_edge_adapters_registration.c edge_runtime.c segment_runtime.c \
		edge_runtime.h segment_runtime.h edge_adapters.h $(SEGMENT_ALL_OBJS)
	$(CC) $(SEGMENT_CPU_CFLAGS) tests/test_edge_adapters_registration.c \
		edge_runtime.c segment_runtime.c $(SEGMENT_ALL_OBJS) -o $@ $(NOCUDA_LDFLAGS)

tests/test_edge_adapters_real$(EXE): tests/test_edge_adapters_real.c \
		edge_runtime.c segment_runtime.c edge_runtime.h segment_runtime.h \
		edge_adapters.h segment_adapters.h json.h $(SEGMENT_ALL_OBJS)
	$(CC) $(SEGMENT_CPU_CFLAGS) tests/test_edge_adapters_real.c \
		edge_runtime.c segment_runtime.c $(SEGMENT_ALL_OBJS) -o $@ $(NOCUDA_LDFLAGS)

.PHONY: segment-adapters segment-adapters-real
segment-adapters: tests/test_segment_adapters_registration$(EXE)
	./tests/test_segment_adapters_registration$(EXE)

.PHONY: edge-adapters edge-adapters-real
edge-adapters: tests/test_edge_adapters_registration$(EXE)
	./tests/test_edge_adapters_registration$(EXE)

# Maintainer/release gate. Paths point at generated/converted tiny Colibri
# containers; see docs/segment-runtime.md for the exact generators.
# OMP_NUM_THREADS defaults to a stable, non-single-threaded 2 for reproducible
# reductions, but callers can override it to exercise a different schedule.
# Kimi's generated fixture is safe at a few MB; overcommit there only prevents
# its production-size reserve heuristic from making the test host-dependent.
segment-adapters-real: tests/test_segment_adapters_real$(EXE)
	@test -n "$(GLM_SEGMENT_MODEL)" || { echo "set GLM_SEGMENT_MODEL" >&2; exit 2; }
	@test -n "$(GLM53_SEGMENT_MODEL)" || { echo "set GLM53_SEGMENT_MODEL" >&2; exit 2; }
	@test -n "$(INKLING_SEGMENT_MODEL)" || { echo "set INKLING_SEGMENT_MODEL" >&2; exit 2; }
	@test -n "$(KIMI_SEGMENT_MODEL)" || { echo "set KIMI_SEGMENT_MODEL" >&2; exit 2; }
	@test -n "$(OLMOE_SEGMENT_MODEL)" || { echo "set OLMOE_SEGMENT_MODEL" >&2; exit 2; }
	@test -n "$(QWEN_SEGMENT_MODEL)" || { echo "set QWEN_SEGMENT_MODEL" >&2; exit 2; }
	@test -n "$(QWEN38_SEGMENT_MODEL)" || { echo "set QWEN38_SEGMENT_MODEL" >&2; exit 2; }
	@test -n "$(DEEPSEEK_SEGMENT_MODEL)" || { echo "set DEEPSEEK_SEGMENT_MODEL" >&2; exit 2; }
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_segment_adapters_real$(EXE) glm "$(GLM_SEGMENT_MODEL)" 0 3 5 8
	GLM53_BITS=32 OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_segment_adapters_real$(EXE) glm53 "$(GLM53_SEGMENT_MODEL)" 0 2 4 16
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_segment_adapters_real$(EXE) inkling "$(INKLING_SEGMENT_MODEL)" 0 4 8 8
	COLI_RAM_OVERCOMMIT=1 OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_segment_adapters_real$(EXE) kimi "$(KIMI_SEGMENT_MODEL)" 0 3 6 8
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_segment_adapters_real$(EXE) olmoe "$(OLMOE_SEGMENT_MODEL)" 0 2 4 8
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_segment_adapters_real$(EXE) qwen36 "$(QWEN_SEGMENT_MODEL)" 0 4 8 8
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_segment_adapters_real$(EXE) qwen38 "$(QWEN38_SEGMENT_MODEL)" 0 2 4 8
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_segment_adapters_real$(EXE) qwen38 "$(QWEN38_SEGMENT_MODEL)" 1 2 4 8
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_segment_adapters_real$(EXE) deepseek_v4 "$(DEEPSEEK_SEGMENT_MODEL)" 0 1 3 8

# End-to-end model-edge gate. Every run performs tokenizer round-trip, embeds
# the independent oracle's prompt, executes the complete real Segment stack
# and compares greedy decode tokens with that oracle. The fixture tokenizer
# helper is only for existing math-only tiny checkpoints that omit one.
edge-adapters-real: tests/test_edge_adapters_real$(EXE)
	@test -n "$(GLM_EDGE_MODEL)" && test -n "$(GLM_EDGE_REF)" || { echo "set GLM_EDGE_MODEL and GLM_EDGE_REF" >&2; exit 2; }
	@test -n "$(GLM53_EDGE_MODEL)" && test -n "$(GLM53_EDGE_REF)" || { echo "set GLM53_EDGE_MODEL and GLM53_EDGE_REF" >&2; exit 2; }
	@test -n "$(INKLING_EDGE_MODEL)" && test -n "$(INKLING_EDGE_REF)" || { echo "set INKLING_EDGE_MODEL and INKLING_EDGE_REF" >&2; exit 2; }
	@test -n "$(KIMI_EDGE_MODEL)" && test -n "$(KIMI_EDGE_REF)" || { echo "set KIMI_EDGE_MODEL and KIMI_EDGE_REF" >&2; exit 2; }
	@test -n "$(OLMOE_EDGE_MODEL)" && test -n "$(OLMOE_EDGE_REF)" || { echo "set OLMOE_EDGE_MODEL and OLMOE_EDGE_REF" >&2; exit 2; }
	@test -n "$(QWEN_EDGE_MODEL)" && test -n "$(QWEN_EDGE_REF)" || { echo "set QWEN_EDGE_MODEL and QWEN_EDGE_REF" >&2; exit 2; }
	@test -n "$(QWEN38_EDGE_MODEL)" && test -n "$(QWEN38_EDGE_REF)" || { echo "set QWEN38_EDGE_MODEL and QWEN38_EDGE_REF" >&2; exit 2; }
	@test -n "$(DEEPSEEK_EDGE_MODEL)" && test -n "$(DEEPSEEK_EDGE_REF)" || { echo "set DEEPSEEK_EDGE_MODEL and DEEPSEEK_EDGE_REF" >&2; exit 2; }
	GLM_SEGMENT_EBITS=16 GLM_SEGMENT_DBITS=16 OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_edge_adapters_real$(EXE) glm "$(GLM_EDGE_MODEL)" "$(GLM_EDGE_REF)" 3
	GLM53_BITS=32 OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_edge_adapters_real$(EXE) glm53 "$(GLM53_EDGE_MODEL)" "$(GLM53_EDGE_REF)" 3
	INK_SEGMENT_BITS=0 OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_edge_adapters_real$(EXE) inkling "$(INKLING_EDGE_MODEL)" "$(INKLING_EDGE_REF)" 3
	COLI_RAM_OVERCOMMIT=1 K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_edge_adapters_real$(EXE) kimi "$(KIMI_EDGE_MODEL)" "$(KIMI_EDGE_REF)" 3
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_edge_adapters_real$(EXE) olmoe "$(OLMOE_EDGE_MODEL)" "$(OLMOE_EDGE_REF)" 3
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_edge_adapters_real$(EXE) qwen36 "$(QWEN_EDGE_MODEL)" "$(QWEN_EDGE_REF)" 3
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_edge_adapters_real$(EXE) qwen38 "$(QWEN38_EDGE_MODEL)" "$(QWEN38_EDGE_REF)" 3
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./tests/test_edge_adapters_real$(EXE) deepseek_v4 "$(DEEPSEEK_EDGE_MODEL)" "$(DEEPSEEK_EDGE_REF)" 3

V4_HOT_TEST_OBJ = COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16_TEST.o
V4_BATCH_TEST_OBJ = COLI_V4_UNIT_NATIVE_QUANT_BATCH_TEST.o
V4_TEST_LINK_OBJS = \
	COLI_V4_UNIT_ST.o \
	COLI_V4_UNIT_CONFIG.o \
	COLI_V4_UNIT_MATH.o \
	COLI_V4_UNIT_SPARSE_ATTENTION.o \
	COLI_V4_UNIT_RESOURCE_PLAN.o \
	COLI_V4_UNIT_PROMPT.o \
	$(filter-out COLI_V4_UNIT_EXPERT_STORE.o COLI_V4_UNIT_EXPERT.o,$(addsuffix .o,$(V4_TEST_UNITS))) \
	$(V4_HOT_TEST_OBJ) \
	$(V4_BATCH_TEST_OBJ) \
	COLI_V4_UNIT_EXPERT_ROWS16.o \
	COLI_V4_UNIT_NATIVE_QUANT.o \
	COLI_V4_UNIT_NATIVE_QUANT_DUAL.o \
	COLI_V4_UNIT_NATIVE_QUANT_ROWS16.o
# Windows engine objects are compiled with -DCOLI_V4_GPU_TIER (the CUDA tier
# is resolved from coli_cuda_dsv4*.dll at runtime), so the matvec references in
# native_quant.h resolve through the GPU unit + the DLL loader. Both are inert
# without a DLL next to the test binary.
ifneq ($(IS_WIN),)
V4_TEST_LINK_OBJS += COLI_V4_UNIT_GPU.o backend_loader_dsv4.o
V4_TEST_EXTRA_TARGETS = COLI_V4_UNIT_GPU.o backend_loader_dsv4.o
endif

tests/test_deepseek_v4$(EXE): tests/test_deepseek_v4.c deepseek_v4.c deepseek_v4.h \
		Makefile.deepseek-v4.units Makefile.deepseek-v4
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4-test-objs \
		COLI_V4_UNIT_ST.o COLI_V4_UNIT_CONFIG.o COLI_V4_UNIT_MATH.o COLI_V4_UNIT_SPARSE_ATTENTION.o \
		COLI_V4_UNIT_RESOURCE_PLAN.o COLI_V4_UNIT_PROMPT.o COLI_V4_UNIT_NATIVE_QUANT.o \
		COLI_V4_UNIT_NATIVE_QUANT_DUAL.o COLI_V4_UNIT_NATIVE_QUANT_ROWS16.o \
		COLI_V4_UNIT_EXPERT_ROWS16.o \
		$(V4_HOT_TEST_OBJ) \
		$(V4_BATCH_TEST_OBJ) \
		$(V4_TEST_EXTRA_TARGETS)
	$(CC) $(CFLAGS) -DCOLI_V4_TEST_HOOKS $< $(V4_TEST_LINK_OBJS) \
		-o $@ -pthread $(LDFLAGS)

V4_OWN_DIR = build/ownership
V4_OWN_CFLAGS = $(CFLAGS) -DCOLI_V4_TEST_HOOKS
V4_OWNERSHIP_OBJS = \
	$(V4_OWN_DIR)/COLI_V4_UNIT_RUNTIME.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_CONFIG.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_ST.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_NATIVE_QUANT.o \
	$(V4_OWN_DIR)/expert_store_registry.o

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

# The engine (RUNTIME unit) opens its expert store through the pluggable
# backend registry, so any test linking RUNTIME also links the registry object
# (which defines coli_expert_store_backend_open_selected + registers "auto").
$(V4_OWN_DIR)/expert_store_registry.o: expert_store_registry.c expert_store_registry.h expert_store.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -c expert_store_registry.c -o $@

tests/test_v4_ownership$(EXE): tests/test_v4_ownership.c $(V4_OWNERSHIP_OBJS)
	$(CC) $(V4_OWN_CFLAGS) $< $(V4_OWNERSHIP_OBJS) -o $@ -pthread $(LDFLAGS)

tests/test_v4_serve_framing$(EXE): tests/test_v4_serve_framing.c deepseek_v4.c \
		deepseek_v4.h deepseek_v4_internal.h serve_codec.h Makefile.deepseek-v4 Makefile.deepseek-v4.units
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) $@

endif

tests/test_route_trace$(EXE): tests/test_route_trace.c route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_cli_args$(EXE): tests/test_cli_args.c cli_args.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_798_guards$(EXE): tests/test_798_guards.c st.h json.h compat.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_dsa_select$(EXE): tests/test_dsa_select.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_corpus_draft$(EXE): tests/test_corpus_draft.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_cap_precedence$(EXE): tests/test_cap_precedence.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
tests/test_mirror_stripe_split$(EXE): tests/test_mirror_stripe_split.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
tests/test_v4_hybrid_policy$(EXE): tests/test_v4_hybrid_policy.c deepseek_v4_hybrid.h hybrid_split.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_k3_fill_budget$(EXE): tests/test_k3_fill_budget.c hybrid_split.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_v4_bank_pair$(EXE): tests/test_v4_bank_pair.c deepseek_v4_bank_pair.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_ram_clamp$(EXE): tests/test_ram_clamp.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_cap_mixed_width$(EXE): tests/test_cap_mixed_width.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_eslot_inflight$(EXE): tests/test_eslot_inflight.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_glm_cache_index$(EXE): tests/test_glm_cache_index.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_ssd_probe$(EXE): tests/test_ssd_probe.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# bench_dsa_select is a microbenchmark (old qsort vs new quickselect partial-select, #356),
# NOT a test gate -- intentionally absent from TEST_BINS. Build on demand: make tests/bench_dsa_select
tests/bench_dsa_select$(EXE): tests/bench_dsa_select.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# bench_indexer_allocations: microbenchmark (DeepSeek V4 indexer malloc vs persistent arena scratch), NOT a test gate.
# Build on demand: make tests/bench_indexer_allocations
tests/bench_indexer_allocations$(EXE): tests/bench_indexer_allocations.c
	$(CC) -O3 $< -o $@

# bench_idot: microbenchmark (single-acc vs independent-acc AVX-VNNI idot), NOT a test gate.
# Build on demand on an AVX-VNNI CPU: make tests/bench_idot ARCH=native
tests/bench_idot$(EXE): tests/bench_idot.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# bench_gemv_stream: microbenchmark (decode-regime GEMV bandwidth vs the read ceiling;
# frozen-baseline + deinterleaved-x candidate A/B), NOT a test gate.
# Build on demand: make tests/bench_gemv_stream ARCH=native
tests/bench_gemv_stream$(EXE): tests/bench_gemv_stream.c colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h quant.h sample.h kv_persist.h telemetry.h route_trace.h
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

install: colibri$(EXE) glm53$(EXE) inkling$(EXE) kimi_k3$(EXE) olmoe$(EXE) qwen36$(EXE) qwen38$(EXE) \
	$(if $(filter 1,$(COLI_V4_SUPPORTED)),deepseek-v4,)
	$(INSTALL) -d $(DESTDIR)$(BINDIR)
	$(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)
	$(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)/tools
	$(INSTALL) -m 755 coli $(DESTDIR)$(BINDIR)/coli
	$(INSTALL) -m 755 colibri$(EXE) $(DESTDIR)$(LIBEXECDIR)/colibri$(EXE)
	$(INSTALL) -m 755 glm53$(EXE) $(DESTDIR)$(LIBEXECDIR)/glm53$(EXE)
	$(INSTALL) -m 755 inkling$(EXE) $(DESTDIR)$(LIBEXECDIR)/inkling$(EXE)
	$(INSTALL) -m 755 kimi_k3$(EXE) $(DESTDIR)$(LIBEXECDIR)/kimi_k3$(EXE)
	$(INSTALL) -m 755 olmoe$(EXE) $(DESTDIR)$(LIBEXECDIR)/olmoe$(EXE)
	$(INSTALL) -m 755 qwen36$(EXE) $(DESTDIR)$(LIBEXECDIR)/qwen36$(EXE)
	$(INSTALL) -m 755 qwen38$(EXE) $(DESTDIR)$(LIBEXECDIR)/qwen38$(EXE)
	@if [ -f deepseek_v4$(EXE) ]; then \
	  $(INSTALL) -m 755 deepseek_v4$(EXE) $(DESTDIR)$(LIBEXECDIR)/deepseek_v4$(EXE); \
	fi
	$(INSTALL) -m 644 family_registry.py resource_plan.py doctor.py autotune.py \
		openai_server.py cluster.py v4_dsml.py version.py $(DESTDIR)$(LIBEXECDIR)/
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
.PHONY: all deepseek-v4 deepseek-v4-clean deepseek-v4-oracle \
	deepseek-v4-tiny-generate deepseek-v4-tiny-check \
	qwen38-tiny-generate qwen38-tiny-check \
	iq3 rans fuzz-rans cuda-test hip-test gpu-compile cuda-bench fp8-bench cuda-dll hip-dll portable \
	test-c test-python test check clean install uninstall bench

# Own line: see the TEST_RULES note above on shared .PHONY lines.
.PHONY: bench-omp-grain
.PHONY: test-asan
.PHONY: dsv4-cuda-test cuda-dsv4-dll cuda-dsv4-dg-dll dsv4-cuda-loader-test

tests/test_kv_prefix$(EXE): tests/test_kv_prefix.c kv_prefix.h
	$(CC) $(CFLAGS) tests/test_kv_prefix.c -o tests/test_kv_prefix$(EXE) $(LDFLAGS)

tests/test_kimi_request_state$(EXE): tests/test_kimi_request_state.c kimi_k3.c kv_prefix.h serve_codec.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h omp_tune.h route_trace.h
	$(CC) $(NOCUDA_CFLAGS) tests/test_kimi_request_state.c $(VK_OBJ) -o tests/test_kimi_request_state$(EXE) $(NOCUDA_LDFLAGS)
