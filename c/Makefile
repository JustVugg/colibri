UNAME_S := $(shell uname -s)
MINGW    := $(findstring MINGW,$(UNAME_S))
MSYS     := $(findstring MSYS,$(UNAME_S))
IS_WIN   := $(MINGW)$(MSYS)

ifeq ($(UNAME_S),Darwin)
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp non trovato: build SINGLE-THREAD. Per il multithread: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm $(OMPL)
EXE     =
else ifneq ($(IS_WIN),)
# --- Windows 11 x86-64 (MinGW-w64 / MSYS2) ---
# GCC + libgomp + winpthreads: pthread, OpenMP, clock_gettime, opendir/readdir,
# AVX2 intrinsics — tutto gratis, nessun porting.
# ARCH default = x86-64-v3 (binario portabile con AVX2). Per max velocita'
# su QUESTA macchina: make ARCH=native
CC      = gcc
ARCH   ?= x86-64-v3
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -static
EXE     = .exe
else
# --- Linux x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp
EXE     =
endif

# CUDA=1 adds an opt-in backend for resident tensors. The default build remains
# pure C and keeps the original zero-dependency runtime.
CUDA      ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
PYTHON    ?= python3
CUDA_OBJ  =
TEST_BINS = tests/test_json$(EXE) tests/test_st$(EXE) tests/test_tier$(EXE) tests/test_grammar$(EXE)
ifeq ($(CUDA),1)
ifeq ($(UNAME_S),Darwin)
$(error CUDA=1 is supported only on Linux and Windows)
endif
ifneq ($(IS_WIN),)
# Windows (Phase G1): backend_cuda.cu is built as coli_cuda.dll by a separate
# rule (nvcc + MSVC host compiler), then loaded at runtime via
# LoadLibrary/GetProcAddress from compat_cuda.c. This side of the Makefile
# only pulls in the CPU-side loader; the DLL itself is built with:
#     make coli_cuda.dll CUDA=1
# from an environment that has both nvcc and cl.exe on PATH (vcvars64.bat).
CFLAGS   += -DCOLI_CUDA
CUDA_OBJ  = compat_cuda.o
else
# Linux: compile-time link with the .cu directly (original path, unchanged).
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
endif
endif

all: glm$(EXE)

# phony 'glm' → 'glm.exe' on Windows (so 'make glm' and 'coli build' work on every platform)
glm: glm$(EXE)

glm$(EXE): glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h $(CUDA_OBJ)
	$(CC) $(CFLAGS) glm.c $(CUDA_OBJ) -o glm$(EXE) $(LDFLAGS)

backend_cuda.o: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -c backend_cuda.cu -o $@

# Windows-only: compile the LoadLibrary shim into an object linkable with the
# MinGW-built glm.exe. compat_cuda.h/.c must be present alongside glm.c.
compat_cuda.o: compat_cuda.c compat_cuda.h backend_cuda.h
	$(CC) $(CFLAGS) -c compat_cuda.c -o $@

# Windows-only DLL build. Uses nvcc (which auto-picks cl.exe as host compiler
# when both are on PATH). Requires running from a shell where vcvars64.bat has
# initialized the MSVC toolchain — see README §Windows for details. The .def
# file drives symbol export; no source changes to backend_cuda.h needed.
# CUDA_ARCH=native lets nvcc pick the sm_XX matching the current GPU; override
# with e.g. CUDA_ARCH=sm_86 for portability across a set of known cards.
coli_cuda.dll: backend_cuda.cu backend_cuda.h coli_cuda.def
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) --shared -o coli_cuda.dll backend_cuda.cu \
		-Xlinker /DEF:coli_cuda.def -Xcompiler /MD

cuda-test: backend_cuda.cu backend_cuda.h tests/test_backend_cuda.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE)
	./backend_cuda_test$(EXE)

# Windows-only: validate coli_cuda.dll via LoadLibrary + GetProcAddress from a
# separate binary. Proves the DLL loading path is functional independently of
# the full glm.exe. Requires coli_cuda.dll built (run `make coli_cuda.dll` first).
cuda-dll-test: tests/test_dll_loadlib.c coli_cuda.dll
	$(CC) $(CFLAGS) -c tests/test_dll_loadlib.c -o tests/test_dll_loadlib.o
	$(CC) tests/test_dll_loadlib.o -o tests/test_dll_loadlib$(EXE) -lm
	./tests/test_dll_loadlib$(EXE)

olmoe$(EXE): olmoe.c st.h json.h compat.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe$(EXE) $(LDFLAGS)

# binario portabile da distribuire su altre macchine x86-64
portable:
	$(MAKE) glm$(EXE) ARCH=x86-64-v3

iobench$(EXE): iobench.c compat.h
	$(CC) $(CFLAGS) iobench.c -o iobench$(EXE) $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st$(EXE): tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier$(EXE): tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_grammar$(EXE): tests/test_grammar.c grammar.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test || exit 1; done

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test: test-c test-python

# Local validation: one portable CPU build and dependency-free tests.
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

clean:
	rm -f olmoe$(EXE) glm$(EXE) iobench$(EXE) backend_cuda.o backend_cuda_test$(EXE) $(TEST_BINS)
	rm -rf tests/__pycache__

.PHONY: all glm cuda-test portable test-c test-python test check clean
