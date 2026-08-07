"""Backend-DLL selection contract for the Windows runtime loader.

``c/backend_loader.c`` picks its DLL name and diagnostic label from
COLI_HIP_DLL: a CUDA_DLL host seeks coli_cuda.dll and says ``[CUDA]``, a
HIP_DLL host seeks coli_hip.dll and says ``[HIP]``. These tests prove that by
building both hosts under distinct EXE names and running each from an empty
directory, so the loader's not-found path is the one exercised.

Nothing here needs a GPU, nvidia-smi, a CUDA or HIP SDK, or a backend DLL: the
contract under test is the *miss*, which is reached before any vendor runtime.

The fixture helpers below are intentionally private copies rather than imports
from ``test_cuda_env``: that module probes the resident ``c/colibri.exe`` at
import time, and this owner must not depend on a binary it never runs.

The stub fixtures below additionally use ``objdump`` for structural inspection
(PE machine type, imports, exports). That is the objdump bundled with the same
MinGW/MSYS2 binutils as the ``gcc`` these tests already require — it is not a
new dependency, and the fixture class skips honestly if it is missing.

Prerequisites (tests skip gracefully when unmet):
- Windows (backend_loader.c is compiled only there)
- MinGW make + gcc, to build the two host artifacts
- MinGW gcc + objdump, for the stub fixtures
"""

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent


def _write_shard(path, tensors):
    """Write a minimal safetensors file to *path*."""
    offset = 0
    header = {}
    payload = b""
    for name, size in tensors:
        header[name] = {"dtype": "U8", "shape": [size],
                        "data_offsets": [offset, offset + size]}
        payload += b"\0" * size
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def _minimal_model(parent):
    """Create a minimal model directory that ``model_init()`` can parse.

    Returns the model Path.  The safetensors payload is dummy zeros — the
    binary will reach CUDA init, print its messages, then fail during
    model loading (fake weights).  The CUDA messages are already on stderr
    at that point.
    """
    model = Path(parent) / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2,
        "n_routed_experts": 2,
        "kv_lora_rank": 4,
        "qk_rope_head_dim": 2,
        "qk_nope_head_dim": 3,
        "v_head_dim": 5,
        "num_attention_heads": 2,
    }))
    _write_shard(model / "model.safetensors", [
        ("model.embed_tokens.weight", 100),
        ("model.layers.0.self_attn.q_a_proj.weight", 200),
        ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
        ("model.layers.1.mlp.experts.0.up_proj.weight", 30),
        ("model.layers.1.mlp.experts.1.gate_proj.weight", 30),
        ("model.layers.1.mlp.experts.1.up_proj.weight", 30),
    ])
    return model


class FixtureBuildError(AssertionError):
    """A stub fixture could not be built — distinct from a loader-contract failure.

    Carrying the compiler command and its complete output means a broken
    fixture is diagnosed as such instead of masquerading as a production bug.
    """


def _derive_backend_abi():
    """Parse the loader's own RESOLVE macros for the export set it requires.

    The ABI is read from c/backend_loader.c rather than restated here, so the
    stub cannot silently drift from the contract it is meant to satisfy: add a
    symbol to the loader and the fixture exports it on the next run.
    """
    source = (HERE / "backend_loader.c").read_text(encoding="utf-8", errors="replace")
    mandatory = ["coli_cuda_" + m for m in
                 re.findall(r"^\s+RESOLVE\((\w+),", source, re.M)]
    optional = ["coli_cuda_" + m for m in
                re.findall(r"^\s+RESOLVE_OPT\((\w+),", source, re.M)]
    if not mandatory:
        raise FixtureBuildError("no RESOLVE symbols parsed from backend_loader.c")
    overlap = set(mandatory) & set(optional)
    if overlap:
        raise FixtureBuildError("symbol in both RESOLVE and RESOLVE_OPT: %s"
                                % sorted(overlap))
    names = mandatory + optional
    bad = [n for n in names if not re.fullmatch(r"coli_cuda_[a-z0-9_]+", n)]
    if bad:
        raise FixtureBuildError("invalid generated export name(s): %s" % bad)
    if len(set(names)) != len(names):
        raise FixtureBuildError("duplicate symbol in the derived ABI")
    return mandatory, optional


# Imports every MinGW-linked DLL legitimately carries. Anything outside this
# set, other than the stub's deliberate amdhip64_7.dll, is a real dependency
# the fixture must not have acquired.
_ALLOWED_IMPORT_PREFIXES = ("kernel32", "api-ms-win-crt-", "msvcrt", "ucrtbase")
_FORBIDDEN_IMPORT_MARKERS = ("cudart", "nvcuda", "hiprtc", "rocblas", "rocwmma",
                             "amd_comgr", "hipblas")

_RUNTIME_MARKER_A = 0xA1
_RUNTIME_MARKER_B = 0xB2
_RUNTIME_BASENAME = "amdhip64_7.dll"


def _same_path(a, b):
    """Compare two Windows paths as paths, not as strings.

    Both separator conventions can be live in one process: under MSYS2 the temp
    root arrives as ``D:/a/_temp/msys64/tmp/...`` with forward slashes, while the
    loader joins with ``\\`` -- which is the correct Windows separator and is not
    something the loader should change. Comparing the two as raw strings then
    fails on the separator alone, so normalise both sides instead.
    """
    return (os.path.normcase(os.path.normpath(str(a)))
            == os.path.normcase(os.path.normpath(str(b))))
_TEST_ACCESSOR = "coli_test_bound_runtime"
_RUNTIME_DIR_VAR = "COLI_HIP_RUNTIME_DIR"

# A second, uniquely named dependency used only to provoke loader errors 126
# and 127 deterministically. It is built in two shapes under one basename: the
# "full" build exports both entry points and provides the import library, the
# "lean" build omits the extra one. A backend linked against full and run
# beside lean therefore loads a DLL that is missing a procedure it imports.
_DEP_BASENAME = "coli_test_dep.dll"
_DEP_PROBE = "coli_test_dep_probe"
_DEP_EXTRA = "coli_test_dep_extra"


class _StubFixture:
    """Two same-named runtime stubs plus one complete fake backend.

    Everything is generated and compiled inside a private temporary root whose
    path deliberately contains a space, so quoting mistakes surface here rather
    than on a user's "C:\\Program Files\\..." install. Nothing is ever written
    into c/.
    """

    def __init__(self):
        self.mandatory, self.optional = _derive_backend_abi()
        self.exports = self.mandatory + self.optional
        # The space is intentional; see the class docstring.
        self._tmp = tempfile.TemporaryDirectory(prefix="coli loader fix ")
        self.root = Path(self._tmp.name)
        self.runtime_a_dir = self.root / "runtime a"
        self.runtime_b_dir = self.root / "runtime-b"
        self.backend_dir = self.root / "backend"
        self.src_dir = self.root / "src"
        self.harness_dir = self.root / "harness"
        self.dep_full_dir = self.root / "dep full"
        self.dep_lean_dir = self.root / "dep-lean"
        for d in (self.runtime_a_dir, self.runtime_b_dir, self.backend_dir,
                  self.src_dir, self.harness_dir,
                  self.dep_full_dir, self.dep_lean_dir):
            d.mkdir(parents=True)
        self.harness_src = self.src_dir / "harness.c"
        self.harness_exe = self.harness_dir / "test_loader_harness.exe"
        self.cuda_harness_exe = self.harness_dir / "test_loader_harness_cuda.exe"
        self.helper_src = self.src_dir / "helpers.c"
        self.helper_exe = self.harness_dir / "test_loader_helpers.exe"
        self.harness_cmd = []
        self.cuda_harness_cmd = []
        self.helper_cmd = []
        self.runtime_a = self.runtime_a_dir / _RUNTIME_BASENAME
        self.runtime_b = self.runtime_b_dir / _RUNTIME_BASENAME
        self.backend = self.backend_dir / "coli_hip.dll"
        # Byte-identical twin under the CUDA name: the two builds differ only
        # in which container the loader looks for, so reusing the same exports
        # keeps the CUDA comparison about COLI_HIP_RUNTIME_DIR and nothing else.
        self.cuda_backend = self.backend_dir / "coli_cuda.dll"
        self.runtime_a_src = self.src_dir / "runtime_a.c"
        self.runtime_b_src = self.src_dir / "runtime_b.c"
        self.backend_src = self.src_dir / "backend.c"
        # Diagnostic-only artifacts (W1-B2c2).
        self.dep_full = self.dep_full_dir / _DEP_BASENAME
        self.dep_lean = self.dep_lean_dir / _DEP_BASENAME
        self.dep_full_src = self.src_dir / "dep_full.c"
        self.dep_lean_src = self.src_dir / "dep_lean.c"
        self.backend_dep_src = self.src_dir / "backend_dep.c"
        # Kept under its own name here; run_harness installs it in a sandbox
        # under the basename the production loader actually looks for.
        self.backend_dep = self.backend_dir / "coli_hip_depvariant.dll"
        self.bad_pe = self.backend_dir / "not_a_pe.bin"
        self._build()

    # --- construction -------------------------------------------------

    HARNESS_SOURCE = r'''
/* Native same-process loader harness (generated).
 *
 * Compiled together with the repository's real c/backend_loader.c, unmodified,
 * so this exercises the production loader rather than a reimplementation. It
 * optionally preloads one absolute amdhip64_7.dll, then calls coli_cuda_init
 * and reports which runtime the fake backend actually bound to.
 *
 * COLI_HIP_RUNTIME_DIR is deliberately NOT read here: the loader compiled in
 * beside this file is the only thing that may interpret it, so the harness
 * proves the production contract rather than a copy of it.
 *
 * usage: test_loader_harness.exe <absolute-preload-path|NONE>
 */
#include <windows.h>
#include <tlhelp32.h>
#include <wchar.h>
#include <stdio.h>
#include <string.h>
#include "backend_cuda.h"

/* Mirrors backend_loader.c's own selection, from the same macro. */
#ifdef COLI_HIP_DLL
#define HARNESS_BACKEND_DLL L"coli_hip.dll"
#else
#define HARNESS_BACKEND_DLL L"coli_cuda.dll"
#endif

#define RUNTIME_BASENAME L"amdhip64_7.dll"
#define WBUF 32768

static void emit_path(const char *key, const wchar_t *wide)
{
    char utf8[WBUF * 2];
    int n = WideCharToMultiByte(CP_UTF8, 0, wide, -1, utf8, (int)sizeof(utf8),
                                NULL, NULL);
    printf("%s=%s\n", key, n > 0 ? utf8 : "<unconvertible>");
}

/* Every loaded module whose basename matches the runtime, by full path.
 * Toolhelp32 can transiently fail with ERROR_BAD_LENGTH while the module list
 * changes, so the snapshot is retried a bounded number of times. */
static int runtime_inventory(const char *prefix)
{
    int attempt;
    for (attempt = 0; attempt < 8; attempt++) {
        HANDLE snap = CreateToolhelp32Snapshot(
            TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
        MODULEENTRY32W entry;
        int count = 0;
        if (snap == INVALID_HANDLE_VALUE) {
            if (GetLastError() == ERROR_BAD_LENGTH) continue;
            printf("%s_count=-1\n", prefix);
            printf("%s_error=%lu\n", prefix, GetLastError());
            return -1;
        }
        entry.dwSize = sizeof(entry);
        if (Module32FirstW(snap, &entry)) {
            do {
                if (_wcsicmp(entry.szModule, RUNTIME_BASENAME) == 0) {
                    wchar_t full[WBUF];
                    char key[128];
                    DWORD got = GetModuleFileNameW(entry.hModule, full, WBUF);
                    sprintf(key, "%s_%d", prefix, count);
                    /* GetModuleFileNameW is preferred; szExePath is MAX_PATH. */
                    emit_path(key, got ? full : entry.szExePath);
                    count++;
                }
            } while (Module32NextW(snap, &entry));
        }
        CloseHandle(snap);
        printf("%s_count=%d\n", prefix, count);
        return count;
    }
    printf("%s_count=-1\n", prefix);
    printf("%s_error=retry_exhausted\n", prefix);
    return -1;
}

int main(int argc, char **argv)
{
    int devices[1];
    int rc, i, requested = 0;

    /* Every argument after argv[0] is an absolute runtime to preload, except
     * the literal NONE. The handles are deliberately NEVER released: holding
     * them through the post-shutdown inventory is what makes production's own
     * reference ownership measurable. */
    for (i = 1; i < argc; i++)
        if (strcmp(argv[i], "NONE") != 0) requested++;
    printf("preload_requested=%d\n", requested);
    runtime_inventory("runtime_before");

    for (i = 1; i < argc; i++) {
        wchar_t wide[WBUF], got[WBUF];
        HMODULE runtime;
        int (*marker)(void);
        char key[64];
        if (strcmp(argv[i], "NONE") == 0) continue;
        MultiByteToWideChar(CP_UTF8, 0, argv[i], -1, wide, WBUF);
        /* Absolute path, Unicode API. No PATH edit, no SetDllDirectory, and
         * the runtime is never copied beside the harness. */
        runtime = LoadLibraryExW(wide, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!runtime) {
            printf("preload_%d_ok=0\n", i - 1);
            printf("preload_%d_error=%lu\n", i - 1, GetLastError());
            return 4;
        }
        printf("preload_%d_ok=1\n", i - 1);
        sprintf(key, "preload_%d_path", i - 1);
        if (GetModuleFileNameW(runtime, got, WBUF)) emit_path(key, got);
        marker = (int (*)(void))(void *)GetProcAddress(runtime,
                                                       "coli_test_runtime_marker");
        if (!marker) { printf("preload_%d_marker_missing=1\n", i - 1); return 5; }
        printf("preload_%d_marker=%d\n", i - 1, marker());
    }
    runtime_inventory("runtime_after_preload");

    devices[0] = 0;
    rc = coli_cuda_init(devices, 1);          /* the real loader runs here */
    printf("loader_init=%d\n", rc);

    /* Reported unconditionally: a configuration rejection must leave the
     * backend DLL untouched even though it is sitting right next to us. */
    printf("backend_loaded=%d\n",
           GetModuleHandleW(HARNESS_BACKEND_DLL) != NULL);

    if (rc) {
        HMODULE backend = GetModuleHandleW(HARNESS_BACKEND_DLL);
        if (backend) {
            wchar_t bpath[WBUF];
            int (*bound)(void);
            if (GetModuleFileNameW(backend, bpath, WBUF))
                emit_path("backend_path", bpath);
            /* Test-only accessor, read straight from the loaded fake backend;
             * production code is not modified to expose it. */
            bound = (int (*)(void))(void *)GetProcAddress(backend,
                                                          "coli_test_bound_runtime");
            if (bound) printf("bound_marker=%d\n", bound());
            else printf("bound_marker_missing=1\n");
        } else {
            printf("backend_handle_missing=1\n");
        }
    }
    runtime_inventory("runtime_after_backend");

    /* Real shutdown, then look again. Production must release exactly its own
     * reference: a runtime only it loaded disappears, while one this harness
     * also holds stays mapped. */
    coli_cuda_shutdown();
    printf("shutdown_called=1\n");
    runtime_inventory("runtime_after_shutdown");
    return 0;
}
'''

    HELPER_SOURCE = r'''
/* Native exerciser for the loader's internal Windows helpers (generated).
 *
 * Compiled together with the repository's real c/backend_loader.c and
 * -DCOLI_LOADER_TEST_API, so the functions under test are production's own.
 * The normal host build never defines that macro and therefore never contains
 * the coli_loader_test_* wrappers this file calls.
 *
 * Loads nothing except the controlled stub runtimes it is explicitly told to
 * preload, by absolute path. No PATH change, no System32, no real ROCm.
 *
 * usage: test_loader_helpers.exe <mode> [args...]
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#define SLOT 4096
#define MAXI 8

int coli_loader_test_same_file(const wchar_t *a, const wchar_t *b);
int coli_loader_test_identity_equal_synthetic(int, int, int, int, int, int);
int coli_loader_test_final_path(const wchar_t *path, char *out, size_t cap);
int coli_loader_test_join(const wchar_t *dir, const wchar_t *child, char *out, size_t cap);
int coli_loader_test_exe_path(char *out, size_t cap);
int coli_loader_test_decide(int configured_valid, int inventory_status,
                            int count, const int *match_equality);
int coli_loader_test_configured(const wchar_t *dir, char *path_out, size_t path_cap,
                                char *final_out, size_t final_cap, unsigned long *error);
int coli_loader_test_inventory(const wchar_t *configured_path,
                               int *status, unsigned long *error, int *count,
                               int max_items, int slot,
                               char *paths, char *finals,
                               unsigned long long *modules, int *equal_to_configured);

/* "@N" means: take the real value from COLI_TEST_ARGN with
 * GetEnvironmentVariableW. Narrow argv is converted by Windows to the console
 * codepage before this program ever sees it, which destroys any character that
 * codepage lacks -- so a Unicode path cannot be passed positionally. The
 * environment is the same Unicode source production reads from, so routing
 * those arguments through it tests the helpers rather than argv. */
static wchar_t *wide(const char *s)
{
    static wchar_t bufs[8][SLOT];
    static int turn = 0;
    wchar_t *w = bufs[turn++ % 8];
    if (s[0] == '@' && s[1] >= '0' && s[1] <= '9' && s[2] == '\0') {
        wchar_t name[] = L"COLI_TEST_ARG0";
        name[13] = (wchar_t)s[1];
        w[0] = L'\0';
        GetEnvironmentVariableW(name, w, SLOT);
        return w;
    }
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w, SLOT);
    return w;
}

/* The prefix keeps the before- and after-preload dumps apart; identical keys
 * would silently overwrite each other in the caller's parsed output. */
static void dump_inventory(const char *prefix, const wchar_t *configured)
{
    char paths[MAXI][SLOT], finals[MAXI][SLOT];
    unsigned long long mods[MAXI];
    int eq[MAXI], status = -1, count = -1, i;
    unsigned long err = 0;
    memset(paths, 0, sizeof paths);
    memset(finals, 0, sizeof finals);
    coli_loader_test_inventory(configured, &status, &err, &count,
                               MAXI, SLOT, &paths[0][0], &finals[0][0], mods, eq);
    printf("%s_status=%d\n", prefix, status);
    printf("%s_error=%lu\n", prefix, err);
    printf("%s_count=%d\n", prefix, count);
    for (i = 0; i < count && i < MAXI; i++) {
        printf("%s_%d_path=%s\n", prefix, i, paths[i]);
        printf("%s_%d_final=%s\n", prefix, i, finals[i]);
        printf("%s_%d_module=%llu\n", prefix, i, mods[i]);
        printf("%s_%d_equal=%d\n", prefix, i, eq[i]);
    }
}

int main(int argc, char **argv)
{
    const char *mode = (argc > 1) ? argv[1] : "";
    char buf[SLOT], buf2[SLOT];

    /* The enum values themselves, so a rename in production cannot leave the
     * tests quietly asserting stale numbers. */
    if (!strcmp(mode, "enums")) {
        int eq1 = 1;
        printf("need_exact_load=%d\n",         coli_loader_test_decide(1, 0, 0, NULL));
        printf("reuse_exact_match=%d\n",       coli_loader_test_decide(1, 0, 1, &eq1));
        { int e0 = 0; printf("reject_wrong_runtime=%d\n", coli_loader_test_decide(1, 0, 1, &e0)); }
        { int two[2] = {1, 1}; printf("reject_multiple_runtimes=%d\n", coli_loader_test_decide(1, 0, 2, two)); }
        printf("reject_unverifiable=%d\n",     coli_loader_test_decide(0, 0, 0, NULL));
        return 0;
    }

    /* decide <configured_valid> <inventory_status> <count> [equality ...] */
    if (!strcmp(mode, "decide")) {
        int eq[MAXI], n = argc - 5, i;
        if (n < 0) n = 0;
        if (n > MAXI) n = MAXI;
        for (i = 0; i < n; i++) eq[i] = atoi(argv[5 + i]);
        printf("decision=%d\n", coli_loader_test_decide(
            atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), n ? eq : NULL));
        return 0;
    }

    if (!strcmp(mode, "identity")) {
        printf("same=%d\n", coli_loader_test_same_file(wide(argv[2]), wide(argv[3])));
        return 0;
    }

    /* a_idi a_bh b_idi b_bh same_idi same_bh -> the symmetric fallback, driven
     * directly instead of hoping the filesystem makes one API fail. */
    if (!strcmp(mode, "synthetic")) {
        printf("same=%d\n", coli_loader_test_identity_equal_synthetic(
            atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),
            atoi(argv[5]), atoi(argv[6]), atoi(argv[7])));
        return 0;
    }

    if (!strcmp(mode, "join")) {
        printf("ok=%d\n", coli_loader_test_join(wide(argv[2]), wide(argv[3]), buf, sizeof buf));
        printf("join=%s\n", buf);
        return 0;
    }

    if (!strcmp(mode, "exepath")) {
        printf("ok=%d\n", coli_loader_test_exe_path(buf, sizeof buf));
        printf("exe=%s\n", buf);
        return 0;
    }

    if (!strcmp(mode, "final")) {
        printf("ok=%d\n", coli_loader_test_final_path(wide(argv[2]), buf, sizeof buf));
        printf("final=%s\n", buf);
        return 0;
    }

    if (!strcmp(mode, "configured")) {
        unsigned long err = 0;
        printf("valid=%d\n", coli_loader_test_configured(wide(argv[2]), buf, sizeof buf,
                                                         buf2, sizeof buf2, &err));
        printf("cfg_path=%s\n", buf);
        printf("cfg_final=%s\n", buf2);
        printf("cfg_error=%lu\n", err);
        return 0;
    }

    /* inventory <configured-or-NONE> [absolute preload ...] */
    if (!strcmp(mode, "inventory")) {
        const wchar_t *configured = strcmp(argv[2], "NONE") ? wide(argv[2]) : NULL;
        int i;
        dump_inventory("before", configured);
        for (i = 3; i < argc; i++) {
            HMODULE h = LoadLibraryExW(wide(argv[i]), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
            int (*marker)(void);
            printf("preload_%d_ok=%d\n", i - 3, h != NULL);
            if (!h) { printf("preload_%d_error=%lu\n", i - 3, GetLastError()); continue; }
            printf("preload_%d_module=%llu\n", i - 3, (unsigned long long)(UINT_PTR)h);
            marker = (int (*)(void))(void *)GetProcAddress(h, "coli_test_runtime_marker");
            if (marker) printf("preload_%d_marker=%d\n", i - 3, marker());
        }
        dump_inventory("after", configured);
        return 0;
    }

    printf("unknown_mode=%s\n", mode);
    return 2;
}
'''

    def _gcc(self, args, what):
        cmd = ["gcc"] + args
        # errors="replace": the toolchain speaks the console codepage, which is
        # not always decodable as the Python default on localized hosts.
        proc = subprocess.run(cmd, text=True, errors="replace",
                              capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise FixtureBuildError(
                "%s failed (rc=%d)\ncommand: %s\nstdout:\n%s\nstderr:\n%s"
                % (what, proc.returncode, " ".join(cmd), proc.stdout, proc.stderr))
        return proc

    def _build_runtime(self, source, out_dir, marker, name):
        # Generated sources stay strictly ASCII: they are fed to a toolchain
        # whose input encoding follows the console codepage.
        source.write_text(
            "/* generated stub runtime: no HIP/ROCm/CUDA header, no GPU work,\n"
            " * no DllMain, just a marker the fake backend can read back. */\n"
            "__declspec(dllexport) int coli_test_runtime_marker(void)\n"
            "{ return 0x%02X; }\n" % marker, encoding="ascii")
        implib = out_dir / "libamdhip64_7.a"
        self._gcc(["-O0", "-shared", str(source), "-o", str(out_dir / _RUNTIME_BASENAME),
                   "-Wl,--out-implib," + str(implib)], "building runtime stub " + name)
        return implib

    def _build_dep(self, source, out_dir, with_extra, name):
        lines = [
            "/* generated test dependency: not a GPU component, no DllMain,",
            " * no vendor runtime. Exists only so a backend can fail to load",
            " * for a precisely known reason. */",
            "__declspec(dllexport) int %s(void) { return 1; }" % _DEP_PROBE,
        ]
        if with_extra:
            lines.append("__declspec(dllexport) int %s(void) { return 2; }" % _DEP_EXTRA)
        source.write_text("\n".join(lines) + "\n", encoding="ascii")
        implib = out_dir / "libcoli_test_dep.a"
        self._gcc(["-O0", "-shared", str(source), "-o", str(out_dir / _DEP_BASENAME),
                   "-Wl,--out-implib," + str(implib)],
                  "building test dependency " + name)
        return implib

    def _backend_source(self, with_dep=False):
        real = {"coli_cuda_init", "coli_cuda_e8_set_grid"}
        lines = [
            "/* generated fake backend: no HIP/ROCm/CUDA header, no GPU work,",
            " * no DllMain. Imports the marker from amdhip64_7.dll by basename",
            " * so the Windows loader binds it exactly as the real DLL would. */",
            "__declspec(dllimport) int coli_test_runtime_marker(void);",
            "",
            "/* Test-only accessor. Not part of the production ABI; the loader",
            " * never resolves it. */",
            "__declspec(dllexport) int %s(void)" % _TEST_ACCESSOR,
            "{ return coli_test_runtime_marker(); }",
            "",
            "/* init returns a plain success signal, never the marker, because",
            " * colibri.c treats this return value as a control signal. */",
            "__declspec(dllexport) int coli_cuda_init(const int *devices, int count)",
            "{ (void)devices; (void)count; (void)coli_test_runtime_marker(); return 1; }",
            "",
            "__declspec(dllexport) int coli_cuda_e8_set_grid(const void *grid)",
            "{ (void)grid; return 1; }",
            "",
            "/* Remaining loader-required exports: never called on the startup",
            " * path under test, so trivial bodies are enough to let symbol",
            " * resolution complete. */",
        ]
        if with_dep:
            lines[3:3] = [
                "",
                "/* Second import, from a uniquely named test-only DLL. Whether",
                " * that DLL is absent or merely lacks this entry point decides",
                " * which Windows loader error the backend fails with. */",
                "__declspec(dllimport) int %s(void);" % _DEP_EXTRA,
                "__declspec(dllexport) int coli_test_dep_touch(void)",
                "{ return %s(); }" % _DEP_EXTRA,
            ]
        for name in self.exports:
            if name not in real:
                lines.append("__declspec(dllexport) int %s(void) { return 0; }" % name)
        return "\n".join(lines) + "\n"

    def _build(self):
        try:
            implib_a = self._build_runtime(self.runtime_a_src, self.runtime_a_dir,
                                           _RUNTIME_MARKER_A, "A")
            self._build_runtime(self.runtime_b_src, self.runtime_b_dir,
                                _RUNTIME_MARKER_B, "B")
            self.backend_src.write_text(self._backend_source(), encoding="ascii")
            self._gcc(["-O0", "-shared", str(self.backend_src),
                       "-o", str(self.backend),
                       "-L" + str(implib_a.parent), "-lamdhip64_7"],
                      "building fake coli_hip.dll")
            shutil.copy2(self.backend, self.cuda_backend)
            self._build_diagnostic_variants(implib_a)
            self._build_harness()
        except Exception:
            self.cleanup()
            raise

    def _build_diagnostic_variants(self, implib_a):
        """Artifacts that make each backend-load failure class deterministic.

        Nothing here depends on the developer's machine: the invalid-PE file is
        generated text, and the two dependency builds are ordinary MinGW DLLs
        with test-only names. No PATH is touched and no real vendor runtime is
        involved — runtime A is still preloaded in the dependency cases, so
        amdhip64_7.dll is provably not the cause of the failure.
        """
        self.bad_pe.write_text(
            "not a portable executable: generated by the loader tests\n",
            encoding="ascii")
        dep_implib = self._build_dep(self.dep_full_src, self.dep_full_dir,
                                     True, "full")
        self._build_dep(self.dep_lean_src, self.dep_lean_dir, False, "lean")
        self.backend_dep_src.write_text(self._backend_source(with_dep=True),
                                        encoding="ascii")
        self._gcc(["-O0", "-shared", str(self.backend_dep_src),
                   "-o", str(self.backend_dep),
                   "-L" + str(implib_a.parent), "-lamdhip64_7",
                   "-L" + str(dep_implib.parent), "-lcoli_test_dep"],
                  "building the dependency-carrying backend variant")

    def _build_harness(self):
        """Compile the harness together with the repository's backend_loader.c.

        The production source is compiled *in place* from c/, never copied into
        the fixture, so the harness always exercises the loader at HEAD.

        Two variants are built from that one source. They differ by exactly the
        macro the Makefile's two mutually exclusive builds differ by, which is
        what lets the CUDA variant serve as the control for "COLI_HIP_RUNTIME_DIR
        must not reach a CUDA host".
        """
        self.harness_src.write_text(self.HARNESS_SOURCE, encoding="ascii")
        base = ["-O0", "-DCOLI_CUDA", "-I" + str(HERE),
                str(self.harness_src), str(HERE / "backend_loader.c")]
        self.harness_cmd = base[:1] + ["-DCOLI_HIP_DLL"] + base[1:] + \
            ["-o", str(self.harness_exe)]
        self.cuda_harness_cmd = base + ["-o", str(self.cuda_harness_exe)]
        self._gcc(self.harness_cmd, "building the native loader harness")
        self._gcc(self.cuda_harness_cmd,
                  "building the native loader harness (CUDA mode)")

        # The helper exerciser is the only build that defines
        # COLI_LOADER_TEST_API; the two harnesses above and the shipped host
        # do not, so the test wrappers exist nowhere else.
        self.helper_src.write_text(self.HELPER_SOURCE, encoding="ascii")
        self.helper_cmd = [
            "-O0", "-DCOLI_CUDA", "-DCOLI_HIP_DLL", "-DCOLI_LOADER_TEST_API",
            "-I" + str(HERE), str(self.helper_src), str(HERE / "backend_loader.c"),
            "-o", str(self.helper_exe),
        ]
        self._gcc(self.helper_cmd, "building the loader helper exerciser")

    # --- inspection (objdump only; nothing is ever loaded) -------------

    @staticmethod
    def objdump(path, flag):
        proc = subprocess.run(["objdump", flag, str(path)], text=True,
                              errors="replace", capture_output=True, timeout=120)
        if proc.returncode != 0:
            raise FixtureBuildError("objdump %s failed on %s:\n%s"
                                    % (flag, path, proc.stderr))
        return proc.stdout

    @classmethod
    def architecture(cls, path):
        for line in cls.objdump(path, "-f").splitlines():
            if "architecture:" in line:
                return line.split("architecture:")[1].split(",")[0].strip()
        return ""

    @classmethod
    def imported_dlls(cls, path):
        return [m.strip().lower() for m in
                re.findall(r"DLL Name:\s*(\S+)", cls.objdump(path, "-p"))]

    @classmethod
    def imported_symbols(cls, path):
        return set(re.findall(r"\bcoli_[a-z0-9_]+", cls.objdump(path, "-p")))

    @classmethod
    def exported_names(cls, path):
        out = cls.objdump(path, "-p")
        block = out.split("Export Address Table")[-1]
        return set(re.findall(r"\b(coli_[a-z0-9_]+)\b", block))

    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def generated_paths(self):
        return [self.runtime_a, self.runtime_b, self.backend, self.cuda_backend,
                self.runtime_a_src, self.runtime_b_src, self.backend_src,
                self.harness_src, self.harness_exe, self.cuda_harness_exe,
                self.runtime_a_dir / "libamdhip64_7.a",
                self.runtime_b_dir / "libamdhip64_7.a",
                self.dep_full, self.dep_lean, self.dep_full_src,
                self.dep_lean_src, self.backend_dep_src, self.backend_dep,
                self.bad_pe, self.dep_full_dir / "libcoli_test_dep.a",
                self.helper_src, self.helper_exe]

    def run_helper(self, *args, cwd=None):
        """Run the helper exerciser in its own process and parse key=value output.

        No DLL is ever loaded into the Python interpreter: every LoadLibraryExW
        happens inside this short-lived subprocess, which has exited before the
        caller inspects the result.

        Any argument containing non-ASCII is hoisted into COLI_TEST_ARG<n> and
        replaced by the token ``@n``. Windows converts a narrow argv through the
        console codepage, which silently mangles characters that codepage
        cannot represent; the environment carries them intact, and it is what
        the production loader reads anyway.
        """
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("COLI_TEST_ARG")}
        argv = []
        for arg in args:
            text = str(arg)
            if not text.isascii():
                slot = len(env.keys() & {"COLI_TEST_ARG%d" % i for i in range(10)})
                env["COLI_TEST_ARG%d" % slot] = text
                text = "@%d" % slot
            argv.append(text)
        # encoding="utf-8": the helper converts every path with
        # WideCharToMultiByte(CP_UTF8, ...), so its output is UTF-8 by
        # construction. Decoding it as the locale codepage instead would mangle
        # exactly the non-ASCII paths these tests exist to check.
        proc = subprocess.run(
            [str(self.helper_exe)] + argv, env=env,
            cwd=str(cwd) if cwd else str(self.harness_dir),
            text=True, encoding="utf-8", errors="replace",
            capture_output=True, timeout=120)
        fields = {}
        for line in (proc.stdout or "").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                fields[key.strip()] = value.strip()
        return proc, fields

    def run_harness(self, case_dir, preload, env=None, vendor="hip",
                    with_backend=True, backend_override=None, extra_files=()):
        """Run the harness in its own sandbox, in a separate native process.

        The harness and the fake backend are co-located because the production
        loader resolves its backend DLL beside the executable; the runtimes stay
        outside that directory so only an explicit preload can bind them. The
        process exits before the caller inspects the result, so no module
        handle survives to block TemporaryDirectory cleanup on Windows.

        ``env`` maps variable names to values for this case only. The child
        always starts from a copy of os.environ with COLI_HIP_RUNTIME_DIR
        *removed*, so an ambient value on the developer's machine can never
        turn a "variable missing" case green; passing "" therefore means a
        genuinely set-but-empty variable, which Windows reports distinctly.

        ``with_backend=False`` leaves the sandbox without a backend DLL, for
        cases that only need the loader's own miss path. ``backend_override``
        installs some other file — a variant DLL, or something that is not a PE
        at all — under the basename the loader looks for. ``extra_files`` are
        copied in beside it, which is how a dependency is made present or
        absent without touching PATH or System32.
        """
        case_dir.mkdir(parents=True, exist_ok=True)
        source_exe = self.harness_exe if vendor == "hip" else self.cuda_harness_exe
        backend = self.backend if vendor == "hip" else self.cuda_backend
        exe = case_dir / source_exe.name
        shutil.copy2(source_exe, exe)
        preloads = [preload] if isinstance(preload, (str, Path)) else list(preload)
        if with_backend:
            shutil.copy2(backend_override or backend, case_dir / backend.name)
        for extra in extra_files:
            shutil.copy2(extra, case_dir / Path(extra).name)
        child = {k: v for k, v in os.environ.items()
                 if k != _RUNTIME_DIR_VAR}
        child.update(env or {})
        # encoding="utf-8": the harness converts every path with
        # WideCharToMultiByte(CP_UTF8, ...), so a Unicode sandbox path survives
        # only if it is decoded as UTF-8 rather than the locale codepage.
        proc = subprocess.run(
            [str(exe)] + [str(p) for p in preloads],
            cwd=str(case_dir), text=True, env=child,
            encoding="utf-8", errors="replace", capture_output=True, timeout=120)
        fields = {}
        for line in (proc.stdout or "").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                fields[key.strip()] = value.strip()
        return proc, fields

    def cleanup(self):
        self._tmp.cleanup()


def _fixture_toolchain_skip():
    """Honest skip reason, or None when the fixture can be built here."""
    if sys.platform != "win32":
        return "stub fixtures target the Windows loader"
    for tool in ("gcc", "objdump"):
        if shutil.which(tool) is None:
            return "MinGW %s (same MSYS2 toolchain as the loader tests) not found" % tool
    return None


class LoaderStubFixtureTest(unittest.TestCase):
    """Structural proof that the stub fixtures are what later slices assume.

    These assertions never load a DLL: everything is read with objdump. A
    failure here means the fixture is broken, not that the production loader
    is — which is exactly why they live apart from the contract tests.
    """

    fixture = None

    @classmethod
    def setUpClass(cls):
        reason = _fixture_toolchain_skip()
        if reason:
            raise unittest.SkipTest(reason)
        # Compiled once per class; individual tests only inspect the result.
        cls.fixture = _StubFixture()

    @classmethod
    def tearDownClass(cls):
        if cls.fixture is not None:
            cls.fixture.cleanup()
            cls.fixture = None

    def test_abi_is_derived_from_the_loader_source(self):
        """47 mandatory + 3 optional, parsed from backend_loader.c.

        The counts are a deliberate tripwire: adding a RESOLVE to the loader
        widens the ABI every Windows DLL must satisfy, and that should be a
        conscious act rather than something noticed by a user. Updating them is
        the intended response, not a nuisance -- but name the symbol you added
        below, so the next person reading a failure gets the reason and not
        just a different integer.
        """
        f = self.fixture
        self.assertEqual(len(f.mandatory), 47)
        self.assertEqual(len(f.optional), 3)
        self.assertEqual(len(f.exports), 50)
        self.assertEqual(len(f.exports), len(f.mandatory) + len(f.optional))
        self.assertIn("coli_cuda_init", f.mandatory)
        self.assertIn("coli_cuda_e8_set_grid", f.optional)
        # attention_project_ragged: paged ragged KV runtime (#795).
        self.assertIn("coli_cuda_attention_project_ragged", f.mandatory)
        # fp8_set_lut: fmt=8 e4m3 dense/expert kernels (#817).
        self.assertIn("coli_cuda_fp8_set_lut", f.optional)

    def test_both_runtimes_exist_with_the_production_basename(self):
        """Same basename, different directories — the conflict precondition."""
        f = self.fixture
        for path in (f.runtime_a, f.runtime_b):
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.name, _RUNTIME_BASENAME)
        self.assertNotEqual(f.runtime_a.parent, f.runtime_b.parent)

    def test_runtime_sources_and_binaries_differ(self):
        """Distinguishable by construction and on disk.

        Structural only: proving a *call* returns 0xA1 or 0xB2 needs a process
        that loads them, which is the next slice's harness.
        """
        f = self.fixture
        src_a = f.runtime_a_src.read_text(encoding="ascii")
        src_b = f.runtime_b_src.read_text(encoding="ascii")
        self.assertIn("0x%02X" % _RUNTIME_MARKER_A, src_a)
        self.assertIn("0x%02X" % _RUNTIME_MARKER_B, src_b)
        self.assertNotIn("0x%02X" % _RUNTIME_MARKER_B, src_a)
        self.assertNotEqual(f.sha256(f.runtime_a), f.sha256(f.runtime_b))

    def test_both_runtimes_export_the_marker(self):
        f = self.fixture
        for path in (f.runtime_a, f.runtime_b):
            self.assertIn("coli_test_runtime_marker", f.exported_names(path))

    def test_backend_imports_the_runtime_marker_by_basename(self):
        """The stub must bind amdhip64_7.dll the way the real backend does."""
        f = self.fixture
        self.assertIn(_RUNTIME_BASENAME.lower(), f.imported_dlls(f.backend))
        self.assertIn("coli_test_runtime_marker", f.imported_symbols(f.backend))

    def test_backend_exports_the_complete_derived_abi(self):
        """Every loader-required symbol, plus exactly one test-only accessor."""
        f = self.fixture
        exported = f.exported_names(f.backend)
        missing = sorted(set(f.exports) - exported)
        self.assertEqual(missing, [], "missing production exports: %s" % missing)
        self.assertIn(_TEST_ACCESSOR, exported)
        extra = sorted(exported - set(f.exports) - {_TEST_ACCESSOR})
        self.assertEqual(extra, [], "unexpected extra exports: %s" % extra)

    def test_exported_names_are_undecorated(self):
        f = self.fixture
        for name in f.exported_names(f.backend):
            self.assertNotIn("@", name)
            self.assertFalse(name.startswith("_"), name)

    def test_all_artifacts_are_pe_x86_64(self):
        f = self.fixture
        for path in (f.runtime_a, f.runtime_b, f.backend):
            self.assertEqual(f.architecture(path), "i386:x86-64", path)

    def test_no_real_gpu_runtime_is_imported(self):
        """The deliberate amdhip64_7.dll stub is allowed; real GPU stacks are not."""
        f = self.fixture
        for path in (f.runtime_a, f.runtime_b, f.backend):
            for dll in f.imported_dlls(path):
                if dll == _RUNTIME_BASENAME.lower():
                    continue  # intentional: the stub uses the production basename
                self.assertFalse(
                    any(bad in dll for bad in _FORBIDDEN_IMPORT_MARKERS),
                    "%s imports a real GPU component: %s" % (path.name, dll))
                self.assertTrue(
                    dll.startswith(_ALLOWED_IMPORT_PREFIXES),
                    "%s has an unexpected import: %s" % (path.name, dll))

    def test_harness_is_built_from_the_repository_loader_source(self):
        """The harness links c/backend_loader.c in place, never a copy."""
        f = self.fixture
        self.assertTrue(f.harness_exe.is_file(), f.harness_exe)
        self.assertEqual(f.architecture(f.harness_exe), "i386:x86-64")
        cmd = " ".join(f.harness_cmd)
        self.assertIn(str(HERE / "backend_loader.c"), cmd)
        self.assertIn("-DCOLI_CUDA", cmd)
        self.assertIn("-DCOLI_HIP_DLL", cmd)
        # The CUDA control is the same source and the same loader, minus the
        # one macro; otherwise it would not be a control.
        cuda = " ".join(f.cuda_harness_cmd)
        self.assertTrue(f.cuda_harness_exe.is_file(), f.cuda_harness_exe)
        self.assertIn(str(HERE / "backend_loader.c"), cuda)
        self.assertIn("-DCOLI_CUDA", cuda)
        self.assertNotIn("-DCOLI_HIP_DLL", cuda)
        # A copied loader would let the test drift from production.
        copies = list(f.root.rglob("backend_loader.c"))
        self.assertEqual(copies, [], "loader source copied into the fixture: %s" % copies)

    def test_harness_imports_no_gpu_runtime(self):
        """The harness itself binds nothing vendor-specific."""
        f = self.fixture
        for dll in f.imported_dlls(f.harness_exe):
            self.assertFalse(any(bad in dll for bad in _FORBIDDEN_IMPORT_MARKERS),
                             "harness imports a real GPU component: %s" % dll)

    def test_fixture_root_contains_a_space_and_avoids_the_repo(self):
        """Quoting bugs must surface here, not on a user's Program Files install."""
        f = self.fixture
        self.assertIn(" ", str(f.root))
        self.assertIn(" ", str(f.runtime_a))
        repo_c = str(HERE.resolve()).lower()
        for path in f.generated_paths():
            self.assertFalse(str(path.resolve()).lower().startswith(repo_c),
                             "generated artifact inside the repo: %s" % path)


class LoaderStubFixtureCleanupTest(unittest.TestCase):
    """Explicit cleanup, not interpreter shutdown, removes every artifact."""

    def test_cleanup_removes_every_generated_artifact(self):
        reason = _fixture_toolchain_skip()
        if reason:
            self.skipTest(reason)
        fixture = _StubFixture()
        paths = fixture.generated_paths()
        root = fixture.root
        for path in paths:
            self.assertTrue(path.exists(), "not built: %s" % path)
        fixture.cleanup()
        self.assertFalse(root.exists(), "temporary root survived: %s" % root)
        for path in paths:
            self.assertFalse(path.exists(), "artifact survived cleanup: %s" % path)
        # The repository must be untouched by a fixture that never targets it.
        self.assertFalse((HERE / "coli_hip.dll").exists())
        self.assertFalse((HERE / _RUNTIME_BASENAME).exists())


class LoaderRuntimeBindingTest(unittest.TestCase):
    """Which amdhip64_7.dll does the real loader end up bound to?

    Every case runs the native harness — which links c/backend_loader.c
    unmodified — in its own process and its own executable-directory sandbox.
    The stub DLLs are never loaded into this Python process, and each
    subprocess has exited before cleanup runs.
    """

    fixture = None

    @classmethod
    def setUpClass(cls):
        reason = _fixture_toolchain_skip()
        if reason:
            raise unittest.SkipTest(reason)
        cls.fixture = _StubFixture()

    @classmethod
    def tearDownClass(cls):
        if cls.fixture is not None:
            cls.fixture.cleanup()
            cls.fixture = None

    def setUp(self):
        self.cases = tempfile.TemporaryDirectory(prefix="coli loader case ")
        self.addCleanup(self.cases.cleanup)

    @staticmethod
    def _same_path(a, b):
        return os.path.normcase(os.path.normpath(a)) == \
               os.path.normcase(os.path.normpath(b))

    def _assert_bound_to(self, name, runtime, marker, other):
        f = self.fixture
        case = Path(self.cases.name) / name
        # W1-B2c1: a HIP host must be configured before the loader will run at
        # all, so every binding case now names the directory it preloads from.
        # Validation does not load anything, so what is proven below is
        # unchanged.
        proc, out = f.run_harness(case, str(runtime),
                                  env={_RUNTIME_DIR_VAR: str(runtime.parent)})
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr)

        self.assertEqual(proc.returncode, 0, "harness failed" + detail)
        self.assertEqual(out.get("preload_requested"), "1", detail)
        self.assertEqual(out.get("preload_0_ok"), "1", detail)
        self.assertEqual(out.get("preload_0_marker"), str(marker), detail)
        self.assertTrue(self._same_path(out["preload_0_path"], str(runtime)), detail)

        self.assertEqual(out.get("loader_init"), "1", "loader init failed" + detail)
        self.assertTrue(self._same_path(out["backend_path"],
                                        str(case / f.backend.name)), detail)
        self.assertEqual(out.get("bound_marker"), str(marker),
                         "bound the wrong runtime" + detail)

        self.assertEqual(out.get("runtime_after_preload_count"), "1", detail)
        self.assertEqual(out.get("runtime_after_backend_count"), "1", detail)
        loaded = out["runtime_after_backend_0"]
        self.assertTrue(self._same_path(loaded, str(runtime)), detail)
        self.assertFalse(self._same_path(loaded, str(other)),
                         "the other runtime was bound" + detail)
        return out

    def test_preloaded_runtime_a_is_the_one_bound(self):
        """Preload A -> the backend's import resolves to A (0xA1)."""
        f = self.fixture
        out = self._assert_bound_to("case_a", f.runtime_a,
                                    _RUNTIME_MARKER_A, f.runtime_b)
        self.assertEqual(out.get("runtime_before_count"), "0")

    def test_preloaded_runtime_b_wins_over_the_linked_runtime(self):
        """Preload B -> B is bound, although the backend was linked against A.

        The fake backend imports amdhip64_7.dll by basename and was linked
        against runtime A's import library, yet the already-loaded same-basename
        runtime B satisfies that import. Same-process preload therefore decides
        the binding, which is exactly why a wrong preloaded runtime has to be
        detected before production loading proceeds.
        """
        f = self.fixture
        out = self._assert_bound_to("case_b", f.runtime_b,
                                    _RUNTIME_MARKER_B, f.runtime_a)
        self.assertEqual(out.get("runtime_before_count"), "0")

    def test_no_preload_is_now_deterministic_because_production_loads_it(self):
        """Nothing preloaded: production loads the configured runtime itself.

        Before enforcement this case was environment-dependent and had to be
        classified rather than asserted. It no longer is: the loader preloads
        exactly the configured file by absolute path, so the outcome is fixed
        on any machine — including one with a system-wide amdhip64_7.dll,
        which is never consulted.

        The post-shutdown inventory is the ownership proof: the harness holds
        no reference here, so releasing production's must drop the count to 0.
        """
        f = self.fixture
        case = Path(self.cases.name) / "case_no_preload"
        proc, out = f.run_harness(case, "NONE",
                                  env={_RUNTIME_DIR_VAR: str(f.runtime_a_dir)})
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr)

        self.assertEqual(proc.returncode, 0, "harness crashed" + detail)
        self.assertEqual(out.get("preload_requested"), "0", detail)
        self.assertEqual(out.get("runtime_before_count"), "0", detail)
        self.assertEqual(out.get("runtime_after_preload_count"), "0", detail)

        self.assertEqual(out.get("loader_init"), "1", detail)
        self.assertEqual(out.get("bound_marker"), str(_RUNTIME_MARKER_A), detail)
        self.assertEqual(out.get("runtime_after_backend_count"), "1", detail)
        self.assertTrue(self._same_path(out["runtime_after_backend_0"],
                                        str(f.runtime_a)), detail)
        self.assertFalse(self._same_path(out["runtime_after_backend_0"],
                                         str(f.runtime_b)), detail)
        self.assertEqual(out.get("shutdown_called"), "1", detail)
        self.assertEqual(out.get("runtime_after_shutdown_count"), "0",
                         "production did not release its own reference" + detail)

    def test_ambient_environment_is_recorded_without_inference(self):
        """Record whether a machine-wide runtime exists; prove nothing from it."""
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        candidate = system32 / _RUNTIME_BASENAME
        exists = candidate.is_file()
        size = candidate.stat().st_size if exists else 0
        # Existence is an environmental fact, never evidence of binding.
        self.assertIsInstance(exists, bool)
        if exists:
            self.assertGreater(size, 0)


class LoaderRuntimeDirContractTest(unittest.TestCase):
    """COLI_HIP_RUNTIME_DIR: which values a HIP host accepts, and which it refuses.

    Every case runs the native harness, which links ``c/backend_loader.c``
    unmodified — so the parsing under test is production's, not a Python
    re-implementation. No ROCm, no HIP SDK, no GPU and no real runtime are
    involved: the rejections happen before any DLL is touched, and the
    acceptances only prove that a directory holding a file of the right name
    was found.
    """

    fixture = None

    @classmethod
    def setUpClass(cls):
        reason = _fixture_toolchain_skip()
        if reason:
            raise unittest.SkipTest(reason)
        cls.fixture = _StubFixture()

    @classmethod
    def tearDownClass(cls):
        if cls.fixture is not None:
            cls.fixture.cleanup()
            cls.fixture = None

    def setUp(self):
        self.cases = tempfile.TemporaryDirectory(prefix="coli runtime dir ")
        self.addCleanup(self.cases.cleanup)
        self.cfg = Path(self.cases.name) / "cfg"
        self.cfg.mkdir()

        # A real file, offered where a directory is required.
        self.a_file = self.cfg / "not a directory.txt"
        self.a_file.write_text("not a runtime", encoding="ascii")
        # A real directory with no runtime in it.
        self.empty_dir = self.cfg / "empty dir"
        self.empty_dir.mkdir()
        # A directory that *is* named amdhip64_7.dll — the shape a naive
        # existence check would happily accept.
        self.dir_child = self.cfg / "child is a dir"
        (self.dir_child / _RUNTIME_BASENAME).mkdir(parents=True)
        # Never created.
        self.missing_dir = self.cfg / "no such directory"
        # A valid runtime directory whose name is outside ASCII.
        self.unicode_dir = self.cfg / "körtid π"
        self.unicode_dir.mkdir()
        shutil.copy2(self.fixture.runtime_a, self.unicode_dir / _RUNTIME_BASENAME)

    # --- shared assertions ---------------------------------------------

    def _run(self, name, value, preload="NONE"):
        env = None if value is None else {_RUNTIME_DIR_VAR: value}
        return self.fixture.run_harness(
            Path(self.cases.name) / name, preload, env=env)

    def _assert_rejected(self, name, value, expected):
        """The loader refused, said why, and never went near a DLL."""
        f = self.fixture
        proc, out = self._run(name, value)
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr)
        err = proc.stderr or ""

        self.assertEqual(proc.returncode, 0, "harness crashed" + detail)
        self.assertEqual(out.get("loader_init"), "0",
                         "loader ran despite invalid configuration" + detail)
        self.assertEqual(out.get("backend_loaded"), "0",
                         "the backend DLL was loaded anyway" + detail)
        self.assertNotIn("backend_path", out, detail)

        # The specific diagnostic, tagged as HIP, and never a claim that a DLL
        # was looked for or loaded.
        self.assertIn("[HIP] ", err, detail)
        self.assertIn(expected, err, detail)
        self.assertNotIn("coli_hip.dll", err, "the backend was reached" + detail)
        self.assertNotIn("missing symbol", err, detail)

        # Module inventory is identical at all three sampling points, and
        # neither controlled runtime is in it.
        counts = [out.get("runtime_before_count"),
                  out.get("runtime_after_preload_count"),
                  out.get("runtime_after_backend_count")]
        self.assertEqual(len(set(counts)), 1,
                         "runtime inventory changed: %s%s" % (counts, detail))
        for key, loaded in out.items():
            if key.startswith("runtime_") and key.rsplit("_", 1)[-1].isdigit():
                for runtime in (f.runtime_a, f.runtime_b):
                    self.assertFalse(
                        os.path.normcase(os.path.normpath(loaded)) ==
                        os.path.normcase(os.path.normpath(str(runtime))),
                        "a controlled runtime was loaded" + detail)
        return proc, out

    def _assert_accepted(self, name, value):
        """Validation passed and execution moved on to the backend-load phase.

        Deliberately makes no claim about whether that load then succeeds:
        W1-B2c1 does not preload the runtime, so the outcome past this point is
        the pre-existing, environment-dependent behaviour.
        """
        proc, out = self._run(name, value)
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr)
        err = proc.stderr or ""

        self.assertEqual(proc.returncode, 0, "harness crashed" + detail)
        self.assertNotIn(_RUNTIME_DIR_VAR, err,
                         "a valid directory was rejected" + detail)
        # Since enforcement landed this is no longer "we got as far as the
        # backend": a valid directory means the runtime is loaded, verified and
        # bound, so the whole path is deterministic.
        self.assertEqual(out.get("loader_init"), "1", detail)
        self.assertEqual(out.get("backend_loaded"), "1", detail)
        self.assertEqual(out.get("runtime_after_backend_count"), "1", detail)
        return proc, out

    # --- rejected values ------------------------------------------------

    def test_missing_variable_is_refused(self):
        self._assert_rejected("miss", None, "COLI_HIP_RUNTIME_DIR is not set")

    def test_empty_variable_is_refused(self):
        """Set-but-empty is distinct from unset, and says so.

        Windows reports the two differently — GetEnvironmentVariableW returns 0
        with ERROR_ENVVAR_NOT_FOUND for one and a clear error code for the
        other — so collapsing them into one message would lose the difference
        between "you forgot" and "your launcher passed an empty value".
        """
        self._assert_rejected("empty", "", "COLI_HIP_RUNTIME_DIR is empty")

    def test_relative_forms_are_refused(self):
        """Nothing that would be resolved against the current directory."""
        cases = {
            "rel_plain": r"relative\directory",
            "rel_dot": r".\directory",
            "rel_parent": r"..\directory",
            "rel_forward": "relative/directory",
        }
        for name, value in cases.items():
            with self.subTest(value=value):
                self._assert_rejected(
                    name, value, "COLI_HIP_RUNTIME_DIR must be an absolute path")

    def test_drive_relative_path_is_refused(self):
        """``C:runtime`` is relative to that drive's own current directory."""
        self._assert_rejected("drive_rel", "C:runtime",
                              "COLI_HIP_RUNTIME_DIR must be an absolute path")

    def test_rooted_path_without_drive_is_refused(self):
        """``\\runtime`` is relative to the current drive, not absolute."""
        self._assert_rejected("rooted", "\\runtime",
                              "COLI_HIP_RUNTIME_DIR must be an absolute path")

    def test_nonexistent_absolute_directory_is_refused(self):
        self._assert_rejected("gone", str(self.missing_dir),
                              "COLI_HIP_RUNTIME_DIR does not exist")

    def test_file_supplied_as_the_directory_is_refused(self):
        self._assert_rejected("file_as_dir", str(self.a_file),
                              "COLI_HIP_RUNTIME_DIR is not a directory")

    def test_directory_without_the_runtime_is_refused(self):
        self._assert_rejected(
            "no_runtime", str(self.empty_dir),
            "amdhip64_7.dll not found in COLI_HIP_RUNTIME_DIR")

    def test_runtime_child_that_is_a_directory_is_refused(self):
        """Existence alone is not enough: it has to be a file."""
        self._assert_rejected("child_dir", str(self.dir_child),
                              "amdhip64_7.dll is a directory, not a file")

    # --- accepted values ------------------------------------------------

    def test_runtime_a_directory_with_a_space_is_accepted(self):
        f = self.fixture
        self.assertIn(" ", str(f.runtime_a_dir))
        self._assert_accepted("ok_a", str(f.runtime_a_dir))

    def test_runtime_b_directory_is_accepted(self):
        self._assert_accepted("ok_b", str(self.fixture.runtime_b_dir))

    def test_unicode_directory_is_accepted(self):
        """The value is read with GetEnvironmentVariableW, so it survives."""
        self.assertFalse(str(self.unicode_dir).isascii())
        self._assert_accepted("ok_unicode", str(self.unicode_dir))

    def test_trailing_separator_is_accepted(self):
        """No doubled separator when the user's value already ends in one."""
        for name, suffix in (("ok_trail_back", "\\"), ("ok_trail_fwd", "/")):
            with self.subTest(suffix=suffix):
                self._assert_accepted(
                    name, str(self.fixture.runtime_a_dir) + suffix)

    # --- configured vs. actually bound ----------------------------------

    @staticmethod
    def _same_path(a, b):
        return os.path.normcase(os.path.normpath(a)) == \
               os.path.normcase(os.path.normpath(b))

    def test_configured_and_preloaded_runtime_agree(self):
        """Configured A, A already loaded: reused, and only one stays mapped.

        Production takes its own counted reference to the very same module
        rather than mapping a second copy — so the inventory shows one module
        before and after, and after shutdown the harness's own reference is
        what keeps it alive.
        """
        f = self.fixture
        proc, out = f.run_harness(
            Path(self.cases.name) / "agree", str(f.runtime_a),
            env={_RUNTIME_DIR_VAR: str(f.runtime_a_dir)})
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr)
        self.assertNotIn(_RUNTIME_DIR_VAR, proc.stderr or "", detail)
        self.assertEqual(out.get("loader_init"), "1", detail)
        self.assertEqual(out.get("bound_marker"), str(_RUNTIME_MARKER_A), detail)
        # Reused, not duplicated.
        self.assertEqual(out.get("runtime_after_preload_count"), "1", detail)
        self.assertEqual(out.get("runtime_after_backend_count"), "1", detail)
        # Production released only its own reference; the harness still holds one.
        self.assertEqual(out.get("runtime_after_shutdown_count"), "1",
                         "production released a reference it did not own" + detail)
        self.assertTrue(self._same_path(out["runtime_after_shutdown_0"],
                                        str(f.runtime_a)), detail)

    def test_configured_a_with_b_preloaded_fails_closed(self):
        """A wrong runtime is already loaded, so the GPU tier must not start.

        Windows binds an import by base name against whatever is already
        mapped, so with runtime B resident the backend would silently get B no
        matter which file the configuration names. There is no safe way to
        pick a winner here — the only honest outcome is to refuse before
        anything GPU-related is loaded, and leave B exactly as it was found.
        """
        f = self.fixture
        proc, out = f.run_harness(
            Path(self.cases.name) / "wrong_runtime", str(f.runtime_b),
            env={_RUNTIME_DIR_VAR: str(f.runtime_a_dir)})
        err = proc.stderr or ""
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, err)

        self.assertEqual(proc.returncode, 0, "harness crashed" + detail)
        self.assertEqual(out.get("preload_0_marker"), str(_RUNTIME_MARKER_B), detail)

        # Fails closed, before the backend.
        self.assertEqual(out.get("loader_init"), "0", detail)
        self.assertEqual(out.get("backend_loaded"), "0", detail)
        self.assertNotIn("backend_path", out, detail)
        self.assertNotIn("bound_marker", out, detail)

        # B is untouched and A was never loaded.
        self.assertEqual(out.get("runtime_after_backend_count"), "1", detail)
        self.assertTrue(self._same_path(out["runtime_after_backend_0"],
                                        str(f.runtime_b)), detail)
        self.assertFalse(self._same_path(out["runtime_after_backend_0"],
                                         str(f.runtime_a)), detail)
        # Nothing was released either: production never took a reference.
        self.assertEqual(out.get("runtime_after_shutdown_count"), "1", detail)

        # The diagnostic names the mismatch and both sides of it.
        self.assertIn("[HIP] a different amdhip64_7.dll is already loaded", err, detail)
        self.assertIn("[HIP]   configured:", err, detail)
        self.assertIn("[HIP]   already loaded:", err, detail)
        self.assertIn("CPU path remains active", err, detail)
        # It never reached the backend, so it must not talk about one.
        self.assertNotIn("could not be loaded", err, detail)

    def test_two_preloaded_runtimes_fail_closed(self):
        """Two same-basename runtimes resident: ambiguous, so refuse.

        Even though one of them IS the configured file, which module an import
        binds to is decided by load order rather than by us. Neither is
        unloaded — they were not ours to unload.
        """
        f = self.fixture
        proc, out = f.run_harness(
            Path(self.cases.name) / "two_runtimes",
            [str(f.runtime_a), str(f.runtime_b)],
            env={_RUNTIME_DIR_VAR: str(f.runtime_a_dir)})
        err = proc.stderr or ""
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, err)

        self.assertEqual(proc.returncode, 0, "harness crashed" + detail)
        self.assertEqual(out.get("preload_0_marker"), str(_RUNTIME_MARKER_A), detail)
        self.assertEqual(out.get("preload_1_marker"), str(_RUNTIME_MARKER_B), detail)
        self.assertEqual(out.get("runtime_after_preload_count"), "2", detail)

        self.assertEqual(out.get("loader_init"), "0", detail)
        self.assertEqual(out.get("backend_loaded"), "0", detail)
        self.assertNotIn("bound_marker", out, detail)

        # Both survive, untouched, and both remain distinct modules.
        self.assertEqual(out.get("runtime_after_backend_count"), "2", detail)
        self.assertEqual(out.get("runtime_after_shutdown_count"), "2", detail)
        paths = {out["runtime_after_backend_0"].lower(),
                 out["runtime_after_backend_1"].lower()}
        self.assertEqual(len(paths), 2, detail)

        self.assertIn("[HIP] 2 different amdhip64_7.dll modules", err, detail)
        self.assertIn("[HIP]   configured:", err, detail)
        self.assertEqual(err.count("[HIP]   already loaded:"), 2, detail)

    def test_configured_through_a_hardlink_alias_is_accepted(self):
        """Same physical file reached by another name: accept, do not compare text.

        The configuration points at a hard link whose path text differs from
        the module Windows reports, so a textual check would refuse a perfectly
        correct setup. Physical identity says they are one file, and that is
        what decides.
        """
        f = self.fixture
        alias_dir = Path(self.cases.name) / "alias runtime dir"
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias = alias_dir / _RUNTIME_BASENAME
        try:
            if not alias.exists():
                os.link(f.runtime_a, alias)
        except (OSError, NotImplementedError, AttributeError) as exc:
            self.skipTest("hard links unavailable here: %s" % exc)

        # Preload through the canonical name, configure through the alias.
        proc, out = f.run_harness(
            Path(self.cases.name) / "hardlink", str(f.runtime_a),
            env={_RUNTIME_DIR_VAR: str(alias_dir)})
        err = proc.stderr or ""
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, err)

        self.assertEqual(out.get("loader_init"), "1",
                         "a hard link to the configured file was refused" + detail)
        self.assertEqual(out.get("bound_marker"), str(_RUNTIME_MARKER_A), detail)
        self.assertEqual(out.get("runtime_after_backend_count"), "1", detail)
        self.assertNotIn("is not the configured file", err, detail)

    def test_unicode_executable_directory_loads_the_backend(self):
        """The HIP backend path is built wide, so a non-ASCII install works.

        The old ANSI construction went through the active code page and would
        mangle exactly this directory name. The runtime stays in its own plain
        directory, so what is under test is the backend path alone.
        """
        f = self.fixture
        case = Path(self.cases.name) / "körtid π sandbox"
        proc, out = f.run_harness(case, "NONE",
                                  env={_RUNTIME_DIR_VAR: str(f.runtime_a_dir)})
        err = proc.stderr or ""
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, err)

        self.assertFalse(str(case).isascii())
        self.assertEqual(out.get("loader_init"), "1", detail)
        self.assertEqual(out.get("bound_marker"), str(_RUNTIME_MARKER_A), detail)
        # The exact Unicode candidate, round-tripped intact through UTF-8.
        self.assertTrue(self._same_path(out["backend_path"],
                                        str(case / f.backend.name)), detail)
        self.assertIn("körtid π", out["backend_path"], detail)
        self.assertNotIn("?", out["backend_path"], detail)
        self.assertEqual(out.get("runtime_after_backend_count"), "1", detail)
        self.assertNotIn("fallback", err, detail)

    def test_configured_runtime_that_is_not_a_valid_pe_fails_before_the_backend(self):
        """Structurally valid configuration, unloadable runtime: 193, no backend.

        The file exists and is a file, so W1-B2c1 is satisfied; it is only when
        the loader tries to map it that the truth appears. Nothing GPU-related
        is attempted afterwards and nothing stays loaded.
        """
        f = self.fixture
        bad_dir = Path(self.cases.name) / "bad runtime dir"
        bad_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f.bad_pe, bad_dir / _RUNTIME_BASENAME)

        proc, out = f.run_harness(
            Path(self.cases.name) / "bad_runtime", "NONE",
            env={_RUNTIME_DIR_VAR: str(bad_dir)})
        err = proc.stderr or ""
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, err)

        self.assertEqual(out.get("loader_init"), "0", detail)
        self.assertEqual(out.get("backend_loaded"), "0", detail)
        self.assertEqual(out.get("runtime_after_backend_count"), "0", detail)
        self.assertEqual(out.get("runtime_after_shutdown_count"), "0", detail)
        self.assertIn("[HIP] the configured amdhip64_7.dll could not be loaded", err, detail)
        self.assertIn("Windows error 193", err, detail)
        self.assertIn("invalid executable format or architecture", err, detail)
        # The runtime never loaded, so the backend must not be mentioned.
        self.assertNotIn("coli_hip.dll could not be loaded", err, detail)


class _HelperTestBase(unittest.TestCase):
    """Shared fixture for the W1-B2d1 helper exercisers."""

    fixture = None

    @classmethod
    def setUpClass(cls):
        reason = _fixture_toolchain_skip()
        if reason:
            raise unittest.SkipTest(reason)
        cls.fixture = _StubFixture()

    @classmethod
    def tearDownClass(cls):
        if cls.fixture is not None:
            cls.fixture.cleanup()
            cls.fixture = None

    def setUp(self):
        self.cases = tempfile.TemporaryDirectory(prefix="coli helper ")
        self.addCleanup(self.cases.cleanup)


class LoaderFileIdentityTest(_HelperTestBase):
    """Does the loader decide "same file" on physical identity, not path text?

    This is the property that decides whether a correctly configured runtime
    reached by a hard link, a dot segment or a different case is accepted or
    wrongly refused. Every case runs the production comparison through the real
    c/backend_loader.c; nothing here loads a DLL.
    """

    def _same(self, a, b):
        proc, out = self.fixture.run_helper("identity", a, b)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return int(out["same"])

    def test_a_file_is_the_same_file_as_itself(self):
        self.assertEqual(self._same(self.fixture.runtime_a, self.fixture.runtime_a), 1)

    def test_two_distinct_runtimes_differ(self):
        self.assertEqual(self._same(self.fixture.runtime_a, self.fixture.runtime_b), 0)

    def test_hardlink_alias_is_the_same_physical_file(self):
        """Different name, different final path — same file, so: equal.

        A textual comparison fails this, which is precisely why the production
        code does not use one.
        """
        alias = Path(self.cases.name) / "aliased runtime.dll"
        try:
            os.link(self.fixture.runtime_a, alias)
        except (OSError, NotImplementedError, AttributeError) as exc:
            self.skipTest("hard links unavailable here: %s" % exc)
        self.assertEqual(self._same(self.fixture.runtime_a, alias), 1)
        # ...and the diagnostic paths really do disagree, so the test is not
        # accidentally comparing one string with itself.
        _, out_a = self.fixture.run_helper("final", self.fixture.runtime_a)
        _, out_b = self.fixture.run_helper("final", alias)
        self.assertEqual(out_a.get("ok"), "1")
        self.assertEqual(out_b.get("ok"), "1")
        self.assertNotEqual(out_a["final"], out_b["final"])

    def test_dot_dot_alias_is_the_same_physical_file(self):
        f = self.fixture
        dotted = f.runtime_a.parent / "sub" / ".." / f.runtime_a.name
        (f.runtime_a.parent / "sub").mkdir(exist_ok=True)
        self.assertEqual(self._same(f.runtime_a, dotted), 1)

    def test_case_variant_is_the_same_physical_file(self):
        f = self.fixture
        upper = Path(str(f.runtime_a.parent).upper()) / f.runtime_a.name.upper()
        self.assertEqual(self._same(f.runtime_a, upper), 1)

    def test_missing_file_is_unavailable_not_unequal(self):
        """Absent or unreadable must never be reported as "a different file"."""
        gone = Path(self.cases.name) / "no such runtime.dll"
        self.assertEqual(self._same(self.fixture.runtime_a, gone), -1)
        self.assertEqual(self._same(gone, gone), -1)

    def test_identity_representations_are_never_mixed(self):
        """The FILE_ID_INFO / by-handle fallback is symmetric.

        Driven through the pure comparison rather than by hoping some
        filesystem makes one API fail: whenever either side lacks the 128-bit
        ID, *both* sides drop to the 64-bit index, and when no representation
        is shared the answer is "unknown".
        """
        f = self.fixture

        def synth(a_idi, a_bh, b_idi, b_bh, same_idi, same_bh):
            _, out = f.run_helper("synthetic", a_idi, a_bh, b_idi, b_bh,
                                  same_idi, same_bh)
            return int(out["same"])

        # Both have the 128-bit ID: it decides, even when the 64-bit indices
        # would have said otherwise.
        self.assertEqual(synth(1, 1, 1, 1, 1, 0), 1)
        self.assertEqual(synth(1, 1, 1, 1, 0, 1), 0)
        # One side lacks it: both fall back to the by-handle index...
        self.assertEqual(synth(1, 1, 0, 1, 0, 1), 1)
        self.assertEqual(synth(0, 1, 1, 1, 1, 0), 0)
        # ...and a matching 128-bit ID on one side is never used against a
        # 64-bit index on the other.
        self.assertEqual(synth(1, 0, 0, 1, 1, 1), -1)
        self.assertEqual(synth(0, 1, 1, 0, 1, 1), -1)
        # Nothing on either side.
        self.assertEqual(synth(0, 0, 0, 0, 1, 1), -1)


class LoaderPathHelperTest(_HelperTestBase):
    """Unicode path construction with no MAX_PATH anywhere in the contract."""

    def _join(self, dir_, child):
        proc, out = self.fixture.run_helper("join", dir_, child)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(out.get("ok"), "1", proc.stdout)
        return out["join"]

    def test_join_inserts_exactly_one_separator(self):
        self.assertEqual(self._join(r"C:\rocm\bin", "amdhip64_7.dll"),
                         r"C:\rocm\bin\amdhip64_7.dll")

    def test_join_does_not_double_an_existing_separator(self):
        for tail in ("\\", "/"):
            with self.subTest(tail=tail):
                joined = self._join(r"C:\rocm\bin" + tail, "amdhip64_7.dll")
                self.assertTrue(joined.endswith("amdhip64_7.dll"), joined)
                self.assertNotIn("\\\\", joined[2:])
                self.assertNotIn("//", joined)

    def test_join_preserves_spaces_and_unicode(self):
        joined = self._join("C:\\Program Files\\körtid π", "amdhip64_7.dll")
        self.assertEqual(joined, "C:\\Program Files\\körtid π\\amdhip64_7.dll")

    def test_join_handles_a_path_far_beyond_max_path(self):
        """MAX_PATH is not a correctness bound anywhere in the helper."""
        deep = "C:\\" + "\\".join("segment%03d" % i for i in range(80))
        self.assertGreater(len(deep), 260)
        joined = self._join(deep, "amdhip64_7.dll")
        self.assertEqual(joined, deep + "\\amdhip64_7.dll")

    def test_exe_path_is_the_running_helper(self):
        """The growing GetModuleFileNameW buffer returns a usable full path."""
        proc, out = self.fixture.run_helper("exepath")
        self.assertEqual(out.get("ok"), "1", proc.stdout)
        self.assertTrue(out["exe"].lower().endswith("test_loader_helpers.exe"),
                        out["exe"])
        self.assertTrue(Path(out["exe"]).is_file(), out["exe"])

    def test_final_path_is_canonical_and_keeps_its_prefix(self):
        """Documented representation: VOLUME_NAME_DOS, \\\\?\\ prefix left intact.

        Stripping the prefix can change how a path is interpreted, so the
        helper does not, and this pins that decision down.
        """
        f = self.fixture
        _, out = f.run_helper("final", f.runtime_a)
        self.assertEqual(out.get("ok"), "1")
        final = out["final"]
        self.assertTrue(final.startswith("\\\\?\\"), final)
        self.assertTrue(final.lower().endswith(_RUNTIME_BASENAME), final)
        # Canonical: a dot-dot alias of the same file yields the same text.
        (f.runtime_a.parent / "sub").mkdir(exist_ok=True)
        _, dotted = f.run_helper("final", f.runtime_a.parent / "sub" / ".." / f.runtime_a.name)
        self.assertEqual(dotted.get("final"), final)

    def test_configured_runtime_foundation(self):
        """Directory in, absolute runtime path plus identity out."""
        f = self.fixture
        proc, out = f.run_helper("configured", f.runtime_a_dir)
        self.assertEqual(out.get("valid"), "1", proc.stdout)
        self.assertTrue(_same_path(out["cfg_path"],
                                   f.runtime_a_dir / _RUNTIME_BASENAME),
                        "%r != %r" % (out["cfg_path"],
                                      str(f.runtime_a_dir / _RUNTIME_BASENAME)))
        self.assertTrue(out["cfg_final"].lower().endswith(_RUNTIME_BASENAME))
        self.assertEqual(out.get("cfg_error"), "0")

    def test_configured_runtime_reports_a_missing_runtime(self):
        missing = Path(self.cases.name) / "empty dir"
        missing.mkdir()
        proc, out = self.fixture.run_helper("configured", missing)
        self.assertEqual(out.get("valid"), "0", proc.stdout)
        # The path is still built, so a diagnostic can name it.
        self.assertTrue(_same_path(out["cfg_path"], missing / _RUNTIME_BASENAME),
                        "%r != %r" % (out["cfg_path"],
                                      str(missing / _RUNTIME_BASENAME)))
        self.assertNotEqual(out.get("cfg_error"), "0")


class LoaderInventoryTest(_HelperTestBase):
    """What does the process actually have loaded under that base name?

    Every module is loaded inside the helper subprocess, by absolute path, from
    the controlled stub fixtures. No real runtime, no PATH, no System32.
    """

    def _inventory(self, name, configured, *preloads):
        case = Path(self.cases.name) / name
        case.mkdir(parents=True, exist_ok=True)
        proc, out = self.fixture.run_helper(
            "inventory", configured, *preloads, cwd=case)
        self.assertEqual(proc.returncode, 0,
                         "helper crashed\n%s\n%s" % (proc.stdout, proc.stderr))
        self.assertEqual(out.get("before_status"), "0", proc.stdout)
        self.assertEqual(out.get("after_status"), "0", proc.stdout)
        return out

    @staticmethod
    def _same_path(a, b):
        return os.path.normcase(os.path.normpath(a)) == \
               os.path.normcase(os.path.normpath(b))

    def test_nothing_preloaded_is_a_successful_empty_inventory(self):
        """Zero matches, reported as success — not as a failure."""
        f = self.fixture
        out = self._inventory("none", f.runtime_a)
        self.assertEqual(out["before_count"], "0")
        self.assertEqual(out["after_count"], "0")
        self.assertEqual(out["before_error"], "0")

    def test_runtime_a_preloaded_is_seen_once_and_identified(self):
        f = self.fixture
        out = self._inventory("a", f.runtime_a, f.runtime_a)
        self.assertEqual(out["preload_0_marker"], str(_RUNTIME_MARKER_A))
        self.assertEqual(out["after_count"], "1")
        self.assertTrue(self._same_path(out["after_0_path"], str(f.runtime_a)))
        self.assertEqual(out["after_0_equal"], "1")
        self.assertTrue(out["after_0_final"].lower().endswith(_RUNTIME_BASENAME))

    def test_runtime_b_preloaded_is_identified_as_not_the_configured_file(self):
        f = self.fixture
        out = self._inventory("b", f.runtime_a, f.runtime_b)
        self.assertEqual(out["preload_0_marker"], str(_RUNTIME_MARKER_B))
        self.assertEqual(out["after_count"], "1")
        self.assertTrue(self._same_path(out["after_0_path"], str(f.runtime_b)))
        self.assertEqual(out["after_0_equal"], "0")

    def test_two_runtimes_are_both_seen_as_distinct_modules(self):
        """The case a base-name lookup cannot report at all.

        Both are mapped, both are callable, both have their own HMODULE and
        identity — and exactly one of them is the configured file.
        """
        f = self.fixture
        out = self._inventory("ab", f.runtime_a, f.runtime_a, f.runtime_b)
        self.assertEqual(out["preload_0_marker"], str(_RUNTIME_MARKER_A))
        self.assertEqual(out["preload_1_marker"], str(_RUNTIME_MARKER_B))
        self.assertEqual(out["after_count"], "2")
        self.assertNotEqual(out["after_0_module"], out["after_1_module"])
        paths = {out["after_0_path"].lower(), out["after_1_path"].lower()}
        self.assertEqual(len(paths), 2, paths)
        self.assertTrue(out["after_0_final"] and out["after_1_final"])
        self.assertEqual({out["after_0_equal"], out["after_1_equal"]}, {"1", "0"})

    def test_repeated_load_of_one_runtime_is_still_one_module(self):
        """References are not modules: three loads, one inventory entry."""
        f = self.fixture
        out = self._inventory("refs", f.runtime_a,
                              f.runtime_a, f.runtime_a, f.runtime_a)
        for i in range(3):
            self.assertEqual(out["preload_%d_ok" % i], "1")
            self.assertEqual(out["preload_%d_marker" % i], str(_RUNTIME_MARKER_A))
        self.assertEqual(out["after_count"], "1")
        self.assertEqual(out["preload_0_module"], out["preload_2_module"])

    def test_runtime_loaded_through_a_hardlink_still_matches_the_configured_file(self):
        """Configured by one path, loaded through another: physically equal.

        The alias keeps the production base name but lives in a different
        directory, because the inventory matches on base name — a runtime
        loaded under some *other* file name is a different module as far as the
        loaded-module list is concerned, which is a separate question from file
        identity. Here the path text differs and the physical file does not, so
        the comparison must say "same", which a textual test could not.
        """
        f = self.fixture
        alias_dir = Path(self.cases.name) / "alias dir"
        alias_dir.mkdir(exist_ok=True)
        alias = alias_dir / _RUNTIME_BASENAME
        try:
            if not alias.exists():
                os.link(f.runtime_a, alias)
        except (OSError, NotImplementedError, AttributeError) as exc:
            self.skipTest("hard links unavailable here: %s" % exc)
        out = self._inventory("hardlink", f.runtime_a, alias)
        self.assertEqual(out["after_count"], "1")
        self.assertNotEqual(os.path.normcase(out["after_0_path"]),
                            os.path.normcase(str(f.runtime_a)))
        self.assertEqual(out["after_0_equal"], "1",
                         "identity comparison rejected a hard link")


class LoaderBindingDecisionTest(_HelperTestBase):
    """The pure decision: inventory plus configured identity in, one verdict out.

    The helper reads the enum values back out of production rather than
    restating them, so renaming or reordering the enum cannot leave these
    assertions quietly checking stale numbers.
    """

    def setUp(self):
        super().setUp()
        _, self.e = self.fixture.run_helper("enums")

    def test_enum_values_are_distinct(self):
        """Five outcomes must be five values, or the matrix below proves nothing."""
        values = [self.e[k] for k in ("need_exact_load", "reuse_exact_match",
                                      "reject_wrong_runtime",
                                      "reject_multiple_runtimes",
                                      "reject_unverifiable")]
        self.assertEqual(len(set(values)), 5, values)

    def test_decision_matrix(self):
        """Every branch, with exact result assertions."""
        f = self.fixture
        cases = [
            # (label, configured_valid, inv_status, count, equality, expected)
            ("zero matches",             1, 0, 0, [],        "need_exact_load"),
            ("one identical",            1, 0, 1, [1],       "reuse_exact_match"),
            ("one different",            1, 0, 1, [0],       "reject_wrong_runtime"),
            ("one undeterminable",       1, 0, 1, [-1],      "reject_unverifiable"),
            ("two different",            1, 0, 2, [1, 0],    "reject_multiple_runtimes"),
            ("two identical",            1, 0, 2, [1, 1],    "reject_multiple_runtimes"),
            ("configured unavailable",   0, 0, 0, [],        "reject_unverifiable"),
            ("snapshot failure",         1, 1, 0, [],        "reject_unverifiable"),
            ("enumeration failure",      1, 2, 0, [],        "reject_unverifiable"),
            ("allocation failure",       1, 3, 0, [],        "reject_unverifiable"),
            ("module-path failure",      1, 4, 0, [],        "reject_unverifiable"),
            ("module-identity failure",  1, 5, 0, [],        "reject_unverifiable"),
        ]
        for label, valid, status, count, eq, expected in cases:
            with self.subTest(case=label):
                _, out = f.run_helper("decide", valid, status, count, *eq)
                self.assertEqual(out.get("decision"), self.e[expected],
                                 "%s -> %s" % (label, out))

    def test_a_failed_inventory_never_looks_like_zero_matches(self):
        """The distinction the whole status model exists for."""
        f = self.fixture
        _, ok = f.run_helper("decide", 1, 0, 0)
        _, bad = f.run_helper("decide", 1, 1, 0)
        self.assertEqual(ok["decision"], self.e["need_exact_load"])
        self.assertEqual(bad["decision"], self.e["reject_unverifiable"])
        self.assertNotEqual(ok["decision"], bad["decision"])


class LoaderFailureDiagnosticTest(unittest.TestCase):
    """When the backend will not load, does the message say what really happened?

    The loader used to report every ``LoadLibraryEx`` failure as "not found",
    which is wrong the moment the file is present and fails for some other
    reason. Each case below engineers one specific Windows loader error with
    generated artifacts, then reads the diagnostic back out of a real
    subprocess. No GPU, no ROCm, no HIP SDK, no PATH or System32 change.
    """

    fixture = None

    @classmethod
    def setUpClass(cls):
        reason = _fixture_toolchain_skip()
        if reason:
            raise unittest.SkipTest(reason)
        cls.fixture = _StubFixture()

    @classmethod
    def tearDownClass(cls):
        if cls.fixture is not None:
            cls.fixture.cleanup()
            cls.fixture = None

    def setUp(self):
        self.cases = tempfile.TemporaryDirectory(prefix="coli diag ")
        self.addCleanup(self.cases.cleanup)

    # --- shared -------------------------------------------------------

    @staticmethod
    def _errors(err):
        """Every numeric loader code the diagnostic reported, in order."""
        return [int(n) for n in
                re.findall(r"Windows (?:loader )?error (\d+)", err)]

    def _fail(self, name, **kwargs):
        """Run one failing HIP case and assert everything common to all of them."""
        f = self.fixture
        kwargs.setdefault("preload", "NONE")
        kwargs.setdefault("env", {_RUNTIME_DIR_VAR: str(f.runtime_a_dir)})
        proc, out = f.run_harness(Path(self.cases.name) / name, **kwargs)
        err = proc.stderr or ""
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, err)

        self.assertEqual(proc.returncode, 0, "harness crashed" + detail)
        self.assertEqual(out.get("loader_init"), "0", detail)
        self.assertNotIn("backend_path", out, detail)
        # Validation passed, so nothing here is a configuration complaint.
        self.assertNotIn(_RUNTIME_DIR_VAR, err, detail)
        # Vendor and container are named, and the CPU fallback is promised.
        self.assertIn("[HIP] coli_hip.dll could not be loaded", err, detail)
        self.assertIn("CPU path remains active", err, detail)
        self.assertNotIn("coli_cuda.dll", err, detail)
        # HIP has exactly one backend attempt now: the absolute
        # executable-directory candidate. No fallback line, and no second code.
        self.assertNotIn("fallback by name", err, detail)
        self.assertEqual(len(self._errors(err)), 1,
                         "HIP should report exactly one backend error" + detail)
        # Claims the loader cannot support.
        self.assertNotIn("System32", err, detail)
        self.assertNotIn("coli_cuda_", err, detail)
        return proc, out, err, detail

    def _assert_only_preloaded_runtime(self, out, detail, preloaded):
        f = self.fixture
        for key, path in out.items():
            if key.startswith("runtime_") and key.rsplit("_", 1)[-1].isdigit():
                for runtime in (f.runtime_a, f.runtime_b):
                    if runtime == preloaded:
                        continue
                    self.assertNotEqual(
                        os.path.normcase(os.path.normpath(path)),
                        os.path.normcase(os.path.normpath(str(runtime))),
                        "an unexpected controlled runtime was loaded" + detail)

    # --- the four classes ---------------------------------------------

    def test_absent_backend_is_reported_as_absent(self):
        """Nothing at the candidate path: say so, and give the code.

        Production loaded the runtime itself here, so the post-shutdown
        inventory also proves that a failed backend load releases it again.
        """
        proc, out, err, detail = self._fail("absent", with_backend=False)
        self.assertRegex(err, r"candidate .*coli_hip\.dll: does not exist", detail)
        self.assertNotIn("exists as a file", err, detail)
        self.assertEqual(self._errors(err)[0], 126, detail)  # ERROR_MOD_NOT_FOUND
        self.assertIn("module or one of its dependencies was not found", err, detail)
        self.assertEqual(out.get("runtime_after_backend_count"), "0",
                         "the runtime reference leaked past a backend failure" + detail)
        self._assert_only_preloaded_runtime(out, detail, None)

    def test_invalid_pe_is_not_reported_as_missing(self):
        """A non-PE file at the exact path: present, and 193 — never 'absent'."""
        f = self.fixture
        proc, out, err, detail = self._fail(
            "badpe", backend_override=f.bad_pe)
        self.assertRegex(err, r"candidate .*coli_hip\.dll: exists as a file", detail)
        self.assertNotIn("does not exist", err, detail)
        self.assertEqual(self._errors(err)[0], 193, detail)  # ERROR_BAD_EXE_FORMAT
        self.assertIn("invalid executable format or architecture", err, detail)
        self.assertEqual(out.get("runtime_after_backend_count"), "0", detail)

    def test_missing_dependency_is_not_reported_as_missing_backend(self):
        """The backend is right there; its dependency is not. That is 126.

        Runtime A is preloaded, so amdhip64_7.dll provably is not the cause —
        the only unsatisfied import left is the test-only dependency, which is
        deliberately absent from the sandbox.
        """
        f = self.fixture
        proc, out, err, detail = self._fail(
            "missing_dep", preload=str(f.runtime_a),
            backend_override=f.backend_dep)
        self.assertRegex(err, r"candidate .*coli_hip\.dll: exists as a file", detail)
        self.assertNotIn("does not exist", err, detail)
        self.assertEqual(self._errors(err)[0], 126, detail)  # ERROR_MOD_NOT_FOUND
        self.assertIn("module or one of its dependencies was not found", err, detail)
        # The message must not name a culprit Windows never identified.
        self.assertNotIn(_DEP_BASENAME, err, detail)
        self.assertNotIn(_DEP_EXTRA, err, detail)
        self.assertEqual(out.get("preload_0_marker"), str(_RUNTIME_MARKER_A), detail)
        # Production reused the harness's runtime and released only its own
        # reference, so the harness-owned module survives.
        self.assertEqual(out.get("runtime_after_shutdown_count"), "1", detail)
        self._assert_only_preloaded_runtime(out, detail, f.runtime_a)

    def test_missing_procedure_is_reported_as_a_procedure_failure(self):
        """Dependency present but one entry point short: that is 127, not 126.

        The lean build of the test dependency exports only the probe, while the
        backend was linked against the full build and imports the extra entry
        point. Both files exist and are valid PEs, so the failure is precisely
        a missing procedure — constructed entirely from generated artifacts,
        with no reliance on whatever amdhip64_7.dll this machine happens to
        have in System32.
        """
        f = self.fixture
        proc, out, err, detail = self._fail(
            "missing_proc", preload=str(f.runtime_a),
            backend_override=f.backend_dep, extra_files=(f.dep_lean,))
        self.assertRegex(err, r"candidate .*coli_hip\.dll: exists as a file", detail)
        self.assertEqual(self._errors(err)[0], 127, detail)  # ERROR_PROC_NOT_FOUND
        self.assertIn("a required procedure was not found", err, detail)
        self.assertNotIn("module or one of its dependencies was not found",
                         err, detail)
        self.assertNotIn(_DEP_EXTRA, err, detail)
        self.assertEqual(out.get("preload_0_marker"), str(_RUNTIME_MARKER_A), detail)
        self.assertEqual(out.get("runtime_after_shutdown_count"), "1", detail)
        self._assert_only_preloaded_runtime(out, detail, f.runtime_a)

    def test_the_four_classes_are_distinguishable_from_each_other(self):
        """The whole point: absent, invalid, dependency and procedure differ."""
        f = self.fixture
        seen = {}
        for name, kwargs in (
            ("d_absent", {"with_backend": False}),
            ("d_badpe", {"backend_override": f.bad_pe}),
            ("d_dep", {"preload": str(f.runtime_a),
                       "backend_override": f.backend_dep}),
            ("d_proc", {"preload": str(f.runtime_a),
                        "backend_override": f.backend_dep,
                        "extra_files": (f.dep_lean,)}),
        ):
            kwargs.setdefault("preload", "NONE")
            kwargs.setdefault("env", {_RUNTIME_DIR_VAR: str(f.runtime_a_dir)})
            proc, _ = f.run_harness(Path(self.cases.name) / name, **kwargs)
            err = proc.stderr or ""
            exists = "exists as a file" in err
            seen[name] = (exists, self._errors(err)[0])
        self.assertEqual(seen["d_absent"], (False, 126), seen)
        self.assertEqual(seen["d_badpe"], (True, 193), seen)
        self.assertEqual(seen["d_dep"], (True, 126), seen)
        self.assertEqual(seen["d_proc"], (True, 127), seen)
        # Absent and missing-dependency share a code; only the candidate
        # classification separates them, which is exactly why it is printed.
        self.assertNotEqual(seen["d_absent"], seen["d_dep"])


class LoaderCudaModeIgnoresRuntimeDirTest(unittest.TestCase):
    """A CUDA host must be completely indifferent to COLI_HIP_RUNTIME_DIR.

    The control is the same harness source and the same ``c/backend_loader.c``,
    built without COLI_HIP_DLL — the one macro the Makefile's two mutually
    exclusive builds differ by. Needs no NVIDIA GPU and no CUDA SDK: the fake
    backend is the fixture's, renamed.
    """

    fixture = None

    @classmethod
    def setUpClass(cls):
        reason = _fixture_toolchain_skip()
        if reason:
            raise unittest.SkipTest(reason)
        cls.fixture = _StubFixture()

    @classmethod
    def tearDownClass(cls):
        if cls.fixture is not None:
            cls.fixture.cleanup()
            cls.fixture = None

    def setUp(self):
        self.cases = tempfile.TemporaryDirectory(prefix="coli cuda ignore ")
        self.addCleanup(self.cases.cleanup)

    def _cuda_run(self, name, env):
        f = self.fixture
        return f.run_harness(Path(self.cases.name) / name, str(f.runtime_a),
                             env=env, vendor="cuda")

    def test_invalid_runtime_dir_does_not_change_cuda_behaviour(self):
        """Garbage in COLI_HIP_RUNTIME_DIR vs. no variable at all: identical."""
        f = self.fixture
        baseline_proc, baseline = self._cuda_run("cuda_clean", None)
        poisoned_proc, poisoned = self._cuda_run(
            "cuda_poisoned", {_RUNTIME_DIR_VAR: r"..\definitely not a runtime"})
        detail = ("\nbaseline:\n%s\n%s\npoisoned:\n%s\n%s"
                  % (baseline_proc.stdout, baseline_proc.stderr,
                     poisoned_proc.stdout, poisoned_proc.stderr))

        for proc in (baseline_proc, poisoned_proc):
            self.assertEqual(proc.returncode, 0, detail)
            # No HIP validation ran, so none of its vocabulary appears.
            self.assertNotIn(_RUNTIME_DIR_VAR, proc.stderr or "", detail)
            self.assertNotIn("[HIP]", proc.stderr or "", detail)

        for key in ("loader_init", "backend_loaded", "bound_marker",
                    "preload_0_marker", "runtime_after_backend_count"):
            self.assertEqual(baseline.get(key), poisoned.get(key),
                             "%s diverged" % key + detail)

        # And the outcome itself is the ordinary CUDA one.
        self.assertEqual(poisoned.get("loader_init"), "1", detail)
        self.assertEqual(poisoned.get("bound_marker"), str(_RUNTIME_MARKER_A), detail)
        self.assertTrue(poisoned["backend_path"].lower().endswith("coli_cuda.dll"),
                        detail)

    def test_cuda_host_seeks_the_cuda_backend_only(self):
        """Even with a poisoned variable, the missing-DLL path stays CUDA's."""
        f = self.fixture
        proc, out = f.run_harness(
            Path(self.cases.name) / "cuda_miss", "NONE",
            env={_RUNTIME_DIR_VAR: "C:relative-nonsense"},
            vendor="cuda", with_backend=False)
        err = proc.stderr or ""
        detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, err)
        self.assertEqual(out.get("loader_init"), "0", detail)
        self.assertIn("[CUDA] coli_cuda.dll could not be loaded", err, detail)
        self.assertIn("[CUDA]   fallback by name coli_cuda.dll:", err, detail)
        self.assertNotIn("[HIP]", err, detail)
        self.assertNotIn(_RUNTIME_DIR_VAR, err, detail)


class LoaderProductionStructureTest(unittest.TestCase):
    """Source-order facts that a behavioural test cannot pin down.

    Reads c/backend_loader.c only — no build, no subprocess, and it runs on any
    platform. These guard the ordering guarantees the runtime tests depend on
    but cannot themselves observe from outside the process.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = (HERE / "backend_loader.c").read_text(encoding="utf-8",
                                                        errors="replace")

    def _index(self, needle):
        at = self.src.find(needle)
        self.assertNotEqual(at, -1, "not found in backend_loader.c: %s" % needle)
        return at

    def test_post_backend_verification_precedes_symbol_resolution(self):
        """Verify the runtime again after the backend maps, before GetProcAddress.

        Loading the backend pulls in its dependencies and is the one moment a
        second amdhip64_7.dll could appear. Checking afterwards is what makes
        the binding claim true; checking before symbol resolution is what stops
        an unverifiable backend from being used at all.
        """
        load = self._index("g_cuda.dll = coli_hip_load_backend();")
        verify = self.src.find('coli_hip_verify_bound(&cfg, runtime, "post-backend', load)
        self.assertNotEqual(verify, -1, "no post-backend verification after the load")
        resolve = self.src.find("RESOLVE(init,", verify)
        self.assertNotEqual(resolve, -1, "no symbol resolution after verification")
        self.assertLess(load, verify)
        self.assertLess(verify, resolve)

    def test_runtime_is_verified_before_the_backend_is_loaded(self):
        """The pre-backend check happens inside acquisition, not after it."""
        acquire = self._index("runtime = coli_hip_acquire_runtime(&cfg);")
        load = self._index("g_cuda.dll = coli_hip_load_backend();")
        self.assertLess(acquire, load)
        self.assertIn('coli_hip_verify_bound(cfg, runtime, "runtime verification")',
                      self.src)

    def test_hip_loads_only_one_backend_candidate(self):
        """No bare-name fallback anywhere in the HIP path."""
        hip = self.src[self._index("static HMODULE coli_hip_load_backend(void)"):]
        hip = hip[:hip.index("\n#endif /* COLI_HIP_DLL */")]
        self.assertEqual(hip.count("LoadLibraryExW("), 1, hip)
        self.assertIn("LOAD_WITH_ALTERED_SEARCH_PATH", hip)
        self.assertNotIn("LOAD_LIBRARY_SEARCH_SYSTEM32", hip)
        self.assertNotIn("LOAD_LIBRARY_SEARCH_APPLICATION_DIR", hip)
        self.assertNotIn("fallback", hip)
        # Unicode throughout: no ANSI loader call and no fixed path buffer.
        self.assertNotIn("LoadLibraryExA(", hip)
        self.assertNotIn("MAX_PATH", hip)

    def test_cuda_keeps_exactly_its_two_historical_attempts(self):
        """CUDA is the control and must be untouched by HIP hardening."""
        self.assertEqual(self.src.count("LoadLibraryExA("), 2)
        primary = self._index(
            "g_cuda.dll = LoadLibraryExA(dllpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);")
        fallback = self._index("g_cuda.dll = LoadLibraryExA(COLI_BACKEND_DLL, NULL,")
        self.assertLess(primary, fallback, "the two attempts changed order")
        tail = self.src[fallback:fallback + 200]
        self.assertIn("LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32",
                      tail)
        # Both errors still captured separately, immediately after each call.
        self.assertIn("if(!g_cuda.dll) primary_err = GetLastError();", self.src)
        self.assertIn("if(!g_cuda.dll) fallback_err = GetLastError();", self.src)

    def test_runtime_reference_is_transferred_only_after_everything_passed(self):
        """g_cuda.hip_runtime is assigned once, past the last failure gate."""
        self.assertEqual(self.src.count("g_cuda.hip_runtime = runtime;"), 1)
        transfer = self._index("g_cuda.hip_runtime = runtime;")
        self.assertLess(self._index("RESOLVE(tensor_update, fn_tensor_update)"), transfer)
        self.assertLess(transfer, self._index("g_cuda.available = 1;"))

    def test_shutdown_releases_backend_before_runtime(self):
        """Order matters: the backend imports the runtime, so it goes first."""
        body = self.src[self._index("void coli_cuda_shutdown(void){"):]
        body = body[:body.index("\n}")]
        self.assertLess(body.index("FreeLibrary(g_cuda.dll)"),
                        body.index("FreeLibrary(g_cuda.hip_runtime)"))

    def test_no_helper_is_publicly_exported(self):
        """Internal by construction: static, no header, no dllexport."""
        self.assertNotIn("__declspec(dllexport)", self.src)
        self.assertEqual(len(re.findall(r"^(?:int|void|wchar_t)\s*\*?coli_win32_loader_",
                                        self.src, re.M)), 0)


class LoaderBackendSelectionTest(unittest.TestCase):
    """Which backend DLL does a Windows host look for, and how does it label its
    own loader messages?

    Deliberately a *miss*-path contract, so it needs no GPU, no vendor SDK and
    no backend DLL: each host is copied into an empty temp directory, and
    ``backend_loader.c`` resolves the DLL next to the executable — so the load
    is guaranteed to fail there and print exactly which file it wanted. That
    makes the assertion behavioural (run the artifact, read its stderr) rather
    than a source-text grep, and keeps it green on a machine with no NVIDIA
    driver, no ROCm and no ``nvidia-smi``.

    The two hosts are built under distinct EXE names so the shared
    ``c/colibri.exe`` other test modules probe is never disturbed.
    """

    _hosts = {}

    @classmethod
    def setUpClass(cls):
        if sys.platform != "win32":
            raise unittest.SkipTest("backend_loader.c is compiled only on Windows")
        if shutil.which("make") is None or shutil.which("gcc") is None:
            raise unittest.SkipTest("MinGW make/gcc needed to build the host matrix")
        # HIP_DLL=1 is a host-only build: it links backend_loader.o and no HIP
        # library, so it must succeed with no HIP SDK, HIP_PATH or HIP_ARCH.
        for key, exe_suffix, flag in (("cuda", "_cudadll.exe", "CUDA_DLL=1"),
                                      ("hip", "_hipdll.exe", "HIP_DLL=1")):
            target = "colibri" + exe_suffix
            # errors="replace": the toolchain speaks the console codepage, which
            # is not always decodable as the Python default on localized hosts.
            proc = subprocess.run(
                ["make", "-C", str(HERE), target, "EXE=" + exe_suffix, flag],
                cwd=str(HERE.parent), text=True, errors="replace",
                capture_output=True, timeout=600,
            )
            built = HERE / target
            if proc.returncode != 0 or not built.exists():
                raise unittest.SkipTest(
                    "could not build the %s host matrix: %s" % (key, proc.stderr[-400:])
                )
            cls._hosts[key] = built

        # Since enforcement landed, a HIP host loads the configured runtime
        # before it ever looks for the backend, so the placeholder file this
        # class used to drop in the sandbox is no longer enough — it has to be
        # a real, loadable DLL. It is a six-line stub with no GPU content, no
        # DllMain and no vendor header, built with the gcc this class already
        # requires, and it is never loaded by Python.
        cls._runtime_tmp = tempfile.TemporaryDirectory(prefix="coli host runtime ")
        src = Path(cls._runtime_tmp.name) / "stub.c"
        src.write_text(
            "/* generated stub runtime for the host matrix: no GPU work. */\n"
            "__declspec(dllexport) int coli_test_runtime_marker(void)\n"
            "{ return 0; }\n", encoding="ascii")
        cls._stub_runtime = Path(cls._runtime_tmp.name) / _RUNTIME_BASENAME
        proc = subprocess.run(
            ["gcc", "-O0", "-shared", str(src), "-o", str(cls._stub_runtime)],
            text=True, errors="replace", capture_output=True, timeout=300)
        if proc.returncode != 0 or not cls._stub_runtime.is_file():
            raise unittest.SkipTest(
                "could not build the host-matrix stub runtime: %s" % proc.stderr[-400:])

    @classmethod
    def tearDownClass(cls):
        for path in cls._hosts.values():
            try:
                path.unlink()
            except OSError:
                pass
        cls._hosts.clear()
        if getattr(cls, "_runtime_tmp", None) is not None:
            cls._runtime_tmp.cleanup()
            cls._runtime_tmp = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run_isolated(self, key, timeout=60, unset=()):
        """Copy one host into an empty dir and run it with the GPU requested.

        The directory holds the executable and the throwaway model only — no
        backend DLL — so the loader's not-found path is the one exercised.

        ``unset`` names environment variables to drop from the child *after*
        the defaults are applied, which is how a caller opts out of the
        ``COLI_CUDA=1`` request set below and tests the not-requested contract
        instead. The default removes nothing, so existing callers are unchanged.

        A HIP host refuses to reach the backend without a valid
        COLI_HIP_RUNTIME_DIR, and since W1-B2d2 it also loads and verifies what
        that directory names — so the sandbox gets a throwaway directory
        holding the generated stub runtime built in setUpClass. No HIP SDK, no
        ROCm and no GPU is involved; the ambient variable is dropped either
        way, so a developer's real install cannot influence the result.
        """
        # One sandbox per host, so a test may exercise both in a single method.
        sandbox = Path(self.tmp.name) / ("bin_" + key)
        sandbox.mkdir(exist_ok=True)
        exe = sandbox / self._hosts[key].name
        shutil.copy2(self._hosts[key], exe)
        model = _minimal_model(str(sandbox))
        merged = {**os.environ, "SNAP": str(model), "COLI_CUDA": "1"}
        merged.pop(_RUNTIME_DIR_VAR, None)
        if key == "hip":
            runtime_dir = sandbox / "runtime"
            runtime_dir.mkdir(exist_ok=True)
            shutil.copy2(self._stub_runtime, runtime_dir / _RUNTIME_BASENAME)
            merged[_RUNTIME_DIR_VAR] = str(runtime_dir)
        for name in unset:
            merged.pop(name, None)
        return subprocess.run(
            [str(exe), "1"],
            env=merged, cwd=str(sandbox), text=True, errors="replace",
            capture_output=True, timeout=timeout,
        )

    def test_cuda_dll_host_reports_cuda_backend(self):
        """CUDA_DLL host: names coli_cuda.dll everywhere, never the HIP name."""
        err = self._run_isolated("cuda").stderr or ""
        self.assertIn("[CUDA] coli_cuda.dll could not be loaded", err)
        self.assertIn("[CUDA]   fallback by name coli_cuda.dll:", err)
        # The sandbox genuinely has no backend, so that is what it must say.
        self.assertIn("does not exist", err)
        self.assertNotIn("coli_hip.dll", err)
        self.assertNotIn("[HIP]", err)

    def test_hip_dll_host_reports_hip_backend(self):
        """HIP_DLL host: names coli_hip.dll everywhere, never the CUDA name.

        Runs with no AMD GPU, no HIP SDK and no nvidia-smi — the loader has not
        reached any vendor runtime at the point this message is printed.
        """
        err = self._run_isolated("hip").stderr or ""
        self.assertIn("[HIP] coli_hip.dll could not be loaded", err)
        self.assertIn("does not exist", err)
        self.assertNotIn("coli_cuda.dll", err)
        # HIP loads only the absolute executable-directory candidate, so there
        # is no by-name fallback line to print and no System32 involvement.
        self.assertNotIn("fallback by name", err)
        self.assertNotIn("System32", err)

    def test_cuda_dll_host_with_coli_cuda_unset_stays_on_cpu(self):
        """CUDA_DLL host, COLI_CUDA absent: no request, no loader, no message.

        This is the opt-in contract that ``test_cuda_env`` can only assert when
        a CUDA host happens to be the resident binary — here it always runs,
        because the host is built by this class and the sandbox guarantees no
        backend DLL. Nothing consults nvidia-smi, a GPU, a CUDA SDK or
        coli_cuda.dll: with COLI_CUDA removed the engine never reaches the
        loader at all, and continues to model validation on the CPU path.
        """
        result = self._run_isolated(
            "cuda", unset=("COLI_CUDA", "COLI_GPU", "COLI_GPUS"))
        err = result.stderr or ""
        self.assertNotEqual(result.returncode, 2)
        self.assertNotIn("[CUDA]", err)
        self.assertNotIn("coli_cuda.dll not found", err)
        self.assertNotIn("requested backend is unavailable", err)
        # Reachability oracle: the throwaway fixture is rejected by model
        # validation, which proves execution got past GPU setup rather than
        # exiting inside it.
        self.assertIn("this engine requires n_group=1", err)

    def test_backend_miss_is_not_silent_and_does_no_gpu_work(self):
        """Both hosts fail closed the same way: exit 2, no GPU touched."""
        for key in ("cuda", "hip"):
            with self.subTest(host=key):
                result = self._run_isolated(key)
                err = result.stderr or ""
                self.assertIn("could not be loaded; GPU tier disabled", err)
                # colibri.c owns this second line and stays vendor-neutral in
                # W1-B1: only the loader's own prefix is discriminated here.
                self.assertIn("[CUDA] requested backend is unavailable", err)
                self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
