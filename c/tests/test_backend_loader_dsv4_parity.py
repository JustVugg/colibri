"""Every dsv4_cuda_* in backend_cuda_dsv4.h must be exported and forwarded on Windows.

On Linux deepseek-v4 links backend_cuda_dsv4.o directly, so a new function in
backend_cuda_dsv4.h is callable the moment backend_cuda_dsv4.cu defines it.
On Windows the kernels live in coli_cuda_dsv4*.dll, built by MSVC with
-Xlinker /DEF:dsv4.def and no __declspec(dllexport): the DLL exports exactly
the names dsv4.def lists, and the MinGW host only sees what
backend_loader_dsv4.c resolves and forwards. A name missing from dsv4.def is
therefore absent from every freshly built DLL. For a RESOLVE'd symbol the
loader rejects the DLL; for an optional one it silently takes the "older DLL"
fallback -- which is how dsv4_cuda_device_unified (unified-memory devices,
where the VRAM reserve guard must stay off) never reached a Windows build,
and how dsv4_cuda_expert_bank_slots (bank sized to the expert count) would
silently keep the 256-slot bank.

test_backend_loader_header_parity.py covers only the GLM loader (coli_cuda_*).
This is its twin for the DeepSeek V4 loader. Pure text, no compiler: it runs
on every CI host, not only on Windows.
"""
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def _strip_comments(src):
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"//[^\n]*", " ", src)


def header_functions():
    src = _strip_comments((HERE / "backend_cuda_dsv4.h").read_text(encoding="utf-8"))
    return set(re.findall(r"\b(dsv4_cuda_\w+)\s*\(", src))


def def_exports():
    src = (HERE / "dsv4.def").read_text(encoding="utf-8")
    body = src.split("EXPORTS", 1)[1]
    names = set()
    for line in body.splitlines():
        line = line.split(";", 1)[0].strip()
        if line:
            # "name", "name @ordinal" or "name=internal": the exported name is first.
            names.add(re.split(r"[\s=@]", line, 1)[0])
    return names


def _loader_src():
    return _strip_comments((HERE / "backend_loader_dsv4.c").read_text(encoding="utf-8"))


def loader_mandatory():
    names = re.findall(r"^\s+RESOLVE\((\w+),", _loader_src(), re.M)
    return {"dsv4_cuda_" + n for n in names}


def loader_optional():
    """(field, symbol) for every GetProcAddress outside the RESOLVE macro."""
    return re.findall(
        r"g_dsv4\.(\w+)\s*=\s*\(\w+\)\s*GetProcAddress\(\s*g_dsv4\.dll\s*,\s*\"(dsv4_cuda_\w+)\"\s*\)",
        _loader_src())


def loader_resolved():
    return loader_mandatory() | {sym for _, sym in loader_optional()}


def loader_defined():
    # Public (non-static) wrapper definitions; the loader's own static helpers
    # (dsv4_cuda_load, dsv4_cuda_resolve) are not part of the ABI.
    return set(re.findall(
        r"^(?!static\b)[A-Za-z_][\w \t\*]*?\b(dsv4_cuda_\w+)\s*\([^;]*?\)\s*\{",
        _loader_src(), re.M))


class Dsv4LoaderExportParityTest(unittest.TestCase):
    def test_parsers_found_the_abi(self):
        # Guard against a regex that silently matches nothing.
        for name, found in (("header", header_functions()), ("dsv4.def", def_exports()),
                            ("loader RESOLVE", loader_mandatory()),
                            ("loader wrappers", loader_defined())):
            self.assertGreater(len(found), 60, name)
            self.assertIn("dsv4_cuda_init", found, name)
        self.assertGreater(len(loader_optional()), 3)

    def test_every_header_function_is_exported_by_the_def(self):
        missing = sorted(header_functions() - def_exports())
        self.assertEqual(missing, [],
                         "declared in backend_cuda_dsv4.h but not listed in dsv4.def; "
                         "the MSVC-built coli_cuda_dsv4*.dll does not export them, so "
                         "the loader rejects the DLL (RESOLVE) or silently takes the "
                         "older-DLL fallback (optional): %s" % missing)

    def test_every_def_export_is_declared_in_the_header(self):
        stale = sorted(def_exports() - header_functions())
        self.assertEqual(stale, [],
                         "dsv4.def exports names backend_cuda_dsv4.h does not declare; "
                         "the DLL link fails on the unresolved export: %s" % stale)

    def test_every_header_function_is_resolved_by_the_loader(self):
        missing = sorted(header_functions() - loader_resolved())
        self.assertEqual(missing, [],
                         "declared in backend_cuda_dsv4.h but never RESOLVE'd or "
                         "GetProcAddress'd in backend_loader_dsv4.c: %s" % missing)

    def test_every_resolved_symbol_is_exported_by_the_def(self):
        missing = sorted(loader_resolved() - def_exports())
        self.assertEqual(missing, [],
                         "backend_loader_dsv4.c resolves names dsv4.def does not export; "
                         "a freshly built DLL never has them: %s" % missing)

    def test_every_header_function_has_a_forwarder(self):
        undefined = sorted(header_functions() - loader_defined())
        self.assertEqual(undefined, [],
                         "declared in backend_cuda_dsv4.h but no public wrapper is "
                         "defined in backend_loader_dsv4.c; a Windows deepseek-v4.exe "
                         "calling them fails to link: %s" % undefined)

    def test_optional_lookups_fill_their_own_slot(self):
        crossed = sorted((field, sym) for field, sym in loader_optional()
                         if "dsv4_cuda_" + field != sym)
        self.assertEqual(crossed, [],
                         "optional GetProcAddress stores a symbol in another "
                         "function's slot: %s" % crossed)


if __name__ == "__main__":
    unittest.main()
