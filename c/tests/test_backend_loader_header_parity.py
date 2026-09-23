"""Every coli_cuda_* the header exports must have a Windows loader forwarder.

On Linux a host links backend_cuda.o directly, so a new COLI_CUDA_DLLEXPORT
prototype in backend_cuda.h is callable the moment backend_cuda.cu defines it.
On Windows the host only sees what backend_loader.c resolves and forwards.
A symbol added to the header but not to the loader therefore builds and runs
everywhere except a CUDA_DLL=1 host, where it surfaces as an undefined
reference -- which is how coli_cuda_available_device_count broke qwen36.exe.

test_backend_loader.py derives its ABI FROM the loader, so it cannot see this
gap. This test compares the loader against the header instead. Pure text, no
compiler: it runs on every CI host, not only on Windows.
"""
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# Header exports deliberately absent from the Windows loader, with the reason.
# Adding a name here should be as conscious as adding a RESOLVE.
NOT_FORWARDED = {
    # COLI_ANS (DietGPU) is wired only for the Linux CUDA=1 path.
    "coli_cuda_tensor_upload_compressed",
}


def header_exports():
    src = (HERE / "backend_cuda.h").read_text(encoding="utf-8")
    return set(re.findall(
        r"COLI_CUDA_DLLEXPORT[^;(]*?\b(coli_cuda_\w+)\s*\(", src))


def loader_resolved():
    src = (HERE / "backend_loader.c").read_text(encoding="utf-8")
    names = re.findall(r"^\s+RESOLVE(?:_OPT)?\((\w+),", src, re.M)
    return {"coli_cuda_" + n for n in names}


def loader_defined():
    src = (HERE / "backend_loader.c").read_text(encoding="utf-8")
    return set(re.findall(
        r"^[A-Za-z_][\w \t\*]*?\b(coli_cuda_\w+)\s*\([^;]*?\)\s*\{", src, re.M))


class LoaderHeaderParityTest(unittest.TestCase):
    def test_parsers_found_the_abi(self):
        # Guard against a regex that silently matches nothing.
        self.assertGreater(len(header_exports()), 40)
        self.assertGreater(len(loader_resolved()), 40)
        self.assertIn("coli_cuda_init", header_exports())

    def test_every_header_export_is_resolved_by_the_loader(self):
        missing = sorted(header_exports() - loader_resolved() - NOT_FORWARDED)
        self.assertEqual(missing, [],
                         "declared COLI_CUDA_DLLEXPORT in backend_cuda.h but never "
                         "RESOLVE/RESOLVE_OPT'd in backend_loader.c; a CUDA_DLL=1 "
                         "host calling them fails to link: %s" % missing)

    def test_every_resolved_symbol_has_a_forwarder(self):
        undefined = sorted(loader_resolved() - loader_defined())
        self.assertEqual(undefined, [],
                         "resolved from the DLL but no public wrapper is defined "
                         "in backend_loader.c: %s" % undefined)

    def test_exemptions_are_still_real(self):
        stale = sorted(NOT_FORWARDED - header_exports())
        self.assertEqual(stale, [], "NOT_FORWARDED names no longer in header: %s"
                         % stale)


if __name__ == "__main__":
    unittest.main()
