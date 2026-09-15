"""Exercise discovery before initialization with a native stub DLL, without CUDA."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent.parent


@unittest.skipUnless(os.name == "nt" and shutil.which("gcc"), "requires Windows and MinGW gcc")
class CudaLoaderDiscoveryTest(unittest.TestCase):
    def run_probe(self, discovery=True, backend=True):
        with tempfile.TemporaryDirectory(prefix="coli discovery ") as tmp:
            root = Path(tmp)
            host = root / "probe.c"
            host.write_text('''#include <stdio.h>
#include "backend_cuda.h"
int main(void) {
    int device = 0;
    printf("visible=%d\\n", coli_cuda_available_device_count());
    printf("before=%d\\n", coli_cuda_device_count());
    printf("init=%d\\n", coli_cuda_init(&device, 1));
    printf("after=%d\\n", coli_cuda_device_count());
    coli_cuda_shutdown();
    return 0;
}
''', encoding="ascii")
            def build(*args):
                proc = subprocess.run(["gcc", *map(str, args)], capture_output=True,
                                      text=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            build("-DCOLI_CUDA", "-I", HERE, host, HERE / "backend_loader.c",
                  "-o", root / "probe.exe")
            if backend:
                source = (HERE / "backend_loader.c").read_text(encoding="utf-8")
                names = re.findall(r"^\s+RESOLVE(?:_OPT)?\((\w+),", source, re.M)
                bodies = {
                    "init": "int coli_cuda_init(const int *d, int n) { (void)d; count=n; return 1; }",
                    "shutdown": "void coli_cuda_shutdown(void) { count=0; }",
                    "device_count": "int coli_cuda_device_count(void) { return count; }",
                    "available_device_count": "int coli_cuda_available_device_count(void) { return 2; }",
                }
                lines = ["static int count;"]
                for name in names:
                    if name == "available_device_count" and not discovery:
                        continue
                    body = bodies.get(name, "int coli_cuda_%s(void) { return 0; }" % name)
                    lines.append("__declspec(dllexport) " + body)
                dll = root / "stub.c"
                dll.write_text("\n".join(lines), encoding="ascii")
                build("-shared", dll, "-o", root / "coli_cuda.dll")
            proc = subprocess.run([str(root / "probe.exe")], capture_output=True,
                                  text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return proc

    def test_discovery_precedes_init(self):
        proc = self.run_probe()
        self.assertEqual(proc.stdout.splitlines(), ["visible=2", "before=0", "init=1", "after=1"])

    def test_old_dll_explains_missing_discovery_but_allows_explicit_init(self):
        proc = self.run_probe(discovery=False)
        self.assertEqual(proc.stdout.splitlines(), ["visible=0", "before=0", "init=1", "after=1"])
        self.assertIn("missing symbol coli_cuda_available_device_count", proc.stderr)
        self.assertIn("COLI_GPUS", proc.stderr)

    def test_missing_dll_returns_zero(self):
        proc = self.run_probe(backend=False)
        self.assertEqual(proc.stdout.splitlines(), ["visible=0", "before=0", "init=0", "after=0"])
        self.assertIn("could not be loaded", proc.stderr)


if __name__ == "__main__":
    unittest.main()
