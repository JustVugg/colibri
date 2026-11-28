"""Verify that coli_launch-launched CUDA chain kernels open with COLI_PDL_WAIT().

On sm_90+ GPUs, coli_launch starts kernels with Programmatic Dependent Launch
(PDL) enabled by default (COLI_CUDA_PDL != 0). Under PDL a dependent kernel may
begin executing while the preceding kernel on the same stream is still finishing,
so every coli_launch target must open with COLI_PDL_WAIT() as its first statement
before reading inputs produced by the previous kernel.
"""
import re
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent

# CUDA chain sources covered by the PDL wait invariant.
COVERED_FILES = (
    "cuda_chain.cu",
    "cuda_bd.cuh",
    "cuda_dflash.cuh",
    "cuda_dsv4c.cuh",
    "cuda_glm53c.cuh",
)

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_LAUNCH_RE = re.compile(r"\bcoli_launch\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)")
_GLOBAL_RE = re.compile(r"\b__global__\b")
_LAUNCH_BOUNDS_RE = re.compile(r"__launch_bounds__\s*\([^)]*\)")
_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def _parse_kernels(text: str) -> dict[str, str]:
    """Return {kernel_name: stripped_body_prefix} for every __global__ definition."""
    clean = _strip_comments(text)
    kernels: dict[str, str] = {}
    for m in _GLOBAL_RE.finditer(clean):
        idx = m.start()
        brace = clean.find("{", idx)
        semi = clean.find(";", idx)
        if brace == -1 or (semi != -1 and semi < brace):
            continue
        sig = _LAUNCH_BOUNDS_RE.sub("", clean[idx:brace])
        nm = _NAME_RE.search(sig)
        if not nm:
            continue
        name = nm.group(1)
        body_prefix = clean[brace + 1 : brace + 256].lstrip()
        kernels[name] = body_prefix
    return kernels


def _parse_coli_launches(text: str) -> set[str]:
    """Return kernel names launched via coli_launch(...) in text."""
    clean = _strip_comments(text)
    return set(_LAUNCH_RE.findall(clean))


class CudaPdlWaitTest(unittest.TestCase):
    def test_coli_launched_kernels_open_with_pdl_wait(self):
        all_kernels: dict[str, tuple[str, str]] = {}
        launches_by_file: dict[str, set[str]] = {}

        for fname in COVERED_FILES:
            path = C_DIR / fname
            self.assertTrue(path.is_file(), f"missing {fname}")
            text = path.read_text(encoding="utf-8")
            kernels = _parse_kernels(text)
            launches = _parse_coli_launches(text)
            self.assertTrue(kernels, f"no __global__ kernels parsed in {fname}")
            self.assertTrue(launches, f"no coli_launch sites parsed in {fname}")
            for name, body in kernels.items():
                all_kernels[name] = (fname, body)
            launches_by_file[fname] = launches

        missing = []
        unresolved = []
        for fname, launches in launches_by_file.items():
            for name in sorted(launches):
                if name not in all_kernels:
                    unresolved.append(f"{fname}: {name}")
                    continue
                def_file, body = all_kernels[name]
                if not re.match(r"^COLI_PDL_WAIT\s*\(\s*\)\s*;", body):
                    missing.append(f"{def_file}: {name}")

        self.assertEqual(unresolved, [], f"coli_launch target not found: {unresolved}")
        self.assertEqual(
            missing,
            [],
            f"coli_launch kernels missing leading COLI_PDL_WAIT(): {missing}",
        )


if __name__ == "__main__":
    unittest.main()
