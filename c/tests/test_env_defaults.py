"""env_for: default Windows misurati (DIRECT/PIPE/PILOT_REAL + blocco OMP).

Carica `coli` come modulo (ha la guardia __main__) e verifica il contratto:
- win32: i tre default I/O e il blocco OMP sono setdefault
- un override esplicito dell'utente vince sempre
- COLI_NO_OMP_TUNE spegne SOLO il blocco OMP, non i default I/O
- non-win32: env_for non tocca nulla di tutto questo
"""
import importlib.machinery
import importlib.util
import os
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

_loader = importlib.machinery.SourceFileLoader("coli_cli", str(HERE / "coli"))
_spec = importlib.util.spec_from_loader("coli_cli", _loader)
coli = importlib.util.module_from_spec(_spec)
_loader.exec_module(coli)

_MODEL_DIR = tempfile.TemporaryDirectory()
_MODEL = Path(_MODEL_DIR.name)
(_MODEL / "config.json").write_text(
    json.dumps({"model_type": "glm_moe_dsa"}), encoding="utf-8")


def args(**over):
    base = dict(model=str(_MODEL), policy="quality", ram=0, ngen=0, topp=0, topk=0,
                temp=None, repin=0, ctx=0, auto_tier=False, gpu=None, vram=0)
    base.update(over)
    return types.SimpleNamespace(**base)


class EnvDefaultsTest(unittest.TestCase):
    def env_for_with(self, environ, platform, cuda=False):
        """Run env_for on a bare-chat args() under a faked env + platform.

        cuda=False by default so the existing default-I/O tests stay
        deterministic: the Windows auto-enable branch calls cuda_binary() and
        (if True) discover_gpus(), both of which reach the real machine — faking
        False keeps these tests independent of the host's GPU."""
        with mock.patch.dict(os.environ, environ, clear=True), \
             mock.patch.object(sys, "platform", platform), \
             mock.patch.object(coli, "cuda_binary", return_value=cuda), \
             mock.patch("resource_plan.physical_cpu_count", return_value=8):
            return coli.env_for(args())

    def test_win32_sets_measured_defaults(self):
        e = self.env_for_with({}, "win32")
        self.assertEqual(e["DIRECT"], "1")
        self.assertEqual(e["PIPE"], "1")
        self.assertEqual(e["PILOT_REAL"], "1")
        self.assertEqual(e["OMP_WAIT_POLICY"], "active")
        self.assertNotIn("OMP_PROC_BIND", e)  # MinGW libgomp: niente affinity

    def test_explicit_override_wins(self):
        e = self.env_for_with({"DIRECT": "0", "PIPE": "0"}, "win32")
        self.assertEqual(e["DIRECT"], "0")
        self.assertEqual(e["PIPE"], "0")
        self.assertEqual(e["PILOT_REAL"], "1")  # non overridden -> default

    def test_kill_switch_scope_is_omp_only(self):
        e = self.env_for_with({"COLI_NO_OMP_TUNE": "1"}, "win32")
        self.assertNotIn("OMP_WAIT_POLICY", e)
        self.assertNotIn("OMP_NUM_THREADS", e)
        self.assertEqual(e["DIRECT"], "1")   # i default I/O restano attivi
        self.assertEqual(e["PIPE"], "1")

    def test_non_windows_untouched(self):
        e = self.env_for_with({}, "linux")
        for k in ("DIRECT", "PIPE", "PILOT_REAL", "OMP_WAIT_POLICY"):
            self.assertNotIn(k, e)


class CudaAutoEnableTest(unittest.TestCase):
    """Windows bare `coli chat` (no --gpu/--vram/--auto-tier) used to ALWAYS run
    CPU-only even on a CUDA build with a GPU present. env_for now auto-enables
    CUDA on win32 when cuda_binary() is True and a GPU is discoverable; falls
    back to CPU with a warning if nvidia-smi (discover_gpus) is missing; stays
    silent on a CPU build; and never touches the Linux path."""

    def _env_for(self, platform, cuda, gpus, plan=None):
        # Patch discover_gpus / build_plan / environment_for_plan at the
        # resource_plan module (env_for imports them lazily on each call, so the
        # patches are live when those imports run). Stubbing the planner keeps
        # the test independent of a real model dir (args().model == "X").
        import resource_plan
        a = args()
        GPB = 1024 ** 3
        if plan is None:
            plan = {"tiers": {"ram": {"budget_bytes": 16 * GPB, "cache_slots_per_layer": 4},
                              "vram": {"budget_bytes": int(8.0 * GPB), "devices": gpus}}}

        def fake_environment_for_plan(p, env, cuda_enabled=True):
            # Mirror the real contract: size CUDA_EXPERT_GB from the plan's VRAM
            # budget (this is the value env_for propagates into the engine env).
            r = dict(env)
            if cuda_enabled and p["tiers"]["vram"]["devices"] and p["tiers"]["vram"]["budget_bytes"] > 0:
                r["CUDA_EXPERT_GB"] = f"{p['tiers']['vram']['budget_bytes'] / GPB:.3f}"
            return r

        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(sys, "platform", platform), \
             mock.patch.object(coli, "cuda_binary", return_value=cuda), \
             mock.patch.object(resource_plan, "discover_gpus", return_value=gpus), \
             mock.patch.object(resource_plan, "physical_cpu_count", return_value=8), \
             mock.patch.object(resource_plan, "build_plan", return_value=plan), \
             mock.patch.object(resource_plan, "environment_for_plan",
                               side_effect=fake_environment_for_plan):
            return coli.env_for(a)

    def _fake_gpu(self, index=0, name="NVIDIA GeForce RTX 5070 Ti",
                  total_mib=16384, free_mib=15000):
        return {"index": index, "name": name,
                "total_bytes": total_mib * 1024 * 1024,
                "free_bytes": free_mib * 1024 * 1024}

    def test_win32_auto_enables_cuda_when_gpu_present(self):
        e = self._env_for("win32", cuda=True, gpus=[self._fake_gpu()])
        self.assertEqual(e["COLI_CUDA"], "1")
        self.assertEqual(e["COLI_GPUS"], "0")
        # VRAM budget is sized from free VRAM by build_plan (real minus reserve),
        # so it must be present and positive — never a guess or zero.
        self.assertIn("CUDA_EXPERT_GB", e)
        self.assertGreater(float(e["CUDA_EXPERT_GB"]), 0.0)
        # Dense offload is an explicit opt-in (matches --auto-tier): not set here.
        self.assertNotIn("CUDA_DENSE", e)

    def test_win32_falls_back_to_cpu_when_nvidia_smi_missing(self):
        # coli_cuda.dll present (cuda=True) but nvidia-smi absent (no GPUs found)
        # -> warn + CPU-only, never crash, never set COLI_CUDA.
        e = self._env_for("win32", cuda=True, gpus=[])
        self.assertNotIn("COLI_CUDA", e)
        self.assertNotIn("COLI_GPUS", e)
        self.assertNotIn("CUDA_EXPERT_GB", e)

    def test_win32_does_not_auto_enable_an_unqualified_gpu(self):
        # A device whose free memory is not qualified as a placement budget
        # (free_bytes is None -- a Windows AMD part found through hipInfo) is
        # discovered and worth reporting, but auto-enable is an automatic
        # placement decision and must not be made from it. Bare `coli chat`
        # stays on the CPU path, exactly as if nothing had been found.
        identity_only = {"index": 0, "name": "AMD Radeon(TM) 8060S Graphics",
                         "arch": "gfx1151", "total_bytes": 78 * 1024 ** 3,
                         "free_bytes": None, "unified_memory": True}
        e = self._env_for("win32", cuda=True, gpus=[identity_only])
        self.assertNotIn("COLI_CUDA", e)
        self.assertNotIn("COLI_GPUS", e)
        self.assertNotIn("CUDA_EXPERT_GB", e)

    def test_win32_cpu_build_stays_silent(self):
        # No coli_cuda.dll (cuda=False) -> CPU build, nothing GPU-related emitted.
        e = self._env_for("win32", cuda=False, gpus=[self._fake_gpu()])
        self.assertNotIn("COLI_CUDA", e)
        self.assertNotIn("COLI_GPUS", e)

    def test_linux_bare_chat_not_auto_enabled(self):
        # The auto-enable is scoped to win32: a Linux bare chat with a GPU
        # present must NOT turn CUDA on (Linux keeps the explicit-flag UX).
        e = self._env_for("linux", cuda=True, gpus=[self._fake_gpu()])
        self.assertNotIn("COLI_CUDA", e)
        self.assertNotIn("CUDA_EXPERT_GB", e)


class Dsv4CudaDetectTest(unittest.TestCase):
    """dsv4_cuda_available: the Windows check is a DLL next to the engine, but
    Linux links the tier straight into the binary (nvcc, -lcudart), so the DLL
    probe there rejected every valid `make deepseek-v4 CUDA=1` build (#1219)."""

    def _detect(self, platform, ldd_stdout=None, dll=False, engine_exists=True,
                ldd_error=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        eng = Path(tmp.name) / "deepseek"
        if engine_exists:
            eng.write_bytes(b"")
        if dll:
            (Path(tmp.name) / "coli_cuda_dsv4.dll").write_bytes(b"")
        fake_run = mock.Mock(return_value=types.SimpleNamespace(stdout=ldd_stdout or ""),
                             side_effect=ldd_error)
        with mock.patch.object(sys, "platform", platform), \
             mock.patch.object(coli, "engine_for", return_value=str(eng)), \
             mock.patch.object(coli.subprocess, "run", fake_run):
            return coli.dsv4_cuda_available(str(_MODEL))

    def test_linux_cuda_link_detected(self):
        out = "\tlibcudart.so.12 => /opt/cuda/lib64/libcudart.so.12 (0x00007f)\n"
        self.assertTrue(self._detect("linux", ldd_stdout=out))

    def test_linux_hip_link_is_not_dsv4_cuda(self):
        out = "\tlibamdhip64.so.6 => /opt/rocm/lib/libamdhip64.so.6 (0x00007f)\n"
        self.assertFalse(self._detect("linux", ldd_stdout=out))

    def test_linux_cpu_build_rejected(self):
        out = "\tlibc.so.6 => /usr/lib/libc.so.6 (0x00007f)\n"
        self.assertFalse(self._detect("linux", ldd_stdout=out))

    def test_linux_broken_link_rejected(self):
        out = "\tlibcudart.so.12 => not found\n"
        self.assertFalse(self._detect("linux", ldd_stdout=out))

    def test_linux_missing_engine_rejected(self):
        self.assertFalse(self._detect("linux", engine_exists=False))

    def test_linux_ldd_failures_are_rejected(self):
        for error in (FileNotFoundError("ldd"),
                      subprocess.TimeoutExpired(["ldd", "deepseek"], 3)):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(self._detect("linux", ldd_error=error))

    def test_unexpected_ldd_errors_are_not_hidden(self):
        with self.assertRaises(RuntimeError):
            self._detect("linux", ldd_error=RuntimeError("test bug"))

    def test_win32_dll_detected(self):
        self.assertTrue(self._detect("win32", dll=True))

    def test_win32_missing_dll_rejected(self):
        self.assertFalse(self._detect("win32", dll=False))

    def test_macos_has_no_dsv4_cuda_tier(self):
        self.assertFalse(self._detect("darwin", ldd_stdout="libcudart.so"))

    def test_build_hint_names_the_right_build(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertIn("make deepseek-v4 CUDA=1", coli.dsv4_cuda_build_hint())
        with mock.patch.object(sys, "platform", "win32"):
            self.assertIn("coli_cuda_dsv4.dll", coli.dsv4_cuda_build_hint())
        with mock.patch.object(sys, "platform", "darwin"):
            hint = coli.dsv4_cuda_build_hint()
            self.assertIn("only on Linux and Windows", hint)
            self.assertNotIn("CUDA=1", hint)


if __name__ == "__main__":
    unittest.main()
