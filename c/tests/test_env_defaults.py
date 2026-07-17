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
import sys
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


def args(**over):
    base = dict(model="X", policy="quality", ram=0, ngen=0, topp=0, topk=0,
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
             mock.patch.object(coli, "cuda_binary", return_value=cuda):
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


if __name__ == "__main__":
    unittest.main()
