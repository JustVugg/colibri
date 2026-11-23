"""A container's CPU limits bound the launcher's OpenMP team (#2914).

``coli`` sets OMP_NUM_THREADS from ``team_cpu_count()`` for the engines, and an
engine given OMP_NUM_THREADS sizes nothing itself, so the launcher is where a
CPU quota (cgroup ``cpu.max`` / ``cpu.cfs_quota_us``) and a cpuset (the affinity
mask) must be seen: lscpu counts the host's cores whatever the container may
use. Every kernel input is a file under a temporary directory, as in
test_cgroup_memory.py; the expectations come from the fixture numbers.
"""

import contextlib
import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import types
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import resource_plan  # noqa: E402
from resource_plan import allowed_cpu_count, bound_team, cgroup_cpu_quota  # noqa: E402

_LINUX_ONLY = unittest.skipUnless(
    sys.platform.startswith("linux"),
    "cgroup fixtures model Linux procfs/cgroupfs paths; the probe is Linux-only",
)


@_LINUX_ONLY
class CgroupCpuQuotaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "cgroup"
        self.root.mkdir()
        self.proc_cgroup = self.base / "proc-self-cgroup"
        self.proc_mountinfo = self.base / "proc-self-mountinfo"

    def _membership(self, *lines):
        self.proc_cgroup.write_text("\n".join(lines) + "\n", encoding="ascii")

    def _mount_line(self, *, mount_id=30, root="/", mount_point=None, fs_type="cgroup2",
                    super_options="rw"):
        mount_point = self.root if mount_point is None else Path(mount_point)
        return (f"{mount_id} 1 0:{mount_id} {root} {mount_point} rw,nosuid,nodev - "
                f"{fs_type} {fs_type} {super_options}")

    def _mounts(self, *lines):
        self.proc_mountinfo.write_text("\n".join(lines) + "\n", encoding="ascii")

    @staticmethod
    def _write(directory, name, text):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text, encoding="ascii")

    def _quota(self, mountinfo=True):
        return cgroup_cpu_quota(
            cgroup_root=self.root,
            proc_cgroup_path=self.proc_cgroup,
            proc_mountinfo_path=self.proc_mountinfo if mountinfo else self.base / "absent",
        )

    def test_v2_quota_in_cpus(self):
        # the issue's pod: cpu.max 765000 100000 is 7.65 CPUs
        self._membership("0::/pod/job")
        self._mounts(self._mount_line())
        self._write(self.root / "pod" / "job", "cpu.max", "765000 100000\n")
        self.assertAlmostEqual(self._quota(), 7.65)

    def test_v2_max_is_no_quota(self):
        self._membership("0::/pod/job")
        self._mounts(self._mount_line())
        self._write(self.root / "pod" / "job", "cpu.max", "max 100000\n")
        self.assertIsNone(self._quota())

    def test_v2_tightest_ancestor_wins(self):
        self._membership("0::/pod/job")
        self._mounts(self._mount_line())
        self._write(self.root / "pod" / "job", "cpu.max", "800000 100000\n")
        self._write(self.root / "pod", "cpu.max", "400000 100000\n")
        self.assertAlmostEqual(self._quota(), 4.0)

    def test_v2_namespace_root(self):
        # a private cgroup namespace: the process is at "/", the container's cgroup is the mount
        self._membership("0::/")
        self._mounts(self._mount_line())
        self._write(self.root, "cpu.max", "250000 100000\n")
        self.assertAlmostEqual(self._quota(), 2.5)

    def test_v2_own_cgroup_mounted_at_the_root(self):
        # no cgroup namespace: procfs names the host's path, the mount shows that cgroup
        self._membership("0::/system.slice/docker-abc.scope")
        self._mounts(self._mount_line(root="/system.slice/docker-abc.scope"))
        self._write(self.root, "cpu.max", "300000 100000\n")
        self.assertAlmostEqual(self._quota(), 3.0)

    def test_v1_cpu_controller(self):
        cpu = self.base / "cpu,cpuacct"
        self._membership("4:cpu,cpuacct:/docker/abc", "3:memory:/docker/abc")
        self._mounts(self._mount_line(fs_type="cgroup", mount_point=cpu,
                                      super_options="rw,cpu,cpuacct"))
        self._write(cpu / "docker" / "abc", "cpu.cfs_quota_us", "150000\n")
        self._write(cpu / "docker" / "abc", "cpu.cfs_period_us", "100000\n")
        self.assertAlmostEqual(self._quota(), 1.5)

    def test_v1_minus_one_is_no_quota(self):
        cpu = self.base / "cpu"
        self._membership("4:cpu,cpuacct:/docker/abc")
        self._mounts(self._mount_line(fs_type="cgroup", mount_point=cpu,
                                      super_options="rw,cpu,cpuacct"))
        self._write(cpu / "docker" / "abc", "cpu.cfs_quota_us", "-1\n")
        self._write(cpu / "docker" / "abc", "cpu.cfs_period_us", "100000\n")
        self.assertIsNone(self._quota())

    def test_without_mountinfo_the_conventional_root(self):
        self._membership("0::/pod/job")
        self._write(self.root / "pod" / "job", "cpu.max", "765000 100000\n")
        self.assertAlmostEqual(self._quota(mountinfo=False), 7.65)

    def test_malformed_input_leaves_the_team_alone(self):
        # fails open: a file it cannot read is no quota, never a smaller team
        self._membership("0::/pod/job")
        self._mounts(self._mount_line())
        self._write(self.root / "pod" / "job", "cpu.max", "lots 100000\n")
        self.assertIsNone(self._quota())
        self._write(self.root / "pod" / "job", "cpu.max", "765000 0\n")
        self.assertIsNone(self._quota())

    def test_no_membership_file(self):
        self.assertIsNone(self._quota())

    def test_not_linux(self):
        self._membership("0::/pod/job")
        self._mounts(self._mount_line())
        self._write(self.root / "pod" / "job", "cpu.max", "765000 100000\n")
        with mock.patch.object(resource_plan.sys, "platform", "darwin"):
            self.assertIsNone(self._quota())


class TeamBoundTest(unittest.TestCase):
    def test_bound_team(self):
        self.assertEqual(bound_team(48, 7.65, None), 7)     # the issue's quota, floored
        self.assertEqual(bound_team(48, None, 8), 8)        # a cpuset of 8
        self.assertEqual(bound_team(48, 7.65, 4), 4)        # both: the tighter
        self.assertEqual(bound_team(48, 0.5, None), 1)      # a fraction of a CPU: one thread
        self.assertEqual(bound_team(6, 16.0, 12), 6)        # limits looser than the cores
        self.assertEqual(bound_team(8, None, None), 8)

    def test_allowed_cpu_count_only_when_restricted(self):
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("no sched_getaffinity on this platform")
        with mock.patch.object(resource_plan.os, "sched_getaffinity", return_value=set(range(8))), \
             mock.patch.object(resource_plan.os, "cpu_count", return_value=96):
            self.assertEqual(allowed_cpu_count(), 8)
        with mock.patch.object(resource_plan.os, "sched_getaffinity", return_value=set(range(96))), \
             mock.patch.object(resource_plan.os, "cpu_count", return_value=96):
            self.assertIsNone(allowed_cpu_count())

    def test_team_cpu_count_sees_the_quota(self):
        with mock.patch("resource_plan.performance_core_count", return_value=None), \
             mock.patch("resource_plan.physical_cpu_count", return_value=48), \
             mock.patch("resource_plan.cgroup_cpu_quota", return_value=7.65), \
             mock.patch("resource_plan.allowed_cpu_count", return_value=None):
            self.assertEqual(resource_plan.team_cpu_count(), 7)


class LauncherTeamTest(unittest.TestCase):
    """The engine's environment: OMP_NUM_THREADS from the bounded team."""

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_cgroup_cpu_under_test", str(HERE / "coli"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    @staticmethod
    def args():
        return types.SimpleNamespace(model="/x", ram=None, ctx=None, ngen=None, temp=None, cap=None)

    def test_qwen36_in_a_quota_container(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("resource_plan.performance_core_count", return_value=None), \
             mock.patch("resource_plan.physical_cpu_count", return_value=48), \
             mock.patch("resource_plan.cgroup_cpu_quota", return_value=7.65), \
             mock.patch("resource_plan.allowed_cpu_count", return_value=None):
            env = self.coli.env_for_engine(self.args(), "qwen36")
        self.assertEqual(env.get("OMP_NUM_THREADS"), "7")

    def test_qwen36_in_a_cpuset_container(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("resource_plan.performance_core_count", return_value=None), \
             mock.patch("resource_plan.physical_cpu_count", return_value=48), \
             mock.patch("resource_plan.cgroup_cpu_quota", return_value=None), \
             mock.patch("resource_plan.allowed_cpu_count", return_value=8):
            env = self.coli.env_for_engine(self.args(), "qwen36")
        self.assertEqual(env.get("OMP_NUM_THREADS"), "8")

    def test_an_explicit_choice_still_wins(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "24"}, clear=True), \
             mock.patch("resource_plan.cgroup_cpu_quota", return_value=7.65):
            env = self.coli.env_for_engine(self.args(), "qwen36")
        self.assertEqual(env.get("OMP_NUM_THREADS"), "24")

    def test_a_profile_measured_without_limits_stays_out_of_a_quota_container(self):
        """--auto-tier lays a measured profile over the default team. A profile's
        fingerprint hashes the plan's cpu block: one measured on the bare host (24
        threads of 48 cores) is found again there, and not in the pod of #2914."""
        import autotune
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"       # the smallest model build_plan reads
            model.mkdir()
            (model / "config.json").write_text(json.dumps({
                "model_type": "glm_moe_dsa", "num_hidden_layers": 2, "n_routed_experts": 2,
                "kv_lora_rank": 4, "qk_rope_head_dim": 2, "qk_nope_head_dim": 3,
                "v_head_dim": 5, "num_attention_heads": 2}), encoding="utf-8")
            header, end = {}, 0
            for name, size in (("model.embed_tokens.weight", 100),
                               ("model.layers.0.self_attn.q_a_proj.weight", 200),
                               ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
                               ("model.layers.1.mlp.experts.0.up_proj.weight", 30)):
                header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [end, end + size]}
                end += size
            raw = json.dumps(header).encode()
            (model / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(end))
            engine = Path(tmp) / "engine"
            engine.write_bytes(b"engine")
            args = types.SimpleNamespace(model=str(model), ram=None, ctx=None, ngen=None,
                                         temp=None, cap=None, auto_tier=True)

            def launch(quota, save=None, **environ):
                """The engine's team under `quota`, and whether a profile was applied;
                `save` is a profile's winner, saved for that machine before the launch."""
                log = io.StringIO()
                with mock.patch("resource_plan.performance_core_count", return_value=None), \
                     mock.patch("resource_plan.physical_cpu_count", return_value=48), \
                     mock.patch("resource_plan.cgroup_cpu_quota", return_value=quota), \
                     mock.patch("resource_plan.allowed_cpu_count", return_value=None), \
                     mock.patch.object(self.coli, "engine_for", return_value=str(engine)), \
                     mock.patch.object(self.coli, "plan_cuda_enabled", return_value=False), \
                     contextlib.redirect_stderr(log):
                    plan = resource_plan.build_plan(model, available_memory=16 * resource_plan.GB,
                                                    available_disk=1, gpus=[], cpu_sockets=1)
                    with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp, "LOCALAPPDATA": tmp,
                                                      **environ}, clear=True):
                        if save:
                            autotune.save_profile({
                                "schema_version": autotune.SCHEMA_VERSION, "accepted": True,
                                "gain": 0.10, "winner": {"env": save},
                                "fingerprint": autotune.machine_fingerprint(plan, str(model),
                                                                            str(engine))})
                        env = self.coli.env_for_engine(args, "qwen36", plan=plan)
                return env.get("OMP_NUM_THREADS"), "applied measured profile" in log.getvalue()

            self.assertEqual(launch(None, save={"OMP_NUM_THREADS": "24"}), ("24", True))
            self.assertEqual(launch(None), ("24", True))        # found again on the bare host
            self.assertEqual(launch(7.65), ("7", False))        # and not in the pod
            self.assertEqual(launch(7.65, OMP_NUM_THREADS="24"), ("24", False))  # asked for by name


if __name__ == "__main__":
    unittest.main()
