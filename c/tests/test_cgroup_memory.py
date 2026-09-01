"""Container-aware memory admission: min(host MemAvailable, finite cgroup headroom).

Pure fixtures: every kernel input (``/proc/self/cgroup``, ``/proc/self/mountinfo``,
``/proc/meminfo`` and the cgroup control files) is a file under a temporary
directory, so the tests run identically on any platform, and every expectation
is derived from the fixture numbers rather than from the implementation.
"""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from doctor import run_doctor
from resource_plan import (
    GB,
    CgroupAccessError,
    CgroupError,
    CgroupFormatError,
    _CGROUP_UINT64_MAX,
    _CGROUP_V1_UNLIMITED_MIN,
    _MEMINFO_MAX_BYTES,
    _cgroup_memory_remaining,
    memory_available,
)

HOST_AVAILABLE = 16_384 * 1024


class CgroupMemoryAdmissionTest(unittest.TestCase):
    """Fixture-driven checks of the cgroup half of the admission minimum."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "cgroup"
        self.root.mkdir()
        self.proc_cgroup = self.base / "proc-self-cgroup"
        self.proc_mountinfo = self.base / "proc-self-mountinfo"
        self.meminfo = self.base / "meminfo"
        self.meminfo.write_text(
            "MemTotal: 32768 kB\nMemAvailable: 16384 kB\n", encoding="ascii"
        )
        # Cgroups are Linux-only and the probe is gated on the platform; pin it
        # so the fixtures exercise the same code everywhere.
        platform = mock.patch("resource_plan.sys.platform", "linux")
        platform.start()
        self.addCleanup(platform.stop)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _proc_escape(value):
        escapes = {"\t": r"\011", "\n": r"\012", " ": r"\040", "\\": r"\134"}
        return "".join(escapes.get(character, character) for character in str(value))

    def _membership(self, *lines):
        self.proc_cgroup.write_text("\n".join(lines) + "\n", encoding="ascii")

    def _mount_line(
        self,
        *,
        mount_id=30,
        parent_id=1,
        root="/",
        mount_point=None,
        fs_type="cgroup2",
        mount_options="rw,nosuid,nodev",
        optional_fields=(),
        super_options="rw",
    ):
        mount_point = self.root if mount_point is None else Path(mount_point)
        optional = "".join(f"{field} " for field in optional_fields)
        return (
            f"{mount_id} {parent_id} 0:{mount_id} {self._proc_escape(root)} "
            f"{self._proc_escape(mount_point)} {mount_options} {optional}- "
            f"{fs_type} {fs_type} {super_options}"
        )

    def _mounts(self, *lines):
        self.proc_mountinfo.write_text("\n".join(lines) + "\n", encoding="ascii")

    @staticmethod
    def _write_pair(directory, limit_name, current_name, limit, current):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / limit_name).write_text(f"{limit}\n", encoding="ascii")
        (directory / current_name).write_text(f"{current}\n", encoding="ascii")

    def _write_v2(self, directory, limit, current):
        self._write_pair(directory, "memory.max", "memory.current", limit, current)

    def _write_v1(self, directory, limit, current):
        self._write_pair(
            directory, "memory.limit_in_bytes", "memory.usage_in_bytes", limit, current
        )

    @staticmethod
    def _write_stat(directory, **counters):
        directory.mkdir(parents=True, exist_ok=True)
        lines = "".join(f"{key} {value}\n" for key, value in counters.items())
        (directory / "memory.stat").write_text(lines, encoding="ascii")

    def _remaining(self):
        return _cgroup_memory_remaining(
            cgroup_root=self.root,
            proc_cgroup_path=self.proc_cgroup,
            proc_mountinfo_path=self.proc_mountinfo,
        )

    def _available(self):
        return memory_available(
            meminfo_path=self.meminfo,
            cgroup_root=self.root,
            proc_cgroup_path=self.proc_cgroup,
            proc_mountinfo_path=self.proc_mountinfo,
        )

    def _standard_v2(self, member="/tenant/job"):
        self._membership(f"0::{member}")
        self._mounts(self._mount_line())

    # -- the admission rule ---------------------------------------------------

    def test_v2_finite_budget_clamps_host_memavailable(self):
        self._standard_v2()
        self._write_v2(self.root / "tenant" / "job", 12_000_000, 3_000_000)
        self.assertEqual(self._remaining(), 9_000_000)
        self.assertEqual(self._available(), 9_000_000)

    def test_host_memavailable_wins_when_lower_than_cgroup_headroom(self):
        self._standard_v2()
        self._write_v2(self.root / "tenant" / "job", 100_000_000, 1)
        self.assertEqual(self._available(), HOST_AVAILABLE)

    def test_mount_root_is_removed_before_suffix_is_appended_to_mount_point(self):
        self._membership("0::/tenant/job")
        self._mounts(self._mount_line(root="/tenant"))
        self._write_v2(self.root / "job", 12_000_000, 4_000_000)
        # A naive mount-point + membership join would look under tenant/job.
        self._write_v2(self.root / "tenant" / "job", 99_000_000, 1)
        self.assertEqual(self._remaining(), 8_000_000)

    def test_namespaced_mount_root_equal_to_membership_maps_to_mount_point(self):
        self._membership("0::/host/tenant/job")
        self._mounts(self._mount_line(root="/host/tenant/job"))
        self._write_v2(self.root, 10_000_000, 4_000_000)
        self.assertEqual(self._remaining(), 6_000_000)

    def test_ancestor_limit_hidden_by_deeper_mount_root_still_binds(self):
        # A3: two cgroup2 mounts expose one kernel tree. The deep one (root
        # /tenant/job) cannot show /tenant, whose 5 MB limit with 1 MB charged
        # is the binding 4 MB; preferring it would over-admit by 5 MB.
        self._membership("0::/tenant/job")
        broad = self.base / "broad"
        deep = self.base / "deep"
        self._write_v2(broad / "tenant", 5_000_000, 1_000_000)
        self._write_v2(broad / "tenant" / "job", 12_000_000, 3_000_000)
        self._write_v2(deep, 12_000_000, 3_000_000)
        broad_line = self._mount_line(mount_id=31, root="/", mount_point=broad)
        deep_line = self._mount_line(mount_id=32, root="/tenant/job", mount_point=deep)
        for lines in ((broad_line, deep_line), (deep_line, broad_line)):
            with self.subTest(first=lines[0].split()[0]):
                self._mounts(*lines)
                self.assertEqual(self._remaining(), 4_000_000)
                self.assertEqual(self._available(), 4_000_000)

    def test_two_mounts_of_one_root_take_the_minimum_instead_of_refusing(self):
        # Mounting cgroupfs twice is legitimate (a read-only copy for a
        # sidecar, say) and must not fail closed; the tighter view wins.
        self._membership("0::/tenant/job", "7:cpu,memory:/tenant/job")
        for version in ("v2", "v1"):
            with self.subTest(version=version):
                first = self.base / f"{version}-first"
                second = self.base / f"{version}-second"
                write = self._write_v2 if version == "v2" else self._write_v1
                write(first / "job", 8_000_000, 1_000_000)
                write(second / "job", 5_000_000, 1_000_000)
                fs_type = "cgroup2" if version == "v2" else "cgroup"
                super_options = "rw" if version == "v2" else "rw,cpu,memory"
                self._mounts(
                    self._mount_line(
                        mount_id=31,
                        root="/tenant",
                        mount_point=first,
                        fs_type=fs_type,
                        super_options=super_options,
                    ),
                    self._mount_line(
                        mount_id=32,
                        root="/tenant",
                        mount_point=second,
                        fs_type=fs_type,
                        super_options=super_options,
                    ),
                )
                self.assertEqual(self._remaining(), 4_000_000)

    def test_hybrid_v1_net_cls_record_beside_v2_membership_is_not_a_refusal(self):
        # This development host's real layout: a VPN client mounts a v1 net_cls
        # hierarchy next to the systemd cgroup2 tree, and mountinfo carries
        # optional "shared:N" fields. Only the memory controller matters; a v1
        # record for another controller is ignored, never a reason to fail.
        scope = (
            "user.slice/user-1000.slice/user@1000.service/app.slice/vte-spawn-1.scope"
        )
        self._membership("1:net_cls:/", f"0::/{scope}")
        net_cls = self.base / "net_cls"
        net_cls.mkdir()
        self._mounts(
            self._mount_line(
                mount_id=38,
                parent_id=28,
                mount_options="rw,nosuid,nodev,noexec,relatime",
                optional_fields=("shared:10",),
                super_options="rw,nsdelegate,memory_recursiveprot",
            ),
            self._mount_line(
                mount_id=2470,
                parent_id=34,
                mount_point=net_cls,
                fs_type="cgroup",
                mount_options="rw,relatime",
                optional_fields=("shared:1651",),
                super_options="rw,net_cls",
            ),
        )
        self._write_v2(
            self.root / "user.slice" / "user-1000.slice", 30_000_000, 10_000_000
        )
        self._write_v2(self.root.joinpath(*scope.split("/")), "max", 2_000_000)
        self.assertEqual(self._remaining(), 20_000_000)
        self.assertEqual(self._available(), HOST_AVAILABLE)

    def test_nested_v2_uses_tightest_visible_ancestor_budget(self):
        self._standard_v2()
        self._write_v2(self.root, "max", 9_000_000)
        self._write_v2(self.root / "tenant", 20_000_000, 14_000_000)
        self._write_v2(self.root / "tenant" / "job", 12_000_000, 3_000_000)
        self.assertEqual(self._remaining(), 6_000_000)

    def test_unlimited_v2_leaf_still_honors_finite_parent(self):
        self._standard_v2()
        self._write_v2(self.root / "tenant", 9_000_000, 4_000_000)
        self._write_v2(self.root / "tenant" / "job", "max", 3_000_000)
        self.assertEqual(self._remaining(), 5_000_000)

    def test_v2_max_is_unlimited_and_does_not_override_host(self):
        self._standard_v2(member="/")
        self._write_v2(self.root, "max", 1234)
        self.assertIsNone(self._remaining())
        self.assertEqual(self._available(), HOST_AVAILABLE)

    def test_v2_counter_accepts_no_trailing_newline(self):
        self._standard_v2(member="/")
        (self.root / "memory.max").write_text("10000", encoding="ascii")
        (self.root / "memory.current").write_text("4000", encoding="ascii")
        self.assertEqual(self._remaining(), 6_000)

    def test_v2_current_at_or_above_limit_is_authoritative_zero_headroom(self):
        self._standard_v2(member="/")
        for current in (10_000, 10_001):
            with self.subTest(current=current):
                self._write_v2(self.root, 10_000, current)
                self.assertEqual(self._remaining(), 0)
                self.assertEqual(self._available(), 0)

    def test_v1_combined_controller_mount_is_discovered_from_super_options(self):
        self._membership("7:cpu,memory,blkio:/tenant/job")
        self._mounts(
            self._mount_line(fs_type="cgroup", super_options="rw,cpu,memory,blkio")
        )
        self._write_v1(self.root / "tenant" / "job", 11_000_000, 4_000_000)
        self.assertEqual(self._remaining(), 7_000_000)

    def test_v1_combined_controller_mount_is_discovered_from_mount_options(self):
        self._membership("7:memory,cpuacct:/tenant/job")
        self._mounts(
            self._mount_line(
                fs_type="cgroup", mount_options="rw,memory,cpuacct", super_options="rw"
            )
        )
        self._write_v1(self.root / "tenant" / "job", 9_000_000, 3_000_000)
        self.assertEqual(self._remaining(), 6_000_000)

    def test_v1_mount_root_and_parent_minimum_are_honored(self):
        self._membership("7:memory,blkio:/tenant/job")
        self._mounts(
            self._mount_line(
                root="/tenant", fs_type="cgroup", super_options="rw,memory,blkio"
            )
        )
        self._write_v1(self.root, 10_000_000, 7_000_000)
        self._write_v1(self.root / "job", 8_000_000, 2_000_000)
        self.assertEqual(self._remaining(), 3_000_000)

    def test_v1_kernel_unlimited_sentinel_does_not_clamp_host(self):
        self._membership("7:memory:/")
        self._mounts(self._mount_line(fs_type="cgroup", super_options="rw,memory"))
        self._write_v1(self.root, _CGROUP_V1_UNLIMITED_MIN, 1234)
        self.assertIsNone(self._remaining())
        self.assertEqual(self._available(), HOST_AVAILABLE)

    def test_v1_current_at_or_above_limit_is_authoritative_zero_headroom(self):
        self._membership("7:memory:/")
        self._mounts(self._mount_line(fs_type="cgroup", super_options="rw,memory"))
        for current in (10_000, 10_001):
            with self.subTest(current=current):
                self._write_v1(self.root, 10_000, current)
                self.assertEqual(self._remaining(), 0)
                self.assertEqual(self._available(), 0)

    def test_v2_controls_are_authoritative_over_stale_v1_membership(self):
        v1_root = self.base / "v1"
        self._membership("0::/", "7:memory:/legacy")
        self._mounts(
            self._mount_line(mount_id=30),
            self._mount_line(
                mount_id=31,
                fs_type="cgroup",
                mount_point=v1_root,
                super_options="rw,memory",
            ),
        )
        self._write_v2(self.root, "max", 1)
        self._write_v1(v1_root / "legacy", 4_000_000, 3_000_000)
        self.assertIsNone(self._remaining())

    def test_v1_is_used_when_v2_mount_has_no_memory_controls(self):
        v1_root = self.base / "v1"
        self._membership("0::/unified", "7:memory:/legacy")
        self._mounts(
            self._mount_line(mount_id=30),
            self._mount_line(
                mount_id=31,
                fs_type="cgroup",
                mount_point=v1_root,
                super_options="rw,cpu,memory",
            ),
        )
        (self.root / "unified").mkdir()
        self._write_v1(v1_root / "legacy", 8_000_000, 3_000_000)
        self.assertEqual(self._remaining(), 5_000_000)

    # -- page cache is reclaimable headroom (A2) -----------------------------

    def test_v2_inactive_file_cache_is_not_spent_budget(self):
        # limit 12 MB, 9 MB charged of which 5 MB is clean file cache: the
        # kernel reclaims that cache before enforcing the limit, so the working
        # set is 4 MB and 8 MB remains -- not 3 MB.
        self._standard_v2(member="/")
        self._write_v2(self.root, 12_000_000, 9_000_000)
        self.assertEqual(self._remaining(), 3_000_000)
        self._write_stat(
            self.root, anon=4_000_000, file=5_000_000, inactive_file=5_000_000
        )
        self.assertEqual(self._remaining(), 8_000_000)
        self.assertEqual(self._available(), 8_000_000)

    def test_v1_subtracts_hierarchical_total_inactive_file_not_the_local_figure(self):
        # usage_in_bytes is hierarchical, so it pairs with total_inactive_file
        # (3 MB), not the cgroup-local inactive_file (1 MB): 10 - (8 - 3) = 5.
        self._membership("7:memory:/")
        self._mounts(self._mount_line(fs_type="cgroup", super_options="rw,memory"))
        self._write_v1(self.root, 10_000_000, 8_000_000)
        self._write_stat(
            self.root,
            cache=3_000_000,
            inactive_file=1_000_000,
            total_inactive_file=3_000_000,
        )
        self.assertEqual(self._remaining(), 5_000_000)

    def test_inactive_file_beyond_charged_usage_leaves_the_whole_limit(self):
        self._standard_v2(member="/")
        self._write_v2(self.root, 12_000_000, 2_000_000)
        self._write_stat(self.root, inactive_file=5_000_000)
        self.assertEqual(self._remaining(), 12_000_000)

    def test_page_cache_is_subtracted_per_ancestor_level(self):
        # Parent: 20 - (15 - 10) = 15 MB; leaf: 12 - 3 = 9 MB. Without the
        # parent's statistics the parent would bind the result at 5 MB.
        self._standard_v2()
        self._write_v2(self.root / "tenant", 20_000_000, 15_000_000)
        self._write_v2(self.root / "tenant" / "job", 12_000_000, 3_000_000)
        self.assertEqual(self._remaining(), 5_000_000)
        self._write_stat(self.root / "tenant", inactive_file=10_000_000)
        self.assertEqual(self._remaining(), 9_000_000)

    def test_memory_stat_without_the_key_subtracts_nothing(self):
        self._standard_v2(member="/")
        self._write_v2(self.root, 12_000_000, 9_000_000)
        self._write_stat(self.root, anon=9_000_000, file=0)
        self.assertEqual(self._remaining(), 3_000_000)

    def test_malformed_memory_stat_is_a_typed_refusal(self):
        self._standard_v2(member="/")
        self._write_v2(self.root, 12_000_000, 9_000_000)
        malformed = (
            "",
            "inactive_file\n",
            "inactive_file x\n",
            "inactive_file 1 2\n",
            "inactive_file -1\n",
            "inactive_file 1\ninactive_file 2\n",
            "inactive_file 1\n\n",
            f"inactive_file {_CGROUP_UINT64_MAX + 1}\n",
            "inactive_file 1\n" + "x" * (64 * 1024),
        )
        for payload in malformed:
            with self.subTest(payload=payload[:30]):
                (self.root / "memory.stat").write_text(payload, encoding="ascii")
                with self.assertRaises(CgroupFormatError):
                    self._remaining()
                with self.assertRaises(CgroupFormatError):
                    self._available()
        (self.root / "memory.stat").write_bytes(b"inactive_file 1\n\xff\n")
        with self.assertRaises(CgroupFormatError):
            self._remaining()

    # -- present but untrustworthy controls fail closed ------------------------

    def test_malformed_control_values_raise_typed_errors_and_fail_closed(self):
        self._standard_v2(member="/")
        malformed = (
            "",
            "-1",
            "+1",
            "1.5",
            "garbage",
            " 1",
            "1 ",
            "\t1",
            "1\n",
            "1\r",
            "max ",
            str(_CGROUP_UINT64_MAX + 1),
        )
        for value in malformed:
            with self.subTest(limit=value):
                self._write_v2(self.root, value, 1)
                with self.assertRaisesRegex(
                    CgroupFormatError, "malformed cgroup memory limit"
                ):
                    self._remaining()
                with self.assertRaises(CgroupFormatError):
                    self._available()
        for value in malformed:
            with self.subTest(current=value):
                self._write_v2(self.root, 10_000, value)
                with self.assertRaisesRegex(
                    CgroupFormatError, "malformed cgroup memory usage"
                ):
                    self._remaining()
                with self.assertRaises(CgroupFormatError):
                    self._available()

    def test_max_and_v1_unlimited_still_validate_present_usage(self):
        cases = (("v2", "max", "bad"), ("v1", _CGROUP_V1_UNLIMITED_MIN, "bad"))
        for version, limit, current in cases:
            with self.subTest(version=version):
                if version == "v2":
                    self._membership("0::/")
                    self._mounts(self._mount_line())
                    self._write_v2(self.root, limit, current)
                else:
                    self._membership("7:memory:/")
                    self._mounts(
                        self._mount_line(fs_type="cgroup", super_options="rw,memory")
                    )
                    self._write_v1(self.root, limit, current)
                with self.assertRaisesRegex(
                    CgroupFormatError, "malformed cgroup memory usage"
                ):
                    self._remaining()

    def test_malformed_leaf_is_not_masked_by_valid_parent(self):
        self._standard_v2()
        self._write_v2(self.root / "tenant", 8_000_000, 3_000_000)
        self._write_v2(self.root / "tenant" / "job", "bad", 1)
        with self.assertRaisesRegex(CgroupFormatError, "malformed cgroup memory limit"):
            self._remaining()

    def test_oversized_and_non_ascii_control_values_fail_closed(self):
        self._standard_v2(member="/")
        for value in (b"9" * 256, b"10\xff\n"):
            with self.subTest(value=value[:8]):
                (self.root / "memory.max").write_bytes(value)
                (self.root / "memory.current").write_bytes(b"1\n")
                with self.assertRaises(CgroupFormatError):
                    self._remaining()
                with self.assertRaises(CgroupFormatError):
                    self._available()

    def test_incomplete_control_pair_fails_closed_in_both_directions(self):
        self._standard_v2(member="/")
        for present, missing in (
            ("memory.max", "memory.current"),
            ("memory.current", "memory.max"),
        ):
            with self.subTest(missing=missing):
                for name in ("memory.max", "memory.current"):
                    (self.root / name).unlink(missing_ok=True)
                (self.root / present).write_text("10000\n", encoding="ascii")
                with self.assertRaisesRegex(CgroupAccessError, "incomplete"):
                    self._remaining()
                with self.assertRaises(CgroupAccessError):
                    self._available()

    def test_unreadable_present_control_file_fails_closed(self):
        self._standard_v2(member="/")
        self._write_v2(self.root, 10_000, 4_000)
        real_open = Path.open

        def deny_usage(path, *args, **kwargs):
            if path.name == "memory.current":
                raise PermissionError("fixture denied")
            return real_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", deny_usage):
            with self.assertRaisesRegex(
                CgroupAccessError, "cannot read cgroup memory usage"
            ):
                self._remaining()
            with self.assertRaises(CgroupAccessError):
                self._available()

    # -- present but malformed procfs inputs are refusals, not "unconstrained" -

    def test_malformed_present_membership_is_never_treated_as_unconstrained(self):
        self._mounts(self._mount_line())
        malformed = (
            b"",
            b"not:a:valid:line\n",
            b"0::relative\n",
            b"0::/a/../b\n",
            b"0::/a/./b\n",
            b"0::/a//b\n",
            b"0::/a/\n",
            b"0:memory:/\n",
            b"1::/\n",
            b"1:memory,memory:/\n",
            b"0::/\n0::/other\n",
            b"0::/\n0:cpu:/\n",
            b"\n",
            b"1:memory:/a\n2:cpu,memory:/b\n",
            b"0::/ok\n\n",
            b"0::/bad\x00name\n",
            b"\xff::/bad-id\n",
            b"1:memo\xffry:/bad-controller\n",
        )
        for payload in malformed:
            with self.subTest(payload=payload[:30]):
                self.proc_cgroup.write_bytes(payload)
                with self.assertRaises(CgroupError):
                    self._remaining()
                with self.assertRaises(CgroupError):
                    self._available()

    def test_oversized_membership_is_a_format_error(self):
        self._mounts(self._mount_line())
        self.proc_cgroup.write_bytes(b"0::/" + b"x" * (64 * 1024) + b"\n")
        with self.assertRaisesRegex(CgroupFormatError, "oversized"):
            self._remaining()

    def test_path_depth_limit_is_enforced_before_filesystem_access(self):
        self._mounts(self._mount_line())
        member = "/" + "/".join(f"level-{index}" for index in range(64))
        self._membership(f"0::{member}")
        with self.assertRaisesRegex(CgroupFormatError, "nesting exceeds"):
            self._remaining()

    def test_malformed_present_mountinfo_is_never_a_compatibility_signal(self):
        self._membership("0::/")
        self._write_v2(self.root, 9_000_000, 2_000_000)
        valid = self._mount_line()
        malformed = (
            "",
            "garbage",
            valid.replace(" - ", " "),
            valid.replace("30 1", "x 1", 1),
            valid.replace("0:30", "bad-device", 1),
            valid.replace(" / ", " /root/../escape ", 1),
            valid.replace(self._proc_escape(self.root), "relative", 1),
            valid.replace(" rw,nosuid", "  rw,nosuid", 1),
            valid + "\n" + valid,
        )
        for payload in malformed:
            with self.subTest(payload=payload[:40]):
                self.proc_mountinfo.write_text(payload + "\n", encoding="ascii")
                with self.assertRaises(CgroupFormatError):
                    self._remaining()
                with self.assertRaises(CgroupFormatError):
                    self._available()

    def test_inherited_cgroup_namespace_mount_root_is_a_typed_refusal(self):
        self._membership("0::/job")
        self._mounts(self._mount_line(root="/../.."))
        with self.assertRaisesRegex(CgroupAccessError, "remount cgroupfs"):
            self._remaining()

    def test_mountinfo_accepts_unrelated_non_path_roots(self):
        self._membership("0::/")
        unrelated = "29 1 0:5 net:[4026531833] /run/netns rw - nsfs nsfs rw"
        self._mounts(unrelated, self._mount_line())
        self._write_v2(self.root, 7_000_000, 2_000_000)
        self.assertEqual(self._remaining(), 5_000_000)

    def test_unrelated_raw_unicode_mount_does_not_block_cgroup_discovery(self):
        self._membership("0::/")
        unrelated = b"29 1 0:5 / /mnt/caf\xc3\xa9-\xff rw - ext4 /dev/example rw\n"
        cgroup = self._mount_line().encode("ascii") + b"\n"
        self.proc_mountinfo.write_bytes(unrelated + cgroup)
        self._write_v2(self.root, 7_000_000, 2_000_000)
        self.assertEqual(self._remaining(), 5_000_000)

    def test_mountinfo_escapes_are_decoded_for_cgroup_mount_paths(self):
        # mountinfo(5) escapes space, tab, newline and backslash as octal; the
        # membership path in /proc/self/cgroup is raw. Both must meet.
        mount_point = self.base / "mount point\twith\nall\\four"
        self._membership("0::/sp ace\ttab\\bs/job")
        self._mounts(self._mount_line(root="/sp ace\ttab\\bs", mount_point=mount_point))
        self._write_v2(mount_point / "job", 7_000_000, 2_000_000)
        self.assertEqual(self._remaining(), 5_000_000)

    # -- absent procfs inputs: compatibility, never a refusal -------------------

    def test_mountinfo_absence_alone_enables_fixed_root_compatibility(self):
        self._membership("0::/tenant/job")
        self._write_v2(self.root / "tenant" / "job", 9_000_000, 2_000_000)
        self.assertFalse(self.proc_mountinfo.exists())
        self.assertEqual(self._remaining(), 7_000_000)

    def test_absent_mountinfo_compatibility_supports_namespaced_root(self):
        self._membership("0::/host/tenant/job")
        self._write_v2(self.root, 9_000_000, 2_000_000)
        self.assertFalse(self.proc_mountinfo.exists())
        self.assertEqual(self._remaining(), 7_000_000)

    def test_absent_mountinfo_compatibility_supports_v1_fixed_root(self):
        self._membership("7:cpu,memory:/tenant/job")
        self._write_v1(self.root / "memory" / "tenant" / "job", 8_000_000, 3_000_000)
        self.assertFalse(self.proc_mountinfo.exists())
        self.assertEqual(self._remaining(), 5_000_000)

    def test_absent_membership_and_mountinfo_can_probe_namespaced_root(self):
        self._write_v2(self.root, 8_000_000, 3_000_000)
        self.assertFalse(self.proc_cgroup.exists())
        self.assertFalse(self.proc_mountinfo.exists())
        self.assertEqual(self._remaining(), 5_000_000)

    def test_absent_membership_with_present_memory_mount_fails_closed(self):
        self._mounts(self._mount_line())
        self._write_v2(self.root, 8_000_000, 3_000_000)
        self.assertFalse(self.proc_cgroup.exists())
        with self.assertRaisesRegex(CgroupAccessError, "membership is unavailable"):
            self._remaining()
        with self.assertRaises(CgroupAccessError):
            self._available()

    def test_present_membership_missing_record_for_v2_mount_fails_closed(self):
        self._membership("1:cpu:/")
        self._mounts(self._mount_line())
        with self.assertRaisesRegex(CgroupAccessError, "v2 mount.*membership"):
            self._remaining()

    def test_present_membership_missing_record_for_v1_memory_mount_fails_closed(self):
        self._membership("1:cpu:/")
        self._mounts(self._mount_line(fs_type="cgroup", super_options="rw,cpu,memory"))
        with self.assertRaisesRegex(CgroupAccessError, "v1 memory mount.*membership"):
            self._remaining()

    def test_membership_hidden_by_every_visible_mount_fails_closed(self):
        # A cgroup2 mount is present but shows a sibling subtree only: the
        # process's own limit is unverifiable, which is not "unlimited".
        self._membership("0::/tenant/job")
        self._mounts(self._mount_line(root="/other"))
        with self.assertRaisesRegex(CgroupAccessError, "no visible cgroup2 mount"):
            self._remaining()

    def test_no_cgroup_mount_at_all_is_host_only_not_a_refusal(self):
        # A sandbox that mounts no cgroupfs offers nothing to read; that is an
        # absent input (the host measurement stands), not a malformed one.
        self._membership("0::/tenant/job")
        self._mounts("29 1 0:5 / /run rw - tmpfs tmpfs rw")
        self.assertIsNone(self._remaining())
        self.assertEqual(self._available(), HOST_AVAILABLE)

    def test_absent_control_files_leave_host_measurement_unchanged(self):
        self._standard_v2()
        (self.root / "tenant" / "job").mkdir(parents=True)
        self.assertIsNone(self._remaining())
        self.assertEqual(self._available(), HOST_AVAILABLE)

    # -- the host half of the minimum -------------------------------------------

    def test_finite_cgroup_is_upper_bound_when_linux_meminfo_is_hidden(self):
        self._standard_v2(member="/")
        self._write_v2(self.root, 7_000_000, 2_000_000)
        self.meminfo.unlink()
        self.assertEqual(self._available(), 5_000_000)

    def test_nothing_measurable_is_unknown_not_zero(self):
        # No /proc/meminfo and no finite cgroup limit is the tri-state's None,
        # so the planner keeps its historical fallback instead of reading
        # "exhausted" (A1).
        self._standard_v2(member="/")
        self._write_v2(self.root, "max", 1)
        self.meminfo.unlink()
        self.assertIsNone(self._available())

    def test_meminfo_without_memavailable_is_unknown_not_malformed(self):
        # Kernels before 3.14 (and Cygwin's emulation) have no MemAvailable;
        # that is the pre-existing fallback path, not a corrupt file.
        self._standard_v2(member="/")
        self._write_v2(self.root, "max", 1)
        self.meminfo.write_text(
            "MemTotal: 32768 kB\nMemFree: 100 kB\n", encoding="ascii"
        )
        self.assertIsNone(self._available())
        self._write_v2(self.root, 7_000_000, 2_000_000)
        self.assertEqual(self._available(), 5_000_000)

    def test_memavailable_is_parsed_as_one_strict_bounded_u64_field(self):
        maximum_kib = _CGROUP_UINT64_MAX // 1024
        for line, expected in (
            ("MemAvailable:\t16384 kB\t", 16_384 * 1024),
            (f"MemAvailable: {maximum_kib} kB", maximum_kib * 1024),
        ):
            with self.subTest(line=line[:40]):
                self.meminfo.write_text(
                    f"MemTotal: 32768 kB\n{line}\n", encoding="ascii"
                )
                self.assertEqual(self._available(), expected)

    def test_malformed_duplicate_and_overflowing_memavailable_fail_closed(self):
        overflow = _CGROUP_UINT64_MAX // 1024 + 1
        malformed = (
            "MemAvailable: not-a-number kB\n",
            "MemAvailable: 100 MB\n",
            "MemAvailable: 100 kB trailing\n",
            "MemAvailable: 100 kB\nMemAvailable: 200 kB\n",
            f"MemAvailable: {overflow} kB\n",
            f"MemAvailable: {'9' * 5000} kB\n",
        )
        for text in malformed:
            with self.subTest(text=text[:50]):
                self.meminfo.write_text(text, encoding="ascii")
                with self.assertRaises(CgroupFormatError):
                    self._available()

    def test_oversized_and_non_ascii_meminfo_fail_closed(self):
        payloads = (
            b"MemAvailable: 100 kB\n" + b"X" * _MEMINFO_MAX_BYTES,
            b"MemAvailable: 100 kB\n\xff\n",
        )
        for payload in payloads:
            with self.subTest(size=len(payload)):
                self.meminfo.write_bytes(payload)
                with self.assertRaises(CgroupFormatError):
                    self._available()

    def test_cgroup_probe_is_linux_only(self):
        # A fixture that would refuse on Linux is never consulted on a platform
        # without cgroups; the host figure stands alone.
        self._standard_v2(member="/")
        self._write_v2(self.root, "bad", 1)
        with mock.patch("resource_plan.sys.platform", "darwin"):
            self.assertEqual(self._available(), HOST_AVAILABLE)


def _write_shard(path, tensors):
    offset = 0
    header = {}
    payload = b""
    for name, size in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        payload += b"\0" * size
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class DoctorMemoryAdmissionTest(unittest.TestCase):
    """The tri-state reaches doctor unchanged (A1).

    unknown (None) is the historical warn + 8 GB fallback; a present but
    malformed cgroup input is a typed refusal whose reason the user sees under
    memory.ram; a finite figure is the ceiling the budget is checked against.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.model = root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "glm_moe_dsa",
                    "num_hidden_layers": 2,
                    "n_routed_experts": 2,
                    "kv_lora_rank": 4,
                    "qk_rope_head_dim": 2,
                    "qk_nope_head_dim": 3,
                    "v_head_dim": 5,
                    "num_attention_heads": 2,
                }
            )
        )
        (self.model / "tokenizer.json").write_text("{}")
        _write_shard(
            self.model / "model.safetensors",
            [
                ("model.embed_tokens.weight", 100),
                ("model.norm.weight", 8),
                ("lm_head.weight", 100),
                ("model.layers.0.self_attn.q_a_proj.weight", 200),
                ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
                ("model.layers.1.mlp.experts.0.up_proj.weight", 30),
                ("model.layers.1.mlp.experts.1.gate_proj.weight", 30),
                ("model.layers.1.mlp.experts.1.up_proj.weight", 30),
            ],
        )
        self.engine = root / "glm"
        self.engine.write_text("#!/bin/sh\nexit 0\n")
        self.engine.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def _doctor(self, probe, **overrides):
        arguments = {
            "model": self.model,
            "ram_gb": 0,
            "context": 32,
            "gpu_indices": [],
            "vram_gb": 0,
            "engine_path": self.engine,
            "available_memory": None,
            "available_disk": 100 * GB,
            "gpus": [],
            "linkage": {"linked": False, "missing": False},
        }
        arguments.update(overrides)
        with mock.patch("resource_plan.memory_available", **probe):
            report = run_doctor(**arguments)
        return report, {check["id"]: check for check in report["checks"]}

    def test_unknown_measurement_is_a_warning_with_the_legacy_fallback(self):
        report, checks = self._doctor({"return_value": None})
        self.assertEqual(checks["memory.ram"]["status"], "warn")
        self.assertEqual(
            checks["memory.ram"]["summary"], "available RAM could not be measured"
        )
        self.assertEqual(checks["memory.ram"]["details"]["available_bytes"], 0)
        self.assertEqual(report["plan"]["tiers"]["ram"]["budget_bytes"], 8 * GB)
        self.assertEqual(report["status"], "warning")

    def test_malformed_cgroup_input_is_a_memory_failure_with_the_reason(self):
        reason = "malformed cgroup memory limit: /sys/fs/cgroup/memory.max"
        report, checks = self._doctor({"side_effect": CgroupFormatError(reason)})
        self.assertEqual(checks["memory.ram"]["status"], "fail")
        self.assertEqual(checks["memory.ram"]["summary"], reason)
        self.assertEqual(checks["model.shards"]["status"], "skip")
        self.assertEqual(checks["placement.plan"]["status"], "skip")
        self.assertIsNone(report["plan"])
        self.assertEqual(report["status"], "error")

    def test_finite_headroom_is_the_ceiling_the_budget_is_checked_against(self):
        report, checks = self._doctor({"return_value": 32 * GB})
        self.assertEqual(checks["memory.ram"]["status"], "pass")
        self.assertEqual(checks["memory.ram"]["details"]["available_bytes"], 32 * GB)
        self.assertEqual(
            report["plan"]["tiers"]["ram"]["budget_bytes"], int(32 * GB * 0.88)
        )

        # 2 GB of headroom: the 12% reserve is kept (1.76 GB, never the old
        # 8 GB floor) and that budget cannot hold the fixed runtime footprint.
        report, checks = self._doctor({"return_value": 2 * GB})
        self.assertEqual(checks["memory.ram"]["details"]["available_bytes"], 2 * GB)
        self.assertEqual(
            report["plan"]["tiers"]["ram"]["budget_bytes"], int(2 * GB * 0.88)
        )
        self.assertEqual(checks["memory.ram"]["status"], "fail")
        self.assertEqual(
            checks["memory.ram"]["summary"],
            "RAM budget cannot hold one expert slot per sparse layer",
        )

        # An explicit --ram above the finite headroom is the container OOM case.
        report, checks = self._doctor({"return_value": 2 * GB}, ram_gb=16)
        self.assertEqual(checks["memory.ram"]["status"], "fail")
        self.assertEqual(
            checks["memory.ram"]["summary"],
            "planned RAM budget exceeds available memory",
        )

    def test_exhausted_budget_is_a_refusal_not_an_unmeasured_warning(self):
        # current >= limit is an authoritative 0; the planner refuses (through
        # doctor's generic planner-error slot) rather than warning "could not
        # be measured" and launching on the 8 GB fallback.
        report, checks = self._doctor({"return_value": 0})
        self.assertEqual(report["status"], "error")
        self.assertIsNone(report["plan"])
        self.assertIn("memory budget is exhausted", checks["model.shards"]["summary"])
        self.assertEqual(checks["memory.ram"]["status"], "skip")


if __name__ == "__main__":
    unittest.main()
