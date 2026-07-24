"""Direct contracts for the extracted RAM-disk planning module."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support import planning


class PlanningModuleTest(unittest.TestCase):
    def _services(self, root):
        platform = argparse.Namespace(is_linux=True)
        return {
            "discover_hardware": ramdisk.discover_hardware,
            "scan_model": ramdisk.scan_model,
            "load_profile": ramdisk._load_profile,
            "select_partial": ramdisk._select_partial,
            "runtime_reserve": ramdisk._runtime_reserve,
            "build_placement": ramdisk._build_placement,
            "reusable_empty_mountpoint": ramdisk._reusable_empty_mountpoint,
            "filesystem_for_path": lambda path: "ext4",
            "state_root": lambda: str(root / "state"),
            "manifest_path": lambda: str(root / "state" / "manifest.json"),
            "benchmarks_path": lambda: str(root / "state" / "benchmarks.json"),
            "current_euid": lambda: 0,
            "get_platform_ops": lambda: platform,
        }

    def test_facade_resolves_every_planning_dependency_at_call_time(self):
        args = mock.sentinel.args
        hardware = mock.sentinel.hardware
        model = mock.sentinel.model
        dependencies = {
            "discover_hardware": mock.sentinel.discover_hardware,
            "scan_model": mock.sentinel.scan_model,
            "_load_profile": mock.sentinel.load_profile,
            "_select_partial": mock.sentinel.select_partial,
            "_runtime_reserve": mock.sentinel.runtime_reserve,
            "_build_placement": mock.sentinel.build_placement,
            "_reusable_empty_mountpoint": mock.sentinel.reusable_empty_mountpoint,
            "_filesystem_for_path": mock.sentinel.filesystem_for_path,
            "_state_root": mock.sentinel.state_root,
            "_manifest_path": mock.sentinel.manifest_path,
            "_benchmarks_path": mock.sentinel.benchmarks_path,
            "current_euid": mock.sentinel.current_euid,
            "get_platform_ops": mock.sentinel.get_platform_ops,
        }
        with mock.patch.multiple(ramdisk, **dependencies), mock.patch.object(
            ramdisk,
            "_planning_build_plan",
            return_value=mock.sentinel.plan,
        ) as implementation:
            result = ramdisk.build_plan(
                args,
                hardware=hardware,
                model=model,
            )

        self.assertIs(result, mock.sentinel.plan)
        implementation.assert_called_once_with(
            args,
            hardware=hardware,
            model=model,
            discover_hardware=mock.sentinel.discover_hardware,
            scan_model=mock.sentinel.scan_model,
            load_profile=mock.sentinel.load_profile,
            select_partial=mock.sentinel.select_partial,
            runtime_reserve=mock.sentinel.runtime_reserve,
            build_placement=mock.sentinel.build_placement,
            reusable_empty_mountpoint=mock.sentinel.reusable_empty_mountpoint,
            filesystem_for_path=mock.sentinel.filesystem_for_path,
            state_root=mock.sentinel.state_root,
            manifest_path=mock.sentinel.manifest_path,
            benchmarks_path=mock.sentinel.benchmarks_path,
            current_euid=mock.sentinel.current_euid,
            get_platform_ops=mock.sentinel.get_platform_ops,
        )

    def test_direct_builder_matches_facade_with_the_same_services(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=2)
            model = ramdisk.scan_model(str(fixture.root))
            args = plan_args(fixture.root, memory_nodes="0-1", cpu_list="0-3")
            services = self._services(fixture.root)
            facade_patches = {
                "_load_profile": services["load_profile"],
                "_select_partial": services["select_partial"],
                "_runtime_reserve": services["runtime_reserve"],
                "_build_placement": services["build_placement"],
                "_reusable_empty_mountpoint": services[
                    "reusable_empty_mountpoint"
                ],
                "_filesystem_for_path": services["filesystem_for_path"],
                "_state_root": services["state_root"],
                "_manifest_path": services["manifest_path"],
                "_benchmarks_path": services["benchmarks_path"],
                "current_euid": services["current_euid"],
                "get_platform_ops": services["get_platform_ops"],
            }
            with mock.patch.multiple(ramdisk, **facade_patches):
                expected = ramdisk.build_plan(
                    args,
                    hardware=hardware,
                    model=model,
                )
            actual = planning.build_plan(
                args,
                hardware=hardware,
                model=model,
                **services,
            )

        expected.pop("created_at")
        actual.pop("created_at")
        self.assertEqual(actual, expected)

    def test_injected_discovery_and_model_callbacks_are_resolved_at_call_time(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture()
            model = ramdisk.scan_model(str(fixture.root))
            calls = []
            services = self._services(fixture.root)

            def discover():
                calls.append("discover")
                return hardware

            def scan(path):
                calls.append(("scan", path))
                return model

            services.update(
                discover_hardware=discover,
                scan_model=scan,
            )
            plan = planning.build_plan(
                plan_args(fixture.root),
                **services,
            )

        self.assertEqual(calls, ["discover", ("scan", str(fixture.root))])
        self.assertEqual(plan["model"]["fingerprint"], model["fingerprint"])
        self.assertNotIn("coli ramdisk is supported only on Linux", plan["blockers"])

    def test_unsupported_hardware_returns_a_plan_without_platform_probing(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture()
            hardware["linux"] = False
            hardware["tmpfs"] = {
                "supported": False,
                "noswap_supported": False,
            }
            model = ramdisk.scan_model(str(fixture.root))
            services = self._services(fixture.root)

            def unexpected_platform_probe():
                raise AssertionError(
                    "unsupported planning must not select Linux platform operations"
                )

            def unexpected_filesystem_probe(path):
                raise AssertionError(
                    "unsupported planning must not inspect Linux mount tables: %s"
                    % path
                )

            services.update(
                get_platform_ops=unexpected_platform_probe,
                filesystem_for_path=unexpected_filesystem_probe,
            )
            plan = planning.build_plan(
                plan_args(fixture.root),
                hardware=hardware,
                model=model,
                **services,
            )

        self.assertEqual(plan["schema"], ramdisk.PLAN_SCHEMA)
        self.assertEqual(plan["version"], ramdisk.MANIFEST_VERSION)
        self.assertIn(
            "coli ramdisk is supported only on Linux",
            plan["blockers"],
        )

    def test_importing_planning_does_not_load_host_or_lifecycle_modules(self):
        code = """
import json
import sys
import ramdisk_support.planning

forbidden = [
    "ramdisk_support.benchmark",
    "ramdisk_support.cli",
    "ramdisk_support.discovery",
    "ramdisk_support.lifecycle",
    "ramdisk_support.linux_ops",
    "ramdisk_support.mounts",
    "ramdisk_support.platform_ops",
    "ramdisk_support.presentation",
    "ramdisk_support.processes",
    "ramdisk_support.state",
]
print(json.dumps([name for name in forbidden if name in sys.modules]))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=C_DIR,
            env=dict(os.environ, PYTHONPATH=str(C_DIR)),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
