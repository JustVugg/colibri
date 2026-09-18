import json
import os
import shutil
import shlex
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch


PYTHON = Path(sys.executable)
FIXTURES = Path(__file__).with_name("fixtures")


def make_release(base, *, web=True, cuda=True, legacy_registry=False):
    root = Path(base) / "release"
    root.mkdir(parents=True)
    shutil.copy2(FIXTURES / "coli", root / "coli")
    shutil.copy2(FIXTURES / "resource_plan.py", root / "resource_plan.py")
    registry = "family_registry_legacy.py" if legacy_registry else "family_registry.py"
    shutil.copy2(FIXTURES / registry, root / "family_registry.py")
    for name in ("fixture-cuda", "fixture-cpu", "fixture-v4"):
        (root / name).write_bytes(b"[CUDA] mode: routed experts\0coli_cuda.dll" if name == "fixture-cuda"
                                 else b"[DSV4 CUDA]" if name == "fixture-v4" else b"engine")
    if cuda:
        (root / "fixture-cuda.cuda").write_text("ready", encoding="utf-8")
        (root / "fixture-v4.dsv4").write_text("ready", encoding="utf-8")
        (root / "coli_cuda.dll").write_bytes(b"fixture backend")
        (root / "coli_cuda_dsv4.dll").write_bytes(b"fixture backend")
    if web:
        (root / "web" / "dist").mkdir(parents=True)
        (root / "web" / "dist" / "index.html").write_text("ok", encoding="utf-8")
    return root


def make_model(root, name="model ; $ unicode 鸟", **config):
    model = Path(root) / name
    model.mkdir()
    payload = {"model_type": "fixture", "display_name": "Fixture Model", **config}
    (model / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


class DomainContractTests(unittest.TestCase):
    def test_preflight_can_start_only_without_failed_checks(self):
        from colibri.launcher.domain import Check, ModelInfo, Preflight

        model = ModelInfo(
            path=Path("model"),
            name="Model",
            family="fixture",
            model_id="fixture-model",
            engine=Path("engine"),
            default_context=4096,
            max_context=8192,
            default_output=1024,
        )
        passing = Preflight(model, (Check("ok", "pass", "ready"),), (), False,
                            "CPU selected", {})
        failing = Preflight(model, (Check("bad", "fail", "broken"),), (), False,
                            "CPU selected", {})

        self.assertTrue(passing.can_start)
        self.assertFalse(failing.can_start)


class InstallationTests(unittest.TestCase):
    def test_release_source_c_and_installed_layouts_are_discovered(self):
        from colibri.launcher.installation import find_installation

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            release = make_release(base / "one")

            source = base / "source"
            source.mkdir()
            shutil.copytree(release, source / "c")
            (source / "web" / "dist").mkdir(parents=True)
            (source / "web" / "dist" / "index.html").write_text("ok")

            prefix = base / "prefix"
            (prefix / "bin").mkdir(parents=True)
            shutil.copy2(FIXTURES / "coli", prefix / "bin" / "coli")
            shutil.copytree(release, prefix / "libexec" / "colibri")

            release_install = find_installation(release, PYTHON)
            source_install = find_installation(source, PYTHON)
            c_install = find_installation(source / "c", PYTHON)
            bin_install = find_installation(prefix / "bin" / "coli", PYTHON)

            self.assertEqual(release_install.support_dir, release.resolve())
            self.assertEqual(source_install.support_dir, (source / "c").resolve())
            self.assertEqual(c_install.root, source.resolve())
            self.assertEqual(bin_install.support_dir,
                             (prefix / "libexec" / "colibri").resolve())
            self.assertEqual(bin_install.version, "9.9.9")

    def test_invalid_installation_and_python_have_actionable_errors(self):
        from colibri.launcher.domain import LauncherError
        from colibri.launcher.installation import find_installation

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(LauncherError, "Colibri launcher"):
                find_installation(tmp, PYTHON)
            root = make_release(Path(tmp) / "valid")
            with self.assertRaisesRegex(LauncherError, "Python"):
                find_installation(root, Path(tmp) / "missing-python")

    def test_external_environment_removes_embedded_python_contamination(self):
        from colibri.launcher.installation import external_environment

        env = external_environment({
            "PATH": "tools", "PYTHONHOME": "embedded", "PYTHONPATH": "bundle",
            "_MEIPASS2": "temp", "LD_LIBRARY_PATH": "bundle-libs",
            "LD_LIBRARY_PATH_ORIG": "system-libs", "CUSTOM": "kept",
        })

        self.assertEqual(env["PATH"], "tools")
        self.assertEqual(env["CUSTOM"], "kept")
        self.assertEqual(env["LD_LIBRARY_PATH"], "system-libs")
        for key in ("PYTHONHOME", "PYTHONPATH", "_MEIPASS2"):
            self.assertNotIn(key, env)

    def test_frozen_environment_cleanup_is_idempotent_with_and_without_original_path(self):
        from colibri.launcher.installation import external_environment

        with patch.object(sys, "frozen", True, create=True):
            for original in (None, "", "/opt/colibri-user-libs"):
                with self.subTest(original=original):
                    ambient = {"LD_LIBRARY_PATH": "/bundle/_internal", "PYTHONHOME": "/bundle"}
                    if original is not None:
                        ambient["LD_LIBRARY_PATH_ORIG"] = original
                    first = external_environment(ambient)
                    second = external_environment(first)
                    self.assertEqual(second, first)
                    self.assertEqual(second.get("LD_LIBRARY_PATH"), original or None)
                    self.assertNotIn("PYTHONHOME", second)

    def test_frozen_diagnostic_boundary_retains_original_library_path_in_real_child(self):
        from colibri.launcher.backend import _controlled_environment
        from colibri.launcher.installation import run_external

        ambient = dict(os.environ, LD_LIBRARY_PATH="/bundle/_internal",
                       LD_LIBRARY_PATH_ORIG="/opt/colibri-user-libs")
        with patch.object(sys, "frozen", True, create=True):
            child = run_external([PYTHON, "-c", "import os; print(os.environ.get('LD_LIBRARY_PATH'))"],
                                 environ=_controlled_environment(ambient))
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stdout.strip(), "/opt/colibri-user-libs")

    def test_frozen_build_launch_to_supervisor_retains_original_library_path(self):
        from colibri.launcher.backend import build_launch
        from colibri.launcher.domain import Installation, LaunchOptions, ModelInfo, Preflight
        from colibri.launcher.supervisor import Supervisor

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            installation = Installation(folder, folder / "coli", PYTHON, folder)
            model = ModelInfo(folder, "Fixture", "fixture", "fixture-model", folder / "engine", 4096, 8192, 2048)
            preflight = Preflight(model, (), (), False, "CPU", {"backend": "cpu"})
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            ambient = dict(os.environ, LD_LIBRARY_PATH="/bundle/_internal",
                           LD_LIBRARY_PATH_ORIG="/opt/colibri-user-libs")
            events = []
            runner = Supervisor(events.append, poll_interval=0.03)
            with patch.object(sys, "frozen", True, create=True):
                spec = build_launch(installation, preflight, LaunchOptions(mode="serve", compute="cpu", port=port), ambient)
                # Replace inference with a real tiny child while preserving the
                # adapter-built environment and the supervisor launch boundary.
                spec = replace(spec, argv=(str(PYTHON), "-u", "-c",
                               "import os; print('library=' + str(os.environ.get('LD_LIBRARY_PATH')), flush=True)"))
                try:
                    runner.start(spec)
                    self.assertTrue(runner.wait(6))
                finally:
                    runner.stop()
                    runner.wait(6)
            self.assertIn("library=/opt/colibri-user-libs", [event.text for event in events if event.kind == "log"])

    def test_frozen_environment_without_original_library_path_clears_bundle_path(self):
        from colibri.launcher.installation import external_environment

        env = external_environment({"PATH": "tools", "_MEIPASS2": "bundle",
                                    "LD_LIBRARY_PATH": "bundle-libs"})

        self.assertNotIn("LD_LIBRARY_PATH", env)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        # Simulated installed engines are not executables. This ldd fixture
        # supplies their documented linkage without a compiler/CUDA install.
        if sys.platform == "linux":
            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            ldd = Path(directory.name, "ldd")
            ldd.write_text('#!/bin/sh\nif [ -f "$1.hip" ]; then\n'
                           '  echo "libamdhip64.so => /fixture/libamdhip64.so (0x1)"\n'
                           'elif [ -f "$1.cuda" ] || [ -f "$1.dsv4" ]; then\n'
                           '  echo "libcudart.so => /fixture/libcudart.so (0x1)"\nfi\n')
            ldd.chmod(0o755)
            environment = patch.dict(os.environ, {"PATH": directory.name + os.pathsep + os.environ["PATH"]})
            environment.start()
            self.addCleanup(environment.stop)

    def test_explicit_cuda_requires_devices_and_positive_accelerator_diagnostic(self):
        from colibri.launcher.backend import build_launch, inspect_model
        from colibri.launcher.domain import LaunchOptions, LauncherError
        from colibri.launcher.installation import find_installation

        device = {"index": 0, "name": "NVIDIA Fixture", "total_bytes": 8 << 30}
        with tempfile.TemporaryDirectory() as tmp:
            root = make_release(Path(tmp) / "install")
            install = find_installation(root, PYTHON)
            options = LaunchOptions(mode="serve", compute="cuda")
            for label, devices, status in (("none", [], None), ("skip", [device], "skip"),
                                           ("warn", [device], "warn"), ("omit", [device], "omit")):
                with self.subTest(label=label), patch.dict(os.environ, {"FAKE_GPU_DATA": json.dumps(devices)}):
                    model = make_model(tmp, label, doctor_accelerator=status)
                    result = inspect_model(install, model, options)
                    self.assertFalse(result.can_start)
                    self.assertFalse(result.cuda_available)
                    with self.assertRaises(LauncherError):
                        build_launch(install, result, options)

    def test_kimi_cuda_enables_its_switch_only_for_supported_device_zero(self):
        from colibri.launcher.backend import build_launch, inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        devices = json.dumps([{"index": index, "name": f"NVIDIA {index}", "total_bytes": 8 << 30}
                              for index in (0, 1)])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FAKE_GPU_DATA": devices}):
            root = make_release(Path(tmp) / "install")
            model = make_model(tmp, model_type="kimi")
            install = find_installation(root, PYTHON)
            options = LaunchOptions(mode="serve", compute="cuda", gpu_ids=(0,))
            result = inspect_model(install, model, options)
            spec = build_launch(install, result, options)
            self.assertTrue(result.cuda_available)
            self.assertEqual(spec.env.get("K3_CUDA"), "1")
            self.assertEqual(result.plan["k3_cuda"], "1")
            self.assertEqual(spec.env.get("K3_VK"), "0")
            for gpu_ids in ((1,), (0, 1), ()):
                blocked = inspect_model(install, model, LaunchOptions(mode="serve", compute="cuda", gpu_ids=gpu_ids))
                self.assertFalse(blocked.can_start)
                self.assertIn("device 0", " ".join(c.message for c in blocked.checks))
            automatic = inspect_model(install, model, LaunchOptions(mode="serve", compute="auto"))
            self.assertEqual(automatic.plan["backend"], "cpu")
            self.assertIn("device 0", automatic.cuda_reason)

    def test_deepseek_v4_uses_dedicated_installed_cuda_probe(self):
        from colibri.launcher.backend import build_launch, inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        devices = json.dumps([{"index": 0, "name": "NVIDIA Fixture", "total_bytes": 8 << 30}])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FAKE_GPU_DATA": devices}):
            root = make_release(Path(tmp) / "install")
            model = make_model(tmp, model_type="deepseek_v4")
            install = find_installation(root, PYTHON)
            options = LaunchOptions(mode="serve", compute="cuda")
            result = inspect_model(install, model, options)
            self.assertTrue(result.can_start)
            self.assertTrue(result.cuda_available)
            self.assertEqual(build_launch(install, result, options).backend, "cuda")
            (root / "fixture-v4.dsv4").unlink()
            self.assertFalse(inspect_model(install, model, options).can_start)

    def _native_family_preflight(self, installation, model, family, options):
        from colibri.launcher.backend import _probe_metadata, inspect_model

        # The installed fixture supplies capabilities/diagnostics without model
        # weights. Only its family identifier changes for the Inkling contract.
        metadata = _probe_metadata(installation, model)
        metadata["family"] = family
        with patch("colibri.launcher.backend._probe_metadata", return_value=metadata):
            return inspect_model(installation, model, options)

    def _upstream_engine_controls(self, spec, family):
        # Exercise the actual CLI environment builder after our subprocess
        # boundary. Capability gates alone are supplied: no engine/model is
        # loaded, and auto-tier needs real weights so is tested elsewhere.
        code = '''import json,runpy,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
namespace = runpy.run_path(str(Path(sys.argv[1]) / 'coli'), run_name='_test_family_launch')
build = namespace['env_for_engine']
build.__globals__.update(dsv4_cuda_available=lambda model: True,
                        cuda_binary=lambda engine=None: True,
                        engine_for_gpu_check=lambda a: 'selected-engine')
args = SimpleNamespace(model='selected-model', ngen=None, temp=None, ram=0,
                       ctx=0, gpu=sys.argv[3], vram=0, auto_tier=False)
env = build(args, sys.argv[2])
print(json.dumps({name: env.get(name) for name in
    ('DSV4_CUDA', 'DSV4_CUDA_DEVICE', 'NOGPU', 'GPU_DEV')}))
'''
        upstream = Path(__file__).resolve().parents[2] / "c"
        gpu = spec.argv[spec.argv.index("--gpu") + 1]
        completed = subprocess.run([str(PYTHON), "-c", code, str(upstream), family, gpu],
                                   env=spec.env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(completed.stdout)

    def test_cpu_and_automatic_fallback_disable_native_single_gpu_engines(self):
        from colibri.launcher.backend import build_launch
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        devices = json.dumps([{"index": 0, "name": "NVIDIA Fixture", "total_bytes": 8 << 30}])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FAKE_GPU_DATA": devices}):
            install = find_installation(make_release(Path(tmp) / "install"), PYTHON)
            model = make_model(tmp, model_type="deepseek_v4", gpu_runtime_missing=True)
            for family in ("deepseek_v4", "inkling"):
                for compute in ("cpu", "auto"):
                    with self.subTest(family=family, compute=compute):
                        options = LaunchOptions(mode="serve", compute=compute)
                        result = self._native_family_preflight(install, model, family, options)
                        self.assertTrue(result.can_start)
                        self.assertEqual(result.plan["backend"], "cpu")
                        controls = self._upstream_engine_controls(build_launch(install, result, options), family)
                        # Native V4 enables CUDA unless DSV4_CUDA=0; native
                        # Inkling enables it whenever NOGPU is absent.
                        if family == "deepseek_v4":
                            self.assertEqual(controls["DSV4_CUDA"], "0")
                        else:
                            self.assertIsNotNone(controls["NOGPU"])

    def test_single_gpu_families_pass_selected_device_to_native_engine(self):
        from colibri.launcher.backend import build_launch
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        for indices, requested in (((0, 1), (1,)), ((1,), ())):
            devices = json.dumps([{"index": index, "name": f"NVIDIA {index}", "total_bytes": 8 << 30}
                                  for index in indices])
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FAKE_GPU_DATA": devices}):
                install = find_installation(make_release(Path(tmp) / "install"), PYTHON)
                model = make_model(tmp, model_type="deepseek_v4")
                for family, variable in (("deepseek_v4", "DSV4_CUDA_DEVICE"), ("inkling", "GPU_DEV")):
                    with self.subTest(family=family, requested=requested):
                        options = LaunchOptions(mode="serve", compute="cuda", gpu_ids=requested)
                        result = self._native_family_preflight(install, model, family, options)
                        self.assertTrue(result.cuda_available)
                        ambient = dict(os.environ, DSV4_CUDA="0", DSV4_CUDA_DEVICE="0",
                                       NOGPU="1", GPU_DEV="0")
                        controls = self._upstream_engine_controls(build_launch(install, result, options, ambient), family)
                        self.assertEqual(controls[variable], "1")
                        if family == "deepseek_v4":
                            self.assertEqual(controls["DSV4_CUDA"], "1")
                        else:
                            self.assertIsNone(controls["NOGPU"])

    def test_single_gpu_families_reject_multiple_devices_and_all_on_multi_gpu_host(self):
        from colibri.launcher.backend import build_launch
        from colibri.launcher.domain import LaunchOptions, LauncherError
        from colibri.launcher.installation import find_installation

        devices = json.dumps([{"index": index, "name": f"NVIDIA {index}", "total_bytes": 8 << 30}
                              for index in (0, 1)])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FAKE_GPU_DATA": devices}):
            install = find_installation(make_release(Path(tmp) / "install"), PYTHON)
            model = make_model(tmp, model_type="deepseek_v4")
            for family in ("deepseek_v4", "inkling"):
                for requested in ((0, 1), ()):
                    with self.subTest(family=family, requested=requested):
                        options = LaunchOptions(mode="serve", compute="cuda", gpu_ids=requested)
                        result = self._native_family_preflight(install, model, family, options)
                        self.assertFalse(result.can_start)
                        self.assertIn("one GPU", " ".join(check.message for check in result.checks))
                        with self.assertRaises(LauncherError):
                            build_launch(install, result, options)
                        automatic = replace(options, compute="auto")
                        fallback = self._native_family_preflight(install, model, family, automatic)
                        self.assertEqual(fallback.plan["backend"], "cpu")
                valid = LaunchOptions(mode="serve", compute="cuda", gpu_ids=(1,))
                result = self._native_family_preflight(install, model, family, valid)
                with self.assertRaisesRegex(LauncherError, "one GPU"):
                    build_launch(install, result, replace(valid, gpu_ids=(0, 1)))

    def test_v4_cuda_bridge_matches_real_upstream_capability_probe(self):
        # Exercise c/coli's actual dedicated function, without model weights.
        # Only engine resolution is supplied by the fixture; Windows DLL and
        # Linux linkage checks execute through the real upstream function.
        with tempfile.TemporaryDirectory() as tmp:
            root = make_release(Path(tmp) / "install")
            upstream = Path(__file__).resolve().parents[2] / "c"
            code = '''import runpy,sys
from pathlib import Path
from colibri.launcher.probe import _cuda_support
sys.path.insert(0, sys.argv[1])
namespace = runpy.run_path(str(Path(sys.argv[1]) / 'coli'), run_name='_test_installed_coli')
engine = Path(sys.argv[2])
dedicated = namespace['dsv4_cuda_available']
dedicated.__globals__['engine_for'] = lambda model: str(engine)
assert dedicated('selected-model')
if sys.platform == 'win32':
    assert not namespace['cuda_binary'](str(engine))
supported, reason = _cuda_support(namespace, engine, 'deepseek_v4', 'selected-model')
assert supported, reason
'''
            completed = subprocess.run([str(PYTHON), '-c', code, str(upstream), str(root / 'fixture-v4')],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_hip_engine_is_not_verified_as_nvidia_cuda(self):
        from colibri.launcher.backend import inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        devices = json.dumps([{"index": 0, "name": "NVIDIA Fixture", "total_bytes": 8 << 30}])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FAKE_GPU_DATA": devices}):
            root = make_release(Path(tmp) / "install")
            (root / "fixture-cuda.cuda").unlink()
            (root / "fixture-cuda.hip").write_text("HIP")
            (root / "fixture-cuda").write_bytes(b"[CUDA] mode: routed experts\0coli_hip.dll")
            (root / "coli_hip.dll").write_bytes(b"fixture HIP")
            install = find_installation(root, PYTHON)
            model = make_model(tmp)
            explicit = inspect_model(install, model, LaunchOptions(mode="serve", compute="cuda"))
            self.assertFalse(explicit.can_start)
            self.assertFalse(explicit.cuda_available)
            self.assertIn("HIP", explicit.cuda_reason)
            automatic = inspect_model(install, model, LaunchOptions(mode="serve", compute="auto"))
            self.assertEqual(automatic.plan["backend"], "cpu")

    def test_automatic_resources_ignore_ambient_family_overrides_in_doctor_and_launch(self):
        from colibri.launcher.backend import build_launch, inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        ambient = {"CTX": "123", "GLM53_MAXT": "987654", "K3_MAXT": "123",
                   "RAM_GB": "777", "GLM53_EXPERT_GB": "999", "K3_EXPERT_GB": "999", "CAP": "999"}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ambient):
            root = make_release(Path(tmp) / "install")
            install = find_installation(root, PYTHON)
            for family, context_var in (("fixture", "CTX"), ("glm53", "GLM53_MAXT"), ("kimi", "K3_MAXT")):
                with self.subTest(family=family):
                    model = make_model(tmp, family, model_type=family)
                    options = LaunchOptions(mode="serve", compute="cpu")
                    result = inspect_model(install, model, options)
                    spec = build_launch(install, result, options)
                    self.assertEqual(result.plan["context"], 4096)
                    self.assertEqual(result.plan["ram"], 0)
                    self.assertNotIn(context_var, spec.env)
                    self.assertNotIn("RAM_GB", spec.env)
                    self.assertNotIn("CAP", spec.env)
                    actual = json.loads(subprocess.run(spec.argv, env=spec.env, capture_output=True, text=True, check=True).stdout)
                    self.assertEqual(actual["context"], 4096)
                    self.assertEqual(actual["ram"], 0)
            model = Path(tmp, "glm53")
            options = LaunchOptions(mode="serve", compute="cpu", ram_gb=20)
            result = inspect_model(install, model, options)
            spec = build_launch(install, result, options)
            actual = json.loads(subprocess.run(spec.argv, env=spec.env, capture_output=True, text=True, check=True).stdout)
            self.assertEqual(float(actual["expert_gb"]), 12)

    def test_advertised_default_output_matches_both_supported_modes(self):
        from colibri.launcher.backend import build_launch, inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        with tempfile.TemporaryDirectory() as tmp:
            root = make_release(Path(tmp) / "install")
            model = make_model(tmp)
            install = find_installation(root, PYTHON)
            for mode in ("web", "serve"):
                options = LaunchOptions(mode=mode, compute="cpu")
                result = inspect_model(install, model, options)
                spec = build_launch(install, result, options)
                actual = json.loads(subprocess.run(spec.argv, env=spec.env, capture_output=True, text=True, check=True).stdout)
                self.assertEqual(result.model.default_output, 2048)
                self.assertEqual(actual["output"], result.model.default_output)

    def test_legacy_registry_without_optional_display_helper_is_supported(self):
        from colibri.launcher.backend import inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        with tempfile.TemporaryDirectory() as tmp:
            root = make_release(Path(tmp) / "install", legacy_registry=True)
            model = make_model(tmp)
            result = inspect_model(find_installation(root, PYTHON), model,
                                   LaunchOptions(mode="serve", compute="cpu"))

            self.assertEqual(result.model.name, "Legacy Fixture")
            self.assertTrue(result.can_start)

    def test_unicode_metacharacter_model_path_is_passed_as_one_argument(self):
        from colibri.launcher.backend import build_launch, inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        with tempfile.TemporaryDirectory() as tmp:
            root = make_release(Path(tmp) / "install")
            model = make_model(tmp)
            install = find_installation(root, PYTHON)
            result = inspect_model(install, model, LaunchOptions(mode="serve", compute="cpu"))
            spec = build_launch(install, result, LaunchOptions(mode="serve", compute="cpu"), {})

            self.assertTrue(result.can_start)
            self.assertIn(str(model.resolve()), spec.argv)
            self.assertEqual(spec.argv.count(str(model.resolve())), 1)

    def test_cpu_launch_disables_accelerators_and_builds_explicit_local_web_argv(self):
        from colibri.launcher.backend import build_launch, inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        with tempfile.TemporaryDirectory() as tmp:
            root = make_release(Path(tmp) / "install")
            model = make_model(tmp)
            install = find_installation(root, PYTHON)
            options = LaunchOptions(compute="cpu")
            result = inspect_model(install, model, options)
            spec = build_launch(install, result, options,
                                {"PATH": "tools", "COLI_CUDA": "1", "K3_VK": "1",
                                 "COLI_API_KEY": "secret", "CLUSTER_WORKERS": "remote"})

            self.assertEqual(spec.argv[0], str(install.python))
            self.assertEqual(spec.env["COLI_CUDA"], "0")
            self.assertEqual(spec.env["K3_VK"], "0")
            self.assertEqual(spec.env["PATH"], "tools")
            self.assertNotIn("COLI_API_KEY", spec.env)
            self.assertNotIn("CLUSTER_WORKERS", spec.env)
            self.assertIn("--no-browser", spec.argv)
            self.assertIn("--gpu", spec.argv)
            self.assertEqual(spec.argv[spec.argv.index("--gpu") + 1], "none")
            self.assertEqual(spec.argv[spec.argv.index("--host") + 1], "127.0.0.1")

    def test_explicit_cuda_rejects_cpu_only_family_and_wrong_device(self):
        from colibri.launcher.backend import inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        devices = json.dumps([{"index": 0, "name": "NVIDIA Fixture", "total_bytes": 8 << 30}])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FAKE_GPU_DATA": devices}):
            root = make_release(Path(tmp) / "install")
            install = find_installation(root, PYTHON)
            cpu_model = make_model(tmp, "cpu", model_type="fixture_cpu")
            cuda_model = make_model(tmp, "cuda")

            cpu = inspect_model(install, cpu_model, LaunchOptions(compute="cuda"))
            wrong = inspect_model(install, cuda_model,
                                  LaunchOptions(compute="cuda", gpu_ids=(9,)))

            self.assertFalse(cpu.can_start)
            self.assertIn("does not support", " ".join(c.message for c in cpu.checks))
            self.assertFalse(wrong.can_start)
            self.assertIn("not detected", " ".join(c.message for c in wrong.checks))

    def test_cuda_launch_clears_ambient_device_remapping_and_tuning(self):
        from colibri.launcher.backend import build_launch, inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        devices = json.dumps([
            {"index": 0, "name": "NVIDIA First", "total_bytes": 8 << 30},
            {"index": 1, "name": "NVIDIA Second", "total_bytes": 16 << 30},
        ])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FAKE_GPU_DATA": devices}):
            root = make_release(Path(tmp) / "install")
            model = make_model(tmp)
            installation = find_installation(root, PYTHON)
            options = LaunchOptions(mode="serve", compute="cuda", gpu_ids=(1,), vram_gb=6.0)
            result = inspect_model(installation, model, options)
            spec = build_launch(installation, result, options, {
                "PATH": "tools", "CUDA_VISIBLE_DEVICES": "0", "CUDA_EXPERT_GB": "99",
                "CUDA_DENSE": "1", "COLI_GPUS": "0", "COLI_CUDA_PIPE": "2",
                "K3_VK_UP": "auto",
            })

            self.assertTrue(result.cuda_available)
            self.assertEqual(spec.argv[spec.argv.index("--gpu") + 1], "1")
            self.assertEqual(spec.env["COLI_GPUS"], "1")
            self.assertEqual(spec.env["COLI_CUDA"], "1")
            self.assertEqual(spec.env["PATH"], "tools")
            for name in ("CUDA_VISIBLE_DEVICES", "CUDA_EXPERT_GB", "CUDA_DENSE",
                         "COLI_CUDA_PIPE", "K3_VK_UP"):
                self.assertNotIn(name, spec.env)

    def test_option_bounds_unsupported_schema_and_missing_web_assets_are_blocking(self):
        from colibri.launcher.backend import inspect_model
        from colibri.launcher.domain import LaunchOptions, LauncherError
        from colibri.launcher.installation import find_installation

        with tempfile.TemporaryDirectory() as tmp:
            root = make_release(Path(tmp) / "install", web=False)
            install = find_installation(root, PYTHON)
            model = make_model(tmp, "normal")
            bad_schema = make_model(tmp, "schema", doctor_schema="bad")

            with self.assertRaisesRegex(LauncherError, "context"):
                inspect_model(install, model, LaunchOptions(context=8193))
            with self.assertRaisesRegex(LauncherError, "diagnostic schema"):
                inspect_model(install, bad_schema, LaunchOptions(mode="serve"))
            missing_web = inspect_model(install, model,
                                        LaunchOptions(mode="web", compute="cpu"))
            self.assertFalse(missing_web.can_start)
            self.assertIn("Web Chat assets", " ".join(c.message for c in missing_web.checks))

    def test_automatic_gpu_failure_rechecks_cpu_and_explains_fallback(self):
        from colibri.launcher.backend import inspect_model
        from colibri.launcher.domain import LaunchOptions
        from colibri.launcher.installation import find_installation

        devices = json.dumps([{"index": 0, "name": "NVIDIA Fixture", "total_bytes": 12 << 30}])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FAKE_GPU_DATA": devices}):
            root = make_release(Path(tmp) / "install")
            model = make_model(tmp, gpu_runtime_missing=True)
            result = inspect_model(find_installation(root, PYTHON), model,
                                   LaunchOptions(mode="serve", compute="auto"))

            self.assertTrue(result.can_start)
            self.assertFalse(result.cuda_available)
            self.assertEqual(result.plan["backend"], "cpu")
            self.assertIn("Automatic selected CPU", " ".join(c.message for c in result.checks))

    def test_command_preview_redacts_secrets_and_quotes_arguments(self):
        from colibri.launcher.backend import command_preview
        from colibri.launcher.domain import LaunchSpec

        spec = LaunchSpec(("python", "coli", "serve", "--model", "a b;$x"),
                          {"COLI_API_KEY": "secret"}, Path("."), 8000,
                          "fixture", "serve", "cpu")
        preview = command_preview(spec)

        if os.name == "nt":
            self.assertTrue(preview.startswith("& "))
        else:
            self.assertEqual(shlex.split(preview), list(spec.argv))
        self.assertNotIn("secret", preview)

    @unittest.skipUnless(os.name == "nt", "PowerShell copy/paste command preview")
    def test_windows_preview_roundtrips_metacharacters_without_expansion_or_secrets(self):
        from colibri.launcher.backend import command_preview
        from colibri.launcher.domain import LaunchSpec

        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp, "capture argv.py")
            capture.write_text('import json,os,sys\nprint(json.dumps({"args":sys.argv[1:], '
                               '"secret":os.environ.get("COLIBRI_PREVIEW_SECRET")}))\n', encoding="utf-8")
            args = (str(PYTHON), str(capture), "model $HOME `n ; & 'literal' 鸟", "plain&name")
            spec = LaunchSpec(args, {"COLIBRI_PREVIEW_SECRET": "hidden"}, Path(tmp), 8000,
                              "fixture", "serve", "cpu")
            env = dict(os.environ)
            env.pop("COLIBRI_PREVIEW_SECRET", None)
            completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                                        command_preview(spec)], env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            received = json.loads(completed.stdout)
            self.assertEqual(received["args"], list(args[2:]))
            self.assertIsNone(received["secret"])


if __name__ == "__main__":
    unittest.main()
