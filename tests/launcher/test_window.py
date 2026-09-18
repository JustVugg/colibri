from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

from colibri.launcher.domain import (
    Check,
    GpuDevice,
    Installation,
    LauncherError,
    LaunchSpec,
    ModelInfo,
    Preflight,
    ProcessEvent,
)
from colibri.launcher.window import LauncherWindow


APP = QApplication.instance() or QApplication([])


def contrast_ratio(first, second):
    def luminance(color):
        channels = [value / 255 for value in (color.red(), color.green(), color.blue())]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                  for value in channels]
        return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))
    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    raise AssertionError("condition did not become true")


class FakeSupervisor:
    instances: list["FakeSupervisor"] = []

    def __init__(self, callback, poll_interval: float = 0.4):
        self.callback = callback
        self.poll_interval = poll_interval
        self.state = "stopped"
        self.spec = None
        self.pid = None
        self.__class__.instances.append(self)

    def start(self, spec):
        self.spec = spec
        self.state = "starting"
        self.callback(ProcessEvent("state", "starting"))

    def stop(self):
        self.state = "stopped"
        self.callback(ProcessEvent("state", "stopped"))

    def wait(self, timeout: float = 10.0):
        return self.state == "stopped"

    def emit(self, kind: str, text: str) -> None:
        if kind == "state":
            self.state = text
        self.callback(ProcessEvent(kind, text))


class LauncherWindowTests(unittest.TestCase):
    def setUp(self):
        FakeSupervisor.instances.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings_path = self.root / "launcher.json"
        self.installation = Installation(
            root=self.root,
            launcher=self.root / "coli",
            python=Path(os.sys.executable),
            support_dir=self.root / "c",
            version="1.11.0",
            web_root=self.root / "web",
        )
        self.model_dir = self.root / "Models" / "Kestrel 7B"
        self.model_dir.mkdir(parents=True)
        self.model = ModelInfo(
            path=self.model_dir,
            name="Kestrel 7B",
            family="qwen36",
            model_id="kestrel-7b",
            engine=self.root / "c" / "engine",
            default_context=8192,
            max_context=32768,
            default_output=1024,
        )
        self.preflight = Preflight(
            model=self.model,
            checks=(Check("model", "pass", "Model is ready"),),
            gpus=(GpuDevice(0, "NVIDIA Test GPU", 24.0),),
            cuda_available=True,
            cuda_reason="CUDA is verified for this model",
            plan={"backend": "cuda", "max_output": 4096},
        )
        self.spec = LaunchSpec(
            argv=(str(self.installation.python), str(self.installation.launcher), "serve"),
            env={"PATH": "test"},
            cwd=self.root,
            port=8000,
            model_id="kestrel-7b",
            mode="serve",
            backend="cuda",
        )

        self.patches = [
            patch("colibri.launcher.window.find_installation", return_value=self.installation),
            patch("colibri.launcher.window.inspect_model", return_value=self.preflight),
            patch("colibri.launcher.window.build_launch", return_value=self.spec),
            patch("colibri.launcher.window.command_preview", return_value='python coli serve --model "Kestrel 7B"'),
            patch("colibri.launcher.window.Supervisor", FakeSupervisor),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.window: LauncherWindow | None = None

    def tearDown(self):
        if self.window is not None:
            self.window._allow_close = True
            self.window.close()
            self.window._workers.waitForDone(2000)
            APP.processEvents()

    def make_window(self) -> LauncherWindow:
        self.window = LauncherWindow(settings_path=self.settings_path)
        self.window.show()
        APP.processEvents()
        return self.window

    def add_model(self, window: LauncherWindow) -> None:
        with patch.object(QFileDialog, "getExistingDirectory", return_value=str(self.model_dir)):
            QTest.mouseClick(window.add_model_button, Qt.MouseButton.LeftButton)
        wait_until(lambda: window.model_list.count() == 1 and window.start_button.isEnabled())

    def test_empty_state_guides_first_run_without_blocking_the_window(self):
        window = self.make_window()
        self.assertTrue(window.empty_state.isVisible())
        self.assertIn("Add", window.empty_state.text())
        self.assertFalse(window.start_button.isEnabled())
        self.assertEqual(window.status_label.text(), "Idle")
        self.assertGreater(window.add_model_button.accessibleName().strip(), "")

    def test_saved_installation_is_ignored_in_favor_of_local_colibri(self):
        from tests.launcher.test_adapter import make_release
        from colibri.launcher.installation import find_installation as real_find_installation
        from colibri.launcher.settings import save_settings

        local = make_release(self.root / "local")
        previous = make_release(self.root / "previous")
        save_settings({
            "version": 1, "installation": str(previous), "python": os.sys.executable,
            "selected_model": str(self.model_dir), "models": [{
                "path": str(self.model_dir), "name": "My saved model", "options": {
                    "mode": "serve", "compute": "cpu", "gpu_ids": [],
                    "ram_gb": 0, "vram_gb": 0.0, "context": 0,
                    "max_tokens": 0, "port": 8012,
                },
            }],
        }, self.settings_path)
        with (
            patch("colibri.launcher.installation.Path.cwd", return_value=local),
            patch("colibri.launcher.window.find_installation", side_effect=real_find_installation),
        ):
            window = self.make_window()
            wait_until(lambda: window._installation is not None and window.start_button.isEnabled())
            self.assertEqual(window._installation.root, local.resolve())
            self.assertEqual(window.model_list.currentItem().text(), "My saved model")
            self.assertEqual(window._options().port, 8012)
            self.assertEqual(window._options().compute, "cpu")
            persisted = json.loads(self.settings_path.read_text(encoding="utf-8"))
            self.assertEqual(Path(persisted["installation"]), local.resolve())

    def test_application_text_remains_readable_with_a_dark_system_palette(self):
        self.check_readable_theme("light")

    def test_saved_dark_theme_remains_readable_with_a_light_system_palette(self):
        self.settings_path.write_text(json.dumps({
            "version": 1, "installation": "", "python": "", "models": [],
            "selected_model": "", "theme": "dark",
        }), encoding="utf-8")
        self.check_readable_theme("dark")

    def check_readable_theme(self, theme):
        from colibri.launcher.app import main

        original_palette = APP.palette()
        original_scheme = APP.styleHints().colorScheme()
        self.addCleanup(APP.setPalette, original_palette)
        self.addCleanup(APP.styleHints().setColorScheme, original_scheme)
        APP.styleHints().setColorScheme(Qt.ColorScheme.Dark if theme == "light" else Qt.ColorScheme.Light)
        dark = QPalette()
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
            dark.setColor(role, QColor("#202020" if theme == "light" else "#ffffff"))
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
            dark.setColor(role, QColor("#ffffff" if theme == "light" else "#202020"))
        APP.setPalette(dark)

        def readable(widget, foreground, background, group=QPalette.ColorGroup.Active, background_widget=None):
            widget.ensurePolished()
            palette = widget.palette()
            surface = (background_widget or widget).palette()
            ratio = contrast_ratio(palette.color(group, foreground), surface.color(group, background))
            self.assertGreaterEqual(ratio, 4.5, f"{widget.objectName() or type(widget).__name__}: {ratio:.2f}:1")

        def verify_window():
            APP.processEvents()
            window = self.window
            self.assertEqual(window.theme_combo.currentData(), theme)
            self.assertEqual(APP.palette().color(QPalette.ColorRole.Window).lightnessF() < 0.5,
                             theme == "dark")
            for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
                for label in (window.empty_state, window.readiness_label):
                    readable(label, QPalette.ColorRole.WindowText, QPalette.ColorRole.Window, group,
                             background_widget=label.parentWidget())
                readable(window.add_model_button, QPalette.ColorRole.ButtonText, QPalette.ColorRole.Button, group)
            for button in (window.start_button, window.rename_model_button):
                readable(button, QPalette.ColorRole.ButtonText, QPalette.ColorRole.Button,
                         QPalette.ColorGroup.Disabled)
            self.add_model(window)
            window.advanced_toggle.setChecked(True)
            readable(window.model_list, QPalette.ColorRole.Text, QPalette.ColorRole.Window,
                     background_widget=window.model_list.parentWidget())
            for widget in (window.theme_combo, window.theme_combo.view(), window.mode_combo, window.mode_combo.view(),
                           window.context_spin, window.python_edit):
                readable(widget, QPalette.ColorRole.Text, QPalette.ColorRole.Base)
            readable(window.mode_combo.view(), QPalette.ColorRole.HighlightedText, QPalette.ColorRole.Highlight)
            dialog = QInputDialog(window)
            dialog.setLabelText("Model name")
            dialog.ensurePolished()
            readable(dialog, QPalette.ColorRole.WindowText, QPalette.ColorRole.Window)
            dialog.close()
            return 0

        def create_window():
            self.window = LauncherWindow(settings_path=self.settings_path)
            return self.window

        with patch("colibri.launcher.window.LauncherWindow", side_effect=create_window), \
                patch.object(QApplication, "exec", side_effect=verify_window):
            self.assertEqual(main(["coli-launcher"]), 0)

    def test_theme_switch_is_saved_without_changing_model_or_running_process(self):
        window = self.make_window()
        self.add_model(window)
        window._preflight_timer.stop()
        before = json.loads(self.settings_path.read_text(encoding="utf-8"))
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        wait_until(lambda: bool(FakeSupervisor.instances) and FakeSupervisor.instances[0].spec is not None)
        supervisor = FakeSupervisor.instances[0]
        supervisor.emit("state", "ready")
        APP.processEvents()
        self.assertTrue(window.theme_combo.isEnabled())
        window.theme_combo.setCurrentIndex(window.theme_combo.findData("dark"))
        APP.processEvents()
        saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["theme"], "dark")
        self.assertEqual(saved["models"], before["models"])
        self.assertEqual(saved["selected_model"], before["selected_model"])
        self.assertEqual(supervisor.state, "ready")
        self.assertEqual(len(FakeSupervisor.instances), 1)
        self.assertLess(APP.palette().color(QPalette.ColorRole.Window).lightnessF(), 0.5)
        QTest.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
        wait_until(lambda: supervisor.state == "stopped")
        window.close()
        window._workers.waitForDone(2000)
        reopened = self.make_window()
        self.assertEqual(reopened.theme_combo.currentData(), "dark")
        reopened.theme_combo.setCurrentIndex(reopened.theme_combo.findData("light"))
        APP.processEvents()
        self.assertGreater(APP.palette().color(QPalette.ColorRole.Window).lightnessF(), 0.5)
        self.assertEqual(json.loads(self.settings_path.read_text(encoding="utf-8"))["theme"], "light")

    def test_theme_save_failure_keeps_the_current_theme(self):
        window = self.make_window()
        original = APP.palette().color(QPalette.ColorRole.Window)
        with patch("colibri.launcher.window.save_settings", side_effect=LauncherError("Disk is full")), \
                patch.object(QMessageBox, "warning") as warning:
            window.theme_combo.setCurrentIndex(window.theme_combo.findData("dark"))
        self.assertEqual(window.theme_combo.currentData(), "light")
        self.assertEqual(APP.palette().color(QPalette.ColorRole.Window), original)
        warning.assert_called_once()
        self.assertIn("Disk is full", warning.call_args.args[2])

    def test_settings_save_failure_does_not_interrupt_startup_readiness(self):
        window = self.make_window()
        self.add_model(window)
        window.close()
        window._workers.waitForDone(2000)

        with (
            patch("colibri.launcher.window.save_settings", side_effect=LauncherError("Disk is full")),
            patch("sys.excepthook") as unhandled,
            patch.object(QMessageBox, "warning") as warning,
        ):
            reopened = self.make_window()
            wait_until(reopened.start_button.isEnabled)
            self.assertTrue(reopened.settings_warning_label.isVisible())
            self.assertIn("Disk is full", reopened.settings_warning_label.text())
            self.assertIn("Ready", reopened.readiness_label.text())
            unhandled.assert_not_called()
            warning.assert_not_called()

    def test_settings_save_failure_keeps_options_usable_and_recovers(self):
        window = self.make_window()
        self.add_model(window)
        with (
            patch("colibri.launcher.window.save_settings", side_effect=LauncherError("Disk is full")),
            patch("sys.excepthook") as unhandled,
            patch.object(QMessageBox, "warning") as warning,
        ):
            window.port_spin.setValue(8123)
            window.context_spin.setValue(4096)
            unhandled.assert_not_called()
            wait_until(window.start_button.isEnabled)
            self.assertTrue(window.settings_warning_label.isVisible())
            self.assertIn("Disk is full", window.settings_warning_label.text())
            self.assertEqual(window._preflight_key[1].port, 8123)
            self.assertEqual(window._preflight_key[1].context, 4096)
            unhandled.assert_not_called()
            warning.assert_not_called()

        window.port_spin.setValue(8124)
        wait_until(window.start_button.isEnabled)
        self.assertFalse(window.settings_warning_label.isVisible())
        saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["models"][0]["options"]["port"], 8124)
        self.assertEqual(saved["models"][0]["options"]["context"], 4096)

    def test_add_rename_and_remove_change_library_only(self):
        window = self.make_window()
        self.add_model(window)
        self.assertEqual(window.model_list.currentItem().text(), "Kestrel 7B")

        with patch.object(QInputDialog, "getText", return_value=("Weekend model", True)):
            QTest.mouseClick(window.rename_model_button, Qt.MouseButton.LeftButton)
        self.assertEqual(window.model_list.currentItem().text(), "Weekend model")

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            QTest.mouseClick(window.remove_model_button, Qt.MouseButton.LeftButton)
        self.assertEqual(window.model_list.count(), 0)
        self.assertTrue(self.model_dir.exists(), "removing a library entry must not delete model files")
        self.assertTrue(window.empty_state.isVisible())

    def test_each_model_persists_its_advanced_settings(self):
        window = self.make_window()
        self.add_model(window)
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("serve"))
        window.compute_combo.setCurrentIndex(window.compute_combo.findData("cpu"))
        window.context_spin.setValue(4096)
        window.port_spin.setValue(8123)
        wait_until(lambda: json.loads(self.settings_path.read_text(encoding="utf-8"))["models"][0]["options"]["port"] == 8123)
        window._allow_close = True
        window.close()
        self.window = None

        reopened = self.make_window()
        wait_until(lambda: reopened.model_list.count() == 1 and reopened.start_button.isEnabled())
        self.assertEqual(reopened.mode_combo.currentData(), "serve")
        self.assertEqual(reopened.compute_combo.currentData(), "cpu")
        self.assertEqual(reopened.context_spin.value(), 4096)
        self.assertEqual(reopened.port_spin.value(), 8123)

    def test_saved_gpu_group_and_missing_ids_roundtrip_without_loss(self):
        saved_options = {
            "mode": "serve",
            "compute": "cuda",
            "gpu_ids": [0, 9],
            "ram_gb": 0,
            "vram_gb": 0.0,
            "context": 0,
            "max_tokens": 0,
            "port": 8000,
        }
        self.settings_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "installation": str(self.root),
                    "python": str(self.installation.python),
                    "models": [{"path": str(self.model_dir), "name": "Saved group", "options": saved_options}],
                    "selected_model": str(self.model_dir),
                }
            ),
            encoding="utf-8",
        )
        detected = Preflight(
            model=self.model,
            checks=(Check("cuda", "fail", "GPU 9 was not detected"),),
            gpus=(GpuDevice(0, "NVIDIA First", 8.0), GpuDevice(1, "NVIDIA Second", 16.0)),
            cuda_available=False,
            cuda_reason="One or more requested NVIDIA GPUs were not detected.",
            plan={"backend": "cuda", "max_output": 4096},
        )

        with patch("colibri.launcher.window.inspect_model", return_value=detected):
            window = self.make_window()
            wait_until(lambda: "not detected" in window.readiness_label.text())

        self.assertEqual(window.gpu_combo.currentData(), (0, 9))
        self.assertIn("not detected", window.gpu_combo.currentText())
        self.assertEqual(window._options().gpu_ids, (0, 9))
        window.port_spin.setValue(8124)
        persisted = json.loads(self.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["models"][0]["options"]["gpu_ids"], [0, 9])

    def test_gpu_zero_selection_reaches_real_kimi_adapter(self):
        from tests.launcher.test_adapter import make_model, make_release
        from colibri.launcher.backend import inspect_model as real_inspect_model
        from colibri.launcher.installation import find_installation as real_find_installation

        release = make_release(self.root / "kimi-install")
        kimi_model = make_model(self.root, "Kimi fixture", model_type="kimi")
        installation = real_find_installation(release, Path(os.sys.executable))
        devices = json.dumps(
            [
                {"index": 0, "name": "NVIDIA First", "total_bytes": 8 << 30},
                {"index": 1, "name": "NVIDIA Second", "total_bytes": 16 << 30},
            ]
        )
        environment = {"FAKE_GPU_DATA": devices}
        if os.sys.platform == "linux":
            # The simulated engine is not an ELF binary. Supply its documented
            # CUDA linkage so this UI-to-adapter boundary test is portable.
            tools = self.root / "tools"
            tools.mkdir()
            ldd = tools / "ldd"
            ldd.write_text(
                '#!/bin/sh\nif [ -f "$1.cuda" ]; then\n'
                '  echo "libcudart.so => /fixture/libcudart.so (0x1)"\nfi\n',
                encoding="utf-8",
            )
            ldd.chmod(0o755)
            environment["PATH"] = str(tools) + os.pathsep + os.environ["PATH"]
        with (
            patch.dict(os.environ, environment),
            patch("colibri.launcher.window.find_installation", return_value=installation),
            patch("colibri.launcher.window.inspect_model", side_effect=real_inspect_model),
        ):
            window = self.make_window()
            wait_until(lambda: window._installation == installation)
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(kimi_model)):
                QTest.mouseClick(window.add_model_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: window.gpu_combo.count() > 1)
            gpu_zero_index = next(
                index for index in range(window.gpu_combo.count()) if window.gpu_combo.itemData(index) == (0,)
            )
            self.assertIn("NVIDIA First", window.gpu_combo.itemText(gpu_zero_index))
            self.assertEqual(window._preflight.plan["backend"], "cpu")

            window.gpu_combo.setCurrentIndex(gpu_zero_index)
            wait_until(lambda: window.start_button.isEnabled() and window._preflight.plan["backend"] == "cuda")

        self.assertEqual(window._options().gpu_ids, (0,))
        self.assertTrue(window._preflight.cuda_available)

    def test_unavailable_cuda_is_disabled_with_a_specific_reason(self):
        no_cuda = Preflight(
            model=self.model,
            checks=(Check("model", "pass", "Model is ready"),),
            gpus=(),
            cuda_available=False,
            cuda_reason="This family has no verified CUDA engine",
            plan={"backend": "cpu"},
        )
        with patch("colibri.launcher.window.inspect_model", return_value=no_cuda):
            window = self.make_window()
            self.add_model(window)
            cuda_index = window.compute_combo.findData("cuda")
            self.assertFalse(window.compute_combo.model().item(cuda_index).isEnabled())
            self.assertIn("no verified CUDA", window.compute_reason.text())

    def test_detected_gpus_remain_visible_when_engine_cannot_use_cuda(self):
        from dataclasses import replace

        result = replace(self.preflight, cuda_available=False,
                         cuda_reason="The selected engine has no CUDA support.",
                         plan={"backend": "cpu", "cuda_capable": False})
        with patch("colibri.launcher.window.inspect_model", return_value=result):
            window = self.make_window()
            self.add_model(window)
            self.assertIn("NVIDIA Test GPU", window.compute_reason.text())
            self.assertIn("no CUDA support", window.compute_reason.text())
            self.assertFalse(window.compute_combo.model().item(
                window.compute_combo.findData("cuda")).isEnabled())

    def test_cpu_choice_allows_switching_back_to_cuda_with_fresh_checks(self):
        from tests.launcher.test_adapter import make_model, make_release
        from colibri.launcher.backend import inspect_model as real_inspect_model
        from colibri.launcher.installation import find_installation as real_find_installation

        release = make_release(self.root / "cuda-install")
        model = make_model(self.root, "CUDA fixture")
        installation = real_find_installation(release, Path(os.sys.executable))
        environment = {"FAKE_GPU_DATA": json.dumps([
            {"index": 0, "name": "NVIDIA First", "total_bytes": 8 << 30},
        ])}
        if os.sys.platform == "linux":
            tools = self.root / "tools"
            tools.mkdir()
            ldd = tools / "ldd"
            ldd.write_text('#!/bin/sh\nif [ -f "$1.cuda" ]; then\n'
                           '  echo "libcudart.so => /fixture/libcudart.so (0x1)"\nfi\n',
                           encoding="utf-8")
            ldd.chmod(0o755)
            environment["PATH"] = str(tools) + os.pathsep + os.environ["PATH"]
        with (
            patch.dict(os.environ, environment),
            patch("colibri.launcher.window.find_installation", return_value=installation),
            patch("colibri.launcher.window.inspect_model", side_effect=real_inspect_model),
        ):
            window = self.make_window()
            wait_until(lambda: window._installation == installation)
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(model)):
                QTest.mouseClick(window.add_model_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: window.start_button.isEnabled())
            window.compute_combo.setCurrentIndex(window.compute_combo.findData("cpu"))
            wait_until(lambda: window.start_button.isEnabled() and
                       window._preflight.plan["backend"] == "cpu")
            cuda_index = window.compute_combo.findData("cuda")
            self.assertTrue(window.compute_combo.model().item(cuda_index).isEnabled())
            self.assertFalse(window._preflight.cuda_available)
            window.compute_combo.setCurrentIndex(cuda_index)
            wait_until(lambda: window.start_button.isEnabled() and
                       window._preflight.cuda_available)
            self.assertEqual(window._preflight.plan["backend"], "cuda")
            self.assertEqual(window._options().compute, "cuda")

    def test_start_stop_and_ready_states_follow_the_supervisor(self):
        window = self.make_window()
        self.add_model(window)
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("serve"))
        wait_until(window.start_button.isEnabled)
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        wait_until(lambda: FakeSupervisor.instances and window.status_label.text() == "Loading")
        runner = FakeSupervisor.instances[-1]
        wait_until(lambda: runner.state == "starting")
        self.assertFalse(window.start_button.isEnabled())
        self.assertTrue(window.stop_button.isEnabled())

        runner.emit("log", "CUDA unavailable; falling back to CPU")
        wait_until(lambda: "may have fallen back" in window.readiness_label.text())
        runner.emit("state", "running")
        wait_until(lambda: window.status_label.text() == "Ready")
        self.assertIn("127.0.0.1:8000", window.api_address.text())
        self.assertIn("fell back", window.readiness_label.text())
        self.assertNotIn("CUDA requested", window.readiness_label.text())

        QTest.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
        wait_until(lambda: window.status_label.text() == "Stopped")
        self.assertFalse(window.stop_button.isEnabled())
        self.assertTrue(window.start_button.isEnabled())

    def test_explicit_cuda_runtime_fallback_remains_a_backend_mismatch(self):
        window = self.make_window()
        self.add_model(window)
        window.compute_combo.setCurrentIndex(window.compute_combo.findData("cuda"))
        wait_until(window.start_button.isEnabled)
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        wait_until(lambda: FakeSupervisor.instances and FakeSupervisor.instances[-1].state == "starting")
        runner = FakeSupervisor.instances[-1]

        runner.emit("log", "[K3-CUDA] device unavailable -- experts stay on CPU")
        runner.emit("state", "running")
        wait_until(lambda: window.status_label.text() == "Backend mismatch")

        self.assertIn("explicitly requested", window.readiness_label.text())
        self.assertIn("CPU fallback", window.readiness_label.text())

    def test_failed_preflight_can_retry_and_recovers(self):
        failed = Preflight(
            model=self.model,
            checks=(Check("engine", "fail", "The selected engine is missing"),),
            gpus=(),
            cuda_available=False,
            cuda_reason="CUDA could not be checked",
            plan={},
        )
        probe_fails = {"value": True}

        def inspect(*_args):
            return failed if probe_fails["value"] else self.preflight

        with patch("colibri.launcher.window.inspect_model", side_effect=inspect):
            window = self.make_window()
            wait_until(lambda: window._installation is not None)
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(self.model_dir)):
                QTest.mouseClick(window.add_model_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: window.retry_button.isVisible())
            self.assertFalse(window.start_button.isEnabled())
            self.assertIn("engine is missing", window.readiness_label.text())
            probe_fails["value"] = False
            QTest.mouseClick(window.retry_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: window.start_button.isEnabled())
            self.assertFalse(window.retry_button.isVisible())

    def test_changed_python_discovery_blocks_old_launch_and_stale_preflight(self):
        window = self.make_window()
        self.add_model(window)
        old_installation = window._installation
        replacement = Installation(
            root=old_installation.root,
            launcher=old_installation.launcher,
            python=self.root / "another-python",
            support_dir=old_installation.support_dir,
            version=old_installation.version,
            web_root=old_installation.web_root,
        )
        old_probe_entered = threading.Event()
        release_old_probe = threading.Event()
        discovery_entered = threading.Event()
        release_discovery = threading.Event()

        def inspect(installation, _path, _options):
            if installation == old_installation:
                old_probe_entered.set()
                release_old_probe.wait(2)
            return self.preflight

        def discover(_folder, _python):
            discovery_entered.set()
            release_discovery.wait(2)
            return replacement

        with (
            patch("colibri.launcher.window.inspect_model", side_effect=inspect),
            patch("colibri.launcher.window.find_installation", side_effect=discover),
        ):
            window._begin_preflight()
            wait_until(old_probe_entered.is_set)
            window.settings_panel.show()
            window.python_edit.setFocus()
            window.python_edit.selectAll()
            QTest.keyClicks(window.python_edit, str(replacement.python))
            release_old_probe.set()
            window._workers.waitForDone(2000)
            APP.processEvents()
            self.assertIsNone(window._installation)
            self.assertFalse(window.start_button.isEnabled())
            self.assertEqual(window.python_edit.text(), str(replacement.python))

            QTest.keyClick(window.python_edit, Qt.Key.Key_Tab)
            wait_until(discovery_entered.is_set)

            self.assertIsNone(window._installation)
            self.assertFalse(window.start_button.isEnabled())
            QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
            self.assertFalse(FakeSupervisor.instances)

            QTest.qWait(100)
            APP.processEvents()
            self.assertFalse(window.start_button.isEnabled())
            self.assertIn("Looking for Colibri", window.readiness_label.text())

            release_discovery.set()
            wait_until(lambda: window._installation == replacement and window.start_button.isEnabled())

    def test_typed_python_change_blocks_start_and_discovers_on_focus_loss(self):
        from dataclasses import replace

        window = self.make_window()
        self.add_model(window)
        replacement = replace(self.installation, python=self.root / "other-python.exe")
        entered = threading.Event()
        release = threading.Event()

        def discover(_folder, python):
            self.assertEqual(python, str(replacement.python))
            entered.set()
            release.wait(2)
            return replacement

        with (
            patch("colibri.launcher.window.find_installation", side_effect=discover) as discovery,
            patch("colibri.launcher.window.build_launch", side_effect=lambda installation, *_args:
                  replace(self.spec, argv=(str(installation.python), "coli", "serve"))),
        ):
            window.settings_panel.show()
            window.python_edit.setFocus()
            window.python_edit.selectAll()
            QTest.keyClicks(window.python_edit, str(replacement.python))
            self.assertIsNone(window._installation)
            self.assertFalse(window.start_button.isEnabled())
            discovery.assert_not_called()

            QTest.keyClick(window.python_edit, Qt.Key.Key_Tab)
            wait_until(entered.is_set)
            self.assertFalse(window.start_button.isEnabled())
            release.set()
            wait_until(window.start_button.isEnabled)
            self.assertEqual(window._installation, replacement)
            discovery.assert_called_once()

            # Writing the discovered path to the field must not start another check.
            window.python_edit.setFocus()
            QTest.keyClick(window.python_edit, Qt.Key.Key_Tab)
            APP.processEvents()
            discovery.assert_called_once()
            QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: FakeSupervisor.instances and FakeSupervisor.instances[-1].spec is not None)
            self.assertEqual(FakeSupervisor.instances[-1].spec.argv[0], str(replacement.python))

    def test_browsing_after_a_python_edit_uses_the_chosen_interpreter(self):
        from dataclasses import replace

        window = self.make_window()
        self.add_model(window)
        replacement = replace(self.installation, python=self.root / "chosen-python.exe")
        with (
            patch("colibri.launcher.window.find_installation", side_effect=lambda _folder, python:
                  replace(self.installation, python=Path(python))),
            patch.object(QFileDialog, "getOpenFileName", return_value=(str(replacement.python), "")),
        ):
            window.settings_panel.show()
            window.python_edit.setFocus()
            window.python_edit.selectAll()
            QTest.keyClicks(window.python_edit, "unfinished-path")
            QTest.mouseClick(window.python_browse_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: window._installation == replacement and window.start_button.isEnabled())
            self.assertEqual(window.python_edit.text(), str(replacement.python))
            self.assertEqual(json.loads(self.settings_path.read_text(encoding="utf-8"))["python"],
                             str(replacement.python))

    def test_failed_python_discovery_never_reuses_old_installation(self):
        from colibri.launcher.domain import LauncherError

        window = self.make_window()
        self.add_model(window)
        discovery_entered = threading.Event()
        release_discovery = threading.Event()

        def fail_discovery(_folder, _python):
            discovery_entered.set()
            release_discovery.wait(2)
            raise LauncherError("The selected Python could not start Colibri")

        with patch("colibri.launcher.window.find_installation", side_effect=fail_discovery):
            window.python_edit.setText(str(self.root / "missing-python"))
            window._discover_installation()
            wait_until(discovery_entered.is_set)
            self.assertIsNone(window._installation)
            self.assertFalse(window.start_button.isEnabled())
            release_discovery.set()
            wait_until(lambda: window.retry_button.isVisible())

        self.assertIsNone(window._installation)
        self.assertFalse(window.start_button.isEnabled())
        self.assertFalse(FakeSupervisor.instances)

    def test_start_rechecks_readiness_instead_of_reusing_the_visible_result(self):
        window = self.make_window()
        self.add_model(window)
        newly_failed = Preflight(
            model=self.model,
            checks=(Check("port", "fail", "The local port is now unavailable"),),
            gpus=self.preflight.gpus,
            cuda_available=True,
            cuda_reason=self.preflight.cuda_reason,
            plan={},
        )
        with patch("colibri.launcher.window.inspect_model", return_value=newly_failed):
            QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: "port is now unavailable" in window.readiness_label.text())
        self.assertFalse(FakeSupervisor.instances)
        self.assertFalse(window.start_button.isEnabled())

    def test_fresh_cuda_failure_preserves_explicit_choice_and_blocks_launch(self):
        window = self.make_window()
        self.add_model(window)
        window.compute_combo.setCurrentIndex(window.compute_combo.findData("cuda"))
        wait_until(window.start_button.isEnabled)
        unavailable = Preflight(
            model=self.model,
            checks=(Check("cuda", "fail", "CUDA driver is unavailable now"),),
            gpus=(),
            cuda_available=False,
            cuda_reason="CUDA driver is unavailable now",
            plan={"backend": "cpu", "max_output": 4096},
        )
        with patch("colibri.launcher.window.inspect_model", return_value=unavailable):
            QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: "CUDA driver is unavailable" in window.readiness_label.text())
            QTest.qWait(180)
            APP.processEvents()

        self.assertEqual(window.compute_combo.currentData(), "cuda")
        self.assertFalse(FakeSupervisor.instances)
        self.assertFalse(window.start_button.isEnabled())
        self.assertTrue(window.compute_combo.isEnabled())
        self.assertTrue(window.model_list.isEnabled())

    def test_stale_launch_build_is_discarded_when_options_change(self):
        window = self.make_window()
        self.add_model(window)
        entered = threading.Event()
        release = threading.Event()
        first = {"value": True}

        def build(_installation, _result, options):
            if first["value"]:
                first["value"] = False
                entered.set()
                release.wait(2)
            return LaunchSpec(
                argv=self.spec.argv,
                env=self.spec.env,
                cwd=self.spec.cwd,
                port=options.port,
                model_id=self.spec.model_id,
                mode=self.spec.mode,
                backend=self.spec.backend,
            )

        with patch("colibri.launcher.window.build_launch", side_effect=build):
            QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
            wait_until(entered.is_set)
            wait_until(lambda: window.status_label.text() == "Preparing")
            self.assertFalse(window.start_button.isEnabled())
            self.assertFalse(window.port_spin.isEnabled())

            window.port_spin.setValue(8123)
            release.set()
            wait_until(window.start_button.isEnabled)
            self.assertFalse(FakeSupervisor.instances, "the completed stale build must not launch")

            QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: FakeSupervisor.instances and FakeSupervisor.instances[-1].spec is not None)
            self.assertEqual(FakeSupervisor.instances[-1].spec.port, 8123)

    def test_model_switch_restores_values_before_applying_each_models_bounds(self):
        window = self.make_window()
        self.add_model(window)
        window.context_spin.setValue(24000)
        window.max_tokens_spin.setValue(3000)
        wait_until(window.start_button.isEnabled)

        second_dir = self.root / "Models" / "Small model"
        second_dir.mkdir()
        second_model = ModelInfo(
            path=second_dir,
            name="Small model",
            family="qwen36",
            model_id="small-model",
            engine=self.model.engine,
            default_context=2048,
            max_context=4096,
            default_output=256,
        )
        second_result = Preflight(
            model=second_model,
            checks=(Check("model", "pass", "Small model is ready"),),
            gpus=(),
            cuda_available=False,
            cuda_reason="CPU is ready",
            plan={"backend": "cpu", "max_output": 512},
        )

        def inspect(_installation, path, _options):
            return second_result if Path(path) == second_dir else self.preflight

        with patch("colibri.launcher.window.inspect_model", side_effect=inspect):
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(second_dir)):
                QTest.mouseClick(window.add_model_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: window.context_spin.maximum() == 4096)
            self.assertEqual(window.max_tokens_spin.maximum(), 512)

            window.model_list.setCurrentRow(0)
            wait_until(lambda: window.context_spin.maximum() == 32768)

        self.assertEqual(window.context_spin.value(), 24000)
        self.assertEqual(window.max_tokens_spin.maximum(), 4096)
        self.assertEqual(window.max_tokens_spin.value(), 3000)

    def test_stop_before_queued_supervisor_start_is_durable(self):
        entered = threading.Event()
        release = threading.Event()

        class DelayedSupervisor(FakeSupervisor):
            def start(self, spec):
                entered.set()
                release.wait(2)
                super().start(spec)

            def stop(self):
                if self.state != "stopped":
                    super().stop()

        with patch("colibri.launcher.window.Supervisor", DelayedSupervisor):
            window = self.make_window()
            self.add_model(window)
            QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
            wait_until(entered.is_set)
            wait_until(window.stop_button.isEnabled)
            QTest.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
            release.set()
            runner = FakeSupervisor.instances[-1]
            wait_until(lambda: runner.state == "stopped" and window.status_label.text() == "Stopped")
            self.assertFalse(window.stop_button.isEnabled())

    def test_stop_and_quit_before_queued_supervisor_start_closes_after_stop(self):
        entered = threading.Event()
        release = threading.Event()

        class DelayedSupervisor(FakeSupervisor):
            def start(self, spec):
                entered.set()
                release.wait(2)
                super().start(spec)

            def stop(self):
                if self.state != "stopped":
                    super().stop()

        with patch("colibri.launcher.window.Supervisor", DelayedSupervisor):
            window = self.make_window()
            self.add_model(window)
            QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
            wait_until(entered.is_set)
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
                window.close()
            release.set()
            runner = FakeSupervisor.instances[-1]
            wait_until(lambda: runner.state == "stopped" and not window.isVisible())
            self.window = None

    def test_late_worker_result_cannot_replace_the_new_selection(self):
        window = self.make_window()
        wait_until(lambda: window._installation is not None)
        second_dir = self.root / "Models" / "Swift 3B"
        second_dir.mkdir()
        second_model = ModelInfo(
            path=second_dir,
            name="Swift 3B",
            family="qwen36",
            model_id="swift-3b",
            engine=self.model.engine,
            default_context=4096,
            max_context=16384,
            default_output=512,
        )
        second_result = Preflight(
            model=second_model,
            checks=(Check("model", "pass", "Second model is ready"),),
            gpus=(),
            cuda_available=False,
            cuda_reason="CPU is ready",
            plan={"backend": "cpu"},
        )
        first_started = threading.Event()
        release_first = threading.Event()

        def inspect(_installation, path, _options):
            if Path(path) == self.model_dir:
                first_started.set()
                release_first.wait(2)
                return self.preflight
            return second_result

        with patch("colibri.launcher.window.inspect_model", side_effect=inspect):
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(self.model_dir)):
                QTest.mouseClick(window.add_model_button, Qt.MouseButton.LeftButton)
            self.assertTrue(first_started.wait(1))
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(second_dir)):
                QTest.mouseClick(window.add_model_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: window.model_list.currentItem().text() == "Swift 3B" and window.start_button.isEnabled())
            release_first.set()
            QTest.qWait(100)
            APP.processEvents()

        self.assertEqual(window.model_path_label.text(), str(second_dir.resolve()))
        self.assertEqual(window.context_spin.maximum(), 16384)

    def test_close_while_loading_can_cancel_or_stop_and_quit(self):
        window = self.make_window()
        self.add_model(window)
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        wait_until(lambda: FakeSupervisor.instances and FakeSupervisor.instances[-1].state == "starting")
        runner = FakeSupervisor.instances[-1]

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel):
            self.assertFalse(window.close())
        self.assertTrue(window.isVisible())
        self.assertEqual(runner.state, "starting")

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            window.close()
        wait_until(lambda: not window.isVisible())
        self.assertEqual(runner.state, "stopped")
        self.window = None


if __name__ == "__main__":
    unittest.main()
