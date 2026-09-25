import json
import tempfile
import unittest
from pathlib import Path


class SettingsTests(unittest.TestCase):
    def test_missing_file_returns_empty_version_one_settings(self):
        from colibri.launcher.settings import load_settings

        with tempfile.TemporaryDirectory() as tmp:
            data, warning = load_settings(Path(tmp) / "missing.json")

        self.assertEqual(data, {"version": 1, "installation": "", "python": "",
                                "models": [], "selected_model": ""})
        self.assertIsNone(warning)

    def test_roundtrip_preserves_valid_model_options(self):
        from colibri.launcher.settings import load_settings, save_settings

        data = {
            "version": 1,
            "installation": "C:/Colibri",
            "python": "C:/Python/python.exe",
            "models": [{
                "path": "D:/Models/model",
                "name": "My model",
                "options": {"mode": "serve", "compute": "cpu", "gpu_ids": [],
                            "ram_gb": 12, "vram_gb": 0.0, "context": 4096,
                            "max_tokens": 512, "port": 8010},
            }],
            "selected_model": "D:/Models/model",
        }
        for theme in (None, "light", "dark"):
            with self.subTest(theme=theme), tempfile.TemporaryDirectory() as tmp:
                document = dict(data)
                if theme is not None:
                    document["theme"] = theme
                path = Path(tmp) / "settings.json"
                save_settings(document, path)
                loaded, warning = load_settings(path)

                self.assertEqual(loaded, document)
                self.assertIsNone(warning)

    def test_invalid_theme_is_rejected_without_replacing_saved_settings(self):
        from colibri.launcher.domain import LauncherError
        from colibri.launcher.settings import save_settings

        data = {"version": 1, "installation": "", "python": "",
                "models": [], "selected_model": "", "theme": "dark"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_settings(data, path)
            original = path.read_bytes()
            for theme in (None, True, 1, [], {}, "system", "DARK", ""):
                with self.subTest(theme=theme):
                    with self.assertRaisesRegex(LauncherError, "theme"):
                        save_settings(dict(data, theme=theme), path)
                    self.assertEqual(path.read_bytes(), original)

    def test_optional_theme_does_not_allow_unrelated_settings_fields(self):
        from colibri.launcher.domain import LauncherError
        from colibri.launcher.settings import save_settings

        data = {"version": 1, "installation": "", "python": "",
                "models": [], "selected_model": "", "theme": "light",
                "unrecognized": "value"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            with self.assertRaisesRegex(LauncherError, "settings"):
                save_settings(data, path)
            self.assertFalse(path.exists())

    def test_unhashable_theme_is_recovered_as_corrupt_settings(self):
        from colibri.launcher.settings import load_settings

        bad = {"version": 1, "installation": "", "python": "",
               "models": [], "selected_model": "", "theme": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            original = json.dumps(bad)
            path.write_text(original, encoding="utf-8")
            loaded, warning = load_settings(path)

            self.assertEqual(loaded.get("theme", "light"), "light")
            self.assertIn("preserved", warning)
            corrupt = list(Path(tmp).glob("settings.json.corrupt-*"))
            self.assertEqual(len(corrupt), 1)
            self.assertEqual(corrupt[0].read_text(encoding="utf-8"), original)

    def test_malformed_file_is_preserved_and_reported(self):
        from colibri.launcher.settings import load_settings

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text("{not json", encoding="utf-8")
            loaded, warning = load_settings(path)
            corrupt = list(Path(tmp).glob("settings.json.corrupt-*"))

            self.assertEqual(loaded["models"], [])
            self.assertIn("preserved", warning)
            self.assertEqual(len(corrupt), 1)
            self.assertEqual(corrupt[0].read_text(encoding="utf-8"), "{not json")
            self.assertFalse(path.exists())

    def test_invalid_types_and_bounds_are_rejected_before_save(self):
        from colibri.launcher.domain import LauncherError
        from colibri.launcher.settings import save_settings

        bad = {"version": 1, "installation": "", "python": "", "models": [{
            "path": "model", "name": "bad", "options": {
                "mode": "web", "compute": "auto", "gpu_ids": [], "ram_gb": True,
                "vram_gb": 0.0, "context": 0, "max_tokens": 0, "port": 70000,
            }}], "selected_model": ""}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(LauncherError, "settings"):
                save_settings(bad, Path(tmp) / "settings.json")

    def test_unhashable_option_value_is_recovered_as_corrupt_settings(self):
        from colibri.launcher.settings import load_settings

        bad = {"version": 1, "installation": "", "python": "", "models": [{
            "path": "model", "name": "bad", "options": {
                "mode": [], "compute": "auto", "gpu_ids": [], "ram_gb": 0,
                "vram_gb": 0.0, "context": 0, "max_tokens": 0, "port": 8000,
            }}], "selected_model": "model"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            loaded, warning = load_settings(path)

            self.assertEqual(loaded["models"], [])
            self.assertIn("preserved", warning)

    def test_parent_creation_failure_is_reported_as_launcher_error(self):
        from colibri.launcher.domain import LauncherError
        from colibri.launcher.settings import save_settings

        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-directory"
            blocker.write_text("file", encoding="utf-8")
            with self.assertRaisesRegex(LauncherError, "could not be saved"):
                save_settings({"version": 1, "installation": "", "python": "",
                               "models": [], "selected_model": ""},
                              blocker / "settings.json")


if __name__ == "__main__":
    unittest.main()
