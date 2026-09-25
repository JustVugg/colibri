"""Run after packaging: verify local-folder discovery and external Python.

Usage: python tests/launcher/packaged_smoke.py <executable> <Colibri checkout>
Uses temporary settings and starts no model or inference process.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def verify_runtime_notices(directory):
    """Verify the shipped runtime and its notices, before executing the GUI."""
    forbidden = [path.name for path in directory.rglob("*")
                 if "virtualkeyboard" in path.name.lower()]
    assert not forbidden, f"Unused GPL-only Qt Virtual Keyboard was bundled: {forbidden}"
    for name in ("python/LICENSE.txt", "openssl/LICENSE.txt", "pyinstaller/COPYING.txt", "qt/THIRD_PARTY_NOTICES.txt",
                 "qt/LGPL-3.0-only.txt", "qt/GPL-3.0-only.txt"):
        path = directory / "licenses" / name
        assert path.is_file() and path.stat().st_size > 100, f"Missing runtime notice: {path}"


def verify_control_assets(directory):
    """Catch omitted, corrupt, or empty arrows in the distributed bundle."""
    from PySide6.QtGui import QColor, QImageReader

    def luminance(color):
        values = (channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
                  for channel in (color.redF(), color.greenF(), color.blueF()))
        return sum(channel * weight for channel, weight in zip(values, (0.2126, 0.7152, 0.0722)))

    for suffix, background in (("", "#EEF3EC"), ("-dark", "#29372F")):
        base = luminance(QColor(background))
        for name in ("chevron-down", "chevron-up", "chevron-down-disabled", "chevron-up-disabled"):
            path = directory / f"{name}{suffix}.svg"
            reader = QImageReader(str(path))
            image = reader.read()
            assert not image.isNull(), f"Cannot decode bundled control arrow {path}: {reader.errorString()}"
            visible_ink = 0
            for y in range(image.height()):
                for x in range(image.width()):
                    color = image.pixelColor(x, y)
                    light, dark = sorted((luminance(color), base), reverse=True)
                    visible_ink += color.alpha() >= 128 and (light + 0.05) / (dark + 0.05) >= 3
            assert visible_ink >= 3, f"Bundled control arrow has no visible ink: {path}"


def run_until(binary, working_directory, env, ready):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([str(binary)], cwd=working_directory, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=flags)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None:
            result = ready()
            if result:
                return result
            time.sleep(0.1)
        raise RuntimeError(f"Packaged GUI did not finish the smoke check; exit={process.poll()}")
    finally:
        if process.poll() is None:
            process.terminate()
        _, stderr = process.communicate(timeout=10)
        if stderr:
            print(stderr.decode("utf-8", errors="replace")[-4000:], file=sys.stderr)


def main():
    binary = Path(sys.argv[1]).resolve()
    checkout = Path(sys.argv[2]).resolve()
    verify_runtime_notices(binary.parent / "_internal")
    verify_control_assets(binary.parent / "_internal" / "colibri" / "launcher" / "icons")
    with tempfile.TemporaryDirectory(prefix="colibri-packaged-smoke-") as directory:
        scratch = Path(directory)
        working_directory = scratch / "unrelated-working-directory"
        working_directory.mkdir()
        env = dict(os.environ)
        env.update(QT_QPA_PLATFORM="offscreen", LOCALAPPDATA=str(scratch),
                   XDG_CONFIG_HOME=str(scratch), COLIBRI_HOME=str(checkout))
        if sys.platform == "linux":
            user_libraries = scratch / "user-libraries"
            user_libraries.mkdir()
            env["LD_LIBRARY_PATH"] = str(user_libraries)
            env.pop("LD_LIBRARY_PATH_ORIG", None)
        settings = scratch / ("Colibri" if os.name == "nt" else "colibri") / "launcher.json"

        # Exercise the archive's normal child-folder layout with real upstream
        # support files. Copy the complete bundle so Qt libraries and Linux
        # symlinks retain the same relative layout as the downloaded package.
        local_checkout = scratch / "local-colibri"
        support = local_checkout / "c"
        support.mkdir(parents=True)
        shutil.copy2(checkout / "c" / "coli", support / "coli")
        for source in (checkout / "c").glob("*.py"):
            shutil.copy2(source, support / source.name)
        package = local_checkout / "ColibriLauncher"
        shutil.copytree(binary.parent, package, symlinks=True)
        local_binary = package / binary.name

        # An old, still valid saved location and COLIBRI_HOME must not redirect
        # this copy of the launcher away from the folder containing it.
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({
            "version": 1, "installation": str(checkout), "python": "",
            "models": [], "selected_model": "",
        }), encoding="utf-8")

        def discovered():
            if settings.is_file():
                data = json.loads(settings.read_text(encoding="utf-8"))
                if data.get("installation") and data.get("python"):
                    return data
            return None

        data = run_until(local_binary, working_directory, env, discovered)
        assert not data["models"], "A fresh launcher must not start models"
        assert Path(data["python"]).name != binary.name
        assert Path(data["installation"]).resolve() == local_checkout, data

        # A selected fixture model exercises the actual frozen application's
        # metadata bridge and diagnostic subprocess without loading any weights.
        # Also cover extracting the executable and _internal directly beside
        # coli, while preserving saved models and ignoring the old location.
        install = scratch / "fixture-install"
        shutil.copytree(binary.parent, install, symlinks=True)
        fixture_binary = install / binary.name
        fixtures = Path(__file__).with_name("fixtures")
        for name in ("family_registry.py", "resource_plan.py"):
            shutil.copy2(fixtures / name, install / name)
        fixture = (fixtures / "coli").read_text(encoding="utf-8")
        marker = '''
if __name__ == "_colibri_desktop_probe":
    (HERE / "probe.json").write_text(json.dumps({
        "source": str(Path(sys.modules["__main__"].__file__).resolve()),
        "python": sys.executable,
        "library_path": os.environ.get("LD_LIBRARY_PATH"),
    }), encoding="utf-8")
elif __name__ == "__main__" and sys.argv[1:2] == ["doctor"]:
    (HERE / "doctor-ran").write_text("ok", encoding="utf-8")
elif __name__ == "__main__" and sys.argv[1:2] in (["serve"], ["web"]):
    (HERE / "unexpected-launch").write_text("error", encoding="utf-8")
'''
        fixture = fixture.replace('if __name__ == "__main__":', marker + '\nif __name__ == "__main__":')
        (install / "coli").write_text(fixture, encoding="utf-8")
        (install / "fixture-cpu").write_bytes(b"fixture, never executed")
        model = scratch / "fixture-model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type": "fixture_cpu"}', encoding="utf-8")
        (model / "tokenizer.json").write_text("{}", encoding="utf-8")
        data.update(theme="dark", selected_model=str(model), models=[{
            "name": "Packaged smoke fixture", "path": str(model), "options": {
                "mode": "serve", "compute": "cpu", "gpu_ids": [], "ram_gb": 0,
                "vram_gb": 0.0, "context": 0, "max_tokens": 0, "port": 8000,
            },
        }])
        settings.write_text(json.dumps(data), encoding="utf-8")
        run_until(fixture_binary, working_directory, env,
                  lambda: (install / "doctor-ran").is_file())
        probe = json.loads((install / "probe.json").read_text(encoding="utf-8"))
        expected = install / "_internal" / "colibri" / "launcher" / "probe.py"
        assert Path(probe["source"]).resolve() == expected.resolve(), probe
        assert Path(probe["python"]).resolve() == Path(data["python"]).resolve(), probe
        saved = json.loads(settings.read_text(encoding="utf-8"))
        assert Path(saved["installation"]).resolve() == install, saved
        assert saved["models"] == data["models"], saved
        assert saved["theme"] == "dark", saved
        if sys.platform == "linux":
            assert probe["library_path"] == str(user_libraries), probe
        assert not (install / "unexpected-launch").exists()
        print("Bundled control arrows, packaged local-folder discovery in both layouts, bundled metadata probe, "
              "and external diagnostics passed; no model started.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
