# Build on the target OS: python -m PyInstaller desktop/launcher.spec
from importlib.metadata import distribution
from pathlib import Path
import os
import runpy
import sys

from PySide6.QtCore import qVersion

root = Path(SPECPATH).parent
if qVersion() != "6.11.2":
    raise RuntimeError("Packaged launcher notices target Qt 6.11.2; install PySide6==6.11.2 to build")
# Prefer Windows' own DLLs to unrelated tools on PATH (for example, Poppler's
# incompatible icuuc.dll). PyInstaller excludes system DLLs from the bundle.
if sys.platform == "win32":
    system_directory = Path(os.environ["SystemRoot"]) / "System32"
    os.environ["PATH"] = str(system_directory) + os.pathsep + os.environ.get("PATH", "")
data = [
    (str(root / "colibri/launcher/probe.py"), "colibri/launcher"),
    (str(root / "colibri/launcher/icons/*.svg"), "colibri/launcher/icons"),
    (str(root / "LICENSE"), "licenses/colibri"),
    (str(root / "NOTICE"), "licenses/colibri"),
    (str(root / "THIRD_PARTY_NOTICES.md"), "licenses/colibri"),
    (str(root / "docs/launcher.md"), "."),
]
for name in ("GPL-3.0-only.txt", "LGPL-3.0-only.txt", "THIRD_PARTY_NOTICES.txt", "SOURCES.json", "README.md"):
    data.append((str(root / "desktop/launcher-licenses" / name), "licenses/qt"))
for name in ("PySide6", "PySide6_Essentials", "shiboken6"):
    package = distribution(name)
    for item in package.files or ():
        if "/licenses/" in str(item).replace("\\", "/"):
            data.append((str(package.locate_file(item)), f"licenses/{name}/{Path(item).parent.name}"))
        elif str(item).endswith(".dist-info/METADATA"):
            data.append((str(package.locate_file(item)), f"licenses/{name}"))

analysis = Analysis(
    [str(root / "desktop/launcher_entry.py")],
    pathex=[str(root)],
    binaries=[],
    datas=data,
    hiddenimports=[],
    hookspath=[str(root / "desktop/launcher-hooks")],
    runtime_hooks=[],
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets"],
    noarchive=False,
)
for name, _, _ in analysis.binaries:
    if "virtualkeyboard" in name.lower():
        raise RuntimeError(f"Unexpected GPL-only Virtual Keyboard runtime: {name}")
collect_notices = runpy.run_path(str(root / "desktop/launcher-licenses/collect.py"))["collect_notices"]
for source, destination in collect_notices(analysis.binaries, workpath):
    analysis.datas.append((f"{destination}/{Path(source).name}", source, "DATA"))
archive = PYZ(analysis.pure)
executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="ColibriLauncher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(root / "desktop/src-tauri/icons/icon.ico") if sys.platform == "win32" else None,
)
collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="ColibriLauncher",
)
