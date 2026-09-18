"""Keep native desktop input/display and SVG; omit unused optional runtimes."""

from pathlib import Path

from PyInstaller.utils.hooks.qt import add_qt6_dependencies


hiddenimports, binaries, datas = add_qt6_dependencies(__file__)
# Filter plugins before PyInstaller's binary dependency scan. Excluding Python
# modules alone does not exclude DLLs pulled in by a Qt plugin. In particular,
# Virtual Keyboard is GPL-only, and neither it nor PDF is used by this GUI.
binaries = [
    (source, destination) for source, destination in binaries
    if "virtualkeyboard" not in Path(source).name.lower()
    and "qgtk3" not in Path(source).name.lower()
    and (Path(destination).name != "imageformats"
         or Path(source).name.lower() in {"qsvg.dll", "libqsvg.so", "qico.dll", "libqico.so"})
]
