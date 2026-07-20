"""Version accessor for the pip package.

The single source of truth is c/version.py (#394): coli --version and the
GitHub Release workflow read it, so the pip metadata must read the SAME file
instead of carrying a second literal that would drift on the first bump.

From a checkout (the supported install: `pip install -e .`) the file is read
directly. From an installed wheel c/ is not on disk, so fall back to the
package metadata that setuptools baked at build time from that same file.
"""
from pathlib import Path

try:
    _ns = {}
    exec((Path(__file__).resolve().parent.parent / "c" / "version.py").read_text(), _ns)
    __version__ = _ns["__version__"]
except OSError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("colibri-engine")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
