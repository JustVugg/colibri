"""Collect notices from the actual build environment without network access."""

import json
from importlib.metadata import distribution
from pathlib import Path
import shutil
import ssl
import subprocess
import sys


def collect_notices(binaries, workpath):
    directory = Path(__file__).resolve().parent
    files = []
    records = []

    def add(source, destination, component):
        source = Path(source)
        if source.is_file():
            files.append((str(source), destination))
            records.append({"component": component, "notice": f"{destination}/{source.name}"})

    # Official CPython installers and Conda include a license beside Python.
    # The reference copy also retains incorporated-software acknowledgements.
    add(directory / "PYTHON-LICENSE.txt", "licenses/python", "Python reference notices")
    candidates = [Path(sys.base_prefix) / name for name in ("LICENSE.txt", "LICENSE_PYTHON.txt", "LICENSE")]
    python_license = next((path for path in candidates if path.is_file()), directory / "PYTHON-LICENSE.txt")
    add(python_license, "licenses/python", "Build interpreter license")
    # Use a stable destination for the smoke check, regardless of installer.
    generated = Path(workpath) / "launcher-notices"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "LICENSE.txt").write_bytes(python_license.read_bytes())
    add(generated / "LICENSE.txt", "licenses/python", "Build interpreter license")

    # All supported build interpreters use OpenSSL 3 (Apache-2.0). Fail rather
    # than attach this license to an older, differently licensed runtime.
    if ssl.OPENSSL_VERSION_INFO[0] != 3:
        raise RuntimeError("Launcher packaging requires an OpenSSL 3 build of Python")
    openssl_directory = generated / "openssl"
    openssl_directory.mkdir(exist_ok=True)
    (openssl_directory / "LICENSE.txt").write_bytes((directory / "OPENSSL-LICENSE.txt").read_bytes())
    add(openssl_directory / "LICENSE.txt", "licenses/openssl", ssl.OPENSSL_VERSION)
    pyinstaller = distribution("PyInstaller")
    bootloader_notices = [pyinstaller.locate_file(item) for item in pyinstaller.files or ()
                         if Path(item).name == "COPYING.txt"]
    if not bootloader_notices:
        raise RuntimeError("The installed PyInstaller is missing its COPYING.txt bootloader notice")
    for path in bootloader_notices:
        add(path, "licenses/pyinstaller", f"PyInstaller {pyinstaller.version}")

    # Include notices from packages supplying the collected system libraries.
    # Debian/Ubuntu builds (including CI) retain their complete copyright files.
    sources = {Path(source).resolve() for _, source, kind in binaries
               if kind != "SYMLINK" and Path(source).is_file()}
    if sys.platform == "linux" and shutil.which("dpkg-query"):
        owners = set()
        extracted_docs = set()
        for source in sources:
            # A local build may use libraries from unpacked .deb packages.
            # Retain notices from that same extraction, not an unrelated
            # installed package with a coincidentally matching filename.
            for index, part in enumerate(source.parts):
                if part == "usr" and index > 1:
                    docs = Path(*source.parts[:index]) / "usr/share/doc"
                    if docs.is_dir():
                        extracted_docs.add(docs)
            query = subprocess.run(["dpkg-query", "--search", str(source)],
                                   capture_output=True, text=True, check=False)
            for line in query.stdout.splitlines():
                if ": " in line:
                    owners.add(line.split(": ", 1)[0].split(":", 1)[0])
        for package in sorted(owners):
            version = subprocess.run(["dpkg-query", "--show", "--showformat=${Version}", package],
                                     capture_output=True, text=True, check=False).stdout.strip()
            add(Path("/usr/share/doc") / package / "copyright",
                f"licenses/system/{package}", f"{package} {version}")
        for docs in sorted(extracted_docs):
            for path in sorted(docs.glob("*/copyright")):
                add(path, f"licenses/system/extracted/{path.parent.name}",
                    f"Extracted runtime package: {path.parent.name}")
        # Debian copyright files can refer to these shared license texts.
        for path in sorted(Path("/usr/share/common-licenses").glob("*")):
            add(path, "licenses/system/common-licenses", "Distribution shared license texts")

    # Conda's metadata maps DLLs to package-specific license directories. This
    # is supplemental; the standard CPython/Qt build needs no Conda files.
    prefix = Path(sys.base_prefix)
    for metadata in sorted((prefix / "conda-meta").glob("*.json")):
        package = json.loads(metadata.read_text(encoding="utf-8"))
        if not any((prefix / name).resolve() in sources for name in package.get("files", [])):
            continue
        extracted = package.get("extracted_package_dir")
        if extracted:
            for path in sorted((Path(extracted) / "info/licenses").rglob("*")):
                add(path, f"licenses/conda/{package['name']}",
                    f"{package['name']} {package['version']}")

    manifest = generated / "runtime-notices.json"
    manifest.write_text(json.dumps({
        "python": sys.version.split()[0], "openssl": ssl.OPENSSL_VERSION,
        "pyside6": distribution("PySide6").version,
        "notices": records,
    }, indent=2) + "\n", encoding="utf-8")
    add(manifest, "licenses", "Build runtime inventory")
    return files
