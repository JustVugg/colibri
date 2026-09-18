# Packaged runtime notices

The launcher source is Apache-2.0. Its unmodified, dynamically linked runtime
libraries retain their own licenses. Notices are installed under `_internal/licenses/`.

## Qt 6.11.2

Packaging requires PySide6 6.11.2. Qt Base, Qt Declarative, Qt SVG, PySide and
Shiboken provide the interface runtime; their LGPLv3 option is accompanied by
the LGPLv3 and incorporated GPLv3 texts. Qt is copyright The Qt Company Ltd.
and other contributors. PySide and Shiboken retain their respective notices.
The build preserves the installed PySide distribution metadata and license files.

`THIRD_PARTY_NOTICES.txt` retains upstream attribution records, copyright
notices and referenced license texts from the Qt Base, Qt Declarative and Qt
SVG v6.11.2 sources. The inventories also describe optional components; their
presence in this notice does not mean each component is bundled. The packaging
hook excludes unused Virtual Keyboard, PDF, extra image codecs and GTK theme
plugins before dependency collection. Virtual Keyboard is GPL-only and is not
part of this launcher package.

Exact source trees, including build instructions and further notices:

- [Qt Base 6.11.2](https://github.com/qt/qtbase/tree/v6.11.2)
- [Qt Declarative 6.11.2](https://github.com/qt/qtdeclarative/tree/v6.11.2)
- [Qt SVG 6.11.2](https://github.com/qt/qtsvg/tree/v6.11.2)
- [PySide/Shiboken 6.11.2 source archive](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.2-src/pyside-setup-everywhere-src-6.11.2.tar.xz)

The one-folder package keeps shared libraries separate under `_internal` and
does not restrict replacing them with compatible builds or debugging those
modifications. Launcher source and rebuilding instructions are in
`colibri/launcher/`, `desktop/launcher.spec`, and `docs/launcher.md`.

## Python and other native libraries

The build copies the Python installer's license when available. The checked-in
Python notice retains the PSF license, historical licenses and incorporated
software acknowledgements from CPython 3.12.12 as a reference and fallback.
The build also includes the OpenSSL 3 Apache-2.0 text and the installed
PyInstaller's copyright, license and bootloader exception.

For Debian/Ubuntu runtime libraries the collector copies installed package
copyright files, including their shared license texts. It also finds copyright
files beside libraries from extracted Debian packages. For Conda libraries it
copies notices from the supplying package's extracted license directory when
available. `runtime-notices.json` records the build versions and collected
notices without including local user paths.

`SOURCES.json` records official source URLs and SHA-256 hashes for the checked-in
notices. Builds do not download notices. Updating Qt requires updating the
version guard, CI pin, attribution inventory and source references together.
When changing the Python/runtime distribution, review its additional binary
dependencies and supply any notices or corresponding sources not present in
that distribution's collected files.
