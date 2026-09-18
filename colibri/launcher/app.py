"""Application entry point for the optional Colibri desktop launcher."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    try:
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        from .window import LauncherWindow
    except ImportError as exc:
        if exc.name and (exc.name == "PySide6" or exc.name.startswith("PySide6.")):
            print(
                "The Colibri desktop launcher needs PySide6. "
                'From the Colibri checkout, install it with: python -m pip install -e ".[launcher]"',
                file=sys.stderr,
            )
            return 1
        raise

    QCoreApplication.setOrganizationName("Colibri")
    QCoreApplication.setApplicationName("Colibri Launcher")
    app = QApplication.instance() or QApplication(list(argv if argv is not None else sys.argv))
    window = LauncherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
