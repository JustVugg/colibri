"""PyInstaller entry point for the optional desktop process launcher."""

from colibri.launcher.app import main

if __name__ == "__main__":
    raise SystemExit(main())
