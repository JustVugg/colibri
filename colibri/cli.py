"""Entry point for `coli` when installed via pip.

Delegates to the original c/coli script which handles all subcommands.
This wrapper exists so `pip install colibri-engine` creates a `coli` console
script that works without the user having to add c/ to PATH manually.
"""

import os
import sys
import runpy


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    engine_dir = os.path.join(os.path.dirname(here), "c")
    coli_script = os.path.join(engine_dir, "coli")

    if not os.path.exists(coli_script):
        sys.exit(
            "colibri engine directory not found.\n"
            "Install from source: git clone + pip install -e ."
        )

    sys.path.insert(0, engine_dir)
    sys.argv[0] = coli_script
    runpy.run_path(coli_script, run_name="__main__")


if __name__ == "__main__":
    main()
