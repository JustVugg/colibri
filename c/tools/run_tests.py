#!/usr/bin/env python3
"""Run C test binaries, exiting non-zero on the first failure.

Used by `make test-c` so the test runner works from any shell (cmd.exe,
PowerShell, Git Bash, MSYS2) without a POSIX `for` loop. Each test binary
is a positional argument.
"""
import os
import subprocess
import sys

failed = []
for binary in sys.argv[1:]:
    # An absolute path: Windows looks a bare relative name such as tests/x.exe
    # up on the search path, not in the current directory, when
    # NoDefaultCurrentDirectoryInExePath is set; and MSYS2's Python keeps the
    # forward slash through normpath (os.sep is '/' under MSYSTEM).
    rc = subprocess.call([os.path.abspath(binary)])
    if rc != 0:
        failed.append(binary)
if failed:
    for f in failed:
        print(f"FAILED: {f}", file=sys.stderr)
    sys.exit(1)
