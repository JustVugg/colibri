#!/usr/bin/env python3
"""Keep coli_env.h in sync with the getenv() calls in the engines.

Run it with `make -C c check-env`. It fails when:

  - the code reads a variable the registry does not list (the check in
    coli_env_check() would then report a real variable as "unknown", which is
    worse than not checking at all);
  - the registry lists a variable no code reads (a stale row that will suggest
    a name that does nothing);
  - the table is not sorted by name, which silently breaks the binary search in
    coli_env_find(), or contains a duplicate name.

c/tests/ is excluded: fixtures read their own variables (EXPERT_RAW, TMPDIR)
which are not part of the engines' surface.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(HERE)

GETENV = re.compile(r'(?:q38_env_bool|(?:compat_)?getenv(?:_utf8)?)\(\s*"([A-Z0-9_]+)"')
ROW = re.compile(r'^\s*\{"([A-Z0-9_]+)"\s*,')


def scan_sources():
    """Every variable the engines actually read, and where."""
    found = {}
    for root, dirs, files in os.walk(C_DIR):
        dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__", "shaders")]
        for f in files:
            if not f.endswith((".c", ".h", ".cu", ".mm")):
                continue
            # coli_env.h is scanned like any other source: its own getenv() calls
            # (COLI_ENV_STRICT) are real reads. The table rows are `{"NAME",` and
            # do not match GETENV, so listing a variable is not mistaken for
            # reading it.
            path = os.path.join(root, f)
            with open(path, errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    for m in GETENV.finditer(line):
                        found.setdefault(m.group(1), []).append(
                            "%s:%d" % (os.path.relpath(path, C_DIR), n))
    return found


def scan_registry():
    """The table's rows, in file order."""
    rows = []
    with open(os.path.join(C_DIR, "coli_env.h"), errors="replace") as fh:
        inside = False
        for line in fh:
            if "coli_env_table[] = {" in line:
                inside = True
                continue
            if inside:
                if line.startswith("};"):
                    break
                m = ROW.match(line)
                if m:
                    rows.append(m.group(1))
    return rows


def main():
    code = scan_sources()
    rows = scan_registry()
    problems = []

    if not rows:
        problems.append("could not parse any rows out of coli_env.h")

    dupes = {n for n in rows if rows.count(n) > 1}
    if dupes:
        problems.append("duplicate rows: " + ", ".join(sorted(dupes)))

    if rows != sorted(rows):
        for a, b in zip(rows, rows[1:]):
            if a > b:
                problems.append(
                    "table is not sorted (%r before %r) -- coli_env_find() "
                    "binary-searches it, so an unsorted table silently fails "
                    "to find valid names" % (a, b))
                break

    missing = sorted(set(code) - set(rows))
    for v in missing:
        problems.append("read by the code but missing from coli_env.h: %s (%s)"
                        % (v, code[v][0]))

    stale = sorted(set(rows) - set(code))
    for v in stale:
        problems.append("in coli_env.h but no code reads it: %s" % v)

    if problems:
        sys.stderr.write("coli_env.h is out of sync with the sources:\n")
        for p in problems:
            sys.stderr.write("  - %s\n" % p)
        sys.stderr.write(
            "\nAdd, remove or re-sort the row in c/coli_env.h so the registry "
            "matches what the engines read.\n")
        return 1

    print("check-env: %d variables, registry matches the sources" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
