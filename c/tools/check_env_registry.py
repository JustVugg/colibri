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
  - a variable in the registry is described nowhere in the reference docs
    (docs/*.md, docs/api-reference/*.md, README.md, GPU_BACKENDS.md) and is not
    in tools/env_undocumented.txt, the list of those that had no description
    when this check began. A new variable needs its line in docs/ENVIRONMENT.md
    (or its engine's page); the list only shrinks: a name in it that is now
    described, or that no longer exists, has to leave it.

c/tests/ is excluded: fixtures read their own variables (EXPERT_RAW, TMPDIR)
which are not part of the engines' surface.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(HERE)

# getenv() and the engines' wrappers that take the literal name first (mimo's
# env_int, qwen38's q38_env_*, vk_tier's env_gb, ...). A new wrapper goes here,
# or the names it reads look unread.
READERS = ("getenv", "compat_getenv", "compat_getenv_utf8", "getenv_utf8",
           "q38_env_bool", "q38_env_int_in_range", "q38_env_positive_int",
           "env_int", "env_gb", "vkc_kv_env", "coli_env_on", "coli_omp_env_on")
GETENV = re.compile(r'\b(?:%s)\s*\(\s*"([A-Z0-9_]+)"' % "|".join(READERS))
# Consume literals and comments before looking for calls. This is a lexical
# scan of literal names, not a C preprocessor or dynamic-name evaluator.
SOURCE_TOKEN = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|'
    + GETENV.pattern, re.DOTALL)
ROW = re.compile(r'^\s*\{"([A-Z0-9_]+)"\s*,')


def scan_sources():
    """Every variable the engines actually read, and where."""
    found = {}
    for root, dirs, files in os.walk(C_DIR):
        dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__", "shaders")]
        for f in files:
            if not f.endswith((".c", ".h", ".cu", ".m", ".mm")):
                continue
            # coli_env.h is scanned like any other source: its own getenv() calls
            # (COLI_ENV_STRICT) are real reads. The table rows are `{"NAME",` and
            # do not match GETENV, so listing a variable is not mistaken for
            # reading it.
            path = os.path.join(root, f)
            with open(path, errors="replace") as fh:
                source = fh.read()
                for m in SOURCE_TOKEN.finditer(source):
                    if m.group(1) is not None:
                        n = source.count("\n", 0, m.start()) + 1
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


def doc_files():
    """The reference docs: the pages a user reads to learn what a variable does."""
    root = os.path.dirname(C_DIR)
    files = [os.path.join(root, name) for name in ("README.md", "GPU_BACKENDS.md")]
    for folder in ("docs", os.path.join("docs", "api-reference")):
        folder = os.path.join(root, folder)
        if os.path.isdir(folder):
            files += [os.path.join(folder, name) for name in sorted(os.listdir(folder))
                      if name.endswith(".md")]
    return [path for path in files if os.path.isfile(path)]


def scan_docs():
    """Every name-shaped word in the reference docs, or None when there are no docs here."""
    files = doc_files()
    if not files:
        return None
    words = set()
    for path in files:
        with open(path, errors="replace") as fh:
            words.update(re.findall(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]+(?![A-Za-z0-9_])", fh.read()))
    return words


def scan_undocumented():
    """tools/env_undocumented.txt: one name a line, `#` comments."""
    path = os.path.join(C_DIR, "tools", "env_undocumented.txt")
    if not os.path.isfile(path):
        return []
    with open(path, errors="replace") as fh:
        return [line.split("#")[0].strip() for line in fh if line.split("#")[0].strip()]


def doc_problems(rows):
    described = scan_docs()
    if described is None:
        return []
    owed = scan_undocumented()
    problems = []
    if owed != sorted(set(owed)):
        problems.append("tools/env_undocumented.txt is not sorted, or repeats a name")
    for v in sorted(set(rows) - described - set(owed)):
        problems.append("in coli_env.h but described in no reference doc: %s "
                        "(add its line to docs/ENVIRONMENT.md)" % v)
    for v in sorted(set(owed) & described):
        problems.append("described in the docs now, remove it from "
                        "tools/env_undocumented.txt: %s" % v)
    for v in sorted(set(owed) - set(rows)):
        problems.append("in tools/env_undocumented.txt but not in coli_env.h: %s" % v)
    return problems


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

    problems = doc_problems(rows)
    if problems:
        sys.stderr.write("the reference docs are out of sync with coli_env.h:\n")
        for p in problems:
            sys.stderr.write("  - %s\n" % p)
        return 1

    owed = len(scan_undocumented())
    print("check-env: %d variables, registry matches the sources%s" % (
        len(rows), "; %d still to describe (tools/env_undocumented.txt)" % owed if owed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
