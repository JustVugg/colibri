#!/usr/bin/env python3
"""The tests build without OpenMP too. On a Mac without libomp the Makefile builds
single-threaded (no -fopenmp: "libomp not found: building single-threaded"), and then
<omp.h> is not on the include path and an omp_* function has no declaration, unless
colibri.c's own stubs come with the test. Five tests included <omp.h> unguarded and one
called omp_set_num_threads, so `make check` stopped at the first of them on such a Mac.
CI builds only with OpenMP (Linux and Windows), so only this check sees it.

The rules, over every tests/test_*.c:
  - <omp.h> is included only where _OPENMP is defined;
  - an omp_*() call outside such a region only in a test that includes colibri.c.
"""
from pathlib import Path
import re
import unittest

C = Path(__file__).resolve().parents[1]
COND = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)")
OMP_H = re.compile(r"^\s*#\s*include\s*<omp\.h>")
OMP_CALL = re.compile(r"\bomp_\w+\s*\(")
STUBS = re.compile(r'^\s*#\s*include\s*"(?:\.\./)?colibri\.c"', re.M)


def _blank_comments(src):
    """comments as spaces, so the line numbers stay"""
    src = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _openmp_test(cond):
    """+1: true only with OpenMP (#ifdef _OPENMP, #if defined(_OPENMP)), -1: only without it,
    0: says nothing about OpenMP"""
    if "_OPENMP" not in cond:
        return 0
    return -1 if re.search(r"!\s*(?:defined\s*\(?\s*)?_OPENMP", cond) else 1


def unguarded(src):
    """(line, what) for each <omp.h> include and omp_*() call outside an _OPENMP region"""
    out, stack = [], []   # per open #if: the _OPENMP sense of its current branch
    for no, line in enumerate(_blank_comments(src).split("\n"), 1):
        m = COND.match(line)
        if m:
            kw, rest = m.groups()
            if kw == "ifdef":
                stack.append(1 if rest.strip() == "_OPENMP" else 0)
            elif kw == "ifndef":
                stack.append(-1 if rest.strip() == "_OPENMP" else 0)
            elif kw == "if":
                stack.append(_openmp_test(rest))
            elif kw == "elif" and stack:
                stack[-1] = _openmp_test(rest)
            elif kw == "else" and stack:
                stack[-1] = -stack[-1]
            elif kw == "endif" and stack:
                stack.pop()
            continue
        if any(s > 0 for s in stack):
            continue
        if OMP_H.match(line):
            out.append((no, "#include <omp.h>"))
        elif OMP_CALL.search(line):
            out.append((no, OMP_CALL.search(line).group(0).rstrip("( ")))
    return out


class OmpGuardSourceTests(unittest.TestCase):
    def test_every_test_builds_without_openmp(self):
        bad = []
        for path in sorted((C / "tests").glob("test_*.c")):
            src = path.read_text(encoding="utf-8", errors="replace")
            stubs = bool(STUBS.search(src))
            for no, what in unguarded(src):
                if what.startswith("omp_") and stubs:
                    continue
                bad.append(f"tests/{path.name}:{no}: {what}")
        self.assertEqual(bad, [], "OpenMP used outside #ifdef _OPENMP (a build without libomp fails)")

    def test_the_rule_reads_the_guards(self):
        self.assertEqual(unguarded("#include <omp.h>\nint main(void){ omp_set_num_threads(1); }\n"),
                         [(1, "#include <omp.h>"), (2, "omp_set_num_threads")])
        self.assertEqual(unguarded("#ifdef _OPENMP\n#include <omp.h>\n#endif\n"
                                   "#if defined(_OPENMP) && 1\nint t = omp_get_max_threads();\n#endif\n"), [])
        # test_omp_tune.c's form: the OpenMP part is the #else of an #ifndef
        self.assertEqual(unguarded("#ifndef _OPENMP\nreturn 0;\n#else\nomp_set_num_threads(2);\n#endif\n"), [])
        self.assertEqual(unguarded("#ifdef _OPENMP\n#else\nomp_set_num_threads(2);\n#endif\n"),
                         [(3, "omp_set_num_threads")])
        self.assertEqual(unguarded("/* omp_set_num_threads(1) */\n// omp_get_max_threads()\n"), [])


if __name__ == "__main__":
    unittest.main()
