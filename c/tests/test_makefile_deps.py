"""Makefile header prerequisites must cover what each engine actually includes.

`make` tracks file timestamps, not #include graphs, and this Makefile has no
-MMD/-include dependency generation. So a header that an engine includes but
that is absent from its rule's prerequisite list produces the worst kind of
build result: `make` reports success, changes nothing, and leaves a STALE
binary that looks freshly built.

Reproduce on a tree without the accompanying fix:

    make colibri                      # builds
    touch edge_runtime.h && make colibri
    -> "make: 'colibri' is up to date."   # colibri.c includes it

whereas touching a header the rule DOES list (st.h) relinks as expected.

This test is the enforcement a comment cannot provide: nothing else compares
the two lists, and the drift is invisible from inside a green build.
"""
import re
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent

# Engine binaries whose rule compiles a single .c directly into the target.
ENGINE_RULES = {
    "colibri$(EXE)": "colibri.c",
    "inkling$(EXE)": "inkling.c",
    "kimi_k3$(EXE)": "kimi_k3.c",
    "qwen36$(EXE)": "qwen36.c",
    "olmoe$(EXE)": "olmoe.c",
}

RULE_RE = re.compile(r"^([A-Za-z0-9_./$()]+)\s*:\s+(.*)$")
INCLUDE_RE = re.compile(r'^#include "([^"]+)"', re.M)


def _prereqs_by_target():
    out = {}
    for line in (C_DIR / "Makefile").read_text().split("\n"):
        m = RULE_RE.match(line)
        if m and m.group(1) in ENGINE_RULES:
            out[m.group(1)] = set(m.group(2).split())
    return out


class MakefileHeaderDepsTest(unittest.TestCase):
    def test_every_engine_rule_lists_the_headers_its_source_includes(self):
        prereqs = _prereqs_by_target()
        self.assertEqual(
            set(prereqs), set(ENGINE_RULES),
            "an engine rule was renamed or removed -- update ENGINE_RULES",
        )
        problems = []
        for target, source in sorted(ENGINE_RULES.items()):
            src = C_DIR / source
            if not src.exists():
                continue
            included = set(INCLUDE_RE.findall(src.read_text()))
            # Only headers that exist in c/ are ours to track; system headers
            # and anything generated elsewhere are not prerequisites.
            local = {h for h in included if (C_DIR / h).exists()}
            missing = sorted(local - prereqs[target])
            if missing:
                problems.append(f"{target} ({source}) is missing: {' '.join(missing)}")
        self.assertEqual(
            problems, [],
            "Headers included by an engine but absent from its Makefile "
            "prerequisites. Editing one of these alone will NOT relink the "
            "binary -- make will report success and leave a stale artifact "
            ":\n  " + "\n  ".join(problems),
        )


if __name__ == "__main__":
    unittest.main()
