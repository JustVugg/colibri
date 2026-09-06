"""Makefile header prerequisites must cover what each engine actually includes.

`make` tracks file timestamps, not #include graphs, and this Makefile has no
-MMD/-include dependency generation. So a header that an engine includes but
that is absent from its rule's prerequisite list produces the worst kind of
build result: `make` reports success, changes nothing, and leaves a STALE
binary that looks freshly built.

Reproduce on a tree without the accompanying fix, without a compiler:

    touch colibri && sleep 1 && touch edge_runtime.h
    make -q colibri; echo $?
    -> 0        # "up to date", though colibri.c includes edge_runtime.h

whereas touching a header the rule DOES list (st.h) exits 1, meaning make
would relink. This test is the enforcement a comment cannot provide: nothing
else compares the two lists, and the drift is invisible from inside a green
build.

The engine list is DERIVED from the Makefile, never kept here. A hand-written
list can only catch a rule that was renamed or removed; it cannot catch a rule
that was never added, which is the case that actually happened -- glm53 and
qwen38 sat with incomplete prerequisites while a hand-listed version of this
test passed green. Anything matching `NAME$(EXE):` with a `NAME.c` beside it
is a target built from one translation unit, so its prerequisites are
checkable, and the next engine added is covered without anyone remembering.
"""
import re
import unittest
from pathlib import Path

from family_registry import FAMILIES

C_DIR = Path(__file__).resolve().parent.parent

# `NAME$(EXE): prereqs`. Targets with a path separator (tests/foo$(EXE)) are
# deliberately excluded: this checks the engines built at the top of c/.
RULE_RE = re.compile(r"(?m)^([A-Za-z0-9_]+)\$\(EXE\):[ \t]*(.*)$")
INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.M)

# deepseek_v4 is built by Makefile.deepseek-v4 through a phony delegating
# target, so it has no `NAME$(EXE):` rule here and no prerequisite list of its
# own to check. test_family_registry pins that separately.
SEPARATE_MAKEFILE = {"deepseek-v4"}


def _engine_rules():
    """Every target built from a single same-named .c, taken from the Makefile.

    Line continuations are folded first: a rule wrapped across lines would
    otherwise present a truncated prerequisite list and report headers as
    missing that are listed on the next line.
    """
    text = (C_DIR / "Makefile").read_text(encoding="utf-8")
    joined = re.sub(r"\\\n[ \t]*", " ", text)
    rules = {}
    for name, prereqs in RULE_RE.findall(joined):
        source = C_DIR / f"{name}.c"
        if source.exists():
            rules[name] = (source, set(prereqs.split("#", 1)[0].split()))
    return rules


class MakefileHeaderDepsTest(unittest.TestCase):
    def test_the_engine_list_is_derived_and_not_empty(self):
        """A parse that matches nothing would make every other check vacuous.

        The rule syntax is the input to this whole file. If it ever changes,
        the header check below starts passing for zero targets and says
        nothing, which reads exactly like a clean tree. The registry is the
        authority on what must be buildable, so cross-check against it rather
        than against a second list kept here.
        """
        rules = _engine_rules()
        self.assertTrue(rules, "no `NAME$(EXE):` rule with a matching NAME.c was "
                               "found -- the Makefile rule syntax changed and "
                               "this file now checks nothing")
        for family in FAMILIES:
            if family.build_target in SEPARATE_MAKEFILE:
                continue
            with self.subTest(family=family.id):
                self.assertIn(
                    family.build_target, rules,
                    f"{family.id}: registered family whose build target is not "
                    f"a checkable `{family.build_target}$(EXE):` rule")

    def test_every_engine_rule_lists_the_headers_its_source_includes(self):
        problems = []
        for target, (source, prereqs) in sorted(_engine_rules().items()):
            included = set(INCLUDE_RE.findall(source.read_text(encoding="utf-8")))
            # Only headers that exist in c/ are ours to track; system headers
            # and anything generated elsewhere are not prerequisites.
            local = {h for h in included if (C_DIR / h).exists()}
            missing = sorted(local - prereqs)
            if missing:
                problems.append(f"{target} ({source.name}) is missing: "
                                f"{' '.join(missing)}")
        self.assertEqual(
            problems, [],
            "Headers included by an engine but absent from its Makefile "
            "prerequisites. Editing one of these alone will NOT relink the "
            "binary -- make will report success and leave a stale artifact "
            ":\n  " + "\n  ".join(problems),
        )


if __name__ == "__main__":
    unittest.main()
