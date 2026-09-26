"""A rule that compiles a Vulkan-hooked engine must link $(VK_OBJ).

Under VK=1, CFLAGS carries -DCOLI_VULKAN. A translation unit that includes an
engine with `#ifdef COLI_VULKAN` hooks (colibri.c, kimi_k3.c, ...) then calls
coli_vk_*, and the link fails unless the rule also links $(VK_OBJ). That
variable is empty unless VK=1, so the default build and every default CI job
stay green while `make test-c VK=1` is broken. #1728 fixed the 43 rules that
`make -k test-c VK=1` reported. Four more gates (test_xdna_qt_state,
test_xdna_failure, test_logprob_status, test_ablate_mode) arrived in pull
requests merged after it, and nine on-demand rules outside TEST_BINS -- the
benches, the XDNA physical probe, the e8x4g64 loader harness -- were never
built by test-c, so no such run could report them.

Nothing here is a hand-kept list. The hooked sources are the files with a
preprocessor conditional on COLI_VULKAN. A link recipe needs $(VK_OBJ) when a
.c file it compiles reaches one of them through quoted #includes, and the rule
must list $(VK_OBJ) as a prerequisite too: otherwise a clean build can reach
the link before backend_vulkan.o exists (#1728 found six rules like that). Flag
variables whose definition filters -DCOLI_VULKAN out, such as
SEGMENT_CPU_CFLAGS, are recognised from that definition, and `-c` compiles
are not links.
"""
import re
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent

VK_HOOK_RE = re.compile(
    r"(?m)^[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif)\b.*\bCOLI_VULKAN\b")
INCLUDE_RE = re.compile(r'(?m)^[ \t]*#[ \t]*include[ \t]*"([^"]+)"')
ASSIGN_RE = re.compile(
    r"^(?:override[ \t]+)?([A-Za-z0-9_]+)[ \t]*(?:\+|\?|!|::?)?=[ \t]*(.*)$")
RULE_RE = re.compile(r"^([^\t#][^=]*?)[ \t]*:(?![=:])[ \t]*(.*)$")
VAR_RE = re.compile(r"\$\(([A-Za-z0-9_]+)\)")
# Lines that are not rules even when they contain a colon, e.g.
# `$(warning mixed HIP_ARCH list: ...)`. A rule may still START with `$(`:
# `$(SEGMENT_BUILD_DIR)/glm.o: colibri.c ...`.
DIRECTIVE_RE = re.compile(
    r"^(?:ifeq|ifneq|ifdef|ifndef|else|endif|(?:-|s)?include|export"
    r"|unexport|vpath)\b|^\$\((?:error|warning|info|file|shell|eval|call)\b")


def _parse_makefile():
    """(rules, variables) from c/Makefile, continuations folded.

    rules: [(targets, prerequisites, recipe_lines)], prerequisites being the
    normal ones only (an order-only $(VK_OBJ) would not relink on change).
    variables: name -> every value it is assigned anywhere; the union is
    enough to find .c files and COLI_VULKAN filters, whatever the platform.
    """
    text = (C_DIR / "Makefile").read_text(encoding="utf-8")
    text = re.sub(r"\\\n[ \t]*", " ", text)
    rules, variables, current, in_define = [], {}, None, False
    for line in text.splitlines():
        if in_define:
            in_define = not line.startswith("endef")
            continue
        if line.startswith("define "):
            in_define, current = True, None
            continue
        if line.startswith("\t"):
            if current is not None:
                current[2].append(line.strip())
            continue
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        assign = ASSIGN_RE.match(stripped)
        if assign:
            variables.setdefault(assign.group(1), []).append(assign.group(2))
            current = None
            continue
        if DIRECTIVE_RE.match(stripped):
            continue
        rule = RULE_RE.match(stripped)
        if rule:
            prereqs = rule.group(2).split(";", 1)[0].split("|", 1)[0].split()
            current = (rule.group(1).split(), prereqs, [])
            rules.append(current)
    return rules, variables


def _expand_words(word, variables, depth=0):
    """Every word a $(VAR) can stand for, across all its definitions."""
    match = VAR_RE.fullmatch(word)
    if not match or depth > 8:
        return [word]
    words = []
    for value in variables.get(match.group(1), []):
        for part in value.split():
            words.extend(_expand_words(part, variables, depth + 1))
    return words


def _vk_stripping_vars(variables):
    return {name for name, values in variables.items()
            if any("filter-out" in v and "-DCOLI_VULKAN" in v for v in values)}


class _IncludeGraph:
    """Quoted-#include reachability, reading each file once."""

    def __init__(self):
        self._files = {}

    def _scan(self, path):
        if path not in self._files:
            text = path.read_text(encoding="utf-8", errors="replace")
            includes = []
            for name in INCLUDE_RE.findall(text):
                for base in (path.parent, C_DIR):
                    candidate = (base / name).resolve()
                    if candidate.is_file():
                        includes.append(candidate)
                        break
            self._files[path] = (bool(VK_HOOK_RE.search(text)), includes)
        return self._files[path]

    def reaches_vk_hook(self, source):
        """True when `source` or anything it quote-includes has a hook."""
        seen, stack = set(), [source.resolve()]
        while stack:
            path = stack.pop()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            hooked, includes = self._scan(path)
            if hooked:
                return True
            stack.extend(includes)
        return False


def _rules_needing_vk_obj():
    """[(target, has_prerequisite, has_link_argument)] for every rule whose
    link recipe compiles a translation unit that reaches a COLI_VULKAN hook."""
    rules, variables = _parse_makefile()
    stripping = _vk_stripping_vars(variables)
    graph = _IncludeGraph()
    found = []
    for targets, prereqs, recipe in rules:
        for line in recipe:
            words = line.split()
            if "$(CC)" not in words or "-c" in words or "-UCOLI_VULKAN" in words:
                continue
            if stripping & {m.group(1) for m in map(VAR_RE.fullmatch, words) if m}:
                continue
            sources = []
            for word in words:
                if word == "$<":
                    sources.extend(prereqs[:1])
                elif word == "$^":
                    sources.extend(prereqs)
                else:
                    sources.extend(_expand_words(word, variables))
            if any(s.endswith(".c") and graph.reaches_vk_hook(C_DIR / s)
                   for s in sources):
                found.append((" ".join(targets), "$(VK_OBJ)" in prereqs,
                              "$(VK_OBJ)" in words or "$^" in words))
    return found


class MakefileVkObjTest(unittest.TestCase):
    def test_the_scan_is_derived_and_not_vacuous(self):
        """A parse that matches nothing would pass the check below silently.

        Cross-check against the engines themselves: every NAME.c carrying a
        COLI_VULKAN hook that has a `NAME$(EXE):` rule must be found by the
        same scan, since that rule compiles it directly.
        """
        hooked = sorted(p.stem for p in C_DIR.glob("*.c")
                        if VK_HOOK_RE.search(p.read_text(encoding="utf-8",
                                                         errors="replace")))
        self.assertTrue(hooked, "no .c file under c/ has a COLI_VULKAN "
                                "conditional -- the hook syntax changed and "
                                "this file now checks nothing")
        rules, _ = _parse_makefile()
        engine_rules = {name for name in hooked
                        if any(f"{name}$(EXE)" in t for t, _, _ in rules)}
        self.assertTrue(engine_rules, f"no `NAME$(EXE):` rule for {hooked}")
        found = {target for target, _, _ in _rules_needing_vk_obj()}
        for name in sorted(engine_rules):
            self.assertIn(f"{name}$(EXE)", found,
                          f"the scan missed {name}$(EXE), which compiles "
                          f"{name}.c itself -- the recipe parse is broken")
        self.assertTrue(any(t.startswith("tests/") for t in found),
                        "the scan found no test rule including an engine")

    def test_every_rule_compiling_a_hooked_engine_links_vk_obj(self):
        missing = []
        for target, has_prereq, has_link in _rules_needing_vk_obj():
            if not has_prereq:
                missing.append(f"{target}: $(VK_OBJ) not a prerequisite")
            if not has_link:
                missing.append(f"{target}: $(VK_OBJ) not on the link line")
        self.assertFalse(
            missing,
            "these rules compile a source with COLI_VULKAN hooks, so under "
            "VK=1 they reference coli_vk_* and must link backend_vulkan.o:\n  "
            + "\n  ".join(missing))


if __name__ == "__main__":
    unittest.main()
