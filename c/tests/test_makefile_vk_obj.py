"""A link that calls the Vulkan backend must link $(VK_OBJ).

Under VK=1, CFLAGS carries -DCOLI_VULKAN. A translation unit that includes an
engine with Vulkan hooks (colibri.c, kimi_k3.c, ...) then calls coli_vk_*, and
the link fails unless the rule also links $(VK_OBJ). That variable is empty
unless VK=1, so the default build and every default CI job stay green while
`make test-c VK=1` is broken. #1728 fixed the 43 rules that
`make -k test-c VK=1` reported. Four more gates (test_xdna_qt_state,
test_xdna_failure, test_logprob_status, test_ablate_mode) arrived in pull
requests merged after it, and nine on-demand rules outside TEST_BINS -- the
benches, the XDNA physical probe, the e8x4g64 loader harness -- were never
built by test-c, so no such run could report them.

Whether a link needs the backend is a preprocessor question, not a textual
one: code can pick CUDA over Vulkan with `#elif`, a hook can be an #ifdef that
only reads an environment variable, and a test can define the coli_vk_*
functions itself to fake the device. So nothing here reads #if lines. It asks
make for the expanded VK=1 link commands (`make -Bn`), preprocesses each .c on
a line with that line's own flags, and looks at what survives: a link needs
$(VK_OBJ) when the code calls a function backend_vulkan.h declares and does
not define it. The rule must then link $(VK_OBJ) and list it as a
prerequisite, or a clean build can reach the link before backend_vulkan.o
exists (#1728 found six rules like that). Nothing is hand-listed: the rules
come from the Makefile and the API from backend_vulkan.h.
"""
import os
import re
import shutil
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
MAKE = shutil.which("make")

ASSIGN_RE = re.compile(
    r"^(?:override[ \t]+)?[A-Za-z0-9_]+[ \t]*(?:\+|\?|!|::?)?=")
RULE_RE = re.compile(r"^([^\t#][^=]*?)[ \t]*:(?![=:])[ \t]*(.*)$")
# Lines that are not rules even when they contain a colon, e.g.
# `$(warning mixed HIP_ARCH list: ...)`. A rule may still START with `$(`.
DIRECTIVE_RE = re.compile(
    r"^(?:ifeq|ifneq|ifdef|ifndef|else|endif|(?:-|s)?include|export"
    r"|unexport|vpath)\b|^\$\((?:error|warning|info|file|shell|eval|call)\b")
LINEMARKER_RE = re.compile(r'^#\s+\d+\s+"([^"]+)"')
STRING_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
NAME_RE = re.compile(r"\bcoli_vk_\w+")
DEF_RE = re.compile(r"\b(coli_vk_\w+)\s*\([^;{}()]*\)\s*\{")
# Link-only and dependency-file arguments; the rest of a link line is what
# the compile saw. -MMD/-MF would make `-E` write .d files beside the sources.
SKIP_WORD = re.compile(r"^(-l|-L|-Wl,|-static|-shared|-flto|-fuse-ld|-MMD$|-MD$|-MP$)")
SKIP_PAIR = {"-o", "-MF", "-MT", "-MQ", "-framework"}
INPUT_SUFFIXES = (".c", ".o", ".a", ".so", ".dll", ".lib", ".dylib")


def _link_rules():
    """{target: normal prerequisites} for every rule with a $(CC) link recipe.

    Prerequisites are the one thing `make -n` does not print. Continuations
    are folded first, and define/endef bodies skipped.
    """
    text = (C_DIR / "Makefile").read_text(encoding="utf-8")
    text = re.sub(r"\\\n[ \t]*", " ", text)
    rules, current, in_define = {}, None, False
    for line in text.splitlines():
        if in_define:
            in_define = not line.startswith("endef")
            continue
        if line.startswith("define "):
            in_define, current = True, None
            continue
        if line.startswith("\t"):
            words = line.split()
            if current and "$(CC)" in words and "-c" not in words:
                for target in current[0]:
                    rules[target] = current[1]
            continue
        stripped = line.split("#", 1)[0].strip()
        if not stripped or DIRECTIVE_RE.match(stripped):
            continue
        if ASSIGN_RE.match(stripped):
            current = None
            continue
        rule = RULE_RE.match(stripped)
        if rule:
            prereqs = rule.group(2).split(";", 1)[0].split("|", 1)[0].split()
            current = (rule.group(1).split(), prereqs)
    return rules


def _backend_api():
    text = (C_DIR / "backend_vulkan.h").read_text(encoding="utf-8")
    return set(re.findall(r"\bcoli_vk_\w+", text))


def _make(*args):
    return subprocess.run([MAKE, "--no-print-directory", *args], cwd=C_DIR,
                          capture_output=True, text=True, errors="replace",
                          timeout=600)


def _unresolved(words, api):
    """Backend functions the link line's sources call without defining them,
    or None when this host cannot preprocess them (and so cannot build them
    either; the platforms that can are where the line gets checked)."""
    cc, flags, sources, i = words[0], [], [], 1
    while i < len(words):
        w = words[i]
        if w in SKIP_PAIR:
            i += 2
            continue
        if w.endswith(INPUT_SUFFIXES):
            if w.endswith(".c"):
                sources.append(w)
        elif w != "-c" and not SKIP_WORD.match(w):
            flags.append(w)
        i += 1
    called, defined = set(), set()
    for source in sources:
        r = subprocess.run([cc, "-E", *flags, source], cwd=C_DIR,
                           capture_output=True, text=True, errors="replace",
                           timeout=300)
        if r.returncode != 0:
            return None
        # Keep only this tree's own code: system headers cannot call the
        # backend, and backend_vulkan.h only declares it.
        kept, keep, own = [], True, {}
        for line in r.stdout.splitlines():
            marker = LINEMARKER_RE.match(line)
            if marker:
                path = marker.group(1)
                if path not in own:
                    resolved = (C_DIR / path).resolve()
                    own[path] = (not path.startswith("<")
                                 and resolved.is_relative_to(C_DIR)
                                 and resolved.name != "backend_vulkan.h")
                keep = own[path]
            elif keep:
                kept.append(line)
        code = STRING_RE.sub('""', "\n".join(kept))
        called |= set(NAME_RE.findall(code)) & api
        defined |= set(DEF_RE.findall(code)) & api
    return called - defined


@unittest.skipUnless(MAKE, "make is required")
class MakefileVkObjTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A dry run still rewrites .build-config at parse time; put it back so
        # running this test does not make the next real build relink.
        stamp = C_DIR / ".build-config"
        saved = (stamp.read_bytes(), stamp.stat()) if stamp.exists() else None
        try:
            # $(EXE) from make's own database. Not --eval: GNU Make 3.81,
            # macOS's /usr/bin/make, predates it; -p is older than that.
            database = _make("-pn", "VK=1", ".build-config").stdout
            exe = re.search(r"(?m)^EXE :?= ?(\S*)", database)
            if exe is None:
                raise AssertionError("make -pn printed no EXE variable")
            cls.rules = {t.replace("$(EXE)", exe.group(1)): p
                         for t, p in _link_rules().items()
                         if "$(" not in t.replace("$(EXE)", "") and "%" not in t}
            dry = _make("-Bnk", "VK=1", *sorted(cls.rules))
        finally:
            if saved is None:
                stamp.unlink(missing_ok=True)
            else:
                stamp.write_bytes(saved[0])
                os.utime(stamp, ns=(saved[1].st_atime_ns, saved[1].st_mtime_ns))
        # A link line is any printed command that compiles a .c into an output
        # without -c -- also for rules that write a differently named file
        # (fuzz-rans) -- and the compiler is whatever make put first on it.
        # `make -n` prints recipe continuations as written, so fold them first.
        lines = {}
        for line in re.sub(r"\\\n[ \t]*", " ", dry.stdout).splitlines():
            words = line.split()
            if ("-o" in words and "-c" not in words
                    and words.index("-o") + 1 < len(words)
                    and any(w.endswith(".c") for w in words)):
                lines[words[words.index("-o") + 1]] = words
        compilers = {words[0] for words in lines.values()}
        if lines and not any(shutil.which(c) for c in compilers):
            raise unittest.SkipTest(f"{', '.join(sorted(compilers))} is required")
        cls.links = {t: w for t, w in lines.items() if shutil.which(w[0])}
        api = _backend_api()
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            results = dict(zip(cls.links, pool.map(
                lambda words: _unresolved(words, api), cls.links.values())))
        cls.needs = {t: names for t, names in results.items() if names}
        cls.unchecked = sorted(t for t, names in results.items() if names is None)

    def test_the_scan_is_not_vacuous(self):
        """A broken dry run or preprocess would pass the check below silently.

        Cross-check against the engines: a NAME.c that calls a backend function
        and has a `NAME$(EXE):` rule compiles that call directly, so the scan
        must report the rule as calling the backend.
        """
        api = _backend_api()
        self.assertTrue(api, "backend_vulkan.h declares no coli_vk_*")
        self.assertTrue(self.links, "make -Bn VK=1 printed no link line")
        call = re.compile(r"\b(%s)\s*\(" % "|".join(sorted(api)))
        engines = sorted(
            t for t in self.links
            if "/" not in t and (C_DIR / (t.split(".")[0] + ".c")).is_file()
            and call.search((C_DIR / (t.split(".")[0] + ".c")).read_text(
                encoding="utf-8", errors="replace")))
        self.assertTrue(engines, "no engine source calls the Vulkan backend")
        for target in engines:
            self.assertNotIn(target, self.unchecked,
                             f"{target} could not be preprocessed here")
            self.assertIn(target, self.needs,
                          f"{target} calls the backend in its source, yet the "
                          f"scan found no call -- the dry run or preprocess "
                          f"is broken")
        self.assertTrue(any(t.startswith("tests/") for t in self.needs),
                        "the scan found no test that calls the backend")

    def test_every_link_that_calls_vulkan_links_vk_obj(self):
        missing = []
        for target, names in sorted(self.needs.items()):
            example = sorted(names)[0]
            if "backend_vulkan.o" not in self.links[target]:
                missing.append(f"{target}: not on the link line (calls {example})")
            if target not in self.rules:
                missing.append(f"{target}: its rule is not named after it; "
                               f"list $(VK_OBJ) as that rule's prerequisite")
            elif "$(VK_OBJ)" not in self.rules[target]:
                missing.append(f"{target}: $(VK_OBJ) not a prerequisite")
        self.assertFalse(
            missing,
            "under VK=1 these links call coli_vk_* without defining it, so they "
            "must link backend_vulkan.o:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    unittest.main()
