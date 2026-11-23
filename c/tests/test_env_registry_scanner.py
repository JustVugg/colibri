"""Name coverage must follow calls, not formatting or source comments."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import check_env_registry as registry


class RegistryScannerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "c"   # the docs are looked for beside it
        self.root.mkdir()
        self.patch = patch.object(registry, "C_DIR", str(self.root))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_calls_across_lines_and_whitespace(self):
        (self.root / "engine.c").write_text(
            'getenv (\n "COLI_A");\nq38_env_bool(\n "Q38_B");\n'
            'compat_getenv_utf8("COLI_C");\nenv_int("MIMO_D", 0);\n')
        self.assertEqual(registry.scan_sources(), {
            "COLI_A": ["engine.c:1"], "Q38_B": ["engine.c:3"],
            "COLI_C": ["engine.c:5"], "MIMO_D": ["engine.c:6"]})

    def test_comments_and_strings_are_not_calls(self):
        (self.root / "engine.c").write_text(
            '// getenv("COLI_COMMENT")\n'
            '/* q38_env_bool("Q38_COMMENT") */\n'
            'const char *s = "getenv(\\"COLI_STRING\\")";\n'
            'my_getenv("COLI_OTHER_FUNCTION");\n'
            'getenv("COLI_REAL");\n')
        self.assertEqual(registry.scan_sources(), {"COLI_REAL": ["engine.c:5"]})

    def test_commented_read_cannot_keep_stale_registry_entry_alive(self):
        (self.root / "engine.c").write_text('// getenv("COLI_OLD")\n')
        (self.root / "coli_env.h").write_text(
            'static const ColiEnvVar coli_env_table[] = {\n'
            '    {"COLI_OLD", CE_BOOL, CE_QWEN, 0, NULL},\n};\n')
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(registry.main(), 1)
        self.assertIn("no code reads it: COLI_OLD", error.getvalue())

    def test_fixture_directories_are_excluded(self):
        (self.root / "tests").mkdir()
        (self.root / "tests" / "fixture.c").write_text('getenv("FIXTURE_ONLY");')
        self.assertEqual(registry.scan_sources(), {})

    def test_objective_c_sources_are_scanned(self):
        (self.root / "glue.m").write_text('getenv("COLI_OBJC");\n')
        self.assertEqual(registry.scan_sources(), {"COLI_OBJC": ["glue.m:1"]})

    def _registry(self, *names):
        (self.root / "engine.c").write_text("".join('getenv("%s");\n' % n for n in names))
        (self.root / "coli_env.h").write_text(
            "static const ColiEnvVar coli_env_table[] = {\n"
            + "".join('    {"%s", CE_BOOL, CE_QWEN, 0, NULL},\n' % n for n in sorted(names)) + "};\n")

    def _run(self):
        out, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(error):
            code = registry.main()
        return code, out.getvalue() + error.getvalue()

    def test_every_variable_is_described_or_owed(self):
        self._registry("COLI_A", "COLI_B", "COLI_C")
        self.assertEqual(self._run()[0], 0)              # no docs beside this tree: not checked
        docs = self.root.parent / "docs"
        docs.mkdir()
        (docs / "ENVIRONMENT.md").write_text("`COLI_A` -- on -- does a.\nNot COLI_BB, nor xCOLI_C.\n")
        (self.root.parent / "README.md").write_text("Set `COLI_B=1`.\n")
        code, said = self._run()
        self.assertEqual(code, 1)
        self.assertIn("described in no reference doc: COLI_C", said)
        self.assertNotIn("COLI_A", said)
        self.assertNotIn("COLI_B ", said)
        (self.root / "tools").mkdir()
        owed = self.root / "tools" / "env_undocumented.txt"
        owed.write_text("# the debt\nCOLI_C\n")
        code, said = self._run()
        self.assertEqual(code, 0)
        self.assertIn("1 still to describe", said)

    def test_the_owed_list_only_shrinks(self):
        self._registry("COLI_A", "COLI_C")
        docs = self.root.parent / "docs"
        docs.mkdir()
        (docs / "ENVIRONMENT.md").write_text("`COLI_A` does a.\n")
        (self.root / "tools").mkdir()
        owed = self.root / "tools" / "env_undocumented.txt"
        for text, expected in (
                ("COLI_A\nCOLI_C\n", "described in the docs now, remove it from tools/env_undocumented.txt: COLI_A"),
                ("COLI_C\nCOLI_GONE\n", "in tools/env_undocumented.txt but not in coli_env.h: COLI_GONE"),
                ("COLI_C\nCOLI_C\n", "not sorted, or repeats a name")):
            with self.subTest(text=text):
                owed.write_text(text)
                code, said = self._run()
                self.assertEqual(code, 1)
                self.assertIn(expected, said)
