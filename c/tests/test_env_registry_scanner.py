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
        self.root = Path(self.directory.name)
        self.patch = patch.object(registry, "C_DIR", str(self.root))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_calls_across_lines_and_whitespace(self):
        (self.root / "engine.c").write_text(
            'getenv (\n "COLI_A");\nq38_env_bool(\n "Q38_B");\n'
            'compat_getenv_utf8("COLI_C");\n')
        self.assertEqual(registry.scan_sources(), {
            "COLI_A": ["engine.c:1"], "Q38_B": ["engine.c:3"],
            "COLI_C": ["engine.c:5"]})

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
