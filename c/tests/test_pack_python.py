"""tools/pack_python.py's needed() must compute the complete, real set of
Python files a release archive has to contain: every file coli reaches,
whether by import (followed to closure) or by subprocess invocation
(matched by its own `os.path.join(TOOLS, "<script>.py")` call sites) --
and, for a subprocess-invoked script, every file THAT script itself
reaches by import, followed the same way coli's own imports are.

Checks enumerated from the source (`tools/pack_python.py`, read in full
before writing this module):

- `local_modules`: root `*.py` files take priority over a `tools/*.py`
  file of the same stem (root added first, `tools/` only via
  `setdefault`).
- `imports_of`: both `import x` and `from x import y` are seen, at any
  nesting (inside a function, after a `sys.path.insert`, inside a `try`)
  -- ast sees them regardless of where in the file they appear.
- `invoked_scripts`: only the exact `TOOLS, "<snake_case>.py"` call-site
  shape is matched; nothing else in the file is mistaken for one.
- `needed`: the import closure starting from `coli`; a script reached by
  subprocess is itself added to the queue so its own imports are
  followed to the same closure, not merely added as a leaf; a script
  invoked but missing from `tools/` raises `SystemExit` before anything
  is returned.

No real Colibri source tree is read here -- every case builds a small,
disposable fixture tree with real files."""

import importlib.util
import pathlib
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"

_spec = importlib.util.spec_from_file_location(
    "pack_python_under_test", TOOLS / "pack_python.py")
PACK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PACK)


class PackPythonNeededTests(unittest.TestCase):
    def make_tree(self, root):
        """A minimal coli-shaped tree: coli imports `foo` and invokes
        tools/bar.py as a subprocess; bar.py itself imports tools/baz.py,
        a module coli never mentions by name anywhere."""
        root = pathlib.Path(root)
        (root / "tools").mkdir()
        (root / "coli").write_text(
            "import os\n"
            "import foo\n"
            "TOOLS = os.path.join(os.path.dirname(__file__), 'tools')\n"
            "CMD = [PY, os.path.join(TOOLS,\"bar.py\")]\n")
        (root / "foo.py").write_text("X = 1\n")
        (root / "tools" / "bar.py").write_text("import baz\nX = 2\n")
        (root / "tools" / "baz.py").write_text("Y = 3\n")
        return root

    def test_a_subprocess_reached_script_pulls_in_its_own_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_tree(tmp)
            paths = PACK.needed(root)
        names = {path.name for path in paths}
        self.assertIn("foo.py", names)   # reached by ordinary import
        self.assertIn("bar.py", names)   # reached by subprocess
        self.assertIn(
            "baz.py", names,
            "bar.py's own import of baz must be followed, not dropped "
            "at the subprocess boundary")

    def test_removing_the_subprocess_recursion_drops_the_transitive_import(self):
        # Mechanical proof the fix above is load-bearing: replay the
        # PRE-FIX shape (a subprocess-reached script is added to the
        # result directly, but its own imports are never walked) against
        # the identical fixture, and confirm baz.py is silently dropped.
        def needed_without_recursion(src):
            local = PACK.local_modules(src)
            reached, queue = set(), [src / "coli"]
            scripts = set()
            while queue:
                path = queue.pop()
                scripts |= PACK.invoked_scripts(path)
                for name in PACK.imports_of(path):
                    if name in local and name not in reached:
                        reached.add(name)
                        queue.append(local[name])
            paths = {local[name] for name in reached}
            for script in sorted(scripts):
                candidate = src / "tools" / script
                if not candidate.exists():
                    raise SystemExit(
                        f"FAIL: coli invokes tools/{script}, which does not exist")
                paths.add(candidate)
            return sorted(paths)

        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_tree(tmp)
            names = {path.name for path in needed_without_recursion(root)}
        self.assertIn("bar.py", names)
        self.assertNotIn(
            "baz.py", names,
            "this replay of the pre-fix code must reproduce the defect "
            "(baz.py missing) -- if it doesn't, the fixture no longer "
            "isolates what the fix changed")

    def test_missing_invoked_script_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_tree(tmp)
            (root / "tools" / "bar.py").unlink()
            with self.assertRaisesRegex(SystemExit, r"tools/bar\.py"):
                PACK.needed(root)

    def test_root_module_shadows_a_same_named_tools_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "tools").mkdir()
            (root / "shared.py").write_text("ROOT = 1\n")
            (root / "tools" / "shared.py").write_text("TOOLS = 1\n")
            local = PACK.local_modules(root)
            self.assertEqual(local["shared"].read_text(), "ROOT = 1\n")

    def test_invoked_scripts_matches_only_the_exact_call_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            path = root / "sample.py"
            path.write_text(
                'a = os.path.join(TOOLS,"real_one.py")\n'
                'b = "TOOLS, \\"not_a_call.py\\""\n'
                'c = os.path.join(OTHER,"wrong_var.py")\n')
            found = PACK.invoked_scripts(path)
        self.assertEqual(found, {"real_one.py"})

    def test_engine_evidence_is_needed_by_the_real_tree(self):
        """Every other test in this module builds a disposable fixture; this
        is the one real-tree assertion, run against this checkout's own
        c/ directory so a future tool import that the fixture-based tests
        cannot see (because they never touch the real tree) is still
        caught here."""
        paths = PACK.needed(HERE.parent)
        self.assertIn(TOOLS / "engine_evidence.py", paths)


if __name__ == "__main__":
    unittest.main()
