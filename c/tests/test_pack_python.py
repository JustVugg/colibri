"""tools/pack_python.py's needed() must compute the complete, real set of
files a release archive has to contain: every file coli reaches, by any of
the three kinds of edge, and everything those files reach in turn.

  - import, followed to closure from `coli`
  - subprocess, matched on the `os.path.join(TOOLS, "<script>.py")` call
    shape -- and the script is then walked like any other reached file, so
    what IT imports is followed too
  - data, a file with a data suffix opened next to a module it belongs to

Each edge was added after a published archive was found missing a file:
the hand-written list (#1296), the subprocess boundary, and the data file
(both #1359). The recurring defect is never the value of one case; it is
that the set is computed one way and the truth is another, so each test
below names the real edge it defends rather than an invented one.

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
- `data_files`: a string literal naming a file that EXISTS next to the
  module and carries a suffix from `DATA_SUFFIXES`; a plain name only, so
  a path or a name nothing matches yields nothing, and a source file
  merely mentioned in a message is not mistaken for data.
- `needed`: the import closure starting from `coli`; a script reached by
  subprocess is itself added to the queue so its own imports are
  followed to the same closure, not merely added as a leaf; data files
  are collected along the same walk as leaves; a script invoked but
  missing from `tools/` raises `SystemExit` before anything is returned.

All but one case builds a small, disposable fixture tree with real files.
The exception is deliberate and marked: one test asserts against this
checkout's own `c/` directory, because a fixture can only defend edges
someone already thought of, and the two files #1359 was about are the
evidence that the real tree grows shapes the fixtures do not have."""

import contextlib
import importlib.util
import io
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

    def test_the_real_tree_carries_the_case_that_started_this(self):
        """Every other test above builds a disposable fixture; this is the one
        real-tree assertion, run against this checkout's own c/ directory so a
        future tool import the fixtures cannot see is still caught here.

        It asserts the two files #1359 was actually about, rather than a name
        invented for the test: coli launches tools/convert_fp8_to_int4.py as a
        subprocess, that script imports iq3_pack inside quant_e8(), and
        iq3_pack opens iq3xxs_grid.json next to itself. Both were absent from
        every published archive. Assert the real pair and the test dies with
        the real bug if either edge is ever dropped again."""
        paths = PACK.needed(HERE.parent)
        self.assertIn(TOOLS / "iq3_pack.py", paths,
                      "reached only through a subprocess-launched script's import")
        self.assertIn(TOOLS / "iq3xxs_grid.json", paths,
                      "reached only as a data file opened next to iq3_pack.py")


class PackPythonDataFileTests(unittest.TestCase):
    """#1359 left a second edge open after the import one was closed: packaging
    iq3_pack.py still does not make `--xbits e8` work from an archive, because
    iq3_pack.py opens a sibling JSON and the packer copied only .py files."""

    def make_tree(self, root, data_name="grid.json"):
        root = pathlib.Path(root)
        (root / "tools").mkdir()
        (root / "coli").write_text(
            "import os\nimport reader\n"
            "TOOLS = os.path.join(os.path.dirname(__file__), 'tools')\n")
        (root / "reader.py").write_text(
            "import os, json\n"
            "def load():\n"
            "    p = os.path.join(os.path.dirname(__file__), %r)\n"
            "    return json.load(open(p))\n" % data_name)
        (root / data_name).write_text("[1, 2, 3]\n")
        return root

    def test_a_data_file_opened_next_to_a_module_is_packaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_tree(tmp)
            names = {path.name for path in PACK.needed(root)}
        self.assertIn("reader.py", names)
        self.assertIn("grid.json", names,
                      "a module's sibling data file is part of what the "
                      "archive must contain, not an optional extra")

    def test_without_data_files_the_module_ships_but_cannot_run(self):
        # Negative control: replay the pre-fix shape (walk the same graph, but
        # never collect data siblings) against the identical fixture. If this
        # ever stops dropping grid.json, the fixture no longer isolates the fix.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_tree(tmp)
            original, PACK.data_files = PACK.data_files, lambda path: set()
            try:
                names = {path.name for path in PACK.needed(root)}
            finally:
                PACK.data_files = original
        self.assertIn("reader.py", names)
        self.assertNotIn("grid.json", names)

    def test_a_source_file_named_in_a_message_is_not_packaged(self):
        """The allowlist earns its keep here. coli names colibri.c and
        family_registry.py names five engine sources, none of which belong in
        a release archive; a rule of "any existing sibling that is not .py"
        would have shipped all of them."""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "engine.c").write_text("int main(void){return 0;}\n")
            (root / "talker.py").write_text(
                'MSG = "build it first: cc engine.c"\n'
                'NAME = "engine.c"\n')
            found = PACK.data_files(root / "talker.py")
        self.assertEqual(found, set())

    def test_a_name_that_is_not_a_real_sibling_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "talker.py").write_text(
                'A = "absent.json"\n'          # nothing of that name exists
                'B = "sub/dir/real.json"\n')   # a path, not a plain sibling
            (root / "sub").mkdir()
            (root / "sub" / "dir").mkdir()
            (root / "sub" / "dir" / "real.json").write_text("{}\n")
            found = PACK.data_files(root / "talker.py")
        self.assertEqual(found, set())

    def test_the_summary_line_counts_data_files_separately(self):
        """A display derived from a set whose meaning changed is how #856 hid
        for a release: the engine's own self-report agreed with the bug."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_tree(tmp)
            dist = pathlib.Path(tmp) / "dist"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                PACK.main(["pack_python.py", str(root), str(dist)])
        # coli is the walk's root and is copied by the workflow, not by
        # needed(), so the fixture's one packaged module is reader.py.
        self.assertIn("1 Python files and 1 data files", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
