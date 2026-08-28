"""Every model type the launcher NAMES, the launcher must also DISPATCH.

#879: `coli` printed "OLMoE · 7B" in its banner and then handed the request to
the GLM engine, because `_BANNER_MODELS` knew about `olmoe` and `model_arch()`
did not. The engine binary shipped in the release archive the whole time; the
launcher structurally could not select it.

The release check that was supposed to catch this (#868, "coli would not
resolve these next to itself") asserted the binary was PRESENT next to `coli`.
It passed. Presence is not dispatch, and a green test that proves the weaker
property is worse than no test, because it is read as proving the stronger one.

So the invariant is stated once, here, over the launcher's own table:

    for every model_type in _BANNER_MODELS:
        model_arch() must classify it, and engine_for() must resolve it to a
        binary name that is not merely the GLM fallback by accident

It is self-updating. Adding a family to the banner roster without teaching the
dispatcher about it fails this file, rather than shipping and being found by
someone who downloaded a release.
"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from family_registry import all_families


HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_dispatch_test", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class LauncherDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def _model_dir(self, model_type):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "config.json").write_text(
            json.dumps({"model_type": model_type}), encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        return str(root)

    def test_every_named_model_type_is_classified(self):
        for family in all_families():
            for model_type in family.model_types:
                with self.subTest(model_type=model_type):
                    self.assertEqual(self.cli.model_arch(self._model_dir(model_type)),
                                     family.id)

    def test_every_named_model_type_resolves_to_a_distinct_engine(self):
        """Two families must not resolve to the same binary. That is the shape
        the bug took: everything unrecognised quietly became the GLM engine."""
        for family in all_families():
            model_type = family.model_types[0]
            with self.subTest(model_type=model_type):
                engine = os.path.basename(
                    self.cli.engine_for(self._model_dir(model_type)))
                self.assertEqual(engine, family.engine_artifact + self.cli._EXE)

    def test_engine_for_has_a_branch_for_every_arch_model_arch_can_return(self):
        """engine_for() indexes a dict with model_arch()'s return value. A value
        model_arch can produce and the dict lacks is a KeyError at runtime, on
        the user's machine, after the banner has already printed."""
        for family in all_families():
            model_type = family.model_types[0]
            with self.subTest(model_type=model_type):
                try:
                    self.cli.engine_for(self._model_dir(model_type))
                except KeyError as error:
                    self.fail(f"{family.display_name}: engine_for raised KeyError({error}) -- "
                              f"model_arch() classifies this family and the "
                              f"engine dict has no entry for it")

    def test_the_roster_is_not_empty(self):
        """Guards the whole file: if _BANNER_MODELS is ever renamed or emptied,
        the loops above pass vacuously and this suite proves nothing."""
        self.assertGreaterEqual(len(all_families()), 5)


class ProjectPythonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_bench_and_convert_use_windows_project_python(self):
        python = self.root / "mio_env" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")
        data = self.root / "data"
        data.mkdir()
        (data / "hellaswag.jsonl").write_text("", encoding="utf-8")

        bench = types.SimpleNamespace(
            model="model", tasks=["hellaswag"], data=str(data), limit=1, ram=None)
        convert = types.SimpleNamespace(
            model="output", repo="repo", ebits=4, io_bits=4,
            group_size=128, xbits=None, no_mtp=True)

        with mock.patch.object(self.cli, "HERE", str(self.root)), \
             mock.patch.object(self.cli.sys, "platform", "win32"), \
             mock.patch.object(self.cli, "need_model"), \
             mock.patch.object(self.cli, "banner"), \
             mock.patch.object(self.cli, "env_for", return_value={}), \
             mock.patch.object(self.cli.subprocess, "call", return_value=0) as call:
            with self.assertRaisesRegex(SystemExit, "0"):
                self.cli.cmd_bench(bench)
            self.assertEqual(call.call_args.args[0][0], str(python))

            call.reset_mock()
            with self.assertRaisesRegex(SystemExit, "0"):
                self.cli.cmd_convert(convert)
            self.assertEqual(call.call_args.args[0][0], str(python))

    def test_posix_project_python_is_unchanged(self):
        python = self.root / "mio_env" / "bin" / "python3"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")
        with mock.patch.object(self.cli, "HERE", str(self.root)), \
             mock.patch.object(self.cli.sys, "platform", "linux"):
            self.assertEqual(self.cli.project_python(), str(python))

    def test_missing_project_python_falls_back_to_current_interpreter(self):
        with mock.patch.object(self.cli, "HERE", str(self.root)), \
             mock.patch.object(self.cli.sys, "platform", "win32"):
            self.assertEqual(self.cli.project_python(), sys.executable)


if __name__ == "__main__":
    unittest.main()
