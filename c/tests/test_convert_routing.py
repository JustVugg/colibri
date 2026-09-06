"""`coli convert` must pick the converter from the checkpoint, not from history.

#1368: it ran `tools/convert_fp8_to_int4.py` on whatever it was given. That
converter is GLM-5.2's, and it classifies tensors by FLAT name:

    if name in ("model.embed_tokens.weight", "lm_head.weight"): return "io"

GLM-5.3-Flash nests its language model under the vision wrapper, so its
embedding is `model.language_model.embed_tokens.weight`. It matched no rule,
fell into the final `endswith(".weight")` fallback, and was quantized. Hours
later `glm53.c` read it with load_f32, refused the U8, and the engine died
inside `coli web` with a traceback about a subprocess exiting.

Nothing along that path was wrong except the first choice.

What the tests below hold:

- the checkpoint decides the converter, and a Flash checkpoint reaches
  convert_glm53.py;
- GLM-5.2's command is byte-for-byte what it was, because the fix must not
  quietly re-tune the path that was already right;
- an option the target converter does not take is REFUSED, not dropped.
  Silently ignoring `--ebits` on a converter that has no such flag is the same
  defect wearing better manners;
- when the family cannot be determined -- no network, no huggingface_hub -- the
  old command is used unchanged, so a working conversion never fails because a
  5 KB metadata fetch did.
"""
import importlib.machinery
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from family_registry import FAMILIES, family_by_id


HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_convert_test", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class Args:
    """The subset of `coli convert`'s namespace that cmd_convert reads."""

    def __init__(self, out, **overrides):
        self.repo = "some/checkpoint"
        self.model = out
        self.ebits = 4
        self.io_bits = 8
        self.xbits = 0
        self.group_size = 64
        self.no_mtp = False
        self.__dict__.update(overrides)


class ConvertRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def run_convert(self, model_type, argv=("coli", "convert"), **overrides):
        """cmd_convert with the subprocess replaced: returns the commands it
        would have run, or the SystemExit it raised instead."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        calls = []

        def fake_call(command):
            calls.append(command)
            return 0

        with mock.patch.object(self.cli, "subprocess") as subprocess_module, \
             mock.patch.object(self.cli, "project_python", return_value="python3"), \
             mock.patch.object(self.cli, "sys") as system, \
             mock.patch.object(self.cli, "checkpoint_family",
                               return_value=(family_by_id(model_type)
                                             if model_type else None)):
            subprocess_module.call = fake_call
            system.argv = list(argv)
            system.exit.side_effect = SystemExit   # sys.exit must RAISE, not build
            try:
                self.cli.cmd_convert(Args(directory.name, **overrides))
            except (SystemExit, TypeError):
                # cmd_convert ends by calling sys.exit; the mock raises the class.
                pass
        return calls

    @staticmethod
    def script_of(command):
        return Path(command[1]).name

    def test_a_flash_checkpoint_reaches_its_own_converter(self):
        calls = self.run_convert("glm53")
        self.assertTrue(calls, "no converter was launched at all")
        self.assertEqual(self.script_of(calls[0]), "convert_glm53.py")

    def test_the_flash_command_carries_no_precision_flags(self):
        """convert_glm53.py has no --ebits/--io-bits/--xbits: it keeps the dense
        weights and the embedding wide and the engine picks the precision at
        load time. Passing them would be an argument error after the download."""
        command = self.run_convert("glm53")[0]
        for flag in ("--ebits", "--io-bits", "--xbits"):
            self.assertNotIn(flag, command)
        self.assertIn("--group-size", command)

    def test_flash_runs_one_pass_because_it_has_no_mtp_flag(self):
        calls = self.run_convert("glm53")
        self.assertEqual(len(calls), 1,
                         "convert_glm53.py has no --mtp; a second pass is an "
                         "argument error after the model is already written")

    def test_glm52_is_untouched(self):
        """The regression guard. This path was already correct and the whole
        point of the change is that it stays identical."""
        calls = self.run_convert("glm")
        self.assertEqual(self.script_of(calls[0]), "convert_fp8_to_int4.py")
        for flag in ("--repo", "--outdir", "--ebits", "--io-bits", "--group-size"):
            self.assertIn(flag, calls[0])
        self.assertEqual(len(calls), 2, "the int8 MTP pass must still run")
        self.assertIn("--mtp", calls[1])
        index = calls[1].index("--ebits")
        self.assertEqual(calls[1][index + 1], "8", "the MTP head is always int8")

    def test_no_mtp_still_skips_the_second_pass_for_glm52(self):
        calls = self.run_convert("glm", no_mtp=True)
        self.assertEqual(len(calls), 1)

    def test_an_option_the_target_does_not_take_is_refused_not_dropped(self):
        calls = self.run_convert(
            "glm53", argv=("coli", "convert", "--ebits", "3"), ebits=3)
        self.assertEqual(calls, [],
                         "--ebits was silently dropped and the conversion ran "
                         "anyway; the user asked for a precision they did not get")

    def test_an_unresolvable_family_keeps_the_old_command(self):
        """A metadata fetch that fails is not a reason to refuse a conversion
        that would have worked. The converter's own guard downloads the same
        config.json and refuses there if the family is genuinely wrong."""
        calls = self.run_convert(None)
        self.assertEqual(self.script_of(calls[0]), "convert_fp8_to_int4.py")
        for flag in ("--ebits", "--io-bits", "--group-size"):
            self.assertIn(flag, calls[0])
        self.assertEqual(len(calls), 2)


class ConverterDeclarationTest(unittest.TestCase):
    """A family that names a converter must also say which options it takes.

    Half a declaration is worse than none: the dispatcher would build a command
    with no precision flags at all and the user would get defaults they never
    chose, silently, which is the shape of #1368 itself.
    """

    def test_every_declared_converter_states_what_it_accepts(self):
        for family in FAMILIES:
            if not family.converter:
                self.assertEqual(
                    family.converter_accepts, (),
                    f"{family.id}: accepts options for a converter it does not name")
                continue
            self.assertTrue(
                family.converter.endswith(".py"),
                f"{family.id}: converter must be a script under tools/")
            self.assertTrue(
                family.converter_accepts,
                f"{family.id}: names {family.converter} but says nothing about "
                f"which coli options it takes, so every one of them is dropped")

    def test_declared_converters_exist_on_disk(self):
        for family in FAMILIES:
            if not family.converter:
                continue
            path = HERE / "tools" / family.converter
            self.assertTrue(path.is_file(),
                            f"{family.id} names tools/{family.converter}, "
                            f"which is not in the tree")

    def test_a_declared_converter_really_takes_what_is_claimed(self):
        """Read the target's own argparse rather than trusting the declaration.
        The two drift the moment someone renames a flag, and the failure would
        land on a user mid-conversion."""
        spelling = {"ebits": "--ebits", "io_bits": "--io-bits",
                    "xbits": "--xbits", "group_size": "--group-size"}
        for family in FAMILIES:
            if not family.converter:
                continue
            source = (HERE / "tools" / family.converter).read_text(
                encoding="utf-8", errors="ignore")
            for option in family.converter_accepts:
                flag = spelling[option]
                self.assertIn(f'add_argument("{flag}"', source,
                              f"{family.id} claims tools/{family.converter} takes "
                              f"{flag}, but that script does not define it")


if __name__ == "__main__":
    unittest.main()
