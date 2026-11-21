"""docs/SETTINGS.md's subcommand table names every `coli` subcommand.

The README sends readers to SETTINGS.md for every option, and
docs/MAINTAINING-DOCS.md names the `add_parser(...)` calls in c/coli as its
source of truth. A subcommand that is added there and not to the table is a
command nobody finds in the reference, as `import`, `ui` and `tui` were.
"""
import re
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
COLI = C_DIR / "coli"
SETTINGS = C_DIR.parent / "docs" / "SETTINGS.md"


def coli_subcommands():
    # Top-level subcommands only: `sub.add_parser("x"`. `cluster`'s own
    # subparsers (coordinator, worker) hang off another object.
    return set(re.findall(r'\bsub\.add_parser\(\s*"([a-z][a-z0-9_-]*)"', COLI.read_text()))


def documented_subcommands():
    text = SETTINGS.read_text(encoding="utf-8")
    table = text.split("### Subcommands", 1)[1].split("\n### ", 1)[0]
    # A row starts with the command's name, maybe followed by its arguments:
    # | `run "<prompt>"` |, | `cluster coordinator` / `cluster worker` |.
    return set(re.findall(r"^\| `([a-z][a-z0-9_-]*)", table, re.M))


class SettingsDocSubcommandsTest(unittest.TestCase):
    def test_parser_scan_finds_the_subcommands(self):
        # Guards the regex itself: a scan that found nothing would pass below.
        self.assertTrue({"serve", "setup", "convert", "cluster"} <= coli_subcommands())

    def test_every_subcommand_is_in_the_table(self):
        missing = sorted(coli_subcommands() - documented_subcommands())
        self.assertEqual(missing, [], "c/coli subcommands missing from docs/SETTINGS.md's "
                                      "Subcommands table: " + ", ".join(missing))

    def test_table_names_no_removed_subcommand(self):
        stale = sorted(documented_subcommands() - coli_subcommands())
        self.assertEqual(stale, [], "docs/SETTINGS.md lists subcommands c/coli no longer has: "
                                    + ", ".join(stale))


if __name__ == "__main__":
    unittest.main()
