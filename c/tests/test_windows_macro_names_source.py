#!/usr/bin/env python3
"""A full <windows.h> (no WIN32_LEAN_AND_MEAN) pulls in <dlgs.h>, and <dlgs.h> #defines the
common dialogs' control IDs as short lowercase words: grp1..grp4, stc1..stc32, lst1..lst16,
psh1..psh16, ... qwen36.c includes such a <windows.h> through serve_poll.h before idot.h, so
the local `const int8_t *grp2` in i8r16_group2 became `const int8_t *0x0431` and the AVX-512
VNNI build failed on Windows (#2611). Linux and macOS never define these names, so only a
Windows build on such a CPU shows it.

The rule checked here, over every C, C++, CUDA and Objective-C source in c/, c/tests, c/tools
and c/train: outside comments and string literals, no identifier is one of <dlgs.h>'s macros.
"""
from pathlib import Path
import re
import unittest

C = Path(__file__).resolve().parents[1]
DIRS = (C, C / "tests", C / "tools", C / "train")
EXTS = (".c", ".h", ".cpp", ".cu", ".cuh", ".m", ".mm")
# every object-like macro <dlgs.h> defines with a lowercase name (mingw-w64 and the Windows SDK)
DLGS = re.compile(r"(?<!\w)(?:ctlFirst|ctlLast|ctl1|psh(?:1[0-6]|[1-9]|Help)|chx(?:1[0-6]|[1-9])"
                  r"|rad(?:1[0-6]|[1-9])|grp[1-4]|frm[1-4]|rct[1-4]|ico[1-4]|stc(?:3[0-2]|[12]\d|[1-9])"
                  r"|lst(?:1[0-6]|[1-9])|cmb(?:1[0-6]|[1-9])|edt(?:1[0-6]|[1-9])|scr[1-8])(?!\w)")
# comments, string and character literals: blanked out, their newlines kept for the line numbers
NOISE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S)


def code_of(src):
    return NOISE.sub(lambda m: "\n" * m.group(0).count("\n"), src)


class WindowsMacroNamesSourceTests(unittest.TestCase):
    def test_no_dlgs_h_macro_used_as_a_name(self):
        bad = []
        for d in DIRS:
            for path in sorted(p for p in d.glob("*") if p.suffix in EXTS):
                code = code_of(path.read_text(encoding="utf-8", errors="replace"))
                for no, line in enumerate(code.split("\n"), 1):
                    for m in DLGS.finditer(line):
                        bad.append(f"{path.relative_to(C).as_posix()}:{no}: {m.group(0)}")
        self.assertEqual(bad, [], "a name <dlgs.h> #defines (a full <windows.h> includes it): rename it")

    def test_the_rule_sees_the_2611_line(self):
        self.assertTrue(DLGS.search(code_of("const int8_t *grp2=grp+(size_t)16*I;")))
        self.assertTrue(DLGS.search(code_of("int stc32, lst16; s->pshHelp=1;")))
        # not a whole name, or not code: nothing for the preprocessor to replace
        self.assertIsNone(DLGS.search(code_of("int grp21, mygrp2, grp_2, stc33, lst0;")))
        self.assertIsNone(DLGS.search(code_of("/* grp2 */ // stc1\nputs(\"lst1\"); c='\\'';")))


if __name__ == "__main__":
    unittest.main()
