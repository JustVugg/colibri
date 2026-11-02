"""c/tui_mascot.h from the chat page's mascot (ui/index.html, const POSES): the same 19 poses,
20 x 20 pixels each, so the terminal's bird is the page's bird. Run it when the page's poses change:

    python3 c/tools/make_tui_mascot.py
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
page = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
poses = json.loads(re.search(r"const POSES=(\{.*?\});", page, re.S).group(1))
assert all(len(p) == 20 and all(len(r) == 20 for r in p) for p in poses.values()), "a pose is not 20 x 20"
out = ["/* The chat page's mascot (ui/index.html, const POSES), its poses as they are: 20 x 20 pixels.",
       "   Written by tools/make_tui_mascot.py; do not edit by hand. A letter is a colour (MPAL in",
       "   tui.c), '.' is empty. */",
       "enum { " + ", ".join("P_" + n.upper() for n in poses) + ", P_COUNT };",
       "static const char *const POSES[P_COUNT][20] = {"]
for name, rows in poses.items():
    out.append("    { /* %s */" % name)
    out += ['        "%s",' % r for r in rows]
    out.append("    },")
out.append("};")
(ROOT / "c" / "tui_mascot.h").write_text("\n".join(out) + "\n", encoding="utf-8")
print("c/tui_mascot.h:", len(poses), "poses")
