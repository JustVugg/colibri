"""The chat UI's children never inherit its stdin (#3140).

On Windows a read waiting on a pipe holds that pipe: the UI under `coli web` had a thread reading
its stdin (to exit when `coli web` closed it), and a child process inheriting that stdin waited on
it before it ran, so the Models page's machine probe (setup_hw), an agent lookup or a download
hung. `coli web` now starts the UI with no pipe on stdin and --exit-with-parent; and each child
the UI process starts, in ui/ and in the setup modules it runs, gets its own stdin (DEVNULL, or
input), so neither path can come back by a new call that forgets.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# what the UI process imports and runs: its own modules and the setup modules ui/models.py calls
FILES = sorted((ROOT / "ui").glob("*.py")) + [ROOT / "c" / n for n in
                                              ("setup_hw.py", "setup_download.py", "setup_catalog.py", "setup_home.py")]
# setup_flow.py: the CLI's flow, where a server started in the foreground keeps the terminal
# (its stdin) on purpose; its other children are held to the rule
FLOW = ROOT / "c" / "setup_flow.py"
FLOW_FOREGROUND = {"start_server"}   # its foreground Popen; the background one passes DEVNULL in **kwargs
SPAWN = {"run", "Popen", "check_output", "call", "check_call"}


def calls_without_stdin(path, skip_functions=()):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name) and child.func.value.id == "subprocess"
                    and child.func.attr in SPAWN and name not in skip_functions):
                kws = {k.arg for k in child.keywords}
                if not kws & {"stdin", "input"} and None not in kws:   # **kwargs: the caller's
                    out.append("%s:%d subprocess.%s" % (path.relative_to(ROOT), child.lineno, child.func.attr))
            visit(child, name)
    visit(tree, None)
    return out


class ChildStdinTest(unittest.TestCase):
    def test_the_ui_process_children_get_their_own_stdin(self):
        missing = []
        for f in FILES:
            if f.exists():
                missing += calls_without_stdin(f)
        self.assertEqual(missing, [], "pass stdin=subprocess.DEVNULL (or input=...) to these")

    def test_setup_flow_children_outside_the_foreground_server(self):
        self.assertEqual(calls_without_stdin(FLOW, FLOW_FOREGROUND), [])

    def test_coli_web_gives_the_ui_no_stdin_pipe(self):
        src = (ROOT / "c" / "coli").read_text(encoding="utf-8")
        at = src.index('"--exit-with-parent", str(os.getpid())]')
        line = src[at:src.index("\n", at)]
        self.assertIn("stdin=subprocess.DEVNULL", line)
        self.assertNotIn("--exit-with-stdin", src)


if __name__ == "__main__":
    unittest.main()
