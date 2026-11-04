"""ui/agents.py with the UI on Windows itself (coli ui from a Windows prompt, the desktop
window): WSL's agents found and opened through wsl.exe, a folder checked as Windows has it.
wsl.exe and the window are stand-ins; the distribution's share is a temporary directory."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

UI = Path(__file__).resolve().parent.parent.parent / "ui"
sys.path.insert(0, str(UI))
import agents  # noqa: E402

FOUND = ("COLI_AGENT claude=/home/u/.local/bin/claude\nCOLI_AGENT pi=not-a-path\n"
         "COLI_HOME=/home/u\nCOLI_DISTRO=Ubuntu\nCOLI_SHARE=%s\n")


class WindowsHost(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.share = os.path.join(self.tmp, "share")
        self.folder = os.path.join(self.tmp, "Desktop")
        os.makedirs(self.folder)
        self.calls, self.started, self.reach = [], [], "ok"
        agents._cache.update(t=0.0, v=None)
        agents._last_launch[0] = 0.0

        def run(cmd, timeout=20):
            self.calls.append(cmd)
            if cmd[:4] == ["wsl.exe", "-e", "bash", "-lic"]:
                return FOUND % self.share
            if "wslpath" in cmd:
                return "/mnt/c/Users/u/Desktop\n"
            if "-c" in cmd and "/dev/tcp/" in cmd[-1]:
                return self.reach
            return ""

        patches = [mock.patch.object(agents, "ON_WIN", True), mock.patch.object(agents, "_run", run),
                   mock.patch.object(agents.shutil, "which", lambda n: "C:\\Windows\\system32\\" + n if n == "wsl.exe" else None),
                   mock.patch.object(agents, "_models", lambda backend: {"q38": 65536}),
                   mock.patch.object(agents, "_wslpath", lambda p, to_unix=True: self.folder if p == "C:\\Users\\u\\Desktop" else None),
                   mock.patch.object(agents, "_start", lambda win, agent, title, command: (self.started.append(command), (True, None))[1])]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_wsl_agents_found_through_wsl_exe(self):
        info = agents.detect(force=True)
        claude = next(a for a in info["agents"] if a["id"] == "claude")
        self.assertEqual((claude["installed"], claude["where"], claude["path"]), (True, "wsl", "/home/u/.local/bin/claude"))
        self.assertFalse(next(a for a in info["agents"] if a["id"] == "pi")["installed"], "a line that is not a path")
        self.assertEqual(info["wsl"], {"distro": "Ubuntu", "home": "/home/u", "share": self.share})
        self.assertNotIn('"', agents._WSL_FIND, "one argument of wsl.exe's command line")

    def test_claude_in_wsl_opens_in_the_folder_as_wsl_has_it(self):
        ok, msg = agents.launch("claude", "q38", "C:\\Users\\u\\Desktop", "", "http://127.0.0.1:8000", "http://127.0.0.1:18790")
        self.assertTrue(ok, msg)
        self.assertEqual(self.started, ["wsl.exe -d Ubuntu -- bash -l /home/u/.cache/colibri-agents/claude.sh"])
        with open(os.path.join(self.share, "home", "u", ".cache", "colibri-agents", "claude.sh"), "rb") as f:
            script = f.read().decode()
        self.assertNotIn("\r", script)
        self.assertIn("cd '/mnt/c/Users/u/Desktop' || exit 1", script)
        self.assertIn("export ANTHROPIC_BASE_URL='http://127.0.0.1:8000'", script)
        self.assertIn(["wsl.exe", "-d", "Ubuntu", "-e", "chmod", "700", "/home/u/.cache/colibri-agents",
                       "/home/u/.cache/colibri-agents/claude.sh"], self.calls)

    def test_a_server_wsl_cannot_reach_is_said_before_any_window(self):
        self.reach = ""
        ok, msg = agents.launch("claude", "q38", "C:\\Users\\u\\Desktop", "", "http://127.0.0.1:8000", "http://127.0.0.1:18790")
        self.assertFalse(ok)
        self.assertIn("networkingMode=mirrored", msg)
        self.assertEqual(self.started, [])

    def test_a_missing_folder(self):
        ok, msg = agents.launch("claude", "q38", "C:\\Users\\u\\NonEsiste", "", "http://127.0.0.1:8000", "http://127.0.0.1:18790")
        self.assertFalse(ok)
        self.assertIn("non esiste", msg)


@unittest.skipUnless(shutil.which("bash"), "bash")
class WslFind(unittest.TestCase):
    def test_an_alias_to_an_installed_program_counts(self):
        with tempfile.TemporaryDirectory() as home:
            prog = os.path.join(home, ".claude", "local", "claude")
            os.makedirs(os.path.dirname(prog))
            with open(prog, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(prog, 0o755)
            script = "alias claude='~/.claude/local/claude --x'; alias codex='/nonexistent/codex'; " + agents._WSL_FIND
            out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                                 env={"HOME": home, "PATH": "/usr/bin:/bin"}).stdout
        self.assertIn("COLI_AGENT claude=%s\n" % prog, out)
        self.assertNotIn("COLI_AGENT codex", out)
        self.assertIn("COLI_HOME=%s\n" % home, out)


if __name__ == "__main__":
    unittest.main()
