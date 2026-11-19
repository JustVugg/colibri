"""`coli web` starts the chat UI as its child, and the UI exits when `coli web` does (its stdin
closes). The Models page switches models by stopping the gateway's process and starting it again:
under `coli web` that process is the UI's parent. The switch must find it (`coli web` serves
too), start a `coli serve` in its place, and neither stop the UI with the parent's children
nor let the UI exit while the switch runs."""
import importlib.util
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C))
spec = importlib.util.spec_from_file_location("ui_web_parent_models", C.parent / "ui/models.py")
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


class CommandTest(unittest.TestCase):
    def test_coli_command(self):
        self.assertEqual(models.coli_command(["python3", "/c/coli", "web", "--model", "/m"]), "web")
        self.assertEqual(models.coli_command(["/c/coli", "serve", "--model", "/m"]), "serve")
        self.assertIsNone(models.coli_command(["python3", "/c/coli", "chat", "--model", "/m"]))
        self.assertIsNone(models.coli_command(["python3", "openai_server.py", "serve"]))
        self.assertIsNone(models.coli_command(["python3", "/c/coli"]))

    def test_web_starts_again_as_serve_without_the_ui_options(self):
        argv = ["python3", "/c/coli", "web", "--model", "/m", "--ui-port", "9000", "--host", "0.0.0.0",
                "--no-browser", "--port", "8123", "--ui-port=9001", "--auto-tier"]
        self.assertEqual(models.serve_argv(argv), ["python3", "/c/coli", "serve", "--model", "/m",
                                                   "--host", "0.0.0.0", "--port", "8123", "--auto-tier"])
        serve = ["python3", "/c/coli", "serve", "--model", "/m"]
        self.assertEqual(models.serve_argv(serve), serve)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc")
class StopTest(unittest.TestCase):
    def test_stop_gateway_leaves_this_process_alone(self):
        """under `coli web` this UI is one of the gateway's children: the switch stops the others"""
        signals = []
        with mock.patch.object(models, "proc_children", return_value=[os.getpid(), 999999]), \
                mock.patch.object(models.os, "kill", side_effect=lambda pid, sig: signals.append(pid)), \
                mock.patch.object(models.os, "waitpid", side_effect=ChildProcessError), \
                mock.patch.object(models.os.path, "exists", side_effect=lambda p: p == "/proc/999999"):
            models.stop_gateway({"pid": 4242}, {})
        self.assertEqual(signals, [4242, 999999])


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc")
class LifetimeTest(unittest.TestCase):
    """ui/server.py --exit-with-stdin, with a stand-in for `coli web`: a parent that holds the
    UI's stdin and exits without closing anything itself"""

    def start(self, port):
        parent = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess, sys, time\n"
             "ui = subprocess.Popen([sys.executable, sys.argv[1], '--port', sys.argv[2], '--backend', "
             "'http://127.0.0.1:9', '--exit-with-stdin'], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, "
             "stderr=subprocess.DEVNULL)\n"
             "print(ui.pid, flush=True)\n"
             "sys.stdin.read()\n",
             str(C.parent / "ui/server.py"), str(port)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        ui = int(parent.stdout.readline())
        self.addCleanup(lambda: os.path.exists("/proc/%d" % ui) and os.kill(ui, 9))
        return parent, ui

    def test_the_ui_ends_with_its_parent(self):
        import socket
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            port = sk.getsockname()[1]
        parent, ui = self.start(port)
        time.sleep(1.0)
        self.assertTrue(os.path.exists("/proc/%d" % ui))
        parent.kill()            # SIGKILL: nothing in the parent runs, its fds close
        parent.wait()
        for _ in range(50):
            if not os.path.exists("/proc/%d" % ui) or "Z" in Path("/proc/%d/stat" % ui).read_text().rsplit(")", 1)[1][:3]:
                break
            time.sleep(0.1)
        else:
            self.fail("the UI outlived its parent")


if __name__ == "__main__":
    unittest.main()
