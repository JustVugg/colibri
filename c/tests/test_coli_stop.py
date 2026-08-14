import argparse
import contextlib
import io
import runpy
import signal
import unittest
from pathlib import Path
from unittest import mock

try:
    from .platform_test_support import requires_linux_stdlib_pidfd
except ImportError:
    from platform_test_support import requires_linux_stdlib_pidfd


C_DIR = Path(__file__).resolve().parents[1]
COLI = runpy.run_path(str(C_DIR / "coli"), run_name="coli_stop_test")
COLI_GLOBALS = COLI["cmd_stop"].__globals__


class ColiStopIdentityTest(unittest.TestCase):
    def test_stale_pidfile_cannot_select_an_unverified_process(self):
        rejected = mock.Mock(return_value=None)
        signal_target = mock.Mock()
        overrides = {
            "_open_stop_target": rejected,
            "_signal_stop_target": signal_target,
            "banner": mock.Mock(),
        }
        with mock.patch.dict(COLI_GLOBALS, overrides), mock.patch(
            "builtins.open",
            mock.mock_open(read_data="4242 /models/old\n"),
        ), mock.patch.object(
            COLI_GLOBALS["os"],
            "listdir",
            return_value=[],
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            COLI["cmd_stop"](
                argparse.Namespace(port=8000, dry_run=False),
                platform_name="linux",
            )

        rejected.assert_called_once_with(
            4242,
            "coli serve (pidfile, port 8000)",
            8000,
        )
        signal_target.assert_not_called()
        self.assertIn("nothing running", output.getvalue())

    def test_delayed_kill_revalidates_and_refuses_a_reused_pid(self):
        target = {
            "pid": 4242,
            "starttime": 99,
            "kind": "wrapper",
            "pidfd": None,
        }
        with mock.patch.dict(
            COLI_GLOBALS,
            {"_stop_target_matches": mock.Mock(side_effect=(True, False))},
        ), mock.patch.object(COLI_GLOBALS["os"], "kill") as kill:
            COLI["_signal_stop_target"](target, signal.SIGTERM)
            with self.assertRaises(ProcessLookupError):
                COLI["_signal_stop_target"](target, 9)

        kill.assert_called_once_with(4242, signal.SIGTERM)

    @requires_linux_stdlib_pidfd
    def test_pidfd_is_used_instead_of_numeric_pid_when_available(self):
        target = {
            "pid": 4242,
            "starttime": 99,
            "kind": "engine",
            "pidfd": 17,
        }
        with mock.patch.dict(
            COLI_GLOBALS,
            {"_stop_target_matches": mock.Mock(return_value=True)},
        ), mock.patch.object(
            COLI_GLOBALS["signal"],
            "pidfd_send_signal",
        ) as send, mock.patch.object(
            COLI_GLOBALS["os"],
            "kill",
        ) as kill:
            COLI["_signal_stop_target"](target, signal.SIGTERM)

        send.assert_called_once_with(17, signal.SIGTERM, None, 0)
        kill.assert_not_called()

    def test_non_linux_stop_is_actionable_before_process_discovery(self):
        banner = mock.Mock()
        open_target = mock.Mock()
        with mock.patch.dict(
            COLI_GLOBALS,
            {"banner": banner, "_open_stop_target": open_target},
        ), mock.patch.object(COLI_GLOBALS["os"], "listdir") as listdir:
            with self.assertRaisesRegex(
                SystemExit,
                "supported only on Linux.*platform process manager",
            ):
                COLI["cmd_stop"](
                    argparse.Namespace(port=8000, dry_run=False),
                    platform_name="win32",
                )

        banner.assert_not_called()
        open_target.assert_not_called()
        listdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
